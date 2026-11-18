"""Accumulating-quantity ladders: parsing on recorded payloads, the as-of nowcast, fills and the book."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from albert.research import counts
from albert.research.xcat import ohlc_rows

FIX = Path(__file__).parent / "fixtures"


def _markets() -> list[dict]:
    return json.loads((FIX / "kalshi" / "counts_markets.json").read_text())["markets"]


# ---- discovery and parsing ----------------------------------------------------------------------------
def test_series_rows_families_and_financial_exclusion():
    payload = json.loads((FIX / "kalshi" / "counts_series.json").read_text())
    rows = {r["series_ticker"]: r for r in counts.series_rows(payload)}
    assert rows["KXRAINNYCM"]["family"] == "rain_month"
    assert rows["KXAVGTKAUS"]["family"] == "temp_avg_week"
    assert rows["KXNYCSNOWM"]["family"] == "snow_month"
    assert "KXSNOWMENTION" not in rows                       # a mention series, not a snowfall ladder
    assert rows["KXTSAW"]["financial"] is True                 # Kalshi files it under Economics
    assert rows["KXRAINNYCM"]["financial"] is False
    assert rows["KXTRUTHSOCIAL"]["source_name"] == "Roll Call"
    assert rows["KXAVGTKAUS"]["source_name"] == "The Weather Company"


def test_market_rows_station_period_and_terms():
    ms = _markets()
    rain = counts.market_row(ms[0], "KXRAINNYCM", "hist")
    assert rain["strike_type"] == "greater" and rain["floor_strike"] == pytest.approx(13)
    assert rain["result"] in ("yes", "no") and rain["close_time"].tzinfo is not None
    assert counts.rules_station(rain["rules_primary"]) == "KNYC"
    assert counts.month_period(rain["event_ticker"]) == (date(2026, 8, 1), date(2026, 8, 31))
    temp = counts.market_row(ms[-1], "KXAVGTKAUS", "live")
    assert counts.rules_station(temp["rules_primary"]) == "KAUS"
    first, last = counts.rules_period(temp["rules_primary"])
    assert (last - first).days == 6
    thr, need = counts.streak_terms(temp["rules_primary"])
    assert thr > 0 and need == temp["floor_strike"]


def test_rules_station_cli_codes():
    assert counts.rules_station("total precipitation at CLIORD in Chicago") == "KORD"
    assert counts.rules_station("at Chicago Midway, IL in Jul") == "KMDW"
    assert counts.rules_station("snowfall in Boston") is None


def test_cli_rows_trace_missing_and_issuance():
    payload = json.loads((FIX / "iem" / "cli_precip.json").read_text())
    rows = counts.cli_rows(payload, "KNYC")
    assert len(rows) == len(payload["results"])
    raw = payload["results"][0]
    assert rows[0]["issued"] == datetime(2026, 8, 19, 6, 17, tzinfo=UTC)     # product id 202608190617
    assert rows[0]["precip"] == pytest.approx(raw["precip"])
    traces = [r for r, x in zip(rows, payload["results"], strict=True) if x["precip"] == "T"]
    assert traces and all(r["precip"] == counts.TRACE for r in traces)
    pl.DataFrame(rows, schema=counts.CLI_COLS)


def test_robots_verdicts_on_recorded_files():
    iem = (FIX / "robots" / "iem.txt").read_text()
    assert all(counts.robots_verdict(iem, "/json/cli.py").values())
    assert not any(counts.robots_verdict((FIX / "robots" / "spc.txt").read_text(), "/climo/reports/").values())
    rc = counts.robots_verdict((FIX / "robots" / "rollcall.txt").read_text(), "/factbase/trump/topic/social/")
    assert rc["anthropic-ai"] is False
    fr = (FIX / "robots" / "federalregister.txt").read_text()            # directives are indented
    assert all(counts.robots_verdict(fr, "/api/v1/documents.json").values())
    assert not any(counts.robots_verdict(fr, "/documents/search").values())
    cg = counts.robots_verdict((FIX / "robots" / "congress.txt").read_text(), "/nominations")
    assert cg["*"] is True and cg["ClaudeBot"] is False


# ---- as-of nowcast ------------------------------------------------------------------------------------
def _history(days: list[date], precip: float, issue_lag_h: int = 30) -> counts.StationHistory:
    df = pl.DataFrame({"station": "KNYC", "day": days,
                       "issued": [datetime(d.year, d.month, d.day, tzinfo=UTC) + timedelta(hours=issue_lag_h) for d in days],
                       "precip": [precip] * len(days), "snow": [0.0] * len(days), "high": [80.0] * len(days),
                       "low": [60.0] * len(days), "precip_month": [None] * len(days), "snow_month": [None] * len(days)},
                      schema=counts.CLI_COLS)
    return counts.StationHistory(df)


def test_rain_nowcast_uses_only_issued_days_and_past_years():
    days = [date(2010, 1, 1) + timedelta(days=i) for i in range((date(2026, 11, 1) - date(2010, 1, 1)).days)]
    h = _history(days, 0.1)
    month = [date(2026, 10, 1) + timedelta(days=i) for i in range(31)]
    t = datetime(2026, 10, 11, 5, tzinfo=UTC)           # CLIs of Oct 1..9 issued (30 h lag), Oct 10 not yet
    nc = counts.rain_nowcast(h, month, t, strike=2.5)
    assert nc["known_days"] == 9 and nc["unknown_days"] == 22
    assert nc["partial"] == pytest.approx(0.9)
    assert nc["clim_mean_rem"] == pytest.approx(2.2)    # 22 analog days at 0.1
    assert nc["p_model"] > 0.5                            # 0.9 + 2.2 > 2.5 on every path
    # a strike already passed by the published partial sum is decided
    assert counts.rain_nowcast(h, month, t, strike=0.5)["decided"] is True
    # future values never enter: changing every day after t leaves the known partial unchanged
    h.precip[h.ix(date(2026, 10, 10)):] = 5.0
    assert counts.rain_nowcast(h, month, t, strike=2.5)["partial"] == pytest.approx(0.9)


def test_streak_runs_and_decision_times():
    assert counts.max_run(np.array([[1, 1, 0, 1, 1, 1, 0]], dtype=bool)).tolist() == [3]
    p0, p1 = counts.lst_bounds("KNYC", date(2026, 7, 1), date(2026, 7, 31))
    assert p0 == datetime(2026, 7, 1, 5, tzinfo=UTC)     # local *standard* midnight (EST) in July
    dts = dict(counts.decision_times(p0, p1))
    assert set(dts) == {"f25", "f50", "f75", "f90", "final"}
    assert dts["f25"] < dts["f50"] < dts["f75"] < dts["f90"] < dts["final"] < p1
    assert all(t.minute == 0 for t in dts.values())


# ---- quotes, fills, fees, book ------------------------------------------------------------------------
def _arrays():
    payload = json.loads((FIX / "kalshi" / "counts_candles_historical.json").read_text())
    df = pl.DataFrame(ohlc_rows(payload, payload["ticker"], 60), schema=counts_ohlc_schema())
    return df, counts._candle_arrays(df)


def counts_ohlc_schema():
    from albert.research.xcat import OHLC_SCHEMA
    return OHLC_SCHEMA


def test_quote_at_never_uses_a_candle_closing_after_the_decision():
    df, ca = _arrays()
    t = df["end_ts"][5] + timedelta(minutes=30)
    q = counts.quote_at(ca, t)
    assert q["bid"] == pytest.approx(df["bid_close"][5]) and q["ask"] == pytest.approx(df["ask_close"][5])
    assert counts.quote_at(ca, df["end_ts"][0] - timedelta(seconds=1))["bid"] is None
    w = counts.window_prints(ca, df["end_ts"][5], df["end_ts"][10])
    sub = df.slice(6, 5)                                   # hours that start at/after the decision
    assert w["w_lo"] == pytest.approx(sub["p_low"].min())


def test_taker_fee_rounds_up_per_order():
    assert counts.taker_fee(1, 0.5, 1.0) == pytest.approx(0.02)          # 1.75c -> 2c
    assert counts.taker_fee(4, 0.5, 1.0) == pytest.approx(0.07)          # 7.0c exactly
    assert counts.taker_fee(10, 0.97, 1.0) == pytest.approx(0.03)        # 2.04c -> 3c


def _row(**kw):
    base = {"ticker": "M", "event_ticker": "E", "family": "rain_month", "split": "dev", "decision": "f75",
            "t": datetime(2026, 9, 20, 15, tzinfo=UTC), "t_next": datetime(2026, 9, 26, 15, tzinfo=UTC),
            "date": date(2026, 9, 20), "y": 1, "settle_ts": datetime(2026, 10, 1, 5, tzinfo=UTC), "fee_type": "quadratic",
            "fee_multiplier": 1.0, "p_model": 0.8, "bid": 0.55, "ask": 0.60, "w_lo": None, "w_hi": None, "w_vol": 0.0}
    return {**base, **kw}


def test_taker_and_maker_rules():
    rows = pl.DataFrame([_row(), _row(ticker="N", p_model=0.4, bid=0.50, ask=0.52, y=0)], infer_schema_length=None)
    tr = counts.taker_trades(rows, 0.08)
    got = {r["ticker"]: r for r in tr.iter_rows(named=True)}
    assert got["M"]["side"] == "yes" and got["M"]["price"] == pytest.approx(0.60) and got["M"]["n"] == 3
    assert got["N"]["side"] == "no" and got["N"]["price"] == pytest.approx(0.50)
    fee_c = 100 * counts.taker_fee(3, 0.60, 1.0) / 3
    assert got["M"]["pnl_c"] == pytest.approx(40 - fee_c)
    # maker: YES bid at floor(0.8 - 0.05) = 0.75 capped one tick under the 0.60 ask -> 0.59; fills only
    # on a print strictly below it
    at = counts.maker_fills(pl.DataFrame([_row(w_lo=0.59, w_hi=0.61)], infer_schema_length=None), 0.05)
    assert at.height == 0
    thr = counts.maker_fills(pl.DataFrame([_row(w_lo=0.58, w_hi=0.61)], infer_schema_length=None), 0.05)
    assert thr.height == 1 and thr["side"][0] == "yes" and thr["price"][0] == pytest.approx(0.59)
    assert thr["pnl_c"][0] == pytest.approx(41.0)


def test_ledger_respects_event_and_open_caps():
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    tr = pl.DataFrame([{**_row(ticker=f"M{i}", event_ticker=f"E{i // 3}", t=t0 + timedelta(hours=i)),
                        "side": "yes", "price": 0.5, "payoff": 1, "n": 4, "fee": 0.07} for i in range(30)],
                      infer_schema_length=None)
    led = counts.ledger(tr, "taker", "x", "dev")
    per_event = led.group_by("event_ticker").agg(pl.col("cost_usd").sum())
    assert per_event["cost_usd"].max() <= counts.EVENT_CAP + 1e-9
    assert led["cost_usd"].sum() <= counts.OPEN_CAP + 1e-9          # nothing settles inside the window
    assert set(led.columns) >= {"struct_id", "event_ticker", "ticker", "side", "n", "t_in", "cost_usd",
                                "settle_ts", "payoff_usd"}


def test_cluster_se_two_way_reduces_to_one_way_when_dates_are_unique():
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    ev = np.repeat(np.arange(12), 5).astype(str)
    dt = np.arange(60).astype(str)
    assert counts.cluster_se(x, ev, dt) == pytest.approx(counts.cluster_se(x, ev), rel=0.35)
    s = counts.summarize(pl.DataFrame({"pnl_c": x, "event_ticker": ev, "date": [date(2026, 1, 1)] * 60}))
    assert s["n"] == 60 and s["events"] == 12


def test_federal_register_rows_and_eo_nowcast_as_of():
    payload = json.loads((FIX / "federalregister" / "eo_documents.json").read_text())
    rows = counts.eo_rows(payload)
    assert len(rows) == len(payload["results"])
    raw = payload["results"][0]
    assert rows[0]["signing_date"].isoformat() == raw["signing_date"]
    assert rows[0]["pi_ts"] == counts.pi_time(raw["public_inspection_pdf_url"])
    assert rows[0]["pi_ts"].date() <= rows[0]["publication_date"]
    eos = pl.DataFrame(rows, schema=counts.EO_COLS)
    first = rows[0]["signing_date"]
    week_last = first + timedelta(days=6)
    before_pi = rows[0]["pi_ts"] - timedelta(minutes=1)
    after_pi = rows[0]["pi_ts"] + timedelta(minutes=1)
    # an EO counts only once it is on public inspection
    assert counts.eo_nowcast(eos, first, week_last, before_pi, 0)["partial"] == 0
    got = counts.eo_nowcast(eos, first, week_last, after_pi, 0)
    assert got["partial"] >= 1 and got["decided"] is True and got["p_model"] == 1.0


def test_mid_calibration_recovers_a_known_logistic_map():
    rng = np.random.default_rng(3)
    mid = rng.uniform(0.05, 0.95, 4000)
    lg = np.log(mid / (1 - mid))
    p = 1 / (1 + np.exp(-(0.3 + 1.4 * lg)))
    y = (rng.uniform(size=mid.size) < p).astype(int)
    rows = pl.DataFrame({"bid": mid - 0.01, "ask": mid + 0.01, "y": y})
    a, b = counts.fit_mid_calibration(rows)
    assert a == pytest.approx(0.3, abs=0.15) and b == pytest.approx(1.4, abs=0.15)
    ctl = counts.with_control(rows.head(3), (a, b))["p_ctrl"].to_numpy()
    assert np.all((ctl > 0) & (ctl < 1))
