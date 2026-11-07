import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from albert.research import series_age as sa
from albert.venues import kalshi as k

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _resp(name: str) -> dict:
    return json.loads((FIX / name).read_text())["response"]


def test_ticker_date_on_real_tickers():
    rain = _resp("series_age_events_rain.json")["events"]
    assert [sa.ticker_date(e["event_ticker"]) for e in rain] == [datetime(2026, 11, 6, tzinfo=UTC),
                                                                 datetime(2026, 7, 15, tzinfo=UTC)]
    assert sa.ticker_date("KXNBAGAME-26NOV02MIALAL") == datetime(2026, 11, 2, tzinfo=UTC)
    assert sa.ticker_date("KXFDAAPPROVE") is None


def test_event_list_rows_real_pages():
    rows = sa.event_list_rows(_resp("series_age_events_approve.json"), "KXTRUMPAPPROVE")
    assert len(rows) == 2 and [r["pos"] for r in rows] == [0, 1]
    assert all(r["mutually_exclusive"] is True and r["series_ticker"] == "KXTRUMPAPPROVE" for r in rows)
    rain = sa.event_list_rows(_resp("series_age_events_rain.json"), "KXRAIN", start_pos=200)
    assert [r["pos"] for r in rain] == [200, 201] and not any(r["mutually_exclusive"] for r in rain)


def test_probe_rows_real_historical_markets():
    rows = sa.probe_rows(_resp("series_age_hist_markets.json"))
    assert len(rows) == 2
    assert rows[0]["event_ticker"] == "KXTRUMPAPPROVE-26SEP02"
    assert rows[0]["open_time"] == datetime(2026, 9, 1, 14, 1, tzinfo=UTC)
    assert rows[0]["created_time"] <= rows[0]["open_time"] < rows[0]["close_time"]


def test_earliest_candidates_only_when_listing_predates_table():
    ev = pl.DataFrame(sa.event_list_rows(_resp("series_age_events_rain.json"), "KXRAIN"), schema=sa.EVENT_LIST_SCHEMA)
    tab = pl.DataFrame({"series_ticker": ["KXRAIN"], "open_min_table": [datetime(2026, 8, 16, 3, 31, tzinfo=UTC)]})
    cand = sa.earliest_candidates(ev, tab)
    assert cand["event_ticker"].to_list() == ["KXRAIN-26JUL15"]
    # table already reaches the earliest listed event: no probe needed
    tab2 = tab.with_columns(pl.lit(datetime(2026, 7, 14, tzinfo=UTC)).alias("open_min_table"))
    assert sa.earliest_candidates(ev, tab2).height == 0


def test_batch_candle_rows_real_payload():
    payload = json.loads((FIX / "candles_batch.json").read_text())
    rows = sa.batch_candle_rows(payload)
    assert rows and all(r["ticker"] for r in rows)
    assert any(r["yes_bid_close"] is not None and r["yes_ask_close"] is not None for r in rows)


def _markets() -> pl.DataFrame:
    rows = k.market_rows(_resp("series_age_hist_markets.json"), {"KXTRUMPAPPROVE": {"category": "Politics",
                                                                                     "fee_type": "quadratic",
                                                                                     "fee_multiplier": 1.0}}, "historical")
    m = pl.DataFrame(rows, schema=k.MARKET_SCHEMA).filter(pl.col("result").is_in(["yes", "no"]))
    return m.with_columns((pl.col("result") == "yes").cast(pl.Int8).alias("y"),
                          pl.coalesce("expected_expiration_time", "close_time").alias("t_ref"))


def test_decisions_on_real_markets_are_open_and_aged():
    m = _markets()
    assert m.height == 2
    launch = pl.DataFrame({"series_ticker": ["KXTRUMPAPPROVE"], "launch": [datetime(2026, 9, 1, 14, 1, tzinfo=UTC)],
                           "launch_censored": [False]})
    d = sa.decisions(m, launch)
    assert d.height > 0
    assert (d["t"] < d["close_time"]).all() and (d["t"] >= d["open_time"] + timedelta(hours=1)).all()
    assert (d["t"] < d["t_ref"]).all()
    assert d["age_bucket"].to_list() == ["a0_lt2w"] * d.height
    assert (d["age_days"] > 0).all() and (d["age_days"] < 14).all()


def test_age_bucket_edges():
    df = pl.DataFrame({"d": [0.0, 13.9, 14.0, 41.9, 42.0, 111.9, 112.0, 900.0]})
    got = df.select(sa.age_bucket(pl.col("d")).alias("b"))["b"].to_list()
    assert got == ["a0_lt2w", "a0_lt2w", "a1_2_6w", "a1_2_6w", "a2_6_16w", "a2_6_16w", "a3_old", "a3_old"]


def test_sell_yes_pnl_math():
    q = pl.DataFrame({"bid": [0.10, 0.0, 0.30], "y": [0, 0, 1], "fee_multiplier": [1.0, 1.0, 0.5]})
    out = sa.sell_yes_pnl(q)
    assert out.height == 2   # no bid -> cannot sell
    # fee on the NO price 0.90: ceil(0.07*0.9*0.1 * 1e4)/1e4 = 0.0063
    assert out["pnl"][0] == pytest.approx(0.10 - 0.0063)
    # half-fee series, NO at 0.70: ceil(0.035*0.7*0.3*1e4)/1e4 = 0.0074; YES won
    assert out["pnl"][1] == pytest.approx(0.30 - 1 - 0.0074)


def test_within_series_recovers_planted_age_effect():
    rng = np.random.default_rng(0)
    rows = []
    for si in range(60):
        level = rng.normal(0, 0.05)
        for j in range(80):
            bucket = "a0_lt2w" if j < 40 else "a3_old"
            mid = rng.uniform(0.05, 0.95)
            p = np.clip(mid + level + (-0.08 if bucket == "a0_lt2w" else 0.0), 0.01, 0.99)
            rows.append({"series_ticker": f"S{si}", "age_bucket": bucket, "mid": mid, "y": int(rng.random() < p),
                         "t": datetime(2026, 9, 1, tzinfo=UTC) + timedelta(hours=j)})
    res = {r["term"]: r for r in sa.within_series(pl.DataFrame(rows), label="x")}
    assert res["a0_lt2w"]["coef_c"] == pytest.approx(-8, abs=3)
    assert res["a0_lt2w"]["t_series"] < -3


def test_logit_fits_calibrated_slope_near_one():
    rng = np.random.default_rng(1)
    mid = rng.uniform(0.03, 0.97, 4000)
    y = (rng.random(4000) < mid).astype(int)
    q = pl.DataFrame({"g": ["x"] * 4000, "mid": mid, "y": y, "event": [f"e{i // 4}" for i in range(4000)],
                      "series_ticker": [f"s{i // 40}" for i in range(4000)]})
    fit = sa.logit_fits(q, ["g"]).row(0, named=True)
    assert fit["b"] == pytest.approx(1.0, abs=0.12) and abs(fit["a"]) < 0.15


def test_ladder_sums_require_complete_exhaustive_ladder():
    t = datetime(2026, 9, 10, tzinfo=UTC)
    o, c = t - timedelta(days=1), t + timedelta(days=1)
    m = pl.DataFrame({"event_ticker": ["E1"] * 3 + ["E2"] * 2, "open_time": [o] * 5, "close_time": [c] * 5,
                      "y": [1, 0, 0, 0, 0]})
    q = pl.DataFrame({"event_ticker": ["E1"] * 3 + ["E2"] * 2, "kind": ["h24"] * 5, "t": [t] * 5,
                      "mid": [0.5, 0.35, 0.25, 0.4, 0.4], "ask": [0.52, 0.37, 0.27, 0.42, 0.42],
                      "bid": [0.48, 0.33, 0.23, 0.38, 0.38], "age_days": [3.0] * 5, "age_bucket": ["a0_lt2w"] * 5,
                      "event_age_days": [1.0] * 5, "series_ticker": ["S"] * 5, "category": ["Politics"] * 5,
                      "split": ["dev"] * 5, "legs_capped": [False] * 5})
    flags = pl.DataFrame({"event_ticker": ["E1", "E2"], "mutually_exclusive": [True, True]})
    lad = sa.ladder_sums(q, m, flags)
    assert lad["event_ticker"].to_list() == ["E1"]   # E2 has no YES leg: not exhaustive
    assert lad["overround_mid"][0] == pytest.approx(0.10)
