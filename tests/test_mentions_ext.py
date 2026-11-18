"""Mentions extensions: pricing, fills, budget replay, salience and post-speech sweep on recorded
payloads (trimmed copies of live Kalshi minute candles and Wikimedia pageviews)."""

import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from albert.research import mentions_ext as X
from albert.research import xcat

FIX = Path(__file__).parent / "fixtures"


def test_word_titles_split_alternatives_and_drop_counts():
    assert X.word_titles("Marco / Rubio (3+ times)") == ["Marco", "Rubio"]
    assert X.word_titles("fake news") == ["Fake_news"]
    assert X.word_titles("AI / Artificial Intelligence") == ["AI", "Artificial_Intelligence"]
    assert X.word_titles(None) == []


def test_no_bid_floor_margin_and_post_only_cap():
    fair = np.array([0.30, 0.30, 0.95])
    bid = np.array([0.25, 0.50, 0.90])
    got = X.no_bid(fair, bid, 0.12)
    assert got[0] == pytest.approx(0.58)            # floor(1 - 0.30 - 0.12)
    assert got[1] == pytest.approx(0.49)            # capped one tick inside the YES bid (1 - 0.50 - 0.01)
    assert math.isnan(got[2])                       # under 1c: no order
    # the frozen calibration lowers fair below the raw mid for cheap legs
    assert X.fair_cal(np.array([0.10]))[0] < 0.10


def test_taker_fee_rounds_up_per_order():
    assert X.taker_fee(0.98, 1) == pytest.approx(0.01)
    assert X.taker_fee(0.98, 10) == pytest.approx(0.02)     # 0.0137 -> 2c for the order
    assert X.taker_fee(0.50, 1) == pytest.approx(0.02)      # 0.0175 -> 2c


def test_wilson_lower_bound():
    lb = X.wilson_lower(95, 100)
    assert 0.89 < lb < 0.95
    assert X.wilson_lower(10, 10) < 1.0
    assert X.wilson_lower(0, 50) == pytest.approx(0.0, abs=1e-12)


def _hours(rows):
    return pl.DataFrame(rows, schema={"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "hi": pl.Float64}, orient="row")


def test_hourly_fills_need_a_print_strictly_through_the_bid_inside_the_window():
    t0 = datetime(2026, 10, 1, 12, tzinfo=UTC)
    O = pl.DataFrame({"row": [0, 1, 2], "ticker": ["A", "B", "C"], "t": [t0] * 3, "end_cb": [t0 + timedelta(hours=3)] * 3,
                      "price": [0.60, 0.60, 0.60]}).with_columns(pl.col("row").cast(pl.UInt32))
    H = _hours([("A", t0 + timedelta(hours=2), 0.40),      # exactly our ask (1 - 0.60): not through
                ("B", t0 + timedelta(hours=2), 0.41),      # through: fill
                ("C", t0 + timedelta(hours=5), 0.90)])     # through, but after the window
    f = X.hourly_fills(O, H)
    assert f.sort("ticker")["filled"].to_list() == [False, True, False]
    fb = f.filter(pl.col("ticker") == "B")["fill_ts"][0]
    assert t0 < fb <= t0 + timedelta(hours=3)


def _orders(event, tau, prices, end_h=6, filled=None):
    n = len(prices)
    filled = filled or [False] * n
    return pl.DataFrame({
        "event": [event] * n, "ticker": [f"{event}-{i}" for i in range(n)], "t": [tau] * n,
        "end_o": [tau + timedelta(hours=end_h)] * n, "price": prices, "filled": filled,
        "fill_ts": [tau + timedelta(minutes=30) if f else None for f in filled],
        "settlement_ts": [tau + timedelta(hours=end_h + 2)] * n, "y": [0] * n,
    }, schema_overrides={"fill_ts": pl.Datetime("us", "UTC")})


def test_capped_book_respects_value_at_risk_and_event_slots():
    t0 = datetime(2026, 10, 1, 12, tzinfo=UTC)
    o = pl.concat([_orders("E1", t0, [0.50] * 20),                     # $10 of bids
                   _orders("E2", t0 + timedelta(hours=1), [0.40] * 20),
                   _orders("E3", t0 + timedelta(hours=2), [0.10] * 5)])
    b = X.capped_book(o, var=15.0, max_events=2)
    by = dict(b.group_by("event").agg(pl.col("price").sum()).iter_rows())
    assert by["E1"] == pytest.approx(10.0)          # the whole event fits
    assert by["E2"] == pytest.approx(4.8)           # only the room left: 12 x 0.40 <= $5
    assert "E3" not in by                           # two events already resting
    later = _orders("E4", t0 + timedelta(hours=9), [0.10] * 5)
    b2 = X.capped_book(pl.concat([o, later]), var=15.0, max_events=2)
    assert "E4" in set(b2["event"])                 # slots free again after E1 and E2 ended


def test_capped_book_counts_unsettled_fills():
    t0 = datetime(2026, 10, 1, 12, tzinfo=UTC)
    first = _orders("E1", t0, [0.70] * 20, end_h=1, filled=[True] * 20)       # $14 filled, settles at t0 + 3 h
    second = _orders("E2", t0 + timedelta(hours=2), [0.50] * 10)
    b = X.capped_book(pl.concat([first, second]), var=15.0, max_events=2)
    assert b.filter(pl.col("event") == "E2")["price"].sum() == pytest.approx(1.0)


def test_pageview_rows_parse_recorded_payload():
    payload = json.loads((FIX / "wikimedia" / "pageviews_tariff.json").read_text())
    rows = X.pageview_rows(payload, "Tariff")
    assert len(rows) == len(payload["items"]) == 50
    assert rows[0]["day"] == date(2026, 8, 1) and rows[0]["views"] == payload["items"][0]["views"]


def test_salience_z_scores_use_only_published_days():
    payload = json.loads((FIX / "wikimedia" / "pageviews_tariff.json").read_text())
    pv = pl.DataFrame(X.pageview_rows(payload, "Tariff"))
    d1 = date(2026, 9, 19)                                      # the fixture's last day
    t = datetime(2026, 9, 20, 6, 0, tzinfo=UTC) + timedelta(hours=1)   # t - 30 h falls on 09-19
    o = pl.DataFrame({"ticker": ["K"], "t": [t], "wiki": ["Tariff"]})
    z = X.salience(o, pv)
    v = {r["day"]: r["views"] for r in X.pageview_rows(payload, "Tariff")}
    base = np.log1p([v[d1 - timedelta(days=k)] for k in range(8, 38)])
    want = (np.log1p(v[d1]) - base.mean()) / max(base.std(ddof=1), 0.05)
    assert z["z1"][0] == pytest.approx(want)
    # one hour earlier, t - 30 h is on 09-18: the 09-19 count is not yet published
    z2 = X.salience(o.with_columns(pl.lit(t - timedelta(hours=2)).alias("t")), pv)
    assert z2["views_d1"][0] == pytest.approx(v[date(2026, 9, 18)])


def _minute_fixture():
    payload = json.loads((FIX / "kalshi" / "mentions_ext_minute_batch.json").read_text())
    rows = [r for v in xcat.batch_rows(payload, 1).values() for r in v]
    Mn = X._ohlc(pl.DataFrame(rows, schema=xcat.OHLC_SCHEMA).lazy()).collect()
    start = datetime.fromtimestamp(1791118200, UTC) + X.END_BEFORE      # the fetch began at start - 10 min
    tk = sorted(Mn["ticker"].unique().to_list())
    legs = pl.DataFrame({"event": ["KXFNSMENTION-26OCT04"] * len(tk), "ticker": tk, "start": [start] * len(tk),
                         "split": ["holdout"] * len(tk), "et_date": [date(2026, 10, 4)] * len(tk),
                         "close_time": [start + timedelta(hours=6)] * len(tk),
                         "settlement_ts": [start + timedelta(hours=7)] * len(tk),
                         "y": [1 if x.endswith(("ICE", "MIDT")) else 0 for x in tk], "fee_multiplier": [1.0] * len(tk)})
    return Mn, legs, start


def test_said_times_and_sweep_on_recorded_minutes():
    Mn, legs, start = _minute_fixture()
    st = X.said_times(Mn, legs)
    assert set(st["ticker"]) == {"KXFNSMENTION-26OCT04-ICE", "KXFNSMENTION-26OCT04-MIDT"}
    assert (st["said_ts"] > start).all()
    sig = X.sweep_signals(legs, Mn)
    assert {"delay60", "quiet30"} <= set(sig["signal"])
    T = X.sweep_trades(legs, Mn, sig)
    assert T.height > 0
    assert (T["bid"] >= X.MIN_BID - 1e-9).all() and (T["t_over"] < T["close_time"]).all()
    r = T.row(0, named=True)
    assert r["no_px"] == pytest.approx(round(1 - r["bid"], 2))
    assert r["pnl"] == pytest.approx((1 - r["y"]) - r["no_px"] - X.taker_fee(r["no_px"], 1))


def test_quiet_time_is_as_of():
    s = datetime(2026, 10, 1, 0, tzinfo=UTC)
    said = [s + timedelta(minutes=m) for m in (2, 5, 40)]
    assert X.quiet_time(said, s, timedelta(minutes=15)) == s + timedelta(minutes=20)
    assert X.quiet_time(said, s, timedelta(minutes=40)) == s + timedelta(minutes=80)
    assert X.quiet_time([], s, timedelta(minutes=15)) is None


def test_word_key_matches_across_venues():
    assert X.word_key("Million / Billion / Trillion 25+ times") == X.word_key("Billion / Million / Trillion (25+ times)")
    assert X.word_key("Fake News") == "fake news"
    assert X.word_key("Job 6+ times") == "job #6"


def test_pm_market_rows_parse_recorded_gamma_event():
    events = json.loads((FIX / "polymarket" / "mentions_ext_gamma_event.json").read_text())
    rows = X.pm_market_rows(events)
    assert len(rows) == len(events[0]["markets"])
    r = rows[0]
    assert r["word_key"] == "afford / affordability / affordable"
    assert r["token"] == json.loads(events[0]["markets"][0]["clobTokenIds"])[0]
    assert r["pm_y"] in (0, 1) and r["pm_end"].tzinfo is not None
    assert X.pm_speaker(events[0]["title"]) == "trump"
    assert X.pm_speaker("What will Trump say this week? (November 9 - November 15)") is None


def test_pm_signal_uses_only_buckets_that_ended_before_t():
    hist = json.loads((FIX / "polymarket" / "mentions_ext_prices_history.json").read_text())
    tok = "6456258930482202730769267237381877247221051099841702273893827256846495384515"
    h = pl.DataFrame([{"token": tok, "bucket_ts": datetime.fromtimestamp(x["timestamp"], UTC),
                       "secs": int(x["resolution_seconds"]), "pm_price": float(x["price"])} for x in hist["data"]])
    b = h.sort("bucket_ts")
    t = b["bucket_ts"][3] + timedelta(minutes=30)          # inside bucket 3: its price is not known yet
    o = pl.DataFrame({"ticker": ["K"], "t": [t], "mid": [0.5]})
    pairs = pl.DataFrame({"ticker": ["K"], "token": [tok]})
    got = X.pm_signal(o, pairs, h)
    assert got["pm_price"][0] == pytest.approx(b["pm_price"][2])
    assert got["pm_gap"][0] == pytest.approx(b["pm_price"][2] - 0.5)


def test_decision_rows_quote_age_and_early_start_exclusion():
    start = datetime(2026, 10, 2, 0, tzinfo=UTC)
    U = pl.DataFrame({"event": ["E", "E", "F"], "ticker": ["E-A", "E-B", "F-A"], "start": [start, start, start],
                      "open_time": [start - timedelta(days=3)] * 3, "close_time": [start + timedelta(hours=3)] * 3})
    t = start - timedelta(hours=6)
    H = pl.DataFrame({"ticker": ["E-A", "E-B", "F-A", "F-A"],
                      "end_ts": [t - timedelta(hours=2), t - timedelta(hours=30), t - timedelta(hours=1), t - timedelta(hours=20)],
                      "bid": [0.20, 0.30, 0.98, 0.10], "ask": [0.25, 0.35, 0.99, 0.15],
                      "hi": [None, None, None, None], "lo": [None, None, None, None], "vol": [0.0] * 4},
                     schema_overrides={"hi": pl.Float64, "lo": pl.Float64})
    Q = X.decision_rows(U, H, 6)
    assert Q["ticker"].to_list() == ["E-A"]            # E-B's quote is > 24 h old; F jumped to 'said' since t - 18 h
    assert Q["end"][0] == start - X.END_BEFORE and Q["fair"][0] < Q["mid"][0]
