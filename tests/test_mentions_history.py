"""Tests for the frozen pre-speech NO-bid rule over the Mentions history.

Fixtures are trimmed copies of real Kalshi responses recorded by this study
(``tests/fixtures/kalshi/mentions_history_*.json``): one 2025 Trump speech event (historical
candlesticks with bare field names, historical prints in the resting window, its milestone) and a
live batch-candles page (``*_dollars`` field names)."""

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import mentions as M
from albert.research import mentions_history as H
from albert.research.fundamental import maker_fills
from albert.venues import kalshi as k

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _load(name: str):
    return json.loads((FIX / f"mentions_history_{name}.json").read_text())


def _event() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Legs (with start), hourly candles and prints of the recorded historical event."""
    mk = k.markets_frame(k.market_rows(_load("markets"), {}, "historical"), non_financial_only=False)
    starts = pl.DataFrame(M.milestone_rows(_load("milestones")))
    lg = M.legs(mk).join(starts.select("event", "start", "ms_updated"), on="event").with_columns(
        pl.lit("quadratic").alias("fee_type"), pl.lit(1.0).alias("fee_multiplier"))
    candles = pl.DataFrame([r for tk, page in _load("candles").items() for r in H.candle_rows(page["candlesticks"], tk)],
                           schema=H.CANDLE_SCHEMA)
    trades = pl.DataFrame([r for page in _load("trades") for r in k.trade_rows(page)], schema=H.TRADE_SCHEMA)
    return lg, candles, trades


def test_frozen_fair_value_is_the_round2_dev_fit():
    mid = np.array([0.02, 0.3, 0.5, 0.8, 0.999])
    z = -0.1156 + 1.0474 * np.log(np.clip(mid, 0.005, 0.995) / (1 - np.clip(mid, 0.005, 0.995)))
    assert np.allclose(H.fair_cal(mid), np.clip(1 / (1 + np.exp(-z)), 0.003, 0.997))
    assert abs(H.fair_cal(np.array([0.5]))[0] - 0.4711) < 1e-3     # YES bias: fair below the mid
    assert H.MARGINS == (0.08, 0.12) and H.LEAD == timedelta(hours=6) and H.END_MARGIN == timedelta(minutes=10)


def test_candle_rows_read_historical_and_live_spellings():
    hist = _load("candles")
    tk, page = next(iter(hist.items()))
    rows = H.candle_rows(page["candlesticks"], tk)
    assert rows and all(r["yes_bid_close"] is not None and r["yes_ask_close"] is not None for r in rows)
    traded = [r for r in rows if r["price_high"] is not None]
    assert traded and all(r["price_low"] <= r["price_high"] for r in traded)
    live = _load("batch_candles")
    m = live["markets"][0]
    lrows = H.candle_rows(m["candlesticks"], m["market_ticker"])
    assert lrows and any(r["price_high"] is not None for r in lrows)
    assert all(0 <= r["yes_bid_close"] <= r["yes_ask_close"] <= 1 for r in lrows if r["yes_ask_close"] is not None)


def test_decision_rows_quote_at_start_minus_6h():
    lg, candles, _ = _event()
    Q, info = H.decision_rows(lg, candles)
    assert Q.height > 0 and info["rows"] == Q.height
    assert (Q["t"] == Q["start"] - H.LEAD).all()
    assert ((Q["t"] - Q["q_ts"]) <= H.MAX_AGE).all() and (Q["q_ts"] <= Q["t"]).all()
    assert (Q["end"] <= Q["start"] - H.END_MARGIN).all()
    assert (Q["open_time"] <= Q["t"]).all()


def test_fast_fills_match_the_harness_rule_on_recorded_prints():
    lg, candles, trades = _event()
    Q, _ = H.decision_rows(lg, candles)
    for m in H.MARGINS:
        f = H.fast_fills(Q, trades, m, p_col="fv_cal", end_col="end")
        ref = maker_fills(Q.with_columns(pl.col("fv_cal").alias("p")), trades, m)
        assert f.height == ref.height > 0
        assert f["filled"].sum() == ref["filled"].sum()
        assert abs(f["pnl"].sum() - ref["pnl"].sum()) < 1e-9


def test_fill_candidates_contain_every_print_fill():
    """Legs left out of the print fetch must provably have no fill: every leg filled from prints
    (both fair values, both margins, both sides) is a candidate from the hourly ranges."""
    lg, candles, trades = _event()
    Q, _ = H.decision_rows(lg, candles)
    cand = set(H.fill_candidates(Q, candles)["ticker"])
    filled = set()
    for fv in ("fv_cal", "fv_mid"):
        for m in H.MARGINS:
            f = H.fast_fills(Q, trades, m, p_col=fv, end_col="end").filter("filled")
            filled |= set(Q.with_row_index("row").join(f, on="row")["ticker"])
    assert filled and filled <= cand


def test_proxy_never_misses_a_print_fill_and_breaker_cuts_the_window():
    lg, candles, trades = _event()
    Q, _ = H.decision_rows(lg, candles)
    for m in H.MARGINS:
        a = H.fast_fills(Q, trades, m, p_col="fv_cal", end_col="end")
        b = H.proxy_fills(Q, candles, m, p_col="fv_cal", end_col="end")
        j = a.join(b, on=["row", "side"], suffix="_x")
        assert j.height == a.height
        assert not (j["filled"] & ~j["filled_x"]).any()
    kt = H.kill_times(Q, trades)
    kp = H.proxy_kill(Q, candles)
    if kt.height and kp.height:
        assert kp["kill_hr"][0] <= kt["kill_ts"][0]


def test_through_volume_counts_only_prints_beyond_our_price():
    lg, candles, trades = _event()
    Q, _ = H.decision_rows(lg, candles)
    f = H.fast_fills(Q, trades, 0.08, p_col="fv_cal", end_col="end").join(
        Q.with_row_index("row").select("row", "ticker", "t", "end"), on="row")
    v = H.through_volume(f, trades, "end")
    assert (v[~f["filled"].to_numpy()] == 0).all()
    assert (v[f["filled"].to_numpy()] > 0).all()


def test_clustered_slope_equals_ols_with_singleton_clusters():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 2, 200)
    y = 3 - 1.5 * x + rng.normal(0, 1, 200)
    a, b, _ta, tb = H.clustered_slope(x, y, np.arange(200))
    bb, aa = np.polyfit(x, y, 1)
    assert abs(a - aa) < 1e-9 and abs(b - bb) < 1e-9 and tb < -5


def test_spread_bands():
    df = pl.DataFrame({"s": [0.01, 0.05, 0.06, 0.15, 0.16, 0.5]})
    got = df.select(H.spread_band(pl.col("s")).alias("b"))["b"].to_list()
    assert got == ["a <=5c", "a <=5c", "b 6-15c", "b 6-15c", "c >15c", "c >15c"]


def test_hours_without_a_trade_range_are_candidates_unless_volume_is_zero():
    """Round-2 candles carry no trade range: an hour with volume > 0 must count as a possible fill;
    an hour with zero volume provably had no print."""
    lg, candles, _ = _event()
    Q, _ = H.decision_rows(lg, candles)
    blind = candles.with_columns(pl.lit(None, dtype=pl.Float64).alias("price_high"), pl.lit(None, dtype=pl.Float64).alias("price_low"))
    traded = set(blind.join(Q.select("ticker", "t", "end"), on="ticker").filter(
        (pl.col("end_ts") > pl.col("t")) & (pl.col("end_ts") - timedelta(hours=1) < pl.col("end")) & (pl.col("volume") > 0))["ticker"])
    assert traded and traded <= set(H.fill_candidates(Q, blind)["ticker"])
    assert set(H.fill_candidates(Q, candles)["ticker"]) <= set(H.fill_candidates(Q, blind)["ticker"])
