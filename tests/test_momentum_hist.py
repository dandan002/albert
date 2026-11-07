"""Momentum/drift research helpers on recorded Kalshi payloads (settled market + its candles)."""

import json
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

from albert.research import momentum_hist as mh
from albert.venues import kalshi as k

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _frames(close_shift_h: float | None = None):
    market = json.loads((FIX / "settled_markets.json").read_text())
    candles = json.loads((FIX / "candles_live.json").read_text())
    info = {"KXNBAGAME": {"category": "Sports", "fee_type": "quadratic", "fee_multiplier": 1.0}}
    m = k.markets_frame(k.market_rows(market, info, "live"))
    c = pl.DataFrame(k.candle_rows(candles, candles["ticker"], 60))
    if close_shift_h is not None:   # move the actual close to just after the 2nd candle
        t2 = c.sort("end_ts")["end_ts"][1]
        m = m.with_columns(pl.lit(t2 + timedelta(hours=close_shift_h)).alias("close_time"))
    return c, m


def test_match_key_merges_maps_with_their_game_but_not_weather_days():
    assert mh.match_key("KXCS2MAP-26SEP031500PAINAYAW-2") == mh.match_key("KXCS2GAME-26SEP031500PAINAYAW")
    assert mh.match_key("KXHIGHNY-26OCT05") != mh.match_key("KXHIGHCHI-26OCT05")


def test_panel_lags_and_post_close_censoring_on_recorded_candles():
    c, m = _frames()
    p = mh.add_lags(mh.build_panel(c, m, prefixes=("KXNBAGAME",)), lags=(1,))
    assert p.height == c.height == 4        # all four candles end before the actual close
    assert p["y"].unique().to_list() == [0.0]   # the fixture market settled NO
    assert (p["h_to_exp"] > 0).all()
    with_both = p.filter(pl.col("ret_1").is_not_null())
    for r in with_both.iter_rows(named=True):
        assert r["ret_1"] == pytest.approx(r["smid"] - r["smid_1"])
    # Candles ending at or after the actual close carry stale, untradeable quotes: dropped.
    c2, m2 = _frames(close_shift_h=0.5)
    assert mh.build_panel(c2, m2, prefixes=("KXNBAGAME",)).height == 2


def test_take_prices_both_directions_with_taker_fees():
    f = pl.DataFrame({"bid": [0.40, 0.40], "ask": [0.43, 0.43], "y": [1.0, 1.0], "mid": [0.415, 0.415],
                      "spread": [0.03, 0.03], "fee_type": ["quadratic"] * 2, "fee_multiplier": [1.0, 1.0],
                      "d": [1, -1]})
    out = mh.take(f, pl.col("d"))
    yes, no = out.row(0, named=True), out.row(1, named=True)
    assert yes["price"] == pytest.approx(0.43) and no["price"] == pytest.approx(0.60)
    assert yes["fee"] == pytest.approx(0.02) and no["fee"] == pytest.approx(0.02)   # ceil(0.07pq)
    assert yes["pnl"] == pytest.approx(1 - 0.43 - 0.02) and no["pnl"] == pytest.approx(-0.62)
    assert yes["gross"] == pytest.approx(1 - 0.415) and no["gross"] == pytest.approx(-(1 - 0.415))
    wide = f.with_columns(pl.lit(0.2).alias("spread"))
    assert mh.take(wide, pl.col("d")).is_empty()      # never cross a wide quote


def test_clustered_t_uses_clusters_not_rows():
    f = pl.DataFrame({"cluster": ["a", "a", "b", "c"], "pnl": [0.1, 0.1, -0.05, 0.2], "risk": [0.5] * 4})
    r = mh.clustered(f)
    assert r["clusters"] == 3 and r["mean"] == pytest.approx(0.0875) and r["t"] is not None
    assert mh.clustered(f, weight="risk")["mean"] == pytest.approx(0.0875 / 0.5)
