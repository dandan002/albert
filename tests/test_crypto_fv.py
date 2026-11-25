"""crypto_fv: parsers on trimmed recorded responses (tests/fixtures/crypto_fv), pricing and fill logic."""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from albert.research import crypto_fv as cf

FX = Path(__file__).parent / "fixtures" / "crypto_fv"


def load(name: str) -> dict:
    return json.loads((FX / f"{name}.json").read_text())["response"]


def test_event_rows():
    rows = cf.event_rows(load("events"))
    assert rows and all(r["series_ticker"] == "KXBTCD" for r in rows)
    assert all(isinstance(r["strike_date"], datetime) for r in rows)
    assert rows[0]["event_ticker"].startswith("KXBTCD-")


@pytest.mark.parametrize("name", ["markets_live", "markets_hist"])
def test_market_rows_daily(name):
    rows = cf.market_rows(load(name), "x")
    assert rows
    r = rows[0]
    assert r["strike_type"] == "greater" and r["floor_strike"] > 0
    assert r["result"] in ("yes", "no")
    assert r["expiration_value"] and r["expiration_value"] > 1000          # BTC index level
    assert r["close_time"] > r["open_time"]
    assert "sixty seconds" in r["rules_primary"]
    # the 5 pm event lists ~25 h before its close
    if r["event_ticker"].endswith("17"):
        assert timedelta(hours=24) <= r["close_time"] - r["open_time"] <= timedelta(hours=26)


def test_market_rows_range_and_barrier():
    rg = cf.market_rows(load("markets_range"), "live")
    assert {r["strike_type"] for r in rg} & {"between", "less", "greater"}
    bt = [r for r in rg if r["strike_type"] == "between"]
    assert bt and all(r["cap_strike"] > r["floor_strike"] for r in bt)
    bar = cf.market_rows(load("markets_barrier"), "live")
    assert bar and "trimmed mean" in (bar[0]["rules_secondary"] or "")
    # barrier expiration values come back as "$80,047.53"
    assert all(r["expiration_value"] is None or r["expiration_value"] > 0 for r in bar)


def test_candle_rows_both_spellings():
    b = load("candles_batch")["markets"][0]
    rows = cf.candle_rows(b["candlesticks"], b["market_ticker"])
    h = load("candles_hist")
    rows_h = cf.candle_rows(h["candlesticks"], "X")
    for rr in (rows, rows_h):
        assert rr and all(r["period_minutes"] == 60 for r in rr)
        assert all(r["yes_bid_close"] is not None and r["yes_ask_close"] is not None for r in rr)
        assert all(r["end_ts"].minute == 0 for r in rr)
    df = pl.DataFrame(rows + rows_h, schema=cf.CANDLE_SCHEMA)
    assert df.height == len(rows) + len(rows_h)


def test_coinbase_and_dvol_rows():
    cb = cf.coinbase_rows(load("coinbase"), "BTC")
    assert len(cb) > 100 and all(r["low"] <= r["close"] <= r["high"] for r in cb)
    dv = cf.dvol_rows(load("dvol"), "BTC")
    assert dv and all(10 < r["dvol"] < 300 for r in dv)


def test_vol_engine_on_recorded_candles():
    cb = pl.DataFrame(cf.coinbase_rows(load("coinbase"), "BTC"), schema=cf.SPOT_SCHEMA)
    e = cf.VolEngine(cb)
    t = cb["ts"].max()
    assert e.spot(t + timedelta(minutes=5)) == pytest.approx(cb.sort("ts")["close"][-1])
    v = e.variance(t.replace(minute=0), 24, "har")
    assert v is not None and 0 < v < 0.05            # daily variance of BTC: well under 5%^2... of 1
    assert e.variance(t.replace(minute=0), 48, "har") > v


def test_pricing_shapes():
    k = np.array([90.0, 100.0, 110.0])
    pa = cf.p_above(100.0, k, 0.01, 5.0)
    assert pa[0] > pa[1] > pa[2]
    assert pa[1] == pytest.approx(0.5, abs=0.03)
    # one-touch above the spot is ~2x finishing above for a driftless walk
    pt = cf.p_touch(100.0, 110.0, 0.01, 1000.0, "max")
    assert pt == pytest.approx(2 * cf.p_above(100.0, 110.0, 0.01, 1000.0), rel=0.15)
    assert cf.p_touch(100.0, 90.0, 0.01, 1000.0, "min") > 1 - cf.p_above(100.0, 90.0, 0.01, 1000.0)
    # fat tails: more mass far out than the normal at the same variance
    assert cf.p_above(100.0, 140.0, 0.01, 4.0) > cf.p_above(100.0, 140.0, 0.01, 1000.0)


def test_maker_fill_requires_trade_through():
    t0 = datetime(2026, 11, 22, 12, tzinfo=UTC)
    row = {"ticker": "M", "event_ticker": "E", "series_ticker": "KXBTCD", "coin": "BTC", "kind": "above",
           "family": "daily", "split": "holdout", "t": t0, "hours": 4.0, "settlement_ts": t0 + timedelta(hours=4),
           "close_time": t0 + timedelta(hours=4), "y": 1, "bid": 0.40, "ask": 0.45, "p_model": 0.52}
    r = pl.DataFrame([row])
    candles = pl.DataFrame({"ticker": ["M", "M"], "end_ts": [t0 + timedelta(hours=1), t0 + timedelta(hours=2)],
                            "price_low": [0.42, 0.40], "price_high": [0.46, 0.47]})
    # YES bid at floor(0.52 - 0.05) = 0.47 >= ask: crossing, so no YES order; NO bid at 0.43 < 0.60
    tr = cf.maker_trades(r, "p_model", 0.05, candles, {"KXBTCD": ("quadratic", 1.0)})
    assert tr.height == 0                       # a NO fill needs a YES print above 0.57
    tr = cf.maker_trades(r, "p_model", 0.10, candles, {"KXBTCD": ("quadratic", 1.0)})
    # YES bid 0.42: the 0.42 low touches but does not trade through; the 0.40 low does (2nd candle)
    yes = tr.filter(pl.col("side") == "yes")
    assert yes.height == 1 and yes["t_in"][0] == t0 + timedelta(hours=2) and yes["price"][0] == pytest.approx(0.42)
    assert yes["fee"][0] == 0.0                  # quadratic series: no maker fee


def test_taker_edge_net_of_fee():
    t0 = datetime(2026, 11, 22, 12, tzinfo=UTC)
    base = {"ticker": "M", "event_ticker": "E", "series_ticker": "KXBTCD", "coin": "BTC", "kind": "above",
            "split": "dev", "t": t0, "hours": 4.0, "settlement_ts": t0, "close_time": t0, "y": 0, "bid": 0.30, "ask": 0.32}
    tr = cf.taker_trades(pl.DataFrame([{**base, "p_model": 0.40}]), "p_model", 0.04, {})
    assert tr.height == 1 and tr["side"][0] == "yes" and tr["price"][0] == pytest.approx(0.32)
    assert tr["fee"][0] == pytest.approx(math.ceil(0.07 * 6 * 0.32 * 0.68 * 100) / 100)
    assert cf.taker_trades(pl.DataFrame([{**base, "p_model": 0.34}]), "p_model", 0.04, {}).height == 0
    no = cf.taker_trades(pl.DataFrame([{**base, "p_model": 0.20}]), "p_model", 0.04, {})
    assert no.height == 1 and no["side"][0] == "no" and no["price"][0] == pytest.approx(0.70)


def test_holm_monotone():
    adj = cf.holm([0.01, 0.04, 0.03, 0.5])
    assert adj[0] == pytest.approx(0.04) and adj[3] == pytest.approx(0.5)
    assert all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03, 0.5], strict=True))
