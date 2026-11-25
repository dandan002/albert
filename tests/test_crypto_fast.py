"""crypto_fast: parsers on trimmed copies of recorded responses (tests/fixtures/crypto_fast)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import crypto_fast as cf

FIX = Path(__file__).parent / "fixtures" / "crypto_fast"


def _resp(name: str):
    return json.loads((FIX / name).read_text())["response"]


def test_market_row_settled_and_strike_chain():
    ms = _resp("kalshi_markets_settled.json")["markets"]
    rows = [cf.market_row(m, "live") for m in ms]
    a, b = rows
    assert a["ticker"] == "KXBTC15M-26NOV221945-45" and a["series"] == "KXBTC15M"
    assert a["close_time"] == datetime(2026, 11, 23, 0, 45, tzinfo=UTC)
    assert (a["close_time"] - a["open_time"]).total_seconds() == 900
    assert a["floor_strike"] == 61088.58 and a["expiration_value"] == 61082.71 and a["result"] == "no"
    # the strike of a window is the previous window's settlement average
    assert b["close_time"] == a["open_time"] and b["expiration_value"] == a["floor_strike"]
    hist = [cf.market_row(m, "historical") for m in _resp("kalshi_hist_markets.json")["markets"]]
    assert all(r["series"] == "KXBTC15M" and r["result"] in ("yes", "no") for r in hist)


def test_ticker_from_close_time_matches_listing():
    for m in _resp("kalshi_markets_settled.json")["markets"]:
        close = datetime.fromisoformat(m["close_time"])
        assert cf.ticker_15m("KXBTC15M", close) == m["ticker"]
        assert cf.window_close(close.replace(minute=close.minute - 1)) == close
    # a boundary belongs to the next window
    t = datetime(2026, 11, 23, 0, 45, tzinfo=UTC)
    assert cf.window_close(t) == datetime(2026, 11, 23, 1, 0, tzinfo=UTC)


def test_k1m_rows_historical_spelling():
    rows = cf.k1m_rows(_resp("kalshi_hist_candles_1m.json")["candlesticks"], "KXBTC15M-26SEP101200-00")
    assert len(rows) == 3
    assert rows[0]["ts"] == datetime.fromtimestamp(1789055160, tz=UTC)
    assert rows[0]["bid"] == 0.53 and rows[0]["ask"] == 0.54
    assert rows[1]["bid"] == 0.23 and rows[1]["volume"] > 0


def test_coinbase_rows_ts_is_minute_end():
    rows = cf.coinbase_rows(_resp("coinbase_candles.json"), "BTC-USD")
    assert rows[0]["ts"] == datetime.fromtimestamp(1789056000 + 60, tz=UTC)
    assert rows[0]["close"] == 77258.82 and rows[0]["low"] <= rows[0]["close"] <= rows[0]["high"]


def test_live_parsers_on_recorded_lines():
    k = cf.kalshi_poll_rows((FIX / "live_kalshi.jsonl").read_text().splitlines())
    assert k and all(r["ticker"].startswith(r["series"]) for r in k)
    assert all(r["bid"] is not None and r["ask"] is not None and 0 < r["bid"] < r["ask"] < 1 for r in k)
    assert all(r["t_recv"] > r["t_send"] for r in k)
    assert cf._book_top(_resp("kalshi_orderbook.json")) == (None, None, None, None)   # settled: empty book
    c = cf.coinbase_tick_rows((FIX / "live_coinbase.jsonl").read_text().splitlines())
    assert c and all(r["bid"] <= r["ask"] for r in c) and c[0]["ts_server"].tzinfo is not None
    kr = cf.kraken_trade_rows((FIX / "live_kraken.jsonl").read_text().splitlines())
    assert kr and kr[0]["symbol"].endswith("/USD") and kr[0]["price"] > 0
    hl = cf.hl_trade_rows((FIX / "live_hl.jsonl").read_text().splitlines())
    assert hl and hl[0]["coin"] in ("BTC", "SOL") and hl[0]["price"] > 0


def test_iso_ns_any_fraction():
    assert cf.iso_ns("2026-11-23T01:08:21.81999Z") == datetime(2026, 11, 23, 1, 8, 21, 819990, tzinfo=UTC)
    assert cf.iso_ns("2026-11-23T01:06:56.824609462Z").microsecond == 824609


def test_fair_prob_properties():
    sd = np.full(3, 0.001)
    tau = np.full(3, 600.0)
    p = cf.fair_prob(np.array([-0.002, 0.0, 0.002]), sd, tau)
    assert abs(p[1] - 0.5) < 1e-12 and p[0] < 0.5 < p[2] and abs(p[0] + p[2] - 1) < 1e-12
    # the averaging window shrinks the variance at tau = 60 s (20 s of variance, not 60)
    near = cf.fair_prob(np.array([0.0005]), np.array([0.001]), np.array([60.0]))
    far = cf.fair_prob(np.array([0.0005]), np.array([0.001]), np.array([100.0]))
    assert near[0] > far[0]


def test_in_sample_is_deterministic_and_volume_blind():
    t = "KXBTC15M-26NOV221945-45"
    assert cf.in_sample(t, 0.15) == cf.in_sample(t, 0.15)
    share = np.mean([cf.in_sample(f"KXBTC15M-X{i}", 0.15) for i in range(4000)])
    assert 0.12 < share < 0.18


def test_take_trades_fee_and_payoff():
    g = pl.DataFrame({
        "ticker": ["A", "A", "B"], "event_ticker": ["A", "A", "B"], "m": [3, 4, 3],
        "fair": [0.70, 0.90, 0.30], "bid": [0.55, 0.60, 0.45], "ask": [0.56, 0.61, 0.46],
        "ask_next": [0.60, 0.80, 0.40], "bid_next": [0.58, 0.79, 0.39],
        "ask_hi_next": [0.65, 0.85, 0.47], "bid_lo_next": [0.50, 0.70, 0.35], "y": [1, 1, 0],
        "date": [datetime(2026, 9, 1).date()] * 3,
    })
    tr = cf.take_trades(g, 0.10, (1, 14)).sort("ticker")
    assert tr["side"].to_list() == ["yes", "no"]            # first signal per window only
    a = tr.row(0, named=True)
    assert a["price"] == 0.56 and a["n"] == 8                 # $5 / 0.56 -> 8 contracts
    assert abs(a["fee"] - 0.14) < 1e-9                        # ceil(0.07*8*0.56*0.44*100)/100
    assert abs(a["pnl_c"] - 100 * (1 - 0.56 - 0.14 / 8)) < 1e-9
    b = tr.row(1, named=True)
    assert abs(b["price"] - 0.55) < 1e-9 and b["payoff"] == 1


def test_holm():
    adj = cf.holm([0.01, 0.04, 0.03])
    assert adj[0] == 0.03 and adj[2] == 0.06 and adj[1] == 0.06


def test_historical_expiration_value_with_thousands_separator():
    m = _resp("kalshi_hist_market_comma.json")["markets"][0]
    r = cf.market_row(m, "historical")
    assert r["expiration_value"] == 77362.10 and r["floor_strike"] == 77362.28


def test_take_favourite_side_and_time_filter():
    sat = datetime(2026, 9, 5, 14, 5, tzinfo=UTC)    # a Saturday, 14:05 UTC
    mon = datetime(2026, 9, 7, 14, 5, tzinfo=UTC)
    g = pl.DataFrame({
        "ticker": ["A", "B", "C"], "event_ticker": ["A", "B", "C"], "m": [5, 5, 5], "t": [sat, sat, mon],
        "mid": [0.705, 0.255, 0.705], "bid": [0.70, 0.25, 0.70], "ask": [0.71, 0.26, 0.71],
        "ask_next": [0.72, 0.27, 0.72], "bid_next": [0.69, 0.24, 0.69], "y": [1, 0, 0],
        "date": [sat.date(), sat.date(), mon.date()],
    })
    tr = cf.take_favourite(g, (5,), weekend=True, hours=(12, 18)).sort("ticker")
    assert tr["ticker"].to_list() == ["A", "B"]
    assert tr["side"].to_list() == ["yes", "no"]
    assert abs(tr["price"][1] - 0.75) < 1e-9 and tr["payoff"].to_list() == [1, 1]
