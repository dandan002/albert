"""Tests for albert.research.deadline_hazard on trimmed copies of recorded Kalshi responses."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from albert.research import deadline_hazard as D
from albert.venues import kalshi as k

FIX = Path(__file__).parent / "fixtures" / "deadline_hazard"


def _markets() -> pl.DataFrame:
    rec = json.loads((FIX / "markets_settled_KXCANUSDEAL.json").read_text())
    info = {"KXCANUSDEAL": {"category": "Politics", "title": "Canada US trade deal", "fee_type": "quadratic",
                            "fee_multiplier": 1.0}}
    rows = k.market_rows(rec["response"], info, "fixture")
    return pl.DataFrame(rows, schema=k.MARKET_SCHEMA)


def test_parse_recorded_strip_and_pairs():
    m = D.parse_markets(_markets())
    assert m.height == 2
    assert set(m["kind"]) == {"event"}
    # "before Nov 1, 2026" ends at Nov 1 00:00 ET = 04:00 UTC
    assert m.filter(pl.col("ticker") == "KXCANUSDEAL-26-26NOV01")["T"][0] == datetime(2026, 11, 1, 4, 0, tzinfo=UTC)
    assert m["skey"].n_unique() == 1          # same question, dates removed
    assert set(m["y"]) == {0}
    pairs = D.strip_pairs(m.with_columns(pl.col("settlement_ts").fill_null(pl.col("close_time"))))
    assert pairs.height == 1
    assert pairs["ticker"][0] == "KXCANUSDEAL-26-26OCT01" and pairs["ticker_2"][0] == "KXCANUSDEAL-26-26NOV01"


def test_quote_at_is_strictly_before_decision():
    t0 = datetime(2026, 9, 1, 12, tzinfo=UTC)
    q = pl.DataFrame({"ticker": ["A", "A"], "q_ts": [t0 - timedelta(hours=1), t0],
                      "bid": [0.10, 0.50], "ask": [0.12, 0.52]})
    df = pl.DataFrame({"ticker": ["A"], "t": [t0]})
    out = D.quote_at(df, q)
    assert out["bid"][0] == pytest.approx(0.10)      # the candle ending exactly at t is not used
    stale = D.quote_at(pl.DataFrame({"ticker": ["A"], "t": [t0 + timedelta(days=5)]}), q, max_age_h=72)
    assert stale["bid"][0] is None


def test_taker_fee_rounds_up_per_order():
    # 5 NO contracts at 0.96: 0.07 * 5 * 0.96 * 0.04 = 0.01344 -> 0.02 per order
    assert D.taker_fee_pc([0.96], [5], [1.0])[0] == pytest.approx(0.02 / 5)
    assert D.order_n([0.96])[0] == 5
    assert D.order_n([0.30])[0] == 16


def test_capped_ledger_respects_caps():
    t0 = datetime(2026, 5, 1, tzinfo=UTC)
    tr = pl.DataFrame({
        "ticker": ["A", "B", "C"], "event_ticker": ["E1", "E1", "E2"], "t": [t0, t0 + timedelta(hours=1), t0 + timedelta(hours=2)],
        "px": [0.60, 0.60, 0.80], "y": [0, 1, 0], "fee_multiplier": [1.0, 1.0, 1.0], "fee_type": ["quadratic"] * 3,
        "settlement_ts": [t0 + timedelta(days=10)] * 3, "split": ["dev"] * 3})
    led = D.capped_ledger(tr, "s", "v", event_cap=5.0, total_cap=8.0)
    assert led["cost_usd"].sum() <= 8.0 + 1e-9
    assert led.filter(pl.col("event_ticker") == "E1")["cost_usd"].sum() <= 5.0 + 1e-9
    a = led.filter(pl.col("ticker") == "A")
    assert a["payoff_usd"][0] == a["n"][0]            # NO on a market that settled NO pays $1 each


def test_book_levels_and_exec_arb_on_recorded_book():
    rec = json.loads((FIX / "orderbook_KXWHPRESSBANEND-26-27MAR01.json").read_text())
    bids, asks = D.book_levels(rec["response"])
    assert bids[0] == (0.21, 7.0)                       # best YES bid
    assert asks[0] == (pytest.approx(0.28), 1.32)      # best NO bid 0.72 -> YES ask 0.28
    assert all(bids[i][0] > bids[i + 1][0] for i in range(len(bids) - 1))
    assert all(asks[i][0] < asks[i + 1][0] for i in range(len(asks) - 1))
    # the same book as both legs never crosses itself
    assert D.exec_arb(bids, asks)["contracts"] == 0
    # a far leg offered below the near bid: walks 0.21x7 and 0.20x250 against 0.10x3 and 0.12x10
    ex = D.exec_arb(bids, [(0.10, 3.0), (0.12, 10.0)])
    assert ex["contracts"] == pytest.approx(13.0)
    assert ex["net_usd"] == pytest.approx(round(3 * 0.092087 + 4 * 0.070995 + 6 * 0.061408, 2), abs=0.011)
