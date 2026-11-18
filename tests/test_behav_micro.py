"""Tests for albert.research.behav_micro on recorded payloads.

Fixtures (copies of real recordings, not invented):
* ``kalshi/behav_micro_trades_KXJOHNSONMENTION-26OCT29-HEAL.json``: the complete public
  ``/markets/trades`` response for that market (50 prints), cached by the flb_flow fetch.
* ``kalshi/behav_micro_stream_KXRAIN-26NOV12-CHI.jsonl.zst``: every recorded WebSocket line of that
  market from the 2026-11-12 14:00 UTC hourly file (``k0`` connection), from its snapshot to 5 s
  after a 6c YES sweep.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from albert.research import behav_micro as b
from albert.venues import kalshi as k

FIX = Path(__file__).parent / "fixtures" / "kalshi"
TRADES = FIX / "behav_micro_trades_KXJOHNSONMENTION-26OCT29-HEAL.json"
STREAM = FIX / "behav_micro_stream_KXRAIN-26NOV12-CHI.jsonl.zst"


def _meta(y: int, maker_fee: bool = False) -> pl.DataFrame:
    # market facts of KXJOHNSONMENTION-26OCT29-HEAL (settled NO; closed 14:23:26 UTC)
    return pl.DataFrame({
        "ticker": ["KXJOHNSONMENTION-26OCT29-HEAL"], "event_ticker": ["KXJOHNSONMENTION-26OCT29"],
        "group": ["Mentions"], "stratum": ["Mentions"],
        "close_time": [datetime(2026, 10, 29, 14, 23, 26, tzinfo=UTC)],
        "settlement_ts": [datetime(2026, 10, 29, 14, 53, 28, tzinfo=UTC)], "y": [y], "mult": [1.0],
        "maker_fee": [maker_fee], "sched": [None], "exp": [datetime(2026, 11, 12, 15, tzinfo=UTC)],
    }, schema_overrides={"y": pl.Int8, "sched": pl.Datetime("us", "UTC")})


@pytest.fixture
def orders() -> pl.DataFrame:
    tr = pl.DataFrame(k.trade_rows(json.loads(TRADES.read_text())))
    return b.orders_from_trades(tr, _meta(0))


def test_fee_rounding() -> None:
    assert b.fee_cents(1, 0.97) == 0.01            # 0.2c model fee rounds up to a cent per order
    assert b.fee_cents(4, 0.50) == 0.07            # 7.0c exactly
    assert b.fee_cents(1, 0.50, rate=b.KALSHI_MAKER) == 0.01
    assert b.fee_cents(10, 0.50, mult=0.0) == 0.0


def test_orders_group_prints_into_taker_orders(orders: pl.DataFrame) -> None:
    # 50 prints -> 20 taker orders (same ticker, timestamp and taker side)
    assert orders.height == 20
    sweep = orders.filter(pl.col("t") == datetime(2026, 10, 29, 6, 13, 53, 629111, tzinfo=UTC)).row(0, named=True)
    assert sweep["d"] == 1 and sweep["fills"] == 6
    assert sweep["q1"] == pytest.approx(0.21) and sweep["qw"] == pytest.approx(0.31)
    assert sweep["contracts"] == pytest.approx(41.42)
    assert sweep["cap10"] == 0.0                   # nothing printed strictly beyond 0.31
    # a NO-taker sweep: YES prints 0.12 -> 0.01 are NO prices 0.88 -> 0.99
    no = orders.filter(pl.col("t") == datetime(2026, 10, 29, 12, 32, 17, 49269, tzinfo=UTC)).row(0, named=True)
    assert no["d"] == -1 and no["q1"] == pytest.approx(0.88) and no["qw"] == pytest.approx(0.99)
    assert no["won"] == 1                          # the market settled NO, so the NO taker won


def test_stub_fills_only_strictly_through(orders: pl.DataFrame) -> None:
    f = b.stub_fills(orders, 10, 2.0).sort("t")
    assert f.height == 4
    # 06:13:53 swept exactly to the stub (0.21 + 0.10 = 0.31) without going beyond: no fill
    assert datetime(2026, 10, 29, 6, 13, 53, 629111, tzinfo=UTC) not in f["t"].to_list()
    yes = f.filter(pl.col("d") == 1)
    assert yes["L"].to_list() == pytest.approx([0.40, 0.40])
    assert yes["pnl"].to_list() == pytest.approx([0.40, 0.40])    # sold YES at 40c, it settled NO
    no = f.filter(pl.col("d") == -1)
    assert no["L"].to_list() == pytest.approx([0.98, 0.97])
    assert no["pnl"].to_list() == pytest.approx([-0.02, -0.03])   # sold NO at 98/97c, NO won
    assert not f["burst"].any()


def test_stub_maker_fee_and_bursts(orders: pl.DataFrame) -> None:
    fee = b.stub_fills(b.orders_from_trades(pl.DataFrame(k.trade_rows(json.loads(TRADES.read_text()))),
                                            _meta(0, maker_fee=True)), 10, 2.0).sort("t")
    assert (fee["gross"] - fee["pnl"]).to_list() == pytest.approx([0.01] * 4)
    # with a 60 s re-peg the 11:28:15 order and the 11:36:49 sweep are separate runs (8.5 min
    # apart), but within an hour-long latency the 11:36 sweep is measured from the 10:41 best
    slow = b.stub_fills(orders, 10, 3600.0)
    assert slow.filter(pl.col("d") == 1)["burst"].any()


def test_cluster_stats() -> None:
    df = pl.DataFrame({"x": [0.01, 0.03, 0.02, 0.04], "event_ticker": ["a", "a", "b", "b"],
                       "etd": [1, 2, 1, 2]})
    r = b.cluster_stats(df, "x", "event_ticker", "etd")
    assert r["n"] == 4 and r["mean_c"] == pytest.approx(2.5) and r["clusters"] == 2


def test_l2_stub_replay_on_recorded_stream() -> None:
    fills, expo = b.l2_stub_file(STREAM, (3, 5, 10), 1000)
    f = pl.DataFrame(fills)
    # one YES taker order walked the asks 0.40 -> 0.46 at 1794492361006 ms (442 contracts)
    assert set(f["D"].to_list()) == {3, 5}
    r = f.filter(pl.col("D") == 5).row(0, named=True)
    assert r["ts_ms"] == 1794492361006 and r["d"] == 1
    assert r["q1"] == pytest.approx(0.40) and r["qw"] == pytest.approx(0.46)
    assert r["pre_bid"] == pytest.approx(0.39) and r["pre_ask"] == pytest.approx(0.40)
    assert r["book_fill"] and r["print_fill"]
    assert r["L_book"] == pytest.approx(0.45)      # best ask 1 s earlier (0.40) + 5c
    assert r["contracts"] == pytest.approx(442.15)
    # exposure: both stubs live for the whole replayed span, collateral below one dollar each
    e = next(x for x in expo if x["D"] == 10)
    assert e["sec_ask"] == pytest.approx(e["sec_bid"]) and e["sec_ask"] > 300
    assert 0 < e["col_ask"] / e["sec_ask"] < 1 and 0 < e["col_bid"] / e["sec_bid"] < 1


def test_stub_ledger_caps(orders: pl.DataFrame) -> None:
    f = b.stub_fills(orders, 10, 2.0)
    led = b.stub_ledger(f, "stub", "D10", "test", per_event=1.0, open_cap=50.0)
    # costs 0.60, 0.60, 0.02, 0.03 (1 - L); the event cap of $1 admits the first 0.60 and both NO
    # legs, and rejects the second 0.60 (all in one event, still open)
    assert led.height == 3
    assert led["cost_usd"].sum() == pytest.approx(0.65)
    assert set(led["side"].to_list()) == {"no", "yes"}
    assert (led["payoff_usd"] - led["cost_usd"]).sum() == pytest.approx(0.40 - 0.02 - 0.03)


def test_refill_after_small_sweep(orders: pl.DataFrame) -> None:
    # triggers: one order spanning >= 10c with <= $50 notional (06:13:53 0.21->0.31 $10.44,
    # 10:19:41 0.21->0.31 $1.89, 11:36:49 0.30->0.49 $4.31, 12:27:39 0.30->0.50); the two NO
    # sweeps are > $50 and do not trigger. Each posts at qw - 1c and fills on the first later
    # same-side order within 60 min that prints strictly beyond it.
    f = b.refill_fills(orders).sort("t")
    assert f["t"].to_list() == [datetime(2026, 10, 29, 6, 15, 38, 151559, tzinfo=UTC),
                                datetime(2026, 10, 29, 10, 41, 12, 130246, tzinfo=UTC),
                                datetime(2026, 10, 29, 12, 27, 39, 780044, tzinfo=UTC)]
    assert f["P"].to_list() == pytest.approx([0.30, 0.30, 0.48])
    assert f["trig_usd"].to_list() == pytest.approx([10.4422, 1.8898, 4.3142], abs=1e-4)
    assert f["pnl"].to_list() == pytest.approx([0.30, 0.30, 0.48])   # sold YES; settled NO
    # the 12:27:39 trigger (P = 0.49) never fills: the only later YES order printed at 0.01
    assert b.refill_fills(orders, max_usd=1.0).height == 0
