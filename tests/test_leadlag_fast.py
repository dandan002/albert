"""Sub-minute lead-lag. Parsers run on recorded frames (Polymarket ws_market.jsonl, Kalshi
ws_stream_wide.jsonl); the event logic runs on top-of-book series built from them."""

import json
from pathlib import Path

import pytest

from albert.backtest.l2 import _mark_executions, events_from_rows
from albert.research.leadlag_fast import (Book, Data, _pm_items, clustered, event_study, kalshi_books,
                                          path_stats, pm_books, pm_moves, taker_fee)

FIX = Path(__file__).parent / "fixtures"


def _pm_rows() -> list[dict]:
    lines = [x for x in (FIX / "polymarket" / "ws_market.jsonl").read_text().splitlines() if x.strip()]
    return [{"recv_ns": 1_000_000_000 * (i + 1), "frame": x} for i, x in enumerate(lines)]


def test_pm_items_carry_server_timestamps_and_complement():
    rows = _pm_rows()
    items = [it for r in rows for it in _pm_items(r)]
    assert items and all(ts is None or ts > 1.7e12 for _, ts, _, _ in items)
    asset, _, bid, ask = next(it for it in items if it[2] is not None and it[3] is not None)
    pairs = {"K-A": {"token_id": asset, "complement": False, "pm_slug": "g"},
             "K-B": {"token_id": asset, "complement": True, "pm_slug": "g"}}
    paths: list = []
    books = pm_books(iter(rows), pairs, paths)
    a, b = books["K-A"], books["K-B"]
    assert a.t and len(a.t) == len(b.t) and paths
    for i in range(len(a.t)):
        if a.bid[i] is not None and a.ask[i] is not None:
            assert b.bid[i] == pytest.approx(1 - a.ask[i]) and b.ask[i] == pytest.approx(1 - a.bid[i])


def test_kalshi_books_from_recorded_stream_have_sizes_and_exchange_times():
    rows = [json.loads(x) for x in (FIX / "kalshi" / "ws_stream_wide.jsonl").read_text().splitlines()]
    evs = events_from_rows(rows)
    _mark_executions(evs)
    paths: list = []
    books, trades = kalshi_books(iter(evs), {evs[0].market}, paths)
    b = books[evs[0].market]
    assert len(b.t) > 5 and trades and paths
    assert all(x is None or x > 0 for x in b.bsz) and any(ts is not None for ts in b.ts)
    assert all(bid < ask for bid, ask in zip(b.bid, b.ask) if bid is not None and ask is not None)
    st = path_stats(paths)
    assert st["n"] == len(paths) and st["median"] is not None


def _book(points):
    b = Book()
    for t_s, bid, ask, size in points:
        b.add(int(t_s * 1e9), int(t_s * 1e3), bid, ask, size, size)
    return b


def test_event_study_measures_lag_staleness_and_taker_pnl():
    # Polymarket jumps 0.50 -> 0.53 at t=100 s; Kalshi follows at t=102 s.
    pm = _book([(0, 0.49, 0.51, 10), (100, 0.52, 0.54, 10)])
    k = _book([(0, 0.48, 0.50, 5), (102, 0.51, 0.53, 5), (300, 0.51, 0.53, 5)])
    d = Data(pairs={"K": {"pm_slug": "g", "complement": False}}, k={"K": k}, p={"K": pm}, trades=[],
             k_paths=[], p_paths=[], start_ns={"K": None}, end_ns=int(300e9))
    moves = pm_moves(pm, 0.02)
    assert len(moves) == 1 and moves[0][2] == 1.0
    f = event_study(d, 0.02)
    r = f.row(0, named=True)
    assert r["lag_s"] == pytest.approx(2.0) and r["stale_50ms"] and r["stale_300ms"]
    assert r["edge_50ms"] == pytest.approx(0.03)            # fair 0.53 vs ask 0.50
    # buy at 0.50 + fee, sell at the bid 0.51 after 10 s, paying the fee again
    assert r["pnl_50ms_10s"] == pytest.approx(0.51 - 0.50 - taker_fee(0.50) - taker_fee(0.51))
    assert r["resp_5s"] == pytest.approx(0.03)


def test_no_lookahead_truncating_future_keeps_decision_inputs():
    pm = _book([(0, 0.49, 0.51, 10), (100, 0.52, 0.54, 10)])
    k_full = _book([(0, 0.48, 0.50, 5), (100.2, 0.51, 0.53, 5), (300, 0.51, 0.53, 5)])
    k_cut = _book([(0, 0.48, 0.50, 5)])
    rows = []
    for k in (k_full, k_cut):
        d = Data(pairs={"K": {"pm_slug": "g"}}, k={"K": k}, p={"K": pm}, trades=[], k_paths=[], p_paths=[],
                 start_ns={}, end_ns=int(300e9))
        rows.append(event_study(d, 0.02).row(0, named=True))
    assert rows[0]["edge_50ms"] == rows[1]["edge_50ms"] and rows[0]["stale_50ms"] == rows[1]["stale_50ms"]
    assert not rows[0]["stale_300ms"] and rows[1]["stale_300ms"]   # Kalshi moved at +200 ms


def test_fee_and_clustered_mean():
    assert taker_fee(0.5) == pytest.approx(0.0175) and taker_fee(0.01) == pytest.approx(0.0007)
    mean, t, n, k = clustered([1.0, 1.0, -1.0, 3.0], ["a", "a", "b", "c"])
    assert mean == pytest.approx(1.0) and n == 4 and k == 3 and t is not None
