import json
from pathlib import Path

import pytest

from albert.backtest.l2 import Event, SimConfig, Simulator, _mark_executions, events_from_rows
from albert.research.intraminute_maker import VacatedLevel, clustered, fill_rows, lifetime_stats

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _events(name: str = "ws_stream_wide.jsonl"):
    rows = [json.loads(line) for line in (FIX / name).read_text().splitlines()]
    evs = events_from_rows(rows)
    _mark_executions(evs)
    return evs


def _run(evs, strat, latency_ms=50.0):
    sim = Simulator(SimConfig(latency_ms=latency_ms, writes_per_s=10.0))
    placed = []
    for ev in evs:
        sim.apply(ev)
        before = len(sim.orders)
        q = sim.markets[ev.market].quote() if ev.market in sim.synced else None
        strat.on_event(sim, ev)
        for oid in list(sim.orders)[before:]:
            placed.append((ev.recv_ns, oid, q))
    return sim, placed


def test_entries_are_inside_the_spread_after_the_vacated_level():
    strat = VacatedLevel(cause="any", offset=0, ttl_s=5, min_spread_after=0.0, lo=0.0, hi=1.0,
                         cooldown_s=0)
    sim, placed = _run(_events(), strat)
    entries = [(t, oid, q) for t, oid, q in placed if strat.role[oid] == "entry"]
    assert strat.triggers > 0 and entries
    for _, oid, q in entries:
        o = sim.orders[oid]
        if o.side == "yes":
            assert q["yes_bid"] < o.price < q["yes_ask"]
        else:
            assert 1 - q["yes_ask"] < o.price < 1 - q["yes_bid"]
    # every entry is cancelled or filled by its time-to-live
    for _, oid, _ in entries:
        o = sim.orders[oid]
        assert o.status in ("cancelled", "filled", "rejected", "cancelling") or sim.now - o.placed_ns < 5e9 + 1e8


def test_decisions_do_not_depend_on_future_events():
    """Replaying only the events up to each decision reproduces the decisions made so far
    (the same labelled events are reused, so this isolates the strategy from the labeller)."""
    evs = _events()
    kw = {"cause": "any", "offset": 0, "ttl_s": 5, "min_spread_after": 0.0, "lo": 0.0, "hi": 1.0, "cooldown_s": 0}
    _, full = _run(evs, VacatedLevel(**kw))
    assert full
    idx = {e.recv_ns: i for i, e in enumerate(evs)}
    for t, _, _ in full:
        j = idx[t]
        _, part = _run(evs[:j + 1], VacatedLevel(**kw))
        assert [(a, b) for a, b, _ in part] == [(a, b) for a, b, _ in full if a <= t]


def test_scalp_exit_does_not_cross_and_pairs_round_trips():
    strat = VacatedLevel(cause="any", offset=0, ttl_s=30, min_spread_after=0.0, lo=0.0, hi=1.0,
                         cooldown_s=0, scalp_k=1, scalp_ttl_s=60)
    sim, placed = _run(_events(), strat)
    for t, oid, q in placed:
        if strat.role[oid] != "scalp":
            continue
        o = sim.orders[oid]
        bid_same = q["yes_bid"] if o.side == "yes" else 1 - q["yes_ask"]
        ask_same = q["yes_ask"] if o.side == "yes" else 1 - q["yes_bid"]
        assert o.price < ask_same + 1e-9 or bid_same is None
    rows = fill_rows(sim, strat, {})
    assert all(r["role"] in ("entry", "scalp") for r in rows)
    for r in rows:
        if r["paired"]:
            assert r["rt_pnl"] is not None and r["rt_pnl"] <= 0.03 + 1e-9   # at most k=1 tick + spread


def test_lifetime_and_clustered_stats():
    strat = VacatedLevel(cause="any", offset=0, ttl_s=5, min_spread_after=0.0, lo=0.0, hi=1.0,
                         cooldown_s=0)
    evs = _events()
    # The fixture's triggers fall in its final sweep; replay its last recorded trade print again
    # 2 s later (a trade print does not change the book) so the orders become active and are seen.
    last_trade = next(e for e in reversed(evs) if e.kind == "trade")
    evs.append(Event(evs[-1].recv_ns + 2 * 10**9, "trade", last_trade.market, last_trade.msg))
    sim, _ = _run(evs, strat)
    st = lifetime_stats(strat.tracker, sim)
    assert st["orders_tracked"] > 0 and st["alone_s_median"] is not None and st["alone_s_median"] >= 0
    rows = [{"event": "a", "qty": 1.0, "x": 0.02}, {"event": "a", "qty": 1.0, "x": 0.04},
            {"event": "b", "qty": 2.0, "x": -0.01}]
    mean, _t, k = clustered(rows, "x")
    assert k == 2 and mean == pytest.approx((0.06 - 0.02) / 4)
