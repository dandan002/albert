import json
from pathlib import Path

from albert.backtest.l2 import (Event, JoinBest, SimConfig, Simulator, _mark_executions,
                                events_from_rows, markouts)

FIX = Path(__file__).parent / "fixtures" / "kalshi"
T0 = 1_793_851_200_000_000_000
MS = 1_000_000


def _real_events() -> list[Event]:
    rows = [json.loads(line) for line in (FIX / "ws_stream_trades.jsonl").read_text().splitlines()]
    return events_from_rows(rows)


def test_every_recorded_trade_matches_one_execution_delta():
    evs = _real_events()
    trades, matched = _mark_executions(evs)
    assert trades > 0 and matched == trades
    assert sum(e.is_exec for e in evs) == trades


def test_replay_of_real_market_keeps_model_consistent_with_recorded_book():
    evs = _real_events()
    _mark_executions(evs)
    sim = Simulator(SimConfig(latency_ms=50))
    strat = JoinBest(min_spread=0.0)
    for ev in evs:
        sim.apply(ev)
        strat.on_event(sim, ev.market)
    m = sim.markets[evs[0].market]
    for side in ("yes", "no"):
        for lev in m.levels[side].values():
            assert all(c[0] > 0 for c in lev.q)
    assert sim.diag["exec_exceeds_model"] == 0
    assert all(f.qty > 0 for f in sim.fills)


def _snap(t, yes, no, market="M"):
    return Event(t, "snapshot", market, {"market_ticker": market,
                                         "yes_dollars_fp": [[f"{p:.4f}", f"{q:.2f}"] for p, q in yes],
                                         "no_dollars_fp": [[f"{p:.4f}", f"{q:.2f}"] for p, q in no]})


def _delta(t, side, price, d, exec_=False, market="M"):
    return Event(t, "delta", market, {"market_ticker": market, "side": side,
                                      "price_dollars": f"{price:.4f}", "delta_fp": f"{d:.2f}"},
                 is_exec=exec_)


def test_queue_fill_needs_the_queue_ahead_to_trade_first():
    sim = Simulator(SimConfig(latency_ms=100))
    sim.apply(_snap(T0, [(0.40, 10)], [(0.50, 5)]))
    sim.now = T0
    oid = sim.place("M", "yes", 0.40)
    sim.apply(_delta(T0 + 200 * MS, "yes", 0.40, -6, exec_=True))   # 6 of the 10 ahead trade
    assert not sim.fills
    sim.apply(_delta(T0 + 300 * MS, "yes", 0.40, -5, exec_=True))   # 4 ahead left, 1 reaches us
    assert len(sim.fills) == 1 and sim.fills[0].qty == 1 and sim.fills[0].how == "queue"
    assert sim.orders[oid].status == "filled"
    assert sim.position["M"] == 1


def test_cancels_ahead_only_count_in_optimistic_mode():
    for mode, filled in (("risk_averse", False), ("optimistic", True)):
        sim = Simulator(SimConfig(latency_ms=0, queue=mode))
        sim.apply(_snap(T0, [(0.40, 3)], [(0.50, 5)]))
        sim.place("M", "yes", 0.40)                                     # active at the next event
        sim.apply(_delta(T0 + 1, "yes", 0.40, 2))                       # 2 more join behind us
        sim.apply(_delta(T0 + 2, "yes", 0.40, -1))                      # a partial cancel, unknown order
        sim.apply(_delta(T0 + 3, "yes", 0.40, -3, exec_=True))          # 3 trade
        assert bool(sim.fills) is filled, mode


def test_post_only_reject_trade_through_and_cancel_latency():
    sim = Simulator(SimConfig(latency_ms=100))
    sim.apply(_snap(T0, [(0.40, 10)], [(0.50, 5)]))   # yes ask = 0.50
    sim.now = T0
    bad = sim.place("M", "yes", 0.50)                 # would cross -> rejected
    good = sim.place("M", "yes", 0.45)                # improves the bid, alone at 0.45
    sim.apply(_delta(T0 + 150 * MS, "yes", 0.40, 0.0001))
    assert sim.orders[bad].status == "rejected" and sim.orders[good].status == "active"
    sim.now = T0 + 150 * MS
    sim.cancel(good)                                   # effective at +250 ms
    sim.apply(_delta(T0 + 200 * MS, "yes", 0.40, -2, exec_=True))   # sells through 0.45 first
    assert sim.fills and sim.fills[0].how == "through" and sim.fills[0].price == 0.45
    rows = markouts(sim)
    assert rows[0]["side"] == "yes" and "mo_1s" in rows[0]
