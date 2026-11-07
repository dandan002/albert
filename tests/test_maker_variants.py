import json
from pathlib import Path

from albert.backtest.l2 import SimConfig, Simulator, _mark_executions, events_from_rows
from albert.research.maker_variants import Variant, clustered, round_trips, score

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _events():
    rows = [json.loads(x) for x in (FIX / "ws_stream_wide.jsonl").read_text().splitlines()]
    evs = events_from_rows(rows)
    _mark_executions(evs)
    return evs


def _replay(strat):
    sim = Simulator(SimConfig(latency_ms=50))
    for ev in _events():
        sim.apply(ev)
        strat.observe(sim, ev)
        strat.on_event(sim, ev.market)
    return sim


def test_baseline_variant_equals_quoter():
    from albert.backtest.l2 import Quoter

    kw = {"min_spread": 0.0, "lo": 0.0, "improve": True, "min_requote_s": 0}
    a = _replay(Variant(**kw))
    b = Simulator(SimConfig(latency_ms=50))
    q = Quoter(**kw)
    for ev in _events():
        b.apply(ev)
        q.on_event(b, ev.market)
    assert len(a.orders) == len(b.orders) > 0
    assert [(f.side, f.price) for f in a.fills] == [(f.side, f.price) for f in b.fills]


def test_levers_change_behaviour_and_never_cross():
    kw = {"min_spread": 0.0, "lo": 0.0, "improve": True, "min_requote_s": 0}
    base = _replay(Variant(**kw))
    tox = Variant(tox_window_s=600, tox_min=1, tox_frac=0.5, **kw)
    sim = _replay(tox)
    assert sum(tox.pulls.values()) > 0 and len(sim.orders) < len(base.orders)
    t2 = _replay(Variant(ticks=3, **kw))
    assert t2.diag.get("post_only_rejects", 0) == 0
    assert {o.price for o in t2.orders.values()} != {o.price for o in base.orders.values()}


def test_round_trips_and_scoring():
    rows = [{"t_ns": 1, "market": "M-A", "side": "yes", "price": 0.40, "qty": 1.0, "fee": 0.0},
            {"t_ns": 2, "market": "M-A", "side": "no", "price": 0.55, "qty": 1.0, "fee": 0.0},
            {"t_ns": 3, "market": "M-A", "side": "no", "price": 0.55, "qty": 1.0, "fee": 0.0}]
    rt = round_trips(rows)
    assert rt["paired"] == 1.0 and abs(rt["paired_pnl"] - 0.05) < 1e-9
    m, _t, k = clustered([{**r, "event": "E", "x": 0.1} for r in rows], "x")
    assert abs(m - 0.1) < 1e-9 and k == 1
    sim = _replay(Variant(min_spread=0.0, lo=0.0, improve=False, min_requote_s=0))
    summ, out = score(sim, {})
    assert summ["fills"] == len(sim.fills) > 0 and all("mx_300s" in r for r in out)


def test_momentum_tilt_drops_a_side():
    kw = {"min_spread": 0.0, "lo": 0.0, "improve": True, "min_requote_s": 0}
    base = _replay(Variant(**kw))
    mom = _replay(Variant(mom_window_s=6 * 3600, mom_move=0.01, **kw))
    assert len(mom.orders) != len(base.orders)
