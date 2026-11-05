import json
from pathlib import Path

from albert.backtest.l2 import SimConfig, _mark_executions, events_from_rows
from albert.research.momentum_l2 import (SignalQuoter, clustered, collect, grid_samples, outcomes,
                                         run_overlay, taker_fee, trigger_samples)

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _events(name: str):
    rows = [json.loads(line) for line in (FIX / name).read_text().splitlines()]
    evs = events_from_rows(rows)
    _mark_executions(evs)
    return evs


def test_collect_records_quotes_and_signed_trades():
    tapes, end = collect(_events("ws_stream_trades.jsonl"))
    tp = next(t for t in tapes.values() if t.tt)
    assert tp.qt and len(tp.qt) == len(tp.bid) == len(tp.ofi)
    assert all(b is None or a is None or b < a for b, a in zip(tp.bid, tp.ask))
    assert tp.tsgn and all(x != 0 for x in tp.tsgn)
    assert end >= tp.qt[-1]


def test_features_use_only_the_past():
    """Changing everything after t must not change the features at t."""
    evs = _events("ws_stream_wide.jsonl")
    tapes, end = collect(evs)
    rows = grid_samples(tapes, end, every_s=60)
    assert rows
    cut = rows[len(rows) // 2]["t_ns"]
    tapes2, _ = collect([e for e in evs if e.recv_ns <= cut])
    feats = [k for k in rows[0] if k.startswith(("mom_", "flow", "imb", "ofi_"))]
    short = {r["t_ns"]: r for r in grid_samples(tapes2, end, every_s=60) if r["t_ns"] <= cut}
    for r in rows:
        if r["t_ns"] <= cut:
            assert all(abs(r[f] - short[r["t_ns"]][f]) < 1e-12 for f in feats), r["t_ns"]


def test_outcomes_sign_and_fees():
    tapes, end = collect(_events("ws_stream_wide.jsonl"))
    tp = next(iter(tapes.values()))
    t = tp.qt[len(tp.qt) // 3]
    up, down = outcomes(tp, t, 1.0, end), outcomes(tp, t, -1.0, end)
    for h in (10, 60):
        if up[f"mid_{h}s"] is not None:
            assert abs(up[f"mid_{h}s"] + down[f"mid_{h}s"]) < 1e-12
            assert up[f"exit_{h}s"] <= up[f"mid_{h}s"] + 1e-12   # crossing costs the spread
    assert taker_fee(0.5) == 0.0175 and taker_fee(0.05) == 0.0034


def test_triggers_and_clustered_t():
    tapes, end = collect(_events("ws_stream_wide.jsonl"))
    rows = trigger_samples(tapes, end, move=0.01, run_n=2, run_s=600, run_min=1, big=10)
    assert rows and {r["kind"] for r in rows} <= {"move", "run", "big"}
    mean, t, n, k = clustered([1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "b"])
    assert mean == 2.5 and n == 4 and k == 2 and t is not None


def test_signal_quoter_withholds_the_side_at_risk():
    evs = _events("ws_stream_wide.jsonl")
    cfg = SimConfig(latency_ms=50)
    q = SignalQuoter(signal="flow", window_s=600, flow_min=1.0, hold_s=1e6, min_spread=0.0, lo=0.0, hi=1.0,
                     min_requote_s=0)
    sim = run_overlay(evs, q, cfg)
    assert q.pressure and sim.orders
    for mk, (d, _) in q.pressure.items():
        withheld = "no" if d > 0 else "yes"
        assert not [o for o in sim.live_orders(mk) if o.side == withheld]
