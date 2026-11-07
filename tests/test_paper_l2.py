import json
from pathlib import Path

from albert.backtest.l2 import (
    OnlineClassifier,
    Quoter,
    SimConfig,
    _mark_executions,
    events_from_rows,
)
from albert.paper_l2 import Guards, PaperMaker

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _rows(name: str = "ws_stream_wide.jsonl") -> list[dict]:
    """Trimmed recording of one market (snapshot + 1 hour of deltas/trades, 5c+ spreads)."""
    return [json.loads(line) for line in (FIX / name).read_text().splitlines()]


def test_online_classifier_labels_like_the_batch_matcher():
    for name in ("ws_stream_wide.jsonl", "ws_stream_trades.jsonl"):
        batch = events_from_rows(_rows(name))
        _mark_executions(batch)
        clf = OnlineClassifier(holdback_ms=500)
        out = []
        for ev in events_from_rows(_rows(name)):
            clf.push(ev)
            out.extend(clf.release(ev.recv_ns))
        out.extend(clf.release(2**62))
        assert [e.is_exec for e in out] == [e.is_exec for e in batch]
        assert clf.stats["trades"] > 0 and clf.stats["trades_matched"] == clf.stats["trades"]


def test_paper_maker_trades_recorded_frames_and_writes_ledger(tmp_path):
    pm = PaperMaker(tmp_path, None, strategy=Quoter(min_spread=0.0, lo=0.0, improve=False, min_requote_s=0),
                    cfg=SimConfig(latency_ms=50), pairs=[], guards=Guards())
    rows = _rows()
    for r in rows:
        pm.on_frame("kalshi", r["recv_ns"], r["frame"])
    end = pm.close()
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert end["orders"] > 0 and kinds[-1] == "end"
    assert end["fills"] == kinds.count("fill")
    assert end["classifier"]["trades"] == end["classifier"]["trades_matched"]
    assert end["diag"].get("exec_exceeds_model", 0) == 0


def test_loss_limit_halts_and_cancels(tmp_path):
    pm = PaperMaker(tmp_path, None, strategy=Quoter(min_spread=0.0), cfg=SimConfig(latency_ms=0),
                    pairs=[], guards=Guards(max_loss=0.0))
    for r in _rows()[:20]:
        pm.on_frame("kalshi", r["recv_ns"], r["frame"])
    pm.halt("test")
    assert pm.state.halted == "test"
    assert not any(pm.sim.live_orders(m) for m in pm.sim.by_market)
    pm.close()


def test_paper_fills_equal_replay_fills_on_the_same_frames(tmp_path):
    from albert.backtest.l2 import Simulator

    evs = events_from_rows(_rows())
    _mark_executions(evs)
    sim = Simulator(SimConfig(latency_ms=50))
    strat = Quoter(min_spread=0.0, lo=0.0, improve=False, min_requote_s=0)
    for ev in evs:
        sim.apply(ev)
        strat.on_event(sim, ev.market)

    pm = PaperMaker(tmp_path, None, strategy=Quoter(min_spread=0.0, lo=0.0, improve=False, min_requote_s=0),
                    cfg=SimConfig(latency_ms=50), pairs=[], guards=Guards())
    for r in _rows():
        pm.on_frame("kalshi", r["recv_ns"], r["frame"])
    pm.close()
    assert sim.fills and [(f.t_ns, f.side, f.price, f.qty) for f in pm.sim.fills] == \
        [(f.t_ns, f.side, f.price, f.qty) for f in sim.fills]
