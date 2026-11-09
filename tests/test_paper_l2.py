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


def test_market_result_on_recorded_markets():
    from albert.paper_l2 import market_result

    payload = json.loads((FIX / "historical_markets.json").read_text())
    markets = payload.get("response", payload)["markets"]
    got = {m["result"]: market_result(m) for m in markets}
    assert got == {"yes": 1.0, "no": 0.0}
    assert market_result({"result": ""}) is None and market_result({}) is None


def test_settled_market_leaves_gross_marks_at_result_and_is_not_quoted(tmp_path):
    pm = PaperMaker(tmp_path, None, strategy=Quoter(min_spread=0.0, lo=0.0, improve=False, min_requote_s=0),
                    cfg=SimConfig(latency_ms=50), pairs=[], guards=Guards())
    rows = _rows()
    for r in rows:
        pm.on_frame("kalshi", r["recv_ns"], r["frame"])
    pm.step(rows[-1]["recv_ns"] + 10**12)          # release the held-back tail (fills land here)
    assert pm.sim.fills
    market = pm.sim.fills[0].market
    assert pm.unsettled_held() == [market] and pm._gross() > 0
    yes = sum(f.qty for f in pm.sim.fills if f.side == "yes")
    no = sum(f.qty for f in pm.sim.fills if f.side == "no")
    cost = sum(f.qty * f.price + f.fee for f in pm.sim.fills)

    pm.settle(market, 1.0)
    assert pm._gross() == 0 and pm.unsettled_held() == []
    assert abs(pm.equity() - (yes - cost)) < 1e-9
    assert not pm.sim.live_orders(market)

    calls = []
    pm.strategy.on_event = lambda sim, m: calls.append(m)    # a resolved market is never quoted
    last = rows[-1]
    pm.on_frame("kalshi", last["recv_ns"] + 2 * 10**12, last["frame"])
    pm.step(last["recv_ns"] + 3 * 10**12)
    assert calls == []

    end = pm.close()
    assert end["results"] == {market: 1.0}
    settle_rows = [r for r in map(json.loads, (tmp_path / "ledger.jsonl").read_text().splitlines())
                   if r["kind"] == "settle"]
    assert len(settle_rows) == 1 and settle_rows[0]["payout"] == yes and settle_rows[0]["no"] == no


def test_settler_polls_held_markets_and_survives_errors(tmp_path):
    import asyncio

    pm = PaperMaker(tmp_path, None, strategy=Quoter(min_spread=0.0, lo=0.0, improve=False, min_requote_s=0),
                    cfg=SimConfig(latency_ms=50), pairs=[], guards=Guards(settle_every_s=0.01))
    rows = _rows()
    for r in rows:
        pm.on_frame("kalshi", r["recv_ns"], r["frame"])
    pm.step(rows[-1]["recv_ns"] + 10**12)
    market = pm.sim.fills[0].market
    answers = iter([RuntimeError("boom"), None, 0.0])

    def fetch(m):
        assert m == market
        a = next(answers, 0.0)
        if isinstance(a, Exception):
            raise a
        return a
    pm.fetch_result = fetch

    async def go():
        task = asyncio.create_task(pm.settler())
        for _ in range(200):
            await asyncio.sleep(0.01)
            if pm.results:
                break
        pm.rec.stop.set()
        await task
    asyncio.run(go())
    assert pm.results == {market: 0.0}
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert kinds.count("settle_error") == 1 and kinds.count("settle") == 1
    pm.close()
