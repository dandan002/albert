import json
from pathlib import Path

from albert.backtest.l2 import (
    Event,
    JoinBest,
    Quoter,
    SimConfig,
    Simulator,
    _mark_executions,
    events_from_rows,
    markouts,
)

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


def test_quoter_improves_one_tick_and_keeps_its_place_when_others_join():
    sim = Simulator(SimConfig(latency_ms=0))
    q = Quoter(min_spread=0.05, improve=True)
    sim.apply(_snap(T0, [(0.40, 10)], [(0.50, 5)]))        # yes 0.40 / 0.50, spread 10c
    q.on_event(sim, "M")
    live = {o.side: o.price for o in sim.live_orders("M")}
    assert live == {"yes": 0.41, "no": 0.51}
    sim.apply(_delta(T0 + MS, "yes", 0.41, 3))               # someone joins behind our yes bid
    q.on_event(sim, "M")
    assert sim.orders[1].status == "active" and sim.orders[1].price == 0.41
    sim.apply(_delta(T0 + 2 * MS, "yes", 0.41, -1, exec_=True))   # front of the level: ours
    assert sim.fills and sim.fills[0].oid == 1 and sim.fills[0].how == "queue"


def test_quoter_respects_spread_band_and_cooldown():
    sim = Simulator(SimConfig(latency_ms=0))
    q = Quoter(min_spread=0.05, improve=True, cooldown_s=60, min_requote_s=0)
    sim.apply(_snap(T0, [(0.40, 10)], [(0.57, 5)]))        # spread 3c: below the band
    q.on_event(sim, "M")
    assert not sim.live_orders("M")
    sim.apply(_delta(T0 + MS, "no", 0.57, -5))                # no-side best now 0.50: spread 10c
    sim.apply(_snap(T0 + MS, [(0.40, 10)], [(0.50, 5)]))
    q.on_event(sim, "M")
    assert len(sim.live_orders("M")) == 2
    sim.apply(_delta(T0 + 2 * MS, "yes", 0.40, -2, exec_=True))   # sells through our 0.41
    q.on_event(sim, "M")
    assert sim.fills and all(o.status != "active" or o.side == "no" for o in sim.orders.values())
    assert not [o for o in sim.live_orders("M")]                     # both pulled for the cooldown


def test_one_sided_cooldown_keeps_the_pairing_side_quoting():
    sim = Simulator(SimConfig(latency_ms=0))
    q = Quoter(min_spread=0.05, improve=True, cooldown_s=60, min_requote_s=0, cooldown_side=True)
    sim.apply(_snap(T0, [(0.40, 10)], [(0.50, 5)]))
    q.on_event(sim, "M")
    assert len(sim.live_orders("M")) == 2
    sim.apply(_delta(T0 + MS, "yes", 0.40, -2, exec_=True))     # sells through our 0.41 YES bid
    q.on_event(sim, "M")
    assert [f.side for f in sim.fills] == ["yes"]
    live = sim.live_orders("M")
    assert [o.side for o in live] == ["no"]                        # NO (pairs the YES) stays
    sim.apply(_snap(T0 + 30 * 10**9, [(0.40, 10)], [(0.50, 5)]))  # 30 s later: YES still paused
    q.on_event(sim, "M")
    assert not [o for o in sim.live_orders("M") if o.side == "yes"]
    sim.apply(_snap(T0 + 61 * 10**9, [(0.40, 10)], [(0.50, 5)]))  # after 60 s: YES requoted
    q.on_event(sim, "M")
    assert [o for o in sim.live_orders("M") if o.side == "yes"]


def test_write_budget_throttles_places():
    sim = Simulator(SimConfig(latency_ms=0, writes_per_s=2))
    sim.apply(_snap(T0, [(0.40, 10)], [(0.50, 5)]))
    ids = [sim.place("M", "yes", 0.40) for _ in range(3)]
    assert ids[2] is None and sim.diag["throttled_places"] == 1
    sim.now = T0 + 10**9                                       # one second refills the bucket
    assert sim.place("M", "yes", 0.40) is not None


def test_pm_frames_become_ext_events_in_kalshi_yes_terms():
    from albert.backtest.l2 import _pm_tops

    lines = (FIX.parent / "polymarket" / "ws_market.jsonl").read_text().splitlines()
    tops = [t for line in lines if line.strip() for t in _pm_tops({"frame": line})]
    assert tops and all(b is None or a is None or b <= a for _, b, a in tops)


def test_lead_lag_detects_kalshi_following_pm():
    from albert.research.leadlag import lead_lag_frame

    sim = Simulator(SimConfig(latency_ms=0, writes_per_s=None))
    sim.apply(_snap(T0, [(0.40, 10)], [(0.50, 5)]))                     # kalshi mid 0.45
    for i, mid in enumerate([0.45, 0.45, 0.50]):                        # pm jumps 5c at +2s
        sim.apply(Event(T0 + i * 10**9, "ext", "M", {"mid": mid}))
    sim.apply(_delta(T0 + 4 * 10**9, "no", 0.50, -5))                   # kalshi ask lifts...
    sim.apply(_delta(T0 + 4 * 10**9 + 1, "no", 0.45, 5))                # ...to 0.55: mid 0.475
    f = lead_lag_frame(sim, {"M": {"pm_slug": "g"}}, move=0.02, window_s=5, gap_s=30,
                       end_ns=T0 + 10 * 10**9)
    pm = f.filter(f["leader"] == "pm")
    assert pm.height == 1 and pm["dir"][0] == 1.0
    assert abs(pm["other_5s"][0] - 0.025) < 1e-9 and pm["other_1s"][0] == 0.0


def test_quoter_pm_band_and_pull():
    sim = Simulator(SimConfig(latency_ms=0, writes_per_s=None))
    q = Quoter(min_spread=0.05, improve=True, pm_band=0.02, pm_pull=0.03, pm_window_s=10,
               min_requote_s=0)
    sim.apply(_snap(T0, [(0.40, 10)], [(0.50, 5)]))              # yes 0.40 / 0.50
    sim.apply(Event(T0 + 1, "ext", "M", {"mid": 0.42}))           # fair 0.42: yes bid 0.41 too rich
    q.on_event(sim, "M")
    assert {o.side for o in sim.live_orders("M")} == {"no"}      # no bid 0.51 <= 0.58 - 0.02
    sim.apply(Event(T0 + 2 * 10**9, "ext", "M", {"mid": 0.47}))   # +5c in 2 s: pull everything
    q.on_event(sim, "M")
    assert not sim.live_orders("M")


def test_quoter_skew_moves_both_bids_toward_flat():
    from albert.backtest.l2 import Quoter

    q = {"yes_bid": 0.40, "yes_ask": 0.45}
    s = Quoter(skew=0.01, pair_edge=0.01)
    s._bought["M"] = {"yes": [2.0, 0.80]}                      # long 2 YES at 0.40
    assert s._skewed("M", q, 2.0, 0.40, 0.55) == (0.38, 0.57)  # NO up 2c, YES down 2c
    s._bought["M"] = {"yes": [2.0, 0.90]}                      # at 0.45: a pair must cost <= 0.99
    assert s._skewed("M", q, 2.0, 0.40, 0.55) == (0.38, 0.55)  # cap 0.54 < base: base kept
    s2 = Quoter(skew=0.05)
    s2._bought["M"] = {"no": [3.0, 1.50]}                      # long 3 NO at 0.50 (pos -3)
    yes, no = s2._skewed("M", q, -3.0, 0.40, 0.55)
    assert yes == 0.44 and no == 0.40                          # YES capped 1c under the ask
    assert Quoter(skew=0.01)._skewed("M", q, 0.4, 0.40, 0.55) == (0.40, 0.55)  # < 1c: no change
    assert Quoter(skew=0.30)._skewed("M", q, 2.0, 0.40, 0.55)[0] is None       # below 1c: pulled


def test_pair_breakdown_parts_add_up():
    import polars as pl

    from albert.backtest.l2 import pair_breakdown

    fills = pl.DataFrame({
        "market": ["E1-A", "E1-A", "E1-A", "E2-B"], "side": ["yes", "no", "yes", "no"],
        "qty": [1.0, 1.0, 1.0, 2.0], "price": [0.40, 0.55, 0.42, 0.30], "fee": [0.0, 0.01, 0.0, 0.0],
        "final_yes": [1.0, 1.0, 1.0, 0.7], "settled": [True, True, True, False]})
    b = pair_breakdown(fills)
    # E1-A: 1 pair at avg YES 0.41 + NO 0.55 -> +0.04; 1 unpaired YES at 0.41 settles 1 -> +0.59
    # E2-B: 2 NO at 0.30 marked at 0.30 -> 0
    assert abs(b["pair_pnl"] - 0.04) < 1e-9 and abs(b["directional_pnl"] - 0.59) < 1e-9
    total = sum((f if s == "yes" else 1 - f) * q - p * q - fee for s, q, p, fee, f in
                zip(fills["side"], fills["qty"], fills["price"], fills["fee"], fills["final_yes"], strict=True))
    assert abs(b["pnl"] - total) < 1e-9
    assert b["paired_contracts"] == 2 and b["contracts"] == 5 and b["events"] == 2
    assert pair_breakdown(fills, settled_only=True)["markets"] == 1
