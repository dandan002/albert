import json
import math
from pathlib import Path

import polars as pl
import pytest

from albert.backtest.l2 import SimConfig, Simulator, _mark_executions, events_from_rows
from albert.research import mm_models as mm

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _events(name: str = "ws_stream_ladder_arb.jsonl"):
    rows = [json.loads(line) for line in (FIX / name).read_text().splitlines()]
    evs = events_from_rows(rows)
    _mark_executions(evs)
    return evs


def _calib(evs, k: float = 0.5, kx: float = 8.0) -> mm.Calib:
    d = mm.extract(evs)
    sig = mm.sigma_buckets(mm.grid(d["tob"], d["active"]))
    sl = mm.SigmaLookup(sig, prior_p=0.01, prior_x=4e-5)
    return mm.Calib(ak={(None, None, None): (1e-3, k)}, akx={(None, None): (1e-3, kx)}, sigma=sl)


def _run(evs, strat, latency_ms: float = 50.0):
    sim = Simulator(SimConfig(latency_ms=latency_ms, writes_per_s=10.0))
    placed = []
    for ev in evs:
        sim.apply(ev)
        before = sim._next
        q = sim.markets[ev.market].quote() if ev.market in sim.synced else None
        strat.on_event(sim, ev.market)
        for oid in range(before, sim._next):
            placed.append((ev.recv_ns, oid, q))
    return sim, placed


def test_extract_takes_the_book_before_each_execution():
    for name in ("ws_stream_wide.jsonl", "ws_stream_ladder_arb.jsonl"):
        d = mm.extract(_events(name))
        tr = d["trades"]
        assert tr.height > 0 and d["tob"].height > 0 and d["active"].height > 0
        both = tr.filter(pl.col("pre_bid").is_not_null() & pl.col("pre_ask").is_not_null())
        assert both.height > 0
        # a taker buying YES pays at least the pre-trade ask; selling, at most the pre-trade bid
        bad = both.filter(((pl.col("taker_side") == "yes") & (pl.col("yes_price") < pl.col("pre_ask") - 1e-9))
                          | ((pl.col("taker_side") == "no") & (pl.col("yes_price") > pl.col("pre_bid") + 1e-9)))
        assert bad.height == 0


def test_taker_reach_and_intensity_are_monotone():
    d = mm.extract(_events("ws_stream_wide.jsonl"))
    o = mm.taker_orders(d["trades"], lo=0.0, hi=1.0)
    # one 388-lot sweep through 7c..12c against a 6/7c book: it reached 12 - 6.5 = 5.5c beyond the mid
    assert o.height == 1 and o["d_c"][0] == pytest.approx(5.5) and o["qty"][0] == pytest.approx(388.0)
    g = mm.grid(d["tob"], d["active"])
    assert g.height > 0
    expo = g.select(pl.len().alias("seconds") * 10, pl.lit(1).alias("markets"))
    lam = mm.intensity(o, expo, [])
    counts = lam["count"].to_list()
    assert counts[:5] == [1] * 5 and counts[5:] == [0] * 5          # strictly beyond delta = 1..5
    assert all(a >= b for a, b in zip(lam["lam"].to_list(), lam["lam"].to_list()[1:]))
    touch = mm.intensity(o, expo, [], touch=True)
    assert touch["count"].sum() >= lam["count"].sum()


def test_fit_curve_recovers_exponential_and_power_law():
    ds = list(range(1, 11))
    f = mm.fit_curve(ds, [2e-3 * math.exp(-0.4 * d) for d in ds], [100] * 10)
    assert f["A"] == pytest.approx(2e-3) and f["k"] == pytest.approx(0.4) and f["r2_exp"] == pytest.approx(1.0)
    p = mm.fit_curve(ds, [1e-3 * d ** -1.3 for d in ds], [100] * 10)
    assert p["alpha"] == pytest.approx(1.3) and p["r2_pow"] == pytest.approx(1.0) and p["r2_exp"] < 1.0
    assert mm.fit_curve([1, 2], [1e-3, 5e-4], [3, 1])["k"] is None


def test_sigma_lookup_uses_only_finished_buckets():
    bns = 300 * 10**9
    sig = pl.DataFrame({"market": ["M"] * 3, "bucket_ns": [0, bns, 2 * bns], "S": [9.0, 0.0, 1e6],
                        "Sx": [0.01, 0.0, 1e3], "T": [300.0, 300.0, 300.0]})
    sl = mm.SigmaLookup(sig, prior_p=0.01, prior_x=1e-5, prior_s=900.0)
    # inside bucket 2 only buckets 0 and 1 count; bucket 2's huge move is invisible
    vp, vx = sl.at("M", 2 * bns + 5)
    assert vp == pytest.approx((9.0 + 0.01 * 900) / (600 + 900))
    assert vx == pytest.approx((0.01 + 1e-5 * 900) / (600 + 900))
    assert sl.at("M", 3 * bns)[0] > 100       # once bucket 2 has ended it is used
    assert sl.at("unknown", 0) == pytest.approx((0.01, 1e-5))


def test_model_formulas():
    b, a = mm.as_quotes(50, 0, 0.05, 0.01, 3600, 0.5)
    assert (a + b) / 2 == pytest.approx(50)
    assert a - b == pytest.approx(0.05 * 0.01 * 3600 + (2 / 0.05) * math.log(1 + 0.05 / 0.5))
    b1, a1 = mm.as_quotes(50, 2, 0.05, 0.01, 3600, 0.5)
    assert (a1 + b1) / 2 == pytest.approx(50 - 2 * 0.05 * 0.01 * 3600) and a1 - b1 == pytest.approx(a - b)
    c = mm.glft_skew(0.01, 0.02, 1e-4, 0.3)
    g0b, g0a = mm.glft_quotes(50, 0, 0.01, 0.02, 1e-4, 0.3)
    g2b, g2a = mm.glft_quotes(50, 2, 0.01, 0.02, 1e-4, 0.3)
    assert (g0a + g0b) / 2 == pytest.approx(50)
    assert ((g0a + g0b) - (g2a + g2b)) / 2 == pytest.approx(2 * c)        # skew c per contract
    assert g2a - g2b == pytest.approx(g0a - g0b)
    assert (g0a - g0b) / 2 == pytest.approx(math.log(1 + 0.01 / 0.3) / 0.01 + c / 2)
    lb, la = mm.as_logit_quotes(0.5, 0, 0.3, 1e-4, 3600, 8.0)
    assert (la + lb) / 2 == pytest.approx(50) and 0 < lb < 50 < la < 100
    tb, ta = mm.as_logit_quotes(0.1, 0, 0.3, 1e-4, 3600, 8.0)
    assert 10 - tb < ta - 10                 # near the boundary the logit map compresses the lower side
    assert mm.to_ticks(47.5, 52.5) == (0.47, 0.53) and mm.to_ticks(47.6, 52.4) == (0.48, 0.52)


def test_model_quoter_never_crosses_stays_in_front_and_fills():
    evs = _events()
    cal = _calib(evs)
    for model in ("as", "glft", "as_logit"):
        strat = mm.ModelQuoter(cal, model=model, gamma=0.01 if model != "as_logit" else 0.3, lo=0.0, hi=1.0,
                               min_requote_s=0.0)
        sim, placed = _run(evs, strat)
        assert placed, model
        for _, oid, q in placed:
            o = sim.orders[oid]
            if o.side == "yes":
                assert q["yes_bid"] < o.price < q["yes_ask"]
            else:
                assert 1 - q["yes_ask"] < o.price < 1 - q["yes_bid"]
        assert sim.diag.get("post_only_rejects", 0) == 0
        assert abs(sim.position[evs[-1].market]) <= 3
    # The ladder fixture's only taker hit inside our quotes comes 7 ms after the book widens (before
    # a 50 ms order is live). In the 1c-wide fixture the model's ask rests behind the best
    # (join="model", at the 9c ask level) and the recorded 388-lot sweep clears that level.
    wide = _events("ws_stream_wide.jsonl")
    sim, _ = _run(wide, mm.ModelQuoter(_calib(wide), model="glft", gamma=0.01, lo=0.0, hi=1.0, join="model"))
    assert sim.fills and all(f.side == "no" and f.price == pytest.approx(0.91) for f in sim.fills)
    rows = mm.fill_rows(sim, {})
    assert rows and all(r["mark"] is not None and r["event"] for r in rows)
    out = mm.summarize(rows, {r["market"]: 1.0 for r in rows})
    assert out["fills"] == len(rows) and out["settled_n"] == len(rows) and out["settle"] is not None


def test_inventory_skews_quotes_and_respects_bounds():
    evs = _events()
    cal = _calib(evs)
    sim = Simulator(SimConfig())
    for ev in evs:
        sim.apply(ev)
    m = "KXTOKENUSE-26NOV09-T207"
    bid, ask = sim.quotes[m][1][-1]
    strat = mm.ModelQuoter(cal, model="glft", gamma=0.01, lo=0.0, hi=1.0)
    b0, a0 = strat.targets(sim, m, bid, ask)
    sim.position[m] = 2.0
    b2, a2 = strat.targets(sim, m, bid, ask)
    assert b2 < b0 and a2 <= a0
    sim.position[m] = 3.0
    assert strat.targets(sim, m, bid, ask)[0] is None
    sim.position[m] = -3.0
    assert strat.targets(sim, m, bid, ask)[1] is None
    noskew = mm.ModelQuoter(cal, model="glft", gamma=0.01, lo=0.0, hi=1.0, skew=False)
    sim.position[m] = 2.0
    assert noskew.targets(sim, m, bid, ask)[0] == b0


def test_decisions_do_not_depend_on_future_events():
    evs = _events()
    cal = _calib(evs)
    kw = dict(model="glft", gamma=0.01, lo=0.0, hi=1.0, min_requote_s=0.0)
    _, full = _run(evs, mm.ModelQuoter(cal, **kw))
    assert full
    idx = {e.recv_ns: i for i, e in enumerate(evs)}
    for t, _, _ in full[:: max(1, len(full) // 8)]:
        _, part = _run(evs[: idx[t] + 1], mm.ModelQuoter(cal, **kw))
        assert [(a, b) for a, b, _ in part] == [(a, b) for a, b, _ in full if a <= t]


@pytest.mark.parametrize("name", ["ws_stream_ladder_arb.jsonl", "ws_stream_wide.jsonl"])
def test_lockstep_equals_separate_runs(name):
    """Simulators sharing one quote history (and pruned) give the same fills and orders as alone."""
    evs = _events(name)
    cal = _calib(evs)
    specs = [("glft", 50.0, lambda: mm.ModelQuoter(cal, model="glft", gamma=0.01, lo=0.0, hi=1.0)),
             ("glft_join", 50.0, lambda: mm.ModelQuoter(cal, model="glft", gamma=0.01, lo=0.0, hi=1.0,
                                                        join="model")),
             ("as", 100.0, lambda: mm.ModelQuoter(cal, model="as", gamma=0.01, lo=0.0, hi=1.0)),
             ("join", 50.0, lambda: mm.Quoter(min_spread=0.0, lo=0.0, hi=1.0)),
             ("base", 50.0, lambda: mm.Quoter(improve=True, min_spread=0.05, lo=0.0, hi=1.0))]
    runs = mm.run_lockstep(evs, specs, {}, prune_every=50)
    for n, lat, make in specs:
        alone = mm.run_lockstep(evs, [(n, lat, make)], {}, prune_every=0)[n][0]
        both = runs[n][0]
        assert [(f.t_ns, f.side, f.price, f.qty) for f in both.fills] == \
               [(f.t_ns, f.side, f.price, f.qty) for f in alone.fills]
        assert both._next == alone._next
    assert sum(s._next - 1 for s, _ in runs.values()) > 0
    if name == "ws_stream_wide.jsonl":
        assert len(runs["glft_join"][0].fills) > 0 and len(runs["join"][0].fills) > 0


def test_clustered_guards_degenerate_dispersion():
    rows = [{"event": e, "qty": 1.0, "x": -0.01} for e in "abc"]
    mean, t, k = mm.clustered(rows, "x")
    assert mean == pytest.approx(-0.01) and t is None and k == 3
    rows.append({"event": "d", "qty": 2.0, "x": 0.03})
    mean, t, k = mm.clustered(rows, "x")
    assert mean == pytest.approx((-0.03 + 0.06) / 5) and t is not None and k == 4


def test_gate_caps_the_price_at_one_tick_inside():
    evs = _events()
    cal = _calib(evs, k=2.0)          # 1/k = 0.5c: uncapped, the model would quote deep inside
    deep = mm.ModelQuoter(cal, model="glft", gamma=0.01, lo=0.0, hi=1.0, min_requote_s=0.0)
    gate = mm.ModelQuoter(cal, model="glft", gamma=0.01, lo=0.0, hi=1.0, min_requote_s=0.0, max_improve=1)
    sim_d, placed_d = _run(evs, deep)
    sim_g, placed_g = _run(evs, gate)
    assert placed_g
    for _, oid, q in placed_g:
        o = sim_g.orders[oid]
        best = q["yes_bid"] if o.side == "yes" else 1 - q["yes_ask"]
        assert o.price == pytest.approx(best + 0.01)
    inside = [sim_d.orders[oid].price - (q["yes_bid"] if sim_d.orders[oid].side == "yes" else 1 - q["yes_ask"])
              for _, oid, q in placed_d]
    assert max(inside) > 0.015                   # the uncapped model does go deeper


def test_fill_context_measures_ticks_inside():
    wide = _events("ws_stream_wide.jsonl")
    sim, _ = _run(wide, mm.ModelQuoter(_calib(wide), model="glft", gamma=0.01, lo=0.0, hi=1.0, join="model"))
    rows = mm.fill_rows(sim, {})
    for r in rows:
        r.update(config="glft", window="dev")
    f = mm.fill_context(pl.DataFrame(rows), mm.extract(wide)["tob"])
    # the NO bid at 91c (YES 9c) rested behind the 7c and 8c asks; by the time the sweep reached
    # it, those levels were gone: it was the best (0 ticks inside) in a 6c/9c book
    assert f.height == len(rows) > 0 and f["ticks_in"].to_list() == [0.0] * f.height
    assert f["spread_c"][0] == pytest.approx(3.0) and f["sbucket"][0] == "s3-4"
    t = mm.context_table(f, "ticks_in_b")
    assert t.height == 1 and t["fills"][0] == f.height
