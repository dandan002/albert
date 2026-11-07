"""Cross-strike curve study on real recordings: six minutes of the KXTOKENUSE-26NOV09 threshold
ladder (7 recorded strikes, 2026-11-05 05:00-05:06 UTC, trimmed from the recorder's stream) and
the recorded public REST metadata of six events (trimmed to the strike fields)."""

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from albert.backtest.l2 import SimConfig, Simulator, _mark_executions, events_from_rows
from albert.research.arb_l2 import taker_fee
from albert.research.strike_curves import (CurveQuoter, TobIndex, _with_end, clustered,
                                           fair_events, fit_batch, fit_ladder, grid_panel,
                                           isotonic_decreasing, ladder_states,
                                           ladders_from_events, leg_probs, residual_dynamics,
                                           slope_clustered, taker_signals, taker_trades,
                                           threshold_nonparam, tob_from_events, weather_legs)

FIX = Path(__file__).parent / "fixtures" / "kalshi"
EVENTS = json.loads((FIX / "events_strike_ladders.json").read_text())
ROWS = [json.loads(line) for line in (FIX / "ws_stream_strike_ladder.jsonl").read_text().splitlines()]
RECORDED = {json.loads(r["frame"])["msg"]["market_ticker"] for r in ROWS}


def _ladder():
    lads = ladders_from_events(EVENTS, RECORDED)
    return next(ld for ld in lads if ld.event == "KXTOKENUSE-26NOV09")


def _tob():
    evs = [(r["recv_ns"], {"orderbook_snapshot": "snapshot", "orderbook_delta": "delta"}.get(
        json.loads(r["frame"])["type"], "trade"), json.loads(r["frame"])["msg"]) for r in ROWS]
    return tob_from_events(_with_end(evs))


@pytest.fixture(scope="module")
def states():
    st = ladder_states(_ladder(), _tob())
    return st.with_columns(pl.lit("dev").alias("split"), pl.lit(False).alias("volatile"),
                           pl.lit(False).alias("volatile_prev"))


def test_ladders_from_recorded_metadata():
    everything = {m["ticker"] for p in EVENTS.values() for m in p["event"]["markets"]}
    lads = {ld.key: ld for ld in ladders_from_events(EVENTS, everything)}
    tok = lads["KXTOKENUSE-26NOV09"]
    assert tok.kind == "threshold" and len(tok.legs) == 13
    assert all(lg.kind == "gt" and lg.hi == math.inf for lg in tok.legs)
    assert [lg.lo for lg in tok.legs] == sorted(lg.lo for lg in tok.legs)
    # whole-degree weather brackets: half-degree edges, complete and mutually exclusive
    ny = lads["KXHIGHNY-26NOV05"]
    assert ny.kind == "bracket" and ny.complete and len(ny.legs) == 6
    edges = [(lg.lo, lg.hi) for lg in ny.legs]
    assert edges[0] == (-math.inf, 69.5) and edges[1] == (69.5, 71.5) and edges[-1][1] == math.inf
    assert all(edges[i][1] == edges[i + 1][0] for i in range(5))           # a partition of X
    # approval in 0.1 steps: "below 38.8" | "38.8" | ... ; the tail starts where the last bin ends
    ap = lads["KXTRUMPAPPROVE-26NOV05"]
    e = [(lg.lo, lg.hi) for lg in ap.legs]
    assert e[0] == (-math.inf, pytest.approx(38.75)) and e[1] == (pytest.approx(38.75), pytest.approx(38.85))
    assert all(e[i][1] == pytest.approx(e[i + 1][0]) for i in range(len(e) - 1))
    # one event, two teams' totals: split into two ladders; rain by city is not a ladder
    assert {"KXNFLTEAMTOTAL-26NOV08GBNE|NE", "KXNFLTEAMTOTAL-26NOV08GBNE|GB"} <= set(lads)
    assert not any(k.startswith("KXRAIN") for k in lads)
    # only recorded legs are kept
    assert len(_ladder().legs) == 7


def test_weather_sub_titles_parse_to_half_degree_edges():
    subs = {m["yes_sub_title"]: m for m in EVENTS["KXHIGHNY-26NOV05"]["event"]["markets"]}
    got = {s: weather_legs(s) for s in subs}
    assert got["69° or below"] == (-math.inf, 69.5)
    assert got["70° to 71°"] == (69.5, 71.5)
    assert all(v is not None for v in got.values())


def test_curve_fits_reproduce_a_parametric_ladder_and_leave_one_out():
    ld = _ladder()
    lo, hi = ld.edges()
    truth = leg_probs(lo, hi, np.array([[0.05, math.log(0.08)]]), "logistic")[0]
    w = np.ones(len(lo))
    full, loo, th = fit_ladder(lo, hi, truth, w, "logistic")
    assert np.allclose(full, truth, atol=1e-4) and np.allclose(loo, truth, atol=1e-3)
    assert th[0] == pytest.approx(0.05, abs=1e-3)
    for fam in ("normal", "skewnormal"):
        f, lo_o, _ = fit_ladder(lo, hi, truth, w, fam)
        assert np.all(np.diff(f) <= 1e-9) and np.max(np.abs(f - truth)) < 0.03


def test_batched_fit_of_unrelated_ladders_matches_one_by_one():
    everything = {m["ticker"] for p in EVENTS.values() for m in p["event"]["markets"]}
    ny = next(ld for ld in ladders_from_events(EVENTS, everything) if ld.event == "KXHIGHNY-26NOV05")
    lo, hi = ny.edges()
    truths = [leg_probs(lo, hi, np.array([[m, math.log(s)]]), "normal")[0]
              for m, s in ((-0.1, 0.15), (0.05, 0.2), (0.2, 0.1))]
    L, H, P = np.tile(lo, (3, 1)), np.tile(hi, (3, 1)), np.array(truths)
    th = fit_batch(L, H, P, np.ones_like(P), "normal", np.tile([0.0, math.log(0.2)], (3, 1)))
    assert np.allclose(leg_probs(L, H, th, "normal"), P, atol=1e-4)
    for i, t in enumerate(truths):
        full, _, th1 = fit_ladder(lo, hi, t, np.ones(6), "normal")
        assert np.allclose(th[i], th1, atol=1e-3)
    assert abs(sum(truths[0]) - 1) < 1e-9            # the six brackets partition the line


def test_isotonic_pools_violations_only():
    y = np.array([0.9, 0.7, 0.75, 0.4])
    fit = isotonic_decreasing(y, np.array([1.0, 1.0, 3.0, 1.0]))
    assert fit[0] == 0.9 and fit[3] == 0.4
    assert fit[1] == fit[2] == pytest.approx((0.7 + 3 * 0.75) / 4)


def test_top_of_book_from_the_recording_is_never_crossed_and_ends_with_the_file():
    tob = _tob()
    assert tob.height > 50 and tob["market"].n_unique() == 7
    q = tob.drop_nulls(["bid", "ask"])
    assert (q["bid"] < q["ask"]).all()
    last = tob.group_by("market").agg(pl.col("bid").last(), pl.col("ask").last())
    assert last["bid"].null_count() == 7 and last["ask"].null_count() == 7    # end markers


def test_states_fit_the_recorded_ladder(states):
    assert states["t_ns"].n_unique() > 8 and states.height == 7 * states["t_ns"].n_unique()
    two = states.filter(pl.col("mid").is_not_null())     # quotes are kept even without a curve
    # monotone fitted survival across strikes at every state
    for _, g in states.group_by("t_ns"):
        f = g.sort("leg")["f_logistic"].to_numpy()
        assert np.all(np.diff(f[np.isfinite(f)]) <= 1e-9)
    # LOO needs an over-determined fit: three other quoted legs for a two-parameter curve
    assert two.filter(pl.col("n_valid") >= 4)["l_logistic"].null_count() == 0
    assert two.filter(pl.col("n_valid") < 4)["l_logistic"].null_count() == two.filter(pl.col("n_valid") < 4).height
    assert two.filter(pl.col("n_valid") == 3)["f_logistic"].null_count() == 0
    resid = (two["l_logistic"] - two["mid"]).abs()
    assert resid.max() < 0.5 and resid.mean() > 0.0
    # leave-one-out differs from the full fit for quoted legs (own price removed)
    assert (two["l_logistic"] - two["f_logistic"]).abs().max() > 1e-4
    assert two["l_nbr"].drop_nulls().len() > 0 and two["lb"].drop_nulls().len() > 0


def test_nonparametric_neighbour_value_is_between_neighbours(states):
    ld = _ladder()
    lo, hi = ld.edges()
    g = states.filter(pl.col("t_ns") == states["t_ns"].max()).sort("leg")
    mid = g["mid"].to_numpy()
    ok = np.isfinite(mid)
    iso, nbr = threshold_nonparam(lo, hi, np.nan_to_num(mid, nan=0.5), ok.astype(float))
    idx = np.flatnonzero(ok)
    for a, b, c in zip(idx, idx[1:], idx[2:]):
        assert min(mid[a], mid[c]) - 1e-9 <= nbr[b] <= max(mid[a], mid[c]) + 1e-9
    assert np.all(np.diff(iso[idx]) <= 1e-12)


def test_grid_panel_and_residual_dynamics(states):
    panel = grid_panel(states, every_s=10, horizons=(10, 60))
    assert panel.height > 50 and {"mid_10", "l_logistic_60", "mid_m10"} <= set(panel.columns)
    panel = panel.with_columns(pl.lit("dev").alias("split"), pl.lit(False).alias("volatile"))
    dyn = residual_dynamics(panel, "logistic", thresholds=(0.0,), horizons=(10,))
    row = dyn.filter((pl.col("split") == "dev") & (pl.col("vol") == "all")).row(0, named=True)
    assert row["rows"] > 0 and row["abs_r_c"] > 0


def test_taker_fills_only_at_or_below_the_signal_price_after_latency(states):
    tob = _tob()
    sig = taker_signals(states, "logistic", -0.05, {})     # loose threshold: exercise execution
    assert sig.height > 0
    trades = taker_trades(sig, TobIndex(tob), {}, latency_ms=50, cooldown_s=60)
    assert trades
    for t in trades:
        assert t["t_fill"] == t["t_ns"] + 50_000_000
        limit = t["ask"] if t["side"] == "yes" else 1 - t["bid"]
        assert t["entry"] <= limit + 1e-9
        assert t["fee"] == pytest.approx(taker_fee(1, t["entry"]))
    by_m: dict[str, list[int]] = {}
    for t in trades:
        by_m.setdefault(t["market"], []).append(t["t_ns"])
    assert all(b - a >= 60e9 for ts in by_m.values() for a, b in zip(ts, ts[1:]))


def test_curve_quoter_quotes_only_the_cheap_side(states):
    rows = [r for r in ROWS]
    evs = events_from_rows(rows)
    _mark_executions(evs)
    fairs = fair_events(states, "logistic")
    assert fairs
    strat = CurveQuoter(mode="cheap", thr=0.01, improve=True, min_spread=0.03, cooldown_s=300,
                        min_requote_s=0.0)
    sim = Simulator(SimConfig(latency_ms=50.0))
    placed = []
    merged = sorted(evs + fairs, key=lambda e: e.recv_ns)
    for ev in merged:
        if ev.kind == "ext":
            strat.fair[ev.market] = ev.msg["mid"]
            sim._run_schedule(ev.recv_ns)
            sim.now = max(sim.now, ev.recv_ns)
        else:
            sim.apply(ev)
        before = len(sim.orders)
        q = sim.markets[ev.market].quote() if ev.market in sim.synced else None
        fair = strat.fair.get(ev.market)
        strat.on_event(sim, ev.market)
        for oid in list(sim.orders)[before:]:
            placed.append((oid, q, fair))
    assert placed
    for oid, q, fair in placed:
        mid = (q["yes_bid"] + q["yes_ask"]) / 2
        side = sim.orders[oid].side
        assert (side == "yes" and fair - mid >= 0.01 - 1e-9) or (side == "no" and mid - fair >= 0.01 - 1e-9)


def test_clustered_statistics():
    m, t, n, k = clustered([1.0, 2.0, 3.0, None], ["a", "a", "b", "b"])
    assert m == pytest.approx(2.0) and n == 3 and k == 2 and t is not None
    x = np.arange(40, dtype=float)
    b, tb = slope_clustered(x, 2 * x + np.sin(x), np.array([str(i % 5) for i in range(40)]))
    assert b == pytest.approx(2.0, abs=0.05) and tb > 10
