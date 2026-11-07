import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from albert.backtest.l2 import Quoter, SimConfig, Simulator, _mark_executions, events_from_rows
from albert.research.hawkes import (
    BASE,
    OnlineTox,
    ToxParams,
    ToxQuoter,
    _nll1_grad,
    _nll2_grad,
    build_tape,
    cl_mean,
    cl_ols,
    exec_markouts,
    features,
    fit_hawkes,
    fit_hawkes2,
    hist_grid,
    hist_maker_markouts,
    hist_taker,
    loglik1,
    loglik1_bruteforce,
    loglik2,
    merged_trades,
    run_many,
    simulate_hawkes,
    tox_regressions,
)

FIX = Path(__file__).parent / "fixtures" / "kalshi"
P = ToxParams(mu=1e-3, n=0.7, beta=0.01, mu_mid=3e-3, n_mid=0.8, beta_mid=0.75)


def _events(name: str = "ws_stream_wide.jsonl"):
    rows = [json.loads(line) for line in (FIX / name).read_text().splitlines()]
    evs = events_from_rows(rows)
    _mark_executions(evs)
    return evs


def _arrivals(name: str = "ws_trades_hawkes.jsonl") -> pl.DataFrame:
    tape = build_tape(_events(name))
    return merged_trades(tape["trades"])


def test_loglik_matches_bruteforce_on_recorded_trades():
    m = _arrivals()
    assert m.height >= 10
    t = (m["ts_ms"].to_numpy() - m["ts_ms"].min()) / 1e3
    T = float(t[-1]) + 30.0
    for mu, n, beta in ((0.005, 0.5, 0.1), (0.01, 0.9, 2.0), (0.002, 0.2, 0.01)):
        assert loglik1(mu, n, beta, [(t, T)]) == pytest.approx(loglik1_bruteforce(mu, n, beta, t, T), rel=1e-9)


def test_bivariate_without_cross_terms_is_two_univariates():
    m = _arrivals()
    t = (m["ts_ms"].to_numpy() - m["ts_ms"].min()) / 1e3
    T = float(t[-1]) + 30.0
    mk = (m["sign"].to_numpy() < 0).astype(int)
    assert 0 < mk.sum() < len(mk)
    nm = np.array([[0.4, 0.0], [0.0, 0.6]])
    two = loglik2(np.array([0.01, 0.02]), nm, 0.5, [(t, mk, T)])
    one = loglik1(0.01, 0.4, 0.5, [(t[mk == 0], T)]) + loglik1(0.02, 0.6, 0.5, [(t[mk == 1], T)])
    assert two == pytest.approx(one, rel=1e-9)


def test_analytic_gradients_match_finite_differences():
    m = _arrivals()
    t = (m["ts_ms"].to_numpy() - m["ts_ms"].min()) / 1e3
    T = float(t[-1]) + 30.0
    mk = (m["sign"].to_numpy() < 0).astype(int)
    for f, x, seqs in ((_nll1_grad, np.log([0.02, 0.5, 0.3]), [(t, T)]),
                       (_nll2_grad, np.log([0.01, 0.02, 0.3, 0.1, 0.2, 0.4, 0.3]), [(t, mk, T)])):
        _, g = f(x, seqs)
        for i in range(len(x)):
            d = np.zeros(len(x))
            d[i] = 1e-6
            num = (f(x + d, seqs)[0] - f(x - d, seqs)[0]) / 2e-6
            assert g[i] == pytest.approx(num, rel=1e-4, abs=1e-4)


def test_mle_recovers_simulated_parameters():
    t = simulate_hawkes(0.5, 0.6, 2.0, 3000, seed=1)
    r = fit_hawkes([(t, 3000.0)])
    assert r["n"] == pytest.approx(0.6, abs=0.05)
    assert r["beta"] == pytest.approx(2.0, rel=0.15)
    assert r["mu"] == pytest.approx(0.5, rel=0.1)
    assert r["ks_p"] > 0.05 and r["ks_p_poisson"] < 1e-6
    t2 = simulate_hawkes(0.4, 0.3, 1.0, 3000, seed=3)
    tt = np.concatenate([t, t2])
    mk = np.concatenate([np.zeros(len(t), int), np.ones(len(t2), int)])
    o = np.argsort(tt, kind="stable")
    r2 = fit_hawkes2([(tt[o], mk[o], 3000.0)])
    assert r2["n_bs"] < 0.05 and r2["n_sb"] < 0.05


def test_tape_matches_fixture_and_merges_sweeps():
    evs = _events()
    tape = build_tape(evs)
    n_trades = sum(1 for e in evs if e.kind == "trade")
    assert tape["trades"].height == n_trades > 0
    assert tape["execs"].height == sum(1 for e in evs if e.is_exec) > 0
    assert tape["quotes"].height > 0 and tape["windows"].height == 1
    m = merged_trades(tape["trades"])
    assert 0 < m.height <= n_trades
    assert m["qty"].sum() == pytest.approx(tape["trades"]["qty"].sum())


def test_features_use_only_events_strictly_before_the_cut():
    tape = build_tape(_events("ws_stream_ladder_arb.jsonl"))
    q = tape["execs"].select("market", pl.col("ts_ms").alias("cut_ms"), "side")
    full = features(tape, q, P)
    assert full.height == q.height > 0
    assert (full["n60"] > 0).any() and (full["lam"] > P.mu).any()     # earlier arrivals seen
    cols = ["lam", "lam_rel", "lam_mid", "ofi10", "ofi60", "rv60", "n60", "vpin", "spread_c"]
    for i, row in enumerate(q.iter_rows(named=True)):
        cut = row["cut_ms"]
        trunc = {"trades": tape["trades"].filter(pl.col("ts_ms") < cut),
                 "quotes": tape["quotes"].filter(pl.col("ts_ms") < cut),
                 "execs": tape["execs"], "windows": tape["windows"]}
        part = features(trunc, q.slice(i, 1), P)
        for c in cols:
            a, b = full[c][i], part[c][0]
            assert (a is None and b is None) or (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b), c


def test_vpin_matches_naive_buckets():
    tape = build_tape(_events())
    p = ToxParams(**{**P.__dict__, "vpin_v": 20.0, "vpin_nb": 3})
    m = merged_trades(tape["trades"])
    end = int(m["ts_ms"].max()) + 1
    f = features(tape, pl.DataFrame({"market": [m["market"][0]], "cut_ms": [end], "side": ["yes"]}), p)
    buckets, cur, buy = [], 0.0, 0.0
    for sign, qty in zip(m["sign"].to_list(), m["qty"].to_list()):
        while qty > 1e-12:
            take = min(qty, 20.0 - cur)
            cur += take
            buy += take if sign > 0 else 0.0
            qty -= take
            if cur >= 20.0 - 1e-12:
                buckets.append(abs(2 * buy - 20.0) / 20.0)
                cur, buy = 0.0, 0.0
    assert len(buckets) >= 3
    assert f["vpin"][0] == pytest.approx(sum(buckets[-3:]) / 3)


def test_exec_markouts_sign_and_horizon():
    tape = build_tape(_events("ws_stream_ladder_arb.jsonl"))
    mo = exec_markouts(tape, tape["execs"])
    assert mo.height == tape["execs"].height
    qu = tape["quotes"].sort("recv_ns")
    r = mo.filter(pl.col("mo_10s").is_not_null()).row(0, named=True)
    at = qu.filter(pl.col("recv_ns") <= r["recv_ns"] + 10 * 10**9).tail(1).row(0, named=True)
    mid = (at["bid"] + at["ask"]) / 2
    want = (mid if r["side"] == "yes" else 1 - mid) - r["price"]
    assert r["mo_10s"] == pytest.approx(want)
    exit_ = at["bid"] if r["side"] == "yes" else 1 - at["ask"]
    assert r["mx_10s"] == pytest.approx(exit_ - r["price"])


def _replay(strats, tox, evs):
    sims, at = run_many(iter(evs), strats, tox)
    return sims, at


def test_tox_quoter_without_filter_is_the_baseline_quoter_and_pull_blocks_quotes():
    evs = _events()
    cfg = SimConfig(latency_ms=50.0, writes_per_s=10.0)
    base_sim = Simulator(cfg)
    q = Quoter(**BASE)
    for ev in evs:
        base_sim.apply(ev)
        q.on_event(base_sim, ev.market)
    assert len(base_sim.orders) > 0
    tox = OnlineTox(P)
    strats = {"same": (ToxQuoter(tox, **BASE), cfg),
              "never": (ToxQuoter(tox, pull_on="lam", pull_thr=0.0, **BASE), cfg),
              "inf": (ToxQuoter(tox, pull_on="lam_rel", pull_thr=math.inf, **BASE), cfg)}
    sims, _ = _replay(strats, tox, evs)
    key = lambda s: [(o.placed_ns, o.side, o.price, o.status) for o in s.orders.values()]
    assert key(sims["same"]) == key(base_sim) == key(sims["inf"])
    assert [(f.t_ns, f.price) for f in sims["same"].fills] == [(f.t_ns, f.price) for f in base_sim.fills]
    assert len(sims["never"].orders) == 0 and strats["never"][0].pulls > 0


def test_online_intensity_matches_closed_form():
    evs = _events("ws_trades_hawkes.jsonl")
    tox = OnlineTox(P)
    sim = Simulator(SimConfig(writes_per_s=None))
    arrivals, last_key = [], None
    for ev in evs:
        sim.apply(ev)
        tox.observe(sim, ev)
        if ev.kind == "trade":
            k = (ev.msg["ts_ms"], ev.msg["taker_side"])
            if k != last_key:
                arrivals.append(ev.recv_ns / 1e9)
                last_key = k
    now = evs[-1].recv_ns + 5 * 10**9
    m = tox.measures(evs[-1].market, now)
    want = P.mu + P.alpha * sum(math.exp(-P.beta * (now / 1e9 - t)) for t in arrivals)
    assert len(arrivals) > 5 and m["lam"] == pytest.approx(want, rel=1e-9)


def test_clustered_stats_and_regression_table():
    y = np.array([1.0, 3.0, -1.0, 1.0])
    g = np.array(["a", "a", "b", "b"])
    m, _t, k = cl_mean(y, g)
    assert m == pytest.approx(1.0) and k == 2
    b, _, n, _ = cl_ols(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([1.0, 3.0, 5.0, 7.0, 9.0]),
                        np.array(["a", "b", "c", "d", "e"]))
    assert b == pytest.approx(2.0) and n == 5
    tapes = [build_tape(_events(n)) for n in ("ws_stream_wide.jsonl", "ws_stream_trades.jsonl",
                                              "ws_stream_ladder_arb.jsonl")]
    tape = {k: pl.concat([t[k] for t in tapes]) for k in tapes[0]}
    ex = exec_markouts(tape, tape["execs"])
    f = features(tape, ex.select("market", pl.col("ts_ms").alias("cut_ms"), "side"), P)
    df = pl.concat([ex, f.drop("market", "cut_ms", "side")], how="horizontal").with_columns(
        (pl.col("recv_ns") < pl.col("recv_ns").median()).alias("dev"), pl.lit(False).alias("volatile"),
        (pl.col("recv_ns") // (60 * 10**9)).cast(pl.Utf8).alias("event"))
    reg = tox_regressions(df, measures=("lam", "n60"), ycols=("mo_10s",))
    assert reg.height > 0 and reg.filter(pl.col("n") > 0).height > 0


def _hist_frame(name: str = "ws_trades_hawkes.jsonl") -> tuple[pl.DataFrame, dict]:
    """Recorded WebSocket prints in the historical-trades shape (merged arrivals)."""
    m = _arrivals(name)
    tr = m.select(pl.col("market").alias("ticker"),
                  pl.from_epoch(pl.col("ts_ms") * 1000, time_unit="us").dt.replace_time_zone("UTC").alias("created_time"),
                  "sign", "qty", pl.col("yes_price").alias("price"))
    us = tr["created_time"].dt.epoch("us")
    return tr, {tr["ticker"][0]: (int(us.min()) - 10**6, int(us.max()) + 400 * 10**6)}


def test_historical_intensity_grid_has_no_look_ahead_and_taker_proxy_pays_fees():
    tr, w = _hist_frame()
    p2 = {"mu_buy": 1e-3, "mu_sell": 1e-3, "n_bb": 0.5, "n_bs": 0.1, "n_sb": 0.1, "n_ss": 0.5, "beta": 0.05}
    g = hist_grid(tr, w, p2, step_s=5.0)
    assert g.height > 0
    # recompute one grid point from trades strictly before it
    row = g.filter(pl.col("p0").is_not_null()).row(len(g) // 2, named=True)
    a = w[tr["ticker"][0]][0]
    t = (tr["created_time"].dt.epoch("us").to_numpy() - a) / 1e6
    sg = tr["sign"].to_numpy()
    before = t < row["t_s"]
    sb = np.exp(-p2["beta"] * (row["t_s"] - t[before & (sg > 0)])).sum()
    ss = np.exp(-p2["beta"] * (row["t_s"] - t[before & (sg < 0)])).sum()
    lb = p2["mu_buy"] + p2["beta"] * (p2["n_bb"] * sb + p2["n_bs"] * ss)
    assert row["lam_buy"] == pytest.approx(lb)
    assert row["p0"] == pytest.approx(tr["price"].to_numpy()[before][-1])
    px = tr["price"].to_numpy()
    mid = (px[before & (sg > 0)][-1] + px[before & (sg < 0)][-1]) / 2
    assert row["m0"] == pytest.approx(mid)
    mk = hist_maker_markouts(tr, w, p2)
    assert mk.height == tr.height
    tk = hist_taker(tr, g, 0.0, h=10, windows=w)
    if tk.height:
        assert ((tk["gross"] - tk["pnl"]) > 0).all()     # fees always paid
        assert (tk["entry"].is_between(0.0, 1.0)).all()


def test_run_many_records_live_measures_per_fill():
    evs = _events()
    tox = OnlineTox(P)
    cfg = SimConfig(latency_ms=50.0, writes_per_s=10.0)
    strat = ToxQuoter(tox, **(BASE | {"min_spread": 0.0, "lo": 0.0, "hi": 1.0, "cooldown_s": 0}))
    sims, at = run_many(iter(evs), {"x": (strat, cfg)}, tox)
    sim = sims["x"]
    assert len(at.get("x", [])) == len(sim.fills) == len(sim.fill_ts)
    assert all(ts > 0 for ts in sim.fill_ts)
    assert sim.fills, "the wide fixture's sweep should fill an improved quote"
