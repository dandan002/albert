import json
import math
from pathlib import Path

import polars as pl
import pytest

from albert.backtest.l2 import SimConfig, _mark_executions, events_from_rows
from albert.research import fair_value as fv

FIX = Path(__file__).parent / "fixtures" / "kalshi"
NAMES = ("ws_stream_wide.jsonl", "ws_stream_trades.jsonl", "ws_stream_ladder_arb.jsonl")


def _events(name: str = "ws_stream_wide.jsonl"):
    rows = [json.loads(line) for line in (FIX / name).read_text().splitlines()]
    evs = events_from_rows(rows)
    _mark_executions(evs)
    return evs


def _all_events():
    return sorted((e for n in NAMES for e in _events(n)), key=lambda e: e.recv_ns)


@pytest.fixture(scope="module")
def tape():
    return fv.tape_from_events(_all_events())


@pytest.fixture(scope="module")
def params(tape):
    q, t = tape
    g, _ = fv.fit_micro(q, min_obs=5)
    (kq, kcm, kst), _ = fv.fit_kalman(fv.kalman_obs(q, t))
    return fv.FairParams(micro_g=g, kal_q=kq, kal_cm=kcm, kal_st=kst,
                         ou_hl={r: 30.0 for r in ("tight_calm", "mid_calm", "wide_calm",
                                                  "tight_vol", "mid_vol", "wide_vol")})


def test_tape_records_top_of_book_changes_and_trades(tape):
    q, t = tape
    assert q.height > 100 and t.height == 49 and q["market"].n_unique() == 4
    # consecutive rows of a market always differ (only changes are recorded)
    d = q.sort("market", "seq").with_columns(
        pl.struct("bid", "ask", "bsz", "asz").shift(1).over("market").alias("prev"))
    same = d.filter(pl.col("prev") == pl.struct("bid", "ask", "bsz", "asz"))
    assert same.is_empty()
    assert set(t["sgn"].unique().to_list()) <= {1.0, -1.0} and (t["qty"] > 0).all()


def test_micro_price_adjustment_is_antisymmetric(tape):
    g, diag = fv.fit_micro(tape[0], min_obs=5)
    assert diag["states_used"] > 0
    assert any(abs(v) > 0 for row in g for v in row)
    for row in g:
        assert row[2] == pytest.approx(0.0, abs=1e-9)
        for k in range(fv.N_IMB):
            assert row[k] == pytest.approx(-row[fv.N_IMB - 1 - k], abs=1e-9)


def test_imbalance_bucket_mirrors():
    for bs, as_ in ((1, 4), (4, 1), (2, 3), (3, 2), (1, 1), (0, 5), (7, 3)):
        assert fv.imb_bucket(bs, as_) == fv.N_IMB - 1 - fv.imb_bucket(as_, bs)


def test_kalman_fit_beats_the_starting_grid_point(tape):
    obs = fv.kalman_obs(*tape)
    assert sum(len(v) for v in obs.values()) > 50
    (q, cm, st), diag = fv.fit_kalman(obs)
    ll0, n = fv.kalman_loglik(obs, 1e-4, 0.3, 0.02)
    assert diag["loglik"] >= ll0 and diag["obs"] == n > 0


def test_features_are_causal(tape, params):
    """Features computed from a truncated tape equal those from the full tape up to the cut."""
    q, t = tape
    full = fv.features_from_tape(q, t, params, {}, {})
    assert full.height > 100
    cut = int(full["t"].quantile(0.5))
    part = fv.features_from_tape(q.filter(pl.col("t") <= cut), t.filter(pl.col("t") <= cut), params, {}, {})
    head = full.filter(pl.col("t") <= cut)
    assert part.height == head.height > 0
    for c in fv.BASE:
        assert part[c].to_list() == pytest.approx(head[c].to_list(), abs=1e-12)


def test_vectorised_prediction_matches_online(tape, params):
    q, t = tape
    feat = fv.features_from_tape(q, t, params, {}, {}).with_columns(pl.lit(False).alias("volatile_prev"))
    p = fv.FairParams(**{**params.__dict__})
    p.ridge = {str(h): [0.01 * (i % 5 - 2) for i in range(len(fv.FEATURES))] for h in fv.HORIZONS_S}
    for est, h in (("micro", 10), ("kal", 60), ("ou", 60), ("ridge", 60), ("ema_trd_300", 1)):
        vec = feat.select(fv.pred_expr(p, est, h).alias("p"))["p"].to_list()
        online = [fv.predict(p, r, est, h, False) for r in feat.iter_rows(named=True)]
        assert vec == pytest.approx(online, abs=1e-12)
        if est != "ema_trd_300":
            assert any(abs(v) > 0 for v in vec)


def test_targets_are_future_book_changes(tape, params):
    q, t = tape
    feat = fv.features_from_tape(q, t, params, {}, {}).filter("thin")
    end = int(q["t"].max())
    tg = fv.with_targets(feat, q, end, horizons=(60,))
    rows = tg.filter(pl.col("y_mid_60").is_not_null())
    assert rows.height > 10
    qs = q.sort("t")
    for r in rows.head(20).iter_rows(named=True):
        live = qs.filter((pl.col("market") == r["market"]) & (pl.col("t") <= r["t"] + 60 * fv.NS)).tail(1)
        mid_h = (live["bid"][0] + live["ask"][0]) / 2
        assert r["y_mid_60"] == pytest.approx(mid_h - r["mid"], abs=1e-12)
        assert r["y_bid_60"] == pytest.approx(live["bid"][0] - r["bid"], abs=1e-12)
    # past the end of the data the target is unknown
    assert tg.filter(pl.col("t") + 60 * fv.NS > end)["y_mid_60"].null_count() == tg.filter(
        pl.col("t") + 60 * fv.NS > end).height


def test_taker_pays_ask_and_both_fees(tape, params):
    q, t = tape
    feat = fv.features_from_tape(q, t, params, {}, {}).with_columns(
        pl.lit(False).alias("volatile_prev"), pl.lit(False).alias("volatile"))
    # a crude fair value (stale trade EWMA) produces signals on the wide fixture
    cands = fv.taker_candidates(feat, params, ["ema_trd_1800", "kal", "micro"], [10, 60], {}, min_thr=-0.05)
    assert cands.height > 0
    tr = fv.taker_trades(cands, q, int(q["t"].max()), thrs=(-0.05,), latencies_ms=(50.0,))
    done = tr.filter(pl.col("pnl").is_not_null())
    assert done.height > 0
    for r in done.iter_rows(named=True):
        fee_in = math.ceil(round(0.07 * r["entry"] * (1 - r["entry"]) * 1e4, 9)) / 1e4
        fee_out = math.ceil(round(0.07 * r["exit"] * (1 - r["exit"]) * 1e4, 9)) / 1e4
        assert r["pnl"] == pytest.approx(r["exit"] - r["entry"] - fee_in - fee_out, abs=1e-9)
        assert r["gross"] == pytest.approx(r["exit"] - r["entry"], abs=1e-12)
    # one open trade per market and strategy
    for _, g in tr.group_by(["est", "h", "thr", "market"]):
        ts = sorted(g["t"].to_list())
        h = g["h"][0]
        assert all(b - a >= h * fv.NS for a, b in zip(ts, ts[1:]))
    s = fv.taker_summary(tr, int(q["t"].max()) + 1)
    assert s.filter((pl.col("split") == "dev") & (pl.col("regime") == "all"))["trades"].sum() == done.height


def test_fair_quoter_drop_and_skew_never_cross():
    evs = _events()
    q, t = fv.tape_from_events(evs)
    g, _ = fv.fit_micro(q, min_obs=5)
    params = fv.FairParams(micro_g=g)
    tracker = fv.Tracker(params)
    specs = [{"mode": "base"}, {"mode": "drop", "est": "micro", "thr": 0.0},
             {"mode": "drop", "est": "micro", "thr": 0.5}, {"mode": "skew", "est": "kal"}]
    runs = [(SimConfig(latency_ms=50.0), fv.FairQuoter(tracker=tracker, improve=True, min_spread=0.05,
                                                       cooldown_s=300, **sp)) for sp in specs]
    placed = {i: [] for i in range(len(specs))}
    orig = [s for _, s in runs]
    for i, strat in enumerate(orig):
        base_wanted = strat._wanted

        def wrapped(sim, market, _bw=base_wanted, _i=i):
            w = _bw(sim, market)
            qq = sim.markets[market].quote()
            placed[_i].append((qq, dict(w)))
            return w
        strat._wanted = wrapped
    sims = fv.run_many(evs, runs, tracker)
    assert len(sims[0].orders) > 0
    # an impossible edge requirement removes every quote
    assert len(sims[2].orders) == 0 and orig[2].dropped > 0
    for i in range(len(specs)):
        for qq, w in placed[i]:
            if w["yes"] is not None:
                assert w["yes"] < qq["yes_ask"] - 1e-9
            if w["no"] is not None:
                assert w["no"] < 1 - qq["yes_bid"] - 1e-9
    # the fair value at decision time is built only from events already applied
    assert all(st.t <= sims[0].now for st in tracker.states.values())
    rows = fv.maker_rows(sims[0], {})
    out = fv.maker_summary(rows, sims[0].now + 1, {}, {"mode": "base"})
    assert out[0]["fills"] == len(rows)


def test_fast_simulator_reproduces_the_stock_simulator():
    """run_many (shared best-price cache, pruned order lists) gives the stock Simulator's
    orders and fills, for the baseline and for overlays."""
    from albert.backtest.l2 import Quoter, Simulator

    evs = _all_events()
    q, t = fv.tape_from_events(evs)
    g, _ = fv.fit_micro(q, min_obs=5)
    params = fv.FairParams(micro_g=g)
    base = dict(improve=False, min_spread=0.01, lo=0.0, hi=1.0, max_pos=50)   # quotes and fills here
    ref = Simulator(SimConfig(latency_ms=50.0))
    quoter = Quoter(**base)
    ref2 = Simulator(SimConfig(latency_ms=50.0))
    tr0 = fv.Tracker(params)
    lean = fv.FairQuoter(tracker=tr0, mode="lean", est="kal", thr=0.005, **base)
    for ev in evs:
        ref.apply(ev)
        quoter.on_event(ref, ev.market)
        ref2.apply(ev)
        tr0.observe(ref2, ev)
        lean.on_event(ref2, ev.market)
    tr = fv.Tracker(params)
    runs = [(SimConfig(latency_ms=50.0), fv.FairQuoter(tracker=tr, mode="base", **base)),
            (SimConfig(latency_ms=50.0), fv.FairQuoter(tracker=tr, mode="lean", est="kal", thr=0.005, **base))]
    sims = fv.run_many(evs, runs, tr)

    def key(sim):
        return ([(f.t_ns, f.market, f.side, f.price, f.qty, f.how) for f in sim.fills]
                + [(o.oid, o.market, o.side, o.price, o.placed_ns, o.status) for o in sim.orders.values()])

    assert len(ref.orders) > 10 and len(ref.fills) > 0
    assert key(sims[0]) == key(ref) and len(sims[0].orders) == len(ref.orders)
    assert key(sims[1]) == key(ref2) and len(sims[1].orders) == len(ref2.orders)
    assert sims[1] is not sims[0] and runs[1][1].dropped == lean.dropped


def test_prediction_table_reports_r2_against_no_change(tape, params):
    q, t = tape
    feat = fv.features_from_tape(q, t, params, {}, {}).filter("thin").with_columns(
        pl.lit(False).alias("volatile_prev"), pl.lit(False).alias("volatile"))
    smp = fv.with_targets(feat, q, int(q["t"].max()))
    split = int(smp["t"].quantile(0.5))
    for by in ("regime", "band"):
        tab = fv.prediction_table(smp, params, split, ("kal", "ema_mid_10"), by=by)
        rows = tab.filter(pl.col("r2_raw").is_not_null())
        assert rows.height > 0
        r = rows.row(0, named=True)
        # recompute one R^2 by hand
        h, est, target = r["h"], r["est"], r["target"]
        d = smp.with_columns(fv.pred_expr(params, est, h).alias("p"),
                             (pl.col("t") >= split).alias("hold"), fv.band_expr().alias("band"))
        d = d.filter(pl.col(f"y_mid_{h}").is_not_null() & (pl.col("hold") == (r["split"] == "holdout")))
        if by == "band":
            d = d.filter(pl.col("band") == r["band"])
        elif r["regime"] == "volatile":
            d = d.filter(pl.col("volatile"))
        if target == "mid":
            y, p = d[f"y_mid_{h}"].to_numpy(), d["p"].to_numpy()
            assert r["r2_raw"] == pytest.approx(1 - ((y - p) ** 2).sum() / (y * y).sum())
