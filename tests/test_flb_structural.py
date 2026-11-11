"""Tests for the structural decomposition (flb_structural): Shin solver, no-arbitrage band, LLO and
Prelec maximum-likelihood fits, outcome-fitted Shin z, Wald test. Synthetic data with known
parameters; the real-panel smoke test runs only where the xs panel exists."""

import math

import numpy as np
import polars as pl
import pytest

from albert.research import flb_structural as F
from albert.research import flb_xs as X


# ---- Shin ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("z", [0.0, 0.02, 0.08, 0.2])
def test_shin_solver_recovers_z_and_probabilities(z):
    pi = np.array([0.55, 0.25, 0.12, 0.05, 0.02, 0.01])
    q = F.shin_forward(pi, z)
    assert q.sum() >= 1 - 1e-12                              # insiders create an overround
    r = F.shin_solve(q)
    assert r["z"] == pytest.approx(z, abs=1e-8)
    assert np.allclose(r["probs"], pi, atol=1e-8)
    # equation eq:shin evaluated at the true z and S = sum q gives the true probabilities
    assert np.allclose(F.shin_probs(q, z, q.sum()), pi, atol=1e-10)


def test_shin_shades_longshots_proportionally_most():
    pi = np.array([0.7, 0.2, 0.08, 0.02])
    q = F.shin_forward(pi, 0.1)
    ratio = q / pi
    assert np.all(np.diff(ratio) > 0)                         # price / probability rises as pi falls


def test_shin_without_overround_is_normalisation():
    q = np.array([0.5, 0.3, 0.15])                           # sums to 0.95: asks below one
    r = F.shin_solve(q)
    assert r["z"] == 0.0 and not r["overround"]
    assert np.allclose(r["probs"], q / q.sum())


def test_shin_fixed_z_scale_free():
    pi = np.array([0.6, 0.3, 0.07, 0.03])
    q = F.shin_forward(pi, 0.12)
    assert np.allclose(F.shin_fixed_z(q, 0.12), pi, atol=1e-9)
    assert np.allclose(F.shin_fixed_z(3.7 * q, 0.12), F.shin_fixed_z(q, 0.12), atol=1e-9)   # only relative prices matter
    assert np.allclose(F.shin_fixed_z(q, 0.0), q / q.sum())
    assert F.shin_fixed_z(q, 0.3).sum() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("z_true", [0.0, 0.10])
def test_outcome_fitted_shin_z_recovers_true_z(z_true):
    rng = np.random.default_rng(7)
    n_ev, legs = 2500, 5
    lw = []
    for _ in range(n_ev):
        pi = rng.dirichlet(np.full(legs, 0.8))
        pi = np.clip(pi, 1e-3, None)
        pi = pi / pi.sum()
        q = F.shin_forward(pi, z_true)
        win = rng.choice(legs, p=pi)
        grid = np.vstack([F.shin_fixed_z(q, z) for z in F.Z_GRID])
        lw.append(np.log(np.clip(grid[:, win], 1e-12, 1)))
    r = F.zfit_grid(np.vstack(lw), np.ones(n_ev), np.arange(n_ev), boots=100)
    assert abs(r["z_ml"] - z_true) <= 0.03
    assert r["ci_lo"] <= r["z_ml"] <= r["ci_hi"] and r["ci_hi"] - r["ci_lo"] < 0.15


# ---- the no-arbitrage band -------------------------------------------------------------------------
def _sides(**cols):
    base = {"subcent": False, "fee_multiplier": 1.0, "maker_fee": False, "lead_h": 24.0}
    n = len(next(iter(cols.values())))
    return pl.DataFrame({**{k: [v] * n for k, v in base.items() if k not in cols}, **cols})


def test_band_components_add_up_and_match_formulas():
    S = _sides(q_mid=[0.035, 0.005, 0.95], q_bid=[0.03, 0.0, 0.94], q_ask=[0.04, 0.01, 0.96], won=[0.0, 0.0, 1.0],
               lead_h=[720.0, 1.0, 168.0], fee_multiplier=[1.0, 1.0, 0.5])
    B = F.band_components(S, rho=0.05)
    tot = B["c_floor"] + B["c_spread"] + B["c_fee"] + B["c_cap"] + B["resid"]
    assert np.allclose(tot.to_numpy(), B["gross"].to_numpy())
    r0 = B.row(0, named=True)
    T = 720 / 8760
    assert r0["c_fee"] == pytest.approx(0.07 * 0.03 * 0.97)
    assert r0["c_cap"] == pytest.approx(0.05 * T * 0.97)
    assert r0["resid"] == pytest.approx(0.03 - 0.07 * 0.03 * 0.97 - 0.05 * T * 0.97 - 0.0)
    assert r0["c_spread"] == pytest.approx(0.005) and r0["c_floor"] == 0
    r1 = B.row(1, named=True)                                 # no bid, 1c ask: pinned at the floor
    assert r1["pinned"] and r1["c_floor"] == pytest.approx(0.005) and r1["c_spread"] == 0 and r1["c_fee"] == 0
    assert B.row(2, named=True)["c_fee"] == pytest.approx(0.035 * 0.94 * 0.06)   # series multiplier 0.5


def test_band_reduces_to_proposition_without_spread():
    """With q_bid = q_ask = q and won replaced by pi, resid <= 0 exactly when pi/q is above the lower
    bound of eq:band."""
    theta, rho, T = 0.07, 0.05, 30 / 365
    for q in (0.02, 0.05, 0.3):
        lower = 1 - theta * (1 - q) - rho * T * (1 - q) / q
        for ratio in (lower - 1e-3, lower + 1e-3):
            S = _sides(q_mid=[q], q_bid=[q], q_ask=[q], won=[ratio * q], lead_h=[T * 8760])
            resid = F.band_components(S, rho=rho)["resid"][0]
            assert (resid <= 0) == (ratio >= lower)


def test_subcent_pinned_tick():
    S = _sides(q_mid=[0.005, 0.0005], q_bid=[0.0, 0.0], q_ask=[0.01, 0.001], won=[0.0, 0.0], subcent=[True, True])
    B = F.band_components(S)
    assert B["pinned"].to_list() == [False, True]             # 0.1c grid: a 1c ask is not the floor


# ---- probability weighting -------------------------------------------------------------------------
def _synthetic(prob, n_ev=3000, legs=3, seed=11):
    rng = np.random.default_rng(seed)
    q = 1 / (1 + np.exp(-rng.normal(-0.6, 1.6, n_ev * legs)))
    q = np.clip(np.round(q, 3), 0.006, 0.994)
    y = (rng.random(q.size) < prob(q)).astype(float)
    g = np.repeat(np.arange(n_ev), legs)
    return q, y, g


def test_llo_fit_recovers_parameters_and_matches_xs_logit():
    a, b = 0.25, 1.35
    q, y, g = _synthetic(lambda q: F.prob_llo(q, np.array([a, b])))
    w = np.ones_like(q)
    f = F.fit_llo(q, y, g, w)
    assert abs(f["alpha"] - a) < 3 * f["se_alpha"] and abs(f["beta"] - b) < 3 * f["se_beta"]
    assert f["gamma"] == pytest.approx(1 / f["beta"]) and f["delta"] == pytest.approx(-f["alpha"] / f["beta"])
    ref = X.logit_fit(y, X.logit(q), g, w)
    assert f["alpha"] == pytest.approx(ref["a"], abs=1e-6) and f["beta"] == pytest.approx(ref["b"], abs=1e-6)
    assert f["se_beta"] == pytest.approx(ref["se_b"], rel=1e-4)
    # the weighting function evaluated at the implied probability returns the price
    pi = F.prob_llo(np.array([0.03, 0.5, 0.9]), np.array([f["alpha"], f["beta"]]))
    assert np.allclose(F.llo_w(pi, f["gamma"], f["delta"]), [0.03, 0.5, 0.9], atol=1e-9)


def test_prelec_fit_recovers_parameters():
    gamma, delta = 0.7, 1.1
    th = np.array([math.log(delta), math.log(1 / gamma)])
    q, y, g = _synthetic(lambda q: F.prob_prelec(q, th), n_ev=6000)
    f = F.fit_prelec(q, y, g, np.ones_like(q))
    assert f["converged"]
    assert abs(f["gamma"] - gamma) < 3 * f["se_gamma"] and abs(f["delta"] - delta) < 3 * f["se_delta"]
    pi = F.prob_prelec(np.array([0.02, 0.4]), th)
    assert np.allclose(F.prelec_w(pi, gamma, delta), [0.02, 0.4], atol=1e-9)


def test_calibrated_data_give_gamma_one_and_no_oos_gain():
    q, y, g = _synthetic(lambda q: q, n_ev=4000)
    f = F.fit_llo(q, y, g, np.ones_like(q))
    assert abs(f["gamma"] - 1) < 3 * f["se_gamma"] and abs(f["delta"]) < 3 * f["se_delta"]
    p = F.fit_prelec(q, y, g, np.ones_like(q))
    assert abs(p["gamma"] - 1) < 3 * p["se_gamma"]


def test_wald_detects_unequal_slopes():
    rows = []
    for grp, b, seed in (("A", 1.0, 1), ("B", 1.0, 2), ("C", 1.6, 3)):
        q, y, g = _synthetic(lambda q, b=b: F.prob_llo(q, np.array([0.0, b])), n_ev=2500, seed=seed)
        rows.append(pl.DataFrame({"mid": q, "y": y, "event_ticker": [f"{grp}{i}" for i in g], "group": grp,
                                  "w_m": np.ones_like(q)}))
    df = pl.concat(rows)
    eq = F.wald_equal(df.filter(pl.col("group") != "C"), "group", ["A", "B"])
    diff = F.wald_equal(df, "group", ["A", "B", "C"])
    assert eq["p"] > 0.01 and diff["p"] < 1e-4
    assert diff["slopes"][2] == pytest.approx(1.6, abs=0.15)


# ---- helpers ---------------------------------------------------------------------------------------
def test_half_split_and_drop_top_events():
    df = pl.DataFrame({"month": ["2026-01"] * 3 + ["2026-02"] * 2 + ["2026-03"] * 5})
    assert F.half_split(df) == "2026-02"
    x = pl.DataFrame({"event_ticker": ["a", "a", "b", "c"], "w_m": [1.0, 1.0, 1.0, 1.0], "v": [5.0, 5.0, -20.0, 1.0]})
    assert F.drop_top_events(x, "v", k=1)["event_ticker"].to_list() == ["a", "a", "c"]


def test_replace_block():
    txt = "a\n% BEGIN AUTO:x\nold\n% END AUTO:x\nb\n"
    assert F.replace_block(txt, "x", "new 1\nnew 2") == "a\n% BEGIN AUTO:x\nnew 1\nnew 2\n% END AUTO:x\nb\n"
    with pytest.raises(KeyError):
        F.replace_block(txt, "y", "z")


# ---- smoke test on the real panel (skipped where the xs panel is absent) ------------------------------
@pytest.mark.skipif(not (X.out_dir() / "panel.parquet").exists(), reason="xs panel not built here")
def test_band_on_real_panel_is_nonempty_and_adds_up():
    pre = F.load_pre()
    assert pre.height > 1000
    B = F.band_components(F.side_frame(pre))
    ls = B.filter(pl.col("q_mid") <= 0.10)
    assert ls.height > 500
    tot = ls["c_floor"] + ls["c_spread"] + ls["c_fee"] + ls["c_cap"] + ls["resid"]
    assert np.allclose(tot.to_numpy(), ls["gross"].to_numpy())
    assert (ls["c_floor"] >= 0).all() and (ls["c_spread"] >= -1e-12).all() and (ls["c_fee"] >= 0).all()
