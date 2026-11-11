"""Pure-math tests of the reduced-form hazard model (no API payloads)."""

import math

import numpy as np
import polars as pl

from albert.research.fundamental import taker_fee
from albert.research.pm_hazard import (
    apply_k,
    cir_survival,
    fee_vec,
    fit_k,
    forward_hazards,
    hazard_multiplier,
    implied_hazard,
    no_event_price,
    prob_from_hazard,
    realized_hazard,
)


def test_implied_hazard_round_trip():
    p = np.array([0.01, 0.2, 0.5, 0.9])
    tau = np.array([1.0, 7.0, 30.0, 2.0])
    h = implied_hazard(p, tau)
    assert np.allclose(prob_from_hazard(h, tau), p)
    # longer horizon at the same price means a lower average hazard
    assert implied_hazard(0.3, 10.0) < implied_hazard(0.3, 5.0)


def test_bootstrap_piecewise_hazard_reprices_the_strip():
    # piecewise-constant hazards on (0,3], (3,10], (10,30] (per day)
    h_true = np.array([0.02, 0.05, 0.01])
    T = np.array([3.0, 10.0, 30.0])
    H = np.cumsum(h_true * np.diff(np.concatenate([[0.0], T])))
    p = 1 - np.exp(-H)
    fh = forward_hazards(p, T)
    assert np.allclose(fh, h_true)
    # repricing from the bootstrapped curve gives the strip back exactly
    assert np.allclose(1 - np.exp(-np.cumsum(fh * np.diff(np.concatenate([[0.0], T])))), p)


def test_monotonicity_violation_is_a_negative_forward_hazard():
    fh = forward_hazards([0.30, 0.25], [5.0, 10.0])
    assert fh[0] > 0 and fh[1] < 0


def test_theta_and_jump_make_a_martingale():
    # flat hazard: E[p(t+dt)] = P(jump)*1 + P(no jump)*p_noevent = p
    for p, tau, dt in ((0.1, 30.0, 1.0), (0.6, 7.0, 3.0), (0.02, 2.0, 0.5)):
        h = float(implied_hazard(p, tau))
        q = float(prob_from_hazard(h, dt))
        p_ne = float(no_event_price(p, h, dt))
        assert p_ne < p                                   # theta: decays without the event
        assert math.isclose(q * 1 + (1 - q) * p_ne, p, rel_tol=1e-12)
        # with no event the decay speed is dp/dt = -h(1-p) at dt -> 0
        eps = 1e-6
        assert math.isclose((float(no_event_price(p, h, eps)) - p) / eps, -h * (1 - p), rel_tol=1e-4)
        # and the remaining-horizon price is consistent: p_ne = 1 - exp(-h (tau - dt))
        assert math.isclose(p_ne, 1 - math.exp(-h * (tau - dt)), rel_tol=1e-12)


def test_simulated_jump_to_resolution_paths():
    rng = np.random.default_rng(7)
    h, T, dt, n = 0.04, 30.0, 1.0, 200_000
    tev = rng.exponential(1 / h, n)
    times = np.arange(0.0, T, dt)
    p_paths = np.where(tev[:, None] <= times[None, :], 1.0, 1 - np.exp(-h * (T - times[None, :])))
    # martingale: the cross-sectional mean price stays at p_0 along the path
    assert np.allclose(p_paths.mean(axis=0), p_paths[0, 0], atol=0.004)
    # settlement frequency equals the time-0 price
    assert abs((tev <= T).mean() - p_paths[0, 0]) < 0.004
    # correctly priced outcomes give a hazard multiplier near 1; overpriced ones (true hazard half the
    # implied) give k near 0.5
    p0 = np.full(n, p_paths[0, 0])
    assert abs(hazard_multiplier(p0, (tev <= T).astype(float)) - 1) < 0.03
    tev_slow = rng.exponential(1 / (h / 2), n)
    assert abs(hazard_multiplier(p0, (tev_slow <= T).astype(float)) - 0.5) < 0.03


def test_realized_hazard_poisson_mle_with_censoring():
    rng = np.random.default_rng(1)
    h, T = 0.1, 5.0
    tev = rng.exponential(1 / h, 100_000)
    events = tev <= T
    exposure = np.minimum(tev, T)
    assert abs(realized_hazard(events, exposure) - h) < 0.003


def test_cir_survival_limits_and_monte_carlo():
    lam, tau = 0.05, 20.0
    # no volatility, no mean reversion: plain exponential survival
    assert math.isclose(cir_survival(lam, tau, 0.0, 0.0, 0.0), math.exp(-lam * tau))
    # small sigma converges to the deterministic mean-reverting integral
    det = cir_survival(lam, tau, 0.5, 0.02, 0.0)
    assert math.isclose(cir_survival(lam, tau, 0.5, 0.02, 1e-5), det, rel_tol=1e-6)
    # Monte Carlo (full-truncation Euler) agrees with the affine closed form
    rng = np.random.default_rng(3)
    kappa, theta, sigma, n, steps = 0.3, 0.04, 0.15, 40_000, 400
    dt = tau / steps
    x = np.full(n, lam)
    integral = np.zeros(n)
    for _ in range(steps):
        xp = np.maximum(x, 0)
        integral += xp * dt
        x = x + kappa * (theta - xp) * dt + sigma * np.sqrt(xp * dt) * rng.standard_normal(n)
    mc = float(np.exp(-integral).mean())
    cf = cir_survival(lam, tau, kappa, theta, sigma)
    assert abs(mc - cf) < 0.01
    # hazard volatility raises survival (Jensen): stochastic intensity < deterministic default prob
    assert cf > cir_survival(lam, tau, kappa, theta, 0.0)


def test_fee_vec_matches_scalar_fee():
    ps = np.array([0.01, 0.07, 0.33, 0.5, 0.91])
    assert np.allclose(fee_vec(ps, np.ones(5)), [taker_fee(p) for p in ps])
    assert np.allclose(fee_vec(ps, np.full(5, 0.5)), [taker_fee(p, 0.5) for p in ps])


def test_shrunk_multipliers_recover_cell_and_fall_back_to_parent():
    rng = np.random.default_rng(5)
    n = 4000
    kind = rng.choice(["a", "b"], n)
    mid = rng.uniform(0.05, 0.6, n)
    k_true = np.where(kind == "a", 0.5, 1.0)
    y = (rng.uniform(size=n) < 1 - (1 - mid) ** k_true).astype(int)
    df = pl.DataFrame({"kind": kind, "mid": mid, "y": y})
    levels = [["kind"]]
    tab = fit_k(df, levels, n0=50)
    ka, kb = math.exp(tab[("kind",)][("a",)]), math.exp(tab[("kind",)][("b",)])
    assert abs(ka - 0.5) < 0.12 and abs(kb - 1.0) < 0.15
    # an unseen cell falls back to the pooled multiplier
    new = pl.DataFrame({"kind": ["c"], "mid": [0.3], "y": [0]})
    p = apply_k(new, tab, levels)
    assert math.isclose(p[0], 1 - 0.7 ** math.exp(tab[()][()]))
