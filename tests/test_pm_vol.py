"""Pure-math checks of the pm_vol model layer on simulated paths (no API payloads)."""

import math

import numpy as np
import pytest
from scipy import integrate, special, stats

from albert.research import pm_vol as pv


def _gh_expect(f, mu, sd):
    """E f(N(mu, sd^2)) by adaptive quadrature (reference values)."""
    return integrate.quad(lambda x: f(x) * stats.norm.pdf(x, mu, sd), mu - 12 * sd, mu + 12 * sd,
                          points=[0.0], limit=400, epsabs=1e-12)[0]


def test_fee_rounding_is_exact():
    assert pv.fee(0.5, pv.TAKER) == pytest.approx(0.0175)
    assert pv.fee(0.5, pv.MAKER) == pytest.approx(0.0044)          # 0.004375 rounds up
    assert pv.fee(0.01, pv.TAKER) == pytest.approx(0.0007)         # 0.000693 rounds up
    assert pv.fee(0.5, pv.TAKER, 0.5) == pytest.approx(0.0088)     # 0.00875 rounds up


@pytest.mark.parametrize("a,b", [(1.0, 1.0), (0.7, 0.6), (3.0, 1.7), (2.0, 1.0 + 1e-7)])
def test_clock_power_matches_integral(a, b):
    tau, h = 10.0, 1.0
    want = integrate.quad(lambda s: a * s ** (-b), tau - h, tau)[0]
    assert pv.clock_power(np.array([tau]), h, a, b)[0] == pytest.approx(want, rel=1e-6)


def test_uniform_clock_is_log_ratio():
    tau = np.array([2.0, 5.0, 100.0])
    assert np.allclose(pv.clock_power(tau, 1.0, 1.0, 1.0), np.log(tau / (tau - 1.0)))


def test_bucket_clock_integrates_piecewise():
    cp = {"clock": {"form": "buckets", "edges": [0.0, 2.0, 4.0, 1e9], "lam": [0.5, 0.2, 0.01], "hour_mult": None}}
    # step [1.5, 2.5]: 0.5 * 0.5 + 0.2 * 0.5
    assert pv.clock_L(cp, np.array([2.5]), 1.0)[0] == pytest.approx(0.35)


@pytest.mark.parametrize("p", [0.03, 0.2, 0.5, 0.85])
@pytest.mark.parametrize("L", [1e-3, 0.1, 1.0, 4.0])
def test_m0_transition_is_a_martingale_with_closed_form_variance(p, L):
    z = special.ndtri(p)
    mu, var = pv.m0_moments(np.array([z]), np.array([L]))
    sd = math.sqrt(var[0])
    mean_p = _gh_expect(special.ndtr, mu[0], sd)
    assert mean_p == pytest.approx(p, abs=1e-6)
    second = _gh_expect(lambda x: special.ndtr(x) ** 2, mu[0], sd)
    rho = pv.info_fraction(np.array([L]))
    assert pv.m0_var_p(np.array([p]), rho)[0] == pytest.approx(second - p * p, abs=1e-6)


def test_m0_variance_limits_and_monotonicity():
    p = np.full(5, 0.3)
    rho = np.array([0.0, 0.1, 0.5, 0.9, 1 - 1e-12])
    v = pv.m0_var_p(p, rho)
    assert v[0] == pytest.approx(0.0, abs=1e-12)
    assert v[-1] == pytest.approx(0.3 * 0.7, abs=1e-5)
    assert np.all(np.diff(v) > 0)


def test_chapman_kolmogorov_composition():
    """Two steps with clock increments L1, L2 equal one step with L1 + L2 (mean and variance)."""
    z, L1, L2 = 0.7, 0.3, 0.45
    m1, v1 = pv.m0_moments(np.array([z]), np.array([L1]))
    mean2 = m1[0] * math.exp(L2 / 2)
    var2 = v1[0] * math.exp(L2) + math.expm1(L2)
    m, v = pv.m0_moments(np.array([z]), np.array([L1 + L2]))
    assert mean2 == pytest.approx(m[0])
    assert var2 == pytest.approx(v[0])


def test_time_change_keeps_martingale():
    p, rho = 0.27, np.array([0.2])
    for alpha2 in (0.25, 1.0, 3.0):
        r = pv.mixture_rho(rho, alpha2)[0]
        z = special.ndtri(p)
        mean_p = _gh_expect(special.ndtr, z / math.sqrt(1 - r), math.sqrt(r / (1 - r)))
        assert mean_p == pytest.approx(p, abs=1e-6)
    assert pv.mixture_rho(rho, 1.0)[0] == pytest.approx(0.2)


def _simulate_m0(n_paths=4000, T=48, seed=0):
    """Bachelier digital with a uniform clock, observed hourly: Y Brownian (unit rate),
    p_t = Phi(Y_t / sqrt(T - t)), outcome 1{Y_T > 0}."""
    rng = np.random.default_rng(seed)
    y0 = rng.normal(0, 3.0, n_paths)
    dw = rng.normal(0, 1.0, (n_paths, T))
    Y = np.concatenate([y0[:, None], y0[:, None] + np.cumsum(dw, axis=1)], axis=1)
    t = np.arange(T + 1)
    with np.errstate(divide="ignore"):
        p = special.ndtr(Y[:, :-1] / np.sqrt(T - t[:-1]))
    return p, (Y[:, -1] > 0).astype(float), T


def test_u_statistic_is_standard_normal_on_simulated_m0():
    p, _, T = _simulate_m0()
    tau = T - np.arange(T)
    z0 = special.ndtri(np.clip(p[:, :-2], 1e-15, 1 - 1e-15))
    z1 = special.ndtri(np.clip(p[:, 1:-1], 1e-15, 1 - 1e-15))
    L = pv.clock_power(np.broadcast_to(tau[:-2], z0.shape), 1.0, 1.0, 1.0)
    u = pv.u_stat(z0, z1, L).ravel()
    assert u.var() == pytest.approx(1.0, abs=0.02)
    assert abs(u.mean()) < 0.01
    assert abs(stats.kurtosis(u)) < 0.1


def test_variance_identity_on_simulated_m0():
    p, y, _ = _simulate_m0(n_paths=6000, seed=1)
    p0 = p[:, 0]
    total = (np.diff(p, axis=1) ** 2).sum(axis=1) + (y - p[:, -1]) ** 2
    se = total.std() / math.sqrt(len(total))
    assert abs(total.mean() - (p0 * (1 - p0)).mean()) < 4 * se
    # orthogonal increments: the outcome surprise alone has the same expectation
    assert abs(((y - p0) ** 2).mean() - (p0 * (1 - p0)).mean()) < 4 * ((y - p0) ** 2).std() / math.sqrt(len(y))


def test_fit_clock_recovers_power_law():
    rng = np.random.default_rng(5)
    n = 60_000
    tau = rng.uniform(5.0, 300.0, n)       # keep p1 inside the 0.5c probit clip
    a_true, b_true = 0.6, 1.4
    L = pv.clock_power(tau, 1.0, a_true, b_true)
    p0 = rng.uniform(0.2, 0.8, n)
    z0 = special.ndtri(p0)
    mu, var = pv.m0_moments(z0, L)
    p1 = special.ndtr(mu + np.sqrt(var) * rng.normal(size=n))
    inc = {"p0": p0, "p1": p1, "spr": np.full(n, 1e-4), "tau": tau, "hour": np.zeros(n, dtype=int)}
    fit = pv.fit_clock(inc, "power", np.arange(n))
    assert fit["clock"]["a"] == pytest.approx(a_true, rel=0.1)
    assert fit["clock"]["b"] == pytest.approx(b_true, abs=0.05)
    ev = pv.clock_eval(inc, fit, np.arange(n))
    assert ev["u"].var() == pytest.approx(1.0, abs=0.03)
    d = pv.drift_test(ev["z0"], ev["z1"], ev["L"], np.arange(n) // 50)
    assert abs(d["slope"] - 1.0) < 4 * d["se"]          # the probit drift that makes p a martingale


def test_em_scale_mixture_recovers_parameters():
    rng = np.random.default_rng(2)
    n = 200_000
    jump = rng.random(n) < 0.05
    u = rng.normal(0, 1.0, n) * np.where(jump, math.sqrt(1.0 + 15.0), 1.0)
    em = pv.em_scale_mixture(u)
    assert em["pi"] == pytest.approx(0.05, abs=0.01)
    assert em["s_d2"] == pytest.approx(1.0, abs=0.05)
    assert em["s_j2"] == pytest.approx(15.0, rel=0.15)
    assert em["jump_share"] == pytest.approx(0.05 * 15 / (1 + 0.05 * 15), abs=0.03)


def test_sv_acf_fit_recovers_decay():
    lags = np.arange(1, 25)
    acf = 0.1 * 0.95 ** lags
    out = pv.sv_from_acf(lags, acf)
    assert out["phi"] == pytest.approx(0.95, abs=1e-3)
    assert out["half_life_h"] == pytest.approx(math.log(0.5) / math.log(0.95), rel=0.01)
    assert out["nu_per_sqrt_h"] > 0


def test_bvn_cdf_matches_scipy():
    for h, k, r in ((0.3, -0.7, 0.5), (-1.2, -0.4, 0.9), (1.5, 2.0, 0.1), (0.0, 0.0, 0.3), (-0.5, 0.8, -0.6)):
        want = stats.multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([h, k])
        assert pv.bvn_cdf(np.array([h]), np.array([k]), np.array([r]))[0] == pytest.approx(want, abs=1e-6)


@pytest.mark.parametrize("p", [0.04, 0.4, 0.5, 0.9])
@pytest.mark.parametrize("rho", [1e-4, 0.05, 0.6, 0.98])
def test_abs_move_closed_form(p, rho):
    z = special.ndtri(p)
    want = _gh_expect(lambda x: np.abs(special.ndtr(x) - p), z / math.sqrt(1 - rho), math.sqrt(rho / (1 - rho)))
    assert pv.m0_abs_dp(np.array([p]), np.array([rho]))[0] == pytest.approx(want, abs=1e-7, rel=1e-5)


@pytest.mark.parametrize("jumps", [False, True])
def test_transition_density_is_a_proper_martingale_density(jumps):
    params = {"categories": {"Mentions": {
        "clock": {"form": "power", "a": 0.8, "b": 1.2, "hour_mult": None}, "scale": 1.0,
        "jump": {"pi": 0.1, "mult_d": 0.5, "mult_j": 5.5}}}}
    tr = pv.transition(0.3, tau=12.0, h=3.0, category="Mentions", alpha_state=1.3, jumps=jumps, params=params)
    xs = np.linspace(1e-6, 1 - 1e-6, 400_001)
    dens = tr.pdf(xs)
    assert integrate.trapezoid(dens, xs) == pytest.approx(1.0, abs=2e-3)
    assert integrate.trapezoid(xs * dens, xs) == pytest.approx(0.3, abs=2e-3)
    assert tr.mean == 0.3
    q = tr.quantile(0.8)
    assert float(tr.cdf(q)) == pytest.approx(0.8, abs=1e-8)
    # variance in Z-space (the p-density has integrable spikes at 0 and 1 when the Z-sd exceeds 1)
    var = sum(w * _gh_expect(lambda x: (special.ndtr(x) - 0.3) ** 2, m, sd)
              for w, (m, sd) in zip(tr.weights, tr.z_moments()))
    assert var == pytest.approx(tr.var, rel=1e-6)
    e_abs = sum(w * _gh_expect(lambda x: np.abs(special.ndtr(x) - 0.3), m, sd)
                for w, (m, sd) in zip(tr.weights, tr.z_moments()))
    assert e_abs == pytest.approx(tr.e_abs, rel=1e-5)


def test_transition_uniform_default_without_params():
    tr = pv.transition(0.5, tau=10.0, h=1.0, category="Politics", params={})
    rho = 1 - 9.0 / 10.0                      # uniform clock: V_{t+h} / V_t = (tau - h) / tau
    assert tr.rho[0] == pytest.approx(rho)
    assert tr.var == pytest.approx(pv.m0_var_p(np.array([0.5]), np.array([rho]))[0])
    with pytest.raises(ValueError):
        pv.transition(0.5, tau=1.0, h=1.0, category="Politics", params={})


def test_window_helpers_match_brute_force():
    tk = np.array([0, 0, 0, 0, 1, 1, 1])
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0])
    gs, ge = pv.group_bounds(tk)
    x0 = np.nan_to_num(x)
    assert pv.trailing_sum(x0, gs, 2).tolist() == [0.0, 1.0, 3.0, 2.0, 0.0, 5.0, 5.0]
    fs, full = pv.forward_sum(x0, ge, 2)
    assert fs.tolist() == [3.0, 2.0, 4.0, 4.0, 5.0, 7.0, 7.0]
    assert full.tolist() == [True, True, True, False, True, True, False]
    e = pv.ewma_past(x, tk, (1,))[1]
    d = 0.5
    assert np.isnan(e[0]) and e[1] == pytest.approx(1.0)
    assert e[2] == pytest.approx((2 + d * 1) / (1 + d))
    assert e[3] == pytest.approx((2 + d * 1) / (1 + d))          # NaN observation is skipped
    assert e[6] == pytest.approx(5.0) and np.isnan(e[4])


def _ewma_loop(x, tk, hl):
    d = math.exp(-math.log(2) / hl)
    out = np.full(len(x), np.nan)
    num = den = 0.0
    prev = None
    for i in range(len(x)):
        if tk[i] != prev:
            num = den = 0.0
            prev = tk[i]
        if den > 0:
            out[i] = num / den
        if not np.isnan(x[i]):
            num, den = x[i] + d * num, 1 + d * den
    return out


def test_vectorised_ewma_matches_loop():
    rng = np.random.default_rng(9)
    tk = np.sort(rng.integers(0, 40, 3000))
    x = rng.exponential(1.0, 3000)
    x[rng.random(3000) < 0.3] = np.nan
    got = pv.ewma_past(x, tk, (3, 48))
    for hl in (3, 48):
        want = _ewma_loop(x, tk, hl)
        assert np.allclose(got[hl], want, equal_nan=True, rtol=1e-5)


def test_round_quote_grid_and_edges():
    b, a = pv.round_quote(np.array([0.503, 0.004, 0.996, 0.5]), np.array([0.02, 0.01, 0.03, 0.01]))
    assert np.allclose(b, [0.49, 0.01, 0.96, 0.50]) and np.allclose(a, [0.51, 0.02, 0.99, 0.51])


def test_m0_paths_vectorised_matches_recursion():
    rng = np.random.default_rng(4)
    tk = np.repeat([0, 1, 2], [5, 1, 7])
    gs, _ = pv.group_bounds(tk)
    L = rng.uniform(0.01, 0.8, len(tk))
    eps = rng.standard_normal(len(tk))
    z0 = np.repeat([0.3, -1.0, 1.4], [5, 1, 7])
    got = pv.m0_paths(z0, L, eps, gs)
    want = np.empty(len(tk))
    for i in range(len(tk)):
        want[i] = z0[i] if gs[i] == i else want[i - 1] * math.exp(L[i - 1] / 2) + math.sqrt(math.expm1(L[i - 1])) * eps[i - 1]
    assert np.allclose(got, want)


def test_drift_slope_is_one_without_ticks_and_rounding_is_measurable():
    rng = np.random.default_rng(8)
    n = 200_000
    p0 = rng.uniform(0.1, 0.9, n)
    L = rng.uniform(0.02, 0.3, n)
    z0 = special.ndtri(p0)
    mu, var = pv.m0_moments(z0, L)
    z1 = mu + np.sqrt(var) * rng.standard_normal(n)
    g = np.arange(n) // 100
    rows = pv.drift_by_z(z0, z1, L, g)
    allrow = rows[-1]
    assert abs(allrow["slope"] - 1) < 4 * allrow["se"]
    b0, a0 = pv.round_quote(p0, np.full(n, 0.02))
    b1, a1 = pv.round_quote(special.ndtr(z1), np.full(n, 0.02))
    rq = pv.drift_by_z(pv.probit((b0 + a0) / 2), pv.probit((b1 + a1) / 2), L, g)[-1]
    assert np.isfinite(rq["slope"])


def test_hf_sample_is_as_of():
    path = {"ns": np.array([10, 25, 40], dtype=np.int64) * 10**9, "bid": np.array([0.40, 0.41, 0.45]),
            "ask": np.array([0.42, 0.43, 0.46]), "lo": 10 * 10**9, "stop": 50 * 10**9, "T": 3600 * 10**9}
    g = pv.hf_sample(path, 10)
    assert np.allclose(g["mid"], [0.41, 0.41, 0.42, 0.455, 0.455])
    assert np.allclose(g["tau"], [3590, 3580, 3570, 3560, 3550])


def test_ols_slope_recovers_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    y = 0.7 * x + rng.normal(scale=0.3, size=5000)
    b, se = pv._ols_slope(x, y, np.arange(5000) // 10)
    assert abs(b - 0.7) < 4 * se
