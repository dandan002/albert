"""Pure-math tests for the cross-strike smile study: leg edges, implied-vol inversion and smile
fits on simulated strips, the interval-censored width MLE and the fee arithmetic. Sub-title
strings are copied from the ingested Kalshi markets table; every strip is simulated."""

import math
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
from scipy.special import ndtr
from scipy.stats import t as student_t

from albert.research.fundamental import taker_fee
from albert.research.pm_smile import (
    build_snapshots,
    fit_jf_implied,
    implied,
    jf_cdf,
    jf_quantile_shape,
    ladder_edges,
    parse_leg,
    pit_table,
    realized_interval,
    strike_points,
    taker_fee_v,
    variance_between,
    width_mle,
)

INF = math.inf


def test_parse_leg_forms():
    assert parse_leg("63° to 64°", "wx_high") == ("between", 63.0, 64.0, "")
    assert parse_leg("62° or below", "wx_high")[:2] == ("le", 62.0)
    assert parse_leg("71° or above", "wx_low")[:2] == ("ge", 71.0)
    assert parse_leg("Butler wins by over 10.5 points", "sp_spread") == ("spread", 10.5, 10.5, "Butler")
    assert parse_leg("UMass wins 1H by over 7.5 points", "sp_spread")[3] == "UMass"
    assert parse_leg("Over 21.5 points", "sp_total")[:2] == ("gt", 21.5)
    assert parse_leg("Northwestern over 9.5 points", "sp_teamtotal") == ("team_gt", 9.5, 9.5, "Northwestern")
    assert parse_leg("Above 38.7%", "politics")[:2] == ("gt", 38.7)
    assert parse_leg("38.1%", "politics")[:3] == ("eq", 38.1, 38.1)
    assert parse_leg("Above 4.25M", "entertainment")[1] == pytest.approx(4.25e6)
    assert parse_leg("Above 100K", "entertainment")[1] == pytest.approx(1e5)
    assert parse_leg("Above 156T tokens", "scitech")[1] == 156.0
    assert parse_leg("At least 9", "politics")[:2] == ("ge", 9.0)
    assert parse_leg("Magnitude of at least 5.2", "climate_other")[:2] == ("ge", 5.2)
    assert parse_leg("10-14", "politics")[:3] == ("between", 10.0, 14.0)
    assert parse_leg("<5", "politics")[:2] == ("lt", 5.0)
    assert parse_leg("40+", "politics")[:2] == ("ge", 40.0)
    assert parse_leg("Maura Sullivan, ≥12%", "elections")[:2] == ("ge", 12.0)
    assert parse_leg("Sherrod Brown", "politics") is None
    assert parse_leg("9", "entertainment") is None


def test_ladder_edges_brackets_and_spreads():
    wx = [("le", 62.0, 62.0, ""), ("between", 63.0, 64.0, ""), ("between", 65.0, 66.0, ""), ("ge", 67.0, 67.0, "")]
    edges, kind = ladder_edges(wx)
    assert kind == "bracket"
    assert edges == [(-INF, 62.5), (62.5, 64.5), (64.5, 66.5), (66.5, INF)]
    sp = [("spread", 2.5, 2.5, "A"), ("spread", 6.5, 6.5, "A"), ("spread", 2.5, 2.5, "B"), ("spread", 6.5, 6.5, "B")]
    edges, kind = ladder_edges(sp)
    assert kind == "threshold"
    assert edges == [(2.5, INF), (6.5, INF), (-INF, -2.5), (-INF, -6.5)]
    rt, _ = ladder_edges([("gt", 45.0, 45.0, ""), ("gt", 50.0, 50.0, ""), ("gt", 55.0, 55.0, "")])
    assert rt[0] == (45.5, INF)          # integer score: "above 45" is X >= 46


def test_realized_interval():
    lo = np.array([-INF, 62.5, 64.5, 66.5])
    hi = np.array([62.5, 64.5, 66.5, INF])
    assert realized_interval(lo, hi, np.array([0, 0, 1, 0]))[:2] == (64.5, 66.5)
    lo = np.array([21.5, 24.5, 27.5])
    hi = np.array([INF, INF, INF])
    L, U, ok = realized_interval(lo, hi, np.array([1, 1, 0]))
    assert (L, U, ok) == (24.5, 27.5, True)
    assert not realized_interval(lo, hi, np.array([0, 1, 0]))[2]      # non-monotone settlement


def _normal_strip(F, sig, K):
    return ndtr((F - K) / sig)


def test_implied_inverts_a_normal_strip():
    K = np.arange(30.5, 70.0, 3.0)
    S = _normal_strip(47.3, 6.0, K)
    fit = implied(K, S, np.ones_like(K))
    assert fit["sigma"] == pytest.approx(6.0, rel=1e-6)
    assert fit["F"] == pytest.approx(47.3, abs=1e-6)
    v = fit["v"][np.isfinite(fit["v"])]
    assert np.allclose(v, 1.0, atol=1e-6)
    assert fit["s"] == pytest.approx(0.0, abs=1e-6) and fit["c"] == pytest.approx(0.0, abs=1e-6)


def test_fat_tails_give_positive_curvature_and_skew_sign():
    K = np.linspace(-4, 4, 17) + 0.25
    S = student_t.sf(K, df=3)
    fit = implied(K, S, np.ones_like(K))
    assert fit["c"] > 0.02
    assert abs(fit["s"]) < 0.02                        # symmetric distribution, no skew
    # right-skewed underlying (log-normal): upper-wing vols exceed lower-wing vols
    K = np.linspace(0.3, 3.0, 19)
    S = 1 - ndtr(np.log(K) / 0.5)
    fit = implied(K, S, np.ones_like(K))
    assert fit["s"] > 0.05


def test_median_outside_strikes_returns_none():
    K = np.array([1.0, 2.0, 3.0])
    assert implied(K, np.array([0.9, 0.8, 0.7]), np.ones(3)) is None


def test_strike_points_bracket_set_removes_overround():
    edges = np.array([-INF, 62.5, 64.5, 66.5, 68.5, INF])
    lo, hi = edges[:-1], edges[1:]
    p = np.diff(-ndtr((65.2 - edges) / 2.0))           # bracket probabilities, sum 1
    bid, ask = p * 1.06 - 0.005, p * 1.06 + 0.005      # 6% overround
    K, S, _ = strike_points("bracket", lo, hi, bid, ask, np.ones(5, bool))
    assert np.allclose(K, [62.5, 64.5, 66.5, 68.5])
    assert np.allclose(S, ndtr((65.2 - K) / 2.0), atol=2e-3)
    fit = implied(K, S, np.ones(4))
    assert fit["F"] == pytest.approx(65.2, abs=0.05) and fit["sigma"] == pytest.approx(2.0, rel=0.05)
    assert strike_points("bracket", lo, hi, bid, ask, np.array([1, 1, 0, 1, 1], bool)) is None


def test_strike_points_threshold_mixed_directions():
    lo = np.array([2.5, 6.5, -INF, -INF])
    hi = np.array([INF, INF, -2.5, -6.5])
    mid = np.array([0.6, 0.3, 0.2, 0.05])
    K, S, _ = strike_points("threshold", lo, hi, mid - 0.01, mid + 0.01, np.ones(4, bool))
    assert np.allclose(K, [-6.5, -2.5, 2.5, 6.5])
    assert np.allclose(S, [0.95, 0.8, 0.6, 0.3])


def test_build_snapshots_orients_spreads_and_records_realized():
    t = datetime(2026, 10, 10, 12, tzinfo=UTC)
    K = np.array([2.5, 6.5, 10.5])
    # favourite is team B (margin X = A - B centred at -5)
    pA = ndtr((-5 - K) / 8)           # P(X > k)
    pB = ndtr((5 - K) / 8)            # P(X < -k)
    lo = np.r_[K, [-INF] * 3]
    hi = np.r_[[INF] * 3, -K]
    mid = np.r_[pA, pB]
    q = pl.DataFrame({"ladder": "E", "event_ticker": "E", "series_ticker": "KXSPREAD", "category": "Sports",
                      "group": "sp_spread", "kind": "threshold", "t_ref": t, "exp_time": t, "t_snap": t, "lead_h": 6,
                      "L": -7.5, "U": -6.5, "n_legs": 6, "ticker": [f"T{i}" for i in range(6)], "lo": lo, "hi": hi,
                      "bid": mid - 0.005, "ask": mid + 0.005, "y": [0, 0, 0, 1, 1, 0], "q_ts": t})
    sn, pt, lg, _ = build_snapshots(q, q)
    assert sn.height == 1
    r = sn.row(0, named=True)
    assert r["orient"] == -1 and r["F"] == pytest.approx(5.0, abs=0.05)
    assert (r["x_lo"], r["x_hi"]) == (6.5, 7.5)
    assert r["sigma_atm"] == pytest.approx(8.0, rel=0.03)
    # realized indicator at each oriented strike: X = 7 (favourite by 7)
    y = dict(zip(pt["K"].to_list(), pt["y_gt"].to_list()))
    assert y[2.5] == 1.0 and y[6.5] == 1.0 and y[10.5] == 0.0 and y[-2.5] == 1.0
    # the underdog's "wins by over 2.5" leg (X > 2.5) becomes X' <= -2.5 after the flip; its own
    # YES contract and settlement are unchanged
    t0 = lg.filter(pl.col("ticker") == "T0").row(0, named=True)
    assert (t0["lo"], t0["hi"], t0["y"]) == (-INF, -2.5, 0)
    assert t0["m_hi"] == pytest.approx((-2.5 - r["F"]) / r["sigma_atm"])


def test_width_mle_recovers_scale():
    rng = np.random.default_rng(0)
    x = rng.normal(0.2, 1.3, 4000)
    edges = np.arange(-6, 6.5, 0.5)
    idx = np.searchsorted(edges, x)
    zl = np.where(idx > 0, edges[np.clip(idx - 1, 0, len(edges) - 1)], -INF)
    zh = np.where(idx < len(edges), edges[np.clip(idx, 0, len(edges) - 1)], INF)
    r = width_mle(zl, zh, np.arange(4000))
    assert r["lambda"] == pytest.approx(1.3, abs=0.05)
    assert r["mu"] == pytest.approx(0.2, abs=0.06)
    assert r["t_lambda_eq_1"] > 5


def test_fractional_pit_is_uniform_under_calibration():
    rng = np.random.default_rng(1)
    n = 3000
    u = rng.uniform(size=n)
    cuts = np.linspace(0, 1, 7)        # implied CDF at six strikes: realized interval = the cell of u
    j = np.searchsorted(cuts, u) - 1
    sn = pl.DataFrame({"group": "g", "lead_h": 6, "split": "dev", "event_ticker": [str(i) for i in range(n)],
                       "pit_lo": cuts[j], "pit_hi": cuts[j + 1]})
    row = pit_table(sn).row(0, named=True)
    assert row["outer_share"] == pytest.approx(0.2, abs=0.02)
    assert all(abs(row[f"pit{k}"] - 0.1) < 0.02 for k in range(10))


def test_vectorised_fee_matches_harness():
    p = np.array([0.01, 0.05, 0.37, 0.5, 0.93, 0.99])
    for mult in (1.0, 0.5):
        assert np.allclose(taker_fee_v(p, np.full(len(p), mult)), [taker_fee(x, mult) for x in p])


def test_jf_skew_t_reduces_to_student_t_and_fit_recovers_shape():
    x = np.linspace(-6, 6, 25)
    assert np.allclose(jf_cdf(x, 2.5, 2.5), student_t.cdf(x, 5), atol=1e-10)
    assert jf_cdf(np.array([-INF, INF]), 2.0, 3.0).tolist() == [0.0, 1.0]
    # right-skewed strip: heavier right tail (b < a) gives positive Bowley skewness
    assert jf_quantile_shape(6.0, 2.0)["bowley"] > 0.05
    m = np.linspace(-3, 3, 41)
    S = 1 - jf_cdf((m - 0.1) / 0.9, 4.0, 2.0)
    th = fit_jf_implied(m, S)
    _, s, a, b = (float(th[0]), math.exp(th[1]), math.exp(th[2]), math.exp(th[3]))
    assert s == pytest.approx(0.9, rel=0.05) and a == pytest.approx(4.0, rel=0.1) and b == pytest.approx(2.0, rel=0.1)


def test_variance_between_martingale_ratio_near_one():
    rng = np.random.default_rng(3)
    rows = []
    for e in range(400):
        F0, s0, s1 = 50.0, 4.0, 3.0
        F1 = F0 + rng.normal(0, math.sqrt(s0**2 - s1**2))
        for lead, F, sig in ((24, F0, s0), (6, F1, s1)):
            rows.append({"ladder": f"L{e}", "event_ticker": f"E{e}", "group": "g", "lead_h": lead, "F": F,
                         "sigma_atm": sig, "s": 0.0, "c": 0.0, "orient": 1, "snap_id": len(rows)})
    out = variance_between(pl.DataFrame(rows), boot=50).row(0, named=True)
    assert out["iv_mean"] == pytest.approx(1 - (3 / 4) ** 2)
    assert out["rv_over_iv"] == pytest.approx(1.0, abs=0.15)
    assert abs(out["t_drift"]) < 3
