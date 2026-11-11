"""Tests for the ladder greeks study: digital/leg greeks against finite differences, strangle leg
selection, the hedge and lot bookkeeping (a hedged position's PnL equals the sum of its ledger
lots and its cash flows; the attribution adds up), the capped book and the width model."""

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from scipy.special import ndtr

from albert.research import vol_greeks as vg

INF = math.inf


def test_digital_greeks_match_finite_differences():
    F, s, K = 70.0, 3.0, 72.5
    D, delta, gamma, vega = vg.digital_greeks(F, s, K)
    h = 1e-4
    def f(F_: float, s_: float) -> float:
        return float(ndtr((F_ - K) / s_))

    assert float(D) == pytest.approx(f(F, s))
    assert float(delta) == pytest.approx((f(F + h, s) - f(F - h, s)) / (2 * h), rel=1e-6)
    assert float(gamma) == pytest.approx((f(F + h, s) - 2 * f(F, s) + f(F - h, s)) / h**2, rel=1e-3)
    assert float(vega) == pytest.approx((f(F, s + h) - f(F, s - h)) / (2 * h), rel=1e-6)
    # out of the money (K > F) is long gamma and vega, in the money short
    assert gamma > 0 and vega > 0
    _, _, g2, v2 = vg.digital_greeks(F, s, 67.5)
    assert g2 < 0 and v2 < 0
    # infinite strikes are constants
    D, d, g, v = vg.digital_greeks(F, s, np.array([-INF, INF]))
    assert D.tolist() == [1.0, 0.0] and d.tolist() == [0.0, 0.0] and g.tolist() == [0.0, 0.0] and v.tolist() == [0.0, 0.0]


def test_leg_greeks_bracket_set_sums():
    edges = [-INF, 66.5, 68.5, 70.5, 72.5, INF]
    lo, hi = np.array(edges[:-1]), np.array(edges[1:])
    v, d, g, ve = vg.leg_greeks(69.3, 2.5, lo, hi)
    assert v.sum() == pytest.approx(1.0)
    assert d.sum() == pytest.approx(0.0, abs=1e-12)
    assert g.sum() == pytest.approx(0.0, abs=1e-12)
    assert ve.sum() == pytest.approx(0.0, abs=1e-12)
    # "below" leg (-inf, K] is 1 - D(K)
    vb, db, _, _ = vg.leg_greeks(69.3, 2.5, -INF, 66.5)
    D, dd, _, _ = vg.digital_greeks(69.3, 2.5, 66.5)
    assert float(vb) == pytest.approx(1 - float(D)) and float(db) == pytest.approx(-float(dd))


def make_path(F_path, sigma=1.0, strikes=(-2.0, -1.0, 0.0, 1.0, 2.0), y_x=0.3, half_spread=0.01, vol=100.0,
              split="dev") -> vg.LadderPath:
    """Threshold ladder of "Over K" legs quoted at Phi((F_t - K)/sigma) +- half_spread, settled at X = y_x."""
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    T = len(F_path)
    hours = [t0 + timedelta(hours=i) for i in range(T)]
    K = np.array(strikes)
    fair = ndtr((np.array(F_path)[:, None] - K[None, :]) / sigma)
    bid = np.clip(np.round(fair - half_spread, 2), 0.01, 0.98)
    ask = np.clip(np.round(fair + half_spread, 2), 0.02, 0.99)
    n = len(K)
    return vg.LadderPath("L1", "EV1", "sp_total", "KXTEST", split, "threshold", hours[-1], [f"T{k}" for k in strikes],
                         K, np.full(n, INF), (y_x > K).astype(int), np.ones(n), ["quadratic"] * n,
                         [hours[-1] + timedelta(hours=2)] * n, hours, bid, ask, np.full((T, n), vol),
                         np.array(F_path, float), np.full(T, sigma))


def test_strangle_selection_threshold_ladder():
    p = make_path([0.0, 0.0])
    core = vg.strangle(p, 0, (0.5, 1.25))
    # lower tail: YES on Over -1 (contains F), upper tail: NO on Over 1
    assert core == [(1, "yes"), (3, "no")]
    long = vg.strangle(p, 0, (0.5, 1.25), short=False)
    assert long == [(1, "no"), (3, "yes")]
    assert vg.tail_strike(0.0, 0.5, 1.5, "bracket") == (0.5, "no")
    assert vg.tail_strike(0.0, -0.5, 0.5, "bracket") is None


@pytest.mark.parametrize("integer", [False, True])
def test_hedged_position_ledger_equals_lots_and_cash(integer):
    F = [0.0, 0.4, 0.9, 1.3, 0.8, 0.2, -0.3, 0.1]
    p = make_path(F, y_x=0.3)
    units = 10.0 if integer else 1.0
    pos = vg.Position(p, [(1, "yes"), (3, "no")], units=units, integer=integer)
    assert pos.enter(0)
    pos.hedge(0)
    for i in range(1, len(F)):
        pos.hedge(i, 0.0)
    pos.settle()
    hedge_lots = [x for x in pos.lots if x.role == "hedge"]
    assert hedge_lots, "the moving median must trigger hedge trades"
    assert any(x.i_out is not None for x in hedge_lots), "an over-hedged lot must be sold back"
    lots_pnl = sum(x.value() - x.cost() for x in pos.lots)
    assert pos.pnl() == pytest.approx(lots_pnl)
    assert pos.cash == pytest.approx(lots_pnl)
    att = pos.attribution()
    assert sum(att.values()) == pytest.approx(pos.pnl())
    if integer:
        assert all(float(x.q).is_integer() for x in pos.lots)


def test_hedge_neutralises_delta_fractional():
    p = make_path([0.0, 0.8])
    pos = vg.Position(p, [(1, "yes"), (3, "no")])
    assert pos.enter(0)
    assert abs(pos.net_delta(1)) > 0.1
    pos.hedge(1)
    assert pos.net_delta(1) == pytest.approx(0.0, abs=1e-9)
    # a threshold above the imbalance trades nothing
    p2 = make_path([0.0, 0.05])
    pos2 = vg.Position(p2, [(1, "yes"), (3, "no")])
    pos2.enter(0)
    assert pos2.hedge(1, thr=0.25) == 0


def test_partial_sale_splits_lot_and_fee():
    p = make_path([0.0, 0.0, 0.0])
    pos = vg.Position(p, [(2, "yes")], units=10.0, integer=True)
    assert pos.enter(0)
    lot = pos.lots[0]
    fee0 = lot.fee
    assert pos.sell(1, lot, 4)
    assert len(pos.lots) == 2
    assert sum(x.q for x in pos.lots) == 10
    assert sum(x.fee for x in pos.lots) == pytest.approx(fee0)
    pos.settle()
    assert pos.cash == pytest.approx(pos.pnl())


def test_volume_cap_in_integer_mode():
    p = make_path([0.0, 0.0], vol=8.0)            # 25% of 8 = 2 contracts per order
    pos = vg.Position(p, [(2, "yes")], units=5.0, integer=True)
    assert not pos.enter(0)                       # all or nothing
    pos = vg.Position(p, [(2, "yes")], units=2.0, integer=True)
    assert pos.enter(0)


def test_book_caps_and_ledger():
    paths = [make_path([0.0, 0.3, 0.6, 0.2], y_x=0.3), make_path([0.0, -0.5, -0.2, 0.1], y_x=-2.5)]
    paths[1] = vg.LadderPath(**{**paths[1].__dict__, "ladder": "L2", "event": "EV2"})
    entries = [vg.Entry(p, 0, vg.strangle(p, 0, (0.5, 1.25)), 3, sid=p.ladder) for p in paths]
    bk = vg.Book(event_cap=5.0, open_cap=20.0)
    led = bk.run(entries, vg.RULES["thr0.1"], "s1_test", "v")
    assert led.height >= 4
    assert set(led.columns) >= {"struct_id", "event_ticker", "ticker", "side", "n", "t_in", "cost_usd", "settle_ts", "payoff_usd"}
    for ev, g in led.group_by("event_ticker"):
        core = g.filter(pl.col("role") == "core")
        assert float(core["cost_usd"].sum()) <= 5.0 * (1 - vg.HEDGE_RESERVE) + 1e-9
    assert led["payoff_usd"].null_count() == 0
    assert (led["settle_ts"] >= led["t_in"]).all()


def test_rule_hours():
    assert vg.Rule("every", k=6).hours(2, 20) == [8, 14, 20]
    assert vg.Rule("thr", thr=0.1).hours(2, 5) == [3, 4, 5]
    assert vg.Rule("entry").hours(2, 5) == [] and vg.RULES["thr0.25"].thr == 0.25


def test_width_fit_recovers_lambda():
    rng = np.random.default_rng(3)
    n = 3000
    fam = rng.choice(["NFL", "MLB"], n)
    lead = rng.choice([6, 24], n)
    lam = np.where(fam == "NFL", 1.2, 0.9)
    x = rng.normal(0, lam)
    zl, zh = np.floor(x * 4) / 4, np.floor(x * 4) / 4 + 0.25     # interval-censored on a 0.25 grid
    sn = pl.DataFrame({"family": fam, "lead_h": lead, "lsig": rng.normal(1.0, 0.1, n), "z_lo": zl, "z_hi": zh,
                       "split": ["dev"] * n})
    fit = vg.fit_width(sn)
    r = vg.width_ratio(sn, fit)
    assert np.median(r[fam == "NFL"]) == pytest.approx(1.2, abs=0.06)
    assert np.median(r[fam == "MLB"]) == pytest.approx(0.9, abs=0.06)
