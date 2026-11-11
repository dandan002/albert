from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from albert.fees import kalshi_fee
from albert.research import vol_theta as V

T0 = datetime(2026, 9, 1, 12, tzinfo=UTC)


@pytest.mark.parametrize(("price", "n", "maker", "fee_type"), [
    (0.95, 5, False, "quadratic"), (0.81, 1, False, "quadratic"), (0.5, 10, False, "quadratic"),
    (0.9, 5, True, "quadratic"), (0.9, 5, True, "quadratic_with_maker_fees"), (0.03, 166, False, "quadratic")])
def test_fee_per_contract_matches_conservative_kalshi_fee(price, n, maker, fee_type):
    got = V.fee_per_contract(np.array([price]), np.array([n]), maker=maker, fee_type=np.array([fee_type]),
                             mult=np.array([1.0]))[0]
    assert got == pytest.approx(kalshi_fee(n, price, maker=maker, fee_type=fee_type, conservative=True) / n)


def test_order_size_respects_event_cap():
    assert V.order_size(np.array([0.95, 0.5, 6.0]))[:2].tolist() == [5.0, 10.0]
    assert V.order_size(np.array([6.0]))[0] == 1.0      # never below one contract


def test_no_bid_is_floored_and_post_only():
    # fair 7c, margin 0: floor(93c) = 93c, but the YES bid is 8c -> NO ask 92c -> cap 91c
    assert V.no_bid_price(np.array([0.07]), np.array([0.08]), 0.0)[0] == pytest.approx(0.91)
    # wide market: floor(1 - 0.071 - 0.02) = 90c, cap 98c
    assert V.no_bid_price(np.array([0.071]), np.array([0.01]), 0.02)[0] == pytest.approx(0.90)
    assert np.isnan(V.no_bid_price(np.array([0.995]), np.array([0.5]), 0.0)[0])


def test_hour_steps_apply_post_delay_and_keep_running_max():
    e = T0
    pr = pl.DataFrame({"ticker": ["A"] * 5,
                       "ts": [e + timedelta(seconds=30), e + timedelta(seconds=61), e + timedelta(minutes=10),
                              e + timedelta(minutes=20), e + timedelta(hours=1)],
                       "px": [0.50, 0.10, 0.08, 0.12, 0.30]})
    steps, hmax = V.hour_steps(pr)
    st = steps.filter(pl.col("e") == e)
    assert st["px"].to_list() == [0.10, 0.12, 0.30]      # 0.50 is inside the post delay; 0.08 not a new max
    assert hmax.filter(pl.col("e") == e)["hmax"][0] == pytest.approx(0.30)


def test_first_through_needs_a_strictly_higher_print():
    steps = pl.DataFrame({"ticker": ["A", "A"], "e": [T0, T0], "ts": [T0 + timedelta(minutes=5), T0 + timedelta(minutes=9)],
                          "px": [0.10, 0.11]})
    orders = pl.DataFrame({"_oid": [0, 1, 2], "ticker": ["A", "A", "A"], "e": [T0, T0, T0], "level": [0.10, 0.09, 0.11]})
    got = V.first_through(orders, steps, ["ticker", "e"]).sort("_oid")
    assert got["filled"].to_list() == [True, True, False]
    assert got["fill_ts"][0] == T0 + timedelta(minutes=9)    # a print AT our level does not fill
    assert got["fill_ts"][1] == T0 + timedelta(minutes=5)


def test_theta_path_and_short_forward_value():
    assert V.theta_path(np.array([0.2]), np.array([3.0]), np.array([3.0]))[0] == pytest.approx(0.2)
    assert V.theta_path(np.array([0.2]), np.array([3.0]), np.array([0.0]))[0] == pytest.approx(0.0)
    assert V.theta_path(np.array([0.2]), np.array([3.0]), np.array([1.5]))[0] < 0.2
    # k = 1 gives the market's own forward probability m2 - m1
    v = V.short_forward_value(np.array([0.15]), np.array([0.40]), np.array([1.0]), np.array([1.0]))[0]
    assert v == pytest.approx(1 - 0.25)
    # a forward hazard multiplier below 1 makes the short forward worth more
    assert V.short_forward_value(np.array([0.15]), np.array([0.40]), np.array([1.0]), np.array([0.7]))[0] > v


def test_train_contracts_are_known_before_the_fold():
    u = pl.DataFrame({"ticker": ["a", "b", "c", "d"],
                      "T": [datetime(2026, 5, 20, tzinfo=UTC), datetime(2026, 5, 31, tzinfo=UTC),
                            datetime(2026, 5, 10, tzinfo=UTC), datetime(2026, 7, 1, tzinfo=UTC)],
                      "settlement_ts": [datetime(2026, 5, 21, tzinfo=UTC), datetime(2026, 5, 31, 12, tzinfo=UTC),
                                        datetime(2026, 6, 3, tzinfo=UTC), datetime(2026, 5, 1, tzinfo=UTC)]})
    # b: deadline inside the 2-day buffer; c: settled after the fold start; d: early YES with a later deadline
    assert V.train_contracts(u, "2026-06").to_list() == ["a"]


def _cand(t, ev, p, payoff, *, filled=True, settle_h=48, maker=False):
    return {"t": t, "t_end": t + timedelta(hours=1), "filled": filled, "t_fill": t if filled else None,
            "settle": t + timedelta(hours=settle_h), "event_ticker": ev, "p1": p, "ft1": "quadratic", "mu1": 1.0,
            "maker": maker, "payoff": payoff, "priority": 0.0, "fold": "2026-09"}


def test_simulation_enforces_event_and_total_caps():
    c = pl.DataFrame([_cand(T0, "E1", 0.9, 1.0), _cand(T0 + timedelta(minutes=1), "E1", 0.9, 1.0)]
                     + [_cand(T0 + timedelta(minutes=2 + i), f"E{i + 2}", 0.9, 0.0) for i in range(5)])
    r = V.simulate(c)
    # E1: 5 contracts at 90c = $4.50 + taker fee ceil(0.07*5*0.09*100)/100 = 4c -> $4.54; the second E1 order has
    # no room; then $15.46 left for E2..E6: E2-E4 take 5 each ($4.54), E5 fits 2 ($1.80 + 2c of the $1.84 left),
    # E6 none
    assert r["trades"] == 5
    assert r["contracts"] == 5 + 5 + 5 + 5 + 2
    assert r["skipped_cap"] == 2
    win = 5 * 1.0 - (4.5 + kalshi_fee(5, 0.9))
    loss = 3 * (4.5 + kalshi_fee(5, 0.9)) + (1.8 + kalshi_fee(2, 0.9))
    assert r["pnl_usd"] == pytest.approx(round(win - loss, 2))


def test_simulation_releases_resting_orders_and_settled_positions():
    unfilled = _cand(T0, "E1", 0.9, 1.0, filled=False, maker=True)
    later = _cand(T0 + timedelta(hours=2), "E1", 0.9, 1.0, maker=True)
    r = V.simulate(pl.DataFrame([unfilled, later]))
    assert r["trades"] == 1 and r["contracts"] == 5        # the unfilled order's reservation was released
