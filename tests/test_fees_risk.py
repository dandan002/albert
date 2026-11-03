import pytest

from albert.fees import kalshi_fee, pmus_fee
from albert.risk import RiskConfig, kelly_fraction, size_order


def test_kalshi_taker_fee_matches_schedule_example():
    # 100 contracts at 50c: 0.07 * 100 * 0.25 = $1.75
    assert kalshi_fee(100, 0.5) == pytest.approx(1.75)


def test_kalshi_conservative_rounding_is_up_to_cent():
    # 1 contract at 50c: 0.0175 -> $0.02 conservative, ~$0.0175 exact
    assert kalshi_fee(1, 0.5) == 0.02
    assert kalshi_fee(1, 0.5, conservative=False) == pytest.approx(0.0175)


def test_kalshi_maker_fee_only_on_maker_series():
    assert kalshi_fee(100, 0.5, maker=True, fee_type="quadratic") == 0
    assert kalshi_fee(100, 0.5, maker=True, fee_type="quadratic_with_maker_fees") == pytest.approx(0.44)


def test_kalshi_fee_multiplier():
    assert kalshi_fee(100, 0.5, multiplier=0.5) == pytest.approx(0.88)
    assert kalshi_fee(100, 0.5, multiplier=0.0) == 0


def test_pmus_fee_matches_docs_example():
    # docs.polymarket.us/fees: 100 contracts at $0.50 -> $1.74
    assert pmus_fee(100, 0.5) == pytest.approx(1.74)
    assert pmus_fee(100, 0.5, maker=True) == 0


def test_pmus_banker_rounding_small_order_can_be_zero():
    # 1 contract at 0.95: 0.0695 * 0.0475 = 0.0033 -> $0.00
    assert pmus_fee(1, 0.95) == 0.0


def test_kelly_fraction():
    assert kelly_fraction(0.6, 0.5) == pytest.approx(0.2)
    assert kelly_fraction(0.4, 0.5) == 0


def test_size_order_rejects_small_edges_and_caps_stake():
    cfg = RiskConfig(kelly_fraction=0.25, max_bet_dollars=5, min_edge=0.03)
    low = size_order(venue="kalshi", prob=0.51, price=0.5, available=None, bankroll=50,
                     event_exposure=0, cfg=cfg)
    assert low.contracts == 0 and low.reason == "edge_below_threshold"
    big = size_order(venue="kalshi", prob=0.9, price=0.5, available=None, bankroll=1000,
                     event_exposure=0, cfg=cfg)
    assert 0 < big.contracts * 0.5 <= 5


def test_size_order_respects_displayed_size_and_event_cap():
    cfg = RiskConfig(max_event_fraction=0.1)
    s = size_order(venue="pmus", prob=0.8, price=0.5, available=2, bankroll=100,
                   event_exposure=0, cfg=cfg)
    assert s.contracts <= 2
    full = size_order(venue="pmus", prob=0.8, price=0.5, available=None, bankroll=100,
                      event_exposure=10, cfg=cfg)
    assert full.contracts == 0
