import pytest
from albert.execution.kelly import kelly_size


def test_positive_edge_returns_positive_size():
    size = kelly_size(
        edge=0.10,
        ask_price=0.40,
        bankroll=10000.0,
        kelly_fraction=0.25,
        confidence=1.0,
        max_position_usd=500.0,
    )
    assert size > 0

def test_negative_edge_returns_zero():
    size = kelly_size(
        edge=0.01,
        ask_price=0.60,  # b = 0.667, f* = (0.01*0.667 - 0.99)/0.667 < 0
        bankroll=10000.0,
        kelly_fraction=0.25,
        confidence=1.0,
        max_position_usd=500.0,
    )
    assert size == 0.0

def test_capped_by_max_position_usd():
    size = kelly_size(
        edge=0.30,
        ask_price=0.20,
        bankroll=1_000_000.0,
        kelly_fraction=1.0,
        confidence=1.0,
        max_position_usd=100.0,
    )
    assert size == pytest.approx(100.0)

def test_confidence_scales_output():
    full = kelly_size(0.10, 0.40, 10000.0, 0.25, 1.0, 500.0)
    half = kelly_size(0.10, 0.40, 10000.0, 0.25, 0.5, 500.0)
    assert half == pytest.approx(full * 0.5)

def test_invalid_ask_price_returns_zero():
    assert kelly_size(0.10, 0.0, 10000.0, 0.25, 1.0, 500.0) == 0.0
    assert kelly_size(0.10, 1.0, 10000.0, 0.25, 1.0, 500.0) == 0.0
    assert kelly_size(0.10, 1.1, 10000.0, 0.25, 1.0, 500.0) == 0.0

def test_zero_edge_returns_zero():
    assert kelly_size(0.0, 0.40, 10000.0, 0.25, 1.0, 500.0) == 0.0
