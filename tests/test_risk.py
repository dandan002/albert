import time
import pytest
from datetime import date, datetime, timezone
from albert.db import get_connection, migrate
from albert.events import OrderIntent
from albert.execution.risk import RiskChecker


def make_intent(market_id: str = "kalshi:X", strategy_id: str = "s1") -> OrderIntent:
    return OrderIntent(
        market_id=market_id,
        strategy_id=strategy_id,
        side="yes",
        edge=0.10,
        confidence=1.0,
    )


def make_checker(conn, overrides: dict = {}) -> RiskChecker:
    config = {
        "max_total_notional_usd": 1000.0,
        "daily_loss_limit_usd": -200.0,
        "order_debounce_seconds": 5,
        **overrides,
    }
    return RiskChecker(conn, config)


def test_allows_normal_order():
    conn = get_connection(":memory:")
    migrate(conn)
    checker = make_checker(conn)
    assert checker.check(make_intent(), position_size_usd=50.0) is True


def test_blocks_on_debounce():
    conn = get_connection(":memory:")
    migrate(conn)
    checker = make_checker(conn, {"order_debounce_seconds": 60})
    intent = make_intent()
    assert checker.check(intent, 50.0) is True
    assert checker.check(intent, 50.0) is False  # second call within debounce window


def test_allows_after_debounce_expires():
    conn = get_connection(":memory:")
    migrate(conn)
    checker = make_checker(conn, {"order_debounce_seconds": 0})
    intent = make_intent()
    assert checker.check(intent, 50.0) is True
    assert checker.check(intent, 50.0) is True  # debounce=0 means always allowed


def test_blocks_when_daily_loss_limit_hit():
    conn = get_connection(":memory:")
    migrate(conn)
    today = date.today().isoformat()
    conn.execute(
        "INSERT INTO daily_pnl (date, strategy_id, realized_pnl, unrealized_pnl) VALUES (?, ?, ?, ?)",
        (today, "s1", -250.0, 0.0)
    )
    conn.commit()
    checker = make_checker(conn, {"daily_loss_limit_usd": -200.0})
    assert checker.check(make_intent(), 50.0) is False


def test_blocks_when_max_notional_exceeded():
    conn = get_connection(":memory:")
    migrate(conn)
    conn.execute(
        "INSERT INTO positions (market_id, strategy_id, side, contracts, avg_entry_price, current_price, unrealized_pnl, opened_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("kalshi:Y", "s1", "yes", 10.0, 0.50, 0.50, 0.0, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    # current notional = 10 * 0.50 = 5.0 USD
    checker = make_checker(conn, {"max_total_notional_usd": 10.0})
    assert checker.check(make_intent(), position_size_usd=6.0) is False
    assert checker.check(make_intent(), position_size_usd=4.0) is True
