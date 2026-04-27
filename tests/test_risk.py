import asyncio
import time
import pytest
from datetime import date, datetime, timezone
from albert.db import get_connection, migrate
from albert.events import EventBus, OrderIntent, StrategyHaltedEvent
from albert.execution.risk import RiskChecker


def make_intent(market_id: str = "kalshi:X", strategy_id: str = "s1") -> OrderIntent:
    return OrderIntent(
        market_id=market_id,
        strategy_id=strategy_id,
        side="yes",
        edge=0.10,
        confidence=1.0,
    )


def make_checker(conn, overrides: dict = {}, bus: EventBus | None = None) -> RiskChecker:
    config = {
        "max_total_notional_usd": 1000.0,
        "daily_loss_limit_usd": -200.0,
        "order_debounce_seconds": 5,
        **overrides,
    }
    return RiskChecker(conn, config, bus)


@pytest.mark.asyncio
async def test_allows_normal_order():
    conn = get_connection(":memory:")
    migrate(conn)
    checker = make_checker(conn)
    assert await checker.check(make_intent(), position_size_usd=50.0) is True


@pytest.mark.asyncio
async def test_blocks_on_debounce():
    conn = get_connection(":memory:")
    migrate(conn)
    checker = make_checker(conn, {"order_debounce_seconds": 60})
    intent = make_intent()
    assert await checker.check(intent, 50.0) is True
    assert await checker.check(intent, 50.0) is False  # second call within debounce window


@pytest.mark.asyncio
async def test_allows_after_debounce_expires():
    conn = get_connection(":memory:")
    migrate(conn)
    checker = make_checker(conn, {"order_debounce_seconds": 0})
    intent = make_intent()
    assert await checker.check(intent, 50.0) is True
    assert await checker.check(intent, 50.0) is True  # debounce=0 means always allowed


@pytest.mark.asyncio
async def test_blocks_when_daily_loss_limit_hit():
    conn = get_connection(":memory:")
    migrate(conn)
    today = date.today().isoformat()
    conn.execute(
        "INSERT INTO daily_pnl (date, strategy_id, realized_pnl, unrealized_pnl) VALUES (?, ?, ?, ?)",
        (today, "s1", -250.0, 0.0)
    )
    conn.commit()
    checker = make_checker(conn, {"daily_loss_limit_usd": -200.0})
    assert await checker.check(make_intent(), 50.0) is False


@pytest.mark.asyncio
async def test_blocks_when_max_notional_exceeded():
    conn = get_connection(":memory:")
    migrate(conn)
    conn.execute(
        "INSERT INTO positions (market_id, strategy_id, side, contracts, avg_entry_price, current_price, unrealized_pnl, opened_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("kalshi:Y", "s1", "yes", 10.0, 0.50, 0.50, 0.0, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    # current notional = 10 * 0.50 = 5.0 USD
    checker = make_checker(conn, {"max_total_notional_usd": 10.0})
    assert await checker.check(make_intent(), position_size_usd=6.0) is False
    assert await checker.check(make_intent(), position_size_usd=4.0) is True


@pytest.mark.asyncio
async def test_circuit_breaker_publishes_halted_event():
    conn = get_connection(":memory:")
    migrate(conn)
    today = date.today().isoformat()
    conn.execute(
        "INSERT INTO daily_pnl (date, strategy_id, realized_pnl, unrealized_pnl) VALUES (?, ?, ?, ?)",
        (today, "s1", -250.0, 0.0)
    )
    conn.commit()

    bus = EventBus()
    halted_queue = bus.subscribe("strategy_halted")

    checker = make_checker(conn, {
        "daily_loss_limit_usd": -200.0,
        "circuit_breaker_violations": 2,
        "order_debounce_seconds": 0,
    }, bus)

    intent = make_intent(strategy_id="s1")

    # First violation — no event yet
    assert await checker.check(intent, 10.0) is False

    # Second violation — circuit breaker triggers
    assert await checker.check(intent, 10.0) is False

    event = await asyncio.wait_for(halted_queue.get(), timeout=1.0)
    assert isinstance(event, StrategyHaltedEvent)
    assert event.strategy_id == "s1"
    assert "circuit_breaker" in event.reason
