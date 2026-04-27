import asyncio
import sqlite3
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import date, datetime

import pytest

from albert.events import EventBus, MarketDataEvent, OrderIntent, FillEvent, StrategyHaltedEvent
from albert.execution.engine import ExecutionEngine
from albert.execution.risk import RiskChecker
from albert.portfolio.tracker import PortfolioTracker
from albert.strategies.engine import StrategyEngine
from albert.db import migrate, get_connection

@pytest.mark.asyncio
async def test_full_pipeline_ingest_to_execution(tmp_path):
    """
    Test the full pipeline:
    1. Market data arrives on EventBus
    2. StrategyEngine picks it up, runs strategy
    3. Strategy emits OrderIntent
    4. ExecutionEngine picks up intent, validates risk
    5. ExecutionEngine calls adapter to place order
    """
    db_path = tmp_path / "test_pipeline.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    migrate(conn)

    # Setup a mock strategy in the DB
    conn.execute(
        "INSERT INTO strategies (strategy_id, name, class_path, config, enabled) VALUES (?, ?, ?, ?, ?)",
        ("test_strat", "Test Momentum", "albert.strategies.examples.momentum.MomentumV1", '{"min_edge": 0.01}', 1)
    )
    conn.commit()

    bus = EventBus()
    shutdown_event = asyncio.Event()

    # Mock adapter
    now_ts = datetime.now()
    fill_event = FillEvent(
        fill_id="test-fill",
        market_id="kalshi:TEST-TICKER",
        strategy_id="test_strat",
        side="yes",
        contracts=10.0,
        fill_price=0.40,
        fee=0.01,
        filled_at=now_ts
    )
    mock_adapter = MagicMock()
    mock_adapter.place_order = AsyncMock(return_value=fill_event)
    mock_adapter.get_bankroll = AsyncMock(return_value=1000.0)
    adapters = {"kalshi": mock_adapter}

    global_config = {
        "max_total_notional_usd": 10000,
        "daily_loss_limit_usd": -500,
        "order_debounce_seconds": 0, # disable debounce for test
    }

    strategy_engine = StrategyEngine(bus, conn, reload_interval=100, shutdown_event=shutdown_event)
    execution_engine = ExecutionEngine(bus, conn, adapters, global_config, shutdown_event)

    # Start engines
    strat_task = asyncio.create_task(strategy_engine.run())
    exec_task = asyncio.create_task(execution_engine.run())

    # Wait for strategy to load
    await asyncio.sleep(0.1)

    # 1. Publish MarketDataEvent that triggers MomentumV1
    # MomentumV1 buys YES if yes_ask < 0.5 - min_edge
    # 0.4 < 0.5 - 0.01 = 0.49 -> SHOULD TRIGGER
    event = MarketDataEvent(
        exchange="kalshi",
        market_id="kalshi:TEST-TICKER",
        yes_bid=0.38,
        yes_ask=0.40,
        no_bid=0.58,
        no_ask=0.60,
        last_price=0.39,
        volume=1000.0,
        timestamp=datetime.now()
    )
    
    await bus.publish("market_data", event)

    # Wait for processing
    # The pipeline is: market_data -> StrategyEngine -> order_intents -> ExecutionEngine -> Adapter
    # We need to wait enough for both engines to process their queues
    for _ in range(20):
        if mock_adapter.place_order.called:
            break
        await asyncio.sleep(0.1)

    # Verify adapter was called
    assert mock_adapter.place_order.called
    args, _ = mock_adapter.place_order.call_args
    intent = args[0]
    assert isinstance(intent, OrderIntent)
    assert intent.market_id == "kalshi:TEST-TICKER"
    assert intent.side == "yes"
    assert intent.strategy_id == "test_strat"

    # Cleanup
    shutdown_event.set()
    # Unblock the handle_market_data queue
    await bus.publish("market_data", event)
    # Unblock the handle_orders queue
    dummy_intent = OrderIntent(
        market_id="dummy",
        strategy_id="dummy",
        side="yes",
        edge=0.0,
        confidence=0.0
    )
    await bus.publish("order_intents", dummy_intent)
    
    # Force cancel if they don't stop
    try:
        await asyncio.wait_for(asyncio.gather(strat_task, exec_task), timeout=1.0)
    except asyncio.TimeoutError:
        strat_task.cancel()
        exec_task.cancel()
    conn.close()


@pytest.mark.asyncio
async def test_graceful_shutdown_cancels_all_tasks_within_5_seconds():
    """
    Verify that all engine tasks exit cleanly within 5 seconds when shutdown_event is set.
    """
    conn = get_connection(":memory:")
    migrate(conn)

    conn.execute(
        "INSERT INTO strategies (strategy_id, name, class_path, config, enabled) VALUES (?, ?, ?, ?, ?)",
        ("s1", "Test", "albert.strategies.examples.momentum.MomentumV1", '{"min_edge": 0.01}', 1)
    )
    conn.commit()

    bus = EventBus()
    shutdown_event = asyncio.Event()

    mock_adapter = MagicMock()
    mock_adapter.place_order = AsyncMock(return_value=FillEvent(
        fill_id="f1",
        market_id="kalshi:X",
        strategy_id="s1",
        side="yes",
        contracts=5.0,
        fill_price=0.47,
        fee=0.01,
        filled_at=datetime.now(),
    ))
    mock_adapter.get_bankroll = AsyncMock(return_value=1000.0)
    adapters = {"kalshi": mock_adapter}

    global_config = {
        "max_total_notional_usd": 10000,
        "daily_loss_limit_usd": -500,
        "order_debounce_seconds": 0,
    }

    strategy_engine = StrategyEngine(bus, conn, reload_interval=100, shutdown_event=shutdown_event)
    execution_engine = ExecutionEngine(bus, conn, adapters, global_config, shutdown_event)
    portfolio_tracker = PortfolioTracker(bus, conn, shutdown_event=shutdown_event)

    tasks = [
        asyncio.create_task(strategy_engine.run()),
        asyncio.create_task(execution_engine.run()),
        asyncio.create_task(portfolio_tracker.run()),
    ]

    # Let tasks start up
    await asyncio.sleep(0.1)

    # Trigger shutdown
    shutdown_event.set()

    # Publish dummy events to unblock any queue.get() calls
    await bus.publish("market_data", MarketDataEvent(
        exchange="kalshi",
        market_id="kalshi:X",
        yes_bid=0.38,
        yes_ask=0.40,
        no_bid=0.58,
        no_ask=0.60,
        last_price=0.39,
        volume=1000.0,
        timestamp=datetime.now(),
    ))
    await bus.publish("order_intents", OrderIntent(
        market_id="kalshi:X",
        strategy_id="s1",
        side="yes",
        edge=0.0,
        confidence=0.0,
    ))
    await bus.publish("fills", FillEvent(
        fill_id="f1",
        market_id="kalshi:X",
        strategy_id="s1",
        side="yes",
        contracts=5.0,
        fill_price=0.47,
        fee=0.01,
        filled_at=datetime.now(),
    ))

    # All tasks should complete within 5 seconds
    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)

    for task in tasks:
        assert task.done()

    conn.close()


@pytest.mark.asyncio
async def test_circuit_breaker_publishes_strategy_halted_event():
    """
    Verify that RiskChecker publishes StrategyHaltedEvent on the EventBus
    when daily loss limit violations exceed the circuit breaker threshold.
    """
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

    global_config = {
        "daily_loss_limit_usd": -200.0,
        "circuit_breaker_violations": 2,
        "order_debounce_seconds": 0,
    }

    checker = RiskChecker(conn, global_config, bus)

    intent = OrderIntent(
        market_id="kalshi:X",
        strategy_id="s1",
        side="yes",
        edge=0.10,
        confidence=1.0,
    )

    # First call: violation count = 1, no event yet
    assert await checker.check(intent, position_size_usd=10.0) is False

    # Second call: violation count = 2, circuit breaker triggers
    assert await checker.check(intent, position_size_usd=10.0) is False

    event = await asyncio.wait_for(halted_queue.get(), timeout=1.0)
    assert isinstance(event, StrategyHaltedEvent)
    assert event.strategy_id == "s1"
    assert "circuit_breaker" in event.reason

    conn.close()
