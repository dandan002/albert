import asyncio
import sqlite3
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

import pytest

from albert.events import EventBus, MarketDataEvent, OrderIntent, FillEvent
from albert.execution.engine import ExecutionEngine
from albert.strategies.engine import StrategyEngine
from albert.db import migrate

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
