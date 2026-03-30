# tests/test_execution_engine.py
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
from albert.db import get_connection, migrate
from albert.events import EventBus, MarketDataEvent, OrderIntent, FillEvent, StrategyHaltedEvent
from albert.execution.engine import ExecutionEngine
from albert.execution.adapters.base import ExchangeAdapter


def make_db_with_strategy(config: dict = None):
    conn = get_connection(":memory:")
    migrate(conn)
    cfg = config or {"kelly_fraction": 0.25, "max_position_usd": 500.0}
    conn.execute(
        "INSERT INTO strategies (strategy_id, name, class_path, config, enabled) VALUES (?, ?, ?, ?, ?)",
        ("s1", "Test", "albert.strategies.examples.momentum.MomentumV1", json.dumps(cfg), 1)
    )
    conn.execute(
        "INSERT INTO orderbook_snapshots (market_id, timestamp, yes_bid, yes_ask, no_bid, no_ask, last_price, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("kalshi:X", datetime.now(timezone.utc).isoformat(), 0.45, 0.47, 0.53, 0.55, 0.46, 1000.0)
    )
    conn.commit()
    return conn


def make_mock_adapter(bankroll: float = 10000.0) -> ExchangeAdapter:
    adapter = MagicMock(spec=ExchangeAdapter)
    adapter.get_bankroll = AsyncMock(return_value=bankroll)
    adapter.place_order = AsyncMock(return_value=FillEvent(
        fill_id="f1",
        market_id="kalshi:X",
        strategy_id="s1",
        side="yes",
        contracts=5.0,
        fill_price=0.47,
        fee=0.01,
        filled_at=datetime.now(timezone.utc),
    ))
    return adapter


async def test_execution_engine_places_order_and_publishes_fill():
    conn = make_db_with_strategy()
    bus = EventBus()
    fills_queue = bus.subscribe("fills")
    adapter = make_mock_adapter()

    engine = ExecutionEngine(
        bus=bus,
        conn=conn,
        adapters={"kalshi": adapter},
        global_config={"max_total_notional_usd": 100000, "daily_loss_limit_usd": -10000, "order_debounce_seconds": 0},
    )
    engine._price_cache["kalshi:X"] = (0.47, 0.55)

    intent = OrderIntent(market_id="kalshi:X", strategy_id="s1", side="yes", edge=0.10, confidence=1.0)
    await bus.publish("order_intents", intent)

    engine_task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.05)
    engine_task.cancel()

    assert not fills_queue.empty()
    fill: FillEvent = fills_queue.get_nowait()
    assert fill.strategy_id == "s1"
    adapter.place_order.assert_awaited_once()


async def test_execution_engine_persists_fill_to_db():
    conn = make_db_with_strategy()
    bus = EventBus()
    adapter = make_mock_adapter()

    engine = ExecutionEngine(
        bus=bus,
        conn=conn,
        adapters={"kalshi": adapter},
        global_config={"max_total_notional_usd": 100000, "daily_loss_limit_usd": -10000, "order_debounce_seconds": 0},
    )
    engine._price_cache["kalshi:X"] = (0.47, 0.55)

    intent = OrderIntent(market_id="kalshi:X", strategy_id="s1", side="yes", edge=0.10, confidence=1.0)
    await bus.publish("order_intents", intent)

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.05)
    task.cancel()

    fill_row = conn.execute("SELECT * FROM fills WHERE fill_id = 'f1'").fetchone()
    assert fill_row is not None
    assert fill_row["strategy_id"] == "s1"


async def test_execution_engine_skips_unknown_exchange():
    conn = make_db_with_strategy()
    bus = EventBus()
    fills_queue = bus.subscribe("fills")
    adapter = make_mock_adapter()

    engine = ExecutionEngine(
        bus=bus,
        conn=conn,
        adapters={"kalshi": adapter},
        global_config={"max_total_notional_usd": 100000, "daily_loss_limit_usd": -10000, "order_debounce_seconds": 0},
    )

    # polymarket intent but no polymarket adapter
    intent = OrderIntent(market_id="polymarket:X:Y", strategy_id="s1", side="yes", edge=0.10, confidence=1.0)
    await bus.publish("order_intents", intent)

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.05)
    task.cancel()

    assert fills_queue.empty()
    adapter.place_order.assert_not_awaited()
