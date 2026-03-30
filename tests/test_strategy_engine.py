import asyncio
import json
import pytest
from datetime import datetime
from albert.db import get_connection, migrate
from albert.events import EventBus, MarketDataEvent, OrderIntent
from albert.strategies.engine import StrategyEngine


def make_db():
    conn = get_connection(":memory:")
    migrate(conn)
    conn.execute(
        "INSERT INTO strategies (strategy_id, name, class_path, config, enabled) VALUES (?, ?, ?, ?, ?)",
        ("m1", "Momentum", "albert.strategies.examples.momentum.MomentumV1", json.dumps({"min_edge": 0.05}), 1)
    )
    conn.commit()
    return conn


def make_event(yes_ask: float) -> MarketDataEvent:
    return MarketDataEvent(
        market_id="kalshi:X",
        exchange="kalshi",
        timestamp=datetime.utcnow(),
        yes_bid=yes_ask - 0.02,
        yes_ask=yes_ask,
        no_bid=0.0,
        no_ask=0.0,
        last_price=yes_ask,
        volume=0.0,
    )


@pytest.mark.asyncio
async def test_engine_publishes_intent_for_active_strategy():
    conn = make_db()
    bus = EventBus()
    intents_queue = bus.subscribe("order_intents")
    engine = StrategyEngine(bus, conn, reload_interval=999)

    await bus.publish("market_data", make_event(0.30))

    engine_task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.05)
    engine_task.cancel()

    assert not intents_queue.empty()
    intent: OrderIntent = intents_queue.get_nowait()
    assert intent.strategy_id == "m1"
    assert intent.side == "yes"


@pytest.mark.asyncio
async def test_engine_skips_disabled_strategy():
    conn = make_db()
    conn.execute("UPDATE strategies SET enabled = 0 WHERE strategy_id = 'm1'")
    conn.commit()

    bus = EventBus()
    intents_queue = bus.subscribe("order_intents")
    engine = StrategyEngine(bus, conn, reload_interval=999)

    await bus.publish("market_data", make_event(0.30))
    engine_task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.05)
    engine_task.cancel()

    assert intents_queue.empty()


@pytest.mark.asyncio
async def test_engine_hot_reloads_config():
    conn = make_db()
    bus = EventBus()
    intents_queue = bus.subscribe("order_intents")
    engine = StrategyEngine(bus, conn, reload_interval=0.01)

    engine_task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.02)

    # Update config to require very high edge — strategy should no longer emit
    conn.execute("UPDATE strategies SET config = ? WHERE strategy_id = 'm1'", (json.dumps({"min_edge": 0.99}),))
    conn.commit()
    await asyncio.sleep(0.05)

    # drain existing intents
    while not intents_queue.empty():
        intents_queue.get_nowait()

    await bus.publish("market_data", make_event(0.30))
    await asyncio.sleep(0.05)
    engine_task.cancel()

    assert intents_queue.empty()
