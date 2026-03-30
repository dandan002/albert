import asyncio
import pytest
from datetime import datetime
from albert.events import EventBus, MarketDataEvent, OrderIntent, FillEvent, StrategyHaltedEvent

def make_market_event():
    return MarketDataEvent(
        market_id="kalshi:TEST-24",
        exchange="kalshi",
        timestamp=datetime.utcnow(),
        yes_bid=0.45,
        yes_ask=0.47,
        no_bid=0.53,
        no_ask=0.55,
        last_price=0.46,
        volume=1000.0,
    )

async def test_publish_subscribe_delivers_event():
    bus = EventBus()
    q = bus.subscribe("market_data")
    event = make_market_event()
    await bus.publish("market_data", event)
    received = q.get_nowait()
    assert received.market_id == "kalshi:TEST-24"
    assert received.yes_ask == 0.47

async def test_multiple_subscribers_each_receive_event():
    bus = EventBus()
    q1 = bus.subscribe("market_data")
    q2 = bus.subscribe("market_data")
    await bus.publish("market_data", make_market_event())
    assert not q1.empty()
    assert not q2.empty()

async def test_publish_to_unsubscribed_channel_does_not_raise():
    bus = EventBus()
    await bus.publish("order_intents", OrderIntent(
        market_id="kalshi:TEST-24",
        strategy_id="s1",
        side="yes",
        edge=0.08,
        confidence=0.9,
    ))

def test_fill_event_fields():
    fill = FillEvent(
        fill_id="f1",
        market_id="kalshi:TEST-24",
        strategy_id="s1",
        side="yes",
        contracts=10.0,
        fill_price=0.47,
        fee=0.01,
        filled_at=datetime.utcnow(),
    )
    assert fill.contracts == 10.0
    assert fill.fill_price == 0.47
