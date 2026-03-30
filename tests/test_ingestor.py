import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from albert.events import EventBus, MarketDataEvent
from albert.ingestor.kalshi import KalshiIngestor


async def test_kalshi_ingestor_publishes_market_data_event():
    bus = EventBus()
    queue = bus.subscribe("market_data")

    ws_message = json.dumps({
        "type": "orderbook_snapshot",
        "msg": {
            "market_ticker": "BTC-24",
            "yes": {"bid": 45, "ask": 47},
            "no": {"bid": 53, "ask": 55},
            "last_price": 46,
            "volume": 1000,
        }
    })

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)
    mock_ws.send = AsyncMock()

    async def fake_aiter(self):
        yield ws_message
        await asyncio.sleep(999)  # hang to simulate live connection

    mock_ws.__aiter__ = fake_aiter

    with patch.dict("os.environ", {"KALSHI_API_TOKEN": "test"}):
        with patch("albert.ingestor.kalshi.websockets.connect", return_value=mock_ws):
            ingestor = KalshiIngestor(bus=bus, market_ids=["kalshi:BTC-24"])
            task = asyncio.create_task(ingestor.run())
            await asyncio.sleep(0.05)
            task.cancel()

    assert not queue.empty()
    event: MarketDataEvent = queue.get_nowait()
    assert event.market_id == "kalshi:BTC-24"
    assert event.yes_ask == pytest.approx(0.47)
    assert event.no_bid == pytest.approx(0.53)


async def test_kalshi_ingestor_ignores_non_orderbook_messages():
    bus = EventBus()
    queue = bus.subscribe("market_data")

    ws_message = json.dumps({"type": "heartbeat", "msg": {}})

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)
    mock_ws.send = AsyncMock()

    async def fake_aiter(self):
        yield ws_message
        await asyncio.sleep(999)

    mock_ws.__aiter__ = fake_aiter

    with patch.dict("os.environ", {"KALSHI_API_TOKEN": "test"}):
        with patch("albert.ingestor.kalshi.websockets.connect", return_value=mock_ws):
            ingestor = KalshiIngestor(bus=bus, market_ids=["kalshi:BTC-24"])
            task = asyncio.create_task(ingestor.run())
            await asyncio.sleep(0.05)
            task.cancel()

    assert queue.empty()
