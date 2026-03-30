import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from albert.events import EventBus, MarketDataEvent
from albert.ingestor.kalshi import KalshiIngestor


def make_private_key():
    key = MagicMock()
    key.sign.return_value = b"sig"
    return key


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

    env = {
        "KALSHI_API_KEY_ID": "test_key_id",
        "KALSHI_PRIVATE_KEY": "test_private_key",
    }
    with patch.dict("os.environ", env):
        with patch("albert.ingestor.kalshi._load_private_key", return_value=make_private_key()):
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

    env = {
        "KALSHI_API_KEY_ID": "test_key_id",
        "KALSHI_PRIVATE_KEY": "test_private_key",
    }
    with patch.dict("os.environ", env):
        with patch("albert.ingestor.kalshi._load_private_key", return_value=make_private_key()):
            with patch("albert.ingestor.kalshi.websockets.connect", return_value=mock_ws):
                ingestor = KalshiIngestor(bus=bus, market_ids=["kalshi:BTC-24"])
                task = asyncio.create_task(ingestor.run())
                await asyncio.sleep(0.05)
                task.cancel()

    assert queue.empty()


async def test_kalshi_ingestor_uses_current_production_websocket_url():
    bus = EventBus()

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)
    mock_ws.send = AsyncMock()

    async def fake_aiter(self):
        if False:
            yield None

    mock_ws.__aiter__ = fake_aiter

    env = {
        "KALSHI_API_KEY_ID": "test_key_id",
        "KALSHI_PRIVATE_KEY": "test_private_key",
    }
    with patch.dict("os.environ", env):
        with patch("albert.ingestor.kalshi._load_private_key", return_value=make_private_key()):
            with patch("albert.ingestor.kalshi.websockets.connect", return_value=mock_ws) as connect:
                ingestor = KalshiIngestor(bus=bus, market_ids=["kalshi:BTC-24"])
                await ingestor._connect_and_stream()

    assert connect.call_args.args[0] == "wss://api.elections.kalshi.com/trade-api/ws/v2"


from albert.ingestor.polymarket import PolymarketIngestor


async def test_polymarket_ingestor_publishes_market_data_event():
    bus = EventBus()
    queue = bus.subscribe("market_data")

    ws_message = json.dumps([{
        "asset_id": "token456",
        "bid_price": "0.44",
        "ask_price": "0.46",
        "price": "0.45",
        "size": "200",
    }])

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)
    mock_ws.send = AsyncMock()

    async def fake_aiter(self):
        yield ws_message
        await asyncio.sleep(999)

    mock_ws.__aiter__ = fake_aiter

    with patch("albert.ingestor.polymarket.websockets.connect", return_value=mock_ws):
        ingestor = PolymarketIngestor(
            bus=bus,
            market_ids=["polymarket:cond123:token456"],
        )
        task = asyncio.create_task(ingestor.run())
        await asyncio.sleep(0.05)
        task.cancel()

    assert not queue.empty()
    event: MarketDataEvent = queue.get_nowait()
    assert event.market_id == "polymarket:cond123:token456"
    assert event.yes_ask == pytest.approx(0.46)
    assert event.no_bid == pytest.approx(0.54)
