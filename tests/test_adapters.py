import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from albert.events import OrderIntent, FillEvent
from albert.execution.adapters.kalshi import KalshiAdapter


def make_intent() -> OrderIntent:
    return OrderIntent(
        market_id="kalshi:BTC-24",
        strategy_id="s1",
        side="yes",
        edge=0.10,
        confidence=1.0,
    )


async def test_kalshi_place_order_returns_fill():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "order": {
            "order_id": "ord_abc123",
            "count": 5,
            "yes_price": 47,
            "fee": 2,
        }
    }

    env = {
        "KALSHI_API_KEY_ID": "test_key_id",
        "KALSHI_PRIVATE_KEY": "test_private_key",
    }
    with patch.dict("os.environ", env):
        with patch("albert.execution.adapters.kalshi._load_private_key", return_value=MagicMock()):
            adapter = KalshiAdapter()
            with patch.object(adapter._client, "post", new=AsyncMock(return_value=mock_response)):
                fill = await adapter.place_order(make_intent(), contracts=5, price=0.47)

    assert fill.fill_id == "ord_abc123"
    assert fill.contracts == 5
    assert fill.fill_price == pytest.approx(0.47)
    assert fill.strategy_id == "s1"


async def test_kalshi_get_bankroll_returns_float():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"balance": {"available": 50000}}

    env = {
        "KALSHI_API_KEY_ID": "test_key_id",
        "KALSHI_PRIVATE_KEY": "test_private_key",
    }
    with patch.dict("os.environ", env):
        with patch("albert.execution.adapters.kalshi._load_private_key", return_value=MagicMock()):
            adapter = KalshiAdapter()
            with patch.object(adapter._client, "get", new=AsyncMock(return_value=mock_response)):
                bankroll = await adapter.get_bankroll()

    assert bankroll == pytest.approx(500.0)


def test_kalshi_adapter_uses_current_production_rest_base_url():
    env = {
        "KALSHI_API_KEY_ID": "test_key_id",
        "KALSHI_PRIVATE_KEY": "test_private_key",
    }
    with patch.dict("os.environ", env):
        with patch("albert.execution.adapters.kalshi._load_private_key", return_value=MagicMock()):
            adapter = KalshiAdapter()

    assert str(adapter._client.base_url) == "https://api.elections.kalshi.com/trade-api/v2/"


from albert.execution.adapters.polymarket import PolymarketAdapter


async def test_polymarket_place_order_returns_fill():
    mock_client = MagicMock()
    mock_client.create_order = AsyncMock()
    mock_client.post_order = AsyncMock(return_value={"orderID": "poly_xyz789", "status": "matched"})

    env = {
        "POLYMARKET_PRIVATE_KEY": "0xtest123456789abcdef",
        "POLYMARKET_CHAIN_ID": "137",
    }
    intent = OrderIntent(
        market_id="polymarket:cond123:token456",
        strategy_id="s1",
        side="yes",
        edge=0.10,
        confidence=1.0,
    )
    with patch.dict("os.environ", env):
        with patch("albert.execution.adapters.polymarket._create_client", return_value=mock_client):
            adapter = PolymarketAdapter()
            fill = await adapter.place_order(intent, contracts=10, price=0.45)

    assert fill.fill_id == "poly_xyz789"
    assert fill.contracts == 10
    assert fill.fill_price == pytest.approx(0.45)
    assert fill.market_id == "polymarket:cond123:token456"


async def test_polymarket_get_bankroll():
    mock_client = MagicMock()
    mock_client.get_balance = AsyncMock(return_value={"balance": "1000.00"})

    env = {
        "POLYMARKET_PRIVATE_KEY": "0xtest123456789abcdef",
        "POLYMARKET_CHAIN_ID": "137",
    }
    with patch.dict("os.environ", env):
        with patch("albert.execution.adapters.polymarket._create_client", return_value=mock_client):
            adapter = PolymarketAdapter()
            balance = await adapter.get_bankroll()

    assert balance == pytest.approx(1000.0)


async def test_polymarket_cancel_order():
    mock_client = MagicMock()
    mock_client.cancel_order = AsyncMock()

    env = {
        "POLYMARKET_PRIVATE_KEY": "0xtest123456789abcdef",
        "POLYMARKET_CHAIN_ID": "137",
    }
    with patch.dict("os.environ", env):
        with patch("albert.execution.adapters.polymarket._create_client", return_value=mock_client):
            adapter = PolymarketAdapter()
            await adapter.cancel_order("order_123")

    mock_client.cancel_order.assert_called_once_with("order_123")


async def test_kalshi_health_check_returns_healthy():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"balance": {"available": 50000}}

    env = {
        "KALSHI_API_KEY_ID": "test_key_id",
        "KALSHI_PRIVATE_KEY": "test_private_key",
    }
    with patch.dict("os.environ", env):
        with patch("albert.execution.adapters.kalshi._load_private_key", return_value=MagicMock()):
            adapter = KalshiAdapter()
            with patch.object(adapter._client, "get", new=AsyncMock(return_value=mock_response)):
                result = await adapter.health_check()

    assert result["status"] == "healthy"
    assert "latency_ms" in result
    assert isinstance(result["latency_ms"], float)


async def test_kalshi_health_check_returns_unhealthy_on_failure():
    env = {
        "KALSHI_API_KEY_ID": "test_key_id",
        "KALSHI_PRIVATE_KEY": "test_private_key",
    }
    with patch.dict("os.environ", env):
        with patch("albert.execution.adapters.kalshi._load_private_key", return_value=MagicMock()):
            adapter = KalshiAdapter()
            with patch.object(adapter._client, "get", new=AsyncMock(side_effect=Exception("API down"))):
                result = await adapter.health_check()

    assert result["status"] == "unhealthy"
    assert "error" in result
    assert "API down" in result["error"]


async def test_polymarket_health_check_returns_healthy():
    mock_client = MagicMock()
    mock_client.get_balance = AsyncMock(return_value={"balance": "1000.00"})

    env = {
        "POLYMARKET_PRIVATE_KEY": "0xtest123456789abcdef",
        "POLYMARKET_CHAIN_ID": "137",
    }
    with patch.dict("os.environ", env):
        with patch("albert.execution.adapters.polymarket._create_client", return_value=mock_client):
            adapter = PolymarketAdapter()
            result = await adapter.health_check()

    assert result["status"] == "healthy"
    assert "latency_ms" in result
    assert isinstance(result["latency_ms"], float)


async def test_polymarket_health_check_returns_unhealthy_on_failure():
    mock_client = MagicMock()
    mock_client.get_balance = AsyncMock(side_effect=Exception("SDK error"))

    env = {
        "POLYMARKET_PRIVATE_KEY": "0xtest123456789abcdef",
        "POLYMARKET_CHAIN_ID": "137",
    }
    with patch.dict("os.environ", env):
        with patch("albert.execution.adapters.polymarket._create_client", return_value=mock_client):
            adapter = PolymarketAdapter()
            result = await adapter.health_check()

    assert result["status"] == "unhealthy"
    assert "error" in result
    assert "SDK error" in result["error"]