import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
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

    with patch.dict("os.environ", {"KALSHI_API_TOKEN": "test_token"}):
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

    with patch.dict("os.environ", {"KALSHI_API_TOKEN": "test_token"}):
        adapter = KalshiAdapter()
        with patch.object(adapter._client, "get", new=AsyncMock(return_value=mock_response)):
            bankroll = await adapter.get_bankroll()

    assert bankroll == pytest.approx(500.0)
