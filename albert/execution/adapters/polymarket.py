import asyncio
import logging
import os
from datetime import datetime, timezone

from albert.events import FillEvent, OrderIntent
from .base import ExchangeAdapter

logger = logging.getLogger(__name__)


def _create_client():
    """Create and return authenticated ClobClient using py-clob-client SDK."""
    try:
        from py_clob_client.client import ClobClient
    except ImportError:
        raise ImportError("py-clob-client not installed: pip install py-clob-client")

    host = os.environ.get("POLYMARKET_HOST", "https://clob.polymarket.com")
    key = os.environ.get("POLYMARKET_PRIVATE_KEY")
    chain_id = int(os.environ.get("POLYMARKET_CHAIN_ID", "137"))

    if not key:
        raise ValueError("POLYMARKET_PRIVATE_KEY environment variable not set")

    client = ClobClient(
        host,
        key=key,
        chain_id=chain_id,
    )
    return client


class PolymarketAdapter(ExchangeAdapter):
    def __init__(self) -> None:
        self._client = _create_client()

    def _token_id(self, market_id: str) -> str:
        parts = market_id.removeprefix("polymarket:").split(":")
        return parts[1] if len(parts) > 1 else parts[0]

    async def place_order(self, intent: OrderIntent, contracts: int, price: float) -> FillEvent:
        token_id = self._token_id(intent.market_id)

        from py_clob_client.clob_types import OrderArgs, OrderType

        side = "BUY" if intent.side == "yes" else "SELL"
        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=float(contracts),
            side=side,
        )

        signed = await self._client.create_order(order_args)
        posted = await self._request_with_retry(self._client.post_order, signed, OrderType.GTC)

        return FillEvent(
            fill_id=posted.get("orderID", "unknown"),
            market_id=intent.market_id,
            strategy_id=intent.strategy_id,
            side=intent.side,
            contracts=float(contracts),
            fill_price=price,
            fee=0.0,
            filled_at=datetime.now(timezone.utc),
        )

    async def cancel_order(self, order_id: str) -> None:
        await self._request_with_retry(self._client.cancel_order, order_id)

    async def get_bankroll(self) -> float:
        balance = await self._request_with_retry(self._client.get_balance)
        if isinstance(balance, dict):
            return float(balance.get("balance", 0))
        return 0.0