import asyncio
import logging
import os
from datetime import datetime, timezone

import httpx

from albert.events import FillEvent, OrderIntent
from .base import ExchangeAdapter

logger = logging.getLogger(__name__)

_BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"
_MAX_RETRIES = 3


class KalshiAdapter(ExchangeAdapter):
    def __init__(self) -> None:
        token = os.environ["KALSHI_API_TOKEN"]
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )

    async def place_order(self, intent: OrderIntent, contracts: int, price: float) -> FillEvent:
        ticker = intent.market_id.removeprefix("kalshi:")
        price_cents = round(price * 100)
        payload = {
            "ticker": ticker,
            "action": "buy",
            "side": intent.side,
            "count": contracts,
            "type": "limit",
            f"{intent.side}_price": price_cents,
        }
        data = await self._post_with_retry("/portfolio/orders", payload)
        order = data["order"]
        fill_price = order.get(f"{intent.side}_price", price_cents) / 100
        return FillEvent(
            fill_id=order["order_id"],
            market_id=intent.market_id,
            strategy_id=intent.strategy_id,
            side=intent.side,
            contracts=float(order["count"]),
            fill_price=fill_price,
            fee=order.get("fee", 0) / 100,
            filled_at=datetime.now(timezone.utc),
        )

    async def cancel_order(self, order_id: str) -> None:
        for attempt in range(_MAX_RETRIES):
            try:
                r = await self._client.delete(f"/portfolio/orders/{order_id}")
                r.raise_for_status()
                return
            except httpx.HTTPError as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def get_bankroll(self) -> float:
        r = await self._client.get("/portfolio/balance")
        r.raise_for_status()
        return r.json()["balance"]["available"] / 100

    async def _post_with_retry(self, path: str, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                r = await self._client.post(path, json=payload)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as e:
                last_exc = e
                logger.warning("kalshi POST %s attempt %d failed: %s", path, attempt + 1, e)
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
        raise last_exc  # type: ignore[misc]  # guaranteed non-None after _MAX_RETRIES > 0
