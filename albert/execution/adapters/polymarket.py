# albert/execution/adapters/polymarket.py
import asyncio
import logging
import os
from datetime import datetime

import httpx

from albert.events import FillEvent, OrderIntent
from .base import ExchangeAdapter

logger = logging.getLogger(__name__)

_BASE_URL = "https://clob.polymarket.com"
_MAX_RETRIES = 3


class PolymarketAdapter(ExchangeAdapter):
    def __init__(self) -> None:
        self._api_key = os.environ["POLYMARKET_API_KEY"]
        self._api_secret = os.environ["POLYMARKET_API_SECRET"]
        self._api_passphrase = os.environ["POLYMARKET_API_PASSPHRASE"]
        # TODO: Production use requires per-request ECDSA signing via api_secret/passphrase.
        # See Polymarket CLOB L2 auth docs for POLY_SIGNATURE / POLY_TIMESTAMP / POLY_NONCE.
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={
                "POLY_ADDRESS": os.environ["POLYMARKET_ADDRESS"],
                "POLY_API_KEY": self._api_key,
                "Content-Type": "application/json",
            },
            timeout=10.0,
        )

    def _token_id(self, market_id: str) -> str:
        # market_id format: polymarket:<condition_id>:<token_id>
        parts = market_id.removeprefix("polymarket:").split(":")
        return parts[1] if len(parts) > 1 else parts[0]

    async def place_order(self, intent: OrderIntent, contracts: int, price: float) -> FillEvent:
        token_id = self._token_id(intent.market_id)
        payload = {
            "orderType": "GTC",
            "tokenID": token_id,
            "price": str(price),
            "size": str(contracts),
            "side": "BUY",
        }
        data = await self._post_with_retry("/order", payload)
        return FillEvent(
            fill_id=data.get("orderID", "unknown"),
            market_id=intent.market_id,
            strategy_id=intent.strategy_id,
            side=intent.side,
            contracts=float(contracts),
            fill_price=price,
            fee=0.0,
            filled_at=datetime.utcnow(),
        )

    async def cancel_order(self, order_id: str) -> None:
        for attempt in range(_MAX_RETRIES):
            try:
                r = await self._client.delete(f"/order/{order_id}")
                r.raise_for_status()
                return
            except httpx.HTTPError as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def get_bankroll(self) -> float:
        r = await self._client.get("/balance")
        r.raise_for_status()
        return float(r.json().get("balance", 0.0))

    async def _post_with_retry(self, path: str, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                r = await self._client.post(path, json=payload)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as e:
                last_exc = e
                logger.warning("polymarket POST %s attempt %d failed: %s", path, attempt + 1, e)
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
        raise last_exc  # type: ignore[misc]
