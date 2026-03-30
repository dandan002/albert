import asyncio
import base64
import logging
import os
from datetime import datetime, timezone

import httpx
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from albert.events import FillEvent, OrderIntent
from .base import ExchangeAdapter

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
_MAX_RETRIES = 3


def _load_private_key(key_str: str):
    """Load an RSA private key from a PEM string.

    Handles keys stored as a single line in env vars (newlines replaced by spaces,
    BEGIN header possibly absent).
    """
    pem = key_str.strip().strip("\"'")

    if "\n" not in pem:
        end_marker = "-----END RSA PRIVATE KEY-----"
        begin_marker = "-----BEGIN RSA PRIVATE KEY-----"
        if end_marker in pem:
            body = pem.replace(end_marker, "").strip()
        else:
            body = pem
        if begin_marker in body:
            body = body.replace(begin_marker, "").strip()
        raw_b64 = body.replace(" ", "")
        wrapped = "\n".join(raw_b64[i:i + 64] for i in range(0, len(raw_b64), 64))
        pem = f"{begin_marker}\n{wrapped}\n{end_marker}"

    return serialization.load_pem_private_key(pem.encode(), password=None, backend=default_backend())


class KalshiAdapter(ExchangeAdapter):
    def __init__(self) -> None:
        self._key_id = os.environ["KALSHI_API_KEY_ID"]
        self._private_key = _load_private_key(os.environ["KALSHI_PRIVATE_KEY"])
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=10.0,
            event_hooks={"request": [self._sign_request]},
        )

    def _sign_request(self, request: httpx.Request) -> None:
        ts = str(int(datetime.now(timezone.utc).timestamp() * 1000))
        path = request.url.path
        msg = f"{ts}{request.method}{path}".encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        request.headers["KALSHI-ACCESS-KEY"] = self._key_id
        request.headers["KALSHI-ACCESS-SIGNATURE"] = base64.b64encode(sig).decode()
        request.headers["KALSHI-ACCESS-TIMESTAMP"] = ts

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
        raise last_exc  # type: ignore[misc]
