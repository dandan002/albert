import asyncio
import base64
import json
import logging
import os
from datetime import datetime, timezone

import websockets
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

from albert.events import EventBus, MarketDataEvent
from albert.execution.adapters.kalshi import _load_private_key
from .base import BaseIngestor

logger = logging.getLogger(__name__)

_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
_WS_PATH = "/trade-api/ws/v2"


class KalshiIngestor(BaseIngestor):
    def __init__(self, bus: EventBus, market_ids: list[str], shutdown_event: asyncio.Event | None = None) -> None:
        super().__init__(bus, shutdown_event=shutdown_event)
        self._key_id = os.environ["KALSHI_API_KEY_ID"]
        self._private_key = _load_private_key(os.environ["KALSHI_PRIVATE_KEY"])
        self._tickers = [mid.removeprefix("kalshi:") for mid in market_ids if mid.startswith("kalshi:")]

    def _make_auth_headers(self) -> dict:
        ts = str(int(datetime.now(timezone.utc).timestamp() * 1000))
        msg = f"{ts}GET{_WS_PATH}".encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    async def _connect_and_stream(self) -> None:
        async with websockets.connect(
            _WS_URL,
            additional_headers=self._make_auth_headers(),
        ) as ws:
            for ticker in self._tickers:
                await ws.send(json.dumps({
                    "id": 1,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["orderbook_delta"],
                        "market_tickers": [ticker],
                    },
                }))
            async for raw_message in ws:
                data = json.loads(raw_message)
                event = self._normalize(data)
                if event:
                    await self._bus.publish("market_data", event)

    def _normalize(self, raw: dict) -> MarketDataEvent | None:
        if raw.get("type") not in ("orderbook_snapshot", "orderbook_delta"):
            return None
        msg = raw.get("msg", {})
        ticker = msg.get("market_ticker")
        if not ticker:
            return None
        yes = msg.get("yes", {})
        no = msg.get("no", {})
        return MarketDataEvent(
            market_id=f"kalshi:{ticker}",
            exchange="kalshi",
            timestamp=datetime.now(timezone.utc),
            yes_bid=yes.get("bid", 0) / 100,
            yes_ask=yes.get("ask", 0) / 100,
            no_bid=no.get("bid", 0) / 100,
            no_ask=no.get("ask", 0) / 100,
            last_price=msg.get("last_price", 0) / 100,
            volume=float(msg.get("volume", 0)),
        )
