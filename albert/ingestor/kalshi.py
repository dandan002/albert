import json
import logging
import os
from datetime import datetime, timezone

import websockets

from albert.events import EventBus, MarketDataEvent
from .base import BaseIngestor

logger = logging.getLogger(__name__)

_WS_URL = "wss://trading-api.kalshi.com/trade-api/ws/v2"


class KalshiIngestor(BaseIngestor):
    def __init__(self, bus: EventBus, market_ids: list[str]) -> None:
        super().__init__(bus)
        self._token = os.environ["KALSHI_API_TOKEN"]
        self._tickers = [mid.removeprefix("kalshi:") for mid in market_ids if mid.startswith("kalshi:")]

    async def _connect_and_stream(self) -> None:
        async with websockets.connect(
            _WS_URL,
            extra_headers={"Authorization": f"Bearer {self._token}"},
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
