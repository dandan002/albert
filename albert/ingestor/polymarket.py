import asyncio
import json
import logging
from datetime import datetime, timezone

import websockets

from albert.events import EventBus, MarketDataEvent
from .base import BaseIngestor

logger = logging.getLogger(__name__)

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class PolymarketIngestor(BaseIngestor):
    def __init__(self, bus: EventBus, market_ids: list[str], shutdown_event: asyncio.Event | None = None) -> None:
        super().__init__(bus, shutdown_event=shutdown_event)
        self._market_ids = [mid for mid in market_ids if mid.startswith("polymarket:")]
        self._token_to_market: dict[str, str] = {}
        for mid in self._market_ids:
            parts = mid.removeprefix("polymarket:").split(":")
            if len(parts) > 1:
                self._token_to_market[parts[1]] = mid

    async def _connect_and_stream(self) -> None:
        async with websockets.connect(_WS_URL) as ws:
            self._connected = True
            try:
                await ws.send(json.dumps({
                    "type": "subscribe",
                    "assets_ids": list(self._token_to_market.keys()),
                    "channel": "price_change",
                }))
                async for raw_message in ws:
                    data = json.loads(raw_message)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        event = self._normalize(item)
                        if event:
                            await self._bus.publish("market_data", event)
            finally:
                self._connected = False

    def _normalize(self, raw: dict) -> MarketDataEvent | None:
        asset_id = raw.get("asset_id")
        if not asset_id:
            return None
        market_id = self._token_to_market.get(asset_id)
        if not market_id:
            return None
        yes_bid = float(raw.get("bid_price", 0))
        yes_ask = float(raw.get("ask_price", 0))
        return MarketDataEvent(
            market_id=market_id,
            exchange="polymarket",
            timestamp=datetime.now(timezone.utc),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=round(1.0 - yes_ask, 6),
            no_ask=round(1.0 - yes_bid, 6),
            last_price=float(raw.get("price", 0)),
            volume=float(raw.get("size", 0)),
        )
