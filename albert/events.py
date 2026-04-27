import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class MarketDataEvent:
    market_id: str
    exchange: Literal["kalshi", "polymarket"]
    timestamp: datetime
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    last_price: float
    volume: float


@dataclass
class OrderIntent:
    market_id: str
    strategy_id: str
    side: Literal["yes", "no"]
    edge: float
    confidence: float


@dataclass
class FillEvent:
    fill_id: str
    market_id: str
    strategy_id: str
    side: Literal["yes", "no"]
    contracts: float
    fill_price: float
    fee: float
    filled_at: datetime


@dataclass
class StrategyHaltedEvent:
    strategy_id: str
    reason: str
    timestamp: datetime


class EventBus:
    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue]] = {}

    def subscribe(self, channel: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.setdefault(channel, []).append(q)
        return q

    async def publish(self, channel: str, event: object) -> None:
        queues = self._queues.get(channel, [])
        if not queues:
            return
        for q in queues:
            await q.put(event)
