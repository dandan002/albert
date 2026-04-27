import asyncio
import importlib
import json
import logging
import sqlite3

from albert.events import EventBus, MarketDataEvent
from albert.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class StrategyEngine:
    def __init__(
        self,
        bus: EventBus,
        conn: sqlite3.Connection,
        reload_interval: float = 30.0,
        shutdown_event: asyncio.Event | None = None,
    ) -> None:
        self._bus = bus
        self._conn = conn
        self._reload_interval = reload_interval
        self._strategies: dict[str, BaseStrategy] = {}
        self._last_reload: float = -1.0
        self._queue = bus.subscribe("market_data")
        self._shutdown_event = shutdown_event or asyncio.Event()

    def _load_strategies(self) -> None:
        rows = self._conn.execute(
            "SELECT strategy_id, class_path, config FROM strategies WHERE enabled = 1"
        ).fetchall()
        active_ids = {row["strategy_id"] for row in rows}

        # Remove disabled strategies
        for sid in list(self._strategies):
            if sid not in active_ids:
                del self._strategies[sid]

        for row in rows:
            sid = row["strategy_id"]
            config = json.loads(row["config"])
            if sid not in self._strategies:
                try:
                    module_path, class_name = row["class_path"].rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    cls = getattr(module, class_name)
                    self._strategies[sid] = cls(strategy_id=sid, config=config)
                    logger.info("loaded strategy %s", sid)
                except Exception:
                    logger.exception("failed to load strategy %s from %s", sid, row["class_path"])
            else:
                self._strategies[sid].config = config

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._load_strategies()
        self._last_reload = loop.time()

        while True:
            # Check for graceful shutdown
            if self._shutdown_event.is_set():
                logger.info("strategy:shutdown engine_stopped")
                return

            event: MarketDataEvent = await self._queue.get()

            if self._shutdown_event.is_set():
                logger.info("strategy:shutdown engine_stopped")
                return

            now = loop.time()
            if now - self._last_reload >= self._reload_interval:
                self._load_strategies()
                self._last_reload = now

            for strategy in list(self._strategies.values()):
                try:
                    intents = await strategy.on_market_data(event)
                    if intents:
                        for intent in intents:
                            await self._bus.publish("order_intents", intent)
                except Exception:
                    logger.exception("strategy %s raised on market data", strategy.id)
