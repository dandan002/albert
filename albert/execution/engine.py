# albert/execution/engine.py
import asyncio
import json
import logging
import sqlite3
from datetime import datetime

from albert.events import EventBus, FillEvent, OrderIntent, StrategyHaltedEvent
from albert.execution.adapters.base import ExchangeAdapter
from albert.execution.kelly import kelly_size
from albert.execution.risk import RiskChecker

logger = logging.getLogger(__name__)


class ExecutionEngine:
    def __init__(
        self,
        bus: EventBus,
        conn: sqlite3.Connection,
        adapters: dict[str, ExchangeAdapter],
        global_config: dict,
    ) -> None:
        self._bus = bus
        self._conn = conn
        self._adapters = adapters
        self._risk = RiskChecker(conn, global_config)
        self._queue = self._bus.subscribe("order_intents")

    async def run(self) -> None:
        queue = self._queue
        while True:
            intent: OrderIntent = await queue.get()
            await self._handle_intent(intent)

    async def _handle_intent(self, intent: OrderIntent) -> None:
        exchange = intent.market_id.split(":")[0]
        adapter = self._adapters.get(exchange)
        if not adapter:
            logger.error("execution:no_adapter exchange=%s", exchange)
            return

        row = self._conn.execute(
            "SELECT yes_ask, no_ask FROM orderbook_snapshots WHERE market_id = ? ORDER BY timestamp DESC LIMIT 1",
            (intent.market_id,),
        ).fetchone()
        if not row:
            logger.warning("execution:no_orderbook market=%s", intent.market_id)
            return

        ask_price = row["yes_ask"] if intent.side == "yes" else row["no_ask"]
        if not ask_price:
            return

        strategy_row = self._conn.execute(
            "SELECT config FROM strategies WHERE strategy_id = ?",
            (intent.strategy_id,),
        ).fetchone()
        strategy_config = json.loads(strategy_row["config"]) if strategy_row else {}
        kelly_fraction = strategy_config.get("kelly_fraction", 0.25)
        max_position_usd = strategy_config.get("max_position_usd", 500.0)

        try:
            bankroll = await adapter.get_bankroll()
        except Exception:
            logger.exception("execution:bankroll_error strategy=%s", intent.strategy_id)
            return

        size_usd = kelly_size(intent.edge, ask_price, bankroll, kelly_fraction, intent.confidence, max_position_usd)
        if size_usd <= 0:
            return

        if not self._risk.check(intent, size_usd):
            return

        contracts = max(1, round(size_usd / ask_price))

        try:
            fill = await adapter.place_order(intent, contracts=contracts, price=ask_price)
        except Exception:
            logger.exception("execution:order_failed strategy=%s market=%s", intent.strategy_id, intent.market_id)
            await self._halt_strategy(intent.strategy_id, "order placement failed after retries")
            return

        self._persist_fill(fill)
        await self._bus.publish("fills", fill)
        logger.info(
            "execution:fill fill_id=%s strategy=%s market=%s contracts=%s price=%.4f",
            fill.fill_id, fill.strategy_id, fill.market_id, fill.contracts, fill.fill_price,
        )

    def _persist_fill(self, fill: FillEvent) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO fills
               (fill_id, market_id, strategy_id, side, contracts, fill_price, fee, filled_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (fill.fill_id, fill.market_id, fill.strategy_id, fill.side,
             fill.contracts, fill.fill_price, fill.fee, fill.filled_at.isoformat()),
        )
        self._conn.commit()

    async def _halt_strategy(self, strategy_id: str, reason: str) -> None:
        self._conn.execute(
            "UPDATE strategies SET enabled = 0 WHERE strategy_id = ?",
            (strategy_id,),
        )
        self._conn.commit()
        await self._bus.publish("strategy_halted", StrategyHaltedEvent(
            strategy_id=strategy_id,
            reason=reason,
            timestamp=datetime.utcnow(),
        ))
        logger.error("execution:strategy_halted strategy=%s reason=%s", strategy_id, reason)
