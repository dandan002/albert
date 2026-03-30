import asyncio
import logging
import sqlite3
from datetime import datetime, date

from albert.events import EventBus, FillEvent, MarketDataEvent

logger = logging.getLogger(__name__)


class PortfolioTracker:
    def __init__(self, bus: EventBus, conn: sqlite3.Connection) -> None:
        self._bus = bus
        self._conn = conn
        self._fills_queue = self._bus.subscribe("fills")
        self._market_data_queue = self._bus.subscribe("market_data")

    async def run(self) -> None:
        fills_queue = self._fills_queue
        market_data_queue = self._market_data_queue

        async def handle_fills() -> None:
            while True:
                fill: FillEvent = await fills_queue.get()
                self._handle_fill(fill)

        async def handle_market_data() -> None:
            while True:
                event: MarketDataEvent = await market_data_queue.get()
                self._handle_market_data(event)

        await asyncio.gather(handle_fills(), handle_market_data())

    def _handle_fill(self, fill: FillEvent) -> None:
        existing = self._conn.execute(
            "SELECT contracts, avg_entry_price FROM positions WHERE market_id = ? AND strategy_id = ?",
            (fill.market_id, fill.strategy_id),
        ).fetchone()

        if existing:
            new_contracts = existing["contracts"] + fill.contracts
            if new_contracts <= 0:
                realized = (fill.fill_price - existing["avg_entry_price"]) * abs(fill.contracts)
                self._conn.execute(
                    "DELETE FROM positions WHERE market_id = ? AND strategy_id = ?",
                    (fill.market_id, fill.strategy_id),
                )
                self._record_realized_pnl(fill.strategy_id, realized)
            else:
                new_avg = (
                    existing["avg_entry_price"] * existing["contracts"]
                    + fill.fill_price * fill.contracts
                ) / new_contracts
                self._conn.execute(
                    "UPDATE positions SET contracts = ?, avg_entry_price = ? WHERE market_id = ? AND strategy_id = ?",
                    (new_contracts, new_avg, fill.market_id, fill.strategy_id),
                )
        else:
            self._conn.execute(
                """INSERT INTO positions
                   (market_id, strategy_id, side, contracts, avg_entry_price, current_price, unrealized_pnl, opened_at)
                   VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
                (fill.market_id, fill.strategy_id, fill.side,
                 fill.contracts, fill.fill_price, fill.fill_price,
                 datetime.utcnow().isoformat()),
            )

        self._conn.commit()
        logger.info(
            "portfolio:fill market=%s strategy=%s contracts=%s price=%.4f",
            fill.market_id, fill.strategy_id, fill.contracts, fill.fill_price,
        )

    def _handle_market_data(self, event: MarketDataEvent) -> None:
        rows = self._conn.execute(
            "SELECT strategy_id, contracts, avg_entry_price, side FROM positions WHERE market_id = ?",
            (event.market_id,),
        ).fetchall()
        for row in rows:
            current_price = event.yes_bid if row["side"] == "yes" else event.no_bid
            unrealized = (current_price - row["avg_entry_price"]) * row["contracts"]
            self._conn.execute(
                "UPDATE positions SET current_price = ?, unrealized_pnl = ? WHERE market_id = ? AND strategy_id = ?",
                (current_price, unrealized, event.market_id, row["strategy_id"]),
            )
        if rows:
            self._conn.commit()

    def _record_realized_pnl(self, strategy_id: str, realized_pnl: float) -> None:
        today = date.today().isoformat()
        self._conn.execute(
            """INSERT INTO daily_pnl (date, strategy_id, realized_pnl, unrealized_pnl)
               VALUES (?, ?, ?, 0)
               ON CONFLICT(date, strategy_id) DO UPDATE SET
                   realized_pnl = realized_pnl + excluded.realized_pnl""",
            (today, strategy_id, realized_pnl),
        )
        self._conn.commit()
