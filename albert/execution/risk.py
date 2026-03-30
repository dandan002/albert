import logging
import sqlite3
import time
from datetime import date

from albert.events import OrderIntent

logger = logging.getLogger(__name__)


class RiskChecker:
    def __init__(self, conn: sqlite3.Connection, global_config: dict) -> None:
        self._conn = conn
        self._config = global_config
        self._last_order_time: dict[tuple[str, str], float] = {}

    def check(self, intent: OrderIntent, position_size_usd: float) -> bool:
        key = (intent.market_id, intent.strategy_id)
        debounce = self._config.get("order_debounce_seconds", 10)
        now = time.monotonic()

        if debounce > 0 and now - self._last_order_time.get(key, 0.0) < debounce:
            logger.info(
                "risk:debounce market=%s strategy=%s", intent.market_id, intent.strategy_id
            )
            return False

        today = date.today().isoformat()
        row = self._conn.execute(
            "SELECT COALESCE(SUM(realized_pnl + unrealized_pnl), 0) AS total FROM daily_pnl WHERE date = ?",
            (today,),
        ).fetchone()
        daily_pnl = row["total"]
        limit = self._config.get("daily_loss_limit_usd", -500.0)
        if daily_pnl < limit:
            logger.warning("risk:daily_loss_limit pnl=%.2f limit=%.2f", daily_pnl, limit)
            return False

        row = self._conn.execute(
            "SELECT COALESCE(SUM(contracts * current_price), 0) AS notional FROM positions"
        ).fetchone()
        current_notional = row["notional"]
        max_notional = self._config.get("max_total_notional_usd", 10000.0)
        if current_notional + position_size_usd > max_notional:
            logger.info(
                "risk:max_notional current=%.2f new=%.2f max=%.2f",
                current_notional, position_size_usd, max_notional,
            )
            return False

        self._last_order_time[key] = now
        return True
