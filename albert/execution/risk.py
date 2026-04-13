import logging
import sqlite3
import time
from datetime import date, datetime, timezone

from albert.events import EventBus, OrderIntent, StrategyHaltedEvent

logger = logging.getLogger(__name__)


class RiskChecker:
    def __init__(
        self,
        conn: sqlite3.Connection,
        global_config: dict,
        bus: EventBus | None = None,
    ) -> None:
        self._conn = conn
        self._config = global_config
        self._bus = bus
        self._last_order_time: dict[tuple[str, str], float] = {}
        self._loss_violation_count: dict[str, int] = {}

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
            # Track consecutive violations for circuit breaker
            violation_count = self._loss_violation_count.get(intent.strategy_id, 0) + 1
            self._loss_violation_count[intent.strategy_id] = violation_count
            max_violations = self._config.get("circuit_breaker_violations", 2)
            if violation_count >= max_violations:
                logger.error(
                    "risk:circuit_breaker_triggered strategy=%s violations=%d",
                    intent.strategy_id, violation_count
                )
                if self._bus:
                    self._bus.publish(
                        "strategy_halted",
                        StrategyHaltedEvent(
                            strategy_id=intent.strategy_id,
                            reason=f"circuit_breaker: daily loss limit reached {violation_count} times",
                            timestamp=datetime.now(timezone.utc),
                        ),
                    )
            return False

        # Clear violation count on successful check
        self._loss_violation_count[intent.strategy_id] = 0

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
