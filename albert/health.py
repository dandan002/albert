import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone

from albert.execution.adapters.base import ExchangeAdapter
from albert.ingestor.base import BaseIngestor

logger = logging.getLogger(__name__)


class HealthMonitor:
    def __init__(
        self,
        adapters: dict[str, ExchangeAdapter],
        ingestors: dict[str, BaseIngestor],
        conn: sqlite3.Connection,
        interval: float = 60.0,
        shutdown_event: asyncio.Event | None = None,
    ) -> None:
        self._adapters = adapters
        self._ingestors = ingestors
        self._conn = conn
        self._interval = interval
        self._shutdown_event = shutdown_event or asyncio.Event()

    async def run(self) -> None:
        while True:
            if self._shutdown_event.is_set():
                return
            await self._check_all()
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=self._interval)
                return
            except asyncio.TimeoutError:
                continue

    async def _check_all(self) -> None:
        now = datetime.now(timezone.utc).isoformat()

        for name, adapter in self._adapters.items():
            try:
                result = await asyncio.wait_for(adapter.health_check(), timeout=5.0)
                status = result.get("status", "unknown")
                details = json.dumps(result)
            except Exception as e:
                status = "unhealthy"
                details = json.dumps({"error": str(e)})
            self._persist(f"adapter:{name}", "adapter", status, details, now)

        for name, ingestor in self._ingestors.items():
            connected = ingestor.is_connected
            status = "healthy" if connected else "unhealthy"
            details = json.dumps({"connected": connected})
            self._persist(f"ingestor:{name}", "ingestor", status, details, now)

    def _persist(self, component: str, component_type: str, status: str, details: str, checked_at: str) -> None:
        self._conn.execute(
            """INSERT INTO health_status (component, component_type, status, details, checked_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(component) DO UPDATE SET
                   component_type = excluded.component_type,
                   status = excluded.status,
                   details = excluded.details,
                   checked_at = excluded.checked_at""",
            (component, component_type, status, details, checked_at),
        )
        self._conn.commit()
