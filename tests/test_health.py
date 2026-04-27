import asyncio
import sqlite3
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from albert.db import migrate, get_connection
from albert.health import HealthMonitor
from albert.ingestor.base import BaseIngestor
from albert.events import EventBus


class DummyIngestor(BaseIngestor):
    async def _connect_and_stream(self) -> None:
        pass

    def _normalize(self, raw: dict):
        return None


@pytest.fixture
def in_memory_db():
    conn = get_connection(":memory:")
    migrate(conn)
    yield conn
    conn.close()


async def test_health_monitor_polls_adapter_and_writes_row(in_memory_db):
    conn = in_memory_db
    mock_adapter = MagicMock()
    mock_adapter.health_check = AsyncMock(return_value={"status": "healthy", "latency_ms": 42.0})

    monitor = HealthMonitor(
        adapters={"kalshi": mock_adapter},
        ingestors={},
        conn=conn,
        interval=1.0,
    )

    await monitor._check_all()

    row = conn.execute("SELECT * FROM health_status WHERE component = 'adapter:kalshi'").fetchone()
    assert row is not None
    assert row["status"] == "healthy"
    assert row["component_type"] == "adapter"
    assert "42.0" in row["details"]


async def test_health_monitor_reads_ingestor_connection(in_memory_db):
    conn = in_memory_db
    bus = EventBus()
    ingestor = DummyIngestor(bus=bus)
    ingestor._connected = True

    monitor = HealthMonitor(
        adapters={},
        ingestors={"kalshi": ingestor},
        conn=conn,
        interval=1.0,
    )

    await monitor._check_all()

    row = conn.execute("SELECT * FROM health_status WHERE component = 'ingestor:kalshi'").fetchone()
    assert row is not None
    assert row["status"] == "healthy"
    assert row["component_type"] == "ingestor"


async def test_health_monitor_respects_interval(in_memory_db):
    conn = in_memory_db
    mock_adapter = MagicMock()
    mock_adapter.health_check = AsyncMock(return_value={"status": "healthy"})

    shutdown = asyncio.Event()
    monitor = HealthMonitor(
        adapters={"kalshi": mock_adapter},
        ingestors={},
        conn=conn,
        interval=0.05,
        shutdown_event=shutdown,
    )

    task = asyncio.create_task(monitor.run())
    await asyncio.sleep(0.25)
    shutdown.set()
    await task

    assert mock_adapter.health_check.call_count >= 2


async def test_health_monitor_handles_adapter_exception(in_memory_db):
    conn = in_memory_db
    mock_adapter = MagicMock()
    mock_adapter.health_check = AsyncMock(side_effect=Exception("API down"))

    monitor = HealthMonitor(
        adapters={"kalshi": mock_adapter},
        ingestors={},
        conn=conn,
        interval=1.0,
    )

    await monitor._check_all()

    row = conn.execute("SELECT * FROM health_status WHERE component = 'adapter:kalshi'").fetchone()
    assert row is not None
    assert row["status"] == "unhealthy"
    assert "API down" in row["details"]


async def test_health_status_table_schema(in_memory_db):
    conn = in_memory_db
    cursor = conn.execute("PRAGMA table_info(health_status)")
    columns = {row["name"]: row["type"] for row in cursor.fetchall()}
    assert "component" in columns
    assert "component_type" in columns
    assert "status" in columns
    assert "details" in columns
    assert "checked_at" in columns
