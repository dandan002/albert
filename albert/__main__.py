# albert/__main__.py
import asyncio
import logging
import logging.handlers
import os
import sqlite3
import sys
from pathlib import Path

from albert.config import load_global_config
from albert.db import get_connection, migrate
from albert.events import EventBus
from albert.execution.adapters.kalshi import KalshiAdapter
from albert.execution.adapters.polymarket import PolymarketAdapter
from albert.execution.engine import ExecutionEngine
from albert.ingestor.kalshi import KalshiIngestor
from albert.ingestor.polymarket import PolymarketIngestor
from albert.portfolio.tracker import PortfolioTracker
from albert.strategies.engine import StrategyEngine
from albert.cli import cmd_status

_LOG_FORMAT = '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "msg": "%(message)s"}'

_REQUIRED_ENV = [
    "KALSHI_API_TOKEN",
    "POLYMARKET_API_KEY",
    "POLYMARKET_API_SECRET",
    "POLYMARKET_API_PASSPHRASE",
    "POLYMARKET_ADDRESS",
]


def _check_env() -> None:
    missing = [v for v in _REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        print(f"ERROR: missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)


def _setup_logging() -> None:
    handler = logging.handlers.RotatingFileHandler(
        "albert.log", maxBytes=10 * 1024 * 1024, backupCount=3
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.basicConfig(level=logging.INFO, handlers=[handler, stream_handler])


async def _ttl_cleanup(conn: sqlite3.Connection, ttl_days: int) -> None:
    while True:
        await asyncio.sleep(3600)  # run every hour
        conn.execute(
            f"DELETE FROM orderbook_snapshots WHERE timestamp < datetime('now', '-{ttl_days} days')"
        )
        conn.commit()


async def _main(conn: sqlite3.Connection, global_config: dict) -> None:
    bus = EventBus()

    rows = conn.execute("SELECT market_id FROM markets WHERE status = 'open'").fetchall()
    market_ids = [row["market_id"] for row in rows]

    adapters = {
        "kalshi": KalshiAdapter(),
        "polymarket": PolymarketAdapter(),
    }

    reload_interval = global_config.get("strategy_reload_interval", 30.0)

    await asyncio.gather(
        KalshiIngestor(bus, market_ids).run(),
        PolymarketIngestor(bus, market_ids).run(),
        StrategyEngine(bus, conn, reload_interval=reload_interval).run(),
        ExecutionEngine(bus, conn, adapters, global_config).run(),
        PortfolioTracker(bus, conn).run(),
        _ttl_cleanup(conn, global_config.get("orderbook_ttl_days", 7)),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        conn = get_connection()
        migrate(conn)
        cmd_status(conn)
        sys.exit(0)

    _setup_logging()
    _check_env()
    conn = get_connection()
    migrate(conn)
    global_config = load_global_config()
    asyncio.run(_main(conn, global_config))
