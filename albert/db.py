import sqlite3
from pathlib import Path

DB_PATH = Path("albert.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    market_id   TEXT PRIMARY KEY,
    exchange    TEXT NOT NULL,
    title       TEXT NOT NULL,
    close_time  DATETIME,
    status      TEXT NOT NULL DEFAULT 'open',
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id   TEXT NOT NULL,
    timestamp   DATETIME NOT NULL,
    yes_bid     REAL,
    yes_ask     REAL,
    no_bid      REAL,
    no_ask      REAL,
    last_price  REAL,
    volume      REAL
);

CREATE TABLE IF NOT EXISTS positions (
    market_id       TEXT NOT NULL,
    strategy_id     TEXT NOT NULL,
    side            TEXT NOT NULL,
    contracts       REAL NOT NULL DEFAULT 0,
    avg_entry_price REAL NOT NULL DEFAULT 0,
    current_price   REAL,
    unrealized_pnl  REAL NOT NULL DEFAULT 0,
    opened_at       DATETIME NOT NULL,
    PRIMARY KEY (market_id, strategy_id)
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id     TEXT PRIMARY KEY,
    market_id   TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    side        TEXT NOT NULL,
    contracts   REAL NOT NULL,
    fill_price  REAL NOT NULL,
    fee         REAL NOT NULL DEFAULT 0,
    filled_at   DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS strategies (
    strategy_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    class_path  TEXT NOT NULL,
    config      TEXT NOT NULL DEFAULT '{}',
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    date        TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    realized_pnl    REAL NOT NULL DEFAULT 0,
    unrealized_pnl  REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (date, strategy_id)
);

CREATE TABLE IF NOT EXISTS health_status (
    component       TEXT PRIMARY KEY,
    component_type  TEXT NOT NULL,
    status          TEXT NOT NULL,
    details         TEXT,
    checked_at      TEXT NOT NULL
);
"""


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()
