import sqlite3
import pytest
from albert.db import get_connection, migrate

def test_migrate_creates_tables():
    conn = get_connection(":memory:")
    migrate(conn)
    tables = {
        row[0] for row in
        conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    expected = {"markets", "orderbook_snapshots", "positions", "fills", "strategies", "daily_pnl"}
    assert tables & expected == expected

def test_migrate_is_idempotent():
    conn = get_connection(":memory:")
    migrate(conn)
    migrate(conn)  # should not raise

def test_strategies_table_columns():
    conn = get_connection(":memory:")
    migrate(conn)
    conn.execute(
        "INSERT INTO strategies (strategy_id, name, class_path, config, enabled) VALUES (?, ?, ?, ?, ?)",
        ("s1", "Test", "albert.strategies.examples.momentum.MomentumV1", '{"min_edge": 0.05}', 1)
    )
    row = conn.execute("SELECT * FROM strategies WHERE strategy_id = 's1'").fetchone()
    assert row["name"] == "Test"
    assert row["enabled"] == 1
