# tests/test_cli.py
import io
import sys
from datetime import datetime, timezone
from albert.db import get_connection, migrate
from albert.cli import cmd_status


def make_db_with_data():
    conn = get_connection(":memory:")
    migrate(conn)
    conn.execute(
        "INSERT INTO positions (market_id, strategy_id, side, contracts, avg_entry_price, current_price, unrealized_pnl, opened_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("kalshi:X", "momentum_v1", "yes", 5.0, 0.40, 0.45, 0.25, datetime.now(timezone.utc).isoformat())
    )
    conn.execute(
        "INSERT INTO daily_pnl (date, strategy_id, realized_pnl, unrealized_pnl) VALUES (date('now'), 'momentum_v1', 1.50, 0.25)"
    )
    conn.commit()
    return conn


def test_status_prints_strategy_row(capsys):
    conn = make_db_with_data()
    cmd_status(conn)
    out = capsys.readouterr().out
    assert "momentum_v1" in out
    lines = [l for l in out.splitlines() if "momentum_v1" in l]
    assert len(lines) == 1
    assert "1" in lines[0]  # 1 position in the momentum_v1 row


def test_status_prints_total_row(capsys):
    conn = make_db_with_data()
    cmd_status(conn)
    out = capsys.readouterr().out
    assert "TOTAL" in out


def test_status_empty_db_prints_totals(capsys):
    conn = get_connection(":memory:")
    migrate(conn)
    cmd_status(conn)
    out = capsys.readouterr().out
    assert "TOTAL" in out
    assert "0" in out
