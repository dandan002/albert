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


from albert.cli import cmd_health


def test_health_includes_adapter_status():
    conn = get_connection(":memory:")
    migrate(conn)
    conn.execute(
        "INSERT INTO health_status (component, component_type, status, details, checked_at) VALUES (?, ?, ?, ?, ?)",
        ("adapter:kalshi", "adapter", "healthy", '{"latency_ms": 42}', "2024-01-01T00:00:00+00:00")
    )
    conn.commit()
    health = cmd_health(conn)
    assert "adapters" in health
    assert health["adapters"]["kalshi"]["status"] == "healthy"


def test_health_includes_ingestor_status():
    conn = get_connection(":memory:")
    migrate(conn)
    conn.execute(
        "INSERT INTO health_status (component, component_type, status, details, checked_at) VALUES (?, ?, ?, ?, ?)",
        ("ingestor:kalshi", "ingestor", "healthy", '{"connected": true}', "2024-01-01T00:00:00+00:00")
    )
    conn.commit()
    health = cmd_health(conn)
    assert "ingestors" in health
    assert health["ingestors"]["kalshi"]["status"] == "healthy"


def test_health_includes_engine_status():
    conn = get_connection(":memory:")
    migrate(conn)
    conn.execute(
        "INSERT INTO health_status (component, component_type, status, details, checked_at) VALUES (?, ?, ?, ?, ?)",
        ("engine:strategy", "engine", "healthy", '{"done": false}', "2024-01-01T00:00:00+00:00")
    )
    conn.commit()
    health = cmd_health(conn)
    assert "engines" in health
    assert health["engines"]["strategy"]["status"] == "healthy"


def test_health_empty_db_has_expected_keys():
    conn = get_connection(":memory:")
    migrate(conn)
    health = cmd_health(conn)
    assert "database" in health
    assert "adapters" in health
    assert "strategies" in health
    assert "positions" in health
    assert "daily_pnl" in health
    assert "market_data" in health
