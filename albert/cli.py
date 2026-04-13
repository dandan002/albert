# albert/cli.py
import json
import sqlite3


def cmd_health(conn: sqlite3.Connection) -> dict:
    """Get health status of all subsystems."""
    health = {
        "database": {"connected": False, "wal_mode": False},
        "adapters": {},
        "strategies": {"enabled": 0, "running": 0},
        "positions": {"count": 0, "notional_usd": 0.0},
        "daily_pnl": {"realized": 0.0, "unrealized": 0.0, "total": 0.0},
        "market_data": {"last_update": None},
    }

    # Database health
    try:
        conn.execute("SELECT 1")
        health["database"]["connected"] = True
        wal = conn.execute("PRAGMA journal_mode").fetchone()
        health["database"]["wal_mode"] = wal["journal_mode"] == "wal"
    except Exception:
        pass

    # Strategy health
    try:
        rows = conn.execute("SELECT COUNT(*) as cnt FROM strategies WHERE enabled = 1").fetchone()
        health["strategies"]["enabled"] = rows["cnt"]
        health["strategies"]["running"] = rows["cnt"]
    except Exception:
        pass

    # Position health
    try:
        rows = conn.execute(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(contracts * current_price), 0) as notional FROM positions"
        ).fetchone()
        health["positions"]["count"] = rows["cnt"]
        health["positions"]["notional_usd"] = rows["notional"]
    except Exception:
        pass

    # Daily P&L
    try:
        today = conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) as realized, COALESCE(SUM(unrealized_pnl), 0) as unrealized FROM daily_pnl WHERE date = date('now')"
        ).fetchone()
        health["daily_pnl"]["realized"] = today["realized"]
        health["daily_pnl"]["unrealized"] = today["unrealized"]
        health["daily_pnl"]["total"] = today["realized"] + today["unrealized"]
    except Exception:
        pass

    # Market data last update
    try:
        row = conn.execute(
            "SELECT MAX(timestamp) as last FROM orderbook_snapshots"
        ).fetchone()
        if row and row["last"]:
            health["market_data"]["last_update"] = row["last"]
    except Exception:
        pass

    return health


def cmd_status(conn: sqlite3.Connection) -> None:
    rows = conn.execute("""
        SELECT
            p.strategy_id,
            COUNT(DISTINCT p.market_id)             AS positions,
            COALESCE(SUM(p.unrealized_pnl), 0)      AS unrealized,
            COALESCE(d.realized_pnl, 0)             AS realized
        FROM positions p
        LEFT JOIN daily_pnl d
            ON p.strategy_id = d.strategy_id AND d.date = date('now')
        GROUP BY p.strategy_id
    """).fetchall()

    col_w = 22
    header = f"{'Strategy':<{col_w}} {'Positions':>10} {'Unrealized PnL':>14} {'Today Realized':>14}"
    sep = "─" * len(header)
    print(header)
    print(sep)

    total_positions = 0
    total_unrealized = 0.0
    total_realized = 0.0

    for row in rows:
        u = row["unrealized"]
        r = row["realized"]
        print(
            f"{row['strategy_id']:<{col_w}} {row['positions']:>10} "
            f"{u:>+14.2f} {r:>+14.2f}"
        )
        total_positions += row["positions"]
        total_unrealized += u
        total_realized += r

    print(sep)
    print(
        f"{'TOTAL':<{col_w}} {total_positions:>10} "
        f"{total_unrealized:>+14.2f} {total_realized:>+14.2f}"
    )
