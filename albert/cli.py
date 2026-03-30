# albert/cli.py
import sqlite3


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
