# albert/cli.py
import sqlite3


def cmd_status(conn: sqlite3.Connection) -> None:
    rows = conn.execute("""
        SELECT
            p.strategy_id,
            COUNT(*)                                                    AS positions,
            COALESCE(SUM(p.unrealized_pnl), 0)                         AS unrealized,
            COALESCE(SUM(d.realized_pnl), 0)                           AS realized,
            COALESCE(SUM(CASE WHEN d.date = date('now') THEN d.realized_pnl ELSE 0 END), 0) AS today
        FROM positions p
        LEFT JOIN daily_pnl d ON p.strategy_id = d.strategy_id
        GROUP BY p.strategy_id
    """).fetchall()

    col_w = 22
    header = f"{'Strategy':<{col_w}} {'Positions':>10} {'Unrealized PnL':>15} {'Realized PnL':>13} {'Today':>10}"
    sep = "─" * len(header)
    print(header)
    print(sep)

    total_positions = 0
    total_unrealized = 0.0
    total_realized = 0.0
    total_today = 0.0

    for row in rows:
        u = row["unrealized"] or 0.0
        r = row["realized"] or 0.0
        t = row["today"] or 0.0
        print(
            f"{row['strategy_id']:<{col_w}} {row['positions']:>10} "
            f"{u:>+14.2f} {r:>+12.2f} {t:>+9.2f}"
        )
        total_positions += row["positions"]
        total_unrealized += u
        total_realized += r
        total_today += t

    print(sep)
    print(
        f"{'TOTAL':<{col_w}} {total_positions:>10} "
        f"{total_unrealized:>+14.2f} {total_realized:>+12.2f} {total_today:>+9.2f}"
    )
