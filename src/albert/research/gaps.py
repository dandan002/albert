"""Cross-venue price gaps between Kalshi and Polymarket on exactly matched markets.

For a matched pair (Kalshi YES == Polymarket outcome i), a locked-in trade exists when

    kalshi_yes_ask + (1 - pm_price_i + half_spread) + fees < 1      (YES on Kalshi, NO via PM)
    (1 - kalshi_yes_bid) + pm_price_i + half_spread + fees < 1      (NO on Kalshi, YES on PM)

Kalshi prices are hourly candles and Polymarket prices are history points, so this is an
*indicative* screen. Real arbitrage needs simultaneous books, which the paper trader checks.
"""

from datetime import timedelta

import polars as pl

from albert.fees import fee


def gap_table(matches: pl.DataFrame, kalshi_candles: pl.DataFrame, pm_history: pl.DataFrame,
              pm_markets: pl.DataFrame, kalshi_markets: pl.DataFrame | None = None, *,
              half_spread: float = 0.01, tolerance_minutes: float = 30.0,
              pre_close_hours: float = 4.0, pm_venue: str = "pmus") -> pl.DataFrame:
    """Gap observations, restricted to quiet pre-close periods.

    In-play prices move faster than hourly candles can follow, and history keeps printing 0/1
    after one venue resolves. Both create fake arbitrage, so we keep only observations at least
    ``pre_close_hours`` before the Kalshi close and with unresolved Polymarket prices.
    """
    tokens = pm_markets.select("condition_id", "token_ids")
    m = matches.join(tokens, left_on="pm_condition_id", right_on="condition_id").with_columns(
        pl.col("token_ids").list.get(pl.col("pm_outcome_index")).alias("token_id"))
    k = kalshi_candles.filter(pl.col("yes_ask_close").is_between(0.01, 0.99)
                              & pl.col("yes_bid_close").is_between(0.01, 0.99))
    pairs = k.join(m.select("kalshi_ticker", "token_id"), left_on="ticker",
                   right_on="kalshi_ticker")
    if kalshi_markets is not None:
        pairs = pairs.join(kalshi_markets.select("ticker", "close_time"), on="ticker").filter(
            pl.col("end_ts") <= pl.col("close_time") - timedelta(hours=pre_close_hours)
        ).drop("close_time")
    pairs = pairs.sort("end_ts")
    j = pairs.join_asof(
        pm_history.sort("ts").select("token_id", pl.col("ts").alias("pm_ts"), pl.col("price").alias("pm_price")),
        left_on="end_ts", right_on="pm_ts", by="token_id", strategy="nearest",
        tolerance=timedelta(minutes=tolerance_minutes),
    ).filter(pl.col("pm_price").is_between(0.02, 0.98))
    rows = []
    for r in j.iter_rows(named=True):
        ka, kb, p = r["yes_ask_close"], r["yes_bid_close"], r["pm_price"]
        # Leg A: YES on Kalshi + NO on PM.  Leg B: NO on Kalshi + YES on PM.
        cost_a = ka + (1 - p + half_spread)
        fees_a = fee("kalshi", 100, ka) / 100 + fee(pm_venue, 100, 1 - p + half_spread) / 100
        cost_b = (1 - kb) + p + half_spread
        fees_b = fee("kalshi", 100, 1 - kb) / 100 + fee(pm_venue, 100, p + half_spread) / 100
        rows.append({
            "ticker": r["ticker"], "ts": r["end_ts"], "kalshi_mid": (ka + kb) / 2, "pm_price": p,
            "gap": (ka + kb) / 2 - p,
            "arb_a": 1 - cost_a - fees_a, "arb_b": 1 - cost_b - fees_b,
        })
    out = pl.DataFrame(rows, schema={"ticker": pl.String, "ts": pl.Datetime("us", "UTC"),
                                     "kalshi_mid": pl.Float64, "pm_price": pl.Float64,
                                     "gap": pl.Float64, "arb_a": pl.Float64, "arb_b": pl.Float64})
    return out.with_columns(pl.max_horizontal("arb_a", "arb_b").alias("best_arb"))


def gap_summary(gaps: pl.DataFrame) -> dict[str, float]:
    if gaps.is_empty():
        return {"observations": 0}
    return {
        "observations": gaps.height,
        "markets": gaps["ticker"].n_unique(),
        "mean_abs_gap": float(gaps["gap"].abs().mean()),
        "p90_abs_gap": float(gaps["gap"].abs().quantile(0.9)),
        "arb_share": float((gaps["best_arb"] > 0).mean()),
        "arb_markets": gaps.filter(pl.col("best_arb") > 0)["ticker"].n_unique(),
        "mean_arb_when_positive": float(gaps.filter(pl.col("best_arb") > 0)["best_arb"].mean() or 0),
    }
