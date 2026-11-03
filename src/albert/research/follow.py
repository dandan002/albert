"""Follow-the-wallet signals: copy a scored wallet's BUY after a realistic delay.

The follower never gets the wallet's price. Entry is the market price at ``ts + delay`` (from
price history) plus a half spread. The wallet's skill estimate comes only from window A, while
the copied trades come from window B, so there is no look-ahead.
"""

from datetime import timedelta

import polars as pl

SIGNAL_SCHEMA = {
    "strategy": pl.String,
    "venue": pl.String,
    "market_id": pl.String,
    "event_id": pl.String,
    "side": pl.String,
    "signal_ts": pl.Datetime("us", "UTC"),
    "resolve_ts": pl.Datetime("us", "UTC"),
    "price": pl.Float64,
    "prob": pl.Float64,
    "available": pl.Float64,
    "payoff": pl.Float64,
    "fee_type": pl.String,
    "fee_multiplier": pl.Float64,
    "note": pl.String,
}


def wallet_edges(rt: pl.DataFrame, wallets: list[str], prior_markets: float = 30.0) -> pl.DataFrame:
    """Per-wallet shrunk edge in probability points (won - price), from window-A trades only."""
    buys = rt.filter(pl.col("wallet").is_in(wallets) & (pl.col("side") == "BUY"))
    per_mkt = buys.group_by("wallet", "condition_id").agg(
        ((pl.col("won") - pl.col("price")) * pl.col("size")).sum().alias("edge_size"),
        pl.col("size").sum(),
    ).with_columns((pl.col("edge_size") / pl.col("size")).alias("edge_pts"))
    return per_mkt.group_by("wallet").agg(
        pl.len().alias("markets_a"), pl.col("edge_pts").mean().alias("edge_pts")
    ).with_columns(
        (pl.col("edge_pts") * pl.col("markets_a") / (pl.col("markets_a") + prior_markets))
        .alias("shrunk_edge_pts"))


def follow_candidates(rt_b: pl.DataFrame, edges: pl.DataFrame, *, min_usd: float = 50.0) -> pl.DataFrame:
    """First BUY per (wallet, market, outcome) in window B by a wallet with positive A-edge."""
    good = edges.filter(pl.col("shrunk_edge_pts") > 0)
    return (rt_b.filter((pl.col("side") == "BUY") & ((pl.col("price") * pl.col("size")) >= min_usd))
            .join(good.select("wallet", "shrunk_edge_pts"), on="wallet")
            .sort("ts")
            .unique(["wallet", "condition_id", "outcome_index"], keep="first", maintain_order=True))


def follow_signals(cands: pl.DataFrame, history: pl.DataFrame, *, delay_minutes: float = 5.0,
                   half_spread: float = 0.01, max_wait_minutes: float = 60.0,
                   venue: str = "pm_proxy") -> pl.DataFrame:
    """Turn candidates into executable signals using the price at ts + delay."""
    c = cands.with_columns((pl.col("ts") + timedelta(minutes=delay_minutes)).alias("entry_ts"))
    joined = c.sort("entry_ts").join_asof(
        history.sort("ts").select("token_id", pl.col("ts").alias("px_ts"), pl.col("price").alias("px")),
        left_on="entry_ts", right_on="px_ts", by="token_id", strategy="forward",
        tolerance=timedelta(minutes=max_wait_minutes),
    ).filter(pl.col("px").is_not_null())
    return joined.select(
        pl.lit("wallet_follow").alias("strategy"),
        pl.lit(venue).alias("venue"),
        (pl.col("condition_id") + ":" + pl.col("outcome_index").cast(pl.String)).alias("market_id"),
        pl.coalesce("event_slug", "condition_id").alias("event_id"),
        pl.lit("yes").alias("side"),
        pl.col("px_ts").alias("signal_ts"),
        pl.col("closed_time").alias("resolve_ts"),
        (pl.col("px") + half_spread).clip(0.01, 0.99).alias("price"),
        (pl.col("price") + pl.col("shrunk_edge_pts")).clip(0.01, 0.99).alias("prob"),
        pl.lit(None, dtype=pl.Float64).alias("available"),
        pl.col("won").alias("payoff"),
        pl.lit(None, dtype=pl.String).alias("fee_type"),
        pl.lit(None, dtype=pl.Float64).alias("fee_multiplier"),
        ("copy " + pl.col("wallet").str.slice(0, 10)).alias("note"),
    ).cast(SIGNAL_SCHEMA)  # type: ignore[arg-type]
