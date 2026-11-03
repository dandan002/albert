"""Wallet skill scoring for Polymarket intl wallets.

Each trade is valued as if held to resolution, against the price the wallet paid:

    BUY  outcome at p:  pnl = size * (won - p),   capital at risk = size * p
    SELL outcome at p:  pnl = size * (p - won),   capital at risk = size * (1 - p)

Exits before resolution are therefore scored on information, not timing. That is the property
we would copy anyway, since a follower enters later and cannot copy the exit.

Why not win rate: buying 95c favorites wins 95% of the time with zero edge. We rank by
**edge per dollar risked**, aggregated per market so that 200 fills in one market count as one
observation. The score is shrunk toward zero for small samples and tested out of sample
(score on window A, evaluate on window B).
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl


@dataclass(frozen=True)
class WalletScoreConfig:
    min_markets: int = 20        # need at least this many resolved markets to be scored
    prior_markets: float = 30.0  # shrinkage strength: pseudo-markets with zero edge
    mm_two_sided: float = 0.5    # flag if this share of markets has both BUY and SELL activity


def resolved_trades(trades: pl.DataFrame, markets: pl.DataFrame) -> pl.DataFrame:
    """Join trades to resolutions. Adds won / pnl / risk per trade."""
    res = markets.select("condition_id", "winner_index", "closed_time", "tags").filter(
        pl.col("winner_index").is_not_null())
    t = trades.join(res, on="condition_id", how="inner").filter(
        pl.col("price").is_between(0.001, 0.999) & (pl.col("size") > 0))
    won = (pl.col("outcome_index") == pl.col("winner_index")).cast(pl.Float64)
    is_buy = pl.col("side") == "BUY"
    return t.with_columns(
        won.alias("won"),
        pl.when(is_buy).then(pl.col("size") * (won - pl.col("price")))
        .otherwise(pl.col("size") * (pl.col("price") - won)).alias("pnl"),
        pl.when(is_buy).then(pl.col("size") * pl.col("price"))
        .otherwise(pl.col("size") * (1 - pl.col("price"))).alias("risk"),
    )


def per_market(rt: pl.DataFrame) -> pl.DataFrame:
    return rt.group_by("wallet", "condition_id").agg(
        pl.col("pnl").sum(),
        pl.col("risk").sum(),
        pl.len().alias("fills"),
        (pl.col("side") == "BUY").any().alias("has_buy"),
        (pl.col("side") == "SELL").any().alias("has_sell"),
        pl.col("outcome_index").n_unique().alias("outcomes_traded"),
        pl.col("ts").min().alias("first_ts"),
        ((pl.col("won") == 1) & (pl.col("side") == "BUY")).any().alias("bought_winner"),
        (pl.col("price") * pl.col("size")).sum().alias("notional"),
        ((pl.col("price") * pl.col("size")) * pl.col("price")).sum().alias("_pw"),
    ).with_columns(
        (pl.col("pnl") / pl.col("risk")).alias("roi"),
        (pl.col("_pw") / pl.col("notional")).alias("avg_price"),
    ).drop("_pw")


def score_wallets(rt: pl.DataFrame, cfg: WalletScoreConfig = WalletScoreConfig()) -> pl.DataFrame:
    pm = per_market(rt)
    scored = pm.group_by("wallet").agg(
        pl.len().alias("markets"),
        pl.col("fills").sum(),
        pl.col("pnl").sum(),
        pl.col("risk").sum(),
        pl.col("roi").mean().alias("mean_roi"),
        pl.col("roi").std().alias("sd_roi"),
        (pl.col("pnl") > 0).mean().alias("market_win_rate"),
        (pl.col("has_buy") & pl.col("has_sell")).mean().alias("two_sided_share"),
        (pl.col("outcomes_traded") > 1).mean().alias("both_outcomes_share"),
        pl.col("avg_price").median().alias("median_entry_price"),
    ).with_columns(
        (pl.col("pnl") / pl.col("risk")).alias("roi"),
        # Shrink the per-market mean ROI toward 0 with `prior_markets` pseudo-observations.
        (pl.col("mean_roi") * pl.col("markets") / (pl.col("markets") + cfg.prior_markets))
        .alias("shrunk_roi"),
        (pl.col("mean_roi") / (pl.col("sd_roi") / pl.col("markets").sqrt())).alias("t_stat"),
        ((pl.col("two_sided_share") > cfg.mm_two_sided)
         | (pl.col("both_outcomes_share") > cfg.mm_two_sided)).alias("mm_like"),
    )
    return scored.filter(pl.col("markets") >= cfg.min_markets).sort("shrunk_roi", descending=True)


@dataclass
class PersistenceResult:
    split: datetime
    scored_a: pl.DataFrame
    evaluated: pl.DataFrame
    spearman: float
    top_mean_roi_b: float
    all_mean_roi_b: float
    top_k: int

    def summary(self) -> str:
        return (f"split={self.split:%Y-%m-%d} wallets_scored={self.scored_a.height} "
                f"evaluated={self.evaluated.height} spearman(A,B)={self.spearman:+.3f} "
                f"top{self.top_k}_roi_B={self.top_mean_roi_b:+.3%} "
                f"all_roi_B={self.all_mean_roi_b:+.3%}")


def persistence(rt: pl.DataFrame, split: datetime, *, top_k: int = 20,
                cfg: WalletScoreConfig = WalletScoreConfig(),
                exclude_mm: bool = True) -> PersistenceResult:
    """Score on markets first traded before ``split``; evaluate on markets first traded after.

    Splitting by the wallet's first trade in a market (not by resolution date) keeps each
    market-level observation entirely on one side of the split.
    """
    first = rt.group_by("wallet", "condition_id").agg(pl.col("ts").min().alias("first_ts"))
    rt = rt.join(first, on=["wallet", "condition_id"])
    a = score_wallets(rt.filter(pl.col("first_ts") < split), cfg)
    b = score_wallets(rt.filter(pl.col("first_ts") >= split),
                      WalletScoreConfig(min_markets=5, prior_markets=cfg.prior_markets))
    if exclude_mm:
        a = a.filter(~pl.col("mm_like"))
    joined = a.select("wallet", pl.col("shrunk_roi").alias("score_a")).join(
        b.select("wallet", pl.col("roi").alias("roi_b"), pl.col("markets").alias("markets_b"),
                 pl.col("pnl").alias("pnl_b")), on="wallet", how="inner")
    if joined.height >= 3:
        spearman = float(np.corrcoef(joined["score_a"].rank().to_numpy(),
                                     joined["roi_b"].rank().to_numpy())[0, 1])
    else:
        spearman = float("nan")
    top = joined.sort("score_a", descending=True).head(top_k)
    return PersistenceResult(
        split=split, scored_a=a, evaluated=joined, spearman=spearman,
        top_mean_roi_b=float(top["pnl_b"].sum() / max(1e-9, _risk_b(rt, split, top["wallet"]))),
        all_mean_roi_b=float(joined["pnl_b"].sum() / max(1e-9, _risk_b(rt, split, joined["wallet"]))),
        top_k=top_k,
    )


def _risk_b(rt: pl.DataFrame, split: datetime, wallets: pl.Series) -> float:
    return float(rt.filter((pl.col("first_ts") >= split) & pl.col("wallet").is_in(wallets.implode()))
                 ["risk"].sum())
