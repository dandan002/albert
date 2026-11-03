"""Calibration (favorite-longshot bias) research and the calibrated-price model.

An *observation* is one market at one decision time: the executable YES ask and bid seen at
``anchor - horizon`` and the realized outcome. Using the last candle at or before that time
keeps prices strictly pre-decision.

The anchor matters. A settled Kalshi market's ``close_time`` is when it *actually* closed
(game over, word said, winner declared), which is not known in advance. Anchoring on it
leaks the event's duration (a game still running 2h before its end is more often close),
so the default anchor is ``expected_expiration_time``, fixed at listing. Markets that had
already closed by the decision time are dropped.
"""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import polars as pl

from albert.fees import fee

OBS_SCHEMA = {
    "venue": pl.String,
    "market_id": pl.String,
    "event_id": pl.String,
    "category": pl.String,
    "decision_ts": pl.Datetime("us", "UTC"),
    "resolve_ts": pl.Datetime("us", "UTC"),
    "yes_bid": pl.Float64,
    "yes_ask": pl.Float64,
    "mid": pl.Float64,
    "y": pl.Float64,
    "fee_type": pl.String,
    "fee_multiplier": pl.Float64,
}


def kalshi_observations(markets: pl.DataFrame, candles: pl.DataFrame, *, horizon_hours: float = 2.0,
                        max_spread: float = 0.10, staleness_hours: float = 6.0,
                        anchor: str = "scheduled") -> pl.DataFrame:
    """One observation per market: the last candle closing at or before anchor - horizon.

    ``anchor="scheduled"`` uses ``expected_expiration_time`` (falling back to ``close_time``
    when missing); ``anchor="start"`` uses the game start encoded in the ticker (markets
    without one are dropped), so ``horizon_hours`` is the lead before the game;
    ``anchor="close"`` reproduces the old, leaky behaviour for comparison.
    """
    if anchor == "start":
        from albert.venues.kalshi import ticker_start

        markets = markets.with_columns(pl.col("event_ticker").map_elements(
            ticker_start, return_dtype=pl.Datetime("us", "UTC")).alias("_start")).drop_nulls("_start")
        anchor_col = pl.col("_start")
    elif anchor == "scheduled" and "expected_expiration_time" in markets.columns:
        anchor_col = pl.coalesce("expected_expiration_time", "close_time")
    elif anchor in ("scheduled", "close"):
        anchor_col = pl.col("close_time")
    else:
        raise ValueError(f"unknown anchor {anchor!r}")
    m = markets.filter(pl.col("result").is_in(["yes", "no"])).with_columns(
        (anchor_col - timedelta(hours=horizon_hours)).alias("decision_ts")
    ).filter(pl.col("decision_ts") < pl.col("close_time"))
    c = candles.filter(pl.col("yes_bid_close").is_not_null() & pl.col("yes_ask_close").is_not_null()
                       & (pl.col("yes_ask_close") > 0) & (pl.col("yes_bid_close") > 0)
                       & (pl.col("yes_ask_close") < 1)).sort("end_ts")
    joined = m.sort("decision_ts").join_asof(
        c.select("ticker", "end_ts", "yes_bid_close", "yes_ask_close"),
        left_on="decision_ts", right_on="end_ts", by="ticker", strategy="backward",
        tolerance=timedelta(hours=staleness_hours),
    ).filter(pl.col("end_ts").is_not_null())
    obs = joined.select(
        pl.lit("kalshi").alias("venue"),
        pl.col("ticker").alias("market_id"),
        pl.col("event_ticker").alias("event_id"),
        "category",
        pl.col("end_ts").alias("decision_ts"),
        pl.col("close_time").alias("resolve_ts"),
        pl.col("yes_bid_close").alias("yes_bid"),
        pl.col("yes_ask_close").alias("yes_ask"),
        ((pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2).alias("mid"),
        (pl.col("result") == "yes").cast(pl.Float64).alias("y"),
        "fee_type", "fee_multiplier",
    ).filter((pl.col("yes_ask") - pl.col("yes_bid")).is_between(0, max_spread))
    return obs.cast(OBS_SCHEMA)  # type: ignore[arg-type]


def pm_observations(markets: pl.DataFrame, history: pl.DataFrame, *, horizon_hours: float = 2.0,
                    half_spread: float = 0.01, staleness_hours: float = 6.0) -> pl.DataFrame:
    """Polymarket intl observations from price history (mid only, so a half spread is assumed)."""
    m = markets.filter(pl.col("winner_index").is_not_null()).with_columns(
        pl.col("token_ids").list.get(0).alias("token_id"),
        pl.coalesce("closed_time", "end_date").alias("resolve_ts"),
    ).with_columns((pl.col("resolve_ts") - timedelta(hours=horizon_hours)).alias("decision_ts"))
    joined = m.sort("decision_ts").join_asof(
        history.sort("ts"), left_on="decision_ts", right_on="ts", by="token_id",
        strategy="backward", tolerance=timedelta(hours=staleness_hours),
    ).filter(pl.col("price").is_not_null())
    return joined.select(
        pl.lit("pm_proxy").alias("venue"),
        pl.col("condition_id").alias("market_id"),
        pl.col("event_slug").alias("event_id"),
        pl.col("tags").list.get(0, null_on_oob=True).alias("category"),
        pl.col("ts").alias("decision_ts"),
        "resolve_ts",
        (pl.col("price") - half_spread).clip(0.001, 0.999).alias("yes_bid"),
        (pl.col("price") + half_spread).clip(0.001, 0.999).alias("yes_ask"),
        pl.col("price").alias("mid"),
        (pl.col("winner_index") == 0).cast(pl.Float64).alias("y"),
        pl.lit(None, dtype=pl.String).alias("fee_type"),
        pl.lit(None, dtype=pl.Float64).alias("fee_multiplier"),
    ).cast(OBS_SCHEMA)  # type: ignore[arg-type]


def _fee_per_contract(venue: str, price: float, fee_type, mult) -> float:
    return fee(venue, 100, price, fee_type=fee_type, multiplier=mult) / 100


def with_edges(obs: pl.DataFrame) -> pl.DataFrame:
    """Realized edge of buying YES at the ask or NO at (1 - bid), after per-contract fees."""
    fee_yes = [_fee_per_contract(v, a, ft, fm) for v, a, ft, fm in
               obs.select("venue", "yes_ask", "fee_type", "fee_multiplier").iter_rows()]
    fee_no = [_fee_per_contract(v, 1 - b, ft, fm) for v, b, ft, fm in
              obs.select("venue", "yes_bid", "fee_type", "fee_multiplier").iter_rows()]
    return obs.with_columns(
        (pl.col("y") - pl.col("yes_ask") - pl.Series(fee_yes)).alias("edge_yes"),
        ((1 - pl.col("y")) - (1 - pl.col("yes_bid")) - pl.Series(fee_no)).alias("edge_no"),
    )


def bucket_expr(width: float) -> pl.Expr:
    return ((pl.col("mid") / width).floor() * width).round(4).alias("bucket")


def calibration_table(obs: pl.DataFrame, *, width: float = 0.05,
                      by_category: bool = True) -> pl.DataFrame:
    keys = (["category"] if by_category else []) + ["bucket"]
    e = with_edges(obs).with_columns(bucket_expr(width))
    return e.group_by(keys).agg(
        pl.len().alias("n"),
        pl.col("mid").mean().alias("mean_price"),
        pl.col("y").mean().alias("hit_rate"),
        pl.col("edge_yes").mean().alias("edge_yes"),
        (pl.col("edge_yes").std() / pl.len().sqrt()).alias("se_yes"),
        pl.col("edge_no").mean().alias("edge_no"),
        (pl.col("edge_no").std() / pl.len().sqrt()).alias("se_no"),
    ).with_columns(
        (pl.col("edge_yes") / pl.col("se_yes")).alias("t_yes"),
        (pl.col("edge_no") / pl.col("se_no")).alias("t_no"),
        (pl.col("hit_rate") - pl.col("mean_price")).alias("miscalibration"),
    ).sort(keys)


@dataclass
class CalibrationModel:
    """P(yes) = market mid shrunk toward the training bucket's hit rate.

    prob = (n * hit_rate + k * mid) / (n + k), per (category, bucket), falling back to the
    all-category bucket. With few observations the model just returns the market price.
    """
    table: pl.DataFrame
    overall: pl.DataFrame
    width: float
    prior: float

    @classmethod
    def fit(cls, obs: pl.DataFrame, *, width: float = 0.05, prior: float = 200.0) -> "CalibrationModel":
        return cls(calibration_table(obs, width=width, by_category=True),
                   calibration_table(obs, width=width, by_category=False), width, prior)

    def predict(self, obs: pl.DataFrame) -> pl.DataFrame:
        o = obs.with_columns(bucket_expr(self.width))
        cat = self.table.select("category", "bucket", pl.col("n").alias("n_c"),
                                pl.col("hit_rate").alias("h_c"))
        allc = self.overall.select("bucket", pl.col("n").alias("n_a"), pl.col("hit_rate").alias("h_a"))
        o = o.join(cat, on=["category", "bucket"], how="left").join(allc, on="bucket", how="left")
        n = pl.coalesce("n_c", "n_a", pl.lit(0)).cast(pl.Float64)
        h = pl.coalesce("h_c", "h_a", pl.col("mid"))
        k = self.prior
        return o.with_columns(((n * h + k * pl.col("mid")) / (n + k)).alias("prob")).drop(
            "n_c", "h_c", "n_a", "h_a")


def brier(obs: pl.DataFrame, col: str = "mid") -> float:
    return float(np.mean((obs[col].to_numpy() - obs["y"].to_numpy()) ** 2))
