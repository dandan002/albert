"""Strategies that turn research tables into backtest signals (see research/follow.SIGNAL_SCHEMA).

Every strategy is fit on a training window and emits signals only for a later test window.
"""

from datetime import datetime

import polars as pl

from albert.research.calibration import CalibrationModel
from albert.research.follow import SIGNAL_SCHEMA


def longshot_signals(train_obs: pl.DataFrame, test_obs: pl.DataFrame, *, width: float = 0.05,
                     prior: float = 200.0) -> pl.DataFrame:
    """Calibration strategy: buy whichever side the calibrated probability favors.

    The model is fit on ``train_obs`` and applied to ``test_obs``. Both sides are emitted as
    candidates and the engine's edge threshold (after fees) decides what trades.
    """
    model = CalibrationModel.fit(train_obs, width=width, prior=prior)
    p = model.predict(test_obs)
    common = [
        pl.lit("longshot").alias("strategy"), "venue", "event_id",
        pl.col("decision_ts").alias("signal_ts"), "resolve_ts",
        pl.lit(None, dtype=pl.Float64).alias("available"), "fee_type", "fee_multiplier",
    ]
    yes = p.select(*common, pl.col("market_id"), pl.lit("yes").alias("side"),
                   pl.col("yes_ask").alias("price"), pl.col("prob"), pl.col("y").alias("payoff"),
                   pl.format("bucket {} p={}", "bucket", pl.col("prob").round(3)).alias("note"))
    no = p.select(*common, pl.col("market_id"), pl.lit("no").alias("side"),
                  (1 - pl.col("yes_bid")).alias("price"), (1 - pl.col("prob")).alias("prob"),
                  (1 - pl.col("y")).alias("payoff"),
                  pl.format("bucket {} p={}", "bucket", pl.col("prob").round(3)).alias("note"))
    # Keep only the better side per market, so the engine never sees both.
    both = pl.concat([yes, no]).with_columns((pl.col("prob") - pl.col("price")).alias("_e"))
    best = both.sort("_e", descending=True).unique("market_id", keep="first").drop("_e")
    return best.select(list(SIGNAL_SCHEMA)).cast(SIGNAL_SCHEMA)  # type: ignore[arg-type]


def split_by_time(obs: pl.DataFrame, split: datetime, col: str = "decision_ts"
                  ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Train on rows resolved before the split; test on rows decided after it (no overlap)."""
    train = obs.filter(pl.col("resolve_ts") < split)
    test = obs.filter(pl.col(col) >= split)
    return train, test
