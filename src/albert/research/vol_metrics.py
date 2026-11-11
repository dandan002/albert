"""Risk-adjusted metrics for strategy backtests: Sharpe ratio and volatility on a fixed bankroll.

Every study that simulates a capped book writes a per-leg **ledger** (one row per filled position
leg) and this module turns it into a daily PnL series and the usual risk statistics, so that all
strategies are measured the same way.

Ledger schema (extra columns are ignored):

    strategy, variant, split   str      labels (split = dev / holdout / nov, or the walk-forward fold)
    struct_id                  str      shared by the legs of one multi-leg position
    event_ticker, ticker       str
    side                       str      the side held: "yes" or "no"
    n                          int      contracts
    t_in                       datetime fill time (UTC)
    cost_usd                   float    n x price paid + fee for this leg
    settle_ts                  datetime settlement (or exit) time (UTC)
    payoff_usd                 float    n x 1 if the held side won else 0 (or exit proceeds net of fees)

Two daily PnL series on the calendar from the first fill to the last settlement (idle days are 0):

* ``mtm`` (primary): each open leg is marked at the end of every UTC day at the hourly-candle mid
  of the held side (last candle at or before the day's end; the entry price if there is none yet);
  on its settlement day it is worth ``payoff_usd``. Daily PnL = change in value, entry day measured
  from ``cost_usd`` (so spread paid and fees hit on day one). This is the series volatility and
  Sharpe are computed from.
* ``settle``: PnL booked only on the settlement day. Lumpier; shown as a check.

Statistics (bankroll B = $100, so $ and % of bankroll coincide):

* annualised return = mean daily PnL x 365 / B; annualised volatility = sd(daily) x sqrt(365) / B;
  **Sharpe = mean / sd x sqrt(365)** (prediction markets trade every day; no risk-free rate, since
  idle cash at Kalshi earns nothing in this backtest);
* the weekly version (7-day sums, x sqrt(52)) as a check against daily autocorrelation in marks;
* Sortino (downside deviation), maximum drawdown of the cumulative MTM equity, skew, kurtosis,
  share of positive weeks;
* a 90% moving-block bootstrap interval for the Sharpe ratio (7-day blocks) and P(Sharpe > 0);
* Sharpe without the five events with the largest total PnL (concentration check).

Run: ``python -m albert.research.vol_metrics run [LEDGER_GLOB ...]`` (default: every
``data/research/*/ledgers/ledger_*.parquet`` and ``data/research/*/ledger_*.parquet``); writes
``data/research/vol_metrics/metrics.{csv,md}``.
"""

from __future__ import annotations

import argparse
import glob
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

BANKROLL = 100.0
DAYS_PER_YEAR = 365
BLOCK_DAYS = 7
N_BOOT = 2000
CANDLE_SOURCES = (
    "data/research/pm_hazard/candles.parquet",
    "data/research/vol_theta/candles_ext.parquet",
    "data/research/pm_smile/candles_fetched.parquet",
    "data/tables/kalshi_candles.parquet",
)
LEDGER_COLS = ["struct_id", "event_ticker", "ticker", "side", "n", "t_in", "cost_usd", "settle_ts", "payoff_usd"]


# ---- marks ---------------------------------------------------------------------------------------
def load_marks(tickers: list[str], root: Path = Path(".")) -> pl.DataFrame:
    """Hourly YES mids (ticker, ts, yes_mid) for ``tickers`` from every cached candle source: the
    bid/ask midpoint when both sides are quoted, else the last trade price."""
    frames = []
    for src in CANDLE_SOURCES:
        p = root / src
        if not p.exists():
            continue
        frames.append(pl.scan_parquet(p).filter(pl.col("ticker").is_in(tickers))
                      .select("ticker", pl.col("end_ts").alias("ts"), "yes_bid_close", "yes_ask_close", "price_close"))
    if not frames:
        return pl.DataFrame(schema={"ticker": pl.String, "ts": pl.Datetime("us", "UTC"), "yes_mid": pl.Float64})
    c = pl.concat(frames, how="diagonal_relaxed").collect()
    two_sided = (pl.col("yes_bid_close").is_not_null() & pl.col("yes_ask_close").is_not_null()
                 & (pl.col("yes_ask_close") > pl.col("yes_bid_close")))
    mid = pl.when(two_sided).then((pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2).otherwise(pl.col("price_close"))
    return (c.with_columns(mid.alias("yes_mid")).filter(pl.col("yes_mid").is_not_null())
            .select("ticker", pl.col("ts").cast(pl.Datetime("us", "UTC")), "yes_mid")
            .unique(["ticker", "ts"]).sort("ticker", "ts"))


# ---- daily PnL -----------------------------------------------------------------------------------
def _calendar(led: pl.DataFrame) -> list[date]:
    d0 = led["t_in"].min().date()
    d1 = led["settle_ts"].max().date()
    return [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]


def _on_calendar(daily: pl.DataFrame, cal: list[date]) -> np.ndarray:
    full = pl.DataFrame({"day": cal}).join(daily, on="day", how="left")
    return full["pnl"].fill_null(0.0).to_numpy()


def settle_daily(led: pl.DataFrame, cal: list[date] | None = None) -> np.ndarray:
    """PnL booked on each leg's settlement day."""
    cal = cal or _calendar(led)
    d = (led.group_by(pl.col("settle_ts").dt.date().alias("day"))
         .agg((pl.col("payoff_usd") - pl.col("cost_usd")).sum().alias("pnl")))
    return _on_calendar(d, cal)


def mtm_values(led: pl.DataFrame, marks: pl.DataFrame) -> pl.DataFrame:
    """One row per (leg, day it is held): end-of-day value of the leg, and the day's PnL."""
    x = led.select(LEDGER_COLS).with_row_index("leg").with_columns(
        (pl.col("cost_usd") / pl.col("n")).alias("unit_in"),
        pl.date_ranges(pl.col("t_in").dt.date(), pl.col("settle_ts").dt.date()).alias("day")).explode("day")
    x = x.with_columns((pl.col("day").cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
                        + timedelta(days=1)).alias("eod")).sort("eod")
    m = marks.sort("ts")
    x = x.join_asof(m, left_on="eod", right_on="ts", by="ticker", strategy="backward", check_sortedness=False)
    held = pl.when(pl.col("side") == "yes").then(pl.col("yes_mid")).otherwise(1 - pl.col("yes_mid"))
    # a mark from before the fill says nothing about the position: use the entry price until one exists
    unit = pl.when(pl.col("ts").is_not_null() & (pl.col("ts") >= pl.col("t_in"))).then(held).otherwise(pl.col("unit_in"))
    last = pl.col("day") == pl.col("settle_ts").dt.date()
    x = x.with_columns(pl.when(last).then(pl.col("payoff_usd")).otherwise(pl.col("n") * unit).alias("value"),
                       (last | (pl.col("ts").is_not_null() & (pl.col("ts") >= pl.col("t_in")))).alias("marked"))
    x = x.sort("leg", "day").with_columns(
        (pl.col("value") - pl.col("value").shift(1).over("leg").fill_null(pl.col("cost_usd"))).alias("pnl"))
    return x.select("leg", "struct_id", "event_ticker", "ticker", "day", "value", "pnl", "marked")


def mtm_daily(led: pl.DataFrame, marks: pl.DataFrame, cal: list[date] | None = None) -> np.ndarray:
    cal = cal or _calendar(led)
    v = mtm_values(led, marks)
    return _on_calendar(v.group_by("day").agg(pl.col("pnl").sum()), cal)


# ---- statistics ----------------------------------------------------------------------------------
def sharpe(x: np.ndarray, per_year: float = DAYS_PER_YEAR) -> float | None:
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return None if sd <= 0 else float(np.mean(x)) / sd * math.sqrt(per_year)


def weekly(x: np.ndarray) -> np.ndarray:
    k = len(x) // BLOCK_DAYS
    return x[: k * BLOCK_DAYS].reshape(k, BLOCK_DAYS).sum(axis=1) if k else np.array([])


def block_bootstrap(x: np.ndarray, *, block: int = BLOCK_DAYS, reps: int = N_BOOT, seed: int = 7) -> np.ndarray:
    """Moving-block bootstrap draws of the annualised Sharpe ratio."""
    n = len(x)
    if n < 2 * block:
        return np.array([])
    rng = np.random.default_rng(seed)
    nb = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    s = x[idx]
    sd = s.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = s.mean(axis=1) / sd * math.sqrt(DAYS_PER_YEAR)
    return out[np.isfinite(out)]


def max_drawdown(x: np.ndarray) -> float:
    eq = np.concatenate([[0.0], np.cumsum(x)])
    return float((eq - np.maximum.accumulate(eq)).min())


def _r(v: Any, k: int = 2) -> Any:
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), k)


def series_stats(x: np.ndarray, bankroll: float = BANKROLL) -> dict[str, Any]:
    n = len(x)
    mu = float(np.mean(x)) if n else 0.0
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    down = float(np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))) if n else 0.0
    w = weekly(x)
    boot = block_bootstrap(x)
    dd = max_drawdown(x)
    ann_ret = mu * DAYS_PER_YEAR / bankroll
    z = (x - mu) / sd if sd > 0 else np.zeros_like(x)
    return {
        "days": n, "pnl_usd": _r(x.sum()), "usd_per_week": _r(mu * 7, 3),
        "ann_return_pct": _r(100 * ann_ret, 1), "ann_vol_pct": _r(100 * sd * math.sqrt(DAYS_PER_YEAR) / bankroll, 1),
        "daily_vol_usd": _r(sd, 3), "sharpe": _r(sharpe(x)),
        "sharpe_ci90": [_r(np.quantile(boot, 0.05)), _r(np.quantile(boot, 0.95))] if len(boot) else None,
        "p_sharpe_pos": _r(float((boot > 0).mean()), 3) if len(boot) else None,
        "sharpe_weekly": _r(sharpe(w, 52)) if len(w) > 2 else None,
        "ann_vol_weekly_pct": _r(100 * float(np.std(w, ddof=1)) * math.sqrt(52) / bankroll, 1) if len(w) > 2 else None,
        "sortino": _r(mu / down * math.sqrt(DAYS_PER_YEAR)) if down > 0 else None,
        "max_dd_usd": _r(dd), "calmar": _r(ann_ret * bankroll / -dd) if dd < 0 else None,
        "worst_day_usd": _r(x.min()) if n else None, "worst_week_usd": _r(w.min()) if len(w) else None,
        "weeks_positive_pct": _r(100 * float((w > 0).mean()), 0) if len(w) else None,
        "skew": _r(float(np.mean(z ** 3)), 2) if n > 2 else None,
        "ex_kurt": _r(float(np.mean(z ** 4)) - 3, 1) if n > 3 else None,
    }


def ledger_metrics(led: pl.DataFrame, marks: pl.DataFrame, bankroll: float = BANKROLL) -> dict[str, Any]:
    """MTM statistics (primary), the settlement-basis Sharpe, and the top-5-events concentration check."""
    cal = _calendar(led)
    v = mtm_values(led, marks)
    x = _on_calendar(v.group_by("day").agg(pl.col("pnl").sum()), cal)
    out = series_stats(x, bankroll)
    xs = settle_daily(led, cal)
    out["sharpe_settle_basis"] = _r(sharpe(xs))
    out["ann_vol_settle_basis_pct"] = _r(100 * float(np.std(xs, ddof=1)) * math.sqrt(DAYS_PER_YEAR) / bankroll, 1)
    ev = led.group_by("event_ticker").agg((pl.col("payoff_usd") - pl.col("cost_usd")).sum().alias("p")).sort("p", descending=True)
    top = ev["event_ticker"].head(5).to_list()
    x5 = _on_calendar(v.filter(~pl.col("event_ticker").is_in(top)).group_by("day").agg(pl.col("pnl").sum()), cal)
    out["sharpe_ex_top5"] = _r(sharpe(x5))
    out["usd_per_week_ex_top5"] = _r(float(np.mean(x5)) * 7, 3)
    out["positions"] = led["struct_id"].n_unique()
    out["events"] = led["event_ticker"].n_unique()
    out["mean_open_mtm_usd"] = _r(float(v.filter(pl.col("day") < pl.col("day").max().over("leg"))["value"].sum()) / len(cal))
    out["first_day"], out["last_day"] = str(cal[0]), str(cal[-1])
    out["marked_share"] = _r(float(v["marked"].mean()), 3)
    return out


# ---- driver --------------------------------------------------------------------------------------
DEFAULT_GLOBS = ("data/research/*/ledgers/ledger_*.parquet", "data/research/*/ledger_*.parquet")
SHOW = ["strategy", "variant", "positions", "events", "usd_per_week", "ann_return_pct", "ann_vol_pct", "sharpe",
        "sharpe_ci90", "p_sharpe_pos", "sharpe_weekly", "sortino", "max_dd_usd", "worst_week_usd",
        "weeks_positive_pct", "sharpe_ex_top5", "sharpe_settle_basis", "skew", "ex_kurt"]


def run(patterns: list[str], out: Path) -> pl.DataFrame:
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    rows = []
    for p in paths:
        led = pl.read_parquet(p)
        if not led.height:
            continue
        marks = load_marks(led["ticker"].unique().to_list())
        m = ledger_metrics(led, marks)
        rows.append({"study": Path(p).parts[-3] if "ledgers" in p else Path(p).parts[-2],
                     "strategy": led["strategy"][0] if "strategy" in led.columns else Path(p).stem,
                     "variant": led["variant"][0] if "variant" in led.columns else "", "file": p, **m})
        print(f"{p}: Sharpe {m['sharpe']} (CI90 {m['sharpe_ci90']}), vol {m['ann_vol_pct']}%, "
              f"${m['usd_per_week']}/wk", flush=True)
    df = pl.DataFrame(rows, infer_schema_length=None)
    out.mkdir(parents=True, exist_ok=True)
    flat = df.with_columns(pl.col("sharpe_ci90").map_elements(lambda v: f"[{v[0]}, {v[1]}]" if v is not None else "",
                                                              return_dtype=pl.String))
    flat.write_csv(out / "metrics.csv")
    cols = ["study"] + [c for c in SHOW if c in flat.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in flat.select(cols).iter_rows():
        lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    (out / "metrics.md").write_text(
        "# Strategy risk metrics (MTM daily PnL on a $100 bankroll; Sharpe annualised x sqrt(365))\n\n"
        + "\n".join(lines) + "\n")
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.vol_metrics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("globs", nargs="*", default=list(DEFAULT_GLOBS))
    r.add_argument("--out", default="data/research/vol_metrics")
    a = ap.parse_args(argv)
    run(a.globs, Path(a.out))


if __name__ == "__main__":
    main()
