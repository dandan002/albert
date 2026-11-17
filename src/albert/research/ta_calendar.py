"""Naive technical-analysis factors and day-of-week effects on settled Kalshi markets.

Question (user, 2026-11-17): do textbook chart signals (RSI, moving-average crossovers, Bollinger
bands, Donchian breakouts, MACD, volume / open-interest confirmation) or the calendar (weekend vs
weekday, US Eastern time) predict Kalshi settlement well enough to pay the taker spread and fee?

Three samples, each used only for what its selection allows:

* **TA and decision-day calibration:** hourly candles of the volume-blind sports and weather series
  (``momentum_hist.VOLUME_BLIND_SERIES_PREFIXES``: every event, or a seeded random set; closes
  2026-08-15..10-31). Markets picked by final volume are excluded, because final volume is not
  known at the decision. ``universe="all"`` adds them as a flagged check.
* **Weekend drift:** the long-dated "by date T" contracts of ``pm_hazard`` (picked by rules text from
  a full scan of settled non-financial markets; politics, mentions, entertainment, ...; Jan-Nov 2026).
* **Taker flow:** the trade-tape windows in ``kalshi_trades`` (top-volume markets: flagged).

Execution: taker at the hourly candle's closing quote (YES at ``yes_ask_close``, NO at
``1 - yes_bid_close``), a ~$1 order of ``n = max(1, floor(1 / price))`` contracts, Kalshi's taker fee
rounded up to the cent per order, held to settlement. Only quotes with spread <= 10c are taken.

No look-ahead: candles go on a regular hourly grid per market (Kalshi omits an hour in which nothing
changed; the quote and open interest are carried forward, volume is 0 there). Every indicator at
hour t uses grid hours <= t, and a decision at t trades the quote of the candle ending at t. Candles
at or after the market's actual close are dropped (whether a market still trades is observable at
t). Time to expiration is measured to ``expected_expiration_time``. Dev = scheduled expiration before
2026-10-01, holdout = October. Standard errors are clustered by match (``momentum_hist.match_key``);
calendar effects are also clustered by ET date, because a weekday effect is a common shock.

    python -m albert.research.ta_calendar ta|calendar|drift|flow|ledgers|all
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.fees import KALSHI_TAKER
from albert.research.momentum_hist import (
    HOLDOUT_FROM,
    VOLUME_BLIND_SERIES_PREFIXES,
    h_bucket,
    kind_of,
    match_key,
)

ET = "America/New_York"
OUT = Path("data/research/ta_calendar")
TABLES = Path("data/tables")
HAZARD = Path("data/research/pm_hazard")
MAX_TAKE_SPREAD = 0.10        # only cross a quote this tight
MAX_SIGNAL_SPREAD = 0.20      # a mid used as a signal must come from a quote this tight
FFILL_LIMIT_H = 6             # a signal mid survives this many hours of wide quotes
PRICE_BAND = (0.05, 0.95)     # TA entries only at mids inside this band
STAKE = 1.0
BANKROLL = 100.0
EVENT_CAP = 3.0               # open cost per event in the capped book
DOW_NAMES = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}


# ---- panel ---------------------------------------------------------------------------------------
def load_markets(tables: Path = TABLES) -> pl.DataFrame:
    return (pl.scan_parquet(tables / "kalshi_markets.parquet")
            .filter(pl.col("result").is_in(["yes", "no"]) & (pl.col("market_type") == "binary"))
            .select("ticker", "event_ticker", "series_ticker", "category", "result", "close_time",
                    "settlement_ts", "expected_expiration_time", "fee_type", "fee_multiplier")
            .collect())


def hourly_grid(c: pl.DataFrame) -> pl.DataFrame:
    """``(ticker, t, bid, ask, volume, oi)`` on a regular hourly grid from each market's first to
    last candle. ``real`` marks hours that had a candle."""
    g = (c.sort("ticker", "t").with_columns(pl.lit(True).alias("real"))
         .upsample("t", every="1h", group_by="ticker", maintain_order=True))
    return g.with_columns(pl.col("bid", "ask", "oi").forward_fill().over("ticker"),
                          pl.col("volume").fill_null(0.0), pl.col("real").fill_null(False))


def with_quote_cols(p: pl.DataFrame) -> pl.DataFrame:
    """mid / spread / signal mid ``smid`` / carried signal mid ``x`` plus ET calendar columns."""
    two = (pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("ask") > pl.col("bid"))
    p = p.with_columns(pl.when(two).then((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                       (pl.col("ask") - pl.col("bid")).alias("spread"))
    p = p.with_columns(pl.when(pl.col("spread") <= MAX_SIGNAL_SPREAD).then(pl.col("mid")).alias("smid"))
    et = pl.col("t").dt.convert_time_zone(ET)
    return p.with_columns(pl.col("smid").forward_fill(limit=FFILL_LIMIT_H).over("ticker").alias("x"),
                          et.dt.weekday().alias("dow"), et.dt.hour().alias("et_hour"), et.dt.date().alias("et_date"),
                          (et.dt.weekday() >= 6).alias("weekend"))


def build_panel(candles: pl.DataFrame, markets: pl.DataFrame,
                prefixes: tuple[str, ...] | None = VOLUME_BLIND_SERIES_PREFIXES) -> pl.DataFrame:
    m = markets
    if prefixes is not None:
        m = m.filter(pl.any_horizontal([pl.col("series_ticker").str.starts_with(x) for x in prefixes]))
    c = (candles.select("ticker", pl.col("end_ts").alias("t"), pl.col("yes_bid_close").alias("bid"),
                        pl.col("yes_ask_close").alias("ask"), "volume", pl.col("open_interest").alias("oi"))
         .join(m.select("ticker", "close_time"), on="ticker")
         .filter(pl.col("t") < pl.col("close_time")).drop("close_time"))
    p = hourly_grid(c).join(m.drop("close_time"), on="ticker")
    ev = m.select("event_ticker").unique().with_columns(
        pl.col("event_ticker").map_elements(match_key, return_dtype=pl.String).alias("cluster"))
    p = p.join(ev, on="event_ticker").with_columns(
        (pl.col("result") == "yes").cast(pl.Float64).alias("y"),
        ((pl.col("expected_expiration_time") - pl.col("t")).dt.total_minutes() / 60).alias("h_to_exp"),
        (pl.col("expected_expiration_time") >= HOLDOUT_FROM).alias("holdout"),
        kind_of(pl.col("series_ticker")).alias("kind"),
    ).filter(pl.col("expected_expiration_time").is_not_null() & (pl.col("h_to_exp") > 0))
    p = p.with_columns(h_bucket(pl.col("h_to_exp")).alias("hb"))
    return with_quote_cols(p.sort("ticker", "t"))


def prepare(universe: str = "blind", tables: Path = TABLES) -> pl.DataFrame:
    prefixes = VOLUME_BLIND_SERIES_PREFIXES if universe == "blind" else None
    return build_panel(pl.read_parquet(tables / "kalshi_candles.parquet"), load_markets(tables), prefixes)


# ---- indicators ----------------------------------------------------------------------------------
def add_indicators(p: pl.DataFrame) -> pl.DataFrame:
    """Classic indicators on the carried signal mid ``x`` (per market, hours <= t only)."""
    x, T = pl.col("x"), "ticker"
    p = p.with_columns(x.diff().over(T).alias("d"),
                       (x - x.shift(6).over(T)).alias("r6"),
                       pl.col("volume").rolling_sum(6, min_samples=6).over(T).alias("v6"),
                       (pl.col("oi") - pl.col("oi").shift(6).over(T)).alias("doi6"))
    d = pl.col("d")
    p = p.with_columns(d.clip(lower_bound=0).alias("up"), (-d).clip(lower_bound=0).alias("dn"),
                       pl.col("v6").shift(6).rolling_mean(48, min_samples=12).over(T).alias("vbase"))
    cols = []
    for n in (6, 14):                                     # Wilder RSI; "act" = total movement in window
        ag = pl.col("up").ewm_mean(alpha=1 / n, adjust=False, min_samples=n).over(T)
        al = pl.col("dn").ewm_mean(alpha=1 / n, adjust=False, min_samples=n).over(T)
        cols += [(100 * ag / (ag + al)).alias(f"rsi{n}"), d.abs().rolling_sum(n, min_samples=n).over(T).alias(f"act{n}")]
    for f, s in ((3, 12), (6, 24)):
        cols.append((x.ewm_mean(span=f, adjust=False, min_samples=s).over(T)
                     - x.ewm_mean(span=s, adjust=False, min_samples=s).over(T)).alias(f"ema{f}_{s}"))
    for n in (12, 24):
        sd = x.rolling_std(n, min_samples=n).over(T)
        cols.append(pl.when(sd >= 0.01).then((x - x.rolling_mean(n, min_samples=n).over(T)) / sd).alias(f"bz{n}"))
    for n in (12, 24, 48):
        cols += [x.shift(1).rolling_max(n, min_samples=n).over(T).alias(f"hi{n}"),
                 x.shift(1).rolling_min(n, min_samples=n).over(T).alias(f"lo{n}")]
    cols.append((x.ewm_mean(span=12, adjust=False, min_samples=26).over(T)
                 - x.ewm_mean(span=26, adjust=False, min_samples=26).over(T)).alias("macd"))
    p = p.with_columns(cols)
    p = p.with_columns((pl.col("macd") - pl.col("macd").ewm_mean(span=9, adjust=False, min_samples=9).over(T)).alias("mh"))
    lag = [pl.col(c).shift(1).over(T).alias(f"{c}_p") for c in ("ema3_12", "ema6_24", "mh")]
    return p.with_columns(lag)


def _cross(col: str, band: float) -> tuple[pl.Expr, pl.Expr]:
    c, cp = pl.col(col), pl.col(f"{col}_p")
    return (c > band) & (cp <= band), (c < -band) & (cp >= -band)


def ta_signals() -> dict[str, tuple[pl.Expr, pl.Expr]]:
    """name -> (up condition, down condition). "Trend" buys YES on up; "fade" buys NO on up."""
    r6 = pl.col("r6")
    sig: dict[str, tuple[pl.Expr, pl.Expr]] = {"tsmom6 5c": (r6 >= 0.05, r6 <= -0.05)}   # benchmark (momentum_hist)
    for n in (6, 14):
        act = pl.col(f"act{n}") >= 0.03                   # RSI is meaningless on a flat quote
        for hi in (70, 80):
            sig[f"rsi{n} {hi}/{100 - hi}"] = (act & (pl.col(f"rsi{n}") >= hi), act & (pl.col(f"rsi{n}") <= 100 - hi))
    sig["ema 3/12 cross"] = _cross("ema3_12", 0.005)
    sig["ema 6/24 cross"] = _cross("ema6_24", 0.005)
    for n in (12, 24):
        sig[f"bollinger{n} 2sd"] = (pl.col(f"bz{n}") >= 2, pl.col(f"bz{n}") <= -2)
    for n in (12, 24, 48):
        sig[f"donchian{n} +2c"] = (pl.col("x") >= pl.col(f"hi{n}") + 0.02, pl.col("x") <= pl.col(f"lo{n}") - 0.02)
    sig["macd hist cross"] = _cross("mh", 0.0025)
    big = r6.abs() >= 0.05
    hv = (pl.col("v6") > 0) & (pl.col("v6") >= 2 * pl.col("vbase"))
    lv = pl.col("v6") <= 0.5 * pl.col("vbase")
    sig["move6 5c high volume"] = (big & hv & (r6 > 0), big & hv & (r6 < 0))
    sig["move6 5c low volume"] = (big & lv & (r6 > 0), big & lv & (r6 < 0))
    sig["move6 5c OI rising"] = (big & (pl.col("doi6") > 0) & (r6 > 0), big & (pl.col("doi6") > 0) & (r6 < 0))
    sig["move6 5c OI falling"] = (big & (pl.col("doi6") < 0) & (r6 > 0), big & (pl.col("doi6") < 0) & (r6 < 0))
    return sig


# ---- execution and scoring -----------------------------------------------------------------------
def taker_fee_expr(n: pl.Expr, price: pl.Expr) -> pl.Expr:
    """Kalshi taker fee for one order of n contracts, rounded up to the cent (``fees.kalshi_fee``)."""
    raw = KALSHI_TAKER * pl.col("fee_multiplier").fill_null(1.0) * n * price * (1 - price)
    return (raw * 100).round(9).ceil() / 100


def take(f: pl.DataFrame, direction: pl.Expr) -> pl.DataFrame:
    """Rows with ``direction`` +1 (buy YES at the ask) or -1 (buy NO at 1 - bid), as a ~$1 taker
    order held to settlement. Per-contract columns: price, fee_c, pnl, risk, gross (vs the mid)."""
    f = f.with_columns(direction.alias("dir")).filter(
        pl.col("dir").is_in([1, -1]) & (pl.col("bid") > 0) & (pl.col("ask") < 1)
        & (pl.col("spread") > 0) & (pl.col("spread") <= MAX_TAKE_SPREAD))
    f = f.with_columns(
        pl.when(pl.col("dir") == 1).then(pl.col("ask")).otherwise(1 - pl.col("bid")).alias("price"),
        pl.when(pl.col("dir") == 1).then(pl.col("y")).otherwise(1 - pl.col("y")).alias("payoff"),
        pl.when(pl.col("dir") == 1).then(pl.col("mid")).otherwise(1 - pl.col("mid")).alias("mid_side"))
    f = f.with_columns(pl.max_horizontal(pl.lit(1.0), (STAKE / pl.col("price")).floor()).alias("n"))
    f = f.with_columns(taker_fee_expr(pl.col("n"), pl.col("price")).alias("fee_order"))
    return f.with_columns(
        (pl.col("fee_order") / pl.col("n")).alias("fee_c"),
        (pl.col("payoff") - pl.col("price") - pl.col("fee_order") / pl.col("n")).alias("pnl"),
        (pl.col("price") + pl.col("fee_order") / pl.col("n")).alias("risk"),
        (pl.col("payoff") - pl.col("mid_side")).alias("gross"))


def first_per_market(f: pl.DataFrame) -> pl.DataFrame:
    return f.sort("t").unique("ticker", keep="first", maintain_order=True)


def cl_mean(f: pl.DataFrame, col: str, cluster: str, weight: str | None = None) -> tuple[float | None, float | None, int]:
    """Ratio mean sum(col)/sum(weight) (weight = rows if None) with a cluster-robust t."""
    if f.is_empty():
        return None, None, 0
    w = pl.col(weight).sum() if weight else pl.len().cast(pl.Float64)
    g = f.group_by(cluster).agg(pl.col(col).sum().alias("s"), w.alias("w"))
    k, tot = g.height, g["w"].sum()
    mean = g["s"].sum() / tot
    if k < 2:
        return mean, None, k
    se = math.sqrt(k / (k - 1) * ((g["s"] - mean * g["w"]) ** 2).sum()) / tot
    return mean, (mean / se if se > 0 else None), k


def _r(v: Any, k: int = 2) -> Any:
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), k)


def score(f: pl.DataFrame, by_date: bool = False, **labels: Any) -> list[dict[str, Any]]:
    rows = []
    for split, sub in (("dev", f.filter(~pl.col("holdout"))), ("holdout", f.filter(pl.col("holdout")))):
        g, tg, _ = cl_mean(sub, "gross", "cluster")
        c, tc, k = cl_mean(sub, "pnl", "cluster")
        d, td, _ = cl_mean(sub, "pnl", "cluster", weight="risk")
        row = {**labels, "split": split, "trades": sub.height, "clusters": k,
               "avg_price": _r(sub["price"].mean(), 3) if sub.height else None,
               "gross_c": _r(None if g is None else 100 * g), "t_gross": _r(tg),
               "net_c": _r(None if c is None else 100 * c), "t_net": _r(tc),
               "net_per_$": _r(d, 4), "t_$": _r(td), "hit": _r(sub["payoff"].mean(), 3) if sub.height else None}
        if by_date:
            _, tgd, kd = cl_mean(sub, "gross", "et_date")
            _, tdd, _ = cl_mean(sub, "pnl", "et_date", weight="risk")
            row |= {"dates": kd, "t_gross_by_date": _r(tgd), "t_$_by_date": _r(tdd)}
        rows.append(row)
    return rows


# ---- TA sweep ------------------------------------------------------------------------------------
def entry_ok(p: pl.DataFrame) -> pl.DataFrame:
    """Rows where a TA trade may be entered: a fresh, takeable quote inside the price band."""
    return p.filter(pl.col("smid").is_not_null() & pl.col("mid").is_between(*PRICE_BAND)
                    & (pl.col("spread") <= MAX_TAKE_SPREAD))


def ta_trades(p: pl.DataFrame) -> pl.DataFrame:
    """First signal per market for every (signal, mode), as one frame of taker trades."""
    q = entry_ok(p)
    frames = []
    for name, (up, dn) in ta_signals().items():
        d = pl.when(up).then(1).when(dn).then(-1).otherwise(0)
        s = first_per_market(q.with_columns(d.alias("sig")).filter(pl.col("sig") != 0))
        for mode, sign in (("trend", 1), ("fade", -1)):
            frames.append(take(s, pl.col("sig") * sign).with_columns(pl.lit(name).alias("signal"),
                                                                    pl.lit(mode).alias("mode")))
    keep = ["signal", "mode", "ticker", "event_ticker", "series_ticker", "cluster", "kind", "hb", "t", "et_date",
            "dow", "weekend", "holdout", "dir", "mid", "spread", "price", "n", "fee_order", "fee_c", "payoff", "pnl",
            "risk", "gross", "settlement_ts", "h_to_exp"]
    return pl.concat([f.select(keep) for f in frames])


def ta_table(tr: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (name, mode), f in tr.group_by(["signal", "mode"], maintain_order=True):
        rows += score(f, signal=name, mode=mode, cut="all")
    return pl.DataFrame(rows, infer_schema_length=None)


def ta_cuts(tr: pl.DataFrame) -> pl.DataFrame:
    """Trend mode by kind and by hours-to-expiration bucket (fade is the mirror image gross)."""
    rows = []
    t = tr.filter(pl.col("mode") == "trend")
    for col in ("kind", "hb"):
        for (name, val), f in t.group_by(["signal", col], maintain_order=True):
            rows += score(f, signal=name, mode="trend", cut=f"{col}={val}")
    return pl.DataFrame(rows, infer_schema_length=None).sort("signal", "cut", "split")


# ---- calendar: calibration by ET weekday of the decision -----------------------------------------
PRICE_BUCKETS = ((0.03, 0.20, "03-20c"), (0.20, 0.50, "20-50c"), (0.50, 0.80, "50-80c"), (0.80, 0.97, "80-97c"))


def price_bucket(e: pl.Expr) -> pl.Expr:
    out = pl.lit(None, dtype=pl.String)
    for lo, hi, name in reversed(PRICE_BUCKETS):
        out = pl.when(e.is_between(lo, hi, closed="left")).then(pl.lit(name)).otherwise(out)
    return out


def snapshots(p: pl.DataFrame, et_hours: tuple[int, ...] = (12,)) -> pl.DataFrame:
    """Every market-day at the given ET hours with a takeable quote and mid in [3c, 97c]."""
    return p.filter(pl.col("et_hour").is_in(et_hours) & pl.col("smid").is_not_null()
                    & (pl.col("spread") <= MAX_TAKE_SPREAD) & pl.col("mid").is_between(0.03, 0.97)
                    ).with_columns(price_bucket(pl.col("mid")).alias("pb"), (pl.col("y") - pl.col("mid")).alias("resid"))


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and cluster-robust (CR1) standard errors."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    _, inv = np.unique(groups, return_inverse=True)
    G = inv.max() + 1
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, inv, X * e[:, None])
    n, k = X.shape
    c = G / (G - 1) * (n - 1) / (n - k)
    V = c * XtX_inv @ (S.T @ S) @ XtX_inv
    return b, np.sqrt(np.clip(np.diag(V), 0.0, None))   # a singleton FE cell can give -1e-17


def weekend_gap(s: pl.DataFrame, col: str = "resid") -> dict[str, Any]:
    """``col`` on weekend minus weekday decisions, with t clustered by ET date and by match."""
    if s.height < 20 or s["weekend"].n_unique() < 2:
        return {"n": s.height}
    y = s[col].to_numpy().astype(float)
    X = np.column_stack([np.ones(s.height), s["weekend"].cast(pl.Float64).to_numpy()])
    b, se_d = ols_cluster(y, X, s["et_date"].cast(pl.String).to_numpy())
    _, se_m = ols_cluster(y, X, s["cluster"].to_numpy())
    return {"n": s.height, "weekday_c": _r(100 * b[0]), "weekend_minus_weekday_c": _r(100 * b[1]),
            "t_by_date": _r(b[1] / se_d[1]) if se_d[1] > 0 else None,
            "t_by_match": _r(b[1] / se_m[1]) if se_m[1] > 0 else None}


def calendar_tables(p: pl.DataFrame) -> dict[str, pl.DataFrame]:
    s = snapshots(p, (12, 20))
    out: dict[str, pl.DataFrame] = {}
    # residual (y - mid) by weekday x price bucket, per split; t by date
    rows = []
    for split, sub in (("dev", s.filter(~pl.col("holdout"))), ("holdout", s.filter(pl.col("holdout"))), ("all", s)):
        for dow in range(1, 8):
            for pb in [None] + [b[2] for b in PRICE_BUCKETS]:
                f = sub.filter(pl.col("dow") == dow)
                if pb:
                    f = f.filter(pl.col("pb") == pb)
                m, td, kd = cl_mean(f, "resid", "et_date")
                _, tm, km = cl_mean(f, "resid", "cluster")
                rows.append({"split": split, "dow": DOW_NAMES[dow], "bucket": pb or "all", "snapshots": f.height,
                             "dates": kd, "matches": km, "avg_mid": _r(f["mid"].mean(), 3) if f.height else None,
                             "resid_c": _r(None if m is None else 100 * m), "t_by_date": _r(td), "t_by_match": _r(tm)})
    out["resid_by_dow"] = pl.DataFrame(rows, infer_schema_length=None)
    # weekend - weekday gap, overall and within price bucket / kind / ET hour
    rows = []
    for split, sub in (("dev", s.filter(~pl.col("holdout"))), ("holdout", s.filter(pl.col("holdout"))), ("all", s)):
        cuts = [("all", sub)] + [(f"bucket={b[2]}", sub.filter(pl.col("pb") == b[2])) for b in PRICE_BUCKETS]
        cuts += [(f"kind={k}", sub.filter(pl.col("kind") == k)) for k in sorted(sub["kind"].unique().to_list())]
        cuts += [(f"et_hour={h}", sub.filter(pl.col("et_hour") == h)) for h in (12, 20)]
        cuts += [(f"hb={h}", sub.filter(pl.col("hb") == h)) for h in ("0-3h", "3-12h", "12-48h", ">48h")]
        for name, f in cuts:
            rows.append({"split": split, "cut": name, **weekend_gap(f)})
    out["weekend_gap"] = pl.DataFrame(rows, infer_schema_length=None)
    # favourite-longshot slope by weekend: resid = a + b * (mid - 0.5), separately on weekend / weekday
    rows = []
    for split, sub in (("dev", s.filter(~pl.col("holdout"))), ("holdout", s.filter(pl.col("holdout"))), ("all", s)):
        for wk in (False, True):
            f = sub.filter(pl.col("weekend") == wk)
            X = np.column_stack([np.ones(f.height), f["mid"].to_numpy() - 0.5])
            b, se = ols_cluster(f["resid"].to_numpy(), X, f["et_date"].cast(pl.String).to_numpy())
            rows.append({"split": split, "weekend": wk, "n": f.height, "resid_at_50c_c": _r(100 * b[0]),
                         "slope_c_per_10c": _r(10 * b[1]), "t_slope_by_date": _r(b[1] / se[1])})
    out["flb_slope"] = pl.DataFrame(rows)
    return out


def calendar_trades(p: pl.DataFrame) -> pl.DataFrame:
    """Taker rules: at the first noon-ET snapshot of each market, buy YES or NO in a price bucket,
    labelled by the snapshot's weekday. One trade per market and rule."""
    s = first_per_market(snapshots(p, (12,)))
    frames = []
    for side, sign in (("YES", 1), ("NO", -1)):
        frames.append(take(s, pl.lit(sign)).with_columns(pl.lit(side).alias("side")))
    return pl.concat(frames, how="diagonal_relaxed")


def calendar_rules(ct: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (side, pb, wk), f in ct.group_by(["side", "pb", "weekend"], maintain_order=True):
        rows += score(f, by_date=True, rule=f"buy {side}", bucket=pb, weekend=wk)
    for (side, wk), f in ct.group_by(["side", "weekend"], maintain_order=True):
        rows += score(f, by_date=True, rule=f"buy {side}", bucket="all", weekend=wk)
    return pl.DataFrame(rows, infer_schema_length=None).sort("rule", "bucket", "weekend", "split")


# ---- calendar: daily drift of long-dated contracts by weekday -------------------------------------
def hazard_panel(hazard: Path = HAZARD) -> pl.DataFrame:
    u = (pl.read_parquet(hazard / "universe_all.parquet")
         .filter(pl.col("result").is_in(["yes", "no"]))
         .select("ticker", "event_ticker", "series_ticker", "category", "kind", "close_time", "settlement_ts", "T",
                 "result", "fee_type", "fee_multiplier"))
    c = (pl.read_parquet(hazard / "candles.parquet")
         .select("ticker", pl.col("end_ts").alias("t"), pl.col("yes_bid_close").alias("bid"),
                 pl.col("yes_ask_close").alias("ask"), "volume", pl.col("open_interest").alias("oi"))
         .unique(["ticker", "t"]).join(u.select("ticker", "close_time"), on="ticker")
         .filter(pl.col("t") < pl.col("close_time")).drop("close_time"))
    p = hourly_grid(c).join(u, on="ticker")
    ev = u.select("event_ticker").unique().with_columns(
        pl.col("event_ticker").map_elements(match_key, return_dtype=pl.String).alias("cluster"))
    p = p.join(ev, on="event_ticker").with_columns((pl.col("result") == "yes").cast(pl.Float64).alias("y"),
                                                   (pl.col("T") >= HOLDOUT_FROM).alias("holdout"))
    return with_quote_cols(p.sort("ticker", "t"))


def daily_moves(hp: pl.DataFrame, et_hour: int = 0) -> pl.DataFrame:
    """Signal-mid change over each ET day [d hh:00, d+1 hh:00), labelled by d's weekday.

    The day's end value is the next ET-midnight signal mid if the market is still open then, or
    its settlement (``y``) if it closed during the day. Dropping close days instead would censor
    the early-YES jumps (an event closes the market), and events cluster on weekdays."""
    s = hp.filter((pl.col("et_hour") == et_hour) & pl.col("smid").is_not_null()).select(
        "ticker", "cluster", "category", "kind", "holdout", "t", "et_date", "dow", "weekend", "smid", "y",
        "close_time")
    # DST days are 23/25 h long: pair each row with the market's next ET-midnight row, not t + 24 h
    s2 = s.sort("ticker", "t").with_columns(pl.col("smid").shift(-1).over("ticker").alias("smid_next"),
                                            pl.col("t").shift(-1).over("ticker").alias("t_next"))
    gap = (pl.col("t_next") - pl.col("t")).dt.total_hours()
    closes = pl.col("close_time") <= pl.col("t") + pl.duration(hours=24)
    s2 = s2.with_columns(pl.when(gap.is_between(23, 25)).then(pl.col("smid_next"))
                         .when(pl.col("t_next").is_null() & closes).then(pl.col("y")).alias("end"),
                         (pl.col("t_next").is_null() & closes).alias("settled_today"))
    return s2.filter(pl.col("end").is_not_null()).with_columns(
        (pl.col("end") - pl.col("smid")).alias("dmid"), price_bucket(pl.col("smid")).alias("pb"))


def drift_tables(dm: pl.DataFrame) -> dict[str, pl.DataFrame]:
    rows = []
    for split, sub in (("dev", dm.filter(~pl.col("holdout"))), ("holdout", dm.filter(pl.col("holdout"))), ("all", dm)):
        for dow in range(1, 8):
            f = sub.filter(pl.col("dow") == dow)
            m, td, kd = cl_mean(f, "dmid", "et_date")
            rows.append({"split": split, "dow": DOW_NAMES[dow], "market_days": f.height, "dates": kd,
                         "mean_dmid_c": _r(None if m is None else 100 * m), "t_by_date": _r(td),
                         "median_dmid_c": _r(100 * f["dmid"].median()) if f.height else None,
                         "share_down": _r((f["dmid"] < 0).mean(), 3) if f.height else None,
                         "share_up": _r((f["dmid"] > 0).mean(), 3) if f.height else None,
                         "abs_move_c": _r(100 * f["dmid"].abs().mean()) if f.height else None})
    by_dow = pl.DataFrame(rows, infer_schema_length=None)
    rows = []
    for split, sub in (("dev", dm.filter(~pl.col("holdout"))), ("holdout", dm.filter(pl.col("holdout"))), ("all", dm)):
        cuts = [("all", sub)] + [(f"bucket={b[2]}", sub.filter(pl.col("pb") == b[2])) for b in PRICE_BUCKETS]
        cuts += [(f"category={c}", sub.filter(pl.col("category") == c))
                 for c in ("Politics", "Mentions", "Entertainment", "Sports")]
        cuts += [("monday only vs rest", sub.with_columns((pl.col("dow") == 1).alias("weekend")))]
        for name, f in cuts:
            rows.append({"split": split, "cut": name, **weekend_gap(f, "dmid")})
    return {"drift_by_dow": by_dow, "drift_weekend_gap": pl.DataFrame(rows, infer_schema_length=None)}


def hazard_calendar_trades(hp: pl.DataFrame) -> pl.DataFrame:
    """Long-dated contracts: at every noon-ET snapshot, buy YES / NO; labelled by weekday. One
    trade per market per weekend/weekday label (its first such snapshot)."""
    s = hp.filter((pl.col("et_hour") == 12) & pl.col("smid").is_not_null() & (pl.col("spread") <= MAX_TAKE_SPREAD)
                  & pl.col("mid").is_between(0.03, 0.97)).with_columns(price_bucket(pl.col("mid")).alias("pb"))
    s = s.sort("t").unique(["ticker", "weekend"], keep="first", maintain_order=True)
    frames = [take(s, pl.lit(sign)).with_columns(pl.lit(side).alias("side")) for side, sign in (("YES", 1), ("NO", -1))]
    return pl.concat(frames, how="diagonal_relaxed")


LEAD_BUCKETS_D = ((0, 1, "<1d"), (1, 3, "1-3d"), (3, 7, "3-7d"), (7, 14, "7-14d"), (14, 30, "14-30d"), (30, 1e9, ">30d"))


def daily_entries(hp: pl.DataFrame, sign: int = -1) -> pl.DataFrame:
    """Every market-day at noon ET (takeable quote, mid 3-97c, before the deadline T) as a taker
    entry on one side (-1 = NO), held to settlement. ``ret`` = net PnL per $ risked."""
    tau = (pl.col("T") - pl.col("t")).dt.total_minutes() / 1440
    lead = pl.lit(None, dtype=pl.String)
    for lo, hi, name in reversed(LEAD_BUCKETS_D):
        lead = pl.when(tau.is_between(lo, hi, closed="left")).then(pl.lit(name)).otherwise(lead)
    s = hp.filter((pl.col("et_hour") == 12) & pl.col("smid").is_not_null() & (pl.col("spread") <= MAX_TAKE_SPREAD)
                  & pl.col("mid").is_between(0.03, 0.97) & (pl.col("T") > pl.col("t")))
    s = s.with_columns(price_bucket(pl.col("mid")).alias("pb"), tau.alias("tau_d"), lead.alias("lead"))
    f = take(s.filter(pl.col("pb").is_not_null()), pl.lit(sign))
    return f.with_columns((pl.col("pnl") / pl.col("risk")).alias("ret"))


def weekend_fe(f: pl.DataFrame, col: str = "ret", fe: tuple[str, ...] = ("lead", "pb")) -> dict[str, Any]:
    """``col`` on weekend minus weekday entries with fixed effects for ``fe`` (lead time, price
    bucket), t clustered by ET date and by match."""
    if f.height < 50 or f["weekend"].n_unique() < 2:
        return {"n": f.height}
    cols = [np.ones(f.height), f["weekend"].cast(pl.Float64).to_numpy()]
    for c in fe:
        for lvl in sorted(f[c].unique().to_list())[1:]:
            dummy = (f[c] == lvl).cast(pl.Float64).to_numpy()
            if dummy.std() > 0:
                cols.append(dummy)
    X = np.column_stack(cols)
    y = f[col].to_numpy().astype(float)
    b, se_d = ols_cluster(y, X, f["et_date"].cast(pl.String).to_numpy())
    _, se_m = ols_cluster(y, X, f["cluster"].to_numpy())
    return {"n": f.height, "markets": f["ticker"].n_unique(), "mean": _r(float(y.mean()), 4),
            "weekend_minus_weekday": _r(b[1], 4), "t_by_date": _r(b[1] / se_d[1]) if se_d[1] > 0 else None,
            "t_by_match": _r(b[1] / se_m[1]) if se_m[1] > 0 else None}


def hazard_dow_tables(de: pl.DataFrame) -> dict[str, pl.DataFrame]:
    rows = []
    for split, sub in (("dev", de.filter(~pl.col("holdout"))), ("holdout", de.filter(pl.col("holdout"))), ("all", de)):
        for dow in range(1, 8):
            f = sub.filter(pl.col("dow") == dow)
            m, td, kd = cl_mean(f, "pnl", "et_date", weight="risk")
            _, tm, _ = cl_mean(f, "pnl", "cluster", weight="risk")
            rows.append({"split": split, "dow": DOW_NAMES[dow], "entries": f.height, "dates": kd,
                         "net_per_$": _r(m, 4), "t_by_date": _r(td), "t_by_match": _r(tm),
                         "gross_c": _r(100 * f["gross"].mean()) if f.height else None})
    by_dow = pl.DataFrame(rows, infer_schema_length=None)
    rows = []
    for split, sub in (("dev", de.filter(~pl.col("holdout"))), ("holdout", de.filter(pl.col("holdout"))), ("all", de)):
        cuts = [("all", sub)] + [(f"category={c}", sub.filter(pl.col("category") == c))
                                 for c in ("Politics", "Mentions", "Entertainment", "Sports")]
        cuts += [(f"lead={b[2]}", sub.filter(pl.col("lead") == b[2])) for b in LEAD_BUCKETS_D]
        for name, f in cuts:
            rows.append({"split": split, "cut": name, **weekend_fe(f)})
    return {"hazard_no_by_dow": by_dow, "hazard_no_weekend_fe": pl.DataFrame(rows, infer_schema_length=None)}


# ---- calendar: who loses, by weekday (trade tape) -------------------------------------------------
def taker_flow(tables: Path = TABLES) -> dict[str, pl.DataFrame]:
    m = load_markets(tables).select("ticker", "event_ticker", "series_ticker", "category", "result")
    t = (pl.scan_parquet(tables / "kalshi_trades.parquet")
         .join(m.lazy(), on="ticker")
         .with_columns(pl.col("created_time").dt.convert_time_zone(ET).alias("et"))
         .with_columns(pl.col("et").dt.weekday().alias("dow"), pl.col("et").dt.date().alias("et_date"),
                       (pl.col("et").dt.weekday() >= 6).alias("weekend"), pl.col("et").dt.hour().alias("et_hour"),
                       pl.when(pl.col("taker_side") == "yes").then(pl.col("yes_price"))
                       .otherwise(1 - pl.col("yes_price")).alias("px"),
                       pl.when(pl.col("taker_side") == "yes").then(pl.col("result") == "yes")
                       .otherwise(pl.col("result") == "no").cast(pl.Float64).alias("win"))
         .filter(pl.col("px").is_between(0.01, 0.99))
         .with_columns(((pl.col("win") - pl.col("px")) * pl.col("count")).alias("pnl_usd"),
                       (pl.col("px") * pl.col("count")).alias("risk_usd"),
                       ((pl.col("taker_side") == "yes").cast(pl.Float64) * pl.col("count")).alias("yes_ct"))
         .collect())
    ev = t.select("event_ticker").unique().with_columns(
        pl.col("event_ticker").map_elements(match_key, return_dtype=pl.String).alias("cluster"))
    t = t.join(ev, on="event_ticker").with_columns(kind_of(pl.col("series_ticker")).alias("kind"))
    rows = []
    for dow in range(1, 8):
        f = t.filter(pl.col("dow") == dow)
        r, td, kd = cl_mean(f, "pnl_usd", "et_date", weight="risk_usd")
        _, tm, _ = cl_mean(f, "pnl_usd", "cluster", weight="risk_usd")
        rows.append({"dow": DOW_NAMES[dow], "trades": f.height, "contracts": _r(f["count"].sum(), 0), "dates": kd,
                     "taker_pnl_per_$": _r(r, 4), "t_by_date": _r(td), "t_by_match": _r(tm),
                     "yes_share": _r(f["yes_ct"].sum() / f["count"].sum(), 3)})
    by_dow = pl.DataFrame(rows)
    rows = []
    for (kd, wk), f in t.group_by(["kind", "weekend"], maintain_order=True):
        r, td, _ = cl_mean(f, "pnl_usd", "et_date", weight="risk_usd")
        rows.append({"kind": kd, "weekend": wk, "trades": f.height, "taker_pnl_per_$": _r(r, 4), "t_by_date": _r(td)})
    by_kind = pl.DataFrame(rows).sort("kind", "weekend")
    return {"taker_by_dow": by_dow, "taker_by_kind_weekend": by_kind,
            "sample": pl.DataFrame({"markets": [t["ticker"].n_unique()], "trades": [t.height],
                                    "first": [t["created_time"].min()], "last": [t["created_time"].max()]})}


# ---- capped-book ledgers (Sharpe / vol via vol_metrics) -------------------------------------------
def capped_ledger(f: pl.DataFrame, strategy: str, variant: str, split: str,
                  bankroll: float = BANKROLL, event_cap: float = EVENT_CAP) -> pl.DataFrame:
    """Trades in time order on a $``bankroll`` book: taken only while open cost (incl. this one)
    stays within the bankroll and the event's open cost within ``event_cap``; capital returns at
    settlement."""
    heap: list[tuple[Any, float, str]] = []
    used, per_ev, rows = 0.0, {}, []
    for r in f.sort("t").iter_rows(named=True):
        while heap and heap[0][0] <= r["t"]:
            _, c, e = heapq.heappop(heap)
            used -= c
            per_ev[e] -= c
        cost = r["n"] * r["price"] + r["fee_order"]
        if used + cost > bankroll + 1e-9 or per_ev.get(r["event_ticker"], 0.0) + cost > event_cap + 1e-9:
            continue
        settle = max(r["settlement_ts"], r["t"])
        used += cost
        per_ev[r["event_ticker"]] = per_ev.get(r["event_ticker"], 0.0) + cost
        heapq.heappush(heap, (settle, cost, r["event_ticker"]))
        rows.append({"strategy": strategy, "variant": variant, "split": split,
                     "struct_id": f"{r['ticker']}|{r['t'].isoformat()}", "event_ticker": r["event_ticker"],
                     "ticker": r["ticker"], "side": "yes" if r["dir"] == 1 else "no", "n": int(r["n"]),
                     "t_in": r["t"], "cost_usd": cost, "settle_ts": settle, "payoff_usd": r["n"] * r["payoff"]})
    return pl.DataFrame(rows)


def ledger_row(led: pl.DataFrame) -> dict[str, Any]:
    from albert.research.vol_metrics import ledger_metrics, load_marks

    if led.is_empty():
        return {"positions": 0}
    m = ledger_metrics(led, load_marks(led["ticker"].unique().to_list()))
    keep = ("positions", "events", "pnl_usd", "usd_per_week", "ann_return_pct", "ann_vol_pct", "sharpe", "sharpe_ci90",
            "p_sharpe_pos", "sharpe_ex_top5", "sharpe_weekly", "max_dd_usd", "weeks_positive_pct", "skew",
            "first_day", "last_day")
    return {k: m.get(k) for k in keep}


# ---- driver --------------------------------------------------------------------------------------
def _save(name: str, df: pl.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT / f"{name}.csv")


def run_ta(universe: str = "blind") -> pl.DataFrame:
    p = add_indicators(prepare(universe))
    if p.is_empty():
        raise RuntimeError("empty panel")
    tr = ta_trades(p)
    if tr.is_empty():
        raise RuntimeError("zero TA trades: investigate before reporting")
    tr.write_parquet(OUT / f"ta_trades_{universe}.parquet")
    _save(f"ta_{universe}", ta_table(tr))
    _save(f"ta_cuts_{universe}", ta_cuts(tr))
    sample = {"universe": universe, "markets": p["ticker"].n_unique(), "matches": p["cluster"].n_unique(),
              "grid_rows": p.height, "real_candles": int(p["real"].sum()),
              "first_t": str(p["t"].min()), "last_t": str(p["t"].max())}
    (OUT / f"sample_{universe}.json").write_text(json.dumps(sample, indent=2))
    return tr


def run_calendar(universe: str = "blind") -> None:
    p = prepare(universe)
    for name, df in calendar_tables(p).items():
        _save(f"{name}_{universe}", df)
    ct = calendar_trades(p)
    ct.write_parquet(OUT / f"cal_trades_{universe}.parquet")
    _save(f"cal_rules_{universe}", calendar_rules(ct))


def run_drift() -> None:
    hp = hazard_panel()
    dm = daily_moves(hp)
    if dm.is_empty():
        raise RuntimeError("no daily moves")
    for name, df in drift_tables(dm).items():
        _save(name, df)
    ht = hazard_calendar_trades(hp)
    ht.write_parquet(OUT / "hazard_cal_trades.parquet")
    _save("hazard_cal_rules", calendar_rules(ht))
    de = daily_entries(hp, -1)
    if de.is_empty():
        raise RuntimeError("no daily NO entries")
    de.write_parquet(OUT / "hazard_no_daily.parquet")
    for name, df in hazard_dow_tables(de).items():
        _save(name, df)
    (OUT / "sample_hazard.json").write_text(json.dumps(
        {"markets": hp["ticker"].n_unique(), "market_days": dm.height, "dates": dm["et_date"].n_unique(),
         "first": str(dm["t"].min()), "last": str(dm["t"].max())}, indent=2))


def run_flow() -> None:
    for name, df in taker_flow().items():
        _save(name, df)


def ledger_specs() -> list[tuple[str, str, Path, pl.Expr]]:
    """(strategy, variant, trade file, filter) for the headline strategies. Picked before looking
    at holdout ledgers: each TA family's least-bad dev rule, the benchmark, the near-event RSI cell,
    and the calendar rules whose gross had the same sign in dev and holdout."""
    ta, cal, hz = OUT / "ta_trades_blind.parquet", OUT / "cal_trades_blind.parquet", OUT / "hazard_no_daily.parquet"
    sig = lambda s, m: (pl.col("signal") == s) & (pl.col("mode") == m)
    return [
        ("ta", "tsmom6 5c trend (benchmark)", ta, sig("tsmom6 5c", "trend")),
        ("ta", "rsi14 70/30 trend", ta, sig("rsi14 70/30", "trend")),
        ("ta", "rsi14 70/30 trend, 3-12h to exp", ta, sig("rsi14 70/30", "trend") & (pl.col("hb") == "3-12h")),
        ("ta", "macd hist cross trend", ta, sig("macd hist cross", "trend")),
        ("ta", "ema 3/12 cross fade", ta, sig("ema 3/12 cross", "fade")),
        ("calendar", "weekend favourites: buy YES 50-97c at noon ET", cal,
         (pl.col("side") == "YES") & pl.col("weekend") & pl.col("pb").is_in(["50-80c", "80-97c"])),
        ("calendar", "weekday favourites: buy YES 50-97c at noon ET (control)", cal,
         (pl.col("side") == "YES") & ~pl.col("weekend") & pl.col("pb").is_in(["50-80c", "80-97c"])),
        ("calendar", "deadline contracts: buy NO daily at noon ET, weekends only", hz, pl.col("weekend")),
        ("calendar", "deadline contracts: buy NO daily at noon ET, weekdays only", hz, ~pl.col("weekend")),
    ]


def run_ledgers() -> pl.DataFrame:
    (OUT / "ledgers").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (strategy, variant, path, filt) in enumerate(ledger_specs()):
        f = pl.read_parquet(path).filter(filt)
        for split, sub in (("all", f), ("holdout", f.filter(pl.col("holdout")))):
            led = capped_ledger(sub, f"ta_calendar/{strategy}", variant, split)
            if split == "all":
                led.write_parquet(OUT / "ledgers" / f"ledger_{i:02d}.parquet")
            rows.append({"strategy": strategy, "variant": variant, "split": split, "signals": sub.height,
                         **ledger_row(led)})
            print(variant, split, rows[-1].get("sharpe"), rows[-1].get("usd_per_week"), flush=True)
    df = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("sharpe_ci90").map_elements(lambda v: f"[{v[0]}, {v[1]}]" if v is not None else "",
                                           return_dtype=pl.String))
    _save("ledger_metrics", df)
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.ta_calendar")
    ap.add_argument("step", choices=["ta", "calendar", "drift", "flow", "ledgers", "all"])
    ap.add_argument("--universe", default="blind", choices=["blind", "all"])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step in ("ta", "all"):
        run_ta(a.universe)
    if a.step in ("calendar", "all"):
        run_calendar(a.universe)
    if a.step in ("drift", "all"):
        run_drift()
    if a.step in ("flow", "all"):
        run_flow()
    if a.step in ("ledgers", "all"):
        run_ledgers()


if __name__ == "__main__":
    main()
