"""Optimal market-making equations fitted to the Kalshi L2 recording and replayed as quoters.

Models (prices in cents, time in seconds, inventory ``q`` in contracts, net YES):

* **Avellaneda–Stoikov (2008)**: reservation price ``r = s - q*gamma*sigma^2*tau`` and total spread
  ``gamma*sigma^2*tau + (2/gamma)*ln(1 + gamma/k)`` around ``r``; ``s`` is the mid or the
  micro-price, ``tau = min(time to expected expiry, horizon_s)``.
* **Guéant–Lehalle–Fernandez-Tapia (2013) / Guéant (2017)**, asymptotic closed form with bounded
  inventory, lot size 1 and ``xi = gamma``::

      c        = sqrt(gamma*sigma^2 / (2*A*k) * (1 + gamma/k)^(1 + k/gamma))
      delta_b  = (1/gamma)*ln(1 + gamma/k) + (2q + 1)/2 * c
      delta_a  = (1/gamma)*ln(1 + gamma/k) - (2q - 1)/2 * c

  no bid at ``q >= max_pos``, no ask at ``q <= -max_pos``.
* **Logit Avellaneda–Stoikov**: the same algebra on ``x = logit(p)`` with ``sigma_x`` and
  ``k_x`` measured in logit units, mapped back with the logistic function.

Calibration (``fit_*``; the caller passes dev-window data only):

* **Fill intensity** ``lambda(delta)``: (1) *taker reach* — each taker order (trades grouped by
  market, ``ts_ms`` and taker side) reaches ``d`` cents beyond the mid of the external book just
  before its first message; a 1-lot resting ``delta`` cents from the mid is filled if ``d > delta``
  (strictly through: inside the spread that is any trade at the touch; at or behind the best only
  a sweep past it, so no queue position is assumed). ``lambda = count / (2 * exposure seconds)``
  per side, exposure from a 10 s grid of two-sided books. (2) *traded-through fraction* — on a
  30 s grid, would a 1-lot at ``mid0 +/- delta`` have been traded through (or crossed by the
  opposite best) within 10 s / 60 s; ``lambda = -ln(1 - P) / H``. Fitted to ``A*exp(-k*delta)``
  and to ``B*delta^-alpha`` (weighted least squares on the log, weights = counts) per regime:
  spread bucket x volatile (previous 5-min bucket, the decision-time flag) x price band.
* **Volatility**: realised variance of the external mid on a 10 s grid per (market, 5-min
  bucket), in cents^2/s and logit^2/s; a 60 s grid gives the signature ratio. A strategy at time
  ``t`` uses only buckets that ended before ``t``'s bucket began (``SigmaLookup``).

``ModelQuoter`` runs in the shared ``backtest.l2.Simulator`` (FIFO queues, latency, post-only,
write budget). 1-lot per side, 1c ticks, model prices rounded to the nearest tick (ties away from
the mid), clamped one tick short of the opposite external best, and only inside a
[``lo``, ``hi``] mid band. Join rule ``front`` (default): a new order is placed only if it is
strictly better than the external best on its side (alone at the front, like the improve
Quoter); an order that others later joined is kept while within ``hyst_ticks`` of the target.
``model``: post at the model price wherever it falls (joining or resting behind the best).
``max_improve=1`` turns the model into a gate on the baseline's improve rule (quote a side at
best + 1 tick only if the model's price reaches it; the inventory skew can switch a side off).

``mm_models(settings(), start=..., split=..., end=...)`` runs everything: calibration on
``[start, split)``, a gamma sweep on dev (``targets`` = inventory term per contract, cents, at the
calm prior variance), dev-chosen configs, ablations (no skew, 300 s cooldown, join rule) and the
gate variants replayed on ``[split, end)`` at both latencies; report ``research-mm-models``.
Replays are cached under ``data/cache/mm_models_run_*`` keyed by window and exact specs.
"""

from __future__ import annotations

import bisect
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.backtest.l2 import (
    EPS,
    Event,
    Quoter,
    SimConfig,
    Simulator,
    iter_events,
    markouts,
    stream_files,
)

TICK = 0.01
HOUR_NS = 3600 * 10**9
DELTAS = tuple(range(1, 11))                                   # cents from the mid
DELTAS_X = tuple(round(0.05 * i, 2) for i in range(1, 11))     # logit units from the logit mid
SPREAD_EDGES = ((2.0, "s<=2"), (4.0, "s3-4"), (6.0, "s5-6"), (10.0, "s7-10"), (1e9, "s>10"))
P_CLIP = 0.005


# ---- small helpers -----------------------------------------------------------------------------
def logit(p: float) -> float:
    p = min(max(p, P_CLIP), 1 - P_CLIP)
    return math.log(p / (1 - p))


def expit(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def spread_bucket(spread_c: float) -> str:
    for edge, name in SPREAD_EDGES:
        if spread_c <= edge + 1e-6:
            return name
    return SPREAD_EDGES[-1][1]


def price_band(mid: float) -> str:
    d = min(mid, 1 - mid)
    return "tail" if d < 0.15 else "shoulder" if d < 0.30 else "center"


def _spread_bucket_expr(col: str = "spread_c") -> pl.Expr:
    e = pl.lit(SPREAD_EDGES[-1][1])
    for edge, name in reversed(SPREAD_EDGES[:-1]):
        e = pl.when(pl.col(col) <= edge + 1e-6).then(pl.lit(name)).otherwise(e)
    return e.alias("sbucket")


def _band_expr(col: str = "mid") -> pl.Expr:
    d = pl.min_horizontal(pl.col(col), 1 - pl.col(col))
    return (pl.when(d < 0.15).then(pl.lit("tail")).when(d < 0.30).then(pl.lit("shoulder"))
            .otherwise(pl.lit("center")).alias("band"))


def _logit_expr(col: str) -> pl.Expr:
    p = pl.col(col).clip(P_CLIP, 1 - P_CLIP)
    return (p / (1 - p)).log()


# ---- extraction (one replay of the external book) ----------------------------------------------
def extract(events: Iterable[Event]) -> dict[str, pl.DataFrame]:
    """External top of book (every change), trades with the book just before their first
    message (same market and ``ts_ms``), and the (market, hour) pairs that had a snapshot."""
    sim = Simulator(SimConfig(writes_per_s=None))
    cur_ts: dict[str, int] = {}
    pre: dict[str, tuple] = {}
    trades: list[dict[str, Any]] = []
    snaps: set[tuple[str, int]] = set()
    for ev in events:
        mk = ev.market
        if ev.kind == "snapshot":
            snaps.add((mk, ev.recv_ns // HOUR_NS * HOUR_NS))
        elif ev.kind in ("delta", "trade"):
            ts = ev.msg.get("ts_ms")
            if ts is not None and ts != cur_ts.get(mk):
                cur_ts[mk] = ts
                q = sim.quotes.get(mk)
                pre[mk] = q[1][-1] if q and q[1] else (None, None)
        sim.apply(ev)
        if ev.kind == "trade" and mk in sim.synced:
            m = ev.msg
            b, a = pre.get(mk, (None, None))
            trades.append({"market": mk, "t_ns": ev.recv_ns, "ts_ms": m.get("ts_ms"),
                           "taker_side": m["taker_side"], "yes_price": float(m["yes_price_dollars"]),
                           "count": float(m["count_fp"]), "pre_bid": b, "pre_ask": a})
    tob = [{"market": mk, "t_ns": t, "bid": b, "ask": a}
           for mk, (times, vals) in sim.quotes.items() for t, (b, a) in zip(times, vals)]
    schema_t = {"market": pl.String, "t_ns": pl.Int64, "bid": pl.Float64, "ask": pl.Float64}
    schema_tr = {"market": pl.String, "t_ns": pl.Int64, "ts_ms": pl.Int64, "taker_side": pl.String,
                 "yes_price": pl.Float64, "count": pl.Float64, "pre_bid": pl.Float64, "pre_ask": pl.Float64}
    return {"tob": pl.DataFrame(tob, schema=schema_t).sort("t_ns"),
            "trades": pl.DataFrame(trades, schema=schema_tr).sort("t_ns"),
            "active": pl.DataFrame([{"market": m, "hour_ns": h} for m, h in sorted(snaps)],
                                   schema={"market": pl.String, "hour_ns": pl.Int64})}


def cached_extract(s, start: str, end: str, root: Path | None = None) -> dict[str, pl.DataFrame]:
    """``extract`` over the recorder's stream, cached under ``data/cache/mm_models_*``."""
    root = root or s.raw_dir / "stream"
    tag = f"{start}_{end}".replace(":", "")
    paths = {k: s.data_dir / "cache" / f"mm_models_{k}_{tag}.parquet" for k in ("tob", "trades", "active")}
    if all(p.exists() for p in paths.values()):
        return {k: pl.read_parquet(p) for k, p in paths.items()}
    stats: dict[str, int] = defaultdict(int)
    out = extract(iter_events(stream_files(root, start, end), stats))
    for k, p in paths.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        out[k].write_parquet(p)
    return out


def vol_flags(vol: pl.DataFrame) -> dict[tuple[str, int], bool]:
    """(market, bucket_ns) -> own-bucket ``volatile`` flag from ``research.volatile``."""
    return {(m, b): bool(v) for m, b, v in vol.select("market", "bucket_ns", "volatile").iter_rows()}


def add_regime(df: pl.DataFrame, vol: pl.DataFrame, bucket_s: int = 300) -> pl.DataFrame:
    """Add ``bucket_ns``, ``volatile`` (own bucket), ``vol_prev`` (previous calendar bucket),
    ``sbucket`` and ``band`` (needs ``mid`` and ``spread_c``)."""
    bns = bucket_s * 10**9
    v = vol.select("market", "bucket_ns", pl.col("volatile").cast(pl.Boolean))
    return (df.with_columns((pl.col("t_ns") // bns * bns).alias("bucket_ns"))
            .join(v, on=["market", "bucket_ns"], how="left")
            .join(v.select("market", (pl.col("bucket_ns") + bns).alias("bucket_ns"),
                           pl.col("volatile").alias("vol_prev")), on=["market", "bucket_ns"], how="left")
            .with_columns(pl.col("volatile").fill_null(False), pl.col("vol_prev").fill_null(False),
                          _spread_bucket_expr(), _band_expr()))


# ---- grid, volatility --------------------------------------------------------------------------
def grid(tob: pl.DataFrame, active: pl.DataFrame, step_s: int = 10) -> pl.DataFrame:
    """External book on a regular grid inside each market's recorded hours (two-sided only)."""
    step = step_s * 10**9
    g = (active.with_columns(pl.int_ranges(pl.col("hour_ns"), pl.col("hour_ns") + HOUR_NS, step,
                                           dtype=pl.Int64).alias("t_ns"))
         .explode("t_ns").select("market", "t_ns").sort("t_ns"))
    g = g.join_asof(tob.sort("t_ns"), on="t_ns", by="market", strategy="backward")
    return (g.filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("ask") > pl.col("bid")))
            .with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                          ((pl.col("ask") - pl.col("bid")) * 100).round(6).alias("spread_c"))
            .sort("market", "t_ns"))


def sigma_buckets(g: pl.DataFrame, step_s: int = 10, bucket_s: int = 300) -> pl.DataFrame:
    """Realised variation of the mid per (market, bucket) on consecutive ``step_s`` grid points:
    ``S`` (sum of squared changes, cents^2), ``Sx`` (logit^2), ``T`` (seconds observed), the same
    on a 60 s subgrid (``S60``, ``T60``), ``var_p = S/T`` and ``var_x = Sx/T``. An interval
    belongs to the bucket in which it starts."""
    bns, step = bucket_s * 10**9, step_s * 10**9

    def rv(frame: pl.DataFrame, st: int, suffix: str) -> pl.DataFrame:
        f = (frame.sort("market", "t_ns")
             .with_columns((pl.col("mid") * 100).diff().over("market").alias("dm"),
                           _logit_expr("mid").diff().over("market").alias("dx"),
                           pl.col("t_ns").diff().over("market").alias("dt"),
                           (pl.col("mid") * (1 - pl.col("mid"))).alias("pq"))
             .filter(pl.col("dt") == st)
             .with_columns(((pl.col("t_ns") - st) // bns * bns).alias("bucket_ns")))
        return f.group_by("market", "bucket_ns").agg(
            pl.len().alias(f"n{suffix}"), (pl.col("dm") ** 2).sum().alias(f"S{suffix}"),
            (pl.col("dx") ** 2).sum().alias(f"Sx{suffix}"), (pl.len() * st / 1e9).alias(f"T{suffix}"),
            pl.col("mid").mean().alias(f"mid{suffix}"), (pl.col("pq") ** 2).mean().alias(f"pq2{suffix}"),
            pl.col("spread_c").median().alias(f"spread{suffix}"))

    a = rv(g, step, "")
    b = rv(g.filter(pl.col("t_ns") % (60 * 10**9) == 0), 60 * 10**9, "60")
    return (a.join(b.select("market", "bucket_ns", "S60", "T60"), on=["market", "bucket_ns"], how="left")
            .with_columns(pl.col("S60").fill_null(0.0), pl.col("T60").fill_null(0.0),
                          (pl.col("S") / pl.col("T")).alias("var_p"), (pl.col("Sx") / pl.col("T")).alias("var_x"))
            .rename({"spread": "spread_c"}).sort("market", "bucket_ns"))


class SigmaLookup:
    """Variance known at time ``t``: the market's realised variation over the ``lookback``
    buckets that ended before the bucket containing ``t`` began, shrunk towards a prior worth
    ``prior_s`` seconds of observation: ``(S + prior * prior_s) / (T + prior_s)``. Most 5-min
    buckets of these thin books have no mid change at all, so a single bucket is not enough."""

    def __init__(self, sig: pl.DataFrame, *, prior_p: float, prior_x: float, prior_s: float = 900.0,
                 lookback: int = 12, bucket_s: int = 300) -> None:
        self.bns, self.lookback, self.prior_s = bucket_s * 10**9, lookback, prior_s
        self.prior = (prior_p, prior_x)
        self.by: dict[str, tuple[list[int], np.ndarray, np.ndarray, np.ndarray]] = {}
        for (m,), f in sig.sort("bucket_ns").group_by(["market"], maintain_order=True):
            z = np.zeros(1)
            self.by[m] = (f["bucket_ns"].to_list(), np.concatenate([z, np.cumsum(f["S"].to_numpy())]),
                          np.concatenate([z, np.cumsum(f["Sx"].to_numpy())]),
                          np.concatenate([z, np.cumsum(f["T"].to_numpy())]))
        self._cache: dict[tuple[str, int], tuple[float, float]] = {}

    def at(self, market: str, t_ns: int) -> tuple[float, float]:
        b = t_ns // self.bns * self.bns
        key = (market, b)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        s = sx = tt = 0.0
        rec = self.by.get(market)
        if rec:
            i0 = bisect.bisect_left(rec[0], b - self.lookback * self.bns)
            i1 = bisect.bisect_left(rec[0], b)           # buckets starting before b (ended by b)
            s, sx, tt = rec[1][i1] - rec[1][i0], rec[2][i1] - rec[2][i0], rec[3][i1] - rec[3][i0]
        w = self.prior_s
        out = ((s + self.prior[0] * w) / (tt + w), (sx + self.prior[1] * w) / (tt + w))
        self._cache[key] = out
        return out


# ---- fill intensity ----------------------------------------------------------------------------
def taker_orders(trades: pl.DataFrame, lo: float = 0.05, hi: float = 0.95) -> pl.DataFrame:
    """One row per taker order: reach ``d_c`` (cents) and ``d_x`` (logit) beyond the pre-trade mid."""
    t = trades.filter(pl.col("pre_bid").is_not_null() & pl.col("pre_ask").is_not_null()
                      & (pl.col("pre_ask") > pl.col("pre_bid")))
    o = t.group_by("market", "ts_ms", "taker_side").agg(
        pl.col("t_ns").min(), pl.col("yes_price").max().alias("pmax"), pl.col("yes_price").min().alias("pmin"),
        pl.col("count").sum().alias("qty"), pl.col("pre_bid").first(), pl.col("pre_ask").first())
    yes = pl.col("taker_side") == "yes"
    o = o.with_columns(((pl.col("pre_bid") + pl.col("pre_ask")) / 2).alias("mid"),
                       ((pl.col("pre_ask") - pl.col("pre_bid")) * 100).round(6).alias("spread_c"),
                       pl.when(yes).then(pl.col("pmax")).otherwise(pl.col("pmin")).alias("p_ext"))
    return (o.with_columns(
        pl.when(yes).then((pl.col("p_ext") - pl.col("mid")) * 100)
        .otherwise((pl.col("mid") - pl.col("p_ext")) * 100).round(6).alias("d_c"),
        pl.when(yes).then(_logit_expr("p_ext") - _logit_expr("mid"))
        .otherwise(_logit_expr("mid") - _logit_expr("p_ext")).alias("d_x"))
        .filter(pl.col("mid").is_between(lo, hi)).sort("t_ns"))


def exposure(g: pl.DataFrame, keys: list[str], step_s: int = 10, lo: float = 0.05,
             hi: float = 0.95) -> pl.DataFrame:
    return (g.filter(pl.col("mid").is_between(lo, hi)).group_by(keys)
            .agg((pl.len() * step_s).alias("seconds"), pl.col("market").n_unique().alias("markets")))


def intensity(orders: pl.DataFrame, expo: pl.DataFrame, keys: list[str], *, col: str = "d_c",
              deltas: tuple = DELTAS, touch: bool = False) -> pl.DataFrame:
    """Long table: keys, delta, count of taker orders reaching strictly beyond ``delta`` (or at
    least ``delta`` with ``touch``), exposure seconds and ``lam`` per side per market-second."""
    aggs = [((pl.col(col) >= d - 1e-6) if touch else (pl.col(col) > d + 1e-6)).sum().alias(f"n_{d}")
            for d in deltas]
    cnt = orders.group_by(keys).agg(pl.len().alias("orders"), *aggs) if keys else \
        orders.select(pl.len().alias("orders"), *aggs)
    e = expo.group_by(keys).agg(pl.col("seconds").sum(), pl.col("markets").sum()) if keys else \
        expo.select(pl.col("seconds").sum(), pl.col("markets").sum())
    j = e.join(cnt, on=keys, how="left") if keys else pl.concat([e, cnt], how="horizontal")
    j = j.fill_null(0)
    long = j.unpivot(index=[*keys, "seconds", "markets", "orders"], on=[f"n_{d}" for d in deltas],
                     variable_name="d", value_name="count")
    return (long.with_columns(pl.col("d").str.slice(2).cast(pl.Float64).alias("delta"),
                              (pl.col("count") / (2 * pl.col("seconds"))).alias("lam"))
            .drop("d").sort(*keys, "delta"))


def fit_curve(deltas: list[float], lam: list[float], counts: list[float]) -> dict[str, float | None]:
    """Weighted least squares on log(lambda): exponential ``A*exp(-k*d)`` and power law
    ``B*d^-alpha``; weights are the counts (Poisson). R^2 is weighted, on the log scale."""
    d = np.asarray(deltas, float)
    lm = np.asarray(lam, float)
    w = np.asarray(counts, float)
    m = (lm > 0) & (w > 0)
    out: dict[str, float | None] = {"A": None, "k": None, "r2_exp": None, "B": None, "alpha": None,
                                    "r2_pow": None, "points": int(m.sum())}
    if m.sum() < 3:
        return out
    y, w, d = np.log(lm[m]), w[m], d[m]

    def wls(x: np.ndarray) -> tuple[float, float, float]:
        X = np.column_stack([np.ones_like(x), x])
        W = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
        yhat = X @ beta
        ybar = np.sum(w * y) / np.sum(w)
        ss_tot = np.sum(w * (y - ybar) ** 2)
        r2 = 1 - np.sum(w * (y - yhat) ** 2) / ss_tot if ss_tot > 0 else float("nan")
        return float(beta[0]), float(beta[1]), float(r2)

    a, b, r2 = wls(d)
    out.update(A=math.exp(a), k=-b, r2_exp=r2)
    a, b, r2 = wls(np.log(d))
    out.update(B=math.exp(a), alpha=-b, r2_pow=r2)
    return out


def fit_table(lam_long: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    rows = []
    groups = lam_long.group_by(keys, maintain_order=True) if keys else [((), lam_long)]
    for key, f in groups:
        r = dict(zip(keys, key)) if keys else {}
        r.update(orders=int(f["orders"][0]), hours=round(float(f["seconds"][0]) / 3600, 1))
        r.update(fit_curve(f["delta"].to_list(), f["lam"].to_list(), f["count"].to_list()))
        r.update({f"lam_{int(d) if float(d).is_integer() else d}": v
                  for d, v in zip(f["delta"].to_list()[:5:2], f["lam"].to_list()[:5:2])})
        rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- traded-through fraction (method 2) ---------------------------------------------------------
def _rmq(times: np.ndarray, vals: np.ndarray, lo_t: np.ndarray, hi_t: np.ndarray, op, fill: float) -> np.ndarray:
    """``op``-reduce of ``vals`` with ``lo_t < times <= hi_t`` for each window (sparse table)."""
    out = np.full(len(lo_t), fill)
    n = len(times)
    if n == 0 or len(lo_t) == 0:
        return out
    lo = np.searchsorted(times, lo_t, side="right")
    hi = np.searchsorted(times, hi_t, side="right")
    table = [vals]
    j = 1
    while (1 << j) <= n:
        prev, half = table[-1], 1 << (j - 1)
        table.append(op(prev[:-half], prev[half:]))
        j += 1
    length = hi - lo
    ok = length > 0
    kk = np.zeros(len(lo_t), dtype=int)
    kk[ok] = np.floor(np.log2(length[ok])).astype(int)
    for k in np.unique(kk[ok]):
        sel = ok & (kk == k)
        t = table[k]
        out[sel] = op(t[lo[sel]], t[hi[sel] - (1 << k)])
    return out


def fill_fraction(g: pl.DataFrame, trades: pl.DataFrame, tob: pl.DataFrame, vol: pl.DataFrame, *,
                  horizons: tuple[int, ...] = (10, 60), sample_s: int = 30, deltas: tuple = DELTAS,
                  lo: float = 0.05, hi: float = 0.95) -> pl.DataFrame:
    """Probability that a 1-lot resting at ``mid0 +/- delta`` (fixed at the grid time) is traded
    strictly through by a later print, or crossed by the opposite external best, within each
    horizon; averaged over both sides, by spread bucket x previous-bucket volatility."""
    pts = g.filter((pl.col("t_ns") % (sample_s * 10**9) == 0) & pl.col("mid").is_between(lo, hi))
    pts = add_regime(pts, vol)
    tr = {m: f for (m,), f in trades.group_by(["market"])}
    tq = {m: f for (m,), f in tob.sort("t_ns").group_by(["market"])}
    out = []
    for (m,), p in pts.group_by(["market"]):
        t0 = p["t_ns"].to_numpy()
        mid = p["mid"].to_numpy() * 100
        f = tr.get(m)
        buy = f.filter(pl.col("taker_side") == "yes") if f is not None else None
        sell = f.filter(pl.col("taker_side") == "no") if f is not None else None
        q = tq.get(m)
        cols: dict[str, np.ndarray] = {}
        for h in horizons:
            t1 = t0 + h * 10**9
            mb = (_rmq(buy["t_ns"].to_numpy(), buy["yes_price"].to_numpy() * 100, t0, t1, np.maximum, -np.inf)
                  if buy is not None else np.full(len(t0), -np.inf))
            ms = (_rmq(sell["t_ns"].to_numpy(), sell["yes_price"].to_numpy() * 100, t0, t1, np.minimum, np.inf)
                  if sell is not None else np.full(len(t0), np.inf))
            qt = q["t_ns"].to_numpy()
            mbid = _rmq(qt, q["bid"].fill_null(-1.0).to_numpy() * 100, t0, t1, np.maximum, -np.inf)
            mask = q["ask"].fill_null(2.0).to_numpy() * 100
            mask_ = _rmq(qt, mask, t0, t1, np.minimum, np.inf)
            for d in deltas:
                ask_fill = (mb > mid + d + 1e-6) | (mbid >= mid + d - 1e-6)
                bid_fill = (ms < mid - d - 1e-6) | (mask_ <= mid - d + 1e-6)
                cols[f"f{h}_{d}"] = (ask_fill.astype(float) + bid_fill.astype(float)) / 2
        out.append(p.select("sbucket", "vol_prev").with_columns(
            *[pl.Series(k, v) for k, v in cols.items()]))
    if not out:
        return pl.DataFrame()
    allp = pl.concat(out)
    rows = []
    for keys in (["sbucket", "vol_prev"], []):
        agg = (allp.group_by(keys).agg(pl.len().alias("points"), *[pl.col(c).mean() for c in allp.columns
                                                                    if c.startswith("f")])
               if keys else allp.select(pl.len().alias("points"), *[pl.col(c).mean() for c in allp.columns
                                                                   if c.startswith("f")]))
        for r in agg.iter_rows(named=True):
            for h in horizons:
                row = {"sbucket": r.get("sbucket", "all"), "vol_prev": r.get("vol_prev"), "horizon_s": h,
                       "points": r["points"]}
                lam = []
                for d in deltas:
                    pf = r[f"f{h}_{d}"]
                    row[f"P_{d}"] = round(pf, 4)
                    lam.append(-math.log(max(1 - pf, 1e-9)) / h)
                fit = fit_curve(list(deltas), lam, [r["points"] * x for x in
                                                    [r[f"f{h}_{d}"] for d in deltas]])
                row.update({f"{k}_fit": v for k, v in fit.items() if k in ("A", "k", "r2_exp", "alpha", "r2_pow")})
                rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None).sort("horizon_s", "sbucket", "vol_prev", nulls_last=True)


# ---- calibration bundle --------------------------------------------------------------------------
@dataclass
class Calib:
    """Dev-fitted inputs. ``ak``: (sbucket, vol_prev, band) -> (A per s, k per cent);
    ``akx``: (sbucket, vol_prev) -> (A, k per logit unit); ``vol``: own-bucket volatile flags,
    read at decision time for the *previous* bucket only."""
    ak: dict[tuple, tuple[float, float]]
    akx: dict[tuple, tuple[float, float]]
    sigma: SigmaLookup
    vol: dict[tuple[str, int], bool] = field(default_factory=dict)
    expiry_ns: dict[str, int] = field(default_factory=dict)
    bucket_s: int = 300

    def ak_at(self, sb: str, volp: bool, band: str) -> tuple[float, float]:
        return self.ak.get((sb, volp, band)) or self.ak.get((sb, volp, None)) or self.ak[(None, None, None)]

    def akx_at(self, sb: str, volp: bool) -> tuple[float, float]:
        return self.akx.get((sb, volp)) or self.akx[(None, None)]


def _ak_from_fits(fits: pl.DataFrame, keys: list[str], min_orders: int, k_min: float) -> dict[tuple, tuple[float, float]]:
    out = {}
    for r in fits.iter_rows(named=True):
        if r["k"] is None or r["orders"] < min_orders or r["A"] is None:
            continue
        out[tuple(r.get(k) for k in keys)] = (r["A"], max(r["k"], k_min))
    return out


def calibrate(data: dict[str, pl.DataFrame], vol: pl.DataFrame, *, until_ns: int, from_ns: int = 0,
              min_orders: int = 30) -> tuple[Calib, dict[str, pl.DataFrame]]:
    """Fit intensity and volatility on data in ``[from_ns, until_ns)`` (the dev window). Returns
    the calibration and the tables for the report."""
    tob = data["tob"].filter(pl.col("t_ns") < until_ns)        # earlier book state feeds the grid
    trades = data["trades"].filter(pl.col("t_ns").is_between(from_ns, until_ns, closed="left"))
    active = data["active"].filter((pl.col("hour_ns") + HOUR_NS <= until_ns) & (pl.col("hour_ns") >= from_ns))
    g = grid(tob, active)
    sig = sigma_buckets(g)
    orders = add_regime(taker_orders(trades), vol)
    gr = add_regime(g, vol)
    tables: dict[str, pl.DataFrame] = {}
    keysets = {"all": [], "spread": ["sbucket"], "spread_vol": ["sbucket", "vol_prev"],
               "regime": ["sbucket", "vol_prev", "band"], "own_vol": ["volatile"], "band": ["band"]}
    fits = {}
    for name, keys in keysets.items():
        lam = intensity(orders, exposure(gr, keys), keys)
        fits[name] = fit_table(lam, keys)
        if name in ("all", "spread"):
            tables[f"lam_{name}"] = lam
        lam_t = intensity(orders, exposure(gr, keys), keys, touch=True)
        fits[f"{name}_touch"] = fit_table(lam_t, keys)
    for name, keys in (("x_all", []), ("x_spread_vol", ["sbucket", "vol_prev"])):
        lam = intensity(orders, exposure(gr, keys), keys, col="d_x", deltas=DELTAS_X)
        fits[name] = fit_table(lam, keys)
    tables.update({f"fit_{k}": v for k, v in fits.items()})
    k_min = 0.05
    ak = _ak_from_fits(fits["regime"], ["sbucket", "vol_prev", "band"], min_orders, k_min)
    ak.update({(sb, vp, None): v for (sb, vp), v in
               _ak_from_fits(fits["spread_vol"], ["sbucket", "vol_prev"], min_orders, k_min).items()})
    a0 = fits["all"].row(0, named=True)
    ak[(None, None, None)] = (a0["A"], max(a0["k"], k_min))
    akx = _ak_from_fits(fits["x_spread_vol"], ["sbucket", "vol_prev"], min_orders, 0.5)
    x0 = fits["x_all"].row(0, named=True)
    akx[(None, None)] = (x0["A"], max(x0["k"], 0.5))
    calm = add_regime(sig.with_columns(pl.col("bucket_ns").alias("t_ns")), vol).filter(~pl.col("volatile"))
    prior_p = float(calm["S"].sum() / calm["T"].sum())
    prior_x = float(calm["Sx"].sum() / calm["T"].sum())
    sl = SigmaLookup(sig, prior_p=prior_p, prior_x=prior_x)
    tables["sigma"] = sig
    tables["orders"] = orders
    tables["grid"] = gr
    return Calib(ak=ak, akx=akx, sigma=sl, vol=vol_flags(vol)), tables


def sigma_summary(sig: pl.DataFrame, vol: pl.DataFrame) -> pl.DataFrame:
    """Pooled realised variance (sum of squared 10 s mid changes / seconds observed) by group:
    ``var_p`` (cents^2/s), ``sd_1min_c`` / ``sd_1h_c`` (implied s.d. of the mid), ``var_x``
    (logit^2/s); ``binary_ratio`` = sum S / sum(Sx * mean[(100 p(1-p))^2]) per bucket (1.0 when
    sigma_p = sigma_logit * p(1-p) holds); ``signature`` = var(10 s) / var(60 s) (> 1: quote
    flicker that reverts within a minute); ``zero_frac``: buckets with no mid change at all."""
    f = add_regime(sig.with_columns(pl.col("bucket_ns").alias("t_ns")), vol)
    rows = []
    for keys in ([], ["volatile"], ["sbucket"], ["band"], ["band", "volatile"]):
        agg = [pl.len().alias("buckets"), pl.col("market").n_unique().alias("markets"),
               (pl.col("T").sum() / 3600).alias("hours"),
               (pl.col("S").sum() / pl.col("T").sum()).alias("var_p"),
               (pl.col("Sx").sum() / pl.col("T").sum()).alias("var_x"),
               (pl.col("S").sum() / (pl.col("Sx") * pl.col("pq2") * 1e4).sum()).alias("binary_ratio"),
               (pl.col("S60").sum() / pl.col("T60").sum()).alias("var_p60"),
               (pl.col("S") == 0).mean().alias("zero_frac")]
        a = f.group_by(keys).agg(*agg) if keys else f.select(*agg)
        a = a.with_columns((pl.col("var_p") * 60).sqrt().alias("sd_1min_c"),
                           (pl.col("var_p") * 3600).sqrt().alias("sd_1h_c"),
                           (pl.col("var_p") / pl.col("var_p60")).alias("signature"))
        for r in a.sort(keys or "buckets").iter_rows(named=True):
            rows.append({"group": "/".join(f"{k}={r[k]}" for k in keys) or "all",
                         **{k: v for k, v in r.items() if k not in keys}})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- model quotes (cents) ------------------------------------------------------------------------
def as_quotes(s_c: float, q: float, gamma: float, var: float, tau: float, k: float) -> tuple[float, float]:
    """Avellaneda–Stoikov bid and ask (cents): reservation ``s - q*gamma*var*tau``, total spread
    ``gamma*var*tau + (2/gamma) ln(1 + gamma/k)``."""
    r = s_c - q * gamma * var * tau
    total = gamma * var * tau + (2 / gamma) * math.log1p(gamma / k)
    return r - total / 2, r + total / 2


def glft_skew(gamma: float, var: float, A: float, k: float) -> float:
    """The GLFT inventory term ``c`` (cents per contract)."""
    return math.sqrt(gamma * var / (2 * A * k) * math.exp((1 + k / gamma) * math.log1p(gamma / k)))


def glft_quotes(s_c: float, q: float, gamma: float, var: float, A: float, k: float) -> tuple[float, float]:
    """Guéant–Lehalle–Fernandez-Tapia asymptotic bid and ask (cents), lot size 1, xi = gamma."""
    base = math.log1p(gamma / k) / gamma
    c = glft_skew(gamma, var, A, k)
    return s_c - (base + (2 * q + 1) / 2 * c), s_c + (base - (2 * q - 1) / 2 * c)


def as_logit_quotes(p: float, q: float, gamma: float, var_x: float, tau: float, k_x: float) -> tuple[float, float]:
    """Avellaneda–Stoikov in logit space, mapped back to cents."""
    x = logit(p)
    r = x - q * gamma * var_x * tau
    total = gamma * var_x * tau + (2 / gamma) * math.log1p(gamma / k_x)
    return 100 * expit(r - total / 2), 100 * expit(r + total / 2)


def to_ticks(bid_c: float | None, ask_c: float | None) -> tuple[float | None, float | None]:
    """Nearest 1c tick, exact half ticks rounded away from the mid (bid down, ask up)."""
    b = None if bid_c is None or not math.isfinite(bid_c) else math.ceil(bid_c - 0.5 - 1e-9) / 100
    a = None if ask_c is None or not math.isfinite(ask_c) else math.floor(ask_c + 0.5 + 1e-9) / 100
    return b, a


# ---- strategy ----------------------------------------------------------------------------------
class ModelQuoter:
    """1-lot two-sided quoting from ``as`` / ``glft`` / ``as_logit`` (see module doc).

    ``fair``: ``mid`` or ``micro`` (size-weighted external top of book). ``skew=False`` drops the
    inventory term (q = 0 in the formulas; the ``max_pos`` bound still applies). ``cooldown_s``:
    pull both sides for this long after any fill (the baseline's rule; 0 = rely on the skew).
    ``max_improve``: cap the price at this many ticks inside the external best (the model then only
    gates each side and skews; 1 = the baseline's improve rule).
    Decisions read only the external book after the current event, the current position and
    calibration fitted on earlier data (``SigmaLookup`` and the previous bucket's volatile flag).
    """

    def __init__(self, calib: Calib, *, model: str = "glft", gamma: float = 0.05, fair: str = "mid",
                 skew: bool = True, min_spread: float = 0.0, join: str = "front", horizon_s: float = 3600.0,
                 lo: float = 0.05, hi: float = 0.95, max_pos: float = 3.0, size: float = 1.0,
                 min_requote_s: float = 1.0, hyst_ticks: int = 0, cooldown_s: float = 0.0,
                 heartbeat_s: float = 5.0, max_improve: int | None = None) -> None:
        assert model in ("as", "glft", "as_logit") and join in ("front", "model") and fair in ("mid", "micro")
        self.c, self.model, self.gamma, self.fair, self.skew = calib, model, gamma, fair, skew
        self.min_spread, self.join, self.horizon_s = min_spread, join, horizon_s
        self.lo, self.hi, self.max_pos, self.size = lo, hi, max_pos, size
        self.requote_ns, self.hyst = int(min_requote_s * 1e9), hyst_ticks * TICK
        self.cool_ns, self.heartbeat_ns = int(cooldown_s * 1e9), int(heartbeat_s * 1e9)
        self.max_improve = max_improve
        self.bns = calib.bucket_s * 10**9
        self.live: dict[tuple[str, str], int] = {}
        self.last_place: dict[tuple[str, str], int] = {}
        self._seen: dict[str, tuple[int, float]] = {}
        self._next_eval: dict[str, int] = {}
        self._cool_until: dict[str, int] = {}
        self._seen_fills = 0
        self.diag: dict[str, int] = defaultdict(int)

    def targets(self, sim: Simulator, market: str, bid: float, ask: float) -> tuple[float | None, float | None]:
        """Model (YES bid, YES ask) in dollars after rounding, inventory bounds and clamping
        (never crossing the external book or ourselves); None = do not quote that side."""
        spread, mid = ask - bid, (ask + bid) / 2
        if spread < self.min_spread - EPS or not self.lo <= mid <= self.hi:
            return None, None
        pos = sim.position[market]
        q = pos if self.skew else 0.0
        s = mid
        if self.fair == "micro":
            qq = sim.markets[market].quote()
            bs, as_ = qq["yes_bid_size"] or 0.0, qq["yes_ask_size"] or 0.0
            if bs + as_ > 0:
                s = (bid * as_ + ask * bs) / (bs + as_)
        now = sim.now
        var_p, var_x = self.c.sigma.at(market, now)
        volp = self.c.vol.get((market, now // self.bns * self.bns - self.bns), False)
        sb = spread_bucket(spread * 100)
        exp_ns = self.c.expiry_ns.get(market)
        tau = self.horizon_s if exp_ns is None else min(self.horizon_s, max(60.0, (exp_ns - now) / 1e9))
        if self.model == "as":
            _, k = self.c.ak_at(sb, volp, price_band(mid))
            b_c, a_c = as_quotes(100 * s, q, self.gamma, var_p, tau, k)
        elif self.model == "glft":
            A, k = self.c.ak_at(sb, volp, price_band(mid))
            b_c, a_c = glft_quotes(100 * s, q, self.gamma, var_p, A, k)
        else:
            _, kx = self.c.akx_at(sb, volp)
            b_c, a_c = as_logit_quotes(s, q, self.gamma, var_x, tau, kx)
        b, a = to_ticks(b_c, a_c)
        if pos >= self.max_pos - EPS:
            b = None
        if pos <= -self.max_pos + EPS:
            a = None
        if b is not None:
            b = round(min(b, ask - TICK), 6)
            b = b if b >= TICK - EPS else None
        if a is not None:
            a = round(max(a, bid + TICK), 6)
            a = a if a <= 1 - TICK + EPS else None
        if self.max_improve is not None:       # the model as a gate: never more than n ticks inside
            b = None if b is None else round(min(b, bid + self.max_improve * TICK), 6)
            a = None if a is None else round(max(a, ask - self.max_improve * TICK), 6)
        if b is not None and a is not None and b >= a - EPS:
            # both sides want the same tick: give way on the side that adds to the position
            if pos > 0:
                b = round(a - TICK, 6) if a - TICK >= TICK - EPS else None
            else:
                a = round(b + TICK, 6) if b + TICK <= 1 - TICK + EPS else None
        return b, a

    def on_event(self, sim: Simulator, market: str) -> None:
        if self.cool_ns:
            for f in sim.fills[self._seen_fills:]:
                self._cool_until[f.market] = f.t_ns + self.cool_ns
        self._seen_fills = len(sim.fills)
        if market not in sim.synced:
            return
        qs = sim.quotes.get(market)
        if not qs or not qs[1]:
            return
        state = (len(qs[0]), sim.position[market])
        if self._seen.get(market) == state and sim.now < self._next_eval.get(market, 0):
            return
        self._seen[market] = state
        self._next_eval[market] = sim.now + self.heartbeat_ns
        self.diag["evals"] += 1
        bid, ask = qs[1][-1]
        b = a = None
        if bid is not None and ask is not None and sim.now >= self._cool_until.get(market, 0):
            b, a = self.targets(sim, market, bid, ask)
        sides = (("yes", b, bid), ("no", None if a is None else round(1 - a, 6),
                                   None if ask is None else round(1 - ask, 6)))
        for side, target, best in sides:
            key = (market, side)
            oid = self.live.get(key)
            o = sim.orders.get(oid) if oid is not None else None
            if o is not None and o.status not in ("pending", "active"):
                o = None
                self.live.pop(key, None)
            if o is not None:
                front = best is None or o.price >= best - EPS
                keep = (target is not None and (self.join == "model" or front)
                        and (abs(o.price - target) <= self.hyst + EPS
                             # capped order that others have since joined: still at the front
                             or (self.max_improve is not None and front
                                 and abs(o.price + TICK - target) < EPS)))
                if keep:
                    continue
                if not sim.cancel(o.oid):
                    self.diag["cancel_throttled"] += 1
                    continue
                self.live.pop(key, None)
            if target is None:
                continue
            if self.join == "front" and best is not None and target <= best + EPS:
                self.diag["behind_best"] += 1
                continue
            last = self.last_place.get(key)
            if last is not None and sim.now - last < self.requote_ns:
                self._next_eval[market] = min(self._next_eval[market], last + self.requote_ns)
                continue
            new = sim.place(market, side, target, self.size)
            if new is None:
                self._next_eval[market] = sim.now + 10**8
                continue
            self.diag["placed"] += 1
            self.live[key] = new
            self.last_place[key] = sim.now


# ---- lockstep replay -----------------------------------------------------------------------------
def prune(sim: Simulator, keep_s: float = 2.0) -> None:
    """Drop finished orders whose scheduled events have all run (memory and ``live_orders``)."""
    cut = sim.now - int(keep_s * 1e9)
    dead = {oid for oid, o in sim.orders.items()
            if o.status in ("filled", "cancelled", "rejected") and o.active_ns < cut and (o.cancel_ns or 0) < cut}
    if not dead:
        return
    for oid in dead:
        del sim.orders[oid]
    for m, ids in sim.by_market.items():
        sim.by_market[m] = [i for i in ids if i not in dead]


def run_lockstep(events: Iterable[Event], specs: list[tuple[str, float, Callable[[], Any]]],
                 fee_info: dict[str, dict], *, writes_per_s: float | None = 10.0,
                 prune_every: int = 20_000, log: Callable[[str], None] | None = None
                 ) -> dict[str, tuple[Simulator, Any]]:
    """Replay one event stream through one Simulator per spec ``(name, latency_ms, factory)``.
    Our orders never enter the external book, so every simulator sees the same top of book and
    they share one quote history (memory)."""
    shared: dict[str, tuple[list[int], list[tuple]]] = defaultdict(lambda: ([], []))
    runs: dict[str, tuple[Simulator, Any]] = {}
    for name, lat, make in specs:
        sim = Simulator(SimConfig(latency_ms=lat, fee_info=fee_info, writes_per_s=writes_per_s))
        sim.quotes = shared
        runs[name] = (sim, make())
    pairs = list(runs.values())
    t0 = time.time()
    for i, ev in enumerate(events):
        mk = ev.market
        for sim, strat in pairs:
            sim.apply(ev)
            strat.on_event(sim, mk)
        if prune_every and i % prune_every == prune_every - 1:
            for sim, _ in pairs:
                prune(sim)
            if log and (i + 1) % (25 * prune_every) == 0:
                log(f"{i + 1} events, {time.time() - t0:.0f}s, fills " +
                    " ".join(str(len(s.fills)) for s, _ in pairs[:6]))
    return runs


# ---- evaluation ----------------------------------------------------------------------------------
def last_mid(sim: Simulator, market: str) -> float | None:
    for b, a in reversed(sim.quotes.get(market, ([], []))[1]):
        if b is not None and a is not None:
            return (b + a) / 2
    return None


def fill_rows(sim: Simulator, vol: dict[tuple[str, int], bool], *, bucket_s: int = 300) -> list[dict[str, Any]]:
    """``backtest.l2.markouts`` per fill plus the event, the fill's own-bucket ``volatile`` flag
    (evaluation split) and ``mark``: value at the last recorded mid minus price and fee."""
    bns = bucket_s * 10**9
    lm: dict[str, float | None] = {}
    rows = markouts(sim)
    for r in rows:
        m = r["market"]
        if m not in lm:
            lm[m] = last_mid(sim, m)
        v = lm[m]
        r["event"] = m.rsplit("-", 1)[0]
        r["volatile"] = vol.get((m, r["t_ns"] // bns * bns), False)
        r["last_mid"] = v
        r["mark"] = None if v is None else (v if r["side"] == "yes" else 1 - v) - r["price"] - r["fee"] / r["qty"]
    return rows


SUMMARY_COLS = ("mo_60s", "mo_300s", "mx_60s", "mx_300s", "mark")


def clustered(rows: list[dict[str, Any]], col: str, key: str = "event") -> tuple[float | None, float | None, int]:
    """Contract-weighted mean of ``col`` and its t-stat with errors clustered by ``key`` (None
    with fewer than two clusters or no dispersion)."""
    g: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        if r.get(col) is None:
            continue
        g[r[key]][0] += r[col] * r["qty"]
        g[r[key]][1] += r["qty"]
    if not g:
        return None, None, 0
    n = sum(v[1] for v in g.values())
    mean = sum(v[0] for v in g.values()) / n
    k = len(g)
    if k < 2:
        return mean, None, k
    se = math.sqrt(k / (k - 1) * sum((a - mean * b) ** 2 for a, b in g.values())) / n
    return mean, (mean / se if se > 1e-9 else None), k


def summarize(rows: list[dict[str, Any]], results: dict[str, float | None]) -> dict[str, Any]:
    """Contract-weighted means in cents with t-stats clustered by event, totals in dollars."""
    from albert.research.maker_variants import round_trips

    out: dict[str, Any] = {"fills": len(rows), "contracts": round(sum(r["qty"] for r in rows), 2),
                           "events": len({r["event"] for r in rows}), "markets": len({r["market"] for r in rows})}
    for col in SUMMARY_COLS:
        mean, t, _ = clustered(rows, col)
        out[col] = None if mean is None else round(100 * mean, 2)
        out[f"t_{col}"] = None if t is None else round(t, 2)
    marked = [r for r in rows if r["mark"] is not None]
    out["pnl_mark"] = round(sum(r["mark"] * r["qty"] for r in marked), 3)
    out["fees"] = round(sum(r["fee"] for r in rows), 3)
    st = []
    for r in rows:
        res = results.get(r["market"])
        if res is not None:
            st.append({**r, "settle": (res if r["side"] == "yes" else 1 - res) - r["price"] - r["fee"] / r["qty"]})
    mean, t, _k = clustered(st, "settle")
    out["settled_n"] = len(st)
    out["settle"] = None if mean is None else round(100 * mean, 2)
    out["t_settle"] = None if t is None else round(t, 2)
    rt = round_trips(rows) if rows else {}
    out["rt_frac"] = round(rt.get("paired_frac", 0.0), 3)
    out["rt_c"] = None if rt.get("paired_c_per_pair") is None else round(rt["paired_c_per_pair"], 2)
    return out


def fill_context(fills: pl.DataFrame, tob: pl.DataFrame) -> pl.DataFrame:
    """Add the external spread just before each fill (``spread_c``, ``sbucket``) and how many
    ticks inside the external best on our side the filled order sat (``ticks_in``; <= 0 when the
    book had caught up with or passed our price)."""
    f = fills.with_columns((pl.col("t_ns") - 1).alias("tq")).sort("tq")
    f = f.join_asof(tob.rename({"t_ns": "tq"}).sort("tq"), on="tq", by="market", strategy="backward")
    f = f.with_columns(((pl.col("ask") - pl.col("bid")) * 100).round(1).alias("spread_c"),
                       pl.when(pl.col("side") == "yes").then((pl.col("price") - pl.col("bid")) * 100)
                       .otherwise((pl.col("price") - (1 - pl.col("ask"))) * 100).round(1).alias("ticks_in"))
    return f.with_columns(
        pl.when(pl.col("ticks_in") <= 0).then(pl.lit("<=0")).when(pl.col("ticks_in") <= 1).then(pl.lit("1"))
        .when(pl.col("ticks_in") <= 2).then(pl.lit("2")).otherwise(pl.lit(">=3")).alias("ticks_in_b"),
        _spread_bucket_expr("spread_c")).drop("tq", "bid", "ask")


def context_table(fills: pl.DataFrame, by: str) -> pl.DataFrame:
    """Markouts by config, window and ``by`` (a ``fill_context`` column)."""
    rows = []
    keys = ["config", "window", by]
    for k, g in fills.group_by(keys, maintain_order=True):
        rr = g.to_dicts()
        r: dict[str, Any] = {**dict(zip(keys, k)), "fills": len(rr)}
        for col in ("mo_60s", "mo_300s", "mx_300s", "mark"):
            m, t, _ = clustered(rr, col)
            r[col] = None if m is None else round(100 * m, 2)
            r[f"t_{col}"] = None if t is None else round(t, 2)
        rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None).sort(keys)


def summary_table(fills: pl.DataFrame, results: dict[str, float | None]) -> pl.DataFrame:
    """One row per (config, latency, window, regime) with regime in all / volatile / calm."""
    out = []
    keys = ["config", "latency_ms", "window"]
    for key, f in fills.group_by(keys, maintain_order=True):
        rows = f.to_dicts()
        for regime, sub in (("all", rows), ("volatile", [r for r in rows if r["volatile"]]),
                            ("calm", [r for r in rows if not r["volatile"]])):
            out.append({**dict(zip(keys, key)), "regime": regime, **summarize(sub, results)})
    return pl.DataFrame(out, infer_schema_length=None)


# ---- metadata (public REST) ----------------------------------------------------------------------
def market_meta(tickers: list[str], cache: Path, *, refresh_unsettled: bool = False,
                pause_s: float = 0.05) -> dict[str, dict[str, Any]]:
    """Public Kalshi market metadata (expiry, status, result), cached as JSON. With
    ``refresh_unsettled`` markets without a result are fetched again."""
    import httpx

    have: dict[str, dict[str, Any]] = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [t for t in tickers if t not in have or (refresh_unsettled and not have[t].get("result"))]
    if todo:
        with httpx.Client(timeout=20) as http:
            for t in todo:
                r = http.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{t}")
                if r.status_code == 200:
                    m = r.json().get("market", {})
                    have[t] = {k: m.get(k) for k in ("expected_expiration_time", "close_time", "status", "result")}
                time.sleep(pause_s)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(have, indent=1))
    return have


def _iso_ns(s: str | None) -> int | None:
    from datetime import datetime

    if not s:
        return None
    return int(datetime.fromisoformat(s).timestamp() * 1e9)


def _hour_ns(key: str) -> int:
    from datetime import UTC, datetime

    return int(datetime.strptime(key, "%Y-%m-%dT%H").replace(tzinfo=UTC).timestamp() * 1e9)


# ---- sweep -------------------------------------------------------------------------------------
FAMILIES: dict[str, dict[str, Any]] = {"as": {"model": "as"}, "asmic": {"model": "as", "fair": "micro"},
                                       "glft": {"model": "glft"}, "aslogit": {"model": "as_logit"}}
BASELINE = {"improve": True, "min_spread": 0.05, "cooldown_s": 300}
CENTS_PER_LOGIT = 25.0          # dp/dx at p = 0.5, in cents


def gamma_grid(calib: Calib, targets: tuple[float, ...] = (0.5, 1.5, 4.0),
               horizon_s: float = 3600.0) -> dict[str, list[float]]:
    """Risk aversion per family chosen so the inventory term per contract equals ``targets``
    (cents) at the calm prior variance (and, for GLFT, the pooled A and k)."""
    pp, px = calib.sigma.prior
    A0, k0 = calib.ak[(None, None, None)]

    def solve(target: float) -> float:
        lo_, hi_ = math.log(1e-7), math.log(10.0)
        for _ in range(100):
            mid = (lo_ + hi_) / 2
            if glft_skew(math.exp(mid), pp, A0, k0) < target:
                lo_ = mid
            else:
                hi_ = mid
        return math.exp((lo_ + hi_) / 2)

    g_as = [t / (pp * horizon_s) for t in targets]
    return {"as": g_as, "asmic": g_as, "glft": [solve(t) for t in targets],
            "aslogit": [(t / CENTS_PER_LOGIT) / (px * horizon_s) for t in targets]}


def sweep_specs(grid_: dict[str, list[float]]) -> dict[str, dict[str, Any] | None]:
    specs: dict[str, dict[str, Any] | None] = {"base": None}
    for fam, kw in FAMILIES.items():
        for i, g in enumerate(grid_[fam]):
            for ms in (0, 5):
                specs[f"{fam}_g{i}_ms{ms}"] = {**kw, "gamma": g, "min_spread": ms / 100}
    return specs


def ablation_specs(chosen: dict[str, str], specs: dict[str, dict[str, Any] | None]) -> dict[str, dict[str, Any]]:
    out = {}
    for name in chosen.values():
        kw = specs[name] or {}
        out[f"{name}_noskew"] = {**kw, "skew": False}
        out[f"{name}_cool"] = {**kw, "cooldown_s": 300.0}
        out[f"{name}_join"] = {**kw, "join": "model"}
    return out


def choose(summ: pl.DataFrame, *, window: str = "dev", latency: float = 50.0, min_fills: int = 30,
           pattern: str = r"^{fam}_g\d_ms\d$", families: Iterable[str] = tuple(FAMILIES)) -> dict[str, str]:
    """Per family, the dev config with the best marked-to-last-mid PnL per contract among those
    with at least ``min_fills`` fills (else the one with most fills)."""
    rows = summ.filter((pl.col("window") == window) & (pl.col("latency_ms") == latency) & (pl.col("regime") == "all"))
    out = {}
    for fam in families:
        cand = rows.filter(pl.col("config").str.contains(pattern.format(fam=fam)))
        if cand.is_empty():
            continue
        ok = cand.filter((pl.col("fills") >= min_fills) & pl.col("mark").is_not_null())
        pick = (ok.sort(["mark", "mx_300s"], descending=True, nulls_last=True) if not ok.is_empty()
                else cand.sort("fills", descending=True))
        out[fam] = pick["config"][0]
    return out


def run_window(s, *, root: Path, start: str, end: str, window: str,
               specs: list[tuple[str, float, dict[str, Any] | None]], calib: Calib,
               fee_info: dict[str, dict], log: Callable[[str], None] | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Replay ``[start, end)`` once for all specs; per-fill rows and per-run diagnostics, cached
    under ``data/cache/mm_models_run_*`` (keyed by the window and the exact specs)."""
    import hashlib

    key = json.dumps([[n, lat, kw] for n, lat, kw in specs], sort_keys=True, default=str)
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    base = s.data_dir / "cache" / f"mm_models_run_{window}_{start}_{end}_{h}".replace(":", "")
    fp, dp = base.with_suffix(".fills.parquet"), base.with_suffix(".diag.parquet")
    if fp.exists() and dp.exists():
        return pl.read_parquet(fp), pl.read_parquet(dp)

    def factory(kw: dict[str, Any] | None) -> Callable[[], Any]:
        if kw is None:
            return lambda: Quoter(**BASELINE)
        return lambda: ModelQuoter(calib, **kw)

    stats: dict[str, int] = defaultdict(int)
    files = stream_files(root, start, end)
    runs = run_lockstep(iter_events(files, stats), [(f"{n}@{lat:g}", lat, factory(kw)) for n, lat, kw in specs],
                        fee_info, log=log)
    rows, diag = [], []
    for (n, lat, kw), (name, (sim, strat)) in zip(specs, runs.items()):
        fr = fill_rows(sim, calib.vol, bucket_s=calib.bucket_s)
        for r in fr:
            r.update(config=n, latency_ms=lat, window=window)
        rows.extend(fr)
        d = {"config": n, "latency_ms": lat, "window": window, "orders": sim._next - 1, "fills": len(sim.fills),
             **{f"sim_{k}": v for k, v in sim.diag.items() if k in ("throttled_places", "throttled_cancels",
                                                                     "post_only_rejects")},
             **{f"q_{k}": v for k, v in getattr(strat, "diag", {}).items()}}
        diag.append(d)
    schema = {"t_ns": pl.Int64, "market": pl.String, "side": pl.String, "price": pl.Float64, "qty": pl.Float64,
              "fee": pl.Float64, "how": pl.String, "mo_1s": pl.Float64, "mo_10s": pl.Float64, "mo_60s": pl.Float64,
              "mo_300s": pl.Float64, "mx_60s": pl.Float64, "mx_300s": pl.Float64, "event": pl.String,
              "volatile": pl.Boolean, "last_mid": pl.Float64, "mark": pl.Float64, "config": pl.String,
              "latency_ms": pl.Float64, "window": pl.String}
    fills = pl.DataFrame(rows, schema=schema)
    dg = pl.DataFrame(diag, infer_schema_length=None)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fills.write_parquet(fp)
    dg.write_parquet(dp)
    return fills, dg


def model_quotes_table(calib: Calib, grid_: dict[str, list[float]], vol_var: tuple[float, float],
                       horizon_s: float = 3600.0) -> pl.DataFrame:
    """Half-spread and inventory skew per contract (cents) each family/gamma implies at a 50c
    mid, for the calm prior and the pooled volatile variance, per spread bucket."""
    rows = []
    for fam in ("as", "glft", "aslogit"):
        for i, g in enumerate(grid_[fam]):
            for label, (vp, vx), volp in (("calm", calib.sigma.prior, False), ("volatile", vol_var, True)):
                for sb in [n for _, n in SPREAD_EDGES]:
                    A, k = calib.ak_at(sb, volp, "center")
                    if fam == "as":
                        b0, a0 = as_quotes(50, 0, g, vp, horizon_s, k)
                        b1, a1 = as_quotes(50, 1, g, vp, horizon_s, k)
                    elif fam == "glft":
                        b0, a0 = glft_quotes(50, 0, g, vp, A, k)
                        b1, a1 = glft_quotes(50, 1, g, vp, A, k)
                    else:
                        _, kx = calib.akx_at(sb, volp)
                        b0, a0 = as_logit_quotes(0.5, 0, g, vx, horizon_s, kx)
                        b1, a1 = as_logit_quotes(0.5, 1, g, vx, horizon_s, kx)
                    rows.append({"family": fam, "g": i, "gamma": g, "sigma": label, "sbucket": sb,
                                 "A": A, "k": k, "half_spread_c": round((a0 - b0) / 2, 2),
                                 "skew_per_contract_c": round(((a0 + b0) - (a1 + b1)) / 2, 2)})
    return pl.DataFrame(rows)


def mm_models(s, *, start: str = "2026-11-05T03", split: str = "2026-11-05T11", end: str = "2026-11-05T18",
              root: Path | None = None, targets: tuple[float, ...] = (0.5, 1.5, 4.0),
              latencies: tuple[float, float] = (50.0, 100.0), fetch_meta: bool = True,
              log: Callable[[str], None] | None = print) -> Path:
    """Calibrate on ``[start, split)``, sweep gamma on dev, replay dev-chosen configs on
    ``[split, end)``; write report ``research-mm-models``."""
    from albert.reports import Report
    from albert.research.volatile import cached_buckets

    root = root or s.raw_dir / "stream"
    lat0, lat1 = latencies
    data = cached_extract(s, start, end, root)
    vol = cached_buckets(s, start, end, root=root)
    split_ns = _hour_ns(split)
    calib, tabs = calibrate(data, vol, until_ns=split_ns)
    markets = sorted(data["tob"]["market"].unique().to_list())
    meta_cache = s.data_dir / "cache" / "mm_models_markets.json"
    meta = market_meta(markets, meta_cache) if fetch_meta else {}
    calib.expiry_ns = {t: n for t, m in meta.items()
                       if (n := _iso_ns(m.get("expected_expiration_time") or m.get("close_time"))) is not None}
    grid_ = gamma_grid(calib, targets)
    specs = sweep_specs(grid_)
    fee_info = {r["series_ticker"]: r for r in pl.read_parquet(s.tables_dir / "kalshi_series.parquet").iter_rows(named=True)}
    if log:
        log(f"calibrated; gamma grid {grid_}")
    # pass 1: dev sweep at the base latency
    dev1, diag1 = run_window(s, root=root, start=start, end=split, window="dev",
                             specs=[(n, lat0, kw) for n, kw in specs.items()], calib=calib, fee_info=fee_info, log=log)
    results: dict[str, float | None] = {}
    summ1 = summary_table(dev1, results)
    chosen = choose(summ1)
    if log:
        log(f"chosen on dev: {chosen}")
    abl = ablation_specs(chosen, specs)
    extra = [(n, lat1, specs[n]) for n in ["base", *chosen.values()]] + [(n, lat0, kw) for n, kw in abl.items()]
    dev2, diag2 = run_window(s, root=root, start=start, end=split, window="dev", specs=extra, calib=calib,
                             fee_info=fee_info, log=log)
    hold, diag3 = run_window(s, root=root, start=split, end=end, window="holdout",
                             specs=[(n, lat0, kw) for n, kw in specs.items()] + extra, calib=calib,
                             fee_info=fee_info, log=log)
    # passes 4-6 (designed after the dev sweep, before any holdout result was read): the model as a
    # gate. Price capped at 1 tick inside the best (the baseline's improve rule), 5c minimum spread;
    # the model decides per side whether to quote (its price must reach best + 1 tick) and skews.
    fams = ("as", "glft", "aslogit")
    gate: dict[str, dict[str, Any]] = {}
    for fam in fams:
        for i, g in enumerate(grid_[fam]):
            kw = {**FAMILIES[fam], "gamma": g, "min_spread": 0.05, "max_improve": 1}
            gate[f"{fam}_g{i}_gate"] = kw
            gate[f"{fam}_g{i}_gate_cool"] = {**kw, "cooldown_s": 300.0}
    devg, diag4 = run_window(s, root=root, start=start, end=split, window="dev",
                             specs=[(n, lat0, kw) for n, kw in gate.items()], calib=calib, fee_info=fee_info, log=log)
    sg = summary_table(devg, {})
    gate_chosen = [*choose(sg, pattern=r"^{fam}_g\d_gate$", families=fams).values(),
                   *choose(sg, pattern=r"^{fam}_g\d_gate_cool$", families=fams).values()]
    if log:
        log(f"gate chosen on dev: {gate_chosen}")
    gextra = ([(f"{n}_noskew", lat0, {**gate[n], "skew": False}) for n in gate_chosen]
              + [(n, lat1, gate[n]) for n in gate_chosen])
    devg2, diag5 = run_window(s, root=root, start=start, end=split, window="dev", specs=gextra, calib=calib,
                              fee_info=fee_info, log=log)
    holdg, diag6 = run_window(s, root=root, start=split, end=end, window="holdout",
                              specs=[(n, lat0, gate[n]) for n in gate_chosen] + gextra, calib=calib,
                              fee_info=fee_info, log=log)
    fills = pl.concat([dev1, dev2, hold, devg, devg2, holdg])
    if fetch_meta:
        meta = market_meta(sorted(fills["market"].unique().to_list()), meta_cache, refresh_unsettled=True)
    results = {t: (1.0 if m.get("result") == "yes" else 0.0 if m.get("result") == "no" else None)
               for t, m in meta.items()}
    summ = summary_table(fills, results)
    diag = pl.concat([diag1, diag2, diag3, diag4, diag5, diag6], how="diagonal_relaxed")

    rep = Report(s.reports_dir, "research-mm-models")
    rep.json("Run", {"start": start, "split": split, "end": end, "latencies_ms": latencies,
                     "tob_changes": data["tob"].height, "trades": data["trades"].height, "markets": len(markets),
                     "dev_taker_orders": tabs["orders"].height, "sigma_prior_calm_c2_per_s": calib.sigma.prior[0],
                     "sigma_prior_calm_logit2_per_s": calib.sigma.prior[1], "gamma_grid": grid_,
                     "skew_targets_c": targets, "chosen_on_dev": chosen, "gate_chosen_on_dev": gate_chosen,
                     "baseline": BASELINE,
                     "settled_markets_with_fills": sum(1 for t in fills["market"].unique().to_list()
                                                       if results.get(t) is not None)})
    rep.text("Reading guide", "Prices and markouts in cents per contract; t-stats clustered by event. "
             "`mo_*` = mid markout, `mx_*` = exit-price markout (best bid on our side), `mark` = value at the "
             "last recorded mid of the window minus price and maker fee, `pnl_mark` in dollars. Regime "
             "split by each fill's own 5-min bucket `volatile` flag (evaluation only); strategies use only "
             "the previous bucket's flag. Calibration (A, k, sigma prior) is fitted on dev only.")
    rep.table("Volatility (dev, pooled realised variance of the mid on a 10 s grid)",
              sigma_summary(tabs["sigma"], vol))
    for name, title in (("fit_all", "all"), ("fit_all_touch", "all, touch (>= delta) instead of through"),
                        ("fit_spread", "by spread bucket"), ("fit_spread_vol", "by spread x previous-bucket volatile"),
                        ("fit_own_vol", "by own-bucket volatile"), ("fit_band", "by price band"),
                        ("fit_x_all", "logit distance, all"), ("fit_x_spread_vol", "logit distance, spread x volatile")):
        rep.table(f"Intensity lambda(delta) per side per market-second, taker reach strictly beyond delta: {title}",
                  tabs[name])
    rep.table("Intensity by regime (spread x previous-bucket volatile x band; cells with >= 30 orders feed the models)",
              tabs["fit_regime"].sort("orders", descending=True), max_rows=40)
    lam = tabs["lam_spread"].pivot(on="delta", index="sbucket", values="lam")
    rep.table("lambda(delta) by spread bucket (per side per market-second; columns = delta in cents)", lam)
    g = tabs["grid"].select("market", "t_ns", "bid", "ask", "mid", "spread_c")
    ff = fill_fraction(g, data["trades"].filter(pl.col("t_ns") < split_ns),
                       data["tob"].filter(pl.col("t_ns") < split_ns), vol)
    rep.table("Traded-through fraction of a 1-lot at mid0 +/- delta within H (dev, 30 s grid)", ff)
    _, htabs = calibrate(data, vol, until_ns=_hour_ns(end), from_ns=split_ns)
    for name in ("fit_all", "fit_spread", "fit_own_vol"):
        rep.table(f"Holdout intensity fit (stability check, not used by any strategy): {name[4:]}", htabs[name])
    vv = add_regime(tabs["sigma"].with_columns(pl.col("bucket_ns").alias("t_ns")), vol).filter(pl.col("volatile"))
    vol_var = (float(vv["S"].sum() / vv["T"].sum()), float(vv["Sx"].sum() / vv["T"].sum())) if vv.height else calib.sigma.prior
    rep.table("Implied quotes at a 50c mid, q = 0 vs 1 (calm prior vs pooled volatile variance)",
              model_quotes_table(calib, grid_, vol_var), max_rows=90)
    cols = ["config", "latency_ms", "window", "regime", "fills", "contracts", "events", "mo_60s", "mo_300s",
            "t_mo_300s", "mx_60s", "mx_300s", "t_mx_300s", "mark", "t_mark", "pnl_mark", "fees", "rt_frac", "rt_c",
            "settled_n", "settle"]
    rep.table(f"Dev sweep ({lat0:g} ms, all regimes)",
              summ.filter((pl.col("window") == "dev") & (pl.col("latency_ms") == lat0) & (pl.col("regime") == "all")
                          & ~pl.col("config").str.contains("_(noskew|cool|join)$|_gate")).select(cols), max_rows=40)
    key_cfgs = ["base", *chosen.values()]
    rep.table("Baseline vs dev-chosen configs: all / volatile / calm x dev / holdout x latency",
              summ.filter(pl.col("config").is_in(key_cfgs)).sort("config", "latency_ms", "window", "regime").select(cols),
              max_rows=80)
    rep.table(f"Ablations of the chosen configs ({lat0:g} ms): no inventory skew, 300 s cooldown, join rule = model",
              summ.filter(pl.col("config").is_in(key_cfgs + list(abl)) & (pl.col("latency_ms") == lat0))
              .sort("config", "window", "regime").select(cols), max_rows=120)
    rep.table(f"Model as a gate (price capped at best + 1 tick, min spread 5c): dev sweep ({lat0:g} ms, all regimes)",
              summ.filter((pl.col("window") == "dev") & (pl.col("latency_ms") == lat0) & (pl.col("regime") == "all")
                          & pl.col("config").str.contains("_gate(_cool)?$")).select(cols), max_rows=40)
    rep.table("Model as a gate, dev-chosen vs baseline: all / volatile / calm x dev / holdout x latency, "
              "with the no-skew ablation",
              summ.filter(pl.col("config").is_in(["base", *gate_chosen, *[f"{n}_noskew" for n in gate_chosen]]))
              .sort("config", "latency_ms", "window", "regime").select(cols), max_rows=120)
    ctx = fill_context(fills.filter(pl.col("config").is_in(["base", *chosen.values(), *gate_chosen])
                                    & (pl.col("latency_ms") == lat0)), data["tob"])
    rep.table(f"Where the fills sit: ticks inside the external best at the fill ({lat0:g} ms)",
              context_table(ctx, "ticks_in_b"), max_rows=80)
    rep.table(f"Fills by external spread at the fill ({lat0:g} ms)", context_table(ctx, "sbucket"), max_rows=80)
    rep.table(f"Holdout, whole grid ({lat0:g} ms, all regimes; not used for selection)",
              summ.filter((pl.col("window") == "holdout") & (pl.col("latency_ms") == lat0) & (pl.col("regime") == "all")
                          & ~pl.col("config").str.contains("_(noskew|cool|join)$|_gate")).select(cols), max_rows=40)
    rep.table("Run diagnostics (orders placed, throttles, post-only rejects, quoter counters)", diag, max_rows=80)
    summ.write_parquet(rep.dir / "summary.parquet")
    fills.write_parquet(rep.dir / "fills.parquet")
    return rep.save()
