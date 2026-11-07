"""Volatile windows in the L2 recordings: the shared definition every strategy study filters on.

The recorded hours are cut into ``bucket_s`` buckets per market. For each bucket we measure, from
the replayed top of book (the Simulator's own book, no strategy):

* ``path_c``: total mid variation in cents, counted only while the spread is at most
  ``TIGHT`` (6c) so quote flicker in wide books does not count as volatility; ``path_all_c``
  counts every two-sided update,
* ``range_c``: high minus low mid, ``spread_med_c``: median quoted spread (time-weighted by
  update, not by duration), ``updates``: top-of-book changes, ``trades`` / ``trade_qty``: prints.

A bucket is **volatile** when ``path_c`` is in the top ``q`` of all buckets with a two-sided book,
or when its market is a sports game **in play** (scheduled start = ``expected_expiration_time``
minus the league's game length, until ``expected_expiration_time`` plus an hour). The volatile
flag of a bucket uses only that bucket's own data, so a strategy that conditions on it at
decision time must use the *previous* bucket's flag (``volatile_prev``), never the current one,
to stay free of look-ahead. Studies that only *select* windows for evaluation may use either, and
must say which.
"""

from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from albert.backtest.l2 import (
    EPS,
    Event,
    SimConfig,
    Simulator,
    iter_events,
    stream_files,
    trade_key,
)

TIGHT = 0.06

# Kalshi game series -> typical game length in hours (scheduled start = expected expiration - this).
GAME_SERIES = {"KXNBAGAME": 2.6, "KXNHLGAME": 2.6, "KXNFLGAME": 3.4, "KXNCAAFGAME": 3.6,
               "KXNCAAMBGAME": 2.2, "KXNCAAWBGAME": 2.2, "KXWNBAGAME": 2.4, "KXMLBGAME": 3.2,
               "KXEPLGAME": 2.0, "KXUCLGAME": 2.0, "KXLALIGAGAME": 2.0, "KXSERIEAGAME": 2.0,
               "KXBUNDESLIGAGAME": 2.0, "KXLIGUE1GAME": 2.0, "KXMLSGAME": 2.0}


def series_of(ticker: str) -> str:
    return ticker.split("-", 1)[0]


def _in_play_bounds(info: dict[str, dict[str, Any]]) -> dict[str, tuple[int, int]]:
    out = {}
    for t, m in info.items():
        hours = GAME_SERIES.get(series_of(t))
        exp = (m or {}).get("expected_expiration_time")
        if hours is None or not exp:
            continue
        e = datetime.fromisoformat(exp)
        out[t] = (int((e - timedelta(hours=hours)).timestamp() * 1e9),
                  int((e + timedelta(hours=1)).timestamp() * 1e9))
    return out


def market_meta(tickers: list[str], cache: Path) -> dict[str, dict[str, Any]]:
    """Public market metadata for game markets only (cached JSON)."""
    from albert.research.leadlag_fast import market_info

    return market_info(sorted(t for t in tickers if series_of(t) in GAME_SERIES), cache)


def buckets(root: Path, start: str | None, end: str | None, *, bucket_s: int = 300,
            meta: dict[str, dict[str, Any]] | None = None, meta_cache: Path | None = None) -> pl.DataFrame:
    """One row per (market, bucket) with activity measures, ``in_play`` and ``volatile`` flags
    (see module doc). ``meta`` (ticker -> {expected_expiration_time}) or ``meta_cache`` supplies
    the in-play schedule; without either, ``in_play`` is false everywhere."""
    stats: dict[str, int] = defaultdict(int)
    return from_events(iter_events(stream_files(root, start, end), stats), bucket_s=bucket_s,
                       meta=meta, meta_cache=meta_cache)


def from_events(events: Iterable[Event], *, bucket_s: int = 300,
                meta: dict[str, dict[str, Any]] | None = None, meta_cache: Path | None = None) -> pl.DataFrame:
    """``buckets`` on an event stream (executions already marked)."""
    sim = Simulator(SimConfig(writes_per_s=None))
    bns = int(bucket_s * 1e9)
    trades: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0, 0.0])
    for ev in events:
        sim.apply(ev)
        if ev.kind == "trade":
            k = (ev.market, ev.recv_ns // bns)
            trades[k][0] += 1
            trades[k][1] += trade_key(ev.msg)[2]
    acc: dict[tuple[str, int], dict[str, Any]] = {}
    for mkt, (times, vals) in sim.quotes.items():
        prev_mid = None
        for t, (bid, ask) in zip(times, vals):
            k = (mkt, t // bns)
            a = acc.setdefault(k, {"path": 0.0, "path_all": 0.0, "hi": None, "lo": None, "spreads": [],
                                 "updates": 0})
            a["updates"] += 1
            if bid is None or ask is None:
                prev_mid = None
                continue
            mid = (bid + ask) / 2
            a["spreads"].append(ask - bid)
            a["hi"] = mid if a["hi"] is None else max(a["hi"], mid)
            a["lo"] = mid if a["lo"] is None else min(a["lo"], mid)
            if prev_mid is not None:
                a["path_all"] += abs(mid - prev_mid)
                if ask - bid <= TIGHT + EPS:
                    a["path"] += abs(mid - prev_mid)
            prev_mid = mid
    if meta is None and meta_cache is not None:
        meta = market_meta(sorted(sim.quotes), meta_cache)
    bounds = _in_play_bounds(meta or {})
    rows = []
    for (mkt, b), a in acc.items():
        sp = sorted(a["spreads"])
        t0 = b * bns
        lo_hi = bounds.get(mkt)
        rows.append({
            "market": mkt, "series": series_of(mkt), "bucket_ns": t0,
            "bucket": datetime.fromtimestamp(t0 / 1e9, UTC),
            "path_c": round(a["path"] * 100, 3), "path_all_c": round(a["path_all"] * 100, 3),
            "range_c": None if a["hi"] is None else round((a["hi"] - a["lo"]) * 100, 3),
            "spread_med_c": round(sp[len(sp) // 2] * 100, 3) if sp else None,
            "updates": a["updates"], "trades": trades.get((mkt, b), [0, 0])[0],
            "trade_qty": trades.get((mkt, b), [0, 0.0])[1],
            "in_play": bool(lo_hi and lo_hi[0] <= t0 + bns // 2 <= lo_hi[1]),
            "game": lo_hi is not None,
        })
    return flag(pl.DataFrame(rows))


def flag(df: pl.DataFrame, q: float = 0.8) -> pl.DataFrame:
    """Add ``vol_pct`` (percentile of ``path_c``), ``volatile`` (``path_c`` in the top ``q`` of
    buckets that moved at all at a tight spread, or in play) and the
    look-ahead-free ``volatile_prev`` (the market's previous bucket was volatile)."""
    two = df.filter(pl.col("path_c") > 0)
    cut = two["path_c"].quantile(q) if two.height else None
    df = df.with_columns(
        pl.col("path_c").rank("average").truediv(pl.len()).alias("vol_pct"),
        ((pl.col("path_c") >= (cut if cut is not None else float("inf"))) & (pl.col("path_c") > 0)
         | pl.col("in_play")).alias("volatile"),
    ).sort("market", "bucket_ns")
    return df.with_columns(pl.col("volatile").shift(1).over("market").fill_null(False).alias("volatile_prev"))


def cached_buckets(s, start: str | None, end: str | None, *, bucket_s: int = 300,
                   root: Path | None = None) -> pl.DataFrame:
    """``buckets`` over the recorder's stream, cached as Parquet under ``data/cache`` (derived,
    rebuildable; the raw stream is never touched)."""
    root = root or s.raw_dir / "stream"
    tag = f"{start or 'all'}_{end or 'all'}_{bucket_s}".replace(":", "")
    path = s.data_dir / "cache" / f"volatile_buckets_{tag}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    df = buckets(root, start, end, bucket_s=bucket_s,
                 meta_cache=s.data_dir / "cache" / "volatile_markets.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def summary(df: pl.DataFrame) -> dict[str, Any]:
    v = df.filter("volatile")
    return {"buckets": df.height, "markets": df["market"].n_unique(), "volatile_buckets": v.height,
            "volatile_markets": v["market"].n_unique(), "in_play_buckets": df["in_play"].sum(),
            "in_play_markets": df.filter("in_play")["market"].n_unique(),
            "path_cut_c": df.filter(pl.col("path_c") > 0)["path_c"].quantile(0.8)}
