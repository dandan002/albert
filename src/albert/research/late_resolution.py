"""Late-resolution drift on non-financial Kalshi markets (research swarm 2026-11-23, slug ``late_resolution``).

Questions (hypotheses in ``data/research/late_resolution/hypotheses.md``, written before outcome data):

* **Census** (H2): how long after close each series settles (capital lock), how often markets close
  early, and how long Kalshi keeps a market trading after its own ``occurrence_datetime`` stamp.
* **Determined-but-open window** (H1): in markets whose occurrence stamp falls strictly inside
  (open, close), the event has happened while the book still trades until Kalshi's manual close.
  Public trade prints and minute candles over [occurrence - 30 min, close] measure how much a
  perfect-knowledge taker or a resting bid on the eventual winner could have earned (an upper bound),
  then frozen dev rules with only real-time inputs are scored on the holdout once.
* **Tail risk / long-lag premium** (H3, H5): reversal rate of legs priced >= 97c shortly before close,
  and whether near-certain legs in long settlement-lag series are cheaper.

Data: ``data/tables/kalshi_markets.parquet`` (non-sports closes 2026-08-15..10-31) plus settled markets
of the same series closing 2026-11-01..11-23 fetched from Kalshi's public ``/markets`` endpoint; public
``/markets/trades`` (or ``/historical/trades``) prints and minute candlesticks for the sampled windows.
Every response is cached raw under ``data/cache/late_resolution/raw`` (read-only reuse of identical
requests already cached by earlier studies under ``data/raw`` / ``data/cache/*/raw``). One client,
>= 0.5 s between requests. Dev = closes before 2026-10-01, holdout = 2026-10-01..11-22.

    python -m albert.research.late_resolution base        # lean market frame from the shared table
    python -m albert.research.late_resolution fetch-nov   # Nov settled markets of the same series
    python -m albert.research.late_resolution census
    python -m albert.research.late_resolution sample
    python -m albert.research.late_resolution fetch-windows
    python -m albert.research.late_resolution analyze
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

OUT = Path("data/research/late_resolution")
CACHE = Path("data/cache/late_resolution")
TABLES = Path("data/tables")
DEV_END = datetime(2026, 10, 1, tzinfo=UTC)
HOLD_END = datetime(2026, 11, 23, tzinfo=UTC)
NOV_START = datetime(2026, 11, 1, tzinfo=UTC)
SEED = 20261123
PRE_MIN = 30                 # minutes of book fetched before the occurrence stamp
MIN_AFTER_MIN = 2.0          # occurrence must be at least this many minutes before close
MAX_WINDOW_H = 72            # longest post-occurrence window fetched (YouTube-views class ~40-60 h)
MAX_EVENTS_PER_CELL = 60     # seeded cap per (category, split) on sampled short-window events
L_EVENTS_PER_SPLIT = 40      # seeded cap per split on YouTube-views events
TAKER = 0.07
MAKER = 0.0175

BASE_COLS = ["ticker", "event_ticker", "series_ticker", "category", "title", "market_type", "open_time",
             "close_time", "occurrence_time", "expected_expiration_time", "settlement_ts", "result",
             "can_close_early", "fee_type", "fee_multiplier", "volume", "source"]

TRADE_SCHEMA = {"trade_id": pl.String, "ticker": pl.String, "ts": pl.Datetime("us", "UTC"), "count": pl.Float64,
                "yes_price": pl.Float64, "taker_side": pl.String}
CANDLE_SCHEMA = {"ticker": pl.String, "ts": pl.Datetime("us", "UTC"), "bid": pl.Float64, "ask": pl.Float64,
                 "p_close": pl.Float64, "volume": pl.Float64}


# ---- client --------------------------------------------------------------------------------------
class ReuseClient:
    """A ``CachedClient`` writing under ``CACHE/raw`` that first looks for the identical request in
    caches earlier studies wrote (read-only), so nothing is fetched twice. Network requests start at
    least ``min_interval`` s apart across threads (the inner client's own throttle is not thread-safe)."""

    def __init__(self, inner, extra_dirs: list[Path], min_interval: float = 0.5) -> None:
        import threading

        self.inner = inner
        self.extra = [d for d in extra_dirs if d.is_dir() and d != inner.raw_dir]
        self.network = 0
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def get(self, path: str, params: dict[str, Any] | None = None, *, cache: bool = True, tag: str | None = None):
        from albert.http import RAW_SUFFIX, Fetched, read_raw

        p = {k: v for k, v in (params or {}).items() if v is not None}
        url = f"{self.inner.base_url}{path}"
        if cache:
            key = self.inner.cache_key(url, p if tag is None else {**p, "__tag": tag})
            for d in [self.inner.raw_dir, *self.extra]:
                for f in (d / f"{key}{RAW_SUFFIX}", d / f"{key}.json"):
                    if f.exists():
                        rec = read_raw(f)
                        return Fetched(url, p, datetime.fromisoformat(rec["fetched_at"]), f, rec["response"], True)
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.network += 1
        return self.inner.get(path, params, cache=cache, tag=tag)


def api(min_interval: float = 0.5):
    import random

    from albert.config import settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    s = settings()

    def nap(x: float) -> None:
        time.sleep(min(x, 30.0) * (0.75 + 0.5 * random.random()))
    inner = CachedClient("kalshi", s.kalshi_base, CACHE / "raw", min_interval=0.0, max_retries=12, sleep=nap)
    extra = [Path("data/raw/kalshi"), *[Path(p) for p in sorted(glob.glob("data/cache/*/raw/kalshi"))]]
    return k.Kalshi(ReuseClient(inner, extra, max(min_interval, 0.5)))


# ---- parsing (tested on recorded payloads) --------------------------------------------------------
def trade_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows of a ``/markets/trades`` or ``/historical/trades`` page (``*_dollars``/``*_fp`` or plain)."""
    from albert.venues.kalshi import _field, num, parse_ts

    out = []
    for t in payload.get("trades") or []:
        yp = num(_field(t, "yes_price"))
        if yp is not None and yp > 1.0:          # old integer-cent payloads
            yp = yp / 100.0
        out.append({"trade_id": t.get("trade_id"), "ticker": t.get("ticker"), "ts": parse_ts(t.get("created_time")),
                    "count": num(_field(t, "count")), "yes_price": yp, "taker_side": t.get("taker_side")})
    return out


def candle_rows(payload: dict[str, Any], ticker: str) -> list[dict[str, Any]]:
    """Minute candles: close of the YES bid / ask (Kalshi reports a missing bid as 0, ask as 1)."""
    from albert.venues.kalshi import _field, num, parse_ts

    out = []
    for c in payload.get("candlesticks") or []:
        b, a, p = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
        out.append({"ticker": ticker, "ts": parse_ts(c.get("end_period_ts")),
                    "bid": num(_field(b, "close")), "ask": num(_field(a, "close")),
                    "p_close": num(_field(p, "close")), "volume": num(_field(c, "volume"))})
    return out


def fee(n: float, price: float, *, maker: bool = False, fee_type: str | None = "quadratic",
        mult: float | None = 1.0) -> float:
    """Kalshi fee per order, rounded up to the cent (``albert.fees.kalshi_fee``)."""
    from albert.fees import kalshi_fee

    return kalshi_fee(n, price, maker=maker, fee_type=fee_type or "quadratic", multiplier=mult)


def split_of(ts: pl.Expr) -> pl.Expr:
    return pl.when(ts < DEV_END).then(pl.lit("dev")).when(ts < HOLD_END).then(pl.lit("holdout")).otherwise(pl.lit("late"))


# ---- base frame ----------------------------------------------------------------------------------
def base() -> None:
    """Non-sports binary settled markets (lean columns) and a sports lag summary, from the shared table."""
    from albert.categories import kalshi_series_is_financial

    OUT.mkdir(parents=True, exist_ok=True)
    lf = pl.scan_parquet(TABLES / "kalshi_markets.parquet")
    ns = (lf.filter((pl.col("category") != "Sports") & pl.col("result").is_in(["yes", "no"])
                    & (pl.col("market_type") == "binary")).select(BASE_COLS).collect())
    ser = pl.read_parquet(TABLES / "kalshi_series.parquet").select("series_ticker", "title").rename({"title": "series_title"})
    ns = ns.join(ser, on="series_ticker", how="left")
    fin = [kalshi_series_is_financial(s, c, t) for s, c, t in ns.select("series_ticker", "category", "series_title").iter_rows()]
    ns = ns.filter(~pl.Series(fin)).drop("series_title")
    ns.write_parquet(OUT / "markets_table.parquet")
    sp = (lf.filter((pl.col("category") == "Sports") & pl.col("result").is_in(["yes", "no"]))
          .select("series_ticker", "close_time", "settlement_ts", "expected_expiration_time", "occurrence_time")
          .with_columns(((pl.col("settlement_ts") - pl.col("close_time")).dt.total_seconds() / 3600).alias("lag_h"))
          .group_by("series_ticker").agg(pl.len().alias("markets"), pl.col("lag_h").median().alias("lag_p50_h"),
                                         pl.col("lag_h").quantile(0.9).alias("lag_p90_h"),
                                         (pl.col("lag_h") > 24).mean().alias("share_lag_gt_24h"))
          .collect())
    sp.write_parquet(OUT / "sports_lag_by_series.parquet")
    print(json.dumps({"non_sports_markets": ns.height, "series": ns["series_ticker"].n_unique(),
                      "sports_series": sp.height}, indent=1))


def fetch_nov(min_interval: float = 0.5) -> None:
    """Settled markets closing 2026-11-01..11-23 for every non-sports series seen in the table."""
    from albert.venues.kalshi import MARKET_SCHEMA, market_rows

    ns = pl.read_parquet(OUT / "markets_table.parquet")
    series = sorted(ns["series_ticker"].unique().to_list())
    info = {r["series_ticker"]: r for r in pl.read_parquet(TABLES / "kalshi_series.parquet").to_dicts()}
    k = api(min_interval)
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, s in enumerate(series):
        try:
            for page in k.series_markets(s, "settled", min_close=NOV_START, max_close=HOLD_END, cache=True):
                rows.extend(market_rows(page.payload, info, "live"))
        except RuntimeError as error:
            print(f"skip {s}: {str(error)[:160]}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(series)} series, {len(rows)} markets, {k.client.network} requests, "
                  f"{time.time() - t0:.0f}s", flush=True)
    f = pl.DataFrame(rows, schema=MARKET_SCHEMA)
    f = f.filter(pl.col("result").is_in(["yes", "no"]) & (pl.col("market_type") == "binary")).select(BASE_COLS)
    f.write_parquet(OUT / "markets_nov.parquet")
    print(json.dumps({"series": len(series), "nov_markets": f.height, "requests": k.client.network}))


def markets() -> pl.DataFrame:
    """Table (Aug 15 - Oct 31 closes) plus November closes, one row per market."""
    a = pl.read_parquet(OUT / "markets_table.parquet")
    p = OUT / "markets_nov.parquet"
    if p.exists():
        cat = a.group_by("series_ticker").agg(pl.col("category").first().alias("cat_t"))
        b = (pl.read_parquet(p).filter(pl.col("close_time") >= datetime(2026, 11, 1, tzinfo=UTC))
             .join(cat, on="series_ticker", how="left")
             .with_columns(pl.coalesce("category", "cat_t").alias("category")).drop("cat_t"))
        a = pl.concat([a, b.select(a.columns)], how="vertical_relaxed").unique("ticker", keep="first")
    return a.with_columns(split_of(pl.col("close_time")).alias("split"))


def with_windows(m: pl.DataFrame) -> pl.DataFrame:
    """Time fields used throughout: settlement lag, early close, the post-occurrence window."""
    return m.with_columns(
        ((pl.col("settlement_ts") - pl.col("close_time")).dt.total_seconds() / 3600).alias("lag_h"),
        ((pl.col("expected_expiration_time") - pl.col("close_time")).dt.total_seconds() / 3600).alias("early_h"),
        ((pl.col("close_time") - pl.col("occurrence_time")).dt.total_seconds() / 60).alias("win_min"),
        ((pl.col("occurrence_time") > pl.col("open_time") + timedelta(minutes=10))
         & (pl.col("occurrence_time") <= pl.col("close_time") - timedelta(minutes=MIN_AFTER_MIN))).fill_null(False)
        .alias("occ_inside"))


def _q(c: str, q: float, name: str) -> pl.Expr:
    return pl.col(c).quantile(q).round(2).alias(name)


def census() -> None:
    """Settlement lag, early closes and post-occurrence windows by category and series (no prices)."""
    m = with_windows(markets())
    agg = [pl.len().alias("markets"), pl.col("event_ticker").n_unique().alias("events"),
           _q("lag_h", 0.5, "lag_p50_h"), _q("lag_h", 0.9, "lag_p90_h"), _q("lag_h", 0.99, "lag_p99_h"),
           (pl.col("lag_h") > 24).mean().round(4).alias("share_lag_gt_24h"),
           (pl.col("lag_h") > 72).mean().round(4).alias("share_lag_gt_72h"),
           (pl.col("early_h") > 1).mean().round(3).alias("share_closed_early_1h"),
           pl.col("occ_inside").mean().round(3).alias("share_occ_inside"),
           pl.col("win_min").filter(pl.col("occ_inside")).median().round(1).alias("occ_window_p50_min"),
           pl.col("win_min").filter(pl.col("occ_inside")).quantile(0.9).round(1).alias("occ_window_p90_min")]
    by_cat = m.group_by("category", "split").agg(agg).sort("category", "split")
    by_ser = (m.group_by("series_ticker", "category").agg(agg).sort("events", descending=True))
    by_cat.write_csv(OUT / "census_category.csv")
    by_ser.write_csv(OUT / "census_series.csv")
    # sibling closes: does one leg of an event close minutes before the others (a public, real-time
    # "the event has happened" signal for the legs still open)?
    early = m.filter(pl.col("early_h") > 1)
    sib = (early.group_by("event_ticker", "category", "split").agg(
        pl.len().alias("legs"), ((pl.col("close_time").max() - pl.col("close_time").min()).dt.total_seconds() / 60)
        .alias("close_spread_min")).filter(pl.col("legs") >= 2))
    sib_t = sib.group_by("category", "split").agg(
        pl.len().alias("events_ge2_early_legs"), _q("close_spread_min", 0.5, "close_spread_p50_min"),
        _q("close_spread_min", 0.9, "close_spread_p90_min"),
        (pl.col("close_spread_min") >= 2).mean().round(3).alias("share_spread_ge_2min")).sort("category", "split")
    sib_t.write_csv(OUT / "census_sibling_close.csv")
    print(sib_t)
    sp = pl.read_parquet(OUT / "sports_lag_by_series.parquet")
    w = sp["markets"].cast(pl.Float64)
    sports = {"series": sp.height, "markets": int(w.sum()),
              "lag_p50_h_weighted": round(float((sp["lag_p50_h"] * w).sum() / w.sum()), 3),
              "share_series_p90_gt_24h": round(float((sp["lag_p90_h"] > 24).mean()), 4)}
    (OUT / "census_sports.json").write_text(json.dumps(sports, indent=1))
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_rows(60)
    print(by_cat)
    print(by_ser.head(40))
    print(sports)


# ---- sample of post-occurrence windows -----------------------------------------------------------
def sample() -> pl.DataFrame:
    """Two samples, events drawn seeded per cell, never on volume, price or result.

    * ``short``: every market whose occurrence stamp lies inside (open + 10 min, close - 2 min] with
      close - occurrence <= 6 h (speech end, race call, award, storm upgrade), all such legs of up to
      ``MAX_EVENTS_PER_CELL`` events per (category, split). Fetch window [occurrence - 30 min, close].
    * ``yt``: YouTube-views markets (``L_SERIES``), the measured day already over at the anchor (ticker
      date + 38 h) and the market still open then; up to ``L_EVENTS_PER_SPLIT`` events per split, every
      leg. Fetch window [anchor - 2 h, close], hourly candles."""
    m = with_windows(markets())
    short = (m.filter(pl.col("occ_inside") & (pl.col("win_min") <= 360) & ~pl.col("series_ticker").is_in(list(L_SERIES)))
             .with_columns(pl.lit("short").alias("wclass"), (pl.col("occurrence_time") - timedelta(minutes=PRE_MIN)).alias("lo"),
                           pl.lit(1).alias("period")))
    yt = (m.filter(pl.col("series_ticker").is_in(list(L_SERIES)))
          .with_columns(l_anchor(pl.col("ticker")).alias("anchor"))
          .filter(pl.col("anchor").is_not_null() & (pl.col("close_time") > pl.col("anchor")))
          .with_columns(pl.lit("yt").alias("wclass"), (pl.col("anchor") - timedelta(hours=2)).alias("lo"),
                        pl.lit(60).alias("period")).drop("anchor"))
    keep = []
    for frame, cap, keys in ((short, MAX_EVENTS_PER_CELL, ["category", "split"]), (yt, L_EVENTS_PER_SPLIT, ["split"])):
        ev = frame.select("event_ticker", *keys).unique()
        for i, (_, g) in enumerate(sorted(ev.group_by(keys), key=lambda x: str(x[0]))):
            e = g["event_ticker"].sort()
            keep.extend(e.sample(min(cap, e.len()), seed=SEED + i).to_list())
    s = (pl.concat([short, yt], how="vertical_relaxed").filter(pl.col("event_ticker").is_in(keep))
         .filter(pl.col("split") != "late")
         .with_columns((pl.col("close_time") + timedelta(minutes=1)).alias("hi")).sort("event_ticker", "ticker"))
    s.write_parquet(OUT / "sample.parquet")
    print(s.group_by("category", "wclass", "split").agg(pl.len().alias("markets"), pl.col("event_ticker").n_unique().alias("events"),
                                                         pl.col("win_min").median().round(1).alias("win_p50_min"))
          .sort("category", "wclass", "split"))
    return s


# ---- fetch trades and candles over each window ---------------------------------------------------
def _chunks(rows: list[dict[str, Any]], period: int, cap: int = 9000, max_n: int = 50,
            max_periods: int = 4000) -> list[list[dict[str, Any]]]:
    """Group legs (sorted by window start) so one batch request spans <= ``max_periods`` periods and
    <= ``cap`` candles summed over its tickers (API limits 5,000 and 10,000)."""
    out: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: x["lo"]):
        nxt = [*cur, r]
        lo, hi = min(x["lo"] for x in nxt), max(x["hi"] for x in nxt)
        per = int((hi - lo).total_seconds() // 60 // period) + 2
        if cur and (per > max_periods or per * len(nxt) > cap or len(nxt) > max_n):
            out.append(cur)
            cur = [r]
        else:
            cur = nxt
    if cur:
        out.append(cur)
    return out


def _fetch_event(k, rows: list[dict[str, Any]], cutoff: datetime | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Trades and candles of one event's sampled legs over their windows."""
    period = int(rows[0]["period"])
    for r in rows:
        r["hist"] = cutoff is not None and r["settlement_ts"] < cutoff
    tr: list[dict[str, Any]] = []
    cd: list[dict[str, Any]] = []
    for r in rows:
        for page in k.trades(r["ticker"], r["hist"], min_ts=r["lo"], max_ts=r["hi"]):
            tr.extend(trade_rows(page.payload))
    live = [r for r in rows if not r["hist"]]
    for ch in _chunks(live, period):
        lo, hi = min(r["lo"] for r in ch), max(r["hi"] for r in ch)
        hi = min(hi, lo + timedelta(minutes=period * 4800))
        if len(ch) == 1:
            page = k.candles(ch[0]["ticker"], lo, hi, period, False)
            cd.extend(candle_rows(page.payload, ch[0]["ticker"]))
        else:
            page = k.candles_batch([r["ticker"] for r in ch], lo, hi, period)
            for mk in page.payload.get("markets") or []:
                cd.extend(candle_rows(mk, mk["market_ticker"]))
    for r in (r for r in rows if r["hist"]):
        try:
            page = k.candles(r["ticker"], r["lo"], r["hi"], period, True)
        except RuntimeError as error:
            if "404" not in str(error):
                raise
            page = k.candles(r["ticker"], r["lo"], r["hi"], period, False)
        cd.extend(candle_rows(page.payload, r["ticker"]))
    return tr, cd


def fetch_windows(min_interval: float = 0.5, workers: int = 2) -> None:
    """Trades in [occurrence - 30 min, close + 1 min] per market and candles over the same window
    (1-min for short windows, 60-min for long ones). Appends parquet parts; resumable. Two threads
    share one client whose requests start >= 0.5 s apart."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    s = pl.read_parquet(OUT / "sample.parquet")
    d = CACHE / "windows"
    d.mkdir(parents=True, exist_ok=True)
    done_p = d / "done.parquet"
    done = set(pl.read_parquet(done_p)["event_ticker"]) if done_p.exists() else set()
    k = api(min_interval)
    cutoff = k.cutoff()["market_settled_ts"]
    events = [e for e in s["event_ticker"].unique().sort().to_list() if e not in done]
    groups = {e: g.to_dicts() for (e,), g in s.filter(pl.col("event_ticker").is_in(events)).group_by("event_ticker")}
    print(f"{len(events)} events to fetch ({len(done)} done), cutoff {cutoff}", flush=True)
    lock = threading.Lock()
    tbuf: list[dict[str, Any]] = []
    cbuf: list[dict[str, Any]] = []
    fin: list[str] = []
    part = len(list(d.glob("trades-*.parquet")))
    t0 = time.time()

    def flush() -> None:
        nonlocal tbuf, cbuf, part
        pl.DataFrame(tbuf, schema=TRADE_SCHEMA).write_parquet(d / f"trades-{part:05d}.parquet")
        pl.DataFrame(cbuf, schema=CANDLE_SCHEMA).write_parquet(d / f"candles-{part:05d}.parquet")
        part += 1
        tbuf, cbuf = [], []
        pl.DataFrame({"event_ticker": sorted(done | set(fin))}).write_parquet(done_p)

    def job(e: str):
        try:
            return e, _fetch_event(k, groups[e], cutoff)
        except RuntimeError as error:
            print(f"skip {e}: {str(error)[:200]}", flush=True)
            return e, None

    n = 0
    with ThreadPoolExecutor(max_workers=min(2, workers)) as pool:
        for e, got in pool.map(job, events):
            with lock:
                n += 1
                if got is not None:
                    tbuf.extend(got[0])
                    cbuf.extend(got[1])
                    fin.append(e)
                if n % 40 == 0:
                    flush()
                    print(f"  {n}/{len(events)} events, {k.client.network} requests, {time.time() - t0:.0f}s", flush=True)
    flush()
    print(f"done: {len(fin)} events, {k.client.network} requests", flush=True)


def load_windows() -> tuple[pl.DataFrame, pl.DataFrame]:
    d = CACHE / "windows"
    t = pl.concat([pl.read_parquet(p) for p in sorted(d.glob("trades-*.parquet"))], how="vertical_relaxed")
    c = pl.concat([pl.read_parquet(p) for p in sorted(d.glob("candles-*.parquet"))], how="vertical_relaxed")
    return t.unique("trade_id").sort("ticker", "ts"), c.unique(["ticker", "ts"]).sort("ticker", "ts")


# ---- H1: what a perfect-knowledge trader could take after the occurrence stamp -------------------
DELTAS_MIN = (0.0, 1.0, 5.0)
MAKER_BIDS = (0.90, 0.95, 0.97, 0.98, 0.99)
MAKER_CAP = 10            # contracts per market for the resting-bid upper bound
TAKER_N = 10              # contracts per taker order in the ask-based bound


def h1_prints(s: pl.DataFrame, t: pl.DataFrame) -> pl.DataFrame:
    """Prints in [occurrence - 30 min, close + 1 min] with the winner's price and who took it."""
    meta = s.select("ticker", "event_ticker", "series_ticker", "category", "wclass", "split", "occurrence_time",
                    "close_time", (pl.col("result") == "yes").alias("y"), "fee_type", "fee_multiplier")
    return (t.join(meta, on="ticker", how="inner")
            .filter((pl.col("ts") >= pl.col("occurrence_time") - timedelta(minutes=PRE_MIN))
                    & (pl.col("ts") <= pl.col("close_time") + timedelta(minutes=1)))
            .with_columns(((pl.col("ts") - pl.col("occurrence_time")).dt.total_seconds() / 60).alias("rel_min"),
                          pl.when(pl.col("y")).then(pl.col("yes_price")).otherwise(1 - pl.col("yes_price")).alias("wp"),
                          ((pl.col("taker_side") == "yes") == pl.col("y")).alias("taker_bought_winner")))


def _fee_col(df: pl.DataFrame, n: str, p: str, maker: bool) -> list[float]:
    return [fee(a, b, maker=maker, fee_type=ft, mult=mu)
            for a, b, ft, mu in df.select(n, p, "fee_type", "fee_multiplier").iter_rows()]


def h1_tables(s: pl.DataFrame, t: pl.DataFrame, c: pl.DataFrame) -> dict[str, pl.DataFrame]:
    p = h1_prints(s, t)
    cell = ["category", "wclass", "split"]
    weeks = (s.group_by(cell).agg(((pl.col("close_time").max() - pl.col("close_time").min()).dt.total_days() / 7)
                                  .clip(1, None).alias("weeks")))
    base = s.group_by(cell).agg(pl.len().alias("markets"), pl.col("event_ticker").n_unique().alias("events"),
                                pl.col("win_min").median().round(1).alias("window_p50_min"))
    post = p.filter(pl.col("rel_min") >= 1.0)
    pre = p.filter(pl.col("rel_min") < 0)
    flow = post.group_by(cell).agg(
        pl.col("ticker").n_unique().alias("markets_with_post_prints"),
        pl.col("count").sum().round(0).alias("post_contracts"),
        (pl.col("count") * (pl.col("wp") <= 0.97)).sum().round(0).alias("post_contracts_winner_le_97c"),
        (pl.col("count") * (pl.col("wp") <= 0.90)).sum().round(0).alias("post_contracts_winner_le_90c"),
        (pl.col("count") * pl.col("taker_bought_winner")).sum().round(0).alias("post_taker_bought_winner"),
        ((pl.col("count") * pl.col("wp")).sum() / pl.col("count").sum()).round(4).alias("post_vwap_winner"))
    preflow = pre.group_by(cell).agg(((pl.col("count") * pl.col("wp")).sum() / pl.col("count").sum()).round(4)
                                     .alias("pre30_vwap_winner"), pl.col("count").sum().round(0).alias("pre30_contracts"))
    # perfect-knowledge taker pool: every winner offer someone lifted after occurrence + delta
    pools = []
    for dlt in DELTAS_MIN:
        x = p.filter((pl.col("rel_min") >= dlt) & pl.col("taker_bought_winner") & (pl.col("wp") <= 0.99))
        x = x.with_columns(pl.Series("fee", _fee_col(x, "count", "wp", False), dtype=pl.Float64))
        pools.append(x.group_by(cell).agg(((pl.col("count") * (1 - pl.col("wp"))) - pl.col("fee")).sum().round(2)
                                          .alias(f"taker_pool_usd_d{int(dlt)}")))
    # resting winner bid at b from occurrence + 1 min, filled by prints strictly through it (cap per market)
    mk = []
    for b in MAKER_BIDS:
        x = (post.filter(~pl.col("taker_bought_winner") & (pl.col("wp") < b - 1e-9))
             .group_by("ticker", *cell, "fee_type", "fee_multiplier").agg(pl.col("count").sum().alias("q"))
             .with_columns(pl.col("q").clip(None, MAKER_CAP).alias("n"), pl.lit(b).alias("b")))
        x = x.with_columns(pl.Series("fee", _fee_col(x, "n", "b", True), dtype=pl.Float64))
        mk.append(x.group_by(cell).agg(pl.len().alias(f"maker{int(b * 100)}_markets"),
                                       (pl.col("n") * (1 - b) - pl.col("fee")).sum().round(2).alias(f"maker{int(b * 100)}_usd")))
    # ask-based taker at occurrence + delta: winner's ask on the last candle ended at or before then
    asks = []
    cc = c.join(s.select("ticker", *cell, "occurrence_time", (pl.col("result") == "yes").alias("y"),
                         "fee_type", "fee_multiplier"), on="ticker")
    for dlt in (1.0, 5.0):
        d = (cc.filter(pl.col("ts") <= pl.col("occurrence_time") + timedelta(minutes=dlt))
             .sort("ts").group_by("ticker", maintain_order=True).last()
             .with_columns(pl.when(pl.col("y")).then(pl.col("ask")).otherwise(1 - pl.col("bid")).alias("ask_w"))
             .filter(pl.col("ask_w").is_not_null() & (pl.col("ask_w") < 0.995)))
        d = d.with_columns(pl.Series("fee", [fee(TAKER_N, a, fee_type=ft, mult=mu) for a, ft, mu in
                                             d.select("ask_w", "fee_type", "fee_multiplier").iter_rows()], dtype=pl.Float64))
        asks.append(d.group_by(cell).agg(pl.len().alias(f"ask_offer_markets_d{int(dlt)}"),
                                         pl.col("ask_w").median().round(3).alias(f"ask_w_p50_d{int(dlt)}"),
                                         ((TAKER_N * (1 - pl.col("ask_w")) - pl.col("fee")) / TAKER_N * 100).mean().round(2)
                                         .alias(f"ask_taker_c_per_contract_d{int(dlt)}")))
    out = base.join(weeks, on=cell, how="left").join(flow, on=cell, how="left").join(preflow, on=cell, how="left")
    for f in [*pools, *mk, *asks]:
        out = out.join(f, on=cell, how="left")
    out = out.sort(cell)
    # concentration: share of the d1 taker pool from the top-5 markets
    return {"h1_cells": out, "h1_prints": p}


# ---- H3 / H5: near-certain legs, tail risk and settlement lag (xcat whole-life hourly candles) -----
XCAT = Path("data/research/xcat")


def xcat_hourly_meta() -> pl.DataFrame:
    return (pl.read_parquet(XCAT / "sample.parquet").filter(pl.col("layer") == "hourly")
            .select("ticker", "event_ticker", "series_ticker", "category", "open_time", "close_time", "settlement_ts",
                    "result", "fee_type", "fee_multiplier").unique("ticker"))


def series_lag_class(m: pl.DataFrame) -> pl.DataFrame:
    """Median close-to-settle lag per series from dev closes only (a structural property of how Kalshi
    settles the series, known before the holdout), bucketed."""
    d = with_windows(m.filter(pl.col("close_time") < DEV_END))
    g = d.group_by("series_ticker").agg(pl.col("lag_h").median().alias("series_lag_p50_h"))
    return g.with_columns(pl.when(pl.col("series_lag_p50_h") <= 1).then(pl.lit("lag<=1h"))
                          .when(pl.col("series_lag_p50_h") <= 6).then(pl.lit("lag1-6h"))
                          .otherwise(pl.lit("lag>6h")).alias("lag_class"))


def near_certain_entries(lo: float = 0.95, hi: float = 0.99, stake: float = 5.0, max_spread: float = 0.05) -> pl.DataFrame:
    """First hourly candle (ended before close) where the leading side's ask is in [lo, hi] with a real
    two-sided book no wider than ``max_spread``; buy floor(stake / ask) of it at the ask (taker fee per
    order), hold to settlement."""
    meta = xcat_hourly_meta()
    parts = sorted((XCAT / "hourly").glob("part-*.parquet"))
    c = (pl.scan_parquet(parts).select("ticker", "end_ts", "bid_close", "ask_close")
         .join(meta.lazy().select("ticker", "open_time", "close_time"), on="ticker")
         .filter((pl.col("end_ts") < pl.col("close_time")) & (pl.col("end_ts") > pl.col("open_time"))
                 & (pl.col("bid_close") > 0) & (pl.col("ask_close") < 1) & (pl.col("ask_close") > pl.col("bid_close"))
                 & (pl.col("ask_close") - pl.col("bid_close") <= max_spread + 1e-9))
         .with_columns(((pl.col("bid_close") + pl.col("ask_close")) / 2).alias("mid"))
         .with_columns(pl.when(pl.col("mid") >= 0.5).then(pl.lit("yes")).otherwise(pl.lit("no")).alias("side"),
                       pl.when(pl.col("mid") >= 0.5).then(pl.col("ask_close")).otherwise(1 - pl.col("bid_close")).alias("ask_l"))
         .filter((pl.col("ask_l") >= lo - 1e-9) & (pl.col("ask_l") <= hi + 1e-9))
         .sort("ticker", "end_ts").group_by("ticker").first().collect())
    e = c.join(meta.drop("open_time", "close_time"), on="ticker", how="left").with_columns(
        (pl.col("result") == pl.col("side")).cast(pl.Float64).alias("win"),
        (stake / pl.col("ask_l")).floor().clip(1, None).alias("n"),
        ((pl.col("settlement_ts") - pl.col("end_ts")).dt.total_seconds() / 3600).alias("hold_h"),
        ((pl.col("close_time") - pl.col("end_ts")).dt.total_seconds() / 3600).alias("to_close_h"),
        split_of(pl.col("end_ts")).alias("split"))
    e = e.with_columns(pl.Series("fee", [fee(n, a, fee_type=ft, mult=mu) for n, a, ft, mu in
                                         e.select("n", "ask_l", "fee_type", "fee_multiplier").iter_rows()], dtype=pl.Float64))
    return e.with_columns((pl.col("n") * pl.col("ask_l") + pl.col("fee")).alias("cost_usd"),
                          (pl.col("n") * pl.col("win")).alias("payoff_usd"))


def cell_stats(df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    from albert.research.fundamental import clustered_mean

    rows = []
    for k, g in df.group_by(keys, maintain_order=True):
        pnl = (g["payoff_usd"] - g["cost_usd"]).to_numpy()
        per_c = (pnl / g["n"].to_numpy()) * 100
        mu, t, ev = clustered_mean(per_c, g["event_ticker"].to_numpy())
        rows.append({**dict(zip(keys, k, strict=True)), "legs": g.height, "events": ev,
                     "win_rate": round(float(g["win"].mean()), 4),
                     "breakeven": round(float((g["cost_usd"] / g["n"]).mean()), 4),
                     "c_per_contract": round(mu, 2), "t_event": None if t is None else round(t, 2),
                     "pnl_per_usd": round(float(pnl.sum() / g["cost_usd"].sum()), 4),
                     "hold_h_p50": round(float(g["hold_h"].median()), 1) if "hold_h" in g.columns else None})
    return pl.DataFrame(rows).sort(keys)


# ---- family L: measured-but-unpublished windows (YouTube views), anchor known in advance ----------
L_SERIES = ("KXYTVIEWSD", "KXYTVIEWSW")
L_DELAYS_H = (0, 6, 12)              # hours after the anchor (measurement day end + slack)
L_TAKER_BANDS = ((0.80, 0.90), (0.90, 0.95), (0.95, 0.98))
L_MAKER_MARGINS = (0.02, 0.05)       # sell YES at mid + m on dead-looking legs (YES mid <= 10c)
STAKE = 5.0


def l_anchor(ticker: pl.Expr) -> pl.Expr:
    """End of the measured day + slack, from the ticker date alone (``KXYTVIEWSD-TAY26SEP19-15.9M`` ->
    2026-09-20 14:00 UTC, i.e. the day's 14:00 UTC stamp + 24 h: after midnight in every US zone)."""
    d = ticker.str.extract(r"-[A-Z]+(\d{2}[A-Z]{3}\d{2})-", 1)
    return (d.str.to_datetime("%y%b%d", strict=False).dt.replace_time_zone("UTC") + timedelta(hours=38))


def l_rules(s: pl.DataFrame, t: pl.DataFrame, c: pl.DataFrame) -> pl.DataFrame:
    """Every family-L cell on every eligible market (one row per market x cell that trades)."""
    meta = (s.filter(pl.col("series_ticker").is_in(list(L_SERIES)))
            .with_columns(l_anchor(pl.col("ticker")).alias("anchor"), (pl.col("result") == "yes").alias("y"))
            .filter(pl.col("anchor").is_not_null()))
    cc = c.join(meta.select("ticker", "anchor", "close_time"), on="ticker").filter(pl.col("ts") < pl.col("close_time"))
    rows = []
    for dly in L_DELAYS_H:
        q = (cc.filter(pl.col("ts") <= pl.col("anchor") + timedelta(hours=dly)).sort("ts")
             .group_by("ticker", maintain_order=True).last().rename({"ts": "q_ts"}))
        x = meta.join(q.select("ticker", "q_ts", "bid", "ask", "p_close"), on="ticker", how="inner").with_columns(
            (pl.col("anchor") + timedelta(hours=dly)).alias("t_dec"))
        x = x.filter(pl.col("t_dec") < pl.col("close_time"))
        for r in x.iter_rows(named=True):
            bid, ask = r["bid"], r["ask"]
            two = bid is not None and ask is not None and 0 < bid < ask < 1
            base_r = {"ticker": r["ticker"], "event_ticker": r["event_ticker"], "series_ticker": r["series_ticker"],
                      "split": r["split"], "delay_h": dly, "t_in": r["t_dec"], "settle_ts": r["settlement_ts"],
                      "y": r["y"], "fee_type": r["fee_type"], "mult": r["fee_multiplier"]}
            if two:
                mid = (bid + ask) / 2
                side = "yes" if mid >= 0.5 else "no"
                px = ask if side == "yes" else 1 - bid
                for lo, hi in L_TAKER_BANDS:
                    if lo <= px < hi:
                        n = max(1.0, math.floor(STAKE / px))
                        f = fee(n, px, fee_type=r["fee_type"], mult=r["fee_multiplier"])
                        win = (side == "yes") == r["y"]
                        rows.append({**base_r, "rule": f"L_taker_{int(lo * 100)}_{int(hi * 100)}", "side": side,
                                     "n": n, "price": px, "cost_usd": n * px + f, "payoff_usd": n * float(win)})
            # maker: sell YES on a leg whose YES mid (or last price without a book) is <= 10c
            ref = (bid + ask) / 2 if two else (r["p_close"] if r["p_close"] is not None else None)
            if ref is None or ref > 0.10:
                continue
            for mgn in L_MAKER_MARGINS:
                ask_yes = min(0.99, math.ceil((ref + mgn) * 100 - 1e-9) / 100)
                tt = t.filter((pl.col("ticker") == r["ticker"]) & (pl.col("ts") > r["t_dec"])
                              & (pl.col("ts") < r["close_time"]) & (pl.col("taker_side") == "yes")
                              & (pl.col("yes_price") > ask_yes + 1e-9)).sort("ts")
                if tt.height == 0:
                    continue
                n = 1.0
                nop = 1 - ask_yes
                f = fee(n, nop, maker=True, fee_type=r["fee_type"], mult=r["fee_multiplier"])
                rows.append({**base_r, "rule": f"L_makerNO_m{int(mgn * 100)}", "side": "no", "n": n, "price": nop,
                             "t_in": tt["ts"][0], "cost_usd": n * nop + f, "payoff_usd": n * float(not r["y"])})
    return pl.DataFrame(rows)


# ---- family P: a sibling leg already closed (public, real-time "the event happened" signal) -------
P_TAKER_BANDS = ((0.80, 0.90), (0.90, 0.95), (0.95, 0.99))
P_MAKER_BIDS = (0.95, 0.97)
P_LAG_S = 60            # seconds after the first sibling's close before acting (poll + order latency)


def p_rules(s: pl.DataFrame, t: pl.DataFrame, c: pl.DataFrame, allm: pl.DataFrame) -> pl.DataFrame:
    """Legs still open >= 2 min after the first leg of their event closed early. Decision at that close
    + 60 s on the last minute candle ended by then: buy the leading side at its ask (bands), or rest a
    bid for it at 95/97c filled by a later print strictly through (``paper_trader.through``)."""
    from albert.paper_trader import through

    first = (allm.filter(pl.col("event_ticker").is_in(s["event_ticker"].unique().implode()))
             .group_by("event_ticker").agg(pl.col("close_time").min().alias("t1")))
    legs = (s.filter(pl.col("wclass") == "short").join(first, on="event_ticker")
            .filter(pl.col("close_time") >= pl.col("t1") + timedelta(minutes=2))
            .with_columns((pl.col("t1") + timedelta(seconds=P_LAG_S)).alias("t_dec")))
    if legs.height == 0:
        return pl.DataFrame()
    q = (c.join(legs.select("ticker", "t_dec"), on="ticker").filter(pl.col("ts") <= pl.col("t_dec"))
         .sort("ts").group_by("ticker", maintain_order=True).last())
    x = legs.join(q.select("ticker", "bid", "ask"), on="ticker", how="inner")
    rows = []
    for r in x.iter_rows(named=True):
        bid, ask = r["bid"], r["ask"]
        if bid is None or ask is None or not (0 < bid < ask < 1):
            continue
        y = r["result"] == "yes"
        side = "yes" if (bid + ask) / 2 >= 0.5 else "no"
        px = ask if side == "yes" else 1 - bid
        win = (side == "yes") == y
        b0 = {"ticker": r["ticker"], "event_ticker": r["event_ticker"], "category": r["category"], "split": r["split"],
              "t_in": r["t_dec"], "settle_ts": r["settlement_ts"], "side": side}
        for lo, hi in P_TAKER_BANDS:
            if lo <= px < hi:
                n = max(1.0, math.floor(STAKE / px))
                f = fee(n, px, fee_type=r["fee_type"], mult=r["fee_multiplier"])
                rows.append({**b0, "rule": f"P_taker_{int(lo * 100)}_{int(hi * 100)}", "n": n, "price": px,
                             "cost_usd": n * px + f, "payoff_usd": n * float(win)})
        tt = t.filter((pl.col("ticker") == r["ticker"]) & (pl.col("ts") > r["t_dec"]) & (pl.col("ts") <= r["close_time"]))
        for b in P_MAKER_BIDS:
            if px <= b:          # the offer is already at or below our bid: a taker would just lift it
                continue
            hit = [ts for ts, yp, ts_side in tt.select("ts", "yes_price", "taker_side").iter_rows()
                   if through(side, b, yp, ts_side)]
            if hit:
                f = fee(1, b, maker=True, fee_type=r["fee_type"], mult=r["fee_multiplier"])
                rows.append({**b0, "rule": f"P_maker_{int(b * 100)}", "n": 1.0, "price": b, "t_in": hit[0],
                             "cost_usd": b + f, "payoff_usd": float(win)})
    return pl.DataFrame(rows)


# ---- books, ledgers, Sharpe ----------------------------------------------------------------------
BOOK_OPEN = 20.0       # $ open at once (mentions uses ~$40 of the $100; this leaves room for others)
BOOK_EVENT = 5.0       # $ per event


def capped_book(f: pl.DataFrame, *, max_open: float = BOOK_OPEN, per_event: float = BOOK_EVENT) -> pl.DataFrame:
    """Keep positions in fill order while open cost <= ``max_open`` and per-event cost <= ``per_event``
    (copied pattern of ``approval2.capped_book`` plus the event cap). Needs t_in, settle_ts, cost_usd."""
    keep, open_pos = [], []
    for r in f.sort("t_in", "ticker").iter_rows(named=True):
        open_pos = [z for z in open_pos if z[0] > r["t_in"]]
        tot = sum(z[1] for z in open_pos)
        evt = sum(z[1] for z in open_pos if z[2] == r["event_ticker"])
        if tot + r["cost_usd"] > max_open + 1e-9 or evt + r["cost_usd"] > per_event + 1e-9:
            continue
        open_pos.append((r["settle_ts"], r["cost_usd"], r["event_ticker"]))
        keep.append(r)
    return pl.DataFrame(keep, schema=f.schema)


def to_ledger(f: pl.DataFrame, strategy: str, variant: str) -> pl.DataFrame:
    from albert.research.vol_metrics import LEDGER_COLS

    return (f.with_columns(pl.lit(strategy).alias("strategy"), pl.lit(variant).alias("variant"),
                           pl.concat_str("ticker", pl.col("t_in").cast(pl.String), separator="@").alias("struct_id"))
            .select("strategy", "variant", "split", *LEDGER_COLS))


def marks_from_candles(c: pl.DataFrame) -> pl.DataFrame:
    """Hourly YES mids (two-sided book, else last price) from this study's candles, forward-filled."""
    x = (c.sort("ticker", "ts").with_columns(
        pl.when((pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("ask") > pl.col("bid")))
        .then((pl.col("bid") + pl.col("ask")) / 2).otherwise(pl.col("p_close")).alias("yes_mid"))
        .filter(pl.col("yes_mid").is_not_null())
        .group_by("ticker", pl.col("ts").dt.truncate("1h").alias("h")).agg(pl.col("yes_mid").last(), pl.col("ts").max()))
    return x.select("ticker", "ts", "yes_mid").sort("ticker", "ts")


def book_metrics(led: pl.DataFrame, marks: pl.DataFrame) -> dict[str, Any]:
    from albert.research.vol_metrics import ledger_metrics

    if led.height < 3:
        return {"positions": led.height}
    m = ledger_metrics(led, marks)
    return {k: m.get(k) for k in ("positions", "events", "usd_per_week", "ann_vol_pct", "sharpe", "sharpe_ci90",
                                  "sharpe_ex_top5", "max_dd_usd", "weeks_positive_pct", "first_day", "last_day")}


# ---- analysis driver ------------------------------------------------------------------------------
def holm(pvals: list[float]) -> list[float]:
    """Holm-adjusted p-values (step-down), in the input order."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj.tolist()


def p_one_sided(t: float | None) -> float:
    return 1.0 if t is None or not math.isfinite(t) else 0.5 * math.erfc(t / math.sqrt(2))


def xcat_marks(tickers: list[str]) -> pl.DataFrame:
    parts = sorted((XCAT / "hourly").glob("part-*.parquet"))
    c = (pl.scan_parquet(parts).filter(pl.col("ticker").is_in(tickers))
         .select("ticker", pl.col("end_ts").alias("ts"), pl.col("bid_close").alias("bid"),
                 pl.col("ask_close").alias("ask"), "p_close", "volume").collect())
    return marks_from_candles(c)


def choose(dev: pl.DataFrame, min_events: int = 10) -> tuple[str | None, pl.DataFrame]:
    """Frozen choice on dev: the cell with the highest c/contract among those with >= ``min_events``
    events and Holm-adjusted one-sided p < 0.05 (over every dev cell tried); None if none passes."""
    if dev.height == 0:
        return None, dev
    d = dev.with_columns(pl.Series("p", [p_one_sided(t) for t in dev["t_event"].to_list()]))
    d = d.with_columns(pl.Series("p_holm", holm(d["p"].to_list())))
    ok = d.filter((pl.col("events") >= min_events) & (pl.col("p_holm") < 0.05)).sort("c_per_contract", descending=True)
    return (ok["cell"][0] if ok.height else None), d


def analyze() -> None:
    res: dict[str, Any] = {}
    s = pl.read_parquet(OUT / "sample.parquet")
    t, c = load_windows()
    s_short = s.filter(pl.col("wclass") == "short")
    # H1 upper bounds
    h1 = h1_tables(s_short, t, c)
    h1["h1_cells"].write_csv(OUT / "h1_cells.csv")
    p = h1["h1_prints"]
    res["h1_prints"] = p.height
    # how fast the winner's offer disappears after the occurrence stamp (short windows, minute candles)
    cc = (c.join(s_short.select("ticker", "category", "split", "occurrence_time", (pl.col("result") == "yes").alias("y")),
                 on="ticker")
          .with_columns(((pl.col("ts") - pl.col("occurrence_time")).dt.total_seconds() / 60).alias("rel_min"),
                        pl.when(pl.col("y")).then(pl.col("ask")).otherwise(1 - pl.col("bid")).alias("ask_w")))
    # as-of semantics (candles are sparse: no candle means no change): the winner's ask on the last
    # candle ended by occurrence + 1 min, then the first later candle where it is >= 99c (or no offer)
    at = (cc.filter(pl.col("rel_min") <= 1).sort("ts").group_by("ticker", maintain_order=True).last()
          .select("ticker", pl.col("ask_w").alias("ask_w_first")))
    later = (cc.filter((pl.col("rel_min") > 1) & (pl.col("ask_w") >= 0.99)).group_by("ticker")
             .agg(pl.col("rel_min").min().alias("t99")))
    clear = (s_short.select("ticker", "category", "split").join(at, on="ticker", how="inner")
             .join(later, on="ticker", how="left")
             .with_columns(pl.when(pl.col("ask_w_first") >= 0.99).then(0.0).otherwise(pl.col("t99")).alias("min_to_99")))
    clear_t = clear.group_by("category", "split").agg(
        pl.len().alias("markets"), (pl.col("ask_w_first") >= 0.99).mean().round(3).alias("share_ask99_at_first_candle"),
        pl.col("min_to_99").median().alias("min_to_99_p50"), pl.col("min_to_99").quantile(0.9).alias("min_to_99_p90"),
        pl.col("min_to_99").is_null().mean().round(3).alias("share_never_99")).sort("category", "split")
    clear_t.write_csv(OUT / "h1_clear_speed.csv")
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(300)
    pl.Config.set_tbl_rows(80)
    print(h1["h1_cells"])
    print(clear_t)
    # family L
    lr_rows = l_rules(s, t, c)
    if lr_rows.height:
        lr_rows = lr_rows.with_columns((pl.col("payoff_usd") > 0).cast(pl.Float64).alias("win"),
                                       pl.concat_str("rule", pl.lit("|d"), pl.col("delay_h").cast(pl.String)).alias("cell"),
                                       ((pl.col("settle_ts") - pl.col("t_in")).dt.total_seconds() / 3600).alias("hold_h"))
        lt = cell_stats(lr_rows, ["cell", "split"])
        lt.write_csv(OUT / "family_L_cells.csv")
        print(lt)
        pick, devt = choose(lt.filter(pl.col("split") == "dev"))
        devt.write_csv(OUT / "family_L_dev_holm.csv")
        res["family_L"] = {"cells_tried": int(lt.filter(pl.col("split") == "dev").height), "frozen": pick}
        if pick:
            ho = lt.filter((pl.col("cell") == pick) & (pl.col("split") == "holdout"))
            res["family_L"]["holdout"] = ho.to_dicts()
        led_rows = lr_rows.filter(pl.col("cell") == (pick or lt.filter(pl.col("split") == "dev")
                                                     .sort("c_per_contract", descending=True)["cell"][0]))
        _ledger_and_metrics(led_rows, "family_L", marks_from_candles(c), res)
    # settlement sources of the sampled series (is the outcome public and timestamped somewhere allowed?)
    try:
        src = series_sources()
        ss = (s.group_by("series_ticker", "category", "wclass").agg(pl.col("event_ticker").n_unique().alias("events"))
              .join(src.group_by("series_ticker").agg(pl.col("source_name").unique().str.join("; ").alias("sources"),
                                                       pl.col("source_url").unique().str.join("; ").alias("urls")),
                    on="series_ticker", how="left").sort("events", descending=True))
        ss.write_csv(OUT / "sample_series_sources.csv")
    except RuntimeError as error:
        print(f"no series sources: {error}")
    # family P: sibling already closed
    allm = markets()
    pr = p_rules(s_short, t, c, allm)
    if pr.height:
        pr = pr.with_columns((pl.col("payoff_usd") > 0).cast(pl.Float64).alias("win"), pl.col("rule").alias("cell"),
                             ((pl.col("settle_ts") - pl.col("t_in")).dt.total_seconds() / 3600).alias("hold_h"))
        pt = cell_stats(pr, ["cell", "split"])
        pt.write_csv(OUT / "family_P_cells.csv")
        print(pt)
        print(cell_stats(pr, ["cell", "category"]))
        pick, devt = choose(pt.filter(pl.col("split") == "dev"))
        devt.write_csv(OUT / "family_P_dev_holm.csv")
        res["family_P"] = {"cells_tried": int(pt.filter(pl.col("split") == "dev").height), "frozen": pick}
        if pick:
            res["family_P"]["holdout"] = pt.filter((pl.col("cell") == pick) & (pl.col("split") == "holdout")).to_dicts()
        best = pick or pt.filter(pl.col("split") == "dev").sort("c_per_contract", descending=True)["cell"][0]
        _ledger_and_metrics(pr.filter(pl.col("cell") == best), "family_P", marks_from_candles(c), res)
    _save_results(res)


def _save_results(res: dict[str, Any]) -> None:
    p = OUT / "results.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(res)
    p.write_text(json.dumps(old, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


def near_certain() -> None:
    """H3 / H5: first touch of a 95-99c leading-side ask on xcat whole-life hourly candles, by series
    settlement-lag class, ask band and category; capped-book ledger and Sharpe."""
    res: dict[str, Any] = {}
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(300)
    pl.Config.set_tbl_rows(80)
    allm = markets()
    # entries inside the study window only (xcat markets close 2026-08-15..10-31; a first touch in 2025
    # would put a 1-year hold on the $20 book)
    nc = near_certain_entries().filter(pl.col("end_ts") >= datetime(2026, 8, 1, tzinfo=UTC))
    lag = series_lag_class(allm)
    nc = nc.join(lag, on="series_ticker", how="left").with_columns(pl.col("lag_class").fill_null("unknown"),
                                                                   pl.when(pl.col("ask_l") < 0.97).then(pl.lit("95-96c"))
                                                                   .otherwise(pl.lit("97-99c")).alias("band"))
    nc.write_parquet(OUT / "near_certain_entries.parquet")
    nc = nc.with_columns(pl.when(pl.col("hold_h") <= 6).then(pl.lit("hold<=6h")).when(pl.col("hold_h") <= 48)
                         .then(pl.lit("hold6-48h")).otherwise(pl.lit("hold>48h")).alias("hold_class"))
    for keys, name in ((["lag_class", "split"], "h5_lag"), (["hold_class", "split"], "h5_hold_descriptive"),
                       (["band", "split"], "h3_band"), (["category", "split"], "h3_category")):
        tab = cell_stats(nc.filter(pl.col("split") != "late"), keys)
        tab.write_csv(OUT / f"{name}.csv")
        print(tab)
    nc_led = nc.filter(pl.col("split") != "late").with_columns(pl.col("end_ts").alias("t_in"),
                                                              pl.col("settlement_ts").alias("settle_ts"))
    _ledger_and_metrics(nc_led.with_columns(pl.lit("all").alias("cell")), "near_certain_95_99", xcat_marks, res)
    _save_results(res)


def _ledger_and_metrics(rows: pl.DataFrame, name: str, marks, res: dict[str, Any]) -> None:
    """Capped book per split, ledger to ``ledgers/``, Sharpe/vol. ``marks`` is a frame or a function of
    the booked tickers (so only those are loaded)."""
    (OUT / "ledgers").mkdir(exist_ok=True)
    out = {}
    for sp in ("dev", "holdout"):
        x = rows.filter(pl.col("split") == sp)
        if x.height == 0:
            continue
        book = capped_book(x.select("ticker", "event_ticker", "side", "n", "t_in", "cost_usd", "settle_ts",
                                    "payoff_usd", "split"))
        led = to_ledger(book, name, str(rows["cell"][0]))
        led.write_parquet(OUT / "ledgers" / f"ledger_{name}_{sp}.parquet")
        mk = marks(led["ticker"].unique().to_list()) if callable(marks) else marks
        out[sp] = {"pnl_usd": round(float((led["payoff_usd"] - led["cost_usd"]).sum()), 2), **book_metrics(led, mk)}
    res[f"book_{name}"] = out


# ---- robots --------------------------------------------------------------------------------------
ROBOTS = {"api.elections.kalshi.com": ["/trade-api/v2/markets", "/trade-api/v2/markets/trades",
                                       "/trade-api/v2/historical/trades", "/trade-api/v2/series"],
          # candidate "outcome is public" sources for the series with post-occurrence windows (checked,
          # not fetched): YouTube views, race calls, hurricanes, speeches, awards
          "charts.youtube.com": ["/artist/", "/charts/TopArtists/us/daily", "/youtubei/v1/browse"],
          "www.youtube.com": ["/watch", "/@TaylorSwift"],
          "apnews.com": ["/projects/election-results-2026/"],
          "www.nhc.noaa.gov": ["/text/", "/gis/"],
          "rollcall.com": ["/factbase/"],
          "deadline.com": ["/"], "variety.com": ["/"], "www.emmys.com": ["/"],
          "ramp.com": ["/data/ai-index"], "arena.ai": ["/leaderboard/text/overall-no-style-control"],
          "lmarena.ai": ["/leaderboard"]}


def series_sources() -> pl.DataFrame:
    """Settlement source (name, url) per series from the newest cached ``/series`` payload."""
    from albert.http import CachedClient, read_raw

    base_url = "https://api.elections.kalshi.com/trade-api/v2/series"
    for i in range(30):
        day = (datetime.now(UTC) - timedelta(days=i)).strftime("%Y-%m-%d")
        key = CachedClient.cache_key(base_url, {"__tag": day})
        for root in [Path("data/raw/kalshi"), *[Path(p) for p in sorted(glob.glob("data/cache/*/raw/kalshi"))]]:
            f = root / f"{key}.json.zst"
            if f.exists():
                rows = []
                for s in read_raw(f)["response"].get("series") or []:
                    for src in s.get("settlement_sources") or [{}]:
                        rows.append({"series_ticker": s.get("ticker"), "source_name": (src or {}).get("name"),
                                     "source_url": (src or {}).get("url"), "series_day": day})
                return pl.DataFrame(rows, schema={"series_ticker": pl.String, "source_name": pl.String,
                                                  "source_url": pl.String, "series_day": pl.String})
    raise RuntimeError("no cached /series payload in the last 30 days")


def robots() -> list[dict[str, Any]]:
    """Each host's own robots.txt, evaluated for every agent the policy names (``counts.robots_verdict``)."""
    import httpx

    from albert.research.counts import AGENTS, robots_verdict

    rows = []
    for host, paths in ROBOTS.items():
        try:
            r = httpx.get(f"https://{host}/robots.txt", timeout=20, follow_redirects=True,
                          headers={"User-Agent": "albert-research/0.2"})
            status, text = r.status_code, (r.text if r.status_code == 200 else "")
        except httpx.HTTPError as error:
            status, text = None, ""
            print(f"robots {host}: {error}")
        for p in paths:
            v = robots_verdict(text, p) if status in (200, 404, 410) else dict.fromkeys(AGENTS, False)
            rows.append({"host": host, "path": p, "http_status": status,
                         "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                         **{f"allow_{a}": v[a] for a in AGENTS}, "allowed": all(v.values())})
    (OUT / "robots.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows, indent=1))
    return rows


# ---- CLI -----------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.late_resolution")
    ap.add_argument("step", choices=["base", "fetch-nov", "census", "sample", "fetch-windows", "analyze", "robots", "near-certain"])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    fn = {"base": base, "fetch-nov": fetch_nov}
    g = globals()
    fn.update({k: g[v] for k, v in {"census": "census", "sample": "sample", "fetch-windows": "fetch_windows",
                                    "analyze": "analyze", "robots": "robots",
                                    "near-certain": "near_certain"}.items() if v in g})
    fn[a.step]()


if __name__ == "__main__":
    main()
