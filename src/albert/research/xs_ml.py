"""Cross-sectional, walk-forward predictive model over all non-financial Kalshi markets (``xs_ml``).

Question (swarm 2026-11-23): instead of testing one hypothesis at a time, fit ONE model over every
non-financial Kalshi market that predicts settlement minus the executable price at fixed decision
leads, refit monthly on an expanding window, every prediction strictly out of sample. Does it beat
the raw price and the naive favourite-longshot (FLB) recalibration, and does what it adds survive
the spread, Kalshi's fees and a capped $100 book, as taker or maker?

Pipeline (``python -m albert.research.xs_ml <cmd>``; outputs ``data/research/xs_ml/``, raw API
responses and candles ``data/cache/xs_ml/``):

* ``scan``: settled-market listings the shared tables lack: the live ``/markets`` endpoint for closes
  2026-11-01 .. now (per UTC day), and ``/historical/markets`` newest-first from the API cutoff
  back to 2026-09-05 (the shared table's historical part misses early-closed markets there).
* ``frame``: every settled binary market of every non-financial series (``flb_common
  .coarse_category``) with close >= 2026-03-01: ``flb_xs`` historical frame (to ~09-10) +
  ``kalshi_markets`` (08-15 .. 10-31) + the scan, deduplicated (newest source wins). The
  scheduled time ``T`` = expected expiration minus the series' modal (expected expiration - close)
  offset on NO-settled markets (the ``flb_xs`` rule; it never uses the actual close of an
  early-closing market); a market without an expected expiration gets ``T`` = close only if it
  cannot close early. One row per event in ``frame_events.parquet`` (legs, YES legs, last
  settlement) feeds the series-history features.
* ``sample``: volume-blind stratified sample of *events* (group x month of T), seeded hash order,
  until the cell's market budget is met; all legs kept up to ``MAX_LEGS`` (seeded subset beyond).
  Nothing is selected on volume, price or result.
* ``fetch``: hourly OHLC candles over ``[max(open, T - 6 d), min(close, T)]``. Candles another
  study already cached with full OHLC over a window that contains this one (``xcat`` hourly =
  whole life, ``ta_recurring`` = last 21 days) are copied for the sampled tickers only; the rest
  come from the API (historical endpoint per market before the cutoff, live batch per event).
* ``panel``: one row per market x lead (T - 72 h, T - 24 h, T - 6 h) while the market is open:
  the quote of the last candle ending at or before t, and features known at t only.
* ``model``: walk-forward monthly refits (train = rows whose decision AND settlement precede the
  month), models: market mid, FLB bucket x group baseline, logit-linear recalibration by group,
  L2 logistic on all features, gradient-boosted stumps/depth-2 trees (numpy only).
* ``dev`` / ``freeze`` / ``holdout`` / ``report``: taker and maker rules from the OOS predictions,
  every cell counted, Holm-corrected; best dev cells frozen in ``frozen.json`` before the holdout
  (decision months 2026-10 and 2026-11) is scored once; ledgers for ``vol_metrics``.

Result (2026-11-23, ``data/research/xs_ml/report.md``): NO EDGE. The model beats the mid and the FLB
curve out of sample in log loss (holdout logistic +19 milli-log-loss per row over the FLB curve, t 4.3),
mostly from one-sided books, but the frozen taker (-5.6c, t -2.8) and maker (-2.7c, t -1.9) rules lose
on the holdout; the only Holm survivor is the known wide maker NO on <= 5c legs (culture_flow), where
the model gate keeps the same holdout fills as the ungated control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

OUT = Path("data/research/xs_ml")
CACHE = Path("data/cache/xs_ml")
TABLES = Path("data/tables")
FLB_FRAME = "data/research/flb_xs/frame/part-*.parquet"
FRAME_FROM = datetime(2026, 3, 1, tzinfo=UTC)       # frame (series history) starts here
T_FROM = datetime(2026, 5, 1, tzinfo=UTC)           # decisions: scheduled time T in [T_FROM, T_TO)
T_TO = datetime(2026, 11, 23, tzinfo=UTC)
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
NOV_SCAN_FROM = datetime(2026, 11, 1, tzinfo=UTC)
HIST_SCAN_STOP = datetime(2026, 9, 5, tzinfo=UTC)
LEADS_H = (72, 24, 6)
WINDOW = timedelta(days=6)
MAX_LEGS = 30
SEED = "xs_ml|20261123"
# market budget per (group, month of T); ~2,000 markets a month
BUDGET = {"Sports": 500, "Weather": 300, "Entertainment": 350, "Mentions": 250, "Politics": 300,
          "Science & Tech": 200, "Other": 100}
FRAME_CHUNKS = 24
COMPACT = {"ticker": pl.String, "event_ticker": pl.String, "series_ticker": pl.String, "market_type": pl.String,
           "open_time": pl.Datetime("us", "UTC"), "close_time": pl.Datetime("us", "UTC"),
           "expected_expiration_time": pl.Datetime("us", "UTC"), "settlement_ts": pl.Datetime("us", "UTC"),
           "result": pl.String, "can_close_early": pl.Boolean, "strike_type": pl.String,
           "floor_strike": pl.Float64, "cap_strike": pl.Float64, "status": pl.String}


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _log(*a: Any) -> None:
    print(*a, flush=True)


def _api(min_interval: float = 0.5):
    """Kalshi public-data client (no keys); raw responses cached under ``data/cache/xs_ml/kalshi``."""
    from albert.config import Settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    def nap(x: float) -> None:
        time.sleep(min(x, 20.0) * (0.75 + 0.5 * random.random()))

    return k.Kalshi(CachedClient("kalshi", Settings().kalshi_base, CACHE, min_interval=max(min_interval, 0.5),
                                 max_retries=40, sleep=nap))


def h(key: str) -> str:
    return hashlib.sha256(f"{SEED}|{key}".encode()).hexdigest()


# ---- scan ----------------------------------------------------------------------------------------
def compact_row(m: dict[str, Any]) -> dict[str, Any]:
    """The fields this study needs from one ``/markets`` or ``/historical/markets`` record."""
    from albert.venues.kalshi import num, parse_ts, series_of

    ev = m.get("event_ticker") or ""
    return {"ticker": m.get("ticker"), "event_ticker": ev, "series_ticker": series_of(ev) if ev else None,
            "market_type": m.get("market_type"), "open_time": parse_ts(m.get("open_time")),
            "close_time": parse_ts(m.get("close_time")),
            "expected_expiration_time": parse_ts(m.get("expected_expiration_time")),
            "settlement_ts": parse_ts(m.get("settlement_ts")), "result": m.get("result") or None,
            "can_close_early": m.get("can_close_early"), "strike_type": m.get("strike_type"),
            "floor_strike": num(m.get("floor_strike")), "cap_strike": num(m.get("cap_strike")),
            "status": m.get("status")}


def _trim() -> None:
    """Hand freed heap back to the OS (big JSON pages fragment it; the light lane caps RSS at 300 MB)."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _rows_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=COMPACT, orient="row") if rows else pl.DataFrame(schema=COMPACT)


def scan_live_day(api, day: datetime) -> Path:
    path = CACHE / "scan" / f"live-{day:%Y%m%d}.parquet"
    if path.exists():
        return path
    frames: list[pl.DataFrame] = []      # one small frame per page keeps memory flat (300 MB lane)
    params = {"status": "settled", "mve_filter": "exclude", "limit": 1000,
              "min_close_ts": int(day.timestamp()), "max_close_ts": int((day + timedelta(days=1)).timestamp()) - 1}
    for page in api._cursor_pages("/markets", params, "markets"):
        frames.append(_rows_frame([compact_row(m) for m in page.payload.get("markets") or []]))
        if len(frames) % 10 == 0:
            _trim()
    out = pl.concat(frames) if frames else _rows_frame([])
    _save(out, path)
    _log(f"live {day:%Y-%m-%d}: {out.height} markets")
    return path


def scan_hist(api, stop: datetime = HIST_SCAN_STOP) -> None:
    """``/historical/markets`` newest first until a whole page closed before ``stop``."""
    from albert.venues.kalshi import parse_ts

    d = CACHE / "scan"
    if (d / "hist.done").exists():
        return
    frames: list[pl.DataFrame] = []
    part = 0
    pages_iter = api._cursor_pages("/historical/markets", {"mve_filter": "exclude", "limit": 1000}, "markets")
    for pages, page in enumerate(pages_iter, start=1):
        ms = page.payload.get("markets") or []
        frames.append(_rows_frame([compact_row(m) for m in ms]))
        if len(frames) % 10 == 0:
            _trim()
        closes = [c for c in (parse_ts(m.get("close_time")) for m in ms) if c is not None]
        if pages % 25 == 0 or (closes and max(closes) < stop):
            _save(pl.concat(frames), d / f"hist-{part:04d}.parquet")
            part, frames = part + 1, []
            _log(f"hist: {pages} pages, oldest close {min(closes) if closes else None}")
        if closes and max(closes) < stop:
            break
    if frames:
        _save(pl.concat(frames), d / f"hist-{part:04d}.parquet")
    (d / "hist.done").write_text(datetime.now(UTC).isoformat())


def scan() -> None:
    """Two client threads (the API limit): one starts on the historical gap, both then work through
    the live days from a shared queue."""
    import queue

    days = []
    day = NOV_SCAN_FROM
    now = datetime.now(UTC)
    while day < now:
        days.append(day)
        day += timedelta(days=1)
    # the last (partial) day is rescanned next time: drop a stale file of today
    today = CACHE / "scan" / f"live-{days[-1]:%Y%m%d}.parquet"
    if today.exists():
        today.unlink()
    q: queue.Queue[datetime] = queue.Queue()
    for d in days:
        q.put(d)

    def worker(first_hist: bool) -> None:
        api = _api(0.5)
        if first_hist:
            scan_hist(api)
        while True:
            try:
                d = q.get_nowait()
            except queue.Empty:
                return
            scan_live_day(api, d)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in [pool.submit(worker, True), pool.submit(worker, False)]:
            f.result()


# ---- frame ---------------------------------------------------------------------------------------
def series_table() -> pl.DataFrame:
    """Non-financial series with their coarse group, frequency and fee schedule."""
    from albert.research.flb_common import coarse_category

    s = pl.read_parquet(TABLES / "kalshi_series.parquet").select(
        "series_ticker", "category", "title", "frequency", "fee_type", "fee_multiplier")
    grp = [coarse_category(c, t, ti) for t, c, ti in s.select("series_ticker", "category", "title").iter_rows()]
    return s.with_columns(pl.Series("group", grp, dtype=pl.String)).filter(
        pl.col("group").is_not_null() & ~pl.col("series_ticker").str.starts_with("KXMVE")).drop("title")


FRAME_COLS = list(COMPACT)


def _sources(ser: pl.Series) -> list[pl.LazyFrame]:
    """Market listings, newest source first, restricted to series ``ser``; compact columns."""
    import glob

    def norm(lf: pl.LazyFrame, src: str, have: set[str]) -> pl.LazyFrame:
        cols = [pl.col(c).cast(t) if c in have else pl.lit(None, t).alias(c) for c, t in COMPACT.items()]
        return lf.filter(pl.col("series_ticker").is_in(ser) & (pl.col("close_time") >= FRAME_FROM)).select(
            *cols, pl.lit(src).alias("src"))

    out = []
    scan_files = sorted(glob.glob(str(CACHE / "scan" / "*.parquet")))
    if scan_files:
        out.append(norm(pl.scan_parquet(scan_files), "scan", set(COMPACT)))
    km = pl.scan_parquet(TABLES / "kalshi_markets.parquet")
    out.append(norm(km, "table", set(km.collect_schema().names())))
    fl = pl.scan_parquet(sorted(glob.glob(FLB_FRAME)))
    out.append(norm(fl, "flb", set(fl.collect_schema().names())))
    return out


def series_offsets(fr: pl.DataFrame) -> pl.DataFrame:
    """Modal (expected expiration - close) in minutes on NO-settled markets per series (dev months
    first; series first seen in the holdout use their own)."""
    x = fr.filter((pl.col("result") == "no") & pl.col("expected_expiration_time").is_not_null()).with_columns(
        (pl.col("expected_expiration_time") - pl.col("close_time")).dt.total_minutes().alias("off"),
        (pl.col("close_time") < HOLDOUT_FROM).alias("dev"))
    dev = x.filter(pl.col("dev")).group_by("series_ticker").agg(pl.col("off").mode().sort().first().alias("o1"))
    allx = x.group_by("series_ticker").agg(pl.col("off").mode().sort().first().alias("o2"))
    return allx.join(dev, on="series_ticker", how="left").select(
        "series_ticker", pl.coalesce("o1", "o2").cast(pl.Float64).alias("offset_min"))


def build_frame_chunk(ser: pl.DataFrame, i: int, k: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    sub = ser.filter(pl.col("series_ticker").hash(seed=11) % k == i)
    lf = pl.concat(_sources(sub["series_ticker"].implode()), how="vertical")
    fr = (lf.filter((pl.col("market_type") == "binary") & pl.col("result").is_in(["yes", "no"]))
          .collect(engine="streaming").unique("ticker", keep="first", maintain_order=True))
    off = series_offsets(fr)
    fr = fr.join(off, on="series_ticker", how="left").with_columns(
        pl.when(pl.col("expected_expiration_time").is_not_null())
        .then(pl.col("expected_expiration_time") - pl.duration(minutes=pl.col("offset_min").fill_null(0).round(0).cast(pl.Int64)))
        .when(pl.col("can_close_early") == False).then(pl.col("close_time"))
        .otherwise(None).alias("T"),
        (pl.col("result") == "yes").cast(pl.Int8).alias("y"),
        pl.coalesce("settlement_ts", pl.col("close_time") + timedelta(days=1)).alias("settle"))
    fr = fr.join(sub.select("series_ticker", "group", "category"), on="series_ticker", how="left")
    keep = ["ticker", "event_ticker", "series_ticker", "group", "open_time", "close_time", "expected_expiration_time",
            "settlement_ts", "settle", "T", "y", "can_close_early", "strike_type", "floor_strike", "cap_strike", "src"]
    fr = fr.select(keep)
    return fr, frame_events_of(fr)


def frame_events_of(fr: pl.DataFrame) -> pl.DataFrame:
    return fr.group_by("event_ticker").agg(
        pl.col("series_ticker").first(), pl.col("group").first(), pl.len().alias("n_legs"),
        pl.col("y").sum().cast(pl.Int32).alias("n_yes"), pl.col("T").min().alias("T_min"),
        pl.col("T").max().alias("T_max"), pl.col("open_time").min().alias("open_min"),
        pl.col("close_time").max().alias("close_max"), pl.col("settle").max().alias("settle_max"),
        pl.col("T").is_null().sum().alias("n_no_T"))


def build_frame() -> None:
    ser = series_table()
    d = OUT / "frame"
    d.mkdir(parents=True, exist_ok=True)
    evs = []
    for i in range(FRAME_CHUNKS):           # resumable: a finished chunk is not rebuilt
        part, evp = d / f"part-{i:02d}.parquet", d / f"ev-{i:02d}.parquet"
        if part.exists() and evp.exists():
            evs.append(pl.read_parquet(evp))
            continue
        if part.exists():
            ev = frame_events_of(pl.read_parquet(part))
        else:
            fr, ev = build_frame_chunk(ser, i, FRAME_CHUNKS)
            _save(fr, part)
            _log(f"frame chunk {i}: {fr.height} markets, {ev.height} events; src {dict(fr.group_by('src').len().iter_rows())}")
            del fr
        _save(ev, evp)
        evs.append(ev)
    ev = pl.concat(evs)
    _save(ev, OUT / "frame_events.parquet")
    m = ev.with_columns(pl.col("T_min").dt.strftime("%Y-%m").alias("m")).group_by("group", "m").agg(
        pl.len().alias("events"), pl.col("n_legs").sum().alias("markets")).sort("group", "m")
    m.write_csv(OUT / "frame_counts.csv")
    _log(str(m))


# ---- sample --------------------------------------------------------------------------------------
def draw_sample() -> pl.DataFrame:
    ev = pl.read_parquet(OUT / "frame_events.parquet").filter(
        pl.col("T_min").is_not_null() & (pl.col("T_min") >= T_FROM) & (pl.col("T_min") < T_TO))
    ev = ev.with_columns(pl.col("T_min").dt.strftime("%Y-%m").alias("month"),
                         pl.col("event_ticker").map_elements(h, return_dtype=pl.String).alias("hk"),
                         pl.col("n_legs").clip(upper_bound=MAX_LEGS).alias("n_cap")).sort("hk")
    ev = ev.with_columns(pl.col("n_cap").cum_sum().over("group", "month").alias("cum"))
    budget = pl.DataFrame({"group": list(BUDGET), "budget": list(BUDGET.values())})
    ev = ev.join(budget, on="group", how="left").with_columns(pl.col("budget").fill_null(100))
    # events in hash order until the cell's budget is met (the event that crosses it is kept)
    pick = ev.filter(pl.col("cum") - pl.col("n_cap") < pl.col("budget"))
    cells = ev.group_by("group", "month").agg(pl.len().alias("pop_events"), pl.col("n_legs").sum().alias("pop_markets"))
    pcount = pick.group_by("group", "month").agg(pl.len().alias("smp_events"), pl.col("n_cap").sum().alias("smp_markets"))
    cells = cells.join(pcount, on=["group", "month"], how="left").with_columns(
        (pl.col("smp_events") / pl.col("pop_events")).alias("p_event")).sort("group", "month")
    cells.write_csv(OUT / "sample_cells.csv")
    import glob

    evset = pick["event_ticker"].implode()
    mk = (pl.scan_parquet(sorted(glob.glob(str(OUT / "frame" / "part-*.parquet"))))
          .filter(pl.col("event_ticker").is_in(evset)).collect())
    mk = mk.with_columns(pl.col("ticker").map_elements(h, return_dtype=pl.String).alias("hk"))
    mk = mk.sort("hk").with_columns(pl.int_range(pl.len()).over("event_ticker").alias("rk"),
                                    pl.len().over("event_ticker").alias("n_legs_frame"))
    mk = mk.filter(pl.col("rk") < MAX_LEGS).drop("rk", "hk")
    mk = mk.join(pick.select("event_ticker", "month", "n_legs"), on="event_ticker").join(
        cells.select("group", "month", "p_event"), on=["group", "month"], how="left")
    ser = series_table().select("series_ticker", "category", "frequency", "fee_type", "fee_multiplier")
    mk = mk.join(ser, on="series_ticker", how="left")
    _save(mk, OUT / "sample.parquet")
    _log(str(cells))
    _log(f"sample: {mk.height} markets, {mk['event_ticker'].n_unique()} events, {mk['series_ticker'].n_unique()} series")
    return mk


# ---- candles -------------------------------------------------------------------------------------
OHLC_COLS = ["ticker", "end_ts", "period", "p_open", "p_high", "p_low", "p_close", "p_mean", "bid_open", "bid_high",
             "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close", "volume", "oi"]
# (name, done list, part glob, window kind): "life" = whole life, "close21" = 21 days before close
REUSE = (("xcat", "data/research/xcat/hourly/done.parquet", "data/research/xcat/hourly/part-*.parquet", "life"),
         ("ta_recurring", "data/cache/ta_recurring/hourly/done.parquet", "data/cache/ta_recurring/hourly/part-*.parquet",
          "close21"))


def candle_plan(s: pl.DataFrame) -> pl.DataFrame:
    lo = pl.max_horizontal("open_time", pl.col("T") - WINDOW)
    hi = pl.min_horizontal("close_time", "T")
    return s.filter(pl.col("T").is_not_null() & (hi > lo)).select(
        "ticker", "event_ticker", pl.col("settlement_ts").fill_null(pl.col("close_time")),
        lo.alias("lo"), hi.alias("hi"), "close_time", pl.lit(60).alias("period"))


def reuse_local(plan: pl.DataFrame) -> set[str]:
    import glob

    d = CACHE / "candles"
    d.mkdir(parents=True, exist_ok=True)
    got: set[str] = set()
    for name, done_path, parts, kind in REUSE:
        if not Path(done_path).exists():
            continue
        done = set(pl.read_parquet(done_path)["ticker"].to_list())
        cand = plan.filter(pl.col("ticker").is_in(list(done - got)))
        if kind == "close21":   # its window [close - 21 d, close] must contain ours
            cand = cand.filter(pl.col("lo") >= pl.col("close_time") - timedelta(days=21))
        have = cand["ticker"].to_list()
        if not have:
            continue
        c = (pl.scan_parquet(sorted(glob.glob(parts))).filter(pl.col("ticker").is_in(have)).select(OHLC_COLS)
             .collect().unique(["ticker", "end_ts"], keep="last"))
        part = len(list(d.glob("part-*.parquet")))
        c.write_parquet(d / f"part-{part:05d}-{name}.parquet")
        got |= set(have)
        _log(f"reuse {name}: {len(have)} markets, {c.height} candles")
    return got


def fetch(workers: int = 2) -> None:
    from albert.research import xcat

    s = pl.read_parquet(OUT / "sample.parquet")
    plan_all = candle_plan(s)
    d = CACHE / "candles"
    d.mkdir(parents=True, exist_ok=True)
    done_path = d / "done.parquet"
    done = set(pl.read_parquet(done_path)["ticker"].to_list()) if done_path.exists() else set()
    if not (d / "reuse.done").exists():
        done |= reuse_local(plan_all.filter(~pl.col("ticker").is_in(list(done))))
        _save(pl.DataFrame({"ticker": sorted(done)}, schema={"ticker": pl.String}), done_path)
        (d / "reuse.done").write_text("1")
    plan = plan_all.filter(~pl.col("ticker").is_in(list(done)))
    apis = [_api(0.5) for _ in range(workers)]
    cutoff = apis[0].cutoff()["market_settled_ts"]
    jobs = xcat._jobs(plan, cutoff)
    _log(f"fetch: {plan_all.height} markets, {len(done)} cached, {plan.height} to fetch in {len(jobs)} requests "
         f"(cutoff {cutoff})")
    lock = threading.Lock()
    buf: list[dict[str, Any]] = []
    fin: list[str] = []
    failed: list[str] = []
    part = [len(list(d.glob("part-*.parquet")))]
    t0 = time.time()

    def flush() -> None:
        nonlocal buf
        if buf:
            pl.DataFrame(buf, schema=xcat.OHLC_SCHEMA).select(OHLC_COLS).write_parquet(d / f"part-{part[0]:05d}.parquet")
            part[0] += 1
        buf = []
        _save(pl.DataFrame({"ticker": sorted(done | set(fin))}, schema={"ticker": pl.String}), done_path)
        if failed:
            _save(pl.DataFrame({"ticker": sorted(set(failed))}), d / "failed.parquet")

    def run(ij: tuple[int, dict[str, Any]]):
        i, j = ij
        return xcat._run_job(apis[i % workers], j)

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(run, enumerate(jobs)):
            with lock:
                for tk, got in res:
                    if got is None:
                        failed.append(tk)
                        continue
                    buf.extend(got)
                    fin.append(tk)
                n += 1
                if n % 250 == 0:
                    flush()
                    rate = n / (time.time() - t0)
                    _log(f"  {n}/{len(jobs)} requests, {rate:.2f} req/s, eta {(len(jobs) - n) / max(rate, 1e-9) / 60:.0f} min")
    flush()
    _log(f"fetch done: {len(fin)} fetched, {len(set(failed))} failed")


def load_candles(tickers: list[str] | None = None) -> pl.DataFrame:
    parts = sorted((CACHE / "candles").glob("part-*.parquet"))
    sc = pl.scan_parquet(parts).select(OHLC_COLS)
    if tickers is not None:
        sc = sc.filter(pl.col("ticker").is_in(tickers))
    return sc.filter(pl.col("period") == 60).unique(["ticker", "end_ts"], keep="last").sort("ticker", "end_ts").collect()


# ---- panel ---------------------------------------------------------------------------------------
WIDE_MARGIN = 0.12           # culture_flow / mentions: sell YES at ceil(mid + 12c)
WIDE_MAX_MID = 0.05          # ... on legs whose real two-sided mid is <= 5c
STAKE = 2.0                  # $ per order, taker and maker: n = max(1, floor(STAKE / price)) contracts
LADDER_TYPES = ("greater", "less", "greater_or_equal", "less_or_equal", "functional")


def me_flags(s: pl.DataFrame) -> pl.DataFrame:
    """Mutually-exclusive (one-winner) flag per sampled event, known at listing: the event's own
    ``/events`` flag cached by ``series_age``, else the series' modal flag there, else a structural
    guess (bracket ``between`` legs, or >= 3 custom legs outside Mentions)."""
    p = Path("data/cache/series_age/events.parquet")
    ev = s.group_by("event_ticker").agg(pl.col("series_ticker").first(), pl.col("group").first(),
                                        pl.col("strike_type").drop_nulls().mode().first().alias("st"),
                                        pl.col("n_legs").first())
    if p.exists():
        f = pl.read_parquet(p).select("event_ticker", "series_ticker", "mutually_exclusive")
        ev = ev.join(f.select("event_ticker", pl.col("mutually_exclusive").alias("me_ev")), on="event_ticker", how="left")
        sm = f.group_by("series_ticker").agg(pl.col("mutually_exclusive").mean().alias("me_share"))
        ev = ev.join(sm, on="series_ticker", how="left")
    else:
        ev = ev.with_columns(pl.lit(None, pl.Boolean).alias("me_ev"), pl.lit(None, pl.Float64).alias("me_share"))
    guess = (pl.col("st") == "between") | (pl.col("st").is_in(["custom", "structured"]) & (pl.col("n_legs") >= 3)
                                          & (pl.col("group") != "Mentions"))
    return ev.select("event_ticker", pl.coalesce("me_ev", pl.col("me_share") >= 0.5, guess).fill_null(False).alias("me"),
                     pl.when(pl.col("me_ev").is_not_null()).then(pl.lit("event"))
                     .when(pl.col("me_share").is_not_null()).then(pl.lit("series")).otherwise(pl.lit("guess")).alias("me_src"))


def series_history(s: pl.DataFrame, d: pl.DataFrame) -> pl.DataFrame:
    """Per decision row: the series' events settled before t (frame, all events, not just the sample):
    count, legs, YES legs; and the same over the 30 days before t."""
    fe = (pl.read_parquet(OUT / "frame_events.parquet")
          .filter(pl.col("series_ticker").is_in(s["series_ticker"].unique().implode()) & pl.col("settle_max").is_not_null())
          .select("series_ticker", "settle_max", "n_legs", "n_yes").sort("settle_max"))
    fe = fe.with_columns(pl.int_range(1, pl.len() + 1).over("series_ticker").alias("c_ev"),
                         pl.col("n_legs").cum_sum().over("series_ticker").alias("c_legs"),
                         pl.col("n_yes").cum_sum().over("series_ticker").alias("c_yes")).drop("n_legs", "n_yes")
    first = pl.read_parquet(OUT / "frame_events.parquet").group_by("series_ticker").agg(
        pl.col("open_min").min().alias("series_first_open"))
    out = d.select("rid", "series_ticker", "t").sort("t")
    a = out.join_asof(fe.rename({"settle_max": "ts_a"}), left_on="t", right_on="ts_a", by="series_ticker",
                      strategy="backward", check_sortedness=False)
    b = out.with_columns((pl.col("t") - timedelta(days=30)).alias("t30")).sort("t30").join_asof(
        fe.rename({"settle_max": "ts_b", "c_ev": "b_ev", "c_legs": "b_legs", "c_yes": "b_yes"}),
        left_on="t30", right_on="ts_b", by="series_ticker", strategy="backward", check_sortedness=False)
    r = a.join(b.select("rid", "b_ev", "b_legs", "b_yes"), on="rid").with_columns(
        [pl.col(c).fill_null(0) for c in ("c_ev", "c_legs", "c_yes", "b_ev", "b_legs", "b_yes")])
    launch_p = Path("data/cache/series_age/launch.parquet")
    r = r.join(first, on="series_ticker", how="left")
    if launch_p.exists():
        r = r.join(pl.read_parquet(launch_p).select("series_ticker", "launch"), on="series_ticker", how="left")
    else:
        r = r.with_columns(pl.lit(None, pl.Datetime("us", "UTC")).alias("launch"))
    return r.select(
        "rid", pl.col("c_ev").alias("s_prev_events"), pl.col("c_legs").alias("s_prev_legs"),
        ((pl.col("c_yes") + 0.5) / (pl.col("c_legs") + 2)).alias("s_yes_rate"),
        (pl.col("c_ev") - pl.col("b_ev")).alias("s_events_30d"),
        ((pl.col("c_yes") - pl.col("b_yes") + 0.5) / (pl.col("c_legs") - pl.col("b_legs") + 2)).alias("s_yes_rate_30d"),
        ((pl.col("t") - pl.min_horizontal(pl.coalesce("launch", "series_first_open"), pl.col("series_first_open")))
         .dt.total_hours() / 24).clip(lower_bound=0).alias("series_age_d"))


def _asof(d: pl.DataFrame, c: pl.DataFrame, shift_h: float, cols: list[str], suffix: str) -> pl.DataFrame:
    left = d.select("rid", "ticker", (pl.col("t") - timedelta(hours=shift_h)).alias("tq")).sort("tq")
    j = left.join_asof(c.select("ticker", "end_ts", *cols).sort("end_ts"), left_on="tq", right_on="end_ts",
                       by="ticker", strategy="backward", check_sortedness=False)
    return j.select("rid", *[pl.col(x).alias(f"{x}{suffix}") for x in cols + ["end_ts"]])


def panel_chunk(s: pl.DataFrame, c: pl.DataFrame) -> pl.DataFrame:
    """Decision rows of one chunk of sampled markets: quote, features, maker fills, target."""
    s = s.filter(pl.col("T").is_not_null())
    c = c.join(s.select("ticker", "close_time", "T"), on="ticker").filter(
        (pl.col("end_ts") < pl.col("close_time")) & (pl.col("end_ts") <= pl.col("T"))).sort("ticker", "end_ts")
    c = c.with_columns(
        pl.col("bid_close").fill_null(0.0).alias("bid"), pl.col("ask_close").fill_null(1.0).alias("ask"),
        pl.col("volume").fill_null(0.0).cum_sum().over("ticker").alias("cumvol"),
        pl.int_range(1, pl.len() + 1).over("ticker").alias("ncandle"),
        pl.when(pl.col("volume") > 0).then(pl.col("end_ts")).otherwise(None).forward_fill().over("ticker").alias("last_trade_ts"),
        pl.when(pl.col("volume") > 0).then(pl.col("p_close")).otherwise(None).forward_fill().over("ticker").alias("last_px"))
    c = c.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    d = (s.select("ticker", "T", "open_time", "close_time").join(pl.DataFrame({"lead": list(LEADS_H)}), how="cross")
         .with_columns((pl.col("T") - pl.duration(hours=pl.col("lead"))).alias("t"))
         .filter((pl.col("open_time") < pl.col("t")) & (pl.col("close_time") > pl.col("t")))
         .with_columns(pl.concat_str("ticker", pl.lit("|"), pl.col("lead").cast(pl.String)).alias("rid")))
    q0 = _asof(d, c, 0, ["bid", "ask", "mid", "oi", "cumvol", "ncandle", "last_trade_ts", "last_px", "p_close"], "")
    q1 = _asof(d, c, 1, ["mid", "cumvol"], "_1h")
    q6 = _asof(d, c, 6, ["mid", "cumvol", "ncandle"], "_6h")
    q24 = _asof(d, c, 24, ["mid", "cumvol", "ncandle"], "_24h")
    d = d.join(q0, on="rid").join(q1, on="rid").join(q6, on="rid").join(q24, on="rid")
    d = d.filter(pl.col("end_ts").is_not_null() & (pl.col("ask") > pl.col("bid")))
    hrs = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_minutes() / 60
    d = d.with_columns(
        (pl.col("ask") - pl.col("bid")).alias("spread"),
        (pl.col("bid") > 0).alias("has_bid"), (pl.col("ask") < 1).alias("has_ask"),
        (pl.col("mid") - pl.col("mid_1h").fill_null(pl.col("mid"))).alias("dmid_1h"),
        (pl.col("mid") - pl.col("mid_6h").fill_null(pl.col("mid"))).alias("dmid_6h"),
        (pl.col("mid") - pl.col("mid_24h").fill_null(pl.col("mid"))).alias("dmid_24h"),
        (pl.col("cumvol") - pl.col("cumvol_1h").fill_null(0)).alias("vol_1h"),
        (pl.col("cumvol") - pl.col("cumvol_6h").fill_null(0)).alias("vol_6h"),
        (pl.col("cumvol") - pl.col("cumvol_24h").fill_null(0)).alias("vol_24h"),
        pl.col("cumvol").alias("vol_window"),
        (pl.col("ncandle") - pl.col("ncandle_24h").fill_null(0)).alias("active_h_24"),
        hrs("t", "end_ts").alias("quote_age_h"),
        hrs("t", "last_trade_ts").fill_null(24 * 6 + 1).alias("trade_age_h"),
        hrs("t", "open_time").alias("listed_h"),
        (pl.col("mid") - pl.col("last_px")).fill_null(0).alias("mid_minus_last"),
        pl.col("t").dt.convert_time_zone("America/New_York").dt.hour().alias("hour_et"),
        (pl.col("t").dt.convert_time_zone("America/New_York").dt.weekday() >= 6).alias("weekend"))
    keep = ["rid", "ticker", "lead", "t", "bid", "ask", "mid", "spread", "has_bid", "has_ask", "oi", "dmid_1h", "dmid_6h",
            "dmid_24h", "vol_1h", "vol_6h", "vol_24h", "vol_window", "active_h_24", "quote_age_h", "trade_age_h",
            "listed_h", "mid_minus_last", "hour_et", "weekend"]
    d = d.select(keep)
    return d.join(maker_fills(d, c, s), on="rid", how="left")


def maker_fills(d: pl.DataFrame, c: pl.DataFrame, s: pl.DataFrame) -> pl.DataFrame:
    """Resting 1-tick-improved bids posted at t on each side, alive until min(T, close). A fill needs a
    later hourly candle whose trade prints went strictly THROUGH our price (YES bid: a trade below it;
    NO bid: a YES trade above 1 - our price) with at least our size traded in that hour."""
    inside = (pl.col("ask") - pl.col("bid")) >= 0.02
    o = d.select("rid", "ticker", "t", "bid", "ask").with_columns(
        pl.when(inside).then(pl.col("bid") + 0.01).otherwise(pl.col("bid")).round(2).alias("mk_yes"),
        pl.when(inside).then(1 - pl.col("ask") + 0.01).otherwise(1 - pl.col("ask")).round(2).alias("mk_no"))
    o = o.with_columns([pl.max_horizontal(pl.lit(1.0), (STAKE / pl.col(x).clip(lower_bound=0.01)).floor()).alias(f"n_{x}")
                        for x in ("mk_yes", "mk_no")])
    o = o.with_columns((((pl.col("bid") + pl.col("ask")) / 2 + WIDE_MARGIN) * 100 - 1e-9).ceil().truediv(100).alias("wide_lvl"))
    later = c.select("ticker", pl.col("end_ts").alias("tf"), "p_low", "p_high", pl.col("volume").fill_null(0).alias("vf"))
    e = o.join(later, on="ticker").filter(pl.col("tf") > pl.col("t"))
    yes_hit = (pl.col("mk_yes") >= 0.01) & (pl.col("p_low") < pl.col("mk_yes") - 1e-9) & (pl.col("vf") >= pl.col("n_mk_yes"))
    no_hit = (pl.col("mk_no") >= 0.01) & (pl.col("p_high") > 1 - pl.col("mk_no") + 1e-9) & (pl.col("vf") >= pl.col("n_mk_no"))
    fy = e.filter(yes_hit).group_by("rid").agg(pl.col("tf").min().alias("fill_yes_ts"))
    fn = e.filter(no_hit).group_by("rid").agg(pl.col("tf").min().alias("fill_no_ts"))
    # the culture_flow / mentions mechanism: a 1-lot NO bid far from the touch, selling YES at
    # ceil(mid + WIDE_MARGIN); filled by a later YES print strictly above that level
    fw = e.filter(pl.col("p_high") > pl.col("wide_lvl") + 1e-9).group_by("rid").agg(pl.col("tf").min().alias("fill_wide_ts"))
    return (o.select("rid", "mk_yes", "mk_no", "wide_lvl").join(fy, on="rid", how="left")
            .join(fn, on="rid", how="left").join(fw, on="rid", how="left"))


def event_features(p: pl.DataFrame) -> pl.DataFrame:
    """Leg rank, sum of mids and share within the event at the same lead (sampled legs quoted then)."""
    g = ["event_ticker", "lead"]
    return p.with_columns(
        pl.len().over(g).alias("ev_quoted"),
        pl.col("mid").sum().over(g).alias("ev_sum_mid"),
        pl.col("mid").max().over(g).alias("ev_max_mid"),
        pl.col("mid").rank("ordinal", descending=True).over(g).alias("ev_rank")).with_columns(
        ((pl.col("ev_rank") - 1) / (pl.col("ev_quoted") - 1).clip(lower_bound=1)).alias("ev_rank_frac"),
        (pl.col("mid") / pl.col("ev_sum_mid").clip(lower_bound=1e-6)).alias("ev_share"),
        (pl.col("mid") >= pl.col("ev_max_mid") - 1e-12).alias("ev_is_fav"))


def build_panel(chunks: int = 8) -> pl.DataFrame:
    s = pl.read_parquet(OUT / "sample.parquet")
    me = me_flags(s)
    s = s.join(me, on="event_ticker", how="left")
    frames = []
    for i in range(chunks):
        sub = s.filter(pl.col("event_ticker").hash(seed=3) % chunks == i)
        c = load_candles(sub["ticker"].to_list())
        frames.append(panel_chunk(sub, c))
        _log(f"panel chunk {i}: {frames[-1].height} rows from {c.height} candles")
        del c
    p = pl.concat(frames)
    meta = s.select("ticker", "event_ticker", "series_ticker", "group", "category", "month", "T", "open_time",
                    "close_time", "settle", "y", "fee_type", "fee_multiplier", "strike_type", "me", "me_src",
                    "n_legs", "p_event", "can_close_early")
    p = p.join(meta, on="ticker", how="left")
    p = event_features(p)
    p = p.join(series_history(s, p.select("rid", "series_ticker", "t")), on="rid", how="left")
    p = p.with_columns(
        pl.col("t").dt.strftime("%Y-%m").alias("t_month"),
        pl.when(pl.col("t") >= HOLDOUT_FROM).then(pl.lit("holdout")).otherwise(pl.lit("dev")).alias("split"),
        pl.col("strike_type").is_in(list(LADDER_TYPES)).fill_null(False).alias("ladder"))
    _save(p, OUT / "panel.parquet")
    cov = p.group_by("group", "split", "lead").agg(pl.len().alias("rows"), pl.col("ticker").n_unique().alias("markets"),
                                                   pl.col("event_ticker").n_unique().alias("events"),
                                                   (pl.col("has_bid") & pl.col("has_ask")).mean().alias("two_sided"),
                                                   (pl.col("has_bid") | pl.col("has_ask")).mean().alias("any_side"),
                                                   pl.col("spread").median().alias("med_spread")).sort("group", "split", "lead")
    cov.write_csv(OUT / "panel_coverage.csv")
    _log(str(cov))
    return p


# ---- features ------------------------------------------------------------------------------------
GROUPS = ("Sports", "Weather", "Politics", "Entertainment", "Mentions", "Science & Tech", "Other")
EDGES = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98, 1.0)   # flb_common buckets
BASE_FEATURES = [
    "lm", "spread", "has_bid", "has_ask", "dmid_1h", "dmid_6h", "dmid_24h", "lvol_1h", "lvol_24h", "lvol_window",
    "loi", "active_h_24", "lquote_age", "ltrade_age", "llisted", "mid_minus_last", "lseries_age", "ls_prev_events",
    "s_yes_rate", "s_yes_rate_30d", "ls_events_30d", "me", "ladder", "me_share", "me_rank_frac", "me_overround",
    "me_is_fav", "lev_quoted", "round5", "hour_sin", "hour_cos", "weekend", "maker_fee_series", "s_resid", "ls_resid_n",
    "lead72", "lead24", *[f"g_{g}" for g in GROUPS[1:]]]
INTERACT = ["spread", "has_bid", "has_ask", "lvol_24h", "ltrade_age", "me", "ladder", "lead72", "lead24", "s_resid",
            *[f"g_{g}" for g in GROUPS[1:]]]


def _logit(x: pl.Expr) -> pl.Expr:
    x = x.clip(0.01, 0.99)
    return (x / (1 - x)).log()


def add_features(p: pl.DataFrame) -> pl.DataFrame:
    l1p = lambda c: pl.col(c).fill_null(0).clip(lower_bound=0).log1p()
    me = pl.col("me").fill_null(False)
    c100 = pl.col("mid") * 100
    r5 = (c100 - (c100 / 5).round(0) * 5).abs()
    return p.with_columns(
        _logit(pl.col("mid")).alias("lm"),
        pl.col("has_bid").cast(pl.Float64), pl.col("has_ask").cast(pl.Float64),
        l1p("vol_1h").alias("lvol_1h"), l1p("vol_24h").alias("lvol_24h"), l1p("vol_window").alias("lvol_window"),
        l1p("oi").alias("loi"), l1p("quote_age_h").alias("lquote_age"), l1p("trade_age_h").alias("ltrade_age"),
        l1p("listed_h").alias("llisted"), l1p("series_age_d").alias("lseries_age"),
        l1p("s_prev_events").alias("ls_prev_events"), l1p("s_events_30d").alias("ls_events_30d"),
        me.cast(pl.Float64).alias("me"), pl.col("ladder").cast(pl.Float64),
        pl.when(me).then(pl.col("ev_share")).otherwise(0.0).alias("me_share"),
        pl.when(me).then(pl.col("ev_rank_frac")).otherwise(0.0).alias("me_rank_frac"),
        pl.when(me).then(pl.col("ev_sum_mid") - 1).otherwise(0.0).clip(-1, 3).alias("me_overround"),
        pl.when(me).then(pl.col("ev_is_fav").cast(pl.Float64)).otherwise(0.0).alias("me_is_fav"),
        pl.col("ev_quoted").cast(pl.Float64).log().alias("lev_quoted"),
        (r5 / 2.5).alias("round5"),
        (pl.col("hour_et") * 2 * math.pi / 24).sin().alias("hour_sin"),
        (pl.col("hour_et") * 2 * math.pi / 24).cos().alias("hour_cos"),
        pl.col("weekend").cast(pl.Float64),
        pl.col("fee_type").fill_null("").str.contains("maker_fee").cast(pl.Float64).alias("maker_fee_series"),
        (pl.col("lead") == 72).cast(pl.Float64).alias("lead72"), (pl.col("lead") == 24).cast(pl.Float64).alias("lead24"),
        *[(pl.col("group") == g).cast(pl.Float64).alias(f"g_{g}") for g in GROUPS[1:]],
        pl.col("dmid_1h").fill_null(0), pl.col("dmid_6h").fill_null(0), pl.col("dmid_24h").fill_null(0),
        pl.col("mid_minus_last").fill_null(0).clip(-0.5, 0.5), pl.col("s_yes_rate").fill_null(0.3),
        pl.col("s_yes_rate_30d").fill_null(0.3))


def series_resid_encoding(p: pl.DataFrame, k: float = 30.0) -> pl.DataFrame:
    """Target encoding of each series' past calibration error, per decision month: from rows that
    SETTLED before the month started, s_resid = sum(y - mid) / (n + k) (shrunk to 0). Joined on
    (series, t_month) so training and test rows see only earlier outcomes."""
    months = sorted(p["t_month"].unique().to_list())
    frames = []
    for m in months:
        start = datetime.strptime(m + "-01", "%Y-%m-%d").replace(tzinfo=UTC)
        q = p.filter(pl.col("settle") < start).group_by("series_ticker").agg(
            (pl.col("y") - pl.col("mid")).sum().alias("rs"), pl.len().alias("n"))
        frames.append(q.with_columns(pl.lit(m).alias("t_month")))
    if not frames:
        return pl.DataFrame(schema={"series_ticker": pl.String, "t_month": pl.String, "s_resid": pl.Float64,
                                    "ls_resid_n": pl.Float64})
    e = pl.concat(frames)
    return e.select("series_ticker", "t_month", (pl.col("rs") / (pl.col("n") + k)).alias("s_resid"),
                    pl.col("n").cast(pl.Float64).log1p().alias("ls_resid_n"))


def model_frame() -> pl.DataFrame:
    """Panel rows with at least one real side quoted (an empty book, bid 0 / ask 1, has no price to
    predict against or trade at), features and the series target encoding."""
    p = pl.read_parquet(OUT / "panel.parquet").filter(pl.col("has_bid") | pl.col("has_ask"))
    p = add_features(p)
    enc = series_resid_encoding(p)
    p = p.join(enc, on=["series_ticker", "t_month"], how="left").with_columns(
        pl.col("s_resid").fill_null(0.0), pl.col("ls_resid_n").fill_null(0.0))
    return p


# ---- models (numpy only) -------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logit(X: np.ndarray, y: np.ndarray, lam: float, offset: np.ndarray | None = None,
              pen: np.ndarray | None = None, iters: int = 50) -> np.ndarray:
    """L2-penalised logistic regression by Newton / IRLS. ``pen`` = per-coefficient penalty
    multipliers (0 for the intercept)."""
    n, k = X.shape
    off = np.zeros(n) if offset is None else offset
    pen = np.ones(k) if pen is None else pen
    b = np.zeros(k)
    for _ in range(iters):
        mu = sigmoid(X @ b + off)
        w = np.clip(mu * (1 - mu), 1e-6, None)
        g = X.T @ (y - mu) - lam * pen * b
        H = (X * w[:, None]).T @ X + lam * np.diag(pen) + 1e-9 * np.eye(k)
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-7:
            break
    return b


def bin_edges(x: np.ndarray, nbins: int = 32) -> np.ndarray:
    qs = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)[1:-1]))
    return qs


def binize(X: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.searchsorted(e, X[:, j], side="right") for j, e in enumerate(edges)], axis=1).astype(np.int32)


def _best_split(Xb: np.ndarray, g: np.ndarray, hs: np.ndarray, idx: np.ndarray, nb: int, lam: float,
                min_leaf: int) -> tuple[float, int, int]:
    G, H, n = g[idx].sum(), hs[idx].sum(), len(idx)
    base = G * G / (H + lam)
    best = (0.0, -1, -1)
    for j in range(Xb.shape[1]):
        b = Xb[idx, j]
        gs = np.bincount(b, weights=g[idx], minlength=nb)
        hb = np.bincount(b, weights=hs[idx], minlength=nb)
        cb = np.bincount(b, minlength=nb)
        gl, hl, cl = np.cumsum(gs)[:-1], np.cumsum(hb)[:-1], np.cumsum(cb)[:-1]
        ok = (cl >= min_leaf) & (n - cl >= min_leaf)
        if not ok.any():
            continue
        gain = gl ** 2 / (hl + lam) + (G - gl) ** 2 / (H - hl + lam) - base
        gain = np.where(ok, gain, -np.inf)
        k = int(np.argmax(gain))
        if gain[k] > best[0]:
            best = (float(gain[k]), j, k)
    return best


class GBM:
    """Gradient-boosted depth-``depth`` trees on binned features, logistic loss, Newton leaves, from
    an offset (the recalibrated price). Small learning rate, row subsampling, min leaf size."""

    def __init__(self, rounds: int = 200, lr: float = 0.05, depth: int = 2, min_leaf: int = 300,
                 lam: float = 50.0, subsample: float = 0.7, nbins: int = 32, seed: int = 0) -> None:
        self.rounds, self.lr, self.depth, self.min_leaf, self.lam = rounds, lr, depth, min_leaf, lam
        self.subsample, self.nbins, self.seed = subsample, nbins, seed
        self.trees: list[list[tuple]] = []
        self.gain: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, offset: np.ndarray) -> GBM:
        rng = np.random.default_rng(self.seed)
        self.edges = [bin_edges(X[:, j], self.nbins) for j in range(X.shape[1])]
        Xb = binize(X, self.edges)
        nb = self.nbins + 1
        f = offset.copy()
        self.gain = np.zeros(X.shape[1])
        for _ in range(self.rounds):
            p = sigmoid(f)
            g, hs = y - p, p * (1 - p)
            rows = np.flatnonzero(rng.random(len(y)) < self.subsample)
            nodes = [rows]
            splits: list[tuple] = []
            for _d in range(self.depth):
                nxt = []
                for idx in nodes:
                    gain, j, k = _best_split(Xb, g, hs, idx, nb, self.lam, self.min_leaf) if len(idx) else (0, -1, -1)
                    if j < 0:
                        splits.append((-1, -1))
                        nxt.extend([idx, idx[:0]])
                        continue
                    self.gain[j] += gain
                    splits.append((j, k))
                    m = Xb[idx, j] <= k
                    nxt.extend([idx[m], idx[~m]])
                nodes = nxt
            leaves = [float(g[i].sum() / (hs[i].sum() + self.lam)) if len(i) else 0.0 for i in nodes]
            tree = (splits, leaves)
            self.trees.append(tree)
            f += self.lr * self._tree_pred(tree, Xb)
        return self

    def _tree_pred(self, tree: tuple, Xb: np.ndarray) -> np.ndarray:
        splits, leaves = tree
        node = np.zeros(len(Xb), dtype=np.int64)       # index within the current level
        pos = 0
        for d in range(self.depth):
            width = 2 ** d
            new = np.zeros_like(node)
            for i in range(width):
                j, k = splits[pos + i]
                m = node == i
                if j < 0:
                    new[m] = 2 * i
                else:
                    new[m] = 2 * i + (Xb[m, j] > k)
            node = new
            pos += width
        return np.asarray(leaves)[node]

    def predict_logit(self, X: np.ndarray, offset: np.ndarray, rounds: int | None = None) -> np.ndarray:
        Xb = binize(X, self.edges)
        f = offset.copy()
        for tree in self.trees[: rounds or len(self.trees)]:
            f += self.lr * self._tree_pred(tree, Xb)
        return f


def _design(df: pl.DataFrame, mu: np.ndarray | None = None, sd: np.ndarray | None = None,
            interact: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X = df.select(BASE_FEATURES).fill_null(0.0).to_numpy().astype(float)
    names = list(BASE_FEATURES)
    if mu is None:
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (X - mu) / sd
    if interact:
        lm = Z[:, names.index("lm")]
        Z = np.column_stack([Z] + [lm * Z[:, names.index(c)] for c in INTERACT])
        names = names + [f"lm*{c}" for c in INTERACT]
    return np.column_stack([np.ones(len(Z)), Z]), mu, sd, ["const"] + names


def bucket_baseline(train: pl.DataFrame, test: pl.DataFrame, k: float = 50.0) -> np.ndarray:
    """Naive FLB curve: mid + shrunk mean (y - mid) of the training rows in the same price bucket x
    group (shrunk to the bucket's pooled residual, itself shrunk to 0)."""
    def bucket(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("mid").cut(list(EDGES[1:-1]), left_closed=False).cast(pl.String).alias("bk"))
    tr, te = bucket(train), bucket(test)
    pooled = tr.group_by("bk").agg(((pl.col("y") - pl.col("mid")).sum() / (pl.len() + k)).alias("rb"))
    cell = tr.group_by("bk", "group").agg((pl.col("y") - pl.col("mid")).sum().alias("rs"), pl.len().alias("n"))
    te = te.join(pooled, on="bk", how="left").join(cell, on=["bk", "group"], how="left").with_columns(
        pl.col("rb").fill_null(0.0), pl.col("rs").fill_null(0.0), pl.col("n").fill_null(0))
    r = (te["rs"] + k * te["rb"]) / (te["n"] + k)
    return np.clip(te["mid"].to_numpy() + r.to_numpy(), 0.002, 0.998)


def llo_design(df: pl.DataFrame) -> np.ndarray:
    lm = df["lm"].to_numpy()
    cols = [np.ones(len(df)), lm]
    for g in GROUPS[1:]:
        d = (df["group"] == g).to_numpy().astype(float)
        cols += [d, d * lm]
    return np.column_stack(cols)


LLO_PEN = np.array([0.0, 0.0] + [1.0] * (2 * (len(GROUPS) - 1)))


def fit_fold(train: pl.DataFrame, test: pl.DataFrame, lams: tuple[float, ...], rounds: tuple[int, ...],
             gbm_kw: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """All models on one walk-forward fold. Returns test predictions and diagnostics."""
    ytr = train["y"].to_numpy().astype(float)
    preds: dict[str, np.ndarray] = {"mid": np.clip(test["mid"].to_numpy(), 0.002, 0.998)}
    preds["bucket"] = bucket_baseline(train, test)
    b_llo = fit_logit(llo_design(train), ytr, 30.0, pen=LLO_PEN)
    preds["llo"] = sigmoid(llo_design(test) @ b_llo)
    diag: dict[str, Any] = {"llo_coef": b_llo.tolist()}
    # L2 logistic on every feature (and logit(mid) interactions) on top of the LLO recalibration offset:
    # the penalty shrinks toward the recalibrated price, not toward the base rate
    Xtr, mu, sd, names = _design(train)
    Xte = _design(test, mu, sd)[0]
    off_tr, off_te = llo_design(train) @ b_llo, llo_design(test) @ b_llo
    pen = np.ones(Xtr.shape[1])
    pen[0] = 0.0
    diag["logit_names"] = names
    for lam in lams:
        b = fit_logit(Xtr, ytr, lam, offset=off_tr, pen=pen)
        preds[f"logit_l{lam:g}"] = sigmoid(Xte @ b + off_te)
        diag[f"logit_l{lam:g}_coef"] = b.tolist()
    # boosting from the recalibrated price (LLO) on the base features (unstandardised; trees are invariant)
    Btr = train.select(BASE_FEATURES).fill_null(0.0).to_numpy().astype(float)
    Bte = test.select(BASE_FEATURES).fill_null(0.0).to_numpy().astype(float)
    gbm = GBM(rounds=max(rounds), **(gbm_kw or {})).fit(Btr, ytr, off_tr)
    for r in rounds:
        preds[f"gbm_r{r}"] = sigmoid(gbm.predict_logit(Bte, off_te, rounds=r))
    diag["gbm_gain"] = dict(zip(BASE_FEATURES, gbm.gain.tolist(), strict=True))
    return preds, diag


# ---- walk-forward --------------------------------------------------------------------------------
DEV_MONTHS = ("2026-06", "2026-07", "2026-08", "2026-09")
HOLDOUT_MONTHS = ("2026-10", "2026-11")
LAMS = (10.0, 100.0, 1000.0)
ROUNDS = (50, 150, 300)


def month_start(m: str) -> datetime:
    return datetime.strptime(m + "-01", "%Y-%m-%d").replace(tzinfo=UTC)


def walk_forward(p: pl.DataFrame, months: tuple[str, ...], lams: tuple[float, ...], rounds: tuple[int, ...]
                 ) -> tuple[pl.DataFrame, dict[str, Any]]:
    outs, diags = [], {}
    for m in months:
        st = month_start(m)
        train = p.filter((pl.col("t") < st) & (pl.col("settle") < st))
        test = p.filter(pl.col("t_month") == m)
        if not test.height or train.height < 500:
            _log(f"fold {m}: skipped (train {train.height}, test {test.height})")
            continue
        t0 = time.time()
        preds, diag = fit_fold(train, test, lams, rounds)
        diag["n_train"], diag["n_test"] = train.height, test.height
        diags[m] = diag
        outs.append(test.select("rid").with_columns(pl.lit(m).alias("fold"),
                                                    *[pl.Series(k, v) for k, v in preds.items()]))
        _log(f"fold {m}: train {train.height} test {test.height} ({time.time() - t0:.0f}s)")
    return pl.concat(outs), diags


def logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def cluster_t_mean(x: np.ndarray, g: np.ndarray) -> tuple[float, float | None]:
    from albert.research.fundamental import clustered_mean

    mu, t, _ = clustered_mean(np.asarray(x, dtype=float), np.asarray(g))
    return mu, t


def score_models(oos: pl.DataFrame, p: pl.DataFrame, models: list[str]) -> pl.DataFrame:
    """OOS log loss / Brier per model, and the per-row gain over the mid and over the FLB bucket
    baseline with event-clustered t."""
    d = oos.join(p.select("rid", "y", "event_ticker", "group", "lead", "mid"), on="rid")
    y, ev = d["y"].to_numpy().astype(float), d["event_ticker"].to_numpy()
    ll_mid, ll_bk = logloss(y, d["mid"].to_numpy()), logloss(y, d["bucket"].to_numpy())
    rows = []
    for m in models:
        pm = d[m].to_numpy()
        ll = logloss(y, pm)
        g1, t1 = cluster_t_mean(ll_mid - ll, ev)
        g2, t2 = cluster_t_mean(ll_bk - ll, ev)
        rows.append({"model": m, "n": len(y), "events": len(np.unique(ev)), "logloss": float(ll.mean()),
                     "brier": float(((pm - y) ** 2).mean()), "gain_vs_mid_mll": 1000 * g1, "t_vs_mid": t1,
                     "gain_vs_bucket_mll": 1000 * g2, "t_vs_bucket": t2})
    return pl.DataFrame(rows)


# ---- trading rules -------------------------------------------------------------------------------
THETAS = (0.0, 0.02, 0.04, 0.07, 0.10)
RULE_MODELS = ("bucket", "llo", "logit", "gbm")
BOOK_CAP = 50.0              # $ of open cost (mentions uses ~$40 of the $100)
EVENT_CAP = 6.0              # $ of open cost per event
MIN_TRADES, MIN_EVENTS = 40, 15


def _n(price: pl.Expr) -> pl.Expr:
    return pl.max_horizontal(pl.lit(1.0), (STAKE / price.clip(lower_bound=0.01)).floor())


def _fee_order(n: pl.Expr, price: pl.Expr, maker: bool) -> pl.Expr:
    """Kalshi fee for one order, rounded up to the cent (taker 0.07, maker 0.0175 on maker-fee series)."""
    mult = pl.col("fee_multiplier").fill_null(1.0)
    if maker:
        on = pl.col("fee_type").fill_null("").str.contains("maker_fee")
        raw = 0.0175 * mult * n * price * (1 - price)
        return pl.when(on).then((raw * 100).round(9).ceil() / 100).otherwise(0.0)
    return ((0.07 * mult * n * price * (1 - price)) * 100).round(9).ceil() / 100


def side_candidates(d: pl.DataFrame, phat: str, exec_: str) -> pl.DataFrame:
    """Both sides of every row with price, size, fee, model edge; long format."""
    maker = exec_ == "maker"
    yes_px = pl.col("mk_yes") if maker else pl.col("ask")
    no_px = pl.col("mk_no") if maker else 1 - pl.col("bid")
    yes_ok = (pl.col("mk_yes") >= 0.01) & (pl.col("mk_yes") < pl.col("ask")) if maker else (
        (pl.col("ask") < 1) & (pl.col("ask") >= 0.02) & (pl.col("ask") <= 0.98))
    no_ok = (pl.col("mk_no") >= 0.01) & (pl.col("mk_no") < 1 - pl.col("bid")) if maker else (
        (pl.col("bid") > 0) & (pl.col("bid") >= 0.02) & (pl.col("bid") <= 0.98))
    base = ["rid", "ticker", "event_ticker", "series_ticker", "group", "lead", "t", "settle", "y", "fee_type",
            "fee_multiplier", "p_event", "mid"]
    fill = {"yes": "fill_yes_ts", "no": "fill_no_ts"}
    out = []
    for side, px, ok, ph in (("yes", yes_px, yes_ok, pl.col(phat)), ("no", no_px, no_ok, 1 - pl.col(phat))):
        x = d.filter(ok).select(*base, px.round(4).alias("price"), ph.alias("p_side"),
                                (pl.col(fill[side]) if maker else pl.col("t")).alias("t_fill"),
                                pl.lit(side).alias("side"))
        x = x.with_columns(_n(pl.col("price")).alias("n"))
        x = x.with_columns(_fee_order(pl.col("n"), pl.col("price"), maker).alias("fee"))
        x = x.with_columns((pl.col("p_side") - pl.col("price") - pl.col("fee") / pl.col("n")).alias("edge"))
        out.append(x)
    c = pl.concat(out)
    return c.with_columns(
        pl.when(pl.col("side") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y")).cast(pl.Float64).alias("payoff"))


def cell_trades(cand: pl.DataFrame, theta: float, lead: int) -> pl.DataFrame:
    """Best side per row if its edge >= theta, at one lead; filled orders only (taker: all)."""
    x = cand.filter((pl.col("lead") == lead) & (pl.col("edge") >= theta)).sort("edge", descending=True).unique(
        "rid", keep="first")
    posted = x.height
    x = x.filter(pl.col("t_fill").is_not_null()).with_columns(
        (pl.col("payoff") - pl.col("price") - pl.col("fee") / pl.col("n")).alias("pnl_c"),
        (pl.col("price") + pl.col("fee") / pl.col("n")).alias("risk_c"))
    return x.with_columns(pl.lit(posted).alias("posted"))


def ratio_t(num: np.ndarray, den: np.ndarray, g: np.ndarray) -> tuple[float, float | None]:
    """Sum(num) / sum(den) with an event-clustered (linearised) standard error."""
    R = num.sum() / den.sum()
    u = num - R * den
    _, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=u)
    k = len(s)
    if k < 2:
        return float(R), None
    se = math.sqrt((s ** 2).sum() * k / (k - 1)) / den.sum()
    return float(R), (float(R / se) if se > 0 else None)


def cell_stats(x: pl.DataFrame) -> dict[str, Any]:
    if not x.height:
        return {"trades": 0, "events": 0}
    n = x["n"].to_numpy()
    pnl, risk, ev = x["pnl_c"].to_numpy() * n, x["risk_c"].to_numpy() * n, x["event_ticker"].to_numpy()
    r, t = ratio_t(pnl, risk, ev)
    c, tc = cluster_t_mean(x["pnl_c"].to_numpy(), ev)
    gross = (x["payoff"] - x["price"]).mean()
    return {"trades": x.height, "events": len(np.unique(ev)), "posted": int(x["posted"][0]),
            "fill_rate": round(x.height / max(int(x["posted"][0]), 1), 3),
            "avg_price": round(float(x["price"].mean()), 3), "avg_edge_c": round(100 * float(x["edge"].mean()), 2),
            "gross_c": round(100 * float(gross), 2), "net_c": round(100 * c, 2), "t_c": None if tc is None else round(tc, 2),
            "pnl_per_$": round(r, 4), "t_$": None if t is None else round(t, 2), "pnl_usd": round(float(pnl.sum()), 2),
            "yes_share": round(float((x["side"] == "yes").mean()), 2), "win": round(float(x["payoff"].mean()), 3),
            # population-equivalent trade count: each sampled event stands for 1 / p_event events of its cell
            "pop_trades": round(float((1 / x["p_event"].fill_null(1.0).clip(lower_bound=1e-3)).sum()), 0)}


def holm(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    order = np.argsort(pvals)
    m = len(pvals)
    passed = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            passed[i] = True
        else:
            break
    return passed


def _p_one_sided(t: float | None) -> float:
    return 1.0 if t is None else 0.5 * math.erfc(t / math.sqrt(2))


def all_cells(oos: pl.DataFrame, p: pl.DataFrame, model_cols: dict[str, str]) -> tuple[pl.DataFrame, dict]:
    d = oos.join(p.select("rid", "ticker", "event_ticker", "series_ticker", "group", "lead", "t", "settle", "y",
                          "fee_type", "fee_multiplier", "p_event", "mid", "bid", "ask", "mk_yes", "mk_no",
                          "fill_yes_ts", "fill_no_ts", "wide_lvl", "fill_wide_ts"), on="rid")
    rows, trades = [], {}
    for exec_ in ("taker", "maker"):
        for mname, col in model_cols.items():
            cand = side_candidates(d, col, exec_)
            for lead in LEADS_H:
                for th in THETAS:
                    x = cell_trades(cand, th, lead)
                    key = f"{exec_}|{mname}|L{lead}|th{round(th * 100)}"
                    trades[key] = x
                    rows.append({"cell": key, "exec": exec_, "model": mname, "lead": lead, "theta_c": th * 100,
                                 **cell_stats(x)})
    wide_rows, wide_trades = wide_cells(d, model_cols)
    rows += wide_rows
    trades.update(wide_trades)
    return pl.DataFrame(rows, infer_schema_length=None), trades


def wide_cells(d: pl.DataFrame, model_cols: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, pl.DataFrame]]:
    """Does the model improve the one maker mechanism that works (culture_flow / mentions)? A 1-lot NO
    bid at 1 - ceil(mid + 12c) on every leg with a real two-sided mid <= 5c (control: no gate), or only
    where the model agrees the leg is overpriced (p_model <= mid) / strongly so (p_model <= mid / 2)."""
    base = d.filter((pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("ask") > pl.col("bid"))
                    & ((pl.col("bid") + pl.col("ask")) / 2 <= WIDE_MAX_MID))
    base = base.with_columns((1 - pl.col("wide_lvl")).round(2).alias("price"), pl.lit(1.0).alias("n"),
                             pl.lit("no").alias("side"), pl.col("fill_wide_ts").alias("t_fill"),
                             (1 - pl.col("y")).cast(pl.Float64).alias("payoff"))
    base = base.with_columns(_fee_order(pl.col("n"), pl.col("price"), True).alias("fee"))
    gates: list[tuple[str, pl.Expr]] = [("none", pl.lit(True))]
    for mname, col in model_cols.items():
        gates += [(f"{mname}_le_mid", pl.col(col) <= pl.col("mid")), (f"{mname}_le_halfmid", pl.col(col) <= pl.col("mid") / 2)]
    rows, trades = [], {}
    for lead in LEADS_H:
        for gname, g in gates:
            x = base.filter((pl.col("lead") == lead) & g)
            posted = x.height
            x = x.filter(pl.col("t_fill").is_not_null()).with_columns(
                (pl.col("payoff") - pl.col("price") - pl.col("fee") / pl.col("n")).alias("pnl_c"),
                (pl.col("price") + pl.col("fee") / pl.col("n")).alias("risk_c"),
                ((1 - pl.col("price")) - pl.col("mid")).alias("edge"), pl.lit(posted).alias("posted"))
            key = f"wide|{gname}|L{lead}"
            trades[key] = x
            rows.append({"cell": key, "exec": "wide", "model": gname, "lead": lead, "theta_c": 100 * WIDE_MARGIN,
                         **cell_stats(x)})
    return rows, trades


# ---- capped book + vol_metrics -------------------------------------------------------------------
def capped_ledger(x: pl.DataFrame, name: str, split: str) -> pl.DataFrame:
    """Chronological capped book: $BOOK_CAP of open cost, $EVENT_CAP per event, positions held to
    settlement; ledger rows in the ``vol_metrics`` schema."""
    x = x.sort("t_fill")
    open_pos: list[tuple[datetime, float, str]] = []
    keep = []
    for r in x.iter_rows(named=True):
        now = r["t_fill"]
        open_pos = [o for o in open_pos if o[0] > now]
        cost = r["n"] * r["price"] + r["fee"]
        tot = sum(o[1] for o in open_pos)
        evc = sum(o[1] for o in open_pos if o[2] == r["event_ticker"])
        if tot + cost > BOOK_CAP or evc + cost > EVENT_CAP:
            continue
        open_pos.append((r["settle"], cost, r["event_ticker"]))
        keep.append({"strategy": "xs_ml", "variant": name, "split": split, "struct_id": r["rid"],
                     "event_ticker": r["event_ticker"], "ticker": r["ticker"], "side": r["side"], "n": int(r["n"]),
                     "t_in": now, "cost_usd": cost, "settle_ts": r["settle"], "payoff_usd": r["n"] * r["payoff"]})
    return pl.DataFrame(keep, infer_schema_length=None)


def ledger_stats(led: pl.DataFrame) -> dict[str, Any]:
    from albert.research import vol_metrics as vm

    if not led.height:
        return {}
    marks = vm.load_marks(led["ticker"].unique().to_list())
    m = vm.ledger_metrics(led, marks)
    keys = ("positions", "pnl_usd", "usd_per_week", "ann_vol_pct", "sharpe", "sharpe_ci90", "p_sharpe_pos",
            "sharpe_ex_top5", "max_dd_usd", "weeks_positive_pct", "days")
    out = {k: m.get(k) for k in keys}
    out["positions"] = led.height
    out["events"] = led["event_ticker"].n_unique()
    return out


# ---- stages --------------------------------------------------------------------------------------
def _json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str))


def importance_tables(diags: dict[str, Any], lam: float) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Logistic standardised coefficients and GBM gain shares per refit, with stability measures."""
    months = sorted(diags)
    names = diags[months[0]]["logit_names"]
    C = np.array([diags[m][f"logit_l{lam:g}_coef"] for m in months])          # folds x coefs
    lg = pl.DataFrame({"feature": names, **{m: C[i] for i, m in enumerate(months)}})
    mean = C.mean(axis=0)
    sign = (np.sign(C) == np.sign(mean)[None, :]).mean(axis=0)
    lg = lg.with_columns(pl.Series("mean", mean), pl.Series("sd", C.std(axis=0)), pl.Series("sign_agree", sign),
                         pl.Series("abs_mean", np.abs(mean))).sort("abs_mean", descending=True)
    G = np.array([[diags[m]["gbm_gain"][f] for f in BASE_FEATURES] for m in months])
    S = G / np.clip(G.sum(axis=1, keepdims=True), 1e-12, None)
    gb = pl.DataFrame({"feature": BASE_FEATURES, **{m: S[i] for i, m in enumerate(months)}}).with_columns(
        pl.Series("mean_share", S.mean(axis=0))).sort("mean_share", descending=True)

    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    stab = {"logit_rank_corr_consecutive": [round(spearman(np.abs(C[i]), np.abs(C[i + 1])), 3) for i in range(len(months) - 1)],
            "gbm_rank_corr_consecutive": [round(spearman(S[i], S[i + 1]), 3) for i in range(len(months) - 1)],
            "months": months}
    return lg, gb, stab


def calibration_table(oos: pl.DataFrame, p: pl.DataFrame, model: str) -> pl.DataFrame:
    """Realised y - mid against the model's predicted p - mid, by price bucket (event-clustered t)."""
    d = oos.join(p.select("rid", "y", "mid", "event_ticker"), on="rid").with_columns(
        pl.col("mid").cut(list(EDGES[1:-1]), left_closed=False).cast(pl.String).alias("bk"))
    rows = []
    for (bk,), g in d.group_by(["bk"], maintain_order=False):
        r = (g["y"] - g["mid"]).to_numpy().astype(float)
        mu, t = cluster_t_mean(r, g["event_ticker"].to_numpy())
        rows.append({"bucket": bk, "rows": g.height, "events": g["event_ticker"].n_unique(),
                     "mid": round(float(g["mid"].mean()), 4), "realised_minus_mid_c": round(100 * mu, 2),
                     "t": None if t is None else round(t, 2),
                     "bucket_pred_c": round(100 * float((g["bucket"] - g["mid"]).mean()), 2),
                     f"{model}_pred_c": round(100 * float((g[model] - g["mid"]).mean()), 2)})
    return pl.DataFrame(rows).sort("mid")


def edge_deciles(oos: pl.DataFrame, p: pl.DataFrame, model: str) -> pl.DataFrame:
    """Does the model's predicted edge over the mid come true? Deciles of (p_model - mid)."""
    d = oos.join(p.select("rid", "y", "mid", "event_ticker"), on="rid").with_columns(
        (pl.col(model) - pl.col("mid")).alias("pe"), (pl.col("bucket") - pl.col("mid")).alias("pb"))
    d = d.with_columns(pl.col("pe").qcut(10, labels=[str(i) for i in range(10)], allow_duplicates=True).alias("dec"))
    rows = []
    for (dec,), g in d.group_by(["dec"]):
        r = (g["y"] - g["mid"]).to_numpy().astype(float)
        mu, t = cluster_t_mean(r, g["event_ticker"].to_numpy())
        rows.append({"decile": int(dec), "rows": g.height, "pred_c": round(100 * float(g["pe"].mean()), 2),
                     "bucket_pred_c": round(100 * float(g["pb"].mean()), 2), "realised_c": round(100 * mu, 2),
                     "t": None if t is None else round(t, 2), "mid": round(float(g["mid"].mean()), 3)})
    return pl.DataFrame(rows).sort("decile")


def run_dev() -> None:
    """Walk-forward on the dev months, model selection by OOS log loss, the full rule grid, Holm,
    and the frozen rules (written before any holdout month is fitted)."""
    p = model_frame()
    oos, diags = walk_forward(p, DEV_MONTHS, LAMS, ROUNDS)
    _save(oos, OUT / "oos_dev.parquet")
    _json(diags, OUT / "diags_dev.json")
    cand = ["mid", "bucket", "llo"] + [f"logit_l{x:g}" for x in LAMS] + [f"gbm_r{r}" for r in ROUNDS]
    sc = score_models(oos, p, cand)
    sc.write_csv(OUT / "model_scores_dev.csv")
    _log(str(sc))
    best_lam = min(LAMS, key=lambda x: sc.filter(pl.col("model") == f"logit_l{x:g}")["logloss"][0])
    best_r = min(ROUNDS, key=lambda r: sc.filter(pl.col("model") == f"gbm_r{r}")["logloss"][0])
    cols = {"bucket": "bucket", "llo": "llo", "logit": f"logit_l{best_lam:g}", "gbm": f"gbm_r{best_r}"}
    # per-month and per-group model scores (chosen models)
    byg = []
    for (g,), sub in oos.join(p.select("rid", "group"), on="rid").group_by(["group"]):
        s2 = score_models(sub.drop("group"), p, list(cols.values()))
        byg.append(s2.with_columns(pl.lit(g).alias("group")))
    pl.concat(byg).sort("group", "model").write_csv(OUT / "model_scores_dev_by_group.csv")
    bym = []
    for (m,), sub in oos.group_by(["fold"]):
        bym.append(score_models(sub, p, list(cols.values())).with_columns(pl.lit(m).alias("fold")))
    pl.concat(bym).sort("fold", "model").write_csv(OUT / "model_scores_dev_by_month.csv")
    lg, gb, stab = importance_tables(diags, best_lam)
    lg.write_csv(OUT / "importance_logit_dev.csv")
    gb.write_csv(OUT / "importance_gbm_dev.csv")
    _json(stab, OUT / "importance_stability_dev.json")
    calibration_table(oos, p, cols["gbm"]).write_csv(OUT / "calibration_dev.csv")
    edge_deciles(oos, p, cols["gbm"]).write_csv(OUT / "edge_deciles_gbm_dev.csv")
    edge_deciles(oos, p, cols["logit"]).write_csv(OUT / "edge_deciles_logit_dev.csv")
    cells, trades = all_cells(oos, p, cols)
    elig = (pl.col("trades") >= MIN_TRADES) & (pl.col("events") >= MIN_EVENTS) & pl.col("t_$").is_not_null()
    cells = cells.with_columns(elig.fill_null(False).alias("eligible"))
    e = cells.filter(pl.col("eligible"))
    pv = np.array([_p_one_sided(t) for t in e["t_$"].to_list()])
    passed = holm(pv) if len(pv) else np.array([], dtype=bool)
    cells = cells.join(e.select("cell").with_columns(pl.Series("holm_pass", passed)), on="cell", how="left")
    cells.write_csv(OUT / "dev_cells.csv")
    n_el = e.height
    bar = None
    if n_el:
        from scipy.stats import norm
        bar = float(norm.isf(0.05 / n_el))
    frozen: dict[str, Any] = {"frozen_at": datetime.now(UTC).isoformat(), "lam": best_lam, "rounds": best_r,
                              "model_cols": cols, "n_cells": cells.height, "n_eligible": n_el,
                              "holm_first_bar_t": bar, "holm_passed": int(passed.sum()) if len(passed) else 0,
                              "rules": []}
    for exec_ in ("taker", "maker", "wide"):
        sub = cells.filter(pl.col("eligible") & (pl.col("exec") == exec_)).sort("t_$", descending=True)
        if sub.height:
            frozen["rules"].append(sub.row(0, named=True))
    # the matching FLB-baseline cell of each frozen rule, as the "what does the model add" control
    ctrl = []
    for r in frozen["rules"]:
        k = (f"wide|none|L{r['lead']}" if r["exec"] == "wide"
             else f"{r['exec']}|bucket|L{r['lead']}|th{round(r['theta_c'])}")
        ctrl.append(cells.filter(pl.col("cell") == k).row(0, named=True) if cells.filter(pl.col("cell") == k).height else {})
    frozen["controls"] = ctrl
    led_stats = {}
    for r in frozen["rules"]:
        led = capped_ledger(trades[r["cell"]], r["cell"].replace("|", "_"), "dev")
        (OUT / "ledgers").mkdir(exist_ok=True)
        if led.height:
            led.write_parquet(OUT / "ledgers" / f"ledger_{r['cell'].replace('|', '_')}_dev.parquet")
        led_stats[r["cell"]] = ledger_stats(led)
    frozen["dev_ledgers"] = led_stats
    _json(frozen, OUT / "frozen.json")
    _log(json.dumps(frozen, indent=1, default=str))


def run_holdout() -> None:
    """Refit monthly for Oct and Nov with the frozen hyperparameters and score the frozen rules once."""
    fz = json.loads((OUT / "frozen.json").read_text())
    if (OUT / "holdout.json").exists():
        raise SystemExit("holdout already scored (holdout.json exists): it is scored once")
    p = model_frame()
    lam, r = fz["lam"], fz["rounds"]
    oos, diags = walk_forward(p, HOLDOUT_MONTHS, (lam,), (r,))
    _save(oos, OUT / "oos_holdout.parquet")
    _json(diags, OUT / "diags_holdout.json")
    cols = fz["model_cols"]
    sc = score_models(oos, p, ["mid", "bucket", "llo", cols["logit"], cols["gbm"]])
    sc.write_csv(OUT / "model_scores_holdout.csv")
    byg = []
    for (g,), sub in oos.join(p.select("rid", "group"), on="rid").group_by(["group"]):
        byg.append(score_models(sub.drop("group"), p, ["llo", cols["logit"], cols["gbm"]]).with_columns(pl.lit(g).alias("group")))
    pl.concat(byg).sort("group", "model").write_csv(OUT / "model_scores_holdout_by_group.csv")
    edge_deciles(oos, p, cols["gbm"]).write_csv(OUT / "edge_deciles_gbm_holdout.csv")
    calibration_table(oos, p, cols["gbm"]).write_csv(OUT / "calibration_holdout.csv")
    cells, trades = all_cells(oos, p, cols)
    cells.write_csv(OUT / "holdout_cells_all.csv")      # descriptive only; the verdict uses the frozen cells
    out: dict[str, Any] = {"scored_at": datetime.now(UTC).isoformat(), "rules": [], "controls": []}
    for r0 in fz["rules"]:
        row = cells.filter(pl.col("cell") == r0["cell"]).row(0, named=True)
        led = capped_ledger(trades[r0["cell"]], r0["cell"].replace("|", "_"), "holdout")
        if led.height:
            led.write_parquet(OUT / "ledgers" / f"ledger_{r0['cell'].replace('|', '_')}_holdout.parquet")
        row["ledger"] = ledger_stats(led)
        by_g = []
        x = trades[r0["cell"]]
        for (g,), sub in x.group_by(["group"]):
            by_g.append({"group": g, **cell_stats(sub)})
        row["by_group"] = by_g
        out["rules"].append(row)
    for c0 in fz.get("controls", []):
        if c0:
            out["controls"].append(cells.filter(pl.col("cell") == c0["cell"]).row(0, named=True))
    _json(out, OUT / "holdout.json")
    _log(json.dumps(out, indent=1, default=str))


# ---- report --------------------------------------------------------------------------------------
def _md(df: pl.DataFrame, n: int | None = None) -> str:
    if n is not None:
        df = df.head(n)
    cols = df.columns

    def f(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4g}" if abs(v) < 1e5 else f"{v:.0f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.iter_rows()]
    return "\n".join(lines)


def _csv(name: str) -> pl.DataFrame | None:
    p = OUT / name
    return pl.read_csv(p) if p.exists() else None


def write_report() -> Path:
    """report.md = verdict.md (written after the holdout) + hypotheses + every table the stages wrote."""
    parts = [(OUT / "verdict.md").read_text() if (OUT / "verdict.md").exists() else "# xs_ml\n\n(verdict pending)\n"]
    parts.append((OUT / "hypotheses.md").read_text().replace("# xs_ml hypotheses", "## Hypotheses"))
    sec: list[tuple[str, str, int | None]] = [
        ("Sample cells (group x month of T; volume-blind event sample)", "sample_cells.csv", None),
        ("Panel coverage", "panel_coverage.csv", None),
        ("Dev OOS model scores (Jun-Sep, monthly refits; gain in milli-log-loss per row, t by event)",
         "model_scores_dev.csv", None),
        ("Dev OOS model scores by month", "model_scores_dev_by_month.csv", None),
        ("Dev OOS model scores by group", "model_scores_dev_by_group.csv", None),
        ("Dev calibration by price bucket (realised y - mid vs predictions)", "calibration_dev.csv", None),
        ("Dev: does the GBM's predicted edge over the mid come true? (deciles)", "edge_deciles_gbm_dev.csv", None),
        ("Dev: same for the logistic model", "edge_deciles_logit_dev.csv", None),
        ("Feature importance, GBM gain share per refit (dev)", "importance_gbm_dev.csv", 20),
        ("Feature importance, logistic standardised coefficients per refit (dev)", "importance_logit_dev.csv", 25),
        ("Holdout OOS model scores (Oct, Nov; frozen hyperparameters)", "model_scores_holdout.csv", None),
        ("Holdout OOS model scores by group", "model_scores_holdout_by_group.csv", None),
        ("Holdout calibration by price bucket", "calibration_holdout.csv", None),
        ("Holdout: GBM edge deciles", "edge_deciles_gbm_holdout.csv", None),
    ]
    for title, f, n in sec:
        df = _csv(f)
        if df is not None:
            parts.append(f"## {title}\n\n{_md(df, n)}\n")
    stab = OUT / "importance_stability_dev.json"
    if stab.exists():
        parts.append(f"## Importance stability (Spearman of importance ranks between consecutive refits)\n\n"
                     f"```\n{stab.read_text()}\n```\n")
    cells = _csv("dev_cells.csv")
    if cells is not None:
        keep = ["cell", "trades", "events", "fill_rate", "avg_price", "avg_edge_c", "gross_c", "net_c", "pnl_per_$",
                "t_$", "yes_share", "eligible", "holm_pass"]
        top = cells.filter(pl.col("eligible")).sort("t_$", descending=True)
        parts.append(f"## Dev rule grid: {cells.height} cells ({top.height} eligible: >= {MIN_TRADES} trades and "
                     f">= {MIN_EVENTS} events), top 25 by event-clustered t of P&L per $\n\n"
                     f"{_md(top.select(keep), 25)}\n")
        for ex in ("taker", "maker"):
            sub = cells.filter(pl.col("eligible") & (pl.col("exec") == ex))
            if sub.height:
                parts.append(f"* {ex}: {sub.height} eligible cells, mean net {sub['net_c'].mean():.2f}c, "
                             f"{(sub['t_$'] >= 2).sum()} with t >= 2, {(sub['t_$'] <= -2).sum()} with t <= -2.\n")
    for name in ("frozen.json", "holdout.json"):
        p = OUT / name
        if p.exists():
            parts.append(f"## {name}\n\n```json\n{p.read_text()}\n```\n")
    hc = _csv("holdout_cells_all.csv")
    if hc is not None:
        parts.append("## Holdout, every cell (descriptive only; the verdict uses the frozen cells)\n\n"
                     + _md(hc.select([c for c in ("cell", "trades", "events", "net_c", "pnl_per_$", "t_$") if c in hc.columns])
                           .sort("t_$", descending=True, nulls_last=True), 20) + "\n")
    out = OUT / "report.md"
    out.write_text("\n\n".join(parts))
    return out


# ---- CLI -----------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m albert.research.xs_ml")
    ap.add_argument("cmd", choices=["scan", "frame", "sample", "fetch", "panel", "prep", "model", "dev", "holdout",
                                        "report"])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "scan":
        scan()
    elif a.cmd == "frame":
        build_frame()
    elif a.cmd == "sample":
        draw_sample()
    elif a.cmd == "fetch":
        fetch()
    elif a.cmd == "panel":
        build_panel()
    elif a.cmd == "prep":           # scan + frame + sample + fetch + panel in one queued job
        import subprocess
        import sys

        for stage in ("scan", "frame", "sample", "fetch", "panel"):   # fresh process (heap) per stage
            r = subprocess.run([sys.executable, "-m", "albert.research.xs_ml", stage], check=False)
            if r.returncode:
                raise SystemExit(f"prep: stage {stage} failed ({r.returncode})")
    elif a.cmd in ("model", "dev"):
        run_dev()
    elif a.cmd == "holdout":
        run_holdout()
    else:
        write_report()


if __name__ == "__main__":
    main()
