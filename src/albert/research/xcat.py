"""Cross-category candle samples with full OHLC (volume-blind), for momentum and resolution spikes.

Two layers fetched from Kalshi's public candlestick endpoints (every response cached raw under
``data/raw/kalshi``, append-only):

* ``hourly``: each market's whole life ``[open, close]`` (capped at 5,000 hours from open), for every
  Politics, Science and Technology, Elections and Mentions market and a seeded random set of
  Entertainment events, closes 2026-08-15..10-31 (the shared ``kalshi_markets`` window).
* ``minute``: each market's first three days ``[open, min(close, open + 3 d)]`` for seeded random
  events: sports (game winners, spreads/totals, other/props strata), weather, mentions,
  entertainment, politics. The window is fixed by the *open* time, so it never depends on when (or
  whether) the market resolved early: a rule evaluated inside it cannot pick spikes by their outcome.

Nothing is selected on volume or on the result. Markets settled before the API's historical cutoff
(2026-09-18 at fetch time) come from ``/historical/markets/{t}/candlesticks`` one by one; later ones
from the live batch endpoint (at most 10,000 requested candles summed over tickers per request).
Candles are sparse: the API omits periods in which nothing changed.

    python -m albert.research.xcat sample | fetch hourly | fetch minute | status
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from albert.venues.kalshi import _field, num, parse_ts

OUT = Path("data/research/xcat")
TABLES = Path("data/tables")
SEED = 20261117
MINUTE_DAYS = 3
MAX_PERIODS = 5000            # per market per request (API limit)
MAX_BATCH_CANDLES = 10000     # summed over tickers in one batch request (API limit)
MAX_BATCH_TICKERS = 50
NON_SPORTS_FULL = ("Politics", "Science and Technology", "Elections", "Mentions")
SAMPLE_EVENTS = {             # minute layer: seeded random events per stratum
    "sports winner": 600, "sports spread/total": 300, "sports other": 300,
    "weather": 500, "mentions": 200, "entertainment": 120, "politics": 120,
}
HOURLY_ENTERTAINMENT_EVENTS = 450

OHLC_SCHEMA = {
    "ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "period": pl.Int64,
    "p_open": pl.Float64, "p_high": pl.Float64, "p_low": pl.Float64, "p_close": pl.Float64, "p_mean": pl.Float64,
    "bid_open": pl.Float64, "bid_high": pl.Float64, "bid_low": pl.Float64, "bid_close": pl.Float64,
    "ask_open": pl.Float64, "ask_high": pl.Float64, "ask_low": pl.Float64, "ask_close": pl.Float64,
    "volume": pl.Float64, "oi": pl.Float64,
}


# ---- parsing -------------------------------------------------------------------------------------
def ohlc_rows(payload: dict[str, Any], ticker: str, period: int) -> list[dict[str, Any]]:
    """Rows of one market's ``candlesticks`` (live ``*_dollars``/``*_fp`` or historical plain keys)."""
    rows = []
    for c in payload.get("candlesticks") or []:
        p, b, a = c.get("price") or {}, c.get("yes_bid") or {}, c.get("yes_ask") or {}
        rows.append({
            "ticker": ticker, "end_ts": parse_ts(c.get("end_period_ts")), "period": period,
            **{f"p_{k}": num(_field(p, k)) for k in ("open", "high", "low", "close", "mean")},
            **{f"bid_{k}": num(_field(b, k)) for k in ("open", "high", "low", "close")},
            **{f"ask_{k}": num(_field(a, k)) for k in ("open", "high", "low", "close")},
            "volume": num(_field(c, "volume")), "oi": num(_field(c, "open_interest")),
        })
    return rows


def batch_rows(payload: dict[str, Any], period: int) -> dict[str, list[dict[str, Any]]]:
    return {m["market_ticker"]: ohlc_rows(m, m["market_ticker"], period) for m in payload.get("markets") or []}


# ---- sample --------------------------------------------------------------------------------------
def sports_group(series: pl.Expr) -> pl.Expr:
    return (pl.when(series.str.contains(r"(GAME|MATCH|MAP|FIGHT|BOUT|RACE)$")).then(pl.lit("sports winner"))
            .when(series.str.contains("SPREAD|TOTAL")).then(pl.lit("sports spread/total"))
            .otherwise(pl.lit("sports other")))


def stratum(category: pl.Expr, series: pl.Expr) -> pl.Expr:
    return (pl.when(category == "Sports").then(sports_group(series))
            .when(category == "Climate and Weather").then(pl.lit("weather"))
            .otherwise(category.str.to_lowercase()))


def _events(m: pl.DataFrame, n: int, seed: int) -> list[str]:
    ev = m["event_ticker"].unique().sort()
    return ev.sample(min(n, ev.len()), seed=seed).to_list()


def build_sample(tables: Path = TABLES) -> pl.DataFrame:
    """One row per (market, layer) with the fetch window. Seeded; never looks at volume or result."""
    m = (pl.read_parquet(tables / "kalshi_markets.parquet")
         .filter(pl.col("result").is_in(["yes", "no"]) & (pl.col("market_type") == "binary")
                 & pl.col("open_time").is_not_null() & (pl.col("close_time") > pl.col("open_time")))
         .with_columns(stratum(pl.col("category"), pl.col("series_ticker")).alias("stratum")))
    frames = []
    # hourly: whole life
    full = m.filter(pl.col("category").is_in(NON_SPORTS_FULL))
    ent = m.filter(pl.col("category") == "Entertainment")
    ent = ent.filter(pl.col("event_ticker").is_in(_events(ent, HOURLY_ENTERTAINMENT_EVENTS, SEED)))
    h = pl.concat([full, ent]).with_columns(
        pl.lit("hourly").alias("layer"), pl.lit(60).alias("period"), pl.col("open_time").alias("lo"),
        pl.min_horizontal("close_time", pl.col("open_time") + timedelta(hours=MAX_PERIODS - 1)).alias("hi"))
    frames.append(h)
    # minute: first three days
    for i, (name, n) in enumerate(SAMPLE_EVENTS.items()):
        sub = m.filter(pl.col("stratum") == name)
        sub = sub.filter(pl.col("event_ticker").is_in(_events(sub, n, SEED + i)))
        frames.append(sub.with_columns(
            pl.lit("minute").alias("layer"), pl.lit(1).alias("period"), pl.col("open_time").alias("lo"),
            pl.min_horizontal("close_time", pl.col("open_time") + timedelta(days=MINUTE_DAYS)).alias("hi")))
    out = pl.concat(frames, how="diagonal_relaxed")
    return out.with_columns(
        ((pl.col("hi") - pl.col("lo")).dt.total_minutes() / pl.col("period")).ceil().cast(pl.Int64).add(1).alias("periods"))


# ---- fetch ---------------------------------------------------------------------------------------
def _jobs(sample: pl.DataFrame, cutoff: datetime) -> list[dict[str, Any]]:
    """Live markets grouped per event into batch requests within the API caps; historical ones alone."""
    jobs: list[dict[str, Any]] = []
    live = sample.filter(pl.col("settlement_ts") >= cutoff)
    hist = sample.filter(pl.col("settlement_ts") < cutoff)
    for r in hist.iter_rows(named=True):
        jobs.append({"kind": "hist", "rows": [r]})
    for _, g in live.sort("event_ticker", "ticker").group_by("event_ticker", maintain_order=True):
        rows = g.to_dicts()
        cur: list[dict[str, Any]] = []
        for r in rows:
            lo = min([x["lo"] for x in cur] + [r["lo"]])
            hi = max([x["hi"] for x in cur] + [r["hi"]])
            span = math.ceil((hi - lo).total_seconds() / 60 / r["period"]) + 1
            if cur and (span * (len(cur) + 1) > MAX_BATCH_CANDLES or len(cur) >= MAX_BATCH_TICKERS):
                jobs.append({"kind": "batch", "rows": cur})
                cur = []
            cur.append(r)
        if cur:
            jobs.append({"kind": "batch", "rows": cur})
    return jobs


def _run_job(api, job: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]] | None]]:
    rows = job["rows"]
    per = rows[0]["period"]
    if job["kind"] == "batch" and len(rows) > 1:
        lo, hi = min(r["lo"] for r in rows), max(r["hi"] for r in rows)
        try:
            got = batch_rows(api.candles_batch([r["ticker"] for r in rows], lo, hi, per).payload, per)
            # a market missing from the batch is retried alone (unknown tickers are silently omitted)
            out = [(r["ticker"], [x for x in got[r["ticker"]] if r["lo"] <= x["end_ts"] <= r["hi"] + timedelta(minutes=per)])
                   for r in rows if r["ticker"] in got]
            return out + [x for r in rows if r["ticker"] not in got for x in _single(api, r, False)]
        except RuntimeError as error:
            print(f"batch failed ({rows[0]['event_ticker']}): {str(error)[:160]}", flush=True)
    return [x for r in rows for x in _single(api, r, job["kind"] == "hist")]


def _single(api, r: dict[str, Any], historical: bool) -> list[tuple[str, list[dict[str, Any]] | None]]:
    """One market; a 404 on one endpoint (markets settled right at the cutoff) tries the other."""
    last = ""
    for hist in (historical, not historical):
        try:
            page = api.candles(r["ticker"], r["lo"], r["hi"], r["period"], hist)
            return [(r["ticker"], ohlc_rows(page.payload, r["ticker"], r["period"]))]
        except RuntimeError as error:
            last = str(error)
            if "404" not in last:
                break
    print(f"skip {r['ticker']}: {last[-220:]}", flush=True)
    return [(r["ticker"], None)]


def fetch(s, layer: str, *, workers: int = 4, min_interval: float = 0.08, save_every: int = 400) -> None:
    from albert.research.pm_hazard import _api

    d = OUT / layer
    d.mkdir(parents=True, exist_ok=True)
    sample = pl.read_parquet(OUT / "sample.parquet").filter(pl.col("layer") == layer)
    done_path, fail_path = d / "done.parquet", d / "failed.parquet"
    done = set(pl.read_parquet(done_path)["ticker"]) if done_path.exists() else set()
    todo = sample.filter(~pl.col("ticker").is_in(list(done)))
    api = _api(s, min_interval, nap_cap=5.0, retries=60)
    cutoff = api.cutoff()["market_settled_ts"]
    jobs = _jobs(todo, cutoff)
    print(f"{layer}: {sample.height} markets, {len(done)} done, {todo.height} to fetch in {len(jobs)} requests "
          f"(cutoff {cutoff:%Y-%m-%d})", flush=True)
    lock = threading.Lock()
    buf: list[dict[str, Any]] = []
    fin: list[str] = []
    failed: list[str] = []
    part = len(list(d.glob("part-*.parquet")))
    t0 = time.time()

    def flush() -> None:
        nonlocal buf, part
        if buf:
            pl.DataFrame(buf, schema=OHLC_SCHEMA).write_parquet(d / f"part-{part:05d}.parquet")
            part += 1
        buf = []
        pl.DataFrame({"ticker": sorted(done | set(fin))}).write_parquet(done_path)
        if failed:
            pl.DataFrame({"ticker": sorted(set(failed))}).write_parquet(fail_path)

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(lambda j: _run_job(api, j), jobs):
            with lock:
                for tk, got in res:
                    if got is None:
                        failed.append(tk)
                        continue
                    buf.extend(got)
                    fin.append(tk)
                n += 1
                if n % save_every == 0:
                    flush()
                    rate = n / (time.time() - t0)
                    print(f"  {layer} {n}/{len(jobs)} requests, {len(fin)} markets, {rate:.1f} req/s, "
                          f"eta {(len(jobs) - n) / max(rate, 1e-9) / 60:.0f} min", flush=True)
    flush()
    print(f"{layer} done: {len(fin)} fetched, {len(set(failed))} failed", flush=True)


def load(layer: str) -> pl.DataFrame:
    parts = sorted((OUT / layer).glob("part-*.parquet"))
    if not parts:
        raise RuntimeError(f"no {layer} candles fetched")
    return pl.concat([pl.read_parquet(p) for p in parts]).unique(["ticker", "end_ts"]).sort("ticker", "end_ts")


def status() -> dict[str, Any]:
    sample = pl.read_parquet(OUT / "sample.parquet")
    out = {}
    for layer in ("hourly", "minute"):
        sub = sample.filter(pl.col("layer") == layer)
        dp = OUT / layer / "done.parquet"
        done = pl.read_parquet(dp)["ticker"] if dp.exists() else pl.Series([], dtype=pl.String)
        out[layer] = {"markets": sub.height, "done": int(sub["ticker"].is_in(done.implode()).sum()),
                      "by_stratum": dict(sub.group_by("stratum").len().iter_rows())}
    return out


def main(argv: list[str] | None = None) -> None:
    from albert.config import settings

    ap = argparse.ArgumentParser(prog="albert.research.xcat")
    ap.add_argument("step", choices=["sample", "fetch", "status"])
    ap.add_argument("layer", nargs="?", choices=["hourly", "minute"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "sample":
        sm = build_sample()
        sm.write_parquet(OUT / "sample.parquet")
        print(json.dumps(sm.group_by("layer", "stratum").agg(pl.len().alias("markets"), pl.col("event_ticker").n_unique()
                                                             .alias("events")).sort("layer", "stratum").to_dicts(), default=str))
    elif a.step == "fetch":
        fetch(settings(), a.layer, workers=a.workers)
    else:
        print(json.dumps(status(), indent=1))


if __name__ == "__main__":
    main()
