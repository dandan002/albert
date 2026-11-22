"""Repetitive non-financial Kalshi series with an alternative-data angle (2026-11-20).

Slice: strait / canal traffic (KXHORMUZWEEKLY, KXHORMUZPEAK, KXHORMUZMAX, KXBABELMANDEBWEEKLY,
KXSUEZWEEKLY, KXPANAMAWEEKLY; settled on IMF PortWatch daily transit calls), the ActBlue top
fundraiser (KXACTBLUETOP; settled on secure.actblue.com/directory) and Trump Truth Social post
counts (KXTRUTHSOCIAL, KXTRUTHSOCIALD; settled on Roll Call, which blocks us, so price/structure
only), plus a universe scan of the other recurring Politics / World / Social / Transportation /
Health / Science series that the other agents of the 2026-11-20 swarm do not cover.

Data policy: Kalshi public API through the repo's cached client (<= 1 request/s, one thread);
PortWatch through the portwatch.imf.org download API (robots allows /api/; it redirects to
hub.arcgis.com /api/v3/, also allowed), one request per vintage with the 60 s crawl delay; the ActBlue
directory page (robots allows /directory); Wayback captures only of hosts that allow us. Every
alt-data value carries the time it became public (PortWatch: the weekly rebuild stamp
``x-amz-meta-contentlastmodified``; ActBlue: the capture or fetch time).

    python -m albert.research.recurring_alt robots
    python -m albert.research.recurring_alt portwatch          # fetch the current vintage (cached per vintage)
    python -m albert.research.recurring_alt discover           # settled + open markets of the slice
    python -m albert.research.recurring_alt candles            # hourly candles, whole market life
    python -m albert.research.recurring_alt universe           # recurring-series feasibility scan
    python -m albert.research.recurring_alt study dev|holdout  # grid on dev, frozen rule, holdout
    python -m albert.research.recurring_alt record             # one forward snapshot (cron-able)
    python -m albert.research.recurring_alt report

Nothing here places, amends or cancels an order.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from albert.venues.kalshi import _field, num, parse_ts

NAME = "recurring_alt"
OUT = Path("data/research/recurring_alt")
CACHE = Path("data/cache/recurring_alt")
FORWARD = OUT / "forward"
CANDLES = OUT / "candles"
UA = "albert-research/0.2 (public-data research)"
ET = ZoneInfo("America/New_York")

STRAIT = {  # series -> (PortWatch portid, kind)
    "KXHORMUZWEEKLY": ("chokepoint6", "sum"),
    "KXHORMUZPEAK": ("chokepoint6", "peak"),
    "KXHORMUZMAX": ("chokepoint6", "maxday"),
    "KXSUEZWEEKLY": ("chokepoint1", "sum"),
    "KXBABELMANDEBWEEKLY": ("chokepoint4", "sum"),
    "KXPANAMAWEEKLY": ("chokepoint2", "sum"),
}
TRUTH = ("KXTRUTHSOCIAL", "KXTRUTHSOCIALD")
ACTBLUE = ("KXACTBLUETOP",)
SLICE = (*STRAIT, *TRUTH, *ACTBLUE)

DEV_END = datetime(2026, 10, 1, tzinfo=UTC)          # dev: period ends <= 2026-09-30
TS_DAILY_SPLIT = datetime(2026, 11, 1, tzinfo=UTC)   # KXTRUTHSOCIALD starts 2026-10-02: own split
MIN_INTERVAL = 1.0

PW_ITEM = "3da2b9ca97684916b75c4013f95d18ab"           # ArcGIS item Daily_Chokepoints_Data (IMF-portwatch)
PW_URL = f"https://portwatch.imf.org/api/download/v1/items/{PW_ITEM}/csv"
PW_PARAMS = {"layers": "0"}
ACTBLUE_URL = "https://secure.actblue.com/directory"

ROBOTS_CHECKS: dict[str, list[str]] = {
    "portwatch.imf.org": [f"/api/download/v1/items/{PW_ITEM}/csv", "/pages/chokepoint6"],
    "hub.arcgis.com": [f"/api/v3/datasets/{PW_ITEM}_0/downloads/data"],
    "www.arcgis.com": [f"/sharing/rest/content/items/{PW_ITEM}"],
    "services9.arcgis.com": ["/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Chokepoints_Data/FeatureServer/0/query"],
    "secure.actblue.com": ["/directory"],
    "web.archive.org": ["/web/", "/cdx/search/cdx"],
    "api.open.fec.gov": ["/v1/schedules/schedule_a/"],
    "www.iea.org": ["/data-and-statistics/data-tools/middle-east-maritime-chokepoints-shipping-monitor"],
    "www.marinetraffic.com": ["/en/ais/home"],
    "tankertrackers.com": ["/"],
    "www.lloydslist.com": ["/"],
    "www.ukmto.org": ["/"],
    "www.windward.ai": ["/blog/"],
    "www.davemanuel.com": ["/shipping-chokepoints.php"],
    "rollcall.com": ["/factbase/trump/topic/social/"],
    "truthsocial.com": ["/@realDonaldTrump"],
    "data.ny.gov": ["/resource/"],
    "www.tsa.gov": ["/travel/passenger-volumes"],
    "data.cdc.gov": ["/resource/"],
    "www.federalregister.gov": ["/api/v1/documents.json"],
    "www.whitehouse.gov": ["/presidential-actions/"],
    "wikimedia.org": ["/api/rest_v1/metrics/pageviews/"],
    "www.faa.gov": ["/"],
    "www.nhc.noaa.gov": ["/"],
}


# ---- small helpers -----------------------------------------------------------------------------
def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _api(min_interval: float = MIN_INTERVAL):
    """Cached public Kalshi client (raw responses under data/raw/kalshi, append-only)."""
    from albert.config import Settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    s = Settings()
    return k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=min_interval, max_retries=12,
                                 sleep=lambda x: time.sleep(min(x, 20.0))))


def _pages(api, path: str, params: dict[str, Any], key: str, *, tag: str | None, cache: bool = True,
           max_pages: int = 100) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor = None
    for _ in range(max_pages):
        page = api.client.get(path, {**params, "cursor": cursor}, tag=tag, cache=cache)
        got = page.payload.get(key) or []
        out.extend(got)
        cursor = page.payload.get("cursor")
        if not cursor or not got:
            break
    return out


# ---- robots ----------------------------------------------------------------------------------------
def check_robots(out: Path = OUT) -> list[dict[str, Any]]:
    """Each host's own robots.txt, evaluated for our paths and the four agents (counts.robots_verdict).
    A robots.txt that does not answer 200/404/410 counts as *not* verified (we do not use the host)."""
    import httpx

    from albert.research.counts import AGENTS, robots_verdict

    rows = []
    for host, paths in ROBOTS_CHECKS.items():
        try:
            r = httpx.get(f"https://{host}/robots.txt", timeout=20, follow_redirects=True, headers={"User-Agent": UA})
            status, text = r.status_code, (r.text if r.status_code == 200 else "")
        except httpx.HTTPError:
            status, text = None, ""
        for p in paths:
            v = robots_verdict(text, p) if status in (200, 404, 410) else dict.fromkeys(AGENTS, False)
            rows.append({"host": host, "path": p, "http_status": status,
                         "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                         **{f"allow_{a}": v[a] for a in AGENTS}, "allowed": all(v.values())})
    out.mkdir(parents=True, exist_ok=True)
    (out / "robots.json").write_text(json.dumps(rows, indent=1))
    return rows


# ---- PortWatch -------------------------------------------------------------------------------------
def parse_portwatch(text: str) -> pl.DataFrame:
    """Daily chokepoint rows (date, portid, portname, n_total, n_tanker, n_cargo, ...)."""
    import io

    d = pl.read_csv(io.StringIO(text.lstrip("﻿")), infer_schema_length=10000)
    return (d.with_columns(pl.col("date").str.strptime(pl.Date, "%Y/%m/%d"))
            .select("date", "portid", "portname", "n_total", "n_tanker", "n_cargo", "n_container", "n_dry_bulk")
            .sort("portid", "date"))


def fetch_portwatch(cache: Path = CACHE / "portwatch", *, log=print) -> dict[str, Any]:
    """Download the current PortWatch vintage once. Each vintage is stored under its rebuild stamp
    (``x-amz-meta-contentlastmodified``) and never overwritten; returns its metadata."""
    import httpx

    cache.mkdir(parents=True, exist_ok=True)
    fetched = datetime.now(UTC)
    with httpx.Client(timeout=180, follow_redirects=True, headers={"User-Agent": UA}) as c:
        r = c.get(PW_URL, params=PW_PARAMS)
        r.raise_for_status()
    stamp = r.headers.get("x-amz-meta-contentlastmodified") or r.headers.get("last-modified") or fetched.isoformat()
    built = parse_ts(stamp) if "T" in stamp else datetime.strptime(stamp, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=UTC)
    key = built.strftime("%Y%m%dT%H%M%SZ")
    path = cache / f"daily_chokepoints_{key}.csv"
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(r.text)
        tmp.replace(path)
    d = parse_portwatch(r.text)
    meta = {"fetched_utc": fetched.isoformat(timespec="seconds"), "content_built_utc": built.isoformat(),
            "last_modified": r.headers.get("last-modified"), "final_url": str(r.url).split("?")[0],
            "rows": d.height, "max_date": str(d["date"].max()), "path": str(path)}
    with (cache / "vintages.jsonl").open("a") as f:
        f.write(json.dumps(meta) + "\n")
    log(f"portwatch vintage built {built:%Y-%m-%d %H:%M} UTC, {d.height} rows through {d['date'].max()}")
    return meta


def load_portwatch(cache: Path = CACHE / "portwatch") -> pl.DataFrame:
    """The latest stored vintage."""
    paths = sorted(cache.glob("daily_chokepoints_*.csv"))
    if not paths:
        raise RuntimeError("no PortWatch vintage; run `portwatch` first")
    return parse_portwatch(paths[-1].read_text())


def published_at(day: date) -> datetime:
    """When a PortWatch day became public: the weekly rebuild on the Tuesday after the Mon-Sun week
    that holds ``day``. Observed rebuild 2026-11-17 10:15 UTC (data through Sun 11-15); we use
    12:00 UTC Tuesday to leave a margin (see report: the price itself confirms the release hour)."""
    sunday = day + timedelta(days=6 - day.weekday())
    tue = sunday + timedelta(days=2)
    return datetime(tue.year, tue.month, tue.day, 12, 0, tzinfo=UTC)


# ---- Kalshi markets -----------------------------------------------------------------------------
MARKET_COLS = {"ticker": pl.String, "event_ticker": pl.String, "series_ticker": pl.String, "title": pl.String,
               "yes_sub_title": pl.String, "strike_type": pl.String, "floor_strike": pl.Float64,
               "cap_strike": pl.Float64, "open_time": pl.Datetime("us", "UTC"), "close_time": pl.Datetime("us", "UTC"),
               "expected_expiration_time": pl.Datetime("us", "UTC"), "settlement_ts": pl.Datetime("us", "UTC"),
               "result": pl.String, "settlement_value": pl.Float64, "expiration_value": pl.String,
               "volume": pl.Float64, "rules_primary": pl.String, "status": pl.String, "src": pl.String}


def market_row(m: dict[str, Any], series: str, src: str) -> dict[str, Any]:
    ev = m.get("expiration_value")
    return {"ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"), "series_ticker": series,
            "title": m.get("title"), "yes_sub_title": m.get("yes_sub_title"), "strike_type": m.get("strike_type"),
            "floor_strike": num(m.get("floor_strike")), "cap_strike": num(m.get("cap_strike")),
            "open_time": parse_ts(m.get("open_time")), "close_time": parse_ts(m.get("close_time")),
            "expected_expiration_time": parse_ts(m.get("expected_expiration_time")),
            "settlement_ts": parse_ts(m.get("settlement_ts")), "result": m.get("result") or None,
            "settlement_value": num(_field(m, "settlement_value")),
            "expiration_value": (str(ev) if ev not in (None, "") else None), "volume": num(_field(m, "volume")),
            "rules_primary": m.get("rules_primary"), "status": m.get("status"), "src": src}


def discover(series: tuple[str, ...] = SLICE, out: Path = OUT, *, log=print) -> pl.DataFrame:
    """Every settled market of the slice (historical + live endpoints), tagged with today's date in
    the cache key so a re-run the same day reuses the responses and a later day sees new ones."""
    api = _api()
    tag = datetime.now(UTC).strftime("%Y-%m-%d")
    rows: list[dict[str, Any]] = []
    for t in series:
        for m in _pages(api, "/historical/markets", {"series_ticker": t, "limit": 1000}, "markets", tag=tag):
            rows.append(market_row(m, t, "hist"))
        for m in _pages(api, "/markets", {"series_ticker": t, "status": "settled", "limit": 1000}, "markets", tag=tag):
            rows.append(market_row(m, t, "live"))
        log(f"  {t}: {len(rows)} markets so far", flush=True)
    mk = pl.DataFrame(rows, schema=MARKET_COLS).unique("ticker", keep="last")
    mk = mk.filter(pl.col("result").is_in(["yes", "no"]) | pl.col("settlement_value").is_not_null())
    _save(mk, out / "markets.parquet")
    log(f"discover: {mk.height} settled markets, {mk['event_ticker'].n_unique()} events")
    return mk


# ---- candles ----------------------------------------------------------------------------------------
def fetch_candles(out: Path = OUT, *, max_requests: int | None = None, log=print) -> None:
    """Hourly candles over each market's whole life (open -> close), for every settled slice market.
    Live markets go in batch requests, historical ones one request each (<= 1 request/s)."""
    from albert.research import xcat
    from albert.research.recurring import pack_live

    CANDLES.mkdir(parents=True, exist_ok=True)
    m = pl.read_parquet(out / "markets.parquet").filter(pl.col("open_time").is_not_null() & pl.col("close_time").is_not_null())
    m = m.with_columns(pl.col("open_time").alias("lo"),
                       pl.min_horizontal("close_time", pl.col("open_time") + timedelta(hours=xcat.MAX_PERIODS - 1)).alias("hi"),
                       pl.lit(60).alias("period"))
    done_path = CANDLES / "done.parquet"
    done = set(pl.read_parquet(done_path)["ticker"].to_list()) if done_path.exists() else set()
    done |= reuse_candles(m["ticker"].to_list(), done, log=log)
    todo = m.filter(~pl.col("ticker").is_in(list(done)))
    api = _api()
    cutoff = api.cutoff()["market_settled_ts"]
    hist = todo.filter(pl.col("settlement_ts") < cutoff).to_dicts()
    live = todo.filter(pl.col("settlement_ts") >= cutoff).to_dicts()
    jobs = [{"kind": "batch", "rows": b} for b in pack_live(live)] + [{"kind": "hist", "rows": [r]} for r in hist]
    if max_requests is not None:
        jobs = jobs[:max_requests]
    log(f"candles: {m.height} markets, {len(done)} done, {len(hist)} historical + {len(live)} live -> {len(jobs)} requests", flush=True)
    part = len(list(CANDLES.glob("part-*.parquet")))
    buf: list[dict[str, Any]] = []
    fin: list[str] = []
    for n, job in enumerate(jobs, 1):
        for tk, got in xcat._run_job(api, job):
            if got is not None:
                buf.extend(got)
                fin.append(tk)
        if n % 20 == 0 or n == len(jobs):
            if buf:
                pl.DataFrame(buf, schema=xcat.OHLC_SCHEMA).write_parquet(CANDLES / f"part-{part:05d}.parquet")
                part += 1
            buf = []
            done |= set(fin)
            _save(pl.DataFrame({"ticker": sorted(done)}, schema={"ticker": pl.String}), done_path)
            log(f"  {n}/{len(jobs)} requests, {len(done)} markets", flush=True)


REUSE = (Path("data/research/recurring/candles"), Path("data/research/xcat/hourly"))


def reuse_candles(tickers: list[str], done: set[str], *, log=print) -> set[str]:
    """Copy whole-life hourly candles other studies already fetched (same endpoint and window:
    open -> close at 60 min) instead of re-requesting them; returns the tickers copied."""
    from albert.research import xcat

    want = set(tickers) - done
    got: set[str] = set()
    for d in REUSE:
        dp = d / "done.parquet"
        if not dp.exists() or not want - got:
            continue
        have = (set(pl.read_parquet(dp)["ticker"].to_list()) & want) - got
        if not have:
            continue
        rows = (pl.scan_parquet(str(d / "part-*.parquet")).filter(pl.col("ticker").is_in(list(have)))
                .select(list(xcat.OHLC_SCHEMA)).collect().cast(xcat.OHLC_SCHEMA))
        if rows.height:
            part = len(list(CANDLES.glob("part-*.parquet")))
            rows.write_parquet(CANDLES / f"part-{part:05d}.parquet")
        got |= have
        log(f"reused {len(have)} markets ({rows.height} candles) from {d}", flush=True)
    if got:
        _save(pl.DataFrame({"ticker": sorted(done | got)}, schema={"ticker": pl.String}), CANDLES / "done.parquet")
    return got


def load_candles(tickers: list[str] | None = None) -> pl.DataFrame:
    parts = sorted(CANDLES.glob("part-*.parquet"))
    if not parts:
        raise RuntimeError("no candles; run `candles` first")
    cols = ["ticker", "end_ts", "bid_close", "ask_close", "p_close", "p_low", "p_high", "volume"]
    lf = pl.scan_parquet([str(p) for p in parts]).select(cols)
    if tickers is not None:
        lf = lf.filter(pl.col("ticker").is_in(tickers))
    return lf.collect().unique(["ticker", "end_ts"]).sort("ticker", "end_ts")


# ---- leg panel --------------------------------------------------------------------------------------
PERIOD_RX = re.compile(r"from ([A-Z][a-z]{2} \d{1,2}, \d{4}) to ([A-Z][a-z]{2} \d{1,2}, \d{4})")
MONTHS = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def family(series: str) -> str:
    if series in STRAIT:
        return STRAIT[series][1]
    return {"KXTRUTHSOCIAL": "ts_week", "KXTRUTHSOCIALD": "ts_day", "KXACTBLUETOP": "actblue"}[series]


def event_end(event_ticker: str) -> date | None:
    """Date coded in the event ticker (``...-26NOV15`` -> 2026-11-15)."""
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", event_ticker)
    return date(2000 + int(m.group(1)), MONTHS[m.group(2)], int(m.group(3))) if m else None


def period_of(rules: str | None, event_ticker: str) -> tuple[date | None, date | None]:
    m = PERIOD_RX.search(rules or "")
    if m:
        a, b = (datetime.strptime(x, "%b %d, %Y").date() for x in m.groups())
        return a, b
    e = event_end(event_ticker)
    return (e - timedelta(days=6), e) if e else (None, None)


def decision_times(fam: str, open_t: datetime, close_t: datetime, p_end: date | None,
                   prev_settle: datetime | None) -> dict[str, datetime]:
    """Fixed decision times per family (UTC). None of them uses the outcome or the actual close."""
    out: dict[str, datetime] = {"open1h": open_t + timedelta(hours=1)}
    if fam in ("sum", "peak", "maxday"):
        if prev_settle is not None:
            out["post"] = prev_settle + timedelta(hours=1)        # previous week public (Kalshi settled it)
        if p_end is not None:
            out["sun"] = datetime(p_end.year, p_end.month, p_end.day, 20, tzinfo=UTC)
        out["final"] = close_t - timedelta(hours=1)               # Tuesday after the week, 1 h before close
    elif fam == "ts_week":
        out["mid"] = open_t + (close_t - open_t) / 2
        out["close24h"] = close_t - timedelta(hours=24)
        out["close6h"] = close_t - timedelta(hours=6)
    elif fam == "ts_day":
        d = (close_t - timedelta(hours=6)).astimezone(ET).date()   # the ET day being counted
        for h in (10, 14, 18, 22):
            out[f"et{h}"] = datetime(d.year, d.month, d.day, h, tzinfo=ET).astimezone(UTC)
    elif fam == "actblue":
        out["close48h"] = close_t - timedelta(hours=48)
        out["close24h"] = close_t - timedelta(hours=24)
        out["close3h"] = close_t - timedelta(hours=3)
    return {k: v for k, v in out.items() if open_t <= v < close_t}


def build_panel(out: Path = OUT) -> pl.DataFrame:
    """One row per (settled leg, decision time) with the quote of the last candle that ended at or
    before t (within 3 h), the payout y (settlement value, 1/N on MAX ties) and leg descriptors."""
    mk = pl.read_parquet(out / "markets.parquet").filter(pl.col("settlement_value").is_not_null())
    ev = (mk.group_by("series_ticker", "event_ticker")
          .agg(pl.col("open_time").min().alias("ev_open"), pl.col("close_time").max().alias("ev_close"),
               pl.col("settlement_ts").max().alias("ev_settle"), pl.col("rules_primary").first())
          .sort("series_ticker", "ev_close"))
    ev = ev.with_columns(pl.col("ev_settle").shift(1).over("series_ticker").alias("prev_settle"))
    rows = []
    for r in ev.iter_rows(named=True):
        fam = family(r["series_ticker"])
        a, b = period_of(r["rules_primary"], r["event_ticker"])
        prev = r["prev_settle"] if fam in ("sum", "peak", "maxday") else None
        for name, t in decision_times(fam, r["ev_open"], r["ev_close"], b, prev).items():
            rows.append({"event_ticker": r["event_ticker"], "fam": fam, "p_start": a, "p_end": b, "dt": name, "t": t})
    dts = pl.DataFrame(rows, schema={"event_ticker": pl.String, "fam": pl.String, "p_start": pl.Date, "p_end": pl.Date,
                                     "dt": pl.String, "t": pl.Datetime("us", "UTC")})
    legs = mk.select("ticker", "event_ticker", "series_ticker", "yes_sub_title", "strike_type", "floor_strike", "cap_strike",
                     "open_time", "close_time", "settlement_ts", pl.col("settlement_value").alias("y"), "volume")
    p = legs.join(dts, on="event_ticker").filter((pl.col("t") >= pl.col("open_time")) & (pl.col("t") < pl.col("close_time")))
    c = load_candles(p["ticker"].unique().to_list())
    q = (c.filter(pl.col("bid_close").is_not_null() & pl.col("ask_close").is_not_null())
         .select("ticker", pl.col("end_ts").alias("q_ts"), pl.col("bid_close").alias("bid"), pl.col("ask_close").alias("ask"))
         .sort("ticker", "q_ts"))
    p = p.sort("ticker", "t").join_asof(q, left_on="t", right_on="q_ts", by="ticker", strategy="backward", check_sortedness=False)
    p = p.with_columns(pl.when((pl.col("t") - pl.col("q_ts")) <= timedelta(hours=3)).then(pl.col("bid")).alias("bid"),
                       pl.when((pl.col("t") - pl.col("q_ts")) <= timedelta(hours=3)).then(pl.col("ask")).alias("ask"))
    p = p.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                       pl.when(pl.col("p_end") < DEV_END.date()).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("split"))
    # KXTRUTHSOCIALD starts 2026-10-02 (no dev period): its own split at 2026-11-01, stated in the report
    p = p.with_columns(pl.when(pl.col("fam") == "ts_day")
                       .then(pl.when(pl.col("close_time") < TS_DAILY_SPLIT).then(pl.lit("dev")).otherwise(pl.lit("holdout")))
                       .otherwise(pl.col("split")).alias("split"))
    # bin rank within the event (0 = lowest strike) for ladders / bins
    lo = pl.coalesce(pl.col("floor_strike"), pl.col("cap_strike") - 1e6)
    p = p.with_columns(lo.rank("dense").over("event_ticker", "dt").alias("rank"),
                       lo.rank("dense", descending=True).over("event_ticker", "dt").alias("rank_top"))
    return p.sort("event_ticker", "dt", "ticker")


# ---- strait model (as-of) -------------------------------------------------------------------------------
REVISION_SAFE = date(2026, 8, 17)    # weeks starting on/after this match Kalshi's first release exactly


def known_weekly(mk: pl.DataFrame) -> pl.DataFrame:
    """First-release weekly values from Kalshi settlements (public at settlement_ts): sum or peak per
    series and week. Values like 'Above 20' (old settlement text) are dropped."""
    e = (mk.filter(pl.col("series_ticker").is_in([s for s, (_, k) in STRAIT.items() if k in ("sum", "peak")]))
         .group_by("series_ticker", "event_ticker")
         .agg(pl.col("expiration_value").drop_nulls().first().alias("ev"), pl.col("settlement_ts").max().alias("known"),
              pl.col("rules_primary").first()))
    e = e.with_columns(pl.col("ev").str.extract(r"^(\d+(?:\.\d+)?)$", 1).cast(pl.Float64).alias("value"))
    ends = [period_of(r, t)[1] for r, t in zip(e["rules_primary"], e["event_ticker"], strict=True)]
    return e.with_columns(pl.Series("p_end", ends, dtype=pl.Date)).drop("rules_primary").sort("series_ticker", "p_end")


def ratio_probs(hist: list[tuple[date, float]], target_end: date, strikes: list[float], *, min_n: int = 6) -> list[float | None]:
    """P(value > strike) for the week ending ``target_end`` from the series' own known weekly values:
    last value x exp(r), r from all past h-week log changes (h = weeks from the last known week)."""
    if len(hist) < 2:
        return [None] * len(strikes)
    hist = sorted(hist)
    last_end, last_v = hist[-1]
    h = max(1, round((target_end - last_end).days / 7))
    by = dict(hist)
    rs = [math.log((by[d + timedelta(days=7 * h)] + 1) / (v + 1)) for d, v in hist if d + timedelta(days=7 * h) in by]
    if len(rs) < min_n:
        return [None] * len(strikes)
    sims = (last_v + 1) * np.exp(np.asarray(rs)) - 1
    return [float((np.sum(sims > k) + 0.5) / (len(sims) + 1)) for k in strikes]


def maxday_probs(daily: dict[date, float], known_through: date, days: list[date], *, weeks: int = 8, k: float = 4.0) -> list[float | None]:
    """P-weighted payout of each day leg: share of the last ``weeks`` published weeks (on/after
    REVISION_SAFE) in which that weekday held the maximum (ties 1/N), shrunk toward 1/7."""
    sunday = known_through
    scores = np.zeros(7)
    n = 0
    for w in range(weeks):
        end = sunday - timedelta(days=7 * w)
        start = end - timedelta(days=6)
        if start < REVISION_SAFE:
            break
        vals = [daily.get(start + timedelta(days=i)) for i in range(7)]
        if any(v is None for v in vals):
            break
        mx = max(vals)
        tied = [i for i, v in enumerate(vals) if v == mx]
        for i in tied:
            scores[(start + timedelta(days=i)).weekday()] += 1 / len(tied)
        n += 1
    if n < 3:
        return [None] * len(days)
    return [float((scores[d.weekday()] + k / 7) / (n + k)) for d in days]


def add_strait_model(p: pl.DataFrame, mk: pl.DataFrame, pw: pl.DataFrame) -> pl.DataFrame:
    """p_model for strait legs, using only values public before each decision time t:
    sum / peak: Kalshi first-release weekly values settled before t (own history; PortWatch publishes
    the same totals the same Tuesday); maxday: PortWatch daily counts of weeks on/after 2026-08-17
    whose weekly release (``published_at``) precedes t."""
    kw = known_weekly(mk)
    hist = {s: list(zip(g["p_end"].to_list(), g["value"].to_list(), g["known"].to_list(), strict=True))
            for (s,), g in kw.filter(pl.col("value").is_not_null()).group_by("series_ticker")}
    hz = pw.filter(pl.col("portid") == "chokepoint6")
    daily = dict(zip(hz["date"].to_list(), hz["n_total"].cast(pl.Float64).to_list(), strict=True))
    out = []
    for (ev, dt), g in p.filter(pl.col("fam").is_in(["sum", "peak", "maxday"])).group_by("event_ticker", "dt"):
        r = g.row(0, named=True)
        t, fam, s = r["t"], r["fam"], r["series_ticker"]
        if fam in ("sum", "peak"):
            h = [(d, v) for d, v, kn in hist.get(s, []) if kn is not None and kn < t]
            pr = ratio_probs(h, r["p_end"], g["floor_strike"].to_list())
        else:
            pub = [d for d in daily if published_at(d) <= t]
            through = max(pub) if pub else None
            legs = [_leg_day(x, r["p_end"]) for x in g["ticker"].to_list()]
            pr = maxday_probs(daily, through, legs) if through and all(legs) else [None] * g.height
        out.append(g.select("ticker", "dt").with_columns(pl.Series("p_model", pr, dtype=pl.Float64)))
    pm = pl.concat(out) if out else pl.DataFrame(schema={"ticker": pl.String, "dt": pl.String, "p_model": pl.Float64})
    return p.join(pm, on=["ticker", "dt"], how="left")


def _leg_day(ticker: str, p_end: date | None) -> date | None:
    m = re.search(r"-([A-Z]{3})(\d{2})$", ticker)
    if not m or p_end is None:
        return None
    d = date(p_end.year, MONTHS[m.group(1)], int(m.group(2)))
    return d if d <= p_end else date(p_end.year - 1, d.month, d.day)


# ---- execution and scoring -------------------------------------------------------------------------------
STAKE = 4.5            # $ per order: n = floor(STAKE / price), fee rounded up per order; <= $5 per event
MAX_SPREAD = 0.10      # only cross books at most 10c wide
EVENT_CAP = 5.0


def take(rows: pl.DataFrame, side: pl.Expr) -> pl.DataFrame:
    """Taker fills: ``side`` is 'yes' / 'no' / null per row. YES at the ask, NO at 1 - bid."""
    from albert.research.counts import taker_fee

    r = rows.with_columns(side.alias("side")).filter(
        pl.col("side").is_not_null() & pl.col("bid").is_not_null() & pl.col("ask").is_not_null()
        & (pl.col("ask") - pl.col("bid") <= MAX_SPREAD + 1e-9))
    r = r.with_columns(pl.when(pl.col("side") == "yes").then(pl.col("ask")).otherwise(1 - pl.col("bid")).alias("price"),
                       pl.when(pl.col("side") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y")).alias("payoff"))
    r = r.filter((pl.col("price") > 0.0) & (pl.col("price") < 1.0))
    if not r.height:
        return r.with_columns(pl.lit(None, pl.Int64).alias("n"), pl.lit(None, pl.Float64).alias("fee_order"),
                              pl.lit(None, pl.Float64).alias("pnl_c"), pl.lit(None, pl.Float64).alias("pnl_usd"),
                              pl.lit(None, pl.Float64).alias("cost_usd"))
    n = [max(1, int(STAKE // pr)) for pr in r["price"].to_list()]
    fee = [taker_fee(k, pr, 1.0) for k, pr in zip(n, r["price"].to_list(), strict=True)]
    r = r.with_columns(pl.Series("n", n, dtype=pl.Int64), pl.Series("fee_order", fee, dtype=pl.Float64))
    return r.with_columns(
        (100 * (pl.col("payoff") - pl.col("price") - pl.col("fee_order") / pl.col("n"))).alias("pnl_c"),
        (pl.col("n") * (pl.col("payoff") - pl.col("price")) - pl.col("fee_order")).alias("pnl_usd"),
        (pl.col("n") * pl.col("price") + pl.col("fee_order")).alias("cost_usd"))


def one_per_event(f: pl.DataFrame) -> pl.DataFrame:
    """At most one order per (event, decision time): the leg with the largest edge column if present,
    else the cheapest price (keeps the per-event exposure within the $5 cap)."""
    if not f.height:
        return f
    key = "edge" if "edge" in f.columns else "price"
    desc = key == "edge"
    return f.sort(key, descending=desc).unique(["event_ticker", "dt"], keep="first", maintain_order=True)


def score(f: pl.DataFrame) -> dict[str, Any]:
    from albert.research.counts import cluster_se

    if not f.height:
        return {"fills": 0, "events": 0}
    x = f["pnl_c"].to_numpy().astype(float)
    ev = f["event_ticker"].to_numpy().astype(str)
    dd = f["t"].dt.date().cast(pl.String).to_numpy().astype(str)
    m = float(x.mean())

    def tt(*g: np.ndarray) -> float | None:
        if any(len(np.unique(k)) < 2 for k in g):
            return None
        se = cluster_se(x, *g)
        return round(m / se, 2) if se and se > 1e-12 and math.isfinite(se) else None
    wk = (f.group_by(pl.col("settlement_ts").dt.truncate("1w").alias("w")).agg(pl.col("pnl_usd").sum()))
    return {"fills": f.height, "events": len(np.unique(ev)), "c_per_contract": round(m, 2),
            "pnl_per_dollar": round(float(f["pnl_usd"].sum() / f["cost_usd"].sum()), 4),
            "pnl_usd": round(float(f["pnl_usd"].sum()), 2), "t_event": tt(ev), "t_2way": tt(ev, dd),
            "weeks": wk.height, "pct_pos_weeks": round(100 * float((wk["pnl_usd"] > 0).mean()), 0),
            "mean_price": round(float(f["price"].mean()), 3), "win_rate": round(float((f["payoff"] > 0.5).mean()), 3)}


# ---- the grid -------------------------------------------------------------------------------------------------
GROUPS = {"strait": ("sum", "peak", "maxday"), "sum": ("sum",), "peak": ("peak",), "maxday": ("maxday",),
          "ts_week": ("ts_week",), "ts_day": ("ts_day",), "actblue": ("actblue",)}
LONG_BUCKETS = ((0.03, 0.10), (0.10, 0.20))
THETAS = (0.05, 0.10)


def cells(p: pl.DataFrame) -> list[tuple[str, dict[str, Any], pl.DataFrame]]:
    """Every dev/holdout cell: (name, spec, fills). Specs are data so the frozen rule can be replayed."""
    out = []
    for lo, hi in LONG_BUCKETS:     # H11 pooled: every family and decision time (added before the holdout)
        spec = {"kind": "longshot_no", "group": "all", "dt": "any", "lo": lo, "hi": hi}
        f = take(p, pl.when((pl.col("mid") >= lo) & (pl.col("mid") < hi)).then(pl.lit("no")))
        out.append((f"longshot_no|all|any|{lo:.2f}-{hi:.2f}", spec, one_per_event(f)))
    for g, fams in GROUPS.items():
        sub = p.filter(pl.col("fam").is_in(list(fams)))
        for dt in sorted(sub["dt"].unique().to_list()):
            s = sub.filter(pl.col("dt") == dt)
            for lo, hi in LONG_BUCKETS:     # H4 / H8 / H11: NO on cheap YES legs
                spec = {"kind": "longshot_no", "group": g, "dt": dt, "lo": lo, "hi": hi}
                f = take(s, pl.when((pl.col("mid") >= lo) & (pl.col("mid") < hi)).then(pl.lit("no")))
                out.append((f"longshot_no|{g}|{dt}|{lo:.2f}-{hi:.2f}", spec, one_per_event(f)))
            spec = {"kind": "favourite_yes", "group": g, "dt": dt, "lo": 0.80, "hi": 0.97}
            f = take(s, pl.when((pl.col("mid") >= 0.80) & (pl.col("mid") < 0.97)).then(pl.lit("yes")))
            out.append((f"favourite_yes|{g}|{dt}", spec, one_per_event(f)))
            if g in ("sum", "peak", "maxday") and "p_model" in s.columns:   # H1 / H5 / H6 / H7: model taker
                for th in THETAS:
                    spec = {"kind": "model", "group": g, "dt": dt, "theta": th}
                    out.append((f"model|{g}|{dt}|{th:.2f}", spec, one_per_event(model_take(s, th))))
    return out


def model_take(s: pl.DataFrame, theta: float) -> pl.DataFrame:
    from albert.research.counts import taker_fee

    s = s.filter(pl.col("p_model").is_not_null())
    fee_y = pl.col("ask").map_elements(lambda a: taker_fee(int(STAKE // a), a, 1.0) / max(1, int(STAKE // a)) if 0 < a < 1 else 1.0,
                                       return_dtype=pl.Float64)
    fee_n = pl.col("bid").map_elements(lambda b: taker_fee(int(STAKE // (1 - b)), 1 - b, 1.0) / max(1, int(STAKE // (1 - b))) if 0 < b < 1 else 1.0,
                                       return_dtype=pl.Float64)
    s = s.with_columns((pl.col("p_model") - pl.col("ask") - fee_y).alias("ey"), (pl.col("bid") - pl.col("p_model") - fee_n).alias("en"))
    side = pl.when(pl.col("ey") >= theta).then(pl.lit("yes")).when(pl.col("en") >= theta).then(pl.lit("no"))
    f = take(s, side)
    return f.with_columns(pl.when(pl.col("side") == "yes").then(pl.col("ey")).otherwise(pl.col("en")).alias("edge"))


def run_grid(p: pl.DataFrame, split: str) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    rows, fills = [], {}
    for name, spec, f in cells(p.filter(pl.col("split") == split)):
        rows.append({"cell": name, "split": split, **{k: spec.get(k) for k in ("kind", "group", "dt")}, **score(f)})
        fills[name] = f
    return pl.DataFrame(rows, infer_schema_length=None), fills


# ---- study driver -----------------------------------------------------------------------------------------
MIN_FILLS, MIN_EVENTS = 30, 10


def load_study_panel(out: Path = OUT) -> pl.DataFrame:
    p = build_panel(out)
    mk = pl.read_parquet(out / "markets.parquet")
    return add_strait_model(p, mk, load_portwatch())


def marks_from_candles(tickers: list[str]) -> pl.DataFrame:
    c = load_candles(tickers)
    two = pl.col("bid_close").is_not_null() & pl.col("ask_close").is_not_null() & (pl.col("ask_close") > pl.col("bid_close"))
    mid = pl.when(two).then((pl.col("bid_close") + pl.col("ask_close")) / 2).otherwise(pl.col("p_close"))
    return (c.with_columns(mid.alias("yes_mid")).filter(pl.col("yes_mid").is_not_null())
            .select("ticker", pl.col("end_ts").alias("ts"), "yes_mid").sort("ticker", "ts"))


def book(f: pl.DataFrame, name: str, split: str) -> tuple[pl.DataFrame, dict[str, Any]]:
    """$100 capped book (<= $5 open per event), daily MTM at the candle mid -> vol_metrics block."""
    from albert.research.ta_calendar import capped_ledger
    from albert.research.vol_metrics import ledger_metrics

    if not f.height:
        return pl.DataFrame(), {"positions": 0}
    g = f.with_columns(pl.when(pl.col("side") == "yes").then(1).otherwise(-1).alias("dir"))
    led = capped_ledger(g, NAME, name, split, event_cap=EVENT_CAP)
    if led.is_empty():
        return led, {"positions": 0}
    m = ledger_metrics(led, marks_from_candles(led["ticker"].unique().to_list()))
    keep = ("positions", "events", "pnl_usd", "usd_per_week", "ann_return_pct", "ann_vol_pct", "sharpe", "sharpe_ci90",
            "p_sharpe_pos", "sharpe_ex_top5", "usd_per_week_ex_top5", "sortino", "max_dd_usd", "min_cum_pnl_usd",
            "weeks_positive_pct", "skew", "first_day", "last_day")
    return led, {k: m.get(k) for k in keep}


def study_dev(out: Path = OUT, *, log=print) -> dict[str, Any]:
    p = load_study_panel(out)
    _save(p, out / "panel.parquet")
    grid, _fills = run_grid(p, "dev")
    grid = grid.sort("t_event", descending=True, nulls_last=True)
    grid.write_csv(out / "grid_dev.csv")
    ok = grid.filter((pl.col("fills") >= MIN_FILLS) & (pl.col("events") >= MIN_EVENTS) & (pl.col("c_per_contract") > 0)
                     & (pl.col("pnl_per_dollar") > 0) & pl.col("t_event").is_not_null())
    picks: list[dict[str, Any]] = []
    for r in ok.iter_rows(named=True):          # primary = best dev t; secondaries = next best of other kinds/groups
        if not picks or (len(picks) < 3 and all((r["kind"], r["group"]) != (q["kind"], q["group"]) for q in picks)):
            picks.append(r)
    spec = {name: s for name, s, _ in cells(p.filter(pl.col("split") == "dev"))}
    frozen = {"frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"),
              "dev": "period ends <= 2026-09-30 (KXTRUTHSOCIALD: closes before 2026-11-01)",
              "holdout": "period ends >= 2026-10-01 (KXTRUTHSOCIALD: closes 2026-11-01 on)",
              "selection": f"highest event-clustered t among dev cells with >= {MIN_FILLS} fills, >= {MIN_EVENTS} events and positive c/contract",
              "dev_cells_tried": grid.height,
              "execution": "taker, YES at the ask / NO at 1 - bid, spread <= 10c, n = floor($4.50 / price), Kalshi taker fee 0.07 x n x P(1-P) rounded up per order; one order per event and decision time; held to settlement",
              "primary": ({"cell": picks[0]["cell"], **spec[picks[0]["cell"]], "dev": picks[0]} if picks else None),
              "secondary": [{"cell": q["cell"], **spec[q["cell"]], "dev": q} for q in picks[1:]]}
    (out / "primary.json").write_text(json.dumps(frozen, indent=1, default=str))
    log(f"dev: {grid.height} cells, {ok.height} pass the floor; primary = {frozen['primary']['cell'] if picks else None}")
    return frozen


def study_holdout(out: Path = OUT, *, log=print) -> dict[str, Any]:
    frozen = json.loads((out / "primary.json").read_text())
    p = pl.read_parquet(out / "panel.parquet")
    res: dict[str, Any] = {"frozen": frozen, "rules": []}
    hgrid, hfills = run_grid(p, "holdout")
    hgrid.write_csv(out / "grid_holdout.csv")
    _, dfills = run_grid(p, "dev")
    (out / "ledgers").mkdir(parents=True, exist_ok=True)
    rules = ([frozen["primary"]] if frozen.get("primary") else []) + frozen.get("secondary", [])
    for i, r in enumerate(rules):
        role = "primary" if i == 0 else f"secondary{i}"
        row = {"role": role, "cell": r["cell"]}
        for split, fl in (("dev", dfills), ("holdout", hfills)):
            f = fl.get(r["cell"], pl.DataFrame())
            row[split] = score(f)
            led, met = book(f, r["cell"], split)
            if led.height:
                led.write_parquet(out / "ledgers" / f"ledger_{role}_{split}.parquet")
            row[f"{split}_book"] = met
        res["rules"].append(row)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    for r in res["rules"]:
        log(f"{r['role']} {r['cell']}: dev {r['dev']} | holdout {r['holdout']} | book {r['holdout_book']}")
    return res


# ---- ActBlue: Wayback captures as as-of snapshots ---------------------------------------------------------
def _norm(name: str | None) -> str:
    return re.sub(r"[^a-z]", "", (name or "").lower().replace('"pat"', "").replace("patricia", "pat"))


def actblue_wayback(out: Path = OUT, cache: Path = CACHE / "wayback" / "actblue") -> pl.DataFrame:
    """For every archived directory capture (capture time = as-of): the leftmost card, the open
    KXACTBLUETOP event that closes next, that card's Kalshi quote at the capture hour, and the result."""
    mk = pl.read_parquet(out / "markets.parquet").filter(pl.col("series_ticker") == "KXACTBLUETOP")
    c = load_candles(mk["ticker"].to_list())
    q = (c.filter(pl.col("bid_close").is_not_null() & pl.col("ask_close").is_not_null())
         .select("ticker", "end_ts", "bid_close", "ask_close").sort("ticker", "end_ts"))
    rows = []
    for path in sorted(cache.glob("dir_*.html")):
        ts = datetime.strptime(path.stem[4:], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        cards = parse_actblue(path.read_text(errors="replace"))
        if not cards:
            continue
        open_ev = mk.filter((pl.col("open_time") <= ts) & (pl.col("close_time") > ts))
        if not open_ev.height:
            rows.append({"capture_utc": ts, "leader": cards[0]["name"], "event_ticker": None})
            continue
        ev = open_ev.sort("close_time")["event_ticker"][0]
        legs = open_ev.filter(pl.col("event_ticker") == ev)
        lead = [r for r in legs.iter_rows(named=True) if _norm(r["yes_sub_title"]) and _norm(r["yes_sub_title"]) in _norm(cards[0]["name"])]
        r = lead[0] if lead else None
        quote = None
        if r is not None:
            qq = q.filter((pl.col("ticker") == r["ticker"]) & (pl.col("end_ts") <= ts)).tail(1)
            quote = qq.row(0, named=True) if qq.height else None
        rows.append({"capture_utc": ts, "leader": cards[0]["name"], "event_ticker": ev,
                     "close_time": legs["close_time"].max(), "hours_to_close": (legs["close_time"].max() - ts).total_seconds() / 3600,
                     "leader_ticker": r["ticker"] if r else None, "leader_y": r["settlement_value"] if r else None,
                     "bid": quote["bid_close"] if quote else None, "ask": quote["ask_close"] if quote else None,
                     "winner": legs.filter(pl.col("settlement_value") >= 0.999)["yes_sub_title"].to_list()})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- descriptive checks (no rule selection) ------------------------------------------------------------
def extras(out: Path = OUT) -> dict[str, Any]:
    """H2 decided window, ladder overround and the ActBlue Wayback snapshots; descriptive only."""
    p = pl.read_parquet(out / "panel.parquet")
    res: dict[str, Any] = {}
    mk = pl.read_parquet(out / "markets.parquet")
    st = mk.filter(pl.col("series_ticker").is_in(list(STRAIT)))
    c = load_candles(st["ticker"].to_list())
    c = c.filter(pl.col("bid_close").is_not_null() & pl.col("ask_close").is_not_null()).join(
        st.select("ticker", "event_ticker", "series_ticker", "close_time", pl.col("settlement_value").alias("y")), on="ticker")
    c = c.with_columns(((pl.col("close_time") - pl.col("end_ts")).dt.total_minutes() / 60).round(0).alias("h_to_close"),
                       ((pl.col("bid_close") + pl.col("ask_close")) / 2).alias("mid"))
    # H2: how far the mids are from the payoff in the last hours before the Tuesday close
    win = c.filter(pl.col("h_to_close").is_between(1, 8))
    res["h2_decided_window"] = (win.group_by("h_to_close")
                                .agg(pl.len().alias("legs"), (pl.col("mid") - pl.col("y")).abs().mean().alias("mean_abs_err"),
                                     ((pl.col("mid") - pl.col("y")).abs() > 0.05).mean().alias("share_off_by_5c"),
                                     # a still-cheap winner or a still-rich loser: the take that the release would allow
                                     (((pl.col("y") > 0.99) & (pl.col("ask_close") < 0.95))
                                      | ((pl.col("y") < 0.01) & (pl.col("bid_close") > 0.05))).mean().alias("share_takeable_5c"))
                                .sort("h_to_close", descending=True).to_dicts())
    # first hour on the Tuesday when every leg of an event is within 3c of its payoff
    conv = (c.with_columns((pl.col("mid") - pl.col("y")).abs().alias("err"))
            .filter(pl.col("end_ts") >= pl.col("close_time") - timedelta(hours=30))
            .group_by("event_ticker", "end_ts").agg(pl.col("err").max().alias("max_err"), pl.col("close_time").first())
            .filter(pl.col("max_err") <= 0.03).group_by("event_ticker")
            .agg(pl.col("end_ts").min().alias("converged"), pl.col("close_time").first()))
    res["h2_converged_hour_utc"] = (conv.with_columns(pl.col("converged").dt.hour().alias("hour"),
                                                      ((pl.col("close_time") - pl.col("converged")).dt.total_minutes() / 60).alias("h_before_close"))
                                    .group_by("hour").agg(pl.len().alias("events"), pl.col("h_before_close").median())
                                    .sort("hour").to_dicts())
    res["h2_events_never_converged_before_close"] = int(st["event_ticker"].n_unique() - conv.height)
    # overround: sum of asks / bids over the legs of mutually exclusive events (TS bins, ActBlue names, MAX days)
    ex = p.filter(pl.col("fam").is_in(["ts_week", "ts_day", "actblue", "maxday"]) & pl.col("ask").is_not_null())
    res["overround"] = (ex.group_by("fam", "dt", "event_ticker").agg(pl.col("ask").sum().alias("sum_ask"), pl.col("bid").sum().alias("sum_bid"),
                                                                    pl.len().alias("legs"))
                        .group_by("fam", "dt").agg(pl.len().alias("events"), pl.col("sum_ask").mean(), pl.col("sum_bid").mean())
                        .sort("fam", "dt").to_dicts())
    # calibration by price bucket (all slice legs, both splits): y - mid
    b = p.filter(pl.col("mid").is_not_null()).with_columns(
        pl.col("mid").cut([0.03, 0.10, 0.20, 0.50, 0.80, 0.90, 0.97], labels=["<3", "3-10", "10-20", "20-50", "50-80", "80-90", "90-97", ">=97"]).alias("bucket"))
    res["calibration"] = (b.group_by("fam", "split", "bucket").agg(pl.len().alias("n"), pl.col("event_ticker").n_unique().alias("events"),
                                                                    (100 * (pl.col("y") - pl.col("mid"))).mean().alias("y_minus_mid_c"))
                          .sort("fam", "split", "bucket").to_dicts())
    ab = actblue_wayback(out)
    res["actblue_wayback"] = ab.to_dicts()
    (out / "extras.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ---- universe scan ------------------------------------------------------------------------------------
UNIVERSE_CATEGORIES = ("Politics", "World", "Social", "Transportation", "Health", "Science and Technology",
                       "Education", "Culture", "Exotics", "Business")
# Other agents of the 2026-11-20 swarm cover these (approval / polls, executive orders and presidential
# actions, speech mentions, culture charts, weather and sports are excluded by category).
OTHER_AGENTS = re.compile(r"APPROVE|APRPOTUS|TRUMPFAV|TRUMPVH|VOTEHUB|GENERICBALLOT|VHGCB|UKPOLL|POLL|"
                          r"\bEO|EOWEEK|EOD$|TRUMPACT|MAMDANIEO|BILLS|PARDON|NOMNUM|EPSTEIN", re.IGNORECASE)
# Money / price / rate series the title filter in albert.categories does not catch (rates, repo, GPU
# rental price indices, freight spot rates, credit-card debt): excluded as financial / economics.
EXTRA_FINANCIAL = re.compile(r"ONRRP|CRED$|A100|B200|H100|H200|RTX|VANRATE|INCOMERATE|TECHLAYOFF", re.IGNORECASE)
WEATHERLIKE = re.compile(r"RAIN|HIGHNY|NYCHOT|SEAICE|PMLA", re.IGNORECASE)


def universe_candidates(series_payload: dict[str, Any], extra: set[str] | None = None) -> pl.DataFrame:
    """Recurring non-financial series in the slice's categories, minus other agents' topics."""
    from albert.categories import kalshi_series_is_financial

    rows = []
    for s in series_payload.get("series") or []:
        t, cat, title = s.get("ticker"), s.get("category"), s.get("title")
        periodic = s.get("frequency") in ("daily", "weekly", "monthly")
        if cat not in UNIVERSE_CATEGORIES or not (periodic or t in (extra or set())):
            continue
        src = (s.get("settlement_sources") or [{}])[0] or {}
        why = ("financial" if kalshi_series_is_financial(t, cat, title) or EXTRA_FINANCIAL.search(t) else
               "other agent" if OTHER_AGENTS.search(t) or re.search(r"approval|poll|executive order", title or "", re.IGNORECASE) else
               "weather" if WEATHERLIKE.search(t) else None)
        rows.append({"series_ticker": t, "category": cat, "frequency": s.get("frequency"), "title": title,
                     "fee_multiplier": num(s.get("fee_multiplier")), "source": src.get("name"), "source_url": src.get("url"),
                     "excluded": why})
    return pl.DataFrame(rows, infer_schema_length=None)


def series_activity(api, series: str, now: datetime, *, days: int = 90, tag: str | None = None) -> dict[str, Any]:
    """Settled events and volume in the last ``days`` and the current open books' typical spread."""
    lo = int((now - timedelta(days=days)).timestamp())
    st = _pages(api, "/markets", {"series_ticker": series, "status": "settled", "min_close_ts": lo, "limit": 1000},
                "markets", tag=tag, max_pages=10)
    op = _pages(api, "/markets", {"series_ticker": series, "status": "open", "limit": 1000}, "markets", tag=tag, max_pages=5)
    ev = {m.get("event_ticker") for m in st}
    vol = sum(num(_field(m, "volume")) or 0.0 for m in st)
    spreads, quoted = [], 0
    for m in op:
        b, a = num(_field(m, "yes_bid")), num(_field(m, "yes_ask"))
        if b is not None and a is not None and b > 0 and a < 1:
            spreads.append(a - b)
            quoted += 1
    last = max((parse_ts(m.get("close_time")) for m in st if m.get("close_time")), default=None)
    return {"series_ticker": series, "events_90d": len(ev), "events_per_month": round(len(ev) / days * 30.4, 2),
            "volume_90d_usd": round(vol, 0), "volume_per_month_usd": round(vol / days * 30.4, 0),
            "open_markets": len(op), "open_quoted": quoted,
            "median_spread_c": round(100 * float(np.median(spreads)), 1) if spreads else None,
            "last_close": last}


def run_universe(out: Path = OUT, *, log=print) -> pl.DataFrame:
    api = _api()
    now = datetime.now(UTC)
    tag = now.strftime("%Y-%m-%d")
    rec = Path("data/research/recurring/series.parquet")
    extra = set(pl.read_parquet(rec)["series_ticker"].to_list()) if rec.exists() else set()
    cand = universe_candidates(api.series().payload, extra)
    todo = cand.filter(pl.col("excluded").is_null())
    log(f"universe: {cand.height} candidates, {todo.height} in scope", flush=True)
    rows = []
    for i, t in enumerate(todo["series_ticker"].to_list(), 1):
        rows.append(series_activity(api, t, now, tag=tag))
        if i % 25 == 0:
            log(f"  {i}/{todo.height}", flush=True)
    act = pl.DataFrame(rows, infer_schema_length=None)
    u = cand.join(act, on="series_ticker", how="left")
    _save(u, out / "universe_raw.parquet")
    return u


# Settlement source -> (alt-data source we may use, robots verdict 2026-11-20, as-of availability, weight).
# weight: 1 = as-of history can be rebuilt from an allowed source; 0.5 = allowed source but no
# history (forward recorder only); 0.1 = settlement source blocked / paid / none (price-only).
FEASIBILITY: list[tuple[str, tuple[str, str, str, float]]] = [
    (r"^KX(HORMUZ|SUEZ|BABELMANDEB|PANAMA|MAXSHIPSHORMUZ)", ("IMF PortWatch daily transit calls (hub.arcgis.com download)", "allowed",
                                                         "weekly Tuesday rebuild, data through the prior Sunday; history before 2026-08-17 restated", 1.0)),
    (r"^KXACTBLUE", ("secure.actblue.com/directory card order", "allowed", "20 Wayback captures since Jul; forward recorder", 0.5)),
    (r"^KXTRUTHSOCIAL|^KXTRUMPDELETE|^KXTRUMPENDORSE", ("Roll Call Factba.se / Truth Social", "blocked", "none (price-only)", 0.1)),
    (r"^KX(TRUMPGOLFD|TRUMPGAGGLED|TRUMPINTERVIEWD|FULLLID|DCGOLF|NJGOLF|LAGO)", ("Roll Call / press pool", "blocked (rollcall.com)", "none", 0.1)),
    (r"^KXKEVCOUNT", ("CISA KEV JSON feed (dateAdded)", "allowed", "exact: dateAdded = publication day", 1.0)),
    (r"^KX(TOKENUSE|TOP3AID|TOPUSAGEAI|OPENSOURCESHAREOR|(GOOG|OPEN|DEEP|ANTH|TENCENT|XIAOMI|BABA|META|MISTRAL|ZAI)SHARE$)",
     ("OpenRouter rankings (openrouter.ai)", "allowed", "page embeds only the last day's top-20 models; share chart loads client-side: forward recorder only", 0.5)),
    (r"^KX(OPENSOURCESHARE$|[A-Z]+V(SPEND|REQ)$)", ("Vercel AI Gateway leaderboards (vercel.com)", "allowed", "daily chart; no history found; forward recorder only", 0.5)),
    (r"^KX(TOPMODEL|LLM1|CHINAAI|TECHRANKLIST)", ("LM Arena leaderboard", "allowed", "Wayback captures (irregular)", 0.5)),
    (r"^KX(SPEECHAI|IMAGEAI|VIDEOAI|OPENINTAI)", ("Artificial Analysis arenas", "not checked", "none", 0.1)),
    (r"^KX(LABSPEND|MODELSPEND|AI|ANTHADOPT|DEEPADOPT|GOOGLADOPT|OPENADOPT|XAIADOPT)", ("Ramp AI Index", "allowed", "monthly release; nothing intra-period", 0.3)),
    (r"^KX(SWENCOUNTERS|ASYLUMCASES)", ("CBP / USCIS monthly releases", "allowed", "monthly release; nothing intra-period", 0.3)),
    (r"^KXMTA", ("MTA daily ridership (data.ny.gov)", "allowed", "dataset update stamps; 1-2 day lag", 1.0)),
    (r"^KXPORTLA", ("Port of LA vessel activity", "allowed", "daily page, no archive; forward only", 0.5)),
    (r"^KXTRUMPPHOTO", ("Getty Images", "unverified (robots 403)", "none", 0.1)),
    (r"^KXLVL4WARNING", ("travel.state.gov", "unverified (robots 403)", "none", 0.1)),
    (r"^KXPRESSBRIEFING", ("whitehouse.gov briefings", "allowed", "post timestamps; monthly, few events", 0.5)),
    (r"^KXWEIGHTDRUGS", ("IQVIA", "paid", "none", 0.1)),
    (r"^KXPOTUSTWEETS", ("x.com", "blocked", "none", 0.1)),
]


def feasibility(series: str) -> tuple[str, str, str, float]:
    for rx, v in FEASIBILITY:
        if re.search(rx, series):
            return v
    return ("news / official statements (no count feed)", "n/a", "none", 0.1)


def rank_universe(out: Path = OUT) -> pl.DataFrame:
    u = pl.read_parquet(out / "universe_raw.parquet").filter(pl.col("excluded").is_null())
    f = [feasibility(s) for s in u["series_ticker"].to_list()]
    u = u.with_columns(pl.Series("alt_source", [x[0] for x in f]), pl.Series("robots", [x[1] for x in f]),
                       pl.Series("as_of", [x[2] for x in f]), pl.Series("feas_weight", [x[3] for x in f], dtype=pl.Float64))
    u = u.with_columns((pl.col("volume_per_month_usd").fill_null(0) * pl.col("feas_weight")).alias("score"))
    u = u.sort("score", descending=True)
    u.write_csv(out / "universe.csv")
    return u


# ---- forward recorder -----------------------------------------------------------------------------------
ARCGIS_ITEM_URL = f"https://www.arcgis.com/sharing/rest/content/items/{PW_ITEM}"


def parse_actblue(text: str) -> list[dict[str, Any]]:
    """The directory's trending-campaign cards in page order (leftmost first). They sit in the
    ``DirectoryExperiments`` React props as ``trendingCampaigns``."""
    import html as _html

    m = re.search(r'data-react-class="DirectoryExperiments" data-react-props="([^"]*)"', text)
    if not m:
        return []
    props = json.loads(_html.unescape(m.group(1)))
    return [{"rank": i + 1, "name": c.get("displayName"), "form": c.get("formName"), "office": c.get("office")}
            for i, c in enumerate(props.get("trendingCampaigns") or [])]


def quote_rows(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"), "sub": m.get("yes_sub_title"),
             "bid": num(_field(m, "yes_bid")), "ask": num(_field(m, "yes_ask")), "last": num(_field(m, "last_price")),
             "volume": num(_field(m, "volume")), "oi": num(_field(m, "open_interest")), "close_time": m.get("close_time")}
            for m in markets]


def record(out: Path = FORWARD, *, series: tuple[str, ...] = SLICE, log=print) -> dict[str, int]:
    """One forward snapshot, appended to ``forward/<UTC day>.jsonl``: the PortWatch item's modified
    stamp (and a new CSV vintage when it changed), the ActBlue directory's card order, and the open
    Kalshi books of the slice series. Each line carries its own fetch time (the as-of stamp)."""
    import httpx

    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    lines: list[dict[str, Any]] = []
    with httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": UA}) as c:
        try:
            # the hub download's own stamp moves with each weekly rebuild (the ArcGIS item "modified"
            # does not: it stayed at 2026-11-10 after the 2026-11-17 rebuild); HEAD costs no download
            h = c.head(PW_URL, params=PW_PARAMS)
            stamp = h.headers.get("x-amz-meta-contentlastmodified")
            item = c.get(ARCGIS_ITEM_URL, params={"f": "json"}).json()
            lines.append({"kind": "portwatch_item", "fetched_utc": now.isoformat(timespec="seconds"),
                          "content_built_utc": stamp, "hub_last_modified": h.headers.get("last-modified"),
                          "item_modified_utc": datetime.fromtimestamp(item["modified"] / 1000, UTC).isoformat()})
            seen = CACHE / "portwatch" / "vintages.jsonl"
            have = {json.loads(x)["content_built_utc"][:19] for x in seen.read_text().splitlines()} if seen.exists() else set()
            if stamp and stamp[:19] not in have:
                meta = fetch_portwatch(log=log)
                lines.append({"kind": "portwatch_vintage", "fetched_utc": now.isoformat(timespec="seconds"), **meta})
        except (httpx.HTTPError, KeyError, ValueError) as error:
            lines.append({"kind": "portwatch_error", "fetched_utc": now.isoformat(timespec="seconds"), "error": str(error)[:200]})
        try:
            r = c.get(ACTBLUE_URL, headers={"Accept": "text/html"})   # the page 404s without an HTML Accept
            cards = parse_actblue(r.text) if r.status_code == 200 else []
            lines.append({"kind": "actblue", "fetched_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                          "status": r.status_code, "leader": cards[0]["name"] if cards else None, "cards": cards})
        except httpx.HTTPError as error:
            lines.append({"kind": "actblue_error", "fetched_utc": now.isoformat(timespec="seconds"), "error": str(error)[:200]})
    api = _api()
    for s in series:
        ms = [m for page in api.series_markets(s, "open") for m in page.payload.get("markets") or []]
        lines.append({"kind": "kalshi_quotes", "fetched_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                      "series_ticker": s, "markets": quote_rows(ms)})
    path = out / f"{now:%Y-%m-%d}.jsonl"
    with path.open("a") as f:
        for x in lines:
            f.write(json.dumps(x, default=str) + "\n")
    counts = {k: sum(1 for x in lines if x["kind"] == k) for k in {x["kind"] for x in lines}}
    counts["quotes"] = sum(len(x["markets"]) for x in lines if x["kind"] == "kalshi_quotes")
    log(f"record: {path} {counts}")
    return counts


# ---- report -----------------------------------------------------------------------------------------------
def _md(rows: list[dict[str, Any]] | pl.DataFrame, cols: list[str] | None = None) -> str:
    df = rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(rows, infer_schema_length=None)
    if not df.height:
        return "(none)\n"
    cols = cols or df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.select(cols).iter_rows():
        out.append("| " + " | ".join("" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out) + "\n"


def write_report(prose: dict[str, str], out: Path = OUT) -> Path:
    """report.md from the stored artifacts; ``prose`` holds the verdict and discussion paragraphs."""
    res = json.loads((out / "results.json").read_text())
    ex = json.loads((out / "extras.json").read_text()) if (out / "extras.json").exists() else {}
    gd = pl.read_csv(out / "grid_dev.csv")
    gh = pl.read_csv(out / "grid_holdout.csv")
    uni = pl.read_csv(out / "universe.csv")
    rev = pl.read_parquet(out / "revisions_vs_kalshi.parquet")
    robots = json.loads((out / "robots.json").read_text())
    s = [f"# Recurring non-financial series with an alternative-data angle (`{NAME}`)\n",
         (f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `python -m albert.research.{NAME}` "
          f"(module `src/albert/research/{NAME}.py`, tests `tests/test_{NAME}.py`).\n"),
         "## Verdict\n", prose["verdict"], "\n"]
    s += ["## Hypotheses (written before any outcome data)\n", (out / "hypotheses.md").read_text().split("\n", 1)[1], "\n"]
    s += ["## Sources and robots (checked 2026-11-20)\n", prose["sources"], "\n",
          _md([{k: r[k] for k in ("host", "path", "http_status", "allowed", "checked_utc")} for r in robots]), "\n"]
    s += ["## Universe: recurring series outside the other agents' families\n", prose["universe"], "\n",
          _md(uni.filter(pl.col("events_90d") > 0).head(15),
              ["series_ticker", "category", "frequency", "events_per_month", "volume_per_month_usd", "median_spread_c",
               "source", "alt_source", "robots", "as_of", "score"]), "\n"]
    s += ["## PortWatch as-of facts\n", prose["portwatch"], "\n",
          _md(rev.group_by("series").agg(pl.len().alias("weeks"), (pl.col("diff") != 0).sum().alias("mismatch"),
                                         pl.col("diff").abs().mean().round(2).alias("mean_abs_diff"),
                                         pl.col("end").filter(pl.col("diff") != 0).max().alias("last_mismatch_week")).sort("series")), "\n"]
    s += ["## Dev screen\n", prose["dev"], "\n",
          _md(gd.head(15), ["cell", "fills", "events", "c_per_contract", "pnl_per_dollar", "t_event", "t_2way", "pct_pos_weeks", "mean_price"]), "\n"]
    fr = res["frozen"]
    s += ["## Frozen rule\n", (f"Frozen {fr['frozen_utc']} (before any holdout row was scored). Selection: {fr['selection']}. "
                                f"Execution: {fr['execution']}. Dev cells tried: {fr['dev_cells_tried']}.\n\n```json\n")
          + json.dumps({k: fr[k] for k in ("primary", "secondary")}, indent=1, default=str) + "\n```\n"]
    rows = []
    for r in res["rules"]:
        for split in ("dev", "holdout"):
            rows.append({"role": r["role"], "cell": r["cell"], "split": split, **r[split]})
    s += ["## Holdout\n", prose["holdout"], "\n",
          _md(rows, ["role", "cell", "split", "fills", "events", "c_per_contract", "pnl_per_dollar", "pnl_usd", "t_event", "t_2way", "pct_pos_weeks"]), "\n"]
    brow = []
    for r in res["rules"]:
        for split in ("dev", "holdout"):
            b = r.get(f"{split}_book") or {}
            brow.append({"role": r["role"], "split": split, **{k: (str(v) if isinstance(v, list) else v) for k, v in b.items()}})
    s += ["## Sharpe and volatility ($100 capped book, <= $5 per event, daily MTM at the mid; `vol_metrics`)\n",
          _md(brow, ["role", "split", "positions", "events", "pnl_usd", "usd_per_week", "ann_vol_pct", "sharpe", "sharpe_ci90",
                     "sharpe_ex_top5", "max_dd_usd", "weeks_positive_pct"]), "\n"]
    s += ["## Holdout grid (exploratory, every cell; not used for selection)\n",
          _md(gh.sort("t_event", descending=True, nulls_last=True).head(12),
              ["cell", "fills", "events", "c_per_contract", "pnl_per_dollar", "t_event", "pct_pos_weeks"]), "\n"]
    if ex:
        s += ["## Descriptive checks\n", prose["extras"], "\n### Strait books in the last hours before the Tuesday close\n",
              _md(ex["h2_decided_window"]), "\nFirst hour (UTC) at which every leg of a strait event sat within 3c of its payoff:\n\n",
              _md(ex["h2_converged_hour_utc"]), f"\nEvents that never converged before close: {ex['h2_events_never_converged_before_close']}.\n",
              "\n### Overround (mean sum of asks / bids over mutually exclusive legs)\n", _md(ex["overround"]),
              "\n### ActBlue: archived directory captures vs the Kalshi quote of the leftmost card\n",
              _md([{k: (str(v) if isinstance(v, list) else v) for k, v in r.items()} for r in ex["actblue_wayback"]]), "\n"]
    s += ["## Forward recorder\n", prose["forward"], "\n", "## What would change the verdict\n", prose["change"], "\n"]
    path = out / "report.md"
    path.write_text("\n".join(s))
    return path


# ---- CLI ---------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m albert.research.{NAME}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("robots")
    sub.add_parser("portwatch")
    sub.add_parser("discover")
    sub.add_parser("universe")
    sub.add_parser("record")
    sub.add_parser("extras")
    sub.add_parser("report")
    st = sub.add_parser("study")
    st.add_argument("split", choices=["dev", "holdout"])
    c = sub.add_parser("candles")
    c.add_argument("--max-requests", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "robots":
        rows = check_robots()
        for r in rows:
            print(r["host"], r["path"], r["http_status"], "allowed" if r["allowed"] else "NOT allowed")
    elif a.cmd == "portwatch":
        print(json.dumps(fetch_portwatch()))
    elif a.cmd == "discover":
        discover()
    elif a.cmd == "study":
        study_dev() if a.split == "dev" else study_holdout()
    elif a.cmd == "report":
        print(write_report(json.loads((OUT / "report_prose.json").read_text())))
    elif a.cmd == "extras":
        extras()
    elif a.cmd == "record":
        record()
    elif a.cmd == "universe":
        if not (OUT / "universe_raw.parquet").exists():
            run_universe()
        print(rank_universe().head(25))
    elif a.cmd == "candles":
        fetch_candles(max_requests=a.max_requests)


if __name__ == "__main__":
    main()
