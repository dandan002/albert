"""Behavioural mispricing in Kalshi culture / entertainment / weather / science-tech markets.

Hypotheses (who trades badly and why) are listed in ``data/research/behav_culture/hypotheses.md``,
written before any price or outcome was examined. This module builds the panels, runs the dev
exploration (< 2026-10-01), and then the pre-registered holdout tests (2026-10-01 .. 2026-11-17).

Panels (all public Kalshi data, every response cached raw under ``data/raw/kalshi``):

* ``weather``: KXHIGH* daily-high ladders (two open tails + four 2-degree brackets). Quotes are the
  last hourly candle that ended at or before fixed decision times (D-1 18:00 ET, D 10:00 local).
  Realised highs and yesterday's high come from the NWS CLI via IEM (``albert.venues.weather``).
* ``culture``: Entertainment and Science/Tech markets (volume-blind ``research/xcat`` hourly sample
  through October, plus every Entertainment / Science/Tech market settled 2026-11-01 .. 11-17
  fetched here) at scheduled close - 24 h and open + 24 h.

Execution in every trade test: taker at the candle's closing ask (YES) or 1 - bid (NO), fee
0.07 x mult x n x P(1-P) rounded up to the cent per order (``ta_calendar.taker_fee_expr``), held
to settlement. t-stats clustered by event and by ET date (weather: one cluster per station-day,
i.e. per event). Sharpe / vol from a capped $100 book through ``vol_metrics``.

    python -m albert.research.behav_culture fetch-weather | fetch-culture | dev | holdout | report
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

from albert.research import weather as wxr
from albert.venues import weather as wx

OUT = Path("data/research/behav_culture")
TABLES = Path("data/tables")
XCAT = Path("data/research/xcat")
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
HOLDOUT_TO = datetime(2026, 11, 18, tzinfo=UTC)          # decisions/settlements up to 11-17 inclusive
ET = ZoneInfo("America/New_York")
KALSHI_TAKER = 0.07
PRICE_BUCKETS = [0.0, 0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 1.0]
BUCKET_LABELS = ["01-05", "05-15", "15-35", "35-65", "65-85", "85-95", "95-99"]
WEATHER_SLOTS = {"d-1 18ET": (-1, 18, "ET"), "d 10L": (0, 10, "local")}


# ---- shared helpers ------------------------------------------------------------------------------
def bucket_expr(price: pl.Expr) -> pl.Expr:
    return price.cut(PRICE_BUCKETS[1:-1], labels=BUCKET_LABELS, left_closed=True).cast(pl.String)


def taker_fee_expr(n: pl.Expr, price: pl.Expr) -> pl.Expr:
    """Kalshi taker fee per order, rounded up to the cent (same as ``ta_calendar.taker_fee_expr``)."""
    raw = KALSHI_TAKER * pl.col("fee_multiplier").fill_null(1.0) * n * price * (1 - price)
    return (raw * 100).round(9).ceil() / 100


def cl_mean(f: pl.DataFrame, col: str, cluster: str, weight: str | None = None) -> tuple[float | None, float | None, int]:
    """Ratio mean sum(col)/sum(weight) (rows if None) with a cluster-robust t (CR1)."""
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


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and cluster-robust (CR1) standard errors."""
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    meat = np.zeros((k, k))
    uniq, inv = np.unique(groups, return_inverse=True)
    g = len(uniq)
    S = np.zeros((g, k))
    np.add.at(S, inv, X * u[:, None])
    meat = S.T @ S
    c = g / (g - 1) * (n - 1) / max(n - k, 1) if g > 1 else 1.0
    V = c * xtx_inv @ meat @ xtx_inv
    return beta, np.sqrt(np.maximum(np.diag(V), 0))


def _r(v: Any, k: int = 2) -> Any:
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), k)


def asof_quotes(points: pl.DataFrame, candles: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """Join each (ticker, t) to the last candle that ended at or before t: bid, ask, mid, spread,
    the candle's age, and the volume traded in the 24 h before t."""
    c = candles.lazy() if isinstance(candles, pl.DataFrame) else candles
    tick = points["ticker"].unique().implode()
    c = (c.filter(pl.col("ticker").is_in(tick))
         .select("ticker", "end_ts", pl.col("yes_bid_close").alias("bid"), pl.col("yes_ask_close").alias("ask"),
                 pl.col("volume").fill_null(0.0).alias("vol"))
         .collect().sort("ticker", "end_ts")
         .with_columns(pl.col("vol").cum_sum().over("ticker").alias("cumvol")))
    p = points.sort("t")
    q = p.join_asof(c.sort("end_ts"), left_on="t", right_on="end_ts", by="ticker", strategy="backward",
                    check_sortedness=False)
    lag = (p.with_columns((pl.col("t") - timedelta(hours=24)).alias("t24")).sort("t24")
           .join_asof(c.select("ticker", pl.col("end_ts").alias("e24"), pl.col("cumvol").alias("cum24"),
                               pl.col("bid").alias("bid24"), pl.col("ask").alias("ask24")).sort("e24"),
                      left_on="t24", right_on="e24", by="ticker", strategy="backward", check_sortedness=False)
           .select("ticker", "t", "cum24", "bid24", "ask24"))
    q = q.join(lag, on=["ticker", "t"], how="left")
    two = pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("ask") > pl.col("bid"))
    return q.with_columns(
        pl.when(two).then((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
        pl.when(two).then(pl.col("ask") - pl.col("bid")).alias("spread"),
        ((pl.col("t") - pl.col("end_ts")).dt.total_minutes() / 60).alias("age_h"),
        (pl.col("cumvol") - pl.col("cum24").fill_null(0.0)).alias("vol24"),
        pl.when(pl.col("bid24").is_not_null() & pl.col("ask24").is_not_null() & (pl.col("ask24") > pl.col("bid24")))
        .then((pl.col("bid24") + pl.col("ask24")) / 2).alias("mid24"),
    ).drop("cum24", "cumvol", "bid24", "ask24")


# ---- weather panel -------------------------------------------------------------------------------
def weather_legs(markets: pl.DataFrame) -> pl.DataFrame:
    """KXHIGH legs with station, day, bracket bounds, position in the ladder and strike features."""
    legs = wxr.high_markets(markets)
    legs = legs.with_columns(
        pl.when(pl.col("lo").is_infinite()).then(pl.lit("low_tail"))
        .when(pl.col("hi").is_infinite()).then(pl.lit("high_tail")).otherwise(pl.lit("interior")).alias("pos"))
    # round-number strike: the bracket (or the tail's finite edge) contains a multiple of 10 / of 5
    lo = pl.when(pl.col("lo").is_infinite()).then(pl.col("hi")).otherwise(pl.col("lo"))
    hi = pl.when(pl.col("hi").is_infinite()).then(pl.col("lo")).otherwise(pl.col("hi"))
    has10 = ((hi / 10).floor() * 10 >= lo)
    has5 = ((hi / 5).floor() * 5 >= lo)
    legs = legs.with_columns(has10.alias("round10"), has5.alias("round5"))
    # rank of each leg from the bottom of the ladder
    legs = legs.with_columns(pl.col("lo").rank("ordinal").over("event_ticker").alias("ladder_idx"),
                             pl.len().over("event_ticker").alias("n_legs"))
    return legs.filter(pl.col("n_legs") == 6)


def weather_decision_time(code: str, day: date, slot: str) -> datetime:
    off, hour, zone = WEATHER_SLOTS[slot]
    d = day + timedelta(days=off)
    tz = ET if zone == "ET" else ZoneInfo(wx.STATION_TZ[code])
    return datetime(d.year, d.month, d.day, hour, tzinfo=tz).astimezone(UTC)


def weather_points(legs: pl.DataFrame) -> pl.DataFrame:
    ev = legs.select("event_ticker", "code", "day").unique()
    rows = []
    for e, code, day in ev.iter_rows():
        for slot in WEATHER_SLOTS:
            rows.append({"event_ticker": e, "slot": slot, "t": weather_decision_time(code, day, slot)})
    pts = pl.DataFrame(rows, schema={"event_ticker": pl.String, "slot": pl.String, "t": pl.Datetime("us", "UTC")})
    return legs.join(pts, on="event_ticker")


def implied_stats(ev: pl.DataFrame) -> dict[str, Any]:
    """Normalised implied distribution of one ladder snapshot (legs sorted by lo): median, mean,
    and the CDF function on integer temperatures (uniform within each 2-degree bracket; tails get
    a 2-degree width)."""
    lo, hi, p = ev["lo"].to_list(), ev["hi"].to_list(), np.array(ev["mid"].to_list(), dtype=float)
    p = p / p.sum()
    edges = []                                     # (a, b) inclusive integer support for each leg
    for a, b in zip(lo, hi):
        if math.isinf(a):
            edges.append((b - 1, b))
        elif math.isinf(b):
            edges.append((a, a + 1))
        else:
            edges.append((a, b))
    support, mass = [], []
    for (a, b), pi in zip(edges, p):
        vals = list(range(int(a), int(b) + 1))
        support += vals
        mass += [pi / len(vals)] * len(vals)
    s, m = np.array(support, dtype=float), np.array(mass)
    cdf = np.cumsum(m)
    med = float(s[np.searchsorted(cdf, 0.5)])
    return {"imp_med": med, "imp_mean": float((s * m).sum()), "overround": float(ev["mid"].sum())}


def build_weather_panel(markets: pl.DataFrame, candles: pl.LazyFrame, cli: pl.DataFrame) -> pl.DataFrame:
    """One row per (leg, decision slot) with the quote, realised high H, yesterday's high H1, the
    snapshot's implied median, and leg features."""
    legs = weather_legs(markets)
    pts = weather_points(legs)
    q = asof_quotes(pts.select("ticker", "t"), candles)
    p = pts.join(q, on=["ticker", "t"], how="left")
    cli = cli.select("code", "day", pl.col("high").alias("H")).unique(["code", "day"])
    p = p.join(cli, on=["code", "day"], how="left").join(
        cli.select("code", (pl.col("day") + timedelta(days=1)).alias("day"), pl.col("H").alias("H1")),
        on=["code", "day"], how="left")
    # event-level implied stats on fully quoted snapshots
    full = p.filter(pl.col("mid").is_not_null()).group_by("event_ticker", "slot").agg(pl.len().alias("nq"))
    p = p.join(full, on=["event_ticker", "slot"], how="left")
    rows = []
    for (e, s), g in p.filter(pl.col("nq") == 6).group_by(["event_ticker", "slot"]):
        rows.append({"event_ticker": e, "slot": s, **implied_stats(g.sort("lo"))})
    imp = pl.DataFrame(rows, schema={"event_ticker": pl.String, "slot": pl.String, "imp_med": pl.Float64,
                                     "imp_mean": pl.Float64, "overround": pl.Float64})
    p = p.join(imp, on=["event_ticker", "slot"], how="left")
    p = p.with_columns(
        pl.col("mid").rank("ordinal", descending=True).over("event_ticker", "slot").alias("mid_rank"),
        (pl.col("vol24") / pl.col("vol24").sum().over("event_ticker", "slot")).alias("vol_share"),
        (pl.col("mid") / pl.col("mid").sum().over("event_ticker", "slot")).alias("mid_share"),
        pl.col("t").dt.convert_time_zone("America/New_York").dt.date().alias("et_date"),
        (pl.col("day") >= HOLDOUT_FROM.date()).alias("holdout"),
        pl.col("day").dt.month().alias("month"),
    )
    # modal bracket and its neighbours
    modal = p.filter(pl.col("mid_rank") == 1).select("event_ticker", "slot", pl.col("ladder_idx").alias("modal_idx"))
    p = p.join(modal, on=["event_ticker", "slot"], how="left").with_columns(
        (pl.col("ladder_idx") - pl.col("modal_idx")).alias("dist_modal"))
    return p


# ---- culture panel -------------------------------------------------------------------------------
FANDOM_NAMES = [
    # pop / organised stan fandoms
    "Taylor Swift", "Beyonce", "Beyoncé", "Ariana Grande", "Lady Gaga", "Nicki Minaj", "Harry Styles", "One Direction",
    "Justin Bieber", "Selena Gomez", "Olivia Rodrigo", "Sabrina Carpenter", "Billie Eilish", "Chappell Roan",
    "Charli xcx", "SZA", "Doja Cat", "Rihanna", "Miley Cyrus", "Dua Lipa", "Tate McRae", "Katy Perry", "Lana Del Rey",
    "Ed Sheeran",
    # K-pop
    "BTS", "Jungkook", "Jimin", "Jin", "Suga", "Agust D", "RM", "J-Hope", "BLACKPINK", "Lisa", "Jennie", "ROSÉ",
    "Jisoo", "Stray Kids", "SEVENTEEN", "TWICE", "ENHYPEN", "ATEEZ", "TXT", "Tomorrow X Together", "NewJeans", "aespa",
    "LE SSERAFIM", "ILLIT", "KATSEYE", "IVE", "NCT", "EXO", "BABYMONSTER", "ZEROBASEONE", "BOYNEXTDOOR", "RIIZE",
    "KPop Demon Hunters", "HUNTR/X",
    # hip-hop / rap
    "Drake", "Kendrick Lamar", "Travis Scott", "Kanye West", "Ye", "Playboi Carti", "NBA YoungBoy",
    "YoungBoy Never Broke Again", "Bad Bunny", "The Weeknd", "Future", "Lil Uzi Vert", "Juice WRLD",
    # country
    "Morgan Wallen", "Zach Bryan",
    # screen / game franchises
    "Stranger Things", "Wednesday", "Squid Game", "Bridgerton", "Marvel", "Avengers", "Star Wars", "Harry Potter",
    "One Piece", "Grand Theft Auto", "GTA", "Pokemon", "Pokémon", "Minecraft", "Hollow Knight",
]
# Case-sensitive for short all-caps acronyms / ambiguous words (BTS, RM, IVE, TXT, EXO, NCT, Ye, Jin, Lisa, Suga,
# Future, Wednesday, TWICE, GTA, ROSÉ); case-insensitive for the rest.
_CASE_SENSITIVE = {"BTS", "RM", "IVE", "TXT", "EXO", "NCT", "Ye", "Jin", "Lisa", "Suga", "Future", "Wednesday", "TWICE",
                   "GTA", "ROSÉ", "SZA", "SEVENTEEN", "ENHYPEN", "ATEEZ", "ILLIT", "KATSEYE", "RIIZE", "Jennie", "Jisoo"}


def fandom_regex() -> tuple[str, str]:
    cs = [re.escape(n) for n in FANDOM_NAMES if n in _CASE_SENSITIVE]
    ci = [re.escape(n) for n in FANDOM_NAMES if n not in _CASE_SENSITIVE]
    return r"(?:^|[^A-Za-z0-9])(?:" + "|".join(cs) + r")(?:$|[^A-Za-z0-9])", \
        r"(?i)(?:^|[^a-z0-9])(?:" + "|".join(ci) + r")(?:$|[^a-z0-9])"


def fandom_expr(text: pl.Expr) -> pl.Expr:
    cs, ci = fandom_regex()
    return text.fill_null("").str.contains(cs) | text.fill_null("").str.contains(ci)


# ---- fetch (public Kalshi API, <= 1 request/s, sequential; raw responses cached append-only) ----------
FAMILY = ("Entertainment", "Science and Technology", "Climate and Weather")
CULTURE = ("Entertainment", "Science and Technology")
NOV_FROM = datetime(2026, 11, 1, tzinfo=UTC)
MARKET_COLS = ["ticker", "event_ticker", "series_ticker", "category", "title", "yes_sub_title", "market_type",
               "open_time", "close_time", "settlement_ts", "expected_expiration_time", "can_close_early", "result",
               "volume", "rules_primary", "fee_type", "fee_multiplier"]


def _api(min_interval: float = 1.0):
    from albert.config import settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    s = settings()
    return k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=min_interval, max_retries=12,
                                 sleep=lambda x: time.sleep(min(x, 20.0))))


def listing_rows(payload: dict[str, Any], info: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """``albert.venues.kalshi.market_rows`` plus the strike fields (for ladder parsing)."""
    from albert.venues import kalshi as k

    rows = k.market_rows(payload, info, "live")
    extra = {m["ticker"]: m for m in payload.get("markets", [])}
    for r in rows:
        m = extra.get(r["ticker"], {})
        r["strike_type"] = m.get("strike_type")
        r["floor_strike"] = k.num(m.get("floor_strike"))
        r["cap_strike"] = k.num(m.get("cap_strike"))
    return rows


def fetch_listing(start: datetime = NOV_FROM, end: datetime = HOLDOUT_TO) -> pl.DataFrame:
    """Every settled single-leg market that closed in [start, end) (all categories are paged, so the
    sample is volume-blind); keeps this study's categories."""
    api = _api()
    series = pl.read_parquet(TABLES / "kalshi_series.parquet")
    info = {r["series_ticker"]: r for r in series.iter_rows(named=True)}
    rows: list[dict[str, Any]] = []
    pages = 0
    for page in api.settled_markets(start, end):
        pages += 1
        rows += [r for r in listing_rows(page.payload, info) if r["category"] in FAMILY]
        if pages % 25 == 0:
            print(f"  listing: {pages} pages, {len(rows)} family markets", flush=True)
    df = pl.DataFrame(rows, infer_schema_length=None)
    df = df.filter(pl.col("result").is_in(["yes", "no"]) & (pl.col("market_type") == "binary"))
    OUT.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT / "markets_nov.parquet")
    print(f"listing done: {pages} pages, {df.height} markets "
          f"{dict(df.group_by('category').len().iter_rows())}", flush=True)
    return df


def table_markets(categories: tuple[str, ...] = FAMILY) -> pl.DataFrame:
    """Aug 15 - Oct 31 settled binaries of this study's categories from the shared markets table."""
    return (pl.scan_parquet(TABLES / "kalshi_markets.parquet").select(MARKET_COLS)
            .filter(pl.col("category").is_in(list(categories)) & pl.col("result").is_in(["yes", "no"])
                    & (pl.col("market_type") == "binary")).collect())


def all_markets(categories: tuple[str, ...] = FAMILY) -> pl.DataFrame:
    a = table_markets(categories)
    p = OUT / "markets_nov.parquet"
    if p.exists():
        b = pl.read_parquet(p).filter(pl.col("category").is_in(list(categories)))
        a = pl.concat([a, b.select([c for c in MARKET_COLS if c in b.columns])], how="diagonal_relaxed")
    return a.unique("ticker", keep="last")


WINDOW = timedelta(days=21)          # candles cover at most the final 21 days before close


def candle_plan(m: pl.DataFrame) -> pl.DataFrame:
    """Hourly windows over each market's life, capped at the final ``WINDOW`` before close."""
    m = m.filter(pl.col("open_time").is_not_null() & (pl.col("close_time") > pl.col("open_time")))
    return m.with_columns(
        pl.max_horizontal("open_time", pl.col("close_time") - WINDOW).alias("lo"),
        pl.col("close_time").alias("hi"), pl.lit(60).alias("period"),
        pl.col("settlement_ts").fill_null(pl.col("close_time")))


def pack_jobs(plan: pl.DataFrame, cutoff: datetime, max_candles: int = 9500, max_tickers: int = 50) -> list[dict[str, Any]]:
    """Historical markets one request each (in ``plan`` order: largest volume first); live ones
    packed across events into batch requests whose shared window x tickers stays under the API cap
    (markets sorted by window start so neighbours share most of their window)."""
    jobs: list[dict[str, Any]] = [{"kind": "hist", "rows": [r]}
                                  for r in plan.filter(pl.col("settlement_ts") < cutoff).iter_rows(named=True)]
    live = plan.filter(pl.col("settlement_ts") >= cutoff).sort("lo", "hi")
    cur: list[dict[str, Any]] = []
    lo = hi = None
    for r in live.iter_rows(named=True):
        nlo = r["lo"] if lo is None else min(lo, r["lo"])
        nhi = r["hi"] if hi is None else max(hi, r["hi"])
        span = math.ceil((nhi - nlo).total_seconds() / 3600) + 1
        if cur and (span * (len(cur) + 1) > max_candles or len(cur) >= max_tickers):
            jobs.append({"kind": "batch", "rows": cur})
            cur, nlo, nhi = [], r["lo"], r["hi"]
        cur.append(r)
        lo, hi = nlo, nhi
    if cur:
        jobs.append({"kind": "batch", "rows": cur})
    return jobs


def fetch_candles(plan: pl.DataFrame, name: str, *, save_every: int = 50, max_requests: int | None = None) -> None:
    """Sequential hourly-candle fetch into ``OUT/candles_<name>/part-*.parquet`` (resumable)."""
    from albert.research import xcat

    d = OUT / f"candles_{name}"
    d.mkdir(parents=True, exist_ok=True)
    done_path = d / "done.parquet"
    done = set(pl.read_parquet(done_path)["ticker"]) if done_path.exists() else set()
    todo = plan.filter(~pl.col("ticker").is_in(list(done)))
    api = _api()
    cutoff = api.cutoff()["market_settled_ts"]
    jobs = pack_jobs(todo, cutoff)
    if max_requests:
        jobs = jobs[:max_requests]
    print(f"{name}: {plan.height} markets, {len(done)} done, {todo.height} to fetch in {len(jobs)} jobs", flush=True)
    buf: list[dict[str, Any]] = []
    fin: list[str] = []
    part = len(list(d.glob("part-*.parquet")))
    t0 = time.time()

    def flush() -> None:
        nonlocal buf, part
        if buf:
            pl.DataFrame(buf, schema=xcat.OHLC_SCHEMA).write_parquet(d / f"part-{part:05d}.parquet")
            part += 1
        buf = []
        pl.DataFrame({"ticker": sorted(done | set(fin))}, schema={"ticker": pl.String}).write_parquet(done_path)

    for i, job in enumerate(jobs, 1):
        for tk, got in xcat._run_job(api, job):
            if got is not None:
                buf.extend(got)
                fin.append(tk)
        if i % save_every == 0:
            flush()
            print(f"  {name} {i}/{len(jobs)} jobs, {len(fin)} markets, {(time.time() - t0) / 60:.1f} min", flush=True)
    flush()
    print(f"{name} done: {len(fin)} markets", flush=True)


def load_candles(names: tuple[str, ...] = ("live", "hist")) -> pl.LazyFrame:
    """Hourly candles from xcat (hourly layer) and this study's fetches, in the ``kalshi_candles``
    column names used by ``asof_quotes``."""
    paths = [str(p) for p in sorted((XCAT / "hourly").glob("part-*.parquet"))]
    for n in names:
        paths += [str(p) for p in sorted((OUT / f"candles_{n}").glob("part-*.parquet"))]
    return (pl.scan_parquet(paths).select("ticker", "end_ts", pl.col("bid_close").alias("yes_bid_close"),
                                         pl.col("ask_close").alias("yes_ask_close"), pl.col("p_close").alias("price_close"),
                                         "volume", pl.col("oi").alias("open_interest")))


# ---- culture panel -------------------------------------------------------------------------------
_NUM = r"\$?(-?[\d,]*\.?\d+)\s*([KMB]?)"
_LADDER_RX = re.compile(r"(?i)^(above|below|at least|at most|more than|less than|over|under|exactly|between)?\s*"
                        + _NUM + r"\s*(\+|or (?:more|above|higher|below|less|fewer|lower))?")
_MULT = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9}


def parse_strike(sub: str | None) -> tuple[str | None, float | None]:
    """``"Above 9M"`` -> ("above", 9e6); ``"At least 5 weeks"`` -> ("above", 5); ``"Exactly 9 songs"``
    -> ("exact", 9); names -> (None, None)."""
    s = (sub or "").strip()
    m = _LADDER_RX.match(s)
    if not m or (m.group(1) is None and m.group(4) is None):
        return None, None
    try:
        v = float(m.group(2).replace(",", "")) * _MULT[(m.group(3) or "").upper()]
    except ValueError:
        return None, None
    w = (m.group(1) or "").lower()
    tail = (m.group(4) or "").lower()
    if w in ("above", "at least", "more than", "over") or any(x in tail for x in ("more", "above", "higher", "+")):
        return "above", v
    if w in ("below", "at most", "less than", "under") or any(x in tail for x in ("below", "less", "fewer", "lower")):
        return "below", v
    if w == "exactly":
        return "exact", v
    return None, None


def scheduled_close(m: pl.DataFrame) -> pl.DataFrame:
    """Scheduled close known in advance: close_time for markets that cannot close early; otherwise
    expected_expiration_time minus the series' median (expected_expiration - close) offset."""
    off = (m.filter(pl.col("expected_expiration_time").is_not_null())
           .group_by("series_ticker")
           .agg((pl.col("expected_expiration_time") - pl.col("close_time")).median().alias("off")))
    m = m.join(off, on="series_ticker", how="left")
    sched = (pl.when(pl.col("can_close_early").not_() | pl.col("expected_expiration_time").is_null())
             .then(pl.col("close_time"))
             .otherwise(pl.col("expected_expiration_time") - pl.col("off").fill_null(timedelta(0))))
    return m.with_columns(sched.alias("sched_close")).drop("off")


CULTURE_SLOTS = ("close-24h", "open+24h")


def culture_points(m: pl.DataFrame) -> pl.DataFrame:
    m = scheduled_close(m)
    a = m.with_columns(pl.lit("close-24h").alias("slot"), (pl.col("sched_close") - timedelta(hours=24)).alias("t"))
    # open + 24 h only for markets scheduled to live <= WINDOW (candles cover the final WINDOW only)
    b = (m.filter(pl.col("sched_close") - pl.col("open_time") <= WINDOW)
         .with_columns(pl.lit("open+24h").alias("slot"), (pl.col("open_time") + timedelta(hours=24)).alias("t")))
    pts = pl.concat([a, b])
    # the market must be open at t (known then); the scheduled close must be >= 1 h away (known ex ante)
    return pts.filter((pl.col("t") > pl.col("open_time")) & (pl.col("t") < pl.col("close_time"))
                      & (pl.col("t") < pl.col("sched_close") - timedelta(hours=1)))


def build_culture_panel(m: pl.DataFrame, candles: pl.LazyFrame) -> pl.DataFrame:
    """One row per (leg, slot): quote, outcome, and the behavioural features."""
    pts = culture_points(m)
    q = asof_quotes(pts.select("ticker", "t"), candles)
    p = pts.join(q, on=["ticker", "t"], how="left").filter(pl.col("mid").is_not_null())
    st = [parse_strike(s) for s in p["yes_sub_title"].to_list()]
    p = p.with_columns(pl.Series("strike_dir", [a for a, _ in st], dtype=pl.String),
                       pl.Series("strike", [b for _, b in st], dtype=pl.Float64))
    p = p.with_columns(
        (pl.col("result") == "yes").cast(pl.Float64).alias("y"),
        fandom_expr(pl.col("title").fill_null("") + " | " + pl.col("yes_sub_title").fill_null("")).alias("fandom"),
        pl.col("strike_dir").is_not_null().alias("is_ladder"),
        pl.len().over("event_ticker", "slot").alias("n_legs_q"),
        (pl.col("vol24") / pl.col("vol24").sum().over("event_ticker", "slot")).alias("vol_share"),
        (pl.col("mid") / pl.col("mid").sum().over("event_ticker", "slot")).alias("mid_share"),
        pl.col("mid").sum().over("event_ticker", "slot").alias("event_mid_sum"),
        pl.col("t").dt.convert_time_zone("America/New_York").dt.date().alias("et_date"),
        (pl.col("t") >= HOLDOUT_FROM).alias("holdout"),
        ((pl.col("t") < HOLDOUT_FROM) & (pl.col("close_time") < HOLDOUT_FROM)).alias("dev"),
    )
    # round-number strike: a leading digit followed only by zeros (10, 100, 5M = no; 10M, 100K = yes)
    mant = pl.col("strike") / (10 ** pl.col("strike").abs().log10().floor())
    p = p.with_columns(((mant - mant.round(0)).abs() < 1e-9).alias("strike_1sig"),
                       ((mant - 1).abs() < 1e-9).alias("strike_pow10"))
    return p


def fetch_plans() -> dict[str, pl.DataFrame]:
    """``live``: every Entertainment / Science-Tech market settled after the historical cutoff that
    xcat lacks, plus November KXHIGH ladders (batch endpoint). ``hist``: the remaining pre-cutoff
    Entertainment markets (one request each), largest volume first so a truncated run keeps the
    markets most likely to have quotes."""

    have = set(pl.read_parquet(XCAT / "hourly" / "done.parquet")["ticker"])
    cut = datetime(2026, 9, 18, tzinfo=UTC)
    cul = all_markets(CULTURE).filter(~pl.col("ticker").is_in(list(have)))
    nov = pl.read_parquet(OUT / "markets_nov.parquet") if (OUT / "markets_nov.parquet").exists() else cul.head(0)
    wx_nov = nov.filter(pl.col("series_ticker").str.starts_with("KXHIGH")).select([c for c in MARKET_COLS if c in nov.columns])
    plan = candle_plan(pl.concat([cul, wx_nov], how="diagonal_relaxed"))
    live = plan.filter(pl.col("settlement_ts") >= cut)
    hist = plan.filter(pl.col("settlement_ts") < cut).sort("volume", descending=True, nulls_last=True)
    return {"live": live, "hist": hist}


def fetch_all(max_hist: int | None = None) -> None:
    plans = fetch_plans()
    fetch_candles(plans["live"], "live")
    fetch_candles(plans["hist"], "hist", max_requests=max_hist)


# ---- trades, scores, ledgers ---------------------------------------------------------------------
STAKE = 5.0            # $ per position (the capped book allows <= $5 per event, <= $100 open)
EVENT_CAP = 5.0
BANKROLL = 100.0


def take(f: pl.DataFrame, side: str, stake: float = STAKE) -> pl.DataFrame:
    """Taker entries held to settlement: YES at the ask, NO at 1 - bid; n = floor(stake / price);
    fee per order rounded up to the cent. Per-contract columns: price, fee_c, pnl, risk, gross (vs mid)."""
    f = f.filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("bid") > 0) & (pl.col("ask") < 1))
    yes = side == "yes"
    f = f.with_columns(
        (pl.col("ask") if yes else 1 - pl.col("bid")).alias("price"),
        (pl.col("y") if yes else 1 - pl.col("y")).alias("payoff"),
        (pl.col("mid") if yes else 1 - pl.col("mid")).alias("mid_side"),
        pl.lit(side).alias("side"))
    f = f.with_columns(pl.max_horizontal(pl.lit(1.0), (stake / pl.col("price")).floor()).alias("n"))
    f = f.with_columns(taker_fee_expr(pl.col("n"), pl.col("price")).alias("fee_order"))
    return f.with_columns(
        (pl.col("fee_order") / pl.col("n")).alias("fee_c"),
        (pl.col("payoff") - pl.col("price") - pl.col("fee_order") / pl.col("n")).alias("pnl"),
        (pl.col("price") + pl.col("fee_order") / pl.col("n")).alias("risk"),
        (pl.col("payoff") - pl.col("mid_side")).alias("gross"),
        (pl.col("n") * (pl.col("payoff") - pl.col("price")) - pl.col("fee_order")).alias("pnl_usd"),
        (pl.col("n") * pl.col("price") + pl.col("fee_order")).alias("risk_usd"))


def score(tr: pl.DataFrame, **labels: Any) -> dict[str, Any]:
    """Per-contract net (c), t by event and by ET date, per-$ return, hit rate (for reference only)."""
    if tr.is_empty():
        return {**labels, "trades": 0}
    c, t_ev, k = cl_mean(tr, "pnl", "event_ticker")
    _, t_dt, kd = cl_mean(tr, "pnl", "et_date")
    g, tg, _ = cl_mean(tr, "gross", "event_ticker")
    d, td, _ = cl_mean(tr, "pnl_usd", "event_ticker", weight="risk_usd")
    return {**labels, "trades": tr.height, "events": k, "dates": kd, "avg_price": _r(tr["price"].mean(), 3),
            "gross_c": _r(100 * g), "t_gross": _r(tg), "net_c": _r(100 * c), "t_event": _r(t_ev), "t_date": _r(t_dt),
            "net_per_$": _r(d, 4), "t_$": _r(td), "pnl_usd": _r(tr["pnl_usd"].sum()), "hit": _r(tr["payoff"].mean(), 3)}


def capped_ledger(tr: pl.DataFrame, strategy: str, variant: str, split: str,
                  bankroll: float = BANKROLL, event_cap: float = EVENT_CAP) -> pl.DataFrame:
    """Trades in time order on a $bankroll book: taken only while open cost stays within the bankroll
    and the event's open cost within ``event_cap``; capital returns at settlement."""
    import heapq

    heap: list[tuple[Any, float, str]] = []
    used, per_ev, rows = 0.0, {}, []
    for r in tr.sort(["t", "ticker"]).iter_rows(named=True):     # ties broken by ticker (deterministic)
        while heap and heap[0][0] <= r["t"]:
            _, c, e = heapq.heappop(heap)
            used -= c
            per_ev[e] -= c
        cost = r["risk_usd"]
        if used + cost > bankroll + 1e-9 or per_ev.get(r["event_ticker"], 0.0) + cost > event_cap + 1e-9:
            continue
        settle = max(r["settlement_ts"], r["t"])
        used += cost
        per_ev[r["event_ticker"]] = per_ev.get(r["event_ticker"], 0.0) + cost
        heapq.heappush(heap, (settle, cost, r["event_ticker"]))
        rows.append({"strategy": strategy, "variant": variant, "split": split,
                     "struct_id": f"{r['ticker']}|{r['t'].isoformat()}", "event_ticker": r["event_ticker"],
                     "ticker": r["ticker"], "side": r["side"], "n": int(r["n"]), "t_in": r["t"], "cost_usd": cost,
                     "settle_ts": settle, "payoff_usd": r["n"] * r["payoff"]})
    return pl.DataFrame(rows, schema={"strategy": pl.String, "variant": pl.String, "split": pl.String,
                                      "struct_id": pl.String, "event_ticker": pl.String, "ticker": pl.String,
                                      "side": pl.String, "n": pl.Int64, "t_in": pl.Datetime("us", "UTC"),
                                      "cost_usd": pl.Float64, "settle_ts": pl.Datetime("us", "UTC"),
                                      "payoff_usd": pl.Float64})


def marks_for(tickers: list[str], candles: pl.LazyFrame) -> pl.DataFrame:
    """Hourly YES mids (bid/ask midpoint if two-sided, else last trade) for ``vol_metrics``."""
    c = candles.filter(pl.col("ticker").is_in(tickers)).collect()
    two = (pl.col("yes_bid_close").is_not_null() & pl.col("yes_ask_close").is_not_null()
           & (pl.col("yes_ask_close") > pl.col("yes_bid_close")))
    mid = pl.when(two).then((pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2).otherwise(pl.col("price_close"))
    return (c.with_columns(mid.alias("yes_mid")).filter(pl.col("yes_mid").is_not_null())
            .select("ticker", pl.col("end_ts").alias("ts"), "yes_mid").unique(["ticker", "ts"]).sort("ticker", "ts"))


def ledger_stats(led: pl.DataFrame, candles: pl.LazyFrame | None = None) -> dict[str, Any]:
    from albert.research.vol_metrics import ledger_metrics

    if led.is_empty():
        return {"positions": 0}
    tk = led["ticker"].unique().to_list()
    m = ledger_metrics(led, marks_for(tk, candles if candles is not None else all_candles(tk)))
    keep = ("positions", "events", "pnl_usd", "usd_per_week", "ann_return_pct", "ann_vol_pct", "sharpe", "sharpe_ci90",
            "p_sharpe_pos", "sharpe_ex_top5", "usd_per_week_ex_top5", "max_dd_usd", "weeks_positive_pct", "skew",
            "first_day", "last_day", "marked_share")
    return {k: m.get(k) for k in keep}


# ---- Wikimedia pageviews (H3; wikimedia.org serves no robots.txt) --------------------------------
WIKI_BASE = "https://wikimedia.org/api/rest_v1"
WIKI_UA = "albert-research/0.2 (quantitative research; public data only)"
PV_FROM, PV_TO = "20260601", "20261117"
YT_ARTIST_ARTICLE = {
    "Drake": "Drake_(musician)", "Ariana Grande": "Ariana_Grande", "Bad Bunny": "Bad_Bunny",
    "Taylor Swift": "Taylor_Swift", "Morgan Wallen": "Morgan_Wallen", "Justin Bieber": "Justin_Bieber",
    "YoungBoy Never Broke Again": "YoungBoy_Never_Broke_Again", "NBA YoungBoy": "YoungBoy_Never_Broke_Again",
    "YoungBoy": "YoungBoy_Never_Broke_Again", "Future": "Future_(rapper)", "The Weeknd": "The_Weeknd",
    "Fuerza Regida": "Fuerza_Regida", "KATSEYE": "Katseye", "Kanye West": "Kanye_West",
    "Olivia Rodrigo": "Olivia_Rodrigo", "Post Malone": "Post_Malone", "Kendrick Lamar": "Kendrick_Lamar",
    "Tate McRae": "Tate_McRae",
}


def pageview_rows(payload: dict[str, Any], title: str) -> list[dict[str, Any]]:
    out = []
    for it in payload.get("items") or []:
        ts = str(it.get("timestamp") or "")
        if len(ts) >= 8:
            out.append({"title": title, "day": date(int(ts[:4]), int(ts[4:6]), int(ts[6:8])),
                        "views": float(it.get("views") or 0)})
    return out


def fetch_pageviews(titles: list[str] | None = None) -> pl.DataFrame:
    from urllib.parse import quote

    from albert.config import settings
    from albert.http import CachedClient

    titles = sorted(set(titles or YT_ARTIST_ARTICLE.values()))
    c = CachedClient("wikimedia", WIKI_BASE, settings().raw_dir, min_interval=3.0, max_retries=8, user_agent=WIKI_UA)
    rows: list[dict[str, Any]] = []
    for t in titles:
        path = f"/metrics/pageviews/per-article/en.wikipedia/all-access/user/{quote(t, safe='')}/daily/{PV_FROM}/{PV_TO}"
        got = pageview_rows(c.get(path).payload, t)
        print(f"  {t}: {len(got)} days", flush=True)
        rows += got
    df = pl.DataFrame(rows, schema={"title": pl.String, "day": pl.Date, "views": pl.Float64})
    df.write_parquet(OUT / "pageviews.parquet")
    return df


def pageview_spike(points: pl.DataFrame, pv: pl.DataFrame) -> pl.DataFrame:
    """Spike ratio per (title, t): mean views over the 3 full days ending >= 24 h before t, over the
    mean of the 28 days before those. Only days that ended before t - 24 h enter."""
    pv = pv.sort("title", "day")
    out = []
    for title, t in points.select("title", "t").unique().iter_rows():
        last = (t - timedelta(hours=24)).date() - timedelta(days=1)      # the last day fully ended by t - 24 h
        g = pv.filter((pl.col("title") == title) & (pl.col("day") <= last) & (pl.col("day") > last - timedelta(days=31)))
        rec = g.filter(pl.col("day") > last - timedelta(days=3))["views"]
        base = g.filter(pl.col("day") <= last - timedelta(days=3))["views"]
        ratio = float(rec.mean()) / float(base.mean()) if len(rec) == 3 and len(base) >= 20 and base.mean() else None
        out.append({"title": title, "t": t, "pv_spike": ratio})
    return pl.DataFrame(out, schema={"title": pl.String, "t": pl.Datetime("us", "UTC"), "pv_spike": pl.Float64})


# ---- panel drivers -------------------------------------------------------------------------------
def all_candles(tickers: list[str] | pl.Series | None = None) -> pl.LazyFrame:
    """Shared table candles (weather and the rest, Aug 15 - Oct 31) + xcat + this study's fetches.
    ``tickers`` filters every source before the union (keeps memory low)."""
    tab = (pl.scan_parquet(TABLES / "kalshi_candles.parquet")
           .select("ticker", "end_ts", "yes_bid_close", "yes_ask_close", "price_close", "volume", "open_interest"))
    srcs = [load_candles().with_columns(pl.lit(0).alias("src")), tab.with_columns(pl.lit(1).alias("src"))]
    if tickers is not None:
        tk = pl.Series(list(tickers), dtype=pl.String).unique().implode()
        srcs = [x.filter(pl.col("ticker").is_in(tk)) for x in srcs]
    # deterministic: a (ticker, end_ts) present in several sources keeps the study's own API fetch
    # (xcat / live / hist) over the shared table
    return (pl.concat(srcs, how="vertical_relaxed").sort("ticker", "end_ts", "src")
            .unique(["ticker", "end_ts"], keep="first", maintain_order=True).drop("src"))


def weather_markets() -> pl.DataFrame:
    a = (pl.scan_parquet(TABLES / "kalshi_markets.parquet").select(MARKET_COLS)
         .filter(pl.col("series_ticker").str.starts_with("KXHIGH")).collect())
    p = OUT / "markets_nov.parquet"
    if p.exists():
        b = pl.read_parquet(p).filter(pl.col("series_ticker").str.starts_with("KXHIGH"))
        a = pl.concat([a, b.select([c for c in MARKET_COLS if c in b.columns])], how="diagonal_relaxed")
    return a.unique("ticker", keep="last")


def build_weather(split: str) -> pl.DataFrame:
    """``dev`` = event days before 2026-10-01; ``holdout`` = 2026-10-01 .. 2026-11-17."""
    m = weather_markets()
    lo, hi = (date(2026, 8, 15), HOLDOUT_FROM.date()) if split == "dev" else (HOLDOUT_FROM.date(), date(2026, 11, 18))
    days = pl.col("event_ticker").map_elements(wxr._event_day, return_dtype=pl.Date)
    m = m.filter(days.is_between(lo, hi, closed="left"))
    cli = pl.read_parquet(TABLES / "wx_cli.parquet")
    p = build_weather_panel(m, all_candles(m["ticker"]), cli)
    return p.filter(pl.col("holdout") == (split == "holdout"))


def build_culture(split: str) -> pl.DataFrame:
    m = all_markets(CULTURE)
    p = build_culture_panel(m, all_candles(m["ticker"]))
    p = p.filter(pl.col("dev")) if split == "dev" else p.filter(pl.col("holdout") & (pl.col("close_time") < HOLDOUT_TO))
    pv_path = OUT / "pageviews.parquet"
    if pv_path.exists():
        pv = pl.read_parquet(pv_path)
        art = (pl.col("title").str.extract(r"^Will (.+?) have above")
               .replace_strict(YT_ARTIST_ARTICLE, default=None, return_dtype=pl.String))
        p = p.with_columns(pl.when(pl.col("series_ticker").str.starts_with("KXYTVIEWS")).then(art).alias("wtitle"))
        sp = pageview_spike(p.filter(pl.col("wtitle").is_not_null()).select(pl.col("wtitle").alias("title"), "t"), pv)
        p = p.join(sp.rename({"title": "wtitle"}), on=["wtitle", "t"], how="left")
    return p.drop("rules_primary")


# ---- descriptive tables (dev exploration and holdout robustness) ---------------------------------
MID_BUCKETS = ["05-15", "15-35", "35-65", "65-85", "85-95"]
AWARD_RX = (r"EMMY|VMA$|GRAMMY|^KXCMA|NOBEL|OSCAR|GOLDENGLOBE|BOOKER|AWARD|TIFF|SEXYMAN|PEOPLESCHOICE|BAFTA"
            r"|GOTHAM|LATINGRAMMY|BALLONDOR|^KXAMAS|^KXEMA|^KXMAMA")
TRIBAL_RX = r"(?i)grok|xai|musk|elon|tesla|spacex|starship"


def with_r(p: pl.DataFrame, max_spread: float = 0.10) -> pl.DataFrame:
    return (p.filter(pl.col("mid").is_not_null() & (pl.col("spread") <= max_spread))
            .with_columns((pl.col("y") - pl.col("mid")).alias("r"), bucket_expr(pl.col("mid")).alias("pb")))


def ymid_table(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Mean y - mid (c) per group with t clustered by event and by ET date."""
    rows = []
    for k, g in df.group_by(by, maintain_order=True):
        m, t, n = cl_mean(g, "r", "event_ticker")
        _, t2, nd = cl_mean(g, "r", "et_date")
        rows.append(dict(zip(by, k)) | {"n": g.height, "events": n, "dates": nd, "mid": _r(g["mid"].mean(), 3),
                                        "y_minus_mid_c": _r(100 * m), "t_event": _r(t), "t_date": _r(t2)})
    return pl.DataFrame(rows).sort(by) if rows else pl.DataFrame()


def fe_reg(df: pl.DataFrame, var: str, label: str, fe: tuple[str, ...] = ("pb",)) -> dict[str, Any]:
    """r = a + b * var + bucket fixed effects; b with t clustered by event and by ET date."""
    d = df.filter(pl.col(var).is_not_null() & pl.col(var).cast(pl.Float64).is_finite())
    if d.height < 20 or d[var].cast(pl.Float64).std() in (None, 0.0):
        return {"test": label, "n": d.height}
    dd = d.to_dummies(list(fe), drop_first=True)
    cols = [c for c in dd.columns if any(c.startswith(f + "_") for f in fe)]
    X = np.c_[np.ones(dd.height), dd[var].cast(pl.Float64).to_numpy(), dd.select(cols).to_numpy().astype(float)]
    y = dd["r"].to_numpy()
    b, se = ols_cluster(y, X, dd["event_ticker"].to_numpy())
    _, se2 = ols_cluster(y, X, dd["et_date"].cast(pl.String).to_numpy())
    return {"test": label, "n": d.height, "events": d["event_ticker"].n_unique(), "coef_c": _r(100 * b[1]),
            "t_event": _r(b[1] / se[1]) if se[1] > 0 else None, "t_date": _r(b[1] / se2[1]) if se2[1] > 0 else None,
            "share": _r(d[var].cast(pl.Float64).mean(), 3)}


def weather_tables(p: pl.DataFrame) -> dict[str, pl.DataFrame]:
    q = with_r(p).with_columns((pl.col("ladder_idx").cast(pl.Int32) - pl.col("modal_idx").cast(pl.Int32)).alias("dm"),
                               (pl.col("vol_share") / pl.col("mid_share")).clip(0.01, 100).log().alias("latt"))
    out = {"wx_position_bucket": ymid_table(q, ["slot", "pos", "pb"]),
           "wx_modal_distance": ymid_table(q, ["slot", "dm"]),
           "wx_round10_interior": ymid_table(q.filter((pl.col("pos") == "interior") & pl.col("pb").is_in(MID_BUCKETS)),
                                             ["slot", "round10", "pb"])}
    regs = []
    for slot in WEATHER_SLOTS:
        s = q.filter((pl.col("slot") == slot) & pl.col("pb").is_in(MID_BUCKETS))
        regs.append({"slot": slot} | fe_reg(s.filter(pl.col("pos") == "interior"), "round10", "H5 round10 (interior)"))
        regs.append({"slot": slot} | fe_reg(s, "round5", "H5 round5 (all legs)", fe=("pb", "pos")))
        regs.append({"slot": slot} | fe_reg(s.filter(pl.col("pos") != "interior"), "latt", "H2 log attention (tails)"))
        regs.append({"slot": slot} | fe_reg(s, "latt", "H2 log attention"))
        tails = s.filter(pl.col("pos") != "interior").with_columns((pl.col("pos") == "high_tail").alias("hot"))
        regs.append({"slot": slot} | fe_reg(tails, "hot", "H4 hot tail vs cold tail"))
        regs.append({"slot": slot} | fe_reg(s.with_columns((pl.col("pos") != "interior").alias("tail")), "tail",
                                            "H8 tail vs interior at equal price"))
    out["wx_regressions"] = pl.DataFrame(regs)
    ev = (q.select("slot", "event_ticker", "code", "day", "month", "H", "H1", "imp_med", "imp_mean")
          .unique(["slot", "event_ticker"]).drop_nulls())
    rows = []
    for slot in WEATHER_SLOTS:
        e = ev.filter(pl.col("slot") == slot).with_columns((pl.col("H") - pl.col("imp_mean")).alias("err"),
                                                           (pl.col("H1") - pl.col("imp_mean")).alias("anch"))
        if e.height < 20:
            continue
        b, se = ols_cluster(e["err"].to_numpy(), np.c_[np.ones(e.height), e["anch"].to_numpy()],
                            e["day"].cast(pl.String).to_numpy())
        for mo, g in e.group_by("month", maintain_order=True):
            rows.append({"slot": slot, "month": mo[0], "events": g.height, "mean_err_F": _r(g["err"].mean()),
                         "p_above_median": _r((g["H"] > g["imp_med"]).mean(), 3),
                         "p_below_median": _r((g["H"] < g["imp_med"]).mean(), 3),
                         "anchor_slope": _r(b[1], 3), "anchor_t_by_day": _r(b[1] / se[1])})
    out["wx_anchoring_drift"] = pl.DataFrame(rows)
    return out


def culture_tables(p: pl.DataFrame) -> dict[str, pl.DataFrame]:
    q = with_r(p)
    ent = q.filter(pl.col("category") == "Entertainment")
    sci = q.filter(pl.col("category") == "Science and Technology").with_columns(
        (pl.col("title").fill_null("") + " " + pl.col("yes_sub_title").fill_null("")).str.contains(TRIBAL_RX).alias("tribal"))
    named = q.filter(~pl.col("is_ladder") & (pl.col("n_legs_q") >= 3)
                     & (pl.col("vol24").sum().over("event_ticker", "slot") > 0)).with_columns(
        (pl.col("vol_share") / pl.col("mid_share")).clip(0.01, 100).log().alias("latt"))
    lad = q.filter(pl.col("strike_dir") == "above").with_columns(
        pl.when(pl.col("series_ticker").str.starts_with("KXYTVIEWS")).then(pl.lit("ytviews"))
        .when(pl.col("category") == "Science and Technology").then(pl.lit("scitech")).otherwise(pl.lit("ent_other")).alias("fam"),
        pl.col("strike").rank("ordinal").over("event_ticker", "slot").alias("srank"),
        pl.len().over("event_ticker", "slot").alias("nl"))
    lad = lad.with_columns((pl.col("srank") == pl.col("nl")).alias("is_top"), (pl.col("srank") == 1).alias("is_bottom"))
    aw = q.filter(pl.col("series_ticker").str.contains(AWARD_RX)).with_columns(
        (pl.col("mid").rank("ordinal", descending=True).over("event_ticker", "slot") == 1).alias("fav"))
    out = {"cul_category_bucket": ymid_table(q, ["category", "slot", "pb"]),
           "cul_fandom_bucket": ymid_table(ent, ["fandom", "pb"]),
           "cul_fandom_series": ymid_table(ent.filter(pl.col("fandom")), ["series_ticker"]),
           "cul_tribal_bucket": ymid_table(sci, ["tribal", "pb"]),
           "cul_ladder_family_bucket": ymid_table(lad, ["fam", "pb"]),
           "cul_awards_fav": ymid_table(aw, ["slot", "fav"])}
    if "pv_spike" in q.columns:
        yt = q.filter(pl.col("pv_spike").is_not_null()).with_columns(pl.col("pv_spike").log().alias("lsp"))
    else:
        yt = q.head(0).with_columns(pl.lit(None, dtype=pl.Float64).alias("lsp"))
    regs = [fe_reg(ent, "fandom", "H1 fandom (Ent, all buckets)"),
            fe_reg(ent.filter(pl.col("pb").is_in(MID_BUCKETS)), "fandom", "H1 fandom (Ent, 5-95c)"),
            fe_reg(ent.filter(~pl.col("is_ladder")), "fandom", "H1 fandom (Ent named legs)"),
            fe_reg(sci, "tribal", "H19 Musk/xAI tribalism (SciTech)"),
            fe_reg(named, "latt", "H2 log volume attention (named legs)"),
            fe_reg(named.filter(pl.col("pb").is_in(MID_BUCKETS)), "latt", "H2 log volume attention (named, 5-95c)"),
            fe_reg(lad.filter(~pl.col("is_bottom") & (pl.col("nl") >= 3)), "is_top", "H8 top strike vs interior"),
            fe_reg(lad.filter(pl.col("pb").is_in(MID_BUCKETS)), "strike_1sig", "H5 round strike (culture ladders)"),
            fe_reg(yt, "lsp", "H3 log pageview spike (YT views)"),
            fe_reg(aw, "fav", "H12 award favourite vs rest")]
    out["cul_regressions"] = pl.DataFrame(regs)
    return out


# ---- dev driver ----------------------------------------------------------------------------------
def _save_tables(tables: dict[str, pl.DataFrame], prefix: str) -> None:
    d = OUT / "tables"
    d.mkdir(parents=True, exist_ok=True)
    for k, v in tables.items():
        if v.height:
            v.write_csv(d / f"{prefix}_{k}.csv")


def run_dev() -> None:
    """Build the dev panels and write every exploration table (data before 2026-10-01 only)."""
    w = build_weather("dev")
    w.write_parquet(OUT / "weather_panel_dev.parquet")
    _save_tables(weather_tables(w), "dev")
    c = build_culture("dev")
    c.write_parquet(OUT / "culture_panel_dev.parquet")
    _save_tables(culture_tables(c), "dev")
    print(f"dev panels: weather {w.height} rows / {w['event_ticker'].n_unique()} events; "
          f"culture {c.height} rows / {c['event_ticker'].n_unique()} events", flush=True)


# ---- candidate rules -----------------------------------------------------------------------------
def attention(p: pl.DataFrame) -> pl.DataFrame:
    """Named (non-ladder) legs of events with >= 3 quoted legs and some volume in the last 24 h, with
    att = (leg's share of the event's 24 h volume) / (leg's share of the event's summed mid)."""
    return (p.filter(~pl.col("is_ladder") & (pl.col("n_legs_q") >= 3)
                     & (pl.col("vol24").sum().over("event_ticker", "slot") > 0))
            .with_columns((pl.col("vol_share") / pl.col("mid_share")).alias("att")))


def rule_trades(p: pl.DataFrame, *, kind: str, side: str, slots: tuple[str, ...], band: tuple[float, float],
                max_spread: float = 0.10, att_min: float | None = None, att_max: float | None = None,
                fandom: bool | None = None, categories: tuple[str, ...] = CULTURE) -> pl.DataFrame:
    """Entries of one rule on a culture panel, as taker trades held to settlement."""
    f = attention(p) if kind == "attention" else p          # shares are computed on the whole event first
    f = f.filter(pl.col("category").is_in(list(categories)) & pl.col("slot").is_in(list(slots))
                 & pl.col("mid").is_between(*band) & (pl.col("spread") <= max_spread))
    if kind == "attention":
        if att_min is not None:
            f = f.filter(pl.col("att") >= att_min)
        if att_max is not None:
            f = f.filter(pl.col("att") <= att_max)
    if fandom is not None:
        f = f.filter(pl.col("fandom") == fandom)
    return take(f, side)


# ---- report helpers ------------------------------------------------------------------------------
def md_table(df: pl.DataFrame, cols: list[str] | None = None, max_rows: int = 60) -> str:
    if df is None or df.is_empty():
        return "(no rows)\n"
    cols = [c for c in (cols or df.columns) if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.select(cols).head(max_rows).iter_rows():
        lines.append("| " + " | ".join("" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines) + "\n"


# ---- evaluation ----------------------------------------------------------------------------------
def robustness(tr: pl.DataFrame, rule: str, split: str, family: pl.Expr) -> pl.DataFrame:
    """Scores of one rule's trades cut by sub-category family, price bucket and month."""
    tr = tr.with_columns(family.alias("family"), bucket_expr(pl.col("mid")).alias("pb"),
                         pl.col("t").dt.strftime("%Y-%m").alias("month"))
    rows = []
    for col in ("family", "pb", "month", "slot"):
        if col not in tr.columns:
            continue
        for k, g in tr.group_by(col, maintain_order=True):
            rows.append(score(g, rule=rule, split=split, cut=col, value=str(k[0])))
    return pl.DataFrame(rows, infer_schema_length=None)


def evaluate(trades: dict[str, pl.DataFrame], rule: str, family: pl.Expr, candles: pl.LazyFrame | None = None,
             ledger_dir: Path | None = None) -> dict[str, Any]:
    """Primary scores per split, robustness cuts, and capped-book ledger metrics."""
    out: dict[str, Any] = {"rule": rule, "scores": [], "robust": [], "ledgers": []}
    for split, tr in trades.items():
        out["scores"].append(score(tr, rule=rule, split=split))
        if tr.height:
            out["robust"].append(robustness(tr, rule, split, family))
            led = capped_ledger(tr, rule, "primary", split)
            if ledger_dir is not None:
                ledger_dir.mkdir(parents=True, exist_ok=True)
                led.write_parquet(ledger_dir / f"ledger_{rule}_{split}.parquet")
            out["ledgers"].append({"rule": rule, "split": split} | ledger_stats(led, candles))
    return out


# ---- pre-registered rules (frozen in hypotheses.md / preregistered.json before the holdout) --------
PREREG = OUT / "preregistered.json"


def r1_fandom_no(p: pl.DataFrame, fandom: bool = True, band: tuple[float, float] = (0.02, 0.35)) -> pl.DataFrame:
    """R1 (fandom=True) and its control C1 (fandom=False): Entertainment legs at either decision slot
    with YES mid in [2c, 35c] and spread <= 10c; buy NO at 1 - bid (taker), held to settlement."""
    f = p.filter((pl.col("category") == "Entertainment") & (pl.col("fandom") == fandom)
                 & pl.col("mid").is_between(*band) & (pl.col("spread") <= 0.10))
    return take(f, "no")


def r1_family() -> pl.Expr:
    return pl.when(pl.col("series_ticker").str.starts_with("KXYTVIEWS")).then(pl.lit("youtube_views")) \
        .otherwise(pl.lit("other_entertainment"))


def d1_round10(p: pl.DataFrame) -> dict[str, Any]:
    """D1: weather interior brackets 5-85c, both slots: y - mid on round10 with bucket and slot FE."""
    q = with_r(p).filter((pl.col("pos") == "interior") & pl.col("mid").is_between(0.05, 0.85))
    return fe_reg(q, "round10", "D1 round-10 bracket premium", fe=("pb", "slot"))


def d2_attention(p: pl.DataFrame) -> dict[str, Any]:
    """D2: culture named legs 5-95c, both slots: y - mid on log(volume share / mid share), bucket FE."""
    q = with_r(attention(p)).filter(pl.col("mid").is_between(0.05, 0.95)).with_columns(
        pl.col("att").clip(0.01, 100).log().alias("latt"))
    return fe_reg(q, "latt", "D2 log volume-attention slope", fe=("pb", "slot"))


def _jsonable(x: Any) -> Any:
    if isinstance(x, pl.DataFrame):
        return x.to_dicts()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def run_holdout() -> dict[str, Any]:
    """Evaluate the frozen rules on dev and holdout (refuses to run without preregistered.json)."""
    if not PREREG.exists():
        raise SystemExit("preregistered.json missing: freeze the rules before reading the holdout")
    frozen = json.loads(PREREG.read_text())
    cand = None                                   # ledgers load only their own tickers' candles
    cul = {"dev": pl.read_parquet(OUT / "culture_panel_dev.parquet"), "holdout": build_culture("holdout")}
    cul["holdout"].write_parquet(OUT / "culture_panel_holdout.parquet")
    wx_ = {"dev": pl.read_parquet(OUT / "weather_panel_dev.parquet"), "holdout": build_weather("holdout")}
    wx_["holdout"].write_parquet(OUT / "weather_panel_holdout.parquet")
    res: dict[str, Any] = {"frozen_at": frozen.get("frozen_at"), "rules": {}}
    led_dir = OUT / "ledgers"
    for name, fan in (("R1_fandom_no", True), ("C1_nonfandom_no", False)):
        trades = {s: r1_fandom_no(cul[s], fan) for s in ("dev", "holdout")}
        for s, tr in trades.items():
            tr.drop([c for c in tr.columns if c.startswith("rules")]).write_parquet(OUT / f"trades_{name}_{s}.parquet")
        ev = evaluate(trades, name, r1_family(), cand, led_dir)
        res["rules"][name] = {"scores": ev["scores"], "ledgers": ev["ledgers"],
                              "robust": pl.concat(ev["robust"], how="diagonal_relaxed") if ev["robust"] else pl.DataFrame()}
    # difference R1 - C1 per contract (holdout), event-clustered by stacking with a sign
    d = []
    for s in ("dev", "holdout"):
        a = pl.read_parquet(OUT / f"trades_R1_fandom_no_{s}.parquet").with_columns(pl.lit(1.0).alias("g"))
        b = pl.read_parquet(OUT / f"trades_C1_nonfandom_no_{s}.parquet").with_columns(pl.lit(0.0).alias("g"))
        ab = pl.concat([a, b], how="diagonal_relaxed").with_columns((pl.col("pnl")).alias("r"),
                                                                     bucket_expr(pl.col("mid")).alias("pb"))
        d.append({"split": s} | fe_reg(ab, "g", "R1 minus C1 (bucket FE)", fe=("pb",)))
    res["r1_minus_c1"] = d
    res["D1"] = [{"split": s} | d1_round10(wx_[s]) for s in ("dev", "holdout")]
    res["D2"] = [{"split": s} | d2_attention(cul[s]) for s in ("dev", "holdout")]
    # post-hoc descriptive tables on the holdout (labelled as such in the report)
    _save_tables(weather_tables(wx_["holdout"]), "holdout")
    _save_tables(culture_tables(cul["holdout"]), "holdout")
    (OUT / "results.json").write_text(json.dumps(_jsonable(res), indent=1, default=str))
    return res


def tables_md(res: dict[str, Any]) -> str:
    """Auto-generated result tables (pasted into report.md)."""
    out = []
    for name, r in res["rules"].items():
        out.append(f"### {name}\n\n" + md_table(pl.DataFrame(r["scores"]), ["rule", "split", "trades", "events", "dates",
                   "avg_price", "gross_c", "net_c", "t_event", "t_date", "net_per_$", "t_$", "pnl_usd", "hit"]))
        led = pl.DataFrame(r["ledgers"], infer_schema_length=None)
        if led.height:
            led = led.with_columns(pl.col("sharpe_ci90").map_elements(lambda v: f"[{v[0]}, {v[1]}]" if v is not None and len(v) == 2 else "",
                                                                      return_dtype=pl.String))
        out.append(md_table(led, ["rule", "split", "positions", "events", "pnl_usd", "usd_per_week", "ann_vol_pct",
                                  "sharpe", "sharpe_ci90", "sharpe_ex_top5", "usd_per_week_ex_top5", "max_dd_usd",
                                  "weeks_positive_pct", "skew"]))
        rob = r["robust"] if isinstance(r["robust"], pl.DataFrame) else pl.DataFrame(r["robust"])
        out.append(md_table(rob, ["split", "cut", "value", "trades", "events", "avg_price", "net_c", "t_event", "t_date",
                                  "net_per_$"], max_rows=80))
    for k in ("r1_minus_c1", "D1", "D2"):
        out.append(f"### {k}\n\n" + md_table(pl.DataFrame(res[k], infer_schema_length=None)))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.behav_culture")
    ap.add_argument("step", choices=["fetch-listing", "fetch-candles", "fetch-pageviews", "dev", "holdout", "tables"])
    ap.add_argument("--max-hist", type=int, default=None, help="cap on pre-cutoff per-market candle requests")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "fetch-listing":
        fetch_listing()
    elif a.step == "fetch-candles":
        fetch_all(a.max_hist)
    elif a.step == "fetch-pageviews":
        fetch_pageviews()
    elif a.step == "dev":
        run_dev()
    elif a.step == "holdout":
        res = run_holdout()
        (OUT / "results_tables.md").write_text(tables_md(res))
    elif a.step == "tables":
        res = json.loads((OUT / "results.json").read_text())
        (OUT / "results_tables.md").write_text(tables_md(res))


if __name__ == "__main__":
    main()
