"""Netflix Top 10 weekly markets on Kalshi: is there a repeatable small-stake edge?

Series (``KXNETFLIX*`` in today's ``/series``): #1 and #2 US show / movie (``KXNETFLIXRANK{SHOW,MOVIE}``,
``...RUNNERUP``), #1 and #2 global show / movie (``...GLOBAL``, ``...GLOBAL2``) and the #1's global views
ladders (``KXNETFLIXTOPVIEWS{TV,MOVIE}``). Rules (``rules_primary`` / ``rules_secondary``, checked
2026-11-20): "If X is #1 on the Netflix Top 10 US Show on the chart published on <Tue>"; the chart covers
the Mon-Sun week ending the previous Sunday; markets close Mon 23:59 ET; settlement source
``netflix.com/tudum/top10/...``. No "other" outcome: if the winner is not listed every market resolves NO.
Outcomes are added mid-week for new releases.

Data sources (robots.txt checked 2026-11-20 for ``*``, ``anthropic-ai``, ``ClaudeBot``, ``Claude-Web``):

* **Kalshi** (our venue): markets, settlements (``expiration_value`` = the published winner) and hourly
  candles. This is the only outcome source used.
* **netflix.com/tudum/top10** (official weekly lists and xlsx files): ``User-agent: *`` is
  ``Disallow: /``; only ClaudeBot is allowed, so the strict policy treats it as **blocked**. Copies
  of the weekly lists elsewhere (Wayback, What's on Netflix's weekly-history tools) are not used either.
* **What's on Netflix** daily top-10 pages (``/most-popular/?dateselect=MM/DD/YYYY``): robots allows
  every agent. Each dated page holds that day's in-app US top 10 (shows, movies), 12 other country
  lists and a global points table (top 50). The site states it tracks the in-app lists itself; Netflix
  publishes no daily list on the web, so this is not a copy of a robots-blocked page. As-of: 12
  Wayback captures of the live page show the archived list labelled D went live between 18:01Z on D and
  00:38Z on D+1 (byte-identical lists, no later revisions), so it is used only from D+1 01:00 UTC.
* FlixPatrol (robots.txt 403 -> blocked), Reelgood (blocks the Claude agents) are not used.

**Result (2026-11-20, report ``data/research/netflix/report.md``): no edge.** By Sun 18:00 ET the eventual
US #1 trades near 0.94 and by Mon 18:00 ET near 0.98 (mid Brier 0.001). The WoN model, which uses lists
public about a day late, picks the weekly winner 45-59% of the time and adds nothing over the price;
model-taker cells lose 88-100% per dollar. Dev-frozen late-week near-certainty rules failed the holdout.
The primary (NO on 1-10c longshots at Mon 18:00 ET) was +3.4%/$ on dev (41/41 legs) but -8.0%/$ on the
holdout (8 legs, t -0.66). The Sunday favourite rule was -27.7%/$ (t -2.25), with its losses in the
global movie ladders. 54 dev cells were tried. The forward recorder (``record``) is built and smoked.

Run (from the repo root, heavy steps under the capped wrapper):

    python -m albert.research.netflix discover | candles | won | panel | screen | holdout | tables | record
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import math
import re
import time
import unicodedata
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from albert.venues.kalshi import _field, num, parse_ts

NAME = "netflix"
OUT = Path("data/research/netflix")
CACHE = Path("data/cache/netflix")
ET = ZoneInfo("America/New_York")
UA = "albert-research/0.2 (public-data research)"
DEV_END = datetime(2026, 10, 1, tzinfo=UTC)          # dev: event closes before this
KALSHI_MIN_INTERVAL = 1.0                            # shared public rate limit: <= 1 request/s per worker
WON_MIN_INTERVAL = 2.0
WON_URL = "https://www.whats-on-netflix.com/most-popular/"

# Kalshi series -> (scope, kind, rank)
RANK_SERIES: dict[str, tuple[str, str, int]] = {
    "KXNETFLIXRANKSHOW": ("us", "tv", 1), "KXNETFLIXRANKSHOWRUNNERUP": ("us", "tv", 2),
    "KXNETFLIXRANKMOVIE": ("us", "movies", 1), "KXNETFLIXRANKMOVIERUNNERUP": ("us", "movies", 2),
    "KXNETFLIXRANKSHOWGLOBAL": ("global", "tv", 1), "KXNETFLIXRANKSHOWGLOBAL2": ("global", "tv", 2),
    "KXNETFLIXRANKMOVIEGLOBAL": ("global", "movies", 1), "KXNETFLIXRANKMOVIEGLOBAL2": ("global", "movies", 2),
}
VIEWS_SERIES = {"KXNETFLIXTOPVIEWSTV": "tv", "KXNETFLIXTOPVIEWSMOVIE": "movies"}


def _api(min_interval: float = KALSHI_MIN_INTERVAL):
    from albert.config import Settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    s = Settings()
    return k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=min_interval, max_retries=12))


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


# ---- Kalshi discovery ------------------------------------------------------------------------------
MARKET_COLS = {"ticker": pl.String, "event_ticker": pl.String, "series_ticker": pl.String,
               "yes_sub_title": pl.String, "strike_type": pl.String, "floor_strike": pl.Float64,
               "open_time": pl.Datetime("us", "UTC"), "close_time": pl.Datetime("us", "UTC"),
               "settlement_ts": pl.Datetime("us", "UTC"), "result": pl.String, "expiration_value": pl.String,
               "volume": pl.Float64, "rules_primary": pl.String, "status": pl.String, "src": pl.String}


def market_row(m: dict[str, Any], series: str, src: str) -> dict[str, Any]:
    return {"ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"), "series_ticker": series,
            "yes_sub_title": m.get("yes_sub_title"), "strike_type": m.get("strike_type"),
            "floor_strike": num(m.get("floor_strike")), "open_time": parse_ts(m.get("open_time")),
            "close_time": parse_ts(m.get("close_time")), "settlement_ts": parse_ts(m.get("settlement_ts")),
            "result": m.get("result") or None,
            "expiration_value": (str(m.get("expiration_value")) if m.get("expiration_value") not in (None, "") else None),
            "volume": num(_field(m, "volume")), "rules_primary": m.get("rules_primary"),
            "status": m.get("status"), "src": src}


def netflix_series(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for s in payload.get("series") or []:
        t = s.get("ticker") or ""
        if t.startswith("KXNETFLIX"):
            src = (s.get("settlement_sources") or [{}])[0] or {}
            out.append({"series_ticker": t, "title": s.get("title"), "category": s.get("category"),
                        "frequency": s.get("frequency"), "fee_type": s.get("fee_type"),
                        "fee_multiplier": num(s.get("fee_multiplier")), "source_url": src.get("url")})
    return out


def _pages(api, path: str, params: dict[str, Any], *, cache: bool):
    cursor = None
    while True:
        page = api.client.get(path, {**params, "cursor": cursor}, cache=cache)
        yield page
        cursor = page.payload.get("cursor")
        if not cursor or not page.payload.get("markets"):
            return


def discover(log=print) -> pl.DataFrame:
    """Every market (historical + live settled + open) of every KXNETFLIX* series."""
    api = _api()
    ser = pl.DataFrame(netflix_series(api.series().payload), infer_schema_length=None)
    _save(ser, OUT / "series.parquet")
    rows: list[dict[str, Any]] = []
    for t in ser["series_ticker"].to_list():
        n0 = len(rows)
        for page in _pages(api, "/historical/markets", {"series_ticker": t, "limit": 1000}, cache=True):
            rows.extend(market_row(m, t, "hist") for m in page.payload.get("markets") or [])
        for status in ("settled", "open", "closed"):
            for page in _pages(api, "/markets", {"series_ticker": t, "status": status, "limit": 1000}, cache=False):
                rows.extend(market_row(m, t, status) for m in page.payload.get("markets") or [])
        log(f"  {t}: {len(rows) - n0} markets", flush=True)
    mk = pl.DataFrame(rows, schema=MARKET_COLS).unique("ticker", keep="last")
    mk = mk.join(ser.select("series_ticker", "fee_type", "fee_multiplier"), on="series_ticker", how="left")
    _save(mk, OUT / "markets.parquet")
    log(f"discover: {ser.height} series, {mk.height} markets, "
        f"{mk.filter(pl.col('result').is_in(['yes', 'no'])).height} settled")
    return mk


# ---- hourly candles ----------------------------------------------------------------------------------
def candle_rows(payload: dict[str, Any], ticker: str) -> list[dict[str, Any]]:
    rows = []
    for c in payload.get("candlesticks", []) or []:
        bid, ask, price = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
        rows.append({"ticker": ticker, "end_ts": parse_ts(c.get("end_period_ts")),
                     "yes_bid_close": num(_field(bid, "close")), "yes_ask_close": num(_field(ask, "close")),
                     "price_close": num(_field(price, "close")), "price_high": num(_field(price, "high")),
                     "price_low": num(_field(price, "low")), "volume": num(_field(c, "volume")),
                     "open_interest": num(_field(c, "open_interest"))})
    return rows


CANDLE_SCHEMA = {"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "yes_bid_close": pl.Float64,
                 "yes_ask_close": pl.Float64, "price_close": pl.Float64, "price_high": pl.Float64,
                 "price_low": pl.Float64, "volume": pl.Float64, "open_interest": pl.Float64}


def load_candles(tickers: list[str] | None = None) -> pl.DataFrame:
    parts = sorted((CACHE / "candles").glob("part-*.parquet"))
    if not parts:
        return pl.DataFrame(schema=CANDLE_SCHEMA)
    lf = pl.scan_parquet(parts)
    if tickers is not None:
        lf = lf.filter(pl.col("ticker").is_in(tickers))
    return lf.collect().unique(["ticker", "end_ts"]).sort("ticker", "end_ts")


def _polite(call, what: str, log=print, tries: int = 40):
    """Run one Kalshi request; on 429 (shared IP saturated) wait a fixed 8 s and retry, so a busy
    limit never turns into the client's exponential back-off."""
    from albert.http import HttpError

    for _ in range(tries):
        try:
            return call()
        except HttpError as error:
            msg = str(error)
            if "429" in msg or "too_many" in msg:
                time.sleep(8)
                continue
            log(f"skip {what}: {msg[:120]}", flush=True)
            return None
    log(f"skip {what}: still rate-limited", flush=True)
    return None


def fetch_candles(mk: pl.DataFrame, *, max_markets: int | None = None, log=print) -> None:
    """Hourly candles over each market's whole life (open -> close + 1h), sequential and polite.
    Live (post-cutoff) events: one batch request per event; historical markets: one request each,
    newest events first. New rows go to new part files under data/cache/netflix/candles (never
    overwritten); raw responses sit in the shared append-only HTTP cache."""
    from albert.config import Settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    d = CACHE / "candles"
    d.mkdir(parents=True, exist_ok=True)
    done_p = sorted(d.glob("done-*.parquet"))
    done = set(pl.read_parquet(done_p)["ticker"].to_list()) if done_p else set()
    sel = (mk.filter(pl.col("result").is_in(["yes", "no"]) & (pl.col("volume") > 0))
           .with_columns(pl.col("series_ticker").is_in(list(VIEWS_SERIES)).alias("_v"))
           .sort("_v", "close_time", descending=[False, True]))
    todo = [r for r in sel.select("ticker", "event_ticker", "open_time", "close_time").iter_rows(named=True)
            if r["ticker"] not in done]
    if max_markets:
        todo = todo[:max_markets]
    log(f"candles: {len(todo)} markets to fetch ({len(done)} done)", flush=True)
    if not todo:
        return
    st = Settings()
    api = k.Kalshi(CachedClient("kalshi", st.kalshi_base, st.raw_dir, min_interval=KALSHI_MIN_INTERVAL, max_retries=1))
    cutoff = api.cutoff().get("market_settled_ts")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    chunk = 0
    rows: list[dict[str, Any]] = []
    got: list[str] = []

    def flush() -> None:
        nonlocal rows, got, chunk
        if got:
            if rows:
                pl.DataFrame(rows, schema=CANDLE_SCHEMA).write_parquet(d / f"part-{stamp}-{chunk:04d}.parquet")
            pl.DataFrame({"ticker": got}).write_parquet(d / f"done-{stamp}-{chunk:04d}.parquet")
            chunk += 1
        rows, got = [], []

    live = [m for m in todo if cutoff and m["close_time"] >= cutoff]
    hist = [m for m in todo if not (cutoff and m["close_time"] >= cutoff)]
    by_event: dict[str, list[dict[str, Any]]] = {}
    for m in live:
        by_event.setdefault(m["event_ticker"], []).append(m)
    for i, (ev, ms) in enumerate(by_event.items(), 1):
        hi = max(m["close_time"] for m in ms) + timedelta(hours=1)
        lo = min(m["open_time"] for m in ms)
        page = _polite(lambda: api.candles_batch([m["ticker"] for m in ms], lo, hi, 60), ev, log)
        if page is None:
            continue
        for node in page.payload.get("markets") or []:
            rows.extend(candle_rows(node, node.get("market_ticker")))
        got.extend(m["ticker"] for m in ms)
        if i % 10 == 0:
            flush()
            log(f"  live events {i}/{len(by_event)}", flush=True)
    flush()
    for i, m in enumerate(hist, 1):
        hi = m["close_time"] + timedelta(hours=1)
        lo = max(m["open_time"] or hi - timedelta(days=10), hi - timedelta(days=60))
        page = _polite(lambda: api.candles(m["ticker"], lo, hi, 60, True), m["ticker"], log)
        if page is not None:
            rows.extend(candle_rows(page.payload, m["ticker"]))
            got.append(m["ticker"])
        if i % 100 == 0:
            flush()
            log(f"  historical {i}/{len(hist)} ({m['close_time']:%Y-%m-%d})", flush=True)
    flush()


# ---- What's on Netflix daily top-10 pages ----------------------------------------------------------
def won_path(d: date) -> Path:
    return CACHE / "won" / f"most-popular_dateselect_{d.isoformat()}.html.gz"


class WonFetcher:
    """Polite GET of dated pages; one gzip file per date, written once and never overwritten."""

    def __init__(self, min_interval: float = WON_MIN_INTERVAL) -> None:
        import httpx

        self.min_interval = min_interval
        self._last = 0.0
        self._http = httpx.Client(timeout=60, headers={"User-Agent": UA}, follow_redirects=True)

    def get(self, d: date | None, retries: int = 4) -> tuple[str | None, datetime]:
        params = {"dateselect": d.strftime("%m/%d/%Y")} if d is not None else {}
        for attempt in range(retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self._http.get(WON_URL, params=params)
            except Exception:  # noqa: BLE001 - network errors: back off and retry
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r.text, datetime.now(UTC)
            if r.status_code in (401, 403, 404):
                return None, datetime.now(UTC)
            time.sleep(10 * (attempt + 1))
        return None, datetime.now(UTC)


def fetch_won(days: list[date], log=print) -> int:
    f = WonFetcher()
    (CACHE / "won").mkdir(parents=True, exist_ok=True)
    n = 0
    todo = [d for d in days if not won_path(d).exists()]
    log(f"won: {len(todo)} dates to fetch ({len(days) - len(todo)} cached)", flush=True)
    for i, d in enumerate(todo, 1):
        text, at = f.get(d)
        if text is None:
            log(f"  won {d}: no page", flush=True)
            continue
        p = won_path(d)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(f"<!-- fetched_at {at.isoformat()} -->\n{text}".encode()))
        tmp.replace(p)
        n += 1
        if i % 50 == 0:
            log(f"  won {i}/{len(todo)}", flush=True)
    return n


_BTN = re.compile(r'<button[^>]*data-won-mp-trend-title="([^"]*)"[^>]*data-won-mp-trend-id="([^"]*)"'
                  r'[^>]*data-won-mp-trend-type="([^"]*)"[^>]*data-won-mp-trend-end-date="([^"]*)"')
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL)
_RANK = re.compile(r'<td class="won-mp-rank-col">\s*(\d+)\s*</td>')
_PTS = re.compile(r'<span class="won-mp-points">\s*([\d,]+)\s*</span>')
_COUNTRY = re.compile(r'<h4 class="won-mp-chart-title" id="won-mp-country-([A-Z]{2})">([^<]*)</h4>')


def parse_won(text: str) -> list[dict[str, Any]]:
    """Rows (list_date, scope, kind, rank, title, title_id, points) from one dated page. ``scope`` is
    ``global`` (points table, top 50) or a country code (in-app top 10)."""
    out: list[dict[str, Any]] = []
    g0 = text.find('won-mp-table--global"')
    if g0 >= 0:
        g1 = text.find("</table>", g0)
        for row in _ROW.findall(text[g0:g1]):
            rk = _RANK.search(row)
            if not rk:
                continue
            btns, pts = _BTN.findall(row), _PTS.findall(row)
            # each side is (button, points); a side may be empty, so pair by order of appearance
            cells = re.split(r'<td class="won-mp-td-move">.*?</td>', row, flags=re.DOTALL)
            for cell in cells:
                b, p = _BTN.search(cell), _PTS.search(cell)
                if b and p:
                    out.append({"list_date": b.group(4), "scope": "global", "kind": b.group(3), "rank": int(rk.group(1)),
                                "title": html.unescape(b.group(1)), "title_id": b.group(2),
                                "points": float(p.group(1).replace(",", ""))})
            del btns, pts
    heads = list(_COUNTRY.finditer(text))
    for i, h in enumerate(heads):
        seg = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        for row in _ROW.findall(seg):
            rk, b = _RANK.search(row), _BTN.search(row)
            if rk and b:
                out.append({"list_date": b.group(4), "scope": h.group(1), "kind": b.group(3), "rank": int(rk.group(1)),
                            "title": html.unescape(b.group(1)), "title_id": b.group(2), "points": None})
    return out


def page_date(text: str) -> str | None:
    m = re.search(r'id="won-mp-country-US">United States top 10 — ([^<]+)</h4>', text)
    if not m:
        return None
    s = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", m.group(1).strip())
    try:
        return datetime.strptime(s, "%b %d, %Y").date().isoformat()
    except ValueError:
        return None


def build_won(log=print) -> pl.DataFrame:
    rows = []
    for p in sorted((CACHE / "won").glob("most-popular_dateselect_*.html.gz")):
        req = p.name.split("_")[-1].split(".")[0]
        text = gzip.decompress(p.read_bytes()).decode("utf-8", "replace")
        pd = page_date(text)
        if pd != req:          # a missing day falls back to another page: drop it
            continue
        for r in parse_won(text):
            if r["list_date"] == req:
                rows.append(r)
    df = pl.DataFrame(rows, schema={"list_date": pl.String, "scope": pl.String, "kind": pl.String, "rank": pl.Int64,
                                    "title": pl.String, "title_id": pl.String, "points": pl.Float64})
    df = df.with_columns(pl.col("list_date").str.to_date()).unique(["list_date", "scope", "kind", "rank"])
    _save(df, OUT / "won_daily.parquet")
    log(f"won daily: {df.height} rows, {df['list_date'].n_unique()} dates "
        f"({df['list_date'].min()} -> {df['list_date'].max()})")
    return df


# ---- title matching ------------------------------------------------------------------------------------
_SUFFIX = re.compile(
    r"(\s*[:\-–]\s*|\s+)(season|series|part|volume|vol\.?|chapter|book|collection)\s*[\divxlc]+\b.*$"
    r"|(\s*[:\-–]\s*)(limited series|the series|miniseries|docuseries)\s*$"
    r"|(\s*[:\-–]\s*)s\d+\b.*$"
    r"|\s*\(\d{4}\)\s*$")


def norm_title(s: str | None) -> str:
    """Comparable key: accents and punctuation dropped, season / limited-series suffixes removed."""
    if not s:
        return ""
    s = re.sub(r"['\u2019\u2018`\u00b4]", "", html.unescape(s))          # Madea's == Madea’s == Madeas
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()
    s = s.replace("&", " and ")
    for _ in range(2):
        s = _SUFFIX.sub("", s).strip()
    s = re.sub(r"\s+-\s+.*\d{4}$", "", s)          # "Raw: 2026 - November 2, 2026" style dated episodes
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def season_of(s: str | None) -> int | None:
    m = re.search(r"\b(?:season|s)\s*(\d+)\b", (s or "").lower())
    return int(m.group(1)) if m else None


def match_key(kalshi_name: str, won_keys: set[str]) -> str | None:
    """Map a Kalshi outcome name to a WoN title key: exact normalized match, then the longest WoN key
    that is a word-prefix of the Kalshi key (or vice versa); keys under 5 characters only when the
    rest is a number ("raw 2026")."""
    k = norm_title(kalshi_name)
    if not k:
        return None
    if k in won_keys:
        return k
    best = None
    for w in won_keys:
        if not (k.startswith(w + " ") or w.startswith(k + " ")) or (best is not None and len(w) <= len(best)):
            continue
        rest = k[len(w):] if k.startswith(w + " ") else w[len(k):]
        # short keys ("raw", "you") match only when the rest is a number ("raw 2026"), never words
        if len(w) >= 5 or re.fullmatch(r"[\d ]+", rest):
            best = w
    return best


def to_et(ts: datetime) -> datetime:
    return ts.astimezone(ET)


def week_of_close(close: datetime) -> tuple[date, date]:
    """(Monday, Sunday) of the tracking week of an event closing Mon 23:59 ET."""
    c = to_et(close)
    sunday = c.date() - timedelta(days=1)
    return sunday - timedelta(days=6), sunday


# ---- panel: one row per (event, outcome, decision time) -------------------------------------------
# decision key -> (days relative to the closing Monday in ET, hour ET)
DECISIONS: dict[str, tuple[int, int]] = {"thu18": (-4, 18), "sat18": (-2, 18), "sun18": (-1, 18),
                                         "mon12": (0, 12), "mon18": (0, 18), "mon22": (0, 22)}
LIST_PUBLIC_HOURS = 25       # the archived WoN list labelled D is treated as public from D+1 01:00 UTC
MIN_WON_DAYS = 4            # the model is used only when >= 4 daily lists of the week are public
QUOTE_MAX_AGE = timedelta(days=7)        # hourly candles are emitted only when the book or tape changes


def decision_time(close: datetime, key: str) -> datetime:
    days, hour = DECISIONS[key]
    mon = to_et(close).date()
    return datetime(mon.year, mon.month, mon.day, hour, tzinfo=ET).astimezone(UTC) + timedelta(days=days)


def last_public_label(t: datetime) -> date:
    """Latest archived WoN list label public at ``t``. Measured on 12 Wayback captures of the live page
    (2025): during day D the live page shows the label D but the content of the archived list D-1; the
    archived list D appeared live between 18:01Z on D (not yet) and 23:08Z on D (shown), and was shown
    at 00:38Z / 01:13Z on D+1. So the archived list D is used only from D+1 01:00 UTC."""
    return (t.astimezone(UTC) - timedelta(hours=LIST_PUBLIC_HOURS)).date()


def events_frame(mk: pl.DataFrame) -> pl.DataFrame:
    m = mk.filter(pl.col("series_ticker").is_in(list(RANK_SERIES)) & pl.col("result").is_in(["yes", "no"]))
    ev = (m.group_by("event_ticker", "series_ticker")
          .agg(pl.col("close_time").max().alias("close"), pl.col("settlement_ts").max().alias("settle"),
               pl.coalesce(pl.col("expiration_value").drop_nulls().first(),
                           pl.col("yes_sub_title").filter(pl.col("result") == "yes").first()).alias("winner"),
               pl.len().alias("n_out"),
               (pl.col("result") == "yes").sum().alias("n_yes"), pl.col("volume").sum().alias("volume")))
    meta = pl.DataFrame([{"series_ticker": k, "scope": v[0], "kind": v[1], "rank": v[2]} for k, v in RANK_SERIES.items()])
    ev = ev.join(meta, on="series_ticker").with_columns(
        pl.when(pl.col("close") < DEV_END).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("split"))
    return ev.sort("close", "series_ticker")


FEATURES = ("x", "x2", "new", "new_x2")
# Netflix paid memberships by country, millions (rounded public estimates, 2024-25); fixed, not fitted
COUNTRY_WEIGHTS = {"US": 80.0, "CA": 9.0, "UK": 17.0, "DE": 15.0, "FR": 12.0, "ES": 8.0, "NL": 4.0,
                   "BR": 22.0, "MX": 13.0, "IN": 12.0, "AU": 7.0, "NZ": 2.0}


def won_window(won: pl.DataFrame, scope: str, kind: str, close: datetime, t: datetime, offset: int) -> pl.DataFrame:
    """Per WoN title key, from the list labels of the tracking week (labels Mon..Sun shifted by
    ``offset`` days) that are public at ``t``: ``x`` mean daily score (US: (11 - rank)/10; global: the
    member-weighted mean of (11 - rank)/10 over 12 country lists; 0 off the lists), ``x2`` the same over the last 2
    public labels (fresh Friday releases win weeks on two or three huge days), ``new`` = not on any list
    in the 7 labels before the window, ``days`` labels used, ``top_days`` days at #1."""
    empty = pl.DataFrame(schema={"key": pl.String, "x": pl.Float64, "x2": pl.Float64, "new": pl.Float64,
                                 "new_x2": pl.Float64, "days": pl.Int64, "top_days": pl.Int64})
    mon, sun = week_of_close(close)
    lo, hi = mon + timedelta(days=offset), min(sun + timedelta(days=offset), last_public_label(t))
    if hi < lo:
        return empty
    if scope == "global":
        # WoN's global points table has a gap (Sep 2025 - Feb 2026) and weights every country equally;
        # the global chart ranks views, so use the 12 country lists weighted by Netflix members instead
        base = won.filter(pl.col("scope").is_in(list(COUNTRY_WEIGHTS)) & (pl.col("kind") == kind))
        tot = sum(COUNTRY_WEIGHTS.values())
        base = (base.with_columns(((11 - pl.col("rank")) / 10 * pl.col("scope").replace_strict(COUNTRY_WEIGHTS, return_dtype=pl.Float64) / tot).alias("pts"))
                .group_by("list_date", "title").agg(pl.col("pts").sum(), pl.col("rank").min()))
    else:
        base = (won.filter((pl.col("scope") == "US") & (pl.col("kind") == kind))
                .with_columns(((11 - pl.col("rank")) / 10).alias("pts")))
    w = base.filter(pl.col("list_date").is_between(lo, hi))
    if not w.height:
        return empty
    days = sorted(w["list_date"].unique().to_list())
    last2 = days[-2:]
    w = w.with_columns(pl.col("title").map_elements(norm_title, return_dtype=pl.String).alias("key"))
    prior = set(base.filter(pl.col("list_date").is_between(lo - timedelta(days=7), lo - timedelta(days=1)))["title"]
                .map_elements(norm_title, return_dtype=pl.String).to_list())
    out = (w.group_by("key").agg((pl.col("pts").sum() / len(days)).alias("x"),
                                 (pl.col("pts").filter(pl.col("list_date").is_in(last2)).sum() / len(last2)).alias("x2"),
                                 (pl.col("rank") == 1).sum().alias("top_days"))
           .with_columns((~pl.col("key").is_in(list(prior))).cast(pl.Float64).alias("new"),
                         pl.lit(len(days)).cast(pl.Int64).alias("days")))
    return out.with_columns((pl.col("new") * pl.col("x2")).alias("new_x2")).select(empty.columns)


def pl_probs(u: np.ndarray, rank: int) -> np.ndarray:
    """Plackett-Luce probabilities of finishing #1 (rank=1) or #2 (rank=2) given utilities ``u``."""
    w = np.exp(u - u.max())
    big_w = w.sum()
    p1 = w / big_w
    if rank == 1:
        return p1
    rest = big_w - w                          # P(i second) = sum_{j != i} P(j first) * w_i / (W - w_j)
    m = p1[None, :] * w[:, None] / np.where(rest[None, :] > 0, rest[None, :], np.inf)
    np.fill_diagonal(m, 0.0)
    return m.sum(axis=1)


def quotes_at(cand: pl.DataFrame, tickers_t: pl.DataFrame) -> pl.DataFrame:
    """Last candle (end_ts <= t, within QUOTE_MAX_AGE) per (ticker, t): bid, ask, last price."""
    c = (cand.select("ticker", pl.col("end_ts").alias("q_ts"), pl.col("yes_bid_close").alias("bid"),
                     pl.col("yes_ask_close").alias("ask"), pl.col("price_close").alias("last"))
         .sort("ticker", "q_ts"))
    out = tickers_t.sort("ticker", "t").join_asof(c, left_on="t", right_on="q_ts", by="ticker", strategy="backward",
                                                  check_sortedness=False)
    stale = pl.col("q_ts").is_null() | ((pl.col("t") - pl.col("q_ts")) > QUOTE_MAX_AGE)
    return out.with_columns(pl.when(stale).then(None).otherwise(pl.col("bid")).alias("bid"),
                            pl.when(stale).then(None).otherwise(pl.col("ask")).alias("ask"))


def build_panel(offset: int = 1, log=print) -> pl.DataFrame:
    """Rows: every listed outcome of every settled ranked event at every decision time when it was open,
    with the executable quote, the WoN week-to-date score of its title, and the result."""
    mk = pl.read_parquet(OUT / "markets.parquet")
    ev = events_frame(mk)
    won = pl.read_parquet(OUT / "won_daily.parquet")
    m = (mk.filter(pl.col("series_ticker").is_in(list(RANK_SERIES)) & pl.col("result").is_in(["yes", "no"]))
         .select("ticker", "event_ticker", "yes_sub_title", "open_time", "close_time", "settlement_ts", "result",
                 "volume", "fee_multiplier"))
    cand = load_candles(m["ticker"].to_list())
    rows = []
    for e in ev.iter_rows(named=True):
        legs = m.filter(pl.col("event_ticker") == e["event_ticker"])
        for key in DECISIONS:
            t = decision_time(e["close"], key)
            if t >= e["close"]:
                continue
            ww = won_window(won, e["scope"], e["kind"], e["close"], t, offset)
            keys = set(ww["key"].to_list())
            for leg in legs.iter_rows(named=True):
                if leg["open_time"] is not None and leg["open_time"] > t:
                    continue
                k = match_key(leg["yes_sub_title"], keys)
                rows.append({"event": e["event_ticker"], "series": e["series_ticker"], "scope": e["scope"],
                             "kind": e["kind"], "rank": e["rank"], "split": e["split"], "close": e["close"],
                             "settle": leg["settlement_ts"] or e["settle"], "decision": key, "t": t,
                             "ticker": leg["ticker"], "name": leg["yes_sub_title"], "key": k,
                             "season": season_of(leg["yes_sub_title"]), "open_time": leg["open_time"],
                             "y": int(leg["result"] == "yes"), "volume": leg["volume"],
                             "fee_multiplier": leg["fee_multiplier"] or 1.0,
                             "winner": e["winner"], "won_days": int(ww["days"].max() or 0) if ww.height else 0})
    panel = pl.DataFrame(rows, infer_schema_length=None)
    q = quotes_at(cand, panel.select("ticker", "t").unique())
    panel = panel.join(q.select("ticker", "t", "bid", "ask", "last", "q_ts"), on=["ticker", "t"], how="left")
    panel = panel.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    _save(panel, OUT / f"panel_off{offset}.parquet")
    log(f"panel: {panel.height} rows, {panel['event'].n_unique()} events, quoted "
        f"{panel.filter(pl.col('bid').is_not_null()).height}")
    return panel


def model_frame(panel: pl.DataFrame, won: pl.DataFrame, offset: int) -> pl.DataFrame:
    """Candidate-title table per (event, decision): WoN window keys plus listed outcomes, features,
    and which candidate won (from the settled winner name)."""
    rows = []
    for (ev, dec), g in panel.group_by(["event", "decision"]):
        r0 = g.row(0, named=True)
        ww = won_window(won, r0["scope"], r0["kind"], r0["close"], r0["t"], offset)
        feats = {r["key"]: r for r in ww.iter_rows(named=True)}
        keys = set(feats)
        cands = set(keys)
        for k, nm in zip(g["key"].to_list(), g["name"].to_list(), strict=True):
            cands.add(k if k is not None else "kalshi:" + norm_title(nm))
        wk = match_key(r0["winner"], keys) if r0["winner"] else None
        wkey = wk if wk is not None else ("kalshi:" + norm_title(r0["winner"]) if r0["winner"] else None)
        if wkey:
            cands.add(wkey)
        for c in cands:
            f = feats.get(c, {})
            rows.append({"event": ev, "decision": dec, "rank": r0["rank"], "scope": r0["scope"], "split": r0["split"],
                         "cand": c, **{k: float(f.get(k) or 0.0) for k in FEATURES},
                         "is_winner": c == wkey, "won_days": r0["won_days"]})
    return pl.DataFrame(rows, infer_schema_length=None)


def _groups(mf: pl.DataFrame) -> list[tuple[np.ndarray, np.ndarray, int]]:
    return [(g.select(FEATURES).to_numpy(), g["is_winner"].to_numpy(), int(g["rank"][0]))
            for _, g in mf.group_by(["event", "decision"]) if g["is_winner"].any() and g["won_days"][0] >= MIN_WON_DAYS]


def _padded(groups: list[tuple[np.ndarray, np.ndarray, int]]) -> tuple[np.ndarray, ...]:
    """Groups -> padded arrays X (G, N, F), mask (G, N), winner index (G,), rank (G,)."""
    g, n = len(groups), max(len(x) for x, _, _ in groups)
    X = np.zeros((g, n, len(FEATURES)))
    mask = np.zeros((g, n), dtype=bool)
    win = np.zeros(g, dtype=int)
    rank = np.zeros(g, dtype=int)
    for i, (x, wv, rk) in enumerate(groups):
        X[i, : len(x)] = x
        mask[i, : len(x)] = True
        win[i] = int(np.argmax(wv))
        rank[i] = rk
    return X, mask, win, rank


def _loglik(b: np.ndarray, X: np.ndarray, mask: np.ndarray, win: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """Per-group log P(winner) under Plackett-Luce (#1 or #2), vectorized over padded groups."""
    u = np.where(mask, X @ b, -np.inf)
    u = u - u.max(axis=1, keepdims=True)
    w = np.where(mask, np.exp(u), 0.0)
    big_w = w.sum(axis=1, keepdims=True)
    p1 = w / big_w
    idx = np.arange(len(win))
    out = np.log(np.clip(p1[idx, win], 1e-12, None))
    two = rank == 2
    if two.any():
        rest = big_w - w
        term = np.where(mask & (rest > 1e-300), p1 / np.where(rest > 1e-300, rest, 1.0), 0.0)
        s = term.sum(axis=1) - term[idx, win]
        p2 = w[idx, win] * s
        out = np.where(two, np.log(np.clip(p2, 1e-12, None)), out)
    return out


def fit_cl(mf: pl.DataFrame) -> tuple[np.ndarray, float]:
    """Plackett-Luce / conditional-logit coefficients on FEATURES by maximum likelihood (dev only),
    with a small ridge penalty. Returns (coefficients, mean log-likelihood per group)."""
    from scipy.optimize import minimize

    groups = _groups(mf)
    if not groups:
        return np.zeros(len(FEATURES)), float("nan")
    arrs = _padded(groups)

    def nll(b: np.ndarray) -> float:
        return -float(_loglik(b, *arrs).sum()) + 0.01 * float(b @ b)

    best = None
    for x0 in ([4.0, 4.0, 0.0, 2.0], [1.0, 8.0, 0.5, 4.0], [8.0, 1.0, 0.0, 0.0]):
        res = minimize(nll, x0=np.array(x0), method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    return best.x, float(_loglik(best.x, *arrs).mean())


def attach_model(panel: pl.DataFrame, mf: pl.DataFrame, coefs: dict[str, np.ndarray]) -> pl.DataFrame:
    """Model probability per listed outcome. A title listed several times (seasons) gets the title's
    probability on the outcome with the highest mid at t (ties: highest season); the others 0.01."""
    probs = []
    for (ev, dec), g in mf.group_by(["event", "decision"]):
        b = coefs[g["scope"][0]]
        p = pl_probs(g.select(FEATURES).to_numpy() @ b, int(g["rank"][0]))
        probs.append(pl.DataFrame({"event": ev, "decision": dec, "cand": g["cand"], "p_title": p,
                                   "x": g["x"], "x2": g["x2"], "new": g["new"]}))
    pt = pl.concat(probs)
    pn = panel.with_columns(pl.when(pl.col("key").is_not_null()).then(pl.col("key"))
                            .otherwise(pl.lit("kalshi:") + pl.col("name").map_elements(norm_title, return_dtype=pl.String))
                            .alias("cand"))
    pn = pn.join(pt, on=["event", "decision", "cand"], how="left")
    pn = pn.with_columns(pl.col("mid").fill_null(-1).alias("_m"), pl.col("season").fill_null(0).alias("_s"))
    pn = pn.with_columns(pl.struct("_m", "_s").rank("ordinal", descending=True).over(["event", "decision", "cand"]).alias("_r"))
    pn = pn.with_columns(pl.when(pl.col("_r") == 1).then(pl.col("p_title")).otherwise(0.01).fill_null(0.01).alias("p"))
    pn = pn.with_columns(pl.when(pl.col("won_days") >= MIN_WON_DAYS).then(pl.col("p")).otherwise(None).alias("p"))
    return pn.drop("_m", "_s", "_r")


# ---- execution, ledgers, statistics -----------------------------------------------------------------
STAKE_PER_LEG = 2.5          # dollars per order (taker); <= 2 legs per event -> <= $5 per event
MAX_LEGS_PER_EVENT = 2


def order_fee(price: float, n: int, mult: float = 1.0) -> float:
    """Kalshi taker fee for one order of n contracts: 0.07 x mult x n x P(1-P), rounded up to the cent."""
    return math.ceil(0.07 * mult * n * price * (1 - price) * 100 - 1e-9) / 100


def take(row: dict[str, Any], side: str, stake: float = STAKE_PER_LEG) -> dict[str, Any] | None:
    """One taker order on ``side`` at the executable price; n contracts sized to ``stake``."""
    price = row["ask"] if side == "yes" else 1 - row["bid"]
    if price is None or not (0.0 < price < 1.0):
        return None
    mult = row.get("fee_multiplier") or 1.0
    n = max(1, int(stake // (price + 0.07 * mult * price * (1 - price) + 1e-9)))
    fee = order_fee(price, n, mult)
    won = row["y"] if side == "yes" else 1 - row["y"]
    return {"event_ticker": row["event"], "ticker": row["ticker"], "series": row["series"], "decision": row["decision"],
            "side": side, "price": price, "n": n, "fee": fee, "t_in": row["t"], "settle_ts": row["settle"],
            "close": row["close"], "split": row["split"], "cost_usd": n * price + fee, "payoff_usd": float(n * won),
            "pnl": n * won - n * price - fee, "c_per_contract": 100 * (won - price - fee / n), "edge": row.get("edge")}


def legs_frame(legs: list[dict[str, Any]]) -> pl.DataFrame:
    if not legs:
        return pl.DataFrame()
    df = pl.DataFrame(legs, infer_schema_length=None)
    # at most MAX_LEGS_PER_EVENT per event (largest edge first, else first come)
    order = [pl.col("edge").fill_null(0.0)] if "edge" in df.columns else []
    df = (df.sort(order or "t_in", descending=bool(order)).with_columns(pl.int_range(pl.len()).over("event_ticker").alias("_i"))
          .filter(pl.col("_i") < MAX_LEGS_PER_EVENT).drop("_i"))
    return df.with_columns(pl.col("close").dt.convert_time_zone("America/New_York").dt.date().alias("week"))


def clustered_t(x: np.ndarray, g: np.ndarray) -> float | None:
    from albert.research.fundamental import clustered_mean

    return clustered_mean(x, g)[1]


def summarize(df: pl.DataFrame) -> dict[str, Any]:
    if df is None or not df.height:
        return {"legs": 0}
    c = df["c_per_contract"].to_numpy()
    # per-leg $ PnL is what the book earns; the c/contract mean is contract-weighted
    contracts = df["n"].to_numpy().astype(float)
    pnl = df["pnl"].to_numpy()
    ev = df["event_ticker"].to_numpy()
    wk = df["week"].cast(pl.String).to_numpy()
    wsum = df.group_by("week").agg(pl.col("pnl").sum())["pnl"].to_numpy()
    return {"legs": df.height, "events": int(df["event_ticker"].n_unique()), "weeks": int(df["week"].n_unique()),
            "contracts": int(contracts.sum()),
            "c_per_contract": round(float((c * contracts).sum() / contracts.sum()), 2),
            "c_per_leg_mean": round(float(c.mean()), 2),
            "pnl_usd": round(float(pnl.sum()), 2), "risked_usd": round(float(df["cost_usd"].sum()), 2),
            "pnl_per_dollar": round(float(pnl.sum() / df["cost_usd"].sum()), 4),
            "t_event": _r2(clustered_t(pnl, ev)), "t_week": _r2(clustered_t(pnl, wk)),
            "pct_pos_weeks": round(100 * float((wsum > 0).mean()), 0), "win_rate": round(float((df["payoff_usd"] > 0).mean()), 3)}


def _r2(v: float | None) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v), 2)


def ledger(df: pl.DataFrame, strategy: str, variant: str) -> pl.DataFrame:
    return df.select(pl.lit(strategy).alias("strategy"), pl.lit(variant).alias("variant"), "split",
                     (pl.col("event_ticker") + ":" + pl.col("ticker") + ":" + pl.col("decision")).alias("struct_id"),
                     "event_ticker", "ticker", "side", "n", "t_in", "cost_usd", "settle_ts", "payoff_usd")


def marks_for(tickers: list[str]) -> pl.DataFrame:
    """Hourly YES mids from this study's candles (vol_metrics' own candle sources lack these series)."""
    c = load_candles(tickers)
    two = pl.col("yes_bid_close").is_not_null() & pl.col("yes_ask_close").is_not_null() & (pl.col("yes_ask_close") > pl.col("yes_bid_close"))
    mid = pl.when(two).then((pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2).otherwise(pl.col("price_close"))
    return (c.with_columns(mid.alias("yes_mid")).filter(pl.col("yes_mid").is_not_null())
            .select("ticker", pl.col("end_ts").alias("ts"), "yes_mid").sort("ticker", "ts"))


def risk_block(led: pl.DataFrame) -> dict[str, Any]:
    from albert.research import vol_metrics as vm

    if not led.height:
        return {}
    m = vm.ledger_metrics(led, marks_for(led["ticker"].unique().to_list()))
    keep = ["sharpe", "sharpe_ci90", "p_sharpe_pos", "ann_vol_pct", "sharpe_ex_top5", "max_dd_usd", "usd_per_week",
            "usd_per_week_ex_top5", "weeks_positive_pct", "sharpe_weekly", "sortino", "sharpe_settle_basis",
            "marked_share", "days", "events"]
    return {k: m.get(k) for k in keep}


# ---- rules ------------------------------------------------------------------------------------------
def rule_model(pn: pl.DataFrame, theta: float) -> list[dict[str, Any]]:
    """Taker on the WoN model: YES if p - ask - fee >= theta, NO if (1-p) - (1-bid) - fee >= theta."""
    legs = []
    for r in pn.filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & pl.col("p").is_not_null()).iter_rows(named=True):
        fy = 0.07 * r["fee_multiplier"] * r["ask"] * (1 - r["ask"]) + 0.005
        fn = 0.07 * r["fee_multiplier"] * r["bid"] * (1 - r["bid"]) + 0.005
        ey, en = r["p"] - r["ask"] - fy, r["bid"] - r["p"] - fn
        if max(ey, en) >= theta:
            side = "yes" if ey >= en else "no"
            leg = take({**r, "edge": max(ey, en)}, side)
            if leg:
                legs.append(leg)
    return legs


def rule_band(pn: pl.DataFrame, side: str, lo: float, hi: float) -> list[dict[str, Any]]:
    """NO on outcomes whose bid is in (lo, hi] (side='no'), or YES on the event favourite whose ask is
    in [lo, hi) (side='yes')."""
    q = pn.filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null())
    legs = []
    if side == "no":
        for r in q.filter((pl.col("bid") > lo) & (pl.col("bid") <= hi)).iter_rows(named=True):
            leg = take({**r, "edge": r["bid"]}, "no")
            if leg:
                legs.append(leg)
    else:
        fav = q.with_columns(pl.col("mid").rank("ordinal", descending=True).over(["event", "decision"]).alias("_r")).filter(pl.col("_r") == 1)
        for r in fav.filter((pl.col("ask") >= lo) & (pl.col("ask") < hi)).iter_rows(named=True):
            leg = take({**r, "edge": 1.0}, "yes")
            if leg:
                legs.append(leg)
    return legs


def incumbents(pn: pl.DataFrame, ev: pl.DataFrame) -> pl.DataFrame:
    """Rows of ``pn`` whose title won the previous week's event of the same series."""
    prev = ev.sort("series_ticker", "close").with_columns(pl.col("winner").shift(1).over("series_ticker").alias("prev_winner"),
                                                          pl.col("close").shift(1).over("series_ticker").alias("prev_close"))
    prev = prev.filter(pl.col("prev_close").is_not_null() & ((pl.col("close") - pl.col("prev_close")) < timedelta(days=8)))
    x = pn.join(prev.select(pl.col("event_ticker").alias("event"), "prev_winner"), on="event")
    return x.filter(pl.col("name").map_elements(norm_title, return_dtype=pl.String)
                    == pl.col("prev_winner").map_elements(norm_title, return_dtype=pl.String))


def late_added(pn_all: pl.DataFrame) -> pl.DataFrame:
    """First quoted decision row of each outcome that opened >= 12 h after its event's first outcome."""
    first_open = pn_all.group_by("event").agg(pl.col("open_time").min().alias("ev_open"))
    x = pn_all.join(first_open, on="event").filter((pl.col("open_time") - pl.col("ev_open")) >= timedelta(hours=12))
    x = x.filter(pl.col("bid").is_not_null()).sort("t")
    return x.group_by("ticker").first()


def overround(pn: pl.DataFrame) -> pl.DataFrame:
    """Per (event, decision): sum of YES bids over listed outcomes. At most one outcome is YES, so NO on
    every outcome pays >= k - 1 for a cost of k - sum(bids): sum(bids) > 1 + fees is an arbitrage; also
    sum of YES asks < 1 is one when the winner is surely listed (not assumed here)."""
    q = pn.filter(pl.col("bid").is_not_null())
    fee = 0.07 * pl.col("bid") * (1 - pl.col("bid")) + 0.01
    return (q.group_by("event", "decision").agg(pl.col("bid").sum().alias("sum_bid"), pl.col("ask").sum().alias("sum_ask"),
                                                fee.sum().alias("fees"), pl.len().alias("k"))
            .with_columns((pl.col("sum_bid") - 1 - pl.col("fees")).alias("no_arb_edge")))


def cross_rank_violations(pn: pl.DataFrame) -> pl.DataFrame:
    """Same title in a #1 and a #2 market of the same scope/kind/week: buying NO in both costs
    (1 - bid1) + (1 - bid2); at most one can be YES, so the pair pays >= $1. Rows where the cost after
    fees is below $1."""
    q = pn.filter(pl.col("bid").is_not_null()).with_columns(
        pl.col("name").map_elements(norm_title, return_dtype=pl.String).alias("nk"),
        pl.col("close").dt.date().alias("cd"))
    a = q.filter(pl.col("rank") == 1).select("scope", "kind", "cd", "decision", "nk", pl.col("bid").alias("bid1"), pl.col("ticker").alias("t1"))
    b = q.filter(pl.col("rank") == 2).select("scope", "kind", "cd", "decision", "nk", pl.col("bid").alias("bid2"), pl.col("ticker").alias("t2"))
    j = a.join(b, on=["scope", "kind", "cd", "decision", "nk"])
    cost = (1 - pl.col("bid1")) + (1 - pl.col("bid2")) + 0.07 * (pl.col("bid1") * (1 - pl.col("bid1")) + pl.col("bid2") * (1 - pl.col("bid2")))
    return j.with_columns(cost.alias("cost")).filter(pl.col("cost") < 0.99)


def complete_events() -> set[str]:
    """Events whose every traded market (volume > 0) has had its candles fetched: quote-based rules
    are evaluated only on these (the historical candle pull is rate-limited and runs newest first)."""
    mk = pl.read_parquet(OUT / "markets.parquet").filter(pl.col("series_ticker").is_in(list(RANK_SERIES))
                                                         & pl.col("result").is_in(["yes", "no"]))
    done_p = sorted((CACHE / "candles").glob("done-*.parquet"))
    done = set(pl.read_parquet(done_p)["ticker"].to_list()) if done_p else set()
    ev = (mk.with_columns(((pl.col("volume") <= 0) | pl.col("ticker").is_in(list(done))).alias("ok"))
          .group_by("event_ticker").agg(pl.col("ok").all()))
    return set(ev.filter(pl.col("ok"))["event_ticker"].to_list())


def convergence(panel: pl.DataFrame) -> pl.DataFrame:
    """Descriptive: per group and decision time, the favourite's mean mid and hit rate, and the mean
    mid of the eventual winner (events whose winner is listed and quoted)."""
    q = panel.filter(pl.col("mid").is_not_null())
    fav = q.with_columns(pl.col("mid").rank("ordinal", descending=True).over(["event", "decision"]).alias("_r")).filter(pl.col("_r") == 1)
    win = q.filter(pl.col("y") == 1)
    g = ["split", "scope", "rank", "decision"]
    a = fav.group_by(g).agg(pl.len().alias("events"), pl.col("mid").mean().round(3).alias("fav_mid"),
                            pl.col("y").mean().round(3).alias("fav_hit"))
    b = win.group_by(g).agg(pl.col("mid").mean().round(3).alias("winner_mid"))
    order = {k: i for i, k in enumerate(DECISIONS)}
    return (a.join(b, on=g, how="left").with_columns(pl.col("decision").replace_strict(order).alias("_o"))
            .sort("split", "scope", "rank", "_o").drop("_o"))


# ---- screening on dev -----------------------------------------------------------------------------
SCREEN_DECISIONS = ("thu18", "sat18", "sun18", "mon12", "mon18")


def crossfit(panel: pl.DataFrame, mf: pl.DataFrame, folds: int = 5) -> pl.DataFrame:
    """Dev predictions with coefficients fitted on the other folds (folds = interleaved weeks)."""
    wk = mf.join(panel.select("event", "close").unique("event"), on="event").with_columns(
        (pl.col("close").dt.ordinal_day() // 7 % folds).alias("fold"))
    out = []
    for f in range(folds):
        train, test = wk.filter(pl.col("fold") != f), wk.filter(pl.col("fold") == f)
        coefs = {sc: fit_cl(train.filter(pl.col("scope") == sc))[0] for sc in ("us", "global")}
        ev = test["event"].unique().to_list()
        out.append(attach_model(panel.filter(pl.col("event").is_in(ev)), test.drop("fold", "close"), coefs))
    return pl.concat(out, how="diagonal_relaxed")


def pred_all_acc(dev: pl.DataFrame, mf: pl.DataFrame, scope: str, decision: str, folds: int = 5) -> dict[str, Any]:
    """Cross-fitted accuracy of the WoN model on every dev week with >= MIN_WON_DAYS public lists."""
    m = mf.filter(pl.col("decision") == decision).join(dev.select("event", "close").unique("event"), on="event")
    m = m.with_columns((pl.col("close").dt.ordinal_day() // 7 % folds).alias("fold"))
    hits, lls = [], []
    for f in range(folds):
        b = fit_cl(m.filter((pl.col("fold") != f) & (pl.col("scope") == scope)).drop("fold", "close"))[0]
        for X, wv, rk in _groups(m.filter((pl.col("fold") == f) & (pl.col("scope") == scope)).drop("fold", "close")):
            p = pl_probs(X @ b, rk)
            hits.append(float(wv[p.argmax()]))
            lls.append(math.log(max(float(p[wv].sum()), 1e-9)))
    if not hits:
        return {}
    return {"rows": len(hits), "argmax_hit": round(float(np.mean(hits)), 3), "mean_loglik": round(float(np.mean(lls)), 3)}


def blend_test(df: pl.DataFrame) -> dict[str, Any]:
    from albert.research.fundamental import fit_logistic, logit

    d = df.filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & pl.col("p").is_not_null()
                  & (pl.col("ask") - pl.col("bid") <= 0.25))
    if d.height < 30 or d["y"].sum() < 3:
        return {"rows": d.height}
    y = d["y"].to_numpy().astype(float)
    X = np.column_stack([np.ones(d.height), logit(d["mid"].to_numpy()), logit(d["p"].to_numpy())])
    try:
        b, t = fit_logistic(X, y, d["event"].to_numpy())
    except np.linalg.LinAlgError:
        return {"rows": d.height, "events": d["event"].n_unique(), "note": "singular (separated)",
                "brier_mid": round(float(((d["mid"].to_numpy() - y) ** 2).mean()), 4),
                "brier_model": round(float(((d["p"].to_numpy() - y) ** 2).mean()), 4)}
    br_mid = float(((d["mid"].to_numpy() - y) ** 2).mean())
    br_mod = float(((d["p"].to_numpy() - y) ** 2).mean())
    return {"rows": d.height, "events": d["event"].n_unique(), "b_mid": round(float(b[1]), 2), "c_model": round(float(b[2]), 2),
            "t_c": round(float(t[2]), 2), "brier_mid": round(br_mid, 4), "brier_model": round(br_mod, 4)}


def screen(log=print) -> dict[str, Any]:
    won = pl.read_parquet(OUT / "won_daily.parquet")
    mk = pl.read_parquet(OUT / "markets.parquet")
    ev = events_frame(mk)
    cells: list[dict[str, Any]] = []
    res: dict[str, Any] = {"cells": cells}
    # 1) label offset (2 cells): dev log-likelihood of the winner at mon18
    fits = {}
    for off in (0, 1):
        panel = pl.read_parquet(OUT / f"panel_off{off}.parquet")
        dev = panel.filter(pl.col("split") == "dev")
        mf = model_frame(dev.filter(pl.col("decision") == "mon18"), won, off)
        lls = {sc: fit_cl(mf.filter(pl.col("scope") == sc)) for sc in ("us", "global")}
        fits[off] = lls
        cells.append({"family": "A0 offset", "cell": f"offset={off}", "metric": "mean loglik mon18",
                      **{f"ll_{sc}": round(v[1], 4) for sc, v in lls.items()},
                      **{f"coef_{sc}": [round(float(c), 2) for c in v[0]] for sc, v in lls.items()}})
    off = max(fits, key=lambda o: sum(v[1] for v in fits[o].values()))
    res["offset"] = off
    log(f"offset chosen on dev: {off}", flush=True)
    panel = pl.read_parquet(OUT / f"panel_off{off}.parquet")
    dev_all = panel.filter(pl.col("split") == "dev")
    mf = model_frame(dev_all, won, off)
    _save(mf, OUT / f"modelframe_dev_off{off}.parquet")
    pred = crossfit(dev_all, mf)            # coefficients use every dev week (no quotes needed)
    full = complete_events()
    dev = dev_all.filter(pl.col("event").is_in(list(full)))
    pred = pred.filter(pl.col("event").is_in(list(full)))
    _save(pred, OUT / "pred_dev_crossfit.parquet")
    res["dev_quoted"] = {"events": dev["event"].n_unique(), "first_close": str(dev["close"].min()),
                         "last_close": str(dev["close"].max()), "event_list": sorted(dev["event"].unique().to_list())}
    _save(convergence(dev), OUT / "convergence_dev.parquet")
    # model accuracy on every dev week (argmax hit, mean log-likelihood of the winner), cross-fitted
    for dec in SCREEN_DECISIONS:
        for sc in ("us", "global"):
            pdev = pred_all_acc(dev_all, mf, sc, dec)
            if pdev:
                cells.append({"family": "A1 model accuracy", "cell": f"{dec} {sc}", **pdev})
    # argmax accuracy of the model vs the market favourite (diagnostic, not a cell)
    for dec in SCREEN_DECISIONS:
        for sc in ("us", "global"):
            for rk in (1, 2):
                d = pred.filter((pl.col("decision") == dec) & (pl.col("scope") == sc) & (pl.col("rank") == rk))
                if not d.height:
                    continue
                cells.append({"family": "A2 blend", "cell": f"{dec} {sc} #{rk}", **blend_test(d)})
    # 2) model taker (8 cells)
    for dec in ("sun18", "mon18"):
        for sc in ("us", "global"):
            for th in (0.05, 0.10):
                d = pred.filter((pl.col("decision") == dec) & (pl.col("scope") == sc))
                cells.append({"family": "A3 model taker", "cell": f"{dec} {sc} theta={th}",
                              **summarize(legs_frame(rule_model(d, th)))})
    # 3) structural (pooled series)
    for dec in ("thu18", "sun18", "mon18"):
        d = dev.filter(pl.col("decision") == dec)
        for lo, hi in ((0.01, 0.10), (0.10, 0.30)):
            cells.append({"family": "B1 NO longshot", "cell": f"{dec} bid({lo},{hi}]", **summarize(legs_frame(rule_band(d, "no", lo, hi)))})
    for dec in ("sun18", "mon18"):
        d = dev.filter(pl.col("decision") == dec)
        for lo, hi in ((0.60, 0.90), (0.90, 0.98)):
            cells.append({"family": "B2 YES favourite", "cell": f"{dec} ask[{lo},{hi})", **summarize(legs_frame(rule_band(d, "yes", lo, hi)))})
    inc = incumbents(dev.filter(pl.col("decision") == "thu18").filter(pl.col("bid").is_not_null()), ev)
    for side in ("yes", "no"):
        legs = [x for x in (take({**r, "edge": 0.0}, side) for r in inc.iter_rows(named=True)) if x]
        cells.append({"family": "C incumbent", "cell": f"thu18 last week's winner {side}", **summarize(legs_frame(legs))})
    la = late_added(dev)
    for side in ("yes", "no"):
        legs = [x for x in (take({**r, "edge": 0.0}, side) for r in la.iter_rows(named=True)) if x]
        cells.append({"family": "D late-added", "cell": f"first quote {side}", **summarize(legs_frame(legs))})
    orr = overround(dev)
    cells.append({"family": "E all-NO overround", "cell": "sum(bid) - 1 - fees > 0", "rows": orr.height,
                  "events": int(orr.filter(pl.col("no_arb_edge") > 0)["event"].n_unique()),
                  "max_edge": round(float(orr["no_arb_edge"].max()), 3) if orr.height else None,
                  "median_sum_bid": round(float(orr["sum_bid"].median()), 3) if orr.height else None})
    cv = cross_rank_violations(dev)
    cells.append({"family": "E #1/#2 consistency", "cell": "NO+NO cost < $0.99", "rows": cv.height,
                  "events": cv.select("scope", "kind", "cd").unique().height if cv.height else 0})
    res["n_cells"] = len(cells)
    (OUT / "screen_dev.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ---- frozen rules: dev vs holdout ---------------------------------------------------------------------
def apply_rule(spec: dict[str, Any], df: pl.DataFrame, ev: pl.DataFrame | None = None) -> pl.DataFrame:
    """Legs of one frozen rule on a panel (``df`` must carry ``p`` for model rules)."""
    d = df.filter(pl.col("decision") == spec["decision"])
    if spec.get("scopes"):
        d = d.filter(pl.col("scope").is_in(spec["scopes"]))
    if spec.get("ranks"):
        d = d.filter(pl.col("rank").is_in(spec["ranks"]))
    fam = spec["family"]
    if fam == "model":
        legs = rule_model(d, spec["theta"])
    elif fam == "band":
        legs = rule_band(d, spec["side"], spec["lo"], spec["hi"])
    elif fam == "incumbent":
        inc = incumbents(d.filter(pl.col("bid").is_not_null()), ev)
        legs = [x for x in (take({**r, "edge": 0.0}, spec["side"]) for r in inc.iter_rows(named=True)) if x]
    else:
        raise ValueError(fam)
    return legs_frame(legs)


def evaluate_frozen(log=print) -> dict[str, Any]:
    spec = json.loads((OUT / "primary.json").read_text())
    off = spec["offset"]
    won = pl.read_parquet(OUT / "won_daily.parquet")
    mk = pl.read_parquet(OUT / "markets.parquet")
    ev = events_frame(mk)
    panel = pl.read_parquet(OUT / f"panel_off{off}.parquet")
    dev, ho = panel.filter(pl.col("split") == "dev"), panel.filter(pl.col("split") == "holdout")
    mf_dev = model_frame(dev, won, off)
    coefs = {sc: fit_cl(mf_dev.filter(pl.col("scope") == sc))[0] for sc in ("us", "global")}
    pred_ho = attach_model(ho, model_frame(ho, won, off), coefs)
    pred_dev = pl.read_parquet(OUT / "pred_dev_crossfit.parquet")
    _save(pred_ho, OUT / "pred_holdout.parquet")
    full = list(complete_events())
    _save(convergence(panel.filter(pl.col("event").is_in(full))), OUT / "convergence.parquet")
    frozen_dev = json.loads((OUT / "screen_dev.json").read_text())["dev_quoted"]["event_list"]   # the screened sample
    dev_q, ho_q = dev.filter(pl.col("event").is_in(frozen_dev)), ho.filter(pl.col("event").is_in(full))
    pred_ho_q = pred_ho.filter(pl.col("event").is_in(full))
    out_meta = {"dev_events": dev_q["event"].n_unique(), "holdout_events": ho_q["event"].n_unique(),
                "holdout_events_all": ho["event"].n_unique()}
    out: dict[str, Any] = {"spec": spec, "coefs": {k: [round(float(x), 3) for x in v] for k, v in coefs.items()},
                           "rules": [], **out_meta}
    (OUT / "ledgers").mkdir(parents=True, exist_ok=True)
    for role, rule in [("primary", spec["primary"])] + [("secondary", r) for r in spec.get("secondaries", [])]:
        row: dict[str, Any] = {"role": role, "rule": rule}
        for split, frame in (("dev", pred_dev if rule["family"] == "model" else dev_q),
                             ("holdout", pred_ho_q if rule["family"] == "model" else ho_q)):
            legs = apply_rule(rule, frame, ev)
            row[split] = summarize(legs)
            if legs.height:
                led = ledger(legs, NAME, rule["name"])
                led.write_parquet(OUT / "ledgers" / f"ledger_{rule['name']}_{split}.parquet")
                legs.write_parquet(OUT / "ledgers" / f"legs_{rule['name']}_{split}.parquet")
                row[f"{split}_risk"] = risk_block(led)
        out["rules"].append(row)
        log(json.dumps(row, default=str), flush=True)
    (OUT / "holdout.json").write_text(json.dumps(out, indent=1, default=str))
    return out


# ---- forward recorder ------------------------------------------------------------------------------
def kalshi_quote_rows(payload: dict[str, Any], fetched_at: datetime) -> list[dict[str, Any]]:
    rows = []
    for m in payload.get("markets") or []:
        rows.append({"fetched_at": fetched_at, "ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"),
                     "yes_sub_title": m.get("yes_sub_title"), "status": m.get("status"),
                     "open_time": parse_ts(m.get("open_time")), "close_time": parse_ts(m.get("close_time")),
                     "yes_bid": num(_field(m, "yes_bid")), "yes_ask": num(_field(m, "yes_ask")),
                     "last_price": num(_field(m, "last_price")), "volume": num(_field(m, "volume")),
                     "open_interest": num(_field(m, "open_interest"))})
    return rows


def record(log=print) -> dict[str, int]:
    """One forward snapshot: today's WoN page (every daily list it shows, stamped with the fetch time)
    and the open Kalshi quotes of every ranked and views Netflix series. Appends new files under
    data/research/netflix/forward/ (never overwrites); the raw WoN page is kept gzipped in the cache."""
    fwd = OUT / "forward"
    fwd.mkdir(parents=True, exist_ok=True)
    (CACHE / "won_live").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    text, at = WonFetcher(min_interval=0.0).get(None)
    n_won = 0
    if text:
        (CACHE / "won_live" / f"most-popular_{stamp}.html.gz").write_bytes(
            gzip.compress(f"<!-- fetched_at {at.isoformat()} -->\n{text}".encode()))
        rows = [{**r, "fetched_at": at, "page_date": page_date(text)} for r in parse_won(text)]
        if rows:
            pl.DataFrame(rows, infer_schema_length=None).write_parquet(fwd / f"won-{stamp}.parquet")
            n_won = len(rows)
    api = _api()
    qrows: list[dict[str, Any]] = []
    for series in list(RANK_SERIES) + list(VIEWS_SERIES):
        page = _polite(lambda s=series: api.client.get("/markets", {"series_ticker": s, "status": "open", "limit": 1000},
                                                        cache=False), series, log)
        if page is not None:
            qrows.extend({**r, "series_ticker": series} for r in kalshi_quote_rows(page.payload, page.fetched_at))
    if qrows:
        pl.DataFrame(qrows, infer_schema_length=None).write_parquet(fwd / f"kalshi-{stamp}.parquet")
    log(f"record {stamp}: won rows {n_won}, kalshi quotes {len(qrows)}")
    return {"won_rows": n_won, "kalshi_rows": len(qrows)}


# ---- report tables -----------------------------------------------------------------------------------
def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4g}" if abs(v) < 1e4 else f"{v:,.0f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def tables(log=print) -> str:
    out = []
    scr = json.loads((OUT / "screen_dev.json").read_text())
    cells = scr["cells"]
    out.append("### A0 label offset (dev, mon18)\n\n" + md_table([c for c in cells if c["family"] == "A0 offset"],
                                                                 ["cell", "ll_us", "ll_global", "coef_us", "coef_global"]))
    out.append("### A2 information test: logit(y) ~ a + b logit(mid) + c logit(model), dev, cross-fitted\n\n" + md_table(
        [c for c in cells if c["family"] == "A2 blend"],
        ["cell", "rows", "events", "b_mid", "c_model", "t_c", "brier_mid", "brier_model"]))
    out.append("### A1 WoN model accuracy on every dev week with >= 4 public lists (cross-fitted, no quotes needed)\n\n"
               + md_table([c for c in cells if c["family"] == "A1 model accuracy"], ["cell", "rows", "argmax_hit", "mean_loglik"]))
    cols = ["family", "cell", "legs", "events", "weeks", "contracts", "c_per_contract", "pnl_per_dollar", "pnl_usd", "t_event",
            "t_week", "pct_pos_weeks"]
    trade = [c for c in cells if c["family"][:2] in ("A3", "B1", "B2", "C ", "D ")]
    out.append("### Trading cells (dev, events with complete candles)\n\n" + md_table(trade, cols))
    out.append("### Structural checks (dev)\n\n" + md_table([c for c in cells if c["family"].startswith("E")],
                                                              ["family", "cell", "rows", "events", "max_edge", "median_sum_bid"]))
    if (OUT / "convergence.parquet").exists():           # dev + holdout, written by `holdout`
        cv = pl.read_parquet(OUT / "convergence.parquet")
        out.append("### Convergence (descriptive)\n\n" + md_table(cv.to_dicts(), cv.columns))
    if (OUT / "holdout.json").exists():
        ho = json.loads((OUT / "holdout.json").read_text())
        rows = []
        for r in ho["rules"]:
            for split in ("dev", "holdout"):
                rows.append({"role": r["role"], "rule": r["rule"]["name"], "split": split, **r.get(split, {}),
                             **{f"risk_{k}": v for k, v in (r.get(f"{split}_risk") or {}).items()}})
        if (OUT / "extra_pre_dev_check.json").exists():
            ex = json.loads((OUT / "extra_pre_dev_check.json").read_text())
            rows += [{"role": "post-freeze check", "rule": r["name"], "split": "Mar-Apr 2026", **r,
                      **{f"risk_{k}": v for k, v in (r.get("risk") or {}).items()}} for r in ex["rules"]]
        out.append("### Frozen rules\n\n" + md_table(rows, ["role", "rule", "split"] + cols[2:]))
        out.append("### Sharpe / vol (vol_metrics, $100 book, daily MTM at the candle mid)\n\n" + md_table(
            rows, ["rule", "split", "risk_sharpe", "risk_sharpe_ci90", "risk_p_sharpe_pos", "risk_ann_vol_pct",
                   "risk_sharpe_ex_top5", "risk_max_dd_usd", "risk_usd_per_week", "risk_weeks_positive_pct", "risk_marked_share"]))
    text = "\n\n".join(out) + "\n"
    (OUT / "tables.md").write_text(text)
    return text


REPORT_STATIC = """
## Markets and rules (checked 2026-11-20: `rules_primary`, `rules_secondary`, `/series`)

* Eight ranked series: #1 and #2 US show / movie (`KXNETFLIXRANK{SHOW,MOVIE}`, `...RUNNERUP`), #1 and #2
  global show / movie (`...GLOBAL`, `...GLOBAL2`), plus two ladders on the global #1's views
  (`KXNETFLIXTOPVIEWS{TV,MOVIE}`). `KXNETFLIX{SHOW,MOVIE}RANKD` (daily #1) had a single event (May 2025).
* "If X is #1 on the Netflix Top 10 US Show on the chart published on <Tue>". The chart covers the Mon-Sun
  week ending the previous Sunday; markets open Tue/Wed, close Mon 23:59 ET and settle on Tuesday after
  Netflix publishes (source `netflix.com/tudum/top10/...`). Global lists rank by views (hours / runtime);
  US lists are ranks only. Fees: quadratic, multiplier 1. Ties are not addressed in the rules.
* No "other" outcome: if the winner is not listed, every market resolves NO. That is common on the global
  ladders (unlisted winner in 7-13 of 48 events per global series, 3-9 per US series). Outcomes for new
  releases are added mid-week (Wed-Fri drops).
* History: US #1 since Jan 2025 (97 weekly events per series), US #2 since Dec 2025 (50-51), global #1/#2
  since Dec 23 2025 (48). Lifetime volume: US #1 $12-18M per series, US #2 $5-9M, global $0.8-2M.
* Displayed size: the L2 recorder samples these series; near the Monday close the top of book held
  100-600 contracts at 0.94-0.99 (e.g. KXNETFLIXRANKSHOW-26NOV16-TAK 0.99 x 176), so $1-5 orders fit.

## Sources, robots.txt (checked 2026-11-20 for `*`, `anthropic-ai`, `ClaudeBot`, `Claude-Web`) and as-of

| source | use | robots verdict | as-of |
|---|---|---|---|
| Kalshi API (markets, settlements, hourly candles) | outcomes, quotes | allowed | quotes from candles closed by the decision time |
| netflix.com/tudum/top10 (+ xlsx weekly files) | official weekly lists | **blocked** (`*`: Disallow /; only ClaudeBot allowed) | not used; Wayback copies and other mirrors of the weekly lists not used either |
| whats-on-netflix.com `/most-popular/?dateselect=` | daily in-app top-10 lists: US + 11 countries, global points top 50 | allowed (all four) | archived list D used from D+1 01:00 UTC (measured, below) |
| web.archive.org (CDX + 12 captures of the WoN page) | publication-lag measurement only | allowed (no robots.txt, 404) | capture timestamps |
| flixpatrol.com | daily lists | blocked (robots.txt 403) | not used |
| reelgood.com | top 10 | blocked for anthropic-ai / ClaudeBot / Claude-Web | not used |
| justwatch.com, api.tvmaze.com, wikimedia pageviews | popularity, schedules, attention | allowed | not used (see hypotheses H8, H10) |

**As-of measurement for the WoN archive.** 12 Wayback captures of the live WoN page (2025,
`won_lag_wayback.csv`): during day D the live page shows the label D but the content of the archived
list D-1 (10 of 12 captures, byte-identical lists); the archived list D was not yet live at 16:36-18:01Z
on D, was live at 23:08Z on D, and was still the live content at 00:38Z / 01:13Z on D+1. No capture showed
a list later revised. Rule used: the archived list D is public from D+1 01:00 UTC. At Mon 18:00 ET the
latest usable list is Sunday's. The forward recorder now measures the 2026 lag directly.

WoN data: 604 dated pages (Jan 6 2025 - Nov 16 2026) from 680 requested; gaps May 19 - Jun 20 2025,
some Aug/Sep 2026 days; the global points table is missing Sep 2025 - Feb 2026 (country lists are not),
so the global model uses the 12 country lists weighted by Netflix members instead (fixed weights).
"""


def write_report(verdict: str, discussion: str, log=print) -> Path:
    """Assemble report.md from the frozen results (tables.md) and the prose passed in."""
    tables()
    hyp = (OUT / "hypotheses.md").read_text().split("\n", 2)[2]
    prim = json.loads((OUT / "primary.json").read_text())
    text = ("# Netflix Top 10 weekly markets (Kalshi): research report\n\nModule `albert.research.netflix`, "
            "run 2026-11-20. Dev = events closing <= 2026-09-30; holdout = closing 2026-10-01 .. 2026-11-17.\n\n"
            "## Verdict\n\n" + verdict.strip() + "\n" + REPORT_STATIC
            + "\n## Hypotheses (written before opening outcome data)\n" + hyp
            + "\n## Frozen rule (`primary.json`)\n\n```json\n" + json.dumps(prim, indent=1) + "\n```\n\n"
            + "## Results\n\n" + (OUT / "tables.md").read_text() + "\n" + discussion.strip() + "\n")
    (OUT / "report.md").write_text(text)
    log(f"wrote {OUT / 'report.md'}")
    return OUT / "report.md"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.netflix")
    ap.add_argument("cmd", choices=["discover", "candles", "won", "won-parse", "panel", "screen", "record", "holdout", "tables"])
    ap.add_argument("--max-markets", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "discover":
        discover()
    elif a.cmd == "candles":
        mk = pl.read_parquet(OUT / "markets.parquet")
        mk = mk.filter(pl.col("series_ticker").is_in(list(RANK_SERIES) + list(VIEWS_SERIES)))
        fetch_candles(mk, max_markets=a.max_markets)
    elif a.cmd == "won":
        mk = pl.read_parquet(OUT / "markets.parquet").filter(pl.col("series_ticker").is_in(list(RANK_SERIES)))
        closes = mk.filter(pl.col("result").is_in(["yes", "no"]))["close_time"].unique().to_list()
        days: set[date] = set()
        for c in closes:
            mon, _ = week_of_close(c)
            days.update(mon + timedelta(days=i) for i in range(8))     # Mon .. next Mon (list labels)
        fetch_won(sorted(days))
        build_won()
    elif a.cmd == "won-parse":
        build_won()
    elif a.cmd == "panel":
        for off in (0, 1):
            build_panel(off)
    elif a.cmd == "record":
        record()
    elif a.cmd == "holdout":
        evaluate_frozen()
    elif a.cmd == "tables":
        print(tables())
    elif a.cmd == "screen":
        r = screen()
        for c in r["cells"]:
            print(c)


if __name__ == "__main__":
    main()
