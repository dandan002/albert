"""Recurring culture / entertainment / science-and-tech families on Kalshi (not Netflix, not the
Billboard / Luminate / Spotify music charts): which ones recur, which have public as-of data, and is
there a repeatable small-stake edge in the best of them?

Steps (all public data; nothing here places, amends or cancels an order):

* ``robots``: robots.txt verdicts for every alt-data host considered (``*``, ``anthropic-ai``,
  ``ClaudeBot``, ``Claude-Web``); a robots.txt that does not answer 200/404/410 counts as blocked.
* ``universe``: every Entertainment / Science and Technology / Culture / Social series in today's
  ``/series`` that settled >= 3 events Aug 17 - Nov 17 2026 (shared markets table + the Nov listing
  of ``behav_culture``), with events per month, volume, median spread (``behav_culture`` panels),
  settlement source, as-of data status and robots verdict, ranked by volume x feasibility.
* ``fetch``: every settled market of the deep-dive series (``/historical/markets`` +
  ``/markets?status=settled`` per series, cached raw and append-only) and hourly candles over each
  market's last ``LOOKBACK`` (batch endpoint after the historical cutoff).
* ``lmarena``: the LMArena leaderboard history (HF ``lmarena-ai/leaderboard-dataset``, ``text``
  config = default view without style control), stamped with a conservative publication time.
* ``openrouter``: OpenRouter's weekly author token totals (``/api/frontend/v1/rankings/author-market-share``).
* ``dev`` / ``holdout``: panels at fixed decision times, the dev screen (closes <= 2026-09-30), the
  frozen ``primary.json`` and the one-shot holdout (closes 2026-10-01 -> latest settled).
* ``record``: forward recorder (OpenRouter running weekly shares, LMArena latest, Kalshi quotes of
  the open markets of the deep-dive series) appended to ``data/research/culture_recurring/forward/``.

    python -m albert.research.culture_recurring robots | universe | all-fetch | alt | fetch
    python -m albert.research.culture_recurring build | dev | holdout | report | record

(``freeze`` is the function :func:`freeze`; ``primary.json`` refuses to be overwritten.)

Result (2026-11-20): no live edge; one thin paper candidate. Rotten Tomatoes is robots-blocked
(403), so KXRT is tested on prices only. The pre-registered primary (YES on the 85-97c KXRT leg at
close - 6 h) won 29/29 dev and 6/6 holdout fills (+6c/contract), but the exact
all-wins-at-these-prices p is 0.17 / 0.68 (0.11 pooled), and it makes ~$0.2/week at $3 an order.
LMArena (as-of from the HF dataset's commit times) adds no information beyond the price; the
challenger-NO secondary has the same shape (29/29, p 0.15). OpenRouter's request-share history
reproduces settlements but its dev window is 7 weeks; the intra-week signal needs the forward
recorder. See data/research/culture_recurring/report.md.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import polars as pl

NAME = "culture_recurring"
OUT = Path("data/research/culture_recurring")
CACHE = Path("data/cache/culture_recurring")
TABLES = Path("data/tables")
SNAP = "2026-11-20"                         # cache tag for listings that grow over time
DEV_END = datetime(2026, 10, 1, tzinfo=UTC)  # dev: closes before this; holdout: from this
UA = "albert-research/0.2 (public-data research)"
AGENTS = ("*", "anthropic-ai", "ClaudeBot", "Claude-Web")
KALSHI_TAKER = 0.07
LOOKBACK = timedelta(days=15)

RT_SERIES = ["KXRT"]
MC_SERIES = ["KXMC"]
LMA_SERIES = ["KXLLM1", "KXTOPMODEL", "KXCHINAAI", "KXTECHRANKLISTAICODE"]
OR_SERIES = ["KXOPENSHARE", "KXGOOGSHARE", "KXDEEPSHARE", "KXANTHSHARE", "KXTENCENTSHARE", "KXXIAOMISHARE",
             "KXBABASHARE", "KXZAISHARE", "KXTOKENUSE", "KXTOPUSAGEAI"]
YT_SERIES = ["KXYTVIEWSW", "KXYTVIEWSD", "KXYTVIEWSHIGH", "KXYTDAILYTOPVIDEO", "KXYTDAILYTOPVIDEOG", "KXYTTOPVIDEO2D",
             "KXYTTOPVIDEOG2D"]
KWORB_YT = "https://kworb.net/youtube/"
DEEP_SERIES = RT_SERIES + MC_SERIES + LMA_SERIES + OR_SERIES
EXCLUDED_RX = r"NETFLIX|BILLBOARD|^KXTOPSONG|^KXTOPALBUM|ALBUM|SONG|SPOTIFY|ARTISTSTREAMS|LUMINATE|^KXBB|^KXHOT100|RANKLIST1SONG|RANKLISTSONG|WEEKSNUM1|TOP10BBSPOTS|^KXTOPMONTHLY"


# ---- small shared helpers ----------------------------------------------------------------------
def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def taker_fee(n: int, price: float, mult: float | None = 1.0) -> float:
    """Kalshi taker fee for one order of ``n`` contracts at ``price``, rounded up to the cent."""
    raw = KALSHI_TAKER * (mult or 1.0) * n * price * (1 - price)
    return math.ceil(round(raw * 100, 9)) / 100


def _client(min_interval: float = 0.7):
    """Cached public Kalshi client (raw responses under data/raw/kalshi, append-only)."""
    from albert.config import Settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    s = Settings()
    return k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=min_interval, max_retries=12,
                                 sleep=lambda x: time.sleep(min(x, 20.0))))


class Web:
    """Polite GET for alt-data hosts: robots-checked hosts only, fixed UA, >= 1 s between requests.
    ``get(url, cache=True)`` stores one file per URL under data/cache/culture_recurring/web and never
    overwrites it; ``cache=False`` (forward recorder) always fetches and returns the body."""

    def __init__(self, cache_dir: Path = CACHE / "web", min_interval: float = 1.0) -> None:
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self._last = 0.0
        self._http = httpx.Client(timeout=120, headers={"User-Agent": UA}, follow_redirects=True)

    def path(self, url: str) -> Path:
        import hashlib

        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", url.split("://", 1)[-1])[:120]
        return self.dir / f"{safe}_{hashlib.sha1(url.encode()).hexdigest()[:12]}.bin"

    def get(self, url: str, *, cache: bool = True, retries: int = 3) -> tuple[bytes | None, datetime]:
        p = self.path(url)
        if cache and p.exists():
            return p.read_bytes(), datetime.fromtimestamp(p.stat().st_mtime, UTC)
        for attempt in range(retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self._http.get(url)
            except httpx.HTTPError:                 # network error: back off and retry
                time.sleep(3 * (attempt + 1))
                continue
            now = datetime.now(UTC)
            if r.status_code == 200:
                if cache:
                    tmp = p.with_suffix(".tmp")
                    tmp.write_bytes(r.content)
                    tmp.replace(p)
                return r.content, now
            if r.status_code in (401, 403, 404):
                return None, now
            time.sleep(3 * (attempt + 1))
        return None, datetime.now(UTC)


# ---- robots ------------------------------------------------------------------------------------
ROBOTS_CHECKS: dict[str, list[str]] = {
    "www.rottentomatoes.com": ["/m/wildwood", "/napi/movie/x/reviews/all"],
    "www.metacritic.com": ["/movie/x/", "/game/x/critic-reviews/"],
    "web.archive.org": ["/web/2026/https://www.metacritic.com/movie/x/", "/cdx/search/cdx"],
    "lmarena.ai": ["/leaderboard/text"],
    "huggingface.co": ["/datasets/lmarena-ai/leaderboard-dataset/resolve/main/text/full-00000-of-00001.parquet",
                       "/api/datasets/lmarena-ai/leaderboard-dataset/commits/main"],
    "openrouter.ai": ["/rankings", "/api/frontend/v1/rankings/author-market-share",
                      "/api/frontend/v1/rankings/models"],
    "kworb.net": ["/youtube/", "/youtube/artist/x.html", "/youtube/video/x.html"],
    "charts.youtube.com": ["/charts/TopVideos/global/daily", "/youtubei/v1/browse"],
    "store.steampowered.com": ["/charts/topsellers/US"],
    "steamdb.info": ["/topsellers/"],
    "artificialanalysis.ai": ["/text-to-image/arena", "/leaderboards/models"],
    "vercel.com": ["/ai-gateway/leaderboards/labs"],
    "deadline.com": ["/category/box-office/"],
    "www.boxofficemojo.com": ["/weekend/"],
    "www.the-numbers.com": ["/weekend-box-office-chart"],
    "www.getcollectr.com": ["/"],
    "wikimedia.org": ["/api/rest_v1/metrics/pageviews/"],
}


def check_robots(out: Path = OUT) -> pl.DataFrame:
    """Fetch each host's own robots.txt now and evaluate our paths for every agent."""
    from albert.research.counts import robots_verdict

    rows = []
    d = out / "robots"
    d.mkdir(parents=True, exist_ok=True)
    for host, paths in ROBOTS_CHECKS.items():
        try:
            r = httpx.get(f"https://{host}/robots.txt", timeout=20, follow_redirects=True, headers={"User-Agent": UA})
            status, text = r.status_code, (r.text if r.status_code == 200 else "")
        except httpx.HTTPError:
            status, text = None, ""
        (d / f"{host}.txt").write_text(text)
        for p in paths:
            v = robots_verdict(text, p) if status in (200, 404, 410) else dict.fromkeys(AGENTS, False)
            rows.append({"host": host, "path": p, "http_status": status,
                         "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                         **{f"allow_{a}": v[a] for a in AGENTS}, "allowed": all(v.values())})
    df = pl.DataFrame(rows)
    (out / "robots.json").write_text(json.dumps(rows, indent=1))
    return df


# ---- universe ----------------------------------------------------------------------------------
def family_of(series: str) -> str:
    """Group per-title / per-lab series into one recurring family."""
    rules = [
        (r"^KXRT$|^KXRTTV$|^KXRTCOMPAR", "Rotten Tomatoes score ladders"),
        (r"^KXMC$", "Metacritic score ladders"),
        (r"^KXYTVIEWS", "YouTube artist daily views ladders"),
        (r"^KXYT(DAILY)?TOP", "YouTube daily/weekly top music video/song"),
        (r"^KXLLM1$|^KXTOPMODEL$|^KXCHINAAI$|^KXTECHRANKLISTAICODE$|^KXAISPIKE$", "LMArena #1 (text / code / China)"),
        (r"^KX(OPEN|GOOG|DEEP|ANTH|TENCENT|XIAOMI|BABA|ZAI|STEALTH)SHARE$|^KXTOKENUSE|^KXTOPUSAGEAI$|^KXOPENSOURCESHAREOR$",
         "OpenRouter weekly shares / tokens"),
        (r"V(REQ|SPEND)$|^KXOPENSOURCESHARE$|^KXOSSPEND$", "Vercel AI Gateway shares"),
        (r"^KX(IMAGE|VIDEO|SPEECH|OPENINT)AI$|^KXCODEAI$", "Artificial Analysis arenas (weekly #1)"),
        (r"DWTS|DANCINGWITH", "Dancing with the Stars votes"),
        (r"BIGBROTHER", "Big Brother votes"),
        (r"AGT", "America's Got Talent votes"),
        (r"SURVIVOR", "Survivor eliminations"),
        (r"EMMY|VMA|GRAMMY|LGRAM|GAMEAWARDS|STREAMERAWARDS|TIFF|FILMFESTIVAL|OSCAR|GOLDENGLOBE|SAG|BAFTA|CRITICS",
         "Award shows (annual categories)"),
        (r"^KXTVSTATS", "Television Stats daily top shows"),
        (r"^KXSTEAM|^KXTOPSELLERS$", "Steam weekly top sellers"),
        (r"^KXPOKEMON$|^KXART$", "Collectible / art prices"),
        (r"^KXAPPRANK", "App Store free ranking"),
        (r"^KX(A100|B200|H100|H200|RTX5090)", "GPU rental price ladders"),
        (r"^KX(GPT|GEMINI|CLAUDE|GROK)$|^KXOAI|RELEASE", "AI / album release dates (one-offs)"),
        (r"SPACEX|STARSHIP|LAUNCH|NEWGLENN", "Space launches"),
    ]
    for rx, fam in rules:
        if re.search(rx, series or ""):
            return fam
    return "other"


# family -> (as-of public data, robots verdict, feasibility 0..1)
FEASIBILITY: dict[str, tuple[str, str, float]] = {
    "Rotten Tomatoes score ladders": (
        "RT pages only; RT robots.txt answers 403 -> blocked, so Wayback copies are blocked too. Price-only tests.",
        "blocked (rottentomatoes.com robots 403)", 0.5),
    "Metacritic score ladders": ("Metacritic pages + Wayback (allowed); weighted Metascore, 21 events", "allowed", 0.4),
    "YouTube artist daily views ladders": (
        "YouTube Charts data API /youtubei/ blocked; kworb.net allowed but keeps ~12 days and official channels only -> forward only",
        "charts.youtube.com/youtubei blocked; kworb allowed", 0.3),
    "YouTube daily/weekly top music video/song": ("same as above; kworb daily charts are a proxy -> forward only",
                                                  "kworb allowed", 0.2),
    "LMArena #1 (text / code / China)": (
        "HF lmarena-ai/leaderboard-dataset: daily snapshots (leaderboard_publish_date) with commit times -> as-of rebuildable",
        "huggingface.co allowed; lmarena.ai Cloudflare 403 (not used)", 0.9),
    "OpenRouter weekly shares / tokens": (
        "openrouter.ai author-market-share API: 52 weekly totals (final weeks); the running week is live -> backtest at week start, forward for intra-week",
        "allowed", 0.7),
    "Vercel AI Gateway shares": ("daily shares embedded in the page (culture.py study A: no edge)", "allowed", 0.8),
    "Artificial Analysis arenas (weekly #1)": ("leaderboard pages, no history API found; 95c spreads", "allowed", 0.2),
    "Dancing with the Stars votes": ("viewer votes secret; judges' scores only during the live show", "n/a", 0.1),
    "Big Brother votes": ("live feeds are paid/login; not usable", "n/a", 0.05),
    "America's Got Talent votes": ("votes secret", "n/a", 0.05),
    "Survivor eliminations": ("pre-taped; only leak-based spoilers (not public-information-safe)", "n/a", 0.0),
    "Award shows (annual categories)": ("one-off categories, no repeatable input", "n/a", 0.1),
    "Television Stats daily top shows": ("4 events, series dormant", "n/a", 0.1),
    "Steam weekly top sellers": ("store.steampowered.com / steamdb allowed; ~11 events, 32c spreads", "allowed", 0.3),
    "Collectible / art prices": ("Collectr app / Artnet prices; no as-of public history found", "getcollectr.com allowed", 0.2),
    "App Store free ranking": ("series ended 2025-11", "n/a", 0.0),
    "GPU rental price ladders": ("series dormant since 2026-09", "n/a", 0.1),
    "AI / album release dates (one-offs)": ("one-off release-date questions", "n/a", 0.1),
    "Space launches": ("few events", "n/a", 0.1),
    "other": ("not assessed", "n/a", 0.1),
}


def universe(out: Path = OUT) -> pl.DataFrame:
    """Recurring culture / tech families from today's /series and settled markets Aug 17 - Nov 17."""
    from albert.config import Settings
    from albert.http import CachedClient, read_raw

    s = Settings()
    key = CachedClient.cache_key(f"{s.kalshi_base}/series", {"__tag": SNAP})
    raw = next(p for p in (s.raw_dir / "kalshi" / f"{key}.json.zst", s.raw_dir / "kalshi" / f"{key}.json") if p.exists())
    rows = []
    for x in read_raw(raw)["response"]["series"]:
        if x.get("category") not in ("Entertainment", "Science and Technology", "Culture", "Social"):
            continue
        src = x.get("settlement_sources") or []
        rows.append({"series_ticker": x["ticker"], "category": x.get("category"), "frequency": x.get("frequency"),
                     "title": x.get("title"), "src_name": "; ".join((y or {}).get("name") or "" for y in src),
                     "fee_multiplier": float(x.get("fee_multiplier") or 1)})
    ser = pl.DataFrame(rows)
    cats = ["Entertainment", "Science and Technology", "Culture", "Social"]
    cols = ["ticker", "event_ticker", "series_ticker", "close_time", "volume"]
    mk = pl.concat([
        pl.scan_parquet(TABLES / "kalshi_markets.parquet").filter(pl.col("category").is_in(cats)).select(cols).collect(),
        pl.read_parquet("data/research/behav_culture/markets_nov.parquet", columns=cols + ["category"])
        .filter(pl.col("category").is_in(cats)).select(cols)]).unique("ticker")
    span_days = (mk["close_time"].max() - mk["close_time"].min()).total_seconds() / 86400
    agg = (mk.group_by("series_ticker").agg(pl.col("event_ticker").n_unique().alias("events"),
                                           pl.col("volume").sum().alias("volume"),
                                           pl.col("close_time").min().alias("first_close"),
                                           pl.col("close_time").max().alias("last_close")))
    pan = pl.concat([pl.read_parquet(f"data/research/behav_culture/culture_panel_{k}.parquet",
                                     columns=["series_ticker", "spread", "mid"]) for k in ("dev", "holdout")])
    spread = (pan.filter(pl.col("mid").is_between(0.05, 0.95)).group_by("series_ticker")
              .agg(pl.col("spread").median().alias("med_spread")))
    u = (ser.join(agg, on="series_ticker", how="inner").join(spread, on="series_ticker", how="left")
         .filter((pl.col("events") >= 3) & ~pl.col("series_ticker").str.contains(EXCLUDED_RX))
         .with_columns(pl.col("series_ticker").map_elements(family_of, return_dtype=pl.String).alias("family")))
    fam = (u.group_by("family").agg(
        pl.col("series_ticker").sort_by("volume", descending=True).str.join(",").alias("series"),
        pl.col("events").sum().alias("events"), pl.col("volume").sum().alias("volume"),
        (pl.col("med_spread") * pl.col("volume")).sum().alias("_sw"), pl.col("volume").filter(pl.col("med_spread").is_not_null()).sum().alias("_vw"),
        pl.col("src_name").first().alias("settlement_source"), pl.col("frequency").mode().first().alias("frequency"),
        pl.col("last_close").max().alias("last_close"))
        .with_columns((pl.col("_sw") / pl.col("_vw")).alias("vol_weighted_spread"),
                      (pl.col("events") / (span_days / 30.4)).alias("events_per_month"))
        .drop("_sw", "_vw"))
    feas = pl.DataFrame([{"family": k, "asof_data": v[0], "robots": v[1], "feasibility": v[2]} for k, v in FEASIBILITY.items()])
    fam = (fam.join(feas, on="family", how="left")
           .with_columns((pl.col("volume") * pl.col("feasibility") * (1 - pl.col("vol_weighted_spread").fill_null(0.5).clip(0, 0.9))).alias("opportunity"))
           .sort("opportunity", descending=True))
    _save(u, out / "universe_series.parquet")
    _save(fam, out / "universe.parquet")
    return fam


# ---- Kalshi markets and candles ----------------------------------------------------------------
MARKET_COLS = {"ticker": pl.String, "event_ticker": pl.String, "series_ticker": pl.String, "title": pl.String,
               "yes_sub_title": pl.String, "strike_type": pl.String, "floor_strike": pl.Float64,
               "cap_strike": pl.Float64, "open_time": pl.Datetime("us", "UTC"), "close_time": pl.Datetime("us", "UTC"),
               "expected_expiration_time": pl.Datetime("us", "UTC"), "settlement_ts": pl.Datetime("us", "UTC"),
               "result": pl.String, "expiration_value": pl.String, "volume": pl.Float64, "rules_primary": pl.String,
               "src": pl.String}


def market_row(m: dict[str, Any], series: str, src: str) -> dict[str, Any]:
    from albert.venues.kalshi import _field, num, parse_ts

    ev = m.get("expiration_value")
    return {"ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"), "series_ticker": series,
            "title": m.get("title"), "yes_sub_title": m.get("yes_sub_title"), "strike_type": m.get("strike_type"),
            "floor_strike": num(m.get("floor_strike")), "cap_strike": num(m.get("cap_strike")),
            "open_time": parse_ts(m.get("open_time")), "close_time": parse_ts(m.get("close_time")),
            "expected_expiration_time": parse_ts(m.get("expected_expiration_time")),
            "settlement_ts": parse_ts(m.get("settlement_ts")), "result": m.get("result") or None,
            "expiration_value": str(ev) if ev not in (None, "") else None, "volume": num(_field(m, "volume")),
            "rules_primary": m.get("rules_primary"), "src": src}


def _pages(api, path: str, params: dict[str, Any], key: str, max_pages: int = 100) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor = None
    for _ in range(max_pages):
        page = api.client.get(path, {**params, "cursor": cursor}, tag=SNAP)
        got = page.payload.get(key) or []
        out.extend(got)
        cursor = page.payload.get("cursor")
        if not cursor or not got:
            break
    return out


def fetch_markets(series: list[str], out: Path = OUT, *, log=print) -> pl.DataFrame:
    """Every settled market of these series (historical + live listings), volume-blind."""
    api = _client()
    rows: list[dict[str, Any]] = []
    for t in series:
        hist = _pages(api, "/historical/markets", {"series_ticker": t, "limit": 1000}, "markets")
        live = _pages(api, "/markets", {"series_ticker": t, "status": "settled", "limit": 1000}, "markets")
        rows += [market_row(m, t, "hist") for m in hist] + [market_row(m, t, "live") for m in live]
        log(f"  markets {t}: {len(hist)} hist + {len(live)} live", flush=True)
    mk = (pl.DataFrame(rows, schema=MARKET_COLS).unique("ticker", keep="last")
          .filter(pl.col("result").is_in(["yes", "no"])))
    path = out / "markets.parquet"
    if path.exists():
        old = pl.read_parquet(path)
        mk = pl.concat([old.filter(~pl.col("ticker").is_in(mk["ticker"].to_list())), mk], how="diagonal_relaxed")
    _save(mk, path)
    return mk


CANDLE_DIR = CACHE / "candles"


def fetch_candles(markets: pl.DataFrame, *, lookback: timedelta = LOOKBACK, chunk: int = 60, log=print) -> int:
    """Hourly candles over each market's last ``lookback``, written in small part files (append-only,
    one per chunk of markets) so memory stays flat: the batch endpoint for markets settled after the
    historical cutoff (joint window <= ~9,000 candles per request), the per-market historical endpoint
    before it. Markets already in a part file are skipped. Returns the number of new candle rows."""
    from concurrent.futures import ThreadPoolExecutor

    from albert.venues import kalshi as k

    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    parts = sorted(CANDLE_DIR.glob("part_*.parquet"))
    done = set(pl.scan_parquet(parts).select("ticker").unique().collect()["ticker"]) if parts else set()
    todo = markets.filter(~pl.col("ticker").is_in(list(done))).select("ticker", "open_time", "close_time")
    api = _client(min_interval=0.25)
    cutoff = api.cutoff().get("market_settled_ts")

    def window(m: dict[str, Any]) -> tuple[datetime, datetime]:
        hi = m["close_time"]
        return max(m["open_time"] or hi - lookback, hi - lookback), hi

    def hist_one(m: dict[str, Any]) -> list[dict[str, Any]]:
        lo, hi = window(m)
        try:
            return k.candle_rows(api.candles(m["ticker"], lo, hi, 60, True).payload, m["ticker"], 60)
        except RuntimeError as error:
            log(f"  skip {m['ticker']}: {str(error)[:100]}", flush=True)
            return []

    def live_batch(ms: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        groups, cur = [], []
        for m in sorted(ms, key=lambda m: window(m)[0]):
            trial = cur + [m]
            lo, hi = min(window(x)[0] for x in trial), max(window(x)[1] for x in trial)
            if cur and (len(trial) > 100 or ((hi - lo).total_seconds() / 3600 + 1) * len(trial) > 9000):
                groups.append(cur)
                cur = [m]
            else:
                cur = trial
        if cur:
            groups.append(cur)
        for g in groups:
            lo, hi = min(window(x)[0] for x in g), max(window(x)[1] for x in g)
            try:
                page = api.candles_batch([x["ticker"] for x in g], lo, hi, 60)
            except RuntimeError as error:
                log(f"  skip batch {g[0]['ticker']}: {str(error)[:100]}", flush=True)
                continue
            out += [r for mk in page.payload.get("markets") or [] for r in k.candle_rows(mk, mk.get("market_ticker"), 60)]
        return out

    rows_all = list(todo.iter_rows(named=True))
    n_new = 0
    for c0 in range(0, len(rows_all), chunk):
        ms = rows_all[c0:c0 + chunk]
        hist = [m for m in ms if cutoff and m["close_time"] < cutoff]
        live = [m for m in ms if not (cutoff and m["close_time"] < cutoff)]
        rows: list[dict[str, Any]] = live_batch(live) if live else []
        if hist:
            with ThreadPoolExecutor(max_workers=2) as pool:
                for r in pool.map(hist_one, hist):
                    rows += r
        df = pl.DataFrame(rows, schema=k.CANDLE_SCHEMA) if rows else pl.DataFrame(schema=k.CANDLE_SCHEMA)
        # mark markets with no candles as done too (an empty row keeps the ticker in the done set)
        missing = {m["ticker"] for m in ms} - set(df["ticker"].unique().to_list())
        if missing:
            df = pl.concat([df, pl.DataFrame({"ticker": sorted(missing)}, schema={"ticker": pl.String})], how="diagonal_relaxed")
        part = CANDLE_DIR / f"part_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}.parquet"
        _save(df, part)
        n_new += len(rows)
        log(f"  candles {min(c0 + chunk, len(rows_all))}/{len(rows_all)} markets, +{len(rows)} rows", flush=True)
    return n_new


def load_candles(tickers: list[str] | None = None) -> pl.DataFrame:
    parts = sorted(CANDLE_DIR.glob("part_*.parquet"))
    if not parts:
        return pl.DataFrame()
    q = pl.scan_parquet(parts).filter(pl.col("end_ts").is_not_null())
    if tickers is not None:
        q = q.filter(pl.col("ticker").is_in(tickers))
    return q.collect().unique(["ticker", "end_ts"])


# ---- LMArena leaderboard history (HF lmarena-ai/leaderboard-dataset) ---------------------------
HF = "https://huggingface.co"
LMA_REPO = "lmarena-ai/leaderboard-dataset"
LMA_CONFIGS = ("text", "webdev")
ORG_ALIASES = {  # Kalshi leg label -> LMArena organization names (lower case)
    "claude": ("anthropic",), "anthropic": ("anthropic",), "gemini": ("google",), "google": ("google",),
    "chatgpt": ("openai",), "openai": ("openai",), "grok": ("xai", "x-ai"), "xai": ("xai", "x-ai"),
    "deepseek": ("deepseek",), "meta": ("meta",), "llama": ("meta",), "mistral": ("mistral",),
    "alibaba": ("alibaba",), "qwen": ("alibaba",), "moonshot": ("moonshot", "moonshotai"), "kimi": ("moonshot", "moonshotai"),
    "xiaomi": ("xiaomi",), "zhipu": ("zhipu", "z.ai", "zai"), "z.ai": ("zhipu", "z.ai", "zai"),
    "tencent": ("tencent",), "bytedance": ("bytedance",), "minimax": ("minimax",), "baidu": ("baidu",),
    "stepfun": ("stepfun",), "meituan": ("meituan",),
}


def lmarena_commits(web: Web | None = None) -> pl.DataFrame:
    """Every commit of the dataset (date, title): 'Update text for D' gives when snapshot D went public."""
    web = web or Web(CACHE / "web" / SNAP)
    rows = []
    for p in range(200):
        body, _ = web.get(f"{HF}/api/datasets/{LMA_REPO}/commits/main?p={p}")
        page = json.loads(body) if body else []
        if not page:
            break
        rows += [{"commit_ts": datetime.fromisoformat(c["date"]), "title": c.get("title") or "",
                  "id": c.get("id")} for c in page]
    return pl.DataFrame(rows, schema={"commit_ts": pl.Datetime("us", "UTC"), "title": pl.String, "id": pl.String})


def lmarena_download(cfg: str, web: Web | None = None) -> Path:
    """The config's full history parquet (one cached copy per snapshot day, never overwritten)."""
    path = CACHE / "lmarena" / f"{cfg}_full_{SNAP}.parquet"
    if not path.exists():
        web = web or Web(CACHE / "web" / SNAP)
        body, _ = web.get(f"{HF}/datasets/{LMA_REPO}/resolve/main/{cfg}/full-00000-of-00001.parquet", cache=False)
        if body is None:
            raise RuntimeError(f"download failed: {cfg}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return path


def lmarena_history(cfg: str = "text", category: str = "overall") -> pl.DataFrame:
    """Overall leaderboard per publish date with ``as_of``: the first commit that published that date
    for this config (snapshots never committed in the visible history get as_of = null)."""
    commits = pl.read_parquet(OUT / "lmarena_commits.parquet")
    rx = rf"^Update {cfg} for (\d{{4}}-\d{{2}}-\d{{2}})"
    pub = (commits.with_columns(pl.col("title").str.extract(rx, 1).alias("d")).filter(pl.col("d").is_not_null())
           .group_by("d").agg(pl.col("commit_ts").min().alias("as_of")))
    h = (pl.scan_parquet(lmarena_download(cfg)).filter(pl.col("category") == category)
         .select("model_name", "organization", "rating", "rating_lower", "rating_upper", "vote_count", "rank",
                 pl.col("leaderboard_publish_date").alias("d"))
         .collect()
         .join(pub, on="d", how="left")
         .with_columns(pl.col("d").str.to_date().alias("pub_date"), pl.col("organization").str.to_lowercase()))
    return h


def leaders(h: pl.DataFrame) -> pl.DataFrame:
    """Per snapshot: #1 model / org by rating, the best model of the runner-up org and the margin."""
    best_org = (h.sort("rating", descending=True).group_by("pub_date", "organization", maintain_order=True)
                .agg(pl.col("model_name").first(), pl.col("rating").first(), pl.col("rating_lower").first(),
                     pl.col("rating_upper").first(), pl.col("as_of").first()))
    r = best_org.sort(["pub_date", "rating"], descending=[False, True]).group_by("pub_date", maintain_order=True)
    return r.agg(pl.col("organization").first().alias("org1"), pl.col("model_name").first().alias("model1"),
                 pl.col("rating").first().alias("r1"), pl.col("rating_lower").first().alias("lo1"),
                 pl.col("organization").slice(1, 1).first().alias("org2"), pl.col("model_name").slice(1, 1).first().alias("model2"),
                 pl.col("rating").slice(1, 1).first().alias("r2"), pl.col("rating_upper").slice(1, 1).first().alias("hi2"),
                 pl.col("as_of").first()).with_columns((pl.col("r1") - pl.col("r2")).alias("margin")).sort("pub_date")


# ---- OpenRouter weekly author totals ---------------------------------------------------------------
OR_API = "https://openrouter.ai/api/frontend/v1/rankings/author-market-share"
# KX*SHARE settle on the text *request* share by author (checked: OpenAI week of 2026-11-09 = 18.99%
# -> "above 18.9%" YES, "above 19.1%" NO); KXTOKENUSE on total tokens (default view: 211T, 207-215T).
OR_PARAMS = {"req_text": "metric=requests&modality=text", "tokens_all": ""}


def openrouter_rows(payload: dict[str, Any], fetched: datetime, params: str) -> list[dict[str, Any]]:
    """One row per (week starting Monday, author): tokens (or the metric asked), plus the fetch time."""
    data = (payload.get("data") or {})
    cached_at = data.get("cachedAt")
    cached = datetime.fromtimestamp(cached_at / 1000, UTC) if cached_at else None
    return [{"week": date.fromisoformat(w["x"][:10]), "author": a, "value": float(v), "fetched": fetched,
             "cached_at": cached, "params": params} for w in data.get("data") or [] for a, v in (w.get("ys") or {}).items()]


def openrouter_weekly(web: Web | None = None, params: str = "") -> pl.DataFrame:
    """Today's weekly history (52 weeks + the running week), cached per snapshot day."""
    web = web or Web(CACHE / "web" / SNAP)
    url = OR_API + (f"?{params}" if params else "")
    body, when = web.get(url)
    if body is None:
        raise RuntimeError("openrouter fetch failed")
    df = pl.DataFrame(openrouter_rows(json.loads(body), when, params))
    tot = df.group_by("week").agg(pl.col("value").sum().alias("total"))
    return df.join(tot, on="week").with_columns((100 * pl.col("value") / pl.col("total")).alias("share_pct"))


# ---- panels: quotes at fixed decision times ---------------------------------------------------------
LADDER_DECISIONS = {"T-14d": timedelta(days=14), "T-72h": timedelta(hours=72), "T-24h": timedelta(hours=24),
                    "T-6h": timedelta(hours=6)}
MAX_QUOTE_AGE = timedelta(hours=3)


def quotes_at(points: pl.DataFrame, candles: pl.DataFrame, max_age: timedelta = MAX_QUOTE_AGE) -> pl.DataFrame:
    """Attach the last hourly candle that *ended* at or before ``t`` (within ``max_age``): yes bid/ask,
    mid, spread and the volume traded in the 24 h before ``t``. Rows without a two-sided quote keep
    null quotes (a taker cannot trade them)."""
    c = (candles.select("ticker", pl.col("end_ts").alias("q_ts"), pl.col("yes_bid_close").alias("bid"),
                        pl.col("yes_ask_close").alias("ask"), "volume")
         .sort("ticker", "q_ts")
         .with_columns(pl.col("volume").fill_null(0).rolling_sum(24, min_samples=1).over("ticker").alias("vol24")))
    out = (points.sort("ticker", "t")
           .join_asof(c, left_on="t", right_on="q_ts", by="ticker", strategy="backward", check_sortedness=False))
    ok = (pl.col("q_ts").is_not_null() & ((pl.col("t") - pl.col("q_ts")) <= max_age)
          & (pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("ask") > pl.col("bid")))
    return out.with_columns(pl.when(ok).then(pl.col("bid")).alias("bid"), pl.when(ok).then(pl.col("ask")).alias("ask")
                            ).with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                                           (pl.col("ask") - pl.col("bid")).alias("spread"))


def decision_points(mk: pl.DataFrame, decisions: dict[str, timedelta]) -> pl.DataFrame:
    """One row per (market, decision label) with t = close - delta, kept only if the market was open."""
    d = pl.DataFrame({"label": list(decisions), "delta": list(decisions.values())})
    return (mk.join(d, how="cross").with_columns((pl.col("close_time") - pl.col("delta")).alias("t"))
            .filter(pl.col("t") >= pl.col("open_time")).drop("delta"))


def pava_decreasing(y: list[float]) -> list[float]:
    """Isotonic (non-increasing) fit by pool-adjacent-violators, equal weights."""
    blocks: list[list[float]] = []
    for v in y:
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, w2 = blocks.pop()
            v1, w1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    out: list[float] = []
    for v, w in blocks:
        out += [v] * int(w)
    return out


def implied_median(ks: list[float], p_above: list[float]) -> float | None:
    """Strike where the (isotonic) P(score > K) crosses 0.5, linearly interpolated."""
    if len(ks) < 2:
        return None
    p = pava_decreasing(p_above)
    for i in range(len(ks) - 1):
        if p[i] >= 0.5 > p[i + 1]:
            return ks[i] + (p[i] - 0.5) / (p[i] - p[i + 1]) * (ks[i + 1] - ks[i]) if p[i] != p[i + 1] else ks[i]
    return None


def rt_markets(mk: pl.DataFrame, series: list[str]) -> pl.DataFrame:
    """Review-score "Above K" ladders with the strike, outcome and the event's final-score bracket."""
    m = (mk.filter(pl.col("series_ticker").is_in(series))
         .with_columns(pl.coalesce(pl.col("floor_strike"),
                                   pl.col("yes_sub_title").str.extract(r"(?i)above\s+([\d.]+)").cast(pl.Float64)).alias("K"),
                       (pl.col("result") == "yes").cast(pl.Int8).alias("y"))
         .filter(pl.col("K").is_not_null()))
    br = m.group_by("event_ticker").agg(pl.col("K").filter(pl.col("y") == 1).max().alias("final_lo"),
                                        pl.col("K").filter(pl.col("y") == 0).min().alias("final_hi"),
                                        pl.col("open_time").min().alias("event_open"))
    return m.join(br, on="event_ticker").with_columns(
        (pl.col("open_time") > pl.col("event_open") + timedelta(hours=24)).alias("late_leg"),
        ((pl.col("K") % 10) == 0).alias("round_k"))


def add_ladder_features(p: pl.DataFrame) -> pl.DataFrame:
    """Per (event, label): implied median and each leg's distance to it (in strike points)."""
    meds = []
    for (ev, lab), g in p.filter(pl.col("mid").is_not_null()).group_by("event_ticker", "label"):
        g = g.sort("K")
        meds.append({"event_ticker": ev, "label": lab, "m50": implied_median(g["K"].to_list(), g["mid"].to_list()),
                     "legs_quoted": g.height})
    md = pl.DataFrame(meds, schema={"event_ticker": pl.String, "label": pl.String, "m50": pl.Float64, "legs_quoted": pl.Int64})
    return p.join(md, on=["event_ticker", "label"], how="left").with_columns((pl.col("K") - pl.col("m50")).alias("k_minus_m50"))


# ---- trades, statistics and ledgers --------------------------------------------------------------------
STAKE = 3.0           # ~$3 per order
MAX_N = 25            # displayed size is not in candles: cap contracts per order
MAX_SPREAD = 0.10     # never cross a book wider than 10c


def take(rows: pl.DataFrame, side: str) -> pl.DataFrame:
    """Taker fills: YES at the ask or NO at 1 - bid, n ~ $3 (<= MAX_N), fee rounded up per order;
    PnL per contract in dollars and the order's dollar PnL / cost."""
    px = pl.col("ask") if side == "yes" else 1 - pl.col("bid")
    win = pl.col("y") if side == "yes" else 1 - pl.col("y")
    r = (rows.filter(pl.col("bid").is_not_null() & (pl.col("spread") <= MAX_SPREAD))
         .with_columns(px.alias("px"), win.cast(pl.Float64).alias("win"), pl.lit(side).alias("side"))
         .with_columns(pl.min_horizontal(pl.lit(MAX_N), (STAKE / pl.col("px")).floor().clip(1, None)).cast(pl.Int64).alias("n")))
    fees = [taker_fee(n, px, m) for n, px, m in zip(r["n"], r["px"], r["fee_multiplier"].fill_null(1.0), strict=True)]
    return (r.with_columns(pl.Series("fee", fees, dtype=pl.Float64))
            .with_columns((pl.col("n") * pl.col("px") + pl.col("fee")).alias("cost_usd"),
                          (pl.col("n") * pl.col("win")).alias("payoff_usd"))
            .with_columns((pl.col("payoff_usd") - pl.col("cost_usd")).alias("pnl_usd"),
                          ((pl.col("payoff_usd") - pl.col("cost_usd")) / pl.col("n")).alias("pnl_c")))


def one_per_event(tr: pl.DataFrame, key: pl.Expr) -> pl.DataFrame:
    """Keep one order per event (the row with the smallest ``key``) so an event risks <= ~$3-5."""
    return tr.sort(key).group_by("event_ticker", maintain_order=True).head(1)


def cluster_t(x: np.ndarray, g: np.ndarray) -> float | None:
    """Mean / cluster-robust (CR1) standard error, clusters ``g``."""
    if len(x) < 2:
        return None
    mu = float(np.mean(x))
    df = pl.DataFrame({"g": g, "e": x - mu}).group_by("g").agg(pl.col("e").sum())
    k = df.height
    if k < 2:
        return None
    se = math.sqrt(k / (k - 1) * float((df["e"] ** 2).sum())) / len(x)
    return mu / se if se > 0 else None


def summarize(tr: pl.DataFrame) -> dict[str, Any]:
    """c/contract, PnL per $ risked, fills, events, t by event and by close week, % positive weeks."""
    if tr.is_empty():
        return {"fills": 0}
    tr = tr.with_columns(pl.col("close_time").dt.truncate("1w").alias("wk"))
    x = tr["pnl_c"].to_numpy()
    wk = tr.group_by("wk").agg(pl.col("pnl_usd").sum())
    return {"fills": tr.height, "events": tr["event_ticker"].n_unique(), "weeks": wk.height,
            "contracts": int(tr["n"].sum()), "c_per_contract": round(100 * float(tr["pnl_usd"].sum() / tr["n"].sum()), 2),
            "mean_c_per_fill": round(100 * float(np.mean(x)), 2),
            "pnl_per_usd": round(float(tr["pnl_usd"].sum() / tr["cost_usd"].sum()), 4),
            "pnl_usd": round(float(tr["pnl_usd"].sum()), 2),
            "t_event": _round(cluster_t(x, tr["event_ticker"].to_numpy())),
            "t_week": _round(cluster_t(x, tr["wk"].cast(pl.String).to_numpy())),
            "pct_pos_weeks": round(100 * float((wk["pnl_usd"] > 0).mean()), 0),
            "mean_px": round(float(tr["px"].mean()), 3), "hit": round(float(tr["win"].mean()), 3),
            "p_fair": _round(p_at_least(tr["win"].sum(), (tr["px"] + tr["fee"] / tr["n"]).clip(0, 1).to_list()), 4)}


def p_at_least(wins: float, probs: list[float]) -> float | None:
    """Exact Poisson-binomial P(W >= wins) if each fill wins with its break-even price (price + fee per
    contract): the honest significance when nearly every fill wins and a t-stat is meaningless."""
    if not probs:
        return None
    dist = np.zeros(len(probs) + 1)
    dist[0] = 1.0
    for q in probs:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= 1 - q
    return float(dist[round(wins):].sum())


def _round(v: float | None, k: int = 2) -> float | None:
    return None if v is None else round(v, k)


def marks_from(candles: pl.DataFrame, tickers: list[str]) -> pl.DataFrame:
    """Hourly YES mids for ``tickers`` (same rule as vol_metrics.load_marks, from this study's cache)."""
    c = candles.filter(pl.col("ticker").is_in(tickers))
    two = (pl.col("yes_bid_close").is_not_null() & pl.col("yes_ask_close").is_not_null()
           & (pl.col("yes_ask_close") > pl.col("yes_bid_close")))
    mid = pl.when(two).then((pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2).otherwise(pl.col("price_close"))
    return (c.with_columns(mid.alias("yes_mid")).filter(pl.col("yes_mid").is_not_null())
            .select("ticker", pl.col("end_ts").alias("ts").cast(pl.Datetime("us", "UTC")), "yes_mid")
            .unique(["ticker", "ts"]).sort("ticker", "ts"))


def ledger(tr: pl.DataFrame, strategy: str, variant: str, split: str) -> pl.DataFrame:
    return tr.select(pl.lit(strategy).alias("strategy"), pl.lit(variant).alias("variant"), pl.lit(split).alias("split"),
                     (pl.col("ticker") + "@" + pl.col("label")).alias("struct_id"), "event_ticker", "ticker", "side",
                     "n", pl.col("t").alias("t_in"), "cost_usd",
                     pl.coalesce(pl.col("settlement_ts"), pl.col("close_time")).alias("settle_ts"), "payoff_usd")


def book_metrics(tr: pl.DataFrame, candles: pl.DataFrame) -> dict[str, Any]:
    from albert.research.vol_metrics import ledger_metrics

    if tr.is_empty():
        return {}
    led = ledger(tr, "x", "x", "x")
    m = ledger_metrics(led, marks_from(candles, led["ticker"].unique().to_list()))
    ends = pl.coalesce(pl.col("settlement_ts"), pl.col("close_time"))
    ev = pl.concat([tr.select(pl.col("t").alias("ts"), pl.col("cost_usd").alias("d")),
                    tr.select(ends.alias("ts"), (-pl.col("cost_usd")).alias("d"))]).sort("ts")
    m["max_open_cost_usd"] = round(float(ev["d"].cum_sum().max()), 2)
    keep = ["max_open_cost_usd", "sharpe", "sharpe_ci90", "sharpe_ex_top5", "ann_vol_pct", "max_dd_usd", "usd_per_week",
            "usd_per_week_ex_top5", "weeks_positive_pct", "sortino", "positions", "events", "first_day", "last_day"]
    return {k: m.get(k) for k in keep}


# ---- LMArena family panel ----------------------------------------------------------------------------
LEG_ORGS = {  # Kalshi leg label (lower case) -> LMArena organization (lower case)
    "gemini": "google", "google": "google", "chatgpt": "openai", "openai": "openai", "gpt": "openai",
    "claude": "anthropic", "anthropic": "anthropic", "grok": "xai", "xai": "xai", "qwen": "alibaba",
    "alibaba": "alibaba", "kimi": "moonshot", "moonshot": "moonshot", "moonshot ai": "moonshot",
    "deepseek": "deepseek", "glm": "zhipu", "zhipu": "zhipu", "z.ai": "zhipu", "zai": "zhipu",
    "minimax": "minimax", "xiaomi": "xiaomi", "mimo": "xiaomi", "tencent": "tencent", "hunyuan": "tencent",
    "bytedance": "bytedance", "dola": "bytedance", "doubao": "bytedance", "seed": "bytedance",
    "meta": "meta", "llama": "meta", "amazon nova": "amazon", "amazon": "amazon", "mistral": "mistral",
    "stepfun": "stepfun", "meituan": "meituan", "longcat": "meituan", "baidu": "baidu", "ernie": "baidu",
}
CHINA_ORGS = {"alibaba", "moonshot", "deepseek", "zhipu", "minimax", "xiaomi", "tencent", "bytedance", "stepfun",
              "meituan", "baidu", "iflytek", "01 ai", "01.ai", "sensetime", "kuaishou", "ant-group", "inclusionai"}


def norm_model(name: str | None) -> str | None:
    """LMArena renamed models mid-2026 ("gemini-4-argon-high" -> "Gemini 4 Argon (High)"): one key."""
    return None if name is None else re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-")
LMA_DECISIONS = {"open+24h": None, "T-72h": timedelta(hours=72), "T-24h": timedelta(hours=24)}


def norm_org(org: str | None) -> str | None:
    """LMArena organization strings vary ("Google", "google", "xAI", "Moonshot AI"): normalise."""
    if org is None:
        return None
    o = org.strip().lower()
    if o in ("zai", "z-ai"):
        return "zhipu"
    for k, v in (("moonshot", "moonshot"), ("z.ai", "zhipu"), ("zhipu", "zhipu"), ("xai", "xai"), ("x-ai", "xai"),
                 ("alibaba", "alibaba"), ("qwen", "alibaba"), ("bytedance", "bytedance"), ("meituan", "meituan"),
                 ("amazon", "amazon"), ("deepseek", "deepseek"), ("minimax", "minimax"), ("tencent", "tencent")):
        if k in o:
            return v
    return o


def org_board(h: pl.DataFrame, china_only: bool = False) -> pl.DataFrame:
    """Per snapshot: best model per org (rating order), the leader org/model and the margin to #2 org."""
    x = h.with_columns(pl.col("organization").map_elements(norm_org, return_dtype=pl.String).alias("org"))
    if china_only:
        x = x.filter(pl.col("org").is_in(list(CHINA_ORGS)))
    best = (x.sort("rating", descending=True).group_by("pub_date", "org", maintain_order=True)
            .agg(pl.col("model_name").first(), pl.col("rating").first(), pl.col("as_of").first()))
    return (best.sort(["pub_date", "rating"], descending=[False, True]).group_by("pub_date", maintain_order=True)
            .agg(pl.col("org").first().alias("org1"), pl.col("model_name").first().alias("model1"),
                 pl.col("rating").first().alias("r1"), pl.col("org").slice(1, 1).first().alias("org2"),
                 pl.col("rating").slice(1, 1).first().alias("r2"), pl.col("as_of").first())
            .with_columns((pl.col("r1") - pl.col("r2")).alias("margin")).sort("pub_date"))


def model_board(h: pl.DataFrame) -> pl.DataFrame:
    """Per snapshot: the #1 model and its margin to the #2 model (any org)."""
    h = h.with_columns(pl.col("model_name").map_elements(norm_model, return_dtype=pl.String))
    return (h.sort(["pub_date", "rating"], descending=[False, True]).group_by("pub_date", maintain_order=True)
            .agg(pl.col("model_name").first().alias("org1"), pl.col("model_name").first().alias("model1"),
                 pl.col("rating").first().alias("r1"), pl.col("model_name").slice(1, 1).first().alias("org2"),
                 pl.col("rating").slice(1, 1).first().alias("r2"), pl.col("as_of").first())
            .with_columns((pl.col("r1") - pl.col("r2")).alias("margin")).sort("pub_date"))


def stay_pairs(board: pl.DataFrame, max_h: int = 45) -> pl.DataFrame:
    """All snapshot pairs (s, s') with 1 <= days <= max_h: did s's leader still lead at s'?"""
    b = board.select("pub_date", "org1", "margin").with_row_index("i")
    a = b.join(b, how="cross", suffix="_2").filter(
        ((pl.col("pub_date_2") - pl.col("pub_date")).dt.total_days()).is_between(1, max_h))
    return a.with_columns((pl.col("pub_date_2") - pl.col("pub_date")).dt.total_days().alias("h"),
                          (pl.col("org1") == pl.col("org1_2")).cast(pl.Int8).alias("stay"))


def fit_stay(pairs: pl.DataFrame) -> np.ndarray:
    """Logistic P(stay) = s(a + b*margin + c*log h), plain IRLS (pairs overlap, so no standard errors)."""
    pairs = pairs.filter(pl.col("margin").is_not_null() & pl.col("margin").is_not_nan())
    x = np.column_stack([np.ones(pairs.height), pairs["margin"].to_numpy(), np.log(pairs["h"].to_numpy())])
    y = pairs["stay"].to_numpy().astype(float)
    w = np.zeros(3)
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))
        g = x.T @ (y - p)
        hess = x.T @ (x * (p * (1 - p) + 1e-9)[:, None]) + 1e-6 * np.eye(3)
        step = np.linalg.solve(hess, g)
        w = w + step
        if np.abs(step).max() < 1e-8:
            break
    return w


def p_stay(w: np.ndarray, margin: float, h: float) -> float:
    return float(1 / (1 + math.exp(-(w[0] + w[1] * margin + w[2] * math.log(max(h, 0.5))))))


def leg_key(series: str, label: str | None) -> str | None:
    """Map a Kalshi leg to the board's key: an org for company legs, a model name for KXTOPMODEL."""
    if label is None:
        return None
    if series == "KXTOPMODEL":
        return norm_model(label)
    lab = label.strip().lower()
    if lab in LEG_ORGS:
        return LEG_ORGS[lab]
    first = lab.split()[0] if lab.split() else lab
    return LEG_ORGS.get(first, lab)


def lma_panel(mk: pl.DataFrame, candles: pl.DataFrame, boards: dict[str, pl.DataFrame],
              weights: dict[str, np.ndarray]) -> pl.DataFrame:
    """LMArena-family legs at open + 24 h, T-72h and T-24h with the as-of leader, margin and P(stay)."""
    m = mk.filter(pl.col("series_ticker").is_in(LMA_SERIES)).with_columns(
        (pl.col("result") == "yes").cast(pl.Int8).alias("y"),
        pl.col("open_time").min().over("event_ticker").alias("event_open"))
    pts = []
    for lab, delta in LMA_DECISIONS.items():
        t = (pl.col("event_open") + timedelta(hours=24)) if delta is None else (pl.col("close_time") - delta)
        pts.append(m.with_columns(t.alias("t"), pl.lit(lab).alias("label")))
    p = pl.concat(pts).filter(pl.col("t") >= pl.col("open_time"))
    p = quotes_at(p, candles)
    rows = []
    for r in p.iter_rows(named=True):
        board_name = {"KXLLM1": "text_org", "KXTOPMODEL": "text_model", "KXCHINAAI": "china_org",
                      "KXTECHRANKLISTAICODE": "webdev_org"}[r["series_ticker"]]
        b = boards[board_name]
        known = b.filter(pl.col("as_of").is_not_null() & (pl.col("as_of") <= r["t"]))
        if known.is_empty():
            continue
        snap = known.row(-1, named=True)
        h = (r["close_time"].date() - snap["pub_date"]).days
        ps = p_stay(weights[board_name], snap["margin"], h)
        key = leg_key(r["series_ticker"], r["yes_sub_title"])
        lead = key is not None and snap["org1"] is not None and key.lower() == str(snap["org1"]).lower()
        rows.append({**{k: r[k] for k in ("ticker", "event_ticker", "series_ticker", "yes_sub_title", "label", "t",
                                         "close_time", "settlement_ts", "y", "bid", "ask", "mid", "spread", "vol24")},
                     "board": board_name, "snap_date": snap["pub_date"], "snap_as_of": snap["as_of"],
                     "leader": snap["org1"], "margin": snap["margin"], "h_days": h, "p_stay": ps,
                     "is_leader": lead, "p_model": ps if lead else None, "p_upper": ps if lead else 1 - ps})
    out = pl.DataFrame(rows, infer_schema_length=None)
    fee = mk.select("ticker").with_columns(pl.lit(1.0).alias("fee_multiplier"))
    return out.join(fee, on="ticker", how="left")


def lma_settle_check(mk: pl.DataFrame, boards: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Does the dataset's leader on the settlement date match Kalshi's YES leg? (proxy check)"""
    names = {"KXLLM1": "text_org", "KXTOPMODEL": "text_model", "KXCHINAAI": "china_org", "KXTECHRANKLISTAICODE": "webdev_org"}
    w = (mk.filter(pl.col("series_ticker").is_in(LMA_SERIES) & (pl.col("result") == "yes"))
         .group_by("event_ticker").agg(pl.col("series_ticker").first(), pl.col("yes_sub_title").first().alias("winner"),
                                       pl.col("close_time").max()))
    rows = []
    for r in w.iter_rows(named=True):
        b = boards[names[r["series_ticker"]]].filter(pl.col("pub_date") <= r["close_time"].date())
        if b.is_empty():
            continue
        snap = b.row(-1, named=True)
        key = leg_key(r["series_ticker"], r["winner"])
        rows.append({"event_ticker": r["event_ticker"], "series_ticker": r["series_ticker"], "winner": r["winner"],
                     "board_leader": snap["org1"], "snap_date": snap["pub_date"],
                     "match": key is not None and key.lower() == str(snap["org1"]).lower()})
    return pl.DataFrame(rows)


# ---- OpenRouter panel ----------------------------------------------------------------------------------
OR_AUTHOR = {"KXOPENSHARE": "openai", "KXGOOGSHARE": "google", "KXDEEPSHARE": "deepseek", "KXANTHSHARE": "anthropic",
             "KXTENCENTSHARE": "tencent", "KXXIAOMISHARE": "xiaomi", "KXBABASHARE": "qwen", "KXZAISHARE": "z-ai"}
OR_DECISIONS = {"open+1h": timedelta(hours=1), "open+24h": timedelta(hours=24)}
_MON = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                         "september", "october", "november", "december"])}
_MON3 = {m[:3]: v for m, v in _MON.items()}


def or_week(rules: str | None, event_ticker: str) -> date | None:
    """Monday that starts the measured week: 'week of Nov 9, 2026' / 'for Nov 9–15, 2026' / 'week ending
    Aug 16, 2026' (Sunday), else ticker date - 7 d."""
    e = re.search(r"week ending\s+([A-Za-z]+)\.?\s+(\d{1,2}),\s*(\d{4})", rules or "")
    if e and e.group(1)[:3].lower() in _MON3:
        return date(int(e.group(3)), _MON3[e.group(1)[:3].lower()], int(e.group(2))) - timedelta(days=6)
    m = re.search(r"(?:week of|for)\s+([A-Za-z]+)\.?\s+(\d{1,2})(?:[^,]*)?,\s*(\d{4})", rules or "")
    if m and m.group(1)[:3].lower() in _MON3:
        return date(int(m.group(3)), _MON3[m.group(1)[:3].lower()], int(m.group(2)))
    t = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", event_ticker or "")
    if t:
        return date(2000 + int(t.group(1)), _MON3[t.group(2).lower()], int(t.group(3))) - timedelta(days=7)
    return None


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def or_residuals(w: pl.DataFrame, value: str, *, end: date, total: bool = False) -> np.ndarray:
    """Week-on-week changes known before ``end``: logit share (authors) or log total (tokens)."""
    if total:
        t = w.group_by("week").agg(pl.col("value").sum().alias("v")).sort("week")
        t = t.filter(pl.col("week") + timedelta(days=7) <= end)
        v = np.log(t["v"].to_numpy())
        return np.diff(v)[1:] if len(v) > 2 else np.array([])
    x = w.filter(pl.col("author") != "Others").sort("week").with_columns(
        pl.col("week").shift(1).over("author").alias("pw"), pl.col(value).shift(1).over("author").alias("pv"))
    x = x.filter((pl.col("week") - pl.col("pw") == timedelta(days=7)) & (pl.col("week") + timedelta(days=7) <= end))
    return _logit(x[value].to_numpy() / 100) - _logit(x["pv"].to_numpy() / 100)


def or_panel(mk: pl.DataFrame, candles: pl.DataFrame, req: pl.DataFrame, tok: pl.DataFrame,
             fit_end: date) -> pl.DataFrame:
    """OpenRouter ladders at open + 1 h / open + 24 h with a last-full-week model (residuals from
    weeks ending before ``fit_end``) and today's final value for the measured week (check only)."""
    m = (mk.filter(pl.col("series_ticker").is_in(list(OR_AUTHOR) + ["KXTOKENUSE"]))
         .with_columns((pl.col("result") == "yes").cast(pl.Int8).alias("y"),
                       pl.col("open_time").min().over("event_ticker").alias("event_open"),
                       pl.coalesce(pl.col("floor_strike"), pl.col("yes_sub_title").str.extract(r"([\d.]+)").cast(pl.Float64)).alias("K")))
    m = m.with_columns(pl.struct("rules_primary", "event_ticker").map_elements(
        lambda r: or_week(r["rules_primary"], r["event_ticker"]), return_dtype=pl.Date).alias("week"))
    pts = [m.with_columns((pl.col("event_open") + d).alias("t"), pl.lit(lab).alias("label")) for lab, d in OR_DECISIONS.items()]
    p = quotes_at(pl.concat(pts).filter(pl.col("t") >= pl.col("open_time")), candles)
    r_share = or_residuals(req, "share_pct", end=fit_end)
    r_tok = or_residuals(tok, "value", end=fit_end, total=True)
    req_ix = {(r["week"], r["author"]): r["share_pct"] for r in req.iter_rows(named=True)}
    tot = tok.group_by("week").agg(pl.col("value").sum().alias("v"))
    tok_ix = dict(zip(tot["week"].to_list(), tot["v"].to_list(), strict=True))
    rows = []
    for r in p.iter_rows(named=True):
        wk = r["week"]
        if wk is None or r["K"] is None:
            continue
        prev = wk - timedelta(days=7)
        known = r["t"] >= datetime(wk.year, wk.month, wk.day, 6, tzinfo=UTC)   # last week final by Mon 06 UTC
        if r["series_ticker"] == "KXTOKENUSE":
            pv, fin = tok_ix.get(prev), tok_ix.get(wk)
            pm = float(np.mean(pv * np.exp(r_tok) / 1e12 > r["K"])) if (pv and known and len(r_tok)) else None
            fin_v = fin / 1e12 if fin else None
        else:
            a = OR_AUTHOR[r["series_ticker"]]
            pv, fin = req_ix.get((prev, a)), req_ix.get((wk, a))
            if pv is not None and known and len(r_share):
                sim = np.round(100 / (1 + np.exp(-(_logit(np.array([pv / 100])) + r_share))), 1)
                pm = float(np.mean(sim > r["K"]))
            else:
                pm = None
            fin_v = fin
        rows.append({**{k: r[k] for k in ("ticker", "event_ticker", "series_ticker", "yes_sub_title", "label", "t",
                                         "close_time", "settlement_ts", "y", "bid", "ask", "mid", "spread", "vol24", "K", "week")},
                     "prev_value": (pv / 1e12 if r["series_ticker"] == "KXTOKENUSE" and pv else pv), "p_model": pm,
                     "final_today": fin_v})
    out = pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.lit(1.0).alias("fee_multiplier"))
    return out.with_columns(
        pl.when(pl.col("final_today").is_null()).then(None)
        .when(pl.col("series_ticker") == "KXTOKENUSE").then(pl.col("final_today") > pl.col("K"))
        .otherwise(pl.col("final_today").round(1) > pl.col("K")).cast(pl.Int8).alias("y_today"))


# ---- rules (shared by dev screen and holdout) -----------------------------------------------------------
def rule_trades(panel: pl.DataFrame, rule: dict[str, Any]) -> pl.DataFrame:
    """Taker fills of one rule. ``rule``: label, side ('yes'/'no'/'model'), mid band [lo, hi), optional
    filters (round_k, late_leg, is_leader, below/above median), theta for model rules, and
    one order per event (the leg whose mid is closest to ``pick_mid``, or the largest model edge)."""
    x = panel.filter(pl.col("label") == rule["label"])
    if "series" in rule:
        x = x.filter(pl.col("series_ticker").is_in(rule["series"]))
    if "mid" in rule:
        lo, hi = rule["mid"]
        x = x.filter(pl.col("mid").is_between(lo, hi, closed="left"))
    for col in ("round_k", "late_leg", "is_leader"):
        if col in rule:
            x = x.filter(pl.col(col) == rule[col])
    if rule.get("k_side") == "above_m50":
        x = x.filter(pl.col("k_minus_m50") > 0)
    elif rule.get("k_side") == "below_m50":
        x = x.filter(pl.col("k_minus_m50") < 0)
    if rule["side"] in ("yes", "no"):
        tr = take(x, rule["side"])
        return one_per_event(tr, (pl.col("mid") - rule.get("pick_mid", 0.5)).abs()) if rule.get("one_per_event", True) else tr
    pcol = rule.get("p_col", "p_model")
    x = x.filter(pl.col(pcol).is_not_null())
    th = rule.get("theta", 0.0)
    ty = take(x, "yes").with_columns((pl.col(pcol) - pl.col("px") - pl.col("fee") / pl.col("n")).alias("edge"))
    tn = take(x, "no").with_columns(((1 - pl.col(pcol)) - pl.col("px") - pl.col("fee") / pl.col("n")).alias("edge"))
    if rule.get("upper_bound_only"):   # p_col is an upper bound on P(YES): only NO is safe
        ty = ty.head(0)
    tr = pl.concat([ty, tn]).filter(pl.col("edge") >= th)
    return one_per_event(tr, -pl.col("edge")) if rule.get("one_per_event", True) else tr


def calibration(panel: pl.DataFrame, by: list[str], buckets=(0.02, 0.15, 0.35, 0.65, 0.85, 0.98)) -> pl.DataFrame:
    """y - mid by mid bucket (two-sided quotes only), clustered t by event."""
    x = panel.filter(pl.col("mid").is_not_null() & pl.col("mid").is_between(buckets[0], buckets[-1]))
    x = x.with_columns(pl.col("mid").cut(list(buckets[1:-1]), left_closed=True).cast(pl.String).alias("bucket"),
                       (pl.col("y") - pl.col("mid")).alias("resid"))
    rows = []
    for key, g in x.group_by(by + ["bucket"]):
        rows.append({**dict(zip(by + ["bucket"], key, strict=True)), "legs": g.height, "events": g["event_ticker"].n_unique(),
                     "mean_mid": round(float(g["mid"].mean()), 3), "yes_rate": round(float(g["y"].mean()), 3),
                     "resid_c": round(100 * float(g["resid"].mean()), 2),
                     "t_event": _round(cluster_t(g["resid"].to_numpy(), g["event_ticker"].to_numpy()))})
    return pl.DataFrame(rows).sort(by + ["bucket"]) if rows else pl.DataFrame()


def ladder_arbs(panel: pl.DataFrame) -> pl.DataFrame:
    """Executable monotonicity violations: bid(K_hi) - ask(K_lo) > fees for one contract each."""
    rows = []
    for (ev, lab), g in panel.filter(pl.col("bid").is_not_null()).group_by("event_ticker", "label"):
        g = g.sort("K")
        ks, bids, asks = g["K"].to_list(), g["bid"].to_list(), g["ask"].to_list()
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                gain = bids[j] - asks[i] - taker_fee(1, asks[i]) - taker_fee(1, 1 - bids[j])
                if gain > 0:
                    rows.append({"event_ticker": ev, "label": lab, "k_lo": ks[i], "k_hi": ks[j], "gain": gain})
    return pl.DataFrame(rows, schema={"event_ticker": pl.String, "label": pl.String, "k_lo": pl.Float64,
                                      "k_hi": pl.Float64, "gain": pl.Float64})


# ---- build panels, dev screen ---------------------------------------------------------------------------
def fee_mults() -> pl.DataFrame:
    return pl.read_parquet(TABLES / "kalshi_series.parquet", columns=["series_ticker", "fee_multiplier"])


def build_panels(out: Path = OUT) -> dict[str, pl.DataFrame]:
    """All three panels from the cached markets/candles/alt data (written to OUT/panel_*.parquet)."""
    mk = pl.read_parquet(out / "markets.parquet")
    candles = load_candles()
    fm = fee_mults()
    panels: dict[str, pl.DataFrame] = {}
    # review-score ladders (price only)
    for fam, series in (("rt", RT_SERIES), ("mc", MC_SERIES)):
        m = rt_markets(mk, series).join(fm, on="series_ticker", how="left")
        p = quotes_at(decision_points(m, LADDER_DECISIONS), candles)
        panels[fam] = add_ladder_features(p)
    # LMArena
    text = pl.read_parquet(out / "lmarena_text_overall.parquet")
    web = pl.read_parquet(out / "lmarena_webdev_overall.parquet")
    boards = {"text_org": org_board(text), "text_model": model_board(text), "china_org": org_board(text, china_only=True),
              "webdev_org": org_board(web)}
    dev_end = DEV_END.date()
    weights = {k: fit_stay(stay_pairs(b).filter(pl.col("pub_date_2") < dev_end)) for k, b in boards.items()}
    (out / "lma_weights.json").write_text(json.dumps({k: [float(v) for v in w] for k, w in weights.items()}, indent=1))
    for k, b in boards.items():
        _save(b, out / f"board_{k}.parquet")
    panels["lma"] = lma_panel(mk, candles, boards, weights)
    _save(lma_settle_check(mk, boards), out / "lma_settle_check.parquet")
    # OpenRouter
    req = pl.read_parquet(out / "openrouter_req_text.parquet")
    tok = pl.read_parquet(out / "openrouter_tokens_all.parquet")
    panels["or"] = or_panel(mk, candles, req, tok, fit_end=dev_end)
    for k, v in panels.items():
        v = v.with_columns((pl.col("close_time") < DEV_END).alias("dev"))
        panels[k] = v
        _save(v, out / f"panel_{k}.parquet")
    return panels


def dev_cells() -> list[dict[str, Any]]:
    """Every dev cell tried, in a fixed order (family, hypothesis, rule)."""
    cells: list[dict[str, Any]] = []
    for fam in ("rt", "mc"):
        for lab in LADDER_DECISIONS:
            base = {"family": fam, "label": lab}
            cells += [
                {**base, "h": "H1/H4", "name": "centre_no", "side": "no", "mid": (0.35, 0.65)},
                {**base, "h": "H1/H4", "name": "centre_yes", "side": "yes", "mid": (0.35, 0.65)},
                {**base, "h": "H1", "name": "above_median_no", "side": "no", "mid": (0.15, 0.5), "k_side": "above_m50", "pick_mid": 0.35},
                {**base, "h": "H1", "name": "below_median_yes", "side": "yes", "mid": (0.5, 0.85), "k_side": "below_m50", "pick_mid": 0.65},
                {**base, "h": "H2", "name": "tail_no", "side": "no", "mid": (0.02, 0.15), "pick_mid": 0.08},
                {**base, "h": "H5", "name": "fav_yes", "side": "yes", "mid": (0.85, 0.97), "pick_mid": 0.91},
                {**base, "h": "H12", "name": "centre_no_round", "side": "no", "mid": (0.35, 0.65), "round_k": True},
                {**base, "h": "H12", "name": "centre_no_nonround", "side": "no", "mid": (0.35, 0.65), "round_k": False},
                {**base, "h": "H6", "name": "late_centre_no", "side": "no", "mid": (0.35, 0.65), "late_leg": True},
                {**base, "h": "H6", "name": "late_centre_yes", "side": "yes", "mid": (0.35, 0.65), "late_leg": True},
            ]
    for lab in LMA_DECISIONS:
        base = {"family": "lma", "label": lab}
        for th in (0.0, 0.03, 0.06):
            cells += [{**base, "h": "H7", "name": f"leader_model_th{th}", "side": "model", "is_leader": True, "theta": th},
                      {**base, "h": "H7", "name": f"challenger_no_ub_th{th}", "side": "model", "is_leader": False,
                       "p_col": "p_upper", "upper_bound_only": True, "theta": th}]
        cells += [{**base, "h": "H8", "name": "challenger_no_5_35", "side": "no", "mid": (0.05, 0.35), "is_leader": False, "pick_mid": 0.2},
                  {**base, "h": "H7", "name": "leader_yes_50_97", "side": "yes", "mid": (0.5, 0.97), "is_leader": True, "pick_mid": 0.8}]
    for lab in OR_DECISIONS:
        base = {"family": "or", "label": lab}
        for th in (0.0, 0.03, 0.06, 0.10):
            cells.append({**base, "h": "H9", "name": f"model_th{th}", "side": "model", "theta": th})
        cells += [{**base, "h": "H9", "name": "centre_no", "side": "no", "mid": (0.35, 0.65)},
                  {**base, "h": "H2", "name": "tail_no", "side": "no", "mid": (0.02, 0.15), "pick_mid": 0.08}]
    return cells


def run_dev(out: Path = OUT, panels: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    panels = panels or {k: pl.read_parquet(out / f"panel_{k}.parquet") for k in ("rt", "mc", "lma", "or")}
    rows = []
    for c in dev_cells():
        p = panels[c["family"]].filter(pl.col("dev"))
        rule = {k: v for k, v in c.items() if k not in ("family", "h", "name")}
        rows.append({"family": c["family"], "h": c["h"], "label": c["label"], "rule": c["name"], **summarize(rule_trades(p, rule))})
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(out / "dev_cells.csv")
    cal = []
    for fam in ("rt", "mc", "lma", "or"):
        by = ["label", "is_leader"] if fam == "lma" else ["label"]
        c = calibration(panels[fam].filter(pl.col("dev")), by)
        if c.height:
            cal.append(c.with_columns(pl.lit(fam).alias("family")))
    if cal:
        pl.concat(cal, how="diagonal_relaxed").write_csv(out / "dev_calibration.csv")
    return df


# ---- frozen rules, holdout ------------------------------------------------------------------------------
def freeze(rules: list[dict[str, Any]], note: str, out: Path = OUT) -> Path:
    """Write primary.json (rule 0 = primary, then <= 2 secondaries); refuses to overwrite."""
    path = out / "primary.json"
    if path.exists():
        raise FileExistsError(f"{path} is frozen; holdout must use it as is")
    if not 1 <= len(rules) <= 3:
        raise ValueError("one primary + at most two secondaries")
    path.write_text(json.dumps({"frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"), "note": note,
                                "dev": "closes < 2026-10-01", "holdout": "closes >= 2026-10-01 (latest settled)",
                                "execution": "taker at ask / 1 - bid, fee 0.07*mult*n*P(1-P) rounded up per order, "
                                             f"~${STAKE:.0f}/order (<= {MAX_N} contracts), one order per event, spread <= {MAX_SPREAD}",
                                "rules": rules}, indent=1, default=str))
    return path


def run_holdout(out: Path = OUT) -> dict[str, Any]:
    spec = json.loads((out / "primary.json").read_text())
    candles = load_candles()
    res: dict[str, Any] = {"frozen_utc": spec["frozen_utc"], "rules": []}
    (out / "ledgers").mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(spec["rules"]):
        panel = pl.read_parquet(out / f"panel_{r['family']}.parquet")
        rule = {k: (tuple(v) if k == "mid" else v) for k, v in r.items() if k not in ("family", "h", "name")}
        row: dict[str, Any] = {"role": "primary" if i == 0 else "secondary", "name": r["name"], "family": r["family"]}
        for split, f in (("dev", pl.col("dev")), ("holdout", ~pl.col("dev")), ("all", pl.lit(True))):
            tr = rule_trades(panel.filter(f), rule)
            row[split] = summarize(tr)
            row[f"{split}_book"] = book_metrics(tr, candles)
            if split != "all" and tr.height:
                _save(ledger(tr, f"{NAME}_{r['name']}", r["family"], split), out / "ledgers" / f"ledger_{r['name']}_{split}.parquet")
                _save(tr, out / f"trades_{r['name']}_{split}.parquet")
        res["rules"].append(row)
    (out / "holdout.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ---- report helpers ----------------------------------------------------------------------------------------
def md_table(df: pl.DataFrame, floats: int = 3) -> str:
    """A plain markdown table (rounded floats, no index)."""
    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.{floats}g}" if abs(v) < 1e-3 or abs(v) >= 1e5 else f"{round(v, floats)}"
        return str(v).replace("|", "/")
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.iter_rows()]
    return "\n".join(lines)


def write_report() -> Path:
    """report.md from the saved study outputs (universe, robots, dev cells, primary.json, holdout.json)."""
    md = md_table
    u = pl.read_parquet(OUT / "universe.parquet")
    uni = u.select(pl.col("family"), pl.col("series").str.split(",").list.head(4).list.join(", ").alias("series (top 4 by volume)"),
                   pl.col("events").alias("events Aug17-Nov17"), pl.col("events_per_month").round(1).alias("events/month"),
                   (pl.col("volume") / 1e6).round(2).alias("volume (M contracts)"),
                   pl.col("vol_weighted_spread").round(3).alias("typical spread"),
                   pl.col("settlement_source").str.slice(0, 38).alias("settles on"), pl.col("asof_data").alias("as-of data"),
                   pl.col("robots"), pl.col("feasibility"), (pl.col("opportunity") / 1e6).round(2).alias("rank score")).head(12)
    rb = pl.DataFrame(json.loads((OUT / "robots.json").read_text())).select("host", "path", "http_status", "allowed")
    dev = pl.read_csv(OUT / "dev_cells.csv")
    keep = ["family", "label", "rule", "fills", "weeks", "c_per_contract", "pnl_per_usd", "t_week", "hit", "mean_px", "p_fair"]
    notable = dev.filter((pl.col("fills") >= 8) & ((pl.col("t_week").abs() >= 1.4) | (pl.col("p_fair") <= 0.2))).select(keep)
    ho = json.loads((OUT / "holdout.json").read_text())
    rows, books = [], []
    for r in ho["rules"]:
        for sp in ("dev", "holdout", "all"):
            s = r[sp]
            rows.append({"rule": r["name"], "role": r["role"], "split": sp, "fills": s.get("fills"), "events": s.get("events"),
                         "weeks": s.get("weeks"), "c/contract": s.get("c_per_contract"), "PnL/$": s.get("pnl_per_usd"),
                         "PnL $": s.get("pnl_usd"), "t event": s.get("t_event"), "t week": s.get("t_week"),
                         "% pos trade-weeks": s.get("pct_pos_weeks"), "hit": s.get("hit"), "mean price": s.get("mean_px"),
                         "exact p (fair)": s.get("p_fair")})
            b = r[sp + "_book"]
            books.append({"rule": r["name"], "split": sp, "Sharpe": b.get("sharpe"), "CI90": str(b.get("sharpe_ci90")),
                          "Sharpe ex top-5": b.get("sharpe_ex_top5"), "ann vol %": b.get("ann_vol_pct"),
                          "max DD $": b.get("max_dd_usd"), "$/week": b.get("usd_per_week"),
                          "$/week ex top-5": b.get("usd_per_week_ex_top5"), "% pos cal. weeks": b.get("weeks_positive_pct"),
                          "max open $": b.get("max_open_cost_usd")})
    res = pl.DataFrame(rows)
    bk = pl.DataFrame(books)
    sc = pl.read_parquet(OUT / "lma_settle_check.parquet")
    sc26 = sc.filter(pl.col("snap_date") >= pl.date(2026, 4, 1))

    text = f"""# Recurring culture / entertainment / science-tech families on Kalshi (`culture_recurring`)

Run 2026-11-20 (UTC). Code: `src/albert/research/culture_recurring.py`; tests: `tests/test_culture_recurring.py`.
Excluded by scope: Netflix and the Billboard / Luminate / Spotify charts (other agents).

## Bottom line

**Verdict: candidate_paper (thin), and needs forward data. Nothing is ready to trade live.**

The rule picked by the pre-written procedure, *buy YES on the Rotten Tomatoes ladder leg priced
85-97c six hours before the Monday 10:00 ET close*, won 29 of 29 dev fills (+6.0c/contract) and
6 of 6 holdout fills (+6.1c/contract). That clears the live bar on paper: holdout t 4.3, 100% positive
trade-weeks, Sharpe ex top-5 3.2. It does not clear it in substance:

* With zero losses, a t-stat and a Sharpe say nothing. The honest test is the exact chance that
  every fill wins when each fill's true win probability equals its price plus fee. That chance is
  p = 0.17 on dev, 0.68 on the holdout and 0.11 pooled (35/35 at a mean 93.5c). Any one loss costs
  about $2.8, which wipes out about 15 wins of about $0.18.
* The holdout has only 6 qualifying events, and the rule makes about $0.15-0.21 a week at $3 an order.
* The LMArena secondary (buy NO at T-72h on challengers priced 5-35c) has the same shape: 29/29
  wins, +6.2c, pooled p 0.15.
* The second RT secondary (T-24h, YES on 65-97c legs) was added after the dev calibration
  table. It lost on the holdout (-4.7c, 13 fills).

What would change this: about 50 more qualifying events at T-6h. At p_fair = 0.935, 50 straight wins
has chance 0.035; at a true 0.99 it happens 60% of the time. RT yields about 0.8 qualifying events a
week, so that is roughly a year. Loosening the time or band to get there faster would need a
fresh pre-registration. The forward recorder below logs displayed size, so a paper test can check fills.

## Step 1: universe

Today's `/series` (Entertainment, Science and Technology, Culture, Social), music charts and Netflix
removed, counting series with >= 3 settled events between 2026-08-17 and 2026-11-17. Sources: the
shared markets table plus the November listing from `behav_culture`. Spreads are volume-weighted
medians from the `behav_culture` panels (mid 5-95c). Rank score = volume x feasibility x
(1 - spread).

{md(uni)}

Not on Kalshi as recurring families on 2026-11-20:
* **Box office.** No opening-weekend or gross series exist, by title and ticker search of today's
  `/series`. For a future family, the-numbers.com allows our paths; deadline.com blocks the Claude
  agents and boxofficemojo.com blocks everyone.
* **Google search / trends and podcasts.** One-off yearly lists only.
* **Bestseller lists.** None.
* **Twitch / Kick and follower counts.** One-offs.
* **App Store free ranking.** `KXAPPRANKFREE`, 302 events, ended 2025-11.
* **GPU rental prices and Television Stats.** Dormant since 2026-09.

## Sources, robots verdicts (checked 2026-11-20 22:27 UTC) and as-of status

{md(rb)}

* **Rotten Tomatoes.** robots.txt answers 403 to our UA, so it counts as blocked, and so do
  Wayback copies. Every KXRT test here uses Kalshi prices only. **Flag for the coordinator:**
  `culture.py` study B (2026-11-05) used Wayback copies of RT pages. Under the current policy that
  data cannot be used, and its RT verdict cannot be revived with more weeks of Wayback data.
* **LMArena.** lmarena.ai serves a Cloudflare block page (403) to our UA, so it is not used. The
  public HF dataset `lmarena-ai/leaderboard-dataset` is allowed. It holds every leaderboard
  snapshot (`leaderboard_publish_date`) since 2023. Each snapshot gets its **as-of time from the
  first HF commit "Update text for D"**: about 04:05 UTC on D+1, sometimes days later. Commit history
  starts 2026-04-02 (backfill before that), so only events from April 2026 are modelled.
  Settlement check: the dataset's leader on the settlement date matches Kalshi's YES leg in
  {sc26['match'].sum()}/{sc26.height} events since April 2026. The misses are KXLLM1-26SEP30 (the lead
  changed that day, and that snapshot was published after the close) and KXTECHRANKLISTAICODE-26JUL27.
  `webdev` = "LM Code Arena".
* **OpenRouter.** robots allows `/api/frontend/`. `author-market-share?metric=requests&modality=text`
  reproduces the KX*SHARE settlements; for example OpenAI in the week of 2026-11-09 is 18.99%, so
  Kalshi's 19.0 (see the test fixture). The default tokens view reproduces KXTOKENUSE. Today's
  history matches every leg's result in about 95% of settled events; the misses are 0.1-point
  rounding at a strike. As-of: last week's final total is usable from Monday 06:00 UTC. The running
  week is live on the site but has no archive (Wayback has 2 captures, with other parameters), so
  intra-week tests need the forward recorder.
* **kworb.net.** Allowed. Its daily list of most-viewed music videos is a proxy for the YouTube
  Charts families. It has no history, so it is forward-only. The YouTube Charts data API
  (`/youtubei/`) is disallowed.
* **Metacritic.** Allowed, Wayback included. With 21 KXMC events ever, it was tested price-only
  alongside RT.

## Hypotheses

Twelve hypotheses were written before any dev outcome was scored; see `hypotheses.md`, which also
holds the pre-written primary-selection procedure. Outcomes:

| # | Hypothesis | Outcome |
|---|---|---|
| H1 | RT score decay (centre / above-median legs overpriced) | rejected: centre-leg NO is -0.3 to -3.3c at T-14d/T-72h; the RT calibration runs the other way (favourites cheap) |
| H2 | longshot YES premium on ladder tails | small and not significant: tail NO -3.8c (T-14d), -1.6c (T-72h), -0.1c (T-24h), +4.1c (T-6h, 19 fills, p 0.45) |
| H3 | ladder monotonicity arbs | none after fees in 353 RT, 68 MC and 169 OpenRouter ladder snapshots |
| H4 | pre-review over-confidence | rejected (T-14d centre NO -3.3c) |
| H5 | favourites cheap into the Monday freeze | **primary**: T-6h +6.0c dev / +6.1c holdout, all wins, exact p 0.17 / 0.68; T-72h +2.1c, T-24h +3.4c on dev |
| H6 | late-added strikes | no effect (+/- noise, 8-30 fills) |
| H7 | LMArena as-of margin model | no information: the leader-leg model cells are within +/-2.5c of zero (t < 0.8); the price already holds the 95% repeat rate |
| H8 | challenger hype | **secondary**: T-72h NO on 5-35c challengers, +6.9c dev / +4.0c holdout, all wins, p 0.20 / 0.75 |
| H9 | OpenRouter last-week model | inconclusive: dev covers 7 weeks; open+1h +20c on 12 fills in 3 weeks (p 0.14), open+24h +3.8c (t 0.05); not frozen |
| H10 | OpenRouter running-share latency | forward only (recorder built) |
| H11 | kworb YouTube views | forward only (recorder built) |
| H12 | round-strike premium | noise: T-72h round +23c on 14 fills (p 0.08), non-round -6.4c; no effect at T-14d/T-24h |

## Dev screen

Dev covers closes before 2026-10-01: RT from 2026-01-26, LMArena from April 2026 (as-of limit),
OpenRouter from 2026-08-10. The screen tried **118 trade cells**: 116 pre-listed plus 2
calibration-inspired cells, `fav_yes_65_97` at T-72h and T-24h. It also produced **65
calibration-bucket cells** (`dev_cells.csv`, `dev_calibration.csv`). No trade cell reaches exact
p < 0.05. With 118 cells, about 6 would show |t| > 2 by chance.

Notable cells (|t| >= 1.4 or exact p <= 0.2, >= 8 fills):

{md(notable)}

Execution in every cell:
* Taker at the ask, or at 1 - bid for NO, from the last hourly candle that ended at or before the
  decision time (<= 3 h old).
* Spread <= 10c.
* About $3 an order (<= 25 contracts); one order per event.
* Kalshi taker fee 0.07 x mult x n x P(1-P), rounded up per order.
* Held to settlement.

## Frozen rules (`primary.json`, frozen 2026-11-20 23:35 UTC, before any holdout statistic)

1. **Primary `rt_fav_yes_T6h`.** At close - 6 h, buy YES on the KXRT leg whose mid is in
   [0.85, 0.97) and closest to 0.91. Chosen by the pre-written procedure: the largest
   week-clustered t among cells with >= 20 fills, >= 8 weeks and c > 0.
2. **Secondary `lma_challenger_no_T72h`.** At close - 72 h, buy NO on the LMArena-family
   non-leader leg (leader = the as-of HF board) whose mid is in [0.05, 0.35) and closest to 0.20.
   It was the best LMArena cell.
3. **Secondary `rt_fav_yes_65_97_T24h`.** At close - 24 h, buy YES on the KXRT leg whose mid is in
   [0.65, 0.97) and closest to 0.80. This is a **deviation** from the written procedure: it was
   added after the dev calibration table. OpenRouter had no cell meeting the >= 8-week filter, so it
   has no secondary.

Prior use of Oct-Nov data on these families:
* `behav_culture` ran pan-Entertainment longshot and fandom tests on the 2026-10-01..11-17
  holdout, including some KXRT legs. Its post-hoc lead was "sell longshots at close - 24 h".
* `culture.py` scored RT on October.
* H5 and H8 are of the same favourite-longshot family, so they are not fully independent of those
  earlier looks.

## Holdout (closes 2026-10-01 -> 2026-11-16, scored once)

{md(res)}

Sharpe / volatility from a capped $100 book. Daily MTM at the hourly-candle mid uses
`vol_metrics.ledger_metrics` with marks from this study's candle cache; Sharpe = mean / sd x
sqrt(365); CI90 is a 7-day block bootstrap. With zero losses the Sharpe is an artefact of no
variance, and the exact p above is the number to read.

{md(bk)}

Exposure: at most $5.8-10.6 open at once (`max open $`), far inside the $100 book and the $5/event cap.

## Forward recorder (built and smoke-tested; not scheduled)

`python -m albert.research.culture_recurring record`, from /home/ubuntu/albert, **hourly**. Each
run is about 30 light requests. It appends one timestamped parquet per source under
`data/research/culture_recurring/forward/<source>/<UTC stamp>.parquet` and never overwrites:

* **openrouter:** weekly author totals including the running week, for both metrics, with the
  site's `cachedAt`.
* **lmarena:** HF latest boards for text and webdev, with the last commit time.
* **kworb_youtube:** the daily most-viewed music videos.
* **kalshi_quotes:** bid, ask, sizes, volume and open interest of every open market in KXRT, KXMC,
  the LMArena family, the OpenRouter family and the YouTube views / top-video series.

Smoke run 2026-11-20 23:36 UTC: 1,060 OpenRouter rows, 587 LMArena rows, 500 kworb rows and
696 Kalshi quotes. Displayed sizes are present; for example KXRT-FUR-77 had an ask of 93c x 50.

What to do with it:
* **(a)** Paper-trade the primary (and secondary 2) on Monday mornings with real displayed size.
* **(b)** After about 6-8 weeks of hourly OpenRouter snapshots, test H10: whether the running
  request share mid-week predicts the KX*SHARE ladder better than its quotes.
* **(c)** Test whether kworb daily views lead the YouTube views / top-video legs (H11).

## Caveats

* Candles carry no depth: size is capped at 25 contracts and the spread at 10c. The recorder now logs
  depth.
* `T-6h` quotes come from overnight hourly candles. Six of 19 holdout RT events had a leg in the
  band at T-6h; dev had 30 of 74.
* The LMArena as-of time uses HF commit times, which is conservative: lmarena.ai itself may show a
  snapshot hours earlier.
* The OpenRouter history is today's view, not as-of. Revisions look small (about 95% of events reproduce
  exactly), but they are not proven.
* The holdout spans 7 weeks.

## Shared-helper note

`vol_metrics.load_marks` reads only its fixed `CANDLE_SOURCES`. This study passes its own marks
(same mid rule) to `ledger_metrics`. If the coordinator wants these ledgers in the global
`vol_metrics run`, add `data/cache/culture_recurring/candles/part_*.parquet` to `CANDLE_SOURCES`.
That needs glob support, because the `exists()` check fails on a pattern.
"""
    (OUT / "report.md").write_text(text)
    return OUT / "report.md"


# ---- forward recorder -------------------------------------------------------------------------------------
FORWARD = OUT / "forward"
RECORD_SERIES = RT_SERIES + MC_SERIES + LMA_SERIES + OR_SERIES + YT_SERIES
QUOTE_FIELDS = ("yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars", "last_price_dollars",
                "yes_bid_size_fp", "yes_ask_size_fp", "volume_fp", "volume_24h_fp", "open_interest_fp", "liquidity_dollars")


def _write_snapshot(kind: str, df: pl.DataFrame, stamp: str) -> Path:
    path = FORWARD / kind / f"{stamp}.parquet"
    if path.exists():                      # append-only: never overwrite a snapshot
        raise FileExistsError(path)
    _save(df, path)
    return path


def kworb_rows(html: str, fetched: datetime) -> list[dict[str, Any]]:
    """kworb.net/youtube/ (today's most viewed music videos): rank, title, video id, 24 h views."""
    rx = re.compile(r'<tr><td>(\d+)</td><td>([^<]*)</td><td class="text"><div><a href="([^"]+)">([^<]*)</a></div></td>\s*'
                    r'<td>([\d,]+)</td>')
    out = []
    for m in rx.finditer(html):
        href = m.group(3)
        vid = re.search(r"(?:v=|video/)([A-Za-z0-9_-]{11})", href)
        out.append({"rank": int(m.group(1)), "move": m.group(2), "title": m.group(4),
                    "video_id": vid.group(1) if vid else None, "views": int(m.group(5).replace(",", "")),
                    "fetched": fetched})
    return out


def record(*, log=print) -> dict[str, int]:
    """One forward snapshot: OpenRouter weekly author totals incl. the running week (both metrics),
    LMArena latest text/webdev boards (HF), and Kalshi quotes of every open market of the recorded
    series. Each source row carries its own fetch time (and the source's cachedAt / commit time)."""
    from albert.venues.kalshi import _field, num, parse_ts

    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    web = Web(CACHE / "web_forward")
    counts: dict[str, int] = {}
    rows = []
    for tag, params in OR_PARAMS.items():
        body, when = web.get(OR_API + (f"?{params}" if params else ""), cache=False)
        if body:
            rows += [{**r, "metric": tag} for r in openrouter_rows(json.loads(body), when, params)]
    if rows:
        _write_snapshot("openrouter", pl.DataFrame(rows), stamp)
        counts["openrouter"] = len(rows)
    lrows = []
    commits, _ = web.get(f"{HF}/api/datasets/{LMA_REPO}/commits/main?p=0", cache=False)
    last_commit = json.loads(commits)[0]["date"] if commits else None
    for cfg in LMA_CONFIGS:
        body, when = web.get(f"{HF}/datasets/{LMA_REPO}/resolve/main/{cfg}/latest-00000-of-00001.parquet", cache=False)
        if body:
            d = pl.read_parquet(io.BytesIO(body))
            d = d.filter(pl.col("category") == "overall") if "category" in d.columns else d
            lrows.append(d.select([c for c in ("model_name", "organization", "rating", "rating_lower", "rating_upper",
                                               "vote_count", "rank", "leaderboard_publish_date") if c in d.columns])
                         .with_columns(pl.lit(cfg).alias("config"), pl.lit(when).alias("fetched"),
                                       pl.lit(last_commit).alias("hf_last_commit")))
    if lrows:
        _write_snapshot("lmarena", pl.concat(lrows, how="diagonal_relaxed"), stamp)
        counts["lmarena"] = sum(x.height for x in lrows)
    body, when = web.get(KWORB_YT, cache=False)
    if body:
        kw = kworb_rows(body.decode("utf-8", "replace"), when)
        if kw:
            _write_snapshot("kworb_youtube", pl.DataFrame(kw), stamp)
            counts["kworb_youtube"] = len(kw)
    api = _client(min_interval=0.5)
    q = []
    for t in RECORD_SERIES:
        try:
            for page in api.series_markets(t, "open"):
                for m in page.payload.get("markets") or []:
                    q.append({"series_ticker": t, "ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"),
                              "yes_sub_title": m.get("yes_sub_title"), "floor_strike": num(m.get("floor_strike")),
                              "close_time": parse_ts(m.get("close_time")), "fetched": page.fetched_at,
                              **{f: num(_field(m, f)) for f in QUOTE_FIELDS}})
        except RuntimeError as error:
            log(f"  quotes {t}: {str(error)[:120]}", flush=True)
    if q:
        _write_snapshot("kalshi_quotes", pl.DataFrame(q, infer_schema_length=None), stamp)
        counts["kalshi_quotes"] = len(q)
    log(f"record {stamp}: {counts}", flush=True)
    return counts


# ---- CLI ---------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m albert.research.{NAME}")
    ap.add_argument("cmd", choices=["robots", "universe", "fetch", "alt", "all-fetch", "build", "dev", "record", "holdout", "report"])
    ap.add_argument("--series", default=",".join(DEEP_SERIES))
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "robots":
        print(check_robots())
    elif a.cmd == "universe":
        with pl.Config(tbl_rows=40, tbl_width_chars=220, fmt_str_lengths=60):
            print(universe().select("family", "events", "events_per_month", "volume", "vol_weighted_spread",
                                    "feasibility", "opportunity", "series"))
    elif a.cmd in ("alt", "all-fetch"):
        _save(lmarena_commits(), OUT / "lmarena_commits.parquet")
        for cfg in LMA_CONFIGS:
            h = lmarena_history(cfg)
            _save(h, OUT / f"lmarena_{cfg}_overall.parquet")
            print(f"lmarena {cfg}: {h['pub_date'].n_unique()} snapshots, {h.filter(pl.col('as_of').is_not_null())['pub_date'].n_unique()} with commit times")
        for tag, params in OR_PARAMS.items():
            o = openrouter_weekly(params=params)
            _save(o, OUT / f"openrouter_{tag}.parquet")
            print(f"openrouter {tag}: {o['week'].n_unique()} weeks, {o.height} rows")
        if a.cmd == "all-fetch":
            universe()
            mk = fetch_markets(a.series.split(","))
            print(f"markets: {mk.height} settled, {mk['event_ticker'].n_unique()} events")
            print(f"candles: {fetch_candles(mk)} new rows")
    elif a.cmd == "holdout":
        print(json.dumps(run_holdout(), indent=1, default=str))
    elif a.cmd == "report":
        print(write_report())
    elif a.cmd == "record":
        record()
    elif a.cmd in ("build", "dev"):
        panels = build_panels() if a.cmd == "build" else None
        if panels:
            for k, v in panels.items():
                print(k, v.height, "rows", v["event_ticker"].n_unique(), "events", v.filter(pl.col("dev"))["event_ticker"].n_unique(), "dev events")
        with pl.Config(tbl_rows=200, tbl_width_chars=250, tbl_hide_column_data_types=True):
            print(run_dev(panels=panels))
    elif a.cmd == "fetch":
        mk = fetch_markets(a.series.split(","))
        print(f"markets: {mk.height} settled, {mk['event_ticker'].n_unique()} events")
        print(f"candles: {fetch_candles(mk.filter(pl.col('series_ticker').is_in(a.series.split(','))))} new rows")


if __name__ == "__main__":
    main()
