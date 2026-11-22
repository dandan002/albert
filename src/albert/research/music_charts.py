"""Billboard / Luminate / streaming chart markets on Kalshi: is there a repeatable small-stake edge?

Question (2026-11-20): the weekly music-chart series (Hot 100 / Billboard 200 #1 and #2, Hot 100
top 10, Luminate unit ladders) are among the most liquid recurring culture markets. Prior work
(``culture.py`` triage, ``recurring.py``, ``behav_culture.py``, ``linked.py``) dropped them for
sample size or found incumbent persistence and chart duplicates priced. This study looks at three
things those did not: (1) the **Billboard 200 Sunday reveal**, which Billboard publishes hours
before the album markets close; (2) **HITS Daily Double first-week projections** (timestamped,
robots-allowed) against the Luminate unit ladders; (3) the price-only fandom / favourite structure
inside these series, plus a forward recorder for the alt-data that has no as-of history.

Timeline for a chart dated Saturday D: Luminate tracking week Fri D-15 .. Thu D-9; Billboard posts
the Billboard 200 top-10 story Sunday D-6 and the Hot 100 top-10 story Monday D-5; full charts
Tuesday D-4. Kalshi chart series (KXTOPSONG, KXTOPALBUM, KXBILLBOARDRUNNERUP*, KXRANKLISTSONGTOP10,
KXALBUMDEBUT) close Sunday D-6 23:59 ET; the Luminate ladders (KXALBUMEQUIV, KXPUREALBUMS,
KXALBUMSTREAMS) close Sunday D-6 10:00 ET. All have ``can_close_early``.

Hypotheses are in ``data/research/music_charts/hypotheses.md`` (written before outcome data).
Dev = events closing before 2026-10-01; holdout = 2026-10-01 .. latest settled.

**Result (2026-11-20): paper candidate.** HITS posts first-week projections on Tuesdays ~16:00 ET
(Activity = album units, Albums = pure sales; ``publishedAt`` stamped, tables unrevised in 319/320
rows checked against Wayback). Luminate lands ~4% above HITS on album units (dev robust sd of
log(Luminate / HITS) 0.065 at the preview, 0.049 at the Friday final). The ladders absorb the
preview over ~24 h: a taker pricing each strike from HITS at the first hour mark >= 5 min after the
post (theta 5c) made +20.8c/contract on dev (60 fills, 54 events, t 3.75, book Sharpe 3.75
[1.67, 5.46], ex top-5 2.17) and +16.7c on the holdout (15 fills, 14 events, t 1.83, Sharpe 3.05
[-2.35, 3.54], ex top-5 -3.4). The Friday HITS final, the #1 / #2 album markets, the Sunday
Billboard 200 reveal (winner already >= 95c by early Sunday; ask <= 98c in 6% of events) and
price-only cells show nothing tradable. Dev cells tried: 83. ``record`` is the forward recorder.

    python -m albert.research.music_charts robots | discover | candles | hits | panel | screen | freeze | holdout | report
    python -m albert.research.music_charts record          # forward recorder: one snapshot per call
"""

from __future__ import annotations

import argparse
import html
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

OUT = Path("data/research/music_charts")
CACHE = Path("data/cache/music_charts")
CANDLES = CACHE / "candles"
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
ET = ZoneInfo("America/New_York")
UA = "albert-research/0.2 (public-data research)"

CHART_SERIES = ["KXTOPSONG", "KXTOPALBUM", "KXBILLBOARDRUNNERUPSONG", "KXBILLBOARDRUNNERUPALBUM",
                "KXRANKLISTSONGTOP10", "KXALBUMDEBUT"]
LADDER_SERIES = ["KXALBUMEQUIV", "KXPUREALBUMS", "KXALBUMSTREAMS"]
SERIES = CHART_SERIES + LADDER_SERIES
# reported in triage only (no market settles in the holdout, or no as-of source)
DEAD_SERIES = ["KXBBCHARTPOSITIONSONG", "KXBBCHARTPOSITIONALBUM", "KXSPOTIFYD", "KXSPOTIFY2D", "KXSPOTIFYGLOBALD",
               "KXSPOTIFYARTISTD", "KXSPOTIFYW", "KXSPOTIFYALBUMW", "KXSPOTIFYARTISTW", "KXARTISTSTREAMS",
               "KXALBUMSTREAMSU", "KXARTISTSTREAMSY", "TOPSONG", "KXRANKLIST1SONG"]
ALBUM_CHART = {"KXTOPALBUM", "KXBILLBOARDRUNNERUPALBUM", "KXALBUMDEBUT"}
SONG_CHART = {"KXTOPSONG", "KXBILLBOARDRUNNERUPSONG", "KXRANKLISTSONGTOP10"}


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


# ---- robots ------------------------------------------------------------------------------------
AGENTS = ("*", "anthropic-ai", "ClaudeBot", "Claude-Web")
ROBOTS_CHECKS: dict[str, list[str]] = {
    "www.billboard.com": ["/charts/hot-100/", "/charts/billboard-200/", "/c/music/chart-beat/", "/wp-json/wp/v2/posts"],
    "www.hitsdailydouble.com": ["/news/charts/", "/charts/hits-top-50/", "/sales_plus_streaming", "/api/"],
    "kworb.net": ["/spotify/country/us_daily.html", "/spotify/track/", "/apple_songs/", "/radio/"],
    "charts.spotify.com": ["/charts/view/regional-us-daily/latest"],
    "charts-spotify-com-service.spotify.com": ["/public/v0/charts"],
    "rss.marketingtools.apple.com": ["/api/v2/us/music/most-played/100/songs.json",
                                     "/api/v2/us/music/most-played/100/albums.json"],
    "itunes.apple.com": ["/us/rss/topsongs/limit=100/json", "/us/rss/topalbums/limit=100/json"],
    "web.archive.org": ["/cdx/search/cdx", "/web/"],
    "chartmasters.org": ["/"],
    "headlineplanet.com": ["/home/category/radio/"],
    "en.wikipedia.org": ["/wiki/List_of_Billboard_Hot_100_number_ones_of_2026"],
    "wikimedia.org": ["/api/rest_v1/metrics/pageviews/"],
    "api.elections.kalshi.com": ["/trade-api/v2/markets"],
}


def robots_verdict(text: str, path: str, agents: tuple[str, ...] = AGENTS) -> dict[str, bool]:
    from albert.research.counts import robots_verdict as rv

    return rv(text, path, agents)


def check_robots(out: Path = OUT) -> list[dict[str, Any]]:
    """Each host's own robots.txt (fetched now, redirects followed) evaluated for our paths."""
    import httpx

    rows = []
    for host, paths in ROBOTS_CHECKS.items():
        try:
            r = httpx.get(f"https://{host}/robots.txt", timeout=20, follow_redirects=True, headers={"User-Agent": UA})
            status, text = r.status_code, (r.text if r.status_code == 200 else "")
        except httpx.HTTPError:
            status, text = None, ""
        if status == 200 and text.lstrip().lower().startswith(("<!doctype", "<html")):
            text = ""          # an HTML 200 page is not a robots file (charts.spotify.com): treat as missing
            status = 404
        for p in paths:
            v = robots_verdict(text, p) if status in (200, 404, 410) else dict.fromkeys(AGENTS, False)
            rows.append({"host": host, "path": p, "http_status": status,
                         "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                         **{f"allow_{a}": v[a] for a in AGENTS}, "allowed": all(v.values())})
    out.mkdir(parents=True, exist_ok=True)
    (out / "robots.json").write_text(json.dumps(rows, indent=1))
    return rows


# ---- discovery ---------------------------------------------------------------------------------
def discover(out: Path = OUT, series: list[str] = SERIES, *, log=print) -> pl.DataFrame:
    """Every settled market of the study series (historical + live endpoints), with rules text."""
    from albert.research.counts import MARKET_COLS, _api, _pages, market_row

    api = _api(min_interval=0.5)
    rows: list[dict[str, Any]] = []
    for t in series:
        n0 = len(rows)
        for page in _pages(api, "/historical/markets", {"series_ticker": t, "limit": 1000}):
            rows.extend(market_row(m, t, "hist") for m in page.payload.get("markets") or [])
        for page in _pages(api, "/markets", {"series_ticker": t, "status": "settled", "limit": 1000}):
            rows.extend(market_row(m, t, "live") for m in page.payload.get("markets") or [])
        log(f"  discover {t}: {len(rows) - n0} markets", flush=True)
    mk = (pl.DataFrame(rows, schema=MARKET_COLS).unique("ticker", keep="last")
          .filter(pl.col("result").is_in(["yes", "no"])))
    _save(mk, out / "markets.parquet")
    log(f"discover: {mk.height} settled markets, {mk['event_ticker'].n_unique()} events")
    return mk


# ---- candles (hourly, whole market life) ---------------------------------------------------------
MAX_PERIODS = 4900


def _candle_parts() -> list[Path]:
    return sorted(CANDLES.glob("part-*.parquet"))


def fetch_candles(out: Path = OUT, *, workers: int = 2, min_interval: float = 0.6, max_retries: int = 6,
                  only: list[str] | None = None, log=print) -> None:
    """Hourly OHLC candles (bid/ask/price) over each market's life. Reuses the xcat / recurring
    hourly layers (same window and schema) and fetches the rest: post-cutoff markets in batches,
    historical ones one request each, ladders and album charts first."""
    from concurrent.futures import ThreadPoolExecutor

    from albert.config import Settings
    from albert.http import CachedClient
    from albert.research import xcat
    from albert.research.recurring import pack_live
    from albert.venues import kalshi as k

    CANDLES.mkdir(parents=True, exist_ok=True)
    mk = pl.read_parquet(out / "markets.parquet").filter(pl.col("open_time").is_not_null() & pl.col("close_time").is_not_null())
    if only:
        mk = mk.filter(pl.col("ticker").str.extract(r"^([A-Z0-9]+)-", 1).is_in(only))
    done_path = CANDLES / "done.parquet"
    done = set(pl.read_parquet(done_path)["ticker"].to_list()) if done_path.exists() else set()
    want = set(mk["ticker"].to_list())
    part = len(_candle_parts())
    for name, d in (("xcat", Path("data/research/xcat/hourly")), ("recurring", Path("data/research/recurring/candles"))):
        dp = d / "done.parquet"
        if not dp.exists():
            continue
        reuse = sorted((set(pl.read_parquet(dp)["ticker"].to_list()) & want) - done)
        if not reuse:
            continue
        c = (pl.scan_parquet(str(d / "part-*.parquet")).filter(pl.col("ticker").is_in(reuse))
             .select(list(xcat.OHLC_SCHEMA)).collect())
        c.write_parquet(CANDLES / f"part-{part:05d}.parquet")
        part += 1
        done |= set(reuse)
        _save(pl.DataFrame({"ticker": sorted(done)}, schema={"ticker": pl.String}), done_path)
        log(f"reused {len(reuse)} markets from {name} ({c.height} candles)", flush=True)
    s = Settings()
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=min_interval, max_retries=max_retries))
    cutoff = api.cutoff()["market_settled_ts"]
    # ladders, then album charts, then song charts; within each, traded markets first (zero-volume
    # markets last: their quotes are fetched too, so the sample stays volume-blind once complete)
    series = pl.col("ticker").str.extract(r"^([A-Z0-9]+)-", 1)
    prio = (pl.when(series.is_in(LADDER_SERIES)).then(0).when(series.is_in(list(ALBUM_CHART))).then(1).otherwise(2)
            + pl.when(pl.col("volume").fill_null(0) > 0).then(0).otherwise(3))
    todo = (mk.filter(~pl.col("ticker").is_in(list(done)))
            .with_columns(pl.col("open_time").alias("lo"),
                          pl.min_horizontal("close_time", pl.col("open_time") + timedelta(hours=MAX_PERIODS - 1)).alias("hi"),
                          pl.lit(60).alias("period"), prio.alias("prio"))
            .sort("prio", "close_time", descending=[False, True]))
    hist = todo.filter(pl.col("settlement_ts") < cutoff).to_dicts()
    live = todo.filter(pl.col("settlement_ts") >= cutoff).to_dicts()
    jobs = [{"kind": "batch", "rows": b} for b in pack_live(live)] + [{"kind": "hist", "rows": [r]} for r in hist]
    log(f"candles: {len(want)} markets, {len(done)} done, {len(hist)} historical + {len(live)} live -> {len(jobs)} requests",
        flush=True)
    buf: list[dict[str, Any]] = []
    fin: list[str] = []
    t0 = time.time()

    def flush() -> None:
        nonlocal part, buf, fin, done
        if buf:
            pl.DataFrame(buf, schema=xcat.OHLC_SCHEMA).write_parquet(CANDLES / f"part-{part:05d}.parquet")
            part += 1
        done |= set(fin)
        _save(pl.DataFrame({"ticker": sorted(done)}, schema={"ticker": pl.String}), done_path)
        buf, fin = [], []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, res in enumerate(pool.map(lambda j: xcat._run_job(api, j), jobs), 1):
            for tk, got in res:
                if got is not None:
                    buf.extend(got)
                    fin.append(tk)
            if n % 150 == 0 or n == len(jobs):
                flush()
                rate = n / (time.time() - t0)
                log(f"  {n}/{len(jobs)} requests, {len(done)} markets, {rate:.2f} req/s, "
                    f"eta {(len(jobs) - n) / max(rate, 1e-9) / 60:.0f} min", flush=True)
    flush()


def load_candles(tickers: list[str] | None = None) -> pl.DataFrame:
    cols = ["ticker", "end_ts", "bid_close", "ask_close", "bid_low", "ask_low", "ask_high", "bid_high",
            "p_close", "p_low", "p_high", "volume"]
    lf = pl.scan_parquet([str(p) for p in _candle_parts()]).select(cols)
    if tickers is not None:
        lf = lf.filter(pl.col("ticker").is_in(tickers))
    return lf.collect().unique(["ticker", "end_ts"]).sort("ticker", "end_ts")


# ---- HITS Daily Double chart posts ---------------------------------------------------------------
# robots.txt (www.hitsdailydouble.com, checked 2026-11-20) disallows only /api/ and /studio/. Posts in
# /news/charts/ embed HITS' chart table (projected or final Activity = album units, Albums = pure
# sales, TEA, SEA) and carry the Sanity ``publishedAt`` of the post. The URL list comes from the
# Wayback CDX index (web.archive.org robots: no file -> allowed); the posts are fetched from HITS.
HITS = "https://www.hitsdailydouble.com"
HITS_CACHE = CACHE / "hits"
_ITEM_RX = re.compile(r'"this_week":(null|\d+),"rank":(?:null|"?\d+"?),"artist":"((?:[^"\\]|\\.)*)","album":"((?:[^"\\]|\\.)*)",'
                      r'"label":"((?:[^"\\]|\\.)*)","data":\{([^}]*)\}')
_MAIN_PUB_RX = re.compile(r'"publishedAt":"([^"]+)","redirectUrl":[^,]*,"sectionName":[^,]*,"sections":\[')


def _num(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _unesc(s: str) -> str:
    """JSON string escapes left in the RSC payload (\\u0026 -> &) and HTML entities."""
    s = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)
    return html.unescape(s.replace("\\\\", "\\"))


def hits_title(album: str | None, artist: str | None) -> str:
    """The title to match a HITS row on: soundtracks are listed as album SOUNDTRACK with the film as artist."""
    a = (album or "").strip().upper()
    return artist or "" if a in ("SOUNDTRACK", "ORIGINAL SOUNDTRACK", "CAST RECORDING") else album or ""


def parse_hits_post(raw: str, slug: str) -> dict[str, Any]:
    """publishedAt of the page's own post, title, kind and the embedded chart rows (first table)."""
    t = raw.replace('\\"', '"')
    pub = _MAIN_PUB_RX.search(t)
    title = re.search(r"<title>([^<]*)</title>", raw)
    desc = re.search(r'<meta name="description" content="([^"]*)"', raw)
    items = []
    # several tables can be embedded (e.g. Top 50 + a second chart); the first chartItems block is the album chart
    start = t.find('"chartItems":[')
    block = t[start:start + 200000] if start >= 0 else ""
    end = block.find("]},")
    for seen, m in enumerate(_ITEM_RX.finditer(block[: end if end > 0 else None]), 1):
        d = dict(re.findall(r'"([^"]+)":"([^"]*)"', m.group(5)))
        items.append({"pos": seen, "this_week": None if m.group(1) == "null" else int(m.group(1)),
                      "artist": _unesc(m.group(2)), "album": _unesc(m.group(3)),
                      "label": _unesc(m.group(4)), "activity": _num(d.get("Activity")),
                      "albums": _num(d.get("Albums")), "tea": _num(d.get("TEA")), "sea": _num(d.get("SEA")),
                      "pct": d.get("+/-")})
    final = any(i["sea"] is not None for i in items)
    s = slug.lower()
    kind = ("final" if final else "preview" if items else
            "other_final" if re.search(r"chart-final|top-50|number-one|chart-story", s) else
            "other_preview" if re.search(r"preview|projection|new-release", s) else "other")
    return {"slug": slug, "published_at": pub.group(1) if pub else None,
            "title": html.unescape(title.group(1)) if title else None,
            "description": html.unescape(desc.group(1)) if desc else None, "kind": kind, "items": items}


def hits_index(out: Path = OUT) -> pl.DataFrame:
    """Every dated /news/charts/ post slug the Wayback CDX knows (2025-), with its first capture."""
    import httpx

    rows: dict[str, dict[str, Any]] = {}
    for host in ("www.hitsdailydouble.com", "hitsdailydouble.com"):
        r = httpx.get("https://web.archive.org/cdx/search/cdx",
                      params={"url": f"{host}/news/charts/", "matchType": "prefix", "from": "2025", "output": "json",
                              "fl": "timestamp,original,statuscode", "limit": 50000},
                      headers={"User-Agent": UA}, timeout=120)
        r.raise_for_status()
        for ts, url, status in (r.json() or [[]])[1:]:
            m = re.search(r"/news/charts/([a-z0-9-]+-(20\d\d-\d\d-\d\d))(?:[?#].*)?$", url)
            if not m or status not in ("200", "301", "302", "308"):
                continue
            slug = m.group(1)
            cap = datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
            cur = rows.get(slug)
            if cur is None or cap < cur["first_capture"]:
                rows[slug] = {"slug": slug, "url_date": date.fromisoformat(m.group(2)), "first_capture": cap}
        time.sleep(2)
    df = pl.DataFrame(list(rows.values())).sort("url_date", "slug")
    _save(df, out / "hits_index.parquet")
    return df


def hits_fetch(out: Path = OUT, *, since: date = date(2025, 1, 1), min_interval: float = 2.0, offline: bool = False,
               log=print) -> pl.DataFrame:
    """Fetch each indexed post from HITS (gzip cache, never overwritten) and parse it."""
    import gzip

    import httpx

    HITS_CACHE.mkdir(parents=True, exist_ok=True)
    idx = pl.read_parquet(out / "hits_index.parquet").filter(pl.col("url_date") >= since)
    cli = httpx.Client(timeout=60, headers={"User-Agent": UA}, follow_redirects=True)
    posts, items = [], []
    last = 0.0
    for i, r in enumerate(idx.iter_rows(named=True)):
        p = HITS_CACHE / f"{r['slug']}.html.gz"
        if not p.exists() and offline:
            continue
        if not p.exists():
            wait = min_interval - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.monotonic()
            try:
                resp = cli.get(f"{HITS}/news/charts/{r['slug']}")
            except httpx.HTTPError as error:
                log(f"  {r['slug']}: {error}")
                continue
            if resp.status_code != 200:
                log(f"  {r['slug']}: HTTP {resp.status_code}")
                continue
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(gzip.compress(resp.content))
            tmp.replace(p)
        raw = gzip.decompress(p.read_bytes()).decode("utf-8", "ignore")
        d = parse_hits_post(raw, r["slug"])
        posts.append({k: v for k, v in d.items() if k != "items"} | {"url_date": r["url_date"], "first_capture": r["first_capture"],
                                                                      "n_items": len(d["items"])})
        items.extend({"slug": r["slug"], **it} for it in d["items"])
        if i % 25 == 0:
            log(f"  hits {i + 1}/{idx.height}", flush=True)
    pdf = pl.DataFrame(posts, infer_schema_length=None).with_columns(
        pl.col("published_at").str.to_datetime(time_zone="UTC", strict=False).dt.cast_time_unit("us"))
    idf = pl.DataFrame(items, infer_schema_length=None)
    _save(pdf, out / "hits_posts.parquet")
    _save(idf, out / "hits_items.parquet")
    log(f"hits: {pdf.height} posts, {idf.height} chart rows; kinds {dict(pdf['kind'].value_counts().iter_rows())}")
    return pdf


# ---- parsing Kalshi markets ----------------------------------------------------------------------
_MONTHS = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
_LADDER_RX = re.compile(r"^If (.+) by (.+?) has above [\d.,]+\s*[KMB]? ?(Album Equivalent Units|Pure Album Sales|streams)",
                        re.IGNORECASE)
_CHART_RX = re.compile(r"^If (.+?) is (?:the )?(#1|#2|Top 10|ranked #\d+)", re.IGNORECASE)


def event_date(event_ticker: str) -> date | None:
    """The yyMMMdd date in an event ticker: the chart date (Billboard series) or the tracking-week
    end (Luminate ladders, e.g. KXALBUMEQUIV-COO26OCT22 -> 2026-10-22)."""
    m = re.search(r"(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})(?:-|$)", event_ticker.split("-", 1)[-1] + "-")
    if not m:
        return None
    return date(2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))


def ladder_terms(rules: str | None) -> tuple[str, str, str] | None:
    """(album, artist, measure) from a ladder rules_primary."""
    m = _LADDER_RX.search((rules or "").strip())
    if not m:
        return None
    meas = m.group(3).lower()
    return m.group(1).strip(), m.group(2).strip(), ("equiv" if "equivalent" in meas else "pure" if "pure" in meas else "streams")


def chart_subject(rules: str | None, sub: str | None) -> str | None:
    m = _CHART_RX.search((rules or "").strip())
    s = m.group(1) if m else sub
    if s and " by " in s and m is None:
        s = s.rsplit(" by ", 1)[0]
    return s.strip() if s else None


def norm_title(s: str | None) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().upper()
    s = s.replace("’", "'").replace("&", " AND ")
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)
    s = re.sub(r"[^A-Z0-9 ]+", "", s)
    s = re.sub(r"^THE ", "", re.sub(r"\s+", " ", s).strip())
    return s


def title_score(a: str, b: str) -> float:
    a, b = norm_title(a), norm_title(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(min(a, b, key=len)) >= 4 and (a.startswith(b) or b.startswith(a)):
        return 0.9
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / len(ta | tb)


def week_end_for_post(published: datetime, kind: str) -> date:
    """Tracking week (ending Thursday) a HITS post describes: a final describes the week that ended
    before its publication; a preview the week in progress (Fri-Sun posts: the week just started)."""
    d = published.astimezone(ET).date()
    if kind == "final":
        back = (d.weekday() - 3) % 7 or 7          # strictly before: Thursday -> previous Thursday
        return d - timedelta(days=back)
    return d + timedelta(days=(3 - d.weekday()) % 7)


def load_markets(out: Path = OUT) -> pl.DataFrame:
    mk = pl.read_parquet(out / "markets.parquet").with_columns(
        pl.col("ticker").str.extract(r"^([A-Z0-9]+)-", 1).alias("series"))
    mk = mk.with_columns(pl.col("event_ticker").map_elements(event_date, return_dtype=pl.Date).alias("edate"),
                         (pl.col("result") == "yes").cast(pl.Int8).alias("y"),
                         pl.when(pl.col("close_time") < HOLDOUT_FROM).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("split"))
    lad = mk["series"].is_in(LADDER_SERIES)
    terms = [ladder_terms(r) if is_l else None for r, is_l in zip(mk["rules_primary"].to_list(), lad.to_list())]
    mk = mk.with_columns(pl.Series("album", [t[0] if t else None for t in terms], dtype=pl.String),
                         pl.Series("artist", [t[1] if t else None for t in terms], dtype=pl.String))
    subj = [chart_subject(r, s) if not is_l else None
            for r, s, is_l in zip(mk["rules_primary"].to_list(), mk["yes_sub_title"].to_list(), lad.to_list())]
    mk = mk.with_columns(pl.coalesce(pl.col("album"), pl.Series("subj", subj, dtype=pl.String)).alias("album"))
    # tracking-week end: ladders carry it in the ticker; chart series carry the chart date (= week end + 9 days)
    mk = mk.with_columns(pl.when(pl.col("series").is_in(LADDER_SERIES)).then(pl.col("edate"))
                         .otherwise(pl.col("edate") - timedelta(days=9)).alias("week_end"))
    return mk


def luminate_value(v: str | None) -> float | None:
    if v is None:
        return None
    s = re.sub(r"[^0-9.]", "", v.replace(",", ""))
    try:
        return float(s) if s else None
    except ValueError:
        return None


NON_ALBUM_TABLES = r"^(song-revenue|country-market-share|canciones|uk-)"


def hits_table(out: Path = OUT) -> pl.DataFrame:
    """One row per (HITS post, album) with the tracking week it describes."""
    posts = pl.read_parquet(out / "hits_posts.parquet").filter(
        pl.col("published_at").is_not_null() & (pl.col("n_items") > 0)
        & ~pl.col("slug").str.contains(NON_ALBUM_TABLES))
    items = pl.read_parquet(out / "hits_items.parquet")
    t = items.join(posts.select("slug", "published_at", "kind", "first_capture"), on="slug")
    wk = [week_end_for_post(p, k) for p, k in zip(t["published_at"].to_list(), t["kind"].to_list())]
    return t.with_columns(pl.Series("week_end", wk, dtype=pl.Date))


def match_hits(mk: pl.DataFrame, hits: pl.DataFrame, min_score: float = 0.6) -> pl.DataFrame:
    """For each market subject (album) and week: the HITS album key it matches (best title score;
    the artist must overlap when the title match is not exact)."""
    rows = []
    hw = {w: g for (w,), g in hits.group_by("week_end")}
    keys = mk.select("week_end", "album", "artist").unique()
    for r in keys.iter_rows(named=True):
        g = hw.get(r["week_end"])
        if g is None or not r["album"]:
            continue
        best, bkey = 0.0, None
        for a, art in g.select("album", "artist").unique().iter_rows():
            sc = title_score(r["album"], hits_title(a, art))
            if sc < 1.0 and r["artist"]:
                ov = set(norm_title(r["artist"]).split()) & set(norm_title(art).split())
                sc = sc if ov else sc * 0.5
            if sc > best:
                best, bkey = sc, (a, art)
        if best >= min_score:
            rows.append({"week_end": r["week_end"], "album": r["album"], "artist": r["artist"],
                         "h_album": bkey[0], "h_artist": bkey[1], "match_score": best})
    return pl.DataFrame(rows, schema={"week_end": pl.Date, "album": pl.String, "artist": pl.String, "h_album": pl.String,
                                      "h_artist": pl.String, "match_score": pl.Float64})


INFO_LAG = timedelta(minutes=5)     # a HITS post enters a decision only >= 5 min after its publishedAt


def latest_hits(hits: pl.DataFrame, week_end: date, t: datetime, *, lag: timedelta = INFO_LAG) -> pl.DataFrame:
    """The newest HITS table for a week published at least ``lag`` before ``t`` (all its albums)."""
    g = hits.filter((pl.col("week_end") == week_end) & (pl.col("published_at") <= t - lag))
    if g.is_empty():
        return g
    last = g["published_at"].max()
    return g.filter(pl.col("published_at") == last)


# ---- quotes and execution --------------------------------------------------------------------------
def order_fee(n: int, p: float, mult: float = 1.0) -> float:
    """Kalshi taker fee for one order of n contracts at price p, rounded up to the cent."""
    return math.ceil(round(0.07 * mult * n * p * (1 - p) * 100, 6)) / 100


def order_size(price: float, budget: float = 4.0, cap: int = 25) -> int:
    return max(1, min(cap, math.floor(budget / max(price, 0.01) + 1e-9)))


def quotes_at(candles: pl.DataFrame, req: pl.DataFrame, max_age: timedelta = timedelta(hours=12)) -> pl.DataFrame:
    """Attach the book at each request time ``t``: the last hourly candle ending at or before t. The API
    omits periods in which nothing changed, so the book is carried forward (up to ``max_age``)."""
    c = (candles.select("ticker", pl.col("end_ts").alias("q_ts"), pl.col("bid_close").alias("bid"),
                        pl.col("ask_close").alias("ask"), pl.col("p_close").alias("last"))
         .sort("ticker", "q_ts"))
    out = req.sort("ticker", "t").join_asof(c, left_on="t", right_on="q_ts", by="ticker", strategy="backward",
                                             check_sortedness=False)
    return out.filter(pl.col("q_ts").is_not_null() & ((pl.col("t") - pl.col("q_ts")) <= max_age))


def taker_fills(rows: pl.DataFrame, p_col: str, theta: float, *, budget: float = 4.0) -> pl.DataFrame:
    """Buy YES at the ask when p - ask - fee >= theta, NO at 1 - bid when (1-p) - (1-bid) - fee >= theta.
    One order per row (the caller limits rows per event); PnL per contract after the per-order fee."""
    out = []
    for r in rows.iter_rows(named=True):
        p, bid, ask = r[p_col], r["bid"], r["ask"]
        if p is None or bid is None or ask is None:
            continue
        cands = []
        if 0 < ask < 1:
            n = order_size(ask, budget)
            fee = order_fee(n, ask)
            cands.append(("yes", ask, n, fee, p - ask - fee / n))
        if 0 < bid < 1:
            q = 1 - bid
            n = order_size(q, budget)
            fee = order_fee(n, q)
            cands.append(("no", q, n, fee, (1 - p) - q - fee / n))
        best = max(cands, key=lambda x: x[4], default=None)
        if best is None or best[4] < theta:
            continue
        side, price, n, fee, edge = best
        win = r["y"] if side == "yes" else 1 - r["y"]
        out.append({**r, "side": side, "price": price, "n": n, "fee": fee, "edge": edge,
                    "pnl_c": win - price - fee / n, "cost_usd": n * price + fee, "payoff_usd": float(n * win)})
    return pl.DataFrame(out, infer_schema_length=None) if out else pl.DataFrame()


def summarize(f: pl.DataFrame, *, cluster: str = "event_ticker", date_col: str | None = "t") -> dict[str, Any]:
    from albert.research.counts import cluster_se

    if f.is_empty():
        return {"fills": 0}
    x = f["pnl_c"].to_numpy()
    groups = [f[cluster].to_numpy()]
    if date_col:
        groups.append(f[date_col].dt.date().cast(pl.String).to_numpy())
    se = cluster_se(x, *groups)
    wk = f.with_columns(pl.col("t").dt.truncate("1w").alias("wk")).group_by("wk").agg(
        ((pl.col("payoff_usd") - pl.col("cost_usd")).sum()).alias("usd"))
    return {"fills": f.height, "events": f[cluster].n_unique(), "c_per_contract": round(100 * float(x.mean()), 2),
            "t": round(float(x.mean()) / se, 2) if se and se > 0 else None,
            "usd_pnl": round(float((f["payoff_usd"] - f["cost_usd"]).sum()), 2),
            "pnl_per_usd": round(float((f["payoff_usd"] - f["cost_usd"]).sum() / f["cost_usd"].sum()), 3),
            "weeks": wk.height, "weeks_pos_pct": round(100 * float((wk["usd"] > 0).mean()), 0),
            "win_rate": round(float((f["payoff_usd"] > 0).mean()), 3)}


# ---- study 2: HITS vs Luminate ------------------------------------------------------------------------
HITS_COL = {"equiv": "activity", "pure": "albums", "streams": "sea"}
MEASURE = {"KXALBUMEQUIV": "equiv", "KXPUREALBUMS": "pure", "KXALBUMSTREAMS": "streams"}


def hits_value(row: dict[str, Any], measure: str) -> float | None:
    """The HITS number comparable to a Luminate measure. Previews carry Activity and Albums only, so
    the streaming proxy there is Activity - Albums (SEA + TEA, in album units)."""
    if measure == "streams":
        if row.get("sea") is not None:
            return row["sea"]
        a, b = row.get("activity"), row.get("albums")
        return a - b if a is not None and b is not None and a > b else None
    return row.get(HITS_COL[measure])


def ladder_events(mk: pl.DataFrame) -> pl.DataFrame:
    """One row per ladder event: album, measure, Luminate value (from expiration_value), close, split."""
    lad = mk.filter(pl.col("series").is_in(LADDER_SERIES))
    ev = (lad.group_by("event_ticker").agg(pl.col("series").first(), pl.col("album").first(), pl.col("artist").first(),
                                            pl.col("week_end").first(), pl.col("close_time").max(),
                                            pl.col("settlement_ts").max(), pl.col("expiration_value").drop_nulls().first(),
                                            pl.col("split").first(), pl.len().alias("strikes")))
    ev = ev.with_columns(pl.col("series").replace_strict(MEASURE).alias("measure"),
                         pl.col("expiration_value").map_elements(luminate_value, return_dtype=pl.Float64).alias("lum_raw"))
    # expiration_value is sometimes written in thousands ("150.3" = 150.3K): pick the scale that agrees
    # with the event's own strike results (YES strikes below the value, NO strikes above)
    bounds = (lad.group_by("event_ticker").agg(
        pl.col("floor_strike").filter(pl.col("result") == "yes").max().alias("lo"),
        pl.col("floor_strike").filter(pl.col("result") == "no").min().alias("hi")))
    ev = ev.join(bounds, on="event_ticker", how="left")
    lum = []
    for v, lo, hi in ev.select("lum_raw", "lo", "hi").iter_rows():
        best = None
        if v is not None:
            for s in (1.0, 1e3, 1e6):
                x = v * s
                if (lo is None or x >= lo) and (hi is None or x <= hi):
                    best = x
                    break
        lum.append(best)
    return ev.with_columns(pl.Series("lum", lum, dtype=pl.Float64))


def calibration_pairs(mk: pl.DataFrame, hits: pl.DataFrame) -> pl.DataFrame:
    """(ladder event, HITS post) pairs: every HITS table for the album's week published before the
    ladder close, with log(Luminate / HITS)."""
    ev = ladder_events(mk).filter(pl.col("lum").is_not_null() & (pl.col("lum") > 0))
    mt = match_hits(ev.select("week_end", "album", "artist"), hits)
    j = ev.join(mt, on=["week_end", "album", "artist"]).join(
        hits.rename({"album": "h_album", "artist": "h_artist"}), on=["week_end", "h_album", "h_artist"])
    j = j.filter(pl.col("published_at") < pl.col("close_time"))
    hv = [hits_value(r, r["measure"]) for r in j.select("measure", "activity", "albums", "sea").iter_rows(named=True)]
    j = j.with_columns(pl.Series("hval", hv, dtype=pl.Float64)).filter(pl.col("hval").is_not_null() & (pl.col("hval") > 0))
    return j.with_columns((pl.col("lum") / pl.col("hval")).log().alias("lr"),
                          pl.when(pl.col("kind") == "final").then(pl.lit("final")).otherwise(pl.lit("preview")).alias("stage"))


def fit_ratio(pairs: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Per measure x stage: mean and robust sd of log(Luminate / HITS) (dev pairs only)."""
    out: dict[str, dict[str, float]] = {}
    for (meas, stage), g in pairs.group_by("measure", "stage"):
        x = g["lr"].to_numpy()
        if len(x) < 5:
            continue
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med))) * 1.4826
        out[f"{meas}|{stage}"] = {"mu": med, "sd": max(mad, float(np.std(x, ddof=1)) * 0.5, 0.02),
                                  "sd_raw": float(np.std(x, ddof=1)), "n": len(x), "events": g["event_ticker"].n_unique()}
    return out


def p_above(k: float, hval: float, mu: float, sd: float, df: float = 5.0) -> float:
    """P(Luminate > k) with log(Luminate / HITS) ~ mu + sd * t_df (fat-tailed)."""
    from scipy import stats

    z = (math.log(k) - math.log(hval) - mu) / sd
    return float(stats.t.sf(z, df))


def et_time(d: date, hour: int) -> datetime:
    """``hour``:00 US Eastern on day ``d``, in UTC (DST-safe: the day is chosen before the zone)."""
    return datetime(d.year, d.month, d.day, hour, tzinfo=ET).astimezone(UTC)


def ceil_hour(t: datetime) -> datetime:
    t0 = t.replace(minute=0, second=0, microsecond=0)
    return t0 if t0 == t else t0 + timedelta(hours=1)


def ladder_decisions(ev: dict[str, Any], hits_week: pl.DataFrame) -> list[tuple[str, datetime]]:
    """Decision times for one ladder event: 1 h after the first and the last HITS preview, 1 h after
    the HITS final, Saturday 12:00 ET and Sunday 09:00 ET (1 h before the 10:00 ET close)."""
    out = []
    pv = hits_week.filter(pl.col("kind") != "final")["published_at"]
    fn = hits_week.filter(pl.col("kind") == "final")["published_at"]
    if pv.len():
        out.append(("preview_fast", ceil_hour(pv.min() + timedelta(minutes=5))))
        out.append(("preview_first", ceil_hour(pv.min()) + timedelta(hours=1)))
        if pv.max() > pv.min():
            out.append(("preview_last", ceil_hour(pv.max()) + timedelta(hours=1)))
    if fn.len():
        out.append(("final", ceil_hour(fn.min()) + timedelta(hours=1)))
    w = ev["week_end"]
    out.append(("sat12", et_time(w + timedelta(days=2), 12)))
    out.append(("sun09", et_time(w + timedelta(days=3), 9)))
    return [(k, t) for k, t in out if t < ev["close_time"]]


def ladder_panel(mk: pl.DataFrame, hits: pl.DataFrame, fit: dict[str, dict[str, float]], candles: pl.DataFrame) -> pl.DataFrame:
    ev = ladder_events(mk)
    mt = match_hits(ev.select("week_end", "album", "artist"), hits)
    ev = ev.join(mt, on=["week_end", "album", "artist"], how="left")
    lad = mk.filter(pl.col("series").is_in(LADDER_SERIES)).select("ticker", "event_ticker", "floor_strike", "y", "settlement_ts")
    by_week = {w: g for (w,), g in hits.group_by("week_end")}
    req = []
    for e in ev.iter_rows(named=True):
        hw = by_week.get(e["week_end"])
        if hw is None or e["h_album"] is None:
            continue
        hw_a = hw.filter((pl.col("album") == e["h_album"]) & (pl.col("artist") == e["h_artist"]))
        for label, t in ladder_decisions(e, hw):
            tab = hw_a.filter(pl.col("published_at") <= t - INFO_LAG).sort("published_at")
            if tab.is_empty():
                continue
            last = tab.row(-1, named=True)
            stage = "final" if last["kind"] == "final" else "preview"
            hv = hits_value(last, e["measure"])
            prm = fit.get(f"{e['measure']}|{stage}")
            if hv is None or hv <= 0 or prm is None:
                continue
            req.append({"event_ticker": e["event_ticker"], "decision": label, "t": t, "stage": stage, "hval": hv,
                        "mu": prm["mu"], "sd": prm["sd"], "measure": e["measure"], "series": e["series"],
                        "split": e["split"], "close_time": e["close_time"], "hits_pub": last["published_at"]})
    if not req:
        return pl.DataFrame()
    r = pl.DataFrame(req).join(lad, on="event_ticker")
    r = r.with_columns(pl.struct("floor_strike", "hval", "mu", "sd").map_elements(
        lambda s: p_above(s["floor_strike"], s["hval"], s["mu"], s["sd"]) if s["floor_strike"] else None,
        return_dtype=pl.Float64).alias("p_model"))
    return quotes_at(candles, r)


# ---- study 2b: HITS -> Billboard 200 #1 / #2 -----------------------------------------------------
def rank_probs(units: np.ndarray, sd: float, n_sim: int = 20000, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """P(rank 1) and P(rank 2) for each album when log(final / HITS) has iid N(0, sd) noise."""
    rng = np.random.default_rng(seed)
    sims = np.log(units)[None, :] + sd * rng.standard_normal((n_sim, len(units)))
    order = np.argsort(-sims, axis=1)
    p1 = np.bincount(order[:, 0], minlength=len(units)) / n_sim
    p2 = np.bincount(order[:, 1], minlength=len(units)) / n_sim if len(units) > 1 else np.zeros(len(units))
    return p1, p2


def chart_decisions(close: datetime, week_end: date, hits_week: pl.DataFrame) -> list[tuple[str, datetime]]:
    w = week_end
    out = []
    pv = hits_week.filter(pl.col("kind") != "final")["published_at"]
    fn = hits_week.filter(pl.col("kind") == "final")["published_at"]
    if pv.len():
        out.append(("preview_fast", ceil_hour(pv.min() + timedelta(minutes=5))))
        out.append(("preview_first", ceil_hour(pv.min()) + timedelta(hours=1)))
        if pv.max() > pv.min():
            out.append(("preview_last", ceil_hour(pv.max()) + timedelta(hours=1)))
    if fn.len():
        out.append(("final", ceil_hour(fn.min()) + timedelta(hours=1)))
    out.append(("sat12", et_time(w + timedelta(days=2), 12)))
    out.append(("sun09", et_time(w + timedelta(days=3), 9)))
    return [(k, t) for k, t in out if t < close]


def album_chart_panel(mk: pl.DataFrame, hits: pl.DataFrame, sd: dict[str, float], candles: pl.DataFrame) -> pl.DataFrame:
    """KXTOPALBUM / KXALBUMDEBUT (#1) and KXBILLBOARDRUNNERUPALBUM (#2): model probabilities from the
    newest HITS table published >= 1 h before each decision; markets not in that table are skipped."""
    ch = mk.filter(pl.col("series").is_in(list(ALBUM_CHART)) & pl.col("week_end").is_not_null())
    by_week = {w: g for (w,), g in hits.group_by("week_end")}
    req = []
    for (ev_t,), g in ch.group_by("event_ticker"):
        w = g["week_end"][0]
        hw = by_week.get(w)
        if hw is None:
            continue
        close = g["close_time"].max()
        series = g["series"][0]
        for label, t in chart_decisions(close, w, hw):
            tab = latest_hits(hw, w, t)
            if tab.is_empty() or tab["activity"].null_count() == tab.height:
                continue
            tab = tab.filter(pl.col("activity").is_not_null() & (pl.col("activity") > 0)).unique(["album", "artist"])
            stage = "final" if tab["kind"][0] == "final" else "preview"
            p1, p2 = rank_probs(tab["activity"].to_numpy(), sd[stage])
            for m in g.iter_rows(named=True):
                best, idx = 0.0, None
                for i, (a, art) in enumerate(tab.select("album", "artist").iter_rows()):
                    sc = title_score(m["album"] or "", hits_title(a, art))
                    if sc > best:
                        best, idx = sc, i
                if idx is None or best < 0.75:
                    continue
                p = p2[idx] if series == "KXBILLBOARDRUNNERUPALBUM" else p1[idx]
                req.append({"ticker": m["ticker"], "event_ticker": ev_t, "series": series, "decision": label, "t": t,
                            "stage": stage, "p_model": float(min(max(p, 0.003), 0.997)), "hits_rank": idx + 1,
                            "hits_units": float(tab["activity"][idx]), "y": m["y"], "split": m["split"],
                            "settlement_ts": m["settlement_ts"], "close_time": close})
    if not req:
        return pl.DataFrame()
    return quotes_at(candles, pl.DataFrame(req))


# ---- study 1: the Billboard 200 Sunday reveal -----------------------------------------------------------
def reveal_scan(mk: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    """Per album-chart event: the first Sunday hour (ET) at which any market's bid reached 95c, the
    winner's ask in the hours after, and the losers' bids (a reveal moves every leg at once)."""
    ch = mk.filter(pl.col("series").is_in(list(ALBUM_CHART)))
    rows = []
    c = candles.filter(pl.col("ticker").is_in(ch["ticker"].implode()))
    cby = {t: g for (t,), g in c.group_by("ticker")}
    for (ev_t,), g in ch.group_by("event_ticker"):
        close = g["close_time"].max()
        sun0 = close.astimezone(ET).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)
        legs = []
        for m in g.iter_rows(named=True):
            cg = cby.get(m["ticker"])
            if cg is None:
                continue
            legs.append((m, cg.filter((pl.col("end_ts") > sun0) & (pl.col("end_ts") <= close + timedelta(hours=1)))))
        if not legs:
            continue
        hours = sorted({ts for _, cg in legs for ts in cg["end_ts"].to_list()})
        first95 = None
        for h in hours:
            if any(((cg.filter(pl.col("end_ts") == h)["bid_close"].fill_null(0) >= 0.95).any()) for _, cg in legs):
                first95 = h
                break
        win = [(m, cg) for m, cg in legs if m["y"] == 1]
        wm, wc = win[0] if win else (None, None)

        def ask_at(h: datetime | None) -> float | None:
            if wc is None or h is None:
                return None
            x = wc.filter(pl.col("end_ts") <= h).sort("end_ts")
            return x["ask_close"][-1] if x.height else None
        pre = None
        if first95 is not None and wc is not None:
            x = wc.filter(pl.col("end_ts") < first95).sort("end_ts")
            pre = x["bid_close"][-1] if x.height else None
        rows.append({"event_ticker": ev_t, "series": g["series"][0], "close": close, "split": g["split"][0],
                     "first95_utc": first95, "first95_et": first95.astimezone(ET).strftime("%a %H:%M") if first95 else None,
                     "winner": wm["ticker"] if wm else None, "winner_bid_before": pre,
                     "winner_ask_at": ask_at(first95), "winner_ask_plus1h": ask_at(first95 + timedelta(hours=1) if first95 else None),
                     "winner_ask_plus2h": ask_at(first95 + timedelta(hours=2) if first95 else None),
                     "sunday_volume_after": float(sum(cg.filter(pl.col("end_ts") > first95)["volume"].fill_null(0).sum()
                                                      for _, cg in legs)) if first95 else None})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- study 3: price-only structure (favourites / fandom longshots) ------------------------------------
def chart_quotes(mk: pl.DataFrame, candles: pl.DataFrame, decisions: dict[str, tuple[int, int]]) -> pl.DataFrame:
    """Book of every chart-series market at fixed ET decision times (label -> (days after week end, hour))."""
    ch = mk.filter(pl.col("series").is_in(list(ALBUM_CHART | SONG_CHART)) & pl.col("week_end").is_not_null())
    req = []
    for m in ch.select("ticker", "event_ticker", "series", "week_end", "close_time", "y", "split", "settlement_ts").iter_rows(named=True):
        for label, (days, hour) in decisions.items():
            t = et_time(m["week_end"] + timedelta(days=days), hour)
            if t < m["close_time"]:
                req.append({**m, "decision": label, "t": t})
    q = quotes_at(candles, pl.DataFrame(req))
    # event rank of each market by bid at that decision (1 = favourite)
    return q.with_columns(pl.col("bid").rank("ordinal", descending=True).over("event_ticker", "decision").alias("bid_rank"),
                          pl.len().over("event_ticker", "decision").alias("n_legs"))


def fixed_side_fills(rows: pl.DataFrame, side: str, *, budget: float = 4.0) -> pl.DataFrame:
    """Take one side at the quote on every row (no model): YES at the ask or NO at 1 - bid."""
    out = []
    for r in rows.iter_rows(named=True):
        price = r["ask"] if side == "yes" else 1 - r["bid"]
        if price is None or not 0 < price < 1:
            continue
        n = order_size(price, budget)
        fee = order_fee(n, price)
        win = r["y"] if side == "yes" else 1 - r["y"]
        out.append({**r, "side": side, "price": price, "n": n, "fee": fee, "pnl_c": win - price - fee / n,
                    "cost_usd": n * price + fee, "payoff_usd": float(n * win)})
    return pl.DataFrame(out, infer_schema_length=None) if out else pl.DataFrame()


# ---- ledgers and risk metrics ----------------------------------------------------------------------------
def to_ledger(f: pl.DataFrame, strategy: str, variant: str, split: str, *, per_event_usd: float = 5.0) -> pl.DataFrame:
    """vol_metrics ledger, one leg per fill, <= $5 cost per event (later fills of an event dropped)."""
    if f.is_empty():
        return pl.DataFrame()
    f = f.sort("t").with_columns(pl.col("cost_usd").cum_sum().over("event_ticker").alias("ev_cost"))
    f = f.filter(pl.col("ev_cost") <= per_event_usd)
    return f.select(pl.lit(strategy).alias("strategy"), pl.lit(variant).alias("variant"), pl.lit(split).alias("split"),
                    pl.concat_str("ticker", pl.lit("|"), pl.col("t").cast(pl.String)).alias("struct_id"),
                    "event_ticker", "ticker", "side", pl.col("n").cast(pl.Int64), pl.col("t").alias("t_in"), "cost_usd",
                    pl.col("settlement_ts").alias("settle_ts"), "payoff_usd")


def candle_marks(candles: pl.DataFrame, tickers: list[str]) -> pl.DataFrame:
    c = candles.filter(pl.col("ticker").is_in(tickers))
    two = pl.col("bid_close").is_not_null() & pl.col("ask_close").is_not_null() & (pl.col("ask_close") > pl.col("bid_close"))
    mid = pl.when(two).then((pl.col("bid_close") + pl.col("ask_close")) / 2).otherwise(pl.col("p_close"))
    return (c.select("ticker", pl.col("end_ts").alias("ts"), mid.alias("yes_mid")).filter(pl.col("yes_mid").is_not_null())
            .sort("ticker", "ts"))


def book_metrics(led: pl.DataFrame, candles: pl.DataFrame) -> dict[str, Any]:
    from albert.research import vol_metrics as vm

    if led.is_empty():
        return {}
    m = vm.ledger_metrics(led, candle_marks(candles, led["ticker"].unique().to_list()))
    keep = ["positions", "events", "usd_per_week", "ann_return_pct", "ann_vol_pct", "sharpe", "sharpe_ci90",
            "p_sharpe_pos", "sharpe_ex_top5", "usd_per_week_ex_top5", "max_dd_usd", "weeks_positive_pct", "worst_week_usd",
            "sortino", "sharpe_settle_basis", "first_day", "last_day"]
    return {k: m.get(k) for k in keep}


# ---- screen / freeze / holdout ------------------------------------------------------------------------
THETAS = (0.03, 0.05, 0.08, 0.12)
LADDER_DECISIONS = ("preview_fast", "preview_first", "preview_last", "final", "sat12", "sun09")
PRICE_DECISIONS = {"sat12": (2, 12), "sun12": (3, 12), "sun18": (3, 18)}
REVEAL_HOURS = (14, 16, 18, 20)
REVEAL_ASKS = (0.95, 0.97, 0.98)


def cap_event(f: pl.DataFrame, per_event_usd: float = 5.0) -> pl.DataFrame:
    """Keep fills in time then edge order while the event's cumulative cost stays <= the cap."""
    if f.is_empty():
        return f
    sort = ["t", "edge"] if "edge" in f.columns else ["t"]
    desc = [False, True] if "edge" in f.columns else [False]
    f = f.sort(sort, descending=desc).with_columns(pl.col("cost_usd").cum_sum().over("event_ticker").alias("_c"))
    return f.filter(pl.col("_c") <= per_event_usd).drop("_c")


def model_cells(panel: pl.DataFrame, family: str, split: str) -> tuple[list[dict[str, Any]], dict[str, pl.DataFrame]]:
    rows, fills = [], {}
    p = panel.filter(pl.col("split") == split)
    for dec in LADDER_DECISIONS:
        for th in THETAS:
            f = cap_event(taker_fills(p.filter(pl.col("decision") == dec), "p_model", th))
            key = f"{family}|{dec}|theta={th}"
            fills[key] = f
            rows.append({"cell": key, "family": family, "split": split, **summarize(f)})
    return rows, fills


def reveal_rule(q: pl.DataFrame, hour: int, max_ask: float, min_bid: float = 0.90) -> pl.DataFrame:
    """Sunday ``hour`` ET (close day): buy YES of each album-chart event's top-bid market when its bid
    >= ``min_bid`` and ask <= ``max_ask`` (a resolved-looking book), held to settlement."""
    x = q.filter((pl.col("decision") == f"rev{hour}") & (pl.col("bid_rank") == 1) & (pl.col("bid") >= min_bid)
                 & (pl.col("ask") <= max_ask))
    return fixed_side_fills(x, "yes")


def price_cells(q: pl.DataFrame, split: str) -> tuple[list[dict[str, Any]], dict[str, pl.DataFrame]]:
    rows, fills = [], {}
    q = q.filter(pl.col("split") == split)
    alb = pl.col("series").is_in(list(ALBUM_CHART))
    for dec in PRICE_DECISIONS:
        for grp, gx in (("album", alb), ("song", ~alb)):
            base = q.filter((pl.col("decision") == dec) & gx)
            fav = fixed_side_fills(base.filter((pl.col("bid_rank") == 1) & pl.col("ask").is_between(0.60, 0.95)), "yes")
            lng = fixed_side_fills(base.filter(pl.col("ask").is_between(0.03, 0.20) & (pl.col("ask") - pl.col("bid") <= 0.10)), "no")
            for name, f in (("fav_yes_60_95", fav), ("longshot_no_3_20", lng)):
                f = cap_event(f)
                key = f"price|{dec}|{grp}|{name}"
                fills[key] = f
                rows.append({"cell": key, "family": "price", "split": split, **summarize(f)})
    return rows, fills


def reveal_cells(q: pl.DataFrame, split: str) -> tuple[list[dict[str, Any]], dict[str, pl.DataFrame]]:
    rows, fills = [], {}
    q = q.filter(pl.col("split") == split)
    for h in REVEAL_HOURS:
        for a in REVEAL_ASKS:
            f = cap_event(reveal_rule(q, h, a))
            key = f"reveal|sun{h}|ask<={a}"
            fills[key] = f
            rows.append({"cell": key, "family": "reveal", "split": split, **summarize(f)})
    return rows, fills


def build_all(out: Path = OUT, *, log=print) -> dict[str, Any]:
    """Panels for every study (dev and holdout rows; the split column separates them)."""
    mk = load_markets(out)
    hits = hits_table(out)
    pairs = calibration_pairs(mk, hits)
    _save(pairs.drop("items", strict=False), out / "calibration_pairs.parquet")
    fit = fit_ratio(pairs.filter(pl.col("split") == "dev"))
    (out / "fit_dev.json").write_text(json.dumps(fit, indent=1))
    log(f"fit (dev): {json.dumps(fit)}")
    candles = load_candles(mk["ticker"].to_list())
    lp = ladder_panel(mk, hits, fit, candles)
    _save(lp, out / "panel_ladder.parquet")
    sd = {"final": fit.get("equiv|final", {}).get("sd", 0.08), "preview": fit.get("equiv|preview", {}).get("sd", 0.2)}
    ap = album_chart_panel(mk, hits, sd, candles)
    _save(ap, out / "panel_album_chart.parquet")
    decs = {**PRICE_DECISIONS, **{f"rev{h}": (3, h) for h in REVEAL_HOURS}}
    cq = chart_quotes(mk, candles, decs)
    _save(cq, out / "panel_chart_quotes.parquet")
    rv = reveal_scan(mk, candles)
    _save(rv, out / "reveal_scan.parquet")
    log(f"panels: ladder {lp.height}, album chart {ap.height}, chart quotes {cq.height}, reveal events {rv.height}")
    return {"fit": fit}


def screen(out: Path = OUT, split: str = "dev", *, log=print) -> pl.DataFrame:
    lp = pl.read_parquet(out / "panel_ladder.parquet")
    ap = pl.read_parquet(out / "panel_album_chart.parquet")
    cq = pl.read_parquet(out / "panel_chart_quotes.parquet")
    rows = []
    for fam, panel in (("ladder", lp), ("album_chart", ap)):
        r, _ = model_cells(panel, fam, split)
        rows += r
    r, _ = price_cells(cq, split)
    rows += r
    r, _ = reveal_cells(cq, split)
    rows += r
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(out / f"screen_{split}.csv")
    log(f"screen {split}: {df.height} cells")
    return df


def rule_fills(rule: dict[str, Any], split: str, out: Path = OUT) -> pl.DataFrame:
    """Fills of one frozen rule (a dev-screen cell key) on one split."""
    fam = rule["cell"].split("|")[0]
    if fam in ("ladder", "album_chart"):
        panel = pl.read_parquet(out / f"panel_{'ladder' if fam == 'ladder' else 'album_chart'}.parquet")
        _, fills = model_cells(panel, fam, split)
    elif fam == "price":
        _, fills = price_cells(pl.read_parquet(out / "panel_chart_quotes.parquet"), split)
    else:
        _, fills = reveal_cells(pl.read_parquet(out / "panel_chart_quotes.parquet"), split)
    return fills[rule["cell"]]


def evaluate_frozen(out: Path = OUT, *, log=print) -> dict[str, Any]:
    """Score the rules frozen in primary.json on dev and holdout: per-contract stats, ledgers and the
    vol_metrics block (capped $100 book, <= $5 per event, daily MTM at the hourly-candle mid)."""
    frozen = json.loads((out / "primary.json").read_text())
    mk = load_markets(out)
    candles = load_candles(mk["ticker"].to_list())
    res: dict[str, Any] = {"frozen_utc": frozen["frozen_utc"], "rules": []}
    (out / "ledgers").mkdir(parents=True, exist_ok=True)
    for role, rule in [("primary", frozen["primary"])] + [("secondary", r) for r in frozen.get("secondaries", [])]:
        entry: dict[str, Any] = {"role": role, "cell": rule["cell"]}
        for split in ("dev", "holdout"):
            f = rule_fills(rule, split, out)
            led = to_ledger(f, "music_charts", rule["cell"], split)
            name = re.sub(r"[^A-Za-z0-9]+", "_", rule["cell"]).strip("_")
            if not led.is_empty():
                led.write_parquet(out / "ledgers" / f"ledger_{name}_{split}.parquet")
            entry[split] = {**summarize(f), "book": book_metrics(led, candles)}
            if not f.is_empty():
                f.select([c for c in ("event_ticker", "ticker", "decision", "t", "side", "price", "n", "fee", "pnl_c",
                                      "p_model", "bid", "ask", "y") if c in f.columns]).write_csv(
                    out / "ledgers" / f"fills_{name}_{split}.csv")
        res["rules"].append(entry)
        log(json.dumps(entry, default=str)[:900])
    (out / "holdout.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ---- forward recorder ----------------------------------------------------------------------------
# One call = one timestamped snapshot of every source that has no as-of history, plus the Kalshi
# quotes of the open study markets. Files are new per snapshot (append-only); nothing is overwritten.
FORWARD = OUT / "forward"
APPLE_FEEDS = {
    "apple_songs": "https://rss.marketingtools.apple.com/api/v2/us/music/most-played/100/songs.json",
    "apple_albums": "https://rss.marketingtools.apple.com/api/v2/us/music/most-played/100/albums.json",
}
ITUNES_FEEDS = {
    "itunes_songs": "https://itunes.apple.com/us/rss/topsongs/limit=100/json",
    "itunes_albums": "https://itunes.apple.com/us/rss/topalbums/limit=100/json",
}


def apple_rows(payload: dict[str, Any], feed: str, fetched: datetime) -> list[dict[str, Any]]:
    f = payload.get("feed") or {}
    return [{"feed": feed, "fetched_utc": fetched, "feed_updated": f.get("updated"), "rank": i + 1,
             "name": r.get("name"), "artist": r.get("artistName"), "id": r.get("id"),
             "release_date": r.get("releaseDate")} for i, r in enumerate(f.get("results") or [])]


def itunes_rows(payload: dict[str, Any], feed: str, fetched: datetime) -> list[dict[str, Any]]:
    f = payload.get("feed") or {}
    upd = (f.get("updated") or {}).get("label")
    out = []
    for i, e in enumerate(f.get("entry") or []):
        out.append({"feed": feed, "fetched_utc": fetched, "feed_updated": upd, "rank": i + 1,
                    "name": (e.get("im:name") or {}).get("label"), "artist": (e.get("im:artist") or {}).get("label"),
                    "id": ((e.get("id") or {}).get("attributes") or {}).get("im:id"),
                    "release_date": ((e.get("im:releaseDate") or {}).get("label"))})
    return out


def hits_listing_slugs(raw: str) -> list[str]:
    """Dated /news/charts/ post slugs linked from a HITS page (home or /charts)."""
    return sorted(set(re.findall(r"/news/charts/([a-z0-9-]+-20\d\d-\d\d-\d\d)", raw)))


def kalshi_quote_rows(payload: dict[str, Any], fetched: datetime) -> list[dict[str, Any]]:
    from albert.venues.kalshi import _field, num, parse_ts

    out = []
    for m in payload.get("markets") or []:
        out.append({"fetched_utc": fetched, "ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"),
                    "yes_sub_title": m.get("yes_sub_title"), "close_time": parse_ts(m.get("close_time")),
                    "yes_bid": num(_field(m, "yes_bid")), "yes_ask": num(_field(m, "yes_ask")),
                    "yes_bid_size": num(_field(m, "yes_bid_size")), "yes_ask_size": num(_field(m, "yes_ask_size")),
                    "last_price": num(_field(m, "last_price")), "volume": num(_field(m, "volume")),
                    "open_interest": num(_field(m, "open_interest")), "status": m.get("status")})
    return out


def record(out: Path = FORWARD, *, log=print) -> dict[str, int]:
    """One snapshot: HITS chart posts (new ones fetched once) and the Top 50 page's latest chart date,
    Apple Music most-played + iTunes top songs/albums (US), Kalshi quotes of open study markets."""
    import gzip

    import httpx

    now = datetime.now(UTC).replace(microsecond=0)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    cli = httpx.Client(timeout=60, headers={"User-Agent": UA}, follow_redirects=True)
    counts: dict[str, int] = {}
    # Apple Music / iTunes (robots: allowed)
    rows: list[dict[str, Any]] = []
    for feed, url in {**APPLE_FEEDS, **ITUNES_FEEDS}.items():
        try:
            r = cli.get(url)
            r.raise_for_status()
            rows.extend((apple_rows if feed.startswith("apple") else itunes_rows)(r.json(), feed, now))
        except (httpx.HTTPError, ValueError) as error:
            log(f"record {feed}: {error}")
        time.sleep(1)
    if rows:
        _save(pl.DataFrame(rows, infer_schema_length=None), out / "apple" / f"{stamp}.parquet")
    counts["apple_itunes_rows"] = len(rows)
    # HITS: listing pages -> new chart posts (fetched once into the shared cache) + Top 50 chart date
    hrows: list[dict[str, Any]] = []
    slugs: set[str] = set()
    top50_date = None
    for path in ("/", "/charts", "/news"):
        try:
            r = cli.get(f"{HITS}{path}")
            if r.status_code == 200:
                slugs |= set(hits_listing_slugs(r.text))
                if path == "/charts":
                    m = re.search(r'"chartDate":"(\d{4}-\d\d-\d\d)"', r.text.replace('\\"', '"'))
                    top50_date = m.group(1) if m else None
        except httpx.HTTPError as error:
            log(f"record hits {path}: {error}")
        time.sleep(2)
    HITS_CACHE.mkdir(parents=True, exist_ok=True)
    for slug in sorted(slugs):
        p = HITS_CACHE / f"{slug}.html.gz"
        new = not p.exists()
        if new:
            try:
                r = cli.get(f"{HITS}/news/charts/{slug}")
            except httpx.HTTPError:
                continue
            if r.status_code != 200:
                continue
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(gzip.compress(r.content))
            tmp.replace(p)
            time.sleep(2)
        d = parse_hits_post(gzip.decompress(p.read_bytes()).decode("utf-8", "ignore"), slug)
        hrows.extend({"fetched_utc": now, "slug": slug, "first_seen_now": new, "published_at": d["published_at"],
                      "kind": d["kind"], "title": d["title"], **it} for it in (d["items"] or [{}]))
    hrows.append({"fetched_utc": now, "slug": "__top50_page__", "first_seen_now": False, "published_at": None,
                  "kind": "top50_latest", "title": top50_date})
    _save(pl.DataFrame(hrows, infer_schema_length=None), out / "hits" / f"{stamp}.parquet")
    counts["hits_rows"] = len(hrows)
    # Kalshi quotes (public market data)
    from albert.research.counts import _api

    api = _api(min_interval=0.5)
    krows: list[dict[str, Any]] = []
    for s in SERIES:
        cursor = None
        while True:
            page = api.client.get("/markets", {"series_ticker": s, "status": "open", "limit": 1000, "cursor": cursor},
                                  cache=False)
            krows.extend(kalshi_quote_rows(page.payload, now))
            cursor = page.payload.get("cursor")
            if not cursor or not page.payload.get("markets"):
                break
    if krows:
        _save(pl.DataFrame(krows, infer_schema_length=None), out / "kalshi" / f"{stamp}.parquet")
    counts["kalshi_rows"] = len(krows)
    log(f"record {stamp}: {counts}")
    return counts


# ---- CLI ---------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.music_charts")
    ap.add_argument("cmd")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None, help="candles: restrict to these series")
    a = ap.parse_args(argv)
    if a.cmd == "robots":
        for r in check_robots():
            print(r["host"], r["path"], r["http_status"], "ALLOWED" if r["allowed"] else "blocked")
    elif a.cmd == "discover":
        discover()
    elif a.cmd == "candles":
        fetch_candles(workers=a.workers, only=a.only)
    elif a.cmd == "hits":
        hits_index()
        hits_fetch()
    elif a.cmd == "hits-reparse":
        hits_fetch(offline=True)
    elif a.cmd == "record":
        record()
    elif a.cmd == "panel":
        build_all()
    elif a.cmd == "screen":
        with pl.Config(tbl_rows=200, tbl_width_chars=220, fmt_str_lengths=60):
            print(screen())
    elif a.cmd == "holdout":
        evaluate_frozen()
    elif a.cmd == "calib":
        mk = load_markets()
        pairs = calibration_pairs(mk, hits_table())
        dev = pairs.filter(pl.col("split") == "dev")
        print(json.dumps(fit_ratio(dev), indent=1))
        with pl.Config(tbl_rows=60, tbl_width_chars=220):
            print(dev.group_by("measure", "stage").agg(pl.len(), pl.col("event_ticker").n_unique().alias("events"),
                                                       pl.col("lr").mean().alias("mean"), pl.col("lr").median().alias("med"),
                                                       pl.col("lr").std().alias("sd"),
                                                       pl.col("lr").abs().quantile(0.9).alias("abs_q90")).sort("measure", "stage"))
    else:
        raise SystemExit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    main()
