"""Where else does the pre-event maker NO carry over? (swarm 2026-11-23, slug ``maker_carry``)

The project's one working mechanism (live on Mentions speeches, paper on 37 culture series as
``cmn``): at a fixed lead before the event, on every leg whose book is really two-sided (best YES
bid >= 1c, best YES ask <= 99c) with mid <= 5c, rest a 1-lot post-only NO bid that sells YES at
ceil(mid + m); cheapest NO first, at most 3 legs per event; filled only by a later print strictly
through the price (queue-free); held to settlement; maker fee per the series' fee_type, rounded up
per order. This module applies the rule, unchanged, volume-blind, to families nobody has tested
with it: weather (daily highs / lows, precipitation, other climate), minor sports (before the
start), politics / world / "by date" deadlines, elections, science and technology, entertainment
outside the 37 cmn series, and series first listed in the last ~3 months.

Anchors (no look-ahead): sports = the scheduled start in the event ticker
(``kalshi.ticker_start``), orders rest from start - h to the start (in-play is a different
market); everything else = the scheduled close, inferred without the actual close as
``expected_expiration_time - lag`` (lag = the series' modal settlement lag over recurring series
with >= 5 events and lags <= 48 h, else the category's), orders rest until the leg's actual close
(or scheduled close + 1 h). The decision time is ``floor_hour(anchor - h)``; a leg must be listed
and open then. Quote = the last hourly candle ending at or before the decision (<= 7 days old:
Kalshi omits hours in which nothing changed). Fill = the first hourly candle ending after the
decision (and by the end) whose trade high is strictly above the sell price; the decision sits on
an hour boundary, so no straddled hour leaks earlier prints in. Trade prints re-check the fills.

Data (Kalshi public GET only; one ``CachedClient``, >= 0.5 s between requests, <= 2 threads; raw
responses under ``data/cache/maker_carry/raw``): settled markets from ``data/tables/kalshi_markets``
(closes 2026-08-15..10-31) plus ``/markets?status=settled`` (closes 10-25..11-23) and per-series
listings for the categories the table lacks (World, Health, Social, Transportation, Exotics,
Culture); hourly OHLC candles over [decision(72 h) - 7 d, end] (live batch endpoint after the
historical cutoff, one historical request per leg before it; the ``xcat`` whole-life hourly layer
is reused where it covers the window). Events are sampled per family by a seeded hash of the event
ticker (never by volume, price path or result).

Splits: dev = anchor before 2026-10-01; holdout = 2026-10-01 .. 2026-11-22. Grid on dev only:
h in {6, 24, 72}, m in {6, 12, 18}c; per family the cell with the highest event-clustered t among
cells with >= 30 fills and >= 15 events is frozen (it "carries" only with dev t >= 2 and >= +3c);
the holdout is scored once for the frozen cell and for the unmodified cmn cell.

    python -m albert.research.maker_carry universe | sample | candles | orders | dev | holdout
    python -m albert.research.maker_carry prints | book | flow | report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

OUT = Path("data/research/maker_carry")
CACHE = Path("data/cache/maker_carry")
CANDLES = CACHE / "candles"
TABLES = Path("data/tables")
BASE = "https://api.elections.kalshi.com/trade-api/v2"
START = datetime(2026, 8, 15, tzinfo=UTC)
DEV_END = datetime(2026, 10, 1, tzinfo=UTC)
HOLD_END = datetime(2026, 11, 23, tzinfo=UTC)
TABLE_END = datetime(2026, 10, 25, tzinfo=UTC)     # table rows closing after this are refreshed from the API
NEW_SINCE = datetime(2026, 8, 23, tzinfo=UTC)      # "first listed in the last ~3 months" (as of 2026-11-23)
SEED = "maker_carry-20261123"
NEW_MIN_EVENTS = 3          # a "new series" is a recurring product (>= 3 events), not a one-off
NEW_EXTRA_CAP = 150         # new-series events added on top of a family's cap, per family and split

LEADS_H = (6, 24, 72)
MARGINS_C = (6, 12, 18)
BAND = 0.05
MAX_LEGS = 3
MAX_QUOTE_AGE = timedelta(days=7)
QUOTE_PAD = timedelta(days=7)
MIN_FILLS, MIN_EVENTS = 30, 15
CARRY_T, CARRY_C = 2.0, 3.0
CMN_CELL = (72, 12)

# the 37 cmn series (albert.paper_forward.CMN_SERIES), excluded: they are already on paper
CMN_SERIES = (
    "KXALBUMEQUIV", "KXAPRPOTUS", "KXBILLBOARDRUNNERUPALBUM", "KXBILLBOARDRUNNERUPSONG", "KXEOWEEK", "KXHORMUZMAX",
    "KXHORMUZPEAK", "KXHORMUZWEEKLY", "KXLLM1", "KXMAMDANIEO", "KXNETFLIXRANKMOVIE", "KXNETFLIXRANKMOVIEGLOBAL",
    "KXNETFLIXRANKMOVIEGLOBAL2", "KXNETFLIXRANKMOVIERUNNERUP", "KXNETFLIXRANKSHOW", "KXNETFLIXRANKSHOWGLOBAL",
    "KXNETFLIXRANKSHOWGLOBAL2", "KXNETFLIXRANKSHOWRUNNERUP", "KXNETFLIXTOPVIEWSMOVIE", "KXNETFLIXTOPVIEWSTV",
    "KXPUREALBUMS", "KXRANKLISTSONGTOP10", "KXRT", "KXTOPALBUM", "KXTOPMODEL", "KXTOPSONG", "KXTRUMPAPPROVE",
    "KXTRUMPENDORSEMENTS", "KXTRUMPNOMNUM", "KXTRUMPVH", "KXTRUTHSOCIAL", "KXTRUTHSOCIALD", "KXVHGCB",
    "KXYTDAILYTOPVIDEO", "KXYTDAILYTOPVIDEOG", "KXYTVIEWSHIGH", "KXYTVIEWSW",
)
TABLE_CATEGORIES = ("Sports", "Climate and Weather", "Entertainment", "Science and Technology", "Politics", "Elections")
EXTRA_CATEGORIES = ("World", "Health", "Social", "Transportation", "Exotics", "Culture")
MAJOR_SPORTS_RX = (r"^KX(NFL|NCAAF|NBA|WNBA|MLB|NHL|NCAAMB|NCAAWB|NCAAB|EPL|LALIGA|SERIEA|BUNDESLIGA|LIGUE1|UCL|UEL|"
                   r"MLS|PGA|UFC|F1|NASCAR|ATP(?!CHALLENGER)|WTA(?!CHALLENGER))")
ESPORTS_RX = r"^KX(CS2|LOL|VALORANT|DOTA2|R6|COD|OW|RL|SC2|ESOCCER|EBASKETBALL|EFOOTBALL)"
FAMILIES = ("wx_daily", "wx_precip", "wx_other", "sports_minor", "politics_world", "elections", "scitech",
            "entertainment", "new_series")
# events per family and split (seeded hash order); None = all
CAPS: dict[str, int | None] = {"wx_daily": 450, "wx_precip": 300, "wx_other": 300, "sports_minor": 700,
                               "politics_world": 900, "elections": 900, "scitech": 900, "entertainment": 700}


# ---- helpers -------------------------------------------------------------------------------------
def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


def _api(min_interval: float = 0.5):
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    # 8 tries: back-off sleeps of 1..64 s; a 429 storm from the shared IP then fails one request
    # (logged and skipped by the callers) instead of sleeping for half an hour
    return k.Kalshi(CachedClient("kalshi", BASE, CACHE / "raw", min_interval=max(0.5, min_interval), max_retries=8))


def _hash(s: str) -> str:
    return hashlib.sha256(f"{SEED}|{s}".encode()).hexdigest()


def cents_up(x: float) -> float:
    """Round a price up to the next cent (exact cents stay)."""
    return math.ceil(round(x * 100, 6)) / 100


def sell_price(mid: float, margin_c: int) -> float:
    return cents_up(mid + margin_c / 100)


def floor_hour(t: datetime) -> datetime:
    return t.replace(minute=0, second=0, microsecond=0)


def family_of(series: str, category: str | None, title: str | None = None) -> str | None:
    """Base family of a series, or None when excluded (financial / crypto / economics, Mentions,
    the 37 cmn series, major-league sports, esports and virtual sports)."""
    from albert.categories import kalshi_series_is_financial

    if not series or series in CMN_SERIES or category in (None, "Mentions"):
        return None
    if kalshi_series_is_financial(series, category, title):
        return None
    if category == "Sports":
        if re.match(MAJOR_SPORTS_RX, series) or re.match(ESPORTS_RX, series):
            return None
        return "sports_minor"
    if category == "Climate and Weather":
        if re.match(r"^KX(HIGH|LOWT)", series):
            return "wx_daily"
        if "RAIN" in series or "SNOW" in series:
            return "wx_precip"
        return "wx_other"
    if category in ("Politics", *EXTRA_CATEGORIES):
        return "politics_world"
    return {"Elections": "elections", "Science and Technology": "scitech", "Entertainment": "entertainment"}.get(category)


# ---- universe ------------------------------------------------------------------------------------
UNI_COLS = ["ticker", "event_ticker", "series_ticker", "category", "yes_sub_title", "market_type", "open_time",
            "close_time", "expected_expiration_time", "settlement_ts", "result", "fee_type", "fee_multiplier", "source"]


def _series_info() -> pl.DataFrame:
    return pl.read_parquet(TABLES / "kalshi_series.parquet").select(
        "series_ticker", pl.col("category").alias("s_category"), pl.col("title").alias("s_title"), "frequency",
        pl.col("fee_type").alias("s_fee_type"), pl.col("fee_multiplier").alias("s_fee_mult"))


def _api_rows(pages, info: dict[str, dict[str, Any]], source: str) -> list[dict[str, Any]]:
    from albert.venues import kalshi as k

    rows: list[dict[str, Any]] = []
    for page in pages:
        rows.extend(k.market_rows(page.payload, info, source))
    return rows


def _family_map(ser: pl.DataFrame) -> dict[str, str]:
    """series -> base family for every series in today's series list (None-family series left out)."""
    out = {}
    for r in ser.iter_rows(named=True):
        f = family_of(r["series_ticker"], r["s_category"], r["s_title"])
        if f is not None:
            out[r["series_ticker"]] = f
    return out


def universe(log=print) -> pl.DataFrame:
    """Settled binary markets closing 2026-08-15 .. 11-23 in the candidate families, one row per
    leg, with its base family. Series are mapped to families from today's series list, so excluded
    series (major sports, financial, Mentions, cmn) are dropped page by page."""
    from albert.venues import kalshi as k

    ser = _series_info()
    fam = _family_map(ser)
    info = {r["series_ticker"]: {"category": r["s_category"], "fee_type": r["s_fee_type"],
                                 "fee_multiplier": r["s_fee_mult"]} for r in ser.iter_rows(named=True)}
    keep = pl.Series(sorted(fam), dtype=pl.String).implode()
    tab = (pl.scan_parquet(TABLES / "kalshi_markets.parquet")
           .filter(pl.col("series_ticker").is_in(keep) & (pl.col("close_time") >= START) & (pl.col("close_time") < TABLE_END))
           .select(UNI_COLS).collect())
    log(f"table: {tab.height} legs closing {START:%m-%d}..{TABLE_END:%m-%d}")
    api = _api()
    frames = [tab]

    def to_frame(rows: list[dict[str, Any]]) -> None:
        if rows:
            df = pl.DataFrame([{c: r.get(c) for c in UNI_COLS} for r in rows if r["series_ticker"] in fam],
                              infer_schema_length=None)
            if df.height:
                frames.append(df)

    # every settled market closing 10-25 .. 11-23 (all categories), kept when its series has a family
    params = {"status": "settled", "mve_filter": "exclude", "limit": 1000, "min_close_ts": int(TABLE_END.timestamp()),
              "max_close_ts": int(HOLD_END.timestamp())}
    rows: list[dict[str, Any]] = []
    n = 0
    live_ck = CACHE / "universe_live.parquet"
    pages = [] if live_ck.exists() else api._cursor_pages("/markets", params, "markets")
    if live_ck.exists():
        frames.append(pl.read_parquet(live_ck))
    for page in pages:
        rows.extend(r for r in k.market_rows(page.payload, info, "live") if r["series_ticker"] in fam)
        n += 1
        if n % 25 == 0:
            to_frame(rows)
            rows = []
            log(f"  live settled pages {n}, {sum(f.height for f in frames[1:])} kept", flush=True)
    to_frame(rows)
    log(f"live: {n} pages")
    if n:
        _save(pl.concat(frames[1:], how="diagonal_relaxed"), live_ck)
    # categories the table lacks: per series, live + historical listings back to START
    extra = [s for s in ser.filter(pl.col("s_category").is_in(list(EXTRA_CATEGORIES)))["series_ticker"].to_list() if s in fam]
    rows = []
    for i, s in enumerate(extra, 1):
        try:
            rows.extend(_api_rows(api.series_markets(s, "settled", min_close=START, max_close=TABLE_END, cache=True), info,
                                  "live"))
            for page in api._cursor_pages("/historical/markets", {"series_ticker": s, "limit": 1000}, "markets"):
                got = k.market_rows(page.payload, info, "historical")
                rows.extend(r for r in got if r["close_time"] is not None and START <= r["close_time"] < TABLE_END)
                closes = [r["close_time"] for r in got if r["close_time"] is not None]
                if not closes or min(closes) < START:
                    break
        except RuntimeError as error:      # one series failing must not stop the build
            log(f"  extra series {s}: {str(error)[:160]}")
        if i % 50 == 0:
            log(f"  extra series {i}/{len(extra)}, {len(rows)} rows", flush=True)
    to_frame(rows)
    mk = pl.concat(frames, how="diagonal_relaxed").unique("ticker", keep="last")
    mk = mk.filter(pl.col("result").is_in(["yes", "no"]) & (pl.col("market_type") == "binary")
                   & pl.col("open_time").is_not_null() & pl.col("close_time").is_not_null()
                   & (pl.col("close_time") > pl.col("open_time")) & (pl.col("close_time") >= START)
                   & (pl.col("close_time") < HOLD_END))
    mk = mk.join(ser, on="series_ticker", how="left").with_columns(
        pl.coalesce("category", "s_category").alias("category"),
        pl.coalesce("fee_type", "s_fee_type").alias("fee_type"),
        pl.coalesce("fee_multiplier", "s_fee_mult").fill_null(1.0).alias("fee_multiplier"),
        pl.col("series_ticker").replace_strict(fam, default=None, return_dtype=pl.String).alias("family"))
    mk = mk.filter(pl.col("family").is_not_null()).with_columns((pl.col("result") == "yes").cast(pl.Int8).alias("y"))
    _save(mk, OUT / "universe.parquet")
    log(f"universe: {mk.height} legs, {mk['event_ticker'].n_unique()} events, {mk['series_ticker'].n_unique()} series")
    log(str(mk.group_by("family").agg(pl.len().alias("legs"), pl.col("event_ticker").n_unique().alias("events"))
            .sort("family")))
    return mk


# ---- anchors and sample --------------------------------------------------------------------------
def anchors(mk: pl.DataFrame) -> pl.DataFrame:
    """One row per event: anchor (scheduled start for sports, scheduled close otherwise), how the
    anchor was found, split. The scheduled close never uses the actual close of that event: it is
    ``expected_expiration_time`` (fixed at listing) minus a settlement lag learned on recurring
    series."""
    from albert.venues.kalshi import ticker_start

    legs = mk.with_columns(((pl.col("expected_expiration_time") - pl.col("close_time")).dt.total_minutes() / 60)
                           .alias("lag_h"))
    ev = legs.group_by("event_ticker").agg(
        pl.col("series_ticker").first(), pl.col("category").first(), pl.col("family").first(),
        pl.col("close_time").max().alias("ev_close"), pl.col("open_time").min().alias("ev_open"),
        pl.col("expected_expiration_time").max().alias("exp_exp"), pl.len().alias("legs"),
        pl.col("lag_h").max().alias("lag_h"))
    # settlement lag: modal hour on recurring series (>= 5 events), lags within [0, 48] h only
    ok = ev.filter(pl.col("lag_h").is_between(0, 48)).with_columns(pl.col("lag_h").round(0).alias("lag_r"))
    n_ev = ev.group_by("series_ticker").agg(pl.len().alias("n_ev"))
    ok = ok.join(n_ev, on="series_ticker").filter(pl.col("n_ev") >= 5)
    ser_lag = (ok.group_by("series_ticker", "lag_r").agg(pl.len().alias("n")).sort("n", descending=True)
               .group_by("series_ticker").first().select("series_ticker", pl.col("lag_r").alias("ser_lag")))
    cat_lag = (ok.group_by("category", "lag_r").agg(pl.len().alias("n")).sort("n", descending=True)
               .group_by("category").first().select("category", pl.col("lag_r").alias("cat_lag")))
    ev = ev.join(ser_lag, on="series_ticker", how="left").join(cat_lag, on="category", how="left").with_columns(
        pl.coalesce("ser_lag", "cat_lag", pl.lit(0.0)).alias("lag_used"))
    starts = {e: ticker_start(e) for e in ev.filter(pl.col("family") == "sports_minor")["event_ticker"].to_list()}
    starts |= {e: election_day(e) for e in ev.filter(pl.col("family") == "elections")["event_ticker"].to_list()}
    ev = ev.with_columns(
        pl.col("event_ticker").replace_strict(starts, default=None, return_dtype=pl.Datetime("us", "UTC")).alias("start"),
        (pl.col("exp_exp") - pl.duration(minutes=(pl.col("lag_used") * 60).cast(pl.Int64))).alias("sched_close"))
    ev = ev.with_columns(
        pl.when(pl.col("family").is_in(list(EVENT_START_FAMILIES))).then(pl.col("start")).otherwise(pl.col("sched_close"))
        .alias("anchor"),
        pl.when(pl.col("family") == "sports_minor").then(pl.lit("ticker_start"))
        .when(pl.col("family") == "elections").then(pl.lit("election_day")).otherwise(pl.lit("exp_minus_lag"))
        .alias("anchor_src"))
    ev = ev.filter(pl.col("anchor").is_not_null() & (pl.col("anchor") >= START) & (pl.col("anchor") < HOLD_END))
    return ev.with_columns(pl.when(pl.col("anchor") < DEV_END).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("split"),
                           ((pl.col("anchor") - pl.col("ev_close")).dt.total_minutes() / 60).alias("early_h"))


EVENT_START_FAMILIES = ("sports_minor", "elections")    # anchor = the event's scheduled start; orders end there
_DATE_RX = re.compile(r"-(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})")
_US26_RX = re.compile(r"(HOUSE|SENATE|GOV|ATTYGEN|SECSTATE|LTGOV|VOTEGENERAL|MIDTERM).*-26(?![0-9])")
_MON = {m: i + 1 for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"))}


def election_day(event_ticker: str) -> datetime | None:
    """Scheduled election day of an Elections event, at 10:00 UTC (before polls close anywhere in
    the Americas or Europe): the date in the ticker (``-26OCT04``), else 2026-11-03 for US 2026
    races whose ticker carries only the year (``KXHOUSERACE-GA05-26``). None when neither."""
    m = _DATE_RX.search(event_ticker or "")
    if m:
        yy, mon, dd = m.groups()
        try:
            return datetime(2000 + int(yy), _MON[mon], int(dd), 10, tzinfo=UTC)
        except ValueError:
            return None
    if _US26_RX.search(event_ticker or ""):
        return datetime(2026, 11, 3, 10, tzinfo=UTC)
    return None


def new_series_flags(mk: pl.DataFrame, log=print) -> dict[str, bool]:
    """Non-sports series first listed on/after NEW_SINCE: every universe leg opened on/after it and
    no settled market of the series closed before START (historical listing, newest first)."""
    first = mk.filter(pl.col("family") != "sports_minor").group_by("series_ticker").agg(pl.col("open_time").min().alias("a"))
    cand = first.filter(pl.col("a") >= NEW_SINCE)["series_ticker"].to_list()
    api = _api()
    out: dict[str, bool] = {}
    for i, s in enumerate(sorted(cand), 1):
        from albert.venues.kalshi import parse_ts

        old = False
        try:
            for j, page in enumerate(api._cursor_pages("/historical/markets", {"series_ticker": s, "limit": 1000}, "markets")):
                for m in page.payload.get("markets") or []:
                    o, c = parse_ts(m.get("open_time")), parse_ts(m.get("close_time"))
                    if (o is not None and o < NEW_SINCE) or (c is not None and c < START):
                        old = True
                if old or j >= 4:
                    break
        except RuntimeError as error:      # unknown: treated as not new
            log(f"  new-series check {s}: {str(error)[:120]}")
            old = True
        out[s] = not old
        if i % 50 == 0:
            log(f"  new-series check {i}/{len(cand)}", flush=True)
    log(f"new series: {sum(out.values())} of {len(cand)} candidates (first universe listing >= {NEW_SINCE:%Y-%m-%d})")
    return out


def sample(log=print) -> pl.DataFrame:
    """Seeded, volume-blind event sample per family and split; legs that can be open at a
    decision time, with their candle window."""
    mk = pl.read_parquet(OUT / "universe.parquet")
    ev = anchors(mk)
    flags_path = OUT / "new_series.json"
    if flags_path.exists():
        flags = json.loads(flags_path.read_text())
    else:
        flags = new_series_flags(mk, log=log)
        _dump(flags, flags_path)
    n_ev = ev.group_by("series_ticker").agg(pl.len().alias("ser_events"))
    ev = ev.join(n_ev, on="series_ticker").with_columns(
        (pl.col("series_ticker").replace_strict(flags, default=False, return_dtype=pl.Boolean)
         & (pl.col("ser_events") >= NEW_MIN_EVENTS)).alias("new"))
    # structural, outcome-blind: an event can trade only if some leg is listed by the latest decision
    # time (anchor - 6 h); hourly-listed series (e.g. hourly temperature) never are
    first_open = mk.group_by("event_ticker").agg(pl.col("open_time").min().alias("first_open"))
    ev = ev.join(first_open, on="event_ticker").filter(
        pl.col("first_open") <= (pl.col("anchor") - timedelta(hours=min(LEADS_H))).dt.truncate("1h"))
    xs = _xcat_windows()
    xev = set(xs["event_ticker"].to_list()) if xs.height else set()
    ev = ev.with_columns(pl.col("event_ticker").map_elements(_hash, return_dtype=pl.String).alias("h"),
                         pl.col("event_ticker").is_in(list(xev)).alias("in_xcat"))
    picked = []
    for fam, cap in CAPS.items():
        for split in ("dev", "holdout"):
            g = ev.filter((pl.col("family") == fam) & (pl.col("split") == split)).sort(["in_xcat", "h"], descending=[True, False])
            picked.append(g if cap is None else g.head(cap))
    ev_s = pl.concat(picked).with_columns(pl.lit(True).alias("in_base"))
    # new-series events are always taken whole (they are few), on top of the family caps; the extra
    # ones stay out of their base family's sample so that sample remains a plain seeded cap
    extra = (ev.filter(pl.col("new") & ~pl.col("event_ticker").is_in(ev_s["event_ticker"].implode()))
             .sort("h").with_columns(pl.int_range(pl.len()).over("family", "split").alias("_k"))
             .filter(pl.col("_k") < NEW_EXTRA_CAP).drop("_k"))
    ev_s = pl.concat([ev_s, extra.with_columns(pl.lit(False).alias("in_base"))], how="vertical_relaxed")
    legs = mk.join(ev_s.select("event_ticker", "anchor", "anchor_src", "split", "new", "in_base", "sched_close", "early_h"),
                   on="event_ticker", how="inner")
    sports = pl.col("family").is_in(list(EVENT_START_FAMILIES))
    t72 = (pl.col("anchor") - timedelta(hours=max(LEADS_H))).dt.truncate("1h")
    t6 = (pl.col("anchor") - timedelta(hours=min(LEADS_H))).dt.truncate("1h")
    end = pl.when(sports).then(pl.min_horizontal("anchor", "close_time")).otherwise(
        pl.min_horizontal("close_time", pl.col("anchor") + timedelta(hours=1)))
    legs = legs.with_columns(end.alias("end"), t6.alias("t6"), t72.alias("t72"))
    legs = legs.filter((pl.col("open_time") <= pl.col("t6")) & (pl.col("end") > pl.col("t6")))
    legs = legs.with_columns(pl.max_horizontal("open_time", pl.col("t72") - QUOTE_PAD).alias("lo"),
                             pl.col("end").dt.truncate("1h").alias("hi_floor"))
    legs = legs.with_columns(pl.when(pl.col("hi_floor") < pl.col("end")).then(pl.col("hi_floor") + timedelta(hours=1))
                             .otherwise(pl.col("hi_floor")).alias("hi")).drop("hi_floor")
    _save(legs, OUT / "sample.parquet")
    log(f"sample: {legs.height} legs, {legs['event_ticker'].n_unique()} events")
    log(str(legs.group_by("family", "split").agg(pl.len().alias("legs"), pl.col("event_ticker").n_unique().alias("events"))
            .sort("family", "split")))
    return legs


# ---- candles -------------------------------------------------------------------------------------
XCAT = Path("data/research/xcat")


def _xcat_windows() -> pl.DataFrame:
    p = XCAT / "sample.parquet"
    if not p.exists():
        return pl.DataFrame(schema={"ticker": pl.String, "event_ticker": pl.String})
    s = pl.read_parquet(p, columns=["ticker", "event_ticker", "layer", "lo", "hi"]).filter(pl.col("layer") == "hourly")
    done_p = XCAT / "hourly" / "done.parquet"
    if done_p.exists():
        s = s.filter(pl.col("ticker").is_in(pl.read_parquet(done_p)["ticker"].implode()))
    return s.select("ticker", "event_ticker", pl.col("lo").alias("x_lo"), pl.col("hi").alias("x_hi"))


def _done() -> set[str]:
    p = CACHE / "candles_done.parquet"
    return set(pl.read_parquet(p)["ticker"].to_list()) if p.exists() else set()


def candle_jobs(legs: pl.DataFrame, cutoff: datetime) -> list[dict[str, Any]]:
    """Historical legs one request each; live legs packed into batch requests (<= 50 tickers and
    <= 10,000 candles summed over tickers, the union window counted for each ticker)."""
    jobs: list[dict[str, Any]] = []
    # historical legs: families interleaved by legs fetched so far (each family gets about the same
    # request budget when the fetch is cut short), events in seeded hash order within a family
    order = (legs.select("event_ticker", "family").unique("event_ticker")
             .with_columns(pl.col("event_ticker").map_elements(_hash, return_dtype=pl.String).alias("eh"))
             .join(legs.group_by("event_ticker").agg(pl.len().alias("nl")), on="event_ticker").sort("eh")
             .with_columns((pl.col("nl").cum_sum().over("family") - pl.col("nl")).alias("legs_before")))
    rank = dict(zip(order["event_ticker"].to_list(), order["legs_before"].to_list(), strict=True))
    rows = (legs.select("ticker", "event_ticker", "lo", "hi", "settlement_ts").with_columns(pl.lit(60).alias("period"))
            .sort("hi", "lo", "ticker").to_dicts())
    cur: list[dict[str, Any]] = []
    for r in rows:
        if r["settlement_ts"] < cutoff:
            jobs.append({"kind": "hist", "rows": [r]})
            continue
        if cur:
            lo = min(min(x["lo"] for x in cur), r["lo"])
            hi = max(max(x["hi"] for x in cur), r["hi"])
            span = math.ceil((hi - lo).total_seconds() / 3600) + 1
            if span * (len(cur) + 1) > 10000 or len(cur) >= 50:
                jobs.append({"kind": "batch", "rows": cur})
                cur = []
        cur.append(r)
    if cur:
        jobs.append({"kind": "batch", "rows": cur})
    # seeded random order, so a fetch stopped early leaves a random subset (not the earliest dates)
    return sorted(jobs, key=lambda j: (j["kind"] != "batch", rank.get(j["rows"][0]["event_ticker"], 0),
                                       _hash(j["rows"][0]["event_ticker"])))


def fetch_candles(*, workers: int = 2, limit: int | None = None, skip_hist: tuple[str, ...] = (), log=print) -> None:
    from albert.research import xcat

    legs = pl.read_parquet(OUT / "sample.parquet", columns=["ticker", "event_ticker", "family", "lo", "hi", "end",
                                                          "settlement_ts"])
    xw = _xcat_windows()
    cover = legs.join(xw.drop("event_ticker"), on="ticker", how="inner").filter(
        (pl.col("x_lo") <= pl.col("lo")) & (pl.col("x_hi") >= pl.col("end") - timedelta(minutes=1)))
    reuse = set(cover["ticker"].to_list())
    done = _done()
    todo = legs.filter(~pl.col("ticker").is_in(list(reuse | done)))
    if limit:
        todo = todo.head(limit)
    api = _api()
    cutoff = api.cutoff()["market_settled_ts"]
    if skip_hist:     # families whose historical (dev) legs are not fetched further: fetched events stay
        todo = todo.filter(~(pl.col("family").is_in(list(skip_hist)) & (pl.col("settlement_ts") < cutoff)))
    jobs = candle_jobs(todo, cutoff)
    log(f"candles: {legs.height} legs, {len(reuse)} covered by xcat, {len(done)} done, {todo.height} to fetch in "
        f"{len(jobs)} requests ({sum(j['kind'] == 'hist' for j in jobs)} historical; cutoff {cutoff:%Y-%m-%d})", flush=True)
    CANDLES.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    buf: list[pl.DataFrame] = []           # one compact frame per response (dict rows are transient)
    fin: list[str] = []
    failed: list[str] = []
    part = len(list(CANDLES.glob("part-*.parquet")))
    t0 = time.time()

    def flush() -> None:
        nonlocal buf, part
        if buf:
            pl.concat(buf).write_parquet(CANDLES / f"part-{part:05d}.parquet")
            part += 1
        buf = []
        _save(pl.DataFrame({"ticker": sorted(done | set(fin))}, schema={"ticker": pl.String}), CACHE / "candles_done.parquet")
        if failed:
            _save(pl.DataFrame({"ticker": sorted(set(failed))}), CACHE / "candles_failed.parquet")

    n = 0

    def chunked():      # bounded submission: map() would queue every result behind one slow (backing-off) job
        with ThreadPoolExecutor(max_workers=min(2, workers)) as pool:
            for i in range(0, len(jobs), 8):
                yield from pool.map(lambda j: xcat._run_job(api, j), jobs[i:i + 8])

    if True:
        for res in chunked():
            with lock:
                for tk, got in res:
                    if got is None:
                        failed.append(tk)
                        continue
                    if got:
                        buf.append(pl.DataFrame(got, schema=xcat.OHLC_SCHEMA))
                    fin.append(tk)
                n += 1
                if n % 50 == 0 or sum(b.height for b in buf) > 100_000:
                    flush()
                if n % 250 == 0:
                    rate = n / (time.time() - t0)
                    log(f"  {n}/{len(jobs)} requests, {len(fin)} legs, {rate:.2f} req/s, "
                        f"eta {(len(jobs) - n) / max(rate, 1e-9) / 60:.0f} min", flush=True)
    flush()
    log(f"candles done: {len(fin)} legs fetched, {len(set(failed))} failed")


CANDLE_COLS = ("ticker", "end_ts", "p_high", "p_close", "bid_close", "ask_close", "volume")


def load_candles(tickers: list[str] | None = None) -> pl.DataFrame:
    """Hourly candles of the sample from this study's cache and the xcat hourly layer (own cache
    first on duplicates)."""
    files = sorted(CANDLES.glob("part-*.parquet")) + sorted((XCAT / "hourly").glob("part-*.parquet"))
    frames = []
    want = None if tickers is None else pl.Series(tickers, dtype=pl.String).implode()
    for f in files:
        q = pl.scan_parquet(f)
        if want is not None:
            q = q.filter(pl.col("ticker").is_in(want))
        frames.append(q.filter(pl.col("period") == 60).select(list(CANDLE_COLS)))
    c = pl.concat([q.collect() for q in frames], how="vertical_relaxed")
    return (c.with_columns(pl.col("end_ts").dt.cast_time_unit("us").dt.convert_time_zone("UTC"))
            .unique(["ticker", "end_ts"], keep="first", maintain_order=True).sort("ticker", "end_ts"))


# ---- orders and fills ----------------------------------------------------------------------------
def decision_quotes(legs: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    """(leg, h) rows open at the decision time with a real two-sided quote and mid <= BAND."""
    rows = (legs.with_columns(pl.lit(list(LEADS_H)).alias("h")).explode("h")
            .with_columns((pl.col("anchor") - pl.duration(hours=pl.col("h"))).dt.truncate("1h").alias("t"))
            .filter((pl.col("open_time") <= pl.col("t")) & (pl.col("end") > pl.col("t"))))
    q = (candles.filter(pl.col("bid_close").is_not_null() & pl.col("ask_close").is_not_null())
         .select("ticker", pl.col("end_ts").alias("q_ts"), pl.col("bid_close").alias("bid"), pl.col("ask_close").alias("ask"))
         .sort("q_ts"))
    rows = rows.sort("t").join_asof(q, left_on="t", right_on="q_ts", by="ticker", strategy="backward", check_sortedness=False)
    rows = rows.filter(pl.col("q_ts").is_not_null() & ((pl.col("t") - pl.col("q_ts")) <= MAX_QUOTE_AGE))
    two = (pl.col("bid") >= 0.01 - 1e-9) & (pl.col("ask") <= 0.99 + 1e-9) & (pl.col("ask") > pl.col("bid"))
    return rows.with_columns(two.alias("two_sided"), ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))


class Tape:
    """Per-leg hourly arrays (end time, trade high, volume) for the through-fill test."""

    def __init__(self, candles: pl.DataFrame) -> None:
        self.by: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        c = candles.filter(pl.col("p_high").is_not_null()).select("ticker", "end_ts", "p_high", "volume")
        for (tk,), g in c.group_by(["ticker"]):
            g = g.sort("end_ts")
            self.by[tk] = (g["end_ts"].dt.epoch("s").to_numpy(), g["p_high"].to_numpy(), g["volume"].fill_null(0).to_numpy())

    def first_through(self, tk: str, t0: datetime, t1: datetime, price: float) -> tuple[datetime, float] | None:
        """First candle ending in (t0, t1] whose trade high is strictly above ``price``."""
        a = self.by.get(tk)
        if a is None:
            return None
        ts, hi, vol = a
        i0 = np.searchsorted(ts, int(t0.timestamp()), side="right")
        i1 = np.searchsorted(ts, int(t1.timestamp()), side="right")
        if i1 <= i0:
            return None
        hit = hi[i0:i1] > price + 1e-9
        if not hit.any():
            return None
        j = i0 + int(np.argmax(hit))
        return datetime.fromtimestamp(int(ts[j]), UTC), float(vol[j])

    def through_volume_after(self, tk: str, t0: datetime, t1: datetime, price: float) -> float:
        """Volume of candles in (t0, t1] whose high is above ``price`` (an upper bound on the
        contracts printed through the price after the fill)."""
        a = self.by.get(tk)
        if a is None:
            return 0.0
        ts, hi, vol = a
        i0 = np.searchsorted(ts, int(t0.timestamp()), side="right")
        i1 = np.searchsorted(ts, int(t1.timestamp()), side="right")
        m = hi[i0:i1] > price + 1e-9
        return float(vol[i0:i1][m].sum())


def fill_end(r: dict[str, Any]) -> datetime:
    """Last candle end that may count: up to the close for non-sports (no prints after a close);
    for sports only whole hours before the start (the start hour holds in-play prints)."""
    end = r["end"]
    if r["family"] in EVENT_START_FAMILIES:
        return floor_hour(end)
    return end if end == floor_hour(end) else floor_hour(end) + timedelta(hours=1)


def build_orders(q: pl.DataFrame, tape: Tape) -> pl.DataFrame:
    """Every order of every (h, m) cell: two-sided, mid <= BAND, sell YES at ceil(mid + m) below
    99c and above the YES bid; per event and cell the cheapest NO first, at most MAX_LEGS."""
    from albert.fees import kalshi_fee

    base = q.filter(pl.col("two_sided") & (pl.col("mid") <= BAND + 1e-9))
    rows = []
    for r in base.iter_rows(named=True):
        fe = fill_end(r)
        for m in MARGINS_C:
            s = sell_price(r["mid"], m)
            if s >= 0.99 - 1e-9 or s <= r["bid"] + 1e-9:
                continue
            price = round(1 - s, 2)
            rows.append({"ticker": r["ticker"], "event_ticker": r["event_ticker"], "series_ticker": r["series_ticker"],
                         "family": r["family"], "new": r["new"], "in_base": r["in_base"], "split": r["split"], "h": r["h"], "m": m,
                         "t": r["t"], "q_ts": r["q_ts"], "bid": r["bid"], "ask": r["ask"], "mid": r["mid"], "s": s,
                         "price": price, "fill_end": fe, "close_time": r["close_time"], "anchor": r["anchor"],
                         "settlement_ts": r["settlement_ts"], "y": r["y"],
                         "fee": kalshi_fee(1, price, maker=True, fee_type=r["fee_type"], multiplier=r["fee_multiplier"])})
    o = pl.DataFrame(rows, infer_schema_length=None)
    o = (o.sort("event_ticker", "h", "m", "price", "ticker")
         .with_columns(pl.int_range(pl.len()).over("event_ticker", "h", "m").alias("rank")))
    o = o.filter(pl.col("rank") < MAX_LEGS)
    fills = []
    for r in o.iter_rows(named=True):
        hit = tape.first_through(r["ticker"], r["t"], r["fill_end"], r["s"])
        if hit is None:
            fills.append((None, None, None))
            continue
        fills.append((hit[0], hit[1], tape.through_volume_after(r["ticker"], hit[0], r["fill_end"], r["s"])))
    o = o.with_columns(pl.Series("fill_ts", [f[0] for f in fills], dtype=pl.Datetime("us", "UTC")),
                       pl.Series("fill_vol", [f[1] for f in fills], dtype=pl.Float64),
                       pl.Series("through_vol_after", [f[2] for f in fills], dtype=pl.Float64))
    return o.with_columns(pl.when(pl.col("fill_ts").is_not_null())
                          .then(100 * ((1 - pl.col("y")) - pl.col("price") - pl.col("fee"))).alias("pnl_c"),
                          pl.col("fill_ts").dt.date().alias("day"),
                          (pl.col("event_ticker").str.replace(r"^[A-Z0-9]+-", "")).alias("game"))


def fetched_events(legs: pl.DataFrame) -> set[str]:
    """Events whose every sample leg was fetched (or is covered by the xcat layer). The fetch runs
    in seeded event-hash order, so stopping it early leaves a random subset of whole events."""
    xw = _xcat_windows()
    cover = legs.join(xw.drop("event_ticker"), on="ticker", how="inner").filter(
        (pl.col("x_lo") <= pl.col("lo")) & (pl.col("x_hi") >= pl.col("end") - timedelta(minutes=1)))
    ok = _done() | set(cover["ticker"].to_list())
    g = legs.group_by("event_ticker").agg(pl.col("ticker").is_in(list(ok)).all().alias("all"))
    return set(g.filter(pl.col("all"))["event_ticker"].to_list())


def run_orders(log=print) -> pl.DataFrame:
    legs = pl.read_parquet(OUT / "sample.parquet")
    complete = fetched_events(legs)
    n0 = legs["event_ticker"].n_unique()
    legs = legs.filter(pl.col("event_ticker").is_in(list(complete)))
    log(f"orders: {len(complete)} of {n0} sample events fully fetched")
    log(str(legs.group_by("family", "split").agg(pl.col("event_ticker").n_unique().alias("events")).sort("family", "split")))
    c = load_candles(legs["ticker"].unique().to_list())
    have = set(c["ticker"].unique().to_list())
    log(f"orders: {legs.height} sample legs, {len(have & set(legs['ticker']))} with candles")
    q = decision_quotes(legs, c)
    _save(q.select("ticker", "event_ticker", "family", "new", "in_base", "split", "h", "t", "q_ts", "bid", "ask", "mid", "two_sided",
                   "y"), OUT / "quotes.parquet")
    o = build_orders(q, Tape(c))
    _save(o, OUT / "orders.parquet")
    log(f"orders: {o.height} (leg, h, m) orders, {o['fill_ts'].is_not_null().sum()} fills")
    return o


# ---- statistics ----------------------------------------------------------------------------------
def cl_t(x: np.ndarray, g: np.ndarray) -> tuple[float, float | None, int]:
    from albert.research.fundamental import clustered_mean

    return clustered_mean(np.asarray(x, dtype=float), np.asarray(g))


def two_way_t(x: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> float | None:
    from albert.research.culture_flow import two_way_t as tw

    return tw(x, g1, g2)


def cluster_key(df: pl.DataFrame) -> np.ndarray:
    """Event, except sports: the game (several events share one match)."""
    return df.select(pl.when(pl.col("family") == "sports_minor").then(pl.col("game")).otherwise(pl.col("event_ticker")))\
        .to_series().to_numpy()


def summarize(f: pl.DataFrame) -> dict[str, Any]:
    """Per-fill statistics of filled orders (``pnl_c`` not null)."""
    f = f.filter(pl.col("pnl_c").is_not_null())
    if f.is_empty():
        return {"fills": 0}
    x = f["pnl_c"].to_numpy()
    mu, t, k = cl_t(x, cluster_key(f))
    t2 = two_way_t(x, cluster_key(f), f["day"].cast(pl.String).to_numpy())
    cost = f["price"] + f["fee"]
    return {"fills": f.height, "events": f["event_ticker"].n_unique(), "clusters": k, "mean_c": round(mu, 2),
            "t": None if t is None else round(t, 2), "t_2way": None if t2 is None else round(t2, 2),
            "pnl_per_usd": round(float(f["pnl_c"].sum() / 100 / cost.sum()), 4),
            "yes_rate": round(float(f["y"].mean()), 4), "breakeven_yes": round(float((1 - cost).mean()), 4),
            "avg_sell_at": round(float(f["s"].mean()), 3), "avg_mid": round(float(f["mid"].mean()), 4)}


def cell_table(o: pl.DataFrame, split: str) -> pl.DataFrame:
    rows = []
    for fam in FAMILIES:
        sub = fam_rows(o, fam).filter(pl.col("split") == split)
        for h in LEADS_H:
            for m in MARGINS_C:
                c = sub.filter((pl.col("h") == h) & (pl.col("m") == m))
                rows.append({"family": fam, "split": split, "h": h, "m": m, "orders": c.height,
                             "order_events": c["event_ticker"].n_unique(), **summarize(c)})
    return pl.DataFrame(rows, infer_schema_length=None)


def fam_rows(df: pl.DataFrame, fam: str) -> pl.DataFrame:
    """Rows of one family: base families use the capped seeded sample only; ``new_series`` uses
    every event of a series first listed since NEW_SINCE."""
    if fam == "new_series":
        return df.filter(pl.col("new"))
    return df.filter((pl.col("family") == fam) & pl.col("in_base"))


def default_cell(cells: pl.DataFrame, fam: str) -> tuple[int, int]:
    """The unmodified cmn cell (72 h, 12c), or 24 h when the family has no 72 h orders on dev."""
    c = cells.filter((pl.col("family") == fam) & (pl.col("h") == 72) & (pl.col("m") == 12))
    if c.height and (c["orders"][0] or 0) > 0:
        return CMN_CELL
    return (24, 12)


def run_dev(log=print) -> dict[str, Any]:
    o = pl.read_parquet(OUT / "orders.parquet")
    cells = cell_table(o, "dev")
    cells.write_csv(OUT / "cells_dev.csv")
    frozen: dict[str, Any] = {"frozen_at": datetime.now(UTC).isoformat(), "n_dev_cells": cells.height,
                              "selection": f"max event-clustered t among dev cells with >= {MIN_FILLS} fills and "
                                           f">= {MIN_EVENTS} events; carries if t >= {CARRY_T} and mean >= {CARRY_C}c",
                              "families": {}}
    for fam in FAMILIES:
        c = cells.filter((pl.col("family") == fam) & (pl.col("fills") >= MIN_FILLS) & (pl.col("events") >= MIN_EVENTS)
                         & pl.col("t").is_not_null())
        best = c.sort("t", descending=True).head(1).to_dicts()
        b = best[0] if best else None
        carries = bool(b and b["t"] >= CARRY_T and b["mean_c"] >= CARRY_C)
        frozen["families"][fam] = {"best": b, "carries": carries, "default_cell": list(default_cell(cells, fam)),
                                   "frozen_cell": [b["h"], b["m"]] if carries else None}
    _dump(frozen, OUT / "frozen.json")
    log(json.dumps(frozen, indent=1, default=str))
    return frozen


def p_few_yes(f: pl.DataFrame) -> float | None:
    """Exact one-sided p of seeing this few YES settlements among the fills if every fill's YES
    probability were its own break-even (1 - NO price - fee): zero edge. Poisson-binomial by DP.
    Unlike a clustered t, it is defined when no fill lost."""
    f = f.filter(pl.col("pnl_c").is_not_null())
    if f.is_empty():
        return None
    q = (1 - f["price"] - f["fee"]).clip(0.0, 1.0).to_numpy()
    dist = np.zeros(len(q) + 1)
    dist[0] = 1.0
    for x in q:
        dist[1:] = dist[1:] * (1 - x) + dist[:-1] * x
        dist[0] *= 1 - x
    return float(dist[: int(f["y"].sum()) + 1].sum())


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvals, key=lambda k: pvals[k])
    n = len(keys)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (n - i) * pvals[k]))
        out[k] = run
    return out


def p_one_sided(t: float | None) -> float:
    if t is None:
        return 1.0
    return 0.5 * math.erfc(t / math.sqrt(2))


def run_holdout(log=print) -> dict[str, Any]:
    fr = json.loads((OUT / "frozen.json").read_text())
    o = pl.read_parquet(OUT / "orders.parquet")
    res: dict[str, Any] = {"frozen_at": fr["frozen_at"], "scored_at": datetime.now(UTC).isoformat(), "tests": {}}
    for fam, f in fr["families"].items():
        sub = fam_rows(o, fam)
        cells = {"default": tuple(f["default_cell"])}
        if f["frozen_cell"]:
            cells["frozen"] = tuple(f["frozen_cell"])
        for kind, (h, m) in cells.items():
            if kind == "default" and f["frozen_cell"] and tuple(f["frozen_cell"]) == (h, m):
                continue
            key = f"{fam}|{kind}|h{h}|m{m}"
            out = {}
            for split in ("dev", "holdout"):
                c = sub.filter((pl.col("split") == split) & (pl.col("h") == h) & (pl.col("m") == m))
                out[split] = {"orders": c.height, **summarize(c), "weeks_positive_pct": weeks_positive(c),
                              "p_few_yes": p_few_yes(c)}
            res["tests"][key] = out
    if "pooled_default" in fr:
        parts = [fam_rows(o, f).filter((pl.col("h") == c[0]) & (pl.col("m") == c[1])) for f, c in fr["pooled_default"]["cells"].items()]
        po = pl.concat(parts, how="vertical_relaxed")
        res["tests"]["pooled_all|default"] = {
            split: {"orders": po.filter(pl.col("split") == split).height, **summarize(po.filter(pl.col("split") == split)),
                    "weeks_positive_pct": weeks_positive(po.filter(pl.col("split") == split)),
                    "p_few_yes": p_few_yes(po.filter(pl.col("split") == split))} for split in ("dev", "holdout")}
    # primary p: the exact few-YES test (a clustered t is degenerate when no fill lost); the t-based
    # normal p is kept alongside
    pv = {k: (v["holdout"].get("p_few_yes") if v["holdout"].get("p_few_yes") is not None else 1.0)
          for k, v in res["tests"].items()}
    pt = {k: p_one_sided(v["holdout"].get("t")) for k, v in res["tests"].items()}
    adj = holm(pv)
    for k in res["tests"]:
        res["tests"][k]["holdout"]["p_t_one_sided"] = round(pt[k], 5)
        res["tests"][k]["holdout"]["p_holm"] = round(adj[k], 5)
    res["n_holdout_tests"] = len(pv)
    _dump(res, OUT / "holdout.json")
    log(json.dumps(res, indent=1, default=str))
    # descriptive only, written after the frozen tests: every holdout cell
    cell_table(o, "holdout").write_csv(OUT / "cells_holdout_descriptive.csv")
    return res


def weeks_positive(c: pl.DataFrame) -> float | None:
    f = c.filter(pl.col("pnl_c").is_not_null())
    if f.is_empty():
        return None
    w = f.group_by(pl.col("fill_ts").dt.truncate("1w")).agg(pl.col("pnl_c").sum())
    return round(100 * float((w["pnl_c"] > 0).mean()), 1)


# ---- cash-aware book -----------------------------------------------------------------------------
def marks_for(tickers: list[str]) -> pl.DataFrame:
    """Hourly YES marks: a real two-sided mid (0 < bid < ask < 1), else the last trade."""
    c = load_candles(tickers)
    two = (pl.col("bid_close") > 0) & (pl.col("ask_close") < 1) & (pl.col("ask_close") > pl.col("bid_close"))
    mid = pl.when(two).then((pl.col("bid_close") + pl.col("ask_close")) / 2).otherwise(pl.col("p_close"))
    return (c.with_columns(mid.alias("yes_mid")).filter(pl.col("yes_mid").is_not_null())
            .select("ticker", pl.col("end_ts").alias("ts"), "yes_mid"))


def resting_book(o: pl.DataFrame, name: str, split: str, *, bankroll: float, event_cap: float = 5.0) -> dict[str, Any]:
    """culture_flow.resting_book with this study's columns: a resting bid reserves its cost from
    placement to its fill or end; a fill holds it to settlement. Orders are taken in time order
    (cheapest NO first) while reserved + open stays within ``bankroll`` and the event within
    ``event_cap``."""
    import heapq

    from albert.research.vol_metrics import ledger_metrics

    heap: list[tuple[datetime, int, float, str]] = []
    used = 0.0
    per_ev: dict[str, float] = {}
    led, seq, placed = [], 0, 0
    for r in o.sort("t", "price", "ticker").iter_rows(named=True):
        t = r["t"]
        while heap and heap[0][0] <= t:
            _, _, amt, ev = heapq.heappop(heap)
            used -= amt
            per_ev[ev] -= amt
        c = r["price"] + r["fee"]
        if used + c > bankroll + 1e-9 or per_ev.get(r["event_ticker"], 0.0) + c > event_cap + 1e-9:
            continue
        used += c
        per_ev[r["event_ticker"]] = per_ev.get(r["event_ticker"], 0.0) + c
        seq += 1
        placed += 1
        if r["fill_ts"] is None:
            heapq.heappush(heap, (r["fill_end"], seq, c, r["event_ticker"]))
            continue
        settle = max(r["settlement_ts"], r["fill_ts"])
        heapq.heappush(heap, (settle, seq, c, r["event_ticker"]))
        led.append({"strategy": "maker_carry", "variant": name, "split": split, "struct_id": f"{r['ticker']}|{r['h']}|{r['m']}",
                    "event_ticker": r["event_ticker"], "ticker": r["ticker"], "side": "no", "n": 1, "t_in": r["fill_ts"],
                    "cost_usd": c, "settle_ts": settle, "payoff_usd": float(1 - r["y"])})
    if not led:
        return {"orders_placed": placed, "positions": 0}
    ledger = pl.DataFrame(led)
    _save(ledger, OUT / "ledgers" / f"ledger_{name}_{split}.parquet")
    m = ledger_metrics(ledger, marks_for(ledger["ticker"].unique().to_list()))
    keep = ("positions", "events", "pnl_usd", "usd_per_week", "ann_return_pct", "ann_vol_pct", "sharpe", "sharpe_ci90",
            "p_sharpe_pos", "sharpe_ex_top5", "usd_per_week_ex_top5", "sortino", "max_dd_usd", "weeks_positive_pct", "skew",
            "first_day", "last_day")
    return {"orders_placed": placed, **{k: m.get(k) for k in keep}}


def run_book(log=print) -> dict[str, Any]:
    """Cash-aware books on the $35 left beside mentions ($40) and cmn ($25), plus $100 for scale:
    each family's tested cells alone, and every carried family's frozen cell pooled."""
    fr = json.loads((OUT / "frozen.json").read_text())
    o = pl.read_parquet(OUT / "orders.parquet")
    res: dict[str, Any] = {}
    pooled = []
    for fam, f in fr["families"].items():
        sub = fam_rows(o, fam)
        for kind in ("frozen", "default"):
            cell = f["frozen_cell"] if kind == "frozen" else f["default_cell"]
            if cell is None:
                continue
            h, m = cell
            c = sub.filter((pl.col("h") == h) & (pl.col("m") == m))
            if kind == "frozen" and fam != "new_series":
                pooled.append(c)
            for split in ("dev", "holdout"):
                for cash in (35.0, 100.0):
                    key = f"{fam}|{kind}|h{h}m{m}|{split}|${cash:.0f}"
                    res[key] = resting_book(c.filter(pl.col("split") == split), f"{fam}_{kind}_h{h}m{m}_c{cash:.0f}", split,
                                            bankroll=cash)
    if "pooled_default" in fr:          # every base family at its unmodified cmn cell, one book
        pd_ = pl.concat([fam_rows(o, f).filter((pl.col("h") == c[0]) & (pl.col("m") == c[1]))
                         for f, c in fr["pooled_default"]["cells"].items()], how="vertical_relaxed")
        for split in ("dev", "holdout"):
            for cash in (35.0, 100.0):
                res[f"pooled_default|{split}|${cash:.0f}"] = resting_book(pd_.filter(pl.col("split") == split),
                                                                          f"pooled_default_c{cash:.0f}", split, bankroll=cash)
    if pooled:
        p = pl.concat(pooled, how="vertical_relaxed")
        for split in ("dev", "holdout"):
            for cash in (20.0, 35.0, 100.0):
                res[f"pooled_frozen|{split}|${cash:.0f}"] = resting_book(p.filter(pl.col("split") == split),
                                                                         f"pooled_c{cash:.0f}", split, bankroll=cash)
    _dump(res, OUT / "book.json")
    log(json.dumps(res, indent=1, default=str))
    return res


# ---- trade prints for the fills ------------------------------------------------------------------
def fetch_prints(which: str = "tested", log=print) -> pl.DataFrame:
    """Kalshi trade prints over [t, fill_end] for the filled orders of the tested cells (holdout
    and dev), one leg at a time. Checks the candle fills and gives the flow after each fill."""
    from albert.venues import kalshi as k

    fr = json.loads((OUT / "frozen.json").read_text())
    o = pl.read_parquet(OUT / "orders.parquet")
    keep = []
    for fam, f in fr["families"].items():
        sub = fam_rows(o, fam)
        for cell in (f["frozen_cell"], f["default_cell"]):
            if cell:
                keep.append(sub.filter((pl.col("h") == cell[0]) & (pl.col("m") == cell[1])))
    o = pl.concat(keep).unique(["ticker", "h", "m"])
    legs = (o.filter(pl.col("fill_ts").is_not_null()).group_by("ticker")
            .agg(pl.col("t").min().alias("lo"), pl.col("fill_end").max().alias("hi"), pl.col("settlement_ts").first()))
    d = CACHE / "trades"
    d.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in d.glob("part-*.parquet"):
        done |= set(pl.read_parquet(f, columns=["leg"])["leg"].to_list())
    todo = legs.filter(~pl.col("ticker").is_in(list(done)))
    api = _api()
    cut = api.cutoff()["trades_created_ts"]
    log(f"prints: {legs.height} filled legs, {todo.height} to fetch (cutoff {cut:%Y-%m-%d})", flush=True)
    rows: list[dict[str, Any]] = []
    nparts = len(list(d.glob("part-*.parquet")))
    schema = {"leg": pl.String, "trade_id": pl.String, "created_time": pl.Datetime("us", "UTC"), "count": pl.Float64,
              "yes_price": pl.Float64, "taker_side": pl.String}

    def flush(names: list[str]) -> None:
        nonlocal rows, nparts
        df = pl.DataFrame(rows, schema=schema)
        empty = sorted(set(names) - set(df["leg"].to_list()))
        if empty:
            df = pl.concat([df, pl.DataFrame({"leg": empty}, schema={"leg": pl.String}).with_columns(
                [pl.lit(None, dtype=v).alias(c) for c, v in schema.items() if c != "leg"])])
        nparts += 1
        _save(df, d / f"part-{nparts:04d}.parquet")
        rows = []

    names: list[str] = []
    for i, r in enumerate(todo.iter_rows(named=True), 1):
        hist = cut is not None and r["settlement_ts"] < cut
        try:
            for page in api.trades(r["ticker"], hist, min_ts=r["lo"] - timedelta(hours=1), max_ts=r["hi"] + timedelta(hours=1)):
                for tr in k.trade_rows(page.payload):
                    rows.append({"leg": r["ticker"], **{c: tr[c] for c in ("trade_id", "created_time", "count", "yes_price",
                                                                          "taker_side")}})
        except RuntimeError as error:
            log(f"  trades {r['ticker']}: {str(error)[:120]}")
            continue
        names.append(r["ticker"])
        if i % 100 == 0:
            flush(names)
            names = []
            log(f"  {i}/{todo.height} legs", flush=True)
    if names:
        flush(names)
    return load_prints()


def load_prints() -> pl.DataFrame:
    files = sorted((CACHE / "trades").glob("part-*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in files]).unique(["leg", "trade_id"], keep="first")


def print_check(o: pl.DataFrame, tr: pl.DataFrame) -> pl.DataFrame:
    """Per filled order: the first print strictly above the sell price in (t, fill window], its
    taker side, and the YES-taker contracts printed through the price after it (informed flow)."""
    by = {k[0]: g.sort("created_time") for k, g in tr.filter(pl.col("trade_id").is_not_null()).group_by(["leg"])}
    rows = []
    for r in o.filter(pl.col("fill_ts").is_not_null()).iter_rows(named=True):
        g = by.get(r["ticker"])
        out = {"ticker": r["ticker"], "h": r["h"], "m": r["m"], "print_fill_ts": None}
        if g is not None:
            end = r["fill_end"] if r["family"] not in EVENT_START_FAMILIES else r["anchor"]
            a = g.filter((pl.col("created_time") > r["t"]) & (pl.col("created_time") <= end))
            thr = a.filter(pl.col("yes_price") > r["s"] + 1e-9)
            if thr.height:
                f = thr.row(0, named=True)
                post = thr.filter(pl.col("created_time") >= f["created_time"])
                out.update({"print_fill_ts": f["created_time"], "fill_taker": f["taker_side"],
                            "sweep_contracts": float(a.filter(pl.col("created_time") == f["created_time"])["count"].sum()),
                            "through_after": float(post.filter(pl.col("taker_side") == "yes")["count"].sum())})
        rows.append(out)
    return pl.DataFrame(rows, infer_schema_length=None)


def run_prints(log=print) -> dict[str, Any]:
    fr = json.loads((OUT / "frozen.json").read_text())
    o = pl.read_parquet(OUT / "orders.parquet")
    tr = load_prints()
    if tr.is_empty():
        tr = fetch_prints(log=log)
    res: dict[str, Any] = {}
    for fam, f in fr["families"].items():
        sub = fam_rows(o, fam)
        for kind in ("frozen", "default"):
            cell = f["frozen_cell"] if kind == "frozen" else f["default_cell"]
            if not cell:
                continue
            c = sub.filter((pl.col("h") == cell[0]) & (pl.col("m") == cell[1]) & pl.col("fill_ts").is_not_null())
            if c.is_empty():
                continue
            pc = print_check(c, tr)
            j = c.join(pc.drop("h", "m"), on="ticker", how="left")
            for split in ("dev", "holdout"):
                s = j.filter(pl.col("split") == split)
                if s.is_empty():
                    continue
                pf = s.filter(pl.col("print_fill_ts").is_not_null())
                heavy = pf.filter(pl.col("through_after") >= 200)
                light = pf.filter(pl.col("through_after") < 200)
                res[f"{fam}|{kind}|h{cell[0]}m{cell[1]}|{split}"] = {
                    "candle_fills": s.height, "print_confirmed": pf.height,
                    "print_pnl": summarize(pf), "yes_taker_fill_share": round(float((pf["fill_taker"] == "yes").mean()), 3)
                    if pf.height else None,
                    "heavy_after_200": summarize(heavy), "light_after": summarize(light),
                    "losers_heavy_share": (round(float(heavy.filter(pl.col("y") == 1).height /
                                                       max(1, pf.filter(pl.col("y") == 1).height)), 3)
                                           if pf.filter(pl.col("y") == 1).height else None)}
    _dump(res, OUT / "prints.json")
    log(json.dumps(res, indent=1, default=str))
    return res


# ---- flow description: where is the retail YES-lottery flow? -------------------------------------
def run_flow(log=print) -> pl.DataFrame:
    """Per family and split (no selection): how often a <= 5c two-sided leg trades strictly
    through mid + 12c before its anchor (fill rate), how often such legs then settle YES (vs the
    break-even), and the share of cheap legs whose book was two-sided at all."""
    q = pl.read_parquet(OUT / "quotes.parquet")
    o = pl.read_parquet(OUT / "orders.parquet").filter(pl.col("m") == 12)
    rows = []
    for fam in FAMILIES:
        qs = fam_rows(q, fam)
        os_ = fam_rows(o, fam)
        for split in ("dev", "holdout"):
            for h in LEADS_H:
                a = qs.filter((pl.col("split") == split) & (pl.col("h") == h))
                cheap = a.filter(pl.col("ask") <= 2 * BAND + 1e-9)
                b = os_.filter((pl.col("split") == split) & (pl.col("h") == h))
                f = b.filter(pl.col("fill_ts").is_not_null())
                rows.append({"family": fam, "split": split, "h": h, "quoted_legs": a.height,
                             "cheap_legs_ask_le_10c": cheap.height,
                             "cheap_two_sided_share": round(float(cheap["two_sided"].mean()), 3) if cheap.height else None,
                             "orders_m12": b.height, "fill_rate": round(f.height / b.height, 4) if b.height else None,
                             "fills": f.height, "yes_rate_fills": round(float(f["y"].mean()), 4) if f.height else None,
                             "yes_rate_unfilled": round(float(b.filter(pl.col("fill_ts").is_null())["y"].mean()), 4)
                             if b.height > f.height else None,
                             "breakeven_yes": round(float((1 - f["price"] - f["fee"]).mean()), 4) if f.height else None,
                             "median_fill_vol": float(f["fill_vol"].median()) if f.height else None})
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(OUT / "flow.csv")
    log(str(df))
    return df


# ---- markdown tables for the report --------------------------------------------------------------
def _f(v: Any, k: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{k}f}"
    return str(v)


def run_tables(log=print) -> str:
    """Markdown tables of every output (dev grid, frozen picks, holdout tests, books, prints, flow)."""
    out: list[str] = []
    cells = pl.read_csv(OUT / "cells_dev.csv")
    out.append("### Dev grid (every cell; c/fill, event-clustered t)\n")
    out.append("| family | h | m | orders | fills | events | c/fill | t | YES rate | break-even |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in cells.iter_rows(named=True):
        out.append(f"| {r['family']} | {r['h']} | {r['m']} | {r['orders']} | {r['fills']} | {_f(r.get('events'))} | "
                   f"{_f(r.get('mean_c'), 1)} | {_f(r.get('t'))} | {_f(r.get('yes_rate'), 3)} | {_f(r.get('breakeven_yes'), 3)} |")
    ho = json.loads((OUT / "holdout.json").read_text()) if (OUT / "holdout.json").exists() else {"tests": {}}
    out.append("\n### Frozen tests: dev -> holdout (scored once)\n")
    out.append("| test | dev fills / events | dev c/fill (t) | holdout orders | holdout fills / events | holdout c/fill (t, 2-way t) "
               "| P&L per $ | YES rate vs break-even | weeks + | p (Holm) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, v in ho["tests"].items():
        d, h = v["dev"], v["holdout"]
        out.append(f"| {k.replace('|', ' ')} | {d.get('fills', 0)} / {_f(d.get('events'))} | {_f(d.get('mean_c'), 1)} ({_f(d.get('t'))}) | "
                   f"{h.get('orders')} | {h.get('fills', 0)} / {_f(h.get('events'))} | {_f(h.get('mean_c'), 1)} "
                   f"({_f(h.get('t'))}, {_f(h.get('t_2way'))}) | {_f(h.get('pnl_per_usd'), 3)} | "
                   f"{_f(h.get('yes_rate'), 3)} vs {_f(h.get('breakeven_yes'), 3)} | {_f(h.get('weeks_positive_pct'), 0)}% | "
                   f"{_f(h.get('p_holm'), 3)} |")
    if (OUT / "book.json").exists():
        bk = json.loads((OUT / "book.json").read_text())
        out.append("\n### Cash-aware books (resting bids reserve collateral; $5 per event)\n")
        out.append("| book | orders placed | fills | $/week | Sharpe [90% CI] | ex top-5 | vol % | max DD $ | weeks + % |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for k, v in bk.items():
            ci = v.get("sharpe_ci90")
            ci_s = f" [{_f(ci[0])}, {_f(ci[1])}]" if isinstance(ci, list | tuple) and len(ci) == 2 else ""
            out.append(f"| {k.replace('|', ' ')} | {v.get('orders_placed')} | {v.get('positions')} | {_f(v.get('usd_per_week'), 2)} | "
                       f"{_f(v.get('sharpe'))}{ci_s} | {_f(v.get('sharpe_ex_top5'))} | {_f(v.get('ann_vol_pct'), 1)} | "
                       f"{_f(v.get('max_dd_usd'))} | {_f(v.get('weeks_positive_pct'), 0)} |")
    md = "\n".join(out)
    (OUT / "tables.md").write_text(md)
    log(md)
    return md


# ---- CLI -----------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.maker_carry")
    ap.add_argument("step", choices=["universe", "sample", "candles", "orders", "dev", "holdout", "prints", "book", "flow",
                                     "fetch-all", "newseries", "tables"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-hist", nargs="*", default=[])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    def log(*x, **kw):
        print(*x, flush=True)

    if a.step == "universe":
        universe(log=log)
    elif a.step == "sample":
        sample(log=log)
    elif a.step == "candles":
        fetch_candles(workers=a.workers, limit=a.limit, skip_hist=tuple(a.skip_hist), log=log)
    elif a.step == "orders":
        run_orders(log=log)
    elif a.step == "dev":
        run_dev(log=log)
    elif a.step == "holdout":
        run_holdout(log=log)
    elif a.step == "prints":
        run_prints(log=log)
    elif a.step == "book":
        run_book(log=log)
    elif a.step == "flow":
        run_flow(log=log)
    elif a.step == "tables":
        run_tables(log=log)
    elif a.step == "newseries":
        mk = pl.read_parquet(OUT / "universe.parquet", columns=["series_ticker", "family", "open_time"])
        _dump(new_series_flags(mk, log=log), OUT / "new_series.json")
    elif a.step == "fetch-all":          # one queue slot for the whole data build
        if not (OUT / "universe.parquet").exists():
            universe(log=log)
        if not (OUT / "sample.parquet").exists():
            sample(log=log)
        fetch_candles(workers=a.workers, limit=a.limit, log=log)


if __name__ == "__main__":
    main()
