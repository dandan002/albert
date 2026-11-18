"""Recurring Kalshi series: does the incumbent (or the last value) carry over more or less often
than the price says?

Kalshi lists many series that ask the same question every day, week or month (chart #1s on
Netflix / Billboard / YouTube, AI leaderboards, post counts, approval and traffic ladders, ...).
Everything this study's model uses comes from *earlier settled events of the same series*, so it is
known at the decision time and no external data is needed.

Steps (all public Kalshi data; nothing here places orders):

1. ``series``: candidate series from ``/series`` (real ``category`` values; financial series dropped
   with ``albert.categories``; Sports, Climate and Weather, Mentions and Elections dropped: games and
   speeches are not the same question, weather is forecast-driven and studied elsewhere), whose
   frequency is daily / weekly / monthly or that settled >= 4 events in the shared Aug 15 - Oct 31
   markets table. ``/events?series_ticker=&status=settled`` counts settled events; series with
   >= ``MIN_EVENTS`` are kept.
2. ``markets``: every market of the kept series (``/historical/markets?series_ticker=`` for markets
   settled before the historical cutoff, ``/markets?series_ticker=&status=settled`` after), with
   ``result``, ``expiration_value`` (the settled value or winner), strikes and times.
3. ``features``: per event (in period order) the winner / settlement value and, per leg, the
   *own-history* state known at a decision time: the last settled event's outcome for the same leg
   (same name, same strike or same bucket), whether the leg is the incumbent (previous winner, or the
   bucket holding the last value), the incumbent's streak, and the series' empirical repeat rate.
   "Known" means the previous event's ``settlement_ts`` is before the decision time.
4. ``candles``: hourly candles for the legs we model (incumbent + the legs next to it) at three
   fixed decision times per event: open + 1 h, mid-life, close - 24 h (only candles that ended at or
   before the decision count).
5. ``study``: model = shrunk empirical repeat rate by streak (fit on dev, < 2026-10-01), compared
   with the price; taker at the ask / bid with the fee rounded up per order; maker at model +- margin
   filled only when a later candle traded strictly through; one primary rule pre-specified on dev;
   holdout 2026-10-01 .. 2026-11-17 and a monthly walk-forward; capped-book ledgers through
   ``albert.research.vol_metrics``.

    python -m albert.research.recurring series | markets | features | candles [--max-requests N]
    python -m albert.research.recurring study dev      # tables on dev + frozen primary.json
    python -m albert.research.recurring study holdout  # holdout, walk-forward, ledgers
    python -m albert.research.recurring report         # data/research/recurring/report.md

Result (2026-11-17): no edge (see ``VERDICT`` / report.md). Outcomes persist strongly, but the
price already holds it wherever the book is tight; own history adds information only at open + 1 h
in 35-65c books, inside the spread. The frozen taker rule loses on dev, holdout and walk-forward
per contract; its positive $ ledgers are lottery skew (Sharpe ex top-5 events < 0).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.venues.kalshi import _field, num, parse_ts

NAME = "recurring"
OUT = Path("data/research/recurring")
TABLES = Path("data/tables")
SNAP = "2026-11-17"                                  # cache tag for listings (they grow over time)
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
END = datetime(2026, 11, 17, tzinfo=UTC)
MIN_EVENTS = 10
PERIODIC = ("daily", "weekly", "monthly")
EXCLUDE_CATEGORIES = ("Sports", "Climate and Weather", "Mentions", "Elections")
MIN_INTERVAL = 1.0                                   # <= 1 request/s, sequential (shared IP)


# ---- API -----------------------------------------------------------------------------------------
def _client(s, min_interval: float = MIN_INTERVAL):
    """Cached public client (raw responses under ``data/raw/kalshi``, append-only). 429s back off
    with waits capped at 20 s; one thread only."""
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    return k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=min_interval, max_retries=12,
                                 sleep=lambda x: time.sleep(min(x, 20.0))))


def _pages(api, path: str, params: dict[str, Any], key: str, *, tag: str | None = SNAP,
           max_pages: int = 200) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor = None
    for _ in range(max_pages):
        page = api.client.get(path, {**params, "cursor": cursor}, tag=tag)
        got = page.payload.get(key) or []
        out.extend(got)
        cursor = page.payload.get("cursor")
        if not cursor or not got:
            break
    return out


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


# ---- step 1: series ------------------------------------------------------------------------------
EVENT_SCHEMA = {"series_ticker": pl.String, "event_ticker": pl.String, "title": pl.String, "sub_title": pl.String,
                "category": pl.String, "mutually_exclusive": pl.Boolean, "strike_date": pl.Datetime("us", "UTC"),
                "strike_period": pl.String}


def event_rows(events: list[dict[str, Any]], series: str) -> list[dict[str, Any]]:
    return [{"series_ticker": e.get("series_ticker") or series, "event_ticker": e.get("event_ticker"),
             "title": e.get("title"), "sub_title": e.get("sub_title"), "category": e.get("category"),
             "mutually_exclusive": e.get("mutually_exclusive"), "strike_date": parse_ts(e.get("strike_date")),
             "strike_period": e.get("strike_period")} for e in events]


def candidate_series(series: pl.DataFrame, table_counts: pl.DataFrame | None = None) -> pl.DataFrame:
    """Non-financial series outside the excluded categories that are periodic by their ``frequency``
    or that settled >= 4 events in the shared markets table (catches mislabelled recurring series)."""
    from albert.categories import kalshi_series_is_financial

    fin = [kalshi_series_is_financial(a, b, c) for a, b, c in series.select("series_ticker", "category", "title").iter_rows()]
    s = series.filter(~pl.Series(fin, dtype=pl.Boolean) & ~pl.col("category").is_in(EXCLUDE_CATEGORIES))
    if table_counts is None:
        table_counts = pl.DataFrame(schema={"series_ticker": pl.String, "table_events": pl.UInt32})
    s = s.join(table_counts, on="series_ticker", how="left").with_columns(pl.col("table_events").fill_null(0))
    return s.filter(pl.col("frequency").is_in(PERIODIC) | (pl.col("table_events") >= 4)).sort("series_ticker")


def table_event_counts(tables: Path = TABLES) -> pl.DataFrame:
    return (pl.scan_parquet(tables / "kalshi_markets.parquet").filter(pl.col("result").is_in(["yes", "no"]))
            .group_by("series_ticker").agg(pl.col("event_ticker").n_unique().cast(pl.UInt32).alias("table_events")).collect())


def run_series(s) -> pl.DataFrame:
    from albert.venues.kalshi import series_frame

    api = _client(s)
    allser = series_frame(api.series().payload)
    cand = candidate_series(allser, table_event_counts())
    print(f"{allser.height} series, {cand.height} candidates", flush=True)
    path = OUT / "events_all.parquet"
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=EVENT_SCHEMA)
    done = set(have["series_ticker"].unique().to_list())
    rows: list[dict[str, Any]] = []
    for i, ser in enumerate(cand["series_ticker"].to_list(), 1):
        if ser in done:
            continue
        evs = _pages(api, "/events", {"series_ticker": ser, "status": "settled", "limit": 200}, "events")
        rows.extend(event_rows(evs, ser) or [{"series_ticker": ser, "event_ticker": None}])
        if i % 50 == 0:
            print(f"  events {i}/{cand.height}", flush=True)
            _save(pl.concat([have, pl.DataFrame(rows, schema=EVENT_SCHEMA)]), path)
    ev = pl.concat([have, pl.DataFrame(rows, schema=EVENT_SCHEMA)]).unique(["series_ticker", "event_ticker"])
    _save(ev, path)
    summ = series_summary(cand, ev)
    _save(summ, OUT / "series.parquet")
    return summ


_DATE_RX = re.compile(r"(?<=[-A-Z])(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})")
_MONTHS = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def ticker_date(event_ticker: str | None) -> datetime | None:
    """Date encoded in the event part of the ticker (``KXNETFLIXRANKSHOW-26NOV09`` -> 2026-11-09,
    ``KXYTVIEWSW-TAY26NOV08`` -> 2026-11-08), else None."""
    m = _DATE_RX.search("-" + (event_ticker or "").partition("-")[2])
    if not m:
        return None
    yy, mon, dd = m.groups()
    try:
        return datetime(2000 + int(yy), _MONTHS[mon], int(dd), tzinfo=UTC)
    except ValueError:
        return None


def series_summary(cand: pl.DataFrame, ev: pl.DataFrame) -> pl.DataFrame:
    ev = ev.filter(pl.col("event_ticker").is_not_null()).with_columns(
        pl.coalesce(pl.col("event_ticker").map_elements(ticker_date, return_dtype=pl.Datetime("us", "UTC")),
                    "strike_date").alias("edate"))
    g = ev.group_by("series_ticker").agg(
        pl.len().alias("settled_events"), pl.col("edate").min().alias("first"), pl.col("edate").max().alias("last"),
        (pl.col("edate") >= HOLDOUT_FROM).sum().alias("events_holdout"))
    out = cand.select("series_ticker", "category", "frequency", "title", "fee_type", "fee_multiplier", "table_events").join(
        g, on="series_ticker", how="left").with_columns(pl.col("settled_events").fill_null(0))
    span_w = ((pl.col("last") - pl.col("first")).dt.total_hours() / 24 / 7).clip(lower_bound=1 / 7)
    return out.with_columns((pl.col("settled_events") / span_w).round(2).alias("events_per_week"),
                            (pl.col("settled_events") >= MIN_EVENTS).alias("keep")).sort(
        "keep", "category", "settled_events", descending=[True, False, True])


# ---- step 2: markets -----------------------------------------------------------------------------
MARKET_SCHEMA = {
    "series_ticker": pl.String, "event_ticker": pl.String, "ticker": pl.String, "name": pl.String,
    "yes_sub_title": pl.String, "strike_type": pl.String, "floor_strike": pl.Float64, "cap_strike": pl.Float64,
    "open_time": pl.Datetime("us", "UTC"), "close_time": pl.Datetime("us", "UTC"),
    "expected_expiration_time": pl.Datetime("us", "UTC"), "settlement_ts": pl.Datetime("us", "UTC"),
    "result": pl.String, "expiration_value": pl.String, "volume": pl.Float64, "market_type": pl.String,
    "source": pl.String,
}


def market_name(m: dict[str, Any]) -> str | None:
    """The outcome a leg names: the ``custom_strike`` value (e.g. the show) or the yes sub-title."""
    cs = m.get("custom_strike")
    if isinstance(cs, dict) and cs:
        vals = [str(v) for v in cs.values() if v not in (None, "")]
        if vals:
            return " | ".join(vals)
    return m.get("yes_sub_title") or m.get("subtitle")


def market_rows(markets: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    rows = []
    for m in markets:
        if m.get("mve_collection_ticker") or m.get("mve_selected_legs"):
            continue
        ev = m.get("event_ticker") or ""
        rows.append({
            "series_ticker": ev.split("-", 1)[0], "event_ticker": ev, "ticker": m.get("ticker"),
            "name": market_name(m), "yes_sub_title": m.get("yes_sub_title"), "strike_type": m.get("strike_type"),
            "floor_strike": num(m.get("floor_strike")), "cap_strike": num(m.get("cap_strike")),
            "open_time": parse_ts(m.get("open_time")), "close_time": parse_ts(m.get("close_time")),
            "expected_expiration_time": parse_ts(m.get("expected_expiration_time")),
            "settlement_ts": parse_ts(m.get("settlement_ts")), "result": m.get("result") or None,
            "expiration_value": m.get("expiration_value") or None, "volume": num(_field(m, "volume")),
            "market_type": m.get("market_type"), "source": source,
        })
    return rows


def run_markets(s) -> pl.DataFrame:
    summ = pl.read_parquet(OUT / "series.parquet").filter("keep")
    api = _client(s)
    path = OUT / "markets.parquet"
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=MARKET_SCHEMA)
    done = set(have["series_ticker"].unique().to_list())
    rows: list[dict[str, Any]] = []
    todo = [x for x in summ["series_ticker"].to_list() if x not in done]
    for i, ser in enumerate(todo, 1):
        hist = _pages(api, "/historical/markets", {"series_ticker": ser, "limit": 1000}, "markets")
        live = _pages(api, "/markets", {"series_ticker": ser, "status": "settled", "limit": 1000, "mve_filter": "exclude"},
                      "markets")
        rows.extend(market_rows(hist, "historical") + market_rows(live, "live"))
        if i % 20 == 0:
            print(f"  markets {i}/{len(todo)} series, {len(rows)} rows", flush=True)
            _save(pl.concat([have, pl.DataFrame(rows, schema=MARKET_SCHEMA)]), path)
    m = pl.concat([have, pl.DataFrame(rows, schema=MARKET_SCHEMA)]).unique("ticker", keep="last")
    _save(m, path)
    return m


# ---- step 3: events, strands, own-history features -----------------------------------------------
LADDER_TYPES = ("greater", "greater_or_equal", "less", "less_or_equal", "between")
_STRAND_RX = re.compile(r"\d{2}(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}(?:\d{4}(?=$|[A-Z-]))?")
_NUM_RX = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?")


def strand_of(event_ticker: str) -> str:
    """Series + the event ticker without its date. Series such as Billboard chart positions list one
    event per album each week (``-26SEP26YOU``) and YouTube views one per artist (``-TAY26NOV08``):
    each subject is its own recurring question. A single-question series
    (``KXNETFLIXRANKSHOW-26NOV09``) is one strand."""
    ser, _, rest = event_ticker.partition("-")
    rest = _STRAND_RX.sub("", rest, count=1)              # the date (and an hour code: ``26NOV171000``)
    return f"{ser}|{rest}" if rest else ser


_SCALE = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}
_NUM_SFX_RX = re.compile(r"(-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?)\s*([kKmMbBtT])?(?![a-zA-Z])")


def parse_value(text: str | None) -> float | None:
    """First number in an ``expiration_value`` (``"1,234"``, ``"45.2%"``, ``"$2.10"``, ``"301.3K"``,
    ``"14.87 M"``), else None."""
    if not text:
        return None
    m = _NUM_SFX_RX.search(str(text).replace("$", ""))
    if not m:
        return None
    try:
        v = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    return v * _SCALE[m.group(2).lower()] if m.group(2) else v


def resolve_value(legs: list[tuple[str | None, float | None, float | None, int]], text: str | None) -> tuple[float | None, str]:
    """The event's settled number, consistent with how its numeric legs resolved. The
    ``expiration_value`` is used when every leg agrees with it (also in thousands / millions:
    some series quote strikes in millions and the value in units); otherwise the value is implied
    by the legs (midpoint of the bracket they pin down). Returns (value, source)."""
    num = [(st, lo, hi, y) for st, lo, hi, y in legs if st in LADDER_TYPES]
    if not num:
        return None, "none"
    v = parse_value(text)
    if v is not None:
        for f in (1.0, 1e-6, 1e-3, 1e3, 1e6):
            if all(leg_outcome(st, lo, hi, v * f) == y for st, lo, hi, y in num):
                return v * f, "value"
    lo_b, hi_b = -math.inf, math.inf
    for st, lo, hi, y in num:
        if st in ("greater", "greater_or_equal") and lo is not None:
            lo_b, hi_b = (max(lo_b, lo), hi_b) if y else (lo_b, min(hi_b, lo))
        elif st in ("less", "less_or_equal") and hi is not None:
            lo_b, hi_b = (lo_b, min(hi_b, hi)) if y else (max(lo_b, hi), hi_b)
        elif st == "between" and lo is not None and hi is not None and y:
            lo_b, hi_b = max(lo_b, lo), min(hi_b, hi)
    ks = sorted({k for _, lo, hi, _ in num for k in (lo, hi) if k is not None})
    step = float(np.median(np.diff(ks))) if len(ks) > 1 else 1.0
    if math.isfinite(lo_b) and math.isfinite(hi_b) and lo_b <= hi_b:
        return (lo_b + hi_b) / 2, "implied"
    if math.isfinite(lo_b):
        return lo_b + step / 2, "implied_open"
    if math.isfinite(hi_b):
        return hi_b - step / 2, "implied_open"
    return None, "none"


_UUID_RX = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_STOP = frozenset({"the", "a", "an", "of", "and", "feat", "ft", "featuring", "with"})


def norm_name(name: str | None) -> str | None:
    if name is None:
        return None
    s = re.sub(r"[\W_]+", " ", _UUID_RX.sub(" ", name.lower()).replace("'", "").replace("\u2019", "")).strip()
    return s or None


def leg_key(ticker: str, event_ticker: str, name: str | None) -> str:
    """Identity of a leg across events: its ticker suffix and the content words of its name
    (``"DEA§death pastors season 1 wife"``). Titles drift between weeks ("Death of the Pastors
    Wife: Season 1" / "Death of a pastors Wife"), so legs are matched with :func:`same_leg`."""
    sfx = ticker[len(event_ticker) + 1:] if ticker.startswith(event_ticker + "-") else ticker
    toks = sorted(set((norm_name(name) or "").split()) - _STOP)
    return f"{sfx}\u00a7{' '.join(toks)}"


@lru_cache(maxsize=1_000_000)
def same_leg(a: str, b: str) -> bool:
    """Same outcome in two events: equal content words, or the same ticker suffix with at least half
    the words shared (Jaccard), or 80% of the words shared. Name-less legs (chart positions keyed by
    ids) match on the suffix."""
    if a == b:
        return True
    sa, _, ta = a.partition("\u00a7")
    sb, _, tb = b.partition("\u00a7")
    wa, wb = set(ta.split()), set(tb.split())
    if not wa and not wb:
        return sa == sb
    j = len(wa & wb) / len(wa | wb)
    return j >= 0.8 or (sa == sb and j >= 0.5)


def matches(key: str, keys: list[str]) -> bool:
    return any(same_leg(key, k) for k in keys)


def leg_outcome(strike_type: str | None, floor: float | None, cap: float | None, v: float | None) -> int | None:
    """YES/NO of a numeric leg at value ``v`` (Kalshi: greater = above floor, less = below cap,
    between = floor <= v <= cap)."""
    if v is None or strike_type is None:
        return None
    if strike_type == "greater" and floor is not None:
        return int(v > floor)
    if strike_type == "greater_or_equal" and floor is not None:
        return int(v >= floor)
    if strike_type == "less" and cap is not None:
        return int(v < cap)
    if strike_type == "less_or_equal" and cap is not None:
        return int(v <= cap)
    if strike_type == "between" and floor is not None and cap is not None:
        return int(floor <= v <= cap)
    return None


def effective_strike() -> pl.Expr:
    """``structured`` legs with only a floor (artist / album stream counts) resolve like ``greater``
    (checked on the settled values: see ``strike_check``)."""
    st = pl.col("strike_type")
    return (pl.when((st == "structured") & pl.col("floor_strike").is_not_null() & pl.col("cap_strike").is_null())
            .then(pl.lit("greater")).otherwise(st))


def strike_check(m: pl.DataFrame) -> pl.DataFrame:
    """Per series: share of numeric legs whose result matches ``leg_outcome`` at the settled value."""
    x = m.filter(pl.col("strike_type").is_in(LADDER_TYPES)).with_columns(
        pl.col("expiration_value").map_elements(parse_value, return_dtype=pl.Float64).alias("v"))
    ok = [leg_outcome(st, lo, hi, v) == y if v is not None else None
          for st, lo, hi, v, y in x.select("strike_type", "floor_strike", "cap_strike", "v", "y").iter_rows()]
    return (x.with_columns(pl.Series("ok", ok, dtype=pl.Boolean)).group_by("series_ticker")
            .agg(pl.len().alias("legs"), pl.col("ok").mean().alias("agree")).sort("agree"))


def load_markets() -> pl.DataFrame:
    m = pl.read_parquet(OUT / "markets.parquet").filter(pl.col("result").is_in(["yes", "no"]))
    ser = pl.read_parquet(OUT / "series.parquet").select("series_ticker", "category", "frequency", "fee_type",
                                                         "fee_multiplier")
    ev = pl.read_parquet(OUT / "events_all.parquet").select("event_ticker", "mutually_exclusive").unique("event_ticker")
    return (m.join(ser, on="series_ticker", how="left").join(ev, on="event_ticker", how="left")
            .with_columns(effective_strike().alias("strike_type"),
                          pl.col("event_ticker").map_elements(strand_of, return_dtype=pl.String).alias("strand"),
                          (pl.col("result") == "yes").cast(pl.Int8).alias("y"),
                          pl.struct("ticker", "event_ticker", "name").map_elements(
                              lambda r: leg_key(r["ticker"], r["event_ticker"], r["name"]), return_dtype=pl.String).alias("key")))


def event_table(m: pl.DataFrame) -> pl.DataFrame:
    """One row per settled event: times, the winner(s) and the numeric value, ordered within strand."""
    e = m.group_by("strand", "event_ticker").agg(
        pl.col("series_ticker").first(), pl.col("category").first(),
        pl.col("open_time").min().alias("open"), pl.col("close_time").max().alias("close"),
        pl.col("close_time").min().alias("close_min"),
        pl.col("expected_expiration_time").max().alias("exp"), pl.col("settlement_ts").max().alias("known"),
        pl.len().alias("legs"), pl.col("y").sum().alias("n_yes"),
        pl.col("key").filter(pl.col("y") == 1).alias("winners"), pl.col("key").alias("listed"),
        pl.col("expiration_value").drop_nulls().mode().first().alias("expiration_value"),
        pl.col("strike_type").is_in(LADDER_TYPES).mean().alias("ladder_share"),
        pl.col("mutually_exclusive").first(),
        pl.struct("strike_type", "floor_strike", "cap_strike", "y").alias("_legs"))
    vals = [resolve_value([(x["strike_type"], x["floor_strike"], x["cap_strike"], x["y"]) for x in legs], text)
            for legs, text in e.select("_legs", "expiration_value").iter_rows()]
    e = e.drop("_legs").with_columns(pl.Series("value", [v for v, _ in vals], dtype=pl.Float64),
                                     pl.Series("value_src", [s for _, s in vals], dtype=pl.String),
                                     pl.coalesce("exp", "close").alias("period_end"))
    return e.sort("strand", "period_end", "event_ticker").with_columns(
        pl.int_range(pl.len()).over("strand").alias("pos"))


def strand_kinds(e: pl.DataFrame) -> pl.DataFrame:
    """``ladder`` (numeric strikes with a parsed value), ``cat`` (mutually exclusive named legs, one
    winner) or ``multi`` (independent named legs or a single binary)."""
    k = e.group_by("strand").agg(
        pl.col("series_ticker").first(), pl.col("category").first(), pl.len().alias("events"),
        pl.col("ladder_share").mean(), pl.col("value").is_not_null().mean().alias("value_share"),
        pl.col("mutually_exclusive").fill_null(False).mean().alias("me_share"), pl.col("legs").mean().alias("legs"),
        (pl.col("n_yes") == 1).mean().alias("one_winner"),
        pl.col("period_end").min().alias("first"), pl.col("period_end").max().alias("last"),
        (pl.col("period_end") >= HOLDOUT_FROM).sum().alias("events_holdout"))
    kind = (pl.when((pl.col("ladder_share") >= 0.8) & (pl.col("value_share") >= 0.8)).then(pl.lit("ladder"))
            .when((pl.col("me_share") >= 0.5) & (pl.col("legs") >= 2) & (pl.col("one_winner") >= 0.5)).then(pl.lit("cat"))
            .otherwise(pl.lit("multi")))
    return k.with_columns(kind.alias("kind"))


def decision_times(open_: datetime, close: datetime) -> dict[str, datetime]:
    """Open + 1 h, mid-life and close - 24 h (only if it is at least an hour after open + 1 h)."""
    out = {"open1h": open_ + timedelta(hours=1)}
    mid = open_ + (close - open_) / 2
    if mid > out["open1h"]:
        out["mid"] = mid
    late = close - timedelta(hours=24)
    if late > out["open1h"] + timedelta(hours=1):
        out["final24h"] = late
    return out


def _streak_name(hist: list[dict[str, Any]], i: int, key: str) -> int:
    s = 0
    while i >= 0 and matches(key, hist[i]["winners"]):
        s += 1
        i -= 1
    return s


def _streak_side(hist: list[dict[str, Any]], i: int, st: str, lo: float | None, hi: float | None, side: int) -> int:
    s = 0
    while i >= 0 and leg_outcome(st, lo, hi, hist[i]["value"]) == side:
        s += 1
        i -= 1
    return s


def leg_panel(m: pl.DataFrame, e: pl.DataFrame, kinds: pl.DataFrame) -> pl.DataFrame:
    """One row per (leg, decision time) with the own-history state known at that time.

    ``prev`` is the latest event of the strand with an earlier period whose outcome was known
    (all its legs settled) before the decision; ``gap`` counts periods between it and this event.
    ``carry`` is what this leg would resolve to if the previous outcome repeated: for named legs 1 =
    previous winner, 0 = listed last time and lost, null = new; for numeric legs the leg evaluated at
    the previous value. ``streak`` counts consecutive earlier periods (ending at ``prev``) with that
    same outcome. Ladder legs also carry the last value and the strand's past changes over ``gap``
    periods (both known at the decision), from which the study builds its model."""
    kinds_d = {r["strand"]: r["kind"] for r in kinds.iter_rows(named=True)}
    legs_by_ev: dict[str, list[dict[str, Any]]] = {}
    for r in m.select("event_ticker", "ticker", "key", "strike_type", "floor_strike", "cap_strike", "y",
                      "close_time").iter_rows(named=True):
        legs_by_ev.setdefault(r["event_ticker"], []).append(r)
    rows: list[dict[str, Any]] = []
    for (strand,), g in e.group_by(["strand"], maintain_order=True):
        kind = kinds_d.get(strand)
        hist = g.sort("pos").to_dicts()
        # Decision times hang on the *scheduled* close: the expected expiration (fixed at listing)
        # minus the strand's typical close-to-expiration offset. The actual close moves earlier when
        # an outcome is decided early, and timing decisions off it would peek at the result.
        offs = [(h["exp"] - h["close"]).total_seconds() for h in hist if h["exp"] is not None and h["close"] is not None]
        off = timedelta(seconds=float(np.median(offs))) if offs else timedelta(0)
        for j, ev in enumerate(hist):
            if ev["open"] is None or ev["close"] is None:
                continue
            sched = ev["exp"] - off if ev["exp"] is not None else ev["close"]
            for tname, tau in decision_times(ev["open"], sched).items():
                prev = next((i for i in range(j - 1, -1, -1) if hist[i]["known"] is not None and hist[i]["known"] < tau), None)
                if prev is None:
                    continue
                gap = j - prev
                base = {"strand": strand, "kind": kind, "event_ticker": ev["event_ticker"], "series_ticker": ev["series_ticker"],
                        "category": ev["category"], "dt": tname, "t": tau, "pos": j, "gap": gap,
                        "prev_event": hist[prev]["event_ticker"], "close": ev["close"], "settle": ev["known"],
                        "period_end": ev["period_end"], "n_known": prev + 1}
                if kind == "ladder":
                    v_last = hist[prev]["value"]
                    if v_last is None:
                        continue
                    # past changes over `gap` periods whose both ends were known before tau
                    ch = [(hist[i]["value"], hist[i - gap]["value"]) for i in range(gap, prev + 1)
                          if hist[i]["value"] is not None and hist[i - gap]["value"] is not None
                          and hist[i]["known"] is not None and hist[i]["known"] < tau][-HIST_MAX:]
                    lv = np.array([a - b for a, b in ch])
                    lg = np.array([math.log(a / b) for a, b in ch if a > 0 and b > 0])
                    legs = [x for x in legs_by_ev.get(ev["event_ticker"], []) if x["close_time"] is not None and x["close_time"] > tau]
                    for leg in sorted(legs, key=lambda x: strike_distance(x, v_last))[:NEAR_LEGS]:
                        st, lo, hi = leg["strike_type"], leg["floor_strike"], leg["cap_strike"]
                        carry = leg_outcome(st, lo, hi, v_last)
                        rows.append({**base, "close": leg["close_time"], "ticker": leg["ticker"], "key": leg["key"], "y": leg["y"],
                                     "carry": carry,
                                     "streak": _streak_side(hist, prev, st, lo, hi, carry) if carry is not None else 0,
                                     "strike_type": st, "floor": lo, "cap": hi, "v_last": v_last,
                                     "dist": strike_distance(leg, v_last), "n_hist": len(lv),
                                     "p_lvl": change_prob(st, lo, hi, v_last + lv),
                                     "p_log": change_prob(st, lo, hi, v_last * np.exp(lg)) if v_last > 0 else None,
                                     "incumbent": None})
                else:
                    winners, listed = hist[prev]["winners"], hist[prev]["listed"]
                    for leg in legs_by_ev.get(ev["event_ticker"], []):
                        if leg["close_time"] is None or leg["close_time"] <= tau:
                            continue
                        key = leg["key"]
                        carry = 1 if matches(key, winners) else (0 if matches(key, listed) else None)
                        rows.append({**base, "close": leg["close_time"], "ticker": leg["ticker"], "key": key, "y": leg["y"],
                                     "carry": carry,
                                     "streak": _streak_name(hist, prev, key) if carry == 1 else 0,
                                     "strike_type": leg["strike_type"], "floor": None, "cap": None, "v_last": None,
                                     "dist": None, "n_hist": prev, "p_lvl": None, "p_log": None,
                                     "incumbent": carry == 1})
    return pl.DataFrame(rows, schema=LEG_SCHEMA)


HIST_MAX = 60        # past changes a ladder model uses (most recent)
NEAR_LEGS = 6        # ladder legs nearest the last value kept per event
LEG_SCHEMA = {
    "strand": pl.String, "kind": pl.String, "event_ticker": pl.String, "series_ticker": pl.String, "category": pl.String,
    "dt": pl.String, "t": pl.Datetime("us", "UTC"), "pos": pl.Int64, "gap": pl.Int64, "prev_event": pl.String,
    "close": pl.Datetime("us", "UTC"), "settle": pl.Datetime("us", "UTC"), "period_end": pl.Datetime("us", "UTC"),
    "n_known": pl.Int64, "ticker": pl.String, "key": pl.String, "y": pl.Int8, "carry": pl.Int8, "streak": pl.Int64,
    "strike_type": pl.String, "floor": pl.Float64, "cap": pl.Float64, "v_last": pl.Float64, "dist": pl.Float64,
    "n_hist": pl.Int64, "p_lvl": pl.Float64, "p_log": pl.Float64, "incumbent": pl.Boolean,
}


def strike_distance(leg: dict[str, Any], v: float) -> float:
    lo, hi = leg.get("floor_strike"), leg.get("cap_strike")
    if lo is not None and hi is not None:
        return 0.0 if lo <= v <= hi else min(abs(v - lo), abs(v - hi))
    k = lo if lo is not None else hi
    return abs(v - k) if k is not None else math.inf


def change_prob(st: str | None, lo: float | None, hi: float | None, sims: np.ndarray, *,
                min_n: int = 5) -> float | None:
    """Share of ``sims`` (the last value moved by each past change) on which the leg is YES,
    with a half-count prior: (k + 0.5) / (n + 1). None with fewer than ``min_n`` past changes."""
    n = len(sims)
    if n < min_n or st is None:
        return None
    if st == "greater" and lo is not None:
        k = (sims > lo).sum()
    elif st == "greater_or_equal" and lo is not None:
        k = (sims >= lo).sum()
    elif st == "less" and hi is not None:
        k = (sims < hi).sum()
    elif st == "less_or_equal" and hi is not None:
        k = (sims <= hi).sum()
    elif st == "between" and lo is not None and hi is not None:
        k = ((sims >= lo) & (sims <= hi)).sum()
    else:
        return None
    return float((k + 0.5) / (n + 1))


MAX_GAP = 3


def streak_bucket(carry: pl.Expr, streak: pl.Expr) -> pl.Expr:
    """Own-history state of a named leg: new, lost last time, or won the last 1 / 2 / 3-4 / 5+."""
    return (pl.when(carry.is_null()).then(pl.lit("new")).when(carry == 0).then(pl.lit("lost"))
            .when(streak <= 1).then(pl.lit("w1")).when(streak == 2).then(pl.lit("w2"))
            .when(streak <= 4).then(pl.lit("w3_4")).otherwise(pl.lit("w5+")))


def transitions(m: pl.DataFrame, e: pl.DataFrame, kinds: pl.DataFrame) -> pl.DataFrame:
    """Outcome history the named-leg model learns from: for every event j and gap g <= ``MAX_GAP``,
    each leg's state relative to event j - g (``carry``, ``streak``) and its result, stamped with the
    time both were known (event j's settlement)."""
    named = set(kinds.filter(pl.col("kind") != "ladder")["strand"].to_list())
    kd = {r["strand"]: r["kind"] for r in kinds.iter_rows(named=True)}
    legs_by_ev: dict[str, list[tuple[str, int]]] = {}
    for ev, key, y in m.filter(pl.col("strand").is_in(list(named))).select("event_ticker", "key", "y").iter_rows():
        legs_by_ev.setdefault(ev, []).append((key, y))
    rows = []
    for (strand,), g in e.filter(pl.col("strand").is_in(list(named))).group_by(["strand"], maintain_order=True):
        hist = g.sort("pos").to_dicts()
        for j in range(1, len(hist)):
            for gap in range(1, min(MAX_GAP, j) + 1):
                prev = j - gap
                winners, listed = hist[prev]["winners"], hist[prev]["listed"]
                for key, y in legs_by_ev.get(hist[j]["event_ticker"], []):
                    carry = 1 if matches(key, winners) else (0 if matches(key, listed) else None)
                    rows.append({"strand": strand, "kind": kd[strand], "gap": gap, "known": hist[j]["known"],
                                 "carry": carry, "streak": _streak_name(hist, prev, key) if carry == 1 else 0, "y": y})
    t = pl.DataFrame(rows, schema={"strand": pl.String, "kind": pl.String, "gap": pl.Int64,
                                   "known": pl.Datetime("us", "UTC"), "carry": pl.Int8, "streak": pl.Int64, "y": pl.Int8})
    return t.with_columns(streak_bucket(pl.col("carry"), pl.col("streak")).alias("bucket"))


def run_features() -> pl.DataFrame:
    m = load_markets()
    e = event_table(m)
    k = strand_kinds(e)
    keep = k.filter(pl.col("events") >= MIN_EVENTS)
    _save(e, OUT / "events.parquet")
    _save(k, OUT / "strands.parquet")
    mk = m.filter(pl.col("strand").is_in(keep["strand"].implode()))
    ek = e.filter(pl.col("strand").is_in(keep["strand"].implode()))
    _save(transitions(mk, ek, keep), OUT / "transitions.parquet")
    p = leg_panel(mk, ek, keep).filter(pl.col("gap") <= MAX_GAP)
    _save(p, OUT / "legs.parquet")
    return p


# ---- step 4: candles ------------------------------------------------------------------------------
MAX_BATCH_CANDLES = 10000    # API: candles summed over tickers per batch request
MAX_BATCH_TICKERS = 50
MAX_PERIODS = 5000           # API: periods per market per request
CANDLES = OUT / "candles"


def wanted_legs(p: pl.DataFrame) -> pl.DataFrame:
    """Legs whose prices the study needs, with a fetch priority (historical candles cost one
    request per market at <= 1 request/s, so the budget goes to the core question first):

    1. incumbents of named strands (``carry`` = 1) settling in 2026;
    2. the two ladder legs nearest the last value (2026), and multi-winner legs that won last time;
    3. the other near ladder legs and multi legs with a known ``carry`` (2026);
    4. the same before 2026 (incumbents first).
    Live (post-cutoff) markets are fetched in batches regardless."""
    near = pl.col("dist").rank("ordinal").over("event_ticker", "dt")
    p = p.with_columns(near.alias("near_rank"))
    tier = (pl.when((pl.col("kind") == "cat") & (pl.col("carry") == 1)).then(1)
            .when((pl.col("kind") == "ladder") & (pl.col("near_rank") <= 2)).then(2)
            .when((pl.col("kind") == "multi") & (pl.col("carry") == 1)).then(2)
            .when(pl.col("kind") == "ladder").then(3)
            .when((pl.col("kind") == "multi") & pl.col("carry").is_not_null()).then(3)
            .otherwise(None))
    old = pl.col("period_end") < datetime(2026, 1, 1, tzinfo=UTC)
    p = p.with_columns(pl.when(old).then(tier + 3).otherwise(tier).alias("prio")).filter(pl.col("prio").is_not_null())
    return (p.group_by("ticker").agg(pl.col("event_ticker").first(), pl.col("strand").first(), pl.col("prio").min(),
                                     pl.col("period_end").max()).sort("prio", pl.col("period_end"), descending=[False, True]))


def pack_live(rows: list[dict[str, Any]], period: int = 60) -> list[list[dict[str, Any]]]:
    """Greedy batches of live markets in time order within the API caps (span x tickers)."""
    jobs: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    lo = hi = None
    for r in sorted(rows, key=lambda x: (x["lo"], x["ticker"])):
        nlo = r["lo"] if lo is None else min(lo, r["lo"])
        nhi = r["hi"] if hi is None else max(hi, r["hi"])
        span = math.ceil((nhi - nlo).total_seconds() / 60 / period) + 1
        if cur and (span * (len(cur) + 1) > MAX_BATCH_CANDLES or len(cur) >= MAX_BATCH_TICKERS):
            jobs.append(cur)
            cur, nlo, nhi = [], r["lo"], r["hi"]
        cur.append(r)
        lo, hi = nlo, nhi
    if cur:
        jobs.append(cur)
    return jobs


def run_candles(s, *, max_requests: int | None = None) -> None:
    from albert.research import xcat

    CANDLES.mkdir(parents=True, exist_ok=True)
    p = pl.read_parquet(OUT / "legs.parquet")
    want = wanted_legs(p)
    m = pl.read_parquet(OUT / "markets.parquet").select("ticker", "event_ticker", "open_time", "close_time", "settlement_ts")
    w = want.join(m.drop("event_ticker"), on="ticker").filter(pl.col("open_time").is_not_null() & pl.col("close_time").is_not_null())
    w = w.with_columns(pl.col("open_time").alias("lo"),
                       pl.min_horizontal("close_time", pl.col("open_time") + timedelta(hours=MAX_PERIODS - 1)).alias("hi"),
                       pl.lit(60).alias("period"))
    done_path = CANDLES / "done.parquet"
    done = set(pl.read_parquet(done_path)["ticker"].to_list()) if done_path.exists() else set()
    # reuse the xcat hourly layer (same window: the market's whole life)
    xd = xcat.OUT / "hourly" / "done.parquet"
    xdone = set(pl.read_parquet(xd)["ticker"].to_list()) if xd.exists() else set()
    reuse = [t for t in w["ticker"].to_list() if t in xdone and t not in done]
    if reuse:
        xc = (pl.scan_parquet(str(xcat.OUT / "hourly" / "part-*.parquet")).filter(pl.col("ticker").is_in(reuse))
              .collect())
        part = len(list(CANDLES.glob("part-*.parquet")))
        xc.write_parquet(CANDLES / f"part-{part:05d}.parquet")
        done |= set(reuse)
        _save(pl.DataFrame({"ticker": sorted(done)}), done_path)
        print(f"reused {len(reuse)} markets from xcat ({xc.height} candles)", flush=True)
    todo = w.filter(~pl.col("ticker").is_in(list(done)))
    api = _client(s)
    cutoff = api.cutoff()["market_settled_ts"]
    hist = todo.filter(pl.col("settlement_ts") < cutoff).sort("prio", "period_end", descending=[False, True]).to_dicts()
    live = todo.filter(pl.col("settlement_ts") >= cutoff).to_dicts()
    jobs = [{"kind": "batch", "rows": b} for b in pack_live(live)] + [{"kind": "hist", "rows": [r]} for r in hist]
    if max_requests is not None:
        jobs = jobs[:max_requests]
    print(f"candles: {w.height} legs wanted, {len(done)} done, {len(hist)} historical + {len(live)} live "
          f"-> {len(jobs)} requests (cutoff {cutoff:%Y-%m-%d})", flush=True)
    buf: list[dict[str, Any]] = []
    fin: list[str] = []
    part = len(list(CANDLES.glob("part-*.parquet")))
    t0 = time.time()
    for n, job in enumerate(jobs, 1):
        for tk, got in xcat._run_job(api, job):
            if got is not None:
                buf.extend(got)
                fin.append(tk)
        if n % 200 == 0 or n == len(jobs):
            if buf:
                pl.DataFrame(buf, schema=xcat.OHLC_SCHEMA).write_parquet(CANDLES / f"part-{part:05d}.parquet")
                part += 1
            buf = []
            done |= set(fin)
            _save(pl.DataFrame({"ticker": sorted(done)}), done_path)
            rate = n / (time.time() - t0)
            print(f"  {n}/{len(jobs)} requests, {len(done)} markets, {rate:.2f} req/s, "
                  f"eta {(len(jobs) - n) / max(rate, 1e-9) / 60:.0f} min", flush=True)


def load_candles(tickers: list[str] | None = None) -> pl.DataFrame:
    """The fetched hourly candles (columns the study uses), one part at a time to bound memory."""
    parts = sorted(CANDLES.glob("part-*.parquet"))
    if not parts:
        raise RuntimeError("no candles fetched")
    cols = ["ticker", "end_ts", "bid_close", "ask_close", "p_close", "p_low", "p_high", "volume"]
    keep = pl.Series(tickers, dtype=pl.String) if tickers is not None else None
    frames = []
    for path in parts:
        f = pl.read_parquet(path, columns=cols)
        frames.append(f.filter(pl.col("ticker").is_in(keep.implode())) if keep is not None else f)
    return pl.concat(frames).unique(["ticker", "end_ts"]).sort("ticker", "end_ts")


# ---- step 5: study --------------------------------------------------------------------------------
STAKE = 4.5                  # $ per order (n = floor(STAKE / price)); + fee stays within EVENT_CAP
BANKROLL = 100.0
EVENT_CAP = 5.0              # open cost per event in the capped book
K_GRID = (1, 3, 10, 30, 100)  # shrinkage strength of a strand's own rate toward the pooled rate
THETAS = (0.02, 0.05, 0.10)  # taker: required edge after fee
MARGINS = (0.03, 0.06, 0.10)  # maker: quote distance from the model
DTS = ("open1h", "mid", "final24h")
KINDS = ("cat", "multi", "ladder")
MIN_DEV_EVENTS = 30
TAKER = 0.07
MAKER = 0.0175


def asof_counts(rows: pl.DataFrame, trans: pl.DataFrame, keys: list[str], prefix: str) -> pl.DataFrame:
    """Cumulative transitions (n, YES count k) with the same ``keys`` known strictly before ``t``."""
    agg = (trans.group_by([*keys, "known"]).agg(pl.len().alias("n"), pl.col("y").cast(pl.Int64).sum().alias("k"))
           .sort("known").with_columns(pl.col("n").cum_sum().over(keys).alias(f"{prefix}_n"),
                                       pl.col("k").cum_sum().over(keys).alias(f"{prefix}_k")))
    out = rows.sort("t").join_asof(agg.select([*keys, "known", f"{prefix}_n", f"{prefix}_k"]), left_on="t",
                                   right_on="known", by=keys, strategy="backward", allow_exact_matches=False,
                                   check_sortedness=False)
    return out.drop("known").with_columns(pl.col(f"{prefix}_n").fill_null(0), pl.col(f"{prefix}_k").fill_null(0))


def named_probs(rows: pl.DataFrame, trans: pl.DataFrame, ks: tuple[int, ...] = K_GRID) -> pl.DataFrame:
    """Model for named legs: P(YES | own-history state) = (k_s + K pi) / (n_s + K), where (k_s, n_s)
    count this strand's past legs in the same state (gap, bucket) and pi is the pooled rate of all
    strands of the same kind in that state, both from outcomes known before ``t``. One column
    ``p_k{K}`` per shrinkage strength."""
    r = rows.with_columns(streak_bucket(pl.col("carry"), pl.col("streak")).alias("bucket"))
    r = asof_counts(r, trans, ["strand", "gap", "bucket"], "s")
    r = asof_counts(r, trans, ["kind", "gap", "bucket"], "g")
    pi = (pl.col("g_k") + 0.5) / (pl.col("g_n") + 1)
    return r.with_columns(pi.alias("p_pool"), *[((pl.col("s_k") + k * pi) / (pl.col("s_n") + k)).alias(f"p_k{k}") for k in ks])


def log_loss(p: pl.Expr, y: pl.Expr) -> pl.Expr:
    q = p.clip(0.005, 0.995)
    return -(y * q.log() + (1 - y) * (1 - q).log())


def choose_k(r: pl.DataFrame, before: datetime, ks: tuple[int, ...] = K_GRID) -> int:
    """Shrinkage strength with the lowest log loss on named rows settled before ``before``."""
    d = r.filter(pl.col("settle") < before)
    if d.is_empty():
        return 10
    ll = {k: d.select(log_loss(pl.col(f"p_k{k}"), pl.col("y")).mean()).item() for k in ks}
    return min(ll, key=lambda k: ll[k])


def choose_transform(r: pl.DataFrame, before: datetime) -> dict[str, str]:
    """Per ladder strand: level or log changes, whichever had the lower log loss on rows settled
    before ``before``; strands without such rows take the majority choice."""
    d = r.filter((pl.col("settle") < before) & pl.col("p_lvl").is_not_null() & pl.col("p_log").is_not_null())
    g = d.group_by("strand").agg(log_loss(pl.col("p_lvl"), pl.col("y")).mean().alias("lvl"),
                                 log_loss(pl.col("p_log"), pl.col("y")).mean().alias("log"))
    out = {s: ("log" if b <= a else "lvl") for s, a, b in g.select("strand", "lvl", "log").iter_rows()}
    major = "log" if sum(v == "log" for v in out.values()) >= len(out) / 2 else "lvl"
    for s in r["strand"].unique().to_list():
        out.setdefault(s, major)
    return out


def all_probs(panel: pl.DataFrame, trans: pl.DataFrame) -> pl.DataFrame:
    """Every candidate model column for every panel row (``p_k*`` for named legs, ``p_lvl`` /
    ``p_log`` for ladder legs); :func:`apply_hyper` picks one."""
    named = named_probs(panel.filter(pl.col("kind") != "ladder"), trans)
    lad = panel.filter(pl.col("kind") == "ladder").with_columns(pl.lit(None, dtype=pl.String).alias("bucket"))
    return pl.concat([named, lad], how="diagonal_relaxed")


def apply_hyper(r: pl.DataFrame, before: datetime) -> tuple[pl.DataFrame, dict[str, Any]]:
    """``p`` with hyper-parameters (K for named legs, level / log per ladder strand) chosen on rows
    settled before ``before`` only."""
    named = r.filter(pl.col("kind") != "ladder")
    lad = r.filter(pl.col("kind") == "ladder")
    k = choose_k(named, before)
    tf = choose_transform(lad, before)
    named = named.with_columns(pl.col(f"p_k{k}").alias("p"), pl.lit(f"K={k}").alias("hyper"))
    lad = lad.with_columns(pl.col("strand").replace_strict(tf, default="log").alias("hyper"))
    lad = lad.with_columns(pl.when(pl.col("hyper") == "log").then(pl.coalesce("p_log", "p_lvl"))
                           .otherwise(pl.col("p_lvl")).alias("p"))
    out = pl.concat([named, lad], how="diagonal_relaxed").filter(pl.col("p").is_not_null())
    return out, {"K": k, "log_strands": sum(v == "log" for v in tf.values()), "level_strands": sum(v == "lvl" for v in tf.values())}


def model_probs(panel: pl.DataFrame, trans: pl.DataFrame, before: datetime) -> pl.DataFrame:
    return apply_hyper(all_probs(panel, trans), before)[0]


def attach_quotes(rows: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    """The last hourly candle that ended at or before ``t`` (Kalshi omits unchanged hours, so the
    quote carries forward): yes bid / ask at its close, mid and spread when two-sided."""
    c = (candles.select("ticker", pl.col("end_ts").alias("q_ts"), pl.col("bid_close").alias("bid"),
                        pl.col("ask_close").alias("ask"), pl.col("p_close").alias("last")).sort("ticker", "q_ts"))
    out = rows.sort("t").join_asof(c, left_on="t", right_on="q_ts", by="ticker", strategy="backward",
                                   check_sortedness=False)
    two = (pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("ask") > pl.col("bid"))
    return out.filter(pl.col("q_ts").is_not_null()).with_columns(
        pl.when(two).then((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
        pl.when(two).then(pl.col("ask") - pl.col("bid")).alias("spread"))


def fee_order(coef: float, n: pl.Expr, price: pl.Expr) -> pl.Expr:
    """Kalshi fee for one order of n contracts, rounded up to the cent."""
    raw = coef * pl.col("fee_multiplier").fill_null(1.0) * n * price * (1 - price)
    return (raw * 100).round(9).ceil() / 100


def taker_trades(q: pl.DataFrame, theta: float) -> pl.DataFrame:
    """Buy YES at the ask when p - ask - fee >= theta, or NO at 1 - bid when bid - p - fee >= theta;
    n = floor(STAKE / price) contracts, fee rounded up per order, held to settlement."""
    ny = (STAKE / pl.col("ask")).floor().clip(lower_bound=1)
    nn = (STAKE / (1 - pl.col("bid"))).floor().clip(lower_bound=1)
    f = q.with_columns(ny.alias("ny"), nn.alias("nn")).with_columns(
        (pl.col("p") - pl.col("ask") - fee_order(TAKER, pl.col("ny"), pl.col("ask")) / pl.col("ny")).alias("ey"),
        (pl.col("bid") - pl.col("p") - fee_order(TAKER, pl.col("nn"), 1 - pl.col("bid")) / pl.col("nn")).alias("en"))
    okay = (pl.col("ask") > 0) & (pl.col("ask") < 1)
    okn = (pl.col("bid") > 0) & (pl.col("bid") < 1)
    f = f.with_columns(pl.when(okay & (pl.col("ey") >= theta)).then(1)
                       .when(okn & (pl.col("en") >= theta)).then(-1).otherwise(0).alias("dir")).filter(pl.col("dir") != 0)
    f = f.with_columns(
        pl.when(pl.col("dir") == 1).then(pl.col("ask")).otherwise(1 - pl.col("bid")).alias("price"),
        pl.when(pl.col("dir") == 1).then(pl.col("ny")).otherwise(pl.col("nn")).alias("n"),
        pl.when(pl.col("dir") == 1).then(pl.col("y")).otherwise(1 - pl.col("y")).cast(pl.Float64).alias("payoff"),
        pl.col("t").alias("t_fill"))
    f = f.with_columns(fee_order(TAKER, pl.col("n"), pl.col("price")).alias("fee_order"))
    return _pnl(f)


def _pnl(f: pl.DataFrame) -> pl.DataFrame:
    side_mid = pl.when(pl.col("dir") == 1).then(pl.col("mid")).otherwise(1 - pl.col("mid"))
    return f.with_columns(
        (pl.col("payoff") - pl.col("price") - pl.col("fee_order") / pl.col("n")).alias("pnl"),
        (pl.col("payoff") - side_mid).alias("gross_mid"),
        (pl.col("n") * pl.col("price") + pl.col("fee_order")).alias("cost_usd"),
        pl.col("settle").dt.date().alias("sdate"))


def maker_orders(q: pl.DataFrame, margin: float) -> pl.DataFrame:
    """Resting bids at the model -+ margin: YES at floor(p - m) below the ask, NO at
    floor(1 - p - m) below the NO ask; non-crossing only (a crossing quote is the taker's)."""
    qy = ((pl.col("p") - margin) * 100).floor() / 100
    qn = ((1 - pl.col("p") - margin) * 100).floor() / 100
    base = q.with_row_index("oid0")
    yes = base.with_columns(qy.alias("price"), pl.lit(1).alias("dir")).filter(
        (pl.col("price") >= 0.01) & (pl.col("price") < pl.col("ask").fill_null(1.0)))
    no = base.with_columns(qn.alias("price"), pl.lit(-1).alias("dir")).filter(
        (pl.col("price") >= 0.01) & (pl.col("price") < 1 - pl.col("bid").fill_null(0.0)))
    return pl.concat([yes, no]).with_row_index("oid")


def maker_fills(orders: pl.DataFrame, candles: pl.DataFrame, chunk: int = 1000) -> pl.DataFrame:
    """An order fills at its price when a later candle (starting at or after ``t``, ending by the
    close) traded strictly through it: YES bid q when the candle's trade low < q; NO bid q when the
    YES trade high > 1 - q. Size: n = floor(STAKE / q), at most that candle's volume."""
    c = candles.select("ticker", "end_ts", "p_low", "p_high", "volume").filter(pl.col("p_low").is_not_null())
    out = []
    for i in range(0, orders.height, chunk):
        o = orders.slice(i, chunk)
        j = o.select("oid", "ticker", "t", "close", "dir", "price").join(c, on="ticker").filter(
            (pl.col("end_ts") >= pl.col("t") + timedelta(hours=1)) & (pl.col("end_ts") <= pl.col("close") + timedelta(hours=1))
            & pl.when(pl.col("dir") == 1).then(pl.col("p_low") < pl.col("price"))
            .otherwise(pl.col("p_high") > 1 - pl.col("price")))
        out.append(j.sort("end_ts").group_by("oid").first().select("oid", pl.col("end_ts").alias("t_fill"),
                                                                   pl.col("volume").alias("fill_vol")))
    fills = pl.concat(out) if out else pl.DataFrame(schema={"oid": pl.UInt32, "t_fill": pl.Datetime("us", "UTC"),
                                                            "fill_vol": pl.Float64})
    f = orders.join(fills, on="oid")
    n = pl.min_horizontal((STAKE / pl.col("price")).floor(), pl.col("fill_vol").fill_null(0).floor())
    maker = pl.col("fee_type").fill_null("") == "quadratic_with_maker_fees"
    f = f.with_columns(n.alias("n")).filter(pl.col("n") >= 1).with_columns(
        pl.when(pl.col("dir") == 1).then(pl.col("y")).otherwise(1 - pl.col("y")).cast(pl.Float64).alias("payoff"),
        pl.when(maker).then(fee_order(MAKER, pl.col("n"), pl.col("price"))).otherwise(0.0).alias("fee_order"))
    return _pnl(f)


def wilson_lo(p: float, n: int, z: float = 1.96) -> float | None:
    if n <= 0:
        return None
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / (1 + z * z / n)


def score(f: pl.DataFrame, col: str = "pnl") -> dict[str, Any]:
    """Mean per contract (each trade weighted equally), t clustered by event and by settlement date,
    win rate with its Wilson lower bound (events as n) against the break-even price."""
    from albert.research.fundamental import clustered_mean

    if f.is_empty():
        return {"trades": 0, "events": 0}
    x = f[col].to_numpy()
    mu, te, ne = clustered_mean(x, f["event_ticker"].to_numpy())
    _, td, nd = clustered_mean(x, f["sdate"].cast(pl.String).to_numpy())
    win = float(f["payoff"].mean())
    be = float((f["price"] + f["fee_order"] / f["n"]).mean())
    r = lambda v, k=2: None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(v, k)
    return {"trades": f.height, "events": ne, "dates": nd, "contracts": int(f["n"].sum()),
            "c_per_contract": r(100 * mu), "t_event": r(te), "t_date": r(td),
            "gross_vs_mid_c": r(100 * float(f["gross_mid"].drop_nulls().mean() or 0.0)) if f["gross_mid"].drop_nulls().len() else None,
            "pnl_usd": r(float((f["n"] * f["pnl"]).sum())), "win": r(win, 3), "breakeven": r(be, 3),
            "win_lo95_events": r(wilson_lo(win, ne), 3), "mean_price": r(float(f["price"].mean()), 3)}


def split_of(t: pl.Expr) -> pl.Expr:
    return pl.when(t < HOLDOUT_FROM).then(pl.lit("dev")).otherwise(pl.lit("holdout"))


# ---- study driver ---------------------------------------------------------------------------------
def load_study() -> tuple[pl.DataFrame, pl.DataFrame]:
    """All rows with every model column and the quote at ``t``; candles of the quoted legs."""
    panel = pl.read_parquet(OUT / "legs.parquet")
    trans = pl.read_parquet(OUT / "transitions.parquet")
    r = all_probs(panel, trans)
    fees = pl.read_parquet(OUT / "series.parquet").select("series_ticker", "fee_type", "fee_multiplier")
    r = r.join(fees, on="series_ticker", how="left").with_columns(
        split_of(pl.col("t")).alias("split"), pl.col("dist").rank("ordinal").over("event_ticker", "dt").alias("near_rank"))
    candles = load_candles(r["ticker"].unique().to_list())
    return r, candles


def in_universe() -> pl.Expr:
    """Rows the strategies trade (the same set at every date, fetched first): legs that won the
    last known period (named strands) and the two ladder legs nearest the last value."""
    return (pl.when(pl.col("kind") == "ladder").then(pl.col("near_rank") <= 2)
            .otherwise(pl.col("carry") == 1))


def strategies(q: pl.DataFrame, candles: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Trades of every grid cell: kind x decision time x (taker theta | maker margin)."""
    out = {}
    for th in THETAS:
        out[f"taker_{th:.2f}"] = taker_trades(q, th)
    for mg in MARGINS:
        out[f"maker_{mg:.2f}"] = maker_fills(maker_orders(q, mg), candles)
    return out


def grid_table(trades: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows = []
    for name, f in trades.items():
        mode, par = name.split("_")
        for split in ("dev", "holdout"):
            fs = f.filter(pl.col("split") == split)
            for kind in (*KINDS, "all"):
                fk = fs if kind == "all" else fs.filter(pl.col("kind") == kind)
                for dt in DTS:
                    rows.append({"mode": mode, "param": float(par), "kind": kind, "dt": dt, "split": split,
                                 **score(fk.filter(pl.col("dt") == dt))})
    return pl.DataFrame(rows, infer_schema_length=None)


def pick_primary(grid: pl.DataFrame) -> dict[str, Any]:
    """Pre-specified on dev only: among cells with >= MIN_DEV_EVENTS dev events, the highest
    min(t by event, t by date); ties by mean. Written to ``primary.json`` before the holdout is read."""
    d = grid.filter((pl.col("split") == "dev") & (pl.col("events") >= MIN_DEV_EVENTS)).with_columns(
        pl.min_horizontal("t_event", "t_date").alias("t_min"))
    best = d.sort(["t_min", "c_per_contract"], descending=True, nulls_last=True).row(0, named=True)
    return {k: best[k] for k in ("mode", "param", "kind", "dt", "trades", "events", "c_per_contract", "t_event", "t_date")}


def rule_trades(trades: dict[str, pl.DataFrame], rule: dict[str, Any]) -> pl.DataFrame:
    f = trades[f"{rule['mode']}_{rule['param']:.2f}"].filter(pl.col("dt") == rule["dt"])
    return f if rule["kind"] == "all" else f.filter(pl.col("kind") == rule["kind"])


def persistence_table(q: pl.DataFrame) -> pl.DataFrame:
    """Does the last outcome carry over more or less often than the price says? Per kind x decision
    time x own-history state: realised YES rate, mean mid and model, y - mid with clustered t."""
    from albert.research.fundamental import clustered_mean

    f = q.filter(pl.col("mid").is_not_null()).with_columns(
        pl.when(pl.col("kind") == "ladder").then(pl.format("carry{}", pl.col("carry"))).otherwise(pl.col("bucket")).alias("state"),
        (pl.col("y") - pl.col("mid")).alias("res"), pl.col("settle").dt.date().cast(pl.String).alias("sdate"))
    rows = []
    for (split, kind, dt, state), g in f.group_by(["split", "kind", "dt", "state"], maintain_order=True):
        mu, te, ne = clustered_mean(g["res"].to_numpy(), g["event_ticker"].to_numpy())
        _, td, _ = clustered_mean(g["res"].to_numpy(), g["sdate"].to_numpy())
        rows.append({"split": split, "kind": kind, "dt": dt, "state": state, "rows": g.height, "events": ne,
                     "yes_rate": round(float(g["y"].mean()), 3), "mid": round(float(g["mid"].mean()), 3),
                     "model": round(float(g["p"].mean()), 3), "y_minus_mid_c": round(100 * mu, 2),
                     "t_event": None if te is None else round(te, 2), "t_date": None if td is None else round(td, 2),
                     "spread_c": round(100 * float(g["spread"].mean()), 1)})
    return pl.DataFrame(rows).sort("split", "kind", "dt", "state")


def band_table(q: pl.DataFrame) -> pl.DataFrame:
    """Own-history state x spread band: y - mid and a taker on the side the state favours (YES on
    incumbents / legs the last value would win, NO on the rest), at the executable price with the
    fee rounded up per order. Shows whether any mispricing survives where the mid is a price."""
    from albert.research.fundamental import clustered_mean

    f = q.filter(pl.col("mid").is_not_null() & pl.col("carry").is_not_null()).with_columns(
        pl.when(pl.col("spread") <= 0.03).then(pl.lit("a <=3c")).when(pl.col("spread") <= 0.06).then(pl.lit("b 4-6c"))
        .when(pl.col("spread") <= 0.15).then(pl.lit("c 7-15c")).otherwise(pl.lit("d >15c")).alias("band"),
        pl.when(pl.col("carry") == 1).then(pl.col("ask")).otherwise(1 - pl.col("bid")).alias("price"),
        pl.when(pl.col("carry") == 1).then(pl.col("y")).otherwise(1 - pl.col("y")).cast(pl.Float64).alias("payoff"))
    f = f.filter((pl.col("price") > 0) & (pl.col("price") < 1)).with_columns(
        (STAKE / pl.col("price")).floor().clip(lower_bound=1).alias("n"))
    f = f.with_columns((pl.col("payoff") - pl.col("price") - fee_order(TAKER, pl.col("n"), pl.col("price")) / pl.col("n")).alias("pnl"),
                       (pl.col("y") - pl.col("mid")).alias("res"), pl.col("settle").dt.date().cast(pl.String).alias("sdate"))
    rows = []
    for (split, kind, carry, band), g in f.group_by(["split", "kind", "carry", "band"], maintain_order=True):
        mr, tr, ne = clustered_mean(g["res"].to_numpy(), g["event_ticker"].to_numpy())
        mp, te, _ = clustered_mean(g["pnl"].to_numpy(), g["event_ticker"].to_numpy())
        _, td, _ = clustered_mean(g["pnl"].to_numpy(), g["sdate"].to_numpy())
        rows.append({"split": split, "kind": kind, "carry": carry, "band": band, "rows": g.height, "events": ne,
                     "y_minus_mid_c": round(100 * mr, 2), "t_res": None if tr is None else round(tr, 2),
                     "taker_c": round(100 * mp, 2), "t_event": None if te is None else round(te, 2),
                     "t_date": None if td is None else round(td, 2)})
    return pl.DataFrame(rows).sort("split", "kind", "carry", "band")


def blend_table(q: pl.DataFrame) -> pl.DataFrame:
    """Logistic blend logit P = a + b logit(mid) + c logit(model) per split / kind / decision time
    (and in books <= 10c wide): c > 0 with |t| > 2 means own history adds what the price lacks."""
    from albert.research.fundamental import fit_logistic, logit

    rows = []
    f = q.filter(pl.col("mid").is_not_null())
    for (split, kind, dt), g in f.group_by(["split", "kind", "dt"], maintain_order=True):
        for band, h in (("all", g), ("spread<=10c", g.filter(pl.col("spread") <= 0.10))):
            if h.height < 30 or h["y"].n_unique() < 2:
                continue
            X = np.column_stack([np.ones(h.height), logit(h["mid"].to_numpy()), logit(h["p"].to_numpy())])
            beta, t = fit_logistic(X, h["y"].to_numpy().astype(float), h["event_ticker"].to_numpy())
            y = h["y"].to_numpy()
            bm = float(((h["mid"].to_numpy() - y) ** 2).mean())
            bp = float(((h["p"].to_numpy() - y) ** 2).mean())
            rows.append({"split": split, "kind": kind, "dt": dt, "band": band, "rows": h.height,
                         "events": h["event_ticker"].n_unique(), "b_mid": round(float(beta[1]), 3),
                         "c_model": round(float(beta[2]), 3), "t_c": round(float(t[2]), 2),
                         "brier_mid": round(bm, 4), "brier_model": round(bp, 4)})
    return pl.DataFrame(rows).sort("split", "kind", "dt", "band")


def walk_forward(r: pl.DataFrame, candles: pl.DataFrame, rule: dict[str, Any], start: datetime) -> pl.DataFrame:
    """Monthly walk-forward of the primary rule's family (same kind, decision time and mode): each
    month's K / transform and theta (or margin) are refit on rows settled before the month starts."""
    months = pl.datetime_range(start, END, interval="1mo", eager=True, time_zone="UTC").to_list()
    out = []
    for m0 in months:
        m1 = (m0 + timedelta(days=32)).replace(day=1)
        rr, hyper = apply_hyper(r, m0)
        rr = rr.filter((pl.col("dt") == rule["dt"]) & in_universe())
        if rule["kind"] != "all":
            rr = rr.filter(pl.col("kind") == rule["kind"])
        q = attach_quotes(rr.filter(pl.col("t") < m1), candles)
        grid = THETAS if rule["mode"] == "taker" else MARGINS
        past = q.filter(pl.col("settle") < m0)
        best, best_t = None, -math.inf
        for par in grid:
            f = taker_trades(past, par) if rule["mode"] == "taker" else maker_fills(maker_orders(past, par), candles)
            sc = score(f)
            tt = min(sc.get("t_event") or -math.inf, sc.get("t_date") or -math.inf)
            if sc.get("events", 0) >= 10 and tt > best_t:
                best, best_t = par, tt
        if best is None:
            continue
        cur = q.filter((pl.col("t") >= m0) & (pl.col("t") < m1))
        f = taker_trades(cur, best) if rule["mode"] == "taker" else maker_fills(maker_orders(cur, best), candles)
        out.append(f.with_columns(pl.lit(m0.strftime("%Y-%m")).alias("fold"), pl.lit(best).alias("param"),
                                  pl.lit(json.dumps(hyper)).alias("hyper")))
    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()


def marks_from(candles: pl.DataFrame) -> pl.DataFrame:
    two = (pl.col("bid_close") > 0) & (pl.col("ask_close") < 1) & (pl.col("ask_close") > pl.col("bid_close"))
    mid = pl.when(two).then((pl.col("bid_close") + pl.col("ask_close")) / 2).otherwise(pl.col("p_close"))
    return candles.select("ticker", pl.col("end_ts").alias("ts"), mid.alias("yes_mid")).filter(pl.col("yes_mid").is_not_null())


def ledger_for(f: pl.DataFrame, name: str, split: str, candles: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Capped $100 book (<= EVENT_CAP open per event) through ``ta_calendar.capped_ledger``; Sharpe /
    vol from ``vol_metrics.ledger_metrics`` with positions marked daily at the hourly-candle mid."""
    from albert.research.ta_calendar import capped_ledger
    from albert.research.vol_metrics import ledger_metrics

    if f.is_empty():
        return pl.DataFrame(), {"positions": 0}
    g = f.select(pl.col("t_fill").alias("t"), "n", "price", "fee_order", "event_ticker", pl.col("settle").alias("settlement_ts"),
                 "ticker", "dir", "payoff")
    led = capped_ledger(g, NAME, name, split, bankroll=BANKROLL, event_cap=EVENT_CAP)
    if led.is_empty():
        return led, {"positions": 0}
    marks = marks_from(candles.filter(pl.col("ticker").is_in(led["ticker"].unique().implode())))
    met = ledger_metrics(led, marks)
    keep = ("positions", "events", "pnl_usd", "usd_per_week", "ann_return_pct", "ann_vol_pct", "sharpe", "sharpe_ci90",
            "p_sharpe_pos", "sharpe_ex_top5", "usd_per_week_ex_top5", "sharpe_weekly", "max_dd_usd", "min_cum_pnl_usd",
            "weeks_positive_pct", "skew", "first_day", "last_day", "marked_share")
    return led, {"strategy": name, "split": split, **{k: met.get(k) for k in keep}}


def run_study(phase: str) -> None:
    """``dev``: tables on dev rows and the pre-specified primary rule (``primary.json``).
    ``holdout``: every table on both splits, the frozen primary on holdout, walk-forward, ledgers."""
    r, candles = load_study()
    rr, hyper = apply_hyper(r, HOLDOUT_FROM)
    q = attach_quotes(rr, candles)
    if phase == "dev":
        q = q.filter(pl.col("split") == "dev")
    trades = strategies(q.filter(in_universe()), candles)
    grid = grid_table(trades)
    _save(grid, OUT / f"grid_{phase}.parquet")
    grid.write_csv(OUT / f"grid_{phase}.csv")
    persistence_table(q).write_csv(OUT / f"persistence_{phase}.csv")
    band_table(q.filter(in_universe() | (pl.col("kind") == "ladder"))).write_csv(OUT / f"bands_{phase}.csv")
    blend_table(q.filter(in_universe())).write_csv(OUT / f"blend_{phase}.csv")
    if phase == "dev":
        rule = {**pick_primary(grid), "hyper": hyper, "frozen_at": datetime.now(UTC).isoformat()}
        (OUT / "primary.json").write_text(json.dumps(rule, indent=1, default=str))
        print(json.dumps(rule, indent=1, default=str))
        return
    rule = json.loads((OUT / "primary.json").read_text())
    prim = rule_trades(trades, rule)
    res = {"rule": rule, "dev": score(prim.filter(pl.col("split") == "dev")),
           "holdout": score(prim.filter(pl.col("split") == "holdout"))}
    (OUT / "ledgers").mkdir(parents=True, exist_ok=True)
    metrics = []
    for split in ("dev", "holdout"):
        led, met = ledger_for(prim.filter(pl.col("split") == split), "primary", split, candles)
        if led.height:
            led.write_parquet(OUT / "ledgers" / f"ledger_primary_{split}.parquet")
        metrics.append(met)
    wf_start = datetime(2026, 1, 1, tzinfo=UTC)
    wf = walk_forward(r, candles, rule, wf_start)
    if wf.height:
        _save(wf, OUT / "walk_forward_trades.parquet")
        res["walk_forward"] = {"all": score(wf), "from_oct": score(wf.filter(pl.col("t") >= HOLDOUT_FROM)),
                               "by_month": [{"fold": k, "param": g["param"][0], **score(g)}
                                            for (k,), g in wf.group_by(["fold"], maintain_order=True)]}
        for split, f in (("walk_forward", wf), ("walk_forward_from_oct", wf.filter(pl.col("t") >= HOLDOUT_FROM))):
            led, met = ledger_for(f, "primary_wf", split, candles)
            if led.height:
                led.write_parquet(OUT / "ledgers" / f"ledger_primary_{split}.parquet")
            metrics.append(met)
    res["ledgers"] = metrics
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


# ---- report ---------------------------------------------------------------------------------------
METHOD = """
* **Series:** 417 candidates from `/series` (non-financial per `albert.categories`; Sports,
  Climate and Weather, Mentions and Elections excluded). A candidate is daily / weekly / monthly,
  or settled >= 4 events Aug 15 - Oct 31. Settled events are counted from `/events`; 143 series
  have >= 10. Markets come from `/historical/markets` and `/markets` by series (99k legs,
  10.7k events).
* **Strands:** a strand is a series plus its subject (one Billboard album, one YouTube artist).
  278 strands have >= 10 events, 9.4k events in all.
* **Kinds:**
  * cat: mutually exclusive named legs with one winner.
  * multi: independent named legs or a single binary.
  * ladder: numeric strikes. The settled value must agree with every leg's result (rescaled
    if the units differ), otherwise it is implied by the legs.
* **Leg identity:** titles drift ("Death of the Pastors Wife: Season 1" / "Death of a pastors
  Wife"), so legs match on the ticker suffix plus content-word Jaccard.
* **Own history at decision t:** taken from the latest strand event fully settled before t, with
  gap = periods back to it. Named legs get carry (won / lost / new last time), the streak of
  consecutive wins, and the strand's and pooled kind's past rates in the same (gap, streak
  bucket). Ladder legs get the leg's outcome at the last value, and the share of the last 60
  past gap-period changes that would make the leg YES (level or log changes).
* **Model:** named legs use (k_s + K pi) / (n_s + K) with K chosen on dev by log loss; ladders
  pick level vs log per strand on dev.
* **Decision times:** open + 1 h, mid-life, scheduled close - 24 h. The scheduled close is the
  expected expiration minus the strand's median close offset, never the actual (possibly early)
  close.
* **Quotes:** the last hourly candle that ended by t.
* **Execution:**
  * Taker: ask / bid, n = floor($4.50 / price), fee 0.07 x mult x n x P(1-P) rounded up per order.
  * Maker: model -+ 3/6/10c, non-crossing. A fill needs a later candle whose trade low / high
    went strictly through the quote, sized at most that candle's volume. Maker fee applies only
    on `quadratic_with_maker_fees` series. Held to settlement.
* **Prices fetched (<= 1 request/s, every response cached in `data/raw/kalshi`):**
  * Every live (post-2026-09-18) leg in the universe.
  * 2026 historical incumbents and the two ladder legs nearest the last value.
  * Pre-2026 events and the rest of 2026 enter only through outcomes (the model's history), not
    through prices.
  * 11.8k markets in all, 3.8k of them reused from `research/xcat`.
* **Statistics:** t clustered by event and by settlement date. Ledgers use the $100 capped book
  (<= $5 open per event) via `ta_calendar.capped_ledger` and `vol_metrics.ledger_metrics`, marked
  daily at the candle mid.
"""


VERDICT = """
**No edge.** Recurring Kalshi series do persist: a named winner repeats 44% of the time after one
win and 88% after five or more in a row; a numeric outcome lands in the bucket the last value would
win far more often than in any other. But the price already knows this wherever the book is tight
enough to trade. Own history adds information only at open + 1 h (blend c = 0.49, t 5.8 dev;
0.75, t 2.8 holdout). At that time named-leg books are 35-65c wide, so the information sits
inside the spread. That repeats the pattern of the earlier fundamental studies.

* **Primary rule (frozen on dev 2026-11-17 21:11 UTC, before the holdout was read):** taker,
  incumbent of a named series (previous winner), open + 1 h, trade when the model beats the
  ask / bid by >= 10c after the fee.
  * Dev: -0.79c per contract (t -0.36 by event, -0.33 by date), 284 events.
  * Holdout: -4.3c (t -0.65 / -0.61), 22 events.
  * Monthly walk-forward Jan-Nov 2026: -0.71c (t -0.34), 317 events.
  * Ledger Sharpe on the $100 capped book: 0.84 [-1.3, 2.4] dev, 2.2 [-5.6, 5.0] holdout,
    1.2 [-0.6, 2.7] walk-forward. Ex top-5 events: -1.7 / -8.7 / -1.3.
  * The positive $ totals are lottery skew (skew about 5). The rule mostly buys 15-20c
    contracts. One ActBlue incumbent bought at about 2c made +$220 of the holdout's +$138.
    Without the top five events, every split loses $6-14 per week. Walk-forward drawdown is
    -$190, i.e. ruin on $100.
* **The whole grid (72 cells):** the primary was the best dev cell. Maker quotes at model
  +- 3/6/10c lose 2-18c per fill in every cell of both splits: adverse selection, since fills come
  when the incumbent's fate has changed. Takers at mid-period and in the final 24 h lose 2.5-9c
  in both splits.
* **Exploratory (post-holdout, not a rule):** in books <= 3c wide, incumbents and
  "last-value" ladder legs were slightly overpriced on dev (y - mid -3.1c, taker -4.3c, t -2.6).
  They were underpriced in Oct-Nov (+5 to +8c, t 1.7-2.3). The sign flips, so it is not
  tradable. Persistence looks underpriced only in books wider than 15c, where the mid is not a
  price.

Nothing here is worth forward paper trading.
"""
def _md(df: pl.DataFrame, digits: int = 3) -> str:
    if df.is_empty():
        return "(none)\n"
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:.{digits}g}" if abs(v) < 1e5 else f"{v:.0f}")
            elif isinstance(v, datetime):
                cells.append(v.strftime("%Y-%m-%d"))
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def outcome_table(trans: pl.DataFrame) -> pl.DataFrame:
    """Own-history base rates without prices: P(YES) by kind x state, gap 1, dev vs holdout."""
    return (trans.filter(pl.col("gap") == 1).with_columns(split_of(pl.col("known")).alias("split"))
            .group_by("split", "kind", "bucket").agg(pl.len().alias("legs"), pl.col("y").mean().round(3).alias("yes_rate"))
            .sort("split", "kind", "bucket"))


def strand_summary() -> pl.DataFrame:
    st = pl.read_parquet(OUT / "strands.parquet").filter(pl.col("events") >= MIN_EVENTS)
    tr = pl.read_parquet(OUT / "transitions.parquet").filter((pl.col("gap") == 1) & (pl.col("carry") == 1))
    rep = tr.group_by("strand").agg(pl.len().alias("incumbent_legs"), pl.col("y").mean().round(3).alias("incumbent_repeat"))
    per = pl.read_parquet(OUT / "series.parquet").select("series_ticker", "frequency")
    return (st.join(rep, on="strand", how="left").join(per, on="series_ticker", how="left")
            .with_columns((pl.col("events") / ((pl.col("last") - pl.col("first")).dt.total_hours() / 24 / 7 + 1))
                          .round(2).alias("events_per_week"))
            .select("series_ticker", "strand", "category", "frequency", "kind", "events", "events_holdout", "events_per_week",
                    pl.col("first").dt.date(), pl.col("last").dt.date(), pl.col("legs").round(1), "incumbent_repeat")
            .sort("category", "series_ticker", "strand"))


def write_report(verdict: str = "") -> Path:
    summ = strand_summary()
    summ.write_csv(OUT / "series_list.csv")
    trans = pl.read_parquet(OUT / "transitions.parquet")
    res = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
    parts = ["# Recurring Kalshi series: does own history beat the price?\n",
             (f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `python -m albert.research.recurring`. "
             "Dev = decisions before 2026-10-01, holdout = 2026-10-01 .. 2026-11-17.\n")]
    if verdict:
        parts.append("## Verdict\n\n" + verdict.strip() + "\n")
    parts.append("## Data and method\n\n" + METHOD.strip() + "\n")
    by_cat = summ.group_by("category", "kind").agg(pl.col("series_ticker").n_unique().alias("series"), pl.len().alias("strands"),
                                                   pl.col("events").sum().alias("events"),
                                                   pl.col("events_holdout").sum().alias("events_holdout")).sort("category", "kind")
    parts += ["## Recurring strands (>= 10 settled events)\n", _md(by_cat), "\n<details><summary>All strands</summary>\n\n",
              _md(summ), "\n</details>\n"]
    parts += ["## Own-history base rates (outcomes only, consecutive periods)\n", _md(outcome_table(trans))]
    for name, title in (("persistence", "Price vs own history: y - mid by state"), ("blend", "Blend logit P = a + b logit(mid) + c logit(model)")):
        for ph in ("holdout",):
            p = OUT / f"{name}_{ph}.csv"
            if p.exists():
                parts += [f"## {title}\n", _md(pl.read_csv(p))]
    bp = OUT / "bands_holdout.csv"
    if bp.exists():
        parts += ["## Own-history state by spread band (y - mid, and a taker on the side the state favours)\n",
                  "Added after the holdout was opened (exploratory, not a pre-specified rule).\n\n", _md(pl.read_csv(bp))]
    gp = OUT / "grid_holdout.csv"
    if gp.exists():
        g = pl.read_csv(gp).select("mode", "param", "kind", "dt", "split", "trades", "events", "c_per_contract", "t_event",
                                   "t_date", "gross_vs_mid_c", "pnl_usd", "win", "breakeven")
        parts += ["## Strategy grid (per contract, after fees)\n", _md(g.filter(pl.col("events") >= 10))]
    if res:
        parts += ["## Primary rule (frozen on dev)\n", "```json\n" + json.dumps(res.get("rule"), indent=1, default=str) + "\n```\n",
                  _md(pl.DataFrame([{"split": k, **res[k]} for k in ("dev", "holdout") if k in res], infer_schema_length=None))]
        wf = res.get("walk_forward")
        if wf:
            parts += ["### Monthly walk-forward\n", _md(pl.DataFrame([{"fold": "all", **wf["all"]}, {"fold": "from Oct", **wf["from_oct"]},
                                                                     *wf["by_month"]], infer_schema_length=None))]
        led = [x for x in res.get("ledgers", []) if x.get("positions")]
        if led:
            led = [{**x, "sharpe_ci90": f"[{x['sharpe_ci90'][0]}, {x['sharpe_ci90'][1]}]" if x.get("sharpe_ci90") else ""}
                   for x in led]
            parts += ["### Capped-book ledgers ($100, <= $5 per event, daily MTM at the candle mid; vol_metrics)\n",
                      _md(pl.DataFrame(led, infer_schema_length=None))]
    path = OUT / "report.md"
    path.write_text("\n".join(parts))
    return path


def main(argv: list[str] | None = None) -> None:
    from albert.config import settings

    ap = argparse.ArgumentParser(prog="albert.research.recurring")
    ap.add_argument("step", choices=["series", "markets", "features", "candles", "study", "report", "all"])
    ap.add_argument("phase", nargs="?", choices=["dev", "holdout"], default="dev")
    ap.add_argument("--max-requests", type=int, default=None)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    s = settings()
    if a.step == "series":
        summ = run_series(s)
        print(json.dumps(summ.filter("keep").group_by("category").agg(pl.len().alias("series"),
              pl.col("settled_events").sum().alias("events")).to_dicts()))
    elif a.step == "markets":
        m = run_markets(s)
        print(f"{m.height} markets, {m['event_ticker'].n_unique()} events, {m['series_ticker'].n_unique()} series")
    elif a.step == "features":
        p = run_features()
        print(p.group_by("kind", "dt").agg(pl.len().alias("rows"), pl.col("event_ticker").n_unique().alias("events"))
              .sort("kind", "dt"))
    elif a.step == "candles":
        run_candles(s, max_requests=a.max_requests)
    elif a.step == "study":
        run_study(a.phase)
    elif a.step == "report":
        print(write_report(VERDICT))
    elif a.step == "all":
        run_series(s)
        run_markets(s)
        run_features()
        run_candles(s, max_requests=a.max_requests)
        run_study("dev")
        run_study("holdout")
        print(write_report(VERDICT))


if __name__ == "__main__":
    main()
