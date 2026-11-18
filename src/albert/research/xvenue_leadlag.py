"""Does Polymarket international lead Kalshi in non-sports markets, enough for a Kalshi taker?

Research only: no orders anywhere. Polymarket intl is a data source (the user is a US resident).

Universe (volume-blind)
-----------------------
* **Kalshi:** every settled binary market in a non-sports, non-financial category (Politics,
  Elections, Entertainment, Mentions, Science and Technology, World, Social, Health, ...; the
  ``categories.kalshi_series_is_financial`` filter removes money series filed elsewhere) that
  closed 2026-08-01 .. 2026-11-17. Aug 15 .. Oct 31 comes from the shared ``kalshi_markets`` table;
  Nov 1 .. 17 from ``/events?status=settled&min_close_ts`` (category is on the event) and
  ``/markets?event_ticker=``; Aug 1 .. 14 per series from ``/historical/markets?series_ticker=``
  for the series families that match below (selected on series identity, never on price/result).
* **Polymarket:** every closed Gamma event (``/events/keyset``) with an end date in
  [2026-07-25, 2026-11-17] under the tags politics, elections, world, geopolitics, pop-culture,
  tech, science, mention-markets and trump; the crypto/finance tags are dropped
  (``categories.PM_FINANCIAL_TAGS``).
* **Matching** (:func:`candidates`, :func:`verify`): candidate pairs from text similarity within a
  date window, grouped into families (Kalshi series x Polymarket event template). A family is
  accepted only after reading both venues' resolution text (``FAMILIES`` documents each decision);
  inside an accepted family a pair must also agree on every entity, number and date of the
  contract (:func:`pair_checks`). Settlement agreement is reported as a quality check, never used
  to select pairs.

Prices
------
* Polymarket: CLOB ``/prices-history?market=<token>&fidelity=1`` (about one point per minute; each
  point carries its own second-resolution stamp). The stamp is calibrated against our own
  recorded Polymarket WebSocket books (:func:`stamp_check`) before use.
* Kalshi: minute candles (``/historical/markets/{t}/candlesticks`` before the 2026-09-18 cutoff,
  batch ``/markets/candlesticks`` after). Candles are omitted when nothing changed, so quotes are
  forward-filled on a minute grid. A candle stamped ``end_ts`` closes at ``end_ts``.

Test (:func:`grid`, :func:`events_for`, :func:`trades_for`)
-----------------------------------------------------------
* Event: Polymarket mid moves >= X (3/5/10c) within N minutes (5/15/60) while Kalshi's mid moved
  < X/2 in that direction over the same window (one event per N minutes per pair).
* Trade: take Kalshi in Polymarket's direction at the ask (YES) or 1 - bid (NO) of the **next**
  minute after the signal minute (lag 1; lag 0 and 2 as sensitivity), ~$5 order, taker fee
  0.07 x mult x n x P(1-P) rounded up per order; exit by taking the bid / 1 - ask at 15 min / 1 h /
  6 h (fee again), or hold to settlement. Execution filter variant: only when the Kalshi spread at
  entry is <= X.
* Control (Kalshi leads Polymarket): the same event with the venues swapped; Polymarket's follow is
  measured and the Kalshi taker trades in the Kalshi move's direction.
* Dev = signals before 2026-10-01; holdout = 2026-10-01 .. 2026-11-17. One primary rule is frozen
  from dev (``primary.json``, :func:`select_primary`) before any holdout table is computed.
* t-statistics are two-way clustered by real-world event (Polymarket event) and ET date; Sharpe/vol
  from a capped $100 ledger through ``vol_metrics``.

    python -m albert.research.xvenue_leadlag universe | hist | match | stamp | fetch-pm | fetch-kalshi
                                             | study | select | report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("POLARS_MAX_THREADS", "2")    # shared 2-CPU box; also bounds parquet decode memory

import numpy as np
import polars as pl

from albert.categories import PM_FINANCIAL_TAGS, kalshi_series_is_financial
from albert.venues.kalshi import _field, num, parse_ts

OUT = Path("data/research/xvenue_leadlag")
TABLES = Path("data/tables")
START = datetime(2026, 8, 1, tzinfo=UTC)
END = datetime(2026, 11, 17, 23, 59, 59, tzinfo=UTC)
HOLDOUT = datetime(2026, 10, 1, tzinfo=UTC)
PM_TAGS = ("politics", "elections", "world", "geopolitics", "pop-culture", "tech", "science",
           "mention-markets", "trump")
PM_END_MIN = datetime(2026, 7, 25, tzinfo=UTC)
KALSHI_EXCLUDE_CATEGORIES = {"Sports", "Climate and Weather", "Financials", "Economics", "Crypto",
                             "Commodities", "Companies"}
KALSHI_INTERVAL = 1.0      # s between Kalshi requests (shared IP with the live bot)
PM_INTERVAL = 0.5          # s between Polymarket requests

PM_MARKET_SCHEMA = {
    "condition_id": pl.String, "question": pl.String, "slug": pl.String, "group_title": pl.String,
    "description": pl.String, "event_id": pl.String, "event_slug": pl.String,
    "event_title": pl.String, "event_description": pl.String, "tags": pl.List(pl.String),
    "start_date": pl.Datetime("us", "UTC"), "end_date": pl.Datetime("us", "UTC"),
    "closed_time": pl.Datetime("us", "UTC"), "outcomes": pl.List(pl.String),
    "token_ids": pl.List(pl.String), "outcome_prices": pl.List(pl.Float64),
    "volume": pl.Float64, "neg_risk": pl.Boolean, "resolution_source": pl.String,
}

KALSHI_MARKET_SCHEMA = {
    "ticker": pl.String, "event_ticker": pl.String, "series_ticker": pl.String,
    "category": pl.String, "title": pl.String, "yes_sub_title": pl.String,
    "event_title": pl.String, "market_type": pl.String, "open_time": pl.Datetime("us", "UTC"),
    "close_time": pl.Datetime("us", "UTC"), "expected_expiration_time": pl.Datetime("us", "UTC"),
    "settlement_ts": pl.Datetime("us", "UTC"), "result": pl.String, "rules_primary": pl.String,
    "rules_secondary": pl.String, "fee_multiplier": pl.Float64, "volume": pl.Float64,
    "source": pl.String,
}


# ---- clients -------------------------------------------------------------------------------------
def _clients(s=None):
    from albert.config import settings
    from albert.http import CachedClient
    from albert.research.pm_hazard import _api

    s = s or settings()
    gamma = CachedClient("pm_gamma", s.pm_gamma_base, s.raw_dir, min_interval=PM_INTERVAL, max_retries=8)
    clob = CachedClient("pm_clob", "https://clob.polymarket.com", s.raw_dir, min_interval=PM_INTERVAL,
                        max_retries=8)
    return _api(s, KALSHI_INTERVAL, nap_cap=20.0, retries=30), gamma, clob


# ---- Polymarket events ---------------------------------------------------------------------------
def _json_list(v: Any) -> list[Any]:
    if v is None:
        return []
    return v if isinstance(v, list) else json.loads(v)


def _loose_ts(v: Any) -> datetime | None:
    if not v:
        return None
    t = str(v).replace(" ", "T").replace("Z", "+00:00")
    if t.endswith("+00"):
        t += ":00"
    try:
        d = datetime.fromisoformat(t)
    except ValueError:
        return None
    return (d if d.tzinfo else d.replace(tzinfo=UTC)).astimezone(UTC)


def pm_market_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per binary market of each Gamma event, with both venues' text needed for matching."""
    rows = []
    for e in events:
        tags = [t.get("slug") for t in e.get("tags") or [] if t.get("slug")]
        for m in e.get("markets") or []:
            outcomes = [str(o) for o in _json_list(m.get("outcomes"))]
            tokens = [str(t) for t in _json_list(m.get("clobTokenIds"))]
            if not m.get("conditionId") or len(outcomes) != 2 or len(tokens) != 2:
                continue
            prices = [float(p) for p in _json_list(m.get("outcomePrices"))] or None
            rows.append({
                "condition_id": m.get("conditionId"), "question": m.get("question"), "slug": m.get("slug"),
                "group_title": m.get("groupItemTitle"), "description": m.get("description"),
                "event_id": str(e.get("id")), "event_slug": e.get("slug"), "event_title": e.get("title"),
                "event_description": e.get("description"), "tags": tags,
                "start_date": _loose_ts(m.get("startDate")), "end_date": _loose_ts(m.get("endDate")),
                "closed_time": _loose_ts(m.get("closedTime")), "outcomes": outcomes, "token_ids": tokens,
                "outcome_prices": prices, "volume": float(m.get("volumeNum") or m.get("volume") or 0),
                "neg_risk": bool(m.get("negRisk")), "resolution_source": m.get("resolutionSource") or None,
            })
    return rows


def fetch_pm_events(gamma, tags: tuple[str, ...] = PM_TAGS) -> pl.DataFrame:
    """Closed events per tag (cached pages), flattened to binary markets; financial tags dropped."""
    seen: dict[str, dict[str, Any]] = {}
    for tag in tags:
        cur, pages = None, 0
        while True:
            p = gamma.get("/events/keyset", {"closed": "true", "limit": 100, "after_cursor": cur,
                                             "end_date_min": PM_END_MIN.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                             "end_date_max": END.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                             "tag_slug": tag})
            ev = p.payload.get("events") or []
            for e in ev:
                seen[str(e.get("id"))] = e
            pages += 1
            cur = p.payload.get("next_cursor")
            if not cur or not ev:
                break
        print(f"pm tag {tag}: {pages} pages, {len(seen)} events so far", flush=True)
    rows = pm_market_rows(list(seen.values()))
    df = pl.DataFrame(rows, schema=PM_MARKET_SCHEMA)
    return df.filter(~pl.col("tags").list.eval(pl.element().is_in(list(PM_FINANCIAL_TAGS))).list.any())


# ---- Kalshi markets ------------------------------------------------------------------------------
def kalshi_market_rows(payload: dict[str, Any], event_info: dict[str, dict[str, Any]],
                       series_info: dict[str, dict[str, Any]], source: str) -> list[dict[str, Any]]:
    rows = []
    for m in payload.get("markets") or []:
        if m.get("mve_collection_ticker") or m.get("mve_selected_legs"):
            continue
        ev = m.get("event_ticker") or ""
        ser = ev.split("-", 1)[0]
        info = series_info.get(ser, {})
        rows.append({
            "ticker": m["ticker"], "event_ticker": ev, "series_ticker": ser,
            "category": event_info.get(ev, {}).get("category") or info.get("category"),
            "title": m.get("title"), "yes_sub_title": m.get("yes_sub_title"),
            "event_title": event_info.get(ev, {}).get("title"), "market_type": m.get("market_type"),
            "open_time": parse_ts(m.get("open_time")), "close_time": parse_ts(m.get("close_time")),
            "expected_expiration_time": parse_ts(m.get("expected_expiration_time")),
            "settlement_ts": parse_ts(m.get("settlement_ts")), "result": m.get("result") or None,
            "rules_primary": m.get("rules_primary"), "rules_secondary": m.get("rules_secondary"),
            "fee_multiplier": info.get("fee_multiplier"), "volume": num(_field(m, "volume")),
            "source": source,
        })
    return rows


def _series_info() -> dict[str, dict[str, Any]]:
    s = pl.read_parquet(TABLES / "kalshi_series.parquet")
    return {r["series_ticker"]: r for r in s.iter_rows(named=True)}


def _keep_kalshi(df: pl.DataFrame, series_info: dict[str, dict[str, Any]]) -> pl.DataFrame:
    fin = {s for s in df["series_ticker"].unique().to_list()
           if kalshi_series_is_financial(s, (series_info.get(s) or {}).get("category"),
                                         (series_info.get(s) or {}).get("title"))}
    return df.filter(~pl.col("category").is_in(list(KALSHI_EXCLUDE_CATEGORIES)) & pl.col("category").is_not_null()
                     & ~pl.col("series_ticker").is_in(list(fin)) & (pl.col("market_type") == "binary")
                     & pl.col("result").is_in(["yes", "no"]))


def kalshi_shared(series_info: dict[str, dict[str, Any]]) -> pl.DataFrame:
    """Aug 15 .. Oct 31 from the shared table (category from the series list)."""
    m = pl.scan_parquet(TABLES / "kalshi_markets.parquet").filter(
        ~pl.col("category").is_in(list(KALSHI_EXCLUDE_CATEGORIES))).collect()
    m = m.with_columns(pl.lit(None, pl.String).alias("rules_secondary"), pl.lit(None, pl.String).alias("event_title"),
                       pl.lit("shared").alias("source"))
    return _keep_kalshi(m.select(list(KALSHI_MARKET_SCHEMA)).cast(KALSHI_MARKET_SCHEMA), series_info)


def fetch_kalshi_recent(api, series_info: dict[str, dict[str, Any]], since: datetime) -> pl.DataFrame:
    """Non-sports markets closing in [since, END]: the settled-events listing (it carries the category)
    names the series, then one ``/markets?series_ticker=&min_close_ts=&max_close_ts=`` query per series
    (no status filter, so closed-but-unsettled markets are seen; unsettled ones drop out below).
    ``/markets`` ignores a comma-separated ``event_ticker`` list (it returns nothing)."""
    events: dict[str, dict[str, Any]] = {}
    cur = None
    tag = "2026-11-17"
    while True:
        p = api.client.get("/events", {"status": "settled", "min_close_ts": int(since.timestamp()), "limit": 200,
                                       "cursor": cur}, tag=tag)
        for e in p.payload.get("events") or []:
            events[e["event_ticker"]] = e
        cur = p.payload.get("cursor")
        if not cur or not p.payload.get("events"):
            break
    keep = {v["series_ticker"] for v in events.values() if v.get("category") not in KALSHI_EXCLUDE_CATEGORIES}
    print(f"kalshi events closing after {since:%Y-%m-%d}: {len(events)}, non-sports/non-financial series {len(keep)}",
          flush=True)
    rows: list[dict[str, Any]] = []
    for i, s in enumerate(sorted(keep)):
        for p in api._cursor_pages("/markets", {"series_ticker": s, "limit": 1000, "mve_filter": "exclude",
                                                "min_close_ts": int(since.timestamp()),
                                                "max_close_ts": int(END.timestamp())}, "markets"):
            rows += kalshi_market_rows(p.payload, events, series_info, "series")
        if i % 100 == 0:
            print(f"  series {i}/{len(keep)}: {len(rows)} markets", flush=True)
    df = pl.DataFrame(rows, schema=KALSHI_MARKET_SCHEMA)
    return _keep_kalshi(df.filter(pl.col("close_time") >= since), series_info)


def fetch_kalshi_series_hist(api, series: list[str], series_info: dict[str, dict[str, Any]],
                             lo: datetime, hi: datetime) -> pl.DataFrame:
    """Historical (pre-cutoff) settled markets of the given series closing in [lo, hi)."""
    rows: list[dict[str, Any]] = []
    for s in series:
        for p in api._cursor_pages("/historical/markets", {"series_ticker": s, "limit": 1000}, "markets"):
            rows += kalshi_market_rows(p.payload, {}, series_info, "historical")
    df = pl.DataFrame(rows, schema=KALSHI_MARKET_SCHEMA)
    return _keep_kalshi(df.filter(pl.col("close_time").is_between(lo, hi, closed="left")), series_info)


# ---- Polymarket history stamp check --------------------------------------------------------------
def pm_stream_quotes(path: Path, assets: set[str] | None = None, top: int = 0) -> pl.DataFrame:
    """(asset, ts_ms, bid, ask) from one recorded Polymarket hour: ``price_change`` items carry the
    new best bid/ask and the server ``timestamp`` (ms). With ``top`` > 0 and no ``assets``, keep the
    ``top`` most active assets."""
    import io

    import zstandard

    rows: list[tuple[str, int, float, float]] = []
    with path.open("rb") as f:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8")
        for line in text:
            if 'price_change' not in line:
                continue
            frame = json.loads(json.loads(line)["frame"])
            for it in frame if isinstance(frame, list) else [frame]:
                if it.get("event_type") != "price_change":
                    continue
                ts = int(it.get("timestamp") or 0)
                for pc in it.get("price_changes") or []:
                    a = pc.get("asset_id")
                    if assets is not None and a not in assets:
                        continue
                    b, k = pc.get("best_bid"), pc.get("best_ask")
                    if b is None or k is None:
                        continue
                    rows.append((a, ts, float(b), float(k)))
    df = pl.DataFrame(rows, schema={"asset": pl.String, "ts_ms": pl.Int64, "bid": pl.Float64, "ask": pl.Float64},
                      orient="row")
    if assets is None and top:
        keep = df.group_by("asset").len().sort("len", descending=True).head(top)["asset"]
        df = df.filter(pl.col("asset").is_in(keep.implode()))
    return df.sort("asset", "ts_ms")


def pm_history_rows(payload: dict[str, Any], token: str) -> list[dict[str, Any]]:
    """CLOB ``/prices-history`` points: ``{"history": [{"t": unix_s, "p": price}, ...]}``."""
    return [{"token": token, "t": int(h["t"]), "p": float(h["p"])} for h in payload.get("history") or []
            if h.get("t") is not None and h.get("p") is not None]


def pm_history(clob, token: str, lo: datetime, hi: datetime, chunk_days: int = 14) -> list[dict[str, Any]]:
    """1-minute history over [lo, hi] in cached chunks of whole UTC days (stable cache keys). One request
    may span at most ~15 days at ``fidelity=1`` (30 days returns HTTP 400)."""
    out: list[dict[str, Any]] = []
    d0 = datetime(lo.year, lo.month, lo.day, tzinfo=UTC)
    while d0 <= hi:
        d1 = d0 + timedelta(days=chunk_days)
        p = clob.get("/prices-history", {"market": token, "startTs": int(d0.timestamp()),
                                         "endTs": int(d1.timestamp()), "fidelity": 1})
        out += [r for r in pm_history_rows(p.payload, token) if lo.timestamp() <= r["t"] <= hi.timestamp()]
        d0 = d1
    seen: set[int] = set()
    return [r for r in sorted(out, key=lambda r: r["t"]) if not (r["t"] in seen or seen.add(r["t"]))]


def stamp_offsets(quotes: pl.DataFrame, hist: pl.DataFrame,
                  offsets: tuple[int, ...] = tuple(range(-120, 121, 5))) -> pl.DataFrame:
    """For each offset d (s): share of history points whose price equals the recorded book's mid at
    ``t + d`` (within 0.5c) and the mean absolute difference. The offset that fits best says what
    instant a point stamped ``t`` describes."""
    q = quotes.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid")).sort("ts_ms")
    rows = []
    for d in offsets:
        h = hist.with_columns(((pl.col("t") + d) * 1000).alias("ts_ms")).sort("ts_ms")
        j = h.join_asof(q.select("asset", "ts_ms", "mid", "bid", "ask"), left_on="ts_ms", right_on="ts_ms",
                        by_left="token", by_right="asset", strategy="backward").drop_nulls("mid")
        if j.is_empty():
            continue
        err = (j["p"] - j["mid"]).abs()
        rows.append({"offset_s": d, "n": j.height, "match_mid": float((err <= 0.005).mean()),
                     "mae_c": float(err.mean()) * 100,
                     "inside": float(((j["p"] >= j["bid"]) & (j["p"] <= j["ask"])).mean())})
    return pl.DataFrame(rows)


# ---- matching: candidates ------------------------------------------------------------------------
STOP = frozenset(["will", "the", "be", "a", "an", "of", "in", "on", "at", "to", "for", "by", "and", "or", "is", "are", "was", "with", "this", "that", "from", "during", "any", "as", "its", "it", "than", "more", "less", "above", "below", "least", "most", "what", "who", "which", "how", "many", "next", "week", "weekly", "day", "daily", "yes", "no", "market", "resolve", "resolves", "2025", "2026", "2027", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december"])
_MONTH_RX = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
             r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_DATE_RX = re.compile(_MONTH_RX + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d\d)?", re.IGNORECASE)


def norm_text(text: str | None) -> str:
    t = (text or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", t).strip()


def tokens(text: str | None) -> list[str]:
    t = _DATE_RX.sub(" ", norm_text(text).lower())
    t = re.sub(r"[^a-z0-9#%.]+", " ", t)
    out = []
    for w in t.split():
        w = w.strip(".")
        if w and w not in STOP and not re.fullmatch(r"20\d\d", w):
            out.append(w)
    return out


def kalshi_text(r: dict[str, Any]) -> str:
    sub = r.get("yes_sub_title") or ""
    title = r.get("title") or ""
    return f"{title} {sub}" if sub and sub.lower() not in title.lower() else title


def pm_text(r: dict[str, Any]) -> str:
    g = r.get("group_title") or ""
    q = r.get("question") or ""
    return f"{q} {r.get('event_title') or ''} {g}" if g and g.lower() not in q.lower() else f"{q} {r.get('event_title') or ''}"


def _kalshi_when(r: dict[str, Any]) -> datetime | None:
    return r.get("expected_expiration_time") or r.get("close_time")


def _pm_when(r: dict[str, Any]) -> list[datetime]:
    return [d for d in (r.get("end_date"), r.get("closed_time")) if d is not None]


def candidates(k: pl.DataFrame, p: pl.DataFrame, *, max_days: float = 5.0, min_score: float = 0.45,
               top: int = 3) -> pl.DataFrame:
    """Top PM candidates per Kalshi market by IDF-weighted token overlap (Dice on weights), with the
    venues' resolution dates within ``max_days``. A screen only: nothing here is a match yet."""
    pm_rows = p.select("condition_id", "question", "group_title", "event_title", "event_slug", "end_date",
                       "closed_time").to_dicts()
    pm_tok = [set(tokens(pm_text(r))) for r in pm_rows]
    df: dict[str, int] = {}
    for ts in pm_tok:
        for w in ts:
            df[w] = df.get(w, 0) + 1
    n = len(pm_rows)
    idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}
    index: dict[str, list[int]] = {}
    for i, ts in enumerate(pm_tok):
        for w in ts:
            if df[w] <= 2000:
                index.setdefault(w, []).append(i)
    pm_w = [sum(idf[w] for w in ts) for ts in pm_tok]
    pm_t = [[d.timestamp() for d in _pm_when(r)] for r in pm_rows]
    out = []
    for r in k.select("ticker", "title", "yes_sub_title", "expected_expiration_time", "close_time").iter_rows(named=True):
        kt = set(tokens(kalshi_text(r)))
        when = _kalshi_when(r)
        if not kt or when is None:
            continue
        kw = sum(idf.get(w, math.log(n + 1)) for w in kt)
        acc: dict[int, float] = {}
        for w in kt:
            for i in index.get(w, ()):
                acc[i] = acc.get(i, 0.0) + idf[w]
        ts = when.timestamp()
        scored = []
        for i, sw in acc.items():
            if not pm_t[i] or min(abs(ts - x) for x in pm_t[i]) > max_days * 86400:
                continue
            s = 2 * sw / (kw + pm_w[i])
            if s >= min_score:
                scored.append((s, i))
        for s, i in sorted(scored, reverse=True)[:top]:
            out.append({"ticker": r["ticker"], "condition_id": pm_rows[i]["condition_id"], "score": s,
                        "rank": len(out)})
    c = pl.DataFrame(out, schema={"ticker": pl.String, "condition_id": pl.String, "score": pl.Float64, "rank": pl.Int64})
    return c.with_columns(pl.col("score").rank("ordinal", descending=True).over("ticker").alias("rank"))


# ---- matching: families and pair checks ----------------------------------------------------------
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                      "dec"], 1)}
_DATE_FULL = re.compile(_MONTH_RX + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d\d)", re.IGNORECASE)
_TICKER_DATE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})(?:\d{4})?(?:-|$)")


def _date(y: int, mon: str, d: int) -> str | None:
    m = MONTHS.get(mon.lower()[:3])
    try:
        return datetime(y, m, d).strftime("%Y-%m-%d") if m else None
    except (TypeError, ValueError):
        return None


def text_dates(text: str | None) -> list[str]:
    """ISO dates written as 'Sep 5, 2026' / 'September 12, 2026' / 'Tuesday, August 25, 2026'."""
    out = []
    for m in re.finditer(r"(" + _MONTH_RX + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d\d)", text or "", re.IGNORECASE):
        d = _date(int(m.group(3)), m.group(1), int(m.group(2)))
        if d:
            out.append(d)
    return out


def ticker_date(ticker: str) -> str | None:
    m = _TICKER_DATE.search(ticker or "")
    return _date(2000 + int(m.group(1)), m.group(2), int(m.group(3))) if m else None


def slug_date(slug: str | None, year: int) -> str | None:
    """'...-week-of-september-12' (year from the end date) or '...-20260825'."""
    s = slug or ""
    m = re.search(r"(20\d\d)(\d\d)(\d\d)(?:\d+)?$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"(" + _MONTH_RX + r")-(\d{1,2})(?:-(20\d\d))?(?:-\d+)?$", s, re.IGNORECASE)
    if m:
        return _date(int(m.group(3) or year), m.group(1), int(m.group(2)))
    return None


def shift(day: str | None, days: int) -> str | None:
    return (datetime.fromisoformat(day) + timedelta(days=days)).strftime("%Y-%m-%d") if day else None


def norm_name(text: str | None) -> str:
    import unicodedata

    t = unicodedata.normalize("NFKD", norm_text(text)).encode("ascii", "ignore").decode().lower()
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(w for w in t.split() if w not in ("the", "a", "an"))


PLACEHOLDER = re.compile(r"^(?:(?:song|movie|show|series|person|candidate|party|album|artist|nominee|player|team|"
                         r"option|company|model|film|individual|contestant|act|celebrity)\s+[a-z0-9]{1,2}|other|"
                         r"others|another party|any other|none|no one|tie|no qualifying event)$")


def name_match(k: str | None, p: str | None) -> float:
    """1.0 equal; 0.95 when one name is a whole-word part of the other (the shorter >= 6 chars and
    >= 2 words, e.g. 'mourinho season 1' vs 'mourinho'... not allowed, but 'below limited series' vs
    'below limited series season 1' is); else the difflib ratio. Placeholders ('Song A') score 0."""
    from difflib import SequenceMatcher

    a, b = norm_name(k), norm_name(p)
    raw = [re.sub(r"[^a-z0-9]+", " ", norm_text(x).lower()).strip() for x in (k, p)]
    if not a or not b or any(PLACEHOLDER.match(x) for x in raw):
        return 0.0
    if a == b:
        return 1.0
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 6 and (" " in short or len(short) >= 8) and re.search(
            r"(^| )" + re.escape(short) + r"( |$)", long_):
        return 0.95
    return SequenceMatcher(None, a, b).ratio()


def pm_quoted(q: str | None) -> str | None:
    m = re.search(r'["“](.+?)["”]', q or "")
    return m.group(1) if m else None


def pm_outcome(r: dict[str, Any]) -> str:
    """Polymarket's outcome label: the group item title, else the quoted name, else the question."""
    return r.get("group_title") or pm_quoted(r.get("question")) or r.get("question") or ""


def _bb_title(r: dict[str, Any]) -> list[str]:
    """'Title - Artist' (songs) or either order (albums): both halves are candidate names."""
    q = pm_quoted(r.get("question")) or r.get("group_title") or ""
    parts = [x.strip() for x in re.split(r"\s+[-–]\s+", q) if x.strip()]
    return parts or [q]


def mention_terms(text: str | None) -> frozenset[str]:
    """Words of a mention contract: Kalshi 'Afford / Affordable' or Polymarket '"Afford" or "Affordable"'."""
    t = norm_text(text)
    q = re.findall(r'"([^"]+)"', t)
    parts = q if q else re.split(r"\s*/\s*", t)
    return frozenset(norm_name(x) for x in parts if norm_name(x))


SPEAKERS = {
    "trump": ("KXTRUMPMENTION", "KXTRUMPSPEECH"), "vance": ("KXVANCEMENTION",),
    "mamdani": ("KXMAMDANIMENTION",), "burnham": ("KXBURNHAMMENTION",), "leavitt": ("KXSECPRESSMENTION",),
}


EVENT_STOP = frozenset(["during", "his", "her", "their", "the", "a", "an", "at", "on", "in", "of", "for", "to", "and", "next", "any", "remarks", "remark", "speech", "address", "event", "appearance", "live", "livestream", "originally", "scheduled", "will", "say"])
EVENT_SYNONYMS = {"pmq": "prime minister s questions", "pmqs": "prime minister s questions", "unga": "united nations general assembly"}


def event_words(text: str | None) -> frozenset[str]:
    t = norm_name(text)
    for k, v in EVENT_SYNONYMS.items():
        t = re.sub(r"(^| )" + k + r"( |$)", r"\1" + v + r"\2", t)
    return frozenset(w for w in t.split() if w not in EVENT_STOP and not re.fullmatch(r"\d+", w) and w not in MONTHS
                     and w not in ("january", "february", "march", "april", "june", "july", "august", "september",
                                   "october", "november", "december"))


def k_event(r: dict[str, Any]) -> str:
    m = re.search(r"say during (.+?)(?: originally scheduled.*)?\??$", norm_text(r.get("title")), re.IGNORECASE)
    return m.group(1) if m else ""


def p_event(r: dict[str, Any]) -> str:
    q = re.sub(r'"[^"]*"', " ", norm_text(r.get("question")))
    m = re.search(r"(?:during|at) (.+?)\??$", q, re.IGNORECASE)
    return m.group(1) if m else ""


def same_speech(k: dict[str, Any], p: dict[str, Any]) -> bool:
    """Every content word of Polymarket's event phrase ('his Midterm Rally in Nebraska') appears in
    Kalshi's ('Midterm Rally in Grand Island, Nebraska'); period markets ('this week') never match."""
    if re.search(r"this-week|-in-(january|february|march|april|may|june|july|august|september|october|november|"
                 r"december)|-this-month|-before-", p.get("event_slug") or ""):
        return False
    pw, kw = event_words(p_event(p)), event_words(k_event(k))
    return bool(pw) and pw <= kw


def _first(xs: list[str]) -> str | None:
    return xs[0] if xs else None


def _rx_date(rx: str, text: str | None, days: int = 0) -> str | None:
    m = re.search(rx, norm_text(text), re.IGNORECASE)
    return shift(_first(text_dates(m.group(1))), days) if m else None


def _cat(rx: str, text: str | None) -> str | None:
    m = re.search(rx, norm_text(text), re.IGNORECASE)
    return norm_name(m.group(1)) if m else None


def _num(rx: str, text: str | None) -> float | None:
    m = re.search(rx, norm_text(text), re.IGNORECASE)
    return float(m.group(1).replace(",", "")) if m else None


def _person(text: str | None) -> str:
    """'Leslie Bibb – “Hacks”' -> 'Leslie Bibb'; 'Jay Feely (R)' -> 'Jay Feely'."""
    t = re.split(r"\s+[–—-]\s+[\"“]", norm_text(text))[0]
    return re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()


NETFLIX = {  # Kalshi series -> (scope, kind, rank)
    "KXNETFLIXRANKSHOW": ("us", "show", "top"), "KXNETFLIXRANKSHOWRUNNERUP": ("us", "show", "2"),
    "KXNETFLIXRANKSHOWGLOBAL": ("global", "show", "top"), "KXNETFLIXRANKSHOWGLOBAL2": ("global", "show", "2"),
    "KXNETFLIXRANKMOVIE": ("us", "movie", "top"), "KXNETFLIXRANKMOVIERUNNERUP": ("us", "movie", "2"),
    "KXNETFLIXRANKMOVIEGLOBAL": ("global", "movie", "top"), "KXNETFLIXRANKMOVIEGLOBAL2": ("global", "movie", "2"),
}
NOBEL = {"chemistry": "chemistry", "physics": "physics", "medicine": "medicine", "physiology": "medicine",
         "literature": "literature", "peace": "peace", "economic": "economics"}


def _nobel(text: str | None) -> str | None:
    t = (text or "").lower()
    return next((v for k, v in NOBEL.items() if k in t), None)


def _netflix_pm(r: dict[str, Any]) -> tuple | None:
    m = re.match(r"what-will-be-the-(top|2)-(us|global)-netflix-(show|movie)-this-week", r["event_slug"] or "")
    d = _first(text_dates(r.get("description")))
    return (m.group(2), m.group(3), m.group(1), d) if m and d else None


def _views_pm(r: dict[str, Any], kind: str) -> tuple | None:
    if not re.match(rf"how-many-views-will-the-1-{kind}-on-netflix", r["event_slug"] or ""):
        return None
    thr = _num(r"have at least ([\d.,]+) million", r.get("question"))
    d = _first(text_dates(r.get("description")))
    return (kind, thr, d) if thr is not None and d else None


def _mention_count(text: str | None) -> int:
    m = re.search(r"(\d+)\+?\s*(?:or more\s+)?times", norm_text(text), re.IGNORECASE)
    return int(m.group(1)) if m else 1


FAMILIES: list[dict[str, Any]] = [
    # --- culture: charts (Billboard dates its charts "Week of <Saturday>"; both venues resolve on that chart)
    {"name": "billboard_hot100_1", "group": "culture", "kalshi": r"^KXTOPSONG$", "pm": r"^billboard-hot-100-1-song-week-of-",
     "k_key": lambda r: ("d", ticker_date(r["event_ticker"])), "p_key": lambda r: ("d", _first(text_dates(r["description"]))),
     "k_names": lambda r: [r["yes_sub_title"]], "p_names": _bb_title, "min_name": 1.0,
     "note": "Kalshi: '#1 on the Billboard Hot 100 chart for the Week of D'; PM: 'number 1 song on the chart titled Week of D'"},
    {"name": "billboard_hot100_2", "group": "culture", "kalshi": r"^KXBILLBOARDRUNNERUPSONG$",
     "pm": r"^billboard-hot-100-2-song-week-of-", "k_key": lambda r: ("d", ticker_date(r["event_ticker"])),
     "p_key": lambda r: ("d", _first(text_dates(r["description"]))), "k_names": lambda r: [r["yes_sub_title"]],
     "p_names": _bb_title, "min_name": 1.0, "note": "#2 song, same chart date"},
    {"name": "billboard_200_1", "group": "culture", "kalshi": r"^KXTOPALBUM$", "pm": r"^billboard-200-1-album-week-of-",
     "k_key": lambda r: ("d", ticker_date(r["event_ticker"])), "p_key": lambda r: ("d", _first(text_dates(r["description"]))),
     "k_names": lambda r: [r["yes_sub_title"]], "p_names": _bb_title, "min_name": 1.0,
     "note": "#1 album, same chart date (exact title: a deluxe edition is a different outcome label)"},
    {"name": "billboard_200_2", "group": "culture", "kalshi": r"^KXBILLBOARDRUNNERUPALBUM$",
     "pm": r"^billboard-200-2-album-week-of-", "k_key": lambda r: ("d", ticker_date(r["event_ticker"])),
     "p_key": lambda r: ("d", _first(text_dates(r["description"]))), "k_names": lambda r: [r["yes_sub_title"]],
     "p_names": _bb_title, "min_name": 1.0, "note": "#2 album, same chart date"},
    # --- culture: Netflix Top 10 (published Tuesdays; both venues name the publication date)
    {"name": "netflix_rank", "group": "culture", "kalshi": r"^KXNETFLIXRANK", "pm": r"^what-will-be-the-(top|2)-(us|global)-netflix-",
     "k_key": lambda r: (*NETFLIX[r["series_ticker"]], _rx_date(r"chart published on (.+?\d{4})", r["rules_primary"]))
     if r["series_ticker"] in NETFLIX else None,
     "p_key": _netflix_pm, "k_names": lambda r: [r["yes_sub_title"]], "p_names": lambda r: [pm_outcome(r)],
     "note": "Kalshi: '#N on the Netflix Top 10 <US|Global> <Show|Movie> on the chart published on D'; PM: same list, same update"},
    {"name": "netflix_views", "group": "culture", "kalshi": r"^KXNETFLIXTOPVIEWS(MOVIE|TV)$", "pm": r"^how-many-views-will-the-1-",
     "k_key": lambda r: ("movie" if r["series_ticker"].endswith("MOVIE") else "show",
                         _num(r"at least ([\d.,]+) million", r["yes_sub_title"]),
                         _rx_date(r"chart published on (.+?\d{4})", r["rules_primary"])),
     "p_key": lambda r: _views_pm(r, "movie") or _views_pm(r, "show"), "k_names": lambda r: ["x"], "p_names": lambda r: ["x"],
     "note": "'#1 movie/show has at least N million views' on the same Tuesday update (views are global-only)"},
    # --- culture: awards
    {"name": "emmys", "group": "culture", "kalshi": r"^KXEMMY", "pm": r"^emmys-2026-",
     "k_key": lambda r: ("emmy", _cat(r"win (.+?) at the (?:\d+\w\w )?(?:Primetime )?Emmy", r["title"])),
     "p_key": lambda r: ("emmy", _cat(r"Emmys 2026: (.+?)\??$", r["question"])),
     "k_names": lambda r: [_person(r["yes_sub_title"]), r["yes_sub_title"]],
     "p_names": lambda r: [_person(pm_outcome(r)), pm_outcome(r)], "note": "same category of the 78th Primetime Emmys"},
    {"name": "vmas", "group": "culture", "kalshi": r"^KXVMA$", "pm": r"^vma-",
     "k_key": lambda r: ("vma", _cat(r"win (.+?) at the 2026 MTV", r["title"])),
     "p_key": lambda r: ("vma", _cat(r"win (.+?) at the 2026 MTV", r["question"])),
     "k_names": lambda r: [r["yes_sub_title"]], "p_names": lambda r: [pm_outcome(r), r["question"]],
     "note": "same category of the 2026 VMAs"},
    {"name": "nobel", "group": "culture", "kalshi": r"^KXNOBEL", "pm": r"^nobel-.*prize",
     "k_key": lambda r: ("nobel", _nobel(r["title"])), "p_key": lambda r: ("nobel", _nobel(r["event_slug"])),
     "k_names": lambda r: [r["yes_sub_title"]], "p_names": lambda r: [pm_outcome(r)],
     "note": "2026 laureate in the same discipline (shared prizes count on both)"},
    # --- culture: reality TV (near-exact: Kalshi 'officially eliminated before D' vs PM 'eliminated by D-1 11:59 PM')
    {"name": "dwts_elim", "group": "culture", "kalshi": r"^KXDWTSELIMINATION$", "pm": r"^dancing-with-the-stars-s\d+-week-\d+-elimination",
     "k_key": lambda r: ("dwts", _rx_date(r"before (.+?\d{4})", r["title"])),
     "p_key": lambda r: ("dwts", _rx_date(r"by (.+?\d{4}), 11:59 PM", r["description"], 1)),
     "k_names": lambda r: [r["yes_sub_title"]], "p_names": lambda r: [pm_outcome(r)], "exact": "near",
     "note": "PM counts withdrawal as elimination; Kalshi says 'officially eliminated'"},
    {"name": "bigbrother_evict", "group": "culture", "kalshi": r"^KXBIGBROTHERELIMINATION$", "pm": r"^who-will-be-evicted-from-big-brother",
     "k_key": lambda r: ("bb", _rx_date(r"before (.+?\d{4})", r["title"])),
     "p_key": lambda r: ("bb", _rx_date(r"and (.+?\d{4}), 11:59 PM", r["description"], 1)),
     "k_names": lambda r: [r["yes_sub_title"]], "p_names": lambda r: [pm_outcome(r)], "exact": "near",
     "note": "PM window starts at market creation and excludes voluntary departure"},
    # --- world: foreign elections (same contest, winner incl. any run-off)
    {"name": "brazil_president", "group": "world", "kalshi": r"^KXBRPRES$", "pm": r"^brazil-presidential-election$",
     "k_key": lambda r: ("br",), "p_key": lambda r: ("br",), "k_names": lambda r: [r["yes_sub_title"]],
     "p_names": lambda r: [pm_outcome(r)], "note": "winner of the 2026 Brazilian presidential election (incl. run-off)"},
    {"name": "israel_place", "group": "world", "kalshi": r"^KXISRAEL(2ND|3RD)$", "pm": r"^israeli-legislative-election-\d\w\w-place",
     "k_key": lambda r: ("il", r["series_ticker"][-3:-2]), "p_key": lambda r: ("il", re.search(r"-(\d)\w\w-place", r["event_slug"]).group(1)),
     "k_names": lambda r: [r["yes_sub_title"]], "p_names": lambda r: [pm_outcome(r)], "exact": "near",
     "note": "Kalshi: certified results; PM: seats with vote/alphabet tie-breaks"},
    # --- mentions: same speaker, same event (closing times within a day, unique), same word set and count
    {"name": "mentions", "group": "mentions", "kalshi": r"^(" + "|".join(s for v in SPEAKERS.values() for s in v) + r")$",
     "pm": r"^what-will-(donald-)?(trump|jd-vance|vance|mamdani|zohran-mamdani|andy-burnham|karoline-leavitt|leavitt)-say",
     "k_key": lambda r: (next(k for k, v in SPEAKERS.items() if r["series_ticker"] in v), _mention_count(r["yes_sub_title"])),
     "p_key": lambda r: (next((k for k in SPEAKERS if k in (r["event_slug"] or "")), None), _mention_count(r["question"])),
     "k_names": lambda r: ["|".join(sorted(mention_terms(re.sub(r"\(.*?\)", "", r["yes_sub_title"] or ""))))],
     "p_names": lambda r: ["|".join(sorted(mention_terms(r["question"])))], "min_name": 1.0, "time_tol_h": 12.0,
     "pair_ok": same_speech, "exact": "near",
     "note": "same speaker, same speech (PM's event words all in Kalshi's event title; closes <= 12 h apart), same "
             "word set and count; both count plural/possessive forms"},
]


def match_family(fam: dict[str, Any], k: pl.DataFrame, p: pl.DataFrame) -> list[dict[str, Any]]:
    """Pairs within one family: equal instance keys, then the best outcome-name match (unique,
    >= ``min_name``), one-to-one. With ``time_tol_h`` the closing times must also be that close."""
    kk = k.filter(pl.col("series_ticker").str.contains(fam["kalshi"])).to_dicts()
    pp = p.filter(pl.col("event_slug").str.contains(fam["pm"])).to_dicts()
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in pp:
        key = fam["p_key"](r)
        if key is not None and None not in key:
            groups.setdefault(key, []).append(r)
    min_name = fam.get("min_name", 0.9)
    tol = fam.get("time_tol_h")
    cand = []
    for r in kk:
        key = fam["k_key"](r)
        if key is None or None in key or key not in groups:
            continue
        scored = []
        for q in groups[key]:
            if tol is not None:
                kt, pt = r["close_time"], q["closed_time"] or q["end_date"]
                if kt is None or pt is None or abs((kt - pt).total_seconds()) > tol * 3600:
                    continue
            if "pair_ok" in fam and not fam["pair_ok"](r, q):
                continue
            s = max((name_match(a, b) for a in fam["k_names"](r) for b in fam["p_names"](q)), default=0.0)
            scored.append((s, q["condition_id"]))
        scored.sort(reverse=True)
        if not scored or scored[0][0] < min_name:
            continue
        if len(scored) > 1 and scored[1][0] >= min_name and scored[1][0] >= scored[0][0] - 0.02:
            continue                                   # ambiguous
        exact = fam["exact_fn"](r) if "exact_fn" in fam else fam.get("exact", "exact")
        cand.append({"ticker": r["ticker"], "condition_id": scored[0][1], "name_score": scored[0][0],
                     "family": fam["name"], "group": fam["group"], "exact": exact})
    # one-to-one on the PM side: keep a PM market only when exactly one Kalshi market attains its best score
    by_pm: dict[str, list[dict[str, Any]]] = {}
    for c in cand:
        by_pm.setdefault(c["condition_id"], []).append(c)
    out = []
    for cs in by_pm.values():
        top = max(c["name_score"] for c in cs)
        winners = [c for c in cs if c["name_score"] == top]
        if len(winners) == 1:
            out.append(winners[0])
    return sorted(out, key=lambda c: c["ticker"])


# ---- pairs ---------------------------------------------------------------------------------------
WINDOW_DAYS = 40            # price paths: at most the last 40 days of the overlap before trading ends
ELECTION_WINDOW_DAYS = 21   # midterm races (812 House legs): the last three weeks (API budget; Oct 13+ is holdout anyway)
SIGNAL_START = datetime(2026, 7, 20, tzinfo=UTC)
PAIR_SCHEMA = {
    "pair_id": pl.String, "family": pl.String, "group": pl.String, "exact": pl.String, "name_score": pl.Float64,
    "ticker": pl.String, "event_ticker": pl.String, "series_ticker": pl.String, "condition_id": pl.String,
    "token": pl.String, "pm_event_slug": pl.String, "k_result": pl.String, "pm_result": pl.String,
    "fee_multiplier": pl.Float64, "k_open": pl.Datetime("us", "UTC"), "k_close": pl.Datetime("us", "UTC"),
    "pm_closed": pl.Datetime("us", "UTC"), "t0": pl.Datetime("us", "UTC"), "t1": pl.Datetime("us", "UTC"),
    "k_end": pl.Datetime("us", "UTC"), "k_settle": pl.Datetime("us", "UTC"),
}
EXIT_TAIL = timedelta(hours=7)     # Kalshi quotes after t1, for exits (fixed horizons up to 6 h)


def kalshi_universe() -> pl.DataFrame:
    parts = [pl.read_parquet(OUT / f) for f in ("kalshi_shared.parquet", "kalshi_recent.parquet", "kalshi_hist.parquet")
             if (OUT / f).exists()]
    k = pl.concat(parts, how="vertical_relaxed").unique("ticker", keep="last", maintain_order=True).sort("ticker")
    return k.filter(pl.col("close_time").is_between(START, END))


def match_all(k: pl.DataFrame, p: pl.DataFrame, families: list[dict[str, Any]] | None = None) -> pl.DataFrame:
    rows = []
    for fam in families or FAMILIES:
        got = match_family(fam, k, p)
        print(f"  {fam['name']}: {len(got)} pairs", flush=True)
        rows += got
    m = pl.DataFrame(rows, schema={"ticker": pl.String, "condition_id": pl.String, "name_score": pl.Float64,
                                   "family": pl.String, "group": pl.String, "exact": pl.String})
    return m.unique("ticker", keep="first", maintain_order=True)


def build_pairs(m: pl.DataFrame, k: pl.DataFrame, p: pl.DataFrame) -> pl.DataFrame:
    """Matched pairs with the YES token, both results and the price-path window [t0, t1]: from the
    later of the two listings (and ``SIGNAL_START``) to the earlier of Kalshi's close and Polymarket's
    close, at most ``WINDOW_DAYS`` long."""
    j = m.join(k.select("ticker", "event_ticker", "series_ticker", "result", "fee_multiplier", "open_time", "close_time",
                        "settlement_ts"),
               on="ticker").join(p.select("condition_id", "token_ids", "outcomes", "outcome_prices", "event_slug",
                                          "start_date", "closed_time", "end_date"), on="condition_id")
    j = j.filter(pl.col("outcomes").list.get(0).str.to_lowercase() == "yes")
    pm_res = (pl.when(pl.col("outcome_prices").list.get(0) >= 0.99).then(pl.lit("yes"))
              .when(pl.col("outcome_prices").list.get(0) <= 0.01).then(pl.lit("no")).otherwise(pl.lit(None)))
    t1 = pl.min_horizontal("close_time", pl.coalesce("closed_time", "end_date"))
    days = pl.when(pl.col("group") == "elections").then(pl.duration(days=ELECTION_WINDOW_DAYS)).otherwise(
        pl.duration(days=WINDOW_DAYS))
    t0 = pl.max_horizontal("open_time", pl.coalesce("start_date", "open_time"), pl.lit(SIGNAL_START), t1 - days)
    out = j.with_columns(
        (pl.col("ticker") + "|" + pl.col("condition_id").str.slice(0, 10)).alias("pair_id"),
        pl.col("token_ids").list.get(0).alias("token"), pl.col("event_slug").alias("pm_event_slug"),
        pl.col("result").alias("k_result"), pm_res.alias("pm_result"), pl.col("open_time").alias("k_open"),
        pl.col("close_time").alias("k_close"), pl.col("closed_time").alias("pm_closed"), t0.alias("t0"), t1.alias("t1"),
        pl.min_horizontal("close_time", t1 + EXIT_TAIL).alias("k_end"),
        pl.coalesce("settlement_ts", "close_time").alias("k_settle"))
    return out.filter(pl.col("t1") > pl.col("t0")).select(list(PAIR_SCHEMA)).cast(PAIR_SCHEMA).sort("pair_id")


# ---- price paths ---------------------------------------------------------------------------------
WARMUP = timedelta(hours=24)       # Kalshi quotes before t0 (candles are omitted when nothing changes)
KALSHI_MAX_PERIODS = 5000
KALSHI_MAX_BATCH = 10000


def fetch_pm_paths(clob, pairs: pl.DataFrame, out: Path, *, save_every: int = 40) -> None:
    """1-minute Polymarket history of each pair's YES token over [t0 - 1 h, t1] (parts, resumable;
    buffered as per-token frames, not row dicts, to keep the process small)."""
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "done.parquet"
    done = set(pl.read_parquet(done_path)["token"]) if done_path.exists() else set()
    todo = pairs.group_by("token").agg(pl.col("t0").min(), pl.col("t1").max()).filter(~pl.col("token").is_in(list(done)))
    part = len(list(out.glob("part-*.parquet")))
    schema = {"token": pl.String, "t": pl.Int64, "p": pl.Float64}
    buf: list[pl.DataFrame] = []
    fin: list[str] = []
    print(f"pm paths: {todo.height} tokens to fetch ({len(done)} done)", flush=True)
    for i, r in enumerate(todo.iter_rows(named=True), 1):
        try:
            buf.append(pl.DataFrame(pm_history(clob, r["token"], r["t0"] - timedelta(hours=1), r["t1"]), schema=schema))
            fin.append(r["token"])
        except RuntimeError as error:
            print(f"  pm skip {r['token'][:12]}: {str(error)[:160]}", flush=True)
        if i % save_every == 0 or i == todo.height:
            if buf:
                pl.concat(buf).write_parquet(out / f"part-{part:05d}.parquet")
                part += 1
            done |= set(fin)
            pl.DataFrame({"token": sorted(done)}).write_parquet(done_path)
            buf, fin = [], []
            print(f"  pm {i}/{todo.height}", flush=True)


def kalshi_jobs(pairs: pl.DataFrame, cutoff: datetime) -> list[dict[str, Any]]:
    """Requests within the API caps: markets settled before the historical cutoff one by one (<= 5,000
    periods each); later ones batched per event, each request <= 5,000 periods and <= 10,000 candles."""
    w = pairs.group_by("ticker", "event_ticker").agg((pl.col("t0").min() - WARMUP).alias("lo"), pl.col("k_end").max().alias("hi"),
                                                     pl.col("k_close").max().alias("close"))
    jobs: list[dict[str, Any]] = []

    def spans(lo: datetime, hi: datetime, n: int) -> Iterator[tuple[datetime, datetime]]:
        step = timedelta(minutes=n - 1)
        while lo < hi:
            yield lo, min(hi, lo + step)
            lo = lo + step + timedelta(minutes=1)

    w = w.join(pairs.group_by("ticker").agg(pl.col("k_settle").max().alias("settle")), on="ticker")
    w = w.with_columns(pl.max_horizontal("close", "settle").alias("close"))
    for r in w.filter(pl.col("close") < cutoff).iter_rows(named=True):
        for a, b in spans(r["lo"], r["hi"], KALSHI_MAX_PERIODS):
            jobs.append({"kind": "hist", "tickers": [r["ticker"]], "lo": a, "hi": b})
    for _, g in w.filter(pl.col("close") >= cutoff).sort("event_ticker", "ticker").group_by("event_ticker", maintain_order=True):
        lo, hi = g["lo"].min(), g["hi"].max()
        tick = g["ticker"].to_list()
        minutes = (hi - lo).total_seconds() / 60 + 1
        per = max(1, min(len(tick), KALSHI_MAX_BATCH // int(min(minutes, KALSHI_MAX_PERIODS))))
        for i in range(0, len(tick), per):
            chunk = tick[i:i + per]
            length = min(KALSHI_MAX_PERIODS, KALSHI_MAX_BATCH // len(chunk))
            for a, b in spans(lo, hi, length):
                jobs.append({"kind": "batch", "tickers": chunk, "lo": a, "hi": b})
    return jobs


def fetch_kalshi_paths(api, pairs: pl.DataFrame, out: Path, *, save_every: int = 200) -> None:
    """Minute candles (bid/ask OHLC) for every pair's ticker over [t0 - 24 h, k_end]; resumable by job."""
    from albert.research.xcat import OHLC_SCHEMA, batch_rows, ohlc_rows

    out.mkdir(parents=True, exist_ok=True)
    cutoff = api.cutoff()["market_settled_ts"]
    jobs = kalshi_jobs(pairs, cutoff)
    done_path = out / "done_jobs.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()

    def jid(j: dict[str, Any]) -> str:
        return f"{','.join(j['tickers'])}|{int(j['lo'].timestamp())}|{int(j['hi'].timestamp())}"

    todo = [j for j in jobs if jid(j) not in done]
    print(f"kalshi paths: {len(jobs)} requests ({len(todo)} to do), cutoff {cutoff:%Y-%m-%d}", flush=True)
    part = len(list(out.glob("part-*.parquet")))
    buf: list[dict[str, Any]] = []
    t_start = time.time()
    for i, j in enumerate(todo, 1):
        try:
            if j["kind"] == "hist":
                t = j["tickers"][0]
                try:
                    rows = ohlc_rows(api.candles(t, j["lo"], j["hi"], 1, True).payload, t, 1)
                except RuntimeError as error:   # settled after the cutoff although closed before it: live only
                    if "404" not in str(error):
                        raise
                    rows = ohlc_rows(api.candles(t, j["lo"], j["hi"], 1, False).payload, t, 1)
            else:
                got = batch_rows(api.candles_batch(j["tickers"], j["lo"], j["hi"], 1).payload, 1)
                rows = [x for v in got.values() for x in v]
                for t in j["tickers"]:               # unknown tickers are silently omitted: try alone, live
                    if t not in got:
                        rows += ohlc_rows(api.candles(t, j["lo"], j["hi"], 1, False).payload, t, 1)
            buf += rows
            done.add(jid(j))
        except RuntimeError as error:
            print(f"  kalshi skip {jid(j)[:60]}: {str(error)[:160]}", flush=True)
        if i % save_every == 0 or i == len(todo):
            if buf:
                pl.DataFrame(buf, schema=OHLC_SCHEMA).write_parquet(out / f"part-{part:05d}.parquet")
                part += 1
            buf = []
            done_path.write_text(json.dumps(sorted(done)))
            rate = i / max(time.time() - t_start, 1e-9)
            print(f"  kalshi {i}/{len(todo)} ({rate:.2f} req/s, eta {(len(todo) - i) / max(rate, 1e-9) / 60:.0f} min)",
                  flush=True)


# ---- minute grid ---------------------------------------------------------------------------------
PM_STALE_S = 600           # a Polymarket point older than this is not a price
KALSHI_TAKER = 0.07
STAKE = 4.5                # $ per taker order (n = floor(STAKE / price), at least 1): cost + fee <= ~$4.82,
                           # inside the ledger's $5-per-event cap
XS = (0.03, 0.05, 0.10)
NS = (5, 15, 60)
HORIZONS = {"15m": 15, "1h": 60, "6h": 360, "settle": None}


def grid(pair: dict[str, Any], pm_t: np.ndarray, pm_p: np.ndarray, k_end: np.ndarray, k_bid: np.ndarray,
         k_ask: np.ndarray) -> dict[str, np.ndarray]:
    """Minute grid over the pair's window. Row i is the instant ``T_i`` = end of minute i (unix s):

    * ``pm``: the last Polymarket point stamped <= T_i (a point stamped t shows the book ~14 s before t,
      see :func:`stamp_offsets`), NaN if older than ``PM_STALE_S``;
    * ``bid``/``ask``: the close of the last Kalshi minute candle ending <= T_i (candles are omitted when
      nothing changed, so this is a forward fill); NaN when that side is empty (no bid / no ask).

    Rows after ``t1`` (one venue stopped trading) are kept for exits only (``live`` False)."""
    m0 = int(pair["t0"].timestamp()) // 60
    m_end = int(pair["k_end"].timestamp()) // 60 - 1      # last minute that ends at or before Kalshi's end
    T = (np.arange(m0, max(m_end, m0) + 1, dtype=np.int64) + 1) * 60
    i = np.searchsorted(pm_t, T, side="right") - 1
    ok = (i >= 0) & (T - pm_t[np.clip(i, 0, None)] <= PM_STALE_S) if len(pm_t) else np.zeros(len(T), bool)
    pm = np.where(ok, pm_p[np.clip(i, 0, None)] if len(pm_p) else np.nan, np.nan)
    j = np.searchsorted(k_end, T, side="right") - 1
    has = j >= 0
    jj = np.clip(j, 0, None)
    bid = np.where(has, k_bid[jj] if len(k_bid) else np.nan, np.nan)
    ask = np.where(has, k_ask[jj] if len(k_ask) else np.nan, np.nan)
    bid = np.where((bid > 0) & (bid < 1), bid, np.nan)
    ask = np.where((ask > 0) & (ask < 1), ask, np.nan)
    mid = np.where(np.isfinite(bid) & np.isfinite(ask) & (ask > bid), (bid + ask) / 2, np.nan)
    live = T <= int(pair["t1"].timestamp())
    return {"T": T, "pm": pm, "bid": bid, "ask": ask, "mid": mid, "live": live}


def taker_fee(n: float, price: float, mult: float | None) -> float:
    """Kalshi taker fee for one order, rounded up to the cent (``ta_calendar.taker_fee_expr``)."""
    raw = KALSHI_TAKER * (1.0 if mult is None or not math.isfinite(mult) else mult) * n * price * (1 - price)
    return math.ceil(round(raw * 100, 9)) / 100


def events_for(g: dict[str, np.ndarray], x: float, n: int, leader: str) -> list[tuple[int, int]]:
    """(row, direction) where the leader's price moved >= x over the last n minutes while the other
    venue's mid moved less than x/2 in that direction; one event per n minutes (cooldown)."""
    a, b = (g["pm"], g["mid"]) if leader == "pm" else (g["mid"], g["pm"])
    if len(a) <= n:
        return []
    da = np.full(len(a), np.nan)
    db = np.full(len(a), np.nan)
    da[n:] = a[n:] - a[:-n]
    db[n:] = b[n:] - b[:-n]
    with np.errstate(invalid="ignore"):
        cand = np.flatnonzero(g["live"] & (np.abs(da) >= x - 1e-9) & (np.sign(da) * db < x / 2))
    out: list[tuple[int, int]] = []
    last = -10**9
    for i in cand:
        if i - last >= n:
            out.append((int(i), int(np.sign(da[i]))))
            last = i
    return out


def trades_for(pair: dict[str, Any], g: dict[str, np.ndarray], ev: list[tuple[int, int]], *, lag: int = 1,
               stake: float = STAKE) -> list[dict[str, Any]]:
    """Kalshi taker in direction d at row i + lag (YES at the ask, NO at 1 - bid), exits per horizon:
    take the bid (YES) / 1 - ask (NO) h minutes after entry while Kalshi still trades, else settlement."""
    T, bid, ask, mid, pm = g["T"], g["bid"], g["ask"], g["mid"], g["pm"]
    last = len(T) - 1
    y = 1.0 if pair["k_result"] == "yes" else 0.0
    mult = pair.get("fee_multiplier")
    out = []
    for i, d in ev:
        e = i + lag
        if e > last or not g["live"][e] or not (np.isfinite(bid[e]) and np.isfinite(ask[e])):
            continue
        price = ask[e] if d > 0 else 1 - bid[e]
        if not 0.01 <= price <= 0.99:
            continue
        n = max(1.0, math.floor(stake / price))
        fee_in = taker_fee(n, price, mult)
        row = {"pair_id": pair["pair_id"], "ticker": pair["ticker"], "event_ticker": pair["event_ticker"],
               "pm_event": pair["pm_event_slug"],
               "family": pair["family"], "group": pair["group"], "exact": pair["exact"], "t_sig": int(T[i]),
               "t_in": int(T[e]), "dir": d, "price": float(price), "n": n, "fee_in": fee_in,
               "spread": float(ask[e] - bid[e]), "pm_move": float(pm[i] - pm[i - 1]) if i > 0 else np.nan,
               "gap_in": float(d * (pm[i] - mid[i])) if np.isfinite(mid[i]) and np.isfinite(pm[i]) else np.nan}
        for h, mins in HORIZONS.items():
            x = e + mins if mins is not None else None
            if x is not None and x <= last and np.isfinite(bid[x]) and np.isfinite(ask[x]):
                exit_ = bid[x] if d > 0 else 1 - ask[x]
                fee_out = taker_fee(n, exit_, mult) if 0 < exit_ < 1 else 0.0
                t_out = int(T[x])
                kmid = mid[x]
                pmx = pm[x]
            else:
                exit_ = y if d > 0 else 1 - y
                fee_out = 0.0
                t_out = int(pair["k_settle"].timestamp()) if pair.get("k_settle") else int(T[last])
                kmid = exit_ if d > 0 else 1 - exit_
                pmx = np.nan
            row[f"pnl_{h}"] = float(exit_ - price - (fee_in + fee_out) / n)
            row[f"gross_{h}"] = float(exit_ - price)
            row[f"exit_{h}"] = float(exit_)
            row[f"fee_out_{h}"] = fee_out
            row[f"t_out_{h}"] = t_out
            # mid follow-through (signed): how far Kalshi's mid moved in the signal's direction
            row[f"kfollow_{h}"] = float(d * (kmid - mid[i])) if np.isfinite(mid[i]) and np.isfinite(kmid) else np.nan
            row[f"pmfollow_{h}"] = float(d * (pmx - pm[i])) if np.isfinite(pm[i]) and np.isfinite(pmx) else np.nan
        out.append(row)
    return out


def cluster_t(df: pl.DataFrame, col: str, a: str = "pm_event", b: str = "date") -> dict[str, float]:
    """Mean of ``col`` with a two-way cluster-robust t (Cameron-Gelbach-Miller: V_a + V_b - V_ab). The
    default clusters are the real-world event (the Polymarket event, which groups every Kalshi series on
    the same race/chart/speech) and the ET date."""
    f = df.filter(pl.col(col).is_not_null() & pl.col(col).is_not_nan())
    n = f.height
    if n < 2:
        return {"n": n, "mean": float(f[col].mean()) if n else float("nan"), "t": float("nan"), "se": float("nan")}
    mu = float(f[col].mean())
    e = f.with_columns((pl.col(col) - mu).alias("_e"))

    def v(keys: list[str]) -> float:
        s = e.group_by(keys).agg(pl.col("_e").sum().alias("s"))
        g = s.height
        adj = g / (g - 1) if g > 1 else 1.0
        return adj * float((s["s"] ** 2).sum()) / n ** 2

    var = v([a]) + v([b]) - v([a, b])
    if var <= 0:
        var = v([a])
    se = math.sqrt(var) if var > 0 else float("nan")
    return {"n": n, "mean": mu, "t": mu / se if se and se > 0 else float("nan"), "se": se,
            "events": f[a].n_unique(), "dates": f[b].n_unique()}


FOLLOW_H = (1, 2, 5, 15, 60)
XCORR_LAGS = tuple(range(-5, 6))


def follow_for(pair: dict[str, Any], g: dict[str, np.ndarray], ev: list[tuple[int, int]], leader: str,
               n: int = 0) -> list[dict[str, Any]]:
    """Signed moves after each event, from the event row i: the other venue (follow) and the leader
    (continuation), in price units, h minutes later (live rows only)."""
    lead, other = (g["pm"], g["mid"]) if leader == "pm" else (g["mid"], g["pm"])
    last = int(np.flatnonzero(g["live"])[-1]) if g["live"].any() else -1
    rows = []
    for i, d in ev:
        r = {"pair_id": pair["pair_id"], "event_ticker": pair["event_ticker"], "pm_event": pair["pm_event_slug"],
             "group": pair["group"],
             "family": pair["family"], "t_sig": int(g["T"][i]), "dir": d, "leader": leader,
             "move": float(d * (lead[i] - lead[i - n])) if n and i >= n else np.nan,
             "other_move": float(d * (other[i] - other[i - n])) if n and i >= n and np.isfinite(other[i - n]) else np.nan,
             "gap": float(d * (lead[i] - other[i])) if np.isfinite(other[i]) else np.nan}
        for h in FOLLOW_H:
            x = i + h
            ok = x <= last
            r[f"other_{h}"] = float(d * (other[x] - other[i])) if ok and np.isfinite(other[x]) and np.isfinite(other[i]) else np.nan
            r[f"lead_{h}"] = float(d * (lead[x] - lead[i])) if ok and np.isfinite(lead[x]) and np.isfinite(lead[i]) else np.nan
        rows.append(r)
    return rows


def xcorr_sums(g: dict[str, np.ndarray]) -> dict[int, tuple[float, float, float, int]]:
    """Sums for the pooled cross-correlation of 1-minute changes: lag k pairs Kalshi's change at t with
    Polymarket's change at t - k (k > 0: Polymarket first)."""
    live = g["live"]
    dk = np.diff(g["mid"])
    dp = np.diff(g["pm"])
    ok = live[1:]
    dk = np.where(ok, dk, np.nan)
    dp = np.where(ok, dp, np.nan)
    out = {}
    n = len(dk)
    for k in XCORR_LAGS:
        if abs(k) >= n:
            continue
        a = dk[k:] if k >= 0 else dk[:k]
        b = dp[:n - k] if k >= 0 else dp[-k:]
        m = np.isfinite(a) & np.isfinite(b)
        out[k] = (float((a[m] * b[m]).sum()), float((a[m] ** 2).sum()), float((b[m] ** 2).sum()), int(m.sum()))
    return out


def load_pm_paths(out: Path = OUT / "pm_paths", tokens: list[str] | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Polymarket points per token (only ``tokens`` when given: a lazy filtered scan keeps memory small)."""
    parts = sorted(out.glob("part-*.parquet"))
    if not parts:
        return {}
    q = pl.scan_parquet(parts)
    if tokens is not None:
        q = q.filter(pl.col("token").is_in(tokens))
    df = q.unique(["token", "t"]).sort("token", "t").collect()
    res = {}
    for (tok,), g in df.partition_by("token", as_dict=True).items():
        res[str(tok)] = (g["t"].to_numpy(), g["p"].to_numpy())
    return res


def load_kalshi_paths(out: Path = OUT / "kalshi_paths",
                      tickers: list[str] | None = None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    parts = sorted(out.glob("part-*.parquet"))
    if not parts:
        return {}
    q = pl.scan_parquet(parts).select("ticker", "end_ts", "bid_close", "ask_close")
    if tickers is not None:
        q = q.filter(pl.col("ticker").is_in(tickers))
    df = q.unique(["ticker", "end_ts"]).sort("ticker", "end_ts").collect()
    res = {}
    for (tk,), g in df.partition_by("ticker", as_dict=True).items():
        res[tk] = ((g["end_ts"].dt.epoch("s")).to_numpy(), g["bid_close"].fill_null(np.nan).to_numpy(),
                   g["ask_close"].fill_null(np.nan).to_numpy())
    return res


def run_pairs(pairs: pl.DataFrame, pm_paths: dict, k_paths: dict) -> dict[str, pl.DataFrame]:
    trades: list[dict[str, Any]] = []
    follows: list[dict[str, Any]] = []
    xc: dict[tuple[str, int], list[float]] = {}
    cover = []
    empty = np.array([], dtype=float)
    for pair in pairs.iter_rows(named=True):
        pm_t, pm_p = pm_paths.get(pair["token"], (np.array([], dtype=np.int64), empty))
        k_e, k_b, k_a = k_paths.get(pair["ticker"], (np.array([], dtype=np.int64), empty, empty))
        g = grid(pair, pm_t, pm_p, k_e, k_b, k_a)
        live = g["live"]
        both = live & np.isfinite(g["pm"]) & np.isfinite(g["mid"])
        cover.append({"pair_id": pair["pair_id"], "minutes": int(live.sum()), "pm_minutes": int((live & np.isfinite(g["pm"])).sum()),
                      "k_minutes": int((live & np.isfinite(g["mid"])).sum()), "both_minutes": int(both.sum()),
                      "pm_moves": int((np.abs(np.diff(g["pm"])) > 1e-9)[live[1:]].sum()),
                      "k_moves": int((np.abs(np.diff(g["mid"])) > 1e-9)[live[1:]].sum()),
                      "mean_abs_gap": float(np.nanmean(np.abs(g["pm"][both] - g["mid"][both]))) if both.any() else np.nan,
                      "median_spread": float(np.nanmedian((g["ask"] - g["bid"])[live])) if np.isfinite(g["ask"][live]).any() else np.nan})
        # measurement-aligned copy (who leads, not tradable): Polymarket's next-minute point (~4 s after T_i)
        # instead of the one ~56 s before it, so both venues are sampled within seconds of each other
        ga = {**g, "pm": np.append(g["pm"][1:], np.nan)}
        for grid_name, gg in (("tradable", g), ("aligned", ga)):
            for k, v in xcorr_sums(gg).items():
                acc = xc.setdefault((grid_name, pair["group"], k), [0.0, 0.0, 0.0, 0])
                for idx in range(4):
                    acc[idx] += v[idx]
        for leader in ("pm", "kalshi"):
            for x, n in ((XS[0], NS[0]), (XS[1], NS[1])):
                for f in follow_for(pair, ga, events_for(ga, x, n, leader), leader, n):
                    follows.append({"leader": leader, "x": x, "n_min": n, "grid": "aligned", **f})
        for leader in ("pm", "kalshi"):
            for x in XS:
                for n in NS:
                    ev = events_for(g, x, n, leader)
                    if not ev:
                        continue
                    tag = {"leader": leader, "x": x, "n_min": n}
                    lags = (0, 1, 2) if leader == "pm" else (1,)
                    for lag in lags:
                        for t in trades_for(pair, g, ev, lag=lag):
                            trades.append({**tag, "lag": lag, **t})
                    for f in follow_for(pair, g, ev, leader, n):
                        follows.append({**tag, "grid": "tradable", **f})
    xr = [{"grid": gr, "group": gname, "lag": k, "sxy": v[0], "sxx": v[1], "syy": v[2], "n": v[3]}
          for (gr, gname, k), v in xc.items()]
    tr = pl.DataFrame(trades, infer_schema_length=None) if trades else pl.DataFrame()
    if tr.height:
        et = pl.from_epoch("t_sig", time_unit="s").dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York")
        tr = tr.with_columns(et.dt.date().cast(pl.String).alias("date"),
                             pl.when(pl.col("t_sig") < int(HOLDOUT.timestamp())).then(pl.lit("dev")).otherwise(pl.lit("holdout"))
                             .alias("split"))
    fo = pl.DataFrame(follows, infer_schema_length=None) if follows else pl.DataFrame()
    if fo.height:
        et = pl.from_epoch("t_sig", time_unit="s").dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York")
        fo = fo.with_columns(et.dt.date().cast(pl.String).alias("date"),
                             pl.when(pl.col("t_sig") < int(HOLDOUT.timestamp())).then(pl.lit("dev")).otherwise(pl.lit("holdout"))
                             .alias("split"))
    return {"trades": tr, "follow": fo, "xcorr": pl.DataFrame(xr), "coverage": pl.DataFrame(cover)}


# ---- summaries, primary rule, ledger -------------------------------------------------------------
MIN_DEV_TRADES = 50
MIN_DEV_EVENTS = 15


CAPS = ("all", "spread<=X")    # execution filter: take only if Kalshi's spread at entry is at most the move X


def apply_cap(tr: pl.DataFrame, cap: str) -> pl.DataFrame:
    return tr if cap == "all" else tr.filter(pl.col("spread") <= pl.col("x") + 1e-9)


def cell_table(tr: pl.DataFrame, by: tuple[str, ...] = ("leader", "lag", "x", "n_min"), split: str | None = None,
               caps: tuple[str, ...] = CAPS) -> pl.DataFrame:
    """One row per cell x cap x horizon: trades, events, net and gross c/contract with the two-way t."""
    f = tr if split is None else tr.filter(pl.col("split") == split)
    rows = []
    for key, g0 in f.group_by(list(by), maintain_order=True):
        for cap in caps:
            g = apply_cap(g0, cap)
            if g.is_empty():
                continue
            for h in HORIZONS:
                c = cluster_t(g, f"pnl_{h}")
                gr = cluster_t(g, f"gross_{h}")
                kf = g[f"kfollow_{h}"].drop_nans().mean()
                fee = (g["fee_in"] / g["n"]).mean() + (g[f"fee_out_{h}"] / g["n"]).mean()
                rows.append({**dict(zip(by, key)), "cap": cap, "horizon": h, "trades": c["n"],
                             "events": c.get("events", 0), "dates": c.get("dates", 0), "net_c": 100 * c["mean"],
                             "t": c["t"], "gross_c": 100 * gr["mean"], "t_gross": gr["t"], "fee_c": 100 * float(fee),
                             "spread_c": 100 * float(g["spread"].mean()),
                             "kfollow_c": 100 * kf if kf is not None else None})
    return pl.DataFrame(rows)


def select_primary(dev_cells: pl.DataFrame) -> dict[str, Any]:
    """Pre-specified rule (dev only): Polymarket-led, 1-minute latency; over X x N x horizon x {all,
    spread<=X}, the cell with the highest two-way-clustered t of net PnL per contract among cells with
    >= 50 trades and >= 15 events (ties: more trades)."""
    c = dev_cells.filter((pl.col("leader") == "pm") & (pl.col("lag") == 1) & (pl.col("trades") >= MIN_DEV_TRADES)
                         & (pl.col("events") >= MIN_DEV_EVENTS) & pl.col("t").is_not_nan())
    if c.is_empty():
        raise RuntimeError("no dev cell meets the minimum size")
    best = c.sort(["t", "trades"], descending=True).row(0, named=True)
    return {k: best[k] for k in ("leader", "lag", "x", "n_min", "cap", "horizon", "trades", "events", "net_c", "t")}


def ledger(tr: pl.DataFrame, horizon: str, *, strategy: str, variant: str, per_event: float = 5.0,
           max_open: float = 20.0) -> pl.DataFrame:
    """Capped $100 book in time order: a trade is skipped when its cost would push the event's open
    cost above ``per_event`` or the total open cost above ``max_open`` (vol_metrics schema)."""
    rows = []
    open_: list[tuple[int, str, float]] = []
    for r in tr.sort("t_in", "pair_id").iter_rows(named=True):
        t_in = r["t_in"]
        open_ = [o for o in open_ if o[0] > t_in]
        cost = r["n"] * r["price"] + r["fee_in"]
        if sum(o[2] for o in open_ if o[1] == r["pm_event"]) + cost > per_event + 1e-9:
            continue
        if sum(o[2] for o in open_) + cost > max_open + 1e-9:
            continue
        t_out = r[f"t_out_{horizon}"]
        open_.append((t_out, r["pm_event"], cost))
        # event_ticker carries the real-world event (Polymarket event slug) so the top-5 check groups it
        rows.append({"strategy": strategy, "variant": variant, "split": r["split"],
                     "struct_id": f"{r['pair_id']}@{t_in}", "event_ticker": r["pm_event"], "ticker": r["ticker"],
                     "side": "yes" if r["dir"] > 0 else "no", "n": int(r["n"]),
                     "t_in": datetime.fromtimestamp(t_in, UTC), "cost_usd": cost,
                     "settle_ts": datetime.fromtimestamp(max(t_out, t_in), UTC),
                     "payoff_usd": r["n"] * r[f"exit_{horizon}"] - r[f"fee_out_{horizon}"], "group": r["group"]})
    return pl.DataFrame(rows)


def kalshi_marks(tickers: list[str], out: Path = OUT / "kalshi_paths") -> pl.DataFrame:
    """YES mids (ticker, ts, yes_mid) from the fetched minute candles, for the daily MTM in vol_metrics."""
    parts = sorted(out.glob("part-*.parquet"))
    c = pl.scan_parquet(parts).select("ticker", "end_ts", "bid_close", "ask_close").filter(
        pl.col("ticker").is_in(tickers)).collect()
    c = c.filter((pl.col("bid_close") > 0) & (pl.col("ask_close") < 1) & (pl.col("ask_close") > pl.col("bid_close")))
    return (c.select("ticker", pl.col("end_ts").alias("ts"), ((pl.col("bid_close") + pl.col("ask_close")) / 2).alias("yes_mid"))
            .unique(["ticker", "ts"]).sort("ticker", "ts"))


def concentration(tr: pl.DataFrame, col: str) -> dict[str, Any]:
    """Share of total PnL (per-contract sums weighted by contracts) in the top 5 events, and the mean
    and t without them."""
    f = tr.with_columns((pl.col(col) * pl.col("n")).alias("_usd"))
    ev = f.group_by("pm_event").agg(pl.col("_usd").sum()).sort("_usd", descending=True)
    total = float(ev["_usd"].sum())
    top = ev["pm_event"].head(5).to_list()
    ex = cluster_t(tr.filter(~pl.col("pm_event").is_in(top)), col)
    return {"total_usd": total, "top5_usd": float(ev["_usd"].head(5).sum()), "top5_events": top,
            "ex_top5_net_c": 100 * ex["mean"], "ex_top5_t": ex["t"], "ex_top5_trades": ex["n"]}


# ---- driver --------------------------------------------------------------------------------------
def step_universe(api, gamma) -> None:
    si = _series_info()
    if not (OUT / "pm_markets.parquet").exists():
        fetch_pm_events(gamma).write_parquet(OUT / "pm_markets.parquet")
    kalshi_shared(si).write_parquet(OUT / "kalshi_shared.parquet")
    if not (OUT / "kalshi_recent.parquet").exists():
        fetch_kalshi_recent(api, si, datetime(2026, 11, 1, tzinfo=UTC)).write_parquet(OUT / "kalshi_recent.parquet")


def step_hist(api) -> None:
    """Aug 1 .. 14 for the series of every family (selected by series identity, not by price/result)."""
    si = _series_info()
    k = kalshi_universe()
    series = sorted({s for s in k["series_ticker"].unique().to_list() if any(re.search(f["kalshi"], s) for f in FAMILIES)})
    print(f"historical Aug 1-14 for {len(series)} family series", flush=True)
    fetch_kalshi_series_hist(api, series, si, START, datetime(2026, 8, 15, 1, tzinfo=UTC)).write_parquet(
        OUT / "kalshi_hist.parquet")


def step_match() -> pl.DataFrame:
    p = pl.read_parquet(OUT / "pm_markets.parquet")
    k = kalshi_universe()
    m = match_all(k, p)
    pairs = build_pairs(m, k, p)
    pairs.write_parquet(OUT / "pairs.parquet")
    q = pairs.join(k.select("ticker", "title", "yes_sub_title", "rules_primary", "close_time"), on="ticker").join(
        p.select("condition_id", "question", "group_title", "description", "closed_time"), on="condition_id")
    q.write_parquet(OUT / "pairs_text.parquet")
    print(f"pairs: {pairs.height}; by family {dict(pairs.group_by('family').len().iter_rows())}", flush=True)
    return pairs


def bucket_pm_paths(tokens: list[str], size: int, out: Path = OUT / "pm_paths") -> Path:
    """Re-file the Polymarket points by integer token id bucket (``tokens`` order, ``size`` per bucket),
    one part at a time, so a study chunk reads only its own small files (a filtered scan of all parts
    decodes every token string: ~700 MB for 200 tokens)."""
    dest = out.parent / "pm_buckets"
    dest.mkdir(exist_ok=True)
    for f in dest.glob("*.parquet"):
        f.unlink()
    ids = pl.DataFrame({"token": tokens, "tid": np.arange(len(tokens), dtype=np.int32)})
    for k, part in enumerate(sorted(out.glob("part-*.parquet"))):
        df = pl.read_parquet(part).join(ids, on="token").select("tid", "t", "p").with_columns(
            (pl.col("tid") // size).alias("b"))
        for (b,), g in df.partition_by("b", as_dict=True).items():
            g.drop("b").write_parquet(dest / f"b{b:04d}-{k:05d}.parquet")
    return dest


def load_pm_bucket(dest: Path, b: int, tokens: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    files = sorted(dest.glob(f"b{b:04d}-*.parquet"))
    if not files:
        return {}
    df = pl.concat([pl.read_parquet(f) for f in files]).unique(["tid", "t"]).sort("tid", "t")
    return {tokens[tid]: (g["t"].to_numpy(), g["p"].to_numpy()) for (tid,), g in df.partition_by("tid", as_dict=True).items()}


def step_study(chunk: int = 60) -> None:
    """Pairs in chunks of Polymarket tokens: each chunk loads only its own price paths (bucketed by
    :func:`bucket_pm_paths`) and writes its trades/follow rows to part files; the cross-correlation sums
    are additive."""
    pairs = pl.read_parquet(OUT / "pairs.parquet").sort("token", "pair_id")
    tokens = pairs["token"].unique(maintain_order=True).to_list()
    dest = bucket_pm_paths(tokens, chunk)
    pairs = pairs.join(pl.DataFrame({"token": tokens, "tid": np.arange(len(tokens))}), on="token").with_columns(
        (pl.col("tid") // chunk).alias("b"))
    tmp = OUT / "study_parts"
    tmp.mkdir(exist_ok=True)
    for f in tmp.glob("*.parquet"):
        f.unlink()
    xcs, covs = [], []
    for c, ((b,), sub) in enumerate(sorted(pairs.partition_by("b", as_dict=True).items())):
        i = int(sub["tid"].min())
        res = run_pairs(sub.drop("tid", "b"), load_pm_bucket(dest, b, tokens),
                        load_kalshi_paths(tickers=sub["ticker"].unique().to_list()))
        for name in ("trades", "follow"):
            if res[name].height:
                res[name].write_parquet(tmp / f"{name}-{c:04d}.parquet")
        xcs.append(res["xcorr"])
        covs.append(res["coverage"])
        print(f"  study chunk {c}: tokens from {i}, pairs {sub.height}, trades {res['trades'].height}", flush=True)
    for name in ("trades", "follow"):                  # streamed: the full tables do not fit the memory budget
        parts = sorted(tmp.glob(f"{name}-*.parquet"))
        pl.scan_parquet(parts, missing_columns="insert", extra_columns="ignore").sink_parquet(OUT / f"{name}.parquet")
    xc = pl.concat([x for x in xcs if x.height]).group_by("grid", "group", "lag").agg(pl.col("sxy", "sxx", "syy", "n").sum())
    xc.write_parquet(OUT / "xcorr.parquet")
    pl.concat(covs).write_parquet(OUT / "coverage.parquet")
    print("study done", flush=True)


def load_trades(split: str | None = None) -> pl.DataFrame:
    """trades.parquet with the string keys as categoricals (memory), optionally one split only."""
    q = pl.scan_parquet(OUT / "trades.parquet")
    if split is not None:
        q = q.filter(pl.col("split") == split)
    cats = ["pair_id", "ticker", "event_ticker", "pm_event", "family", "group", "exact", "date", "split", "leader"]
    return q.drop("pm_move", "exact").with_columns(pl.col(c).cast(pl.Categorical) for c in cats if c != "exact").collect()


def step_select() -> dict[str, Any]:
    """Dev only: cell table on dev and the frozen primary rule (written once, never overwritten)."""
    tr = load_trades("dev")
    dev = cell_table(tr, split="dev")
    dev.write_csv(OUT / "cells_dev.csv")
    path = OUT / "primary.json"
    if path.exists():
        prim = json.loads(path.read_text())
        print(f"primary already frozen: {prim}", flush=True)
        return prim
    prim = select_primary(dev)
    prim["frozen_at"] = datetime.now(UTC).isoformat()
    prim["rule"] = select_primary.__doc__.strip()
    path.write_text(json.dumps(prim, indent=1))
    print(f"primary frozen: {prim}", flush=True)
    return prim


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.xvenue_leadlag")
    ap.add_argument("step", choices=["universe", "hist", "match", "stamp", "fetch-pm", "fetch-kalshi", "study", "select",
                                     "report", "all"])
    ap.add_argument("--stream-hour", nargs=3, metavar=("DAY", "HOUR", "CONN"), default=("2026-11-12", "3", "p0"))
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    steps = ["universe", "hist", "match", "fetch-pm", "fetch-kalshi", "study", "select", "report"] if a.step == "all" else [a.step]
    api = gamma = clob = None
    if any(s in steps for s in ("universe", "hist", "fetch-pm", "fetch-kalshi", "stamp")):
        api, gamma, clob = _clients()
    for s in steps:
        if s == "universe":
            step_universe(api, gamma)
        elif s == "hist":
            step_hist(api)
        elif s == "match":
            step_match()
        elif s == "stamp":
            step_stamp(clob, *a.stream_hour)
        elif s == "fetch-pm":
            fetch_pm_paths(clob, pl.read_parquet(OUT / "pairs.parquet"), OUT / "pm_paths")
        elif s == "fetch-kalshi":
            fetch_kalshi_paths(api, pl.read_parquet(OUT / "pairs.parquet"), OUT / "kalshi_paths")
        elif s == "study":
            step_study()
        elif s == "select":
            step_select()
        elif s == "report":
            step_report()


def step_stamp(clob, day: str, hour: str, conn: str, top: int = 12) -> pl.DataFrame:
    """What instant does a CLOB history point describe? Compare it with our recorded book (one hour)."""
    path = Path(f"data/raw/stream/pm/{day}/{int(hour):02d}-{conn}.jsonl.zst")
    q = pm_stream_quotes(path, top=top)
    lo = datetime.fromisoformat(f"{day}T{int(hour):02d}:00:00+00:00")
    rows = [r for a in q["asset"].unique().to_list() for r in pm_history(clob, a, lo, lo + timedelta(hours=1), chunk_days=1)]
    res = stamp_offsets(q, pl.DataFrame(rows), tuple(range(-60, 31, 2)))
    res.write_csv(OUT / f"stamp_{day}T{int(hour):02d}_{conn}.csv")
    best = res.sort("mae_c").row(0, named=True)
    print(f"stamp check {day} {hour}h {conn}: {q['asset'].n_unique()} tokens, best offset {best['offset_s']} s "
          f"(match {best['match_mid']:.2f}, MAE {best['mae_c']:.2f}c)", flush=True)
    return res



# ---- report --------------------------------------------------------------------------------------
def _md(df: pl.DataFrame, digits: int = 2) -> str:
    if df.is_empty():
        return "_(none)_\n"
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append("" if not math.isfinite(v) else f"{v:.{digits}f}")
            else:
                cells.append("" if v is None else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def xcorr_table(xc: pl.DataFrame, grid_name: str) -> pl.DataFrame:
    """Pooled correlation of 1-minute changes, Kalshi(t) vs Polymarket(t - lag), by group."""
    xc = xc.filter(pl.col("grid") == grid_name).drop("grid")
    allg = xc.group_by("lag").agg(pl.col("sxy", "sxx", "syy", "n").sum()).with_columns(pl.lit("all").alias("group"))
    x = pl.concat([xc, allg.select(xc.columns)])
    x = x.with_columns((pl.col("sxy") / (pl.col("sxx") * pl.col("syy")).sqrt()).alias("corr"))
    return x.pivot(on="lag", index="group", values="corr", sort_columns=True).sort("group")


def follow_table(fo: pl.DataFrame, x: float, n: int, grid_name: str = "aligned") -> pl.DataFrame:
    """Mean signed moves (c) after events of size >= x in n minutes: the other venue's follow and the
    leader's continuation, by leader and group, with the share of the gap the other venue closes."""
    f = fo.filter((pl.col("x") == x) & (pl.col("n_min") == n) & (pl.col("grid") == grid_name)).fill_nan(None)
    rows = []
    for (leader, grp), g in f.group_by("leader", "group", maintain_order=True):
        r = {"leader": leader, "group": grp, "events": g.height, "move_c": 100 * float(g["move"].mean() or 0),
             "gap_c": 100 * float(g["gap"].mean() or 0)}
        for h in FOLLOW_H:
            r[f"other_{h}m_c"] = 100 * float(g[f"other_{h}"].mean() or 0)
        r["lead_15m_c"] = 100 * float(g["lead_15"].mean() or 0)
        r["lead_60m_c"] = 100 * float(g["lead_60"].mean() or 0)
        c = cluster_t(g, "other_5")
        r["t_other_5m"] = c["t"]
        rows.append(r)
    return pl.DataFrame(rows).sort("leader", "group") if rows else pl.DataFrame()


# ---- elections (Nov 3 2026 midterms) -------------------------------------------------------------
STATES = {"AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california", "CO": "colorado",
          "CT": "connecticut", "DE": "delaware", "FL": "florida", "GA": "georgia", "HI": "hawaii", "ID": "idaho",
          "IL": "illinois", "IN": "indiana", "IA": "iowa", "KS": "kansas", "KY": "kentucky", "LA": "louisiana",
          "ME": "maine", "MD": "maryland", "MA": "massachusetts", "MI": "michigan", "MN": "minnesota",
          "MS": "mississippi", "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
          "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york", "NC": "north carolina",
          "ND": "north dakota", "OH": "ohio", "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania",
          "RI": "rhode island", "SC": "south carolina", "SD": "south dakota", "TN": "tennessee", "TX": "texas",
          "UT": "utah", "VT": "vermont", "VA": "virginia", "WA": "washington", "WV": "west virginia",
          "WI": "wisconsin", "WY": "wyoming", "DC": "district of columbia"}
STATE_CODE = {v: k for k, v in STATES.items()}


def _district(x: str) -> int:
    return 0 if x.upper() in ("AL", "00") else int(x)


def _house_k(r: dict[str, Any]) -> tuple | None:
    m = re.match(r"^(?:KXHOUSERACE-|HOUSE)([A-Z]{2})(\d{1,2}|AL)-26-([RD])$", r["ticker"])
    return ("house", m.group(1), _district(m.group(2))) if m else None


def _house_p(r: dict[str, Any]) -> tuple | None:
    m = re.match(r"^([a-z]{2})-(\d{2}|al)-house-election-winner(?:-\d+)?$", r["event_slug"] or "")
    return ("house", m.group(1).upper(), _district(m.group(2))) if m else None


def _party_k(r: dict[str, Any]) -> list[str]:
    m = re.search(r"-([RD])$", r["ticker"])
    return [m.group(1)] if m else []


def _party_p(r: dict[str, Any]) -> list[str]:
    q = (r.get("question") or "").lower()
    if "another party" in q or "other" == (r.get("group_title") or "").lower():
        return []
    if "republican" in q:
        return ["R"]
    if "democrat" in q:
        return ["D"]
    return []


def _state_from(text: str | None) -> str | None:
    t = norm_name(text)
    for name in sorted(STATE_CODE, key=len, reverse=True):
        if re.search(r"(^| )" + name + r"( |$)", t):
            return STATE_CODE[name]
    return None


def _gov_k(r: dict[str, Any]) -> tuple | None:
    m = re.match(r"^(?:KXGOV|GOVPARTY)([A-Z]{2})-26-", r["ticker"])
    return ("gov", m.group(1)) if m and m.group(1) in STATES else None


def _gov_p(r: dict[str, Any]) -> tuple | None:
    m = re.match(r"^([a-z-]+)-governor-election-winner(?:-\d+)?$", r["event_slug"] or "")
    st = STATE_CODE.get(m.group(1).replace("-", " ")) if m else None
    return ("gov", st) if st else None


def _senate_k(r: dict[str, Any]) -> tuple | None:
    m = re.match(r"^(?:KX)?SENATE([A-Z]{2})(S?)-26-([RD])$", r["ticker"])
    if not m or m.group(1) not in STATES:
        return None
    return ("senate", m.group(1), "special" if m.group(2) or "special" in (r["title"] or "").lower() else "regular")


def _senate_p(r: dict[str, Any]) -> tuple | None:
    m = re.match(r"^([a-z-]+?)-senate-(special-)?election-winner(?:-\d+)?$", r["event_slug"] or "")
    st = STATE_CODE.get(m.group(1).replace("-", " ")) if m else None
    special = bool(m and m.group(2)) or "special" in (r.get("question") or "").lower()
    return ("senate", st, "special" if special else "regular") if st else None


FAMILIES += [
    {"name": "house_party", "group": "elections", "kalshi": r"^(KXHOUSERACE|HOUSE[A-Z]{2}\d{0,2}|HOUSE[A-Z]{2}AL)$",
     "pm": r"^[a-z]{2}-(\d{2}|al)-house-election-winner", "k_key": _house_k, "p_key": _house_p,
     "k_names": _party_k, "p_names": _party_p, "min_name": 1.0,
     "note": "party of the winner of the district (Kalshi: party of the member sworn in, accelerated determination "
             "on a media consensus; PM: consensus of credible reporting)"},
    {"name": "governor", "group": "elections", "kalshi": r"^(KXGOV[A-Z]{2}|GOVPARTY[A-Z]{2})$",
     "pm": r"^[a-z-]+-governor-election-winner", "k_key": _gov_k, "p_key": _gov_p,
     "k_names": lambda r: [r["yes_sub_title"]], "p_names": lambda r: [_person(pm_outcome(r))],
     "exact_fn": lambda r: "near" if r["series_ticker"].startswith("GOVPARTY") else "exact",
     "note": "2026 governor winner by candidate (Kalshi GOVPARTY markets name the party's candidate: near-exact)"},
    {"name": "senate_party", "group": "elections", "kalshi": r"^(KX)?SENATE[A-Z]{2}S?$",
     "pm": r"^[a-z-]+-senate-(special-)?election-winner", "k_key": _senate_k, "p_key": _senate_p,
     "k_names": _party_k, "p_names": _party_p, "min_name": 1.0,
     "note": "party of the winner of the state's 2026 Senate race (regular vs special kept apart)"},
]


# Families read and rejected (resolution text differs), recorded for the report.
REJECTED = [
    ("Truth Social post counts", "KXTRUTHSOCIAL / donald-trump-of-truth-social-posts",
     "Kalshi Sun-Sat week vs Polymarket Tue noon - Tue noon ET (xtracker counter)"),
    ("Trump says X before date", "KXTRUMPSAY / what-will-trump-say-this-week",
     "Kalshi window starts Sun 8 pm ET, Polymarket Sun 00:00 ET; PM excludes written usages"),
    ("Rotten Tomatoes", "KXRT / *-rotten-tomatoes-score", "Kalshi 'above X' (> X) vs Polymarket 'at least X' (>= X)"),
    ("Top AI model", "KXTOPMODEL / best-ai-model-on", "10:00 vs 12:00 ET check; Kalshi leaderboard setting differs"),
    ("Text-to-image/video AI", "KXIMAGEAI, KXVIDEOAI / which-company-has-the-best-*",
     "Artificial Analysis vs arena.ai; different dates"),
    ("Chinese AI company", "KXCHINAAI / second-best-chinese-ai-company", "first vs second place; different dates"),
    ("RCP approval", "KXAPRPOTUS / trump-approval-rating-on", "11:00 vs 17:00 ET snapshot; 'above' vs 'at least'"),
    ("Album sales", "KXALBUMEQUIV, KXPUREALBUMS / *-first-week-album-sales", "Luminate units vs Hits Daily Double"),
    ("AGT", "KXAGTELIMINATION / americas-got-talent-*-advances", "'eliminated before D' is not the complement of 'advances'"),
    ("OpenRouter tokens", "KXTOKENUSE / openrouter-token-usage", "thresholds vs brackets, no common contract"),
    ("Trump mention 'B' series", "KXTRUMPMENTIONB", "conditional scopes ('any UN event excluding ...', 'next livestream')"),
]


def _ledger_stats(led: pl.DataFrame) -> dict[str, Any]:
    from albert.research.vol_metrics import ledger_metrics

    if led.is_empty():
        return {}
    marks = kalshi_marks(led["ticker"].unique().to_list())
    return ledger_metrics(led, marks)


def step_report() -> None:
    pairs = pl.read_parquet(OUT / "pairs.parquet")
    text = pl.read_parquet(OUT / "pairs_text.parquet")
    tr = load_trades()
    fo = pl.read_parquet(OUT / "follow.parquet")
    xc = pl.read_parquet(OUT / "xcorr.parquet")
    cov = pl.read_parquet(OUT / "coverage.parquet")
    prim = json.loads((OUT / "primary.json").read_text())
    h = prim["horizon"]
    cell = (pl.col("leader") == prim["leader"]) & (pl.col("lag") == prim["lag"]) & (pl.col("x") == prim["x"]) & (
        pl.col("n_min") == prim["n_min"])
    pt = apply_cap(tr.filter(cell), prim["cap"])
    hold = cell_table(tr, split="holdout")
    hold.write_csv(OUT / "cells_holdout.csv")
    dev = pl.read_csv(OUT / "cells_dev.csv")
    L: list[str] = []
    add = L.append
    add("# Does Polymarket intl lead Kalshi in non-sports markets? (xvenue_leadlag)\n")
    add(f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `python -m albert.research.xvenue_leadlag report`. "
        "Research only; Polymarket intl is a data source (no orders anywhere).\n")
    if (OUT / "verdict.md").exists():
        add((OUT / "verdict.md").read_text())
    add("Data sources: Kalshi public API (`/events`, `/markets`, `/historical/markets`, minute candlesticks; <= 1 req/s, "
        "sequential); Polymarket Gamma `/events/keyset` and CLOB `/prices-history` (<= 2 req/s; "
        "robots.txt re-checked 2026-11-17: `gamma-api.polymarket.com` and `clob.polymarket.com` serve none (HTTP 404), "
        "so nothing is disallowed; no other host used); our own recorded Polymarket WebSocket books for the clock "
        "check. Every response is cached raw under `data/raw/{kalshi,pm_gamma,pm_clob}`.\n")
    # --- sample
    add("## 1. Sample and match quality\n")
    agree = text.with_columns((pl.col("k_result") == pl.col("pm_result")).alias("agree"))
    fam = agree.group_by("group", "family", "exact").agg(pl.len().alias("pairs"), pl.col("pm_event_slug").n_unique().alias(
        "pm_events"), pl.col("agree").sum().alias("settle_agree")).sort("group", "family")
    add(f"{pairs.height} exactly matched pairs (Kalshi YES = Polymarket 'Yes'), {pairs['pm_event_slug'].n_unique()} "
        f"real-world events, Kalshi markets settled {START:%Y-%m-%d} .. {END:%Y-%m-%d}; settlement agrees on "
        f"{int(agree['agree'].sum())}/{agree.height}. 'near' = same outcome, an edge-case rule differs.\n")
    add(_md(fam))
    add("\nRejected after reading both venues' rules:\n")
    add(_md(pl.DataFrame(REJECTED, schema=["family", "series / slug", "why"], orient="row")))
    sample = text.sample(min(12, text.height), seed=20261117).sort("family")
    add("\nSpot-check sample (seeded, 12 pairs): Kalshi title | Polymarket question | results\n")
    add(_md(sample.select("family", pl.col("ticker"), pl.col("title").str.slice(0, 70).alias("kalshi"),
                          pl.col("yes_sub_title").str.slice(0, 30).alias("k_outcome"),
                          pl.col("question").str.slice(0, 80).alias("polymarket"), "k_result", "pm_result")))
    add("\nResolution text side by side (one pair per group, seeded):\n")
    for (grp,), g in text.sort("pair_id").group_by("group", maintain_order=True):
        r = g.sample(1, seed=7).row(0, named=True)
        add(f"* **{grp} / {r['family']}** `{r['ticker']}`: Kalshi: \"{norm_text(r['rules_primary'])[:260]}\" | "
            f"Polymarket: \"{norm_text(r['description'])[:420]}\"")
    add("")
    c = cov.join(pairs.select("pair_id", "group"), on="pair_id").with_columns(
        pl.col("mean_abs_gap", "median_spread").fill_nan(None)).group_by("group").agg(
        pl.len().alias("pairs"), pl.col("both_minutes").sum().alias("minutes_both_quoted"),
        pl.col("pm_moves").sum(), pl.col("k_moves").sum(), pl.col("mean_abs_gap").median().alias("median_|PM-K|"),
        pl.col("median_spread").median().alias("median_K_spread")).sort("group")
    add("\nPrice-path coverage (live window; moves = minutes with a mid change):\n")
    add(_md(c, 3))
    # --- stamp
    add("\n## 2. Clocks\n")
    for f in sorted(OUT.glob("stamp_*.csv")):
        st = pl.read_csv(f).sort("mae_c").row(0, named=True)
        add(f"* `{f.name}`: CLOB point stamped t matches our recorded book mid at t {st['offset_s']:+d} s "
            f"({100 * st['match_mid']:.0f}% within 0.5c, MAE {st['mae_c']:.2f}c).")
    if (OUT / "pm_price_semantics.csv").exists():
        ps = pl.read_csv(OUT / "pm_price_semantics.csv")
        add(f"* What a point is: in {ps.height} open markets checked live ({int(ps['wide'].sum())} with spread > 10c, "
            f"last trade >= 2c from the mid), the latest point equals the book mid in {int(ps['is_mid'].sum())} and "
            f"the last trade in {int(ps['is_last'].sum())}: the history is a mid series even in wide books.")
    add("* Grid minute i ends at T_i. Polymarket = last point stamped <= T_i (it shows the book ~14 s earlier); "
        "Kalshi = close of the last candle ending <= T_i. A signal at T_i trades at Kalshi's quote at T_(i+1) (lag 1), "
        "so the Polymarket information is 60-120 s old at the fill.\n")
    # --- lead
    add("## 3. Who leads\n")
    add("Pooled correlation of 1-minute mid changes, Kalshi(t) vs Polymarket(t - lag) (lag > 0: Polymarket first). "
        "**Aligned** grid (for this question only): Polymarket's next-minute point, which shows the book ~4 s after "
        "Kalshi's candle close, so both venues are sampled within seconds. On the **tradable** grid Polymarket is "
        "sampled ~56 s before Kalshi, which shifts simultaneous moves to lag -1.\n")
    for gname in ("aligned", "tradable"):
        add(f"\n{gname}:\n")
        add(_md(xcorr_table(xc, gname), 3))
    for x, n in ((XS[0], NS[0]), (XS[1], NS[1])):
        ft = follow_table(fo, x, n, "aligned")
        add(f"\nAligned grid. After a >= {100 * x:.0f}c move in {n} min on the leader that the other venue had not "
            "matched (< half): the leader's move and the gap it left (c), the other venue's signed move h minutes "
            "later, the leader's own continuation (negative = reversal), and t of the 5-minute follow:\n")
        add(_md(ft))
    # --- primary
    add("\n## 4. Primary rule (frozen on dev before the holdout was computed)\n")
    add(f"`primary.json` ({prim['frozen_at']}): leader {prim['leader']}, latency {prim['lag']} min, X = "
        f"{100 * prim['x']:.0f}c, N = {prim['n_min']} min, filter {prim['cap']}, exit {h}. Rule: {prim['rule']}\n")
    rows = []
    stats = {}
    for split in ("dev", "holdout"):
        f = pt.filter(pl.col("split") == split)
        ct = cluster_t(f, f"pnl_{h}")
        gr = cluster_t(f, f"gross_{h}")
        led = ledger(f, h, strategy="xvenue_leadlag", variant=f"pm{100 * prim['x']:.0f}c_{prim['n_min']}m_{h}_{split}")
        if not led.is_empty():
            (OUT / "ledgers").mkdir(exist_ok=True)
            led.write_parquet(OUT / "ledgers" / f"ledger_primary_{split}.parquet")
        m = _ledger_stats(led)
        stats[split] = m
        conc = concentration(f, f"pnl_{h}") if f.height else {}
        rows.append({"split": split, "trades": ct["n"], "events": ct.get("events", 0), "net_c": 100 * ct["mean"],
                     "t": ct["t"], "gross_c": 100 * gr["mean"], "book_trades": m.get("positions"),
                     "sharpe": m.get("sharpe"), "sharpe_ci90": str(m.get("sharpe_ci90")),
                     "sharpe_ex_top5": m.get("sharpe_ex_top5"), "usd_week": m.get("usd_per_week"),
                     "ann_vol_pct": m.get("ann_vol_pct"), "max_dd": m.get("max_dd_usd"),
                     "pnl_usd_all_trades": conc.get("total_usd"), "pnl_usd_top5_events": conc.get("top5_usd"),
                     "ex_top5_net_c": conc.get("ex_top5_net_c"), "ex_top5_t": conc.get("ex_top5_t")})
    add(_md(pl.DataFrame(rows)))
    (OUT / "primary_stats.json").write_text(json.dumps({"rows": rows, "ledger": stats}, indent=1, default=str))
    add("\nnet_c / t: per contract after taker fees (rounded up per order), two-way clustered by real-world event and "
        "ET date. Ledger: $100 book, <= $5 per trade and per event, <= $20 open, daily MTM at the Kalshi mid "
        "(`vol_metrics`).\n")
    add("By category (primary rule):\n")
    bc = []
    for (grp, split), g in pt.group_by("group", "split", maintain_order=True):
        ct = cluster_t(g, f"pnl_{h}")
        bc.append({"group": grp, "split": split, "trades": ct["n"], "events": ct.get("events", 0),
                   "net_c": 100 * ct["mean"], "t": ct["t"], "gross_c": 100 * float(g[f"gross_{h}"].mean())})
    add(_md(pl.DataFrame(bc).sort("group", "split")))
    bf = []
    for (fam, split), g in pt.group_by("family", "split", maintain_order=True):
        ct = cluster_t(g, f"pnl_{h}")
        bf.append({"family": fam, "split": split, "trades": ct["n"], "events": ct.get("events", 0),
                   "net_c": 100 * ct["mean"], "t": ct["t"], "spread_c": 100 * float(g["spread"].mean())})
    add("\nBy family (primary rule; post hoc, for context):\n")
    add(_md(pl.DataFrame(bf).sort("family", "split")))
    gp = []
    for (big, split), g in pt.with_columns((pl.col("gap_in") >= pl.col("x") / 2).fill_null(False).alias("gap_open")).group_by(
            "gap_open", "split", maintain_order=True):
        ct = cluster_t(g, f"pnl_{h}")
        gp.append({"pm_minus_kalshi_gap>=X/2": big, "split": split, "trades": ct["n"], "net_c": 100 * ct["mean"],
                   "t": ct["t"], "kfollow_c": 100 * float(g[f"kfollow_{h}"].drop_nans().mean() or 0)})
    add("\nDid the Polymarket move open a cross-venue gap (PM mid - Kalshi mid >= X/2 in the signal's direction)? "
        "(post hoc):\n")
    add(_md(pl.DataFrame(gp).sort("pm_minus_kalshi_gap>=X/2", "split")))
    add("\nLatency sensitivity (same cell; lag 0 = Kalshi quote 0-60 s after the Polymarket snapshot, optimistic):\n")
    lat = tr.filter((pl.col("leader") == "pm") & (pl.col("x") == prim["x"]) & (pl.col("n_min") == prim["n_min"]))
    add(_md(cell_table(lat, by=("lag", "split"), caps=(prim["cap"],)).filter(pl.col("horizon") == h)))
    # --- grids
    add("\n## 5. All cells (Polymarket-led, lag 1): dev, then holdout (context only, not pre-specified)\n")
    for name, df in (("dev", dev), ("holdout", hold)):
        g = df.filter((pl.col("leader") == "pm") & (pl.col("lag") == 1)).select(
            "x", "n_min", "cap", "horizon", "trades", "events", "net_c", "t", "gross_c", "fee_c", "spread_c", "kfollow_c")
        add(f"\n{name}:\n")
        add(_md(g))
    # --- control
    add("\n## 6. Control: Kalshi leads Polymarket\n")
    add("Same event definition with the venues swapped; the Kalshi taker trades in the Kalshi move's direction "
        "(lag 1). If Polymarket led, unconfirmed Kalshi moves should revert (negative).\n")
    for name, split in (("dev", "dev"), ("holdout", "holdout")):
        g = cell_table(tr.filter(pl.col("leader") == "kalshi"), split=split, caps=(prim["cap"],)).filter(
            (pl.col("x") == prim["x"]) & (pl.col("n_min") == prim["n_min"]))
        add(f"\n{name} (X, N and filter of the primary rule):\n")
        add(_md(g.select("horizon", "trades", "events", "net_c", "t", "gross_c", "kfollow_c")))
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print(f"report: {OUT / 'report.md'}", flush=True)


def pm_price_semantics(gamma, clob, *, per_bucket: int = 20, tag: str = "politics") -> pl.DataFrame:
    """Live check of what a CLOB history point is in a wide book: for open markets with a spread above
    and below 10c (Gamma ``bestBid``/``bestAsk``), the latest 1-minute point vs the live book's mid and
    the last trade (Polymarket displays the last trade when the spread exceeds 10c)."""
    page = gamma.get("/events/keyset", {"closed": "false", "limit": 100, "tag_slug": tag}, cache=False)
    rows = []
    for e in page.payload.get("events") or []:
        for m in e.get("markets") or []:
            b, a, lt = m.get("bestBid"), m.get("bestAsk"), m.get("lastTradePrice")
            toks = _json_list(m.get("clobTokenIds"))
            if b is None or a is None or lt is None or not toks or not m.get("acceptingOrders", True):
                continue
            rows.append({"token": str(toks[0]), "bid": float(b), "ask": float(a), "last": float(lt)})
    df = pl.DataFrame(rows).with_columns((pl.col("ask") - pl.col("bid")).alias("spread"))
    df = df.filter((pl.col("bid") > 0) & (pl.col("ask") < 1) & ((pl.col("last") - (pl.col("bid") + pl.col("ask")) / 2).abs() > 0.02))
    pick = pl.concat([df.filter(pl.col("spread") > 0.10).head(per_bucket), df.filter(pl.col("spread") <= 0.10).head(per_bucket)])
    out = []
    now = int(time.time())
    for r in pick.iter_rows(named=True):
        book = clob.get("/book", {"token_id": r["token"]}, cache=False).payload
        bids = [float(x["price"]) for x in book.get("bids") or []]
        asks = [float(x["price"]) for x in book.get("asks") or []]
        lt = clob.get("/last-trade-price", {"token_id": r["token"]}, cache=False).payload
        h = clob.get("/prices-history", {"market": r["token"], "startTs": now - 1800, "endTs": now, "fidelity": 1},
                     cache=False).payload.get("history") or []
        if not bids or not asks or not h:
            continue
        mid = (max(bids) + min(asks)) / 2
        out.append({"spread": min(asks) - max(bids), "mid": mid, "last": float(lt.get("price") or "nan"),
                    "hist": float(h[-1]["p"]), "hist_age_s": now - int(h[-1]["t"])})
    res = pl.DataFrame(out).with_columns((pl.col("spread") > 0.10).alias("wide"),
                                         ((pl.col("hist") - pl.col("mid")).abs() <= 0.005).alias("is_mid"),
                                         ((pl.col("hist") - pl.col("last")).abs() <= 0.005).alias("is_last"))
    return res


if __name__ == "__main__":
    main()
