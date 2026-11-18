"""Mentions extensions: can the pre-speech NO-bid edge be sharpened or extended?

The one robust edge (``mentions_history``; live in ``live_mentions``): one post-only NO bid per
quoted word leg of a Kalshi Mentions speech event at ``floor(1 - fair - 12c)``, ``fair =
sigma(-0.1156 + 1.0474 logit(mid))``, resting from start - 6 h to start - 10 min, circuit breaker on
a >= 95c print of a leg asked <= 80c at posting, held to settlement. This study asks, in order:

1. **Salience filter.** Does public attention to a leg's word before the speech (Wikimedia daily
   pageviews of the word's article, z-score of the last published days against the word's own
   prior 30 days) predict which NO-bid fills lose? Feature and threshold chosen on dev only.
   (GDELT DOC 2.0 was the first choice; its API host has no robots.txt but answered HTTP 429 to
   every request from this machine during the study, so it is not used.)
2. **Entry window and margin.** Post at start - L (L in 6/12/24/48 h) at margin m (8-20c), either
   holding that order to start - 10 min (``hold``) or re-pricing at start - 6 h with the live rule
   (``extend``: the early order rests until start - 6 h; an unfilled leg then gets the live order);
   plus ``late``: the live rule and, for events listed after start - 6 h (which the bot never
   posts), a bid at the end of the first full hour after listing. Whole grid on dev; holdout only
   for the dev-chosen cell (rule: :func:`choose_cell`) and the live rule.
3. **Post-speech NO sweep.** After an as-of "speech over" signal, buy NO as a taker at the ask
   (1 - YES bid) on legs whose YES bid is still in [2c, cap], taker fee rounded up per order,
   held to settlement. Wilson lower bound on the win rate (events as n) against breakeven.
4. **Polymarket intl mention markets** (data only): the Polymarket YES price of the same speech
   and word as of the decision vs the Kalshi mid, as a skip filter on the live rule's fills.

Conventions (``AGENTS.md``): dev = events starting before 2026-10-01, holdout = 2026-10-01 to
2026-11-16 (the last settled data); no look-ahead (every input is timestamped before the decision);
maker fills only on prints strictly through the bid (hourly candle trade high above our ask, checked
against the public prints ``mentions_history`` cached); Mentions series are ``quadratic`` (taker fee
only, ``0.07 x mult x n x P(1 - P)`` rounded up to the cent per order); t-stats clustered by event
and by ET date of the start; capped-book ledgers ($15 value at risk, 2 events at once, $15 daily
fill cap, as the live bot runs) scored with ``vol_metrics.ledger_metrics`` on the $100 bankroll.

Data: cached Kalshi responses of earlier studies (``mentions_history`` markets / milestones /
candles / prints, ``xcat`` whole-life hourly and first-3-days minute candles, ``pre_event`` hourly
and the November forward listing); new public Kalshi requests (<= 1 / s, sequential, cached under
``data/cache/mentions_ext/raw``) only for hourly candles back to start - 51 h where missing and for
minute candles after the start (and over the late-listed events' windows); Wikimedia REST
pageviews (wikimedia.org has no robots.txt; ~10 requests / min allowed); Polymarket gamma events
and data-api price history (no robots.txt on either host). Outputs in ``data/research/mentions_ext``
(``report.md``, tables as CSV, ``ledgers/``).

    python -m albert.research.mentions_ext universe
    python -m albert.research.mentions_ext fetch-hourly|fetch-minute|fetch-late|fetch-pageviews|fetch-pm
    python -m albert.research.mentions_ext run
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import polars as pl

OUT = Path("data/research/mentions_ext")
CACHE = Path("data/cache/mentions_ext")
MH = Path("data/cache/mentions_history")
PRE = Path("data/research/pre_event")
XCAT = Path("data/research/xcat")
TABLES = Path("data/tables")

ET = "America/New_York"
FROZEN_A, FROZEN_B = -0.1156, 1.0474
LIVE_L, LIVE_M = 6, 0.12
LEADS = (6, 12, 24, 48)
MARGINS = (0.08, 0.10, 0.12, 0.15, 0.20)
END_BEFORE = timedelta(minutes=10)
QUOTE_AGE = timedelta(hours=24)
SAID_BID, SAID_JUMP, SAID_LOOKBACK = 0.97, 0.12, timedelta(hours=18)
CB_PRINT, CB_ASK = 0.95, 0.80
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
DATA_FROM, DATA_TO = datetime(2025, 3, 1, tzinfo=UTC), datetime(2026, 11, 16, 12, tzinfo=UTC)
GRID_FROM = datetime(2026, 8, 17, tzinfo=UTC)   # whole-life hourly candles (48 h entries) from here
HOURLY_BACK = timedelta(hours=51)
VAR, MAX_EVENTS, DAILY_CAP = 15.0, 2, 15.0       # deploy/mentions-prod.conf
EPS = 1e-9


# ---- small helpers --------------------------------------------------------------------------------
def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _strs(values) -> pl.Series:
    return pl.Series(sorted(set(values)), dtype=pl.String).implode()


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0.005, 0.995)
    return np.log(p / (1 - p))


def fair_cal(mid: np.ndarray) -> np.ndarray:
    """Frozen round-2 calibrated mid (the live rule's fair value)."""
    return np.clip(1 / (1 + np.exp(-(FROZEN_A + FROZEN_B * logit(mid)))), 0.003, 0.997)


def no_bid(fair: np.ndarray, bid: np.ndarray, margin: float) -> np.ndarray:
    """``floor(1 - fair - m)`` capped one tick inside the YES bid (post-only); NaN under 1c."""
    fair, bid = np.asarray(fair, dtype=float), np.asarray(bid, dtype=float)
    with np.errstate(invalid="ignore"):
        nb = np.minimum(np.floor((1 - fair - margin) * 100 + 1e-9) / 100, np.round(1 - bid - 0.01, 2))
        return np.where(np.isfinite(nb) & (nb >= 0.01 - EPS), np.round(nb, 2), np.nan)


def taker_fee(price: float, n: int = 1, mult: float | None = 1.0) -> float:
    """Kalshi taker fee for one order, rounded up to the cent (``albert.fees``)."""
    from albert.fees import kalshi_fee

    return kalshi_fee(n, price, maker=False, multiplier=1.0 if mult is None else mult, conservative=True)


def wilson_lower(k: float, n: float, z: float = 1.6449) -> float:
    """One-sided 95% Wilson lower bound of a binomial share ``k / n``."""
    if n <= 0:
        return float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - rad) / den


def clustered_t(x: np.ndarray, g: np.ndarray) -> tuple[float, float | None, int]:
    from albert.research.fundamental import clustered_mean

    return clustered_mean(np.asarray(x, dtype=float), np.asarray(g))


def et_date(col: str = "start") -> pl.Expr:
    return pl.col(col).dt.convert_time_zone(ET).dt.date()


# ---- universe -------------------------------------------------------------------------------------
MARKET_COLS = ["ticker", "event_ticker", "series_ticker", "title", "yes_sub_title", "open_time", "close_time",
               "settlement_ts", "volume", "result", "fee_type", "fee_multiplier"]


def _markets() -> pl.DataFrame:
    """Settled Mentions markets: ``mentions_history`` listing (closes to 2026-11-05) plus the
    ``pre_event`` November forward listing (events with a November Mentions milestone)."""
    a = pl.read_parquet(MH / "markets.parquet").select(MARKET_COLS)
    frames = [a]
    p = PRE / "forward" / "kalshi_markets.parquet"
    if p.exists():
        frames.append(pl.scan_parquet(p).filter(pl.col("category") == "Mentions").select(MARKET_COLS).collect())
    return pl.concat(frames, how="vertical_relaxed").unique("ticker", keep="first")


def _starts() -> pl.DataFrame:
    """Earliest Mentions milestone start per event (``mentions_history`` first, then the forward
    milestones for events it lacks)."""
    a = pl.read_parquet(MH / "starts.parquet").select("event", "start", "ms_updated")
    frames = [a]
    p = PRE / "forward" / "milestones.parquet"
    if p.exists():
        b = (pl.read_parquet(p).filter(pl.col("ms_category") == "Mentions").sort("start")
             .unique("event", keep="first").select("event", "start", "ms_updated"))
        frames.append(b.filter(~pl.col("event").is_in(a["event"].implode())))
    return pl.concat(frames, how="vertical_relaxed")


def financial_series(series: list[str]) -> set[str]:
    """Series the live bot's finance filter skips (``categories.kalshi_series_is_financial`` with
    the Mentions category and the series title)."""
    from albert.categories import kalshi_series_is_financial

    titles = dict(pl.read_parquet(TABLES / "kalshi_series.parquet").select("series_ticker", "title").iter_rows())
    return {s for s in series if kalshi_series_is_financial(s, "Mentions", titles.get(s))}


def word_titles(sub_title: str | None) -> list[str]:
    """Wikipedia article guesses for a leg: each ``/`` alternative of the subtitle with its own
    capitalisation (``"Marco / Rubio (3+ times)"`` -> ``["Marco", "Rubio"]``), spaces as ``_``."""
    s = re.sub(r"\(\d+\+\s*times?\)", "", sub_title or "", flags=re.IGNORECASE).strip()
    out = []
    for a in s.split("/"):
        a = " ".join(a.split())
        if a:
            out.append((a[0].upper() + a[1:]).replace(" ", "_"))
    return sorted(set(out))


def universe() -> pl.DataFrame:
    """Settled word legs of Mentions speech events (window series, earnings calls and the bot's
    finance-filtered series excluded) starting in [2025-03-01, 2026-11-16 12:00 UTC)."""
    from albert.research import mentions as M
    from albert.research import mentions_history as H

    mk = _markets()
    lg = M.legs(mk).filter(~pl.col("series").is_in(M.WINDOW_SERIES))
    lg = lg.join(_starts(), on="event").join(mk.select("ticker", "yes_sub_title", "fee_type", "fee_multiplier"), on="ticker")
    lg = lg.with_columns(H.speaker_group(pl.col("series")).alias("sgroup"))
    fin = financial_series(lg["series"].unique().to_list())
    lg = lg.filter((pl.col("sgroup") != "earnings") & ~pl.col("series").is_in(_strs(fin) if fin else _strs([""])))
    lg = lg.filter((pl.col("start") >= DATA_FROM) & (pl.col("start") < DATA_TO) & pl.col("close_time").is_not_null())
    return lg.with_columns(
        pl.when(pl.col("start") < HOLDOUT_FROM).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("split"),
        et_date("start").alias("et_date"),
        pl.col("yes_sub_title").map_elements(lambda x: "|".join(word_titles(x)), return_dtype=pl.String).alias("wiki"),
    ).sort("start", "event", "ticker")


# ---- hourly candles -------------------------------------------------------------------------------
HSCHEMA = {"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "bid": pl.Float64, "ask": pl.Float64,
           "hi": pl.Float64, "lo": pl.Float64, "vol": pl.Float64}


def _ohlc(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.select("ticker", pl.col("end_ts").cast(pl.Datetime("us", "UTC")), pl.col("bid_close").alias("bid"),
                     pl.col("ask_close").alias("ask"), pl.col("p_high").alias("hi"), pl.col("p_low").alias("lo"),
                     pl.col("volume").alias("vol"))


def hourly_sources() -> list[tuple[str, pl.LazyFrame]]:
    """Every cached hourly candle layer with the trade range first (priority order)."""
    out: list[tuple[str, pl.LazyFrame]] = []
    mine = CACHE / "hourly"
    if mine.exists() and list(mine.glob("part-*.parquet")):
        out.append(("mentions_ext", _ohlc(pl.scan_parquet(str(mine / "part-*.parquet")))))
    if (XCAT / "hourly").exists():
        out.append(("xcat", _ohlc(pl.scan_parquet(str(XCAT / "hourly" / "part-*.parquet")))))
    for name, p in (("pre_event", PRE / "hourly.parquet"), ("pre_event_fwd", PRE / "forward" / "hourly.parquet")):
        if p.exists():
            out.append((name, _ohlc(pl.scan_parquet(p))))
    if (MH / "candles.parquet").exists():
        out.append(("mentions_history", pl.scan_parquet(MH / "candles.parquet").select(
            "ticker", pl.col("end_ts").cast(pl.Datetime("us", "UTC")), pl.col("yes_bid_close").alias("bid"),
            pl.col("yes_ask_close").alias("ask"), pl.col("price_high").alias("hi"), pl.col("price_low").alias("lo"),
            pl.col("volume").alias("vol"))))
    return out


def load_hourly(tickers: list[str]) -> pl.DataFrame:
    """Hourly candles of ``tickers`` from every source; a (ticker, hour) keeps the first source
    that has the trade range (all sources are the same Kalshi candles)."""
    tk = _strs(tickers)
    frames = []
    for i, (name, lf) in enumerate(hourly_sources()):
        frames.append(lf.filter(pl.col("ticker").is_in(tk)).with_columns(pl.lit(i).alias("_pri"), pl.lit(name).alias("src"))
                      .collect())
    if not frames:
        return pl.DataFrame(schema={**HSCHEMA, "src": pl.String})
    c = pl.concat(frames, how="vertical_relaxed")
    c = c.with_columns(pl.col("hi").is_null().alias("_norange")).sort("ticker", "end_ts", "_norange", "_pri")
    return c.unique(["ticker", "end_ts"], keep="first").drop("_pri", "_norange").sort("ticker", "end_ts")


# ---- fetching (public Kalshi, <= 1 request / s, cached) ------------------------------------------
def kalshi_api(min_interval: float = 1.0):
    import random

    from albert.config import Settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    s = Settings()

    def nap(x: float) -> None:
        time.sleep(min(max(x, 2.0), 30.0) * (0.75 + 0.5 * random.random()))
    return k.Kalshi(CachedClient("kalshi", s.kalshi_base, CACHE / "raw", min_interval=min_interval,
                                 max_retries=12, sleep=nap))


def _batch_jobs(rows: pl.DataFrame, period: int, max_candles: int = 9_500, max_tickers: int = 50) -> list[list[dict]]:
    """Group (ticker, lo, hi) rows per event into batch requests within the API caps."""
    jobs: list[list[dict]] = []
    for _, g in rows.sort("event", "ticker").group_by("event", maintain_order=True):
        cur: list[dict] = []
        for r in g.iter_rows(named=True):
            lo = min([x["lo"] for x in cur] + [r["lo"]])
            hi = max([x["hi"] for x in cur] + [r["hi"]])
            span = math.ceil((hi - lo).total_seconds() / 60 / period) + 1
            if cur and (span * (len(cur) + 1) > max_candles or len(cur) >= max_tickers):
                jobs.append(cur)
                cur = []
            cur.append(r)
        if cur:
            jobs.append(cur)
    return jobs


def fetch_candles(rows: pl.DataFrame, period: int, out_dir: Path, *, label: str) -> None:
    """Candles for ``rows`` (event, ticker, lo, hi, settlement_ts): live batch requests for markets
    settled after the historical cutoff, one historical request per market before it. Rows saved
    as ``part-*.parquet`` in ``out_dir`` (xcat OHLC schema); finished tickers in ``done.parquet``."""
    from albert.research import xcat

    out_dir.mkdir(parents=True, exist_ok=True)
    done_p = out_dir / "done.parquet"
    done = set(pl.read_parquet(done_p)["ticker"]) if done_p.exists() else set()
    rows = rows.filter(~pl.col("ticker").is_in(_strs(done) if done else _strs([""])))
    api = kalshi_api(1.0)
    cutoff = api.cutoff()["market_settled_ts"]
    live = rows.filter(pl.col("settlement_ts") >= cutoff)
    hist = rows.filter(pl.col("settlement_ts") < cutoff)
    jobs = [("batch", j) for j in _batch_jobs(live, period)] + [("hist", [r]) for r in hist.iter_rows(named=True)]
    print(f"{label}: {rows.height} markets in {len(jobs)} requests (cutoff {cutoff:%Y-%m-%d})", flush=True)
    buf: list[dict] = []
    fin: list[str] = []
    part = len(list(out_dir.glob("part-*.parquet")))

    def flush() -> None:
        nonlocal buf, part
        if buf:
            _save(pl.DataFrame(buf, schema=xcat.OHLC_SCHEMA), out_dir / f"part-{part:05d}.parquet")
            part += 1
        buf = []
        _save(pl.DataFrame({"ticker": sorted(done | set(fin))}, schema={"ticker": pl.String}), done_p)

    for i, (kind, job) in enumerate(jobs, 1):
        try:
            if kind == "batch" and len(job) > 1:
                lo, hi = min(r["lo"] for r in job), max(r["hi"] for r in job)
                got = xcat.batch_rows(api.candles_batch([r["ticker"] for r in job], lo, hi, period).payload, period)
                for r in job:
                    if r["ticker"] in got:
                        buf.extend(got[r["ticker"]])
                        fin.append(r["ticker"])
                missing = [r for r in job if r["ticker"] not in got]
            else:
                missing = job
            for r in missing:
                for res_tk, res in xcat._single(api, r, kind == "hist"):
                    if res is not None:
                        buf.extend(res)
                        fin.append(res_tk)
        except RuntimeError as error:
            print(f"skip {job[0]['event']}: {str(error)[-200:]}", flush=True)
        if i % 50 == 0:
            flush()
            print(f"  {label} {i}/{len(jobs)} requests", flush=True)
    flush()
    print(f"{label} done: {len(fin)} markets", flush=True)


def hourly_gaps(U: pl.DataFrame) -> pl.DataFrame:
    """Legs of events from ``GRID_FROM`` whose hourly candles do not reach back to start - 49 h
    (or the leg's open): fetch [max(open, start - 51 h), start]."""
    U = U.filter(pl.col("start") >= GRID_FROM)
    H = load_hourly(U["ticker"].to_list())
    first = H.group_by("ticker").agg(pl.col("end_ts").min().alias("first"))
    d = U.join(first, on="ticker", how="left").with_columns(
        pl.max_horizontal(pl.col("open_time"), pl.col("start") - HOURLY_BACK).alias("lo"), pl.col("start").alias("hi"))
    need = d.filter(pl.col("first").is_null() | (pl.col("first") > pl.col("lo") + timedelta(hours=3)))
    need = need.filter(pl.col("hi") > pl.col("lo"))
    return need.select("event", "ticker", "lo", "hi", "settlement_ts", pl.lit(60).alias("period"),
                       pl.col("event").alias("event_ticker"))


MINUTE_AFTER = timedelta(hours=8)


def minute_jobs(U: pl.DataFrame) -> pl.DataFrame:
    """Legs needing minute candles after the start (post-speech sweep): [start - 10 min,
    min(close, start + 8 h)], events from 2026-09-18 (live batch endpoint) not in the xcat minute
    layer."""
    U = U.filter(pl.col("start") >= datetime(2026, 9, 18, tzinfo=UTC))
    have = set()
    dp = XCAT / "minute" / "done.parquet"
    if dp.exists():
        have = set(pl.read_parquet(dp)["ticker"])
    d = U.filter(~pl.col("ticker").is_in(_strs(have) if have else _strs([""])))
    d = d.with_columns((pl.col("start") - END_BEFORE).alias("lo"),
                       pl.min_horizontal(pl.col("close_time"), pl.col("start") + MINUTE_AFTER).alias("hi"))
    d = d.filter(pl.col("hi") > pl.col("lo"))
    return d.select("event", "ticker", "lo", "hi", "settlement_ts", pl.lit(1).alias("period"),
                    pl.col("event").alias("event_ticker"))


def late_minute_jobs(U: pl.DataFrame) -> pl.DataFrame:
    """Legs of late-listed events (posted at :func:`late_times`): minute candles over [t, start]
    to check the hourly fill proxy, whose last hour reaches past ``end = start - 10 min``."""
    G = U.filter(pl.col("start") >= GRID_FROM)
    d = G.join(late_times(G), on="event").filter(pl.col("open_time") <= pl.col("t"))
    d = d.with_columns(pl.col("t").alias("lo"), pl.col("start").alias("hi"))
    return d.select("event", "ticker", "lo", "hi", "settlement_ts", pl.lit(1).alias("period"),
                    pl.col("event").alias("event_ticker"))


def load_minute(tickers: list[str], dirs: tuple[Path, ...] = (CACHE / "minute", CACHE / "minute_late", XCAT / "minute")
                ) -> pl.DataFrame:
    """Minute candles (xcat first-3-days layer plus this study's post-start and late-event fetches)."""
    tk = _strs(tickers)
    frames = []
    for d in dirs:
        if d.exists() and list(d.glob("part-*.parquet")):
            frames.append(_ohlc(pl.scan_parquet(str(d / "part-*.parquet"))).filter(pl.col("ticker").is_in(tk)).collect())
    if not frames:
        return pl.DataFrame(schema=HSCHEMA)
    return pl.concat(frames).unique(["ticker", "end_ts"], keep="first").sort("ticker", "end_ts")


# ---- Wikimedia pageviews ----------------------------------------------------------------------------
WIKI_BASE = "https://wikimedia.org/api/rest_v1"
PV_FROM, PV_TO = "20250101", "20261116"
UA = "albert-research/0.2 (quantitative research; public data only)"


def pageview_rows(payload: dict[str, Any], title: str) -> list[dict[str, Any]]:
    """Daily user pageviews of one article from a REST ``per-article`` response."""
    out = []
    for it in payload.get("items") or []:
        ts = str(it.get("timestamp") or "")
        if len(ts) >= 8:
            out.append({"title": title, "day": date(int(ts[:4]), int(ts[4:6]), int(ts[6:8])),
                        "views": float(it.get("views") or 0)})
    return out


def fetch_pageviews(titles: list[str], *, min_interval: float = 3.0) -> pl.DataFrame:
    """Daily en.wikipedia user pageviews per title, 2025-01-01..2026-11-16 (one cached request per
    title; titles that do not exist are recorded in ``pageviews_missing.parquet``)."""
    from albert.http import CachedClient, HttpError

    c = CachedClient("wikimedia", WIKI_BASE, CACHE / "raw", min_interval=min_interval, max_retries=8, user_agent=UA)
    miss_p, rows_p = CACHE / "pageviews_missing.parquet", CACHE / "pageviews.parquet"
    missing = set(pl.read_parquet(miss_p)["title"]) if miss_p.exists() else set()
    have = pl.read_parquet(rows_p) if rows_p.exists() else pl.DataFrame(
        schema={"title": pl.String, "day": pl.Date, "views": pl.Float64})
    done = set(have["title"].unique()) | missing
    todo = [t for t in dict.fromkeys(titles) if t not in done]
    print(f"pageviews: {len(todo)} titles to fetch ({len(done)} done)", flush=True)
    new: list[dict[str, Any]] = []
    for i, t in enumerate(todo, 1):
        path = f"/metrics/pageviews/per-article/en.wikipedia/all-access/user/{quote(t, safe='')}/daily/{PV_FROM}/{PV_TO}"
        try:
            rows = pageview_rows(c.get(path).payload, t)
            new.extend(rows)
            if not rows:
                missing.add(t)
        except HttpError as error:
            if "404" in str(error) or "400" in str(error):
                missing.add(t)
            else:
                print(f"skip {t}: {str(error)[-160:]}", flush=True)
        if i % 50 == 0 or i == len(todo):
            if new:
                have = pl.concat([have, pl.DataFrame(new, schema=have.schema)])
                new = []
            _save(have, rows_p)
            _save(pl.DataFrame({"title": sorted(missing)}, schema={"title": pl.String}), miss_p)
            print(f"  pageviews {i}/{len(todo)}", flush=True)
    return have


# ---- decision rows (the live rule's inputs at t = start - L) ---------------------------------------
def quote_at(d: pl.DataFrame, H: pl.DataFrame, tcol: str = "t") -> pl.DataFrame:
    """The last hourly candle closed at or before ``tcol`` (<= 24 h old; Kalshi omits unchanged
    hours) with a usable quote: ``bid`` / ``ask`` (an empty side is 0 / 1)."""
    q = (H.filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("ask") > pl.col("bid") + EPS)
                  & ((pl.col("bid") > 0) | (pl.col("ask") < 1)))
         .select("ticker", pl.col("end_ts").alias("q_ts"), "bid", "ask").sort("q_ts"))
    out = d.sort(tcol).join_asof(q, left_on=tcol, right_on="q_ts", by="ticker", strategy="backward",
                                 check_sortedness=False)
    return out.filter(pl.col("q_ts").is_not_null() & ((pl.col(tcol) - pl.col("q_ts")) <= QUOTE_AGE))


LATE_MIN_LEAD = timedelta(hours=1)


def late_times(U: pl.DataFrame) -> pl.DataFrame:
    """Events listed after start - 6 h (the live bot never posts them): post at the end of the
    first full hour after the event's first leg opened, if that is >= 1 h before the start."""
    ev = U.group_by("event").agg(pl.col("open_time").min().alias("ev_open"), pl.col("start").first())
    ev = ev.filter(pl.col("ev_open") > pl.col("start") - timedelta(hours=LIVE_L))
    ev = ev.with_columns((pl.col("ev_open").dt.truncate("1h") + timedelta(hours=1)).alias("t"))
    return ev.filter(pl.col("t") <= pl.col("start") - LATE_MIN_LEAD).select("event", "t")


def decision_rows(U: pl.DataFrame, H: pl.DataFrame, lead_h: int, *, late: bool = False) -> pl.DataFrame:
    """Legs listed and open at ``t = start - L`` with a quote; the outcome-blind early-start
    exclusion (share of legs bid >= 97c at t vs 18 h earlier jumps > 12 points); ``end =
    min(start - 10 min, close)``; ``fair`` = frozen calibrated mid. ``late``: the events listed
    after start - 6 h, at :func:`late_times` (``lead_h`` is then only a label)."""
    if late:
        d = U.join(late_times(U), on="event")
    else:
        d = U.with_columns((pl.col("start") - timedelta(hours=lead_h)).alias("t"))
    d = d.filter((pl.col("open_time") <= pl.col("t")) & (pl.col("close_time") > pl.col("t")))
    Q = quote_at(d, H, "t")
    if not Q.height:
        return Q
    d0 = d.select("event", "ticker", (pl.col("t") - SAID_LOOKBACK).alias("t0"))
    Q0 = quote_at(d0, H, "t0")
    s = Q.group_by("event").agg((pl.col("bid") >= SAID_BID).mean().alias("said"))
    s0 = Q0.group_by("event").agg((pl.col("bid") >= SAID_BID).mean().alias("said0"))
    sh = s.join(s0, on="event", how="left").with_columns(pl.col("said0").fill_null(0.0))
    bad = sh.filter(pl.col("said") > pl.col("said0") + SAID_JUMP + 1e-12)["event"]
    Q = Q.filter(~pl.col("event").is_in(bad.implode()))
    Q = Q.with_columns(pl.min_horizontal(pl.col("start") - END_BEFORE, pl.col("close_time")).alias("end"),
                       ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("ask") - pl.col("bid")).alias("spread"),
                       pl.lit(lead_h).alias("lead_h"))
    Q = Q.filter(pl.col("end") > pl.col("t"))
    return Q.with_columns(pl.Series("fair", fair_cal(Q["mid"].to_numpy()))).sort("event", "ticker")


def breaker_kill(Q: pl.DataFrame, H: pl.DataFrame, period: timedelta = timedelta(hours=1)) -> pl.DataFrame:
    """Circuit breaker from candles: the START of the first candle after ``t`` whose trade high is
    >= 95c on a leg asked <= 80c at ``t`` (the whole candle is cancelled: conservative)."""
    legs = Q.filter(pl.col("ask") <= CB_ASK + EPS).select("event", "ticker", "t")
    h = H.filter(pl.col("hi") >= CB_PRINT - EPS).select("ticker", "end_ts").join(legs, on="ticker")
    h = h.filter(pl.col("end_ts") > pl.col("t"))
    return h.group_by("event").agg((pl.col("end_ts").min() - period).alias("kill"))


def with_breaker(Q: pl.DataFrame, H: pl.DataFrame, period: timedelta = timedelta(hours=1)) -> pl.DataFrame:
    k = breaker_kill(Q, H, period)
    return Q.join(k, on="event", how="left").with_columns(
        pl.min_horizontal("end", pl.col("kill").fill_null(pl.col("end"))).alias("end_cb"))


def hourly_fills(O: pl.DataFrame, H: pl.DataFrame, end_col: str = "end_cb",
                 period: timedelta = timedelta(hours=1)) -> pl.DataFrame:
    """NO-bid fills from candle trade ranges: the bid at ``price`` fills if a candle overlapping
    ``(t, end]`` printed strictly above ``1 - price`` (a print through our offer). ``fill_ts`` = the
    middle of the first such candle, clipped to the window. ``O`` needs row, ticker, t, end, price.
    ``period`` = the candle length (hourly by default; minute candles for the checks)."""
    o = O.select("row", "ticker", "t", pl.col(end_col).alias("_end"), (1 - pl.col("price")).alias("level"))
    h = H.select("ticker", "end_ts", "hi").filter(pl.col("hi").is_not_null()).join(o, on="ticker")
    h = h.filter((pl.col("end_ts") > pl.col("t")) & (pl.col("end_ts") - period < pl.col("_end"))
                 & (pl.col("hi") > pl.col("level") + EPS))
    f = h.group_by("row").agg(pl.col("end_ts").min().alias("hit_end"), pl.col("t").first(), pl.col("_end").first())
    f = f.with_columns(pl.min_horizontal(pl.max_horizontal(pl.col("hit_end") - period / 2, pl.col("t")
                                                           + timedelta(minutes=1)), pl.col("_end")).alias("fill_ts"))
    return O.join(f.select("row", "fill_ts"), on="row", how="left").with_columns(
        pl.col("fill_ts").is_not_null().alias("filled"))


def minute_check(o: pl.DataFrame, Q: pl.DataFrame, Mn: pl.DataFrame, covered: set[str], phase: str = "late") -> pl.DataFrame:
    """The hourly proxy vs minute candles on one phase of a cell's orders, on events whose every
    leg's minute candles were fetched over the window (``covered``; a minute without a change has
    no candle): breaker and fills recomputed at one-minute resolution."""
    one = timedelta(minutes=1)
    ev_ok = Q.group_by("event").agg(pl.col("ticker").is_in(_strs(covered) if covered else _strs([""])).all().alias("ok")
                                    ).filter("ok")["event"]
    qm = with_breaker(Q.drop("kill", "end_cb").filter(pl.col("event").is_in(ev_ok.implode())), Mn, one)
    oo = o.filter((pl.col("phase") == phase) & pl.col("event").is_in(ev_ok.implode()))
    oo = oo.join(qm.select("ticker", pl.col("end_cb").alias("end_m")), on="ticker").with_row_index("row")
    fm = hourly_fills(oo.drop("fill_ts", "filled").rename({"end_m": "end_x"}), Mn, "end_x", one)
    P = oo.select("row", "event", "split", "y", "price", "pnl", pl.col("filled").alias("filled_hourly")).join(
        fm.select("row", pl.col("filled").alias("filled_minute")), on="row")
    P = P.with_columns(pl.when(pl.col("filled_minute")).then((1 - pl.col("y")) - pl.col("price")).otherwise(0.0).alias("pnl_minute"))
    rows = []
    for (sp,), g in P.group_by(["split"]):
        a, b = g.filter("filled_hourly"), g.filter("filled_minute")
        mu, te, _ = clustered_t(b["pnl_minute"].to_numpy(), b["event"].to_numpy()) if b.height else (float("nan"), None, 0)
        rows.append({"split": sp, "phase": phase, "events": g["event"].n_unique(), "orders": g.height,
                     "fills_hourly": a.height, "fills_minute": b.height,
                     "agree": round(float((g["filled_hourly"] == g["filled_minute"]).mean()), 4),
                     "c_per_fill_hourly": round(float(a["pnl"].mean()) * 100, 2) if a.height else None,
                     "c_per_fill_minute": round(mu * 100, 2) if b.height else None,
                     "t_event_minute": None if te is None else round(te, 2),
                     "pnl_usd_minute": round(float(b["pnl_minute"].sum()), 2)})
    return pl.DataFrame(rows, infer_schema_length=None).sort("split")


ORDER_COLS = ["event", "ticker", "series", "sgroup", "split", "et_date", "start", "settlement_ts", "y", "wiki",
              "yes_sub_title", "t", "bid", "ask", "mid", "fair", "spread"]


def cell_orders(Qs: dict[int, pl.DataFrame], H: pl.DataFrame, lead_h: int, margin: float, variant: str) -> pl.DataFrame:
    """Orders of one grid cell with fills and PnL. ``hold``: one NO bid from ``start - L`` to
    ``end_cb``. ``extend`` (L > 6): the early bid rests to ``start - 6 h`` (or the breaker); a leg it
    did not fill then gets the live rule's 6 h order, unless the event was cancelled early."""
    def orders(Q: pl.DataFrame, end_col: str, phase: str) -> pl.DataFrame:
        px = no_bid(Q["fair"].to_numpy(), Q["bid"].to_numpy(), margin)
        o = Q.select(*ORDER_COLS, pl.col(end_col).alias("end_o"), "kill").with_columns(
            pl.Series("price", px), pl.lit(phase).alias("phase"))
        o = o.filter(pl.col("price").is_not_null()).with_row_index("row")
        return hourly_fills(o, H, "end_o")

    QL = Qs[lead_h]
    if variant == "late":
        out = pl.concat([orders(QL, "end_cb", "single").drop("row"), orders(Qs["late"], "end_cb", "late").drop("row")],
                        how="vertical_relaxed")
    elif variant == "hold" or lead_h == LIVE_L:
        out = orders(QL, "end_cb", "single")
    else:
        six = QL.with_columns(pl.min_horizontal(pl.col("end_cb"), pl.col("start") - timedelta(hours=LIVE_L)).alias("end_e"))
        early = orders(six, "end_e", "early")
        killed = QL.filter(pl.col("kill").is_not_null() & (pl.col("kill") <= pl.col("start") - timedelta(hours=LIVE_L)))
        Q6 = Qs[LIVE_L].filter(~pl.col("event").is_in(killed["event"].unique().implode())
                               & ~pl.col("ticker").is_in(early.filter("filled")["ticker"].implode()))
        late = orders(Q6, "end_cb", "late")
        out = pl.concat([early.drop("row"), late.drop("row")], how="vertical_relaxed")
    out = out.with_columns(pl.when(pl.col("filled")).then((1 - pl.col("y")) - pl.col("price")).otherwise(0.0).alias("pnl"))
    return out.drop("row", strict=False).with_columns(
        pl.lit(lead_h).alias("lead_h"), pl.lit(margin).alias("margin"), pl.lit(_variant(lead_h, variant)).alias("variant"))


def _variant(lead_h: int, variant: str) -> str:
    return "hold" if lead_h == LIVE_L and variant == "extend" else variant


def cell_name(lead_h: int, margin: float, variant: str) -> str:
    return f"L{lead_h}h/m{round(margin * 100)}c/{_variant(lead_h, variant)}"


GRID_CELLS = ([(L, m, "hold") for L in LEADS for m in MARGINS] + [(L, m, "extend") for L in LEADS[1:] for m in MARGINS]
              + [(LIVE_L, m, "late") for m in MARGINS])


def build_qs(U: pl.DataFrame, H: pl.DataFrame) -> dict[Any, pl.DataFrame]:
    Qs: dict[Any, pl.DataFrame] = {L: with_breaker(decision_rows(U, H, L), H) for L in LEADS}
    Qs["late"] = with_breaker(decision_rows(U, H, LIVE_L, late=True), H)
    return Qs


def summarize(o: pl.DataFrame) -> dict[str, Any]:
    """Orders, fills, PnL per fill (c) with t clustered by event and by ET date of the start."""
    f = o.filter("filled")
    row: dict[str, Any] = {"events": o["event"].n_unique(), "orders": o.height, "fills": f.height,
                           "fill_rate": round(f.height / o.height, 3) if o.height else None}
    if f.height:
        x = f["pnl"].to_numpy()
        mu, te, ke = clustered_t(x, f["event"].to_numpy())
        _, td, kd = clustered_t(x, f["et_date"].cast(pl.String).to_numpy())
        row.update({"events_filled": ke, "dates_filled": kd, "c_per_fill": round(mu * 100, 2),
                    "t_event": None if te is None else round(te, 2), "t_date": None if td is None else round(td, 2),
                    "pnl_usd": round(float(x.sum()), 2), "usd_per_event": round(float(x.sum()) / o["event"].n_unique(), 3),
                    "win_rate": round(float((f["y"] == 0).mean()), 3), "avg_price": round(float(f["price"].mean()), 3)})
    return row


# ---- capped book (the live bot's budget) --------------------------------------------------------------
def capped_book(o: pl.DataFrame, *, var: float = VAR, max_events: int = MAX_EVENTS, daily_cap: float = DAILY_CAP,
                max_orders: int = 40) -> pl.DataFrame:
    """Replay of ``live_mentions`` budgeting on a cell's orders: postings in time order (an event's
    first posting needs a free event slot; ``extend`` re-posts at start - 6 h); at each posting the
    room is ``var - resting bids - max(today's fill cost, unsettled fill cost)``, legs are taken
    cheapest NO bid first until the room is used; no posting once today's fill cost reaches
    ``daily_cap``. Returns the accepted orders (``filled`` / ``pnl`` as in ``o``)."""
    o = o.with_columns(pl.col("t").alias("tau")).sort("tau", "event", "price")
    acc: list[dict[str, Any]] = []
    first_seen: dict[str, bool] = {}
    skipped: set[str] = set()
    ev_end: dict[str, datetime] = {}
    for (ev, tau), g in o.group_by(["event", "tau"], maintain_order=True):
        if ev in skipped:
            continue
        open_ev = {e for e, end in ev_end.items() if end > tau and e != ev}
        if ev not in first_seen and len(open_ev) >= max_events:
            skipped.add(ev)
            continue
        day = tau.date()
        today = sum(a["price"] for a in acc if a["fill_ts"] is not None and a["fill_ts"] <= tau and a["fill_ts"].date() == day)
        if today >= daily_cap - EPS:
            if ev not in first_seen:
                skipped.add(ev)
            continue
        resting = sum(a["price"] for a in acc if a["tau"] <= tau < a["end_o"] and (a["fill_ts"] is None or a["fill_ts"] > tau))
        unsettled = sum(a["price"] for a in acc if a["fill_ts"] is not None and a["fill_ts"] <= tau < a["settlement_ts"])
        room = var - resting - max(today, unsettled)
        first_seen[ev] = True
        total = 0.0
        for r in g.head(max_orders).iter_rows(named=True):
            if total + r["price"] > room + EPS:
                break
            total += r["price"]
            acc.append(r)
            ev_end[ev] = max(ev_end.get(ev, r["end_o"]), r["end_o"])
    if not acc:
        return o.head(0)
    return pl.DataFrame(acc, schema=o.schema)


def ledger(book: pl.DataFrame, strategy: str, variant: str, split: str) -> pl.DataFrame:
    """``vol_metrics`` ledger rows of a book's fills (1 NO contract each, no maker fee)."""
    f = book.filter("filled")
    return f.select(pl.lit(strategy).alias("strategy"), pl.lit(variant).alias("variant"), pl.lit(split).alias("split"),
                    pl.col("ticker").alias("struct_id"), pl.col("event").alias("event_ticker"), "ticker",
                    pl.lit("no").alias("side"), pl.lit(1).alias("n"), pl.col("fill_ts").alias("t_in"),
                    pl.col("price").alias("cost_usd"), "settlement_ts",
                    (1 - pl.col("y")).cast(pl.Float64).alias("payoff_usd")).rename({"settlement_ts": "settle_ts"})


def marks_from(H: pl.DataFrame) -> pl.DataFrame:
    two = pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("ask") > pl.col("bid"))
    return (H.filter(two).select("ticker", pl.col("end_ts").alias("ts"), ((pl.col("bid") + pl.col("ask")) / 2).alias("yes_mid"))
            .sort("ticker", "ts"))


def book_metrics(led: pl.DataFrame, marks: pl.DataFrame) -> dict[str, Any]:
    from albert.research import vol_metrics as V

    if not led.height:
        return {}
    m = V.ledger_metrics(led, marks.filter(pl.col("ticker").is_in(led["ticker"].implode())))
    keep = ["positions", "events", "pnl_usd", "usd_per_week", "ann_vol_pct", "sharpe", "sharpe_ci90", "p_sharpe_pos",
            "sharpe_ex_top5", "usd_per_week_ex_top5", "max_dd_usd", "worst_week_usd", "weeks_positive_pct", "days"]
    out = {k: m.get(k) for k in keep}
    ci = out.pop("sharpe_ci90")
    out["sharpe_ci90"] = f"[{ci[0]}, {ci[1]}]" if ci else ""
    return out


def _slug(name: str) -> str:
    return name.replace("/", "_")


def write_ledger(book: pl.DataFrame, name: str, split: str, strategy: str = "mentions_ext") -> pl.DataFrame:
    led = ledger(book, strategy, name, split)
    if led.height:
        _save(led, OUT / "ledgers" / f"ledger_{_slug(name)}_{split}.parquet")
    return led


def cell_row(o: pl.DataFrame, marks: pl.DataFrame, name: str, split: str, *, save: bool = True) -> dict[str, Any]:
    """One table row: uncapped fills (every order) and the $15 capped book with its ledger metrics."""
    s = o.filter(pl.col("split") == split)
    book = capped_book(s)
    led = write_ledger(book, name, split) if save else ledger(book, "mentions_ext", name, split)
    bm = book_metrics(led, marks)
    bs = summarize(book) if book.height else {}
    return {"cell": name, "split": split, **summarize(s),
            "book_fills": bs.get("fills"), "book_c_per_fill": bs.get("c_per_fill"),
            **{f"book_{k}": v for k, v in bm.items()}}


def choose_cell(T: pl.DataFrame) -> str:
    """Pre-specified (before any holdout number was computed): the dev cell with the largest $15
    capped-book PnL among cells whose uncapped dev PnL per fill has t (by ET date) >= 2; ties by
    the book's Sharpe."""
    ok = T.filter((pl.col("split") == "dev") & (pl.col("t_date").fill_null(0) >= 2))
    return ok.sort(["book_pnl_usd", "book_sharpe"], descending=True, nulls_last=True)["cell"][0]


def run_grid(U: pl.DataFrame) -> dict[str, Any]:
    """Q2: the entry-window x margin grid on events from 2026-08-17 (whole-life hourly candles):
    dev table for every cell; holdout only for the dev-chosen cell and the live rule."""
    G = U.filter(pl.col("start") >= GRID_FROM)
    H = load_hourly(G["ticker"].to_list())
    marks = marks_from(H)
    Qs = build_qs(G, H)
    cov = pl.concat([q.group_by("split").agg(pl.col("event").n_unique().alias("events"), pl.len().alias("legs"))
                     .with_columns(pl.lit(str(k)).alias("entry")) for k, q in Qs.items()]).sort("entry", "split")
    rows, orders = [], {}
    for L, m, v in GRID_CELLS:
        name = cell_name(L, m, v)
        o = cell_orders(Qs, H, L, m, v)
        orders[name] = o
        rows.append(cell_row(o, marks, name, "dev", save=False))
        print(f"  {name}: dev {rows[-1].get('c_per_fill')}c x {rows[-1].get('fills')} fills, "
              f"book ${rows[-1].get('book_pnl_usd')}", flush=True)
    T = pl.DataFrame(rows, infer_schema_length=None)
    live = cell_name(LIVE_L, LIVE_M, "hold")
    chosen = choose_cell(T)
    for c in dict.fromkeys([live, chosen]):          # ledgers only for the two cells carried to holdout
        cell_row(orders[c], marks, c, "dev")
    hold = [cell_row(orders[c], marks, c, "holdout") for c in dict.fromkeys([live, chosen])]
    TH = pl.DataFrame(hold, infer_schema_length=None)
    return {"H": H, "marks": marks, "Qs": Qs, "orders": orders, "dev": T, "holdout": TH, "chosen": chosen,
            "live": live, "coverage": cov}


# ---- fill-source check: hourly proxy vs public prints (6 h cells) ---------------------------------------
def prints_check(o: pl.DataFrame, Q: pl.DataFrame) -> pl.DataFrame:
    """On legs whose (t, end] the ``mentions_history`` print fetch covers: fills from prints
    strictly through the bid (breaker from prints) vs the hourly proxy, with the contracts printed
    through our price while resting (capacity)."""
    from albert.research import mentions_history as MH_

    trades, windows = _mh_trades(Q["ticker"].to_list())
    if not trades.height or not windows.height:
        return pl.DataFrame()
    q = Q.select("event", "ticker", "t", "end", "ask")
    cov = q.join(windows, on="ticker").filter((pl.col("t_from") <= pl.col("t") + timedelta(minutes=1))
                                              & (pl.col("t_to") >= pl.col("end") - timedelta(minutes=1)))
    ok = cov["ticker"].unique()
    # events whose every candidate leg is covered: the breaker needs all cheap legs' prints
    evs = q.group_by("event").agg(pl.col("ticker").is_in(ok.implode()).all().alias("all"))
    evs = evs.filter("all")["event"]
    qq = Q.filter(pl.col("event").is_in(evs.implode()))
    tr = trades.filter(pl.col("ticker").is_in(qq["ticker"].implode()))
    kt = MH_.kill_times(qq, tr)
    qq = qq.join(kt, on="event", how="left").with_columns(
        pl.min_horizontal("end", pl.col("kill_ts").fill_null(pl.col("end"))).alias("end_pr"))
    oo = o.filter(pl.col("ticker").is_in(qq["ticker"].implode()) & (pl.col("phase") == "single"))
    oo = oo.join(qq.select("ticker", "end_pr"), on="ticker")
    P = oo.with_columns(pl.lit("no").alias("side"), pl.col("filled").alias("filled_hourly"))
    by = {k[0]: g for k, g in tr.sort("ticker", "created_time").group_by(["ticker"])}
    fp, thru = [], []
    for r in P.iter_rows(named=True):
        g = by.get(r["ticker"])
        if g is None:
            fp.append(False)
            thru.append(0.0)
            continue
        w = g.filter((pl.col("created_time") > r["t"]) & (pl.col("created_time") <= r["end_pr"])
                     & (pl.col("yes_price") > 1 - r["price"] + EPS))
        fp.append(w.height > 0)
        thru.append(float(w.filter(pl.col("taker_side") == "yes")["count"].sum()))
    P = P.with_columns(pl.Series("filled_prints", fp), pl.Series("thru", thru))
    P = P.with_columns(pl.when(pl.col("filled_prints")).then((1 - pl.col("y")) - pl.col("price")).otherwise(0.0).alias("pnl_prints"))
    rows = []
    for sp, g in P.group_by("split"):
        a, b = g.filter("filled_hourly"), g.filter("filled_prints")
        rows.append({"split": sp[0], "events": g["event"].n_unique(), "orders": g.height, "fills_hourly": a.height,
                     "fills_prints": b.height, "agree": round(float((g["filled_hourly"] == g["filled_prints"]).mean()), 4),
                     "c_per_fill_hourly": round(float(a["pnl"].mean()) * 100, 2) if a.height else None,
                     "c_per_fill_prints": round(float(b["pnl_prints"].mean()) * 100, 2) if b.height else None,
                     "thru_median": float(b["thru"].median()) if b.height else None,
                     "thru_p10": float(b["thru"].quantile(0.10)) if b.height else None,
                     "share_thru_ge_5": round(float((b["thru"] >= 5).mean()), 3) if b.height else None})
    return pl.DataFrame(rows).sort("split")


def _mh_trades(tickers: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``mentions_history`` prints of ``tickers`` and the windows they cover (its own fetch plus
    round 2's); scanned lazily and filtered by ticker to keep memory low."""
    tk = _strs(tickers)
    frames, wins = [], []
    cols = ["trade_id", "ticker", "created_time", "count", "yes_price", "taker_side"]
    if (MH / "trades.parquet").exists():
        frames.append(pl.scan_parquet(MH / "trades.parquet").select(cols).filter(pl.col("ticker").is_in(tk)))
        wins.append(pl.read_parquet(MH / "trade_windows.parquet").filter(pl.col("ticker").is_in(tk)))
    c = Path("data/cache")
    bad = set()
    if (c / "fundamental_mentions_maker_failed.parquet").exists():
        bad = set(pl.read_parquet(c / "fundamental_mentions_maker_failed.parquet")["ticker"])
    for name in ("mentions_maker", "mentions_maker_retry"):
        tp, wp = c / f"fundamental_{name}_trades.parquet", c / f"fundamental_{name}_trades_windows.parquet"
        if tp.exists():
            frames.append(pl.scan_parquet(tp).select(cols).filter(pl.col("ticker").is_in(tk)))
        if wp.exists():
            wins.append(pl.read_parquet(wp).filter(pl.col("ticker").is_in(tk)
                                                   & ~pl.col("ticker").is_in(_strs(bad) if bad else _strs([""]))))
    tr = (pl.concat(frames, how="vertical_relaxed").unique("trade_id").collect() if frames else pl.DataFrame())
    w = pl.concat(wins, how="vertical_relaxed").unique() if wins else pl.DataFrame()
    return tr, w


def long_history(U: pl.DataFrame) -> pl.DataFrame:
    """Supplementary dev check on the older history (events before 2026-08-17 with
    ``mentions_history`` candles over [start - 27 h, start]: Trump and Vance complete, a random 6%
    of other speakers): hold cells L in 6/12/24 h and extend cells L in 12/24 h, uncapped."""
    A = U.filter(pl.col("start") < GRID_FROM)
    H = load_hourly(A["ticker"].to_list())
    A = A.filter(pl.col("ticker").is_in(H["ticker"].unique().implode()))
    Qs = {L: with_breaker(decision_rows(A, H, L), H) for L in (6, 12, 24)}
    rows = []
    for L, v in ((6, "hold"), (12, "hold"), (24, "hold"), (12, "extend"), (24, "extend")):
        for m in MARGINS:
            o = cell_orders(Qs, H, L, m, v)
            rows.append({"cell": cell_name(L, m, v), **summarize(o)})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- Q1: salience filter (Wikimedia pageviews) ------------------------------------------------------
PV_LAG = timedelta(hours=30)          # the last day used ended >= 6 h before the decision
PV_BASE = (8, 37)                     # baseline days D1-37 .. D1-8 (30 days)
SAL_FEATURES = ("z1", "z3", "z7")
SAL_K = (0.5, 1.0, 1.5, 2.0, 3.0)


def live_orders(U: pl.DataFrame, margin: float = LIVE_M, *, late: bool = False) -> pl.DataFrame:
    """The live rule's orders (6 h, ``margin``) over the whole universe with candles: the grid
    sample (whole-life candles) and the older ``mentions_history`` sample; ``late`` adds the
    events listed after start - 6 h."""
    H = load_hourly(U["ticker"].to_list())
    U = U.filter(pl.col("ticker").is_in(H["ticker"].unique().implode()))
    Qs: dict[Any, pl.DataFrame] = {LIVE_L: with_breaker(decision_rows(U, H, LIVE_L), H)}
    if late:
        Qs["late"] = with_breaker(decision_rows(U, H, LIVE_L, late=True), H)
    return cell_orders(Qs, H, LIVE_L, margin, "late" if late else "hold")


_JUNK_TITLE = re.compile(r"^[\d.,]+[kKmM]?$|times|or_above|or_more|or_fewer|^\d+\+?$")


def pageview_titles(U: pl.DataFrame) -> list[str]:
    """Article guesses of every leg the live rule (6 h, 12c) filled, count legs ("45+ times") and
    bare numbers dropped; holdout titles first, then dev titles by how many dev fills use them
    (ties in hash order), so a fetch cut short by Wikimedia's rate limit covers most dev fills."""
    import hashlib
    from collections import Counter

    o = live_orders(U, LIVE_M).filter("filled")
    def titles(w: str | None) -> list[str]:
        return [t for t in (w or "").split("|") if t and not _JUNK_TITLE.search(t)]
    ho = {t for w in o.filter(pl.col("split") == "holdout")["wiki"].to_list() for t in titles(w)}
    n = Counter(t for w in o.filter(pl.col("split") == "dev")["wiki"].to_list() for t in titles(w))
    key = lambda t: hashlib.sha256(t.encode()).hexdigest()
    return sorted(ho, key=key) + sorted(set(n) - ho, key=lambda t: (-n[t], key(t)))


def salience(o: pl.DataFrame, pv: pl.DataFrame) -> pl.DataFrame:
    """Per order: z-scores of log(1 + daily views summed over the leg's article guesses) on the
    last published day (``z1``), the last 3 (``z3``) and 7 days (``z7``) against the same leg's
    30 days before them; ``D1 = date(t - 30 h)``. Null when no article or < 20 baseline days."""
    lo, hi = PV_BASE
    keys = o.select("ticker", "t", "wiki").unique(["ticker", "t"]).with_columns(
        (pl.col("t") - PV_LAG).dt.date().alias("d1"), pl.col("wiki").str.split("|").alias("title")).explode("title")
    keys = keys.filter(pl.col("title").is_not_null() & (pl.col("title") != ""))
    days = keys.with_columns(pl.date_ranges(pl.col("d1") - timedelta(days=hi), pl.col("d1")).alias("day")).explode("day")
    v = days.join(pv, on=["title", "day"], how="left").group_by("ticker", "t", "d1", "day").agg(
        pl.col("views").sum().alias("v"), pl.col("views").is_not_null().any().alias("has"))
    v = v.filter("has").with_columns(pl.col("v").log1p().alias("lv"), (pl.col("d1") - pl.col("day")).dt.total_days().alias("lag"))
    base = v.filter((pl.col("lag") >= lo) & (pl.col("lag") <= hi)).group_by("ticker", "t").agg(
        pl.col("lv").mean().alias("mu"), pl.col("lv").std().alias("sd"), pl.len().alias("nb"))
    rec = v.filter(pl.col("lag") <= 6).group_by("ticker", "t").agg(
        pl.col("lv").filter(pl.col("lag") == 0).mean().alias("r1"), pl.col("lv").filter(pl.col("lag") <= 2).mean().alias("r3"),
        pl.col("lv").mean().alias("r7"), pl.col("v").filter(pl.col("lag") == 0).sum().alias("views_d1"))
    z = base.join(rec, on=["ticker", "t"]).filter(pl.col("nb") >= 20).with_columns(
        pl.max_horizontal(pl.col("sd"), pl.lit(0.05)).alias("sd"))
    z = z.with_columns(*[((pl.col(f"r{k}") - pl.col("mu")) / pl.col("sd")).alias(f"z{k}") for k in (1, 3, 7)])
    return o.join(z.select("ticker", "t", "mu", "views_d1", *SAL_FEATURES), on=["ticker", "t"], how="left")


def salience_grid(F: pl.DataFrame) -> pl.DataFrame:
    """Dev fills only: for every (feature, k), the fills a 'skip when z > k' filter removes, their
    PnL, and the kept fills' PnL per fill."""
    rows = []
    d = F.filter("filled")
    for feat in SAL_FEATURES:
        for k in SAL_K:
            skip = d.filter(pl.col(feat).fill_null(-99) > k)
            keep = d.filter(~(pl.col(feat).fill_null(-99) > k))
            mu, te, _ = clustered_t(skip["pnl"].to_numpy(), skip["event"].to_numpy()) if skip.height else (float("nan"), None, 0)
            rows.append({"feature": feat, "k": k, "fills_skipped": skip.height, "events_skipped": skip["event"].n_unique(),
                         "skipped_c_per_fill": round(mu * 100, 2) if skip.height else None,
                         "skipped_t_event": None if te is None else round(te, 2),
                         "skipped_pnl_usd": round(float(skip["pnl"].sum()), 2),
                         "kept_fills": keep.height, "kept_c_per_fill": round(float(keep["pnl"].mean()) * 100, 2),
                         "kept_pnl_usd": round(float(keep["pnl"].sum()), 2)})
    return pl.DataFrame(rows)


def choose_salience(G: pl.DataFrame, min_skipped: int = 30) -> tuple[str, float] | None:
    """Pre-specified: the (feature, k) whose skipped dev fills lost the most money in total
    (skipped PnL < 0) among filters removing >= ``min_skipped`` fills; None if no filter's skipped
    fills lost money (then the verdict is: no filter)."""
    ok = G.filter((pl.col("fills_skipped") >= min_skipped) & (pl.col("skipped_pnl_usd") < 0)).sort("skipped_pnl_usd")
    return None if not ok.height else (ok["feature"][0], float(ok["k"][0]))


def salience_bins(F: pl.DataFrame, feat: str = "z3") -> pl.DataFrame:
    """Fills by salience bin and split: share of words said, average price and PnL."""
    f = F.filter("filled").with_columns(
        pl.when(pl.col(feat).is_null()).then(pl.lit("none"))
        .when(pl.col(feat) <= 0).then(pl.lit("a <=0")).when(pl.col(feat) <= 1).then(pl.lit("b 0-1"))
        .when(pl.col(feat) <= 2).then(pl.lit("c 1-2")).otherwise(pl.lit("d >2")).alias("bin"))
    rows = []
    for (sp, b), g in f.group_by("split", "bin"):
        mu, te, k = clustered_t(g["pnl"].to_numpy(), g["event"].to_numpy())
        rows.append({"split": sp, f"{feat}_bin": b, "fills": g.height, "events": k, "said_share": round(float(g["y"].mean()), 3),
                     "yes_price": round(float((1 - g["price"]).mean()), 3), "c_per_fill": round(mu * 100, 2),
                     "t_event": None if te is None else round(te, 2)})
    return pl.DataFrame(rows).sort("split", f"{feat}_bin")


def salience_slope(F: pl.DataFrame, feat: str) -> pl.DataFrame:
    """OLS of fill PnL (c) on the feature, standard errors clustered by event, per split."""
    from albert.research.mentions_history import clustered_slope

    rows = []
    for (sp,), g in F.filter(pl.col("filled") & pl.col(feat).is_not_null()).group_by(["split"]):
        _a, b, _ta, tb = clustered_slope(g[feat].to_numpy(), g["pnl"].to_numpy() * 100, g["event"].to_numpy())
        rows.append({"split": sp, "feature": feat, "fills": g.height, "slope_c_per_z": round(b, 2), "t_slope": round(tb, 2)})
    return pl.DataFrame(rows).sort("split")


# ---- Q3: post-speech NO sweep --------------------------------------------------------------------------
SAID_PRINT = 0.97
DELAYS = (45, 60, 90, 120, 180)       # minutes after the scheduled start
QUIETS = (15, 30, 60)                 # minutes without a new leg first printing >= 97c
CAPS = (0.05, 0.10, 0.20, 0.50)       # YES bid range [2c, cap]
MIN_BID = 0.02


def said_times(Mn: pl.DataFrame, legs: pl.DataFrame) -> pl.DataFrame:
    """First minute after the start in which each leg traded >= 97c (a word was said)."""
    m = Mn.join(legs.select("ticker", "event", "start"), on="ticker").filter(
        (pl.col("end_ts") > pl.col("start")) & (pl.col("hi") >= SAID_PRINT - EPS))
    return m.group_by("event", "ticker").agg(pl.col("end_ts").min().alias("said_ts"))


def quiet_time(said: list[datetime], start: datetime, q: timedelta, floor: timedelta = timedelta(minutes=20)) -> datetime | None:
    """First time >= start + ``floor`` after at least one leg was said with no new leg first said
    in the previous ``q``: ``s_j + q`` for the first gap > ``q`` in the sorted said times."""
    if not said:
        return None
    s = sorted(said)
    for a, b in zip(s, s[1:] + [None]):
        if b is None or b > a + q:
            return max(a + q, start + floor)
    return None


def sweep_signals(legs: pl.DataFrame, Mn: pl.DataFrame) -> pl.DataFrame:
    """Per event and signal: the 'speech over' time (fixed delays after the start, and quiet
    periods after the last leg first printed >= 97c, both as of that time)."""
    st = said_times(Mn, legs)
    ev = legs.group_by("event").agg(pl.col("start").first())
    by = {e: g["said_ts"].to_list() for (e,), g in st.group_by(["event"])}
    rows = []
    for r in ev.iter_rows(named=True):
        for d in DELAYS:
            rows.append({"event": r["event"], "signal": f"delay{d}", "t_over": r["start"] + timedelta(minutes=d)})
        for q in QUIETS:
            said = [x for x in by.get(r["event"], []) if x is not None]
            # only said prints before the candidate time are known: quiet_time uses the sorted
            # sequence and stops at the first gap, so no later print enters the decision
            t = quiet_time(said, r["start"], timedelta(minutes=q))
            if t is not None:
                rows.append({"event": r["event"], "signal": f"quiet{q}", "t_over": t})
    return pl.DataFrame(rows, schema={"event": pl.String, "signal": pl.String, "t_over": pl.Datetime("us", "UTC")})


def sweep_trades(legs: pl.DataFrame, Mn: pl.DataFrame, sig: pl.DataFrame) -> pl.DataFrame:
    """At each signal time: every leg still open (close after t_over) whose last minute quote at or
    before t_over has YES bid >= 2c; buy 1 NO at 1 - bid (taker), fee rounded up per order."""
    q = (Mn.filter(pl.col("bid").is_not_null()).select("ticker", pl.col("end_ts").alias("q_ts"), "bid", "ask")
         .sort("q_ts"))
    d = legs.select("event", "ticker", "split", "et_date", "start", "close_time", "settlement_ts", "y",
                    "fee_multiplier").join(sig, on="event")
    d = d.filter(pl.col("close_time") > pl.col("t_over") + timedelta(minutes=1))
    d = d.sort("t_over").join_asof(q, left_on="t_over", right_on="q_ts", by="ticker", strategy="backward",
                                   check_sortedness=False)
    d = d.filter(pl.col("q_ts").is_not_null() & (pl.col("q_ts") > pl.col("start") - timedelta(minutes=30))
                 & (pl.col("bid") >= MIN_BID - EPS))
    px = (1 - d["bid"]).round(2).to_list()
    mult = d["fee_multiplier"].fill_null(1.0).to_list()
    fee = [taker_fee(p, 1, m) for p, m in zip(px, mult)]
    return d.with_columns(pl.Series("no_px", px), pl.Series("fee", fee)).with_columns(
        ((1 - pl.col("y")) - pl.col("no_px") - pl.col("fee")).alias("pnl"),
        (pl.col("t_over") + timedelta(minutes=1)).alias("t_in"))


def sweep_table(T: pl.DataFrame, splits: tuple[str, ...] = ("dev",)) -> pl.DataFrame:
    rows = []
    for sp in splits:
        for (sig,), g0 in T.filter(pl.col("split") == sp).group_by(["signal"]):
            for cap in CAPS:
                g = g0.filter(pl.col("bid") <= cap + EPS)
                if not g.height:
                    continue
                rows.append({"split": sp, "signal": sig, "cap": cap, **sweep_stats(g)})
    return pl.DataFrame(rows, infer_schema_length=None).sort("split", "signal", "cap")


def sweep_stats(g: pl.DataFrame) -> dict[str, Any]:
    x = g["pnl"].to_numpy()
    mu, te, ke = clustered_t(x, g["event"].to_numpy())
    _, td, _ = clustered_t(x, g["et_date"].cast(pl.String).to_numpy())
    win = float((g["y"] == 0).mean())
    be = float((g["no_px"] + g["fee"]).mean())
    return {"events": ke, "trades": g.height, "win_rate": round(win, 4), "breakeven": round(be, 4),
            "wilson_lb_events": round(wilson_lower(win * ke, ke), 4), "lb_minus_be": round(wilson_lower(win * ke, ke) - be, 4),
            "c_per_trade": round(mu * 100, 2), "t_event": None if te is None else round(te, 2),
            "t_date": None if td is None else round(td, 2), "pnl_usd": round(float(x.sum()), 2),
            "losses": int((g["y"] == 1).sum()), "avg_no_px": round(float(g["no_px"].mean()), 3)}


def choose_sweep(S: pl.DataFrame, min_events: int = 20) -> tuple[str, float] | None:
    """Pre-specified: among dev (signal, cap) cells with >= ``min_events`` events, the one with the
    largest Wilson-lower-bound margin over breakeven; None when no cell's bound clears breakeven."""
    ok = S.filter((pl.col("split") == "dev") & (pl.col("events") >= min_events)).sort("lb_minus_be", descending=True)
    if not ok.height or ok["lb_minus_be"][0] <= 0:
        return None
    return ok["signal"][0], float(ok["cap"][0])


def duration_quantiles(legs: pl.DataFrame, Mn: pl.DataFrame) -> pl.DataFrame:
    """Dev events: minutes from the scheduled start to the last first-said print (a lower bound on
    the speech's end), quantiles."""
    st = said_times(Mn, legs).join(legs.select("event", "start", "split").unique("event"), on="event")
    d = st.group_by("event").agg(((pl.col("said_ts").max() - pl.col("start").first()).dt.total_minutes()).alias("m"),
                                 pl.col("split").first())
    return d.group_by("split").agg(pl.len().alias("events"), *[pl.col("m").quantile(q).alias(f"q{int(q * 100)}")
                                                               for q in (0.5, 0.75, 0.9, 0.95)]).sort("split")


def sweep_ledger(g: pl.DataFrame, name: str, split: str) -> pl.DataFrame:
    led = g.select(pl.lit("mentions_ext").alias("strategy"), pl.lit(name).alias("variant"), pl.lit(split).alias("split"),
                   pl.col("ticker").alias("struct_id"), pl.col("event").alias("event_ticker"), "ticker",
                   pl.lit("no").alias("side"), pl.lit(1).alias("n"), "t_in", (pl.col("no_px") + pl.col("fee")).alias("cost_usd"),
                   pl.col("settlement_ts").alias("settle_ts"), (1 - pl.col("y")).cast(pl.Float64).alias("payoff_usd"))
    if led.height:
        _save(led, OUT / "ledgers" / f"ledger_{_slug(name)}_{split}.parquet")
    return led


def post_start_coverage(G: pl.DataFrame) -> pl.Series:
    """Legs whose minute candles were fetched over the post-start window: this study's fetch
    ([start - 10 min, min(close, start + 8 h)]) or an xcat first-3-days window spanning [start,
    min(close, start + 3 h)]."""
    have: set[str] = set()
    dp = CACHE / "minute" / "done.parquet"
    if dp.exists():
        have |= set(pl.read_parquet(dp)["ticker"])
    xs = XCAT / "sample.parquet"
    if xs.exists():
        w = (pl.scan_parquet(xs).filter(pl.col("layer") == "minute").select("ticker", "lo", "hi").collect()
             .join(G.select("ticker", "start", "close_time"), on="ticker"))
        w = w.filter((pl.col("lo") <= pl.col("start"))
                     & (pl.col("hi") >= pl.min_horizontal(pl.col("close_time"), pl.col("start") + timedelta(hours=3))))
        have |= set(w["ticker"])
    return pl.Series(sorted(have), dtype=pl.String)


def run_sweep(U: pl.DataFrame) -> dict[str, Any]:
    G = U.filter(pl.col("start") >= GRID_FROM)
    cov = post_start_coverage(G)
    share = G.group_by("event").agg(pl.col("ticker").is_in(cov.implode()).mean().alias("s"))
    legs = G.filter(pl.col("event").is_in(share.filter(pl.col("s") >= 0.8)["event"].implode())
                    & pl.col("ticker").is_in(cov.implode()))
    Mn = load_minute(legs["ticker"].to_list(), (CACHE / "minute", XCAT / "minute"))
    sig = sweep_signals(legs, Mn)
    T = sweep_trades(legs, Mn, sig)
    S = sweep_table(T)
    pick = choose_sweep(S)
    cv = legs.group_by("split").agg(pl.col("event").n_unique().alias("events"), pl.len().alias("legs"))
    out = {"trades": T, "dev": S, "pick": pick, "coverage": cv, "durations": duration_quantiles(legs, Mn)}
    if pick is not None:
        sg, cap = pick
        name = f"sweep/{sg}/cap{round(cap * 100)}c"
        rows, leds = [], {}
        for sp in ("dev", "holdout"):
            g = T.filter((pl.col("split") == sp) & (pl.col("signal") == sg) & (pl.col("bid") <= cap + EPS))
            if g.height:
                rows.append({"split": sp, "cell": name, **sweep_stats(g)})
                leds[sp] = sweep_ledger(g, name, sp)
        out["holdout"] = pl.DataFrame(rows, infer_schema_length=None)
        out["ledgers"] = leds
        out["name"] = name
    return out


# ---- Q4: Polymarket international mention markets as a signal (data only, never traded) -------------
PM_FROM, PM_TO = datetime(2026, 8, 10, tzinfo=UTC), datetime(2026, 11, 17, tzinfo=UTC)
PM_GAPS = (0.02, 0.05, 0.10)
_PM_COUNT = re.compile(r"\(?\s*(\d+)\+\s*times?\s*\)?", re.IGNORECASE)


def pm_clients():
    from albert.config import Settings
    from albert.http import CachedClient

    s = Settings()
    gamma = CachedClient("polymarket_gamma", s.pm_gamma_base, CACHE / "raw", min_interval=0.5, max_retries=6)
    data = CachedClient("polymarket_data", s.pm_data_base, CACHE / "raw", min_interval=0.5, max_retries=6)
    return gamma, data


def word_key(text: str | None) -> str:
    """Comparable leg key across venues: lower-cased ``/`` alternatives, sorted, plus the count
    threshold (``"Million / Billion 25+ times"`` and ``"Billion / Million (25+ times)"`` agree)."""
    s = text or ""
    m = _PM_COUNT.search(s)
    k = int(m.group(1)) if m else 1
    s = _PM_COUNT.sub("", s)
    alts = sorted({" ".join(a.lower().replace('"', "").split()) for a in s.split("/") if a.strip()})
    return " / ".join(alts) + (f" #{k}" if k > 1 else "")


def pm_market_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per word market of Polymarket mention events: event slug / title / end, word key,
    YES token and the resolved winner (1 = YES)."""
    from albert.venues.kalshi import parse_ts

    rows = []
    for e in events:
        for m in e.get("markets") or []:
            try:
                toks = json.loads(m.get("clobTokenIds") or "[]")
                outs = json.loads(m.get("outcomes") or "[]")
                prices = json.loads(m.get("outcomePrices") or "[]")
            except (TypeError, ValueError):
                continue
            if len(toks) != 2 or [o.lower() for o in outs] != ["yes", "no"]:
                continue
            win = None
            if prices and float(prices[0]) >= 0.99:
                win = 1
            elif prices and float(prices[1]) >= 0.99:
                win = 0
            rows.append({"pm_event": e.get("slug"), "pm_title": e.get("title"), "pm_end": parse_ts(e.get("endDate")),
                         "pm_start": parse_ts(e.get("startDate")), "word_key": word_key(m.get("groupItemTitle")),
                         "pm_word": m.get("groupItemTitle"), "token": str(toks[0]), "pm_y": win})
    return rows


def fetch_pm_events() -> pl.DataFrame:
    """Closed Polymarket events tagged ``mention-markets`` ending in [2026-08-10, 2026-11-17]
    (gamma ``/events`` ordered by end date, offset pages of 100, cached)."""
    gamma, _ = pm_clients()
    rows: list[dict[str, Any]] = []
    for off in range(0, 3000, 100):
        page = gamma.get("/events", {"tag_slug": "mention-markets", "closed": "true", "limit": 100, "offset": off,
                                     "order": "endDate", "ascending": "false",
                                     "end_date_min": PM_FROM.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                     "end_date_max": PM_TO.strftime("%Y-%m-%dT%H:%M:%SZ")})
        evs = page.payload if isinstance(page.payload, list) else []
        rows += pm_market_rows(evs)
        if len(evs) < 100:
            break
    df = pl.DataFrame(rows, infer_schema_length=None)
    _save(df, CACHE / "pm_markets.parquet")
    return df


def pm_speaker(title: str | None) -> str | None:
    t = (title or "").lower()
    if any(w in t for w in ("this week", " in october", " in november", " in september", " in august", " post ")):
        return None                                   # window markets, not one speech
    if "trump" in t:
        return "trump"
    if "vance" in t:
        return "vance"
    return None


def match_pm(U: pl.DataFrame, pm: pl.DataFrame) -> pl.DataFrame:
    """Kalshi leg <-> Polymarket market pairs: same speaker (Trump or Vance; window markets
    excluded), the Kalshi start within 30 h before to 2 h after the Polymarket end, >= 3 common
    word keys, and the same key."""
    K = U.filter(pl.col("sgroup").is_in(["trump", "vance"]) & (pl.col("start") >= GRID_FROM)).with_columns(
        pl.col("yes_sub_title").map_elements(word_key, return_dtype=pl.String).alias("word_key"))
    P = pm.with_columns(pl.col("pm_title").map_elements(pm_speaker, return_dtype=pl.String).alias("sgroup")).filter(
        pl.col("sgroup").is_not_null())
    ke = K.group_by("event").agg(pl.col("sgroup").first(), pl.col("start").first(), pl.col("word_key"))
    pe = P.group_by("pm_event").agg(pl.col("sgroup").first(), pl.col("pm_end").first(), pl.col("word_key").alias("pm_keys"))
    c = ke.join(pe, on="sgroup").filter((pl.col("start") >= pl.col("pm_end") - timedelta(hours=30))
                                        & (pl.col("start") <= pl.col("pm_end") + timedelta(hours=2)))
    c = c.with_columns(pl.col("word_key").list.set_intersection("pm_keys").list.len().alias("common"))
    c = c.filter(pl.col("common") >= 3).sort("common", descending=True).unique("event", keep="first").unique("pm_event", keep="first")
    pairs = K.join(c.select("event", "pm_event"), on="event").join(P.select("pm_event", "word_key", "token", "pm_y", "pm_word"),
                                                                   on=["pm_event", "word_key"])
    return pairs.select("event", "ticker", "pm_event", "token", "pm_word", "pm_y", "y", "start")


def fetch_pm_history(pairs: pl.DataFrame) -> pl.DataFrame:
    """Hourly Polymarket YES price buckets over [start - 30 h, start] per matched token."""
    _, data = pm_clients()
    rows = []
    for i, r in enumerate(pairs.unique("token").iter_rows(named=True), 1):
        lo, hi = int((r["start"] - timedelta(hours=30)).timestamp()), int(r["start"].timestamp())
        try:
            p = data.get("/prices-history", {"token_id": r["token"], "start": lo, "end": hi, "bucket_seconds": 3600})
        except RuntimeError as error:
            print(f"skip {r['token'][:12]}: {str(error)[-120:]}", flush=True)
            continue
        for x in p.payload.get("data") or []:
            if x.get("timestamp") is not None and x.get("price") is not None:
                rows.append({"token": r["token"], "bucket_ts": datetime.fromtimestamp(int(x["timestamp"]), UTC),
                             "secs": int(x.get("resolution_seconds") or 0), "pm_price": float(x["price"])})
        if i % 100 == 0:
            print(f"  pm history {i}", flush=True)
    df = pl.DataFrame(rows, schema={"token": pl.String, "bucket_ts": pl.Datetime("us", "UTC"), "secs": pl.Int64,
                                    "pm_price": pl.Float64})
    _save(df, CACHE / "pm_history.parquet")
    return df


def pm_signal(o: pl.DataFrame, pairs: pl.DataFrame, hist: pl.DataFrame) -> pl.DataFrame:
    """Polymarket YES price as of each order's ``t`` (the last hourly bucket that ENDED by t) and
    the gap Polymarket - Kalshi mid."""
    h = hist.with_columns((pl.col("bucket_ts") + pl.duration(seconds=pl.col("secs"))).alias("known_ts")).sort("known_ts")
    x = o.join(pairs.select("ticker", "token"), on="ticker").sort("t")
    x = x.join_asof(h.select("token", "known_ts", "pm_price"), left_on="t", right_on="known_ts", by="token",
                    strategy="backward", check_sortedness=False)
    x = x.filter(pl.col("pm_price").is_not_null() & (pl.col("t") - pl.col("known_ts") <= timedelta(hours=6)))
    return x.with_columns((pl.col("pm_price") - pl.col("mid")).alias("pm_gap"))


def pm_tables(F: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Live-rule fills with a Polymarket price: PnL by gap sign and the dev-chosen skip filter."""
    from albert.research.mentions_history import clustered_slope

    f = F.filter("filled")
    out: dict[str, pl.DataFrame] = {}
    rows = []
    for (sp,), g in f.group_by(["split"]):
        _a, b, _ta, tb = clustered_slope(g["pm_gap"].to_numpy(), g["pnl"].to_numpy() * 100, g["event"].to_numpy()) \
            if g["event"].n_unique() > 2 else (0, float("nan"), 0, float("nan"))
        rows.append({"split": sp, "fills": g.height, "events": g["event"].n_unique(), **summarize(g.with_columns(pl.lit(True).alias("filled"))),
                     "slope_c_per_gap": round(b, 2), "t_slope": round(tb, 2)})
    out["q4_fills"] = pl.DataFrame(rows, infer_schema_length=None).sort("split")
    grid = []
    d = f.filter(pl.col("split") == "dev")
    for gp in PM_GAPS:
        sk = d.filter(pl.col("pm_gap") > gp)
        grid.append({"gap": gp, "fills_skipped": sk.height, "skipped_pnl_usd": round(float(sk["pnl"].sum()), 2),
                     "skipped_c_per_fill": round(float(sk["pnl"].mean()) * 100, 2) if sk.height else None,
                     "said_share_skipped": round(float(sk["y"].mean()), 3) if sk.height else None})
    G = pl.DataFrame(grid)
    out["q4_grid_dev"] = G
    ok = G.filter((pl.col("fills_skipped") >= 15) & (pl.col("skipped_pnl_usd") < 0)).sort("skipped_pnl_usd")
    if ok.height:
        gp = float(ok["gap"][0])
        h = f.filter(pl.col("split") == "holdout")
        sk, kp = h.filter(pl.col("pm_gap") > gp), h.filter(~(pl.col("pm_gap") > gp))
        out["q4_holdout"] = pl.DataFrame([{"gap": gp, "set": "skipped", **summarize(sk.with_columns(pl.lit(True).alias("filled")))},
                                          {"gap": gp, "set": "kept", **summarize(kp)}], infer_schema_length=None)
    return out


# ---- driver and report -----------------------------------------------------------------------------------
def md_table(df: pl.DataFrame, max_rows: int = 80) -> str:
    if df is None or not df.height:
        return "(no rows)\n"
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.head(max_rows).iter_rows():
        lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def _salience_book(F: pl.DataFrame, pick: tuple[str, float] | None, marks: pl.DataFrame) -> pl.DataFrame:
    """Live-rule capped book with and without the chosen salience filter (grid sample only, where
    the book is complete), dev and holdout."""
    rows = []
    G = F.filter(pl.col("start") >= GRID_FROM)
    variants = [("live", G)]
    if pick is not None:
        feat, k = pick
        variants.append((f"live+skip {feat}>{k}", G.filter(~(pl.col(feat).fill_null(-99) > k))))
    for name, o in variants:
        for sp in ("dev", "holdout"):
            rows.append(cell_row(o, marks, f"salience/{name}", sp, save=pick is not None))
    return pl.DataFrame(rows, infer_schema_length=None)


def run_all(U: pl.DataFrame, skip: tuple[str, ...] = ()) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pl.DataFrame] = {}
    info: dict[str, Any] = {}
    # Q2 ------------------------------------------------------------------------------------------
    print("Q2 grid", flush=True)
    g = run_grid(U)
    info["q2_chosen"], info["q2_live"] = g["chosen"], g["live"]
    tables["q2_coverage"] = g["coverage"]
    tables["q2_grid_dev"] = g["dev"]
    tables["q2_holdout"] = g["holdout"]
    live6 = g["orders"][g["live"]]
    tables["q2_prints_check"] = prints_check(live6, g["Qs"][LIVE_L])
    tables["q2_prints_check_chosen"] = (prints_check(g["orders"][g["chosen"]], g["Qs"][LIVE_L])
                                        if g["chosen"].startswith(f"L{LIVE_L}h/") else pl.DataFrame())
    ch = g["orders"][g["chosen"]]
    tables["q2_chosen_by_phase"] = pl.DataFrame(
        [{"cell": g["chosen"], "split": sp, "phase": ph, **summarize(ch.filter((pl.col("split") == sp) & (pl.col("phase") == ph)))}
         for sp in ("dev", "holdout") for ph in sorted(ch["phase"].unique().to_list())
         if ch.filter((pl.col("split") == sp) & (pl.col("phase") == ph)).height], infer_schema_length=None)
    if "late" in g["Qs"] and g["chosen"].endswith("/late"):
        late_tk = g["Qs"]["late"]["ticker"].to_list()
        dp = CACHE / "minute_late" / "done.parquet"
        covered = set(pl.read_parquet(dp)["ticker"]) if dp.exists() else set()
        tables["q2_minute_check_late"] = minute_check(ch, g["Qs"]["late"], load_minute(late_tk, (CACHE / "minute_late",)),
                                                      covered)
    marks = g["marks"]
    del g, ch, live6
    gc.collect()
    print("Q2 long history", flush=True)
    tables["q2_long_history_dev"] = long_history(U)
    gc.collect()
    # Q1 ------------------------------------------------------------------------------------------
    pvp = CACHE / "pageviews.parquet"
    if pvp.exists() and "q1" not in skip:
        print("Q1 salience", flush=True)
        pv = pl.read_parquet(pvp)
        F = salience(live_orders(U, LIVE_M), pv)
        _save(F.drop("yes_sub_title"), OUT / "q1_live_orders_salience.parquet")
        cov = F.filter("filled").group_by("split").agg(pl.len().alias("fills"), pl.col("z3").is_not_null().sum().alias("with_salience"),
                                                       pl.col("event").n_unique().alias("events"))
        tables["q1_coverage"] = cov.sort("split")
        SG = salience_grid(F.filter(pl.col("split") == "dev"))
        tables["q1_grid_dev"] = SG
        pick = choose_salience(SG)
        info["q1_pick"] = pick
        tables["q1_bins_z3"] = salience_bins(F, "z3")
        tables["q1_slopes"] = pl.concat([salience_slope(F, f) for f in SAL_FEATURES])
        if pick is not None:
            feat, k = pick
            ho = F.filter((pl.col("split") == "holdout") & pl.col("filled"))
            skip, keep = ho.filter(pl.col(feat).fill_null(-99) > k), ho.filter(~(pl.col(feat).fill_null(-99) > k))
            tables["q1_holdout"] = pl.DataFrame([{"set": "skipped", **summarize(skip.with_columns(pl.lit(True).alias("filled")))},
                                                 {"set": "kept", **summarize(keep)}], infer_schema_length=None)
        tables["q1_book"] = _salience_book(F, pick, marks)
        del F
        gc.collect()
    # Q3 ------------------------------------------------------------------------------------------
    print("Q3 sweep", flush=True)
    sw = run_sweep(U)
    info["q3_pick"] = sw["pick"]
    tables["q3_coverage"] = sw["coverage"]
    tables["q3_durations"] = sw["durations"]
    tables["q3_grid_dev"] = sw["dev"]
    if "holdout" in sw:
        tables["q3_holdout"] = sw["holdout"]
        tables["q3_book"] = pl.DataFrame([{"split": sp, **book_metrics(led, marks_from(load_hourly(led["ticker"].to_list())))}
                                          for sp, led in sw["ledgers"].items()], infer_schema_length=None)
    _save(sw["trades"], OUT / "q3_trades.parquet")
    # Q4 ------------------------------------------------------------------------------------------
    if (CACHE / "pm_history.parquet").exists():
        print("Q4 Polymarket", flush=True)
        pairs = pl.read_parquet(CACHE / "pm_pairs.parquet")
        F4 = pm_signal(live_orders(U, LIVE_M), pairs, pl.read_parquet(CACHE / "pm_history.parquet"))
        tables["q4_coverage"] = pairs.group_by(pl.when(pl.col("start") < HOLDOUT_FROM).then(pl.lit("dev")).otherwise(pl.lit("holdout"))
                                               .alias("split")).agg(pl.col("event").n_unique().alias("events"), pl.len().alias("legs"),
                                                                    (pl.col("pm_y") == pl.col("y")).mean().alias("same_result")).sort("split")
        tables.update(pm_tables(F4))
    for name, t in tables.items():
        if t is not None and t.height:
            t.write_csv(OUT / f"{name}.csv")
    (OUT / "results.json").write_text(json.dumps(info, indent=1, default=str))
    write_report(tables, info)


SECTIONS = [
    ("Q2. Entry window and margin", ["q2_coverage", "q2_grid_dev", "q2_holdout", "q2_chosen_by_phase",
                                     "q2_minute_check_late", "q2_prints_check", "q2_prints_check_chosen",
                                     "q2_long_history_dev"]),
    ("Q1. Salience filter (Wikimedia pageviews)", ["q1_coverage", "q1_grid_dev", "q1_holdout", "q1_bins_z3", "q1_slopes",
                                                   "q1_book"]),
    ("Q3. Post-speech NO sweep", ["q3_coverage", "q3_durations", "q3_grid_dev", "q3_holdout", "q3_book"]),
    ("Q4. Polymarket mention markets as a signal", ["q4_coverage", "q4_fills", "q4_grid_dev", "q4_holdout"]),
]


BOOK_COLS = ["cell", "split", "events", "orders", "fills", "c_per_fill", "t_date", "t_event", "pnl_usd", "book_fills",
             "book_c_per_fill", "book_pnl_usd", "book_usd_per_week", "book_ann_vol_pct", "book_sharpe", "book_sharpe_ci90",
             "book_sharpe_ex_top5", "book_max_dd_usd", "book_weeks_positive_pct"]
MD_COLS = {"q2_grid_dev": BOOK_COLS, "q2_holdout": BOOK_COLS, "q1_book": BOOK_COLS}


def write_report(tables: dict[str, pl.DataFrame], info: dict[str, Any]) -> Path:
    parts = ["# Mentions extensions (`research/mentions_ext.py`)\n"]
    v = OUT / "verdict.md"
    if v.exists():
        parts.append(v.read_text())
    parts.append("\n## Run info\n\n```\n" + json.dumps(info, indent=1, default=str) + "\n```\n")
    for title, keys in SECTIONS:
        parts.append(f"\n## {title} (tables)\n")
        for k in keys:
            if k in tables and tables[k] is not None and tables[k].height:
                t = tables[k]
                if k in MD_COLS:
                    t = t.select([c for c in MD_COLS[k] if c in t.columns])
                parts.append(f"\n### {k}\n\n" + md_table(t, 120))
    parts.append("\nFull columns (book = $15 capped-book ledger metrics) are in the CSV files next to this report.\n")
    parts.append("\n## Design\n\n" + (__doc__ or ""))
    p = OUT / "report.md"
    p.write_text("\n".join(parts))
    return p


# ---- CLI ---------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.mentions_ext")
    ap.add_argument("step", choices=["universe", "fetch-hourly", "fetch-minute", "fetch-late", "fetch-pageviews", "fetch-pm",
                                     "run"])
    ap.add_argument("--skip", nargs="*", default=[], help="questions to leave out of `run` (e.g. q1)")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "universe":
        U = universe()
        _save(U, OUT / "universe.parquet")
        print(U.group_by("split", "sgroup").agg(pl.len().alias("legs"), pl.col("event").n_unique().alias("events"))
              .sort("split", "sgroup"))
        return
    U = pl.read_parquet(OUT / "universe.parquet")
    if a.step == "fetch-hourly":
        fetch_candles(hourly_gaps(U), 60, CACHE / "hourly", label="hourly")
    elif a.step == "fetch-minute":
        fetch_candles(minute_jobs(U), 1, CACHE / "minute", label="minute")
    elif a.step == "fetch-late":
        fetch_candles(late_minute_jobs(U), 1, CACHE / "minute_late", label="minute_late")
    elif a.step == "fetch-pageviews":
        fetch_pageviews(pageview_titles(U))
    elif a.step == "fetch-pm":
        pm = fetch_pm_events()
        pairs = match_pm(U, pm)
        _save(pairs, CACHE / "pm_pairs.parquet")
        print(f"pm markets {pm.height}, matched legs {pairs.height} in {pairs['event'].n_unique()} events", flush=True)
        fetch_pm_history(pairs)
    else:
        run_all(U, tuple(a.skip))


if __name__ == "__main__":
    main()
