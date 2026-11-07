"""Mentions history: is the pre-speech NO-bid pattern stable over the whole Kalshi history?

Round 2 (:mod:`albert.research.mentions_maker`) found one holdout-positive lead: resting wide NO
bids (selling YES 8–12c above fair) on speech legs from 6 h before the scheduled start, filled
only by prints that trade through the bid, held to settlement. October 2026: +4.8c per fill
(t 2.8) at the pre-specified 8c margin; Aug–Sep weaker; and quoting around the YES-bias-corrected
mid alone earned the same, so it is a *price* pattern, not transcript skill. This study freezes
that rule and runs it month by month over every settled Mentions-category event Kalshi has
(series launches in 2025 to November 2026), per series and per speaker. Nothing is re-tuned.

Frozen rule (every input as of the decision time ``t``):

* **Events / legs.** Every settled leg of every Mentions-category series (volume-blind), speech
  events only (window series "say X before <date>" have no start and are excluded, as in round 2;
  "Event does not qualify" legs dropped). Event start = the earliest Kalshi milestone
  ``start_date`` listing the event (round 1/2 rule), from ``/milestones?category=Mentions``.
* **Decision time** ``t = start − 6 h``; the leg must be listed and open at ``t``; quote = last
  hourly candle closed by ``t`` (≤ 2 h old, ask > bid). Outcome-blind early-start exclusion
  (round 2): drop an event if the share of its quoted legs bid ≥ 0.97 at ``t`` exceeds the share
  at ``start − 24 h`` by > 0.12 (the speech probably came earlier than the milestone).
* **Fair value** = the round-2 dev fit of the calibrated mid, frozen:
  ``logit fair = −0.1156 + 1.0474·logit(mid)`` (``fv_cal``, fitted on Aug 17–Sep 30 2026 rows; the
  logit clips the mid to [0.005, 0.995] and the fair value to [0.003, 0.997], as in round 2).
  Control: the raw mid (``fv_mid``).
* **Orders.** NO bid at ``floor(1 − fair − m)`` (sell YES at ``fair + m`` or more), ``m`` ∈ {8c,
  12c}, capped one tick inside the yes bid (post-only); optional YES bid at ``floor(fair − m)``,
  capped one tick under the ask. One contract each; bids < 1c are not posted. Orders rest from
  ``t`` until ``end = start − 10 min`` (or the leg's close if earlier).
* **Circuit breaker** (``cb``, the primary variant): cancel every order of the event at the first
  print ≥ 95c on a leg whose ask at ``t`` was ≤ 80c (a word was just said: the speech started
  early). Live-implementable from the public trade feed.
* **Fills.** A bid fills only if a later public print trades strictly THROUGH it (a YES taker at a
  yes price above our ask for the NO bid; a NO taker below our bid for the YES bid); the queue at
  our own price never clears. Held to settlement; Mentions series carry no maker fee
  (``quadratic`` fee type, taker-only), applied by series via the harness ``maker_fee``.

Outputs: monthly PnL per fill and per order, fills, events, t clustered by event; the share of
months positive; a trend test of PnL per fill on calendar time (competition / decay); capacity
(contracts printed through our price per filled order and per event); per series and speaker;
split by quoted spread at ``t`` (≤ 5c, 6–15c, > 15c) and by price band; a taker reference (buy
NO at ``1 − bid`` at ``t``).

Data (public Kalshi API only, cached under ``data/cache/mentions_history/``): ``/series``,
``/historical/markets`` + ``/markets`` per series, ``/milestones`` by category, hourly
candlesticks per leg (live batch endpoint for events from Aug 15 2026; one historical request per
leg before that, because the batch endpoint returns nothing for older markets), and public prints
in ``(t, end]``. Events of Aug 15 – Nov 1 2026 reuse round 2's cached Kalshi candles and prints.

Fill source, per event: **prints** when every leg whose hourly trade range could hold a fill
(:func:`fill_candidates`, exact: the hourly high / low bound every print) has fetched prints;
otherwise the **hourly-candle proxy** (:func:`proxy_fills`: the bid fills if an hour overlapping
``(t, end]`` traded beyond it; breaker = the whole hour of the first ≥ 95c print on a cheap leg
is cancelled). The proxy is checked against prints wherever both exist (report table P).

Sampling (the shared public rate limit allowed ~0.5–7 requests / s): candles are fetched in a
hash-random event order with weight 1 for Trump and Vance and 0.06 for every other speaker and
for earnings calls (``u / w``, ``u`` uniform from the event ticker), and the fetch was stopped
after Trump / Vance were complete and the other groups had reached a volume-blind random sample
(run of 2026-11-06: Trump 285/285 events, Vance 62/62, other speakers 411/1,733, earnings
192/483, every event of Aug 15 – Nov 5 2026 included; coverage in report table 0).

Entry point: ``python -m albert.research.mentions_history {markets,candles,trades [N],run}``.
"""

from __future__ import annotations

import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.research import mentions as M

NAME = "mentions_history"
LEAD = timedelta(hours=6)
H24 = timedelta(hours=24)
END_MARGIN = timedelta(minutes=10)
MAX_AGE = timedelta(hours=2)
FROZEN_A, FROZEN_B = -0.1156, 1.0474        # fv_cal, research-fundamental-mentions_maker-20261106T010604Z
MARGINS = (0.08, 0.12)
CB_PRINT, CB_ASK = 0.95, 0.80
SAID_BID, SAID_JUMP = 0.97, 0.12
ROUND2_FROM = datetime(2026, 8, 15, tzinfo=UTC)  # round-2 caches cover legs settled from here
ROUND2_TO = datetime(2026, 11, 1, tzinfo=UTC)
DEV_FROM, HOLDOUT_FROM = datetime(2026, 8, 17, tzinfo=UTC), datetime(2026, 10, 1, tzinfo=UTC)


def cache_dir(s) -> Path:
    d = s.data_dir / "cache" / NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def client(s, min_interval: float = 0.25):
    from albert.http import CachedClient

    return CachedClient("kalshi", s.kalshi_base, cache_dir(s) / "raw", min_interval=min_interval)


def _save(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _day() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


# ---- series, markets, milestones ----------------------------------------------------------------
def mention_series(s) -> pl.DataFrame:
    """Every Mentions-category series (``/series?category=Mentions``, refreshed daily)."""
    from albert.venues import kalshi as k

    page = client(s).get("/series", {"category": "Mentions"}, tag=_day())
    return k.series_frame(page.payload).filter(pl.col("category") == "Mentions")


def _pages(c, path: str, params: dict[str, Any], key: str, *, tag: str | None = None) -> list[dict[str, Any]]:
    out, cursor = [], None
    while True:
        page = c.get(path, {**params, "cursor": cursor}, tag=tag)
        out.append(page.payload)
        cursor = page.payload.get("cursor")
        if not cursor or not page.payload.get(key):
            return out


def fetch_markets(s, series: pl.DataFrame, *, workers: int = 2) -> pl.DataFrame:
    """Every settled leg of every series: ``/historical/markets?series_ticker=`` (before the
    historical cutoff; immutable, cached) plus ``/markets?series_ticker=&status=settled`` (cached
    per day). Saved to ``data/cache/mentions_history/markets.parquet``."""
    from albert.venues import kalshi as k

    c = client(s)
    info = {r["series_ticker"]: r for r in series.iter_rows(named=True)}
    day = _day()

    def one(sr: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            for p in _pages(c, "/historical/markets", {"series_ticker": sr, "limit": 1000}, "markets"):
                rows.extend(k.market_rows(p, info, "historical"))
            for p in _pages(c, "/markets", {"series_ticker": sr, "status": "settled", "limit": 1000},
                                            "markets", tag=day):
                rows.extend(k.market_rows(p, info, "live"))
        except RuntimeError as error:
            print(f"skip series {sr}: {str(error)[:120]}", flush=True)
        return rows

    rows: list[dict[str, Any]] = []
    names = series["series_ticker"].to_list()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, got in enumerate(pool.map(one, names), 1):
            rows.extend(got)
            if i % 50 == 0:
                print(f"  series {i}/{len(names)} legs {len(rows)}", flush=True)
    df = k.markets_frame(rows, non_financial_only=False).unique("ticker", keep="first")
    _save(df, cache_dir(s) / "markets.parquet")
    return df


def fetch_milestones(s, events: list[str] | None = None) -> pl.DataFrame:
    """Earliest milestone start per event from ``/milestones?category=Mentions`` (paged, cached
    per day), plus a per-event ``related_event_ticker`` query for events the listing misses."""
    c = client(s)
    rows: list[dict[str, Any]] = []
    for p in _pages(c, "/milestones", {"category": "Mentions", "limit": 500}, "milestones", tag=_day()):
        rows.extend(M.milestone_rows(p))
    df = pl.DataFrame(rows, infer_schema_length=None)
    missing = sorted(set(events or []) - set(df["event"]))
    extra = []
    for i, ev in enumerate(missing, 1):
        try:
            page = c.get("/milestones", {"related_event_ticker": ev, "limit": 10})
        except RuntimeError as error:
            print(f"milestone {ev}: {str(error)[:100]}", flush=True)
            continue
        extra.extend(r for r in M.milestone_rows(page.payload) if r["event"] == ev)
        if i % 100 == 0:
            print(f"  milestones {i}/{len(missing)}", flush=True)
    if extra:
        df = pl.concat([df, pl.DataFrame(extra, infer_schema_length=None)], how="diagonal_relaxed")
    df = df.sort("start").unique("event", keep="first")
    _save(df, cache_dir(s) / "starts.parquet")
    return df


def speech_legs(s) -> pl.DataFrame:
    """Settled speech legs (round-1 parser) with their event start; window series dropped."""
    mk = pl.read_parquet(cache_dir(s) / "markets.parquet")
    starts = pl.read_parquet(cache_dir(s) / "starts.parquet")
    lg = M.legs(mk).filter(~pl.col("series").is_in(M.WINDOW_SERIES))
    fees = mk.select("ticker", "fee_type", "fee_multiplier", "source")
    return (lg.join(starts.select("event", "start", pl.col("end").alias("ms_end"), "ms_title", "ms_updated"), on="event")
            .join(fees, on="ticker", how="left"))


# ---- candles (quotes at t, start − 24 h; hourly trade range) ------------------------------------
CANDLE_SCHEMA = {"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "yes_bid_close": pl.Float64,
                 "yes_ask_close": pl.Float64, "price_low": pl.Float64, "price_high": pl.Float64,
                 "volume": pl.Float64}


def candle_rows(candles: list[dict[str, Any]], ticker: str) -> list[dict[str, Any]]:
    """Hourly candles of one market (live ``*_dollars`` or historical bare field names): closing
    yes bid / ask and the hour's trade-price low / high (null when nothing traded)."""
    from albert.venues import kalshi as k

    rows = []
    for c in candles or []:
        price, bid, ask = c.get("price") or {}, c.get("yes_bid") or {}, c.get("yes_ask") or {}
        rows.append({"ticker": ticker, "end_ts": k.parse_ts(c.get("end_period_ts")),
                     "yes_bid_close": k.num(k._field(bid, "close")), "yes_ask_close": k.num(k._field(ask, "close")),
                     "price_low": k.num(k._field(price, "low")), "price_high": k.num(k._field(price, "high")),
                     "volume": k.num(k._field(c, "volume"))})
    return rows


BATCH_FROM = datetime(2026, 8, 15, tzinfo=UTC)   # live batch candles return nothing for older events


def candle_jobs(lg: pl.DataFrame) -> list[dict[str, Any]]:
    """One job per event: the legs open at ``start − 6 h`` and the window [start − 27 h, start];
    ``batch`` (live batch endpoint, ≤ 100 legs per request) for events from Aug 15 2026, else one
    historical request per leg."""
    d = lg.filter((pl.col("open_time") <= pl.col("start") - LEAD) & (pl.col("close_time") > pl.col("start") - LEAD))
    jobs = []
    for (ev,), g in d.group_by(["event"]):
        st = g["start"][0]
        ticks = sorted(g["ticker"].to_list())
        for j in range(0, len(ticks), 100):
            jobs.append({"job": f"{ev}#{j // 100}", "event": ev, "tickers": ticks[j:j + 100], "batch": st >= BATCH_FROM,
                         "lo": st - timedelta(hours=27), "hi": st})
    return sorted(jobs, key=lambda x: x["job"])


def fetch_candles(s, jobs: list[dict[str, Any]], *, workers: int = 2, save_every: int = 20) -> pl.DataFrame:
    """Hourly candles per job: batch jobs use one live ``/markets/candlesticks`` request; legs it
    omits (and every leg of older events) are fetched one by one — historical
    ``/historical/markets/{ticker}/candlesticks`` before the market-settlement cutoff, else the live
    per-market endpoint. Every response is cached under data/cache/mentions_history/raw, rows in
    ``candles.parquet``, finished jobs in ``candle_jobs.parquet`` (failed jobs retried later)."""
    from albert.venues import kalshi as k

    c = client(s)
    cutoff = k.Kalshi(c).cutoff().get("market_settled_ts")
    path, done_path = cache_dir(s) / "candles.parquet", cache_dir(s) / "candle_jobs.parquet"
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=CANDLE_SCHEMA)
    done = set(pl.read_parquet(done_path)["job"]) if done_path.exists() else set()
    todo = [j for j in jobs if j["job"] not in done]

    def get(path_: str, prm: dict[str, Any]):
        """``c.get`` with an outer retry on rate limiting (the client's own retries last ~15 s)."""
        for attempt in range(5):
            try:
                return c.get(path_, prm)
            except RuntimeError as error:
                if "429" not in str(error) or attempt == 4:
                    raise
                time.sleep(15 * (attempt + 1))
        raise RuntimeError("unreachable")

    def one(job: dict[str, Any]) -> tuple[str, list[dict[str, Any]], bool]:
        rows: list[dict[str, Any]] = []
        lo, hi = int(job["lo"].timestamp()), int(job["hi"].timestamp())
        prm = {"start_ts": lo, "end_ts": hi, "period_interval": 60}
        try:
            got = set()
            if job["batch"]:
                page = get("/markets/candlesticks", {"market_tickers": ",".join(job["tickers"]), **prm})
                for m in page.payload.get("markets") or []:
                    got.add(m["market_ticker"])
                    rows.extend(candle_rows(m.get("candlesticks"), m["market_ticker"]))
            for tk in sorted(set(job["tickers"]) - got) if job.get("fallback", True) else []:
                live = cutoff is not None and job["hi"] >= cutoff
                path_ = f"/series/{k.series_of(tk)}/markets/{tk}/candlesticks" if live else f"/historical/markets/{tk}/candlesticks"
                try:
                    p = get(path_, prm)
                except RuntimeError as error:
                    if "404" not in str(error):
                        raise
                    p = get(f"/historical/markets/{tk}/candlesticks" if live else
                            f"/series/{k.series_of(tk)}/markets/{tk}/candlesticks", prm)
                rows.extend(candle_rows(p.payload.get("candlesticks"), tk))
        except RuntimeError as error:
            print(f"skip candles {job['job']}: {str(error)[-160:]}", flush=True)
            return job["job"], rows, False
        return job["job"], rows, True

    new_rows: list[dict[str, Any]] = []
    new_done: list[str] = []

    def flush() -> None:
        nonlocal have
        if new_rows:
            have = pl.concat([have, pl.DataFrame(new_rows, schema=CANDLE_SCHEMA)]).unique(["ticker", "end_ts"])
            _save(have, path)
        _save(pl.DataFrame({"job": sorted(done | set(new_done))}, schema={"job": pl.String}), done_path)
        new_rows.clear()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (job, rows, ok) in enumerate(pool.map(one, todo), 1):
            if ok:
                new_rows.extend(rows)
                new_done.append(job)
            if i % save_every == 0:
                flush()
                print(f"  candle jobs {i}/{len(todo)} {job}", flush=True)
    flush()
    return have


def round2_candles(s) -> pl.DataFrame:
    """Round 2's cached hourly Kalshi candles (``fundamental_mentions_candles*.parquet``: quotes and
    volume, no trade range) in this study's schema; ``price_low/high`` are null, so an hour with
    volume > 0 counts as a possible fill (:func:`fill_candidates`)."""
    frames = []
    for name in ("fundamental_mentions_candles.parquet", "fundamental_mentions_candles_batch.parquet"):
        p = s.data_dir / "cache" / name
        if p.exists():
            frames.append(pl.read_parquet(p))
    if not frames:
        return pl.DataFrame(schema=CANDLE_SCHEMA)
    c = pl.concat(frames, how="diagonal_relaxed").filter(pl.col("period_minutes") == 60)
    c = c.with_columns(pl.lit(None, dtype=pl.Float64).alias("price_low"), pl.lit(None, dtype=pl.Float64).alias("price_high"))
    return c.select(list(CANDLE_SCHEMA)).unique(["ticker", "end_ts"])


def all_candles(s) -> pl.DataFrame:
    """This study's candles, plus round 2's for tickers this study did not fetch."""
    p = cache_dir(s) / "candles.parquet"
    mine = pl.read_parquet(p) if p.exists() else pl.DataFrame(schema=CANDLE_SCHEMA)
    r2 = round2_candles(s).filter(~pl.col("ticker").is_in(mine["ticker"].unique().implode()))
    return pl.concat([mine, r2])


# ---- trades in [t, end] ----------------------------------------------------------------------------
TRADE_SCHEMA = {"trade_id": pl.String, "ticker": pl.String, "created_time": pl.Datetime("us", "UTC"),
                "count": pl.Float64, "yes_price": pl.Float64, "taker_side": pl.String}


def fetch_trades(s, windows: pl.DataFrame, *, workers: int = 2, save_every: int = 300) -> pl.DataFrame:
    """Every public print of each leg in its window (``ticker, t_from, t_to``): historical
    ``/historical/trades`` before the trades cutoff, live ``/markets/trades`` after. Rows in
    ``trades.parquet``, finished windows in ``trade_windows.parquet`` (failures retried)."""
    from albert.venues import kalshi as k

    c = client(s)
    api = k.Kalshi(c)
    cutoff = api.cutoff().get("trades_created_ts")
    path, done_path = cache_dir(s) / "trades.parquet", cache_dir(s) / "trade_windows.parquet"
    wschema = {"ticker": pl.String, "t_from": pl.Datetime("us", "UTC"), "t_to": pl.Datetime("us", "UTC")}
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=TRADE_SCHEMA)
    done = pl.read_parquet(done_path) if done_path.exists() else pl.DataFrame(schema=wschema)
    todo = windows.select("ticker", "t_from", "t_to").unique().join(done, on=["ticker", "t_from", "t_to"], how="anti")

    def one(r: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
        historical = bool(cutoff and r["t_to"] < cutoff)
        out: list[dict[str, Any]] = []
        try:
            for page in api.trades(r["ticker"], historical, min_ts=r["t_from"], max_ts=r["t_to"] + timedelta(seconds=1)):
                out.extend(k.trade_rows(page.payload))
        except RuntimeError as error:
            print(f"skip trades {r['ticker']}: {str(error)[-120:]}", flush=True)
            return r, out, False
        return r, out, True

    new_rows: list[dict[str, Any]] = []
    new_done: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal have, done
        if new_rows:
            have = pl.concat([have, pl.DataFrame(new_rows, schema=TRADE_SCHEMA)]).unique("trade_id")
            _save(have, path)
        if new_done:
            done = pl.concat([done, pl.DataFrame(new_done, schema=wschema)])
            _save(done, done_path)
        new_rows.clear()
        new_done.clear()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (r, rows, ok) in enumerate(pool.map(one, todo.iter_rows(named=True)), 1):
            if ok:
                new_rows.extend(rows)
                new_done.append(r)
            if i % save_every == 0:
                flush()
                print(f"  trade windows {i}/{todo.height}", flush=True)
    flush()
    return have


# ---- decision rows (frozen rule) ------------------------------------------------------------------
def fair_cal(mid: np.ndarray) -> np.ndarray:
    """Frozen round-2 calibrated mid: ``logit fair = A + B·logit(mid)`` (harness logit clip)."""
    from albert.research.fundamental import logit

    return np.clip(1 / (1 + np.exp(-(FROZEN_A + FROZEN_B * logit(np.asarray(mid, dtype=float))))), 0.003, 0.997)


def spread_band(sp: pl.Expr) -> pl.Expr:
    return (pl.when(sp <= 0.05 + 1e-9).then(pl.lit("a <=5c")).when(sp <= 0.15 + 1e-9).then(pl.lit("b 6-15c"))
            .otherwise(pl.lit("c >15c")))


def speaker_group(series: pl.Expr) -> pl.Expr:
    """Trump (main + secondary events), Vance, earnings calls, TV / news shows, sports
    broadcasts, other politicians / people."""
    s = series
    return (pl.when(s.is_in(["KXTRUMPMENTION", "KXTRUMPMENTIONB"])).then(pl.lit("trump"))
            .when(s == "KXVANCEMENTION").then(pl.lit("vance"))
            .when(s.str.starts_with("KXEARNINGSMENTION") | s.str.starts_with("KXMENTIONEARN")).then(pl.lit("earnings"))
            .otherwise(pl.lit("other")))


def decision_rows(lg: pl.DataFrame, candles: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    """One row per speech leg open at ``t = start − 6 h`` with a quote (≤ 2 h old), the
    outcome-blind early-start exclusion (said share at t vs at start − 24 h, round 2), ``end``,
    ``mid``, frozen fair values and segments."""
    from albert.research.fundamental import attach_quotes

    c = candles.select("ticker", "end_ts", "yes_bid_close", "yes_ask_close")
    rows = {}
    for lab, dt in (("h6", LEAD), ("h24", H24)):
        d = lg.with_columns((pl.col("start") - dt).alias("t"))
        d = d.filter((pl.col("open_time") <= pl.col("t")) & (pl.col("close_time") > pl.col("t")))
        rows[lab] = attach_quotes(d, c, max_age=MAX_AGE)
    s6 = rows["h6"].group_by("event").agg((pl.col("bid") >= SAID_BID).mean().alias("said"))
    s24 = rows["h24"].group_by("event").agg((pl.col("bid") >= SAID_BID).mean().alias("said24"))
    sh = s6.join(s24, on="event", how="left").with_columns(pl.col("said24").fill_null(0.0))
    bad = sh.filter(pl.col("said") > pl.col("said24") + SAID_JUMP)["event"]
    Q = rows["h6"].filter(~pl.col("event").is_in(bad.implode()))
    Q = Q.with_columns(pl.min_horizontal(pl.col("start") - END_MARGIN, pl.col("close_time")).alias("end"),
                       ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("ask") - pl.col("bid")).alias("spread"),
                       pl.lit("h6").alias("when"))
    Q = Q.filter(pl.col("end") > pl.col("t"))
    Q = Q.with_columns(pl.Series("fv_cal", fair_cal(Q["mid"].to_numpy())), pl.col("mid").alias("fv_mid"),
                       pl.col("start").dt.strftime("%Y-%m").alias("month"),
                       (pl.col("start").dt.year().cast(pl.String) + "-Q" + pl.col("start").dt.quarter().cast(pl.String)).alias("quarter"),
                       pl.when(pl.col("t") < datetime(2025, 7, 1, tzinfo=UTC)).then(pl.lit("a1 2025H1"))
                       .when(pl.col("t") < datetime(2026, 1, 1, tzinfo=UTC)).then(pl.lit("a2 2025H2"))
                       .when(pl.col("t") < datetime(2026, 7, 1, tzinfo=UTC)).then(pl.lit("a3 2026H1"))
                       .when(pl.col("t") < DEV_FROM).then(pl.lit("a4 2026Jul-Aug16"))
                       .when(pl.col("t") < HOLDOUT_FROM).then(pl.lit("b dev Aug17-Sep"))
                       .when(pl.col("t") < ROUND2_TO).then(pl.lit("c holdout Oct")).otherwise(pl.lit("d Nov")).alias("period"),
                       pl.when(pl.col("t") < DEV_FROM).then(pl.lit("1 history <Aug17-2026"))
                       .when(pl.col("t") < HOLDOUT_FROM).then(pl.lit("2 r2 dev Aug17-Sep"))
                       .when(pl.col("t") < ROUND2_TO).then(pl.lit("3 r2 holdout Oct")).otherwise(pl.lit("4 Nov 1-5")).alias("split"),
                       spread_band(pl.col("ask") - pl.col("bid")).alias("sband"), M.price_band(pl.col("mid")).alias("band"),
                       speaker_group(pl.col("series")).alias("sgroup"),
                       (pl.col("ms_updated") <= pl.col("t")).alias("ms_known"))
    info = {"legs_open_at_t": int(lg.filter((pl.col("open_time") <= pl.col("start") - LEAD)
                                            & (pl.col("close_time") > pl.col("start") - LEAD)).height),
            "rows_quoted_at_t": rows["h6"].height, "events_quoted": rows["h6"]["event"].n_unique(),
            "events_excluded_early_start": int(bad.len()), "rows": Q.height, "events": Q["event"].n_unique()}
    return Q, info


def order_levels(Q: pl.DataFrame, fv: str, margin: float) -> pl.DataFrame:
    """YES-bid and NO-bid prices of the rule (harness rounding and post-only caps)."""
    p, bid, ask = Q[fv].to_numpy(), Q["bid"].to_numpy(), Q["ask"].to_numpy()
    yb = np.minimum(np.floor((p - margin) * 100 + 1e-9) / 100, np.round(ask - 0.01, 2))
    nb = np.minimum(np.floor((1 - p - margin) * 100 + 1e-9) / 100, np.round(1 - bid - 0.01, 2))
    return Q.select("ticker").with_columns(pl.Series("yes_bid_px", yb), pl.Series("no_bid_px", nb))


def fill_candidates(Q: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    """Legs whose hourly trade range could contain a print through any of the rule's bids (both
    fair values, 8c margin = the most aggressive level) in an hour overlapping (t, end], or a
    ≥ 95c print on a cheap leg (circuit breaker), or an hour missing from the candles. Every other
    leg provably has no fill (the hourly high / low bound every print), so its prints are not
    fetched."""
    lv = [order_levels(Q, fv, min(MARGINS)) for fv in ("fv_cal", "fv_mid")]
    nb = np.minimum(lv[0]["no_bid_px"].to_numpy(), lv[1]["no_bid_px"].to_numpy())
    yb = np.maximum(lv[0]["yes_bid_px"].to_numpy(), lv[1]["yes_bid_px"].to_numpy())
    q = Q.select("ticker", "t", "end", "ask").with_columns(pl.Series("ask_lvl", 1 - nb), pl.Series("yes_lvl", yb))
    unknown = pl.col("price_high").is_null() & (pl.col("volume").is_null() | (pl.col("volume") > 0))
    h = candles.select("ticker", "end_ts", "price_low", "price_high", "volume").join(q, on="ticker")
    h = h.filter((pl.col("end_ts") > pl.col("t")) & (pl.col("end_ts") - timedelta(hours=1) < pl.col("end")))
    hit = h.group_by("ticker").agg(
        ((pl.col("price_high") > pl.col("ask_lvl") + 1e-9).any() | (pl.col("price_low") < pl.col("yes_lvl") - 1e-9).any()
         | ((pl.col("ask") <= CB_ASK) & (pl.col("price_high") >= CB_PRINT - 1e-9)).any() | unknown.any()).alias("hit"),
        pl.len().alias("hours"))
    # hourly candles ending at h overlap (t, end] iff t < h < end + 1 h
    hr = 3_600_000_000
    t_us, e_us = pl.col("t").dt.epoch("us"), pl.col("end").dt.epoch("us")
    need = q.join(hit, on="ticker", how="left").with_columns(
        ((e_us + hr - 1) // hr - t_us // hr).alias("hours_needed"))
    # an hour without a candle (or an unparsed range) is treated as a candidate
    return need.filter(pl.col("hit").fill_null(True) | (pl.col("hours").fill_null(0) < pl.col("hours_needed")))


# ---- fills, capacity ----------------------------------------------------------------------------
R2_NAME = "mentions_maker"      # round-2 study whose cached Kalshi prints are reused


def load_all_trades(s) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Prints from this study plus round 2's cached Kalshi prints (``fundamental_mentions_maker_
    trades*.parquet``), and the windows each covers (``ticker, t_from, t_to``); round-2 tickers
    whose fetch failed (``..._failed.parquet``) are not covered."""
    frames, wins = [], []
    p = cache_dir(s) / "trades.parquet"
    if p.exists():
        frames.append(pl.read_parquet(p).select(list(TRADE_SCHEMA)))
        wins.append(pl.read_parquet(cache_dir(s) / "trade_windows.parquet"))
    c = s.data_dir / "cache"
    failed = c / f"fundamental_{R2_NAME}_failed.parquet"
    bad = set(pl.read_parquet(failed)["ticker"]) if failed.exists() else set()
    for name in (R2_NAME, f"{R2_NAME}_retry"):
        tp, wp = c / f"fundamental_{name}_trades.parquet", c / f"fundamental_{name}_trades_windows.parquet"
        if tp.exists():
            frames.append(pl.read_parquet(tp).select(list(TRADE_SCHEMA)))
        if wp.exists():
            wins.append(pl.read_parquet(wp).filter(~pl.col("ticker").is_in(list(bad))))
    tr = pl.concat(frames, how="diagonal_relaxed").unique("trade_id") if frames else pl.DataFrame(schema=TRADE_SCHEMA)
    w = pl.concat(wins, how="diagonal_relaxed").unique() if wins else pl.DataFrame()
    return tr, w


def covered_rows(Q: pl.DataFrame, windows: pl.DataFrame) -> pl.Series:
    """Rows whose (t, end] lies inside a fetched trade window of the same ticker."""
    if not windows.height:
        return pl.Series([False] * Q.height)
    j = Q.select("ticker", "t", "end").with_row_index("r").join(windows, on="ticker", how="left")
    ok = j.with_columns(((pl.col("t_from") <= pl.col("t")) & (pl.col("t_to") >= pl.col("end"))).fill_null(False).alias("ok"))
    return ok.group_by("r").agg(pl.col("ok").any()).sort("r")["ok"]


def through_volume(P: pl.DataFrame, trades: pl.DataFrame, end_col: str) -> np.ndarray:
    """Contracts printed strictly through each FILLED order's price in (t, end] (``P`` has
    ticker, t, end, side, price, filled): the most our resting order could have sold at its
    price (0 for unfilled orders)."""
    tr = trades.select("ticker", pl.col("created_time").dt.epoch("us").alias("ts"), "yes_price", "taker_side", "count")
    by = {k[0]: (g["ts"].to_numpy(), g["yes_price"].to_numpy(), (g["taker_side"] == "yes").to_numpy(), g["count"].to_numpy())
          for k, g in tr.sort("ticker", "ts").group_by(["ticker"])}
    out = np.zeros(P.height)
    tick, side, price, filled = P["ticker"].to_list(), P["side"].to_list(), P["price"].to_numpy(), P["filled"].to_numpy()
    t0, t1 = P["t"].dt.epoch("us").to_numpy(), P[end_col].dt.epoch("us").to_numpy()
    for i in np.flatnonzero(filled):
        g = by.get(tick[i])
        if g is None:
            continue
        lo, hi = np.searchsorted(g[0], t0[i], "right"), np.searchsorted(g[0], t1[i], "right")
        yp, ys, n = g[1][lo:hi], g[2][lo:hi], g[3][lo:hi]
        m = (ys & (yp > 1 - price[i] + 1e-9)) if side[i] == "no" else (~ys & (yp < price[i] - 1e-9))
        out[i] = float(n[m].sum())
    return out


# Copied from round 2 (:mod:`albert.research.mentions_maker`, Kalshi-only code) so this study does
# not depend on that module, which also holds transcript code.
def kill_times(Q: pl.DataFrame, trades: pl.DataFrame, *, cb_print: float = CB_PRINT,
               cb_ask: float = CB_ASK) -> pl.DataFrame:
    """Circuit breaker per event: the first print after ``t`` at ≥ ``cb_print`` on any leg of the
    event whose ask at ``t`` was ≤ ``cb_ask`` (a word was just said). Public trade feed only."""
    legs = Q.filter(pl.col("ask") <= cb_ask).select("event", "ticker", "t")
    tr = trades.filter(pl.col("yes_price") >= cb_print).select("ticker", pl.col("created_time").alias("ts"))
    k = legs.join(tr, on="ticker").filter(pl.col("ts") > pl.col("t"))
    return k.group_by("event").agg(pl.col("ts").min().alias("kill_ts"))


def fast_fills(pred: pl.DataFrame, trades: pl.DataFrame, margin: float, p_col: str = "p",
               end_col: str = "end") -> pl.DataFrame:
    """The harness rule (:func:`albert.research.fundamental.maker_fills`), vectorised per ticker:
    one YES bid ``floor(p − margin)`` (≤ ask − 1c) and one NO bid ``floor(1 − p − margin)``
    (≤ 1 − bid − 1c) per row, resting in (t, end]; a bid fills only on a later print strictly
    through it; held to settlement with the series maker fee. ``row`` = row index in ``pred``."""
    from albert.research.fundamental import maker_fee

    tr = trades.select("ticker", pl.col("created_time").alias("ts"), "yes_price", "taker_side").sort("ticker", "ts")
    by: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for (tk,), g in tr.group_by(["ticker"]):
        by[tk] = (g["ts"].dt.epoch("us").to_numpy(), g["yes_price"].to_numpy().astype(float),
                  (g["taker_side"] == "yes").to_numpy())
    P = pred.with_row_index("row")
    tick = P["ticker"].to_list()
    t0, t1 = P["t"].dt.epoch("us").to_numpy(), P[end_col].dt.epoch("us").to_numpy()
    p = P[p_col].to_numpy().astype(float)
    bid, ask, y = P["bid"].to_numpy(), P["ask"].to_numpy(), P["y"].to_numpy().astype(float)
    ft = P["fee_type"].to_list() if "fee_type" in P.columns else [None] * P.height
    fm = P["fee_multiplier"].to_list() if "fee_multiplier" in P.columns else [1.0] * P.height
    rows = []
    for i in range(P.height):
        yb = min(math.floor((p[i] - margin) * 100 + 1e-9) / 100, round(ask[i] - 0.01, 2))
        nb = min(math.floor((1 - p[i] - margin) * 100 + 1e-9) / 100, round(1 - bid[i] - 0.01, 2))
        g = by.get(tick[i])
        if g is not None:
            lo, hi = np.searchsorted(g[0], t0[i], "right"), np.searchsorted(g[0], t1[i], "right")
            yp, ys, ts = g[1][lo:hi], g[2][lo:hi], g[0][lo:hi]
        else:
            yp = ys = ts = np.empty(0)
        mult = fm[i] or 1.0
        for side, price in (("yes", yb), ("no", nb)):
            if price < 0.01:
                continue
            hit = None
            if len(ts):
                m = (~ys & (yp < price - 1e-9)) if side == "yes" else (ys & (yp > 1 - price + 1e-9))
                if m.any():
                    hit = int(ts[np.argmax(m)])
            won = y[i] if side == "yes" else 1 - y[i]
            rows.append({"row": i, "side": side, "price": price, "filled": hit is not None, "fill_us": hit,
                         "pnl": (won - price - maker_fee(price, ft[i], mult)) if hit is not None else 0.0})
    return pl.DataFrame(rows, schema={"row": pl.UInt32, "side": pl.String, "price": pl.Float64, "filled": pl.Boolean,
                                      "fill_us": pl.Int64, "pnl": pl.Float64})


def summarize(f: pl.DataFrame) -> dict[str, Any]:
    """Orders, fills, fill rate, PnL per fill and per order (cents), t by event."""
    from albert.research.fundamental import clustered_mean

    filled = f.filter("filled")
    row: dict[str, Any] = {"orders": f.height, "fills": filled.height,
                           "fill_rate": round(filled.height / f.height, 3) if f.height else None}
    if filled.height:
        mu, tt, k = clustered_mean(filled["pnl"].to_numpy(), filled["event"].to_numpy())
        row.update({"events": k, "pnl_per_fill_c": round(mu * 100, 3), "t": None if tt is None else round(tt, 2),
                    "pnl_per_order_c": round(float(f["pnl"].sum()) / f.height * 100, 3),
                    "pnl_total_$": round(float(filled["pnl"].sum()), 2),
                    "yes_share": round(float((filled["side"] == "yes").mean()), 3)})
    return row


GRID = tuple((fv, endc, m) for fv in ("fv_cal", "fv_mid") for endc in ("end_cb", "end") for m in MARGINS)


ATTRS = ["ticker", "event", "series", "speaker", "sgroup", "t", "start", "month", "quarter", "period", "split", "sband", "band",
         "mid", "bid", "ask", "spread", "end", "end_cb", "y", "ms_known"]


def fills_grid(Q: pl.DataFrame, trades: pl.DataFrame | None, candles: pl.DataFrame | None = None, *,
               inner: bool = False) -> pl.DataFrame:
    """Orders of every (fair value, end, margin) configuration with fills, PnL and through-volume:
    from prints (:func:`fast_fills`, trade-through) when ``trades`` is given, else from the hourly
    candle proxy (:func:`proxy_fills`; ``Q.end_cb`` must then be the proxy kill time)."""
    base = Q.with_row_index("row").select("row", *ATTRS)
    out = []
    for fv, endc, m in GRID:
        if trades is not None:
            f = fast_fills(Q, trades, m, p_col=fv, end_col=endc).join(base, on="row")
            f = f.with_columns(pl.Series("thru", through_volume(f, trades, endc)), pl.lit("prints").alias("source"))
        else:
            f = proxy_fills(Q, candles, m, p_col=fv, end_col=endc, inner=inner).join(base, on="row")
            f = f.with_columns(pl.lit(None, dtype=pl.Float64).alias("thru"),
                               pl.lit("proxy_inner" if inner else "proxy").alias("source"))
        out.append(f.with_columns(pl.lit(fv).alias("fair"), pl.lit(endc).alias("endc"), pl.lit(m).alias("margin")))
    return pl.concat(out)


def proxy_kill(Q: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    """Circuit breaker from hourly candles: the START of the first hour after ``t`` whose trade
    high is ≥ 95c on a cheap leg (ask ≤ 80c at ``t``); the whole hour is cancelled (conservative)."""
    legs = Q.filter(pl.col("ask") <= CB_ASK).select("event", "ticker", "t")
    h = candles.filter(pl.col("price_high") >= CB_PRINT - 1e-9).select("ticker", "end_ts").join(legs, on="ticker")
    h = h.filter(pl.col("end_ts") > pl.col("t"))
    return h.group_by("event").agg((pl.col("end_ts").min() - timedelta(hours=1)).alias("kill_hr"))


def proxy_fills(Q: pl.DataFrame, candles: pl.DataFrame, margin: float, p_col: str, end_col: str,
                *, inner: bool = False) -> pl.DataFrame:
    """Fill proxy from hourly candles (same order prices as :func:`fast_fills`): the NO bid fills if
    an hour overlapping (t, end] traded above our ask (1 − NO bid), the YES bid if one traded below
    it. Taker side is unknown and partial first / last hours count whole, so the proxy can only
    over-fill relative to prints; validated against prints where both exist. ``inner=True`` counts
    only hours lying wholly inside (t, end] (a lower bound: it drops the partial hours)."""
    from albert.research.fundamental import maker_fee

    lv = order_levels(Q, p_col, margin)
    P = Q.with_row_index("row").select("row", "ticker", "t", pl.col(end_col).alias("_end"), "y", "fee_type", "fee_multiplier")
    P = P.with_columns(lv["yes_bid_px"], lv["no_bid_px"])
    h = candles.select("ticker", "end_ts", "price_low", "price_high").join(P.select("row", "ticker", "t", "_end"), on="ticker")
    if inner:
        h = h.filter((pl.col("end_ts") - timedelta(hours=1) >= pl.col("t")) & (pl.col("end_ts") <= pl.col("_end")))
    else:
        h = h.filter((pl.col("end_ts") > pl.col("t")) & (pl.col("end_ts") - timedelta(hours=1) < pl.col("_end")))
    hl = h.group_by("row").agg(pl.col("price_high").max().alias("hi"), pl.col("price_low").min().alias("lo"))
    P = P.join(hl, on="row", how="left")
    rows = []
    for r in P.iter_rows(named=True):
        mult = r["fee_multiplier"] or 1.0
        for side, price in (("yes", r["yes_bid_px"]), ("no", r["no_bid_px"])):
            if price < 0.01:
                continue
            if side == "yes":
                hit = r["lo"] is not None and r["lo"] < price - 1e-9
            else:
                hit = r["hi"] is not None and r["hi"] > 1 - price + 1e-9
            won = r["y"] if side == "yes" else 1 - r["y"]
            rows.append({"row": r["row"], "side": side, "price": price, "filled": bool(hit), "fill_us": None,
                         "pnl": (won - price - maker_fee(price, r["fee_type"], mult)) if hit else 0.0})
    return pl.DataFrame(rows, schema={"row": pl.UInt32, "side": pl.String, "price": pl.Float64, "filled": pl.Boolean,
                                      "fill_us": pl.Int64, "pnl": pl.Float64})


# ---- statistics -----------------------------------------------------------------------------------
def summ(g: pl.DataFrame) -> dict[str, Any]:
    """Round-2 summary (orders, fills, PnL per fill / order in cents, t by event) plus events and
    capacity (contracts printed through our price per filled order, median; per event, median)."""
    row = summarize(g)
    row["events_filled"] = row.pop("events", 0)
    row["events"] = g["event"].n_unique()
    f = g.filter(pl.col("filled") & pl.col("thru").is_not_null())      # capacity from prints only
    if f.height:
        row["thru_per_fill_med"] = float(f["thru"].median())
        row["thru_per_event_med"] = float(f.group_by("event").agg(pl.col("thru").sum())["thru"].median())
    return row


def table(F: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    rows = [{**dict(zip(keys, k)), **summ(g)} for k, g in F.group_by(keys)]
    cols = [*keys, "events", "orders", "fills", "fill_rate", "events_filled", "pnl_per_fill_c", "t", "pnl_per_order_c",
            "pnl_total_$", "yes_share", "thru_per_fill_med", "thru_per_event_med"]
    T = pl.DataFrame(rows, infer_schema_length=None)
    return T.select([c for c in cols if c in T.columns]).sort(keys)


def clustered_slope(x: np.ndarray, y: np.ndarray, g: np.ndarray) -> tuple[float, float, float, float]:
    """OLS ``y = a + b·x`` with standard errors clustered by ``g``: (a, b, t_a, t_b)."""
    X = np.column_stack([np.ones_like(x), x])
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    u = y - X @ beta
    _, inv = np.unique(g, return_inverse=True)
    G = inv.max() + 1
    S = np.zeros((G, 2))
    np.add.at(S, inv, X * u[:, None])
    V = bread @ (S.T @ S) @ bread * G / max(G - 1, 1)
    se = np.sqrt(np.diag(V))
    return float(beta[0]), float(beta[1]), float(beta[0] / se[0]), float(beta[1] / se[1])


def trend(F: pl.DataFrame) -> dict[str, Any]:
    """PnL per fill on calendar time (years since the first fill), clustered by event; and the
    same on the event-level PnL per order."""
    f = F.filter("filled")
    if f.height < 10:
        return {}
    t0 = f["t"].min()
    x = ((f["t"] - t0).dt.total_seconds() / (365.25 * 86400)).to_numpy()
    a, b, ta, tb = clustered_slope(x, f["pnl"].to_numpy() * 100, f["event"].to_numpy())
    return {"fills": f.height, "first": str(t0)[:10], "intercept_c": round(a, 2), "slope_c_per_year": round(b, 2),
            "t_intercept": round(ta, 2), "t_slope": round(tb, 2)}


def months_positive(T: pl.DataFrame, min_fills: int = 10, min_events: int = 3) -> dict[str, Any]:
    """Months with ≥ ``min_fills`` fills in ≥ ``min_events`` events: how many had positive PnL."""
    m = T.filter((pl.col("fills") >= min_fills) & (pl.col("events_filled") >= min_events))
    return {"months": m.height, "positive": int((m["pnl_per_fill_c"] > 0).sum()),
            "share": round(float((m["pnl_per_fill_c"] > 0).mean()), 3) if m.height else None,
            "months_t_above_2": int((m["t"].fill_null(0) > 2).sum()), "months_t_below_-2": int((m["t"].fill_null(0) < -2).sum()),
            "min_fills": min_fills, "min_events": min_events}


def taker_reference(Q: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Buy NO at ``1 − bid`` at ``t`` on every quoted leg (taker fee by series), held to settlement."""
    from albert.research.fundamental import clustered_mean, taker_fee

    px = 1 - Q["bid"].to_numpy()
    mult = Q["fee_multiplier"].fill_null(1.0).to_numpy()
    fee = np.array([taker_fee(p, m) if p < 1 else 0.0 for p, m in zip(px, mult)])
    d = Q.with_columns(pl.Series("pnl_no", (1 - Q["y"].to_numpy()) - px - fee)).filter(pl.col("bid") > 0)
    rows = []
    for k, g in d.group_by(keys):
        mu, tt, n = clustered_mean(g["pnl_no"].to_numpy(), g["event"].to_numpy())
        rows.append({**dict(zip(keys, k)), "legs": g.height, "events": n, "pnl_c": round(mu * 100, 2),
                     "t": None if tt is None else round(tt, 2)})
    return pl.DataFrame(rows, infer_schema_length=None).sort(keys)


# ---- the study --------------------------------------------------------------------------------------
PRIORITY = {"trump": 0, "vance": 1, "other": 2, "earnings": 3}
SAMPLE_WEIGHT = {"trump": 1.0, "vance": 1.0, "other": 0.06, "earnings": 0.06}


def job_priority(job: dict[str, Any], grp: dict[str, str]) -> tuple:
    """Fetch order: cheap batch jobs first; then events by ``u / w`` with ``u`` a uniform hash of
    the event ticker (volume-blind) and ``w`` the group weight (Trump, Vance 1; other speakers and
    earnings calls 0.06), so when Trump and Vance are complete a random ~6% of the other events is
    too, and any partial fetch is a random sample of events across all months."""
    import hashlib

    u = int(hashlib.sha256(job["event"].encode()).hexdigest()[:12], 16) / 16 ** 12
    return (0 if job["batch"] else 1, u / SAMPLE_WEIGHT[grp[job["event"]]], job["job"])


def round2_done(s, jobs: list[dict[str, Any]]) -> set[str]:
    """Jobs whose every leg has round-2 candles (events Aug 15 – Nov 1 2026): not refetched."""
    have = set(round2_candles(s)["ticker"].unique())
    return {j["job"] for j in jobs if set(j["tickers"]) <= have}


def build(s) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    lg = speech_legs(s)
    candles = all_candles(s)
    jobs = candle_jobs(lg)
    done = set(pl.read_parquet(cache_dir(s) / "candle_jobs.parquet")["job"]) | round2_done(s, jobs)
    ev_ok = {j["event"] for j in jobs if j["job"] in done} - {j["event"] for j in jobs if j["job"] not in done}
    Q, info = decision_rows(lg.filter(pl.col("event").is_in(list(ev_ok))), candles)
    info = {"speech_legs": lg.height, "speech_events": lg["event"].n_unique(), "candle_events_done": len(ev_ok),
            "candle_events_total": len({j["event"] for j in jobs}), **info}
    ev = (lg.filter(pl.col("event").is_in(list({j["event"] for j in jobs})))
          .group_by("event").agg(pl.col("series").first(), pl.col("start").first())
          .with_columns(speaker_group(pl.col("series")).alias("sgroup"), pl.col("start").dt.strftime("%Y-%m").alias("month"),
                        pl.col("event").is_in(list(ev_ok)).alias("fetched"), pl.col("event").is_in(Q["event"].unique().implode()).alias("quoted")))
    info["_coverage"] = ev
    return Q, candles, info


def flag_sources(Q: pl.DataFrame, candles: pl.DataFrame, windows: pl.DataFrame) -> pl.DataFrame:
    """Flag fill candidates (:func:`fill_candidates`) and trade coverage; ``src`` = ``prints`` for
    events whose every candidate leg has fetched prints, else ``proxy`` (hourly candle ranges)."""
    cand = fill_candidates(Q, candles).select("ticker").with_columns(pl.lit(True).alias("cand"))
    Q = Q.join(cand, on="ticker", how="left").with_columns(pl.col("cand").fill_null(False))
    Q = Q.with_columns(covered_rows(Q, windows).alias("covered"))
    bad = Q.filter(pl.col("cand") & ~pl.col("covered"))["event"].unique()
    # the proxy needs trade ranges: an event with an uncovered candidate leg whose candles lack
    # the range (round-2 candles) is not evaluable
    nr = candles.filter(pl.col("price_high").is_null() & (pl.col("volume").is_null() | (pl.col("volume") > 0)))["ticker"].unique()
    none = Q.filter(pl.col("cand") & ~pl.col("covered") & pl.col("ticker").is_in(nr.implode()))["event"].unique()
    return Q.with_columns(pl.when(pl.col("event").is_in(none.implode())).then(pl.lit("none"))
                          .when(pl.col("event").is_in(bad.implode())).then(pl.lit("proxy")).otherwise(pl.lit("prints")).alias("src"))


def with_config(F: pl.DataFrame) -> pl.DataFrame:
    return F.with_columns(pl.concat_str([pl.col("fair"), pl.col("endc"), (pl.col("margin") * 100).round(0).cast(pl.Int64)
                                         .cast(pl.String) + pl.lit("c")], separator="/").alias("config"))


def proxy_check(Qp: pl.DataFrame, trades: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    """On events with prints whose candles all carry trade ranges (this study's fetch): candle-proxy
    fills (all overlapping hours, and inner hours only) vs print fills, per configuration (NO side)."""
    from albert.research.fundamental import clustered_mean

    nr = candles.filter(pl.col("price_high").is_null() & (pl.col("volume").is_null() | (pl.col("volume") > 0)))["ticker"].unique()
    bad = Qp.filter(pl.col("ticker").is_in(nr.implode()) | ~pl.col("ticker").is_in(candles["ticker"].unique().implode()))["event"].unique()
    Qp = Qp.filter(~pl.col("event").is_in(bad.implode()))
    if not Qp.height:
        return pl.DataFrame()
    kp = proxy_kill(Qp, candles)
    Qx = Qp.drop("end_cb").join(kp, on="event", how="left").with_columns(
        pl.min_horizontal("end", pl.col("kill_hr").fill_null(pl.col("end"))).alias("end_cb"))
    A = with_config(fills_grid(Qp, trades)).filter(pl.col("side") == "no")
    rows = []
    for inner in (False, True):
        B = with_config(fills_grid(Qx, None, candles, inner=inner)).filter(pl.col("side") == "no")
        j = A.select("config", "ticker", "event", "filled", "pnl").join(
            B.select("config", "ticker", pl.col("filled").alias("filled_x"), pl.col("pnl").alias("pnl_x")), on=["config", "ticker"])
        for (cfg,), g in j.group_by(["config"]):
            f, fx = g.filter("filled"), g.filter("filled_x")
            mu, tt, _ = clustered_mean(fx["pnl_x"].to_numpy(), fx["event"].to_numpy()) if fx.height else (float("nan"), None, 0)
            rows.append({"proxy": "inner hours" if inner else "all overlapping hours", "config": cfg,
                         "events": Qp["event"].n_unique(), "orders": g.height, "fills_prints": f.height,
                         "fills_proxy": fx.height, "both": int((g["filled"] & g["filled_x"]).sum()),
                         "agree": round(float((g["filled"] == g["filled_x"]).mean()), 4),
                         "pnl_per_fill_prints_c": round(float(f["pnl"].mean()) * 100, 2) if f.height else None,
                         "pnl_per_fill_proxy_c": round(mu * 100, 2) if fx.height else None,
                         "t_proxy": None if tt is None else round(tt, 2),
                         "pnl_$_prints": round(float(f["pnl"].sum()), 2), "pnl_$_proxy": round(float(fx["pnl_x"].sum()), 2)})
    return pl.DataFrame(rows).sort("proxy", "config")


def run(s) -> dict[str, Any]:
    from albert.research.fundamental import clustered_mean

    Q, candles, info = build(s)
    cov = info.pop("_coverage")
    trades, windows = load_all_trades(s)
    Q = flag_sources(Q, candles, windows)
    Qp, Qx = Q.filter(pl.col("src") == "prints"), Q.filter(pl.col("src") == "proxy")
    info.update({"candidate_legs": int(Q["cand"].sum()), "events_prints": Qp["event"].n_unique(),
                 "events_proxy": Qx["event"].n_unique(), "events_unevaluable": Q.filter(pl.col("src") == "none")["event"].n_unique(),
                 "prints": trades.height})
    kt = kill_times(Qp, trades)
    Qp = Qp.join(kt, on="event", how="left").with_columns(
        pl.min_horizontal("end", pl.col("kill_ts").fill_null(pl.col("end"))).alias("end_cb"))
    kp = proxy_kill(Qx, candles)
    Qx = Qx.join(kp, on="event", how="left").with_columns(
        pl.min_horizontal("end", pl.col("kill_hr").fill_null(pl.col("end"))).alias("end_cb"))
    info["circuit_breaker_tripped_events"] = {
        "prints": int(Qp.unique("event").filter(pl.col("kill_ts") < pl.col("end")).height),
        "proxy": int(Qx.unique("event").filter(pl.col("kill_hr") < pl.col("end")).height)}
    parts = [fills_grid(Qp, trades)] if Qp.height else []
    if Qx.height:
        parts.append(fills_grid(Qx, None, candles))
    F = with_config(pl.concat(parts, how="diagonal_relaxed"))
    no = F.filter(pl.col("side") == "no")
    prim = no.filter((pl.col("fair") == "fv_cal") & (pl.col("endc") == "end_cb"))
    tables: dict[str, pl.DataFrame] = {}
    evald = prim.filter(pl.col("margin") == 0.08).group_by("event").agg(pl.col("source").first())
    cov = cov.join(evald, on="event", how="left")
    tables["0. Coverage: events with legs open at start − 6 h, fetched, quoted and evaluated (by group and quarter)"] = (
        cov.with_columns((pl.col("start").dt.year().cast(pl.String) + "-Q" + pl.col("start").dt.quarter().cast(pl.String)).alias("quarter"))
        .group_by("sgroup", "quarter")
        .agg(pl.len().alias("events"), pl.col("fetched").sum(), pl.col("quoted").sum(),
             (pl.col("source") == "prints").sum().alias("eval_prints"), (pl.col("source") == "proxy").sum().alias("eval_proxy"))
        .sort("sgroup", "quarter"))
    tables["A0. Out-of-sample summary: NO only, frozen rule and raw-mid control, by split and group"] = pl.concat([
        table(no.filter(pl.col("endc") == "end_cb"), ["config", "split"]).with_columns(pl.lit("all").alias("sgroup")),
        table(no.filter(pl.col("endc") == "end_cb"), ["config", "split", "sgroup"])], how="diagonal_relaxed").sort("config", "sgroup", "split")
    tables["A. Headline by period: NO bid only (rows) and both sides, every configuration"] = pl.concat([
        table(no, ["config", "period"]).with_columns(pl.lit("NO only").alias("sides")),
        table(F, ["config", "period"]).with_columns(pl.lit("both").alias("sides"))], how="diagonal_relaxed").sort("sides", "config", "period")
    tables["B. All periods pooled: NO only"] = table(no, ["config"])
    tables["B2. Frozen rule (fv_cal, breaker, NO only) by period and fill source"] = table(prim, ["margin", "period", "source"])
    if Qp.height:
        tables["P. Fill-source check on events with prints: hourly-candle proxy vs prints (NO side)"] = proxy_check(Qp, trades, candles)
    mon = {}
    for m in MARGINS:
        T = table(prim.filter(pl.col("margin") == m), ["month"])
        mon[m] = T
        tables[f"C{int(m * 100)}. Monthly, frozen rule (fv_cal, circuit breaker, NO only, {int(m * 100)}c)"] = T
    tables["D0. Quarterly by speaker group, both margins (frozen rule)"] = pl.concat([
        table(prim, ["margin", "sgroup", "quarter"]),
        table(prim, ["margin", "quarter"]).with_columns(pl.lit("all").alias("sgroup"))], how="diagonal_relaxed").sort("margin", "sgroup", "quarter")
    tables["D. Monthly by speaker group (frozen rule, 8c)"] = table(prim.filter(pl.col("margin") == 0.08), ["sgroup", "month"])
    tables["E. Speaker group x period (frozen rule, 8c and 12c)"] = table(prim, ["margin", "sgroup", "period"])
    ser = table(prim.filter(pl.col("margin") == 0.08), ["series"]).filter(pl.col("fills") >= 30).sort("fills", descending=True)
    tables["F. Per series with >= 30 fills, all periods (frozen rule, 8c)"] = ser
    tables["G. Spread at t x split (frozen rule, 8c and 12c)"] = table(prim, ["margin", "sband", "split"])
    tables["H. Price band at t x split (frozen rule, 8c)"] = table(prim.filter(pl.col("margin") == 0.08), ["band", "split"])
    tables["I. Control: raw mid, circuit breaker, NO only, by month (8c)"] = table(
        no.filter((pl.col("fair") == "fv_mid") & (pl.col("endc") == "end_cb") & (pl.col("margin") == 0.08)), ["month"])
    tr_rows, mp_rows = [], []
    for m in MARGINS:
        for grp in ("all", "trump", "vance", "other", "earnings"):
            g = prim.filter(pl.col("margin") == m)
            if grp != "all":
                g = g.filter(pl.col("sgroup") == grp)
            tr_rows.append({"margin": m, "group": grp, **trend(g)})
            mp_rows.append({"margin": m, "group": grp, **(months_positive(table(g, ["month"])) if g.height else {})})
    tables["J. Trend: PnL per fill (c) on years since first fill, clustered by event (frozen rule)"] = pl.DataFrame(tr_rows, infer_schema_length=None)
    tables["K. Share of months positive (months with >= 10 fills in >= 3 events)"] = pl.DataFrame(mp_rows, infer_schema_length=None)
    rob = []
    for (m, per), g in prim.filter("filled").group_by(["margin", "split"]):
        ev = g.group_by("event").agg(pl.col("pnl").sum().alias("p")).sort("p", descending=True)
        x = g.filter(~pl.col("event").is_in(ev.head(3)["event"].implode()))
        mu, tt, _ = clustered_mean(x["pnl"].to_numpy(), x["event"].to_numpy())
        rob.append({"margin": m, "split": per, "events_filled": ev.height, "events_positive": int((ev["p"] > 0).sum()),
                    "pnl_$": round(float(g["pnl"].sum()), 2), "top3_$": round(float(ev.head(3)["p"].sum()), 2),
                    "pnl_per_fill_ex_top3_c": round(mu * 100, 2), "t_ex_top3": None if tt is None else round(tt, 2)})
    tables["L. Robustness: events positive and without the 3 best events (frozen rule)"] = pl.DataFrame(rob).sort("margin", "split")
    big = prim.with_columns((pl.col("filled") & (pl.col("thru") >= 1)).alias("filled"))
    big = big.with_columns(pl.when(pl.col("filled")).then(pl.col("pnl")).otherwise(0.0).alias("pnl"))
    tables["R. Robustness: milestone start last edited before t (known) or after; fills needing >= 1 contract through"] = pl.concat([
        table(prim, ["margin", "split", "ms_known"]).with_columns(pl.lit("all fills").alias("variant")),
        table(big, ["margin", "split"]).with_columns(pl.lit(">= 1 contract through").alias("variant"))],
        how="diagonal_relaxed").sort("variant", "margin", "split")
    cap = prim.filter(pl.col("filled") & (pl.col("source") == "prints"))
    tables["M. Capacity: contracts printed through our NO price while resting (frozen rule)"] = pl.DataFrame([
        {"margin": m, "split": per, "fills": g.height, "thru_p25": float(g["thru"].quantile(0.25)),
         "thru_median": float(g["thru"].median()), "thru_p75": float(g["thru"].quantile(0.75)),
         "per_event_median": float(g.group_by("event").agg(pl.col("thru").sum())["thru"].median()),
         "per_event_filled_legs_median": float(g.group_by("event").len()["len"].median())}
        for (m, per), g in cap.group_by(["margin", "split"])]).sort("margin", "split")
    tables["N. Taker reference: buy NO at 1 - bid at t (fee), by split and group"] = pl.concat([
        taker_reference(Q, ["split"]).with_columns(pl.lit("all").alias("sgroup")), taker_reference(Q, ["split", "sgroup"])],
        how="diagonal_relaxed").sort("sgroup", "split")
    # round-2 reproduction: Trump/Vance, calibrated mid, no breaker, 12c, both sides, October
    tv = F.filter(pl.col("sgroup").is_in(["trump", "vance"]) & (pl.col("fair") == "fv_cal") & (pl.col("endc") == "end")
                  & (pl.col("margin") == 0.12))
    tables["O. Round-2 cross-check: Trump/Vance, fv_cal, no breaker, 12c, both sides (round 2 h6 Oct: +7.3c)"] = table(tv, ["period"])
    # rows and frozen-rule orders for the coordinator
    c = s.data_dir / "cache"
    pl.concat([Qp, Qx], how="diagonal_relaxed").select(
        "ticker", "event", "series", "sgroup", "t", "start", "end", "end_cb", "bid", "ask", "mid", "fv_cal", "y", "split",
        "month", "src").write_parquet(c / f"fundamental_{NAME}_rows.parquet")
    prim.select("ticker", "event", "series", "sgroup", "t", "month", "split", "margin", "side", "price", "filled", "pnl",
                "thru", "source").write_parquet(c / f"fundamental_{NAME}_orders.parquet")
    info["saved"] = [f"data/cache/fundamental_{NAME}_rows.parquet", f"data/cache/fundamental_{NAME}_orders.parquet"]
    info["months_positive_8c"] = months_positive(mon[0.08])
    info["months_positive_12c"] = months_positive(mon[0.12])
    return {"Q": Q, "F": F, "info": info, "tables": tables}


def write_report(s, out: dict[str, Any], verdict: str = "") -> Path:
    from albert.reports import Report

    rep = Report(s.reports_dir, f"research-fundamental-{NAME}")
    if verdict:
        rep.text("Verdict", verdict)
    rep.text("Design", __doc__)
    rep.json("Run info", out["info"])
    for i, (title, tab) in enumerate(out["tables"].items()):
        if tab.height:
            rep.table(title, tab, max_rows=120, save_as=f"t{i:02d}")
    return rep.save()


# ---- CLI ---------------------------------------------------------------------------------------------
def main(argv: list[str]) -> None:
    from albert.config import Settings

    s = Settings()
    step = argv[1] if len(argv) > 1 else "run"
    if step == "markets":
        mk = fetch_markets(s, mention_series(s))
        print("legs", mk.height, "events", mk["event_ticker"].n_unique())
        st = fetch_milestones(s, mk["event_ticker"].unique().to_list())
        print("starts", st.height)
    elif step == "candles":
        lg = speech_legs(s).with_columns(speaker_group(pl.col("series")).alias("sgroup"))
        grp = dict(lg.select("event", "sgroup").unique().iter_rows())
        # events with round-2 candles get the cheap batch request only (trade ranges for the proxy
        # check), never the per-leg fallback
        skip = round2_done(s, candle_jobs(lg))
        jobs = [{**j, "fallback": j["job"] not in skip} for j in candle_jobs(lg)]
        jobs = sorted([j for j in jobs if j["batch"] or j["fallback"]], key=lambda j: job_priority(j, grp))
        print("candle jobs covered by round 2", len(skip), flush=True)
        print("candle jobs", len(jobs), flush=True)
        fetch_candles(s, jobs)
    elif step == "trades":
        # prints for fill-candidate legs not yet covered; ``trades N`` = only N events, a hash-random
        # sample of the fetched Trump / Vance events before Aug 15 2026 (a check of the candle proxy
        # in older markets)
        Q, candles, info = build(s)
        info.pop("_coverage")
        _, windows = load_all_trades(s)
        cand = fill_candidates(Q, candles).select("ticker").with_columns(pl.lit(True).alias("cand"))
        Q = Q.join(cand, on="ticker", how="left").filter(pl.col("cand").fill_null(False))
        Q = Q.filter(~covered_rows(Q, windows))
        if len(argv) > 2:
            import hashlib

            old = Q.filter(pl.col("sgroup").is_in(["trump", "vance"]) & (pl.col("start") < BATCH_FROM))
            evs = sorted(old["event"].unique().to_list(),
                         key=lambda e: hashlib.sha256(f"prints:{e}".encode()).hexdigest())[:int(argv[2])]
            Q = Q.filter(pl.col("event").is_in(evs))
        Q = Q.with_columns(pl.col("sgroup").replace_strict(PRIORITY, return_dtype=pl.Int64).alias("pri")).sort("pri", "t")
        w = Q.select("ticker", pl.col("t").alias("t_from"), pl.col("end").alias("t_to"))
        print("trade windows to fetch", w.height, "events", Q["event"].n_unique(), info, flush=True)
        fetch_trades(s, w, save_every=50)
    else:
        out = run(s)
        print(out["info"])
        for title, tab in out["tables"].items():
            print(title)
            print(tab)
        print(write_report(s, out, argv[2] if len(argv) > 2 else ""))


if __name__ == "__main__":
    main(sys.argv)
