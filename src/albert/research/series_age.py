"""Series age and calibration: are NEW Kalshi series (or young ladders) miscalibrated, with
YES / longshots overpriced, and does the bias fade as the series matures? If so, is there a
rule (sell longshots in young series) that survives fees out of sample?

Steps (all public data; nothing here places orders):

1. **Launch date per series** = first ``open_time`` of any market the series ever listed. The
   per-series endpoint ``GET /series/{ticker}`` carries no creation date (only
   ``last_updated_ts``), and ``/historical/markets`` ignores close-time filters and lists newest
   first, so we page ``GET /events?series_ticker=`` (small objects, every event the series ever
   had, old ones included), take the events with the earliest ticker-encoded dates (plus the last
   ones listed), and read their markets' ``open_time`` from ``/historical/markets?event_ticker=``
   (or the live ``/markets`` after the cutoff). The minimum with the ingested table is the launch.
   Responses are cached under ``data/cache/series_age/`` (append-only, keyed by path + params).
2. **Volume-blind sample:** markets of ``kalshi_markets`` (every settled non-financial market
   closing Aug 15–Oct 31), decisions at fixed points relative to the *scheduled* expiry
   (``expected_expiration_time``, which does not move when a market closes early, so decision
   times never peek at when the outcome happened): 24 h and 6 h before it and at mid-life;
   a decision exists only while the market is open. Events are sampled with a fixed seed per
   (series × age bucket) cell, all legs of each sampled event kept (ladder sums need them).
   Quote = last hourly candle closed by ``t`` (``fundamental.attach_quotes``).
3. Calibration of the mid by age bucket × price band and logistic ``logit y ~ a + b·logit(mid)``
   per bucket (t by event, category fixed effects), ladder sums by age; then a dev-chosen rule
   (taker at the bid with series fees; maker with trade-through fills) tested on October.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

NAME = "series_age"
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
DATA_START = datetime(2026, 8, 15, tzinfo=UTC)
OLD_BEFORE = datetime(2026, 3, 1, tzinfo=UTC)   # launched before this: "older" at every decision
SNAPSHOT = "2026-11-05"                          # cache tag for listings (they grow over time)
AGE_EDGES_WEEKS = (2.0, 6.0, 16.0)
AGE_LABELS = ("a0_lt2w", "a1_2_6w", "a2_6_16w", "a3_old")
SEED = 20261105

_DATE_RX = re.compile(r"-(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})")
_MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


def cache_dir(s) -> Path:
    return s.data_dir / "cache" / NAME


def _client(s, min_interval: float = 0.15):
    """Client caching raw responses under ``data/cache/series_age/kalshi``. Retry waits are
    capped at 1 s: the API's limit refills per second, and the client's doubling backoff
    otherwise idles for up to 15 s on a shared, busy limit."""
    import time

    from albert.http import CachedClient
    return CachedClient("kalshi", s.kalshi_base, cache_dir(s), min_interval=min_interval,
                        sleep=lambda x: time.sleep(min(x, 1.0)))


def _patient_get(api, path: str, params: dict[str, Any], *, tries: int = 7, **kw):
    """``api.get`` that waits out the shared rate limit (429 after the client's own retries)."""
    import time

    from albert.http import HttpError
    for attempt in range(tries):
        try:
            return api.get(path, params, **kw)
        except HttpError as error:
            if "429" not in str(error) or attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def _atomic_write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


# ---- step 1: launch dates ---------------------------------------------------------------------
def ticker_date(event_ticker: str | None) -> datetime | None:
    """Date encoded after the first dash (``KXRAIN-26AUG16NYC`` -> 2026-08-16), else None."""
    m = _DATE_RX.search(event_ticker or "")
    if not m:
        return None
    yy, mon, dd = m.groups()
    try:
        return datetime(2000 + int(yy), _MONTHS[mon], int(dd), tzinfo=UTC)
    except ValueError:
        return None


EVENT_LIST_SCHEMA = {"series_ticker": pl.String, "event_ticker": pl.String, "mutually_exclusive": pl.Boolean,
                     "category": pl.String, "title": pl.String, "sub_title": pl.String,
                     "strike_date": pl.Datetime("us", "UTC"), "pos": pl.Int64}


def event_list_rows(payload: dict[str, Any], series: str, start_pos: int = 0) -> list[dict[str, Any]]:
    """Rows of one ``GET /events`` page (``pos`` = order in the listing, newest first)."""
    from albert.venues.kalshi import parse_ts
    return [{"series_ticker": e.get("series_ticker") or series, "event_ticker": e.get("event_ticker"),
             "mutually_exclusive": e.get("mutually_exclusive"), "category": e.get("category"),
             "title": e.get("title"), "sub_title": e.get("sub_title"),
             "strike_date": parse_ts(e.get("strike_date")), "pos": start_pos + i}
            for i, e in enumerate(payload.get("events") or [])]


def fetch_series_events(s, series: list[str], *, max_pages: int = 10, workers: int = 2) -> pl.DataFrame:
    """Every event each series ever listed (``GET /events``, 200 per page, at most ``max_pages``),
    cached in ``data/cache/series_age/events.parquet`` with a per-series ``complete`` flag."""
    from albert.http import HttpError
    path = cache_dir(s) / "events.parquet"
    meta_path = cache_dir(s) / "events_series.parquet"
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=EVENT_LIST_SCHEMA)
    meta = pl.read_parquet(meta_path) if meta_path.exists() else pl.DataFrame(
        schema={"series_ticker": pl.String, "pages": pl.Int64, "complete": pl.Boolean, "error": pl.String})
    meta = meta.filter(pl.col("error").is_null())
    have = have.filter(pl.col("series_ticker").is_in(meta["series_ticker"].to_list()))
    seen = set(meta["series_ticker"].to_list())
    todo = [x for x in dict.fromkeys(series) if x not in seen]   # caller's priority order
    if not todo:
        return have
    api = _client(s)
    old_enough = OLD_BEFORE - timedelta(days=30)

    def one(ser: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        cursor, pages, err, reached_old = None, 0, None, False
        while True:
            try:
                page = _patient_get(api, "/events", {"series_ticker": ser, "limit": 200, "cursor": cursor}, tag=SNAPSHOT)
            except HttpError as error:
                err = str(error)[:200]
                break
            got = event_list_rows(page.payload, ser, len(rows))
            rows.extend(got)
            pages += 1
            cursor = page.payload.get("cursor")
            dates = [d for d in (ticker_date(r["event_ticker"]) or r["strike_date"] for r in got) if d is not None]
            # The listing is roughly newest first: once a whole page predates OLD_BEFORE by a
            # month, the series is old at every decision time and the rest is not needed.
            reached_old = bool(dates) and max(dates) < old_enough
            if not cursor or not page.payload.get("events") or pages >= max_pages or reached_old:
                break
        complete = err is None and (not cursor or reached_old or pages < max_pages)
        return rows, {"series_ticker": ser, "pages": pages, "complete": complete, "error": err}

    rows, metas = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (r, mrow) in enumerate(pool.map(one, todo), 1):
            rows.extend(r)
            metas.append(mrow)
            if i % 100 == 0:
                print(f"  events {i}/{len(todo)} series", flush=True)
                _atomic_write(pl.concat([have, pl.DataFrame(rows, schema=EVENT_LIST_SCHEMA)]), path)
                _atomic_write(pl.concat([meta, pl.DataFrame(metas, schema=meta.schema)]), meta_path)
    have = pl.concat([have, pl.DataFrame(rows, schema=EVENT_LIST_SCHEMA)]).unique(["series_ticker", "event_ticker"])
    _atomic_write(have, path)
    _atomic_write(pl.concat([meta, pl.DataFrame(metas, schema=meta.schema)]).unique("series_ticker", keep="last"), meta_path)
    return have


def event_dates(events: pl.DataFrame) -> pl.DataFrame:
    """Events with ``edate``: the date encoded in the ticker, else the strike date."""
    return events.with_columns(
        pl.coalesce(pl.col("event_ticker").map_elements(ticker_date, return_dtype=pl.Datetime("us", "UTC")),
                    "strike_date").alias("edate"))


def earliest_candidates(events: pl.DataFrame, table_min: pl.DataFrame) -> pl.DataFrame:
    """One probe event per series whose listing reaches earlier than the ingested table: the
    event with the earliest encoded date (or, with no dates at all, the one listed last). No
    probe when that date is already covered by the table (launch = table minimum) or is more
    than a month before ``OLD_BEFORE`` (the series is old at every decision time)."""
    ev = event_dates(events).join(table_min, on="series_ticker", how="left")
    dated = (ev.filter(pl.col("edate").is_not_null()).sort("edate").group_by("series_ticker").head(1)
             .filter((pl.col("edate") >= OLD_BEFORE - timedelta(days=30))
                     & (pl.col("edate") < pl.col("open_min_table") - timedelta(days=2))))
    undated_series = ev.group_by("series_ticker").agg(pl.col("edate").is_null().all().alias("nodate")).filter("nodate")
    last = (ev.join(undated_series, on="series_ticker").sort("pos", descending=True).group_by("series_ticker").head(1))
    return pl.concat([dated.select("series_ticker", "event_ticker"), last.select("series_ticker", "event_ticker")]).unique()


MKT_PROBE_SCHEMA = {"event_ticker": pl.String, "ticker": pl.String, "open_time": pl.Datetime("us", "UTC"),
                    "created_time": pl.Datetime("us", "UTC"), "close_time": pl.Datetime("us", "UTC")}


def probe_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    from albert.venues.kalshi import parse_ts
    return [{"event_ticker": m.get("event_ticker"), "ticker": m.get("ticker"), "open_time": parse_ts(m.get("open_time")),
             "created_time": parse_ts(m.get("created_time")), "close_time": parse_ts(m.get("close_time"))}
            for m in payload.get("markets") or []]


def fetch_event_markets(s, events: list[str], *, workers: int = 2) -> pl.DataFrame:
    """Markets of each event (historical archive first, then live ``/markets``): open times."""
    from albert.http import HttpError
    path = cache_dir(s) / "probe_markets.parquet"
    done_path = cache_dir(s) / "probe_events.parquet"
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=MKT_PROBE_SCHEMA)
    done = set(pl.read_parquet(done_path)["event_ticker"].to_list()) if done_path.exists() else set()
    todo = sorted(set(events) - done)
    if not todo:
        return have
    api = _client(s)

    def one(ev: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in ("/historical/markets", "/markets"):
            try:
                page = _patient_get(api, p, {"event_ticker": ev, "limit": 1000}, tag=SNAPSHOT)
            except HttpError as error:
                print(f"skip {p} {ev}: {str(error)[:100]}", flush=True)
                continue
            out.extend(probe_rows(page.payload))
            if out:
                break
        return out

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(one, todo), 1):
            rows.extend(r)
            if i % 500 == 0:
                print(f"  probe markets {i}/{len(todo)} events", flush=True)
    have = pl.concat([have, pl.DataFrame(rows, schema=MKT_PROBE_SCHEMA)]).unique("ticker")
    _atomic_write(have, path)
    _atomic_write(pl.DataFrame({"event_ticker": sorted(done | set(todo))}), done_path)
    return have


def build_launch(s, markets: pl.DataFrame, *, workers: int = 2, max_pages: int = 10) -> pl.DataFrame:
    """Launch date per series = min of: first open in the ingested table, first open of the
    probed earliest event, and (for series whose listing reaches far back) that event's encoded
    date, which is never earlier than its markets opened. ``launch_censored`` = the listing hit
    its page cap without reaching a pre-``OLD_BEFORE`` event (launch is then an upper bound)."""
    tab = markets.group_by("series_ticker").agg(pl.col("open_time").min().alias("open_min_table"),
                                                pl.col("category").first().alias("category"),
                                                pl.len().alias("n_markets"))
    # biggest series first, so a slow (shared, rate-limited) run covers most of the sample early
    need = (tab.filter(pl.col("open_min_table") >= OLD_BEFORE).sort("n_markets", descending=True)
            ["series_ticker"].to_list())
    events = fetch_series_events(s, need, workers=workers, max_pages=max_pages)
    events = events.filter(pl.col("series_ticker").is_in(need))
    cand = earliest_candidates(events, tab.select("series_ticker", "open_min_table"))
    probe = fetch_event_markets(s, cand["event_ticker"].to_list(), workers=workers)
    pmin = (probe.join(cand, on="event_ticker").group_by("series_ticker")
            .agg(pl.col("open_time").min().alias("open_min_probe")))
    emin = (event_dates(events).group_by("series_ticker")
            .agg(pl.col("edate").min().alias("edate_min"), pl.len().alias("events_listed")))
    meta = pl.read_parquet(cache_dir(s) / "events_series.parquet")
    out = (tab.join(pmin, on="series_ticker", how="left").join(meta, on="series_ticker", how="left")
           .join(emin, on="series_ticker", how="left")
           .with_columns(pl.when(pl.col("edate_min") < OLD_BEFORE - timedelta(days=30)).then("edate_min")
                         .alias("edate_old"))
           .with_columns(pl.min_horizontal("open_min_table", "open_min_probe", "edate_old").alias("launch"),
                         (pl.col("complete").not_() & pl.col("complete").is_not_null()).alias("launch_censored")))
    _atomic_write(out, cache_dir(s) / "launch.parquet")
    return out


# ---- step 2: decisions, sample, quotes --------------------------------------------------------
DATA_END = datetime(2026, 11, 1, tzinfo=UTC)
MARKET_COLS = ("ticker", "event_ticker", "series_ticker", "category", "title", "yes_sub_title", "open_time",
               "close_time", "expected_expiration_time", "settlement_ts", "result", "fee_type", "fee_multiplier")


def age_bucket(age_days: pl.Expr) -> pl.Expr:
    w = age_days / 7
    e0, e1, e2 = AGE_EDGES_WEEKS
    return (pl.when(w < e0).then(pl.lit(AGE_LABELS[0])).when(w < e1).then(pl.lit(AGE_LABELS[1]))
            .when(w < e2).then(pl.lit(AGE_LABELS[2])).otherwise(pl.lit(AGE_LABELS[3])))


def universe(s) -> pl.DataFrame:
    """Every settled yes/no market whose *scheduled* expiry falls in the data window. Using the
    scheduled expiry (not the actual close) keeps the universe free of outcome timing: a market
    still unsettled on Nov 1 can only be one scheduled after the window."""
    m = (pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet").select(*MARKET_COLS)
         .filter(pl.col("result").is_in(["yes", "no"])).collect())
    return (m.with_columns((pl.col("result") == "yes").cast(pl.Int8).alias("y"),
                           pl.coalesce("expected_expiration_time", "close_time").alias("t_ref"))
            .filter((pl.col("t_ref") >= DATA_START) & (pl.col("t_ref") < DATA_END)))


def decisions(m: pl.DataFrame, launch: pl.DataFrame) -> pl.DataFrame:
    """Decision rows: 24 h and 6 h before the scheduled expiry and at mid-life (open + half the
    scheduled life; dropped when within 6 h of another point). A decision exists only if the
    market is open (open + 1 h ≤ t < close) and t is inside the data window. Ages at ``t``:
    series (since launch), event (since the event's first leg opened) and market."""
    ev_open = m.group_by("event_ticker").agg(pl.col("open_time").min().alias("event_open"))
    base = m.join(ev_open, on="event_ticker").join(launch.select("series_ticker", "launch", "launch_censored"),
                                                   on="series_ticker", how="left")
    h24 = base.with_columns((pl.col("t_ref") - timedelta(hours=24)).alias("t"), pl.lit("h24").alias("kind"))
    h6 = base.with_columns((pl.col("t_ref") - timedelta(hours=6)).alias("t"), pl.lit("h6").alias("kind"))
    mid = (base.with_columns((pl.col("open_time") + (pl.col("t_ref") - pl.col("open_time")) / 2).alias("t"),
                             pl.lit("mid").alias("kind"))
           .filter(((pl.col("t") - (pl.col("t_ref") - timedelta(hours=24))).abs() >= timedelta(hours=6))
                   & ((pl.col("t") - (pl.col("t_ref") - timedelta(hours=6))).abs() >= timedelta(hours=6))))
    d = pl.concat([h24, h6, mid]).with_columns(pl.col("t").dt.cast_time_unit("us"))
    d = d.filter((pl.col("t") >= DATA_START) & (pl.col("t") >= pl.col("open_time") + timedelta(hours=1))
                 & (pl.col("t") < pl.col("close_time")))
    return d.with_columns(
        ((pl.col("t") - pl.col("launch")).dt.total_seconds() / 86400).alias("age_days"),
        ((pl.col("t") - pl.col("event_open")).dt.total_seconds() / 86400).alias("event_age_days"),
        ((pl.col("t") - pl.col("open_time")).dt.total_seconds() / 86400).alias("market_age_days"),
        (pl.col("t") >= HOLDOUT_FROM).alias("holdout"),
    ).with_columns(age_bucket(pl.col("age_days")).alias("age_bucket"),
                   pl.col("event_ticker").alias("event"))


def _rank(key: str) -> int:
    import hashlib
    return int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()[:12], 16)


HIST_CUTOFF = datetime(2026, 9, 6, tzinfo=UTC)   # Kalshi market_settled_ts on 2026-11-05


def sample_events(d: pl.DataFrame, *, k_young: int = 3, k_old: int = 2, k_hist: int = 1, max_legs: int = 6,
                  hist_legs: int = 2, hist_before: datetime = HIST_CUTOFF) -> pl.DataFrame:
    """Seeded, volume-blind sample of events per (series × age bucket of the event's first
    decision × stratum): ``k_young`` events per bucket younger than 16 weeks, ``k_old`` in the
    oldest. Strata: holdout, dev settled after the historical cutoff, and dev settled before it,
    which costs one candle request per market (no batch endpoint) and so draws only ``k_hist``
    events per cell and ``hist_legs`` legs per event. Otherwise all legs of a sampled event are
    kept up to ``max_legs`` (else a seeded subset, flagged ``legs_capped`` so ladder sums skip it)."""
    first = (d.sort("t").group_by("event_ticker", "holdout")
             .agg(pl.col("series_ticker").first(), pl.col("age_bucket").first().alias("cell_bucket"),
                  pl.col("close_time").max().alias("ev_close")))
    first = first.with_columns(
        pl.when(pl.col("holdout")).then(pl.lit("holdout")).when(pl.col("ev_close") < hist_before)
        .then(pl.lit("dev_hist")).otherwise(pl.lit("dev_live")).alias("stratum"),
        pl.col("event_ticker").map_elements(_rank, return_dtype=pl.Int64).alias("rk"))
    first = first.with_columns(pl.col("rk").rank("ordinal").over("series_ticker", "cell_bucket", "stratum").alias("r"))
    k = (pl.when(pl.col("stratum") == "dev_hist").then(pl.lit(k_hist))
         .when(pl.col("cell_bucket") == AGE_LABELS[3]).then(pl.lit(k_old)).otherwise(pl.lit(k_young)))
    keep = first.filter(pl.col("r") <= k)
    sd = d.join(keep.select("event_ticker", "holdout", "cell_bucket", "stratum"), on=["event_ticker", "holdout"])
    legs = (d.filter(pl.col("event_ticker").is_in(keep["event_ticker"].to_list())).select("event_ticker", "ticker").unique()
            .join(keep.select("event_ticker", "stratum").unique("event_ticker", keep="first"), on="event_ticker"))
    legs = legs.with_columns(pl.col("ticker").map_elements(_rank, return_dtype=pl.Int64).alias("lr"),
                             pl.len().over("event_ticker").alias("n_legs"),
                             pl.when(pl.col("stratum") == "dev_hist").then(pl.lit(hist_legs)).otherwise(pl.lit(max_legs))
                             .alias("cap"))
    legs = legs.with_columns(pl.col("lr").rank("ordinal").over("event_ticker").alias("lrank"))
    legs = legs.filter(pl.col("lrank") <= pl.col("cap")).with_columns((pl.col("n_legs") > pl.col("cap")).alias("legs_capped"))
    return sd.join(legs.select("ticker", "n_legs", "legs_capped"), on="ticker")


def candle_windows(sd: pl.DataFrame, *, pre: timedelta = timedelta(hours=3),
                   split_after: timedelta = timedelta(days=3), hist_before: datetime = HIST_CUTOFF) -> pl.DataFrame:
    """One window per market covering its decisions (``[first t − pre, last t]``). Markets on
    the batch endpoint (settled after the cutoff) get one window per decision instead when the
    decisions are more than ``split_after`` apart (the batch shares one window and caps candles);
    historical markets always get one joint window (one request either way)."""
    g = sd.group_by("ticker").agg(pl.col("t").min().alias("t0"), pl.col("t").max().alias("t1"),
                                  pl.col("settlement_ts").first(), pl.col("close_time").first())
    is_hist = pl.coalesce("settlement_ts", "close_time") < hist_before
    one = is_hist | (pl.col("t1") - pl.col("t0") <= split_after)
    joint = g.filter(one).select(
        "ticker", (pl.col("t0") - pre).alias("lo"), pl.col("t1").alias("hi"), "settlement_ts", "close_time")
    split = (sd.join(g.filter(~one).select("ticker"), on="ticker")
             .select("ticker", (pl.col("t") - pre).alias("lo"), pl.col("t").alias("hi"), "settlement_ts", "close_time"))
    w = pl.concat([joint, split]).unique(["ticker", "lo", "hi"])
    return w.with_columns(pl.col("lo").dt.truncate("1h"), pl.col("hi").dt.truncate("1h") + timedelta(hours=1))


def batch_candle_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows of a ``GET /markets/candlesticks`` (batch) response, harness candle format."""
    from albert.venues import kalshi as k
    return [r for mk in payload.get("markets") or [] for r in k.candle_rows(mk, mk.get("market_ticker"), 60)]


def fetch_window_candles(s, windows: pl.DataFrame, *, max_candles: int = 9000, max_batch: int = 100,
                         workers: int = 2) -> pl.DataFrame:
    """Hourly candles for each (ticker, lo, hi) window, cached in
    ``data/cache/fundamental_series_age_candles.parquet`` (same rows as the harness). Markets
    settled after the historical cutoff go through the batch endpoint (up to 100 tickers and
    10,000 candles per request); older ones through ``/historical/markets/{t}/candlesticks``."""
    from albert.http import HttpError
    from albert.venues import kalshi as k

    path = s.data_dir / "cache" / f"fundamental_{NAME}_candles.parquet"
    done_path = cache_dir(s) / "candle_windows_done.parquet"
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=k.CANDLE_SCHEMA)
    done = pl.read_parquet(done_path) if done_path.exists() else pl.DataFrame(
        schema={"ticker": pl.String, "lo": pl.Datetime("us", "UTC"), "hi": pl.Datetime("us", "UTC")})
    todo = windows.join(done, on=["ticker", "lo", "hi"], how="anti")
    if not todo.height:
        return have
    api = _client(s, min_interval=0.12)
    kal = k.Kalshi(api)
    cutoff = kal.cutoff().get("market_settled_ts")
    settled = pl.coalesce("settlement_ts", "close_time")
    hist = todo.filter(settled < cutoff) if cutoff else todo.head(0)
    live = todo.filter(settled >= cutoff) if cutoff else todo

    def flush(rows: list[dict[str, Any]], finished: list[dict[str, Any]]) -> None:
        nonlocal have, done
        if rows:
            have = pl.concat([have, pl.DataFrame(rows, schema=k.CANDLE_SCHEMA)]).unique(["ticker", "end_ts", "period_minutes"])
            _atomic_write(have, path)
        if finished:
            done = pl.concat([done, pl.DataFrame(finished, schema=done.schema)]).unique()
            _atomic_write(done, done_path)

    # live: batch requests over windows sorted by start, one ticker per chunk at most once
    ws = sorted(live.select("ticker", "lo", "hi").iter_rows(named=True), key=lambda r: (r["lo"], r["ticker"]))
    chunks, cur = [], []
    for w in ws:
        trial = cur + [w]
        lo, hi = min(x["lo"] for x in trial), max(x["hi"] for x in trial)
        hours = (hi - lo).total_seconds() / 3600 + 1
        if cur and (len(trial) > max_batch or hours * len(trial) > max_candles
                    or w["ticker"] in {x["ticker"] for x in cur}):
            chunks.append(cur)
            cur = [w]
        else:
            cur = trial
    if cur:
        chunks.append(cur)

    def batch(ch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        lo, hi = min(x["lo"] for x in ch), max(x["hi"] for x in ch)
        try:
            page = _patient_get(api, "/markets/candlesticks", {
                "market_tickers": ",".join(sorted(x["ticker"] for x in ch)), "start_ts": int(lo.timestamp()),
                "end_ts": int(hi.timestamp()), "period_interval": 60})
        except HttpError as error:
            print(f"skip batch at {ch[0]['ticker']}: {str(error)[:160]}", flush=True)
            return []
        return batch_candle_rows(page.payload)

    def single(w: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            page = _patient_get(api, f"/historical/markets/{w['ticker']}/candlesticks", {
                "start_ts": int(w["lo"].timestamp()), "end_ts": int(w["hi"].timestamp()), "period_interval": 60})
        except HttpError as error:
            print(f"skip candles {w['ticker']}: {str(error)[:120]}", flush=True)
            return []
        return k.candle_rows(page.payload, w["ticker"], 60)

    jobs: list[tuple[Any, Any, list[dict[str, Any]]]] = [(batch, ch, ch) for ch in chunks]
    jobs += [(single, w, [w]) for w in hist.select("ticker", "lo", "hi").iter_rows(named=True)]
    print(f"  candles: {len(chunks)} batch requests for {live.height} live windows, {hist.height} historical", flush=True)
    rows: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (got, covered) in enumerate(pool.map(lambda j: (j[0](j[1]), j[2]), jobs), 1):
            rows.extend(got)
            finished.extend({"ticker": x["ticker"], "lo": x["lo"], "hi": x["hi"]} for x in covered)
            if i % 300 == 0:
                print(f"  candle jobs {i}/{len(jobs)}", flush=True)
                flush(rows, finished)
                rows, finished = [], []
    flush(rows, finished)
    return have


def fetch_event_flags(s, events: list[str], *, workers: int = 2) -> pl.DataFrame:
    """``mutually_exclusive`` for events not covered by the series listings (``GET /events/{t}``)."""
    from albert.http import HttpError
    path = cache_dir(s) / "event_flags.parquet"
    schema = {"event_ticker": pl.String, "mutually_exclusive": pl.Boolean}
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=schema)
    todo = sorted(set(events) - set(have["event_ticker"].to_list()))
    if not todo:
        return have
    api = _client(s)

    def one(ev: str) -> dict[str, Any]:
        try:
            e = (_patient_get(api, f"/events/{ev}", {}).payload or {}).get("event") or {}
        except HttpError as error:
            print(f"skip event {ev}: {str(error)[:100]}", flush=True)
            return {"event_ticker": ev, "mutually_exclusive": None}
        return {"event_ticker": ev, "mutually_exclusive": e.get("mutually_exclusive")}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, todo))
    have = pl.concat([have, pl.DataFrame(rows, schema=schema)]).unique("event_ticker")
    _atomic_write(have, path)
    return have


# ---- step 2b: calibration ---------------------------------------------------------------------
BAND_EDGES = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95)


def price_band(mid: pl.Expr) -> pl.Expr:
    labels = ["b00_05", "b05_10", "b10_20", "b20_40", "b40_60", "b60_80", "b80_90", "b90_95", "b95_100"]
    expr = pl.when(mid < BAND_EDGES[0]).then(pl.lit(labels[0]))
    for edge, lab in zip(BAND_EDGES[1:], labels[1:-1], strict=True):
        expr = expr.when(mid < edge).then(pl.lit(lab))
    return expr.otherwise(pl.lit(labels[-1]))


def split_expr() -> pl.Expr:
    return pl.when(pl.col("holdout")).then(pl.lit("holdout")).otherwise(pl.lit("dev"))


def with_quotes(sd: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    """Attach the executable quote at ``t`` (harness rule) and derived columns. Dev rows must
    also have closed before the holdout starts, so no dev outcome is learned in October."""
    from albert.research.fundamental import attach_quotes
    q = attach_quotes(sd, candles)
    q = q.filter(pl.col("holdout") | (pl.col("close_time") < HOLDOUT_FROM))
    return q.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                          (pl.col("ask") - pl.col("bid")).alias("spread"), split_expr().alias("split")
                          ).with_columns(price_band(pl.col("mid")).alias("band"))


MIN_CLUSTERS = 10


def _t(t: float | None, clusters: int) -> float | None:
    """A clustered t-stat, or None with fewer than ``MIN_CLUSTERS`` clusters (meaningless)."""
    return None if t is None or clusters < MIN_CLUSTERS else round(float(t), 2)


def calib_table(q: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Per group: n, events, series, mean mid, mean y, y − mid in cents with t by event and by series."""
    from albert.research.fundamental import clustered_mean
    out = []
    for key, part in q.group_by(by, maintain_order=False):
        r = part["y"].to_numpy().astype(float) - part["mid"].to_numpy()
        mu, te, ke = clustered_mean(r, part["event"].to_numpy())
        _, ts, ks = clustered_mean(r, part["series_ticker"].to_numpy())
        out.append({**dict(zip(by, key, strict=True)), "n": part.height, "events": ke, "series": ks,
                    "mean_mid": round(float(part["mid"].mean()), 4), "mean_y": round(float(part["y"].mean()), 4),
                    "y_minus_mid_c": round(mu * 100, 2), "t_event": _t(te, ke), "t_series": _t(ts, ks),
                    "mean_spread_c": round(float(part["spread"].mean()) * 100, 2)})
    return pl.DataFrame(out).sort(by) if out else pl.DataFrame()


def _cluster_t(X: np.ndarray, y: np.ndarray, beta: np.ndarray, g: np.ndarray) -> np.ndarray:
    mu = 1 / (1 + np.exp(-X @ beta))
    Hinv = np.linalg.pinv(X.T @ (X * (mu * (1 - mu))[:, None]))
    _, inv = np.unique(g, return_inverse=True)
    S = np.zeros((inv.max() + 1, X.shape[1]))
    np.add.at(S, inv, X * (y - mu)[:, None])
    V = Hinv @ (S.T @ S) @ Hinv
    return np.sqrt(np.clip(np.diag(V), 1e-18, None))


def logit_fits(q: pl.DataFrame, by: list[str], *, min_n: int = 50) -> pl.DataFrame:
    """``logit P(y) = a + b·logit(mid)`` per group; calibrated is a = 0, b = 1. t of a and of
    (b − 1), clustered by event and by series. b > 1: longshots overpriced (and favourites
    underpriced); a < 0: YES overpriced at every price."""
    from albert.research.fundamental import fit_logistic, logit
    out = []
    for key, part in q.group_by(by):
        if part.height < min_n or part["y"].n_unique() < 2:
            continue
        X = np.column_stack([np.ones(part.height), logit(part["mid"].to_numpy(), eps=0.01)])
        y = part["y"].to_numpy().astype(float)
        beta, _ = fit_logistic(X, y, part["event"].to_numpy())
        se_e = _cluster_t(X, y, beta, part["event"].to_numpy())
        se_s = _cluster_t(X, y, beta, part["series_ticker"].to_numpy())
        ne, ns = part["event"].n_unique(), part["series_ticker"].n_unique()
        out.append({**dict(zip(by, key, strict=True)), "n": part.height, "events": ne, "series": ns,
                    "a": round(float(beta[0]), 3), "b": round(float(beta[1]), 3),
                    "t_a_event": _t(beta[0] / se_e[0], ne), "t_b1_event": _t((beta[1] - 1) / se_e[1], ne),
                    "t_a_series": _t(beta[0] / se_s[0], ns), "t_b1_series": _t((beta[1] - 1) / se_s[1], ns)})
    return pl.DataFrame(out).sort(by) if out else pl.DataFrame()


def design(q: pl.DataFrame, cats: list[str], *, young: tuple[str, ...] = AGE_LABELS[:3]) -> tuple[np.ndarray, list[str]]:
    """Category intercepts + common slope on logit(mid) + per-young-bucket shifts in intercept
    and slope (the oldest bucket is the reference)."""
    from albert.research.fundamental import logit
    L = logit(q["mid"].to_numpy(), eps=0.01)
    cols, names = [], []
    cat = q["category"].to_numpy()
    for c in cats:
        cols.append((cat == c).astype(float))
        names.append(f"cat:{c}")
    cols.append(L)
    names.append("logit_mid")
    ab = q["age_bucket"].to_numpy()
    for k in young:
        d = (ab == k).astype(float)
        cols += [d, d * L]
        names += [f"a:{k}", f"b:{k}"]
    return np.column_stack(cols), names


def pooled_fit(q: pl.DataFrame, cats: list[str] | None = None) -> tuple[pl.DataFrame, np.ndarray, list[str]]:
    """Pooled logistic with category fixed effects and young-bucket intercept/slope shifts,
    t clustered by event and by series."""
    from albert.research.fundamental import fit_logistic
    cats = cats or sorted(q["category"].unique().to_list())
    # a category whose outcomes are all one value separates perfectly (no finite intercept)
    varied = q.group_by("category").agg(pl.col("y").n_unique().alias("k")).filter(pl.col("k") > 1)["category"].to_list()
    cats = [c for c in cats if c in varied]
    q = q.filter(pl.col("category").is_in(cats))
    X, names = design(q, cats)
    keep = X.std(axis=0) > 0
    keep[: len(cats)] = X[:, : len(cats)].sum(axis=0) > 0
    X, names = X[:, keep], [n for n, k in zip(names, keep, strict=True) if k]
    y = q["y"].to_numpy().astype(float)
    beta, _ = fit_logistic(X, y, q["event"].to_numpy())
    se_e = _cluster_t(X, y, beta, q["event"].to_numpy())
    se_s = _cluster_t(X, y, beta, q["series_ticker"].to_numpy())
    null = np.array([1.0 if n == "logit_mid" else 0.0 for n in names])
    tab = pl.DataFrame({"term": names, "coef": np.round(beta, 3),
                        "t_event": np.round((beta - null) / se_e, 2), "t_series": np.round((beta - null) / se_s, 2)})
    return tab, beta, names


def predict_pooled(q: pl.DataFrame, beta: np.ndarray, names: list[str], cats: list[str]) -> np.ndarray:
    X, all_names = design(q, cats)
    idx = [all_names.index(n) for n in names]
    return 1 / (1 + np.exp(-X[:, idx] @ beta))


def ladder_sums(q: pl.DataFrame, m: pl.DataFrame, flags: pl.DataFrame) -> pl.DataFrame:
    """Σ mid − 1 (and Σ ask − 1, 1 − Σ bid) for mutually exclusive events at each decision kind,
    only when every leg open at ``t`` (per the markets table) has a quote and the legs that
    settled contain exactly one YES (a complete, exhaustive ladder)."""
    me = flags.filter(pl.col("mutually_exclusive").fill_null(False))["event_ticker"].to_list()
    qq = q.filter(pl.col("event_ticker").is_in(me) & ~pl.col("legs_capped"))
    g = qq.group_by("event_ticker", "kind").agg(
        pl.col("t").min(), pl.col("t").max().alias("t_max"), pl.len().alias("quoted"), pl.col("mid").sum().alias("sum_mid"),
        pl.col("ask").sum().alias("sum_ask"), pl.col("bid").sum().alias("sum_bid"), pl.col("age_days").first(),
        (pl.col("ask") - pl.col("bid")).max().alias("max_leg_spread"),
        pl.col("age_bucket").first(), pl.col("event_age_days").first(), pl.col("series_ticker").first(),
        pl.col("category").first(), pl.col("split").first())
    legs = m.filter(pl.col("event_ticker").is_in(g["event_ticker"].unique().to_list())).select(
        "event_ticker", "open_time", "close_time", "y")
    yes = legs.group_by("event_ticker").agg(pl.col("y").sum().alias("n_yes"), pl.len().alias("n_legs_total"))
    open_at = (g.select("event_ticker", "kind", "t").join(legs, on="event_ticker")
               .filter((pl.col("open_time") <= pl.col("t")) & (pl.col("close_time") > pl.col("t")))
               .group_by("event_ticker", "kind").agg(pl.len().alias("open_legs")))
    out = (g.join(open_at, on=["event_ticker", "kind"]).join(yes, on="event_ticker")
           .filter((pl.col("t_max") - pl.col("t") < timedelta(hours=1)) & (pl.col("quoted") == pl.col("open_legs"))
                   & (pl.col("n_yes") == 1) & (pl.col("open_legs") >= 2)))
    return out.with_columns((pl.col("sum_mid") - 1).alias("overround_mid"), (pl.col("sum_ask") - 1).alias("overround_ask"),
                            (1 - pl.col("sum_bid")).alias("underround_bid"), pl.col("event_ticker").alias("event"))


def ladder_table(lad: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    from albert.research.fundamental import clustered_mean
    out = []
    for key, part in lad.group_by(by):
        x = part["overround_mid"].to_numpy()
        mu, ts, ks = clustered_mean(x, part["series_ticker"].to_numpy())
        out.append({**dict(zip(by, key, strict=True)), "ladders": part.height, "events": part["event"].n_unique(),
                    "series": ks, "mean_legs": round(float(part["open_legs"].mean()), 1),
                    "sum_mid_minus_1_c": round(mu * 100, 2), "median_c": round(float(np.median(x)) * 100, 2),
                    "t_series": _t(ts, ks),
                    "sum_ask_minus_1_c": round(float(part["overround_ask"].mean()) * 100, 2),
                    "one_minus_sum_bid_c": round(float(part["underround_bid"].mean()) * 100, 2)})
    return pl.DataFrame(out).sort(by) if out else pl.DataFrame()


def within_series(q: pl.DataFrame, *, label: str) -> list[dict[str, Any]]:
    """OLS of (y − mid) on age-bucket dummies + calendar-month dummies with series fixed
    effects (both sides demeaned within series), SE clustered by series: does the same series
    become better calibrated as it ages, net of the calendar month?"""
    if q.height < 50:
        return []
    q = q.with_columns((pl.col("y").cast(pl.Float64) - pl.col("mid")).alias("r"),
                       pl.col("t").dt.strftime("%Y-%m").alias("month"))
    buckets = [b for b in AGE_LABELS[:3] if (q["age_bucket"] == b).any()]
    months = sorted(q["month"].unique().to_list())[1:]
    X = np.column_stack([(q["age_bucket"] == b).to_numpy().astype(float) for b in buckets]
                        + [(q["month"] == mo).to_numpy().astype(float) for mo in months])
    r = q["r"].to_numpy()
    _, g = np.unique(q["series_ticker"].to_numpy(), return_inverse=True)
    cnt = np.bincount(g)

    def demean(v: np.ndarray) -> np.ndarray:
        if v.ndim == 1:
            return v - (np.bincount(g, weights=v) / cnt)[g]
        return np.column_stack([demean(v[:, j]) for j in range(v.shape[1])])

    Xd, rd = demean(X), demean(r)
    keep = Xd.std(axis=0) > 1e-12
    Xd = Xd[:, keep]
    names = [n for n, k in zip(buckets + months, keep, strict=True) if k]
    if not names:
        return []
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ rd
    e = rd - Xd @ beta
    S = np.zeros((cnt.size, Xd.shape[1]))
    np.add.at(S, g, Xd * e[:, None])
    V = XtX_inv @ (S.T @ S) @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 1e-18, None))
    movers = int((q.group_by("series_ticker").agg(pl.col("age_bucket").n_unique().alias("k"))["k"] > 1).sum())
    return [{"subset": label, "term": n, "coef_c": round(float(b) * 100, 2), "t_series": _t(b / s_, int(cnt.size)),
             "n": q.height, "series": int(cnt.size), "series_spanning_buckets": movers}
            for n, b, s_ in zip(names, beta, se, strict=True)]


# ---- step 3: rules ----------------------------------------------------------------------------
AGE_RULES = {"lt2w": AGE_LABELS[:1], "lt6w": AGE_LABELS[:2], "lt16w": AGE_LABELS[:3], "all": AGE_LABELS}
PRICE_RULES = (0.05, 0.10, 0.20, 0.30)


def sell_yes_pnl(q: pl.DataFrame) -> pl.DataFrame:
    """Sell YES at the bid (= buy NO at 1 − bid) as a taker, one contract, held to settlement,
    series taker fee on the NO price: pnl = bid − y − fee."""
    from albert.research.fundamental import taker_fee
    q = q.filter(pl.col("bid") >= 0.01)
    fee = [taker_fee(1 - b, m or 1.0) for b, m in zip(q["bid"].to_list(), q["fee_multiplier"].to_list(), strict=True)]
    return q.with_columns((pl.col("bid") - pl.col("y") - pl.Series(fee)).alias("pnl"))


def rule_grid(q: pl.DataFrame) -> pl.DataFrame:
    """Every (age condition × mid < X) sell-YES taker rule, per split: trades, PnL per contract
    (cents) with t by event and by series."""
    from albert.research.fundamental import clustered_mean
    tr = sell_yes_pnl(q)
    out = []
    for an, ab in AGE_RULES.items():
        for x in PRICE_RULES:
            for split in ("dev", "holdout"):
                part = tr.filter(pl.col("age_bucket").is_in(list(ab)) & (pl.col("mid") < x) & (pl.col("split") == split))
                if not part.height:
                    continue
                mu, te, ke = clustered_mean(part["pnl"].to_numpy(), part["event"].to_numpy())
                _, ts, ks = clustered_mean(part["pnl"].to_numpy(), part["series_ticker"].to_numpy())
                out.append({"age": an, "mid_below": x, "split": split, "trades": part.height, "events": ke, "series": ks,
                            "pnl_c": round(mu * 100, 3), "t_event": _t(te, ke), "t_series": _t(ts, ks),
                            "gross_c": round(float((part["bid"] - part["y"]).mean()) * 100, 3),
                            "mean_bid_c": round(float(part["bid"].mean()) * 100, 2)})
    return pl.DataFrame(out)


def choose_rule(grid: pl.DataFrame, *, min_trades: int = 200) -> dict[str, Any] | None:
    """Best dev PnL per contract among rules with ≥ ``min_trades`` dev trades."""
    dev = grid.filter((pl.col("split") == "dev") & (pl.col("trades") >= min_trades)).sort("pnl_c", descending=True)
    return dev.row(0, named=True) if dev.height else None


def event_age_bucket(days: pl.Expr) -> pl.Expr:
    return (pl.when(days < 1).then(pl.lit("e0_lt1d")).when(days < 3).then(pl.lit("e1_1_3d"))
            .when(days < 7).then(pl.lit("e2_3_7d")).otherwise(pl.lit("e3_ge7d")))


def maker_sample(rows: pl.DataFrame, n: int) -> pl.DataFrame:
    """Seeded subsample of at most ``n`` rows per split (volume-blind)."""
    rows = rows.with_columns(pl.concat_str("ticker", pl.col("t").dt.strftime("%Y%m%d%H"), separator="@")
                             .map_elements(_rank, return_dtype=pl.Int64).alias("_rk"))
    return rows.sort("_rk").group_by("split", maintain_order=True).head(n).drop("_rk")


def run(s, *, k_young: int = 3, k_old: int = 2, k_hist: int = 1, max_legs: int = 6, maker_rows: int = 400,
        max_spread: float = 0.10, workers: int = 2, fetch: bool = True) -> dict[str, Any]:
    """Full study; returns every table (also written by :func:`write_report`). Calibration uses
    quotes with spread ≤ ``max_spread`` (a mid between an empty bid and a 99c ask is not a
    price); rules trade every quoted row at its own bid."""
    from albert.research.fundamental import evaluate, evaluate_maker, fetch_trades, maker_fills, clustered_mean

    m = universe(s)
    lpath = cache_dir(s) / "launch.parquet"
    launch = pl.read_parquet(lpath) if lpath.exists() else build_launch(s, m, workers=workers)
    full_min = (pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet").group_by("series_ticker")
                .agg(pl.col("open_time").min().alias("open_min_full")).collect())
    launch = (launch.join(full_min, on="series_ticker", how="left")
              .with_columns(pl.min_horizontal("launch", "open_min_full").alias("launch")))
    # A capped listing bounds the launch from above only. If the series' first market in the
    # table is well inside the window (a busy series that would otherwise have had markets
    # since Aug 15), take that as the launch and flag it ("table_only", sensitivity check);
    # otherwise keep only decisions the bound already proves are ≥ 16 weeks old.
    launch = launch.with_columns(
        (pl.col("launch_censored") & (pl.col("open_min_full") >= DATA_START + timedelta(days=3))).alias("launch_table_only"))
    ambiguous = launch.filter(pl.col("launch_censored") & ~pl.col("launch_table_only"))["series_ticker"].to_list()
    launch = launch.filter(pl.col("launch").is_not_null())
    m = m.filter(pl.col("series_ticker").is_in(launch["series_ticker"].to_list()))
    d = decisions(m, launch)
    n_before = d.height
    d = d.filter(~pl.col("series_ticker").is_in(ambiguous) | (pl.col("age_bucket") == AGE_LABELS[3]))
    unknown = n_before - d.height
    d = d.join(launch.select("series_ticker", "launch_table_only"), on="series_ticker", how="left")
    sd = sample_events(d, k_young=k_young, k_old=k_old, k_hist=k_hist, max_legs=max_legs)
    print(f"sample: {sd.height} decisions, {sd['ticker'].n_unique()} markets, {sd['event_ticker'].n_unique()} events, "
          f"{sd['series_ticker'].n_unique()} series", flush=True)
    candles = fetch_window_candles(s, candle_windows(sd), workers=workers) if fetch else pl.read_parquet(
        s.data_dir / "cache" / f"fundamental_{NAME}_candles.parquet")
    qa = with_quotes(sd, candles).with_columns(event_age_bucket(pl.col("event_age_days")).alias("event_age_bucket"))
    q = qa.filter(pl.col("spread") <= max_spread + 1e-9)
    out: dict[str, Any] = {"launch": launch, "sample_rows": sd.height, "quoted_rows": qa.height,
                           "calib_rows": q.height, "decisions_age_ambiguous_dropped": unknown,
                           "series_launch_table_only": launch.filter("launch_table_only")["series_ticker"].to_list()}
    cov = (sd.with_columns(split_expr().alias("split")).group_by("split", "age_bucket").agg(pl.len().alias("decisions"))
           .join(qa.group_by("split", "age_bucket").agg(pl.len().alias("quoted")), on=["split", "age_bucket"], how="left")
           .join(q.group_by("split", "age_bucket").agg(pl.len().alias("spread_ok")), on=["split", "age_bucket"], how="left")
           .with_columns((pl.col("quoted") / pl.col("decisions")).round(3).alias("coverage"),
                         (pl.col("spread_ok") / pl.col("decisions")).round(3).alias("coverage_spread_ok"))
           .sort("split", "age_bucket"))
    out["coverage"] = cov
    out["launch_summary"] = (launch.with_columns(pl.col("launch").dt.strftime("%Y-%m").alias("launch_month"))
                             .with_columns(pl.when(pl.col("launch") < OLD_BEFORE).then(pl.lit("<2026-03"))
                                           .otherwise(pl.col("launch_month")).alias("launch_month"))
                             .group_by("launch_month").agg(pl.len().alias("series"), pl.col("launch_censored").sum().alias("censored"))
                             .sort("launch_month"))
    out["calib_bucket"] = calib_table(q, ["split", "age_bucket"])
    out["calib_bucket_allspreads"] = calib_table(qa, ["split", "age_bucket"])
    out["calib_bucket_band"] = calib_table(q, ["split", "age_bucket", "band"])
    out["calib_cat_bucket"] = calib_table(q, ["split", "category", "age_bucket"])
    out["calib_eventage"] = calib_table(q, ["split", "event_age_bucket"])
    out["calib_eventage_young"] = calib_table(q.filter(pl.col("age_bucket") != AGE_LABELS[3]), ["split", "event_age_bucket"])
    period = (pl.when(pl.col("holdout")).then(pl.lit("p3_oct")).when(pl.col("t") < HIST_CUTOFF)
              .then(pl.lit("p1_aug15_sep6")).otherwise(pl.lit("p2_sep6_30")).alias("period"))
    yo = pl.when(pl.col("age_bucket") == AGE_LABELS[3]).then(pl.lit("old")).otherwise(pl.lit("young")).alias("young_old")
    out["calib_period"] = calib_table(q.with_columns(period), ["period", "age_bucket"])
    out["calib_period_longshots"] = calib_table(q.with_columns(period).filter(pl.col("mid") < 0.2), ["period", "age_bucket"])
    sband = (pl.when(pl.col("spread") <= 0.05).then(pl.lit("s0_le5c")).when(pl.col("spread") <= 0.10)
             .then(pl.lit("s1_5_10c")).when(pl.col("spread") <= 0.30).then(pl.lit("s2_10_30c"))
             .otherwise(pl.lit("s3_gt30c")).alias("spread_band"))
    out["calib_spread_band"] = calib_table(qa.with_columns(sband, yo), ["split", "young_old", "spread_band"])
    out["calib_cat_longshots"] = calib_table(q.with_columns(yo).filter(pl.col("mid") < 0.2),
                                             ["split", "category", "young_old"])
    out["fits_bucket"] = logit_fits(q, ["split", "age_bucket"])
    out["fits_cat_bucket"] = logit_fits(q, ["split", "category", "age_bucket"], min_n=100)
    cats = sorted(q["category"].unique().to_list())
    pooled = {}
    for split in ("dev", "holdout"):
        part = q.filter(pl.col("split") == split)
        if part.height > 100:
            pooled[split] = pooled_fit(part, cats)
    out["pooled"] = {k: v[0] for k, v in pooled.items()}
    ws = []
    for split in ("dev", "holdout", "all"):
        part = q if split == "all" else q.filter(pl.col("split") == split)
        for label, sub in (("all_prices", part), ("mid_lt_20", part.filter(pl.col("mid") < 0.2)),
                           ("mid_gt_80", part.filter(pl.col("mid") > 0.8))):
            ws += [{"split": split, **r} for r in within_series(sub, label=label)]
    out["within_series"] = pl.DataFrame(ws) if ws else pl.DataFrame()

    # ladders: mutually exclusive flag from the series listings, else GET /events/{t}
    ev_path = cache_dir(s) / "events.parquet"
    listed = pl.read_parquet(ev_path).select("event_ticker", "mutually_exclusive") if ev_path.exists() else pl.DataFrame(
        schema={"event_ticker": pl.String, "mutually_exclusive": pl.Boolean})
    sampled = qa["event_ticker"].unique()
    flags = listed.filter(pl.col("event_ticker").is_in(sampled)).unique("event_ticker")
    missing = sampled.filter(~sampled.is_in(flags["event_ticker"].to_list())).to_list()
    if missing and fetch:
        flags = pl.concat([flags, fetch_event_flags(s, missing, workers=workers)]).unique("event_ticker")
    lad = ladder_sums(qa, m, flags)
    out["ladders_all"] = ladder_table(lad, ["split", "age_bucket"])
    lad = lad.filter(pl.col("max_leg_spread") <= max_spread + 1e-9)
    out["ladders"] = ladder_table(lad, ["split", "age_bucket"])
    out["ladders_eventage"] = ladder_table(lad.with_columns(event_age_bucket(pl.col("event_age_days")).alias("event_age_bucket")),
                                           ["split", "event_age_bucket"])
    out["ladders_cat"] = ladder_table(lad, ["split", "category", "age_bucket"])

    # rules: sell YES below X in series younger than N, taker at the bid with series fees
    grid = rule_grid(qa)
    out["rule_grid"] = grid
    chosen = choose_rule(grid)
    out["rule_chosen"] = chosen

    # fair value: pooled dev fit (category intercepts, young-bucket intercept/slope shifts)
    if "dev" in pooled:
        _, beta, names = pooled["dev"]
        qa = qa.filter(pl.col("category").is_in(cats))
        qa = qa.with_columns(pl.Series("p", predict_pooled(qa, beta, names, cats)))
        ev = evaluate(qa, HOLDOUT_FROM)
        out["fair_eval"] = ev
        out["holdout_preds"] = (qa.filter(pl.col("holdout")).select("ticker", "t", pl.col("p").alias("p_fair"), "mid",
                                                                    "bid", "ask", "y", "age_bucket", "series_ticker", "category"))
        # maker: young-series longshots (< 16 weeks, mid < 30c) — the hypothesis' cell, broad
        # enough to fill — with old-series longshots as the control. h24/h6 decisions only: the
        # order rests until the close, and mid-life rows would rest for weeks.
        base = (pl.col("mid") < 0.3) & (pl.col("kind") != "mid")
        young = pl.col("age_bucket").is_in(list(AGE_RULES["lt16w"]))
        mk_y = maker_sample(qa.filter(base & young), maker_rows)
        mk_o = maker_sample(qa.filter(base & ~young), maker_rows // 2)
        mk = pl.concat([mk_y.with_columns(pl.lit("young").alias("group")), mk_o.with_columns(pl.lit("old").alias("group"))])
        mk = mk.with_columns(pl.col("close_time").alias("end"))
        out["maker_rows"] = {"young": mk_y.height, "old": mk_o.height}
        if fetch and mk.height:
            trades = fetch_trades(s, mk.select("ticker", pl.col("t").alias("t_from"), pl.col("end").alias("t_to")),
                                  name=NAME, workers=workers)
            if trades.height:
                out["maker"] = evaluate_maker(mk.filter(pl.col("group") == "young"), trades, HOLDOUT_FROM)
                variants = []
                for group in ("young", "old"):
                    g = mk.filter(pl.col("group") == group)
                    for label, rows in (("all", g), ("gap_ge_2c", g.filter((pl.col("p") - pl.col("mid")).abs() >= 0.02)),
                                        ("gap_ge_5c", g.filter((pl.col("p") - pl.col("mid")).abs() >= 0.05))):
                        for margin in (0.0, 0.01, 0.02, 0.03):
                            f = maker_fills(rows, trades, margin) if rows.height else pl.DataFrame()
                            if not f.height:
                                continue
                            f = f.with_columns(pl.when(pl.col("t") >= HOLDOUT_FROM).then(pl.lit("holdout"))
                                               .otherwise(pl.lit("dev")).alias("split"))
                            for (split, side), part in f.group_by("split", "side"):
                                filled = part.filter("filled")
                                row = {"group": group, "rows": label, "margin": margin, "split": split, "side": side,
                                       "orders": part.height, "fills": filled.height}
                                if filled.height:
                                    mu, tt, k = clustered_mean(filled["pnl"].to_numpy(), filled["event"].to_numpy())
                                    row.update({"events": k, "pnl_per_fill_c": round(mu * 100, 3), "t": _t(tt, k),
                                                "pnl_per_order_c": round(float(part["pnl"].sum()) / part.height * 100, 3)})
                                variants.append(row)
                out["maker_variants"] = (pl.DataFrame(variants).sort("group", "rows", "side", "margin", "split")
                                         if variants else pl.DataFrame())
    out["q"] = qa
    out["ladder_rows"] = lad
    return out


def write_report(s, out: dict[str, Any], notes: str = "") -> Path:
    """Markdown + Parquet report under ``data/reports/research-fundamental-series_age-*``, and the
    October prediction rows in ``data/cache/fundamental_series_age_holdout_preds.parquet``."""
    from albert.reports import Report

    rep = Report(s.reports_dir, f"research-fundamental-{NAME}")
    if notes:
        rep.text("Notes", notes)
    rep.json("Run", {"sample_rows": out.get("sample_rows"), "quoted_rows": out.get("quoted_rows"),
                     "calib_rows": out.get("calib_rows"),
                     "decisions_age_ambiguous_dropped": out.get("decisions_age_ambiguous_dropped"),
                     "series_launch_table_only": out.get("series_launch_table_only"),
                     "rule_chosen_on_dev": out.get("rule_chosen"), "maker_rows": out.get("maker_rows")})
    rep.table("Series launch month (series in the universe)", out["launch_summary"], save_as="launch_summary")
    rep.table("Quote coverage of sampled decisions", out["coverage"])
    rep.table("Calibration by series age, spread ≤ 10c (y − mid, cents; t by event / by series)", out["calib_bucket"],
              save_as="calib_bucket")
    rep.table("Calibration by series age, any spread", out["calib_bucket_allspreads"])
    rep.table("Calibration by age × price band", out["calib_bucket_band"], save_as="calib_bucket_band", max_rows=80)
    rep.table("logit y = a + b·logit(mid) by age bucket (t of a and of b − 1)", out["fits_bucket"], save_as="fits_bucket")
    rep.table("Calibration by period × age (spread ≤ 10c): Aug 15–Sep 6 / Sep 6–30 / October", out["calib_period"])
    rep.table("Longshots (mid < 20c) by period × age", out["calib_period_longshots"])
    rep.table("y − mid by spread band, young (< 16 w) vs old: the wide-book artefact", out["calib_spread_band"],
              save_as="calib_spread_band")
    rep.table("Longshots (mid < 20c, spread ≤ 10c) by category, young vs old", out["calib_cat_longshots"], max_rows=40)
    for split, tab in out.get("pooled", {}).items():
        rep.table(f"Pooled logistic, {split}: category intercepts + young-bucket shifts (vs ≥16 w)", tab)
    rep.table("Within-series: (y − mid) on age bucket + month, series fixed effects (cents)", out["within_series"],
              save_as="within_series", max_rows=60)
    rep.table("Calibration by category × age", out["calib_cat_bucket"], save_as="calib_cat_bucket", max_rows=80)
    rep.table("logit fits by category × age", out["fits_cat_bucket"], save_as="fits_cat_bucket", max_rows=60)
    rep.table("Calibration by event (ladder) age", out["calib_eventage"])
    rep.table("Calibration by event age, series younger than 16 weeks", out["calib_eventage_young"])
    rep.table("Ladder sums (mutually exclusive, complete, every leg spread ≤ 10c): Σ mid − 1 by series age",
              out["ladders"], save_as="ladders")
    rep.table("Ladder sums, any leg spread", out["ladders_all"])
    rep.table("Ladder sums by event age", out["ladders_eventage"])
    rep.table("Ladder sums by category × age", out["ladders_cat"], max_rows=60)
    rep.table("Sell-YES taker rules (bid − y − fee, cents per contract)", out["rule_grid"], save_as="rule_grid", max_rows=40)
    fe = out.get("fair_eval")
    if fe:
        rep.table("Fair value (pooled dev fit) vs mid: scores", pl.DataFrame(fe["scores"]))
        if fe.get("blend"):
            rep.json("Blend logit P = a + b·logit(mid) + c·logit(fair), fitted on dev", fe["blend"])
        rep.table("Fair-value taker (harness, θ chosen on dev)", pl.DataFrame(fe["taker"], infer_schema_length=None))
        rep.json("θ chosen on dev", {"theta": fe.get("theta_chosen_on_dev")})
    mk = out.get("maker")
    if mk:
        rep.json("Maker margin chosen on dev", {"margin": mk.get("margin_chosen_on_dev")})
        rep.table("Maker around the fair value, trade-through fills (harness)", pl.DataFrame(mk["maker"], infer_schema_length=None))
    if isinstance(out.get("maker_variants"), pl.DataFrame) and out["maker_variants"].height:
        rep.table("Maker variants by side and |fair − mid| gate", out["maker_variants"], save_as="maker_variants", max_rows=80)
    if isinstance(out.get("holdout_preds"), pl.DataFrame):
        path = s.data_dir / "cache" / f"fundamental_{NAME}_holdout_preds.parquet"
        _atomic_write(out["holdout_preds"], path)
    return rep.save()
