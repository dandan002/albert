"""Cross-section of the favourite-longshot bias on Kalshi (workstream ``xs`` of ``paper_flb/``).

Question: how large is the FLB, and how does it vary by category, horizon, contract structure,
liquidity and calendar time; how much survives fees and the 1c tick?

Pipeline (``python -m albert.research.flb_xs run <step>``; outputs under ``data/research/flb_xs/``):

* ``scan``: the sampling frame. Every settled market of every in-scope (non-financial) series from
  ``/historical/markets?series_ticker=`` (newest first; sports series stop at ``SPORTS_FROM``),
  compact rows only. Markets settled after the historical cutoff come from the shared
  ``kalshi_markets`` table (2026-08-15..10-31).
* ``events``: ``/events?series_ticker=`` for every series with multi-leg events in the sample
  (the ``mutually_exclusive`` flag).
* ``sample``: scheduled time ``T`` per market (see ``scheduled_time``), structure, group; a
  stratified random sample of *events* by group x month x structure (all legs of a sampled event).
* ``candles``: hourly candles of each sampled market over ``[T - 37 d, T + 2.5 h]`` (one request per
  market on the historical endpoint; live batch endpoint after the cutoff; shared tables reused).
* ``panel``: one row per market x lead (``LEADS_H`` before ``T``; sports in-play leads are negative),
  quote = last candle that ended at or before the lead time. Writes ``panel.parquet``,
  ``panel_schema.json`` and ``PANEL_READY``.
* ``analyze``: calibration, FLB measures, term structure, structure, liquidity, calendar time,
  limits to arbitrage, robustness -> ``summary.json``, ``tables/*.parquet``, ``verdict.md``.

No-look-ahead rules: leads are measured from the *scheduled* time (sports: scheduled start; daily
weather: start of the measurement day; deadline contracts: the deadline parsed from the rules;
otherwise the scheduled close); quotes come from candles that ended at or before the lead time;
an observation needs the market open at the lead time and still open 15 min later (a market that
closes within minutes of a quote has usually just learned its outcome). ``last_price`` is never
used as a price.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import threading
import time
import warnings
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from albert.research.flb_common import (
    BUCKET_LABELS,
    FAVOURITE,
    GROUPS,
    LONGSHOT,
    bucket_expr,
    clustered,
    coarse_category,
    fee_pc,
)

OUT = Path("data/research/flb_xs")
ET = ZoneInfo("America/New_York")
SPORTS_FROM = datetime(2024, 7, 1, tzinfo=UTC)      # sports series are scanned back to here
MAX_PAGES = {"Sports": 60, "default": 200}
CUTOFF_FALLBACK = datetime(2026, 9, 11, tzinfo=UTC)
TABLE_END = datetime(2026, 10, 31, 23, 59, 59, tzinfo=UTC)
MIN_RPS_INTERVAL = 0.34                             # <= 3 requests / s across threads
LEADS_H = (1, 6, 24, 72, 168, 720)                  # hours before the scheduled time
INPLAY_H = (-0.5, -1.0, -2.0)                       # sports: hours *after* the scheduled start
LEAD_LABEL = {1: "1h", 6: "6h", 24: "24h", 72: "3d", 168: "7d", 720: "30d",
              -0.5: "in-play 0.5h", -1.0: "in-play 1h", -2.0: "in-play 2h"}
WINDOW_BEFORE = timedelta(days=37)                  # candles from T - 37 d (7 d before the 30 d lead)
WINDOW_AFTER = timedelta(hours=2, minutes=30)       # ... to T + 2.5 h (in-play leads)
CLOSE_GUARD = timedelta(minutes=15)                 # market must stay open 15 min after a quote
STRUCTURES = ("binary", "multi", "ladder", "deadline")
MAX_LEGS = 60


def out_dir() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def _save(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


# ---- API client (shared cache, <= 3 req/s across threads) ----------------------------------------
def api(min_interval: float = MIN_RPS_INTERVAL):
    from albert.config import settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    s = settings()
    lock = threading.Lock()

    def nap(x: float) -> None:
        time.sleep(min(x, 20.0) * (0.75 + 0.5 * random.random()))

    client = CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=min_interval,
                          max_retries=40, sleep=nap)

    def throttle() -> None:                  # thread-safe version of CachedClient._throttle
        with lock:
            wait = client.min_interval - (time.monotonic() - client._last_request)
            if wait > 0:
                time.sleep(wait)
            client._last_request = time.monotonic()

    client._throttle = throttle               # type: ignore[method-assign]
    return k.Kalshi(client)


# ---- frame rows --------------------------------------------------------------------------------
FRAME_SCHEMA: dict[str, Any] = {
    "ticker": pl.String, "event_ticker": pl.String, "series_ticker": pl.String,
    "title": pl.String, "yes_sub_title": pl.String, "market_type": pl.String,
    "open_time": pl.Datetime("us", "UTC"), "close_time": pl.Datetime("us", "UTC"),
    "occurrence_time": pl.Datetime("us", "UTC"), "expected_expiration_time": pl.Datetime("us", "UTC"),
    "can_close_early": pl.Boolean, "result": pl.String, "volume": pl.Float64,
    "open_interest": pl.Float64, "strike_type": pl.String, "floor_strike": pl.Float64,
    "cap_strike": pl.Float64, "dl_kind": pl.String, "dl_T": pl.Datetime("us", "UTC"),
    "source": pl.String,
}


def deadline_info(category: str | None, title: str | None, rules: str | None) -> tuple[str | None, datetime | None]:
    """('event'|'mention'|'elimination'|'count', deadline T) for a "by date T" contract, else (None, None).
    Same text rules as the hazard study (``pm_hazard.classify`` / ``parse_window``)."""
    from albert.research.pm_hazard import DEADLINE_HINT, HAZARD_TYPES, classify, parse_window

    if not rules or not DEADLINE_HINT.search(rules):
        return None, None
    kind = classify(category, title, rules)
    if kind not in HAZARD_TYPES:
        return None, None
    try:
        _, T, _ = parse_window(rules)
    except (ValueError, KeyError):
        return None, None
    return (kind, T) if T is not None else (None, None)


def frame_rows(payload: dict[str, Any], category: str | None, source: str) -> list[dict[str, Any]]:
    from albert.venues import kalshi as k

    rows = []
    for m in payload.get("markets", []):
        if m.get("mve_collection_ticker") or m.get("mve_selected_legs"):
            continue
        res = m.get("result") or None
        if res not in ("yes", "no"):
            continue
        kind, T = deadline_info(category, m.get("title"), m.get("rules_primary"))
        rows.append({
            "ticker": m["ticker"], "event_ticker": m.get("event_ticker"),
            "series_ticker": None, "title": m.get("title"), "yes_sub_title": m.get("yes_sub_title"),
            "market_type": m.get("market_type"),
            "open_time": k.parse_ts(m.get("open_time")), "close_time": k.parse_ts(m.get("close_time")),
            "occurrence_time": k.parse_ts(m.get("occurrence_datetime")),
            "expected_expiration_time": k.parse_ts(m.get("expected_expiration_time")),
            "can_close_early": m.get("can_close_early"), "result": res,
            "volume": k.num(k._field(m, "volume")), "open_interest": k.num(k._field(m, "open_interest")),
            "strike_type": m.get("strike_type"), "floor_strike": k.num(m.get("floor_strike")),
            "cap_strike": k.num(m.get("cap_strike")), "dl_kind": kind, "dl_T": T, "source": source,
        })
    return rows


def scan_plan() -> pl.DataFrame:
    """In-scope series (coarse group not None) with their group; non-sports first. Multivariate
    (parlay) collections ``KXMVE*`` are skipped: their markets are combos, outside the study."""
    from albert.config import settings

    ser = pl.read_parquet(settings().tables_dir / "kalshi_series.parquet")
    grp = [coarse_category(c, s, t) for s, c, t in ser.select("series_ticker", "category", "title").iter_rows()]
    plan = ser.with_columns(pl.Series("group", grp, dtype=pl.String)).filter(
        pl.col("group").is_not_null() & ~pl.col("series_ticker").str.starts_with("KXMVE"))   # parlay collections
    return plan.sort(pl.col("group") == "Sports", "series_ticker").select(
        "series_ticker", "category", "title", "frequency", "fee_type", "fee_multiplier", "group")


def scan(workers: int = 6) -> None:
    """Historical frame: every settled market of every in-scope series. Checkpointed in parts."""
    from concurrent.futures import ThreadPoolExecutor

    d = out_dir() / "frame"
    d.mkdir(parents=True, exist_ok=True)
    plan = scan_plan()
    _save(plan, out_dir() / "scan_plan.parquet")
    done_path = d / "done.parquet"
    done = set(pl.read_parquet(done_path)["series_ticker"]) if done_path.exists() else set()
    todo = [r for r in plan.iter_rows(named=True) if r["series_ticker"] not in done]
    print(f"scan: {len(todo)} series to scan ({plan.height} planned, {len(done)} done)", flush=True)
    k = api()

    def one(r: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None, int]:
        sr, sports = r["series_ticker"], r["group"] == "Sports"
        rows: list[dict[str, Any]] = []
        pages = 0
        cap = MAX_PAGES["Sports" if sports else "default"]
        try:
            # same params as pm_hazard.scan, so its cached pages are reused
            for page in k._cursor_pages("/historical/markets", {"series_ticker": sr, "limit": 1000}, "markets"):
                pages += 1
                got = frame_rows(page.payload, r["category"], "historical")
                for x in got:
                    x["series_ticker"] = sr
                rows.extend(got)
                closes = [x["close_time"] for x in got if x["close_time"] is not None]
                if pages >= cap or (sports and closes and min(closes) < SPORTS_FROM):
                    break
        except RuntimeError as error:
            print(f"skip {sr}: {str(error)[:160]}", flush=True)
            return sr, None, pages
        return sr, rows, pages

    buf: list[dict[str, Any]] = []
    fin: list[str] = []
    part = len(list(d.glob("part-*.parquet")))
    n_req = 0
    t0 = time.monotonic()

    def flush() -> None:
        nonlocal buf, fin, part
        if buf:
            _save(pl.DataFrame(buf, schema=FRAME_SCHEMA), d / f"part-{part:05d}.parquet")
            part += 1
        done.update(fin)
        _save(pl.DataFrame({"series_ticker": sorted(done)}, schema={"series_ticker": pl.String}), done_path)
        buf, fin = [], []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (sr, rows, pages) in enumerate(pool.map(one, todo), 1):
            n_req += pages
            if rows is None:
                continue
            buf.extend(rows)
            fin.append(sr)
            if len(buf) >= 100_000 or i % 500 == 0 or i == len(todo):
                flush()
                el = time.monotonic() - t0
                print(f"  series {i}/{len(todo)}  pages {n_req}  ({n_req / max(el, 1):.2f} pages/s)", flush=True)
    flush()
    print("scan done", flush=True)


# ---- scheduled time, structure ------------------------------------------------------------------
_MONTHS = {m: i + 1 for i, m in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"))}
_TICKER_DATE = re.compile(r"(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})")
_STRIKE_SUFFIX = re.compile(r"(\d+(?:\.\d+)?)$")
NUMERIC_STRIKES = frozenset({"greater", "greater_or_equal", "less", "less_or_equal", "between"})
UTC_PLUS_14 = timedelta(hours=14)


def ticker_date(event_ticker: str | None) -> datetime | None:
    """Calendar date encoded in an event ticker (``KXVOTESWEDEN-26SEP13SDP`` -> 2026-09-13), as a
    naive midnight (time zone applied by the caller), or None."""
    if not event_ticker or "-" not in event_ticker:
        return None
    m = _TICKER_DATE.search(event_ticker.split("-", 1)[1])
    if not m:
        return None
    yy, mon, dd = int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3))
    if not 21 <= yy <= 27:
        return None
    try:
        return datetime(2000 + yy, mon, dd)
    except ValueError:
        return None


def leg_strike(ticker: str, event_ticker: str | None) -> tuple[str, float | None]:
    """(ladder base, numeric strike) from the leg suffix: ``...-T86.99`` -> ('', 86.99),
    ``...-CAR8`` -> ('CAR', 8), ``...-PHITMAXEY0-20`` -> ('PHITMAXEY0-', 20), ``...-DJT`` -> ('DJT', None)."""
    et = event_ticker or ""
    suffix = ticker[len(et) + 1:] if et and ticker.startswith(et + "-") else ticker.rsplit("-", 1)[-1]
    m = _STRIKE_SUFFIX.search(suffix)
    if not m:
        return suffix, None
    base = suffix[:m.start()]
    if base in ("T", "B"):
        base = ""
    return base, float(m.group(1))


def scheduled_time(group: str, frequency: str | None, event_ticker: str | None, sports_start: datetime | None,
                   t_close_sched: datetime | None, dl_T: datetime | None) -> tuple[datetime | None, str]:
    """The scheduled time leads are measured from, and its source.

    * ``deadline``: the deadline parsed from the rules ("by date T" contracts).
    * ``start``: sports with a start time in the ticker (pre-game leads > 0, in-play leads < 0).
    * ``date``: the event date in the ticker. Sports: midnight at UTC+14 (no listed date has
      started anywhere before it); daily weather: the earlier of midnight ET and the scheduled
      close minus 24 h (the local start of the measurement day abroad); otherwise midnight ET,
      unless the scheduled close comes first.
    * ``close``: the scheduled close (expected expiration minus the series' modal offset).
    """
    if dl_T is not None:
        return dl_T, "deadline"
    if group == "Sports" and sports_start is not None:
        return sports_start, "start"
    d = ticker_date(event_ticker)
    if d is not None:
        if group == "Sports":
            t = (d - UTC_PLUS_14).replace(tzinfo=UTC)
        else:
            t = d.replace(tzinfo=ET).astimezone(UTC)
            if group == "Weather" and frequency == "daily" and t_close_sched is not None:
                t = min(t, t_close_sched - timedelta(hours=24))
        if t_close_sched is None or t <= t_close_sched or group == "Sports":
            return t, "date"
    if t_close_sched is not None:
        return t_close_sched, "close"
    return None, "none"


def leg_strike_exprs() -> list[pl.Expr]:
    """Vectorised ``leg_strike`` plus the API strike fields: columns ``ladder_base`` and ``strike``."""
    et, tk = pl.col("event_ticker"), pl.col("ticker")
    suffix = (pl.when(et.is_not_null() & tk.str.starts_with(et + "-"))
              .then(tk.str.slice(et.str.len_chars() + 1)).otherwise(tk.str.split("-").list.last()))
    num = suffix.str.extract(r"(\d+(?:\.\d+)?)$", 1).cast(pl.Float64)
    base = suffix.str.replace(r"(\d+(?:\.\d+)?)$", "")
    base = pl.when(base.is_in(["T", "B"])).then(pl.lit("")).otherwise(base)
    fs, cs, st = pl.col("floor_strike"), pl.col("cap_strike"), pl.col("strike_type")
    api = pl.when(fs.is_not_null() & cs.is_not_null()).then((fs + cs) / 2).otherwise(pl.coalesce(fs, cs))
    strike = (pl.when(st.is_in(list(NUMERIC_STRIKES))).then(api)
              .when(st.is_not_null()).then(pl.lit(None, pl.Float64)).otherwise(num))
    return [base.alias("ladder_base"), strike.alias("strike")]


def scheduled_time_exprs() -> list[pl.Expr]:
    """Vectorised ``scheduled_time`` on frame columns (needs ``sports_start``): ``T`` and ``sched_src``."""
    rest = pl.col("event_ticker").str.replace(r"^[^-]*-", "")
    g = rest.str.extract_groups(r"(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})")
    yy = g.struct.field("1").cast(pl.Int32, strict=False)
    mon = g.struct.field("2").replace_strict(_MONTHS, default=None, return_dtype=pl.Int32)
    dd = g.struct.field("3").cast(pl.Int32, strict=False)
    datestr = pl.format("20{}-{}-{}", yy.cast(pl.String), mon.cast(pl.String).str.zfill(2), dd.cast(pl.String).str.zfill(2))
    day = (pl.when(pl.col("event_ticker").str.contains("-") & (yy >= 21) & (yy <= 27))
           .then(datestr.str.to_date("%Y-%m-%d", strict=False)).otherwise(None))
    naive = day.cast(pl.Datetime("us"))
    t_et = naive.dt.replace_time_zone("America/New_York").dt.convert_time_zone("UTC")
    t_sport = (naive - UTC_PLUS_14).dt.replace_time_zone("UTC")
    tcs = pl.col("t_close_sched")
    weather_daily = (pl.col("group") == "Weather") & (pl.col("frequency") == "daily") & tcs.is_not_null()
    t_date = (pl.when(pl.col("group") == "Sports").then(t_sport)
              .when(weather_daily).then(pl.min_horizontal(t_et, tcs - timedelta(hours=24))).otherwise(t_et))
    date_ok = day.is_not_null() & ((pl.col("group") == "Sports") | tcs.is_null() | (t_date <= tcs))
    sports_start = (pl.col("group") == "Sports") & pl.col("sports_start").is_not_null()
    T = (pl.when(pl.col("dl_T").is_not_null()).then(pl.col("dl_T"))
         .when(sports_start).then(pl.col("sports_start"))
         .when(date_ok).then(t_date).otherwise(tcs))
    src = (pl.when(pl.col("dl_T").is_not_null()).then(pl.lit("deadline"))
           .when(sports_start).then(pl.lit("start"))
           .when(date_ok).then(pl.lit("date"))
           .when(tcs.is_not_null()).then(pl.lit("close")).otherwise(pl.lit("none")))
    return [T.dt.cast_time_unit("us").alias("T"), src.alias("sched_src")]


def series_offsets(frame: pl.DataFrame) -> pl.DataFrame:
    """Scheduled close per series: offset = expected_expiration - close, modal (>= half the markets)
    else median, over NO-settled markets (they run to the scheduled close); group median fallback."""
    x = frame.filter((pl.col("result") == "no") & pl.col("expected_expiration_time").is_not_null()).with_columns(
        ((pl.col("expected_expiration_time") - pl.col("close_time")).dt.total_minutes()).alias("off"))
    per = x.group_by("series_ticker").agg(
        pl.col("off").mode().first().alias("mode"), pl.col("off").median().alias("med"), pl.len().alias("n_no"),
        pl.col("group").first())
    share = x.join(per.select("series_ticker", "mode"), on="series_ticker").group_by("series_ticker").agg(
        (pl.col("off") == pl.col("mode")).mean().alias("mode_share"))
    per = per.join(share, on="series_ticker").with_columns(
        pl.when(pl.col("mode_share") >= 0.5).then(pl.col("mode").cast(pl.Float64)).otherwise(pl.col("med")).alias("offset_min"))
    gmed = per.group_by("group").agg(pl.col("offset_min").median().alias("g_off"))
    allser = frame.select("series_ticker", "group").unique("series_ticker")
    return (allser.join(per.select("series_ticker", "offset_min"), on="series_ticker", how="left")
            .join(gmed, on="group", how="left")
            .select("series_ticker", pl.coalesce("offset_min", "g_off").fill_null(0.0).alias("offset_min")))


FRAME_COLS = ["ticker", "event_ticker", "series_ticker", "open_time", "close_time", "expected_expiration_time",
              "result", "volume", "open_interest", "strike_type", "floor_strike", "cap_strike", "dl_kind", "dl_T", "source"]


def load_frame(group: str | None = None, chunk: tuple[int, int] = (0, 1)) -> pl.DataFrame:
    """Historical scan (closed before the cutoff) + shared table (closed from the cutoff to 10-31),
    binary markets only, with group, series fields and the deadline classification. ``group``
    restricts to one coarse group and ``chunk=(i, k)`` to the i-th of k series-hash chunks (memory)."""
    from albert.config import settings
    from albert.research.pm_hazard import DEADLINE_HINT

    s = settings()
    plan = pl.read_parquet(out_dir() / "scan_plan.parquet")
    if group is not None:
        plan = plan.filter(pl.col("group") == group)
    if chunk[1] > 1:
        plan = plan.filter(pl.col("series_ticker").hash(seed=1) % chunk[1] == chunk[0])
    ser = plan["series_ticker"].implode()
    cutoff = CUTOFF_FALLBACK
    hist = (pl.scan_parquet(str(out_dir() / "frame" / "part-*.parquet"))
            .filter((pl.col("close_time") < cutoff) & pl.col("series_ticker").is_in(ser) & (pl.col("market_type") == "binary"))
            .select(FRAME_COLS).collect())
    hint = pl.col("rules_primary").str.contains("(?i)" + DEADLINE_HINT.pattern)
    tq = pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet").filter(
        (pl.col("close_time") >= cutoff) & pl.col("result").is_in(["yes", "no"])
        & pl.col("series_ticker").is_in(ser) & (pl.col("market_type") == "binary"))
    # two passes so that rules text is only held for the few rows with a deadline phrase
    tab = tq.select("ticker", "event_ticker", "series_ticker", "open_time", "close_time", "expected_expiration_time",
                    "result", "volume", "open_interest", "category", hint.alias("hint")).collect(engine="streaming")
    txt = (tq.filter(pl.col("ticker").is_in(tab.filter(pl.col("hint"))["ticker"].implode()))
           .select("ticker", pl.col("rules_primary").alias("rules"), pl.col("title").alias("title_h"))
           .collect(engine="streaming"))
    cats = dict(zip(plan["series_ticker"], plan["category"], strict=True))
    dl = [(tk, *deadline_info(cats.get(sr), t, r)) for tk, sr, t, r in
          txt.join(tab.select("ticker", "series_ticker"), on="ticker").select("ticker", "series_ticker", "title_h", "rules").iter_rows()]
    dl_df = pl.DataFrame(dl, schema={"ticker": pl.String, "dl_kind": pl.String, "dl_T": pl.Datetime("us", "UTC")}, orient="row")
    tab = tab.drop("hint", "category").join(dl_df, on="ticker", how="left").with_columns(
        pl.lit(None, pl.String).alias("strike_type"), pl.lit(None, pl.Float64).alias("floor_strike"),
        pl.lit(None, pl.Float64).alias("cap_strike"), pl.lit("table").alias("source"))
    fr = pl.concat([hist, tab.select(FRAME_COLS)]).unique("ticker", keep="first")
    return fr.join(plan.select("series_ticker", "group", "category", "frequency", "fee_type", "fee_multiplier"),
                   on="series_ticker", how="inner")


def classify_frame(fr: pl.DataFrame) -> pl.DataFrame:
    """Scheduled time T, structure proxy, ladder keys and sampling cells for every frame market."""
    from albert.venues.kalshi import ticker_start

    off = series_offsets(fr)
    fr = fr.join(off, on="series_ticker", how="left").with_columns(
        pl.when(pl.col("expected_expiration_time").is_not_null())
        .then(pl.col("expected_expiration_time") - pl.duration(minutes=pl.col("offset_min").round(0).cast(pl.Int64)))
        .when(pl.col("result") == "no").then(pl.col("close_time")).otherwise(None).alias("t_close_sched"))
    # ladder legs: numeric strike in the leg suffix (historical rows: the API strike fields)
    fr = fr.with_columns(*leg_strike_exprs())
    fr = fr.with_columns(pl.concat_str("event_ticker", pl.lit("|"), "ladder_base").alias("ladder_key"))
    lad = fr.filter(pl.col("strike").is_not_null()).group_by("ladder_key").agg(
        pl.col("strike").n_unique().alias("n_strikes"))
    fr = fr.join(lad, on="ladder_key", how="left").with_columns(
        (pl.col("n_strikes").fill_null(0) >= 3).alias("is_ladder"))
    ev = fr.group_by("event_ticker").agg(pl.len().alias("n_legs"), pl.col("volume").sum().alias("ev_volume"),
                                         pl.col("is_ladder").any().alias("ev_ladder"))
    fr = fr.join(ev, on="event_ticker", how="left")
    # scheduled time
    sev = fr.filter(pl.col("group") == "Sports")["event_ticker"].unique()
    starts = pl.DataFrame({"event_ticker": sev, "sports_start": [ticker_start(e) for e in sev.to_list()]},
                          schema={"event_ticker": pl.String, "sports_start": pl.Datetime("us", "UTC")})
    fr = fr.join(starts, on="event_ticker", how="left").with_columns(*scheduled_time_exprs()).drop("sports_start")
    fr = fr.with_columns(
        pl.when(pl.col("dl_kind").is_not_null()).then(pl.lit("deadline"))
        .when(pl.col("is_ladder")).then(pl.lit("ladder"))
        .when(pl.col("n_legs") >= 3).then(pl.lit("multi"))      # provisional: needs the ME flag
        .otherwise(pl.lit("binary")).alias("structure0"),
        pl.col("T").dt.strftime("%Y-%m").alias("month"),
        (pl.col("result") == "yes").cast(pl.Int8).alias("y"))
    return fr


def eligible(fr: pl.DataFrame) -> pl.DataFrame:
    """Markets whose scheduled time is known, in the sample window, with some pre-T life and trading."""
    return fr.filter(pl.col("T").is_not_null() & (pl.col("T") <= TABLE_END)
                     & (pl.col("open_time") < pl.col("T") - timedelta(hours=1))
                     & (pl.col("ev_volume") >= 10)
                     & ((pl.col("group") != "Sports") | (pl.col("T") >= SPORTS_FROM)))


def water_fill(avail: dict[str, float], total: float) -> dict[str, float]:
    """Split ``total`` equally across keys, capping each at its availability and handing the unused
    quota to the others (structures that are rare in a cell do not waste the cell's budget)."""
    quota = dict.fromkeys(avail, 0.0)
    active, rem = set(avail), float(total)
    while active and rem > 1e-9:
        share = rem / len(active)
        small = {k for k in active if avail[k] - quota[k] <= share}
        if not small:
            for k in active:
                quota[k] += share
            break
        for k in small:
            rem -= avail[k] - quota[k]
            quota[k] = avail[k]
        active -= small
    return quota


def draw_sample(el: pl.DataFrame, target: dict[str, int], seed: int = 20261111) -> pl.DataFrame:
    """Stratified random sample of events: cells group x month x structure0; within a cell, events in
    random order until the cell's market quota is met. A group's budget is spread evenly over its
    months and split across structures by water-filling. All legs of a drawn event are kept, except
    that multi-outcome candidates keep a random MULTI_MAX legs and other events whole ladders up to
    MAX_LEGS legs. Returns markets with the event inclusion probability."""
    rng = np.random.default_rng(seed)
    # the sampling cell is an event attribute: its group, the month of its earliest scheduled time and
    # its most common leg structure (legs of one event can differ in T and structure)
    cell = (el.sort("T").group_by("event_ticker", maintain_order=True)
            .agg(pl.col("group").first().alias("c_group"), pl.col("month").first().alias("c_month"),
                 pl.col("structure0").mode().sort().first().alias("c_struct"), pl.len().alias("n")))
    el = el.join(cell.select("event_ticker", "c_group", "c_month", "c_struct"), on="event_ticker")
    evs = cell.rename({"c_group": "group", "c_month": "month", "c_struct": "structure0"})
    cell_n = evs.group_by("group", "month", "structure0").agg(pl.len().alias("pop_events"),
                                                             pl.col("n").sum().alias("pop_markets"))
    evs = evs.with_columns(pl.when(pl.col("structure0") == "multi").then(pl.col("n").clip(upper_bound=MULTI_MAX))
                           .otherwise(pl.col("n").clip(upper_bound=MAX_LEGS)).alias("n_cap"))
    avail = evs.group_by("group", "month", "structure0").agg(pl.col("n_cap").sum().alias("avail"))
    months = avail.group_by("group").agg(pl.col("month").n_unique().alias("m"))
    per_month = {g: max(MIN_PER_MONTH, target.get(g, target["default"]) / m) for g, m in months.iter_rows()}
    quota: dict[tuple[str, str, str], float] = {}
    for (g, mo), sub in avail.group_by(["group", "month"]):
        quota.update({(g, mo, k): v for k, v in water_fill(dict(zip(sub["structure0"], sub["avail"], strict=True)),
                                                              per_month[g]).items()})
    picked: list[tuple[str, float]] = []
    for (g, mo, st), sub in evs.group_by(["group", "month", "structure0"]):
        tgt = quota[(g, mo, st)]
        order = rng.permutation(sub.height)
        names, sizes = sub["event_ticker"].to_list(), sub["n_cap"].to_list()
        tot, chosen = 0, []
        for i in order:
            if tot >= tgt:
                break
            chosen.append(names[i])
            tot += sizes[i]
        pi = len(chosen) / sub.height
        picked.extend((e, pi) for e in chosen)
    pk = pl.DataFrame(picked, schema={"event_ticker": pl.String, "p_event": pl.Float64}, orient="row")
    smp = el.join(pk, on="event_ticker", how="inner")
    # cap very large events: keep whole ladders (random order) up to MAX_LEGS legs; ME candidate lists kept whole
    # multi-outcome candidates with more than MULTI_MAX legs: a random MULTI_MAX legs (incomplete events)
    bigm = smp.filter((pl.col("n_legs") > MULTI_MAX) & (pl.col("c_struct") == "multi"))
    if bigm.height:
        drop = bigm.with_columns(pl.int_range(pl.len()).shuffle(seed=seed).over("event_ticker").alias("_r")).filter(
            pl.col("_r") >= MULTI_MAX)["ticker"]
        smp = smp.filter(~pl.col("ticker").is_in(drop.implode()))
    big = smp.filter((pl.col("n_legs") > MAX_LEGS) & (pl.col("c_struct") != "multi"))
    if big.height:
        keep = []
        for _, g in big.group_by("event_ticker"):
            keys = g["ladder_key"].unique().to_list()
            rng.shuffle(keys)
            sizes = dict(g.group_by("ladder_key").len().iter_rows())
            tot = 0
            for kk in keys:
                if tot and tot + sizes[kk] > MAX_LEGS:
                    continue
                keep.append(kk)
                tot += sizes[kk]
        smp = smp.filter((pl.col("n_legs") <= MAX_LEGS) | (pl.col("c_struct") == "multi")
                         | pl.col("ladder_key").is_in(keep))
    cell_n = cell_n.rename({"group": "c_group", "month": "c_month", "structure0": "c_struct"})
    return smp.join(cell_n, on=["c_group", "c_month", "c_struct"], how="left")


# total market budget per group, spread evenly over the group's active months (at least MIN_PER_MONTH)
SAMPLE_TARGET = {"Sports": 7000, "Weather": 3500, "Politics": 4500, "Entertainment": 3500, "Mentions": 2000,
                 "Science & Tech": 2500, "default": 600}
MIN_PER_MONTH = 40
GROUP_CHUNKS = {"Sports": 6}   # Sports is classified and sampled in 6 series-hash chunks (memory)
MULTI_MAX = 30


def sample() -> None:
    """Classify the frame and draw the sample, one group at a time (keeps memory low)."""
    import gc

    d = out_dir()
    keep_cols = ["ticker", "event_ticker", "series_ticker", "group", "category", "frequency",
                 "open_time", "close_time", "expected_expiration_time", "t_close_sched", "T", "sched_src", "month",
                 "result", "y", "volume", "open_interest", "fee_type", "fee_multiplier", "strike_type", "strike",
                 "ladder_key", "is_ladder", "n_legs", "ev_volume", "dl_kind", "structure0", "source"]
    (d / "frame_classified").mkdir(exist_ok=True)
    cells, smps = [], []
    jobs = [(grp, i, GROUP_CHUNKS.get(grp, 1)) for grp in GROUPS for i in range(GROUP_CHUNKS.get(grp, 1))]
    for g, i, k in jobs:
        fr = classify_frame(load_frame(g, (i, k)))
        _save(fr.select(keep_cols), d / "frame_classified" / f"{re.sub(r'[^a-z]', '', g.lower())}_{i}.parquet")
        el = eligible(fr)
        cells.append(fr.group_by("group", "month").agg(pl.len().alias("frame_markets"), pl.col("volume").sum().alias("frame_volume"),
                                                       pl.col("event_ticker").n_unique().alias("frame_events"))
                     .join(el.group_by("group", "month").agg(pl.len().alias("eligible_markets"),
                                                             pl.col("event_ticker").n_unique().alias("eligible_events")),
                           on=["group", "month"], how="left"))
        if el.height:
            budget = {kk: v / k for kk, v in SAMPLE_TARGET.items()}
            smps.append(draw_sample(el, budget, seed=20261111 + i).unique("ticker").select(
                *keep_cols, "c_month", "c_struct", "p_event", "pop_events", "pop_markets"))
        print(f"{g}: frame {fr.height:,}, eligible {el.height:,}, sampled {smps[-1].height if el.height else 0:,}", flush=True)
        del fr, el
        gc.collect()
    cells_df = pl.concat(cells).group_by("group", "month").agg(pl.all().sum()).sort("group", "month")
    _save(cells_df, d / "frame_cells.parquet")
    smp = pl.concat(smps)
    _save(smp, d / "sample.parquet")
    print(f"sample {smp.height:,} markets in {smp['event_ticker'].n_unique():,} events", flush=True)
    print(smp.group_by("group").agg(pl.len(), pl.col("event_ticker").n_unique().alias("events"),
                                    pl.col("month").min().alias("first"), pl.col("month").max().alias("last")).sort("group"), flush=True)
    print(smp.group_by("c_struct").len(), smp.group_by("sched_src").len(), flush=True)


# ---- events (mutually_exclusive flag) -------------------------------------------------------------
def events(workers: int = 3) -> None:
    """``/events/{event_ticker}`` for every sampled event with >= 3 non-ladder legs (the structure needs
    the flag) and a random 1,500 two-leg events (head-to-head overround)."""
    from concurrent.futures import ThreadPoolExecutor

    from albert.venues import kalshi as k

    d = out_dir()
    smp = pl.read_parquet(d / "sample.parquet")
    base = smp.filter(~pl.col("is_ladder") & pl.col("dl_kind").is_null())
    multi = base.filter(pl.col("n_legs") >= 3)["event_ticker"].unique().sort().to_list()
    two = base.filter(pl.col("n_legs") == 2)["event_ticker"].unique().sort()
    two = two.sample(min(1500, two.len()), seed=7).sort().to_list() if two.len() else []
    todo = multi + two
    path = d / "events.parquet"
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=k.EVENT_SCHEMA)
    todo = [e for e in todo if e not in set(have["event_ticker"])]
    print(f"events: {len(todo)} to fetch ({have.height} cached)", flush=True)
    api_ = api()

    def one(e: str) -> dict[str, Any] | None:
        try:
            return k.event_row(api_.event(e).payload)
        except RuntimeError as error:
            print(f"skip {e}: {str(error)[:120]}", flush=True)
            return None

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(one, todo), 1):
            if r is not None and r.get("event_ticker"):
                rows.append(r)
            if i % 1000 == 0 or i == len(todo):
                have = pl.concat([have, pl.DataFrame(rows, schema=k.EVENT_SCHEMA)]).unique("event_ticker")
                rows = []
                _save(have, path)
                print(f"  events {i}/{len(todo)}", flush=True)
    if rows:
        _save(pl.concat([have, pl.DataFrame(rows, schema=k.EVENT_SCHEMA)]).unique("event_ticker"), path)


# ---- candles ------------------------------------------------------------------------------------
CANDLE_COLS = {"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "yes_bid_close": pl.Float64,
               "yes_ask_close": pl.Float64, "price_close": pl.Float64, "volume": pl.Float64,
               "open_interest": pl.Float64}


def windows(smp: pl.DataFrame) -> pl.DataFrame:
    """Candle window per sampled market: [max(open, T - 37 d), min(close, T + 2.5 h)]."""
    return smp.select("ticker", "event_ticker", "open_time", "close_time", "T").with_columns(
        pl.max_horizontal("open_time", pl.col("T") - WINDOW_BEFORE).alias("lo"),
        pl.min_horizontal("close_time", pl.col("T") + WINDOW_AFTER).alias("hi")).filter(pl.col("lo") < pl.col("hi"))


def _local_candles(w: pl.DataFrame) -> pl.DataFrame:
    """Candles already on disk that cover a market's whole window (shared table, hazard and smile
    studies). Coverage: first candle within 2 h of the window start (or of the open), last within 3 h
    of its end."""
    from albert.config import settings

    s = settings()
    srcs = [s.tables_dir / "kalshi_candles.parquet", Path("data/research/pm_hazard/candles.parquet"),
            Path("data/research/pm_smile/candles_fetched.parquet")]
    out = []
    got: set[str] = set()
    for src in srcs:
        if not src.exists():
            continue
        ww = w.filter(~pl.col("ticker").is_in(list(got)))
        c = (pl.scan_parquet(src).filter(pl.col("ticker").is_in(ww["ticker"].implode()))
             .select(list(CANDLE_COLS)).collect().join(ww.select("ticker", "lo", "hi"), on="ticker")
             .filter((pl.col("end_ts") >= pl.col("lo") - timedelta(days=30)) & (pl.col("end_ts") <= pl.col("hi"))))
        cov = c.group_by("ticker").agg(pl.col("end_ts").min().alias("a"), pl.col("end_ts").max().alias("b"),
                                       pl.col("lo").first(), pl.col("hi").first())
        ok = cov.filter((pl.col("a") <= pl.col("lo") + timedelta(hours=2)) & (pl.col("b") >= pl.col("hi") - timedelta(hours=3)))
        keep = set(ok["ticker"])
        if keep:
            out.append(c.filter(pl.col("ticker").is_in(list(keep))).select(list(CANDLE_COLS)))
            got |= keep
        print(f"  local {src.name}: {len(keep)} markets covered", flush=True)
    return pl.concat(out) if out else pl.DataFrame(schema=CANDLE_COLS)


def candles(workers: int = 6) -> None:
    """Hourly candles for every sampled market (local first, then one API request per market)."""
    from concurrent.futures import ThreadPoolExecutor

    from albert.venues import kalshi as k

    d = out_dir() / "candles"
    d.mkdir(parents=True, exist_ok=True)
    done_path = d / "done.parquet"
    done = set(pl.read_parquet(done_path)["ticker"]) if done_path.exists() else set()
    smp = pl.read_parquet(out_dir() / "sample.parquet")
    w = windows(smp).filter(~pl.col("ticker").is_in(list(done)))
    part = len(list(d.glob("part-*.parquet")))
    if w.height and not (d / "local.done").exists():
        loc = _local_candles(w)
        if loc.height:
            _save(loc, d / f"part-{part:05d}.parquet")
            part += 1
            done |= set(loc["ticker"].unique())
            _save(pl.DataFrame({"ticker": sorted(done)}, schema={"ticker": pl.String}), done_path)
        (d / "local.done").write_text("ok")
        w = w.filter(~pl.col("ticker").is_in(list(done)))
    todo = w.sort("lo", descending=True).to_dicts()
    print(f"candles: {len(todo)} markets to fetch ({len(done)} done)", flush=True)
    if not todo:
        return
    api_ = api()
    cutoff = CUTOFF_FALLBACK
    keep = list(CANDLE_COLS)

    def one(r: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None]:
        hist = r["close_time"] < cutoff
        page = None
        for attempt in (hist, not hist):          # settled on the other side of the cutoff: other endpoint
            try:
                page = api_.candles(r["ticker"], r["lo"] - timedelta(hours=1), r["hi"], 60, attempt)
                break
            except RuntimeError as error:
                msg = str(error)
                if "404" not in msg:
                    print(f"skip {r['ticker']}: {msg[:140]}", flush=True)
                    return r["ticker"], None
        if page is None:
            print(f"no candles {r['ticker']} (404 on both endpoints)", flush=True)
            return r["ticker"], []
        return r["ticker"], [{c: x[c] for c in keep} for x in k.candle_rows(page.payload, r["ticker"], 60)]

    rows: list[dict[str, Any]] = []
    fin: list[str] = []
    t0 = time.monotonic()

    def flush() -> None:
        nonlocal rows, fin, part
        if rows:
            _save(pl.DataFrame(rows, schema=CANDLE_COLS), d / f"part-{part:05d}.parquet")
            part += 1
        done.update(fin)
        _save(pl.DataFrame({"ticker": sorted(done)}, schema={"ticker": pl.String}), done_path)
        rows, fin = [], []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (tk, got) in enumerate(pool.map(one, todo), 1):
            if got is not None:
                rows.extend(got)
                fin.append(tk)
            if i % 1000 == 0 or i == len(todo):
                flush()
                el = time.monotonic() - t0
                print(f"  candles {i}/{len(todo)} ({i / max(el, 1):.2f}/s)", flush=True)
    flush()
    print("candles done", flush=True)


# ---- panel --------------------------------------------------------------------------------------
PANEL_DOC: dict[str, str] = {
    "ticker": "Kalshi market ticker",
    "event_ticker": "event ticker (cluster unit)",
    "series_ticker": "series ticker",
    "group": "coarse category (flb_common.coarse_category)",
    "category": "Kalshi series category",
    "structure": "binary | multi (mutually exclusive event, >=3 non-ladder legs) | ladder (strike-ladder leg) | deadline ('by date T')",
    "n_legs": "settled legs of the event in the frame",
    "mutually_exclusive": "event flag from /events (null if not fetched: ladders, deadlines, single-leg events)",
    "ladder_key": "event|ladder base: legs of one strike ladder share it",
    "strike": "numeric strike of a ladder leg (null otherwise)",
    "lead_h": "hours before the scheduled time (negative: sports in-play, hours after the scheduled start)",
    "phase": "pre | inplay",
    "ts": "observation time = scheduled_ts - lead_h (UTC)",
    "quote_ts": "end of the hourly candle the quote comes from (<= ts)",
    "quote_age_h": "ts - quote_ts in hours (Kalshi omits hours without activity; the last close stands)",
    "yes_bid": "best YES bid at quote_ts, dollars (0 = no bid)",
    "yes_ask": "best YES ask at quote_ts, dollars (1 = no ask)",
    "mid": "(yes_bid + yes_ask) / 2",
    "spread": "yes_ask - yes_bid",
    "subcent": "market quoted on a 0.1c grid somewhere in its candle window (else the standard 1c grid)",
    "volume": "lifetime contracts traded (market record; includes post-ts trading, do not use as a regressor)",
    "vol_7d": "contracts traded in the 7 days up to ts (ex ante)",
    "vol_24h": "contracts traded in the 24 hours up to ts (ex ante)",
    "open_interest": "open interest at quote_ts (contracts)",
    "scheduled_ts": "scheduled time T leads are measured from",
    "sched_src": "start | date | close | deadline (see flb_xs.scheduled_time)",
    "open_time": "market open", "close_time": "market close (actual)",
    "y": "1 if the market settled YES",
    "fee_type": "Kalshi series fee type", "fee_multiplier": "series fee multiplier",
    "maker_fee": "series charges maker fees",
    "month": "YYYY-MM of scheduled_ts",
    "p_event": "inclusion probability of the event in its sampling cell",
    "dl_kind": "deadline contract type (pm_hazard.classify)",
}


def _structure(smp: pl.DataFrame) -> pl.DataFrame:
    d = out_dir()
    ev = pl.read_parquet(d / "events.parquet").select("event_ticker", "mutually_exclusive") if (d / "events.parquet").exists() \
        else pl.DataFrame(schema={"event_ticker": pl.String, "mutually_exclusive": pl.Boolean})
    smp = smp.join(ev, on="event_ticker", how="left")
    return smp.with_columns(
        pl.when(pl.col("structure0") == "multi")
        .then(pl.when(pl.col("mutually_exclusive").fill_null(False)).then(pl.lit("multi")).otherwise(pl.lit("binary")))
        .otherwise(pl.col("structure0")).alias("structure"))


def lead_rows(smp: pl.DataFrame) -> pl.DataFrame:
    """Market x lead grid with the observation time ts; the market must be open at ts and stay open
    CLOSE_GUARD longer."""
    leads = pl.DataFrame({"lead_h": [float(x) for x in LEADS_H]})
    pre = smp.join(leads, how="cross")
    inp = smp.filter((pl.col("group") == "Sports") & (pl.col("sched_src") == "start")).join(
        pl.DataFrame({"lead_h": list(INPLAY_H)}), how="cross")
    g = pl.concat([pre, inp]).with_columns(
        (pl.col("T") - pl.duration(seconds=(pl.col("lead_h") * 3600).cast(pl.Int64))).alias("ts"),
        pl.when(pl.col("lead_h") > 0).then(pl.lit("pre")).otherwise(pl.lit("inplay")).alias("phase"))
    return g.filter((pl.col("ts") >= pl.col("open_time")) & (pl.col("ts") <= pl.col("close_time") - CLOSE_GUARD))


warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")


def attach_quotes(g: pl.DataFrame, c: pl.DataFrame) -> pl.DataFrame:
    """Quote at each observation time: the last hourly candle that ended at or before ts (Kalshi
    omits periods in which nothing changed, so the last candle's close is the standing quote), its
    open interest, and contracts traded in the 7 days / 24 hours up to ts."""
    g = g.sort("ticker", "ts")
    c = (c.unique(["ticker", "end_ts"]).sort("ticker", "end_ts")
         .with_columns(pl.col("volume").fill_null(0).cum_sum().over("ticker").alias("cumvol")))
    # markets quoted on a 0.1c grid (Kalshi's sub-cent price levels) vs the standard 1c grid
    frac = lambda x: ((x * 100 - (x * 100).round(0)).abs() > 0.05)
    tick = c.group_by("ticker").agg((frac(pl.col("yes_bid_close")) | frac(pl.col("yes_ask_close"))).any().alias("subcent"))
    g = g.join(tick, on="ticker", how="left").with_columns(pl.col("subcent").fill_null(False))
    q = g.join_asof(c.rename({"end_ts": "quote_ts", "volume": "c_volume", "open_interest": "c_oi"}),
                    left_on="ts", right_on="quote_ts", by="ticker", strategy="backward")
    for lab, hrs in (("cv7", 168), ("cv24", 24)):
        back = g.select("ticker", "ts").with_columns((pl.col("ts") - timedelta(hours=hrs)).alias("t0")).sort("ticker", "t0")
        b = back.join_asof(c.select("ticker", "end_ts", "cumvol"), left_on="t0", right_on="end_ts", by="ticker",
                           strategy="backward").select("ticker", "ts", pl.col("cumvol").fill_null(0).alias(lab))
        q = q.join(b, on=["ticker", "ts"], how="left")
    q = q.filter(pl.col("quote_ts").is_not_null() & pl.col("yes_bid_close").is_not_null()
                 & pl.col("yes_ask_close").is_not_null())
    return q.with_columns((pl.col("cumvol") - pl.col("cv7")).alias("vol_7d"),
                          (pl.col("cumvol") - pl.col("cv24")).alias("vol_24h"), pl.col("c_oi").alias("oi_ts"))


def panel(chunk: int = 4000) -> None:
    d = out_dir()
    smp = _structure(pl.read_parquet(d / "sample.parquet"))
    cand = pl.scan_parquet(str(d / "candles" / "part-*.parquet"))
    have = set(pl.read_parquet(d / "candles" / "done.parquet")["ticker"])
    smp = smp.filter(pl.col("ticker").is_in(list(have)))
    tickers = smp["ticker"].unique().sort().to_list()
    parts = []
    for i in range(0, len(tickers), chunk):
        tk = tickers[i:i + chunk]
        g = lead_rows(smp.filter(pl.col("ticker").is_in(tk)))
        c = cand.filter(pl.col("ticker").is_in(tk)).collect()
        parts.append(attach_quotes(g, c))
        print(f"  panel {min(i + chunk, len(tickers))}/{len(tickers)} markets", flush=True)
    p = pl.concat(parts, how="diagonal_relaxed")
    p = p.select(
        "ticker", "event_ticker", "series_ticker", "group", "category", "structure", "n_legs", "mutually_exclusive",
        "ladder_key", "strike", "lead_h", "phase", "ts", "quote_ts",
        ((pl.col("ts") - pl.col("quote_ts")).dt.total_minutes() / 60).alias("quote_age_h"),
        pl.col("yes_bid_close").alias("yes_bid"), pl.col("yes_ask_close").alias("yes_ask"),
        ((pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2).alias("mid"),
        (pl.col("yes_ask_close") - pl.col("yes_bid_close")).alias("spread"), "subcent",
        pl.col("volume"), "vol_7d", "vol_24h", pl.col("oi_ts").alias("open_interest"),
        pl.col("T").alias("scheduled_ts"), "sched_src", "open_time", "close_time", "y", "fee_type",
        pl.col("fee_multiplier").fill_null(1.0), pl.col("fee_type").fill_null("").str.contains("maker").alias("maker_fee"),
        "month", "p_event", "dl_kind").sort("ticker", "lead_h")
    _save(p, d / "panel.parquet")
    schema = {c: {"dtype": str(t), "doc": PANEL_DOC.get(c, "")} for c, t in p.schema.items()}
    (d / "panel_schema.json").write_text(json.dumps({"rows": p.height, "markets": p["ticker"].n_unique(),
                                                     "events": p["event_ticker"].n_unique(),
                                                     "leads_h": sorted(set(p["lead_h"].to_list())),
                                                     "columns": schema}, indent=1))
    print(f"panel {p.height:,} rows, {p['ticker'].n_unique():,} markets", flush=True)
    print(p.group_by("group", "phase").agg(pl.len(), pl.col("ticker").n_unique().alias("markets")).sort("group"), flush=True)


# ---- statistics ---------------------------------------------------------------------------------
Q_CLIP = 0.005       # logit(q) is evaluated on q clipped to [0.5c, 99.5c] (the extreme mid on a 1c grid)
RHO = 0.05           # opportunity cost of capital per year in the no-arbitrage band (framework example)
MAX_SPREAD = 0.10    # a quote is usable when ask - bid <= 10c (one-sided quotes at the 1c floor qualify)


def logit(q: Any) -> np.ndarray:
    q = np.clip(np.asarray(q, dtype=float), Q_CLIP, 1 - Q_CLIP)
    return np.log(q / (1 - q))


def _cluster_meat(scores: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, int]:
    _, inv = np.unique(g, return_inverse=True)
    k = int(inv.max()) + 1
    s = np.zeros((k, scores.shape[1]))
    np.add.at(s, inv, scores)
    return s.T @ s * (k / (k - 1) if k > 1 else 1.0), k


def logit_fit(y: Any, x: Any, g: Any, w: Any | None = None, iters: int = 50) -> dict[str, Any]:
    """Weighted logit of y on [1, x] (x = logit q) with standard errors clustered by g.
    Calibration: a = 0, b = 1. b > 1: prices are too moderate (favourites underpriced, longshots
    overpriced) - the favourite-longshot bias."""
    y, x, g = np.asarray(y, dtype=float), np.asarray(x, dtype=float), np.asarray(g)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x) & (w > 0)
    y, x, g, w = y[ok], x[ok], g[ok], w[ok]
    if len(y) < 20 or len(np.unique(g)) < 5:
        return {"n": len(y), "clusters": len(np.unique(g)), "a": None, "b": None}
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        H = (X * (w * p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H + 1e-10 * np.eye(2), X.T @ (w * (y - p)))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
    H = (X * (w * p * (1 - p))[:, None]).T @ X
    Hi = np.linalg.inv(H)
    meat, k = _cluster_meat(X * (w * (y - p))[:, None], g)
    V = Hi @ meat @ Hi
    se = np.sqrt(np.diag(V))
    return {"n": len(y), "clusters": k, "a": float(beta[0]), "b": float(beta[1]),
            "se_a": float(se[0]), "se_b": float(se[1]), "t_b1": float((beta[1] - 1) / se[1]) if se[1] > 0 else None,
            "t_a0": float(beta[0] / se[0]) if se[0] > 0 else None}


def ols_fit(yv: Any, X: Any, g: Any, w: Any | None = None) -> dict[str, Any]:
    """Weighted least squares with standard errors clustered by g; returns coefficients, SEs, t."""
    yv, X, g = np.asarray(yv, dtype=float), np.asarray(X, dtype=float), np.asarray(g)
    w = np.ones_like(yv) if w is None else np.asarray(w, dtype=float)
    ok = np.isfinite(yv) & np.all(np.isfinite(X), axis=1) & (w > 0)
    yv, X, g, w = yv[ok], X[ok], g[ok], w[ok]
    if len(yv) <= X.shape[1] or len(np.unique(g)) < 3:
        return {"n": len(yv), "coef": None}
    XtW = (X * w[:, None]).T
    A = np.linalg.pinv(XtW @ X)
    beta = A @ (XtW @ yv)
    meat, k = _cluster_meat(X * (w * (yv - X @ beta))[:, None], g)
    se = np.sqrt(np.diag(A @ meat @ A))
    return {"n": len(yv), "clusters": k, "coef": beta.tolist(), "se": se.tolist(),
            "t": [float(b / s) if s > 0 else None for b, s in zip(beta, se, strict=True)]}


def shin_z(prices: Any, tol: float = 1e-12) -> tuple[float, np.ndarray]:
    """Shin (1993) insider share z and implied probabilities from quoted prices of an exhaustive,
    mutually exclusive set of outcomes with sum Pi > 1:
    p_i(z) = [sqrt(z^2 + 4 (1 - z) pi_i^2 / Pi) - z] / (2 (1 - z)), z solving sum p_i(z) = 1.
    Returns (0, pi / Pi) when Pi <= 1."""
    pi = np.asarray(prices, dtype=float)
    S = pi.sum()
    if S <= 1 + 1e-12 or len(pi) < 2:
        return 0.0, pi / S if S > 0 else pi

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4 * (1 - z) * pi * pi / S) - z) / (2 * (1 - z))

    lo, hi = 0.0, 0.999
    if probs(hi).sum() > 1:           # cannot reach 1 (degenerate); cap
        return hi, probs(hi) / probs(hi).sum()
    for _ in range(200):
        mid = (lo + hi) / 2
        if probs(mid).sum() > 1:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    z = (lo + hi) / 2
    return z, probs(z)


def fee_per_dollar(q: Any, contracts: int | None = None, mult: float = 1.0) -> np.ndarray:
    """Taker fee per dollar of stake at price q: exact (contracts=None) or for one order of
    ``contracts`` contracts with the fee rounded up to the cent (Kalshi's per-order rounding)."""
    q = np.asarray(q, dtype=float)
    f = fee_pc(q, mult=mult)
    if contracts is None:
        return f / q
    tot = np.ceil(np.round(f * contracts * 100, 9)) / 100
    return tot / (contracts * q)


def usable(p: pl.DataFrame, max_spread: float = MAX_SPREAD) -> pl.DataFrame:
    """Quotes that carry a price: ask > bid, spread <= max_spread, not an empty book."""
    return p.filter((pl.col("yes_ask") > pl.col("yes_bid")) & (pl.col("spread") <= max_spread + 1e-9)
                    & (pl.col("yes_ask") <= 1) & (pl.col("yes_bid") >= 0))


def sides(p: pl.DataFrame) -> pl.DataFrame:
    """Two side-observations per quote (YES and NO). q_mid: side price at the mid; q_ask: taker
    price (YES ask, or 1 - YES bid for NO); q_bid: resting (maker) price. Returns per dollar:
    r_mid (gross at mid), r_ask (gross at ask), r_net (taker at ask net of the exact taker fee)."""
    base = ["ticker", "event_ticker", "series_ticker", "group", "structure", "lead_h", "phase", "ts", "month",
            "vol_7d", "open_interest", "spread", "fee_multiplier", "maker_fee", "w_m", "w_v", "sched_src",
            "yes_bid", "yes_ask", "n_legs", "subcent"]
    base = [c for c in base if c in p.columns] + [c for c in p.columns if c.startswith(("w_", "t_")) and c not in base]
    yes = p.select(*base, pl.lit("yes").alias("side"), pl.col("mid").alias("q_mid"), pl.col("yes_ask").alias("q_ask"),
                   pl.col("yes_bid").alias("q_bid"), pl.col("y").cast(pl.Float64).alias("won"))
    no = p.select(*base, pl.lit("no").alias("side"), (1 - pl.col("mid")).alias("q_mid"), (1 - pl.col("yes_bid")).alias("q_ask"),
                  (1 - pl.col("yes_ask")).alias("q_bid"), (1 - pl.col("y").cast(pl.Float64)).alias("won"))
    s = pl.concat([yes, no])
    fee = fee_pc(s["q_ask"].to_numpy(), mult=s["fee_multiplier"].to_numpy())
    mfee = fee_pc(s["q_bid"].to_numpy(), maker=True, mult=s["fee_multiplier"].to_numpy(),
                  maker_fee_series=s["maker_fee"].to_numpy())
    s = s.with_columns(pl.Series("fee", fee), pl.Series("mfee", mfee))
    return s.with_columns(
        ((pl.col("won") - pl.col("q_mid")) / pl.col("q_mid")).alias("r_mid"),
        pl.when(pl.col("q_ask") < 1).then((pl.col("won") - pl.col("q_ask")) / pl.col("q_ask")).alias("r_ask"),
        pl.when(pl.col("q_ask") < 1).then((pl.col("won") - pl.col("q_ask") - pl.col("fee")) / (pl.col("q_ask") + pl.col("fee"))).alias("r_net"),
        pl.when(pl.col("q_bid") > 0).then((pl.col("won") - pl.col("q_bid") - pl.col("mfee")) / (pl.col("q_bid") + pl.col("mfee"))).alias("r_maker"),
        (pl.col("won") - pl.col("q_mid")).alias("pc_mid"),
        bucket_expr("q_mid").alias("bucket"),
        bucket_expr("q_ask").alias("bucket_ask"))


def with_weights(p: pl.DataFrame) -> pl.DataFrame:
    """w_m: each market weighs 1 in total across its observations in ``p``; w_v: w_m x contracts
    traded in the 7 days up to the quote (ex ante volume weight)."""
    return p.with_columns((1.0 / pl.len().over("ticker")).alias("w_m")).with_columns(
        (pl.col("w_m") * pl.col("vol_7d").fill_null(0).clip(lower_bound=0)).alias("w_v"))


def _cm(df: pl.DataFrame, col: str, w: str | None = None) -> dict[str, Any]:
    x = df[col].to_numpy()
    r = clustered(x, df["event_ticker"].to_numpy(), None if w is None else df[w].to_numpy())
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}


def bucket_table(s: pl.DataFrame, w: str = "w_m") -> list[dict[str, Any]]:
    """Per side-price bucket: n, clusters, mean side price, win rate (weighted), and returns per
    dollar: gross at mid (bucketed by q_mid), gross at ask and net of taker fees (bucketed by q_ask)."""
    out = []
    for b in BUCKET_LABELS:
        x = s.filter(pl.col("bucket") == b)
        xa = s.filter((pl.col("bucket_ask") == b) & pl.col("r_net").is_not_null())
        if x.height == 0:
            continue
        ww = x[w].to_numpy()
        if ww.sum() <= 0:
            continue
        won = _cm(x, "won", w)
        rec = {"bucket": b, "n": x.height, "clusters": won["clusters"],
               "q_mid": float(np.average(x["q_mid"].to_numpy(), weights=ww)),
               "win": won["mean"], "win_se": won["se"],
               "r_mid": _cm(x, "r_mid", w), "pc_mid": _cm(x, "pc_mid", w)}
        if xa.height and xa[w].sum() > 0:
            rec.update({"n_ask": xa.height, "q_ask": float(np.average(xa["q_ask"].to_numpy(), weights=xa[w].to_numpy())),
                        "r_ask": _cm(xa, "r_ask", w), "r_net": _cm(xa, "r_net", w)})
        out.append(rec)
    return out


def flb_measures(p: pl.DataFrame, s: pl.DataFrame | None = None, w: str = "w_m") -> dict[str, Any]:
    """The three FLB measures on a quote panel ``p`` (with weights): logit slope (YES side), the
    longshot and favourite returns per dollar at mid and net, and the longshot-minus-favourite spread."""
    if p.height == 0:
        return {"n_quotes": 0}
    s = sides(p) if s is None else s
    fit = logit_fit(p["y"].to_numpy(), logit(p["mid"].to_numpy()), p["event_ticker"].to_numpy(), p[w].to_numpy())
    out: dict[str, Any] = {"n_quotes": p.height, "markets": p["ticker"].n_unique(), "events": p["event_ticker"].n_unique(),
                           "logit": fit}
    ls = s.filter(pl.col("q_mid") <= LONGSHOT + 1e-9)
    fav = s.filter(pl.col("q_mid") >= FAVOURITE - 1e-9)
    out["ls_r_mid"] = _cm(ls, "r_mid", w) if ls.height else None
    out["fav_r_mid"] = _cm(fav, "r_mid", w) if fav.height else None
    lsa = s.filter((pl.col("q_ask") <= LONGSHOT + 1e-9) & pl.col("r_net").is_not_null())
    fava = s.filter((pl.col("q_ask") >= FAVOURITE - 1e-9) & pl.col("r_net").is_not_null())
    out["ls_r_net"] = _cm(lsa, "r_net", w) if lsa.height else None
    out["fav_r_net"] = _cm(fava, "r_net", w) if fava.height else None
    for lab, a, b, col in (("spread_mid", ls, fav, "r_mid"), ("spread_net", lsa, fava, "r_net")):
        both = pl.concat([a.with_columns(pl.lit(1.0).alias("is_ls")), b.with_columns(pl.lit(0.0).alias("is_ls"))])
        if both.height and a.height and b.height:
            r = ols_fit(both[col].to_numpy(), np.column_stack([np.ones(both.height), both["is_ls"].to_numpy()]),
                        both["event_ticker"].to_numpy(), both[w].to_numpy())
            out[lab] = {"n": r["n"], "clusters": r.get("clusters"), "diff": r["coef"][1] if r["coef"] else None,
                        "se": r["se"][1] if r["coef"] else None, "t": r["t"][1] if r["coef"] else None}
    return out


def both_weights(p: pl.DataFrame, s: pl.DataFrame) -> dict[str, Any]:
    return {"mkt": flb_measures(p, s, "w_m"), "vol": flb_measures(p, s, "w_v")}


def _terciles(col: pl.Expr, by: str = "group") -> pl.Expr:
    """Tercile 1..3 of ``col`` within ``by`` (average ranks, so ties share a tercile)."""
    r = col.rank("average").over(by) / pl.len().over(by)
    return pl.when(r <= 1 / 3).then(1).when(r <= 2 / 3).then(2).otherwise(3)


# ---- analyses ------------------------------------------------------------------------------------
def multi_outcome(p: pl.DataFrame) -> dict[str, Any]:
    """Mutually exclusive events with every leg quoted and exactly one YES: overround at mid and at
    the ask, Shin z (from asks), and where the mispricing sits (leg rank and the longshot field)."""
    m = p.filter(pl.col("mutually_exclusive").fill_null(False) & (pl.col("n_legs") >= 2)
                 & (pl.col("structure").is_in(["multi", "binary"])))
    g = m.group_by("event_ticker", "lead_h").agg(
        pl.len().alias("k"), pl.col("n_legs").first(), pl.col("y").sum().alias("ys"),
        pl.col("mid").sum().alias("sum_mid"), pl.col("yes_ask").sum().alias("sum_ask"),
        pl.col("yes_bid").sum().alias("sum_bid"), pl.col("group").first(), pl.col("structure").first(),
        pl.col("ticker").alias("tks"), pl.col("yes_ask").alias("asks"), pl.col("mid").alias("mids"), pl.col("y").alias("yy"))
    g = g.filter((pl.col("k") == pl.col("n_legs")) & (pl.col("ys") == 1))
    if g.height == 0:
        return {"events": 0}
    zs, legs = [], []
    for r in g.iter_rows(named=True):
        z, ps = shin_z(r["asks"])
        zs.append(z)
        mids = np.asarray(r["mids"], dtype=float)
        order = np.argsort(-mids, kind="stable")
        rank = np.empty(len(mids), dtype=int)
        rank[order] = np.arange(1, len(mids) + 1)
        for i in range(len(mids)):
            legs.append({"event_ticker": r["event_ticker"], "lead_h": r["lead_h"], "n_legs": r["n_legs"],
                         "structure": r["structure"], "group": r["group"],
                         "rank": int(rank[i]), "mid": float(mids[i]), "norm": float(mids[i] / r["sum_mid"]),
                         "shin": float(ps[i]), "y": float(r["yy"][i])})
    g = g.with_columns(pl.Series("shin_z", zs), (pl.col("sum_mid") - 1).alias("over_mid"),
                       (pl.col("sum_ask") - 1).alias("over_ask"), (pl.col("sum_bid") - 1).alias("under_bid"))
    L = pl.DataFrame(legs).with_columns(
        pl.when(pl.col("rank") == 1).then(pl.lit("1 (favourite)")).when(pl.col("rank") == 2).then(pl.lit("2"))
        .when(pl.col("rank") <= 5).then(pl.lit("3-5")).otherwise(pl.lit("6+")).alias("rank_class"),
        (pl.col("y") - pl.col("mid")).alias("e_mid"), (pl.col("y") - pl.col("norm")).alias("e_norm"),
        (pl.col("y") - pl.col("shin")).alias("e_shin"))
    out: dict[str, Any] = {}
    for lab, sub in (("two_leg", g.filter(pl.col("n_legs") == 2)), ("multi", g.filter(pl.col("n_legs") >= 3))):
        if sub.height == 0:
            continue
        ev = sub["event_ticker"].to_numpy()
        out[lab] = {"event_leads": sub.height, "events": sub["event_ticker"].n_unique(),
                    "over_mid": clustered(sub["over_mid"].to_numpy(), ev), "over_ask": clustered(sub["over_ask"].to_numpy(), ev),
                    "under_bid": clustered(sub["under_bid"].to_numpy(), ev),
                    "shin_z": clustered(sub["shin_z"].to_numpy(), ev),
                    "shin_z_median": float(np.median(sub["shin_z"].to_numpy())),
                    "over_ask_median": float(np.median(sub["over_ask"].to_numpy())),
                    "legs_median": float(np.median(sub["n_legs"].to_numpy()))}
        ls = L.filter(pl.col("n_legs") == 2) if lab == "two_leg" else L.filter(pl.col("n_legs") >= 3)
        out[lab]["by_rank"] = []
        for rc in sorted(ls["rank_class"].unique().to_list()):
            x = ls.filter(pl.col("rank_class") == rc)
            e = x["event_ticker"].to_numpy()
            out[lab]["by_rank"].append({"rank_class": rc, "n": x.height, "mid": float(x["mid"].mean()),
                                        "win": float(x["y"].mean()), "e_mid": clustered(x["e_mid"].to_numpy(), e),
                                        "e_norm": clustered(x["e_norm"].to_numpy(), e),
                                        "e_shin": clustered(x["e_shin"].to_numpy(), e)})
        # the field: all legs priced at or below 10c, summed per event-lead
        fld = ls.filter(pl.col("mid") <= LONGSHOT).group_by("event_ticker", "lead_h").agg(
            pl.col("mid").sum().alias("price"), pl.col("y").sum().alias("win"), pl.col("norm").sum().alias("norm"),
            pl.col("shin").sum().alias("shin"), pl.len().alias("legs"))
        if fld.height:
            e = fld["event_ticker"].to_numpy()
            out[lab]["field"] = {"n": fld.height, "legs_mean": float(fld["legs"].mean()), "price": float(fld["price"].mean()),
                                 "win": float(fld["win"].mean()),
                                 "price_minus_win": clustered((fld["price"] - fld["win"]).to_numpy(), e),
                                 "norm_minus_win": clustered((fld["norm"] - fld["win"]).to_numpy(), e),
                                 "shin_minus_win": clustered((fld["shin"] - fld["win"]).to_numpy(), e)}
        # proper scoring: raw mid vs normalised vs Shin (Brier, per leg)
        out[lab]["brier"] = {k: float(((ls["y"] - ls[c]) ** 2).mean()) for k, c in (("mid", "mid"), ("norm", "norm"), ("shin", "shin"))}
    z_by_lead = []
    for ld in sorted(g["lead_h"].unique().to_list(), reverse=True):
        x = g.filter((pl.col("lead_h") == ld) & (pl.col("n_legs") >= 3))
        if x.height >= 20:
            z_by_lead.append({"lead_h": ld, "n": x.height, "shin_z_median": float(x["shin_z"].median()),
                              "over_ask": clustered(x["over_ask"].to_numpy(), x["event_ticker"].to_numpy()),
                              "over_mid": clustered(x["over_mid"].to_numpy(), x["event_ticker"].to_numpy())})
    out["by_lead"] = z_by_lead
    zg = []
    for (grp, two_leg), x in g.with_columns((pl.col("n_legs") == 2).alias("two")).group_by(["group", "two"]):
        if x.height >= 20 and x["event_ticker"].n_unique() >= 10:
            e = x["event_ticker"].to_numpy()
            zg.append({"group": grp, "legs": "2" if two_leg else ">=3", "event_leads": x.height,
                       "events": x["event_ticker"].n_unique(), "shin_z_median": float(x["shin_z"].median()),
                       "shin_z": clustered(x["shin_z"].to_numpy(), e), "over_ask": clustered(x["over_ask"].to_numpy(), e),
                       "over_mid": clustered(x["over_mid"].to_numpy(), e)})
    out["by_group"] = sorted(zg, key=lambda r: (r["legs"], r["group"]))
    (out_dir() / "tables").mkdir(exist_ok=True)
    _save(g.drop("tks", "asks", "mids", "yy"), out_dir() / "tables" / "me_events.parquet")
    _save(L, out_dir() / "tables" / "me_legs.parquet")
    return out


def floor_analysis(pre: pl.DataFrame, S: pl.DataFrame) -> dict[str, Any]:
    """The 1c tick floor: mass at 1-2c, and the 1-2c loss split into the part a price at the floor
    must lose given the realised win rate (mechanical) and the part from prices above the floor."""
    yes = S.filter(pl.col("side") == "yes")
    tot_w = float(yes["w_m"].sum())
    out: dict[str, Any] = {
        "share_quotes_yes_mid_le_2c": float(yes.filter(pl.col("q_mid") <= 0.02 + 1e-9)["w_m"].sum() / tot_w),
        "share_quotes_either_side_le_2c": float(S.filter(pl.col("q_mid") <= 0.02 + 1e-9)["w_m"].sum() / tot_w),
        "share_quotes_pinned": float(pre.filter((pl.col("yes_bid") == 0) & (pl.col("yes_ask") <= 0.01 + 1e-9))["w_m"].sum() / tot_w)}
    ls = S.filter((pl.col("q_mid") <= 0.02 + 1e-9) & ~pl.col("subcent"))
    pinned = ls.filter(pl.col("q_mid") <= 0.005 + 1e-9)
    above = ls.filter(pl.col("q_mid") > 0.005 + 1e-9)
    sub = S.filter((pl.col("q_mid") <= 0.02 + 1e-9) & pl.col("subcent"))
    out["subcent_markets_share"] = float(pre.filter(pl.col("subcent"))["ticker"].n_unique() / max(pre["ticker"].n_unique(), 1))
    def stats(x: pl.DataFrame, price: str) -> dict[str, Any]:
        w = x["w_m"].to_numpy()
        win = _cm(x, "won", "w_m")
        return {"n": x.height, "clusters": win["clusters"], "w": float(w.sum()),
                "q": float(np.average(x[price].to_numpy(), weights=w)), "win": win["mean"], "win_se": win["se"]}

    def decompose(x: pl.DataFrame, at_floor: pl.Expr, floor: float, price: str, ret: str) -> dict[str, Any]:
        """Loss per contract (price - win) of all sides in x; the mechanical part is the loss of the
        sides quoted at the floor, max(floor - their win rate, 0), times their weight share."""
        a, f = stats(x, price), x.filter(at_floor)
        loss = a["q"] - a["win"]
        mech = 0.0
        if f.height:
            fs = stats(f, price)
            mech = fs["w"] / a["w"] * max(floor - fs["win"], 0.0)
        mech = min(mech, loss) if loss > 0 else 0.0
        return {**a, "q_mid" if price == "q_mid" else "q_ask": a["q"], "loss_pc": loss, "mechanical_pc": mech,
                "mechanical_share": mech / loss if loss > 0 else None, ret: _cm(x, ret, "w_m")}

    floor_mid = pl.col("q_mid") <= 0.005 + 1e-9
    rows = {"all_1_2c": decompose(ls, floor_mid, 0.005, "q_mid", "r_mid")}
    for lab, x in (("pinned_at_floor", pinned), ("above_floor", above)):
        if x.height:
            rows[lab] = decompose(x, floor_mid, 0.005, "q_mid", "r_mid")
    if sub.height:
        rows["subcent_grid_0_2c"] = decompose(sub, pl.col("q_mid") <= 0.0005 + 1e-9, 0.0005, "q_mid", "r_mid")
    # taker at a 1-2c ask on the 1c grid: the floor is the 1c ask itself
    la = S.filter((pl.col("q_ask") <= 0.02 + 1e-9) & pl.col("r_net").is_not_null() & ~pl.col("subcent"))
    if la.height:
        rows["taker_ask_1_2c"] = decompose(la, pl.col("q_ask") <= 0.01 + 1e-9, 0.01, "q_ask", "r_net")
    out["decomposition"] = rows
    return out


def no_band(S: pl.DataFrame) -> list[dict[str, Any]]:
    """Per bucket of the side price: mispricing per contract at mid (q - win) against the cost of
    taking the other side (half-spread + taker fee) or resting on it (maker fee only)."""
    out = []
    for b in BUCKET_LABELS:
        x = S.filter(pl.col("bucket") == b)
        if x.height < 50:
            continue
        w = x["w_m"].to_numpy()
        e = _cm(x, "pc_mid", "w_m")
        other_q = 1 - x["q_mid"].to_numpy()
        half = x["spread"].to_numpy() / 2
        fee_other = fee_pc(np.clip(other_q + half, 0, 1), mult=x["fee_multiplier"].to_numpy())
        capital = RHO * (x["lead_h"].clip(lower_bound=0).to_numpy() / 8760.0) * other_q
        out.append({"bucket": b, "n": x.height, "q_mid": float(np.average(x["q_mid"].to_numpy(), weights=w)),
                    "capital": float(np.average(capital, weights=w)),
                    "band_taker_capital": float(np.average(half + fee_other + capital, weights=w)),
                    "mispricing_pc": -e["mean"], "mispricing_se": e["se"],
                    "half_spread": float(np.average(half, weights=w)),
                    "taker_fee_other": float(np.average(fee_other, weights=w)),
                    "band_taker": float(np.average(half + fee_other, weights=w)),
                    "maker_fee_other": float(np.average(fee_pc(other_q, maker=True, mult=x["fee_multiplier"].to_numpy(),
                                                               maker_fee_series=x["maker_fee"].to_numpy()), weights=w))})
    return out


def other_side(S: pl.DataFrame) -> dict[str, Any]:
    """Buying the other side of a longshot (NO when YES mid <= 10c, and the mirror case): as taker at
    1 - bid net of taker fees; resting at the bid as maker (fill assumed at the quote, no adverse
    selection: an upper bound) net of maker fees where the series charges them."""
    fav = S.filter(pl.col("q_mid") >= FAVOURITE - 1e-9)
    out: dict[str, Any] = {}
    for lab, x in (("all", fav), *((g, fav.filter(pl.col("group") == g)) for g in GROUPS)):
        if x.height < 50:
            continue
        xt = x.filter(pl.col("r_net").is_not_null())
        xm = x.filter(pl.col("r_maker").is_not_null())
        out[lab] = {"n": x.height, "taker_net": _cm(xt, "r_net", "w_m"), "taker_gross": _cm(xt, "r_ask", "w_m"),
                    "maker_net": _cm(xm, "r_maker", "w_m"), "mid_gross": _cm(x, "r_mid", "w_m"),
                    "q_ask": float(xt["q_ask"].mean()) if xt.height else None,
                    "q_bid": float(xm["q_bid"].mean()) if xm.height else None,
                    "taker_net_vol": _cm(xt, "r_net", "w_v") if xt["w_v"].sum() > 0 else None}
    return out


def _qtr(month: pl.Expr) -> pl.Expr:
    return month.str.slice(0, 4) + pl.lit("Q") + ((month.str.slice(5, 2).cast(pl.Int32) - 1) // 3 + 1).cast(pl.String)


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.floating | float):
        v = float(x)
        return None if not math.isfinite(v) else round(v, 7)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, datetime):
        return x.isoformat()
    return x


def _write(name: str, obj: Any) -> None:
    d = out_dir() / "tables"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(_jsonable(obj), indent=1))


def analyze() -> None:
    d = out_dir()
    (d / "tables").mkdir(parents=True, exist_ok=True)
    raw = pl.read_parquet(d / "panel.parquet")
    p = usable(raw)
    pre = with_weights(p.filter(pl.col("phase") == "pre"))
    inp = with_weights(p.filter(pl.col("phase") == "inplay"))
    S = sides(pre)
    Si = sides(inp) if inp.height else None
    summary: dict[str, Any] = {"generated": datetime.now(UTC).isoformat(timespec="seconds")}
    cells = pl.read_parquet(d / "frame_cells.parquet")
    summary["coverage"] = {
        "panel_rows": raw.height, "usable_rows": p.height, "pre_rows": pre.height, "inplay_rows": inp.height,
        "markets": pre["ticker"].n_unique(), "events": pre["event_ticker"].n_unique(),
        "series": pre["series_ticker"].n_unique(),
        "first_month": pre["month"].min(), "last_month": pre["month"].max(),
        "by_group": pre.group_by("group").agg(pl.len().alias("quotes"), pl.col("ticker").n_unique().alias("markets"),
                                              pl.col("event_ticker").n_unique().alias("events"),
                                              pl.col("month").min().alias("first"), pl.col("month").max().alias("last")).sort("group").to_dicts(),
        "by_structure": pre.group_by("structure").agg(pl.len().alias("quotes"), pl.col("ticker").n_unique().alias("markets"),
                                                      pl.col("event_ticker").n_unique().alias("events")).sort("structure").to_dicts(),
        "by_lead": pre.group_by("lead_h").agg(pl.len().alias("quotes")).sort("lead_h").to_dicts(),
        "by_year": pre.with_columns(pl.col("month").str.slice(0, 4).alias("year")).group_by("year", "group").agg(
            pl.col("ticker").n_unique().alias("markets")).sort("year", "group").to_dicts(),
        "frame": cells.group_by("group").agg(pl.col("frame_markets").sum(), pl.col("eligible_markets").sum(),
                                             pl.col("eligible_events").sum(), pl.col("month").min().alias("first"),
                                             pl.col("month").max().alias("last")).sort("group").to_dicts(),
        "frame_total_markets": int(cells["frame_markets"].sum()),
        "sampled_markets": pl.read_parquet(d / "sample.parquet", columns=["ticker"]).height,
        "sampled_events": pl.read_parquet(d / "sample.parquet", columns=["event_ticker"])["event_ticker"].n_unique(),
        "usable_share": p.height / raw.height if raw.height else None,
        "sched_src": pre.group_by("sched_src").agg(pl.col("ticker").n_unique().alias("markets")).to_dicts()}
    print("coverage done", flush=True)

    # 1. calibration and FLB measures, pooled and by group
    calib = {"All": {"mkt": bucket_table(S, "w_m"), "vol": bucket_table(S, "w_v")}}
    meas = {"All": both_weights(pre, S)}
    for g in GROUPS:
        pg, sg = pre.filter(pl.col("group") == g), S.filter(pl.col("group") == g)
        if pg.height < 200:
            continue
        calib[g] = {"mkt": bucket_table(sg, "w_m"), "vol": bucket_table(sg, "w_v")}
        meas[g] = both_weights(pg, sg)
    if Si is not None:
        calib["Sports in-play"] = {"mkt": bucket_table(Si, "w_m"), "vol": bucket_table(Si, "w_v")}
        meas["Sports in-play"] = both_weights(inp, Si)
    _write("calibration", calib)
    _write("measures_group", meas)
    summary["measures_group"] = meas
    print("calibration done", flush=True)

    # 2. term structure: each lead separately (one quote per market and lead)
    term = []
    groups_t = ["All", *GROUPS, "Sports (exact start)"]
    for g in groups_t:
        for ld in (*LEADS_H, *INPLAY_H):
            src = p.filter(pl.col("lead_h") == float(ld))
            if g == "Sports (exact start)":
                src = src.filter((pl.col("group") == "Sports") & (pl.col("sched_src") == "start"))
            elif g != "All":
                src = src.filter(pl.col("group") == g)
            elif ld < 0:
                continue
            if src.height < 100 or src["event_ticker"].n_unique() < 30:
                continue
            src = with_weights(src)
            m = flb_measures(src, None, "w_m")
            term.append({"group": g, "lead_h": ld, "label": LEAD_LABEL[ld], **m})
    _write("term", term)
    summary["term"] = term
    print("term done", flush=True)

    # 3. structure
    struct = {}
    for st in STRUCTURES:
        ps, ss = pre.filter(pl.col("structure") == st), S.filter(pl.col("structure") == st)
        if ps.height < 100:
            continue
        struct[st] = {"all": both_weights(ps, ss), "bucket": bucket_table(ss, "w_m"),
                      "by_group": {g: flb_measures(ps.filter(pl.col("group") == g), ss.filter(pl.col("group") == g), "w_m")
                                   for g in GROUPS if ps.filter(pl.col("group") == g).height >= 200}}
        regions = {}
        for lab, lo, hi in (("LS <=10c", 0.0, LONGSHOT), ("body 35-65c", 0.35, 0.65), ("FAV >=90c", FAVOURITE, 1.0)):
            x = ss.filter((pl.col("q_mid") >= lo - 1e-9) & (pl.col("q_mid") <= hi + 1e-9))
            xa = ss.filter((pl.col("q_ask") >= lo - 1e-9) & (pl.col("q_ask") <= hi + 1e-9) & pl.col("r_net").is_not_null())
            regions[lab] = {"n": x.height, "r_mid": _cm(x, "r_mid", "w_m") if x.height else None,
                            "pc_mid": _cm(x, "pc_mid", "w_m") if x.height else None,
                            "r_net": _cm(xa, "r_net", "w_m") if xa.height else None}
        struct[st]["regions"] = regions
    struct["multi_outcome"] = multi_outcome(pre)
    legs_cls = pl.when(pl.col("n_legs") <= 5).then(pl.lit("3-5")).when(pl.col("n_legs") <= 10).then(pl.lit("6-10")) \
        .when(pl.col("n_legs") <= 20).then(pl.lit("11-20")).otherwise(pl.lit("21+"))
    pm = pre.filter(pl.col("structure") == "multi").with_columns(legs_cls.alias("legs_cls"))
    struct["multi_by_legs"] = [{"legs": c, **flb_measures(pm.filter(pl.col("legs_cls") == c), None, "w_m")}
                               for c in ("3-5", "6-10", "11-20", "21+") if pm.filter(pl.col("legs_cls") == c).height >= 100]
    _write("structure", struct)
    summary["structure"] = struct
    print("structure done", flush=True)

    # 4. liquidity terciles (within group) and calendar time
    liq = {}
    pt = pre.with_columns(_terciles(pl.col("vol_7d").fill_null(0)).alias("t_vol"), _terciles(pl.col("spread")).alias("t_spread"),
                          _terciles(pl.col("open_interest").fill_null(0)).alias("t_oi"))
    St = sides(pt)
    for var in ("t_vol", "t_spread", "t_oi"):
        liq[var] = []
        for t in (1, 2, 3):
            pp = pt.filter(pl.col(var) == t)
            ss = St.filter(pl.col(var) == t)
            liq[var].append({"tercile": t, "median_value": float(pp[{"t_vol": "vol_7d", "t_spread": "spread", "t_oi": "open_interest"}[var]].median() or 0),
                             **flb_measures(pp, ss, "w_m")})
    _write("liquidity", liq)
    summary["liquidity"] = liq
    pq = pre.with_columns(_qtr(pl.col("month")).alias("qtr"))
    Sq = S.with_columns(_qtr(pl.col("month")).alias("qtr"))
    cal = []
    for g in ("All", *GROUPS):
        for q in sorted(pq["qtr"].unique().to_list()):
            pp = pq.filter(pl.col("qtr") == q)
            ss = Sq.filter(pl.col("qtr") == q)
            if g != "All":
                pp, ss = pp.filter(pl.col("group") == g), ss.filter(pl.col("group") == g)
            if pp.height < 150 or pp["event_ticker"].n_unique() < 20:
                continue
            cal.append({"group": g, "qtr": q, **flb_measures(pp, ss, "w_m")})
    venue = (cells.filter(pl.col("month").is_not_null()).with_columns(_qtr(pl.col("month")).alias("qtr"))
             .group_by("qtr").agg(pl.col("frame_markets").sum().alias("markets"),
                                  pl.col("frame_volume").sum().alias("contracts")).sort("qtr"))
    _write("calendar", {"cells": cal, "venue": venue.to_dicts()})
    summary["calendar"] = {"cells": cal, "venue": venue.to_dicts()}
    print("liquidity/calendar done", flush=True)

    # 5. limits to arbitrage
    qgrid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    fees = {"q": qgrid.tolist(), "exact": fee_per_dollar(qgrid).tolist(),
            **{f"order_{n}": fee_per_dollar(qgrid, n).tolist() for n in (1, 10, 100)}}
    limits = {"fee_per_dollar": fees, "band": no_band(S), "other_side": other_side(S), "floor": floor_analysis(pre, S)}
    fee_paid = S.filter(pl.col("r_net").is_not_null())
    limits["fee_share_by_bucket"] = [
        {"bucket": b, "fee_per_dollar": float((x["fee"] / x["q_ask"]).mean()), "n": x.height}
        for b in BUCKET_LABELS if (x := fee_paid.filter(pl.col("bucket_ask") == b)).height]
    _write("limits", limits)
    summary["limits"] = limits
    print("limits done", flush=True)

    # 6. robustness
    rob: dict[str, Any] = {}
    med = pre["ts"].sort()[pre.height // 2]
    for lab, cond in (("first_half", pl.col("ts") < med), ("second_half", pl.col("ts") >= med)):
        pp = with_weights(pre.filter(cond))
        rob[lab] = {"split_ts": med.isoformat(), "all": flb_measures(pp, None, "w_m"),
                    "by_group": {g: flb_measures(pp.filter(pl.col("group") == g), None, "w_m") for g in GROUPS
                                 if pp.filter(pl.col("group") == g).height >= 200}}
    inner = with_weights(pre.filter((pl.col("mid") > 0.02 + 1e-9) & (pl.col("mid") < 0.98 - 1e-9)))
    rob["excl_extreme_buckets"] = flb_measures(inner, None, "w_m")
    rob["quote_age_le_24h"] = flb_measures(with_weights(pre.filter(pl.col("quote_age_h") <= 24)), None, "w_m")
    rob["spread_le_5c"] = flb_measures(with_weights(pre.filter(pl.col("spread") <= 0.05 + 1e-9)), None, "w_m")
    rob["design_weighted"] = flb_measures(pre.with_columns((pl.col("w_m") / pl.col("p_event")).alias("w_d")), None, "w_d")
    rob["by_sched_src"] = {src: flb_measures(with_weights(pre.filter(pl.col("sched_src") == src)), None, "w_m")
                           for src in pre["sched_src"].unique().to_list() if pre.filter(pl.col("sched_src") == src).height >= 200}
    y, x, w = pre["y"].to_numpy(), logit(pre["mid"].to_numpy()), pre["w_m"].to_numpy()
    ls = S.filter(pl.col("q_mid") <= LONGSHOT + 1e-9)
    rob["clustering"] = {}
    for lab, col in (("event", "event_ticker"), ("market", "ticker"), ("series", "series_ticker"), ("month", "month")):
        f = logit_fit(y, x, pre[col].to_numpy(), w)
        c = clustered(ls["r_mid"].to_numpy(), ls[col].to_numpy(), ls["w_m"].to_numpy())
        rob["clustering"][lab] = {"b": f["b"], "se_b": f["se_b"], "t_b1": f["t_b1"], "clusters": f["clusters"],
                                  "ls_r_mid": c["mean"], "ls_se": c["se"], "ls_t": c["t"], "ls_clusters": c["clusters"]}
    _write("robustness", rob)
    summary["robustness"] = rob
    (d / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1))
    print("analyze done", flush=True)


# ---- CLI -----------------------------------------------------------------------------------------
# ---- LaTeX table snippets (pasted into paper_flb/sections/*_xs.tex) --------------------------------
def _f(x: Any, scale: float = 100.0, d: int = 1, sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    v = x * scale
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return f"${s.replace('-', '-')}$" if sign else s


def _t(x: Any) -> str:
    return "" if x is None else f"(${x:.1f}$)"


def _int(n: Any) -> str:
    return "--" if n is None else f"{int(n):,}".replace(",", "{,}")


def _m(d: dict | None, key: str = "mean") -> Any:
    return None if not d else d.get(key)


def _beta_cells(m: dict) -> tuple[str, str]:
    lg = m.get("logit") or {}
    if lg.get("b") is None:
        return "--", ""
    return f"{lg['b']:.3f}", _t(lg.get("t_b1"))


def _measure_rows(label: str, m: dict) -> list[str]:
    b, tb = _beta_cells(m)
    est = [label, _int(m.get("markets")), _int(m.get("events")), b,
           _f(_m(m.get("ls_r_mid"))), _f(_m(m.get("fav_r_mid"))), _f(_m(m.get("spread_mid"), "diff")),
           _f(_m(m.get("ls_r_net"))), _f(_m(m.get("fav_r_net"))), _f(_m(m.get("spread_net"), "diff"))]
    ts = ["", "", "", tb, _t(_m(m.get("ls_r_mid"), "t")), _t(_m(m.get("fav_r_mid"), "t")),
          _t(_m(m.get("spread_mid"), "t")), _t(_m(m.get("ls_r_net"), "t")), _t(_m(m.get("fav_r_net"), "t")),
          _t(_m(m.get("spread_net"), "t"))]
    return [" & ".join(est) + r" \\", " & ".join(ts) + r" \\[2pt]"]


def latex() -> None:
    d = out_dir()
    S = json.loads((d / "summary.json").read_text())
    T = {n: json.loads((d / "tables" / f"{n}.json").read_text())
         for n in ("calibration", "term", "structure", "liquidity", "calendar", "limits", "robustness")}
    L = d / "latex"
    L.mkdir(exist_ok=True)
    head = (r"Group & Markets & Events & $\hat\beta$ & \LS & \FAV & \LS$-$\FAV & \LS & \FAV & \LS$-$\FAV \\" "\n"
            r" & & & & \multicolumn{3}{c}{gross, at mid} & \multicolumn{3}{c}{net of taker fee, at ask} \\" "\n"
            r"\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
    out = []
    for wkey, title in (("mkt", "A. Market-weighted"), ("vol", "B. Volume-weighted (contracts traded in the 7 days to $t$)")):
        out.append(r"\midrule" + "\n" + rf"\multicolumn{{10}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for g in ("All", *GROUPS, "Sports in-play"):
            if g in S["measures_group"]:
                lab = "All groups" if g == "All" else g.replace("&", r"\&")
                out.extend(_measure_rows(lab, S["measures_group"][g][wkey]))
    (L / "tab_group.tex").write_text("% source: data/research/flb_xs/summary.json:measures_group\n" + head + "\n" + "\n".join(out) + "\n")

    rows = []
    for r in T["calibration"]["All"]["mkt"]:
        rows.append(" & ".join([r["bucket"], _int(r["n"]), _int(r["clusters"]), _f(r["q_mid"], 100, 2, False),
                                _f(r["win"], 100, 2, False), _f(_m(r["r_mid"])) + " " + _t(_m(r["r_mid"], "t")),
                                _f(_m(r.get("r_ask"))), _f(_m(r.get("r_net"))) + " " + _t(_m(r.get("r_net"), "t"))]) + r" \\")
    (L / "tab_bucket.tex").write_text("% source: data/research/flb_xs/tables/calibration.json:All.mkt\n" + "\n".join(rows) + "\n")

    # term structure: beta by lead (rows) x group (columns), t(beta=1) below
    cols = ["All", *GROUPS[:-1]]          # "Other" is too thin for lead-by-lead cells
    by = {(r["group"], r["lead_h"]): r for r in T["term"]}
    rows = []
    def tkey(g: str, ld: float) -> tuple[str | None, float]:
        # sports: events with a start time in the ticker only (date-only events have no exact horizon)
        if g == "Sports":
            return "Sports (exact start)", ld
        return (g if ld > 0 else None), ld

    for ld in (*LEADS_H, *INPLAY_H):
        if not any(tkey(g, ld) in by for g in cols):
            continue
        est, ts = [LEAD_LABEL[ld]], [""]
        for g in cols:
            key = tkey(g, ld)
            r = by.get(key)
            b, tb = _beta_cells(r) if r else ("", "")
            est.append(b)
            ts.append(tb)
        rows += [" & ".join(est) + r" \\", " & ".join(ts) + r" \\[1pt]"]
    (L / "tab_term.tex").write_text("% source: data/research/flb_xs/tables/term.json\n" + "\n".join(rows) + "\n")
    # term structure of the longshot return at mid
    rows = []
    for ld in (*LEADS_H, *INPLAY_H):
        est, ts = [LEAD_LABEL[ld]], [""]
        hit = False
        for g in cols:
            r = by.get(tkey(g, ld))
            if r and r.get("ls_r_mid"):
                hit = True
            est.append(_f(_m(r.get("ls_r_mid"))) if r else "")
            ts.append(_t(_m(r.get("ls_r_mid"), "t")) if r else "")
        if hit:
            rows += [" & ".join(est) + r" \\", " & ".join(ts) + r" \\[1pt]"]
    (L / "tab_term_ls.tex").write_text("% source: data/research/flb_xs/tables/term.json (ls_r_mid)\n" + "\n".join(rows) + "\n")

    st = T["structure"]
    rows = []
    names = {"binary": "Binary", "multi": "Multi-outcome", "ladder": "Ladder leg", "deadline": "Deadline"}
    for s_ in STRUCTURES:
        if s_ not in st:
            continue
        m = st[s_]["all"]["mkt"]
        rg = st[s_]["regions"]
        b, tb = _beta_cells(m)
        est = [names[s_], _int(m["markets"]), _int(m["events"]), b,
               _f(_m(rg["LS <=10c"]["r_mid"])), _f(_m(rg["body 35-65c"]["r_mid"])), _f(_m(rg["FAV >=90c"]["r_mid"])),
               _f(_m(rg["LS <=10c"]["r_net"])), _f(_m(rg["FAV >=90c"]["r_net"]))]
        ts = ["", "", "", tb, _t(_m(rg["LS <=10c"]["r_mid"], "t")), _t(_m(rg["body 35-65c"]["r_mid"], "t")),
              _t(_m(rg["FAV >=90c"]["r_mid"], "t")), _t(_m(rg["LS <=10c"]["r_net"], "t")), _t(_m(rg["FAV >=90c"]["r_net"], "t"))]
        rows += [" & ".join(est) + r" \\", " & ".join(ts) + r" \\[2pt]"]
    (L / "tab_structure.tex").write_text("% source: data/research/flb_xs/tables/structure.json\n" + "\n".join(rows) + "\n")

    # liquidity terciles and robustness
    rows = []
    for var, lab in (("t_vol", "7-day volume"), ("t_spread", "Quoted spread"), ("t_oi", "Open interest")):
        for r in T["liquidity"][var]:
            b, tb = _beta_cells(r)
            rows.append(" & ".join([lab if r["tercile"] == 1 else "", ["Low", "Mid", "High"][r["tercile"] - 1],
                                    _int(r["markets"]), b + " " + tb, _f(_m(r.get("ls_r_mid"))) + " " + _t(_m(r.get("ls_r_mid"), "t")),
                                    _f(_m(r.get("fav_r_mid"))) + " " + _t(_m(r.get("fav_r_mid"), "t")),
                                    _f(_m(r.get("ls_r_net"))) + " " + _t(_m(r.get("ls_r_net"), "t"))]) + r" \\")
    (L / "tab_liquidity.tex").write_text("% source: data/research/flb_xs/tables/liquidity.json\n" + "\n".join(rows) + "\n")
    rob = T["robustness"]
    rows = []
    items = [("Full sample", S["measures_group"]["All"]["mkt"]), ("First half", rob["first_half"]["all"]),
             ("Second half", rob["second_half"]["all"]), (r"Excluding 1--2\cents{} and 99\cents", rob["excl_extreme_buckets"]),
             (r"Spread $\le 5\cents$", rob["spread_le_5c"]), ("Quote age $\\le 24$ h", rob["quote_age_le_24h"]),
             ("Design-weighted", rob["design_weighted"])]
    items += [(f"Scheduled time: {k}", v) for k, v in sorted(rob["by_sched_src"].items())]
    for lab, m in items:
        b, tb = _beta_cells(m)
        rows.append(" & ".join([lab, _int(m.get("markets")), b + " " + tb,
                                _f(_m(m.get("ls_r_mid"))) + " " + _t(_m(m.get("ls_r_mid"), "t")),
                                _f(_m(m.get("fav_r_mid"))) + " " + _t(_m(m.get("fav_r_mid"), "t")),
                                _f(_m(m.get("ls_r_net"))) + " " + _t(_m(m.get("ls_r_net"), "t"))]) + r" \\")
    (L / "tab_robust.tex").write_text("% source: data/research/flb_xs/tables/robustness.json, summary.json\n" + "\n".join(rows) + "\n")
    rows = []
    for lab, r in rob["clustering"].items():
        rows.append(" & ".join([lab.capitalize(), _int(r["clusters"]), f"{r['b']:.3f}", f"{r['se_b']:.3f}", _t(r["t_b1"]),
                                _f(r["ls_r_mid"]), _f(r["ls_se"], 100, 1, False), _t(r["ls_t"])]) + r" \\")
    (L / "tab_cluster.tex").write_text("% source: data/research/flb_xs/tables/robustness.json:clustering\n" + "\n".join(rows) + "\n")
    # per-group gross return at mid by bucket (appendix)
    cal = T["calibration"]
    gcols = [g for g in ("All", *GROUPS[:-1], "Sports in-play") if g in cal]     # "Other" too thin
    rows = []
    for b_ in BUCKET_LABELS:
        est, ts = [b_], [""]
        for g in gcols:
            r = next((x for x in cal[g]["mkt"] if x["bucket"] == b_), None)
            est.append(_f(_m(r["r_mid"])) if r and r["n"] >= 30 else "")
            ts.append(_t(_m(r["r_mid"], "t")) if r and r["n"] >= 30 else "")
        rows += [" & ".join(est) + r" \\", " & ".join(ts) + r" \\[1pt]"]
    (L / "tab_bucket_group.tex").write_text("% source: data/research/flb_xs/tables/calibration.json (r_mid, market-weighted; cells with >= 30 sides)\n"
                                            + "\n".join(rows) + "\n")
    # calendar (pooled) by quarter
    rows = []
    for r in T["calendar"]["cells"]:
        if r["group"] != "All":
            continue
        b, tb = _beta_cells(r)
        rows.append(" & ".join([r["qtr"], _int(r["markets"]), _int(r["events"]), b + " " + tb,
                                _f(_m(r.get("ls_r_mid"))) + " " + _t(_m(r.get("ls_r_mid"), "t")),
                                _f(_m(r.get("ls_r_net"))) + " " + _t(_m(r.get("ls_r_net"), "t"))]) + r" \\")
    (L / "tab_calendar.tex").write_text("% source: data/research/flb_xs/tables/calendar.json (group All)\n" + "\n".join(rows) + "\n")

    # coverage: frame, eligible, analysed, by group; analysed markets by year
    cov = S["coverage"]
    fr = {r["group"]: r for r in cov["frame"]}
    an = {r["group"]: r for r in cov["by_group"]}
    years = sorted({r["year"] for r in cov["by_year"]})
    by_y = {(r["group"], r["year"]): r["markets"] for r in cov["by_year"]}
    rows = []
    for g in GROUPS:
        if g not in fr:
            continue
        a = an.get(g, {})
        rows.append(" & ".join([g.replace("&", r"\&"), _int(fr[g]["frame_markets"]), _int(fr[g]["eligible_markets"]),
                                _int(a.get("markets")), _int(a.get("events")), str(a.get("first", "--")), str(a.get("last", "--")),
                                *[_int(by_y.get((g, y))) if by_y.get((g, y)) else "--" for y in years]]) + r" \\")
    tot = [r"\midrule All", _int(sum(r["frame_markets"] for r in cov["frame"])),
           _int(sum(r["eligible_markets"] or 0 for r in cov["frame"])), _int(cov["markets"]), _int(cov["events"]),
           str(cov["first_month"]), str(cov["last_month"]),
           *[_int(sum(v for (g, y), v in by_y.items() if y == yy)) for yy in years]]
    rows.append(" & ".join(tot) + r" \\")
    (L / "tab_coverage.tex").write_text("% source: data/research/flb_xs/summary.json:coverage (frame_cells.parquet)\n"
                                        f"% years: {' '.join(years)}\n" + "\n".join(rows) + "\n")

    # multi-outcome events: overround, Shin z, by rank, field
    mo = T["structure"].get("multi_outcome", {})
    rows = []
    for lab, key in (("Two-leg events", "two_leg"), (r"Multi-outcome events ($\geq 3$ legs)", "multi")):
        if key not in mo:
            continue
        r = mo[key]
        rows.append(" & ".join([lab, _int(r["events"]), _int(r["event_leads"]), f"{r['legs_median']:.0f}",
                                _f(_m(r["over_ask"]), 100, 1) + " " + _t(_m(r["over_ask"], "t")),
                                _f(_m(r["over_mid"]), 100, 1) + " " + _t(_m(r["over_mid"], "t")),
                                _f(_m(r["under_bid"]), 100, 1),
                                f"{_m(r['shin_z']):.3f}", f"{r['shin_z_median']:.3f}"]) + r" \\")
    (L / "tab_overround.tex").write_text("% source: data/research/flb_xs/tables/structure.json:multi_outcome\n" + "\n".join(rows) + "\n")
    rows = []
    if "multi" in mo:
        for r in mo["multi"]["by_rank"]:
            rows.append(" & ".join([r["rank_class"], _int(r["n"]), _f(r["mid"], 100, 1, False), _f(r["win"], 100, 1, False),
                                    _f(_m(r["e_mid"]), 100, 2) + " " + _t(_m(r["e_mid"], "t")),
                                    _f(_m(r["e_norm"]), 100, 2) + " " + _t(_m(r["e_norm"], "t")),
                                    _f(_m(r["e_shin"]), 100, 2) + " " + _t(_m(r["e_shin"], "t"))]) + r" \\")
        fl = mo["multi"].get("field")
        if fl:
            rows.append(r"\midrule" + "\n" + " & ".join([r"Field ($\leq 10\cents$ legs)", _int(fl["n"]), _f(fl["price"], 100, 1, False),
                                                 _f(fl["win"], 100, 1, False),
                                                 _f(-_m(fl["price_minus_win"]), 100, 2) + " " + _t(-_m(fl["price_minus_win"], "t")),
                                                 _f(-_m(fl["norm_minus_win"]), 100, 2) + " " + _t(-_m(fl["norm_minus_win"], "t")),
                                                 _f(-_m(fl["shin_minus_win"]), 100, 2) + " " + _t(-_m(fl["shin_minus_win"], "t"))]) + r" \\")
    (L / "tab_multi_rank.tex").write_text("% source: data/research/flb_xs/tables/structure.json:multi_outcome.multi (by_rank, field)\n"
                                          + "\n".join(rows) + "\n")
    rows = []
    for r in mo.get("by_group", []):
        rows.append(" & ".join([r["group"].replace("&", r"\&"), r["legs"], _int(r["events"]), _int(r["event_leads"]),
                                f"{_m(r['shin_z']):.3f}", f"{r['shin_z_median']:.3f}",
                                _f(_m(r["over_ask"]), 100, 1), _f(_m(r["over_mid"]), 100, 1)]) + r" \\")
    (L / "tab_shin_group.tex").write_text("% source: data/research/flb_xs/tables/structure.json:multi_outcome.by_group\n" + "\n".join(rows) + "\n")
    rows = []
    for r in T["structure"].get("multi_by_legs", []):
        b, tb = _beta_cells(r)
        rows.append(" & ".join([r["legs"], _int(r["markets"]), _int(r["events"]), b + " " + tb,
                                _f(_m(r.get("ls_r_mid"))) + " " + _t(_m(r.get("ls_r_mid"), "t"))]) + r" \\")
    (L / "tab_multi_legs.tex").write_text("% source: data/research/flb_xs/tables/structure.json:multi_by_legs\n" + "\n".join(rows) + "\n")

    # limits: tick floor decomposition and the other side of longshots
    fl = T["limits"]["floor"]
    rows = []
    names = {"all_1_2c": r"All 1--2\cents{} sides (1\cents{} grid)", "pinned_at_floor": r"\quad at the floor (mid 0.5\cents)",
             "above_floor": r"\quad above the floor", "subcent_grid_0_2c": r"0--2\cents{} sides, 0.1\cents{} grid"}
    for k_, lab in names.items():
        r = fl["decomposition"].get(k_)
        if not r:
            continue
        rows.append(" & ".join([lab, _int(r["n"]), _f(r["q_mid"], 100, 2, False), _f(r["win"], 100, 2, False),
                                _f(r["loss_pc"], 100, 2, False), _f(r["mechanical_pc"], 100, 2, False),
                                "--" if r["mechanical_share"] is None else f"{100 * r['mechanical_share']:.0f}\\%",
                                _f(_m(r["r_mid"])) + " " + _t(_m(r["r_mid"], "t"))]) + r" \\")
    r = fl["decomposition"].get("taker_ask_1_2c")
    if r:
        rows.append(" & ".join([r"Taker at a 1--2\cents{} ask", _int(r["n"]), _f(r["q_ask"], 100, 2, False), _f(r["win"], 100, 2, False),
                                _f(r["loss_pc"], 100, 2, False), _f(r["mechanical_pc"], 100, 2, False),
                                "--" if r["mechanical_share"] is None else f"{100 * r['mechanical_share']:.0f}\\%",
                                _f(_m(r["r_net"])) + " " + _t(_m(r["r_net"], "t"))]) + r" \\")
    (L / "tab_floor.tex").write_text("% source: data/research/flb_xs/tables/limits.json:floor\n" + "\n".join(rows) + "\n")
    rows = []
    for g, r in T["limits"]["other_side"].items():
        rows.append(" & ".join(["All groups" if g == "all" else g.replace("&", r"\&"), _int(r["n"]),
                                _f(_m(r["mid_gross"]), 100, 2) + " " + _t(_m(r["mid_gross"], "t")),
                                _f(_m(r["taker_gross"]), 100, 2) + " " + _t(_m(r["taker_gross"], "t")),
                                _f(_m(r["taker_net"]), 100, 2) + " " + _t(_m(r["taker_net"], "t")),
                                _f(_m(r["maker_net"]), 100, 2) + " " + _t(_m(r["maker_net"], "t"))]) + r" \\")
    (L / "tab_other_side.tex").write_text("% source: data/research/flb_xs/tables/limits.json:other_side\n" + "\n".join(rows) + "\n")
    (L / "macros.tex").write_text(number_macros(S, T))
    (d / "verdict.md").write_text(verdict_text(S, T))
    print(f"latex snippets -> {L}", flush=True)


_GKEY = {"All": "All", "Sports": "Sports", "Weather": "Weather", "Politics": "Politics", "Entertainment": "Ent",
         "Mentions": "Mentions", "Science & Tech": "Sci", "Other": "Other", "Sports in-play": "Inplay",
         "Sports (exact start)": "SportsStart"}
_LKEY = {1: "OneH", 6: "SixH", 24: "OneD", 72: "ThreeD", 168: "SevenD", 720: "ThirtyD",
         -0.5: "InHalf", -1.0: "InOne", -2.0: "InTwo"}
_SKEY = {"binary": "Binary", "multi": "Multi", "ladder": "Ladder", "deadline": "Deadline"}


def number_macros(S: dict[str, Any], T: dict[str, Any]) -> str:
    """``\\newcommand{\\xs...}`` for every number quoted in the xs prose, with its source."""
    out: list[str] = ["% Numbers quoted in the xs text (generated by `python -m albert.research.flb_xs run latex`)."]

    def put(name: str, val: str, src: str) -> None:
        out.append(f"\\newcommand{{\\xs{name}}}{{{val}}} % source: data/research/flb_xs/{src}")

    def pct(x: Any, d: int = 1, sign: bool = False) -> str:
        if x is None:
            return "--"
        v = 100 * x
        s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
        return s.replace("-", r"\ensuremath{-}").replace("+", r"\ensuremath{+}")

    def tt(x: Any) -> str:
        return "--" if x is None else f"{x:.1f}".replace("-", r"\ensuremath{-}")

    def num(x: Any, d: int = 2) -> str:
        return "--" if x is None else f"{x:.{d}f}".replace("-", r"\ensuremath{-}")

    def measures(prefix: str, m: dict, src: str) -> None:
        lg = m.get("logit") or {}
        put(prefix + "Beta", num(lg.get("b")), src)
        put(prefix + "BetaT", tt(lg.get("t_b1")), src)
        put(prefix + "Alpha", num(lg.get("a")), src)
        put(prefix + "AlphaT", tt(lg.get("t_a0")), src)
        put(prefix + "Mkts", _int(m.get("markets")), src)
        put(prefix + "Events", _int(m.get("events")), src)
        for k, lab in (("ls_r_mid", "LsMid"), ("fav_r_mid", "FavMid"), ("ls_r_net", "LsNet"), ("fav_r_net", "FavNet")):
            put(prefix + lab, pct(_m(m.get(k))), src)
            put(prefix + lab + "T", tt(_m(m.get(k), "t")), src)
        for k, lab in (("spread_mid", "SpreadMid"), ("spread_net", "SpreadNet")):
            put(prefix + lab, pct(_m(m.get(k), "diff")), src)
            put(prefix + lab + "T", tt(_m(m.get(k), "t")), src)

    cov = S["coverage"]
    put("FrameMkts", _int(cov["frame_total_markets"]), "summary.json:coverage")
    put("EligibleMkts", _int(sum(r["eligible_markets"] or 0 for r in cov["frame"])), "summary.json:coverage")
    put("Mkts", _int(cov["markets"]), "summary.json:coverage")
    put("Events", _int(cov["events"]), "summary.json:coverage")
    put("Series", _int(cov["series"]), "summary.json:coverage")
    put("PreQuotes", _int(cov["pre_rows"]), "summary.json:coverage")
    put("InplayQuotes", _int(cov["inplay_rows"]), "summary.json:coverage")
    put("PanelRows", _int(cov["panel_rows"]), "summary.json:coverage")
    put("UsableShare", pct(cov["usable_share"], 0), "summary.json:coverage")
    put("SampledMkts", _int(cov.get("sampled_markets")), "summary.json:coverage")
    put("SampledEvents", _int(cov.get("sampled_events")), "summary.json:coverage")
    for g, mm in S["measures_group"].items():
        measures(_GKEY[g], mm["mkt"], "summary.json:measures_group." + g + ".mkt")
        measures(_GKEY[g] + "Vol", mm["vol"], "summary.json:measures_group." + g + ".vol")
    cal = T["calibration"]["All"]["mkt"]
    for r in cal:
        key = {"1-2c": "BOne", "3-5c": "BThree", "6-10c": "BSix", "11-20c": "BEleven", "91-95c": "BNinetyOne",
               "96-98c": "BNinetySix", "99c": "BNinetyNine"}.get(r["bucket"])
        if key:
            put(key + "Q", num(100 * r["q_mid"]), "tables/calibration.json:All.mkt")
            put(key + "Win", num(100 * r["win"]), "tables/calibration.json:All.mkt")
            put(key + "Mid", pct(_m(r["r_mid"])), "tables/calibration.json:All.mkt")
            put(key + "MidT", tt(_m(r["r_mid"], "t")), "tables/calibration.json:All.mkt")
            put(key + "Net", pct(_m(r.get("r_net"))), "tables/calibration.json:All.mkt")
            put(key + "NetT", tt(_m(r.get("r_net"), "t")), "tables/calibration.json:All.mkt")
            put(key + "N", _int(r["n"]), "tables/calibration.json:All.mkt")
    for r in T["term"]:
        if r["group"] in _GKEY and r["lead_h"] in _LKEY:
            p_ = "Term" + _GKEY[r["group"]] + _LKEY[r["lead_h"]]
            lg = r.get("logit") or {}
            put(p_ + "Beta", num(lg.get("b")), "tables/term.json")
            put(p_ + "BetaT", tt(lg.get("t_b1")), "tables/term.json")
            put(p_ + "LsMid", pct(_m(r.get("ls_r_mid"))), "tables/term.json")
            put(p_ + "LsMidT", tt(_m(r.get("ls_r_mid"), "t")), "tables/term.json")
            put(p_ + "Mkts", _int(r.get("markets")), "tables/term.json")
    st = T["structure"]
    for s_, key in _SKEY.items():
        if s_ in st:
            measures("St" + key, st[s_]["all"]["mkt"], f"tables/structure.json:{s_}.all.mkt")
            for reg, rk in (("LS <=10c", "Ls"), ("body 35-65c", "Body"), ("FAV >=90c", "Fav")):
                r = st[s_]["regions"][reg]
                put("St" + key + "Reg" + rk + "Mid", pct(_m(r["r_mid"])), f"tables/structure.json:{s_}.regions")
                put("St" + key + "Reg" + rk + "MidT", tt(_m(r["r_mid"], "t")), f"tables/structure.json:{s_}.regions")
                put("St" + key + "Reg" + rk + "Pc", pct(_m(r["pc_mid"]), 2), f"tables/structure.json:{s_}.regions")
                put("St" + key + "Reg" + rk + "PcT", tt(_m(r["pc_mid"], "t")), f"tables/structure.json:{s_}.regions")
    mo = st.get("multi_outcome", {})
    for k, key in (("two_leg", "Two"), ("multi", "Multi")):
        if k in mo:
            r = mo[k]
            src = f"tables/structure.json:multi_outcome.{k}"
            put("Me" + key + "Events", _int(r["events"]), src)
            put("Me" + key + "OverAsk", pct(_m(r["over_ask"])), src)
            put("Me" + key + "OverAskT", tt(_m(r["over_ask"], "t")), src)
            put("Me" + key + "OverMid", pct(_m(r["over_mid"])), src)
            put("Me" + key + "OverMidT", tt(_m(r["over_mid"], "t")), src)
            put("Me" + key + "UnderBid", pct(_m(r["under_bid"])), src)
            put("Me" + key + "ZMean", num(_m(r["shin_z"]), 3), src)
            put("Me" + key + "ZMedian", num(r["shin_z_median"], 3), src)
            put("Me" + key + "Legs", num(r["legs_median"], 0), src)
            for kk, b in r.get("brier", {}).items():
                put("Me" + key + "Brier" + kk.capitalize(), num(b, 4), src)
            fl = r.get("field")
            if fl:
                put("Me" + key + "FieldPrice", pct(fl["price"]), src)
                put("Me" + key + "FieldWin", pct(fl["win"]), src)
                put("Me" + key + "FieldGap", pct(_m(fl["price_minus_win"])), src)
                put("Me" + key + "FieldGapT", tt(_m(fl["price_minus_win"], "t")), src)
                put("Me" + key + "FieldShinGap", pct(_m(fl["shin_minus_win"])), src)
                put("Me" + key + "FieldShinGapT", tt(_m(fl["shin_minus_win"], "t")), src)
            for rr in r.get("by_rank", []):
                rk = {"1 (favourite)": "RankOne", "2": "RankTwo", "3-5": "RankThree", "6+": "RankSix"}[rr["rank_class"]]
                put("Me" + key + rk + "Emid", pct(_m(rr["e_mid"]), 2), src)
                put("Me" + key + rk + "EmidT", tt(_m(rr["e_mid"], "t")), src)
                put("Me" + key + rk + "Eshin", pct(_m(rr["e_shin"]), 2), src)
                put("Me" + key + rk + "EshinT", tt(_m(rr["e_shin"], "t")), src)
    for r in st.get("multi_by_legs", []):
        key = {"3-5": "Three", "6-10": "Six", "11-20": "Eleven", "21+": "TwentyOne"}[r["legs"]]
        put("MultiLegs" + key + "Beta", num((r.get("logit") or {}).get("b")), "tables/structure.json:multi_by_legs")
        put("MultiLegs" + key + "BetaT", tt((r.get("logit") or {}).get("t_b1")), "tables/structure.json:multi_by_legs")
    for r in mo.get("by_group", []):
        key = _GKEY.get(r["group"], "Other") + ("Two" if r["legs"] == "2" else "Multi")
        put("Z" + key, num(r["shin_z_median"], 3), "tables/structure.json:multi_outcome.by_group")
    for var, key in (("t_vol", "Vol"), ("t_spread", "Spread"), ("t_oi", "Oi")):
        for r in T["liquidity"][var]:
            p_ = "Liq" + key + ["Low", "Mid", "High"][r["tercile"] - 1]
            lg = r.get("logit") or {}
            put(p_ + "Beta", num(lg.get("b")), "tables/liquidity.json")
            put(p_ + "BetaT", tt(lg.get("t_b1")), "tables/liquidity.json")
            put(p_ + "LsMid", pct(_m(r.get("ls_r_mid"))), "tables/liquidity.json")
            put(p_ + "LsNet", pct(_m(r.get("ls_r_net"))), "tables/liquidity.json")
            put(p_ + "Median", num(r.get("median_value"), 3), "tables/liquidity.json")
    calc = [r for r in T["calendar"]["cells"] if r["group"] == "All"]
    if calc:
        put("CalFirstQtr", calc[0]["qtr"], "tables/calendar.json")
        put("CalLastQtr", calc[-1]["qtr"], "tables/calendar.json")
        put("CalNQtr", str(len(calc)), "tables/calendar.json")
        bs = [(r["logit"] or {}).get("b") for r in calc if (r.get("logit") or {}).get("b") is not None]
        put("CalBetaMin", num(min(bs)), "tables/calendar.json")
        put("CalBetaMax", num(max(bs)), "tables/calendar.json")
        above = sum(1 for r in calc if (r.get("logit") or {}).get("b") is not None and r["logit"]["b"] > 1)
        put("CalBetaAboveOne", str(above), "tables/calendar.json")
        for lab, sel in (("Recent", [r for r in calc if r["qtr"] >= "2025Q1"]), ("Early", [r for r in calc if r["qtr"] < "2025Q1"])):
            bs = [r["logit"]["b"] for r in sel if (r.get("logit") or {}).get("b") is not None]
            if bs:
                put(f"Cal{lab}Min", num(min(bs)), "tables/calendar.json")
                put(f"Cal{lab}Max", num(max(bs)), "tables/calendar.json")
                put(f"Cal{lab}N", str(len(bs)), "tables/calendar.json")
    venue = [v for v in T["calendar"]["venue"] if "2021Q3" <= v["qtr"] <= "2026Q3"]
    if venue:
        early = [v["markets"] for v in venue if v["qtr"].startswith("2022")]
        peak = max(venue, key=lambda v: v["markets"])
        put("VenueEarlyMkts", _int(round(sum(early) / max(len(early), 1))), "tables/calendar.json:venue (frame markets per quarter, 2022 mean)")
        put("VenuePeakMkts", _int(peak["markets"]), "tables/calendar.json:venue")
        put("VenuePeakQtr", peak["qtr"], "tables/calendar.json:venue")
    main6 = [g for g in GROUPS[:-1] if g in S["measures_group"]]
    for wk, lab in (("mkt", ""), ("vol", "Vol")):
        bs = [S["measures_group"][g][wk]["logit"]["b"] for g in main6 if S["measures_group"][g][wk]["logit"].get("b")]
        put(f"Group{lab}BetaMin", num(min(bs)), "summary.json:measures_group")
        put(f"Group{lab}BetaMax", num(max(bs)), "summary.json:measures_group")
    lim = T["limits"]
    shares = [r["mispricing_pc"] / r["band_taker"] for r in lim["band"] if r["bucket"] in ("1-2c", "3-5c", "6-10c")]
    if shares:
        put("BandShareMin", pct(min(shares), 0), "tables/limits.json:band")
        put("BandShareMax", pct(max(shares), 0), "tables/limits.json:band")
    fl = lim["floor"]
    put("FloorShareYes", pct(fl["share_quotes_yes_mid_le_2c"], 0), "tables/limits.json:floor")
    put("FloorShareEither", pct(fl["share_quotes_either_side_le_2c"], 0), "tables/limits.json:floor")
    put("FloorSharePinned", pct(fl["share_quotes_pinned"], 0), "tables/limits.json:floor")
    put("SubcentShare", pct(fl["subcent_markets_share"], 1), "tables/limits.json:floor")
    for k, key in (("all_1_2c", "All"), ("pinned_at_floor", "Pinned"), ("above_floor", "Above"), ("taker_ask_1_2c", "Taker"),
                   ("subcent_grid_0_2c", "Subcent")):
        r = fl["decomposition"].get(k)
        if r:
            put("Floor" + key + "Win", num(100 * r["win"], 2), "tables/limits.json:floor.decomposition")
            put("Floor" + key + "Q", num(100 * r["q"], 2), "tables/limits.json:floor.decomposition")
            put("Floor" + key + "Loss", num(100 * r["loss_pc"], 2), "tables/limits.json:floor.decomposition")
            put("Floor" + key + "Mech", num(100 * r["mechanical_pc"], 2), "tables/limits.json:floor.decomposition")
            put("Floor" + key + "MechShare", pct(r["mechanical_share"], 0), "tables/limits.json:floor.decomposition")
            put("Floor" + key + "N", _int(r["n"]), "tables/limits.json:floor.decomposition")
    for g, r in lim["other_side"].items():
        key = "All" if g == "all" else _GKEY.get(g, "Other")
        for k, lab in (("taker_net", "TakerNet"), ("maker_net", "MakerNet"), ("mid_gross", "Mid"), ("taker_gross", "TakerGross")):
            put("Other" + key + lab, pct(_m(r[k]), 2), "tables/limits.json:other_side")
            put("Other" + key + lab + "T", tt(_m(r[k], "t")), "tables/limits.json:other_side")
    for r in lim["band"]:
        key = {"1-2c": "BOne", "3-5c": "BThree", "6-10c": "BSix", "96-98c": "BNinetySix", "99c": "BNinetyNine"}.get(r["bucket"])
        if key:
            put("Band" + key + "Mis", num(100 * r["mispricing_pc"], 2), "tables/limits.json:band")
            put("Band" + key + "Cost", num(100 * r["band_taker"], 2), "tables/limits.json:band")
            put("Band" + key + "CostCap", num(100 * r["band_taker_capital"], 2), "tables/limits.json:band")
    inside = sum(1 for r in lim["band"] if abs(r["mispricing_pc"]) <= r["band_taker"])
    put("BandInside", str(inside), "tables/limits.json:band")
    put("BandN", str(len(lim["band"])), "tables/limits.json:band")
    rob = T["robustness"]
    for k, key in (("first_half", "First"), ("second_half", "Second")):
        measures("Rob" + key, rob[k]["all"], f"tables/robustness.json:{k}.all")
        n_ok = sum(1 for mm in rob[k]["by_group"].values() if (mm.get("logit") or {}).get("b") and mm["logit"]["b"] > 1)
        put("Rob" + key + "GroupsAboveOne", str(n_ok), f"tables/robustness.json:{k}.by_group")
        put("Rob" + key + "Groups", str(len(rob[k]["by_group"])), f"tables/robustness.json:{k}.by_group")
    put("RobSplit", rob["first_half"]["split_ts"][:10], "tables/robustness.json:first_half.split_ts")
    for k, key in (("excl_extreme_buckets", "Inner"), ("spread_le_5c", "Spread"), ("quote_age_le_24h", "Fresh"),
                   ("design_weighted", "Design")):
        measures("Rob" + key, rob[k], f"tables/robustness.json:{k}")
    for k, r in rob["clustering"].items():
        put("Clu" + k.capitalize() + "BetaT", tt(r["t_b1"]), "tables/robustness.json:clustering")
        put("Clu" + k.capitalize() + "LsT", tt(r["ls_t"]), "tables/robustness.json:clustering")
    return "\n".join(out) + "\n"


STEPS: dict[str, Any] = {"scan": scan, "sample": sample, "events": events, "candles": candles, "panel": panel,
                         "analyze": analyze, "latex": latex}


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m albert.research.flb_xs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("step", choices=sorted(STEPS))
    a = ap.parse_args(list(argv) if argv is not None else None)
    STEPS[a.step]()



def verdict_text(S: dict[str, Any], T: dict[str, Any]) -> str:
    """Short verdict (markdown) from summary.json and tables/*.json."""
    def m(d: dict | None, k: str = "mean", sc: float = 100.0) -> str:
        return "n/a" if not d or d.get(k) is None else f"{sc * d[k]:+.1f}"

    def t(d: dict | None, k: str = "t") -> str:
        return "n/a" if not d or d.get(k) is None else f"{d[k]:.1f}"

    def b(x: dict) -> str:
        lg = x.get("logit") or {}
        return "n/a" if lg.get("b") is None else f"{lg['b']:.2f} (t vs 1 = {lg['t_b1']:.1f})"

    c, A = S["coverage"], S["measures_group"]["All"]
    lines = [
        "# Verdict: cross-section of the favourite-longshot bias on Kalshi (workstream xs)",
        "",
        (f"Sample: {c['markets']:,} markets in {c['events']:,} events ({c['first_month']} to {c['last_month']}), "
        f"{c['pre_rows']:,} pre-event quotes at leads 1 h-30 d and {c['inplay_rows']:,} sports in-play quotes; "
        f"stratified from a frame of {c['frame_total_markets']:,} settled non-financial markets. "
        "Market-weighted, t clustered by event unless noted."),
        "",
        "## Headline",
        (f"- Calibration slope beta = {b(A['mkt'])}; volume-weighted {b(A['vol'])}. YES-only intercept "
        f"alpha = {A['mkt']['logit']['a']:+.2f} (t {A['mkt']['logit']['t_a0']:.1f})."),
        (f"- Longshots (q <= 10c): {m(A['mkt']['ls_r_mid'])}% per $ at the mid (t {t(A['mkt']['ls_r_mid'])}), "
        f"{m(A['mkt']['ls_r_net'])}% net of spread and taker fee (t {t(A['mkt']['ls_r_net'])})."),
        (f"- Favourites (q >= 90c): {m(A['mkt']['fav_r_mid'])}% at the mid (t {t(A['mkt']['fav_r_mid'])}), "
        f"{m(A['mkt']['fav_r_net'])}% net (t {t(A['mkt']['fav_r_net'])})."),
        "",
        "## By group (beta, longshot gross per $)",
    ]
    for g, mm in S["measures_group"].items():
        if g != "All":
            lines.append(f"- {g}: beta {b(mm['mkt'])}; LS {m(mm['mkt']['ls_r_mid'])}% (t {t(mm['mkt']['ls_r_mid'])})")
    lines += ["", "## By structure"]
    for s_ in STRUCTURES:
        if s_ in T["structure"]:
            x = T["structure"][s_]["all"]["mkt"]
            lines.append(f"- {s_}: beta {b(x)}; LS {m(x.get('ls_r_mid'))}% (t {t(x.get('ls_r_mid'))})")
    mo = T["structure"].get("multi_outcome", {})
    if "multi" in mo:
        r = mo["multi"]
        fl = r.get("field") or {}
        lines.append(f"- Multi-outcome events: overround at asks {m(r['over_ask'])}%, at mids {m(r['over_mid'])}%; "
                     f"median Shin z {r['shin_z_median']:.3f}; field (legs <= 10c) price minus win "
                     f"{m(fl.get('price_minus_win'))}c (t {t(fl.get('price_minus_win'))}), after Shin "
                     f"{m(fl.get('shin_minus_win'))}c.")
    lines += ["", "## Horizon (pooled beta by lead)"]
    lines.append("- " + "; ".join(f"{r['label']}: {b(r)}" for r in T["term"] if r["group"] == "All"))
    lines.append("- Sports, exact start: " + "; ".join(f"{r['label']}: {b(r)}" for r in T["term"]
                                                       if r["group"] == "Sports (exact start)"))
    fl = T["limits"]["floor"]["decomposition"]
    os_ = T["limits"]["other_side"].get("all", {})
    lines += ["", "## Mechanical vs behavioural",
              (f"- Tick floor: {100 * fl['all_1_2c']['mechanical_share']:.0f}% of the 1-2c loss per contract (mid) is "
              f"mechanical; {100 * fl['taker_ask_1_2c']['mechanical_share']:.0f}% for takers at a 1-2c ask."),
              f"- Without the 1-2c/99c buckets: beta {b(T['robustness']['excl_extreme_buckets'])}.",
              (f"- Buying the favourite side of every longshot: taker net {m(os_.get('taker_net'), sc=100)}% per $ "
              f"(t {t(os_.get('taker_net'))}); maker (fill at quote, upper bound) {m(os_.get('maker_net'))}% "
              f"(t {t(os_.get('maker_net'))})."),
              f"- Halves: first {b(T['robustness']['first_half']['all'])}, second {b(T['robustness']['second_half']['all'])}.",
              "",
              ("Reading: a large per-dollar, small per-contract bias at every horizon and in every group; about half "
              "of the cheapest bucket's loss is the tick floor and the rest comes from prices above it; the slope "
              "is unchanged without the floor buckets; the mispricing lies inside the taker no-arbitrage band, "
              "close to its edge, so it survives fees for buyers but not for an arbitrageur. Details: "
              "summary.json, tables/*.json; paper_flb/sections/results_xs.tex.")]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
