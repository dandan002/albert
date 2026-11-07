"""Forward paper test of the three fundamental-model leads (no orders are ever placed).

Every rule below is frozen from rounds 1-2; nothing is refitted here. Each snapshot records the
public Kalshi order book at the decision time ``t`` and the hypothetical bids the frozen rule
would rest, next to the same bids anchored on the plain (and the bias-corrected) mid as controls.
Scoring later fills each bid only when a public trade after ``t`` prints strictly THROUGH it
(the harness rule of ``fundamental.maker_fills``), until the order's end, and holds it to
settlement with the series maker fee.

Leads (``data/cache/forward/<lead>/<UTC stamp>.parquet``, one row per market):

* ``mentions`` (``mentions_maker``): Kalshi Mentions speech events (window series such as
  ``KXTRUMPSAY`` excluded) whose milestone ``start_date`` is 5-7 h away when first seen (the hourly
  loop sees each event once at 6-7 h). Fair value = the dev-calibrated mid ``fv_cal`` =
  sigmoid(-0.1156 + 1.0474 logit(mid)), clipped to [0.003, 0.997]; rule = one NO bid at
  floor(1 - fair - m) for m = 8c and 12c, post-only (capped one tick inside the YES bid), resting
  until start - 10 min. Control: the same NO bids around the raw mid. The study's outcome-blind
  exclusion is applied (the share of legs bid >= 97c at ``t`` more than 12 points above that share
  at start - 24 h, from one batch-candle request); the circuit-breaker variant (cancel the event
  at the first print >= 95c on a leg asked <= 80c at ``t``) is scored as ``+cb``.
* ``rain`` (``weather_maker`` / ``rain``): at 12:00-14:00 UTC (the loop runs 12:05), today's
  ``KXRAIN`` legs whose book is <= 3c wide. Model = the IEM-only ridge logit (``rain.IEM_VARIANT``,
  re-frozen 2026-11-06 after Open-Meteo was dropped) trained exactly as
  ``rain.expanding_predictions`` trains the November model (slot d 12z, station-days before
  2026-10-30), saved once to ``rain/frozen_model.json``; inputs fetched live from IEM only (ASOS
  precipitation, NBS / GFS MOS) through the study's own feature code with its as-of lags. Fair =
  the dev blend sigmoid(a + b logit(mid) + c logit(p)) (``RAIN_BLEND``); a YES and a NO bid at 8c;
  controls: the dev mid-recalibration (``RAIN_MIDCAL``) and the raw mid. Rows where rain was
  already reported at ``t`` are excluded (as in the study); orders rest until the first ASOS
  report of >= 0.01" + 10 min, the end of the CLI day, or the close.
* ``fcs`` (``ncaaf_maker``): FCS spread legs 5-7 h before ESPN kickoff (an hourly check of the
  ``football_game`` milestones), books 6-15c wide. ``ncaaf_maker.snapshot_open`` computes the
  frozen model, blend (``frozen.json``) and the frozen maker bids (margin 3c, |fair - mid| >= 2c,
  both sides) = ``rule``; ``blend3`` = the same bids without the 2c gate; controls ``midcal3``
  (dev mid-only fit) and ``mid3`` (raw mid), all resting until kickoff. The bids are priced on the
  listing quote but start at the depth read after the model ran (the ratings refit takes minutes);
  a bid that would cross that book is dropped.

Rows recorded with ``--force`` (whatever is open now, off schedule) carry ``scheduled = False``,
never enter the ledger and are left out of the score unless ``--include-adhoc``.

    nice -n 10 python -m albert.research.forward snapshot all      # what the hourly loop runs
    nice -n 10 python -m albert.research.forward snapshot rain --force
    nice -n 10 python -m albert.research.forward score all

Public data only: Kalshi market / order book / trade / milestone endpoints, IEM, ESPN (inside
``ncaaf_maker``). Raw responses are recorded under ``data/cache/forward/raw``.
"""

from __future__ import annotations

import fcntl
import json
import re
import sys
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

NAME = "forward"
LEADS = ("mentions", "rain", "fcs")
DUE = (timedelta(hours=5), timedelta(hours=7))   # mentions, fcs: first sight with start - now in (5 h, 7 h]
FORCE_HORIZON = timedelta(hours=48)

# mentions: research-fundamental-mentions_maker-20261106T010604Z ("fv_cal", dev fit)
MENTIONS_CAL = (-0.1156, 1.0474)
MENTIONS_MARGINS = (0.08, 0.12)
MENTIONS_END = timedelta(minutes=10)
SAID_BID, SAID_JUMP = 0.97, 0.12
CB_PRINT, CB_ASK = 0.95, 0.80
SPEECH_GROUPS = ("trump", "vance", "earnings", "worldnews", "bernie")

# rain: research-fundamental-rain-iem-20261107T010544Z (rain.run_iem: IEM-only model, group "d 12z",
# dev fits; the Open-Meteo era blend was (-0.308, 0.706, 0.587), the mid-only fit is unchanged)
RAIN_BLEND = (-0.244, 0.788, 0.485)
RAIN_MIDCAL = (-0.362, 1.292)
RAIN_MARGIN = 0.08
RAIN_MAX_SPREAD_C = 3
RAIN_HOURS = (12, 14)                              # due 12:00 <= now < 14:00 UTC
RAIN_SLOT, RAIN_VARIANT, RAIN_LAMBDA = "d 12z", "mos_obs", 1.0      # rain.IEM_VARIANT / IEM_LAMBDA
RAIN_TRAIN_BEFORE = date(2026, 10, 30)             # rain.expanding_predictions' November model
OBS_LAG = timedelta(minutes=10)

# fcs: data/cache/ncaaf_maker/frozen.json (margin 3c, gate 2c, both sides, FCS)
FCS_SPREAD_C = (6, 15)

ORDER_RX = re.compile(r"^o_(.+)_(yes|no)$")
LEVELS = pl.List(pl.List(pl.Float64))
BOOK_SCHEMA = {"ticker": pl.String, "book_bid": pl.Float64, "book_bid_size": pl.Float64, "book_ask": pl.Float64,
               "book_ask_size": pl.Float64, "yes_levels": LEVELS, "no_levels": LEVELS, "book_ts": pl.Datetime("us", "UTC")}


# ---- paths, clients, ledger -------------------------------------------------------------------
def root(s) -> Path:
    return s.data_dir / "cache" / NAME


def lead_dir(s, lead: str) -> Path:
    d = root(s) / lead
    d.mkdir(parents=True, exist_ok=True)
    return d


def client(s, venue: str, base: str, *, record: bool = True, min_interval: float = 0.5, transport=None):
    from albert.http import CachedClient

    return CachedClient(venue, base, root(s) / "raw", min_interval=min_interval, max_retries=8,
                        record_live=record, transport=transport)


def load_ledger(s, lead: str) -> dict[str, str]:
    p = lead_dir(s, lead) / "ledger.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_ledger(s, lead: str, ledger: dict[str, str]) -> None:
    p = lead_dir(s, lead) / "ledger.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=1, sort_keys=True))
    tmp.replace(p)


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


@contextmanager
def locked(s, lead: str):
    """Non-blocking per-lead lock: a second concurrent snapshot of the same lead is skipped."""
    with open(lead_dir(s, lead) / ".lock", "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ---- order book, prices -------------------------------------------------------------------------
def book_rows(payload: dict[str, Any], fetched_at: datetime | None = None, depth: int = 10) -> list[dict[str, Any]]:
    """``/markets/orderbooks`` -> best YES bid / ask and sizes (a NO bid at q is a YES ask at
    1 - q) plus the top ``depth`` levels of each side, best first, as [price, size]."""
    from albert.venues import kalshi as k

    out = []
    for ob in payload.get("orderbooks") or []:
        book = ob.get("orderbook_fp") or ob.get("orderbook") or {}
        top = k.orderbook_top(ob)
        lv = {side: sorted(((float(p), float(q)) for p, q in (book.get(f"{side}_dollars") or book.get(side) or [])),
                           reverse=True)[:depth] for side in ("yes", "no")}
        out.append({"ticker": ob.get("ticker"), "book_bid": top["yes_bid"], "book_bid_size": top["yes_bid_size"],
                    "book_ask": top["yes_ask"], "book_ask_size": top["yes_ask_size"],
                    "yes_levels": [list(x) for x in lv["yes"]], "no_levels": [list(x) for x in lv["no"]],
                    "book_ts": fetched_at})
    return out


def fetch_books(c, tickers: list[str], *, chunk: int = 50) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(0, len(tickers), chunk):
        page = c.get("/markets/orderbooks", {"tickers": sorted(tickers[i:i + chunk])}, cache=False)
        rows.extend(book_rows(page.payload, page.fetched_at))
    return pl.DataFrame(rows, schema=BOOK_SCHEMA)


def with_quote(df: pl.DataFrame) -> pl.DataFrame:
    """Quote at t from the order book (an empty side is 0 / 1, as Kalshi lists it), falling back
    to the listing quote when the book is missing; ``quoted`` = a two-sided-or-one-sided book."""
    bid = pl.when(pl.col("book_ts").is_not_null()).then(pl.col("book_bid").fill_null(0.0)).otherwise(pl.col("yes_bid"))
    ask = pl.when(pl.col("book_ts").is_not_null()).then(pl.col("book_ask").fill_null(1.0)).otherwise(pl.col("yes_ask"))
    df = df.with_columns(bid.alias("bid"), ask.alias("ask"))
    return df.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                           ((pl.col("ask") - pl.col("bid")) * 100).round(0).alias("spread_c"),
                           (pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("ask") > pl.col("bid") + 1e-9)
                            & ((pl.col("bid") > 0) | (pl.col("ask") < 1))).alias("quoted"))


def logit(p: np.ndarray) -> np.ndarray:
    from albert.research.fundamental import logit as lg

    return lg(np.asarray(p, dtype=float))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def bid_prices(fair: np.ndarray, bid: np.ndarray, ask: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """The harness quote: YES bid at floor(fair - m), NO bid at floor(1 - fair - m), each capped one
    tick inside the opposite quote (post-only); NaN where no order (fair missing or price < 1c)."""
    fair, bid, ask = (np.asarray(x, dtype=float) for x in (fair, bid, ask))
    with np.errstate(invalid="ignore"):
        yes = np.minimum(np.floor((fair - margin) * 100 + 1e-9) / 100, np.round(ask - 0.01, 2))
        no = np.minimum(np.floor((1 - fair - margin) * 100 + 1e-9) / 100, np.round(1 - bid - 0.01, 2))
        yes = np.where(np.isfinite(yes) & (yes >= 0.01 - 1e-9), np.round(yes, 2), np.nan)
        no = np.where(np.isfinite(no) & (no >= 0.01 - 1e-9), np.round(no, 2), np.nan)
    return yes, no


def _strs(values) -> pl.Series:
    """A string list for ``is_in`` (imploded, as polars >= 1.3x wants)."""
    return pl.Series(sorted(values), dtype=pl.String).implode()


def _col(x: np.ndarray, name: str, mask: np.ndarray | None = None) -> pl.Series:
    x = np.where(mask, x, np.nan) if mask is not None else x
    return pl.Series(name, x, dtype=pl.Float64, nan_to_null=True)


def _fees(c, series: list[str], day: str) -> pl.DataFrame:
    """Fee type / multiplier per series (``/series/<ticker>``, cached once per UTC day)."""
    rows = []
    for se in sorted(set(series)):
        info = (c.get(f"/series/{se}", cache=True, tag=day).payload or {}).get("series") or {}
        rows.append({"series": se, "fee_type": info.get("fee_type"), "fee_multiplier": float(info.get("fee_multiplier") or 1.0)})
    return pl.DataFrame(rows, schema={"series": pl.String, "fee_type": pl.String, "fee_multiplier": pl.Float64})


def _markets(c, params: dict[str, Any]) -> tuple[pl.DataFrame, datetime | None]:
    """Every page of a live ``/markets`` listing (status open by default) with displayed sizes."""
    from albert.venues import kalshi as k

    rows: list[dict[str, Any]] = []
    cursor, fetched = None, None
    while True:
        page = c.get("/markets", {"limit": 1000, **params, "cursor": cursor}, cache=False)
        fetched = fetched or page.fetched_at
        size = {m["ticker"]: (k.num(k._field(m, "yes_bid_size")), k.num(k._field(m, "yes_ask_size")))
                for m in page.payload.get("markets") or []}
        for r in k.market_rows(page.payload, {}, "forward"):
            r["yes_bid_size"], r["yes_ask_size"] = size.get(r["ticker"], (None, None))
            r["series"], r["event"] = r.pop("series_ticker"), r.pop("event_ticker")
            rows.append(r)
        cursor = page.payload.get("cursor")
        if not cursor or not page.payload.get("markets"):
            break
    rename = {"series_ticker": "series", "event_ticker": "event"}
    schema = {**{rename.get(k_, k_): v for k_, v in k.MARKET_SCHEMA.items()},
              "yes_bid_size": pl.Float64, "yes_ask_size": pl.Float64}
    return pl.DataFrame(rows, schema=schema).drop("fee_type", "fee_multiplier", "source"), fetched


# ---- lead 1: mentions -------------------------------------------------------------------------------
def speech_group(series: str) -> str:
    from albert.research import mentions as M

    sp = M.speaker_of(series)
    return sp if sp in SPEECH_GROUPS else "other"


def mention_due(payload: dict[str, Any], now: datetime, lo: timedelta, hi: timedelta, done: set[str]) -> pl.DataFrame:
    """Speech events (window series excluded) whose milestone start is in (now + lo, now + hi] and
    that are not in the ledger; the earliest start per event (as ``mentions.fetch_starts``)."""
    from albert.research import mentions as M

    schema = {"event": pl.String, "start": pl.Datetime("us", "UTC"), "end": pl.Datetime("us", "UTC"),
              "ms_title": pl.String, "ms_type": pl.String, "ms_updated": pl.Datetime("us", "UTC")}
    df = pl.DataFrame(M.milestone_rows(payload), schema=schema)
    df = df.with_columns(pl.col("event").str.split("-").list.first().alias("series"))
    df = df.filter(pl.col("start").is_not_null() & ~pl.col("series").is_in(_strs(M.WINDOW_SERIES))
                   & ~pl.col("event").is_in(_strs(done))
                   & (pl.col("start") > now + lo) & (pl.col("start") <= now + hi))
    return df.sort("start").unique("event", keep="first", maintain_order=True)


def said_share(rows: pl.DataFrame) -> float:
    """Share of an event's quoted legs bid >= 97c (a said word goes to 99c at once)."""
    q = rows.filter(pl.col("quoted"))
    return float((q["bid"] >= SAID_BID).mean()) if q.height else 0.0


def said24_from_candles(payload: dict[str, Any], at: datetime, max_age: timedelta = timedelta(hours=2)) -> float:
    """The same share at ``at`` = start - 24 h from batch hourly candles: each leg's last candle
    closed by ``at`` (at most ``max_age`` old) with a two-sided quote; 0 when no leg was quoted."""
    from albert.venues import kalshi as k

    said = []
    for m in payload.get("markets") or []:
        cs = [c for c in k.candle_rows({"candlesticks": m.get("candlesticks") or []}, m.get("market_ticker"), 60)
              if c["end_ts"] is not None and at - max_age <= c["end_ts"] <= at
              and c["yes_bid_close"] is not None and c["yes_ask_close"] is not None
              and c["yes_ask_close"] > c["yes_bid_close"]]
        if cs:
            said.append(max(cs, key=lambda c: c["end_ts"])["yes_bid_close"] >= SAID_BID)
    return float(np.mean(said)) if said else 0.0


def mention_orders(df: pl.DataFrame) -> pl.DataFrame:
    """Fair value (frozen fv_cal) and the hypothetical NO bids of the rule (``cal8``, ``cal12``)
    and of the raw-mid control (``mid8``, ``mid12``) on eligible rows."""
    a, b = MENTIONS_CAL
    mid, bid, ask = (df[c].to_numpy().astype(float) for c in ("mid", "bid", "ask"))
    fair = np.clip(sigmoid(a + b * logit(mid)), 0.003, 0.997)
    ok = df["eligible"].to_numpy()
    cols = [_col(fair, "fair_cal")]
    for m in MENTIONS_MARGINS:
        tag = round(m * 100)
        cols.append(_col(bid_prices(fair, bid, ask, m)[1], f"o_cal{tag}_no", ok))
        cols.append(_col(bid_prices(mid, bid, ask, m)[1], f"o_mid{tag}_no", ok))
    return df.with_columns(cols)


def _snap_mentions(s, now: datetime, force: bool, done: set[str], *, transport=None, log=print):
    from albert.research import mentions as M
    from albert.venues import kalshi as k

    pre = client(s, "kalshi", s.kalshi_base, record=False, transport=transport)
    ms = pre.get("/milestones", {"minimum_start_date": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "category": "Mentions",
                                 "limit": 500}, cache=False)
    lo, hi = (timedelta(0), FORCE_HORIZON) if force else DUE
    due = mention_due(ms.payload, now, lo, hi, set() if force else done)
    if not due.height:
        return None, []
    c = client(s, "kalshi", s.kalshi_base, transport=transport)
    api = k.Kalshi(c)
    frames = []
    for ev in due.iter_rows(named=True):
        mk, _ = _markets(c, {"event_ticker": ev["event"]})
        mk = mk.filter(pl.col("status").is_in(_strs(["active", "open"])))
        mk = mk.with_columns(pl.col("yes_sub_title").map_elements(lambda w: M.parse_word(w)[0], return_dtype=pl.String)
                             .alias("word"))
        mk = mk.filter((pl.col("word").str.len_chars() > 0) & ~pl.col("word").str.starts_with("event does not qualify"))
        if not mk.height:
            continue
        tickers = mk["ticker"].to_list()
        rows = with_quote(mk.join(fetch_books(c, tickers), on="ticker", how="left"))
        at24 = ev["start"] - timedelta(hours=24)
        said24 = 0.0
        if at24 < now:
            said24 = said24_from_candles(api.candles_batch(tickers, at24 - timedelta(hours=3), at24, 60).payload, at24)
        said = said_share(rows)
        frames.append(rows.with_columns(
            pl.lit(ev["start"]).alias("start"), pl.min_horizontal(pl.lit(ev["start"] - MENTIONS_END), pl.col("close_time")).alias("end"),
            pl.lit(ev["ms_title"]).alias("ms_title"), pl.lit(ev["ms_type"]).alias("ms_type"),
            pl.lit(ev["ms_updated"]).alias("ms_updated"), pl.lit(said).alias("said"), pl.lit(said24).alias("said24"),
            pl.lit(said > said24 + SAID_JUMP + 1e-12).alias("excluded")))
    if not frames:
        return None, due["event"].to_list()
    df = pl.concat(frames, how="diagonal_relaxed")
    df = df.join(_fees(c, df["series"].unique().to_list(), now.strftime("%Y-%m-%d")), on="series", how="left")
    df = df.with_columns(
        pl.col("book_ts").fill_null(now).alias("t"),
        pl.col("series").map_elements(speech_group, return_dtype=pl.String).alias("group"),
        ((pl.col("start") - now).dt.total_seconds() / 3600).round(3).alias("lead_h"),
        (pl.col("quoted") & ~pl.col("excluded") & (pl.col("end") > now)).alias("eligible"))
    return mention_orders(df), due["event"].to_list()


# ---- lead 2: rain -----------------------------------------------------------------------------------
def station_ids(code: str) -> list[str]:
    from albert.research import rain

    return [code] + ([rain.OLD_ID[code]] if code in rain.OLD_ID else [])


def asos_multi(text: str) -> pl.DataFrame:
    """IEM ``asos.py`` CSV for several stations -> rain.asos_precip_rows per station, with the
    station id mapped to its Kalshi code (renamed stations keep their code)."""
    from albert.research import rain

    back = {old: new for new, old in rain.OLD_ID.items()}
    lines = text.splitlines()
    schema = {"code": pl.String, "ts": pl.Datetime("us", "UTC"), "p01i": pl.Float64, "lat": pl.Float64, "lon": pl.Float64}
    if not lines or not lines[0].startswith("station"):
        return pl.DataFrame(schema=schema)
    by: dict[str, list[str]] = {}
    for line in lines[1:]:
        by.setdefault(line.split(",", 1)[0], []).append(line)
    rows: list[dict[str, Any]] = []
    for sid, part in by.items():
        rows.extend(rain.asos_precip_rows("\n".join([lines[0], *part]), back.get(sid, sid)))
    return pl.DataFrame(rows, schema=schema)


def rain_windows(codes: list[str], day: date, t: datetime) -> pl.DataFrame:
    from albert.research import rain

    rows = []
    for code in codes:
        lo, hi = rain.lst_bounds(code, day)
        rows.append({"code": code, "day": day, "slot": RAIN_SLOT, "t": t, "lo": lo, "hi": hi})
    return pl.DataFrame(rows, schema={"code": pl.String, "day": pl.Date, "slot": pl.String, "t": pl.Datetime("us", "UTC"),
                                      "lo": pl.Datetime("us", "UTC"), "hi": pl.Datetime("us", "UTC")})


def fetch_asos(s, codes: list[str], lo: datetime, hi: datetime, *, now: datetime, transport=None) -> pl.DataFrame:
    """ASOS precipitation reports of ``codes`` in [lo, hi] (one IEM request); cached once the
    window is 3 h in the past."""
    iem = client(s, "iem", s.iem_base, min_interval=1.5, transport=transport)
    ids = [sid for c_ in codes for sid in station_ids(c_)]
    text = iem.get("/cgi-bin/request/asos.py", {
        "station": ids, "data": "p01i", "tz": "Etc/UTC", "format": "onlycomma", "latlon": "yes", "missing": "M",
        "trace": "T", "report_type": "3,4", "sts": lo.strftime("%Y-%m-%dT%H:%MZ"), "ets": hi.strftime("%Y-%m-%dT%H:%MZ")},
        cache=hi < now - timedelta(hours=3)).payload
    return asos_multi(text if isinstance(text, str) else "")


def rain_inputs(s, codes: list[str], day: date, t: datetime, *, transport=None, log=print) -> pl.DataFrame:
    """The study's station-day features for ``codes`` at decision time ``t`` (``rain.build_features``
    on live IEM data; every lag and as-of rule is the study's). A station whose request failed gets
    no row."""
    from albert.http import HttpError
    from albert.research import rain

    win = rain_windows(codes, day, t)
    if not win.height:
        return win
    sts = min(win["lo"].min() - timedelta(hours=1), t - rain.OBS_LAG - rain.RECENT - rain.RECENT_STALE)
    obs = fetch_asos(s, win["code"].to_list(), sts, t, now=t, transport=transport)
    iem = client(s, "iem", s.iem_base, min_interval=1.5, transport=transport)
    mos_rows: list[dict[str, Any]] = []
    failed: set[str] = set()
    for code in win["code"].to_list():
        for model in ("NBS", "GFS"):
            got = []
            for sid in station_ids(code):
                try:
                    text = iem.get("/cgi-bin/request/mos.py", {
                        "station": f"K{sid}", "model": model, "format": "csv",
                        "sts": (t - timedelta(hours=30)).strftime("%Y-%m-%dT%H:%MZ"),
                        "ets": t.strftime("%Y-%m-%dT%H:%MZ")}, cache=False).payload
                except HttpError as error:
                    log(f"rain: mos {model} K{sid}: {str(error)[:120]}")
                    failed.add(code)
                    break
                got = rain.mos_rows(text, code) if isinstance(text, str) else []
                if got:
                    break
            mos_rows.extend(got)
    mos = pl.DataFrame(mos_rows, schema={"code": pl.String, "model": pl.String, "runtime": pl.Datetime("us", "UTC"),
                                         "ftime": pl.Datetime("us", "UTC"), "p06": pl.Float64, "p12": pl.Float64,
                                         "q06": pl.Float64, "q12": pl.Float64})
    clim = rain.climatology(rain.load(s, "cli"), False)
    f = rain.build_features(win.filter(~pl.col("code").is_in(_strs(failed))), obs, mos, clim)
    return f.with_columns((pl.col("seen") >= 0.01).alias("decided"))


def rain_model(s):
    """The IEM-only ridge logit (RAIN_VARIANT, RAIN_LAMBDA) for slot d 12z, trained once on the
    cached station-day features before RAIN_TRAIN_BEFORE (the model ``rain.expanding_predictions``
    would use for November) and frozen to rain/frozen_model.json. A frozen file for another
    variant, lambda or feature list (e.g. the removed Open-Meteo model) is refused, not reused."""
    from albert.research import rain

    path = lead_dir(s, "rain") / "frozen_model.json"
    if path.exists():
        d = json.loads(path.read_text())
        want = rain.design(rain_feature_stub(), rain.VARIANTS[RAIN_VARIANT], d.get("cities") or [])[1]
        if (d.get("variant"), d.get("lam"), d.get("features")) != (RAIN_VARIANT, RAIN_LAMBDA, want):
            raise RuntimeError(f"{path} is for {d.get('variant')} / {d.get('lam')}, not {RAIN_VARIANT} / "
                               f"{RAIN_LAMBDA} with features {want}; move it away to re-freeze")
        m = rain.Logit(lam=d["lam"])
        m.beta, m.mu, m.sd = (np.array(d[x]) for x in ("beta", "mu", "sd"))
        m.cities = d.get("cities") or []
        return m
    feats = pl.read_parquet(rain._table(s, "features_iem_001"))
    fs = feats.filter((pl.col("slot") == RAIN_SLOT) & ~pl.col("decided") & pl.col("wet").is_not_null()
                      & (pl.col("day") < RAIN_TRAIN_BEFORE))
    parts = rain.VARIANTS[RAIN_VARIANT]
    cities = sorted(feats["code"].unique().to_list()) if "city" in parts else []
    X, names = rain.design(fs, parts, cities)
    m = rain.Logit(lam=RAIN_LAMBDA).fit(X, fs["wet"].to_numpy().astype(float))
    m.cities = cities
    write_json = {"variant": RAIN_VARIANT, "slot": RAIN_SLOT, "lam": RAIN_LAMBDA, "features": names, "cities": cities,
                  "inputs": "IEM only (CLI climatology, ASOS p01i, NBS + GFS MOS)",
                  "trained_days": [str(fs["day"].min()), str(fs["day"].max())], "rows": fs.height,
                  "frozen_at": datetime.now(UTC).isoformat(), "beta": m.beta.tolist(), "mu": m.mu.tolist(),
                  "sd": m.sd.tolist()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(write_json, indent=1))
    tmp.replace(path)
    return m


def rain_feature_stub() -> pl.DataFrame:
    """One all-missing feature row: enough for ``rain.design`` to list a variant's feature names."""
    return pl.DataFrame({"code": ["X"], "clim": [None], "left": [1.0], "seen": [0.0], "pre_max": [None],
                         **{f"{m}_{k}": [None] for m in ("nbs", "gfs") for k in ("pany", "pmax", "q")}},
                        schema_overrides={"clim": pl.Float64, "pre_max": pl.Float64,
                                          **{f"{m}_{k}": pl.Float64 for m in ("nbs", "gfs") for k in ("pany", "pmax", "q")}})


def rain_predict(model, feats: pl.DataFrame) -> np.ndarray:
    from albert.research import rain

    X, _ = rain.design(feats, rain.VARIANTS[RAIN_VARIANT], getattr(model, "cities", []))
    return model.predict(X)


def rain_orders(df: pl.DataFrame) -> pl.DataFrame:
    """Frozen blend fair value and the hypothetical YES and NO bids at 8c (``blend8``), with the
    mid-recalibration (``midcal8``) and raw-mid (``mid8``) controls, on eligible rows."""
    a, b, c_ = RAIN_BLEND
    a2, b2 = RAIN_MIDCAL
    mid, bid, ask = (df[c].to_numpy().astype(float) for c in ("mid", "bid", "ask"))
    p = df["p_model"].to_numpy().astype(float)
    fair = sigmoid(a + b * logit(mid) + c_ * logit(p))
    midcal = sigmoid(a2 + b2 * logit(mid))
    ok = df["eligible"].to_numpy()
    tag = round(RAIN_MARGIN * 100)
    cols = [_col(fair, "fair_blend"), _col(midcal, "fair_midcal")]
    for name, f in (("blend", fair), ("midcal", midcal), ("mid", mid)):
        y, n = bid_prices(f, bid, ask, RAIN_MARGIN)
        cols += [_col(y, f"o_{name}{tag}_yes", ok), _col(n, f"o_{name}{tag}_no", ok)]
    return df.with_columns(cols)


def _snap_rain(s, now: datetime, force: bool, done: set[str], *, transport=None, log=print):
    from albert.research import rain

    day = now.date()
    if not force and (not RAIN_HOURS[0] <= now.hour < RAIN_HOURS[1] or day.isoformat() in done):
        return None, []
    c = client(s, "kalshi", s.kalshi_base, transport=transport)
    mk, _ = _markets(c, {"series_ticker": "KXRAIN", "status": "open"})
    mk = mk.with_columns(pl.col("rules_primary").map_elements(rain.rule_code, return_dtype=pl.String).alias("code"),
                         pl.col("event").map_elements(rain.event_day, return_dtype=pl.Date).alias("day"))
    mk = mk.filter((pl.col("day") == day) & pl.col("code").is_in(_strs(rain.STATION_TZ)))
    if not mk.height:
        log(f"rain: no open KXRAIN legs for {day}")
        return None, []
    rows = with_quote(mk.join(fetch_books(c, mk["ticker"].to_list()), on="ticker", how="left"))
    bounds = rain_windows(sorted(rows["code"].unique().to_list()), day, now).select("code", "lo", "hi")
    rows = rows.join(bounds, on="code", how="left").with_columns(
        (pl.col("quoted") & (pl.col("spread_c") <= RAIN_MAX_SPREAD_C)).alias("tight"))
    t = rows["book_ts"].drop_nulls().max() or now
    codes = sorted(rows.filter("tight")["code"].unique().to_list())
    feats = rain_inputs(s, codes, day, t, transport=transport, log=log) if codes else pl.DataFrame()
    if feats.height:
        feats = feats.with_columns(pl.Series("p_model", rain_predict(rain_model(s), feats)))
        keep = ["code", "p_model", "decided", "seen", "left", "clim", "nbs_pany", "gfs_pany", "nbs_q", "pre_max", "pre_n"]
        rows = rows.join(feats.select(keep), on="code", how="left")
    else:
        rows = rows.with_columns(pl.lit(None, dtype=pl.Float64).alias("p_model"), pl.lit(None, dtype=pl.Boolean).alias("decided"))
    rows = rows.join(_fees(c, ["KXRAIN"], now.strftime("%Y-%m-%d")), on="series", how="left")
    rows = rows.with_columns(
        pl.col("book_ts").fill_null(now).alias("t"), pl.col("code").alias("group"),
        pl.min_horizontal("hi", "close_time").alias("end_max"),
        (pl.col("tight") & pl.col("p_model").is_not_null() & ~pl.col("decided").fill_null(True)).alias("eligible"))
    return rain_orders(rows), [day.isoformat()]


# ---- lead 3: FCS spreads -------------------------------------------------------------------------------
def football_due(payload: dict[str, Any], now: datetime, lo: timedelta, hi: timedelta, done: set[str]) -> pl.DataFrame:
    """College-football games (``football_game`` milestones whose main event is KXNCAAFGAME) that
    kick off in (now + lo, now + hi] and are not in the ledger: game key, start, division, and the
    game's spread event tickers."""
    from albert.venues import kalshi as k

    rows = []
    for ms in payload.get("milestones") or []:
        evs = (ms.get("primary_event_tickers") or []) + (ms.get("related_event_tickers") or [])
        main = (ms.get("details") or {}).get("main_game_event_ticker") or next(
            (e for e in evs if e.startswith("KXNCAAFGAME-")), None)
        if not main or not main.startswith("KXNCAAFGAME-"):
            continue
        rows.append({"key": main.split("-", 1)[1], "start": k.parse_ts(ms.get("start_date")), "title": ms.get("title"),
                     "division": (ms.get("details") or {}).get("division"),
                     "spread_events": sorted({e for e in evs if e.startswith("KXNCAAFSPREAD-")})})
    df = pl.DataFrame(rows, schema={"key": pl.String, "start": pl.Datetime("us", "UTC"), "title": pl.String,
                                    "division": pl.String, "spread_events": pl.List(pl.String)})
    return (df.filter(pl.col("start").is_not_null() & ~pl.col("key").is_in(_strs(done))
                      & (pl.col("start") > now + lo) & (pl.col("start") <= now + hi))
            .sort("start").unique("key", keep="first", maintain_order=True))


def fcs_orders(df: pl.DataFrame) -> pl.DataFrame:
    """The frozen rule's bids (``rule``: ``ncaaf_maker`` px where it posts), the same bids without
    the 2c gate (``blend3``) and the mid-only (``midcal3``) and raw-mid (``mid3``) controls."""
    bid, ask, mid = (df[c].to_numpy().astype(float) for c in ("bid", "ask", "mid"))
    ok = df["eligible"].to_numpy()
    margin = float(df["cfg_margin"][0]) if df.height else 0.03
    tag = round(margin * 100)
    cols = []
    for side in ("yes", "no"):
        px = df[f"px_{side}"].to_numpy().astype(float)
        post = df[f"post_{side}"].fill_null(False).to_numpy()
        cols += [_col(px, f"o_rule_{side}", ok & post), _col(px, f"o_blend{tag}_{side}", ok)]
    for name, f in (("midcal", df["fair_mid"].to_numpy().astype(float)), ("mid", mid)):
        y, n = bid_prices(f, bid, ask, margin)
        cols += [_col(y, f"o_{name}{tag}_yes", ok), _col(n, f"o_{name}{tag}_no", ok)]
    df = df.with_columns(cols)
    if "book_ts" not in df.columns:
        return df
    # The bids are priced on the listing quote, but a live quoter could post only after the model
    # ran (the depth read, ``book_ts``): a bid that would cross the book by then is dropped
    # (post-only would reject it).
    seen = pl.col("book_ts").is_not_null()
    b_ask, b_bid = pl.col("book_ask").fill_null(1.0), pl.col("book_bid").fill_null(0.0)
    keep = []
    for c_ in df.columns:
        m = ORDER_RX.match(c_)
        if m:
            cross = (pl.col(c_) >= b_ask - 1e-9) if m.group(2) == "yes" else (pl.col(c_) >= 1 - b_bid - 1e-9)
            keep.append(pl.when(seen & cross).then(None).otherwise(pl.col(c_)).alias(c_))
    return df.with_columns(keep)


class _OwnRaw:
    """Settings view whose ``raw_dir`` is data/cache/forward/raw/ncaaf, so the responses that
    ``ncaaf_maker.snapshot_open`` records stay out of the shared data/raw tree."""

    def __init__(self, s) -> None:
        self._s = s

    def __getattr__(self, name: str):
        return getattr(self._s, name)

    @property
    def raw_dir(self) -> Path:
        return root(self._s) / "raw" / "ncaaf"


def _snap_fcs(s, now: datetime, force: bool, done: set[str], *, transport=None, log=print):
    from albert.research import ncaaf_maker as nm

    pre = client(s, "kalshi", s.kalshi_base, record=False, transport=transport)
    ms = pre.get("/milestones", {"minimum_start_date": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "category": "Sports",
                                 "type": "football_game", "limit": 500}, cache=False)
    lo, hi = (timedelta(0), FORCE_HORIZON) if force else DUE
    due = football_due(ms.payload, now, lo, hi, set() if force else done)
    if not due.height:
        return None, []
    horizon = due["start"].max() - now + timedelta(hours=2)    # ESPN's kickoff can differ from Kalshi's start
    path = nm.snapshot_open(_OwnRaw(s), horizon=horizon, now=now, out_dir=lead_dir(s, "fcs") / "ncaaf_full")
    snap = pl.read_parquet(path)
    spread_events = sorted({e for evs in due["spread_events"].to_list() for e in evs})
    start_of = {e: st for evs, st in zip(due["spread_events"].to_list(), due["start"].to_list(), strict=True) for e in evs}
    legs = snap.filter(pl.col("event").is_in(_strs(spread_events)) & (pl.col("kind") == "spread") & (pl.col("tier") == "FCS"))
    if not legs.height:
        log(f"fcs: {due.height} game(s) due, no FCS spread legs among them")
        return None, due["key"].to_list()
    legs = legs.with_columns(((pl.col("ask") - pl.col("bid")) * 100).round(0).alias("spread_c"))
    legs = legs.with_columns((pl.col("fair").is_not_null() & pl.col("spread_c").is_between(*FCS_SPREAD_C)).alias("eligible"))
    c = client(s, "kalshi", s.kalshi_base, transport=transport)
    books = fetch_books(c, legs.filter("eligible")["ticker"].to_list())
    legs = legs.join(books, on="ticker", how="left").with_columns(
        pl.coalesce("book_ts", "quote_ts").alias("t"), pl.col("game_id").cast(pl.String).alias("group"),
        pl.col("event").replace_strict(start_of, default=None, return_dtype=pl.Datetime("us", "UTC")).alias("start"),
        ((pl.col("kickoff") - now).dt.total_seconds() / 3600).round(3).alias("lead_h"),
        pl.lit(str(path)).alias("ncaaf_snapshot"))
    return fcs_orders(legs), due["key"].to_list()


# ---- snapshot driver ---------------------------------------------------------------------------------
SNAPPERS = {"mentions": _snap_mentions, "rain": _snap_rain, "fcs": _snap_fcs}


def snapshot(lead: str, s=None, *, now: datetime | None = None, force: bool = False, transport=None,
             log=print) -> Path | None:
    """Record ``lead`` if it is due now (or whatever is open with ``force``). Idempotent: a key
    (mentions event, rain day, football game) is recorded once; returns the file or None."""
    from albert.config import settings

    s = s or settings()
    now = now or datetime.now(UTC)
    with locked(s, lead) as ok:
        if not ok:
            log(f"{lead}: another snapshot is running; skipped")
            return None
        ledger = load_ledger(s, lead)
        df, keys = SNAPPERS[lead](s, now, force, set(ledger), transport=transport, log=log)
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        if df is None or not df.height:
            if keys and not force:     # due but nothing to record (e.g. only FBS games): mark done
                ledger.update({k_: f"{stamp} (nothing recorded)" for k_ in keys})
                save_ledger(s, lead, ledger)
            log(f"{lead}: due ({', '.join(keys[:6])}) but nothing to record" if keys else f"{lead}: nothing due")
            return None
        df = df.with_columns(pl.lit(lead).alias("lead"), pl.lit(now).alias("snapshot_ts"), pl.lit(not force).alias("scheduled"))
        path = lead_dir(s, lead) / f"{stamp}.parquet"
        write_parquet(df, path)
        if not force:
            ledger.update({k_: stamp for k_ in keys})
            save_ledger(s, lead, ledger)
        n_orders = sum(int(df[c_].is_not_null().sum()) for c_ in df.columns if ORDER_RX.match(c_))
        log(f"{lead}: wrote {path} ({df.height} markets, {int(df['eligible'].sum())} eligible, {n_orders} bids, "
            f"keys {keys[:6]}{'...' if len(keys) > 6 else ''})")
        return path


# ---- scoring ------------------------------------------------------------------------------------------
def load_snapshots(s, lead: str, *, include_adhoc: bool = False) -> pl.DataFrame:
    files = sorted(lead_dir(s, lead).glob("*.parquet"))
    files = [f for f in files if re.match(r"^\d{8}T\d{6}Z\.parquet$", f.name)]
    if not files:
        return pl.DataFrame()
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    return df if include_adhoc else df.filter(pl.col("scheduled"))


def order_frame(snaps: pl.DataFrame) -> pl.DataFrame:
    """Long frame of every hypothetical bid: one row per (market, snapshot, config, side)."""
    base = [c_ for c_ in ("lead", "ticker", "event", "series", "group", "snapshot_ts", "t", "end", "bid", "ask", "mid",
                          "spread_c", "fee_type", "fee_multiplier", "scheduled") if c_ in snaps.columns]
    parts = []
    for col in snaps.columns:
        m = ORDER_RX.match(col)
        if m:
            parts.append(snaps.select(*base, pl.lit(m.group(1)).alias("cfg"), pl.lit(m.group(2)).alias("side"),
                                      pl.col(col).alias("price")).filter(pl.col("price").is_not_null()))
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def first_fills(orders: pl.DataFrame, trades: pl.DataFrame, end_col: str = "end") -> pl.DataFrame:
    """The harness rule: a YES bid fills at the first print after ``t`` (and by ``end``) where a
    NO taker sold YES strictly below it; a NO bid where a YES taker paid strictly above 1 - price.
    The queue at our own price is never assumed to clear."""
    o = orders.with_row_index("oid")
    if not trades.height:
        return o.with_columns(pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("fill_ts"), pl.lit(False).alias("filled"))
    tr = trades.select("ticker", pl.col("created_time").alias("ts"), "yes_price", "taker_side")
    j = (o.select("oid", "ticker", "t", pl.col(end_col).alias("_end"), "side", "price").join(tr, on="ticker")
         .filter((pl.col("ts") > pl.col("t")) & (pl.col("ts") <= pl.col("_end"))))
    through = (((pl.col("side") == "yes") & (pl.col("taker_side") == "no") & (pl.col("yes_price") < pl.col("price") - 1e-9))
               | ((pl.col("side") == "no") & (pl.col("taker_side") == "yes") & (pl.col("yes_price") > 1 - pl.col("price") + 1e-9)))
    hit = j.filter(through).group_by("oid").agg(pl.col("ts").min().alias("fill_ts"))
    return o.join(hit, on="oid", how="left").with_columns(pl.col("fill_ts").is_not_null().alias("filled"))


def settle(o: pl.DataFrame) -> pl.DataFrame:
    """PnL per contract held to settlement, minus the series maker fee (0.0175 x mult x P(1-P),
    rounded up to $0.0001, only on ``*maker_fee*`` series)."""
    p = pl.col("price")
    fee = ((0.0175 * pl.col("fee_multiplier").fill_null(1.0) * p * (1 - p) * 1e4 - 1e-9).ceil() / 1e4)
    fee = pl.when(pl.col("fee_type").fill_null("").str.contains("maker_fee")).then(fee).otherwise(0.0)
    won = pl.when(pl.col("side") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y")).cast(pl.Float64)
    return o.with_columns(pl.when(pl.col("filled")).then(won - p - fee).otherwise(0.0).alias("pnl"))


def kill_times(snaps: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    """Mentions circuit breaker per (event, snapshot): the first print after ``t`` at >= 95c on a
    leg of the event whose ask at ``t`` was <= 80c (a word was just said)."""
    legs = snaps.filter(pl.col("quoted") & (pl.col("ask") <= CB_ASK)).select("event", "snapshot_ts", "ticker", "t")
    tr = trades.filter(pl.col("yes_price") >= CB_PRINT).select("ticker", pl.col("created_time").alias("ts"))
    k_ = legs.join(tr, on="ticker").filter(pl.col("ts") > pl.col("t"))
    return k_.group_by("event", "snapshot_ts").agg(pl.col("ts").min().alias("kill_ts"))


def fetch_results(s, tickers: list[str], *, transport=None, chunk: int = 50) -> pl.DataFrame:
    """Settled result (y) per ticker from ``/markets?tickers=``; settled rows are kept in
    data/cache/forward/results.parquet and never re-requested."""
    from albert.venues import kalshi as k

    path = root(s) / "results.parquet"
    schema = {"ticker": pl.String, "result": pl.String, "y": pl.Int8, "settlement_ts": pl.Datetime("us", "UTC")}
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=schema)
    todo = sorted(set(tickers) - set(have["ticker"].to_list()))
    if todo:
        c = client(s, "kalshi", s.kalshi_base, transport=transport)
        rows = []
        for i in range(0, len(todo), chunk):
            page = c.get("/markets", {"tickers": ",".join(todo[i:i + chunk]), "limit": 1000}, cache=False)
            for r in k.market_rows(page.payload, {}, "forward-results"):
                if r["result"] in ("yes", "no"):
                    rows.append({"ticker": r["ticker"], "result": r["result"], "y": int(r["result"] == "yes"),
                                 "settlement_ts": r["settlement_ts"]})
        if rows:
            have = pl.concat([have, pl.DataFrame(rows, schema=schema)]).unique("ticker", keep="first")
            write_parquet(have, path)
    return have.filter(pl.col("ticker").is_in(_strs(tickers)))


def fetch_trades(s, windows: pl.DataFrame, *, transport=None, log=print) -> pl.DataFrame:
    """Every public print of each ticker in (t_from, t_to] (closed windows only, so the pages
    are cached as immutable)."""
    from albert.http import HttpError
    from albert.venues import kalshi as k

    api = k.Kalshi(client(s, "kalshi", s.kalshi_base, transport=transport))
    cut = api.cutoff()
    cutoff = cut.get("trades_created_ts") or cut.get("market_settled_ts")
    rows: list[dict[str, Any]] = []
    for r in windows.iter_rows(named=True):
        hist = bool(cutoff and r["t_to"] < cutoff)
        try:
            for page in api.trades(r["ticker"], hist, min_ts=r["t_from"] - timedelta(seconds=1), max_ts=r["t_to"] + timedelta(seconds=1)):
                rows.extend(k.trade_rows(page.payload))
        except HttpError as error:
            log(f"trades {r['ticker']}: {str(error)[:120]}")
            raise
    return pl.DataFrame(rows, schema=k.TRADE_SCHEMA).unique("trade_id")


def rain_ends(s, snaps: pl.DataFrame, now: datetime, *, transport=None) -> pl.DataFrame:
    """Order end per rain row: the first ASOS report of >= 0.01" in the CLI day + 10 min, the end
    of the CLI day, or the close (``weather_maker.first_rain``); only days that are over."""
    from albert.research import weather_maker as wm

    done = snaps.filter(pl.col("hi").is_not_null() & (pl.col("hi") < now - timedelta(minutes=10)))
    if not done.height:
        return done.with_columns(pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("end"))
    obs = fetch_asos(s, sorted(done["code"].unique().to_list()), done["lo"].min(), done["hi"].max(), now=now, transport=transport)
    win = done.select("code", pl.col("day").cast(pl.String).alias("key"), "lo", "hi").unique()
    wet = wm.first_rain(obs.select("code", "ts", "p01i"), win)
    done = done.with_columns(pl.col("day").cast(pl.String).alias("key")).join(
        wet.select("code", "key", "wet_ts"), on=["code", "key"], how="left")
    return done.with_columns(pl.min_horizontal(pl.col("wet_ts") + OBS_LAG, "hi", "close_time").alias("end"))


def summarize(o: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Per group: bids, fills, fill rate, PnL per fill (cents, t clustered by event), events and
    UTC days (of the decision time t) with fills, total $ at one contract per bid."""
    from albert.research.fundamental import clustered_mean

    rows = []
    for key, g in o.group_by(by, maintain_order=True):
        f = g.filter("filled")
        mu, tt, k_ = clustered_mean(f["pnl"].to_numpy(), f["event"].to_numpy()) if f.height else (float("nan"), None, 0)
        rows.append({**dict(zip(by, key, strict=True)), "bids": g.height, "fills": f.height,
                     "fill_rate": round(f.height / g.height, 3) if g.height else None,
                     "pnl_per_fill_c": round(mu * 100, 2) if f.height else None, "t": None if tt is None else round(tt, 2),
                     "events": k_, "days": f["t"].dt.date().n_unique() if f.height else 0,
                     "pnl_usd": round(float(f["pnl"].sum()), 2),
                     "yes_share": round(float((f["side"] == "yes").mean()), 2) if f.height else None})
    return pl.DataFrame(rows, infer_schema_length=None).sort(by) if rows else pl.DataFrame()


def score(lead: str, s=None, *, now: datetime | None = None, include_adhoc: bool = False, transport=None,
          log=print) -> pl.DataFrame:
    """Score every recorded bid whose order window has ended and whose market has settled;
    writes data/cache/forward/<lead>/scored.parquet and returns the running table."""
    from albert.config import settings

    s = s or settings()
    now = now or datetime.now(UTC)
    snaps = load_snapshots(s, lead, include_adhoc=include_adhoc)
    if not snaps.height:
        log(f"{lead}: no {'' if include_adhoc else 'scheduled '}snapshots yet")
        return pl.DataFrame()
    if lead == "rain":
        snaps = rain_ends(s, snaps, now, transport=transport)
        snaps = snaps.filter(pl.col("end").is_not_null() & (pl.col("end") > pl.col("t")))
    snaps = snaps.filter(pl.col("end") < now - timedelta(minutes=10))
    orders = order_frame(snaps)
    if not orders.height:
        log(f"{lead}: nothing to score yet (no closed order windows)")
        return pl.DataFrame()
    res = fetch_results(s, orders["ticker"].unique().to_list(), transport=transport)
    orders = orders.join(res.select("ticker", "y"), on="ticker")
    if not orders.height:
        log(f"{lead}: order windows closed, markets not settled yet")
        return pl.DataFrame()
    legs = snaps.filter(pl.col("quoted").fill_null(True)) if lead == "mentions" else snaps
    legs = legs.filter(pl.col("ticker").is_in(orders["ticker"].unique().implode()) | (pl.lit(lead == "mentions")))
    win = legs.group_by("ticker").agg(pl.col("t").min().alias("t_from"), pl.col("end").max().alias("t_to"))
    trades = fetch_trades(s, win, transport=transport, log=log)
    scored = settle(first_fills(orders, trades)).with_columns(pl.lit("end").alias("cancel"))
    if lead == "mentions":
        kt = kill_times(snaps, trades)
        o2 = orders.join(kt, on=["event", "snapshot_ts"], how="left").with_columns(
            pl.min_horizontal("end", pl.col("kill_ts").fill_null(pl.col("end"))).alias("end_cb")).drop("kill_ts")
        cb = settle(first_fills(o2, trades, "end_cb")).with_columns(pl.lit("cb").alias("cancel")).drop("end_cb")
        scored = pl.concat([scored, cb], how="diagonal_relaxed").with_columns(
            pl.when(pl.col("group").is_in(_strs(["trump", "vance"]))).then(pl.lit("trump/vance")).otherwise(pl.lit("other")).alias("scope"))
        table = pl.concat([summarize(scored, ["cfg", "cancel", "scope"]),
                           summarize(scored.with_columns(pl.lit("all").alias("scope")), ["cfg", "cancel", "scope"])],
                          how="diagonal_relaxed").sort("cfg", "cancel", "scope")
    else:
        table = summarize(scored, ["cfg"])
    write_parquet(scored, lead_dir(s, lead) / "scored.parquet")
    return table


# ---- CLI ---------------------------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(f"{datetime.now(UTC):%Y-%m-%d %H:%M:%S}Z {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from albert.config import settings

    ap = argparse.ArgumentParser(prog="python -m albert.research.forward", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["snapshot", "score"])
    ap.add_argument("lead", nargs="?", default="all", choices=["all", *LEADS])
    ap.add_argument("--force", action="store_true", help="snapshot whatever is open now (rows marked scheduled=False)")
    ap.add_argument("--include-adhoc", action="store_true", help="score forced (off-schedule) rows too")
    args = ap.parse_args(argv)
    s = settings()
    leads = list(LEADS) if args.lead == "all" else [args.lead]
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    failed = 0
    for lead in leads:
        try:
            if args.cmd == "snapshot":
                snapshot(lead, s, force=args.force, log=_log)
            else:
                table = score(lead, s, include_adhoc=args.include_adhoc, log=_log)
                if table.height:
                    _log(f"{lead}: running table (one contract per bid; PnL held to settlement, maker fee by series)")
                    print(table, flush=True)
        except Exception as error:  # noqa: BLE001  (one lead failing must not stop the others)
            failed += 1
            _log(f"{lead}: FAILED {type(error).__name__}: {str(error)[:300]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
