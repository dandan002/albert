"""Model-free behavioural and flow edges in Kalshi crypto markets, plus non-price crypto events.

Research exception (user decision 2026-11-23): crypto contracts are in scope for this study only;
any live use needs a separate explicit user decision. ``albert.categories`` still marks crypto as
financial; this module selects series itself (``/series`` category "Crypto").

Kalshi is the only venue. Coinbase's public market-data endpoint (api.coinbase.com, robots.txt 404 =
allowed) supplies hourly spot candles, used only as a control: to pick ladder legs by percentage
distance from spot at the decision time (a sampling rule, not a fair value) and to tag spot moves.
No fitted fair value (``crypto_fv``), nothing sub-hour (``crypto_fast``).

Families (hypotheses in ``data/research/crypto_flow/hypotheses.md``):

* ``daily`` / ``weekly``: the 5 pm ET above/below ladders of KX{BTC,ETH,SOL,XRP,DOGE,BNB,HYPE}D (the
  daily event lives 25 h, the Friday event 169 h). Event tickers are built from the date
  (``KXBTCD-26JUN0117``), so the sample never depends on volume or outcome.
* ``onetouch``: monthly / weekly one-touch max/min ladders (close early when touched).
* ``other``: milestones and non-price crypto events (airdrops, token launches, FDV, reserves, ATH...).

Dev = events closing before 2026-10-01; holdout = 2026-10-01 .. 2026-11-22.

    python -m albert.research.crypto_flow series | universe | spot | select | candles | panel
    python -m albert.research.crypto_flow screen | freeze | holdout | book
"""

from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

OUT = Path("data/research/crypto_flow")
CACHE = Path("data/cache/crypto_flow")
RAW = CACHE / "raw"
CANDLES = CACHE / "candles"          # vol_metrics.load_marks reads data/cache/*/candles/*.parquet
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
COINBASE_BASE = "https://api.coinbase.com"
UA = "albert-research/0.2"

START = datetime(2026, 3, 1, tzinfo=UTC)
DEV_END = datetime(2026, 10, 1, tzinfo=UTC)
HOLD_END = datetime(2026, 11, 23, tzinfo=UTC)
SPOT_START = datetime(2025, 12, 1, tzinfo=UTC)
SEED = 20261123
ET = ZoneInfo("America/New_York")

COINS: dict[str, str] = {"BTC": "KXBTCD", "ETH": "KXETHD", "SOL": "KXSOLD", "XRP": "KXXRPD",
                         "DOGE": "KXDOGED", "BNB": "KXBNBD", "HYPE": "KXHYPED"}
SPOT_PRODUCT = {c: f"{c}-USD" for c in COINS}
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


MAX_BACKOFF_S = 20.0


def _client(venue: str = "kalshi", min_interval: float = 0.5):
    """``CachedClient`` whose exponential retry sleep is capped at MAX_BACKOFF_S (the shared client's
    uncapped doubling stalled single requests for minutes while 12 agents shared Kalshi's per-IP limit).
    The cap only shortens sleeps after a 429/5xx; min_interval and the 2-thread limit are unchanged."""
    from albert.http import CachedClient

    base = KALSHI_BASE if venue == "kalshi" else COINBASE_BASE

    def capped_sleep(sec: float) -> None:
        time.sleep(min(sec, MAX_BACKOFF_S))

    return CachedClient(venue, base, RAW, min_interval=min_interval, max_retries=30, user_agent=UA,
                        sleep=capped_sleep)


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


def split_of(close: datetime) -> str:
    return "dev" if close < DEV_END else ("holdout" if close < HOLD_END else "later")


# ---- series and fees -----------------------------------------------------------------------------
SERIES_SCHEMA = {"series_ticker": pl.String, "title": pl.String, "frequency": pl.String,
                 "fee_type": pl.String, "fee_multiplier": pl.Float64, "tags": pl.String}


def series_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Category-Crypto series from a ``/series`` payload."""
    from albert.venues.kalshi import num

    out = []
    for s in payload.get("series") or []:
        if s.get("category") != "Crypto":
            continue
        out.append({"series_ticker": s.get("ticker"), "title": s.get("title"), "frequency": s.get("frequency"),
                    "fee_type": s.get("fee_type"), "fee_multiplier": num(s.get("fee_multiplier")),
                    "tags": ",".join(s.get("tags") or [])})
    return out


def fee_change_rows(payload: dict[str, Any], series: set[str]) -> list[dict[str, Any]]:
    return [r for r in payload.get("series_fee_change_arr") or [] if r.get("series_ticker") in series]


def fetch_series(log=print) -> pl.DataFrame:
    api = _client()
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    ser = pl.DataFrame(series_rows(api.get("/series", tag=day).payload), schema=SERIES_SCHEMA)
    fc = fee_change_rows(api.get("/series/fee_changes", {"show_historical": "true"}, tag=day).payload,
                         set(ser["series_ticker"]))
    _save(ser, OUT / "series.parquet")
    _save_json(fc, OUT / "fee_changes.json")
    log(f"series: {ser.height} crypto series; fee types {ser.group_by('fee_type', 'fee_multiplier').len().rows()}; "
        f"{len(fc)} scheduled fee changes on crypto series")
    return ser


# ---- universe ------------------------------------------------------------------------------------
MARKET_SCHEMA = {"ticker": pl.String, "event_ticker": pl.String, "series_ticker": pl.String,
                 "title": pl.String, "yes_sub_title": pl.String, "strike_type": pl.String,
                 "floor_strike": pl.Float64, "cap_strike": pl.Float64, "open_time": pl.Datetime("us", "UTC"),
                 "close_time": pl.Datetime("us", "UTC"), "expected_expiration_time": pl.Datetime("us", "UTC"),
                 "settlement_ts": pl.Datetime("us", "UTC"), "result": pl.String, "expiration_value": pl.String,
                 "can_close_early": pl.Boolean, "volume": pl.Float64, "rules_primary": pl.String,
                 "status": pl.String, "src": pl.String}
SKIP_FREQ = ("hourly", "fifteen_min")   # sub-hour events; the coin D series are rebuilt per event
MAX_PAGES = 40


def market_row(m: dict[str, Any], series: str, src: str) -> dict[str, Any]:
    from albert.research.counts import market_row as _row

    return _row(m, series, src)


def daily_event_tickers(series: str, d0: date, d1: date) -> list[tuple[str, date]]:
    """5 pm ET event tickers (``KXBTCD-26JUN0117``) for every date in [d0, d1)."""
    out = []
    d = d0
    while d < d1:
        out.append((f"{series}-{d:%y}{MONTHS[d.month - 1]}{d:%d}17", d))
        d += timedelta(days=1)
    return out


def _pages(api, path: str, params: dict[str, Any], *, cache: bool = True, max_pages: int = MAX_PAGES):
    cursor = None
    for _ in range(max_pages):
        page = api.get(path, {**params, "cursor": cursor}, cache=cache)
        yield page
        cursor = page.payload.get("cursor")
        if not cursor or not page.payload.get("markets"):
            return


def _series_markets(api, s: str) -> list[dict[str, Any]]:
    """Settled markets of one series closing on/after START: historical (newest first, stop once past
    START) plus live settled."""
    rows: list[dict[str, Any]] = []
    for page in _pages(api, "/historical/markets", {"series_ticker": s, "limit": 1000}):
        ms = page.payload.get("markets") or []
        rows.extend(market_row(m, s, "hist") for m in ms)
        closes = [m.get("close_time") for m in ms if m.get("close_time")]
        if closes and min(closes) < START.isoformat().replace("+00:00", "Z"):
            break
    for page in _pages(api, "/markets", {"series_ticker": s, "status": "settled", "limit": 1000,
                                         "min_close_ts": int(START.timestamp())}):
        rows.extend(market_row(m, s, "live") for m in page.payload.get("markets") or [])
    return rows


def _event_markets(api, event: str, series: str, live_first: bool) -> list[dict[str, Any]]:
    order = (("/markets", "live"), ("/historical/markets", "hist"))
    for path, src in (order if live_first else order[::-1]):
        ms = api.get(path, {"event_ticker": event, "limit": 1000}).payload.get("markets") or []
        if ms:
            return [market_row(m, series, src) for m in ms]
    return []


def fetch_universe(*, workers: int = 2, log=print) -> pl.DataFrame:
    ser = pl.read_parquet(OUT / "series.parquet")
    api = _client()
    cutoff = datetime.fromisoformat(api.get("/historical/cutoff", cache=False).payload["market_settled_ts"])
    general = ser.filter(~pl.col("frequency").is_in(SKIP_FREQ) & ~pl.col("series_ticker").is_in(list(COINS.values())))
    rows: list[dict[str, Any]] = []
    lock = threading.Lock()
    t0 = time.time()

    def run_series(s: str) -> None:
        got = _series_markets(api, s)
        with lock:
            rows.extend(got)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(run_series, general["series_ticker"].to_list())):
            if i % 25 == 0:
                log(f"  series {i + 1}/{general.height}: {len(rows)} markets, {time.time() - t0:.0f}s", flush=True)
    log(f"general series: {len(rows)} markets", flush=True)
    jobs = [(e, s, d) for s in COINS.values() for e, d in daily_event_tickers(s, START.date(), HOLD_END.date())]

    def run_event(job: tuple[str, str, date]) -> None:
        e, s, d = job
        got = _event_markets(api, e, s, live_first=datetime(d.year, d.month, d.day, tzinfo=UTC) >= cutoff - timedelta(days=2))
        with lock:
            rows.extend(got)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(run_event, jobs)):
            if i % 200 == 0:
                log(f"  coin events {i + 1}/{len(jobs)}: {len(rows)} markets, {time.time() - t0:.0f}s", flush=True)
    mk = (pl.DataFrame(rows, schema=MARKET_SCHEMA).unique("ticker", keep="first")
          .filter(pl.col("result").is_in(["yes", "no"]) & (pl.col("close_time") >= START)))
    _save(mk, OUT / "markets.parquet")
    log(f"universe: {mk.height} settled markets, {mk['event_ticker'].n_unique()} events, "
        f"{mk['series_ticker'].n_unique()} series")
    return mk


# ---- spot (Coinbase public candles; control only) -------------------------------------------------
SPOT_SCHEMA = {"coin": pl.String, "ts": pl.Datetime("us", "UTC"), "open": pl.Float64, "high": pl.Float64,
               "low": pl.Float64, "close": pl.Float64}


def spot_rows(payload: dict[str, Any], coin: str) -> list[dict[str, Any]]:
    """Hourly candles from ``/api/v3/brokerage/market/products/{id}/candles`` (``start`` = epoch s)."""
    return [{"coin": coin, "ts": datetime.fromtimestamp(int(c["start"]), UTC), "open": float(c["open"]),
             "high": float(c["high"]), "low": float(c["low"]), "close": float(c["close"])}
            for c in payload.get("candles") or []]


def fetch_spot(log=print) -> pl.DataFrame:
    api = _client("coinbase", min_interval=1.0)
    rows: list[dict[str, Any]] = []
    step = 349 * 3600
    for coin, pid in SPOT_PRODUCT.items():
        s = int(SPOT_START.timestamp())
        end = int(HOLD_END.timestamp())
        while s < end:
            e = min(s + step, end)
            # completed windows are immutable, so cached; the last (still filling) window is not
            done = e < int(datetime.now(UTC).timestamp()) - 7200
            page = api.get(f"/api/v3/brokerage/market/products/{pid}/candles",
                           {"start": s, "end": e, "granularity": "ONE_HOUR"}, cache=done)
            rows.extend(spot_rows(page.payload, coin))
            s = e
        log(f"  spot {coin}: {len(rows)} rows so far", flush=True)
    sp = pl.DataFrame(rows, schema=SPOT_SCHEMA).unique(["coin", "ts"]).sort("coin", "ts")
    _save(sp, OUT / "spot.parquet")
    log(f"spot: {sp.height} hourly candles, {sp['coin'].n_unique()} coins, {sp['ts'].min()} .. {sp['ts'].max()}")
    return sp


# ---- leg selection (volume-blind; spot used only as a distance rule) ------------------------------
FAMILY_RX: tuple[tuple[str, str], ...] = (
    ("onetouch", r"^KX[A-Z]+(MAXMON|MINMON|MAXM|MINM|MAXW|MINW)$|^KXBTXMINW$"),
    ("annual", r"^(KX)?[A-Z]+(MAXY|MINY|MINMAXY|Y|YEAR|D26)$"),
)
LEADS_H: dict[str, tuple[int, ...]] = {"daily": (24, 6), "weekly": (72, 24), "onetouch": (168, 72, 24),
                                      "other": (720, 168, 72, 24)}
DIST_Q = (0.45, 0.99)        # legs whose |ln K/S_t| lies between these quantiles of |h-hour moves|
DEV_DAILY_DAYS = 15          # seeded random dev days per coin (daily ladders); holdout: every day
MAX_QUOTE_AGE = timedelta(days=7)


SKIP_RX = r"^(KXSHIBA|KXSHIBAD|KX[A-Z]+EVENT)$"   # legacy SHIB daily ladders; price-after-econ-release


def family_of(series: str, event: str, life_h: float) -> str:
    """daily / weekly coin ladders, onetouch, other (milestones and non-price events), skip (annual
    ladders: only their touched legs settle inside the window, a biased sample; SHIB; econ-event)."""
    if series in COINS.values():
        return "weekly" if life_h > 48 else "daily"
    if re.match(SKIP_RX, series):
        return "skip"
    for fam, rx in FAMILY_RX:
        if re.match(rx, series):
            return "skip" if fam == "annual" else fam
    return "other"


def normalize(mk: pl.DataFrame) -> pl.DataFrame:
    """KXDOGED lists its above-strikes as strike_type 'custom' with no floor_strike; the strike is the
    ticker suffix (``-T0.1349999``, rules: 'above 0.1349999')."""
    t = pl.col("ticker").str.extract(r"-T([0-9.]+)$", 1).cast(pl.Float64)
    fix = (pl.col("strike_type") == "custom") & pl.col("floor_strike").is_null() & t.is_not_null() \
        & pl.col("series_ticker").is_in(list(COINS.values()))
    return mk.with_columns(pl.when(fix).then(t).otherwise(pl.col("floor_strike")).alias("floor_strike"),
                           pl.when(fix).then(pl.lit("greater")).otherwise(pl.col("strike_type")).alias("strike_type"))


def coin_of(series: str) -> str | None:
    for c in COINS:
        if series.startswith(("KX" + c, c)) or (c == "BTC" and series.startswith(("KXBITCOIN", "BITCOIN", "KXBTX"))):
            return c
    return None


def with_family(mk: pl.DataFrame) -> pl.DataFrame:
    mk = normalize(mk)
    life = ((pl.col("close_time").max() - pl.col("open_time").min()).dt.total_minutes() / 60).over("event_ticker")
    mk = mk.with_columns(life.alias("event_life_h"))
    fam = [family_of(s, e, h) for s, e, h in mk.select("series_ticker", "event_ticker", "event_life_h").iter_rows()]
    coin = [coin_of(s) for s in mk["series_ticker"]]
    return mk.with_columns(pl.Series("family", fam), pl.Series("coin", coin, dtype=pl.String))


def sched_close(mk: pl.DataFrame) -> pl.DataFrame:
    """Scheduled close per event, never the actual (early) close: expected_expiration_time minus the
    series' modal lag between expected expiration and close (measured on all legs, outcome-blind)."""
    lag = (pl.col("expected_expiration_time") - pl.col("close_time")).dt.total_seconds()
    modal = (mk.filter(pl.col("expected_expiration_time").is_not_null())
             .with_columns(lag.alias("lag")).group_by("series_ticker")
             .agg(pl.col("lag").mode().min().alias("lag")))
    mk = mk.join(modal, on="series_ticker", how="left").with_columns(pl.col("lag").fill_null(0))
    sc = pl.coalesce(pl.col("expected_expiration_time") - pl.duration(seconds=pl.col("lag")), pl.col("close_time"))
    return mk.with_columns(sc.max().over("event_ticker").alias("sched_close")).drop("lag")


def spot_quantiles(sp: pl.DataFrame) -> dict[tuple[str, int], tuple[float, float]]:
    """|ln S(t+h)/S(t)| quantiles per coin and horizon from spot before START (no sample data)."""
    out = {}
    pre = sp.filter(pl.col("ts") < START).sort("coin", "ts")
    hs = sorted({h for v in LEADS_H.values() for h in v})
    for coin in COINS:
        x = pre.filter(pl.col("coin") == coin)["close"].to_numpy()
        for h in hs:
            if len(x) <= h + 10:
                continue
            r = np.abs(np.log(x[h:] / x[:-h]))
            out[(coin, h)] = (float(np.quantile(r, DIST_Q[0])), float(np.quantile(r, DIST_Q[1])))
    return out


def spot_at(sp: pl.DataFrame, rows: pl.DataFrame) -> pl.DataFrame:
    """Spot at decision time t: close of the hourly candle that ends at or before t (start <= t - 1h)."""
    s = sp.select("coin", (pl.col("ts") + timedelta(hours=1)).alias("s_ts"), pl.col("close").alias("spot")).sort("s_ts")
    return rows.sort("t").join_asof(s, left_on="t", right_on="s_ts", by="coin", strategy="backward",
                                    tolerance=timedelta(hours=2), check_sortedness=False)


def ladder_decisions(mk: pl.DataFrame, sp: pl.DataFrame) -> pl.DataFrame:
    """(leg, lead) rows of the daily/weekly coin ladders with spot at t and the distance filter.
    Dev daily events are a seeded random sample of days per coin; weekly and holdout events all."""
    lad = mk.filter(pl.col("family").is_in(["daily", "weekly"]))
    ev = lad.group_by("event_ticker").agg(pl.col("coin").first(), pl.col("family").first(),
                                          pl.col("sched_close").first())
    ev = ev.with_columns(pl.col("sched_close").map_elements(split_of, return_dtype=pl.String).alias("split"))
    rng = np.random.default_rng(SEED)
    keep = []
    for coin in COINS:
        d = ev.filter((pl.col("coin") == coin) & (pl.col("family") == "daily") & (pl.col("split") == "dev")).sort("event_ticker")
        if d.height:
            keep += list(rng.choice(d["event_ticker"].to_list(), size=min(DEV_DAILY_DAYS, d.height), replace=False))
    ev = ev.filter((pl.col("family") == "weekly") | (pl.col("split") != "dev") | pl.col("event_ticker").is_in(keep))
    rows = (lad.join(ev.select("event_ticker", "split"), on="event_ticker")
            .with_columns(pl.col("family").replace_strict({k: list(v) for k, v in LEADS_H.items()}, default=None,
                                                          return_dtype=pl.List(pl.Int64)).alias("lead_h"))
            .explode("lead_h")
            .with_columns((pl.col("sched_close") - pl.duration(hours=pl.col("lead_h"))).alias("t"))
            .filter((pl.col("open_time") <= pl.col("t")) & (pl.col("close_time") > pl.col("t"))))
    rows = spot_at(sp, rows)
    q = spot_quantiles(sp)
    lo = [q.get((c, h), (None, None))[0] for c, h in rows.select("coin", "lead_h").iter_rows()]
    hi = [q.get((c, h), (None, None))[1] for c, h in rows.select("coin", "lead_h").iter_rows()]
    rows = rows.with_columns(pl.Series("d_lo", lo, dtype=pl.Float64), pl.Series("d_hi", hi, dtype=pl.Float64),
                             (pl.col("floor_strike") / pl.col("spot")).log().alias("log_k_s"))
    return rows.filter(pl.col("spot").is_not_null() & (pl.col("log_k_s").abs() >= pl.col("d_lo"))
                       & (pl.col("log_k_s").abs() <= pl.col("d_hi")))


def other_decisions(mk: pl.DataFrame) -> pl.DataFrame:
    """(leg, lead) rows of every non-ladder crypto market open at sched_close - lead."""
    o = mk.filter(~pl.col("family").is_in(["daily", "weekly"]))
    o = o.with_columns(pl.col("sched_close").map_elements(split_of, return_dtype=pl.String).alias("split"))
    rows = (o.with_columns(pl.col("family").replace_strict({k: list(v) for k, v in LEADS_H.items()},
                                                           default=list(LEADS_H["other"]),
                                                           return_dtype=pl.List(pl.Int64)).alias("lead_h"))
            .explode("lead_h")
            .with_columns((pl.col("sched_close") - pl.duration(hours=pl.col("lead_h"))).alias("t"))
            .filter((pl.col("open_time") <= pl.col("t")) & (pl.col("close_time") > pl.col("t"))))
    return rows


def other_universe(log=print) -> pl.DataFrame:
    """Every settled market (any date) of the milestone / non-price series: there are too few after
    START for anything else. Pages come from the raw cache when ``universe`` already fetched them."""
    ser = pl.read_parquet(OUT / "series.parquet")
    api = _client()
    rows: list[dict[str, Any]] = []
    for s in ser["series_ticker"].to_list():
        if ser.filter(pl.col("series_ticker") == s)["frequency"][0] in SKIP_FREQ or family_of(s, "", 1e4) != "other":
            continue
        for page in _pages(api, "/historical/markets", {"series_ticker": s, "limit": 1000}):
            rows.extend(market_row(m, s, "hist") for m in page.payload.get("markets") or [])
        for page in _pages(api, "/markets", {"series_ticker": s, "status": "settled", "limit": 1000,
                                             "min_close_ts": int(START.timestamp())}):
            rows.extend(market_row(m, s, "live") for m in page.payload.get("markets") or [])
    mk = (pl.DataFrame(rows, schema=MARKET_SCHEMA).unique("ticker", keep="first")
          .filter(pl.col("result").is_in(["yes", "no"])))
    _save(mk, OUT / "markets_other.parquet")
    log(f"other universe: {mk.height} settled markets, {mk['event_ticker'].n_unique()} events, "
        f"{mk['series_ticker'].n_unique()} series")
    return mk


def select(log=print) -> pl.DataFrame:
    base = pl.read_parquet(OUT / "markets.parquet")
    oth = other_universe(log=log)
    mk = pl.concat([base, oth.filter(~pl.col("ticker").is_in(base["ticker"].implode()))], how="vertical_relaxed")
    mk = sched_close(with_family(mk)).filter(pl.col("family") != "skip")
    _save(mk, OUT / "markets_fam.parquet")
    sp = pl.read_parquet(OUT / "spot.parquet")
    lad = ladder_decisions(mk, sp)
    oth = other_decisions(mk)
    dec = pl.concat([lad, oth], how="diagonal_relaxed").filter(pl.col("split") != "later")
    _save(dec, OUT / "decisions.parquet")
    prio = pl.col("family").replace_strict({"other": 0, "onetouch": 1, "weekly": 2, "daily": 3}, default=4)
    need = (dec.group_by("ticker").agg(pl.col("family").first(), pl.col("event_ticker").first(),
                                       pl.col("settlement_ts").first(),
                                       pl.col("open_time").first(), pl.col("close_time").first(),
                                       pl.col("t").min().alias("t_first"))
            .with_columns(pl.max_horizontal("open_time", pl.col("t_first") - timedelta(hours=2)).alias("lo"),
                          pl.col("close_time").alias("hi"), pl.lit(60).alias("period"))
            .with_columns(prio.alias("prio")).sort("prio", "event_ticker", "ticker"))
    _save(need, OUT / "candle_todo.parquet")
    by = dec.group_by("family", "split").agg(pl.len().alias("rows"), pl.col("ticker").n_unique().alias("legs"),
                                            pl.col("event_ticker").n_unique().alias("events")).sort("family", "split")
    log(f"select: {dec.height} decision rows, {need.height} legs need candles\n{by}")
    return need


# ---- candles -----------------------------------------------------------------------------------------
def _kalshi():
    from albert.venues import kalshi as k

    return k.Kalshi(_client())


def fetch_candles(*, workers: int = 2, log=print) -> None:
    from albert.research import xcat

    need = pl.read_parquet(OUT / "candle_todo.parquet")
    CANDLES.mkdir(parents=True, exist_ok=True)
    done_path = CACHE / "candles_done.parquet"
    done = set(pl.read_parquet(done_path)["ticker"]) if done_path.exists() else set()
    todo = need.filter(~pl.col("ticker").is_in(list(done)))
    api = _kalshi()
    cutoff = api.cutoff()["market_settled_ts"]
    jobs = xcat._jobs(todo, cutoff)
    log(f"candles: {need.height} legs, {len(done)} done, {todo.height} to fetch in {len(jobs)} requests", flush=True)
    lock = threading.Lock()
    buf: list[dict[str, Any]] = []
    fin: list[str] = []
    failed: list[str] = []
    part = len(list(CANDLES.glob("part-*.parquet")))
    t0 = time.time()

    def flush() -> None:
        nonlocal buf, part
        if buf:
            pl.DataFrame(buf, schema=xcat.OHLC_SCHEMA).write_parquet(CANDLES / f"part-{part:05d}.parquet")
            part += 1
        buf = []
        pl.DataFrame({"ticker": sorted(done | set(fin))}, schema={"ticker": pl.String}).write_parquet(done_path)
        if failed:
            pl.DataFrame({"ticker": sorted(set(failed))}).write_parquet(CACHE / "candles_failed.parquet")

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(lambda j: xcat._run_job(api, j), jobs):
            with lock:
                for tk, got in res:
                    if got is None:
                        failed.append(tk)
                        continue
                    buf.extend(got)
                    fin.append(tk)
                n += 1
                if len(buf) > 40_000:      # live batch responses hold up to 50 legs each
                    flush()
                if n % 250 == 0:
                    flush()
                    rate = n / (time.time() - t0)
                    log(f"  candles {n}/{len(jobs)} requests, {rate:.2f} req/s, eta {(len(jobs) - n) / rate / 60:.0f} min",
                        flush=True)
    flush()
    log(f"candles done: {len(fin)} legs fetched, {len(set(failed))} failed")


def load_candles(tickers: list[str] | None = None) -> pl.DataFrame:
    parts = sorted(CANDLES.glob("part-*.parquet"))
    q = pl.scan_parquet(parts).select("ticker", "end_ts", "p_high", "p_low", "p_close", "bid_close", "ask_close", "volume")
    if tickers is not None:
        q = q.filter(pl.col("ticker").is_in(tickers))
    return q.collect().unique(["ticker", "end_ts"]).sort("ticker", "end_ts")


# ---- panel: quotes at the decision time ------------------------------------------------------------
ROUND_UNIT = {"BTC": 5000.0, "ETH": 500.0, "SOL": 25.0, "XRP": 0.5, "DOGE": 0.05, "BNB": 100.0, "HYPE": 10.0}


def quotes_at(dec: pl.DataFrame, c: pl.DataFrame) -> pl.DataFrame:
    """Last hourly candle ending at or before t with a bid/ask (<= MAX_QUOTE_AGE old). Kalshi candles
    report a missing bid as 0 and a missing ask as 1; ``two_sided`` needs 0 < bid < ask < 1."""
    q = (c.filter(pl.col("bid_close").is_not_null() & pl.col("ask_close").is_not_null())
         .select("ticker", pl.col("end_ts").alias("q_ts"), pl.col("bid_close").alias("bid"),
                 pl.col("ask_close").alias("ask")).sort("q_ts"))
    out = dec.sort("t").join_asof(q, left_on="t", right_on="q_ts", by="ticker", strategy="backward",
                                  check_sortedness=False)
    out = out.filter(pl.col("q_ts").is_not_null() & ((pl.col("t") - pl.col("q_ts")) <= MAX_QUOTE_AGE)
                     & (pl.col("ask") > pl.col("bid")))
    return out.with_columns(((pl.col("bid") > 0) & (pl.col("ask") < 1)).alias("two_sided"),
                            ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))


def strike_of(r: dict[str, Any]) -> float | None:
    """Leg strike (``greater`` legs list 66799.99 for a $66,800 level; tolerance handles that)."""
    return r.get("floor_strike") if r.get("floor_strike") is not None else r.get("cap_strike")


def panel(log=print) -> pl.DataFrame:
    dec = pl.read_parquet(OUT / "decisions.parquet")
    c = load_candles(dec["ticker"].unique().to_list())
    q = quotes_at(dec, c)
    sp = pl.read_parquet(OUT / "spot.parquet")
    # spot control: move over the 6 h before t (no look-ahead: candles that end by t)
    s = sp.select("coin", (pl.col("ts") + timedelta(hours=1)).alias("s_ts"), "close").sort("s_ts")
    q = q.with_columns((pl.col("t") - timedelta(hours=6)).alias("t6"))
    q6 = (q.select("ticker", "lead_h", "coin", "t6").sort("t6")
          .join_asof(s.rename({"s_ts": "s6", "close": "spot6"}), left_on="t6", right_on="s6", by="coin",
                     strategy="backward", tolerance=timedelta(hours=2), check_sortedness=False))
    q = q.join(q6.select("ticker", "lead_h", "spot6"), on=["ticker", "lead_h"], how="left")
    q = q.with_columns(pl.struct("floor_strike", "cap_strike").map_elements(strike_of, return_dtype=pl.Float64).alias("strike"),
                       (pl.col("result") == "yes").cast(pl.Int8).alias("y"),
                       (pl.col("spot") / pl.col("spot6")).log().alias("ret6"),
                       pl.col("t").dt.convert_time_zone("America/New_York").dt.weekday().alias("wd_et"))
    unit = pl.col("coin").replace_strict(ROUND_UNIT, default=None, return_dtype=pl.Float64)
    q = q.with_columns((((pl.col("strike") / unit) - (pl.col("strike") / unit).round()).abs() < 1e-3).fill_null(False)
                       .alias("round_strike"))
    _save(q, OUT / "panel.parquet")
    by = q.group_by("family", "split").agg(pl.len().alias("quoted"), pl.col("two_sided").sum().alias("two_sided"),
                                          pl.col("event_ticker").n_unique().alias("events")).sort("family", "split")
    log(f"panel: {dec.height} decision rows, {q.height} quoted\n{by}")
    return q


# ---- fills and P&L -------------------------------------------------------------------------------------
class Tape:
    """Per-leg hourly candle arrays (end time, trade high, trade low) for strict trade-through fills."""

    def __init__(self, c: pl.DataFrame) -> None:
        self.by: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        c = c.filter(pl.col("p_high").is_not_null() & pl.col("p_low").is_not_null())
        for (tk,), g in c.select("ticker", "end_ts", "p_high", "p_low").group_by(["ticker"]):
            g = g.sort("end_ts")
            self.by[tk] = (g["end_ts"].dt.epoch("s").to_numpy(), g["p_high"].to_numpy(), g["p_low"].to_numpy())

    def first_through(self, tk: str, t0: datetime, t1: datetime, yes_price: float, above: bool) -> datetime | None:
        """End of the first hourly candle that starts at or after ``t0`` and ends by ``t1`` (+1 h) with a
        YES print strictly above ``yes_price`` (``above``) or strictly below it."""
        a = self.by.get(tk)
        if a is None:
            return None
        ts, hi, lo = a
        i0 = np.searchsorted(ts, int(t0.timestamp()) + 3600, side="left")
        i1 = np.searchsorted(ts, int(t1.timestamp()) + 3600, side="right")
        if i1 <= i0:
            return None
        hit = hi[i0:i1] > yes_price + 1e-9 if above else lo[i0:i1] < yes_price - 1e-9
        if not hit.any():
            return None
        return datetime.fromtimestamp(int(ts[i0 + int(np.argmax(hit))]), UTC)


def taker_fee_order(n: int, price: float, mult: float | None) -> float:
    from albert.fees import kalshi_fee

    return kalshi_fee(n, price, fee_type="quadratic", multiplier=1.0 if mult is None else mult)


def maker_fee_contract(price: float, fee_type: str | None, mult: float | None) -> float:
    from albert.fees import kalshi_fee

    return kalshi_fee(1, price, maker=True, fee_type=fee_type, multiplier=1.0 if mult is None else mult)


def cheap_side(q: pl.DataFrame) -> pl.DataFrame:
    """Cheap side of each quoted leg: YES when mid <= 50c, else NO; its mid price ``c``."""
    return q.with_columns(pl.when(pl.col("mid") <= 0.5).then(pl.lit("yes")).otherwise(pl.lit("no")).alias("cheap"),
                          pl.when(pl.col("mid") <= 0.5).then(pl.col("mid")).otherwise(1 - pl.col("mid")).alias("c"))


def maker_orders(q: pl.DataFrame, *, band: tuple[float, float], sides: str, per_event: int = 3) -> pl.DataFrame:
    """Two-sided legs whose cheap side lies in ``band``; ``sides`` = 'yes' (cheap YES only, the culture
    rule) or 'both'; at most ``per_event`` legs per (event, lead), cheapest first (ties by ticker)."""
    o = cheap_side(q.filter(pl.col("two_sided")))
    o = o.filter((pl.col("c") > band[0] + 1e-9) & (pl.col("c") <= band[1] + 1e-9))
    if sides == "yes":
        o = o.filter(pl.col("cheap") == "yes")
    o = o.sort("event_ticker", "lead_h", "c", "ticker")
    return o.with_columns(pl.int_range(pl.len()).over("event_ticker", "lead_h").alias("rank")).filter(pl.col("rank") < per_event)


def maker_fills(o: pl.DataFrame, tape: Tape, margin_c: int) -> pl.DataFrame:
    """Sell the cheap side at ceil(c + m) with a post-only bid on the expensive side, 1 contract; filled
    on the first later hourly candle with a YES print strictly through; held to settlement. Returns
    every placed order (``filled`` flag) so a cash-aware book can reserve collateral for resting ones."""
    rows = []
    for r in o.iter_rows(named=True):
        s = math.ceil(round((r["c"] + margin_c / 100) * 100, 6)) / 100
        if s >= 0.99:
            continue
        held_price = round(1 - s, 2)
        if r["cheap"] == "yes":       # NO bid at 1 - s: filled by a YES print above s
            hit = tape.first_through(r["ticker"], r["t"], r["close_time"], s, above=True)
            won = 1 - r["y"]
        else:                         # YES bid at 1 - s: filled by a YES print below 1 - s
            hit = tape.first_through(r["ticker"], r["t"], r["close_time"], 1 - s, above=False)
            won = r["y"]
        fee = maker_fee_contract(held_price, r.get("fee_type"), r.get("fee_multiplier"))
        rows.append({**{k: r[k] for k in PASS_COLS if k in r}, "filled": hit is not None, "t_fill": hit,
                     "sell_at": s, "price": held_price, "held": "no" if r["cheap"] == "yes" else "yes", "n": 1,
                     "won": won, "pnl_c": 100 * (won - held_price - fee), "fee_c": 100 * fee})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


PASS_COLS = ("ticker", "event_ticker", "series_ticker", "family", "coin", "split", "lead_h", "t", "close_time",
             "settlement_ts", "sched_close", "strike_type", "strike", "spot", "log_k_s", "ret6", "wd_et",
             "round_strike", "cheap", "c", "mid", "bid", "ask", "y", "rank")


def taker_fills(q: pl.DataFrame, *, band: tuple[float, float], action: str, budget: float = 5.0) -> pl.DataFrame:
    """Taker at the quote on two-sided legs with the cheap side in ``band``: 'fade' buys the expensive
    side at its ask, 'back' buys the cheap side at its ask; floor(budget / price) contracts (>= 1), fee
    per order rounded up to the cent; held to settlement."""
    o = cheap_side(q.filter(pl.col("two_sided")))
    o = o.filter((pl.col("c") > band[0] + 1e-9) & (pl.col("c") <= band[1] + 1e-9))
    rows = []
    for r in o.iter_rows(named=True):
        buy_yes = (r["cheap"] == "yes") == (action == "back")
        price = r["ask"] if buy_yes else round(1 - r["bid"], 4)
        if not 0 < price < 1:
            continue
        n = max(1, int(budget // price))
        fee = taker_fee_order(n, price, r.get("fee_multiplier"))
        won = r["y"] if buy_yes else 1 - r["y"]
        rows.append({**{k: r[k] for k in PASS_COLS if k in r}, "t_fill": r["t"], "price": price,
                     "held": "yes" if buy_yes else "no", "n": n, "won": won,
                     "pnl_c": 100 * (won - price - fee / n), "fee_c": 100 * fee / n})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


# ---- statistics ----------------------------------------------------------------------------------------
def cl_t(x: np.ndarray, g: np.ndarray) -> tuple[float, float | None, int]:
    from albert.research.fundamental import clustered_mean

    return clustered_mean(x, g)


def summarize(df: pl.DataFrame) -> dict[str, Any]:
    """c/contract with t clustered by event and by settlement day (crypto legs co-move across coins)."""
    if df.is_empty():
        return {"n": 0, "events": 0}
    x = df["pnl_c"].to_numpy().astype(float)
    mu, t_ev, k_ev = cl_t(x, df["event_ticker"].to_numpy())
    _, t_day, k_day = cl_t(x, df["sched_close"].dt.date().cast(pl.String).to_numpy())
    ts = [v for v in (t_ev, t_day) if v is not None]
    cost = (df["price"] + df["fee_c"] / 100).to_numpy()
    return {"n": len(x), "events": int(k_ev), "days": int(k_day), "c_per": round(mu, 2),
            "t_event": None if t_ev is None else round(t_ev, 2), "t_day": None if t_day is None else round(t_day, 2),
            "t_min": round(min(ts), 2) if ts else None, "won": round(float(df["won"].mean()), 4),
            "avg_price": round(float(df["price"].mean()), 4),
            "pnl_per_dollar": round(float(x.sum() / 100 / cost.sum()), 4) if cost.sum() > 0 else None}


def p_two_sided(t: float | None) -> float:
    if t is None or not math.isfinite(t):
        return 1.0
    return math.erfc(abs(t) / math.sqrt(2))


def p_exact_events(df: pl.DataFrame) -> float:
    """Two-sided exact check that does not depend on a variance estimate (cells where every fill won
    make the clustered t infinite): one fill per event (the first), wins ~ Poisson-binomial with each
    event's break-even probability (price + fee) under H0. Two-sided = 2 x the tail in the direction
    of the observed mean, capped at 1."""
    if df.is_empty():
        return 1.0
    f = df.sort("t_fill", "c").group_by("event_ticker", maintain_order=True).first()
    p0 = np.clip((f["price"] + f["fee_c"] / 100).to_numpy().astype(float), 0.0, 1.0)
    w = int(f["won"].sum())
    dist = np.zeros(len(p0) + 1)
    dist[0] = 1.0
    for q in p0:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= 1 - q
    tail = dist[w:].sum() if w >= p0.sum() else dist[: w + 1].sum()
    return float(min(1.0, 2 * tail))


def cell_p(df: pl.DataFrame, t_min: float | None) -> float:
    """Cell p-value used for Holm: the larger of the clustered-t p and the event-level exact p."""
    return max(p_two_sided(t_min), p_exact_events(df))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * pvals[i]))
        adj[i] = run
    return adj


# ---- screen (dev only), freeze, holdout ------------------------------------------------------------
MAKER_BANDS = ((0.0, 0.05), (0.05, 0.15))
MARGINS_C = (6, 12)
TAKER_BANDS = ((0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.35))
MIN_FILLS, MIN_EVENTS = 60, 20
# the culture_flow / mentions mechanism carried over unchanged (named before any outcome was opened)
CARRY_OVER = {"kind": "maker", "family": "weekly", "lead_h": 72, "margin_c": 12, "band": [0.0, 0.05], "sides": "yes"}
CARRY_OVER_1T = {"kind": "maker", "family": "onetouch", "lead_h": 72, "margin_c": 12, "band": [0.0, 0.05], "sides": "yes"}


def _band_s(b: tuple[float, float] | list[float]) -> str:
    return f"{round(100 * b[0])}-{round(100 * b[1])}c"


def cell_name(c: dict[str, Any]) -> str:
    if c["kind"] == "maker":
        return f"maker|{c['family']}|{c['lead_h']}h|m{c['margin_c']}|{_band_s(c['band'])}|{c['sides']}"
    return f"taker|{c['family']}|{c['lead_h']}h|{c['action']}|{_band_s(c['band'])}"


def all_cells() -> list[dict[str, Any]]:
    cells = []
    for fam, leads in LEADS_H.items():
        for h in leads:
            for m in MARGINS_C:
                for b in MAKER_BANDS:
                    for sides in ("yes", "both"):
                        cells.append({"kind": "maker", "family": fam, "lead_h": h, "margin_c": m, "band": list(b),
                                      "sides": sides})
            for b in TAKER_BANDS:
                for action in ("fade", "back"):
                    cells.append({"kind": "taker", "family": fam, "lead_h": h, "action": action, "band": list(b)})
    return cells


def run_cell(cell: dict[str, Any], q: pl.DataFrame, tape: Tape) -> pl.DataFrame:
    sub = q.filter((pl.col("family") == cell["family"]) & (pl.col("lead_h") == cell["lead_h"]))
    if sub.is_empty():
        return pl.DataFrame()
    if cell["kind"] == "maker":
        o = maker_orders(sub, band=tuple(cell["band"]), sides=cell["sides"])
        return maker_fills(o, tape, cell["margin_c"]) if o.height else pl.DataFrame()
    return taker_fills(sub, band=tuple(cell["band"]), action=cell["action"])


def filled(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    return df.filter(pl.col("filled")) if "filled" in df.columns else df


def load_panel(split: str) -> tuple[pl.DataFrame, Tape]:
    q = pl.read_parquet(OUT / "panel.parquet").filter(pl.col("split") == split)
    mk = pl.read_parquet(OUT / "markets_fam.parquet").select("ticker", "fee_type", "fee_multiplier") \
        if "fee_type" in pl.read_parquet_schema(OUT / "markets_fam.parquet") else None
    ser = pl.read_parquet(OUT / "series.parquet").select("series_ticker", "fee_type", "fee_multiplier")
    q = q.join(ser, on="series_ticker", how="left") if mk is None else q.join(mk, on="ticker", how="left")
    tape = Tape(load_candles(q["ticker"].unique().to_list()))
    return q, tape


def screen(log=print) -> pl.DataFrame:
    """Every pre-declared cell on dev only. Fills are saved per cell for the diagnostics."""
    q, tape = load_panel("dev")
    rows, keep = [], []
    for cell in all_cells():
        df = run_cell(cell, q, tape)
        f = filled(df)
        st = summarize(f)
        st["p"] = cell_p(f, st.get("t_min")) if not f.is_empty() else 1.0
        st.update({"cell": cell_name(cell), "orders": df.height if "filled" in df.columns else st.get("n", 0),
                   **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in cell.items()}})
        rows.append(st)
        if not f.is_empty():
            g = f.with_columns(pl.lit(cell_name(cell)).alias("cell"))
            keep.append(g.select([c for c in (*PASS_COLS, "cell", "t_fill", "price", "held", "n", "won", "pnl_c", "fee_c")
                                  if c in g.columns]))
    cells = pl.DataFrame(rows, infer_schema_length=None)
    cells = cells.with_columns(pl.Series("p_holm", holm(cells["p"].to_list())))
    _save(cells, OUT / "cells_dev.parquet")
    if keep:
        _save(pl.concat(keep, how="diagonal_relaxed"), OUT / "fills_dev.parquet")
    show = cells.filter(pl.col("n") >= 20).sort("p")
    log(f"screen: {cells.height} dev cells ({(cells['n'] > 0).sum()} with fills)")
    with pl.Config(tbl_rows=60, tbl_width_chars=220, fmt_str_lengths=60):
        log(show.select("cell", "orders", "n", "events", "days", "c_per", "t_event", "t_day", "won", "avg_price",
                        "pnl_per_dollar", "p_holm").head(40))
    return cells


def diagnostics(fills: pl.DataFrame, cell: str) -> list[dict[str, Any]]:
    """H3-H7 splits of one cell's fills (each split is counted as a cell in the report)."""
    f = fills.filter(pl.col("cell") == cell)
    if f.is_empty():
        return []
    up = (pl.when(pl.col("log_k_s").is_not_null()).then(pl.col("log_k_s") > 0)
          .otherwise(pl.col("series_ticker").str.contains("MAX")))
    big = pl.col("ret6").abs() > pl.col("ret6").abs().median()
    splits = {
        "H3 upside tail": up, "H3 downside tail": ~up,
        "H4 round strike": pl.col("round_strike"), "H4 non-round strike": ~pl.col("round_strike"),
        "H5 weekend (ET)": pl.col("wd_et") >= 6, "H5 weekday (ET)": pl.col("wd_et") < 6,
        "H7 big 6h spot move": big, "H7 small 6h spot move": ~big,
    }
    out = []
    for name, e in splits.items():
        st = summarize(f.filter(e.fill_null(False)))
        out.append({"cell": cell, "split": name, **st})
    for coin in COINS:
        st = summarize(f.filter(pl.col("coin") == coin))
        if st["n"]:
            out.append({"cell": cell, "split": f"coin {coin}", **st})
    return out


def freeze(log=print) -> dict[str, Any]:
    """Primary rule: the dev cell with the smallest cell p-value (clustered t by event and by day, and the
    event-level exact test) among cells with >= MIN_FILLS fills, >= MIN_EVENTS events and a positive
    mean. The two pre-registered carry-over cells are scored too."""
    cells = pl.read_parquet(OUT / "cells_dev.parquet")
    # 'other' has 2 events after 2026-10-01 (known from the listing, not from outcomes): no holdout test
    ok = cells.filter((pl.col("n") >= MIN_FILLS) & (pl.col("events") >= MIN_EVENTS) & (pl.col("c_per") > 0)
                      & (pl.col("family") != "other"))
    best = ok.sort(["p", "n"], descending=[False, True]).head(1).to_dicts()
    cmap = {cell_name(c): c for c in all_cells()}
    rules = {"carry_over": {**CARRY_OVER, "name": cell_name(CARRY_OVER)},
             "carry_over_onetouch": {**CARRY_OVER_1T, "name": cell_name(CARRY_OVER_1T)}}
    if best:
        rules["primary"] = {**cmap[best[0]["cell"]], "name": best[0]["cell"], "dev": best[0]}
    # best maker-family rule per family (descriptive holdout check; not confirmatory beyond Holm)
    for fam in ("daily", "weekly", "onetouch"):
        b = ok.filter(pl.col("family") == fam).sort(["p", "n"], descending=[False, True]).head(1).to_dicts()
        if b and b[0]["cell"] != rules.get("primary", {}).get("name"):
            rules[f"best_{fam}"] = {**cmap[b[0]["cell"]], "name": b[0]["cell"], "dev": b[0]}
    # exploratory (named before the holdout, below the n bar): the culture mechanism at a longer lead on
    # one-touch ladders, the largest positive dev mean among ladder cells
    ex = {"kind": "maker", "family": "onetouch", "lead_h": 168, "margin_c": 12, "band": [0.05, 0.15], "sides": "yes"}
    rules["explore_onetouch_maker"] = {**ex, "name": cell_name(ex),
                                       "dev": cells.filter(pl.col("cell") == cell_name(ex)).to_dicts()[0]}
    frozen = {"frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"), "n_dev_cells": cells.height,
              "selection": ("min cell p = max(p of min(t_event, t_day), event-level exact p) among dev cells "
                            "of daily/weekly/onetouch with >= 60 fills, >= 20 events, mean > 0"),
              "holm_survivors_dev": cells.filter(pl.col("p_holm") < 0.05)["cell"].to_list(), "rules": rules}
    _save_json(frozen, OUT / "frozen.json")
    log(json.dumps(frozen, indent=1, default=str))
    return frozen


def holdout(log=print) -> dict[str, Any]:
    frozen = json.loads((OUT / "frozen.json").read_text())
    if (OUT / "holdout.json").exists():
        log("holdout already scored; refusing to rescore (delete holdout.json deliberately to rerun)")
        return json.loads((OUT / "holdout.json").read_text())
    q, tape = load_panel("holdout")
    res, keep = {}, []
    for key, rule in frozen["rules"].items():
        cell = {k: v for k, v in rule.items() if k not in ("name", "dev")}
        df = run_cell(cell, q, tape)
        f = filled(df)
        res[key] = {"name": rule["name"], "orders": df.height, **summarize(f)}
        if not f.is_empty():
            keep.append(f.with_columns(pl.lit(key).alias("rule"), pl.lit(rule["name"]).alias("cell")))
        if "filled" in df.columns and not df.is_empty():
            _save(df.with_columns(pl.lit(key).alias("rule")), OUT / f"orders_holdout_{key}.parquet")
    if keep:
        _save(pl.concat(keep, how="diagonal_relaxed"), OUT / "fills_holdout.parquet")
    out = {"scored_utc": datetime.now(UTC).isoformat(timespec="seconds"), "results": res}
    _save_json(out, OUT / "holdout.json")
    log(json.dumps(out, indent=1, default=str))
    return out


# ---- H6: sweeps on sampled ladder legs (trade prints; spot used only to tag spot-driven flow) ------
TRADE_SCHEMA = {"trade_id": pl.String, "ticker": pl.String, "created_time": pl.Datetime("us", "UTC"),
                "count": pl.Float64, "yes_price": pl.Float64, "taker_side": pl.String}
SWEEP_LEGS_PER_SPLIT = 150
SWEEP_BUCKETS = ((0.0, 50.0), (50.0, 200.0), (200.0, 1e12))


def trade_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    from albert.venues.kalshi import trade_rows as _rows

    return _rows(payload)


def sweep_sample(q: pl.DataFrame) -> pl.DataFrame:
    """Seeded random ladder legs (weekly + daily) with a two-sided quote and cheap side <= 15c at the
    first decision time; outcome- and volume-blind."""
    o = cheap_side(q.filter(pl.col("two_sided") & pl.col("family").is_in(["daily", "weekly"])))
    o = o.filter(pl.col("c") <= 0.15).sort("t").group_by("ticker", maintain_order=True).first()
    frames = []
    for i, split in enumerate(("dev", "holdout")):
        s = o.filter(pl.col("split") == split).sort("ticker")
        frames.append(s.sample(min(SWEEP_LEGS_PER_SPLIT, s.height), seed=SEED + i))
    return pl.concat(frames)


def fetch_trades(*, workers: int = 2, log=print) -> pl.DataFrame:
    q = pl.read_parquet(OUT / "panel.parquet")
    legs = sweep_sample(q)
    _save(legs, OUT / "sweep_legs.parquet")
    api = _kalshi()
    cutoff = api.cutoff()["trades_created_ts"]
    rows: list[dict[str, Any]] = []
    lock = threading.Lock()

    def one(r: dict[str, Any]) -> None:
        got: list[dict[str, Any]] = []
        for hist in ((True, False) if r["close_time"] <= cutoff else (False,)):
            for page in api.trades(r["ticker"], hist, min_ts=r["t"], max_ts=r["close_time"]):
                got.extend(trade_rows(page.payload))
        with lock:
            rows.extend(got)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, legs.iter_rows(named=True)))
    tr = pl.DataFrame(rows, schema=TRADE_SCHEMA).unique("trade_id")
    _save(tr, OUT / "sweep_trades.parquet")
    log(f"trades: {legs.height} legs, {tr.height} prints")
    return tr


def sweeps(tr: pl.DataFrame, legs: pl.DataFrame, sp: pl.DataFrame) -> pl.DataFrame:
    """One row per sweep (same leg, taker side and second): notional paid, the taker's settlement
    edge per contract, and whether spot moved toward the strike in the sweep's clock hour."""
    t = tr.with_columns(pl.col("created_time").dt.truncate("1s").alias("sec"),
                        pl.when(pl.col("taker_side") == "yes").then(pl.col("yes_price"))
                        .otherwise(1 - pl.col("yes_price")).alias("paid"))
    sw = (t.group_by("ticker", "taker_side", "sec")
          .agg(pl.col("count").sum().alias("contracts"), (pl.col("count") * pl.col("paid")).sum().alias("usd"),
               (pl.col("count") * pl.col("paid")).sum().truediv(pl.col("count").sum()).alias("avg_paid")))
    sw = sw.join(legs.select("ticker", "event_ticker", "coin", "split", "strike", "y", "cheap", "sched_close"), on="ticker")
    hr = sp.select("coin", pl.col("ts").alias("hour"), ((pl.col("close") / pl.col("open")).log()).alias("ret_h"))
    sw = sw.with_columns(pl.col("sec").dt.truncate("1h").alias("hour")).join(hr, on=["coin", "hour"], how="left")
    # taker bought YES (bets on price >= strike for 'greater' legs) / NO; spot-aligned when the hour's move
    # went the taker's way by more than the coin's median absolute hourly move
    med = sp.with_columns((pl.col("close") / pl.col("open")).log().abs().alias("a")).group_by("coin").agg(pl.col("a").median())
    sw = sw.join(med, on="coin", how="left")
    toward = pl.when(pl.col("taker_side") == "yes").then(pl.col("ret_h")).otherwise(-pl.col("ret_h"))
    won = pl.when(pl.col("taker_side") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y"))
    return sw.with_columns((toward > pl.col("a")).fill_null(False).alias("spot_aligned"), won.alias("taker_won"),
                           (100 * (won - pl.col("avg_paid"))).alias("taker_edge_c"),
                           ((pl.col("taker_side") == "yes") == (pl.col("cheap") == "yes")).alias("bought_cheap"))


def sweep_table(log=print) -> pl.DataFrame:
    tr = pl.read_parquet(OUT / "sweep_trades.parquet")
    legs = pl.read_parquet(OUT / "sweep_legs.parquet")
    sw = sweeps(tr, legs, pl.read_parquet(OUT / "spot.parquet"))
    rows = []
    for split in ("dev", "holdout"):
        for cheap in (True, False):
            for lo, hi in SWEEP_BUCKETS:
                for aligned in (None, True, False):
                    s = sw.filter((pl.col("split") == split) & (pl.col("bought_cheap") == cheap)
                                  & (pl.col("usd") > lo) & (pl.col("usd") <= hi))
                    if aligned is not None:
                        s = s.filter(pl.col("spot_aligned") == aligned)
                    if s.is_empty():
                        continue
                    x = s["taker_edge_c"].to_numpy()
                    w = s["contracts"].to_numpy()
                    mu, t, k = cl_t(x, s["event_ticker"].to_numpy())
                    rows.append({"split": split, "taker_bought": "cheap side" if cheap else "expensive side",
                                 "usd_bucket": f"{lo:g}-{hi:g}" if hi < 1e11 else f">{lo:g}",
                                 "spot": {None: "all", True: "aligned", False: "not aligned"}[aligned],
                                 "sweeps": s.height, "events": k, "taker_edge_c": round(mu, 2),
                                 "t_event": None if t is None else round(t, 2),
                                 "contract_wtd_edge_c": round(float((x * w).sum() / w.sum()), 2)})
    tab = pl.DataFrame(rows)
    _save(tab, OUT / "sweeps_table.parquet")
    with pl.Config(tbl_rows=80, tbl_width_chars=200):
        log(tab)
    return tab


# ---- fill check against trade prints (holdout orders of a frozen maker rule) ----------------------
def print_check(key: str, *, workers: int = 2, log=print) -> dict[str, Any]:
    """Re-run the fill test of every holdout order of maker rule ``key`` on Kalshi trade prints: the
    first print after placement that trades strictly through the resting bid (``paper_trader.through``)."""
    from albert.paper_trader import through

    o = pl.read_parquet(OUT / f"orders_holdout_{key}.parquet")
    api = _kalshi()
    cutoff = api.cutoff()["trades_created_ts"]
    lock = threading.Lock()
    res: list[dict[str, Any]] = []

    def one(r: dict[str, Any]) -> None:
        side = "no" if r["cheap"] == "yes" else "yes"
        bid = 1 - r["sell_at"]                  # our bid price on the held side
        first = None
        for hist in ((True, False) if r["close_time"] <= cutoff else (False,)):
            for page in api.trades(r["ticker"], hist, min_ts=r["t"], max_ts=r["close_time"]):
                for tr in trade_rows(page.payload):
                    if tr["created_time"] > r["t"] and through(side, bid, tr["yes_price"], tr["taker_side"]):
                        first = tr["created_time"] if first is None else min(first, tr["created_time"])
        with lock:
            res.append({"ticker": r["ticker"], "lead_h": r["lead_h"], "print_fill": first is not None,
                        "t_print": first})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, o.iter_rows(named=True)))
    chk = o.join(pl.DataFrame(res), on=["ticker", "lead_h"])
    _save(chk, OUT / f"print_check_{key}.parquet")
    both = chk.filter(pl.col("filled") & pl.col("print_fill"))
    pf = chk.filter(pl.col("print_fill"))
    out = {"orders": chk.height, "candle_fills": int(chk["filled"].sum()), "print_fills": int(chk["print_fill"].sum()),
           "both": both.height, "print_rule": summarize(pf.with_columns(pl.col("t_print").alias("t_fill")))}
    _save_json(out, OUT / f"print_check_{key}.json")
    log(json.dumps(out, indent=1, default=str))
    return out


# ---- capped book, ledgers, Sharpe / vol -----------------------------------------------------------
CAPS = (20.0, 57.0)    # $20 beside the live mentions bot; $57 = $97 - mentions' $40


def capped(df: pl.DataFrame, cap: float) -> pl.DataFrame:
    """Accept orders in time order while reserved cash stays <= cap. A resting maker order reserves its
    price from t until it fills (then until settlement) or the leg closes; a taker position reserves its
    cost from t to settlement."""
    if df.is_empty():
        return df
    maker = "filled" in df.columns
    d = df.with_columns((pl.col("n") * pl.col("price") + pl.col("n") * pl.col("fee_c") / 100).alias("reserve"))
    end = (pl.when(pl.col("filled")).then(pl.col("settlement_ts")).otherwise(pl.col("close_time")) if maker
           else pl.col("settlement_ts"))
    d = d.with_columns(end.alias("t_end")).sort("t", "event_ticker", "c", "ticker")
    held: list[tuple[datetime, float]] = []
    ok = []
    for t, t_end, r in d.select("t", "t_end", "reserve").iter_rows():
        held = [(e, x) for e, x in held if e > t]
        if sum(x for _, x in held) + r <= cap + 1e-9:
            held.append((t_end, r))
            ok.append(True)
        else:
            ok.append(False)
    out = d.filter(pl.Series(ok))
    return filled(out)


def ledger(f: pl.DataFrame, strategy: str, variant: str, split: str) -> pl.DataFrame:
    return f.select(pl.lit(strategy).alias("strategy"), pl.lit(variant).alias("variant"), pl.lit(split).alias("split"),
                    (pl.col("ticker") + "|" + pl.col("lead_h").cast(pl.String)).alias("struct_id"), "event_ticker",
                    "ticker", pl.col("held").alias("side"), "n", pl.col("t_fill").alias("t_in"),
                    (pl.col("n") * (pl.col("price") + pl.col("fee_c") / 100)).alias("cost_usd"),
                    pl.col("settlement_ts").alias("settle_ts"), (pl.col("n") * pl.col("won")).cast(pl.Float64).alias("payoff_usd"))


def own_marks(tickers: list[str]) -> pl.DataFrame:
    """Hourly YES mids from this study's candles, same rule as ``vol_metrics.load_marks`` (mid only when
    0 < bid < ask < 1, else the last trade)."""
    c = load_candles(tickers)
    two = (pl.col("bid_close") > 0) & (pl.col("ask_close") < 1) & (pl.col("ask_close") > pl.col("bid_close"))
    mid = pl.when(two.fill_null(False)).then((pl.col("bid_close") + pl.col("ask_close")) / 2).otherwise(pl.col("p_close"))
    return (c.with_columns(mid.alias("yes_mid")).filter(pl.col("yes_mid").is_not_null())
            .select("ticker", pl.col("end_ts").alias("ts"), "yes_mid").sort("ticker", "ts"))


def weekly_pos(f: pl.DataFrame) -> float | None:
    if f.is_empty():
        return None
    w = f.group_by(pl.col("settlement_ts").dt.truncate("1w")).agg(pl.col("pnl_c").sum())
    return round(float((w["pnl_c"] > 0).mean()), 3)


def book(log=print) -> dict[str, Any]:
    from albert.research.vol_metrics import ledger_metrics

    frozen = json.loads((OUT / "frozen.json").read_text())
    out: dict[str, Any] = {}
    (OUT / "ledgers").mkdir(parents=True, exist_ok=True)
    qd, tape_d = load_panel("dev")
    for key, rule in frozen["rules"].items():
        cell = {k: v for k, v in rule.items() if k not in ("name", "dev")}
        sets = {"dev": run_cell(cell, qd, tape_d)}
        hp = OUT / f"orders_holdout_{key}.parquet"
        if hp.exists():
            sets["holdout"] = pl.read_parquet(hp)
        elif (OUT / "fills_holdout.parquet").exists() and cell["kind"] == "taker":
            fh = pl.read_parquet(OUT / "fills_holdout.parquet")
            sets["holdout"] = fh.filter(pl.col("rule") == key)
        for split, df in sets.items():
            if df.is_empty():
                continue
            if cell["kind"] == "taker" and "filled" in df.columns:   # concat with maker fills adds a null column
                df = df.drop("filled")
            for cap in (*CAPS, 1e9):
                f = capped(df, cap)
                if f.is_empty():
                    continue
                led = ledger(f, "crypto_flow", f"{key}|cap{cap:g}", split)
                name = f"ledger_{key}_{split}_cap{'all' if cap > 1e6 else int(cap)}"
                _save(led, OUT / "ledgers" / f"{name}.parquet")
                m = ledger_metrics(led, own_marks(led["ticker"].unique().to_list()))
                m.update({"fills": f.height, "c_per_fill": round(float(f["pnl_c"].mean()), 2),
                          "weeks_pos_settle": weekly_pos(f)})
                out[name] = m
                log(f"{name}: fills {f.height}, c/fill {m['c_per_fill']}, $/wk {m['usd_per_week']}, Sharpe {m['sharpe']} "
                    f"{m['sharpe_ci90']}, ex-top5 {m['sharpe_ex_top5']}, vol {m['ann_vol_pct']}%, maxDD {m['max_dd_usd']}, "
                    f"weeks+ {m['weeks_positive_pct']}%", flush=True)
    _save_json(out, OUT / "book.json")
    return out


# ---- report tables ---------------------------------------------------------------------------------
def md_table(df: pl.DataFrame, cols: list[str] | None = None) -> str:
    cols = cols or df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    for r in df.select(cols).iter_rows():
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


def calibration(split: str = "dev") -> pl.DataFrame:
    """Descriptive: cheap-side mid vs how often the cheap side won, by family and bucket (two-sided legs,
    one row per leg and lead; legs of an event are correlated, so read with the clustered cells)."""
    q = cheap_side(pl.read_parquet(OUT / "panel.parquet").filter((pl.col("split") == split) & pl.col("two_sided")))
    q = q.with_columns(pl.when(pl.col("cheap") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y")).alias("cheap_won"),
                       pl.col("c").cut([0.05, 0.10, 0.20, 0.35], labels=["0-5", "5-10", "10-20", "20-35", "35-50"])
                       .alias("bucket"))
    return (q.group_by("family", "bucket").agg(pl.len().alias("legs"), pl.col("event_ticker").n_unique().alias("events"),
                                               (100 * pl.col("c").mean()).round(2).alias("avg_mid_c"),
                                               (100 * pl.col("cheap_won").mean()).round(2).alias("won_pct"))
            .with_columns((pl.col("won_pct") - pl.col("avg_mid_c")).round(2).alias("won_minus_mid"))
            .sort("family", "bucket"))


def tables(log=print) -> None:
    parts = []
    cells = pl.read_parquet(OUT / "cells_dev.parquet")
    parts.append("## Dev cells (every cell tried; sorted by cell p)\n\n" + md_table(
        cells.sort("p"),
        ["cell", "orders", "n", "events", "days", "c_per", "t_event", "t_day", "won", "avg_price", "pnl_per_dollar", "p", "p_holm"]))
    for split in ("dev", "holdout"):
        if split == "holdout" and not (OUT / "holdout.json").exists():
            continue
        parts.append(f"## Calibration of the cheap side ({split})\n\n" + md_table(calibration(split)))
    if (OUT / "diagnostics.parquet").exists():
        parts.append("## Diagnostics (H3-H7 splits)\n\n" + md_table(pl.read_parquet(OUT / "diagnostics.parquet")))
    if (OUT / "sweeps_table.parquet").exists():
        parts.append("## Sweeps (H6)\n\n" + md_table(pl.read_parquet(OUT / "sweeps_table.parquet")))
    (OUT / "tables.md").write_text("\n\n".join(parts) + "\n")
    log(f"tables written: {OUT / 'tables.md'}")


def diag(log=print) -> pl.DataFrame:
    """H3-H7 splits on dev for the pooled fade and maker cells of each ladder family (counted as cells)."""
    f = pl.read_parquet(OUT / "fills_dev.parquet")
    frozen = json.loads((OUT / "frozen.json").read_text()) if (OUT / "frozen.json").exists() else {"rules": {}}
    names = {r["name"] for r in frozen["rules"].values()}
    names |= {n for n in f["cell"].unique().to_list() if "|fade|0-5c" in n or "|fade|5-10c" in n}
    rows = []
    for n in sorted(names):
        rows += diagnostics(f, n)
    d = pl.DataFrame(rows, infer_schema_length=None)
    _save(d, OUT / "diagnostics.parquet")
    with pl.Config(tbl_rows=200, tbl_width_chars=200, fmt_str_lengths=50):
        log(d.select("cell", "split", "n", "events", "c_per", "t_event", "t_day", "won"))
    return d


# ---- CLI -------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.crypto_flow")
    ap.add_argument("cmd")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--key", default="carry_over")
    a = ap.parse_args(argv)
    log = print
    if a.cmd == "series":
        fetch_series(log=log)
    elif a.cmd == "universe":
        fetch_universe(workers=a.workers, log=log)
    elif a.cmd == "spot":
        fetch_spot(log=log)
    elif a.cmd == "select":
        select(log=log)
    elif a.cmd == "candles":
        fetch_candles(workers=a.workers, log=log)
    elif a.cmd == "panel":
        panel(log=log)
    elif a.cmd == "screen":
        screen(log=log)
    elif a.cmd == "freeze":
        freeze(log=log)
    elif a.cmd == "holdout":
        holdout(log=log)
    elif a.cmd == "book":
        book(log=log)
    elif a.cmd == "diag":
        diag(log=log)
    elif a.cmd == "tables":
        tables(log=log)
    elif a.cmd == "printcheck":
        print_check(a.key, workers=a.workers, log=log)
    elif a.cmd == "trades":
        fetch_trades(workers=a.workers, log=log)
    elif a.cmd == "sweeps":
        sweep_table(log=log)
    else:
        raise SystemExit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    main()
