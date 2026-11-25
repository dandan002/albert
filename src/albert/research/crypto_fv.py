"""Fair value of Kalshi crypto price ladders, hours to months (swarm 2026-11-23, slug ``crypto_fv``).

Research exception (user decision 2026-11-23): crypto contracts are in scope for this study only;
``albert.categories`` still marks them financial and is not changed: series are selected here.
Sub-hour / 15-minute contracts (``crypto_fast``) and model-free flow effects (``crypto_flow``) are
out of scope.

**Question.** Does a price distribution built at decision time from public spot data (Coinbase
5-minute candles; BRTI is a composite of Coinbase, Kraken, Bitstamp, Gemini, ...), a realized-vol
forecast (HAR weights, hour-of-week seasonality, Student-t tails) and/or Deribit DVOL price Kalshi's
daily above/below and range ladders (5 pm ET event, 2-24 h before the close) and monthly one-touch
max/min barriers better than the Kalshi quote, and does any gap survive taker fees / adverse
selection out of sample?

**Settlement mechanics (rules text, 2026-11-23).** Daily ladders (KXBTCD/KXBTC/KXETHD/...): simple
average of the 60 one-second BRTI (ETHUSD_RTI, ...) values before the close. Monthly barriers
(KXBTCMAXMON/MINMON, ...): each minute a trimmed mean (top/bottom 20% of that minute's RTI prints
removed) is computed; YES as soon as any minute's value is above (below) the strike between the
market's issuance and the period end; the market closes early when hit.

Data: Kalshi public REST (``/events``, ``/markets``, ``/historical/markets``, candlesticks),
api.coinbase.com public market candles, Deribit public DVOL. Robots verdicts in ``robots.json``.
Coinbase Exchange (api.exchange.coinbase.com, robots 401) and Bitstamp (robots disallows /api/) are
treated as blocked and not used.

**Result (2026-11-23, report ``data/research/crypto_fv/report.md``): EDGE CANDIDATE (marginal).**
Dev = Apr-Sep 2026 quotes (Jan-Sep for the vol fit), holdout = 2026-10-01..11-22, 60 dev cells.
Frozen primary: taker on the 5 pm above/below ladders 2 and 4 h before the close when the dev-fitted
blend logit P = a + b logit(mid) + c logit(model) beats the ask (or 1 - bid) by >= 2c after the taker
fee. Dev +4.1c/contract (t 3.1, 486 fills / 348 events, Holm p 0.065 over 60 cells); holdout
+5.8c (t 2.2 by event, 2.05 two-way, 117 fills / 84 events), book Sharpe 5.7 [1.3, 10.4], ex top-5
4.0, ~$2/week at $2 orders. It buys favourites (avg 83c): the ladders' implied variance is ~1.2-1.6x
the HAR forecast and realized variance only ~0.4-0.7x the implied on dev, so the favourite side is
cheap. A model-free "buy any 55-92c favourite" control was flat on dev (+1.3c, t 0.9) but +8.6c
(t 3.2) on holdout, and holdout profit came from the NO side during the November sell-off.
Makers lose everywhere (adverse selection); range and barrier rules do not hold up.

    python -m albert.research.crypto_fv robots | events | fetchall | select | build | calib
                                       | evaluate | diag | tables | fixtures
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

OUT = Path("data/research/crypto_fv")
CACHE = Path("data/cache/crypto_fv")
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
COINBASE_BASE = "https://api.coinbase.com/api/v3/brokerage/market"
DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
UA = "albert-research/0.2"

START = datetime(2026, 1, 1, tzinfo=UTC)          # first daily event close in the sample
SPOT_START = datetime(2025, 9, 1, tzinfo=UTC)      # warm-up for 90-day vol windows
DEV_END = datetime(2026, 10, 1, tzinfo=UTC)        # dev: close < DEV_END
HOLD_END = datetime(2026, 11, 23, tzinfo=UTC)      # holdout: DEV_END <= close < HOLD_END
CUTOFF = datetime(2026, 9, 22, tzinfo=UTC)         # Kalshi /historical cutoff (market_settled_ts)

COINS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD", "DOGE": "DOGE-USD"}
DVOL_COINS = ("BTC", "ETH")
# daily ladders: series -> (coin, kind, strikes sampled per event, z targets at the first decision)
DAILY: dict[str, tuple[str, str, tuple[float, ...]]] = {
    # (cut 2026-11-23 02:50 from 6/6/4/4/5/4 targets to a subset because the shared Kalshi API ran at
    # ~0.5 req/s; the nearest-strike rule is per target, so the subset picks the same strikes)
    "KXBTCD": ("BTC", "above", (-1.2, -0.4, 0.4, 1.2)),
    "KXETHD": ("ETH", "above", (-1.2, -0.4, 0.4, 1.2)),
    "KXSOLD": ("SOL", "above", (-0.6, 0.6)),
    "KXXRPD": ("XRP", "above", (-0.6, 0.6)),
    "KXBTC": ("BTC", "range", (-1.0, 0.0, 1.0)),
    "KXETH": ("ETH", "range", (-0.5, 0.5)),
}
DAILY_HOUR = "17"                                   # the 5 pm ET event (lists ~25 h before its close)
HORIZONS = (2, 4, 8, 16, 24)                         # hours before the close
# monthly one-touch barriers: series -> (coin, "max" | "min")
BARRIER: dict[str, tuple[str, str]] = {
    "KXBTCMAXMON": ("BTC", "max"), "KXBTCMINMON": ("BTC", "min"),
    "KXETHMAXMON": ("ETH", "max"), "KXETHMINMON": ("ETH", "min"),
    "KXSOLMAXMON": ("SOL", "max"), "KXSOLMINMON": ("SOL", "min"),
    "KXXRPMAXMON": ("XRP", "max"), "KXXRPMINMON": ("XRP", "min"),
}
BARRIER_Z = 3.0            # barrier strikes sampled within |ln K/S| <= z * monthly sigma at listing
BARRIER_MAX_STRIKES = 5
BARRIER_DECISION_HOUR = 16  # one decision per day at 16:00 UTC while the market is open

ROBOTS_CHECKS = {
    "api.elections.kalshi.com": ["/trade-api/v2/markets", "/trade-api/v2/events", "/trade-api/v2/historical/markets"],
    "api.coinbase.com": ["/api/v3/brokerage/market/products/BTC-USD/candles"],
    "www.deribit.com": ["/api/v2/public/get_volatility_index_data"],
    "api.exchange.coinbase.com": ["/products/BTC-USD/candles"],
    "www.bitstamp.net": ["/api/v2/ohlc/btcusd/"],
    "api.kraken.com": ["/0/public/OHLC"],
    "api.gemini.com": ["/v2/candles/btcusd/1hr"],
    "www.cfbenchmarks.com": ["/data/indices/BRTI"],
}


# ---- small utils ---------------------------------------------------------------------------------
def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def parse_ts(v: Any) -> datetime | None:
    if v in (None, "", "0001-01-01T00:00:00Z"):
        return None
    if isinstance(v, int | float):
        return datetime.fromtimestamp(v, tz=UTC)
    return datetime.fromisoformat(str(v)).astimezone(UTC)


def num(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(str(v).replace("$", "").replace(",", ""))
    except ValueError:
        return None


def _f(node: dict[str, Any], name: str) -> Any:
    for key in (f"{name}_dollars", f"{name}_fp", name):
        if key in node:
            return node[key]
    return None


def _capped_sleep(seconds: float) -> None:
    """Backoff sleep capped at 20 s: the shared client doubles its delay on every 429, which under
    swarm load left workers asleep for many minutes after the API had recovered."""
    time.sleep(min(seconds, 20.0))


def kalshi_client(min_interval: float = 0.5):
    from albert.http import CachedClient
    return CachedClient("kalshi", KALSHI_BASE, CACHE / "raw", min_interval=min_interval, max_retries=14,
                        sleep=_capped_sleep)


def _http_client(venue: str, base: str, min_interval: float):
    from albert.http import CachedClient
    return CachedClient(venue, base, CACHE / "raw", min_interval=min_interval, max_retries=8)


# ---- robots ----------------------------------------------------------------------------------------
def check_robots(out: Path = OUT) -> list[dict[str, Any]]:
    import httpx

    from albert.research.counts import AGENTS, robots_verdict
    rows = []
    for host, paths in ROBOTS_CHECKS.items():
        try:
            r = httpx.get(f"https://{host}/robots.txt", timeout=20, follow_redirects=True, headers={"User-Agent": UA})
            status, text = r.status_code, (r.text if r.status_code == 200 else "")
        except httpx.HTTPError:
            status, text = None, ""
        for p in paths:
            v = robots_verdict(text, p) if status in (200, 404, 410) else dict.fromkeys(AGENTS, False)
            rows.append({"host": host, "path": p, "http_status": status,
                         "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                         **{f"allow_{a}": v[a] for a in AGENTS}, "allowed": all(v.values())})
        time.sleep(1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "robots.json").write_text(json.dumps(rows, indent=1))
    return rows


# ---- Kalshi events / markets -----------------------------------------------------------------------
def event_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"event_ticker": e.get("event_ticker"), "series_ticker": e.get("series_ticker"),
             "strike_date": parse_ts(e.get("strike_date")), "sub_title": e.get("sub_title"),
             "title": e.get("title")} for e in payload.get("events") or []]


EVENT_SCHEMA = {"event_ticker": pl.String, "series_ticker": pl.String, "strike_date": pl.Datetime("us", "UTC"),
                "sub_title": pl.String, "title": pl.String}


def fetch_events(series: list[str], *, log=print) -> pl.DataFrame:
    c = kalshi_client()
    rows: list[dict[str, Any]] = []
    for s in series:
        cur = None
        while True:
            p = c.get("/events", {"series_ticker": s, "status": "settled", "limit": 200, "cursor": cur}).payload
            rows += event_rows(p)
            cur = p.get("cursor")
            if not cur or not p.get("events"):
                break
        log(f"events {s}: {sum(r['series_ticker'] == s for r in rows)}")
    df = pl.DataFrame(rows, schema=EVENT_SCHEMA).unique("event_ticker")
    _save(df, CACHE / "events.parquet")
    return df


MARKET_SCHEMA = {"ticker": pl.String, "event_ticker": pl.String, "series_ticker": pl.String,
                 "strike_type": pl.String, "floor_strike": pl.Float64, "cap_strike": pl.Float64,
                 "open_time": pl.Datetime("us", "UTC"), "close_time": pl.Datetime("us", "UTC"),
                 "expected_expiration_time": pl.Datetime("us", "UTC"), "settlement_ts": pl.Datetime("us", "UTC"),
                 "result": pl.String, "expiration_value": pl.Float64, "volume": pl.Float64,
                 "rules_primary": pl.String, "rules_secondary": pl.String, "src": pl.String}


def market_rows(payload: dict[str, Any], src: str) -> list[dict[str, Any]]:
    out = []
    for m in payload.get("markets") or []:
        out.append({"ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"),
                    "series_ticker": (m.get("event_ticker") or "").split("-", 1)[0],
                    "strike_type": m.get("strike_type"), "floor_strike": num(m.get("floor_strike")),
                    "cap_strike": num(m.get("cap_strike")), "open_time": parse_ts(m.get("open_time")),
                    "close_time": parse_ts(m.get("close_time")),
                    "expected_expiration_time": parse_ts(m.get("expected_expiration_time")),
                    "settlement_ts": parse_ts(m.get("settlement_ts")), "result": m.get("result") or None,
                    "expiration_value": num(m.get("expiration_value")), "volume": num(_f(m, "volume")),
                    "rules_primary": m.get("rules_primary"), "rules_secondary": m.get("rules_secondary"),
                    "src": src})
    return out


def event_markets(c, event_ticker: str, strike_date: datetime | None = None) -> list[dict[str, Any]]:
    """All markets of one settled event: the live endpoint first (the historical one first for events
    that ended over two days before the cutoff), else the other."""
    rows = []
    order = (("/markets", "live"), ("/historical/markets", "historical"))
    if strike_date is not None and strike_date < CUTOFF - timedelta(days=2):
        order = order[::-1]
    for path, src in order:
        cur = None
        while True:
            p = c.get(path, {"event_ticker": event_ticker, "limit": 1000, "cursor": cur}).payload
            rows += market_rows(p, src)
            cur = p.get("cursor")
            if not cur or not p.get("markets"):
                break
        if rows:
            return rows
    return rows


def daily_events(ev: pl.DataFrame) -> pl.DataFrame:
    """The 5 pm ET event of each daily-ladder series with a strike date in [START, HOLD_END)."""
    return ev.filter(pl.col("series_ticker").is_in(list(DAILY)) & pl.col("event_ticker").str.ends_with(DAILY_HOUR)
                     & (pl.col("strike_date") >= START) & (pl.col("strike_date") < HOLD_END)).sort("strike_date")


def barrier_events(ev: pl.DataFrame) -> pl.DataFrame:
    return ev.filter(pl.col("series_ticker").is_in(list(BARRIER)))


def fetch_markets(*, workers: int = 2, log=print) -> pl.DataFrame:
    ev = pl.read_parquet(CACHE / "events.parquet")
    sel = pl.concat([barrier_events(ev), daily_events(ev).sort("strike_date", descending=True)])
    path = CACHE / "markets.parquet"
    done = set(pl.read_parquet(path, columns=["event_ticker"])["event_ticker"].to_list()) if path.exists() else set()
    todo = [(e, d) for e, d in sel.select("event_ticker", "strike_date").iter_rows() if e not in done]
    log(f"markets: {len(todo)} events to fetch ({len(done)} done)")
    clients = [kalshi_client() for _ in range(workers)]
    frames: list[pl.DataFrame] = []
    rows: list[dict[str, Any]] = []

    def job(i_e):
        i, (e, d) = i_e
        r = event_markets(clients[i % workers], e, d)
        for x in r[1:]:          # rules text is per event: keep it on the first market only (memory)
            x["rules_primary"] = x["rules_secondary"] = None
        return r
    n = 0
    with ThreadPoolExecutor(workers) as pool:
        for k, r in enumerate(pool.map(job, enumerate(todo))):
            rows += r
            n += len(r)
            if (k + 1) % 100 == 0 or k + 1 == len(todo):
                frames.append(pl.DataFrame(rows, schema=MARKET_SCHEMA))
                rows = []
            if (k + 1) % 200 == 0:
                log(f"  {k + 1}/{len(todo)} events, {n} markets")
    new = pl.concat(frames) if frames else pl.DataFrame(schema=MARKET_SCHEMA)
    old = pl.read_parquet(path) if path.exists() else new.clear()
    df = pl.concat([old, new]).unique("ticker", keep="first")
    _save(df, path)
    log(f"markets: {df.height} rows, {df['event_ticker'].n_unique()} events")
    return df


# ---- spot (Coinbase public candles) and DVOL (Deribit public) ---------------------------------------
def coinbase_rows(payload: dict[str, Any], coin: str) -> list[dict[str, Any]]:
    return [{"coin": coin, "ts": datetime.fromtimestamp(int(c["start"]), tz=UTC), "open": float(c["open"]),
             "high": float(c["high"]), "low": float(c["low"]), "close": float(c["close"]),
             "volume": float(c["volume"])} for c in payload.get("candles") or []]


SPOT_SCHEMA = {"coin": pl.String, "ts": pl.Datetime("us", "UTC"), "open": pl.Float64, "high": pl.Float64,
               "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64}


def fetch_spot(coins: list[str] | None = None, end: datetime = HOLD_END + timedelta(days=1), *, log=print) -> pl.DataFrame:
    """5-minute candles (``ts`` = candle start, UTC) from api.coinbase.com, 350 per request. Windows
    that end in the past are cached; the last (still open) window is fetched live."""
    c = _http_client("coinbase", COINBASE_BASE, 0.35)
    step = 300 * 350
    frames = []
    for coin in coins or list(COINS):
        rows: list[dict[str, Any]] = []
        s = int(SPOT_START.timestamp())
        now = time.time()
        while s < min(end.timestamp(), now - 300):
            e = int(min(s + step, now - 60))
            p = c.get(f"/products/{COINS[coin]}/candles", {"start": s, "end": e, "granularity": "FIVE_MINUTE"},
                      cache=e < now - 600).payload
            rows += coinbase_rows(p, coin)
            s = e
        df = pl.DataFrame(rows, schema=SPOT_SCHEMA).unique("ts").sort("ts")
        log(f"spot {coin}: {df.height} candles {df['ts'].min()} .. {df['ts'].max()}")
        frames.append(df)
    out = pl.concat(frames)
    _save(out, CACHE / "spot_5m.parquet")
    return out


def dvol_rows(payload: dict[str, Any], coin: str) -> list[dict[str, Any]]:
    res = payload.get("result") or {}
    return [{"coin": coin, "ts": datetime.fromtimestamp(d[0] / 1000, tz=UTC), "dvol": float(d[4])}
            for d in res.get("data") or []]


def fetch_dvol(*, log=print) -> pl.DataFrame:
    """Hourly DVOL (30-day implied vol index, %) close per coin; ``ts`` = candle start."""
    c = _http_client("deribit", DERIBIT_BASE, 1.0)
    frames = []
    for coin in DVOL_COINS:
        rows: list[dict[str, Any]] = []
        s = int(SPOT_START.timestamp() * 1000)
        end = int(min((HOLD_END + timedelta(days=1)).timestamp(), time.time() - 60) * 1000)
        while s < end:
            e = min(s + 900 * 3600 * 1000, end)
            p = c.get("/get_volatility_index_data", {"currency": coin, "start_timestamp": s, "end_timestamp": e,
                                                     "resolution": "3600"}, cache=e < time.time() * 1000 - 3.6e6).payload
            rows += dvol_rows(p, coin)
            s = e
        df = pl.DataFrame(rows, schema={"coin": pl.String, "ts": pl.Datetime("us", "UTC"), "dvol": pl.Float64})
        df = df.unique("ts").sort("ts")
        log(f"dvol {coin}: {df.height} hours {df['ts'].min()} .. {df['ts'].max()}")
        frames.append(df)
    out = pl.concat(frames)
    _save(out, CACHE / "dvol_1h.parquet")
    return out



# ---- vol engine ------------------------------------------------------------------------------------
HAR_W = (0.3, 0.4, 0.3)    # daily / weekly / monthly RV weights (fixed, Corsi-style; not fitted)
SEASON_WEEKS = 8
LONG_TAU_DAYS = 7.0        # mean reversion of the HAR level toward the 90-day RV for multi-day horizons


class VolEngine:
    """Hourly realized variance of one coin from 5-minute Coinbase closes, and forecasts of the
    variance of ln S between two hour marks using only hours that ended at or before ``t``."""

    def __init__(self, spot: pl.DataFrame, dvol: pl.DataFrame | None = None) -> None:
        s = spot.sort("ts").with_columns((pl.col("ts") + timedelta(minutes=5)).alias("tc"))
        tc = s["tc"].to_numpy().astype("datetime64[s]").astype(np.int64)
        px = s["close"].to_numpy()
        self.tc, self.px = tc, px
        self.t0 = int(tc[0] // 3600 * 3600)            # first hour mark
        nh = int((tc[-1] - self.t0) // 3600) + 2
        r = np.diff(np.log(px))
        ok = np.diff(tc) == 300
        hour_idx = ((tc[1:] - 1 - self.t0) // 3600 + 1).astype(np.int64)   # return ending in (H-1h, H] -> H
        rv = np.zeros(nh)
        cnt = np.zeros(nh)
        np.add.at(rv, hour_idx[ok], r[ok] ** 2)
        np.add.at(cnt, hour_idx[ok], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rv = np.where(cnt >= 6, rv * 12 / np.maximum(cnt, 1), np.nan)
        # fill missing hours with the trailing 24-hour mean
        fill = pl.Series(rv).fill_null(strategy="forward").to_numpy() if np.isnan(rv).any() else rv
        rv = np.where(np.isnan(rv), fill, rv)
        rv = np.nan_to_num(rv, nan=float(np.nanmean(rv)))
        self.rv = rv
        self.cum = np.concatenate([[0.0], np.cumsum(rv)])
        self.dvol_ts = self.dvol = None
        if dvol is not None and dvol.height:
            d = dvol.sort("ts")
            self.dvol_ts = (d["ts"].to_numpy().astype("datetime64[s]").astype(np.int64) + 3600)  # candle close
            self.dvol = d["dvol"].to_numpy()

    def hidx(self, t: datetime | int) -> int:
        ts = int(t.timestamp()) if isinstance(t, datetime) else int(t)
        return (ts - self.t0) // 3600

    def spot(self, t: datetime | int) -> float | None:
        """Last 5-minute close at or before ``t`` (no older than 15 minutes)."""
        ts = int(t.timestamp()) if isinstance(t, datetime) else int(t)
        i = int(np.searchsorted(self.tc, ts, side="right")) - 1
        if i < 0 or ts - self.tc[i] > 900:
            return None
        return float(self.px[i])

    def path_extreme(self, t0: int, t1: int, kind: str) -> float | None:
        """Max (or min) 5-minute close in (t0, t1]."""
        i0 = int(np.searchsorted(self.tc, t0, side="right"))
        i1 = int(np.searchsorted(self.tc, t1, side="right"))
        if i1 <= i0:
            return None
        seg = self.px[i0:i1]
        return float(seg.max() if kind == "max" else seg.min())

    def _sum(self, i: int, n: int) -> float:
        lo = max(i - n, 0)
        return float(self.cum[i + 1] - self.cum[lo + 1]) * n / max(i - lo, 1)

    def daily_levels(self, t: datetime | int) -> dict[str, float]:
        i = self.hidx(t)        # hour ending at t has index i (its rv uses returns ending <= t)
        d = self._sum(i, 24)
        w = self._sum(i, 168) / 7
        m = self._sum(i, 672) / 28
        q = self._sum(i, 2160) / 90
        return {"rv_d": d, "rv_w": w, "rv_m": m, "rv_q": q, "har": HAR_W[0] * d + HAR_W[1] * w + HAR_W[2] * m}

    def season(self, t: datetime | int) -> np.ndarray:
        """168 hour-of-week variance multipliers (mean 1) from the last SEASON_WEEKS weeks; index by
        hour-of-week of the hour *ending* at a mark (``(H / 3600) % 168``, epoch-aligned)."""
        i = self.hidx(t)
        n = 168 * SEASON_WEEKS
        lo = i - n + 1
        if lo < 0:
            return np.ones(168)
        seg = self.rv[lo:i + 1].copy()
        cap = np.quantile(seg, 0.99)
        seg = np.minimum(seg, cap)
        # hour-of-week of each element: absolute hour index -> epoch hours
        hours = (self.t0 // 3600 + np.arange(lo, i + 1)) % 168
        how = np.bincount(hours, weights=seg, minlength=168) / np.maximum(np.bincount(hours, minlength=168), 1)
        hod24 = np.bincount(hours % 24, weights=seg, minlength=24) / np.maximum(np.bincount(hours % 24, minlength=24), 1)
        hod = hod24[np.arange(168) % 24]
        f = 0.5 * how / how.mean() + 0.5 * hod / hod.mean()
        return f / f.mean()

    def dvol_at(self, t: datetime | int) -> float | None:
        if self.dvol is None:
            return None
        ts = int(t.timestamp()) if isinstance(t, datetime) else int(t)
        j = int(np.searchsorted(self.dvol_ts, ts, side="right")) - 1
        if j < 0 or ts - self.dvol_ts[j] > 3 * 3600:
            return None
        return float(self.dvol[j])

    def variance(self, t: datetime | int, hours: float, source: str = "har") -> float | None:
        """Forecast variance of ln S over (t, t + hours]. ``source``: har | dvol | mix (geometric mean)."""
        ts = int(t.timestamp()) if isinstance(t, datetime) else int(t)
        if not 0 <= self.hidx(ts) < len(self.rv):
            return None
        lev = self.daily_levels(ts)
        if source == "har":
            daily = lev["har"]
        elif source == "dvol":
            dv = self.dvol_at(ts)
            if dv is None:
                return None
            daily = (dv / 100) ** 2 / 365
        else:
            dv = self.dvol_at(ts)
            if dv is None:
                return None
            daily = math.sqrt(lev["har"] * (dv / 100) ** 2 / 365)
        days = hours / 24
        if days <= 7:
            s = self.season(ts)
            h0 = ts // 3600
            nh = round(hours)
            mult = float(s[(h0 + np.arange(1, nh + 1)) % 168].sum()) / 24 if nh > 0 else days
            return daily * mult
        if source == "har":   # multi-day: the HAR level decays toward the 90-day mean
            q = lev["rv_q"]
            avg = q + (daily - q) * (1 - math.exp(-days / LONG_TAU_DAYS)) * LONG_TAU_DAYS / days
            return avg * days
        return daily * days


# ---- distribution / pricing ------------------------------------------------------------------------
def t_cdf(x: np.ndarray | float, nu: float) -> np.ndarray:
    """CDF of the Student t with ``nu`` dof scaled to unit variance (nu > 2); nu >= 200 -> normal."""
    from scipy import stats
    x = np.asarray(x, dtype=float)
    if nu >= 200:
        return stats.norm.cdf(x)
    return stats.t.cdf(x * math.sqrt(nu / (nu - 2)), nu)


def p_above(spot: float, strike: np.ndarray | float, var: float, nu: float) -> np.ndarray:
    """P(S_T > K) with ln(S_T/S) = -var/2 + sqrt(var) * unit-variance t_nu (a martingale-ish centre)."""
    sd = math.sqrt(max(var, 1e-12))
    z = (np.log(np.asarray(strike, dtype=float) / spot) + var / 2) / sd
    return 1 - t_cdf(z, nu)


def p_touch(spot: float, strike: np.ndarray | float, var: float, nu: float, kind: str) -> np.ndarray:
    """One-touch probability (not yet touched) by the reflection formula for a log-Brownian motion with
    drift -var/2, the normal CDF swapped for the unit-variance t_nu (heuristic fat-tail version)."""
    sd = math.sqrt(max(var, 1e-12))
    k = np.asarray(strike, dtype=float)

    def g(x):
        return t_cdf(x / sd, nu)
    if kind == "max":
        b = np.log(k / spot)
        p = g(-b - var / 2) + np.exp(-b) * g(-b + var / 2)
    else:
        b = np.log(spot / k)
        p = g(-b + var / 2) + np.exp(b) * g(-b - var / 2)
    return np.clip(p, 0.0, 1.0)


def model_prob(row_kind: str, strike_type: str, floor: float | None, cap: float | None, spot: float,
               var: float, nu: float) -> float:
    if row_kind == "barrier":
        if strike_type == "greater":
            return float(p_touch(spot, floor, var, nu, "max"))
        return float(p_touch(spot, cap, var, nu, "min"))
    if strike_type == "greater":
        return float(p_above(spot, floor, var, nu))
    if strike_type == "less":
        return float(1 - p_above(spot, cap, var, nu))
    return float(p_above(spot, floor, var, nu) - p_above(spot, cap, var, nu))



# ---- sample selection (decision-time information only) and Kalshi candles ---------------------------
def engines() -> dict[str, VolEngine]:
    spot = pl.read_parquet(CACHE / "spot_5m.parquet")
    dv_path = CACHE / "dvol_1h.parquet"
    dvol = pl.read_parquet(dv_path) if dv_path.exists() else None
    out = {}
    for coin in spot["coin"].unique().to_list():
        d = dvol.filter(pl.col("coin") == coin) if dvol is not None else None
        out[coin] = VolEngine(spot.filter(pl.col("coin") == coin), d if d is not None and d.height else None)
    return out


def select_daily(mk: pl.DataFrame, eng: dict[str, VolEngine]) -> pl.DataFrame:
    """Per 5 pm event, the strikes nearest to fixed z-scores of ln(K/S) at the first decision
    (close - 24 h), z in units of that hour's HAR sigma. Uses spot and vol known at that time only."""
    keep = []
    d = mk.filter(pl.col("series_ticker").is_in(list(DAILY)))
    for (ev, series), g in d.group_by(["event_ticker", "series_ticker"]):
        coin, kind, zs = DAILY[series]
        if coin not in eng:
            continue
        close = g["close_time"].max()
        t0 = close - timedelta(hours=24)
        s0 = eng[coin].spot(t0)
        var = eng[coin].variance(t0, 24, "har")
        if s0 is None or not var:
            continue
        sd = math.sqrt(var)
        if kind == "above":
            c = g.filter(pl.col("strike_type") == "greater").with_columns(
                ((pl.col("floor_strike") / s0).log() / sd).alias("z0"))
        else:
            c = g.filter(pl.col("strike_type") == "between").with_columns(
                (((pl.col("floor_strike") + pl.col("cap_strike")) / 2 / s0).log() / sd).alias("z0"))
        if not c.height:
            continue
        z0 = c["z0"].to_numpy()
        idx = sorted({int(np.argmin(np.abs(z0 - z))) for z in zs})
        keep.append(c[idx].select("ticker", "z0"))
    sel = pl.concat(keep) if keep else pl.DataFrame(schema={"ticker": pl.String, "z0": pl.Float64})
    return mk.join(sel, on="ticker", how="inner").with_columns(pl.lit("daily").alias("family"))


def select_barrier(mk: pl.DataFrame, eng: dict[str, VolEngine]) -> pl.DataFrame:
    """Barrier strikes within BARRIER_Z monthly sigmas of spot at the market's own listing time,
    at most BARRIER_MAX_STRIKES per event (evenly spread by z)."""
    # only periods that ended inside the sample: an unfinished month lists only its already-hit
    # (settled YES) strikes, which would be an outcome-selected sample
    b = mk.filter(pl.col("series_ticker").is_in(list(BARRIER)) & (pl.col("expected_expiration_time") < HOLD_END))
    rows = []
    for r in b.iter_rows(named=True):
        coin, _kind = BARRIER[r["series_ticker"]]
        e = eng.get(coin)
        if e is None or r["open_time"] is None:
            continue
        t = r["open_time"]
        s0 = e.spot(t)
        var = e.variance(t.replace(minute=0, second=0, microsecond=0), 30 * 24, "har")
        k = r["floor_strike"] if r["strike_type"] == "greater" else r["cap_strike"]
        if s0 is None or not var or k is None:
            continue
        rows.append({"ticker": r["ticker"], "event_ticker": r["event_ticker"], "z0": math.log(k / s0) / math.sqrt(var)})
    z = pl.DataFrame(rows, schema={"ticker": pl.String, "event_ticker": pl.String, "z0": pl.Float64})
    z = z.filter(pl.col("z0").abs() <= BARRIER_Z)
    keep = []
    for _, g in z.group_by("event_ticker"):
        g = g.sort("z0")
        if g.height > BARRIER_MAX_STRIKES:
            idx = np.unique(np.round(np.linspace(0, g.height - 1, BARRIER_MAX_STRIKES)).astype(int))
            g = g[idx.tolist()]
        keep.append(g.select("ticker", "z0"))
    sel = pl.concat(keep) if keep else pl.DataFrame(schema={"ticker": pl.String, "z0": pl.Float64})
    return mk.join(sel, on="ticker", how="inner").with_columns(pl.lit("barrier").alias("family"))


CANDLE_SCHEMA = {"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "period_minutes": pl.Int64,
                 "yes_bid_close": pl.Float64, "yes_ask_close": pl.Float64, "price_close": pl.Float64,
                 "price_low": pl.Float64, "price_high": pl.Float64, "volume": pl.Float64}


def candle_rows(candles: list[dict[str, Any]], ticker: str) -> list[dict[str, Any]]:
    out = []
    for c in candles or []:
        bid, ask, pr = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
        out.append({"ticker": ticker, "end_ts": parse_ts(c.get("end_period_ts")), "period_minutes": 60,
                    "yes_bid_close": num(_f(bid, "close")), "yes_ask_close": num(_f(ask, "close")),
                    "price_close": num(_f(pr, "close")), "price_low": num(_f(pr, "low")),
                    "price_high": num(_f(pr, "high")), "volume": num(_f(c, "volume"))})
    return out


def fetch_candles(sel: pl.DataFrame, *, workers: int = 2, part: str = "a", log=print) -> pl.DataFrame:
    """Hourly candles over each selected market's life. Pre-cutoff markets: one historical call each;
    the rest: ``/markets/candlesticks`` batches of markets with the same life. Written as append-only
    parts under ``candles/`` (also read by ``vol_metrics`` for daily marks)."""
    cdir = CACHE / "candles"
    cdir.mkdir(parents=True, exist_ok=True)
    have: set[str] = set()
    for p in cdir.glob("candles_*.parquet"):
        have |= set(pl.read_parquet(p, columns=["ticker"])["ticker"].unique().to_list())
    sel = sel.filter(~pl.col("ticker").is_in(list(have)))
    log(f"candles: {sel.height} markets to fetch ({len(have)} cached)")
    if not sel.height:
        return sel
    clients = [kalshi_client() for _ in range(workers)]
    jobs: list[tuple[str, list[str], int, int]] = []
    live = sel.filter(pl.col("src") != "historical")
    for _, g in live.group_by(["event_ticker", "open_time", "close_time"]):
        tick = g["ticker"].to_list()
        o, c = g["open_time"][0], g["close_time"][0]
        per = max(1, min(20, 8000 // max(1, int((c - o).total_seconds() // 3600) + 2)))
        for i in range(0, len(tick), per):
            jobs.append(("b", tick[i:i + per], int(o.timestamp()), int(c.timestamp()) + 3600))
    # historical: newest first across both families (a cut-short run keeps a recent, contiguous block)
    hist = sel.filter(pl.col("src") == "historical").sort("close_time", descending=True)
    for r in hist.iter_rows(named=True):
        jobs.append(("h", [r["ticker"]], int(r["open_time"].timestamp()), int(r["close_time"].timestamp()) + 3600))

    def run(i_job):
        i, (kind, tick, s, e) = i_job
        cl = clients[i % workers]
        try:
            if kind == "h":
                p = cl.get(f"/historical/markets/{tick[0]}/candlesticks", {"start_ts": s, "end_ts": e, "period_interval": 60}).payload
                return candle_rows(p.get("candlesticks"), tick[0])
            p = cl.get("/markets/candlesticks", {"market_tickers": ",".join(sorted(tick)), "start_ts": s, "end_ts": e,
                                                 "period_interval": 60}).payload
            out = []
            for m in p.get("markets") or []:
                out += candle_rows(m.get("candlesticks"), m.get("market_ticker"))
            return out
        except Exception as error:   # noqa: BLE001 - log and continue; the market is retried next run
            log(f"  candle error {tick[:2]}: {str(error)[:160]}")
            return []
    rows: list[dict[str, Any]] = []
    n_part = 0
    with ThreadPoolExecutor(workers) as pool:
        for k, r in enumerate(pool.map(run, enumerate(jobs))):
            rows += r
            if (k + 1) % 500 == 0 or k + 1 == len(jobs):
                if rows:
                    _save(pl.DataFrame(rows, schema=CANDLE_SCHEMA),
                          cdir / f"candles_{part}_{datetime.now(UTC):%Y%m%dT%H%M%S}_{n_part}.parquet")
                    n_part += 1
                    rows = []
                log(f"  {k + 1}/{len(jobs)} candle jobs")
    return sel


def load_candles() -> pl.DataFrame:
    parts = sorted((CACHE / "candles").glob("candles_*.parquet"))
    if not parts:
        return pl.DataFrame(schema=CANDLE_SCHEMA)
    return pl.concat([pl.read_parquet(p) for p in parts]).unique(["ticker", "end_ts"]).sort("ticker", "end_ts")


def selection(eng: dict[str, VolEngine] | None = None) -> pl.DataFrame:
    eng = eng or engines()
    mk = pl.read_parquet(CACHE / "markets.parquet")
    sel = pl.concat([select_daily(mk, eng), select_barrier(mk, eng)], how="diagonal_relaxed")
    _save(sel, CACHE / "selection.parquet")
    return sel



# ---- decision rows ---------------------------------------------------------------------------------
VAR_SOURCES = ("har", "dvol", "mix")


def _quotes(c: pl.DataFrame, keys: pl.DataFrame) -> pl.DataFrame:
    """Attach the hourly candle that ended exactly at the decision hour ``t`` (its closing bid/ask is
    the book at ``t``; the decision is taken just after)."""
    q = c.select("ticker", pl.col("end_ts").alias("t"), pl.col("yes_bid_close").alias("bid"),
                 pl.col("yes_ask_close").alias("ask"), pl.col("price_close").alias("last"))
    return keys.join(q, on=["ticker", "t"], how="left")


def build_rows(eng: dict[str, VolEngine] | None = None, *, log=print) -> pl.DataFrame:
    eng = eng or engines()
    sel = pl.read_parquet(CACHE / "selection.parquet")
    candles = load_candles()
    coin_of = {**{k: v[0] for k, v in DAILY.items()}, **{k: v[0] for k, v in BARRIER.items()}}
    # daily: one row per (market, horizon)
    d = sel.filter(pl.col("family") == "daily").join(pl.DataFrame({"h": list(HORIZONS)}), how="cross")
    d = d.with_columns((pl.col("close_time") - pl.duration(hours=pl.col("h"))).alias("t"),
                       pl.col("h").cast(pl.Float64).alias("hours"))
    d = d.filter(pl.col("t") >= pl.col("open_time"))
    # barrier: one row per (market, day at BARRIER_DECISION_HOUR UTC) while open, >= 2 h before the end
    brows = []
    for r in sel.filter(pl.col("family") == "barrier").iter_rows(named=True):
        o, c, x = r["open_time"], r["close_time"], r["expected_expiration_time"]
        t = (o + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        t = t.replace(hour=BARRIER_DECISION_HOUR) if t.hour <= BARRIER_DECISION_HOUR else \
            (t + timedelta(days=1)).replace(hour=BARRIER_DECISION_HOUR)
        while t < c and t <= x - timedelta(hours=2):
            brows.append({"ticker": r["ticker"], "t": t, "hours": (x - t).total_seconds() / 3600})
            t += timedelta(days=1)
    b = sel.filter(pl.col("family") == "barrier").join(
        pl.DataFrame(brows, schema={"ticker": pl.String, "t": pl.Datetime("us", "UTC"), "hours": pl.Float64}),
        on="ticker", how="inner")
    rows = pl.concat([d.drop("h"), b], how="diagonal_relaxed")
    rows = rows.with_columns(pl.col("series_ticker").replace_strict(coin_of, default=None).alias("coin"))
    # spot and variance per unique (coin, t, hours)
    keys = rows.select("coin", "t", "hours").unique()
    vals = []
    for coin, t, hours in keys.iter_rows():
        e = eng.get(coin)
        rec = {"coin": coin, "t": t, "hours": hours, "spot": None, **{f"var_{s}": None for s in VAR_SOURCES}}
        if e is not None:
            rec["spot"] = e.spot(t)
            for s in VAR_SOURCES:
                rec[f"var_{s}"] = e.variance(t, hours, s)
            if rec["var_har"] is not None:
                lev = e.daily_levels(t)
                rec["rv_d"], rec["rv_m"] = lev["rv_d"], lev["rv_m"]
        vals.append(rec)
    vals_df = pl.DataFrame(vals, schema={"coin": pl.String, "t": pl.Datetime("us", "UTC"), "hours": pl.Float64,
                                         "spot": pl.Float64, **{f"var_{s}": pl.Float64 for s in VAR_SOURCES},
                                         "rv_d": pl.Float64, "rv_m": pl.Float64})
    rows = rows.join(vals_df, on=["coin", "t", "hours"], how="left")
    # already touched (barrier) on Coinbase closes since issuance
    touched = []
    for r in rows.select("ticker", "coin", "family", "strike_type", "floor_strike", "cap_strike", "open_time", "t").iter_rows(named=True):
        flag = None
        if r["family"] == "barrier" and r["coin"] in eng:
            kind = "max" if r["strike_type"] == "greater" else "min"
            ext = eng[r["coin"]].path_extreme(int(r["open_time"].timestamp()), int(r["t"].timestamp()), kind)
            k = r["floor_strike"] if kind == "max" else r["cap_strike"]
            flag = ext is not None and ((ext >= k) if kind == "max" else (ext <= k))
        touched.append(flag)
    rows = rows.with_columns(pl.Series("touched_cb", touched, dtype=pl.Boolean))
    rows = _quotes(candles, rows)
    rows = rows.with_columns(
        (pl.col("result") == "yes").cast(pl.Int8).alias("y"),
        pl.when(pl.col("close_time") < DEV_END).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("split"),
        pl.when(pl.col("family") == "barrier").then(pl.lit("barrier"))
        .when(pl.col("strike_type") == "between").then(pl.lit("range")).otherwise(pl.lit("above")).alias("kind"),
        ((pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("ask") > pl.col("bid"))).fill_null(False).alias("two_sided"),
    ).with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    rows = rows.filter(pl.col("result").is_in(["yes", "no"]) & pl.col("spot").is_not_null())
    # quote coverage: candles were fetched newest-first and the run was cut by the API rate, so the
    # dev months before the coverage boundary have model rows (used for the vol fit) but no quotes
    have = set(candles["ticker"].unique().to_list())
    cov = (sel.with_columns(pl.col("ticker").is_in(list(have)).alias("has"), pl.col("close_time").dt.strftime("%Y-%m").alias("month"))
           .group_by("family", "month").agg(pl.len().alias("markets"), pl.col("has").mean().alias("with_candles")).sort("family", "month"))
    _save(cov, OUT / "coverage.parquet")
    log(str(cov))
    _save(rows, OUT / "rows.parquet")
    log(f"rows: {rows.height} ({rows.filter(pl.col('two_sided')).height} two-sided), by kind/split:")
    log(str(rows.group_by("kind", "split").agg(pl.len(), pl.col("two_sided").sum(), pl.col("event_ticker").n_unique().alias("events")).sort("kind", "split")))
    return rows



# ---- model fit and calibration (dev only) -----------------------------------------------------------
K_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.3)
NU_GRID = (3.0, 4.0, 5.0, 7.0, 10.0, 30.0, 1000.0)


def p_vec(df: pl.DataFrame, var: np.ndarray, nu: float) -> np.ndarray:
    """Model YES probability for every row given a variance vector (vectorised ``model_prob``)."""
    s = df["spot"].to_numpy()
    st = df["strike_type"].to_numpy()
    fl = df["floor_strike"].fill_null(np.nan).to_numpy()
    cp = df["cap_strike"].fill_null(np.nan).to_numpy()
    bar = (df["family"] == "barrier").to_numpy()
    var = np.maximum(var, 1e-12)
    sd = np.sqrt(var)

    def above(k):
        return 1 - t_cdf((np.log(k / s) + var / 2) / sd, nu)
    out = np.full(len(s), np.nan)
    g = (st == "greater") & ~bar
    out[g] = above(fl)[g]
    lt = (st == "less") & ~bar
    out[lt] = 1 - above(cp)[lt]
    bt = (st == "between") & ~bar
    out[bt] = (above(fl) - above(cp))[bt]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        bmax = bar & (st == "greater")
        b = np.log(fl / s)
        pm = t_cdf((-b - var / 2) / sd, nu) + np.exp(-b) * t_cdf((-b + var / 2) / sd, nu)
        out[bmax] = pm[bmax]
        bmin = bar & (st == "less")
        b2 = np.log(s / cp)
        pn = t_cdf((-b2 + var / 2) / sd, nu) + np.exp(b2) * t_cdf((-b2 - var / 2) / sd, nu)
        out[bmin] = pn[bmin]
    return np.clip(out, 1e-4, 1 - 1e-4)


def scores(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    q = np.clip(p, 0.005, 0.995)
    return {"brier": float(((p - y) ** 2).mean()), "logloss": float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())}


def fit_params(dev: pl.DataFrame, source: str) -> dict[str, Any]:
    """(variance scale k, tail dof nu) minimising dev log loss over all dev rows with a variance."""
    d = dev.filter(pl.col(f"var_{source}").is_not_null())
    y = d["y"].to_numpy().astype(float)
    v = d[f"var_{source}"].to_numpy()
    best = None
    for nu in NU_GRID:
        for k in K_GRID:
            ll = scores(p_vec(d, v * k, nu), y)["logloss"]
            if best is None or ll < best["logloss"]:
                best = {"source": source, "k": k, "nu": nu, "logloss": ll, "n": d.height}
    return best


def calibrate(*, log=print) -> dict[str, Any]:
    from albert.research.fundamental import fit_logistic, logit
    rows = pl.read_parquet(OUT / "rows.parquet").filter(~pl.col("touched_cb").fill_null(False))
    dev = rows.filter(pl.col("split") == "dev")
    res: dict[str, Any] = {"fits": {}, "chosen": {}, "blend": {}}
    for fam in ("daily", "barrier"):
        dd = dev.filter(pl.col("family") == fam)
        bte = dd.filter(pl.col("coin").is_in(list(DVOL_COINS)))
        fits = {s: fit_params(bte, s) for s in VAR_SOURCES}
        fits_all = fit_params(dd, "har")
        res["fits"][fam] = {"btc_eth": fits, "all_har": fits_all}
        best = dict(min(fits.values(), key=lambda f: f["logloss"]))
        other = dict(fits_all)
        if fam == "daily":   # variance scale per horizon, tail shape fixed (dev only)
            for f, sub in ((best, bte), (other, dd.filter(~pl.col("coin").is_in(list(DVOL_COINS))))):
                f["k_by_h"] = {}
                for h in HORIZONS:
                    x = sub.filter((pl.col("hours") == float(h)) & pl.col(f"var_{f['source']}").is_not_null())
                    y = x["y"].to_numpy().astype(float)
                    v = x[f"var_{f['source']}"].to_numpy()
                    lls = [scores(p_vec(x, v * k, f["nu"]), y)["logloss"] for k in K_GRID]
                    f["k_by_h"][str(h)] = K_GRID[int(np.argmin(lls))]
        res["chosen"][fam] = {"btc_eth": best, "other": other}
        log(f"{fam}: fits {json.dumps(fits)}; all-coins har {json.dumps(fits_all)}; chosen {best['source']}")
    rows = add_model(rows, res["chosen"])
    # blend fitted on dev two-sided rows per kind
    for kind in ("above", "range", "barrier"):
        d = rows.filter((pl.col("split") == "dev") & pl.col("two_sided") & (pl.col("kind") == kind))
        if d.height < 50:
            continue
        X = np.column_stack([np.ones(d.height), logit(d["mid"].to_numpy()), logit(d["p_model"].to_numpy())])
        beta, tstat = fit_logistic(X, d["y"].to_numpy().astype(float), d["event_ticker"].to_numpy())
        res["blend"][kind] = {"a": float(beta[0]), "b": float(beta[1]), "c": float(beta[2]),
                              "t_a": float(tstat[0]), "t_b": float(tstat[1]), "t_c": float(tstat[2]), "n": d.height,
                              "events": d["event_ticker"].n_unique()}
        log(f"blend {kind}: {json.dumps(res['blend'][kind])}")
    rows = add_blend(rows, res["blend"])
    rows = add_xladder(rows, res["chosen"]["daily"]["btc_eth"]["nu"])
    _save(rows, OUT / "rows_model.parquet")
    (OUT / "calibration.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def add_model(rows: pl.DataFrame, chosen: dict[str, Any]) -> pl.DataFrame:
    parts = []
    for fam in ("daily", "barrier"):
        for grp, coins in (("btc_eth", list(DVOL_COINS)), ("other", None)):
            f = chosen[fam][grp]
            m = (pl.col("family") == fam) & (pl.col("coin").is_in(coins) if coins else ~pl.col("coin").is_in(list(DVOL_COINS)))
            d = rows.filter(m & pl.col(f"var_{f['source']}").is_not_null())
            if not d.height:
                continue
            k = np.full(d.height, f["k"])
            if f.get("k_by_h"):
                hrs = d["hours"].to_numpy()
                for h, kh in f["k_by_h"].items():
                    k[hrs == float(h)] = kh
            p = p_vec(d, d[f"var_{f['source']}"].to_numpy() * k, f["nu"])
            parts.append(d.with_columns(pl.Series("p_model", p), pl.lit(f["source"]).alias("model_src")))
    return pl.concat(parts)


def add_blend(rows: pl.DataFrame, blend: dict[str, Any]) -> pl.DataFrame:
    from albert.research.fundamental import logit
    pb = np.full(rows.height, np.nan)
    kind = rows["kind"].to_numpy()
    mid = rows["mid"].fill_null(0.5).to_numpy()
    pm = rows["p_model"].to_numpy()
    for k, b in blend.items():
        m = kind == k
        z = b["a"] + b["b"] * logit(mid[m]) + b["c"] * logit(pm[m])
        pb[m] = 1 / (1 + np.exp(-z))
    return rows.with_columns(pl.Series("p_blend", pb).fill_nan(None))




K_IMPLIED_GRID = np.exp(np.linspace(np.log(0.15), np.log(6.0), 80))


def implied_k(g: pl.DataFrame, nu: float) -> float | None:
    """Variance scale on the HAR forecast that best fits the two-sided above-ladder mids of one
    (event, t), squared error in probability; None without a usable strike."""
    g = g.filter(pl.col("two_sided") & (pl.col("mid") > 0.05) & (pl.col("mid") < 0.95)
                 & (pl.col("strike_type") == "greater"))
    if not g.height:
        return None
    v = g["var_har"].to_numpy()[None, :] * K_IMPLIED_GRID[:, None]
    x = np.log(g["floor_strike"].to_numpy() / g["spot"].to_numpy())[None, :]
    p = 1 - t_cdf((x + v / 2) / np.sqrt(v), nu)
    err = ((p - g["mid"].to_numpy()[None, :]) ** 2).sum(axis=1)
    return float(K_IMPLIED_GRID[int(np.argmin(err))])


def add_xladder(rows: pl.DataFrame, nu: float) -> pl.DataFrame:
    """Cross-ladder fair value for range buckets: the variance implied by the same coin's above/below
    ladder (same 5 pm close, same decision hour, quotes at ``t`` only) applied to the bucket."""
    rows = rows.with_columns(pl.col("event_ticker").str.split("-").list.get(1).alias("ev_key"))
    ab = rows.filter(pl.col("kind") == "above")
    imp = []
    for (coin, key, t), g in ab.group_by(["coin", "ev_key", "t"]):
        k = implied_k(g, nu)
        if k is not None:
            imp.append({"coin": coin, "ev_key": key, "t": t, "k_xl": k})
    imp_df = pl.DataFrame(imp, schema={"coin": pl.String, "ev_key": pl.String, "t": pl.Datetime("us", "UTC"), "k_xl": pl.Float64})
    rows = rows.join(imp_df, on=["coin", "ev_key", "t"], how="left")
    rg = (rows["kind"] == "range").to_numpy() & rows["k_xl"].is_not_null().to_numpy()
    pxl = np.full(rows.height, np.nan)
    if rg.any():
        sub = rows.filter(pl.Series(rg))
        pxl[rg] = p_vec(sub, sub["var_har"].to_numpy() * sub["k_xl"].to_numpy(), nu)
    return rows.with_columns(pl.Series("p_xl", pxl).fill_nan(None))


# ---- trading rules ---------------------------------------------------------------------------------
STAKE_USD = 2.0
MAX_CONTRACTS = 20
MAX_SPREAD = 0.15
MAKER_LIFE_H = {"daily": 2, "barrier": 24}
TAKER_MARGINS = (0.02, 0.04, 0.08)
MAKER_MARGINS = (0.02, 0.05)
HGROUPS = {"short": (2, 4), "long": (8, 16, 24)}


def _n(price: float) -> int:
    return int(min(MAX_CONTRACTS, max(1, math.floor(STAKE_USD / max(price, 0.01)))))


def _fee(n: int, price: float, fee_type: str | None, mult: float | None, maker: bool) -> float:
    from albert.fees import kalshi_fee
    return kalshi_fee(n, price, maker=maker, fee_type=fee_type or "quadratic", multiplier=1.0 if mult is None else mult)


def cell_rows(rows: pl.DataFrame, kind: str, hgroup: str) -> pl.DataFrame:
    r = rows.filter((pl.col("kind") == kind) & pl.col("two_sided") & ((pl.col("ask") - pl.col("bid")) <= MAX_SPREAD + 1e-9))
    if kind != "barrier":
        r = r.filter(pl.col("hours").is_in([float(h) for h in HGROUPS[hgroup]]))
    return r


def taker_trades(r: pl.DataFrame, pcol: str, margin: float, fees: dict[str, tuple[str, float]]) -> pl.DataFrame:
    """First decision per market where the model edge net of the taker fee is >= margin."""
    out = []
    for x in r.filter(pl.col(pcol).is_not_null()).sort("t").iter_rows(named=True):
        ft, mult = fees.get(x["series_ticker"], ("quadratic", 1.0))
        p = x[pcol]
        for side, price, win in (("yes", x["ask"], x["y"] == 1), ("no", 1 - x["bid"], x["y"] == 0)):
            n = _n(price)
            fee = _fee(n, price, ft, mult, False)
            edge = (p if side == "yes" else 1 - p) - price - fee / n
            if edge >= margin - 1e-12:
                out.append({**{k: x[k] for k in ("ticker", "event_ticker", "series_ticker", "coin", "kind", "split", "t",
                                                  "hours", "settlement_ts", "close_time")},
                            "side": side, "price": price, "n": n, "fee": fee, "edge": edge, "p": p,
                            "t_in": x["t"], "payoff": float(win)})
                break
    df = pl.DataFrame(out) if out else pl.DataFrame()
    if df.height:
        df = df.sort("t").unique(["ticker"], keep="first")
    return df


def maker_trades(r: pl.DataFrame, pcol: str, margin: float, candles: pl.DataFrame,
                 fees: dict[str, tuple[str, float]]) -> pl.DataFrame:
    """Rest a YES bid at floor(p - m) and a NO bid at floor(1 - p - m) (never crossing) for
    MAKER_LIFE_H hours; filled only if a later hourly candle traded strictly through (YES: a print
    below our bid; NO: a print above 1 - our NO bid). First fill per (market, side), held to settlement."""
    orders = []
    for x in r.filter(pl.col(pcol).is_not_null()).iter_rows(named=True):
        p = x[pcol]
        b = math.floor((p - margin) * 100 + 1e-9) / 100
        if 0.01 <= b < x["ask"]:
            orders.append({**x, "side": "yes", "price": b})
        q = math.floor((1 - p - margin) * 100 + 1e-9) / 100
        if 0.01 <= q < 1 - x["bid"]:
            orders.append({**x, "side": "no", "price": q})
    if not orders:
        return pl.DataFrame()
    o = pl.DataFrame(orders, infer_schema_length=None).select(
        "ticker", "event_ticker", "series_ticker", "coin", "kind", "family", "split", "t", "hours", "settlement_ts",
        "close_time", "y", "side", "price", pl.col(pcol).alias("p")).with_row_index("oid")
    life = o["family"].replace_strict(MAKER_LIFE_H, default=2).cast(pl.Int64)
    o = o.with_columns((pl.col("t") + pl.duration(hours=life)).alias("t_end"))
    # the order's hourly candles: end_ts = t + 1h ... t + life (equi-join, no cross product)
    slots = o.select("oid", "ticker", pl.datetime_ranges(pl.col("t") + timedelta(hours=1), pl.col("t_end"), "1h")
                     .alias("end_ts"), "close_time").explode("end_ts")
    slots = slots.filter(pl.col("end_ts") <= pl.col("close_time") + timedelta(hours=1)).with_columns(
        pl.col("end_ts").cast(pl.Datetime("us", "UTC")))
    c = candles.select("ticker", pl.col("end_ts").cast(pl.Datetime("us", "UTC")), "price_low", "price_high")
    j = slots.join(c, on=["ticker", "end_ts"], how="inner").join(o.select("oid", "side", "price"), on="oid")
    hit = pl.when(pl.col("side") == "yes").then(pl.col("price_low") < pl.col("price") - 1e-9) \
        .otherwise(pl.col("price_high") > 1 - pl.col("price") + 1e-9)
    f = j.filter(hit.fill_null(False)).group_by("oid").agg(pl.col("end_ts").min().alias("t_in"))
    fills = o.join(f, on="oid", how="inner").sort("t_in").unique(["ticker", "side"], keep="first")
    if not fills.height:
        return pl.DataFrame()
    n = [_n(pr) for pr in fills["price"].to_list()]
    fee = [_fee(k, pr, *fees.get(sr, ("quadratic", 1.0)), True)
           for k, pr, sr in zip(n, fills["price"].to_list(), fills["series_ticker"].to_list(), strict=True)]
    return fills.with_columns(pl.Series("n", n), pl.Series("fee", fee),
                              pl.when(pl.col("side") == "yes").then(pl.col("y") == 1).otherwise(pl.col("y") == 0)
                              .cast(pl.Float64).alias("payoff"),
                              (pl.when(pl.col("side") == "yes").then(pl.col("p")).otherwise(1 - pl.col("p")) - pl.col("price")).alias("edge"))



def favourite_trades(r: pl.DataFrame, fees: dict[str, tuple[str, float]], lo: float = 0.55, hi: float = 0.92) -> pl.DataFrame:
    """Model-free control: buy the favourite side at the touch when its price is in [lo, hi]."""
    x = r.with_columns(pl.lit(None, pl.Float64).alias("p_none"))
    out = []
    for row in x.sort("t").iter_rows(named=True):
        ft, mult = fees.get(row["series_ticker"], ("quadratic", 1.0))
        for side, price, win in (("yes", row["ask"], row["y"] == 1), ("no", 1 - row["bid"], row["y"] == 0)):
            if lo <= price <= hi:
                n = _n(price)
                out.append({**{k: row[k] for k in ("ticker", "event_ticker", "series_ticker", "coin", "kind", "split", "t",
                                                    "hours", "settlement_ts", "close_time")},
                            "side": side, "price": price, "n": n, "fee": _fee(n, price, ft, mult, False), "edge": None,
                            "p": None, "t_in": row["t"], "payoff": float(win)})
                break
    df = pl.DataFrame(out) if out else pl.DataFrame()
    return df.sort("t").unique(["ticker"], keep="first") if df.height else df


def pnl_stats(tr: pl.DataFrame) -> dict[str, Any]:
    from albert.research.counts import cluster_se
    if not tr.height:
        return {"fills": 0, "events": 0, "c_per_contract": None, "t_event": None, "t_2way": None, "pnl_per_usd": None}
    x = (100 * (tr["payoff"] - tr["price"] - tr["fee"] / tr["n"])).to_numpy()
    ev = tr["event_ticker"].to_numpy().astype(str)
    dt = tr["t_in"].dt.date().cast(pl.String).to_numpy().astype(str)
    m = float(x.mean())
    se_e = cluster_se(x, ev) if len(np.unique(ev)) > 1 else float("nan")
    se_2 = cluster_se(x, ev, dt) if len(np.unique(ev)) > 1 and len(np.unique(dt)) > 1 else float("nan")
    cost = (tr["n"] * tr["price"] + tr["fee"]).sum()
    pnl = (tr["n"] * tr["payoff"] - tr["n"] * tr["price"] - tr["fee"]).sum()
    return {"fills": tr.height, "events": len(np.unique(ev)), "c_per_contract": round(m, 2),
            "t_event": round(m / se_e, 2) if se_e and math.isfinite(se_e) and se_e > 0 else None,
            "t_2way": round(m / se_2, 2) if se_2 and math.isfinite(se_2) and se_2 > 0 else None,
            "pnl_per_usd": round(float(pnl / cost), 4) if cost else None,
            "win_pct": round(100 * float((tr["payoff"] > 0).mean()), 1), "avg_price": round(float(tr["price"].mean()), 3)}


def fee_map() -> dict[str, tuple[str, float]]:
    """Each series' own fee_type / fee_multiplier from today's /series payload."""
    c = kalshi_client()
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    out = {}
    for s in [*DAILY, *BARRIER]:
        p = c.get(f"/series/{s}", tag=day).payload.get("series") or {}
        out[s] = (p.get("fee_type") or "quadratic", float(p.get("fee_multiplier") if p.get("fee_multiplier") is not None else 1.0))
    return out


def cells() -> list[dict[str, Any]]:
    out = []
    for kind, hgs in (("above", ("short", "long")), ("range", ("short", "long")), ("barrier", ("all",))):
        for hg in hgs:
            for pcol in ("p_model", "p_blend", "p_xl") if kind == "range" else ("p_model", "p_blend"):
                for m in TAKER_MARGINS:
                    out.append({"mode": "taker", "kind": kind, "hgroup": hg, "pcol": pcol, "margin": m})
                for m in MAKER_MARGINS:
                    out.append({"mode": "maker", "kind": kind, "hgroup": hg, "pcol": pcol, "margin": m})
    return out


def cell_name(c: dict[str, Any]) -> str:
    return f"{c['mode']}|{c['kind']}|{c['hgroup']}|{c['pcol']}|m{round(c['margin'] * 100)}"


def run_cell(rows: pl.DataFrame, c: dict[str, Any], candles: pl.DataFrame, fees) -> pl.DataFrame:
    r = cell_rows(rows, c["kind"], c["hgroup"])
    if c["mode"] == "taker":
        return taker_trades(r, c["pcol"], c["margin"], fees)
    return maker_trades(r, c["pcol"], c["margin"], candles, fees)


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj.tolist()



# ---- evaluation --------------------------------------------------------------------------------------
MIN_FILLS, MIN_EVENTS = 50, 30
EVENT_CAP_USD, OPEN_CAP_USD = 5.0, 20.0


def ledger(tr: pl.DataFrame, variant: str, split: str) -> pl.DataFrame:
    """Capped $100 book: <= $5 cost per event, <= $20 open; trades in fill order."""
    rows, open_pos, per_event = [], [], {}
    for x in tr.sort("t_in").iter_rows(named=True):
        settle = x["settlement_ts"] or x["close_time"]
        open_pos = [(s, c) for s, c in open_pos if s > x["t_in"]]
        cost = x["n"] * x["price"] + x["fee"]
        if per_event.get(x["event_ticker"], 0.0) + cost > EVENT_CAP_USD + 1e-9:
            continue
        if sum(c for _, c in open_pos) + cost > OPEN_CAP_USD + 1e-9:
            continue
        per_event[x["event_ticker"]] = per_event.get(x["event_ticker"], 0.0) + cost
        open_pos.append((settle, cost))
        rows.append({"strategy": "crypto_fv", "variant": variant, "split": split,
                     "struct_id": f"{x['ticker']}|{x['side']}|{x['t_in']:%Y%m%dT%H}", "event_ticker": x["event_ticker"],
                     "ticker": x["ticker"], "side": x["side"], "n": int(x["n"]), "t_in": x["t_in"], "cost_usd": cost,
                     "settle_ts": settle, "payoff_usd": x["n"] * x["payoff"]})
    return pl.DataFrame(rows, schema={"strategy": pl.String, "variant": pl.String, "split": pl.String,
                                      "struct_id": pl.String, "event_ticker": pl.String, "ticker": pl.String,
                                      "side": pl.String, "n": pl.Int64, "t_in": pl.Datetime("us", "UTC"),
                                      "cost_usd": pl.Float64, "settle_ts": pl.Datetime("us", "UTC"),
                                      "payoff_usd": pl.Float64})


def book_metrics(led: pl.DataFrame) -> dict[str, Any]:
    from albert.research.vol_metrics import ledger_metrics, load_marks
    if not led.height:
        return {}
    marks = load_marks(led["ticker"].unique().to_list())
    return ledger_metrics(led, marks)


def evaluate(*, log=print) -> dict[str, Any]:
    from scipy import stats
    rows = pl.read_parquet(OUT / "rows_model.parquet")
    candles = load_candles()
    fees = fee_map()
    dev = rows.filter(pl.col("split") == "dev")
    res = []
    for c in cells():
        st = pnl_stats(run_cell(dev, c, candles, fees))
        res.append({"name": cell_name(c), **c, **st})
    t = np.array([r["t_event"] if r["t_event"] is not None else 0.0 for r in res])
    ev = np.array([max(r["events"], 2) for r in res])
    pv = [float(stats.t.sf(ti, df=e - 1)) for ti, e in zip(t, ev, strict=True)]   # one-sided, H1: mean > 0
    adj = holm(pv)
    for r, p_, a_ in zip(res, pv, adj, strict=True):
        r["p_one_sided"], r["p_holm"] = round(p_, 4), round(a_, 4)
    grid = pl.DataFrame(res)
    _save(grid, OUT / "grid_dev.parquet")
    eligible = [r for r in res if r["fills"] >= MIN_FILLS and r["events"] >= MIN_EVENTS and (r["c_per_contract"] or 0) > 0
                and r["t_event"] is not None]
    # freeze two rules on dev: the best eligible cell (event-clustered t) and the best eligible cell of
    # another contract kind; the holdout is then scored once for each (Holm over the two)
    picks = []
    if eligible:
        first = max(eligible, key=lambda r: r["t_event"])
        picks.append(first)
        other = [r for r in eligible if r["kind"] != first["kind"]]
        if other:
            picks.append(max(other, key=lambda r: r["t_event"]))
    specs = []
    for i, r in enumerate(picks):
        spec = {k: r[k] for k in ("name", "mode", "kind", "hgroup", "pcol", "margin")}
        spec["role"] = "primary" if i == 0 else "secondary"
        specs.append(spec)
    frozen_doc = {"frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"), "rules": specs,
                  "criteria": f"max dev t_event among cells with >= {MIN_FILLS} fills, >= {MIN_EVENTS} events, mean > 0",
                  "calibration": json.loads((OUT / "calibration.json").read_text())["chosen"]}
    (OUT / "frozen.json").write_text(json.dumps(frozen_doc, indent=1, default=str))
    log(f"frozen on dev: {json.dumps(specs)}")
    # ---- holdout, scored once per frozen rule
    out: dict[str, Any] = {"cells": len(res), "frozen": specs, "results": []}
    hold = rows.filter(pl.col("split") == "holdout")
    (OUT / "ledgers").mkdir(parents=True, exist_ok=True)
    hp = []
    for spec, devrow in zip(specs, picks, strict=True):
        tr_d = run_cell(dev, spec, candles, fees)
        tr_h = run_cell(hold, spec, candles, fees)
        rec: dict[str, Any] = {"rule": spec, "dev": {k: devrow[k] for k in devrow if k not in spec}, "holdout": pnl_stats(tr_h)}
        th = rec["holdout"]["t_event"]
        hp.append(float(stats.t.sf(th, df=max(rec["holdout"]["events"], 2) - 1)) if th is not None else 1.0)
        # model-free control on the same rows: buy the favourite side (55-92c) at the touch
        rec["control_favourite"] = {"dev": pnl_stats(favourite_trades(cell_rows(dev, spec["kind"], spec["hgroup"]), fees)),
                                    "holdout": pnl_stats(favourite_trades(cell_rows(hold, spec["kind"], spec["hgroup"]), fees))}
        tag = spec["role"]
        led_d, led_h = ledger(tr_d, spec["name"], "dev"), ledger(tr_h, spec["name"], "holdout")
        for nm, led in (("dev", led_d), ("holdout", led_h)):
            if led.height:
                _save(led, OUT / "ledgers" / f"ledger_{tag}_{nm}.parquet")
        rec["book_dev"], rec["book_holdout"] = book_metrics(led_d), book_metrics(led_h)
        if led_d.height and led_h.height:
            rec["book_all"] = book_metrics(pl.concat([led_d, led_h]))
        for nm, tr in (("dev", tr_d), ("holdout", tr_h)):
            if tr.height:
                _save(tr, OUT / f"trades_{tag}_{nm}.parquet")
        out["results"].append(rec)
    for rec, a_ in zip(out["results"], holm(hp) if hp else [], strict=True):
        rec["holdout_p_one_sided_holm"] = round(a_, 4)
    (OUT / "evaluation.json").write_text(json.dumps(out, indent=1, default=str))
    log(json.dumps([{"rule": r["rule"]["name"], "dev": r["dev"], "holdout": r["holdout"]} for r in out["results"]], default=str))
    return out


# ---- diagnostics -------------------------------------------------------------------------------------
def hbucket(col: str = "hours") -> pl.Expr:
    return (pl.when(pl.col(col) <= 4).then(pl.lit("02-04h")).when(pl.col(col) <= 24).then(pl.lit("08-24h"))
            .when(pl.col(col) <= 24 * 7).then(pl.lit("1-7d")).otherwise(pl.lit(">7d")))


def score_table(rows: pl.DataFrame) -> pl.DataFrame:
    """Brier / log loss of mid, model and blend on two-sided rows, by kind x split x horizon bucket."""
    r = rows.filter(pl.col("two_sided") & pl.col("p_blend").is_not_null()).with_columns(hbucket().alias("hb"))
    out = []
    for (kind, split, hb), g in r.group_by(["kind", "split", "hb"]):
        y = g["y"].to_numpy().astype(float)
        rec = {"kind": kind, "split": split, "horizon": hb, "rows": g.height, "events": g["event_ticker"].n_unique(),
               "mean_spread_c": round(100 * float((g["ask"] - g["bid"]).mean()), 2)}
        for nm in ("mid", "p_model", "p_blend"):
            sc = scores(g[nm].to_numpy(), y)
            rec[f"brier_{nm}"], rec[f"ll_{nm}"] = round(sc["brier"], 4), round(sc["logloss"], 4)
        out.append(rec)
    return pl.DataFrame(out).sort("kind", "split", "horizon")


def bucket_table(rows: pl.DataFrame) -> pl.DataFrame:
    """Calibration by mid bucket: realized YES rate vs mean mid vs mean model (two-sided rows, dev)."""
    edges = [0, 0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 1.0]
    labels = ["<5c", "5-15c", "15-35c", "35-65c", "65-85c", "85-95c", ">95c"]
    r = rows.filter(pl.col("two_sided")).with_columns(
        pl.col("mid").cut(edges[1:-1], labels=labels).alias("bucket"))
    return (r.group_by("kind", "split", "bucket").agg(
        pl.len().alias("rows"), pl.col("event_ticker").n_unique().alias("events"), pl.col("y").mean().alias("yes_rate"),
        pl.col("mid").mean().alias("mid"), pl.col("p_model").mean().alias("model"),
        ((pl.col("y") - pl.col("mid")) * 100).mean().alias("y_minus_mid_c"))
        .sort("kind", "split", "bucket"))


def skew_table(rows: pl.DataFrame) -> pl.DataFrame:
    """Above ladders: model - mid (c) and y - mid (c) by strike side and distance (z at decision)."""
    r = rows.filter(pl.col("two_sided") & (pl.col("kind") == "above")).with_columns(
        ((pl.col("floor_strike") / pl.col("spot")).log() / pl.col("var_har").sqrt()).alias("z"))
    r = r.with_columns(pl.col("z").cut([-2, -1, -0.25, 0.25, 1, 2], labels=["<-2", "-2..-1", "-1..-.25", "atm", ".25..1", "1..2", ">2"]).alias("zb"))
    return (r.group_by("split", "zb").agg(pl.len().alias("rows"), ((pl.col("p_model") - pl.col("mid")) * 100).mean().alias("model_minus_mid_c"),
                                          ((pl.col("y") - pl.col("mid")) * 100).mean().alias("y_minus_mid_c"),
                                          pl.col("mid").mean().alias("mid"))
            .sort("split", "zb"))


def implied_vol_table(rows: pl.DataFrame) -> pl.DataFrame:
    """Per (event, t) of the above ladders: the variance scale that best fits the two-sided mids under
    the chosen tail shape, vs the HAR forecast and the realized squared log return to settlement."""
    cal = json.loads((OUT / "calibration.json").read_text())
    nu = cal["chosen"]["daily"]["btc_eth"]["nu"]
    r = rows.filter(pl.col("kind") == "above")
    out = []
    for (ev, t), g in r.group_by(["event_ticker", "t"]):
        k = implied_k(g, nu)
        if k is None:
            continue
        v = g["var_har"].to_numpy()
        ev_val = g["expiration_value"][0]
        out.append({"event_ticker": ev, "t": t, "coin": g["coin"][0], "split": g["split"][0], "hours": g["hours"][0],
                    "k_implied": k, "var_har": float(v[0]), "var_dvol": g["var_dvol"][0],
                    "real_sq": math.log(ev_val / g["spot"][0]) ** 2 if ev_val else None})
    d = pl.DataFrame(out)
    if not d.height:
        return d
    return (d.with_columns(hbucket().alias("hb"))
            .group_by("coin", "split", "hb").agg(
                pl.len().alias("ladders"), pl.col("k_implied").median().alias("k_implied_median"),
                (pl.col("real_sq").mean() / pl.col("var_har").mean()).alias("realized_over_har"),
                (pl.col("real_sq").mean() / (pl.col("var_har") * pl.col("k_implied")).mean()).alias("realized_over_implied"),
                (pl.col("var_dvol").mean() / pl.col("var_har").mean()).alias("dvol_over_har"))
            .sort("coin", "split", "hb"))


def monotone_table(rows: pl.DataFrame) -> pl.DataFrame:
    """Above ladders: pairs K1 < K2 at the same (event, t) where ask(K1) < bid(K2), i.e. buying YES K1
    and NO K2 costs < $1 for a payoff >= $1 (hourly candle closes; not simultaneous, size unknown)."""
    r = rows.filter(pl.col("two_sided") & (pl.col("kind") == "above")).select(
        "event_ticker", "t", "split", "floor_strike", "bid", "ask")
    j = r.join(r, on=["event_ticker", "t", "split"], suffix="_2").filter(pl.col("floor_strike") < pl.col("floor_strike_2"))
    j = j.with_columns((pl.col("bid_2") - pl.col("ask")).alias("gap"))
    return j.group_by("split").agg(pl.len().alias("pairs"), (pl.col("gap") > 0).sum().alias("violations"),
                                   pl.col("gap").filter(pl.col("gap") > 0).mean().alias("mean_gap"))




# ---- report ------------------------------------------------------------------------------------------
def _md(df: pl.DataFrame, floats: int = 3) -> str:
    if not df.height:
        return "(none)\n"
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        cells_ = []
        for v in r:
            if isinstance(v, float):
                cells_.append(f"{v:.{floats}f}" if math.isfinite(v) else "")
            else:
                cells_.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells_) + " |")
    return "\n".join(lines) + "\n"


def diagnostics(*, log=print) -> dict[str, pl.DataFrame]:
    rows = pl.read_parquet(OUT / "rows_model.parquet")
    out = {"scores": score_table(rows), "buckets": bucket_table(rows.filter(pl.col("split") == "dev")),
           "skew": skew_table(rows.filter(pl.col("split") == "dev")), "implied": implied_vol_table(rows),
           "monotone": monotone_table(rows)}
    for k, v in out.items():
        _save(v, OUT / f"diag_{k}.parquet")
        log(f"--- {k}\n{v}")
    return out



def write_tables() -> Path:
    """All tables (calibration fits, scores, buckets, skew, implied vol, consistency, dev grid,
    frozen-rule results) into ``tables.md``; ``report.md`` (hand-written) cites them."""
    parts = ["# crypto_fv tables (generated)\n"]
    cal = json.loads((OUT / "calibration.json").read_text())
    parts.append("## Model fits on dev (variance scale k on the forecast, Student-t dof nu, dev log loss)\n")
    for fam, f in cal["fits"].items():
        for src, v in f["btc_eth"].items():
            parts.append(f"- {fam} BTC/ETH {src}: k={v['k']}, nu={v['nu']}, logloss={v['logloss']:.4f}, n={v['n']}")
        v = f["all_har"]
        parts.append(f"- {fam} all coins har: k={v['k']}, nu={v['nu']}, logloss={v['logloss']:.4f}, n={v['n']}")
    parts.append("\n## Blend logit P = a + b logit(mid) + c logit(model), dev two-sided rows (t by event)\n")
    for kind, b in cal["blend"].items():
        parts.append(f"- {kind}: a={b['a']:.3f} (t {b['t_a']:.1f}), b={b['b']:.3f} (t {b['t_b']:.1f}), "
                     f"c={b['c']:.3f} (t {b['t_c']:.1f}), rows={b['n']}, events={b['events']}")
    for name, title in (("scores", "Scores on two-sided rows (mid vs model vs blend)"),
                        ("buckets", "Calibration by mid bucket (dev)"),
                        ("skew", "Above ladders: model - mid and outcome - mid by strike distance z (dev)"),
                        ("implied", "Implied variance scale k (fit to above-ladder mids) and realized / forecast variance"),
                        ("monotone", "Above-ladder monotonicity violations (ask(K1) < bid(K2), K1 < K2)")):
        p = OUT / f"diag_{name}.parquet"
        if p.exists():
            parts.append(f"\n## {title}\n")
            parts.append(_md(pl.read_parquet(p), 4))
    g = OUT / "grid_dev.parquet"
    if g.exists():
        parts.append("\n## Dev grid (every cell tried; one-sided p and Holm-adjusted p)\n")
        parts.append(_md(pl.read_parquet(g).select("name", "fills", "events", "c_per_contract", "t_event", "t_2way",
                                                   "pnl_per_usd", "win_pct", "avg_price", "p_one_sided", "p_holm")
                         .sort("t_event", descending=True, nulls_last=True), 3))
    ev = OUT / "evaluation.json"
    if ev.exists():
        parts.append("\n## Frozen rule: dev and holdout\n")
        parts.append("```\n" + ev.read_text() + "\n```\n")
    out = OUT / "tables.md"
    out.write_text("\n".join(parts))
    return out


# ---- test fixtures: trimmed copies of recorded responses ----------------------------------------------
FIXTURES = Path("tests/fixtures/crypto_fv")


def make_fixtures() -> list[str]:
    """Trim one recorded raw response of each kind into ``tests/fixtures/crypto_fv``."""
    from albert.http import read_raw
    FIXTURES.mkdir(parents=True, exist_ok=True)
    want = {
        "events": lambda r: r["url"].endswith("/events") and r["params"].get("series_ticker") == "KXBTCD",
        "markets_live": lambda r: r["url"].endswith("/v2/markets") and "event_ticker" in r["params"]
        and (r["response"].get("markets") or [{}])[0].get("strike_type") == "greater" and "KXBTCD" in r["params"]["event_ticker"],
        "markets_hist": lambda r: r["url"].endswith("/historical/markets") and "event_ticker" in r["params"]
        and bool(r["response"].get("markets")),
        "markets_range": lambda r: r["url"].endswith("/v2/markets") and r["params"].get("event_ticker", "").startswith("KXBTC-")
        and bool(r["response"].get("markets")),
        "markets_barrier": lambda r: "MAXMON" in str(r["params"].get("event_ticker", "")) and bool(r["response"].get("markets")),
        "candles_batch": lambda r: r["url"].endswith("/markets/candlesticks")
        and any(m.get("candlesticks") for m in r["response"].get("markets") or []),
        "candles_hist": lambda r: "/historical/markets/" in r["url"] and r["url"].endswith("/candlesticks")
        and len(r["response"].get("candlesticks") or []) > 3,
        "coinbase": lambda r: "coinbase" in r["url"] and bool(r["response"].get("candles")),
        "dvol": lambda r: "deribit" in r["url"] and bool((r["response"].get("result") or {}).get("data")),
    }
    got: dict[str, dict[str, Any]] = {}
    for venue in ("kalshi", "coinbase", "deribit"):
        for path in sorted((CACHE / "raw" / venue).glob("*.json*")):
            if len(got) == len(want):
                break
            rec = read_raw(path)
            for name, pred in want.items():
                if name not in got and isinstance(rec.get("response"), dict):
                    try:
                        ok = pred(rec)
                    except Exception:   # noqa: BLE001
                        ok = False
                    if ok:
                        got[name] = rec
    written = []
    for name, rec in got.items():
        resp = json.loads(json.dumps(rec["response"]))
        for key in ("events", "markets", "candlesticks", "candles"):
            if isinstance(resp.get(key), list):
                if key == "markets" and name == "candles_batch":
                    resp[key] = [{**m, "candlesticks": m["candlesticks"][:6]} for m in resp[key] if m.get("candlesticks")][:2]
                else:
                    resp[key] = resp[key][:350 if name == "coinbase" else 6]
        if name == "dvol":
            resp["result"]["data"] = resp["result"]["data"][:6]
        (FIXTURES / f"{name}.json").write_text(json.dumps({"url": rec["url"], "params": rec["params"],
                                                           "fetched_at": rec["fetched_at"], "response": resp}, indent=1))
        written.append(name)
    return written


# ---- CLI -------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m albert.research.crypto_fv")
    ap.add_argument("cmd")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--part", default="a")
    a = ap.parse_args(argv)
    if a.cmd == "robots":
        for r in check_robots():
            print(r["host"], r["path"], r["http_status"], r["allowed"])
    elif a.cmd == "events":
        fetch_events(list(DAILY) + list(BARRIER))
    elif a.cmd == "spot":
        fetch_spot()
    elif a.cmd == "dvol":
        fetch_dvol()
    elif a.cmd == "markets":
        fetch_markets(workers=a.workers)
    elif a.cmd == "select":
        sel = selection()
        print(sel.group_by("family", "series_ticker").len().sort("family", "series_ticker"))
    elif a.cmd == "candles":
        sel = pl.read_parquet(CACHE / "selection.parquet")
        fetch_candles(sel, workers=a.workers, part=a.part)
    elif a.cmd == "build":
        build_rows()
    elif a.cmd == "fetchall":   # one light-lane download job: markets || (spot, dvol), then selection, candles
        import threading

        def spot_dvol():
            if not (CACHE / "spot_5m.parquet").exists():
                fetch_spot()
            if not (CACHE / "dvol_1h.parquet").exists():
                fetch_dvol()
        th = threading.Thread(target=spot_dvol)
        th.start()
        if not (CACHE / "markets.parquet").exists():
            fetch_markets(workers=a.workers)
        th.join()
        sel = selection()
        print(sel.group_by("family", "series_ticker").len().sort("family", "series_ticker"), flush=True)
        fetch_candles(sel, workers=a.workers, part=a.part)
    elif a.cmd == "calib":
        calibrate()
    elif a.cmd == "evaluate":
        evaluate()
    elif a.cmd == "diag":
        diagnostics()
    elif a.cmd == "tables":
        print(write_tables())
    elif a.cmd == "fixtures":
        print(make_fixtures())
    else:
        raise SystemExit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    main()
