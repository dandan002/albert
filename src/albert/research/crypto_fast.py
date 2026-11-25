"""Short-horizon Kalshi crypto contracts: settlement mechanics, staleness and lead-lag (crypto_fast).

Research exception (user decision 2026-11-23): crypto contracts are in scope for this study only;
any live use needs a separate explicit user decision. Kalshi public GET endpoints only; spot from
public market-data endpoints (Coinbase Advanced Trade public candles, Kraken public trades and
WebSockets); no account, no authentication, no orders anywhere.

Series (selected here by ticker, not through ``albert.categories``, which marks crypto as financial):
15-minute up/down ``KX{BTC,ETH,SOL,XRP,DOGE}15M``. Rules (read from the API, 2026-11-23): YES if the
simple average of the 60 one-second CF Benchmarks RTI values before the close is at least the same
60-second average before the open (``floor_strike`` = the previous window's ``expiration_value``).
Fees: ``fee_type`` quadratic, multiplier 1 (taker 0.07 P(1-P) per contract, no maker fee).

Pipeline (``python -m albert.research.crypto_fast <cmd>``):

* ``markets``  settled markets per series (``/historical/markets?series_ticker=`` before the
  historical cutoff, ``/markets?series_ticker=&status=settled`` after).
* ``spot``     Coinbase 1-minute candles (``api.coinbase.com/api/v3/brokerage/market/...``).
* ``sample``   volume-blind sample: a fixed share of windows chosen by a hash of the ticker.
* ``candles``  Kalshi 1-minute candles for the sample (historical per market, live in batches).
* ``build``    minute grid per sampled window: Kalshi bid/ask, spot, fair value, outcome.
* ``study``    calibration, lead-lag, staleness rules on dev; frozen rule scored once on holdout.
* ``record``   live recording: Kalshi order book by REST (<= 2 req/s) + Coinbase/Kraken WebSockets.
* ``live``     second-level staleness and latency analysis of the recording.

Result (2026-11-23; numbers in data/research/crypto_fast/summary.json): Kalshi moves with Coinbase
spot within the same second (no staleness at our latency). The frozen minute-scale rule (spot-model
gap >= 12c at minutes 5-9, taker) was flat on dev (+0.8c, t 0.17) and +7.7c/contract on the holdout
(t 2.25, 153 windows, tape-verified), concentrated in DOGE/XRP: a marginal, regime-dependent candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
COINBASE_BASE = "https://api.coinbase.com/api/v3/brokerage/market"
KRAKEN_BASE = "https://api.kraken.com/0/public"
CACHE = Path("data/cache/crypto_fast")
OUT = Path("data/research/crypto_fast")
UA = "albert-research/0.2"

SERIES_15M: dict[str, str] = {   # Kalshi series -> Coinbase product (spot proxy for the CF RTI)
    "KXBTC15M": "BTC-USD", "KXETH15M": "ETH-USD", "KXSOL15M": "SOL-USD",
    "KXXRP15M": "XRP-USD", "KXDOGE15M": "DOGE-USD",
}
START = datetime(2026, 7, 1, tzinfo=UTC)
DEV_END = datetime(2026, 10, 1, tzinfo=UTC)       # dev: close < DEV_END
HOLD_END = datetime(2026, 11, 23, tzinfo=UTC)     # holdout: DEV_END <= close < HOLD_END

MARKET_COLS = {
    "ticker": pl.String, "event_ticker": pl.String, "series": pl.String,
    "open_time": pl.Datetime("us", "UTC"), "close_time": pl.Datetime("us", "UTC"),
    "settlement_ts": pl.Datetime("us", "UTC"), "strike_type": pl.String,
    "floor_strike": pl.Float64, "cap_strike": pl.Float64, "expiration_value": pl.Float64,
    "result": pl.String, "volume": pl.Float64, "status": pl.String, "src": pl.String,
}


# ---- shared helpers ------------------------------------------------------------------------------
def _ts(v: Any) -> datetime | None:
    if v in (None, "", "0001-01-01T00:00:00Z"):
        return None
    if isinstance(v, int | float):
        return datetime.fromtimestamp(v, tz=UTC)
    return datetime.fromisoformat(str(v)).astimezone(UTC)


def _f(v: Any) -> float | None:
    if v in (None, ""):
        return None
    if isinstance(v, str):
        v = v.replace(",", "")   # some historical expiration values read "77,362.10"
    return float(v)


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def kalshi_client(min_interval: float = 0.6):
    from albert.http import CachedClient
    return CachedClient("kalshi", KALSHI_BASE, CACHE, min_interval=max(0.5, min_interval),
                        max_retries=12, user_agent=UA)


# ---- markets -------------------------------------------------------------------------------------
def market_row(m: dict[str, Any], src: str) -> dict[str, Any]:
    """One settled market from a ``/markets`` or ``/historical/markets`` node."""
    ev = m.get("event_ticker") or m["ticker"].rsplit("-", 1)[0]
    return {
        "ticker": m["ticker"], "event_ticker": ev, "series": ev.split("-", 1)[0],
        "open_time": _ts(m.get("open_time")), "close_time": _ts(m.get("close_time")),
        "settlement_ts": _ts(m.get("settlement_ts")), "strike_type": m.get("strike_type"),
        "floor_strike": _f(m.get("floor_strike")), "cap_strike": _f(m.get("cap_strike")),
        "expiration_value": _f(m.get("expiration_value")), "result": m.get("result") or None,
        "volume": _f(m.get("volume_fp", m.get("volume"))), "status": m.get("status"), "src": src,
    }


def fetch_markets(series: str, start: datetime = START, end: datetime = HOLD_END, *, log=print) -> pl.DataFrame:
    """Every settled market of ``series`` that closed in [start, end) (selected on series identity
    and close time only)."""
    from albert.venues.kalshi import Kalshi
    c = kalshi_client()
    k = Kalshi(c)
    cutoff = k.cutoff()["market_settled_ts"]
    rows: list[dict[str, Any]] = []
    if cutoff is None or end > cutoff:
        lo = max(start, cutoff) if cutoff else start
        for page in k.series_markets(series, "settled", min_close=lo, max_close=end, cache=False):
            rows += [market_row(m, "live") for m in page.payload.get("markets", [])]
        log(f"{series}: live {len(rows)}")
    if cutoff is not None and start < cutoff:
        cursor, n0 = None, len(rows)
        while True:
            page = c.get("/historical/markets", {"series_ticker": series, "limit": 1000, "cursor": cursor},
                         cache=False)
            ms = page.payload.get("markets", [])
            rows += [market_row(m, "historical") for m in ms]
            closes = [_ts(m.get("close_time")) for m in ms]
            closes = [x for x in closes if x is not None]
            cursor = page.payload.get("cursor")
            if not ms or not cursor or (closes and min(closes) < start):
                break
        log(f"{series}: historical {len(rows) - n0}")
    df = pl.DataFrame(rows, schema=MARKET_COLS)
    return (df.filter((pl.col("close_time") >= start) & (pl.col("close_time") < end))
            .unique("ticker", keep="first").sort("close_time"))


# ---- spot (Coinbase Advanced Trade public candles) ----------------------------------------------
def coinbase_client():
    from albert.http import CachedClient
    return CachedClient("coinbase", COINBASE_BASE, CACHE, min_interval=0.35, max_retries=8, user_agent=UA)


def coinbase_rows(payload: dict[str, Any], product: str) -> list[dict[str, Any]]:
    """``{"candles": [{"start", "low", "high", "open", "close", "volume"}]}`` -> rows; ``ts`` is the
    minute's END (start + 60 s), the first moment its close is known."""
    return [{"product": product, "ts": datetime.fromtimestamp(int(c["start"]) + 60, tz=UTC),
             "open": float(c["open"]), "high": float(c["high"]), "low": float(c["low"]),
             "close": float(c["close"]), "volume": float(c["volume"])}
            for c in payload.get("candles", [])]


SPOT_SCHEMA = {"product": pl.String, "ts": pl.Datetime("us", "UTC"), "open": pl.Float64, "high": pl.Float64,
               "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64}


def fetch_spot(product: str, start: datetime = START - timedelta(hours=3), end: datetime = HOLD_END,
               *, log=print) -> pl.DataFrame:
    """1-minute candles in 300-minute requests. Complete past windows are immutable, so cached."""
    c = coinbase_client()
    rows: list[dict[str, Any]] = []
    t = start
    step = timedelta(minutes=300)
    i = 0
    now = datetime.now(UTC)
    while t < end:
        t1 = min(t + step, end)
        page = c.get(f"/products/{product}/candles", {"start": int(t.timestamp()), "end": int(t1.timestamp()) - 1,
                                                      "granularity": "ONE_MINUTE", "limit": 350},
                     cache=t1 < now - timedelta(minutes=5))
        rows += coinbase_rows(page.payload, product)
        t = t1
        i += 1
        if i % 100 == 0:
            log(f"{product}: {i} pages, through {t:%Y-%m-%d %H:%M}")
    return pl.DataFrame(rows, schema=SPOT_SCHEMA).unique(["product", "ts"]).sort("ts")


# ---- sample --------------------------------------------------------------------------------------
def in_sample(ticker: str, share: float, salt: str = "crypto_fast-2026-11-23") -> bool:
    """Volume-blind: keep a ticker when a hash of it (and a fixed salt) falls below ``share``."""
    h = int(hashlib.sha256(f"{salt}|{ticker}".encode()).hexdigest()[:12], 16)
    return h / 16 ** 12 < share


def split_of(close: datetime) -> str:
    return "dev" if close < DEV_END else ("holdout" if close < HOLD_END else "forward")


# ---- Kalshi 1-minute candles ---------------------------------------------------------------------
K1M_SCHEMA = {"ticker": pl.String, "ts": pl.Datetime("us", "UTC"), "bid": pl.Float64, "ask": pl.Float64,
              "bid_lo": pl.Float64, "bid_hi": pl.Float64, "ask_lo": pl.Float64, "ask_hi": pl.Float64,
              "p_close": pl.Float64, "p_mean": pl.Float64, "volume": pl.Float64}


def _node(n: dict[str, Any] | None, k: str) -> float | None:
    if not n:
        return None
    return _f(n.get(f"{k}_dollars", n.get(k)))


def k1m_rows(candles: list[dict[str, Any]], ticker: str) -> list[dict[str, Any]]:
    """Kalshi candlesticks (historical or live spelling) -> rows; ``ts`` = ``end_period_ts`` (the
    candle covers the minute ending there)."""
    out = []
    for c in candles:
        b, a, p = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
        out.append({"ticker": ticker, "ts": _ts(c.get("end_period_ts")),
                    "bid": _node(b, "close"), "ask": _node(a, "close"),
                    "bid_lo": _node(b, "low"), "bid_hi": _node(b, "high"),
                    "ask_lo": _node(a, "low"), "ask_hi": _node(a, "high"),
                    "p_close": _node(p, "close"), "p_mean": _node(p, "mean"),
                    "volume": _f(c.get("volume_fp", c.get("volume")))})
    return out


def fetch_k1m(sample: pl.DataFrame, cutoff: datetime, *, cached_hist_only: bool = False, log=print) -> pl.DataFrame:
    """1-minute candles for every sampled market from open - 1 min to close (cached per request).
    Live (post-cutoff) windows first, batched across series in 10-hour buckets under the endpoint's
    cap of 10,000 candles = (span in minutes) x (markets); then historical windows one request each
    (``cached_hist_only``: only those already in the raw cache)."""
    from albert.http import RAW_SUFFIX, CachedClient
    c = kalshi_client()
    rows: list[dict[str, Any]] = []
    live = sample.filter(pl.col("close_time") >= cutoff)
    lv = live.with_columns(pl.col("open_time").dt.truncate("10h").alias("bucket")).sort("open_time")
    groups = list(lv.partition_by("bucket", maintain_order=True)) if lv.height else []
    for j, part in enumerate(groups):
        chunks = [part]
        while any((ch["close_time"].max() - ch["open_time"].min()).total_seconds() / 60 * ch.height > 9000
                  for ch in chunks):
            chunks = [x for ch in chunks for x in (ch.slice(0, ch.height // 2), ch.slice(ch.height // 2))]
        for ch in chunks:
            t0 = ch["open_time"].min() - timedelta(minutes=1)
            t1 = ch["close_time"].max()
            page = c.get("/markets/candlesticks", {"market_tickers": ",".join(sorted(ch["ticker"].to_list())),
                                                   "start_ts": int(t0.timestamp()), "end_ts": int(t1.timestamp()),
                                                   "period_interval": 1})
            for m in page.payload.get("markets", []):
                rows += k1m_rows(m.get("candlesticks", []), m["market_ticker"])
        if j % 20 == 0:
            log(f"live candle buckets {j}/{len(groups)}")
    hist = sample.filter(pl.col("close_time") < cutoff)
    n_skip = 0
    for i, r in enumerate(hist.iter_rows(named=True)):
        t0, t1 = r["open_time"] - timedelta(minutes=1), r["close_time"]
        path = f"/historical/markets/{r['ticker']}/candlesticks"
        params = {"start_ts": int(t0.timestamp()), "end_ts": int(t1.timestamp()), "period_interval": 1}
        if cached_hist_only:
            key = CachedClient.cache_key(f"{KALSHI_BASE}{path}", params)
            if not (c.raw_dir / f"{key}{RAW_SUFFIX}").exists():
                n_skip += 1
                continue
        page = c.get(path, params)
        rows += k1m_rows(page.payload.get("candlesticks", []), r["ticker"])
        if i % 200 == 0:
            log(f"historical candles {i}/{hist.height} (skipped uncached {n_skip})")
    log(f"historical windows skipped (not cached): {n_skip}")
    return pl.DataFrame(rows, schema=K1M_SCHEMA).unique(["ticker", "ts"]).sort("ticker", "ts")


# ---- minute grid -----------------------------------------------------------------------------------
SHARE = {"KXBTC15M": 0.15, "KXETH15M": 0.15, "KXSOL15M": 0.08, "KXXRP15M": 0.08, "KXDOGE15M": 0.08}
# Windows before the historical cutoff cost one rate-limited request each (50 tokens of a shared
# per-IP bucket), so they are sub-sampled further with the same hash (a subset of SHARE).
HIST_SHARE = {"KXBTC15M": 0.05, "KXETH15M": 0.05, "KXSOL15M": 0.03, "KXXRP15M": 0.03, "KXDOGE15M": 0.03}
# The per-IP bucket was shared by a dozen agents (~0.1 request/s for us at worst), so the live
# (batched) part is sub-sampled too and historical windows are taken only where already cached.
LIVE_SHARE = {"KXBTC15M": 0.08, "KXETH15M": 0.08, "KXSOL15M": 0.05, "KXXRP15M": 0.05, "KXDOGE15M": 0.05}
AVG_S = 60          # settlement averages the RTI over the final 60 s
VOL_WIN = 60        # minutes of spot returns for the volatility estimate (ending at the decision)


def load_markets(series: str) -> pl.DataFrame:
    """Settled, cleanly specified 15-minute windows of one series (no selection on price/volume)."""
    m = pl.read_parquet(OUT / f"markets_{series}.parquet")
    return m.filter(pl.col("result").is_in(["yes", "no"]) & pl.col("floor_strike").is_not_null()
                    & pl.col("open_time").is_not_null()
                    & ((pl.col("close_time") - pl.col("open_time")) == pl.duration(minutes=15)))


def spot_features(spot: pl.DataFrame) -> pl.DataFrame:
    """Per minute end ``ts``: close, ohlc4 (proxy for a 60-s average), and the 1-minute log-return
    volatility over the ``VOL_WIN`` minutes ending at ``ts`` (data known at ``ts``)."""
    full = pl.DataFrame({"ts": pl.datetime_range(spot["ts"].min(), spot["ts"].max(), "1m", time_zone="UTC",
                                                 eager=True)})
    s = full.join(spot.select("ts", "open", "high", "low", "close"), on="ts", how="left").sort("ts")
    s = s.with_columns(pl.col("close").forward_fill().alias("close_ff"))   # no trade in a minute: last price
    s = s.with_columns(
        pl.when(pl.col("open").is_null()).then(pl.col("close_ff"))
        .otherwise((pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4).alias("ohlc4"),
        pl.col("close_ff").log().diff().alias("r1"))
    return s.with_columns(pl.col("r1").rolling_std(VOL_WIN, min_samples=40).alias("sd1m")).select(
        "ts", pl.col("close_ff").alias("spot"), "ohlc4", "r1", "sd1m")


def fair_prob(x: np.ndarray, sd1m: np.ndarray, tau_s: np.ndarray) -> np.ndarray:
    """P(average of the last 60 s >= strike) for a driftless log-price, ``x`` = log(spot / strike) now,
    ``tau_s`` seconds to the close (>= 60): Var = s^2 (tau - 60) + s^2 60 / 3 = s^2 (tau - 40)."""
    from scipy.special import ndtr
    s2 = (sd1m ** 2) / 60.0
    v = s2 * np.maximum(tau_s - 40.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = x / np.sqrt(v)
    return ndtr(z)


def build_grid(mk: pl.DataFrame, k1m: pl.DataFrame, sf: pl.DataFrame) -> pl.DataFrame:
    """One row per (window, minute m = 1..15): Kalshi quote at the minute end (last candle at or before
    it, after the open), spot and fair value at the same instant, the next minute's quote, outcome."""
    g = (mk.select("ticker", "event_ticker", "series", "open_time", "close_time", "floor_strike",
                   "expiration_value", "result", "volume")
         .with_columns(pl.int_ranges(1, 16).alias("m")).explode("m")
         .with_columns((pl.col("open_time") + pl.duration(minutes=pl.col("m"))).alias("t")))
    g = g.with_columns(((pl.col("close_time") - pl.col("t")).dt.total_seconds()).alias("tau_s"))
    q = k1m.select("ticker", "ts", "bid", "ask", "ask_hi", "bid_lo", "volume").rename({"volume": "kvol"}).sort("ts")
    g = g.sort("t").join_asof(q, left_on="t", right_on="ts", by="ticker", strategy="backward",
                              tolerance=timedelta(minutes=3))
    g = g.with_columns(pl.when(pl.col("ts") > pl.col("open_time")).then(pl.col("bid")).alias("bid"),
                       pl.when(pl.col("ts") > pl.col("open_time")).then(pl.col("ask")).alias("ask")).drop("ts")
    # spot at t, at the open (proxy strike) and at the close (proxy settlement)
    sp = sf.select("ts", "spot", "ohlc4", "sd1m")
    g = g.join(sp.rename({"ts": "t"}), on="t", how="left")
    g = g.join(sp.select(pl.col("ts").alias("open_time"), pl.col("ohlc4").alias("k_proxy")), on="open_time", how="left")
    g = g.join(sp.select(pl.col("ts").alias("close_time"), pl.col("ohlc4").alias("e_proxy")), on="close_time", how="left")
    g = g.sort("ticker", "m").with_columns(
        pl.col("bid").shift(-1).over("ticker").alias("bid_next"), pl.col("ask").shift(-1).over("ticker").alias("ask_next"),
        pl.col("ask_hi").shift(-1).over("ticker").alias("ask_hi_next"),
        pl.col("bid_lo").shift(-1).over("ticker").alias("bid_lo_next"))
    two = (pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("ask") > pl.col("bid"))
    g = g.with_columns(pl.when(two).then((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                       (pl.col("result") == "yes").cast(pl.Int8).alias("y"),
                       (pl.col("spot") / pl.col("k_proxy")).log().alias("x"),
                       pl.col("close_time").dt.date().alias("date"))
    x = g["x"].to_numpy()
    p = fair_prob(np.nan_to_num(x, nan=0.0), g["sd1m"].fill_null(np.nan).to_numpy(), g["tau_s"].to_numpy().astype(float))
    p = np.where(np.isfinite(x) & (g["tau_s"].to_numpy() >= AVG_S), p, np.nan)
    g = g.with_columns(pl.Series("fair", p).fill_nan(None))
    return g.with_columns(pl.col("close_time").map_elements(split_of, return_dtype=pl.String).alias("split"))


# ---- statistics and rules ------------------------------------------------------------------------
def cluster_t(x: np.ndarray, *groups: np.ndarray) -> float | None:
    """Mean / clustered SE (one- or two-way, ``albert.research.counts.cluster_se``)."""
    from albert.research.counts import cluster_se
    if len(x) < 3:
        return None
    se = cluster_se(x.astype(float), *groups)
    return float(x.mean() / se) if se and math.isfinite(se) and se > 0 else None


def kfee(n: int, price: float) -> float:
    """Kalshi taker fee for one order (quadratic, multiplier 1 for every series here)."""
    from albert.fees import kalshi_fee
    return kalshi_fee(n, price, fee_type="quadratic", multiplier=1.0, conservative=True)


STAKE = 5.0        # dollars per order
MAX_N = 25         # contracts per order


def size_for(price: float) -> int:
    return max(1, min(MAX_N, int(STAKE // max(price, 0.01))))


def take_trades(g: pl.DataFrame, theta: float, minutes: tuple[int, int], *, fill: str = "now",
                one_per_window: bool = True) -> pl.DataFrame:
    """Taker rule on the minute grid: at minute ends m in [m0, m1], buy YES at the ask when
    fair - ask >= theta, NO at 1 - bid when bid - fair >= theta; the first signal per window.
    ``fill``: "now" = quote at the signal minute end (needs the quote to still be there a moment
    later), "next" = quote one minute later (a 60-s latency), "worst" = the worse of the next
    minute's high ask / low bid (pessimistic bound)."""
    m0, m1 = minutes
    r = g.filter(pl.col("m").is_between(m0, m1) & pl.col("fair").is_not_null() & pl.col("bid").is_not_null()
                 & pl.col("ask").is_not_null() & (pl.col("bid") > 0) & (pl.col("ask") < 1))
    r = r.with_columns(pl.when(pl.col("fair") - pl.col("ask") >= theta - 1e-9).then(pl.lit("yes"))
                       .when(pl.col("bid") - pl.col("fair") >= theta - 1e-9).then(pl.lit("no"))
                       .otherwise(None).alias("side")).filter(pl.col("side").is_not_null())
    if one_per_window:
        r = r.sort("ticker", "m").unique("ticker", keep="first")
    if fill == "now":
        ya, nb = pl.col("ask"), pl.col("bid")
    elif fill == "next":
        ya, nb = pl.col("ask_next"), pl.col("bid_next")
    else:
        ya, nb = pl.col("ask_hi_next"), pl.col("bid_lo_next")
    r = r.with_columns(pl.when(pl.col("side") == "yes").then(ya).otherwise(1 - nb).alias("price"),
                       pl.when(pl.col("side") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y")).alias("payoff"))
    r = r.filter(pl.col("price").is_not_null() & (pl.col("price") > 0) & (pl.col("price") < 1))
    if not r.height:
        return r.with_columns(pl.lit(None, pl.Int64).alias("n"), pl.lit(None, pl.Float64).alias("fee"),
                              pl.lit(None, pl.Float64).alias("pnl_c"))
    n = [size_for(p) for p in r["price"].to_list()]
    fee = [kfee(k, p) for k, p in zip(n, r["price"].to_list(), strict=True)]
    r = r.with_columns(pl.Series("n", n), pl.Series("fee", fee))
    return r.with_columns((100 * (pl.col("payoff") - pl.col("price") - pl.col("fee") / pl.col("n"))).alias("pnl_c"))


def summarize(tr: pl.DataFrame) -> dict[str, Any]:
    if not tr.height:
        return {"n": 0, "windows": 0, "days": 0, "c_per_contract": None, "t_2way": None, "pnl_per_usd": None}
    x = tr["pnl_c"].to_numpy()
    ev = tr["ticker"].to_numpy()
    d = tr["date"].cast(pl.String).to_numpy()
    usd = (tr["n"] * tr["price"] + tr["fee"]).sum()
    pnl = (tr["n"] * tr["pnl_c"] / 100).sum()
    t2 = cluster_t(x, ev, d)
    return {"n": tr.height, "windows": tr["ticker"].n_unique(), "days": tr["date"].n_unique(),
            "c_per_contract": round(float(x.mean()), 3), "t_2way": None if t2 is None else round(t2, 2),
            "t_date": None if (td := cluster_t(x, d)) is None else round(td, 2),
            "pnl_per_usd": round(float(pnl / usd), 4) if usd else None,
            "win_pct": round(100 * float((x > 0).mean()), 1)}


def ledger_rows(tr: pl.DataFrame, strategy: str, variant: str) -> pl.DataFrame:
    """``vol_metrics`` ledger: one leg per trade, held to the 15-minute settlement."""
    return tr.select(
        pl.lit(strategy).alias("strategy"), pl.lit(variant).alias("variant"), "split",
        pl.col("ticker").alias("struct_id"), "event_ticker", "ticker", "side", pl.col("n").cast(pl.Int64),
        pl.col("t").alias("t_in"), (pl.col("n") * pl.col("price") + pl.col("fee")).alias("cost_usd"),
        pl.col("close_time").alias("settle_ts"), (pl.col("n") * pl.col("payoff")).cast(pl.Float64).alias("payoff_usd"))


def holm(pvals: list[float]) -> list[float]:
    """Holm-adjusted p-values (same order as the input)."""
    k = len(pvals)
    order = sorted(range(k), key=lambda i: pvals[i])
    adj = [0.0] * k
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (k - rank) * pvals[i]))
        adj[i] = run
    return adj


def p_one_sided(t: float | None) -> float:
    from scipy.stats import norm
    return 1.0 if t is None else float(norm.sf(t))


# ---- study ---------------------------------------------------------------------------------------
THETAS = (0.03, 0.05, 0.08, 0.12)
MINUTE_BANDS = ((1, 4), (5, 9), (10, 14))
MIN_DEV_TRADES = 100


def load_grid(series: list[str] | None = None) -> pl.DataFrame:
    frames = [pl.read_parquet(p) for p in sorted(OUT.glob("grid_*.parquet"))
              if series is None or p.stem.removeprefix("grid_") in series]
    return pl.concat(frames, how="diagonal_relaxed")


def mechanics(series: str) -> dict[str, Any]:
    """Settlement mechanics checks on the full market list (no outcome selection)."""
    m = load_markets(series).sort("close_time")
    prev = m.with_columns(pl.col("expiration_value").shift(1).alias("prev_e"),
                          pl.col("close_time").shift(1).alias("prev_close"))
    adj = prev.filter(pl.col("prev_close") == pl.col("open_time"))
    same = (adj["floor_strike"] - adj["prev_e"]).abs() <= 1e-6 * adj["floor_strike"].abs()
    lag = (m["settlement_ts"] - m["close_time"]).dt.total_seconds()
    ties = m.filter((pl.col("expiration_value") - pl.col("floor_strike")).abs() < 1e-9)
    return {"series": series, "markets": m.height, "first_close": str(m["close_time"].min()),
            "strike_eq_prev_settle_pct": round(100 * float(same.mean()), 2) if adj.height else None,
            "adjacent_pairs": adj.height, "yes_pct": round(100 * float((m["result"] == "yes").mean()), 2),
            "ties": ties.height, "ties_yes": int((ties["result"] == "yes").sum()),
            "settle_lag_s_median": float(np.median(lag.drop_nulls().to_numpy())) if lag.drop_nulls().len() else None,
            "settle_lag_s_p95": float(np.quantile(lag.drop_nulls().to_numpy(), 0.95)) if lag.drop_nulls().len() else None}


def proxy_quality(g: pl.DataFrame) -> dict[str, Any]:
    """How well Coinbase minute ohlc4 tracks the CF RTI 60-s averages (strike and settlement)."""
    w = g.filter(pl.col("m") == 15).filter(pl.col("k_proxy").is_not_null() & pl.col("e_proxy").is_not_null())
    ek = 1e4 * (w["e_proxy"] / w["expiration_value"] - 1)
    kk = 1e4 * (w["k_proxy"] / w["floor_strike"] - 1)
    agree = ((w["e_proxy"] >= w["k_proxy"]) == (w["result"] == "yes"))
    return {"windows": w.height, "settle_err_bp_median": round(float(ek.median()), 2),
            "settle_err_bp_mad": round(float((ek - ek.median()).abs().median()), 2),
            "strike_err_bp_median": round(float(kk.median()), 2),
            "sign_agree_pct": round(100 * float(agree.mean()), 2)}


def calibration(g: pl.DataFrame) -> list[dict[str, Any]]:
    """By minute: mean (y - mid), Brier of mid vs fair, and the logistic of y on logit(mid) and
    logit(fair) (does spot information add to the Kalshi quote at the same minute end?)."""
    from albert.research.fundamental import fit_logistic, logit
    out = []
    for m in range(1, 15):
        r = g.filter((pl.col("m") == m) & pl.col("mid").is_not_null() & pl.col("fair").is_not_null())
        if r.height < 50:
            continue
        y = r["y"].to_numpy().astype(float)
        mid = r["mid"].to_numpy()
        fair = r["fair"].to_numpy()
        d = r["date"].cast(pl.String).to_numpy()
        X = np.column_stack([np.ones(len(y)), logit(mid), logit(fair) - logit(mid)])
        beta, tst = fit_logistic(X, y, d)
        out.append({"m": m, "n": r.height, "y_minus_mid_c": round(100 * float((y - mid).mean()), 2),
                    "t_date": None if (t := cluster_t(y - mid, d)) is None else round(t, 2),
                    "brier_mid": round(float(((mid - y) ** 2).mean()), 4),
                    "brier_fair": round(float(((fair - y) ** 2).mean()), 4),
                    "b_mid": round(float(beta[1]), 3), "b_gap": round(float(beta[2]), 3), "t_gap": round(float(tst[2]), 2),
                    "mean_abs_gap_c": round(100 * float(np.abs(fair - mid).mean()), 2)})
    return out


def leadlag(g: pl.DataFrame) -> dict[str, Any]:
    """Minute-scale lead-lag in probability units: does the gap (fair - mid) at a minute end, or the
    fair-value change of the minute just ended, predict the next minute's change in the Kalshi mid?"""
    r = g.sort("ticker", "m").with_columns(
        (pl.col("mid").shift(-1).over("ticker") - pl.col("mid")).alias("dmid_next"),
        (pl.col("mid") - pl.col("mid").shift(1).over("ticker")).alias("dmid"),
        (pl.col("fair") - pl.col("fair").shift(1).over("ticker")).alias("dfair"),
        (pl.col("fair").shift(-1).over("ticker") - pl.col("fair")).alias("dfair_next"),
        (pl.col("fair") - pl.col("mid")).alias("gap"))
    r = r.filter(pl.col("m").is_between(2, 13)).drop_nulls(["dmid_next", "dmid", "dfair", "dfair_next", "gap"])
    out: dict[str, Any] = {"n": r.height}
    d = r["date"].cast(pl.String).to_numpy()

    def ols(ycol: str, xcols: list[str]) -> dict[str, Any]:
        Y = r[ycol].to_numpy()
        X = np.column_stack([np.ones(r.height)] + [r[c].to_numpy() for c in xcols])
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        e = Y - X @ beta
        _, inv = np.unique(d, return_inverse=True)
        S = np.zeros((inv.max() + 1, X.shape[1]))
        np.add.at(S, inv, X * e[:, None])
        XtXi = np.linalg.inv(X.T @ X)
        V = XtXi @ (S.T @ S) @ XtXi
        se = np.sqrt(np.diag(V))
        return {c: {"b": round(float(beta[i + 1]), 4), "t": round(float(beta[i + 1] / se[i + 1]), 2)}
                for i, c in enumerate(xcols)}
    out["dmid_next ~ gap"] = ols("dmid_next", ["gap"])
    out["dmid_next ~ dfair + dfair_next"] = ols("dmid_next", ["dfair", "dfair_next"])
    out["dmid ~ dfair"] = ols("dmid", ["dfair"])
    out["dfair_next ~ dmid (Kalshi leads spot?)"] = ols("dfair_next", ["dmid"])
    return out


def time_structure(g: pl.DataFrame) -> list[dict[str, Any]]:
    """Kalshi pricing error (y - mid) at m = 5 and 10 by UTC hour block and weekend."""
    r = g.filter(pl.col("m").is_in([5, 10]) & pl.col("mid").is_not_null()).with_columns(
        (pl.col("t").dt.hour() // 6).alias("hblk"), (pl.col("t").dt.weekday() >= 6).alias("weekend"))
    out = []
    for (hb, we), part in r.group_by(["hblk", "weekend"], maintain_order=False):
        x = (part["y"] - part["mid"]).to_numpy().astype(float)
        # signed error toward the favourite: positive = favourites under-priced
        fav = np.where(part["mid"].to_numpy() >= 0.5, x, -x)
        dd = part["date"].cast(pl.String).to_numpy()
        out.append({"utc_hours": f"{6 * hb:02d}-{6 * hb + 6:02d}", "weekend": we, "n": part.height,
                    "y_minus_mid_c": round(100 * float(x.mean()), 2),
                    "fav_err_c": round(100 * float(fav.mean()), 2),
                    "t_fav_date": None if (t := cluster_t(fav, dd)) is None else round(t, 2),
                    "spread_c": round(100 * float((part["ask"] - part["bid"]).mean()), 2)})
    return sorted(out, key=lambda z: (z["weekend"], z["utc_hours"]))


def print_fills(tr: pl.DataFrame, delays: tuple[float, ...] = (1.0, 5.0), horizon: int = 20, *, log=print) -> pl.DataFrame:
    """Execution check from the tape: for each signal (minute end ``t``), the first Kalshi print on
    our side (a taker buying the same outcome) at or after ``t + delay``; its price is what a taker
    actually paid then. ``None`` when nobody took that side within ``horizon`` seconds."""
    from albert.venues.kalshi import Kalshi
    c = kalshi_client()
    cutoff = Kalshi(c).cutoff()["trades_created_ts"]
    out = []
    for i, r in enumerate(tr.iter_rows(named=True)):
        t = r["t"]
        path = "/historical/trades" if cutoff is not None and t < cutoff else "/markets/trades"
        prints: list[tuple[float, float, str]] = []
        cursor = None
        for _ in range(30):
            page = c.get(path, {"ticker": r["ticker"], "limit": 1000, "min_ts": int(t.timestamp()),
                                "max_ts": int(t.timestamp()) + horizon, "cursor": cursor})
            for x in page.payload.get("trades", []):
                ts = _ts(x["created_time"]).timestamp()
                yp = _f(x.get("yes_price_dollars", x.get("yes_price")))
                prints.append((ts, yp, x.get("taker_side")))
            cursor = page.payload.get("cursor")
            if not cursor or not page.payload.get("trades"):
                break
        prints.sort()
        row = {"ticker": r["ticker"], "t": t, "side": r["side"]}
        for d in delays:
            hit = next((p for p in prints if p[0] >= t.timestamp() + d and p[2] == r["side"]), None)
            row[f"px_{int(d)}s"] = None if hit is None else (hit[1] if r["side"] == "yes" else round(1 - hit[1], 4))
        out.append(row)
        if i % 50 == 0:
            log(f"prints {i}/{tr.height}")
    return pl.DataFrame(out, infer_schema_length=None)


# ---- hourly above/below (KXBTCD): the last 20 minutes of near-the-money strikes ----------------------
HOURLY = {"KXBTCD": ("KXBTC15M", "BTC-USD")}   # series -> (15-minute series giving the RTI basis, spot)
HOURLY_SHARE = 0.10
HOURLY_BAND = 0.006      # strikes within +-0.6% of spot at the decision window's start (known then)
HOURLY_LAST = 20         # minutes before the close on the grid


def fetch_hourly_markets(series: str, *, log=print) -> pl.DataFrame:
    """Every settled market of the hourly events (all strikes; selection happens per event later)."""
    return fetch_markets(series, log=log)


def rti_basis(m15: pl.DataFrame, sf: pl.DataFrame) -> pl.DataFrame:
    """log(CF RTI 60-s average / Coinbase ohlc4) at every 15-minute boundary (the 15-minute strikes are
    RTI averages), known from that boundary on."""
    b = m15.select(pl.col("open_time").alias("ts"), pl.col("floor_strike").alias("rti")).join(
        sf.select("ts", "ohlc4"), on="ts", how="inner")
    return b.with_columns((pl.col("rti") / pl.col("ohlc4")).log().alias("basis")).select("ts", "basis").sort("ts")


def hourly_sample(mk: pl.DataFrame, sf: pl.DataFrame) -> pl.DataFrame:
    """Events chosen by a hash of the event ticker; within each, the strikes within HOURLY_BAND of the
    Coinbase spot HOURLY_LAST minutes before the close (data known at the first decision)."""
    mk = mk.filter(pl.col("result").is_in(["yes", "no"]) & pl.col("strike_type").is_in(["greater", "greater_or_equal"])
                   & pl.col("floor_strike").is_not_null())
    ev = mk.select("event_ticker").unique()
    ev = ev.filter(pl.Series([in_sample(e, HOURLY_SHARE) for e in ev["event_ticker"].to_list()], dtype=pl.Boolean))
    m = mk.join(ev, on="event_ticker").with_columns((pl.col("close_time") - pl.duration(minutes=HOURLY_LAST)).alias("t0"))
    m = m.join(sf.select(pl.col("ts").alias("t0"), pl.col("spot").alias("spot0")), on="t0", how="left")
    return m.filter((pl.col("floor_strike") / pl.col("spot0")).log().abs() <= HOURLY_BAND)


def fetch_k1m_window(sample: pl.DataFrame, cutoff: datetime, minutes: int, *, log=print) -> pl.DataFrame:
    """1-minute candles for the last ``minutes`` of each sampled market (historical per market; live
    batched per event)."""
    c = kalshi_client()
    rows: list[dict[str, Any]] = []
    hist = sample.filter(pl.col("close_time") < cutoff)
    for i, r in enumerate(hist.iter_rows(named=True)):
        t0, t1 = r["close_time"] - timedelta(minutes=minutes + 1), r["close_time"]
        page = c.get(f"/historical/markets/{r['ticker']}/candlesticks",
                     {"start_ts": int(t0.timestamp()), "end_ts": int(t1.timestamp()), "period_interval": 1})
        rows += k1m_rows(page.payload.get("candlesticks", []), r["ticker"])
        if i % 200 == 0:
            log(f"hourly historical candles {i}/{hist.height}")
    live = sample.filter(pl.col("close_time") >= cutoff)
    for j, part in enumerate(live.partition_by("event_ticker", maintain_order=True)):
        t1 = part["close_time"].max()
        t0 = t1 - timedelta(minutes=minutes + 1)
        page = c.get("/markets/candlesticks", {"market_tickers": ",".join(sorted(part["ticker"].to_list())),
                                               "start_ts": int(t0.timestamp()), "end_ts": int(t1.timestamp()),
                                               "period_interval": 1})
        for m in page.payload.get("markets", []):
            rows += k1m_rows(m.get("candlesticks", []), m["market_ticker"])
        if j % 50 == 0:
            log(f"hourly live events {j}")
    return pl.DataFrame(rows, schema=K1M_SCHEMA).unique(["ticker", "ts"]).sort("ticker", "ts")


def build_hourly_grid(sm: pl.DataFrame, k1m: pl.DataFrame, sf: pl.DataFrame, basis: pl.DataFrame) -> pl.DataFrame:
    """Rows (market, m = minutes before close in HOURLY_LAST .. 2): quote, spot, RTI-basis-adjusted fair."""
    g = (sm.select("ticker", "event_ticker", "series", "open_time", "close_time", "floor_strike", "result", "volume")
         .with_columns(pl.int_ranges(2, HOURLY_LAST + 1).alias("mb")).explode("mb")
         .with_columns((pl.col("close_time") - pl.duration(minutes=pl.col("mb"))).alias("t")))
    g = g.with_columns(((pl.col("close_time") - pl.col("t")).dt.total_seconds()).alias("tau_s"),
                       (HOURLY_LAST + 1 - pl.col("mb")).alias("m"))   # m increases toward the close
    q = k1m.select("ticker", "ts", "bid", "ask", "ask_hi", "bid_lo").sort("ts")
    g = g.sort("t").join_asof(q, left_on="t", right_on="ts", by="ticker", strategy="backward",
                              tolerance=timedelta(minutes=3)).drop("ts")
    g = g.join(sf.select(pl.col("ts").alias("t"), "spot", "sd1m"), on="t", how="left")
    g = g.sort("t").join_asof(basis.rename({"ts": "b_ts"}), left_on="t", right_on="b_ts", strategy="backward",
                              tolerance=timedelta(minutes=30)).drop("b_ts")
    g = g.sort("ticker", "t").with_columns(
        pl.col("bid").shift(-1).over("ticker").alias("bid_next"), pl.col("ask").shift(-1).over("ticker").alias("ask_next"),
        pl.col("ask_hi").shift(-1).over("ticker").alias("ask_hi_next"),
        pl.col("bid_lo").shift(-1).over("ticker").alias("bid_lo_next"))
    two = (pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("ask") > pl.col("bid"))
    g = g.with_columns(pl.when(two).then((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                       (pl.col("result") == "yes").cast(pl.Int8).alias("y"),
                       ((pl.col("spot") / pl.col("floor_strike")).log() + pl.col("basis").fill_null(0.0)).alias("x"),
                       pl.col("close_time").dt.date().alias("date"))
    x = g["x"].to_numpy()
    p = fair_prob(np.nan_to_num(x, nan=0.0), g["sd1m"].fill_null(np.nan).to_numpy(), g["tau_s"].to_numpy().astype(float))
    p = np.where(np.isfinite(x) & (g["tau_s"].to_numpy() >= AVG_S), p, np.nan)
    g = g.with_columns(pl.Series("fair", p).fill_nan(None))
    return g.with_columns(pl.col("close_time").map_elements(split_of, return_dtype=pl.String).alias("split"))


def take_favourite(g: pl.DataFrame, minutes: tuple[int, ...], *, weekend: bool | None, hours: tuple[int, int] | None,
                   lo: float = 0.55, hi: float = 0.90, fill: str = "now") -> pl.DataFrame:
    """Time-structure rule: at the first listed minute end with a two-sided quote, buy the favourite
    (YES when the mid is in [lo, hi], NO when it is in [1 - hi, 1 - lo]) at the ask / 1 - bid, inside the
    given UTC hour block and weekend flag. Same fee/size model as ``take_trades``."""
    r = g.filter(pl.col("m").is_in(list(minutes)) & pl.col("mid").is_not_null())
    if weekend is not None:
        r = r.filter((pl.col("t").dt.weekday() >= 6) == weekend)
    if hours is not None:
        r = r.filter(pl.col("t").dt.hour().is_between(hours[0], hours[1] - 1))
    r = r.with_columns(pl.when(pl.col("mid").is_between(lo, hi)).then(pl.lit("yes"))
                       .when(pl.col("mid").is_between(1 - hi, 1 - lo)).then(pl.lit("no")).otherwise(None).alias("side"))
    r = r.filter(pl.col("side").is_not_null()).sort("ticker", "m").unique("ticker", keep="first")
    ya, nb = (pl.col("ask"), pl.col("bid")) if fill == "now" else (pl.col("ask_next"), pl.col("bid_next"))
    r = r.with_columns(pl.when(pl.col("side") == "yes").then(ya).otherwise(1 - nb).alias("price"),
                       pl.when(pl.col("side") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y")).alias("payoff"))
    r = r.filter(pl.col("price").is_not_null() & (pl.col("price") > 0) & (pl.col("price") < 1))
    if not r.height:
        return r.with_columns(pl.lit(None, pl.Int64).alias("n"), pl.lit(None, pl.Float64).alias("fee"),
                              pl.lit(None, pl.Float64).alias("pnl_c"))
    n = [size_for(p) for p in r["price"].to_list()]
    fee = [kfee(k, p) for k, p in zip(n, r["price"].to_list(), strict=True)]
    r = r.with_columns(pl.Series("n", n), pl.Series("fee", fee))
    return r.with_columns((100 * (pl.col("payoff") - pl.col("price") - pl.col("fee") / pl.col("n"))).alias("pnl_c"))


# ---- live recorder -------------------------------------------------------------------------------
_ET = "America/New_York"
_MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def window_close(now: datetime) -> datetime:
    """Close of the 15-minute window that is open at ``now`` (a boundary itself belongs to the next)."""
    q = now.replace(second=0, microsecond=0, minute=(now.minute // 15) * 15)
    return q + timedelta(minutes=15)


def ticker_15m(series: str, close: datetime) -> str:
    """``KXBTC15M-26NOV221945-45``: the event is named by the close in US Eastern time."""
    from zoneinfo import ZoneInfo
    et = close.astimezone(ZoneInfo(_ET))
    return f"{series}-{et:%y}{_MON[et.month - 1]}{et:%d%H%M}-{et:%M}"


class _Sink:
    """Append-only JSONL per source and UTC hour, zstd-compressed with a frame flushed every few
    seconds (a crash loses at most the last few seconds; concatenated frames stay valid)."""

    def __init__(self, root: Path, src: str) -> None:
        self.root, self.src = root, src
        self.hour: str | None = None
        self.fh = self.w = None
        self.last_flush = 0.0

    def write(self, rec: dict[str, Any]) -> None:
        import time

        import zstandard
        hour = datetime.now(UTC).strftime("%Y%m%dT%H")
        if hour != self.hour:
            self.close()
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / f"{hour}-{self.src}.jsonl.zst"
            n = 0
            while path.exists():   # never append to (or overwrite) an existing raw file
                n += 1
                path = self.root / f"{hour}-{self.src}-{n}.jsonl.zst"
            self.fh = open(path, "wb")  # noqa: SIM115 - closed in close()
            self.w = zstandard.ZstdCompressor(level=6).stream_writer(self.fh)
            self.hour = hour
        self.w.write((json.dumps(rec, separators=(",", ":")) + "\n").encode())
        now = time.monotonic()
        if now - self.last_flush > 5:
            import zstandard
            self.w.flush(zstandard.FLUSH_FRAME)
            self.fh.flush()
            self.last_flush = now

    def close(self) -> None:
        if self.w is not None:
            self.w.close()
            self.w = self.fh = None


async def _kalshi_poller(series: list[str], sink: _Sink, stop, interval: float = 0.5) -> None:
    """Poll the open window's order book, one series per tick (``interval`` >= 0.5 s => <= 2 req/s)."""
    import asyncio
    import time

    import httpx
    interval = max(0.5, interval)
    backoff = 0.0
    async with httpx.AsyncClient(headers={"User-Agent": UA}, timeout=5.0) as cl:
        i = 0
        nxt = time.monotonic()
        while not stop():
            nxt += interval + backoff
            s = series[i % len(series)]
            i += 1
            now = datetime.now(UTC)
            tk = ticker_15m(s, window_close(now))
            t_send = time.time_ns()
            try:
                r = await cl.get(f"{KALSHI_BASE}/markets/{tk}/orderbook", params={"depth": 10})
                t_recv = time.time_ns()
                rec = {"t_send": t_send, "t_recv": t_recv, "series": s, "ticker": tk, "status": r.status_code,
                       "payload": r.json() if r.status_code == 200 else r.text[:200]}
                backoff = min(30.0, max(2.0, 2 * backoff)) if r.status_code == 429 else 0.0
            except Exception as e:  # noqa: BLE001 - keep recording through transient errors
                rec = {"t_send": t_send, "t_recv": time.time_ns(), "series": s, "ticker": tk, "error": repr(e)[:200]}
            sink.write(rec)
            await asyncio.sleep(max(0.0, nxt - time.monotonic()))


async def _ws_feed(name: str, url: str, subs: list[dict[str, Any]], sink: _Sink, stop) -> None:
    import asyncio
    import time

    import websockets
    while not stop():
        try:
            async with websockets.connect(url, user_agent_header=UA, max_size=2 ** 23, ping_interval=20) as ws:
                for sub in subs:
                    await ws.send(json.dumps(sub))
                sink.write({"t_recv": time.time_ns(), "event": "connected"})
                while not stop():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    except TimeoutError:
                        continue
                    sink.write({"t_recv": time.time_ns(), "msg": msg})
        except Exception as e:  # noqa: BLE001
            sink.write({"t_recv": time.time_ns(), "event": "error", "error": repr(e)[:200]})
            await asyncio.sleep(5)


def record(hours: float, series: list[str], products: list[str], root: Path = CACHE / "live",
           interval: float = 0.5) -> None:
    """Kalshi books by REST plus Coinbase / Kraken / Hyperliquid public trade feeds (data only)."""
    import asyncio
    import time
    end = time.monotonic() + hours * 3600
    stop_file = root / "STOP"

    def stop() -> bool:
        return time.monotonic() > end or stop_file.exists()
    sinks = {k: _Sink(root, k) for k in ("kalshi", "coinbase", "kraken", "hl")}
    kr = [p.replace("-USD", "/USD") for p in products]
    coins = [p.split("-")[0] for p in products]
    feeds = [
        ("coinbase", "wss://advanced-trade-ws.coinbase.com",
         [{"type": "subscribe", "product_ids": products, "channel": "ticker"},
          {"type": "subscribe", "product_ids": products, "channel": "heartbeats"}]),
        ("kraken", "wss://ws.kraken.com/v2",
         [{"method": "subscribe", "params": {"channel": "trade", "symbol": kr}}]),
        ("hl", "wss://api.hyperliquid.xyz/ws",
         [{"method": "subscribe", "subscription": {"type": "trades", "coin": c}} for c in coins]),
    ]

    async def run() -> None:
        tasks = [_kalshi_poller(series, sinks["kalshi"], stop, interval)]
        tasks += [_ws_feed(n, u, s, sinks[n], stop) for n, u, s in feeds]
        await asyncio.gather(*tasks)
    try:
        asyncio.run(run())
    finally:
        for s in sinks.values():
            s.close()


# ---- live recording: parse and analyse ----------------------------------------------------------
def _book_top(payload: dict[str, Any]) -> tuple[float | None, float | None, float | None, float | None]:
    """(yes bid, yes ask, bid size, ask size) from an ``orderbook_fp`` payload (levels ascending)."""
    book = payload.get("orderbook_fp") or payload.get("orderbook") or {}
    yes = [(float(p), float(q)) for p, q in (book.get("yes_dollars") or [])]
    no = [(float(p), float(q)) for p, q in (book.get("no_dollars") or [])]
    by = max(yes, default=None)
    bn = max(no, default=None)
    return (by[0] if by else None, round(1 - bn[0], 4) if bn else None,
            by[1] if by else None, bn[1] if bn else None)


def iso_ns(v: str) -> datetime:
    """RFC 3339 with any number of fractional digits (Coinbase sends up to 9) -> datetime (us)."""
    v = v.rstrip("Z")
    head, _, frac = v.partition(".")
    return datetime.fromisoformat(f"{head}.{(frac + '000000')[:6]}+00:00")


def kalshi_poll_rows(lines: list[str]) -> list[dict[str, Any]]:
    rows = []
    for ln in lines:
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue   # a torn last line after a crash
        if r.get("status") != 200:
            continue
        b, a, bs, as_ = _book_top(r["payload"])
        rows.append({"t_send": r["t_send"], "t_recv": r["t_recv"], "series": r["series"], "ticker": r["ticker"],
                     "bid": b, "ask": a, "bid_sz": bs, "ask_sz": as_})
    return rows


def coinbase_tick_rows(lines: list[str]) -> list[dict[str, Any]]:
    rows = []
    for ln in lines:
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        msg = r.get("msg")
        if not msg or '"ticker"' not in msg[:40]:
            continue
        m = json.loads(msg)
        if m.get("channel") != "ticker":
            continue
        ts = iso_ns(m["timestamp"]) if m.get("timestamp") else None
        for ev in m.get("events", []):
            for t in ev.get("tickers", []):
                rows.append({"t_recv": r["t_recv"], "ts_server": ts, "product": t["product_id"],
                             "price": float(t["price"]), "bid": _f(t.get("best_bid")), "ask": _f(t.get("best_ask"))})
    return rows


def kraken_trade_rows(lines: list[str]) -> list[dict[str, Any]]:
    rows = []
    for ln in lines:
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        msg = r.get("msg")
        if not msg or '"trade"' not in msg[:30]:
            continue
        m = json.loads(msg)
        for t in m.get("data", []):
            rows.append({"t_recv": r["t_recv"], "ts_server": _ts(t["timestamp"]), "symbol": t["symbol"],
                         "price": float(t["price"]), "qty": float(t["qty"]), "side": t["side"]})
    return rows


def hl_trade_rows(lines: list[str]) -> list[dict[str, Any]]:
    rows = []
    for ln in lines:
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        msg = r.get("msg")
        if not msg or '"trades"' not in msg[:30]:
            continue
        m = json.loads(msg)
        for t in m.get("data", []):
            rows.append({"t_recv": r["t_recv"], "ts_server": datetime.fromtimestamp(t["time"] / 1000, tz=UTC),
                         "coin": t["coin"], "price": float(t["px"]), "qty": float(t["sz"]), "side": t["side"]})
    return rows


def read_live(src: str, root: Path = CACHE / "live") -> list[str]:
    import zstandard
    lines: list[str] = []
    for p in sorted(root.glob(f"*-{src}*.jsonl.zst")):
        with open(p, "rb") as fh:
            data = zstandard.ZstdDecompressor().stream_reader(fh, read_across_frames=True).read()
        lines += data.decode(errors="replace").splitlines()
    return lines


LIVE_PRODUCT = {"KXBTC15M": "BTC-USD", "KXETH15M": "ETH-USD", "KXSOL15M": "SOL-USD"}


def live_markets(tickers: list[str], *, log=print) -> pl.DataFrame:
    """Settled metadata (strike, settlement value, result) of the recorded windows."""
    c = kalshi_client()
    rows = []
    for tk in sorted(set(tickers)):
        try:
            page = c.get(f"/markets/{tk}")
        except Exception as e:  # noqa: BLE001 - a window that never listed
            log(f"{tk}: {e!r}"[:200])
            continue
        m = page.payload.get("market") or {}
        if m.get("result") in ("yes", "no"):
            rows.append(market_row(m, "live"))
    return pl.DataFrame(rows, schema=MARKET_COLS)


def live_frame(root: Path = CACHE / "live") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Kalshi polls (one row per 200 response) and Coinbase mids (one row per ticker message)."""
    k = pl.DataFrame(kalshi_poll_rows(read_live("kalshi", root))).sort("t_recv")
    c = pl.DataFrame(coinbase_tick_rows(read_live("coinbase", root))).sort("t_recv")
    c = c.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    return k, c


def live_fair(polls: pl.DataFrame, cb: pl.DataFrame, mk: pl.DataFrame, sd1m: pl.DataFrame, *,
              t_col: str = "t_recv") -> pl.DataFrame:
    """Fair value at each poll's ``t_col`` (ns) from the Coinbase mid known at that time.

    Strike in Coinbase units = mean of 1-s Coinbase mids over the 60 s before the open (the CF RTI
    strike is the same 60-s average of the index). From 60 s before the close the known part of the
    settlement average is used: final = (a * A_known + tau * A_future) / 60 with
    sd(A_future) = S sigma sqrt(tau / 3)."""
    from scipy.special import ndtr
    out = []
    for (series, ticker), part in polls.group_by(["series", "ticker"], maintain_order=True):
        meta = mk.filter(pl.col("ticker") == ticker)
        if not meta.height:
            continue
        prod = LIVE_PRODUCT[series]
        c = cb.filter(pl.col("product") == prod)
        if not c.height:
            continue
        ct = c["t_recv"].to_numpy()
        cm = c["mid"].to_numpy()
        t0 = int(meta["open_time"][0].timestamp() * 1e9)
        t1 = int(meta["close_time"][0].timestamp() * 1e9)
        grid = np.arange(t0 - 60 * 10**9, t1 + 1, 10**9)            # 1-s grid of Coinbase mids
        gi = np.searchsorted(ct, grid, side="right") - 1
        gm = np.where(gi >= 0, cm[np.clip(gi, 0, None)], np.nan)
        pre = gm[:61]
        if np.isnan(pre).mean() > 0.2:
            continue
        k_cb = float(np.nanmean(pre))
        v = sd1m.filter(pl.col("product") == prod).sort("ts")
        vts = v["ts"].dt.epoch("ns").to_numpy()
        vsd = v["sd1m"].to_numpy()
        tt = part[t_col].to_numpy()
        si = np.searchsorted(ct, tt, side="right") - 1
        spot = np.where(si >= 0, cm[np.clip(si, 0, None)], np.nan)
        vi = np.searchsorted(vts, tt, side="right") - 1
        sd = np.where(vi >= 0, vsd[np.clip(vi, 0, None)], np.nan)
        s2 = sd ** 2 / 60.0
        tau = (t1 - tt) / 1e9
        fair = np.full(len(tt), np.nan)
        early = tau >= 60
        fair[early] = ndtr(np.log(spot[early] / k_cb) / np.sqrt(s2[early] * (tau[early] - 40.0)))
        late = (tau > 0) & (tau < 60)
        for i in np.where(late)[0]:
            a_s = 60.0 - tau[i]
            lo = int((t1 - 60 * 10**9 - grid[0]) // 10**9)
            hi = int((tt[i] - grid[0]) // 10**9)
            known = gm[lo:hi + 1]
            a_known = float(np.nanmean(known)) if np.isfinite(known).any() else spot[i]
            mean_final = (a_s * a_known + tau[i] * spot[i]) / 60.0
            sd_final = spot[i] * math.sqrt(s2[i]) * tau[i] * math.sqrt(tau[i] / 3.0) / 60.0
            fair[i] = ndtr((mean_final - k_cb) / max(sd_final, 1e-12))
        y = 1 if meta["result"][0] == "yes" else 0
        out.append(part.with_columns(pl.Series("spot", spot), pl.Series("fair", fair), pl.Series("tau", tau),
                                     pl.lit(k_cb).alias("k_cb"), pl.lit(y).alias("y"),
                                     pl.lit(meta["floor_strike"][0]).alias("strike"),
                                     pl.lit(meta["close_time"][0]).alias("close_time")))
    return pl.concat(out) if out else pl.DataFrame()


LIVE_THETAS = (0.05, 0.10)


def live_takers(f: pl.DataFrame, theta: float, tau_band: tuple[float, float]) -> pl.DataFrame:
    """First signal per window at a poll (decision at our receive time, spot known then): buy YES at
    the polled ask when fair - ask >= theta, NO at 1 - bid when bid - fair >= theta. ``persist``:
    the next poll of the same market (about 1 s later) still offers that price or better."""
    r = f.sort("ticker", "t_recv").with_columns(
        pl.col("ask").shift(-1).over("ticker").alias("ask_n1"), pl.col("bid").shift(-1).over("ticker").alias("bid_n1"))
    r = r.filter(pl.col("tau").is_between(*tau_band, closed="right") & pl.col("fair").is_not_null()
                 & pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("bid") > 0) & (pl.col("ask") < 1))
    r = r.with_columns(pl.when(pl.col("fair") - pl.col("ask") >= theta).then(pl.lit("yes"))
                       .when(pl.col("bid") - pl.col("fair") >= theta).then(pl.lit("no")).otherwise(None).alias("side"))
    r = r.filter(pl.col("side").is_not_null()).unique("ticker", keep="first", maintain_order=True)
    if not r.height:
        return r
    r = r.with_columns(
        pl.when(pl.col("side") == "yes").then(pl.col("ask")).otherwise(1 - pl.col("bid")).alias("price"),
        pl.when(pl.col("side") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y")).alias("payoff"),
        pl.when(pl.col("side") == "yes").then(pl.col("ask_n1") <= pl.col("ask") + 1e-9)
        .otherwise(pl.col("bid_n1") >= pl.col("bid") - 1e-9).fill_null(False).alias("persist"))
    n = [size_for(p) for p in r["price"].to_list()]
    fee = [kfee(k, p) for k, p in zip(n, r["price"].to_list(), strict=True)]
    return r.with_columns(pl.Series("n", n), pl.Series("fee", fee)).with_columns(
        (100 * (pl.col("payoff") - pl.col("price") - pl.col("fee") / pl.col("n"))).alias("pnl_c"))


def live_study(f: pl.DataFrame) -> dict[str, Any]:
    from albert.research.fundamental import fit_logistic, logit
    res: dict[str, Any] = {}
    two = (pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("ask") > pl.col("bid"))
    f = f.with_columns(pl.when(two).then((pl.col("bid") + pl.col("ask")) / 2).alias("mid")).sort("ticker", "t_recv")
    f = f.with_columns((pl.col("mid").shift(-1).over("ticker") - pl.col("mid")).alias("dmid_next"),
                       (pl.col("mid") - pl.col("mid").shift(1).over("ticker")).alias("dmid"),
                       (pl.col("fair") - pl.col("fair").shift(1).over("ticker")).alias("dfair"),
                       (pl.col("fair") - pl.col("mid")).alias("gap"))
    for sr in sorted(f["series"].unique().to_list()):
        part = f.filter((pl.col("series") == sr) & pl.col("tau").is_between(60, 840))
        r = part.drop_nulls(["dmid_next", "dmid", "dfair", "gap"])
        if r.height < 100:
            continue
        out: dict[str, Any] = {"polls": part.height, "windows": part["ticker"].n_unique()}
        # lead-lag on ~1-s poll steps: corr(dmid at step +k, dfair at step 0)
        xc = {}
        for k in (-2, -1, 0, 1, 2, 3, 5):
            q = part.with_columns(pl.col("dmid").shift(-k).over("ticker").alias("dm_k")).drop_nulls(["dm_k", "dfair"])
            xc[str(k)] = round(float(np.corrcoef(q["dfair"].to_numpy(), q["dm_k"].to_numpy())[0, 1]), 3)
        out["corr_dmid_k_dfair"] = xc
        # jump events: |dfair| >= 5c between consecutive polls (~1 s); response of the mid over the next polls
        ev = []
        arr = part.select("ticker", "t_recv", "mid", "fair", "dfair").to_dicts()
        by: dict[str, list[dict[str, Any]]] = {}
        for row in arr:
            by.setdefault(row["ticker"], []).append(row)
        for rows in by.values():
            for i in range(1, len(rows)):
                d = rows[i]["dfair"]
                if d is None or abs(d) < 0.05 or rows[i - 1]["mid"] is None:
                    continue
                base = rows[i - 1]["mid"]
                resp = {}
                for k in (0, 1, 2, 3, 5, 10):
                    j = i + k
                    if j < len(rows) and rows[j]["mid"] is not None:
                        resp[k] = (rows[j]["mid"] - base) / d
                ev.append(resp)
        out["jump_events"] = len(ev)
        out["jump_response_share"] = {str(k): round(float(np.median([e[k] for e in ev if k in e])), 3)
                                      for k in (0, 1, 2, 3, 5, 10) if any(k in e for e in ev)}
        res[sr] = out
    # calibration: does fair add to the mid? (logistic, clustered by window)
    cal = {}
    for name, band in (("tau_60_300", (60, 300)), ("tau_300_840", (300, 840)), ("final_minute", (0, 60))):
        r = f.filter(pl.col("tau").is_between(*band, closed="right") & pl.col("mid").is_not_null() & pl.col("fair").is_not_null())
        if r.height < 100 or r["ticker"].n_unique() < 8:
            continue
        y = r["y"].to_numpy().astype(float)
        X = np.column_stack([np.ones(r.height), logit(r["mid"].to_numpy()), logit(r["fair"].to_numpy()) - logit(r["mid"].to_numpy())])
        beta, tst = fit_logistic(X, y, r["ticker"].to_numpy())
        cal[name] = {"polls": r.height, "windows": r["ticker"].n_unique(), "b_mid": round(float(beta[1]), 3),
                     "b_gap": round(float(beta[2]), 3), "t_gap": round(float(tst[2]), 2),
                     "brier_mid": round(float(((r["mid"] - r["y"]) ** 2).mean()), 4),
                     "brier_fair": round(float(((r["fair"] - r["y"]) ** 2).mean()), 4)}
    res["calibration"] = cal
    # gap persistence: of polls where |fair - mid| >= 5c, how much of the gap is left one poll (~1 s) later
    pers = {}
    for sr in sorted(f["series"].unique().to_list()):
        q = f.filter((pl.col("series") == sr) & pl.col("tau").is_between(60, 840)).sort("ticker", "t_recv").with_columns(
            pl.col("gap").shift(-1).over("ticker").alias("gap_n1"), pl.col("gap").shift(-3).over("ticker").alias("gap_n3"))
        big = q.filter(pl.col("gap").abs() >= 0.05).drop_nulls(["gap_n1"])
        if big.height:
            pers[sr] = {"polls_gap_ge_5c": big.height, "share_of_polls": round(big.height / max(q.height, 1), 3),
                        "median_gap_left_1poll": round(float((big["gap_n1"] / big["gap"]).median()), 3),
                        "median_gap_left_3polls": round(float((big["gap_n3"] / big["gap"]).drop_nulls().median()), 3)}
    res["gap_persistence"] = pers
    # final minute: does the book follow the running-average fair or a spot-only fair?
    fm = f.filter(pl.col("tau").is_between(1, 59) & pl.col("mid").is_not_null() & pl.col("fair").is_not_null()
                  & pl.col("spot").is_not_null())
    if fm.height > 50:
        from scipy.stats import spearmanr
        tau = fm["tau"].to_numpy()
        # rank correlations, so the (unknown) volatility scale of a spot-only model does not matter
        naive_x = np.log(fm["spot"].to_numpy() / fm["k_cb"].to_numpy()) / np.sqrt(np.maximum(tau, 1.0))
        res["final_minute"] = {"polls": fm.height, "windows": fm["ticker"].n_unique(),
                               "spearman_mid_vs_avg_fair": round(float(spearmanr(fm["mid"], fm["fair"])[0]), 3),
                               "spearman_mid_vs_spot_only": round(float(spearmanr(fm["mid"], naive_x)[0]), 3),
                               "mean_abs_gap_c": round(100 * float((fm["fair"] - fm["mid"]).abs().mean()), 2)}
    tk = []
    for th in LIVE_THETAS:
        for band in ((60, 840), (0, 60)):
            tr = live_takers(f, th, band)
            for fill in ("optimistic", "persist"):
                x = tr if fill == "optimistic" else tr.filter(pl.col("persist")) if tr.height else tr
                if not x.height:
                    tk.append({"theta": th, "tau": list(band), "fill": fill, "n": 0})
                    continue
                v = x["pnl_c"].to_numpy()
                tk.append({"theta": th, "tau": list(band), "fill": fill, "n": x.height,
                           "c_per_contract": round(float(v.mean()), 2),
                           "t_window": None if (t := cluster_t(v, x["ticker"].to_numpy())) is None else round(t, 2),
                           "persist_share": round(float(tr["persist"].mean()), 3)})
    res["takers"] = tk
    return res


def spot_leadlag(root: Path = CACHE / "live", step_ms: int = 500) -> dict[str, Any]:
    """Cross-venue lead-lag of spot/perp prices on a ``step_ms`` grid (receive times): corr(r_coinbase(t),
    r_other(t + k)) for k steps; a peak at k > 0 means Coinbase leads."""
    cb = pl.DataFrame(coinbase_tick_rows(read_live("coinbase", root)))
    kr = pl.DataFrame(kraken_trade_rows(read_live("kraken", root)))
    hl = pl.DataFrame(hl_trade_rows(read_live("hl", root)))
    out: dict[str, Any] = {}
    for coin in ("BTC", "SOL"):
        c = cb.filter(pl.col("product") == f"{coin}-USD").with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("px"))
        others = {"kraken": kr.filter(pl.col("symbol") == f"{coin}/USD").rename({"price": "px"}),
                  "hyperliquid_perp": hl.filter(pl.col("coin") == coin).rename({"price": "px"})}
        if c.height < 100:
            continue
        t0, t1 = int(c["t_recv"].min()), int(c["t_recv"].max())
        grid = np.arange(t0, t1, step_ms * 10**6)

        def on_grid(df: pl.DataFrame) -> np.ndarray:
            d = df.sort("t_recv")
            i = np.searchsorted(d["t_recv"].to_numpy(), grid, side="right") - 1
            v = d["px"].to_numpy()
            return np.where(i >= 0, v[np.clip(i, 0, None)], np.nan)
        rc = np.diff(np.log(on_grid(c)))
        for name, o in others.items():
            if o.height < 100:
                continue
            ro = np.diff(np.log(on_grid(o)))
            cc = {}
            for k in range(-6, 7):
                a, b = (rc[: len(rc) - k], ro[k:]) if k >= 0 else (rc[-k:], ro[: len(ro) + k])
                ok = np.isfinite(a) & np.isfinite(b)
                cc[str(k * step_ms)] = round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3) if ok.sum() > 100 else None
            out[f"{coin}_coinbase_vs_{name}"] = cc
    return out


# ---- CLI -----------------------------------------------------------------------------------------
def cmd_markets(a: argparse.Namespace) -> None:
    for s in a.series:
        df = fetch_markets(s)
        _save(df, OUT / f"markets_{s}.parquet")
        print(s, df.height, df["close_time"].min(), df["close_time"].max(),
              df.group_by("result").len().sort("result").rows())


def sampled_markets() -> pl.DataFrame:
    frames = []
    for sr, share in SHARE.items():
        p = OUT / f"markets_{sr}.parquet"
        if not p.exists():
            continue
        m = load_markets(sr)
        frames.append(m.filter(pl.Series([in_sample(t, share) for t in m["ticker"].to_list()], dtype=pl.Boolean)))
    return pl.concat(frames).sort("close_time")


def cmd_candles(a: argparse.Namespace) -> None:
    from albert.venues.kalshi import Kalshi
    sm = sampled_markets()
    if a.series:
        sm = sm.filter(pl.col("series").is_in(a.series))
    cutoff = Kalshi(kalshi_client()).cutoff()["market_settled_ts"]
    keep = [in_sample(t, LIVE_SHARE[s]) if c >= cutoff else in_sample(t, HIST_SHARE[s]) for t, s, c in
            zip(sm["ticker"].to_list(), sm["series"].to_list(), sm["close_time"].to_list(), strict=True)]
    sm = sm.filter(pl.Series(keep, dtype=pl.Boolean))
    _save(sm, OUT / "sample.parquet")
    print("sample", sm.group_by("series").len().sort("series").rows(), "cutoff", cutoff, flush=True)
    # all series in one pass: historical windows one request each, live windows batched across series
    df = fetch_k1m(sm, cutoff, cached_hist_only=a.cached_hist_only, log=lambda x: print(x, flush=True))
    have = set(df["ticker"].unique().to_list())
    sm = sm.filter(pl.col("ticker").is_in(have))
    _save(sm, OUT / "sample.parquet")   # the windows that actually have candles
    for sr in sorted(sm["series"].unique().to_list()):
        part = df.filter(pl.col("ticker").str.starts_with(sr + "-"))
        _save(part, CACHE / "k1m" / f"{sr}.parquet")
        print(sr, "candles", part.height, "markets", part["ticker"].n_unique(), flush=True)


def cmd_build(a: argparse.Namespace) -> None:
    sm = pl.read_parquet(OUT / "sample.parquet")   # written by ``candles``
    for sr, prod in SERIES_15M.items():
        kp = CACHE / "k1m" / f"{sr}.parquet"
        sp = CACHE / "spot" / f"{prod}.parquet"
        if not (kp.exists() and sp.exists()):
            print(sr, "missing inputs")
            continue
        mk = sm.filter(pl.col("series") == sr)
        g = build_grid(mk, pl.read_parquet(kp), spot_features(pl.read_parquet(sp)))
        _save(g, OUT / f"grid_{sr}.parquet")
        print(sr, g.height, "rows", g["ticker"].n_unique(), "windows; quoted share",
              round(float(g["mid"].is_not_null().mean()), 3), "fair share", round(float(g["fair"].is_not_null().mean()), 3))


def cmd_study(a: argparse.Namespace) -> None:
    """Descriptives on dev, the rule grid on dev (cells counted), the frozen rule; holdout only with
    ``--holdout`` after ``frozen.json`` exists (scored once)."""
    g = load_grid()
    dev = g.filter(pl.col("split") == "dev")
    res: dict[str, Any] = {"mechanics": [mechanics(sr) for sr in SERIES_15M if (OUT / f"markets_{sr}.parquet").exists()]}
    res["proxy"] = {sr: proxy_quality(dev.filter(pl.col("series") == sr)) for sr in dev["series"].unique().sort()}
    res["calibration_dev"] = {sr: calibration(dev.filter(pl.col("series") == sr)) for sr in dev["series"].unique().sort()}
    res["leadlag_dev"] = {sr: leadlag(dev.filter(pl.col("series") == sr)) for sr in dev["series"].unique().sort()}
    res["time_dev"] = time_structure(dev)
    cells = []
    for sr in [*sorted(dev["series"].unique().to_list()), "ALL"]:
        part = dev if sr == "ALL" else dev.filter(pl.col("series") == sr)
        for th in THETAS:
            for band in MINUTE_BANDS:
                tr = take_trades(part, th, band)
                cells.append({"series": sr, "theta": th, "minutes": list(band), **summarize(tr)})
    pv = [p_one_sided(c["t_2way"]) if c["n"] >= MIN_DEV_TRADES else 1.0 for c in cells]
    for c, q in zip(cells, holm(pv), strict=True):
        c["holm_p"] = round(q, 4)
    res["dev_cells"] = cells
    res["n_cells"] = len(cells)
    # second family: buy the favourite by UTC hour block x weekend (8 cells + the all-times baseline)
    tcells = []
    for we in (False, True):
        for hb in range(4):
            tr = take_favourite(dev, (5,), weekend=we, hours=(6 * hb, 6 * hb + 6))
            tcells.append({"weekend": we, "utc_hours": [6 * hb, 6 * hb + 6], **summarize(tr)})
    tp = [p_one_sided(c["t_2way"]) if c["n"] >= 50 else 1.0 for c in tcells]
    for c, q in zip(tcells, holm(tp), strict=True):
        c["holm_p"] = round(q, 4)
    res["time_fav_cells"] = tcells
    res["time_fav_all_times"] = summarize(take_favourite(dev, (5,), weekend=None, hours=None))
    (OUT / "study_dev.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("mechanics", "proxy")}, indent=1, default=str))
    for sr, rows in res["calibration_dev"].items():
        print(sr, [(r["m"], r["y_minus_mid_c"], r["b_gap"], r["t_gap"], r["brier_mid"], r["brier_fair"]) for r in rows])
    print(json.dumps(res["leadlag_dev"], indent=0))
    for r in res["time_dev"]:
        print(r)
    ok = sorted([c for c in cells if c["n"] >= MIN_DEV_TRADES], key=lambda c: -(c["t_2way"] or -99))
    for c in ok[:15]:
        print(c)
    print("time_fav all", res["time_fav_all_times"])
    for c in tcells:
        print("time_fav", c)


def cmd_live(a: argparse.Namespace) -> None:
    k, cb = live_frame()
    now_ns = datetime.now(UTC).timestamp() * 1e9
    rtt = ((k["t_recv"] - k["t_send"]) / 1e6).to_numpy()
    cbl = ((cb["t_recv"] / 1e3 - cb["ts_server"].dt.epoch("us")) / 1e3).to_numpy()
    tickers = k["ticker"].unique().to_list()
    mk = live_markets(tickers)
    mk = mk.filter(pl.col("close_time").dt.epoch("ns") < now_ns - 120e9)
    t_lo = datetime.fromtimestamp(k["t_send"].min() / 1e9, tz=UTC) - timedelta(hours=2)
    t_hi = datetime.fromtimestamp(k["t_send"].max() / 1e9, tz=UTC)
    sds = []
    for sr in sorted(k["series"].unique().to_list()):
        prod = LIVE_PRODUCT[sr]
        sp = fetch_spot(prod, start=t_lo.replace(second=0, microsecond=0), end=t_hi.replace(second=0, microsecond=0))
        sds.append(spot_features(sp).select("ts", "sd1m").with_columns(pl.lit(prod).alias("product")))
    f = live_fair(k.filter(pl.col("ticker").is_in(mk["ticker"].to_list())), cb, mk, pl.concat(sds))
    _save(f, OUT / "live_polls.parquet")
    res = {"polls": k.height, "windows_settled": mk.height, "hours": round((t_hi - t_lo).total_seconds() / 3600 - 2, 2),
           "kalshi_rest_rtt_ms": {q: round(float(np.quantile(rtt, q)), 1) for q in (0.05, 0.5, 0.95, 0.99)},
           "coinbase_ws_delay_ms": {q: round(float(np.nanquantile(cbl, q)), 1) for q in (0.05, 0.5, 0.95)},
           **live_study(f), "spot_leadlag_corr_by_lag_ms": spot_leadlag()}
    (OUT / "live.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


def cmd_freeze(a: argparse.Namespace) -> None:
    """Freeze the best dev cell (highest two-way t with >= MIN_DEV_TRADES trades) before any holdout."""
    res = json.loads((OUT / "study_dev.json").read_text())
    ok = [c for c in res["dev_cells"] if c["n"] >= MIN_DEV_TRADES and c["t_2way"] is not None]
    best = max(ok, key=lambda c: c["t_2way"])
    tf = [c for c in res["time_fav_cells"] if c["n"] >= 50 and c["t_2way"] is not None]
    tbest = max(tf, key=lambda c: c["t_2way"])
    frozen = {"frozen_at": datetime.now(UTC).isoformat(), "rule": {k: best[k] for k in ("series", "theta", "minutes")},
              "fill": "now", "stake_usd": STAKE, "max_contracts": MAX_N, "dev": best, "n_cells_searched": res["n_cells"],
              "passes_holm_on_dev": best["holm_p"] < 0.05,
              "secondary_time_fav": {"rule": {"weekend": tbest["weekend"], "utc_hours": tbest["utc_hours"], "minutes": [5],
                                              "mid_band": [0.55, 0.90]},
                                     "dev": tbest, "n_cells_searched": len(res["time_fav_cells"]),
                                     "note": "family defined after seeing the dev time_structure table (8 descriptive "
                                             "cells); total cells looked at = 72 + 8 + 8"}}
    if (OUT / "frozen.json").exists():
        raise SystemExit("frozen.json exists; the holdout rule is frozen once")
    (OUT / "frozen.json").write_text(json.dumps(frozen, indent=1, default=str))
    print(json.dumps(frozen, indent=1, default=str))


def _metrics(tr: pl.DataFrame, variant: str) -> dict[str, Any]:
    """``vol_metrics.ledger_metrics`` on the trade ledger. Positions last at most 15 minutes, so no leg
    is open at a UTC day end (a window closing at 00:00 settles at that instant): the daily MTM series
    equals settlement PnL and no candle marks are needed (an empty mark table is passed)."""
    from albert.research.vol_metrics import ledger_metrics
    led = ledger_rows(tr, "crypto_fast_taker", variant)
    (OUT / "ledgers").mkdir(parents=True, exist_ok=True)
    _save(led, OUT / "ledgers" / f"ledger_{variant}.parquet")
    marks = pl.DataFrame(schema={"ticker": pl.String, "ts": pl.Datetime("us", "UTC"), "yes_mid": pl.Float64})
    return ledger_metrics(led, marks)


def cmd_holdout(a: argparse.Namespace) -> None:
    fr = json.loads((OUT / "frozen.json").read_text())
    rule = fr["rule"]
    g = load_grid()
    if rule["series"] != "ALL":
        g = g.filter(pl.col("series") == rule["series"])
    res: dict[str, Any] = {"frozen": fr}
    for split in ("dev", "holdout"):
        part = g.filter(pl.col("split") == split)
        for fill in ("now", "next", "worst"):
            tr = take_trades(part, rule["theta"], tuple(rule["minutes"]), fill=fill)
            res[f"{split}_{fill}"] = summarize(tr)
            if fill == "now" and tr.height:
                res[f"{split}_metrics"] = _metrics(tr, f"{split}_{rule['series']}_th{rule['theta']}_m{rule['minutes'][0]}-{rule['minutes'][1]}")
                if split == "holdout":
                    pf = print_fills(tr)
                    _save(pf, OUT / "holdout_print_fills.parquet")
                    j = tr.join(pf.select("ticker", "px_1s", "px_5s"), on="ticker", how="left")
                    for col in ("px_1s", "px_5s"):
                        x = j.filter(pl.col(col).is_not_null()).with_columns(pl.col(col).alias("price")).drop("fee", "n", "pnl_c")
                        if not x.height:
                            continue
                        n = [size_for(p) for p in x["price"].to_list()]
                        fee = [kfee(k, p) for k, p in zip(n, x["price"].to_list(), strict=True)]
                        x = x.with_columns(pl.Series("n", n), pl.Series("fee", fee)).with_columns(
                            (100 * (pl.col("payoff") - pl.col("price") - pl.col("fee") / pl.col("n"))).alias("pnl_c"))
                        res[f"holdout_tape_{col}"] = {**summarize(x), "coverage": round(x.height / tr.height, 3),
                                                      "mean_slip_c": round(100 * float((j[col] - j["price"]).drop_nulls().mean()), 2)}
    sec = fr.get("secondary_time_fav")
    if sec:
        sr_ = sec["rule"]
        gall = load_grid()
        for split in ("dev", "holdout"):
            part = gall.filter(pl.col("split") == split)
            for fill in ("now", "next"):
                tr = take_favourite(part, tuple(sr_["minutes"]), weekend=sr_["weekend"], hours=tuple(sr_["utc_hours"]),
                                    lo=sr_["mid_band"][0], hi=sr_["mid_band"][1], fill=fill)
                res[f"time_fav_{split}_{fill}"] = summarize(tr)
                if fill == "now" and tr.height:
                    res[f"time_fav_{split}_metrics"] = _metrics(tr, f"timefav_{split}_we{int(sr_['weekend'])}_h{sr_['utc_hours'][0]}")
            # the same rule on every other block (context, not a test)
            if split == "holdout":
                res["time_fav_holdout_all_times"] = summarize(take_favourite(part, (5,), weekend=None, hours=None))
    (OUT / "holdout.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


def cmd_hourly_fetch(a: argparse.Namespace) -> None:
    from albert.venues.kalshi import Kalshi
    for sr, (_m15, prod) in HOURLY.items():
        p = OUT / f"markets_{sr}.parquet"
        if not p.exists():
            _save(fetch_hourly_markets(sr), p)
        mk = pl.read_parquet(p)
        sf = spot_features(pl.read_parquet(CACHE / "spot" / f"{prod}.parquet"))
        sm = hourly_sample(mk, sf)
        _save(sm, OUT / f"hourly_sample_{sr}.parquet")
        print(sr, "events", sm["event_ticker"].n_unique(), "markets", sm.height, flush=True)
        cutoff = Kalshi(kalshi_client()).cutoff()["market_settled_ts"]
        k = fetch_k1m_window(sm, cutoff, HOURLY_LAST, log=lambda x: print(x, flush=True))
        _save(k, CACHE / "k1m" / f"{sr}.parquet")
        print(sr, "candles", k.height, flush=True)


def cmd_hourly_build(a: argparse.Namespace) -> None:
    for sr, (m15, prod) in HOURLY.items():
        sm = pl.read_parquet(OUT / f"hourly_sample_{sr}.parquet")
        sf = spot_features(pl.read_parquet(CACHE / "spot" / f"{prod}.parquet"))
        basis = rti_basis(load_markets(m15), sf)
        g = build_hourly_grid(sm, pl.read_parquet(CACHE / "k1m" / f"{sr}.parquet"), sf, basis)
        _save(g, OUT / f"hgrid_{sr}.parquet")
        print(sr, g.height, "rows; quoted", round(float(g["mid"].is_not_null().mean()), 3),
              "basis bp median", round(1e4 * float(basis["basis"].median()), 2))


def cmd_proxy(a: argparse.Namespace) -> None:
    """Coinbase minute ohlc4 vs the CF RTI 60-s averages on every window (no Kalshi prices needed)."""
    out = {}
    for sr, prod in SERIES_15M.items():
        m = load_markets(sr)
        sf = spot_features(pl.read_parquet(CACHE / "spot" / f"{prod}.parquet")).select("ts", "ohlc4")
        w = (m.join(sf.rename({"ts": "open_time", "ohlc4": "k_proxy"}), on="open_time", how="left")
             .join(sf.rename({"ts": "close_time", "ohlc4": "e_proxy"}), on="close_time", how="left")
             .drop_nulls(["k_proxy", "e_proxy"]))
        w = w.with_columns(pl.Series("m", [15] * w.height), pl.col("close_time").map_elements(split_of, return_dtype=pl.String).alias("split"))
        b = 1e4 * (w["floor_strike"] / w["k_proxy"]).log()
        out[sr] = {**proxy_quality(w), "basis_bp_p05_p95": [round(float(b.quantile(0.05)), 2), round(float(b.quantile(0.95)), 2)],
                   "move_bp_median_abs": round(float((1e4 * (w["expiration_value"] / w["floor_strike"]).log()).abs().median()), 1)}
        print(sr, out[sr], flush=True)
    (OUT / "proxy.json").write_text(json.dumps(out, indent=1))


def cmd_fairmodel(a: argparse.Namespace) -> None:
    """Spot-only fair value on EVERY window (no Kalshi prices): calibration slope of the outcome on
    logit(fair) by minute, and whether recent spot momentum adds (H4: drift / mean reversion)."""
    from albert.research.fundamental import fit_logistic, logit
    out: dict[str, Any] = {}
    for sr, prod in SERIES_15M.items():
        mk = load_markets(sr)
        sf = spot_features(pl.read_parquet(CACHE / "spot" / f"{prod}.parquet"))
        g = build_grid(mk, pl.DataFrame(schema=K1M_SCHEMA), sf)
        g = g.join(sf.select(pl.col("ts").alias("t"), pl.col("spot").alias("s_now")), on="t", how="left").join(
            sf.select((pl.col("ts") + pl.duration(minutes=3)).alias("t"), pl.col("spot").alias("s_3m")), on="t", how="left")
        g = g.with_columns((pl.col("s_now") / pl.col("s_3m")).log().alias("r3"))
        rows = []
        for m in (1, 3, 5, 8, 11, 13, 14):
            r = g.filter((pl.col("m") == m) & (pl.col("split") == "dev")).drop_nulls(["fair", "r3", "sd1m"])
            if r.height < 200:
                continue
            y = r["y"].to_numpy().astype(float)
            lf = logit(r["fair"].to_numpy(), 1e-4)
            mom = (r["r3"] / r["sd1m"]).to_numpy()   # 3-minute return in 1-minute sd units
            beta, tst = fit_logistic(np.column_stack([np.ones(r.height), lf, mom]), y, r["date"].cast(pl.String).to_numpy())
            rows.append({"m": m, "n": r.height, "slope_logit_fair": round(float(beta[1]), 3), "t_slope": round(float(tst[1]), 1),
                         "b_mom3": round(float(beta[2]), 3), "t_mom3": round(float(tst[2]), 2),
                         "brier_fair": round(float(((r["fair"] - r["y"]) ** 2).mean()), 4),
                         "brier_half": round(float(((0.5 - r["y"]) ** 2).mean()), 4)})
        out[sr] = rows
        print(sr, rows, flush=True)
    (OUT / "fairmodel_dev.json").write_text(json.dumps(out, indent=1))


def cmd_spot(a: argparse.Namespace) -> None:
    for p in a.products:
        df = fetch_spot(p)
        _save(df, CACHE / "spot" / f"{p}.parquet")
        print(p, df.height, df["ts"].min(), df["ts"].max())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.crypto_fast")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("markets")
    m.add_argument("--series", nargs="+", default=list(SERIES_15M))
    s = sub.add_parser("spot")
    s.add_argument("--products", nargs="+", default=sorted(set(SERIES_15M.values())))
    c = sub.add_parser("candles")
    c.add_argument("--series", nargs="*", default=None)
    c.add_argument("--cached-hist-only", action="store_true")
    sub.add_parser("build")
    sub.add_parser("live")
    sub.add_parser("freeze")
    sub.add_parser("proxy")
    sub.add_parser("fairmodel")
    sub.add_parser("hourly-fetch")
    sub.add_parser("hourly-build")
    sub.add_parser("holdout")
    sub.add_parser("study")
    r = sub.add_parser("record")
    r.add_argument("--hours", type=float, default=6.0)
    r.add_argument("--series", nargs="+", default=["KXBTC15M", "KXSOL15M"])
    r.add_argument("--products", nargs="+", default=["BTC-USD", "SOL-USD"])
    r.add_argument("--interval", type=float, default=0.5, help="seconds between Kalshi polls (>= 0.5)")
    a = ap.parse_args(argv)
    if a.cmd == "record":
        record(a.hours, a.series, a.products, interval=a.interval)
        return
    {"markets": cmd_markets, "spot": cmd_spot, "candles": cmd_candles, "build": cmd_build,
     "study": cmd_study, "live": cmd_live, "freeze": cmd_freeze, "holdout": cmd_holdout, "proxy": cmd_proxy, "fairmodel": cmd_fairmodel,
     "hourly-fetch": cmd_hourly_fetch, "hourly-build": cmd_hourly_build}[a.cmd](a)


if __name__ == "__main__":
    main()
