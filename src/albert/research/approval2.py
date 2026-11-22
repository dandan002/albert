"""Presidential-approval and poll-average markets on Kalshi, round 2 (after ``approval.py``).

Run: ``python -m albert.research.approval2 <subcommand>`` (see ``main``). Outputs go to
``data/research/approval2/``; raw Kalshi responses go through ``albert.http.CachedClient``.
The report (``data/research/approval2/report.md``) holds the hypotheses, the dev screen, the frozen
rule (``primary.json``) and the holdout.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from albert.venues.kalshi import _field, num, parse_ts

OUT = Path("data/research/approval2")
CACHE = Path("data/cache/approval2")
UA = "albert-research/0.2 (public-data research)"
ET = ZoneInfo("America/New_York")
DEV_END = datetime(2026, 10, 1, tzinfo=UTC)          # dev: settles before this; holdout: on/after
KALSHI_MIN_INTERVAL = 1.0                            # shared public rate limit: sequential, 1 req/s

# Candidate series (today's /series, filtered by title/ticker for approval, favourability, poll
# averages and generic ballot; reviewed 2026-11-20). Source = settlement source per the rules.
SERIES: dict[str, str] = {
    "KXTRUMPAPPROVE": "RCP", "KXAPRPOTUS": "RCP", "KXAPRPOTUSD": "RCP", "APRPOTUS": "RCP",
    "KXAPRPOTUSEOY": "RCP", "KXTRUMPAPPROVALBELOW": "?", "KXTRUMPAPPROVALYEAR": "?",
    "KXTRUMPPOLLDAILY": "?", "KXTRUMPVH": "VoteHub", "KXVHAPRPOTUS": "VoteHub", "KXVHGCB": "VoteHub",
    "KXGENERICBALLOTVOTEHUB": "VoteHub", "KXVOTEHUBTRUMPUPDOWN": "VoteHub",
    "KXVOTEHUBVANCEAPPROVE": "VoteHub", "KXSILVERAPPROVE": "Silver Bulletin",
    "KXSILVERAPPROVALMUSK": "Silver Bulletin", "KXTRUMPFAV": "?", "KXGENERICBALLOT": "?",
    "KXNYTTRUMPAPPROVAL": "NYT", "KXTRUMPGALLUP30": "Gallup", "KXTRUMPGALLUP50": "Gallup",
    "KXLEADPOLL": "?", "KXUKPOLLFIRST": "?", "KXMAMDANIPOLL": "Siena", "KX538APPROVE": "538",
}

MARKET_COLS = {"ticker": pl.String, "event_ticker": pl.String, "series": pl.String, "title": pl.String,
               "yes_sub_title": pl.String, "strike_type": pl.String, "floor_strike": pl.Float64,
               "cap_strike": pl.Float64, "open_time": pl.Datetime("us", "UTC"),
               "close_time": pl.Datetime("us", "UTC"), "expected_expiration_time": pl.Datetime("us", "UTC"),
               "settlement_ts": pl.Datetime("us", "UTC"), "result": pl.String, "expiration_value": pl.String,
               "volume": pl.Float64, "rules_primary": pl.String, "status": pl.String, "src": pl.String}


def _api(min_interval: float = KALSHI_MIN_INTERVAL, record_live: bool = True):
    from albert.config import Settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    s = Settings()
    return k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=min_interval, max_retries=12,
                                 record_live=record_live))


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def market_row(m: dict[str, Any], series: str, src: str) -> dict[str, Any]:
    ev = m.get("expiration_value")
    return {"ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"), "series": series,
            "title": m.get("title"), "yes_sub_title": m.get("yes_sub_title"), "strike_type": m.get("strike_type"),
            "floor_strike": num(m.get("floor_strike")), "cap_strike": num(m.get("cap_strike")),
            "open_time": parse_ts(m.get("open_time")), "close_time": parse_ts(m.get("close_time")),
            "expected_expiration_time": parse_ts(m.get("expected_expiration_time")),
            "settlement_ts": parse_ts(m.get("settlement_ts")), "result": m.get("result") or None,
            "expiration_value": str(ev) if ev not in (None, "") else None,
            "volume": num(_field(m, "volume")), "rules_primary": m.get("rules_primary"),
            "status": m.get("status"), "src": src}


def _pages(api, path: str, params: dict[str, Any], tag: str | None = None):
    cursor = None
    while True:
        page = api.client.get(path, {**params, "cursor": cursor}, cache=True, tag=tag)
        yield page
        cursor = page.payload.get("cursor")
        if not cursor or not page.payload.get("markets"):
            return


def discover(out: Path = OUT, series: list[str] | None = None, *, log=print) -> pl.DataFrame:
    """Series metadata (today's /series) and every settled + open market of each candidate series:
    ``/historical/markets`` (immutable, cached) plus ``/markets?status=settled|open`` (cache keyed by
    today's date, so a re-run on a later day sees new settlements; older responses are kept)."""
    api = _api()
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    ser = api.series().payload.get("series") or []
    meta = []
    names = series or list(SERIES)
    for s in ser:
        if s.get("ticker") in names:
            src = (s.get("settlement_sources") or [{}])[0] or {}
            meta.append({"series": s["ticker"], "title": s.get("title"), "category": s.get("category"),
                         "frequency": s.get("frequency"), "fee_type": s.get("fee_type"),
                         "fee_multiplier": num(s.get("fee_multiplier")), "source_name": src.get("name"),
                         "source_url": src.get("url"), "contract_url": s.get("contract_url")})
    meta_df = pl.DataFrame(meta)
    _save(meta_df, out / "series.parquet")
    rows: list[dict[str, Any]] = []
    for t in names:
        n0 = len(rows)
        for page in _pages(api, "/historical/markets", {"series_ticker": t, "limit": 1000}):
            rows.extend(market_row(m, t, "hist") for m in page.payload.get("markets") or [])
        for status in ("settled", "open"):
            for page in _pages(api, "/markets", {"series_ticker": t, "status": status, "limit": 1000}, tag=day):
                rows.extend(market_row(m, t, status) for m in page.payload.get("markets") or [])
        log(f"  {t}: {len(rows) - n0} markets", flush=True)
    mk = pl.DataFrame(rows, schema=MARKET_COLS).unique("ticker", keep="last")
    _save(mk, out / "markets.parquet")
    log(f"discover: {meta_df.height} series, {mk.height} markets")
    return mk


# ---- candles -----------------------------------------------------------------------------------
def _ohlc(node: dict[str, Any] | None, key: str) -> float | None:
    return num(_field(node or {}, key))


def candle_rows(payload: dict[str, Any], ticker: str, period: int) -> list[dict[str, Any]]:
    """Candles with bid/ask closes and the traded price range. Kalshi returns only periods in
    which something changed (1-minute candles are sparse); callers forward-fill quotes."""
    rows = []
    for c in payload.get("candlesticks") or []:
        bid, ask, pr = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
        rows.append({"ticker": ticker, "end_ts": parse_ts(c.get("end_period_ts")), "period": period,
                     "bid": _ohlc(bid, "close"), "ask": _ohlc(ask, "close"),
                     "bid_lo": _ohlc(bid, "low"), "ask_hi": _ohlc(ask, "high"),
                     "p_lo": _ohlc(pr, "low"), "p_hi": _ohlc(pr, "high"), "p_close": _ohlc(pr, "close"),
                     "volume": num(_field(c, "volume")), "oi": num(_field(c, "open_interest"))})
    return rows


CANDLE_SCHEMA = {"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "period": pl.Int64,
                 "bid": pl.Float64, "ask": pl.Float64, "bid_lo": pl.Float64, "ask_hi": pl.Float64,
                 "p_lo": pl.Float64, "p_hi": pl.Float64, "p_close": pl.Float64, "volume": pl.Float64,
                 "oi": pl.Float64}

# (series, period minutes, how much of each market's life: None = all, else the last N hours,
#  minimum close time)
CANDLE_PLAN: list[tuple[str, int, int | None, datetime | None]] = [
    ("KXTRUMPAPPROVE", 1, None, None),
    ("KXAPRPOTUSD", 1, None, None),
    ("KXVHGCB", 1, None, None),
    ("KXAPRPOTUS", 60, None, datetime(2025, 1, 20, tzinfo=UTC)),     # Trump second term only
    ("KXTRUMPVH", 60, None, None),
    ("KXVHAPRPOTUS", 60, None, None),
    ("KXGENERICBALLOTVOTEHUB", 60, None, None),
    ("KXVOTEHUBTRUMPUPDOWN", 60, None, None),
    ("KXAPRPOTUS", 1, 72, datetime(2026, 8, 28, tzinfo=UTC)),       # Fridays overlapping the daily series
]


def _candle_jobs(out: Path, plan, cutoff) -> list[tuple[str, int, str, datetime, datetime, bool]]:
    mk = pl.read_parquet(out / "markets.parquet").filter(pl.col("result").is_in(["yes", "no"]))
    jobs = []
    for series, period, last_h, min_close in plan:
        sel = mk.filter(pl.col("series") == series)
        if min_close is not None:
            sel = sel.filter(pl.col("close_time") >= min_close)
        for m in sel.sort("close_time").iter_rows(named=True):
            hi = m["close_time"]
            lo = m["open_time"] or hi - timedelta(days=8)
            if last_h is not None:
                lo = max(lo, hi - timedelta(hours=last_h))
            lo = max(lo, hi - timedelta(minutes=period * 4900))
            jobs.append((series, period, m["ticker"], lo, hi + timedelta(minutes=period), bool(cutoff and hi < cutoff)))
    return jobs


def fetch_candles(out: Path = OUT, plan: list[tuple[str, int, int | None, datetime | None]] = CANDLE_PLAN,
                  *, min_interval: float = 0.5, parse: bool = False, log=print) -> None:
    """Two phases. ``parse=False`` (network only, holds nothing): request every planned candle
    window once; ``CachedClient`` stores each response (append-only), so re-runs only fetch what is
    missing. ``parse=True`` (no network for cached windows): rebuild ``candles_{period}m.parquet``
    from the cache one (series, period) part at a time. Settled markets only."""
    from albert.http import HttpError

    api = _api(min_interval)
    cutoff = api.cutoff().get("market_settled_ts")
    jobs = _candle_jobs(out, plan, cutoff)
    parts: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for i, (series, period, ticker, lo, hi, hist) in enumerate(jobs, 1):
        try:
            page = api.candles(ticker, lo, hi, period, hist)
        except (HttpError, RuntimeError) as error:
            log(f"skip {ticker} {period}m: {str(error)[:100]}")
            page = None
        if parse:
            rows = candle_rows(page.payload, ticker, period) if page is not None else []
            parts.setdefault((series, period), []).extend(rows)
        if i % 250 == 0:
            log(f"  candles {i}/{len(jobs)} ({series} {period}m)", flush=True)
        if parse and (i == len(jobs) or jobs[i][:2] != (series, period)):
            df = pl.DataFrame(parts.pop((series, period)), schema=CANDLE_SCHEMA).unique(["ticker", "end_ts"], keep="last")
            _save(df.sort("ticker", "end_ts"), out / "candle_parts" / f"{series}_{period}m.parquet")
    if parse:
        for period in sorted({p for _, p, *_ in plan}):
            files = sorted((out / "candle_parts").glob(f"*_{period}m.parquet"))
            if files:
                _save(pl.concat([pl.read_parquet(f) for f in files]), out / f"candles_{period}m.parquet")
                log(f"wrote candles_{period}m.parquet from {len(files)} parts")


# ---- legs, settled values, quotes --------------------------------------------------------------
_DASH = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%?\s*-\s*(\d+(?:\.\d+)?)\s*%?\s*$")


def bracket(sub: str | None) -> tuple[int, int] | None:
    """Inclusive [lo, hi] in tenths for a leg title ("38.5%", "38.8 to 39.0", "37.5-37.7%",
    "Above 38.5%" -> [386, inf], "Below 37.9%" -> [-inf, 378]); see ``approval.bracket_tenths``."""
    from albert.research.approval import bracket_tenths

    s = (sub or "").strip()
    if m := _DASH.match(s):
        return round(float(m.group(1)) * 10), round(float(m.group(2)) * 10)
    return bracket_tenths(s.replace("%", "")) if s else None


def settle_instant(close: datetime) -> datetime:
    """Scheduled settlement instant: the close rounded up to the hour (1 PM ET legs close at :59)."""
    if close.minute == 0 and close.second == 0:
        return close
    return close.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)


def legs(mk: pl.DataFrame, series: list[str]) -> pl.DataFrame:
    """Settled legs with bounds (tenths), settlement instant ``T`` and outcome ``y``."""
    rows = []
    for r in mk.filter(pl.col("series").is_in(series) & pl.col("result").is_in(["yes", "no"])).iter_rows(named=True):
        b = bracket(r["yes_sub_title"])
        if b is None:
            continue
        rows.append({"ticker": r["ticker"], "event": r["event_ticker"], "series": r["series"], "lo": b[0], "hi": b[1],
                     "T": settle_instant(r["close_time"]), "open_time": r["open_time"], "settled_at": r["settlement_ts"],
                     "y": int(r["result"] == "yes"), "ev": r["expiration_value"]})
    return pl.DataFrame(rows).sort("T", "event", "lo")


def _ev_tenths(s: str | None) -> int | None:
    try:
        return round(float(str(s).replace("%", "").strip()) * 10)
    except (TypeError, ValueError):
        return None


def event_values(lg: pl.DataFrame) -> pl.DataFrame:
    """Per event: exact settled value in tenths when known (``expiration_value`` or a winning
    single-value leg), else null; ``win_lo``/``win_hi`` = the winning leg's bounds; public at
    ``settled_at`` (the latest settlement stamp of its legs)."""
    rows = []
    for (ev,), g in lg.group_by(["event"], maintain_order=True):
        w = g.filter(pl.col("y") == 1)
        exact = None
        for s in g["ev"].to_list():
            if (v := _ev_tenths(s)) is not None:
                exact = v
                break
        lo = w["lo"][0] if w.height else None
        hi = w["hi"][0] if w.height else None
        if exact is None and lo is not None and lo == hi:
            exact = lo
        rows.append({"event": ev, "series": g["series"][0], "T": g["T"][0], "settled_at": g["settled_at"].max(),
                     "value": exact, "win_lo": lo, "win_hi": hi, "n_legs": g.height, "n_yes": w.height})
    return pl.DataFrame(rows).sort("series", "T")


def load_candles(period: int, tickers: list[str] | None = None, out: Path = OUT) -> pl.DataFrame:
    c = pl.scan_parquet(out / f"candles_{period}m.parquet")
    if tickers is not None:
        c = c.filter(pl.col("ticker").is_in(tickers))
    return c.collect()


def quotes_at(cand: pl.DataFrame, frame: pl.DataFrame, *, max_age: timedelta | None = None) -> pl.DataFrame:
    """Attach the YES bid/ask in force at ``t`` (last candle that ended at or before ``t``; quotes
    are forward-filled because Kalshi omits periods without changes). ``frame`` needs ticker, t."""
    q = (cand.sort("ticker", "end_ts")
         .with_columns(pl.col("bid", "ask").forward_fill().over("ticker"))
         .select("ticker", pl.col("end_ts").alias("q_ts"), "bid", "ask"))
    out = frame.sort("ticker", "t").join_asof(q, left_on="t", right_on="q_ts", by="ticker", strategy="backward",
                                              check_sortedness=False)
    if max_age is not None:
        out = out.filter((pl.col("t") - pl.col("q_ts")) <= max_age)
    return out


# ---- execution and statistics ------------------------------------------------------------------
def order_fee(n: int, p: float, mult: float = 1.0) -> float:
    """Kalshi taker fee for one order of ``n`` contracts at ``p``, rounded up to the cent."""
    return math.ceil(0.07 * mult * n * p * (1 - p) * 100 - 1e-9) / 100


def contracts(price: float, budget: float) -> int:
    return max(1, int(budget // price)) if price > 0 else 0


def taker_fill(side: str, bid: float | None, ask: float | None, budget: float = 2.0) -> dict[str, Any] | None:
    """Taker order of about ``budget`` dollars: YES at the ask or NO at 1 - bid."""
    p = ask if side == "yes" else (1 - bid if bid is not None else None)
    if p is None or not (0 < p < 1):
        return None
    n = contracts(p, budget)
    fee = order_fee(n, p)
    return {"side": side, "price": p, "n": n, "fee": fee, "cost": n * p + fee}


def cluster_t(x: np.ndarray, g: np.ndarray) -> tuple[float, float | None, int]:
    from albert.research.fundamental import clustered_mean

    return clustered_mean(np.asarray(x, float), np.asarray(g))


def two_way_t(x: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> float | None:
    """t of the mean with Cameron-Gelbach-Miller two-way clustered SE (``g1`` events, ``g2`` dates)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return None
    e = x - x.mean()

    def v(g: np.ndarray) -> float:
        _, inv = np.unique(g, return_inverse=True)
        s = np.bincount(inv, weights=e)
        return float((s ** 2).sum()) / n ** 2

    g12 = np.array([f"{a}|{b}" for a, b in zip(g1, g2, strict=True)])
    var = v(np.asarray(g1)) + v(np.asarray(g2)) - v(g12)
    return float(x.mean() / math.sqrt(var)) if var > 0 else None


def summarize(f: pl.DataFrame) -> dict[str, Any]:
    """Fills table -> per-contract and per-$ numbers, event-clustered and two-way t, % positive
    weeks. ``f`` columns: event, T, n, price, fee, cost, payoff (dollars)."""
    if f.height == 0:
        return {"fills": 0}
    pnl = (f["payoff"] - f["cost"]).to_numpy()
    per_c = ((f["payoff"] - f["cost"]) / f["n"]).to_numpy()
    mu_c, t_c, k = cluster_t(per_c, f["event"].to_numpy())
    dates = f["T"].dt.date().cast(pl.String).to_numpy()
    wk = f.with_columns((pl.col("payoff") - pl.col("cost")).alias("pnl"), pl.col("T").dt.truncate("1w").alias("wk")) \
          .group_by("wk").agg(pl.col("pnl").sum())
    return {"fills": f.height, "events": k, "contracts": int(f["n"].sum()), "c_per_contract": round(100 * mu_c, 2),
            "t_event": None if t_c is None else round(t_c, 2),
            "t_two_way": None if (t2 := two_way_t(per_c, f["event"].to_numpy(), dates)) is None else round(t2, 2),
            "pnl_usd": round(float(pnl.sum()), 2), "risked_usd": round(float(f["cost"].sum()), 2),
            "pnl_per_dollar": round(float(pnl.sum() / f["cost"].sum()), 4),
            "weeks": wk.height, "pct_pos_weeks": round(100 * float((wk["pnl"] > 0).mean()), 1),
            "win_rate": round(float((f["payoff"] > 0).mean()), 3)}


# ---- ladder decision frame and rules -----------------------------------------------------------
LEADS: dict[str, list[float]] = {     # hours before the scheduled settlement instant
    "KXTRUMPAPPROVE": [22, 12, 6, 4, 3, 2, 1, 0.5, 1 / 6],
    "KXAPRPOTUS": [144, 96, 48, 24, 12, 4, 2, 1],
}


def ladder_frame(lg: pl.DataFrame, vals: pl.DataFrame, cand: pl.DataFrame, leads: list[float],
                 min_open: timedelta = timedelta(minutes=30)) -> pl.DataFrame:
    """One row per (leg, decision time t = T - lead) with the quote in force at t and context that
    was public at t: ``prev`` = the series' latest settled value (tenths; exact or the winning
    bracket's centre when only a bracket is known) whose settlement stamp is <= t."""
    dec = (lg.join(pl.DataFrame({"lead_h": [float(x) for x in leads]}), how="cross")
           .with_columns((pl.col("T") - pl.duration(seconds=(pl.col("lead_h") * 3600).cast(pl.Int64))).alias("t"))
           .filter(pl.col("t") >= pl.col("open_time") + min_open))
    q = quotes_at(cand, dec.select("ticker", "t", "lead_h"))
    dec = dec.join(q.select("ticker", "t", "bid", "ask", "q_ts"), on=["ticker", "t"], how="left")
    pv = (vals.with_columns(pl.coalesce(pl.col("value"), ((pl.col("win_lo") + pl.col("win_hi")) / 2)
                                        .round().cast(pl.Int64)).alias("pv"))
          .filter(pl.col("pv").is_not_null() & (pl.col("pv") > 0) & (pl.col("pv") < 1000))
          .select("series", pl.col("settled_at").alias("pv_at"), "pv").sort("pv_at"))
    dec = dec.sort("t").join_asof(pv, left_on="t", right_on="pv_at", by="series", strategy="backward",
                                  check_sortedness=False)
    dec = dec.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                           pl.when(pl.col("lo") == pl.col("hi")).then(pl.col("lo"))
                           .when(pl.col("lo") <= 0).then(pl.col("hi"))
                           .when(pl.col("hi") >= 1000).then(pl.col("lo"))
                           .otherwise((pl.col("lo") + pl.col("hi")) // 2).alias("centre"))
    rank = pl.col("mid").rank("ordinal", descending=True).over("event", "lead_h")
    dec = dec.with_columns(rank.alias("mid_rank"),
                           pl.col("mid").sum().over("event", "lead_h").alias("mid_sum"),
                           pl.col("T").dt.convert_time_zone("America/New_York").dt.weekday().alias("wd"))
    modal = (dec.filter(pl.col("mid_rank") == 1)
             .select("event", "lead_h", pl.col("centre").alias("modal_centre"), pl.col("mid").alias("modal_mid")))
    dec = dec.join(modal, on=["event", "lead_h"], how="left")
    contains_prev = (pl.col("pv") >= pl.col("lo")) & (pl.col("pv") <= pl.col("hi"))
    dist_prev = pl.when(contains_prev).then(0).when(pl.col("pv") < pl.col("lo")).then(pl.col("lo") - pl.col("pv")) \
        .otherwise(pl.col("hi") - pl.col("pv"))
    return dec.with_columns(contains_prev.alias("has_prev"), dist_prev.alias("d_prev"),
                            (pl.col("centre") - pl.col("modal_centre")).alias("d_modal"))


def fills_from(sel: pl.DataFrame, side: str, budget: float = 2.0, per_event: float = 5.0) -> pl.DataFrame:
    """Taker fills for selected (leg, t) rows: YES at the ask or NO at 1 - bid, about ``budget``
    dollars each, at most ``per_event`` dollars per event (cheapest orders first), held to
    settlement. One position per (ticker): the earliest qualifying decision."""
    rows = []
    sel = sel.sort("t").unique(["ticker"], keep="first", maintain_order=True)
    spent: dict[str, float] = {}
    for r in sel.sort("t", "ticker").iter_rows(named=True):
        f = taker_fill(side, r["bid"], r["ask"], budget)
        if f is None:
            continue
        if spent.get(r["event"], 0.0) + f["cost"] > per_event + 1e-9:
            continue
        spent[r["event"]] = spent.get(r["event"], 0.0) + f["cost"]
        won = r["y"] == 1 if side == "yes" else r["y"] == 0
        rows.append({"event": r["event"], "ticker": r["ticker"], "series": r["series"], "T": r["T"], "t": r["t"],
                     "lead_h": r["lead_h"], "side": side, "n": f["n"], "price": f["price"], "fee": f["fee"],
                     "cost": f["cost"], "payoff": float(f["n"]) if won else 0.0})
    schema = {"event": pl.String, "ticker": pl.String, "series": pl.String, "T": pl.Datetime("us", "UTC"),
              "t": pl.Datetime("us", "UTC"), "lead_h": pl.Float64, "side": pl.String, "n": pl.Int64,
              "price": pl.Float64, "fee": pl.Float64, "cost": pl.Float64, "payoff": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


def priced(df: pl.DataFrame, max_spread: float = 0.10) -> pl.DataFrame:
    return df.filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("ask") > pl.col("bid"))
                     & ((pl.col("ask") - pl.col("bid")) <= max_spread))


def rule_cells(dec: pl.DataFrame) -> list[tuple[str, str, pl.DataFrame]]:
    """The dev screen: (cell name, side, selected rows) per hypothesis x lead. Every cell is scored."""
    cells = []
    p = priced(dec)
    for L in sorted(dec["lead_h"].unique().to_list(), reverse=True):
        d = p.filter(pl.col("lead_h") == L)
        tag = f"L{L:g}h"
        cells.append((f"H4 modal YES {tag}", "yes", d.filter(pl.col("mid_rank") == 1)))
        cells.append((f"H11 modal NO {tag}", "no", d.filter(pl.col("mid_rank") == 1)))
        cells.append((f"H3 longshot NO ask<=5c {tag}", "no", d.filter(pl.col("ask") <= 0.05)))
        cells.append((f"H3 longshot NO ask<=10c {tag}", "no", d.filter(pl.col("ask") <= 0.10)))
        cells.append((f"H2 prev YES weekend {tag}", "yes", d.filter(pl.col("has_prev") & pl.col("wd").is_in([6, 7, 1]))))
        cells.append((f"H2 prev YES weekday {tag}", "yes", d.filter(pl.col("has_prev") & pl.col("wd").is_in([2, 3, 4, 5]))))
        # added after the dev calibration table (legs at 5-15c settle YES far below their price):
        cells.append((f"H3b shoulder NO 5c<mid<=15c {tag}", "no", d.filter((pl.col("mid") > 0.05) & (pl.col("mid") <= 0.15))))
        cells.append((f"H3c non-modal NO mid<=15c {tag}", "no", d.filter((pl.col("mid") <= 0.15) & (pl.col("mid_rank") > 1))))
        cells.append((f"H9 above-prev YES {tag}", "yes", d.filter(pl.col("d_prev") > 0)))
        cells.append((f"H9 below-prev YES {tag}", "yes", d.filter(pl.col("d_prev") < 0)))
    return cells


def _md(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        lines.append("| " + " | ".join("" if v is None else (f"{v:g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def build_frames(out: Path = OUT) -> dict[str, pl.DataFrame]:
    """Decision frames for the daily (1-minute quotes) and weekly (hourly quotes) RCP ladders."""
    mk = pl.read_parquet(out / "markets.parquet")
    frames = {}
    for series, period in (("KXTRUMPAPPROVE", 1), ("KXAPRPOTUS", 60)):
        lg = legs(mk, [series])
        if series == "KXAPRPOTUS":
            lg = lg.filter(pl.col("T") >= datetime(2025, 1, 20, tzinfo=UTC))
        path = out / f"candles_{period}m.parquet"
        if not path.exists():
            continue
        cand = load_candles(period, lg["ticker"].to_list(), out)
        if cand.height == 0:
            continue
        frames[series] = ladder_frame(lg, event_values(lg), cand, LEADS[series])
    return frames


def screen(out: Path = OUT) -> pl.DataFrame:
    """Dev screen (settlement before ``DEV_END`` only): every rule cell of ``rule_cells`` on both RCP
    ladders, plus the base rates the hypotheses rest on. Holdout rows are dropped before scoring."""
    frames = build_frames(out)
    res = []
    for series, dec in frames.items():
        dev = dec.filter(pl.col("T") < DEV_END)
        for name, side, sel in rule_cells(dev):
            s = summarize(fills_from(sel, side))
            res.append({"series": series, "cell": name, "side": side, **s})
    tab = pl.DataFrame(res, infer_schema_length=None)
    _save(tab, out / "dev_cells.parquet")
    (out / "dev_cells.md").write_text(_md(tab.select("series", "cell", "fills", "events", "c_per_contract", "t_event",
                                                     "pnl_per_dollar", "pct_pos_weeks")))
    return tab


def score_frozen(out: Path = OUT) -> dict[str, Any]:
    """Score the rules frozen in ``primary.json`` on dev and holdout (same code path as the
    screen: the cell is re-selected by name from ``rule_cells``), write ledgers and Sharpe/vol."""
    spec = json.loads((out / "primary.json").read_text())
    frames = build_frames(out)
    res: dict[str, Any] = {}
    led_dir = out / "ledgers"
    for role, rule in [("primary", spec["primary"])] + [(f"secondary{i + 1}", r) for i, r in enumerate(spec.get("secondaries", []))]:
        dec = frames[rule["series"]]
        cand = load_candles(1 if rule["series"] == "KXTRUMPAPPROVE" else 60, dec["ticker"].unique().to_list(), out)
        marks = marks_from(cand)
        for split in ("dev", "holdout"):
            part = dec.filter((pl.col("T") < DEV_END) if split == "dev" else (pl.col("T") >= DEV_END))
            cells = {n: (side, sel) for n, side, sel in rule_cells(part)}
            side, sel = cells[rule["cell"]]
            f = fills_from(sel, side)
            book = capped_book(f)
            led = to_ledger(book, "approval2", f"{role}:{rule['cell']}", split)
            _save(led, led_dir / f"ledger_{role}_{split}.parquet")
            res[f"{role}_{split}"] = {"rule": rule, **summarize(f), "book": {**summarize(book), **vol_block(led, marks)}}
    (out / "holdout_scores.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def friday_gaps(out: Path = OUT) -> pl.DataFrame:
    """H5: on Fridays the weekly ladder (11 AM ET, 0.3 brackets) and the daily ladder (1 PM ET, exact
    values) price nearly the same number. Per weekly bracket and decision time: weekly mid vs the
    daily-implied probability (sum of daily mids of values inside the bracket / sum of all daily
    mids; brackets cut by a daily tail are skipped), and the weekly outcome."""
    mk = pl.read_parquet(out / "markets.parquet")
    lw, ld = legs(mk, ["KXAPRPOTUS"]), legs(mk, ["KXTRUMPAPPROVE"])
    lw = lw.filter(pl.col("T") >= ld["T"].min() - timedelta(days=1))
    cw = pl.concat([load_candles(1, lw["ticker"].to_list(), out).select("ticker", "end_ts", "bid", "ask"),
                    load_candles(60, lw["ticker"].to_list(), out).select("ticker", "end_ts", "bid", "ask")])
    cd = load_candles(1, ld["ticker"].to_list(), out).select("ticker", "end_ts", "bid", "ask")
    rows = []
    for (ev,), g in lw.group_by(["event"]):
        Tw = g["T"][0]
        d = ld.filter(pl.col("T").dt.date() == Tw.date())
        if d.height == 0:
            continue
        for lead in (2.0, 1.0, 0.25):
            t = Tw - timedelta(hours=lead)
            qw = quotes_at(cw, g.select("ticker", pl.lit(t).alias("t")))
            qd = quotes_at(cd, d.select("ticker", pl.lit(t).alias("t")))
            qd = qd.join(d.select("ticker", "lo", "hi"), on="ticker").with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
            if qd["mid"].null_count() or qd.height == 0:
                continue
            tot = qd["mid"].sum()
            for r in g.join(qw.select("ticker", "bid", "ask"), on="ticker").iter_rows(named=True):
                if r["bid"] is None or r["ask"] is None:
                    continue
                cut = qd.filter((pl.col("lo") < r["lo"]) & (pl.col("hi") >= r["lo"]) | (pl.col("lo") <= r["hi"]) & (pl.col("hi") > r["hi"]))
                if cut.height:
                    continue
                inside = qd.filter((pl.col("lo") >= r["lo"]) & (pl.col("hi") <= r["hi"]))["mid"].sum()
                rows.append({"event": ev, "T": Tw, "lead_h": lead, "ticker": r["ticker"], "lo": r["lo"], "hi": r["hi"],
                             "w_bid": r["bid"], "w_ask": r["ask"], "w_mid": (r["bid"] + r["ask"]) / 2,
                             "d_implied": inside / tot, "y": r["y"]})
    df = pl.DataFrame(rows).with_columns((pl.col("d_implied") - pl.col("w_mid")).alias("gap"))
    _save(df, out / "friday_gaps.parquet")
    return df


def base_rates(out: Path = OUT, split: str = "dev") -> dict[str, Any]:
    """Descriptive dev facts: day-to-day change of the settled daily value by weekday, and the
    calibration of the modal leg / far legs by lead (market mid vs realised rate)."""
    frames = build_frames(out)
    res: dict[str, Any] = {}
    keep = (pl.col("T") < DEV_END) if split == "dev" else (pl.col("T") >= DEV_END)
    mk = pl.read_parquet(out / "markets.parquet")
    for series in frames:
        vals = event_values(legs(mk, [series])).filter(keep).sort("T")
        vals = vals.with_columns((pl.col("value") - pl.col("value").shift(1)).alias("dv"),
                                 pl.col("T").dt.convert_time_zone("America/New_York").dt.weekday().alias("wd"))
        res[f"{series}_changes"] = vals.group_by("wd").agg(pl.len().alias("n"), (pl.col("dv") == 0).mean().alias("p_same"),
                                                            pl.col("dv").abs().mean().alias("mean_abs_dv")).sort("wd").to_dicts()
        dec = priced(frames[series].filter(keep))
        cal = (dec.with_columns(pl.when(pl.col("mid_rank") == 1).then(pl.lit("modal"))
                                .when(pl.col("mid") <= 0.05).then(pl.lit("mid<=5c"))
                                .when(pl.col("mid") <= 0.15).then(pl.lit("mid 5-15c")).otherwise(pl.lit("other")).alias("grp"))
               .group_by("lead_h", "grp").agg(pl.len().alias("n"), pl.col("event").n_unique().alias("events"),
                                              pl.col("mid").mean().alias("mid"), pl.col("y").mean().alias("rate"),
                                              pl.col("ask").mean().alias("ask"), pl.col("bid").mean().alias("bid"))
               .sort("grp", "lead_h", descending=[False, True]))
        res[f"{series}_calibration"] = cal.to_dicts()
    return res


# ---- capped book, ledger, Sharpe/vol ------------------------------------------------------------
def capped_book(f: pl.DataFrame, *, bankroll: float = 100.0, max_open: float = 20.0) -> pl.DataFrame:
    """Keep fills in time order while cash covers them and open cost stays <= ``max_open``."""
    keep, open_pos, cash = [], [], bankroll
    for r in f.sort("t").iter_rows(named=True):
        still = []
        for T, c, pay in open_pos:
            if T <= r["t"]:
                cash += pay
            else:
                still.append((T, c, pay))
        open_pos = still
        if r["cost"] > cash or sum(c for _, c, _ in open_pos) + r["cost"] > max_open:
            continue
        cash -= r["cost"]
        open_pos.append((r["T"], r["cost"], r["payoff"]))
        keep.append(r)
    return pl.DataFrame(keep, schema=f.schema)


def to_ledger(f: pl.DataFrame, strategy: str, variant: str, split: str) -> pl.DataFrame:
    return f.select(pl.lit(strategy).alias("strategy"), pl.lit(variant).alias("variant"), pl.lit(split).alias("split"),
                    pl.concat_str("ticker", pl.col("t").cast(pl.String), separator="@").alias("struct_id"),
                    pl.col("event").alias("event_ticker"), "ticker", "side", "n", pl.col("t").alias("t_in"),
                    pl.col("cost").alias("cost_usd"), pl.col("T").alias("settle_ts"), pl.col("payoff").alias("payoff_usd"))


def marks_from(cand: pl.DataFrame) -> pl.DataFrame:
    """Hourly YES mids for ``vol_metrics`` from this study's candles (forward-filled quotes)."""
    c = (cand.sort("ticker", "end_ts").with_columns(pl.col("bid", "ask").forward_fill().over("ticker"))
         .with_columns(pl.when(pl.col("bid").is_not_null() & pl.col("ask").is_not_null())
                       .then((pl.col("bid") + pl.col("ask")) / 2).otherwise(pl.col("p_close")).alias("yes_mid"))
         .filter(pl.col("yes_mid").is_not_null())
         .with_columns(pl.col("end_ts").dt.truncate("1h").alias("h"))
         .group_by("ticker", "h").agg(pl.col("yes_mid").last(), pl.col("end_ts").max().alias("ts")))
    return c.select("ticker", "ts", "yes_mid").sort("ticker", "ts")


def vol_block(led: pl.DataFrame, marks: pl.DataFrame) -> dict[str, Any]:
    from albert.research.vol_metrics import ledger_metrics

    if led.height == 0:
        return {}
    m = ledger_metrics(led, marks)
    keys = ["sharpe", "sharpe_ci90", "p_sharpe_pos", "ann_vol_pct", "sharpe_ex_top5", "sortino", "max_dd_usd",
            "usd_per_week", "usd_per_week_ex_top5", "weeks_positive_pct", "sharpe_settle_basis", "days", "events"]
    return {k: m.get(k) for k in keys}


# ---- H1 event study: Rasmussen release -> Kalshi ladder -> RCP entry ----------------------------
def ras_history(cache: Path = CACHE / "rasmussen") -> pl.DataFrame:
    """Newest cached copy of Rasmussen's public history table (release date -> approve)."""
    from albert.research.approval import parse_rasmussen

    files = sorted(cache.glob("trump_approval_index_history_second_term_fetched_*.html.gz"))
    return parse_rasmussen(gzip.decompress(files[-1].read_bytes()).decode("utf-8", "replace"))


def ladder_mean_path(lg: pl.DataFrame, cand: pl.DataFrame, event: str, times: list[datetime]) -> list[float | None]:
    """Market-implied mean of the settled value (tenths) at each time: sum(mid x centre) / sum(mid)
    over the event's legs (tails at their boundary), from forward-filled minute quotes."""
    g = lg.filter(pl.col("event") == event).with_columns(
        pl.when(pl.col("lo") <= 0).then(pl.col("hi")).when(pl.col("hi") >= 1000).then(pl.col("lo"))
        .otherwise((pl.col("lo") + pl.col("hi")) / 2).alias("centre"))
    frame = g.select("ticker", "centre").join(pl.DataFrame({"t": times}).with_columns(pl.col("t").dt.cast_time_unit("us")),
                                              how="cross")
    q = quotes_at(cand.filter(pl.col("ticker").is_in(g["ticker"].to_list())), frame.select("ticker", "t"))
    q = q.join(frame, on=["ticker", "t"]).with_columns(((pl.col("bid").fill_null(0) + pl.col("ask").fill_null(1)) / 2).alias("mid"))
    agg = q.group_by("t").agg(((pl.col("mid") * pl.col("centre")).sum() / pl.col("mid").sum()).alias("m"),
                              pl.len().alias("k")).sort("t")
    got = dict(zip(agg["t"].to_list(), agg["m"].to_list(), strict=True))
    return [got.get(t) for t in times]


def ras_event_study(out: Path = OUT, split: str = "dev") -> pl.DataFrame:
    """Per weekday KXTRUMPAPPROVE event whose Rasmussen release was entered by RCP on that day:
    RCP entry time ``e`` (the row's ``updated`` stamp), Rasmussen's change vs its previous release,
    the homepage publication bracket when known, and the market-implied mean (tenths) at e-180,
    e-60, e-5 and e+30 minutes."""
    mk = pl.read_parquet(out / "markets.parquet")
    lg = legs(mk, ["KXTRUMPAPPROVE"])
    cand = load_candles(1, lg["ticker"].to_list(), out)
    v = pl.read_parquet(out / "rcp_versions.parquet").filter(pl.col("fam") == "Rasmussen Reports")
    entries = v.group_by("id").agg(pl.col("known_at").min().alias("e"), pl.col("end").first(), pl.col("approve").first())
    ras = ras_history().sort("date").with_columns(pl.col("approve").shift(1).alias("prev_approve"))
    rel = pl.read_parquet(out / "ras_release.parquet") if (out / "ras_release.parquet").exists() else None
    vals = event_values(lg)
    rows = []
    for r in entries.sort("e").iter_rows(named=True):
        e = r["e"]
        d_et = e.astimezone(ET).date()
        ev = vals.filter(pl.col("T").dt.convert_time_zone("America/New_York").dt.date() == d_et)
        if ev.height == 0 or e >= ev["T"][0]:
            continue
        rd = ras.filter(pl.col("date") == r["end"] + timedelta(days=1))
        times = [e + timedelta(minutes=m) for m in (-180, -60, -5, 30)]
        path = ladder_mean_path(lg, cand, ev["event"][0], times)
        bracket_hi = None
        if rel is not None:
            b = rel.filter(pl.col("date") == d_et)
            bracket_hi = b["hi"][0] if b.height else None
        rows.append({"event": ev["event"][0], "e": e, "T": ev["T"][0], "ras_new": rd["approve"][0] if rd.height else r["approve"],
                     "ras_prev": rd["prev_approve"][0] if rd.height else None, "home_hi": bracket_hi,
                     "m_m180": path[0], "m_m60": path[1], "m_m5": path[2], "m_p30": path[3],
                     "settled": ev["value"][0]})
    df = pl.DataFrame(rows).filter((pl.col("T") < DEV_END) if split == "dev" else (pl.col("T") >= DEV_END))
    df = df.with_columns((pl.col("ras_new") - pl.col("ras_prev")).alias("d_ras"),
                                         (pl.col("m_m5") - pl.col("m_m180")).alias("pre_move"),
                                         (pl.col("m_p30") - pl.col("m_m5")).alias("entry_move"))
    _save(df, out / f"ras_event_study_{split}.parquet")
    return df


# ---- Rasmussen release times (Wayback copies of rasmussenreports.com, robots-allowed) ----------
RAS_HOME = "rasmussenreports.com/"
RAS_TRACK = "https://www.rasmussenreports.com/public_content/politics/trump_administration_second_term/prez_track_"
_MON = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def track_slug(d: date) -> str:
    """Rasmussen's daily tracker URL slug for release date ``d`` ("prez_track_nov20")."""
    return f"prez_track_{_MON[d.month - 1]}{d.day:02d}"


def home_links_tracker(html: str, d: date) -> bool | None:
    """Does a copy of Rasmussen's homepage link the tracker for ``d``? None if it links none."""
    slugs = set(re.findall(r"prez_track_([a-z]{3}\d{2})", html))
    if not slugs:
        return None
    return track_slug(d).removeprefix("prez_track_") in slugs


def _wb():
    from albert.research.approval import Fetcher

    return Fetcher(CACHE / "ras_home", min_interval=1.0)


def ras_release_times(out: Path = OUT, start: str = "20261001", end: str = "20261018", *, log=print) -> pl.DataFrame:
    """Bracket the publication time of Rasmussen's daily tracker on each weekday with dense Wayback
    copies of the homepage: ``lo`` = last copy that still links an older tracker, ``hi`` = first
    copy linking that day's tracker (a binary search over the day's copies, 00-18 UTC).
    Copies are cached forever (immutable); a copy is public at its capture time."""
    f = _wb()
    caps = [c for c in f.cdx(RAS_HOME, start, end) if c["statuscode"] == "200"]
    by_day: dict[date, list[dict[str, str]]] = {}
    for c in caps:
        ts = datetime.strptime(c["timestamp"], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        if ts.hour < 18:
            by_day.setdefault(ts.date(), []).append({**c, "ts": ts})
    rows = []
    for d, cs in sorted(by_day.items()):
        if d.weekday() >= 5 or len(cs) < 8:
            continue
        cs = sorted(cs, key=lambda c: c["ts"])

        def shows(i: int) -> bool | None:
            body = f.snapshot(cs[i]["timestamp"], cs[i]["original"])
            return None if body is None else home_links_tracker(body.decode("utf-8", "replace"), d)

        lo, hi = 0, len(cs) - 1
        s_lo, s_hi = shows(lo), shows(hi)
        if s_hi is not True or s_lo is True:
            rows.append({"date": d, "lo": None, "hi": cs[hi]["ts"] if s_hi else None, "copies": len(cs),
                         "note": "already linked at first copy" if s_lo else "not linked by 18 UTC"})
            continue
        while hi - lo > 1:
            mid = (lo + hi) // 2
            s = shows(mid)
            if s is True:
                hi = mid
            else:
                lo = mid
        rows.append({"date": d, "lo": cs[lo]["ts"], "hi": cs[hi]["ts"], "copies": len(cs), "note": ""})
        log(f"  {d}: published in ({cs[lo]['ts']:%H:%M}, {cs[hi]['ts']:%H:%M}] UTC", flush=True)
    df = pl.DataFrame(rows, schema={"date": pl.Date, "lo": pl.Datetime("us", "UTC"), "hi": pl.Datetime("us", "UTC"),
                                    "copies": pl.Int64, "note": pl.String})
    _save(df, out / "ras_release.parquet")
    return df


# ---- forward recorder ----------------------------------------------------------------------------
_RAS_HEAD = re.compile(r"for\s+(\w+day)\s+shows\s+that\s+(\d+)%.*?\((\d+)%\)\s+disapprove", re.DOTALL)
_RAS_CSV = re.compile(r'<pre id="series-total_approve\d+"[^>]*>\s*Date,Total Approve\s*(.*?)</pre>', re.DOTALL)
REC_SERIES = ("KXTRUMPAPPROVE", "KXAPRPOTUS", "KXTRUMPVH", "KXVHAPRPOTUS")
BOOK_SERIES = ("KXTRUMPAPPROVE", "KXAPRPOTUS")


def parse_ras_tracker(html: str) -> dict[str, Any]:
    """Rasmussen daily tracker page -> headline weekday / approve / disapprove and the newest row
    of its embedded ``Date,Total Approve`` series (epoch seconds of the release date)."""
    text = re.sub(r"<[^>]+>", " ", html)
    m = _RAS_HEAD.search(text)
    out: dict[str, Any] = {"weekday": m.group(1) if m else None, "approve": int(m.group(2)) if m else None,
                           "disapprove": int(m.group(3)) if m else None, "csv_last_date": None, "csv_last_approve": None}
    if c := _RAS_CSV.search(html):
        pts = re.findall(r"(\d{9,11}),(\d+)", c.group(1))
        if pts:
            ts, v = pts[-1]
            out["csv_last_date"] = datetime.fromtimestamp(int(ts), UTC).astimezone(ET).date().isoformat()
            out["csv_last_approve"] = int(v)
    return out


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")


def record(out: Path = OUT, *, books: bool = True, log=print) -> int:
    """One forward snapshot (append-only JSONL under ``forward/<UTC date>.jsonl``):

    * Rasmussen (robots-allowed): homepage -> is today's (ET) tracker linked; today's tracker URL ->
      HTTP status, headline numbers, newest series row, and the CDN ``Age`` header (seconds the copy
      had been cached, so ``fetched_at - age`` bounds when that copy was produced);
    * Kalshi (public API, uncached, raw responses kept): every open market of ``REC_SERIES`` with
      bid/ask/last/volume, plus depth-10 books of the open ``BOOK_SERIES`` legs.

    RCP itself is not fetched (DataDome captcha; never bypassed). Returns rows written."""
    import httpx

    from albert.venues.kalshi import orderbook_top

    now = datetime.now(UTC)
    today = now.astimezone(ET).date()
    rows: list[dict[str, Any]] = []
    http = httpx.Client(timeout=30, headers={"User-Agent": UA}, follow_redirects=False)
    for kind, url in (("ras_home", "https://www.rasmussenreports.com/"),
                      ("ras_track", RAS_TRACK + track_slug(today).removeprefix("prez_track_"))):
        t0 = datetime.now(UTC)
        try:
            r = http.get(url)
            body = r.text if r.status_code == 200 else ""
            row = {"kind": kind, "fetched_at": t0.isoformat(), "url": url, "status": r.status_code,
                   "age_s": num(r.headers.get("age")), "date_hdr": r.headers.get("date"),
                   "last_modified": r.headers.get("last-modified"), "et_date": today.isoformat()}
            if kind == "ras_home":
                row["linked_today"] = home_links_tracker(body, today) if body else None
            elif body:
                row |= parse_ras_tracker(body)
                raw = CACHE / "forward_raw" / f"{track_slug(today)}_{t0:%Y%m%dT%H%M%SZ}.html.gz"
                raw.parent.mkdir(parents=True, exist_ok=True)
                raw.write_bytes(gzip.compress(r.content))
        except httpx.HTTPError as error:
            row = {"kind": kind, "fetched_at": t0.isoformat(), "url": url, "error": str(error)[:200]}
        rows.append(row)
    api = _api(record_live=False)       # one raw file per poll would flood data/raw; the JSONL keeps the books
    for series in REC_SERIES:
        t0 = datetime.now(UTC)
        try:
            page = api.client.get("/markets", {"series_ticker": series, "status": "open", "limit": 200}, cache=False)
        except Exception as error:  # noqa: BLE001 - a failed series must not stop the snapshot
            rows.append({"kind": "kalshi_error", "fetched_at": t0.isoformat(), "series": series, "error": str(error)[:200]})
            continue
        for m in page.payload.get("markets") or []:
            rows.append({"kind": "kalshi_market", "fetched_at": t0.isoformat(), "series": series,
                         "ticker": m.get("ticker"), "event": m.get("event_ticker"), "sub": m.get("yes_sub_title"),
                         "close_time": m.get("close_time"), "yes_bid": num(_field(m, "yes_bid")),
                         "yes_ask": num(_field(m, "yes_ask")), "last": num(_field(m, "last_price")),
                         "volume": num(_field(m, "volume")), "oi": num(_field(m, "open_interest"))})
            if books and series in BOOK_SERIES:
                t1 = datetime.now(UTC)
                try:
                    ob = api.orderbook(m["ticker"], depth=10).payload
                except Exception as error:  # noqa: BLE001
                    rows.append({"kind": "book_error", "fetched_at": t1.isoformat(), "ticker": m.get("ticker"),
                                 "error": str(error)[:200]})
                    continue
                book = ob.get("orderbook_fp") or ob.get("orderbook") or {}
                rows.append({"kind": "book", "fetched_at": t1.isoformat(), "ticker": m.get("ticker"),
                             **orderbook_top(ob), "yes_levels": book.get("yes_dollars") or book.get("yes"),
                             "no_levels": book.get("no_dollars") or book.get("no")})
    _append_jsonl(out / "forward" / f"{now:%Y-%m-%d}.jsonl", rows)
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    log(f"record {now:%Y-%m-%dT%H:%M:%SZ}: {len(rows)} rows {kinds}")
    return len(rows)


def forward_summary(out: Path = OUT) -> dict[str, pl.DataFrame]:
    """Read the forward snapshots: per ET date, the bracket of Rasmussen's release (last snapshot
    without / first with that day's tracker), and per snapshot the KXTRUMPAPPROVE ladder's implied
    mean (tenths) from top-of-book mids, so the market's repricing can be timed against the release
    (and, once Wayback copies exist, against RCP's entry stamp)."""
    rows = []
    for f in sorted((out / "forward").glob("*.jsonl")):
        rows.extend(json.loads(line) for line in f.read_text().splitlines() if line.strip())
    ras = [r for r in rows if r["kind"] == "ras_track"]
    rel = []
    for d in sorted({r["et_date"] for r in ras}):
        rs = sorted((r for r in ras if r["et_date"] == d), key=lambda r: r["fetched_at"])
        seen = [r for r in rs if r.get("status") == 200 and r.get("csv_last_date") == d]
        first = seen[0]["fetched_at"] if seen else None
        before = [r["fetched_at"] for r in rs if first is None or r["fetched_at"] < first]
        rel.append({"et_date": d, "last_missing": before[-1] if before else None, "first_seen": first,
                    "approve": seen[0]["approve"] if seen else None, "snapshots": len(rs)})
    mk = [r for r in rows if r["kind"] == "kalshi_market" and r["series"] == "KXTRUMPAPPROVE"]
    lad = []
    for r in mk:
        b = bracket(r.get("sub"))
        if b is None or r.get("yes_bid") is None or r.get("yes_ask") is None:
            continue
        c = b[1] if b[0] <= 0 else b[0] if b[1] >= 1000 else (b[0] + b[1]) / 2
        lad.append({"fetched_at": r["fetched_at"], "event": r["event"], "centre": c,
                    "mid": (r["yes_bid"] + r["yes_ask"]) / 2})
    lad_df = pl.DataFrame(lad)
    if lad_df.height:
        lad_df = (lad_df.group_by("fetched_at", "event")
                  .agg(((pl.col("mid") * pl.col("centre")).sum() / pl.col("mid").sum()).alias("implied_mean"),
                       pl.col("mid").max().alias("top_mid")).sort("event", "fetched_at"))
    return {"release": pl.DataFrame(rel), "ladder": lad_df}


def paper_fills(out: Path = OUT, lead_h: float = 12.0, window: timedelta = timedelta(minutes=30)) -> pl.DataFrame:
    """Forward paper test of the frozen primary from the recorder's depth-10 books: for each open
    KXTRUMPAPPROVE event, the first snapshot in [T - lead, T - lead + window]; every leg with
    5c < mid <= 15c and spread <= 10c gets a NO order of floor(2 / (1 - bid)) contracts at 1 - bid,
    limited to the displayed YES-bid size, <= $5 per event. Outcomes come from ``markets.parquet``
    (re-run ``discover`` after settlement); unsettled legs keep a null payoff."""
    rows = []
    for f in sorted((out / "forward").glob("*.jsonl")):
        rows.extend(r for r in (json.loads(x) for x in f.read_text().splitlines() if x.strip())
                    if r["kind"] == "book" and str(r.get("ticker", "")).startswith("KXTRUMPAPPROVE-"))
    if not rows:
        return pl.DataFrame()
    mk = pl.read_parquet(out / "markets.parquet").select("ticker", "event_ticker", "close_time", "result")
    b = pl.DataFrame([{k: r.get(k) for k in ("fetched_at", "ticker", "yes_bid", "yes_bid_size", "yes_ask")} for r in rows])
    b = (b.with_columns(pl.col("fetched_at").str.to_datetime(time_zone="UTC"))
         .join(mk, on="ticker", how="inner")
         .with_columns(pl.col("close_time").map_elements(settle_instant, return_dtype=pl.Datetime("us", "UTC")).alias("T")))
    t0 = pl.col("T") - pl.duration(seconds=int(lead_h * 3600))
    b = b.filter((pl.col("fetched_at") >= t0) & (pl.col("fetched_at") <= t0 + window))
    first = b.group_by("event_ticker").agg(pl.col("fetched_at").min().alias("snap"))
    b = b.join(first, on="event_ticker").filter(pl.col("fetched_at") - pl.col("snap") < timedelta(minutes=2))
    b = b.with_columns(((pl.col("yes_bid") + pl.col("yes_ask")) / 2).alias("mid"))
    sel = b.filter((pl.col("mid") > 0.05) & (pl.col("mid") <= 0.15) & (pl.col("yes_ask") - pl.col("yes_bid") <= 0.10))
    out_rows, spent = [], {}
    for r in sel.sort("fetched_at").iter_rows(named=True):
        p = 1 - r["yes_bid"]
        n = min(contracts(p, 2.0), int(r["yes_bid_size"] or 0))
        if n < 1:
            continue
        fee = order_fee(n, p)
        if spent.get(r["event_ticker"], 0) + n * p + fee > 5:
            continue
        spent[r["event_ticker"]] = spent.get(r["event_ticker"], 0) + n * p + fee
        pay = None if r["result"] not in ("yes", "no") else float(n) * (r["result"] == "no")
        out_rows.append({"event": r["event_ticker"], "ticker": r["ticker"], "t": r["fetched_at"], "T": r["T"], "n": n,
                         "price": p, "fee": fee, "cost": n * p + fee, "payoff": pay, "bid_size": r["yes_bid_size"]})
    df = pl.DataFrame(out_rows)
    if df.height:
        _save(df, out / "paper_fills.parquet")
    return df


# ---- CLI ---------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m albert.research.approval2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("discover")
    c = sub.add_parser("candles")
    c.add_argument("--parse", action="store_true", help="rebuild parquet from the cache (run under rjob)")
    c.add_argument("--series", nargs="*", help="limit the plan to these series")
    sub.add_parser("ras-release")
    sub.add_parser("screen", help="dev screen of every rule cell (dev settlements only)")
    b = sub.add_parser("base-rates")
    b.add_argument("--split", default="dev")
    sub.add_parser("ras-study")
    sub.add_parser("dev", help="base rates (dev) + dev screen + Rasmussen event study, one process")
    sub.add_parser("holdout", help="score the rules frozen in primary.json (dev and holdout) + ledgers")
    sub.add_parser("forward-summary", help="release brackets + ladder implied mean from the forward snapshots")
    sub.add_parser("paper", help="forward paper fills of the frozen primary from recorded books")
    sub.add_parser("post", help="after freezing: holdout scores, Friday gaps (H5), holdout event study, holdout base rates")
    r = sub.add_parser("record", help="one forward snapshot (Rasmussen + Kalshi quotes/books)")
    r.add_argument("--no-books", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "discover":
        discover()
    elif a.cmd == "candles":
        plan = [p for p in CANDLE_PLAN if not a.series or p[0] in a.series]
        fetch_candles(plan=plan, parse=a.parse)
    elif a.cmd == "ras-release":
        print(ras_release_times())
    elif a.cmd == "record":
        record(books=not a.no_books)
    elif a.cmd == "screen":
        pl.Config.set_tbl_rows(200)
        pl.Config.set_tbl_width_chars(220)
        print(screen().select("series", "cell", "fills", "events", "c_per_contract", "t_event", "pnl_per_dollar", "pct_pos_weeks", "win_rate").filter(pl.col("cell").str.contains("H3")))
    elif a.cmd == "base-rates":
        res = base_rates(split=a.split)
        (OUT / f"base_rates_{a.split}.json").write_text(json.dumps(res, indent=1, default=str))
        for k, v in res.items():
            print(k)
            print(pl.DataFrame(v))
    elif a.cmd == "dev":
        pl.Config.set_tbl_rows(200)
        pl.Config.set_tbl_width_chars(220)
        res = base_rates(split="dev")
        (OUT / "base_rates_dev.json").write_text(json.dumps(res, indent=1, default=str))
        for k, v in res.items():
            print(k)
            print(pl.DataFrame(v))
        print(screen().select("series", "cell", "fills", "events", "c_per_contract", "t_event", "pnl_per_dollar",
                              "pct_pos_weeks"))
        print(ras_event_study())
    elif a.cmd == "forward-summary":
        res = forward_summary()
        for k, v in res.items():
            print(k)
            print(v)
    elif a.cmd == "paper":
        print(paper_fills())
    elif a.cmd == "post":
        pl.Config.set_tbl_rows(100)
        pl.Config.set_tbl_width_chars(220)
        print(json.dumps(score_frozen(), indent=1, default=str))
        g = friday_gaps()
        print(g.group_by("lead_h", (pl.col("T") < DEV_END).alias("dev")).agg(
            pl.len(), pl.col("event").n_unique().alias("fridays"), pl.col("gap").abs().mean().alias("mean_abs_gap"),
            pl.col("gap").abs().quantile(0.9).alias("p90_abs_gap"),
            ((pl.col("y") - pl.col("w_mid")) * pl.col("gap").sign()).mean().alias("edge_toward_daily")).sort("dev", "lead_h"))
        print(ras_event_study(split="holdout"))
        res = base_rates(split="holdout")
        (OUT / "base_rates_holdout.json").write_text(json.dumps(res, indent=1, default=str))
    elif a.cmd == "holdout":
        print(json.dumps(score_frozen(), indent=1, default=str))
    elif a.cmd == "ras-study":
        pl.Config.set_tbl_rows(200)
        pl.Config.set_tbl_width_chars(220)
        print(ras_event_study())


if __name__ == "__main__":
    main()
