"""Deadline-hazard study: "will X happen by date T" contracts on Kalshi (non-financial).

Builds on ``pm_hazard`` (universe, deadline parsing, classification, hourly candles; read only, not
edited) and ``vol_theta`` (maker theta harvest). New here (hypotheses in
``data/research/deadline_hazard/hypotheses.md``, written before outcome data):

* H1/H2 the status-quo (deadline) premium measured as a taker NO at fixed leads, per $ and per
  $-day after fees, and its calendar-time trend;
* H3 roll demand on the next leg of a deadline strip after the near leg expires NO;
* H4 stale far legs after a front-leg move;
* H5 static term-structure arbitrage (hourly history plus a live L2 snapshot of open ladders);
* H6 question-type base rates.

Steps (``python -m albert.research.deadline_hazard <step>``):

* ``scan``: settled markets closing on/after 2026-10-01 of every candidate series (pm_hazard's
  hazard series plus non-financial deadline-ish series in ``kalshi_series``), public
  ``/markets?series_ticker=&status=settled``; raw cache under ``data/cache/deadline_hazard/raw``.
* ``universe``: pm_hazard's ``universe_all`` plus the scan, parsed with pm_hazard's functions.
* ``candles``: hourly candles for in-scope markets not covered by pm_hazard / vol_theta caches.
* ``panel`` / ``analyze``: decision panel and tests; writes ``data/research/deadline_hazard/``.
* ``live``: one L2 snapshot of every open multi-deadline strip (executable violations + depth).

Dev = decisions before 2026-10-01 (deadline T before 2026-10-01); holdout = decisions
2026-10-01 .. 2026-11-22 with T <= 2026-11-22. Markets are selected by T (outcome- and
volume-blind); quotes come from candles that closed strictly before the decision.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.research import pm_hazard as H

SLUG = "deadline_hazard"
OUT = Path("data/research") / SLUG
CACHE = Path("data/cache") / SLUG
SCAN_FROM = datetime(2026, 10, 1, tzinfo=UTC)        # holdout markets were open after this
HOLDOUT = datetime(2026, 10, 1, 4, 0, tzinfo=UTC)     # dev: decisions and T before; holdout after
HOLDOUT_END = datetime(2026, 11, 22, 5, 0, tzinfo=UTC)  # latest deadline scored (Nov 22 00:00 ET)
DEV_START = H.START
LOOKBACK = H.LOOKBACK
LEADS_D = (1, 3, 7, 14, 30)                          # H1 decision leads (days before T)
PRICE_BANDS = ((0.02, 0.10, "02-10"), (0.10, 0.30, "10-30"), (0.30, 0.60, "30-60"), (0.60, 0.90, "60-90"))
MAX_SPREAD = 0.10
ORDER_USD = 1.0     # research order size for per-contract stats (fee rounding at this size)


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


def _api(min_interval: float = 0.5, retries: int = 12):
    """Public, unauthenticated Kalshi client with this study's own raw cache."""
    import random
    import time

    from albert.config import Settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    def nap(x: float) -> None:
        time.sleep(min(x, 30.0) * (0.75 + 0.5 * random.random()))
    s = Settings()
    return k.Kalshi(CachedClient("kalshi", s.kalshi_base, CACHE / "raw", min_interval=max(min_interval, 0.5),
                                 max_retries=retries, sleep=nap))


# ---- scan ----------------------------------------------------------------------------------------
def scan_plan() -> pl.DataFrame:
    """Candidate series: every series with a hazard-typed market in pm_hazard's universe, plus
    non-financial one-off/custom/weekly/monthly/annual series whose title has a deadline-ish word
    (the same SERIES_HINT pm_hazard used), from the current ``kalshi_series`` table."""
    from albert.categories import kalshi_series_is_financial

    u = pl.read_parquet(Path("data/research/pm_hazard/universe_all.parquet"), columns=["series_ticker", "kind"])
    hz = set(u.filter(pl.col("kind").is_in(H.HAZARD_TYPES))["series_ticker"])
    ser = pl.read_parquet(Path("data/tables/kalshi_series.parquet"))
    cand = set()
    for r in ser.iter_rows(named=True):
        if kalshi_series_is_financial(r["series_ticker"], r["category"], r["title"]):
            continue
        if r["frequency"] in ("one_off", "custom", "weekly", "monthly", "annual") and H.SERIES_HINT.search(r["title"] or ""):
            cand.add(r["series_ticker"])
    allser = sorted(hz | cand)
    return pl.DataFrame({"series_ticker": allser, "hz": [x in hz for x in allser]})


def _series_info() -> dict[str, dict[str, Any]]:
    df = pl.read_parquet(Path("data/tables/kalshi_series.parquet"))
    return {r["series_ticker"]: r for r in df.iter_rows(named=True)}


def scan(limit: int | None = None) -> None:
    """Settled markets closing after SCAN_FROM of each planned series, kept when the rules carry a
    deadline phrase. Checkpointed every 100 series."""
    from albert.venues import kalshi as k

    plan = scan_plan()
    if limit:
        plan = plan.head(limit)
    path, done_path = OUT / "scan_new.parquet", OUT / "scan_new_done.parquet"
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=k.MARKET_SCHEMA)
    done = set(pl.read_parquet(done_path)["series_ticker"]) if done_path.exists() else set()
    info = _series_info()
    api = _api(0.5)
    rows: list[dict[str, Any]] = []
    todo = [r["series_ticker"] for r in plan.iter_rows(named=True) if r["series_ticker"] not in done]
    print(f"scan: {len(todo)} series to query ({plan.height} planned, {len(done)} done)", flush=True)
    for i, sr in enumerate(todo, 1):
        try:
            for page in api.series_markets(sr, "settled", min_close=SCAN_FROM, cache=True):
                rows.extend(x for x in k.market_rows(page.payload, info, "dh-live")
                            if H.DEADLINE_HINT.search(x.get("rules_primary") or ""))
        except RuntimeError as error:
            print(f"skip {sr}: {str(error)[:120]}", flush=True)
            continue
        done.add(sr)
        if i % 100 == 0 or i == len(todo):
            if rows:
                have = pl.concat([have, pl.DataFrame(rows, schema=k.MARKET_SCHEMA)]).unique("ticker", keep="last")
            rows = []
            _save(have, path)
            _save(pl.DataFrame({"series_ticker": sorted(done)}), done_path)
            print(f"  series {i}/{len(todo)}: {have.height} candidate markets", flush=True)


# ---- universe and candles -------------------------------------------------------------------------
def parse_markets(m: pl.DataFrame) -> pl.DataFrame:
    """pm_hazard's financial filters, deadline window, kind and strip key for raw market rows."""
    from albert.categories import kalshi_series_is_financial

    info = _series_info()
    fin = {r: kalshi_series_is_financial(r, (info.get(r) or {}).get("category"), (info.get(r) or {}).get("title"))
           for r in m["series_ticker"].unique().to_list()}
    m = m.filter(pl.col("series_ticker").map_elements(lambda x: not fin.get(x, True), return_dtype=pl.Boolean))
    m = m.filter(~pl.col("title").fill_null("").map_elements(
        lambda t: kalshi_series_is_financial(None, "Politics", t), return_dtype=pl.Boolean))
    recs = []
    for r in m.select("ticker", "category", "title", "rules_primary", "series_ticker").iter_rows(named=True):
        try:
            st, T, kw = H.parse_window(r["rules_primary"])
        except (ValueError, KeyError):
            st, T, kw = None, None, None
        recs.append({"ticker": r["ticker"], "w_start": st, "T": T, "kw": kw,
                     "kind": H.classify(r["category"], r["title"], r["rules_primary"]),
                     "skey": H.strip_key(r["series_ticker"], r["rules_primary"])})
    meta = pl.DataFrame(recs, schema={"ticker": pl.String, "w_start": pl.Datetime("us", "UTC"),
                                      "T": pl.Datetime("us", "UTC"), "kw": pl.String, "kind": pl.String,
                                      "skey": pl.String})
    return m.join(meta, on="ticker").with_columns(
        (pl.col("result") == "yes").cast(pl.Int8).alias("y"), pl.col("fee_multiplier").fill_null(1.0))


def build_universe() -> pl.DataFrame:
    """pm_hazard's parsed universe plus this study's scan (the scan's row wins for a ticker in both:
    it is the settled state). In scope: hazard kinds, settled yes/no, START < T <= HOLDOUT_END."""
    from albert.venues import kalshi as k

    old = pl.read_parquet(Path("data/research/pm_hazard/universe_all.parquet"))
    frames = []
    if (OUT / "scan_new.parquet").exists():
        new = pl.read_parquet(OUT / "scan_new.parquet").select(list(k.MARKET_SCHEMA))
        info = _series_info()
        new = new.with_columns(
            pl.col("series_ticker").replace_strict({x: (info.get(x) or {}).get("category") for x in new["series_ticker"].unique()},
                                                   default=None, return_dtype=pl.String).fill_null(pl.col("category")).alias("category"))
        frames.append(parse_markets(new))
    frames.append(old)
    u = pl.concat(frames, how="diagonal_relaxed").unique("ticker", keep="first", maintain_order=True)
    u = u.filter(pl.col("kind").is_in(H.HAZARD_TYPES) & pl.col("T").is_not_null()
                 & (pl.col("T") <= HOLDOUT_END) & (pl.col("T") > DEV_START + timedelta(hours=1))
                 & pl.col("result").is_in(["yes", "no"])
                 & (pl.col("close_time") > DEV_START) & (pl.col("open_time") < pl.col("T")))
    u = u.with_columns(pl.col("settlement_ts").fill_null(pl.col("close_time")),
                       pl.col("fee_type").fill_null("quadratic"))
    _save(u, OUT / "universe.parquet")
    print(f"universe: {u.height} markets, {u['event_ticker'].n_unique()} events; "
          f"T >= holdout start: {u.filter(pl.col('T') >= HOLDOUT).height}")
    return u


CANDLE_SRC = (Path("data/research/pm_hazard/candles.parquet"), Path("data/research/vol_theta/candles_ext.parquet"))
CANDLE_NEW = CACHE / "candles" / "candles_new.parquet"
CANDLE_COLS = ["ticker", "end_ts", "yes_bid_close", "yes_ask_close", "price_close", "volume", "open_interest"]


def load_candles(tickers: list[str] | None = None) -> pl.DataFrame:
    frames = [pl.scan_parquet(p).select(CANDLE_COLS) for p in (*CANDLE_SRC, CANDLE_NEW) if p.exists()]
    c = pl.concat(frames, how="vertical_relaxed")
    if tickers is not None:
        c = c.filter(pl.col("ticker").is_in(tickers))
    return c.collect().unique(["ticker", "end_ts"]).sort("ticker", "end_ts")


def fetch_candles(u: pl.DataFrame) -> None:
    """Hourly candles over [max(open, T - LOOKBACK, START - 1 d), min(close, T) + 1 h] for in-scope
    markets that no cached source covers. Live endpoint after Kalshi's historical cutoff."""
    from albert.venues import kalshi as k

    have = set()
    for p in (*CANDLE_SRC, CANDLE_NEW):
        if p.exists():
            have |= set(pl.scan_parquet(p).select("ticker").unique().collect()["ticker"])
    todo = u.filter(~pl.col("ticker").is_in(list(have)))
    print(f"candles: {todo.height} markets to fetch ({u.height - todo.height} cached)", flush=True)
    if not todo.height:
        return
    api = _api(0.5, retries=60)
    cutoff = api.cutoff().get("market_settled_ts")
    rows: list[dict[str, Any]] = []
    old = pl.read_parquet(CANDLE_NEW) if CANDLE_NEW.exists() else None

    def flush() -> None:
        nonlocal old, rows
        if not rows:
            return
        new = pl.DataFrame(rows, schema={"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"),
                                         "yes_bid_close": pl.Float64, "yes_ask_close": pl.Float64,
                                         "price_close": pl.Float64, "volume": pl.Float64, "open_interest": pl.Float64})
        old = pl.concat([f for f in (old, new) if f is not None]).unique(["ticker", "end_ts"])
        _save(old, CANDLE_NEW)
        rows = []
    for i, r in enumerate(todo.iter_rows(named=True), 1):
        lo = max(r["open_time"], r["T"] - LOOKBACK, DEV_START - timedelta(days=1))
        hi = min(r["close_time"], r["T"]) + timedelta(hours=1)
        if lo >= hi:
            continue
        try:
            page = api.candles(r["ticker"], lo, hi, 60, bool(cutoff and r["settlement_ts"] and r["settlement_ts"] < cutoff))
        except RuntimeError as error:
            print(f"skip {r['ticker']}: {str(error)[:120]}", flush=True)
            continue
        rows.extend({c: x[c] for c in CANDLE_COLS} for x in k.candle_rows(page.payload, r["ticker"], 60))
        if i % 200 == 0:
            flush()
            print(f"  candles {i}/{todo.height}", flush=True)
    flush()


# ---- quotes and the fixed-lead panel -------------------------------------------------------------
def quotes(c: pl.DataFrame) -> pl.DataFrame:
    """Two-sided hourly quotes (0 < bid < ask < 1). Kalshi emits a candle whenever the book changes
    (68% have zero volume), so a missing hour means an unchanged quote: carry it forward."""
    return (c.filter(pl.col("yes_bid_close").is_not_null() & pl.col("yes_ask_close").is_not_null()
                     & (pl.col("yes_bid_close") > 0) & (pl.col("yes_ask_close") < 1)
                     & (pl.col("yes_ask_close") > pl.col("yes_bid_close")))
            .select("ticker", pl.col("end_ts").alias("q_ts"), pl.col("yes_bid_close").alias("bid"),
                    pl.col("yes_ask_close").alias("ask")).sort("ticker", "q_ts"))


def quote_at(df: pl.DataFrame, q: pl.DataFrame, t_col: str = "t", max_age_h: float = 72.0,
             ticker_col: str = "ticker", suffix: str = "") -> pl.DataFrame:
    """The last quote that ended strictly before ``t_col`` (at most ``max_age_h`` old)."""
    qq = q.rename({"q_ts": f"q_ts{suffix}", "bid": f"bid{suffix}", "ask": f"ask{suffix}"})
    if ticker_col != "ticker":
        qq = qq.rename({"ticker": ticker_col})
    x = df.with_columns((pl.col(t_col) - timedelta(seconds=1)).alias("_tq")).sort("_tq")
    x = x.join_asof(qq, left_on="_tq", right_on=f"q_ts{suffix}", by=ticker_col, strategy="backward",
                    check_sortedness=False).drop("_tq")
    stale = (pl.col(t_col) - pl.col(f"q_ts{suffix}")) > timedelta(hours=max_age_h)
    return x.with_columns([pl.when(stale).then(None).otherwise(pl.col(c)).alias(c)
                           for c in (f"bid{suffix}", f"ask{suffix}")])


def order_n(price: Any, usd: float = 5.0) -> np.ndarray:
    """Contracts a $usd order buys at ``price`` (at least 1)."""
    p = np.asarray(price, dtype=float)
    return np.maximum(np.floor(usd / np.maximum(p, 0.01) + 1e-9), 1)


def taker_fee_pc(price: Any, n: Any, mult: Any) -> np.ndarray:
    """Kalshi taker fee per contract: 0.07 x mult x n x P(1-P), rounded up to the cent per order."""
    P, N, M = (np.asarray(x, dtype=float) for x in (price, n, mult))
    raw = 0.07 * M * N * P * (1 - P)
    return np.ceil(np.round(raw * 100, 9)) / 100 / N


QTYPES = (   # question type from title + rules (first match wins)
    ("exit", r"\b(leave|leaves|out as|out\b|resign|step(s)? down|fired|removed|depart|ousted|replaced|impeach|retire)"),
    ("legal", r"\b(indict|arrest|charged|convicted|sentenced|pardon|plead|lawsuit|sue[sd]?|court rul)"),
    ("military", r"\b(strike|attack|invade|invasion|ceasefire|troops|military|war\b|missile|bomb|hostage)"),
    ("legislation", r"\b(pass(es)?|sign(s|ed)?|become law|bill\b|act\b|confirm(ed)?|nominat|vote|shutdown|fund)"),
    ("meeting", r"\b(meet|meeting|visit|summit|talk(s)?|call with|speak with|travel)"),
    ("release", r"\b(release|launch|announce|unveil|debut|drop|trailer|album|model|ipo|open(s)?\b)"),
)


def qtype(title: str | None, rules: str | None) -> str:
    text = f"{title or ''} {(rules or '').split('then the market resolves')[0]}"
    for name, rx in QTYPES:
        if re.search(rx, text, re.IGNORECASE):
            return name
    return "other"


def split_of(t: pl.Expr, T: pl.Expr) -> pl.Expr:
    return (pl.when((T < HOLDOUT) & (t < HOLDOUT)).then(pl.lit("dev"))
            .when((t >= HOLDOUT) & (T <= HOLDOUT_END)).then(pl.lit("holdout")).otherwise(pl.lit("straddle")))


def fixed_lead_panel(u: pl.DataFrame, q: pl.DataFrame, leads_d: tuple[int, ...] = LEADS_D) -> pl.DataFrame:
    """One row per (market, lead): decision t = T - lead, market open at t (it has not closed
    early), quote strictly before t; taker NO at 1 - bid, $5 order, fee rounded up per order;
    held to settlement."""
    base = u.select("ticker", "event_ticker", "series_ticker", "category", "kind", "skey", "T", "w_start", "open_time",
                    "close_time", "settlement_ts", "y", "fee_type", "fee_multiplier", "title", "rules_primary")
    base = base.with_columns(pl.struct("title", "rules_primary").map_elements(
        lambda r: qtype(r["title"], r["rules_primary"]), return_dtype=pl.String).alias("qtype")).drop("rules_primary")
    g = base.join(pl.DataFrame({"lead_d": list(leads_d)}), how="cross").with_columns(
        (pl.col("T") - pl.duration(days=pl.col("lead_d"))).alias("t"))
    g = g.filter((pl.col("t") >= DEV_START) & (pl.col("t") >= pl.col("open_time") + timedelta(hours=1))
                 & (pl.col("t") < pl.col("close_time")))
    g = quote_at(g, q).filter(pl.col("bid").is_not_null())
    return add_no_trade(g)


def add_no_trade(g: pl.DataFrame, side: str = "no") -> pl.DataFrame:
    """Taker cost, fee and settlement PnL of one $5 order on ``side`` at the row's quote."""
    g = g.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("ask") - pl.col("bid")).alias("spread"),
                       split_of(pl.col("t"), pl.col("T")).alias("split"))
    px = (1 - g["bid"].to_numpy()) if side == "no" else g["ask"].to_numpy()
    n = order_n(px)
    fee = taker_fee_pc(px, n, g["fee_multiplier"].to_numpy())
    won = (1 - g["y"].to_numpy()) if side == "no" else g["y"].to_numpy()
    days = np.maximum((g["settlement_ts"] - g["t"]).dt.total_seconds().to_numpy() / 86400, 1 / 24)
    pnl = won - px - fee
    return g.with_columns(pl.Series("px", px), pl.Series("n", n), pl.Series("fee", fee), pl.Series("pnl", pnl),
                          pl.Series("per_usd", pnl / (px + fee)), pl.Series("days", days),
                          pl.Series("per_usd_day", pnl / (px + fee) / days))


def _band(mid: pl.Expr, bands: tuple = PRICE_BANDS) -> pl.Expr:
    out = pl.lit(None, dtype=pl.String)
    for lo, hi, lab in reversed(bands):
        out = pl.when((mid >= lo) & (mid < hi)).then(pl.lit(lab)).otherwise(out)
    return out


def cell_stats(x: pl.DataFrame, col: str = "pnl") -> dict[str, Any]:
    """Mean per contract (c), t clustered by event, per $, per $-day, n and events."""
    from albert.research.counts import cluster_se

    if x.height < 2:
        return {"n": x.height, "events": x["event_ticker"].n_unique() if x.height else 0}
    v = x[col].to_numpy().astype(float)
    se = cluster_se(v, x["event_ticker"].to_numpy())
    cost = (x["px"] + x["fee"]).to_numpy()
    dd = (x["days"] * (x["px"] + x["fee"])).to_numpy()
    return {"n": x.height, "events": x["event_ticker"].n_unique(), "c": round(100 * v.mean(), 2),
            "t": round(v.mean() / se, 2) if se and se > 0 else None,
            "per_usd_pct": round(100 * v.sum() / cost.sum(), 2),
            "per_usd_day_bp": round(1e4 * v.sum() / dd.sum(), 2),
            "mean_days": round(float(x["days"].mean()), 1), "mean_px": round(float(x["px"].mean()), 3),
            "yes_rate": round(float(x["y"].mean()), 3), "mean_mid": round(float(x["mid"].mean()), 3)}


# ---- H1 / H2 / H6: taker NO at fixed leads -------------------------------------------------------
def kind_group(kind: pl.Expr) -> pl.Expr:
    return pl.when(kind == "event").then(pl.lit("event")).otherwise(pl.lit("sched"))   # mention/elim/count


def h1_cells(pn: pl.DataFrame, split: str) -> pl.DataFrame:
    """Taker NO by lead x YES price band x kind group (event vs scheduled-type)."""
    x = pn.filter((pl.col("split") == split) & (pl.col("spread") <= MAX_SPREAD)).with_columns(
        _band(pl.col("mid")).alias("band"), kind_group(pl.col("kind")).alias("kg")).filter(pl.col("band").is_not_null())
    rows = []
    for (lead, band, kg), g in x.group_by("lead_d", "band", "kg"):
        rows.append({"family": "H1", "lead_d": lead, "band": band, "kg": kg, **cell_stats(g)})
    return pl.DataFrame(rows).sort("kg", "band", "lead_d")


def h6_cells(pn: pl.DataFrame, split: str) -> pl.DataFrame:
    """Event-kind taker NO by question type x lead group (<= 3 d vs >= 7 d), YES 2-60c."""
    x = pn.filter((pl.col("split") == split) & (pl.col("spread") <= MAX_SPREAD) & (pl.col("kind") == "event")
                  & (pl.col("mid") >= 0.02) & (pl.col("mid") < 0.60)).with_columns(
        pl.when(pl.col("lead_d") <= 3).then(pl.lit("short")).otherwise(pl.lit("long")).alias("lg"))
    rows = []
    for (qt, lg), g in x.group_by("qtype", "lg"):
        rows.append({"family": "H6", "qtype": qt, "lg": lg, **cell_stats(g)})
    return pl.DataFrame(rows).sort("qtype", "lg")


def h2_monthly(pn: pl.DataFrame, splits: tuple[str, ...]) -> pl.DataFrame:
    """Calendar-time trend: YES residual (y - mid) and taker-NO c by decision month and lead group."""
    from albert.research.counts import cluster_se

    x = pn.filter(pl.col("split").is_in(list(splits)) & (pl.col("spread") <= MAX_SPREAD)
                  & (pl.col("mid") >= 0.02) & (pl.col("mid") < 0.90)).with_columns(
        pl.col("t").dt.strftime("%Y-%m").alias("month"),
        pl.when(pl.col("lead_d") <= 3).then(pl.lit("short")).otherwise(pl.lit("long")).alias("lg"),
        (pl.col("y") - pl.col("mid")).alias("resid"))
    rows = []
    for (mo, lg), g in x.group_by("month", "lg"):
        v = g["resid"].to_numpy()
        se = cluster_se(v, g["event_ticker"].to_numpy()) if g.height > 1 else float("nan")
        rows.append({"month": mo, "lg": lg, "n": g.height, "events": g["event_ticker"].n_unique(),
                     "resid_c": round(100 * v.mean(), 2), "t_resid": round(v.mean() / se, 2) if se > 0 else None,
                     **{f"no_{k}": v2 for k, v2 in cell_stats(g).items() if k in ("c", "t", "per_usd_pct")}})
    return pl.DataFrame(rows).sort("lg", "month")


# ---- strips: H3 roll, H4 stale far legs, H5 static arbitrage --------------------------------------
def strip_pairs(u: pl.DataFrame) -> pl.DataFrame:
    """Adjacent legs (T1 < T2, the nearest later deadline) of one question (same strip key: rules
    identical once dates are removed), single-occurrence kinds (event, mention). Unlike pm_hazard,
    a far leg listed after the near one is kept: every decision below is taken after both legs are
    open and the near leg is unresolved, when the remaining windows are nested."""
    legs = u.filter(pl.col("kind").is_in(["event", "mention"])).select(
        "ticker", "event_ticker", "series_ticker", "skey", "T", "open_time", "close_time", "settlement_ts", "y",
        "fee_multiplier", "kind", "category")
    pairs = legs.join(legs.select("ticker", "skey", "T", "open_time", "close_time", "settlement_ts", "y", "fee_multiplier",
                                  "event_ticker"), on="skey", suffix="_2").filter(pl.col("T_2") > pl.col("T"))
    nearest = pairs.group_by("ticker").agg(pl.col("T_2").min().alias("Tn"))
    pairs = pairs.join(nearest, on="ticker").filter(pl.col("T_2") == pl.col("Tn")).drop("Tn")
    return pairs.unique(["ticker", "ticker_2"])


def h3_roll(pairs: pl.DataFrame, q: pl.DataFrame, offsets_h: tuple[int, ...] = (-24, 6, 24, 72)) -> pl.DataFrame:
    """Far leg of each pair at T1 + offset. Post-expiry rows (offset > 0) need the near leg to have
    reached T1 unresolved (its last quote before T1 <= 10c and it closed no earlier than T1 - 2 h),
    all observable at the decision. Taker NO on the far leg, held to settlement."""
    x = pairs.filter((pl.col("T_2") - pl.col("T")) >= timedelta(days=2))
    x = x.join(pl.DataFrame({"off_h": list(offsets_h)}), how="cross").with_columns(
        (pl.col("T") + pl.duration(hours=pl.col("off_h"))).alias("t"))
    x = x.filter((pl.col("t") < pl.col("close_time_2")) & (pl.col("t") >= pl.col("open_time_2") + timedelta(hours=1))
                 & (pl.col("t") < pl.col("T_2")))
    # near leg's last quote before T1 (observable at T1)
    near = x.select("ticker", "T").unique().rename({"T": "t_near"})
    near = quote_at(near, q, "t_near", max_age_h=24 * 7, suffix="_n")
    x = x.join(near.select("ticker", "t_near", "bid_n", "ask_n"), left_on=["ticker", "T"], right_on=["ticker", "t_near"],
               how="left")
    expired = ((pl.col("bid_n") + pl.col("ask_n")) / 2 <= 0.10) & (pl.col("close_time") >= pl.col("T") - timedelta(hours=2))
    x = x.filter(pl.when(pl.col("off_h") > 0).then(expired).otherwise(
        (pl.col("t") >= pl.col("open_time") + timedelta(hours=1)) & (pl.col("t") < pl.col("close_time"))))
    far = x.select(pl.col("ticker_2").alias("ticker"), "event_ticker_2", "T_2", "open_time_2", "close_time_2",
                   "settlement_ts_2", "y_2", "fee_multiplier_2", "t", "off_h", pl.col("ticker").alias("near"),
                   "event_ticker", "category", "kind")
    far = far.rename({"event_ticker_2": "far_event", "T_2": "T", "open_time_2": "open_time", "close_time_2": "close_time",
                      "settlement_ts_2": "settlement_ts", "y_2": "y", "fee_multiplier_2": "fee_multiplier"})
    far = quote_at(far, q).filter(pl.col("bid").is_not_null())
    return add_no_trade(far)


def hourly_pair_grid(pairs: pl.DataFrame, q: pl.DataFrame) -> pl.DataFrame:
    """Both legs' carried-forward quotes on an hourly grid over the hours both are open and the near
    leg's deadline is ahead: hour h uses the candles that ended at or before h."""
    w = pairs.with_columns(
        pl.max_horizontal("open_time", "open_time_2", pl.lit(DEV_START)).dt.truncate("1h").alias("a"),
        (pl.min_horizontal("close_time", "close_time_2", "T") - timedelta(hours=1)).dt.truncate("1h").alias("b"))
    w = w.filter(pl.col("b") > pl.col("a"))
    w = w.with_columns(pl.datetime_ranges("a", "b", "1h").alias("h")).explode("h")
    w = w.select("ticker", "ticker_2", "event_ticker", "skey", "T", "T_2", "y", "y_2", "settlement_ts", "settlement_ts_2",
                 "fee_multiplier", "fee_multiplier_2", "close_time", "close_time_2", "category", "kind",
                 (pl.col("h") + timedelta(seconds=1)).alias("hq"))
    w = quote_at(w, q, "hq", max_age_h=72, suffix="_1")
    w = quote_at(w, q, "hq", max_age_h=72, ticker_col="ticker_2", suffix="_2")
    return w.filter(pl.col("bid_1").is_not_null() & pl.col("bid_2").is_not_null()).with_columns(
        (pl.col("hq") - timedelta(seconds=1)).alias("h")).drop("hq").sort("ticker", "ticker_2", "h")


def h4_signals(gr: pl.DataFrame, move: float = 0.05, still: float = 0.01) -> pl.DataFrame:
    """Front-leg moves of >= ``move`` in the last hour while the far leg moved <= ``still``: trade the
    far leg in the front leg's direction at the hour's quote (YES at the ask after an up-move, NO at
    1 - bid after a down-move); first signal per pair, direction and UTC day; held to settlement."""
    g = gr.with_columns(((pl.col("bid_1") + pl.col("ask_1")) / 2).alias("m1"), ((pl.col("bid_2") + pl.col("ask_2")) / 2).alias("m2"))
    g = g.with_columns((pl.col("m1") - pl.col("m1").shift(1).over("ticker", "ticker_2")).alias("d1"),
                       (pl.col("m2") - pl.col("m2").shift(1).over("ticker", "ticker_2")).alias("d2"),
                       (pl.col("h") - pl.col("h").shift(1).over("ticker", "ticker_2")).alias("dh"))
    for k in (1, 6, 24):
        g = g.with_columns(pl.col("m2").shift(-k).over("ticker", "ticker_2").alias(f"m2_{k}"),
                           (pl.col("h").shift(-k).over("ticker", "ticker_2") - pl.col("h")).alias(f"dh_{k}"))
    sig = g.filter((pl.col("dh") == timedelta(hours=1)) & (pl.col("d1").abs() >= move) & (pl.col("d2").abs() <= still)
                   & ((pl.col("ask_2") - pl.col("bid_2")) <= MAX_SPREAD) & (pl.col("m2") >= 0.02) & (pl.col("m2") <= 0.90)
                   & (pl.col("m1") <= 0.95))
    sig = sig.with_columns(pl.when(pl.col("d1") > 0).then(pl.lit("yes")).otherwise(pl.lit("no")).alias("side"),
                           pl.col("h").dt.date().alias("day"))
    sig = sig.sort("h").unique(["ticker", "ticker_2", "side", "day"], keep="first")
    out = []
    for side in ("yes", "no"):
        x = sig.filter(pl.col("side") == side).select(
            pl.col("ticker_2").alias("ticker"), "event_ticker", "category", "kind", pl.col("T_2").alias("T"),
            pl.col("y_2").alias("y"), pl.col("settlement_ts_2").alias("settlement_ts"),
            pl.col("fee_multiplier_2").alias("fee_multiplier"), (pl.col("h") + timedelta(minutes=1)).alias("t"),
            pl.col("bid_2").alias("bid"), pl.col("ask_2").alias("ask"), "d1", "d2", "m2", "m2_1", "m2_6", "m2_24",
            "dh_1", "dh_6", "dh_24", "side")
        if not x.height:
            continue
        x = add_no_trade(x, side)
        sgn = 1.0 if side == "yes" else -1.0
        x = x.with_columns([(sgn * (pl.col(f"m2_{k}") - pl.col("m2"))).alias(f"mk_{k}h") for k in (1, 6, 24)])
        out.append(x.with_columns(pl.lit(move).alias("move")))
    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()


def h5_arbs(gr: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Executable monotonicity violations: near YES bid > far YES ask after both taker fees ($5
    orders, rounded up per order). Profit locked per pair-contract = bid_1 - ask_2 - fees."""
    g = gr.with_columns((pl.col("bid_1") - pl.col("ask_2")).alias("gross"))
    px_y, px_n = g["ask_2"].to_numpy(), 1 - g["bid_1"].to_numpy()
    n = np.minimum(order_n(px_y), order_n(px_n))
    fee = taker_fee_pc(px_y, n, g["fee_multiplier_2"].to_numpy()) + taker_fee_pc(px_n, n, g["fee_multiplier"].to_numpy())
    g = g.with_columns(pl.Series("fees", fee), (pl.col("gross") - pl.Series(fee)).alias("net"))
    v = g.filter(pl.col("net") > 0).sort("ticker", "ticker_2", "h")
    v = v.with_columns(((pl.col("h") - pl.col("h").shift(1).over("ticker", "ticker_2")) != timedelta(hours=1))
                       .fill_null(True).cum_sum().over("ticker", "ticker_2").alias("run"))
    runs = v.group_by("ticker", "ticker_2", "run").agg(pl.len().alias("hours"), pl.col("net").first().alias("net0"),
                                                       pl.col("net").max().alias("net_max"), pl.col("h").first().alias("h0"),
                                                       pl.col("event_ticker").first(), pl.col("y").first(),
                                                       pl.col("y_2").first(), pl.col("bid_1").first(), pl.col("ask_2").first())
    summ = {"pair_hours": g.height, "pairs": g.select("ticker", "ticker_2").n_unique(),
            "viol_mid_share": round(float((g["bid_1"] + g["ask_1"] > g["bid_2"] + g["ask_2"]).mean()), 4) if g.height else None,
            "exec_hours": v.height, "exec_share": round(v.height / max(g.height, 1), 5),
            "exec_pairs": v.select("ticker", "ticker_2").n_unique() if v.height else 0,
            "runs": runs.height, "runs_ge2h": runs.filter(pl.col("hours") >= 2).height,
            "median_net0_c": round(100 * float(runs["net0"].median()), 2) if runs.height else None,
            "mean_net0_c": round(100 * float(runs["net0"].mean()), 2) if runs.height else None,
            "basis_breaks": runs.filter((pl.col("y") == 1) & (pl.col("y_2") == 0)).height}
    return runs, summ


def explore_strips(split: str = "dev") -> None:
    """Dev-only screens for H3, H4, H5."""
    u = pl.read_parquet(OUT / "universe.parquet")
    pairs = strip_pairs(u)
    c = load_candles(sorted(set(pairs["ticker"]) | set(pairs["ticker_2"])))
    q = quotes(c)
    rows = []
    r3 = h3_roll(pairs, q)
    _save(r3, OUT / "h3_rows.parquet")
    r3d = r3.filter(pl.col("split") == split)
    for (off, band), g in r3d.with_columns(_band(pl.col("mid"), ((0.02, 0.10, "02-10"), (0.10, 0.90, "10-90")))
                                         .alias("band")).filter(pl.col("band").is_not_null()).group_by("off_h", "band"):
        rows.append({"family": "H3", "cell": f"off{off}_{band}", **cell_stats(g.filter(pl.col("spread") <= MAX_SPREAD))})
    # paired drift: far mid at T1 + 6 h minus at T1 - 24 h on the same far leg
    pre = r3d.filter(pl.col("off_h") == -24).select("ticker", "near", pl.col("mid").alias("m_pre"))
    for off in (6, 24, 72):
        post = r3d.filter(pl.col("off_h") == off).select("ticker", "near", pl.col("mid").alias("m_post"), "event_ticker")
        d = pre.join(post, on=["ticker", "near"])
        if d.height:
            from albert.research.counts import cluster_se
            v = (d["m_post"] - d["m_pre"]).to_numpy()
            se = cluster_se(v, d["event_ticker"].to_numpy())
            rows.append({"family": "H3drift", "cell": f"mid(T1+{off}h)-mid(T1-24h)", "n": d.height,
                         "events": d["event_ticker"].n_unique(), "c": round(100 * v.mean(), 2),
                         "t": round(v.mean() / se, 2) if se > 0 else None})
    gr = hourly_pair_grid(pairs, q)
    gr = gr.with_columns(split_of(pl.col("h"), pl.col("T_2")).alias("split"))
    _save(gr, OUT / "pair_grid.parquet")
    grd = gr.filter(pl.col("split") == split)
    for mv in (0.05, 0.10):
        s4 = h4_signals(grd, mv)
        for side in ("yes", "no"):
            x = s4.filter(pl.col("side") == side) if s4.height else s4
            st = cell_stats(x) if x.height else {"n": 0}
            mk = {f"mk_{k}h_c": round(100 * float(x[f"mk_{k}h"].drop_nulls().mean()), 2) for k in (1, 6, 24)} if x.height else {}
            rows.append({"family": "H4", "cell": f"move{int(mv * 100)}_{side}", **st, **mk})
    runs, summ = h5_arbs(grd)
    _save(runs, OUT / f"h5_runs_{split}.parquet")
    print("H5", split, summ)
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(OUT / f"explore_{split}_strips.csv")
    with pl.Config(tbl_rows=100, tbl_cols=30, tbl_width_chars=250):
        print(f"pairs {pairs.height}, grid {gr.height}")
        print(df)


# ---- H5 live: one L2 snapshot of every open multi-deadline strip -----------------------------------
def book_levels(payload: dict[str, Any]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """(YES bids high->low, YES asks low->high) as (price $, size) from an orderbook payload; a NO
    bid at q is a YES ask at 1 - q."""
    book = payload.get("orderbook_fp") or payload.get("orderbook") or {}
    yes = sorted(((float(p), float(z)) for p, z in (book.get("yes_dollars") or book.get("yes") or [])), reverse=True)
    no = sorted(((float(p), float(z)) for p, z in (book.get("no_dollars") or book.get("no") or [])), reverse=True)
    return yes, [(round(1 - p, 4), z) for p, z in no]


def exec_arb(bids_near: list[tuple[float, float]], asks_far: list[tuple[float, float]], mult_near: float = 1.0,
             mult_far: float = 1.0) -> dict[str, float]:
    """Sell near YES (buy NO at 1 - bid) and buy far YES at the ask, level by level, while each
    marginal contract still nets > 0 after taker fees (fee at the marginal level, 1-lot rounding
    upper bound ignored). Returns contracts, net $ and the best-level gross edge."""
    i = j = 0
    bn = [list(x) for x in bids_near]
    af = [list(x) for x in asks_far]
    qty = net = 0.0
    best = (bn[0][0] - af[0][0]) if bn and af else None
    while i < len(bn) and j < len(af):
        b, a = bn[i][0], af[j][0]
        edge = b - a - 0.07 * mult_near * b * (1 - b) - 0.07 * mult_far * a * (1 - a)
        if edge <= 0:
            break
        k = min(bn[i][1], af[j][1])
        qty += k
        net += k * edge
        bn[i][1] -= k
        af[j][1] -= k
        if bn[i][1] <= 1e-9:
            i += 1
        if af[j][1] <= 1e-9:
            j += 1
    return {"contracts": qty, "net_usd": round(net, 2), "best_gross": best}


def live_snapshot() -> pl.DataFrame:
    """Open markets of every series that had a multi-deadline strip (or a hazard market) in the
    universe, grouped into strips by rules text, then both books of each adjacent pair."""
    from albert.venues import kalshi as k

    u = pl.read_parquet(OUT / "universe.parquet")
    ser = sorted(set(strip_pairs(u)["series_ticker"]) | set(scan_plan().filter(pl.col("hz"))["series_ticker"]))
    api = _api(0.5)
    info = _series_info()
    rows: list[dict[str, Any]] = []
    for sr in ser:
        try:
            for page in api.series_markets(sr, "open", cache=False):
                rows.extend(k.market_rows(page.payload, info, "dh-open"))
        except RuntimeError as error:
            print(f"skip {sr}: {str(error)[:100]}", flush=True)
    if not rows:
        return pl.DataFrame()
    m = parse_markets(pl.DataFrame(rows, schema=k.MARKET_SCHEMA)).filter(
        pl.col("kind").is_in(["event", "mention"]) & pl.col("T").is_not_null())
    m = m.with_columns(pl.col("settlement_ts").fill_null(pl.col("close_time")))
    pairs = strip_pairs(m.with_columns(pl.lit(0, dtype=pl.Int8).alias("y")))
    legs = sorted(set(pairs["ticker"]) | set(pairs["ticker_2"]))
    print(f"live: {m.height} open deadline markets, {pairs.height} adjacent pairs, {len(legs)} books", flush=True)
    books: dict[str, Any] = {}
    stamp = datetime.now(UTC)
    for tk in legs:
        try:
            books[tk] = api.orderbook(tk, depth=20).payload
        except RuntimeError as error:
            print(f"skip book {tk}: {str(error)[:100]}", flush=True)
    out = []
    for r in pairs.iter_rows(named=True):
        if r["ticker"] not in books or r["ticker_2"] not in books:
            continue
        bn, an = book_levels(books[r["ticker"]])
        bf, af = book_levels(books[r["ticker_2"]])
        ex = exec_arb(bn, af, r["fee_multiplier"] or 1.0, r["fee_multiplier_2"] or 1.0)
        out.append({"snap_ts": stamp, "near": r["ticker"], "far": r["ticker_2"], "skey": r["skey"], "T": r["T"], "T_2": r["T_2"],
                    "near_bid": bn[0][0] if bn else None, "near_ask": an[0][0] if an else None,
                    "far_bid": bf[0][0] if bf else None, "far_ask": af[0][0] if af else None,
                    "near_bid_size": bn[0][1] if bn else None, "far_ask_size": af[0][1] if af else None, **ex})
    df = pl.DataFrame(out, infer_schema_length=None)
    path = OUT / "live_snapshots.parquet"
    old = pl.read_parquet(path) if path.exists() else None
    _save(pl.concat([f for f in (old, df) if f is not None], how="diagonal_relaxed"), path)
    viol = df.filter(pl.col("contracts") > 0)
    mid_viol = df.filter((pl.col("near_bid") + pl.col("near_ask")) > (pl.col("far_bid") + pl.col("far_ask")))
    print(f"live snapshot {stamp:%Y-%m-%d %H:%M}Z: {df.height} pairs quoted, {mid_viol.height} mid violations, "
          f"{viol.height} executable after fees, {viol['contracts'].sum() if viol.height else 0} contracts, "
          f"${viol['net_usd'].sum() if viol.height else 0} net", flush=True)
    with pl.Config(tbl_rows=50, tbl_cols=20, tbl_width_chars=250):
        print(viol.select("near", "far", "near_bid", "far_ask", "near_bid_size", "far_ask_size", "contracts", "net_usd"))
    return df


# ---- capped-book ledger -----------------------------------------------------------------------------
EVENT_CAP = 5.0
TOTAL_CAP = 30.0


def capped_ledger(tr: pl.DataFrame, strategy: str, variant: str, *, event_cap: float = EVENT_CAP,
                  total_cap: float = TOTAL_CAP) -> pl.DataFrame:
    """Chronological book: one position per market (its first qualifying decision), <= event_cap of
    cost per event and <= total_cap open in total (cost released at settlement). Each fill buys
    floor(room / unit cost) contracts at the trade's taker price, fee recomputed and rounded up
    per order at that size."""
    from albert.fees import kalshi_fee

    x = tr.sort("t", "ticker").unique("ticker", keep="first", maintain_order=True)   # deterministic tie-break
    open_pos: list[tuple[datetime, str, float]] = []   # (settle_ts, event, cost)
    led = []
    for r in x.iter_rows(named=True):
        t = r["t"]
        open_pos = [p for p in open_pos if p[0] > t]
        used = sum(p[2] for p in open_pos)
        ev_used = sum(p[2] for p in open_pos if p[1] == r["event_ticker"])
        room = min(event_cap - ev_used, total_cap - used)
        px = float(r["px"])
        if room < px + 0.02:
            continue
        n = math.floor(room / (px + 0.01))
        fee = kalshi_fee(n, px, fee_type=r.get("fee_type") or "quadratic", multiplier=r["fee_multiplier"])
        while n > 0 and n * px + fee > room:
            n -= 1
            fee = kalshi_fee(n, px, fee_type=r.get("fee_type") or "quadratic", multiplier=r["fee_multiplier"])
        if n <= 0:
            continue
        side = r.get("side") or "no"
        won = (r["y"] == 1) if side == "yes" else (r["y"] == 0)
        cost = n * px + fee
        settle = max(r["settlement_ts"], t + timedelta(minutes=1))
        open_pos.append((settle, r["event_ticker"], cost))
        led.append({"strategy": strategy, "variant": variant, "split": r["split"], "struct_id": r["ticker"],
                    "event_ticker": r["event_ticker"], "ticker": r["ticker"], "side": side, "n": n, "t_in": t,
                    "cost_usd": round(cost, 4), "settle_ts": settle, "payoff_usd": float(n if won else 0)})
    return pl.DataFrame(led, schema={"strategy": pl.String, "variant": pl.String, "split": pl.String, "struct_id": pl.String,
                                     "event_ticker": pl.String, "ticker": pl.String, "side": pl.String, "n": pl.Int64,
                                     "t_in": pl.Datetime("us", "UTC"), "cost_usd": pl.Float64,
                                     "settle_ts": pl.Datetime("us", "UTC"), "payoff_usd": pl.Float64})


def book_metrics(led: pl.DataFrame) -> dict[str, Any]:
    from albert.research import vol_metrics as V

    if not led.height:
        return {"positions": 0}
    marks = V.load_marks(led["ticker"].unique().to_list())
    m = V.ledger_metrics(led, marks)
    keep = ("positions", "events", "pnl_usd", "usd_per_week", "ann_vol_pct", "sharpe", "sharpe_ci90", "sharpe_ex_top5",
            "usd_per_week_ex_top5", "max_dd_usd", "weeks_positive_pct", "mean_open_mtm_usd", "skew", "marked_share",
            "first_day", "last_day")
    return {k: m.get(k) for k in keep}


def load_panel(u: pl.DataFrame | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(universe, quotes, fixed-lead panel); the panel is cached in OUT/panel.parquet."""
    u = pl.read_parquet(OUT / "universe.parquet") if u is None else u
    c = load_candles(u["ticker"].to_list())
    q = quotes(c)
    path = OUT / "panel.parquet"
    if path.exists():
        pn = pl.read_parquet(path)
    else:
        pn = fixed_lead_panel(u, q)
        _save(pn, path)
    return u, q, pn


def explore(split: str = "dev") -> None:
    """Dev-only screens for H1, H2, H6 (the holdout is never read here)."""
    _, _, pn = load_panel()
    assert split == "dev"
    h1 = h1_cells(pn, split)
    h6 = h6_cells(pn, split)
    h2 = h2_monthly(pn, ("dev",))
    for name, df in (("h1", h1), ("h6", h6), ("h2", h2)):
        df.write_csv(OUT / f"explore_dev_{name}.csv")
    with pl.Config(tbl_rows=200, tbl_cols=30, tbl_width_chars=250):
        print(h1.select("lead_d", "band", "kg", "n", "events", "c", "t", "per_usd_pct", "per_usd_day_bp", "mean_days",
                        "yes_rate", "mean_mid"))
        print(h6.select("qtype", "lg", "n", "events", "c", "t", "per_usd_pct", "per_usd_day_bp", "yes_rate", "mean_mid"))
        print(h2)


# ---- frozen rules: dev -> holdout, scored once ---------------------------------------------------
# Frozen 2026-11-23 from the dev screens (explore_dev_*.csv), before any holdout outcome was read.
_EV = (pl.col("kind") == "event") & (pl.col("spread") <= MAX_SPREAD)
FROZEN: dict[str, dict[str, Any]] = {
    # F1 last-day longshot NO on one-off events: dev H1 cell lead 1 d, YES 2-10c (+2.0c, t 7.5, 169 events)
    "F1_lastday_longshot_no": {"src": "panel", "filter": _EV & (pl.col("lead_d") == 1)
                               & (pl.col("mid") >= 0.02) & (pl.col("mid") < 0.10)},
    # F2 far-from-deadline status-quo NO: event kind, leads 7/14/30 d, YES 10-90c (dev H1 cells +4 to +19c)
    "F2_far_statusquo_no": {"src": "panel", "filter": _EV & pl.col("lead_d").is_in([7, 14, 30])
                            & (pl.col("mid") >= 0.10) & (pl.col("mid") < 0.90)},
    # F3 front-leg drop >= 5c in an hour, far leg unchanged: buy the far leg's NO (dev H4 +18c, t 4.2)
    "F3_frontdrop_far_no": {"src": "h4", "filter": (pl.col("side") == "no") & (pl.col("move") == 0.05)},
}


def rule_trades(name: str, pn: pl.DataFrame, extra: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Trades of one frozen rule on every split (filters only use information at the decision)."""
    r = FROZEN[name]
    src = pn if r["src"] == "panel" else extra[r["src"]]
    x = src.filter(r["filter"])
    if "side" not in x.columns:
        x = x.with_columns(pl.lit("no").alias("side"))
    return x


def score_rules(pn: pl.DataFrame, extra: dict[str, pl.DataFrame]) -> list[dict[str, Any]]:
    out = []
    led_dir = OUT / "ledgers"
    led_dir.mkdir(parents=True, exist_ok=True)
    for name in FROZEN:
        x = rule_trades(name, pn, extra)
        for split in ("dev", "holdout"):
            xs = x.filter(pl.col("split") == split)
            st = cell_stats(xs) if xs.height > 1 else {"n": xs.height}
            led = capped_ledger(xs, name, "frozen")
            if led.height:
                led.write_parquet(led_dir / f"ledger_{name}_{split}.parquet")
            bm = book_metrics(led)
            wk = None
            if xs.height:
                wk = (xs.with_columns(pl.col("t").dt.truncate("1w").alias("wk")).group_by("wk")
                      .agg(pl.col("pnl").sum()).select((pl.col("pnl") > 0).mean()).item())
            out.append({"rule": name, "split": split, **st, "trade_weeks_pos_pct": round(100 * wk, 0) if wk is not None else None,
                        **{f"book_{k}": v for k, v in bm.items()}})
            print(name, split, out[-1], flush=True)
    return out


def analyze() -> None:
    """Score the frozen rules on dev and holdout once, then the descriptive holdout tables."""
    u, q, pn = load_panel()
    pairs = strip_pairs(u)
    gr = hourly_pair_grid(pairs, q).with_columns(split_of(pl.col("h"), pl.col("T_2")).alias("split"))
    _save(gr, OUT / "pair_grid.parquet")
    s4 = pl.concat([h4_signals(gr, mv) for mv in (0.05, 0.10)], how="diagonal_relaxed")
    _save(s4, OUT / "h4_signals.parquet")
    res = score_rules(pn, {"h4": s4})
    _dump(res, OUT / "frozen_results.json")
    pl.DataFrame(res, infer_schema_length=None).drop([c for c in ("book_sharpe_ci90",) if c in res[0]]).write_csv(
        OUT / "frozen_results.csv")
    # descriptive (after the rules were frozen and scored): every screen cell on the holdout
    h1_cells(pn, "holdout").write_csv(OUT / "holdout_h1.csv")
    h6_cells(pn, "holdout").write_csv(OUT / "holdout_h6.csv")
    h2_monthly(pn, ("dev", "holdout")).write_csv(OUT / "h2_monthly_all.csv")
    rows = []
    for mv in (0.05, 0.10):
        for side in ("yes", "no"):
            x = s4.filter((pl.col("split") == "holdout") & (pl.col("move") == mv) & (pl.col("side") == side))
            rows.append({"cell": f"move{int(mv * 100)}_{side}", **(cell_stats(x) if x.height > 1 else {"n": x.height}),
                         **({f"mk_{k}h_c": round(100 * float(x[f"mk_{k}h"].drop_nulls().mean()), 2) for k in (1, 6, 24)}
                            if x.height else {})})
    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "holdout_h4.csv")
    runs, summ = h5_arbs(gr.filter(pl.col("split") == "holdout"))
    _save(runs, OUT / "h5_runs_holdout.parquet")
    _dump(summ, OUT / "h5_holdout.json")
    print("H5 holdout", summ)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog=f"albert.research.{SLUG}")
    ap.add_argument("step", choices=["scan", "universe", "candles", "explore", "strips", "panel", "analyze", "live", "all"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if a.step == "scan":
        scan(a.limit)
    elif a.step == "universe":
        build_universe()
    elif a.step == "candles":
        fetch_candles(pl.read_parquet(OUT / "universe.parquet"))
    elif a.step == "explore":
        explore()
    elif a.step == "strips":
        explore_strips()
    elif a.step == "live":
        live_snapshot()
    elif a.step == "analyze":
        analyze()


if __name__ == "__main__":
    main()
