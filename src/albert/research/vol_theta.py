"""Time-premium ("theta") and hazard-term-structure strategies on Kalshi deadline contracts.

Research only: no orders of any kind. Builds on :mod:`albert.research.pm_hazard` (its universe,
deadline parsing from the rules text, hourly candles, decision panel and hazard estimators) and
tests whether the documented mispricings of "will X happen by T" contracts survive execution:

* implied hazards exceed realized (calibration multiplier k ≈ 0.6–0.8 at every lead);
* sticky theta: with no event, prices decay at only 0.28–0.69 of the flat-hazard rate, then fall
  faster than a martingale in the final hours;
* forward segments of date strips are overpriced (0.27 implied vs 0.21 realized).

Strategies (each with a small pre-declared grid, ≤ 12 variants):

* **S1 maker theta harvest** — every hour from T − 30 d to T − 24 h, on each contract with YES mid
  in [2c, 60c] and spread ≤ 10c, rest one post-only NO bid at ``floor(1 − fair − margin)`` (capped
  one tick inside the YES bid) for that hour (re-quoted hourly). Filled only if a public print
  with ``taker_side = yes`` trades STRICTLY through our level (YES price > 1 − bid) between t + 60 s
  and t + 1 h; a through-print means every resting contract at our better level was consumed, so
  the whole (small) order fills. Held to settlement. fair = 1 − (1 − mid)^k with k fitted on
  training contracts only: ``hz`` (hazard calibration, k by kind → kind×lead → kind×lead×price,
  shrunk) or ``pm`` (price map, k by price bucket). Grid: fair × margin ∈ {0,1,2,4,8,12}c.
* **S2 forward-hazard calendar** — at the near leg's decision grid (leads 1 h–30 d) of adjacent
  nested legs T1 < T2 of one strip: short the forward segment = buy YES near at its ask + buy NO
  far at 1 − far bid (taker, both fees). Payoff 1 − 1[event in (T1, T2]]. Model:
  P(fwd) = (1 − p1_model)·(1 − r^{k_f}), r = (1 − m2)/(1 − m1), k_f fitted on training pairs with no
  near event; trade when expected payoff − cost ≥ θ. Grid: k_f pooled / by category × θ.
* **S3 final-hours decay** — at a fixed lead L ∈ {24, 12, 6, 3} h before T, when YES mid ≤ 50c and
  (optionally) above the flat-hazard theta path from the 72 h anchor by ≥ x: buy NO at the NO ask
  (taker, ``S3t``) or rest a NO bid one tick better than the best NO bid until T − 10 min with
  through-print fills (maker, ``S3m``). Held to settlement.

Walk-forward by month: fold M holds every decision made in calendar month M (Feb … Oct 2026, plus
the Nov 1–2 decisions in October's fold, and an extra ``2026-11x`` fold of contracts with
deadlines Nov 2–8 fetched for this study). Parameters (k tables, the variant) come only from
training contracts — deadline T before M − 2 d and settled (``settlement_ts``) before M — so every
input of a fold-M decision was known before the decision. Variant chosen on training data by PnL
per dollar-day of collateral (≥ 30 trades and ≥ 10 events). Capital is locked from the entry until
``settlement_ts`` (NO markets settle a median 11 h after T).

Execution: taker at the executable candle bid/ask (last hourly candle that closed at or before
the decision), Kalshi fee via :func:`albert.fees.kalshi_fee` (``conservative=True``: rounded up
to the cent per order) at the order size a $5-per-event cap allows; maker fee only on
``*_with_maker_fees`` series. Displayed size is not in the history (no order books for these
markets); orders are a few contracts.

Steps (``python -m albert.research.vol_theta run <step>``):

* ``plan``: trade windows per market (in-scope universe + the Nov 2–8 extension).
* ``candles``: hourly candles of the extension markets (pm_hazard's cover deadlines ≤ Nov 2).
* ``trades``: public trades of every planned window (``/historical/trades`` before the cutoff,
  ``/markets/trades`` after; raw pages cached append-only), checkpointed in parts.
* ``analyze``: S1–S3 grids, walk-forward, markouts, capped bankroll simulation, breakdowns,
  ``summary.json`` and ``verdict.md`` inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.research import pm_hazard as H

EXT_END = datetime(2026, 11, 8, 5, 0, tzinfo=UTC)    # extension: deadlines in (H.END, EXT_END]
TRADE_LOOKBACK = timedelta(days=31)
BAND_FETCH = (0.01, 0.70)        # a market's trades are fetched if it ever quoted a mid in here
POST_DELAY = timedelta(seconds=60)    # an order posted at t is live from t + 60 s
EVENT_CAP = 5.0                  # $ collateral per event
TOTAL_CAP = 20.0                 # $ total open collateral (positions + resting orders)
MIN_TRADES, MIN_EVENTS = 30, 10  # variant selection floor on training data

# S1
S1_LEADS = (24, 720)             # hours before T
S1_BAND = (0.02, 0.60)
S1_MARGINS = (0.0, 0.01, 0.02, 0.04, 0.08, 0.12)
FAIRS = ("hz", "pm")
LEVELS = {"hz": [["kind"], ["kind", "lead_b"], ["kind", "lead_b", "price_b"]], "pm": [["price_b"]]}
# S2
S2_THETAS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08)
S2_MAX_LEAD = 720
# S3
S3_LEADS = (24, 12, 6, 3)
S3_GAPS: tuple[float | None, ...] = (None, 0.0, 0.03)
S3_ANCHOR = 72
S3_MAX_MID = 0.50
END_BEFORE = timedelta(minutes=10)


def out_dir(s) -> Path:
    d = s.data_dir / "research" / "vol_theta"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


# ---- universe and data ---------------------------------------------------------------------------
def extension(u_all: pl.DataFrame) -> pl.DataFrame:
    """Hazard contracts with deadlines in (H.END, EXT_END], settled yes/no (all such NO markets had
    settled by the scan: 98% close within 26 h of T). Selected by T only (outcome-blind)."""
    return u_all.filter(pl.col("kind").is_in(H.HAZARD_TYPES) & pl.col("T").is_not_null()
                        & (pl.col("T") > H.END) & (pl.col("T") <= EXT_END)
                        & pl.col("result").is_in(["yes", "no"])
                        & (pl.col("close_time") > H.START) & (pl.col("open_time") < pl.col("T")))


def load_universe(s) -> pl.DataFrame:
    u_all = pl.read_parquet(H.out_dir(s) / "universe_all.parquet")
    u = pl.concat([H.in_scope(u_all).with_columns(pl.lit(False).alias("ext")),
                   extension(u_all).with_columns(pl.lit(True).alias("ext"))])
    return u.with_columns(pl.col("settlement_ts").fill_null(pl.col("close_time")))


def load_candles(s, tickers: pl.Series | None = None) -> pl.DataFrame:
    frames = [pl.scan_parquet(H.out_dir(s) / "candles.parquet")]
    ext = out_dir(s) / "candles_ext.parquet"
    if ext.exists():
        frames.append(pl.scan_parquet(ext))
    c = pl.concat(frames, how="diagonal_relaxed")
    if tickers is not None:
        c = c.filter(pl.col("ticker").is_in(tickers.implode()))
    return c.collect().unique(["ticker", "end_ts"])


def fetch_ext_candles(s, u: pl.DataFrame) -> None:
    """Hourly candles of the extension markets over [max(open, T − LOOKBACK), min(close, T) + 1 h]."""
    from albert.venues import kalshi as k

    d = out_dir(s)
    path = d / "candles_ext.parquet"
    have = set(pl.read_parquet(path)["ticker"]) if path.exists() else set()
    ext = u.filter(pl.col("ext") & ~pl.col("ticker").is_in(list(have)))
    api = H._api(s, 0.22, nap_cap=3.0, retries=120)
    cutoff = api.cutoff().get("market_settled_ts")
    keep = ["ticker", "end_ts", "yes_bid_close", "yes_ask_close", "price_close", "volume", "open_interest"]
    rows: list[dict[str, Any]] = []
    for r in ext.iter_rows(named=True):
        lo = max(r["open_time"], r["T"] - H.LOOKBACK)
        hi = min(r["close_time"], r["T"]) + timedelta(hours=1)
        try:
            page = api.candles(r["ticker"], lo, hi, 60, bool(cutoff and r["close_time"] < cutoff))
        except RuntimeError as error:
            print(f"skip {r['ticker']}: {str(error)[:120]}", flush=True)
            continue
        rows.extend({c: x[c] for c in keep} for x in k.candle_rows(page.payload, r["ticker"], 60))
    if rows:
        new = pl.DataFrame(rows, schema={"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"),
                                         "yes_bid_close": pl.Float64, "yes_ask_close": pl.Float64,
                                         "price_close": pl.Float64, "volume": pl.Float64,
                                         "open_interest": pl.Float64})
        old = pl.read_parquet(path) if path.exists() else None
        _save(pl.concat([f for f in (old, new) if f is not None]).unique(["ticker", "end_ts"]), path)
    print(f"ext candles: {ext.height} markets requested, {len(rows)} rows", flush=True)


def trade_plan(s, u: pl.DataFrame) -> pl.DataFrame:
    """One window per market that ever quoted a mid in BAND_FETCH (spread ≤ 15c) during its last
    TRADE_LOOKBACK: [max(open, T − lookback, START − 1 d), min(close, T) + 1 min]."""
    c = load_candles(s, u["ticker"])
    q = (c.join(u.select("ticker", "T"), on="ticker")
         .filter((pl.col("end_ts") >= pl.col("T") - TRADE_LOOKBACK) & pl.col("yes_bid_close").is_not_null()
                 & pl.col("yes_ask_close").is_not_null())
         .with_columns(((pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2).alias("mid"))
         .filter(pl.col("mid").is_between(*BAND_FETCH) & ((pl.col("yes_ask_close") - pl.col("yes_bid_close")) <= 0.15)))
    tick = q["ticker"].unique()
    w = (u.filter(pl.col("ticker").is_in(tick.implode()))
         .select("ticker", "event_ticker", "T", "close_time",
                 pl.max_horizontal("open_time", pl.col("T") - TRADE_LOOKBACK,
                                   pl.lit(H.START - timedelta(days=1))).alias("t_from"),
                 (pl.min_horizontal("close_time", "T") + timedelta(minutes=1)).alias("t_to"))
         .filter(pl.col("t_to") > pl.col("t_from")))
    return w.sort("T", descending=True)


TRADE_SCHEMA = {"trade_id": pl.String, "ticker": pl.String, "created_time": pl.Datetime("us", "UTC"),
                "count": pl.Float64, "yes_price": pl.Float64, "taker_side": pl.String}


def fetch_trades(s, plan: pl.DataFrame, *, workers: int = 2, min_interval: float = 0.45,
                 flush_every: int = 100) -> None:
    """Every public trade of each planned window. A window that straddles the historical cutoff is
    split (``/historical/trades`` before, ``/markets/trades`` from it). Pages are cached raw
    (append-only); rows are written to ``trades/part_*.parquet``, markets to ``trades_done``.
    Each worker has its own client (≤ 1/min_interval requests/s each)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from albert.venues import kalshi as k

    d = out_dir(s)
    parts = d / "trades"
    parts.mkdir(exist_ok=True)
    done_path = d / "trades_done.parquet"
    done = set(pl.read_parquet(done_path)["ticker"]) if done_path.exists() else set()
    todo = plan.filter(~pl.col("ticker").is_in(list(done)))
    local = threading.local()

    def api():
        if not hasattr(local, "api"):
            local.api = H._api(s, min_interval, nap_cap=5.0, retries=120)
        return local.api

    cutoff = api().cutoff().get("trades_created_ts")
    print(f"trades: {todo.height} windows to fetch ({len(done)} done), cutoff {cutoff}, {workers} workers", flush=True)

    def one(r: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None]:
        segs = []
        if cutoff is None or r["t_to"] <= cutoff:
            segs.append((True, r["t_from"], r["t_to"]))
        elif r["t_from"] >= cutoff:
            segs.append((False, r["t_from"], r["t_to"]))
        else:
            segs += [(True, r["t_from"], cutoff), (False, cutoff, r["t_to"])]
        got: list[dict[str, Any]] = []
        for hist, a, b in segs:
            try:
                for page in api().trades(r["ticker"], hist, min_ts=a, max_ts=b):
                    got.extend(k.trade_rows(page.payload))
            except RuntimeError as error:
                print(f"skip {r['ticker']}: {str(error)[:120]}", flush=True)
                return r["ticker"], None
        return r["ticker"], got

    rows: list[dict[str, Any]] = []
    fin: list[str] = []
    n_part = len(list(parts.glob("part_*.parquet")))

    def flush() -> None:
        nonlocal rows, fin, n_part
        if rows:
            df = pl.DataFrame(rows, schema=TRADE_SCHEMA).unique("trade_id")
            _save(df, parts / f"part_{n_part:05d}.parquet")
            n_part += 1
        done.update(fin)
        _save(pl.DataFrame({"ticker": sorted(done)}, schema={"ticker": pl.String}), done_path)
        rows, fin = [], []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (tk, got) in enumerate(pool.map(one, todo.iter_rows(named=True)), 1):
            if got is not None:
                rows.extend(got)
                fin.append(tk)
            if i % flush_every == 0 or len(rows) > 400_000:
                flush()
                print(f"  trades {i}/{todo.height} windows", flush=True)
    flush()
    print("trades done", flush=True)


def load_trades(s, tickers: pl.Series | None = None) -> pl.LazyFrame:
    t = pl.scan_parquet(out_dir(s) / "trades" / "part_*.parquet")
    if tickers is not None:
        t = t.filter(pl.col("ticker").is_in(tickers.implode()))
    return t


# ---- execution helpers ---------------------------------------------------------------------------
def order_size(price: Any) -> np.ndarray:
    """Contracts a single order buys under the per-event cap at this unit cost (≥ 1)."""
    p = np.maximum(np.asarray(price, dtype=float), 0.01)
    return np.maximum(np.floor(EVENT_CAP / p + 1e-9), 1.0)


def fee_per_contract(price: Any, n: Any, *, maker: bool = False, fee_type: Any = "quadratic",
                     mult: Any = 1.0) -> np.ndarray:
    """Vectorized :func:`albert.fees.kalshi_fee` (conservative: the order's fee rounded up to the
    cent) divided by the order size n. Maker fee only on ``*_with_maker_fees`` series."""
    from albert.fees import KALSHI_MAKER, KALSHI_TAKER

    p = np.asarray(price, dtype=float)
    n = np.asarray(n, dtype=float)
    ft = np.broadcast_to(np.asarray(fee_type, dtype=object), p.shape)
    m = np.broadcast_to(np.asarray(mult, dtype=float), p.shape)
    if maker:
        coef = np.array([KALSHI_MAKER if (f and "maker_fee" in f) else 0.0 for f in ft], dtype=float)
    else:
        coef = np.full(p.shape, KALSHI_TAKER)
    raw = coef * np.nan_to_num(m, nan=1.0) * n * p * (1 - p)
    return np.ceil(np.round(raw * 100, 9)) / 100 / n


def whole_cents_down(x: Any) -> np.ndarray:
    return np.floor(np.asarray(x, dtype=float) * 100 + 1e-9) / 100


def no_bid_price(fair: Any, bid: Any, margin: float) -> np.ndarray:
    """Post-only NO bid at floor(1 − fair − margin), capped one cent inside the NO ask (1 − YES bid);
    NaN when below 1c."""
    nb = np.minimum(whole_cents_down(1 - np.asarray(fair, dtype=float) - margin),
                    whole_cents_down(1 - np.asarray(bid, dtype=float) - 0.01))
    return np.where(nb >= 0.01, nb, np.nan)


def theta_path(p_anchor: Any, tau_anchor: Any, tau_now: Any) -> np.ndarray:
    """Flat-hazard no-event price after decaying from p_anchor (time left tau_anchor) to tau_now:
    1 − (1 − p_A)^{tau_now/tau_A}."""
    pa = np.clip(np.asarray(p_anchor, dtype=float), H.EPS, 1 - H.EPS)
    return 1 - (1 - pa) ** (np.asarray(tau_now, dtype=float) / np.asarray(tau_anchor, dtype=float))


def short_forward_value(m1: Any, m2: Any, k1: Any, kf: Any) -> np.ndarray:
    """Model expected payoff of the short forward (YES near + NO far, payoff 1 − 1[event in
    (T1, T2]]): 1 − (1 − p1)·(1 − r^{k_f}), p1 = 1 − (1 − m1)^{k1}, r = (1 − m2)/(1 − m1) ∈ (0, 1]."""
    m1 = np.clip(np.asarray(m1, dtype=float), H.EPS, 1 - H.EPS)
    m2 = np.clip(np.asarray(m2, dtype=float), H.EPS, 1 - H.EPS)
    r = np.clip((1 - m2) / (1 - m1), 1e-3, 1.0)
    surv1 = (1 - m1) ** np.asarray(k1, dtype=float)
    return 1 - surv1 * (1 - r ** np.asarray(kf, dtype=float))


# ---- walk-forward --------------------------------------------------------------------------------
FOLDS = ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09",
         "2026-10", "2026-11"]


def fold_start(m: str) -> datetime:
    y, mo = (int(x) for x in m.split("-"))
    return datetime(y, mo, 1, tzinfo=UTC)


def train_contracts(u: pl.DataFrame, m: str) -> pl.Series:
    """Tickers usable for training before fold m: deadline before m − 2 d (outcome-blind) and
    settled before m."""
    t0 = fold_start(m)
    return u.filter((pl.col("T") < t0 - timedelta(days=2)) & (pl.col("settlement_ts") < t0))["ticker"]


def fold_col(t: str = "t") -> pl.Expr:
    """Decision month; Nov 1–8 decisions form the '2026-11' fold."""
    return pl.col(t).dt.strftime("%Y-%m").alias("fold")


def apply_k_vec(df: pl.DataFrame, tables: dict[tuple, dict], levels: list[list[str]],
                mid_col: str = "mid") -> np.ndarray:
    """Vectorized :func:`pm_hazard.apply_k`: p = 1 − (1 − mid)^k, k from the finest level present."""
    lk = np.full(df.height, tables[()][()])
    for cols in levels:
        tab = tables[tuple(cols)]
        if not tab:
            continue
        keys = list(tab)
        t = pl.DataFrame({c: [k[i] for k in keys] for i, c in enumerate(cols)} | {"_lk": list(tab.values())},
                         schema={**{c: df.schema[c] for c in cols}, "_lk": pl.Float64})
        got = (df.select(cols).with_row_index("_i").join(t, on=cols, how="left").sort("_i")["_lk"]
               .to_numpy())
        lk = np.where(np.isfinite(got), got, lk)
    mid = np.clip(df[mid_col].to_numpy(), H.EPS, 1 - H.EPS)
    return 1 - (1 - mid) ** np.exp(lk)


def k_of(df: pl.DataFrame, tables: dict[tuple, dict], levels: list[list[str]]) -> np.ndarray:
    """The hazard multiplier k each row gets (for model hazards)."""
    lk = np.full(df.height, tables[()][()])
    for cols in levels:
        tab = tables[tuple(cols)]
        keys = list(tab)
        if not keys:
            continue
        t = pl.DataFrame({c: [k[i] for k in keys] for i, c in enumerate(cols)} | {"_lk": list(tab.values())},
                         schema={**{c: df.schema[c] for c in cols}, "_lk": pl.Float64})
        got = df.select(cols).with_row_index("_i").join(t, on=cols, how="left").sort("_i")["_lk"].to_numpy()
        lk = np.where(np.isfinite(got), got, lk)
    return np.exp(lk)


def fit_tables(panel: pl.DataFrame, train: pl.Series) -> dict[str, dict]:
    tr = panel.filter(pl.col("ticker").is_in(train.implode()) & (pl.col("spread") <= H.MAX_SPREAD))
    return {f: H.fit_k(tr, LEVELS[f]) for f in FAIRS}


# ---- quotes and prints ---------------------------------------------------------------------------
def quote_frame(c: pl.DataFrame) -> pl.DataFrame:
    return (c.filter(pl.col("yes_bid_close").is_not_null() & pl.col("yes_ask_close").is_not_null()
                     & (pl.col("yes_ask_close") > pl.col("yes_bid_close")))
            .select("ticker", pl.col("end_ts").alias("q_ts"), pl.col("yes_bid_close").alias("bid"),
                    pl.col("yes_ask_close").alias("ask"))
            .sort("ticker", "q_ts"))


def quote_at(frame: pl.DataFrame, q: pl.DataFrame, t_col: str, *, ticker_col: str = "ticker",
             max_age: timedelta = timedelta(hours=2), suffix: str = "") -> pl.DataFrame:
    """Attach the last quote whose candle ended at or before ``t_col`` (≤ max_age old)."""
    qq = q.rename({"ticker": ticker_col, "q_ts": f"q_ts{suffix}", "bid": f"bid{suffix}", "ask": f"ask{suffix}"})
    out = frame.sort(ticker_col, t_col).join_asof(qq, left_on=t_col, right_on=f"q_ts{suffix}", by=ticker_col,
                                                  strategy="backward", check_sortedness=False)
    return out.filter(pl.col(f"q_ts{suffix}").is_not_null()
                      & ((pl.col(t_col) - pl.col(f"q_ts{suffix}")) <= max_age))


def mid_after(frame: pl.DataFrame, q: pl.DataFrame, at_col: str, out_col: str) -> pl.DataFrame:
    """YES mid of the last candle ending at or before ``at_col`` (the settlement value 1·y when the
    market had closed by then)."""
    qm = q.select("ticker", pl.col("q_ts").alias("_qts"), ((pl.col("bid") + pl.col("ask")) / 2).alias("_m"))
    out = frame.sort("ticker", at_col).join_asof(qm, left_on=at_col, right_on="_qts", by="ticker",
                                                 strategy="backward", check_sortedness=False)
    return out.with_columns(pl.when(pl.col(at_col) >= pl.col("close_time")).then(pl.col("y").cast(pl.Float64))
                            .otherwise(pl.col("_m")).alias(out_col)).drop("_qts", "_m")


def taker_yes_prints(s, tickers: pl.Series) -> pl.DataFrame:
    """Public prints whose taker bought YES (the flow that hits a resting NO bid)."""
    return (load_trades(s, tickers).filter(pl.col("taker_side") == "yes")
            .select("ticker", pl.col("created_time").alias("ts"), pl.col("yes_price").alias("px"))
            .unique().collect().sort("ticker", "ts"))


def running_max_steps(pr: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Prints that set a new running maximum within each group (time order): the first print
    through any level L is the first step with px > L."""
    pr = pr.sort([*by, "ts"]).with_columns(pl.col("px").cum_max().over(by).alias("_rm"))
    pr = pr.with_columns(pl.col("_rm").shift(1).over(by).alias("_prev"))
    return pr.filter(pl.col("_prev").is_null() | (pl.col("px") > pl.col("_prev"))).drop("_rm", "_prev")


def first_through(orders: pl.DataFrame, steps: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """fill_ts = time of the first print strictly through the order's YES level (``level``)."""
    cols = list(dict.fromkeys(["_oid", *by, "level"]))
    j = (orders.select(cols).join(steps.select(*by, "ts", "px"), on=by)
         .filter(pl.col("px") > pl.col("level") + 1e-9)
         .group_by("_oid").agg(pl.col("ts").min().alias("fill_ts")))
    return orders.join(j, on="_oid", how="left").with_columns(pl.col("fill_ts").is_not_null().alias("filled"))


# ---- statistics ----------------------------------------------------------------------------------
def stats(df: pl.DataFrame, *, orders: int | None = None) -> dict[str, Any]:
    """Per filled contract (1 lot per order): mean PnL in cents with t clustered by event, PnL per
    dollar risked and per dollar-day of collateral; markouts when present."""
    if df.height == 0:
        return {"orders": orders, "n": 0, "events": 0}
    mu, t, k = H._cm(df["pnl"].to_numpy() * 100, df["event_ticker"].to_numpy())
    _, t_ser, k_ser = H._cm(df["pnl"].to_numpy() * 100, df["series_ticker"].to_numpy())
    out = {"orders": orders, "n": df.height, "events": k, "series": k_ser, "pnl_c": H._r(mu, 3), "t": H._r(t, 2),
           "t_series": H._r(t_ser, 2),
           "per_dollar": H._r(float(df["pnl"].sum() / df["cost"].sum()), 4),
           "per_dollar_day": H._r(float(df["pnl"].sum() / max(df["dollar_days"].sum(), 1e-9)), 5),
           "mean_cost": H._r(float(df["cost"].mean()), 3), "mean_days": H._r(float((df["dollar_days"] / df["cost"]).mean()), 2),
           "win_rate": H._r(float((df["pnl"] > 0).mean()), 3)}
    if orders:
        out["fill_rate"] = H._r(df.height / orders, 4)
    for col in ("mk_1h", "mk_1d", "mk_settle"):
        if col in df.columns:
            out[f"{col}_c"] = H._r(float(df[col].mean()) * 100, 3)
    return out


def score(st: dict[str, Any]) -> float | None:
    """Training score used to choose a variant: PnL per dollar-day (needs ≥ MIN_TRADES trades and
    ≥ MIN_EVENTS events)."""
    if (st.get("n") or 0) < MIN_TRADES or (st.get("events") or 0) < MIN_EVENTS:
        return None
    return st.get("per_dollar_day")


def monthly(oos: pl.DataFrame) -> pl.DataFrame:
    if not oos.height:
        return pl.DataFrame()
    f = oos.filter("filled")
    rows = []
    for (m,), g in f.group_by("fold"):
        st = stats(g, orders=oos.filter(pl.col("fold") == m).height)
        rows.append({"fold": m, "variant": g["variant"][0], "pnl_sum_c": H._r(float(g["pnl"].sum()) * 100, 2), **st})
    return pl.DataFrame(rows, infer_schema_length=None).sort("fold")


# ---- capped bankroll simulation ------------------------------------------------------------------
def simulate(c: pl.DataFrame, *, event_cap: float = EVENT_CAP, total_cap: float = TOTAL_CAP,
             return_trades: bool = False) -> dict[str, Any]:
    """Chronological $100-bankroll book with ≤ event_cap collateral per event and ≤ total_cap open
    (positions + resting orders). Candidates (sorted by t, then priority) need: t, t_end (unfilled
    order cancel; = t for takers), filled, settle (release of a filled position), event_ticker,
    p1/ft1/mu1 and optional p2/ft2/mu2 (leg unit prices and fee params), maker, payoff (realized
    payoff per contract), priority. Size = the most whole contracts the caps allow, fee per order
    with :func:`albert.fees.kalshi_fee` (conservative)."""
    import heapq

    from albert.fees import kalshi_fee

    if not c.height:
        return {"trades": 0}
    c = c.sort(["t", "priority"], descending=[False, True])
    heap: list[tuple] = []
    open_total = 0.0
    ev_open: dict[str, float] = {}
    trades = []
    skipped = 0
    two = "p2" in c.columns

    def release(until) -> None:
        nonlocal open_total
        while heap and heap[0][0] <= until:
            _, _, amt, ev = heapq.heappop(heap)
            open_total -= amt
            ev_open[ev] = ev_open.get(ev, 0.0) - amt

    ledger: list[dict[str, Any]] = []
    has_ledger = "ticker1" in c.columns
    seq = 0
    for r in c.iter_rows(named=True):
        release(r["t"])
        ev = r["event_ticker"]
        room = min(event_cap - ev_open.get(ev, 0.0), total_cap - open_total)
        legs = [(r["p1"], r["ft1"], r["mu1"])]
        if two and r.get("p2") is not None:
            legs.append((r["p2"], r["ft2"], r["mu2"]))
        unit = sum(p for p, _, _ in legs)
        n = math.floor(room / unit + 1e-9) if unit > 0 else 0

        def fee(k: int) -> float:
            return sum(kalshi_fee(k, p, maker=bool(r["maker"]), fee_type=ft or "quadratic", multiplier=mu)
                       for p, ft, mu in legs)
        while n >= 1 and n * unit + fee(n) > room + 1e-9:
            n -= 1
        if n < 1:
            skipped += 1
            continue
        seq += 1
        if not r["filled"]:
            amt = n * unit
            heapq.heappush(heap, (r["t_end"], seq, amt, ev))
        else:
            cost = n * unit + fee(n)
            t_in = r.get("t_fill") or r["t"]
            heapq.heappush(heap, (r["settle"], seq, cost, ev))
            pnl = n * r["payoff"] - cost
            if has_ledger:
                for i, (p, ft, mu) in enumerate(legs, start=1):
                    ledger.append({"struct_id": str(seq), "fold": r["fold"], "event_ticker": ev,
                                   "ticker": r[f"ticker{i}"], "side": r[f"side{i}"], "n": n, "t_in": t_in,
                                   "cost_usd": n * p + kalshi_fee(n, p, maker=bool(r["maker"]),
                                                                  fee_type=ft or "quadratic", multiplier=mu),
                                   "settle_ts": r[f"settle{i}"], "payoff_usd": n * float(r[f"pay{i}"])})
            trades.append({"t": r["t"], "fold": r["fold"], "settle": r["settle"], "event_ticker": ev, "n": n,
                           "cost": cost, "pnl": pnl, "p1": r["p1"], "priority": r["priority"],
                           "dollar_days": cost * max((r["settle"] - t_in).total_seconds() / 86400, 1 / 24)})
        open_total += (n * unit if not r["filled"] else n * unit + fee(n))
        ev_open[ev] = ev_open.get(ev, 0.0) + (n * unit if not r["filled"] else n * unit + fee(n))
    tr = pl.DataFrame(trades, infer_schema_length=None)
    if not tr.height:
        return {"trades": 0, "skipped": skipped}
    t0 = min(fold_start(FOLDS[0]), tr["t"].min())
    t1 = tr["t"].max()
    weeks = max((t1 - t0).total_seconds() / 604800, 1e-9)
    by_m = tr.group_by("fold").agg(pl.col("pnl").sum().round(2).alias("pnl_usd"), pl.len().alias("orders"),
                                   pl.col("n").sum().alias("contracts")).sort("fold")
    eq = tr.sort("settle").with_columns(pl.col("pnl").cum_sum().alias("eq"))
    dd = float((eq["eq"] - eq["eq"].cum_max().clip(lower_bound=0)).min())
    _, t, k = H._cm(tr["pnl"].to_numpy(), tr["event_ticker"].to_numpy())
    ev = tr.group_by("event_ticker").agg(pl.col("pnl").sum()).sort("pnl", descending=True)
    top5 = float(ev["pnl"].head(5).sum())
    extra = {}
    if return_trades:
        extra = {"trades_frame": tr, "ledger_frame": pl.DataFrame(ledger, infer_schema_length=None)}
    return {**extra, "top5_events_pnl_usd": H._r(top5, 2),
            "usd_per_week_ex_top5": H._r((float(tr["pnl"].sum()) - top5) / weeks, 3),
            "trades": tr.height, "contracts": int(tr["n"].sum()), "events": k, "skipped_cap": skipped,
            "pnl_usd": H._r(float(tr["pnl"].sum()), 2), "weeks": H._r(weeks, 1),
            "usd_per_week": H._r(float(tr["pnl"].sum()) / weeks, 3), "t_event": H._r(t, 2),
            "per_dollar_day": H._r(float(tr["pnl"].sum() / tr["dollar_days"].sum()), 5),
            "mean_open_usd": H._r(float(tr["dollar_days"].sum()) / (weeks * 7), 2),
            "max_drawdown_usd": H._r(dd, 2), "months_positive": int((by_m["pnl_usd"] > 0).sum()),
            "months": by_m.height, "by_month": by_m.to_dicts()}


# ---- S1: hourly maker NO bids --------------------------------------------------------------------
META = ["ticker", "event_ticker", "series_ticker", "category", "kind", "T", "w_start", "open_time", "close_time",
        "settlement_ts", "y", "fee_type", "fee_multiplier"]


def _ceil_hour(e: pl.Expr) -> pl.Expr:
    return (e - timedelta(microseconds=1)).dt.truncate("1h") + timedelta(hours=1)


def _lead_cols(df: pl.DataFrame, t_col: str) -> pl.DataFrame:
    df = df.with_columns(
        ((pl.col("T") - pl.col(t_col)).dt.total_seconds() / 3600).alias("lead_h"),
        ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("ask") - pl.col("bid")).alias("spread"),
        ((pl.col("T") - pl.max_horizontal(t_col, pl.col("w_start").fill_null(pl.col(t_col)))).dt.total_seconds()
         / 86400).alias("tau_d"))
    return df.with_columns(H._bucket(pl.col("lead_h"), H.LEAD_BUCKETS, "lead_b"),
                           H._bucket(pl.col("mid"), H.PRICE_BUCKETS, "price_b"))


def s1_orders(u: pl.DataFrame, q: pl.DataFrame) -> pl.DataFrame:
    """Order hours e in [T − 720 h, T − 24 h] while the market is open, quote = last candle ending
    at or before e (≤ 6 h old: hours without a candle had no update), in the S1 band."""
    lo_h, hi_h = S1_LEADS
    m = u.select(META).with_columns(
        _ceil_hour(pl.max_horizontal(pl.col("open_time") + timedelta(hours=1), pl.col("T") - timedelta(hours=hi_h),
                                     pl.lit(H.START))).alias("e0"),
        (pl.min_horizontal(pl.col("close_time") - timedelta(seconds=1), pl.col("T") - timedelta(hours=lo_h))
         .dt.truncate("1h")).alias("e1")).filter(pl.col("e1") >= pl.col("e0"))
    hrs = (m.select("ticker", pl.datetime_ranges("e0", "e1", interval="1h").alias("e")).explode("e"))
    o = quote_at(hrs, q, "e", max_age=timedelta(hours=6))
    o = o.join(m.drop("e0", "e1"), on="ticker")
    o = _lead_cols(o, "e").filter((pl.col("spread") <= H.MAX_SPREAD) & pl.col("mid").is_between(*S1_BAND))
    return o.with_columns(fold_col("e"), ((pl.col("e") - pl.col("q_ts")).dt.total_hours()).alias("q_age_h")).drop("q_ts")


def hour_steps(pr: pl.DataFrame, delay: timedelta = POST_DELAY) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Taker-YES prints bucketed by the order hour they could fill: an order posted at e is live
    (e + delay, e + 1 h]. Returns the running-max steps per (ticker, e) and the hourly max."""
    x = pr.with_columns((pl.col("ts") - timedelta(microseconds=1)).dt.truncate("1h").alias("e"))
    x = x.filter((pl.col("ts") - pl.col("e")) > delay)
    st = running_max_steps(x, ["ticker", "e"])
    hmax = st.group_by("ticker", "e").agg(pl.col("px").max().alias("hmax"))
    return st, hmax


def _markouts(f: pl.DataFrame, q: pl.DataFrame, price_col: str) -> pl.DataFrame:
    if not f.height:
        return f.with_columns(*(pl.lit(None, dtype=pl.Float64).alias(c) for c in ("mk_1h", "mk_1d", "mk_settle")))
    f = f.with_columns((pl.col("fill_ts") + timedelta(hours=1)).alias("_a1"),
                       (pl.col("fill_ts") + timedelta(days=1)).alias("_a2"))
    f = mid_after(f, q, "_a1", "_m1")
    f = mid_after(f, q, "_a2", "_m2")
    return f.with_columns(((1 - pl.col("_m1")) - pl.col(price_col)).alias("mk_1h"),
                          ((1 - pl.col("_m2")) - pl.col(price_col)).alias("mk_1d"),
                          ((1 - pl.col("y")) - pl.col(price_col)).alias("mk_settle")).drop("_a1", "_a2", "_m1", "_m2")


def _maker_trade_cols(f: pl.DataFrame, price_col: str) -> pl.DataFrame:
    """PnL per contract of a filled NO bid held to settlement (maker fee at the $5-cap order size)."""
    nb = f[price_col].to_numpy()
    fee = fee_per_contract(nb, order_size(nb), maker=True, fee_type=f["fee_type"].to_numpy(),
                           mult=f["fee_multiplier"].fill_null(1.0).to_numpy())
    f = f.with_columns(pl.Series("fee_pc", fee))
    return f.with_columns(
        ((1 - pl.col("y")) - pl.col(price_col) - pl.col("fee_pc")).alias("pnl"),
        (pl.col(price_col) + pl.col("fee_pc")).alias("cost")).with_columns(
        (pl.col("cost") * ((pl.col("settlement_ts") - pl.col("fill_ts")).dt.total_seconds() / 86400)
         .clip(lower_bound=1 / 24)).alias("dollar_days"))


def s1_fold(o: pl.DataFrame, tables: dict[str, dict], steps: pl.DataFrame, q: pl.DataFrame,
            margin: float, fair: str, *, markouts: bool = True, fills_only: bool = False):
    """Price the fold's orders for one variant; returns every order with filled/fill_ts and, for
    fills, pnl/cost/dollar_days (and markouts). ``fills_only``: (fills, number of orders)."""
    p_fair = apply_k_vec(o, tables[fair], LEVELS[fair])
    nb = no_bid_price(p_fair, o["bid"].to_numpy(), margin)
    x = o.with_columns(pl.Series("fair", p_fair), pl.Series("nb", nb)).filter(pl.col("nb").is_not_nan())
    x = x.with_columns((1 - pl.col("nb")).alias("level"),
                       (((1 - pl.col("fair")) - pl.col("nb")) / pl.col("nb")).alias("priority"))
    may = x.filter(pl.col("hmax") > pl.col("level") + 1e-9).with_row_index("_oid")
    got = first_through(may, steps, ["ticker", "e"]).filter("filled").drop("_oid")
    got = _maker_trade_cols(got, "nb")
    if fills_only:
        return got.with_columns(pl.lit(True).alias("filled")), x.height
    if markouts:
        got = _markouts(got, q, "nb")
    rest = x.join(got.select("ticker", "e"), on=["ticker", "e"], how="anti").with_columns(
        pl.lit(False).alias("filled"), pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("fill_ts"))
    return pl.concat([got, rest], how="diagonal_relaxed")


class _LazySims(dict):
    """Per-variant sim frames read from disk on access (S1's are large): {variant: directory}."""

    def __getitem__(self, key):
        return pl.read_parquet(super().__getitem__(key) / "*.parquet")


SIM_COLS = ["t", "t_end", "filled", "t_fill", "settle", "event_ticker", "p1", "ft1", "mu1", "maker", "payoff",
            "priority", "fold"]


def s1_sim_frame(x: pl.DataFrame) -> pl.DataFrame:
    return x.select((pl.col("e") + POST_DELAY).alias("t"), (pl.col("e") + timedelta(hours=1)).alias("t_end"),
                    "filled", pl.col("fill_ts").alias("t_fill"), pl.col("settlement_ts").alias("settle"),
                    "event_ticker", pl.col("nb").alias("p1"), pl.col("fee_type").alias("ft1"),
                    pl.col("fee_multiplier").fill_null(1.0).alias("mu1"), pl.lit(True).alias("maker"),
                    (1 - pl.col("y")).cast(pl.Float64).alias("payoff"), "priority", "fold",
                    *_leg_cols(1, "ticker", "no", 1 - pl.col("y"), "settlement_ts"))


def _leg_cols(i: int, ticker: str, side: str, pay: pl.Expr, settle: str) -> list[pl.Expr]:
    """Ledger identity of leg i: the ticker, the side we hold, its unit payoff and settlement."""
    return [pl.col(ticker).alias(f"ticker{i}"), pl.lit(side).alias(f"side{i}"),
            pay.cast(pl.Float64).alias(f"pay{i}"), pl.col(settle).alias(f"settle{i}")]


S1_KEEP = ["ticker", "event_ticker", "series_ticker", "category", "kind", "e", "bid", "ask", "mid", "lead_b",
           "price_b", "fold", "hmax", "T", "close_time", "settlement_ts", "y", "fee_type", "fee_multiplier", "q_age_h"]


def run_s1(u, q, panel, prints, simdir: Path) -> dict[str, Any]:
    import shutil

    shutil.rmtree(simdir, ignore_errors=True)
    o_all = s1_orders(u, q)
    steps, hmax = hour_steps(prints)
    o_all = (o_all.join(hmax, on=["ticker", "e"], how="left").with_columns(pl.col("hmax").fill_null(0.0))
             .select(S1_KEEP))
    del hmax
    print(f"S1: {o_all.height} order-hours, {o_all['ticker'].n_unique()} markets, "
          f"{o_all.estimated_size('mb'):.0f} MB", flush=True)
    variants = [f"{f}_m{round(m * 100)}" for f in FAIRS for m in S1_MARGINS]
    grid, chosen, oos, test_filled = [], [], [], []
    for m in FOLDS:
        tr_tk = train_contracts(u, m)
        tables = fit_tables(panel, tr_tk)
        is_tr = pl.col("ticker").is_in(tr_tk.implode())
        tr_o, te_o = o_all.filter(is_tr), o_all.filter(pl.col("fold") == m)
        best = None
        for f in FAIRS:
            for mg in S1_MARGINS:
                v = f"{f}_m{round(mg * 100)}"
                tr, n_tr = s1_fold(tr_o, tables, steps, q, mg, f, fills_only=True)
                te = s1_fold(te_o, tables, steps, q, mg, f).with_columns(pl.lit(v).alias("variant"),
                                                                       pl.lit(m).alias("fold_eval"))
                st_tr, st_te = stats(tr, orders=n_tr), stats(te.filter("filled"), orders=te.height)
                del tr
                grid += [{"fold": m, "variant": v, "role": "train", **st_tr}, {"fold": m, "variant": v, "role": "test", **st_te}]
                test_filled.append(te.filter("filled"))
                if te.height:
                    vd = simdir / v
                    vd.mkdir(parents=True, exist_ok=True)
                    s1_sim_frame(te).write_parquet(vd / f"{m}.parquet")
                sc = score(st_tr)
                if sc is not None and (best is None or sc > best[1]):
                    best = (v, sc, f, mg)
        trade = best is not None and best[1] > 0
        chosen.append({"fold": m, "variant": best[0] if best else None, "train_score": best[1] if best else None,
                       "trades": trade, "k_global_hz": H._r(math.exp(tables["hz"][()][()]), 3),
                       "train_order_hours": tr_o.height, "test_order_hours": te_o.height})
        if trade:
            oos.append(s1_fold(te_o, tables, steps, q, best[3], best[2]).with_columns(
                pl.lit(best[0]).alias("variant"), pl.lit(m).alias("fold_eval")))
        print(f"  S1 fold {m}: train {tr_o.height} / test {te_o.height} order-hours, chosen "
              f"{best[:2] if best else None}", flush=True)
    del o_all
    sens = s1_sensitivity(pl.concat(oos, how="diagonal_relaxed") if oos else pl.DataFrame(), prints, steps)
    del steps
    sims = {v: simdir / v for v in variants if (simdir / v).exists()}
    res = _finish("S1", grid, chosen, oos, test_filled, variants, s1_sim_frame, variant_sims=_LazySims(sims))
    res["fill_rule_sensitivity"] = sens
    return res


def s1_sensitivity(oos: pl.DataFrame, prints: pl.DataFrame, steps: pl.DataFrame) -> list[dict[str, Any]]:
    """The chosen out-of-sample S1 orders re-filled under stricter rules: (a) the order goes live 5
    min after the quote instead of 60 s; (b) the print must be ≥ 1c through our level; (c) both."""
    if not oos.height:
        return []
    base = oos.drop("filled", "fill_ts", "pnl", "cost", "dollar_days", "fee_pc", "mk_1h", "mk_1d", "mk_settle",
                    strict=False)
    steps5 = hour_steps(prints.filter(pl.col("ticker").is_in(oos["ticker"].unique().implode())),
                        delay=timedelta(minutes=5))[0]
    out = [{"rule": "base (60 s, strictly through)", **stats(oos.filter("filled"), orders=oos.height)}]
    for name, st, extra in (("live after 5 min", steps5, 0.0), ("print >= 1c through", steps, 0.01),
                            ("both", steps5, 0.01)):
        x = base.with_columns((pl.col("level") + extra).alias("level")).with_row_index("_oid")
        f = first_through(x, st, ["ticker", "e"]).filter("filled").drop("_oid")
        f = _maker_trade_cols(f.with_columns((1 - pl.col("level") + extra).alias("nb")), "nb")
        out.append({"rule": name, **stats(f, orders=oos.height)})
    return out


REF_VARIANTS = {"S1": ("hz_m2", "hz_m8"), "S3t": ("L12_all", "L6_all"), "S3m": ("L12_all", "L6_all"),
                "S2": ("pool_th0",)}


LEDGER_DIR: Path | None = None   # set by analyze(): capped-book ledgers for vol_metrics


def _write_ledger(sm: dict[str, Any], name: str, variant: str) -> None:
    """Pop the capped book's frames and write its per-leg ledger (``vol_metrics`` schema)."""
    sm.pop("trades_frame", None)
    led = sm.pop("ledger_frame", None)
    if LEDGER_DIR is None or led is None or not led.height:
        return
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    split = pl.when(pl.col("fold") < "2026-10").then(pl.lit("dev")).when(pl.col("fold") == "2026-10") \
        .then(pl.lit("holdout")).otherwise(pl.lit("nov"))
    led.with_columns(pl.lit(name).alias("strategy"), pl.lit(variant).alias("variant"), split.alias("split")) \
        .write_parquet(LEDGER_DIR / f"ledger_{name}_{variant}.parquet")


def _finish(name: str, grid: list, chosen: list, oos: list, test_filled: list, variants: list[str],
            sim_frame, variant_sims: dict[str, pl.DataFrame] | None = None) -> dict[str, Any]:
    G = pl.DataFrame(grid, infer_schema_length=None)
    C = pl.DataFrame(chosen, infer_schema_length=None)
    O = pl.concat(oos, how="diagonal_relaxed") if oos else pl.DataFrame()
    TF = pl.concat([t for t in test_filled if t.height], how="diagonal_relaxed") if test_filled else pl.DataFrame()
    pooled = []
    for v in variants:
        te = G.filter((pl.col("variant") == v) & (pl.col("role") == "test"))
        orders = int(te["orders"].fill_null(0).sum()) if te.height else 0
        f = TF.filter(pl.col("variant") == v) if TF.height else TF
        st = stats(f, orders=orders) if f.height else {"orders": orders, "n": 0}
        if f.height:
            mo = f.group_by("fold").agg(pl.col("pnl").sum().alias("p"))
            st["months_positive"], st["months"] = int((mo["p"] > 0).sum()), mo.height
        if variant_sims is not None and v in variant_sims:
            sm = simulate(variant_sims[v], return_trades=LEDGER_DIR is not None)
            _write_ledger(sm, name, v)
            st.update({"sim_usd_per_week": sm.get("usd_per_week"), "sim_max_dd_usd": sm.get("max_drawdown_usd"),
                       "sim_months_positive": sm.get("months_positive"), "sim_trades": sm.get("trades")})
        pooled.append({"variant": v, **st})
    P = pl.DataFrame(pooled, infer_schema_length=None)
    res = {"grid": G, "chosen": C, "oos": O, "pooled": P}
    for rv in REF_VARIANTS.get(name, ()):
        f = TF.filter(pl.col("variant") == rv) if TF.height else TF
        for by in ("category", "band", "kind"):
            if f.height:
                res[f"ref_by_{by}_{rv}"] = breakdown(f, by)
    if O.height:
        res["oos_stats"] = stats(O.filter("filled"), orders=O.height)
        res["monthly"] = monthly(O)
        res["sim"] = simulate(sim_frame(O), return_trades=LEDGER_DIR is not None)
        _write_ledger(res["sim"], name, "wf")
    else:
        res["oos_stats"], res["monthly"], res["sim"] = {"n": 0}, pl.DataFrame(), {"trades": 0}
    print(f"{name}: OOS {res['oos_stats']}", flush=True)
    print(f"{name}: sim {({k: v for k, v in res['sim'].items() if k != 'by_month'})}", flush=True)
    return res


# ---- S3: final-hours decay -----------------------------------------------------------------------
def s3_rows(u: pl.DataFrame, q: pl.DataFrame) -> pl.DataFrame:
    """Decision rows at T − L (quote ≤ 2 h old at the decision, executed 60 s later), with the
    72 h anchor and the flat-hazard theta path from it."""
    m = u.select(META)
    g = m.join(pl.DataFrame({"L": list(S3_LEADS)}), how="cross").with_columns(
        (pl.col("T") - pl.duration(hours=pl.col("L"))).alias("t"),
        (pl.col("T") - timedelta(hours=S3_ANCHOR)).alias("tA"))
    g = g.filter((pl.col("t") >= H.START) & (pl.col("t") >= pl.col("open_time") + timedelta(hours=1))
                 & (pl.col("t") < pl.col("close_time")))
    g = quote_at(g, q, "t")
    g = _lead_cols(g, "t").filter((pl.col("spread") <= H.MAX_SPREAD) & pl.col("mid").is_between(0.02, S3_MAX_MID))
    a = quote_at(g.select("ticker", "L", "tA").filter(pl.col("tA") >= H.START), q, "tA", suffix="_A")
    a = a.select("ticker", "L", ((pl.col("bid_A") + pl.col("ask_A")) / 2).alias("pA"))
    g = g.join(a, on=["ticker", "L"], how="left").with_columns(
        ((pl.col("T") - pl.max_horizontal("tA", pl.col("w_start").fill_null(pl.col("tA")))).dt.total_seconds() / 86400)
        .alias("tauA"))
    ok = g["pA"].is_not_null().to_numpy() & (g["tauA"].to_numpy() > 0)
    path = np.where(ok, theta_path(g["pA"].fill_null(0.5).to_numpy(), np.where(ok, g["tauA"].to_numpy(), 1.0),
                                   np.maximum(g["tau_d"].to_numpy(), 0)), np.nan)
    g = g.with_columns(pl.Series("p_path", path)).with_columns((pl.col("mid") - pl.col("p_path")).alias("gap"),
                                                                fold_col("t"),
                                                                ((pl.col("t") - pl.col("q_ts")).dt.total_hours()).alias("q_age_h"))
    return g.drop("q_ts")


def s3_select(g: pl.DataFrame, L: int, gap: float | None) -> pl.DataFrame:
    x = g.filter(pl.col("L") == L)
    if gap is not None:
        x = x.filter(pl.col("gap").is_not_nan() & (pl.col("gap") >= gap - 1e-9))
    return x


def s3_taker(x: pl.DataFrame, q: pl.DataFrame) -> pl.DataFrame:
    x = x.filter(pl.col("bid") >= 0.01).with_columns((1 - pl.col("bid")).alias("px_in"),
                                                     (pl.col("t") + POST_DELAY).alias("fill_ts"))
    pin = x["px_in"].to_numpy()
    fee = fee_per_contract(pin, order_size(pin), mult=x["fee_multiplier"].fill_null(1.0).to_numpy())
    x = x.with_columns(pl.Series("fee_pc", fee), pl.lit(True).alias("filled")).with_columns(
        ((1 - pl.col("y")) - pl.col("px_in") - pl.col("fee_pc")).alias("pnl"),
        (pl.col("px_in") + pl.col("fee_pc")).alias("cost")).with_columns(
        (pl.col("cost") * ((pl.col("settlement_ts") - pl.col("fill_ts")).dt.total_seconds() / 86400)
         .clip(lower_bound=1 / 24)).alias("dollar_days"))
    return _markouts(x, q, "px_in")


def s3_maker(x: pl.DataFrame, u: pl.DataFrame, q: pl.DataFrame, steps: pl.DataFrame) -> pl.DataFrame:
    """From the decision T − L, rest a NO bid pegged one cent above the best NO bid (1 − YES ask),
    capped one cent inside the NO ask, re-pegged every hour e (live (e + 60 s, e + 1 h], prints
    after T − 10 min excluded) while that hour's quote has mid ≤ 50c and spread ≤ 10c; the first
    print strictly through the hour's level is the (single) entry. Held to settlement."""
    L_of = x.select("ticker", "L", pl.col("t").alias("t_dec"), "mid", "gap", "p_path", "fold")
    hrs = (L_of.join(u.select("ticker", "T", "close_time"), on="ticker")
           .with_columns(pl.datetime_ranges(pl.col("t_dec"),
                                            pl.min_horizontal(pl.col("T") - timedelta(hours=1),
                                                              pl.col("close_time") - timedelta(seconds=1)),
                                            interval="1h").alias("e"))
           .explode("e").filter(pl.col("e").is_not_null()).select("ticker", "L", "t_dec", "e"))
    hq = quote_at(hrs, q, "e", max_age=timedelta(hours=6)).with_columns(
        ((pl.col("bid") + pl.col("ask")) / 2).alias("mid_e"), (pl.col("ask") - pl.col("bid")).alias("sp_e"))
    hq = hq.filter((pl.col("mid_e") <= S3_MAX_MID) & (pl.col("sp_e") <= H.MAX_SPREAD))
    nb = np.minimum(whole_cents_down(1 - hq["ask"].to_numpy() + 0.01), whole_cents_down(1 - hq["bid"].to_numpy() - 0.01))
    hq = hq.with_columns(pl.Series("nb", np.where(nb >= 0.01, nb, np.nan))).filter(pl.col("nb").is_not_nan())
    hq = hq.with_columns((1 - pl.col("nb")).alias("level")).with_row_index("_oid")
    got = first_through(hq, steps, ["ticker", "e"]).filter("filled")
    first = got.sort("fill_ts").unique(["ticker", "L"], keep="first")
    nb0 = np.minimum(whole_cents_down(1 - x["ask"].to_numpy() + 0.01), whole_cents_down(1 - x["bid"].to_numpy() - 0.01))
    x = x.with_columns(pl.Series("nb0", np.clip(nb0, 0.01, 0.99)))
    out = (x.join(first.select("ticker", "L", "nb", "fill_ts", "e"), on=["ticker", "L"], how="left")
           .with_columns(pl.col("fill_ts").is_not_null().alias("filled"),
                         pl.min_horizontal(pl.col("T") - END_BEFORE, "close_time").alias("t1")))
    f = out.filter("filled")
    f = _markouts(_maker_trade_cols(f, "nb"), q, "nb")
    return pl.concat([f, out.filter(~pl.col("filled"))], how="diagonal_relaxed")


def s3m_steps(prints: pl.DataFrame, u: pl.DataFrame) -> pl.DataFrame:
    pr = prints.join(u.select("ticker", "T"), on="ticker").filter(
        (pl.col("ts") >= pl.col("T") - timedelta(hours=max(S3_LEADS) + 1)) & (pl.col("ts") <= pl.col("T") - END_BEFORE))
    return hour_steps(pr.drop("T"))[0]


def s3_sim_frame(maker: bool):
    def fr(x: pl.DataFrame) -> pl.DataFrame:
        px = pl.coalesce("nb", "nb0") if maker else pl.col("px_in")
        return x.select((pl.col("t") + POST_DELAY).alias("t"),
                        (pl.col("t1") if maker else pl.col("t") + POST_DELAY).alias("t_end"),
                        "filled", pl.col("fill_ts").alias("t_fill"), pl.col("settlement_ts").alias("settle"),
                        "event_ticker", px.alias("p1"), pl.col("fee_type").alias("ft1"),
                        pl.col("fee_multiplier").fill_null(1.0).alias("mu1"), pl.lit(maker).alias("maker"),
                        (1 - pl.col("y")).cast(pl.Float64).alias("payoff"),
                        pl.col("gap").fill_nan(0.0).fill_null(0.0).alias("priority"), "fold",
                        *_leg_cols(1, "ticker", "no", 1 - pl.col("y"), "settlement_ts"))
    return fr


def run_s3(u, q, prints, maker: bool) -> dict[str, Any]:
    g = s3_rows(u, q)
    name = "S3m" if maker else "S3t"
    print(f"{name}: {g.height} decision rows, {g['ticker'].n_unique()} markets", flush=True)
    variants, built = [], {}
    steps = s3m_steps(prints, u) if maker else None
    for L in S3_LEADS:
        for gp in S3_GAPS:
            v = f"L{L}_" + ("all" if gp is None else f"gap{round(gp * 100)}")
            variants.append(v)
            x = s3_select(g, L, gp)
            built[v] = (s3_maker(x, u, q, steps) if maker else s3_taker(x, q)).with_columns(pl.lit(v).alias("variant"))
    grid, chosen, oos, test_filled, test_all = [], [], [], [], {}
    for m in FOLDS:
        tr_tk = train_contracts(u, m)
        best, keep = None, {}
        for v in variants:
            x = built[v]
            tr = x.filter(pl.col("ticker").is_in(tr_tk.implode()))
            te = x.filter(pl.col("fold") == m).with_columns(pl.lit(m).alias("fold_eval"))
            st_tr, st_te = stats(tr.filter("filled"), orders=tr.height), stats(te.filter("filled"), orders=te.height)
            grid += [{"fold": m, "variant": v, "role": "train", **st_tr}, {"fold": m, "variant": v, "role": "test", **st_te}]
            test_filled.append(te.filter("filled"))
            test_all.setdefault(v, []).append(te)
            keep[v] = te
            sc = score(st_tr)
            if sc is not None and (best is None or sc > best[1]):
                best = (v, sc)
        trade = best is not None and best[1] > 0
        chosen.append({"fold": m, "variant": best[0] if best else None, "train_score": best[1] if best else None,
                       "trades": trade})
        if trade:
            oos.append(keep[best[0]])
    sims = {v: s3_sim_frame(maker)(pl.concat(fr, how="diagonal_relaxed")) for v, fr in test_all.items()}
    return _finish(name, grid, chosen, oos, test_filled, variants, s3_sim_frame(maker), variant_sims=sims)


# ---- S2: forward-hazard calendar -----------------------------------------------------------------
KF_GROUPS = ("Politics", "Mentions")


def s2_rows(u: pl.DataFrame, q: pl.DataFrame) -> pl.DataFrame:
    """Short-forward decisions at the near leg's grid (leads ≤ S2_MAX_LEAD): both legs open and
    quoted (≤ 2 h old, spread ≤ 10c), windows nested, near mid ≤ 50c and far mid ≤ 90c
    (outcome-blind 'no event yet' filter). Taker on both legs, fees at the $5-cap order size."""
    pairs = H.strip_pairs(u)
    extra = u.select("ticker", "series_ticker", "category", "kind", "settlement_ts", "fee_type")
    pairs = pairs.join(extra, on="ticker").join(
        u.select(pl.col("ticker").alias("ticker_2"), pl.col("settlement_ts").alias("settlement_ts_2"),
                 pl.col("fee_type").alias("fee_type_2")), on="ticker_2")
    leads = [L for L in H.LEADS_H if L <= S2_MAX_LEAD]
    g = pairs.join(pl.DataFrame({"L": leads}), how="cross").with_columns(
        (pl.col("T") - pl.duration(hours=pl.col("L"))).alias("t"))
    g = g.filter((pl.col("t") >= H.START) & (pl.col("t") >= pl.col("open_time") + timedelta(hours=1))
                 & (pl.col("t") >= pl.col("open_time_2") + timedelta(hours=1))
                 & (pl.col("t") < pl.col("close_time")) & (pl.col("t") < pl.col("close_time_2"))
                 & (pl.col("w_start_2").is_null()
                    | (pl.col("w_start_2") <= pl.max_horizontal("t", pl.col("w_start").fill_null(pl.col("t"))))))
    g = quote_at(g, q, "t")
    g = quote_at(g, q, "t", ticker_col="ticker_2", suffix="_2")
    g = _lead_cols(g, "t").with_columns(((pl.col("bid_2") + pl.col("ask_2")) / 2).alias("mid_2"))
    g = g.filter((pl.col("spread") <= H.MAX_SPREAD) & ((pl.col("ask_2") - pl.col("bid_2")) <= H.MAX_SPREAD)
                 & (pl.col("mid") <= 0.5) & (pl.col("mid_2") <= 0.9) & (pl.col("bid_2") >= 0.01) & (pl.col("ask") < 1))
    p1, p2 = g["ask"].to_numpy(), 1 - g["bid_2"].to_numpy()
    n = order_size(p1 + p2)
    f1 = fee_per_contract(p1, n, mult=g["fee_multiplier"].fill_null(1.0).to_numpy())
    f2 = fee_per_contract(p2, n, mult=g["fee_multiplier_2"].fill_null(1.0).to_numpy())
    y1, y2 = g["y"].to_numpy().astype(float), g["y_2"].to_numpy().astype(float)
    t_s = g["t"].dt.epoch("s").to_numpy()
    s1_ = g["settlement_ts"].dt.epoch("s").to_numpy()
    s2_ = g["settlement_ts_2"].dt.epoch("s").to_numpy()
    dd = ((p1 + f1) * np.maximum(s1_ - t_s - 60, 3600) + (p2 + f2) * np.maximum(s2_ - t_s - 60, 3600)) / 86400
    cost = p1 + p2 + f1 + f2
    g = g.with_columns(pl.Series("p1", p1), pl.Series("p2", p2), pl.Series("cost", cost),
                       pl.Series("pnl", y1 + 1 - y2 - cost), pl.Series("dollar_days", dd),
                       pl.Series("pnl_far_only", (1 - y2) - p2 - fee_per_contract(p2, order_size(p2))),
                       pl.Series("r", np.clip((1 - g["mid_2"].to_numpy()) / (1 - g["mid"].to_numpy()), 1e-3, 1.0)),
                       pl.when(pl.col("category").is_in(list(KF_GROUPS))).then(pl.col("category"))
                       .otherwise(pl.lit("Other")).alias("kf_group"),
                       pl.lit(True).alias("filled"), (pl.col("t") + POST_DELAY).alias("fill_ts"),
                       fold_col("t"),
                       pl.max_horizontal((pl.col("t") - pl.col("q_ts")).dt.total_hours(),
                                         (pl.col("t") - pl.col("q_ts_2")).dt.total_hours()).alias("q_age_h"))
    return g.drop("q_ts", "q_ts_2")


def fit_kf(tr: pl.DataFrame, n0: float = 50.0) -> dict[str, float]:
    """Forward-hazard multiplier: Σ[1 − r^{k_f}] = Σ 1[far YES] over training decisions with no near
    event, pooled ('_pool') and by group (shrunk in log toward the pooled value)."""
    x = tr.filter(pl.col("y") == 0)
    if x.height < 10:
        return {"_pool": 1.0}
    pool = H.hazard_multiplier(1 - x["r"].to_numpy(), x["y_2"].to_numpy())
    out = {"_pool": pool}
    for (grp,), g in x.group_by("kf_group"):
        k = H.hazard_multiplier(1 - g["r"].to_numpy(), g["y_2"].to_numpy())
        out[grp] = math.exp((g.height * math.log(k) + n0 * math.log(pool)) / (g.height + n0))
    return out


def run_s2(u, q, panel) -> dict[str, Any]:
    g = s2_rows(u, q)
    print(f"S2: {g.height} pair decisions, {g['ticker'].n_unique()} pairs, {g['skey'].n_unique()} strips", flush=True)
    variants = [f"{mode}_th{round(th * 100)}" for mode in ("pool", "cat") for th in S2_THETAS]
    grid, chosen, oos, test_filled, test_all = [], [], [], [], {}
    for m in FOLDS:
        t0 = fold_start(m)
        tr_tk = train_contracts(u, m)
        tables = fit_tables(panel, tr_tk)
        tr_mask = pl.col("ticker").is_in(tr_tk.implode()) & pl.col("ticker_2").is_in(tr_tk.implode())
        sub = g.filter(tr_mask | (pl.col("fold") == m)).with_columns(
            pl.when(tr_mask).then(pl.lit("train")).otherwise(pl.lit("test")).alias("role"))
        if not sub.height:
            continue
        kf = fit_kf(sub.filter(pl.col("role") == "train"))
        k1 = k_of(sub, tables["hz"], LEVELS["hz"])
        best, keep = None, {}
        for mode in ("pool", "cat"):
            kfv = (np.full(sub.height, kf["_pool"]) if mode == "pool"
                   else np.array([kf.get(x, kf["_pool"]) for x in sub["kf_group"].to_list()]))
            val = short_forward_value(sub["mid"].to_numpy(), sub["mid_2"].to_numpy(), k1, kfv)
            sv = sub.with_columns(pl.Series("value", val)).with_columns((pl.col("value") - pl.col("cost")).alias("edge"))
            for th in S2_THETAS:
                v = f"{mode}_th{round(th * 100)}"
                x = sv.filter(pl.col("edge") >= th).with_columns(pl.lit(v).alias("variant"), pl.lit(m).alias("fold_eval"))
                tr, te = x.filter(pl.col("role") == "train"), x.filter(pl.col("role") == "test")
                st_tr, st_te = stats(tr, orders=tr.height), stats(te, orders=te.height)
                grid += [{"fold": m, "variant": v, "role": "train", **st_tr}, {"fold": m, "variant": v, "role": "test", **st_te}]
                test_filled.append(te)
                test_all.setdefault(v, []).append(te)
                keep[v] = te
                sc = score(st_tr)
                if sc is not None and (best is None or sc > best[1]):
                    best = (v, sc)
        trade = best is not None and best[1] > 0
        chosen.append({"fold": m, "variant": best[0] if best else None, "train_score": best[1] if best else None,
                       "trades": trade, "kf_pool": H._r(kf["_pool"], 3),
                       **{f"kf_{k}": H._r(v, 3) for k, v in kf.items() if k != "_pool"}})
        if trade:
            oos.append(keep[best[0]])
        del t0
    sims = {v: s2_sim_frame(pl.concat(fr, how="diagonal_relaxed")) for v, fr in test_all.items()}
    return _finish("S2", grid, chosen, oos, test_filled, variants, s2_sim_frame, variant_sims=sims)


def s2_sim_frame(x: pl.DataFrame) -> pl.DataFrame:
    return x.select((pl.col("t") + POST_DELAY).alias("t"), (pl.col("t") + POST_DELAY).alias("t_end"), "filled",
                    pl.col("fill_ts").alias("t_fill"),
                    pl.max_horizontal("settlement_ts", "settlement_ts_2").alias("settle"), "event_ticker",
                    "p1", pl.col("fee_type").alias("ft1"), pl.col("fee_multiplier").fill_null(1.0).alias("mu1"),
                    "p2", pl.col("fee_type_2").alias("ft2"), pl.col("fee_multiplier_2").fill_null(1.0).alias("mu2"),
                    pl.lit(False).alias("maker"), (pl.col("y") + 1 - pl.col("y_2")).cast(pl.Float64).alias("payoff"),
                    pl.col("edge").alias("priority"), "fold",
                    *_leg_cols(1, "ticker", "yes", pl.col("y"), "settlement_ts"),
                    *_leg_cols(2, "ticker_2", "no", 1 - pl.col("y_2"), "settlement_ts_2"))


# ---- breakdowns and driver -----------------------------------------------------------------------
def band_expr() -> pl.Expr:
    return (pl.when(pl.col("mid") < 0.15).then(pl.lit("02-15c")).when(pl.col("mid") < 0.5).then(pl.lit("15-50c"))
            .otherwise(pl.lit("50-60c")).alias("band"))


def breakdown(oos: pl.DataFrame, by: str) -> pl.DataFrame:
    if not oos.height:
        return pl.DataFrame()
    x = oos.with_columns(band_expr()) if by == "band" else oos
    rows = []
    for (k,), g in x.group_by(by):
        rows.append({by: k, **stats(g.filter("filled"), orders=g.height)})
    return pl.DataFrame(rows, infer_schema_length=None).sort(by)


def _jsonable(res: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in res.items():
        if isinstance(v, pl.DataFrame):
            if k in ("oos",):
                continue
            out[k] = v.to_dicts()
        else:
            out[k] = v
    return out


EXEC = {"S1": "maker: hourly NO bid at floor(1 − fair − m), trade-through fills, hold to settlement",
        "S2": "taker both legs (YES near at ask, NO far at 1 − bid), hold to settlement",
        "S3t": "taker: NO at 1 − bid at T − L, hold to settlement",
        "S3m": "maker: NO bid pegged 1c above best NO bid from T − L, re-pegged hourly, 1 entry"}


def headline(name: str, res: dict[str, Any]) -> dict[str, Any]:
    """One row of the verdict table: train-chosen variants, in-sample (training set of the last
    regular fold, 2026-10) and pooled walk-forward out-of-sample stats, capped-sim $/week."""
    ch = res.get("chosen") or []
    g = res.get("grid") or []
    used = [c["variant"] for c in ch if c.get("trades")]
    last = next((c for c in ch if c["fold"] == "2026-10"), None)
    ins = next((r for r in g if last and r["fold"] == "2026-10" and r["role"] == "train"
                and r["variant"] == last.get("variant")), {})
    o = res.get("oos_stats") or {}
    sm = res.get("sim") or {}
    mo = res.get("monthly") or []
    return {"strategy": name, "execution": EXEC[name],
            "chosen_variants": ", ".join(f"{c['fold'][2:]}:{c['variant'] if c.get('trades') else '-'}" for c in ch),
            "chosen_last": last.get("variant") if last else None,
            "is_n": ins.get("n"), "is_events": ins.get("events"), "is_c": ins.get("pnl_c"), "is_t": ins.get("t"),
            "oos_n": o.get("n"), "oos_events": o.get("events"), "oos_c": o.get("pnl_c"), "oos_t": o.get("t"),
            "oos_t_series": o.get("t_series"), "per_dollar": o.get("per_dollar"),
            "per_dollar_day": o.get("per_dollar_day"), "fill_rate": o.get("fill_rate"),
            "mk_1h_c": o.get("mk_1h_c"), "mk_1d_c": o.get("mk_1d_c"),
            "oos_months_pos": f"{sum(1 for m in mo if (m.get('pnl_sum_c') or 0) > 0)}/{len(mo)}",
            "folds_traded": len(used),
            "sim_usd_week": sm.get("usd_per_week"), "sim_usd_week_ex_top5": sm.get("usd_per_week_ex_top5"),
            "sim_t_event": sm.get("t_event"), "sim_pnl_usd": sm.get("pnl_usd"),
            "sim_max_dd_usd": sm.get("max_drawdown_usd"), "sim_mean_open_usd": sm.get("mean_open_usd"),
            "sim_months_pos": f"{sm.get('months_positive')}/{sm.get('months')}"}


def write_results(d: Path, summary: dict[str, Any]) -> Path:
    from albert.reports import md_table

    lines = ["# vol_theta results (generated by `python -m albert.research.vol_theta run analyze`)", "",
             "Universe: " + json.dumps(summary["universe"]), ""]
    heads = [headline(k, summary[k]) for k in ("S1", "S2", "S3t", "S3m") if k in summary]
    lines += ["## Headline (walk-forward, train-chosen variant per fold)", "",
              md_table(pl.DataFrame(heads, infer_schema_length=None).drop("execution", "chosen_variants"), max_rows=10), ""]
    for h in heads:
        lines += [f"- **{h['strategy']}** ({h['execution']}): chosen per fold {h['chosen_variants']}"]
    lines.append("")
    for k in ("S1", "S2", "S3t", "S3m"):
        if k not in summary:
            continue
        r = summary[k]
        lines += [f"## {k}", "", "Out-of-sample by month (chosen variant):", "",
                  md_table(pl.DataFrame(r["monthly"], infer_schema_length=None), max_rows=20) if r.get("monthly") else "(none)",
                  "", "Capped bankroll simulation ($5/event, $20 open incl. resting orders), by decision month:", "",
                  md_table(pl.DataFrame(r["sim"].get("by_month") or [], infer_schema_length=None), max_rows=20)
                  if r["sim"].get("by_month") else "(none)", "",
                  ("Full grid, pooled out-of-sample (each fold priced with its own training fit; ex post, not a "
                   "selection):"), "", md_table(pl.DataFrame(r["pooled"], infer_schema_length=None), max_rows=20), "",
                  "Chosen per fold:", "", md_table(pl.DataFrame(r["chosen"], infer_schema_length=None), max_rows=20), ""]
        if r.get("fill_rule_sensitivity"):
            lines += ["Fill-rule sensitivity of the chosen out-of-sample orders:", "",
                      md_table(pl.DataFrame(r["fill_rule_sensitivity"], infer_schema_length=None)), ""]
        if r.get("oos_fresh_quotes"):
            lines += [f"Fresh quotes only (candle closed in the decision hour; {r.get('oos_fresh_share')} of orders): "
                      + json.dumps(r["oos_fresh_quotes"]), ""]
        for by in ("category", "band", "kind"):
            if r.get(f"by_{by}"):
                lines += [f"OOS (chosen) by {by}:", "", md_table(pl.DataFrame(r[f"by_{by}"], infer_schema_length=None)), ""]
        for key in [x for x in r if x.startswith("ref_by_")]:
            lines += [f"Reference variant {key[7:]}:", "", md_table(pl.DataFrame(r[key], infer_schema_length=None)), ""]
    path = d / "results.md"
    path.write_text("\n".join(lines))
    return path


def analyze(s) -> None:
    global LEDGER_DIR
    d = out_dir(s)
    LEDGER_DIR = d / "ledgers"
    u = load_universe(s)
    c = load_candles(s, u["ticker"])
    q = quote_frame(c)
    del c
    panel = pl.read_parquet(H.out_dir(s) / "panel.parquet")
    done = set(pl.read_parquet(d / "trades_done.parquet")["ticker"])
    plan = pl.read_parquet(d / "trade_plan.parquet")
    missing = plan.filter(~pl.col("ticker").is_in(list(done)))
    print(f"trades: {len(done)} of {plan.height} planned windows fetched; {missing.height} missing", flush=True)
    # markets whose trades we do not have cannot be evaluated as makers: drop them from the maker tests
    u_mk = u.filter(pl.col("ticker").is_in(list(done)))
    prints = taker_yes_prints(s, u_mk["ticker"])
    print(f"taker-YES prints: {prints.height}", flush=True)
    summary: dict[str, Any] = {"universe": {"markets": u.height, "events": u["event_ticker"].n_unique(),
                                            "ext_markets": u.filter("ext").height,
                                            "trade_windows": plan.height, "trade_windows_fetched": len(done),
                                            "taker_yes_prints": prints.height}}
    import resource

    def mem(tag: str) -> None:
        print(f"[mem] {tag}: peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)

    mem("loaded")
    results = {}
    results["S1"] = run_s1(u_mk, q, panel, prints, d / "s1_sim")
    mem("S1")
    results["S3t"] = run_s3(u, q, prints, maker=False)
    results["S3m"] = run_s3(u_mk, q, prints, maker=True)
    mem("S3")
    results["S2"] = run_s2(u, q, panel)
    mem("S2")
    for name, res in results.items():
        res["grid"].write_parquet(d / f"grid_{name}.parquet")
        res["grid"].write_csv(d / f"grid_{name}.csv")
        res["pooled"].write_csv(d / f"grid_pooled_{name}.csv")
        if res["oos"].height:
            res["oos"].write_parquet(d / f"oos_{name}.parquet")
        for by in ("category", "band", "kind"):
            if res["oos"].height:
                res[f"by_{by}"] = breakdown(res["oos"], by)
        if res["oos"].height and "q_age_h" in res["oos"].columns:
            fr = res["oos"].filter(pl.col("q_age_h") == 0)
            res["oos_fresh_quotes"] = stats(fr.filter("filled"), orders=fr.height)
            res["oos_fresh_share"] = H._r(fr.height / res["oos"].height, 3)
        summary[name] = _jsonable(res)
    (d / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"wrote {d / 'summary.json'} and {write_results(d, json.loads(json.dumps(summary, default=str)))}",
          flush=True)


# ---- entry point -------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    from albert.config import settings

    ap = argparse.ArgumentParser(prog="albert.research.vol_theta")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("step", choices=["plan", "candles", "trades", "analyze"])
    a = ap.parse_args(argv)
    s = settings()
    u = load_universe(s)
    if a.step == "candles":
        fetch_ext_candles(s, u)
    if a.step in ("plan", "trades"):
        plan = trade_plan(s, u)
        _save(plan, out_dir(s) / "trade_plan.parquet")
        print(f"plan: {plan.height} windows, {plan['event_ticker'].n_unique()} events, "
              f"{u.filter('ext').height} extension markets", flush=True)
        if a.step == "trades":
            fetch_trades(s, plan)
    if a.step == "analyze":
        analyze(s)


if __name__ == "__main__":
    main()
