"""Shared harness for fundamental-model studies: does a model of the *underlying event* beat the
Kalshi price, out of sample and after costs?

Every study produces one frame of predictions, one row per (market, decision time):

    ticker, event, t (UTC decision time), p (model probability of YES, known at ``t``)

and calls :func:`evaluate`, which attaches the executable quote at ``t`` (the last 60-minute
candle that *ended* at or before ``t``: its closing yes bid / yes ask), the settled result, and
reports, per split (dev / holdout):

* **Scores:** Brier and log loss of the model, of the market mid, and of a logistic blend
  ``logit P = a + b·logit(mid) + c·logit(p)`` fitted on dev only. ``c`` (t clustered by event)
  is the test that matters: does the model add information the price lacks? A model can lose
  to the market on Brier and still add information (c > 0), or beat a stale mid and add none.
* **Taker:** buy YES at the ask when ``p − ask − fee ≥ θ``, or NO at ``1 − bid`` when
  ``(1 − p) − (1 − bid) − fee ≥ θ``; one contract, held to settlement, Kalshi taker fee
  0.07·P·(1−P) (exact to $0.0001, this account's precision). θ is chosen on dev from a fixed
  grid; holdout reports that θ. Mean PnL per contract and t clustered by event.

Rules this harness enforces or makes visible (AGENTS.md): no look-ahead (quotes strictly from
candles that closed by ``t``; the model must only use data timestamped before ``t``, which the
study itself must guarantee), executable prices only (never the mid for PnL), and volume-blind
samples (:func:`settled_markets` takes every market of a series, never a volume-ranked subset).

Candles are cached per study under ``data/cache/fundamental_<name>_candles.parquet``; studies
never write the shared ``data/tables`` (concurrent read-merge-write would drop rows).
"""

import math
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

THETAS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)


def taker_fee(p: float, mult: float = 1.0) -> float:
    """Kalshi taker fee for one contract (0.07 × series multiplier × P(1−P)), to $0.0001."""
    return math.ceil(0.07 * mult * p * (1 - p) * 1e4 - 1e-9) / 1e4


def maker_fee(p: float, fee_type: str | None, mult: float = 1.0) -> float:
    """Kalshi maker fee for one contract: 0.0175 × multiplier × P(1−P) on maker-fee series only."""
    if not fee_type or "maker_fee" not in fee_type:
        return 0.0
    return math.ceil(0.0175 * mult * p * (1 - p) * 1e4 - 1e-9) / 1e4


# ---- data ------------------------------------------------------------------------------------
def settled_markets(s, series: list[str], start: datetime | None = None,
                    end: datetime | None = None) -> pl.DataFrame:
    """Every settled market of these series (volume-blind) from the ingested markets table, with
    ``y`` (1 = YES won). Markets settled without a yes/no result are dropped."""
    m = pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet").filter(pl.col("series_ticker").is_in(series))
    if start is not None:
        m = m.filter(pl.col("close_time") >= start)
    if end is not None:
        m = m.filter(pl.col("close_time") < end)
    m = m.filter(pl.col("result").is_in(["yes", "no"])).with_columns((pl.col("result") == "yes").cast(pl.Int8).alias("y"))
    return m.collect()


def fetch_candles(s, markets: pl.DataFrame, name: str, *, period_minutes: int = 60,
                  lookback: timedelta = timedelta(days=3), workers: int = 4) -> pl.DataFrame:
    """Candles for ``markets`` (needs ticker, open_time, close_time), cached per study. Each
    market's window is its last ``lookback`` before close (and at most ~4,900 periods)."""
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    path = s.data_dir / "cache" / f"fundamental_{name}_candles.parquet"
    have = pl.read_parquet(path) if path.exists() else None
    done = set(have["ticker"].unique()) if have is not None and have.height else set()
    todo = [m for m in markets.select("ticker", "open_time", "close_time").iter_rows(named=True)
            if m["ticker"] not in done]
    if not todo:
        return have
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.1))
    cutoff = api.cutoff().get("market_settled_ts")

    def one(m: dict[str, Any]) -> list[dict[str, Any]]:
        hi = m["close_time"]
        lo = max(m["open_time"] or hi - lookback, hi - lookback,
                 hi - timedelta(minutes=period_minutes * 4900))
        historical = bool(cutoff and hi < cutoff)
        try:
            page = api.candles(m["ticker"], lo, hi, period_minutes, historical)
        except RuntimeError as error:
            print(f"skip candles {m['ticker']}: {str(error)[:120]}")
            return []
        return k.candle_rows(page.payload, m["ticker"], period_minutes)

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(one, todo), 1):
            rows.extend(r)
            if i % 200 == 0:
                print(f"  candles {i}/{len(todo)}", flush=True)
    new = pl.DataFrame(rows, infer_schema_length=None) if rows else None
    frames = [f for f in (have, new) if f is not None and f.height]
    out = pl.concat(frames, how="diagonal_relaxed").unique(["ticker", "end_ts", "period_minutes"]) if frames else pl.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    out.write_parquet(tmp)
    tmp.replace(path)
    return out


def attach_quotes(pred: pl.DataFrame, candles: pl.DataFrame, *, max_age: timedelta = timedelta(hours=2)) -> pl.DataFrame:
    """Join the last candle that ended at or before ``t`` (within ``max_age``): yes bid / ask."""
    c = (candles.select("ticker", pl.col("end_ts").alias("q_ts"), pl.col("yes_bid_close").alias("bid"),
                        pl.col("yes_ask_close").alias("ask"))
         .filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("ask") > pl.col("bid")))
         .sort("ticker", "q_ts"))
    out = pred.sort("ticker", "t").join_asof(c, left_on="t", right_on="q_ts", by="ticker", strategy="backward",
                                                           check_sortedness=False)
    return out.filter(pl.col("q_ts").is_not_null() & ((pl.col("t") - pl.col("q_ts")) <= max_age))


# ---- statistics ------------------------------------------------------------------------------
def clustered_mean(x: np.ndarray, g: np.ndarray) -> tuple[float, float | None, int]:
    """Mean, t-stat with standard error clustered by ``g``, number of clusters."""
    if len(x) == 0:
        return float("nan"), None, 0
    mu = float(x.mean())
    _, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=x - mu)
    k = len(sums)
    if k < 2:
        return mu, None, k
    se = math.sqrt((sums ** 2).sum() * k / (k - 1)) / len(x)
    return mu, (mu / se if se > 0 else None), k


def logit(p: np.ndarray, eps: float = 0.005) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_logistic(X: np.ndarray, y: np.ndarray, g: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression by Newton steps; returns coefficients and cluster-robust t-stats."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ beta))
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    mu = 1 / (1 + np.exp(-X @ beta))
    Hinv = np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))
    _, inv = np.unique(g, return_inverse=True)
    S = np.zeros((inv.max() + 1, X.shape[1]))
    np.add.at(S, inv, X * (y - mu)[:, None])
    V = Hinv @ (S.T @ S) @ Hinv
    return beta, beta / np.sqrt(np.clip(np.diag(V), 1e-18, None))


def _scores(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    q = np.clip(p, 0.005, 0.995)
    return float(((p - y) ** 2).mean()), float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def taker_pnl(df: pl.DataFrame, theta: float) -> pl.DataFrame:
    """One contract per row where the model's edge after the taker fee is at least ``theta``. A
    ``fee_multiplier`` column (from :func:`settled_markets`) applies the series' own fee."""
    rows = []
    for r in df.iter_rows(named=True):
        p, bid, ask, y = r["p"], r["bid"], r["ask"], r["y"]
        mult = r.get("fee_multiplier") or 1.0
        ey = p - ask - taker_fee(ask, mult)
        en = (1 - p) - (1 - bid) - taker_fee(1 - bid, mult)
        if max(ey, en) < theta:
            continue
        if ey >= en:
            rows.append({**r, "side": "yes", "edge": ey, "pnl": y - ask - taker_fee(ask, mult)})
        else:
            rows.append({**r, "side": "no", "edge": en, "pnl": (1 - y) - (1 - bid) - taker_fee(1 - bid, mult)})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def evaluate(pred: pl.DataFrame, holdout_from: datetime, *, min_trades: int = 30) -> dict[str, Any]:
    """Score a prediction frame that already carries ``bid``, ``ask`` and ``y`` (see module doc).
    Returns {"scores": [...], "blend": {...}, "taker": [...]} for the dev/holdout split."""
    df = pred.filter(pl.col("p").is_not_null() & pl.col("bid").is_not_null() & pl.col("ask").is_not_null())
    df = df.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                         (pl.col("t") >= holdout_from).alias("holdout"))
    out: dict[str, Any] = {"rows": df.height, "events": df["event"].n_unique() if df.height else 0}
    scores = []
    for split, part in (("dev", df.filter(~pl.col("holdout"))), ("holdout", df.filter(pl.col("holdout")))):
        if not part.height:
            continue
        y = part["y"].to_numpy().astype(float)
        bm, lm = _scores(part["p"].to_numpy(), y)
        bk, lk = _scores(part["mid"].to_numpy(), y)
        scores.append({"split": split, "n": part.height, "events": part["event"].n_unique(),
                       "brier_model": round(bm, 5), "brier_mid": round(bk, 5), "logloss_model": round(lm, 5),
                       "logloss_mid": round(lk, 5), "mean_p": round(float(part["p"].mean()), 4),
                       "mean_mid": round(float(part["mid"].mean()), 4), "mean_y": round(float(y.mean()), 4)})
    out["scores"] = scores
    dev, hold = df.filter(~pl.col("holdout")), df.filter(pl.col("holdout"))
    if dev.height >= 20:
        def design(part: pl.DataFrame) -> np.ndarray:
            return np.column_stack([np.ones(part.height), logit(part["mid"].to_numpy()), logit(part["p"].to_numpy())])
        beta, t = fit_logistic(design(dev), dev["y"].to_numpy().astype(float), dev["event"].to_numpy())
        blend = {"a": round(float(beta[0]), 4), "b_mid": round(float(beta[1]), 4), "c_model": round(float(beta[2]), 4),
                 "t_b": round(float(t[1]), 2), "t_c": round(float(t[2]), 2)}
        if hold.height:
            ph = 1 / (1 + np.exp(-design(hold) @ beta))
            yh = hold["y"].to_numpy().astype(float)
            blend["brier_blend_holdout"] = round(_scores(ph, yh)[0], 5)
            blend["brier_mid_holdout"] = round(_scores(hold["mid"].to_numpy(), yh)[0], 5)
            if hold.height >= 20:
                b2, t2 = fit_logistic(design(hold), yh, hold["event"].to_numpy())
                blend["c_model_holdout_refit"] = round(float(b2[2]), 4)
                blend["t_c_holdout_refit"] = round(float(t2[2]), 2)
        out["blend"] = blend
    taker = []
    best = None
    for theta in THETAS:
        for split, part in (("dev", dev), ("holdout", hold)):
            tr = taker_pnl(part, theta)
            if not tr.height:
                taker.append({"theta": theta, "split": split, "trades": 0})
                continue
            mu, tt, k = clustered_mean(tr["pnl"].to_numpy(), tr["event"].to_numpy())
            taker.append({"theta": theta, "split": split, "trades": tr.height, "events": k,
                          "pnl_c": round(mu * 100, 3), "t": None if tt is None else round(tt, 2),
                          "yes_share": round(float((tr["side"] == "yes").mean()), 3)})
            if split == "dev" and tr.height >= min_trades and (best is None or mu > best[1]):
                best = (theta, mu)
    out["taker"] = taker
    out["theta_chosen_on_dev"] = None if best is None else best[0]
    return out


# ---- passive (maker) test with trade-print fills ---------------------------------------------
MARGINS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08)


def fetch_trades(s, windows: pl.DataFrame, name: str, *, workers: int = 2) -> pl.DataFrame:
    """Every public trade of each market inside its window (columns ticker, t_from, t_to), cached
    per study under ``data/cache/fundamental_<name>_trades.parquet`` (one entry per ticker+window)."""
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    path = s.data_dir / "cache" / f"fundamental_{name}_trades.parquet"
    done_path = path.with_name(path.stem + "_windows.parquet")
    have = pl.read_parquet(path) if path.exists() else None
    done = pl.read_parquet(done_path) if done_path.exists() else pl.DataFrame(
        schema={"ticker": pl.String, "t_from": pl.Datetime("us", "UTC"), "t_to": pl.Datetime("us", "UTC")})
    w = windows.select("ticker", "t_from", "t_to").unique()
    todo = w.join(done, on=["ticker", "t_from", "t_to"], how="anti")
    if todo.height:
        api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.1))
        cutoff = api.cutoff().get("trades_created_ts") or api.cutoff().get("market_settled_ts")

        def one(r: dict[str, Any]) -> list[dict[str, Any]]:
            historical = bool(cutoff and r["t_to"] < cutoff)
            out: list[dict[str, Any]] = []
            try:
                for page in api.trades(r["ticker"], historical, min_ts=r["t_from"], max_ts=r["t_to"]):
                    out.extend(k.trade_rows(page.payload))
            except RuntimeError as error:
                print(f"skip trades {r['ticker']}: {str(error)[:120]}")
            return out

        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, got in enumerate(pool.map(one, todo.iter_rows(named=True)), 1):
                rows.extend(got)
                if i % 200 == 0:
                    print(f"  trades {i}/{todo.height}", flush=True)
        new = pl.DataFrame(rows, infer_schema_length=None) if rows else None
        frames = [f for f in (have, new) if f is not None and f.height]
        have = pl.concat(frames, how="diagonal_relaxed").unique("trade_id") if frames else pl.DataFrame()
        path.parent.mkdir(parents=True, exist_ok=True)
        for f, target in ((have, path), (pl.concat([done, todo]), done_path)):
            tmp = target.with_suffix(".tmp")
            f.write_parquet(tmp)
            tmp.replace(target)
    return have if have is not None else pl.DataFrame()


def maker_fills(pred: pl.DataFrame, trades: pl.DataFrame, margin: float) -> pl.DataFrame:
    """Rest one YES bid and one NO bid per row from ``t`` until ``end`` around the fair value ``p``:
    YES bid ``floor(p − margin)`` and NO bid ``floor(1 − p − margin)`` (whole cents), each capped one
    tick inside the opposite quote so it is post-only. A bid fills only if a later trade prints
    strictly THROUGH it (a YES-seller at a lower YES price, or a YES-buyer at a higher one): the
    queue at our own price is never assumed to clear. Held to settlement; maker fee by series.

    ``pred`` needs ticker, event, t, end, p, bid, ask, y (and optionally fee_type,
    fee_multiplier); ``trades`` needs ticker, created_time, yes_price, taker_side."""
    tr = trades.select("ticker", pl.col("created_time").alias("ts"), "yes_price", "taker_side")
    by = {k[0]: v for k, v in tr.partition_by("ticker", as_dict=True).items()}
    rows = []
    for r in pred.iter_rows(named=True):
        p, bid, ask = r["p"], r["bid"], r["ask"]
        yb = min(math.floor((p - margin) * 100 + 1e-9) / 100, round(ask - 0.01, 2))
        nb = min(math.floor((1 - p - margin) * 100 + 1e-9) / 100, round(1 - bid - 0.01, 2))
        t_tr = by.get(r["ticker"])
        window = (t_tr.filter((pl.col("ts") > r["t"]) & (pl.col("ts") <= r["end"]))
                  if t_tr is not None else None)
        ft, mult = r.get("fee_type"), r.get("fee_multiplier") or 1.0
        for side, price in (("yes", yb), ("no", nb)):
            if price < 0.01:
                continue
            hit = None
            if window is not None and window.height:
                if side == "yes":
                    w = window.filter((pl.col("taker_side") == "no") & (pl.col("yes_price") < price - 1e-9))
                else:
                    w = window.filter((pl.col("taker_side") == "yes") & (pl.col("yes_price") > 1 - price + 1e-9))
                if w.height:
                    hit = w["ts"].min()
            won = r["y"] if side == "yes" else 1 - r["y"]
            rows.append({"ticker": r["ticker"], "event": r["event"], "t": r["t"], "side": side, "price": price,
                         "filled": hit is not None, "fill_ts": hit,
                         "pnl": (won - price - maker_fee(price, ft, mult)) if hit is not None else 0.0})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def evaluate_maker(pred: pl.DataFrame, trades: pl.DataFrame, holdout_from: datetime,
                   *, margins: tuple[float, ...] = MARGINS, min_fills: int = 30) -> dict[str, Any]:
    """Maker test per margin and split: orders, fills, PnL per filled contract and per order
    posted (t by event). The margin is chosen on dev (best PnL per fill with ≥ ``min_fills``)."""
    table, best = [], None
    for m in margins:
        f = maker_fills(pred, trades, m)
        if not f.height:
            continue
        f = f.with_columns((pl.col("t") >= holdout_from).alias("holdout"))
        for split, part in (("dev", f.filter(~pl.col("holdout"))), ("holdout", f.filter(pl.col("holdout")))):
            filled = part.filter("filled")
            row = {"margin": m, "split": split, "orders": part.height, "fills": filled.height,
                   "fill_rate": round(filled.height / part.height, 3) if part.height else None}
            if filled.height:
                mu, tt, k = clustered_mean(filled["pnl"].to_numpy(), filled["event"].to_numpy())
                row.update({"events": k, "pnl_per_fill_c": round(mu * 100, 3), "t": None if tt is None else round(tt, 2),
                            "pnl_per_order_c": round(float(part["pnl"].sum()) / part.height * 100, 3),
                            "yes_share": round(float((filled["side"] == "yes").mean()), 3)})
                if split == "dev" and filled.height >= min_fills and (best is None or mu > best[1]):
                    best = (m, mu)
            table.append(row)
    return {"maker": table, "margin_chosen_on_dev": None if best is None else best[0]}


def report(s, name: str, results: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
    from albert.reports import Report

    rep = Report(s.reports_dir, f"research-fundamental-{name}")
    rep.json("Run", {"rows": results.get("rows"), "events": results.get("events"),
                     "theta_chosen_on_dev": results.get("theta_chosen_on_dev"), **(extra or {})})
    if results.get("scores"):
        rep.table("Scores: model vs market mid (lower is better)", pl.DataFrame(results["scores"]))
    if results.get("blend"):
        rep.json("Blend logit P = a + b·logit(mid) + c·logit(model), fitted on dev", results["blend"])
    if results.get("taker"):
        rep.table("Taker at the ask/bid with fees, held to settlement (cents per contract)",
                  pl.DataFrame(results["taker"], infer_schema_length=None))
    if results.get("maker"):
        rep.json("Maker margin chosen on dev", {"margin": results.get("margin_chosen_on_dev")})
        rep.table("Maker: bids around the fair value, trade-through fills, held to settlement",
                  pl.DataFrame(results["maker"], infer_schema_length=None))
    return rep.save()


def now_utc() -> datetime:
    return datetime.now(UTC)
