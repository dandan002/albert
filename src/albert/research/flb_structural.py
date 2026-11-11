"""Structural decomposition of the favourite-longshot bias (workstream ``structural`` of ``paper_flb/``).

Question (research question 4): how much of the longshot bias on Kalshi is *mechanical* (tick floor,
fees, spread, capital), how much fits *probability weighting* (LLO / Prelec), how much fits Shin-type
*insiders*, and how much fits *heterogeneous beliefs*? By group, horizon and contract structure.

Input: the quote panel of the ``xs`` workstream (``data/research/flb_xs/panel.parquet``), with the
xs rules for usable quotes, sides, market and volume weights (imported from ``flb_xs``). Strike
types of ladder legs (to find ladders that partition the outcome space) come from the xs sample.

Steps (``python -m albert.research.flb_structural run <step>``; outputs in
``data/research/flb_structural/``):

* ``band``: per side observation, the no-arbitrage band of ``eq:band`` with the series taker fee, the
  bid-ask spread and capital at rho in {0, 5%, 10%} over the lead. The gross mispricing at the mid of
  a longshot side, ``q_mid - won``, splits exactly into tick floor + half-spread + fee + capital +
  residual, where the residual ``q_bid - fee - capital - won`` is the net profit per contract of
  selling the longshot (buying the other side as a taker). Residual > 0 means the bias lies outside
  the band.
* ``weighting``: maximum-likelihood fits of calibrated, constant-shift, LLO and Prelec models of
  ``P(y = 1 | q)`` on YES sides; parameters by group, lead and structure (event-clustered sandwich
  SEs); equality tests (cluster-robust Wald); in-sample log likelihood and out-of-sample log score
  (train on the first half of months, test on the second, and the reverse).
* ``shin``: events whose legs partition the outcome space (mutually exclusive multi-outcome events,
  two-leg events and partition ladders) with every leg quoted at the lead: Shin z from the asks and
  the mids, z fitted to outcomes, and log scores of raw, normalised and Shin probabilities.
* ``beliefs``: calibration-level regressions (logit slope interactions; longshot mispricing in cents)
  on ex-ante characteristics: number of outcomes, turnover, spread, horizon, forecast anchor.
* ``synthesis``: hypothesis scorecard by group and counterfactual shares of the gross longshot loss.
* ``all``: every step above, then ``summary.json`` and ``verdict.md``.
* ``latex``: rewrites the auto-generated blocks (macros, tables) in ``paper_flb/sections/*_structural.tex``.

No trading strategy is tested here, so no capped-book ledger or Sharpe ratio is computed (AGENTS.md
rule 6 applies to strategies). No look-ahead: every regressor is known at the quote time (volume up to
``ts``, open interest and spread at the quote, the lead); the number of legs is the event design.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import optimize
from scipy import stats as sps

from albert.research.flb_common import (
    BUCKET_LABELS,
    GROUPS,
    LONGSHOT,
    TAKER_RATE,
    clustered,
    fee_pc,
)
from albert.research.flb_xs import (
    LEAD_LABEL,
    LEADS_H,
    Q_CLIP,
    _cluster_meat,
    _jsonable,
    logit,
    shin_z,
    sides,
    usable,
    with_weights,
)
from albert.research.flb_xs import out_dir as xs_dir

OUT = Path("data/research/flb_structural")
PAPER = Path("paper_flb/sections")
RHOS = (0.0, 0.05, 0.10)
RHO_MAIN = 0.05
HOURS_PER_YEAR = 8760.0
LS_BUCKETS = BUCKET_LABELS[:3]                       # 1-2c, 3-5c, 6-10c
ANCHORED = ("Sports", "Weather")                     # sportsbook lines; NWS forecasts
INFO_GROUPS = ("Politics", "Entertainment", "Mentions", "Science & Tech")   # private information plausible
PUBLIC_GROUPS = ("Sports", "Weather")                # outcomes publicly observed
MAIN_GROUPS = tuple(g for g in GROUPS if g != "Other")
Z_GRID = np.round(np.arange(0.0, 0.6001, 0.005), 4)    # Shin z grid for outcome-fitted z
P_CLIP = Q_CLIP                                      # forecasts clipped to [0.5c, 99.5c] for log scores
MIN_FIT_N, MIN_FIT_CLUSTERS = 150, 20
SAME_BASIS = {"shin_ask": "norm_ask", "shin_mid": "norm_mid", "shin_fit": "norm_mid", "norm_mid": "mid"}


def out_dir() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def _write_json(name: str, obj: Any) -> None:
    (out_dir() / f"{name}.json").write_text(json.dumps(_jsonable(obj), indent=1))


def _read_json(name: str) -> Any:
    return json.loads((out_dir() / f"{name}.json").read_text())


def _write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with (out_dir() / f"{name}.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: _fmt_csv(r.get(c)) for c in cols})


def _fmt_csv(v: Any) -> Any:
    if isinstance(v, float | np.floating):
        return "" if not math.isfinite(float(v)) else round(float(v), 7)
    return v


# ---- data ------------------------------------------------------------------------------------------
def panel_status() -> dict[str, Any]:
    d = xs_dir()
    ready = (d / "PANEL_READY").exists()
    st = (d / "panel.parquet").stat()
    return {"panel_ready": ready, "panel_mtime": datetime.fromtimestamp(st.st_mtime, UTC).isoformat(timespec="seconds"),
            "panel_bytes": st.st_size}


def load_pre(path: Path | None = None) -> pl.DataFrame:
    """Usable pre-event quotes of the xs panel with the xs market / volume weights, plus the number of
    outcomes of the event and the forecast-anchor flag."""
    raw = pl.read_parquet(path or (xs_dir() / "panel.parquet"))
    pre = usable(raw).filter((pl.col("phase") == "pre") & (pl.col("lead_h") > 0) & pl.col("group").is_in(GROUPS))
    pre = with_weights(pre)
    me = pl.col("mutually_exclusive").fill_null(False)
    return pre.with_columns(
        pl.when(pl.col("structure").is_in(["multi", "ladder"]) | (me & (pl.col("n_legs") >= 2)))
        .then(pl.col("n_legs").cast(pl.Float64).clip(lower_bound=2)).otherwise(2.0).alias("n_out"),
        pl.col("group").is_in(list(ANCHORED)).alias("anchored"))


def side_frame(pre: pl.DataFrame) -> pl.DataFrame:
    """xs side observations joined with the extra quote columns this workstream needs."""
    extra = pre.select("ticker", "lead_h", "vol_24h", "quote_age_h", "n_out", "anchored", "mutually_exclusive")
    return sides(pre).join(extra, on=["ticker", "lead_h"], how="left")


def half_split(df: pl.DataFrame, col: str = "month") -> str:
    """Last month of the first half: the month boundary that splits observations closest to 50/50."""
    vc = df.group_by(col).agg(pl.len().alias("n")).sort(col)
    cum = np.cumsum(vc["n"].to_numpy()) / df.height
    i = int(np.argmin(np.abs(cum - 0.5)))
    return str(vc[col][i])


# ---- 1. the no-arbitrage band ----------------------------------------------------------------------
def band_components(S: pl.DataFrame, rho: float = RHO_MAIN) -> pl.DataFrame:
    """Per side, in dollars per contract: gross = q_mid - won = floor + spread + fee + capital + resid.

    * floor: the half-spread of a side pinned at the tick floor (no bid, ask at one tick);
    * spread: the half-spread q_mid - q_bid of other sides;
    * fee: taker fee theta q_bid (1 - q_bid) for buying the other side at 1 - q_bid;
    * capital: rho T (1 - q_bid), T = lead in years;
    * resid: q_bid - fee - capital - won, the net profit per contract of selling the side as a taker.
    resid_mk is the maker variant (sell at the ask: rest a bid for the other side at 1 - q_ask),
    an upper bound because it ignores adverse selection of fills."""
    tick = pl.when(pl.col("subcent")).then(0.001).otherwise(0.01)
    pinned = (pl.col("q_bid") <= 1e-9) & (pl.col("q_ask") <= tick + 1e-9)
    T = pl.col("lead_h").clip(lower_bound=0) / HOURS_PER_YEAR
    theta = TAKER_RATE * pl.col("fee_multiplier")
    out = S.with_columns(pinned.alias("pinned"), T.alias("T_years")).with_columns(
        (pl.col("q_mid") - pl.col("won")).alias("gross"),
        pl.when(pl.col("pinned")).then(pl.col("q_mid") - pl.col("q_bid")).otherwise(0.0).alias("c_floor"),
        pl.when(pl.col("pinned")).then(0.0).otherwise(pl.col("q_mid") - pl.col("q_bid")).alias("c_spread"),
        (theta * pl.col("q_bid") * (1 - pl.col("q_bid"))).alias("c_fee"),
        (rho * pl.col("T_years") * (1 - pl.col("q_bid"))).alias("c_cap"))
    mfee = fee_pc(out["q_ask"].to_numpy(), maker=True, mult=out["fee_multiplier"].to_numpy(),
                  maker_fee_series=out["maker_fee"].to_numpy())
    return out.with_columns(pl.Series("c_mfee", mfee)).with_columns(
        (pl.col("q_bid") - pl.col("c_fee") - pl.col("c_cap") - pl.col("won")).alias("resid"),
        (pl.col("q_ask") - pl.col("c_mfee") - rho * pl.col("T_years") * (1 - pl.col("q_ask")) - pl.col("won")).alias("resid_mk"))


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float((x * w).sum() / w.sum()) if w.sum() > 0 else float("nan")


def band_cell(x: pl.DataFrame, w: str = "w_m") -> dict[str, Any] | None:
    """Mean decomposition of a cell of sides, in cents per contract, with event-clustered inference on
    the gross mispricing and the residual (taker and maker)."""
    if x.height == 0 or x[w].sum() <= 0:
        return None
    wt = x[w].to_numpy()
    ev = x["event_ticker"].to_numpy()
    g = clustered(100 * x["gross"].to_numpy(), ev, wt)
    r = clustered(100 * x["resid"].to_numpy(), ev, wt)
    rm = clustered(100 * x["resid_mk"].to_numpy(), ev, wt)
    comp = {c: 100 * _wmean(x[c].to_numpy(), wt) for c in ("c_floor", "c_spread", "c_fee", "c_cap")}
    q = 100 * _wmean(x["q_mid"].to_numpy(), wt)
    band = sum(comp.values())
    gross = g["mean"]
    return {"n": x.height, "markets": x["ticker"].n_unique(), "clusters": g["clusters"], "q_mid_c": q,
            "win_c": 100 * _wmean(x["won"].to_numpy(), wt), "gross_c": gross, "gross_se": g["se"], "gross_t": g["t"],
            "gross_rel": gross / q if q > 0 else None,
            "floor_c": comp["c_floor"], "spread_c": comp["c_spread"], "fee_c": comp["c_fee"], "cap_c": comp["c_cap"],
            "band_c": band, "resid_c": r["mean"], "resid_se": r["se"], "resid_t": r["t"],
            "resid_mk_c": rm["mean"], "resid_mk_se": rm["se"], "resid_mk_t": rm["t"],
            "share_in_band": (min(band, gross) / gross) if gross and gross > 0 else None,
            "pinned_share": _wmean(x["pinned"].cast(pl.Float64).to_numpy(), wt)}


def _lead_weights(x: pl.DataFrame) -> pl.DataFrame:
    """Within a single lead, every market counts once (as in the xs term structure)."""
    return x.with_columns((1.0 / pl.len().over("ticker")).alias("w_m")).with_columns(
        (pl.col("w_m") * pl.col("vol_7d").fill_null(0).clip(lower_bound=0)).alias("w_v"))


def drop_top_events(x: pl.DataFrame, col: str, k: int = 5, w: str = "w_m") -> pl.DataFrame:
    """Drop the k events with the largest absolute weighted contribution to sum(w * col)."""
    top = (x.group_by("event_ticker").agg((pl.col(w) * pl.col(col)).sum().abs().alias("c"))
           .sort("c", descending=True).head(k)["event_ticker"])
    return x.filter(~pl.col("event_ticker").is_in(top.to_list()))


def band(pre: pl.DataFrame | None = None) -> dict[str, Any]:
    pre = load_pre() if pre is None else pre
    S0 = side_frame(pre)
    ls_expr = pl.col("q_mid") <= LONGSHOT + 1e-9
    out: dict[str, Any] = {"rho_main": RHO_MAIN, "rhos": list(RHOS)}
    rows: list[dict[str, Any]] = []
    for rho in RHOS:
        B = band_components(S0, rho)
        LS = B.filter(ls_expr)
        for wname in ("w_m", "w_v"):
            def add(label: dict[str, Any], x: pl.DataFrame, per_lead: bool, wname: str = wname) -> dict[str, Any] | None:
                xx = _lead_weights(x) if per_lead else x
                c = band_cell(xx, wname)
                if c is not None:
                    rows.append({"rho": rho, "weight": "market" if wname == "w_m" else "volume", **label, **c})
                return c
            # pooled leads: bucket x group, and longshots pooled
            for grp in ("All", *GROUPS):
                xg = B if grp == "All" else B.filter(pl.col("group") == grp)
                for b in (*BUCKET_LABELS, "LS <=10c"):
                    xb = xg.filter(ls_expr) if b == "LS <=10c" else xg.filter(pl.col("bucket") == b)
                    add({"group": grp, "lead": "pooled", "bucket": b}, xb, False)
                # by lead, longshots and longshot buckets
                for ld in LEADS_H:
                    xl = xg.filter(pl.col("lead_h") == float(ld))
                    add({"group": grp, "lead": LEAD_LABEL[ld], "bucket": "LS <=10c"}, xl.filter(ls_expr), True)
                    for b in LS_BUCKETS:
                        add({"group": grp, "lead": LEAD_LABEL[ld], "bucket": b}, xl.filter(pl.col("bucket") == b), True)
            for st in ("binary", "multi", "ladder", "deadline"):
                add({"group": "All", "structure": st, "lead": "pooled", "bucket": "LS <=10c"},
                    LS.filter(pl.col("structure") == st), False)
        if rho == RHO_MAIN:
            # robustness on the pooled longshot cell (market-weighted)
            split = half_split(pre)
            out["split_month"] = split
            rob = {"first_half": band_cell(with_weights_sides(LS.filter(pl.col("month") <= split))),
                   "second_half": band_cell(with_weights_sides(LS.filter(pl.col("month") > split))),
                   "drop_top5_events": band_cell(drop_top_events(LS, "resid")),
                   "quote_age_le_24h": band_cell(LS.filter(pl.col("quote_age_h") <= 24)),
                   "spread_le_5c": band_cell(LS.filter(pl.col("spread") <= 0.05 + 1e-9)),
                   "not_pinned": band_cell(LS.filter(~pl.col("pinned"))),
                   "with_bid": band_cell(LS.filter(pl.col("q_bid") > 1e-9)),
                   "pinned": band_cell(LS.filter(pl.col("pinned")))}
            out["robustness"] = rob
            # tick floor: subcent vs 1c grid
            fl = []
            for lab, cond in (("1c grid", ~pl.col("subcent")), ("0.1c grid", pl.col("subcent"))):
                for b in ("1-2c", "LS <=10c"):
                    x = LS.filter(cond) if b == "LS <=10c" else B.filter(cond & (pl.col("bucket") == b))
                    c = band_cell(x)
                    if c:
                        fl.append({"grid": lab, "bucket": b, **c})
            out["tick_grid"] = fl
            # horizon gradients (H1: gross rises with T, residual does not), longshot sides, bucket FE
            out["horizon_slope"] = {col: horizon_slope(LS, col) for col in ("gross", "resid", "c_cap")}
            out["horizon_slope_by_group"] = {g: {col: horizon_slope(LS.filter(pl.col("group") == g), col)
                                                 for col in ("gross", "resid")} for g in MAIN_GROUPS}
            out["n_ls_sides"] = LS.height
            out["n_quotes"] = pre.height
    out["cells"] = rows
    _write_csv("band_cells", rows)
    _write_csv("band_tick_grid", out.get("tick_grid", []))
    _write_json("band", out)
    return out


def with_weights_sides(x: pl.DataFrame) -> pl.DataFrame:
    """Market weights recomputed within a subsample of sides (each market weighs one in total)."""
    return x.with_columns((1.0 / pl.len().over("ticker")).alias("w_m"))


def horizon_slope(x: pl.DataFrame, col: str, w: str = "w_m") -> dict[str, Any]:
    """WLS of 100 * col (cents) on log(lead hours) with bucket fixed effects, clustered by event."""
    if x.height < 50:
        return {"n": x.height}
    bk = x["bucket"].to_numpy()
    levels = sorted(set(bk))
    X = np.column_stack([np.log(x["lead_h"].to_numpy())] + [(bk == b).astype(float) for b in levels])
    r = ols_cluster(100 * x[col].to_numpy(), X, x["event_ticker"].to_numpy(), x[w].to_numpy())
    if r.get("coef") is None:
        return {"n": x.height}
    return {"n": r["n"], "clusters": r["clusters"], "slope": r["coef"][0], "se": r["se"][0], "t": r["t"][0],
            "per_log_lead": "cents per unit of log(lead hours)"}


def ols_cluster(yv: Any, X: Any, g: Any, w: Any | None = None) -> dict[str, Any]:
    """Weighted least squares with event-clustered SEs (no constant added)."""
    yv, X, g = np.asarray(yv, dtype=float), np.asarray(X, dtype=float), np.asarray(g)
    w = np.ones_like(yv) if w is None else np.asarray(w, dtype=float)
    ok = np.isfinite(yv) & np.all(np.isfinite(X), axis=1) & (w > 0)
    yv, X, g, w = yv[ok], X[ok], g[ok], w[ok]
    if len(yv) <= X.shape[1] or len(np.unique(g)) < 5:
        return {"n": len(yv), "coef": None}
    XtW = (X * w[:, None]).T
    A = np.linalg.pinv(XtW @ X)
    beta = A @ (XtW @ yv)
    meat, k = _cluster_meat(X * (w * (yv - X @ beta))[:, None], g)
    V = A @ meat @ A
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return {"n": len(yv), "clusters": k, "coef": beta.tolist(), "se": se.tolist(), "V": V.tolist(),
            "t": [float(b / s) if s > 0 else None for b, s in zip(beta, se, strict=True)]}


# ---- 2. probability weighting ----------------------------------------------------------------------
def _sig(v: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(v, -35, 35)))


def logit_design(y: Any, X: Any, g: Any, w: Any | None = None, iters: int = 100) -> dict[str, Any]:
    """Weighted logit of y on the columns of X (Newton), event-clustered sandwich covariance."""
    y, X, g = np.asarray(y, dtype=float), np.asarray(X, dtype=float), np.asarray(g)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & (w > 0)
    y, X, g, w = y[ok], X[ok], g[ok], w[ok]
    k = X.shape[1]
    if len(y) < max(20, 3 * k) or len(np.unique(g)) < 5:
        return {"n": len(y), "coef": None}
    b = np.zeros(k)
    for _ in range(iters):
        p = _sig(X @ b)
        H = (X * (w * p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H + 1e-9 * np.eye(k), X.T @ (w * (y - p)))
        lam = 1.0
        while lam > 1e-4 and _wll_logit(y, X @ (b + lam * step), w) < _wll_logit(y, X @ b, w) - 1e-12:
            lam /= 2
        b = b + lam * step
        if np.max(np.abs(lam * step)) < 1e-10:
            break
    p = _sig(X @ b)
    H = (X * (w * p * (1 - p))[:, None]).T @ X
    Hi = np.linalg.pinv(H)
    meat, nc = _cluster_meat(X * (w * (y - p))[:, None], g)
    V = Hi @ meat @ Hi
    ll = y * np.log(np.clip(p, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-15, 1))
    return {"n": len(y), "clusters": nc, "coef": b.tolist(), "V": V.tolist(),
            "se": np.sqrt(np.clip(np.diag(V), 0, None)).tolist(), "mean_ll": float((w * ll).sum() / w.sum()),
            "sum_w": float(w.sum())}


def _wll_logit(y: np.ndarray, eta: np.ndarray, w: np.ndarray) -> float:
    p = _sig(eta)
    return float((w * (y * np.log(np.clip(p, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-15, 1)))).sum())


# Models of P(y = 1 | q) for the YES side priced q (clipped to [Q_CLIP, 1 - Q_CLIP]).
def prob_calibrated(q: np.ndarray, th: np.ndarray | None = None) -> np.ndarray:
    return np.clip(q, P_CLIP, 1 - P_CLIP)


def prob_shift(q: np.ndarray, th: np.ndarray) -> np.ndarray:
    return _sig(th[0] + logit(q))


def prob_llo(q: np.ndarray, th: np.ndarray) -> np.ndarray:
    """LLO: logit P = alpha + beta logit q  (beta = 1/gamma, alpha = -delta/gamma)."""
    return _sig(th[0] + th[1] * logit(q))


def prob_prelec(q: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Prelec (1998) two-parameter weighting w(pi) = exp(-delta (-ln pi)^gamma) inverted at q = w(pi):
    pi = exp(-((-ln q) / delta)^(1/gamma)); th = (ln delta, ln(1/gamma))."""
    L = -np.log(np.clip(q, Q_CLIP, 1 - Q_CLIP))
    u = np.exp(np.exp(th[1]) * (np.log(L) - th[0]))
    return np.exp(-u)


def prelec_w(pi: Any, gamma: float, delta: float) -> np.ndarray:
    pi = np.asarray(pi, dtype=float)
    return np.exp(-delta * (-np.log(pi)) ** gamma)


def llo_w(pi: Any, gamma: float, delta: float) -> np.ndarray:
    pi = np.asarray(pi, dtype=float)
    return _sig(delta + gamma * np.log(pi / (1 - pi)))


def _ll_score_prelec(th: np.ndarray, q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L = -np.log(np.clip(q, Q_CLIP, 1 - Q_CLIP))
    kappa = np.exp(th[1])
    lr = np.log(L) - th[0]
    u = np.exp(kappa * lr)
    P = np.clip(np.exp(-u), 1e-15, 1 - 1e-15)
    ll = y * np.log(P) + (1 - y) * np.log(1 - P)
    common = (y - P) / (1 - P) * u * kappa
    return ll, np.column_stack([common, -common * lr])


def mle_cluster(ll_score: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], th0: Any, g: Any,
                w: np.ndarray) -> dict[str, Any]:
    """Weighted M-estimation with an analytic score; numerical Hessian; event-clustered sandwich."""
    W = w.sum()

    def f(th: np.ndarray) -> tuple[float, np.ndarray]:
        ll, s = ll_score(th)
        return -float((w * ll).sum() / W), -(w[:, None] * s).sum(0) / W

    res = optimize.minimize(f, np.asarray(th0, dtype=float), jac=True, method="BFGS",
                            options={"gtol": 1e-9, "maxiter": 500})
    th = res.x
    k = len(th)
    H = np.zeros((k, k))
    for j in range(k):
        e = np.zeros(k)
        e[j] = 1e-5
        H[:, j] = (f(th + e)[1] - f(th - e)[1]) / 2e-5
    H = (H + H.T) / 2 * W
    _, s = ll_score(th)
    meat, nc = _cluster_meat(s * w[:, None], np.asarray(g))
    Hi = np.linalg.pinv(H)
    V = Hi @ meat @ Hi
    return {"coef": th.tolist(), "V": V.tolist(), "se": np.sqrt(np.clip(np.diag(V), 0, None)).tolist(),
            "clusters": nc, "n": len(w), "mean_ll": -res.fun, "converged": bool(res.success)}


def fit_llo(q: Any, y: Any, g: Any, w: Any) -> dict[str, Any]:
    """LLO fit with alpha, beta and the weighting parameters gamma = 1/beta, delta = -alpha/beta."""
    q, y = np.asarray(q, dtype=float), np.asarray(y, dtype=float)
    r = logit_design(y, np.column_stack([np.ones_like(q), logit(q)]), g, w)
    if r.get("coef") is None:
        return {"n": r["n"]}
    a, b = r["coef"]
    V = np.asarray(r["V"])
    J = np.array([[0.0, -1 / b ** 2], [-1 / b, a / b ** 2]])          # d(gamma, delta)/d(alpha, beta)
    Vg = J @ V @ J.T
    sg, sd = math.sqrt(max(Vg[0, 0], 0)), math.sqrt(max(Vg[1, 1], 0))
    sa, sb = r["se"]
    return {"n": r["n"], "clusters": r["clusters"], "alpha": a, "beta": b, "se_alpha": sa, "se_beta": sb,
            "t_beta1": (b - 1) / sb if sb > 0 else None, "gamma": 1 / b, "se_gamma": sg,
            "t_gamma1": (1 / b - 1) / sg if sg > 0 else None, "delta": -a / b, "se_delta": sd,
            "mean_ll": r["mean_ll"], "V": r["V"]}


def fit_prelec(q: Any, y: Any, g: Any, w: Any) -> dict[str, Any]:
    q, y, w = np.asarray(q, dtype=float), np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    r = mle_cluster(lambda th: _ll_score_prelec(th, q, y), [0.0, 0.0], g, w)
    ld, lk = r["coef"]
    s_ld, s_lk = r["se"]
    gamma, delta = math.exp(-lk), math.exp(ld)
    return {"n": r["n"], "clusters": r["clusters"], "gamma": gamma, "se_gamma": gamma * s_lk,
            "t_gamma1": (gamma - 1) / (gamma * s_lk) if s_lk > 0 else None, "delta": delta, "se_delta": delta * s_ld,
            "ln_delta": ld, "ln_kappa": lk, "mean_ll": r["mean_ll"], "converged": r["converged"]}


def fit_shift(q: Any, y: Any, g: Any, w: Any) -> dict[str, Any]:
    q, y, w = np.asarray(q, dtype=float), np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    x = logit(q)

    def ls(th: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.clip(_sig(th[0] + x), 1e-15, 1 - 1e-15)
        return y * np.log(p) + (1 - y) * np.log(1 - p), (y - p)[:, None]

    r = mle_cluster(ls, [0.0], g, w)
    return {"n": r["n"], "clusters": r["clusters"], "alpha": r["coef"][0], "se_alpha": r["se"][0], "mean_ll": r["mean_ll"]}


def log_score(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, P_CLIP, 1 - P_CLIP)
    return y * np.log(p) + (1 - y) * np.log(1 - p)


def _yes(pre: pl.DataFrame, w: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (pre["mid"].to_numpy(), pre["y"].cast(pl.Float64).to_numpy(), pre["event_ticker"].to_numpy(),
            pre[w].to_numpy())


def _enough(x: pl.DataFrame) -> bool:
    return x.height >= MIN_FIT_N and x["event_ticker"].n_unique() >= MIN_FIT_CLUSTERS


def weighting_cells(pre: pl.DataFrame, w: str = "w_m") -> list[dict[str, Any]]:
    """LLO and Prelec fits by group, lead, structure, and group x lead (YES sides)."""
    cells: list[tuple[dict[str, Any], pl.DataFrame]] = [({"dim": "all", "cell": "All"}, pre)]
    cells += [({"dim": "group", "cell": g}, pre.filter(pl.col("group") == g)) for g in GROUPS]
    cells += [({"dim": "lead", "cell": LEAD_LABEL[ld]}, _lead_weights(pre.filter(pl.col("lead_h") == float(ld))))
              for ld in LEADS_H]
    cells += [({"dim": "structure", "cell": st}, pre.filter(pl.col("structure") == st))
              for st in ("binary", "multi", "ladder", "deadline")]
    cells += [({"dim": "group_lead", "cell": f"{g}|{LEAD_LABEL[ld]}"},
               _lead_weights(pre.filter((pl.col("group") == g) & (pl.col("lead_h") == float(ld)))))
              for g in MAIN_GROUPS for ld in LEADS_H]
    rows = []
    for lab, x in cells:
        if not _enough(x) or x[w].sum() <= 0:
            continue
        q, y, g, ww = _yes(x, w)
        llo = fit_llo(q, y, g, ww)
        pr = fit_prelec(q, y, g, ww)
        rows.append({**lab, "weight": "market" if w == "w_m" else "volume", "n": llo.get("n"),
                     "clusters": llo.get("clusters"),
                     **{f"llo_{k}": v for k, v in llo.items() if k not in ("n", "clusters", "V")},
                     **{f"prelec_{k}": v for k, v in pr.items() if k not in ("n", "clusters")}})
    return rows


def wald_equal(pre: pl.DataFrame, dim: str, levels: list[str], w: str = "w_m", what: str = "slope") -> dict[str, Any]:
    """Cluster-robust Wald test that the LLO slope (what='slope') or slope and intercept (what='both')
    are equal across the levels of ``dim``, from one fully interacted logit (events clustered across
    cells, so leads of the same market are handled jointly)."""
    x = pre.filter(pl.col(dim).is_in(levels))
    lv = [lv for lv in levels if x.filter(pl.col(dim) == lv).height >= MIN_FIT_N]
    x = x.filter(pl.col(dim).is_in(lv))
    if len(lv) < 2:
        return {"levels": lv}
    d = x[dim].to_numpy()
    lq = logit(x["mid"].to_numpy())
    D = np.column_stack([(d == v).astype(float) for v in lv])
    X = np.column_stack([D, D * lq[:, None]])
    r = logit_design(x["y"].cast(pl.Float64).to_numpy(), X, x["event_ticker"].to_numpy(), x[w].to_numpy())
    if r.get("coef") is None:
        return {"levels": lv}
    b, V = np.asarray(r["coef"]), np.asarray(r["V"])
    k = len(lv)
    rows = []
    for j in range(1, k):
        for off in ((k,) if what == "slope" else (0, k)):
            e = np.zeros(2 * k)
            e[off + j], e[off] = 1.0, -1.0
            rows.append(e)
    R = np.asarray(rows)
    Rb = R @ b
    W = float(Rb @ np.linalg.pinv(R @ V @ R.T) @ Rb)
    df = R.shape[0]
    return {"levels": lv, "what": what, "wald": W, "df": df, "p": float(sps.chi2.sf(W, df)),
            "slopes": b[k:].tolist(), "slope_se": np.sqrt(np.diag(V)[k:]).tolist(), "clusters": r["clusters"],
            "n": r["n"]}


def oos_compare(train: pl.DataFrame, test: pl.DataFrame, w: str = "w_m") -> dict[str, Any]:
    """Out-of-sample log score per (weighted) YES observation of each model fitted on ``train``;
    differences against the calibrated model with event-clustered t-statistics."""
    train, test = with_weights(train), with_weights(test)
    q, y, g, ww = _yes(train, w)
    qt, yt, gt, wt = _yes(test, w)
    fits = {"shift": fit_shift(q, y, g, ww), "llo": fit_llo(q, y, g, ww), "prelec": fit_prelec(q, y, g, ww)}
    preds = {"calibrated": prob_calibrated(qt),
             "shift": prob_shift(qt, np.array([fits["shift"]["alpha"]])),
             "llo": prob_llo(qt, np.array([fits["llo"]["alpha"], fits["llo"]["beta"]])),
             "prelec": prob_prelec(qt, np.array([fits["prelec"]["ln_delta"], fits["prelec"]["ln_kappa"]]))}
    # group-specific LLO (one weighting function per group) against the single pooled function
    pg = np.full(len(qt), np.nan)
    tg = test["group"].to_numpy()
    for grp in np.unique(tg):
        tr = train.filter(pl.col("group") == grp)
        f = fit_llo(*_yes(tr, w)) if _enough(tr) else fits["llo"]
        if f.get("beta") is None:
            f = fits["llo"]
        m = tg == grp
        pg[m] = prob_llo(qt[m], np.array([f["alpha"], f["beta"]]))
    preds["llo_by_group"] = pg
    base = log_score(preds["calibrated"], yt)
    out: dict[str, Any] = {"n_train": train.height, "n_test": test.height,
                           "clusters_test": int(np.unique(gt).size), "fits": fits, "models": {}}
    llo_ls = log_score(preds["llo"], yt)
    for m, p in preds.items():
        ls = log_score(p, yt)
        d = clustered(ls - base, gt, wt)
        dl = clustered(ls - llo_ls, gt, wt)
        out["models"][m] = {"mean_ls": float((wt * ls).sum() / wt.sum()), "diff_vs_cal": d["mean"],
                            "diff_se": d["se"], "diff_t": d["t"], "diff_vs_llo": dl["mean"], "diff_vs_llo_t": dl["t"]}
    return out


def weighting(pre: pl.DataFrame | None = None) -> dict[str, Any]:
    pre = load_pre() if pre is None else pre
    out: dict[str, Any] = {}
    rows = weighting_cells(pre, "w_m") + [r for r in weighting_cells(pre, "w_v") if r["dim"] in ("all", "group")]
    out["cells"] = rows
    _write_csv("weighting_fits", rows)
    # in-sample comparison, pooled and by group (market-weighted)
    ins = []
    for grp in ("All", *GROUPS):
        x = pre if grp == "All" else pre.filter(pl.col("group") == grp)
        if not _enough(x):
            continue
        q, y, g, ww = _yes(x, "w_m")
        cal = float((ww * log_score(prob_calibrated(q), y)).sum() / ww.sum())
        sh, llo, pr = fit_shift(q, y, g, ww), fit_llo(q, y, g, ww), fit_prelec(q, y, g, ww)
        nm = float(x["ticker"].n_unique())
        ins.append({"group": grp, "n": x.height, "markets": int(nm), "clusters": llo["clusters"],
                    "ll_calibrated": cal, "ll_shift": sh["mean_ll"], "ll_llo": llo["mean_ll"], "ll_prelec": pr["mean_ll"],
                    # weighted log likelihood totals (each market weighs one): LR-style gains, not chi2 under clustering
                    "gain_shift_total": (sh["mean_ll"] - cal) * nm, "gain_llo_total": (llo["mean_ll"] - cal) * nm,
                    "gain_prelec_total": (pr["mean_ll"] - cal) * nm})
    out["in_sample"] = ins
    _write_csv("weighting_insample", ins)
    # out of sample: first half of months -> second half, and the reverse
    split = half_split(pre)
    first, second = pre.filter(pl.col("month") <= split), pre.filter(pl.col("month") > split)
    out["split_month"] = split
    out["oos"] = {"first_to_second": oos_compare(first, second), "second_to_first": oos_compare(second, first)}
    oos_rows = [{"direction": d, "model": m, **v} for d, r in out["oos"].items() for m, v in r["models"].items()]
    _write_csv("weighting_oos", oos_rows)
    # fits by half (stability)
    out["halves"] = {lab: {k: v for k, v in fit_llo(*_yes(with_weights(x), "w_m")).items() if k != "V"}
                     for lab, x in (("first", first), ("second", second))}
    # equality tests
    out["wald"] = {"group_slope": wald_equal(pre, "group", list(MAIN_GROUPS)),
                   "group_both": wald_equal(pre, "group", list(MAIN_GROUPS), what="both"),
                   "structure_slope": wald_equal(pre, "structure", ["binary", "multi", "ladder", "deadline"]),
                   "lead_slope": wald_equal(pre.with_columns(pl.col("lead_h").cast(pl.String).alias("lead_s")), "lead_s",
                                            [str(float(ld)) for ld in LEADS_H]),
                   "group_slope_vol": wald_equal(pre, "group", list(MAIN_GROUPS), w="w_v")}
    # fitted curves for the figure: pi on a grid of q, LLO and Prelec by group
    curves = []
    qg = np.round(np.concatenate([np.arange(0.005, 0.1, 0.005), np.arange(0.1, 0.9, 0.02), np.arange(0.9, 0.996, 0.005)]), 4)
    for r in rows:
        if r["dim"] not in ("all", "group") or r["weight"] != "market" or r.get("llo_beta") is None:
            continue
        p_llo = prob_llo(qg, np.array([r["llo_alpha"], r["llo_beta"]]))
        p_pr = prob_prelec(qg, np.array([r["prelec_ln_delta"], r["prelec_ln_kappa"]]))
        curves += [{"cell": r["cell"], "q": float(a), "pi_llo": float(b), "pi_prelec": float(c)}
                   for a, b, c in zip(qg, p_llo, p_pr, strict=True)]
    _write_csv("weighting_curves", curves)
    # calibration points by YES-price bucket (for the figure overlay)
    pts = []
    for grp in ("All", *MAIN_GROUPS):
        x = pre if grp == "All" else pre.filter(pl.col("group") == grp)
        xb = x.with_columns(pl.col("mid").alias("q")).with_columns(_bucket("q"))
        for b in BUCKET_LABELS:
            xx = xb.filter(pl.col("bucket") == b)
            if xx.height < 30:
                continue
            c = clustered(xx["y"].cast(pl.Float64).to_numpy(), xx["event_ticker"].to_numpy(), xx["w_m"].to_numpy())
            pts.append({"cell": grp, "bucket": b, "n": xx.height, "q": _wmean(xx["mid"].to_numpy(), xx["w_m"].to_numpy()),
                        "win": c["mean"], "win_se": c["se"]})
    _write_csv("weighting_points", pts)
    _write_json("weighting", out)
    return out


def _bucket(col: str) -> pl.Expr:
    from albert.research.flb_common import bucket_expr
    return bucket_expr(col)


# ---- 3. Shin ---------------------------------------------------------------------------------------
def shin_forward(pi: Any, z: float) -> np.ndarray:
    """Shin (1993) prices of outcomes with true probabilities pi and insider share z:
    q_i = sqrt(z pi_i + (1 - z) pi_i^2) * sum_j sqrt(z pi_j + (1 - z) pi_j^2)."""
    pi = np.asarray(pi, dtype=float)
    r = np.sqrt(z * pi + (1 - z) * pi ** 2)
    return r * r.sum()


def shin_probs(prices: Any, z: float, S: float) -> np.ndarray:
    """Equation eq:shin: pi_i = [sqrt(z^2 + 4 (1 - z) q_i^2 / S) - z] / (2 (1 - z))."""
    q = np.asarray(prices, dtype=float)
    return (np.sqrt(z * z + 4 * (1 - z) * q * q / S) - z) / (2 * (1 - z))


def shin_solve(prices: Any) -> dict[str, Any]:
    """z and implied probabilities from the quoted prices of a partition (eq:shin with S = sum q).
    S <= 1 (no overround) gives z = 0 and plain normalisation, flagged."""
    q = np.asarray(prices, dtype=float)
    S = float(q.sum())
    z, p = shin_z(q)
    return {"z": float(z), "probs": np.asarray(p, dtype=float), "S": S, "overround": S > 1 + 1e-12}


def shin_fixed_z(prices: Any, z: float) -> np.ndarray:
    """Shin probabilities for a given z with the scale S chosen so that they sum to one (z = 0 is the
    normalisation q_i / sum q). Used to fit z to outcomes from relative prices."""
    q = np.asarray(prices, dtype=float)
    if z <= 0:
        return q / q.sum()
    f = lambda ls: shin_probs(q, z, math.exp(ls)).sum() - 1.0   # decreasing in ln S
    lo, hi = math.log(q.sum() ** 2) - 5, math.log(q.sum() ** 2) + 5
    while f(lo) < 0:
        lo -= 5
    while f(hi) > 0:
        hi += 5
    ls = optimize.brentq(f, lo, hi, xtol=1e-12)
    return shin_probs(q, z, math.exp(ls))


def partition_ladders(pre: pl.DataFrame) -> pl.DataFrame:
    """Ladder keys whose legs partition the outcome space: one lower tail ('less'), one upper tail
    ('greater'), the rest 'between' buckets; the ladder is the whole event. Strike types are read from
    the xs sample (a static market attribute)."""
    st = pl.read_parquet(xs_dir() / "sample.parquet", columns=["ticker", "strike_type"])
    lad = pre.filter(pl.col("structure") == "ladder").select("ticker", "event_ticker", "ladder_key").unique("ticker")
    lad = lad.join(st, on="ticker", how="left")
    k = lad.group_by("ladder_key").agg(
        pl.col("strike_type").str.starts_with("less").sum().alias("n_less"),
        pl.col("strike_type").str.starts_with("greater").sum().alias("n_greater"),
        (pl.col("strike_type") == "between").sum().alias("n_between"), pl.len().alias("k"),
        pl.col("event_ticker").n_unique().alias("n_ev"), pl.col("event_ticker").first())
    return k.filter((pl.col("n_less") == 1) & (pl.col("n_greater") == 1) & (pl.col("n_between") == pl.col("k") - 2)
                    & (pl.col("n_ev") == 1)).select("ladder_key")


def shin_events(pre: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Event-leads whose legs partition the outcome space and are all quoted. Returns (event-leads,
    legs, completeness counts)."""
    me = pre.filter(pl.col("mutually_exclusive").fill_null(False) & (pl.col("n_legs") >= 2)
                    & pl.col("structure").is_in(["multi", "binary"]))
    me = me.with_columns(pl.when(pl.col("n_legs") == 2).then(pl.lit("two-leg")).otherwise(pl.lit("multi")).alias("kind"),
                         pl.col("event_ticker").alias("unit"))
    pk = partition_ladders(pre)
    ld = pre.filter(pl.col("ladder_key").is_in(pk["ladder_key"].to_list())).with_columns(
        pl.lit("partition ladder").alias("kind"), pl.col("ladder_key").alias("unit"))
    legs = pl.concat([me, ld], how="diagonal_relaxed")
    g = legs.group_by("unit", "lead_h").agg(
        pl.len().alias("k"), pl.col("n_legs").first(), pl.col("y").sum().alias("ys"), pl.col("kind").first(),
        pl.col("group").first(), pl.col("event_ticker").first(), pl.col("month").first(),
        pl.col("yes_ask").sum().alias("sum_ask"), pl.col("mid").sum().alias("sum_mid"),
        pl.col("yes_bid").sum().alias("sum_bid"), pl.col("spread").mean().alias("mean_spread"),
        pl.col("ts").first(), pl.col("ticker"), pl.col("yes_ask").alias("asks"), pl.col("mid").alias("mids"),
        pl.col("y").alias("yy"), pl.col("vol_7d").sum().alias("vol_7d"))
    counts = {"event_leads_all": g.height,
              "complete": int((g["k"] == g["n_legs"]).sum()),
              "complete_one_yes": int(((g["k"] == g["n_legs"]) & (g["ys"] == 1)).sum()),
              "by_kind": g.with_columns((pl.col("k") == pl.col("n_legs")).alias("complete"))
              .group_by("kind", "complete").agg(pl.len().alias("event_leads"), (pl.col("ys") == 1).sum().alias("one_yes"))
              .sort("kind", "complete").to_dicts()}
    g = g.filter((pl.col("k") == pl.col("n_legs")) & (pl.col("ys") == 1))
    counts["used_event_leads"] = g.height
    counts["used_units"] = g["unit"].n_unique()
    counts["sum_ask_below_1"] = int((g["sum_ask"] < 1 - 1e-9).sum())
    counts["sum_mid_below_1"] = int((g["sum_mid"] < 1 - 1e-9).sum())
    return g, legs, counts


def shin(pre: pl.DataFrame | None = None) -> dict[str, Any]:
    pre = load_pre() if pre is None else pre
    g, _, counts = shin_events(pre)
    out: dict[str, Any] = {"counts": counts}
    if g.height == 0:
        _write_json("shin", out)
        return out
    zs_a, zs_m, leg_rows, grid_rows = [], [], [], []
    for r in g.iter_rows(named=True):
        asks, mids, yy = (np.asarray(r[c], dtype=float) for c in ("asks", "mids", "yy"))
        sa, sm = shin_solve(asks), shin_solve(mids)
        zs_a.append(sa["z"])
        zs_m.append(sm["z"])
        grid = np.vstack([shin_fixed_z(mids, z) for z in Z_GRID])          # (nz, k) probabilities from mids
        win = int(np.argmax(yy))
        grid_rows.append(np.log(np.clip(grid[:, win], P_CLIP, 1)))
        for i in range(len(mids)):
            leg_rows.append({"unit": r["unit"], "event_ticker": r["event_ticker"], "lead_h": r["lead_h"],
                             "kind": r["kind"], "group": r["group"], "month": r["month"], "n_legs": r["n_legs"],
                             "y": float(yy[i]), "mid": float(mids[i]), "ask": float(asks[i]),
                             "norm_mid": float(mids[i] / r["sum_mid"]), "norm_ask": float(asks[i] / r["sum_ask"]),
                             "shin_ask": float(sa["probs"][i]), "shin_mid": float(sm["probs"][i]),
                             "z_ask": sa["z"], "grid": grid[:, i]})
    g = g.with_columns(pl.Series("z_ask", zs_a), pl.Series("z_mid", zs_m),
                       (pl.col("sum_ask") - 1).alias("over_ask"), (pl.col("sum_mid") - 1).alias("over_mid"))
    # each unit (event / ladder) weighs one across its leads
    g = g.with_columns((1.0 / pl.len().over("unit")).alias("w"))
    LW = np.vstack(grid_rows)                                               # (event-leads, nz) log p(winner)
    w = g["w"].to_numpy()
    units = g["unit"].to_numpy()
    L = pl.DataFrame([{k: v for k, v in r.items() if k != "grid"} for r in leg_rows])
    LG = np.vstack([r["grid"] for r in leg_rows])                           # (legs, nz) leg probabilities
    uw = g.select("unit", "lead_h", "w")
    L = L.join(uw, on=["unit", "lead_h"], how="left", maintain_order="left")   # rows stay aligned with LG

    def zfit(mask: np.ndarray) -> dict[str, Any]:
        r = zfit_grid(LW[mask], w[mask], units[mask])
        r.pop("draws", None)
        return r

    def zsum(x: pl.DataFrame) -> dict[str, Any]:
        e = x["unit"].to_numpy()
        ww = x["w"].to_numpy()
        za, zm = clustered(x["z_ask"].to_numpy(), e, ww), clustered(x["z_mid"].to_numpy(), e, ww)
        oa, om = clustered(100 * x["over_ask"].to_numpy(), e, ww), clustered(100 * x["over_mid"].to_numpy(), e, ww)
        return {"event_leads": x.height, "units": x["unit"].n_unique(), "legs_median": float(x["n_legs"].median()),
                "z_ask": za["mean"], "z_ask_se": za["se"], "z_ask_median": float(x["z_ask"].median()),
                "z_mid": zm["mean"], "z_mid_se": zm["se"], "z_mid_median": float(x["z_mid"].median()),
                "over_ask_pct": oa["mean"], "over_ask_se": oa["se"], "over_mid_pct": om["mean"], "over_mid_se": om["se"],
                "spread_c": 100 * _wmean(x["mean_spread"].to_numpy(), ww),
                "sum_ask_below_1": int((x["sum_ask"] < 1 - 1e-9).sum())}

    rows = []
    cuts = [("All", "all", np.ones(g.height, dtype=bool))]
    for kind in ("two-leg", "multi", "partition ladder"):
        cuts.append((kind, "kind", (g["kind"] == kind).to_numpy()))
    for grp in GROUPS:
        cuts.append((grp, "group", (g["group"] == grp).to_numpy()))
        for kind in ("two-leg", "multi", "partition ladder"):
            cuts.append((f"{grp}|{kind}", "group_kind", ((g["group"] == grp) & (g["kind"] == kind)).to_numpy()))
    for ld in LEADS_H:
        cuts.append((LEAD_LABEL[ld], "lead", (g["lead_h"] == float(ld)).to_numpy()))
    for lab, dim, m in cuts:
        x = g.filter(pl.Series(m))
        if x.height < 10 or x["unit"].n_unique() < 5:
            continue
        rows.append({"dim": dim, "cell": lab, **zsum(x), **{f"fit_{k}": v for k, v in zfit(m).items() if k != "n"}})
    out["cells"] = rows
    _write_csv("shin_cells", rows)
    _write_csv("shin_event_leads", g.drop("ticker", "asks", "mids", "yy").to_dicts())

    # does z mostly restate the spread? z_ask on mean half-spread, log legs and group dummies (multi + ladders)
    reg = g.filter(pl.col("kind") != "two-leg")
    out["z_regression"] = z_regression(reg)
    out["z_regression_all"] = z_regression(g)
    out["z_regression_info"] = z_regression(g, info_dummy=True)
    # H3's comparison: outcome-fitted z where private information is plausible vs where outcomes are public
    gi = g["group"].is_in(list(INFO_GROUPS)).to_numpy()
    gp = g["group"].is_in(list(PUBLIC_GROUPS)).to_numpy()
    fi, fp = zfit_grid(LW[gi], w[gi], units[gi]), zfit_grid(LW[gp], w[gp], units[gp])
    cmp_: dict[str, Any] = {"info": {k: v for k, v in fi.items() if k != "draws"},
                            "public": {k: v for k, v in fp.items() if k != "draws"}}
    if "draws" in fi and "draws" in fp:
        d = fi["draws"] - fp["draws"]
        cmp_["diff"] = {"z_ml": fi["z_ml"] - fp["z_ml"], "ci_lo": float(np.percentile(d, 2.5)),
                        "ci_hi": float(np.percentile(d, 97.5)), "share_draws_positive": float((d > 0).mean())}
    za_i = clustered(g.filter(pl.Series(gi))["z_ask"].to_numpy(), units[gi], w[gi])
    za_p = clustered(g.filter(pl.Series(gp))["z_ask"].to_numpy(), units[gp], w[gp])
    cmp_["z_ask_info"], cmp_["z_ask_public"] = za_i, za_p
    out["info_vs_public"] = cmp_

    # forecast comparison per leg: log score (binary), Brier, categorical log score of the winner
    split = half_split(pre)                  # the same month split as the weighting step
    out["split_month"] = split
    tr = (g["month"] <= split).to_numpy()
    fit_tr = zfit(tr)
    z_tr = fit_tr.get("z_ml", 0.0)
    jz = int(np.argmin(np.abs(Z_GRID - z_tr)))
    L = L.with_columns(pl.Series("shin_fit", LG[:, jz]))
    out["z_fit_train"] = fit_tr
    out["z_fit_test"] = zfit(~tr)
    sc = {}
    for lab, sub in (("all", L), ("test_half", L.filter(pl.col("month") > split)),
                     *((k, L.filter(pl.col("kind") == k)) for k in ("two-leg", "multi", "partition ladder"))):
        if sub.height == 0:
            continue
        e = sub["unit"].to_numpy()
        wl = sub["w"].to_numpy()          # every leg carries its event-lead's weight (per-leg average)
        y = sub["y"].to_numpy()
        base = log_score(sub["mid"].to_numpy(), y)
        res = {"legs": sub.height, "units": sub["unit"].n_unique()}
        for m in ("mid", "norm_mid", "norm_ask", "shin_ask", "shin_mid", "shin_fit"):
            p = sub[m].to_numpy()
            ls = log_score(p, y)
            d = clustered(ls - base, e, wl)
            ref = SAME_BASIS.get(m)
            dn = clustered(ls - log_score(sub[ref].to_numpy(), y), e, wl) if ref else {"mean": None, "t": None}
            res[m] = {"mean_ls": _wmean(ls, wl), "brier": _wmean((y - p) ** 2, wl), "diff_vs_mid": d["mean"],
                      "diff_t": d["t"], "vs_norm": ref, "diff_vs_norm": dn["mean"], "diff_vs_norm_t": dn["t"]}
        sc[lab] = res
    out["scores"] = sc
    sc_rows = [{"sample": s, "forecast": m, **v} for s, r in sc.items() for m, v in r.items() if isinstance(v, dict)]
    _write_csv("shin_scores", sc_rows)

    # where Shin moves probability: longshot legs (mid <= 10c), price minus outcome before and after
    ls = L.filter(pl.col("mid") <= LONGSHOT + 1e-9)
    lsr = {}
    for lab, sub in (("all", ls), ("test_half", ls.filter(pl.col("month") > split)),
                     *((k, ls.filter(pl.col("kind") == k)) for k in ("two-leg", "multi", "partition ladder"))):
        if sub.height < 20:
            continue
        e, wl = sub["unit"].to_numpy(), sub["w"].to_numpy()
        y = sub["y"].to_numpy()
        rr = {"legs": sub.height, "units": sub["unit"].n_unique()}
        for m in ("mid", "norm_mid", "shin_ask", "shin_fit"):
            c = clustered(100 * (sub[m].to_numpy() - y), e, wl)
            rr[m] = {"mispricing_c": c["mean"], "se": c["se"], "t": c["t"]}
        lsr[lab] = rr
    out["longshot_legs"] = lsr
    # Shin's mechanism: is longshot overpricing (normalised mid minus outcome) larger where z_ask is larger?
    out["ls_on_z"] = ls_on_z(ls)
    _write_json("shin", out)
    return out


def zfit_grid(lw: np.ndarray, w: np.ndarray, units: np.ndarray, boots: int = 400, seed: int = 20261111) -> dict[str, Any]:
    """Outcome-fitted Shin z: maximise the weighted categorical log score of the winning leg over
    ``Z_GRID`` (``lw``: event-leads x grid of log p(winner)); event-bootstrap percentile interval."""
    if len(w) < 20:
        return {"n": len(w)}
    obj = (w[:, None] * lw).sum(0) / w.sum()
    j = int(np.argmax(obj))
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(units, return_inverse=True)
    sums = np.zeros((len(uniq), lw.shape[1]))
    np.add.at(sums, inv, w[:, None] * lw)
    wsum = np.bincount(inv, weights=w)
    draws = np.empty(boots)
    for b in range(boots):
        idx = rng.integers(0, len(uniq), len(uniq))
        draws[b] = Z_GRID[int(np.argmax(sums[idx].sum(0) / wsum[idx].sum()))]
    return {"n": len(w), "units": len(uniq), "z_ml": float(Z_GRID[j]),
            "ci_lo": float(np.percentile(draws, 2.5)), "ci_hi": float(np.percentile(draws, 97.5)),
            "ls_gain_vs_norm": float(obj[j] - obj[0]), "at_upper_bound": j == len(Z_GRID) - 1, "draws": draws}


def z_regression(g: pl.DataFrame, info_dummy: bool = False) -> dict[str, Any]:
    if g.height < 30:
        return {"n": g.height}
    grp = g["group"].to_numpy()
    levels = [x for x in MAIN_GROUPS if (grp == x).sum() >= 10]
    base = levels[0] if "Sports" not in levels else "Sports"
    others = [x for x in levels if x != base]
    keep = np.isin(grp, levels)
    dummies = ([np.isin(grp[keep], INFO_GROUPS).astype(float)] if info_dummy
               else [(grp[keep] == x).astype(float) for x in others])
    X = np.column_stack([np.ones(keep.sum()), 100 * g["mean_spread"].to_numpy()[keep] / 2,
                         np.log(g["n_legs"].to_numpy()[keep].astype(float))] + dummies)
    r = ols_cluster(g["z_ask"].to_numpy()[keep], X, g["unit"].to_numpy()[keep], g["w"].to_numpy()[keep])
    if r.get("coef") is None:
        return {"n": int(keep.sum())}
    names = ["const", "half_spread_c", "log_legs", *(["info_group"] if info_dummy else [f"group:{x}" for x in others])]
    pred = np.asarray(X) @ np.asarray(r["coef"])
    yv = g["z_ask"].to_numpy()[keep]
    ww = g["w"].to_numpy()[keep]
    r2 = 1 - (ww * (yv - pred) ** 2).sum() / (ww * (yv - _wmean(yv, ww)) ** 2).sum()
    return {"n": r["n"], "clusters": r["clusters"], "base_group": base, "r2": float(r2),
            "coef": dict(zip(names, r["coef"], strict=True)), "se": dict(zip(names, r["se"], strict=True)),
            "t": dict(zip(names, r["t"], strict=True))}


def ls_on_z(ls: pl.DataFrame) -> dict[str, Any]:
    if ls.height < 30:
        return {"n": ls.height}
    y = 100 * (ls["norm_mid"] - ls["y"]).to_numpy()
    X = np.column_stack([np.ones(ls.height), ls["z_ask"].to_numpy(), np.log(ls["n_legs"].to_numpy().astype(float))])
    r = ols_cluster(y, X, ls["unit"].to_numpy(), ls["w"].to_numpy())
    if r.get("coef") is None:
        return {"n": ls.height}
    return {"n": r["n"], "clusters": r["clusters"], "coef_z": r["coef"][1], "se_z": r["se"][1], "t_z": r["t"][1],
            "units": "cents of longshot overpricing per unit of z_ask"}


# ---- 4. heterogeneous beliefs ----------------------------------------------------------------------
CHARS = ("log_outcomes", "turnover", "spread_c", "log_lead", "anchored")
CHARS_SPLIT = ("log_outcomes", "turnover", "spread_c", "log_lead", "anchor_sports", "anchor_weather")
DUMMIES = {"anchored": "anchored_f", "anchor_sports": "anchor_sports", "anchor_weather": "anchor_weather"}


def characteristics(pre: pl.DataFrame) -> pl.DataFrame:
    """Ex-ante characteristics of each quote: log number of outcomes, turnover = log((1 + 24h volume)
    / (1 + open interest)), quoted spread (cents), log lead (hours), forecast anchor (0/1)."""
    return pre.with_columns(
        pl.col("n_out").log().alias("log_outcomes"),
        ((1 + pl.col("vol_24h").fill_null(0).clip(lower_bound=0)).log()
         - (1 + pl.col("open_interest").fill_null(0).clip(lower_bound=0)).log()).alias("turnover"),
        (100 * pl.col("spread")).alias("spread_c"), pl.col("lead_h").log().alias("log_lead"),
        pl.col("anchored").cast(pl.Float64).alias("anchored_f"),
        (pl.col("group") == "Sports").cast(pl.Float64).alias("anchor_sports"),
        (pl.col("group") == "Weather").cast(pl.Float64).alias("anchor_weather"))


def _std(a: np.ndarray) -> tuple[np.ndarray, float, float]:
    m, s = float(np.mean(a)), float(np.std(a))
    return (a - m) / (s if s > 0 else 1.0), m, s


def slope_interactions(x: pl.DataFrame, chars: list[str], group_slopes: bool, w: str = "w_m") -> dict[str, Any]:
    """Logit of y on logit(mid) with interactions: logit P = a_g + (b + sum_k b_k Z_k) logit q + sum_k c_k Z_k,
    Z_k standardised (anchored is 0/1, unstandardised). b_k > 0: a steeper calibration slope (stronger bias)
    where Z_k is higher. With group_slopes, the base slope is group-specific."""
    lq = logit(x["mid"].to_numpy())
    grp = x["group"].to_numpy()
    levels = [g for g in GROUPS if (grp == g).sum() > 0]
    D = np.column_stack([(grp == g).astype(float) for g in levels])
    Z, sd = [], {}
    for c in chars:
        a = x[DUMMIES.get(c, c)].to_numpy().astype(float)
        if c in DUMMIES:
            Z.append(a)
            sd[c] = 1.0
        else:
            z, _, s = _std(a)
            Z.append(z)
            sd[c] = s
    Zm = np.column_stack(Z)
    slope = D * lq[:, None] if group_slopes else lq[:, None]
    X = np.column_stack([D, slope, Zm * lq[:, None], Zm])
    r = logit_design(x["y"].cast(pl.Float64).to_numpy(), X, x["event_ticker"].to_numpy(), x[w].to_numpy())
    if r.get("coef") is None:
        return {"n": r["n"]}
    b, se = np.asarray(r["coef"]), np.asarray(r["se"])
    o = len(levels) + (len(levels) if group_slopes else 1)
    res = {"n": r["n"], "clusters": r["clusters"], "sd": sd, "interactions": {}}
    for j, c in enumerate(chars):
        res["interactions"][c] = {"coef": float(b[o + j]), "se": float(se[o + j]),
                                  "t": float(b[o + j] / se[o + j]) if se[o + j] > 0 else None}
    if not group_slopes:
        res["base_slope"] = {"coef": float(b[len(levels)]), "se": float(se[len(levels)])}
    return res


def ls_regression(S: pl.DataFrame, chars: list[str], group_fe: bool, w: str = "w_m") -> dict[str, Any]:
    """Longshot sides (q_mid <= 10c): 100 x (q_mid - won), cents of overpricing per contract, on
    standardised characteristics with bucket (and group) fixed effects; event-clustered."""
    x = S.filter(pl.col("q_mid") <= LONGSHOT + 1e-9)
    if x.height < 100:
        return {"n": x.height}
    bk = x["bucket"].to_numpy()
    cols = [(bk == b).astype(float) for b in LS_BUCKETS if (bk == b).any()]
    if group_fe:
        grp = x["group"].to_numpy()
        cols += [(grp == g).astype(float) for g in GROUPS[1:] if (grp == g).any()]
    Z = []
    for c in chars:
        a = x[DUMMIES.get(c, c)].to_numpy().astype(float)
        Z.append(a if c in DUMMIES else _std(a)[0])
    X = np.column_stack(cols + Z)
    r = ols_cluster(100 * (x["q_mid"] - x["won"]).to_numpy(), X, x["event_ticker"].to_numpy(), x[w].to_numpy())
    if r.get("coef") is None:
        return {"n": x.height}
    o = len(cols)
    return {"n": r["n"], "clusters": r["clusters"],
            "coef": {c: {"coef": r["coef"][o + j], "se": r["se"][o + j], "t": r["t"][o + j]} for j, c in enumerate(chars)}}


def _varies(x: pl.DataFrame, col: str, min_events: int = 10) -> bool:
    """A characteristic is identified within a sample when at least ``min_events`` events lie strictly
    above and strictly below its median (event means)."""
    ev = x.group_by("event_ticker").agg(pl.col(col).mean())[col]
    med = ev.median()
    return med is not None and int((ev > med + 1e-12).sum()) >= min_events and int((ev < med - 1e-12).sum()) >= min_events


def beliefs(pre: pl.DataFrame | None = None) -> dict[str, Any]:
    pre = load_pre() if pre is None else pre
    X = characteristics(pre)
    S = side_frame(pre).join(X.select("ticker", "lead_h", "log_outcomes", "turnover", "spread_c", "log_lead",
                                      "anchored_f", "anchor_sports", "anchor_weather"), on=["ticker", "lead_h"], how="left")
    within = [c for c in CHARS if c != "anchored"]
    out: dict[str, Any] = {
        "describe": {c: {"mean": float(X[c].mean()), "sd": float(X[c].std()), "p10": float(X[c].quantile(0.1)),
                         "p90": float(X[c].quantile(0.9))} for c in (*within, "anchored_f")},
        "pooled_anchor": slope_interactions(X, list(CHARS), group_slopes=False),
        "within_group": slope_interactions(X, within, group_slopes=True),
        "pooled_anchor_vol": slope_interactions(X, list(CHARS), group_slopes=False, w="w_v"),
        "pooled_anchor_split": slope_interactions(X, list(CHARS_SPLIT), group_slopes=False),
        "ls_pooled_anchor": ls_regression(S, list(CHARS), group_fe=False),
        "ls_pooled_anchor_split": ls_regression(S, list(CHARS_SPLIT), group_fe=False),
        "ls_within_group": ls_regression(S, within, group_fe=True),
        "ls_within_group_drop_top5": ls_regression(drop_top_events(S.filter(pl.col("q_mid") <= LONGSHOT + 1e-9),
                                                                   "pc_mid"), within, group_fe=True)}
    split = half_split(pre)
    out["split_month"] = split
    out["halves"] = {lab: slope_interactions(X.filter(cond), list(CHARS), group_slopes=False)
                     for lab, cond in (("first", pl.col("month") <= split), ("second", pl.col("month") > split))}
    # by group: within-group slope interactions (sign test across groups)
    by_g = {}
    for grp in MAIN_GROUPS:
        xg = X.filter(pl.col("group") == grp)
        if not _enough(xg):
            continue
        chars_g = [c for c in within if _varies(xg, c)]
        by_g[grp] = slope_interactions(xg, chars_g, group_slopes=False)
    out["by_group"] = by_g
    signs = {}
    for c in within:
        cs = [v["interactions"][c] for v in by_g.values() if "interactions" in v and c in v["interactions"]]
        pos = sum(1 for v in cs if v["coef"] > 0)
        signs[c] = {"groups": len(cs), "positive": pos,
                    "sign_test_p": float(sps.binomtest(pos, len(cs), 0.5).pvalue) if cs else None,
                    "sig_pos": sum(1 for v in cs if (v["t"] or 0) > 1.96), "sig_neg": sum(1 for v in cs if (v["t"] or 0) < -1.96)}
    out["sign_test"] = signs
    rows = []
    for spec in ("pooled_anchor", "within_group", "pooled_anchor_vol", "pooled_anchor_split"):
        for c, v in out[spec].get("interactions", {}).items():
            rows.append({"spec": spec, "variable": c, **v, "n": out[spec]["n"], "clusters": out[spec]["clusters"]})
    for grp, r in by_g.items():
        for c, v in r.get("interactions", {}).items():
            rows.append({"spec": f"group:{grp}", "variable": c, **v, "n": r["n"], "clusters": r["clusters"]})
    for spec in ("ls_pooled_anchor", "ls_pooled_anchor_split", "ls_within_group", "ls_within_group_drop_top5"):
        for c, v in out[spec].get("coef", {}).items():
            rows.append({"spec": spec, "variable": c, **v, "n": out[spec]["n"], "clusters": out[spec]["clusters"]})
    _write_csv("beliefs_coef", rows)
    _write_json("beliefs", out)
    return out


# ---- 5. synthesis ----------------------------------------------------------------------------------
def _cell(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any] | None:
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def counterfactual(pre: pl.DataFrame, wfit: dict[str, Any]) -> dict[str, Any]:
    """Share of the gross longshot mispricing (q_mid - won, longshot sides of the test half) removed by
    (a) the taker band, (b) the LLO map fitted on the first half, (c) the Prelec map, (d) Shin (ask) and
    normalisation in partition events. The maps are alternatives, not additive components."""
    split = wfit["split_month"]
    test = with_weights(pre.filter(pl.col("month") > split))
    B = band_components(side_frame(test), RHO_MAIN).filter(pl.col("q_mid") <= LONGSHOT + 1e-9)
    f = wfit["oos"]["first_to_second"]["fits"]
    mid_yes = ((B["yes_bid"] + B["yes_ask"]) / 2).to_numpy()
    p_yes_llo = prob_llo(mid_yes, np.array([f["llo"]["alpha"], f["llo"]["beta"]]))
    p_yes_pr = prob_prelec(mid_yes, np.array([f["prelec"]["ln_delta"], f["prelec"]["ln_kappa"]]))
    is_yes = (B["side"] == "yes").to_numpy()
    pi_llo = np.where(is_yes, p_yes_llo, 1 - p_yes_llo)
    pi_pr = np.where(is_yes, p_yes_pr, 1 - p_yes_pr)
    w, e = B["w_m"].to_numpy(), B["event_ticker"].to_numpy()
    q, won = B["q_mid"].to_numpy(), B["won"].to_numpy()
    gross = clustered(100 * (q - won), e, w)
    out: dict[str, Any] = {"sample": f"longshot sides (q_mid <= 10c), months after {split}", "n": B.height,
                           "clusters": gross["clusters"], "gross_c": gross["mean"], "gross_se": gross["se"]}
    band_c = 100 * _wmean((B["c_floor"] + B["c_spread"] + B["c_fee"] + B["c_cap"]).to_numpy(), w)
    out["band"] = {"removed_c": min(band_c, gross["mean"]), "band_c": band_c,
                   "share": min(band_c, gross["mean"]) / gross["mean"] if gross["mean"] > 0 else None}
    for lab, pi in (("llo", pi_llo), ("prelec", pi_pr)):
        res = clustered(100 * (pi - won), e, w)
        out[lab] = {"residual_c": res["mean"], "residual_se": res["se"], "residual_t": res["t"],
                    "removed_c": gross["mean"] - res["mean"], "share": 1 - res["mean"] / gross["mean"] if gross["mean"] > 0 else None}
    # the same with per-group LLO maps fitted on the first half
    first = with_weights(pre.filter(pl.col("month") <= split))
    pig = np.full(B.height, np.nan)
    gg = B["group"].to_numpy()
    for grp in np.unique(gg):
        tr = first.filter(pl.col("group") == grp)
        ff = fit_llo(*_yes(tr, "w_m")) if _enough(tr) else f["llo"]
        if ff.get("beta") is None:
            ff = f["llo"]
        m = gg == grp
        py = prob_llo(mid_yes[m], np.array([ff["alpha"], ff["beta"]]))
        pig[m] = np.where(is_yes[m], py, 1 - py)
    res = clustered(100 * (pig - won), e, w)
    out["llo_by_group"] = {"residual_c": res["mean"], "residual_t": res["t"], "removed_c": gross["mean"] - res["mean"],
                           "share": 1 - res["mean"] / gross["mean"] if gross["mean"] > 0 else None}
    return out


def synthesis() -> dict[str, Any]:
    pre = load_pre()
    bd, wf, sh, be = (_read_json(n) for n in ("band", "weighting", "shin", "beliefs"))
    out: dict[str, Any] = {"counterfactual": counterfactual(pre, wf)}
    # Shin counterfactual in partition events (longshot legs, second half: z is fitted on the first)
    lsl = sh.get("longshot_legs", {}).get("test_half")
    if lsl:
        g0 = lsl["mid"]["mispricing_c"]
        out["counterfactual"]["shin_partition"] = {
            "legs": lsl["legs"], "units": lsl["units"], "gross_c": g0,
            **{m: {"residual_c": lsl[m]["mispricing_c"], "residual_t": lsl[m]["t"],
                   "share": 1 - lsl[m]["mispricing_c"] / g0 if g0 > 0 else None} for m in ("norm_mid", "shin_ask", "shin_fit")}}
    out["criteria"] = criteria(bd, wf, sh, be)
    out["scorecard"] = scorecard(bd, wf, sh, be)
    _write_csv("scorecard", out["scorecard"])
    _write_csv("criteria", [{"hypothesis": h, **c} for h, v in out["criteria"].items() for c in v["tests"]])
    cf = out["counterfactual"]
    dec = [{"map": "taker band (rho=5%)", "share": cf["band"]["share"], "removed_c": cf["band"]["removed_c"]},
           {"map": "LLO, pooled, first-half fit", "share": cf["llo"]["share"], "removed_c": cf["llo"]["removed_c"],
            "residual_t": cf["llo"]["residual_t"]},
           {"map": "LLO, by group, first-half fit", "share": cf["llo_by_group"]["share"],
            "removed_c": cf["llo_by_group"]["removed_c"], "residual_t": cf["llo_by_group"]["residual_t"]},
           {"map": "Prelec, pooled, first-half fit", "share": cf["prelec"]["share"], "removed_c": cf["prelec"]["removed_c"],
            "residual_t": cf["prelec"]["residual_t"]}]
    if "shin_partition" in cf:
        for m, lab in (("norm_mid", "normalised mids (partition events)"), ("shin_ask", "Shin from asks (partition events)"),
                       ("shin_fit", "Shin, outcome-fitted z (partition events)")):
            dec.append({"map": lab, "share": cf["shin_partition"][m]["share"],
                        "residual_t": cf["shin_partition"][m]["residual_t"]})
    _write_csv("decomposition", dec)
    _write_json("synthesis", out)
    return out


CRITERIA_TEX = {   # short LaTeX labels of the criteria for tab:structural-scorecard
    'longshot mispricing inside the taker band (residual t < 1.96)':
        r'Longshot mispricing inside the taker band ($\bar R$, \cents)',
    'inside the taker band, volume-weighted':
        r'Inside the taker band, volume-weighted ($\bar R$, \cents)',
    'inside the taker band in every group':
        'Inside the taker band in every group (Bonferroni)',
    'gross longshot mispricing rises with horizon (slope t > 1.96)':
        r'Gross mispricing rises with the lead (\cents{} per log hour)',
    'concentrated at the tick floor (floor share of longshot mispricing >= 50%)':
        r'Concentrated at the tick floor (floor share $\ge0.5$)',
    'pooled LLO gamma < 1 (t > 1.96)':
        r'Pooled $\gamma<1$ ($\gamma$, $t_{\gamma<1}$)',
    'gamma < 1 in every group (point estimates)':
        r'$\gamma<1$ in every group (point estimates)',
    'equal slopes across groups not rejected (Wald p > 0.05)':
        r'Equal $\gamma$ across groups not rejected (Wald $\chi^2$)',
    'LLO beats calibrated prices out of sample, both directions (t > 1.96)':
        'LLO beats calibrated prices out of sample, both directions',
    'one pooled weighting function predicts as well as group-specific ones out of sample':
        'Pooled LLO predicts as well as group-specific LLO',
    'Shin probabilities from the asks remove the longshot overpricing in partition events':
        r'Shin from the asks removes the longshot-leg overpricing (\cents)',
    'z from the asks is higher in information groups, given the spread (t > 1.96)':
        r'$\shinz$ from the asks higher in information groups, given the spread',
    'outcome-fitted z higher in information than in public-outcome groups (bootstrap CI > 0)':
        r'Fitted $\shinz$ higher in information than in public groups',
    "longshot overpricing rises with an event's z (t > 1.96)":
        r'Longshot overpricing rises with $\shinz$ (\cents{} per unit)',
    'steeper slope with more outcomes, within groups (t > 1.96)':
        'Steeper slope with more outcomes, within groups',
    'steeper slope with higher turnover, within groups (t > 1.96)':
        'Steeper slope with more turnover, within groups',
    'flatter slope where a public forecast anchors beliefs (t < -1.96)':
        'Flatter slope with a public forecast anchor',
    'turnover effect positive in most groups (sign test)':
        'Turnover interaction positive in most groups',
}


def _verdict(tests: list[dict[str, Any]]) -> str:
    ok = [t["pass"] for t in tests if t["pass"] is not None]
    if not ok:
        return "not identified"
    return "supported" if all(ok) else ("partly supported" if any(ok) else "not supported")


def criteria(bd: dict[str, Any], wf: dict[str, Any], sh: dict[str, Any], be: dict[str, Any]) -> dict[str, Any]:
    """Each hypothesis's predictions as pass/fail tests on the pooled evidence (5% two-sided level)."""
    cells = bd["cells"]
    ls = _cell(cells, rho=RHO_MAIN, weight="market", group="All", lead="pooled", bucket="LS <=10c") or {}
    lsv = _cell(cells, rho=RHO_MAIN, weight="volume", group="All", lead="pooled", bucket="LS <=10c") or {}
    hz = bd["horizon_slope"]
    grp_cells = [_cell(cells, rho=RHO_MAIN, weight="market", group=g, lead="pooled", bucket="LS <=10c") for g in MAIN_GROUPS]
    n_g = len([c for c in grp_cells if c])
    t_bonf = float(sps.norm.ppf(1 - 0.05 / max(n_g, 1)))      # one-sided, Bonferroni across groups
    inside = [c for c in grp_cells if c and (c["resid_t"] or 0) < t_bonf]
    floor_share = (ls.get("floor_c") or 0) / ls["gross_c"] if ls.get("gross_c") else None
    h1 = [
        {"test": "longshot mispricing inside the taker band (residual t < 1.96)", "stat": ls.get("resid_c"),
         "t": ls.get("resid_t"), "pass": (ls.get("resid_t") or 0) < 1.96},
        {"test": "inside the taker band, volume-weighted", "stat": lsv.get("resid_c"), "t": lsv.get("resid_t"),
         "pass": (lsv.get("resid_t") or 0) < 1.96 if lsv else None},
        {"test": "inside the taker band in every group", "stat": len(inside), "of": n_g, "t_crit": t_bonf,
         "pass": len(inside) == n_g},
        {"test": "gross longshot mispricing rises with horizon (slope t > 1.96)", "stat": hz["gross"].get("slope"),
         "t": hz["gross"].get("t"), "pass": (hz["gross"].get("t") or 0) > 1.96},
        {"test": "concentrated at the tick floor (floor share of longshot mispricing >= 50%)", "stat": floor_share,
         "pass": floor_share is not None and floor_share >= 0.5}]
    wc = {r["cell"]: r for r in wf["cells"] if r["weight"] == "market" and r["dim"] in ("all", "group")}
    allc = wc.get("All", {})
    gams = [wc[g]["llo_gamma"] for g in MAIN_GROUPS if g in wc and wc[g].get("llo_gamma") is not None]
    wg = wf["wald"]["group_slope"]
    o1, o2 = wf["oos"]["first_to_second"]["models"], wf["oos"]["second_to_first"]["models"]
    h2 = [
        {"test": "pooled LLO gamma < 1 (t > 1.96)", "stat": allc.get("llo_gamma"), "t": -(allc.get("llo_t_gamma1") or 0),
         "pass": -(allc.get("llo_t_gamma1") or 0) > 1.96},
        {"test": "gamma < 1 in every group (point estimates)", "stat": sum(1 for x in gams if x < 1), "of": len(gams),
         "pass": all(x < 1 for x in gams)},
        {"test": "equal slopes across groups not rejected (Wald p > 0.05)", "stat": wg.get("wald"), "p": wg.get("p"),
         "pass": (wg.get("p") or 0) > 0.05},
        {"test": "LLO beats calibrated prices out of sample, both directions (t > 1.96)",
         "stat": [o1["llo"]["diff_vs_cal"], o2["llo"]["diff_vs_cal"]], "t": [o1["llo"]["diff_t"], o2["llo"]["diff_t"]],
         "pass": min(o1["llo"]["diff_t"] or 0, o2["llo"]["diff_t"] or 0) > 1.96},
        {"test": "one pooled weighting function predicts as well as group-specific ones out of sample",
         "stat": [o1["llo_by_group"]["diff_vs_llo"], o2["llo_by_group"]["diff_vs_llo"]],
         "t": [o1["llo_by_group"]["diff_vs_llo_t"], o2["llo_by_group"]["diff_vs_llo_t"]],
         "pass": max(o1["llo_by_group"]["diff_vs_llo_t"] or 0, o2["llo_by_group"]["diff_vs_llo_t"] or 0) < 1.96}]
    ivp = sh.get("info_vs_public", {})
    lsl = sh.get("longshot_legs", {}).get("all", {})
    zr = sh.get("z_regression_info", {})
    h3 = [
        {"test": "Shin probabilities from the asks remove the longshot overpricing in partition events",
         "stat": (lsl.get("shin_ask") or {}).get("mispricing_c"), "t": (lsl.get("shin_ask") or {}).get("t"),
         "pass": abs((lsl.get("shin_ask") or {}).get("t") or 99) < 1.96 if lsl else None},
        {"test": "z from the asks is higher in information groups, given the spread (t > 1.96)",
         "stat": (zr.get("coef") or {}).get("info_group"), "t": (zr.get("t") or {}).get("info_group"),
         "pass": ((zr.get("t") or {}).get("info_group") or 0) > 1.96 if zr.get("coef") else None},
        {"test": "outcome-fitted z higher in information than in public-outcome groups (bootstrap CI > 0)",
         "stat": (ivp.get("diff") or {}).get("z_ml"), "ci": [(ivp.get("diff") or {}).get("ci_lo"), (ivp.get("diff") or {}).get("ci_hi")],
         "pass": ((ivp.get("diff") or {}).get("ci_lo") or -1) > 0 if ivp.get("diff") else None},
        {"test": "longshot overpricing rises with an event's z (t > 1.96)", "stat": sh.get("ls_on_z", {}).get("coef_z"),
         "t": sh.get("ls_on_z", {}).get("t_z"), "pass": (sh.get("ls_on_z", {}).get("t_z") or 0) > 1.96 if sh.get("ls_on_z", {}).get("coef_z") is not None else None}]
    pa, wgp = be["pooled_anchor"]["interactions"], be["within_group"]["interactions"]
    st = be["sign_test"]
    h4 = [
        {"test": "steeper slope with more outcomes, within groups (t > 1.96)", "stat": wgp["log_outcomes"]["coef"],
         "t": wgp["log_outcomes"]["t"], "pass": (wgp["log_outcomes"]["t"] or 0) > 1.96},
        {"test": "steeper slope with higher turnover, within groups (t > 1.96)", "stat": wgp["turnover"]["coef"],
         "t": wgp["turnover"]["t"], "pass": (wgp["turnover"]["t"] or 0) > 1.96},
        {"test": "flatter slope where a public forecast anchors beliefs (t < -1.96)", "stat": pa["anchored"]["coef"],
         "t": pa["anchored"]["t"], "pass": (pa["anchored"]["t"] or 0) < -1.96},
        {"test": "turnover effect positive in most groups (sign test)", "stat": st["turnover"]["positive"],
         "of": st["turnover"]["groups"], "pass": st["turnover"]["positive"] > st["turnover"]["groups"] / 2}]
    out = {}
    for h, tests in (("H1", h1), ("H2", h2), ("H3", h3), ("H4", h4)):
        out[h] = {"tests": tests, "verdict": _verdict(tests), "passed": sum(1 for t in tests if t["pass"]),
                  "of": sum(1 for t in tests if t["pass"] is not None)}
    return out


def scorecard(bd: dict[str, Any], wf: dict[str, Any], sh: dict[str, Any], be: dict[str, Any]) -> list[dict[str, Any]]:
    """The statistic behind each hypothesis, by group, with a mark: + consistent with the hypothesis's
    prediction for that group, - inconsistent, 0 inconclusive (|t| < 1.96 or CI covering the benchmark)."""
    cells = bd["cells"]
    wcells = [r for r in wf["cells"] if r["weight"] == "market"]
    rows = []
    for grp in ("All", *MAIN_GROUPS):
        b = _cell(cells, rho=RHO_MAIN, weight="market", group=grp, lead="pooled", bucket="LS <=10c")
        wc = _cell(wcells, dim="all" if grp == "All" else "group", cell=grp)
        sc = next((r for r in sh.get("cells", []) if r["dim"] == ("all" if grp == "All" else "group") and r["cell"] == grp), None)
        bg = be["by_group"].get(grp) if grp != "All" else be["within_group"]
        row: dict[str, Any] = {"group": grp}
        if b:      # H1: inside the taker band; + if the residual is not significantly positive
            t = b["resid_t"] or 0.0
            row.update({"h1_gross_c": b["gross_c"], "h1_band_c": b["band_c"], "h1_share": b["share_in_band"],
                        "h1_resid_c": b["resid_c"], "h1_resid_t": t, "h1_resid_mk_c": b["resid_mk_c"],
                        "h1_resid_mk_t": b["resid_mk_t"], "h1_floor_c": b["floor_c"], "h1_n": b["n"],
                        "h1_clusters": b["clusters"], "h1_mark": "+" if t < 1.96 else "-"})
        if wc and wc.get("llo_gamma") is not None:   # H2: gamma < 1
            tg = -(wc["llo_t_gamma1"] or 0.0)
            row.update({"h2_gamma": wc["llo_gamma"], "h2_gamma_se": wc["llo_se_gamma"], "h2_t_gamma_lt1": tg,
                        "h2_prelec_gamma": wc.get("prelec_gamma"), "h2_prelec_se": wc.get("prelec_se_gamma"),
                        "h2_clusters": wc.get("clusters"),
                        "h2_mark": "+" if tg > 1.96 else ("-" if tg < -1.96 else "0")})
        if sc and sc.get("fit_z_ml") is not None:     # H3: z high where information is private, low where public
            lo, hi = sc.get("fit_ci_lo"), sc.get("fit_ci_hi")
            row.update({"h3_units": sc["units"], "h3_z_ask": sc["z_ask"], "h3_z_mid": sc["z_mid"],
                        "h3_z_ml": sc.get("fit_z_ml"), "h3_z_ml_lo": lo, "h3_z_ml_hi": hi})
            if grp in INFO_GROUPS:
                row["h3_mark"] = "+" if lo > 0 else "0"
            elif grp in PUBLIC_GROUPS:
                row["h3_mark"] = "-" if lo > 0 else "+"
        if bg and "interactions" in bg:               # H4: steeper slope with turnover (and outcomes)
            it = bg["interactions"]
            for c in ("log_outcomes", "turnover"):
                if c in it:
                    row[f"h4_{c}_coef"], row[f"h4_{c}_t"] = it[c]["coef"], it[c]["t"]
            tt = it.get("turnover", {}).get("t")
            if tt is not None:
                row["h4_mark"] = "+" if tt > 1.96 else ("-" if tt < -1.96 else "0")
        rows.append(row)
    return rows


# ---- summary, verdict -----------------------------------------------------------------------------
def summarize() -> dict[str, Any]:
    S: dict[str, Any] = {"generated": datetime.now(UTC).isoformat(timespec="seconds"), "input": panel_status()}
    for n in ("band", "weighting", "shin", "beliefs", "synthesis"):
        S[n] = _read_json(n)
    S["band"].pop("cells", None)             # in band_cells.csv
    S["weighting"].pop("cells", None)        # in weighting_fits.csv
    pre = load_pre()
    S["coverage"] = {"quotes": pre.height, "markets": pre["ticker"].n_unique(), "events": pre["event_ticker"].n_unique(),
                     "first_month": pre["month"].min(), "last_month": pre["month"].max(),
                     "by_group": pre.group_by("group").agg(pl.len().alias("quotes"), pl.col("ticker").n_unique().alias("markets"),
                                                           pl.col("event_ticker").n_unique().alias("events")).sort("group").to_dicts()}
    S["strategy_metrics"] = ("No trading strategy is tested in this workstream; AGENTS.md rule 6 (Sharpe/vol from "
                             "capped-book ledgers) does not apply and no ledger is built.")
    (out_dir() / "summary.json").write_text(json.dumps(_jsonable(S), indent=1))
    (out_dir() / "verdict.md").write_text(verdict_text(S))
    return S


def _p(x: Any, d: int = 2) -> str:
    if isinstance(x, list | tuple):
        return "[" + ", ".join(_p(v, d) for v in x) + "]"
    return "n/a" if x is None else (f"{x:.{d}f}" if isinstance(x, float | int) else str(x))


NOT_IDENTIFIED = ("Not identified: risk-love vs probability weighting (no bet-size or exotic-bet data); probability "
                  "weighting vs heterogeneous beliefs from prices and outcomes alone (both fit any monotone calibration "
                  "curve); the insider share separately from spread compensation for adverse selection; the true cost "
                  "of capital (band shown for rho = 0, 5%, 10%).")


def verdict_text(S: dict[str, Any]) -> str:
    sy = S["synthesis"]
    cf = sy["counterfactual"]
    cov, inp = S["coverage"], S["input"]
    status = "final" if inp["panel_ready"] else "PROVISIONAL"
    lines = ["# Structural decomposition of the longshot bias: verdict", "",
             (f"Input: xs panel ({status}, {inp['panel_mtime']}); {cov['quotes']:,} pre-event quotes, "
              f"{cov['markets']:,} markets, {cov['events']:,} events, {cov['first_month']}..{cov['last_month']}."), ""]
    names = {"H1": "mechanical (tick, spread, fee, capital band)", "H2": "probability weighting (LLO / Prelec)",
             "H3": "Shin insiders", "H4": "heterogeneous beliefs"}
    for h, v in sy["criteria"].items():
        lines.append(f"## {h} {names[h]}: {v['verdict']} ({v['passed']}/{v['of']} predictions)")
        for t in v["tests"]:
            mark = "n/a" if t["pass"] is None else ("yes" if t["pass"] else "no")
            extra = "".join(f", {k}={_p(t[k], 3)}" for k in ("t", "p", "of", "ci") if k in t)
            lines.append(f"- [{mark}] {t['test']}: {_p(t.get('stat'), 3)}{extra}")
        lines.append("")
    lines.append("## Counterfactual shares of the gross longshot mispricing (test half, alternative maps)")
    lines.append(f"- gross {_p(cf['gross_c'])}c per contract on {cf['n']:,} longshot sides ({cf['clusters']} events)")
    for k in ("band", "llo", "llo_by_group", "prelec"):
        lines.append(f"- {k}: {_p(100 * (cf[k]['share'] or 0), 0)}%")
    if "shin_partition" in cf:
        sp = cf["shin_partition"]
        for k in ("norm_mid", "shin_ask", "shin_fit"):
            lines.append(f"- partition events, {k}: {_p(100 * (sp[k]['share'] or 0), 0)}%")
    lines += ["", "## Scorecard by group (+ consistent, - inconsistent, 0 inconclusive)", "",
              "| group | H1 resid c (t) | H2 gamma (t<1) | H3 z_ml [CI] | H4 turnover (t) |", "|---|---|---|---|---|"]
    for r in sy["scorecard"]:
        lines.append(
            f"| {r['group']} | {r.get('h1_mark', '')} {_p(r.get('h1_resid_c'))} ({_p(r.get('h1_resid_t'), 1)}) "
            f"| {r.get('h2_mark', '')} {_p(r.get('h2_gamma'))} ({_p(r.get('h2_t_gamma_lt1'), 1)}) "
            f"| {r.get('h3_mark', '')} {_p(r.get('h3_z_ml'), 3)} [{_p(r.get('h3_z_ml_lo'), 3)}, {_p(r.get('h3_z_ml_hi'), 3)}] "
            f"| {r.get('h4_mark', '')} {_p(r.get('h4_turnover_coef'), 3)} ({_p(r.get('h4_turnover_t'), 1)}) |")
    lines += ["", NOT_IDENTIFIED, "", S["strategy_metrics"], ""]
    return "\n".join(lines)


def run_all() -> None:
    pre = load_pre()
    print(f"structural: {pre.height:,} pre-event quotes, {pre['ticker'].n_unique():,} markets ({panel_status()})", flush=True)
    for name, fn in (("band", band), ("weighting", weighting), ("shin", shin), ("beliefs", beliefs)):
        fn(pre)
        print(f"  {name} done", flush=True)
    synthesis()
    print("  synthesis done", flush=True)
    summarize()
    print("summary.json, verdict.md written", flush=True)


# ---- LaTeX blocks (auto-generated, between % BEGIN AUTO:<name> and % END AUTO:<name>) ----------------
def replace_block(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(% BEGIN AUTO:{re.escape(name)}\n).*?(% END AUTO:{re.escape(name)})", re.DOTALL)
    if not pat.search(text):
        raise KeyError(f"no AUTO block {name}")
    return pat.sub(lambda m: m.group(1) + body.rstrip("\n") + "\n" + m.group(2), text)


def latex() -> None:
    write_latex()


SRC = "data/research/flb_structural"
GROUP_TEX = {"All": "All groups", "Sports": "Sports", "Weather": "Weather", "Politics": "Politics",
             "Entertainment": "Entertainment", "Mentions": "Mentions", "Science & Tech": r"Science \& Tech"}
GKEY = {"All": "All", "Sports": "Sports", "Weather": "Weather", "Politics": "Politics", "Entertainment": "Ent",
        "Mentions": "Mentions", "Science & Tech": "Sci"}
LKEY = {"1h": "OneH", "6h": "SixH", "24h": "OneD", "3d": "ThreeD", "7d": "SevenD", "30d": "ThirtyD"}


def _num(x: Any, d: int = 2, sign: bool = False, math_: bool = True) -> str:
    r"""A number for LaTeX: '--' when missing; minus as $-$ (tables) or \ensuremath{-} (macros)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:] if not sign else "+" + s[1:]
    neg = s.startswith("-")
    body = s.lstrip("+-")
    if "." not in body and len(body) > 3:
        body = f"{int(body):,}".replace(",", "{,}")
    if neg:
        return ("$-$" if math_ else r"\ensuremath{-}") + body
    if s.startswith("+"):
        return ("$+$" if math_ else "+") + body
    return body


def _int(n: Any) -> str:
    return "--" if n is None else f"{int(n):,}".replace(",", "{,}")


def _pc(x: Any, d: int = 0) -> str:
    return "--" if x is None else _num(100 * x, d)


class Macros:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.names: set[str] = set()

    def add(self, name: str, value: str, source: str) -> None:
        assert re.fullmatch(r"[A-Za-z]+", name), name
        if name in self.names:
            raise ValueError(f"duplicate macro {name}")
        self.names.add(name)
        self.lines.append(rf"\newcommand{{\sd{name}}}{{{value}}} % source: {SRC}/{source}")

    def num(self, name: str, x: Any, source: str, d: int = 2) -> None:
        self.add(name, _num(x, d, math_=False), source)


def _csv(name: str) -> list[dict[str, Any]]:
    def conv(v: str) -> Any:
        if v == "":
            return None
        for f in (int, float):
            try:
                return f(v)
            except ValueError:
                pass
        return v
    with (out_dir() / f"{name}.csv").open() as fh:
        return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _band_get(cells: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    base = {"rho": RHO_MAIN, "weight": "market", "lead": "pooled", "bucket": "LS <=10c", "group": "All"}
    base.update(kw)
    for r in cells:
        if all(r.get(k) == v for k, v in base.items()) and (("structure" in kw) or not r.get("structure")):
            return r
    return {}


def _tc(x: Any, t: Any, d: int = 2) -> str:
    """value (t) cell."""
    if x is None:
        return "--"
    return f"{_num(x, d)} ({_num(t, 1)})" if t is not None else _num(x, d)


def write_latex() -> None:
    S = json.loads((out_dir() / "summary.json").read_text())
    band_cells = _csv("band_cells")
    wfits = _csv("weighting_fits")
    shc = _csv("shin_cells")
    M = Macros()
    bd, wf, sh, be, sy = (S[k] for k in ("band", "weighting", "shin", "beliefs", "synthesis"))
    cov = S["coverage"]
    # coverage
    M.add("Quotes", _int(cov["quotes"]), "summary.json:coverage")
    M.add("Markets", _int(cov["markets"]), "summary.json:coverage")
    M.add("Events", _int(cov["events"]), "summary.json:coverage")
    M.add("LsSides", _int(bd["n_ls_sides"]), "summary.json:band")
    M.add("SplitMonth", wf["split_month"], "summary.json:weighting.split_month")
    # ---- band
    ls = _band_get(band_cells)
    for key, nm in (("gross_c", "LsGross"), ("band_c", "LsBand"), ("resid_c", "LsResid"), ("resid_mk_c", "LsResidMk"),
                    ("floor_c", "LsFloor"), ("spread_c", "LsSpread"), ("fee_c", "LsFee"), ("cap_c", "LsCap"),
                    ("q_mid_c", "LsPrice"), ("win_c", "LsWin")):
        M.num(nm, ls.get(key), "band_cells.csv:All.pooled.LS")
    for key, nm in (("gross_t", "LsGrossT"), ("resid_t", "LsResidT"), ("resid_mk_t", "LsResidMkT")):
        M.num(nm, ls.get(key), "band_cells.csv:All.pooled.LS", 1)
    M.add("LsShare", _pc(ls.get("share_in_band")), "band_cells.csv:All.pooled.LS")
    M.add("LsFloorShare", _pc((ls.get("floor_c") or 0) / ls["gross_c"] if ls.get("gross_c") else None),
          "band_cells.csv:All.pooled.LS")
    M.add("LsRel", _pc(ls.get("gross_rel")), "band_cells.csv:All.pooled.LS")
    lsv = _band_get(band_cells, weight="volume")
    M.num("LsVolGross", lsv.get("gross_c"), "band_cells.csv:All.pooled.LS.volume")
    M.num("LsVolBand", lsv.get("band_c"), "band_cells.csv:All.pooled.LS.volume")
    M.num("LsVolResid", lsv.get("resid_c"), "band_cells.csv:All.pooled.LS.volume")
    M.num("LsVolResidT", lsv.get("resid_t"), "band_cells.csv:All.pooled.LS.volume", 1)
    for rho, nm in ((0.0, "RhoZero"), (0.10, "RhoTen")):
        r = _band_get(band_cells, rho=rho)
        M.num(f"{nm}Resid", r.get("resid_c"), f"band_cells.csv:rho={rho}")
        M.num(f"{nm}ResidT", r.get("resid_t"), f"band_cells.csv:rho={rho}", 1)
        M.num(f"{nm}Cap", r.get("cap_c"), f"band_cells.csv:rho={rho}")
    for b, nm in (("1-2c", "BOne"), ("3-5c", "BThree"), ("6-10c", "BSix")):
        r = _band_get(band_cells, bucket=b)
        M.num(f"{nm}Gross", r.get("gross_c"), f"band_cells.csv:All.pooled.{b}")
        M.num(f"{nm}Band", r.get("band_c"), f"band_cells.csv:All.pooled.{b}")
        M.num(f"{nm}Resid", r.get("resid_c"), f"band_cells.csv:All.pooled.{b}")
        M.num(f"{nm}ResidT", r.get("resid_t"), f"band_cells.csv:All.pooled.{b}", 1)
        M.num(f"{nm}Floor", r.get("floor_c"), f"band_cells.csv:All.pooled.{b}")
    r1 = _band_get(band_cells, bucket="1-2c")
    M.add("BOneFloorShare", _pc((r1.get("floor_c") or 0) / r1["gross_c"] if r1.get("gross_c") else None),
          "band_cells.csv:All.pooled.1-2c")
    for ld, nm in LKEY.items():
        r = _band_get(band_cells, lead=ld)
        M.num(f"Lead{nm}Gross", r.get("gross_c"), f"band_cells.csv:All.{ld}.LS")
        M.num(f"Lead{nm}Band", r.get("band_c"), f"band_cells.csv:All.{ld}.LS")
        M.num(f"Lead{nm}Cap", r.get("cap_c"), f"band_cells.csv:All.{ld}.LS")
        M.num(f"Lead{nm}Price", r.get("q_mid_c"), f"band_cells.csv:All.{ld}.LS")
        M.num(f"Lead{nm}Resid", r.get("resid_c"), f"band_cells.csv:All.{ld}.LS")
        M.num(f"Lead{nm}ResidT", r.get("resid_t"), f"band_cells.csv:All.{ld}.LS", 1)
    inside = 0
    for g in MAIN_GROUPS:
        r = _band_get(band_cells, group=g)
        k = GKEY[g]
        M.num(f"{k}Gross", r.get("gross_c"), f"band_cells.csv:{g}.pooled.LS")
        M.num(f"{k}Resid", r.get("resid_c"), f"band_cells.csv:{g}.pooled.LS")
        M.num(f"{k}ResidT", r.get("resid_t"), f"band_cells.csv:{g}.pooled.LS", 1)
        M.add(f"{k}Share", _pc(r.get("share_in_band")), f"band_cells.csv:{g}.pooled.LS")
        inside += int((r.get("resid_t") or 0) < 1.96)
    M.add("GroupsInside", str(inside), "band_cells.csv:<group>.pooled.LS")
    M.add("NGroups", str(len(MAIN_GROUPS)), "band_cells.csv")
    hz = bd["horizon_slope"]
    for k, nm in (("gross", "HzGross"), ("resid", "HzResid"), ("c_cap", "HzCap")):
        M.num(nm, hz[k].get("slope"), "summary.json:band.horizon_slope", 2 if k != "c_cap" else 3)
        M.num(f"{nm}T", hz[k].get("t"), "summary.json:band.horizon_slope", 1)
    rob = bd["robustness"]
    for k, nm in (("first_half", "First"), ("second_half", "Second"), ("drop_top5_events", "DropTop"), ("with_bid", "Bid"),
                  ("spread_le_5c", "Tight"), ("quote_age_le_24h", "Fresh"), ("pinned", "Pinned"), ("not_pinned", "NotPinned")):
        r = rob.get(k) or {}
        M.num(f"Rob{nm}Resid", r.get("resid_c"), f"summary.json:band.robustness.{k}")
        M.num(f"Rob{nm}ResidT", r.get("resid_t"), f"summary.json:band.robustness.{k}", 1)
        M.num(f"Rob{nm}Gross", r.get("gross_c"), f"summary.json:band.robustness.{k}")
    tg = {(r["grid"], r["bucket"]): r for r in bd.get("tick_grid", [])}
    for (gr, b), nm in ((("1c grid", "1-2c"), "OneCentB"), (("0.1c grid", "1-2c"), "SubCentB"),
                        (("1c grid", "LS <=10c"), "OneCentLs"), (("0.1c grid", "LS <=10c"), "SubCentLs")):
        r = tg.get((gr, b), {})
        M.num(f"{nm}Gross", r.get("gross_c"), "band_tick_grid.csv")
        M.num(f"{nm}Floor", r.get("floor_c"), "band_tick_grid.csv")
        M.num(f"{nm}Resid", r.get("resid_c"), "band_tick_grid.csv")
        M.num(f"{nm}ResidT", r.get("resid_t"), "band_tick_grid.csv", 1)
        M.add(f"{nm}N", _int(r.get("n")), "band_tick_grid.csv")
        M.add(f"{nm}Clusters", _int(r.get("clusters")), "band_tick_grid.csv")
    # ---- weighting
    wc = {(r["dim"], r["cell"], r["weight"]): r for r in wfits}
    for g in ("All", *MAIN_GROUPS):
        r = wc.get(("all" if g == "All" else "group", g, "market"), {})
        k = GKEY[g]
        M.num(f"{k}Gamma", r.get("llo_gamma"), f"weighting_fits.csv:{g}")
        M.num(f"{k}GammaSe", r.get("llo_se_gamma"), f"weighting_fits.csv:{g}", 3)
        M.num(f"{k}GammaT", -(r.get("llo_t_gamma1") or 0), f"weighting_fits.csv:{g}", 1)
        M.num(f"{k}Delta", r.get("llo_delta"), f"weighting_fits.csv:{g}")
        M.num(f"{k}PrGamma", r.get("prelec_gamma"), f"weighting_fits.csv:{g}")
        M.num(f"{k}PrDelta", r.get("prelec_delta"), f"weighting_fits.csv:{g}")
    rv = wc.get(("all", "All", "volume"), {})
    M.num("AllVolGamma", rv.get("llo_gamma"), "weighting_fits.csv:All.volume")
    M.num("AllVolGammaSe", rv.get("llo_se_gamma"), "weighting_fits.csv:All.volume", 3)
    gams = [wc[("group", g, "market")]["llo_gamma"] for g in MAIN_GROUPS if ("group", g, "market") in wc]
    M.num("GammaMin", min(gams), "weighting_fits.csv:<group>")
    M.num("GammaMax", max(gams), "weighting_fits.csv:<group>")
    M.add("GammaBelowOne", str(sum(1 for x in gams if x < 1)), "weighting_fits.csv:<group>")
    lg = [wc[("lead", lab, "market")]["llo_gamma"] for lab in LKEY if ("lead", lab, "market") in wc]
    M.num("LeadGammaMin", min(lg), "weighting_fits.csv:<lead>")
    M.num("LeadGammaMax", max(lg), "weighting_fits.csv:<lead>")
    for k, nm in (("group_slope", "WaldGroup"), ("group_both", "WaldGroupBoth"), ("structure_slope", "WaldStruct"),
                  ("lead_slope", "WaldLead"), ("group_slope_vol", "WaldGroupVol")):
        r = wf["wald"][k]
        M.num(nm, r.get("wald"), f"summary.json:weighting.wald.{k}", 1)
        M.add(f"{nm}Df", str(r.get("df")), f"summary.json:weighting.wald.{k}")
        M.num(f"{nm}P", r.get("p"), f"summary.json:weighting.wald.{k}", 3)
    for d, nm in (("first_to_second", "Fs"), ("second_to_first", "Sf")):
        mm = wf["oos"][d]["models"]
        for m, mn in (("shift", "Shift"), ("llo", "Llo"), ("prelec", "Prelec"), ("llo_by_group", "LloGroup")):
            M.num(f"Oos{nm}{mn}", 1000 * (mm[m]["diff_vs_cal"] or 0), f"summary.json:weighting.oos.{d}", 2)
            M.num(f"Oos{nm}{mn}T", mm[m]["diff_t"], f"summary.json:weighting.oos.{d}", 1)
        M.num(f"Oos{nm}GroupVsPooled", 1000 * (mm["llo_by_group"]["diff_vs_llo"] or 0), f"summary.json:weighting.oos.{d}", 2)
        M.num(f"Oos{nm}GroupVsPooledT", mm["llo_by_group"]["diff_vs_llo_t"], f"summary.json:weighting.oos.{d}", 1)
        M.num(f"Oos{nm}PrelecVsLloT", mm["prelec"]["diff_vs_llo_t"], f"summary.json:weighting.oos.{d}", 1)
        M.num(f"Oos{nm}CalLs", -mm["calibrated"]["mean_ls"], f"summary.json:weighting.oos.{d}", 3)
    for h, nm in (("first", "HalfFirst"), ("second", "HalfSecond")):
        r = wf["halves"][h]
        M.num(f"{nm}Gamma", r.get("gamma"), f"summary.json:weighting.halves.{h}")
        M.num(f"{nm}GammaT", -(r.get("t_gamma1") or 0), f"summary.json:weighting.halves.{h}", 1)
    # ---- Shin
    cnt = sh["counts"]
    M.add("ShinEventLeads", _int(cnt["used_event_leads"]), "summary.json:shin.counts")
    M.add("ShinUnits", _int(cnt["used_units"]), "summary.json:shin.counts")
    M.add("ShinAllEventLeads", _int(cnt["event_leads_all"]), "summary.json:shin.counts")
    M.add("ShinIncomplete", _int(cnt["event_leads_all"] - cnt["complete"]), "summary.json:shin.counts")
    M.add("ShinNotOneYes", _int(cnt["complete"] - cnt["complete_one_yes"]), "summary.json:shin.counts")
    M.add("ShinAskBelow", _int(cnt["sum_ask_below_1"]), "summary.json:shin.counts")
    M.add("ShinMidBelow", _int(cnt["sum_mid_below_1"]), "summary.json:shin.counts")
    scd = {(r["dim"], r["cell"]): r for r in shc}
    for key, nm in ((("all", "All"), "All"), (("kind", "two-leg"), "Two"), (("kind", "multi"), "Multi"),
                    (("kind", "partition ladder"), "Ladder"), *(((("group", g), GKEY[g])) for g in MAIN_GROUPS)):
        r = scd.get(key, {})
        M.num(f"Shin{nm}Zask", r.get("z_ask"), f"shin_cells.csv:{key[1]}", 3)
        M.num(f"Shin{nm}Zmid", r.get("z_mid"), f"shin_cells.csv:{key[1]}", 3)
        M.num(f"Shin{nm}Zml", r.get("fit_z_ml"), f"shin_cells.csv:{key[1]}", 3)
        M.num(f"Shin{nm}ZmlLo", r.get("fit_ci_lo"), f"shin_cells.csv:{key[1]}", 3)
        M.num(f"Shin{nm}ZmlHi", r.get("fit_ci_hi"), f"shin_cells.csv:{key[1]}", 3)
        M.num(f"Shin{nm}OverAsk", r.get("over_ask_pct"), f"shin_cells.csv:{key[1]}", 1)
        M.num(f"Shin{nm}OverMid", r.get("over_mid_pct"), f"shin_cells.csv:{key[1]}", 1)
        M.add(f"Shin{nm}Units", _int(r.get("units")), f"shin_cells.csv:{key[1]}")
    zr = sh["z_regression_info"]
    M.num("ZregSpread", zr["coef"]["half_spread_c"], "summary.json:shin.z_regression_info", 4)
    M.num("ZregSpreadT", zr["t"]["half_spread_c"], "summary.json:shin.z_regression_info", 1)
    M.num("ZregInfo", zr["coef"]["info_group"], "summary.json:shin.z_regression_info", 3)
    M.num("ZregInfoT", zr["t"]["info_group"], "summary.json:shin.z_regression_info", 1)
    M.num("ZregRsq", zr["r2"], "summary.json:shin.z_regression_info", 2)
    zr2 = sh["z_regression_all"]
    M.num("ZregGroupRsq", zr2["r2"], "summary.json:shin.z_regression_all", 2)
    ivp = sh["info_vs_public"]
    M.num("ZmlInfo", ivp["info"].get("z_ml"), "summary.json:shin.info_vs_public", 3)
    M.num("ZmlPublic", ivp["public"].get("z_ml"), "summary.json:shin.info_vs_public", 3)
    M.num("ZmlDiff", ivp.get("diff", {}).get("z_ml"), "summary.json:shin.info_vs_public", 3)
    M.num("ZmlDiffLo", ivp.get("diff", {}).get("ci_lo"), "summary.json:shin.info_vs_public", 3)
    M.num("ZmlDiffHi", ivp.get("diff", {}).get("ci_hi"), "summary.json:shin.info_vs_public", 3)
    M.num("ZaskInfo", ivp["z_ask_info"]["mean"], "summary.json:shin.info_vs_public", 3)
    M.num("ZaskPublic", ivp["z_ask_public"]["mean"], "summary.json:shin.info_vs_public", 3)
    M.num("ZaskDiffRaw", ivp["z_ask_info"]["mean"] - ivp["z_ask_public"]["mean"], "summary.json:shin.info_vs_public", 3)
    M.num("ZTrain", sh["z_fit_train"].get("z_ml"), "summary.json:shin.z_fit_train", 3)
    M.num("ZTrainLo", sh["z_fit_train"].get("ci_lo"), "summary.json:shin.z_fit_train", 3)
    M.num("ZTrainHi", sh["z_fit_train"].get("ci_hi"), "summary.json:shin.z_fit_train", 3)
    M.num("ZTest", sh["z_fit_test"].get("z_ml"), "summary.json:shin.z_fit_test", 3)
    M.num("ZTestLo", sh["z_fit_test"].get("ci_lo"), "summary.json:shin.z_fit_test", 3)
    M.num("ZTestHi", sh["z_fit_test"].get("ci_hi"), "summary.json:shin.z_fit_test", 3)
    lsl = sh["longshot_legs"]
    for smp, nm in (("all", "All"), ("test_half", "Test")):
        r = lsl.get(smp, {})
        for m, mn in (("mid", "Mid"), ("norm_mid", "Norm"), ("shin_ask", "ShinAsk"), ("shin_fit", "ShinFit")):
            M.num(f"Ls{nm}{mn}", (r.get(m) or {}).get("mispricing_c"), f"summary.json:shin.longshot_legs.{smp}")
            M.num(f"Ls{nm}{mn}T", (r.get(m) or {}).get("t"), f"summary.json:shin.longshot_legs.{smp}", 1)
        M.add(f"Ls{nm}Legs", _int(r.get("legs")), f"summary.json:shin.longshot_legs.{smp}")
    scs = sh["scores"]
    for smp, nm in (("all", "All"), ("test_half", "Test")):
        r = scs.get(smp, {})
        for m, mn in (("norm_mid", "Norm"), ("shin_ask", "ShinAsk"), ("shin_mid", "ShinMid"), ("shin_fit", "ShinFit")):
            M.num(f"Sc{nm}{mn}", 1000 * ((r.get(m) or {}).get("diff_vs_norm") or 0), f"shin_scores.csv:{smp}", 2)
            M.num(f"Sc{nm}{mn}T", (r.get(m) or {}).get("diff_vs_norm_t"), f"shin_scores.csv:{smp}", 1)
    lz = sh["ls_on_z"]
    M.num("LsOnZ", lz.get("coef_z"), "summary.json:shin.ls_on_z", 1)
    M.num("LsOnZT", lz.get("t_z"), "summary.json:shin.ls_on_z", 1)
    # ---- beliefs
    for spec, nm in (("pooled_anchor", "Pool"), ("within_group", "Within"), ("pooled_anchor_vol", "Vol"),
                     ("pooled_anchor_split", "Split")):
        for c, v in be[spec]["interactions"].items():
            cn = "".join(w.capitalize() for w in c.split("_"))
            M.num(f"{nm}{cn}", v["coef"], f"beliefs_coef.csv:{spec}", 3)
            M.num(f"{nm}{cn}T", v["t"], f"beliefs_coef.csv:{spec}", 1)
    M.num("PoolBaseSlope", be["pooled_anchor"]["base_slope"]["coef"], "summary.json:beliefs.pooled_anchor", 2)
    for c, v in be["sign_test"].items():
        cn = "".join(w.capitalize() for w in c.split("_"))
        M.add(f"Sign{cn}Pos", str(v["positive"]), "summary.json:beliefs.sign_test")
        M.add(f"Sign{cn}Of", str(v["groups"]), "summary.json:beliefs.sign_test")
        M.add(f"Sign{cn}SigNeg", str(v["sig_neg"]), "summary.json:beliefs.sign_test")
    for h, nm in (("first", "HalfFirst"), ("second", "HalfSecond")):
        for c in ("turnover", "anchored", "log_outcomes"):
            v = be["halves"][h]["interactions"][c]
            cn = "".join(w.capitalize() for w in c.split("_"))
            M.num(f"{nm}{cn}", v["coef"], f"summary.json:beliefs.halves.{h}", 3)
            M.num(f"{nm}{cn}T", v["t"], f"summary.json:beliefs.halves.{h}", 1)
    for spec, nm in (("ls_within_group", "LsW"), ("ls_pooled_anchor", "LsP"), ("ls_within_group_drop_top5", "LsD"),
                     ("ls_pooled_anchor_split", "LsS")):
        for c, v in be[spec]["coef"].items():
            cn = "".join(w.capitalize() for w in c.split("_"))
            M.num(f"{nm}{cn}", v["coef"], f"beliefs_coef.csv:{spec}", 2)
            M.num(f"{nm}{cn}T", v["t"], f"beliefs_coef.csv:{spec}", 1)
    # ---- synthesis
    cf = sy["counterfactual"]
    M.num("CfGross", cf["gross_c"], "summary.json:synthesis.counterfactual")
    M.add("CfN", _int(cf["n"]), "summary.json:synthesis.counterfactual")
    M.add("CfClusters", _int(cf["clusters"]), "summary.json:synthesis.counterfactual")
    for k, nm in (("band", "Band"), ("llo", "Llo"), ("llo_by_group", "LloGroup"), ("prelec", "Prelec")):
        M.add(f"Cf{nm}Share", _pc(cf[k]["share"]), "decomposition.csv")
        if "residual_c" in cf[k]:
            M.num(f"Cf{nm}Resid", cf[k]["residual_c"], "summary.json:synthesis.counterfactual")
            M.num(f"Cf{nm}ResidT", cf[k]["residual_t"], "summary.json:synthesis.counterfactual", 1)
    sp = cf.get("shin_partition", {})
    for k, nm in (("norm_mid", "Norm"), ("shin_ask", "ShinAsk"), ("shin_fit", "ShinFit")):
        M.add(f"Cf{nm}Share", _pc((sp.get(k) or {}).get("share")), "decomposition.csv")
    M.num("CfPartGross", sp.get("gross_c"), "summary.json:synthesis.counterfactual.shin_partition")
    for h, v in sy["criteria"].items():
        M.add(f"Verdict{h[1]}".replace("1", "One").replace("2", "Two").replace("3", "Three").replace("4", "Four"),
              v["verdict"], "summary.json:synthesis.criteria")
        M.add(f"Passed{h[1]}".replace("1", "One").replace("2", "Two").replace("3", "Three").replace("4", "Four"),
              f"{v['passed']} of {v['of']}", "summary.json:synthesis.criteria")
    macros = "% Numbers in the structural sections, generated by `python -m albert.research.flb_structural run latex`.\n" \
             + "\n".join(M.lines)
    files = {"method": PAPER / "method_structural.tex", "results": PAPER / "results_structural.tex",
             "appendix": PAPER / "appendix_structural.tex"}
    txt = {k: p.read_text() for k, p in files.items()}
    txt["method"] = replace_block(txt["method"], "macros", macros)
    blocks = latex_tables(S, band_cells, wfits, shc)
    for name, body in blocks.items():
        for k, t in list(txt.items()):
            if f"% BEGIN AUTO:{name}\n" in t:
                txt[k] = replace_block(t, name, body)
    for k, p in files.items():
        p.write_text(txt[k])
    print(f"latex: {len(M.lines)} macros, {len(blocks)} tables", flush=True)


def _table(label: str, caption: str, source: str, colspec: str, header: list[str], body: list[str], notes: str,
           size: str = r"\scriptsize", sep: str = "3pt") -> str:
    return "\n".join([
        r"\begin{table}[!htbp]", r"\centering", r"\begin{threeparttable}",
        rf"\caption{{{caption}}}\label{{{label}}}", size, rf"\setlength{{\tabcolsep}}{{{sep}}}",
        rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *header, r"\midrule", f"% source: {SRC}/{source}", *body,
        r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\footnotesize", rf"\item {notes}", r"\end{tablenotes}",
        r"\end{threeparttable}", r"\end{table}"])


def _band_row(label: str, r: dict[str, Any], maker: bool = True) -> str:
    if not r:
        return f"{label} & " + " & ".join(["--"] * (11 if maker else 10)) + r" \\"
    cells = [label, _int(r["n"]), _int(r["clusters"]), _num(r["q_mid_c"]), _tc(r["gross_c"], r["gross_t"]),
             _num(r["floor_c"]), _num(r["spread_c"]), _num(r["fee_c"]), _num(r["cap_c"]),
             _tc(r["resid_c"], r["resid_t"])]
    if maker:
        cells.append(_tc(r["resid_mk_c"], r["resid_mk_t"]))
    cells.append(_pc(r.get("share_in_band")) if r.get("share_in_band") is not None else "--")
    return " & ".join(cells) + r" \\"


BAND_HEAD = [r" & & & & Gross & \multicolumn{4}{c}{Band components} & Residual & Maker & In band \\",
             r"\cmidrule(lr){6-9}",
             (r"Cell & Sides & Events & $\bar\q$ & $\bar\q-\bar\won$ & Floor & Half-spr. & Fee & Capital & $\bar R$ & "
             r"$\bar R^{mk}$ & (\%) \\")]


def latex_tables(S: dict[str, Any], band_cells: list[dict[str, Any]], wfits: list[dict[str, Any]],
                 shc: list[dict[str, Any]]) -> dict[str, str]:
    T: dict[str, str] = {}
    bd, wf, sh, be, sy = (S[k] for k in ("band", "weighting", "shin", "beliefs", "synthesis"))
    # ---- band (main)
    body = [_band_row(r"All longshots", _band_get(band_cells)),
            r"\multicolumn{12}{@{}l}{\emph{By lead, all groups}} \\"]
    for ld in LKEY:
        body.append(_band_row(r"\quad " + ld, _band_get(band_cells, lead=ld)))
    body.append(r"\multicolumn{12}{@{}l}{\emph{By group, leads pooled}} \\")
    for g in MAIN_GROUPS:
        body.append(_band_row(r"\quad " + GROUP_TEX[g], _band_get(band_cells, group=g)))
    T["tab-band"] = _table(
        "tab:structural-band", "Longshot mispricing and the no-arbitrage band", "band_cells.csv",
        "@{}lrrrrrrrrrrr@{}", BAND_HEAD, body,
        r"Side observations with $\q\le10\cents$ at the mid, pre-event quotes. All entries in cents per contract. "
        r"Gross: price at the mid minus the realised win rate. The band components (\cref{eq:structural-decomp}) are "
        r"the half-spread of sides pinned at the tick floor (Floor), the half-spread of other sides, the taker fee and "
        r"the capital cost at $\rho=5\%$ a year over the lead. $\bar R$: residual beyond the taker band, the mean net "
        r"profit per contract of selling the side at its bid; $\bar R^{mk}$: selling at its ask as a maker (an upper "
        r"bound: no adverse selection). In band: share of the gross mispricing the taker band covers, "
        r"$\min(\text{band},\text{gross})/\text{gross}$. Market-weighted (each market once per cell; within a lead, "
        r"equal weights). Cells by price bucket and structure are in \cref{tab:structural-band-groups,tab:structural-band-robust}; "
        r"the band per bucket agrees with \cref{tab:xs-limits}. $t$-statistics clustered by event in parentheses.")
    # ---- band by group x bucket (appendix)
    body = []
    for g in ("All", *MAIN_GROUPS):
        body.append(rf"\multicolumn{{12}}{{@{{}}l}}{{\emph{{{GROUP_TEX[g]}}}}} \\")
        for b, lab in (("1-2c", r"1--2\cents"), ("3-5c", r"3--5\cents"), ("6-10c", r"6--10\cents")):
            body.append(_band_row(r"\quad " + lab, _band_get(band_cells, group=g, bucket=b)))
    T["tab-band-groups"] = _table(
        "tab:structural-band-groups", "The band by group and longshot bucket", "band_cells.csv",
        "@{}lrrrrrrrrrrr@{}", BAND_HEAD, body,
        r"As \cref{tab:structural-band}, by group and price bucket, leads pooled, $\rho=5\%$, market-weighted. "
        r"$t$-statistics clustered by event in parentheses.")
    # ---- band robustness (appendix)
    rob = bd["robustness"]
    rows_ = [("Main", _band_get(band_cells)),
             ("$\\rho=0$", _band_get(band_cells, rho=0.0)), ("$\\rho=10\\%$", _band_get(band_cells, rho=0.10)),
             ("Volume-weighted", _band_get(band_cells, weight="volume")),
             (f"First half (to {bd['split_month']})", rob.get("first_half")), ("Second half", rob.get("second_half")),
             ("Drop 5 influential events", rob.get("drop_top5_events")),
             ("Quote age $\\le$ 24 h", rob.get("quote_age_le_24h")),
             ("Spread $\\le$ 5\\cents", rob.get("spread_le_5c")),
             ("Pinned at the floor", rob.get("pinned")), ("Above the floor", rob.get("not_pinned")),
             ("With a bid ($\\q^b>0$)", rob.get("with_bid")),
             *((lab, _band_get(band_cells, structure=st)) for st, lab in
               (("binary", "Binary"), ("multi", "Multi-outcome"), ("ladder", "Ladder leg"), ("deadline", "Deadline")))]
    tgd = {(r["grid"], r["bucket"]): r for r in bd.get("tick_grid", [])}
    rows_ += [("1\\cents{} grid, 1--2\\cents", tgd.get(("1c grid", "1-2c"))),
              ("0.1\\cents{} grid, 1--2\\cents", tgd.get(("0.1c grid", "1-2c"))),
              ("1\\cents{} grid, all \\LS", tgd.get(("1c grid", "LS <=10c"))),
              ("0.1\\cents{} grid, all \\LS", tgd.get(("0.1c grid", "LS <=10c")))]
    body = [_band_row(lab, r or {}) for lab, r in rows_]
    T["tab-band-robust"] = _table(
        "tab:structural-band-robust", "The band: cost of capital, weights, halves and the tick grid",
        "band_cells.csv, summary.json:band.robustness, band_tick_grid.csv", "@{}lrrrrrrrrrrr@{}", BAND_HEAD, body,
        r"Longshot sides ($\q\le10\cents$ at the mid), leads pooled; columns as in \cref{tab:structural-band}. Halves "
        r"split at the month that divides the quotes most evenly. The influential events are the five with the largest "
        r"absolute contribution to the residual. Grid: markets quoted on Kalshi's 0.1\cents{} price grid against the "
        r"standard 1\cents{} grid; the floor is one tick on each grid. $t$-statistics clustered by event in parentheses.",
        sep="2.5pt")
    # ---- weighting (main)
    wc = {(r["dim"], r["cell"], r["weight"]): r for r in wfits}
    ins = {r["group"]: r for r in wf["in_sample"]}
    body = [r"\multicolumn{10}{@{}l}{\emph{Panel A. Fitted weighting functions}} \\"]
    for g in ("All", *MAIN_GROUPS):
        r = wc.get(("all" if g == "All" else "group", g, "market"))
        if not r:
            continue
        i = ins.get(g, {})
        body.append(" & ".join([GROUP_TEX[g], _int(r["n"]), _int(r["clusters"]),
                                f"{_num(r['llo_gamma'])} ({_num(r['llo_se_gamma'], 3)})",
                                f"{_num(r['llo_delta'])} ({_num(r['llo_se_delta'], 3)})", _num(-(r['llo_t_gamma1'] or 0), 1),
                                f"{_num(r['prelec_gamma'])} ({_num(r['prelec_se_gamma'], 3)})",
                                f"{_num(r['prelec_delta'])} ({_num(r['prelec_se_delta'], 3)})",
                                _num(1000 * (i.get("ll_llo", 0) - i.get("ll_calibrated", 0)), 1),
                                _num(1000 * (i.get("ll_prelec", 0) - i.get("ll_calibrated", 0)), 1)]) + r" \\")
    body += [r"\midrule", r"\multicolumn{10}{@{}l}{\emph{Panel B. Equal slopes ($\gamma$) across cells: Wald tests}} \\"]
    for k, lab in (("group_slope", "Groups"), ("group_both", r"Groups, $\gamma$ and $\delta$"),
                   ("structure_slope", "Structures"), ("lead_slope", "Leads"), ("group_slope_vol", "Groups, volume-weighted")):
        r = wf["wald"][k]
        body.append(rf"\multicolumn{{3}}{{@{{}}l}}{{{lab}}} & \multicolumn{{7}}{{l}}{{$\chi^2_{{{r.get('df')}}}={_num(r.get('wald'), 1)}$, "
                    rf"$p={_num(r.get('p'), 3)}$, {_int(r.get('clusters'))} events}} \\")
    body += [r"\midrule", (r"\multicolumn{10}{@{}l}{\emph{Panel C. Out-of-sample log score against calibrated prices "
             r"(milli-nats per quote)}} \\")]
    for m, lab in (("shift", "Constant shift"), ("llo", "LLO, pooled"), ("prelec", "Prelec, pooled"),
                   ("llo_by_group", "LLO, by group")):
        a, b2 = wf["oos"]["first_to_second"]["models"][m], wf["oos"]["second_to_first"]["models"][m]
        body.append(rf"\multicolumn{{3}}{{@{{}}l}}{{{lab}}} & \multicolumn{{7}}{{l}}{{first $\to$ second half: "
                    rf"{_tc(1000 * a['diff_vs_cal'], a['diff_t'])}; second $\to$ first: {_tc(1000 * b2['diff_vs_cal'], b2['diff_t'])}}} \\")
    T["tab-weighting"] = _table(
        "tab:structural-weighting", "Probability weighting: fitted functions, equality tests and forecasts",
        "weighting_fits.csv, weighting_insample.csv, weighting_oos.csv, summary.json:weighting.wald",
        "@{}lrrrrrrrrr@{}",
        [r" & & & \multicolumn{3}{c}{LLO} & \multicolumn{2}{c}{Prelec} & \multicolumn{2}{c}{$\Delta\log L$} \\",
         r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}",
         r"Group & Quotes & Events & $\gamma$ & $\delta$ & $t_{\gamma<1}$ & $\gamma$ & $\delta$ & LLO & Prelec \\"],
        body,
        r"YES sides of pre-event quotes, market-weighted. LLO: $\operatorname{logit}\q=\delta+\gamma\operatorname{logit}\pi$, "
        r"estimated as~\eqref{eq:logit} with $\gamma=1/\beta$ and $\delta=-\alpha/\beta$. Prelec: $\q=\exp(-\delta(-\ln\pi)^\gamma)$. "
        r"Maximum likelihood on outcomes; standard errors in parentheses clustered by event (delta method for $\gamma$, "
        r"$\delta$). $\Delta\log L$: in-sample gain in mean log likelihood over calibrated prices, milli-nats per "
        r"market. Panel B: cluster-robust Wald tests from one fully interacted logit. Panel C: models fitted on one half of "
        r"the months and scored on the other; differences against $\Prob(y=1)=\q$ with event-clustered $t$ in parentheses.",
        sep="3.5pt")
    # ---- weighting cells (appendix)
    body = []
    for dim, lab in (("lead", "By lead"), ("structure", "By structure"), ("group", "By group, volume-weighted")):
        body.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{{lab}}}}} \\")
        wt = "volume" if dim == "group" else "market"
        cells = [r for r in wfits if r["dim"] == dim and r["weight"] == wt]
        if dim == "group":
            cells = [wc[("all", "All", "volume")], *cells] if ("all", "All", "volume") in wc else cells
        for r in cells:
            name = {"binary": "Binary", "multi": "Multi-outcome", "ladder": "Ladder leg", "deadline": "Deadline"}.get(
                str(r["cell"]), GROUP_TEX.get(str(r["cell"]), str(r["cell"])))
            body.append(" & ".join([r"\quad " + name, _int(r["n"]), _int(r["clusters"]), _num(r["llo_alpha"]),
                                    _tc(r["llo_beta"], r["llo_t_beta1"]),
                                    f"{_num(r['llo_gamma'])} ({_num(r['llo_se_gamma'], 3)})",
                                    f"{_num(r['llo_delta'])} ({_num(r['llo_se_delta'], 3)})",
                                    f"{_num(r['prelec_gamma'])} ({_num(r['prelec_se_gamma'], 3)})",
                                    f"{_num(r['prelec_delta'])} ({_num(r['prelec_se_delta'], 3)})"]) + r" \\")
    T["tab-weighting-cells"] = _table(
        "tab:structural-weighting-cells", "Weighting-function parameters by lead and structure", "weighting_fits.csv",
        "@{}lrrrrrrrr@{}",
        [r" & & & \multicolumn{4}{c}{LLO} & \multicolumn{2}{c}{Prelec} \\",
         r"\cmidrule(lr){4-7}\cmidrule(l){8-9}",
         r"Cell & Quotes & Events & $\alpha$ & $\beta$ ($t_{\beta=1}$) & $\gamma$ & $\delta$ & $\gamma$ & $\delta$ \\"],
        body,
        r"As \cref{tab:structural-weighting}. Within a lead each market counts once. Standard errors (or $t$) clustered by "
        r"event in parentheses.")
    # ---- Shin (main)
    scd = {(r["dim"], r["cell"]): r for r in shc}
    def shin_row(lab: str, r: dict[str, Any] | None) -> str:
        if not r:
            return f"{lab} & " + " & ".join(["--"] * 8) + r" \\"
        ci = (f"{_num(r.get('fit_z_ml'), 3)} [{_num(r.get('fit_ci_lo'), 3)}, {_num(r.get('fit_ci_hi'), 3)}]"
              if r.get("fit_z_ml") is not None else "--")
        return " & ".join([lab, _int(r["event_leads"]), _int(r["units"]), _num(r["legs_median"], 0),
                           _tc(r["over_ask_pct"], (r["over_ask_pct"] / r["over_ask_se"]) if r.get("over_ask_se") else None, 1),
                           _tc(r["over_mid_pct"], (r["over_mid_pct"] / r["over_mid_se"]) if r.get("over_mid_se") else None, 1),
                           f"{_num(r['z_ask'], 3)} ({_num(r['z_ask_se'], 4)})", _num(r["z_mid"], 3), ci]) + r" \\"
    body = [shin_row("All partition events", scd.get(("all", "All")))]
    for k, lab in (("two-leg", r"\quad Two-leg events"), ("multi", r"\quad Multi-outcome events"),
                   ("partition ladder", r"\quad Partition ladders")):
        body.append(shin_row(lab, scd.get(("kind", k))))
    for g in MAIN_GROUPS:
        if ("group", g) in scd:
            body.append(shin_row(r"\quad " + GROUP_TEX[g], scd[("group", g)]))
    ivp = sh["info_vs_public"]
    d = ivp.get("diff", {})
    body.append(rf"\quad Information minus public & & & & & & {_num(ivp['z_ask_info']['mean'] - ivp['z_ask_public']['mean'], 3)} & & "
                rf"{_num(d.get('z_ml'), 3)} [{_num(d.get('ci_lo'), 3)}, {_num(d.get('ci_hi'), 3)}] \\")
    T["tab-shin"] = _table(
        "tab:structural-shin", "Shin's insider share in events whose legs partition the outcomes",
        "shin_cells.csv, summary.json:shin.info_vs_public", "@{}lrrrrrrrr@{}",
        [r" & Event- & & Legs & $\sum a-1$ & $\sum p-1$ & $\shinz$ from & $\shinz$ from & $\shinz$ fitted \\",
         r"Sample & leads & Events & (med.) & (\%) & (\%) & asks & mids & to outcomes [95\% CI] \\"],
        body,
        r"Mutually exclusive events (Kalshi flag) and strike ladders that partition the outcome space, at leads where "
        r"every leg has a usable quote and exactly one leg settles YES. $a$, $p$: YES ask and mid. $\shinz$ from "
        r"asks or mids solves~\eqref{eq:shin} ($\shinz=0$ when the prices sum to at most one). Fitted $\shinz$: the "
        r"value on a grid over $[0,0.6]$ that maximises the log score of the winning leg, with the scale chosen so that "
        r"probabilities sum to one; 95\% event-bootstrap interval. Each event weighs one across its leads; "
        r"\cref{tab:xs-multi} weights event-leads equally, hence the small differences in the multi-outcome row. Information "
        r"groups: Politics, Entertainment, Mentions, Science \& Tech; public: Sports, Weather. Standard errors (of "
        r"$\shinz$) or $t$-statistics (of the overrounds) clustered by event in parentheses.", sep="2.8pt")
    lsl, scs = sh["longshot_legs"], sh["scores"]
    against = {"mid": None, "norm_mid": "mid", "norm_ask": None, "shin_ask": "norm_ask", "shin_mid": "norm_mid",
               "shin_fit": "norm_mid"}
    against_tex = {None: "--", "mid": "raw mid", "norm_ask": "normalised asks", "norm_mid": "normalised mids"}
    fbody = []
    for m, lab in (("mid", "Raw mid"), ("norm_mid", "Normalised mids"), ("norm_ask", "Normalised asks"),
                   ("shin_ask", "Shin, from the asks"), ("shin_mid", "Shin, from the mids"),
                   ("shin_fit", r"Shin, $\shinz$ fitted on the first half")):
        cells = [lab]
        for smp in ("all", "test_half"):
            v = (lsl.get(smp) or {}).get(m)
            cells.append(_tc(v["mispricing_c"], v["t"]) if v else "--")
        for smp in ("all", "test_half"):
            v = (scs.get(smp) or {}).get(m) or {}
            ok = against[m] is not None and v.get("diff_vs_norm") is not None
            cells.append(_tc(1000 * v["diff_vs_norm"], v.get("diff_vs_norm_t")) if ok else "--")
        cells.append(against_tex[against[m]])
        fbody.append(" & ".join(cells) + r" \\")
    la, lt = lsl.get("all", {}), lsl.get("test_half", {})
    T["tab-shin-forecasts"] = _table(
        "tab:structural-shin-forecasts", "Partition events: raw, normalised and Shin probabilities",
        "summary.json:shin.longshot_legs, shin_scores.csv", "@{}lrrrrl@{}",
        [r" & \multicolumn{2}{c}{Longshot legs (\cents)} & \multicolumn{3}{c}{Log score gain (milli-nats)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(l){4-6}",
         r"Forecast & All months & Second half & All months & Second half & against \\"],
        fbody,
        rf"Legs of the partition events of \cref{{tab:structural-shin}}. Longshot legs: mid at most 10\cents{{}} "
        rf"({_int(la.get('legs'))} legs in all months, {_int(lt.get('legs'))} in the second half); mean forecast minus "
        r"outcome in cents. Log score: mean gain per leg, over all legs, against the normalisation of the same prices. "
        r"The fitted $\shinz$ is estimated on the first half of the months, so the second half is out of sample. Each "
        r"event weighs one across its leads; $t$-statistics clustered by event in parentheses.",
        size=r"\footnotesize", sep="3.5pt")
    # ---- Shin by lead and group x kind (appendix)
    body = [r"\multicolumn{9}{@{}l}{\emph{By lead}} \\"]
    for ld in LKEY:
        if ("lead", ld) in scd:
            body.append(shin_row(r"\quad " + ld, scd[("lead", ld)]))
    body.append(r"\multicolumn{9}{@{}l}{\emph{By group and structure}} \\")
    for r in shc:
        if r["dim"] == "group_kind":
            g, k = str(r["cell"]).split("|")
            body.append(shin_row(rf"\quad {GROUP_TEX[g]}, {k}", r))
    zr = sh["z_regression_all"]
    names = {"half_spread_c": "half-spread (\\cents)", "log_legs": "log legs"}
    terms = "; ".join(f"{names.get(k, k.replace('group:', '').replace('&', r'\&'))} {_num(v, 4)} ({_num(zr['t'][k], 1)})"
                      for k, v in zr["coef"].items() if k != "const")
    T["tab-shin-full"] = _table(
        "tab:structural-shin-full", "Shin's insider share by lead, group and structure", "shin_cells.csv",
        "@{}lrrrrrrrr@{}",
        [r" & Event- & & Legs & $\sum a-1$ & $\sum p-1$ & $\shinz$ from & $\shinz$ from & $\shinz$ fitted \\",
         r"Sample & leads & Events & (med.) & (\%) & (\%) & asks & mids & to outcomes [95\% CI] \\"],
        body,
        rf"As \cref{{tab:structural-shin}}. Regression of $\shinz$ from the asks on the mean half-spread, log legs and group "
        rf"dummies (Sports omitted), weighted, event-clustered $t$ in parentheses: {terms}; $R^2={_num(zr['r2'], 2)}$, "
        rf"{_int(zr['n'])} event-leads.", sep="2.8pt")
    # ---- beliefs (main)
    lab_c = {"log_outcomes": "Number of outcomes (log)", "turnover": "Turnover", "spread_c": "Quoted spread",
             "log_lead": "Horizon (log lead)", "anchored": "Forecast anchor (0/1)", "anchor_sports": "Anchor: Sports (0/1)",
             "anchor_weather": "Anchor: Weather (0/1)"}
    specs = [("pooled_anchor", "interactions"), ("within_group", "interactions"), ("pooled_anchor_vol", "interactions"),
             ("pooled_anchor_split", "interactions"), ("ls_within_group", "coef"), ("ls_pooled_anchor_split", "coef")]
    body = []
    for c in ("log_outcomes", "turnover", "spread_c", "log_lead", "anchored", "anchor_sports", "anchor_weather"):
        cells = [lab_c[c]]
        for sp, key in specs:
            v = be[sp].get(key, {}).get(c)
            cells.append(_tc(v["coef"], v["t"], 3 if key == "interactions" else 2) if v else "")
        body.append(" & ".join(cells) + r" \\")
    body.append(r"\midrule")
    body.append(" & ".join(["Events", *(_int(be[sp]["clusters"]) for sp, _ in specs)]) + r" \\")
    body.append(" & ".join(["Observations", *(_int(be[sp]["n"]) for sp, _ in specs)]) + r" \\")
    T["tab-beliefs"] = _table(
        "tab:structural-beliefs", "Where is the bias steeper? Calibration-slope interactions", "beliefs_coef.csv",
        "@{}lrrrrrr@{}",
        [r" & \multicolumn{4}{c}{Slope interaction (logit)} & \multicolumn{2}{c}{Longshot overpricing (\cents)} \\",
         r"\cmidrule(lr){2-5}\cmidrule(l){6-7}",
         r" & Pooled & Within group & Volume-wtd. & Split anchor & Within group & Split anchor \\"],
        body,
        r"Columns 1--4: coefficients $b_k$ of~\eqref{eq:structural-interact}, the change in the calibration slope per "
        r"standard deviation of each characteristic (per unit for 0/1 anchors), YES sides of pre-event quotes; "
        r"$b_k>0$ means a stronger bias. Pooled: one slope plus interactions, group intercepts; within group: "
        r"group-specific slopes, so the group-level anchor is absorbed. Columns 5--6: least squares of the price minus "
        r"outcome of longshot sides (cents per contract) on the standardised characteristics with price-bucket (and in "
        r"column 5 group) fixed effects. Market-weighted unless stated; $t$-statistics clustered by event in parentheses.",
        sep="3.5pt")
    # ---- beliefs by group (appendix)
    body = []
    for g, r in be["by_group"].items():
        it = r.get("interactions", {})
        body.append(" & ".join([GROUP_TEX[g], *(_tc(it[c]["coef"], it[c]["t"], 3) if c in it else "--"
                                               for c in ("log_outcomes", "turnover", "spread_c", "log_lead")),
                                _int(r["n"]), _int(r["clusters"])]) + r" \\")
    body.append(r"\midrule")
    for h, lab in (("first", f"Pooled, first half (to {be['split_month']})"), ("second", "Pooled, second half")):
        it = be["halves"][h]["interactions"]
        body.append(" & ".join([lab, *(_tc(it[c]["coef"], it[c]["t"], 3) for c in ("log_outcomes", "turnover", "spread_c",
                                                                                   "log_lead")),
                                _int(be["halves"][h]["n"]), _int(be["halves"][h]["clusters"])]) + r" \\")
        body.append(rf"\quad anchor & \multicolumn{{4}}{{l}}{{{_tc(it['anchored']['coef'], it['anchored']['t'], 3)}}} & & \\")
    st = be["sign_test"]
    T["tab-beliefs-groups"] = _table(
        "tab:structural-beliefs-groups", "Calibration-slope interactions by group and by half", "beliefs_coef.csv, summary.json:beliefs",
        "@{}lrrrrrr@{}",
        [r"Sample & Outcomes & Turnover & Spread & Horizon & Quotes & Events \\"], body,
        r"As \cref{tab:structural-beliefs}, column 1, estimated separately by group (a characteristic enters when at least "
        r"ten events lie on each side of its group median) and by half of the sample. Positive interactions across groups: "
        + ", ".join(f"{lab_c[c].split(' (')[0].lower()} {st[c]['positive']} of {st[c]['groups']}" for c in st)
        + r". $t$-statistics clustered by event in parentheses.", sep="3.5pt")
    # ---- scorecard (main)
    body = []
    for h, v in sy["criteria"].items():
        body.append(rf"\multicolumn{{3}}{{@{{}}l}}{{\emph{{{h}: {v['verdict']} ({v['passed']} of {v['of']} predictions)}}}} \\")
        for t in v["tests"]:
            stat = t.get("stat")
            if isinstance(stat, list):
                st_ = ", ".join(_num(1000 * x, 2) for x in stat)
                tt = ", ".join(_num(x, 1) for x in t.get("t", []))
                val = f"{st_} ({tt})"
            elif "ci" in t:
                val = f"{_num(stat, 3)} [{_num(t['ci'][0], 3)}, {_num(t['ci'][1], 3)}]"
            elif "of" in t:
                val = f"{_int(stat)} of {_int(t['of'])}"
            elif "p" in t:
                val = f"{_num(stat, 1)} ($p={_num(t['p'], 3)}$)"
            else:
                val = _tc(stat, t.get("t"), 1 if "per unit" in CRITERIA_TEX.get(t["test"], "") else
                          (3 if "z from" in t["test"] or "slope" in t["test"] else 2))
            mark = "--" if t["pass"] is None else ("yes" if t["pass"] else "no")
            body.append(rf"\quad {CRITERIA_TEX.get(t['test'], _tex_escape(t['test']))} & {val} & {mark} \\")
    T["tab-scorecard"] = _table(
        "tab:structural-scorecard", "The four explanations scored against their predictions",
        "criteria.csv, summary.json:synthesis.criteria", r"@{}p{0.6\textwidth}rr@{}",
        [r"Prediction & Statistic & Holds \\"], body,
        r"Each row is a prediction of Hypotheses~\ref{hyp:mech}--\ref{hyp:beliefs} as made operational in "
        r"\cref{sec:method-structural}, tested at the 5\% level; the every-group band test uses a one-sided Bonferroni "
        r"threshold across groups. Statistics: "
        r"cents per contract with $t$ (H1 band and horizon rows; the floor row is a share), $\gamma$ with $t_{\gamma<1}$, "
        r"$\chi^2$ with $p$, out-of-sample log-score differences in milli-nats per quote for the two directions with $t$, "
        r"cents per contract, $\shinz$ differences with 95\% bootstrap intervals, and slope interactions with $t$. All "
        r"$t$-statistics and intervals clustered by event. Verdict: supported when every prediction holds, partly "
        r"supported when some do.".replace("H1", r"\cref{hyp:mech}"), size=r"\footnotesize", sep="4pt")
    # ---- scorecard by group
    rows = [{k: ({"+": "$+$", "-": "$-$", "0": "$0$"}.get(v, v) if k.endswith("_mark") else v) for k, v in r.items()}
            for r in sy["scorecard"]]
    body = []
    for r in rows:
        g = r["group"]
        h3 = (f"{r.get('h3_mark', '')} {_num(r.get('h3_z_ml'), 3)} [{_num(r.get('h3_z_ml_lo'), 2)}, {_num(r.get('h3_z_ml_hi'), 2)}]"
              if r.get("h3_z_ml") is not None else "--")
        body.append(" & ".join([
            GROUP_TEX[g],
            f"{r.get('h1_mark', '')} {_tc(r.get('h1_resid_c'), r.get('h1_resid_t'))}" if r.get("h1_resid_c") is not None else "--",
            _pc(r.get("h1_share")),
            f"{r.get('h2_mark', '')} {_num(r.get('h2_gamma'))} ({_num(r.get('h2_t_gamma_lt1'), 1)})" if r.get("h2_gamma") is not None else "--",
            h3,
            f"{r.get('h4_mark', '')} {_tc(r.get('h4_turnover_coef'), r.get('h4_turnover_t'), 3)}" if r.get("h4_turnover_coef") is not None else "--"])
            + r" \\")
    T["tab-scorecard-groups"] = _table(
        "tab:structural-scorecard-groups", "The evidence by group", "scorecard.csv", "@{}lrrrrr@{}",
        [r" & \multicolumn{2}{c}{\Cref{hyp:mech}: band} & \Cref{hyp:pref}: weighting & \Cref{hyp:info}: insiders & \Cref{hyp:beliefs}: dispersion \\",
         r"\cmidrule(lr){2-3}",
         r"Group & Residual $\bar R$ (\cents) & In band (\%) & $\gamma$ ($t_{\gamma<1}$) & fitted $\shinz$ [95\% CI] & Turnover \\"],
        body,
        r"Mark: $+$ consistent with the hypothesis's prediction for the group, $-$ inconsistent, $0$ inconclusive. "
        r"\Cref{hyp:mech}: $+$ when the longshot residual beyond the taker band is not significantly positive. "
        r"\Cref{hyp:pref}: $+$ when $\gamma<1$ significantly. \Cref{hyp:info}: in Politics, Entertainment, Mentions and "
        r"Science \& Tech $+$ when the interval for the fitted $\shinz$ excludes zero; in Sports and Weather, where "
        r"outcomes are public, $+$ when it includes zero and $-$ when it does not. \Cref{hyp:beliefs}: sign and "
        r"significance of the turnover interaction (within-group slopes for all groups). Sources and clustering as in "
        r"\cref{tab:structural-band,tab:structural-weighting,tab:structural-shin,tab:structural-beliefs}.", sep="3.5pt")
    # ---- counterfactual
    cf = sy["counterfactual"]
    body = [(rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{Longshot sides, second half ({_int(cf['n'])} sides, {_int(cf['clusters'])} "
            rf"events); gross mispricing {_tc(cf['gross_c'], cf['gross_c'] / cf['gross_se'] if cf.get('gross_se') else None)} \cents}}}} \\")]
    body.append(rf"\quad Taker band, $\rho=5\%$ (band {_num(cf['band']['band_c'])}\cents) & {_num(cf['band']['removed_c'])} & -- & "
                rf"{_pc(cf['band']['share'])} \\")
    for k, lab in (("llo", "LLO, pooled, fitted on the first half"), ("llo_by_group", "LLO by group, fitted on the first half"),
                   ("prelec", "Prelec, pooled, fitted on the first half")):
        body.append(rf"\quad {lab} & {_num(cf[k]['removed_c'])} & {_tc(cf[k]['residual_c'], cf[k]['residual_t'])} & {_pc(cf[k]['share'])} \\")
    sp = cf.get("shin_partition")
    if sp:
        body.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{Longshot legs of partition events, second half ({_int(sp['legs'])} legs, "
                    rf"{_int(sp['units'])} events); gross mispricing {_num(sp['gross_c'])} \cents}}}} \\")
        for k, lab in (("norm_mid", "Normalised mids"), ("shin_ask", "Shin from the asks"),
                       ("shin_fit", "Shin with $\\shinz$ fitted on the first half")):
            body.append(rf"\quad {lab} & {_num(sp['gross_c'] - sp[k]['residual_c'])} & {_tc(sp[k]['residual_c'], sp[k]['residual_t'])} & {_pc(sp[k]['share'])} \\")
    T["tab-decomp"] = _table(
        "tab:structural-decomp", "How much of the longshot loss does each explanation map remove?",
        "decomposition.csv, summary.json:synthesis.counterfactual", "@{}lrrr@{}",
        [r"Map from prices to probabilities & Removed (\cents) & Residual (\cents) & Share removed (\%) \\"], body,
        r"Gross mispricing: price at the mid minus outcome, cents per contract. Band: the share of the gross mispricing "
        r"that lies inside the taker band (\cref{tab:structural-band}). Each other row replaces the price by a fitted "
        r"probability and reports the remaining price-minus-outcome gap (residual) and the share removed. The maps are "
        r"alternative readings of the same calibration curve, not additive components. Market-weighted (partition "
        r"events: each event once); $t$-statistics clustered by event in parentheses.", size=r"\footnotesize", sep="4pt")
    return T


def _tex_escape(s: str) -> str:
    return (s.replace("&", r"\&").replace("%", r"\%").replace(">=", r"$\ge$").replace("<=", r"$\le$")
            .replace(" < ", r" $<$ ").replace(" > ", r" $>$ ").replace("t $<$ -1.96", r"$t<-1.96$")
            .replace("'s z", r"'s $\shinz$").replace("gamma", r"$\gamma$").replace("z from", r"$\shinz$ from")
            .replace("fitted z", r"fitted $\shinz$"))



STEPS: dict[str, Callable[[], Any]] = {"band": band, "weighting": weighting, "shin": shin, "beliefs": beliefs,
                                       "synthesis": synthesis, "summary": summarize, "all": run_all, "latex": latex}


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m albert.research.flb_structural")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("step", choices=sorted(STEPS))
    a = ap.parse_args(list(argv) if argv is not None else None)
    STEPS[a.step]()


if __name__ == "__main__":
    main()
