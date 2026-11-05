"""Self-exciting intensity (Hawkes) and flow toxicity as a volatility-aware quote filter.

Four parts (report ``research-hawkes``):

1. **Hawkes fits.** Exponential-kernel Hawkes processes fitted by maximum likelihood with the
   O(n) recursive log-likelihood (``_excite`` computes ``A_i = sum_{j<i} exp(-beta (t_i - t_j))``
   with a cumulative log-sum-exp, so there is no Python loop over events):
   ``lambda(t) = mu + alpha * sum_{t_j < t} exp(-beta (t - t_j))``, branching ratio
   ``n = alpha / beta``, decay time ``1 / beta``. Univariate and bivariate (taker buys YES /
   taker sells YES, shared ``beta``) on trade arrivals, and univariate on mid-change events.
   Goodness of fit: time-rescaling residuals (compensator increments) against Exp(1), KS test,
   next to the same test for a homogeneous Poisson fit. A taker order that fills several resting
   orders prints several trades with one timestamp; those are merged into one arrival.
2. **Toxicity at the maker's fill.** Hawkes intensity (absolute and relative to the market's own
   trailing one-hour arrival rate), mid-change intensity, VPIN (equal-volume buckets, taker side
   known), signed taker-flow imbalance over 10 / 60 s (signed against the maker: positive = more
   flow in the direction of the taker who hit us), short-horizon realised volatility, all measured
   from the tape strictly before the fill (exchange timestamps ``< ts_ms`` of the event that filled
   us, so the sweep that fills us is never in its own feature). Regressed against maker markouts
   (``mo`` to the mid, ``mx`` to the exit price = best bid on our side) at 10 / 60 / 300 s, for
   every recorded execution and for the baseline Quoter's simulated fills, with standard errors
   clustered by event.
3. **Quote filters on the Simulator** (``ToxQuoter``, the baseline ``Quoter`` plus one filter):
   pull while a live measure (computed online in receive time, ``OnlineTox``) exceeds a threshold
   chosen on dev; volatility-scaled minimum spread; or quote one tick wider while intensity is
   high. Thresholds are dev quantiles of the measure at the baseline's dev fills.
4. **Historical trades** (``kalshi_trades``): Hawkes fits on game markets, and whether intensity
   bursts predict the direction or only the magnitude of the next print-price move, with a taker
   proxy (enter at the next same-direction taker print, exit at the next opposite one, taker fees)
   and the makers' print markouts against prior intensity. The table holds mostly *pre-game*
   windows (``ingest_kalshi_trade_windows``: anchor - 25 h to anchor - 4 h); only a few dozen game
   markets have trades through the game, and the in-play sample is analysed separately.

No look-ahead: features use data before the decision; thresholds and Hawkes parameters used
online are fitted on dev only; evaluation splits by the fill's own 5-minute volatile flag are
labelled as such.
"""

import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

from albert.backtest.l2 import (EPS, Event, Quoter, SimConfig, Simulator, iter_events,
                                stream_files)

NS = 10**9
TICK = 0.01
NMAX = 0.999                      # branching-ratio bound (stationary Hawkes)
BETA_BOUNDS = (1e-4, 1e3)         # decay time 1 ms .. 2.8 h
HORIZONS = (10, 60, 300)
MEASURES = ("lam", "lam_rel", "lam_mid", "vpin", "ofi_adv10", "ofi_adv60", "abs_ofi60", "rv60",
            "n60")
# heavy-tailed non-negative measures are regressed on log1p(x); signed ones on sign*log1p|x|
_LOG = {"lam", "lam_rel", "lam_mid", "abs_ofi60", "rv60", "n60"}
_SLOG = {"ofi_adv10", "ofi_adv60"}


# ---- Hawkes likelihood ---------------------------------------------------------------------------
def _excite(t: np.ndarray, beta: float, src: np.ndarray | None = None) -> np.ndarray:
    """``A_i = sum_{j < i, src_j} exp(-beta (t_i - t_j))`` for sorted ``t`` (index order, so
    ties count as earlier). O(n) via a cumulative log-sum-exp of ``beta t``."""
    x = beta * t
    if src is not None:
        x = np.where(src, x, -np.inf)
    out = np.zeros(len(t))
    if len(t) > 1:
        lse = np.logaddexp.accumulate(x)
        out[1:] = np.exp(lse[:-1] - beta * t[1:])
    return out


def _excite_ab(t: np.ndarray, beta: float, src: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``A_i`` as in ``_excite`` and ``B_i = sum_{j<i} (t_i - t_j) exp(-beta (t_i - t_j))``
    (= -dA_i/dbeta), via ``B_i = t_i A_i - C_i`` with ``C_i = sum_{j<i} t_j exp(...)``."""
    a = _excite(t, beta, src)
    x = beta * t + np.log(np.maximum(t, 1e-300))
    if src is not None:
        x = np.where(src, x, -np.inf)
    c = np.zeros(len(t))
    if len(t) > 1:
        lse = np.logaddexp.accumulate(x)
        c[1:] = np.exp(lse[:-1] - beta * t[1:])
    return a, np.maximum(t * a - c, 0.0)


def _nll1_grad(x: np.ndarray, seqs: list[tuple[np.ndarray, float]]) -> tuple[float, np.ndarray]:
    """Negative log-likelihood and its gradient in (log mu, log n, log beta)."""
    mu, n, beta = np.exp(x)
    ll, g_mu, g_n, g_b = 0.0, 0.0, 0.0, 0.0
    for t, T in seqs:
        a, b = _excite_ab(t, beta)
        lam = mu + n * beta * a
        e = np.exp(-beta * (T - t))
        ll += float(np.log(lam).sum()) - mu * T - n * float((1 - e).sum())
        g_mu += float((1 / lam).sum()) - T
        g_n += float((beta * a / lam).sum()) - float((1 - e).sum())
        g_b += float((n * (a - beta * b) / lam).sum()) - n * float(((T - t) * e).sum())
    return -ll, -np.array([g_mu * mu, g_n * n, g_b * beta])


def _nll2_grad(x: np.ndarray, seqs: list[tuple[np.ndarray, np.ndarray, float]]) -> tuple[float, np.ndarray]:
    """Bivariate negative log-likelihood and gradient in (log mu_0, log mu_1, log n_00, log n_01,
    log n_10, log n_11, log beta)."""
    e = np.exp(x)
    mu, nm, beta = e[0:2], e[2:6].reshape(2, 2), float(e[6])
    ll = 0.0
    g_mu, g_n, g_b = np.zeros(2), np.zeros((2, 2)), 0.0
    for t, m, T in seqs:
        ab = [_excite_ab(t, beta, m == j) for j in (0, 1)]
        lam = mu[m] + beta * (nm[m, 0] * ab[0][0] + nm[m, 1] * ab[1][0])
        inv = 1 / lam
        ee = np.exp(-beta * (T - t))
        tail = 1 - ee
        ll += float(np.log(lam).sum()) - float(mu.sum()) * T
        for j in (0, 1):
            sj = float(tail[m == j].sum())
            ll -= float(nm[:, j].sum()) * sj
            for k in (0, 1):
                g_n[k, j] += float((beta * ab[j][0][m == k] * inv[m == k]).sum()) - sj
            g_b -= float(nm[:, j].sum()) * float(((T - t) * ee)[m == j].sum())
            g_b += float((nm[m, j] * (ab[j][0] - beta * ab[j][1]) * inv).sum())
        for k in (0, 1):
            g_mu[k] += float(inv[m == k].sum()) - T
    grad = np.concatenate([g_mu * mu, (g_n * nm).ravel(), [g_b * beta]])
    return -ll, -grad


def loglik1(mu: float, n: float, beta: float, seqs: list[tuple[np.ndarray, float]]) -> float:
    """Univariate log-likelihood; ``seqs`` = [(event times from the window start, window length)]."""
    alpha = n * beta
    ll = 0.0
    for t, T in seqs:
        lam = mu + alpha * _excite(t, beta)
        ll += float(np.log(lam).sum()) - mu * T - n * float((1 - np.exp(-beta * (T - t))).sum())
    return ll


def loglik1_bruteforce(mu: float, n: float, beta: float, t: np.ndarray, T: float) -> float:
    """O(n^2) reference implementation (tests)."""
    alpha = n * beta
    ll = 0.0
    for i in range(len(t)):
        ll += math.log(mu + alpha * sum(math.exp(-beta * (t[i] - t[j])) for j in range(i)))
    return ll - mu * T - n * sum(1 - math.exp(-beta * (T - x)) for x in t)


def loglik2(mu: np.ndarray, nm: np.ndarray, beta: float,
            seqs: list[tuple[np.ndarray, np.ndarray, float]]) -> float:
    """Bivariate log-likelihood; marks 0/1, ``nm[k, j]`` = branching from type j to type k."""
    ll = 0.0
    for t, m, T in seqs:
        a0 = _excite(t, beta, m == 0)
        a1 = _excite(t, beta, m == 1)
        lam = mu[m] + beta * (nm[m, 0] * a0 + nm[m, 1] * a1)
        tail = 1 - np.exp(-beta * (T - t))
        ll += (float(np.log(lam).sum()) - float(mu.sum()) * T
               - float(nm[:, 0].sum()) * float(tail[m == 0].sum())
               - float(nm[:, 1].sum()) * float(tail[m == 1].sum()))
    return ll


def _ks(x: np.ndarray) -> tuple[float | None, float | None]:
    from scipy.stats import kstest

    if len(x) < 5:
        return None, None
    r = kstest(x, "expon")
    return float(r.statistic), float(r.pvalue)


def residuals1(seqs: list[tuple[np.ndarray, float]], mu: float, n: float, beta: float) -> np.ndarray:
    """Compensator increments ``Lambda(t_i) - Lambda(t_{i-1})`` (Exp(1) if the model is right)."""
    out = []
    for t, _ in seqs:
        lam_cum = mu * t + n * (np.arange(len(t)) - _excite(t, beta))
        out.append(np.diff(lam_cum, prepend=0.0))
    return np.concatenate(out) if out else np.zeros(0)


def fit_hawkes(seqs: list[tuple[np.ndarray, float]], beta0: tuple[float, ...] = (1.0, 0.1, 0.01),
               ) -> dict[str, Any]:
    """MLE of a univariate exponential Hawkes (shared parameters over all sequences)."""
    from scipy.optimize import minimize

    seqs = [(np.asarray(t, float), float(T)) for t, T in seqs if T > 0]
    N = sum(len(t) for t, _ in seqs)
    T = sum(T for _, T in seqs)
    out: dict[str, Any] = {"events": N, "seqs": len(seqs), "T_s": T}
    if N < 10 or T <= 0:
        return out
    rate = N / T

    bounds = [(math.log(1e-9), math.log(1e3)), (math.log(1e-6), math.log(NMAX)),
              (math.log(BETA_BOUNDS[0]), math.log(BETA_BOUNDS[1]))]
    best = None
    for b0 in beta0:
        r = minimize(_nll1_grad, [math.log(rate * 0.5), math.log(0.5), math.log(b0)], args=(seqs,),
                     jac=True, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    mu, n, beta = (float(v) for v in np.exp(best.x))
    ll_p = N * math.log(rate) - N
    d, p = _ks(residuals1(seqs, mu, n, beta))
    dp, pp = _ks(np.concatenate([np.diff(t, prepend=0.0) * rate for t, _ in seqs]))
    out.update(mu=mu, n=n, beta=beta, alpha=n * beta, decay_s=1 / beta, ll=-float(best.fun),
               ll_poisson=ll_p, ll_gain_per_event=(-float(best.fun) - ll_p) / N, ks_D=d, ks_p=p,
               ks_D_poisson=dp, ks_p_poisson=pp, n_at_bound=n > NMAX * 0.995,
               converged=bool(best.success))
    return out


def fit_hawkes2(seqs: list[tuple[np.ndarray, np.ndarray, float]],
                beta0: tuple[float, ...] = (1.0, 0.1, 0.01)) -> dict[str, Any]:
    """MLE of a bivariate exponential Hawkes with a shared decay (marks 0 = taker bought YES,
    1 = taker sold YES). ``n_kj`` = expected type-k children of one type-j event."""
    from scipy.optimize import minimize

    seqs = [(np.asarray(t, float), np.asarray(m, int), float(T)) for t, m, T in seqs if T > 0]
    N = [sum(int((m == k).sum()) for _, m, _ in seqs) for k in (0, 1)]
    T = sum(T for _, _, T in seqs)
    out: dict[str, Any] = {"events_buy": N[0], "events_sell": N[1], "seqs": len(seqs), "T_s": T}
    if min(N) < 10:
        return out

    def unpack(x):
        e = np.exp(x)
        return e[0:2], e[2:6].reshape(2, 2), float(e[6])

    lb = [math.log(1e-9)] * 2 + [math.log(1e-6)] * 4 + [math.log(BETA_BOUNDS[0])]
    ub = [math.log(1e3)] * 2 + [math.log(NMAX)] * 4 + [math.log(BETA_BOUNDS[1])]
    best = None
    for b0 in beta0:
        x0 = [math.log(N[0] / T * 0.5), math.log(N[1] / T * 0.5)] + [math.log(0.2)] * 4 + [math.log(b0)]
        r = minimize(_nll2_grad, x0, args=(seqs,), jac=True, method="L-BFGS-B", bounds=list(zip(lb, ub)))
        if best is None or r.fun < best.fun:
            best = r
    mu, nm, beta = unpack(best.x)
    res: list[list[np.ndarray]] = [[], []]
    for t, m, _ in seqs:
        a = [_excite(t, beta, m == j) for j in (0, 1)]
        cnt = [np.cumsum(m == j) - (m == j) for j in (0, 1)]       # type-j events before i
        for k in (0, 1):
            lam_cum = mu[k] * t + sum(nm[k, j] * (cnt[j] - a[j]) for j in (0, 1))
            res[k].append(np.diff(lam_cum[m == k], prepend=0.0))
    ks = [_ks(np.concatenate(r)) for r in res]
    out.update(mu_buy=float(mu[0]), mu_sell=float(mu[1]), n_bb=float(nm[0, 0]),
               n_bs=float(nm[0, 1]), n_sb=float(nm[1, 0]), n_ss=float(nm[1, 1]),
               rho=float(max(abs(np.linalg.eigvals(nm)))), beta=beta, decay_s=1 / beta,
               ll=-float(best.fun), ks_D_buy=ks[0][0], ks_p_buy=ks[0][1], ks_D_sell=ks[1][0],
               ks_p_sell=ks[1][1], converged=bool(best.success))
    return out


def simulate_hawkes(mu: float, n: float, beta: float, T: float, seed: int = 0) -> np.ndarray:
    """Ogata thinning (estimator tests only)."""
    rng = np.random.default_rng(seed)
    alpha = n * beta
    t, s, out = 0.0, 0.0, []           # s = excitation sum at time t
    while True:
        lam_bar = mu + alpha * s
        w = rng.exponential(1 / lam_bar)
        s *= math.exp(-beta * w)
        t += w
        if t > T:
            break
        if rng.uniform() * lam_bar <= mu + alpha * s:
            out.append(t)
            s += 1.0
    return np.array(out)


# ---- the L2 tape -------------------------------------------------------------------------------
def build_tape(events: Iterable[Event]) -> dict[str, pl.DataFrame]:
    """One pass over recorded events: trades, top-of-book changes and recorded executions, with
    receive and exchange times (snapshots carry no ``ts_ms``: their receive time is used)."""
    sim = Simulator(SimConfig(writes_per_s=None))
    trades, quotes, execs = [], [], []
    first: dict[str, int] = {}
    last: dict[str, int] = {}
    nq: dict[str, int] = defaultdict(int)
    for ev in events:
        if ev.kind not in ("snapshot", "delta", "trade"):
            continue
        sim.apply(ev)
        m = ev.market
        first.setdefault(m, ev.recv_ns)
        last[m] = ev.recv_ns
        ts = ev.msg.get("ts_ms") or ev.recv_ns // 10**6
        if ev.kind == "trade":
            trades.append((m, ev.recv_ns, ts, 1 if ev.msg["taker_side"] == "yes" else -1,
                           float(ev.msg["count_fp"]), float(ev.msg["yes_price_dollars"])))
            continue
        if ev.kind == "delta" and ev.is_exec:
            side, price = ev.msg["side"], float(ev.msg["price_dollars"])
            lev = sim.markets[m].levels[side].get(price)
            cleared = (lev is None or lev.ext_total() <= EPS) if m in sim.synced else None
            execs.append((m, ev.recv_ns, ts, side, price, -float(ev.msg["delta_fp"]), cleared))
        times, vals = sim.quotes.get(m, ([], []))
        if len(times) > nq[m]:
            nq[m] = len(times)
            quotes.append((m, ev.recv_ns, ts, vals[-1][0], vals[-1][1]))
    tr = pl.DataFrame(trades, schema={"market": pl.Utf8, "recv_ns": pl.Int64, "ts_ms": pl.Int64,
                                      "sign": pl.Int8, "qty": pl.Float64, "yes_price": pl.Float64},
                      orient="row")
    qu = pl.DataFrame(quotes, schema={"market": pl.Utf8, "recv_ns": pl.Int64, "ts_ms": pl.Int64,
                                      "bid": pl.Float64, "ask": pl.Float64}, orient="row")
    ex = pl.DataFrame(execs, schema={"market": pl.Utf8, "recv_ns": pl.Int64, "ts_ms": pl.Int64,
                                     "side": pl.Utf8, "price": pl.Float64, "qty": pl.Float64,
                                     "cleared": pl.Boolean}, orient="row")
    win = pl.DataFrame({"market": list(first), "first_ns": [first[m] for m in first],
                        "last_ns": [last[m] for m in first]})
    return {"trades": tr, "quotes": qu, "execs": ex, "windows": win}


def cached_tape(s, start: str, end: str, root: Path | None = None) -> dict[str, pl.DataFrame]:
    root = root or s.raw_dir / "stream"
    base = s.data_dir / "cache" / f"hawkes_tape_{start}_{end}".replace(":", "")
    names = ("trades", "quotes", "execs", "windows")
    if all(Path(f"{base}_{n}.parquet").exists() for n in names):
        return {n: pl.read_parquet(f"{base}_{n}.parquet") for n in names}
    stats: dict[str, int] = defaultdict(int)
    tape = build_tape(iter_events(stream_files(root, start, end), stats))
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    for n in names:
        tape[n].write_parquet(f"{base}_{n}.parquet")
    return tape


def merged_trades(trades: pl.DataFrame) -> pl.DataFrame:
    """One arrival per (market, exchange ms, taker side): a sweep is one event."""
    return (trades.group_by("market", "ts_ms", "sign", maintain_order=True)
            .agg(pl.col("qty").sum(), pl.col("yes_price").last(), pl.col("recv_ns").first())
            .sort("market", "ts_ms"))


def mid_events(quotes: pl.DataFrame) -> pl.DataFrame:
    """Mid changes (two-sided book on both sides of the change), one per (market, ms)."""
    q = quotes.sort("market", "recv_ns").with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    q = q.with_columns(pl.col("mid").shift(1).over("market").alias("prev"))
    q = q.filter(pl.col("mid").is_not_null() & pl.col("prev").is_not_null()
                 & ((pl.col("mid") - pl.col("prev")).abs() > 1e-9))
    return q.group_by("market", "ts_ms", maintain_order=True).agg(
        (pl.col("mid").last() - pl.col("prev").first()).alias("dmid")).sort("market", "ts_ms")


def series_of(ticker: str) -> str:
    return ticker.split("-", 1)[0]


def event_of(ticker: str) -> str:
    return ticker.rsplit("-", 1)[0]


def l2_sequences(ev: pl.DataFrame, windows: pl.DataFrame, t0_ns: int, t1_ns: int,
                 marks: bool = False) -> dict[str, tuple]:
    """market -> (t seconds from the window start[, marks], T seconds) for events with exchange
    time inside the market's observed window clipped to [t0, t1)."""
    w = {r["market"]: (max(r["first_ns"], t0_ns) // 10**6, min(r["last_ns"], t1_ns) // 10**6)
         for r in windows.iter_rows(named=True)}
    out = {}
    for (mkt,), g in ev.group_by("market"):
        if mkt not in w:
            continue
        a, b = w[mkt]
        if b <= a:
            continue
        g = g.filter((pl.col("ts_ms") >= a) & (pl.col("ts_ms") < b)).sort("ts_ms")
        if g.is_empty():
            continue
        t = (g["ts_ms"].to_numpy() - a) / 1e3
        if marks:
            out[mkt] = (t, (g["sign"].to_numpy() < 0).astype(int), (b - a) / 1e3)
        else:
            out[mkt] = (t, (b - a) / 1e3)
    return out


def fit_table(seqs: dict[str, tuple], *, label: dict[str, str], min_events: int = 30,
              per_market: bool = True, bivariate: bool = False) -> list[dict[str, Any]]:
    """Fits per market (>= ``min_events``), pooled per series, and pooled over everything."""
    fit = fit_hawkes2 if bivariate else fit_hawkes
    rows = []
    n_of = {k: len(v[0]) for k, v in seqs.items()}
    if per_market:
        for k, v in seqs.items():
            if n_of[k] >= min_events:
                rows.append({"scope": "market", "key": k} | fit([v]))
    by_series: dict[str, list] = defaultdict(list)
    for k, v in seqs.items():
        by_series[series_of(k)].append(v)
    for ser, vs in by_series.items():
        if sum(len(v[0]) for v in vs) >= min_events:
            rows.append({"scope": "series", "key": ser, "markets": len(vs)} | fit(vs))
    rows.append({"scope": "all", "key": "all", "markets": len(seqs)} | fit(list(seqs.values())))
    return [r | {"label": json.dumps(label)} for r in rows]


# ---- toxicity features from the tape (offline, exchange time) -------------------------------------
@dataclass
class _Arrays:
    tr_t: np.ndarray            # merged trade arrival times (s, exchange)
    tr_sq: np.ndarray           # signed qty per arrival (+ taker bought YES)
    cum_sq: np.ndarray
    cum_n: np.ndarray
    lse_tr: np.ndarray          # cumulative log-sum-exp of beta * t (trades)
    vpin_cum: np.ndarray        # cumulative bucket imbalance (VPIN)
    cum_v: float
    tr_cumv: np.ndarray         # cumulative volume after each arrival (leading 0)
    md_t: np.ndarray            # mid-change times (s, exchange)
    cum_d2: np.ndarray
    lse_md: np.ndarray
    q_t: np.ndarray             # quote change times (s, exchange) and the spread after them
    q_spread: np.ndarray
    first_s: float


@dataclass
class ToxParams:
    mu: float
    n: float
    beta: float
    mu_mid: float
    n_mid: float
    beta_mid: float
    vpin_v: float = 50.0
    vpin_nb: int = 10
    rate_s: float = 3600.0
    min_rate_s: float = 600.0

    @property
    def alpha(self) -> float:
        return self.n * self.beta

    @property
    def alpha_mid(self) -> float:
        return self.n_mid * self.beta_mid


def _arrays(tr: pl.DataFrame, md: pl.DataFrame, qu: pl.DataFrame, first_s: float,
            p: ToxParams) -> _Arrays:
    t = tr["ts_ms"].to_numpy() / 1e3 if tr.height else np.zeros(0)
    sq = (tr["sign"].to_numpy() * tr["qty"].to_numpy()) if tr.height else np.zeros(0)
    q = tr["qty"].to_numpy() if tr.height else np.zeros(0)
    t0 = first_s
    cum_v = np.concatenate([[0.0], np.cumsum(q)])
    cum_b = np.concatenate([[0.0], np.cumsum(np.where(sq > 0, q, 0.0))])
    kmax = int(cum_v[-1] // p.vpin_v)
    edges = np.arange(kmax + 1) * p.vpin_v
    b_edges = np.interp(edges, cum_v, cum_b) if kmax else np.zeros(1)
    imb = np.abs(2 * np.diff(b_edges) - p.vpin_v) / p.vpin_v if kmax else np.zeros(0)
    mt = md["ts_ms"].to_numpy() / 1e3 if md.height else np.zeros(0)
    d2 = md["dmid"].to_numpy() ** 2 if md.height else np.zeros(0)
    qt = qu["ts_ms"].to_numpy() / 1e3 if qu.height else np.zeros(0)
    spread = (qu["ask"] - qu["bid"]).fill_null(np.nan).to_numpy() if qu.height else np.zeros(0)
    return _Arrays(
        tr_t=t, tr_sq=sq, cum_sq=np.concatenate([[0.0], np.cumsum(sq)]),
        cum_n=np.arange(len(t) + 1, dtype=float),
        lse_tr=np.logaddexp.accumulate(p.beta * (t - t0)) if len(t) else np.zeros(0),
        vpin_cum=np.concatenate([[0.0], np.cumsum(imb)]), cum_v=float(cum_v[-1]), tr_cumv=cum_v,
        md_t=mt, cum_d2=np.concatenate([[0.0], np.cumsum(d2)]),
        lse_md=np.logaddexp.accumulate(p.beta_mid * (mt - t0)) if len(mt) else np.zeros(0),
        q_t=qt, q_spread=spread, first_s=t0)


def _decayed(lse: np.ndarray, idx: np.ndarray, bq: np.ndarray) -> np.ndarray:
    """``sum_{j < idx} exp(beta (t_j - q))`` from the cumulative log-sum-exp ``lse`` of
    ``beta t``; ``bq`` = beta * q. Zero where no event precedes."""
    if len(lse) == 0:
        return np.zeros(len(idx))
    x = np.where(idx > 0, lse[np.maximum(idx - 1, 0)] - bq, -np.inf)
    return np.exp(x)


def _feat_market(a: _Arrays, cut: np.ndarray, sign_taker: np.ndarray, p: ToxParams) -> dict[str, np.ndarray]:
    """Features at exchange times ``cut`` (s) from events strictly before ``cut``.
    ``sign_taker`` = +1 if the taker who hits the maker bought YES (maker sold YES / bought NO)."""
    hi = np.searchsorted(a.tr_t, cut, "left")                 # arrivals strictly before cut
    lo10 = np.searchsorted(a.tr_t, cut - 10, "left")
    lo60 = np.searchsorted(a.tr_t, cut - 60, "left")
    lo_r = np.searchsorted(a.tr_t, cut - p.rate_s, "left")
    rel = cut - a.first_s
    burst = _decayed(a.lse_tr, hi, p.beta * rel)
    lam = p.mu + p.alpha * burst
    span = np.clip(rel, p.min_rate_s, p.rate_s)
    rate = (a.cum_n[hi] - a.cum_n[lo_r]) / span
    ofi10 = a.cum_sq[hi] - a.cum_sq[lo10]
    ofi60 = a.cum_sq[hi] - a.cum_sq[lo60]
    k = np.floor(a.tr_cumv[hi] / p.vpin_v).astype(int)
    vpin = np.where(k >= p.vpin_nb,
                    (a.vpin_cum[np.minimum(k, len(a.vpin_cum) - 1)]
                     - a.vpin_cum[np.clip(k - p.vpin_nb, 0, len(a.vpin_cum) - 1)]) / p.vpin_nb, np.nan)
    mhi = np.searchsorted(a.md_t, cut, "left")
    mlo = np.searchsorted(a.md_t, cut - 60, "left")
    burst_m = _decayed(a.lse_md, mhi, p.beta_mid * rel)
    qi = np.searchsorted(a.q_t, cut, "left") - 1
    spread = np.where(qi >= 0, a.q_spread[np.maximum(qi, 0)] if len(a.q_spread) else np.nan, np.nan)
    return {"lam": lam, "burst": burst, "lam_rel": lam / np.maximum(rate, p.mu), "rate1h": rate,
            "lam_mid": p.mu_mid + p.alpha_mid * burst_m, "vpin": vpin,
            "ofi10": ofi10, "ofi60": ofi60, "ofi_adv10": sign_taker * ofi10,
            "ofi_adv60": sign_taker * ofi60, "abs_ofi60": np.abs(ofi60),
            "rv60": 100 * np.sqrt(a.cum_d2[mhi] - a.cum_d2[mlo]), "n10": a.cum_n[hi] - a.cum_n[lo10],
            "n60": a.cum_n[hi] - a.cum_n[lo60], "spread_c": 100 * spread}


def features(tape: dict[str, pl.DataFrame], queries: pl.DataFrame, p: ToxParams) -> pl.DataFrame:
    """``queries`` has market, cut_ms (exchange ms; features use events with ts < cut_ms) and
    side (the maker's side: yes = the maker bought YES, hit by a taker selling YES)."""
    tr = merged_trades(tape["trades"])
    md = mid_events(tape["quotes"])
    qu = tape["quotes"].sort("market", "ts_ms")
    first = {r["market"]: r["first_ns"] / 1e9 for r in tape["windows"].iter_rows(named=True)}
    trg = {k[0]: g for k, g in tr.group_by("market")}
    mdg = {k[0]: g for k, g in md.group_by("market")}
    qug = {k[0]: g for k, g in qu.group_by("market")}
    empty_tr, empty_md, empty_qu = tr.head(0), md.head(0), qu.head(0)
    parts = []
    q = queries.with_row_index("_i")
    for (mkt,), g in q.group_by("market"):
        a = _arrays(trg.get(mkt, empty_tr), mdg.get(mkt, empty_md), qug.get(mkt, empty_qu),
                    first.get(mkt, g["cut_ms"].min() / 1e3), p)
        cut = g["cut_ms"].to_numpy() / 1e3
        st = np.where(g["side"].to_numpy() == "yes", -1.0, 1.0)
        f = _feat_market(a, cut, st, p)
        parts.append(pl.DataFrame({"_i": g["_i"]} | {k: v.astype(float) for k, v in f.items()}))
    if not parts:
        return queries
    return q.join(pl.concat(parts), on="_i").sort("_i").drop("_i")


# ---- markouts of recorded executions -------------------------------------------------------------
def exec_markouts(tape: dict[str, pl.DataFrame], execs: pl.DataFrame,
                  horizons: tuple[int, ...] = HORIZONS) -> pl.DataFrame:
    """Per recorded execution: the resting order bought ``side`` at ``price``; ``mo_h`` = mid
    value at recv + h minus price, ``mx_h`` = best bid on that side minus price (dollars)."""
    qu = tape["quotes"].sort("market", "recv_ns")
    end = int(qu["recv_ns"].max()) if qu.height else 0
    qg = {k[0]: g for k, g in qu.group_by("market")}
    parts = []
    e = execs.with_row_index("_i")
    for (mkt,), g in e.group_by("market"):
        if mkt not in qg:
            continue
        qq = qg[mkt]
        qt, bid, ask = qq["recv_ns"].to_numpy(), qq["bid"].fill_null(np.nan).to_numpy(), qq["ask"].fill_null(np.nan).to_numpy()
        t = g["recv_ns"].to_numpy()
        yes = g["side"].to_numpy() == "yes"
        price = g["price"].to_numpy()
        cols = {"_i": g["_i"]}
        for h in horizons:
            th = t + h * NS
            i = np.searchsorted(qt, th, "right") - 1
            ok = (i >= 0) & (th <= end)
            b = np.where(ok, bid[np.maximum(i, 0)], np.nan)
            a = np.where(ok, ask[np.maximum(i, 0)], np.nan)
            mid = (a + b) / 2
            cols[f"mo_{h}s"] = np.where(yes, mid, 1 - mid) - price
            cols[f"mx_{h}s"] = np.where(yes, b, 1 - a) - price
        parts.append(pl.DataFrame(cols))
    out = e.join(pl.concat(parts), on="_i", how="left").sort("_i").drop("_i")
    return out.with_columns([pl.col(c).fill_nan(None) for c in out.columns if c[:3] in ("mo_", "mx_")])


# ---- clustered statistics ------------------------------------------------------------------------
def cl_mean(y: np.ndarray, g: np.ndarray) -> tuple[float | None, float | None, int]:
    """Mean and t-stat with SE clustered by ``g`` (equal weight per observation)."""
    ok = ~np.isnan(y)
    y, g = y[ok], g[ok]
    if len(y) == 0:
        return None, None, 0
    m = float(y.mean())
    _, inv = np.unique(g, return_inverse=True)
    k = int(inv.max()) + 1
    if k < 2:
        return m, None, k
    s = np.bincount(inv, weights=y - m)
    se = math.sqrt(k / (k - 1) * float((s ** 2).sum())) / len(y)
    return m, (m / se if se > 0 else None), k


def cl_ols(x: np.ndarray, y: np.ndarray, g: np.ndarray) -> tuple[float | None, float | None, int, int]:
    """Slope of y on x with SE clustered by g: (b, t, n, clusters)."""
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y, g = x[ok], y[ok], g[ok]
    if len(y) < 3:
        return None, None, len(y), 0
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    if sxx <= 0:
        return None, None, len(y), 0
    b = float((xc * (y - y.mean())).sum()) / sxx
    e = y - y.mean() - b * xc
    _, inv = np.unique(g, return_inverse=True)
    k = int(inv.max()) + 1
    if k < 2:
        return b, None, len(y), k
    s = np.bincount(inv, weights=xc * e)
    se = math.sqrt(k / (k - 1) * float((s ** 2).sum())) / sxx
    return b, (b / se if se > 0 else None), len(y), k


def _transform(name: str, x: np.ndarray) -> np.ndarray:
    if name in _LOG:
        return np.log1p(np.maximum(x, 0))
    if name in _SLOG:
        return np.sign(x) * np.log1p(np.abs(x))
    return x


def spread_bin(spread_c: np.ndarray) -> np.ndarray:
    """Quoted-spread bins (cents) for the fixed-effect control: 1, 2, 3, 4, 5-6, 7-10, 11+, none."""
    x = np.round(spread_c)
    return np.select([np.isnan(x), x <= 4, x <= 6, x <= 10], [-1, x, 5, 7], 11).astype(int)


def _demean(v: np.ndarray, grp: np.ndarray) -> np.ndarray:
    """Subtract group means (ignoring NaN) from ``v``."""
    ok = ~np.isnan(v)
    _, inv = np.unique(grp, return_inverse=True)
    s = np.bincount(inv[ok], weights=v[ok], minlength=inv.max() + 1)
    c = np.bincount(inv[ok], minlength=inv.max() + 1)
    return v - np.where(c[inv] > 0, s[inv] / np.maximum(c[inv], 1), 0.0)


def tox_regressions(df: pl.DataFrame, *, measures=MEASURES, ycols: tuple[str, ...],
                    dev_col: str = "dev", vol_col: str = "volatile",
                    cluster: str = "event", control: str | None = None) -> pl.DataFrame:
    """Per measure x markout x {all, volatile, calm} x {dev, holdout}: slope (cents per dev SD of
    the transformed measure) and the high-minus-rest gap (high = above the dev 80th percentile),
    both with t-stats clustered by ``cluster``. ``control="spread"`` absorbs quoted-spread bins
    (fixed effects within each subset): wide markets have mechanically worse exit markouts and
    different activity, which otherwise confounds every activity measure."""
    rows = []
    dev = df[dev_col].to_numpy()
    fe = spread_bin(df["spread_c"].fill_null(np.nan).to_numpy()) if control == "spread" else None
    for name in measures:
        if name not in df.columns:
            continue
        x = _transform(name, df[name].cast(pl.Float64).fill_null(np.nan).to_numpy())
        xd = x[dev & ~np.isnan(x)]
        if len(xd) < 10 or xd.std() == 0:
            continue
        z = (x - xd.mean()) / xd.std()
        cut = float(np.quantile(xd, 0.8))
        hi = np.where(np.isnan(x), np.nan, (x > cut).astype(float))
        for yc in ycols:
            y = 100 * df[yc].cast(pl.Float64).fill_null(np.nan).to_numpy()
            for regime in ("all", "volatile", "calm"):
                for split in ("dev", "holdout"):
                    sel = dev if split == "dev" else ~dev
                    if regime != "all":
                        v = df[vol_col].fill_null(False).to_numpy()
                        sel = sel & (v if regime == "volatile" else ~v)
                    g = df[cluster].to_numpy()[sel]
                    xs, ys, hs = z[sel], y[sel], hi[sel]
                    if fe is not None and sel.any():
                        both = ~(np.isnan(xs) | np.isnan(ys))
                        xs, ys, hs = (np.where(both, v, np.nan) for v in (xs, ys, hs))
                        xs, ys, hs = (_demean(v, fe[sel]) for v in (xs, ys, hs))
                    b, t, n, k = cl_ols(xs, ys, g)
                    bh, th, _, _ = cl_ols(hs, ys, g)
                    nh = int(np.nansum(hi[sel]))
                    rows.append({"measure": name, "y": yc, "regime": regime, "split": split,
                                 "control": control or "none", "n": n,
                                 "clusters": k, "slope_c_per_sd": b, "t_slope": t,
                                 "n_high": nh, "high_minus_rest_c": bh, "t_high": th})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- online measures and the filtered quoter -------------------------------------------------------
@dataclass
class _MState:
    s: float = 0.0               # Hawkes excitation at t_last
    t_last: float = 0.0
    key: tuple = ()
    first: float | None = None
    arrivals: deque = field(default_factory=deque)     # arrival times (1 h window)
    flow: deque = field(default_factory=deque)         # (t, signed qty) (60 s window)
    mids: deque = field(default_factory=deque)         # (t, dmid^2) (60 s window)
    last_mid: float | None = None
    d2: float = 0.0
    sq: float = 0.0


class OnlineTox:
    """Per-market live measures in receive time, from market data only (shared by all
    strategies of a run): Hawkes intensity ``lam`` (dev-fitted parameters), ``lam_rel`` (relative
    to the trailing hour's arrival rate), ``rv60`` (cents, sqrt of summed squared mid changes over
    60 s) and ``ofi60`` (|taker buy-YES minus sell-YES contracts| over 60 s)."""

    def __init__(self, p: ToxParams, window_s: float = 60.0) -> None:
        self.p, self.w = p, window_s
        self.st: dict[str, _MState] = defaultdict(_MState)

    def observe(self, sim: Simulator, ev: Event) -> None:
        if ev.kind not in ("trade", "delta", "snapshot"):
            return
        st = self.st[ev.market]
        t = ev.recv_ns / 1e9
        if st.first is None:
            st.first = t
        if ev.kind == "trade":
            sign = 1.0 if ev.msg["taker_side"] == "yes" else -1.0
            key = (ev.msg.get("ts_ms"), sign)
            if key != st.key:
                st.s = st.s * math.exp(-self.p.beta * max(t - st.t_last, 0.0)) + 1.0
                st.t_last, st.key = t, key
                st.arrivals.append(t)
            x = sign * float(ev.msg["count_fp"])
            st.flow.append((t, x))
            st.sq += x
            return
        times, vals = sim.quotes.get(ev.market, ([], []))
        if not vals or times[-1] != ev.recv_ns:
            return
        b, a = vals[-1]
        if b is None or a is None:
            return
        mid = (a + b) / 2
        if st.last_mid is not None and abs(mid - st.last_mid) > 1e-9:
            d = (mid - st.last_mid) ** 2
            st.mids.append((t, d))
            st.d2 += d
        st.last_mid = mid

    def measures(self, market: str, now_ns: int) -> dict[str, float]:
        st = self.st.get(market)
        if st is None or st.first is None:
            return {"lam": self.p.mu, "lam_rel": 1.0, "rv60": 0.0, "ofi60": 0.0}
        t = now_ns / 1e9
        while st.flow and st.flow[0][0] < t - self.w:
            st.sq -= st.flow.popleft()[1]
        while st.mids and st.mids[0][0] < t - self.w:
            st.d2 -= st.mids.popleft()[1]
        while st.arrivals and st.arrivals[0] < t - self.p.rate_s:
            st.arrivals.popleft()
        lam = self.p.mu + self.p.alpha * st.s * math.exp(-self.p.beta * max(t - st.t_last, 0.0))
        span = min(max(t - st.first, self.p.min_rate_s), self.p.rate_s)
        rate = len(st.arrivals) / span
        return {"lam": lam, "lam_rel": lam / max(rate, self.p.mu),
                "rv60": 100 * math.sqrt(max(st.d2, 0.0)), "ofi60": abs(st.sq) if st.flow else 0.0}


class ToxQuoter(Quoter):
    """Baseline ``Quoter`` plus one toxicity filter:

    * ``pull_on`` / ``pull_thr``: no quotes (existing ones cancelled) while the live measure
      exceeds the threshold;
    * ``vol_c``: minimum spread ``max(min_spread, vol_c * rv60)`` (rv60 in dollars);
    * ``widen_on`` / ``widen_thr``: quote one tick less aggressively (at the external best, i.e.
      join instead of improving) while the measure exceeds the threshold.
    """

    def __init__(self, tox: OnlineTox | None = None, *, pull_on: str | None = None,
                 pull_thr: float = math.inf, widen_on: str | None = None,
                 widen_thr: float = math.inf, vol_c: float = 0.0, **kw: Any) -> None:
        super().__init__(**kw)
        self.tox = tox
        self.pull_on, self.pull_thr = pull_on, pull_thr
        self.widen_on, self.widen_thr = widen_on, widen_thr
        self.vol_c = vol_c
        self.pulls = self.widens = self.vol_blocks = 0

    def _wanted(self, sim: Simulator, market: str) -> dict[str, float | None]:
        want = super()._wanted(sim, market)
        if self.tox is None or (want["yes"] is None and want["no"] is None):
            return want
        if not (self.pull_on or self.widen_on or self.vol_c):
            return want
        m = self.tox.measures(market, sim.now)
        if self.pull_on and m[self.pull_on] > self.pull_thr:
            self.pulls += 1
            return {"yes": None, "no": None}
        if self.vol_c:
            q = sim.markets[market].quote()
            if q["yes_ask"] - q["yes_bid"] < max(self.min_spread, self.vol_c * m["rv60"] / 100) - EPS:
                self.vol_blocks += 1
                return {"yes": None, "no": None}
        if self.widen_on and m[self.widen_on] > self.widen_thr:
            self.widens += 1
            return {k: (round(v - TICK, 6) if v is not None and v - TICK > EPS else None)
                    for k, v in want.items()}
        return want


class TSim(Simulator):
    """Simulator that keeps each fill's exchange time (``ts_ms`` of the event that filled it) and
    drops finished orders from its per-market order lists (identical results, faster)."""

    def __init__(self, cfg: SimConfig) -> None:
        super().__init__(cfg)
        self.fill_ts: list[int] = []
        self._ts = 0

    def apply(self, ev: Event) -> None:
        self._ts = (ev.msg.get("ts_ms") if ev.kind != "ext" else None) or ev.recv_ns // 10**6
        super().apply(ev)
        ids = self.by_market.get(ev.market)
        if ids is not None and len(ids) > 32:
            self.by_market[ev.market] = [i for i in ids if self.orders[i].status
                                         not in ("filled", "cancelled", "rejected")]

    def _fill(self, o, qty: float, how: str) -> None:
        k = len(self.fills)
        super()._fill(o, qty, how)
        if len(self.fills) > k:
            self.fill_ts.append(self._ts)


def run_many(events: Iterable[Event], strategies: dict[str, tuple[ToxQuoter, SimConfig]],
             tox: OnlineTox | None) -> tuple[dict[str, TSim], dict[str, list[dict[str, float]]]]:
    """Replay once, driving every strategy in lockstep on its own simulator (the recorded book,
    hence the quote history, is the same for all and is shared). Returns the simulators and, per
    strategy, the live measures at each fill (state before the event that filled it)."""
    sims = {k: TSim(cfg) for k, (_, cfg) in strategies.items()}
    shared = next(iter(sims.values())).quotes if sims else None
    for sim in sims.values():
        sim.quotes = shared
    at_fill: dict[str, list[dict[str, float]]] = defaultdict(list)
    for ev in events:
        for k, (strat, _) in strategies.items():
            sim = sims[k]
            n = len(sim.fills)
            sim.apply(ev)
            if tox is not None:
                for f in sim.fills[n:]:
                    at_fill[k].append(tox.measures(f.market, f.t_ns))
        if tox is not None and sims:
            tox.observe(next(iter(sims.values())), ev)
        for k, (strat, _) in strategies.items():
            strat.on_event(sims[k], ev.market)
    return sims, at_fill


def fill_frame(sim: TSim, at_fill: list[dict[str, float]] | None = None,
               horizons: tuple[int, ...] = HORIZONS) -> pl.DataFrame:
    end = max((t[-1] for t, _ in sim.quotes.values() if t), default=0)
    rows = []
    for i, f in enumerate(sim.fills):
        r = {"t_ns": f.t_ns, "ts_ms": sim.fill_ts[i], "market": f.market, "event": event_of(f.market),
             "side": f.side, "price": f.price, "qty": f.qty, "fee": f.fee, "how": f.how}
        for h in horizons:
            t = f.t_ns + h * NS
            mid = sim.mid_at(f.market, t) if t <= end else None
            ex = sim.exit_at(f.market, f.side, t) if t <= end else None
            r[f"mo_{h}s"] = None if mid is None else (mid if f.side == "yes" else 1 - mid) - f.price
            r[f"mx_{h}s"] = None if ex is None else ex - f.price
        r["last_mid"] = sim.mid_at(f.market, 2**62)
        if at_fill:
            r |= {f"live_{k}": v for k, v in at_fill[i].items()}
        rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- settlement and scoring -------------------------------------------------------------------------
def market_results(tickers: Iterable[str], cache: Path) -> dict[str, float | None]:
    """Settled YES result (1.0 / 0.0) per market from the public REST API, cached; unsettled
    markets are re-queried on each call."""
    import httpx

    have = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [t for t in sorted(set(tickers)) if have.get(t) is None]
    if todo:
        with httpx.Client(timeout=20) as http:
            for t in todo:
                try:
                    r = http.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{t}")
                except httpx.HTTPError:
                    continue
                if r.status_code == 200:
                    res = r.json().get("market", {}).get("result")
                    have[t] = {"yes": 1.0, "no": 0.0}.get(res)
                time.sleep(0.05)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(have))
    return {t: have.get(t) for t in tickers}


def taker_fee(price: float, mult: float = 1.0) -> float:
    """Kalshi taker fee per contract, 0.07 P (1-P), rounded up to $0.0001 (balance precision)."""
    return math.ceil(0.07 * mult * price * (1 - price) * 1e4 - 1e-9) / 1e4


def with_flags(fills: pl.DataFrame, vol: pl.DataFrame, split_ns: int, results: dict[str, float | None],
               bucket_s: int = 300) -> pl.DataFrame:
    """Adds dev (t < split), volatile (the fill's own 5-min bucket flag: an evaluation split),
    settlement and PnL per contract net of the maker fee (settlement, else the last recorded mid),
    and ``mxn_*``: exit markout net of the maker fee and the taker fee to cross out."""
    bns = bucket_s * NS
    v = vol.select("market", "bucket_ns", "volatile")
    f = fills.with_columns((pl.col("t_ns") // bns * bns).alias("bucket_ns"), (pl.col("t_ns") < split_ns).alias("dev"))
    f = f.join(v, on=["market", "bucket_ns"], how="left").with_columns(pl.col("volatile").fill_null(False))
    f = f.with_columns(pl.col("market").replace_strict(results, default=None, return_dtype=pl.Float64).alias("settle_yes"))
    f = f.with_columns(pl.coalesce("settle_yes", "last_mid").alias("final_yes"))
    val = pl.when(pl.col("side") == "yes").then(pl.col("final_yes")).otherwise(1 - pl.col("final_yes"))
    f = f.with_columns((val - pl.col("price") - pl.col("fee") / pl.col("qty")).alias("pnl"),
                       pl.col("settle_yes").is_not_null().alias("settled"))
    for h in HORIZONS:
        c = f"mx_{h}s"
        if c in f.columns:
            exitp = pl.col(c) + pl.col("price")
            f = f.with_columns((pl.col(c) - pl.col("fee") / pl.col("qty")
                                - exitp.map_elements(lambda p: taker_fee(p) if p is not None else None,
                                                     return_dtype=pl.Float64)).alias(f"mxn_{h}s"))
    return f


def score_frame(f: pl.DataFrame, cols: tuple[str, ...] = ("mo_60s", "mo_300s", "mx_60s", "mx_300s",
                                                           "mxn_300s", "pnl")) -> list[dict[str, Any]]:
    rows = []
    for regime in ("all", "volatile", "calm"):
        for split in ("dev", "holdout"):
            g = f.filter(pl.col("dev") if split == "dev" else ~pl.col("dev"))
            if regime != "all":
                g = g.filter(pl.col("volatile") if regime == "volatile" else ~pl.col("volatile"))
            r: dict[str, Any] = {"regime": regime, "split": split, "fills": g.height,
                                 "contracts": float(g["qty"].sum()) if g.height else 0.0,
                                 "events": g["event"].n_unique() if g.height else 0,
                                 "pnl_usd": float((g["qty"] * g["pnl"]).sum()) if g.height else 0.0}
            for c in cols:
                if g.height == 0:
                    r[c], r[f"t_{c}"] = None, None
                    continue
                y = g[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
                m, t, _ = cl_mean(y, g["event"].to_numpy())
                r[c] = None if m is None else round(100 * m, 2)
                r[f"t_{c}"] = None if t is None else round(t, 2)
            rows.append(r)
    return rows


# ---- historical trades -----------------------------------------------------------------------------
GAMES = ("KXNFLGAME", "KXNBAGAME", "KXMLBGAME", "KXNCAAFGAME")
INPLAY_ONSET_H = 3.0     # trade rate doubles between exp-3.25h and exp-3.0h (NBA/NFL/NCAAF, verified)
PREGAME_END_H = 3.5      # pre-game sample ends here (before the onset)


def hist_coverage(s, series: tuple[str, ...] = GAMES) -> pl.DataFrame:
    """Per game market: trades, in-play trades ([exp - onset, close]) and whether the trades reach
    the close (most markets only have the pre-game maker window, anchor - 25 h .. anchor - 4 h)."""
    m = (pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet")
         .filter(pl.col("series_ticker").is_in(list(series)))
         .select("ticker", "event_ticker", "series_ticker", "expected_expiration_time", "close_time",
                 "result", "fee_multiplier").collect())
    tr = (pl.scan_parquet(s.tables_dir / "kalshi_trades.parquet").join(m.lazy(), on="ticker")
          .group_by("ticker").agg(pl.len().alias("trades"), pl.col("created_time").min().alias("first"),
                                  pl.col("created_time").max().alias("last"),
                                  (pl.col("created_time") >= pl.col("expected_expiration_time")
                                   - pl.duration(minutes=int(INPLAY_ONSET_H * 60))).sum().alias("inplay_trades"))
          .collect())
    out = m.join(tr, on="ticker")
    return out.with_columns(
        (pl.col("last") >= pl.col("close_time") - pl.duration(minutes=15)).alias("to_close"),
        (pl.col("expected_expiration_time") < pl.datetime(2026, 10, 1, time_zone="UTC")).alias("dev"))


def hist_trades(s, tickers: list[str]) -> pl.DataFrame:
    """Merged arrivals (one per ticker, microsecond, taker side) of the given markets: sign +1 =
    taker bought YES; price = the sweep's last print (max for buys, min for sells)."""
    t = (pl.scan_parquet(s.tables_dir / "kalshi_trades.parquet")
         .filter(pl.col("ticker").is_in(tickers)).collect())
    t = t.with_columns(pl.when(pl.col("taker_side") == "yes").then(1).otherwise(-1).cast(pl.Int8).alias("sign"))
    return (t.group_by("ticker", "created_time", "sign")
            .agg(pl.col("count").sum().alias("qty"),
                 pl.when(pl.col("sign").first() > 0).then(pl.col("yes_price").max())
                 .otherwise(pl.col("yes_price").min()).alias("price"))
            .sort("ticker", "created_time"))


def hist_sequences(tr: pl.DataFrame, windows: dict[str, tuple[int, int]], marks: bool = False) -> dict[str, tuple]:
    """ticker -> (t seconds from window start[, marks], T) for arrivals inside [a, b) (epoch us)."""
    out = {}
    for (tk,), g in tr.group_by("ticker"):
        if tk not in windows:
            continue
        a, b = windows[tk]
        us = g["created_time"].dt.epoch("us").to_numpy()
        sel = (us >= a) & (us < b)
        if sel.sum() == 0 or b <= a:
            continue
        o = np.argsort(us[sel], kind="stable")
        t = (us[sel][o] - a) / 1e6
        if marks:
            out[tk] = (t, (g["sign"].to_numpy()[sel][o] < 0).astype(int), (b - a) / 1e6)
        else:
            out[tk] = (t, (b - a) / 1e6)
    return out


def _print_mid(t: np.ndarray, sg: np.ndarray, px: np.ndarray, q: np.ndarray, strict: bool) -> np.ndarray:
    """Mid proxy from prints: mean of the last taker-buy price (ask proxy) and the last
    taker-sell price (bid proxy) before ``q`` (``strict``) or at/before it. Removes the bid-ask
    bounce of the last print; NaN until both sides have printed."""
    side = "left" if strict else "right"
    out = []
    for mask in (sg > 0, sg < 0):
        ts, ps = t[mask], px[mask]
        i = np.searchsorted(ts, q, side) - 1
        out.append(np.where(i >= 0, ps[np.maximum(i, 0)] if len(ps) else np.nan, np.nan))
    return (out[0] + out[1]) / 2


def hist_grid(tr: pl.DataFrame, windows: dict[str, tuple[int, int]], p2: dict[str, Any],
              step_s: float = 10.0, horizons: tuple[int, ...] = (60, 300)) -> pl.DataFrame:
    """On a time grid inside each window: causal bivariate intensities (dev parameters ``p2``),
    burst = total intensity / baseline, imbalance = (lam_buy - lam_sell) / total, the last print
    price strictly before t and the price change to the last print at t + h."""
    mu = np.array([p2["mu_buy"], p2["mu_sell"]])
    nm = np.array([[p2["n_bb"], p2["n_bs"]], [p2["n_sb"], p2["n_ss"]]])
    beta = p2["beta"]
    parts = []
    for (tk,), g in tr.group_by("ticker"):
        if tk not in windows:
            continue
        a, b = windows[tk]
        g = g.sort("created_time")
        t = (g["created_time"].dt.epoch("us").to_numpy() - a) / 1e6
        T = (b - a) / 1e6
        keep = (t >= 0) & (t < T)
        t, sg, px = t[keep], g["sign"].to_numpy()[keep], g["price"].to_numpy()[keep]
        if len(t) < 20:
            continue
        q = np.arange(step_s, T - max(horizons), step_s)
        if len(q) == 0:
            continue
        idx = np.searchsorted(t, q, "left")
        s = []
        for j, mask in ((0, sg > 0), (1, sg < 0)):
            lse = np.logaddexp.accumulate(np.where(mask, beta * t, -np.inf))
            s.append(np.where(idx > 0, np.exp(lse[np.maximum(idx - 1, 0)] - beta * q), 0.0))
        lb = mu[0] + beta * (nm[0, 0] * s[0] + nm[0, 1] * s[1])
        ls = mu[1] + beta * (nm[1, 0] * s[0] + nm[1, 1] * s[1])
        p0 = np.where(idx > 0, px[np.maximum(idx - 1, 0)], np.nan)
        m0 = _print_mid(t, sg, px, q, strict=True)
        cols = {"ticker": tk, "t_s": q, "lam_buy": lb, "lam_sell": ls,
                "burst": (lb + ls) / mu.sum(), "imb": (lb - ls) / (lb + ls), "p0": p0, "m0": m0}
        for h in horizons:
            j = np.searchsorted(t, q + h, "right") - 1
            cols[f"dp_{h}s"] = np.where(j >= 0, px[np.maximum(j, 0)], np.nan) - p0
            cols[f"dm_{h}s"] = _print_mid(t, sg, px, q + h, strict=False) - m0
        parts.append(pl.DataFrame(cols))
    return pl.concat(parts) if parts else pl.DataFrame()


def hist_maker_markouts(tr: pl.DataFrame, windows: dict[str, tuple[int, int]], p2: dict[str, Any],
                        horizons: tuple[int, ...] = (60, 300)) -> pl.DataFrame:
    """Per arrival (a maker was filled at the print): markout to a print mid proxy (average of
    the last taker-buy and last taker-sell print prices) at t + h, with the intensity state
    strictly before the print. ``adv`` = imbalance signed in the taker's direction."""
    mu = np.array([p2["mu_buy"], p2["mu_sell"]])
    nm = np.array([[p2["n_bb"], p2["n_bs"]], [p2["n_sb"], p2["n_ss"]]])
    beta = p2["beta"]
    parts = []
    for (tk,), g in tr.group_by("ticker"):
        if tk not in windows:
            continue
        a, b = windows[tk]
        g = g.sort("created_time")
        t = (g["created_time"].dt.epoch("us").to_numpy() - a) / 1e6
        keep = (t >= 0) & (t < (b - a) / 1e6)
        t, sg, px = t[keep], g["sign"].to_numpy()[keep], g["price"].to_numpy()[keep]
        if len(t) < 20:
            continue
        s = [_excite(t, beta, sg > 0), _excite(t, beta, sg < 0)]
        lb = mu[0] + beta * (nm[0, 0] * s[0] + nm[0, 1] * s[1])
        ls = mu[1] + beta * (nm[1, 0] * s[0] + nm[1, 1] * s[1])
        cols = {"ticker": tk, "t_s": t, "sign": sg.astype(float), "price": px,
                "burst": (lb + ls) / mu.sum(), "adv": sg * (lb - ls) / (lb + ls)}
        tb, pb = t[sg > 0], px[sg > 0]
        ts_, ps = t[sg < 0], px[sg < 0]
        for h in horizons:
            ib = np.searchsorted(tb, t + h, "right") - 1
            is_ = np.searchsorted(ts_, t + h, "right") - 1
            mid = np.where((ib >= 0) & (is_ >= 0),
                           (pb[np.maximum(ib, 0)] if len(pb) else np.nan)
                           + (ps[np.maximum(is_, 0)] if len(ps) else np.nan), np.nan) / 2
            ok = t + h < (b - a) / 1e6
            cols[f"mk_{h}s"] = np.where(ok, -sg * (mid - px), np.nan)
        parts.append(pl.DataFrame(cols))
    return pl.concat(parts) if parts else pl.DataFrame()


def hist_taker(tr: pl.DataFrame, grid: pl.DataFrame, thr: float, *, h: int = 300,
               latency_s: float = 0.1, max_wait_s: float = 60.0, fee_mult: dict[str, float] | None = None,
               windows: dict[str, tuple[int, int]] | None = None) -> pl.DataFrame:
    """Taker proxy on signed intensity imbalance: |imb| > thr buys in its direction at the next
    print where a taker traded that way (>= latency after the signal, the ask proxy), and exits
    ``h`` later at the next print where a taker traded the other way (the bid proxy); taker fee
    both ways. One position per market at a time."""
    rows = []
    sig = grid.filter(pl.col("imb").abs() > thr)
    trg = {k[0]: g.sort("created_time") for k, g in tr.group_by("ticker")}
    for (tk,), g in sig.group_by("ticker"):
        if tk not in trg or windows is None:
            continue
        a, _ = windows[tk]
        x = trg[tk]
        t = (x["created_time"].dt.epoch("us").to_numpy() - a) / 1e6
        sg, px = x["sign"].to_numpy(), x["price"].to_numpy()
        tb, pb, ts_, ps = t[sg > 0], px[sg > 0], t[sg < 0], px[sg < 0]
        busy_until = -1.0
        mult = (fee_mult or {}).get(tk, 1.0)
        for q, imb in zip(g["t_s"].to_numpy(), g["imb"].to_numpy()):
            if q < busy_until:
                continue
            d = 1 if imb > 0 else -1
            et, ep = (tb, pb) if d > 0 else (ts_, ps)
            xt, xp = (ts_, ps) if d > 0 else (tb, pb)
            i = np.searchsorted(et, q + latency_s, "left")
            if i >= len(et) or et[i] > q + latency_s + max_wait_s:
                continue
            j = np.searchsorted(xt, et[i] + h, "left")
            if j >= len(xt) or xt[j] > et[i] + h + 600:
                continue
            entry = ep[i] if d > 0 else 1 - ep[i]        # price of the contract bought
            exitp = xp[j] if d > 0 else 1 - xp[j]
            pnl = exitp - entry - taker_fee(entry, mult) - taker_fee(exitp, mult)
            rows.append({"ticker": tk, "t_s": q, "dir": d, "entry": entry, "exit": exitp,
                         "gross": exitp - entry, "pnl": pnl})
            busy_until = xt[j]
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- study orchestration ------------------------------------------------------------------------------
BASE = {"improve": True, "min_spread": 0.05, "cooldown_s": 300}
YCOLS = ("mo_10s", "mo_60s", "mo_300s", "mx_10s", "mx_60s", "mx_300s")


def _ns(hour: str) -> int:
    from datetime import UTC, datetime

    return int(datetime.strptime(hour, "%Y-%m-%dT%H").replace(tzinfo=UTC).timestamp() * NS)


def l2_fits(tape: dict[str, pl.DataFrame], t0: int, split: int, t1: int) -> pl.DataFrame:
    """Part 1 on the L2 tape: univariate and bivariate trade fits, mid-change fits, dev/holdout."""
    tr, md, win = merged_trades(tape["trades"]), mid_events(tape["quotes"]), tape["windows"]
    rows = []
    for name, (a, b) in (("dev", (t0, split)), ("holdout", (split, t1))):
        rows += fit_table(l2_sequences(tr, win, a, b),
                          label={"data": "l2_trades", "model": "uni", "split": name})
        rows += fit_table(l2_sequences(tr, win, a, b, marks=True), per_market=False, bivariate=True,
                          label={"data": "l2_trades", "model": "bi", "split": name})
        rows += fit_table(l2_sequences(md, win, a, b), min_events=50,
                          label={"data": "l2_mid", "model": "uni", "split": name})
    df = pl.DataFrame(rows, infer_schema_length=None)
    lab = df["label"].str.json_decode(pl.Struct({"data": pl.Utf8, "model": pl.Utf8, "split": pl.Utf8}))
    return df.with_columns(lab.struct.field("data"), lab.struct.field("model"),
                           lab.struct.field("split")).drop("label")


def tox_params(fits: pl.DataFrame) -> ToxParams:
    """Dev, all-markets pooled univariate fits (trades and mid changes) -> online parameters."""
    def pick(data):
        r = fits.filter((pl.col("scope") == "all") & (pl.col("data") == data)
                        & (pl.col("model") == "uni") & (pl.col("split") == "dev")).row(0, named=True)
        return r["mu"], r["n"], r["beta"]

    mu, n, beta = pick("l2_trades")
    mm, nn, bm = pick("l2_mid")
    return ToxParams(mu=mu, n=n, beta=beta, mu_mid=mm, n_mid=nn, beta_mid=bm)


def fit_summary(fits: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per data/model/split/scope: the pooled fit, or the per-market distribution (medians,
    share of markets whose time-rescaling KS test does not reject at 5%)."""
    pooled = fits.filter(pl.col("scope") == "all")
    per = (fits.filter((pl.col("scope") == "market") & pl.col("n").is_not_null())
           .group_by("data", "model", "split")
           .agg(pl.len().alias("markets"), pl.col("events").median().alias("events_med"),
                pl.col("mu").median().alias("mu_med"), pl.col("n").median().alias("n_med"),
                pl.col("n").quantile(0.25).alias("n_p25"), pl.col("n").quantile(0.75).alias("n_p75"),
                pl.col("decay_s").median().alias("decay_s_med"),
                (pl.col("ks_p") > 0.05).mean().alias("ks_pass_share"),
                (pl.col("ks_p_poisson") > 0.05).mean().alias("ks_pass_share_poisson"),
                pl.col("n_at_bound").mean().alias("n_at_bound_share"))
           .sort("data", "model", "split"))
    return pooled, per


def l2_tox(tape: dict[str, pl.DataFrame], p: ToxParams, vol: pl.DataFrame, split: int,
           lo: float = 0.05, hi: float = 0.95) -> pl.DataFrame:
    """Recorded executions (maker price in [lo, hi]) with markouts, toxicity features at
    ``ts < ts_ms`` of the execution, dev and volatile flags."""
    ex = tape["execs"].filter(pl.col("price").is_between(lo, hi))
    ex = exec_markouts(tape, ex)
    f = features(tape, ex.select("market", pl.col("ts_ms").alias("cut_ms"), "side"), p)
    df = pl.concat([ex, f.drop("market", "cut_ms", "side")], how="horizontal")
    bns = 300 * NS
    df = df.with_columns((pl.col("recv_ns") // bns * bns).alias("bucket_ns"),
                         (pl.col("recv_ns") < split).alias("dev"),
                         pl.col("market").map_elements(event_of, return_dtype=pl.Utf8).alias("event"))
    return (df.join(vol.select("market", "bucket_ns", "volatile"), on=["market", "bucket_ns"], how="left")
            .with_columns(pl.col("volatile").fill_null(False)))


def _strategies(tox: OnlineTox, fee_info: dict, specs: dict[str, dict[str, Any]]) -> dict[str, tuple]:
    out = {}
    for name, sp in specs.items():
        sp = dict(sp)
        lat = sp.pop("latency_ms", 50.0)
        cfg = SimConfig(latency_ms=lat, queue="risk_averse", fee_info=fee_info, writes_per_s=10.0)
        out[name] = (ToxQuoter(tox, **(BASE | sp)), cfg)
    return out


def run_specs(s, files: list[Path], p: ToxParams, specs: dict[str, dict[str, Any]], tag: str,
              log=print) -> dict[str, pl.DataFrame]:
    """Run (or load from cache) each spec over ``files``; one replay for all uncached specs."""
    cache = s.data_dir / "cache"
    out, todo = {}, {}
    for name, sp in specs.items():
        path = cache / f"hawkes_fills_{tag}_{name}.parquet"
        meta = cache / f"hawkes_fills_{tag}_{name}.json"
        if path.exists() and meta.exists() and json.loads(meta.read_text()).get("spec") == sp:
            out[name] = pl.read_parquet(path)
        else:
            todo[name] = sp
    if todo:
        fee_info = {r["series_ticker"]: r for r in
                    pl.read_parquet(s.tables_dir / "kalshi_series.parquet").iter_rows(named=True)}
        tox = OnlineTox(p)
        strats = _strategies(tox, fee_info, todo)
        t0 = time.time()
        stats: dict[str, int] = defaultdict(int)
        sims, at_fill = run_many(iter_events(files, stats), strats, tox)
        log(f"replayed {len(todo)} specs in {time.time() - t0:.0f}s")
        for name, sim in sims.items():
            strat = strats[name][0]
            f = fill_frame(sim, at_fill.get(name))
            path = cache / f"hawkes_fills_{tag}_{name}.parquet"
            f.write_parquet(path)
            (cache / f"hawkes_fills_{tag}_{name}.json").write_text(json.dumps(
                {"spec": todo[name], "orders": len(sim.orders), "pulls": strat.pulls,
                 "widens": strat.widens, "vol_blocks": strat.vol_blocks, "diag": dict(sim.diag)}))
            out[name] = f
    return out


def spec_meta(s, tag: str, name: str) -> dict[str, Any]:
    p = s.data_dir / "cache" / f"hawkes_fills_{tag}_{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def variant_specs(base_dev: pl.DataFrame, qs: tuple[float, ...] = (0.5, 0.75, 0.9),
                  vol_cs: tuple[float, ...] = (1.0, 2.0, 3.0)) -> dict[str, dict[str, Any]]:
    """Filter variants with thresholds at dev quantiles of the live measure at baseline dev fills."""
    specs: dict[str, dict[str, Any]] = {}
    for m in ("lam", "lam_rel", "rv60", "ofi60"):
        x = base_dev[f"live_{m}"].drop_nulls()
        for q in qs:
            specs[f"pull_{m}_q{int(q * 100)}"] = {"pull_on": m, "pull_thr": float(x.quantile(q))}
    for m in ("lam", "lam_rel"):
        x = base_dev[f"live_{m}"].drop_nulls()
        for q in qs[1:]:
            specs[f"widen_{m}_q{int(q * 100)}"] = {"widen_on": m, "widen_thr": float(x.quantile(q))}
    for c in vol_cs:
        specs[f"volspread_c{c:g}"] = {"vol_c": c}
    return specs


def score_specs(s, fills: dict[str, pl.DataFrame], vol: pl.DataFrame, split: int, tag: str
                ) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    markets = set()
    for f in fills.values():
        if f.height:
            markets |= set(f["market"].to_list())
    res = market_results(sorted(markets), s.data_dir / "cache" / "hawkes_results.json")
    rows, flagged = [], {}
    for name, f in fills.items():
        if f.height == 0:
            rows.append({"variant": name, "fills": 0})
            continue
        g = with_flags(f, vol, split, res)
        flagged[name] = g
        meta = spec_meta(s, tag, name)
        for r in score_frame(g):
            rows.append({"variant": name, "orders": meta.get("orders"), "pulls": meta.get("pulls"),
                         "settled": int(g.filter(pl.col("settled") & (pl.col("dev") == (r["split"] == "dev")))
                                        .height)} | r)
    return pl.DataFrame(rows, infer_schema_length=None), flagged


def diff_vs_base(flagged: dict[str, pl.DataFrame], base: str, cols: tuple[str, ...] = ("mx_300s", "pnl"),
                 n_boot: int = 2000, seed: int = 0) -> pl.DataFrame:
    """Variant minus baseline mean per contract (c) per regime and split, with an SE from a
    bootstrap over events (the union of both runs' events is resampled, so fills the two runs
    share move together)."""
    rng = np.random.default_rng(seed)
    rows = []
    b = flagged[base]
    for name, f in flagged.items():
        if name == base or f.height == 0:
            continue
        for regime in ("all", "volatile", "calm"):
            for split in ("dev", "holdout"):
                def sub(x):
                    x = x.filter(pl.col("dev") if split == "dev" else ~pl.col("dev"))
                    return x if regime == "all" else x.filter(pl.col("volatile") if regime == "volatile" else ~pl.col("volatile"))
                fv, fb = sub(f), sub(b)
                r: dict[str, Any] = {"variant": name, "regime": regime, "split": split,
                                     "fills": fv.height, "base_fills": fb.height}
                evs = sorted(set(fv["event"].to_list()) | set(fb["event"].to_list()))
                if not evs or fv.height == 0 or fb.height == 0:
                    rows.append(r)
                    continue
                idx = {e: i for i, e in enumerate(evs)}
                for c in cols:
                    sums = []
                    for x in (fv, fb):
                        y = x[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
                        g = np.array([idx[e] for e in x["event"].to_list()])
                        ok = ~np.isnan(y)
                        sums.append((np.bincount(g[ok], weights=y[ok], minlength=len(evs)),
                                     np.bincount(g[ok], minlength=len(evs)).astype(float)))
                    d0 = sums[0][0].sum() / max(sums[0][1].sum(), 1) - sums[1][0].sum() / max(sums[1][1].sum(), 1)
                    w = rng.multinomial(len(evs), np.full(len(evs), 1 / len(evs)), size=n_boot).astype(float)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        mv = (w @ sums[0][0]) / (w @ sums[0][1])
                        mb = (w @ sums[1][0]) / (w @ sums[1][1])
                    se = float(np.nanstd(mv - mb))
                    r[f"d_{c}"] = round(100 * d0, 2)
                    r[f"t_d_{c}"] = round(d0 / se, 2) if se > 0 else None
                rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None)


def choose_on_dev(table: pl.DataFrame, base: str, min_keep: float = 0.4, top: int = 3) -> list[str]:
    """Best dev variants by exit markout at 5 min (all windows), keeping >= ``min_keep`` of the
    baseline's dev fills."""
    dev = table.filter((pl.col("regime") == "all") & (pl.col("split") == "dev"))
    nb = dev.filter(pl.col("variant") == base)["fills"][0]
    cand = dev.filter((pl.col("fills") >= min_keep * nb) & pl.col("mx_300s").is_not_null()
                      & (pl.col("variant") != base) & ~pl.col("variant").str.starts_with("base"))
    return cand.sort("mx_300s", descending=True)["variant"].head(top).to_list()


def hist_study(s, *, split: str = "2026-10-01", cap: int = 150, seed: int = 0,
               log=print) -> dict[str, pl.DataFrame]:
    """Part 4 on ``kalshi_trades``: pre-game windows (exp - 25 h .. exp - 4 h, one leg per game,
    up to ``cap`` games per series and split, volume-blind) and the in-play sample (markets whose
    trades reach the close; [exp - 3 h, min(close, exp + 1 h)])."""
    from datetime import UTC, datetime

    cache = s.data_dir / "cache"
    names = ("coverage", "fits", "grid", "maker", "taker", "params_pregame", "params_inplay", "inplay_split")
    tag = f"hawkes_hist_{split}_cap{cap}_seed{seed}"
    done = cache / f"{tag}.done"
    if done.exists():
        return {n: pl.read_parquet(cache / f"{tag}_{n}.parquet") for n in names
                if (cache / f"{tag}_{n}.parquet").exists()}
    split_dt = datetime.fromisoformat(split).replace(tzinfo=UTC)
    cov = (hist_coverage(s).filter(pl.col("expected_expiration_time").is_not_null())
           .with_columns((pl.col("expected_expiration_time") < split_dt).alias("dev")))
    us = lambda c: pl.col(c).dt.epoch("us")  # noqa: E731
    h = 3600 * 10**6
    legs = cov.sort("ticker").group_by("event_ticker", maintain_order=True).first()
    pre_pool = legs.with_columns((us("expected_expiration_time") - 25 * h).alias("a"),
                                 (us("expected_expiration_time") - 4 * h).alias("b"))
    parts = []
    for (ser, dev), g in pre_pool.group_by("series_ticker", "dev"):
        parts.append(g.sample(n=min(cap, g.height), seed=seed))
    pre = pl.concat(parts)
    inp = cov.filter(pl.col("to_close") & (pl.col("inplay_trades") >= 500)).with_columns(
        (us("expected_expiration_time") - int(INPLAY_ONSET_H * h)).alias("a"),
        pl.min_horizontal(us("close_time"), us("expected_expiration_time") + h).alias("b"))
    out: dict[str, pl.DataFrame] = {"coverage": cov.group_by("series_ticker", "dev").agg(
        pl.len().alias("markets"), pl.col("trades").sum(), pl.col("to_close").sum(),
        (pl.col("to_close") & (pl.col("inplay_trades") >= 500)).sum().alias("inplay_markets"),
        pl.col("inplay_trades").sum()).sort("series_ticker", "dev")}
    fit_rows, grid_rows, mk_rows, tk_rows = [], [], [], []
    fee_mult = dict(zip(cov["ticker"].to_list(), cov["fee_multiplier"].fill_null(1.0).to_list()))
    for kind, sample in (("pregame", pre), ("inplay", inp)):
        tr = hist_trades(s, sample["ticker"].to_list())
        game = dict(zip(sample["ticker"].to_list(), sample["event_ticker"].to_list()))
        w = {r["ticker"]: (r["a"], r["b"]) for r in sample.iter_rows(named=True)}
        dev_t = set(sample.filter("dev")["ticker"].to_list())
        if kind == "inplay" and sample.filter("dev")["event_ticker"].n_unique() < 3:
            # no in-play games before the split: split in-play games by date instead (earlier half
            # of the games = dev), still strictly out of sample in time
            ev = sample.group_by("event_ticker").agg(pl.col("expected_expiration_time").min()).sort("expected_expiration_time")
            first = set(ev["event_ticker"].head(ev.height // 2).to_list())
            dev_t = set(sample.filter(pl.col("event_ticker").is_in(list(first)))["ticker"].to_list())
            log(f"hist inplay: {ev.height} games, none before {split}; dev = first {len(first)} games by date "
                f"(through {ev['expected_expiration_time'][len(first) - 1]})")
            out["inplay_split"] = pl.DataFrame({"dev_games": [len(first)], "holdout_games": [ev.height - len(first)],
                                                "dev_through": [ev["expected_expiration_time"][len(first) - 1]]})
        p2_dev = None
        for split_name, keep in (("dev", True), ("holdout", False)):
            ws = {k: v for k, v in w.items() if (k in dev_t) == keep}
            if not ws:
                continue
            seqs = hist_sequences(tr, ws)
            seqs2 = hist_sequences(tr, ws, marks=True)
            lab = {"data": kind, "split": split_name}
            fit_rows += fit_table(seqs, label=lab | {"model": "uni"}, min_events=300)
            bi = fit_table(seqs2, label=lab | {"model": "bi"}, per_market=False, bivariate=True)
            fit_rows += bi
            if split_name == "dev":
                p2_dev = next((r for r in bi if r["scope"] == "all" and r.get("beta")), None)
            log(f"hist {kind} {split_name}: {len(seqs)} markets, {sum(len(v[0]) for v in seqs.values())} arrivals")
        if p2_dev is None:
            log(f"hist {kind}: no dev bivariate fit; skipping intensity tests")
            continue
        out[f"params_{kind}"] = pl.DataFrame([{k: v for k, v in p2_dev.items() if not isinstance(v, dict)}])
        step = 60.0 if kind == "pregame" else 10.0
        grid = hist_grid(tr, w, p2_dev, step_s=step)
        mk = hist_maker_markouts(tr, w, p2_dev)
        for frame, rows in ((grid, grid_rows), (mk, mk_rows)):
            if frame.is_empty():
                continue
            frame = frame.with_columns(
                pl.col("ticker").is_in(list(dev_t)).alias("dev"),
                pl.col("ticker").replace_strict(game, default=None).alias("game"))
            rows.append(frame.with_columns(pl.lit(kind).alias("data")))
        # taker proxy on the imbalance: thresholds from dev grid points
        gd = grid.filter(pl.col("ticker").is_in(list(dev_t)))
        if gd.height:
            for q in (0.9, 0.97):
                thr = float(gd["imb"].abs().quantile(q))
                tk = hist_taker(tr, grid, thr, fee_mult=fee_mult, windows=w)
                if tk.height:
                    tk_rows.append(tk.with_columns(pl.lit(kind).alias("data"), pl.lit(q).alias("q"),
                                                   pl.lit(thr).alias("thr"),
                                                   pl.col("ticker").is_in(list(dev_t)).alias("dev"),
                                                   pl.col("ticker").replace_strict(game, default=None).alias("game")))
    fits = pl.DataFrame(fit_rows, infer_schema_length=None)
    lab = fits["label"].str.json_decode(pl.Struct({"data": pl.Utf8, "model": pl.Utf8, "split": pl.Utf8}))
    out["fits"] = fits.with_columns(lab.struct.field("data"), lab.struct.field("model"),
                                    lab.struct.field("split")).drop("label")
    out["grid"] = pl.concat(grid_rows) if grid_rows else pl.DataFrame()
    out["maker"] = pl.concat(mk_rows) if mk_rows else pl.DataFrame()
    out["taker"] = pl.concat(tk_rows) if tk_rows else pl.DataFrame()
    cache.mkdir(parents=True, exist_ok=True)
    for n, v in out.items():
        v.write_parquet(cache / f"{tag}_{n}.parquet")
    done.write_text("ok")
    return out


def hist_tests(grid: pl.DataFrame, maker: pl.DataFrame, taker: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Direction vs magnitude on the grid, maker markouts vs prior intensity, taker proxy PnL;
    z-scores use dev moments of each data kind; t-stats clustered by game."""
    rows, mrows, trows = [], [], []
    for kind in grid["data"].unique().to_list() if grid.height else []:
        g = grid.filter((pl.col("data") == kind) & pl.col("p0").is_not_null())
        dev = g["dev"].to_numpy()
        lb = np.log(g["burst"].to_numpy())
        imb = g["imb"].to_numpy()
        zb = (lb - lb[dev].mean()) / lb[dev].std()
        zi = (imb - imb[dev].mean()) / imb[dev].std()
        hb = lb > np.quantile(lb[dev], 0.9)
        for hcol in [c for c in g.columns if c.startswith(("dp_", "dm_"))]:
            dp = 100 * g[hcol].to_numpy()
            for split in ("dev", "holdout"):
                sel = dev if split == "dev" else ~dev
                for test, x, y, sub in (("magnitude: |dp| ~ z(log burst)", zb, np.abs(dp), None),
                                        ("direction: dp ~ z(imbalance)", zi, dp, None),
                                        ("direction in bursts (top 10% burst)", zi, dp, hb)):
                    s2 = sel if sub is None else sel & sub
                    b, t, n, k = cl_ols(x[s2], y[s2], g["game"].to_numpy()[s2])
                    rows.append({"data": kind, "h": hcol, "split": split, "test": test, "n": n,
                                 "games": k, "slope_c_per_sd": b, "t": t,
                                 "mean_abs_dp_c": float(np.nanmean(np.abs(dp[s2]))) if s2.any() else None})
    for kind in maker["data"].unique().to_list() if maker.height else []:
        g = maker.filter(pl.col("data") == kind)
        dev = g["dev"].to_numpy()
        lb = np.log(g["burst"].to_numpy())
        zb = (lb - lb[dev].mean()) / lb[dev].std()
        adv = g["adv"].to_numpy()
        za = (adv - adv[dev].mean()) / adv[dev].std()
        for hcol in [c for c in g.columns if c.startswith("mk_")]:
            y = 100 * g[hcol].to_numpy()
            for split in ("dev", "holdout"):
                sel = dev if split == "dev" else ~dev
                m, tm, k = cl_mean(y[sel], g["game"].to_numpy()[sel])
                for name, x in (("z(log burst)", zb), ("z(adverse imbalance)", za)):
                    b, t, n, _ = cl_ols(x[sel], y[sel], g["game"].to_numpy()[sel])
                    mrows.append({"data": kind, "h": hcol, "split": split, "x": name, "n": n,
                                  "games": k, "mean_markout_c": m, "t_mean": tm,
                                  "slope_c_per_sd": b, "t": t})
    if taker.height:
        for (kind, q, split), g in taker.group_by("data", "q", "dev"):
            m, t, k = cl_mean(100 * g["pnl"].to_numpy(), g["game"].to_numpy())
            mg, tg, _ = cl_mean(100 * g["gross"].to_numpy(), g["game"].to_numpy())
            trows.append({"data": kind, "q": q, "thr": g["thr"][0], "split": "dev" if split else "holdout",
                          "trades": g.height, "games": k, "gross_c": mg, "t_gross": tg,
                          "net_c": m, "t_net": t})
    return {"direction_magnitude": pl.DataFrame(rows, infer_schema_length=None),
            "maker": pl.DataFrame(mrows, infer_schema_length=None),
            "taker": pl.DataFrame(trows, infer_schema_length=None).sort("data", "q", "split") if trows else pl.DataFrame()}


def _r(df: pl.DataFrame, digits: int = 3) -> pl.DataFrame:
    return df.with_columns([pl.col(c).round(digits) for c, t in df.schema.items() if t == pl.Float64])


def hawkes_study(s, *, start: str = "2026-11-05T03", end: str = "2026-11-05T18",
                 split: str = "2026-11-05T11", hist: bool = True, hist_cap: int = 60,
                 root: Path | None = None, log=print) -> Path:
    """Run all four parts and write report ``research-hawkes``. ``start``/``end``/``split`` are
    ``YYYY-MM-DDTHH`` hours (dev = [start, split), holdout = [split, end)); every expensive step
    is cached under ``data/cache/hawkes_*`` keyed by the window. Rerun on tonight's in-play hours
    with e.g. ``hawkes_study(settings(), start="2026-11-05T23", end="2026-11-06T07",
    split="2026-11-06T03", hist=False)``."""
    from albert.reports import Report
    from albert.research.volatile import cached_buckets

    root = root or s.raw_dir / "stream"
    tag = f"{start}_{end}"
    t0, t_split, t1 = _ns(start), _ns(split), _ns(end)
    files = stream_files(root, start, end)
    if not files:
        raise RuntimeError("no recorded Kalshi stream files in range")
    tape = cached_tape(s, start, end, root)
    fits_path = s.data_dir / "cache" / f"hawkes_l2fits_{tag}.parquet"
    if fits_path.exists():
        fits = pl.read_parquet(fits_path)
    else:
        fits = l2_fits(tape, t0, t_split, t1)
        fits.write_parquet(fits_path)
    p = tox_params(fits)
    vol = cached_buckets(s, start, end, root=root)
    rep = Report(s.reports_dir, "research-hawkes")
    rep.json("Run", {"start": start, "end": end, "split": split, "files": len(files),
                     "markets": tape["windows"].height, "trade_prints": tape["trades"].height,
                     "arrivals": merged_trades(tape["trades"]).height,
                     "quote_changes": tape["quotes"].height, "executions": tape["execs"].height,
                     "volatile_buckets": int(vol["volatile"].sum()), "buckets": vol.height,
                     "online_params": p.__dict__})
    # part 1
    pooled, per = fit_summary(fits)
    cols = ["data", "model", "split", "markets", "events", "events_buy", "events_sell", "mu", "n",
            "decay_s", "ll_gain_per_event", "ks_D", "ks_p", "ks_D_poisson", "n_bb", "n_bs", "n_sb",
            "n_ss", "rho", "n_at_bound"]
    rep.table("Part 1. Hawkes fits pooled over all markets (exponential kernel, MLE)",
              _r(pooled.select([c for c in cols if c in pooled.columns]), 5), save_as="l2_fits")
    rep.table("Part 1. Per-market fits (markets with >= 30 trade arrivals / 50 mid changes)", _r(per))
    ser = (fits.filter((pl.col("scope") == "series") & (pl.col("model") == "uni"))
           .sort("events", descending=True)
           .select("data", "split", "key", "markets", "events", "mu", "n", "decay_s", "ks_D", "ks_p",
                   "ks_D_poisson"))
    rep.table("Part 1. Pooled by series (largest)", _r(ser, 5), max_rows=30)
    # part 2: recorded executions
    ex = l2_tox(tape, p, vol, t_split)
    reg = pl.concat([tox_regressions(ex, ycols=YCOLS), tox_regressions(ex, ycols=YCOLS, control="spread")])
    rep.table("Part 2. Recorded executions (maker price 5-95c): counts and mean markouts (c)",
              _r(ex.group_by("dev", "volatile").agg(
                  pl.len().alias("executions"), pl.col("event").n_unique().alias("events"),
                  *[(100 * pl.col(c)).mean().alias(c) for c in YCOLS],
                  pl.col("cleared").mean().alias("cleared_share")).sort("dev", "volatile"), 2))
    key = reg.filter((pl.col("control") == "spread") & pl.col("y").is_in(["mx_60s", "mx_300s"])
                     & pl.col("measure").is_in(["lam", "lam_rel", "lam_mid", "rv60", "abs_ofi60",
                                                "ofi_adv60", "vpin"]))
    rep.table("Part 2. Recorded executions: exit markout on toxicity at the fill (spread-bin fixed "
              "effects; slope = c per dev SD; high = above dev 80th pct; t by event)",
              _r(key.drop("control"), 2), save_as="exec_regressions", max_rows=90)
    reg.write_parquet(rep.dir / "exec_regressions_all.parquet")
    # part 3: quote filters
    base = run_specs(s, files, p, {"base_50": {"latency_ms": 50.0}, "base_100": {"latency_ms": 100.0}}, tag, log)
    b = base["base_50"]
    specs = variant_specs(b.filter(pl.col("t_ns") < t_split))
    var = run_specs(s, files, p, {k: v | {"latency_ms": 50.0} for k, v in specs.items()}, tag, log)
    table, flagged = score_specs(s, base | var, vol, t_split, tag)
    picks = choose_on_dev(table, "base_50")
    r100 = run_specs(s, files, p, {f"{k}_100": specs[k] | {"latency_ms": 100.0} for k in picks}, tag, log)
    table, flagged = score_specs(s, base | var | r100, vol, t_split, tag)
    if "base_50" in flagged and flagged["base_50"].height:
        bf = flagged["base_50"]
        fq = features(tape, bf.select("market", pl.col("ts_ms").alias("cut_ms"), "side"), p)
        bfx = pl.concat([bf, fq.drop("market", "cut_ms", "side")], how="horizontal")
        breg = tox_regressions(bfx, ycols=YCOLS, control="spread")
        rep.table("Part 2. Baseline Quoter fills (50 ms): markout on toxicity at the fill (spread FE)",
                  _r(breg.filter(pl.col("y").is_in(["mo_300s", "mx_60s", "mx_300s"])
                                 & (pl.col("regime") == "all")).drop("control"), 2),
                  save_as="quoter_regressions", max_rows=60)
    rep.json("Part 3. Thresholds (dev quantiles of the live measure at baseline dev fills)", specs)
    rep.text("Part 3. Dev picks (best dev mx_300s keeping >= 40% of dev fills), rerun at 100 ms",
             ", ".join(picks) or "none")
    show = ["variant", "regime", "split", "fills", "events", "orders", "pulls", "mo_300s", "t_mo_300s",
            "mx_60s", "t_mx_60s", "mx_300s", "t_mx_300s", "mxn_300s", "t_mxn_300s", "pnl", "t_pnl",
            "pnl_usd", "settled"]
    rep.table("Part 3. Quote filters on the Simulator (c per contract; t by event; volatile = the "
              "fill's own 5-min bucket, an evaluation split)",
              table.filter(pl.col("regime") == "all").select([c for c in show if c in table.columns]),
              save_as="variants", max_rows=80)
    if "base_50" in flagged:
        d50 = diff_vs_base({k: v for k, v in flagged.items() if not k.endswith("_100") and k != "base_100"}, "base_50")
        d100 = diff_vs_base({k: v for k, v in flagged.items() if k.endswith("_100")}, "base_100")
        rep.table("Part 3. Variant minus baseline (same latency), c per contract, t from an event "
                  "bootstrap", pl.concat([d50, d100], how="diagonal"), save_as="variant_diffs", max_rows=200)
    rep.table("Part 3. Volatile vs calm windows",
              table.filter(pl.col("regime") != "all").select([c for c in show if c in table.columns]),
              max_rows=200)
    if hist:
        h = hist_study(s, cap=hist_cap, log=log)
        tests = hist_tests(h["grid"], h["maker"], h["taker"])
        rep.table("Part 4. Historical coverage (game markets; in-play = trades reach the close)", h["coverage"])
        rep.text("Part 4. Samples", "Pre-game: one leg per game, exp - 25 h .. exp - 4 h (the ingested maker "
                 f"window), up to {hist_cap} games per series and split, dev = before 2026-10-01. In-play: every "
                 "leg whose trades reach the close, [exp - 3 h, min(close, exp + 1 h)] (onset verified: the "
                 "trade rate doubles between exp - 3.25 h and exp - 3 h). Mid proxy = mean of the last taker-buy "
                 "and last taker-sell print (removes the bid-ask bounce of the last print, `dp_*`).")
        if "inplay_split" in h:
            rep.table("Part 4. In-play split (no in-play games before October: split by game date)", h["inplay_split"])
        hf = h["fits"].filter(pl.col("scope") != "market")
        hcols = ["data", "model", "split", "scope", "key", "markets", "events", "events_buy",
                 "events_sell", "mu", "n", "decay_s", "ks_D", "ks_p", "ks_D_poisson", "n_bb", "n_bs",
                 "n_sb", "n_ss", "rho", "n_at_bound"]
        rep.table("Part 4. Historical Hawkes fits (pooled)", _r(hf.select([c for c in hcols if c in hf.columns]), 5),
                  save_as="hist_fits", max_rows=60)
        hm = (h["fits"].filter((pl.col("scope") == "market") & pl.col("n").is_not_null())
              .group_by("data", "split").agg(pl.len().alias("markets"), pl.col("events").median().alias("events_med"),
                                             pl.col("n").median().alias("n_med"), pl.col("decay_s").median().alias("decay_s_med"),
                                             (pl.col("ks_p") > 0.05).mean().alias("ks_pass_share"),
                                             (pl.col("ks_p_poisson") > 0.05).mean().alias("ks_pass_share_poisson")))
        rep.table("Part 4. Historical per-market fits (>= 300 arrivals)", _r(hm))
        rep.table("Part 4. Do intensity bursts predict direction or magnitude of print moves? (c; t by game)",
                  _r(tests["direction_magnitude"], 3), save_as="hist_direction", max_rows=60)
        rep.table("Part 4. Maker print markouts (to a print mid proxy) vs prior intensity (c; t by game)",
                  _r(tests["maker"], 3), save_as="hist_maker", max_rows=60)
        if tests["taker"].height:
            rep.table("Part 4. Taker proxy on intensity imbalance (next same-side print in, next "
                      "opposite print out, 300 s, taker fees; c per contract)", _r(tests["taker"], 3))
    return rep.save()
