"""Information-based market making for binary contracts: PIN, Kyle's lambda, Glosten–Milgrom
zero-profit spreads and a Das (2005)-style Bayesian market maker, scored on the L2 replay and on
the historical trade tape.

**PIN** (Easley, Kiefer, O'Hara & Paperman 1996). Each period (here a 5-minute bucket of one
market) has an information event with probability ``alpha``; it is bad news with probability
``delta``. Uninformed takers buy at rate ``eb`` and sell at ``es``; informed takers add ``mu`` on
the news side. ``PIN = alpha*mu / (alpha*mu + eb + es)`` is the probability that a taker order is
informed. The likelihood is evaluated in the numerically stable factorisation of Lin & Ke (2011)
(``pin_nll``), maximised from a small grid of starts (Yan & Zhang 2012 style). Taker *orders*,
not prints, are counted: prints of one ticker sharing timestamp and taker side are one order.
``vpin`` is the raw moment version ``sum|B-S| / sum(B+S)`` (biased towards 1 when buckets hold
few orders); ``pin_mom`` subtracts the imbalance fair coin flips would produce and is used where
an MLE per decision is too expensive (rolling / expanding, look-ahead free).

**Kyle's lambda**: OLS slope of the 1-minute mid change on signed taker contracts (cents per 100
contracts). **Sign autocorrelation**: lag-1 autocorrelation of taker-order signs.

**Glosten–Milgrom** for a binary asset. With prior ``p`` and informed fraction ``pi``:

* terminal (``gm_binary``): informed traders know the outcome V in {0, 1};
  ``ask = p(1+pi)/(1+pi(2p-1))``, ``bid = p(1-pi)/(1-pi(2p-1))``;
* short horizon (``gm_half_two_point``): informed traders know the next move, V = p +- Delta
  with Delta the typical mid move over the markout horizon; zero-profit half-spread ``pi*Delta``.

A 1-lot maker should earn where the half-spread it captures (mid minus its bid) exceeds the GM
half-spread plus the maker fee, and lose where it does not (``excess`` columns).

**Das MM** (``DasMM``): a discretised belief over the true value (0..1 by 1c) per market, reset
towards the current mid on each trade (``reset``), updated on every public taker trade with the
informed-trader likelihood ``alpha*Phi((v - price)/sigma) + (1 - alpha)/2`` (buy; mirrored for a
sell). Quotes are the zero-profit conditional expectations ``bid* = max b: E[V | sell at b] - b >=
fee + margin`` and ``ask* = min a: a - E[V | buy at a] >= fee + margin``. It is a ``Quoter``
(1-lot, cooldown, position cap, write budget, post-only via the Simulator) that improves the
best by one tick on a side only when that price is inside the Bayesian zero-profit quote, so it
never joins or crosses.

Look-ahead rules: everything a strategy or rule uses is built from data before its decision.
``*_prev`` columns use the market's previous buckets only; PIN fitted on the dev window is used
on holdout. Evaluation splits (volatile / calm) use each fill's own bucket and say so.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import gammaln, ndtr

from albert.backtest.l2 import (EPS, Event, Quoter, SimConfig, Simulator, iter_events, markouts,
                                stream_files)

TICK = 0.01
MAKER_RATE = 0.0175
BUCKET_S = 300


# ---- PIN (EKOP) ---------------------------------------------------------------------------------
def _unpack(x: np.ndarray) -> tuple[float, float, float, float, float]:
    a = 1.0 / (1.0 + math.exp(-x[0]))
    d = 1.0 / (1.0 + math.exp(-x[1]))
    return a, d, math.exp(x[2]), math.exp(x[3]), math.exp(x[4])


def pin_nll(x: np.ndarray, B: np.ndarray, S: np.ndarray, lgf: np.ndarray) -> float:
    """Negative EKOP log-likelihood in the Lin & Ke (2011) form. ``x`` = (logit alpha, logit
    delta, log mu, log eb, log es); ``lgf`` = log(B!) + log(S!)."""
    a, d, mu, eb, es = _unpack(x)
    lb, ls = math.log1p(mu / eb), math.log1p(mu / es)
    e1 = -mu - S * ls            # good news: buys elevated
    e2 = -mu - B * lb            # bad news: sells elevated
    e3 = -B * lb - S * ls        # no news
    em = np.maximum(np.maximum(e1, e2), e3)
    inner = a * (1 - d) * np.exp(e1 - em) + a * d * np.exp(e2 - em) + (1 - a) * np.exp(e3 - em)
    ll = np.log(inner) + B * math.log(eb + mu) + S * math.log(es + mu) - (eb + es) + em - lgf
    return float(-ll.sum())


def fit_pin(B: Iterable[float], S: Iterable[float], *, min_buckets: int = 12,
            min_orders: int = 30) -> dict[str, float] | None:
    """EKOP maximum likelihood on per-period buy/sell order counts. None when the sample is too
    small (fewer than ``min_buckets`` periods or ``min_orders`` orders)."""
    B = np.asarray(list(B), dtype=float)
    S = np.asarray(list(S), dtype=float)
    if len(B) < min_buckets or B.sum() + S.sum() < min_orders:
        return None
    lgf = gammaln(B + 1) + gammaln(S + 1)
    mb, ms = max(B.mean(), 1e-3), max(S.mean(), 1e-3)
    best = None
    for a in (0.15, 0.5, 0.85):
        for d in (0.25, 0.75):
            for g in (0.5,):
                eb = g * mb
                mu = max((mb - eb) / (a * (1 - d)), 1e-3)
                es = ms - a * d * mu
                if es <= 0:
                    es = g * ms
                x0 = np.array([math.log(a / (1 - a)), math.log(d / (1 - d)), math.log(mu),
                               math.log(max(eb, 1e-4)), math.log(max(es, 1e-4))])
                r = minimize(pin_nll, x0, args=(B, S, lgf), method="L-BFGS-B",
                             bounds=[(-10, 10), (-10, 10), (-9, 9), (-9, 9), (-9, 9)])
                if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                    best = r
    if best is None:
        return None
    a, d, mu, eb, es = _unpack(best.x)
    return {"alpha": a, "delta": d, "mu": mu, "eb": eb, "es": es,
            "pin": a * mu / (a * mu + eb + es), "ll": -best.fun, "buckets": float(len(B)),
            "orders": float(B.sum() + S.sum()), "vpin": vpin(B, S), "pin_mom": pin_mom(B, S)}


def vpin(B: Iterable[float], S: Iterable[float]) -> float | None:
    """Volume-synchronised style estimate sum |B - S| / sum (B + S). Biased towards 1 when
    buckets hold few orders (a single order is always fully one-sided): see ``pin_mom``."""
    B = np.asarray(list(B), dtype=float)
    S = np.asarray(list(S), dtype=float)
    tot = (B + S).sum()
    return float(np.abs(B - S).sum() / tot) if tot > 0 else None


def _noise_abs_imbalance(n: np.ndarray) -> np.ndarray:
    """E|B - S| when n orders are independent fair coin flips (no information), exact for
    n <= 400 and sqrt(2n/pi) beyond."""
    n = np.asarray(n, dtype=np.int64)
    out = np.sqrt(2 * np.maximum(n, 0) / math.pi)
    small = n <= 400
    if small.any():
        table = _NOISE_TABLE
        out[small] = table[n[small]]
    return out


def _noise_table(nmax: int = 400) -> np.ndarray:
    from scipy.stats import binom

    t = np.zeros(nmax + 1)
    for n in range(1, nmax + 1):
        k = np.arange(n + 1)
        t[n] = float((binom.pmf(k, n, 0.5) * np.abs(2 * k - n)).sum())
    return t


_NOISE_TABLE = _noise_table()


def noise_expr(n: pl.Expr) -> pl.Expr:
    """Polars version of ``_noise_abs_imbalance`` for an order-count column."""
    tab = {i: float(v) for i, v in enumerate(_NOISE_TABLE)}
    n = n.cast(pl.Int64)
    return (pl.when(n <= 400).then(n.replace_strict(tab, default=0.0, return_dtype=pl.Float64))
            .otherwise((2 * n.cast(pl.Float64) / math.pi).sqrt()))


def pin_mom(B: Iterable[float], S: Iterable[float]) -> float | None:
    """Bias-adjusted moment estimate of the informed share: the order imbalance in excess of
    what fair coin flips would produce, sum(|B - S| - E0|B - S| given B + S) / sum(B + S),
    floored at 0. Zero in expectation without informed flow at any bucket size."""
    B = np.asarray(list(B), dtype=float)
    S = np.asarray(list(S), dtype=float)
    tot = (B + S).sum()
    if tot <= 0:
        return None
    ex = np.abs(B - S) - _noise_abs_imbalance((B + S).astype(np.int64))
    return max(0.0, float(ex.sum() / tot))


# ---- Glosten–Milgrom ----------------------------------------------------------------------------
def gm_binary(p: float, pi: float) -> tuple[float, float]:
    """(bid, ask) zero-profit quotes when a fraction ``pi`` of takers know the outcome."""
    k = pi * (2 * p - 1)
    return p * (1 - pi) / (1 - k), p * (1 + pi) / (1 + k)


def gm_half_binary(p, pi):
    """Half of the terminal GM spread (works on floats and numpy arrays)."""
    k = pi * (2 * p - 1)
    return (p * (1 + pi) / (1 + k) - p * (1 - pi) / (1 - k)) / 2


def gm_half_two_point(pi, delta):
    """Zero-profit half-spread when informed takers know the next move of size ``delta``."""
    return pi * delta


def maker_fee_rate(fee_type: str | None, multiplier: float | None) -> float:
    return MAKER_RATE * (multiplier or 1.0) if fee_type and "maker_fee" in fee_type else 0.0


# ---- price impact and order-flow persistence -----------------------------------------------------
def kyle_lambda(dp: Iterable[float], q: Iterable[float]) -> dict[str, float | None]:
    """OLS slope (with intercept) of price changes on signed volume, its t-stat and n."""
    dp = np.asarray(list(dp), dtype=float)
    q = np.asarray(list(q), dtype=float)
    n = len(dp)
    if n < 10 or np.var(q) <= 0:
        return {"lam": None, "t_lam": None, "n": n}
    qc, dc = q - q.mean(), dp - dp.mean()
    lam = float((qc * dc).sum() / (qc * qc).sum())
    resid = dc - lam * qc
    se = math.sqrt((resid * resid).sum() / (n - 2) / (qc * qc).sum())
    return {"lam": lam, "t_lam": lam / se if se > 0 else None, "n": n}


def sign_autocorr(signs: Iterable[float]) -> float | None:
    s = np.asarray(list(signs), dtype=float)
    if len(s) < 10 or s[:-1].std() == 0 or s[1:].std() == 0:
        return None
    return float(np.corrcoef(s[:-1], s[1:])[0, 1])


# ---- generic counting / estimation -------------------------------------------------------------
def bucket_counts(orders: pl.DataFrame, unit: list[str], *, bucket_col: str = "bucket") -> pl.DataFrame:
    """Buy/sell taker-order counts per unit and integer bucket, with the empty buckets between
    a unit's first and last trade filled with zeros. ``orders`` needs ``unit``, ``bucket_col``
    (int) and ``buy`` (bool)."""
    c = orders.group_by(unit + [bucket_col]).agg(pl.col("buy").sum().cast(pl.Float64).alias("B"),
                                                 (~pl.col("buy")).sum().cast(pl.Float64).alias("S"))
    rng = c.group_by(unit).agg(pl.col(bucket_col).min().alias("_b0"), pl.col(bucket_col).max().alias("_b1"))
    full = rng.with_columns(pl.int_ranges("_b0", pl.col("_b1") + 1).alias(bucket_col)).explode(bucket_col)
    return (full.drop("_b0", "_b1").join(c, on=unit + [bucket_col], how="left")
            .with_columns(pl.col("B").fill_null(0.0), pl.col("S").fill_null(0.0))
            .sort(unit + [bucket_col]))


def fit_units(counts: pl.DataFrame, unit: list[str], **kw: Any) -> pl.DataFrame:
    """``fit_pin`` per unit of ``bucket_counts`` output (units that fail the size filters are
    dropped)."""
    rows = []
    for key, g in counts.group_by(unit, maintain_order=True):
        r = fit_pin(g["B"].to_numpy(), g["S"].to_numpy(), **kw)
        if r is not None:
            rows.append(dict(zip(unit, key)) | r)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def clustered(rows: pl.DataFrame, col: str, key: str = "event",
              w: str = "qty") -> tuple[float | None, float | None, int]:
    """Weighted mean of ``col`` and its t-stat clustered by ``key``."""
    f = rows.filter(pl.col(col).is_not_null() & pl.col(col).is_not_nan())
    if f.is_empty():
        return None, None, 0
    g = f.group_by(key).agg((pl.col(col) * pl.col(w)).sum().alias("s"), pl.col(w).sum().alias("n"))
    n = g["n"].sum()
    if n <= 0:
        return None, None, 0
    mean = g["s"].sum() / n
    k = g.height
    if k < 2:
        return mean, None, k
    se = math.sqrt(k / (k - 1) * ((g["s"] - mean * g["n"]) ** 2).sum()) / n
    return mean, (mean / se if se > 0 else None), k


def split_table(rows: pl.DataFrame, cols: list[str], *, groups: dict[str, pl.Expr] | None = None,
                key: str = "event", w: str = "qty") -> pl.DataFrame:
    """all / volatile / calm x dev / holdout (x optional ``groups``) table of clustered means in
    cents. ``rows`` needs ``window`` (dev/holdout), ``volatile`` (bool, the fill's own bucket)."""
    out = []
    regimes = {"all": pl.lit(True), "volatile": pl.col("volatile"), "calm": ~pl.col("volatile")}
    for gname, gexpr in (groups or {"all": pl.lit(True)}).items():
        for win in ("dev", "holdout"):
            for reg, rexpr in regimes.items():
                sub = rows.filter((pl.col("window") == win) & rexpr.fill_null(False) & gexpr.fill_null(False))
                r: dict[str, Any] = {"group": gname, "window": win, "regime": reg, "fills": sub.height,
                                     "contracts": round(float(sub[w].sum()), 2) if sub.height else 0.0,
                                     "events": sub[key].n_unique() if sub.height else 0}
                for col in cols:
                    m, t, _ = clustered(sub, col, key, w)
                    r[col] = None if m is None else round(100 * m, 2)
                    r[f"t_{col}"] = None if t is None else round(t, 2)
                out.append(r)
    return pl.DataFrame(out)


# ---- Das (2005) Bayesian market maker ----------------------------------------------------------
class DasMM(Quoter):
    """Bayesian zero-profit quoting on top of ``Quoter`` (see module doc).

    ``alpha``: informed share of takers (or per-market ``alpha_of``); ``sigma``: noise of the
    informed signal; ``reset``: weight of a fresh prior around the current mid mixed in before
    each trade update (the true value moves); ``prior_sd``: sd of that prior; ``margin``: extra
    edge required beyond the maker fee; ``fee_info``: series -> {fee_type, fee_multiplier}."""

    def __init__(self, *, alpha: float = 0.3, sigma: float = 0.03, reset: float = 0.1,
                 prior_sd: float = 0.05, margin: float = 0.0, alpha_of: dict[str, float] | None = None,
                 fee_info: dict[str, dict] | None = None, **kw: Any) -> None:
        kw.setdefault("min_spread", 0.03)
        kw.setdefault("improve", True)
        super().__init__(**kw)
        self.alpha, self.sigma, self.reset, self.prior_sd, self.margin = alpha, sigma, reset, prior_sd, margin
        self.alpha_of = alpha_of or {}
        self.fee_info = fee_info or {}
        self.v = np.linspace(0.0, 1.0, 101)
        self.ticks = np.round(np.arange(1, 100) / 100, 2)
        # P(informed signal v + N(0, sigma) above tick a): rows ticks, columns values
        self._up = ndtr((self.v[None, :] - self.ticks[:, None]) / sigma)
        self.bel: dict[str, np.ndarray] = {}
        self.zp: dict[str, tuple[float | None, float | None]] = {}
        self.trades_seen = 0
        self.gate_blocks: dict[str, int] = defaultdict(int)

    def _alpha(self, market: str) -> float:
        return self.alpha_of.get(market, self.alpha)

    def _fee_rate(self, market: str) -> float:
        info = self.fee_info.get(market.split("-", 1)[0], {})
        return maker_fee_rate(info.get("fee_type"), info.get("fee_multiplier"))

    def _prior(self, mid: float) -> np.ndarray:
        w = np.exp(-0.5 * ((self.v - mid) / self.prior_sd) ** 2)
        return w / w.sum()

    def zero_profit(self, bel: np.ndarray, alpha: float, fee_rate: float) -> tuple[float | None, float | None]:
        """(bid*, ask*) on the 1c grid: the best prices with non-negative expected profit net of
        the maker fee and ``margin`` given a taker arrives (GM conditional expectations)."""
        lb = alpha * self._up + (1 - alpha) / 2               # P(buy at a | v)
        ls = alpha * (1 - self._up) + (1 - alpha) / 2         # P(sell at b | v)
        e_buy = (lb @ (self.v * bel)) / (lb @ bel)
        e_sell = (ls @ (self.v * bel)) / (ls @ bel)
        cost = fee_rate * self.ticks * (1 - self.ticks) + self.margin
        ok_ask = self.ticks - e_buy >= cost - EPS
        ok_bid = e_sell - self.ticks >= cost - EPS
        ask = float(self.ticks[np.argmax(ok_ask)]) if ok_ask.any() else None
        bid = float(self.ticks[len(ok_bid) - 1 - np.argmax(ok_bid[::-1])]) if ok_bid.any() else None
        return bid, ask

    def _ensure(self, sim: Simulator, market: str) -> np.ndarray | None:
        bel = self.bel.get(market)
        if bel is None:
            q = sim.markets[market].quote()
            if q["yes_bid"] is None or q["yes_ask"] is None:
                return None
            bel = self._prior((q["yes_bid"] + q["yes_ask"]) / 2)
            self.bel[market] = bel
            self.zp[market] = self.zero_profit(bel, self._alpha(market), self._fee_rate(market))
        return bel

    def observe(self, sim: Simulator, ev: Event) -> None:
        if ev.kind != "trade" or ev.market not in sim.synced:
            return
        bel = self._ensure(sim, ev.market)
        if bel is None:
            return
        q = sim.markets[ev.market].quote()
        if self.reset and q["yes_bid"] is not None and q["yes_ask"] is not None:
            bel = (1 - self.reset) * bel + self.reset * self._prior((q["yes_bid"] + q["yes_ask"]) / 2)
        a = self._alpha(ev.market)
        price = float(ev.msg["yes_price_dollars"])
        i = int(round(price * 100)) - 1
        if 0 <= i < len(self.ticks):
            up = self._up[i]
            like = a * up + (1 - a) / 2 if ev.msg["taker_side"] == "yes" else a * (1 - up) + (1 - a) / 2
            bel = bel * like
            bel = bel / bel.sum()
        self.bel[ev.market] = bel
        self.zp[ev.market] = self.zero_profit(bel, a, self._fee_rate(ev.market))
        self.trades_seen += 1

    def _wanted(self, sim: Simulator, market: str) -> dict[str, float | None]:
        want = super()._wanted(sim, market)
        if want["yes"] is None and want["no"] is None:
            return want
        if self._ensure(sim, market) is None:
            return {"yes": None, "no": None}
        bid_star, ask_star = self.zp[market]
        if want["yes"] is not None and (bid_star is None or want["yes"] > bid_star + EPS):
            want["yes"] = None
            self.gate_blocks["yes"] += 1
        if want["no"] is not None and (ask_star is None or 1 - want["no"] < ask_star - EPS):
            want["no"] = None
            self.gate_blocks["no"] += 1
        return want

    def belief_mean(self, market: str) -> float | None:
        bel = self.bel.get(market)
        return None if bel is None else float(self.v @ bel)


class Recorder:
    """Collects the public tape during a replay: taker trades and recorded executions with the
    external quote just before them (for PIN, lambda and the maker-flow markouts)."""

    def __init__(self) -> None:
        self.trades: list[tuple] = []
        self.execs: list[tuple] = []
        self._pre: tuple[float | None, float | None] = (None, None)

    def before(self, sim: Simulator, ev: Event) -> None:
        if ev.kind == "delta" and ev.is_exec and ev.market in sim.synced:
            q = sim.markets[ev.market].quote()
            self._pre = (q["yes_bid"], q["yes_ask"])

    def after(self, sim: Simulator, ev: Event) -> None:
        if ev.kind == "trade":
            m = ev.msg
            self.trades.append((ev.market, ev.recv_ns, m.get("ts_ms") or 0, m["taker_side"] == "yes",
                                float(m["count_fp"]), float(m["yes_price_dollars"])))
        elif ev.kind == "delta" and ev.is_exec and ev.market in sim.synced:
            side, price = ev.msg["side"], float(ev.msg["price_dollars"])
            lev = sim.markets[ev.market].levels[side].get(price)
            cleared = lev is None or lev.ext_total() <= EPS
            self.execs.append((ev.market, ev.recv_ns, side, price, -float(ev.msg["delta_fp"]),
                               self._pre[0], self._pre[1], cleared))

    def trades_frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.trades, schema={"market": pl.String, "t_ns": pl.Int64, "ts_ms": pl.Int64,
                                                 "buy": pl.Boolean, "qty": pl.Float64, "yes_price": pl.Float64},
                            orient="row")


def replay(events: Iterator[Event], strategy: Any | None, cfg: SimConfig,
           recorder: Recorder | None = None) -> tuple[Simulator, dict[str, int]]:
    """Simulator loop: optional recorder, then ``strategy.observe`` (if any) and
    ``strategy.on_event`` after each event, exactly as ``backtest.l2.run``."""
    sim = Simulator(cfg)
    stats: dict[str, int] = defaultdict(int)
    obs = getattr(strategy, "observe", None)
    for ev in events:
        if recorder is not None:
            recorder.before(sim, ev)
        sim.apply(ev)
        stats["events"] += 1
        if recorder is not None:
            recorder.after(sim, ev)
        if strategy is not None:
            if obs is not None:
                obs(sim, ev)
            strategy.on_event(sim, ev.market)
    return sim, stats


# ---- L2 scoring ----------------------------------------------------------------------------------
def _event_of(market: str) -> str:
    return market.rsplit("-", 1)[0]


def fill_frame(sim: Simulator, results: dict[str, float | None], fee_info: dict[str, dict]) -> pl.DataFrame:
    """Per simulated fill: markouts (``backtest.l2.markouts``: mo_* to the mid, mx_* to the exit
    price), the external quote when the order was placed and just before the fill, the exact
    maker fee rate, and PnL per contract net of fees to settlement (``settle``, None if
    unsettled) and to settlement-or-last-mid (``pnl_mark``)."""
    rows = markouts(sim)          # one row per sim.fills entry, in order
    if not rows:
        return pl.DataFrame()
    out = []
    for r, fl in zip(rows, sim.fills):
        o = sim.orders.get(fl.oid)
        b0, a0 = sim.quote_at(r["market"], o.placed_ns) if o else (None, None)
        b1, a1 = sim.quote_at(r["market"], r["t_ns"] - 1)
        info = fee_info.get(r["market"].split("-", 1)[0], {})
        res = results.get(r["market"])
        last = sim.mid_at(r["market"], 2**62)
        final = res if res is not None else last
        fee_pc = r["fee"] / r["qty"]

        def val(x: float | None, side: str = r["side"]) -> float | None:
            return None if x is None else (x if side == "yes" else 1 - x)

        out.append(r | {"event": _event_of(r["market"]), "placed_ns": o.placed_ns if o else None,
                        "bid0": b0, "ask0": a0, "bid1": b1, "ask1": a1,
                        "fee_rate": maker_fee_rate(info.get("fee_type"), info.get("fee_multiplier")),
                        "settle": None if res is None else val(res) - r["price"] - fee_pc,
                        "pnl_mark": None if final is None else val(final) - r["price"] - fee_pc})
    return pl.DataFrame(out, infer_schema_length=None)


def exec_frame(sim: Simulator, rec: Recorder, results: dict[str, float | None],
               fee_info: dict[str, dict]) -> pl.DataFrame:
    """Recorded executions as maker fills (the resting order bought ``side`` at ``price``):
    markouts at 60/300 s to the mid and to the exit price, and settlement net of the exact
    maker fee. ``cleared``: the execution emptied its price level (a back-of-queue fill)."""
    end = max((t[-1] for t, _ in sim.quotes.values() if t), default=0)
    out = []
    for mkt, t, side, price, qty, b, a, cleared in rec.execs:
        info = fee_info.get(mkt.split("-", 1)[0], {})
        rate = maker_fee_rate(info.get("fee_type"), info.get("fee_multiplier"))
        r: dict[str, Any] = {"market": mkt, "event": _event_of(mkt), "t_ns": t, "side": side, "price": price,
                             "size": qty, "qty": 1.0, "bid1": b, "ask1": a, "cleared": cleared,
                             "fee_rate": rate, "fee": rate * price * (1 - price)}
        for h in (60, 300):
            th = t + h * 10**9
            mid = sim.mid_at(mkt, th) if th <= end else None
            ex = sim.exit_at(mkt, side, th) if th <= end else None
            r[f"mo_{h}s"] = None if mid is None else (mid if side == "yes" else 1 - mid) - price
            r[f"mx_{h}s"] = None if ex is None else ex - price
        res = results.get(mkt)
        r["settle"] = None if res is None else (res if side == "yes" else 1 - res) - price - r["fee"]
        out.append(r)
    return pl.DataFrame(out, infer_schema_length=None) if out else pl.DataFrame()


def l2_orders(trades: pl.DataFrame) -> pl.DataFrame:
    """Prints -> taker orders (same market, exchange ms and taker side), with the 5-min bucket."""
    return (trades.group_by("market", "ts_ms", "buy")
            .agg(pl.col("t_ns").min(), pl.col("qty").sum(), pl.col("yes_price").mean())
            .with_columns((pl.col("t_ns") // (BUCKET_S * 10**9)).alias("bucket"))
            .sort("market", "t_ns"))


def l2_market_stats(sim: Simulator, orders: pl.DataFrame, t0: int, t1: int) -> pl.DataFrame:
    """Per market over [t0, t1): EKOP fit on 5-min order counts, VPIN, Kyle's lambda (1-min mid
    change on signed contracts), sign autocorrelation, mean |mid change| over 300 s and median
    quoted spread (both sampled every 60 s)."""
    o = orders.filter(pl.col("t_ns").is_between(t0, t1, closed="left"))
    counts = bucket_counts(o, ["market"])
    fits = fit_units(counts, ["market"])
    rows = []
    for mkt, g in o.group_by("market"):
        mkt = mkt[0]
        times, vals = sim.quotes.get(mkt, ([], []))
        grid = list(range(t0 - t0 % 60_000_000_000 + 60_000_000_000, t1, 60_000_000_000))
        mids = [sim.mid_at(mkt, t) for t in grid] if times else []
        spreads = [(a - b) for b, a in (sim.quote_at(mkt, t) for t in grid) if b is not None and a is not None] if times else []
        # signed contracts per minute
        g = g.with_columns((pl.col("t_ns") // 60_000_000_000).alias("m"),
                           pl.when(pl.col("buy")).then(pl.col("qty")).otherwise(-pl.col("qty")).alias("sq"))
        sq = dict(g.group_by("m").agg(pl.col("sq").sum()).iter_rows())
        dp, qv = [], []
        for i in range(1, len(grid)):
            if mids[i] is not None and mids[i - 1] is not None:
                dp.append(mids[i] - mids[i - 1])
                qv.append(sq.get(grid[i - 1] // 60_000_000_000, 0.0))
        kl = kyle_lambda(dp, qv)
        d300 = [abs(mids[i + 5] - mids[i]) for i in range(len(mids) - 5)
                if mids[i] is not None and mids[i + 5] is not None]
        signs = g.sort("t_ns")["buy"].cast(pl.Float64).to_numpy() * 2 - 1
        rows.append({"market": mkt, "orders": g.height, "lam_c_per_100": None if kl["lam"] is None else kl["lam"] * 1e4,
                     "t_lam": kl["t_lam"], "sign_ac1": sign_autocorr(signs),
                     "absmove300_c": 100 * float(np.mean(d300)) if d300 else None,
                     "spread_med_c": 100 * float(np.median(spreads)) if spreads else None,
                     "mid_med": float(np.median([m for m in mids if m is not None])) if any(m is not None for m in mids) else None})
    st = pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame({"market": []}, schema={"market": pl.String})
    if fits.height:
        st = st.join(fits.drop("orders"), on="market", how="left")
    return st


def static_alpha(stats_dev: pl.DataFrame, orders_dev: pl.DataFrame, markets: Iterable[str]) -> pl.DataFrame:
    """Dev-fitted informed share and move size for every market in ``markets``: the market's own
    EKOP fit and dev |mid move| over 300 s where it has one, else the series' pooled EKOP fit
    (each market-bucket of the series one observation) and the series median move, else the pooled
    fit over all markets. Columns: market, pin, pin_mom, absmove300_c, pin_src."""
    ser = lambda c: pl.col(c).str.split("-").list.first()   # noqa: E731
    o = orders_dev.with_columns(ser("market").alias("series"))
    cnt = bucket_counts(o, ["series", "market"])
    pooled = fit_units(cnt.drop("market"), ["series"])
    allfit = fit_pin(cnt["B"].to_numpy(), cnt["S"].to_numpy())
    if "absmove300_c" not in stats_dev.columns:      # no dev data at all
        stats_dev = stats_dev.with_columns(pl.lit(None, pl.Float64).alias("absmove300_c"))
    mv = stats_dev.select("market", "absmove300_c").with_columns(ser("market").alias("series"))
    mv_ser = mv.group_by("series").agg(pl.col("absmove300_c").median().alias("_mv_s"))
    mv_all = mv["absmove300_c"].median() if mv.height else None
    own = stats_dev.select("market", pl.col("pin").alias("_p"), pl.col("pin_mom").alias("_pm")) if "pin" in stats_dev.columns else None
    base = pl.DataFrame({"market": sorted(set(markets))}).with_columns(ser("market").alias("series"))
    if own is not None:
        base = base.join(own, on="market", how="left")
    else:
        base = base.with_columns(pl.lit(None, pl.Float64).alias("_p"), pl.lit(None, pl.Float64).alias("_pm"))
    if pooled.height:
        base = base.join(pooled.select("series", pl.col("pin").alias("_ps"), pl.col("pin_mom").alias("_pms")), on="series", how="left")
    else:
        base = base.with_columns(pl.lit(None, pl.Float64).alias("_ps"), pl.lit(None, pl.Float64).alias("_pms"))
    base = base.join(mv.select("market", "absmove300_c"), on="market", how="left").join(mv_ser, on="series", how="left")
    ga = allfit["pin"] if allfit else None
    gm = allfit["pin_mom"] if allfit else None
    return base.select(
        "market",
        pl.coalesce("_p", "_ps", pl.lit(ga, pl.Float64)).alias("pin"),
        pl.coalesce("_pm", "_pms", pl.lit(gm, pl.Float64)).alias("pin_mom"),
        pl.coalesce("absmove300_c", "_mv_s", pl.lit(mv_all, pl.Float64)).alias("absmove300_c"),
        pl.when(pl.col("_p").is_not_null()).then(pl.lit("market")).when(pl.col("_ps").is_not_null())
        .then(pl.lit("series")).otherwise(pl.lit("all")).alias("pin_src"))


def rolling_alpha(orders: pl.DataFrame, buckets: pl.DataFrame, n_prev: int = 12) -> pl.DataFrame:
    """Per (market, bucket): ``pin_prev`` (bias-adjusted moment PIN, see ``pin_mom``) and mean
    mid range over the market's previous ``n_prev`` buckets (strictly before the bucket: usable
    at decision time)."""
    c = (orders.group_by("market", "bucket").agg(pl.col("buy").sum().cast(pl.Float64).alias("B"),
                                                 (~pl.col("buy")).sum().cast(pl.Float64).alias("S")))
    b = (buckets.select("market", (pl.col("bucket_ns") // (BUCKET_S * 10**9)).alias("bucket"), "range_c", "volatile")
         .join(c, on=["market", "bucket"], how="full", coalesce=True)
         .with_columns(pl.col("B").fill_null(0.0), pl.col("S").fill_null(0.0))
         .sort("market", "bucket"))
    roll = dict(window_size=n_prev, min_samples=1)
    b = b.with_columns(
        ((pl.col("B") - pl.col("S")).abs() - noise_expr(pl.col("B") + pl.col("S"))).alias("_imb"),
        (pl.col("B") + pl.col("S")).alias("_tot"),
    ).with_columns(
        pl.col("_imb").shift(1).rolling_sum(**roll).over("market").alias("_imb_p"),
        pl.col("_tot").shift(1).rolling_sum(**roll).over("market").alias("_tot_p"),
        pl.col("range_c").shift(1).rolling_mean(**roll).over("market").alias("range_prev_c"),
    ).with_columns(
        pl.when(pl.col("_tot_p") >= 10).then((pl.col("_imb_p") / pl.col("_tot_p")).clip(0.0, 1.0)).otherwise(None).alias("pin_prev"),
    )
    return b.select("market", "bucket", "pin_prev", "range_prev_c", "volatile")


def adverse_tables(execs: pl.DataFrame, split_ns: int, n_prev: int = 12) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Empirical GM half-spread: the adverse move a resting order suffers when a taker hits it,
    ``adverse_c`` = own-side mid just before the execution minus own-side mid 300 s later (cents),
    i.e. E[V | trade] - p measured. Needs ``exec_frame`` rows with ``bid1``/``ask1``/``mo_300s``.

    Returns (rolling, static): rolling per (market, bucket) = mean adverse_c of executions at
    least two buckets earlier (their 300 s horizon has ended before the bucket starts: no
    look-ahead), over ``n_prev`` buckets; static per market = dev mean (market, else series,
    else all)."""
    mid = (pl.col("bid1") + pl.col("ask1")) / 2
    own = pl.when(pl.col("side") == "yes").then(mid).otherwise(1 - mid)
    e = (execs.filter(pl.col("mo_300s").is_not_null() & pl.col("bid1").is_not_null() & pl.col("ask1").is_not_null())
         .with_columns((100 * (own - pl.col("price") - pl.col("mo_300s"))).alias("adverse_c"),
                       (pl.col("t_ns") // (BUCKET_S * 10**9)).alias("bucket"),
                       pl.col("market").str.split("-").list.first().alias("series")))
    g = e.group_by("market", "bucket").agg(pl.col("adverse_c").sum().alias("_s"), pl.len().alias("_n")).sort("market", "bucket")
    # shift by one more bucket: executions in the previous bucket may still be inside their horizon
    nxt = g.with_columns((pl.col("bucket") + 2).alias("bucket"))
    roll = nxt.with_columns(
        pl.col("_s").rolling_sum(n_prev, min_samples=1).over("market").alias("_rs"),
        pl.col("_n").rolling_sum(n_prev, min_samples=1).over("market").alias("_rn"),
    ).filter(pl.col("_rn") >= 5).select("market", "bucket", (pl.col("_rs") / pl.col("_rn")).alias("adverse_prev_c"))
    d = e.filter(pl.col("t_ns") + 300 * 10**9 < split_ns)
    m = d.group_by("market").agg(pl.col("adverse_c").mean().alias("_m"), pl.len().alias("_n")).filter(pl.col("_n") >= 10)
    sr = d.group_by("series").agg(pl.col("adverse_c").mean().alias("_sr"), pl.len().alias("_n")).filter(pl.col("_n") >= 10)
    allm = d["adverse_c"].mean() if d.height else None
    base = e.select("market", "series").unique().join(m.drop("_n"), on="market", how="left").join(sr.drop("_n"), on="series", how="left")
    static = base.select("market", pl.coalesce("_m", "_sr", pl.lit(allm, pl.Float64)).alias("adverse_dev_c"))
    return roll, static


def _asof_bucket(rows: pl.DataFrame, roll: pl.DataFrame, col: str) -> pl.DataFrame:
    """Join the latest ``roll`` value at or before each row's bucket (rolling tables only have
    rows for buckets with data)."""
    r = roll.select("market", pl.col("bucket").alias("_rb"), col).sort("_rb")
    return (rows.sort("bucket").join_asof(r, left_on="bucket", right_on="_rb", by="market", strategy="backward")
            .drop("_rb"))


def attach_info(rows: pl.DataFrame, *, t_col: str, mid_cols: tuple[str, str], roll: pl.DataFrame,
                static: pl.DataFrame | None, buckets: pl.DataFrame, split_ns: int,
                adverse: tuple[pl.DataFrame, pl.DataFrame] | None = None) -> pl.DataFrame:
    """Add the fill's own-bucket ``volatile`` flag, ``window`` (dev/holdout), the rolling
    (``*_prev``) and static (dev-fitted) informed share and move size, and the GM excesses:

    ``h_c``: half-spread captured at decision time (mid minus our price, own side), in cents;
    ``x2_roll`` / ``x2_dev``: h - pi*Delta - fee (two-point GM), rolling or dev-fitted;
    ``xb_dev``: h - terminal GM half-spread(mid, PIN_dev) - fee;
    ``xi_roll`` / ``xi_dev``: h - measured adverse move (``adverse_tables``) - fee."""
    b, a = mid_cols
    bkt = (pl.col(t_col) // (BUCKET_S * 10**9)).alias("bucket")
    vol = buckets.select("market", (pl.col("bucket_ns") // (BUCKET_S * 10**9)).alias("bucket"), "volatile")
    rows = rows.with_columns(bkt, (pl.col("t_ns") // (BUCKET_S * 10**9)).alias("_fb"))
    rows = (rows.join(vol.rename({"bucket": "_fb"}), on=["market", "_fb"], how="left")
            .join(roll.drop("volatile"), on=["market", "bucket"], how="left")
            .with_columns(pl.col("volatile").fill_null(False),
                          pl.when(pl.col("t_ns") < split_ns).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("window")))
    mid = (pl.col(b) + pl.col(a)) / 2
    own_mid = pl.when(pl.col("side") == "yes").then(mid).otherwise(1 - mid)
    rows = rows.with_columns(mid.alias("mid_dec"), (100 * (own_mid - pl.col("price"))).alias("h_c"),
                             (100 * (pl.col(a) - pl.col(b))).alias("spread_dec_c"),
                             (100 * pl.col("fee_rate") * pl.col("price") * (1 - pl.col("price"))).alias("fee_c"))
    rows = rows.with_columns(
        (pl.col("h_c") - pl.col("pin_prev") * pl.col("range_prev_c") - pl.col("fee_c")).alias("x2_roll"))
    if static is not None and static.height:
        st = static.select("market", pl.col("pin").alias("pin_dev"), pl.col("pin_mom").alias("pin_mom_dev"),
                           pl.col("absmove300_c").alias("move_dev_c"))
        rows = rows.join(st, on="market", how="left").with_columns(
            (pl.col("h_c") - pl.col("pin_dev") * pl.col("move_dev_c") - pl.col("fee_c")).alias("x2_dev"),
            (pl.col("h_c") - 100 * gm_half_binary(pl.col("mid_dec"), pl.col("pin_dev")) - pl.col("fee_c")).alias("xb_dev"))
    if adverse is not None:
        aroll, astat = adverse
        rows = _asof_bucket(rows, aroll, "adverse_prev_c").join(astat, on="market", how="left").with_columns(
            (pl.col("h_c") - pl.col("adverse_prev_c") - pl.col("fee_c")).alias("xi_roll"),
            (pl.col("h_c") - pl.col("adverse_dev_c") - pl.col("fee_c")).alias("xi_dev"))
    return rows.drop("_fb").sort("t_ns")


_REFRESHED: set[str] = set()     # unsettled tickers already re-checked by this process


def market_results(tickers: Iterable[str], cache: Path, *, refresh_unsettled: bool = True) -> dict[str, float | None]:
    """Settled YES result (1.0 / 0.0, None if open) per ticker from the public market endpoint,
    cached as JSON (unsettled markets are re-checked once per process)."""
    import httpx

    data = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [t for t in sorted(set(tickers)) if t not in data
            or (refresh_unsettled and data[t] is None and t not in _REFRESHED)]
    _REFRESHED.update(todo)
    if todo:
        with httpx.Client(timeout=20) as h:
            for t in todo:
                try:
                    r = h.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{t}")
                    res = r.json().get("market", {}).get("result") if r.status_code == 200 else None
                except (httpx.HTTPError, ValueError):
                    res = None
                data[t] = {"yes": 1.0, "no": 0.0}.get(res) if res else None
                time.sleep(0.05)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
    return {t: data.get(t) for t in tickers}


def _ns(hour: str) -> int:
    return int(datetime.fromisoformat(hour + ":00:00").replace(tzinfo=UTC).timestamp() * 1e9)


def fee_table(s) -> dict[str, dict]:
    ser = pl.read_parquet(s.tables_dir / "kalshi_series.parquet")
    return {r["series_ticker"]: r for r in ser.iter_rows(named=True)}


def run_l2(s, start: str, end: str, strategy: Any, *, latency_ms: float, record: bool = False,
           fee_info: dict[str, dict] | None = None) -> tuple[Simulator, Recorder | None, dict[str, int]]:
    files = stream_files(s.raw_dir / "stream", start, end)
    if not files:
        raise RuntimeError(f"no recorded stream files in [{start}, {end})")
    cfg = SimConfig(latency_ms=latency_ms, queue="risk_averse", fee_info=fee_info or fee_table(s),
                    writes_per_s=10.0)
    rec = Recorder() if record else None
    stats: dict[str, int] = defaultdict(int)
    sim, st = replay(iter_events(files, stats), strategy, cfg, rec)
    return sim, rec, dict(stats) | st


def baseline_quoter(fee_info: dict[str, dict]) -> Quoter:
    return Quoter(improve=True, min_spread=0.05, cooldown_s=300,
                  category_of={k: v.get("category") for k, v in fee_info.items()})


def das_mm(fee_info: dict[str, dict], **params: Any) -> DasMM:
    return DasMM(fee_info=fee_info, cooldown_s=300,
                 category_of={k: v.get("category") for k, v in fee_info.items()}, **params)


FILL_COLS = ["mo_60s", "mo_300s", "mx_60s", "mx_300s", "pnl_mark", "settle"]


def strategy_summary(f: pl.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"fills": f.height}
    if not f.height:
        return out
    out.update({"contracts": round(float(f["qty"].sum()), 2), "events": f["event"].n_unique(),
                "fees": round(float(f["fee"].sum()), 2),
                "pnl_mark_usd": round(float((f["qty"] * f["pnl_mark"].fill_null(0)).sum()), 2),
                "settled_contracts": round(float(f.filter(pl.col("settle").is_not_null())["qty"].sum()), 2),
                "settled_usd": round(float((f["qty"] * f["settle"].fill_null(0)).sum()), 2)})
    return out


def l2_study(s, start: str = "2026-11-05T03", end: str = "2026-11-05T18", split: str = "2026-11-05T11", *,
             latencies: tuple[float, ...] = (50.0, 100.0), das_params: dict[str, Any] | None = None,
             tag: str | None = None, report: bool = True) -> Path | dict[str, Any]:
    """The L2 half of the study on recorded hours [start, end), dev before ``split``:

    1. baseline ``Quoter`` at each latency (the first run also records the tape);
    2. PIN / VPIN / lambda / sign autocorrelation per market on dev and on holdout;
    3. GM excess for recorded executions and Quoter fills, markouts by excess sign;
    4. ``DasMM`` (``das_params``, chosen on dev) with a global alpha and with the market's
       dev PIN as alpha, at each latency.
    Derived tables are cached under ``data/cache/info_mm_<tag>_*.parquet``."""
    from albert.research.volatile import cached_buckets

    das_params = das_params or {"alpha": 0.3, "sigma": 0.03, "reset": 0.1, "prior_sd": 0.05}
    tag = tag or f"{start}_{end}".replace(":", "")
    cache = s.data_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    fee_info = fee_table(s)
    split_ns = _ns(split)
    buckets = cached_buckets(s, start, end)
    res_cache = cache / "info_mm_results.json"

    fills: dict[str, pl.DataFrame] = {}
    meta: dict[str, Any] = {}
    t_lo, t_hi = _ns(start), _ns(end)
    # first latency: baseline Quoter + the public tape (everything that needs the replayed quotes)
    lat0 = latencies[0]
    sim0, rec, st = run_l2(s, start, end, baseline_quoter(fee_info), latency_ms=lat0, record=True, fee_info=fee_info)
    assert rec is not None
    res = market_results({f.market for f in sim0.fills}, res_cache)
    fills[f"quoter_{int(lat0)}"] = fill_frame(sim0, res, fee_info)
    meta[f"quoter_{int(lat0)}"] = {"orders": len(sim0.orders), "replay_events": st.get("events"), "diag": dict(sim0.diag)}
    trades = rec.trades_frame()
    orders = l2_orders(trades)
    stats_dev = l2_market_stats(sim0, orders, t_lo, split_ns).with_columns(pl.lit("dev").alias("window"))
    stats_hold = l2_market_stats(sim0, orders, split_ns, t_hi).with_columns(pl.lit("holdout").alias("window"))
    static = static_alpha(stats_dev, orders.filter(pl.col("t_ns") < split_ns),
                          set(buckets["market"].to_list()) | set(orders["market"].to_list()))
    roll = rolling_alpha(orders, buckets)
    ex_res = market_results({m for m, *_ in rec.execs}, res_cache)
    execs = exec_frame(sim0, rec, ex_res, fee_info)
    del sim0, rec
    trades.write_parquet(cache / f"info_mm_{tag}_trades.parquet")
    pl.concat([stats_dev, stats_hold], how="diagonal_relaxed").write_parquet(cache / f"info_mm_{tag}_pin.parquet")
    for lat in latencies[1:]:
        sim, _, st = run_l2(s, start, end, baseline_quoter(fee_info), latency_ms=lat, fee_info=fee_info)
        res = market_results({f.market for f in sim.fills}, res_cache)
        fills[f"quoter_{int(lat)}"] = fill_frame(sim, res, fee_info)
        meta[f"quoter_{int(lat)}"] = {"orders": len(sim.orders), "replay_events": st.get("events"), "diag": dict(sim.diag)}
        del sim

    adverse = adverse_tables(execs, split_ns)
    execs = attach_info(execs, t_col="t_ns", mid_cols=("bid1", "ask1"), roll=roll, static=static,
                        buckets=buckets, split_ns=split_ns, adverse=adverse)
    for k in list(fills):
        if fills[k].height:
            fills[k] = attach_info(fills[k], t_col="placed_ns", mid_cols=("bid0", "ask0"), roll=roll,
                                   static=static, buckets=buckets, split_ns=split_ns, adverse=adverse)

    # Das MM: global alpha and dev-fitted alpha (market, else series, else pooled; clean on holdout only)
    alpha_dev = {r["market"]: r["pin"] for r in static.iter_rows(named=True) if r.get("pin") is not None}
    for lat in latencies:
        for name, extra in (("das", {}), ("das_pin", {"alpha_of": alpha_dev})):
            strat = das_mm(fee_info, **(das_params | extra))
            sim, _, st = run_l2(s, start, end, strat, latency_ms=lat, fee_info=fee_info)
            res = market_results({f.market for f in sim.fills}, res_cache)
            f = fill_frame(sim, res, fee_info)
            if f.height:
                f = attach_info(f, t_col="placed_ns", mid_cols=("bid0", "ask0"), roll=roll, static=static,
                                buckets=buckets, split_ns=split_ns, adverse=adverse)
            fills[f"{name}_{int(lat)}"] = f
            meta[f"{name}_{int(lat)}"] = {"orders": len(sim.orders), "replay_events": st.get("events"),
                                          "trades_seen": strat.trades_seen, "gate_blocks": dict(strat.gate_blocks),
                                          "diag": dict(sim.diag)}
            del sim
    execs.write_parquet(cache / f"info_mm_{tag}_execs.parquet")
    for k, f in fills.items():
        if f.height:
            f.write_parquet(cache / f"info_mm_{tag}_fills_{k}.parquet")
    out = {"execs": execs, "fills": fills, "stats_dev": stats_dev, "stats_hold": stats_hold, "meta": meta,
           "static": static}
    if not report:
        return out
    return l2_report(s, out, start=start, end=end, split=split, das_params=das_params)


def l2_families(f: pl.DataFrame) -> dict[str, dict[str, pl.Expr]]:
    """Candidate alpha-aware rules for ``pick_rules``: each GM-excess sign and the terciles of
    the rolling and dev-fitted informed share."""
    fam = {f"{c} sign": {f"{c}>0": pl.col(c) > 0, f"{c}<=0": pl.col(c) <= 0}
           for c in ("x2_roll", "x2_dev", "xb_dev", "xi_roll", "xi_dev") if c in f.columns}
    for c in ("pin_prev", "pin_dev"):
        if c in f.columns:
            fam[f"{c} terciles"] = tercile_groups(f, c)
    return fam


def excess_groups() -> dict[str, pl.Expr]:
    """Fill groups by the sign of each GM excess (see ``attach_info``)."""
    g: dict[str, pl.Expr] = {"all": pl.lit(True)}
    for c in ("x2_roll", "x2_dev", "xb_dev", "xi_roll", "xi_dev"):
        g[f"{c}>0"] = pl.col(c) > 0
        g[f"{c}<=0"] = pl.col(c) <= 0
    return g


def l2_report(s, out: dict[str, Any], *, start: str, end: str, split: str, das_params: dict[str, Any]) -> Path:
    from albert.reports import Report

    rep = Report(s.reports_dir, "research-info-mm")
    rep.text("Scope", f"L2 replay {start}..{end} (end exclusive), dev < {split} <= holdout. Volatile/calm "
                      "splits use each fill's own 5-min bucket (`volatile` of `research.volatile`, an "
                      "evaluation split). Markouts in cents per contract; t clustered by event.")
    sd, sh = out["stats_dev"], out["stats_hold"]
    desc = []
    for name, st in (("dev", sd), ("holdout", sh)):
        f = st.filter(pl.col("pin").is_not_null()) if "pin" in st.columns else st.head(0)
        desc.append({"window": name, "markets_fit": f.height,
                     **{f"{c}_{q}": (round(float(f[c].quantile(v)), 4) if f.height and c in f.columns and f[c].drop_nulls().len() else None)
                        for c in ("pin", "alpha", "pin_mom", "vpin", "lam_c_per_100", "sign_ac1", "absmove300_c", "spread_med_c")
                        for q, v in (("p25", 0.25), ("med", 0.5), ("p75", 0.75))}})
    rep.table("PIN / VPIN / lambda per market (quartiles)", pl.DataFrame(desc))
    if "pin" in sd.columns and "pin" in sh.columns:
        j = sd.select("market", "pin", "pin_mom").join(sh.select("market", pl.col("pin").alias("pin_h"), pl.col("pin_mom").alias("pin_mom_h")), on="market").drop_nulls()
        rep.json("PIN persistence dev -> holdout (same market)", {
            "markets": j.height,
            "spearman_pin": j.select(pl.corr("pin", "pin_h", method="spearman")).item() if j.height > 3 else None,
            "spearman_pin_mom": j.select(pl.corr("pin_mom", "pin_mom_h", method="spearman")).item() if j.height > 3 else None})
    ex = out["execs"]
    if ex.height:
        reg = pl.when(pl.col("volatile")).then(pl.lit("volatile")).otherwise(pl.lit("calm"))
        comp = (ex.with_columns(reg.alias("regime")).group_by("window", "regime").agg(
            pl.len().alias("execs"),
            (pl.col("spread_dec_c") / 2).median().alias("half_spread_c"),
            (pl.col("pin_prev") * pl.col("range_prev_c")).median().alias("gm_two_point_roll_c"),
            (pl.col("pin_dev") * pl.col("move_dev_c")).median().alias("gm_two_point_dev_c"),
            (100 * gm_half_binary(pl.col("mid_dec"), pl.col("pin_dev"))).median().alias("gm_terminal_c"),
            pl.col("adverse_prev_c").median().alias("gm_empirical_roll_c"),
            pl.col("fee_c").median().alias("fee_c"),
            *[(pl.col(c) > 0).mean().round(3).alias(f"share_{c}>0") for c in ("x2_roll", "xb_dev", "xi_roll")],
        ).sort("window", "regime"))
        rep.table("Actual half-spread vs GM zero-profit half-spreads at recorded executions (medians, cents)", comp)
        grp = excess_groups() | {"cleared": pl.col("cleared"),
                                 "cleared & xi_roll>0": pl.col("cleared") & (pl.col("xi_roll") > 0),
                                 "cleared & xi_roll<=0": pl.col("cleared") & (pl.col("xi_roll") <= 0)}
        rep.table("Recorded executions (1 lot each) by GM excess", split_table(ex, ["mo_60s", "mo_300s", "mx_60s", "mx_300s", "settle"], groups=grp), max_rows=200)
        rules = pl.concat([pick_rules(ex, l2_families(ex), m) for m in ("mo_300s", "mx_300s", "settle")], how="diagonal_relaxed")
        rep.table("Recorded executions: alpha-aware rules chosen on dev, scored on holdout (cents)", rules)
    comp = []
    for k, f in out["fills"].items():
        if f.height:
            comp.append(split_table(f, FILL_COLS).drop("group").with_columns(pl.lit(k).alias("strategy")))
    if comp:
        c = pl.concat(comp, how="diagonal_relaxed")
        rep.table("Strategies compared (cents per contract, t by event)", c.select("strategy", *[x for x in c.columns if x != "strategy"]), max_rows=60)
    for k, f in out["fills"].items():
        if not f.height:
            rep.text(k, "No fills.")
            continue
        rep.table(f"{k}: fills by GM excess at placement", split_table(f, FILL_COLS, groups=excess_groups()), max_rows=120)
        rules = pl.concat([pick_rules(f, l2_families(f), m, min_fills=20) for m in ("mx_300s", "pnl_mark")], how="diagonal_relaxed")
        rep.table(f"{k}: alpha-aware rules chosen on dev, scored on holdout (cents)", rules)
        rep.json(f"{k}: totals", strategy_summary(f) | out["meta"].get(k, {}))
    rep.json("Das parameters (chosen on dev)", das_params)
    return rep.save()


# ---- historical trade tape (trade-only) ----------------------------------------------------------
HIST_SPLIT = datetime(2026, 10, 1, tzinfo=UTC)
_US = 10**6                       # microseconds per second (historical times are datetime[us])


def hist_orders(trades: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Prints -> taker orders (same ticker, timestamp and taker side) with the lowest and highest
    print price (a sweep's extent), VWAP, the 5-min bucket and epoch microseconds ``us``."""
    return (trades.lazy().group_by("ticker", "created_time", "taker_side")
            .agg(pl.col("count").sum().alias("qty"), pl.col("yes_price").min().alias("p_lo"),
                 pl.col("yes_price").max().alias("p_hi"),
                 ((pl.col("yes_price") * pl.col("count")).sum() / pl.col("count").sum()).alias("vwap"),
                 pl.len().alias("prints"))
            .rename({"created_time": "t"})
            .with_columns((pl.col("taker_side") == "yes").alias("buy"),
                          pl.col("t").dt.epoch("us").alias("us"))
            .with_columns((pl.col("us") // (BUCKET_S * _US)).alias("bucket"))
            .drop("taker_side").sort("ticker", "us"))


def ws_trade_rows(rows: Iterable[dict[str, Any]]) -> pl.DataFrame:
    """Recorded WebSocket ``trade`` frames -> rows in the ``kalshi_trades`` table schema."""
    out = []
    for r in rows:
        d = json.loads(r["frame"])
        if d.get("type") != "trade":
            continue
        m = d["msg"]
        out.append({"trade_id": m["trade_id"], "ticker": m["market_ticker"],
                    "created_time": datetime.fromtimestamp(m["ts_ms"] / 1000, UTC), "count": float(m["count_fp"]),
                    "yes_price": float(m["yes_price_dollars"]), "taker_side": m["taker_side"]})
    return pl.DataFrame(out, schema={"trade_id": pl.String, "ticker": pl.String,
                                     "created_time": pl.Datetime("us", "UTC"), "count": pl.Float64,
                                     "yes_price": pl.Float64, "taker_side": pl.String})


def tape_state(orders: pl.DataFrame, stale_s: float = 900.0) -> pl.DataFrame:
    """Quote proxies from the tape after each order: ``ask_px`` = highest price of the last
    taker buy, ``bid_px`` = lowest price of the last taker sell (each forward-filled within the
    ticker and dropped once older than ``stale_s``); ``mid_px`` where both exist and bid < ask."""
    st = orders.with_columns(
        pl.when(pl.col("buy")).then(pl.col("p_hi")).alias("_a"), pl.when(pl.col("buy")).then(pl.col("us")).alias("_at"),
        pl.when(~pl.col("buy")).then(pl.col("p_lo")).alias("_b"), pl.when(~pl.col("buy")).then(pl.col("us")).alias("_bt"),
    ).with_columns(pl.col("_a", "_at", "_b", "_bt").forward_fill().over("ticker"))
    fresh = lambda c: (pl.col("us") - pl.col(c)) <= int(stale_s * _US)   # noqa: E731
    st = st.with_columns(pl.when(fresh("_at")).then(pl.col("_a")).alias("ask_px"),
                         pl.when(fresh("_bt")).then(pl.col("_b")).alias("bid_px"))
    return st.with_columns(
        pl.when(pl.col("bid_px") < pl.col("ask_px")).then((pl.col("bid_px") + pl.col("ask_px")) / 2).alias("mid_px"),
    ).drop("_a", "_at", "_b", "_bt")


def in_play_bounds(markets: pl.DataFrame) -> pl.DataFrame:
    """Game tickers -> [start, expected_expiration + 1 h]: start from the ticker's encoded time
    when present, else expected expiration minus the league's game length."""
    from albert.research.volatile import GAME_SERIES
    from albert.venues.kalshi import ticker_start

    g = markets.filter(pl.col("series_ticker").is_in(list(GAME_SERIES)) | pl.col("event_ticker").str.contains(r"-\d\d[A-Z]{3}\d{6}"))
    g = g.with_columns(pl.col("event_ticker").map_elements(ticker_start, return_dtype=pl.Datetime("us", "UTC")).alias("_st"),
                       pl.col("series_ticker").replace_strict(GAME_SERIES, default=None, return_dtype=pl.Float64).alias("_gl"))
    g = g.with_columns(pl.coalesce("_st", pl.col("expected_expiration_time") - pl.duration(minutes=(pl.col("_gl") * 60).cast(pl.Int64))).alias("play_start"),
                       (pl.col("expected_expiration_time") + pl.duration(hours=1)).alias("play_end"))
    return g.filter(pl.col("play_start").is_not_null() & pl.col("play_end").is_not_null()).select(
        "ticker", pl.col("play_start").dt.epoch("us").alias("ps_us"), pl.col("play_end").dt.epoch("us").alias("pe_us"))


def hist_units_pin(orders: pl.DataFrame, play: pl.DataFrame) -> pl.DataFrame:
    """EKOP + VPIN + lambda + sign autocorrelation per ticker-day and per in-play window."""
    o = orders.with_columns(pl.col("t").dt.date().alias("day"))
    st = tape_state(o)
    counts = bucket_counts(o, ["ticker", "day"])
    fits = fit_units(counts, ["ticker", "day"]).with_columns(pl.lit("ticker_day").alias("unit"))
    io = o.join(play, on="ticker").filter(pl.col("us").is_between(pl.col("ps_us"), pl.col("pe_us")))
    ifits = fit_units(bucket_counts(io, ["ticker"]), ["ticker"])
    if ifits.height:
        first_day = io.group_by("ticker").agg(pl.col("day").min())
        ifits = ifits.join(first_day, on="ticker").with_columns(pl.lit("in_play").alias("unit"))
    # lambda and sign autocorrelation on 1-minute bars of the tape mid proxy
    st = st.with_columns((pl.col("us") // (60 * _US)).alias("minute"),
                         pl.when(pl.col("buy")).then(pl.col("qty")).otherwise(-pl.col("qty")).alias("sq"))
    ip = play.select("ticker", "ps_us", "pe_us")
    st = st.join(ip, on="ticker", how="left").with_columns(
        pl.col("us").is_between(pl.col("ps_us"), pl.col("pe_us")).fill_null(False).alias("in_play"))
    bars = (st.group_by("ticker", "day", "minute").agg(pl.col("mid_px").last(), pl.col("sq").sum(), pl.col("in_play").first())
            .sort("ticker", "minute")
            .with_columns(pl.col("mid_px").diff().over("ticker").alias("dmid")))
    sg = st.with_columns((pl.col("buy").cast(pl.Float64) * 2 - 1).alias("sgn")).with_columns(
        pl.col("sgn").shift(1).over("ticker", "day").alias("sgn_l"))

    def agg(frame: pl.DataFrame, sframe: pl.DataFrame, key: list[str]) -> pl.DataFrame:
        b = frame.drop_nulls("dmid").group_by(key).agg(
            pl.cov("dmid", "sq").alias("_c"), pl.col("sq").var().alias("_v"), pl.len().alias("bars"))
        b = b.with_columns(pl.when((pl.col("bars") >= 10) & (pl.col("_v") > 0)).then(pl.col("_c") / pl.col("_v") * 1e4)
                           .alias("lam_c_per_100")).drop("_c", "_v")
        a = sframe.drop_nulls("sgn_l").group_by(key).agg(pl.corr("sgn", "sgn_l").alias("sign_ac1"))
        return b.join(a, on=key, how="full", coalesce=True)

    td = agg(bars, sg, ["ticker", "day"])
    fits = fits.join(td, on=["ticker", "day"], how="left")
    if ifits.height:
        ipl = agg(bars.filter("in_play"), sg.filter("in_play"), ["ticker"])
        ifits = ifits.join(ipl, on="ticker", how="left")
    out = pl.concat([fits, ifits], how="diagonal_relaxed") if ifits.height else fits
    return out.with_columns(pl.when(pl.col("day") < HIST_SPLIT.date()).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("window"),
                            pl.col("ticker").str.split("-").list.first().alias("series"))


def print_maker(orders: pl.DataFrame, *, min_spread: float = 0.05, latency_s: float = 1.0,
                cooldown_s: float = 300.0, max_pos: float = 3.0, lo: float = 0.05, hi: float = 0.95,
                stale_s: float = 900.0) -> pl.DataFrame:
    """Passive 1-lot maker on the trade tape (no book). After each taker order it rests a YES
    bid one tick above the last taker sell's lowest price and a YES ask one tick below the last
    taker buy's highest price, when that spread proxy is at least ``min_spread`` and both prints
    are fresher than ``stale_s``. A quote is live ``latency_s`` after it is (re)priced and fills
    only when a later taker order trades strictly THROUGH it (a sell printing below our bid, a
    buy printing above our ask). After a fill both sides are pulled for ``cooldown_s``.
    Returns one row per fill with the own-side price (YES bid fill -> side yes; ask fill -> we
    bought NO at 1 - ask) and the quote proxies at placement."""
    o = orders.sort("ticker", "us")
    tick = o["ticker"].to_list()
    us = o["us"].to_numpy()
    buy = o["buy"].to_numpy()
    plo = o["p_lo"].to_numpy()
    phi = o["p_hi"].to_numpy()
    lat, cool, stale = int(latency_s * _US), int(cooldown_s * _US), int(stale_s * _US)
    out: list[tuple] = []
    cur = None
    for i in range(len(us)):
        tk = tick[i]
        if tk != cur:
            cur = tk
            a_px = b_px = None
            a_t = b_t = 0
            bid = ask = None
            bid_on = ask_on = 0
            bid_ctx = ask_ctx = None
            pos = 0.0
            cool_until = 0
        t = us[i]
        # 1) fills of resting quotes by this order
        if buy[i]:
            if ask is not None and t >= ask_on and phi[i] > ask + EPS:
                out.append((tk, t, "no", round(1 - ask, 4), ask_ctx))
                pos -= 1
                cool_until = t + cool
                bid = ask = None
        else:
            if bid is not None and t >= bid_on and plo[i] < bid - EPS:
                out.append((tk, t, "yes", bid, bid_ctx))
                pos += 1
                cool_until = t + cool
                bid = ask = None
        # 2) tape state
        if buy[i]:
            a_px, a_t = phi[i], t
        else:
            b_px, b_t = plo[i], t
        # 3) re-price
        want_b = want_a = None
        if (t >= cool_until and a_px is not None and b_px is not None
                and t - a_t <= stale and t - b_t <= stale and a_px - b_px >= min_spread - EPS):
            mid = (a_px + b_px) / 2
            if lo <= mid <= hi:
                if pos < max_pos:
                    want_b = round(b_px + TICK, 4)
                if pos > -max_pos:
                    want_a = round(a_px - TICK, 4)
        ctx = (a_px, b_px)
        if want_b is None:
            bid = None
        elif bid is None or abs(want_b - bid) > EPS:
            bid, bid_on, bid_ctx = want_b, t + lat, ctx
        if want_a is None:
            ask = None
        elif ask is None or abs(want_a - ask) > EPS:
            ask, ask_on, ask_ctx = want_a, t + lat, ctx
    return pl.DataFrame([(tk, t, side, p, c[0], c[1]) for tk, t, side, p, c in out],
                        schema={"ticker": pl.String, "us": pl.Int64, "side": pl.String, "price": pl.Float64,
                                "ask0": pl.Float64, "bid0": pl.Float64}, orient="row")


def hist_markouts(fills: pl.DataFrame, state: pl.DataFrame, markets: pl.DataFrame,
                  horizons_s: tuple[int, ...] = (60, 300)) -> pl.DataFrame:
    """Exit (own-side bid proxy) and mid-proxy markouts at each horizon, and settlement PnL net
    of the exact maker fee, for historical maker fills (``side``, ``price`` own-side)."""
    s = state.select("ticker", "us", "bid_px", "ask_px", "mid_px").sort("us")
    f = fills.with_columns(pl.col("us").cast(pl.Int64)).sort("us")
    for h in horizons_s:
        f = (f.with_columns((pl.col("us") + h * _US).alias("_th")).sort("_th")
             .join_asof(s.rename({"us": "_ts", "bid_px": "_b", "ask_px": "_a", "mid_px": "_m"}), left_on="_th",
                        right_on="_ts", by="ticker", strategy="backward")
             .with_columns(
                 pl.when(pl.col("side") == "yes").then(pl.col("_b") - pl.col("price"))
                 .otherwise(1 - pl.col("_a") - pl.col("price")).alias(f"mx_{h}s"),
                 pl.when(pl.col("side") == "yes").then(pl.col("_m") - pl.col("price"))
                 .otherwise(1 - pl.col("_m") - pl.col("price")).alias(f"mo_{h}s"))
             .drop("_th", "_ts", "_b", "_a", "_m"))
    m = markets.select("ticker", "event_ticker", "result", "fee_type", "fee_multiplier")
    f = f.join(m, on="ticker", how="left").with_columns(
        pl.when(pl.col("fee_type").str.contains("maker_fee")).then(MAKER_RATE * pl.col("fee_multiplier").fill_null(1.0))
        .otherwise(0.0).alias("fee_rate"))
    yes = pl.when(pl.col("result") == "yes").then(1.0).when(pl.col("result") == "no").then(0.0)
    val = pl.when(pl.col("side") == "yes").then(yes).otherwise(1 - yes)
    return f.with_columns(
        (pl.col("fee_rate") * pl.col("price") * (1 - pl.col("price"))).alias("fee"),
    ).with_columns((val - pl.col("price") - pl.col("fee")).alias("settle"),
                   pl.col("event_ticker").alias("event"), pl.lit(1.0).alias("qty"))


def hist_alpha_prior(orders: pl.DataFrame, pins: pl.DataFrame) -> pl.DataFrame:
    """Look-ahead-free informed share per (ticker, bucket): ``pin_exp`` (expanding moment PIN,
    ``pin_mom``, over the ticker's earlier buckets, >= 30 orders), ``pin_prevday`` (EKOP PIN of the ticker's
    previous UTC day), and ``move_exp_c`` (expanding mean |mid-proxy change| per 5-min bucket)."""
    st = tape_state(orders)
    b = (st.group_by("ticker", "bucket").agg(pl.col("buy").sum().cast(pl.Float64).alias("B"),
                                             (~pl.col("buy")).sum().cast(pl.Float64).alias("S"),
                                             pl.col("mid_px").drop_nulls().first().alias("_m0"),
                                             pl.col("mid_px").drop_nulls().last().alias("_m1"),
                                             pl.col("t").first().dt.date().alias("day"))
         .sort("ticker", "bucket")
         .with_columns(((pl.col("B") - pl.col("S")).abs() - noise_expr(pl.col("B") + pl.col("S"))).alias("_imb"),
                       (pl.col("B") + pl.col("S")).alias("_tot"),
                       (100 * (pl.col("_m1") - pl.col("_m0")).abs()).alias("_mv")))
    b = b.with_columns(
        pl.col("_imb").cum_sum().shift(1).over("ticker").alias("_ci"),
        pl.col("_tot").cum_sum().shift(1).over("ticker").alias("_ct"),
        pl.col("_mv").cum_sum().shift(1).over("ticker").alias("_cm"),
        pl.col("_mv").is_not_null().cast(pl.Int64).cum_sum().shift(1).over("ticker").alias("_cn"),
    ).with_columns(
        pl.when(pl.col("_ct") >= 30).then((pl.col("_ci") / pl.col("_ct")).clip(0.0, 1.0)).alias("pin_exp"),
        pl.when(pl.col("_cn") >= 6).then(pl.col("_cm") / pl.col("_cn")).alias("move_exp_c"),
    )
    pd = (pins.filter(pl.col("unit") == "ticker_day").select("ticker", (pl.col("day") + pl.duration(days=1)).cast(pl.Date).alias("day"),
                                                             pl.col("pin").alias("pin_prevday")))
    return b.join(pd, on=["ticker", "day"], how="left").select("ticker", "bucket", "pin_exp", "move_exp_c", "pin_prevday")


def hist_volatile(orders: pl.DataFrame, play: pl.DataFrame, q: float = 0.8) -> pl.DataFrame:
    """Evaluation regime per (ticker, bucket): volatile when the tape mid-proxy path is in the top
    ``1 - q`` of buckets that moved, or the bucket is in play (game tickers)."""
    st = tape_state(orders)
    b = (st.with_columns(pl.col("mid_px").diff().abs().over("ticker").alias("_d"))
         .group_by("ticker", "bucket").agg((100 * pl.col("_d").sum()).alias("path_c"), pl.col("us").first()))
    cut = b.filter(pl.col("path_c") > 0)["path_c"].quantile(q)
    b = b.join(play, on="ticker", how="left").with_columns(
        pl.col("us").is_between(pl.col("ps_us"), pl.col("pe_us")).fill_null(False).alias("in_play"))
    return b.with_columns((((pl.col("path_c") >= cut) & (pl.col("path_c") > 0)) | pl.col("in_play")).alias("volatile")).select(
        "ticker", "bucket", "path_c", "in_play", "volatile")


def hist_attach(fills: pl.DataFrame, prior: pl.DataFrame, vol: pl.DataFrame, series_pin: pl.DataFrame) -> pl.DataFrame:
    """Add regime, window, look-ahead-free alpha sources and GM excesses (cents) to historical
    maker fills. ``h_c`` = own-side mid proxy at placement minus our price."""
    f = fills.with_columns((pl.col("us") // (BUCKET_S * _US)).alias("bucket"),
                           pl.col("ticker").str.split("-").list.first().alias("series"))
    f = (f.join(prior, on=["ticker", "bucket"], how="left")
         .join(vol.select("ticker", "bucket", "volatile", "in_play"), on=["ticker", "bucket"], how="left")
         .join(series_pin, on="series", how="left")
         .with_columns(pl.col("volatile").fill_null(False), pl.col("in_play").fill_null(False),
                       pl.when(pl.col("us") < int(HIST_SPLIT.timestamp() * _US)).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("window")))
    mid = (pl.col("ask0") + pl.col("bid0")) / 2
    own = pl.when(pl.col("side") == "yes").then(mid).otherwise(1 - mid)
    f = f.with_columns(mid.alias("mid_dec"), (100 * (own - pl.col("price"))).alias("h_c"),
                       (100 * pl.col("fee")).alias("fee_c"))
    return f.with_columns(
        (pl.col("h_c") - pl.col("pin_exp") * pl.col("move_exp_c") - pl.col("fee_c")).alias("x2_exp"),
        (pl.col("h_c") - 100 * gm_half_binary(pl.col("mid_dec"), pl.col("pin_prevday")) - pl.col("fee_c")).alias("xb_prevday"),
        (pl.col("h_c") - 100 * gm_half_binary(pl.col("mid_dec"), pl.col("pin_series_dev")) - pl.col("fee_c")).alias("xb_series"),
    )


def hist_study(s, *, min_spread: float = 0.05, latency_s: float = 1.0, sample: int = 3,
               report: bool = True) -> Path | dict[str, Any]:
    """The historical half: PIN per ticker-day and in-play window; all taker orders as maker
    fills and the passive tape maker, both scored by GM excess and alpha terciles (cut points
    from dev) to settlement and to exit proxies; dev = before October, holdout = October."""
    cache = s.data_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    markets = pl.read_parquet(s.tables_dir / "kalshi_markets.parquet",
                              columns=["ticker", "event_ticker", "series_ticker", "result", "fee_type", "fee_multiplier",
                                       "expected_expiration_time"])
    orders = hist_orders(pl.scan_parquet(s.tables_dir / "kalshi_trades.parquet")).collect()
    markets = markets.join(orders.select("ticker").unique(), on="ticker")
    play = in_play_bounds(markets)
    pin_path = cache / "info_mm_hist_pin.parquet"
    if pin_path.exists():
        pins = pl.read_parquet(pin_path)
    else:
        pins = hist_units_pin(orders, play)
        pins.write_parquet(pin_path)
    series_pin = (pins.filter((pl.col("unit") == "ticker_day") & (pl.col("window") == "dev"))
                  .group_by("series").agg(pl.col("pin").median().alias("pin_series_dev"), pl.len().alias("_n"))
                  .filter(pl.col("_n") >= 5).drop("_n"))
    prior = hist_alpha_prior(orders, pins)
    vol = hist_volatile(orders, play)
    # (b) the passive tape maker
    pm = print_maker(orders, min_spread=min_spread, latency_s=latency_s)
    state = tape_state(orders).select("ticker", "us", "buy", "vwap", "bid_px", "ask_px", "mid_px")
    del orders
    pm = hist_attach(hist_markouts(pm, state, markets), prior, vol, series_pin)
    # (a) every taker order as a 1-lot maker fill on the other side, at the order's VWAP; quote
    # proxies from before the order; prices inside [0.05, 0.95]; a deterministic 1/``sample``
    # subsample (hash of ticker and time) keeps memory bounded.
    allm = state.with_columns(
        pl.col("ask_px").shift(1).over("ticker").alias("ask0"), pl.col("bid_px").shift(1).over("ticker").alias("bid0"),
        pl.when(pl.col("buy")).then(pl.lit("no")).otherwise(pl.lit("yes")).alias("side"),
        pl.when(pl.col("buy")).then(1 - pl.col("vwap")).otherwise(pl.col("vwap")).alias("price"),
    ).filter(pl.col("ask0").is_not_null() & pl.col("bid0").is_not_null() & (pl.col("ask0") > pl.col("bid0"))
             & pl.col("price").is_between(0.05, 0.95)
             & (pl.struct("ticker", "us").hash(11) % sample == 0)).select("ticker", "us", "side", "price", "ask0", "bid0")
    keep = ["ticker", "event", "series", "us", "side", "price", "qty", "fee", "window", "volatile", "in_play",
            "mo_60s", "mo_300s", "mx_60s", "mx_300s", "settle", "h_c", "mid_dec", "pin_exp", "pin_prevday",
            "pin_series_dev", "move_exp_c", "x2_exp", "xb_prevday", "xb_series"]
    allm = hist_attach(hist_markouts(allm, state, markets), prior, vol, series_pin).select(keep)
    pm = pm.select(keep + ["ask0", "bid0"])
    out = {"pins": pins, "all_orders": allm, "print_maker": pm, "series_pin": series_pin}
    allm.write_parquet(cache / "info_mm_hist_allmaker.parquet")
    pm.write_parquet(cache / "info_mm_hist_printmaker.parquet")
    if not report:
        return out
    return hist_report(s, out)


def tercile_groups(f: pl.DataFrame, col: str) -> dict[str, pl.Expr]:
    """Groups by ``col`` terciles with cut points from the dev window only."""
    d = f.filter((pl.col("window") == "dev") & pl.col(col).is_not_null())[col]
    if d.len() < 3:
        return {}
    q1, q2 = d.quantile(1 / 3), d.quantile(2 / 3)
    return {f"{col}<={q1:.3f}": pl.col(col) <= q1, f"{col} mid": pl.col(col).is_between(q1, q2, closed="right"),
            f"{col}>{q2:.3f}": pl.col(col) > q2}


def pick_rules(f: pl.DataFrame, families: dict[str, dict[str, pl.Expr]], metric: str,
               min_fills: int = 30) -> pl.DataFrame:
    """For each family of candidate groups (an alpha source's terciles or a GM-excess sign),
    choose the group with the best dev ``metric`` (at least ``min_fills`` dev fills) and report
    it on dev and holdout next to the unfiltered maker: the out-of-sample test of an
    alpha-aware rule."""
    rows = []
    base = split_table(f, [metric]).filter(pl.col("regime") == "all")
    b = {r["window"]: r for r in base.iter_rows(named=True)}
    for fam, groups in families.items():
        if not groups:
            continue
        t = split_table(f, [metric], groups=groups).filter(pl.col("regime") == "all")
        dev = t.filter((pl.col("window") == "dev") & (pl.col("fills") >= min_fills) & pl.col(metric).is_not_null())
        if dev.is_empty():
            continue
        best = dev.sort(metric, descending=True).row(0, named=True)
        hold = t.filter((pl.col("window") == "holdout") & (pl.col("group") == best["group"])).row(0, named=True)
        rows.append({"family": fam, "chosen_on_dev": best["group"], "metric": metric,
                     "dev": best[metric], "t_dev": best[f"t_{metric}"], "dev_fills": best["fills"],
                     "holdout": hold[metric], "t_holdout": hold[f"t_{metric}"], "holdout_fills": hold["fills"],
                     "holdout_events": hold["events"],
                     "unfiltered_dev": b["dev"][metric], "unfiltered_holdout": b["holdout"][metric]})
    return pl.DataFrame(rows)


def hist_report(s, out: dict[str, Any]) -> Path:
    from albert.reports import Report

    rep = Report(s.reports_dir, "research-info-mm-hist")
    pins = out["pins"]
    desc = (pins.group_by("unit", "window").agg(
        pl.len().alias("units"), pl.col("orders").sum().alias("orders"),
        *[pl.col(c).quantile(q).round(4).alias(f"{c}_{n}") for c in ("pin", "alpha", "pin_mom", "vpin", "lam_c_per_100", "sign_ac1")
          for n, q in (("p25", 0.25), ("med", 0.5), ("p75", 0.75))]).sort("unit", "window"))
    rep.table("PIN per ticker-day / in-play window (quartiles)", desc)
    top = (pins.filter(pl.col("unit") == "ticker_day").group_by("series").agg(
        pl.len().alias("ticker_days"), pl.col("orders").sum(), pl.col("pin").median().round(3), pl.col("pin_mom").median().round(3),
        pl.col("lam_c_per_100").median().round(3), pl.col("sign_ac1").median().round(3)).sort("orders", descending=True))
    rep.table("PIN by series (ticker-days, medians)", top, max_rows=25)
    cols = ["mo_60s", "mo_300s", "mx_60s", "mx_300s", "settle"]
    for name, f in (("All taker orders as 1-lot maker fills", out["all_orders"]),
                    ("Passive tape maker (1 tick inside the print spread, fills strictly through)", out["print_maker"])):
        groups = {"all": pl.lit(True), "x2_exp>0": pl.col("x2_exp") > 0, "x2_exp<=0": pl.col("x2_exp") <= 0,
                  "xb_prevday>0": pl.col("xb_prevday") > 0, "xb_prevday<=0": pl.col("xb_prevday") <= 0}
        groups |= tercile_groups(f, "pin_exp") | tercile_groups(f, "pin_prevday") | tercile_groups(f, "pin_series_dev")
        rep.table(name, split_table(f, cols, groups=groups), max_rows=120)
        fams = {"pin_exp terciles": tercile_groups(f, "pin_exp"), "pin_prevday terciles": tercile_groups(f, "pin_prevday"),
                "pin_series_dev terciles": tercile_groups(f, "pin_series_dev"),
                "x2_exp sign": {"x2_exp>0": pl.col("x2_exp") > 0, "x2_exp<=0": pl.col("x2_exp") <= 0},
                "xb_prevday sign": {"xb_prevday>0": pl.col("xb_prevday") > 0, "xb_prevday<=0": pl.col("xb_prevday") <= 0}}
        rules = pl.concat([pick_rules(f, fams, m) for m in ("mx_300s", "settle")], how="diagonal_relaxed")
        rep.table(f"{name}: alpha-aware rules chosen on dev, scored on holdout (cents)", rules)
    return rep.save()
