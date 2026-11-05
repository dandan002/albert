"""Curve-fitted fair-value estimators on recorded Kalshi L2 books: do they predict the price,
and do they make a maker (or a taker) better?

Estimators, evaluated after every top-of-book change (price or size) of a market, causally
(``FairState`` sees only events up to the decision time; every parameter is fitted on dev only):

* ``mid``; ``wmid`` (size-weighted mid ``(b*as + a*bs)/(bs+as)``);
* ``micro``: Stoikov (2018) micro-price. States are (imbalance bucket, spread bucket); the
  discrete Markov model of state transitions and mid changes between consecutive top-of-book
  updates is fitted on dev (symmetrised), and the adjustment ``G = sum_k B^k G1`` with
  ``G1 = (I-Q)^-1 r``, ``B = (I-Q)^-1 T`` is solved once (``fit_micro``);
* ``ema_mid_<hl>`` / ``ema_trd_<hl>``: time-decayed EWMA of the mid path, and a time-decayed
  size-weighted average of trade prints shrunk to the mid with weight one contract;
* ``kal``: local-level Kalman filter on logit(price); observations are mid changes (noise sd
  ``cm * spread``, floored at half a tick) and trade prints (noise sd ``st``); process variance
  ``q`` per second. ``(q, cm, st)`` maximise the innovation likelihood on dev (``fit_kalman``);
* ``ou``: mean reversion of the mid toward a slow EWMA (half-life ``SLOW_HL``): predicted change
  at horizon h is ``(slow - mid) * (1 - 2^(-h/HL))`` with HL fitted per regime (spread band x
  previous bucket volatile) on dev (``fit_ou``);
* ``ridge``: ridge regression of the future mid change at each horizon on all the above
  deviations from the mid, exponentially-decayed order-flow imbalance (OFI) and signed trade
  volume, and their interactions with a wide-spread (>= 5c) dummy (``fit_ridge``).

Every estimator is expressed as a predicted change of the mid ``p = fair - mid``.

Tests:

1. Prediction (``prediction_table``): samples thinned to at most one per market per second.
   Targets: the mid change and the *exit-price* change (the change of the best bid if ``p > 0``,
   i.e. the price a YES holder sells at, of the best ask if ``p < 0``) at 1/10/60/300 s. Reported:
   R^2 against "no change" (``1 - SSE/sum y^2``) with ``p`` used raw (beta = 1) and scaled by
   the dev-fitted beta, the size-weighted hit rate ``sum |y| 1[sign p = sign y] / sum |y|``, and
   the mean signed move when ``|p| >= 0.5c`` with t clustered by event.
2. Taker (``taker_trades``): buy YES when ``fair - ask >= fee + thr`` (NO symmetrically), IOC at the
   seen price after ``latency``, displayed size >= 1, exit by selling at the bid after h seconds,
   taker fee ``0.07 P (1-P)`` (rounded up to $0.0001) on both legs; one open trade per market.
3. Maker (``FairQuoter``): the baseline ``Quoter(improve=True, min_spread=0.05,
   cooldown_s=300)`` with a side dropped when the fair value says the quote has less than ``thr``
   of edge (``mode='drop'``; with ``thr > 0`` this is partly a spread filter), dropped when the
   fair value moved against that side by ``thr`` relative to the mid (``mode='lean'``), or with
   both quotes re-centred on the fair value at the baseline distance (``mode='skew'``). Scored by
   ``backtest.l2.markouts`` (mid and exit price, fees deducted) and settlement where known.
   ``run_many`` replays all variants in one pass on ``FastSimulator`` (the stock Simulator with
   a shared best-price cache; tests check identical orders and fills).

Rerun on other hours: ``fair_value_study(settings(), start=..., split=..., end=...)`` (fits on
``[start, split)``, evaluates on both halves; hours as ``YYYY-MM-DDTHH``, end exclusive).

Regimes: every sample, trade and fill is split by its *own* 5-minute bucket's ``volatile`` flag
from ``research.volatile`` (an evaluation split, not a decision input). The OU regime uses
``volatile_prev`` (the previous bucket), which is known at decision time.
"""

import hashlib
import heapq
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

from albert.backtest.l2 import (EPS, Event, Market, Quoter, SimConfig, Simulator, iter_events,
                                markouts, stream_files)

NS = 10**9
HORIZONS_S = (1, 10, 60, 300)
EMA_MID_HL = (10, 60, 300)
EMA_TRD_HL = (60, 300, 1800)
SLOW_HL = 600
FLOW_HL = (10, 60)
N_IMB = 5
N_SPREAD = 7
WIDE = 0.05
BUCKET_NS = 300 * NS
PRICE_FEATS = ("wmid", "micro", "ema_mid_10", "ema_mid_60", "ema_mid_300", "ema_trd_60",
               "ema_trd_300", "ema_trd_1800", "kal", "slow")
FLOW_FEATS = ("ofi_10", "ofi_60", "flow_10", "flow_60")
BASE = PRICE_FEATS + FLOW_FEATS
FEATURES = BASE + tuple(f"{x}_w" for x in BASE)
ESTIMATORS = ("wmid", "micro", "ema_mid_10", "ema_mid_60", "ema_mid_300", "ema_trd_60",
              "ema_trd_300", "ema_trd_1800", "kal", "ou", "ridge")


# ---- parameters ------------------------------------------------------------------------------
@dataclass
class FairParams:
    micro_g: list[list[float]] | None = None    # [spread bucket][imbalance bucket] -> adjustment
    kal_q: float = 1e-4                          # logit variance per second
    kal_cm: float = 0.25                         # mid noise sd as a fraction of the spread
    kal_st: float = 0.02                         # trade print noise sd (price units)
    kal_floor: float = 0.005                     # minimum mid noise sd (half a tick)
    ou_hl: dict[str, float] = field(default_factory=dict)      # regime -> half-life (s)
    ridge: dict[str, list[float]] = field(default_factory=dict)  # str(horizon) -> coefs (FEATURES)

    def tag(self) -> str:
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


def imb_bucket(bs: float, as_: float) -> int:
    """Symmetric imbalance bucket of I = bs/(bs+as): mirrors exactly under I -> 1-I."""
    tot = bs + as_
    u = (bs / tot - 0.5) if tot > 0 else 0.0
    m = 0 if abs(u) < 0.1 else 1 if abs(u) < 0.3 else 2
    return 2 + (m if u > 0 else -m)


def spread_bucket(spread: float) -> int:
    """Spread in cents -> 0..6 for 1, 2, 3, 4, 5-6, 7-10, 11+."""
    c = round(spread * 100)
    return 0 if c <= 1 else 1 if c == 2 else 2 if c == 3 else 3 if c == 4 else 4 if c <= 6 else 5 if c <= 10 else 6


def ou_regime(spread: float, vol_prev: bool) -> str:
    band = "tight" if spread <= 0.02 + EPS else "mid" if spread <= 0.04 + EPS else "wide"
    return f"{band}_{'vol' if vol_prev else 'calm'}"


def _clip(p: float) -> float:
    return min(max(p, 0.01), 0.99)


def kal_step(x: float | None, P: float, kt: int, t: int, z_price: float, r_price: float,
             q: float) -> tuple[float, float, int, float | None]:
    """One local-level Kalman update in logit space. Returns (x, P, t, loglik or None at init)."""
    pz = _clip(z_price)
    z = math.log(pz / (1 - pz))
    if x is None:
        return z, r_price / (pz * (1 - pz)) ** 2, t, None
    P += q * (t - kt) / NS
    px = _clip(1 / (1 + math.exp(-x)))
    R = r_price / (px * (1 - px)) ** 2
    S = P + R
    v = z - x
    K = P / S
    return x + K * v, P * (1 - K), t, -0.5 * (math.log(2 * math.pi * S) + v * v / S)


def _ofi_step(pb, pa, pbs, pas, b, a, bs, as_) -> float:
    """Cont-Kukanov-Stoikov order-flow increment at the top level (YES bid / YES ask)."""
    if None in (pb, pa, b, a):
        return 0.0
    e = 0.0
    e += bs if b >= pb - EPS else 0.0
    e -= pbs if b <= pb + EPS else 0.0
    e -= as_ if a <= pa + EPS else 0.0
    e += pas if a >= pa - EPS else 0.0
    return e


# ---- online state ----------------------------------------------------------------------------
class FairState:
    """Estimator state of one market, updated on top-of-book changes and trade prints."""

    def __init__(self, p: FairParams) -> None:
        self.p = p
        self.t: int | None = None
        self.b: float | None = None
        self.a: float | None = None
        self.bs = self.as_ = 0.0
        self.mid: float | None = None
        self.ema_t: int | None = None
        self.ema: list[float] | None = None
        self.slow: float | None = None
        self.trd_t: int | None = None
        self.tnum = [0.0] * len(EMA_TRD_HL)
        self.tden = [0.0] * len(EMA_TRD_HL)
        self.kx: float | None = None
        self.kP = 0.0
        self.kt = 0
        self.ll = 0.0
        self.nll_obs = 0
        self.ofi = [0.0] * len(FLOW_HL)
        self.ofi_t: int | None = None
        self.flow = [0.0] * len(FLOW_HL)
        self.gross = [0.0] * len(FLOW_HL)
        self.flow_t: int | None = None

    def _kal(self, t: int, z: float, r: float) -> None:
        self.kx, self.kP, self.kt, ll = kal_step(self.kx, self.kP, self.kt, t, z, r, self.p.kal_q)
        if ll is not None:
            self.ll += ll
            self.nll_obs += 1

    def on_quote(self, t: int, b: float | None, a: float | None, bs: float, as_: float) -> None:
        if self.mid is not None and self.ema_t is not None and self.ema is not None:
            dt = (t - self.ema_t) / NS
            for i, hl in enumerate(EMA_MID_HL):
                self.ema[i] += (1 - 2 ** (-dt / hl)) * (self.mid - self.ema[i])
            self.slow += (1 - 2 ** (-dt / SLOW_HL)) * (self.mid - self.slow)
        step = _ofi_step(self.b, self.a, self.bs, self.as_, b, a, bs, as_) if self.t is not None else 0.0
        if self.ofi_t is not None:
            dt = (t - self.ofi_t) / NS
            for i, hl in enumerate(FLOW_HL):
                self.ofi[i] *= 2 ** (-dt / hl)
        for i in range(len(FLOW_HL)):
            self.ofi[i] += step
        self.ofi_t = t
        mid = (a + b) / 2 if a is not None and b is not None else None
        if mid is not None:
            if self.ema is None:
                self.ema = [mid] * len(EMA_MID_HL)
                self.slow = mid
            if mid != self.mid:
                r = (self.p.kal_cm * (a - b)) ** 2 + self.p.kal_floor ** 2
                self._kal(t, mid, r)
        self.ema_t = t
        self.t, self.b, self.a, self.bs, self.as_, self.mid = t, b, a, bs, as_, mid

    def on_trade(self, t: int, sgn: float, qty: float, px: float) -> None:
        if self.trd_t is not None:
            dt = (t - self.trd_t) / NS
            for i, hl in enumerate(EMA_TRD_HL):
                d = 2 ** (-dt / hl)
                self.tnum[i] *= d
                self.tden[i] *= d
        for i in range(len(EMA_TRD_HL)):
            self.tnum[i] += qty * px
            self.tden[i] += qty
        self.trd_t = t
        if self.flow_t is not None:
            dt = (t - self.flow_t) / NS
            for i, hl in enumerate(FLOW_HL):
                d = 2 ** (-dt / hl)
                self.flow[i] *= d
                self.gross[i] *= d
        for i in range(len(FLOW_HL)):
            self.flow[i] += sgn * qty
            self.gross[i] += qty
        self.flow_t = t
        self._kal(t, px, self.p.kal_st ** 2)

    def features(self, now: int) -> dict[str, float] | None:
        """Deviations from the mid of every price estimator, plus flow features, as of ``now``
        (EWMAs integrate the current mid up to ``now``; nothing after ``now`` is used)."""
        mid = self.mid
        if mid is None or self.ema is None:
            return None
        b, a, bs, as_ = self.b, self.a, self.bs, self.as_
        spread = a - b
        f: dict[str, float] = {"mid": mid, "spread": spread, "bid": b, "ask": a, "bsz": bs, "asz": as_}
        f["wmid"] = ((b * as_ + a * bs) / (bs + as_) - mid) if bs + as_ > 0 else 0.0
        g = self.p.micro_g
        f["micro"] = g[spread_bucket(spread)][imb_bucket(bs, as_)] if g else 0.0
        dt = (now - self.ema_t) / NS if self.ema_t is not None else 0.0
        for i, hl in enumerate(EMA_MID_HL):
            e = self.ema[i] + (1 - 2 ** (-dt / hl)) * (mid - self.ema[i])
            f[f"ema_mid_{hl}"] = e - mid
        slow = self.slow + (1 - 2 ** (-dt / SLOW_HL)) * (mid - self.slow)
        f["slow"] = slow - mid
        tdt = (now - self.trd_t) / NS if self.trd_t is not None else 0.0
        for i, hl in enumerate(EMA_TRD_HL):
            d = 2 ** (-tdt / hl)
            num, den = self.tnum[i] * d, self.tden[i] * d
            f[f"ema_trd_{hl}"] = (num + mid) / (den + 1.0) - mid
        f["kal"] = (1 / (1 + math.exp(-self.kx)) - mid) if self.kx is not None else 0.0
        depth = (bs + as_) / 2
        odt = (now - self.ofi_t) / NS if self.ofi_t is not None else 0.0
        fdt = (now - self.flow_t) / NS if self.flow_t is not None else 0.0
        for i, hl in enumerate(FLOW_HL):
            f[f"ofi_{hl}"] = self.ofi[i] * 2 ** (-odt / hl) / (depth + 1.0)
            d = 2 ** (-fdt / hl)
            f[f"flow_{hl}"] = self.flow[i] * d / (self.gross[i] * d + 5.0)
        return f


def predict(p: FairParams, f: dict[str, float], est: str, h: int, vol_prev: bool = False) -> float:
    """Predicted mid change of estimator ``est`` at horizon ``h`` seconds (``fair - mid``)."""
    if est == "mid":
        return 0.0
    if est in PRICE_FEATS:
        return f[est]
    if est == "ou":
        hl = p.ou_hl.get(ou_regime(f["spread"], vol_prev))
        return f["slow"] * (1 - 2 ** (-h / hl)) if hl else 0.0
    if est == "ridge":
        coefs = p.ridge.get(str(h))
        if not coefs:
            return 0.0
        w = 1.0 if f["spread"] >= WIDE - EPS else 0.0
        return (sum(c * f[x] for c, x in zip(coefs, BASE))
                + w * sum(c * f[x] for c, x in zip(coefs[len(BASE):], BASE)))
    raise ValueError(est)


# ---- tapes (top of book + trades), cached per closed hour ------------------------------------
Q_SCHEMA = {"t": pl.Int64, "seq": pl.Int64, "market": pl.Utf8, "bid": pl.Float64,
            "ask": pl.Float64, "bsz": pl.Float64, "asz": pl.Float64}
T_SCHEMA = {"t": pl.Int64, "seq": pl.Int64, "market": pl.Utf8, "sgn": pl.Float64,
            "qty": pl.Float64, "px": pl.Float64}


class Tracker:
    """Top-of-book change detector + ``FairState`` per market, fed from a Simulator's external
    book (the strategy's view). Shared by every strategy of a multi-run pass."""

    def __init__(self, params: FairParams | None = None, track: bool = True) -> None:
        self.params = params or FairParams()
        self.track = track
        self.states: dict[str, FairState] = {}
        self.last: dict[str, tuple] = {}
        self._ev: Event | None = None

    def state(self, market: str) -> FairState:
        st = self.states.get(market)
        if st is None:
            st = self.states[market] = FairState(self.params)
        return st

    def observe(self, sim: Simulator, ev: Event) -> tuple | None:
        """Update from ``ev`` (already applied to ``sim``). Returns ('q', bid, ask, bsz, asz) on a
        top-of-book change, ('t', sgn, qty, px) on a trade, else None. Idempotent per event."""
        if ev is self._ev:
            return None
        self._ev = ev
        if ev.kind == "trade":
            m = ev.msg
            out = ("t", 1.0 if m["taker_side"] == "yes" else -1.0, float(m["count_fp"]),
                   float(m["yes_price_dollars"]))
            if self.track:
                self.state(ev.market).on_trade(ev.recv_ns, *out[1:])
            return out
        if ev.kind not in ("snapshot", "delta") or ev.market not in sim.synced:
            return None
        q = sim.markets[ev.market].quote()
        key = (q["yes_bid"], q["yes_ask"], q["yes_bid_size"], q["yes_ask_size"])
        if self.last.get(ev.market) == key:
            return None
        self.last[ev.market] = key
        if self.track:
            self.state(ev.market).on_quote(ev.recv_ns, key[0], key[1], key[2] or 0.0, key[3] or 0.0)
        return ("q", *key)


def tape_from_events(events: Iterable[Event]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Top-of-book changes (price or size) and trade prints of every market, in event order."""
    sim = Simulator(SimConfig(writes_per_s=None))
    tr = Tracker(track=False)
    q: dict[str, list] = {k: [] for k in Q_SCHEMA}
    t: dict[str, list] = {k: [] for k in T_SCHEMA}
    for seq, ev in enumerate(events):
        sim.apply(ev)
        out = tr.observe(sim, ev)
        if out is None:
            continue
        cols = q if out[0] == "q" else t
        cols["t"].append(ev.recv_ns)
        cols["seq"].append(seq)
        cols["market"].append(ev.market)
        for k, v in zip(list(cols)[3:], out[1:]):
            cols[k].append(v)
    return pl.DataFrame(q, schema=Q_SCHEMA), pl.DataFrame(t, schema=T_SCHEMA)


def hour_files(root: Path, start: str, end: str) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = defaultdict(list)
    for p in sorted((root / "kalshi").glob("*/*.jsonl.zst")):
        if "-rest" in p.name:
            continue
        key = f"{p.parent.name}T{p.name[:2]}"
        if start <= key < end:
            out[key].append(p)
    return dict(sorted(out.items()))


def _closed(files: list[Path]) -> bool:
    hh = files[0].name[:2]
    return not list(files[0].parent.glob(f"{hh}-*.part"))


def load_tapes(root: Path, start: str, end: str, cache: Path) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    """Tapes per hour (closed hours cached as Parquet under ``cache``)."""
    tag = hashlib.md5(str(root.resolve()).encode()).hexdigest()[:6]
    out = {}
    for hour, files in hour_files(root, start, end).items():
        qp = cache / f"fair_value_tape_{tag}_{hour}_q.parquet"
        tp = cache / f"fair_value_tape_{tag}_{hour}_t.parquet"
        if qp.exists() and tp.exists():
            out[hour] = (pl.read_parquet(qp), pl.read_parquet(tp))
            continue
        stats: dict[str, int] = defaultdict(int)
        q, t = tape_from_events(iter_events(files, stats))
        if _closed(files):
            cache.mkdir(parents=True, exist_ok=True)
            q.write_parquet(qp)
            t.write_parquet(tp)
        out[hour] = (q, t)
        _log(f"tape {hour}: {q.height} quotes, {t.height} trades")
    return out


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---- fitting ---------------------------------------------------------------------------------
def fit_micro(q: pl.DataFrame, min_obs: int = 30) -> tuple[list[list[float]], dict[str, Any]]:
    """Stoikov micro-price adjustment G(spread bucket, imbalance bucket) from consecutive
    top-of-book updates (symmetrised: every observation is also added mirrored)."""
    d = (q.sort("market", "t", "seq")
         .with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
         .with_columns(pl.col("mid").shift(-1).over("market").alias("mid1"),
                       pl.col("bid").shift(-1).over("market").alias("bid1"),
                       pl.col("ask").shift(-1).over("market").alias("ask1"),
                       pl.col("bsz").shift(-1).over("market").alias("bsz1"),
                       pl.col("asz").shift(-1).over("market").alias("asz1"))
         .filter(pl.col("mid").is_not_null() & pl.col("mid1").is_not_null()))
    n = N_IMB * N_SPREAD
    cnt = np.zeros(n)
    Q = np.zeros((n, n))
    T = np.zeros((n, n))
    r = np.zeros(n)
    for b, a, bs, as_, b1, a1, bs1, as1 in d.select("bid", "ask", "bsz", "asz", "bid1", "ask1",
                                                     "bsz1", "asz1").iter_rows():
        sb, ib = spread_bucket(a - b), imb_bucket(bs or 0.0, as_ or 0.0)
        sb1, ib1 = spread_bucket(a1 - b1), imb_bucket(bs1 or 0.0, as1 or 0.0)
        dm = round((a1 + b1) / 2 - (a + b) / 2, 6)
        for x, y, dd in ((sb * N_IMB + ib, sb1 * N_IMB + ib1, dm),
                         (sb * N_IMB + N_IMB - 1 - ib, sb1 * N_IMB + N_IMB - 1 - ib1, -dm)):
            cnt[x] += 1
            if dd == 0:
                Q[x, y] += 1
            else:
                T[x, y] += 1
                r[x] += dd
    ok = cnt >= min_obs
    ok &= T.sum(axis=1) > 0
    den = np.where(ok, cnt, 1.0)
    Q = np.where(ok[:, None], Q / den[:, None], 0.0)
    T = np.where(ok[:, None], T / den[:, None], 0.0)
    r = np.where(ok, r / den, 0.0)
    inv = np.linalg.inv(np.eye(n) - Q)
    G1 = inv @ r
    B = inv @ T
    G, term = G1.copy(), G1.copy()
    for k in range(500):
        term = B @ term
        G += term
        if np.abs(term).max() < 1e-8:
            break
    g = G.reshape(N_SPREAD, N_IMB)
    diag = {"transitions": int(d.height), "states_used": int(ok.sum()), "iterations": k + 1,
            "G1_c": np.round(G1.reshape(N_SPREAD, N_IMB) * 100, 3).tolist()}
    return [[round(float(v), 6) for v in row] for row in g], diag


def kalman_obs(q: pl.DataFrame, tr: pl.DataFrame) -> dict[str, list[tuple[int, float, float]]]:
    """Kalman observations per market in event order: (t, price, noise kind) where kind is the
    spread for a mid change and -1 for a trade print."""
    mids = (q.sort("market", "t", "seq")
            .with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
            .with_columns(pl.col("mid").shift(1).over("market").alias("prev"))
            .filter(pl.col("mid").is_not_null() & (pl.col("prev").is_null() | (pl.col("mid") != pl.col("prev"))))
            .select("t", "seq", "market", pl.col("mid").alias("z"), (pl.col("ask") - pl.col("bid")).alias("k")))
    trs = tr.select("t", "seq", "market", pl.col("px").alias("z"), pl.lit(-1.0).alias("k"))
    allo = pl.concat([mids, trs]).sort("market", "t", "seq")
    out: dict[str, list] = defaultdict(list)
    for t, _, m, z, k in allo.iter_rows():
        out[m].append((t, z, k))
    return out


def kalman_loglik(obs: dict[str, list[tuple[int, float, float]]], q: float, cm: float, st: float,
                  floor: float = 0.005) -> tuple[float, int]:
    total, n = 0.0, 0
    st2, fl2 = st * st, floor * floor
    for seq in obs.values():
        x, P, kt = None, 0.0, 0
        for t, z, k in seq:
            r = st2 if k < 0 else (cm * k) ** 2 + fl2
            x, P, kt, ll = kal_step(x, P, kt, t, z, r, q)
            if ll is not None:
                total += ll
                n += 1
    return total, n


def fit_kalman(obs: dict[str, list[tuple[int, float, float]]]) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Grid then coordinate search (log steps) for the innovation-likelihood maximum."""
    best, best_ll = None, -math.inf
    tried: dict[tuple, float] = {}

    def ll(th: tuple[float, float, float]) -> float:
        key = tuple(round(v, 12) for v in th)
        if key not in tried:
            tried[key] = kalman_loglik(obs, *th)[0]
        return tried[key]

    for q in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        for cm in (0.05, 0.15, 0.3, 0.6):
            for st in (0.005, 0.01, 0.02, 0.04):
                v = ll((q, cm, st))
                if v > best_ll:
                    best, best_ll = (q, cm, st), v
    step = 2.0
    while step > 1.1:
        improved = False
        for i in range(3):
            for f in (step, 1 / step):
                th = list(best)
                th[i] *= f
                v = ll(tuple(th))
                if v > best_ll + 1e-6:
                    best, best_ll, improved = tuple(th), v, True
        if not improved:
            step = math.sqrt(step)
    n = kalman_loglik(obs, *best)[1]
    return best, {"loglik": best_ll, "obs": n, "evaluations": len(tried)}


def fit_ou(dev: pl.DataFrame) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Half-life per regime minimising the squared error of ``(slow - mid)(1 - 2^(-h/HL))`` as a
    forecast of the mid change, jointly over the horizons."""
    grid = [0.5 * 1.25 ** k for k in range(60)]   # 0.5 s .. ~280 h
    out, diag = {}, []
    for reg, g in dev.group_by("regime"):
        reg = reg[0]
        x = g["slow"].to_numpy()
        best, best_sse = None, math.inf
        for hl in grid:
            sse = 0.0
            for h in HORIZONS_S:
                y = g[f"y_mid_{h}"].to_numpy()
                ok = ~np.isnan(y)
                sse += float(((y[ok] - x[ok] * (1 - 2 ** (-h / hl))) ** 2).sum())
            if sse < best_sse:
                best, best_sse = hl, sse
        out[reg] = best
        diag.append({"regime": reg, "samples": g.height, "half_life_s": round(best, 2)})
    return out, sorted(diag, key=lambda r: r["regime"])


def _design(df: pl.DataFrame) -> np.ndarray:
    base = df.select(BASE).to_numpy()
    w = (df["spread"].to_numpy() >= WIDE - EPS).astype(float)[:, None]
    return np.hstack([base, base * w])


def fit_ridge(dev: pl.DataFrame, lam: float = 10.0) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    """Ridge (standardised features, no intercept) of the mid change at each horizon."""
    X = _design(dev)
    out, diag = {}, []
    for h in HORIZONS_S:
        y = dev[f"y_mid_{h}"].to_numpy()
        ok = ~np.isnan(y)
        Xo, yo = X[ok], y[ok]
        sd = Xo.std(axis=0)
        sd[sd == 0] = 1.0
        Z = Xo / sd
        beta = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ yo) / sd
        out[str(h)] = [float(v) for v in beta]
        diag.append({"h": h, "n": int(ok.sum()),
                     **{f: round(float(b * s), 6) for f, b, s in zip(FEATURES, beta, sd)}})
    return out, diag


# ---- features at every top-of-book update ----------------------------------------------------
def features_from_tape(q: pl.DataFrame, tr: pl.DataFrame, params: FairParams,
                       states: dict[str, FairState], last_thin: dict[str, int],
                       thin_ns: int = NS) -> pl.DataFrame:
    """Replay a tape through ``FairState`` (``states`` carries over between hours) and return
    one row per two-sided top-of-book update with every feature and a ``thin`` flag."""
    cols = ["t", "market", "thin", "mid", "spread", "bid", "ask", "bsz", "asz", *BASE]
    out: dict[str, list] = {c: [] for c in cols}
    qi = ((r[0], r[1], 0, r) for r in q.sort("t", "seq").iter_rows())
    ti = ((r[0], r[1], 1, r) for r in tr.sort("t", "seq").iter_rows())
    for t, _, kind, r in heapq.merge(qi, ti):
        m = r[2]
        st = states.get(m)
        if st is None:
            st = states[m] = FairState(params)
        if kind == 1:
            st.on_trade(t, r[3], r[4], r[5])
            continue
        st.on_quote(t, r[3], r[4], r[5] or 0.0, r[6] or 0.0)
        f = st.features(t)
        if f is None:
            continue
        thin = t - last_thin.get(m, -(2**62)) >= thin_ns
        if thin:
            last_thin[m] = t
        out["t"].append(t)
        out["market"].append(m)
        out["thin"].append(thin)
        for c in cols[3:]:
            out[c].append(f[c])
    schema = {c: pl.Float64 for c in cols}
    schema.update({"t": pl.Int64, "market": pl.Utf8, "thin": pl.Boolean})
    return pl.DataFrame(out, schema=schema)


def with_targets(samples: pl.DataFrame, quotes: pl.DataFrame, end_ns: int,
                 horizons: Iterable[int] = HORIZONS_S) -> pl.DataFrame:
    """Add mid / bid / ask changes at each horizon (book live at t+h; null past ``end_ns`` or
    when the book is one-sided then)."""
    qq = quotes.select("t", "market", "bid", "ask").sort("t")
    out = samples.sort("t")
    for h in horizons:
        fut = (out.select("t", "market").with_columns((pl.col("t") + h * NS).alias("th")).sort("th")
               .join_asof(qq.rename({"t": "tq", "bid": "b_h", "ask": "a_h"}), left_on="th",
                          right_on="tq", by="market", strategy="backward")
               .select("t", "market", "th", "b_h", "a_h"))
        out = out.join(fut, on=["t", "market"], how="left").with_columns(
            pl.when(pl.col("th") <= end_ns).then(pl.col("b_h") - pl.col("bid")).alias(f"y_bid_{h}"),
            pl.when(pl.col("th") <= end_ns).then(pl.col("a_h") - pl.col("ask")).alias(f"y_ask_{h}"),
        ).with_columns(((pl.col(f"y_bid_{h}") + pl.col(f"y_ask_{h}")) / 2).alias(f"y_mid_{h}")
                       ).drop("th", "b_h", "a_h")
    return out


def add_regimes(df: pl.DataFrame, buckets: pl.DataFrame) -> pl.DataFrame:
    """Own-bucket ``volatile`` (evaluation split) and ``volatile_prev`` (decision-time)."""
    b = buckets.select("market", "bucket_ns", "volatile", "volatile_prev")
    return (df.with_columns((pl.col("t") // BUCKET_NS * BUCKET_NS).alias("bucket_ns"))
            .join(b, on=["market", "bucket_ns"], how="left")
            .with_columns(pl.col("volatile").fill_null(False), pl.col("volatile_prev").fill_null(False)))


def regime_expr() -> pl.Expr:
    """Vectorised ``ou_regime`` (needs ``spread`` and ``volatile_prev``)."""
    band = (pl.when(pl.col("spread") <= 0.02 + EPS).then(pl.lit("tight"))
            .when(pl.col("spread") <= 0.04 + EPS).then(pl.lit("mid")).otherwise(pl.lit("wide")))
    return pl.concat_str([band, pl.when(pl.col("volatile_prev")).then(pl.lit("vol")).otherwise(pl.lit("calm"))],
                         separator="_")


def pred_expr(params: FairParams, est: str, h: int) -> pl.Expr:
    """Vectorised ``predict`` (same formulas) over a feature frame with ``volatile_prev``."""
    if est in PRICE_FEATS:
        return pl.col(est)
    if est == "ou":
        hl = regime_expr().replace_strict(params.ou_hl, default=None, return_dtype=pl.Float64)
        return (pl.col("slow") * (1 - pl.lit(2.0).pow(-h / hl))).fill_null(0.0)
    if est == "ridge":
        coefs = params.ridge.get(str(h)) or [0.0] * len(FEATURES)
        w = (pl.col("spread") >= WIDE - EPS).cast(pl.Float64)
        lin = pl.sum_horizontal([pl.col(x) * c for c, x in zip(coefs, BASE)])
        lin_w = pl.sum_horizontal([pl.col(x) * c for c, x in zip(coefs[len(BASE):], BASE)])
        return lin + w * lin_w
    raise ValueError(est)


# ---- prediction test -------------------------------------------------------------------------
def _clustered(y: np.ndarray, groups: np.ndarray) -> tuple[float | None, float | None, int]:
    if y.size == 0:
        return None, None, 0
    uniq, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=y)
    n = np.bincount(inv)
    mean = s.sum() / n.sum()
    k = uniq.size
    if k < 2:
        return float(mean), None, int(k)
    se = math.sqrt(k / (k - 1) * ((s - mean * n) ** 2).sum()) / n.sum()
    return float(mean), (float(mean / se) if se > 0 else None), int(k)


def _metrics(p: np.ndarray, y: np.ndarray, beta: float, events: np.ndarray,
             min_pred: float) -> dict[str, Any]:
    ss0 = float((y * y).sum())
    if ss0 <= 0:
        return {}
    nz = (p != 0) & (y != 0)
    ay = np.abs(y[nz])
    sig = np.abs(p) >= min_pred - EPS
    mean, t, k = _clustered(np.sign(p[sig]) * y[sig], events[sig])
    return {"r2_raw": 1 - float(((y - p) ** 2).sum()) / ss0,
            "r2_cal": 1 - float(((y - beta * p) ** 2).sum()) / ss0,
            "hit_w": float((ay * (np.sign(p[nz]) == np.sign(y[nz]))).sum() / ay.sum()) if ay.sum() > 0 else None,
            "n_sig": int(sig.sum()), "edge_c": None if mean is None else 100 * mean, "t_event": t,
            "events": k, "mean_abs_p_c": 100 * float(np.abs(p).mean()) if p.size else None}


def _beta(d: pl.DataFrame, target: str, h: int, split_ns: int) -> float:
    """Dev slope through the origin, from samples whose horizon ends before the split."""
    dv = d.filter(~pl.col("holdout") & (pl.col("t") + h * NS < split_ns))
    pp, yy = dv["p"].to_numpy(), dv[f"y_{target}"].to_numpy()
    return float((pp * yy).sum() / (pp * pp).sum()) if (pp * pp).sum() > 0 else 0.0


def band_expr() -> pl.Expr:
    return (pl.when(pl.col("spread") <= 0.02 + EPS).then(pl.lit("1-2c"))
            .when(pl.col("spread") <= 0.04 + EPS).then(pl.lit("3-4c")).otherwise(pl.lit(">=5c")))


def prediction_table(samples: pl.DataFrame, params: FairParams, split_ns: int,
                     estimators: Iterable[str] = ESTIMATORS, min_pred: float = 0.005,
                     by: str = "regime") -> pl.DataFrame:
    """R^2 vs no change (raw and dev-beta), size-weighted hit rate and the mean signed move when
    ``|p| >= min_pred``, per estimator x horizon x target x split x group. ``by='regime'``:
    all / volatile / calm (own bucket), one dev beta for all; ``by='band'``: quoted spread at
    the sample (1-2c / 3-4c / >=5c), dev beta fitted within each band."""
    rows = []
    base = samples.with_columns((pl.col("t") >= split_ns).alias("holdout"),
                                pl.col("market").str.replace(r"-[^-]+$", "").alias("event"),
                                band_expr().alias("band"))
    groups = (["all", "volatile", "calm"] if by == "regime" else ["1-2c", "3-4c", ">=5c"])
    for est in estimators:
        for h in HORIZONS_S:
            d = base.with_columns(pred_expr(params, est, h).alias("p"))
            d = d.with_columns(
                pl.col(f"y_mid_{h}").alias("y_mid"),
                pl.when(pl.col("p") > 0).then(pl.col(f"y_bid_{h}"))
                .when(pl.col("p") < 0).then(pl.col(f"y_ask_{h}"))
                .otherwise(pl.col(f"y_mid_{h}")).alias("y_exit"),
            ).select("t", "p", "y_mid", "y_exit", "holdout", "volatile", "event", "band")
            for target in ("mid", "exit"):
                dt = d.filter(pl.col(f"y_{target}").is_not_null())
                beta_all = _beta(dt, target, h, split_ns)
                for grp in groups:
                    if by == "regime":
                        g0 = dt if grp == "all" else dt.filter(pl.col("volatile") == (grp == "volatile"))
                        beta = beta_all
                    else:
                        g0 = dt.filter(pl.col("band") == grp)
                        beta = _beta(g0, target, h, split_ns)
                    for split in ("dev", "holdout"):
                        g = g0.filter(pl.col("holdout") == (split == "holdout"))
                        p, y = g["p"].to_numpy(), g[f"y_{target}"].to_numpy()
                        rows.append({"est": est, "h": h, "target": target, "split": split, by: grp,
                                     "n": int(y.size), "beta_dev": round(beta, 4),
                                     **_metrics(p, y, beta, g["event"].to_numpy(), min_pred)})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- taker test ------------------------------------------------------------------------------
def _fee_expr(price: pl.Expr, mult: pl.Expr) -> pl.Expr:
    return (0.07 * mult * price * (1 - price) * 1e4).round(9).ceil() / 1e4


def taker_candidates(feat: pl.DataFrame, params: FairParams, ests: Iterable[str],
                     exits_s: Iterable[int], fee_mult: dict[str, float], min_thr: float = 0.0) -> pl.DataFrame:
    """Updates where some estimator's fair value clears the ask (bid) by fee + ``min_thr``."""
    f = feat.with_columns(pl.col("market").str.replace(r"-.*$", "")
                          .replace_strict(fee_mult, default=1.0, return_dtype=pl.Float64).alias("mult"))
    out = []
    for est in ests:
        for h in exits_s:
            d = f.with_columns(pred_expr(params, est, h).alias("p")).with_columns(
                (pl.col("mid") + pl.col("p") - pl.col("ask") - _fee_expr(pl.col("ask"), pl.col("mult"))).alias("e_yes"),
                (pl.col("bid") - pl.col("mid") - pl.col("p") - _fee_expr(1 - pl.col("bid"), pl.col("mult"))).alias("e_no"),
            ).with_columns(pl.max_horizontal("e_yes", "e_no").alias("edge"),
                           pl.when(pl.col("e_yes") >= pl.col("e_no")).then(1).otherwise(-1).alias("dir"))
            out.append(d.filter(pl.col("edge") >= min_thr - EPS)
                       .select("t", "market", "bid", "ask", "mult", "edge", "dir", "volatile",
                               pl.lit(est).alias("est"), pl.lit(h).alias("h")))
    return pl.concat(out) if out else pl.DataFrame()


def taker_trades(cands: pl.DataFrame, quotes: pl.DataFrame, end_ns: int, *, thrs=(0.0, 0.01, 0.02),
                 latencies_ms=(50.0, 100.0), results: dict[str, float | None] | None = None) -> pl.DataFrame:
    """De-cluster (one open trade per market and strategy: next entry only after the exit),
    fill IOC at the seen price after the latency with displayed size >= 1, exit at the bid."""
    if cands.is_empty():
        return pl.DataFrame()
    picked = []
    for (est, h), g in cands.sort("t").group_by(["est", "h"], maintain_order=True):
        for thr in thrs:
            busy: dict[str, int] = {}
            keep = []
            for i, (t, m, e) in enumerate(g.select("t", "market", "edge").iter_rows()):
                if e < thr - EPS or t < busy.get(m, -(2**62)):
                    continue
                busy[m] = t + h * NS
                keep.append(i)
            picked.append(g.select(pl.all().gather(keep)).with_columns(pl.lit(thr).alias("thr")))
    tr = pl.concat(picked)
    qq = quotes.select("t", "market", "bid", "ask", "bsz", "asz").sort("t")
    rows = []
    for lat in latencies_ms:
        ln = int(lat * 1e6)
        d = (tr.with_columns((pl.col("t") + ln).alias("te")).sort("te")
             .join_asof(qq.rename({"t": "tq", "bid": "b1", "ask": "a1", "bsz": "bs1", "asz": "as1"}),
                        left_on="te", right_on="tq", by="market", strategy="backward")
             .with_columns((pl.col("te") + pl.col("h") * NS).alias("tx")).sort("tx")
             .join_asof(qq.rename({"t": "tq2", "bid": "b2", "ask": "a2"}).select("tq2", "market", "b2", "a2"),
                        left_on="tx", right_on="tq2", by="market", strategy="backward"))
        d = d.with_columns(
            pl.when((pl.col("dir") > 0) & (pl.col("a1") <= pl.col("ask") + EPS) & (pl.col("as1") >= 1 - EPS))
            .then(pl.col("a1"))
            .when((pl.col("dir") < 0) & (pl.col("b1") >= pl.col("bid") - EPS) & (pl.col("bs1") >= 1 - EPS))
            .then(1 - pl.col("b1")).alias("entry"),
            pl.when(pl.col("tx") > end_ns).then(None)
            .when(pl.col("dir") > 0).then(pl.col("b2")).otherwise(1 - pl.col("a2")).alias("exit"),
        ).with_columns(
            (pl.col("exit") - pl.col("entry") - _fee_expr(pl.col("entry"), pl.col("mult"))
             - _fee_expr(pl.col("exit"), pl.col("mult"))).alias("pnl"),
            (pl.col("exit") - pl.col("entry")).alias("gross"),
            pl.lit(lat).alias("latency_ms"),
        )
        if results:
            d = d.with_columns(pl.col("market").replace_strict(results, default=None, return_dtype=pl.Float64)
                               .alias("res")).with_columns(
                (pl.when(pl.col("dir") > 0).then(pl.col("res")).otherwise(1 - pl.col("res"))
                 - pl.col("entry") - _fee_expr(pl.col("entry"), pl.col("mult"))).alias("settle_pnl"))
        else:
            d = d.with_columns(pl.lit(None, dtype=pl.Float64).alias("settle_pnl"))
        rows.append(d.select("t", "market", "est", "h", "thr", "latency_ms", "dir", "edge", "volatile",
                             "entry", "exit", "gross", "pnl", "settle_pnl"))
    return pl.concat(rows)


def taker_summary(trades: pl.DataFrame, split_ns: int) -> pl.DataFrame:
    rows = []
    if trades.is_empty():
        return pl.DataFrame()
    d = trades.with_columns((pl.col("t") >= split_ns).alias("holdout"),
                            pl.col("market").str.replace(r"-[^-]+$", "").alias("event"))
    for key, g in d.group_by(["est", "h", "thr", "latency_ms"], maintain_order=True):
        for split in ("dev", "holdout"):
            gs = g.filter(pl.col("holdout") == (split == "holdout"))
            for reg in ("all", "volatile", "calm"):
                gr = gs if reg == "all" else gs.filter(pl.col("volatile") == (reg == "volatile"))
                filled = gr.filter(pl.col("entry").is_not_null())
                done = filled.filter(pl.col("pnl").is_not_null())
                mean, t, k = _clustered(done["pnl"].to_numpy(), done["event"].to_numpy())
                gm, _, _ = _clustered(done["gross"].to_numpy(), done["event"].to_numpy())
                st = filled.filter(pl.col("settle_pnl").is_not_null())
                sm, stt, _ = _clustered(st["settle_pnl"].to_numpy(), st["event"].to_numpy())
                rows.append({"est": key[0], "h": key[1], "thr_c": 100 * key[2], "latency_ms": key[3],
                             "split": split, "regime": reg, "signals": gr.height, "filled": filled.height,
                             "trades": done.height, "events": k,
                             "gross_c": None if gm is None else 100 * gm,
                             "pnl_c": None if mean is None else 100 * mean, "t_event": t,
                             "settled": st.height, "settle_pnl_c": None if sm is None else 100 * sm,
                             "t_settle": stt})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- maker overlay ---------------------------------------------------------------------------
def _round_c(x: float) -> float:
    return math.floor(x * 100 + 0.5) / 100


class FairQuoter(Quoter):
    """Baseline ``Quoter`` with a fair-value overlay.

    * ``mode='drop'``: keep a side only if ``fair_side - our_price >= thr`` (YES: fair - p;
      NO: (1 - fair) - p). ``thr < 0`` tolerates quotes up to ``-thr`` on the wrong side of fair.
    * ``mode='lean'``: drop the YES bid when ``fair - mid <= -thr`` and the NO bid when
      ``fair - mid >= thr`` (the estimator moved against that side by ``thr``). Unlike ``drop``
      with ``thr > 0`` this is not also a spread filter.
    * ``mode='skew'``: centre both quotes on the fair value at the baseline's distance from the
      mid (``spread/2 - 1 tick``), rounded to the cent and kept at least a tick from crossing.
    ``mode='base'`` reproduces the baseline (the fair value is ignored)."""

    def __init__(self, *, tracker: Tracker, est: str = "micro", horizon: int = 300,
                 mode: str = "drop", thr: float = 0.0,
                 vol_prev: dict[tuple[str, int], bool] | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.tracker, self.est, self.h, self.mode, self.thr = tracker, est, horizon, mode, thr
        self.vol_prev = vol_prev or {}
        self.dropped = 0

    def fair(self, sim: Simulator, market: str) -> float | None:
        st = self.tracker.states.get(market)
        f = st.features(sim.now) if st is not None else None
        if f is None:
            return None
        vp = self.vol_prev.get((market, sim.now // BUCKET_NS * BUCKET_NS), False)
        return f["mid"] + predict(self.tracker.params, f, self.est, self.h, vp)

    def _wanted(self, sim: Simulator, market: str) -> dict[str, float | None]:
        want = super()._wanted(sim, market)
        if self.mode == "base" or (want["yes"] is None and want["no"] is None):
            return want
        fair = self.fair(sim, market)
        if fair is None:
            return want
        if self.mode == "drop":
            for side in ("yes", "no"):
                p = want[side]
                if p is None:
                    continue
                fv = fair if side == "yes" else 1 - fair
                if fv - p < self.thr - EPS:
                    want[side] = None
                    self.dropped += 1
            return want
        q = sim.markets[market].quote()
        if self.mode == "lean":
            dev = fair - (q["yes_bid"] + q["yes_ask"]) / 2
            for side, bad in (("yes", dev <= -self.thr + EPS), ("no", dev >= self.thr - EPS)):
                if want[side] is not None and bad:
                    want[side] = None
                    self.dropped += 1
            return want
        b, a = q["yes_bid"], q["yes_ask"]
        d = (a - b) / 2 - 0.01
        for side, fv, cap in (("yes", fair, a - 0.01), ("no", 1 - fair, 1 - b - 0.01)):
            if want[side] is None:
                continue
            p = round(min(_round_c(fv - d), cap), 6)
            want[side] = p if 0.0 < p < 1.0 else None
        return want


class CachedMarket(Market):
    """``Market`` whose ``best``/``quote`` (external liquidity only) are cached in ``cache``
    until the next event that can change this market's external book. Simulators replaying
    the same events have identical external books, so they may share one cache."""

    def __init__(self, cache: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._c: dict[str, Any] = {} if cache is None else cache

    def invalidate(self) -> None:
        self._c.clear()

    def best(self, side: str) -> tuple[float, float] | None:
        c = self._c
        if side not in c:
            c[side] = super().best(side)
        return c[side]

    def quote(self) -> dict[str, float | None]:
        q = self._c.get("q")
        if q is None:
            q = self._c["q"] = super().quote()
        return dict(q)


class _Markets(dict):
    def __init__(self, shared: dict[str, dict[str, Any]] | None) -> None:
        super().__init__()
        self.shared = shared

    def __missing__(self, key: str) -> CachedMarket:
        m = self[key] = CachedMarket(None if self.shared is None else self.shared.setdefault(key, {}))
        return m


class FastSimulator(Simulator):
    """``Simulator`` with identical behaviour, faster: external best prices are cached per market
    (only a snapshot or delta of that market changes external sizes; our own chunks never count),
    and finished orders are pruned from the per-market order lists (every reader of those lists
    filters on live statuses). With ``shared``, the caller owns invalidation (``run_many``)."""

    def __init__(self, cfg: SimConfig, shared: dict[str, dict[str, Any]] | None = None) -> None:
        super().__init__(cfg)
        self.markets = _Markets(shared)
        self.owns_cache = shared is None
        self._n = 0

    def apply(self, ev: Event) -> None:
        # Due activations/cancels first, against the book as of the previous event (as the base
        # class does); then this event may change this market's external book.
        self._run_schedule(ev.recv_ns)
        if self.owns_cache and ev.kind in ("snapshot", "delta"):
            self.markets[ev.market].invalidate()
        super().apply(ev)
        self._n += 1
        if self._n % 20000 == 0:
            live = ("pending", "active", "cancelling", "cancel_pending")
            for m, ids in self.by_market.items():
                self.by_market[m] = [i for i in ids if self.orders[i].status in live]


def run_many(events: Iterable[Event], runs: list[tuple[SimConfig, Quoter]], tracker: Tracker) -> list[Simulator]:
    """Replay once for several (config, strategy) pairs. Every Simulator sees the same external
    book (our orders never enter it), so they share the quote history, the best-price cache and
    one ``Tracker``. Per event: every simulator runs its due schedule against the pre-event book,
    then the event's market cache is invalidated once, then the event is applied to each."""
    shared: dict[str, dict[str, Any]] = {}
    sims = [FastSimulator(cfg, shared) for cfg, _ in runs]
    for s in sims[1:]:
        s.quotes = sims[0].quotes
    for ev in events:
        for s in sims:
            s._run_schedule(ev.recv_ns)
        if ev.kind in ("snapshot", "delta"):
            c = shared.get(ev.market)
            if c:
                c.clear()
        for s in sims:
            s.apply(ev)
        tracker.observe(sims[0], ev)
        for s, (_, strat) in zip(sims, runs):
            strat.on_event(s, ev.market)
    return sims


def maker_rows(sim: Simulator, results: dict[str, float | None]) -> list[dict[str, Any]]:
    rows = markouts(sim)
    for r in rows:
        r["event"] = r["market"].rsplit("-", 1)[0]
        res = results.get(r["market"])
        r["settle"] = (None if res is None else
                       (res if r["side"] == "yes" else 1 - res) - r["price"] - r["fee"] / r["qty"])
    return rows


def maker_summary(rows: list[dict[str, Any]], split_ns: int, vol: dict[tuple[str, int], bool],
                  label: dict[str, Any]) -> list[dict[str, Any]]:
    """all / volatile / calm (fill's own bucket) x dev / holdout, contract-weighted, t by event."""
    out = []
    for split in ("dev", "holdout"):
        for reg in ("all", "volatile", "calm"):
            sel = [r for r in rows if (r["t_ns"] >= split_ns) == (split == "holdout")
                   and (reg == "all" or vol.get((r["market"], r["t_ns"] // BUCKET_NS * BUCKET_NS), False)
                        == (reg == "volatile"))]
            o: dict[str, Any] = {**label, "split": split, "regime": reg, "fills": len(sel),
                                 "contracts": round(sum(r["qty"] for r in sel), 2),
                                 "events": len({r["event"] for r in sel})}
            for col in ("mo_60s", "mo_300s", "mx_60s", "mx_300s", "settle"):
                vals = [r for r in sel if r.get(col) is not None]
                y = np.array([(r[col] - (0.0 if col == "settle" else r["fee"] / r["qty"])) * r["qty"]
                              for r in vals])
                w = np.array([r["qty"] for r in vals])
                ev = np.array([r["event"] for r in vals])
                if not len(vals):
                    o[f"{col}_c"] = o[f"t_{col}"] = None
                    continue
                uniq, inv = np.unique(ev, return_inverse=True)
                s, n = np.bincount(inv, weights=y), np.bincount(inv, weights=w)
                mean = s.sum() / n.sum()
                k = uniq.size
                se = math.sqrt(k / (k - 1) * ((s - mean * n) ** 2).sum()) / n.sum() if k > 1 else 0.0
                o[f"{col}_c"] = round(100 * mean, 3)
                o[f"t_{col}"] = round(mean / se, 2) if se > 0 else None
            o["settled"] = sum(1 for r in sel if r.get("settle") is not None)
            o["pnl_mark_mid300_usd"] = round(sum((r["mo_300s"] - r["fee"] / r["qty"]) * r["qty"]
                                                 for r in sel if r.get("mo_300s") is not None), 3)
            out.append(o)
    return out


# ---- top level -------------------------------------------------------------------------------
def _results(tickers: list[str], cache: Path) -> dict[str, float | None]:
    """Settled YES results (public REST, cached; unsettled entries are refetched)."""
    from albert.research.leadlag_fast import market_info

    have = json.loads(cache.read_text()) if cache.exists() else {}
    have = {k: v for k, v in have.items() if v.get("result") in ("yes", "no")}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(have))
    info = market_info(sorted(tickers), cache)
    return {t: (1.0 if (info.get(t) or {}).get("result") == "yes" else
                0.0 if (info.get(t) or {}).get("result") == "no" else None) for t in tickers}


def _fee_info(s) -> dict[str, dict]:
    p = s.tables_dir / "kalshi_series.parquet"
    if not p.exists():
        return {}
    return {r["series_ticker"]: r for r in pl.read_parquet(p).iter_rows(named=True)}


def fit_params(tapes: dict[str, tuple[pl.DataFrame, pl.DataFrame]], dev_hours: list[str],
               buckets: pl.DataFrame, cache: Path) -> tuple[FairParams, dict[str, Any], pl.DataFrame]:
    """Fit micro-price, Kalman, OU and ridge on dev hours. Returns (params, diagnostics,
    dev thinned samples with targets)."""
    qd = pl.concat([tapes[h][0] for h in dev_hours])
    td = pl.concat([tapes[h][1] for h in dev_hours])
    g, mdiag = fit_micro(qd)
    _log(f"micro fitted on {mdiag['transitions']} transitions")
    obs = kalman_obs(qd, td)
    (kq, kcm, kst), kdiag = fit_kalman(obs)
    _log(f"kalman q={kq:.3g} cm={kcm:.3g} st={kst:.3g} ({kdiag['evaluations']} evals)")
    params = FairParams(micro_g=g, kal_q=kq, kal_cm=kcm, kal_st=kst)
    states: dict[str, FairState] = {}
    thin: dict[str, int] = {}
    feats = [features_from_tape(*tapes[h], params, states, thin).filter("thin") for h in dev_hours]
    end_ns = int(qd["t"].max())
    dev = add_regimes(with_targets(pl.concat(feats), qd, end_ns), buckets).with_columns(
        regime_expr().alias("regime"))
    params.ou_hl, odiag = fit_ou(dev)
    params.ridge, rdiag = fit_ridge(dev)
    diag = {"micro": mdiag, "kalman": {"q": kq, "cm": kcm, "st": kst, **kdiag}, "ou": odiag,
            "ridge": rdiag, "dev_samples": dev.height}
    return params, diag, dev


def all_features(tapes: dict[str, tuple[pl.DataFrame, pl.DataFrame]], params: FairParams,
                 cache: Path, start: str) -> list[Path]:
    """Features at every update for every hour (cached per hour, keyed by params + start)."""
    tag = hashlib.md5((params.tag() + start + ",".join(tapes)).encode()).hexdigest()[:10]
    paths = [cache / f"fair_value_feat_{tag}_{h}.parquet" for h in tapes]
    if all(p.exists() for p in paths):
        return paths
    states: dict[str, FairState] = {}
    thin: dict[str, int] = {}
    cache.mkdir(parents=True, exist_ok=True)
    for (h, (q, t)), p in zip(tapes.items(), paths):
        features_from_tape(q, t, params, states, thin).write_parquet(p)
        _log(f"features {h}")
    return paths


def _label(spec: dict[str, Any], lat: float, strat: "FairQuoter") -> dict[str, Any]:
    return {"mode": spec["mode"], "est": spec.get("est", ""), "thr_c": round(100 * spec.get("thr", 0.0), 2),
            "latency_ms": lat, "dropped": strat.dropped}


def maker_test(root: Path, start: str, split: str, end: str, params: FairParams,
               fee_info: dict[str, dict], results: dict[str, float | None],
               vol: dict[tuple[str, int], bool], vprev: dict[tuple[str, int], bool], *,
               latencies_ms: tuple[float, ...] = (50.0, 100.0),
               ests: tuple[str, ...] = ("micro", "kal", "ridge"),
               thrs: tuple[float, ...] = (-0.01, 0.0, 0.01, 0.02, 0.03),
               lean_thrs: tuple[float, ...] = (0.005, 0.01, 0.02), min_fill_frac: float = 0.4,
               out_dir: Path | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Dev grid (baseline; per estimator drop x thr, lean x thr, skew) at the first latency;
    per estimator and mode keep the thr with the best dev exit markout (net of fees) among
    variants that keep ``min_fill_frac`` of the baseline's fills; replay the chosen variants on
    holdout (fresh state at the split) at every latency, and on dev at the other latencies.
    Returns (dev grid, chosen table)."""
    split_ns = int(datetime.strptime(split, "%Y-%m-%dT%H").replace(tzinfo=UTC).timestamp() * NS)

    def run(specs: list[dict[str, Any]], lat: float, a: str, b: str) -> list[tuple[dict, list, FairQuoter]]:
        tr = Tracker(params)
        runs = [(SimConfig(latency_ms=lat, fee_info=fee_info),
                 FairQuoter(tracker=tr, vol_prev=vprev, **MAKER_BASE, **spec)) for spec in specs]
        stats: dict[str, int] = defaultdict(int)
        sims = run_many(iter_events(stream_files(root, a, b), stats), runs, tr)
        _log(f"maker pass {a}..{b} latency {lat:g}: {len(specs)} variants, {stats['trades']} trades")
        return [(spec, maker_rows(sim, results), strat) for spec, sim, (_, strat) in zip(specs, sims, runs)]

    lat0 = latencies_ms[0]
    grid = [{"mode": "base"}]
    for est in ests:
        grid += [{"mode": "drop", "est": est, "thr": thr} for thr in thrs]
        grid += [{"mode": "lean", "est": est, "thr": thr} for thr in lean_thrs]
        grid.append({"mode": "skew", "est": est})
    dev_out = run(grid, lat0, start, split)
    dt = pl.DataFrame([r for spec, rows, strat in dev_out
                       for r in maker_summary(rows, split_ns, vol, _label(spec, lat0, strat))
                       if r["split"] == "dev"], infer_schema_length=None)
    allr = dt.filter(pl.col("regime") == "all")
    base_fills = allr.filter(pl.col("mode") == "base")["fills"][0]
    chosen: list[dict[str, Any]] = [{"mode": "base"}]
    for est in ests:
        for mode in ("drop", "lean"):
            c = (allr.filter((pl.col("mode") == mode) & (pl.col("est") == est)
                             & (pl.col("fills") >= min_fill_frac * base_fills) & pl.col("mx_300s_c").is_not_null())
                 .sort("mx_300s_c", descending=True))
            if c.height:
                chosen.append({"mode": mode, "est": est, "thr": round(c["thr_c"][0] / 100, 4)})
        chosen.append({"mode": "skew", "est": est})
    keep = {json.dumps(sp, sort_keys=True) for sp in chosen}
    table = [r for spec, rows, strat in dev_out if json.dumps(spec, sort_keys=True) in keep
             for r in maker_summary(rows, split_ns, vol, _label(spec, lat0, strat)) if r["split"] == "dev"]
    for lat in latencies_ms:
        passes = [(split, end)] if lat == lat0 else [(start, split), (split, end)]
        for a, b in passes:
            for spec, rows, strat in run(chosen, lat, a, b):
                table += [r for r in maker_summary(rows, split_ns, vol, _label(spec, lat, strat))
                          if r["split"] == ("dev" if a == start else "holdout")]
                if out_dir is not None and spec["mode"] == "base" and lat == lat0:
                    pl.DataFrame(rows, infer_schema_length=None).write_parquet(
                        out_dir / f"maker_base_fills_{'dev' if a == start else 'holdout'}.parquet")
    if out_dir is not None:
        for spec, rows, strat in dev_out:
            if spec["mode"] == "base" and rows:
                pl.DataFrame(rows, infer_schema_length=None).write_parquet(out_dir / "maker_base_fills_dev.parquet")
    return dt, pl.DataFrame(table, infer_schema_length=None).sort("split", "latency_ms", "mode", "est", "regime")


MAKER_BASE = dict(improve=True, min_spread=0.05, cooldown_s=300)


def fair_value_study(s, *, start: str = "2026-11-05T03", split: str = "2026-11-05T11",
                     end: str = "2026-11-05T18", root: Path | None = None,
                     latencies_ms: tuple[float, ...] = (50.0, 100.0), maker: bool = True,
                     maker_ests: tuple[str, ...] = ("micro", "kal", "ridge"),
                     maker_thrs: tuple[float, ...] = (-0.01, 0.0, 0.01, 0.02, 0.03),
                     lean_thrs: tuple[float, ...] = (0.005, 0.01, 0.02),
                     taker_ests: tuple[str, ...] = ("wmid", "micro", "kal", "ema_trd_300", "ou", "ridge"),
                     taker_exits: tuple[int, ...] = (10, 60, 300)) -> Path:
    """Fit on [start, split), evaluate on both halves, write report ``research-fair-value``.
    Hours are ``YYYY-MM-DDTHH`` strings (end exclusive)."""
    from albert.reports import Report
    from albert.research.volatile import cached_buckets

    root = root or s.raw_dir / "stream"
    cache = s.data_dir / "cache"
    split_ns = int(datetime.strptime(split, "%Y-%m-%dT%H").replace(tzinfo=UTC).timestamp() * NS)
    buckets = cached_buckets(s, start, end, root=root)
    vol = {(m, b): v for m, b, v in buckets.select("market", "bucket_ns", "volatile").iter_rows()}
    vprev = {(m, b): v for m, b, v in buckets.select("market", "bucket_ns", "volatile_prev").iter_rows()}
    tapes = load_tapes(root, start, end, cache)
    hours = list(tapes)
    dev_hours = [h for h in hours if h < split]
    if not dev_hours or len(dev_hours) == len(hours):
        raise RuntimeError(f"need hours on both sides of {split}: {hours}")
    params, diag, _ = fit_params(tapes, dev_hours, buckets, cache)
    rep = Report(s.reports_dir, "research-fair-value")
    (rep.dir / "params.json").write_text(json.dumps(asdict(params), indent=1))
    quotes = pl.concat([q.select("t", "market", "bid", "ask", "bsz", "asz") for q, _ in tapes.values()])
    end_ns = int(quotes["t"].max())
    rep.json("Run", {"root": str(root), "start": start, "split": split, "end": end, "hours": hours,
                     "quote_updates": quotes.height,
                     "trades": sum(t.height for _, t in tapes.values()),
                     "markets": quotes["market"].n_unique()})
    rep.json("Fitted on dev", {k: v for k, v in diag.items() if k != "ridge"})
    rep.table("Micro-price adjustment G (c) by spread bucket (rows: 1,2,3,4,5-6,7-10,11+ c) x "
              "imbalance bucket (bid-heavy right)",
              pl.DataFrame([{"spread_bucket": i, **{f"imb{j}": round(100 * v, 3) for j, v in enumerate(r)}}
                            for i, r in enumerate(params.micro_g or [])]))
    rep.table("Ridge coefficients (per 1 sd of feature, in price units)", pl.DataFrame(diag["ridge"]),
              save_as="ridge")

    # -- prediction test
    paths = all_features(tapes, params, cache, start)
    thin = pl.concat([pl.read_parquet(p).filter("thin") for p in paths])
    samples = add_regimes(with_targets(thin, quotes, end_ns), buckets)
    pt = prediction_table(samples, params, split_ns)
    rep.table("Prediction: all estimators (full table)", pt, save_as="prediction", max_rows=0)
    for target in ("mid", "exit"):
        for col in ("r2_raw", "r2_cal", "hit_w", "edge_c"):
            piv = (pt.filter((pl.col("target") == target) & (pl.col("regime") == "all"))
                   .with_columns(pl.concat_str([pl.col("split"), pl.col("h").cast(pl.Utf8)], separator="_").alias("k"))
                   .pivot(on="k", index="est", values=col))
            rep.table(f"{col}, target {target}, all regimes (columns: split_horizon s)", piv)
    for target in ("mid", "exit"):
        piv = (pt.filter((pl.col("target") == target) & (pl.col("split") == "holdout"))
               .with_columns(pl.concat_str([pl.col("regime"), pl.col("h").cast(pl.Utf8)], separator="_").alias("k"))
               .pivot(on="k", index="est", values="r2_cal"))
        rep.table(f"Holdout r2_cal by regime, target {target} (columns: regime_horizon s)", piv)
    bt = prediction_table(samples, params, split_ns, ("wmid", "micro", "ema_mid_10", "kal", "ou", "ridge"),
                          by="band")
    rep.table("Prediction by quoted spread band (dev beta fitted within the band; full table)", bt,
              save_as="prediction_by_band", max_rows=0)
    for target in ("mid", "exit"):
        for col in ("r2_cal", "edge_c"):
            piv = (bt.filter((pl.col("target") == target) & (pl.col("split") == "holdout"))
                   .with_columns(pl.concat_str([pl.col("band"), pl.col("h").cast(pl.Utf8)], separator="_").alias("k"))
                   .pivot(on="k", index="est", values=col))
            rep.table(f"Holdout {col} by spread band, target {target} (columns: band_horizon s)", piv)
    sp = (samples.with_columns(band_expr().alias("band"), (pl.col("t") >= split_ns).alias("holdout"))
          .group_by("holdout", "band").agg(pl.len().alias("samples"), (pl.col("spread").mean() * 100).alias("mean_spread_c"))
          .sort("holdout", "band"))
    rep.table("Samples per spread band", sp)
    del samples, thin

    # -- taker test
    fee_info = _fee_info(s)
    fee_mult = {k: float(v.get("fee_multiplier") or 1.0) for k, v in fee_info.items()}
    cands = []
    for p in paths:
        f = add_regimes(pl.read_parquet(p), buckets)
        cands.append(taker_candidates(f, params, taker_ests, taker_exits, fee_mult))
    cands = pl.concat([c for c in cands if not c.is_empty()]) if any(not c.is_empty() for c in cands) else pl.DataFrame()
    results = _results(sorted(quotes["market"].unique().to_list()), cache / "fair_value_markets.json")
    trades = taker_trades(cands, quotes, end_ns, latencies_ms=latencies_ms, results=results)
    ts = taker_summary(trades, split_ns)
    rep.table("Taker: buy when fair - ask >= fee + thr; exit at the bid after h s (net c/contract, "
              "t by event)", ts, save_as="taker", max_rows=0)
    if not ts.is_empty():
        rep.table("Taker, all regimes, latency 50 ms",
                  ts.filter((pl.col("regime") == "all") & (pl.col("latency_ms") == latencies_ms[0]))
                  .select("est", "h", "thr_c", "split", "filled", "trades", "events", "gross_c", "pnl_c", "t_event"),
                  max_rows=200)
    if not trades.is_empty():
        trades.write_parquet(rep.dir / "taker_trades.parquet")

    # -- maker test
    if maker:
        dt, mt = maker_test(root, start, split, end, params, fee_info, results, vol, vprev,
                            latencies_ms=latencies_ms, ests=maker_ests, thrs=maker_thrs,
                            lean_thrs=lean_thrs, out_dir=rep.dir)
        rep.table(f"Maker dev selection grid (latency {latencies_ms[0]:g} ms; dev hours only; markouts "
                  "net of fees, c/contract, t by event)", dt, save_as="maker_dev_grid", max_rows=200)
        rep.table("Maker, chosen variants (thr chosen on dev), dev and holdout (fees deducted from every "
                  "markout; mx = exit price, settle = settled markets only)", mt, save_as="maker", max_rows=200)
    return rep.save()
