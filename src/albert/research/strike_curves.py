"""Cross-strike curves on Kalshi ladders: does a fitted implied distribution find mispriced strikes?

Kalshi lists ladders of markets on one underlying: "above X" thresholds (token usage, shipping
transits, approval, YouTube views, Rotten Tomatoes...) and mutually exclusive brackets (daily
high/low temperature, Truth Social posts, approval bins). At every top-of-book change of any leg we
fit a smooth monotone implied distribution of the underlying X across the strikes and compare each
strike's price with it.

**Legs on one latent scale.** A leg pays YES iff ``lo < X <= hi`` for a continuous latent X, so its
fair price is ``S(lo) - S(hi)`` with ``S`` the survival function (S(-inf) = 1, S(inf) = 0):
"greater f" is (f, inf), "less c" is (-inf, c), "between [f, c]" is (f, c). In bracket events the
underlying is discrete (whole degrees, 0.1 points of approval, post counts), so edges move half a
grid step outwards: "between [70, 71]" is (69.5, 71.5), "less 70" is (-inf, 69.5), "greater 71" is
(71.5, inf); the step is the gap between adjacent brackets. Threshold-only ladders keep the raw
strikes (a uniform shift is absorbed by the location). Strikes are centred and scaled per ladder.

**Fits** (per state, weighted least squares in probability space, weight 1 / max(spread, 1c)^2,
legs without a two-sided book get weight 0):

* parametric: ``logistic``, ``normal`` and ``skewnormal`` survival curves, Levenberg-Marquardt with
  a numeric Jacobian, batched so the full fit and every leave-one-out (LOO) fit run together;
* nonparametric: ``iso`` (weighted isotonic regression of the threshold survival, full fit),
  ``nbr`` (LOO: logit-linear interpolation between the two neighbouring strikes, thresholds only),
  ``sum1`` (LOO for a complete bracket set: 1 minus the other brackets' mids);
* ``bid`` / ``ask`` curves: the logistic LOO fit to the other legs' bids (asks) only.

A strike's **residual** is LOO fair minus its own mid: what the other strikes say it is worth. The
full fit lets a tight leg pull the curve to itself, so the LOO value is the honest signal. A LOO
value needs an over-determined fit (k + 1 other quoted legs for k parameters).

**Coverage.** The recorder rotates markets, so a leg is only quoted while its file streams it: an
``end`` marker after each file's last event clears its book (otherwise a dropped leg's last quote
would live on in the curve). Ladders whose legs were recorded at different times have no states.

**No look-ahead.** A state at time t uses only books at t. Evaluation (``volatile`` splits) uses
the leg's own 5-minute bucket flag; strategy filters use ``volatile_prev``.

**Trading tests.** Taker: buy the side the curve calls cheap when ``fair - ask >= fee + thr`` (entry
price 5-95c); the order reaches the book after the latency and fills at the then-best price only if
it is still at or below the signal's ask with size; exit by selling at the bid after 60/300 s
(taker fee both ways) or hold to settlement. Maker: the baseline ``Quoter(improve=True, min_spread=0.05,
cooldown_s=300)`` on ladder legs, quoting only the side the curve calls cheap (``cheap``) or
dropping only the side it calls rich (``skew``). Fits are fed to the replay as events at the
state time + 1 ns, computed from the recorded book, which is the book the simulated strategy sees.

**Historical.** Settled daily-high bracket sets (six legs) at each 60-minute candle close
(``historical_weather``, fits batched across all candles): does the LOO residual predict
settlement beyond the mid, and does buying the cheap side at the candle's close ask/bid pay when
held to settlement. Dev = before 2026-10-01, holdout = October.
"""

import bisect
import heapq
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.special import expit, ndtr, owens_t

from albert.backtest.l2 import Event as L2Event
from albert.backtest.l2 import Quoter, SimConfig, Simulator, _mark_executions
from albert.research.arb_l2 import Books, _ts_ns, build_sets, taker_fee

API = "https://api.elections.kalshi.com/trade-api/v2"
FAMILIES = ("logistic", "normal", "skewnormal")
FAIRS = ("logistic", "normal", "skewnormal", "nbr", "sum1")
HORIZONS_S = (10, 60, 300)
LATENCIES_MS = (50, 100)
STRIKE_TYPES = ("greater", "greater_or_equal", "less", "less_or_equal", "between")
MIN_SPREAD = 0.01
INF = math.inf


# ---- metadata ------------------------------------------------------------------------------------
def _get(http: Any, url: str, params: dict | None = None, tries: int = 6) -> Any:
    """Public REST GET with backoff on 429 (the API is shared with every other process)."""
    import time

    r = None
    for i in range(tries):
        r = http.get(url, params=params)
        if r.status_code != 429:
            return r
        time.sleep(0.5 * 2**i)
    return r


def fetch_events(tickers: Iterable[str], cache: Path, refresh: bool = False) -> dict[str, dict]:
    """event_ticker -> public REST event payload with nested markets (cached JSON)."""
    import httpx

    have = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [e for e in sorted(set(tickers)) if refresh or e not in have]
    if todo:
        with httpx.Client(timeout=20) as http:
            for e in todo:
                r = _get(http, f"{API}/events/{e}", params={"with_nested_markets": "true"})
                if r.status_code == 200:
                    have[e] = r.json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(have))
    return have


def market_results(tickers: Iterable[str], cache: Path, refresh: bool = True) -> dict[str, float | None]:
    """market -> 1.0 / 0.0 settled result, None while open (public REST, cached; open markets are
    re-fetched when ``refresh``)."""
    import httpx

    have = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [t for t in sorted(set(tickers))
            if t not in have or (refresh and have[t].get("result") not in ("yes", "no"))]
    if todo:
        with httpx.Client(timeout=20) as http:
            for t in todo:
                r = _get(http, f"{API}/markets/{t}")
                if r.status_code == 200:
                    m = r.json().get("market", {})
                    have[t] = {k: m.get(k) for k in ("status", "result", "close_time", "settlement_ts")}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(have, indent=0))
    return {t: (1.0 if have.get(t, {}).get("result") == "yes" else 0.0
                if have.get(t, {}).get("result") == "no" else None) for t in tickers}


@dataclass
class Leg:
    market: str
    lo: float            # YES iff lo < X <= hi (raw strike units)
    hi: float
    kind: str            # gt | lt | between
    close_ns: int | None = None


@dataclass
class Ladder:
    key: str
    event: str
    kind: str            # threshold | bracket
    legs: list[Leg]
    complete: bool = False     # bracket set with every leg of the event recorded
    mult: float = 1.0
    center: float = 0.0
    scale: float = 1.0
    n_event: int = 0

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([(lg.lo - self.center) / self.scale for lg in self.legs])
        hi = np.array([(lg.hi - self.center) / self.scale for lg in self.legs])
        return lo, hi


def _unit(vals: list[float]) -> float:
    for d in range(4):
        if all(abs(v * 10**d - round(v * 10**d)) < 1e-6 for v in vals):
            return 10.0**-d
    return 1e-4


def _step(between: list[tuple[float, float]], vals: list[float]) -> float:
    """Grid step of a discrete bracket underlying: the gap between adjacent brackets."""
    b = sorted(between)
    gaps = [round(b[i + 1][0] - b[i][1], 9) for i in range(len(b) - 1)]
    gaps = [g for g in gaps if g > 1e-9]
    return min(gaps) if gaps else _unit(vals)


def ladders_from_events(payloads: dict[str, dict], recorded: set[str],
                        fee_mult: dict[str, float] | None = None, min_legs: int = 3) -> list[Ladder]:
    """Threshold ladders and bracket sets with at least ``min_legs`` recorded legs. Events whose
    strikes repeat (two teams' totals in one event, one rain market per city) are split by the
    ticker suffix's letter prefix first. ``arb_l2.build_sets`` decides which threshold events are
    ladders (two or more recorded "greater" strikes), the same grouping the arbitrage scan uses."""
    out: list[Ladder] = []
    for e, p in sorted(payloads.items()):
        ev = p.get("event", p)
        ms = [m for m in (ev.get("markets") or p.get("markets") or [])
              if m.get("strike_type") in STRIKE_TYPES]
        if len(ms) < 2:
            continue
        keys = [(m.get("strike_type"), m.get("floor_strike"), m.get("cap_strike")) for m in ms]
        groups: dict[str, list[dict]] = defaultdict(list)
        if len(set(keys)) < len(keys):
            for m in ms:
                groups[re.sub(r"[\d.]+$", "", m["ticker"].rsplit("-", 1)[1])].append(m)
        else:
            groups[""] = ms
        for gname, gm in groups.items():
            ld = _ladder(f"{e}{'|' + gname if gname else ''}", e, gm, bool(ev.get("mutually_exclusive")),
                         recorded, (fee_mult or {}).get(e.split("-")[0], 1.0))
            if ld is not None and len(ld.legs) >= min_legs:
                out.append(ld)
    # cross-check thresholds with the arbitrage scanner's grouping
    meta = {ld.key: {"me": False, "legs": [{"t": lg.market, "strike_type": "greater", "floor": lg.lo}
                                            for lg in ld.legs if lg.kind == "gt"]}
            for ld in out if ld.kind == "threshold"}
    laddered = {s.event for s in build_sets(meta, recorded) if s.kind == "ladder"}
    return [ld for ld in out if ld.kind == "bracket" or ld.key in laddered]


def _ladder(key: str, event: str, ms: list[dict], me: bool, recorded: set[str],
            mult: float) -> Ladder | None:
    types = {m["strike_type"] for m in ms}
    between = [(float(m["floor_strike"]), float(m["cap_strike"])) for m in ms
               if m["strike_type"] == "between"]
    vals = [float(v) for m in ms for v in (m.get("floor_strike"), m.get("cap_strike")) if v is not None]
    bracket = bool(between) or (types & {"less", "less_or_equal"} and types & {"greater", "greater_or_equal"})
    d = _step(between, vals) / 2 if between else 0.0
    legs = []
    for m in ms:
        st, f, c = m["strike_type"], m.get("floor_strike"), m.get("cap_strike")
        if st == "greater":
            lo, hi, kind = float(f) + d, INF, "gt"
        elif st == "greater_or_equal":
            lo, hi, kind = float(f) - d, INF, "gt"
        elif st == "less":
            lo, hi, kind = -INF, float(c) - d, "lt"
        elif st == "less_or_equal":
            lo, hi, kind = -INF, float(c) + d, "lt"
        else:
            lo, hi, kind = float(f) - d, float(c) + d, "between"
        legs.append(Leg(m["ticker"], lo, hi, kind, _ts_ns(m.get("close_time"))))
    rec = [lg for lg in legs if lg.market in recorded]
    if not rec:
        return None
    rec.sort(key=lambda lg: (lg.lo if lg.lo > -INF else lg.hi - 1e9, lg.hi))
    fin = [v for lg in legs for v in (lg.lo, lg.hi) if abs(v) < INF]
    center = float(np.median(fin))
    scale = float(max(fin) - min(fin)) or 1.0
    return Ladder(key, event, "bracket" if bracket else "threshold", rec,
                  complete=bool(bracket and me and len(rec) == len(legs)), mult=mult,
                  center=center, scale=scale, n_event=len(legs))


# ---- curve fitting -------------------------------------------------------------------------------
def survival(x: np.ndarray, theta: np.ndarray, family: str) -> np.ndarray:
    """S(x) for parameter rows ``theta`` (B, k) = (mu, log sigma[, alpha]) -> (B, n); ``x`` is
    (n,) shared by all rows or (B, n) per row."""
    mu = theta[:, 0:1]
    sig = np.exp(theta[:, 1:2])
    with np.errstate(invalid="ignore", over="ignore"):
        z = ((x if x.ndim == 2 else x[None, :]) - mu) / sig
        if family == "logistic":
            return expit(-z)
        if family == "normal":
            return ndtr(-z)
        s = 1.0 - ndtr(z) + 2.0 * owens_t(z, theta[:, 2:3])
    return np.clip(np.nan_to_num(s, nan=0.0), 0.0, 1.0)


def leg_probs(lo: np.ndarray, hi: np.ndarray, theta: np.ndarray, family: str) -> np.ndarray:
    return survival(lo, theta, family) - survival(hi, theta, family)


def init_theta(lo: np.ndarray, hi: np.ndarray, p: np.ndarray, w: np.ndarray, family: str) -> np.ndarray:
    """Starting values: logit regression on threshold-type legs, else bracket moments."""
    ok = w > 0
    xs, ss, ws = [], [], []
    for i in np.flatnonzero(ok):
        if np.isinf(hi[i]) and np.isfinite(lo[i]):
            xs.append(lo[i]), ss.append(p[i]), ws.append(w[i])
        elif np.isinf(lo[i]) and np.isfinite(hi[i]):
            xs.append(hi[i]), ss.append(1 - p[i]), ws.append(w[i])
    mu = sig = None
    if len(set(xs)) >= 2:
        x, s = np.array(xs), np.clip(np.array(ss), 0.02, 0.98)
        y = np.log(s / (1 - s))
        ww = np.array(ws) * (s * (1 - s)) ** 2
        xm, ym = np.average(x, weights=ww), np.average(y, weights=ww)
        b = np.sum(ww * (x - xm) * (y - ym)) / max(np.sum(ww * (x - xm) ** 2), 1e-12)
        if b < -1e-9:
            sig, mu = -1.0 / b, xm - ym / b
    if mu is None:
        c = np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2,
                     np.where(np.isfinite(lo), lo + 0.1, hi - 0.1))
        q = np.clip(p, 0, None) * ok
        if q.sum() > 0:
            mu = float(np.sum(q * c) / q.sum())
            sig = float(np.sqrt(np.sum(q * (c - mu) ** 2) / q.sum())) or 0.1
        else:
            mu, sig = 0.0, 0.25
    sig = float(np.clip(sig, 0.01, 10.0))
    if family != "logistic":
        sig *= 1.7
    th = [float(mu), math.log(sig)]
    return np.array(th + ([0.0] if family == "skewnormal" else []))


def fit_batch(lo: np.ndarray, hi: np.ndarray, p: np.ndarray, W: np.ndarray, family: str,
              theta0: np.ndarray, iters: int = 25, tol: float = 1e-6) -> np.ndarray:
    """Levenberg-Marquardt for B problems sharing strikes and prices but not weights (rows of
    ``W``), e.g. a full fit plus every leave-one-out fit; ``lo``, ``hi`` and ``p`` may also be
    (B, n) for B unrelated problems. Returns theta (B, k)."""
    th = theta0.astype(float).copy()
    k = th.shape[1]
    steps = np.array([1e-5, 1e-5, 1e-4])[:k]

    def res(t: np.ndarray) -> np.ndarray:
        return leg_probs(lo, hi, t, family) - (p if p.ndim == 2 else p[None, :])

    r = res(th)
    cost = (W * r * r).sum(1)
    lam = np.full(len(th), 1e-3)
    eye = np.eye(k)[None]
    for _ in range(iters):
        J = np.empty(r.shape + (k,))
        for j in range(k):
            t2 = th.copy()
            t2[:, j] += steps[j]
            J[:, :, j] = (res(t2) - r) / steps[j]
        A = np.einsum("bnk,bn,bnl->bkl", J, W, J)
        g = np.einsum("bnk,bn,bn->bk", J, W, r)
        D = np.einsum("bkk->bk", A)
        A2 = A + eye * (lam[:, None, None] * D[:, :, None] + 1e-12)
        try:
            step = -np.linalg.solve(A2, g[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        tn = th + step
        tn[:, 0] = np.clip(tn[:, 0], -3.0, 3.0)        # location within a few strike ranges
        tn[:, 1] = np.clip(tn[:, 1], -7.0, 3.0)
        if k == 3:
            tn[:, 2] = np.clip(tn[:, 2], -8.0, 8.0)
        rn = res(tn)
        cn = (W * rn * rn).sum(1)
        ok = np.isfinite(cn) & (cn < cost)
        gain = np.where(ok, cost - cn, 0.0)
        th[ok], r[ok], cost[ok] = tn[ok], rn[ok], cn[ok]
        lam = np.clip(np.where(ok, lam * 0.3, lam * 5.0), 1e-9, 1e9)
        done = (np.abs(step).max(1) < tol) | (lam >= 1e4) | (ok & (gain <= 1e-9 * (cost + 1e-9)))
        if np.all(done):
            break
    return th


def fit_ladder(lo: np.ndarray, hi: np.ndarray, p: np.ndarray, w: np.ndarray, family: str,
               theta_prev: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(full-fit prices (n,), leave-one-out prices (n,), full theta). The full fit needs k + 1
    quoted legs (k parameters); the LOO price of a leg is the fit to the other legs evaluated at its
    edges and needs k + 1 *other* quoted legs (an over-determined fit), else NaN. Legs with zero
    weight get the full fit as their LOO value. The start is the previous state's solution when it
    fits the current prices better than the default start."""
    n = len(p)
    k = 3 if family == "skewnormal" else 2
    valid = np.flatnonzero(w > 0)
    nan = np.full(n, np.nan)
    if len(valid) < k + 1:
        return nan, nan.copy(), np.full(k, np.nan)
    do_loo = len(valid) - 1 >= k + 1
    rows = [w.copy()]
    if do_loo:
        for i in valid:
            wi = w.copy()
            wi[i] = 0.0
            rows.append(wi)
    W = np.array(rows)
    t0 = init_theta(lo, hi, p, w, family)
    if theta_prev is not None and np.all(np.isfinite(theta_prev)) and len(theta_prev) == k:
        cands = np.array([t0, theta_prev])
        cost = (w * (leg_probs(lo, hi, cands, family) - p) ** 2).sum(1)
        t0 = cands[int(np.argmin(cost))]
    th = fit_batch(lo, hi, p, W, family, np.repeat(t0[None], len(rows), 0))
    probs = leg_probs(lo, hi, th, family)
    full = probs[0]
    loo = full.copy()
    if do_loo:
        for j, i in enumerate(valid, start=1):
            loo[i] = probs[j, i]
    else:
        loo[valid] = np.nan
    return full, loo, th[0]


def isotonic_decreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares non-increasing fit (pool adjacent violators)."""
    blocks: list[list[float]] = []
    for yi, wi in zip(y, w):
        blocks.append([float(yi), float(wi), 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0] - 1e-12:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            ww = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / ww, ww, c1 + c2])
    out: list[float] = []
    for v, _, c in blocks:
        out += [v] * int(c)
    return np.array(out)


def threshold_nonparam(lo: np.ndarray, hi: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(isotonic full fit, neighbour LOO interpolation) on survival-type legs (gt / lt); NaN for
    brackets and invalid legs."""
    n = len(p)
    iso, nbr = np.full(n, np.nan), np.full(n, np.nan)
    surv = []          # (x, S, w, i, is_lt)
    for i in range(n):
        if w[i] <= 0:
            continue
        if np.isinf(hi[i]) and np.isfinite(lo[i]):
            surv.append((lo[i], p[i], w[i], i, False))
        elif np.isinf(lo[i]) and np.isfinite(hi[i]):
            surv.append((hi[i], 1 - p[i], w[i], i, True))
    if len(surv) < 2:
        return iso, nbr
    surv.sort()
    xs = np.array([s[0] for s in surv])
    ss = np.array([s[1] for s in surv])
    fit = isotonic_decreasing(ss, np.array([s[2] for s in surv]))
    lg = np.log(np.clip(ss, 0.005, 0.995) / (1 - np.clip(ss, 0.005, 0.995)))
    for j, (x, _, _, i, is_lt) in enumerate(surv):
        iso[i] = 1 - fit[j] if is_lt else fit[j]
        if 0 < j < len(surv) - 1 and xs[j + 1] > xs[j - 1]:
            f = lg[j - 1] + (x - xs[j - 1]) / (xs[j + 1] - xs[j - 1]) * (lg[j + 1] - lg[j - 1])
            v = 1 / (1 + math.exp(-f))
            nbr[i] = 1 - v if is_lt else v
    return iso, nbr


# ---- recorded books ------------------------------------------------------------------------------
_MKT = re.compile(r'"market_ticker":"([^"]+)"')
_KINDS = {"orderbook_snapshot": "snapshot", "orderbook_delta": "delta", "trade": "trade"}


def _hour_files(files: list[Path]) -> dict[str, list[Path]]:
    by_hour: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        by_hour[f"{p.parent.name}T{p.name[:2]}"].append(p)
    return dict(sorted(by_hour.items()))


def _rows_for(path: Path, markets: set[str]) -> list[tuple[int, str, dict]]:
    from albert.stream import read_stream

    out = []
    for row in read_stream(path):
        frame = row["frame"]
        m = _MKT.search(frame)
        if m is None or m.group(1) not in markets:
            continue
        d = json.loads(frame)
        kind = _KINDS.get(d.get("type"))
        if kind:
            out.append((row["recv_ns"], kind, d["msg"]))
    return out


def l2_events(files: list[Path], markets: set[str]) -> Iterator[L2Event]:
    """``backtest.l2.iter_events`` restricted to ``markets`` (executions are labelled per market,
    so filtering first gives the same labels), merged hour by hour in receive order."""
    for _, paths in _hour_files(files).items():
        streams = []
        for p in paths:
            evs = [L2Event(t, k, msg["market_ticker"], msg) for t, k, msg in _rows_for(p, markets)]
            _mark_executions(evs)
            streams.append(evs)
        yield from heapq.merge(*streams, key=lambda e: e.recv_ns)


def tob_from_events(events: Iterable[Any], close_ns: dict[str, int] | None = None) -> pl.DataFrame:
    """Top-of-book changes (price or size) per market: t_ns, bid, bid_sz, ask, ask_sz. Crossed
    books, books after the market's close time and markets no longer streamed (``end`` events)
    are written as empty (null) quotes."""
    books = Books()
    last: dict[str, tuple] = {}
    rows: list[tuple] = []
    close_ns = close_ns or {}
    for ev in events:
        m = ev.market
        if ev.kind == "end":            # the recorder stopped streaming this market
            books.lv.pop(m, None)
            if last.get(m, (None,) * 4) != (None,) * 4:
                last[m] = (None,) * 4
                rows.append((m, ev.recv_ns, None, None, None, None))
            continue
        if ev.kind not in ("snapshot", "delta"):
            continue
        books.apply(ev.kind, m, ev.msg)
        if m not in books.lv:
            continue
        y, n = books.best(m, "yes"), books.best(m, "no")
        q = (y[0] if y else None, y[1] if y else None,
             round(1 - n[0], 6) if n else None, n[1] if n else None)
        c = close_ns.get(m)
        if (c is not None and ev.recv_ns >= c) or (q[0] is not None and q[2] is not None and q[0] >= q[2] - 1e-9):
            q = (None, None, None, None)
        if last.get(m) != q:
            last[m] = q
            rows.append((m, ev.recv_ns, *q))
    return pl.DataFrame(rows, schema={"market": pl.String, "t_ns": pl.Int64, "bid": pl.Float64,
                                      "bid_sz": pl.Float64, "ask": pl.Float64, "ask_sz": pl.Float64},
                        orient="row")


def cached_tob(files: list[Path], markets: set[str], cache_dir: Path,
               close_ns: dict[str, int] | None = None) -> pl.DataFrame:
    """Top of book hour by hour, cached per hour (and per market set) under ``cache_dir``."""
    import hashlib

    tag = hashlib.sha1("\n".join(sorted(markets)).encode()).hexdigest()[:10]
    frames = []
    for hour, paths in _hour_files(files).items():
        path = cache_dir / f"strike_curves_tob_{hour}_{tag}.parquet"
        if not path.exists():
            streams = [_with_end(_rows_for(p, markets)) for p in paths]
            df = tob_from_events(heapq.merge(*streams, key=lambda e: e.recv_ns), close_ns)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(path)
        frames.append(pl.read_parquet(path))
    return pl.concat(frames).sort("t_ns", maintain_order=True) if frames else tob_from_events([])


@dataclass
class _Ev:
    recv_ns: int
    kind: str
    market: str
    msg: dict = field(default_factory=dict)


def _with_end(rows: list[tuple[int, str, dict]]) -> list[_Ev]:
    """Events of one recorded file plus an ``end`` marker per market after the file's last
    event: a market is only observed while its connection streams it (the recorder rotates
    markets and reconnects), so its last quote must not be carried past the file."""
    evs = [_Ev(t, k, msg["market_ticker"], msg) for t, k, msg in rows]
    if evs:
        t_end = max(e.recv_ns for e in evs) + 1
        evs += [_Ev(t_end, "end", m) for m in sorted({e.market for e in evs})]
    return evs


# ---- ladder states -------------------------------------------------------------------------------
def _fees(ask: float, mult: float) -> float:
    return taker_fee(1, ask, mult)


def ladder_states(ld: Ladder, tob: pl.DataFrame, families: tuple[str, ...] = FAMILIES,
                  bidask: bool = True) -> pl.DataFrame:
    """One row per (state, leg): every time a leg's best bid or ask changes (or a leg closes), the
    ladder is refitted. Columns: quotes, full fits ``f_<family>``, leave-one-out fits ``l_<fair>``,
    ``iso``, bid/ask-curve LOO fits ``lb`` / ``la``, the logistic location/scale (scaled units) and
    the sums of bids / mids / asks across legs."""
    idx = {lg.market: i for i, lg in enumerate(ld.legs)}
    n = len(ld.legs)
    lo, hi = ld.edges()
    q = (tob.filter(pl.col("market").is_in(list(idx)))
         .select("market", "t_ns", "bid", "ask")
         .with_columns(pl.col("market").replace_strict(idx, return_dtype=pl.Int64).alias("i")))
    # keep price changes only (sizes do not move the fit)
    q = q.sort("t_ns", maintain_order=True).filter(
        (pl.col("bid") != pl.col("bid").shift(1).over("i")).fill_null(True)
        | (pl.col("ask") != pl.col("ask").shift(1).over("i")).fill_null(True)
        | (pl.col("bid").is_null() != pl.col("bid").shift(1).over("i").is_null()).fill_null(True)
        | (pl.col("ask").is_null() != pl.col("ask").shift(1).over("i").is_null()).fill_null(True))
    closes = [(lg.close_ns, i) for i, lg in enumerate(ld.legs) if lg.close_ns]
    t_arr = q["t_ns"].to_list()
    i_arr = q["i"].to_list()
    b_arr = q["bid"].to_list()
    a_arr = q["ask"].to_list()
    ev = sorted([(t, 0, i, b, a) for t, i, b, a in zip(t_arr, i_arr, b_arr, a_arr)]
                + [(c, 1, i, None, None) for c, i in closes if t_arr and t_arr[0] <= c <= t_arr[-1]],
                key=lambda e: (e[0], e[1]))
    bid = np.full(n, np.nan)
    ask = np.full(n, np.nan)
    closed = np.zeros(n, bool)
    prev: dict[str, np.ndarray | None] = {f: None for f in families}
    prev_b: dict[str, np.ndarray | None] = {"lb": None, "la": None}
    out: dict[str, list] = defaultdict(list)
    was_active = False
    j = 0
    while j < len(ev):
        t = ev[j][0]
        while j < len(ev) and ev[j][0] == t:
            _, is_close, i, b, a = ev[j]
            if is_close:
                closed[i] = True
            bid[i] = np.nan if b is None else b
            ask[i] = np.nan if a is None else a
            j += 1
        ok = np.isfinite(bid) & np.isfinite(ask) & ~closed
        mid = np.where(ok, (bid + ask) / 2, np.nan)
        if ok.sum() < 3:
            # too few legs for a curve: record the quotes without fits, so later look-ups of
            # this ladder never reuse an older state's quotes or fits
            if was_active or ok.any():
                _emit(out, ld, t, bid, ask, ok, {}, (np.nan, np.nan, np.nan))
            was_active = False
            continue
        was_active = True
        spr = np.where(ok, ask - bid, np.nan)
        w = np.where(ok, 1.0 / np.maximum(np.nan_to_num(spr, nan=1.0), MIN_SPREAD) ** 2, 0.0)
        p = np.nan_to_num(mid, nan=0.5)
        row: dict[str, Any] = {}
        for fam in families:
            full, loo, th = fit_ladder(lo, hi, p, w, fam, prev[fam])
            prev[fam] = th if np.all(np.isfinite(th)) else None
            row[f"f_{fam}"], row[f"l_{fam}"] = full, loo
            if fam == "logistic":
                row["mu"], row["sig"] = th[0], math.exp(th[1]) if np.isfinite(th[1]) else np.nan
        iso, nbr = threshold_nonparam(lo, hi, p, w)
        row["iso"], row["l_nbr"] = iso, nbr
        if ld.complete and ok.all():
            row["l_sum1"] = 1.0 - (np.nansum(mid) - mid)
        else:
            row["l_sum1"] = np.full(n, np.nan)
        if bidask:
            for key, px in (("lb", bid), ("la", ask)):
                _, loo, th = fit_ladder(lo, hi, np.nan_to_num(px, nan=0.5), w, "logistic", prev_b[key])
                prev_b[key] = th if np.all(np.isfinite(th)) else None
                row[key] = loo
        sums = (float(np.nansum(np.where(ok, bid, np.nan))), float(np.nansum(mid)),
                float(np.nansum(np.where(ok, ask, np.nan))))
        _emit(out, ld, t, bid, ask, ok, row, sums)
    if not out:
        return pl.DataFrame()
    df = pl.DataFrame(dict(out), strict=False)
    return df.with_columns(pl.lit(ld.key).alias("ladder"), pl.lit(ld.event).alias("event"),
                           pl.lit(ld.kind).alias("kind"), pl.lit(ld.complete).alias("complete"),
                           ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                           (pl.col("ask") - pl.col("bid")).alias("spread"))


_FIT_KEYS = ("f_logistic", "l_logistic", "f_normal", "l_normal", "f_skewnormal", "l_skewnormal",
             "mu", "sig", "iso", "l_nbr", "l_sum1", "lb", "la")


def _emit(out: dict[str, list], ld: Ladder, t: int, bid: np.ndarray, ask: np.ndarray, ok: np.ndarray,
          row: dict[str, Any], sums: tuple[float, float, float]) -> None:
    for i in range(len(ld.legs)):
        out["t_ns"].append(t)
        out["leg"].append(i)
        out["market"].append(ld.legs[i].market)
        out["bid"].append(float(bid[i]) if ok[i] else None)
        out["ask"].append(float(ask[i]) if ok[i] else None)
        out["n_valid"].append(int(ok.sum()))
        out["pos"].append(int(ok[:i].sum()) if ok[i] else None)
        for k, v in zip(("sum_bid", "sum_mid", "sum_ask"), sums):
            out[k].append(None if not np.isfinite(v) else v)
        for key in sorted(set(_FIT_KEYS) | set(row)):
            v = row.get(key)
            x = v[i] if isinstance(v, np.ndarray) else v
            out[key].append(None if x is None or not np.isfinite(x) else float(x))


def all_states(ladders: list[Ladder], tob: pl.DataFrame, **kw: Any) -> pl.DataFrame:
    frames = [ladder_states(ld, tob, **kw) for ld in ladders]
    frames = [f for f in frames if f.height]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


# ---- statistics ----------------------------------------------------------------------------------
def clustered(values: list[float] | np.ndarray, clusters: list[str] | np.ndarray) -> tuple[float | None, float | None, int, int]:
    """(mean, t-stat clustered by ``clusters``, observations, clusters); None/NaN values skipped."""
    v = np.array([np.nan if x is None else x for x in values], dtype=float) if isinstance(values, list) else values.astype(float)
    c = np.asarray(clusters)
    ok = np.isfinite(v)
    if not ok.any():
        return None, None, 0, 0
    v, c = v[ok], c[ok]
    _, codes = np.unique(c, return_inverse=True)
    sums = np.bincount(codes, weights=v)
    cnts = np.bincount(codes).astype(float)
    n = cnts.sum()
    mean = float(sums.sum() / n)
    k = len(sums)
    if k < 2:
        return mean, None, int(n), k
    se = math.sqrt(k / (k - 1) * float(np.sum((sums - mean * cnts) ** 2))) / n
    return mean, (mean / se if se > 0 else None), int(n), k


def slope_clustered(x: np.ndarray, y: np.ndarray, clusters: np.ndarray) -> tuple[float | None, float | None]:
    """OLS slope of y on x (with intercept) and its t-stat with cluster-robust standard errors."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, c = x[ok], y[ok], np.asarray(clusters)[ok]
    if len(x) < 10 or np.var(x) <= 0:
        return None, None
    X = np.column_stack([np.ones_like(x), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    _, codes = np.unique(c, return_inverse=True)
    k = int(codes.max()) + 1
    if k < 2:
        return float(beta[1]), None
    S = np.column_stack([np.bincount(codes, weights=X[:, 0] * u, minlength=k),
                         np.bincount(codes, weights=X[:, 1] * u, minlength=k)])
    V = XtX_inv @ (S.T @ S) @ XtX_inv * k / (k - 1)
    se = math.sqrt(V[1, 1]) if V[1, 1] > 0 else 0.0
    return float(beta[1]), (float(beta[1] / se) if se > 0 else None)


def _r(v: float | None, d: int = 2, scale: float = 100.0) -> float | None:
    return None if v is None else round(v * scale, d)


# ---- residual dynamics on a grid -----------------------------------------------------------------
def grid_panel(states: pl.DataFrame, *, every_s: int = 10, horizons: tuple[int, ...] = HORIZONS_S,
               end_ns: int | None = None, fairs: tuple[str, ...] = FAIRS) -> pl.DataFrame:
    """Sample each leg's latest state on a fixed grid and attach its state ``h`` seconds later
    (mid, bid, ask, fair) and 10 s earlier. Rows only where the leg has a two-sided book."""
    cols = ["t_ns", "ladder", "event", "market", "bid", "ask", "mid", "f_logistic", "mu", "sig",
            *[f"l_{f}" for f in fairs if f"l_{f}" in states.columns], "lb", "la"]
    cols = [c for c in cols if c in states.columns]
    st = states.select(cols).sort("t_ns")
    end_ns = end_ns or int(st["t_ns"].max())
    step = every_s * 10**9
    spans = st.group_by("ladder", "market").agg(pl.col("t_ns").min().alias("t0"))
    grids = spans.with_columns(
        pl.int_ranges((pl.col("t0") // step + 1) * step, end_ns, step).alias("t_ns")).explode("t_ns").drop("t0")
    grids = grids.filter(pl.col("t_ns").is_not_null()).with_columns(pl.col("t_ns").cast(pl.Int64)).sort("t_ns")
    base = grids.join_asof(st, on="t_ns", by=["ladder", "market"], strategy="backward")
    base = base.filter(pl.col("mid").is_not_null())
    fair_cols = [c for c in cols if c.startswith("l_") or c in ("lb", "la", "f_logistic", "mu")]
    for h in (-10, *horizons):
        tag = f"m{-h}" if h < 0 else f"{h}"
        fut = base.select("ladder", "market", "t_ns").with_columns((pl.col("t_ns") + h * 10**9).alias("t_h"))
        fut = fut.filter(pl.col("t_h") <= end_ns).sort("t_h")
        right = st.select("ladder", "market", pl.col("t_ns").alias("t_h"), "bid", "ask", "mid", *fair_cols)
        right = right.rename({c: f"{c}_{tag}" for c in right.columns if c not in ("ladder", "market", "t_h")})
        fut = fut.join_asof(right.sort("t_h"), on="t_h", by=["ladder", "market"], strategy="backward").drop("t_h")
        base = base.join(fut, on=["ladder", "market", "t_ns"], how="left")
    return base


def add_flags(df: pl.DataFrame, buckets: pl.DataFrame, split_ns: int, *, t_col: str = "t_ns",
              bucket_s: int = 300) -> pl.DataFrame:
    """Attach the leg's own bucket ``volatile`` (evaluation) and ``volatile_prev`` (decision-time)
    flags, and the dev/holdout split."""
    b = buckets.select("market", "bucket_ns", "volatile", "volatile_prev")
    return (df.with_columns((pl.col(t_col) // (bucket_s * 10**9) * (bucket_s * 10**9)).alias("bucket_ns"))
            .join(b, on=["market", "bucket_ns"], how="left")
            .with_columns(pl.col("volatile").fill_null(False), pl.col("volatile_prev").fill_null(False),
                          pl.when(pl.col(t_col) < split_ns).then(pl.lit("dev")).otherwise(pl.lit("holdout"))
                          .alias("split")))


SPLITS = (("all", None), ("volatile", True), ("calm", False))


def _subsets(df: pl.DataFrame) -> Iterator[tuple[str, str, pl.DataFrame]]:
    for sp in ("dev", "holdout"):
        d = df.filter(pl.col("split") == sp)
        for name, flag in SPLITS:
            yield sp, name, d if flag is None else d.filter(pl.col("volatile") == flag)


def residual_dynamics(panel: pl.DataFrame, fair: str = "logistic", thresholds: tuple[float, ...] = (0.0, 0.02, 0.04),
                      horizons: tuple[int, ...] = HORIZONS_S, mult: dict[str, float] | None = None) -> pl.DataFrame:
    """Per split: residual r = fair - mid (LOO). For |r| >= thr: how much of the gap the mid closes
    (``mid_toward``), how much the fair closes (``fit_toward``), the regression slopes of the mid and
    fair changes on r, and the exit-price markout of buying the cheap side as a taker at t and
    selling at the bid at t + h (fees both ways), in cents, t-stats clustered by event."""
    fc = f"l_{fair}"
    rows = []
    d0 = panel.filter(pl.col(fc).is_not_null()).with_columns((pl.col(fc) - pl.col("mid")).alias("r"))
    for sp, vol, d in _subsets(d0):
        for thr in thresholds:
            x = d.filter(pl.col("r").abs() >= thr - 1e-9)
            if x.height == 0:
                continue
            s = np.sign(x["r"].to_numpy())
            ev = x["event"].to_list()
            for h in horizons:
                if f"mid_{h}" not in x.columns:
                    continue
                dm = (x[f"mid_{h}"] - x["mid"]).to_numpy().astype(float)
                df_ = (x[f"{fc}_{h}"] - x[fc]).to_numpy().astype(float)
                rr = x["r"].to_numpy()
                m_mid, t_mid, n, k = clustered(dm * s, ev)
                m_fit, t_fit, _, _ = clustered(-df_ * s, ev)
                b_mid, tb_mid = slope_clustered(rr, dm, np.array(ev))
                b_fit, tb_fit = slope_clustered(rr, -df_, np.array(ev))
                mx = _exit_markout(x, h, s, mult or {})
                m_mx, t_mx, n_mx, _ = clustered(mx, ev)
                rows.append({"fair": fair, "split": sp, "vol": vol, "thr_c": thr * 100, "h_s": h,
                             "rows": n, "events": k, "abs_r_c": _r(float(np.mean(np.abs(rr)))),
                             "mid_toward_c": _r(m_mid), "t_mid": _r(t_mid, 2, 1),
                             "fit_toward_c": _r(m_fit), "t_fit": _r(t_fit, 2, 1),
                             "beta_mid": _r(b_mid, 3, 1), "tb_mid": _r(tb_mid, 2, 1),
                             "beta_fit": _r(b_fit, 3, 1), "tb_fit": _r(tb_fit, 2, 1),
                             "taker_exit_c": _r(m_mx), "t_exit": _r(t_mx, 2, 1), "n_exit": n_mx})
    return pl.DataFrame(rows)


def _vfee(price: np.ndarray, mult: np.ndarray) -> np.ndarray:
    """Vectorized ``arb_l2.taker_fee`` for one contract."""
    return np.ceil(np.round(0.07 * mult * price * (1 - price) * 1e4, 6)) / 1e4


def _exit_markout(x: pl.DataFrame, h: int, s: np.ndarray, mult: dict[str, float]) -> np.ndarray:
    """Buy YES at the ask (s > 0) or NO at 1 - bid (s < 0) now, sell at the bid h later; fees."""
    bid, ask = x["bid"].to_numpy(), x["ask"].to_numpy()
    if f"bid_{h}" not in x.columns:
        return np.full(len(s), np.nan)
    bh = x[f"bid_{h}"].to_numpy().astype(float)
    ah = x[f"ask_{h}"].to_numpy().astype(float)
    mu = np.array([mult.get(e.split("-")[0], 1.0) for e in x["event"].to_list()])
    entry = np.where(s > 0, ask, 1 - bid)
    ex = np.where(s > 0, bh, 1 - ah)
    out = ex - entry - _vfee(entry, mu) - _vfee(np.nan_to_num(ex, nan=0.5), mu)
    return np.where((s != 0) & np.isfinite(ex), out, np.nan)


def shift_response(panel: pl.DataFrame, fair: str = "logistic", horizons: tuple[int, ...] = (10, 60, 300),
                   min_move: float = 0.02) -> pl.DataFrame:
    """Which strikes lag when the curve shifts. ``d_others`` = change of the leg's LOO fair over the
    last 10 s (the other strikes moved), ``d_own`` = its own mid change over the same 10 s.
    Contemporaneous beta of d_own on d_others, and for fresh gaps (|d_others| >= ``min_move``, own
    mid unchanged within 0.5c) the share of the move the leg's mid makes over the next h seconds,
    split by moneyness of the full logistic fit (wing < 15% or > 85%, shoulder, near 30-70%)."""
    fc = f"l_{fair}"
    d = panel.filter(pl.col(fc).is_not_null() & pl.col(f"{fc}_m10").is_not_null() & pl.col("mid_m10").is_not_null())
    d = d.with_columns((pl.col(fc) - pl.col(f"{fc}_m10")).alias("d_others"),
                       (pl.col("mid") - pl.col("mid_m10")).alias("d_own"),
                       pl.when((pl.col("f_logistic") < 0.15) | (pl.col("f_logistic") > 0.85)).then(pl.lit("wing"))
                       .when((pl.col("f_logistic") >= 0.3) & (pl.col("f_logistic") <= 0.7)).then(pl.lit("near"))
                       .otherwise(pl.lit("shoulder")).alias("money"))
    rows = []
    for sp in ("dev", "holdout"):
        for money in ("wing", "shoulder", "near"):
            x = d.filter((pl.col("split") == sp) & (pl.col("money") == money))
            if x.height < 10:
                continue
            ev = np.array(x["event"].to_list())
            b, tb = slope_clustered(x["d_others"].to_numpy(), x["d_own"].to_numpy(), ev)
            fresh = x.filter((pl.col("d_others").abs() >= min_move) & (pl.col("d_own").abs() < 0.005))
            row = {"fair": fair, "split": sp, "money": money, "rows": x.height, "beta_contemp": _r(b, 3, 1),
                   "t_contemp": _r(tb, 2, 1), "fresh_moves": fresh.height,
                   "fresh_events": fresh["event"].n_unique() if fresh.height else 0}
            if fresh.height:
                s = np.sign(fresh["d_others"].to_numpy())
                size = float(np.mean(np.abs(fresh["d_others"].to_numpy())))
                row["fresh_size_c"] = _r(size)
                for h in horizons:
                    dm = (fresh[f"mid_{h}"] - fresh["mid"]).to_numpy().astype(float)
                    m, t, _, _ = clustered(dm * s, fresh["event"].to_list())
                    row[f"follow_{h}s_c"] = _r(m)
                    row[f"t_{h}s"] = _r(t, 2, 1)
            rows.append(row)
    return pl.DataFrame(rows)


def fit_quality(panel: pl.DataFrame, fairs: tuple[str, ...] = FAIRS, h: int = 300) -> pl.DataFrame:
    """Which fair predicts the leg's mid h seconds later best (dev and holdout): RMSE in cents of
    fair(t) vs mid(t+h), against the persistence forecast mid(t)."""
    rows = []
    for sp in ("dev", "holdout"):
        x = panel.filter((pl.col("split") == sp) & pl.col(f"mid_{h}").is_not_null())
        base = x.select(((pl.col(f"mid_{h}") - pl.col("mid")) ** 2).mean().sqrt()).item()
        for f in fairs:
            c = f"l_{f}"
            if c not in x.columns:
                continue
            y = x.filter(pl.col(c).is_not_null())
            if y.height == 0:
                continue
            rm = y.select(((pl.col(f"mid_{h}") - pl.col(c)) ** 2).mean().sqrt()).item()
            rp = y.select(((pl.col(f"mid_{h}") - pl.col("mid")) ** 2).mean().sqrt()).item()
            mix = y.select(((pl.col(f"mid_{h}") - 0.5 * pl.col("mid") - 0.5 * pl.col(c)) ** 2).mean().sqrt()).item()
            rows.append({"split": sp, "fair": f, "rows": y.height, "events": y["event"].n_unique(),
                         "rmse_fair_c": _r(rm), "rmse_mid_c": _r(rp), "rmse_half_c": _r(mix),
                         "rmse_mid_all_c": _r(base)})
    return pl.DataFrame(rows)


def sum_to_one(states: pl.DataFrame) -> pl.DataFrame:
    """Complete bracket sets: time-weighted distribution of sum(bids), sum(mids), sum(asks)."""
    x = states.filter(pl.col("complete")).with_columns((pl.col("leg").max().over("ladder") + 1).alias("n_legs"))
    x = x.filter((pl.col("leg") == 0)).sort("ladder", "t_ns")
    x = x.with_columns(pl.col("n_valid") == pl.col("n_legs")).rename({"n_valid": "full"})
    if x.height == 0:
        return pl.DataFrame()
    x = x.with_columns((pl.col("t_ns").shift(-1).over("ladder") - pl.col("t_ns")).fill_null(0).alias("dur"))
    out = []
    for lad, g in x.group_by("ladder"):
        dur = g["dur"].to_numpy().astype(float) * g["full"].to_numpy()
        if dur.sum() <= 0:
            continue
        g = g.with_columns(pl.col("sum_bid").fill_null(0.0), pl.col("sum_mid").fill_null(0.0),
                           pl.col("sum_ask").fill_null(0.0))
        row = {"ladder": lad[0], "states": g.height, "hours": round(dur.sum() / 3.6e12, 2)}
        for c in ("sum_bid", "sum_mid", "sum_ask"):
            v = g[c].to_numpy()
            row[f"{c}_avg"] = round(float(np.sum(v * dur) / dur.sum()), 4)
        row["share_ask_lt_1"] = round(float(np.sum(dur * (g["sum_ask"].to_numpy() < 1 - 1e-9)) / dur.sum()), 4)
        row["share_bid_gt_1"] = round(float(np.sum(dur * (g["sum_bid"].to_numpy() > 1 + 1e-9)) / dur.sum()), 4)
        out.append(row)
    return pl.DataFrame(out)


# ---- taker test ----------------------------------------------------------------------------------
class TobIndex:
    """Fast as-of lookups of a market's top of book (with sizes)."""

    def __init__(self, tob: pl.DataFrame) -> None:
        self.d: dict[str, tuple[list[int], list[tuple]]] = {}
        for (m,), g in tob.sort("t_ns").group_by("market", maintain_order=True):
            self.d[m] = (g["t_ns"].to_list(), list(zip(g["bid"].to_list(), g["bid_sz"].to_list(),
                                                       g["ask"].to_list(), g["ask_sz"].to_list())))
        self.end = int(tob["t_ns"].max()) if tob.height else 0

    def at(self, m: str, t: int) -> tuple | None:
        ts, vs = self.d.get(m, ([], []))
        i = bisect.bisect_right(ts, t) - 1
        return vs[i] if i >= 0 else None


def taker_signals(states: pl.DataFrame, fair: str, thr: float, mult: dict[str, float],
                  lo: float = 0.05, hi: float = 0.95) -> pl.DataFrame:
    """State rows where buying a side is cheap by at least fee + thr against the fair: YES when
    fair - ask >= fee(ask) + thr, NO when bid - fair >= fee(1 - bid) + thr, with the entry price in
    [lo, hi] (thin-tailed curves cannot price the 1-cent tick floor of far wings). ``fair`` is a
    LOO fair name or ``bidask`` (YES against the bid curve, NO against the ask curve)."""
    fy, fn = ("lb", "la") if fair == "bidask" else (f"l_{fair}", f"l_{fair}")
    x = states.filter(pl.col(fy).is_not_null() & pl.col(fn).is_not_null() & pl.col("bid").is_not_null())
    x = x.with_columns(pl.col("event").str.split("-").list.first().replace_strict(mult, default=1.0,
                                                                                    return_dtype=pl.Float64).alias("mult"))
    fee_y = (0.07 * pl.col("mult") * pl.col("ask") * (1 - pl.col("ask")) * 1e4 - 1e-9).ceil() / 1e4
    fee_n = (0.07 * pl.col("mult") * pl.col("bid") * (1 - pl.col("bid")) * 1e4 - 1e-9).ceil() / 1e4
    x = x.with_columns(pl.when(pl.col("ask").is_between(lo, hi)).then(pl.col(fy) - pl.col("ask") - fee_y)
                       .otherwise(None).alias("edge_yes"),
                       pl.when((1 - pl.col("bid")).is_between(lo, hi)).then(pl.col("bid") - pl.col(fn) - fee_n)
                       .otherwise(None).alias("edge_no"))
    x = x.filter((pl.col("edge_yes") >= thr - 1e-9).fill_null(False) | (pl.col("edge_no") >= thr - 1e-9).fill_null(False))
    return x.with_columns(pl.when(pl.col("edge_yes").fill_null(-9) >= pl.col("edge_no").fill_null(-9)).then(pl.lit("yes"))
                          .otherwise(pl.lit("no")).alias("side"),
                          pl.max_horizontal("edge_yes", "edge_no").alias("edge")).sort("t_ns")


def taker_trades(signals: pl.DataFrame, tob: TobIndex, results: dict[str, float | None], *,
                 latency_ms: float, cooldown_s: float = 300.0, exits_s: tuple[int, ...] = (60, 300),
                 mult: dict[str, float] | None = None, skip_volatile_prev: bool = False) -> list[dict[str, Any]]:
    """Execute signals: the order arrives ``latency_ms`` later and fills at the then-best price if
    it is still at or below the signal's price with at least one contract shown. One entry per
    market per ``cooldown_s``. Exits sell at the bid h seconds after the fill (taker fee), or hold
    to settlement."""
    lat = int(latency_ms * 1e6)
    last_entry: dict[str, int] = {}
    out = []
    cols = ["t_ns", "market", "event", "ladder", "side", "bid", "ask", "edge", "volatile", "volatile_prev", "split"]
    cols = [c for c in cols if c in signals.columns]
    for r in signals.select(cols).iter_rows(named=True):
        m, t = r["market"], r["t_ns"]
        if skip_volatile_prev and r.get("volatile_prev"):
            continue
        if t - last_entry.get(m, -(2**62)) < cooldown_s * 1e9:
            continue
        q = tob.at(m, t + lat)
        if q is None:
            continue
        b, bs, a, as_ = q
        mu = (mult or {}).get(r["event"].split("-")[0], 1.0)
        if r["side"] == "yes":
            if a is None or a > r["ask"] + 1e-9 or (as_ or 0) < 1:
                continue
            entry = a
        else:
            if b is None or b < r["bid"] - 1e-9 or (bs or 0) < 1:
                continue
            entry = round(1 - b, 6)
        last_entry[m] = t
        t_fill = t + lat
        fee = taker_fee(1, entry, mu)
        row = {**r, "t_fill": t_fill, "entry": entry, "fee": fee, "qty": 1.0}
        for h in exits_s:
            th = t_fill + h * 10**9
            qh = tob.at(m, th) if th <= tob.end else None
            ex = None
            mid = None
            if qh is not None:
                bh, _, ah, _ = qh
                if r["side"] == "yes":
                    ex = bh
                    mid = None if bh is None or ah is None else (bh + ah) / 2
                else:
                    ex = None if ah is None else 1 - ah
                    mid = None if bh is None or ah is None else 1 - (bh + ah) / 2
            row[f"mx_{h}s"] = None if ex is None else ex - entry - fee - taker_fee(1, ex, mu)
            row[f"mo_{h}s"] = None if mid is None else mid - entry
        res = results.get(m)
        row["settle"] = None if res is None else (res if r["side"] == "yes" else 1 - res) - entry - fee
        out.append(row)
    return out


def summarize_trades(rows: list[dict[str, Any]], cols: tuple[str, ...] = ("mo_60s", "mx_60s", "mo_300s", "mx_300s", "settle")) -> list[dict[str, Any]]:
    out = []
    for sp in ("dev", "holdout"):
        for name, flag in SPLITS:
            sub = [r for r in rows if r.get("split") == sp and (flag is None or bool(r.get("volatile")) == flag)]
            d: dict[str, Any] = {"split": sp, "vol": name, "fills": len(sub),
                                 "events": len({r["event"] for r in sub})}
            for c in cols:
                m, t, n, _ = clustered([r.get(c) for r in sub], [r["event"] for r in sub])
                d[f"{c}_c"] = _r(m)
                d[f"t_{c}"] = _r(t, 2, 1)
                if c == "settle":
                    d["settled"] = n
                    d["settle_pnl"] = round(sum(r["settle"] for r in sub if r.get("settle") is not None), 4)
            out.append(d)
    return out


# ---- maker test ----------------------------------------------------------------------------------
class CurveQuoter(Quoter):
    """``Quoter`` that reads the curve's fair value per leg (``fair`` events) and

    * ``mode="cheap"``: quotes only the side the curve calls cheap by at least ``thr`` (YES bid when
      fair - mid >= thr, NO bid when mid - fair >= thr), nothing otherwise;
    * ``mode="skew"``: quotes both sides but drops the side the curve calls rich by ``thr``;
    * ``mode="band"``: keeps every bid at least ``thr`` below the fair (YES <= fair - thr, NO <=
      1 - fair - thr);
    * ``mode="base"``: ignores the curve (baseline on the same markets).
    ``skip_volatile_prev``: no quotes in a market whose previous 5-minute bucket was volatile."""

    def __init__(self, *, mode: str = "cheap", thr: float = 0.02, vol_prev: dict[tuple[str, int], bool] | None = None,
                 skip_volatile_prev: bool = False, **kw: Any) -> None:
        super().__init__(**kw)
        self.mode, self.thr = mode, thr
        self.fair: dict[str, float | None] = {}
        self.vol_prev = vol_prev or {}
        self.skip_vp = skip_volatile_prev

    def _wanted(self, sim: Simulator, market: str) -> dict[str, float | None]:
        want = super()._wanted(sim, market)
        if self.skip_vp and self.vol_prev.get((market, sim.now // (300 * 10**9) * 300 * 10**9)):
            return {"yes": None, "no": None}
        if self.mode == "base" or (want["yes"] is None and want["no"] is None):
            return want
        fair = self.fair.get(market)
        if fair is None:
            return {"yes": None, "no": None}
        q = sim.markets[market].quote()
        mid = (q["yes_bid"] + q["yes_ask"]) / 2
        r = fair - mid
        if self.mode == "cheap":
            if r >= self.thr - 1e-9:
                want["no"] = None
            elif r <= -self.thr + 1e-9:
                want["yes"] = None
            else:
                return {"yes": None, "no": None}
        elif self.mode == "skew":
            if r >= self.thr - 1e-9:
                want["no"] = None
            elif r <= -self.thr + 1e-9:
                want["yes"] = None
        elif self.mode == "band":
            if want["yes"] is not None and want["yes"] > fair - self.thr + 1e-9:
                want["yes"] = None
            if want["no"] is not None and want["no"] > 1 - fair - self.thr + 1e-9:
                want["no"] = None
        return want


def fair_events(states: pl.DataFrame, fair: str) -> list[L2Event]:
    """The fair of each leg as ``ext`` events at state time + 1 ns (after the book event that
    produced it); None when the leg has no fair."""
    c = f"l_{fair}"
    x = states.select("t_ns", "market", c).sort("t_ns")
    x = x.filter((pl.col(c) != pl.col(c).shift(1).over("market")).fill_null(True)
                 | (pl.col(c).is_null() != pl.col(c).shift(1).over("market").is_null()).fill_null(True))
    return [L2Event(t + 1, "ext", m, {"mid": v}) for t, m, v in x.iter_rows()]


def run_maker(files: list[Path], markets: set[str], strategy: CurveQuoter, cfg: SimConfig,
              fairs: list[L2Event]) -> Simulator:
    sim = Simulator(cfg)
    events = heapq.merge(l2_events(files, markets), iter(fairs), key=lambda e: e.recv_ns)
    for ev in events:
        if ev.kind == "ext":
            strategy.fair[ev.market] = ev.msg.get("mid")
            sim._run_schedule(ev.recv_ns)
            sim.now = max(sim.now, ev.recv_ns)
        else:
            sim.apply(ev)
        strategy.on_event(sim, ev.market)
    return sim


def maker_rows(sim: Simulator, results: dict[str, float | None], strategy: CurveQuoter) -> list[dict[str, Any]]:
    from albert.backtest.l2 import markouts

    rows = markouts(sim)
    for r in rows:
        r["event"] = r["market"].rsplit("-", 1)[0]
        res = results.get(r["market"])
        r["settle"] = None if res is None else (res if r["side"] == "yes" else 1 - res) - r["price"] - r["fee"] / r["qty"]
        r["fair"] = None
    return rows


def summarize_maker(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for sp in ("dev", "holdout"):
        for name, flag in SPLITS:
            sub = [r for r in rows if r.get("split") == sp and (flag is None or bool(r.get("volatile")) == flag)]
            d: dict[str, Any] = {"split": sp, "vol": name, "fills": len(sub), "contracts": round(sum(r["qty"] for r in sub), 2),
                                 "events": len({r["event"] for r in sub})}
            for c in ("mo_60s", "mo_300s", "mx_60s", "mx_300s", "settle"):
                vals, cl = [], []
                for r in sub:
                    if r.get(c) is not None:
                        vals += [r[c]] * max(1, round(r["qty"]))
                        cl += [r["event"]] * max(1, round(r["qty"]))
                m, t, n, _ = clustered(vals, cl)
                d[f"{c}_c"] = _r(m)
                d[f"t_{c}"] = _r(t, 2, 1)
            d["settled_contracts"] = round(sum(r["qty"] for r in sub if r.get("settle") is not None), 2)
            d["settled_pnl"] = round(sum(r["settle"] * r["qty"] for r in sub if r.get("settle") is not None), 4)
            out.append(d)
    return out


# ---- historical: settled weather brackets (60-min candles) ----------------------------------------
_SUB = (re.compile(r"(-?\d+)\D+to (-?\d+)"), re.compile(r"(-?\d+)\D+or below"), re.compile(r"(-?\d+)\D+or above"))


def weather_legs(sub_title: str) -> tuple[float, float] | None:
    """(lo, hi) latent edges of a whole-degree weather bracket from its yes_sub_title."""
    s = sub_title or ""
    m = _SUB[0].search(s)
    if m:
        return float(m.group(1)) - 0.5, float(m.group(2)) + 0.5
    m = _SUB[1].search(s)
    if m:
        return -INF, float(m.group(1)) + 0.5
    m = _SUB[2].search(s)
    if m:
        return float(m.group(1)) - 0.5, INF
    return None


def historical_weather(tables: Path, *, holdout_from: str = "2026-10-01", families: tuple[str, ...] = ("logistic", "normal"),
                       thresholds: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10), every_h: int = 1,
                       series_prefix: tuple[str, ...] = ("KXHIGH", "KXLOWT"),
                       cache: Path | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Settled daily temperature bracket sets (complete, 6 legs) at each hourly candle close: fit
    the curve to the bracket mids, then ask (a) does the LOO residual predict settlement beyond the
    mid (slope of result - mid on fair - mid) and (b) does buying the cheap side at the candle's
    close ask/bid when fair - ask >= fee + thr pay at settlement. Returns (panel, summary)."""
    if cache is not None and cache.exists():
        return _hist_summary(pl.read_parquet(cache), families, thresholds)
    mk = (pl.scan_parquet(tables / "kalshi_markets.parquet")
          .filter(pl.col("series_ticker").str.starts_with(series_prefix[0])
                  | pl.col("series_ticker").str.starts_with(series_prefix[-1]))
          .filter(pl.col("result").is_in(["yes", "no"]))
          .select("ticker", "event_ticker", "series_ticker", "yes_sub_title", "result", "close_time",
                  "fee_multiplier").collect())
    mk = mk.with_columns(pl.col("yes_sub_title").map_elements(lambda s: list(weather_legs(s) or (None, None)),
                                                              return_dtype=pl.List(pl.Float64)).alias("edges"))
    mk = mk.with_columns(pl.col("edges").list.get(0).alias("lo"), pl.col("edges").list.get(1).alias("hi")).drop("edges")
    mk = mk.filter(pl.col("lo").is_not_null() | pl.col("hi").is_not_null())
    ev_n = mk.group_by("event_ticker").agg(pl.len().alias("n"), (pl.col("result") == "yes").sum().alias("yes"))
    good = ev_n.filter((pl.col("n") == 6) & (pl.col("yes") == 1))["event_ticker"]
    mk = mk.filter(pl.col("event_ticker").is_in(good.to_list()))
    cd = (pl.scan_parquet(tables / "kalshi_candles.parquet")
          .filter(pl.col("ticker").is_in(mk["ticker"].to_list()) & (pl.col("period_minutes") == 60))
          .select("ticker", "end_ts", "yes_bid_close", "yes_ask_close").collect())
    cd = cd.join(mk, on="ticker").filter(pl.col("end_ts") < pl.col("close_time"))
    if every_h > 1:
        cd = cd.filter(pl.col("end_ts").dt.hour() % every_h == 0)
    panel = _hist_fit(cd, families)
    if panel.is_empty():
        return panel, pl.DataFrame()
    panel = panel.with_columns(
        pl.when(pl.col("t") < datetime.fromisoformat(holdout_from).replace(tzinfo=UTC)).then(pl.lit("dev"))
        .otherwise(pl.lit("holdout")).alias("split"),
        ((pl.col("close") - pl.col("t")).dt.total_minutes() / 60).alias("h_to_close"))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(cache)
    return _hist_summary(panel, families, thresholds)


def _hist_fit(cd: pl.DataFrame, families: tuple[str, ...], chunk: int = 20000) -> pl.DataFrame:
    """Vectorized fits for every (event, candle) with all six legs quoted: one batched LM per
    family over all groups (full fit + six LOO fits each), in chunks of groups."""
    cd = cd.with_columns(pl.col("yes_bid_close").fill_null(0.0).alias("bid"),
                         pl.col("yes_ask_close").fill_null(1.0).alias("ask"),
                         pl.col("lo").fill_null(-INF), pl.col("hi").fill_null(INF))
    cd = cd.filter((pl.col("ask") > pl.col("bid")) & (pl.col("ask") > 0) & (pl.col("bid") < 1))
    cd = cd.with_columns(pl.len().over("event_ticker", "end_ts").alias("n")).filter(pl.col("n") == 6)
    cd = cd.sort("event_ticker", "end_ts", "lo")
    G = cd.height // 6
    if G == 0:
        return pl.DataFrame()
    shape = (G, 6)
    lo = cd["lo"].to_numpy().reshape(shape)
    hi = cd["hi"].to_numpy().reshape(shape)
    bid = cd["bid"].to_numpy().reshape(shape)
    ask = cd["ask"].to_numpy().reshape(shape)
    fin_lo = np.where(np.isfinite(lo), lo, np.nan)
    fin_hi = np.where(np.isfinite(hi), hi, np.nan)
    both = np.concatenate([fin_lo, fin_hi], axis=1)
    c = np.nanmedian(both, axis=1)[:, None]
    sc = (np.nanmax(both, axis=1) - np.nanmin(both, axis=1))[:, None]
    sc[sc <= 0] = 1.0
    lo, hi = (lo - c) / sc, (hi - c) / sc
    mid = (bid + ask) / 2
    w = 1.0 / np.maximum(ask - bid, MIN_SPREAD) ** 2
    # starting values: bracket moments of the mids
    cen = np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2,
                   np.where(np.isfinite(lo), lo + 0.1, hi - 0.1))
    q = np.clip(mid, 1e-6, None)
    mu0 = (q * cen).sum(1) / q.sum(1)
    sg0 = np.sqrt((q * (cen - mu0[:, None]) ** 2).sum(1) / q.sum(1)).clip(0.02, 5.0)
    out: dict[str, np.ndarray] = {}
    for fam in families:
        k = 3 if fam == "skewnormal" else 2
        loo = np.full(shape, np.nan)
        for a in range(0, G, chunk):
            b = min(G, a + chunk)
            g = b - a
            # rows: per group, LOO for each of the six legs
            L = np.repeat(lo[a:b], 6, axis=0)
            H = np.repeat(hi[a:b], 6, axis=0)
            P = np.repeat(mid[a:b], 6, axis=0)
            Wt = np.repeat(w[a:b], 6, axis=0)
            Wt[np.arange(g * 6), np.tile(np.arange(6), g)] = 0.0
            th0 = np.zeros((g * 6, k))
            th0[:, 0] = np.repeat(mu0[a:b], 6)
            th0[:, 1] = np.log(np.repeat(sg0[a:b], 6) * (1.0 if fam == "logistic" else 1.7))
            th = fit_batch(L, H, P, Wt, fam, th0)
            pr = leg_probs(L, H, th, fam)
            loo[a:b] = pr[np.arange(g * 6), np.tile(np.arange(6), g)].reshape(g, 6)
        out[f"l_{fam}"] = loo
    out["l_sum1"] = 1.0 - (mid.sum(1, keepdims=True) - mid)
    res = (cd["result"] == "yes").cast(pl.Float64).to_numpy()
    df = pl.DataFrame({"event": cd["event_ticker"], "t": cd["end_ts"], "close": cd["close_time"],
                       "series": cd["series_ticker"], "mult": cd["fee_multiplier"].fill_null(1.0).cast(pl.Float64),
                       "market": cd["ticker"], "leg": np.tile(np.arange(6), G), "bid": bid.ravel(),
                       "ask": ask.ravel(), "mid": mid.ravel(), "result": res,
                       "sum_mid": np.repeat(mid.sum(1), 6),
                       **{k: v.ravel() for k, v in out.items()}})
    return df.with_columns([pl.col(k).fill_nan(None) for k in out])


def _hist_summary(panel: pl.DataFrame, families: tuple[str, ...],
                  thresholds: tuple[float, ...]) -> tuple[pl.DataFrame, pl.DataFrame]:
    out = []
    spread = pl.col("ask") - pl.col("bid")
    for sp in ("dev", "holdout"):
        for win, cond in (("all", pl.lit(True)), ("spread<=5c", spread <= 0.05 + 1e-9),
                          ("spread<=2c", spread <= 0.02 + 1e-9), (">=24h", pl.col("h_to_close") >= 24),
                          ("12-24h", pl.col("h_to_close").is_between(12, 24, closed="left")),
                          ("<12h", pl.col("h_to_close") < 12)):
            x = panel.filter((pl.col("split") == sp) & cond)
            for fam in (*families, "sum1"):
                y = x.filter(pl.col(f"l_{fam}").is_not_null())
                if y.height == 0:
                    continue
                r = (y[f"l_{fam}"] - y["mid"]).to_numpy()
                z = (y["result"] - y["mid"]).to_numpy()
                ev = np.array(y["event"].to_list())
                # sum1's residual (1 - sum of mids) is shared by all six legs and sum(result) = 1, so
                # its pooled slope is 1/6 by construction: not a test
                b, tb = slope_clustered(r, z, ev) if fam != "sum1" else (None, None)
                res = y["result"].to_numpy()
                f = np.clip(y[f"l_{fam}"].to_numpy(), 0, 1)
                row = {"split": sp, "window": win, "fair": fam, "rows": y.height, "events": len(set(ev)),
                       "beta_settle_on_resid": _r(b, 3, 1), "t_beta": _r(tb, 2, 1),
                       "brier_mid": round(float(np.mean((y["mid"].to_numpy() - res) ** 2)), 5),
                       "brier_fair": round(float(np.mean((f - res) ** 2)), 5),
                       "brier_80mid_20fair": round(float(np.mean((0.8 * y["mid"].to_numpy() + 0.2 * f - res) ** 2)), 5)}
                for thr in thresholds:
                    pnl, cl = _hist_taker(y, fam, thr)
                    m, t, n, k = clustered(pnl, cl)
                    row[f"n_thr{int(thr * 100)}"] = n
                    row[f"pnl_thr{int(thr * 100)}_c"] = _r(m)
                    row[f"t_thr{int(thr * 100)}"] = _r(t, 2, 1)
                out.append(row)
    return panel, pl.DataFrame(out)


def _hist_taker(y: pl.DataFrame, fam: str, thr: float) -> tuple[list[float], list[str]]:
    """One entry per (event, leg, side): the first candle where the cheap side clears fee + thr,
    held to settlement."""
    seen: set[tuple[str, int, str]] = set()
    pnl, cl = [], []
    for r in y.sort("t").select("event", "leg", "bid", "ask", "result", "mult", f"l_{fam}").iter_rows():
        e, leg, b, a, res, mu, f = r
        if f is None:
            continue
        if f - a - taker_fee(1, a, mu) >= thr - 1e-9 and (e, leg, "yes") not in seen:
            seen.add((e, leg, "yes"))
            pnl.append(res - a - taker_fee(1, a, mu))
            cl.append(e)
        elif b - f - taker_fee(1, 1 - b, mu) >= thr - 1e-9 and (e, leg, "no") not in seen:
            seen.add((e, leg, "no"))
            pnl.append((1 - res) - (1 - b) - taker_fee(1, 1 - b, mu))
            cl.append(e)
    return pnl, cl


# ---- top level -----------------------------------------------------------------------------------
def strike_curves(s, *, start: str = "2026-11-05T03", end: str = "2026-11-05T18", split: str = "2026-11-05T11",
                  root: Path | None = None, families: tuple[str, ...] = FAMILIES, grid_s: int = 10,
                  taker_fairs: tuple[str, ...] = ("logistic", "normal", "skewnormal", "nbr", "sum1", "bidask"),
                  thresholds: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05),
                  maker: bool = True, maker_variants: tuple[tuple[str, float], ...] | None = None,
                  maker_fair: str | None = None, history: bool = True, refresh_results: bool = True) -> Path:
    """Run every part and write report ``research-strike-curves``. Parameters (fair, threshold)
    for the trading tests are chosen on dev (``start``..``split``) and reported on holdout."""
    from albert.backtest.l2 import stream_files
    from albert.reports import Report
    from albert.research.volatile import cached_buckets

    root = root or s.raw_dir / "stream"
    cache = s.data_dir / "cache"
    files = stream_files(root, start, end)
    if not files:
        raise RuntimeError("no recorded Kalshi stream files in range")
    split_ns = _ts_ns(f"{split[:10]}T{split[11:13]}:00:00Z")
    buckets = cached_buckets(s, start, end, root=root)
    recorded = set(buckets["market"].unique().to_list())
    by_event: dict[str, int] = defaultdict(int)
    for m in recorded:
        by_event[m.rsplit("-", 1)[0]] += 1
    payloads = fetch_events([e for e, k in by_event.items() if k >= 3], cache / "strike_curves_events.json")
    ser = pl.read_parquet(s.tables_dir / "kalshi_series.parquet")
    mult = {r["series_ticker"]: float(r["fee_multiplier"] or 1.0) for r in ser.iter_rows(named=True)}
    fee_info = {r["series_ticker"]: r for r in ser.iter_rows(named=True)}
    ladders = ladders_from_events(payloads, recorded, mult)
    markets = {lg.market for ld in ladders for lg in ld.legs}
    close = {lg.market: lg.close_ns for ld in ladders for lg in ld.legs if lg.close_ns}
    tob = cached_tob(files, markets, cache, close)
    tag = f"{start}_{end}".replace(":", "")
    st_path = cache / f"strike_curves_states_{tag}.parquet"
    if st_path.exists():
        states = pl.read_parquet(st_path)
    else:
        states = all_states(ladders, tob, families=families)
        states.write_parquet(st_path)
    states = add_flags(states, buckets, split_ns)
    results = market_results(sorted(markets), cache / "strike_curves_markets.json", refresh=refresh_results)
    rep = Report(s.reports_dir, "research-strike-curves")
    lad_tab = pl.DataFrame([{"ladder": ld.key, "kind": ld.kind, "legs": len(ld.legs), "event_legs": ld.n_event,
                             "complete": ld.complete,
                             "states": states.filter(pl.col("ladder") == ld.key)["t_ns"].n_unique(),
                             "settled_legs": sum(results.get(lg.market) is not None for lg in ld.legs)}
                            for ld in ladders]).sort("states", descending=True)
    rep.json("Run", {"start": start, "end": end, "split": split, "files": len(files), "ladders": len(ladders),
                     "threshold_ladders": sum(ld.kind == "threshold" for ld in ladders),
                     "bracket_sets": sum(ld.kind == "bracket" for ld in ladders),
                     "complete_bracket_sets": sum(ld.complete for ld in ladders), "legs": len(markets),
                     "tob_rows": tob.height, "states": states["t_ns"].n_unique(), "state_rows": states.height,
                     "settled_legs": sum(v is not None for v in results.values())})
    rep.table("Ladders", lad_tab, save_as="ladders", max_rows=80)
    rep.table("Sum-to-one on complete bracket sets (time-weighted)", sum_to_one(states))
    # residuals: size and drivers
    end_ns = int(tob["t_ns"].max())
    panel = add_flags(grid_panel(states, every_s=grid_s, end_ns=end_ns), buckets, split_ns)
    res_cols = [c for c in panel.columns if c.startswith("l_") and "_" not in c[2:]]
    size_rows = []
    for sp in ("dev", "holdout"):
        x = panel.filter(pl.col("split") == sp)
        for c in res_cols:
            r = (x[c] - x["mid"]).drop_nulls().abs().to_numpy()
            if len(r):
                size_rows.append({"split": sp, "fair": c[2:], "rows": len(r), "median_abs_c": _r(float(np.median(r))),
                                  "p90_abs_c": _r(float(np.quantile(r, 0.9))), "share_ge_2c": round(float(np.mean(r >= 0.02)), 3),
                                  "share_ge_5c": round(float(np.mean(r >= 0.05)), 3)})
    rep.table("Residual size (LOO fair - mid, grid samples)", pl.DataFrame(size_rows))
    rep.table("Which fair forecasts the mid 300 s ahead (RMSE, cents)", fit_quality(panel))
    dyn = pl.concat([residual_dynamics(panel, f, mult=mult) for f in FAIRS if f"l_{f}" in panel.columns],
                    how="diagonal_relaxed")
    dyn.write_parquet(rep.dir / "residual_dynamics.parquet")
    rep.table("Residual dynamics at 60 s: does the mid move to the fit or the fit to the mid (cents; t by "
              "event; all horizons in residual_dynamics.parquet)", dyn.filter(pl.col("h_s") == 60), max_rows=200)
    rep.table("Residual dynamics, logistic fair, all horizons",
              dyn.filter((pl.col("fair") == "logistic") & (pl.col("vol") == "all")), max_rows=60)
    rep.table("Curve shifts: which strikes lag (logistic LOO fair)", shift_response(panel), save_as="shift_response")
    # taker test
    tob_ix = TobIndex(tob)
    grid_rows, all_trades = [], {}
    for fair in taker_fairs:
        if fair != "bidask" and f"l_{fair}" not in states.columns:
            continue
        for thr in thresholds:
            sig = taker_signals(states, fair, thr, mult)
            for lat in LATENCIES_MS:
                tr = taker_trades(sig, tob_ix, results, latency_ms=lat, mult=mult)
                all_trades[(fair, thr, lat)] = tr
                for d in summarize_trades(tr):
                    if d["vol"] == "all":
                        grid_rows.append({"fair": fair, "thr_c": thr * 100, "lat_ms": lat, "signals": sig.height, **d})
    grid = pl.DataFrame(grid_rows)
    rep.table("Taker grid (all windows; choose on dev)", grid, save_as="taker_grid", max_rows=200)
    dev = grid.filter((pl.col("split") == "dev") & (pl.col("lat_ms") == 50) & (pl.col("fills") >= 20))
    best = None
    if dev.height:
        best_row = dev.sort("mx_300s_c", descending=True, nulls_last=True).row(0, named=True)
        best = (best_row["fair"], best_row["thr_c"] / 100)
        for lat in LATENCIES_MS:
            rep.table(f"Taker, chosen on dev: fair={best[0]} thr={best[1] * 100:.0f}c, latency {lat} ms",
                      pl.DataFrame(summarize_trades(all_trades[(best[0], best[1], lat)])))
    all_rows = [dict(r, fair_=k[0], thr=k[1], lat=k[2]) for k, v in all_trades.items() for r in v]
    if all_rows:
        pl.DataFrame(all_rows, infer_schema_length=None).write_parquet(rep.dir / "taker_trades.parquet")
    # maker test
    if maker:
        mf = maker_fair or "logistic"
        fairs = fair_events(states, mf)
        vol_prev = {(m, b): bool(v) for m, b, v in buckets.select("market", "bucket_ns", "volatile_prev").iter_rows()}
        # ("skew", 1.0) never skews: the baseline restricted to moments with a curve
        variants = maker_variants or (("base", 0.0), ("skew", 1.0), ("cheap", 0.02), ("cheap", 0.04),
                                      ("skew", 0.02), ("band", 0.0))
        mrows = []
        for mode, thr in variants:
            for lat in LATENCIES_MS:
                strat = CurveQuoter(mode=mode, thr=thr, improve=True, min_spread=0.05, cooldown_s=300,
                                    markets=markets, vol_prev=vol_prev)
                sim = run_maker(files, markets, strat, SimConfig(latency_ms=lat, fee_info=fee_info), fairs)
                rows = maker_rows(sim, results, strat)
                if rows:
                    fr = add_flags(pl.DataFrame(rows, infer_schema_length=None), buckets, split_ns).to_dicts()
                else:
                    fr = []
                for d in summarize_maker(fr):
                    mrows.append({"mode": mode, "thr_c": thr * 100, "lat_ms": lat, "orders": len(sim.orders), **d})
        rep.table(f"Maker: baseline Quoter vs curve-aware ({mf} LOO fair), fills split by own-bucket volatility",
                  pl.DataFrame(mrows), save_as="maker", max_rows=200)
    if history:
        panel_h, summ = historical_weather(s.tables_dir, cache=cache / "strike_curves_hist_panel.parquet")
        rep.table("Historical weather brackets (60-min candles): residual -> settlement, taker at candle close",
                  summ, save_as="historical", max_rows=60)
        rep.json("Historical coverage", {"rows": panel_h.height, "events": panel_h["event"].n_unique() if panel_h.height else 0})
    return rep.save()
