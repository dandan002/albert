"""Short-horizon momentum and order flow on recorded Kalshi L2 streams.

Does the Kalshi mid continue or revert after

* (a) a mid move of at least ``k`` over the last N seconds,
* (b) signed trade flow (taker-YES volume minus taker-NO volume) or a run of same-side trades,
* (c) top-of-book imbalance and order-flow imbalance (OFI, Cont-Kukanov-Stoikov) at the top level,
* (d) a single large trade?

Two sampling schemes: a fixed 60 s grid per market (continuous features, direction = sign) and
event triggers (a, b-run, d), de-clustered to one per market per ``gap_s``. Every feature uses
only events with ``recv_ns <= t`` (the decision time); outcomes are the signed mid change at
10 s / 60 s / 5 min / 30 min, the executable-price change (exit at the best bid on the side
bought), and a taker trade: enter at the executable price ``latency`` after ``t`` (displayed size
required), pay the taker fee ``0.07 P (1-P)`` (rounded up to $0.0001, this account's balance
precision), exit by crossing at the horizon (fee again) or hold to settlement. Standard errors
are clustered by event (and by market).

The maker overlay (``SignalQuoter``) is the backtest ``Quoter`` that stops quoting the side
about to be run over while a flow or momentum signal is active.
"""

import bisect
import math
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from albert.backtest.l2 import EPS, Event, Quoter, SimConfig, Simulator

HORIZONS_S = (10, 60, 300, 1800)
NS = 10**9


@dataclass
class Tape:
    """Top-of-book and trade history of one market (receive times)."""
    qt: list[int] = field(default_factory=list)
    bid: list[float | None] = field(default_factory=list)
    ask: list[float | None] = field(default_factory=list)
    bsz: list[float] = field(default_factory=list)
    asz: list[float] = field(default_factory=list)
    ofi: list[float] = field(default_factory=list)       # cumulative OFI at each quote change
    tt: list[int] = field(default_factory=list)
    tsgn: list[float] = field(default_factory=list)      # + taker bought YES, - taker bought NO
    tpx: list[float] = field(default_factory=list)       # yes price

    def i_at(self, t: int) -> int:
        return bisect.bisect_right(self.qt, t) - 1

    def quote(self, t: int) -> tuple[float | None, float | None, float, float]:
        i = self.i_at(t)
        if i < 0:
            return None, None, 0.0, 0.0
        return self.bid[i], self.ask[i], self.bsz[i], self.asz[i]

    def mid(self, t: int) -> float | None:
        b, a, _, _ = self.quote(t)
        return (b + a) / 2 if b is not None and a is not None else None

    def ofi_window(self, t0: int, t1: int) -> float:
        i, j = self.i_at(t0), self.i_at(t1)
        if j < 0:
            return 0.0
        return self.ofi[j] - (self.ofi[i] if i >= 0 else 0.0)

    def flow(self, t0: int, t1: int) -> tuple[float, float, int]:
        """(net signed volume, gross volume, trades) for trades with t0 < t <= t1."""
        i, j = bisect.bisect_right(self.tt, t0), bisect.bisect_right(self.tt, t1)
        s = self.tsgn[i:j]
        return sum(s), sum(abs(x) for x in s), len(s)


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


def collect(events: Iterable[Event]) -> tuple[dict[str, Tape], int]:
    """Replay events into a book and record top-of-book changes and trades per market.
    Returns (tapes, last recv_ns)."""
    sim = Simulator(SimConfig(writes_per_s=None))
    tapes: dict[str, Tape] = defaultdict(Tape)
    end = 0
    for ev in events:
        sim.apply(ev)
        end = ev.recv_ns
        tp = tapes[ev.market]
        if ev.kind == "trade":
            m = ev.msg
            sgn = 1.0 if m["taker_side"] == "yes" else -1.0
            tp.tt.append(ev.recv_ns)
            tp.tsgn.append(sgn * float(m["count_fp"]))
            tp.tpx.append(float(m["yes_price_dollars"]))
            continue
        if ev.market not in sim.synced:
            continue
        q = sim.markets[ev.market].quote()
        b, a = q["yes_bid"], q["yes_ask"]
        bs, as_ = q["yes_bid_size"] or 0.0, q["yes_ask_size"] or 0.0
        if tp.qt and (tp.bid[-1], tp.ask[-1], tp.bsz[-1], tp.asz[-1]) == (b, a, bs, as_):
            continue
        step = _ofi_step(tp.bid[-1], tp.ask[-1], tp.bsz[-1], tp.asz[-1], b, a, bs, as_) if tp.qt else 0.0
        tp.ofi.append((tp.ofi[-1] if tp.ofi else 0.0) + step)
        tp.qt.append(ev.recv_ns)
        tp.bid.append(b)
        tp.ask.append(a)
        tp.bsz.append(bs)
        tp.asz.append(as_)
    return dict(tapes), end


# ---- samples ---------------------------------------------------------------------------------
def taker_fee(p: float, multiplier: float = 1.0) -> float:
    raw = 0.07 * multiplier * p * (1 - p)
    return math.ceil(round(raw * 1e4, 9)) / 1e4


def outcomes(tp: Tape, t: int, d: float, end: int, *, latency_ns: int = 100_000_000,
             fee_mult: float = 1.0, result: float | None = None) -> dict[str, Any] | None:
    """Signed outcomes of a direction-``d`` signal at ``t`` (d=+1: YES expected to rise)."""
    b0, a0, _, _ = tp.quote(t)
    if b0 is None or a0 is None:
        return None
    mid0 = (b0 + a0) / 2
    row: dict[str, Any] = {"spread": a0 - b0, "mid": mid0}
    eb, ea, ebs, eas = tp.quote(t + latency_ns)
    # Taker entry: buy YES at the ask (d>0) or NO at 1-bid (d<0), displayed size >= 1.
    if d > 0:
        entry = ea if ea is not None and eas >= 1 - EPS else None
    else:
        entry = (1 - eb) if eb is not None and ebs >= 1 - EPS else None
    fee_in = taker_fee(entry, fee_mult) if entry is not None else None
    row["entry"] = entry
    for h in HORIZONS_S:
        th = t + h * NS
        if th > end:
            row[f"mid_{h}s"] = row[f"exit_{h}s"] = row[f"taker_{h}s"] = None
            continue
        b, a, _, _ = tp.quote(th)
        m = (b + a) / 2 if b is not None and a is not None else None
        row[f"mid_{h}s"] = None if m is None else d * (m - mid0)
        # Value of the contract bought now, sold by crossing at the horizon.
        sell = (b if d > 0 else (None if a is None else 1 - a))
        start = (a0 if d > 0 else 1 - b0)
        row[f"exit_{h}s"] = None if sell is None else sell - start
        row[f"taker_{h}s"] = (None if entry is None or sell is None else
                              sell - entry - fee_in - taker_fee(sell, fee_mult))
    row["settle"] = None if result is None else d * (result - mid0)
    row["taker_settle"] = (None if result is None or entry is None else
                           (result if d > 0 else 1 - result) - entry - fee_in)
    return row


def grid_samples(tapes: dict[str, Tape], end: int, *, every_s: float = 60.0,
                 fee_info: dict[str, dict] | None = None,
                 results: dict[str, float | None] | None = None) -> list[dict[str, Any]]:
    """Every market every ``every_s`` while two-sided: features from the past, outcomes after."""
    rows = []
    step = int(every_s * NS)
    for mk, tp in tapes.items():
        if not tp.qt:
            continue
        mult = ((fee_info or {}).get(mk.split("-", 1)[0], {}) or {}).get("fee_multiplier") or 1.0
        res = (results or {}).get(mk)
        t = (tp.qt[0] // step + 1) * step
        while t <= end - 10 * NS:
            m0 = tp.mid(t)
            if m0 is not None:
                _b, _a, bs, as_ = tp.quote(t)
                feats: dict[str, float] = {}
                for n in (60, 300):
                    prev = tp.mid(t - n * NS)
                    feats[f"mom_{n}"] = 0.0 if prev is None else m0 - prev
                    net, gross, _k = tp.flow(t - n * NS, t)
                    feats[f"flow_{n}"] = net
                    feats[f"flowimb_{n}"] = net / gross if gross > 0 else 0.0
                feats["imb"] = (bs - as_) / (bs + as_) if bs + as_ > 0 else 0.0
                for n in (10, 60):
                    depth = (bs + as_) / 2 or 1.0
                    feats[f"ofi_{n}"] = tp.ofi_window(t - n * NS, t) / depth
                base = outcomes(tp, t, 1.0, end, fee_mult=mult, result=res)
                if base is not None:
                    rows.append({"market": mk, "t_ns": t, **feats, **base})
            t += step
    return rows


def trigger_samples(tapes: dict[str, Tape], end: int, *, move: float = 0.02, move_s: float = 30.0,
                    run_n: int = 3, run_s: float = 30.0, run_min: float = 10.0,
                    big: float = 250.0, gap_s: float = 60.0,
                    fee_info: dict[str, dict] | None = None,
                    results: dict[str, float | None] | None = None) -> list[dict[str, Any]]:
    """Event triggers, one per (market, kind) per ``gap_s``:

    * ``move``: mid changed by >= ``move`` over the last ``move_s`` (checked at quote changes),
    * ``run``: the last ``run_n`` trades within ``run_s`` were all on one side, net >= ``run_min``,
    * ``big``: a single trade of >= ``big`` contracts.
    Direction is the direction of the move / taker flow."""
    rows = []
    gap = int(gap_s * NS)
    for mk, tp in tapes.items():
        mult = ((fee_info or {}).get(mk.split("-", 1)[0], {}) or {}).get("fee_multiplier") or 1.0
        res = (results or {}).get(mk)
        last: dict[str, int] = defaultdict(lambda: -(2**62))
        hits: list[tuple[int, str, float, float]] = []
        win = int(move_s * NS)
        for i, t in enumerate(tp.qt):
            b, a = tp.bid[i], tp.ask[i]
            if b is None or a is None:
                continue
            prev = tp.mid(t - win)
            if prev is None:
                continue
            dm = (b + a) / 2 - prev
            if abs(dm) >= move - EPS:
                hits.append((t, "move", math.copysign(1.0, dm), abs(dm)))
        rwin = int(run_s * NS)
        for i, t in enumerate(tp.tt):
            s = tp.tsgn[i]
            if abs(s) >= big:
                hits.append((t, "big", math.copysign(1.0, s), abs(s)))
            if i + 1 >= run_n and tp.tt[i + 1 - run_n] >= t - rwin:
                seg = tp.tsgn[i + 1 - run_n:i + 1]
                if (all(x > 0 for x in seg) or all(x < 0 for x in seg)) and abs(sum(seg)) >= run_min:
                    hits.append((t, "run", math.copysign(1.0, seg[0]), abs(sum(seg))))
        hits.sort()
        for t, kind, d, size in hits:
            if t - last[kind] < gap:
                continue
            last[kind] = t
            out = outcomes(tp, t, d, end, fee_mult=mult, result=res)
            if out is not None:
                rows.append({"market": mk, "t_ns": t, "kind": kind, "dir": d, "size": size, **out})
    return rows


def clustered(values: list[float], clusters: list[str]) -> tuple[float | None, float | None, int, int]:
    """(mean, t, n, clusters) with standard errors clustered by ``clusters`` (equal weights)."""
    s: dict[str, float] = defaultdict(float)
    n: dict[str, int] = defaultdict(int)
    for v, c in zip(values, clusters):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        s[c] += v
        n[c] += 1
    total = sum(n.values())
    if not total:
        return None, None, 0, 0
    mean = sum(s.values()) / total
    k = len(s)
    if k < 2:
        return mean, None, total, k
    var = k / (k - 1) * sum((s[c] - mean * n[c]) ** 2 for c in s)
    se = math.sqrt(var) / total
    return mean, (mean / se if se > 0 else None), total, k


def event_of(market: str) -> str:
    return market.rsplit("-", 1)[0]


# ---- maker overlay ---------------------------------------------------------------------------
class SignalQuoter(Quoter):
    """``Quoter`` that drops the side about to be run over while a signal is active.

    Up-pressure (taker YES buying, or the mid rising) lifts asks, i.e. fills our NO bid just
    before the price moves against it, so the NO side is withheld (and cancelled); down-pressure
    withholds the YES bid. ``signal``: ``flow`` (net taker volume over ``window_s`` >= ``flow_min``
    with >= 2 trades) or ``mom`` (mid move over ``window_s`` >= ``move``). Pressure lasts
    ``hold_s`` after the last trigger."""

    def __init__(self, *, signal: str = "flow", window_s: float = 60.0, flow_min: float = 20.0,
                 move: float = 0.02, hold_s: float = 60.0, **kw: Any) -> None:
        super().__init__(**kw)
        self.signal, self.window_ns, self.flow_min = signal, int(window_s * NS), flow_min
        self.move, self.hold_ns = move, int(hold_s * NS)
        self.trades: dict[str, deque] = defaultdict(deque)
        self.pressure: dict[str, tuple[float, int]] = {}

    def observe(self, sim: Simulator, ev: Event) -> None:
        t = ev.recv_ns
        if ev.kind == "trade" and self.signal == "flow":
            dq = self.trades[ev.market]
            dq.append((t, (1.0 if ev.msg["taker_side"] == "yes" else -1.0) * float(ev.msg["count_fp"])))
            while dq and dq[0][0] < t - self.window_ns:
                dq.popleft()
            net = sum(x for _, x in dq)
            if len(dq) >= 2 and abs(net) >= self.flow_min - EPS:
                self.pressure[ev.market] = (math.copysign(1.0, net), t + self.hold_ns)
        elif ev.kind in ("snapshot", "delta") and self.signal == "mom":
            now, prev = sim.mid_at(ev.market, t), sim.mid_at(ev.market, t - self.window_ns)
            if now is not None and prev is not None and abs(now - prev) >= self.move - EPS:
                self.pressure[ev.market] = (math.copysign(1.0, now - prev), t + self.hold_ns)

    def _wanted(self, sim: Simulator, market: str) -> dict[str, float | None]:
        want = super()._wanted(sim, market)
        p = self.pressure.get(market)
        if p is not None and sim.now <= p[1]:
            want["no" if p[0] > 0 else "yes"] = None
        return want


def run_overlay(events: Iterable[Event], strategy: Quoter, cfg: SimConfig) -> Simulator:
    """``backtest.l2.run`` with an ``observe`` hook so the strategy sees each event."""
    sim = Simulator(cfg)
    observe = getattr(strategy, "observe", None)
    for ev in events:
        sim.apply(ev)
        if observe is not None:
            observe(sim, ev)
        strategy.on_event(sim, ev.market)
    return sim


def stream_roots(paths: list[Path]) -> list[Path]:
    return [p for p in paths if (p / "kalshi").exists()]
