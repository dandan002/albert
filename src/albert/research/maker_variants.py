"""Maker strategy variants on the L2 replay, scored on what is realizable, not just the mid.

``Variant`` extends ``backtest.l2.Quoter`` with the levers the exit-price markouts point at:

* ``ticks``: improve this many ticks inside the best (1 = baseline). Falls back to 1 tick when
  the spread cannot hold both sides at that depth.
* ``micro_edge``: quote a side only if our bid sits at least this far below the microprice
  (size-weighted fair value of the external top of book).
* ``skew``: after a fill, quote the position-reducing side this many ticks more aggressively
  (never crossing) and let the cooldown block only the position-increasing side, to round-trip.
* ``tox_window_s`` / ``tox_min`` / ``tox_frac``: pull both sides of a market while recent taker
  flow is one-sided (at least ``tox_min`` contracts, ``tox_frac`` of the volume in the window).
* ``clear_window_s``: pull a market for this long after a recorded execution cleared a level.
* ``hist_min_fills`` / ``hist_floor``: stop quoting a market once its own matured fills (older
  than ``hist_h_s``, so no look-ahead) average a mid markout below ``hist_floor``.
* ``min_behind``: improve only if the level we step in front of shows at least this size.
* ``exclude_series``: series prefixes not to trade (e.g. KXTRUMPSAY); ``categories`` as in Quoter.
* ``mom_window_s`` / ``mom_move``: momentum tilt. When the (external) mid moved at least
  ``mom_move`` over the last ``mom_window_s`` (or since the first recorded quote, if shorter),
  do not quote the side that bets against the move (falling mid: no YES bid; rising: no NO bid).

``run_variant`` replays with the same Simulator (latency, write budget, risk-averse queue) and
calls ``observe(sim, ev)`` after each event so the strategy can read trades and executions.
``score`` adds settlement, exit-price markouts (from ``backtest.l2.markouts``) and round trips
(YES and NO bought in one market net to $1: realized spread = 1 - p_yes - p_no - fees).
"""

import heapq
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterator

from albert.backtest.l2 import (EPS, Event, Quoter, SimConfig, Simulator, iter_events, markouts)


class Variant(Quoter):
    def __init__(self, *, ticks: int = 1, micro_edge: float | None = None, skew: int = 0,
                 tox_window_s: float = 0.0, tox_min: float = 10.0, tox_frac: float = 0.8,
                 clear_window_s: float = 0.0, hist_min_fills: int = 0, hist_floor: float = 0.0,
                 hist_h_s: float = 300.0, min_behind: float = 0.0,
                 exclude_series: tuple[str, ...] = (), mom_window_s: float = 0.0,
                 mom_move: float = 0.03, **kw: Any) -> None:
        super().__init__(**kw)
        self.ticks, self.micro_edge, self.skew = ticks, micro_edge, skew
        self.tox_ns, self.tox_min, self.tox_frac = int(tox_window_s * 1e9), tox_min, tox_frac
        self.clear_ns = int(clear_window_s * 1e9)
        self.hist_min_fills, self.hist_floor = hist_min_fills, hist_floor
        self.hist_ns = int(hist_h_s * 1e9)
        self.min_behind = min_behind
        self.exclude_series = tuple(exclude_series)
        self.mom_ns, self.mom_move = int(mom_window_s * 1e9), mom_move
        self._trades: dict[str, deque] = defaultdict(deque)       # (t, taker_side, count)
        self._last_clear: dict[str, int] = {}
        self._hist_next = 0                                        # index into sim.fills
        self._hist: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])   # sum mo, n
        self.banned: set[str] = set()
        self.pulls: dict[str, int] = defaultdict(int)

    # -- inputs --
    def observe(self, sim: Simulator, ev: Event) -> None:
        if ev.kind == "trade" and self.tox_ns:
            q = self._trades[ev.market]
            q.append((ev.recv_ns, ev.msg.get("taker_outcome_side") or ev.msg.get("taker_side"),
                      float(ev.msg.get("count_fp") or 0)))
            while q and q[0][0] < ev.recv_ns - self.tox_ns:
                q.popleft()
        elif ev.kind == "delta" and ev.is_exec and self.clear_ns:
            lev = sim.markets[ev.market].levels[ev.msg["side"]].get(float(ev.msg["price_dollars"]))
            if lev is None or lev.ext_total() <= EPS:
                self._last_clear[ev.market] = ev.recv_ns

    def _update_hist(self, sim: Simulator) -> None:
        fills = sim.fills
        while self._hist_next < len(fills) and fills[self._hist_next].t_ns + self.hist_ns <= sim.now:
            f = fills[self._hist_next]
            mid = sim.mid_at(f.market, f.t_ns + self.hist_ns)
            if mid is not None:
                h = self._hist[f.market]
                h[0] += ((mid if f.side == "yes" else 1 - mid) - f.price) * f.qty
                h[1] += f.qty
                if h[1] >= self.hist_min_fills and h[0] / h[1] < self.hist_floor:
                    self.banned.add(f.market)
            self._hist_next += 1

    def _toxic(self, sim: Simulator, market: str) -> bool:
        if self.tox_ns:
            q = self._trades.get(market)
            if q:
                cut = sim.now - self.tox_ns
                yes = sum(c for t, s, c in q if t >= cut and s == "yes")
                no = sum(c for t, s, c in q if t >= cut and s == "no")
                tot = yes + no
                if tot >= self.tox_min and max(yes, no) >= self.tox_frac * tot:
                    return True
        if self.clear_ns and sim.now - self._last_clear.get(market, -(2**62)) < self.clear_ns:
            return True
        return False

    def _eligible(self, market: str) -> bool:
        if market in self.banned or (self.exclude_series and market.startswith(self.exclude_series)):
            return False
        return super()._eligible(market)

    # -- prices --
    def _wanted(self, sim: Simulator, market: str) -> dict[str, float | None]:
        base = super()._wanted(sim, market)
        pos = sim.position[market]
        reducing = "no" if pos > EPS else "yes" if pos < -EPS else None
        cooling = sim.now < self._cool_until.get(market, 0)
        if cooling and not (self.skew and reducing):
            return base
        q = sim.markets[market].quote()
        if q["yes_bid"] is None or q["yes_ask"] is None:
            return base
        if self._toxic(sim, market):
            self.pulls[market] += 1
            return {"yes": None, "no": None}
        if cooling:      # skew: only the reducing side may quote during the cooldown
            want: dict[str, float | None] = {"yes": None, "no": None}
            spread = q["yes_ask"] - q["yes_bid"]
            if not (self.lo <= (q["yes_ask"] + q["yes_bid"]) / 2 <= self.hi) or spread < 0.02 - EPS:
                return want
            want[reducing] = self._price(q, reducing, 1 + self.skew)
            return want
        want = dict(base)
        spread = q["yes_ask"] - q["yes_bid"]
        k = self.ticks if spread >= (2 * self.ticks + 1) * 0.01 - EPS else 1
        extra = {"yes": 0, "no": 0}
        if self.skew and reducing:
            extra[reducing] = self.skew
        for side in ("yes", "no"):
            if want[side] is None:
                continue
            p = self._price(q, side, k + extra[side]) if self.improve else want[side]
            if self.min_behind:
                size = q["yes_bid_size"] if side == "yes" else q["yes_ask_size"]
                if (size or 0) < self.min_behind:
                    p = None
            if p is not None and self.micro_edge is not None:
                fair = _microprice(q)
                fv = fair if side == "yes" else 1 - fair
                if p > fv - self.micro_edge + EPS:
                    p = None
            want[side] = p
        if self.mom_ns:
            move = self._momentum(sim, market)
            if move >= self.mom_move - EPS:
                want["no"] = None
            elif move <= -self.mom_move + EPS:
                want["yes"] = None
        return want

    def _momentum(self, sim: Simulator, market: str) -> float:
        """Mid now minus the mid ``mom_window_s`` ago (or at the first two-sided quote)."""
        now_mid = sim.mid_at(market, sim.now)
        if now_mid is None:
            return 0.0
        then = sim.mid_at(market, sim.now - self.mom_ns)
        if then is None:
            times, vals = sim.quotes.get(market, ([], []))
            then = next(((b + a) / 2 for b, a in vals if b is not None and a is not None), now_mid)
        return now_mid - then

    @staticmethod
    def _price(q: dict[str, Any], side: str, ticks: int) -> float | None:
        """``ticks`` inside our side's best bid, capped one tick short of crossing."""
        if side == "yes":
            p, cap = q["yes_bid"] + 0.01 * ticks, q["yes_ask"] - 0.01
        else:
            p, cap = 1 - q["yes_ask"] + 0.01 * ticks, 1 - q["yes_bid"] - 0.01
        p = round(min(p, cap), 6)
        return p if 0.0 < p < 1.0 else None

    def on_event(self, sim: Simulator, market: str) -> None:
        if self.hist_min_fills:
            self._update_hist(sim)
        if self.skew:
            # The base class starts a both-sides cooldown on every fill; keep that, and let
            # _wanted open the reducing side through it.
            pass
        super().on_event(sim, market)


def _microprice(q: dict[str, Any]) -> float:
    b, a = q["yes_bid"], q["yes_ask"]
    bs, as_ = q["yes_bid_size"] or 0.0, q["yes_ask_size"] or 0.0
    if bs + as_ <= 0:
        return (a + b) / 2
    return (b * as_ + a * bs) / (bs + as_)


def run_variant(files: list[Path], strategy: Variant, cfg: SimConfig,
                ext: Iterator[Event] | None = None) -> tuple[Simulator, dict[str, int]]:
    sim = Simulator(cfg)
    stats: dict[str, int] = defaultdict(int)
    events: Iterator[Event] = iter_events(files, stats)
    if ext is not None:
        events = heapq.merge(events, ext, key=lambda e: e.recv_ns)
    for ev in events:
        sim.apply(ev)
        stats["events"] += 1
        strategy.observe(sim, ev)
        strategy.on_event(sim, ev.market)
    return sim, stats


def round_trips(fills: list[dict[str, Any]]) -> dict[str, float]:
    """Pair YES and NO purchases within each market (FIFO): every pair settles to $1 for sure."""
    by_m: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: {"yes": [], "no": []})
    pnl = paired = 0.0
    for f in sorted(fills, key=lambda r: r["t_ns"]):
        book = by_m[f["market"]]
        other = book["no" if f["side"] == "yes" else "yes"]
        q, fee_pc = f["qty"], f["fee"] / f["qty"]
        while q > EPS and other:
            o = other[0]
            take = min(q, o[0])
            pnl += take * (1 - f["price"] - o[1] - fee_pc - o[2])
            paired += take
            o[0] -= take
            q -= take
            if o[0] <= EPS:
                other.pop(0)
        if q > EPS:
            book[f["side"]].append([q, f["price"], fee_pc])
    total = sum(f["qty"] for f in fills)
    return {"contracts": total, "paired": paired, "paired_frac": paired / total if total else 0.0,
            "paired_pnl": pnl, "paired_c_per_pair": 100 * pnl / paired if paired else None}


def clustered(rows: list[dict[str, Any]], col: str, key: str = "event") -> tuple[float | None, float | None, int]:
    """Contract-weighted mean of ``col`` and a t-stat clustered by ``key``."""
    g: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        if r.get(col) is None:
            continue
        g[r[key]][0] += r[col] * r["qty"]
        g[r[key]][1] += r["qty"]
    if not g:
        return None, None, 0
    n = sum(v[1] for v in g.values())
    mean = sum(v[0] for v in g.values()) / n
    k = len(g)
    if k < 2:
        return mean, None, k
    se = math.sqrt(k / (k - 1) * sum((s - mean * m) ** 2 for s, m in g.values())) / n
    return mean, (mean / se if se > 0 else None), k


def score(sim: Simulator, results: dict[str, float | None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summary and per-fill rows. ``results``: market -> 1.0/0.0 settled YES result or None."""
    rows = markouts(sim)
    for r in rows:
        r["event"] = r["market"].rsplit("-", 1)[0]
        res = results.get(r["market"])
        r["settle"] = None if res is None else (res if r["side"] == "yes" else 1 - res) - r["price"] - r["fee"] / r["qty"]
    out: dict[str, Any] = {"fills": len(rows), "contracts": sum(r["qty"] for r in rows),
                           "orders": len(sim.orders)}
    for col in ("mo_300s", "mx_300s", "settle"):
        mean, t, k = clustered(rows, col)
        out[col] = None if mean is None else round(100 * mean, 2)
        out[f"t_{col}"] = None if t is None else round(t, 2)
        out[f"events_{col}"] = k
    settled = [r for r in rows if r["settle"] is not None]
    out["settled_contracts"] = sum(r["qty"] for r in settled)
    out["settled_pnl"] = round(sum(r["settle"] * r["qty"] for r in settled), 3)
    out.update({f"rt_{k}": (round(v, 3) if isinstance(v, float) else v)
                for k, v in round_trips(rows).items()})
    return out, rows
