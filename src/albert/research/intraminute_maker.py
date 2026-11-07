"""Intra-minute maker strategies in the L2 replay (seconds to enter, seconds to a minute to hold).

Entry triggers (``VacatedLevel``), on the event where the best price on one side disappears:

* ``cause="pull"``: the level was cancelled, not traded (quote flicker). Post near the old
  price inside the new, wider spread and expect the snap-back.
* ``cause="exec"``: a taker cleared the level (refill). Post at the cleared price, or behind it.

``offset`` is in ticks behind the vacated price (0 = at it). An unfilled entry is cancelled
after ``ttl_s``. ``scalp_k`` (ticks): after an entry fills, post the opposite side at entry +
k ticks to flip the contract within ``scalp_ttl_s``; leftover inventory is held (valued at the
mid, the exit price, or settlement).

``LifetimeTracker`` measures, for every resting order, how long it stays alone at the front
(until someone joins at its price or steps ahead) and when it fills.

Decisions use only the book after the current event (no look-ahead). The simulator is the
backtest's (queue position, latency, post-only, write budget).
"""

import heapq
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from albert.backtest.l2 import EPS, Event, SimConfig, Simulator, iter_events

TICK = 0.01


class LifetimeTracker:
    """Per order: active, first joined (external size at our price), first stepped ahead
    (external best better than our price), filled, and ended (cancelled/filled)."""

    def __init__(self) -> None:
        self.rec: dict[int, dict[str, Any]] = {}

    def observe(self, sim: Simulator, market: str) -> None:
        m = sim.markets[market]
        for oid in sim.by_market.get(market, ()):
            o = sim.orders[oid]
            if o.status not in ("active", "cancelling"):
                if oid in self.rec and self.rec[oid].get("end_ns") is None and o.status in (
                        "filled", "cancelled"):
                    self.rec[oid]["end_ns"] = sim.now
                continue
            r = self.rec.setdefault(oid, {"market": market, "side": o.side, "price": o.price,
                                          "active_ns": o.active_ns, "joined_ns": None,
                                          "ahead_ns": None, "end_ns": None})
            lev = m.levels[o.side].get(o.price)
            if r["joined_ns"] is None and lev is not None:
                # external size queued behind us at our price
                seen = False
                for c in lev.q:
                    if c[1] == oid:
                        seen = True
                    elif seen and c[1] is None and c[0] > EPS:
                        r["joined_ns"] = sim.now
                        break
            best = m.best(o.side)
            if r["ahead_ns"] is None and best is not None and best[0] > o.price + EPS:
                r["ahead_ns"] = sim.now


class VacatedLevel:
    def __init__(self, *, cause: str = "pull", offset: int = 0, ttl_s: float = 10.0,
                 min_spread_after: float = 0.03, max_spread_before: float = 1.0,
                 lo: float = 0.05, hi: float = 0.95, max_pos: float = 3.0, size: float = 1.0,
                 cooldown_s: float = 30.0, scalp_k: int | None = None, scalp_ttl_s: float = 60.0,
                 exclude: tuple[str, ...] = (), direction: str = "fade") -> None:
        assert cause in ("pull", "exec", "any") and direction in ("fade", "lean")
        self.direction = direction
        self.cause, self.offset, self.ttl = cause, offset, int(ttl_s * 1e9)
        self.min_after, self.max_before = min_spread_after, max_spread_before
        self.lo, self.hi, self.max_pos, self.size = lo, hi, max_pos, size
        self.cool = int(cooldown_s * 1e9)
        self.scalp_k, self.scalp_ttl = scalp_k, int(scalp_ttl_s * 1e9)
        self.exclude = exclude
        self.prev: dict[str, tuple[float | None, float | None]] = {}
        self.expiry: list[tuple[int, int]] = []
        self.last_entry: dict[str, int] = {}
        self.role: dict[int, str] = {}            # oid -> entry | scalp
        self.scalp_of: dict[int, int] = {}        # scalp oid -> entry fill index
        self._seen_fills = 0
        self.triggers = 0
        self.tracker = LifetimeTracker()

    def _expire(self, sim: Simulator) -> None:
        while self.expiry and self.expiry[0][0] <= sim.now:
            _, oid = heapq.heappop(self.expiry)
            if sim.orders[oid].status in ("pending", "active"):
                sim.cancel(oid)

    def _post(self, sim: Simulator, market: str, side: str, price: float, ttl: int, role: str) -> int | None:
        price = round(price, 2)
        if not 0.01 - EPS <= price <= 0.99 + EPS:
            return None
        oid = sim.place(market, side, price, self.size)
        if oid is not None:
            self.role[oid] = role
            heapq.heappush(self.expiry, (sim.now + ttl, oid))
        return oid

    def _scalps(self, sim: Simulator) -> None:
        for i in range(self._seen_fills, len(sim.fills)):
            f = sim.fills[i]
            if self.scalp_k is None or self.role.get(f.oid) != "entry":
                continue
            # Sell what we bought at entry + k ticks: a bid on the other side at 1 - target.
            target = f.price + self.scalp_k * TICK
            q = sim.markets[f.market].quote()
            other = "no" if f.side == "yes" else "yes"
            # our side's best bid: posting the other side at >= 1 - that would cross
            own_bid = q["yes_bid"] if f.side == "yes" else (
                None if q["yes_ask"] is None else round(1 - q["yes_ask"], 6))
            price = 1 - target
            if own_bid is not None:
                price = min(price, 1 - own_bid - TICK)
            oid = self._post(sim, f.market, other, price, self.scalp_ttl, "scalp")
            if oid is not None:
                self.scalp_of[oid] = i
        self._seen_fills = len(sim.fills)

    def on_event(self, sim: Simulator, ev: Event) -> None:
        self._expire(sim)
        self._scalps(sim)
        market = ev.market
        if ev.kind == "ext" or market not in sim.synced:
            return
        q = sim.markets[market].quote()
        prev = self.prev.get(market)
        self.prev[market] = (q["yes_bid"], q["yes_ask"])
        self.tracker.observe(sim, market)
        if prev is None or ev.kind != "delta" or float(ev.msg["delta_fp"]) >= 0:
            return
        if self.cause == "pull" and ev.is_exec or self.cause == "exec" and not ev.is_exec:
            return
        if market.split("-", 1)[0] in self.exclude:
            return
        pb, pa = prev
        nb, na = q["yes_bid"], q["yes_ask"]
        if None in (pb, pa, nb, na):
            return
        if pa - pb > self.max_before + EPS or na - nb < self.min_after - EPS:
            return
        mid = (nb + na) / 2
        if not self.lo <= mid <= self.hi:
            return
        if sim.now - self.last_entry.get(market, -(2**62)) < self.cool:
            return
        pos = sim.position[market]
        # Which side's best price vanished on this event?
        side = ev.msg["side"]
        if self.direction == "lean":
            # Lean with the move: the vacated side signals the price is heading away from it,
            # so improve the *other* side by one tick (sell into the move) instead.
            if side == "yes" and nb < pb - EPS and pos > -self.max_pos:
                price = (1 - na) + TICK                   # NO bid one tick above the NO best
                if price < (1 - nb) - EPS:
                    self.triggers += 1
                    if self._post(sim, market, "no", price, self.ttl, "entry") is not None:
                        self.last_entry[market] = sim.now
            elif side == "no" and na > pa + EPS and pos < self.max_pos:
                price = nb + TICK
                if price < na - EPS:
                    self.triggers += 1
                    if self._post(sim, market, "yes", price, self.ttl, "entry") is not None:
                        self.last_entry[market] = sim.now
            return
        if side == "yes" and nb < pb - EPS and pos < self.max_pos:
            price = pb - self.offset * TICK
            if nb + EPS < price < na - EPS:
                self.triggers += 1
                if self._post(sim, market, "yes", price, self.ttl, "entry") is not None:
                    self.last_entry[market] = sim.now
        elif side == "no" and na > pa + EPS and pos > -self.max_pos:
            price = (1 - pa) - self.offset * TICK         # NO bid near the old NO best
            if (1 - na) + EPS < price < (1 - nb) - EPS:
                self.triggers += 1
                if self._post(sim, market, "no", price, self.ttl, "entry") is not None:
                    self.last_entry[market] = sim.now


class QuoterTracked:
    """Wraps backtest.l2.Quoter so the lifetime tracker sees its orders."""

    def __init__(self, quoter) -> None:
        self.q = quoter
        self.tracker = LifetimeTracker()
        self.role: dict[int, str] = {}
        self.scalp_of: dict[int, int] = {}
        self.triggers = 0

    def on_event(self, sim: Simulator, ev: Event) -> None:
        self.q.on_event(sim, ev.market)
        if ev.kind != "ext" and ev.market in sim.synced:
            self.tracker.observe(sim, ev.market)


def run_intra(files: list[Path], strategy, cfg: SimConfig) -> Simulator:
    sim = Simulator(cfg)
    stats: dict[str, int] = defaultdict(int)
    for ev in iter_events(files, stats):
        sim.apply(ev)
        strategy.on_event(sim, ev)
    return sim


# ---- scoring -----------------------------------------------------------------------------------
def taker_fee(price: float) -> float:
    return math.ceil(0.07 * price * (1 - price) * 1e4 - 1e-9) / 1e4


def fill_rows(sim: Simulator, strategy, results: dict[str, float | None],
              horizons: tuple[int, ...] = (10, 60, 300)) -> list[dict[str, Any]]:
    end = max((t[-1] for t, _ in sim.quotes.values() if t), default=0)
    role = getattr(strategy, "role", {})
    scalp_of = getattr(strategy, "scalp_of", {})
    paired = {scalp_of[f.oid]: j for j, f in enumerate(sim.fills) if f.oid in scalp_of}
    rows = []
    for i, f in enumerate(sim.fills):
        r = {"t_ns": f.t_ns, "market": f.market, "event": f.market.rsplit("-", 1)[0], "side": f.side,
             "price": f.price, "qty": f.qty, "fee": f.fee, "how": f.how,
             "role": role.get(f.oid, "entry"), "paired": i in paired,
             "age_s": (f.t_ns - sim.orders[f.oid].active_ns) / 1e9}
        for h in horizons:
            t = f.t_ns + h * 10**9
            mid = sim.mid_at(f.market, t) if t <= end else None
            ex = sim.exit_at(f.market, f.side, t) if t <= end else None
            r[f"mo_{h}s"] = None if mid is None else (mid if f.side == "yes" else 1 - mid) - f.price
            # exit = sell at the bid and pay the taker fee
            r[f"mx_{h}s"] = None if ex is None else ex - taker_fee(ex) - f.price
        res = results.get(f.market)
        r["settle"] = None if res is None else (res if f.side == "yes" else 1 - res) - f.price - f.fee / f.qty
        if i in paired:
            g = sim.fills[paired[i]]
            r["rt_pnl"] = 1 - f.price - g.price - (f.fee + g.fee) / f.qty
        rows.append(r)
    return rows


def clustered(rows: list[dict[str, Any]], col: str, cluster: str = "event") -> tuple[float | None, float | None, int]:
    g: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        v = r.get(col)
        if v is None:
            continue
        g[r[cluster]][0] += v * r["qty"]
        g[r[cluster]][1] += r["qty"]
    if not g:
        return None, None, 0
    n = sum(x[1] for x in g.values())
    mean = sum(x[0] for x in g.values()) / n
    k = len(g)
    if k < 2:
        return mean, None, k
    se = math.sqrt(k / (k - 1) * sum((s - mean * c) ** 2 for s, c in g.values())) / n
    return mean, (mean / se if se > 0 else None), k


def lifetime_stats(tracker: LifetimeTracker, sim: Simulator) -> dict[str, Any]:
    """Seconds alone at the front (until joined, stepped ahead, or ended) and the fill share by
    time since activation."""
    filled_at = {}
    for f in sim.fills:
        filled_at.setdefault(f.oid, f.t_ns)
    alone, fills_by = [], defaultdict(int)
    n = 0
    for oid, r in tracker.rec.items():
        n += 1
        ends = [t for t in (r["joined_ns"], r["ahead_ns"], r["end_ns"], filled_at.get(oid)) if t is not None]
        # still alone and live when the replay ends: censored at the end of the data
        alone.append((min(ends) if ends else sim.now) / 1e9 - r["active_ns"] / 1e9)
        if oid in filled_at:
            dt = (filled_at[oid] - r["active_ns"]) / 1e9
            for edge in (1, 5, 10, 30, 60, 300, 1e9):
                if dt <= edge:
                    fills_by[edge] += 1
                    break
    alone.sort()

    def pct(xs: list[float], p: float) -> float | None:
        xs = sorted(xs)
        return round(xs[int(p * (len(xs) - 1))], 2) if xs else None

    join = [(r["joined_ns"] - r["active_ns"]) / 1e9 for r in tracker.rec.values() if r["joined_ns"]]
    ahead = [(r["ahead_ns"] - r["active_ns"]) / 1e9 for r in tracker.rec.values() if r["ahead_ns"]]
    # fill hazard: of orders still live (unfilled, not ended) at t, the share that fill by 2t
    life = {}
    for oid, r in tracker.rec.items():
        end = filled_at.get(oid) or r["end_ns"] or sim.now
        life[oid] = ((end - r["active_ns"]) / 1e9, oid in filled_at)
    hazard = {}
    for a, b in ((0, 1), (1, 5), (5, 10), (10, 30), (30, 60), (60, 300)):
        at_risk = [(d, f) for d, f in life.values() if d > a]
        filled = sum(1 for d, f in at_risk if f and d <= b)
        hazard[f"{a}-{b}s"] = {"at_risk": len(at_risk), "filled": filled,
                               "rate": round(filled / len(at_risk), 4) if at_risk else None}
    # was the order alone at the front when it filled?
    alone_fill = sum(1 for oid, r in tracker.rec.items() if oid in filled_at
                     and all(t is None or t > filled_at[oid] for t in (r["joined_ns"], r["ahead_ns"])))
    return {"orders_tracked": n, "alone_s_p25": pct(alone, 0.25), "alone_s_median": pct(alone, 0.5),
            "alone_s_p75": pct(alone, 0.75),
            "frac_joined": round(len(join) / n, 3) if n else None,
            "join_s_p25_med_p75": [pct(join, 0.25), pct(join, 0.5), pct(join, 0.75)],
            "frac_stepped_ahead": round(len(ahead) / n, 3) if n else None,
            "ahead_s_p25_med_p75": [pct(ahead, 0.25), pct(ahead, 0.5), pct(ahead, 0.75)],
            "fills": sum(fills_by.values()), "fills_while_alone": alone_fill,
            "fill_time_buckets_s": {("inf" if k >= 1e9 else k): v for k, v in sorted(fills_by.items())},
            "fill_hazard": hazard}


def summarize(rows: list[dict[str, Any]], split_ns: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, sub in (("dev", [r for r in rows if r["t_ns"] < split_ns]),
                      ("hold", [r for r in rows if r["t_ns"] >= split_ns])):
        entries = [r for r in sub if r["role"] == "entry"]
        d: dict[str, Any] = {"fills": len(sub), "entry_fills": len(entries),
                             "events": len({r["event"] for r in sub})}
        for col in ("mo_10s", "mo_60s", "mx_10s", "mx_60s", "mo_300s", "settle"):
            m, t, _k = clustered(entries, col)
            d[col] = None if m is None else round(m * 100, 2)
            d[f"t_{col}"] = None if t is None else round(t, 2)
        d["settled_entries"] = sum(1 for r in entries if r["settle"] is not None)
        pr = [r for r in entries if r["paired"]]
        d["round_trips"] = len(pr)
        d["rt_frac"] = round(len(pr) / len(entries), 3) if entries else None
        m, t, _ = clustered(pr, "rt_pnl")
        d["rt_pnl_c"] = None if m is None else round(m * 100, 2)
        d["t_rt"] = None if t is None else round(t, 2)
        out[name] = d
    return out
