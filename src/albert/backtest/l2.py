"""Order-book replay simulator for Kalshi maker strategies (recorded WebSocket streams).

Input: ``data/raw/stream/kalshi/<date>/<HH>-<conn>.jsonl.zst`` written by ``albert record``.

Facts from the recorded feed that the model relies on (verified on the 2026-11-05 pilot):

* The book has bids only: YES bids and NO bids. A taker buying YES at p fills NO bids at 1 - p.
  A taker buying NO at p (YES price) fills YES bids at p.
* Every resting order that trades produces its own ``orderbook_delta`` (negative, the filled
  size), followed by a ``trade`` print with the same ``ts_ms`` and size. Deltas that a trade
  explains are executions; the other negative deltas are cancels or decreases.
* Positive deltas are new orders joining the back of a price level.

Queue model: each price level is a FIFO of chunks (order-sized after the snapshot, one
aggregated chunk per level from the snapshot itself). Our simulated order joins the back of its
level after ``latency_ms``. An execution of size x at our level fills us by
``min(remaining, x - ext_ahead)`` and then removes x of external size from the front (the
recorded book is never contradicted). Cancels remove an exact-size external chunk, from behind
us (``risk_averse``) or ahead of us (``optimistic``). Executions at a worse price than ours on
the same side, or a new opposite-side order that would cross ours, fill us (we would have been
first). Our orders never appear in the book that strategies see.

All times are our receive times (``recv_ns``). No look-ahead: the strategy is called after each
event with only the state built from events up to that moment; the execution classifier looks
ahead a few frames in the same connection only to label the *recorded* delta, never to inform
the strategy.
"""

import bisect
import heapq
import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from albert.fees import kalshi_fee
from albert.stream import read_stream

EPS = 1e-9


# ---- event loading --------------------------------------------------------------------------
@dataclass
class Event:
    recv_ns: int
    kind: str                # snapshot | delta | trade
    market: str
    msg: dict[str, Any]
    is_exec: bool = False


def _mark_executions(events: list[Event]) -> tuple[int, int]:
    """Label negative deltas that a trade print explains (same ts_ms, maker side, price, size).

    The trade usually follows its deltas, but about one in ten arrives first, so unmatched
    deltas and unmatched trades both wait (up to 2 s of exchange time) for their partner.
    Returns (trades, trades_matched)."""
    deltas: dict[str, list[Event]] = defaultdict(list)
    trades_waiting: dict[str, list[tuple[str, float, float, int]]] = defaultdict(list)
    trades = matched = 0

    def key_of_trade(m: dict[str, Any]) -> tuple[str, float, float, int]:
        side = "yes" if m["taker_side"] == "no" else "no"
        price = float(m["yes_price_dollars"] if side == "yes" else m["no_price_dollars"])
        return side, price, float(m["count_fp"]), m.get("ts_ms") or 0

    def same(d: Event, k: tuple[str, float, float, int]) -> bool:
        dm = d.msg
        return (dm.get("ts_ms") == k[3] and dm["side"] == k[0]
                and abs(float(dm["price_dollars"]) - k[1]) < EPS
                and abs(-float(dm["delta_fp"]) - k[2]) < 1e-6)

    for ev in events:
        if ev.kind == "delta" and float(ev.msg["delta_fp"]) < 0:
            waiting = trades_waiting[ev.market]
            hit = next((i for i, k in enumerate(waiting) if same(ev, k)), None)
            if hit is not None:
                ev.is_exec = True
                matched += 1
                del waiting[hit]
            else:
                deltas[ev.market].append(ev)
        elif ev.kind == "trade":
            trades += 1
            k = key_of_trade(ev.msg)
            cands = deltas[ev.market]
            hit = next((i for i, d in enumerate(cands) if same(d, k)), None)
            if hit is not None:
                cands[hit].is_exec = True
                matched += 1
                del cands[hit]
            else:
                trades_waiting[ev.market].append(k)
            ts = k[3]
            deltas[ev.market] = [d for d in cands if (d.msg.get("ts_ms") or ts) >= ts - 2000]
            trades_waiting[ev.market] = [w for w in trades_waiting[ev.market] if w[3] >= ts - 2000]
    return trades, matched


def _file_events(path: Path) -> list[Event]:
    return events_from_rows(read_stream(path))


def events_from_rows(rows: list[dict[str, Any]]) -> list[Event]:
    out: list[Event] = []
    for row in rows:
        d = json.loads(row["frame"])
        typ = d.get("type")
        kind = {"orderbook_snapshot": "snapshot", "orderbook_delta": "delta", "trade": "trade"}.get(typ)
        if kind:
            out.append(Event(row["recv_ns"], kind, d["msg"]["market_ticker"], d["msg"]))
    return out


def stream_files(root: Path, start: str | None = None, end: str | None = None) -> list[Path]:
    """Closed Kalshi stream files (excluding REST checks) with ``start <= date-hour < end``
    where both bounds are ``YYYY-MM-DDTHH`` strings."""
    files = []
    for p in sorted((root / "kalshi").glob("*/*.jsonl.zst")):
        if "-rest" in p.name:
            continue
        key = f"{p.parent.name}T{p.name[:2]}"
        if (start is None or key >= start) and (end is None or key < end):
            files.append(p)
    return files


def iter_events(files: list[Path], stats: dict[str, int]) -> Iterator[Event]:
    """Merge files hour by hour in receive order. Each market lives on one connection, so its
    own order is the connection's order; the merge only interleaves different markets."""
    by_hour: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        by_hour[f"{p.parent.name}T{p.name[:2]}"].append(p)
    for hour in sorted(by_hour):
        streams = []
        for p in by_hour[hour]:
            evs = _file_events(p)
            t, m = _mark_executions(evs)
            stats["trades"] += t
            stats["trades_matched"] += m
            streams.append(evs)
        yield from heapq.merge(*streams, key=lambda e: e.recv_ns)


# ---- book with modeled queues -----------------------------------------------------------------
@dataclass
class Order:
    oid: int
    market: str
    side: str            # yes | no  (a bid on that side)
    price: float
    qty: float
    placed_ns: int
    active_ns: int
    rem: float = 0.0
    status: str = "pending"   # pending | active | cancelling | cancelled | filled | rejected
    cancel_ns: int | None = None


@dataclass
class Fill:
    t_ns: int
    oid: int
    market: str
    side: str
    price: float
    qty: float
    fee: float
    how: str             # queue | through | cross


class Level:
    """FIFO of [size, oid]; oid None marks external (recorded) liquidity."""
    __slots__ = ("q",)

    def __init__(self) -> None:
        self.q: list[list] = []

    def ext_total(self) -> float:
        return sum(c[0] for c in self.q if c[1] is None)


class Market:
    def __init__(self) -> None:
        self.levels: dict[str, dict[float, Level]] = {"yes": {}, "no": {}}

    def best(self, side: str) -> tuple[float, float] | None:
        lv = [(p, lev.ext_total()) for p, lev in self.levels[side].items()]
        lv = [(p, s) for p, s in lv if s > EPS]
        return max(lv) if lv else None

    def quote(self) -> dict[str, float | None]:
        by, bn = self.best("yes"), self.best("no")
        return {"yes_bid": by[0] if by else None, "yes_bid_size": by[1] if by else None,
                "yes_ask": round(1 - bn[0], 6) if bn else None, "yes_ask_size": bn[1] if bn else None}


@dataclass
class SimConfig:
    latency_ms: float = 100.0
    queue: str = "risk_averse"          # risk_averse | optimistic
    fee_info: dict[str, dict] = field(default_factory=dict)   # series -> {fee_type, fee_multiplier}


class Simulator:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.markets: dict[str, Market] = defaultdict(Market)
        self.orders: dict[int, Order] = {}
        self.by_market: dict[str, list[int]] = defaultdict(list)
        self.fills: list[Fill] = []
        self.position: dict[str, float] = defaultdict(float)   # net YES contracts
        self.cash = 0.0
        self.now = 0
        self._next = 1
        self._sched: list[tuple[int, int, str]] = []   # (t_ns, oid, action)
        self.quotes: dict[str, tuple[list[int], list[tuple]]] = defaultdict(lambda: ([], []))
        self.diag: dict[str, int] = defaultdict(int)
        self.synced: set[str] = set()   # markets whose snapshot we have seen

    # -- strategy API --
    def place(self, market: str, side: str, price: float, qty: float = 1.0) -> int:
        oid = self._next
        self._next += 1
        lat = int(self.cfg.latency_ms * 1e6)
        o = Order(oid, market, side, round(price, 6), qty, self.now, self.now + lat, rem=qty)
        self.orders[oid] = o
        self.by_market[market].append(oid)
        heapq.heappush(self._sched, (o.active_ns, oid, "activate"))
        return oid

    def cancel(self, oid: int) -> None:
        o = self.orders[oid]
        if o.status in ("pending", "active"):
            o.status = "cancelling" if o.status == "active" else "cancel_pending"
            o.cancel_ns = self.now + int(self.cfg.latency_ms * 1e6)
            heapq.heappush(self._sched, (o.cancel_ns, oid, "cancel"))

    def live_orders(self, market: str) -> list[Order]:
        return [self.orders[i] for i in self.by_market[market]
                if self.orders[i].status in ("pending", "active")]

    # -- internals --
    def _fee(self, market: str, qty: float, price: float) -> float:
        info = self.cfg.fee_info.get(market.split("-", 1)[0], {})
        return kalshi_fee(qty, price, maker=True, fee_type=info.get("fee_type") or "quadratic",
                          multiplier=info.get("fee_multiplier") or 1.0)

    def _fill(self, o: Order, qty: float, how: str) -> None:
        qty = min(qty, o.rem)
        if qty <= EPS:
            return
        o.rem -= qty
        fee = self._fee(o.market, qty, o.price)
        self.fills.append(Fill(self.now, o.oid, o.market, o.side, o.price, qty, fee, how))
        self.cash -= qty * o.price + fee
        self.position[o.market] += qty if o.side == "yes" else -qty
        lev = self.markets[o.market].levels[o.side].get(o.price)
        if lev:
            for c in lev.q:
                if c[1] == o.oid:
                    c[0] -= qty
            lev.q = [c for c in lev.q if c[0] > EPS]
        if o.rem <= EPS:
            o.status = "filled"

    def _run_schedule(self, until: int) -> None:
        while self._sched and self._sched[0][0] <= until:
            t, oid, action = heapq.heappop(self._sched)
            self.now = max(self.now, t)
            o = self.orders[oid]
            m = self.markets[o.market]
            if action == "activate":
                if o.status == "cancel_pending":
                    o.status = "cancelled"
                    continue
                if o.status != "pending":
                    continue
                opp = m.best("no" if o.side == "yes" else "yes")
                if opp and o.price + opp[0] >= 1 - EPS:   # post-only: would cross
                    o.status = "rejected"
                    self.diag["post_only_rejects"] += 1
                    continue
                o.status = "active"
                m.levels[o.side].setdefault(o.price, Level()).q.append([o.rem, oid])
            elif action == "cancel" and o.status == "cancelling":
                lev = m.levels[o.side].get(o.price)
                if lev:
                    lev.q = [c for c in lev.q if c[1] != oid]
                o.status = "cancelled"

    def _ours_at(self, market: str, side: str) -> list[Order]:
        return [self.orders[i] for i in self.by_market[market]
                if self.orders[i].status in ("active", "cancelling") and self.orders[i].side == side]

    def apply(self, ev: Event) -> None:
        self._run_schedule(ev.recv_ns)
        self.now = ev.recv_ns
        if ev.kind == "snapshot":
            self.synced.add(ev.market)
        elif ev.market not in self.synced:
            self.diag["pre_snapshot_events"] += 1   # replay started mid-stream: wait for a snapshot
            return
        m = self.markets[ev.market]
        msg = ev.msg
        if ev.kind == "snapshot":
            mine = {o.oid: self._ahead(m, o)
                    for o in self._ours_at(ev.market, "yes") + self._ours_at(ev.market, "no")}
            for side in ("yes", "no"):
                m.levels[side] = {}
                for p, q in msg.get(f"{side}_dollars_fp") or []:
                    m.levels[side].setdefault(float(p), Level()).q.append([float(q), None])
            for oid, ahead in mine.items():
                o = self.orders[oid]
                lev = m.levels[o.side].setdefault(o.price, Level())
                self._insert_at(lev, [o.rem, oid], ahead)
            self.diag["snapshots"] += 1
        elif ev.kind == "delta":
            side, price, d = msg["side"], float(msg["price_dollars"]), float(msg["delta_fp"])
            lev = m.levels[side].setdefault(price, Level())
            if d > 0:
                lev.q.append([d, None])
                self._cross_check(ev.market, side, price, d)
            elif ev.is_exec:
                self._execution(ev.market, side, price, -d, lev)
            else:
                self._cancel_ext(lev, -d)
            if not lev.q:
                m.levels[side].pop(price, None)
        q = m.quote()
        times, vals = self.quotes[ev.market]
        if not vals or vals[-1] != (q["yes_bid"], q["yes_ask"]):
            times.append(ev.recv_ns)
            vals.append((q["yes_bid"], q["yes_ask"]))

    def _ahead(self, m: Market, o: Order) -> float:
        lev = m.levels[o.side].get(o.price)
        ahead = 0.0
        for c in lev.q if lev else []:
            if c[1] == o.oid:
                break
            if c[1] is None:
                ahead += c[0]
        return ahead

    @staticmethod
    def _insert_at(lev: Level, chunk: list, ahead: float) -> None:
        """Insert ``chunk`` after ``ahead`` units of external size (splitting a chunk if needed)."""
        acc, out, placed = 0.0, [], False
        for c in lev.q:
            if not placed and c[1] is None and acc + c[0] > ahead + EPS:
                take = ahead - acc
                if take > EPS:
                    out.append([take, None])
                out.append(chunk)
                out.append([c[0] - take, None])
                placed = True
                acc += c[0]
                continue
            out.append(c)
            if c[1] is None:
                acc += c[0]
        if not placed:
            out.append(chunk)
        lev.q = out

    def _execution(self, market: str, side: str, price: float, x: float, lev: Level) -> None:
        self.diag["executions"] += 1
        ext = [c for c in lev.q if c[1] is None]
        if ext and abs(ext[0][0] - x) < 1e-6:
            self.diag["exec_front_match"] += 1
        # Our orders at this level: filled by the part of x that reaches them.
        ahead = 0.0
        for c in list(lev.q):
            if c[1] is None:
                ahead += c[0]
            else:
                o = self.orders[c[1]]
                if o.status in ("active", "cancelling"):
                    self._fill(o, x - ahead, "queue")
        # Trade-through: the taker sold through a level worse than our better-priced bid.
        for o in self._ours_at(market, side):
            if o.price > price + EPS:
                self._fill(o, x, "through")
        # Remove x of external size from the front, as the recorded book did.
        need = x
        for c in lev.q:
            if c[1] is None and need > EPS:
                take = min(c[0], need)
                c[0] -= take
                need -= take
        lev.q = [c for c in lev.q if c[0] > EPS]
        if need > 1e-6:
            self.diag["exec_exceeds_model"] += 1

    def _cancel_ext(self, lev: Level, x: float) -> None:
        idx = [i for i, c in enumerate(lev.q) if c[1] is None]
        exact = [i for i in idx if abs(lev.q[i][0] - x) < 1e-6]
        order = list(reversed(idx)) if self.cfg.queue == "risk_averse" else idx
        if exact:
            i = exact[-1] if self.cfg.queue == "risk_averse" else exact[0]
            lev.q[i][0] = 0.0
        else:
            need = x
            for i in order:
                take = min(lev.q[i][0], need)
                lev.q[i][0] -= take
                need -= take
                if need <= EPS:
                    break
        lev.q = [c for c in lev.q if c[0] > EPS]

    def _cross_check(self, market: str, side: str, price: float, d: float) -> None:
        """A new external bid that crosses our opposite-side bid would have traded with us."""
        other = "no" if side == "yes" else "yes"
        for o in self._ours_at(market, other):
            if o.price + price >= 1 - EPS:
                self._fill(o, d, "cross")

    def mid_at(self, market: str, t_ns: int) -> float | None:
        times, vals = self.quotes.get(market, ([], []))
        i = bisect.bisect_right(times, t_ns) - 1
        if i < 0:
            return None
        b, a = vals[i]
        return (b + a) / 2 if b is not None and a is not None else None


# ---- strategies -----------------------------------------------------------------------------
class JoinBest:
    """Baseline: rest 1 contract at the best YES bid and the best NO bid (join, never improve)
    while the spread is at least ``min_spread`` and the mid is inside [lo, hi]. Re-peg when the
    best price moves. Stop adding to a side when |net position| reaches ``max_pos``."""

    def __init__(self, min_spread: float = 0.03, lo: float = 0.05, hi: float = 0.95,
                 max_pos: float = 3.0, size: float = 1.0, markets: set[str] | None = None) -> None:
        self.min_spread, self.lo, self.hi = min_spread, lo, hi
        self.max_pos, self.size, self.markets = max_pos, size, markets

    def on_event(self, sim: Simulator, market: str) -> None:
        if (self.markets is not None and market not in self.markets) or market not in sim.synced:
            return
        q = sim.markets[market].quote()
        want: dict[str, float | None] = {"yes": None, "no": None}
        if q["yes_bid"] is not None and q["yes_ask"] is not None:
            spread = q["yes_ask"] - q["yes_bid"]
            mid = (q["yes_ask"] + q["yes_bid"]) / 2
            if spread >= self.min_spread - EPS and self.lo <= mid <= self.hi:
                pos = sim.position[market]
                if pos < self.max_pos:
                    want["yes"] = q["yes_bid"]
                if pos > -self.max_pos:
                    want["no"] = round(1 - q["yes_ask"], 6)
        live = sim.live_orders(market)
        for side in ("yes", "no"):
            mine = [o for o in live if o.side == side]
            keep = [o for o in mine if want[side] is not None and abs(o.price - want[side]) < EPS]
            for o in mine:
                if o not in keep:
                    sim.cancel(o.oid)
            if want[side] is not None and not keep:
                sim.place(market, side, want[side], self.size)


STRATEGIES = {"join": JoinBest}


def run(files: list[Path], strategy, cfg: SimConfig) -> tuple[Simulator, dict[str, int]]:
    sim = Simulator(cfg)
    stats: dict[str, int] = defaultdict(int)
    for ev in iter_events(files, stats):
        sim.apply(ev)
        stats["events"] += 1
        strategy.on_event(sim, ev.market)
    return sim, stats


# ---- evaluation ------------------------------------------------------------------------------
HORIZONS_S = (1, 10, 60, 300)


def markouts(sim: Simulator) -> list[dict[str, Any]]:
    """Per fill: value change of the bought contract at each horizon, per contract, vs the mid.
    ``None`` where the horizon runs past the recorded data."""
    end = max((t[-1] for t, _ in sim.quotes.values() if t), default=0)
    rows = []
    for f in sim.fills:
        row = {"t_ns": f.t_ns, "market": f.market, "side": f.side, "price": f.price, "qty": f.qty,
               "fee": f.fee, "how": f.how}
        for h in HORIZONS_S:
            t = f.t_ns + h * 10**9
            mid = sim.mid_at(f.market, t) if t <= end else None
            if mid is None:
                row[f"mo_{h}s"] = None
            else:
                value = mid if f.side == "yes" else 1 - mid
                row[f"mo_{h}s"] = value - f.price
        rows.append(row)
    return rows


def _cluster_t(frame, col: str, cluster: str = "market") -> tuple[float | None, float | None, int]:
    """Mean of ``col`` and a t-stat with standard errors clustered by ``cluster``."""
    import math

    import polars as pl

    f = frame.filter(pl.col(col).is_not_null())
    if f.is_empty():
        return None, None, 0
    g = f.group_by(cluster).agg((pl.col(col) * pl.col("qty")).sum().alias("s"), pl.col("qty").sum().alias("n"))
    n_total = g["n"].sum()
    mean = g["s"].sum() / n_total
    k = g.height
    if k < 2:
        return mean, None, k
    resid = ((g["s"] - mean * g["n"]) ** 2).sum()
    se = math.sqrt(k / (k - 1) * resid) / n_total
    return mean, (mean / se if se > 0 else None), k


def backtest_l2(s, c, *, start: str | None, end: str | None, strategy: str, latency_ms: float,
                queue: str, params: dict[str, Any]) -> Path:
    import polars as pl

    from albert.ingest import kalshi_series
    from albert.reports import Report

    files = stream_files(s.raw_dir / "stream", start, end)
    if not files:
        raise RuntimeError("no recorded Kalshi stream files in range; run `albert record` first")
    cfg = SimConfig(latency_ms=latency_ms, queue=queue, fee_info=kalshi_series(s, c))
    strat = STRATEGIES[strategy](**params)
    sim, stats = run(files, strat, cfg)
    rows = markouts(sim)
    rep = Report(s.reports_dir, f"backtest-l2-{strategy}")
    rep.json("Run", {"files": len(files), "first": str(files[0]), "last": str(files[-1]),
                     "strategy": strategy, "params": params, "latency_ms": latency_ms,
                     "queue": queue, "events": stats["events"],
                     "trades": stats["trades"], "trades_matched_to_deltas": stats["trades_matched"],
                     "orders": len(sim.orders), "diag": dict(sim.diag)})
    if not rows:
        rep.text("Fills", "No fills.")
        return rep.save()
    fills = pl.DataFrame(rows, infer_schema_length=None)
    # Settle with the market result where known, else mark at the last recorded mid.
    results: dict[str, float | None] = {}
    for t in fills["market"].unique().to_list():
        m = c.kalshi.client.get(f"/markets/{t}", cache=False).payload.get("market", {})
        res = m.get("result")
        results[t] = 1.0 if res == "yes" else 0.0 if res == "no" else None
    last_mid = {t: sim.mid_at(t, 2**62) for t in results}
    fills = fills.with_columns(
        pl.col("market").replace_strict(results, default=None, return_dtype=pl.Float64).alias("settle_yes"),
        pl.col("market").replace_strict(last_mid, default=None, return_dtype=pl.Float64).alias("last_mid"),
    ).with_columns(
        pl.coalesce("settle_yes", "last_mid").alias("final_yes"),
        pl.col("settle_yes").is_not_null().alias("settled"),
    ).with_columns(
        (pl.when(pl.col("side") == "yes").then(pl.col("final_yes")).otherwise(1 - pl.col("final_yes"))
         - pl.col("price") - pl.col("fee") / pl.col("qty")).alias("pnl_per_contract"),
    )
    summary = []
    for col in [f"mo_{h}s" for h in HORIZONS_S] + ["pnl_per_contract"]:
        mean, t, k = _cluster_t(fills, col)
        summary.append({"metric": col, "mean_per_contract": mean, "t_clustered": t, "markets": k})
    rep.table("Markouts and PnL per contract (clustered by market)", pl.DataFrame(summary))
    rep.json("Totals", {
        "fills": fills.height, "contracts": fills["qty"].sum(),
        "notional": (fills["qty"] * fills["price"]).sum(), "fees": fills["fee"].sum(),
        "pnl": (fills["qty"] * fills["pnl_per_contract"]).sum(),
        "settled_contracts": fills.filter(pl.col("settled"))["qty"].sum(),
        "by_how": dict(fills.group_by("how").agg(pl.col("qty").sum()).iter_rows()),
    })
    rep.table("Fills", fills.drop("t_ns"), save_as="fills", max_rows=40)
    return rep.save()
