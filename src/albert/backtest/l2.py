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


def trade_key(m: dict[str, Any]) -> tuple[str, float, float, int]:
    """(maker side, maker-side price, size, ts_ms) of a trade print."""
    side = "yes" if m["taker_side"] == "no" else "no"
    price = float(m["yes_price_dollars"] if side == "yes" else m["no_price_dollars"])
    return side, price, float(m["count_fp"]), m.get("ts_ms") or 0


def delta_matches(dm: dict[str, Any], k: tuple[str, float, float, int]) -> bool:
    return (dm.get("ts_ms") == k[3] and dm["side"] == k[0]
            and abs(float(dm["price_dollars"]) - k[1]) < EPS
            and abs(-float(dm["delta_fp"]) - k[2]) < 1e-6)


def _mark_executions(events: list[Event]) -> tuple[int, int]:
    """Label negative deltas that a trade print explains (same ts_ms, maker side, price, size).

    The trade usually follows its deltas, but about one in ten arrives first, so unmatched
    deltas and unmatched trades both wait (up to 2 s of exchange time) for their partner.
    Returns (trades, trades_matched)."""
    deltas: dict[str, list[Event]] = defaultdict(list)
    trades_waiting: dict[str, list[tuple[str, float, float, int]]] = defaultdict(list)
    trades = matched = 0
    for ev in events:
        if ev.kind == "delta" and float(ev.msg["delta_fp"]) < 0:
            waiting = trades_waiting[ev.market]
            hit = next((i for i, k in enumerate(waiting) if delta_matches(ev.msg, k)), None)
            if hit is not None:
                ev.is_exec = True
                matched += 1
                del waiting[hit]
            else:
                deltas[ev.market].append(ev)
        elif ev.kind == "trade":
            trades += 1
            k = trade_key(ev.msg)
            cands = deltas[ev.market]
            hit = next((i for i, d in enumerate(cands) if delta_matches(d.msg, k)), None)
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


class OnlineClassifier:
    """Streaming version of ``_mark_executions`` for live paper trading: events are held back
    ``holdback_ms`` of receive time so a trade arriving just after its delta can still label it.
    Released events keep their receive times, so the simulator runs in event time exactly as
    in replay (the holdback delays our bookkeeping, not the modeled decisions)."""

    def __init__(self, holdback_ms: float = 500.0) -> None:
        self.hold = int(holdback_ms * 1e6)
        self.buf: list[Event] = []
        self.waiting: dict[str, list[tuple[tuple[str, float, float, int], int]]] = defaultdict(list)
        self.stats: dict[str, int] = defaultdict(int)

    def push(self, ev: Event) -> None:
        if ev.kind == "delta" and float(ev.msg["delta_fp"]) < 0:
            w = self.waiting[ev.market]
            hit = next((i for i, (k, _) in enumerate(w) if delta_matches(ev.msg, k)), None)
            if hit is not None:
                ev.is_exec = True
                self.stats["trades_matched"] += 1
                del w[hit]
        elif ev.kind == "trade":
            self.stats["trades"] += 1
            k = trade_key(ev.msg)
            hit = next((d for d in self.buf if d.kind == "delta" and not d.is_exec
                        and d.market == ev.market and float(d.msg["delta_fp"]) < 0
                        and delta_matches(d.msg, k)), None)
            if hit is not None:
                hit.is_exec = True
                self.stats["trades_matched"] += 1
            else:
                self.waiting[ev.market].append((k, ev.recv_ns))
        self.buf.append(ev)

    def release(self, now_ns: int) -> list[Event]:
        cut = now_ns - self.hold
        i = 0
        while i < len(self.buf) and self.buf[i].recv_ns <= cut:
            i += 1
        out, self.buf = self.buf[:i], self.buf[i:]
        old = now_ns - 2 * 10**9
        for m, w in self.waiting.items():
            if w and w[0][1] < old:
                self.stats["trades_unmatched"] += sum(1 for _, t in w if t < old)
                self.waiting[m] = [x for x in w if x[1] >= old]
        return out


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


# ---- Polymarket intl (data only) as an external signal ---------------------------------------
def load_pairs(root: Path) -> dict[str, dict[str, Any]]:
    """Kalshi ticker -> {token_id, complement, pm_slug} from ``<root>/pairs/*.json`` (written by
    ``albert record --paired-days``); later recordings override earlier ones."""
    out: dict[str, dict[str, Any]] = {}
    for p in sorted((root / "pairs").glob("*.json")):
        for row in json.loads(p.read_text()):
            out[row["kalshi_ticker"]] = row
    return out


def pm_files(root: Path, start: str | None = None, end: str | None = None) -> list[Path]:
    files = []
    for p in sorted((root / "pm").glob("*/*.jsonl.zst")):
        key = f"{p.parent.name}T{p.name[:2]}"
        if (start is None or key >= start) and (end is None or key < end):
            files.append(p)
    return files


def _pm_tops(row: dict[str, Any]) -> Iterator[tuple[str, float | None, float | None]]:
    """(asset_id, best_bid, best_ask) updates carried by one recorded Polymarket frame."""
    try:
        d = json.loads(row["frame"])
    except ValueError:
        return
    for it in d if isinstance(d, list) else [d]:
        if not isinstance(it, dict):
            continue
        if it.get("event_type", "book") == "book" and "bids" in it:
            bids = [float(x["price"]) for x in it.get("bids") or []]
            asks = [float(x["price"]) for x in it.get("asks") or []]
            yield it["asset_id"], max(bids, default=None), min(asks, default=None)
        elif it.get("event_type") == "price_change":
            for ch in it.get("price_changes") or []:
                bb, ba = ch.get("best_bid"), ch.get("best_ask")
                yield (ch["asset_id"], float(bb) if bb not in (None, "") else None,
                       float(ba) if ba not in (None, "") else None)


def pm_events(files: list[Path], pairs: dict[str, dict[str, Any]]) -> Iterator[Event]:
    """Top-of-book changes of paired Polymarket tokens as ``ext`` events keyed by the Kalshi
    ticker, in the Kalshi market's YES terms (complement applied), at our receive time."""
    by_token: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for t, row in pairs.items():
        by_token[row["token_id"]].append((t, bool(row.get("complement"))))
    last: dict[str, tuple] = {}
    by_hour: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        by_hour[f"{p.parent.name}T{p.name[:2]}"].append(p)
    for hour in sorted(by_hour):
        evs: list[Event] = []
        for p in by_hour[hour]:
            for row in read_stream(p):
                for asset, bid, ask in _pm_tops(row):
                    if asset not in by_token or last.get(asset) == (bid, ask):
                        continue
                    last[asset] = (bid, ask)
                    for ticker, comp in by_token[asset]:
                        b, a = (None if ask is None else 1 - ask, None if bid is None else 1 - bid) if comp else (bid, ask)
                        mid = (a + b) / 2 if a is not None and b is not None else None
                        evs.append(Event(row["recv_ns"], "ext", ticker, {"bid": b, "ask": a, "mid": mid}))
        evs.sort(key=lambda e: e.recv_ns)
        yield from evs


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
    # Write budget (token bucket): Kalshi Basic tier allows 10 order writes/s; a cancel costs
    # ``cancel_cost`` of a write (50 cancels/s). None disables the limit.
    writes_per_s: float | None = 10.0
    cancel_cost: float = 0.2


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
        self.ext: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
        self._tokens = cfg.writes_per_s or 0.0
        self._tokens_ns = 0

    # -- strategy API --
    def _spend(self, cost: float) -> bool:
        rate = self.cfg.writes_per_s
        if rate is None:
            return True
        self._tokens = min(rate, self._tokens + (self.now - self._tokens_ns) / 1e9 * rate)
        self._tokens_ns = self.now
        if self._tokens < cost - EPS:
            return False
        self._tokens -= cost
        return True

    def place(self, market: str, side: str, price: float, qty: float = 1.0) -> int | None:
        """Returns the order id, or None when the write budget is exhausted (nothing sent)."""
        if not self._spend(1.0):
            self.diag["throttled_places"] += 1
            return None
        oid = self._next
        self._next += 1
        lat = int(self.cfg.latency_ms * 1e6)
        o = Order(oid, market, side, round(price, 6), qty, self.now, self.now + lat, rem=qty)
        self.orders[oid] = o
        self.by_market[market].append(oid)
        heapq.heappush(self._sched, (o.active_ns, oid, "activate"))
        return oid

    def cancel(self, oid: int) -> bool:
        """Returns False when the write budget is exhausted (the order stays live)."""
        o = self.orders[oid]
        if o.status in ("pending", "active"):
            if not self._spend(self.cfg.cancel_cost):
                self.diag["throttled_cancels"] += 1
                return False
            o.status = "cancelling" if o.status == "active" else "cancel_pending"
            o.cancel_ns = self.now + int(self.cfg.latency_ms * 1e6)
            heapq.heappush(self._sched, (o.cancel_ns, oid, "cancel"))
        return True

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
        if ev.kind == "ext":            # external fair-value signal (Polymarket intl mid)
            if ev.msg.get("mid") is not None:
                times, mids = self.ext[ev.market]
                times.append(ev.recv_ns)
                mids.append(ev.msg["mid"])
            return
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

    def ext_at(self, market: str, t_ns: int) -> float | None:
        times, mids = self.ext.get(market, ([], []))
        i = bisect.bisect_right(times, t_ns) - 1
        return mids[i] if i >= 0 else None

    def ext_range(self, market: str, t0: int, t1: int) -> float:
        """High minus low of the external mid over [t0, t1] (including the value live at t0)."""
        times, mids = self.ext.get(market, ([], []))
        i = max(bisect.bisect_right(times, t0) - 1, 0)
        j = bisect.bisect_right(times, t1)
        window = mids[i:j]
        return max(window) - min(window) if window else 0.0

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


class Quoter:
    """Two-sided 1-lot quoting with the levers the recorded maker markouts point at.

    * ``improve``: quote one tick inside the best (alone at the front of a new level) instead
      of joining the back of the queue. Recorded executions that clear the whole level (the only
      ones a back-of-queue order receives) lose 2-4c at spreads of 4c or less.
    * ``min_spread`` / ``max_spread``: quote only in this spread band (measured before us).
    * ``pull_frac``: when joining, cancel if the external size ahead of us shrinks by this
      fraction (the queue is being eaten or pulled: the price is about to move).
    * ``cooldown_s``: after any fill in a market, pull both sides and wait.
    * ``min_requote_s``: at most one new order per market side in this interval (rate limits).
    * ``categories``: series categories to trade (None = all).
    * Paired markets (Polymarket intl mid replayed as ``ext`` events): ``pm_band`` keeps each bid
      at least this far below the Polymarket fair value (``None`` = ignore it); ``pm_pull`` pulls
      both sides for ``pm_window_s`` when Polymarket moved this much in the last ``pm_window_s``;
      ``pm_only`` trades only markets with a Polymarket price.
    """

    def __init__(self, min_spread: float = 0.03, max_spread: float = 1.0, lo: float = 0.05,
                 hi: float = 0.95, max_pos: float = 3.0, size: float = 1.0, improve: bool = False,
                 pull_frac: float = 0.0, cooldown_s: float = 0.0, min_requote_s: float = 1.0,
                 categories: set[str] | None = None, category_of: dict[str, str] | None = None,
                 markets: set[str] | None = None, pm_band: float | None = None,
                 pm_pull: float = 0.0, pm_window_s: float = 10.0, pm_only: bool = False) -> None:
        self.min_spread, self.max_spread, self.lo, self.hi = min_spread, max_spread, lo, hi
        self.max_pos, self.size, self.improve = max_pos, size, improve
        self.pull_frac, self.cooldown_ns = pull_frac, int(cooldown_s * 1e9)
        self.requote_ns = int(min_requote_s * 1e9)
        self.categories, self.category_of, self.markets = categories, category_of or {}, markets
        self._last_place: dict[tuple[str, str], int] = {}
        self._ahead0: dict[int, float] = {}
        self._cool_until: dict[str, int] = {}
        self._seen_fills = 0
        self.pm_band, self.pm_pull, self.pm_only = pm_band, pm_pull, pm_only
        self.pm_window_ns = int(pm_window_s * 1e9)

    def _eligible(self, market: str) -> bool:
        if self.markets is not None and market not in self.markets:
            return False
        if self.categories is not None:
            return self.category_of.get(market.split("-", 1)[0]) in self.categories
        return True

    def _wanted(self, sim: Simulator, market: str) -> dict[str, float | None]:
        want: dict[str, float | None] = {"yes": None, "no": None}
        if sim.now < self._cool_until.get(market, 0):
            return want
        q = sim.markets[market].quote()
        if q["yes_bid"] is None or q["yes_ask"] is None:
            return want
        spread = q["yes_ask"] - q["yes_bid"]
        mid = (q["yes_ask"] + q["yes_bid"]) / 2
        if not (self.min_spread - EPS <= spread <= self.max_spread + EPS and self.lo <= mid <= self.hi):
            return want
        fair = sim.ext_at(market, sim.now)
        if fair is None and self.pm_only:
            return want
        if fair is not None and self.pm_pull:
            if sim.ext_range(market, sim.now - self.pm_window_ns, sim.now) >= self.pm_pull - EPS:
                self._cool_until[market] = max(self._cool_until.get(market, 0),
                                               sim.now + self.pm_window_ns)
                return want
        tick = 0.01 if self.improve and spread >= 0.03 - EPS else 0.0
        pos = sim.position[market]
        yes, no = round(q["yes_bid"] + tick, 6), round(1 - q["yes_ask"] + tick, 6)
        if fair is not None and self.pm_band is not None:
            yes = yes if yes <= fair - self.pm_band + EPS else None
            no = no if no <= 1 - fair - self.pm_band + EPS else None
        if pos < self.max_pos:
            want["yes"] = yes
        if pos > -self.max_pos:
            want["no"] = no
        return want

    def on_event(self, sim: Simulator, market: str) -> None:
        for f in sim.fills[self._seen_fills:]:
            if self.cooldown_ns:
                self._cool_until[f.market] = f.t_ns + self.cooldown_ns
        self._seen_fills = len(sim.fills)
        if market not in sim.synced or not self._eligible(market):
            return
        want = self._wanted(sim, market)
        m = sim.markets[market]
        for side in ("yes", "no"):
            mine = [o for o in sim.live_orders(market) if o.side == side]
            keep = []
            for o in mine:
                ok = want[side] is not None and (
                    abs(o.price - want[side]) < EPS
                    # improved order that others have since joined: we are still at the front
                    or (self.improve and abs(o.price + 0.01 - want[side]) < EPS
                        and o.price >= (m.best(side) or (0.0,))[0] - EPS))
                if ok and self.pull_frac and o.status == "active" and o.oid in self._ahead0:
                    lev = m.levels[side].get(o.price)
                    ext = lev.ext_total() if lev else 0.0
                    if ext < (1 - self.pull_frac) * self._ahead0[o.oid] - EPS:
                        ok = False
                        self._cool_until[market] = sim.now + max(self.cooldown_ns, self.requote_ns)
                if ok:
                    keep.append(o)
                elif not sim.cancel(o.oid):
                    keep.append(o)   # throttled: still live
            if want[side] is None or keep:
                continue
            if sim.now - self._last_place.get((market, side), -(2**62)) < self.requote_ns:
                continue
            if sim.now < self._cool_until.get(market, 0):
                continue
            oid = sim.place(market, side, want[side], self.size)
            if oid is not None:
                self._last_place[(market, side)] = sim.now
                lev = m.levels[side].get(want[side])
                self._ahead0[oid] = lev.ext_total() if lev else 0.0


STRATEGIES = {"join": JoinBest, "quote": Quoter}


def run(files: list[Path], strategy, cfg: SimConfig,
        ext: Iterator[Event] | None = None) -> tuple[Simulator, dict[str, int]]:
    sim = Simulator(cfg)
    stats: dict[str, int] = defaultdict(int)
    events = iter_events(files, stats)
    if ext is not None:
        events = heapq.merge(events, ext, key=lambda e: e.recv_ns)
    for ev in events:
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
                queue: str, params: dict[str, Any], writes_per_s: float | None = 10.0,
                root: Path | None = None) -> Path:
    import polars as pl

    from albert.ingest import kalshi_series
    from albert.reports import Report

    root = root or s.raw_dir / "stream"
    files = stream_files(root, start, end)
    if not files:
        raise RuntimeError("no recorded Kalshi stream files in range; run `albert record` first")
    series = kalshi_series(s, c)
    cfg = SimConfig(latency_ms=latency_ms, queue=queue, fee_info=series, writes_per_s=writes_per_s)
    if strategy == "quote":
        params = params | {"category_of": {k: v.get("category") for k, v in series.items()}}
    strat = STRATEGIES[strategy](**params)
    pairs = load_pairs(root)
    ext = pm_events(pm_files(root, start, end), pairs) if pairs and strategy == "quote" else None
    sim, stats = run(files, strat, cfg, ext)
    rows = markouts(sim)
    rep = Report(s.reports_dir, f"backtest-l2-{strategy}")
    rep.json("Run", {"files": len(files), "first": str(files[0]), "last": str(files[-1]),
                     "strategy": strategy, "latency_ms": latency_ms, "writes_per_s": writes_per_s,
                     "params": {k: sorted(v) if isinstance(v, set) else v
                                for k, v in params.items() if k != "category_of"},
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
    fills = fills.with_columns(pl.col("market").str.replace(r"-[^-]+$", "").alias("event"))
    summary = []
    for col in [f"mo_{h}s" for h in HORIZONS_S] + ["pnl_per_contract"]:
        mean, t, k = _cluster_t(fills, col)
        _, te, ke = _cluster_t(fills, col, "event")
        summary.append({"metric": col, "mean_per_contract": mean, "t_by_market": t, "markets": k,
                        "t_by_event": te, "events": ke})
    rep.table("Markouts and PnL per contract (clustered by market and by event)", pl.DataFrame(summary))
    rep.json("Totals", {
        "fills": fills.height, "contracts": fills["qty"].sum(),
        "notional": (fills["qty"] * fills["price"]).sum(), "fees": fills["fee"].sum(),
        "pnl": (fills["qty"] * fills["pnl_per_contract"]).sum(),
        "settled_contracts": fills.filter(pl.col("settled"))["qty"].sum(),
        "by_how": dict(fills.group_by("how").agg(pl.col("qty").sum()).iter_rows()),
    })
    rep.table("Fills", fills.drop("t_ns"), save_as="fills", max_rows=40)
    return rep.save()
