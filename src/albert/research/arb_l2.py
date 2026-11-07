"""Structural arbitrage in the recorded Kalshi L2 stream: how often, how big, how long.

Three kinds of set, built from event metadata (only sets whose every leg was recorded):

* ``me_yes``: an exhaustive, mutually exclusive event (exactly one leg settles YES, e.g. a
  two-team game moneyline). Buying YES on every leg costs ``sum(yes_ask)`` and pays exactly $1.
* ``me_no``: a mutually exclusive event (at most one YES). Buying NO on every leg costs
  ``sum(1 - yes_bid)`` and pays at least ``n - 1``. Needs mutual exclusivity only.
* ``ladder``: two "greater than" strikes of one event, X1 < X2. P(>X1) >= P(>X2), so buying YES
  at X1 and NO at X2 pays at least $1. It's an arbitrage when ``yes_ask(X1) < yes_bid(X2)``.

Every leg is a taker order, so it pays the Kalshi taker fee 0.07 x multiplier x C x P x (1 - P),
rounded up to $0.0001 per leg (this account's balance precision). An *episode* is a stretch of
time in which the set is executable for one lot after fees. Each episode records:

* start, duration, best edge, and the size at the best level on every leg;
* whether a trade printed on a leg during the episode (someone else took it);
* whether the legs sit on one recorder connection (cross-connection lag can fake an arbitrage).

**Latency test.** At ``start + L``, each leg is filled if its current price is still at or below
the observed limit, with size. If only some legs fill, they are unwound at ``start + 2L`` by
selling at the then-current bid, after fees. Orders are sent in parallel; sequential legging only
adds latency.

**Timing caveats.**
* ``recv_ns`` is our receive time. The Mac clock runs about 44 ms behind the exchange.
* The coordinator measured about 50 ms from an exchange event to our order reaching the
  exchange. So survival at L = 50 ms is the realistic case; 100 and 250 ms are stress cases.

Stale books after a market closes would fake arbitrages, so legs are ignored after their
``close_time``. For a settled game that is the actual end.
"""

import heapq
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

LATENCIES_MS = (25, 50, 100, 250, 500)
TAKER = 0.07


def taker_fee(contracts: float, price: float, mult: float = 1.0) -> float:
    """Kalshi taker fee for one leg, rounded up to $0.0001."""
    if contracts <= 0 or not 0 < price < 1:
        return 0.0
    raw = TAKER * mult * contracts * price * (1 - price)
    return math.ceil(round(raw * 1e4, 6)) / 1e4


# ---- books ---------------------------------------------------------------------------------------
class Books:
    """Per-market price -> size on both bid sides (YES bids, NO bids; NO in NO-leg prices)."""

    def __init__(self) -> None:
        self.lv: dict[str, dict[str, dict[float, float]]] = {}

    def apply(self, kind: str, m: str, msg: dict[str, Any]) -> None:
        if kind == "snapshot":
            self.lv[m] = {s: {float(p): float(q) for p, q in msg.get(f"{s}_dollars_fp") or []}
                          for s in ("yes", "no")}
        elif kind == "delta" and m in self.lv:
            side = self.lv[m][msg["side"]]
            p = float(msg["price_dollars"])
            q = side.get(p, 0.0) + float(msg["delta_fp"])
            if q > 1e-9:
                side[p] = q
            else:
                side.pop(p, None)

    def best(self, m: str, side: str) -> tuple[float, float] | None:
        lv = self.lv.get(m, {}).get(side)
        if not lv:
            return None
        p = max(lv)
        return p, lv[p]

    def buy_price(self, m: str, outcome: str) -> tuple[float, float] | None:
        """(price, size) to buy ``outcome`` now: YES at 1 - best NO bid, NO at 1 - best YES bid."""
        b = self.best(m, "no" if outcome == "yes" else "yes")
        return None if b is None else (round(1 - b[0], 6), b[1])

    def sell_price(self, m: str, outcome: str) -> float | None:
        b = self.best(m, outcome)
        return None if b is None else b[0]

    def crossed(self, m: str) -> bool:
        y, n = self.best(m, "yes"), self.best(m, "no")
        return bool(y and n and y[0] + n[0] > 1 + 1e-9)


# ---- sets ----------------------------------------------------------------------------------------
@dataclass
class ArbSet:
    key: str
    kind: str                 # me_yes | me_no | ladder
    event: str
    legs: list[tuple[str, str]]   # (market, outcome bought)
    payoff: float             # guaranteed payout per lot
    close_ns: dict[str, int] = field(default_factory=dict)
    mult: float = 1.0


def _ts_ns(s: str | None) -> int | None:
    if not s:
        return None
    return int(datetime.fromisoformat(s).timestamp() * 1e9)


def build_sets(events: dict[str, dict[str, Any]], recorded: set[str],
               fee_mult: dict[str, float] | None = None) -> list[ArbSet]:
    """``events``: event_ticker -> {"me": bool, "legs": [{t, strike_type, floor, close_time}],
    "exhaustive": bool (optional, default = me and every leg recorded)}."""
    out: list[ArbSet] = []
    for e, meta in events.items():
        legs = meta.get("legs") or []
        tickers = [leg["t"] for leg in legs]
        if len(tickers) < 2:
            continue
        mult = (fee_mult or {}).get(e.split("-")[0], 1.0)
        close = {leg["t"]: _ts_ns(leg.get("close_time")) for leg in legs if leg.get("close_time")}
        if meta.get("me") and set(tickers) <= recorded:
            if meta.get("exhaustive", True):
                out.append(ArbSet(f"{e}|yes", "me_yes", e, [(t, "yes") for t in tickers], 1.0, close, mult))
            out.append(ArbSet(f"{e}|no", "me_no", e, [(t, "no") for t in tickers],
                              float(len(tickers) - 1), close, mult))
        if not meta.get("me"):
            strikes = sorted((float(leg["floor"]), leg["t"]) for leg in legs
                             if leg.get("strike_type") in ("greater", "greater_or_equal")
                             and leg.get("floor") is not None and leg["t"] in recorded)
            for i in range(len(strikes)):
                for j in range(i + 1, len(strikes)):
                    (x1, lo), (x2, hi) = strikes[i], strikes[j]
                    if x2 > x1:
                        out.append(ArbSet(f"{e}|{x1:g}<{x2:g}", "ladder", e,
                                          [(lo, "yes"), (hi, "no")], 1.0, close, mult))
    return out


def set_edge(s: ArbSet, books: Books, now_ns: int, qty: float = 1.0
             ) -> tuple[float, float, list[tuple[float, float]]] | None:
    """(edge per lot after fees, max size, [(price, size) per leg]) or None if not quotable."""
    quotes = []
    for m, outcome in s.legs:
        c = s.close_ns.get(m)
        if c is not None and now_ns >= c:
            return None
        q = books.buy_price(m, outcome)
        if q is None or q[0] <= 0 or q[0] >= 1:
            return None
        quotes.append(q)
    cost = sum(p for p, _ in quotes)
    fees = sum(taker_fee(qty, p, s.mult) for p, _ in quotes) / qty
    return s.payoff - cost - fees, min(sz for _, sz in quotes), quotes


# ---- replay --------------------------------------------------------------------------------------
@dataclass
class Episode:
    key: str
    kind: str
    event: str
    start_ns: int
    end_ns: int | None = None
    edge0: float = 0.0
    best_edge: float = 0.0
    size0: float = 0.0
    best_dollars: float = 0.0     # max over the episode of edge x size (displayed-size capture)
    limits: list[float] = field(default_factory=list)
    trades_on_legs: int = 0
    same_conn: bool = True
    latency: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass
class Event:
    recv_ns: int
    kind: str
    market: str
    msg: dict[str, Any]
    conn: str = ""


def replay(events: Iterator[Event], sets: list[ArbSet], conn_of: dict[str, str] | None = None,
           latencies_ms: tuple[int, ...] = LATENCIES_MS) -> tuple[list[Episode], dict[str, int]]:
    books = Books()
    by_market: dict[str, list[ArbSet]] = defaultdict(list)
    for s in sets:
        for m, _ in s.legs:
            by_market[m].append(s)
    conn_of = conn_of or {}
    open_ep: dict[str, Episode] = {}
    done: list[Episode] = []
    checks: list[tuple[int, int, str, int, Episode]] = []   # (due, seq, stage, L, ep)
    seq = itertools.count()
    diag: dict[str, int] = defaultdict(int)
    set_by_key = {s.key: s for s in sets}

    def run_checks(until: int) -> None:
        while checks and checks[0][0] < until:
            due, _, stage, lat, ep = heapq.heappop(checks)
            s = set_by_key[ep.key]
            r = ep.latency[lat]
            if stage == "fill":
                filled, cost = [], 0.0
                for (m, outcome), lim in zip(s.legs, ep.limits):
                    q = books.buy_price(m, outcome)
                    ok = q is not None and q[0] <= lim + 1e-9 and q[1] >= 1
                    filled.append(ok)
                    if ok:
                        cost += q[0] + taker_fee(1, q[0], s.mult)
                r["filled"] = filled
                if all(filled):
                    r["pnl"] = s.payoff - cost
                elif not any(filled):
                    r["pnl"] = 0.0
                else:
                    r["cost"] = cost
                    heapq.heappush(checks, (due + lat * 10**6, next(seq), "unwind", lat, ep))
            else:   # unwind the filled legs by selling at the bid
                proceeds = 0.0
                for (m, outcome), ok in zip(s.legs, r["filled"]):
                    if ok:
                        b = books.sell_price(m, outcome) or 0.0
                        proceeds += b - taker_fee(1, b, s.mult)
                r["pnl"] = proceeds - r["cost"]

    last_conn: dict[str, str] = dict(conn_of)
    for ev in events:
        run_checks(ev.recv_ns)
        if ev.conn:
            last_conn[ev.market] = ev.conn
        if ev.kind == "trade":
            for s in by_market.get(ev.market, ()):
                if s.key in open_ep:
                    open_ep[s.key].trades_on_legs += 1
            continue
        books.apply(ev.kind, ev.market, ev.msg)
        if books.crossed(ev.market):
            diag["crossed_single_market_events"] += 1
            continue
        for s in by_market.get(ev.market, ()):
            r = set_edge(s, books, ev.recv_ns)
            ep = open_ep.get(s.key)
            live = r is not None and r[0] > 1e-9
            if live:
                edge, size, quotes = r
                if ep is None:
                    conns = {last_conn.get(m) for m, _ in s.legs}
                    ep = Episode(s.key, s.kind, s.event, ev.recv_ns, edge0=edge, best_edge=edge,
                                 size0=size, limits=[p for p, _ in quotes], same_conn=len(conns) == 1)
                    open_ep[s.key] = ep
                    for lat in latencies_ms:
                        ep.latency[lat] = {}
                        heapq.heappush(checks, (ev.recv_ns + lat * 10**6, next(seq), "fill", lat, ep))
                ep.best_edge = max(ep.best_edge, edge)
                ep.best_dollars = max(ep.best_dollars, edge * size)
            elif ep is not None:
                ep.end_ns = ev.recv_ns
                done.append(open_ep.pop(s.key))
    run_checks(2**62)
    for ep in open_ep.values():
        diag["episodes_open_at_end"] += 1
        done.append(ep)
    return done, diag


# ---- loading -------------------------------------------------------------------------------------
def iter_stream(files: list[Path]) -> Iterator[Event]:
    """Book and trade events from recorded files, merged hour by hour in receive order, tagged
    with the recorder connection (file name ``HH-<conn>``)."""
    from albert.stream import read_stream

    by_hour: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        by_hour[f"{p.parent.name}T{p.name[:2]}"].append(p)
    kinds = {"orderbook_snapshot": "snapshot", "orderbook_delta": "delta", "trade": "trade"}
    for hour in sorted(by_hour):
        streams = []
        for p in by_hour[hour]:
            conn = p.name.split(".")[0][3:].split("-")[0]
            evs = []
            for row in read_stream(p):
                d = json.loads(row["frame"])
                kind = kinds.get(d.get("type"))
                if kind:
                    evs.append(Event(row["recv_ns"], kind, d["msg"]["market_ticker"], d["msg"], conn))
            streams.append(evs)
        yield from heapq.merge(*streams, key=lambda e: e.recv_ns)


def summarize(eps: list[Episode], hours: float) -> dict[str, Any]:
    """Counts, durations, edges and latency survival per kind (plain dicts, for reports)."""
    out: dict[str, Any] = {}
    for kind in sorted({e.kind for e in eps}):
        ks = [e for e in eps if e.kind == kind]
        durs = sorted((e.end_ns - e.start_ns) / 1e6 for e in ks if e.end_ns is not None)
        row: dict[str, Any] = {
            "episodes": len(ks), "events": len({e.event for e in ks}),
            "same_conn_share": sum(e.same_conn for e in ks) / len(ks),
            "dur_ms_median": durs[len(durs) // 2] if durs else None,
            "dur_ms_p90": durs[int(len(durs) * 0.9)] if durs else None,
            "edge0_c_median": 100 * sorted(e.edge0 for e in ks)[len(ks) // 2],
            "best_dollars_sum": sum(e.best_dollars for e in ks),
            "taken_by_trade_share": sum(e.trades_on_legs > 0 for e in ks) / len(ks),
        }
        for lat in LATENCIES_MS:
            rs = [e.latency.get(lat, {}) for e in ks if "pnl" in e.latency.get(lat, {})]
            full = [r for r in rs if all(r["filled"])]
            part = [r for r in rs if any(r["filled"]) and not all(r["filled"])]
            row[f"L{lat}"] = {"tested": len(rs), "all_filled": len(full),
                              "partial": len(part), "pnl_1lot": sum(r["pnl"] for r in rs),
                              "partial_pnl": sum(r["pnl"] for r in part)}
        row["per_day_L50_1lot"] = row["L50"]["pnl_1lot"] * 24 / hours if hours else None
        out[kind] = row
    return out
