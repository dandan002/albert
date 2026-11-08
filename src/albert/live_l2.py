"""Live Kalshi maker: the paper maker's Quoter driving real orders (demo or prod).

``LiveBroker`` gives the strategy the same interface as ``backtest.l2.Simulator`` (``place``,
``cancel``, ``live_orders``, ``position``, ``fills``, ``markets``, ``synced``, ``now``,
``ext_at``), so ``Quoter`` runs unchanged. The differences from the simulator:

* **Our orders are subtracted from the book the strategy sees.** The live WebSocket book
  includes our own resting orders; without the subtraction an improved quote looks like the
  external best and the strategy would bid against itself. Every order we have sent and not seen
  cancelled counts (pending, resting, cancelling), clamped at zero, so the view is conservative
  for the few hundred ms between our request and the book delta.
* **Orders and fills come from the exchange.** ``place``/``cancel`` queue a request for the
  action worker (one at a time, so a cancel never overtakes its create) and return at once, like
  the simulator. Order state follows the REST responses and the ``user_orders`` channel; fills,
  position and cash follow the ``fill`` channel only (deduplicated by trade id).

``LiveMaker`` runs it: market-data and private WebSocket connections (recorded, as in the paper
run), the action worker, guards and reconciliation. Guards, in order of defence:

1. Strategy limits (``Quoter.max_pos``) and broker limits: per-order size, open orders, gross
   contracts in unsettled markets, deployed dollars (``max_cost``: cost of fills in unsettled
   markets, including those carried over from earlier runs of the same tag, plus the collateral
   of open orders), blocked markets (any market where the account already has a position or a
   resting order that is not this tag's, e.g. the user's manual trades or another bot's).
   Held markets are checked for a result every reconcile; a settled market leaves gross,
   deployed and reconciliation and is not quoted again.
2. Run guards: daily loss limit on mark-to-mid PnL, stale feed (no market data for N s: cancel
   everything and pause), sequence gaps (cancel everything, pause), foreign or taker fills in a
   bot market (halt), reconciliation against REST every minute (two consecutive mismatches halt),
   kill switch file and SIGTERM.
3. Exchange-side backstops that work even if this process dies: every order is post-only, has an
   ``expiration_time`` (dead-man's switch) and ``cancel_order_on_pause``, and belongs to the run's
   order group, whose ``contracts_limit`` cancels all of its orders when exceeded.

A halt cancels the bot's own orders only (by order id, then by client_order_id prefix). The bot
never calls Cancel All Orders: the account also holds the user's manual orders. Several bots share
the account: each live maker has a one-letter ``tag`` (client ids ``alb<tag><run_id>-<n>``) and only
ever sweeps its own tag; the mentions bot uses ``albm``, so tag ``m`` is refused.
"""

import asyncio
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from albert.backtest.l2 import (
    EPS,
    Event,
    Fill,
    Level,
    Market,
    OnlineClassifier,
    Order,
    SimConfig,
    Simulator,
    _pm_tops,
)
from albert.timing import LoopLag, OrderTiming
from albert.venues.kalshi_trade import (
    ENVS,
    NotAllowed,
    TradeError,
    order_body,
    parse_fill,
    parse_order,
)

BOT_PREFIX = "alb"
SEND_CONCURRENCY = 4                # order requests in flight at once (warm connections kept)
WARM_EVERY_S = 20.0                 # re-warm the pool before the load balancer's idle timeout                 # every bot client_order_id starts with this
MARKOUT_S = (10, 60, 300)
OPEN_STATES = ("pending", "active", "cancelling", "cancel_pending")

# Ceilings for the pilot. Raising them is a code change, not a flag.
HARD_LIMITS = {"max_order_qty": 1.0, "max_gross": 30.0, "max_loss": 25.0, "max_open_orders": 80,
               "max_cost": 25.0}


def check_tag(tag: str) -> str:
    """A maker tag is one lowercase letter other than ``m`` (the mentions bot's ``albm``), so no
    bot's client_order_id prefix is a prefix of another's."""
    if len(tag) != 1 or not tag.isascii() or not tag.islower() or tag == "m":
        raise NotAllowed(f"maker tag must be one lowercase letter other than 'm', got {tag!r}")
    return tag


def carried_from_ledgers(paths: list[Path]) -> dict[str, dict[str, float]]:
    """Own fills of earlier runs per market: ``cost`` (price x qty + fees) and ``net`` YES contracts.
    The caller drops markets that have settled since."""
    out: dict[str, dict[str, float]] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") != "fill":
                continue
            c = out.setdefault(r["market"], {"cost": 0.0, "net": 0.0})
            c["cost"] += r["qty"] * r["price"] + r["fee"]
            c["net"] += r["qty"] if r["side"] == "yes" else -r["qty"]
    return out


@dataclass
class LiveGuards:
    max_order_qty: float = 1.0
    max_gross: float = 10.0            # open contracts across markets; no new orders above
    max_loss: float = 10.0             # halt when run PnL (marked to mid, fees in) < -$X
    max_open_orders: int = 40
    stale_feed_s: float = 120.0        # no market data at all this long: cancel all, pause
    gap_pause_s: float = 5.0           # after a sequence gap or disconnect: cancel all, pause
    reconcile_every_s: float = 60.0
    reconcile_grace_s: float = 5.0     # ignore orders touched this recently when reconciling
    order_expiry_s: float = 900.0      # expiration_time on every order
    group_contracts_limit: int = 5     # exchange-side: matched contracts per rolling 15 s
    max_cost: float | None = None      # $ deployed (see module doc); no new orders above
    status_every_s: float = 60.0
    queue_every_s: float = 30.0        # sample exchange queue positions of our resting orders

    def check(self) -> None:
        for k, ceiling in HARD_LIMITS.items():
            if getattr(self, k) is not None and getattr(self, k) > ceiling:
                raise NotAllowed(f"{k}={getattr(self, k)} exceeds the pilot ceiling {ceiling}")


class LiveBroker:
    """Simulator-compatible strategy interface backed by real orders. Not thread-safe; everything
    runs on one asyncio loop."""

    def __init__(self, cfg: SimConfig, guards: LiveGuards, *, client_prefix: str,
                 order_group_id: str | None = None, blocked: set[str] | None = None) -> None:
        self.cfg, self.guards = cfg, guards
        self.prefix = client_prefix
        self.group = order_group_id
        self.blocked = set(blocked or ())
        self.book = Simulator(SimConfig(writes_per_s=None))   # raw book + ext; never places
        self.orders: dict[int, Order] = {}
        self.by_market: dict[str, list[int]] = defaultdict(list)
        self.fills: list[Fill] = []
        self.position: dict[str, float] = defaultdict(float)
        self.cash = 0.0
        self.quotes: dict[str, tuple[list[int], list[tuple]]] = defaultdict(lambda: ([], []))
        self.diag: dict[str, int] = defaultdict(int)
        self.actions: list[tuple[str, int]] = []
        self.exch_id: dict[int, str] = {}
        self.by_exch: dict[str, int] = {}
        self.by_coid: dict[str, int] = {}
        self.touched_ns: dict[int, int] = {}
        self.trade_ids: set[str] = set()
        self.halted: str | None = None
        self.paused_until = 0
        self.events: list[tuple[str, dict[str, Any]]] = []   # for the ledger
        self._next = 1
        self._tokens = cfg.writes_per_s or 0.0
        self._tokens_ns = 0
        self.timing = OrderTiming()
        self.shard: dict[str, int] = {}      # exchange_index per market; missing -> auto-route
        self.results: dict[str, float] = {}  # settled market -> YES payout
        self.carried: dict[str, dict[str, float]] = {}   # earlier runs' unsettled own fills
        self.trigger_ns: int | None = None   # recv_ns of the market-data event being processed

    # -- strategy API (same as Simulator) --
    @property
    def now(self) -> int:
        return self.book.now

    @property
    def synced(self) -> set[str]:
        return self.book.synced

    @property
    def markets(self) -> "_NetMarkets":
        return _NetMarkets(self)

    def ext_at(self, market: str, t_ns: int) -> float | None:
        return self.book.ext_at(market, t_ns)

    def ext_range(self, market: str, t0: int, t1: int) -> float:
        return self.book.ext_range(market, t0, t1)

    def mid_at(self, market: str, t_ns: int) -> float | None:
        return Simulator.mid_at(self, market, t_ns)   # over our net quotes

    def quote_at(self, market: str, t_ns: int) -> tuple[float | None, float | None]:
        return Simulator.quote_at(self, market, t_ns)

    def exit_at(self, market: str, side: str, t_ns: int) -> float | None:
        return Simulator.exit_at(self, market, side, t_ns)

    def _spend(self, cost: float) -> bool:
        return Simulator._spend(self, cost)

    def open_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if o.status in OPEN_STATES]

    def gross(self) -> float:
        """Open contracts in unsettled markets, this run's and carried ones."""
        return (sum(abs(q) for m, q in self.position.items() if m not in self.results)
                + sum(abs(c["net"]) for m, c in self.carried.items() if m not in self.results))

    def deployed(self) -> float:
        """Dollars tied up: cost of every own fill in an unsettled market (a YES+NO pair stays
        counted until settlement), carried fills of earlier runs, and open orders' collateral."""
        fills = sum(f.qty * f.price + f.fee for f in self.fills if f.market not in self.results)
        carried = sum(c["cost"] for m, c in self.carried.items() if m not in self.results)
        resting = sum(o.price * o.rem for o in self.open_orders())
        return fills + carried + resting

    def settle(self, market: str, result: float) -> None:
        """Record a market's result: it leaves gross, deployed and reconciliation, and is blocked."""
        if market in self.results:
            return
        self.results[market] = result
        self.blocked.add(market)
        for o in self.live_orders(market):
            self.cancel(o.oid)
        yes = sum(f.qty for f in self.fills if f.market == market and f.side == "yes")
        no = sum(f.qty for f in self.fills if f.market == market and f.side == "no")
        self.events.append(("settle", {"market": market, "result": result, "yes": yes, "no": no,
                                       "payout": yes * result + no * (1 - result),
                                       "carried": self.carried.get(market)}))

    def place(self, market: str, side: str, price: float, qty: float = 1.0) -> int | None:
        g = self.guards
        why = ("halted" if self.halted else
               "paused" if self.now < self.paused_until else
               "blocked_market" if market in self.blocked else
               "order_qty" if qty > g.max_order_qty + EPS else
               "open_orders" if len(self.open_orders()) >= g.max_open_orders else
               "gross" if self.gross() >= g.max_gross - EPS else
               "budget" if g.max_cost is not None and self.deployed() + price * qty > g.max_cost + EPS
               else None)
        if why:
            self.diag[f"refused_{why}"] += 1
            return None
        if not self._spend(1.0):
            self.diag["throttled_places"] += 1
            return None
        oid = self._next
        self._next += 1
        o = Order(oid, market, side, round(price, 6), qty, self.now, self.now, rem=qty)
        self.orders[oid] = o
        self.by_market[market].append(oid)
        self.by_coid[self.client_id(oid)] = oid
        self.touched_ns[oid] = self.now
        self._queue("create", oid)
        return oid

    def cancel(self, oid: int) -> bool:
        o = self.orders[oid]
        if o.status not in ("pending", "active"):
            return True
        if not self._spend(self.cfg.cancel_cost):
            self.diag["throttled_cancels"] += 1
            return False
        self.touched_ns[oid] = self.now
        if o.status == "pending" and oid not in self.exch_id:
            o.status = "cancel_pending"            # cancelled as soon as the create is acked
        else:
            o.status = "cancelling"
            self._queue("cancel", oid)
        return True

    def live_orders(self, market: str) -> list[Order]:
        return [self.orders[i] for i in self.by_market[market]
                if self.orders[i].status in ("pending", "active")]

    def _queue(self, kind: str, oid: int) -> None:
        self.actions.append((kind, oid))
        self.timing.queued(kind, oid, self.orders[oid].market, self.trigger_ns)

    def client_id(self, oid: int) -> str:
        return f"{self.prefix}-{oid}"

    # -- book --
    def ours(self, market: str) -> dict[tuple[str, float], float]:
        out: dict[tuple[str, float], float] = defaultdict(float)
        for i in self.by_market.get(market, ()):
            o = self.orders[i]
            if o.status in OPEN_STATES:
                out[(o.side, o.price)] += o.rem
        return out

    def net_market(self, market: str) -> Market:
        raw = self.book.markets[market]
        mine = self.ours(market)
        m = Market()
        for side in ("yes", "no"):
            for p, lev in raw.levels[side].items():
                net = lev.ext_total() - mine.get((side, p), 0.0)
                if net > EPS:
                    nl = Level()
                    nl.q.append([net, None])
                    m.levels[side][p] = nl
        return m

    def apply(self, ev: Event) -> None:
        self.book.apply(ev)
        if ev.kind in ("snapshot", "delta") and ev.market in self.book.synced:
            self._record_quote(ev.market)

    def _record_quote(self, market: str) -> None:
        q = self.net_market(market).quote()
        times, vals = self.quotes[market]
        if not vals or vals[-1] != (q["yes_bid"], q["yes_ask"]):
            times.append(self.now)
            vals.append((q["yes_bid"], q["yes_ask"]))

    # -- exchange updates --
    def on_created(self, oid: int, resp: dict[str, Any]) -> None:
        o = self.orders[oid]
        self.exch_id[oid] = resp["order_id"]
        self.by_exch[resp["order_id"]] = oid
        if o.status == "cancel_pending":
            o.status = "cancelling"
            self._queue("cancel", oid)
        elif o.status == "pending":
            o.status = "active"

    def on_rejected(self, oid: int, err: TradeError) -> None:
        o = self.orders[oid]
        if o.status in ("pending", "cancel_pending"):
            o.status = "rejected"
        self.diag[f"rejects_{err.status}"] += 1
        self.events.append(("reject", {"oid": oid, "market": o.market, "side": o.side,
                                       "price": o.price, "status": err.status, "body": err.body}))

    def on_cancelled(self, oid: int) -> None:
        o = self.orders[oid]
        if o.status not in ("filled",):
            o.status = "cancelled"
        o.cancel_ns = self.now

    def _own_oid(self, coid: str, order_id: str) -> int | None:
        if order_id in self.by_exch:
            return self.by_exch[order_id]
        return self.by_coid.get(coid)

    def on_private(self, d: dict[str, Any], now_ns: int) -> None:
        typ, msg = d.get("type"), d.get("msg") or {}
        if typ == "user_order":
            st = parse_order(msg)
            oid = self._own_oid(st.client_order_id, st.order_id)
            if oid is None:
                self.diag["foreign_order_updates" if not st.client_order_id.startswith(BOT_PREFIX)
                          else "unknown_bot_order_updates"] += 1
                if st.ticker in self.by_market and st.status == "resting":
                    self.halt(f"foreign resting order appeared in bot market {st.ticker}")
                return
            o = self.orders[oid]
            if oid not in self.exch_id:
                self.exch_id[oid] = st.order_id
                self.by_exch[st.order_id] = oid
            if st.status == "resting":
                self.timing.ws("create", oid, now_ns)
            elif st.status == "canceled":
                self.timing.ws("cancel", oid, now_ns)
            if st.status == "resting" and o.status == "pending":
                o.status = "active"
            elif st.status == "resting" and o.status == "cancel_pending":
                o.status = "cancelling"
                self._queue("cancel", oid)
            elif st.status == "canceled":
                self.on_cancelled(oid)
            elif st.status == "executed" and o.rem <= EPS:
                o.status = "filled"
        elif typ == "fill":
            fr = parse_fill(msg)
            if fr.trade_id in self.trade_ids:
                self.diag["duplicate_fills"] += 1
                return
            self.trade_ids.add(fr.trade_id)
            oid = self._own_oid(fr.client_order_id, fr.order_id)
            if oid is None:
                self.diag["foreign_fills"] += 1
                if fr.ticker in self.by_market:
                    self.halt(f"foreign fill in bot market {fr.ticker}")
                return
            o = self.orders[oid]
            if fr.is_taker:
                self.halt(f"taker fill on post-only order {oid}")
            qty = fr.qty
            o.rem = max(0.0, o.rem - qty)
            if o.rem <= EPS:
                o.status = "filled"
            self.fills.append(Fill(now_ns, oid, o.market, o.side, fr.price, qty, fr.fee, "live"))
            self.cash -= qty * fr.price + fr.fee
            self.position[o.market] += qty if o.side == "yes" else -qty
            if abs(fr.price - o.price) > EPS:
                self.diag["fill_price_differs"] += 1
            if fr.post_position is not None and abs(fr.post_position - self.position[o.market]) > 1e-6:
                self.diag["post_position_mismatch"] += 1
            self.touched_ns[oid] = self.now
        elif (typ == "order_group_updates" and msg.get("order_group_id") == self.group
              and msg.get("event_type") in ("triggered", "deleted")):
            self.halt(f"order group {msg.get('event_type')}")

    # -- halts --
    def cancel_all(self, reason: str) -> int:
        n = 0
        for o in self.open_orders():
            if o.status in ("pending", "active"):
                self.touched_ns[o.oid] = self.now
                if o.status == "pending" and o.oid not in self.exch_id:
                    o.status = "cancel_pending"
                else:
                    o.status = "cancelling"
                    self._queue("cancel", o.oid)
                n += 1
        self.diag[f"cancel_all_{reason}"] += 1
        return n

    def halt(self, reason: str) -> None:
        if self.halted:
            return
        self.halted = reason
        self.cancel_all("halt")
        self.events.append(("halt", {"reason": reason}))

    def pause(self, seconds: float, reason: str) -> None:
        self.paused_until = max(self.paused_until, self.now + int(seconds * 1e9))
        self.cancel_all(reason)

    def equity(self) -> float:
        """Mark-to-net-mid PnL of the run (per fill; YES and NO in one market net to $1)."""
        held: dict[str, list[float]] = {}
        pnl = 0.0
        for f in self.fills:
            h = held.setdefault(f.market, [0.0, 0.0])
            h[0 if f.side == "yes" else 1] += f.qty
            pnl -= f.qty * f.price + f.fee
        for m, (yes, no) in held.items():
            mid = self.results.get(m)
            if mid is None:
                mid = self.mid_at(m, 2**62)
            if mid is None:
                mid = self.book.mid_at(m, 2**62)
            if mid is None:
                mid = 0.0 if yes >= no else 1.0   # unmarkable: assume the worst for the bigger leg
            pnl += yes * mid + no * (1 - mid)
        return pnl


class _NetMarkets:
    def __init__(self, broker: LiveBroker) -> None:
        self.b = broker

    def __getitem__(self, market: str) -> Market:
        return self.b.net_market(market)


async def run_actions(broker: LiveBroker, trader, *, expiry_s: float,
                      wall_s=time.time, max_errors: int = 5, concurrency: int = 4) -> int:
    """Send every queued action, up to ``concurrency`` requests in flight at once (a burst of
    N orders no longer waits N round trips). Returns the number sent once the queue is empty and
    nothing is in flight. Order per oid is safe: a cancel is queued only after its create was
    acked (``cancel`` marks an unacked create ``cancel_pending``; ``on_created`` queues it)."""
    state = {"sent": 0, "errors": 0}
    shard = getattr(broker, "shard", {})

    def error(e_count: int = 1) -> None:
        state["errors"] += e_count
        if state["errors"] >= max_errors:
            broker.halt(f"{state['errors']} order errors")

    async def one(kind: str, oid: int) -> None:
        o = broker.orders[oid]
        tm = broker.timing
        try:
            if kind == "create":
                if o.status not in ("pending", "cancel_pending"):
                    tm.drop(kind, oid)
                    return
                body = order_body(o.market, o.side, o.price, o.qty,
                                  client_order_id=broker.client_id(oid),
                                  order_group_id=broker.group,
                                  expiration_ts=int(wall_s() + expiry_s),
                                  exchange_index=shard.get(o.market, -1))
                tm.sent(kind, oid)
                resp = await trader.create(body)
                tm.acked(kind, oid, resp)
                broker.on_created(oid, resp)
            elif kind == "cancel":
                if oid not in broker.exch_id or o.status in ("cancelled", "filled", "rejected"):
                    tm.drop(kind, oid)
                    return
                tm.sent(kind, oid)
                resp = await trader.cancel(broker.exch_id[oid], o.market)
                tm.acked(kind, oid, resp)
                broker.on_cancelled(oid)
            state["sent"] += 1
        except httpx.HTTPError as e:
            # The request may or may not have reached the exchange. A create is treated as live
            # and to be cancelled: if it exists, the user_orders channel or reconciliation finds
            # it by client_order_id and cancels it. A cancel is retried by the strategy.
            tm.failed(kind, oid, "transport")
            broker.diag[f"transport_errors_{kind}"] += 1
            broker.events.append(("error", {"kind": kind, "oid": oid, "transport": repr(e)[:200]}))
            if kind == "create":
                o.status = "cancel_pending"
            elif o.status == "cancelling":
                o.status = "active"
            error()
        except TradeError as e:
            tm.failed(kind, oid, f"http {e.status}")
            if kind == "create":
                broker.on_rejected(oid, e)
                if e.status >= 500 or e.status == 429:
                    error()
            elif e.status == 404:
                broker.on_cancelled(oid)      # already gone: the user_orders channel has the truth
            else:
                broker.events.append(("error", {"kind": kind, "oid": oid, "status": e.status,
                                                "body": e.body}))
                o.status = "active" if o.status == "cancelling" else o.status
                error()

    inflight: set[asyncio.Task] = set()
    while broker.actions or inflight:
        while broker.actions and len(inflight) < concurrency:
            kind, oid = broker.actions.pop(0)
            inflight.add(asyncio.create_task(one(kind, oid)))
        if inflight:
            done, _ = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
            inflight -= done
            for d in done:
                d.result()                      # surface bugs (NotAllowed etc.) instead of hiding them
    return state["sent"]


def reconcile(broker: LiveBroker, resting: list, positions: dict[str, float],
              baseline: dict[str, float]) -> list[str]:
    """Differences between our state and the exchange's, ignoring recently touched orders.
    ``resting``: OrderState list from REST; ``baseline``: positions held before the run."""
    grace = int(broker.guards.reconcile_grace_s * 1e9)
    recent = {oid for oid, t in broker.touched_ns.items() if broker.now - t < grace}
    ours_exch = {o.order_id for o in resting if o.client_order_id.startswith(broker.prefix)}
    issues = []
    for oid, o in broker.orders.items():
        if oid in recent:
            continue
        eid = broker.exch_id.get(oid)
        live_local = o.status in ("active", "cancelling")
        if live_local and eid not in ours_exch:
            issues.append(f"order {oid} ({o.market}) {o.status} locally but not resting")
        if not live_local and eid in ours_exch:
            issues.append(f"order {oid} ({o.market}) resting on the exchange but {o.status} locally")
    for o in resting:
        if not o.client_order_id.startswith(broker.prefix) or o.order_id in broker.by_exch:
            continue
        oid = broker.by_coid.get(o.client_order_id)
        if oid is None:
            issues.append(f"unknown bot order {o.order_id} resting in {o.ticker}")
            continue
        # A create whose response was lost: adopt it, and cancel it unless we still want it.
        broker.exch_id[oid] = o.order_id
        broker.by_exch[o.order_id] = oid
        lo = broker.orders[oid]
        if lo.status in ("cancel_pending", "cancelled", "rejected"):
            lo.status = "cancelling"
            broker.actions.append(("cancel", oid))
        elif lo.status == "pending":
            lo.status = "active"
    for m in broker.by_market:
        if m in broker.results:               # settled: the exchange drops the position
            continue
        if any(broker.touched_ns.get(i, 0) > broker.now - grace for i in broker.by_market[m]):
            continue
        ex = positions.get(m, 0.0) - baseline.get(m, 0.0)
        if abs(ex - broker.position[m]) > 1e-6:
            issues.append(f"position {m}: exchange {ex:+g} vs local {broker.position[m]:+g}")
    return issues


class LiveMaker:
    """The live run: feeds, worker, guards, ledger. ``trader`` is a ``KalshiTrader``."""

    def __init__(self, run_dir: Path, env: str, signer, trader, *, run_id: str, strategy,
                 cfg: SimConfig, guards: LiveGuards, pairs: list[dict[str, Any]],
                 holdback_ms: float = 0.0, stop_file: Path | None = None, tag: str = "q",
                 carried: dict[str, dict[str, float]] | None = None,
                 shards_allowed: frozenset[int] = frozenset({0})) -> None:
        from albert.stream import Recorder

        guards.check()
        self.dir, self.env, self.trader = run_dir, env, trader
        self.strategy, self.guards = strategy, guards
        self.tag_prefix = f"{BOT_PREFIX}{check_tag(tag)}"     # every order of this maker, any run
        self.prefix = f"{self.tag_prefix}{run_id}"
        self.carried_in = dict(carried or {})
        # The account's cash sits on shard 0; an order on a shard without balance is rejected
        # (insufficient_shard_balance, e.g. NBA on shard 3), so such markets are blocked.
        self.shards_allowed = shards_allowed
        self.broker = LiveBroker(cfg, guards, client_prefix=self.prefix)
        # No holdback live (2026-11-07): the classifier's labels only move external size between
        # chunks of a level, and the live book holds no own orders, so the level totals the
        # strategy reads are the same either way. A holdback only delayed every decision.
        self.clf = OnlineClassifier(holdback_ms)
        self.stop_file = stop_file
        self.baseline: dict[str, float] = {}
        self.by_token: dict[str, list[tuple[str, bool]]] = {}
        for p in pairs:
            self.by_token.setdefault(p["token_id"], []).append((p["kalshi_ticker"], bool(p.get("complement"))))
        self.rec = Recorder(run_dir / "stream", signer, on_frame=self.on_frame,
                            kalshi_ws=ENVS[env]["ws"], kalshi_private_ws=ENVS[env]["ws_private"])
        self.ledger = open(run_dir / "ledger.jsonl", "a", buffering=1)  # noqa: SIM115 - closed in close()
        self.last_md_ns = time.time_ns()
        self.gaps_seen = 0
        self.drops_seen = 0
        self.mismatches = 0
        self.fills_logged = 0
        self.pending_markouts: list[tuple[int, int, int]] = []
        self.wake = asyncio.Event()
        self.lag = LoopLag()

    def write(self, kind: str, data: dict[str, Any]) -> None:
        self.ledger.write(json.dumps({"kind": kind, "wall": datetime.now(UTC).isoformat(),
                                      "now_ns": self.broker.now, **data}, default=str) + "\n")

    # -- startup and shutdown --
    async def prepare(self, markets: list[str]) -> list[str]:
        """Cancel leftover bot orders, block markets the account already holds or has manual
        orders in, create the run's order group. Returns the tradable markets."""
        stale = await self.trader.cancel_own(self.tag_prefix)
        resting = await self.trader.orders("resting")
        self.baseline = await self.trader.positions()
        manual = {o.ticker for o in resting if not o.client_order_id.startswith(self.tag_prefix)}
        self.broker.carried = dict(self.carried_in)
        await self.check_settlements(list(self.carried_in))
        self.broker.events.clear()   # carried markets that settled earlier are not this run's
        blocked = (manual | set(self.baseline)) & set(markets)
        self.broker.blocked |= manual | set(self.baseline)
        self.broker.shard = await self.trader.shards([m for m in markets if m not in self.broker.blocked])
        off_shard = {m for m in markets if m not in self.broker.blocked
                     and self.broker.shard.get(m) not in self.shards_allowed}
        self.broker.blocked |= off_shard
        if hasattr(self.trader, "warm"):
            await self.trader.warm(SEND_CONCURRENCY)
        self.broker.group = await self.trader.create_order_group(self.guards.group_contracts_limit)
        balance = await self.trader.balance()
        self.write("prepare", {"cancelled_stale_bot_orders": stale, "manual_order_markets": sorted(manual),
                               "carried": {m: c for m, c in self.broker.carried.items()
                                           if m not in self.broker.results},
                               "deployed": self.broker.deployed(),
                               "positions_before": self.baseline, "blocked": sorted(blocked),
                               "off_shard": sorted(off_shard),
                               "order_group": self.broker.group, "balance": balance,
                               "shards": dict(sorted(Counter(self.broker.shard.values()).items()))})
        return [m for m in markets if m not in self.broker.blocked]

    async def check_settlements(self, markets: list[str] | None = None) -> None:
        """Ask the exchange for the result of every held, unsettled market (own fills this run and
        carried ones) and settle the resolved ones. A failed lookup is retried next time."""
        from albert.paper_l2 import market_result

        b = self.broker
        if markets is None:
            markets = sorted(({f.market for f in b.fills} | set(b.carried)) - set(b.results))
        for m in markets:
            if m in b.results:
                continue
            try:
                page = await self.trader.get(f"/markets/{m}")
            except (TradeError, httpx.HTTPError) as e:
                self.write("error", {"what": "settlement lookup", "market": m, "error": repr(e)[:200]})
                continue
            r = market_result((page or {}).get("market") or {})
            if r is not None:
                b.settle(m, r)

    async def shutdown(self) -> dict[str, Any]:
        b = self.broker
        b.halt(b.halted or "shutdown")
        await run_actions(b, self.trader, expiry_s=self.guards.order_expiry_s,
                          concurrency=SEND_CONCURRENCY)
        leftover = await self.trader.cancel_own(self.prefix)
        if b.group:
            try:
                await self.trader.delete_order_group(b.group)
            except TradeError as e:
                self.write("error", {"what": "delete order group", "status": e.status})
        positions = await self.trader.positions()
        balance = await self.trader.balance()
        for rec in b.timing.flush(time.time_ns() + 10**12):     # whatever is left, WS ack or not
            self.write("timing", rec)
        end = {"halted": b.halted, "stopped_reason": self.rec.stopped_reason,
               "equity": b.equity(), "cash": b.cash, "fills": len(b.fills), "orders": len(b.orders),
               "positions_local": {m: q for m, q in b.position.items() if abs(q) > 1e-9},
               "results": dict(b.results), "deployed": b.deployed(),
               "positions_exchange": positions, "balance": balance,
               "cancelled_at_close": leftover, "diag": dict(b.diag),
               "classifier": dict(self.clf.stats),
               "latency_ms": b.timing.summary(whole_run=True),
               "loop_lag_ms": self.lag.summary(whole_run=True),
               "conn_stats": {c: st.as_dict() for c, st in self.rec.stats.items()}}
        files = self.rec.close()
        end["files"] = [asdict(f) for f in files]
        self.write("end", end)
        self.ledger.close()
        return end

    # -- input --
    def on_frame(self, venue: str, recv_ns: int, frame: str) -> None:
        if venue == "kalshi":
            d = json.loads(frame)
            typ = d.get("type")
            if typ in ("user_order", "fill", "order_group_updates"):
                self.broker.on_private(d, recv_ns)
                self.wake.set()
                return
            kind = {"orderbook_snapshot": "snapshot", "orderbook_delta": "delta",
                    "trade": "trade"}.get(typ)
            if kind:
                self.last_md_ns = recv_ns
                self.clf.push(Event(recv_ns, kind, d["msg"]["market_ticker"], d["msg"]))
        elif venue == "pm":
            for asset, bid, ask in _pm_tops({"frame": frame}):
                for ticker, comp in self.by_token.get(asset, []):
                    b_, a_ = ((None if ask is None else 1 - ask, None if bid is None else 1 - bid)
                              if comp else (bid, ask))
                    mid = (a_ + b_) / 2 if a_ is not None and b_ is not None else None
                    self.clf.push(Event(recv_ns, "ext", ticker, {"bid": b_, "ask": a_, "mid": mid}))
        self.step(recv_ns)

    def step(self, now_ns: int) -> None:
        b = self.broker
        for ev in self.clf.release(now_ns):
            t0 = time.perf_counter_ns()
            b.trigger_ns = ev.recv_ns
            try:
                b.apply(ev)
                if b.halted or ev.market in b.blocked:
                    continue
                if b.gross() >= self.guards.max_gross - EPS:
                    for o in b.live_orders(ev.market):
                        b.cancel(o.oid)
                    continue
                self.strategy.on_event(b, ev.market)
            finally:
                b.trigger_ns = None
                b.timing.samples.add("event_process", (time.perf_counter_ns() - t0) / 1e6)
        if b.actions:
            self.wake.set()
        self._log()

    def _log(self) -> None:
        b = self.broker
        for kind, data in b.events:
            self.write(kind, data)
        b.events.clear()
        for rec in b.timing.flush():
            self.write("timing", rec)
        for i in range(self.fills_logged, len(b.fills)):
            f = b.fills[i]
            self.write("fill", {**asdict(f), "quote": b.net_market(f.market).quote(),
                                "pm_mid": b.ext_at(f.market, f.t_ns), "position": b.position[f.market]})
            for h in MARKOUT_S:
                self.pending_markouts.append((f.t_ns + h * 10**9, i, h))
        self.fills_logged = len(b.fills)
        due = [x for x in self.pending_markouts if x[0] <= b.now]
        if due:
            self.pending_markouts = [x for x in self.pending_markouts if x[0] > b.now]
            for t, i, h in due:
                f = b.fills[i]
                mid = b.mid_at(f.market, t)
                mo = None if mid is None else (mid if f.side == "yes" else 1 - mid) - f.price
                ex = b.exit_at(f.market, f.side, t)
                self.write("markout", {"fill": i, "market": f.market, "h": h, "mo": mo,
                                       "mx": None if ex is None else ex - f.price})

    # -- loops --
    async def worker(self) -> None:
        while not self.rec.stop.is_set():
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=1.0)
            except TimeoutError:
                pass
            self.wake.clear()
            if self.broker.actions:
                await run_actions(self.broker, self.trader, expiry_s=self.guards.order_expiry_s,
                                  concurrency=SEND_CONCURRENCY)
                self._log()

    async def guard_loop(self) -> None:
        g, b = self.guards, self.broker
        next_status = next_recon = next_queue = next_warm = 0.0
        while not self.rec.stop.is_set():
            await self.rec._sleep_or_stop(0.2)
            self.step(time.time_ns())
            if self.stop_file is not None and self.stop_file.exists():
                b.halt("kill switch file")
                self.rec.stopped_reason = "kill switch"
                self.rec.stop.set()
            eq = b.equity()
            if eq < -g.max_loss:
                b.halt(f"loss limit: equity {eq:.2f} < -{g.max_loss}")
            if (time.time_ns() - self.last_md_ns) / 1e9 > g.stale_feed_s and b.open_orders():
                b.pause(g.stale_feed_s, "stale_feed")
            gaps = sum(len(st.gaps) for st in self.rec.stats.values())
            if gaps > self.gaps_seen:
                self.gaps_seen = gaps
                b.pause(g.gap_pause_s, "seq_gap")
                next_recon = 0.0
            drops = sum(len(st.errors) for c, st in self.rec.stats.items() if c.startswith(("k", "u")))
            if drops > self.drops_seen:       # a Kalshi connection failed (it reconnects itself)
                self.drops_seen = drops
                b.pause(g.gap_pause_s, "disconnect")
                next_recon = 0.0
            if b.actions:
                self.wake.set()
            mono = time.monotonic()
            try:
                if mono >= next_recon:
                    next_recon = mono + g.reconcile_every_s
                    await self.check_settlements()
                    issues = reconcile(b, await self.trader.orders("resting"),
                                       await self.trader.positions(), self.baseline)
                    self.mismatches = self.mismatches + 1 if issues else 0
                    if issues:
                        self.write("reconcile", {"issues": issues, "consecutive": self.mismatches})
                    if self.mismatches >= 2:
                        b.halt("reconciliation mismatch")
                if mono >= next_queue:
                    next_queue = mono + g.queue_every_s
                    await self._sample_queue()
                if mono >= next_warm and hasattr(self.trader, "warm"):
                    next_warm = mono + WARM_EVERY_S
                    await self.trader.warm(SEND_CONCURRENCY)
            except TradeError as e:
                self.write("error", {"what": "guard REST", "status": e.status, "body": e.body})
            self._log()
            if mono >= next_status:
                next_status = mono + g.status_every_s
                self.write("status", {
                    "equity": round(eq, 4), "cash": round(b.cash, 4), "gross": b.gross(),
                    "deployed": round(b.deployed(), 4), "settled": len(b.results),
                    "open_orders": len(b.open_orders()), "orders": len(b.orders), "fills": len(b.fills),
                    "synced": len(b.synced), "halted": b.halted, "queued_actions": len(b.actions),
                    "classifier": dict(self.clf.stats), "diag": dict(b.diag),
                    "latency_ms": b.timing.summary(), "loop_lag_ms": self.lag.summary()})

    async def _sample_queue(self) -> None:
        """Exchange queue position of each resting order vs the size the net book shows at its
        price (the most that can be ahead of us): a direct check of the queue model."""
        b = self.broker
        active = [o for o in b.open_orders() if o.status == "active" and o.oid in b.exch_id]
        if not active:
            return
        qp = await self.trader.queue_positions(sorted({o.market for o in active}))
        rows = []
        for o in active:
            eid = b.exch_id[o.oid]
            if eid in qp:
                lev = b.net_market(o.market).levels[o.side].get(o.price)
                rows.append({"oid": o.oid, "market": o.market, "side": o.side, "price": o.price,
                             "age_s": (b.now - o.placed_ns) / 1e9, "queue_position": qp[eid],
                             "level_ex_us": lev.ext_total() if lev else 0.0})
        if rows:
            self.write("queue", {"orders": rows})

    async def run(self, *, minutes: float, kalshi: list[str], pm_tokens: list[str],
                  chunk: int = 100, min_free_gb: float = 10.0) -> None:
        from albert.stream import disk_guard

        tasks = [disk_guard(self.rec, min_free_gb), self.guard_loop(), self.worker(),
                 self.lag.run(self.rec.stop),
                 self.rec.kalshi_private_conn("u0"),
                 self.rec.kalshi_private_conn("u1", ("order_group_updates",), self.rec.kalshi_ws)]
        for i in range(0, len(kalshi), chunk):
            tasks.append(self.rec.kalshi_conn(f"k{i // chunk}", kalshi[i:i + chunk]))
        for i in range(0, len(pm_tokens), chunk):
            tasks.append(self.rec.pm_conn(f"p{i // chunk}", pm_tokens[i:i + chunk]))

        async def timer() -> None:
            await self.rec._sleep_or_stop(minutes * 60)
            self.rec.stop.set()

        await asyncio.gather(timer(), *tasks)


@dataclass
class PlumbingStep:
    name: str
    ok: bool
    detail: dict[str, Any] = field(default_factory=dict)


async def pick_demo_market(trader) -> dict[str, Any]:
    """An open market with a two-sided quote at least 3c wide (room for a non-crossing bid)."""
    cursor = None
    for _ in range(10):
        params: dict[str, Any] = {"status": "open", "mve_filter": "exclude", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        page = await trader.get("/markets", params)
        for m in page.get("markets") or []:
            bid, ask = float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 0)
            if 0.05 <= bid and ask <= 0.95 and ask - bid >= 0.03:
                return m
        cursor = page.get("cursor")
        if not cursor:
            break
    raise RuntimeError("no open demo market with a two-sided quote")


async def plumbing(trader, signer, out_dir: Path, *, market: str | None = None,
                   wait_s: float = 3.0) -> list[PlumbingStep]:
    """Demo-only end-to-end check of every order path the live maker uses. Saves every REST
    response and private WebSocket frame under ``out_dir`` (the source for real fixtures).

    Steps: balance; order group; resting post-only bid (REST + user_orders); queue position;
    cancel; post-only reject (a bid at the ask); expiry (expiration_time 10 s ahead); a 1-contract
    taker fill (IOC; demo only: records the fill channel); order-group trigger cancels the group's
    orders; cancel_own sweep; group delete."""
    from albert.stream import Recorder

    if trader.env != "demo":
        raise NotAllowed("the plumbing test runs on demo only")
    out_dir.mkdir(parents=True, exist_ok=True)
    rest_log = open(out_dir / "rest.jsonl", "a", buffering=1)  # noqa: SIM115, ASYNC230 - one-off test log
    frames: list[dict[str, Any]] = []

    def on_frame(venue: str, recv_ns: int, frame: str) -> None:
        frames.append(json.loads(frame))

    rec = Recorder(out_dir / "stream", signer, on_frame=on_frame, kalshi_ws=ENVS["demo"]["ws"],
                   kalshi_private_ws=ENVS["demo"]["ws_private"])
    ws_task = asyncio.create_task(rec.kalshi_private_conn("u0"))
    ws_task_groups = asyncio.create_task(
        rec.kalshi_private_conn("u1", ("order_group_updates",), rec.kalshi_ws))
    steps: list[PlumbingStep] = []
    prefix = f"{BOT_PREFIX}plumb{int(time.time())}"

    def log(name: str, payload: Any) -> None:
        rest_log.write(json.dumps({"step": name, "payload": payload}, default=str) + "\n")

    def ws_seen(kind: str, coid: str) -> list[dict[str, Any]]:
        return [f for f in frames if f.get("type") == kind
                and (f.get("msg") or {}).get("client_order_id") == coid]

    async def step(name: str, coro) -> Any:
        try:
            res = await coro
            log(name, res)
            return res
        except (TradeError, httpx.HTTPError) as e:
            log(name, {"error": repr(e), "body": getattr(e, "body", None)})
            steps.append(PlumbingStep(name, False, {"error": repr(e)[:300]}))
            return None

    try:
        await asyncio.sleep(wait_s)          # let the private subscription settle
        steps.append(PlumbingStep("balance", True, {"balance": await step("balance", trader.balance())}))
        m = await pick_demo_market(trader) if market is None else \
            (await trader.get(f"/markets/{market}"))["market"]
        log("market", m)
        ticker = m["ticker"]
        bid, ask = float(m["yes_bid_dollars"]), float(m["yes_ask_dollars"])
        group = await step("create_group", trader.create_order_group(2))
        steps.append(PlumbingStep("order_group", bool(group), {"id": group}))

        # resting order, queue position, cancel
        coid = f"{prefix}-1"
        deep = round(max(0.01, bid - 0.02), 2)
        r = await step("create_resting", trader.create(order_body(
            ticker, "yes", deep, 1, client_order_id=coid, order_group_id=group,
            expiration_ts=int(time.time() + 600))))
        await asyncio.sleep(wait_s)
        resting = await trader.orders("resting", ticker)
        log("orders_resting", [asdict(o) for o in resting])
        mine = [o for o in resting if o.client_order_id == coid]
        steps.append(PlumbingStep("resting_order", bool(r and mine and mine[0].price == deep),
                                  {"rest": bool(mine), "ws_user_orders": len(ws_seen("user_order", coid))}))
        qp = await step("queue_positions", trader.queue_positions([ticker]))
        steps.append(PlumbingStep("queue_position", bool(qp and r and r["order_id"] in qp), {"qp": qp}))
        if r:
            await step("cancel", trader.cancel(r["order_id"], ticker))
            await asyncio.sleep(wait_s)
            gone = not [o for o in await trader.orders("resting", ticker) if o.client_order_id == coid]
            steps.append(PlumbingStep("cancel", gone, {"ws": [f["msg"].get("status")
                                                             for f in ws_seen("user_order", coid)]}))

        # post-only reject
        try:
            rej = await trader.create(order_body(ticker, "yes", ask, 1, client_order_id=f"{prefix}-2",
                                                 order_group_id=group))
            log("post_only_cross", rej)
            steps.append(PlumbingStep("post_only_reject", False, {"unexpected": rej}))
            await trader.cancel(rej["order_id"], ticker)
        except TradeError as e:
            log("post_only_cross", {"status": e.status, "body": e.body})
            code = (e.body.get("error") or {}).get("code") if isinstance(e.body, dict) else None
            # Only a crossing rejection counts; e.g. insufficient_balance would mask the check.
            steps.append(PlumbingStep("post_only_reject", e.status in (400, 409) and code != "insufficient_balance",
                                      {"status": e.status, "body": e.body}))

        # expiry
        coid = f"{prefix}-3"
        r = await step("create_expiring", trader.create(order_body(
            ticker, "yes", deep, 1, client_order_id=coid, order_group_id=group,
            expiration_ts=int(time.time() + 10))))
        await asyncio.sleep(15 + wait_s)
        still = [o for o in await trader.orders("resting", ticker) if o.client_order_id == coid]
        steps.append(PlumbingStep("expiry", bool(r) and not still,
                                  {"ws": [f["msg"].get("status") for f in ws_seen("user_order", coid)]}))

        # one taker fill (demo money): records the fill channel
        coid = f"{prefix}-4"
        body = order_body(ticker, "yes", ask, 1, client_order_id=coid, post_only=False)
        body["time_in_force"] = "immediate_or_cancel"
        body.pop("cancel_order_on_pause")
        r = await step("taker_fill", trader.create(body))
        await asyncio.sleep(wait_s)
        steps.append(PlumbingStep("fill", bool(r and float(r.get("fill_count") or 0) > 0),
                                  {"ws_fills": len(ws_seen("fill", coid)), "resp": r}))
        steps.append(PlumbingStep("positions", True, {"positions": await step("positions", trader.positions())}))

        # order-group trigger cancels the group's resting orders
        if group:
            ids = []
            for i in (5, 6):
                r = await step(f"create_group_{i}", trader.create(order_body(
                    ticker, "yes", deep, 1, client_order_id=f"{prefix}-{i}", order_group_id=group,
                    expiration_ts=int(time.time() + 600))))
                ids.append(r and r["order_id"])
            await step("trigger_group", trader.trigger_order_group(group))
            await asyncio.sleep(wait_s)
            left = [o for o in await trader.orders("resting", ticker) if o.order_id in ids]
            steps.append(PlumbingStep("group_trigger", all(ids) and not left,
                                      {"ws_group": [f["msg"] for f in frames
                                                    if f.get("type") == "order_group_updates"]}))
        swept = await step("cancel_own", trader.cancel_own(prefix))
        steps.append(PlumbingStep("cancel_own_sweep", swept is not None, {"cancelled": swept}))
        if group:
            await step("delete_group", trader.delete_order_group(group))
    finally:
        rec.stop.set()
        await asyncio.gather(ws_task, ws_task_groups, return_exceptions=True)
        rec.close()
        (out_dir / "ws_private.json").write_text(json.dumps(frames, indent=1))
        (out_dir / "steps.json").write_text(json.dumps([asdict(s) for s in steps], indent=1, default=str))
        rest_log.close()
    return steps
