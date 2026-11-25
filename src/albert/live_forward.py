"""Live (real-money) versions of two forward paper rules (``albert.paper_forward``).

Started 2026-11-25 at the user's request ("Let's just run the approval NO and culture maker live
please") after four days of paper trading (apr 11 fills, cmn 1 fill). The paper pass bars were not met;
the user chose to go live anyway. The paper service keeps running alongside as the control.

``cmn`` (client_order_id prefix ``albc``): the paper rule unchanged.
    At each event's scheduled close - 72 h in the 37 culture_flow series, on legs whose live book is
    really two-sided with mid <= 5c, rest a 1-contract post-only NO bid that sells YES at
    ceil(mid + 12c). Cheapest NO first, <= 3 legs per event. Each order expires at its leg's close.
    It is cancelled when the leg leaves the open listing or the event's close moves. Fills are the
    exchange's, held to settlement.
``apr`` (prefix ``alba``): the paper rule unchanged.
    12 h before each KXTRUMPAPPROVE settlement, buy 1 NO at 1 - best YES bid on every leg with
    5c < mid <= 15c, spread <= 10c and at least 1 contract bid, except legs where NO plus the fee
    costs 99c or more (no possible profit). The order is immediate-or-cancel
    with that limit price. These are the only taker orders on the account
    (``KalshiTrader(allow_taker=True)`` allows IOC/FOK only).

Safety, on top of the order layer (prod gate, write allowlist, own-orders-only cancels, per-order
caps, global KILL/HALT refusals, audit log):

* **Per-rule worst-case caps.** Resting own NO bids (price + fee) plus the cost of own fills whose
  market has not settled stay within ``cmn_cap`` / ``apr_cap`` (default $15 / $5). With the mentions
  bot's $40 cap, the bots' worst cases sum to the guard's $60 KILL line.
* **Hands off other orders.** A market where the account holds any position or any other resting
  order (manual, another bot or the other rule) is never touched.
* **Hard stops.** The global KILL, this bot's STOP file (``data/live/forward/STOP``) or the run time
  ending cancels every own order and exits.
* **Daily-loss HALT.** The guard's halt cancels own resting orders, and no new decision is taken
  until it lapses.
* **SIGTERM (a service stop or restart).** Cancels every own order. The next start reposts the
  cancelled cmn bids at the same price while their leg is open. They are post-only, so a bid that
  would now cross is rejected and dropped. A restart therefore does not change the rule.
* **Strays.** Own orders the ledger does not know (from a crashed run) are cancelled.

State is an append-only ledger ``data/live/forward/<env>/ledger.jsonl``, replayed on start, plus
``status.json`` each pass. The guard books the fills under the tags ``albc`` / ``alba``.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from albert.fees import kalshi_fee
from albert.paper_forward import (
    APR_LEAD,
    APR_SERIES,
    CMN_LEAD,
    CMN_MAX_LEGS,
    CMN_PREFILTER_ASK,
    CMN_SERIES,
    DECIDE_WINDOW,
    apr_pick,
    book_top,
    cmn_quote,
    cmn_rank,
    iso,
    listing_quote,
    no_size_at,
    parse_t,
    settle_instant,
)
from albert.venues.kalshi_trade import TradeError, order_body

PREFIX = {"cmn": "albc", "apr": "alba"}
RULES = tuple(PREFIX)
EVERY = {"cmn": timedelta(minutes=10), "apr": timedelta(minutes=5)}
CLOSE_MOVE_S = 60
MIN_LIFE = timedelta(minutes=5)     # no cmn bid is (re)posted with less than this left before its leg closes
MAX_ERRORS = 10                     # consecutive failed passes -> cancel everything and exit (systemd restarts)
EPS = 1e-9
# The one change from the paper rule: an apr leg whose NO costs 1 - bid + fee >= $0.99 (a 1c YES bid)
# can only lose or pay 0 (99c + the 1c fee for $1), so it is skipped (found in the dry smoke 2026-11-25).
MIN_UPSIDE = 0.01


@dataclass(frozen=True)
class ForwardLimits:
    cmn_cap: float = 15.0
    apr_cap: float = 5.0

    def cap(self, rule: str) -> float:
        return self.cmn_cap if rule == "cmn" else self.apr_cap


def rule_of(coid: str | None) -> str | None:
    for rule, p in PREFIX.items():
        if (coid or "").startswith(p):
            return rule
    return None


def order_fill(o: dict[str, Any]) -> tuple[float, float]:
    """(contracts filled, total cost including fees) of a ``GET /portfolio/orders/{id}`` order."""
    def f(k: str) -> float:
        v = o.get(k)
        return float(v) if v not in (None, "") else 0.0

    cost = (f("maker_fill_cost_dollars") + f("taker_fill_cost_dollars")
            + f("maker_fees_dollars") + f("taker_fees_dollars"))
    return f("fill_count_fp"), round(cost, 6)


def apr_body(ticker: str, price: float, coid: str, exchange_index: int = -1) -> dict[str, Any]:
    """1 NO at limit ``price``, immediate-or-cancel (never rests)."""
    body = order_body(ticker, "no", price, 1, client_order_id=coid, post_only=False, exchange_index=exchange_index)
    body["time_in_force"] = "immediate_or_cancel"
    body.pop("cancel_order_on_pause", None)       # as in the demo plumbing taker test (live_l2)
    return body


class ForwardBot:
    def __init__(self, env: str, traders: dict[str, Any], api: Any, root: Path, *,
                 limits: ForwardLimits | None = None, rules: tuple[str, ...] = RULES, switches=None,
                 stop_file: Path | None = None, clock: Callable[[], datetime] | None = None,
                 log: Callable[[str], None] = print) -> None:
        if not rules or any(r not in PREFIX for r in rules):
            raise ValueError(f"rules must be a non-empty subset of {RULES}")
        self.env, self.traders, self.api = env, traders, api
        self.rules, self.limits, self.switches = tuple(rules), limits or ForwardLimits(), switches
        self.reader = traders[self.rules[0]]
        self.dir = root / env
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "ledger.jsonl"
        self.stop_file = stop_file or root / "STOP"
        self.clock = clock or (lambda: datetime.now(UTC))
        self.say = log
        self.run_id = self.clock().strftime("%Y%m%dT%H%M%S")
        self._n = 0
        self.decided: set[str] = set()
        self.orders: dict[str, dict[str, Any]] = {}     # order_id -> order record + status
        self.fills: dict[str, dict[str, Any]] = {}      # order_id -> fill record
        self.settled: dict[str, str] = {}               # ticker -> result
        self.due: dict[str, datetime] = {}
        self.halted_by: str | None = None
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    try:
                        self._apply(json.loads(line))
                    except ValueError:
                        continue                         # a torn last line after a crash

    # -- ledger --
    def write(self, kind: str, **rec: Any) -> dict[str, Any]:
        r = {"kind": kind, "ts": iso(self.clock()), **rec}
        with self.path.open("a") as fh:
            fh.write(json.dumps(r, default=str) + "\n")
        self._apply(r)
        return r

    def _apply(self, r: dict[str, Any]) -> None:
        k, oid = r["kind"], r.get("order_id")
        if k == "decision":
            self.decided.add(f"{r['rule']}:{r['key']}")
        elif k == "order" and oid:
            self.orders[oid] = {**r, "status": "resting"}
        elif k == "fill" and oid in self.orders:
            self.fills[oid] = r
            self.orders[oid]["status"] = "filled"
        elif k in ("end", "park") and oid in self.orders and self.orders[oid]["status"] != "filled":
            self.orders[oid]["status"] = "ended" if k == "end" else "parked"
        elif k == "settle":
            self.settled[r["ticker"]] = r["result"]

    def _coid(self, rule: str) -> str:
        self._n += 1
        return f"{PREFIX[rule]}{self.run_id}-{self._n}"

    # -- accounting --
    def resting(self, rule: str | None = None) -> list[dict[str, Any]]:
        return [o for o in self.orders.values() if o["status"] == "resting" and rule in (None, o["rule"])]

    def used(self, rule: str) -> float:
        """Worst case: resting NO bids (price + fee) plus own fills not settled yet."""
        rest = sum(o["price"] + o.get("fee", 0.0) for o in self.resting(rule))
        pos = sum(f["cost"] for f in self.fills.values() if f["rule"] == rule and f["ticker"] not in self.settled)
        return round(rest + pos, 6)

    async def _pub(self, fn: Callable[..., Any], *args: Any) -> Any:
        return await asyncio.to_thread(fn, *args)

    async def blocked(self) -> set[str]:
        """Markets with any position or any resting order (every source)."""
        resting = await self.reader.orders("resting")
        pos = await self.reader.positions()
        return {o.ticker for o in resting} | {t for t, q in pos.items() if abs(q) > EPS}

    # -- exchange sync --
    async def _resolve(self, oid: str) -> str:
        """Settle an own order that left the book: logs its fill. Returns filled | ended | resting."""
        o = self.orders[oid]
        try:
            d = (await self.reader.get(f"/portfolio/orders/{oid}")).get("order") or {}
        except TradeError as err:
            if err.status != 404:
                raise
            return "ended"
        if d.get("status") == "resting":
            return "resting"
        qty, cost = order_fill(d)
        if qty <= EPS:
            o["exchange_status"] = d.get("status")
            return "ended"
        self.write("fill", rule=o["rule"], order_id=oid, event=o["event"], ticker=o["ticker"], side=o["side"],
                   price=o["price"], qty=qty, cost=cost, close_time=o.get("close_time"),
                   taker=float(d.get("taker_fill_cost_dollars") or 0) > 0,
                   exchange_ts=d.get("last_update_time"))
        self.say(f"{o['rule']} fill {o['ticker']} NO @{o['price']:.2f} (cost ${cost:.2f})")
        return "filled"

    async def sync(self) -> None:
        """Reconcile own orders with the exchange: strays are cancelled, gone orders resolved."""
        resting = await self.reader.orders("resting")
        on_book = {o.order_id for o in resting}
        for o in resting:
            rule = rule_of(o.client_order_id)
            if rule and o.order_id not in self.orders:
                try:
                    await self.traders.get(rule, self.reader).cancel(o.order_id, o.ticker)
                except TradeError as err:
                    if err.status != 404:
                        raise
                self.write("stray_cancel", order_id_stray=o.order_id, coid=o.client_order_id, ticker=o.ticker)
        for oid, o in list(self.orders.items()):
            if o["status"] == "resting" and oid not in on_book:
                st = await self._resolve(oid)
                if st == "ended":
                    self.write("end", order_id=oid, reason=o.get("exchange_status") or "left the book")

    async def _cancel(self, oid: str, reason: str, *, park: bool = False) -> None:
        o = self.orders[oid]
        try:
            await self.traders.get(o["rule"], self.reader).cancel(oid, o["ticker"])
        except TradeError as err:
            if err.status != 404:
                raise
        st = await self._resolve(oid)
        if st != "filled":
            self.write("park" if park else "end", order_id=oid, reason=reason)

    async def settle(self, now: datetime) -> None:
        todo = sorted({f["ticker"] for f in self.fills.values() if f["ticker"] not in self.settled
                       and (parse_t(f.get("close_time")) or now) <= now})
        if not todo:
            return
        for m in await self._pub(self.api.markets, todo):
            res = m.get("result")
            if res not in ("yes", "no") or m.get("ticker") in self.settled:
                continue
            mine = [f for f in self.fills.values() if f["ticker"] == m["ticker"]]
            pnl = sum(f["qty"] * (1.0 if res == f["side"] else 0.0) - f["cost"] for f in mine)
            self.write("settle", ticker=m["ticker"], result=res, pnl=round(pnl, 4),
                       rules=sorted({f["rule"] for f in mine}))
            self.say(f"settled {m['ticker']} {res}: {pnl:+.2f}")

    # -- cmn --
    async def cmn_scan(self, now: datetime) -> None:
        open_by_ticker: dict[str, dict[str, Any]] = {}
        events: dict[str, list[dict[str, Any]]] = {}
        listed: set[str] = set()
        for s in CMN_SERIES:
            try:
                ms = await self._pub(self.api.open_markets, s)
            except Exception as err:  # noqa: BLE001 - one series failing must not stop the scan
                self.write("error", what=f"cmn list {s}", error=repr(err)[:200])
                continue
            listed.add(s)
            for m in ms:
                open_by_ticker[m["ticker"]] = m
                events.setdefault(m["event_ticker"], []).append(m)
        for o in self.resting("cmn"):
            if o["series"] not in listed:
                continue
            ev = events.get(o["event"])
            reason = None
            if o["ticker"] not in open_by_ticker:
                reason = "left open listing"
            elif ev and abs((max(parse_t(x["close_time"]) for x in ev) - parse_t(o["ev_close"])).total_seconds()) > CLOSE_MOVE_S:
                reason = "event close moved"
            if reason:
                await self._cancel(o["order_id"], reason)
        due = []
        for ev_t, legs in sorted(events.items()):
            if f"cmn:{ev_t}" in self.decided:
                continue
            close = max(parse_t(m["close_time"]) for m in legs)
            if close - CMN_LEAD <= now < close - CMN_LEAD + DECIDE_WINDOW:
                due.append((ev_t, legs, close))
        if due:
            blocked = await self.blocked()
            for ev_t, legs, close in due:
                await self._cmn_decide(ev_t, legs, close, now, blocked)

    async def _cmn_decide(self, ev_t: str, legs: list[dict[str, Any]], close: datetime, now: datetime,
                          blocked: set[str]) -> None:
        series = ev_t.split("-")[0]
        fee_type, mult = await self._pub(self.api.fee, series)
        seen, cands = [], []
        for m in legs:
            q = listing_quote(m)
            row: dict[str, Any] = {"ticker": m["ticker"], "listing": q}
            if q["yes_ask"] is not None and q["yes_ask"] <= CMN_PREFILTER_ASK:
                ob = await self._pub(self.api.book, m["ticker"])
                top = book_top(ob)
                row["book"] = top
                quote = cmn_quote(top)
                if quote is not None:
                    quote["ahead"] = no_size_at(ob, quote["price"])
                    cands.append({"ticker": m["ticker"], "close_time": m["close_time"], **quote, "top": top})
                    row["quote"] = quote
            seen.append(row)
        placed, skipped = [], []
        shards = await self._shards([c["ticker"] for c in cands])
        for c in cmn_rank(cands):
            fee = kalshi_fee(1, c["price"], maker=True, fee_type=fee_type, multiplier=mult)
            why = None
            if len(placed) >= CMN_MAX_LEGS:
                why = "max legs"
            elif c["ticker"] in blocked:
                why = "market has a position or another order"
            elif parse_t(c["close_time"]) - now < MIN_LIFE:
                why = "leg closing"
            elif self.used("cmn") + c["price"] + fee > self.limits.cmn_cap + EPS:
                why = f"cap ${self.limits.cmn_cap:.2f}"
            if why:
                skipped.append({"ticker": c["ticker"], "price": c["price"], "why": why})
                continue
            oid = await self._post_cmn(ticker=c["ticker"], event=ev_t, series=series, price=c["price"], fee=fee,
                                       fee_type=fee_type, ev_close=iso(close), close_time=c["close_time"],
                                       shard=shards.get(c["ticker"], -1),
                                       extra={"mid": c["mid"], "sell_at": c["sell_at"], "ahead": c["ahead"],
                                              "top": c["top"]})
            if oid:
                placed.append(oid)
                blocked.add(c["ticker"])
        self.write("decision", rule="cmn", key=ev_t, series=series, ev_close=iso(close), legs=len(legs),
                   candidates=len(cands), placed=placed, skipped=skipped, seen=seen)
        if cands:
            self.say(f"cmn {ev_t}: {len(cands)} qualifying legs, {len(placed)} NO bids posted")

    async def _post_cmn(self, *, ticker: str, event: str, series: str, price: float, fee: float,
                        fee_type: str | None, ev_close: str, close_time: str, shard: int = -1,
                        extra: dict[str, Any] | None = None) -> str | None:
        coid = self._coid("cmn")
        exp = math.floor(parse_t(close_time).timestamp())
        body = order_body(ticker, "no", price, 1, client_order_id=coid, expiration_ts=exp, exchange_index=shard)
        try:
            resp = await self.traders["cmn"].create(body)
        except TradeError as err:
            self.write("reject", rule="cmn", event=event, ticker=ticker, price=price, error=str(err)[:300])
            return None
        oid = resp.get("order_id")
        if not oid:
            self.write("reject", rule="cmn", event=event, ticker=ticker, price=price, error=f"no order_id: {resp}")
            return None
        self.write("order", rule="cmn", order_id=oid, coid=coid, event=event, series=series, ticker=ticker,
                   side="no", price=price, fee=fee, fee_type=fee_type, ev_close=ev_close, close_time=close_time,
                   expiration=exp, **(extra or {}))
        return oid

    async def _shards(self, tickers: list[str]) -> dict[str, int]:
        if not tickers:
            return {}
        try:
            return await self.reader.shards(tickers)
        except TradeError:
            return {}                     # auto-route (-1)

    # -- apr --
    async def apr_scan(self, now: datetime) -> None:
        events: dict[str, list[dict[str, Any]]] = {}
        for m in await self._pub(self.api.open_markets, APR_SERIES):
            events.setdefault(m["event_ticker"], []).append(m)
        for ev_t, legs in sorted(events.items()):
            if f"apr:{ev_t}" in self.decided:
                continue
            settle = settle_instant(max(parse_t(m["close_time"]) for m in legs))
            if not settle - APR_LEAD <= now < settle - APR_LEAD + DECIDE_WINDOW:
                continue
            _, mult = await self._pub(self.api.fee, APR_SERIES)
            blocked = await self.blocked()
            shards = await self._shards([m["ticker"] for m in legs])
            seen, filled = [], 0
            for m in sorted(legs, key=lambda m: m["ticker"]):
                top = book_top(await self._pub(self.api.book, m["ticker"]))
                pick = apr_pick(top, mult)
                row: dict[str, Any] = {"ticker": m["ticker"], "top": top, "pick": pick is not None}
                if pick is not None:
                    if 1 - pick["price"] - pick["fee"] < MIN_UPSIDE - EPS:
                        row["skip"] = "no upside after fee"
                    elif m["ticker"] in blocked:
                        row["skip"] = "market has a position or another order"
                    elif self.used("apr") + pick["price"] + pick["fee"] > self.limits.apr_cap + EPS:
                        row["skip"] = f"cap ${self.limits.apr_cap:.2f}"
                    else:
                        row["result"] = await self._take_apr(ev_t, m, pick, top, shards.get(m["ticker"], -1))
                        filled += row["result"] == "filled"
                seen.append(row)
            self.write("decision", rule="apr", key=ev_t, settle=iso(settle), legs=len(legs), filled=filled, seen=seen)
            self.say(f"apr {ev_t}: {filled} NO bought")

    async def _take_apr(self, event: str, m: dict[str, Any], pick: dict[str, Any], top: dict[str, Any],
                        shard: int) -> str:
        coid = self._coid("apr")
        try:
            resp = await self.traders["apr"].create(apr_body(m["ticker"], pick["price"], coid, shard))
        except TradeError as err:
            self.write("reject", rule="apr", event=event, ticker=m["ticker"], price=pick["price"], error=str(err)[:300])
            return "rejected"
        oid = resp.get("order_id")
        if not oid:
            self.write("reject", rule="apr", event=event, ticker=m["ticker"], price=pick["price"],
                       error=f"no order_id: {resp}")
            return "rejected"
        self.write("order", rule="apr", order_id=oid, coid=coid, event=event, series=APR_SERIES, ticker=m["ticker"],
                   side="no", price=pick["price"], fee=pick["fee"], mid=pick["mid"], top=top,
                   close_time=m["close_time"], response=resp)
        try:
            st = await self._resolve(oid)
        except TradeError as err:         # resolved on the next pass (the order is not on the book)
            self.write("error", what=f"resolve {oid}", error=str(err)[:200])
            return "unknown"
        if st == "ended":
            self.write("end", order_id=oid, reason="ioc not filled")
        return st

    # -- lifecycle --
    async def prepare(self) -> None:
        """Resolve what happened while down, cancel strays, then repost the cmn bids a SIGTERM parked."""
        await self.sync()
        now = self.clock()
        parked = [o for o in self.orders.values() if o["status"] == "parked" and o["rule"] in self.rules]
        if not parked:
            return
        blocked = await self.blocked()
        for o in parked:
            why = None
            if parse_t(o["close_time"]) - now < MIN_LIFE:
                why = "leg closing"
            elif o["ticker"] in blocked:
                why = "market has a position or another order"
            elif self.used("cmn") + o["price"] + o.get("fee", 0.0) > self.limits.cmn_cap + EPS:
                why = "cap"
            new = None
            if why is None:
                extra = {k: o[k] for k in ("mid", "sell_at", "ahead", "top") if k in o}
                new = await self._post_cmn(ticker=o["ticker"], event=o["event"], series=o["series"], price=o["price"],
                                           fee=o.get("fee", 0.0), fee_type=o.get("fee_type"), ev_close=o["ev_close"],
                                           close_time=o["close_time"], extra={**extra, "reposts": o["order_id"]})
            self.write("end", order_id=o["order_id"], reason=f"reposted as {new}" if new else f"not reposted: {why or 'rejected'}")

    async def shutdown(self, reason: str, *, park: bool = False) -> None:
        """Cancel every own order. ``park``: cmn bids are reposted at the next start."""
        cancelled: list[str] = []
        for rule in self.rules:
            try:
                cancelled += await self.traders[rule].cancel_own(PREFIX[rule])
            except Exception as err:  # noqa: BLE001 - keep going: the guard sweeps on KILL as a backstop
                self.write("error", what=f"cancel_own {rule}", error=repr(err)[:200])
        for o in self.resting():
            try:
                st = await self._resolve(o["order_id"])
            except Exception as err:  # noqa: BLE001
                self.write("error", what=f"resolve {o['order_id']}", error=repr(err)[:200])
                continue
            if st != "filled":
                kind = "park" if park and o["rule"] == "cmn" and st == "ended" else "end"
                self.write(kind, order_id=o["order_id"], reason=f"shutdown: {reason}")
        self.write("shutdown", reason=reason, cancelled=cancelled, park=park)
        self.say(f"live forward stopped: {reason} ({len(cancelled)} orders cancelled)")

    async def tick(self) -> bool:
        """One pass; False when the bot should stop."""
        now = self.clock()
        stop = "stop file" if self.stop_file.exists() else None
        if stop is None and self.switches is not None and (k := self.switches.killed()):
            stop = f"global kill: {k}"
        if stop:
            await self.shutdown(stop)
            return False
        await self.sync()
        await self.settle(now)
        halt = self.switches.halted(now) if self.switches is not None else None
        if halt:
            if halt != self.halted_by:
                self.halted_by = halt
                self.write("halt", reason=halt)
            for o in self.resting():
                await self._cancel(o["order_id"], f"halt: {halt}")
            self.status(now)
            return True
        if self.halted_by:
            self.write("halt_lifted", reason=self.halted_by)
            self.halted_by = None
        for rule in self.rules:
            if now < self.due.get(rule, now):
                continue
            try:
                await (self.cmn_scan(now) if rule == "cmn" else self.apr_scan(now))
            except Exception as err:  # noqa: BLE001 - one rule failing must not stop the other
                self.write("error", what=f"{rule} scan", error=repr(err)[:300])
                self.say(f"{rule} scan failed: {err!r}"[:300])
            self.due[rule] = now + EVERY[rule]
        self.status(now)
        return True

    def status(self, now: datetime) -> dict[str, Any]:
        st: dict[str, Any] = {"ts": iso(now), "env": self.env, "rules": list(self.rules), "limits": asdict(self.limits),
                              "halted": self.halted_by}
        for rule in self.rules:
            fills = [f for f in self.fills.values() if f["rule"] == rule]
            st[rule] = {"used": self.used(rule), "cap": self.limits.cap(rule), "resting": len(self.resting(rule)),
                        "fills": len(fills), "open_fills": sum(f["ticker"] not in self.settled for f in fills),
                        "fill_cost": round(sum(f["cost"] for f in fills), 4)}
        tmp = self.dir / ".status.json.tmp"
        tmp.write_text(json.dumps(st, default=str, indent=1) + "\n")
        tmp.replace(self.dir / "status.json")
        return st

    async def run(self, minutes: float, poll_s: float = 60.0) -> None:
        self.write("start", env=self.env, rules=list(self.rules), limits=asdict(self.limits), pid=os.getpid())
        stop_at = self.clock() + timedelta(minutes=minutes)
        errors = 0
        try:
            await self.prepare()
            while self.clock() < stop_at:
                try:
                    if not await self.tick():
                        return
                    errors = 0
                except asyncio.CancelledError:
                    raise
                except Exception as err:
                    errors += 1
                    self.write("error", what="tick", error=repr(err)[:500], consecutive=errors)
                    self.say(f"live forward error: {err!r}"[:300])
                    if errors >= MAX_ERRORS:
                        await self.shutdown(f"{errors} consecutive errors", park=True)
                        raise
                await asyncio.sleep(poll_s)
            await self.shutdown("run time over")
        except asyncio.CancelledError:
            await self.shutdown("signal", park=True)
            raise


class DryTrader:
    """Real account reads, simulated writes: creates and cancels are logged, never sent."""

    def __init__(self, reader: Any) -> None:
        self.r = reader
        self.writable = True
        self.sent: list[dict[str, Any]] = []

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if path.startswith("/portfolio/orders/dry-"):
            raise TradeError(404, "dry run", path)
        return await self.r.get(path, params)

    async def orders(self, status: str = "resting", ticker: str | None = None):
        return await self.r.orders(status, ticker)

    async def positions(self) -> dict[str, float]:
        return await self.r.positions()

    async def shards(self, tickers: list[str]) -> dict[str, int]:
        return await self.r.shards(tickers)

    async def create(self, body: dict[str, Any]) -> dict[str, Any]:
        self.sent.append(body)
        return {"order_id": f"dry-{len(self.sent)}", "client_order_id": body["client_order_id"], "dry_run": True}

    async def cancel(self, order_id: str, ticker: str) -> dict[str, Any]:
        return {}

    async def cancel_own(self, prefix: str) -> list[str]:
        return []

    async def aclose(self) -> None:
        await self.r.aclose()
