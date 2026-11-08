"""Live pre-speech NO-bid maker on Kalshi mention markets (demo, or prod behind the live gate).

The rule is the one frozen in ``research/mentions_maker`` and confirmed out of sample on 548 unseen
events by ``research/mentions_history`` (and recorded hourly by ``research/forward``): for a speech
event in the Mentions category (window series and earnings calls excluded) whose milestone start is
about 6 h away, rest ONE post-only NO bid per quoted word leg at ``floor(1 − fair − margin)`` with
``fair = σ(−0.1156 + 1.0474·logit(mid))`` (capped one tick inside the YES bid), until start − 10
min, and hold any fill to settlement. Events whose share of "already said" legs (bid ≥ 97c) jumped
more than 12 points since start − 24 h are skipped (the study's outcome-blind exclusion).

Safety, on top of the order layer's own (``venues/kalshi_trade``: write allowlist, own-orders-only
cancels, prod gate ``KALSHI_LIVE_ENABLED=1`` + ``--live``):

* every order is 1 contract, carries the ``albm`` client_order_id prefix (under the bot prefix
  ``alb``), sits in a per-event order group (exchange-side cap on contracts matched per 15 s) and
  has ``expiration_time`` = start − 10 min, so the exchange cancels it even if this process dies;
* markets where the account holds a position or a non-bot order are never touched;
* collateral cap: resting NO bids plus today's fill cost never exceed ``max_collateral``; no new
  events once today's fill cost exceeds ``max_daily_loss`` (worst case: every fill loses);
* circuit breaker: if a leg asked ≤ 80c at posting trades or bids at ≥ 95c (a word was said, so
  the speech has started or leaked), the event's order group is triggered (all its orders cancel);
* if the milestone start moves earlier than 10 min from now, the event's orders are cancelled;
* kill switch file and SIGTERM cancel every ``albm`` order and stop.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from albert.venues.kalshi_trade import TradeError, order_body, parse_fill

PREFIX = "albm"

EXCLUDED_GROUPS = ("earnings",)
DUE = (timedelta(hours=5), timedelta(hours=6))      # post when start − now is in (5 h, 6 h]
END_BEFORE = timedelta(minutes=10)
BREAKER_PRICE, CHEAP_ASK = 0.95, 0.80


@dataclass
class MentionLimits:
    margin: float = 0.12
    max_collateral: float = 20.0      # $: resting NO bid prices + today's fill cost
    max_daily_loss: float = 10.0      # $: today's fill cost (worst case) before new events stop
    max_events: int = 3               # events with resting orders at once
    max_orders_per_event: int = 40
    group_limit: int = 5              # exchange-side: contracts matched per rolling 15 s


@dataclass
class EventState:
    event: str
    start: datetime
    end: datetime
    group: str | None
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)   # order_id -> {ticker, price, cheap}
    closed: bool = False
    breaker: bool = False


def _timing(t_send_ns: int, resp: Any) -> dict[str, float]:
    """REST round trip and send -> matching-engine ``ts_ms`` (ms) for one order request."""
    out = {"rest_ms": round((time.time_ns() - t_send_ns) / 1e6, 3)}
    ts = resp.get("ts_ms") if isinstance(resp, dict) else None
    if ts:
        out["send_to_engine_ms"] = round(int(ts) - t_send_ns / 1e6, 3)
    return out


def plan_event(s, base: str, event: str, start: datetime, now: datetime, margin: float, *,
               transport=None) -> pl.DataFrame:
    """The frozen rule's NO bids for one event, priced from the env's public order books (the same
    helpers the forward test records with). Columns: ticker, word, bid, ask, price, cheap, eligible."""
    from albert.research import forward as F
    from albert.research import mentions as M
    from albert.venues import kalshi as k

    c = F.client(s, "kalshi", base, record=False, transport=transport)
    mk, _ = F._markets(c, {"event_ticker": event})
    mk = mk.filter(pl.col("status").is_in(F._strs(["active", "open"])))
    mk = mk.with_columns(pl.col("yes_sub_title").map_elements(lambda w: M.parse_word(w)[0], return_dtype=pl.String)
                         .alias("word"))
    mk = mk.filter((pl.col("word").str.len_chars() > 0) & ~pl.col("word").str.starts_with("event does not qualify"))
    if not mk.height:
        return pl.DataFrame()
    tickers = mk["ticker"].to_list()
    rows = F.with_quote(mk.join(F.fetch_books(c, tickers), on="ticker", how="left"))
    at24 = start - timedelta(hours=24)
    said24 = 0.0
    if at24 < now:
        said24 = F.said24_from_candles(k.Kalshi(c).candles_batch(tickers, at24 - timedelta(hours=3), at24, 60).payload, at24)
    said = F.said_share(rows)
    excluded = said > said24 + F.SAID_JUMP + 1e-12
    rows = rows.with_columns(pl.lit(excluded).alias("excluded"), (pl.col("quoted") & pl.lit(not excluded)).alias("eligible"))
    rows = F.mention_orders(rows)
    tag = round(margin * 100)
    col = f"o_cal{tag}_no"
    if col not in rows.columns:          # a margin outside the frozen pair: same formula
        fair = rows["fair_cal"].to_numpy()
        no = F.bid_prices(fair, rows["bid"].to_numpy(), rows["ask"].to_numpy(), margin)[1]
        rows = rows.with_columns(F._col(no, col, rows["eligible"].to_numpy()))
    return rows.select("ticker", "word", "bid", "ask", "fair_cal", "eligible",
                       pl.col(col).alias("price"), (pl.col("ask") <= CHEAP_ASK).alias("cheap"))


class MentionsBot:
    def __init__(self, s, env: str, trader, run_dir: Path, *, limits: MentionLimits, base: str,
                 stop_file: Path, transport=None, clock: Callable[[], datetime] | None = None,
                 log: Callable[[str], None] = print,
                 series_loader: Callable[[], dict[str, dict]] | None = None) -> None:
        self.s, self.env, self.trader, self.run_dir = s, env, trader, run_dir
        self.limits, self.base, self.stop_file, self.transport = limits, base, stop_file, transport
        self.clock = clock or (lambda: datetime.now(UTC))
        self.log = log
        self.run_id = run_dir.name.replace("-", "")[:16]
        self.events: dict[str, EventState] = {}
        self.done: set[str] = set()
        self.blocked: set[str] = set()
        self.fill_ids: set[str] = set()
        self.fill_cost_today: dict[str, float] = {}
        self.halted: str | None = None
        self._n = 0
        # Finance filter (albert.categories): the series list is reloaded once a day; an event whose
        # series is not in it yet is skipped, so a new finance series cannot slip through untitled.
        self.series_loader = series_loader
        self.series_info: dict[str, dict] | None = None
        self.series_day: str | None = None
        self.skipped_financial: set[str] = set()
        run_dir.mkdir(parents=True, exist_ok=True)

    # -- ledger --
    def write(self, kind: str, data: dict[str, Any]) -> None:
        rec = {"ts": self.clock().isoformat(), "kind": kind, **data}
        with (self.run_dir / "ledger.jsonl").open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    # -- accounting --
    def resting_collateral(self) -> float:
        return sum(o["price"] for e in self.events.values() if not e.closed for o in e.orders.values()
                   if o.get("status", "resting") == "resting")

    def spent_today(self) -> float:
        return self.fill_cost_today.get(self.clock().strftime("%Y-%m-%d"), 0.0)

    # -- lifecycle --
    async def prepare(self) -> None:
        """Sweep stale bot orders, then block every market with a position or a non-bot order."""
        swept = await self.trader.cancel_own(PREFIX)
        resting = await self.trader.orders("resting")
        pos = await self.trader.positions()
        manual = {o.ticker for o in resting if not o.client_order_id.startswith(PREFIX)}
        self.blocked = manual | {t for t, q in pos.items() if abs(q) > 1e-9}
        self.write("prepare", {"swept": swept, "blocked": sorted(self.blocked)})

    async def shutdown(self, reason: str) -> None:
        cancelled = await self.trader.cancel_own(PREFIX)
        for e in self.events.values():
            if e.group and not e.closed:
                try:
                    await self.trader.delete_order_group(e.group)
                except TradeError as err:
                    self.write("error", {"what": "delete group", "event": e.event, "error": str(err)})
            e.closed = True
        self.write("shutdown", {"reason": reason, "cancelled": cancelled})
        self.log(f"mentions bot stopped: {reason} ({len(cancelled)} orders cancelled)")

    def halt(self, reason: str) -> None:
        if not self.halted:
            self.halted = reason
            self.write("halt", {"reason": reason})

    # -- one pass --
    async def tick(self, *, force_events: list[str] | None = None) -> bool:
        """One pass; False when the bot should stop."""
        now = self.clock()
        if self.stop_file.exists():
            self.halt("kill file")
        if self.halted:
            await self.shutdown(self.halted)
            return False
        if self.series_loader is not None and self.series_day != now.strftime("%Y-%m-%d"):
            try:
                self.series_info = await asyncio.to_thread(self.series_loader)
                self.series_day = now.strftime("%Y-%m-%d")
            except Exception as e:  # noqa: BLE001 - keep yesterday's list (or skip every event)
                self.write("error", {"what": "series list", "error": repr(e)[:200]})
        starts = await self._milestones(now)
        await self._fills()
        await self._watch(now, starts)
        if self.spent_today() >= self.limits.max_daily_loss:
            self.write("no_new_events", {"spent_today": self.spent_today()})
            return True
        due = self._due(now, starts, force_events)
        for ev, start in due:
            if sum(not e.closed for e in self.events.values()) >= self.limits.max_events:
                break
            await self._post_event(ev, start, now, forced=bool(force_events and ev in force_events))
        return True

    async def _milestones(self, now: datetime) -> dict[str, datetime]:
        from albert.research import forward as F

        page = await self.trader.get("/milestones", {"minimum_start_date": (now - timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"), "category": "Mentions", "limit": 500})
        due = F.mention_due(page, now - timedelta(hours=1), timedelta(0), timedelta(days=2), set())
        return {r["event"]: r["start"] for r in due.iter_rows(named=True)}

    def is_financial(self, series: str) -> bool:
        """Finance-related series (albert.categories) are never traded. Without a series list
        (no loader) only the ticker and Mentions-name rules apply; with one, unknown series are
        skipped."""
        from albert.categories import kalshi_series_is_financial

        if self.series_info is None:
            return self.series_loader is not None or kalshi_series_is_financial(series, "Mentions", None)
        info = self.series_info.get(series)
        if info is None:
            return True
        return kalshi_series_is_financial(series, info.get("category"), info.get("title"))

    def _due(self, now: datetime, starts: dict[str, datetime], force: list[str] | None) -> list[tuple[str, datetime]]:
        from albert.research import forward as F

        if force:
            return [(ev, now + timedelta(minutes=25)) for ev in force if ev not in self.done]
        out = []
        for ev, start in sorted(starts.items(), key=lambda kv: kv[1]):
            if ev in self.done or F.speech_group(ev.split("-")[0]) in EXCLUDED_GROUPS:
                continue
            if self.is_financial(ev.split("-")[0]):
                if ev not in self.skipped_financial:
                    self.skipped_financial.add(ev)
                    self.write("skip_financial", {"event": ev})
                continue
            if DUE[0] < start - now <= DUE[1]:
                out.append((ev, start))
        return out

    async def _post_event(self, ev: str, start: datetime, now: datetime, *, forced: bool) -> None:
        self.done.add(ev)
        try:
            plan = await asyncio.to_thread(plan_event, self.s, self.base, ev, start, now, self.limits.margin,
                                           transport=self.transport)
        except Exception as err:  # noqa: BLE001 - a bad listing must not stop the bot
            self.write("error", {"what": "plan", "event": ev, "error": repr(err)[:300]})
            return
        if not plan.height:
            self.write("skip", {"event": ev, "reason": "no legs"})
            return
        from albert.research import forward as F

        legs = plan.filter(pl.col("eligible") & pl.col("price").is_not_null()
                           & ~pl.col("ticker").is_in(F._strs(self.blocked)))
        legs = legs.sort("price").head(self.limits.max_orders_per_event)
        room = self.limits.max_collateral - self.resting_collateral() - self.spent_today()
        keep, total = [], 0.0
        for r in legs.iter_rows(named=True):
            if total + r["price"] > room + 1e-9:
                break
            keep.append(r)
            total += r["price"]
        end = start - END_BEFORE
        self.write("plan", {"event": ev, "start": start, "end": end, "forced": forced, "legs": plan.height,
                            "eligible": legs.height, "posting": len(keep), "collateral": round(total, 4),
                            "rows": plan.to_dicts()})
        if not keep:
            return
        group = await self.trader.create_order_group(self.limits.group_limit)
        st = EventState(ev, start, end, group)
        self.events[ev] = st
        exp = math.floor(end.timestamp())
        try:
            shard = await self.trader.shards([r["ticker"] for r in keep])
        except TradeError:
            shard = {}                     # auto-route (-1) instead
        for r in keep:
            self._n += 1
            coid = f"{PREFIX}{self.run_id}-{self._n}"
            body = order_body(r["ticker"], "no", r["price"], 1, client_order_id=coid, order_group_id=group,
                              expiration_ts=exp, exchange_index=shard.get(r["ticker"], -1))
            t_send = time.time_ns()
            try:
                resp = await self.trader.create(body)
            except TradeError as err:
                self.write("reject", {"event": ev, "ticker": r["ticker"], "price": r["price"], "error": str(err)})
                continue
            oid = resp.get("order_id")
            st.orders[oid] = {"ticker": r["ticker"], "price": r["price"], "cheap": bool(r["cheap"]),
                              "coid": coid, "status": "resting"}
            self.write("order", {"event": ev, "order_id": oid, "client_order_id": coid, "ticker": r["ticker"],
                                 "side": "no", "price": r["price"], "fair": r["fair_cal"], "bid": r["bid"],
                                 "ask": r["ask"], "expiration": exp, **_timing(t_send, resp)})
        self.log(f"{ev}: {len(st.orders)} NO bids, ${self.resting_collateral():.2f} resting")

    async def _watch(self, now: datetime, starts: dict[str, datetime]) -> None:
        for e in self.events.values():
            if e.closed:
                continue
            moved = starts.get(e.event)
            reason = None
            if now >= e.end:
                reason = "end"
            elif moved is not None and moved - END_BEFORE <= now:
                reason = f"start moved to {moved.isoformat()}"
            elif not e.breaker and await self._breaker(e):
                reason = "circuit breaker"
                e.breaker = True
            if reason:
                await self._close(e, reason)

    async def _breaker(self, e: EventState) -> bool:
        cheap = {o["ticker"] for o in e.orders.values() if o["cheap"]}
        if not cheap:
            return False
        page = await self.trader.get("/markets", {"event_ticker": e.event, "limit": 200})
        for m in page.get("markets") or []:
            if m.get("ticker") not in cheap:
                continue
            hi = max(float(m.get("yes_bid_dollars") or 0), float(m.get("last_price_dollars") or 0))
            if hi >= BREAKER_PRICE - 1e-9:
                self.write("breaker", {"event": e.event, "ticker": m.get("ticker"), "price": hi})
                return True
        return False

    async def _close(self, e: EventState, reason: str) -> None:
        if e.group:
            try:
                await self.trader.trigger_order_group(e.group)
            except TradeError as err:
                self.write("error", {"what": "trigger", "event": e.event, "error": str(err)})
        resting = await self.trader.orders("resting")
        for o in resting:
            if o.order_id in e.orders and o.client_order_id.startswith(PREFIX):
                t_send = time.time_ns()
                try:
                    resp = await self.trader.cancel(o.order_id, o.ticker)
                    self.write("cancel", {"event": e.event, "order_id": o.order_id, **_timing(t_send, resp)})
                except TradeError as err:
                    self.write("error", {"what": "cancel", "order_id": o.order_id, "error": str(err)})
        if e.group:
            try:
                await self.trader.delete_order_group(e.group)
            except TradeError as err:
                self.write("error", {"what": "delete group", "event": e.event, "error": str(err)})
        e.closed = True
        for o in e.orders.values():
            if o.get("status") == "resting":
                o["status"] = "closed"
        self.write("close", {"event": e.event, "reason": reason})

    async def _fills(self) -> None:
        page = await self.trader.get("/portfolio/fills", {"limit": 200})
        ours = {oid: (e, o) for e in self.events.values() for oid, o in e.orders.items()}
        for raw in page.get("fills") or []:
            fr = parse_fill(raw)
            if fr.trade_id in self.fill_ids or fr.order_id not in ours:
                continue
            self.fill_ids.add(fr.trade_id)
            e, o = ours[fr.order_id]
            if fr.is_taker:
                self.halt(f"taker fill on post-only order {fr.order_id}")
            cost = fr.qty * o["price"] + fr.fee      # a NO bid's fill costs its price; worst case lost
            day = self.clock().strftime("%Y-%m-%d")
            self.fill_cost_today[day] = self.fill_cost_today.get(day, 0.0) + cost
            o["status"] = "filled"
            self.write("fill", {"event": e.event, **asdict(fr), "cost": round(cost, 4)})
            self.log(f"fill {fr.ticker} NO @{o['price']:.2f}")

    async def run(self, minutes: float, poll_s: float = 60.0, *, force_events: list[str] | None = None) -> None:
        await self.prepare()
        stop_at = self.clock() + timedelta(minutes=minutes)
        try:
            while self.clock() < stop_at:
                if not await self.tick(force_events=force_events):
                    return
                force_events = None if force_events and all(ev in self.done for ev in force_events) else force_events
                await asyncio.sleep(poll_s)
            self.halt("run time over")
            await self.shutdown(self.halted)
        except asyncio.CancelledError:
            await self.shutdown("cancelled")
            raise
        except Exception as err:
            self.write("error", {"what": "run", "error": repr(err)[:500]})
            await self.shutdown(f"exception {err!r}"[:200])
            raise
