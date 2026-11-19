"""Live pre-speech NO-bid maker on Kalshi mention markets (demo, or prod behind the live gate).

The rule is the one frozen in ``research/mentions_maker`` and confirmed out of sample on 548 unseen
events by ``research/mentions_history`` (and recorded hourly by ``research/forward``): for a speech
event in the Mentions category (window series and earnings calls excluded) whose milestone start is
about 6 h away, rest ONE post-only NO bid per quoted word leg at ``floor(1 − fair − margin)`` with
``fair = σ(−0.1156 + 1.0474·logit(mid))`` (capped one tick inside the YES bid), until start − 10
min, and hold any fill to settlement. Events whose share of "already said" legs (bid ≥ 97c) jumped
more than 12 points since start − 24 h are skipped (the study's outcome-blind exclusion).
Finance-related series (``albert.categories``) are skipped, except a business executive's series
when its milestone venue ("… at NYT DealBook Summit") also hosts, within a day, a speaker the filter
passes (a mostly political event; user request 2026-11-18).

Safety, on top of the order layer's own (``venues/kalshi_trade``: write allowlist, own-orders-only
cancels, prod gate ``KALSHI_LIVE_ENABLED=1`` + ``--live``):

* every order is 1 contract, carries the ``albm`` client_order_id prefix (under the bot prefix
  ``alb``), sits in a per-event order group (exchange-side cap on contracts matched per 15 s) and
  has ``expiration_time`` = start − 10 min, so the exchange cancels it even if this process dies;
* markets where the account holds a position or a non-bot order are never touched;
* collateral cap (the bot's worst case): resting NO bids plus the larger of today's fill cost and
  the cost of every bot fill whose market has not settled yet never exceed ``max_collateral`` (so a
  fill from before midnight UTC still counts, and a restart rebuilds unsettled fills from earlier
  runs' ledgers); no new events once today's fill cost exceeds ``max_daily_loss``;
* expected-fill budget (``max_expected`` > 0; user request 2026-11-19, ``research/mentions_budget``):
  postings also keep the expected fill cost, max(today's, unsettled fill cost) + the sum of each
  resting bid's fill probability x price, within ``max_expected``, and after fills push it over,
  resting bids are cancelled (largest expected cost first) until it is back under; so realized fill
  cost passes ``max_expected`` only by fills between two passes, while more bids rest than the
  collateral cap alone would allow;
* circuit breaker: if a leg asked ≤ 80c at posting trades or bids at ≥ 95c (a word was said, so
  the speech has started or leaked), the event's order group is triggered (all its orders cancel);
* if the milestone start moves earlier than 10 min from now, the event's orders are cancelled;
* kill switch file and SIGTERM cancel every ``albm`` order and stop.

Early phase (``early_hours``; live since 2026-11-18 at the user's request, with the paper bot
running alongside; ``research/mentions_ext`` Q2 ``L12h/m12c/extend``): the same NO bids are also posted ``early_hours`` before the start and rest
until start − 6 h (their own order group and expiration); at 6 h the frozen order above is posted
only on legs the early phase did not fill, and not at all if the early phase's breaker fired.
Main (6 h) phases are served before early ones, and both count against ``max_events`` and the cap.
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
EARLY_END = DUE[1]                                  # an early phase rests until start − 6 h
BREAKER_PRICE, CHEAP_ASK = 0.95, 0.80
# Fill model (research/mentions_budget, fitted on every replayed bid 2026-08-17 .. 11-16): P(a bid
# fills before its phase ends) = logistic(w . [1, main phase, price, price^2, log(spread)]), and the
# share of a phase's fills by elapsed window fraction 0, 0.1, ..., 1 (fills are front-loaded).
FILL_W = (-2.1400, 0.5058, 11.4240, -11.8087, 0.4444)
FILL_CURVE = {"early": (0.0, 0.2807, 0.3427, 0.4819, 0.5380, 0.6737, 0.7673, 0.7953, 0.8854, 0.8936, 1.0),
              "main": (0.0, 0.2479, 0.2923, 0.4267, 0.4485, 0.5545, 0.6484, 0.6732, 0.7761, 0.8167, 1.0)}
SETTLE_GRACE = timedelta(minutes=15)  # a fill leaves the open exposure only this long after it, once
                                      # its market shows no position (positions can lag the fill feed)


@dataclass
class MentionLimits:
    margin: float = 0.12
    max_collateral: float = 20.0      # $: resting NO bids + max(today's fills, unsettled fills)
    max_daily_loss: float = 10.0      # $: today's fill cost (worst case) before new events stop
    max_events: int = 3               # events with resting orders at once
    max_orders_per_event: int = 40
    group_limit: int = 5              # exchange-side: contracts matched per rolling 15 s
    early_hours: float = 0.0          # > 7: also bid from start − early_hours to start − 6 h (0 = off)
    max_expected: float = 0.0         # $: expected fill cost target (0 = off: max_collateral only)


@dataclass
class EventState:
    event: str
    start: datetime
    end: datetime
    group: str | None
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)   # order_id -> {ticker, price, cheap}
    closed: bool = False
    breaker: bool = False
    phase: str = "main"               # "main" (6 h -> start − 10 min) or "early" (early_hours -> 6 h)
    lead: timedelta = END_BEFORE      # the phase ends this long before the start


def venue(milestone_title: str | None) -> str | None:
    """Where a speech happens, from its milestone title: the text after the last " at "
    ("Elon Musk at NYT DealBook Summit" -> "nyt dealbook summit"); None without one."""
    t = " ".join((milestone_title or "").split())
    i = t.lower().rfind(" at ")
    return t[i + 4:].strip().lower() or None if i >= 0 else None


def fill_prob(phase: str, price: float, spread: float | None) -> float:
    """P(a NO bid at ``price`` fills before its phase ends), ``FILL_W``."""
    s = min(max(spread if spread is not None and math.isfinite(spread) else 0.5, 0.01), 0.5)
    x = (1.0, 1.0 if phase == "main" else 0.0, price, price * price, math.log(s))
    return 1 / (1 + math.exp(-sum(w * v for w, v in zip(FILL_W, x, strict=True))))


def p_remaining(p0: float, elapsed: float, phase: str) -> float:
    """P(fill in the rest of the window | not filled after ``elapsed`` (fraction) of it)."""
    curve = FILL_CURVE["main" if phase == "main" else "early"]
    x = min(max(elapsed, 0.0), 1.0) * 10
    i = min(int(x), 9)
    F = curve[i] + (curve[i + 1] - curve[i]) * (x - i)
    return p0 * (1 - F) / max(1 - p0 * F, 1e-9)


def _key(event: str, phase: str) -> str:
    """``events`` / ``done`` key: the event ticker for the main phase, ``<event>@early`` otherwise."""
    return event if phase == "main" else f"{event}@{phase}"


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
        self.open_fills: dict[str, dict[str, Any]] = {}   # ticker -> {cost, since}: fills not yet settled
        self.halted: str | None = None
        self._n = 0
        # Finance filter (albert.categories): the series list is reloaded once a day; an event whose
        # series is not in it yet is skipped, so a new finance series cannot slip through untitled.
        self.series_loader = series_loader
        self.series_info: dict[str, dict] | None = None
        self.series_day: str | None = None
        self.skipped_financial: set[str] = set()
        # Executives' series pass at a mostly political event (milestone venue shared with a
        # speaker the filter passes; user request 2026-11-18).
        self.venues: dict[str, str | None] = {}
        self.allowed_exec: set[str] = set()
        self.capped: set[str] = set()      # (event, phase) keys already logged as skipped by max_events
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

    def open_exposure(self) -> float:
        """Cost of bot fills whose market has not settled (worst case: they all lose)."""
        return sum(v["cost"] for v in self.open_fills.values())

    def realized(self) -> float:
        """Fill cost that can still lose: today's, or every unsettled fill's if more."""
        return max(self.spent_today(), self.open_exposure())

    def at_risk(self) -> float:
        """Resting NO bids plus fills that can still lose: the worst case."""
        return self.resting_collateral() + self.realized()

    def _resting(self) -> list[tuple[EventState, str, dict[str, Any]]]:
        return [(e, oid, o) for e in self.events.values() if not e.closed for oid, o in e.orders.items()
                if o.get("status", "resting") == "resting"]

    def expected_cost(self, e: EventState, o: dict[str, Any], now: datetime) -> float:
        """A resting bid's expected fill cost from now to its phase end."""
        posted = o.get("posted", now)
        span = (e.end - posted).total_seconds()
        elapsed = (now - posted).total_seconds() / span if span > 0 else 1.0
        return p_remaining(o.get("p0", 1.0), elapsed, e.phase) * o["price"]

    def expected(self, now: datetime) -> float:
        """Fills that can still lose plus every resting bid's expected fill cost."""
        return self.realized() + sum(self.expected_cost(e, o, now) for e, _, o in self._resting())

    async def _rebalance(self, now: datetime) -> None:
        """After fills: cancel resting bids, largest expected cost first, until the expected fill
        cost is back within ``max_expected``."""
        target = self.limits.max_expected
        if target <= 0 or self.expected(now) <= target + 1e-9:
            return
        before = self.expected(now)
        cancelled = []
        for e, oid, o in sorted(self._resting(), key=lambda r: -self.expected_cost(r[0], r[2], now)):
            if self.expected(now) <= target + 1e-9:
                break
            t_send = time.time_ns()
            try:
                resp = await self.trader.cancel(oid, o["ticker"])
            except TradeError as err:      # filled or expired meanwhile: the next pass sees it
                self.write("error", {"what": "budget cancel", "order_id": oid, "error": str(err)[:200]})
                continue
            o["status"] = "budget_cancel"
            cancelled.append(oid)
            self.write("cancel", {"event": e.event, "phase": e.phase, "order_id": oid, "reason": "expected budget",
                                  **_timing(t_send, resp)})
        self.write("rebalance", {"expected_before": round(before, 4), "expected_after": round(self.expected(now), 4),
                                 "realized": round(self.realized(), 4), "cancelled": len(cancelled)})

    def _add_open(self, ticker: str, cost: float, since: datetime) -> None:
        cur = self.open_fills.get(ticker)
        self.open_fills[ticker] = {"cost": (cur["cost"] if cur else 0.0) + cost,
                                   "since": min(cur["since"], since) if cur else since}

    def _earlier_fills(self, pos: dict[str, float]) -> None:
        """Unsettled fills of earlier runs in this env (crash restarts must not reset the budget):
        every fill in a sibling run's ledger whose market still shows a position."""
        for led in sorted(self.run_dir.parent.glob("*/ledger.jsonl")):
            if led.parent == self.run_dir:
                continue
            for line in led.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("kind") == "fill" and abs(pos.get(rec.get("ticker"), 0.0)) > 1e-9:
                    self._add_open(rec["ticker"], float(rec.get("cost") or 0.0), datetime.fromisoformat(rec["ts"]))

    async def _settled(self, now: datetime) -> None:
        """Drop fills whose market no longer shows a position (settled), after SETTLE_GRACE."""
        old = [t for t, v in self.open_fills.items() if now - v["since"] >= SETTLE_GRACE]
        if not old:
            return
        pos = await self.trader.positions()
        for t in old:
            if abs(pos.get(t, 0.0)) <= 1e-9:
                self.write("settled", {"ticker": t, "cost": round(self.open_fills.pop(t)["cost"], 4)})

    # -- lifecycle --
    async def prepare(self) -> None:
        """Sweep stale bot orders, then block every market with a position or a non-bot order."""
        swept = await self.trader.cancel_own(PREFIX)
        resting = await self.trader.orders("resting")
        pos = await self.trader.positions()
        manual = {o.ticker for o in resting if not o.client_order_id.startswith(PREFIX)}
        self.blocked = manual | {t for t, q in pos.items() if abs(q) > 1e-9}
        self._earlier_fills(pos)
        self.write("prepare", {"swept": swept, "blocked": sorted(self.blocked),
                               "open_exposure": round(self.open_exposure(), 4)})

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
        await self._settled(now)
        await self._watch(now, starts)
        await self._rebalance(now)
        if self.spent_today() >= self.limits.max_daily_loss:
            self.write("no_new_events", {"spent_today": self.spent_today()})
            return True
        due = self._due(now, starts, force_events)
        for ev, start, phase in due:
            busy = {e.event for e in self.events.values() if not e.closed} - {ev}
            if len(busy) >= self.limits.max_events:
                if _key(ev, phase) not in self.capped:          # retried every pass while due
                    self.capped.add(_key(ev, phase))
                    self.write("skip", {"event": ev, "phase": phase, "reason": "max events",
                                        "open_events": sorted(busy)})
                continue
            await self._post_event(ev, start, now, forced=bool(force_events and ev in force_events), phase=phase)
        return True

    async def _milestones(self, now: datetime) -> dict[str, datetime]:
        from albert.research import forward as F

        page = await self.trader.get("/milestones", {"minimum_start_date": (now - timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"), "category": "Mentions", "limit": 500})
        due = F.mention_due(page, now - timedelta(hours=1), timedelta(0), timedelta(days=2), set())
        self.venues = {r["event"]: venue(r["ms_title"]) for r in due.iter_rows(named=True)}
        return {r["event"]: r["start"] for r in due.iter_rows(named=True)}

    def is_executive(self, series: str) -> bool:
        """A business executive's series (``categories.kalshi_mention_is_executive``); False when
        the series list is missing or does not know the series (fail closed)."""
        from albert.categories import kalshi_mention_is_executive

        if self.series_info is None:
            return self.series_loader is None and kalshi_mention_is_executive(series, "Mentions", None)
        info = self.series_info.get(series)
        return info is not None and kalshi_mention_is_executive(series, info.get("category"), info.get("title"))

    def political_venue(self, ev: str, starts: dict[str, datetime]) -> bool:
        """``ev``'s milestone venue ("… at NYT DealBook Summit") also hosts, within a day, a speaker
        the finance filter passes: a mostly political event, where executives are traded too."""
        from albert.research import forward as F

        v = self.venues.get(ev)
        if not v or ev not in starts:
            return False
        return any(o != ev and self.venues.get(o) == v and abs(s - starts[ev]) <= timedelta(days=1)
                   and F.speech_group(o.split("-")[0]) not in EXCLUDED_GROUPS
                   and not self.is_financial(o.split("-")[0]) for o, s in starts.items())

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

    def _due(self, now: datetime, starts: dict[str, datetime],
             force: list[str] | None) -> list[tuple[str, datetime, str]]:
        """(event, start, phase) to post now: main phases (start in (5 h, 6 h]) first, then early
        ones (start in (early_hours − 1 h, early_hours]), each by start time."""
        from albert.research import forward as F

        if force:
            return [(ev, now + timedelta(minutes=25), "main") for ev in force if ev not in self.done]
        early = timedelta(hours=self.limits.early_hours)
        out = []
        for ev, start in sorted(starts.items(), key=lambda kv: kv[1]):
            if ev in self.done or F.speech_group(ev.split("-")[0]) in EXCLUDED_GROUPS:
                continue
            series = ev.split("-")[0]
            if self.is_financial(series) and self.is_executive(series) and self.political_venue(ev, starts):
                if ev not in self.allowed_exec:
                    self.allowed_exec.add(ev)
                    self.write("executive_at_political_event", {"event": ev, "venue": self.venues.get(ev)})
            elif self.is_financial(series):
                if ev not in self.skipped_financial:
                    self.skipped_financial.add(ev)
                    self.write("skip_financial", {"event": ev})
                continue
            left = start - now
            if DUE[0] < left <= DUE[1]:
                out.append((ev, start, "main"))
            elif (early > EARLY_END + timedelta(hours=1) and _key(ev, "early") not in self.done
                  and early - timedelta(hours=1) < left <= early):
                out.append((ev, start, "early"))
        return sorted(out, key=lambda r: r[2] != "main")

    async def _post_event(self, ev: str, start: datetime, now: datetime, *, forced: bool,
                          phase: str = "main") -> None:
        self.done.add(_key(ev, phase))
        early = self.events.get(_key(ev, "early")) if phase == "main" else None
        if early is not None and early.breaker:
            self.write("skip", {"event": ev, "phase": phase, "reason": "early phase circuit breaker"})
            return
        filled = {o["ticker"] for o in early.orders.values() if o.get("status") == "filled"} if early else set()
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
                           & ~pl.col("ticker").is_in(F._strs(self.blocked | filled)))
        legs = legs.sort("price").head(self.limits.max_orders_per_event)
        room = self.limits.max_collateral - self.at_risk()
        exp_room = self.limits.max_expected - self.expected(now) if self.limits.max_expected > 0 else math.inf
        keep, total, exp_total = [], 0.0, 0.0
        for r in legs.iter_rows(named=True):
            r["p0"] = fill_prob(phase, r["price"], (r["ask"] - r["bid"]) if r["ask"] is not None and r["bid"] is not None
                                else None)
            c = r["p0"] * r["price"]
            if total + r["price"] > room + 1e-9:     # cheapest first: no later leg fits the hard cap
                break
            if exp_total + c > exp_room + 1e-9:     # a dearer leg can still have a smaller p x price
                continue
            keep.append(r)
            total += r["price"]
            exp_total += c
        lead = EARLY_END if phase == "early" else END_BEFORE
        end = start - lead
        self.write("plan", {"event": ev, "phase": phase, "start": start, "end": end, "forced": forced,
                            "early_filled": sorted(filled), "legs": plan.height,
                            "eligible": legs.height, "posting": len(keep), "collateral": round(total, 4),
                            "expected": round(exp_total, 4),
                            "rows": plan.to_dicts()})
        if not keep:
            return
        group = await self.trader.create_order_group(self.limits.group_limit)
        st = EventState(ev, start, end, group, phase=phase, lead=lead)
        self.events[_key(ev, phase)] = st
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
                              "coid": coid, "status": "resting", "p0": r["p0"], "posted": now}
            self.write("order", {"event": ev, "phase": phase, "order_id": oid, "client_order_id": coid,
                                 "ticker": r["ticker"],
                                 "side": "no", "price": r["price"], "fair": r["fair_cal"], "bid": r["bid"],
                                 "ask": r["ask"], "p_fill": round(r["p0"], 4), "expiration": exp,
                                 **_timing(t_send, resp)})
        self.log(f"{ev} ({phase}): {len(st.orders)} NO bids, ${self.resting_collateral():.2f} resting")

    async def _watch(self, now: datetime, starts: dict[str, datetime]) -> None:
        for e in self.events.values():
            if e.closed:
                continue
            moved = starts.get(e.event)
            reason = None
            if now >= e.end:
                reason = "end"
            elif moved is not None and moved - e.lead <= now:
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
        self.write("close", {"event": e.event, "phase": e.phase, "reason": reason})

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
            self._add_open(fr.ticker, cost, self.clock())
            o["status"] = "filled"
            self.write("fill", {"event": e.event, "phase": e.phase, **asdict(fr), "cost": round(cost, 4)})
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
