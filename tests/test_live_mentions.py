"""Live mentions bot (live_mentions.py): planning on recorded public payloads (the forward test's
``forward_*`` fixtures for KXTRUMPMENTION-26NOV06) and the safety paths on a fake exchange."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from albert.config import Settings
from albert.live_mentions import END_BEFORE, PREFIX, MentionLimits, MentionsBot, plan_event
from albert.venues.kalshi_trade import OrderState

FIX = Path(__file__).parent / "fixtures" / "kalshi"
EV = "KXTRUMPMENTION-26NOV06"
START = datetime(2026, 11, 6, 22, 30, tzinfo=UTC)
BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _j(name: str):
    return json.loads((FIX / name).read_text())


ROUTES = {"/markets/orderbooks": _j(f"forward_orderbooks_{EV}.json"),
          "/markets/candlesticks": _j(f"forward_candles_{EV}.json"),
          "/markets": _j(f"forward_markets_{EV}.json")}


def _transport() -> httpx.MockTransport:
    def handler(req: httpx.Request) -> httpx.Response:
        for suffix, body in ROUTES.items():
            if req.url.path.endswith(suffix):
                return httpx.Response(200, json=body)
        return httpx.Response(404, json={})
    return httpx.MockTransport(handler)


class FakeTrader:
    """Accepts creates (they rest), records every write; reads come from recorded payloads."""

    def __init__(self, resting=None, positions=None, fills=None, markets=None):
        self.resting: list[OrderState] = list(resting or [])
        self.pos = positions or {}
        self.fills = fills or []
        self.markets = markets
        self.created: list[dict] = []
        self.calls: list[tuple] = []
        self._id = 0

    async def get(self, path, params=None):
        self.calls.append(("GET", path))
        if path == "/milestones":
            return _j("forward_milestones_mentions.json")
        if path == "/portfolio/fills":
            return {"fills": self.fills}
        if path == "/markets":
            return self.markets or _j(f"forward_markets_{EV}.json")
        return {}

    async def orders(self, status="resting", ticker=None):
        return [o for o in self.resting if o.status == "resting"]

    async def positions(self):
        return dict(self.pos)

    async def create(self, body):
        self._id += 1
        oid = f"o{self._id}"
        self.created.append(body)
        price = float(body["price"])
        self.resting.append(OrderState(oid, body["client_order_id"], body["ticker"],
                                       "no" if body["side"] == "ask" else "yes",
                                       1 - price if body["side"] == "ask" else price,
                                       "resting", 1.0, 0.0, 1.0, 0.0, body.get("order_group_id")))
        return {"order_id": oid, "client_order_id": body["client_order_id"]}

    async def cancel(self, oid, ticker):
        self.calls.append(("DELETE", oid))
        for o in self.resting:
            if o.order_id == oid:
                o.status = "canceled"
        return {}

    async def cancel_own(self, prefix):
        assert prefix
        done = []
        for o in self.resting:
            if o.status == "resting" and o.client_order_id.startswith(prefix):
                o.status = "canceled"
                done.append(o.order_id)
        self.calls.append(("cancel_own", prefix))
        return done

    async def shards(self, tickers):
        page = self.markets or _j(f"forward_markets_{EV}.json")      # recorded GET /markets
        return {m["ticker"]: m["exchange_index"] for m in page["markets"] if m["ticker"] in tickers}

    async def create_order_group(self, limit):
        self.calls.append(("group", limit))
        return "g-1"

    async def trigger_order_group(self, gid):
        self.calls.append(("trigger", gid))
        for o in self.resting:
            if o.order_group_id == gid:
                o.status = "canceled"
        return {}

    async def delete_order_group(self, gid):
        self.calls.append(("delete_group", gid))


def _bot(tmp_path, trader, now, **lim):
    s = Settings(data_dir=tmp_path)
    clock = {"now": now}
    bot = MentionsBot(s, "demo", trader, tmp_path / "run", limits=MentionLimits(**lim), base=BASE,
                      stop_file=tmp_path / "STOP", transport=_transport(), clock=lambda: clock["now"],
                      log=lambda *_: None)
    return bot, clock


def test_plan_prices_match_the_frozen_forward_rule(tmp_path):
    plan = plan_event(Settings(data_dir=tmp_path), BASE, EV, START, START - timedelta(hours=6), 0.12,
                      transport=_transport())
    assert plan.height == 6 and plan["eligible"].all()
    row = plan.filter(plan["ticker"] == f"{EV}-AUTO").row(0, named=True)
    # recorded book 0.28 / 0.32 -> fair 0.2683 -> NO bid floor(1 - 0.2683 - 0.12) = 0.61 (forward test)
    assert row["price"] == 0.61 and row["cheap"]
    assert plan_event(Settings(data_dir=tmp_path), BASE, EV, START, START - timedelta(hours=6), 0.08,
                      transport=_transport()).filter(plan["ticker"] == f"{EV}-AUTO")["price"][0] == 0.65


def test_posts_one_no_bid_per_leg_with_group_expiry_and_prefix(tmp_path):
    t = FakeTrader()
    bot, clock = _bot(tmp_path, t, START - timedelta(hours=5, minutes=30))
    asyncio.run(bot.prepare())
    assert asyncio.run(bot.tick())
    assert len(t.created) == 6 and ("group", 5) in t.calls
    for b in t.created:
        assert b["client_order_id"].startswith(PREFIX) and b["side"] == "ask" and b["post_only"]
        assert b["count"] == "1.00" and b["order_group_id"] == "g-1"
        assert b["expiration_time"] == int((START - END_BEFORE).timestamp())
        assert b["exchange_index"] == 0          # routed to the shard the recorded markets list
    # the event is posted once; at start - 10 min it is closed (group triggered, orders cancelled)
    asyncio.run(bot.tick())
    assert len(t.created) == 6
    clock["now"] = START - END_BEFORE
    asyncio.run(bot.tick())
    assert ("trigger", "g-1") in t.calls and ("delete_group", "g-1") in t.calls
    assert not [o for o in t.resting if o.status == "resting"]
    kinds = [json.loads(x)["kind"] for x in (tmp_path / "run" / "ledger.jsonl").read_text().splitlines()]
    assert kinds.count("order") == 6 and "close" in kinds


def test_collateral_cap_and_blocked_manual_markets(tmp_path):
    manual = OrderState("m1", "manual-xyz", f"{EV}-AUTO", "yes", 0.5, "resting", 1, 0, 1, 0, None)
    t = FakeTrader(resting=[manual])
    bot, _ = _bot(tmp_path, t, START - timedelta(hours=5, minutes=30), max_collateral=1.5)
    asyncio.run(bot.prepare())
    asyncio.run(bot.tick())
    assert all(b["ticker"] != f"{EV}-AUTO" for b in t.created)          # manual market untouched
    assert sum(1 - float(b["price"]) for b in t.created) <= 1.5 + 1e-9    # NO prices within the cap
    assert manual.status == "resting"                                    # never cancelled


def test_circuit_breaker_triggers_the_event_group(tmp_path):
    t = FakeTrader()
    bot, clock = _bot(tmp_path, t, START - timedelta(hours=5, minutes=30))
    asyncio.run(bot.prepare())
    asyncio.run(bot.tick())
    page = _j(f"forward_markets_{EV}.json")
    hot = json.loads(json.dumps(page))
    for m in hot["markets"]:
        if m["ticker"] == f"{EV}-AUTO":
            m["yes_bid_dollars"] = "0.9600"
    t.markets = hot
    clock["now"] += timedelta(minutes=5)
    asyncio.run(bot.tick())
    assert ("trigger", "g-1") in t.calls
    assert not [o for o in t.resting if o.status == "resting"]


def test_kill_file_cancels_only_bot_orders_and_stops(tmp_path):
    manual = OrderState("m1", "manual-xyz", "KXOTHER-1", "yes", 0.5, "resting", 1, 0, 1, 0, None)
    t = FakeTrader(resting=[manual])
    bot, _ = _bot(tmp_path, t, START - timedelta(hours=5, minutes=30))
    asyncio.run(bot.prepare())
    asyncio.run(bot.tick())
    (tmp_path / "STOP").touch()
    assert asyncio.run(bot.tick()) is False
    assert manual.status == "resting"
    assert not [o for o in t.resting if o.status == "resting" and o.client_order_id.startswith(PREFIX)]


def test_a_taker_fill_on_our_order_halts(tmp_path):
    t = FakeTrader()
    bot, clock = _bot(tmp_path, t, START - timedelta(hours=5, minutes=30))
    asyncio.run(bot.prepare())
    asyncio.run(bot.tick())
    oid = next(iter(bot.events[EV].orders))
    t.fills = [{"trade_id": "tr1", "order_id": oid, "client_order_id": t.created[0]["client_order_id"],
                "ticker": t.created[0]["ticker"], "side": "no", "outcome_side": "no", "is_taker": True,
                "yes_price_dollars": "0.35", "count_fp": "1.00", "fee_cost": "0.0100", "ts": 1794000000}]
    clock["now"] += timedelta(minutes=1)
    assert asyncio.run(bot.tick()) is True          # halts on the next pass
    assert bot.halted and "taker fill" in bot.halted
    assert asyncio.run(bot.tick()) is False
    assert bot.spent_today() == pytest.approx(bot.events[EV].orders[oid]["price"] + 0.01)


def test_finance_mention_events_are_skipped(tmp_path):
    bot, _ = _bot(tmp_path, FakeTrader(), START - timedelta(hours=6))
    # no series list: ticker and Mentions-name rules only
    assert bot.is_financial("KXFEDMENTION") and bot.is_financial("KXEARNINGSMENTIONORCL")
    assert not bot.is_financial("KXTRUMPMENTION")
    # with a loader: the list decides, unknown series are skipped (fail closed)
    bot.series_loader = dict
    bot.series_info = {"KXTRUMPMENTION": {"category": "Mentions", "title": "What will Trump say?"},
                       "KXWALLERMENTION": {"category": "Mentions", "title": "Waller Mention"}}
    assert not bot.is_financial("KXTRUMPMENTION")
    assert bot.is_financial("KXWALLERMENTION") and bot.is_financial("KXNEWSERIESMENTION")
    bot.series_info = None
    assert bot.is_financial("KXTRUMPMENTION")       # loader set but the list failed to load


def _maker_fill(t, oid, created, tid="tr1"):
    return {"trade_id": tid, "order_id": oid, "client_order_id": created["client_order_id"],
            "ticker": created["ticker"], "side": "no", "outcome_side": "no", "is_taker": False,
            "yes_price_dollars": "0.10", "count_fp": "1.00", "fee_cost": "0.0000", "ts": 1794000000}


def test_unsettled_fill_counts_after_midnight_until_its_market_settles(tmp_path):
    t = FakeTrader()
    bot, clock = _bot(tmp_path, t, START - timedelta(hours=5, minutes=30))
    asyncio.run(bot.prepare())
    asyncio.run(bot.tick())
    oid = next(iter(bot.events[EV].orders))
    created = next(b for b in t.created)
    t.fills = [_maker_fill(t, oid, created)]
    t.pos = {created["ticker"]: -1.0}
    clock["now"] += timedelta(minutes=1)
    asyncio.run(bot.tick())
    cost = bot.events[EV].orders[oid]["price"]
    assert bot.open_exposure() == pytest.approx(cost)
    clock["now"] += timedelta(days=1)                    # a new UTC day: today's fill cost is 0
    asyncio.run(bot.tick())
    assert bot.spent_today() == 0.0
    assert bot.at_risk() >= cost - 1e-9                   # the open position still counts
    t.pos = {}                                            # the market settled
    clock["now"] += timedelta(minutes=1)
    asyncio.run(bot.tick())
    assert bot.open_exposure() == 0.0
    led = [json.loads(x) for x in (tmp_path / "run" / "ledger.jsonl").read_text().splitlines()]
    assert any(r["kind"] == "settled" and r["ticker"] == created["ticker"] for r in led)


def test_settlement_waits_for_the_grace_period(tmp_path):
    t = FakeTrader()
    bot, clock = _bot(tmp_path, t, START - timedelta(hours=5, minutes=30))
    asyncio.run(bot.prepare())
    asyncio.run(bot.tick())
    oid = next(iter(bot.events[EV].orders))
    t.fills = [_maker_fill(t, oid, t.created[0])]
    t.pos = {}                                            # positions lag the fill feed
    clock["now"] += timedelta(minutes=1)
    asyncio.run(bot.tick())
    clock["now"] += timedelta(minutes=5)
    asyncio.run(bot.tick())
    assert bot.open_exposure() > 0                        # not dropped inside SETTLE_GRACE


def test_restart_rebuilds_unsettled_fills_and_shrinks_the_room(tmp_path):
    old = tmp_path / "20261101T000000Z"
    old.mkdir()
    rec = {"ts": "2026-11-06T10:00:00+00:00", "kind": "fill", "ticker": "KXOLD-1", "cost": 0.9}
    gone = {"ts": "2026-11-06T10:00:00+00:00", "kind": "fill", "ticker": "KXOLD-2", "cost": 0.8}
    (old / "ledger.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps(gone) + "\n")
    t = FakeTrader(positions={"KXOLD-1": -1.0})          # KXOLD-2 already settled
    s = Settings(data_dir=tmp_path)
    clock = {"now": START - timedelta(hours=5, minutes=30)}
    bot = MentionsBot(s, "demo", t, tmp_path / "20261117T000000Z", limits=MentionLimits(max_collateral=1.5),
                      base=BASE, stop_file=tmp_path / "STOP", transport=_transport(),
                      clock=lambda: clock["now"], log=lambda *_: None)
    asyncio.run(bot.prepare())
    assert bot.open_exposure() == pytest.approx(0.9)
    asyncio.run(bot.tick())
    assert sum(1 - float(b["price"]) for b in t.created) <= 1.5 - 0.9 + 1e-9
