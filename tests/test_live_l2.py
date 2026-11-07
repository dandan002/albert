"""Order layer and live broker, offline. Parsers run on recorded portfolio payloads (read-only
GETs on the real account, user id redacted); the exchange is a fake that answers in those shapes.
The private WebSocket messages (user_order, fill) are built from the recorded REST order and fill
objects, which carry the same fields; replace them with demo recordings once demo keys exist."""

import asyncio
import copy
import json
import uuid
from pathlib import Path

import httpx
import pytest

from albert.backtest.l2 import Event, Quoter, SimConfig, _mark_executions, events_from_rows
from albert.live_l2 import BOT_PREFIX, LiveBroker, LiveGuards, LiveMaker, reconcile, run_actions
from albert.venues import kalshi_trade as kt
from albert.venues.kalshi_trade import (
    KalshiTrader,
    NotAllowed,
    TradeError,
    balance_dollars,
    order_body,
    parse_fill,
    parse_order,
    positions_by_ticker,
)

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _load(name: str):
    return json.loads((FIX / f"{name}.json").read_text())


def _rows(name: str = "ws_stream_wide.jsonl") -> list[dict]:
    return [json.loads(line) for line in (FIX / name).read_text().splitlines()]


# ---- parsers on recorded payloads ---------------------------------------------------------------
def test_parse_recorded_orders_fills_positions_balance():
    manual = parse_order(_load("portfolio_orders_resting")["orders"][0])
    # The user's manual order: sell YES at 0.92 == a bid for NO at 0.08.
    assert manual.ticker == "KXIMPEACH-29-JAN20" and manual.side == "no"
    assert manual.price == pytest.approx(0.08) and manual.status == "resting"
    assert manual.remaining == pytest.approx(67.22) and manual.client_order_id == ""
    executed = parse_order(_load("portfolio_orders_executed")["orders"][0])
    assert executed.side == "yes" and executed.filled == executed.initial > 0 and executed.remaining == 0
    f = parse_fill(_load("portfolio_fills")["fills"][0])
    assert f.side == "yes" and f.price == pytest.approx(0.73) and f.qty > 0 and f.fee > 0
    assert f.is_taker and f.ts_ms and f.trade_id
    pos = positions_by_ticker(_load("portfolio_positions"))
    assert pos == {"KXIMPEACH-29-JAN20": pytest.approx(67.22)}      # zero positions dropped
    assert balance_dollars(_load("portfolio_balance")) == pytest.approx(95.5422)
    assert _load("account_limits")["write"]["refill_rate"] == 100   # 10 creates/s, 50 cancels/s


def test_order_body_quotes_the_yes_leg():
    yes = order_body("T-1", "yes", 0.42, 1, client_order_id="alb1-1", order_group_id="g",
                     expiration_ts=1700000000)
    assert (yes["side"], yes["price"], yes["count"]) == ("bid", "0.4200", "1.00")
    assert yes["post_only"] and yes["cancel_order_on_pause"] and yes["expiration_time"] == 1700000000
    assert yes["time_in_force"] == "good_till_canceled" and yes["order_group_id"] == "g"
    no = order_body("T-1", "no", 0.40, 1, client_order_id="alb1-2")
    assert (no["side"], no["price"]) == ("ask", "0.6000")
    with pytest.raises(ValueError):
        order_body("T-1", "no", 1.0, 1, client_order_id="x")


# ---- safety rails --------------------------------------------------------------------------------
def test_write_allowlist_excludes_cancel_all_and_batches():
    oid = "01a1a73c-99a8-79b6-8f8a-d46d62dbffd9"
    assert kt._write_allowed("POST", "/portfolio/events/orders")
    assert kt._write_allowed("DELETE", f"/portfolio/events/orders/{oid}")
    for method, path in [("DELETE", "/portfolio/events/orders"),
                         ("DELETE", "/portfolio/events/orders/batched"),
                         ("DELETE", "/portfolio/orders"), ("POST", "/portfolio/orders"),
                         ("POST", "/portfolio/events/orders/batched"),
                         ("POST", f"/portfolio/events/orders/{oid}/amend")]:
        assert not kt._write_allowed(method, path), (method, path)


def test_prod_writes_need_env_flag_and_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(kt, "secret", lambda name: None)
    with pytest.raises(NotAllowed):
        kt.check_writable("prod", allow_prod=True)
    monkeypatch.setattr(kt, "secret", lambda name: "1" if name == kt.LIVE_FLAG else None)
    with pytest.raises(NotAllowed):
        kt.check_writable("prod", allow_prod=False)
    kt.check_writable("prod", allow_prod=True)
    kt.check_writable("demo", allow_prod=False)


class _Signer:
    def headers(self, method, path, now_ms=None):
        return {"KALSHI-ACCESS-KEY": "test"}


def test_read_only_client_refuses_writes(monkeypatch):
    monkeypatch.setattr(kt, "secret", lambda name: None)
    t = KalshiTrader("prod", _Signer(), transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(NotAllowed):
        asyncio.run(t.cancel("01a1a73c-99a8-79b6-8f8a-d46d62dbffd9", "X"))
    assert t.requests == []


def _resting_with_bot_orders() -> dict:
    """The recorded resting-orders page plus two copies re-labelled as bot orders."""
    page = _load("portfolio_orders_resting")
    bot = []
    for i, coid in enumerate([f"{BOT_PREFIX}20261105T000000Z-1", f"{BOT_PREFIX}20261104T000000Z-7"]):
        o = copy.deepcopy(page["orders"][0])
        o.update(order_id=f"0000000{i}-0000-0000-0000-000000000000", client_order_id=coid,
                 ticker="KXHIGHDEN-26NOV05-B60.5")
        bot.append(o)
    return {"cursor": "", "orders": page["orders"] + bot}


def test_cancel_own_never_touches_manual_orders():
    page = _resting_with_bot_orders()

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            return httpx.Response(200, json=page)
        return httpx.Response(200, json={"order_id": req.url.path.rsplit("/", 1)[-1]})

    t = KalshiTrader("demo", _Signer(), transport=httpx.MockTransport(handler))
    done = asyncio.run(t.cancel_own(f"{BOT_PREFIX}20261105"))
    assert done == ["00000000-0000-0000-0000-000000000000"]
    deletes = [p for m, p in t.requests if m == "DELETE"]
    assert deletes == ["/portfolio/events/orders/00000000-0000-0000-0000-000000000000"]
    with pytest.raises(NotAllowed):
        asyncio.run(t.cancel_own(""))


def test_429_is_retried():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(429, json={"error": "too many requests"}) if len(calls) < 3 else \
            httpx.Response(201, json={"order_id": "abc"})

    t = KalshiTrader("demo", _Signer(), transport=httpx.MockTransport(handler))
    assert asyncio.run(t.create(order_body("T", "yes", 0.5, 1, client_order_id="alb-1")))["order_id"] == "abc"
    assert len(calls) == 3


# ---- broker --------------------------------------------------------------------------------------
class FakeTrader:
    """Answers like the exchange: creates rest (no matching), cancels succeed once."""

    def __init__(self, resting: dict | None = None, positions: dict | None = None,
                 reject: set[str] | None = None) -> None:
        self.resting = [parse_order(o) for o in (resting or {"orders": []})["orders"]]
        self.pos = positions_by_ticker(positions or {})
        self.reject = reject or set()
        self.log: list[tuple] = []

    async def create(self, body):
        self.log.append(("create", body["client_order_id"], body["side"], body["price"]))
        if body["client_order_id"] in self.reject:
            raise TradeError(400, {"error": {"code": "post_only_cross"}}, "create")
        eid = str(uuid.uuid4())
        yes_p = float(body["price"])
        self.resting.append(parse_order({
            "order_id": eid, "client_order_id": body["client_order_id"], "ticker": body["ticker"],
            "book_side": body["side"], "yes_price_dollars": body["price"], "status": "resting",
            "initial_count_fp": body["count"], "remaining_count_fp": body["count"],
            "fill_count_fp": "0.00"}))
        assert 0 < yes_p < 1
        return {"order_id": eid, "client_order_id": body["client_order_id"], "fill_count": "0.00",
                "remaining_count": body["count"], "ts_ms": 0}

    async def cancel(self, order_id, ticker):
        self.log.append(("cancel", order_id))
        before = len(self.resting)
        self.resting = [o for o in self.resting if o.order_id != order_id]
        if len(self.resting) == before:
            raise TradeError(404, {"error": "not found"}, "cancel")
        return {"order_id": order_id}

    async def orders(self, status="resting", ticker=None):
        return list(self.resting)

    async def positions(self):
        return dict(self.pos)

    async def balance(self):
        return 95.54

    async def cancel_own(self, prefix):
        out = [o.order_id for o in self.resting if o.client_order_id.startswith(prefix)]
        for i in out:
            await self.cancel(i, "")
        return out

    async def create_order_group(self, limit):
        self.log.append(("group", limit))
        return "11111111-2222-3333-4444-555555555555"

    async def delete_order_group(self, gid):
        self.log.append(("delete_group", gid))

    async def queue_positions(self, tickers):
        return {}


def _broker(**kw) -> LiveBroker:
    return LiveBroker(SimConfig(writes_per_s=10.0), LiveGuards(**kw), client_prefix=f"{BOT_PREFIX}T")


def _events():
    evs = events_from_rows(_rows())
    _mark_executions(evs)
    return evs


def test_strategy_never_bids_against_its_own_resting_order():
    """After our improved bid shows up in the exchange book, the strategy keeps it instead of
    stepping in front of it (which it would do if the book still contained our order)."""
    b, fake = _broker(), FakeTrader()
    strat = Quoter(min_spread=0.0, lo=0.0, improve=True, min_requote_s=0)
    evs = _events()
    market = evs[0].market
    for ev in evs[:3]:
        b.apply(ev)
        strat.on_event(b, ev.market)
    assert b.actions and all(k == "create" for k, _ in b.actions)
    asyncio.run(run_actions(b, fake, expiry_s=900))
    mine = b.live_orders(market)
    assert mine and all(o.status == "active" for o in mine)
    n_orders = len(b.orders)
    raw_before = {s: b.book.markets[market].best(s) for s in ("yes", "no")}
    t = b.now
    for i, o in enumerate(mine):   # the exchange book now shows our orders
        side_price = o.price
        b.apply(Event(t + i + 1, "delta", market,
                      {"market_ticker": market, "side": o.side, "price_dollars": f"{side_price:.4f}",
                       "delta_fp": "1.00"}))
        strat.on_event(b, market)
    for s in ("yes", "no"):   # the raw book has our price as best; the strategy's view does not
        ours = [o for o in mine if o.side == s]
        if ours:
            assert b.book.markets[market].best(s)[0] == pytest.approx(ours[0].price)
            assert b.markets[market].best(s) == raw_before[s]
    assert len(b.orders) == n_orders and not b.actions
    assert all(o.status == "active" for o in mine)


def test_cancel_before_ack_is_sent_after_the_ack():
    b, fake = _broker(), FakeTrader()
    evs = _events()
    b.apply(evs[0])
    oid = b.place(evs[0].market, "yes", 0.10)
    assert b.cancel(oid) and b.orders[oid].status == "cancel_pending"
    asyncio.run(run_actions(b, fake, expiry_s=900))
    assert [x[0] for x in fake.log] == ["create", "cancel"]
    assert b.orders[oid].status == "cancelled" and not fake.resting


def test_post_only_reject_and_limits():
    b = _broker(max_open_orders=1)
    fake = FakeTrader(reject={f"{BOT_PREFIX}T-1"})
    b.apply(_events()[0])
    m = _events()[0].market
    assert b.place(m, "yes", 0.10, qty=2) is None and b.diag["refused_order_qty"] == 1
    oid = b.place(m, "yes", 0.10)
    assert b.place(m, "no", 0.10) is None and b.diag["refused_open_orders"] == 1
    asyncio.run(run_actions(b, fake, expiry_s=900))
    assert b.orders[oid].status == "rejected" and b.diag["rejects_400"] == 1
    b.blocked.add(m)
    assert b.place(m, "yes", 0.10) is None and b.diag["refused_blocked_market"] == 1


def _fill_msg(coid: str, order_id: str, ticker: str, *, taker: bool = False, trade_id: str = "t1",
              post: str | None = None) -> dict:
    """A ``fill`` channel message built from the recorded REST fill (same field names)."""
    f = copy.deepcopy(_load("portfolio_fills")["fills"][0])
    f.update(client_order_id=coid, order_id=order_id, market_ticker=ticker, is_taker=taker,
             trade_id=trade_id, count_fp="1.00", fee_cost="0.010000", yes_price_dollars="0.1000",
             outcome_side="yes", book_side="bid")
    if post is not None:
        f["post_position_fp"] = post
    return {"type": "fill", "sid": 9, "msg": f}


def test_fills_update_position_once_and_halt_on_taker_or_foreign_fills():
    b, fake = _broker(), FakeTrader()
    ev = _events()[0]
    b.apply(ev)
    oid = b.place(ev.market, "yes", 0.10)
    asyncio.run(run_actions(b, fake, expiry_s=900))
    eid = b.exch_id[oid]
    msg = _fill_msg(b.client_id(oid), eid, ev.market, post="1.00")
    b.on_private(msg, b.now)
    b.on_private(msg, b.now)          # duplicate delivery
    assert len(b.fills) == 1 and b.position[ev.market] == 1.0 and b.diag["duplicate_fills"] == 1
    assert b.orders[oid].status == "filled" and b.cash == pytest.approx(-0.11)
    assert b.diag.get("post_position_mismatch", 0) == 0 and not b.halted
    # The user's manual fill in another market is ignored ...
    b.on_private(_fill_msg("", "x", "KXIMPEACH-29-JAN20", trade_id="t2"), b.now)
    assert not b.halted and b.diag["foreign_fills"] == 1
    # ... but a foreign fill in a market the bot trades halts it.
    b.on_private(_fill_msg("", "y", ev.market, trade_id="t3"), b.now)
    assert b.halted and "foreign fill" in b.halted
    b2 = _broker()
    b2.apply(ev)
    oid2 = b2.place(ev.market, "yes", 0.10)
    asyncio.run(run_actions(b2, fake, expiry_s=900))
    b2.on_private(_fill_msg(b2.client_id(oid2), b2.exch_id[oid2], ev.market, taker=True, trade_id="t4"), b2.now)
    assert b2.halted and "taker" in b2.halted


def test_halt_cancels_only_bot_orders_and_reconcile_flags_mismatches():
    manual = _load("portfolio_orders_resting")
    fake = FakeTrader(resting=manual)
    b = _broker()
    for ev in _events()[:2]:
        b.apply(ev)
    m = _events()[0].market
    o1, o2 = b.place(m, "yes", 0.10), b.place(m, "no", 0.10)
    asyncio.run(run_actions(b, fake, expiry_s=900))
    b.book.now += 10 * 10**9            # past the reconcile grace period
    assert reconcile(b, asyncio.run(fake.orders()), {}, {}) == []
    fake.resting = [o for o in fake.resting if o.order_id != b.exch_id[o1]]   # vanished
    issues = reconcile(b, asyncio.run(fake.orders()), {m: 1.0}, {})
    assert any("not resting" in i for i in issues) and any("position" in i for i in issues)
    b.halt("test")
    asyncio.run(run_actions(b, fake, expiry_s=900))
    assert b.orders[o2].status == "cancelled"
    assert [o.ticker for o in fake.resting] == ["KXIMPEACH-29-JAN20"]       # manual order untouched
    assert all(k != "cancel" or eid != manual["orders"][0]["order_id"] for k, eid, *_ in fake.log)


def test_reconcile_adopts_a_create_whose_response_was_lost():
    b, fake = _broker(), FakeTrader()
    b.apply(_events()[0])
    m = _events()[0].market
    oid = b.place(m, "yes", 0.10)
    asyncio.run(fake.create(order_body(m, "yes", 0.10, 1, client_order_id=b.client_id(oid))))
    b.actions.clear()
    b.orders[oid].status = "cancel_pending"     # what run_actions does after a transport error
    b.book.now += 10 * 10**9
    issues = reconcile(b, asyncio.run(fake.orders()), {}, {})
    assert issues == [] and b.actions == [("cancel", oid)]
    asyncio.run(run_actions(b, fake, expiry_s=900))
    assert not fake.resting


def test_guard_ceilings():
    with pytest.raises(NotAllowed):
        LiveGuards(max_order_qty=2).check()
    with pytest.raises(NotAllowed):
        LiveGuards(max_loss=100).check()
    LiveGuards().check()


def test_prepare_blocks_markets_with_manual_orders_or_positions(tmp_path):
    fake = FakeTrader(resting=_resting_with_bot_orders(), positions=_load("portfolio_positions"))
    lm = LiveMaker(tmp_path, "demo", None, fake, run_id="R", strategy=Quoter(),
                   cfg=SimConfig(), guards=LiveGuards(), pairs=[])
    ok = asyncio.run(lm.prepare(["KXIMPEACH-29-JAN20", "KXHIGHDEN-26NOV05-B60.5"]))
    assert ok == ["KXHIGHDEN-26NOV05-B60.5"]
    assert "KXIMPEACH-29-JAN20" in lm.broker.blocked and lm.broker.group
    # leftover bot orders from earlier runs are cancelled; the manual order is not
    assert sorted(o.client_order_id for o in fake.resting) == [""]
    end = asyncio.run(lm.shutdown())
    assert ("delete_group", lm.broker.group) in fake.log and end["fills"] == 0
    kinds = [json.loads(x)["kind"] for x in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert kinds[0] == "prepare" and kinds[-1] == "end"


def test_live_maker_trades_recorded_frames_end_to_end(tmp_path):
    """Recorded market data in, real-shaped order traffic out, own orders netted from the book."""
    fake = FakeTrader()
    lm = LiveMaker(tmp_path, "demo", None, fake, run_id="R",
                   strategy=Quoter(min_spread=0.0, lo=0.0, improve=True, min_requote_s=0),
                   cfg=SimConfig(writes_per_s=10.0), guards=LiveGuards(), pairs=[])
    asyncio.run(lm.prepare([_events()[0].market]))
    for r in _rows()[:200]:
        lm.on_frame("kalshi", r["recv_ns"], r["frame"])
        asyncio.run(run_actions(lm.broker, fake, expiry_s=900))
    creates = [x for x in fake.log if x[0] == "create"]
    assert creates and all(c[1].startswith(f"{BOT_PREFIX}R-") for c in creates)
    resting_ids = {o.order_id for o in fake.resting}
    local_live = {lm.broker.exch_id[o.oid] for o in lm.broker.open_orders() if o.oid in lm.broker.exch_id}
    assert resting_ids == local_live
    end = asyncio.run(lm.shutdown())
    assert not fake.resting and end["halted"] == "shutdown"


def test_order_group_trigger_sends_a_json_body():
    """Kalshi demo answered a body-less trigger PUT with 400 invalid_content_type (plumbing run
    2026-11-06): the PUT must carry an empty JSON object."""
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path, req.headers.get("content-type"), req.content))
        return httpx.Response(200, json={})

    t = KalshiTrader("demo", _Signer(), transport=httpx.MockTransport(handler))
    gid = "9ba666a5-9453-42c6-8a9b-0419dc456723"
    asyncio.run(t.trigger_order_group(gid))
    asyncio.run(t.reset_order_group(gid))
    assert [s[1].rsplit("/", 1)[-1] for s in seen] == ["trigger", "reset"]
    assert all(s[0] == "PUT" and s[2] == "application/json" and s[3] == b"{}" for s in seen)


def test_private_frames_recorded_on_demo_parse():
    """Real demo plumbing frames (2026-11-07): user_order lifecycle, a taker fill and an
    order-group trigger parse into the broker's types."""
    import json
    from pathlib import Path

    from albert.venues.kalshi_trade import parse_fill, parse_order
    frames = json.loads((Path(__file__).parent / "fixtures" / "kalshi" / "ws_private_demo.json").read_text())
    orders = [parse_order(f["msg"]) for f in frames if f["type"] == "user_order"]
    fills = [parse_fill(f["msg"]) for f in frames if f["type"] == "fill"]
    groups = [f["msg"]["event_type"] for f in frames if f["type"] == "order_group_updates"]
    assert {o.status for o in orders} >= {"resting", "canceled", "executed"}
    assert all(o.client_order_id.startswith("albplumb") for o in orders)
    assert len(fills) == 1 and fills[0].qty == 1.0 and fills[0].is_taker and 0 < fills[0].fee < 0.02
    assert groups == ["created", "triggered"]
    assert not any("user_id" in f["msg"] for f in frames)


# ---- latency instrumentation ---------------------------------------------------------------------
def test_order_timing_legs_from_a_recorded_prod_ack():
    """Clock values and ts_ms from the 2026-11-07 prod create (data/live/manual ledger)."""
    from albert.timing import OrderTiming

    t = {"now": 0}
    tm = OrderTiming(clock=lambda: t["now"])
    send = 1794044031970345000
    t["now"] = send - 2_000_000
    tm.queued("create", 1, "KXLOLGAME-26NOV071700GENIG-GEN", event_ns=send - 9_000_000)
    t["now"] = send
    tm.sent("create", 1)
    t["now"] = send + 6_395_029
    tm.acked("create", 1, {"order_id": "x", "ts_ms": 1794044031975})
    assert tm.flush(t["now"]) == []                     # waiting for the WS update
    tm.ws("create", 1, 1794044031976134000)
    (rec,) = tm.flush(t["now"])
    assert rec["decide_to_send"] == 2.0 and rec["event_to_decide"] == 7.0
    assert rec["rest"] == pytest.approx(6.395) and rec["send_to_engine"] == pytest.approx(4.655)
    assert rec["send_to_ws"] == pytest.approx(5.789) and rec["event_to_ws"] == pytest.approx(14.789)
    s = tm.summary()
    assert s["create.rest"]["n"] == 1 and tm.summary() == {}     # the window resets
    assert tm.summary(whole_run=True)["create.rest"]["p50"] == pytest.approx(6.395)


def test_live_maker_logs_timing_and_the_holdback_shows_in_event_to_decide(tmp_path):
    """On recorded frames, every order is decided at least the classifier holdback after the
    frame that triggered it (the strategy sees events only once released)."""
    fake = FakeTrader()
    lm = LiveMaker(tmp_path, "demo", None, fake, run_id="R",
                   strategy=Quoter(min_spread=0.0, lo=0.0, improve=True, min_requote_s=0),
                   cfg=SimConfig(writes_per_s=10.0), guards=LiveGuards(), pairs=[], holdback_ms=500)
    asyncio.run(lm.prepare([_events()[0].market]))
    clock = {"now": 0}
    lm.broker.timing.clock = lambda: clock["now"]
    for r in _rows()[:200]:
        clock["now"] = r["recv_ns"]
        lm.on_frame("kalshi", r["recv_ns"], r["frame"])
        asyncio.run(run_actions(lm.broker, fake, expiry_s=900))
    recs = lm.broker.timing.flush(clock["now"] + 10**12)
    recs += [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines()
             if json.loads(x)["kind"] == "timing"]
    driven = [x for x in recs if "event_to_decide" in x]
    assert driven and all(x["event_to_decide"] >= 500 for x in driven)
    assert all(x["action"] in ("create", "cancel") and "rest" in x for x in recs)
