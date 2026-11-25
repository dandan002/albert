"""albert.live_forward on recorded payloads (paper_forward books, Kalshi portfolio orders, the demo
IOC response) and a fake exchange; plus the order client's taker exception."""

import asyncio
import json
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from albert import paper_forward as pf
from albert.live_forward import PREFIX, ForwardBot, ForwardLimits, apr_body, order_fill, rule_of
from albert.live_guard import Switches
from albert.venues.kalshi_trade import KalshiTrader, OrderState, Refused, TradeError, order_body

FIX = Path(__file__).parent / "fixtures"


def _j(rel):
    return json.loads((FIX / rel).read_text())


CMN = _j("paper_forward/cmn_two_sided.json")
CMN_CLOSE = pf.parse_t(CMN["market"]["close_time"])
APR_BOOK = _j("paper_forward/apr_qualifying_book.json")
APR_EVENT = _j("paper_forward/apr_event_books.json")
ORDER = _j("kalshi/portfolio_order_single.json")["order"]          # an executed 1-lot NO fill, as recorded
IOC = _j("live_forward/ioc_create_response.json")["response"]


class FakeApi:
    def __init__(self, markets, books, results=None):
        self.ms, self.books, self.results = markets, books, results or {}

    def open_markets(self, series):
        return [m for m in self.ms if m["event_ticker"].startswith(series + "-")]

    def book(self, ticker):
        return self.books[ticker]

    def fee(self, series):
        return ("quadratic", 1.0)

    def markets(self, tickers):
        return [{**m, "result": self.results.get(m["ticker"], "")} for m in self.ms if m["ticker"] in tickers]


class FakeTrader:
    """Creates rest (maker) or fill at once (IOC); order details come from the recorded order."""

    def __init__(self, positions=None, other=()):
        self.resting: list[OrderState] = list(other)
        self.pos = dict(positions or {})
        self.created: list[dict] = []
        self.cancelled: list[str] = []
        self.details: dict[str, dict] = {}
        self._id = 0

    async def get(self, path, params=None):
        oid = path.rsplit("/", 1)[-1]
        if path.startswith("/portfolio/orders/") and oid in self.details:
            return {"order": self.details[oid]}
        raise TradeError(404, {}, path)

    async def orders(self, status="resting", ticker=None):
        return [o for o in self.resting if o.status == "resting"]

    async def positions(self):
        return dict(self.pos)

    async def shards(self, tickers):
        return {}

    async def create(self, body):
        self._id += 1
        oid = f"o{self._id}"
        self.created.append(body)
        price = 1 - float(body["price"])            # NO price
        if body["time_in_force"] == "immediate_or_cancel":
            self.details[oid] = {**ORDER, "order_id": oid, "ticker": body["ticker"],
                                 "client_order_id": body["client_order_id"]}
            return {**IOC, "order_id": oid, "client_order_id": body["client_order_id"]}
        self.resting.append(OrderState(oid, body["client_order_id"], body["ticker"], "no", price, "resting",
                                       1.0, 0.0, 1.0, 0.0, None))
        self.details[oid] = {**ORDER, "order_id": oid, "status": "resting", "fill_count_fp": "0.00",
                             "maker_fill_cost_dollars": "0.000000"}
        return {"order_id": oid, "client_order_id": body["client_order_id"]}

    def fill(self, oid):
        for o in self.resting:
            if o.order_id == oid:
                o.status = "executed"
        self.details[oid] = {**ORDER, "order_id": oid}

    async def cancel(self, oid, ticker):
        self.cancelled.append(oid)
        for o in self.resting:
            if o.order_id == oid:
                o.status = "canceled"
        if oid in self.details and self.details[oid]["status"] == "resting":
            self.details[oid] = {**self.details[oid], "status": "canceled"}
        return {}

    async def cancel_own(self, prefix):
        done = [o.order_id for o in self.resting if o.status == "resting" and o.client_order_id.startswith(prefix)]
        for oid in done:
            await self.cancel(oid, "")
        return done


def _bot(tmp_path, trader, api, now, **kw):
    clock = {"t": now}
    bot = ForwardBot("prod", {"cmn": trader, "apr": trader}, api, tmp_path, clock=lambda: clock["t"],
                     log=lambda *_: None, **kw)
    return bot, clock


def _cmn_api():
    return FakeApi([CMN["market"]], {CMN["market"]["ticker"]: CMN["orderbook"]})


def test_rule_of_and_order_fill_on_the_recorded_order():
    assert rule_of("albc2026-1") == "cmn" and rule_of("alba2026-1") == "apr" and rule_of("albm1-1") is None
    assert order_fill(ORDER) == (1.0, 0.29)


def test_cmn_posts_the_paper_rule_with_real_orders_then_books_the_fill(tmp_path):
    t = FakeTrader()
    bot, clock = _bot(tmp_path, t, _cmn_api(), CMN_CLOSE - pf.CMN_LEAD + timedelta(minutes=3), rules=("cmn",))
    asyncio.run(bot.prepare())
    assert asyncio.run(bot.tick())
    [body] = t.created
    assert body["client_order_id"].startswith(PREFIX["cmn"]) and body["post_only"]
    assert body["side"] == "ask" and body["price"] == "0.1400"          # NO at 0.86 = sells YES at 0.14
    assert body["expiration_time"] == int(CMN_CLOSE.timestamp())
    assert bot.used("cmn") == pytest.approx(0.86)
    clock["t"] += timedelta(minutes=20)
    asyncio.run(bot.tick())
    assert len(t.created) == 1                                        # decided once
    t.fill("o1")
    asyncio.run(bot.tick())
    [f] = bot.fills.values()
    assert f["cost"] == 0.29 and f["rule"] == "cmn" and f["side"] == "no"     # the exchange's numbers
    bot.api.results = {CMN["market"]["ticker"]: "no"}
    clock["t"] = CMN_CLOSE + timedelta(hours=1)
    asyncio.run(bot.tick())
    assert bot.settled[CMN["market"]["ticker"]] == "no" and bot.used("cmn") == 0
    # the ledger replays to the same state
    b2, _ = _bot(tmp_path, FakeTrader(), _cmn_api(), clock["t"], rules=("cmn",))
    assert "cmn:" + CMN["market"]["event_ticker"] in b2.decided and b2.orders["o1"]["status"] == "filled"


def test_cmn_respects_the_cap_and_never_touches_a_market_with_a_position(tmp_path):
    now = CMN_CLOSE - pf.CMN_LEAD + timedelta(minutes=3)
    t = FakeTrader()
    bot, _ = _bot(tmp_path / "a", t, _cmn_api(), now, rules=("cmn",), limits=ForwardLimits(cmn_cap=0.5))
    asyncio.run(bot.tick())
    assert not t.created
    [d] = [r for r in map(json.loads, bot.path.read_text().splitlines()) if r["kind"] == "decision"]
    assert d["skipped"][0]["why"].startswith("cap")
    t = FakeTrader(positions={CMN["market"]["ticker"]: -3.0})
    bot, _ = _bot(tmp_path / "b", t, _cmn_api(), now, rules=("cmn",))
    asyncio.run(bot.tick())
    assert not t.created


def test_sigterm_parks_cmn_bids_and_the_next_start_reposts_them(tmp_path):
    t = FakeTrader()
    bot, clock = _bot(tmp_path, t, _cmn_api(), CMN_CLOSE - pf.CMN_LEAD + timedelta(minutes=3), rules=("cmn",))
    asyncio.run(bot.tick())
    asyncio.run(bot.shutdown("signal", park=True))
    assert t.cancelled == ["o1"] and bot.orders["o1"]["status"] == "parked"
    clock["t"] += timedelta(hours=1)
    b2, _ = _bot(tmp_path, t, _cmn_api(), clock["t"], rules=("cmn",))
    asyncio.run(b2.prepare())
    assert len(t.created) == 2 and t.created[1]["price"] == t.created[0]["price"]
    assert b2.orders["o1"]["status"] == "ended" and b2.orders["o2"]["reposts"] == "o1"
    asyncio.run(b2.shutdown("stop file"))                        # not a signal: nothing is parked
    b3, _ = _bot(tmp_path, t, _cmn_api(), clock["t"], rules=("cmn",))
    asyncio.run(b3.prepare())
    assert len(t.created) == 2


def test_global_kill_cancels_and_stops_and_strays_are_cancelled(tmp_path):
    stray = OrderState("x9", "albc-old-1", "T-1", "no", 0.9, "resting", 1.0, 0.0, 1.0, 0.0, None)
    manual = OrderState("m1", "", "T-2", "no", 0.08, "resting", 67.0, 0.0, 67.0, 0.0, None)
    t = FakeTrader(other=[stray, manual])
    sw = Switches(tmp_path / "live")
    bot, _ = _bot(tmp_path, t, _cmn_api(), CMN_CLOSE - pf.CMN_LEAD + timedelta(minutes=3), rules=("cmn",),
                  switches=sw)
    asyncio.run(bot.prepare())
    assert t.cancelled == ["x9"]                                  # the manual order is untouched
    asyncio.run(bot.tick())
    sw.kill("test")
    assert asyncio.run(bot.tick()) is False
    assert "o1" in t.cancelled and "m1" not in t.cancelled


def test_apr_takes_ioc_no_on_the_qualifying_leg_only(tmp_path):
    markets = APR_EVENT["markets"]
    books = {**APR_EVENT["orderbooks"], APR_BOOK["ticker"]: APR_BOOK["orderbook"]}
    if APR_BOOK["ticker"] not in {m["ticker"] for m in markets}:
        markets = markets + [{**markets[0], "ticker": APR_BOOK["ticker"]}]
    close = max(pf.parse_t(m["close_time"]) for m in markets)
    now = pf.settle_instant(close) - pf.APR_LEAD + timedelta(minutes=2)
    t = FakeTrader()
    bot, _ = _bot(tmp_path, t, FakeApi(markets, books), now, rules=("apr",))
    asyncio.run(bot.tick())
    [body] = t.created
    assert body["ticker"] == APR_BOOK["ticker"] and body["client_order_id"].startswith(PREFIX["apr"])
    assert body["side"] == "ask" and body["price"] == "0.0200" and not body["post_only"]      # NO at 0.98
    assert body["time_in_force"] == "immediate_or_cancel" and "cancel_order_on_pause" not in body
    [f] = bot.fills.values()
    assert f["rule"] == "apr" and f["cost"] == 0.29
    asyncio.run(bot.tick())
    assert len(t.created) == 1


def _prod(monkeypatch, **kw):
    monkeypatch.setenv("KALSHI_LIVE_ENABLED", "1")

    class _Signer:
        def headers(self, method, path):
            return {}

    ok = httpx.MockTransport(lambda req: httpx.Response(201, json={"order_id": "abc"}))
    return KalshiTrader("prod", _Signer(), allow_prod=True, transport=ok, **kw)


def test_order_client_allows_prod_takers_only_ioc_and_only_when_enabled(monkeypatch):
    maker = _prod(monkeypatch)
    with pytest.raises(Refused, match="post-only"):
        asyncio.run(maker.create(apr_body("T-1", 0.98, "alba1-1")))
    taker = _prod(monkeypatch, allow_taker=True)
    assert asyncio.run(taker.create(apr_body("T-1", 0.98, "alba1-2")))["order_id"] == "abc"
    gtc = order_body("T-1", "no", 0.98, 1, client_order_id="alba1-3", post_only=False)
    with pytest.raises(Refused, match="immediate_or_cancel"):
        asyncio.run(taker.create(gtc))
    big = apr_body("T-1", 0.98, "alba1-4")
    big["count"] = "3.00"                                        # $2.94 > the $2 per-order cap
    with pytest.raises(Refused, match="cap"):
        asyncio.run(taker.create(big))
    assert asyncio.run(taker.create(order_body("T-1", "no", 0.86, 1, client_order_id="albc1-1")))["order_id"]
    assert len(taker.requests) == 2 and not maker.requests


def test_apr_skips_a_leg_with_no_upside_after_the_fee(tmp_path):
    rec = _j("live_forward/apr_no_upside_book.json")             # recorded: YES bid 1c, ask 10c
    top = pf.book_top(rec["orderbook"])
    assert pf.apr_pick(top)["price"] == 0.99                     # the paper rule would take it
    m = {**APR_EVENT["markets"][0], "ticker": rec["ticker"], "event_ticker": rec["ticker"].rsplit("-", 1)[0]}
    now = pf.settle_instant(pf.parse_t(m["close_time"])) - pf.APR_LEAD + timedelta(minutes=2)
    t = FakeTrader()
    bot, _ = _bot(tmp_path, t, FakeApi([m], {rec["ticker"]: rec["orderbook"]}), now, rules=("apr",))
    asyncio.run(bot.tick())
    assert not t.created
    [d] = [r for r in map(json.loads, bot.path.read_text().splitlines()) if r["kind"] == "decision"]
    assert d["seen"][0]["skip"] == "no upside after fee"
