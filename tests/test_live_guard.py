"""Phase 6 account-level safety (live_guard.py, kalshi_trade.py): global switches, the order/fill
audit log, per-order caps, and the guard's daily-loss halt and at-risk kill switch. Exchange shapes
are the recorded ``portfolio_fills``, ``portfolio_order_single``, ``portfolio_orders_resting`` and
``markets_by_tickers`` payloads (copies re-labelled where a scenario needs another price or ticker)."""

import asyncio
import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

import albert.venues.kalshi_trade as kt
from albert.live_guard import Audit, Guard, GuardLimits, Switches, next_midnight, tag_of, yes_value
from albert.venues.kalshi_trade import KalshiTrader, Refused, TradeError, order_body, parse_order

FIX = Path(__file__).parent / "fixtures" / "kalshi"
NOW = datetime(2026, 11, 19, 15, 0, tzinfo=UTC)


def _j(name: str):
    return json.loads((FIX / f"{name}.json").read_text())


class _Signer:
    def headers(self, method, path, now_ms=None):
        return {"KALSHI-ACCESS-KEY": "key-id-123", "KALSHI-ACCESS-SIGNATURE": "sekrit-signature"}


def _prod(monkeypatch, handler, **kw) -> KalshiTrader:
    monkeypatch.setattr(kt, "secret", lambda name: "1" if name == kt.LIVE_FLAG else None)
    return KalshiTrader("prod", _Signer(), allow_prod=True, transport=httpx.MockTransport(handler), **kw)


def _ok(req: httpx.Request) -> httpx.Response:
    if req.method == "POST":
        return httpx.Response(201, json={"order_id": "01a1ed75-6828-7519-82b6-fb98ce7e681e",
                                         "client_order_id": json.loads(req.content)["client_order_id"]})
    return httpx.Response(200, json={"order_id": req.url.path.rsplit("/", 1)[-1], "reduced_by": "1.00"})


def _audit_lines(root: Path) -> list[dict]:
    return [json.loads(x) for p in sorted(root.glob("orders-*.jsonl")) for x in p.read_text().splitlines()]


OID = "01a1a73c-99a8-79b6-8f8a-d46d62dbffd9"


# ---- switches ----------------------------------------------------------------------------------
def test_switches_kill_halt_lapse_and_fail_closed(tmp_path):
    sw = Switches(tmp_path)
    assert sw.blocked(NOW) is None
    (tmp_path / "KILL").touch()                         # a bare `touch` is enough
    assert sw.killed() == "kill file" and sw.blocked(NOW).startswith("kill")
    sw.kill("manual: test", NOW)
    assert sw.killed() == "manual: test"
    assert sw.resume() == ["KILL"] and sw.killed() is None
    until = next_midnight(NOW)
    assert until == datetime(2026, 11, 20, tzinfo=UTC)
    sw.halt("guard: loss", until, NOW)
    assert "guard: loss" in sw.halted(NOW) and sw.blocked(NOW).startswith("halt")
    assert sw.halted(until) is None                     # lapses by itself at midnight
    sw.halt_path.write_text("not json")
    assert sw.halted(NOW) == "halt file (unreadable)"   # fail closed
    assert sw.resume(halt=True) == ["HALT.json"] and sw.blocked(NOW) is None


def test_tags_name_the_bot():
    assert tag_of("albm20261117T164556Z-9") == "albm"
    assert tag_of("albi20261109T142124Z-3") == "albi"
    assert tag_of("") is None and tag_of(None) is None and tag_of("manual-1") is None


# ---- order client --------------------------------------------------------------------------------
def test_every_write_is_audited_without_credentials(tmp_path, monkeypatch):
    calls = []

    def handler(req):
        calls.append(req.method)
        if req.method == "POST" and len(calls) == 3:
            return httpx.Response(400, json={"error": {"code": "post_only_cross"}})
        return _ok(req)

    audit = Audit(tmp_path / "audit")
    t = _prod(monkeypatch, handler, actor="albm", audit=audit)
    body = order_body("KXVANCEMENTION-26NOV18-WORK", "no", 0.29, 1, client_order_id="albm1-1")
    asyncio.run(t.create(body))
    asyncio.run(t.cancel(OID, "KXVANCEMENTION-26NOV18-WORK"))
    with pytest.raises(TradeError):
        asyncio.run(t.create({**body, "client_order_id": "albm1-2"}))
    asyncio.run(t.get("/portfolio/fills"))              # reads are not audited
    recs = _audit_lines(audit.root)
    assert [(r["method"], r["status"]) for r in recs] == [("POST", "ok"), ("DELETE", "ok"), ("POST", 400)]
    assert all(r["actor"] == "albm" and r["env"] == "prod" and r["ms"] >= 0 for r in recs)
    assert recs[0]["body"]["client_order_id"] == "albm1-1" and recs[0]["response"]["order_id"]
    assert "post_only_cross" in recs[2]["error"]
    text = "".join(p.read_text() for p in audit.root.glob("*.jsonl"))
    assert "sekrit" not in text and "KALSHI-ACCESS" not in text and "key-id-123" not in text


def test_global_switch_refuses_new_orders_but_lets_cancels_through(tmp_path, monkeypatch):
    sw = Switches(tmp_path / "live")
    audit = Audit(tmp_path / "audit")
    t = _prod(monkeypatch, _ok, audit=audit, switches=sw)
    body = order_body("T-1", "no", 0.30, 1, client_order_id="albm1-1")
    sw.kill("manual: test")
    with pytest.raises(Refused) as e:
        asyncio.run(t.create(body))
    assert isinstance(e.value, TradeError) and e.value.status == 0 and "kill" in e.value.reason
    asyncio.run(t.cancel(OID, "T-1"))
    assert [m for m, _ in t.requests] == ["DELETE"]     # nothing was sent for the create
    sw.resume()
    sw.halt("guard: loss", datetime.now(UTC) + timedelta(hours=1))
    with pytest.raises(Refused):
        asyncio.run(t.create(body))
    sw.halt("guard: loss", datetime.now(UTC) - timedelta(seconds=1))    # lapsed
    assert asyncio.run(t.create(body))["order_id"]
    recs = _audit_lines(audit.root)
    assert [r.get("refused", "")[:20] for r in recs if "refused" in r] == ["global switch: kill:", "global switch: halt:"]


def test_per_order_caps_and_post_only_on_prod(tmp_path, monkeypatch):
    t = _prod(monkeypatch, _ok)
    for count, price in [(6, 0.10), (3, 0.80)]:          # 6 contracts; $2.40 of cost
        with pytest.raises(Refused):
            asyncio.run(t.create(order_body("T-1", "no", price, count, client_order_id="albm1-1")))
    asyncio.run(t.create(order_body("T-1", "yes", 0.50, 2, client_order_id="albm1-2")))
    with pytest.raises(Refused, match="post-only"):
        asyncio.run(t.create(order_body("T-1", "yes", 0.50, 1, client_order_id="albm1-3", post_only=False)))
    demo = KalshiTrader("demo", _Signer(), transport=httpx.MockTransport(_ok))
    asyncio.run(demo.create(order_body("T-1", "yes", 0.50, 1, client_order_id="albplumb-1", post_only=False)))
    assert [m for m, _ in t.requests] == ["POST"] and [m for m, _ in demo.requests] == ["POST"]


def test_cancel_only_client_cannot_create(tmp_path, monkeypatch):
    t = _prod(monkeypatch, _ok, cancel_only=True)
    with pytest.raises(Refused):
        asyncio.run(t.create(order_body("T-1", "no", 0.30, 1, client_order_id="albm1-1")))
    with pytest.raises(Refused):
        asyncio.run(t.create_order_group(5))
    asyncio.run(t.cancel(OID, "T-1"))
    assert [m for m, _ in t.requests] == ["DELETE"]


# ---- guard ---------------------------------------------------------------------------------------
BOT_ORDER = _j("portfolio_order_single")["order"]             # albm NO @0.29, KXVANCEMENTION-26NOV18-WORK
MARKETS = _j("markets_by_tickers")["markets"]                 # that market finalized (no) + an active one
OPEN_TICKER = MARKETS[1]["ticker"]


def _bot_fill(trade_id: str, ticker: str, no_price: float, count: float, ts: int, order_id: str) -> dict:
    """A recorded fill re-labelled as a maker NO fill of ours."""
    f = copy.deepcopy(_j("portfolio_fills")["fills"][0])
    f.update(trade_id=trade_id, fill_id=trade_id, order_id=order_id, ticker=ticker, market_ticker=ticker,
             outcome_side="no", side="no", book_side="ask", action="sell", is_taker=False, fee_cost="0.000000",
             yes_price_dollars=f"{1 - no_price:.4f}", no_price_dollars=f"{no_price:.4f}",
             count_fp=f"{count:.2f}", ts=ts)
    return f


class GuardTrader:
    """Answers the guard's reads like the exchange; ``cancel_own`` cancels by prefix."""

    writable = True

    def __init__(self, fills, orders, markets, resting=()):
        self.fills, self.orders_by_id, self.markets = fills, orders, markets
        self.resting = [parse_order(o) for o in resting]
        self.calls: list[tuple] = []

    async def get(self, path, params=None):
        params = params or {}
        self.calls.append((path, dict(params)))
        if path == "/portfolio/fills":
            fills = [f for f in self.fills if f["ts"] >= params.get("min_ts", 0)]
            if "cursor" not in params and len(fills) > 2:        # two pages
                return {"fills": fills[:2], "cursor": "next"}
            return {"fills": fills[2:] if "cursor" in params else fills, "cursor": ""}
        if path.startswith("/portfolio/orders/"):
            oid = path.rsplit("/", 1)[-1]
            if oid not in self.orders_by_id:
                raise TradeError(404, {"error": {"code": "not_found"}}, f"GET {path}")
            return {"order": self.orders_by_id[oid]}
        if path == "/markets":
            want = set(params["tickers"].split(","))
            return {"markets": [m for m in self.markets if m["ticker"] in want]}
        raise AssertionError(path)

    async def orders(self, status="resting"):
        return [o for o in self.resting if o.status == "resting"]

    async def cancel_own(self, prefix):
        done = []
        for o in self.resting:
            if o.status == "resting" and o.client_order_id.startswith(prefix):
                o.status = "canceled"
                done.append(o.order_id)
        self.calls.append(("cancel_own", prefix))
        return done


def _orders() -> dict:
    manual = copy.deepcopy(BOT_ORDER)
    manual_oid = _j("portfolio_fills")["fills"][0]["order_id"]
    manual.update(order_id=manual_oid, client_order_id="", ticker="KXIMPEACH-29-JAN20")
    return {BOT_ORDER["order_id"]: BOT_ORDER, manual_oid: manual}


def _guard(tmp_path, trader, clock, **lim):
    return Guard("prod", trader, tmp_path / "live", limits=GuardLimits(**lim), clock=lambda: clock["now"],
                 log=lambda *_: None, lookup_pause_s=0)


def test_guard_books_bot_fills_from_the_exchange_and_leaves_manual_trades_out(tmp_path):
    vance = _bot_fill("t-vance", BOT_ORDER["ticker"], 0.29, 1, 1795000000, BOT_ORDER["order_id"])
    fills = [*_j("portfolio_fills")["fills"], vance]          # three manual fills + ours
    t = GuardTrader(fills, _orders(), MARKETS)
    clock = {"now": NOW}
    st = asyncio.run(_guard(tmp_path, t, clock).tick())
    assert set(st["book"]) == {"albm"}
    b = st["book"]["albm"]
    assert (b["fills"], b["cost"], b["value"], b["equity"], b["unsettled_cost"]) == (1, 0.29, 1.0, 0.71, 0.0)
    assert st["fills_total"] == 4 and st["untagged"] == 0 and st["new_fills"] == 4
    assert st["day_pnl"] == 0.0 and st["at_risk"] == 0.0 and st["blocked"] is None
    audit = tmp_path / "live" / "audit" / "prod"
    assert len((audit / "fills.jsonl").read_text().splitlines()) == 4
    assert len((audit / "order_tags.jsonl").read_text().splitlines()) == 2
    # A restart reloads the audit: no duplicate fills, no repeat order lookups, min_ts resumes.
    t.calls.clear()
    st2 = asyncio.run(_guard(tmp_path, t, clock).tick())
    assert st2["new_fills"] == 0 and len((audit / "fills.jsonl").read_text().splitlines()) == 4
    assert not [c for c in t.calls if c[0].startswith("/portfolio/orders/")]
    assert t.calls[0][1]["min_ts"] == 1795000000 - 3600
    assert json.loads((tmp_path / "live" / "guard" / "prod" / "status.json").read_text())["book"]["albm"]["equity"] == 0.71


def _open_market(yes_bid: float, yes_ask: float) -> list[dict]:
    m = copy.deepcopy(MARKETS[1])
    m.update(yes_bid_dollars=f"{yes_bid:.4f}", yes_ask_dollars=f"{yes_ask:.4f}")
    return [MARKETS[0], m]


def _resting_bot_and_manual() -> list[dict]:
    page = _j("portfolio_orders_resting")["orders"]
    bot = copy.deepcopy(page[0])
    bot.update(order_id="00000001-0000-0000-0000-000000000000", client_order_id="albm20261119T000000Z-1")
    return [page[0], bot]


def test_day_loss_halts_until_midnight_sweeping_bot_orders_only(tmp_path):
    fill = _bot_fill("t-open", OPEN_TICKER, 0.85, 20, 1795000000, BOT_ORDER["order_id"])
    t = GuardTrader([fill], _orders(), _open_market(0.10, 0.14), _resting_bot_and_manual())
    clock = {"now": NOW}
    g = _guard(tmp_path, t, clock, max_day_loss=15.0, max_at_risk=60.0)
    st = asyncio.run(g.tick())
    assert st["equity"] == pytest.approx(20 * 0.88 - 17.0) and st["day_pnl"] == 0.0
    assert st["book"]["albm"]["unsettled_cost"] == 17.0 and st["blocked"] is None
    t.markets = _open_market(0.90, 0.94)                  # the speech goes badly: NO marked at 0.08
    g.marks.clear()
    clock["now"] = NOW + timedelta(hours=2)
    st = asyncio.run(g.tick())
    assert st["day_pnl"] == pytest.approx(20 * 0.08 - 20 * 0.88)
    assert st["blocked"].startswith("halt") and g.switches.halted(clock["now"])
    assert ("cancel_own", "alb") in t.calls
    left = [o for o in t.resting if o.status == "resting"]
    assert [o.client_order_id for o in left] == [_j("portfolio_orders_resting")["orders"][0]["client_order_id"]]
    assert not g.switches.killed()
    # The next UTC day: the halt has lapsed and the day's loss starts from zero again.
    clock["now"] = datetime(2026, 11, 20, 0, 1, tzinfo=UTC)
    g.marks.clear()
    st = asyncio.run(g.tick())
    assert st["blocked"] is None and st["day_pnl"] == 0.0 and st["day"] == "2026-11-20"
    kinds = [json.loads(x)["kind"] for x in (tmp_path / "live" / "guard" / "prod" / "ledger.jsonl").read_text().splitlines()]
    assert kinds.count("halt") == 1 and kinds.count("day_start") == 2 and "sweep" in kinds


def test_at_risk_above_the_cap_trips_the_global_kill_switch(tmp_path):
    fill = _bot_fill("t-open", OPEN_TICKER, 0.85, 20, 1795000000, BOT_ORDER["order_id"])
    t = GuardTrader([fill], _orders(), _open_market(0.10, 0.14), _resting_bot_and_manual())
    clock = {"now": NOW}
    g = _guard(tmp_path, t, clock, max_day_loss=15.0, max_at_risk=10.0)
    st = asyncio.run(g.tick())
    assert st["at_risk"] > 10.0 and "at risk" in g.switches.killed()
    assert st["blocked"].startswith("kill") and ("cancel_own", "alb") in t.calls
    clock["now"] += timedelta(minutes=1)
    t.calls.clear()
    asyncio.run(g.tick())
    assert ("cancel_own", "alb") not in t.calls          # nothing left resting: no repeat sweep


def test_an_order_the_exchange_cannot_find_stays_untagged_and_out_of_the_book(tmp_path):
    fill = _bot_fill("t-x", OPEN_TICKER, 0.85, 1, 1795000000, "01a1ffff-0000-0000-0000-000000000000")
    t = GuardTrader([fill], _orders(), MARKETS)
    st = asyncio.run(_guard(tmp_path, t, {"now": NOW}).tick())
    assert st["untagged"] == 1 and st["book"] == {}


def test_marks_from_recorded_markets():
    assert yes_value(MARKETS[0]) == (0.0, True)            # finalized, result no
    assert yes_value(MARKETS[1]) == (None, False)          # active, no quotes, no trades: at cost
    m = dict(MARKETS[1], yes_bid_dollars="0.2000", yes_ask_dollars="0.3000")
    assert yes_value(m) == (0.25, False)
