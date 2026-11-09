"""PaperTrader: simulated orders on recorded prod prints (KXTRUMPMENTION-26OCT05-TRAN, 2026-10-05)."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from albert.live_mentions import PREFIX
from albert.paper_trader import PaperTrader, ledger_report, maker_fee, through
from albert.venues.kalshi_trade import TradeError, order_body, parse_fill

FIX = Path(__file__).parent / "fixtures" / "kalshi"
TICKER = "KXTRUMPMENTION-26OCT05-TRAN"
TRADES = json.loads((FIX / "mentions_maker_trades_KXTRUMPMENTION-26OCT05-TRAN.json").read_text())
SERIES = json.loads((FIX / "forward_series_KXTRUMPMENTION.json").read_text())


class Reader:
    writable = False

    def __init__(self):
        self.calls = []

    async def get(self, path, params=None):
        self.calls.append((path, params))
        if path == "/markets/trades":
            lo = params.get("min_ts", 0)
            ts = [t for t in TRADES["trades"] if t["ticker"] == params["ticker"]
                  and datetime.fromisoformat(t["created_time"]).timestamp() >= lo]
            return {"trades": ts, "cursor": ""}
        if path.startswith("/series/"):
            return SERIES
        raise AssertionError(path)

    async def shards(self, tickers):
        return {}

    async def aclose(self):
        pass


class Clock:
    def __init__(self, iso):
        self.t = datetime.fromisoformat(iso)

    def __call__(self):
        return self.t

    def set(self, iso):
        self.t = datetime.fromisoformat(iso)


def _trader(at="2026-10-05T18:45:00+00:00"):
    clock = Clock(at)
    return PaperTrader(Reader(), clock=clock), clock


def test_rules():
    assert through("no", 0.04, 0.97, "yes") and not through("no", 0.03, 0.97, "yes")
    assert not through("no", 0.04, 0.97, "no")
    assert through("yes", 0.50, 0.49, "no") and not through("yes", 0.50, 0.50, "no")
    assert maker_fee(0.5, 1, "quadratic_with_maker_fees", 1.0) == 0.0044
    assert maker_fee(0.5, 1, "quadratic", 1.0) == 0.0


def test_refuses_writable_reader():
    r = Reader()
    r.writable = True
    with pytest.raises(ValueError):
        PaperTrader(r)


def test_no_bid_fills_at_first_print_through_it():
    t, clock = _trader()

    async def go():
        body = order_body(TICKER, "no", 0.04, 1, client_order_id=f"{PREFIX}x-1",
                          expiration_ts=int(datetime(2026, 10, 5, 23, tzinfo=UTC).timestamp()))
        resp = await t.create(body)
        cheap = await t.create(order_body(TICKER, "no", 0.03, 1, client_order_id=f"{PREFIX}x-2"))
        assert [o.order_id for o in await t.orders("resting")] == [resp["order_id"], cheap["order_id"]]
        clock.set("2026-10-05T23:00:00+00:00")
        page = await t.get("/portfolio/fills", {"limit": 200})
        assert len(page["fills"]) == 1
        f = parse_fill(page["fills"][0])
        assert (f.order_id, f.ticker, f.side, f.price, f.qty, f.is_taker) == (resp["order_id"], TICKER, "no", 0.04, 1.0, False)
        # the first YES taker print above 0.96 (0.97 at 18:50:54.9)
        assert f.ts_ms == int(datetime.fromisoformat("2026-10-05T18:50:54.905986+00:00").timestamp() * 1000)
        assert await t.positions() == {TICKER: -1.0}
        # the 0.03 bid (0.97 YES) is never traded through: still resting, nothing more on a re-poll
        assert [o.order_id for o in await t.orders("resting")] == [cheap["order_id"]]
        assert len((await t.get("/portfolio/fills"))["fills"]) == 1

    asyncio.run(go())


def test_cancelled_or_expired_orders_only_see_prints_before_their_end():
    t, clock = _trader()

    async def go():
        a = await t.create(order_body(TICKER, "no", 0.04, 1, client_order_id=f"{PREFIX}x-1"))
        g = await t.create_order_group(5)
        b = await t.create(order_body(TICKER, "no", 0.04, 1, client_order_id=f"{PREFIX}x-2", order_group_id=g))
        c = await t.create(order_body(TICKER, "no", 0.04, 1, client_order_id=f"{PREFIX}x-3",
                                      expiration_ts=int(datetime(2026, 10, 5, 18, 50, tzinfo=UTC).timestamp())))
        clock.set("2026-10-05T18:50:30+00:00")            # before the first through-print
        await t.cancel(a["order_id"], TICKER)
        await t.trigger_order_group(g)
        clock.set("2026-10-05T23:00:00+00:00")
        assert (await t.get("/portfolio/fills"))["fills"] == []
        assert await t.orders("resting") == []
        assert {o.order_id for o in await t.orders("canceled")} == {a["order_id"], b["order_id"], c["order_id"]}
        assert await t.cancel_own(PREFIX) == []

    asyncio.run(go())


def test_cancel_after_the_print_still_fills_on_the_next_poll():
    t, clock = _trader()

    async def go():
        a = await t.create(order_body(TICKER, "no", 0.04, 1, client_order_id=f"{PREFIX}x-1"))
        clock.set("2026-10-05T18:52:00+00:00")            # the bot cancels before it polled fills
        await t.cancel(a["order_id"], TICKER)
        clock.set("2026-10-05T18:53:00+00:00")
        fills = (await t.get("/portfolio/fills"))["fills"]
        assert [f["order_id"] for f in fills] == [a["order_id"]]

    asyncio.run(go())


def test_portfolio_reads_are_never_forwarded():
    t, _ = _trader()

    async def go():
        with pytest.raises(TradeError):
            await t.get("/portfolio/balance")
        assert t.reader.calls == []

    asyncio.run(go())


def test_ledger_report_settles_paper_fills(tmp_path):
    import httpx

    from albert.config import Settings

    results = json.loads((FIX / "forward_results_KXTRUMPMENTION-26OCT05.json").read_text())
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=results))
    run = tmp_path / "20261005T184500Z"
    run.mkdir()
    ev = "KXTRUMPMENTION-26OCT05"
    recs = [{"kind": "order", "ts": "2026-10-05T18:45:00+00:00", "event": ev, "order_id": "o1",
             "ticker": f"{ev}-TRAN", "side": "no", "price": 0.04},
            {"kind": "order", "ts": "2026-10-05T18:45:00+00:00", "event": ev, "order_id": "o2",
             "ticker": f"{ev}-CORN", "side": "no", "price": 0.10},
            {"kind": "order", "ts": "2026-10-05T18:45:00+00:00", "event": ev, "order_id": "o3",
             "ticker": f"{ev}-CORN", "side": "no", "price": 0.05},
            {"kind": "fill", "order_id": "o1", "fee": 0.0007}, {"kind": "fill", "order_id": "o2", "fee": 0.0016}]
    (run / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    df = ledger_report(Settings(data_dir=tmp_path), [run / "ledger.jsonl"], transport=transport)
    got = {r["price"]: (r["filled"], r["y"], r["pnl"]) for r in df.iter_rows(named=True)}
    assert got[0.04] == (True, 1, pytest.approx(-0.0407))          # TRAN settled YES: the NO bid lost
    assert got[0.10] == (True, 0, pytest.approx(0.8984))           # CORN settled NO: won 1 - price - fee
    assert got[0.05] == (False, 0, 0.0)
