import itertools
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from albert import paper_forward as pf
from albert.research import music_charts as mc

FIX = Path(__file__).parent / "fixtures" / "paper_forward"
TRADES = Path(__file__).parent / "fixtures" / "kalshi" / "mentions_maker_trades_KXTRUMPMENTION-26OCT05-CORN.json"


def load(name):
    return json.loads((FIX / name).read_text())


def test_cmn_quote_on_a_real_two_sided_book():
    d = load("cmn_two_sided.json")
    top = pf.book_top(d["orderbook"])
    assert top["yes_bid"] == 0.01 and top["yes_ask"] == 0.02
    q = pf.cmn_quote(top)
    assert q["mid"] == pytest.approx(0.015) and q["sell_at"] == 0.14 and q["price"] == 0.86
    assert pf.no_size_at(d["orderbook"], 0.86) == 0          # nobody rests at our price
    assert pf.no_size_at(d["orderbook"], 0.87) == 41


def test_cmn_quote_skips_a_one_sided_book():
    d = load("cmn_one_sided.json")
    top = pf.book_top(d["orderbook"])
    assert top["yes_bid"] is None and not pf.two_sided(top)
    assert pf.cmn_quote(top) is None


def test_cmn_rank_cheapest_no_first():
    c = [{"ticker": "B", "price": 0.86}, {"ticker": "A", "price": 0.87}, {"ticker": "C", "price": 0.86}]
    assert [x["ticker"] for x in pf.cmn_rank(c)] == ["B", "C", "A"]


def test_first_through_is_strict_and_inside_the_window():
    prints = json.loads(TRADES.read_text())["trades"]
    t0 = datetime(2026, 10, 5, 21, 0, tzinfo=UTC)
    end = datetime(2026, 10, 5, 23, 0, tzinfo=UTC)
    # NO bid at 0.37 sells YES at 0.63: the first YES print above 0.63 is 0.64 at 21:10:52
    hit = pf.first_through("no", 0.37, t0, end, prints)
    assert hit["created_time"].startswith("2026-10-05T21:10:52") and hit["yes_price_dollars"] == "0.6400"
    # at 0.36 (sell at 0.64) a 0.64 print is not through; the first 0.66 print is
    hit = pf.first_through("no", 0.36, t0, end, prints)
    assert float(hit["yes_price_dollars"]) > 0.64
    # nothing after the order's end counts
    assert pf.first_through("no", 0.37, t0, t0 + timedelta(minutes=5), prints) is None


def test_hits_prices_match_the_preview_table_to_the_weeks_ladders():
    d = mc.parse_hits_post((FIX / "hits_preview_20261027_trimmed.html").read_text(),
                           "chart-preview-marching-orders-2026-10-27")
    week = mc.week_end_for_post(pf.parse_t(d["published_at"]), "preview")
    assert week == date(2026, 10, 29)
    markets = load("kxalbumequiv_26oct29_markets.json")["markets"]
    priced = pf.hits_prices(d["items"], markets, week)
    assert priced and {r["event_ticker"] for r in priced} == {"KXALBUMEQUIV-HEAVYCROWN26OCT29"}
    assert all(r["activity"] == 33000 for r in priced)
    p = {r["strike"]: r["p_model"] for r in priced}
    ks = sorted(p)
    assert all(p[a] > p[b] for a, b in itertools.pairwise(ks))      # higher strike, lower probability
    assert p[20000] > 0.95 and p[40000] < 0.05
    assert pf.hits_prices(d["items"], markets, date(2026, 11, 5)) == []      # another week: nothing


def test_taker_pick_and_event_cap():
    top = {"yes_bid": 0.55, "yes_bid_size": 10.0, "yes_ask": 0.60, "yes_ask_size": 100.0}
    o = pf.taker_pick(0.80, top, budget=1.0)
    assert o["side"] == "yes" and o["n"] == 1 and o["fee"] == 0.02 and o["edge"] == pytest.approx(0.18)
    o = pf.taker_pick(0.20, top, budget=1.0)
    assert o["side"] == "no" and o["price"] == 0.45 and o["n"] == 2          # floor(1 / 0.45) = 2
    assert pf.taker_pick(0.58, top, budget=1.0) is None
    thin = {**top, "yes_ask_size": 0.5}
    assert pf.taker_pick(0.80, thin, budget=1.0) is None                    # less than one displayed contract
    a = {"n": 1, "price": 0.6, "fee": 0.02, "edge": 0.10}
    b = {"n": 1, "price": 0.5, "fee": 0.02, "edge": 0.20}
    assert pf.cap_event([a, b], 1.25) == [b, a] and pf.cap_event([a, b], 1.0) == [b]


def test_apr_pick_on_recorded_books():
    q = load("apr_qualifying_book.json")
    o = pf.apr_pick(pf.book_top(q["orderbook"]))
    assert o == {"side": "no", "price": 0.98, "n": 1, "fee": 0.01, "mid": pytest.approx(0.06)}
    ev = load("apr_event_books.json")
    tops = {t: pf.book_top(b) for t, b in ev["orderbooks"].items()}
    assert tops["KXTRUMPAPPROVE-26NOV21-E38.3"]["yes_bid"] == 0.16      # mid 18c: above the band
    assert all(pf.apr_pick(t) is None for t in tops.values())


def test_settle_instant_rounds_the_1259_close_up():
    assert pf.settle_instant(datetime(2026, 11, 21, 17, 59, tzinfo=UTC)) == datetime(2026, 11, 21, 18, 0, tzinfo=UTC)


class FakeApi:
    def __init__(self, market, book, prints=()):
        self.m, self.ob, self.prints = market, book, list(prints)
        self.calls = []

    def open_markets(self, series):
        return [self.m] if self.m["event_ticker"].startswith(series + "-") else []

    def book(self, ticker):
        self.calls.append(("book", ticker))
        return self.ob

    def fee(self, series):
        return ("quadratic", 1.0)

    def trades(self, ticker, min_ts):
        return [p for p in self.prints if p["ticker"] == ticker]

    def markets(self, tickers):
        return [self.m]

    def market(self, ticker):
        return {**self.m, "result": "no", "settlement_ts": "2026-11-24T06:00:00Z"}


def test_runner_cmn_end_to_end(tmp_path):
    d = load("cmn_two_sided.json")
    close = pf.parse_t(d["market"]["close_time"])
    now = {"t": close - pf.CMN_LEAD + timedelta(minutes=3)}
    print_t = now["t"] + timedelta(hours=5)
    prints = [{"ticker": d["market"]["ticker"], "trade_id": "p1", "created_time": print_t.isoformat(),
               "taker_side": "yes", "yes_price_dollars": "0.1500", "count_fp": "3.00"}]
    api = FakeApi(d["market"], d["orderbook"], prints)
    r = pf.Runner(api, tmp_path, bankroll=25.0, clock=lambda: now["t"], log=lambda s: None)
    r.cmn_scan()
    b = r.books["cmn"]
    [o] = b.orders.values()
    assert o["price"] == 0.86 and o["in_book"] and o["sell_at"] == 0.14 and o["ahead"] == 0
    assert b.cash_used(now["t"]) == pytest.approx(0.86)
    r.cmn_scan()                                              # decided once
    assert len(b.orders) == 1
    now["t"] = print_t + timedelta(minutes=10)
    r.cmn_match()
    assert b.orders[o["id"]]["fill"]["print_id"] == "p1"
    now["t"] = close + timedelta(hours=1)
    r.settle()
    assert b.settled[d["market"]["ticker"]]["result"] == "no"
    # the ledger replays to the same state
    b2 = pf.Book("cmn", tmp_path, bankroll=25.0)
    assert b2.orders[o["id"]]["fill"] is not None and d["market"]["ticker"] in b2.settled
    s = pf.summary(b2, tmp_path)
    assert s["fills"] == 1 and s["c_per_contract"] == pytest.approx(14.0) and s["losing_fills"] == 0


def test_runner_cmn_skips_outside_the_decision_window(tmp_path):
    d = load("cmn_two_sided.json")
    close = pf.parse_t(d["market"]["close_time"])
    api = FakeApi(d["market"], d["orderbook"])
    r = pf.Runner(api, tmp_path, clock=lambda: close - pf.CMN_LEAD - timedelta(minutes=5), log=lambda s: None)
    r.cmn_scan()
    assert not r.books["cmn"].orders and not api.calls
