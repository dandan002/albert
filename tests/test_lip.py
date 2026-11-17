import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from albert.research import lip

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


# ---- parsers on recorded payloads ---------------------------------------------------------------
def test_program_rows_recorded():
    rows = lip.program_rows(_load("lip_incentive_programs.json"))
    assert len(rows) == 4
    r = rows[0]
    assert r["market_ticker"] == "KXZYNPOS-26DEC03-T97.5"
    assert r["pool_usd"] == pytest.approx(95.0)          # 950000 centi-cents
    assert r["discount_factor"] == pytest.approx(0.5)    # 5000 bps
    assert r["target_size"] == pytest.approx(1000.0)
    assert r["start"] == datetime(2026, 11, 17, 14, 11, 21, tzinfo=UTC)
    assert r["max_reward_usd"] is None and r["paid_out"] is False
    assert {x["incentive_description"] for x in rows} == {"series_lip", "new_event", None}
    df = lip.programs_frame(rows)
    rt = df.filter(pl.col("market_ticker") == "KXRT-JUM-65").row(0, named=True)
    assert rt["days"] == pytest.approx(1.0) and rt["pool_per_day"] == pytest.approx(100.0)


def test_schedule_rows_recorded():
    rows = lip.schedule_rows(_load("lip_incentive_schedules.json"))
    assert len(rows) == 3
    assert rows[0]["series_ticker"] == "ECMOV" and rows[0]["reward_usd"] == pytest.approx(100.0)
    assert rows[0]["window_days"] == 7 and rows[0]["all_markets"] is False
    assert rows[1]["event_ticker"] is None                 # covers the whole series
    assert rows[0]["ends"] == datetime(2027, 1, 1, 5, tzinfo=UTC)


def test_book_levels_recorded_batch():
    levels = list(lip.book_levels(_load("lip_orderbooks.json")))
    tickers = {t for t, *_ in levels}
    assert tickers == {"KXZYNPOS-26DEC03-T97.5", "KXZYNPOS-26DEC03-T94.5"}   # unknown ticker: empty
    b = lip.books_dict(pl.DataFrame(levels, schema=["ticker", "side", "price", "size"], orient="row"))
    assert b["KXZYNPOS-26DEC03-T94.5"]["no"] == {0.01: 7151.0}
    assert max(b["KXZYNPOS-26DEC03-T97.5"]["yes"]) == pytest.approx(0.79)


# ---- scoring rules (help.kalshi.com article 13823851) -----------------------------------------
def test_practitioner_example_level_and_truncate():
    # target 1,000, DF 0.9: 500 at the best, 300 one tick back, 400 two ticks back
    book = {0.50: 500.0, 0.49: 300.0, 0.48: 400.0}
    s = lip.score_side(book, {}, 1000, 0.9)
    assert s.ref == 0.50
    assert s.total == pytest.approx(500 + 270 + 324)
    t = lip.score_side(book, {}, 1000, 0.9, "truncate")
    assert t.total == pytest.approx(500 + 270 + 200 * 0.81)


def test_reference_price_and_discount():
    # cumulative size reaches 1/5 of 1,000 at 0.09; the 0.08 level is one tick below it
    book = {0.20: 100.0, 0.19: 25.0, 0.10: 20.0, 0.09: 64.87, 0.08: 410.0, 0.04: 45.0, 0.03: 252.0, 0.01: 1001.0}
    s = lip.score_side(book, {}, 1000, 0.5)
    assert s.ref == 0.09
    expected = 209.87 + 410 * 0.5 + 45 * 0.5**5 + 252 * 0.5**6 + 1001 * 0.5**8
    assert s.total == pytest.approx(expected)
    assert lip.ref_price(book, 1000) == 0.09


def test_thin_side_scores_nothing_and_voids_snapshot():
    assert lip.score_side({0.40: 150.0}, {}, 1000, 0.5).total == 0.0
    valid, sy, sn = lip.score_snapshot({0.40: 150.0}, {0.55: 5000.0}, 1000, 0.5)
    assert (valid, sy, sn) == (False, 0.0, 0.0)
    # our 1,000 at 1c completes the YES side: valid, and it scores full credit (it sets the ref)
    valid, sy, sn = lip.score_snapshot({0.40: 150.0}, {0.55: 5000.0}, 1000, 0.5, our_yes={0.01: 1000.0})
    assert valid and sy == pytest.approx(1000 / 1150) and sn == 0.0


def test_stepping_ahead_of_a_wall_excludes_it():
    wall = {0.01: 7151.0}
    assert lip.score_side(wall, {0.02: 100.0}, 1000, 0.5).share == pytest.approx(100 / 7251)
    assert lip.score_side(wall, {0.02: 100.0}, 1000, 0.5, "truncate").share == pytest.approx(0.1)
    # 1,000 one tick above reach Target Size: the wall no longer helps reach it
    assert lip.score_side(wall, {0.02: 1000.0}, 1000, 0.5).share == pytest.approx(1.0)


def test_help_page_payout_example():
    # $100 period, 10,000 snapshots, 8,000 valid, our share 20% of each valid snapshot -> $16
    pool, n = 100.0, 10_000
    total = sum(lip.reward_rate(pool, n, 0.2, 0.2) for _ in range(8_000))
    assert total == pytest.approx(16.0)
    assert lip.paid(total) == 16.0
    assert lip.paid(0.999) == 0.0 and lip.paid(1.239) == 1.23


def test_full_credit_price_post_only():
    assert lip.full_credit_price({0.97: 1220.0}, 0.01, 1000) == 0.97
    assert lip.full_credit_price({0.40: 50.0}, None, 1000) == 0.01    # others never reach 200
    assert lip.full_credit_price({0.97: 1220.0}, 0.03, 1000) is None  # 0.97 + 0.03 would cross


def test_candidate_prices_stop_at_window():
    book = {0.30: 600.0, 0.29: 600.0, 0.10: 5000.0}
    c = lip.candidate_prices(book, 0.65, 1000)
    assert 0.31 in c and 0.30 in c and 0.29 in c and 0.01 in c
    assert 0.10 not in c and 0.11 not in c       # below the level where 1,000 is reached
    assert all(p + 0.65 < 1 for p in c)


# ---- allocation and evaluation on synthetic books ---------------------------------------------
def _progs(t0: datetime) -> pl.DataFrame:
    rows = [{"id": f"p{i}", "market_ticker": tk, "series_ticker": "KXTEST", "event_ticker": "KXTEST-E",
             "incentive_type": "liquidity", "incentive_description": "series_lip", "start": t0 - timedelta(days=1),
             "end": t0 + timedelta(days=6), "pool_usd": 70.0, "discount_factor": 0.5, "target_size": 1000.0,
             "max_reward_usd": None, "paid_out": False} for i, tk in enumerate(("A", "B"))]
    return lip.programs_frame(rows)


def test_allocate_respects_budget_and_min_payout():
    t0 = datetime(2026, 11, 17, tzinfo=UTC)
    progs = _progs(t0)
    books = {"A": {"yes": {0.01: 50000.0}, "no": {0.97: 3000.0}},
             "B": {"yes": {0.50: 20000.0}, "no": {0.45: 20000.0}}}
    meta = {"KXTEST": {"financial": False, "category": "Entertainment"}}
    opts = lip.market_options(progs, books, meta)
    a = lip.allocate(opts, 100.0, t0)
    assert a["cost"].sum() <= 100.0 + 1e-9
    top = a.row(0, named=True)
    assert (top["ticker"], top["side"], top["price"]) == ("A", "yes", 0.02)   # step ahead of the 1c wall
    assert top["share"] == pytest.approx(1.0)
    assert (a["rpd"] * 6 >= 1.0).all()
    ev = lip.evaluate(a, progs, books)
    assert ev.filter(pl.col("ticker") == "A")["rpd_real"][0] == pytest.approx(top["rpd"])
    # next poll: someone else steps ahead with 1,000 at 3c, our 2c order no longer qualifies
    books2 = {**books, "A": {"yes": {0.03: 1000.0, 0.01: 50000.0}, "no": {0.96: 3000.0}}}
    ev2 = lip.evaluate(a, progs, books2)
    assert ev2.filter(pl.col("ticker") == "A")["rpd_real"][0] == pytest.approx(0.0)


def test_financial_series_skipped():
    t0 = datetime(2026, 11, 17, tzinfo=UTC)
    books = {"A": {"yes": {0.01: 50000.0}, "no": {0.97: 3000.0}}, "B": {"yes": {}, "no": {}}}
    opts = lip.market_options(_progs(t0), books, {"KXTEST": {"financial": True}})
    assert opts.is_empty()


def test_farmer_locked_capital_nets_pairs():
    f = lip.Farmer(lip.Variant("x"))
    f.pos["A"] = {"yes": 10.0, "no": 10.0, "yes_cost": 3.0, "no_cost": 6.5}
    f.pos["B"] = {"yes": 5.0, "no": 0.0, "yes_cost": 1.0, "no_cost": 0.0}
    assert f.locked() == pytest.approx(0.0 + 1.0)     # the A pair is worth $10 >= its $9.50 cost
    assert f.locked(settled={"B"}) == pytest.approx(0.0)


# ---- reaction events on recorded stream lines ---------------------------------------------------
def test_reaction_events_recorded():
    lines = (FIX / "ws_stream_lip.jsonl").read_text().splitlines()
    ev = lip.reaction_events(lines, big=500)
    assert len(ev) == 1
    e = ev[0]
    # KXRT-EBE-35, 2026-11-12 10:05 UTC: 515.94 YES at 0.74 over a 46-lot best bid at 0.71,
    # cancelled 0.35 s later (a flicker, not a resting farmer)
    assert (e["side"], e["price"], e["size"], e["kind"]) == ("yes", 0.74, 515.94, "improve")
    assert (e["prev_best"], e["prev_best_size"]) == (0.71, 46.0)
    assert e["withdraw_s"] == pytest.approx(0.354, abs=0.01) and e["counter_s"] is None


def test_trade_hits_sides():
    tr = pl.DataFrame({"taker_side": ["no", "no", "yes", "yes"], "yes_price": [0.02, 0.03, 0.98, 0.97],
                       "count": [10.0, 5.0, 7.0, 3.0]})
    # a YES bid at 2c is reached by takers buying NO at yes_price <= 2c
    assert lip.trade_hits(tr, "yes", 0.02)["count"].sum() == 10.0
    # a NO bid at 2c is reached by takers buying YES at a NO price (1 - yes_price) <= 2c
    assert lip.trade_hits(tr, "no", 0.02)["count"].sum() == 7.0
