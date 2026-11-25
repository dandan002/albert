"""Tests for albert.research.exec_actions (fixtures are trimmed copies of responses recorded 2026-11-20)."""

import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from albert.research import exec_actions as E

FX = Path(__file__).parent / "fixtures"
FEED = (FX / "whitehouse" / "presidential_actions_feed_20261120_trim.xml").read_text()
MARKETS = json.loads((FX / "kalshi_exec_actions" / "markets_20261120_trim.json").read_text())["markets"]


def _m(ticker: str) -> dict:
    m = next(x for x in MARKETS if x["ticker"] == ticker)
    return {**m, "series_ticker": ticker.split("-")[0]}


def test_wh_feed_items_kinds_times_and_nominations():
    rows = E.wh_items(FEED, "2026-11-20T22:00:00+00:00")
    by = {r["title"]: r for r in rows}
    nom = by["Nominations and Withdrawals Sent to the Senate"]
    assert nom["kind"] == "nomination"
    assert nom["pub_ts"] == datetime(2026, 11, 10, 21, 56, 49, tzinfo=UTC)
    assert nom["pub_date_et"] == date(2026, 11, 10)              # 16:56 ET
    assert nom["n_nom"] >= 1 and nom["n_withdraw"] == 1
    assert nom["guid"].startswith("https://www.whitehouse.gov/?p=")
    assert by["VETERANS DAY, 2026"]["kind"] == "proclamation"
    assert by["DECREASING HOUSING COSTS FOR THE AMERICAN PEOPLE"]["kind"] == "eo"
    assert not any(r["round_time"] for r in rows)


def test_count_nominations_splits_withdrawals():
    content = ("<p>NOMINATIONS SENT TO THE SENATE:</p><p>Jane Roe, of Ohio, to be a Judge.</p>"
               "<p>John Doe, of Iowa, to be an Ambassador.</p><p>WITHDRAWAL SENT TO THE SENATE:</p>"
               "<p>Ann Poe, of Utah, to be a Member, which was sent to the Senate on May 1.</p>"
               "<p>The post Nominations Sent to the Senate appeared first on The White House.</p>")
    assert E.count_nominations(content) == (2, 1)


def test_senate_schedule_days_and_stamp():
    rows = E.senate_rows((FX / "senate" / "2026_schedule_trim.html").read_text(), 2026)
    days = {r["day"] for r in rows}
    assert date(2026, 11, 26) in days and date(2026, 11, 23) in days      # Thanksgiving state work period
    assert date(2026, 11, 17) not in days                                   # in session
    assert rows[0]["page_updated"] == "2025-11-21"                          # published before the year
    assert E.recess_week(E._days(date(2026, 11, 22), date(2026, 11, 28)), days)
    assert not E.recess_week(E._days(date(2026, 11, 15), date(2026, 11, 21)), days)


def test_period_parsing_weekly_daily_formats():
    assert E.period_of(_m("KXEOWEEK-26NOV28-4")) == ("2026-11-22", "2026-11-28")
    assert E.period_of(_m("KXTRUMPNOMNUM-26NOV22-T6")) == ("2026-11-22", "2026-11-28")
    assert E.period_of(_m("KXTRUMPACT-26NOV22-T9")) == ("2026-11-22", "2026-11-28")
    assert E.period_of(_m("KXMAMDANIEO-26NOV28-T0")) == ("2026-11-22", "2026-11-28")
    assert E.period_of(_m("KXTRUMPACTD-26NOV21-YES")) == ("2026-11-21", "2026-11-21")
    # Feb 2026 KXTRUMPACT markets were daily ("on Feb 28 (as checked ... Mar 1)")
    assert E.period_of(_m("KXTRUMPACT-26FEB22-26FEB28")) == ("2026-02-28", "2026-02-28")


def test_strikes():
    assert E.strike_yes(5, "greater", 4, None) == 1 and E.strike_yes(4, "greater", 4, None) == 0
    assert E.strike_yes(9, "greater_or_equal", 9, None) == 1
    assert E.strike_yes(7, "between", 7, 7) == 1 and E.strike_yes(6, "between", 7, 7) == 0
    assert E.need_for_yes("greater", 4, None) == (5, math.inf)
    assert E.need_for_yes("greater_or_equal", 9, None) == (9, math.inf)
    assert E.need_for_yes("between", 3, 3) == (3, 3)


def _wh(times: list[datetime], kind: str = "proclamation", n_nom: int = 0) -> pl.DataFrame:
    return pl.DataFrame({"guid": [str(i) for i in range(len(times))], "pub_ts": times,
                         "pub_date_et": [t.astimezone(E.ET).date() for t in times], "kind": [kind] * len(times),
                         "n_nom": [n_nom] * len(times)}, schema_overrides={"pub_ts": pl.Datetime("us", "UTC")})


def test_feed_window_and_nowcast_is_as_of():
    # one item every Tuesday 20:00 UTC for 14 weeks, ending Tue 2026-11-17
    tue = datetime(2026, 11, 17, 20, tzinfo=UTC)
    times = [tue - timedelta(days=7 * w) for w in range(14)]
    feeds = {m: E.Feed(_wh(times), m) for m in ("all", "nom", "eo")}
    fr = pl.DataFrame(schema={"signing_date": pl.Date, "publication_date": pl.Date})
    p0, p1 = date(2026, 11, 15), date(2026, 11, 21)
    # Monday: nothing published yet in the week; every analog week had exactly 1 item after Monday
    nc = E.nowcast("KXTRUMPACT", p0, p1, datetime(2026, 11, 16, 15, tzinfo=UTC), 1, math.inf, feeds, fr, set())
    assert nc["partial"] == 0 and not nc["decided"] and nc["p_model"] > 0.9
    assert E.nowcast("KXTRUMPACT", p0, p1, datetime(2026, 11, 16, 15, tzinfo=UTC), 2, math.inf, feeds, fr, set())["p_model"] < 0.1
    # Wednesday: the Tuesday item is public, so ">= 1" is decided, and the item is not visible before its time
    nc = E.nowcast("KXTRUMPACT", p0, p1, datetime(2026, 11, 18, 12, tzinfo=UTC), 1, math.inf, feeds, fr, set())
    assert nc["decided"] and nc["p_model"] == 1.0
    assert feeds["all"].window(datetime(2026, 11, 15, 5, tzinfo=UTC).timestamp(), tue.timestamp()) == 0.0
    assert feeds["all"].window(datetime(2026, 11, 15, 5, tzinfo=UTC).timestamp(), tue.timestamp() + 1) == 1.0


def _row(**kw) -> dict:
    base = {"ticker": "T-1", "event_ticker": "T", "series_ticker": "KXTRUMPACT", "decision": "f75", "split": "dev",
            "t": datetime(2026, 9, 1, 12, tzinfo=UTC), "t_next": datetime(2026, 9, 2, 12, tzinfo=UTC),
            "date": date(2026, 9, 1), "y": 0, "fee_type": "quadratic", "fee_multiplier": 1.0,
            "settle_ts": datetime(2026, 9, 7, tzinfo=UTC), "bid": 0.20, "ask": 0.24, "mid": 0.22,
            "w_lo": 0.18, "w_hi": 0.30, "w_vol": 500.0, "p_model": 0.05}
    return {**base, **kw}


def test_taker_fee_rounded_up_per_order_and_side():
    f = E.taker(pl.DataFrame([_row()]), "p_model", 0.03)
    assert f.height == 1 and f["side"][0] == "no"
    # NO at 1 - bid = 0.80, n = 2 contracts, fee = ceil(0.07*2*0.8*0.2*100)/100 = 0.03
    assert abs(f["price"][0] - 0.80) < 1e-9 and f["n"][0] == 2 and abs(f["fee"][0] - 0.03) < 1e-9
    assert abs(f["pnl_c"][0] - 100 * (1 - 0.80 - 0.015)) < 1e-6
    assert E.taker(pl.DataFrame([_row(p_model=0.22)]), "p_model", 0.03).height == 0


def test_maker_needs_strict_trade_through():
    # NO bid at floor(1 - 0.22 - 0.05) = 0.73 -> fills only if a later print is above 0.27
    f = E.maker(pl.DataFrame([_row()]), "mid", 0.05, ("no",), (0.02, 0.40))
    assert f.height == 1 and abs(f["price"][0] - 0.73) < 1e-9 and abs(f["pnl_c"][0] - 27.0) < 1e-6
    assert E.maker(pl.DataFrame([_row(w_hi=0.27)]), "mid", 0.05, ("no",), (0.02, 0.40)).height == 0
    assert E.maker(pl.DataFrame([_row(mid=0.55)]), "mid", 0.05, ("no",), (0.02, 0.40)).height == 0


def test_ledger_caps_event_exposure():
    rows = [_row(ticker=f"T-{i}", t=datetime(2026, 9, 1, 12 + i, tzinfo=UTC)) for i in range(5)]
    f = E.taker(pl.DataFrame(rows), "p_model", 0.03)
    led = E.ledger(f, "taker", "x", "dev")
    assert led["cost_usd"].sum() <= E.EVENT_CAP + 1e-9 and led.height < 5
    assert set(led.columns) >= {"struct_id", "event_ticker", "ticker", "side", "n", "t_in", "cost_usd", "settle_ts", "payoff_usd"}
