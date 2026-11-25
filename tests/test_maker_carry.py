"""maker_carry: family mapping, scheduled-close anchors, the maker-NO order rule (two-sided band,
post-only, cheapest NO first, 3 legs per event, strict through-fill after an hour-aligned decision),
fees, Holm, on trimmed copies of recorded Kalshi responses."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from albert.research import maker_carry as mc
from albert.research import xcat
from albert.venues import kalshi as k

FIX = Path(__file__).parent / "fixtures" / "maker_carry"


def _markets() -> pl.DataFrame:
    rec = json.loads((FIX / "historical_markets_KXHIGHNY.json").read_text())
    info = {"KXHIGHNY": {"category": "Climate and Weather", "fee_type": "quadratic", "fee_multiplier": 1.0}}
    return pl.DataFrame(k.market_rows(rec["response"], info, "historical"), infer_schema_length=None)


def _candles() -> tuple[str, pl.DataFrame]:
    rec = json.loads((FIX / "candles_leg.json").read_text())
    tk = rec["ticker"]
    return tk, pl.DataFrame(xcat.ohlc_rows(rec["response"], tk, 60), schema=xcat.OHLC_SCHEMA)


def test_family_mapping():
    assert mc.family_of("KXHIGHNY", "Climate and Weather", "Highest temperature in NYC") == "wx_daily"
    assert mc.family_of("KXLOWTCHI", "Climate and Weather", "Lowest Temperature Chicago") == "wx_daily"
    assert mc.family_of("KXRAINNYCM", "Climate and Weather", "Monthly rain in New York") == "wx_precip"
    assert mc.family_of("KXHURCAT", "Climate and Weather", "Hurricane category") == "wx_other"
    # minor vs major sports, esports and virtual excluded
    assert mc.family_of("KXATPCHALLENGERMATCH", "Sports", "ATP Challenger") == "sports_minor"
    assert mc.family_of("KXITFMATCH", "Sports", "ITF") == "sports_minor"
    assert mc.family_of("KXATPMATCH", "Sports", "ATP") is None
    assert mc.family_of("KXNFLGAME", "Sports", "NFL") is None
    assert mc.family_of("KXSERIEAGAME", "Sports", "Serie A") is None
    assert mc.family_of("KXSERIECGAME", "Sports", "Serie C") == "sports_minor"
    assert mc.family_of("KXCS2MAP", "Sports", "CS2") is None
    assert mc.family_of("KXESOCCERGAME", "Sports", "eSoccer") is None
    # cmn series, Mentions, financial and crypto out; other categories in
    assert mc.family_of("KXTOPSONG", "Entertainment", "Top song") is None
    assert mc.family_of("KXTRUMPMENTION", "Mentions", "Trump mention") is None
    assert mc.family_of("KXFEDDECISION", "Politics", "Fed decision") is None
    assert mc.family_of("KXBTCD", "Crypto", "Bitcoin") is None
    assert mc.family_of("KXOSCARPIC", "Entertainment", "Oscar best picture") == "entertainment"
    assert mc.family_of("KXNEXTPOPE", "World", "Next Pope") == "politics_world"
    assert mc.family_of("KXGOVNY", "Elections", "NY governor") == "elections"
    assert len(mc.CMN_SERIES) == 37


def test_prices_round_up_to_the_cent():
    assert mc.sell_price(0.025, 12) == pytest.approx(0.15)
    assert mc.sell_price(0.03, 12) == pytest.approx(0.15)
    assert mc.sell_price(0.035, 6) == pytest.approx(0.10)
    assert mc.cents_up(0.1500000001) == pytest.approx(0.15)


def test_anchor_is_scheduled_close_not_actual_close():
    m = _markets().with_columns(pl.lit("wx_daily").alias("family"), pl.lit("Climate and Weather").alias("category"))
    base = m.row(0, named=True)
    rows = []
    # six recurring events of the series, each closing 14 h before its expected expiration
    for d in range(6):
        for leg in ("A", "B"):
            close = base["close_time"] + timedelta(days=d)
            rows.append({**base, "ticker": f"KXHIGHNY-X{d}-{leg}", "event_ticker": f"KXHIGHNY-X{d}",
                         "close_time": close, "expected_expiration_time": close + timedelta(hours=14),
                         "open_time": close - timedelta(hours=39)})
    # a seventh event that closed 20 h early: its anchor must still be the scheduled close
    sched = base["close_time"] + timedelta(days=6)
    rows.append({**base, "ticker": "KXHIGHNY-X6-A", "event_ticker": "KXHIGHNY-X6", "close_time": sched - timedelta(hours=20),
                 "expected_expiration_time": sched + timedelta(hours=14), "open_time": sched - timedelta(hours=39)})
    ev = mc.anchors(pl.DataFrame(rows))
    early = ev.filter(pl.col("event_ticker") == "KXHIGHNY-X6").row(0, named=True)
    assert early["anchor"] == sched
    assert early["early_h"] == pytest.approx(20)
    assert ev.filter(pl.col("event_ticker") == "KXHIGHNY-X0")["anchor"][0] == base["close_time"]


def test_sports_anchor_is_ticker_start():
    m = _markets().with_columns(pl.lit("sports_minor").alias("family"), pl.lit("Sports").alias("category"),
                                pl.lit("KXKBOGAME-26SEP040430SAMKTW").alias("event_ticker"))
    ev = mc.anchors(m)
    assert ev["anchor"][0] == datetime(2026, 9, 4, 8, 30, tzinfo=UTC)
    assert ev["anchor_src"][0] == "ticker_start"


def _legs(tk: str, c: pl.DataFrame, t: datetime, end: datetime, fam: str = "wx_daily") -> pl.DataFrame:
    return pl.DataFrame([{"ticker": tk, "event_ticker": tk.rsplit("-", 1)[0], "series_ticker": tk.split("-")[0],
                          "family": fam, "new": False, "in_base": True, "split": "dev", "anchor": t + timedelta(hours=24),
                          "open_time": t - timedelta(days=3), "close_time": end, "end": end,
                          "settlement_ts": end + timedelta(hours=14), "y": 0, "fee_type": "quadratic",
                          "fee_multiplier": 1.0}])


def test_decision_quote_and_through_fill_on_recorded_candles():
    tk, c = _candles()
    two = c.filter((pl.col("bid_close") >= 0.01) & (pl.col("ask_close") <= 0.99) & (pl.col("ask_close") > pl.col("bid_close"))
                   & ((pl.col("bid_close") + pl.col("ask_close")) / 2 <= 0.05))
    assert two.height, "fixture must hold a cheap two-sided hour"
    t = two["end_ts"][0]
    end = c["end_ts"].max()
    legs = _legs(tk, c, t, end).with_columns(pl.lit(t + timedelta(hours=24)).alias("anchor"))
    q = mc.decision_quotes(legs, c).filter(pl.col("h") == 24)
    assert q.height == 1 and q["q_ts"][0] <= q["t"][0]
    r = q.row(0, named=True)
    assert r["two_sided"] and r["mid"] <= 0.05
    o = mc.build_orders(q, mc.Tape(c))
    assert set(o["m"].to_list()) <= set(mc.MARGINS_C)
    for row in o.iter_rows(named=True):
        s = row["s"]
        assert s > row["bid"] and s < 0.99
        after = c.filter((pl.col("end_ts") > row["t"]) & (pl.col("end_ts") <= row["fill_end"]) & (pl.col("p_high") > s + 1e-9))
        if after.height:
            assert row["fill_ts"] == after["end_ts"][0]
            assert row["pnl_c"] == pytest.approx(100 * (1 - row["price"]))   # y = 0, no maker fee
        else:
            assert row["fill_ts"] is None and row["pnl_c"] is None


def test_strict_through_and_hour_alignment():
    tk, c = _candles()
    tape = mc.Tape(c)
    traded = c.filter(pl.col("p_high").is_not_null()).sort("end_ts")
    first = traded.row(0, named=True)
    hi = first["p_high"]
    t0 = first["end_ts"] - timedelta(hours=1)
    # a print exactly at the price does not fill; one cent below does, in that candle
    got = tape.first_through(tk, t0, first["end_ts"], hi)
    assert got is None
    got = tape.first_through(tk, t0, first["end_ts"], round(hi - 0.01, 2))
    assert got is not None and got[0] == first["end_ts"]
    # the candle ending at the decision time is not after it
    assert tape.first_through(tk, first["end_ts"], first["end_ts"], 0.0) is None


def test_fill_window_end():
    r = {"family": "sports_minor", "end": datetime(2026, 10, 3, 19, 5, tzinfo=UTC)}
    assert mc.fill_end(r) == datetime(2026, 10, 3, 19, 0, tzinfo=UTC)     # start hour is in-play
    r = {"family": "wx_daily", "end": datetime(2026, 10, 3, 4, 59, 58, tzinfo=UTC)}
    assert mc.fill_end(r) == datetime(2026, 10, 3, 5, 0, tzinfo=UTC)
    r = {"family": "wx_daily", "end": datetime(2026, 10, 3, 5, 0, tzinfo=UTC)}
    assert mc.fill_end(r) == r["end"]


def test_three_legs_cheapest_no_first_and_maker_fee():
    tk, c = _candles()
    two = c.filter((pl.col("bid_close") >= 0.01) & (pl.col("ask_close") <= 0.99) & (pl.col("ask_close") > pl.col("bid_close"))
                   & ((pl.col("bid_close") + pl.col("ask_close")) / 2 <= 0.05))
    t = two["end_ts"][0]
    q = mc.decision_quotes(_legs(tk, c, t, c["end_ts"].max()), c).filter(pl.col("h") == 24)
    r = q.row(0, named=True)
    rows = []
    for i, (bid, ask) in enumerate([(0.01, 0.02), (0.02, 0.04), (0.03, 0.07), (0.01, 0.09), (0.04, 0.06)]):
        rows.append({**r, "ticker": f"{tk}-L{i}", "bid": bid, "ask": ask, "mid": (bid + ask) / 2})
    qq = pl.DataFrame(rows).with_columns(pl.lit("quadratic_with_maker_fees").alias("fee_type"))
    o = mc.build_orders(qq, mc.Tape(c)).filter(pl.col("m") == 12)
    assert o.height == 3
    # cheapest NO first: the three 5c-mid legs (NO at 83c) are kept, the 3c and 1.5c legs (85c, 86c) are not
    assert sorted(o["price"].to_list()) == o.sort("rank")["price"].to_list()
    assert o["price"].max() <= 0.85 + 1e-9
    assert (o["fee"] == 0.01).all()      # 0.0175 x p(1-p) rounds up to 1c per 1-lot order


def test_holm_and_p():
    adj = mc.holm({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.04) and adj["c"] == pytest.approx(0.5)
    assert mc.p_one_sided(1.645) == pytest.approx(0.05, abs=1e-3)
    assert mc.p_one_sided(None) == 1.0


def test_election_day_anchor():
    assert mc.election_day("KXPARAIBAGOV-26OCT04") == datetime(2026, 10, 4, 10, tzinfo=UTC)
    assert mc.election_day("KXHOUSERACE-GA05-26") == datetime(2026, 11, 3, 10, tzinfo=UTC)
    assert mc.election_day("KXVOTEGENERAL-KXHOUSERACE-FL07-26RELI") == datetime(2026, 11, 3, 10, tzinfo=UTC)
    assert mc.election_day("KXNEXTROMANIAPM") is None


def test_exact_few_yes_p():
    # two fills at NO 85c (break-even YES 15%), neither lost: P(0 YES) = 0.85^2
    f = pl.DataFrame({"price": [0.85, 0.85], "fee": [0.0, 0.0], "y": [0, 0], "pnl_c": [15.0, 15.0]})
    assert mc.p_few_yes(f) == pytest.approx(0.85 ** 2)
    f = f.with_columns(pl.Series("y", [1, 0]))
    assert mc.p_few_yes(f) == pytest.approx(1 - 0.15 ** 2)     # P(at most one YES)
    assert mc.p_few_yes(f.filter(pl.col("y") == 2)) is None
