"""Behavioural politics study: frozen labels on recorded Kalshi markets, election-day anchors,
as-of quotes without look-ahead, taker fees rounded per order, and the capped ledger."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from albert.research import behav_politics as bp
from albert.venues.kalshi import MARKET_SCHEMA, market_rows

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _markets() -> pl.DataFrame:
    payload = json.loads((FIX / "behav_politics_historical_markets.json").read_text())
    info = {"KXVOTEHEGSETH": {"category": "Politics"}, "PRESPARTYWI": {"category": "Elections"},
            "SENATEOH": {"category": "Elections"}, "KXPRESCHILE": {"category": "Elections"},
            "KXLEADEROUT": {"category": "Elections"}, "KXSENATEKYR": {"category": "Elections"}}
    return pl.DataFrame(market_rows(payload, info, "historical"), schema=MARKET_SCHEMA)


@pytest.fixture(scope="module")
def labelled() -> dict[str, dict]:
    m = bp.label_markets(_markets())
    assert m.height == 11
    return {r["ticker"]: r for r in m.iter_rows(named=True)}


# ---- frozen labels on recorded markets -------------------------------------------------------------
def test_party_line_follows_member_party(labelled):
    assert labelled["KXVOTEHEGSETH-26-JF"]["party_line"] == "no"      # Fetterman (D) on a Trump nominee
    assert labelled["KXVOTEHEGSETH-26-JL"]["party_line"] == "yes"     # Lankford (R)
    assert labelled["KXVOTEHEGSETH-26-TT"]["party_line"] == "yes"     # Tillis (R)


def test_partisan_lean_on_general_election_legs(labelled):
    assert labelled["PRESPARTYWI-24-R"]["lean"] == "R" and labelled["PRESPARTYWI-24-D"]["lean"] == "D"
    assert labelled["SENATEOH-24-R"]["lean"] == "R" and labelled["SENATEOH-24-D"]["lean"] == "D"
    assert labelled["PRESPARTYWI-24-R"]["ekind"] == "general"
    # a same-party primary has no lean
    assert labelled["KXSENATEKYR-26-SJEN"]["ekind"] == "primary" and labelled["KXSENATEKYR-26-SJEN"]["lean"] is None


def test_foreign_election_and_populist_list(labelled):
    kast, parisi = labelled["KXPRESCHILE-25-JKAS"], labelled["KXPRESCHILE-25-FPAR"]
    assert kast["ekind"] == parisi["ekind"] == "foreign"
    assert kast["populist"] is True and parisi["populist"] is False
    assert kast["lean"] is None


def test_drama_lexicon_matches_leader_out_rules(labelled):
    leader = labelled["KXLEADEROUT-26-AKHA"]
    assert leader["drama"] is True and leader["ekind"] is None
    assert bp.drama("Will Trump fire Jerome Powell?", None)
    assert not bp.drama("Will the Senate confirm Kevin Warsh?", None)
    assert bp.procedural("Will the Senate confirm Kevin Warsh?", None)
    assert bp.default_yes("Will the Senate confirm Kevin Warsh?", None)


def test_member_party_and_vote_direction():
    assert bp.member_party("Lisa Blunt Rochester") == "D"
    assert bp.member_party("Cindy Hyde-Smith") == "R"
    assert bp.member_party("Rick Scott") == "R"
    assert bp.member_party("Anna Paulina Luna") is None
    assert bp.vote_party_line("KXVOTEX", "Will Susan Collins vote against the bill?", None) == "no"
    assert bp.vote_party_line("KXOTEEPSTEIN", "Will Andy Ogles vote for releasing the Epstein files?", None) is None


def test_approval_and_generic_ballot_direction():
    late = datetime(2026, 9, 1, tzinfo=UTC)
    assert bp.approval_lean("KXTRUMPAPPROVE", "Will Trump's RCP approval average be above 40.2 on Sep 7, 2026?",
                            None, late) == "R"
    assert bp.approval_lean("KXTRUMPAPPROVE", "Will Trump's RCP approval average be below 39.3 on Sep 3, 2026?",
                            None, late) == "D"
    assert bp.approval_lean("KXVHGCB", "Will the VoteHub 2026 generic ballot margin (D − R, time-weighted) be below 8.2?",
                            None, late) == "R"


# ---- election-day anchors ----------------------------------------------------------------------------
def test_ticker_date_and_call_date():
    assert bp.ticker_date("KXSANTACATARINA-26OCT04") == date(2026, 10, 4)
    assert bp.ticker_date("KXILPRIMARY-17R26") is None
    # a call at 01:30 ET belongs to the previous election night; one at 22:00 ET to the same day
    assert bp._call_date(datetime(2026, 3, 18, 5, 30, tzinfo=UTC)) == date(2026, 3, 17)
    assert bp._call_date(datetime(2026, 3, 18, 2, 0, tzinfo=UTC)) == date(2026, 3, 17)


def test_election_dates_general_and_state_mode(labelled):
    m = bp.label_markets(_markets())
    ed = {r["event_ticker"]: r for r in bp.election_dates(m).iter_rows(named=True)}
    assert ed["PRESPARTYWI-24"]["e_date"] == date(2024, 11, 5) and ed["PRESPARTYWI-24"]["e_src"] == "general"
    assert ed["SENATEOH-24"]["e_date"] == date(2024, 11, 5)
    # a primary needs >= 3 same-state events for the modal date: none here
    assert ed["KXSENATEKYR-26"]["e_date"] is None
    # three Kentucky primary events closing on the same night give that night to a slow fourth one
    rows = []
    for i, close in enumerate(["2026-05-20T01:10:00Z", "2026-05-20T02:30:00Z", "2026-05-20T03:00:00Z",
                               "2026-05-23T15:00:00Z"]):
        rows.append({"ticker": f"KXKYPRIMARY-0{i}R26-A", "event_ticker": f"KXKYPRIMARY-0{i}R26",
                     "series_ticker": "KXKYPRIMARY", "ekind": "primary", "title": f"Will A be the Republican nominee for KY-0{i}?",
                     "yes_sub_title": "A", "rules_primary": "If A wins the nomination for the Republican Party to "
                     f"contest the 2026 KY-0{i} House seat, then the market resolves to Yes.",
                     "close_time": datetime.fromisoformat(close)})
    e = bp.election_dates(pl.DataFrame(rows))
    assert e["e_date"].to_list() == [date(2026, 5, 19)] * 4
    assert set(e["e_src"].to_list()) == {"state_mode"}


# ---- quotes, fees, ledger -------------------------------------------------------------------------
def _candles() -> pl.DataFrame:
    t0 = datetime(2026, 9, 1, 12, tzinfo=UTC)
    return pl.DataFrame({"ticker": ["A"] * 3, "end_ts": [t0, t0 + timedelta(hours=1), t0 + timedelta(hours=2)],
                         "bid": [0.40, 0.50, 0.90], "ask": [0.44, 0.52, 0.95], "price": [0.42, 0.51, 0.93],
                         "volume": [1.0, 1.0, 1.0], "oi": [1.0, 1.0, 1.0]}, schema=bp.CANDLE_SCHEMA)


def test_quotes_use_only_candles_ended_by_t():
    c = _candles()
    t = datetime(2026, 9, 1, 13, 59, tzinfo=UTC)          # the 14:00 candle has not ended yet
    q = bp.quotes_at(c, pl.DataFrame({"ticker": ["A"], "t": [t]}))
    assert q["bid"][0] == 0.50 and q["ask"][0] == 0.52 and q["q_ts"][0] == datetime(2026, 9, 1, 13, tzinfo=UTC)
    q = bp.quotes_at(c, pl.DataFrame({"ticker": ["A"], "t": [datetime(2026, 9, 1, 11, tzinfo=UTC)]}))
    assert q["q_ts"][0] is None


def test_candle_rows_parse_recorded_payload():
    payload = json.loads((FIX / "behav_politics_candles_historical.json").read_text())
    rows = bp.candle_rows(payload, payload["ticker"])
    assert len(rows) >= 10
    df = pl.DataFrame(rows, schema=bp.CANDLE_SCHEMA)
    two = df.filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null())
    assert two.height and ((two["ask"] >= two["bid"]) & (two["ask"] <= 1) & (two["bid"] >= 0)).all()
    assert df["end_ts"].is_sorted()


def test_taker_fee_rounds_up_per_order():
    df = pl.DataFrame({"side": ["yes", "no"], "bid": [0.48, 0.48], "ask": [0.50, 0.50], "mid": [0.49, 0.49],
                       "y": [1, 1], "fee_multiplier": [1.0, 1.0]})
    t = bp.take(df, stake=2.5)
    yes, no = t.row(0, named=True), t.row(1, named=True)
    assert yes["n"] == 5 and yes["fee_order"] == pytest.approx(0.09)        # 0.0875 -> 9c
    assert yes["pnl"] == pytest.approx(1 - 0.50 - 0.09 / 5)
    assert no["price"] == pytest.approx(0.52) and no["pnl"] == pytest.approx(-0.52 - no["fee_order"] / no["n"])


def test_capped_ledger_respects_event_cap_and_releases_capital():
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    tr = pl.DataFrame({"t": [t0, t0 + timedelta(hours=1), t0 + timedelta(days=3)],
                       "ticker": ["A", "B", "C"], "event_ticker": ["E", "E", "E"], "side": ["no"] * 3,
                       "n": [8, 8, 8], "price": [0.5, 0.5, 0.5], "payoff": [1, 1, 0],
                       "settle_ts": [t0 + timedelta(days=2)] * 2 + [t0 + timedelta(days=4)], "fee_multiplier": [1.0] * 3})
    led = bp.capped_ledger(tr, "s", "v", "dev", event_cap=5.0)
    assert led["cost_usd"].max() <= 5.0 + 1e-9
    assert led.height == 3                       # B is shrunk to the room left; C fits after settlement
    assert led.filter(pl.col("ticker") == "A")["cost_usd"][0] + led.filter(pl.col("ticker") == "B")["cost_usd"][0] <= 5.0
