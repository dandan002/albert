"""Parsers and helpers of albert.research.netflix, tested on trimmed copies of recorded responses."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest

from albert.research import netflix as nf

FIX = Path(__file__).parent / "fixtures"


def test_parse_won_trimmed_page():
    text = (FIX / "whatsonnetflix" / "most-popular_2026-11-20_trimmed.html").read_text()
    rows = nf.parse_won(text)
    assert rows, "a parser that returns zero rows is a failure"
    assert nf.page_date(text) == "2026-11-20"
    glob_rows = [r for r in rows if r["scope"] == "global"]
    assert {(r["kind"], r["rank"]) for r in glob_rows} == {("movies", 1), ("tv", 1), ("movies", 2), ("tv", 2),
                                                          ("movies", 3), ("tv", 3)}
    top_tv = next(r for r in glob_rows if r["kind"] == "tv" and r["rank"] == 1)
    assert top_tv["title"] == "The Body" and top_tv["points"] == 786.0 and top_tv["list_date"] == "2026-11-20"
    us = [r for r in rows if r["scope"] == "US"]
    assert [(r["kind"], r["rank"], r["title"]) for r in us if r["kind"] == "tv"] == [
        ("tv", 1, "The Body"), ("tv", 2, "Taken: Natalee Holloway"), ("tv", 3, "From")]
    assert {r["kind"] for r in us} == {"tv", "movies"}
    assert all(r["points"] is None for r in us)
    assert {r["scope"] for r in rows} == {"global", "US", "UK"}


@pytest.mark.parametrize("name,key", [
    ("Taken: Natalee Holloway: Season 1", "taken natalee holloway"),
    ("Below: Limited Series", "below"),
    ("Love Is Blind: S11: Boston", "love is blind"),
    ("Black Doves Season 2", "black doves"),
    ("Dr. Seuss&#039; The Grinch", "dr seuss the grinch"),
    ("KPop Demon Hunters", "kpop demon hunters"),
])
def test_norm_title(name, key):
    assert nf.norm_title(name) == key


def test_match_key_prefix_and_exact():
    keys = {"raw", "stranger things", "from", "the body"}
    assert nf.match_key("Raw: 2026 - November 2, 2026", keys) == "raw"
    assert nf.match_key("Stranger Things 5", keys) == "stranger things"
    assert nf.match_key("The Body: Season 1", keys) == "the body"
    assert nf.match_key("Frontier", keys) is None


def test_pl_probs_are_distributions():
    u = np.array([2.0, 1.0, 0.5, -1.0])
    for rank in (1, 2):
        p = nf.pl_probs(u, rank)
        assert p.sum() == pytest.approx(1.0)
        assert (p >= 0).all()
    assert nf.pl_probs(u, 1).argmax() == 0
    # the strongest title is less likely to be second than the runner-up
    p2 = nf.pl_probs(u, 2)
    assert p2[1] > p2[3]


def test_timing_helpers():
    close = datetime(2026, 11, 17, 4, 59, tzinfo=UTC)          # Mon Nov 16 23:59 ET
    assert nf.week_of_close(close) == (date(2026, 11, 9), date(2026, 11, 15))
    assert nf.decision_time(close, "mon18") == datetime(2026, 11, 16, 23, 0, tzinfo=UTC)
    assert nf.decision_time(close, "sun18") == datetime(2026, 11, 15, 23, 0, tzinfo=UTC)
    # the archived list labelled D is public from D+1 01:00 UTC
    assert nf.last_public_label(datetime(2026, 11, 17, 0, 59, tzinfo=UTC)) == date(2026, 11, 15)
    assert nf.last_public_label(datetime(2026, 11, 17, 1, 0, tzinfo=UTC)) == date(2026, 11, 16)
    # Mon 18:00 ET: Sunday's list is the latest one
    assert nf.last_public_label(nf.decision_time(close, "mon18")) == date(2026, 11, 15)


def test_order_fee_rounds_up_per_order():
    assert nf.order_fee(0.50, 1) == 0.02            # 0.0175 -> 0.02
    assert nf.order_fee(0.50, 10) == 0.18           # 0.175 -> 0.18
    assert nf.order_fee(0.95, 2) == 0.01


def test_candle_rows_on_recorded_payload():
    rec = json.loads((FIX / "kalshi" / "netflix_candles_historical_trimmed.json").read_text())
    rows = nf.candle_rows(rec["response"], rec["ticker"])
    assert len(rows) == 4
    last = rows[-1]
    assert last["ticker"] == "KXNETFLIXRANKSHOW-26SEP14-CRE"
    assert last["yes_ask_close"] == 1.0 and last["yes_bid_close"] == 0.0
    assert last["end_ts"] == datetime.fromtimestamp(1789444800, tz=UTC)


def test_market_row_on_recorded_payload():
    rec = json.loads((FIX / "kalshi" / "netflix_historical_markets_trimmed.json").read_text())
    rows = [nf.market_row(m, "KXNETFLIXRANKSHOWGLOBAL2", "hist") for m in rec["response"]["markets"]]
    assert rows and rows[0]["ticker"] == "KXNETFLIXRANKSHOWGLOBAL2-26SEP07-EAR"
    assert rows[0]["result"] == "no" and rows[0]["expiration_value"] == "Beauty in Black: Season 3"
    assert rows[0]["volume"] == 0.0


def test_vectorized_loglik_matches_pl_probs():
    rng = np.random.default_rng(3)
    groups = []
    for rank in (1, 2, 1, 2):
        n = int(rng.integers(3, 7))
        X = rng.random((n, len(nf.FEATURES)))
        wv = np.zeros(n, dtype=bool)
        wv[int(rng.integers(0, n))] = True
        groups.append((X, wv, rank))
    b = np.array([2.0, 3.0, -1.0, 1.5])
    got = nf._loglik(b, *nf._padded(groups))
    want = [np.log(nf.pl_probs(X @ b, rk)[wv].sum()) for X, wv, rk in groups]
    assert np.allclose(got, want)


def test_take_sizes_and_fees():
    row = {"ask": 0.95, "bid": 0.93, "y": 1, "fee_multiplier": 1.0, "event": "E", "ticker": "T", "series": "S",
           "decision": "sun18", "t": datetime(2026, 11, 15, 23, tzinfo=UTC), "settle": datetime(2026, 11, 17, 21, tzinfo=UTC),
           "close": datetime(2026, 11, 17, 4, 59, tzinfo=UTC), "split": "holdout"}
    leg = nf.take(row, "yes")
    assert leg["n"] == 2 and leg["fee"] == 0.01
    assert leg["pnl"] == pytest.approx(2 * (1 - 0.95) - 0.01)
    no = nf.take(row, "no")                       # NO at 1 - bid = 0.07, loses
    assert no["price"] == pytest.approx(0.07) and no["pnl"] < 0
