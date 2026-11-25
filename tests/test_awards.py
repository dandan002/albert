"""Tests for ``albert.research.awards``: universe classification, precursor matching, quotes, fills, books.

Fixtures in ``tests/fixtures/awards/`` are trimmed copies of recorded Kalshi responses (``/historical/markets``
for Golden Globe and Oscar series, and hourly ``candlesticks`` of one leg)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from albert.research import awards as aw
from albert.research.counts import MARKET_COLS, market_row
from albert.research.xcat import OHLC_SCHEMA, ohlc_rows

FIX = Path(__file__).parent / "fixtures" / "awards"


def _markets(name: str) -> pl.DataFrame:
    rec = json.loads((FIX / name).read_text())
    rows = [market_row(m, rec["series_ticker"], "hist") for m in rec["response"]["markets"]]
    mk = pl.DataFrame(rows, schema=MARKET_COLS)
    return mk.with_columns(pl.col("series_ticker").map_elements(aw.family_of, return_dtype=pl.String).alias("family"),
                           (pl.col("result") == "yes").cast(pl.Int8).alias("y"))


def test_family_classification() -> None:
    assert aw.family_of("KXOSCARACTO") == "oscar"
    assert aw.family_of("OSCARPIC") == "oscar"
    assert aw.family_of("KXOSCARNOMACTO") is None          # nominations are not winner-take-all
    assert aw.family_of("KXOSCARCOUNTANORA") is None
    assert aw.family_of("KXGGDRAMAFILMACTO") == "gg"
    assert aw.family_of("KXGGGNOMDIR") is None
    assert aw.family_of("KXGRAMMYCOUNTSWIFT") is None
    assert aw.family_of("KXNOBELPEACE") == "nobel"
    assert aw.family_of("KXMLBWSMVP") is None               # in-game MVPs are excluded
    assert aw.family_of("KXMLBALCY") == "sports"


def test_category_keys_link_precursors() -> None:
    assert aw.category_keys("KXGGDRAMAFILMACTO", "gg") == ["actor"]
    assert aw.category_keys("KXOSCARACTO", "oscar") == ["actor"]
    assert aw.category_keys("KXSAGAWARDSUPACTR", "sag") == ["sup_actress"]
    assert aw.category_keys("KXBAFTAFILM", "bafta") == ["picture"]
    assert aw.category_keys("KXGGSCREEN", "gg") == ["orig_screenplay", "adapt_screenplay"]
    assert aw.category_keys("KXGGLSERIES", "gg") == ["tv_limited"]
    assert aw.category_keys("KXEMMYDSERIES", "emmy") == ["tv_drama"]
    assert aw.category_keys("KXDGAAWARDS", "guild") == ["director"]
    assert aw.category_keys("KXGRAMAOTY", "grammy") == []


def test_names_match() -> None:
    assert aw.norm_name("Timothée Chalamet") == "timothee chalamet"
    assert aw.names_match(aw.norm_name("Timothée Chalamet"), aw.norm_name("Timothee Chalamet"))
    assert aw.names_match(aw.norm_name("The Brutalist"), aw.norm_name("Brutalist"))
    assert aw.names_match(aw.norm_name("Adrien Brody (The Brutalist)"), aw.norm_name("Adrien Brody"))
    assert not aw.names_match(aw.norm_name("Tie"), aw.norm_name("Tie"))
    assert not aw.names_match(aw.norm_name("Emma Stone"), aw.norm_name("Emma Thompson"))


def test_holm() -> None:
    p = {"a": 0.001, "b": 0.02, "c": 0.04, "d": None}
    out = aw.holm(p)
    assert out == {"a": True, "b": False, "c": False, "d": False}


def test_capped_book_respects_caps() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [{"ticker": f"T{i}", "event_ticker": f"E{i // 3}", "t": t0 + timedelta(hours=i), "t_in": t0 + timedelta(hours=i),
             "cost_usd": 2.0, "price": 0.5, "settlement_ts": t0 + timedelta(days=10), "close_time": t0 + timedelta(days=10)}
            for i in range(40)]
    b = aw.capped_book(pl.DataFrame(rows), cash=10.0, event_cap=5.0)
    assert b.height == 5                    # $10 / $2, at most 2 per event ($5 cap)
    assert b.group_by("event_ticker").len()["len"].max() == 2


def test_precursor_pairs_from_recorded_markets() -> None:
    mk = pl.concat([_markets("gg_drama_actor_26.json"), _markets("oscar_actor_26.json")])
    mk = mk.with_columns(pl.col("family").replace_strict(aw.GROUP).alias("group"))
    assert mk.height == 32
    ev = aw.events_table(mk)
    assert ev.filter(pl.col("event_ticker") == "KXOSCARACTO-26")["n_yes"].item() == 1
    pairs = aw.precursor_pairs(mk, ev)
    assert set(pairs["p_event"]) == {"KXGGDRAMAFILMACTO-26"} and set(pairs["f_event"]) == {"KXOSCARACTO-26"}
    win = pairs.filter(pl.col("p_y") == 1)
    assert win.height == 1 and win["f_ticker"].item() == "KXOSCARACTO-26-WAG"     # Wagner Moura, GG winner
    assert (pairs["p_close"] < pairs["f_close"]).all()
    assert pairs.filter(pl.col("f_name") == "tie").height == 0


def test_breakeven_p_is_exact_poisson_binomial() -> None:
    f = pl.DataFrame({"price": [0.9] * 20, "fee": [0.01] * 20, "n": [1] * 20, "payoff": [1] * 20})
    assert aw.breakeven_p(f) == pytest.approx(0.91 ** 20 * 0 + (1 - 0.09) ** 20)
    g = f.with_columns(pl.Series("payoff", [1] * 15 + [0] * 5))
    assert aw.breakeven_p(g) > 0.9          # 5 losses against 1.8 expected: no evidence of edge


def _candles() -> pl.DataFrame:
    rec = json.loads((FIX / "candles_gg_drama_film_26_sen.json").read_text())
    rows = ohlc_rows(rec["response"], rec["ticker"], 60)
    assert rows, "fixture parsed to zero candles"
    return pl.DataFrame(rows, schema=OHLC_SCHEMA)


def _point(t: datetime, h: int = 24) -> pl.DataFrame:
    return pl.DataFrame({"ticker": ["KXGGDRAMAFILM-26-SEN"], "event_ticker": ["KXGGDRAMAFILM-26"], "t": [t], "h": [h],
                         "close_time": [datetime(2026, 1, 12, 4, 0, tzinfo=UTC)], "y": [0], "fee_type": ["quadratic"],
                         "fee_multiplier": [1.0]})


def test_quotes_at_uses_last_candle_before_t() -> None:
    c = _candles()
    q = aw.quotes_at(_point(datetime(2026, 1, 11, 12, 30, tzinfo=UTC)), c)
    assert q["end_ts"].item() == datetime(2026, 1, 11, 12, 0, tzinfo=UTC)
    assert q["bid"].item() == pytest.approx(0.08) and q["ask"].item() == pytest.approx(0.09)
    assert q["mid"].item() == pytest.approx(0.085) and q["two_sided"].item()
    stale = aw.quotes_at(_point(datetime(2026, 1, 25, tzinfo=UTC)), c)       # > 7 days after the last candle
    assert stale["mid"].item() is None and not stale["two_sided"].item()


def test_maker_fill_needs_a_later_print_through_the_price() -> None:
    c = _candles()
    panel = aw.quotes_at(_point(datetime(2026, 1, 11, 12, 30, tzinfo=UTC)), c)
    o = aw.maker_rows(panel, c, 24, (0.05, 0.15), 0.06)
    assert o.height == 1
    r = o.row(0, named=True)
    assert r["sell_at"] == pytest.approx(0.15) and r["price"] == pytest.approx(0.85)
    assert r["fill_ts"] == datetime(2026, 1, 11, 19, 0, tzinfo=UTC)          # first hour with a print above 15c
    assert r["pnl_c"] == pytest.approx(15.0)                                    # leg lost: NO pays 1
    assert aw.maker_rows(panel, c, 24, (0.0, 0.05), 0.12).height == 0          # mid 8.5c is outside the band
    late = aw.quotes_at(_point(datetime(2026, 1, 12, 3, 30, tzinfo=UTC)), c)    # 03:00 candle: bid 5c ask 10c
    o2 = aw.maker_rows(late, c, 24, (0.05, 0.15), 0.12)
    assert o2.height == 1 and o2["fill_ts"].item() is None                     # 04:00 high 10c < 20c: no fill
