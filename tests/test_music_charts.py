"""Tests for albert.research.music_charts (fixtures are trimmed copies of recorded responses)."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import music_charts as mc

FIX = Path(__file__).parent / "fixtures" / "music_charts"


def test_parse_hits_final_uses_the_posts_own_timestamp_and_table():
    raw = (FIX / "hits_chart_final_trimmed.html").read_text()
    d = mc.parse_hits_post(raw, "chart-final-parade-rout-2026-10-30")
    assert d["published_at"] == "2026-10-30T20:45:00.000Z"        # not the sidebar post's stamp
    assert d["kind"] == "final"
    assert d["title"] == "Chart Final: Parade Rout"
    top = d["items"][0]
    assert (top["artist"], top["album"]) == ("MY CHEMICAL ROMANCE", "THE BLACK PARADE")
    assert top["activity"] == 70270 and top["albums"] == 61078 and top["sea"] == 9087 and top["this_week"] == 1
    assert len(d["items"]) == 3


def test_parse_hits_preview_has_no_sea():
    d = mc.parse_hits_post((FIX / "hits_chart_preview_trimmed.html").read_text(), "chart-preview-marching-orders-2026-10-27")
    assert d["kind"] == "preview" and d["published_at"] == "2026-10-27T20:38:18.167Z"
    assert d["items"][0]["activity"] == 65000 and d["items"][0]["albums"] == 56000 and d["items"][0]["sea"] is None
    assert mc.hits_value(d["items"][0], "streams") == 9000          # Activity - Albums proxy
    assert mc.hits_value(d["items"][0], "pure") == 56000


def test_week_end_for_post():
    fri = datetime(2026, 10, 30, 20, 45, tzinfo=UTC)               # Friday final -> week ended Thu Oct 29
    assert mc.week_end_for_post(fri, "final") == date(2026, 10, 29)
    tue = datetime(2026, 10, 27, 20, 38, tzinfo=UTC)               # Tuesday preview -> week ending Thu Oct 29
    assert mc.week_end_for_post(tue, "preview") == date(2026, 10, 29)
    sun = datetime(2026, 9, 6, 16, 0, tzinfo=UTC)                  # Sunday preview -> the week that started Friday
    assert mc.week_end_for_post(sun, "preview") == date(2026, 9, 10)
    thu_late = datetime(2026, 10, 30, 3, 0, tzinfo=UTC)            # Thu 23:00 ET final -> previous Thursday
    assert mc.week_end_for_post(thu_late, "final") == date(2026, 10, 22)


def test_event_date_and_ladder_terms():
    assert mc.event_date("KXALBUMEQUIV-COO26OCT22") == date(2026, 10, 22)
    assert mc.event_date("KXTOPALBUM-26OCT31") == date(2026, 10, 31)
    rules = ("If Cookie Happened by Eels has above 10K Album Equivalent Units during the October 16, 2026 - "
             "October 22, 2026, tracking week, then the market resolves to Yes.")
    assert mc.ladder_terms(rules) == ("Cookie Happened", "Eels", "equiv")
    pure = ("If The Great Divide by Noah Kahan has above 80K Pure Album Sales (aka Product Sales) during the April 24, "
            "2026 - April 30, 2026, tracking week, then the market resolves to Yes.")
    assert mc.ladder_terms(pure) == ("The Great Divide", "Noah Kahan", "pure")
    streams = ("If Don’t Look Down by Rod Wave has above 100K streams during the August 28, 2026 - September 03, 2026, "
               "tracking week, then the market resolves to Yes.")
    assert mc.ladder_terms(streams)[2] == "streams"
    assert mc.chart_subject("If Damascus Road is #1 on the Billboard 200  chart for the Week of Oct 31, 2026, then",
                            "Damascus Road") == "Damascus Road"
    assert mc.luminate_value("2,790.254") == 2790.254 and mc.luminate_value("1,678 ") == 1678


def test_title_matching():
    assert mc.title_score("Don’t Look Down", "DON'T LOOK DOWN") == 1.0
    assert mc.title_score("The Black Parade", "BLACK PARADE (20TH ANNIVERSARY)") == 1.0
    assert mc.title_score("you seem pretty sad", "YOU SEEM PRETTY SAD FOR A GIRL SO IN LOVE") == 0.9
    assert mc.title_score("Dandelion", "THE GREAT DIVIDE") == 0.0


def test_fees_and_sizes():
    assert mc.order_fee(5, 0.99) == 0.01           # 0.0035 rounds up to a cent
    assert mc.order_fee(10, 0.50) == 0.18          # 0.175 -> 0.18
    assert mc.order_size(0.40) == 10 and mc.order_size(0.02) == 25


def test_taker_fills_side_and_pnl():
    rows = pl.DataFrame({"event_ticker": ["E", "E"], "ticker": ["A", "B"], "t": [datetime(2026, 5, 1, tzinfo=UTC)] * 2,
                         "bid": [0.30, 0.80], "ask": [0.35, 0.85], "y": [1, 0], "p": [0.60, 0.50]})
    f = mc.taker_fills(rows, "p", 0.05)
    a = f.filter(pl.col("ticker") == "A").row(0, named=True)
    assert a["side"] == "yes" and a["price"] == 0.35 and a["n"] == 11
    assert abs(a["pnl_c"] - (1 - 0.35 - a["fee"] / 11)) < 1e-9
    b = f.filter(pl.col("ticker") == "B").row(0, named=True)
    assert b["side"] == "no" and abs(b["price"] - 0.20) < 1e-9 and b["payoff_usd"] == b["n"]


def test_rank_probs_orders_by_units():
    p1, p2 = mc.rank_probs(np.array([100.0, 50.0, 49.0]), 0.05)
    assert p1[0] > 0.99 and abs(p2[1] - p2[2]) < 0.3 and abs(p1.sum() - 1) < 1e-9


def test_p_above_monotone():
    a = mc.p_above(100_000, 100_000, 0.0, 0.1)
    assert abs(a - 0.5) < 1e-9
    assert mc.p_above(80_000, 100_000, 0.0, 0.1) > 0.95
    assert mc.p_above(130_000, 100_000, 0.0, 0.1) < 0.05


def test_quotes_at_uses_last_candle_within_age():
    c = pl.DataFrame({"ticker": ["A", "A"], "end_ts": [datetime(2026, 5, 1, 10, tzinfo=UTC), datetime(2026, 5, 1, 11, tzinfo=UTC)],
                      "bid_close": [0.1, 0.2], "ask_close": [0.15, 0.25], "p_close": [0.12, 0.22]})
    req = pl.DataFrame({"ticker": ["A", "A", "A"], "t": [datetime(2026, 5, 1, 9, tzinfo=UTC), datetime(2026, 5, 1, 11, 30, tzinfo=UTC),
                                                         datetime(2026, 5, 2, 13, tzinfo=UTC)]})
    q = mc.quotes_at(c, req).sort("t")
    assert q.height == 1 and q["bid"][0] == 0.2          # nothing before the first candle, nothing older than 12 h


def test_feed_rows():
    now = datetime(2026, 11, 20, 22, 40, tzinfo=UTC)
    a = mc.apple_rows(json.loads((FIX / "apple_most_played_songs_trimmed.json").read_text()), "apple_songs", now)
    assert len(a) == 3 and a[0]["rank"] == 1 and a[0]["name"] and a[0]["artist"]
    it = mc.itunes_rows(json.loads((FIX / "itunes_topsongs_trimmed.json").read_text()), "itunes_songs", now)
    assert len(it) == 3 and it[0]["name"] and it[0]["feed_updated"].startswith("2026-11-20")


def test_listing_slugs():
    raw = 'href="/news/charts/chart-preview-x-2026-11-10" href=\\"/news/charts/top-50-2026-11-06\\" href="/news/live/a-2026-11-10"'
    assert mc.hits_listing_slugs(raw) == ["chart-preview-x-2026-11-10", "top-50-2026-11-06"]


def test_ladder_decisions_respect_close():
    ev = {"week_end": date(2026, 10, 29), "close_time": datetime(2026, 11, 1, 15, 0, tzinfo=UTC)}
    hw = pl.DataFrame({"kind": ["preview", "final"],
                       "published_at": [datetime(2026, 10, 27, 20, 38, tzinfo=UTC), datetime(2026, 10, 30, 20, 45, tzinfo=UTC)]})
    d = dict(mc.ladder_decisions(ev, hw))
    assert d["preview_first"] == datetime(2026, 10, 27, 22, 0, tzinfo=UTC)
    assert d["preview_fast"] == datetime(2026, 10, 27, 21, 0, tzinfo=UTC)          # >= 5 min after 20:38
    assert d["final"] == datetime(2026, 10, 30, 22, 0, tzinfo=UTC)
    assert d["sun09"] == datetime(2026, 11, 1, 14, 0, tzinfo=UTC) and all(t < ev["close_time"] for t in d.values())
    assert d["sat12"] - d["final"] < timedelta(days=1)


def test_unescape_and_soundtrack_titles():
    assert mc._unesc(r"THIS & THAT") == "THIS & THAT"
    assert mc.hits_title("SOUNDTRACK", "KPOP DEMON HUNTERS") == "KPOP DEMON HUNTERS"
    assert mc.hits_title("DANDELION", "ELLA LANGLEY") == "DANDELION"
    assert mc.title_score("KPop Demon Hunters", mc.hits_title("SOUNDTRACK", "KPOP DEMON HUNTERS")) == 1.0
