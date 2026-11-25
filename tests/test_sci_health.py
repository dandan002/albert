"""Tests for albert.research.sci_health (parsers on trimmed recorded responses; model logic on small frames)."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from albert.research import sci_health as sh
from albert.research.counts import market_row

FIX = Path(__file__).parent / "fixtures" / "sci_health"


def _resp(name: str):
    return json.loads((FIX / name).read_text())["response"]


def test_family_of_topic_series():
    assert sh.family_of("KXBIGGESTQUAKE") == "quake_day"
    assert sh.family_of("KXHURCAT") == "hurricane_storm"
    assert sh.family_of("KXMEASLES") == "disease_count"
    assert sh.family_of("KXROCKET") == "launch_count"
    assert sh.family_of("KXHIGHNY") is None


def test_usgs_rows_parse_recorded_query():
    rows = sh.usgs_rows(_resp("usgs_query_2026-09-02.json"))
    assert len(rows) == 4
    assert rows[0]["id"] == "us7000tdhl" and rows[0]["mag"] == 5.3 and rows[0]["mag_type"] == "mww"
    assert rows[0]["time"].tzinfo is not None and rows[0]["time"].date() == date(2026, 9, 2)
    assert max(r["mag"] for r in rows) == 6.2


def test_usgs_first_publication_from_superseded_detail():
    pub = sh.usgs_first_publication(_resp("usgs_detail_us7000tdms.json"))
    assert pub["pub_any"] < pub["pub_origin"]
    assert pub["pub_origin"] == datetime(2026, 9, 2, 12, 55, 12, 40000, tzinfo=UTC)
    assert pub["mag_first"] == 5.9


def test_kalshi_quake_markets_and_event_day():
    p = _resp("kalshi_markets_quake.json")
    rows = [market_row(m, "KXBIGGESTQUAKE", "live") for m in p["markets"]]
    assert {r["floor_strike"] for r in rows} == {7.0, 6.8, 6.6}
    assert all(r["result"] == "no" for r in rows)
    assert sh.quake_day(rows[0]["event_ticker"]) == date(2026, 11, 13)
    assert sh.quake_day("KXBIGGESTQUAKE-25") is None


def test_hurdat_and_bdeck_parsers():
    hd = sh.hurdat_rows((FIX / "hurdat2_excerpt.txt").read_text())
    assert hd and hd[0]["storm_id"].startswith(("AL", "EP")) and hd[0]["vmax"] is not None
    bd = sh.bdeck_rows((FIX / "bdeck_excerpt.dat").read_text())
    assert bd and all(r["basin"] in ("AL", "EP", "CP") for r in bd)
    assert any(r["name"] not in ("Invest", "") for r in bd)


def test_quake_climate_counts_conditional_days():
    d0 = datetime(2020, 1, 1, tzinfo=UTC)
    # one quake per day: 6.0 at 06:00 on even days, 5.0 at 18:00 on odd days, for 800 days
    rows = []
    for i in range(800):
        t = d0 + timedelta(days=i, hours=6 if i % 2 == 0 else 18)
        rows.append({"id": str(i), "time": t, "mag": 6.0 if i % 2 == 0 else 5.0})
    q = pl.DataFrame(rows).with_columns(pl.col("time").cast(pl.Datetime("ms", "UTC")))
    c = sh.QuakeClimate(q, first=date(2020, 1, 1), last=date(2022, 3, 10))
    d = date(2022, 3, 1)
    p0, n0 = c.p(d, datetime(2022, 2, 28, 21, tzinfo=UTC), 6.0)
    assert n0 >= 700 and abs(p0 - 0.5) < 0.01            # half of all days reach 6.0
    p12, _ = c.p(d, datetime(2022, 3, 1, 12, tzinfo=UTC), 6.0)
    assert p12 < 0.01                                    # a 6.0 only ever comes at 06:00: none left by noon
    assert c.known_max(date(2021, 1, 2), datetime(2021, 1, 2, 23, tzinfo=UTC)) in (5.0, 6.0)


def test_tape_fill_strictly_through():
    tr = pl.DataFrame({"ticker": ["A"] * 4,
                       "created_time": [datetime(2026, 9, 1, h, tzinfo=UTC) for h in (1, 2, 3, 4)],
                       "yes_price": [0.20, 0.21, 0.25, 0.10], "count": [5.0, 3.0, 4.0, 9.0],
                       "taker_side": ["yes", "yes", "yes", "no"]})
    tape = sh.Tape(tr)
    t0, t1 = datetime(2026, 9, 1, 0, tzinfo=UTC), datetime(2026, 9, 1, 5, tzinfo=UTC)
    got, ts = tape.fill("A", t0, t1, "no", 0.20, 6)      # sell YES at 20c: 0.20 is a touch, not through
    assert got == 6 and ts == datetime(2026, 9, 1, 3, tzinfo=UTC)
    got, _ = tape.fill("A", t0, t1, "yes", 0.10, 3)      # YES bid at 10c: the NO print at 0.10 only touches
    assert got == 0


def test_holm_and_fee():
    import numpy as np

    adj = sh.holm(np.array([0.01, 0.04, 0.03]))
    assert abs(adj[0] - 0.03) < 1e-12 and adj[1] >= adj[2]
    assert sh.taker_fee(10, 0.5) == 0.18                 # 0.07 x 10 x 0.25 = 0.175 -> 0.18


def test_hurricane_names():
    assert sh.hur_event_name("KXHURCAT-26FAY") == (2026, "Fay")
    assert sh.hur_event_name("KXHURCAT-JERRY") == (2025, "Jerry")
    assert sh.name_basin(2026, "Vance") == ("EP", 20)
    assert sh.name_basin(2026, "Lala") == ("CP", 1)


def test_origin_versions_and_displayed_magnitude():
    vs = sh.origin_versions(_resp("usgs_detail_us7000tdms.json"))
    assert len(vs) == 3 and all(v["source"] == "us" and v["mag"] == 5.9 for v in vs)
    frame = pl.DataFrame([{**v, "event_id": "us7000tdms", "origin": datetime(2026, 9, 2, 12, 35, 56, tzinfo=UTC)}
                          for v in vs])
    rt = sh.Realtime(frame)
    d = date(2026, 9, 2)
    assert rt.known_max(d, datetime(2026, 9, 2, 12, 50, tzinfo=UTC)) == float("-inf")   # not yet published
    assert rt.known_max(d, datetime(2026, 9, 2, 13, 0, tzinfo=UTC)) == 5.9
    assert rt.first_cross(d, 5.8) == datetime(2026, 9, 2, 12, 55, 12, 40000, tzinfo=UTC)
