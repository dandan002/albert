from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from albert.research import recurring_alt as ra

FIX = Path(__file__).parent / "fixtures"


def test_parse_portwatch_trimmed_real_csv():
    d = ra.parse_portwatch((FIX / "portwatch" / "daily_chokepoints_trim.csv").read_text(encoding="utf-8-sig"))
    hz = d.filter(pl.col("portid") == "chokepoint6")
    assert hz.height == 7
    assert hz["date"].min() == date(2026, 11, 9) and hz["date"].max() == date(2026, 11, 15)
    # Kalshi settled KXHORMUZWEEKLY-26NOV15 (Nov 9-15) at 16 and KXHORMUZPEAK-26NOV15 at 6
    assert hz["n_total"].sum() == 16
    assert hz["n_total"].max() == 6
    assert d.filter(pl.col("portid") == "chokepoint1")["portname"][0] == "Suez Canal"


def test_published_at_is_tuesday_after_the_week():
    # Mon Nov 9 .. Sun Nov 15 all become public on Tue Nov 17 (rebuild seen 10:15 UTC; 12:00 used)
    for d in (date(2026, 11, 9), date(2026, 11, 15)):
        assert ra.published_at(d) == datetime(2026, 11, 17, 12, tzinfo=UTC)
    assert ra.published_at(date(2026, 11, 16)) == datetime(2026, 11, 24, 12, tzinfo=UTC)


def test_parse_actblue_trimmed_real_page():
    cards = ra.parse_actblue((FIX / "actblue" / "directory_trim.html").read_text())
    assert [c["rank"] for c in cards] == [1, 2, 3, 4, 5]
    assert cards[0]["name"] == 'Patricia "Pat" Moore'
    assert cards[0]["office"] == "LA-05 2026"
    assert ra.parse_actblue("<html></html>") == []


def test_period_and_leg_day():
    rules = ("If the number of transit calls through the Strait of Hormuz as reported by the IMF PortWatch is "
             "above 200 from Apr 20, 2026 to Apr 26, 2026, then the market resolves to Yes.")
    assert ra.period_of(rules, "KXHORMUZWEEKLY-26APR26") == (date(2026, 4, 20), date(2026, 4, 26))
    assert ra.period_of(None, "KXHORMUZMAX-26NOV22") == (date(2026, 11, 16), date(2026, 11, 22))
    assert ra._leg_day("KXHORMUZMAX-26NOV22-NOV16", date(2026, 11, 22)) == date(2026, 11, 16)
    assert ra._leg_day("KXHORMUZMAX-26JAN03-DEC28", date(2027, 1, 3)) == date(2026, 12, 28)


def test_ratio_probs_uses_only_given_history():
    hist = [(date(2026, 8, 2) + (date(2026, 8, 9) - date(2026, 8, 2)) * i, v)
            for i, v in enumerate([30, 33, 30, 33, 30, 33, 30, 33])]
    p = ra.ratio_probs(hist, date(2026, 9, 27), [20.0, 31.0, 60.0])
    assert p[0] > 0.9 and p[2] < 0.1 and 0.2 < p[1] < 0.8
    assert ra.ratio_probs(hist[:3], date(2026, 9, 27), [20.0]) == [None]


def test_maxday_probs_ties_and_revision_cut():
    # three clean weeks after REVISION_SAFE: Monday always the max, one week tied Mon/Tue
    daily = {}
    start = date(2026, 8, 17)
    for w in range(3):
        for i in range(7):
            daily[date.fromordinal(start.toordinal() + 7 * w + i)] = 1.0
        daily[date.fromordinal(start.toordinal() + 7 * w)] = 5.0
    daily[date(2026, 8, 25)] = 5.0       # Tue of week 2 ties Monday
    through = date(2026, 9, 6)
    days = [date(2026, 9, 7 + i) for i in range(7)]
    p = ra.maxday_probs(daily, through, days, k=0.0)
    assert abs(p[0] - (2.5 / 3)) < 1e-9 and abs(p[1] - (0.5 / 3)) < 1e-9
    assert abs(sum(p) - 1.0) < 1e-9
    assert ra.maxday_probs(daily, date(2026, 8, 23), days) == [None] * 7   # < 3 weeks after the cut


def test_take_fee_rounding_and_pnl():
    rows = pl.DataFrame({"event_ticker": ["E"], "ticker": ["E-T1"], "bid": [0.05], "ask": [0.07], "y": [0.0],
                         "t": [datetime(2026, 11, 1, tzinfo=UTC)], "settlement_ts": [datetime(2026, 11, 3, tzinfo=UTC)]})
    f = ra.take(rows, pl.lit("no"))
    assert f["price"][0] == 0.95 and f["n"][0] == 4
    # 0.07 * 4 * 0.95 * 0.05 = 0.0133 -> rounded up to $0.02 per order
    assert abs(f["fee_order"][0] - 0.02) < 1e-12
    assert abs(f["pnl_usd"][0] - (4 * 0.05 - 0.02)) < 1e-9
    wide = rows.with_columns(pl.lit(0.30).alias("ask"))
    assert ra.take(wide, pl.lit("no")).height == 0          # spread > 10c is never crossed
