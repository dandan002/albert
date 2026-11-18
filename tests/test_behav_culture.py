"""Behavioural culture / weather study: parsers on recorded payloads (Kalshi listing and candles,
Wikimedia pageviews), features without look-ahead, taker fills with per-order fee rounding, and the
capped-book ledger."""

import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from albert.research import behav_culture as bc
from albert.research import xcat

FIX = Path(__file__).parent / "fixtures"


# ---- recorded payloads -----------------------------------------------------------------------------
def _listing() -> pl.DataFrame:
    payload = json.loads((FIX / "kalshi" / "behav_culture_markets_nov.json").read_text())
    series = pl.read_parquet(bc.TABLES / "kalshi_series.parquet") if (bc.TABLES / "kalshi_series.parquet").exists() else None
    info = {"KXHIGHNY": {"category": "Climate and Weather", "fee_type": "quadratic", "fee_multiplier": 1.0}}
    if series is not None:
        info |= {r["series_ticker"]: r for r in series.iter_rows(named=True)}
    return pl.DataFrame(bc.listing_rows(payload, info), infer_schema_length=None)


def test_listing_rows_keep_strikes_and_results():
    m = _listing()
    assert m.height >= 10
    assert set(m["result"].unique().to_list()) <= {"yes", "no"}
    assert m["settlement_ts"].dtype == pl.Datetime("us", "UTC")
    # threshold ladders carry floor strikes; the parsed yes_sub_title agrees with them
    lad = m.filter((pl.col("strike_type") == "greater") & ~pl.col("series_ticker").str.starts_with("KXHIGH"))
    assert lad.height >= 3
    for sub, fl in lad.select("yes_sub_title", "floor_strike").iter_rows():
        d, v = bc.parse_strike(sub)
        assert d == "above" and v == pytest.approx(fl, rel=1e-6)


def test_weather_legs_from_recorded_listing():
    m = _listing().filter(pl.col("series_ticker").str.starts_with("KXHIGH"))
    legs = bc.weather_legs(m)
    assert legs.height % 6 == 0 and legs.height >= 6
    one = legs.filter(pl.col("event_ticker") == legs["event_ticker"][0]).sort("lo")
    assert one["pos"].to_list() == ["low_tail"] + ["interior"] * 4 + ["high_tail"]
    assert one["y"].sum() == 1.0                     # exactly one bracket settles YES
    # brackets tile the integers: each interior bracket starts one above the previous end
    his = one["hi"].to_list()
    los = one["lo"].to_list()
    assert all(los[i + 1] == his[i] + 1 for i in range(1, 4))


def test_batch_candles_parse_and_asof_quotes_never_look_ahead():
    payload = json.loads((FIX / "kalshi" / "behav_culture_candles_batch.json").read_text())
    rows = xcat.batch_rows(payload, 60)
    c = pl.DataFrame([r for v in rows.values() for r in v], schema=xcat.OHLC_SCHEMA)
    assert c.height > 0 and c["ticker"].n_unique() >= 2
    cand = c.select("ticker", "end_ts", pl.col("bid_close").alias("yes_bid_close"), pl.col("ask_close").alias("yes_ask_close"),
                    pl.col("p_close").alias("price_close"), "volume", pl.col("oi").alias("open_interest"))
    tk = c["ticker"][0]
    one = c.filter(pl.col("ticker") == tk).sort("end_ts")
    t = one["end_ts"][len(one) // 2] + timedelta(minutes=30)          # between two candles
    q = bc.asof_quotes(pl.DataFrame({"ticker": [tk], "t": [t]}), cand)
    assert q["end_ts"][0] <= t
    assert q["end_ts"][0] == one.filter(pl.col("end_ts") <= t)["end_ts"].max()
    # vol24 = volume of candles ending in (t - 24 h, t]
    exp = one.filter((pl.col("end_ts") > t - timedelta(hours=24)) & (pl.col("end_ts") <= t))["volume"].fill_null(0).sum()
    assert q["vol24"][0] == pytest.approx(exp)


def test_pageviews_parse_and_spike_uses_only_finished_days():
    payload = json.loads((FIX / "wikimedia" / "behav_culture_pageviews_taylor_swift.json").read_text())
    pv = pl.DataFrame(bc.pageview_rows(payload, "Taylor_Swift"))
    assert pv.height == len(payload["items"]) and pv["views"].min() > 0
    t = datetime(2026, 9, 20, 15, tzinfo=UTC)
    s = bc.pageview_spike(pl.DataFrame({"title": ["Taylor_Swift"], "t": [t]}), pv)["pv_spike"][0]
    last = date(2026, 9, 18)                                          # t - 24 h is Sep 19; last full day Sep 18
    rec = pv.filter(pl.col("day").is_between(last - timedelta(days=2), last))["views"].mean()
    base = pv.filter(pl.col("day").is_between(last - timedelta(days=30), last - timedelta(days=3)))["views"].mean()
    assert s == pytest.approx(rec / base)
    # changing a later day (Sep 19, still in progress 24 h before t) cannot change the signal
    pv2 = pv.with_columns(pl.when(pl.col("day") >= date(2026, 9, 19)).then(pl.col("views") * 50).otherwise(pl.col("views")))
    assert bc.pageview_spike(pl.DataFrame({"title": ["Taylor_Swift"], "t": [t]}), pv2)["pv_spike"][0] == pytest.approx(s)


# ---- features --------------------------------------------------------------------------------------
@pytest.mark.parametrize("sub,exp", [
    ("Above 9M", ("above", 9e6)), ("Above 16.5B", ("above", 16.5e9)), ("At least 5 weeks", ("above", 5.0)),
    ("Exactly 9 songs", ("exact", 9.0)), ("Above $121.39", ("above", 121.39)), ("Below 40%", ("below", 40.0)),
    ("Above 3,000", ("above", 3000.0)), ("Taylor Swift", (None, None)), ("Before Oct 24, 2026", (None, None)),
])
def test_parse_strike(sub, exp):
    assert bc.parse_strike(sub) == exp


def test_fandom_list_matches_whole_names_only():
    s = pl.Series(["Will Taylor Swift be #1?", "Will BTS chart?", "Will Jinx win?", "Will Drake release?",
                   "Will the Future of AI arrive?", "Will Future have above 5M daily views?", "Mutta Kalakki",
                   "Will Lisanne win?", "ROSÉ - APT."])
    got = pl.DataFrame({"s": s}).select(bc.fandom_expr(pl.col("s")))["s"].to_list()
    # "Future" is case-sensitive: the capitalised artist matches, but so does a capitalised common noun,
    # which is why the list is checked series by series in the report
    assert got == [True, True, False, True, True, True, False, False, True]


def test_implied_stats_median_and_overround():
    ev = pl.DataFrame({"lo": [-math.inf, 70.0, 72.0, 74.0, 76.0, 78.0], "hi": [69.0, 71.0, 73.0, 75.0, 77.0, math.inf],
                       "mid": [0.02, 0.10, 0.40, 0.40, 0.08, 0.02]})
    s = bc.implied_stats(ev)
    assert s["overround"] == pytest.approx(1.02)
    assert 73 <= s["imp_med"] <= 74


def test_scheduled_close_ignores_actual_early_close():
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    m = pl.DataFrame({
        "series_ticker": ["S"] * 3, "can_close_early": [True] * 3,
        "close_time": [t0 + timedelta(days=7), t0 + timedelta(days=14), t0 + timedelta(days=16)],   # last closed early
        "expected_expiration_time": [t0 + timedelta(days=7, hours=2), t0 + timedelta(days=14, hours=2),
                                     t0 + timedelta(days=21, hours=2)]})
    out = bc.scheduled_close(m)
    assert out["sched_close"][2] == t0 + timedelta(days=21)


# ---- execution and ledger --------------------------------------------------------------------------
def _quotes() -> pl.DataFrame:
    t = datetime(2026, 9, 1, 12, tzinfo=UTC)
    return pl.DataFrame({
        "ticker": ["A", "B"], "event_ticker": ["E1", "E2"], "t": [t, t + timedelta(hours=1)],
        "bid": [0.30, 0.04], "ask": [0.32, 0.06], "mid": [0.31, 0.05], "y": [0.0, 1.0],
        "fee_multiplier": [1.0, 1.0], "settlement_ts": [t + timedelta(days=2), t + timedelta(days=1)],
        "et_date": [date(2026, 9, 1)] * 2})


def test_take_no_side_fee_rounded_up_per_order():
    tr = bc.take(_quotes(), "no")
    a = tr.filter(pl.col("ticker") == "A").row(0, named=True)
    assert a["price"] == pytest.approx(0.70) and a["n"] == 7                 # floor($5 / 0.70)
    raw = 0.07 * 7 * 0.70 * 0.30
    assert a["fee_order"] == pytest.approx(math.ceil(raw * 100) / 100)      # 0.1029 -> 0.11
    assert a["pnl_usd"] == pytest.approx(7 * 0.30 - a["fee_order"])
    b = tr.filter(pl.col("ticker") == "B").row(0, named=True)
    assert b["payoff"] == 0.0 and b["pnl"] < -0.9


def test_capped_ledger_respects_event_cap_and_frees_capital():
    q = _quotes()
    q = pl.concat([q, q.with_columns(pl.col("ticker") + "2", pl.col("t") + timedelta(minutes=5))])  # second leg, same events
    tr = bc.take(q, "no")
    led = bc.capped_ledger(tr, "s", "v", "dev", bankroll=100.0, event_cap=5.0)
    assert led["event_ticker"].value_counts()["count"].max() == 1             # $4.9 + $4.9 > $5 per event
    led2 = bc.capped_ledger(tr, "s", "v", "dev", bankroll=5.0, event_cap=100.0)
    assert led2.height == 1                                                   # bankroll binds until settlement
    assert set(led.columns) >= {"struct_id", "event_ticker", "ticker", "side", "n", "t_in", "cost_usd", "settle_ts", "payoff_usd"}


def test_cluster_mean_t():
    f = pl.DataFrame({"x": [1.0, 1.0, -1.0, 3.0], "g": ["a", "a", "b", "c"]})
    m, t, k = bc.cl_mean(f, "x", "g")
    assert k == 3 and m == pytest.approx(1.0) and t is not None


def test_pack_jobs_respects_batch_caps_and_keeps_historical_single():
    t0 = datetime(2026, 10, 1, tzinfo=UTC)
    cut = datetime(2026, 9, 18, tzinfo=UTC)
    rows = []
    for i in range(120):
        lo = t0 + timedelta(hours=7 * i)
        rows.append({"ticker": f"T{i}", "event_ticker": f"E{i // 6}", "lo": lo, "hi": lo + timedelta(hours=48 + (i % 5) * 40),
                     "period": 60, "settlement_ts": lo + timedelta(days=3)})
    rows.append({"ticker": "H", "event_ticker": "EH", "lo": t0 - timedelta(days=40), "hi": t0 - timedelta(days=20),
                 "period": 60, "settlement_ts": t0 - timedelta(days=20)})
    jobs = bc.pack_jobs(pl.DataFrame(rows), cut, max_candles=1000, max_tickers=10)
    hist = [j for j in jobs if j["kind"] == "hist"]
    assert len(hist) == 1 and hist[0]["rows"][0]["ticker"] == "H"
    seen = []
    for j in (j for j in jobs if j["kind"] == "batch"):
        lo = min(r["lo"] for r in j["rows"])
        hi = max(r["hi"] for r in j["rows"])
        span = math.ceil((hi - lo).total_seconds() / 3600) + 1
        assert len(j["rows"]) <= 10 and (len(j["rows"]) == 1 or span * len(j["rows"]) <= 1000)
        seen += [r["ticker"] for r in j["rows"]]
    assert sorted(seen) == sorted(f"T{i}" for i in range(120))


def test_r1_rule_selects_priced_fandom_entertainment_legs_only():
    t = datetime(2026, 9, 1, 12, tzinfo=UTC)
    base = {"t": t, "settlement_ts": t + timedelta(days=1), "fee_multiplier": 1.0, "y": 0.0, "et_date": date(2026, 9, 1),
            "event_ticker": "E", "series_ticker": "KXYTVIEWSW"}
    rows = [
        {**base, "ticker": "A", "category": "Entertainment", "fandom": True, "bid": 0.09, "ask": 0.11, "mid": 0.10, "spread": 0.02},
        {**base, "ticker": "B", "category": "Entertainment", "fandom": True, "bid": 0.40, "ask": 0.44, "mid": 0.42, "spread": 0.04},
        {**base, "ticker": "C", "category": "Entertainment", "fandom": True, "bid": 0.05, "ask": 0.25, "mid": 0.15, "spread": 0.20},
        {**base, "ticker": "D", "category": "Entertainment", "fandom": False, "bid": 0.09, "ask": 0.11, "mid": 0.10, "spread": 0.02},
        {**base, "ticker": "E", "category": "Science and Technology", "fandom": True, "bid": 0.09, "ask": 0.11, "mid": 0.10,
         "spread": 0.02},
    ]
    p = pl.DataFrame(rows)
    assert bc.r1_fandom_no(p)["ticker"].to_list() == ["A"]
    assert bc.r1_fandom_no(p, fandom=False)["ticker"].to_list() == ["D"]
    a = bc.r1_fandom_no(p).row(0, named=True)
    assert a["side"] == "no" and a["price"] == pytest.approx(0.91) and a["payoff"] == 1.0
