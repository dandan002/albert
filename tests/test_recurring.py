"""Recurring-series persistence study: parsers on recorded Kalshi payloads, own-history features
without look-ahead, the models, and taker / maker fills."""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from albert.research import recurring as rc
from albert.research import xcat

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _markets_frame(*names: str) -> pl.DataFrame:
    rows = []
    for name in names:
        payload = json.loads((FIX / name).read_text())
        rows += rc.market_rows(payload["markets"], "historical" if "historical" in name else "live")
    return pl.DataFrame(rows, schema=rc.MARKET_SCHEMA)


def _prepared(*names: str) -> pl.DataFrame:
    m = _markets_frame(*names).filter(pl.col("result").is_in(["yes", "no"]))
    return m.with_columns(
        rc.effective_strike().alias("strike_type"),
        pl.col("event_ticker").map_elements(rc.strand_of, return_dtype=pl.String).alias("strand"),
        (pl.col("result") == "yes").cast(pl.Int8).alias("y"),
        pl.struct("ticker", "event_ticker", "name").map_elements(
            lambda r: rc.leg_key(r["ticker"], r["event_ticker"], r["name"]), return_dtype=pl.String).alias("key"),
        pl.lit("Entertainment").alias("category"), pl.lit(True).alias("mutually_exclusive"))


# ---- parsers on recorded payloads -----------------------------------------------------------------
def test_event_rows_parse_recorded_listing():
    payload = json.loads((FIX / "recurring_events_KXNETFLIXRANKSHOW.json").read_text())
    rows = rc.event_rows(payload["events"], "KXNETFLIXRANKSHOW")
    assert len(rows) == len(payload["events"]) >= 3
    assert all(r["series_ticker"] == "KXNETFLIXRANKSHOW" and r["category"] == "Entertainment" for r in rows)
    pl.DataFrame(rows, schema=rc.EVENT_SCHEMA)


def test_market_rows_named_legs_use_custom_strike():
    payload = json.loads((FIX / "recurring_historical_markets_KXNETFLIXRANKSHOW.json").read_text())
    rows = rc.market_rows(payload["markets"], "historical")
    raw = payload["markets"][0]
    assert rows[0]["name"] == next(iter(raw["custom_strike"].values()))
    assert rows[0]["strike_type"] == "custom" and rows[0]["result"] in ("yes", "no")
    assert rows[0]["settlement_ts"].tzinfo is not None
    # titles drift between weeks; the winner still matches the settled value as the same leg
    win = [r for r in rows if r["result"] == "yes"]
    assert any(r["name"] != r["expiration_value"] for r in win)
    assert win and all(rc.same_leg(rc.leg_key(r["ticker"], r["event_ticker"], r["name"]),
                                   rc.leg_key(r["ticker"], r["event_ticker"], r["expiration_value"])) for r in win)


def test_market_rows_ladder_strikes_and_value():
    payload = json.loads((FIX / "recurring_markets_KXTRUTHSOCIALD.json").read_text())
    rows = rc.market_rows(payload["markets"], "live")
    assert all(r["strike_type"] in rc.LADDER_TYPES for r in rows)
    for r in rows:
        v = rc.parse_value(r["expiration_value"])
        assert v is not None
        # every recorded leg resolves exactly as its strikes say at the settled value
        assert rc.leg_outcome(r["strike_type"], r["floor_strike"], r["cap_strike"], v) == (r["result"] == "yes")


def test_parse_value_and_strand():
    assert rc.parse_value("1,234") == 1234.0
    assert rc.parse_value("45.2%") == 45.2
    assert rc.parse_value("$2.10") == 2.10
    assert rc.parse_value("301.3K") == pytest.approx(301300)
    assert rc.parse_value("14.87 M") == pytest.approx(14.87e6)
    assert rc.parse_value("Taken") is None
    assert rc.strand_of("KXNETFLIXRANKSHOW-26NOV09") == "KXNETFLIXRANKSHOW"
    assert rc.strand_of("KXBBCHARTPOSITIONALBUM-26SEP26YOU") == "KXBBCHARTPOSITIONALBUM|YOU"
    assert rc.strand_of("KXAPPRANKFREE-25NOV14") == "KXAPPRANKFREE"


def test_leg_outcome_conventions():
    assert rc.leg_outcome("greater", 10, None, 10.5) == 1
    assert rc.leg_outcome("greater", 10, None, 10) == 0
    assert rc.leg_outcome("less", None, 10, 9.9) == 1
    assert rc.leg_outcome("between", 10, 19, 19) == 1
    assert rc.leg_outcome("between", 10, 19, 19.5) == 0
    assert rc.leg_outcome("custom", None, None, 3) is None


# ---- features: no look-ahead ------------------------------------------------------------------------
def test_named_panel_carry_streak_and_no_lookahead():
    m = _prepared("recurring_historical_markets_KXNETFLIXRANKSHOW.json")
    e = rc.event_table(m)
    k = rc.strand_kinds(e)
    assert k["kind"].to_list() == ["cat"]
    p = rc.leg_panel(m, e, k)
    assert p.height > 0
    # the previous event's outcome was known before every decision
    known = dict(e.select("event_ticker", "known").iter_rows())
    assert all(known[pe] < t for pe, t in p.select("prev_event", "t").iter_rows())
    # incumbent = previous winner; its streak counts consecutive earlier wins
    hist = e.sort("pos").to_dicts()
    for r in p.filter(pl.col("carry") == 1).iter_rows(named=True):
        prev = next(h for h in hist if h["event_ticker"] == r["prev_event"])
        assert rc.matches(r["key"], prev["winners"]) and r["streak"] >= 1
    tr = rc.transitions(m, e, k)
    assert set(tr["gap"].unique()) <= {1, 2, 3}
    assert tr.filter(pl.col("carry") == 1)["bucket"].is_in(["w1", "w2", "w3_4", "w5+"]).all()


def test_ladder_panel_uses_only_past_changes():
    m = _prepared("recurring_markets_KXTRUTHSOCIALD.json").with_columns(pl.lit("Politics").alias("category"),
                                                                       pl.lit(False).alias("mutually_exclusive"))
    e = rc.event_table(m)
    k = rc.strand_kinds(e)
    assert k["kind"].to_list() == ["ladder"]
    p = rc.leg_panel(m, e, k)
    assert p.height > 0
    vals = dict(e.select("event_ticker", "value").iter_rows())
    for r in p.iter_rows(named=True):
        assert r["v_last"] == vals[r["prev_event"]]
        # history length grows only with events known before the decision
        assert r["n_hist"] <= r["n_known"] - r["gap"]
    assert p.group_by("event_ticker", "dt").len()["len"].max() <= rc.NEAR_LEGS


def test_change_prob_counts_and_prior():
    sims = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    assert rc.change_prob("greater", 12, None, sims) == pytest.approx((3 + 0.5) / 6)
    assert rc.change_prob("between", 10, 20, sims) == pytest.approx((3 + 0.5) / 6)
    assert rc.change_prob("greater", 12, None, sims[:3]) is None


def test_named_probs_shrink_toward_pool_and_use_past_only():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    trans = pl.DataFrame({"strand": ["A"] * 4 + ["B"] * 4, "kind": ["cat"] * 8, "gap": [1] * 8,
                          "known": [t0 + timedelta(days=i) for i in range(4)] * 2,
                          "carry": [1] * 8, "streak": [1] * 8, "y": [1, 1, 1, 1, 0, 0, 0, 0]},
                         schema_overrides={"carry": pl.Int8, "y": pl.Int8}).with_columns(pl.lit("w1").alias("bucket"))
    rows = pl.DataFrame({"strand": ["A", "A"], "kind": ["cat", "cat"], "gap": [1, 1], "carry": [1, 1], "streak": [1, 1],
                         "t": [t0 + timedelta(days=2), t0 + timedelta(days=10)]}, schema_overrides={"carry": pl.Int8})
    out = rc.named_probs(rows, trans, ks=(1, 100)).sort("t")
    # at day 2 only days 0 and 1 are known (strictly before): A 2/2, pool 2/4
    first = out.row(0, named=True)
    assert first["s_n"] == 2 and first["s_k"] == 2 and first["g_n"] == 4
    pool = (2 + 0.5) / 5
    assert first["p_k1"] == pytest.approx((2 + pool) / 3)
    assert out["p_k100"][1] < out["p_k1"][1]        # stronger shrinkage pulls toward the pool


# ---- execution ------------------------------------------------------------------------------------
def _quote_rows() -> pl.DataFrame:
    t = datetime(2026, 10, 5, 12, tzinfo=UTC)
    return pl.DataFrame({
        "ticker": ["X-A", "X-B", "X-C"], "event_ticker": ["X-1", "X-1", "X-2"], "kind": ["cat"] * 3, "dt": ["mid"] * 3,
        "t": [t] * 3, "close": [t + timedelta(hours=10)] * 3, "settle": [t + timedelta(hours=12)] * 3,
        "y": [1, 0, 1], "p": [0.80, 0.30, 0.50], "bid": [0.60, 0.40, 0.45], "ask": [0.65, 0.45, 0.55],
        "mid": [0.625, 0.425, 0.50], "fee_multiplier": [1.0] * 3, "fee_type": ["quadratic"] * 3, "split": ["holdout"] * 3},
        schema_overrides={"y": pl.Int8})


def test_taker_fee_rounded_up_per_order_and_sides():
    f = rc.taker_trades(_quote_rows(), 0.05).sort("ticker")
    assert f["ticker"].to_list() == ["X-A", "X-B"]
    a, b = f.row(0, named=True), f.row(1, named=True)
    assert a["dir"] == 1 and a["price"] == 0.65 and a["n"] == int(rc.STAKE // 0.65)
    raw = 0.07 * a["n"] * 0.65 * 0.35
    assert a["fee_order"] == pytest.approx(math.ceil(raw * 100) / 100) and a["fee_order"] > raw   # rounded up
    assert a["pnl"] == pytest.approx(1 - 0.65 - a["fee_order"] / a["n"])
    assert b["dir"] == -1 and b["price"] == pytest.approx(0.60) and b["payoff"] == 1


def test_maker_fills_need_a_later_trade_strictly_through():
    payload = json.loads((FIX / "recurring_candles_incumbent.json").read_text())
    c = pl.DataFrame(xcat.ohlc_rows(payload, payload["ticker"], 60), schema=xcat.OHLC_SCHEMA)
    lows = c.filter(pl.col("p_low").is_not_null())
    assert lows.height >= 2
    t = lows["end_ts"][0] - timedelta(hours=2)
    lo = float(lows["p_low"].min())
    q = pl.DataFrame({"ticker": [payload["ticker"]] * 2, "event_ticker": ["E"] * 2, "kind": ["cat"] * 2, "dt": ["mid"] * 2,
                      "t": [t] * 2, "close": [c["end_ts"].max()] * 2, "settle": [c["end_ts"].max()] * 2, "y": [1, 1],
                      "p": [lo + 0.045, lo + 0.035], "bid": [0.0, 0.0], "ask": [1.0, 1.0], "mid": [None, None],
                      "fee_multiplier": [1.0, 1.0], "fee_type": ["quadratic"] * 2, "split": ["dev"] * 2},
                     schema_overrides={"y": pl.Int8, "mid": pl.Float64})
    o = rc.maker_orders(q, 0.03).filter(pl.col("dir") == 1)
    f = rc.maker_fills(o, c)
    # a YES bid at the low + 1c fills (the low traded strictly below it); a bid at the low itself does not
    assert f.height == 1 and f["price"][0] == pytest.approx(round(lo + 0.01, 2))
    assert f["t_fill"][0] >= t + timedelta(hours=1) and f["fee_order"][0] == 0.0


def test_pack_live_respects_api_caps():
    t0 = datetime(2026, 10, 1, tzinfo=UTC)
    rows = [{"ticker": f"T{i}", "lo": t0 + timedelta(hours=i), "hi": t0 + timedelta(hours=i + 48)} for i in range(200)]
    jobs = rc.pack_live(rows)
    assert sum(len(j) for j in jobs) == 200
    for j in jobs:
        lo, hi = min(r["lo"] for r in j), max(r["hi"] for r in j)
        assert len(j) <= rc.MAX_BATCH_TICKERS
        assert (int((hi - lo).total_seconds() // 3600) + 1) * len(j) <= rc.MAX_BATCH_CANDLES


def test_decision_times_order():
    o = datetime(2026, 10, 1, tzinfo=UTC)
    d = rc.decision_times(o, o + timedelta(days=7))
    assert d["open1h"] < d["mid"] < d["final24h"]
    assert "final24h" not in rc.decision_times(o, o + timedelta(hours=20))
