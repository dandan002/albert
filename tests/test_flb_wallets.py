"""Wallets workstream (paper_flb): parsers on recorded Polymarket payloads, sampling, accounting, inference."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from albert.research import flb_wallets as W

FX = Path(__file__).parent / "fixtures" / "polymarket"


def _fx(name: str):
    return json.loads((FX / name).read_text())


def test_frame_rows_from_recorded_events():
    rows = W.frame_rows(_fx("flb_events_keyset.json")["events"])
    f = pl.DataFrame(rows, schema=W.FRAME_SCHEMA)
    assert f.height == 4
    by_slug = {r["event_slug"]: r for r in rows}
    nhl = by_slug["nhl-tor-nyr-2025-02-28"]
    assert nhl["group"] == "Sports" and nhl["winner_index"] == 0
    assert nhl["game_start_time"] == datetime(2025, 3, 1, tzinfo=UTC)
    assert nhl["token0"] != nhl["token1"] and nhl["volume"] > 1000
    assert not nhl["fees_enabled"] and nhl["fee_rate"] == 0.0
    assert by_slug["highest-temperature-in-nyc-on-march-1"]["group"] == "Weather"
    assert by_slug["will-trump-issue-an-executive-order-on-march-1"]["group"] == "Politics"
    assert f["condition_id"].n_unique() == 4


def test_trade_rows_and_taker_labels_from_recorded_feeds():
    all_rows = W.trade_rows(_fx("flb_trades_all.json"))
    taker_rows = W.trade_rows(_fx("flb_trades_taker.json"))
    assert len(all_rows) > len(taker_rows) >= 3
    r = all_rows[0]
    assert r["wallet"] == r["wallet"].lower() and r["outcome_index"] in (0, 1) and isinstance(r["ts"], int)
    f = pl.DataFrame(all_rows, schema=W.TRADE_SCHEMA)
    t = pl.DataFrame(taker_rows, schema=W.TRADE_SCHEMA).select("tx", "wallet").unique()
    f = f.join(t.with_columns(pl.lit(True).alias("taker")), on=["tx", "wallet"], how="left").with_columns(
        pl.col("taker").fill_null(False))
    # every transaction has exactly one taker wallet, and maker rows offset it in YES-exposure terms
    per_tx = f.group_by("tx").agg(pl.col("taker").sum().alias("n_taker"))
    assert per_tx["n_taker"].to_list() == [1] * per_tx.height
    norm = W.normalize(f)
    sums = norm.group_by("tx").agg(pl.col("expo").sum(), pl.col("cash").sum())
    assert np.allclose(sums["expo"].to_numpy(), 0, atol=1e-6)
    assert np.allclose(sums["cash"].to_numpy(), 0, atol=1e-6)


def test_user_stats_row_from_recorded_payload():
    p = _fx("flb_user_stats.json")
    row = W.user_stats_row(p, p["data"]["proxy_wallet"].upper())
    assert row["known"] and row["join_date"] == 1736524761
    assert row["wallet"] == p["data"]["proxy_wallet"].lower() and row["markets_traded_now"] == 87
    assert W.user_stats_row({"data": None}, "0xabc") == {
        "wallet": "0xabc", "known": False, "join_date": None, "markets_traded_now": None,
        "biggest_win_now": None, "pnl_now": None, "trade_count_now": None}


def test_price_points_from_recorded_history():
    pts = W.price_points(_fx("price_history.json"))
    assert len(pts) == 5 and pts[0] == (1793595600, 0.495)


def test_stratified_sample_weights():
    rng = np.random.default_rng(0)
    n = 300
    frame = pl.DataFrame({
        "condition_id": [f"c{i}" for i in range(n)],
        "group": rng.choice(["Sports", "Politics"], n).tolist(),
        "tier": rng.choice(list(W.TIER_LABELS), n).tolist(),
        "month": rng.choice(["2025-01", "2025-02"], n).tolist(),
        "volume": rng.uniform(1e3, 1e7, n).tolist(),
    })
    s = W.sample_markets(frame)
    again = W.sample_markets(frame)
    assert s["condition_id"].to_list() == again["condition_id"].to_list()     # seeded
    sizes = frame.with_columns(W.period_expr()).group_by(W.STRATUM).len()
    chk = s.group_by(W.STRATUM).agg(pl.len().alias("n"), pl.col("weight").first()).join(sizes, on=W.STRATUM)
    assert set(s.filter(pl.col("tier") == "$1M-6M")["period"].to_list()) <= {"2025H1"}
    assert s.filter(pl.col("tier") == ">$6M").height == 0
    for r in chk.iter_rows(named=True):
        assert r["n"] == min(W.PER_STRATUM[r["tier"]], r["len"])
        assert r["weight"] == pytest.approx(r["len"] / r["n"])


def test_asof_features_use_only_prior_fills():
    f = pl.DataFrame({
        "wid": [1, 1, 1, 1, 2], "mid": [10, 10, 11, 12, 10], "buy": [True, False, True, True, True],
        "outcome_index": [0, 0, 1, 0, 0], "price": [0.5, 0.6, 0.05, 0.9, 0.5],
        "size": [10.0, 10.0, 100.0, 2000.0, 1.0], "ts": [100, 200, 300, 400, 150],
        "taker": [True, False, False, True, True]})
    a = W.asof_features(f).sort("wid", "ts")
    w1 = a.filter(pl.col("wid") == 1)
    assert w1["n_prev"].to_list() == [0, 1, 2, 3]
    assert w1["maker_prev"].to_list() == [0, 0, 1, 2]
    assert w1["mkts_prev"].to_list() == [0, 1, 1, 2]
    # market 10 becomes two-sided at ts 200 (buy then sell), so it counts from the fill after it
    assert w1["two_prev"].to_list() == [0, 0, 1, 1]
    # changing a later fill must not change earlier features
    b = W.asof_features(f.with_columns(pl.when(pl.col("ts") == 400).then(99999.0).otherwise(pl.col("size"))
                                       .alias("size"))).sort("wid", "ts")
    cols = ["n_prev", "maker_prev", "mkts_prev", "two_prev", "typ_size"]
    assert a.filter(pl.col("ts") < 400).select(cols).equals(b.filter(pl.col("ts") < 400).select(cols))


def test_classes_are_exclusive_and_ordered():
    f = pl.DataFrame({"n_prev": [30, 0, 3, 50, 5], "maker_prev": [29, 0, 0, 0, 0], "mkts_prev": [10, 0, 1, 5, 1],
                      "two_prev": [9, 0, 0, 0, 0], "typ_size": [50.0, 5000.0, 20.0, 20.0, 300.0]})
    got = f.with_columns(W.class_expr(W.DEFAULT_TH))["cls"].to_list()
    assert got == ["MM", "Whale", "Retail", "Active retail", "Mid"]


def test_logit_slope_recovers_known_slope():
    rng = np.random.default_rng(1)
    n = 20000
    q = rng.uniform(0.02, 0.98, n)
    lg = np.log(q / (1 - q))
    true = 1 / (1 + np.exp(-(0.1 + 1.3 * lg)))
    y = (rng.random(n) < true).astype(float)
    res = W.logit_slope(q, y, np.arange(n) // 2)
    assert res["b"] == pytest.approx(1.3, abs=0.06) and res["se_b"] < 0.05


def test_gini_and_lorenz():
    assert W.gini(np.ones(10)) == pytest.approx(0.0)
    assert W.gini(np.r_[np.zeros(99), 1.0]) == pytest.approx(0.99, abs=1e-9)
    x, y = W.lorenz(np.array([1.0, 1.0, 2.0]))
    assert x[0] == 0 and y[-1] == pytest.approx(1.0) and y[1] == pytest.approx(0.25)


def test_lsv_herding_zero_for_binomial_flow():
    rng = np.random.default_rng(2)
    n = np.full(4000, 20)
    k = rng.binomial(n, 0.6)
    h = W.lsv_herding(k, n, np.full(4000, 0.6))
    assert abs(h.mean()) < 0.01
    k2 = np.where(rng.random(4000) < 0.5, n, 0)        # all-or-nothing windows: strong herding
    assert W.lsv_herding(k2, n, np.full(4000, 0.5)).mean() > 0.3


def test_grouped_cl_matches_clustered():
    from albert.research.flb_common import clustered
    rng = np.random.default_rng(3)
    n = 3000
    df = pl.DataFrame({"g": rng.choice(["a", "b"], n), "ev": rng.integers(0, 150, n),
                       "x": rng.normal(0.1, 1, n), "w": rng.uniform(0.5, 2, n)})
    got = W.grouped_cl(df, ["g"], "x", "w").sort("g")
    for r in got.iter_rows(named=True):
        s = df.filter(pl.col("g") == r["g"])
        ref = clustered(s["x"].to_numpy(), s["ev"].to_numpy(), s["w"].to_numpy())
        assert r["n"] == ref["n"] and r["clusters"] == ref["clusters"]
        assert r["mean"] == pytest.approx(ref["mean"]) and r["se"] == pytest.approx(ref["se"])
