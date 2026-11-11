import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from albert.research import flb_common as C
from albert.research import flb_flow as F
from albert.venues import kalshi as k

FIX = Path(__file__).parent / "fixtures" / "kalshi"
TICKER = "KXTRUMPMENTION-26OCT05-CORN"


def _recorded_trades() -> pl.DataFrame:
    payload = json.loads((FIX / f"mentions_maker_trades_{TICKER}.json").read_text())
    return pl.DataFrame(k.trade_rows(payload), schema=F.TRADE_SCHEMA)


def _recorded_meta() -> pl.DataFrame:
    ev = json.loads((FIX / "mention_event_KXTRUMPMENTION-26OCT05.json").read_text())["event"]
    m = next(x for x in ev["markets"] if x["ticker"] == TICKER)
    close, exp = k.parse_ts(m["close_time"]), k.parse_ts(m["expected_expiration_time"])
    cut, early = F.resolution_cut(close, exp)
    return pl.DataFrame([{
        "ticker": TICKER, "event_ticker": m["event_ticker"], "group": "Mentions", "stratum": "Mentions",
        "sample": "rs", "w_design": 2.0, "y": int(m["result"] == "yes"), "mult": 1.0, "maker_fee": False,
        "open_time": k.parse_ts(m["open_time"]), "close_time": close, "exp": exp, "cut": cut,
        "strict_cut": min(exp, close) - timedelta(hours=24), "sched": None, "early": early,
        "primetime": None}], schema_overrides={"sched": pl.Datetime("us", "UTC"), "primetime": pl.Boolean})


def test_cell_stats_matches_flb_common_clustered():
    rng = np.random.default_rng(1)
    n = 600
    df = pl.DataFrame({"cell": rng.choice(["a", "b"], n), "ev": rng.integers(0, 40, n).astype(str),
                       "x": rng.normal(size=n), "w": rng.uniform(0.1, 3, n)})
    out = F.cell_stats(df, ["cell"], {"m": (pl.col("x"), pl.col("w"))}, cluster="ev",
                       sums={"rows": pl.len()})
    for cell in ("a", "b"):
        sub = df.filter(pl.col("cell") == cell)
        ref = C.clustered(sub["x"].to_numpy(), sub["ev"].to_numpy(), sub["w"].to_numpy())
        row = out.filter(pl.col("cell") == cell).row(0, named=True)
        assert row["m"] == pytest.approx(ref["mean"])
        assert row["m_se"] == pytest.approx(ref["se"])
        assert row["m_clusters"] == ref["clusters"] and row["rows"] == ref["n"]
    pooled = F.cell_stats(df, [], {"m": (pl.col("x"), pl.col("w"))}, cluster="ev")
    ref = C.clustered(df["x"].to_numpy(), df["ev"].to_numpy(), df["w"].to_numpy())
    assert pooled["m_t"][0] == pytest.approx(ref["t"])


def test_diff_stats_equals_weighted_regression_sandwich():
    rng = np.random.default_rng(2)
    n = 800
    g = rng.integers(0, 50, n)
    f = rng.random(n) < 0.4
    x = rng.normal(size=n) + 0.3 * f + 0.2 * (g % 3)
    w = rng.uniform(0.5, 2, n)
    df = pl.DataFrame({"g": g.astype(str), "f": f, "x": x, "w": w})
    out = F.diff_stats(df, [], pl.col("f"), pl.col("x"), pl.col("w"), cluster="g").row(0, named=True)
    # WLS of x on [1, f] with weights w: the slope is the difference of the weighted means, and its
    # clustered sandwich (G/(G-1)) is what diff_stats computes from the two means' influence functions
    X = np.column_stack([np.ones(n), f.astype(float)])
    W = w[:, None]
    bread = np.linalg.inv(X.T @ (W * X))
    beta = bread @ X.T @ (w * x)
    e = x - X @ beta
    _, inv = np.unique(g, return_inverse=True)
    sc = np.zeros((inv.max() + 1, 2))
    np.add.at(sc, inv, X * (w * e)[:, None])
    G = sc.shape[0]
    V = G / (G - 1) * bread @ sc.T @ sc @ bread
    assert out["diff"] == pytest.approx(beta[1])
    assert out["se"] == pytest.approx(np.sqrt(V[1, 1]))


def test_ols_cluster_recovers_coefficients():
    rng = np.random.default_rng(3)
    n = 4000
    g = rng.integers(0, 200, n)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    y = X @ np.array([0.5, 2.0, -1.0]) + rng.normal(size=n) + rng.normal(size=200)[g]
    r = F.ols_cluster(X, y, g, ["c", "a", "b"])
    assert r["a"]["coef"] == pytest.approx(2.0, abs=0.1) and r["b"]["coef"] == pytest.approx(-1.0, abs=0.1)
    assert abs(r["a"]["t"]) > 20 and r["_clusters"]["coef"] == 200


def test_segments_split_at_the_trade_cutoff():
    cut = datetime(2026, 9, 11, tzinfo=UTC)
    a, b = datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 20, tzinfo=UTC)
    assert F.segments(a, a + timedelta(days=2), cut) == [(True, None, None)]
    assert F.segments(b, b + timedelta(days=2), cut) == [(False, None, None)]
    assert F.segments(a, b, cut) == [(True, None, cut), (False, cut, None)]


def test_resolution_cut_and_scheduled_start():
    close = datetime(2026, 10, 6, 1, 47, tzinfo=UTC)
    cut, early = F.resolution_cut(close, close + timedelta(days=13))
    assert early and cut == close - F.EARLY_CLOSE_DROP
    cut, early = F.resolution_cut(close, close + timedelta(days=13), sports=True)
    assert early and cut == close - F.FINAL_DROP
    cut, early = F.resolution_cut(close, close)
    assert not early and cut == close - F.FINAL_DROP
    # the ticker's start (ET) wins; otherwise occurrence - 3 h; a rewritten occurrence is ignored
    st = F.scheduled_start("KXKBOGAME-26NOV040430SAMKTW", None, None)
    assert st == datetime(2026, 11, 4, 9, 30, tzinfo=UTC)
    occ = datetime(2026, 9, 14, 3, 0, tzinfo=UTC)
    assert F.scheduled_start("KXNFLGAME-26SEP13KCPHI", occ, occ) == occ - timedelta(hours=3)
    odd = occ.replace(second=17)
    assert F.scheduled_start("KXNFLGAME-26SEP13KCPHI", odd, occ) == occ - timedelta(hours=3)
    assert F.league_of("KXNFLGAME") == "NFL" and F.league_of("KXATPMATCH") == "Tennis"
    assert F.league_of("KXEPLGAME") == "Soccer" and F.league_of("KXF1RACE") == "Other sports"


def test_draw_sample_is_seeded_volume_blind_and_capped():
    rows = []
    for st, n_ev in (("A", 30), ("B", 5)):
        for e in range(n_ev):
            for leg in range(25 if e == 0 else 2):
                rows.append({"stratum": st, "event_ticker": f"{st}-{e}", "ticker": f"{st}-{e}-{leg}",
                             "volume": float(leg), "close_time": datetime(2026, 9, 1, tzinfo=UTC)})
    fr = pl.DataFrame(rows)
    s1 = F.draw_sample(fr, {"A": 10, "B": 10})
    s2 = F.draw_sample(fr, {"A": 10, "B": 10})
    assert s1.equals(s2)
    ev = s1.group_by("stratum").agg(pl.col("event_ticker").n_unique())
    assert dict(ev.iter_rows()) == {"A": 10, "B": 5}
    assert s1.group_by("event_ticker").len()["len"].max() <= F.MAX_LEGS
    assert (s1["volume"] > 0).all()   # untraded legs are never drawn


def test_build_chunk_on_recorded_prints():
    tr = _recorded_trades()
    meta = _recorded_meta()
    pr, hours = F.build_chunk(tr, meta)
    assert pr.height == tr.height > 0
    # taker side price and outcome
    no = pr.filter(~pl.col("taker_yes"))
    assert (no["q"] - (1 - no["yes_price"])).abs().max() < 1e-12
    assert (pr["won"] == np.where(pr["taker_yes"], 0, 1)).all()   # the market resolved NO
    # sweeps: prints at one timestamp with one taker side are one order; size is the order's notional
    o = pr.group_by("order_id").agg(pl.col("created_time").n_unique(), pl.col("taker_yes").n_unique(),
                                    (pl.col("count") * pl.col("q")).sum().alias("n"), pl.col("notional").first())
    assert (o["created_time"] == 1).all() and (o["taker_yes"] == 1).all()
    assert (o["n"] - o["notional"]).abs().max() < 1e-9
    assert pr["order_id"].n_unique() < pr.height    # the fixture has multi-fill sweeps
    sb = pl.DataFrame({"notional": pr["notional"]}).with_columns(C.size_bucket_expr())["size_bucket"]
    assert (sb == pr["size_bucket"]).all()
    # fees: taker 0.07 q(1-q), no maker fee on this (quadratic) series
    assert (pr["fee_t"] - 0.07 * pr["q"] * (1 - pr["q"])).abs().max() < 1e-12
    assert (pr["fee_m"] == 0).all()
    # the recorded prints span 4.6 h to 2.8 h before an early close: the cut (close - 4 h) splits
    # them, and all are within 24 h of the close (not strict)
    cut = meta["cut"][0]
    assert (pr["pre_res"] == (pr["created_time"] < cut)).all()
    assert pr["pre_res"].any() and not pr["pre_res"].all() and not pr["strict"].any()
    # gross PnL per contract of taker and maker are mirror images
    pos = F.positions(pr.lazy()).collect()
    tk = pos.filter(pl.col("role") == "taker")
    mk = pos.filter(pl.col("role") == "maker")
    assert ((tk["wo"] - tk["qo"]) + (mk["wo"] - mk["qo"])).abs().max() < 1e-12
    # mid proxy: only from both sides' prints, bounded by them
    m = pr.filter(pl.col("m_pre").is_not_null())
    assert m.height > 0 and m["m_pre"].is_between(0, 1).all()
    # no mark may look past the cut: marks at t + h are null once t + h is after it
    late = pr.filter(pl.col("created_time") + timedelta(hours=1) > cut)
    assert late["m_1h"].null_count() == late.height
    assert hours.filter(pl.col("h") + timedelta(hours=1) > cut)["m_end"].null_count() == \
        hours.filter(pl.col("h") + timedelta(hours=1) > cut).height
    assert hours.height > 0 and set(hours["ticker"]) == {TICKER}
    assert hours["vol"].sum() == pytest.approx(pr.filter(pl.col("pre_res"))["count"].sum())


def test_mid_state_and_markouts_by_hand():
    t0 = datetime(2026, 10, 1, 12, tzinfo=UTC)
    od = pl.DataFrame({"order_id": [1, 2, 3, 4], "ticker": ["X"] * 4,
                       "t": [t0, t0 + timedelta(seconds=10), t0 + timedelta(seconds=30), t0 + timedelta(hours=3)],
                       "d": [1, -1, 1, -1], "contracts": [1.0, 2.0, 3.0, 4.0], "p_last": [0.42, 0.38, 0.44, 0.30]})
    st = F.mid_state(od)
    assert st["ask"].to_list() == [0.42, 0.42, 0.44, 0.44]
    assert st["bid"].to_list()[1:] == [0.38, 0.38, 0.30]
    pre = F.state_at(od, st, timedelta(0), "pre", strict_before=True).sort("order_id")
    assert pre["m_pre"].to_list()[:3] == [None, None, pytest.approx(0.40)]
    assert pre["cumv_pre"].to_list() == [0.0, 1.0, 3.0, 6.0]
    one = F.state_at(od, st, timedelta(minutes=1), "1m").sort("order_id")
    assert one["m_1m"].to_list()[2] == pytest.approx(0.41)
    # three hours later the ask print is stale (> STALE): no mid
    assert one["m_1m"].to_list()[3] is None
