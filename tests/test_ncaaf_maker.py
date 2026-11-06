from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from albert.research import ncaaf_maker as nm
from albert.research.fundamental import maker_fee, maker_fills, taker_fee
from albert.venues import kalshi as k


def _trades(fixture) -> pl.DataFrame:
    """Recorded /historical/trades page of one FCS spread leg (18 prints, both taker sides)."""
    return pl.DataFrame(k.trade_rows(fixture("kalshi/ncaaf_trades_spread.json")))


def _pred(tr: pl.DataFrame, ps, *, bid: float = 0.80, ask: float = 0.88, y: int = 1,
          t: datetime | None = None, end: datetime | None = None) -> pl.DataFrame:
    t = t or tr["created_time"].min() - timedelta(minutes=1)
    end = end or tr["created_time"].max() + timedelta(minutes=1)
    n = len(ps)
    return pl.DataFrame({"ticker": [tr["ticker"][0]] * n, "event": ["g1"] * n, "game_id": ["g1"] * n,
                         "t": [t + timedelta(seconds=i) for i in range(n)], "end": [end] * n, "p": list(ps),
                         "bid": [bid] * n, "ask": [ask] * n, "y": [y] * n,
                         "fee_type": ["quadratic_with_maker_fees"] * n, "fee_multiplier": [1.0] * n})


def test_vectorised_fills_equal_the_harness(fixture):
    tr = _trades(fixture)
    assert tr.height == 18 and set(tr["taker_side"]) == {"yes", "no"}
    pred = _pred(tr, [0.70, 0.80, 0.84, 0.845, 0.86, 0.90])
    total_fills = 0
    for m in (0.0, 0.01, 0.03):
        a = maker_fills(pred, tr, m).sort("t", "side")
        v = nm.fills(pred, tr, m).sort("t", "side")
        assert a.height == v.height > 0
        assert a["filled"].to_list() == v["filled"].to_list()
        assert np.allclose(a["price"].to_numpy(), v["price"].to_numpy())
        assert np.allclose(a["pnl"].to_numpy(), v["pnl"].to_numpy())
        assert a["fill_ts"].to_list() == v["fill_ts"].to_list()
        total_fills += int(v["filled"].sum())
    assert total_fills > 0


def test_strict_needs_a_print_through_our_bid(fixture):
    tr = _trades(fixture)
    # YES bid at 0.84: the YES-sellers at 0.83 print through it; at 0.83 nothing prints below.
    o = nm.fills(_pred(tr, [0.845, 0.835]), tr, 0.0).filter(pl.col("side") == "yes").sort("price")
    assert o["price"].to_list() == pytest.approx([0.83, 0.84])
    assert o["filled"].to_list() == [False, True]
    first_seller = tr.filter(pl.col("taker_side") == "no")["created_time"].min()
    assert o["fill_ts"][1] == first_seller
    # Filled YES at 0.84 on a leg that settled YES: 1 − 0.84 − maker fee.
    assert o["pnl"][1] == pytest.approx(1 - 0.84 - maker_fee(0.84, "quadratic_with_maker_fees"))


def test_improve_rule_fills_at_our_price_only_when_we_improved_the_quote(fixture):
    tr = _trades(fixture)
    sell = tr.filter(pl.col("taker_side") == "no").sort("created_time").row(0, named=True)
    t, end = sell["created_time"] - timedelta(seconds=5), sell["created_time"] + timedelta(seconds=1)
    assert sell["yes_price"] == pytest.approx(0.83)
    # Our YES bid of 0.83 above a 0.80 bid: first in the 0.83 queue, so the seller at 0.83 hits us.
    better = _pred(tr, [0.835], bid=0.80, ask=0.88, t=t, end=end)
    assert not nm.fills(better, tr, 0.0, "strict").filter(pl.col("side") == "yes")["filled"][0]
    assert nm.fills(better, tr, 0.0, "improve").filter(pl.col("side") == "yes")["filled"][0]
    # Joining an existing 0.83 bid: behind the queue, not filled at our own price.
    join = _pred(tr, [0.835], bid=0.83, ask=0.88, t=t, end=end)
    assert not nm.fills(join, tr, 0.0, "improve").filter(pl.col("side") == "yes")["filled"][0]


def test_post_only_cap_and_no_bid_fill(fixture):
    tr = _trades(fixture)
    # Fair 0.95 above the 0.88 ask: the YES bid is capped one tick inside the ask.
    o = nm.maker_orders(_pred(tr, [0.95]), 0.0)
    assert o.filter(pl.col("side") == "yes")["price"][0] == pytest.approx(0.87)
    # NO bid at 0.15 (YES ask 0.85): the YES-buyer at 0.86 prints through it.
    f = nm.fills(_pred(tr, [0.85], bid=0.80, ask=0.88), tr, 0.0).filter(pl.col("side") == "no")
    assert f["price"][0] == pytest.approx(0.15) and f["filled"][0]
    assert f["fill_ts"][0] == tr.filter((pl.col("taker_side") == "yes") & (pl.col("yes_price") > 0.85))["created_time"].min()
    assert f["pnl"][0] == pytest.approx(0 - 0.15 - maker_fee(0.15, "quadratic_with_maker_fees"))


def test_maker_fee_expression_matches_the_harness():
    prices = [0.01, 0.07, 0.33, 0.5, 0.84, 0.99]
    df = pl.DataFrame({"price": prices * 2, "fee_type": ["quadratic_with_maker_fees"] * 6 + ["quadratic"] * 6,
                       "fee_multiplier": [1.0] * 12})
    got = df.select(nm.maker_fee_expr(pl.col("price")).alias("fee"))["fee"].to_list()
    want = [maker_fee(p, "quadratic_with_maker_fees") for p in prices] + [0.0] * 6
    assert got == pytest.approx(want, abs=1e-12)


def test_blend_fit_recovers_coefficients_and_falls_back_for_small_groups():
    rng = np.random.default_rng(7)
    n_games, legs = 400, 10
    rows = []
    for g in range(n_games):
        for j in range(legs):
            lm, lp = rng.normal(0, 1.2), rng.normal(0, 1.2)
            pr = 1 / (1 + np.exp(-(0.1 + 0.6 * lm + 0.5 * lp)))
            rows.append({"game_id": str(g), "kind": "spread", "tgroup": "FCS", "t": datetime(2026, 9, 1, tzinfo=UTC),
                         "mid": float(1 / (1 + np.exp(-lm))), "p": float(1 / (1 + np.exp(-lp))), "y": int(rng.random() < pr)})
    for g in range(5):  # a kind with too few dev games
        rows.append({"game_id": f"x{g}", "kind": "team_total", "tgroup": "FCS", "t": datetime(2026, 9, 1, tzinfo=UTC),
                     "mid": 0.5, "p": 0.6, "y": g % 2})
    df = pl.DataFrame(rows)
    b = nm.fit_blends(df)
    a, bm, c = b["spread|FCS"]["blend"]
    assert abs(bm - 0.6) < 0.1 and abs(c - 0.5) < 0.1 and b["spread|FCS"]["t_blend"][2] > 5
    assert b["team_total|FCS"]["source"].startswith("all|FCS")
    fv = nm.apply_fair(df, b)
    assert fv.height == df.height and fv["fair"].is_between(0, 1).all()
    assert fv.select(pl.corr("fair", "mid")).item() > 0.5


def test_open_rows_parse_quote_and_size(fixture):
    page = fixture("kalshi/ncaaf_open_markets.json")
    rows = nm.open_rows(page, datetime(2026, 11, 5, 23, 30, tzinfo=UTC))
    assert len(rows) == len(page["markets"]) > 2
    r = next(r for r in rows if r["ticker"] == "KXNCAAFSPREAD-26NOV07AAMUPV-PV5")
    assert (r["yes_bid"], r["yes_ask"], r["yes_bid_size"], r["yes_ask_size"]) == (0.25, 0.35, 100.0, 55.0)
    assert r["status"] == "active" and r["quote_ts"].tzinfo is not None
    assert {x["series_ticker"] for x in rows} == {"KXNCAAFGAME", "KXNCAAFSPREAD"}


def _frozen(sides: str = "favoured", x: float = 0.02) -> dict:
    blends = {f"{kind}|{tg}": {"blend": [0.0, 0.5, 0.5], "mid_only": [0.0, 1.0]}
              for kind in nm.KINDS for tg in ("FCS", "FBS")}
    return {"blends": blends, "maker": {"source": "blend", "margin": 0.0, "x": x, "sides": sides, "tiers": ["FCS"]},
            "taker_theta": 0.0}


def test_forward_prices_record_bids_and_taker_decision(fixture):
    page = fixture("kalshi/ncaaf_open_markets.json")
    legs = pl.DataFrame(nm.open_rows(page, datetime(2026, 11, 5, 23, 30, tzinfo=UTC)), infer_schema_length=None)
    legs = legs.select("ticker", pl.col("yes_bid").alias("bid"), pl.col("yes_ask").alias("ask"),
                       pl.when(pl.col("series_ticker") == "KXNCAAFGAME").then(pl.lit("ml")).otherwise(pl.lit("spread")).alias("kind"),
                       pl.lit("FCS").alias("tier"), pl.lit("FCS").alias("tgroup"),
                       pl.lit("quadratic_with_maker_fees").alias("fee_type"), pl.lit(1.0).alias("fee_multiplier"))
    legs = legs.with_columns(pl.lit(0.5).alias("p"))
    out = nm.forward_prices(legs, _frozen())
    assert out.height == legs.height
    two = out.filter(pl.col("fair").is_not_null())
    assert two.height > 0
    # Legs without a two-sided quote get no fair value and no bids.
    one_sided = out.filter((pl.col("bid") <= 0) | (pl.col("ask") >= 1))
    assert one_sided["fair"].null_count() == one_sided.height
    r = two.filter(pl.col("ticker") == "KXNCAAFSPREAD-26NOV07AAMUPV-PV5").row(0, named=True)
    # fair = σ(0.5·logit 0.30 + 0.5·logit 0.5) ≈ 0.396 → YES bid 0.34 (post-only cap: ask − 1c), NO bid 0.60.
    assert r["mid"] == pytest.approx(0.30) and r["fair"] == pytest.approx(0.3956, abs=1e-3)
    assert r["px_yes"] == pytest.approx(0.34) and r["px_no"] == pytest.approx(0.60)
    # Only the favoured side (fair > mid → YES) is posted; its edge 9.6c ≥ x.
    assert r["post_yes"] and not r["post_no"] and r["improves_yes"]
    assert r["taker_side"] == "yes" and r["taker_price"] == pytest.approx(0.35)
    assert r["taker_edge"] == pytest.approx(r["fair"] - 0.35 - taker_fee(0.35))
    assert r["cfg_sides"] == "favoured" and r["cfg_margin"] == 0.0


def test_forward_orders_score_like_the_study_fills(fixture):
    tr = _trades(fixture)
    pred = _pred(tr, [0.845]).with_columns(pl.lit("FCS").alias("tier"), pl.lit("spread").alias("kind"),
                                           ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    o = nm.maker_orders(pred, 0.0)
    snap = pred.with_columns(pl.col("t").alias("snapshot_ts"), pl.col("end").alias("kickoff"), pl.col("p").alias("fair"),
                             pl.lit(o.filter(pl.col("side") == "yes")["price"][0]).alias("px_yes"),
                             pl.lit(o.filter(pl.col("side") == "no")["price"][0]).alias("px_no"),
                             pl.lit(True).alias("improves_yes"), pl.lit(True).alias("improves_no"),
                             pl.lit(True).alias("post_yes"), pl.lit(False).alias("post_no"))
    long = nm.forward_orders(snap)
    assert long.height == 2 and set(long["side"]) == {"yes", "no"}
    hits = nm._first_fill(long, tr, "strict")
    want = nm.fills(pred, tr, 0.0).filter("filled")
    assert sorted(hits["side"].to_list()) == sorted(want["side"].to_list())
    assert hits.height > 0


def test_trade_windows_end_at_kickoff():
    kick = datetime(2026, 10, 4, 19, tzinfo=UTC)
    df = pl.DataFrame({"ticker": ["a", "a", "b"], "kickoff": [kick, kick, kick + timedelta(hours=3)]})
    w = nm.trade_windows(df).sort("ticker")
    assert w.height == 2
    assert w["t_to"].to_list() == [kick, kick + timedelta(hours=3)]
    assert (w["t_to"] - w["t_from"]).to_list() == [timedelta(hours=24)] * 2


def _hl(fixture) -> pl.DataFrame:
    """Recorded batch hourly candles of the same leg as the trades fixture."""
    page = fixture("kalshi/ncaaf_candles_batch_spread.json")
    return pl.DataFrame(nm.hl_rows(page["markets"][0]), schema=nm.HL_SCHEMA)


def test_hourly_high_low_equal_the_trade_prints(fixture):
    tr, hl = _trades(fixture), _hl(fixture)
    by_hour = (tr.with_columns(pl.col("created_time").dt.truncate("1h").alias("start"))
               .group_by("start").agg(pl.col("yes_price").min().alias("lo"), pl.col("yes_price").max().alias("hi")))
    j = by_hour.join(hl.with_columns((pl.col("end_ts") - pl.duration(hours=1)).alias("start")), on="start")
    assert j.height == by_hour.height >= 8
    assert np.allclose(j["lo"].to_numpy(), j["price_low"].to_numpy())
    assert np.allclose(j["hi"].to_numpy(), j["price_high"].to_numpy())


def test_pseudo_trades_use_only_hours_that_start_after_t(fixture):
    hl = _hl(fixture)
    pseudo = nm.hl_pseudo_trades(hl)
    assert pseudo.height == 2 * hl["price_low"].is_not_null().sum() > 0
    tr = _trades(fixture)
    end = datetime(2026, 9, 5, 22, tzinfo=UTC)
    # The 21:00-22:00 hour printed 0.86 (a YES-buyer through a 0.15 NO bid, i.e. a 0.85 YES ask).
    on_hour = _pred(tr, [0.85], bid=0.80, ask=0.88, t=datetime(2026, 9, 5, 21, tzinfo=UTC), end=end)
    mid_hour = _pred(tr, [0.85], bid=0.80, ask=0.88, t=datetime(2026, 9, 5, 21, 30, tzinfo=UTC), end=end)
    assert nm.fills(on_hour, pseudo, 0.0).filter(pl.col("side") == "no")["filled"][0]
    # From 21:30 the hour started before t: it may hold prints before t, so it is not used.
    assert not nm.fills(mid_hour, pseudo, 0.0).filter(pl.col("side") == "no")["filled"][0]


def test_hourly_and_print_fills_agree_on_hour_aligned_orders(fixture):
    tr, pseudo = _trades(fixture), nm.hl_pseudo_trades(_hl(fixture))
    t, end = datetime(2026, 9, 5, 4, tzinfo=UTC), datetime(2026, 9, 5, 23, tzinfo=UTC)
    pred = _pred(tr, [0.80, 0.82, 0.835, 0.845, 0.86, 0.88], bid=0.70, ask=0.95, t=t, end=end)
    b = nm.fills(pred, pseudo, 0.0).sort("t", "side")
    # Every print counts regardless of its taker side (a print below a resting bid means the bid
    # was hit first), so the hourly rule equals the print rule applied to both taker sides ...
    both = pl.concat([tr.with_columns(pl.lit(x).alias("taker_side")) for x in ("yes", "no")])
    a = nm.fills(pred, both, 0.0).sort("t", "side")
    assert a["filled"].sum() > 0
    assert a["filled"].to_list() == b["filled"].to_list()
    # ... and fills at least every order the harness's taker-side rule fills.
    h = nm.fills(pred, tr, 0.0).sort("t", "side")
    assert all(hb or not ha for ha, hb in zip(h["filled"], b["filled"]))
    assert h["filled"].sum() < b["filled"].sum()


def test_choose_on_dev_uses_only_dev_rows():
    base = {"source": "blend", "rule": "strict", "tier": "FCS", "lead": 24, "x": 0.0, "sides": "both"}
    grid = pl.DataFrame([
        {**base, "margin": 0.0, "split": "dev", "fills": 100, "total_usd": -3.0},
        {**base, "margin": 0.0, "split": "holdout", "fills": 90, "total_usd": 50.0},
        {**base, "margin": 0.02, "split": "dev", "fills": 60, "total_usd": 1.5},
        {**base, "margin": 0.02, "split": "holdout", "fills": 50, "total_usd": -2.0},
        {**base, "margin": 0.05, "split": "dev", "fills": 10, "total_usd": 9.0},   # too few dev fills
        {**base, "margin": 0.05, "split": "holdout", "fills": 12, "total_usd": 9.0},
    ])
    c = nm.choose_on_dev(grid)
    assert c["margin"].unique().to_list() == [0.02]
    assert c.sort("split")["total_usd"].to_list() == [1.5, -2.0]


def test_spread_buckets_in_whole_cents():
    sp = pl.DataFrame({"s": [0.01, 0.05, 0.06, 0.10, 0.11, 0.15, 0.16, 0.98]})
    got = sp.select(nm.spread_bucket(pl.col("s")).alias("b"))["b"].to_list()
    assert got == ["a ≤5c", "a ≤5c", "b 6-10c", "b 6-10c", "c 11-15c", "c 11-15c", "d >15c", "d >15c"]
