from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from albert.research import thin_sports as T
from albert.venues import kalshi as k


def _games(fixture) -> pl.DataFrame:
    """Recorded ESPN scoreboards: volleyball 2026-10-31 and English National League 2026-10-10,
    each trimmed to two games."""
    rows = T.game_rows(fixture("espn/scoreboard_wvb.json"), "wvb") + T.game_rows(fixture("espn/scoreboard_engnl.json"), "engnl")
    return pl.DataFrame(rows, schema=T.GAME_SCHEMA)


def _legs(fixture) -> pl.DataFrame:
    """Recorded Kalshi /markets pages for the same games (all series), as study legs."""
    rows = k.market_rows(fixture("kalshi/thin_sports_markets.json"), {}, "fixture")
    m = pl.DataFrame(rows, schema=k.MARKET_SCHEMA).with_columns((pl.col("result") == "yes").cast(pl.Int8).alias("y"))
    out = []
    for r in m.iter_rows(named=True):
        lg = "wvb" if r["series_ticker"].startswith("KXNCAAWV") else "engnl"
        x = T.leg_row(r, lg)
        if x is not None:
            out.append(x)
    return pl.DataFrame(out, infer_schema_length=None)


def test_game_rows_volleyball_points_and_soccer_scores(fixture):
    g = _games(fixture)
    assert g.height == 4
    v = g.filter(pl.col("league") == "wvb")
    assert v.height == 2 and v["completed"].all()
    for r in v.iter_rows(named=True):
        assert r["home_pts"] > 40 and r["away_pts"] > 40 and r["periods"] >= 3
        assert max(r["home_score"], r["away_score"]) == 3          # best of five sets
    neb = v.filter(pl.col("away_abbr") == "NEB").row(0, named=True)
    assert neb["home_abbr"] == "MSU" and neb["start"] == datetime(2026, 10, 31, 20, 30, tzinfo=UTC)
    s = g.filter(pl.col("league") == "engnl")
    assert set(s["status"]) == {"STATUS_FULL_TIME"}
    bos = s.filter(pl.col("home_name").str.contains("Boston")).row(0, named=True)
    assert (bos["home_score"], bos["away_score"]) == (1.0, 2.0) and "Southend" in bos["away_name"]


def test_season_dates_day_and_list_calendars(fixture):
    days = T.season_dates(fixture("espn/scoreboard_wvb.json"))
    assert days and all(len(d) == 8 for d in days) and days == sorted(days)
    months = T.season_dates(fixture("espn/scoreboard_usl_calendar.json"))
    assert "202603" in months and "202610" in months and all(len(m) == 6 for m in months)


def test_leg_rows_parse_every_contract_kind(fixture):
    legs = _legs(fixture)
    assert legs.height == 27
    assert set(legs["kind"]) == {"win", "spread", "total", "btts"}
    tot = legs.filter(pl.col("ticker") == "KXENGNLTOTAL-26OCT10BOSSOU-3").row(0, named=True)
    assert tot["strike"] == 2 and tot["name_a"] == "Boston" and tot["name_b"] == "Southend"
    sp = legs.filter(pl.col("kind") == "spread").row(0, named=True)
    assert sp["side_name"] in ("Boston", "Southend") and sp["strike"] >= 1 and sp["code"] in ("BOS", "SOU")
    v = legs.filter(pl.col("league") == "wvb")
    assert v.height == 2 and v["ticker_start"][0] == datetime(2026, 10, 31, 20, 30, tzinfo=UTC)
    assert set(legs["game_key"]) == {"wvb:26OCT311630NEBMSU", "engnl:26OCT10BOSSOU", "engnl:26OCT10TAMBAA"}


def test_match_games_agrees_with_every_settled_leg(fixture):
    g, legs = _games(fixture), _legs(fixture)
    m = T.match_games(legs, g)
    assert m["game_id"].null_count() == 0
    j = m.join(g.select("league", "game_id", "home_score", "away_score"), on=["league", "game_id"])
    implied = [T.leg_outcome(r["kind"], r["side"], r["strike"], r["home_score"], r["away_score"]) for r in j.iter_rows(named=True)]
    assert implied == j["y"].to_list()
    neb = j.filter(pl.col("ticker").str.ends_with("-NEB")).row(0, named=True)
    assert neb["side"] == "away"                                   # ticker order is not home/away
    tie = j.filter(pl.col("ticker").str.ends_with("-TIE"))
    assert set(tie["side"]) == {"tie"}


def test_leg_prob_is_a_proper_distribution():
    pmf = T.score_pmf(1.6, 1.1)
    assert abs(pmf.sum() - 1) < 1e-6
    h, d, a = (T.leg_prob("win", x, None, pmf) for x in ("home", "tie", "away"))
    assert abs(h + d + a - 1) < 1e-6 and h > a
    over = [T.leg_prob("total", None, kk, pmf) for kk in range(0, 7)]
    assert all(x > y for x, y in zip(over, over[1:]))
    assert T.leg_prob("spread", "home", 0, pmf) == pytest.approx(h)
    assert T.leg_prob("btts", None, None, pmf) == pytest.approx((1 - np.exp(-1.6)) * (1 - np.exp(-1.1)))
    hw = T.leg_prob("win", "home", None, pmf, ot_home=0.5)
    assert hw == pytest.approx(h + 0.5 * d)


def _synthetic_games(rng, n_teams=8, n_games=400, kind="soccer"):
    strength = np.linspace(-0.5, 0.5, n_teams)
    rows = []
    t0 = datetime(2026, 8, 1, tzinfo=UTC)
    for i in range(n_games):
        h, a = rng.choice(n_teams, 2, replace=False)
        lh, la = np.exp(0.3 + 0.2 + strength[h] - strength[a] * 0.0), np.exp(0.3 + strength[a])
        hs, as_ = rng.poisson(lh), rng.poisson(la)
        hp, ap = 75 * (1 + strength[h] - strength[a]), 75.0
        rows.append({"league": "engnl" if kind == "soccer" else "wvb", "game_id": str(i), "season": 2026,
                     "start": t0 + timedelta(hours=6 * i), "completed": True, "status": "STATUS_FULL_TIME",
                     "neutral": False, "home_id": str(h), "home_abbr": None, "home_loc": None, "home_name": None,
                     "home_short": None, "home_score": float(hs), "home_pts": hp, "away_id": str(a), "away_abbr": None,
                     "away_loc": None, "away_name": None, "away_short": None, "away_score": float(as_), "away_pts": ap,
                     "periods": 3})
    return pl.DataFrame(rows, schema=T.GAME_SCHEMA), strength


def test_models_recover_strength_and_never_see_later_games():
    rng = np.random.default_rng(3)
    g, strength = _synthetic_games(rng)
    t = g["start"].max() + timedelta(days=1)
    q = pl.DataFrame({"game_id": ["q"], "home_id": ["7"], "away_id": ["0"], "neutral": [False], "t": [t]})
    prm = {"half_life": 365.0, "lam": 1.0, "kappa": 1.0}
    out = T.predict_queries(g, q, "engnl", prm)
    assert out["lam_h"][0] > 1.5 * out["lam_a"][0]               # strongest at home vs weakest
    # a result that ends after t - RESULT_LAG must not move the prediction
    late = g.tail(1).with_columns(pl.lit("late").alias("game_id"), pl.lit(t - timedelta(hours=1)).alias("start"),
                                  pl.lit(9.0).alias("away_score"))
    out2 = T.predict_queries(pl.concat([g, late]), q, "engnl", prm)
    assert out2["lam_h"][0] == pytest.approx(out["lam_h"][0]) and out2["lam_a"][0] == pytest.approx(out["lam_a"][0])
    gv, _ = _synthetic_games(rng, kind="volleyball")
    v = T.predict_queries(gv.with_columns(pl.lit("wvb").alias("league")), q, "wvb", prm)
    assert v["zhat"][0] > 0.3


def test_maker_fills_strict_and_lenient():
    t = datetime(2026, 10, 10, 12, tzinfo=UTC)
    pred = pl.DataFrame({"ticker": ["A"], "event": ["e"], "league": ["engnl"], "kind": ["win"], "lead": [2],
                         "t": [t], "end": [t + timedelta(hours=2)], "bid": [0.40], "ask": [0.50], "mid": [0.45],
                         "fair": [0.47], "y": [1], "fee_type": ["quadratic"], "fee_multiplier": [1.0]})
    o = T.maker_orders(pred, 0.01)
    yes = o.filter(pl.col("side") == "yes").row(0, named=True)
    no = o.filter(pl.col("side") == "no").row(0, named=True)
    assert yes["price"] == pytest.approx(0.46) and yes["improves"] is True
    assert no["price"] == pytest.approx(0.52) and no["improves"] is True    # sells YES at 0.48 < ask 0.50
    tr = pl.DataFrame({"ticker": ["A", "A", "A"], "created_time": [t - timedelta(minutes=5), t + timedelta(minutes=10),
                                                                   t + timedelta(minutes=30)],
                       "yes_price": [0.30, 0.46, 0.49], "taker_side": ["no", "no", "yes"], "count": [1.0, 1.0, 1.0]})
    f = T.fill_orders(o, tr)
    fy = f.filter(pl.col("side") == "yes").row(0, named=True)
    assert fy["fill_strict"] is None                               # the print through us came before t
    assert fy["fill_lenient"] == t + timedelta(minutes=10)         # at our improving price
    fn = f.filter(pl.col("side") == "no").row(0, named=True)
    assert fn["fill_strict"] == t + timedelta(minutes=30)          # YES buyer at 0.49 > 0.48
    assert fn["pnl_if_filled"] == pytest.approx(0 - 0.52)
    fav = T.maker_orders(pred, 0.01, sides="favoured")
    assert fav["side"].to_list() == ["yes"]
    assert T.maker_orders(pred, 0.01, gate=0.05).height == 0


def test_season_of_and_holdout():
    assert T.season_of(datetime(2026, 10, 3, tzinfo=UTC), "ncaah") == 2027
    assert T.season_of(datetime(2026, 9, 3, tzinfo=UTC), "wvb") == 2026
    assert T.holdout_from("ncaah") == T.HOCKEY_SPLIT and T.holdout_from("usl") == T.HOLDOUT
    assert T.leg_kind("KXUSLSPREAD") == "spread" and T.leg_kind("KXNCAAWVMATCH") == "win"
    assert date(2026, 10, 1) == T.HOLDOUT.date()


def test_candle_fills_use_only_whole_hours_inside_the_window():
    t = datetime(2026, 10, 10, 12, 30, tzinfo=UTC)
    pred = pl.DataFrame({"ticker": ["A"], "event": ["e"], "league": ["usl"], "kind": ["total"], "lead": [2],
                         "t": [t], "end": [t + timedelta(hours=2)], "bid": [0.30], "ask": [0.40], "mid": [0.35],
                         "fair": [0.36], "y": [0], "fee_type": ["quadratic"], "fee_multiplier": [1.0]})
    o = T.maker_orders(pred, 0.0)                                   # YES bid 0.36, NO bid 0.64 (sells YES at 0.36)
    ranges = pl.DataFrame({"ticker": ["A"] * 3,
                           "end_ts": [t + timedelta(minutes=30), t + timedelta(minutes=90), t + timedelta(minutes=150)],
                           "low": [0.20, 0.36, 0.10], "high": [0.90, 0.45, 0.20], "volume": [5.0, 3.0, 2.0]})
    f = T.candle_fill_orders(o, ranges)
    yes = f.filter(pl.col("side") == "yes").row(0, named=True)
    assert yes["fill_strict"] is None                               # 12:00-13:00 straddles t; 14:00-15:00 ends after `end`
    assert yes["fill_lenient"] == t + timedelta(minutes=90)         # printed at our improving price
    no = f.filter(pl.col("side") == "no").row(0, named=True)
    assert no["price"] == pytest.approx(0.64)
    assert no["fill_strict"] == t + timedelta(minutes=90)           # a YES print at 0.45 > 0.36 in 13:00-14:00
    assert no["pnl_if_filled"] == pytest.approx(1 - 0.64)
