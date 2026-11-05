import json
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from albert.research import soccer as S
from albert.venues import kalshi as k


def _markets(fixture) -> pl.DataFrame:
    """Real settled Kalshi markets (GAME / TOTAL / SPREAD) of two EPL games, recorded 2026-11-05."""
    rows = k.market_rows(fixture("kalshi/soccer_epl_avlbre_markets.json"), {}, "test")
    return pl.DataFrame(rows, schema=k.MARKET_SCHEMA)


def test_split_home_away():
    assert S.split_home_away("AVLBRE", ["BRE", "AVL"]) == ("AVL", "BRE")
    assert S.split_home_away("LEENEW", ["LEE", "NEW"]) == ("LEE", "NEW")
    assert S.split_home_away("AVLBRE", ["AVL", "CRY"]) is None


def test_game_frame_and_scores_from_real_markets(fixture):
    m = _markets(fixture)
    games = S.game_frame(m.filter(pl.col("event_ticker").str.starts_with("KXEPLGAME")))
    assert games.height == 2
    avl = games.filter(pl.col("event") == "KXEPLGAME-26OCT10AVLBRE").row(0, named=True)
    assert (avl["home"], avl["away"], avl["outcome"]) == ("Aston Villa", "Brentford", "D")
    # kickoff = expected expiration − 3h (15:00 BST = 14:00 UTC; Polymarket gameStartTime agrees)
    assert avl["kickoff"] == datetime(2026, 10, 10, 14, tzinfo=UTC)
    sc = S.kalshi_scores(m, games).sort("event")
    got = dict(zip(sc["event"], zip(sc["hg"], sc["ag"])))
    assert got == {"KXEPLGAME-26OCT10AVLBRE": (2, 2), "KXEPLGAME-26SEP14LEENEW": (4, 1)}


def test_score_from_ladders_needs_a_unique_answer():
    yes_to = {k: k <= 3 for k in range(1, 7)}            # total exactly 3
    # margin exactly 2 contradicts an odd total: the ladders are inconsistent
    assert S.score_from_ladders("H", yes_to, {2: True, 3: False}) is None
    # "more than 1.5" NO pins the margin to 1 -> 2-1
    assert S.score_from_ladders("H", yes_to, {2: False, 3: False}) == (2, 1)
    assert S.score_from_ladders("A", yes_to, {2: True, 3: True}) == (0, 3)
    # total ladder never says NO -> total unknown
    assert S.score_from_ladders("H", {1: True, 2: True}, {2: False}) is None
    # odd total is impossible for a draw
    assert S.score_from_ladders("D", yes_to, {}) is None


def test_openfootball_rows_real_file(fixture):
    rows = S.openfootball_rows(fixture("openfootball/en.1_2026-27_trim.json"), "en.1", "2026-27")
    assert rows, "fixture must contain played matches"
    first = rows[0]
    assert (first["home"], first["away"], first["hg"], first["ag"]) == ("Arsenal FC", "Coventry City FC", 3, 0)
    assert all(r["hg"] >= 0 and r["ag"] >= 0 for r in rows)


def test_name_mapping_on_real_names(fixture):
    rows = S.openfootball_rows(fixture("openfootball/en.1_2026-27_trim.json"), "en.1", "2026-27")
    ref = sorted({r["home"] for r in rows} | {r["away"] for r in rows})
    mp = S.map_names(["Arsenal", "Coventry", "Manchester United", "Manchester City"], ref)
    assert mp.get("Arsenal") == "Arsenal FC" and mp.get("Coventry") == "Coventry City FC"
    assert mp.get("Manchester United") != mp.get("Manchester City")


def test_pm_game_rows_keeps_only_the_1x2_event(fixture):
    rows = S.pm_game_rows(fixture("polymarket/soccer_events_trim.json")["events"])
    assert len(rows) == 1
    r = rows[0]
    assert r["pm_slug"] == "epl-ast-bre-2026-10-10"
    assert {r["pm_team1"], r["pm_team2"]} == {"Aston Villa FC", "Brentford FC"}
    assert r["pm_start"] == datetime(2026, 10, 10, 14, tzinfo=UTC)
    assert r["pm_draw"] not in (r["pm_tok1"], r["pm_tok2"])


def test_pm_quote_is_only_used_after_its_bucket_ends(fixture):
    from albert.venues.polymarket import price_history_rows

    fx = fixture("polymarket/soccer_prices_history_trim.json")
    hist = pl.DataFrame(price_history_rows(fx["response"], "H"))
    hist = pl.concat([hist, hist.with_columns(pl.lit("D").alias("token_id")), hist.with_columns(pl.lit("A").alias("token_id"))])
    hist = hist.filter(pl.col("price") > 0)
    assert hist.height > 0
    last = hist["ts"].max()
    matched = pl.DataFrame({"event": ["E"], "pm_home": ["H"], "pm_draw": ["D"], "pm_away": ["A"]})
    # at last bucket start + 299 s the last bucket has not ended: the previous one is used
    for dt, expect_ts in ((299, last - timedelta(seconds=S.PM_BUCKET)), (300, last)):
        dec = pl.DataFrame({"event": ["E"], "t": [last + timedelta(seconds=dt)], "lead": [2]})
        w = S.pm_wide(matched, hist, dec)
        price = hist.filter((pl.col("token_id") == "H") & (pl.col("ts") == expect_ts))["price"][0]
        assert w["q_home"][0] == price
        assert abs(w["pm_mult_h"][0] + w["pm_mult_d"][0] + w["pm_mult_a"][0] - 1) < 1e-9


def test_devig_methods():
    q = np.array([[0.62, 0.25, 0.18]])                     # overround 1.05
    for m in ("mult", "power", "shin"):
        p = S.devig(q, m)
        assert abs(p.sum() - 1) < 1e-9 and (p > 0).all()
    mult, power, shin = (S.devig(q, m)[0] for m in ("mult", "power", "shin"))
    # power and Shin take more of the margin from the longshot than proportional scaling
    assert power[2] < mult[2] and shin[2] < mult[2]
    assert power[0] > mult[0] and shin[0] > mult[0]


def test_dixon_coles_recovers_the_stronger_team():
    rng = np.random.default_rng(5)
    n, games = 4, 600
    att = np.array([0.4, 0.0, 0.0, -0.4])
    hi, ai = rng.integers(0, n, games), rng.integers(0, n, games)
    keep = hi != ai
    hi, ai = hi[keep], ai[keep]
    hg = rng.poisson(np.exp(0.1 + 0.3 + att[hi] - 0.0))
    ag = rng.poisson(np.exp(0.1 + att[ai]))
    fit = S.fit_dc(hi, ai, hg.astype(float), ag.astype(float), np.ones(len(hi)), n, np.zeros(n), np.zeros(n), 0.5)
    assert fit["att"][0] > fit["att"][1] > fit["att"][3]
    assert 0.15 < fit["h"] < 0.45
    ph, pd, pa = S.dc_outcome_probs(1.6, 1.1, -0.05)
    assert abs(ph + pd + pa - 1) < 1e-9 and ph > pa


def test_kelly_portfolio_bets_only_with_an_edge():
    bid, ask = np.array([0.48, 0.26, 0.24]), np.array([0.50, 0.28, 0.26])
    f, g = S.kelly_portfolio(np.array([0.49, 0.27, 0.24]), bid, ask)
    assert f.sum() == 0 and g == 0
    f, g = S.kelly_portfolio(np.array([0.45, 0.36, 0.19]), bid, ask)
    assert f[1] > 0 and g > 0          # buys the underpriced draw
    assert f.sum() <= 0.25 + 1e-9


def test_pool_fits_weights_on_logs():
    rng = np.random.default_rng(1)
    p = rng.dirichlet([4, 3, 3], 3000)
    y = np.array([rng.choice(3, p=r) for r in p])
    Y = np.eye(3)[y]
    noise = rng.dirichlet([1, 1, 1], 3000)
    z = S.fit_pool([p, noise], Y)
    assert z[2] > 0.7 and abs(z[3]) < 0.2


@pytest.mark.parametrize("lead", S.LEADS)
def test_decision_frame_is_before_kickoff(fixture, lead):
    games = S.game_frame(_markets(fixture).filter(pl.col("event_ticker").str.starts_with("KXEPLGAME")))
    d = S.decision_frame(games).filter(pl.col("lead") == lead).join(games.select("event", "kickoff"), on="event")
    assert d.height == 2
    assert (d["t"] <= d["kickoff"] - timedelta(hours=lead)).all()
    assert (d["t"] > d["kickoff"] - timedelta(hours=lead + 1)).all()
    assert (d["t"].dt.minute() == 0).all()


def test_fixture_files_are_real_recordings(fixture):
    raw = fixture("kalshi/soccer_epl_avlbre_markets.json")
    assert all("rules_primary" in m and m["result"] in ("yes", "no") for m in raw["markets"])
    assert json.dumps(raw).count("professional EPL soccer game") >= 10
