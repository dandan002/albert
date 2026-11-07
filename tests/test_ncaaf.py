from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from albert.research import ncaaf as nc
from albert.venues import kalshi as k


def _games(fixture) -> pl.DataFrame:
    """Recorded ESPN scoreboard (2026 week 9, FBS feed), trimmed to two games."""
    return pl.DataFrame(nc.game_rows(fixture("espn/scoreboard_ncaaf.json"), "80"), schema=nc.GAME_SCHEMA)


def _markets(fixture) -> pl.DataFrame:
    """Recorded Kalshi /markets pages of the four series, trimmed to the same two games."""
    rows = k.market_rows(fixture("kalshi/ncaaf_markets.json"), {}, "fixture")
    m = pl.DataFrame(rows, schema=k.MARKET_SCHEMA)
    return m.with_columns((pl.col("result") == "yes").cast(pl.Int8).alias("y"))


def test_scoreboard_rows(fixture):
    g = _games(fixture)
    assert g.height == 2
    fla = g.filter(pl.col("home_abbr").is_in(["UGA", "FLA"])).row(0, named=True)
    assert fla["neutral"] is True and fla["completed"] is True and fla["season"] == 2026
    assert fla["kickoff"].tzinfo is not None and fla["kickoff"].minute in (0, 30)
    assert fla["home_score"] is not None and fla["away_score"] is not None
    usu = g.filter(pl.col("home_abbr") == "USU").row(0, named=True)
    assert usu["away_abbr"] == "CSU" and usu["neutral"] is False


def test_event_key_and_names():
    assert nc.event_key("KXNCAAFSPREAD-26AUG27MHUYSU") == (datetime(2026, 8, 27), "MHUYSU")
    assert nc.event_key("KXNCAAFGAME-26OCT31FLAUGA") == (datetime(2026, 10, 31), "FLAUGA")
    assert nc.norm_name("Youngstown St.") == nc.norm_name("Youngstown State")
    assert nc.norm_name("St. Thomas") == "saint thomas"
    assert nc.norm_name("North Carolina St.") == nc.norm_name("NC State")
    assert nc.norm_name("Miami (OH)") == "miami oh"


def test_match_and_ladder_results_agree_with_espn_scores(fixture):
    g, m = _games(fixture), _markets(fixture)
    match = nc.match_events(m.filter(pl.col("series_ticker") == "KXNCAAFGAME"), g)
    assert match.height == 2 and match["game_id"].null_count() == 0
    legs = nc.ladder_legs(m, match, g)
    assert set(legs["kind"]) == {"ml", "spread", "total", "team_total"}
    assert legs.height == m.height
    # Every leg's settled result equals what ESPN's final score implies (parsing check).
    assert (legs["y"] == legs["y_espn"]).all()
    # The moneyline legs map to opposite sides of the same game.
    ml = legs.filter(pl.col("kind") == "ml")
    assert ml.group_by("game_id").agg(pl.col("team_side").n_unique())["team_side"].to_list() == [2, 2]


def test_odds_row(fixture):
    row = nc.odds_row(fixture("espn/odds_ncaaf.json"), "401864494")
    assert row["provider"] == "DraftKings"
    assert row["spread_open"] == -35.5 and row["spread_close"] == -37.5
    assert row["total_open"] == 57.5 and row["total_close"] == 61.5
    assert row["ml_home_close"] == -100000 and row["ml_away_close"] == 5000


def _synthetic_games(n_teams=8, n_games=60, seed=3):
    """Pure math check: random round-robin with known strengths (no venue data involved)."""
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 10, n_teams)
    rows = []
    t0 = datetime(2025, 9, 1, tzinfo=UTC)
    for i in range(n_games):
        h, a = rng.choice(n_teams, 2, replace=False)
        margin = 3 + strength[h] - strength[a] + rng.normal(0, 12)
        rows.append({"game_id": str(i), "season": 2025, "kickoff": t0 + timedelta(hours=6 * i), "completed": True,
                     "neutral": False, "home_id": str(h), "away_id": str(a),
                     "home_score": 30 + margin / 2, "away_score": 30 - margin / 2})
    return pl.DataFrame(rows)


def test_kalman_without_drift_equals_ridge_with_prior():
    g = _synthetic_games()
    prm = nc.KFParams(split=False, sigma=12.0, q_day=0.0, p0=100.0, home=3.0,
                      tier_init=(("P4", 0.0), ("G5", 0.0), ("FCS", 0.0), ("other", 0.0)))
    # Queries before the first game introduce every team at the prior mean (0), then one after
    # the last game asks for team 0 at home to team 1.
    t0, t1 = g["kickoff"].min() - timedelta(hours=1), g["kickoff"].max() + timedelta(days=1)
    q = pl.DataFrame([{"game_id": f"q{i}", "season": 2025, "home_id": str(2 * i), "away_id": str(2 * i + 1),
                       "neutral": False, "t": t0} for i in range(4)]
                     + [{"game_id": "final", "season": 2025, "home_id": "0", "away_id": "1", "neutral": False, "t": t1}])
    pred = nc.run_kf(g, q, prm, {}).filter(pl.col("game_id") == "final").row(0, named=True)
    # Closed form: posterior mean of (r, h) with prior r ~ N(0, p0), h ~ N(3, 1), noise sigma^2.
    teams = sorted(set(g["home_id"]) | set(g["away_id"]))
    idx = {t: i for i, t in enumerate(teams)}
    X = np.zeros((g.height, len(teams) + 1))
    for j, r in enumerate(g.iter_rows(named=True)):
        X[j, idx[r["home_id"]]], X[j, idx[r["away_id"]]], X[j, -1] = 1.0, -1.0, 1.0
    y = (g["home_score"] - g["away_score"]).to_numpy()
    prior_prec = np.diag([1 / 100.0] * len(teams) + [1.0])
    prior_mean = np.array([0.0] * len(teams) + [3.0])
    A = X.T @ X / 144.0 + prior_prec
    beta = np.linalg.solve(A, X.T @ y / 144.0 + prior_prec @ prior_mean)
    want = beta[idx["0"]] - beta[idx["1"]] + beta[-1]
    assert pred["mu_margin"] == pytest.approx(want, abs=1e-6)


def test_query_never_sees_a_game_that_has_not_finished():
    g = _synthetic_games()
    prm = nc.KFParams(split=False, sigma=12.0, q_day=0.01)
    last = g.sort("kickoff").tail(1).row(0, named=True)
    # A query 1 h after the last kickoff (result not public until kickoff + 4.5 h) ...
    q = pl.DataFrame([{"game_id": last["game_id"], "season": 2025, "home_id": last["home_id"],
                       "away_id": last["away_id"], "neutral": False, "t": last["kickoff"] + timedelta(hours=1)}])
    with_game = nc.run_kf(g, q, prm, {}).row(0, named=True)
    without = nc.run_kf(g.filter(pl.col("game_id") != last["game_id"]), q, prm, {}).row(0, named=True)
    assert with_game["mu_margin"] == pytest.approx(without["mu_margin"])
    # ... while a query after kickoff + 4.5 h does see it.
    q_late = q.with_columns((pl.col("t") + timedelta(hours=4)).alias("t"))
    assert nc.run_kf(g, q_late, prm, {})["mu_margin"][0] != pytest.approx(without["mu_margin"])


def test_dist_fit_and_probabilities():
    rng = np.random.default_rng(0)
    z = rng.standard_t(5, 20000) * 1.3
    fit = nc.fit_dist(z)
    assert fit["ll_t"] > fit["ll_normal"]
    assert 3.5 < fit["t"].df < 7.5 and fit["t"].scale == pytest.approx(1.3, rel=0.08)
    d = nc.Dist(1.0, None)
    assert d.sf(np.array([0.0]))[0] == pytest.approx(0.5)


def test_implied_location_recovers_ladder_center():
    from scipy import stats

    strikes = np.arange(-20.5, 21.0, 3.0)
    mids = stats.norm.sf((strikes - 6.0) / 15.0)
    up = np.ones(strikes.size, dtype=bool)
    loc, scale = nc.implied_location(strikes, up, mids, np.full(strikes.size, 0.02), 16.0)
    assert loc == pytest.approx(6.0, abs=0.05) and scale == pytest.approx(15.0, abs=0.1)


def test_kelly_ladder_bets_only_with_an_edge_after_fees():
    grid = np.arange(-60, 61)
    pmf = nc.outcome_pmf(7.0, 15.0, nc.Dist(1.0, None), grid)
    p_over = float(pmf[grid > 3.5].sum())
    fair = pl.DataFrame([{"ticker": "A", "x_strike": 3.5, "upper": True, "bid": round(p_over - 0.01, 2),
                          "ask": round(p_over + 0.01, 2)}])
    R, _, _ = nc.ladder_options(fair, grid)
    assert nc.kelly(R, pmf, 0.25).sum() == 0          # fair price: the fee and spread kill the bet
    cheap = fair.with_columns(pl.lit(0.40).alias("bid"), pl.lit(0.42).alias("ask"))
    R, _, labels = nc.ladder_options(cheap, grid)
    f = nc.kelly(R, pmf, 0.25)
    assert f[labels.index("A:yes")] > 0 and f[labels.index("A:no")] == 0
    single = nc.single_kelly(R, pmf, 0.25)
    assert single == pytest.approx(f, abs=1e-3)       # one contract: both choices agree


def test_implied_location_refuses_a_ladder_that_stops_short_of_the_line():
    from scipy import stats

    strikes = np.arange(2.5, 23.0, 4.0)                 # a 35-point favourite
    mids = np.clip(stats.norm.sf((strikes - 35.0) / 8.0), 0.0, 0.99)  # 94-99c
    up = np.ones(strikes.size, dtype=bool)
    assert nc.implied_location(strikes, up, mids, np.full(strikes.size, 0.02), 16.0) is None
