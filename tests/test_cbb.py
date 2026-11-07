from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from albert.research import cbb
from albert.venues import kalshi as k


def _games(fixture) -> pl.DataFrame:
    """Recorded ESPN men's college basketball scoreboard (2026-02-19, D-I feed), two games."""
    return pl.DataFrame(cbb.game_rows(fixture("espn/scoreboard_cbb.json")), schema=cbb.GAME_SCHEMA)


def _markets(fixture) -> pl.DataFrame:
    """Recorded Kalshi /historical/markets legs of Lindenwood at Tennessee St. (trimmed)."""
    m = pl.DataFrame(k.market_rows(fixture("kalshi/cbb_markets.json"), {}, "fixture"), schema=k.MARKET_SCHEMA)
    return m.with_columns((pl.col("result") == "yes").cast(pl.Int8).alias("y"))


def test_scoreboard_rows_carry_box_totals(fixture):
    g = _games(fixture)
    assert g.height == 2
    r = g.filter(pl.col("game_id") == "401806493").row(0, named=True)
    assert r["home_loc"] == "Tennessee State" and r["away_loc"] == "Lindenwood"
    assert r["completed"] is True and r["status"] == "STATUS_FINAL" and r["season"] == 2026
    assert r["tip"].tzinfo is not None and r["neutral"] is False
    for side in ("home", "away"):
        assert r[f"{side}_score"] > 40 and r[f"{side}_fga"] > 40 and r[f"{side}_fta"] >= 0
        assert r[f"{side}_conf"] is not None


def test_possessions_estimate_is_sane(fixture):
    g = _games(fixture)
    tiers = {(2026, t): "low" for t in g["home_id"].to_list() + g["away_id"].to_list()}
    mg = cbb.model_games(g, tiers)
    assert mg.height == 2
    assert mg["poss"].is_between(55, 90).all() and mg["poss40"].is_between(55, 90).all()
    eff = 100 * (mg["home_score"] + mg["away_score"]) / (2 * mg["poss"])
    assert eff.is_between(80, 135).all()


def test_match_and_ladder_results_agree_with_espn_scores(fixture):
    g, m = _games(fixture), _markets(fixture)
    match = cbb.match_events(m.filter(pl.col("series_ticker") == "KXNCAAMBGAME"), g)
    assert match.height == 1 and match["game_id"][0] == "401806493"
    legs = cbb.ladder_legs(m, match, g)
    assert set(legs["kind"]) == {"ml", "spread", "total"}
    assert legs.height == m.height
    assert legs.filter(pl.col("kind") != "ml")["strike"].is_not_null().all()
    assert (legs["y"] == legs["y_espn"]).all()
    ml = legs.filter(pl.col("kind") == "ml")
    assert sorted(ml["team_side"].to_list()) == ["away", "home"]


def test_sampling_is_seeded_and_stratified(fixture):
    g = pl.DataFrame({"game_id": [str(i) for i in range(40)],
                      "tier": ["low"] * 20 + ["high"] * 20,
                      "tip": [datetime(2025, 12, 1, tzinfo=UTC) + timedelta(days=4 * i) for i in range(40)]})
    a = cbb.sample_games(g, {"low": 3, "mid": 3, "high": 2}, seed=1)
    b = cbb.sample_games(g, {"low": 3, "mid": 3, "high": 2}, seed=1)
    assert a.to_list() == b.to_list()
    # cells: low dev (16 games -> 3), low holdout (4 -> 3), high holdout (20 -> 2)
    assert a.len() == 8
    assert a.filter(a.cast(pl.Int64) >= 20).len() == 2


def _synthetic(n_teams=8, n_games=80, seed=5):
    """Pure math: random schedule with known strengths and tempos (no venue data)."""
    rng = np.random.default_rng(seed)
    o, d, tau = rng.normal(0, 6, n_teams), rng.normal(0, 6, n_teams), rng.normal(0, 3, n_teams)
    rows = []
    t0 = datetime(2025, 11, 3, 23, tzinfo=UTC)
    for i in range(n_games):
        h, a = (int(x) for x in rng.choice(n_teams, 2, replace=False))
        poss = 68 + tau[h] + tau[a] + rng.normal(0, 4)
        eh = 105 + o[h] - d[a] + 2 + rng.normal(0, 10)
        ea = 105 + o[a] - d[h] - 2 + rng.normal(0, 10)
        rows.append({"game_id": str(i), "season": 2026, "tip": t0 + timedelta(hours=8 * i), "neutral": False,
                     "home_id": str(h), "away_id": str(a), "home_score": eh * poss / 100, "away_score": ea * poss / 100,
                     "poss": poss, "poss40": poss})
    truth = {"o": o, "d": d, "tau": tau}
    return pl.DataFrame(rows), {(2026, str(i)): "low" for i in range(n_teams)}, truth


def test_joint_update_equals_sequential_updates():
    games, tiers, _ = _synthetic()
    last = games.tail(1)
    final_q = (last.select("game_id", "season", "home_id", "away_id", "neutral")
               .with_columns((last["tip"] + timedelta(days=3)).alias("t")))
    # Seed every team before the first result (new teams start at the tier mean of the moment,
    # which differs between batched and sequential runs; the update math must not).
    seed = pl.DataFrame({"game_id": [f"s{i}" for i in range(4)], "season": [2026] * 4,
                         "home_id": ["0", "2", "4", "6"], "away_id": ["1", "3", "5", "7"], "neutral": [True] * 4,
                         "t": [games["tip"].min() - timedelta(days=1)] * 4})
    final_q = pl.concat([seed, final_q])
    for mode in ("eff", "pts", "massey"):
        prm = cbb.EffParams(mode=mode, q_day=0.0, q_day_tempo=0.0, mu=105.0 if mode == "eff" else 72.0)
        batched = cbb.run_model(games, final_q, prm, tiers).filter(pl.col("game_id") == last["game_id"][0])
        # a query between every result forces one Kalman update per game
        between = (games.select("game_id", "season", "home_id", "away_id", "neutral")
                   .with_columns((games["tip"] + timedelta(hours=7)).alias("t"), (pl.col("game_id") + "q").alias("game_id")))
        seq = cbb.run_model(games, pl.concat([final_q, between]), prm, tiers).filter(pl.col("game_id") == last["game_id"][0])
        for col in ("mu_margin", "sd_margin"):
            assert abs(batched[col][0] - seq[col][0]) < 1e-6, (mode, col)


def test_efficiency_model_learns_strength_order():
    games, tiers, truth = _synthetic(n_games=600)
    teams = [str(i) for i in range(8)]
    q = pl.DataFrame({"game_id": [f"q{i}" for i in range(8)], "season": [2026] * 8, "home_id": teams,
                      "away_id": teams[1:] + teams[:1], "neutral": [True] * 8,
                      "t": [games["tip"].max() + timedelta(days=1)] * 8})
    prm = cbb.EffParams(mode="eff", mu=105.0, q_day=0.0, q_day_tempo=0.0)
    pred = cbb.run_model(games, q, prm, tiers)
    assert pred.height == 8
    o, d, tau = truth["o"], truth["d"], truth["tau"]
    a = np.arange(8)
    b = np.roll(a, -1)
    true_margin = 68 * ((o[a] - d[b]) - (o[b] - d[a])) / 100
    true_pace = 68 + tau[a] + tau[b]
    assert np.corrcoef(pred["mu_margin"].to_numpy(), true_margin)[0, 1] > 0.8
    assert np.corrcoef(pred["poss_exp"].to_numpy(), true_pace)[0, 1] > 0.8
    assert pred["mu_total"].is_between(100, 180).all() and (pred["sd_margin"] > 5).all()


def test_vectorised_maker_fills_equal_the_harness_rule():
    """Pure logic: the vectorised trade-through fills equal fundamental.maker_fills, and the
    lenient trade-at rule fills at least as often."""
    from albert.research.fundamental import maker_fills

    t0 = datetime(2026, 2, 1, 12, tzinfo=UTC)
    rng = np.random.default_rng(2)
    rows, trades = [], []
    for i in range(30):
        tk = f"T{i}"
        mid = float(rng.uniform(0.15, 0.85))
        bid, ask = round(mid - 0.02, 2), round(mid + 0.02, 2)
        rows.append({"ticker": tk, "event": f"g{i // 2}", "t": t0, "end": t0 + timedelta(hours=2),
                     "p_fair": float(np.clip(mid + rng.normal(0, 0.05), 0.02, 0.98)), "bid": bid, "ask": ask,
                     "y": int(rng.random() < mid), "fee_type": "quadratic_with_maker_fees", "fee_multiplier": 1.0,
                     "mid": mid, "holdout": False, "kind": "ml"})
        for j in range(int(rng.integers(0, 8))):
            trades.append({"ticker": tk, "created_time": t0 + timedelta(minutes=int(rng.integers(-30, 150))),
                           "yes_price": round(float(np.clip(mid + rng.normal(0, 0.05), 0.01, 0.99)), 2),
                           "taker_side": "yes" if rng.random() < 0.5 else "no", "trade_id": f"{i}-{j}"})
    df, tr = pl.DataFrame(rows), pl.DataFrame(trades)
    F = cbb.maker_fills_all(df, tr, ["kind"], margins=(0.0, 0.02))
    assert F.height > 0
    for m in (0.0, 0.02):
        ref = maker_fills(df.rename({"p_fair": "p"}), tr, m).sort("ticker", "side")
        got = F.filter((pl.col("margin") == m) & (pl.col("rule") == "through")).sort("ticker", "side")
        assert np.allclose(ref["price"].to_numpy(), got["price"].to_numpy())
        assert ref["filled"].to_list() == got["filled"].to_list()
        assert np.allclose(ref["pnl"].to_numpy(), got["pnl"].to_numpy())
        at = F.filter((pl.col("margin") == m) & (pl.col("rule") == "at"))
        assert at["filled"].sum() >= got["filled"].sum()
