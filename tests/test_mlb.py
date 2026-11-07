"""MLB sabermetric study: parsers on trimmed real MLB Stats API / Kalshi payloads, model math."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import mlb

FIX = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


def _games() -> pl.DataFrame:
    return pl.DataFrame(mlb.parse_schedule(_load("mlb/schedule_2026-09-03.json")), infer_schema_length=None)


def _kalshi() -> pl.DataFrame:
    rows = _load("kalshi/mlb_markets_torcle_2026-09-03.json")["markets"]
    return pl.DataFrame(rows).with_columns(
        pl.col("open_time").str.to_datetime(time_zone="UTC"), pl.col("close_time").str.to_datetime(time_zone="UTC"))


def test_parse_schedule_linescore_and_probables():
    g = _games()
    assert g.height == 2
    r = g.filter(pl.col("game_pk") == 823337).row(0, named=True)
    assert (r["home_score"], r["away_score"]) == (5, 2)
    assert r["inn_home"][:8] == [2, 0, 0, 0, 0, 2, 0, 1] and r["inn_home"][8] is None   # home skipped the 9th
    assert sum(r["inn_away"]) == 2 and len(r["inn_away"]) == 9
    assert r["home_prob_pitcher"] == 669199 and r["coded_state"] == "F"
    assert r["game_date"] == datetime(2026, 9, 3, 16, 35, tzinfo=UTC)


def test_parse_boxscore_starters_and_totals():
    teams, pitchers = mlb.parse_boxscore(_load("mlb/boxscore_823337.json"), 823337)
    home = next(t for t in teams if t["side"] == "home")
    assert home["r"] == 5 and home["pa"] == 37 and home["bb"] == 7
    hp = [p for p in pitchers if p["side"] == "home"]
    assert hp[0]["pitcher_id"] == 669199 and hp[0]["gs"] == 1 and hp[0]["order"] == 0
    assert sum(p["outs"] for p in hp) == 27                       # home pitched all 9 innings
    assert sum(p["outs"] for p in pitchers if p["side"] == "away") == 24
    num, den = mlb.woba_parts(home)
    assert den > 0 and 0.2 < num / den < 0.5
    f, bf = mlb.fip_parts(hp[0])
    assert bf == 9 and f == 13 * 0 + 3 * (2 + 1) - 2 * 3


def test_parse_season_people_team():
    ps = mlb.parse_season_pitching(_load("mlb/season_pitching_2025_sample.json"))
    th = mlb.parse_team_hitting(_load("mlb/team_hitting_2025_sample.json"))
    pp = mlb.parse_people(_load("mlb/people_sample.json"))
    assert len(ps) == 5 and all(r["season"] == 2025 for r in ps)
    assert len(th) == 3 and th[0]["pa"] > 5000
    assert {r["pitcher_id"] for r in pp} == {669199, 694918} and all(r["throws"] in ("L", "R") for r in pp)


def test_codes_strikes_and_mapping():
    assert mlb.split_codes("TORCLE") == ("TOR", "CLE")
    assert mlb.split_codes("STLCING2") == ("STL", "CIN")
    assert mlb.strike_of({"yes_sub_title": "Over 8.5 runs scored"}) == 8.5
    assert mlb.strike_of({"yes_sub_title": "Cleveland wins by over 1.5 runs"}) == 1.5
    k = _kalshi()
    mapping = mlb.match_games(k, _games())
    assert mapping.height == 1 and mapping["game_pk"][0] == 824388 and mapping["home_code"][0] == "CLE"
    specs = mlb.market_specs(k, mapping)
    assert specs.height == 8                                       # KS legs are not priced
    by = {r["ticker"]: r for r in specs.iter_rows(named=True)}
    assert by["KXMLBGAME-26SEP031310TORCLE-CLE"]["side"] == "home" and by["KXMLBGAME-26SEP031310TORCLE-CLE"]["y"] == 0
    assert by["KXMLBSPREAD-26SEP031310TORCLE-CLE2"]["side"] == "home"
    assert by["KXMLBTOTAL-26SEP031310TORCLE-9"]["strike"] == 8.5 and by["KXMLBTOTAL-26SEP031310TORCLE-9"]["y"] == 1
    times = mlb.decision_times(mapping)
    assert set(times["decision"]) == {"morning", "pre1h"}
    assert times.filter(pl.col("decision") == "pre1h")["t"][0] == datetime(2026, 9, 3, 16, 10, tzinfo=UTC)


def test_starters_public_from_strikeout_markets():
    k = _kalshi()
    mapping = mlb.match_games(k, _games())
    feat = pl.DataFrame({"game_pk": [824388, 824388], "side": ["away", "home"],
                         "opp_starter_name": ["Tanner Bibee", "José Soriano"]})
    st = mlb.starters_public(k, mapping, feat)
    r = st.row(0, named=True)
    assert r["n_unlisted"] == 1                                    # only Bibee's legs are in the fixture
    assert r["starters_listed_at"] == datetime(2026, 9, 3, 4, 25, tzinfo=UTC)


def test_starter_share():
    s = mlb.starter_share(np.array([27.0, 15.0, 3.0]), sd=0.01)
    assert np.allclose(s[0], 1.0)
    assert np.allclose(s[1], [1, 1, 1, 1, 1, 0, 0, 0, 0])
    assert np.allclose(s[2], [1, 0, 0, 0, 0, 0, 0, 0, 0])
    w = mlb.starter_share(np.array([16.0]))
    assert np.all(np.diff(w[0]) <= 1e-12) and abs(w[0].sum() * 3 - 16.0) < 0.5


def test_fit_nb_recovers_parameters():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(40000), rng.normal(size=40000)])
    mu = np.exp(X @ np.array([-0.7, 0.3]))
    y = rng.poisson(rng.gamma(1 / 2.5, 2.5 * mu))
    beta, alpha, _ = mlb.fit_nb(X, y.astype(float))
    assert np.allclose(beta, [-0.7, 0.3], atol=0.05) and abs(alpha - 2.5) < 0.25


def test_simulate_game_rules_and_probabilities():
    model = {"alpha": 2.5, "alpha_x": 1.0, "delta_x": 0.8, "sg2": 0.0, "st2": 0.03}
    lam = np.full(9, 0.5)
    sim = mlb.simulate(lam, lam, model, ghost=True, n=20000, rng=np.random.default_rng(1))
    margin = sim["home"] - sim["away"]
    assert not np.any(margin == 0)                                 # no ties
    summ = mlb.summarize(sim)
    assert 0.45 < summ["p_home"] < 0.56 and 7.0 < summ["mean_total"] < 10.5
    assert abs(summ["total_pmf"].sum() - 1) < 1e-9 and abs(summ["f5_pmf"].sum() - 1) < 1e-9
    spec = {"series": "KXMLBTOTAL", "strike": 8.5}
    assert abs(mlb.prob_for(spec, summ) - float(((sim["home"] + sim["away"]) > 8.5).mean())) < 1e-9
    assert abs(mlb.prob_for({"series": "KXMLBGAME", "side": "away", "strike": None}, summ) - (1 - summ["p_home"])) < 1e-12
    p_in = mlb.prob_for({"series": "KXMLBINNINGTOTAL", "strike": 0.5, "inning": 1}, summ)
    assert 0.3 < p_in < 0.7
    strong = mlb.summarize(mlb.simulate(lam, lam * 1.3, model, ghost=True, n=20000, rng=np.random.default_rng(2)))
    assert strong["p_home"] > summ["p_home"] + 0.05


def test_taker_real_uses_series_fee_multiplier():
    df = pl.DataFrame({"series": ["KXMLBTOTAL", "KXMLBINNINGTOTAL"], "p": [0.6, 0.6], "bid": [0.48, 0.48],
                       "ask": [0.50, 0.50], "y": [1, 1], "event": ["a", "b"], "t": [datetime(2026, 9, 1, tzinfo=UTC)] * 2})
    tr = mlb.taker_real(df, 0.0)
    fees = (tr["cost"] - 0.5).to_list()
    assert abs(fees[0] - 0.0088) < 1e-9 and abs(fees[1] - 0.0175) < 1e-9


def test_parse_batting_uses_player_codes_not_final_lineup():
    box = _load("mlb/boxscore_823337.json")
    rows = mlb.parse_batting(box, 823337)
    for side in ("home", "away"):
        slots = sorted(r["slot"] for r in rows if r["side"] == side and r["slot"])
        assert slots == list(range(1, 10))
    away = {r["player_id"]: r for r in rows if r["side"] == "away"}
    # 808982 sits in the team-level battingOrder list (the final lineup) but entered as a substitute
    assert 808982 in box["teams"]["away"]["battingOrder"] and away[808982]["slot"] is None
    assert sum(r["pa"] for r in rows if r["side"] == "home") == 37


def test_parse_weather_diagnostic():
    w = mlb.parse_weather(_load("mlb/boxscore_823337.json"), 823337)
    assert w["temp"] == 77.0 and w["wind_out"] == -7.0 and not w["roof"]
