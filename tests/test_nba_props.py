"""NBA props study: parsers on trimmed real payloads (ESPN scoreboard / summary of DAL @ DEN,
2026-03-25; Kalshi historical markets and candles of the same game) and the count-law math."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from albert.research import nba_props as n

FIX = Path(__file__).parent / "fixtures"


def _games() -> pl.DataFrame:
    payload = json.loads((FIX / "espn" / "nba_scoreboard_20260325.json").read_text())
    return pl.DataFrame(n.game_rows(payload), schema=n.GAME_SCHEMA)


def _boxes() -> tuple[pl.DataFrame, pl.DataFrame]:
    payload = json.loads((FIX / "espn" / "nba_summary_401810914.json").read_text())
    p, t = n.box_rows("401810914", payload)
    return pl.DataFrame(p, schema=n.BOX_SCHEMA), pl.DataFrame(t, schema=n.TEAM_BOX_SCHEMA)


def _markets() -> pl.DataFrame:
    payload = json.loads((FIX / "kalshi" / "nba_props_markets_26MAR25DALDEN.json").read_text())
    info = {"fee_type": "quadratic", "fee_multiplier": 1}
    return pl.DataFrame([n.market_row(m, info) for m in payload["markets"]], schema=n.MARKET_SCHEMA)


def test_scoreboard_rows():
    g = _games()
    assert g.height >= 1
    r = g.filter(pl.col("game_id") == "401810914").row(0, named=True)
    assert (r["away_abbr"], r["home_abbr"]) == ("DAL", "DEN")
    assert r["tip"] == datetime(2026, 3, 26, 2, tzinfo=UTC)
    assert r["completed"] and r["home_score"] == 142 and r["away_score"] == 135


def test_box_rows_players_and_teams():
    p, t = _boxes()
    assert p.height >= 8 and t.height == 2
    played = p.filter(~pl.col("dnp"))
    assert played.height >= 6
    assert played["min"].min() > 0 and played["pts"].is_not_null().all()
    # FG / 3PT / FT are "made-attempted" strings
    assert (played["fg3m"] <= played["fg3a"]).all() and (played["fgm"] <= played["fga"]).all()
    assert p.filter(pl.col("dnp"))["min"].is_null().all()
    assert t["fga"].min() > 50 and t["tov"].is_not_null().all() and t["oreb"].is_not_null().all()


def test_market_rows_and_matching():
    m = _markets()
    assert m.height >= 9
    assert set(m["series_ticker"]) >= {"KXNBAPTS", "KXNBATEAMTOTAL", "KXNBATOTAL"}
    assert m["floor_strike"].is_not_null().all() and m["result"].is_in(["yes", "no"]).all()
    p, _ = _boxes()
    legs = n.legs_frame(m, _games(), p)
    assert legs.height == m.height
    assert (legs["game_id"] == "401810914").all()
    pts = legs.filter(pl.col("kind") == "pts")
    assert pts.height and (pts["player"] == "Cooper Flagg").all() and pts["athlete_id"].is_not_null().all()
    assert (pts["team_side"] == "away").all()
    # "20+" pays when the stat is >= 20 (floor_strike 19.5); check against the ESPN box
    flagg = p.filter(pl.col("name") == "Cooper Flagg").row(0, named=True)
    for r in pts.iter_rows(named=True):
        assert r["y"] == int(flagg["pts"] >= r["k"])
    tot = legs.filter(pl.col("kind") == "total")
    for r in tot.iter_rows(named=True):
        assert r["y"] == int(142 + 135 >= r["k"])
    tt = legs.filter(pl.col("kind") == "team_total")
    for r in tt.iter_rows(named=True):
        assert r["team_side"] == "away" and r["y"] == int(135 >= r["k"])


def test_candle_rows():
    payload = json.loads((FIX / "kalshi" / "nba_props_candles_historical.json").read_text())
    rows = n.candle_rows(payload, "KXNBAPTS-26MAR25DALDEN-DALKTHOMPSON31-15")
    assert len(rows) == 6
    assert all(r["bid"] is not None and r["ask"] is not None and r["ask"] > r["bid"] for r in rows)
    traded = [r for r in rows if (r["volume"] or 0) > 0]
    assert traded and all(r["price_low"] <= r["price_high"] for r in traded)


def test_names_and_events():
    assert n.norm_person("Nikola Jokić") == "nikola jokic"
    assert n.norm_person("P.J. Washington Jr.") == "pj washington"
    assert n.parse_event("KXNBAPTS-26JAN06LALNOP") == (date(2026, 1, 6), "LAL", "NOP")
    assert n.parse_event("KXNBAPTS-26XYZ06LALNOP") is None


def test_nb_pmf_sums_to_one():
    x = np.arange(0, 400)
    for mu, r in ((3.0, 2.0), (25.0, 7.0), (0.4, 50.0)):
        assert abs(np.exp(n.nb_logpmf(x, np.full(len(x), mu), r)).sum() - 1) < 1e-6


def _model() -> n.PropModel:
    beta = np.zeros(14)
    beta[1] = 1.0   # minutes = recent average
    prior = {(p, r): {"pts": 0.45, "reb": 0.2, "ast": 0.1, "fg3a": 0.15} for p in "GFC" for r in (False, True)}
    return n.PropModel(beta_min=beta, sd_min={i: 4.0 for i in range(5)}, rookie_min=(15.0, 8.0), lam=0.98, m0=100.0,
                       prior=prior, p3_lg=0.36, a0=100.0, calib={c: 1.0 for c in ("pts", "reb", "ast", "fg3a")},
                       r={"pts": 8.0, "reb": 40.0, "ast": 30.0, "fg3a": 400.0, "pra": 20.0}, adjust=False)


@pytest.mark.parametrize("kind", ["pts", "reb", "ast", "fg3m", "pra"])
def test_prop_probs_monotone_in_strike(kind):
    ks = [1.0, 3.0, 5.0, 10.0, 20.0, 30.0]
    df = pl.DataFrame({"kind": [kind] * len(ks), "k": ks, "n_hist": [20] * len(ks), "m_last5": [32.0] * len(ks),
                       "pos": ["G"] * len(ks), "m_season": [32.0] * len(ks), "M_0.98": [600.0] * len(ks),
                       "S_pts_0.98": [300.0] * len(ks), "S_reb_0.98": [120.0] * len(ks), "S_ast_0.98": [90.0] * len(ks),
                       "S_fg3a_0.98": [180.0] * len(ks), "S_fg3m_0.98": [70.0] * len(ks)})
    p = n.prop_probs(df, _model())
    assert np.all(np.diff(p) <= 1e-12) and p[0] > 0.5 and p[-1] < 0.5
    assert np.all((p > 0) & (p < 1))


def test_minutes_law_conditions_on_playing():
    df = pl.DataFrame({"n_hist": [10], "m_last5": [3.0]})
    nodes, w = n.minutes_law(df, _model())
    assert abs(w.sum() - 1) < 1e-9 and nodes.min() >= 1 and nodes.max() <= 48
