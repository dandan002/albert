from datetime import UTC, datetime, timedelta

import polars as pl

from albert.research import esports as es

T = datetime(2026, 10, 1, 12, tzinfo=UTC)


def test_series_prob():
    assert abs(es.series_prob(0.5, 3) - 0.5) < 1e-12
    assert abs(es.series_prob(0.6, 1) - 0.6) < 1e-12
    assert abs(es.series_prob(0.6, 3) - (0.6 ** 2 + 2 * 0.6 ** 2 * 0.4)) < 1e-12


def test_predictions_only_use_maps_that_ended_before_the_decision():
    ps_matches = pl.DataFrame({"match_id": [1, 2], "game": ["csgo", "csgo"], "a_id": [10, 10], "b_id": [20, 20],
                               "end_at": [T - timedelta(days=1), T + timedelta(hours=3)]})
    ps_games = pl.DataFrame({"match_id": [1, 1, 2, 2], "game": ["csgo"] * 4, "position": [1, 2, 1, 2],
                             "status": ["finished"] * 4,
                             "end_at": [T - timedelta(days=1, hours=1), T - timedelta(days=1), T + timedelta(hours=1),
                                        T + timedelta(hours=2)],
                             "winner_id": [10, 10, 20, 20], "forfeit": [False] * 4})
    mapping = pl.DataFrame({"kalshi_ticker": ["K"], "game": ["csgo"], "map_no": [2], "first_team_id": [10],
                            "second_team_id": [20], "number_of_games": [3], "start": [T]})
    p = es.predictions(mapping, ps_matches, ps_games, lead_hours=1.0)
    # Yesterday's two maps (team 10 won) count; today's maps of the same match (team 20 won) do not.
    assert p["elo_p"][0] > 0.5 and p["n_min"][0] == 2
