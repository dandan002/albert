from datetime import UTC, datetime, timedelta

import polars as pl

from albert.research.ratings import elo_predictions


def _event(i, a, b, winner, series="KXTTELITEMATCH"):
    start = datetime(2026, 10, 1, 12, tzinfo=UTC) + timedelta(hours=i)
    stamp = (start.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York"))
             .strftime("%y%b%d%H%M").upper())
    ev = f"{series}-{stamp}{a[:3].upper()}{b[:3].upper()}"
    return [{"ticker": f"{ev}-{x[:3].upper()}", "event_ticker": ev, "series_ticker": series,
             "yes_sub_title": x, "result": "yes" if x == winner else "no"} for x in (a, b)]


def test_elo_uses_only_earlier_matches_and_learns_the_stronger_player():
    rows = []
    for i in range(30):
        rows += _event(i, "Alpha", "Bravo", "Alpha")
    preds = elo_predictions(pl.DataFrame(rows)).sort("start")
    assert preds.height == 30
    assert preds["elo_p"][0] == 0.5  # first meeting: no information yet
    # Alpha is the first leg alphabetically and keeps winning: its probability rises.
    assert preds["elo_p"][-1] > preds["elo_p"][1] > 0.5
    assert preds["n_min"].to_list() == list(range(30))


def test_map_markets_share_ratings_with_match_markets():
    rows = _event(0, "Alpha", "Bravo", "Alpha", "KXCS2GAME") + _event(1, "Alpha", "Bravo", "Alpha", "KXCS2MAP")
    preds = elo_predictions(pl.DataFrame(rows)).sort("start")
    assert preds["elo_p"][1] > 0.5
