from datetime import UTC, datetime, timedelta

import polars as pl

from albert.research.ratings import elo_predictions


def _event(i, a, b, winner, series="KXTTELITEMATCH"):
    start = datetime(2026, 10, 1, 12, tzinfo=UTC) + timedelta(hours=3 * i)
    stamp = (start.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York"))
             .strftime("%y%b%d%H%M").upper())
    ev = f"{series}-{stamp}{a[:3].upper()}{b[:3].upper()}"
    return [{"ticker": f"{ev}-{x[:3].upper()}", "event_ticker": ev, "series_ticker": series,
             "yes_sub_title": x, "result": "yes" if x == winner else "no",
             "close_time": start + timedelta(minutes=30)} for x in (a, b)]


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
    # A later match's map market uses the earlier (closed) match result.
    rows = _event(0, "Alpha", "Bravo", "Alpha", "KXCS2GAME") + _event(8, "Alpha", "Bravo", "Alpha", "KXCS2MAP")
    preds = elo_predictions(pl.DataFrame(rows)).sort("start")
    assert preds["elo_p"][1] > 0.5


def test_walk_forward_fits_only_on_earlier_months():
    import numpy as np

    from albert.research.ratings import walk_forward

    rng = np.random.default_rng(3)
    rows = []
    for month in (1, 2, 3, 4):
        for i in range(400):
            elo = rng.uniform(0.2, 0.8)
            mid = float(np.clip(0.5 + 0.5 * (elo - 0.5) + rng.normal(0, 0.05), 0.05, 0.95))
            truth = elo  # Elo is the true probability; the market is too timid
            rows.append({"decision_ts": datetime(2026, month, 1 + i % 27, 12, tzinfo=UTC), "mid": mid,
                         "yes_bid": mid - 0.01, "yes_ask": mid + 0.01, "elo_p": elo,
                         "y": float(rng.random() < truth), "fee_type": "quadratic", "fee_multiplier": 1.0})
    per_month, bets = walk_forward(pl.DataFrame(rows), margin=0.03, first_test="2026-02", min_train=300)
    assert per_month["month"].to_list() == ["2026-02", "2026-03", "2026-04"]
    assert per_month["train"].to_list() == [400, 800, 1200]
    assert all(w > 0 for w in per_month["elo_weight"].to_list())
    assert (per_month["brier_blend"] < per_month["brier_market"]).all()
    assert bets["pnl"].sum() > 0


def test_same_match_maps_and_match_do_not_leak_into_each_other():
    # Match + map 1 + map 2 share one start time. Alpha wins all three. With no history,
    # every prediction must be 0.5: none of these results is known an hour before start.
    rows = (_event(0, "Alpha", "Bravo", "Alpha", "KXCS2GAME")
            + _event(0, "Alpha", "Bravo", "Alpha", "KXCS2MAP")
            + [dict(r, event_ticker=r["event_ticker"] + "-2", ticker=r["ticker"].replace("-", "-2-", 1))
               for r in _event(0, "Alpha", "Bravo", "Alpha", "KXCS2MAP")])
    for r in rows:
        r["close_time"] = datetime(2026, 10, 1, 15, tzinfo=UTC)
    preds = elo_predictions(pl.DataFrame(rows))
    assert preds.height == 3 and (preds["elo_p"] == 0.5).all()
