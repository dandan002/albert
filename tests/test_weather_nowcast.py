from datetime import UTC, date, datetime

import polars as pl
import pytest

from albert.research import weather_nowcast as wn
from albert.venues import weather as wx


def _obs(fixture_text) -> pl.DataFrame:
    """Recorded IEM ASOS: NYC (Central Park), 2026-10-31, hourly METARs."""
    rows = wx.asos_rows(fixture_text("iem/asos.csv"), "NYC")
    return pl.DataFrame(rows, schema={"code": pl.String, "ts": pl.Datetime("us", "UTC"), "tmpf": pl.Float64})


def test_running_extremes_use_the_standard_time_cli_day(fixture_text):
    obs = _obs(fixture_text)
    hi = wn.running_extreme(obs, "NYC", date(2026, 10, 31), "high")
    lo = wn.running_extreme(obs, "NYC", date(2026, 10, 31), "low")
    # CLI day in EDT starts 05:00 UTC (local standard midnight): the 00:51-04:51 obs belong to Oct 30.
    assert hi[0][0] == datetime(2026, 10, 31, 5, 51, tzinfo=UTC)
    assert len(hi) == 19 and hi[-1][1] == 58.0 and lo[-1][1] == 47.0
    assert all(b[1] >= a[1] for a, b in zip(hi, hi[1:]))
    assert all(b[1] <= a[1] for a, b in zip(lo, lo[1:]))


def test_resolution_events_on_recorded_observations(fixture_text):
    obs = _obs(fixture_text)
    hi = wn.running_extreme(obs, "NYC", date(2026, 10, 31), "high")
    lo = wn.running_extreme(obs, "NYC", date(2026, 10, 31), "low")
    dead = wn.resolution("high", 55, 56, hi, margin=0)
    assert dead[0] == "dead" and dead[1] == datetime(2026, 10, 31, 17, 51, tzinfo=UTC) and dead[2] == 57
    assert wn.resolution("high", 55, 56, hi, margin=1)[1] == datetime(2026, 10, 31, 18, 51, tzinfo=UTC)
    won = wn.resolution("high", 58, float("inf"), hi, margin=0)
    assert won[0] == "won" and won[1] == datetime(2026, 10, 31, 18, 51, tzinfo=UTC)
    assert wn.resolution("high", 59, 60, hi, margin=0) is None          # still open
    low_dead = wn.resolution("low", 50, 51, lo, margin=0)
    assert low_dead[0] == "dead" and low_dead[1] == datetime(2026, 10, 31, 8, 51, tzinfo=UTC)
    assert wn.resolution("low", float("-inf"), 47, lo, margin=0)[0] == "won"


def test_taker_scoring_and_fee():
    assert wn.taker_fee(0.5) == pytest.approx(0.0175)
    assert wn.taker_fee(0.02) == pytest.approx(0.0014)
    q = pl.DataFrame({"ticker": ["A", "B", "C"], "code": ["NYC"] * 3, "day": [date(2026, 10, 1)] * 3,
                      "kind": ["high"] * 3, "what": ["dead", "dead", "won"], "y": [0.0, 1.0, 1.0],
                      "bid_5": [0.10, 0.05, None], "ask_5": [0.12, 0.07, 0.95]})
    t = wn.taker_trades(q, 5)
    assert t["pnl"].to_list() == pytest.approx([0.10 - 0.0063, 0.05 - 0.0034 - 1, 0.05 - 0.0034])
    r = wn.clustered(t)
    assert r["n"] == 3 and r["clusters"] == 1
