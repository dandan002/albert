from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from albert.research.approval import (
    GRID,
    Params,
    Rule,
    arrival_prob,
    asof,
    bracket_tenths,
    build_context,
    forecast_mechanical,
    forecast_rw,
    kalman_house,
    ladder_kelly,
    members_at,
    parse_rasmussen,
    parse_snapshot,
    poll_versions,
    rcp_mean,
    track,
    with_rasmussen,
)

FIX = Path(__file__).parent / "fixtures" / "rcp"
RAS = Path(__file__).parent / "fixtures" / "rasmussen" / "approval_history_trimmed.html"
SNAP_TS = datetime(2026, 10, 1, 16, 12, 21, tzinfo=UTC)


def _snap():
    # Trimmed copy of the Wayback capture 20261001161221 (id_ raw mode) of the RCP approval page.
    return parse_snapshot((FIX / "wayback_20261001161221_trimmed.html").read_text(), SNAP_TS)


def test_parse_recorded_rcp_page_and_reproduce_its_average():
    s = _snap()
    assert s is not None and len(s.in_avg) == 18 and len(s.polls) > len(s.in_avg) and len(s.chart) == 20
    assert s.avg_approve == 38.8 and s.avg_window == "9/8 - 9/29"
    # RCP's number is the plain mean of the in-average polls, rounded to one decimal.
    assert rcp_mean([p["approve"] for p in s.in_avg]) == s.avg_approve
    ras = next(p for p in s.in_avg if p["pollster"] == "Rasmussen Reports")
    assert ras["approve"] == 44 and ras["end"] == date(2026, 9, 29) and ras["pop"] == "LV" and ras["n"] == 1500
    # "updated" is US Central (-0500) -> UTC, and never after the capture.
    assert ras["updated"] == datetime(2026, 9, 30, 14, 11, 17, tzinfo=UTC)
    assert all(p["updated"] is None or p["updated"] <= SNAP_TS for p in s.polls)
    assert s.chart[0]["approve"] == 38.8


def test_asof_and_membership_use_only_polls_entered_before_t():
    s = _snap()
    v = poll_versions([s])
    assert v.height == len(s.polls)
    t = datetime(2026, 9, 29, 18, 0, tzinfo=UTC)            # before the 9/29 afternoon entries
    p = asof(v, t)
    assert p.height > 0 and (p["known_at"] <= t).all()
    assert "157946" not in p["id"].to_list()                  # Harvard-Harris entered 9/29 23:58 UTC
    fams = set(p["fam"].to_list())
    m = members_at(p, t, fams, Rule(window=21))
    assert m.height > 0 and m["fam"].n_unique() == m.height
    assert (m["end"] >= date(2026, 9, 29) - timedelta(days=21)).all()


def test_bracket_parsing_on_recorded_subtitles():
    assert bracket_tenths("39.5%") == (395, 395)
    assert bracket_tenths("39.1") == (391, 391)
    assert bracket_tenths("38.8 to 39.0") == (388, 390)
    assert bracket_tenths("Above 40.5") == (406, 1000)
    assert bracket_tenths("Above 39.8%") == (399, 1000)
    assert bracket_tenths("Below 38.8") == (0, 387)
    assert bracket_tenths("Below 39.2%") == (0, 391)
    assert rcp_mean([38, 39]) == 38.5 and rcp_mean([38.84]) == 38.8 and rcp_mean([38.85]) == 38.9


def test_kalman_recovers_house_effects_on_synthetic_polls():
    rng = np.random.default_rng(1)
    days = [date(2026, 3, 1) + timedelta(days=i) for i in range(120)]
    x = 40 + np.cumsum(rng.normal(0, 0.15, len(days)))
    house = {"A": 2.0, "B": -2.0, "C": 0.0}
    rows = []
    for i, d in enumerate(days):
        for f, h in house.items():
            if (i + "ABC".index(f)) % 3 == 0:
                rows.append({"fam": f, "start": d, "end": d, "approve": x[i] + h + rng.normal(0, 1.0), "n": 100000})
    fit = kalman_house(pl.DataFrame(rows), days[-1], q=0.15 ** 2, tau=1.0, days=130)
    assert abs((fit.house["A"] - fit.house["B"]) - 4.0) < 0.6
    assert abs(fit.x - x[-1]) < 1.0


def test_ladder_kelly_buys_only_the_mispriced_bracket():
    grid = np.arange(380, 391)
    pmf = np.zeros(len(grid))
    pmf[grid == 385] = 0.6
    pmf[grid == 386] = 0.4
    legs = [{"lo": 0, "hi": 384, "bid": 0.01, "ask": 0.02},
            {"lo": 385, "hi": 385, "bid": 0.30, "ask": 0.32},        # model 0.60: buy YES
            {"lo": 386, "hi": 386, "bid": 0.38, "ask": 0.42},        # fair
            {"lo": 387, "hi": 1000, "bid": 0.01, "ask": 0.03}]
    picks = ladder_kelly(grid, pmf, legs, theta=0.02)
    assert picks and all(p["stake"] > 0 for p in picks)
    assert any(p["lo"] == 385 and p["side"] == "yes" for p in picks)
    assert not any(p["lo"] == 386 for p in picks)
    assert ladder_kelly(grid, pmf, legs[2:3], theta=0.02) == []


def _empty_settled() -> pl.DataFrame:
    return pl.DataFrame(schema={"event": pl.String, "T": pl.Datetime("us", "UTC"),
                                "settled_at": pl.Datetime("us", "UTC"), "value": pl.Int64})


def test_rasmussen_history_restores_pruned_tracker_rows():
    ras = parse_rasmussen(RAS.read_text())
    assert ras.height == 12 and ras["date"].is_sorted()
    assert ras.filter(pl.col("date") == date(2026, 11, 5))["approve"][0] == 43
    v = with_rasmussen(poll_versions([_snap()]), ras)
    restored = v.filter(pl.col("id").str.starts_with("ras-"))
    assert restored.height == 12
    # a release dated D is only "known" on D (15:00 UTC), with field dates ending D-1
    r = restored.filter(pl.col("id") == "ras-20261105").row(0, named=True)
    assert r["known_at"] == datetime(2026, 11, 5, 15, tzinfo=UTC) and r["end"] == date(2026, 11, 4)


def test_mechanical_forecast_uses_only_captures_public_at_t():
    s = _snap()
    ctx = build_context([s], parse_rasmussen(RAS.read_text()).head(0), _empty_settled())
    with pytest.raises(IndexError):                          # nothing was public before the capture
        track(ctx, SNAP_TS - timedelta(hours=1), SNAP_TS - timedelta(hours=1))
    mem = track(ctx, SNAP_TS, SNAP_TS)
    assert rcp_mean([m["approve"] for m in mem.values()]) == 38.8
    rng = np.random.default_rng(3)
    t = SNAP_TS + timedelta(hours=1)
    pmf, info = forecast_mechanical(ctx, t, t + timedelta(hours=2), Params(sims=800, prune_rate=0.0, sigma_res=0.0), rng)
    assert abs(pmf.sum() - 1) < 1e-9 and info["anchor_value"] == 388
    assert GRID[np.argmax(pmf)] in (387, 388, 389)
    rw = forecast_rw(ctx, t, t + timedelta(days=3), 0.2, rng, 800)
    assert abs(rw.sum() - 1) < 1e-9 and rw[GRID == 388][0] > 0.05


def test_arrival_probability_follows_weekly_cadence():
    # A pollster that enters every Tuesday 14:00 UTC: a Tue 12:00 -> Tue 18:00 window is ~certain,
    # a Thursday window ~impossible.
    tues = [np.datetime64(datetime(2026, 9, 1, 14) + timedelta(days=7 * k)) for k in range(8)]
    rel = np.array(tues, dtype="datetime64[us]")
    t = datetime(2026, 10, 27, 12, tzinfo=UTC)
    assert arrival_prob(rel, t, t + timedelta(hours=6)) > 0.9
    assert arrival_prob(rel, t + timedelta(days=2), t + timedelta(days=2, hours=6)) < 0.05
