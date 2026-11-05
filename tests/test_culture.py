"""Culture/tech fundamental study on real recorded payloads: the Vercel AI Gateway labs page
(trimmed to the requests metric, 2026-06-01..2026-10-28), a Wayback capture of the Rotten Tomatoes
page of Verity (2026-10-03 21:40 UTC, trimmed around the critics scorecard) and a recorded Wayback
CDX listing. Synthetic numbers only for pure math (beta-binomial fit, rounding)."""

import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from albert.research.culture import (Fetcher, daily_panel, fit_freshrate, point_forecast, predictive,
                                     prob_above, rt_prob_above, rt_round, rt_scorecard, vercel_rawdata,
                                     wayback_cdx)

FIX = Path(__file__).parent / "fixtures"
LABS = (FIX / "vercel" / "labs_page_trimmed.html").read_text()
RT = (FIX / "rottentomatoes" / "wayback_verity_20261003214032_trimmed.html").read_text()


@pytest.fixture(scope="module")
def hist():
    return vercel_rawdata(LABS)


def test_vercel_rawdata_parses_daily_shares(hist):
    assert hist.height > 1000
    assert set(hist["metric"].unique()) == {"requests"}
    assert hist["day"].min() == date(2026, 6, 1) and hist["day"].max() == date(2026, 10, 28)
    sums = hist.group_by("day").agg(pl.col("v").sum())["v"]
    assert sums.min() > 99.9 and sums.max() < 100.1  # every lab is listed: shares sum to 100
    anth = hist.filter((pl.col("name") == "anthropic") & (pl.col("day") == date(2026, 10, 12)))
    assert 7.2 < anth["v"][0] <= 7.4  # KXANTHVREQ-12OCT26 settled in (7.2, 7.3]; page now 7.33


def test_forecast_uses_only_data_up_to_latest_known_day(hist):
    days, x = daily_panel(hist)[("anthropic", "requests")]
    i = len(x) - 3
    for h in (1, 2):
        for variant in ("last", "dow", "mean7_dow"):
            full = predictive(x, i, h, variant)
            cut = predictive(x[: i - h + 1], i, h, variant)
            assert full is not None and cut == full
            y = x.copy()
            y[i - h + 1:] += 5.0  # future values must not matter
            assert predictive(y, i, h, variant) == full
    assert point_forecast(x, i, 1, "last") == x[i - 1]


def test_prob_above_is_a_survival_curve(hist):
    days, x = daily_panel(hist)[("openai", "requests")]
    mu, sc = predictive(x, len(x) - 1, 1, "dow")
    ps = [prob_above(mu, sc, k) for k in (5, 15, 20, 25, 40)]
    assert all(a >= b for a, b in zip(ps, ps[1:], strict=False))
    assert ps[0] > 0.9 and ps[-1] < 0.1


def test_rt_scorecard_from_wayback_capture():
    card = rt_scorecard(RT)
    assert card == {"liked": 43, "not_liked": 85, "total": 128, "score": 34, "review_count": 128}
    assert rt_round(card["liked"], card["total"]) == card["score"]  # 33.6 -> 34: rounds, not floors


def test_rt_round_half_up():
    assert rt_round(1, 8) == 13  # 12.5
    assert rt_round(2, 3) == 67
    assert list(rt_round(np.array([0, 5]), np.array([10, 10]))) == [0, 50]


def test_rt_prob_above_monotone_and_sharp_when_settled():
    growth = {"a": 0.0, "b": 0.0, "c": 0.0, "sd": 1e-6}  # dN = 0: nothing left to come
    fresh = {"delta": 0.0, "kappa": 20.0}
    assert rt_prob_above(43, 128, 1.0, 33, growth, fresh) == 1.0
    assert rt_prob_above(43, 128, 1.0, 34, growth, fresh) == 0.0
    growth = {"a": math.log(41), "b": 0.0, "c": 0.0, "sd": 0.3}  # ~40 more reviews
    ps = [rt_prob_above(43, 128, 48.0, k, growth, fresh) for k in (20, 30, 34, 40, 60)]
    assert all(a >= b for a, b in zip(ps, ps[1:], strict=False)) and ps[0] > 0.95 and ps[-1] < 0.05


def test_fit_freshrate_recovers_beta_binomial():
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(400):
        total = int(rng.integers(20, 150))
        r = rng.uniform(0.2, 0.9)
        liked = int(round(r * total))
        dn = int(rng.integers(5, 60))
        m = 1 / (1 + math.exp(-(math.log((liked + 0.5) / (total - liked + 0.5)) - 0.3)))
        q = rng.beta(m * 15, (1 - m) * 15)
        rows.append({"liked": liked, "total": total, "dN": dn, "dL": int(rng.binomial(dn, q))})
    fit = fit_freshrate(pl.DataFrame(rows))
    assert abs(fit["delta"] + 0.3) < 0.12 and 8 < fit["kappa"] < 30


def test_wayback_cdx_parses_recorded_listing(tmp_path):
    f = Fetcher(tmp_path, min_interval=0.0)
    url = ("https://web.archive.org/cdx/search/cdx?url=rottentomatoes.com/m/verity&from=20261001"
           "&to=20261006&output=json&fl=timestamp,original,length&filter=statuscode:200")
    f.path(url).write_bytes((FIX / "rottentomatoes" / "cdx_verity.json").read_bytes())
    rows = wayback_cdx(f, "rottentomatoes.com/m/verity", "20261001", "20261006")
    assert len(rows) >= 5 and all(r["timestamp"].startswith("202610") for r in rows)


def test_ladder_trades_mean_variance():
    """Two strikes on one underlying; scenarios say X > 10 always, X > 20 half the time."""
    from albert.research.culture import ladder_trades

    g = pl.DataFrame({"ticker": ["A-10", "A-20"], "event": ["A", "A"], "t": [0, 0], "K": [10.0, 20.0],
                      "bid": [0.80, 0.48], "ask": [0.85, 0.52], "y": [1, 0]})
    scen = np.array([[1.0, 1.0], [1.0, 0.0]] * 50)
    tr = ladder_trades(g, scen, lam=0.0, theta=0.0)
    assert [(r["ticker"], r["side"]) for r in tr] == [("A-10", "yes")]  # 1 - 0.85 - fee > 0; A-20 has no edge
    assert abs(tr[0]["pnl"] - (1 - 0.85 - 0.0090)) < 1e-9
    assert ladder_trades(g, scen, lam=0.0, theta=0.2) == []
