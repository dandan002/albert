from datetime import UTC, date, datetime

from albert.venues import weather as wx


def test_cli_rows_from_recorded_iem(fixture):
    rows = wx.cli_rows(fixture("iem/cli.json"), "NYC")
    assert rows and all(r["code"] == "NYC" and isinstance(r["day"], date) for r in rows)
    assert any(isinstance(r["high"], int) for r in rows)


def test_asos_and_nbm_rows_from_recorded_csv(fixture_text):
    obs = wx.asos_rows(fixture_text("iem/asos.csv"), "NYC")
    assert obs and obs[0]["ts"].tzinfo == UTC and 0 < obs[0]["tmpf"] < 120
    nbm = wx.nbm_rows(fixture_text("iem/nbm.csv"), "NYC")
    assert nbm and all(r["ftime"] > r["runtime"] for r in nbm)
    assert any(r["txn"] is not None and r["xnd"] is not None for r in nbm)


def test_nws_observations_convert_to_fahrenheit(fixture):
    rows = wx.nws_obs_rows(fixture("nws/observations.json"), "NYC")
    assert rows and all(20 < r["tmpf"] < 120 for r in rows)


def test_cli_code_and_lst_bounds():
    assert wx.cli_code("recorded at New York City (CLINYC) for Sep 2") == "NYC"
    # CLI days run on local *standard* time, even in summer.
    assert wx.lst_day_bounds("NYC", date(2026, 9, 2))[0] == datetime(2026, 9, 2, 5, tzinfo=UTC)
    assert wx.lst_day_bounds("PHX", date(2026, 9, 2))[0] == datetime(2026, 9, 2, 7, tzinfo=UTC)


def _model(**kw):
    import numpy as np

    from albert.research.weather import HighModel

    base = dict(bias=0.0, sigma=2.0, u_values=np.array([0.0, 1.0, 2.0]),
                u_probs=np.array([0.4, 0.5, 0.1]))
    return HighModel(**{**base, **kw})


def test_brackets_parse_kalshi_subtitles():
    import math

    from albert.research.weather import bracket

    assert bracket("72° to 73°") == (72.0, 73.0)
    assert bracket("83° or above") == (83.0, math.inf)
    assert bracket("65° or below") == (-math.inf, 65.0)
    assert bracket("something else") is None


def test_bracket_probabilities_sum_to_one_and_respect_observed_max():
    import math

    m = _model()
    legs = [(-math.inf, 65), (66, 67), (68, 69), (70, 71), (72, 73), (74, math.inf)]
    pre = [m.prob(lo, hi, "NYC", None, 70.0) for lo, hi in legs]
    assert abs(sum(pre) - 1) < 1e-9 and max(pre) == pre[3]
    # Intraday: observed 72F already, rest-of-day forecast lower -> high is 72-74.
    intra = [m.prob(lo, hi, "NYC", 72.0, 68.0) for lo, hi in legs]
    assert abs(sum(intra) - 1) < 1e-9
    assert intra[0] == intra[1] == intra[2] == intra[3] == 0  # below the observed max: dead
    assert intra[4] > 0.85


def test_day_over_means_high_is_observed_max_plus_offset():
    import math

    m = _model()
    p = m.prob(72, 73, "NYC", 72.0, -math.inf)  # nothing left to forecast
    assert abs(p - 0.9) < 1e-9  # U in {0, 1} -> 72 or 73
