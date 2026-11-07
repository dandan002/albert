from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from albert.research import rain

# Real Kalshi rules_primary strings (KXRAIN-26AUG16-NYC, KXRAIN-26SEP16-MIL, KXRAIN-26OCT30-WPB).
RULES = [
    ("If the total precipitation at CLINYC in New York City in Aug 16, 2026 is strictly greater than "
     "0 inches, then the market resolves to Yes.", "NYC"),
    ("If the total precipitation at KMKE in Milwaukee in Sep 16, 2026 is strictly greater than 0 "
     "inches, then the market resolves to Yes.", "MKE"),
    ("If the total precipitation at CLIDJT in West Palm Beach in Oct 30, 2026 is strictly greater "
     "than 0 inches, then the market resolves to Yes.", "DJT"),
]


def test_rule_code_and_event_day():
    for rules, code in RULES:
        assert rain.rule_code(rules) == code
    assert rain.event_day("KXRAIN-26AUG16") == date(2026, 8, 16)
    assert rain.event_day("KXRAINWKND-26OCT03") == date(2026, 10, 3)
    assert rain.event_day("nonsense") is None


def test_lst_bounds_use_standard_time():
    assert rain.lst_bounds("NYC", date(2026, 9, 2))[0] == datetime(2026, 9, 2, 5, tzinfo=UTC)
    assert rain.lst_bounds("SEA", date(2026, 9, 2))[0] == datetime(2026, 9, 2, 8, tzinfo=UTC)
    assert rain.lst_bounds("PHX", date(2026, 12, 2))[0] == datetime(2026, 12, 2, 7, tzinfo=UTC)
    assert rain.slot_time(date(2026, 9, 2), "d-1 12z") == datetime(2026, 9, 1, 12, tzinfo=UTC)


def test_cli_precip_rows_parse_trace(fixture):
    rows = rain.cli_precip_rows(fixture("iem/cli_precip.json"), "NYC")
    assert len(rows) == 7
    by = {r["day"]: r["precip"] for r in rows}
    assert by[date(2026, 8, 21)] == rain.TRACE          # "T"
    assert by[date(2026, 8, 19)] == 0.0
    assert by[date(2026, 8, 20)] == pytest.approx(2.87)
    lab = pl.DataFrame(rows).select(rain.label(pl.col("precip"), True).alias("t"),
                                    rain.label(pl.col("precip"), False).alias("m"))
    assert lab["t"].sum() == 5 and lab["m"].sum() == 3


def test_asos_precip_rows(fixture_text):
    rows = rain.asos_precip_rows(fixture_text("iem/asos_precip.csv"), "NYC")
    assert len(rows) > 50 and rows[0]["ts"].tzinfo == UTC
    assert rows[0]["p01i"] == rain.TRACE and rows[0]["lat"] == pytest.approx(40.779)
    assert any(r["p01i"] and r["p01i"] >= 0.1 for r in rows)


def test_mos_rows_nbs(fixture_text):
    rows = rain.mos_rows(fixture_text("iem/nbs_pop.csv"), "NYC")
    assert rows and all(r["ftime"] > r["runtime"] for r in rows)
    pops = [r for r in rows if r["p06"] is not None]
    assert len(pops) >= 8 and max(r["p06"] for r in pops) == 39.0


def _win(code: str, day: date, slot: str) -> pl.DataFrame:
    lo, hi = rain.lst_bounds(code, day)
    return pl.DataFrame({"code": [code], "day": [day], "slot": [slot], "t": [rain.slot_time(day, slot)],
                         "lo": [lo], "hi": [hi]})


def test_recent_rain_uses_reports_valid_by_t_minus_lag(fixture_text):
    obs = pl.DataFrame(rain.asos_precip_rows(fixture_text("iem/asos_precip.csv"), "NYC"))
    win = pl.concat([_win("NYC", date(2026, 9, 1), "d 12z"), _win("NYC", date(2026, 9, 2), "d 00z")])
    got = rain.recent_rain(win, obs).sort("slot")
    for r in got.iter_rows(named=True):
        u = rain.slot_time(r["day"], r["slot"]) - rain.OBS_LAG
        last = obs.filter(pl.col("ts") <= u)["ts"].max()
        span = obs.filter((pl.col("ts") > last - rain.RECENT) & (pl.col("ts") <= last))
        assert r["pre_max"] == pytest.approx(span["p01i"].max()) and r["pre_n"] == span.height
    by = dict(zip(got["slot"], got["pre_max"], strict=True))
    assert by["d 12z"] == pytest.approx(0.11)          # the 05:51Z hour of 2026-09-01, before the 12z decision
    # A wet report after t - OBS_LAG is not seen; a stale station (nothing within 2 h) is missing.
    late = pl.DataFrame({"code": ["NYC"], "ts": [datetime(2026, 9, 1, 11, 55, tzinfo=UTC)], "p01i": [0.5],
                         "lat": [None], "lon": [None]}, schema=obs.schema)
    assert rain.recent_rain(win, pl.concat([obs, late])).sort("slot")["pre_max"].to_list() == got["pre_max"].to_list()
    old = obs.filter(pl.col("ts") < datetime(2026, 9, 1, 9, tzinfo=UTC))
    stale = rain.recent_rain(win.filter(pl.col("slot") == "d 12z"), old)
    assert stale["pre_max"][0] is None and stale["pre_n"][0] == 0


def test_build_features_and_design_are_iem_only(fixture, fixture_text):
    obs = pl.DataFrame(rain.asos_precip_rows(fixture_text("iem/asos_precip.csv"), "NYC"))
    mos = pl.DataFrame(rain.mos_rows(fixture_text("iem/nbs_pop.csv"), "NYC"))
    cli = pl.DataFrame(rain.cli_precip_rows(fixture("iem/cli_precip.json"), "NYC"), infer_schema_length=None)
    clim = rain.climatology(cli.with_columns(pl.col("day").dt.offset_by("-1y")), False, years=(2025,))
    win = _win("NYC", date(2026, 9, 1), "d 12z")
    f = rain.build_features(win, obs, mos, clim)
    assert f.height == 1 and not any("om" in c for c in f.columns)
    X, names = rain.design(f, rain.VARIANTS[rain.IEM_VARIANT], [])
    assert X.shape == (1, len(names)) and np.isfinite(X).all()
    assert {"log_pre_max", "pre_wet", "pre_missing", "trace_today", "nbs_missing", "gfs_missing"} <= set(names)
    row = dict(zip(names, X[0], strict=True))
    assert row["pre_wet"] == 1.0 and row["pre_missing"] == 0.0
    assert row["nbs_missing"] == 1.0                    # the 2025 NBS run is not for this day: missing, filled
    assert set(rain.VARIANTS) == {"climo", "nbm", "mos", "mos_obs", "full"}


def test_mos_features_respect_runtime_lag(fixture_text):
    mos = pl.DataFrame(rain.mos_rows(fixture_text("iem/nbs_pop.csv"), "NYC"))  # run 2025-09-02 13Z
    win = _win("NYC", date(2025, 9, 3), "d 00z")                               # t = 09-03 00Z
    f = rain.mos_features(win, mos, "NBS")
    assert f.height == 1 and f["nbs_pany"][0] == 0.0 and f["nbs_cov"][0] == 1.0
    # A run whose runtime + lag is after t must not be used.
    late = mos.with_columns((pl.col("runtime") + timedelta(hours=10)).alias("runtime"),
                            pl.lit(100.0).alias("p06"))
    f2 = rain.mos_features(win, pl.concat([mos, late]), "NBS")
    assert f2["nbs_pany"][0] == 0.0
    # ...but it is used once public (t = 09-04 00Z).
    f4 = rain.mos_features(_win("NYC", date(2025, 9, 4), "d 00z"), pl.concat([mos, late]), "NBS")
    assert f4["nbs_pany"][0] > 0.99
    # 2025-09-04 LST day (05Z-05Z): the 4% block ending 09-05 00Z and 5/6 of the 39% block.
    f3 = rain.mos_features(_win("NYC", date(2025, 9, 4), "d 00z"), mos, "NBS")
    assert f3["nbs_pany"][0] == pytest.approx(1 - 0.96 * (1 - 0.39 * 5 / 6), abs=1e-6)
    assert f3["nbs_pmax"][0] == pytest.approx(0.39)


def test_rain_so_far_ignores_reports_after_t_and_first_hour(fixture_text):
    obs = pl.DataFrame(rain.asos_precip_rows(fixture_text("iem/asos_precip.csv"), "NYC"))
    # 2026-09-01: rain 04:51-05:51Z; the 05:xx reports partly sum the previous LST day.
    win = pl.concat([_win("NYC", date(2026, 9, 1), "d 12z"), _win("NYC", date(2026, 9, 1), "d 00z")])
    f = rain.rain_so_far(win, obs).sort("slot")
    seen = dict(zip(f["slot"], f["seen"], strict=True))
    assert seen["d 00z"] == 0.0                     # before the day: nothing seen
    assert seen["d 12z"] >= 0.01                    # 06:51-08:51Z reports are inside the day
    only_late = obs.filter(pl.col("ts") > datetime(2026, 9, 1, 11, 55, tzinfo=UTC))
    assert rain.rain_so_far(win, only_late).filter(pl.col("slot") == "d 12z")["seen"][0] == 0.0


def test_logit_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 2))
    p = 1 / (1 + np.exp(-(0.5 + 1.5 * X[:, 0] - 1.0 * X[:, 1])))
    y = (rng.random(4000) < p).astype(float)
    m = rain.Logit(lam=0.1).fit(X, y)
    b = m.beta[1:] / m.sd
    assert b[0] == pytest.approx(1.5, abs=0.2) and b[1] == pytest.approx(-1.0, abs=0.2)
    assert rain.brier_logloss(m.predict(X), y)[0] < rain.brier_logloss(np.full(4000, y.mean()), y)[0]


def test_maker_sim_trade_through_rules():
    t = datetime(2026, 10, 5, 12, tzinfo=UTC)
    rows = pl.DataFrame({"ticker": ["A", "B"], "event": ["E", "E"], "t": [t, t], "p": [0.40, 0.40],
                         "bid": [0.30, 0.30], "ask": [0.50, 0.50], "y": [1, 0], "slot": ["d 12z"] * 2,
                         "close_time": [t + timedelta(hours=12)] * 2})
    ext = pl.DataFrame({"ticker": ["A", "A", "B"],
                        "end_ts": [t + timedelta(hours=1), t + timedelta(hours=8), t + timedelta(hours=1)],
                        "p_low": [0.38, 0.01, 0.42], "p_high": [0.39, 0.99, 0.45]})
    f = rain.maker_sim(rows, ext, 0.02)
    # A: YES bid 0.38 improves the 0.30 bid and a trade printed at 0.38 -> filled, wins 0.62.
    # The 8h candle is beyond the 6h horizon. B: ask 0.42 < 0.50 ask, high 0.45 -> NO filled.
    got = {(r["ticker"], r["side"]): r["pnl"] for r in f.iter_rows(named=True)}
    assert got == {("A", "yes"): pytest.approx(0.62), ("B", "no"): pytest.approx(0.42)}


def test_expanding_fit_ignores_labels_not_public_at_the_first_decision():
    rng = np.random.default_rng(1)
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(243)]      # Jan-Aug
    n = len(days)
    clim = rng.uniform(0.1, 0.6, n)
    feats = pl.DataFrame({"code": ["NYC"] * n, "day": days, "slot": ["d 00z"] * n,
                          "t": [rain.slot_time(d, "d 00z") for d in days], "clim": clim,
                          "left": np.ones(n), "decided": [False] * n,
                          "wet": (rng.random(n) < clim).astype(np.int8)})
    month = [date(2026, 8, 1)]
    base = rain.expanding_predictions(feats, "climo", month)
    assert base.height == 31 and base["p"].is_between(0, 1).all()
    # Labels of Jul 30-31 are not public at Aug 1's first decision (d-2 12z = Jul 30 12Z).
    late = feats.with_columns(pl.when(pl.col("day") >= date(2026, 7, 30)).then(1 - pl.col("wet"))
                              .otherwise(pl.col("wet")).alias("wet"))
    assert rain.expanding_predictions(late, "climo", month)["p"].to_list() == pytest.approx(base["p"].to_list())
    early = feats.with_columns(pl.when(pl.col("day") == date(2026, 7, 29)).then(1 - pl.col("wet"))
                               .otherwise(pl.col("wet")).alias("wet"))
    assert rain.expanding_predictions(early, "climo", month)["p"].to_list() != pytest.approx(base["p"].to_list())
