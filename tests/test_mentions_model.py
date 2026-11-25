"""Tests for ``albert.research.mentions_model`` (parsers on trimmed recorded Kalshi responses)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import mentions_model as MM

FIX = Path(__file__).parent / "fixtures" / "mentions_model"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text())["response"]


def test_milestone_frame_reads_recorded_page() -> None:
    ms = MM.milestone_frame([_load("milestones_mentions_20261123_trim.json")])
    assert ms.height >= 2
    v = ms.filter(pl.col("event") == "KXVANCEMENTION-26NOV18")
    assert v.height == 1
    assert v["start"][0] == datetime(2026, 11, 18, 13, tzinfo=UTC)
    assert v["ms_category"][0] == "Mentions"


def test_market_frame_and_legs_drop_non_qualifying_leg() -> None:
    from albert.research import mentions as M

    mk = MM.market_frame([_load("markets_vance_26nov18_trim.json")])
    assert mk.height == 3
    assert set(mk["category"].to_list()) == {"Mentions"}
    assert set(mk["result"].to_list()) <= {"yes", "no"}
    lg = M.legs(mk)
    assert lg.height == 2                       # "Event does not qualify" dropped
    assert not lg["word"].str.starts_with("event does not qualify").any()


def test_candles_batch_parses() -> None:
    from albert.research import xcat

    got = xcat.batch_rows(_load("candles_batch_vance_26nov18_trim.json"), 60)
    assert len(got) == 2
    rows = [r for v in got.values() for r in v]
    assert rows and all(r["end_ts"] is not None for r in rows)
    assert any(r["p_high"] is not None for r in rows)


def _u(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"speaker": pl.String, "etype": pl.String, "word": pl.String, "k": pl.Int64,
                                      "settlement_ts": pl.Datetime("us", "UTC"), "start": pl.Datetime("us", "UTC"),
                                      "y": pl.Int64}, orient="row")


def test_history_features_use_only_legs_settled_before_t() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    U = _u([("trump", "rally", "tariff", 1, t0, t0 - timedelta(hours=2), 1),
            ("trump", "rally", "tariff", 1, t0 + timedelta(days=1), t0 + timedelta(hours=22), 0),
            ("trump", "rally", "tariff", 1, t0 + timedelta(days=3), t0 + timedelta(days=2), 1),     # after t
            ("vance", "rally", "tariff", 1, t0, t0 - timedelta(hours=2), 0)])
    D = pl.DataFrame({"speaker": ["trump"], "etype": ["rally"], "word": ["tariff"], "k": [1],
                      "t": [t0 + timedelta(days=1)]}, schema_overrides={"t": pl.Datetime("us", "UTC")})
    F = MM.history_features(D, U)
    assert F["n_sw"][0] == 1 and F["h_sw"][0] == 1        # the leg settling exactly at t is excluded
    assert F["n_w"][0] == 2 and F["h_w"][0] == 1
    assert F["last_y"][0] == 1
    assert 0 < F["r_sw"][0] < 1


def test_fit_logit_recovers_direction() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(0.5 + 2 * X[:, 0])))).astype(float)
    m = MM.fit_logit(X, y, l2=0.1)
    p = MM.predict(m, X)
    assert np.corrcoef(p, X[:, 0])[0, 1] > 0.9
    assert abs(p.mean() - y.mean()) < 0.02


def _q() -> pl.DataFrame:
    return pl.DataFrame({"fair": [0.30, 0.30, 0.60], "bid": [0.28, 0.28, 0.58], "ask": [0.32, 0.32, 0.62],
                         "p_model": [0.20, 0.45, 0.75]})


def test_price_cols_policies() -> None:
    live = MM.price_cols(_q(), {"kind": "live"})
    assert live["price"].to_list() == [0.58, 0.58, 0.28]        # floor(1 - fair - 12c)
    skip = MM.price_cols(_q(), {"kind": "skip", "theta": 0.0})
    # sell YES at 1 - price = 0.42 / 0.42 / 0.72; skip where p_model >= that
    assert np.isnan(skip["price"][1]) and np.isnan(skip["price"][2]) and skip["price"][0] == 0.58
    mf = MM.price_cols(_q(), {"kind": "mfair", "m": 0.08})
    assert mf["price"].to_list() == [0.71, 0.47, 0.17]
    yes = MM.price_cols(_q(), {"kind": "yes", "g": 0.10, "m": 0.04})
    assert np.isnan(yes["yes_price"][0])
    assert yes["yes_price"][1] == 0.31                          # min(floor(0.45 - 0.04), ask - 1c)
    assert yes["yes_price"][2] == 0.61


def test_yes_fills_need_a_print_strictly_below() -> None:
    t = datetime(2026, 1, 1, 10, tzinfo=UTC)
    O = pl.DataFrame({"row": [0, 1], "ticker": ["A", "B"], "t": [t, t], "end_o": [t + timedelta(hours=3)] * 2,
                      "price": [0.40, 0.40]}).with_columns(pl.col("row").cast(pl.UInt32))
    H = pl.DataFrame({"ticker": ["A", "A", "B"], "end_ts": [t + timedelta(hours=1), t + timedelta(hours=2), t + timedelta(hours=1)],
                      "lo": [0.42, 0.39, 0.40]})
    f = MM.yes_fills(O, H).sort("row")
    assert f["filled"].to_list() == [True, False]                 # B printed AT our price: no fill


def test_split_of() -> None:
    d = pl.DataFrame({"start": [datetime(2026, 9, 30, tzinfo=UTC), datetime(2026, 10, 2, tzinfo=UTC),
                                datetime(2026, 11, 16, 20, tzinfo=UTC), datetime(2026, 11, 18, tzinfo=UTC)]})
    assert d.select(MM.split_of(pl.col("start")).alias("s"))["s"].to_list() == ["dev", "seen", "gap", "confirm"]
