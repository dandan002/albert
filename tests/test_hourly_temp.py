"""hourly_temp: Kalshi hourly temperature ladders settled on the Kalshi Weather Index.

Parsers run on trimmed copies of real responses: the minute index (GET /live_data/weather/miami),
settled markets with ``expiration_value``, 1-minute batch candles, and the IEM ASOS archives.
Model and ladder maths use small synthetic inputs.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import hourly_temp as h
from albert.venues import kalshi as k

FIX = Path(__file__).parent / "fixtures"


def _index() -> pl.DataFrame:
    payload = json.loads((FIX / "kalshi" / "hourly_temp_index_miami.json").read_text())
    return pl.DataFrame(h.index_rows(payload, "MIA"), schema=h.INDEX_SCHEMA)


def _markets() -> pl.DataFrame:
    payload = json.loads((FIX / "kalshi" / "hourly_temp_markets.json").read_text())
    rows = [{"ticker": m["ticker"], "event": m["event_ticker"], "series": "KXTEMPMIAH",
             "exp_value": float(m["expiration_value"]), "result": m["result"],
             "floor_strike": m["floor_strike"], "close_time": k.parse_ts(m["close_time"])}
            for m in payload["markets"]]
    return pl.DataFrame(rows)


def test_index_parses_and_settles_the_ladder():
    idx = _index()
    assert idx.height > 60 and (idx["status"] == "normal").all()
    S = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    at_s = idx.filter(pl.col("t") == S)["v"].item()
    ev = h.events(_markets())
    assert ev.height == 10
    # Kalshi's expiration value is the published index point at the settlement minute ...
    assert (ev["exp_value"] == at_s).all() and (ev["S"] == S).all()
    # ... and every leg resolved YES iff that value is above the strike (k - 0.01).
    assert ev.select(((pl.col("exp_value") > pl.col("k") - 0.01).cast(pl.Int8) == pl.col("y")).all()).item()
    # The Miami index (5 stations, whole deg C) lives on the 0.36 F lattice.
    frac = (idx["v"].to_numpy() - 32) / (1.8 / 5)
    assert np.allclose(frac, np.round(frac), atol=0.03)


def test_strike_and_lattice_threshold():
    assert h.strike("KXTEMPMIAH-26SEP2308-T76.99") == 77.0
    # lattice 32 + 0.36 j: 76.64, 77.00 -> YES above 76.99 <=> index >= 77.00; midpoint 76.82
    assert abs(h.threshold(77.0, 5) - 76.82) < 1e-9
    # NYC has 8 members: step 0.225 F; 75.875 and 76.100 straddle 75.99 -> midpoint 75.9875
    assert abs(h.threshold(76.0, 8) - 75.9875) < 1e-9


def test_features_use_only_published_minutes():
    idx = _index()
    grid = h.minute_grid(idx)
    lo, v = grid["MIA"]
    S = np.array([int(datetime(2026, 9, 23, 12, 0, tzinfo=UTC).timestamp() // 60)])
    f = h.features(grid, "MIA", S, 10, None)
    # decision at 11:50; station readings arrive <= 4 min after the minute -> newest usable 11:46
    assert h.DELAY_MIN == 4
    assert f["u0"][0] == S[0] - 14 and f["I0"][0] == v[S[0] - 14 - lo]
    assert f["h"][0] == 14
    later = h.features(grid, "MIA", S, 10, None, delay=0)
    assert later["u0"][0] == S[0] - 10      # the minute at t itself only with zero delay


def test_minute_candles_and_quote_at_t():
    payload = json.loads((FIX / "kalshi" / "hourly_temp_candles_1m.json").read_text())
    rows = []
    for m in payload["markets"]:
        rows.extend(k.candle_rows(m, m["market_ticker"], 1))
    c = pl.DataFrame(rows, schema=k.CANDLE_SCHEMA)
    assert c.height > 50 and c["ticker"].n_unique() == 3
    t = datetime(2026, 9, 23, 11, 30, tzinfo=UTC)
    pred = pl.DataFrame({"ticker": c["ticker"].unique().sort().to_list(), "t": [t] * 3})
    q = h.quotes_at(pred, c)
    assert q.height >= 1 and (q["q_ts"] <= t).all() and (q["ask"] > q["bid"]).all()


def test_iem_parsers_on_recorded_csv():
    hf = h.obs_rows((FIX / "iem" / "hourly_temp_asos_hf.csv").read_text(), "MIA")
    assert len(hf) >= 5 and any(r["hf"] for r in hf) and any(r["precise"] for r in hf)
    assert all(abs(r["tf"] - (r["tc"] * 9 / 5 + 32)) < 1e-9 for r in hf)
    one = h.one_min_rows((FIX / "iem" / "hourly_temp_asos1min.csv").read_text(), "MIA")
    assert len(one) >= 5 and all(60 < r["tf"] < 100 for r in one)
    assert h.metar_temp_c("KMIA 231153Z 17004KT 25/23 A2995 RMK AO2 T02500233 $") == (25.0, True)
    assert h.metar_temp_c("KORD 051551Z 27012KT M02/M05 A3001") == (-2, False)


def test_fit_and_probability_are_calibrated_on_synthetic_ou():
    rng = np.random.default_rng(0)
    n = 4000
    f = {"I0": rng.normal(70, 5, n), "clim": rng.normal(0, 1, n), "d15": rng.normal(0, 0.5, n),
         "d30": rng.normal(0, 0.7, n), "d60": rng.normal(0, 1, n), "nbm": rng.normal(0, 1, n),
         "anom": rng.normal(0, 2, n), "vol": np.abs(rng.normal(0.3, 0.1, n)), "h": np.full(n, 46.0)}
    dy = 0.8 * f["clim"] + 0.3 * f["anom"] + rng.normal(0, 0.6, n)
    fit = h.fit_variant(f, dy, "ou")
    assert abs(fit["beta"][1] - 0.8) < 0.05 and abs(fit["beta"][6] - 0.3) < 0.03
    mu, sg = h.predict_variant(fit, f)
    p = h.prob_above(mu, sg, f["I0"] + 0.5, fit["z"])
    y = (f["I0"] + dy > f["I0"] + 0.5).astype(float)
    assert abs(p.mean() - y.mean()) < 0.02
    assert np.all(np.diff(h.prob_above(70.0, 1.0, np.array([68.0, 69.0, 70.0, 71.0]))) < 0)


def test_ladder_choice():
    rows = [{"ticker": f"T{x}", "event": "E", "lead": 25, "k": float(x), "p": p, "bid": b, "ask": a, "y": y,
             "t": None}
            for x, p, b, a, y in [(70, 0.95, 0.80, 0.84, 1), (71, 0.70, 0.55, 0.60, 1), (72, 0.20, 0.30, 0.35, 0)]]
    every = h.ladder_trades(rows, 0.0, None)
    sides = {(c["ticker"], c["side"]) for c in every}
    assert sides == {("T70", "yes"), ("T71", "yes"), ("T72", "no")}
    best = h.ladder_trades(rows, 0.0, -1.0)
    assert len(best) == 1 and best[0]["edge"] == max(c["edge"] for c in every)
    mv = h.ladder_trades(rows, 0.0, 0.0)          # lam 0 = max expected PnL = every positive edge
    assert {(c["ticker"], c["side"]) for c in mv} == sides
    assert all(abs(c["pnl"] - (c["y"] if c["side"] == "yes" else 1 - c["y"]) + c["cost"]) < 0.02 for c in every)
