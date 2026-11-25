import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import xs_ml as xm
from albert.research.xcat import OHLC_SCHEMA, ohlc_rows

FIX = Path(__file__).parent / "fixtures" / "xs_ml"
TICKER = "KXTRUMPACT-26MAY24-T4"


def _candles() -> pl.DataFrame:
    rec = json.loads((FIX / "candles_hourly_KXTRUMPACT-26MAY24-T4.json").read_text())
    return pl.DataFrame(ohlc_rows(rec["response"], TICKER, 60), schema=OHLC_SCHEMA).select(xm.OHLC_COLS)


def _sample() -> pl.DataFrame:
    # the market's real listing fields (recorded /historical/markets row); T = expected expiration
    t_close = datetime(2026, 5, 31, 14, tzinfo=UTC)
    return pl.DataFrame({
        "ticker": [TICKER], "event_ticker": ["KXTRUMPACT-26MAY24"], "series_ticker": ["KXTRUMPACT"],
        "open_time": [datetime(2026, 5, 24, 14, tzinfo=UTC)], "close_time": [t_close], "T": [t_close]})


def test_compact_row_parses_recorded_page():
    rec = json.loads((FIX / "markets_settled_page.json").read_text())
    rows = [xm.compact_row(m) for m in rec["response"]["markets"]]
    df = xm._rows_frame(rows)
    assert df.height == 3
    assert df["series_ticker"].to_list() == [e.split("-")[0] for e in df["event_ticker"].to_list()]
    assert set(df["result"].to_list()) <= {"yes", "no"}
    assert df["close_time"].dtype == pl.Datetime("us", "UTC")
    assert (df["expected_expiration_time"] >= df["open_time"]).all()
    assert df["market_type"].to_list() == ["binary"] * 3


def test_panel_quotes_are_causal_and_fills_strictly_through():
    c, s = _candles(), _sample()
    assert c.height == 72
    d = xm.panel_chunk(s, c)
    assert sorted(d["lead"].to_list()) == [6, 24, 72]
    cs = c.sort("end_ts")
    for r in d.iter_rows(named=True):
        before = cs.filter(pl.col("end_ts") <= r["t"])
        last = before.row(-1, named=True)
        assert abs(r["bid"] - (last["bid_close"] or 0.0)) < 1e-12 and abs(r["ask"] - (last["ask_close"] or 1.0)) < 1e-12
        assert r["quote_age_h"] >= 0
        vol24 = before.filter(pl.col("end_ts") > r["t"] - timedelta(hours=24))["volume"].sum()
        assert abs(r["vol_24h"] - vol24) < 1e-9
        # maker YES bid: first later candle (before T) that printed strictly below it with >= n traded
        n = max(1.0, np.floor(xm.STAKE / r["mk_yes"]))
        later = cs.filter((pl.col("end_ts") > r["t"]) & (pl.col("end_ts") <= s["T"][0]))
        hits = later.filter((pl.col("p_low") < r["mk_yes"] - 1e-9) & (pl.col("volume") >= n))
        assert r["fill_yes_ts"] == (hits["end_ts"][0] if hits.height else None)
        n_no = max(1.0, np.floor(xm.STAKE / r["mk_no"]))
        hits = later.filter((pl.col("p_high") > 1 - r["mk_no"] + 1e-9) & (pl.col("volume") >= n_no))
        assert r["fill_no_ts"] == (hits["end_ts"][0] if hits.height else None)
        # wide NO bid (sell YES at ceil(mid + 12c)): first later print strictly above that level
        lvl = np.ceil(round((r["mid"] + xm.WIDE_MARGIN) * 100, 6)) / 100
        assert abs(r["wide_lvl"] - lvl) < 1e-9
        hits = later.filter(pl.col("p_high") > lvl + 1e-9)
        assert r["fill_wide_ts"] == (hits["end_ts"][0] if hits.height else None)


def test_candles_at_or_after_close_are_dropped():
    c, s = _candles(), _sample()
    early = s.with_columns(pl.lit(datetime(2026, 5, 30, 20, tzinfo=UTC)).alias("close_time"))
    d = xm.panel_chunk(early, c)
    assert d["lead"].to_list() == [72, 24]                      # the 6 h lead is after the actual close
    assert d["fill_yes_ts"].drop_nulls().le(datetime(2026, 5, 30, 20, tzinfo=UTC)).all()


def test_logit_and_gbm_learn_on_synthetic_data():
    rng = np.random.default_rng(1)
    n = 6000
    x = rng.normal(size=(n, 3))
    z = 0.8 * x[:, 0] - 0.5 * x[:, 1] + 1.2 * (x[:, 2] > 0.5)
    y = (rng.random(n) < xm.sigmoid(z)).astype(float)
    X = np.column_stack([np.ones(n), x])
    b = xm.fit_logit(X, y, lam=1.0, pen=np.array([0, 1, 1, 1.0]))
    assert b[1] > 0.5 and b[2] < -0.3
    g = xm.GBM(rounds=150, lr=0.1, min_leaf=100, lam=5.0).fit(x, y, np.zeros(n))
    ll0 = xm.logloss(y, np.full(n, y.mean())).mean()
    ll1 = xm.logloss(y, xm.sigmoid(g.predict_logit(x, np.zeros(n)))).mean()
    assert ll1 < ll0 - 0.05
    assert g.gain.argmax() in (0, 2)


def test_ratio_t_holm_and_capped_book():
    num = np.array([1.0, -0.5, 2.0, 0.5])
    den = np.ones(4)
    r, t = xm.ratio_t(num, den, np.array(["a", "a", "b", "c"]))
    assert abs(r - 0.75) < 1e-12 and t is not None
    assert xm.holm(np.array([0.001, 0.2, 0.01])).tolist() == [True, False, True]
    t0 = datetime(2026, 6, 1, tzinfo=UTC)
    x = pl.DataFrame({"rid": ["a", "b", "c"], "event_ticker": ["E", "E", "F"], "ticker": ["A", "B", "C"],
                      "side": ["no", "no", "yes"], "n": [5.0, 5.0, 2.0], "price": [0.9, 0.9, 0.5], "fee": [0.01, 0.01, 0.02],
                      "t_fill": [t0, t0 + timedelta(hours=1), t0], "settle": [t0 + timedelta(days=1)] * 3,
                      "payoff": [1.0, 1.0, 0.0]})
    led = xm.capped_ledger(x, "v", "dev")
    assert led["ticker"].to_list() == ["A", "C"]              # B would push event E over EVENT_CAP
    assert abs(led["cost_usd"][0] - 4.51) < 1e-9
