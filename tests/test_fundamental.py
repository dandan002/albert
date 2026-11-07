import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from albert.research.fundamental import (
    attach_quotes,
    clustered_mean,
    evaluate,
    fit_logistic,
    taker_fee,
    taker_pnl,
)
from albert.venues.kalshi import candle_rows

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _candles() -> pl.DataFrame:
    d = json.loads((FIX / "candles_historical.json").read_text())
    return pl.DataFrame(candle_rows(d, d["ticker"], 60))


def test_quote_comes_from_a_candle_that_closed_before_t():
    c = _candles()
    assert c.height > 0
    t_end = c["end_ts"].sort()
    t = t_end[1] + timedelta(minutes=30)          # between the 2nd and 3rd candle closes
    pred = pl.DataFrame({"ticker": [c["ticker"][0]], "event": ["e"], "t": [t], "p": [0.5]})
    q = attach_quotes(pred, c, max_age=timedelta(days=30))
    assert q.height == 1 and q["q_ts"][0] <= t
    # A decision before the first close gets no quote (no look-ahead).
    early = pred.with_columns(pl.lit(t_end[0] - timedelta(minutes=1)).alias("t"))
    assert attach_quotes(early, c, max_age=timedelta(days=30)).height == 0


def test_taker_pays_the_ask_and_fee():
    df = pl.DataFrame({"ticker": ["a", "b"], "event": ["e1", "e2"], "p": [0.9, 0.1], "bid": [0.6, 0.6],
                       "ask": [0.7, 0.7], "y": [1, 1]})
    tr = taker_pnl(df, 0.0).sort("ticker")
    assert tr["side"].to_list() == ["yes", "no"]
    assert abs(tr["pnl"][0] - (1 - 0.7 - taker_fee(0.7))) < 1e-12
    assert abs(tr["pnl"][1] - (0 - 0.4 - taker_fee(0.4))) < 1e-12


def test_blend_recovers_informative_model_and_split():
    rng = np.random.default_rng(0)
    n = 4000
    truth = rng.uniform(0.1, 0.9, n)
    y = (rng.uniform(size=n) < truth).astype(int)
    mid = np.clip(truth + rng.normal(0, 0.15, n), 0.03, 0.97)       # noisy market
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    df = pl.DataFrame({"ticker": [f"m{i}" for i in range(n)], "event": [f"e{i // 4}" for i in range(n)],
                       "t": [t0 + timedelta(hours=i) for i in range(n)], "p": truth, "bid": mid - 0.01,
                       "ask": mid + 0.01, "y": y})
    out = evaluate(df, holdout_from=t0 + timedelta(hours=n // 2))
    assert out["blend"]["c_model"] > 0.5 and out["blend"]["t_c"] > 3
    hold = [r for r in out["taker"] if r["split"] == "holdout" and r["trades"]]
    assert hold and max(r["pnl_c"] for r in hold) > 0
    beta, _t = fit_logistic(np.column_stack([np.ones(n), np.log(truth / (1 - truth))]), y.astype(float), np.arange(n))
    assert abs(beta[1] - 1) < 0.2
    mu, _tt, k = clustered_mean(np.array([1.0, 1.0, -1.0, 3.0]), np.array([1, 1, 2, 3]))
    assert k == 3 and abs(mu - 1.0) < 1e-12


def _trades() -> pl.DataFrame:
    from albert.venues.kalshi import trade_rows
    d = json.loads((FIX / "trades_inplay_mlb.json").read_text())
    return pl.DataFrame(trade_rows(d))


def test_maker_fills_only_on_trades_strictly_through_our_bid():
    from albert.research.fundamental import evaluate_maker, maker_fee, maker_fills
    tr = _trades()
    assert tr.height > 100
    t0, t1 = tr["created_time"].min(), tr["created_time"].max()
    ticker = tr["ticker"][0]
    # Fair value 0.60: YES bid at 0.59 (cap: ask 0.70 − 0.01), NO bid at 0.39.
    pred = pl.DataFrame({"ticker": [ticker], "event": ["g"], "t": [t0 - timedelta(seconds=1)], "end": [t1],
                         "p": [0.60], "bid": [0.50], "ask": [0.70], "y": [1],
                         "fee_type": ["quadratic"], "fee_multiplier": [1.0]})
    f = maker_fills(pred, tr, 0.01).sort("side")
    no, yes = f.row(0, named=True), f.row(1, named=True)
    assert yes["price"] == 0.59 and no["price"] == 0.39
    sold_below = tr.filter((pl.col("taker_side") == "no") & (pl.col("yes_price") < 0.59))
    bought_above = tr.filter((pl.col("taker_side") == "yes") & (pl.col("yes_price") > 0.61))
    assert yes["filled"] == (sold_below.height > 0) and no["filled"] == (bought_above.height > 0)
    if yes["filled"]:
        assert yes["fill_ts"] == sold_below["created_time"].min()
        assert abs(yes["pnl"] - (1 - 0.59)) < 1e-12          # no maker fee on this series
    # Orders resting after the last trade never fill.
    late = pred.with_columns(pl.lit(t1).alias("t"))
    assert not maker_fills(late, tr, 0.01)["filled"].any()
    assert maker_fee(0.5, "quadratic_with_maker_fees") > 0 and maker_fee(0.5, "quadratic") == 0
    res = evaluate_maker(pred, tr, holdout_from=t1 + timedelta(days=1), min_fills=1)
    assert res["maker"] and res["maker"][0]["split"] == "dev"
