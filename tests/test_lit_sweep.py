"""Tests for the literature-sweep module (fixtures are trimmed copies of recorded files/rows)."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import lit_sweep as ls
from albert.research.counts import robots_verdict

FIX = Path(__file__).parent / "fixtures" / "lit_sweep"


def test_robots_verdicts_on_recorded_files():
    arxiv = (FIX / "robots_arxiv.org.txt").read_text()
    assert all(robots_verdict(arxiv, "/abs/2607.14430").values())
    assert not any(robots_verdict(arxiv, "/e-print/2607.14430").values())
    cepr = (FIX / "robots_cepr.org.txt").read_text()
    v = robots_verdict(cepr, "/voxeu/columns/economics-kalshi-prediction-market")
    assert v["*"] and not v["ClaudeBot"] and not v["anthropic-ai"] and not v["Claude-Web"]


def test_weighted_logistic_recovers_slope_and_weights_equal_duplication():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 20000)
    x = ls.logit(p)
    y = (rng.uniform(size=p.size) < 1 / (1 + np.exp(-2.0 * x))).astype(float)
    a, b = ls.weighted_logistic(x, y, np.ones_like(x))
    assert abs(b - 2.0) < 0.1 and abs(a) < 0.1
    # weight 2 on every row equals duplicating every row
    a2, b2 = ls.weighted_logistic(x, y, 2 * np.ones_like(x))
    assert abs(b2 - b) < 1e-6 and abs(a2 - a) < 1e-6


def test_taker_fill_rounds_fee_up_per_order_and_caps_size():
    n, fee = ls.taker_fill(0.90, 100.0, "quadratic_with_maker_fees", 1.0)
    assert n == 5 and fee == 0.04          # 0.07 * 5 * 0.9 * 0.1 = 3.15c -> 4c
    n, fee = ls.taker_fill(0.60, 2.4, "quadratic", 1.0)
    assert n == 2                           # capped by the size the print shows
    assert ls.taker_fill(0.60, 0.5, "quadratic", 1.0) == (0, 0.0)


def _market() -> pl.DataFrame:
    row = json.loads((FIX / "market_row.json").read_text())
    for k in ("close_time", "expected_expiration_time"):
        row[k] = datetime.fromisoformat(row[k])
    m = pl.DataFrame([row]).with_columns(pl.col("close_time", "expected_expiration_time").dt.cast_time_unit("us"))
    return m.with_columns((pl.col("expected_expiration_time") - pl.duration(minutes=180)).alias("sched_start"),
                          (pl.col("result") == "yes").cast(pl.Int8).alias("y"), pl.lit("holdout").alias("split"))


def test_t1_taker_on_recorded_prints_enters_after_the_decision_on_the_favourite_side():
    m = _market()
    trades = pl.read_parquet(FIX / "inplay_trades_trimmed.parquet")
    assert trades.height > 100
    sched = m["sched_start"][0]
    t0 = trades["created_time"].min()
    # a decision 3 minutes into the recorded slice, for both offsets (L chosen so that L - k lands there)
    k_min = (t0 - sched).total_seconds() / 60 + 3
    f = ls.t1_taker(m, {"KXNHLGAME": (k_min + 30, 1)}, offsets=(30,), trades=trades)
    assert f.height == 1
    r = f.row(0, named=True)
    d = sched + timedelta(minutes=k_min)
    assert r["t"] >= d + timedelta(seconds=2)
    assert r["price"] >= 0.5 and r["n"] >= 1 and r["fee"] > 0
    # the entry is a print on the favourite side at that instant
    hit = trades.filter(pl.col("created_time") == r["t"])
    assert (hit["taker_side"] == r["side"]).any()
    assert abs(r["pnl"] - (r["payoff"] - r["cost"])) < 1e-9
    # a decision after the actual close produces nothing
    late = ls.t1_taker(m, {"KXNHLGAME": (600.0, 1)}, offsets=(30,), trades=trades)
    assert late.height == 0


def test_band_expr_labels_entry_prices():
    f = pl.DataFrame({"price": [0.5, 0.69, 0.7, 0.9, 0.97, 0.999]}).with_columns(ls._band_expr("price"))
    assert f["band"].to_list() == ["[0.50,0.70)", "[0.50,0.70)", "[0.70,0.85)", "[0.85,0.95)", "[0.95,0.99)", None]
