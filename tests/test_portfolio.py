"""Tests for research/portfolio: the cmn rule on trimmed real order rows (culture_flow_review prints,
holdout), the cash replay's accounting, and the small statistics helpers."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import portfolio as P

FIX = Path(__file__).parent / "fixtures" / "portfolio" / "cmn_prints_trim.parquet"


def rule_rows() -> pl.DataFrame:
    return P.cmn_rule(pl.read_parquet(FIX)).with_columns(pl.lit("holdout").alias("split"))


def test_cmn_rule_on_real_rows():
    raw = pl.read_parquet(FIX)
    q = rule_rows()
    assert q.height > 0
    # two-sided books only, mid <= 5c, sell YES above the bid at ceil(mid + 12c), 1 - that as the NO price
    assert (q["bid"] >= 0.01 - 1e-9).all() and (q["ask"] <= 0.99 + 1e-9).all() and (q["mid"] <= 0.05 + 1e-9).all()
    assert ((q["sell_at"] - q["mid"] - 0.12) >= -1e-9).all() and ((q["sell_at"] - q["mid"] - 0.12) < 0.01).all()
    assert ((q["price"] + q["sell_at"] - 1).abs() < 1e-9).all()
    # at most 3 legs per event, the cheapest NO first; the 5-leg YouTube event is cut to 3
    per = q.group_by("event_ticker").len()
    assert per["len"].max() == 3
    yt = q.filter(pl.col("event_ticker") == "KXYTVIEWSW-YOU26NOV15")
    two = raw.filter((pl.col("event_ticker") == "KXYTVIEWSW-YOU26NOV15") & (pl.col("bid") >= 0.01)
                     & (pl.col("ask") <= 0.99) & (pl.col("mid") <= 0.05))
    assert yt.height == 3 and two.height == 5
    assert yt["price"].max() <= (1 - (two["mid"] + 0.12)).sort().to_list()[2] + 0.011
    # fills only on prints after the decision and before the leg closes
    f = q.filter(pl.col("fill_ts").is_not_null())
    assert f.height >= 3
    assert ((f["fill_ts"] > f["t"]) & (f["fill_ts"] <= f["close_time"])).all()


def test_replay_cash_cap_and_accounting():
    q = rule_rows()
    big, st_big = P.cmn_replay(q, P.CashPolicy("cap100", cap=100))
    assert big.height == q.height and st_big["skipped_cash"] == 0
    # a cap of one order's collateral: only one order rests at a time
    _, st = P.cmn_replay(q, P.CashPolicy("cap0.9", cap=0.9))
    assert st["peak_used"] <= 0.9 + 1e-9 and st["skipped_cash"] > 0
    # PnL of a fill is 1 - y - price; unfilled orders book 0
    f = big.filter("filled")
    assert np.allclose(f["pnl"].to_numpy(), (1 - f["y"].cast(pl.Float64) - f["price"]).to_numpy())
    assert (big.filter(~pl.col("filled"))["pnl"] == 0).all()
    led = P.cmn_ledger(big, "cap100", "holdout")
    assert led.height == f.height and set(led.columns) >= {"t_in", "settle_ts", "cost_usd", "payoff_usd", "event_ticker"}


def test_replay_shared_pool_cancels_for_mentions():
    q = rule_rows()
    t0 = q["t"].min().replace(minute=0, second=0, microsecond=0)
    hours = pl.datetime_range(t0, q["settle_ts"].max(), "1h", time_zone="UTC", eager=True)
    # mentions holds all but $1 of the account during the first day, then nothing
    usd = [P.ACCOUNT_CASH - 5 - 1.0 if h < t0 + timedelta(days=1) else 0.0 for h in hours]
    mu = pl.DataFrame({"ts": hours, "usd": usd})
    b, st = P.cmn_replay(q, P.CashPolicy("shared", cap=100, shared=True), mentions_usage=mu)
    b0, _ = P.cmn_replay(q, P.CashPolicy("cap100", cap=100))
    assert b.height < b0.height or st["cancels"] > 0


def test_cash_intervals_and_usage():
    t = datetime(2026, 10, 1, tzinfo=UTC)
    rows = [{"post_ts": t, "fill_ts": t + timedelta(hours=2), "cancelled_ts": None, "end": t + timedelta(hours=5),
             "settle_ts": t + timedelta(hours=10), "price": 0.85},
            {"post_ts": t, "fill_ts": None, "cancelled_ts": t + timedelta(hours=1), "end": t + timedelta(hours=5),
             "settle_ts": t + timedelta(hours=10), "price": 0.5}]
    iv = P.cash_intervals(rows)
    u = P.hourly_usage(iv, t, t + timedelta(hours=12))
    v = dict(zip(u["ts"].to_list(), u["usd"].to_list(), strict=True))
    assert abs(v[t] - 1.35) < 1e-9                       # both resting
    assert abs(v[t + timedelta(hours=1)] - 0.85) < 1e-9  # the second cancelled
    assert abs(v[t + timedelta(hours=9)] - 0.85) < 1e-9  # first is a position until settlement
    assert v[t + timedelta(hours=10)] == 0


def test_kelly_and_sample_size():
    # a bet that wins +1 per $ with p = 0.6 and loses the stake otherwise: Kelly f = 2p - 1 = 0.2
    r = np.array([1.0] * 6 + [-1.0] * 4)
    assert abs(P.kelly(r) - 0.2) < 0.002
    assert P.required_fills(0.05, 0.2) == 64
    assert P.required_fills(0.05, 0.2, deff=2.0) == 128
    assert P.required_fills(-0.01, 0.2) == -1


def test_month_loss_prob_shapes():
    x = np.r_[np.full(60, 1.0), np.full(10, -3.0)]
    m = P.month_loss_prob(x, reps=500)
    assert 0 <= m["p_loss_month"] <= 1 and m["month_p05"] <= m["month_median"]
    mc = P.event_mc({"a": (np.array([1.0, 1.0, -2.0]), 1.0)}, reps=500, months=6)
    assert 0 <= mc["p_loss_month"] <= 1 and mc["dd12_p95"] <= 0
