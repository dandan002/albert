"""TA-factor and day-of-week research helpers: panel on recorded payloads, causality of the
indicators, fees, settlement-day drift, and the capped book."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from albert.fees import kalshi_fee
from albert.research import ta_calendar as tc
from albert.venues import kalshi as k

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _recorded():
    market = json.loads((FIX / "settled_markets.json").read_text())
    candles = json.loads((FIX / "candles_live.json").read_text())
    info = {"KXNBAGAME": {"category": "Sports", "fee_type": "quadratic", "fee_multiplier": 1.0}}
    m = k.markets_frame(k.market_rows(market, info, "live"))
    return pl.DataFrame(k.candle_rows(candles, candles["ticker"], 60)), m


def test_panel_on_recorded_candles_is_gridded_and_labelled_in_eastern_time():
    c, m = _recorded()
    p = tc.build_panel(c, m, prefixes=("KXNBAGAME",))
    assert p.height >= c.height > 0
    assert p["real"].sum() == c.height                      # every recorded candle survives
    gaps = p.sort("t")["t"].diff().drop_nulls().dt.total_minutes().unique().to_list()
    assert gaps == [60]                                      # a regular hourly grid
    assert (p["h_to_exp"] > 0).all() and p["y"].unique().to_list() == [0.0]
    for r in p.iter_rows(named=True):
        et = r["t"].astimezone(ZoneInfo(tc.ET))
        assert (r["dow"], r["et_hour"], r["et_date"], r["weekend"]) == (et.isoweekday(), et.hour, et.date(),
                                                                         et.isoweekday() >= 6)


def _series(mids: list[float], t0: datetime = datetime(2026, 9, 1, tzinfo=UTC)) -> pl.DataFrame:
    n = len(mids)
    return pl.DataFrame({
        "ticker": ["A"] * n, "t": [t0 + timedelta(hours=i) for i in range(n)],
        "bid": [m - 0.01 for m in mids], "ask": [m + 0.01 for m in mids],
        "volume": [float(i % 3) for i in range(n)], "oi": [100.0 + i for i in range(n)],
    })


def test_hourly_grid_carries_quotes_forward_and_zeroes_volume():
    s = _series([0.40, 0.42, 0.45]).filter(pl.col("t") != datetime(2026, 9, 1, 1, tzinfo=UTC))
    g = tc.hourly_grid(s)
    assert g.height == 3
    filled = g.filter(~pl.col("real")).row(0, named=True)
    assert filled["bid"] == pytest.approx(0.39) and filled["volume"] == 0.0


def test_indicators_use_only_past_hours():
    rng = np.random.default_rng(3)
    mids = list(np.clip(0.5 + np.cumsum(rng.normal(0, 0.02, 80)), 0.05, 0.95).round(2))
    full = tc.add_indicators(tc.with_quote_cols(_series(mids)))
    cut = 50
    part = tc.add_indicators(tc.with_quote_cols(_series(mids[:cut])))
    cols = ["rsi6", "rsi14", "ema3_12", "ema6_24", "bz12", "bz24", "hi12", "lo48", "macd", "mh", "r6", "v6", "vbase"]
    a = full.head(cut).select(cols).fill_nan(None)
    b = part.select(cols).fill_nan(None)
    for col in cols:
        x, y = a[col].to_list(), b[col].to_list()
        assert [v is None for v in x] == [v is None for v in y], col
        assert [v for v in x if v is not None] == pytest.approx([v for v in y if v is not None]), col
    # every signal is computable on this series and fires at least once
    sigs = tc.ta_signals()
    q = full.with_columns([(up | dn).alias(name) for name, (up, dn) in sigs.items()])
    fired = [name for name in sigs if q[name].fill_null(False).any()]
    assert "rsi6 70/30" in fired and "ema 3/12 cross" in fired


def test_taker_fee_expression_matches_kalshi_fee_per_order():
    df = pl.DataFrame({"n": [1.0, 2.0, 10.0, 19.0, 3.0], "price": [0.5, 0.5, 0.1, 0.05, 0.33],
                       "fee_multiplier": [1.0, 1.0, 1.0, 0.5, None]})
    got = df.with_columns(tc.taker_fee_expr(pl.col("n"), pl.col("price")).alias("f"))["f"].to_list()
    exp = [kalshi_fee(n, p, multiplier=1.0 if mu is None else mu) for n, p, mu in df.iter_rows()]
    assert got == pytest.approx(exp)


def test_take_sizes_a_dollar_order_and_scores_both_sides():
    f = pl.DataFrame({"bid": [0.30, 0.30], "ask": [0.34, 0.34], "spread": [0.04, 0.04], "mid": [0.32, 0.32],
                      "y": [1.0, 1.0], "fee_multiplier": [1.0, 1.0], "d": [1, -1]})
    t = tc.take(f, pl.col("d")).sort("dir", descending=True)
    yes, no = t.row(0, named=True), t.row(1, named=True)
    assert yes["price"] == pytest.approx(0.34) and yes["n"] == 2
    assert yes["fee_order"] == pytest.approx(kalshi_fee(2, 0.34))
    assert yes["pnl"] == pytest.approx(1 - 0.34 - yes["fee_order"] / 2)
    assert no["price"] == pytest.approx(0.70) and no["n"] == 1 and no["payoff"] == 0.0
    assert yes["gross"] == pytest.approx(1 - 0.32) and no["gross"] == pytest.approx(-(1 - 0.32))


def test_daily_moves_value_a_closing_day_at_settlement():
    t0 = datetime(2026, 9, 7, 4, tzinfo=UTC)               # 00:00 EDT, a Monday
    hp = pl.DataFrame({
        "ticker": ["A", "A"], "cluster": ["A", "A"], "category": ["Politics"] * 2, "kind": ["event"] * 2,
        "holdout": [False, False], "t": [t0, t0 + timedelta(hours=24)], "smid": [0.30, 0.20],
        "et_date": [date(2026, 9, 7), date(2026, 9, 8)], "dow": [1, 2], "weekend": [False, False],
        "et_hour": [0, 0], "y": [1.0, 1.0], "close_time": [t0 + timedelta(hours=30)] * 2})
    dm = tc.daily_moves(hp).sort("t")
    assert dm["dmid"].to_list() == pytest.approx([-0.10, 0.80])   # day 2 ends at the YES settlement
    assert dm["settled_today"].to_list() == [False, True]


def test_capped_ledger_respects_bankroll_and_event_cap():
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    rows = [{"t": t0 + timedelta(minutes=i), "ticker": f"T{i}", "event_ticker": "E1" if i < 5 else f"E{i}",
             "n": 1.0, "price": 0.95, "fee_order": 0.01, "dir": 1, "payoff": 1.0,
             "settlement_ts": t0 + timedelta(days=1)} for i in range(120)]
    led = tc.capped_ledger(pl.DataFrame(rows), "s", "v", "all", bankroll=100.0, event_cap=3.0)
    assert (led.filter(pl.col("event_ticker") == "E1")["cost_usd"].sum()) <= 3.0 + 1e-9
    assert led["cost_usd"].sum() <= 100.0 + 1e-9
    assert set(led.columns) >= {"struct_id", "event_ticker", "ticker", "side", "n", "t_in", "cost_usd",
                                "settle_ts", "payoff_usd"}


def test_weekend_gap_recovers_a_planted_difference():
    rng = np.random.default_rng(0)
    days = [date(2026, 9, 1) + timedelta(days=i) for i in range(56)]
    rows = []
    for d in days:
        wk = d.isoweekday() >= 6
        for j in range(30):
            rows.append({"et_date": d, "weekend": wk, "cluster": f"{d}-{j}",
                         "resid": (0.05 if wk else 0.0) + rng.normal(0, 0.1)})
    g = tc.weekend_gap(pl.DataFrame(rows))
    assert g["weekend_minus_weekday_c"] == pytest.approx(5.0, abs=1.0)
    assert g["t_by_date"] > 3
