from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from albert.research import literature as lit

T0 = datetime(2026, 10, 1, tzinfo=UTC)


def _markets(rows):
    return pl.DataFrame([{"fee_type": "quadratic", "fee_multiplier": 1.0, "category": "Sports",
                          "close_time": T0 + timedelta(days=5), **r} for r in rows])


def test_trade_legs_split_taker_and_maker_sides():
    markets = _markets([{"ticker": "A", "event_ticker": "E", "result": "yes"}])
    trades = pl.DataFrame({"trade_id": ["1"], "ticker": ["A"], "created_time": [T0],
                           "count": [10.0], "yes_price": [0.70], "taker_side": ["no"]})
    legs = lit.trade_legs(trades, markets)
    taker = legs.filter(pl.col("role") == "taker").row(0, named=True)
    maker = legs.filter(pl.col("role") == "maker").row(0, named=True)
    assert (taker["side"], maker["side"]) == ("no", "yes")
    assert abs(taker["price"] - 0.30) < 1e-9 and abs(maker["price"] - 0.70) < 1e-9
    assert taker["fee"] > 0 and maker["fee"] == 0  # no maker fee on plain quadratic series
    assert taker["pnl"] < 0 < maker["pnl"]
    assert abs(maker["pnl"] - 10 * 0.30) < 1e-9


def test_cluster_ratio_is_pnl_over_stake():
    f = pl.DataFrame({"g": ["a"] * 4, "event_ticker": ["e1", "e1", "e2", "e3"],
                      "pnl": [1.0, -1.0, 2.0, 0.0], "stake": [2.0, 2.0, 2.0, 2.0]})
    out = lit.cluster_ratio(f, ["g"]).row(0, named=True)
    assert out["roi"] == 2.0 / 8.0 and out["clusters"] == 3 and out["se"] > 0


def _candle(ticker, bid, ask, hour=0):
    return {"ticker": ticker, "end_ts": T0 + timedelta(hours=hour), "period_minutes": 60,
            "yes_bid_close": bid, "yes_ask_close": ask}


def test_leg_sum_arbitrage_needs_complete_exclusive_events():
    markets = _markets([{"ticker": t, "event_ticker": e, "result": "no"}
                        for t, e in [("A1", "A"), ("A2", "A"), ("A3", "A"),
                                     ("B1", "B"), ("B2", "B"), ("C1", "C"), ("C2", "C")]])
    events = pl.DataFrame({"event_ticker": ["A", "B", "C"], "mutually_exclusive": [True, True, False],
                           "n_markets": [3, 3, 2]})
    candles = pl.DataFrame([_candle("A1", 0.28, 0.30), _candle("A2", 0.28, 0.30),
                            _candle("A3", 0.28, 0.30), _candle("B1", 0.1, 0.2),
                            _candle("B2", 0.1, 0.2), _candle("C1", 0.1, 0.2), _candle("C2", 0.1, 0.2)])
    hours = lit.leg_sum_arbitrage(candles, markets, events)
    assert hours["event_ticker"].to_list() == ["A"]  # B incomplete, C not exclusive
    row = hours.row(0, named=True)
    fee = 0.0147  # ceil(0.07 * 100 * 0.3 * 0.7) / 100 per contract on a 100-lot
    assert abs(row["long_profit"] - (1 - 3 * (0.30 + fee))) < 1e-9 and row["long_profit"] > 0
    # No leg won (mutually exclusive is "at most one"), so the long basket actually lost.
    assert row["n_yes"] == 0 and row["long_realized"] < 0


def test_logistic_fit_recovers_underconfidence_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1.5, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(0.2 + 1.5 * x)))).astype(float)
    a, b = lit.logistic_fit(x, y)
    assert abs(b - 1.5) < 0.1 and abs(a - 0.2) < 0.1


def test_rolling_min_entries_use_only_past_prices():
    markets = _markets([{"ticker": "A", "event_ticker": "E", "result": "yes"}])
    mids = [0.60, 0.58, 0.56, 0.50, 0.52, 0.70, 0.71, 0.72]
    candles = pl.DataFrame([_candle("A", m - 0.01, m + 0.01, h) for h, m in enumerate(mids)])
    trades = lit.rolling_min_reversion(candles, markets, lookback_hours=2, hold_hours=2)
    # Hour 0-1 lack history; hours 2 and 3 make new lows but 3 is inside hour 2's holding window.
    assert trades["entry_ts"].to_list() == [T0 + timedelta(hours=2)]
    t = trades.row(0, named=True)
    assert abs(t["entry_ask"] - 0.57) < 1e-9 and abs(t["exit_price"] - 0.51) < 1e-9
    assert not t["settled"] and t["pnl"] < 0


def test_move_reversal_detects_overreaction():
    rng = np.random.default_rng(2)
    n = 4000
    early = rng.uniform(0.3, 0.7, n)
    move = rng.normal(0, 0.1, n)
    late = np.clip(early + move, 0.02, 0.98)
    truth = np.clip(early + 0.5 * move, 0.01, 0.99)  # half the move is noise
    y = (rng.random(n) < truth).astype(float)
    base = {"market_id": [f"m{i}" for i in range(n)], "event_id": [f"e{i}" for i in range(n)],
            "category": ["Sports"] * n}
    obs_late = pl.DataFrame({**base, "mid": late, "y": y, "decision_ts": [T0 + timedelta(hours=24)] * n})
    obs_early = pl.DataFrame({**base, "mid": early, "y": y, "decision_ts": [T0] * n})
    slopes = lit.move_reversal_slopes(lit.move_reversal(obs_late, obs_early))
    row = slopes.filter(pl.col("category") == "ALL").row(0, named=True)
    assert row["slope"] < 0 and row["t"] < -3
