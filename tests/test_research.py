from datetime import UTC, datetime, timedelta

import polars as pl

from albert.matching import match_kalshi_pm, parse_kalshi_game
from albert.research.calibration import CalibrationModel, kalshi_observations
from albert.research.wallets import resolved_trades

T0 = datetime(2026, 10, 1, 12, tzinfo=UTC)


def test_calibration_uses_only_pre_decision_candles():
    markets = pl.DataFrame([{"ticker": "X", "event_ticker": "E", "category": "Sports",
                             "close_time": T0, "result": "yes", "fee_type": "quadratic",
                             "fee_multiplier": 1.0}])
    candles = pl.DataFrame([
        {"ticker": "X", "end_ts": T0 - timedelta(hours=3), "yes_bid_close": 0.40, "yes_ask_close": 0.42},
        # After the decision time (close - 2h); must never be used.
        {"ticker": "X", "end_ts": T0 - timedelta(hours=1), "yes_bid_close": 0.95, "yes_ask_close": 0.97},
    ])
    obs = kalshi_observations(markets, candles, horizon_hours=2)
    assert obs.height == 1
    assert obs["yes_ask"][0] == 0.42
    assert obs["decision_ts"][0] <= T0 - timedelta(hours=2)


def test_calibration_model_shrinks_to_market_price_without_data():
    empty = pl.DataFrame(schema={"category": pl.String, "mid": pl.Float64, "y": pl.Float64,
                                 "yes_ask": pl.Float64, "yes_bid": pl.Float64, "venue": pl.String,
                                 "fee_type": pl.String, "fee_multiplier": pl.Float64})
    model = CalibrationModel.fit(empty)
    out = model.predict(pl.DataFrame([{"category": "Sports", "mid": 0.3}]))
    assert out["prob"][0] == 0.3


def test_wallet_trade_valuation():
    trades = pl.DataFrame([
        {"wallet": "w", "condition_id": "c", "outcome_index": 0, "side": "BUY", "price": 0.4,
         "size": 10.0, "ts": T0, "token_id": "t0"},
        {"wallet": "w", "condition_id": "c", "outcome_index": 1, "side": "SELL", "price": 0.7,
         "size": 10.0, "ts": T0, "token_id": "t1"},
    ])
    markets = pl.DataFrame([{"condition_id": "c", "winner_index": 0, "closed_time": T0, "tags": ["x"]}])
    rt = resolved_trades(trades, markets).sort("side")
    buy, sell = rt.row(0, named=True), rt.row(1, named=True)
    assert buy["pnl"] == 10 * (1 - 0.4) and buy["risk"] == 10 * 0.4
    # Sold outcome 1 at 0.7 and outcome 1 lost: keeps 0.7 per contract.
    assert abs(sell["pnl"] - 7.0) < 1e-9 and abs(sell["risk"] - 3.0) < 1e-9


def test_parse_kalshi_game_ticker():
    g = parse_kalshi_game("KXNBAGAME-26OCT30LALGSW-LAL")
    assert g == {"league": "nba", "date": "2026-10-30", "teams": "lalgsw", "pick": "lal"}
    assert parse_kalshi_game("KXBTC15M-26APR012000-00") is None


def test_real_matched_pair_agrees_on_outcome(fixture):
    pair = fixture("matching_pair.json")
    kalshi = pl.DataFrame(pair["kalshi"]).select("ticker")
    pm = pl.DataFrame(pair["polymarket"])
    out = match_kalshi_pm(kalshi, pm)
    assert out.height == 2
    winner = pair["polymarket"][0]["winner_index"]
    for row in out.iter_rows(named=True):
        k = next(r for r in pair["kalshi"] if r["ticker"] == row["kalshi_ticker"])
        assert (k["result"] == "yes") == (row["pm_outcome_index"] == winner)
