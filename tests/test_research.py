from datetime import UTC, datetime, timedelta

import polars as pl

from albert.matching import kalshi_ticker_for, match_kalshi_pm, parse_kalshi_game
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


def test_decisions_anchor_on_scheduled_expiry_not_actual_close():
    base = {"event_ticker": "E", "category": "Sports", "result": "no", "fee_type": "quadratic",
            "fee_multiplier": 1.0, "expected_expiration_time": T0}
    markets = pl.DataFrame([
        # Closed 1h early: decision must still be scheduled - 2h, not actual close - 2h.
        {**base, "ticker": "EARLY", "close_time": T0 - timedelta(hours=1)},
        # Closed before the decision time: no tradable observation exists.
        {**base, "ticker": "GONE", "close_time": T0 - timedelta(hours=3)},
    ])
    candles = pl.DataFrame([
        {"ticker": t, "end_ts": T0 - timedelta(hours=h), "yes_bid_close": b, "yes_ask_close": b + 0.02}
        for t in ("EARLY", "GONE") for h, b in ((4, 0.30), (2, 0.40), (3, 0.35))
    ])
    obs = kalshi_observations(markets, candles, horizon_hours=2)
    assert obs["market_id"].to_list() == ["EARLY"]
    assert obs["decision_ts"][0] == T0 - timedelta(hours=2) and obs["yes_bid"][0] == 0.40
    leaky = kalshi_observations(markets, candles, horizon_hours=2, anchor="close")
    assert leaky.filter(pl.col("market_id") == "EARLY")["yes_bid"][0] == 0.35


def test_start_anchor_uses_ticker_game_time():
    # KXKBOGAME-26OCT020430... starts 04:30 EDT = 08:30 UTC on Oct 2.
    start = datetime(2026, 10, 2, 8, 30, tzinfo=UTC)
    markets = pl.DataFrame([{"ticker": "KXKBOGAME-26OCT020430SAMKTW-SAM",
                             "event_ticker": "KXKBOGAME-26OCT020430SAMKTW", "category": "Sports",
                             "close_time": start + timedelta(hours=4), "result": "yes",
                             "fee_type": "quadratic", "fee_multiplier": 1.0,
                             "expected_expiration_time": start + timedelta(hours=3)}])
    candles = pl.DataFrame([
        {"ticker": "KXKBOGAME-26OCT020430SAMKTW-SAM", "end_ts": start - timedelta(hours=1, minutes=30),
         "yes_bid_close": 0.55, "yes_ask_close": 0.57},
        {"ticker": "KXKBOGAME-26OCT020430SAMKTW-SAM", "end_ts": start - timedelta(minutes=30),
         "yes_bid_close": 0.80, "yes_ask_close": 0.82},  # after the 1h-lead decision
    ])
    obs = kalshi_observations(markets, candles, horizon_hours=1.0, anchor="start", staleness_hours=2)
    assert obs.height == 1 and obs["yes_bid"][0] == 0.55


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


def test_kalshi_ticker_for_inverts_the_real_match(fixture):
    pair = fixture("matching_pair.json")
    event = pair["polymarket"][0]["event_slug"]
    built = {kalshi_ticker_for(event, i) for i in (0, 1)}
    assert built == {r["ticker"] for r in pair["kalshi"]}
    assert kalshi_ticker_for("ucl-ars-bay-2026-10-21", 0) is None  # soccer (3-way) unsupported
