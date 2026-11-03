from datetime import UTC, datetime, timedelta

import polars as pl

from albert.backtest import maker

T0 = datetime(2026, 10, 1, tzinfo=UTC)


def _market(**kw):
    base = {"ticker": "M", "event_ticker": "E", "category": "Sports", "result": "yes",
            "expected_expiration_time": T0 + timedelta(hours=24), "close_time": T0 + timedelta(hours=23),
            "settlement_ts": T0 + timedelta(hours=24), "fee_type": "quadratic", "fee_multiplier": 1.0}
    return pl.DataFrame([{**base, **kw}])


def _candle(bid, ask, hours=0.0, ticker="M"):
    return {"ticker": ticker, "end_ts": T0 + timedelta(hours=hours), "period_minutes": 60,
            "yes_bid_close": bid, "yes_ask_close": ask}


def _trade(minutes, yes_price, taker_side, count=10.0, ticker="M"):
    return {"trade_id": f"{ticker}{minutes}{taker_side}", "ticker": ticker,
            "created_time": T0 + timedelta(minutes=minutes), "count": count,
            "yes_price": yes_price, "taker_side": taker_side}


def test_improves_wide_spread_and_fills_on_takers_selling_to_us():
    markets = _market()
    candles = pl.DataFrame([_candle(0.60, 0.65)])
    trades = pl.DataFrame([
        _trade(-5, 0.60, "no"),            # before decision: only proves trade coverage
        _trade(10, 0.62, "yes", 50),       # a taker *buying* YES never fills our YES bid
        _trade(20, 0.62, "no", 3),         # seller at 62c: above our 61c bid, no fill
        _trade(30, 0.61, "no", 4),         # seller at our price, we are first in queue
        _trade(40, 0.60, "no", 10),        # through our price
    ])
    orders = maker.maker_orders(markets, candles, trades, horizon_hours=24)
    o = orders.row(0, named=True)
    assert (o["side"], o["bid"], o["improved"]) == ("yes", 0.61, True)
    fills = maker.with_pnl(maker.simulate_fills(orders, trades, size=6)).row(0, named=True)
    assert fills["filled"] == 6 and fills["fill_ts"] == T0 + timedelta(minutes=40)
    assert abs(fills["pnl"] - 6 * (1 - 0.61)) < 1e-9 and fills["fee"] == 0  # no maker fee here


def test_joining_a_queue_needs_a_trade_through_our_price_and_no_side_maps_prices():
    # YES 30/31: favorite is NO, NO bid = 0.69, spread 1c -> join, need NO trades below 69c.
    markets = _market(result="no")
    candles = pl.DataFrame([_candle(0.30, 0.31)])
    trades = pl.DataFrame([
        _trade(-5, 0.30, "yes"),
        _trade(10, 0.31, "yes", 20),   # NO at 69c: at our price, but we are behind the queue
        _trade(20, 0.32, "yes", 2),    # NO at 68c: through us
    ])
    orders = maker.maker_orders(markets, candles, trades, horizon_hours=24)
    o = orders.row(0, named=True)
    assert (o["side"], o["bid"], o["improved"]) == ("no", 0.69, False)
    fills = maker.simulate_fills(orders, trades, size=5).row(0, named=True)
    assert fills["filled"] == 2


def test_orders_need_trade_coverage_and_respect_close_time():
    markets = _market()
    candles = pl.DataFrame([_candle(0.60, 0.65)])
    late_coverage = pl.DataFrame([_trade(30, 0.61, "no")])  # first recorded trade after decision
    assert maker.maker_orders(markets, candles, late_coverage, horizon_hours=24).is_empty()
    trades = pl.DataFrame([_trade(-5, 0.6, "no"), _trade(23 * 60 + 10, 0.55, "no")])  # after close
    orders = maker.maker_orders(markets, candles, trades, horizon_hours=24)
    assert maker.simulate_fills(orders, trades, size=5)["filled"][0] == 0


def test_bankroll_reserves_collateral_and_releases_unfilled():
    markets = _market()
    candles = pl.DataFrame([_candle(0.60, 0.65)])
    trades = pl.DataFrame([_trade(-5, 0.6, "no"), _trade(30, 0.61, "no", 3)])
    orders = maker.maker_orders(markets, candles, trades, horizon_hours=24)
    b = maker.run_maker_bankroll(orders, trades, cash=100.0, max_bet_dollars=5.0)
    # 8 contracts posted at 61c (5 / 0.6161), 3 filled and won: +3 * 0.39.
    assert b.orders_placed == 1 and b.orders_filled == 1
    assert abs(b.end_cash - (100 + 3 * 0.39)) < 1e-9


def test_live_quote_rule_matches_backtest_rule():
    for bid, ask in ((0.60, 0.65), (0.30, 0.31), (0.48, 0.53), (0.05, 0.09), (0.97, 0.98)):
        markets = _market()
        candles = pl.DataFrame([_candle(bid, ask)])
        trades = pl.DataFrame([_trade(-5, bid, "no")])
        orders = maker.maker_orders(markets, candles, trades, horizon_hours=24)
        live = maker.quote_order(bid, ask)
        if orders.is_empty():
            assert live is None
        else:
            o = orders.row(0, named=True)
            assert live == {"side": o["side"], "bid": o["bid"], "ask": o["ask"], "improved": o["improved"]}
