from datetime import UTC, datetime, timedelta

import polars as pl

from albert.backtest.engine import run_backtest
from albert.research.follow import SIGNAL_SCHEMA
from albert.risk import RiskConfig

T0 = datetime(2026, 10, 1, tzinfo=UTC)


def sig(market, ts_h, resolve_h, price, prob, payoff, venue="kalshi", event=None):
    return {"strategy": "t", "venue": venue, "market_id": market, "event_id": event or market,
            "side": "yes", "signal_ts": T0 + timedelta(hours=ts_h),
            "resolve_ts": T0 + timedelta(hours=resolve_h), "price": price, "prob": prob,
            "available": None, "payoff": payoff, "fee_type": "quadratic",
            "fee_multiplier": 1.0, "note": ""}


def frame(rows):
    return pl.DataFrame(rows, schema=SIGNAL_SCHEMA)


def test_win_and_loss_settle_into_the_right_venue():
    cfg = RiskConfig(kelly_fraction=1.0, max_bet_fraction=0.2, max_bet_dollars=10)
    res = run_backtest(frame([sig("a", 0, 1, 0.5, 0.7, 1.0), sig("b", 0, 1, 0.5, 0.7, 0.0, venue="pmus")]),
                       cash={"kalshi": 50, "pmus": 50}, cfg=cfg)
    a, b = res.fills
    assert a.pnl == a.contracts - a.cost and a.pnl > 0
    assert b.pnl == -b.cost
    assert res.end_cash["kalshi"] == 50 + a.pnl
    assert res.end_cash["pmus"] == 50 + b.pnl


def test_sizing_does_not_depend_on_payoff():
    cfg = RiskConfig()
    win = run_backtest(frame([sig("a", 0, 1, 0.4, 0.6, 1.0)]), cash={"kalshi": 50, "pmus": 0}, cfg=cfg)
    loss = run_backtest(frame([sig("a", 0, 1, 0.4, 0.6, 0.0)]), cash={"kalshi": 50, "pmus": 0}, cfg=cfg)
    assert win.fills[0].contracts == loss.fills[0].contracts > 0


def test_no_pyramiding_and_cash_released_only_after_resolution():
    cfg = RiskConfig(kelly_fraction=1.0, max_bet_fraction=1.0, max_bet_dollars=100, max_event_fraction=1.0)
    rows = [sig("a", 0, 10, 0.5, 0.9, 1.0), sig("a", 1, 10, 0.5, 0.9, 1.0), sig("c", 2, 20, 0.5, 0.9, 1.0)]
    res = run_backtest(frame(rows), cash={"kalshi": 10, "pmus": 0}, cfg=cfg)
    assert res.skipped.get("already_held") == 1
    # "c" arrives before "a" resolves, so it can only use cash left over after "a".
    assert sum(f.cost for f in res.fills) <= 10 + 1e-9


def test_drawdown_kill_switch():
    cfg = RiskConfig(kelly_fraction=1.0, max_bet_fraction=0.6, max_bet_dollars=100,
                     max_event_fraction=1.0, drawdown_kill=0.5, daily_loss_fraction=1.0)
    rows = [sig(f"m{i}", i * 2, i * 2 + 1, 0.5, 0.95, 0.0) for i in range(6)]
    res = run_backtest(frame(rows), cash={"kalshi": 100, "pmus": 0}, cfg=cfg)
    assert res.halted_at is not None
    assert res.end_equity > 0
