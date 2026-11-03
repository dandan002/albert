"""Paper trader bookkeeping (no network: quotes and resolutions are stubbed)."""

import json
from dataclasses import replace

import pytest

from albert.config import Settings
from albert.paper import PaperState, PaperTrader
from albert.risk import RiskConfig


@pytest.fixture
def trader(tmp_path):
    s = Settings(data_dir=tmp_path)
    PaperTrader.reset(s, {"kalshi": 50.0, "pmus": 50.0})
    return PaperTrader(s, c=None, cfg=RiskConfig(), strategies=())


def test_fill_moves_cash_and_logs(trader):
    ok = trader._try_fill(strategy="t", venue="kalshi", market_id="K1", side="yes", event_id="E",
                          price=0.40, prob=0.60, available=100)
    assert ok
    pos = trader.state.positions[0]
    assert trader.state.cash["kalshi"] == pytest.approx(50 - pos.cost)
    assert trader.state.cash["pmus"] == 50
    events = [json.loads(line)["event"] for line in trader.ledger_path.read_text().splitlines()]
    assert events[-1] == "fill"
    # no pyramiding
    assert not trader._try_fill(strategy="t", venue="kalshi", market_id="K1", side="yes",
                                event_id="E", price=0.40, prob=0.60, available=100)


def test_settlement_credits_the_right_venue(trader, monkeypatch):
    trader._try_fill(strategy="t", venue="pmus", market_id="U1", side="long", event_id="E",
                     price=0.40, prob=0.60, available=100)
    pos = trader.state.positions[0]
    monkeypatch.setattr(trader, "_resolution", lambda p: 1.0)
    assert trader.settle() == 1
    assert trader.state.positions == []
    assert trader.state.cash["pmus"] == pytest.approx(50 - pos.cost + pos.contracts)
    assert trader.state.realized_pnl == pytest.approx(pos.contracts - pos.cost)


def test_drawdown_halts_new_trading(trader, monkeypatch):
    trader.state.peak_equity = 300.0  # equity 100 < 50% of peak
    trader.strategies = ("longshot",)
    called = []
    monkeypatch.setattr(trader, "run_longshot", lambda: called.append(1) or 0)
    trader.tick()
    assert trader.state.halted and not called
    reloaded = PaperState.load(trader.state_path)
    assert reloaded.halted


def test_requires_reset(tmp_path):
    with pytest.raises(RuntimeError):
        PaperTrader(replace(Settings(), data_dir=tmp_path / "none"), c=None, strategies=())


def _order(trader, *, contracts=10, bid=0.61, improved=True, hours_left=5.0, placed_minutes_ago=30):
    from datetime import UTC, datetime, timedelta

    from albert.fees import kalshi_fee
    from albert.paper import RestingOrder

    now = datetime.now(UTC)
    reserve = contracts * bid + kalshi_fee(contracts, bid, maker=True)
    o = RestingOrder("K1", "E1", "Sports", "yes", bid, contracts, improved, reserve,
                     (now - timedelta(minutes=placed_minutes_ago)).isoformat(),
                     (now + timedelta(hours=hours_left)).isoformat(),
                     (now - timedelta(minutes=placed_minutes_ago)).isoformat(), "quadratic", 1.0)
    trader.state.cash["kalshi"] -= reserve
    trader.state.orders.append(o)
    return o


def _trades(*rows):
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from albert.venues import kalshi as k

    now = datetime.now(UTC)
    return pl.DataFrame([{"trade_id": str(i), "ticker": "K1", "created_time": now - timedelta(minutes=m),
                          "count": n, "yes_price": px, "taker_side": side}
                         for i, (m, px, side, n) in enumerate(rows)], schema=k.TRADE_SCHEMA)


def test_maker_order_fills_only_from_trades_into_our_price(trader, monkeypatch):
    equity0 = trader.state.equity()
    _order(trader, contracts=10)
    assert trader.state.equity() == pytest.approx(equity0)  # reserved collateral counts as equity
    monkeypatch.setattr(trader, "_new_trades", lambda ticker, after: _trades(
        (20, 0.61, "yes", 50.0),   # taker bought YES: not a fill for our YES bid
        (10, 0.61, "no", 4.0),     # taker sold YES at our price: fill 4
    ))
    assert trader.update_orders() == 1
    o, pos = trader.state.orders[0], trader.state.positions[0]
    assert (o.filled, pos.contracts, pos.price, pos.strategy) == (4, 4, 0.61, "maker")
    assert trader.state.equity() == pytest.approx(equity0)  # filling moves value, not equity
    # Later polls only see newer trades; the same prints are not counted twice.
    monkeypatch.setattr(trader, "_new_trades", lambda ticker, after: _trades()[:0])
    assert trader.update_orders() == 0 and trader.state.orders[0].filled == 4


def test_maker_order_ttl_releases_unfilled_collateral(trader, monkeypatch):
    _order(trader, contracts=10, hours_left=-0.1)
    monkeypatch.setattr(trader, "_new_trades", lambda ticker, after: _trades(
        (10, 0.60, "no", 3.0),  # before the cancel time: fills
        (2, 0.55, "no", 9.0),   # after the cancel time: ignored
    ))
    trader.update_orders()
    assert trader.state.orders == []
    pos = trader.state.positions[0]
    assert pos.contracts == 3
    assert trader.state.cash["kalshi"] == pytest.approx(50 - pos.cost)


def test_halt_cancels_resting_orders(trader):
    _order(trader)
    trader.state.peak_equity = 300.0
    trader.strategies = ("maker",)
    counts = trader.tick()
    assert counts["cancelled"] == 1 and trader.state.orders == []
    assert trader.state.cash["kalshi"] == pytest.approx(50.0)


def test_old_state_files_without_orders_still_load(trader):
    raw = json.loads(trader.state_path.read_text())
    raw.pop("orders", None)
    raw.pop("maker_scan_ts", None)
    trader.state_path.write_text(json.dumps(raw))
    assert PaperState.load(trader.state_path).orders == []


def test_game_key_groups_markets_on_the_same_match():
    from albert.paper import _game_key

    assert _game_key("KXCS2SPREAD-26NOV040430PRVMOUZ") == _game_key("KXCS2GAME-26NOV040430PRVMOUZ")
    assert _game_key("KXHIGHNY-26NOV04") != _game_key("KXHIGHCHI-26NOV05")


def test_elo_scan_buys_when_blend_beats_ask(trader, monkeypatch):
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace
    from zoneinfo import ZoneInfo

    from albert import paper
    from albert.research.ratings import Ratings

    start = (datetime.now(UTC) + timedelta(hours=1)).astimezone(ZoneInfo("America/New_York"))
    ev = "KXCS2MAP-" + start.strftime("%y%b%d%H%M").upper() + "AAABBB-1"
    legs = [{"ticker": f"{ev}-AAA", "event_ticker": ev, "yes_sub_title": "Alpha", "fee_type": "quadratic",
             "fee_multiplier": 1.0, "category": "Sports"},
            {"ticker": f"{ev}-BBB", "event_ticker": ev, "yes_sub_title": "Bravo", "fee_type": "quadratic",
             "fee_multiplier": 1.0, "category": "Sports"}]
    pages = {"KXCS2MAP": [SimpleNamespace(payload=legs)]}
    trader.c = SimpleNamespace(kalshi=SimpleNamespace(series_markets=lambda ser, status: pages.get(ser, [])))
    monkeypatch.setattr(paper.k, "market_rows", lambda payload, info, src: payload)
    trader._series = {}
    trader._ratings = Ratings({("KXCS2GAME", "alpha"): 1800.0, ("KXCS2GAME", "bravo"): 1400.0},
                              {("KXCS2GAME", "alpha"): 50, ("KXCS2GAME", "bravo"): 50})
    trader._ratings_at = datetime.now(UTC)
    monkeypatch.setattr(trader, "kalshi_quote", lambda t: {"yes_bid": 0.50, "yes_ask": 0.52,
                                                            "yes_bid_size": 100.0, "yes_ask_size": 100.0})
    assert trader.run_elo() == 1
    pos = trader.state.positions[0]
    assert (pos.strategy, pos.market_id, pos.side) == ("elo", f"{ev}-AAA", "yes")
    assert pos.prob > 0.6  # Elo says Alpha is a strong favourite; the market mid is 0.51
    assert trader.run_elo() == 0  # scans are throttled to every 10 minutes
