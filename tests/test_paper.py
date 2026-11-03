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
