import pytest
from datetime import datetime
from albert.strategies.base import BaseStrategy
from albert.events import MarketDataEvent, OrderIntent


class DoubleEdgeStrategy(BaseStrategy):
    async def on_market_data(self, event: MarketDataEvent) -> list[OrderIntent] | None:
        edge = self.estimate_edge(event)
        if edge <= 0:
            return None
        return [OrderIntent(
            market_id=event.market_id,
            strategy_id=self.id,
            side="yes",
            edge=edge,
            confidence=1.0,
        )]

    def estimate_edge(self, event: MarketDataEvent) -> float:
        return max(0.0, 0.5 - event.yes_ask)


def make_event(yes_ask: float) -> MarketDataEvent:
    return MarketDataEvent(
        market_id="kalshi:X",
        exchange="kalshi",
        timestamp=datetime.utcnow(),
        yes_bid=yes_ask - 0.02,
        yes_ask=yes_ask,
        no_bid=0.0,
        no_ask=0.0,
        last_price=yes_ask,
        volume=0.0,
    )


async def test_strategy_returns_intent_when_edge_positive():
    s = DoubleEdgeStrategy(strategy_id="s1", config={"min_edge": 0.0})
    intents = await s.on_market_data(make_event(0.40))
    assert intents is not None
    assert len(intents) == 1
    assert intents[0].side == "yes"
    assert intents[0].edge == pytest.approx(0.10)


async def test_strategy_returns_none_when_no_edge():
    s = DoubleEdgeStrategy(strategy_id="s1", config={})
    intents = await s.on_market_data(make_event(0.55))
    assert intents is None


async def test_momentum_strategy_returns_intent_below_threshold():
    from albert.strategies.examples.momentum import MomentumV1
    s = MomentumV1(strategy_id="momentum_v1", config={"min_edge": 0.05})
    intents = await s.on_market_data(make_event(0.30))
    assert intents is not None
    assert intents[0].edge == pytest.approx(0.20)


async def test_momentum_strategy_returns_none_above_threshold():
    from albert.strategies.examples.momentum import MomentumV1
    s = MomentumV1(strategy_id="momentum_v1", config={"min_edge": 0.05})
    intents = await s.on_market_data(make_event(0.50))
    assert intents is None
