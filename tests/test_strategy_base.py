import pytest
from datetime import datetime, timezone
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
        timestamp=datetime.now(timezone.utc),
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


async def test_mean_reversion_strategy_emits_when_price_below_mean():
    from albert.strategies.examples.mean_reversion import MeanReversionStrategy
    s = MeanReversionStrategy(
        strategy_id="mean_rev_v1",
        config={"window_size": 10, "min_edge": 0.02}
    )
    # Add initial prices to populate window
    for price in [0.50, 0.52, 0.48, 0.51, 0.49, 0.50, 0.51, 0.49, 0.50, 0.51]:
        s.estimate_edge(MarketDataEvent(
            market_id="kalshi:X",
            exchange="kalshi",
            timestamp=datetime.now(timezone.utc),
            yes_bid=price - 0.01,
            yes_ask=price,
            no_bid=0.0,
            no_ask=0.0,
            last_price=price,
            volume=0.0,
        ))
    # Now price drops below mean - should emit
    intents = await s.on_market_data(make_event(0.30))
    assert intents is not None
    assert intents[0].side == "yes"


async def test_mean_reversion_strategy_returns_none_above_mean():
    from albert.strategies.examples.mean_reversion import MeanReversionStrategy
    s = MeanReversionStrategy(
        strategy_id="mean_rev_v1",
        config={"window_size": 10, "min_edge": 0.02}
    )
    # Add prices
    for price in [0.40, 0.42, 0.38, 0.41, 0.39, 0.40]:
        s.estimate_edge(MarketDataEvent(
            market_id="kalshi:X",
            exchange="kalshi",
            timestamp=datetime.now(timezone.utc),
            yes_bid=price - 0.01,
            yes_ask=price,
            no_bid=0.0,
            no_ask=0.0,
            last_price=price,
            volume=0.0,
        ))
    # Now price rises above mean - should return None
    intents = await s.on_market_data(make_event(0.60))
    assert intents is None
