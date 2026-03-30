from albert.strategies.base import BaseStrategy
from albert.events import MarketDataEvent, OrderIntent


class MomentumV1(BaseStrategy):
    """
    Example strategy: buy YES when the ask is below 0.5 by at least min_edge.
    Edge = 0.5 - yes_ask (assumes true probability is 0.5 for any event near even odds).
    """

    async def on_market_data(self, event: MarketDataEvent) -> list[OrderIntent] | None:
        min_edge = self.config.get("min_edge", 0.05)
        edge = self.estimate_edge(event)
        if edge < min_edge:
            return None
        return [OrderIntent(
            market_id=event.market_id,
            strategy_id=self.id,
            side="yes",
            edge=edge,
            confidence=min(1.0, edge / 0.2),
        )]

    def estimate_edge(self, event: MarketDataEvent) -> float:
        if event.yes_ask <= 0:
            return 0.0
        return max(0.0, 0.5 - event.yes_ask)
