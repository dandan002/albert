from albert.strategies.base import BaseStrategy
from albert.events import MarketDataEvent, OrderIntent
from albert.execution.kelly import kelly_size


class MomentumV1(BaseStrategy):
    """
    Momentum strategy: buy YES when the ask is below 0.5 by at least min_edge.
    Uses Kelly sizing for position sizing.
    """

    def __init__(self, strategy_id: str, config: dict) -> None:
        super().__init__(strategy_id, config)
        self._kelly_fraction = config.get("kelly_fraction", 0.25)
        self._max_position_usd = config.get("max_position_usd", 1000)
        self._bankroll = config.get("bankroll", 10000)

    async def on_market_data(self, event: MarketDataEvent) -> list[OrderIntent] | None:
        min_edge = self.config.get("min_edge", 0.05)
        edge = self.estimate_edge(event)
        if edge < min_edge:
            return None

        size = kelly_size(
            edge=edge,
            ask_price=event.yes_ask,
            bankroll=self._bankroll,
            kelly_fraction=self._kelly_fraction,
            confidence=min(1.0, edge / 0.2),
            max_position_usd=self._max_position_usd,
        )
        if size <= 0:
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
