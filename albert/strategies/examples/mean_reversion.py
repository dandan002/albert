from collections import deque
from albert.strategies.base import BaseStrategy
from albert.events import MarketDataEvent, OrderIntent


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy: buy YES when price is below moving average, sell when above.
    Edge = deviation from running mean of last N prices.
    """

    def __init__(self, strategy_id: str, config: dict) -> None:
        super().__init__(strategy_id, config)
        self._window_size = config.get("window_size", 20)
        self._min_edge = config.get("min_edge", 0.03)
        self._prices: deque[float] = deque(maxlen=self._window_size)

    async def on_market_data(self, event: MarketDataEvent) -> list[OrderIntent] | None:
        edge = self.estimate_edge(event)
        if edge < self._min_edge:
            return None
        return [OrderIntent(
            market_id=event.market_id,
            strategy_id=self.id,
            side="yes" if edge > 0 else "no",
            edge=abs(edge),
            confidence=min(1.0, abs(edge) / 0.2),
        )]

    def estimate_edge(self, event: MarketDataEvent) -> float:
        price = event.yes_ask
        if price <= 0:
            return 0.0

        self._prices.append(price)

        if len(self._prices) < self._window_size // 2:
            return 0.0

        mean = sum(self._prices) / len(self._prices)
        return mean - price