from abc import ABC, abstractmethod
from albert.events import MarketDataEvent, OrderIntent


class BaseStrategy(ABC):
    def __init__(self, strategy_id: str, config: dict) -> None:
        self.id = strategy_id
        self.config = config

    @abstractmethod
    async def on_market_data(self, event: MarketDataEvent) -> list[OrderIntent] | None:
        """Called on every orderbook update. Return order intents or None."""
        ...

    @abstractmethod
    def estimate_edge(self, event: MarketDataEvent) -> float:
        """Return estimated probability edge (0.0–1.0)."""
        ...
