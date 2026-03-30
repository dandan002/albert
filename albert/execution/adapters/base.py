from abc import ABC, abstractmethod
from albert.events import OrderIntent, FillEvent


class ExchangeAdapter(ABC):
    @abstractmethod
    async def place_order(self, intent: OrderIntent, contracts: int, price: float) -> FillEvent:
        """Place a limit order. Returns a FillEvent on success."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> None:
        """Cancel an open order by exchange order ID."""
        ...

    @abstractmethod
    async def get_bankroll(self) -> float:
        """Return available balance in USD."""
        ...
