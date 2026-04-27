import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine

from albert.events import OrderIntent, FillEvent

logger = logging.getLogger(__name__)


class ExchangeAdapter(ABC):
    _MAX_RETRIES = 3

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

    async def _request_with_retry(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Execute an async request function with exponential backoff retry logic."""
        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                logger.warning(
                    "%s request attempt %d failed: %s",
                    self.__class__.__name__, attempt + 1, e
                )
                if attempt < self._MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)

        if last_exc:
            raise last_exc
        raise RuntimeError("Request failed without exception")
