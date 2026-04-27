import asyncio
import logging
from abc import ABC, abstractmethod

from albert.events import EventBus, MarketDataEvent

logger = logging.getLogger(__name__)


class BaseIngestor(ABC):
    def __init__(self, bus: EventBus, reconnect_delay: float = 5.0, shutdown_event: asyncio.Event | None = None) -> None:
        self._bus = bus
        self._reconnect_delay = reconnect_delay
        self._shutdown_event = shutdown_event
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def run(self) -> None:
        """Connect and stream indefinitely, reconnecting on failure."""
        while True:
            if self._shutdown_event and self._shutdown_event.is_set():
                return
            try:
                await self._connect_and_stream()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._connected = False
                logger.exception(
                    "%s disconnected, reconnecting in %.1fs",
                    self.__class__.__name__, self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)

    @abstractmethod
    async def _connect_and_stream(self) -> None: ...

    @abstractmethod
    def _normalize(self, raw: dict) -> MarketDataEvent | None: ...
