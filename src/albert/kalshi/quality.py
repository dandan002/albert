from albert.kalshi.schemas import (
    KalshiMarketRow,
    KalshiOrderbookObservationRow,
    KalshiOutcomeRow,
    KalshiPriceObservationRow,
)


def assess_market_quality(
    *,
    markets: list[KalshiMarketRow],
    outcomes: list[KalshiOutcomeRow],
    price_observations: list[KalshiPriceObservationRow],
    orderbook_observations: list[KalshiOrderbookObservationRow],
) -> dict[str, int]:
    raise NotImplementedError
