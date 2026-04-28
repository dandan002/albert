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
    outcome_by_ticker = {outcome.market_ticker: outcome for outcome in outcomes}
    price_tickers = {row.market_ticker for row in price_observations}
    orderbook_tickers = {row.market_ticker for row in orderbook_observations}

    missing_outcome_count = 0
    missing_price_history_count = 0
    missing_orderbook_count = 0
    for market in markets:
        outcome = outcome_by_ticker.get(market.market_ticker)
        if outcome is None or outcome.outcome_missing:
            missing_outcome_count += 1
        if market.market_ticker not in price_tickers:
            missing_price_history_count += 1
        if market.market_ticker not in orderbook_tickers:
            missing_orderbook_count += 1

    return {
        "missing_outcome_count": missing_outcome_count,
        "missing_price_history_count": missing_price_history_count,
        "missing_orderbook_count": missing_orderbook_count,
    }


def quality_summary_text(result: object) -> str:
    return (
        f"missing_outcome_count={getattr(result, 'missing_outcome_count')} "
        f"missing_price_history_count={getattr(result, 'missing_price_history_count')} "
        f"missing_orderbook_count={getattr(result, 'missing_orderbook_count')}"
    )


__all__ = ["assess_market_quality", "quality_summary_text"]
