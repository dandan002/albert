from datetime import datetime
from pathlib import Path
from typing import Any

from albert.kalshi.schemas import (
    KalshiMarketRow,
    KalshiOrderbookObservationRow,
    KalshiOutcomeRow,
    KalshiPriceObservationRow,
    KalshiQualitySummaryRow,
    KalshiSkippedMarketRow,
    KalshiTradeRow,
)


def normalize_markets_page(
    payload: dict[str, Any],
    *,
    source_endpoint: str,
    source_raw_path: Path,
    series_ticker: str = "KXBTC15M",
) -> tuple[list[KalshiMarketRow], list[KalshiOutcomeRow], list[KalshiSkippedMarketRow]]:
    raise NotImplementedError


def normalize_candlesticks_page(
    payload: dict[str, Any],
    *,
    market_ticker: str,
    series_ticker: str,
    source_endpoint: str,
    source_raw_path: Path,
    period_interval_minutes: int = 1,
) -> list[KalshiPriceObservationRow]:
    raise NotImplementedError


def normalize_trades_page(
    payload: dict[str, Any],
    *,
    source_endpoint: str,
    source_raw_path: Path,
) -> list[KalshiTradeRow]:
    raise NotImplementedError


def normalize_orderbooks_page(
    payload: dict[str, Any],
    *,
    source_endpoint: str,
    source_raw_path: Path,
    snapshot_time: datetime,
) -> list[KalshiOrderbookObservationRow]:
    raise NotImplementedError


def build_quality_summary(
    run_id: str,
    markets: list[KalshiMarketRow],
    skipped_markets: list[KalshiSkippedMarketRow],
    outcomes: list[KalshiOutcomeRow],
    price_observations: list[KalshiPriceObservationRow],
    orderbook_observations: list[KalshiOrderbookObservationRow],
    historical_window_count: int,
    live_window_count: int,
) -> KalshiQualitySummaryRow:
    raise NotImplementedError
