from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


TABLE_MARKETS = "kalshi_markets"
TABLE_PRICE_OBSERVATIONS = "kalshi_price_observations"
TABLE_TRADES = "kalshi_trades"
TABLE_ORDERBOOK_OBSERVATIONS = "kalshi_orderbook_observations"
TABLE_OUTCOMES = "kalshi_outcomes"
TABLE_SKIPPED_MARKETS = "kalshi_skipped_markets"
TABLE_QUALITY_SUMMARY = "kalshi_quality_summary"


def _require_aware_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


def _decimal_from_source(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise ValueError("boolean values are not valid decimals")
    return Decimal(str(value))


class StrictRow(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProvenanceRow(StrictRow):
    source_endpoint: str
    source_raw_path: Path


class KalshiMarketRow(ProvenanceRow):
    market_ticker: str
    event_ticker: str
    series_ticker: str
    title: str | None = None
    subtitle: str | None = None
    status: str | None = None
    open_time_utc: datetime | None = None
    close_time_utc: datetime | None = None
    settlement_time_utc: datetime | None = None
    expiration_time_utc: datetime | None = None
    can_close_early: bool | None = None
    price_level_structure: str | None = None
    fractional_trading_enabled: bool | None = None

    @field_validator(
        "open_time_utc",
        "close_time_utc",
        "settlement_time_utc",
        "expiration_time_utc",
    )
    @classmethod
    def normalize_datetimes(cls, value: datetime | None) -> datetime | None:
        return _require_aware_utc(value)


class KalshiPriceObservationRow(ProvenanceRow):
    market_ticker: str
    series_ticker: str
    observed_at_utc: datetime
    period_interval_minutes: int
    yes_bid_open: Decimal | None = None
    yes_bid_high: Decimal | None = None
    yes_bid_low: Decimal | None = None
    yes_bid_close: Decimal | None = None
    yes_ask_open: Decimal | None = None
    yes_ask_high: Decimal | None = None
    yes_ask_low: Decimal | None = None
    yes_ask_close: Decimal | None = None
    last_price_open: Decimal | None = None
    last_price_close: Decimal | None = None
    volume_contracts: Decimal | None = None
    open_interest_contracts: Decimal | None = None

    @field_validator("observed_at_utc")
    @classmethod
    def normalize_observed_at(cls, value: datetime) -> datetime:
        normalized = _require_aware_utc(value)
        if normalized is None:
            raise ValueError("observed_at_utc is required")
        return normalized

    @field_validator(
        "yes_bid_open",
        "yes_bid_high",
        "yes_bid_low",
        "yes_bid_close",
        "yes_ask_open",
        "yes_ask_high",
        "yes_ask_low",
        "yes_ask_close",
        "last_price_open",
        "last_price_close",
        "volume_contracts",
        "open_interest_contracts",
        mode="before",
    )
    @classmethod
    def parse_decimal_fields(cls, value: Any) -> Decimal | None:
        return _decimal_from_source(value)


class KalshiTradeRow(ProvenanceRow):
    trade_id: str
    market_ticker: str
    created_time_utc: datetime
    count_contracts: Decimal | None = None
    yes_price: Decimal | None = None
    no_price: Decimal | None = None
    taker_side: str | None = None

    @field_validator("created_time_utc")
    @classmethod
    def normalize_created_time(cls, value: datetime) -> datetime:
        normalized = _require_aware_utc(value)
        if normalized is None:
            raise ValueError("created_time_utc is required")
        return normalized

    @field_validator("count_contracts", "yes_price", "no_price", mode="before")
    @classmethod
    def parse_decimal_fields(cls, value: Any) -> Decimal | None:
        return _decimal_from_source(value)


class KalshiOrderbookObservationRow(ProvenanceRow):
    market_ticker: str
    snapshot_time_utc: datetime
    side: Literal["yes", "no"]
    level_index: int
    bid_price: Decimal | None = None
    bid_count_contracts: Decimal | None = None
    implied_ask_price: Decimal | None = None

    @field_validator("snapshot_time_utc")
    @classmethod
    def normalize_snapshot_time(cls, value: datetime) -> datetime:
        normalized = _require_aware_utc(value)
        if normalized is None:
            raise ValueError("snapshot_time_utc is required")
        return normalized

    @field_validator("bid_price", "bid_count_contracts", "implied_ask_price", mode="before")
    @classmethod
    def parse_decimal_fields(cls, value: Any) -> Decimal | None:
        return _decimal_from_source(value)


class KalshiOutcomeRow(ProvenanceRow):
    market_ticker: str
    settlement_time_utc: datetime | None = None
    outcome: str | None = None
    result: str | None = None
    settlement_value_dollars: Decimal | None = None
    settlement_ts: datetime | None = None
    outcome_missing: bool

    @field_validator("settlement_time_utc", "settlement_ts")
    @classmethod
    def normalize_datetimes(cls, value: datetime | None) -> datetime | None:
        return _require_aware_utc(value)

    @field_validator("settlement_value_dollars", mode="before")
    @classmethod
    def parse_decimal_fields(cls, value: Any) -> Decimal | None:
        return _decimal_from_source(value)


class KalshiSkippedMarketRow(ProvenanceRow):
    market_ticker: str | None = None
    event_ticker: str | None = None
    series_ticker: str | None = None
    skip_reason: str
    raw_title: str | None = None
    raw_subtitle: str | None = None


class KalshiQualitySummaryRow(StrictRow):
    run_id: str
    market_count: int
    eligible_market_count: int
    skipped_market_count: int
    missing_outcome_count: int
    missing_price_history_count: int
    missing_orderbook_count: int
    historical_window_count: int
    live_window_count: int


__all__ = [
    "TABLE_MARKETS",
    "TABLE_ORDERBOOK_OBSERVATIONS",
    "TABLE_OUTCOMES",
    "TABLE_PRICE_OBSERVATIONS",
    "TABLE_QUALITY_SUMMARY",
    "TABLE_SKIPPED_MARKETS",
    "TABLE_TRADES",
    "KalshiMarketRow",
    "KalshiOrderbookObservationRow",
    "KalshiOutcomeRow",
    "KalshiPriceObservationRow",
    "KalshiQualitySummaryRow",
    "KalshiSkippedMarketRow",
    "KalshiTradeRow",
]
