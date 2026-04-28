import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from albert.kalshi.normalize import (
    build_quality_summary,
    normalize_candlesticks_page,
    normalize_markets_page,
    normalize_orderbooks_page,
    normalize_trades_page,
)
from albert.kalshi.quality import assess_market_quality
from albert.kalshi.schemas import (
    TABLE_MARKETS,
    TABLE_ORDERBOOK_OBSERVATIONS,
    TABLE_OUTCOMES,
    TABLE_PRICE_OBSERVATIONS,
    TABLE_QUALITY_SUMMARY,
    TABLE_SKIPPED_MARKETS,
    TABLE_TRADES,
    KalshiMarketRow,
)


FIXTURES = Path(__file__).parent / "fixtures" / "kalshi"
UTC = timezone.utc


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _raw_path(name: str) -> Path:
    return FIXTURES / name


def test_normalized_row_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="unexpected_field"):
        KalshiMarketRow(
            market_ticker="KXBTC15M-26APR011200",
            event_ticker="KXBTC15M-26APR0112",
            series_ticker="KXBTC15M",
            title="Bitcoin above $100,000 on Apr 1 at 12:00?",
            subtitle="15 minute Bitcoin range",
            status="settled",
            open_time_utc=datetime(2026, 4, 1, 11, 45, tzinfo=UTC),
            close_time_utc=datetime(2026, 4, 1, 12, 0, tzinfo=UTC),
            settlement_time_utc=datetime(2026, 4, 1, 12, 1, tzinfo=UTC),
            expiration_time_utc=datetime(2026, 4, 1, 12, 0, tzinfo=UTC),
            can_close_early=False,
            price_level_structure="binary",
            fractional_trading_enabled=True,
            source_endpoint="live_markets",
            source_raw_path=_raw_path("live_markets_page.json"),
            unexpected_field="must be rejected",
        )


def test_normalizes_markets_and_outcomes_from_live_and_historical_pages() -> None:
    live_markets, live_outcomes, live_skipped = normalize_markets_page(
        _fixture("live_markets_page.json"),
        source_endpoint="live_markets",
        source_raw_path=_raw_path("live_markets_page.json"),
    )
    historical_markets, historical_outcomes, historical_skipped = normalize_markets_page(
        _fixture("historical_markets_page.json"),
        source_endpoint="historical_markets",
        source_raw_path=_raw_path("historical_markets_page.json"),
    )

    markets = [*live_markets, *historical_markets]
    outcomes = [*live_outcomes, *historical_outcomes]

    assert TABLE_MARKETS == "kalshi_markets"
    assert TABLE_OUTCOMES == "kalshi_outcomes"
    assert not live_skipped
    assert not historical_skipped
    assert {market.series_ticker for market in markets} == {"KXBTC15M"}
    assert all(market.market_ticker for market in markets)
    assert all(market.event_ticker for market in markets)
    assert all(market.source_endpoint in {"live_markets", "historical_markets"} for market in markets)
    assert all(market.source_raw_path.name.endswith(".json") for market in markets)
    assert all(market.close_time_utc.tzinfo is not None for market in markets)
    assert any(outcome.result is not None for outcome in outcomes)
    assert any(outcome.settlement_value_dollars is not None for outcome in outcomes)
    assert any(outcome.settlement_ts is not None for outcome in outcomes)
    assert all(hasattr(outcome, "outcome_missing") for outcome in outcomes)


def test_records_ambiguous_btc_markets_as_skipped() -> None:
    markets, outcomes, skipped = normalize_markets_page(
        _fixture("ambiguous_btc_market.json"),
        source_endpoint="live_markets",
        source_raw_path=_raw_path("ambiguous_btc_market.json"),
    )

    assert TABLE_SKIPPED_MARKETS == "kalshi_skipped_markets"
    assert markets == []
    assert outcomes == []
    assert skipped
    assert all(row.skip_reason for row in skipped)
    assert all(row.series_ticker != "KXBTC15M" or row.skip_reason for row in skipped)


def test_normalizes_live_and_historical_candlesticks() -> None:
    live_rows = normalize_candlesticks_page(
        _fixture("live_candlesticks.json"),
        market_ticker="KXBTC15M-26APR011200",
        series_ticker="KXBTC15M",
        source_endpoint="live_candlesticks",
        source_raw_path=_raw_path("live_candlesticks.json"),
        period_interval_minutes=1,
    )
    historical_rows = normalize_candlesticks_page(
        _fixture("historical_candlesticks.json"),
        market_ticker="KXBTC15M-26MAR011200",
        series_ticker="KXBTC15M",
        source_endpoint="historical_candlesticks",
        source_raw_path=_raw_path("historical_candlesticks.json"),
        period_interval_minutes=1,
    )

    rows = [*live_rows, *historical_rows]

    assert TABLE_PRICE_OBSERVATIONS == "kalshi_price_observations"
    assert {row.source_endpoint for row in rows} == {
        "live_candlesticks",
        "historical_candlesticks",
    }
    assert all(row.yes_bid_open is None or isinstance(row.yes_bid_open, Decimal) for row in rows)
    assert all(not isinstance(row.yes_bid_open, float) for row in rows)
    assert all(hasattr(row, "yes_ask_close") for row in rows)
    assert all(hasattr(row, "last_price_close") for row in rows)


def test_normalizes_trade_pages_with_source_endpoint() -> None:
    live_rows = normalize_trades_page(
        _fixture("live_trades_page.json"),
        source_endpoint="live_trades",
        source_raw_path=_raw_path("live_trades_page.json"),
    )
    historical_rows = normalize_trades_page(
        _fixture("historical_trades_page.json"),
        source_endpoint="historical_trades",
        source_raw_path=_raw_path("historical_trades_page.json"),
    )

    rows = [*live_rows, *historical_rows]

    assert TABLE_TRADES == "kalshi_trades"
    assert {row.source_endpoint for row in rows} == {"live_trades", "historical_trades"}
    assert all(row.trade_id for row in rows)
    assert all(row.count_contracts is None or isinstance(row.count_contracts, Decimal) for row in rows)
    assert all(not isinstance(row.count_contracts, float) for row in rows)


def test_normalizes_orderbook_bid_levels_and_implied_asks() -> None:
    rows = normalize_orderbooks_page(
        _fixture("orderbooks.json"),
        source_endpoint="live_orderbooks",
        source_raw_path=_raw_path("orderbooks.json"),
        snapshot_time=datetime(2026, 4, 1, 12, tzinfo=UTC),
    )

    assert TABLE_ORDERBOOK_OBSERVATIONS == "kalshi_orderbook_observations"
    assert {row.side for row in rows} >= {"yes", "no"}
    assert all(row.bid_price is None or isinstance(row.bid_price, Decimal) for row in rows)
    assert any(row.implied_ask_price is not None for row in rows)
    assert all(not isinstance(row.implied_ask_price, float) for row in rows)


def test_missing_historical_orderbooks_are_quality_flags_not_synthetic_depth() -> None:
    markets, outcomes, _skipped = normalize_markets_page(
        _fixture("missing_history_market.json"),
        source_endpoint="historical_markets",
        source_raw_path=_raw_path("missing_history_market.json"),
    )
    quality_counts = assess_market_quality(
        markets=markets,
        outcomes=outcomes,
        price_observations=[],
        orderbook_observations=[],
    )
    quality_row = build_quality_summary(
        run_id="test-run",
        markets=markets,
        skipped_markets=[],
        outcomes=outcomes,
        price_observations=[],
        orderbook_observations=[],
        historical_window_count=1,
        live_window_count=0,
    )

    assert TABLE_QUALITY_SUMMARY == "kalshi_quality_summary"
    assert quality_counts["missing_orderbook_count"] == len(markets)
    assert quality_counts["missing_price_history_count"] == len(markets)
    assert quality_row.missing_orderbook_count == len(markets)
