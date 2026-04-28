from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable

import polars as pl
from pydantic import BaseModel

from albert.cache import normalized_cache_path, run_manifest_path
from albert.kalshi.client import FetchedPage, KalshiClient
from albert.kalshi.normalize import (
    build_quality_summary,
    normalize_candlesticks_page,
    normalize_markets_page,
    normalize_orderbooks_page,
    normalize_trades_page,
)
from albert.kalshi.routing import (
    EndpointWindow,
    parse_historical_cutoff,
    plan_candlestick_windows,
    plan_market_windows,
    plan_trade_windows,
)
from albert.kalshi.schemas import (
    TABLE_MARKETS,
    TABLE_ORDERBOOK_OBSERVATIONS,
    TABLE_OUTCOMES,
    TABLE_PRICE_OBSERVATIONS,
    TABLE_QUALITY_SUMMARY,
    TABLE_SKIPPED_MARKETS,
    TABLE_TRADES,
    KalshiMarketRow,
    KalshiOrderbookObservationRow,
    KalshiOutcomeRow,
    KalshiPriceObservationRow,
    KalshiSkippedMarketRow,
    KalshiTradeRow,
)
from albert.manifest import OutputMetadata, RunManifest, SourceMetadata, write_manifest


UTC = timezone.utc
TABLES = (
    TABLE_MARKETS,
    TABLE_PRICE_OBSERVATIONS,
    TABLE_TRADES,
    TABLE_ORDERBOOK_OBSERVATIONS,
    TABLE_OUTCOMES,
    TABLE_SKIPPED_MARKETS,
    TABLE_QUALITY_SUMMARY,
)

TABLE_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    TABLE_MARKETS: {
        "market_ticker": pl.String,
        "event_ticker": pl.String,
        "series_ticker": pl.String,
        "title": pl.String,
        "subtitle": pl.String,
        "status": pl.String,
        "open_time_utc": pl.String,
        "close_time_utc": pl.String,
        "settlement_time_utc": pl.String,
        "expiration_time_utc": pl.String,
        "can_close_early": pl.Boolean,
        "price_level_structure": pl.String,
        "fractional_trading_enabled": pl.Boolean,
        "source_endpoint": pl.String,
        "source_raw_path": pl.String,
    },
    TABLE_PRICE_OBSERVATIONS: {
        "market_ticker": pl.String,
        "series_ticker": pl.String,
        "observed_at_utc": pl.String,
        "period_interval_minutes": pl.Int64,
        "yes_bid_open": pl.String,
        "yes_bid_high": pl.String,
        "yes_bid_low": pl.String,
        "yes_bid_close": pl.String,
        "yes_ask_open": pl.String,
        "yes_ask_high": pl.String,
        "yes_ask_low": pl.String,
        "yes_ask_close": pl.String,
        "last_price_open": pl.String,
        "last_price_close": pl.String,
        "volume_contracts": pl.String,
        "open_interest_contracts": pl.String,
        "source_endpoint": pl.String,
        "source_raw_path": pl.String,
    },
    TABLE_TRADES: {
        "trade_id": pl.String,
        "market_ticker": pl.String,
        "created_time_utc": pl.String,
        "count_contracts": pl.String,
        "yes_price": pl.String,
        "no_price": pl.String,
        "taker_side": pl.String,
        "source_endpoint": pl.String,
        "source_raw_path": pl.String,
    },
    TABLE_ORDERBOOK_OBSERVATIONS: {
        "market_ticker": pl.String,
        "snapshot_time_utc": pl.String,
        "side": pl.String,
        "level_index": pl.Int64,
        "bid_price": pl.String,
        "bid_count_contracts": pl.String,
        "implied_ask_price": pl.String,
        "source_endpoint": pl.String,
        "source_raw_path": pl.String,
    },
    TABLE_OUTCOMES: {
        "market_ticker": pl.String,
        "settlement_time_utc": pl.String,
        "outcome": pl.String,
        "result": pl.String,
        "settlement_value_dollars": pl.String,
        "settlement_ts": pl.String,
        "outcome_missing": pl.Boolean,
        "source_endpoint": pl.String,
        "source_raw_path": pl.String,
    },
    TABLE_SKIPPED_MARKETS: {
        "market_ticker": pl.String,
        "event_ticker": pl.String,
        "series_ticker": pl.String,
        "skip_reason": pl.String,
        "raw_title": pl.String,
        "raw_subtitle": pl.String,
        "source_endpoint": pl.String,
        "source_raw_path": pl.String,
    },
    TABLE_QUALITY_SUMMARY: {
        "run_id": pl.String,
        "market_count": pl.Int64,
        "eligible_market_count": pl.Int64,
        "skipped_market_count": pl.Int64,
        "missing_outcome_count": pl.Int64,
        "missing_price_history_count": pl.Int64,
        "missing_orderbook_count": pl.Int64,
        "historical_window_count": pl.Int64,
        "live_window_count": pl.Int64,
    },
}


@dataclass(frozen=True)
class KalshiDiscoveryResult:
    run_id: str
    manifest_path: Path
    output_paths: dict[str, Path]
    market_count: int
    eligible_market_count: int
    skipped_market_count: int
    missing_outcome_count: int
    missing_price_history_count: int
    missing_orderbook_count: int


def run_kalshi_discovery(
    start_date: date,
    end_date: date,
    *,
    cache_root: Path,
    run_root: Path,
    config_path: Path,
    config_hash: str,
    series_ticker: str = "KXBTC15M",
    run_id: str,
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
    client: KalshiClient | None = None,
) -> KalshiDiscoveryResult:
    start = _date_start_utc(start_date)
    end = _date_start_utc(end_date)
    if end <= start:
        raise ValueError("end_date must be after start_date")

    kalshi_client = client or KalshiClient(cache_root=cache_root, base_url=base_url)
    pages: list[FetchedPage] = []

    cutoff_page = kalshi_client.get_json_page(
        endpoint_segment="historical_cutoff",
        path="/historical/cutoff",
        parameters={},
        covered_start=start,
        covered_end=end,
    )
    pages.append(cutoff_page)
    cutoff = parse_historical_cutoff(cutoff_page.payload)

    market_rows: list[KalshiMarketRow] = []
    outcome_rows: list[KalshiOutcomeRow] = []
    skipped_rows: list[KalshiSkippedMarketRow] = []
    market_windows = plan_market_windows(start, end, cutoff)
    for window in market_windows:
        parameters = _window_parameters(
            window,
            {"series_ticker": series_ticker, "limit": 1000},
        )
        for page in kalshi_client.iter_cursor_pages(
            endpoint_segment=window.endpoint_segment,
            path=window.path,
            parameters=parameters,
            collection_key="markets",
            covered_start=window.start,
            covered_end=window.end,
        ):
            pages.append(page)
            markets, outcomes, skipped = normalize_markets_page(
                page.payload,
                source_endpoint=page.endpoint_segment,
                source_raw_path=page.raw_path,
                series_ticker=series_ticker,
            )
            market_rows.extend(markets)
            outcome_rows.extend(outcomes)
            skipped_rows.extend(skipped)

    price_rows = _fetch_price_observations(
        kalshi_client,
        pages,
        market_rows,
        cutoff,
        series_ticker,
    )
    trade_rows = _fetch_trades(kalshi_client, pages, start, end, cutoff, series_ticker)
    orderbook_rows = _fetch_current_orderbooks(kalshi_client, pages, market_rows, start, end)

    historical_window_count = sum(1 for window in market_windows if window.kind == "historical")
    live_window_count = sum(1 for window in market_windows if window.kind == "live")
    quality_row = build_quality_summary(
        run_id=run_id,
        markets=market_rows,
        skipped_markets=skipped_rows,
        outcomes=outcome_rows,
        price_observations=price_rows,
        orderbook_observations=orderbook_rows,
        historical_window_count=historical_window_count,
        live_window_count=live_window_count,
    )

    table_rows: dict[str, list[BaseModel]] = {
        TABLE_MARKETS: market_rows,
        TABLE_PRICE_OBSERVATIONS: price_rows,
        TABLE_TRADES: trade_rows,
        TABLE_ORDERBOOK_OBSERVATIONS: orderbook_rows,
        TABLE_OUTCOMES: outcome_rows,
        TABLE_SKIPPED_MARKETS: skipped_rows,
        TABLE_QUALITY_SUMMARY: [quality_row],
    }
    output_paths = _write_normalized_outputs(cache_root, run_id, series_ticker, start, end, table_rows)
    manifest_path = run_manifest_path(run_root, run_id)
    manifest = RunManifest(
        run_id=run_id,
        created_at=datetime.now(UTC),
        command="albert ingest kalshi discover",
        parameters={
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "series_ticker": series_ticker,
            "run_id": run_id,
        },
        config_path=config_path,
        config_hash=config_hash,
        cache_root=cache_root,
        run_root=run_root,
        sources=[_source_metadata(page) for page in pages],
        outputs=[
            OutputMetadata(
                kind="normalized_parquet",
                source="kalshi",
                table=table_name,
                path=output_paths[table_name],
                row_count=len(rows),
            )
            for table_name, rows in table_rows.items()
        ],
    )
    write_manifest(manifest, manifest_path)

    return KalshiDiscoveryResult(
        run_id=run_id,
        manifest_path=manifest_path,
        output_paths=output_paths,
        market_count=len(market_rows),
        eligible_market_count=len(market_rows),
        skipped_market_count=len(skipped_rows),
        missing_outcome_count=quality_row.missing_outcome_count,
        missing_price_history_count=quality_row.missing_price_history_count,
        missing_orderbook_count=quality_row.missing_orderbook_count,
    )


def _date_start_utc(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=UTC)


def _epoch_seconds(value: datetime) -> int:
    return int(value.astimezone(UTC).timestamp())


def _window_parameters(
    window: EndpointWindow,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    start_name, end_name = window.parameter_names
    return {
        **(extra or {}),
        start_name: _epoch_seconds(window.start),
        end_name: _epoch_seconds(window.end),
    }


def _fetch_price_observations(
    client: KalshiClient,
    pages: list[FetchedPage],
    markets: Iterable[KalshiMarketRow],
    cutoff: object,
    series_ticker: str,
) -> list[KalshiPriceObservationRow]:
    rows: list[KalshiPriceObservationRow] = []
    for market in markets:
        if market.open_time_utc is None or market.expiration_time_utc is None:
            continue
        windows = plan_candlestick_windows(
            market.market_ticker,
            market.open_time_utc,
            market.expiration_time_utc,
            cutoff,
        )
        for window in windows:
            page = client.get_json_page(
                endpoint_segment=window.endpoint_segment,
                path=window.path,
                parameters=_window_parameters(
                    window,
                    {"period_interval_minutes": 1},
                ),
                covered_start=window.start,
                covered_end=window.end,
            )
            pages.append(page)
            rows.extend(
                normalize_candlesticks_page(
                    page.payload,
                    market_ticker=market.market_ticker,
                    series_ticker=series_ticker,
                    source_endpoint=page.endpoint_segment,
                    source_raw_path=page.raw_path,
                    period_interval_minutes=1,
                )
            )
    return rows


def _fetch_trades(
    client: KalshiClient,
    pages: list[FetchedPage],
    start: datetime,
    end: datetime,
    cutoff: object,
    series_ticker: str,
) -> list[KalshiTradeRow]:
    rows: list[KalshiTradeRow] = []
    for window in plan_trade_windows(start, end, cutoff):
        for page in client.iter_cursor_pages(
            endpoint_segment=window.endpoint_segment,
            path=window.path,
            parameters=_window_parameters(
                window,
                {"series_ticker": series_ticker, "limit": 1000},
            ),
            collection_key="trades",
            covered_start=window.start,
            covered_end=window.end,
        ):
            pages.append(page)
            rows.extend(
                normalize_trades_page(
                    page.payload,
                    source_endpoint=page.endpoint_segment,
                    source_raw_path=page.raw_path,
                )
            )
    return rows


def _fetch_current_orderbooks(
    client: KalshiClient,
    pages: list[FetchedPage],
    markets: list[KalshiMarketRow],
    start: datetime,
    end: datetime,
) -> list[KalshiOrderbookObservationRow]:
    tickers = sorted(
        {
            market.market_ticker
            for market in markets
            if market.status is None or market.status.lower() not in {"settled", "finalized"}
        }
    )
    if not tickers:
        return []
    page = client.get_json_page(
        endpoint_segment="live_orderbooks",
        path="/markets/orderbooks",
        parameters={"tickers": ",".join(tickers)},
        covered_start=start,
        covered_end=end,
    )
    pages.append(page)
    return normalize_orderbooks_page(
        page.payload,
        source_endpoint=page.endpoint_segment,
        source_raw_path=page.raw_path,
        snapshot_time=page.fetched_at,
    )


def _write_normalized_outputs(
    cache_root: Path,
    run_id: str,
    series_ticker: str,
    start: datetime,
    end: datetime,
    table_rows: dict[str, list[BaseModel]],
) -> dict[str, Path]:
    output_paths: dict[str, Path] = {}
    parameters = {
        "run_id": run_id,
        "series_ticker": series_ticker,
        "start_ts": _epoch_seconds(start),
        "end_ts": _epoch_seconds(end),
    }
    for table_name in TABLES:
        rows = table_rows[table_name]
        path = normalized_cache_path(cache_root, "kalshi", table_name, parameters)
        path.parent.mkdir(parents=True, exist_ok=True)
        _rows_dataframe(table_name, rows).write_parquet(path)
        output_paths[table_name] = path
    return output_paths


def _rows_dataframe(table_name: str, rows: list[BaseModel]) -> pl.DataFrame:
    if rows:
        return pl.DataFrame([_row_dict(row) for row in rows], schema=TABLE_SCHEMAS[table_name])
    return pl.DataFrame(schema=TABLE_SCHEMAS[table_name])


def _row_dict(row: BaseModel) -> dict[str, Any]:
    return row.model_dump(mode="json")


def _source_metadata(page: FetchedPage) -> SourceMetadata:
    return SourceMetadata(
        source=page.source,
        endpoint=page.endpoint,
        parameters=page.parameters,
        fetched_at=page.fetched_at,
        covered_start=page.covered_start,
        covered_end=page.covered_end,
        raw_path=page.raw_path,
    )


__all__ = ["KalshiDiscoveryResult", "run_kalshi_discovery"]
