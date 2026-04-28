import json
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from albert.cache import raw_cache_path
from albert.manifest import RunManifest, SourceMetadata, config_hash_from_mapping
from albert.kalshi.client import FetchedPage
from albert.kalshi.pipeline import _write_normalized_outputs, run_kalshi_discovery
from albert.kalshi.schemas import (
    TABLE_MARKETS,
    TABLE_ORDERBOOK_OBSERVATIONS,
    TABLE_OUTCOMES,
    TABLE_PRICE_OBSERVATIONS,
    TABLE_QUALITY_SUMMARY,
    TABLE_SKIPPED_MARKETS,
    TABLE_TRADES,
)


FIXTURES = Path(__file__).parent / "fixtures" / "kalshi"
UTC = timezone.utc


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FixtureKalshiClient:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.fetched_at = datetime(2026, 4, 28, 12, tzinfo=UTC)

    def _page(
        self,
        *,
        endpoint_segment: str,
        path: str,
        parameters: dict[str, object],
        covered_start: datetime,
        covered_end: datetime,
        payload: dict[str, object],
    ) -> FetchedPage:
        raw_path = raw_cache_path(self.cache_root, "kalshi", endpoint_segment, parameters)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return FetchedPage(
            source="kalshi",
            endpoint=path,
            endpoint_segment=endpoint_segment,
            parameters=parameters,
            fetched_at=self.fetched_at,
            covered_start=covered_start,
            covered_end=covered_end,
            raw_path=raw_path,
            payload=payload,
        )

    def get_json_page(
        self,
        *,
        endpoint_segment: str,
        path: str,
        parameters: dict[str, object],
        covered_start: datetime,
        covered_end: datetime,
    ) -> FetchedPage:
        payloads = {
            "historical_cutoff": {
                "market_settled_ts": "2026-04-01T12:00:00Z",
                "trades_created_ts": "2026-04-01T12:00:00Z",
            },
            "live_markets": _fixture("live_markets_page.json"),
            "historical_markets": _fixture("historical_markets_page.json"),
            "live_candlesticks": _fixture("live_candlesticks.json"),
            "historical_candlesticks": _fixture("historical_candlesticks.json"),
            "live_trades": _fixture("live_trades_page.json"),
            "historical_trades": _fixture("historical_trades_page.json"),
            "live_orderbooks": _fixture("orderbooks.json"),
        }
        return self._page(
            endpoint_segment=endpoint_segment,
            path=path,
            parameters=dict(parameters),
            covered_start=covered_start,
            covered_end=covered_end,
            payload=payloads[endpoint_segment],
        )

    def iter_cursor_pages(
        self,
        *,
        endpoint_segment: str,
        path: str,
        parameters: dict[str, object],
        collection_key: str,
        covered_start: datetime,
        covered_end: datetime,
    ):
        if collection_key == "markets":
            yield self.get_json_page(
                endpoint_segment=endpoint_segment,
                path=path,
                parameters=parameters,
                covered_start=covered_start,
                covered_end=covered_end,
            )
        elif collection_key == "trades":
            # First page: original parameters, payload from fixture with cursor="trade-page-2"
            base_payload = dict(self.get_json_page(
                endpoint_segment=endpoint_segment,
                path=path,
                parameters=parameters,
                covered_start=covered_start,
                covered_end=covered_end,
            ).payload)
            first_page_payload = dict(base_payload)
            first_page_payload["cursor"] = "trade-page-2"

            first_page = self._page(
                endpoint_segment=endpoint_segment,
                path=path,
                parameters=dict(parameters),
                covered_start=covered_start,
                covered_end=covered_end,
                payload=first_page_payload,
            )
            yield first_page

            # Second page: parameters with cursor="trade-page-2", payload with cursor=None
            # and one trade whose trade_id is "TRADE-PAGE-2"
            second_page_payload = {
                "cursor": None,
                "trades": [
                    {
                        "trade_id": "TRADE-PAGE-2",
                        "ticker": "KXBTC15M-26APR281900",
                        "created_time": "2026-04-28T19:05:00.000Z",
                        "count": 3,
                        "yes_price": "0.50",
                        "no_price": "0.50",
                        "taker_side": "yes",
                    }
                ],
            }
            second_page_params = dict(parameters)
            second_page_params["cursor"] = "trade-page-2"

            second_page = self._page(
                endpoint_segment=endpoint_segment,
                path=path,
                parameters=second_page_params,
                covered_start=covered_start,
                covered_end=covered_end,
                payload=second_page_payload,
            )
            yield second_page
        else:
            raise ValueError(f"unexpected collection_key: {collection_key}")


def test_discover_writes_manifest_sources_and_outputs(
    tmp_path: Path,
    example_config_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    run_root = tmp_path / "runs"
    config_hash = config_hash_from_mapping(
        {"cache_dir": str(cache_root), "run_dir": str(run_root), "timezone": "UTC"}
    )

    result = run_kalshi_discovery(
        date(2026, 4, 1),
        date(2026, 4, 2),
        cache_root=cache_root,
        run_root=run_root,
        config_path=example_config_path,
        config_hash=config_hash,
        series_ticker="KXBTC15M",
        run_id="test-run",
        client=FixtureKalshiClient(cache_root),
    )

    manifest = RunManifest.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )
    output_tables = {output.table for output in manifest.outputs}

    assert manifest.sources
    assert all(isinstance(source, SourceMetadata) for source in manifest.sources)
    assert all(source.source == "kalshi" for source in manifest.sources)
    assert all(source.covered_start.tzinfo is not None for source in manifest.sources)
    assert all(source.covered_end.tzinfo is not None for source in manifest.sources)
    assert all("api_key" not in source.parameters for source in manifest.sources)
    assert output_tables == {
        TABLE_MARKETS,
        TABLE_PRICE_OBSERVATIONS,
        TABLE_TRADES,
        TABLE_ORDERBOOK_OBSERVATIONS,
        TABLE_OUTCOMES,
        TABLE_SKIPPED_MARKETS,
        TABLE_QUALITY_SUMMARY,
    }
    assert set(result.output_paths) == output_tables
    assert all(path.exists() for path in result.output_paths.values())


def test_discover_paginates_trade_pages_and_records_each_source(
    tmp_path: Path,
    example_config_path: Path,
) -> None:
    """Gap G-01: trade retrieval must use cursor pagination and append every
    trade page to manifest sources."""
    cache_root = tmp_path / "cache"
    run_root = tmp_path / "runs"
    config_hash = config_hash_from_mapping(
        {"cache_dir": str(cache_root), "run_dir": str(run_root), "timezone": "UTC"}
    )

    result = run_kalshi_discovery(
        date(2026, 4, 1),
        date(2026, 4, 2),
        cache_root=cache_root,
        run_root=run_root,
        config_path=example_config_path,
        config_hash=config_hash,
        series_ticker="KXBTC15M",
        run_id="test-pagination",
        client=FixtureKalshiClient(cache_root),
    )

    manifest = RunManifest.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )

    # At least one manifest source must be a trade endpoint with cursor="trade-page-2"
    trade_cursor_sources = [
        source
        for source in manifest.sources
        if source.endpoint in ("/markets/trades", "/historical/trades")
        and source.parameters.get("cursor") == "trade-page-2"
    ]
    assert len(trade_cursor_sources) >= 1, (
        f"expected at least one trade source with cursor='trade-page-2', "
        f"found endpoints: {[s.endpoint for s in manifest.sources]}"
    )

    # Trades Parquet must contain the TRADE-PAGE-2 trade from the second page
    trades_path = result.output_paths[TABLE_TRADES]
    trades_df = pl.read_parquet(trades_path)
    trade_ids = trades_df["trade_id"].to_list()
    assert "TRADE-PAGE-2" in trade_ids, (
        f"expected TRADE-PAGE-2 in trade_id column, found: {trade_ids}"
    )

    # Manifest must have more sources than the pre-fix single-page count of 7
    assert len(manifest.sources) > 7, (
        f"expected more than 7 manifest sources for paginated trades, "
        f"got {len(manifest.sources)}"
    )


def test_discover_writes_empty_parquet_tables_with_declared_columns(
    tmp_path: Path,
) -> None:
    """Gap G-02: empty normalized tables must keep declared columns in their
    Parquet files."""
    cache_root = tmp_path / "cache"
    start = datetime(2026, 4, 1, tzinfo=UTC)
    end = datetime(2026, 4, 2, tzinfo=UTC)

    # Provide empty row lists for the tables we want to check
    table_rows: dict[str, list] = {
        TABLE_MARKETS: [],
        TABLE_PRICE_OBSERVATIONS: [],
        TABLE_TRADES: [],
        TABLE_ORDERBOOK_OBSERVATIONS: [],
        TABLE_OUTCOMES: [],
        TABLE_SKIPPED_MARKETS: [],
        TABLE_QUALITY_SUMMARY: [],
    }

    output_paths = _write_normalized_outputs(
        cache_root=cache_root,
        run_id="empty-test",
        series_ticker="KXBTC15M",
        start=start,
        end=end,
        table_rows=table_rows,
    )

    # Trades table must retain declared columns when empty
    trades_df = pl.read_parquet(output_paths[TABLE_TRADES])
    assert trades_df.height == 0, f"expected 0 rows, got {trades_df.height}"
    assert set(trades_df.columns) >= {
        "source_endpoint",
        "source_raw_path",
        "trade_id",
        "market_ticker",
        "created_time_utc",
        "count_contracts",
        "yes_price",
        "no_price",
        "taker_side",
    }, f"trades columns missing: {set(trades_df.columns)}"

    # Orderbook observations table must retain declared columns when empty
    orderbooks_df = pl.read_parquet(output_paths[TABLE_ORDERBOOK_OBSERVATIONS])
    assert orderbooks_df.height == 0, f"expected 0 rows, got {orderbooks_df.height}"
    assert set(orderbooks_df.columns) >= {
        "source_endpoint",
        "source_raw_path",
        "market_ticker",
        "snapshot_time_utc",
        "side",
        "level_index",
        "bid_price",
        "bid_count_contracts",
        "implied_ask_price",
    }, f"orderbooks columns missing: {set(orderbooks_df.columns)}"

    # Skipped markets table must retain declared columns when empty
    skipped_df = pl.read_parquet(output_paths[TABLE_SKIPPED_MARKETS])
    assert skipped_df.height == 0, f"expected 0 rows, got {skipped_df.height}"
    assert set(skipped_df.columns) >= {
        "source_endpoint",
        "source_raw_path",
        "market_ticker",
        "event_ticker",
        "series_ticker",
        "skip_reason",
        "raw_title",
        "raw_subtitle",
    }, f"skipped_markets columns missing: {set(skipped_df.columns)}"
