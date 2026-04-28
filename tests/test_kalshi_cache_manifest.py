import json
from datetime import date, datetime, timezone
from pathlib import Path

from albert.cache import raw_cache_path
from albert.manifest import RunManifest, SourceMetadata, config_hash_from_mapping
from albert.kalshi.client import FetchedPage
from albert.kalshi.pipeline import run_kalshi_discovery
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
        assert collection_key == "markets"
        yield self.get_json_page(
            endpoint_segment=endpoint_segment,
            path=path,
            parameters=parameters,
            covered_start=covered_start,
            covered_end=covered_end,
        )


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
