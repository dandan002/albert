from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from albert.manifest import (
    OutputMetadata,
    RunManifest,
    SourceMetadata,
    config_hash_from_mapping,
    write_manifest,
)


NOW = datetime(2026, 4, 28, 12, 0, tzinfo=timezone.utc)


def _source(raw_path: Path) -> SourceMetadata:
    return SourceMetadata(
        source="kalshi",
        endpoint="markets",
        parameters={"ticker": "KXBTCD-26APR281200"},
        fetched_at=NOW,
        covered_start=NOW,
        covered_end=NOW,
        raw_path=raw_path,
    )


def _manifest(tmp_path: Path) -> RunManifest:
    config = {"cache_dir": "cache", "run_dir": "runs", "timezone": "UTC"}
    return RunManifest(
        run_id="test-run",
        created_at=NOW,
        command="albert ingest kalshi",
        parameters={"limit": 100},
        config_path=tmp_path / "config" / "albert.toml",
        config_hash=config_hash_from_mapping(config),
        cache_root=tmp_path / "cache",
        run_root=tmp_path / "runs",
        sources=[_source(tmp_path / "cache" / "raw" / "kalshi" / "markets.json")],
        outputs=[
            OutputMetadata(
                kind="normalized_parquet",
                source="kalshi",
                table="markets",
                path=tmp_path / "cache" / "normalized" / "kalshi" / "markets.parquet",
                row_count=1,
            )
        ],
    )


def test_source_metadata_requires_data_provenance_fields(tmp_path: Path) -> None:
    source = _source(tmp_path / "raw.json")

    assert source.source == "kalshi"
    assert source.endpoint == "markets"
    assert source.parameters == {"ticker": "KXBTCD-26APR281200"}
    assert source.fetched_at == NOW
    assert source.covered_start == NOW
    assert source.covered_end == NOW
    assert source.raw_path == tmp_path / "raw.json"


def test_run_manifest_requires_config_provenance_and_outputs(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    assert manifest.run_id == "test-run"
    assert manifest.created_at == NOW
    assert manifest.config_path == tmp_path / "config" / "albert.toml"
    assert len(manifest.config_hash) == 64
    assert manifest.cache_root == tmp_path / "cache"
    assert manifest.run_root == tmp_path / "runs"
    assert manifest.sources[0].endpoint == "markets"
    assert manifest.outputs[0].kind == "normalized_parquet"


def test_manifest_json_includes_config_path_and_config_hash(tmp_path: Path) -> None:
    payload = _manifest(tmp_path).model_dump_json(indent=2)

    assert '"config_path"' in payload
    assert '"config_hash"' in payload
    assert '"endpoint": "markets"' in payload
    assert '"parameters"' in payload
    assert '"fetched_at"' in payload
    assert '"covered_start"' in payload
    assert '"covered_end"' in payload
    assert '"raw_path"' in payload


def test_config_hash_from_mapping_is_deterministic() -> None:
    left = config_hash_from_mapping(
        {"cache_dir": "cache", "run_dir": "runs", "timezone": "UTC"}
    )
    right = config_hash_from_mapping(
        {"timezone": "UTC", "run_dir": "runs", "cache_dir": "cache"}
    )

    assert left == right
    assert len(left) == 64


def test_source_metadata_rejects_covered_end_before_covered_start(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="covered_end must be"):
        SourceMetadata(
            source="kalshi",
            endpoint="markets",
            parameters={},
            fetched_at=NOW,
            covered_start=NOW,
            covered_end=datetime(2026, 4, 28, 11, 59, tzinfo=timezone.utc),
            raw_path=tmp_path / "raw.json",
        )


def test_manifest_models_reject_naive_datetimes(tmp_path: Path) -> None:
    naive = datetime(2026, 4, 28, 12, 0)

    with pytest.raises(ValidationError, match="timezone-aware"):
        SourceMetadata(
            source="kalshi",
            endpoint="markets",
            parameters={},
            fetched_at=naive,
            covered_start=NOW,
            covered_end=NOW,
            raw_path=tmp_path / "raw.json",
        )

    with pytest.raises(ValidationError, match="timezone-aware"):
        RunManifest(
            run_id="test-run",
            created_at=naive,
            command="albert ingest kalshi",
            parameters={},
            config_path=tmp_path / "config.toml",
            config_hash="a" * 64,
            cache_root=tmp_path / "cache",
            run_root=tmp_path / "runs",
        )


def test_output_metadata_rejects_negative_row_count(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="row_count"):
        OutputMetadata(
            kind="normalized_parquet",
            source="kalshi",
            table="markets",
            path=tmp_path / "markets.parquet",
            row_count=-1,
        )


def test_secret_like_keys_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="api_key"):
        config_hash_from_mapping({"cache_dir": "cache", "api_key": "secret"})

    with pytest.raises(ValidationError, match="token"):
        SourceMetadata(
            source="kalshi",
            endpoint="markets",
            parameters={"token": "secret"},
            fetched_at=NOW,
            covered_start=NOW,
            covered_end=NOW,
            raw_path=tmp_path / "raw.json",
        )

    with pytest.raises(ValidationError, match="api_key"):
        RunManifest(
            run_id="test-run",
            created_at=NOW,
            command="albert ingest kalshi",
            parameters={"api_key": "secret"},
            config_path=tmp_path / "config.toml",
            config_hash="a" * 64,
            cache_root=tmp_path / "cache",
            run_root=tmp_path / "runs",
        )


def test_write_manifest_creates_deterministic_json_file(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "test-run" / "manifest.json"

    written = write_manifest(_manifest(tmp_path), path)

    assert written == path
    payload = path.read_text(encoding="utf-8")
    assert payload.endswith("\n")
    assert '"config_path"' in payload
    assert '"config_hash"' in payload
