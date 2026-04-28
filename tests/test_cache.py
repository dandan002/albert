from pathlib import Path

import pytest

from albert.cache import (
    normalized_cache_path,
    raw_cache_path,
    run_manifest_path,
    sanitize_segment,
    stable_params_hash,
)


def test_sanitize_segment_lowercases_safe_input() -> None:
    assert sanitize_segment("Kalshi_Markets-1") == "kalshi_markets-1"


@pytest.mark.parametrize("unsafe", ["../outside", "bad/source", "bad source", "..", ""])
def test_sanitize_segment_rejects_unsafe_segments(unsafe: str) -> None:
    with pytest.raises(ValueError, match="unsafe cache path segment"):
        sanitize_segment(unsafe)


def test_stable_params_hash_is_order_independent() -> None:
    left = stable_params_hash({"ticker": "KXBTCD-26APR281200", "limit": 100})
    right = stable_params_hash({"limit": 100, "ticker": "KXBTCD-26APR281200"})

    assert left == right
    assert len(left) == 16


def test_raw_cache_path_is_deterministic_json_under_root(tmp_path: Path) -> None:
    params = {"ticker": "KXBTCD-26APR281200", "limit": 100}

    first = raw_cache_path(tmp_path, "Kalshi", "Markets", params)
    second = raw_cache_path(tmp_path, "kalshi", "markets", dict(reversed(params.items())))

    assert first == second
    assert first.suffix == ".json"
    assert first.is_relative_to(tmp_path)
    assert first.parent == tmp_path / "raw" / "kalshi" / "markets"


def test_normalized_cache_path_is_deterministic_parquet_under_root(
    tmp_path: Path,
) -> None:
    params = {"date": "2026-04-28", "source": "kalshi"}

    first = normalized_cache_path(tmp_path, "Kalshi", "Markets", params)
    second = normalized_cache_path(tmp_path, "kalshi", "markets", dict(reversed(params.items())))

    assert first == second
    assert first.suffix == ".parquet"
    assert first.is_relative_to(tmp_path)
    assert first.parent == tmp_path / "normalized" / "kalshi" / "markets"


def test_run_manifest_path_is_under_run_root(tmp_path: Path) -> None:
    path = run_manifest_path(tmp_path / "runs", "test-run")

    assert path == tmp_path / "runs" / "test-run" / "manifest.json"
    assert path.is_relative_to(tmp_path / "runs")
