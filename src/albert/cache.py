import hashlib
import json
import re
from pathlib import Path
from typing import Mapping


SAFE_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def sanitize_segment(value: str) -> str:
    segment = value.strip().lower()
    if not segment or any(character.isspace() for character in value):
        raise ValueError(f"unsafe cache path segment: {value!r}")
    if not SAFE_SEGMENT.fullmatch(segment):
        raise ValueError(f"unsafe cache path segment: {value!r}")
    return segment


def stable_params_hash(parameters: Mapping[str, object]) -> str:
    payload = json.dumps(
        parameters,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _ensure_under_root(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve(strict=False)
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ValueError("cache path escaped configured root")
    return candidate


def raw_cache_path(
    root: Path,
    source: str,
    endpoint: str,
    parameters: Mapping[str, object],
) -> Path:
    candidate = (
        root
        / "raw"
        / sanitize_segment(source)
        / sanitize_segment(endpoint)
        / f"{stable_params_hash(parameters)}.json"
    )
    return _ensure_under_root(root, candidate)


def normalized_cache_path(
    root: Path,
    source: str,
    table: str,
    parameters: Mapping[str, object],
) -> Path:
    candidate = (
        root
        / "normalized"
        / sanitize_segment(source)
        / sanitize_segment(table)
        / f"{stable_params_hash(parameters)}.parquet"
    )
    return _ensure_under_root(root, candidate)


def run_manifest_path(run_root: Path, run_id: str) -> Path:
    candidate = run_root / sanitize_segment(run_id) / "manifest.json"
    return _ensure_under_root(run_root, candidate)


__all__ = [
    "SAFE_SEGMENT",
    "normalized_cache_path",
    "raw_cache_path",
    "run_manifest_path",
    "sanitize_segment",
    "stable_params_hash",
]
