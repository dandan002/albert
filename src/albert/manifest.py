import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SECRET_KEY_PARTS = ("secret", "token", "password", "api_key", "apikey")


def _reject_secret_keys(mapping: Mapping[str, Any], location: str) -> Mapping[str, Any]:
    for key, value in mapping.items():
        lowered = str(key).lower()
        if any(part in lowered for part in SECRET_KEY_PARTS):
            raise ValueError(f"secret-like key is not allowed in {location}: {key}")
        if isinstance(value, Mapping):
            _reject_secret_keys(value, f"{location}.{key}")
    return mapping


def _require_aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


def config_hash_from_mapping(config: Mapping[str, Any]) -> str:
    _reject_secret_keys(config, "config")
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    endpoint: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime
    covered_start: datetime
    covered_end: datetime
    raw_path: Path

    @field_validator("parameters")
    @classmethod
    def reject_secret_parameter_keys(cls, value: dict[str, Any]) -> dict[str, Any]:
        _reject_secret_keys(value, "source parameters")
        return value

    @field_validator("fetched_at", "covered_start", "covered_end")
    @classmethod
    def require_aware_datetimes(cls, value: datetime) -> datetime:
        return _require_aware_datetime(value)

    @model_validator(mode="after")
    def require_ordered_covered_range(self) -> "SourceMetadata":
        if self.covered_end < self.covered_start:
            raise ValueError("covered_end must be greater than or equal to covered_start")
        return self


class OutputMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["raw_json", "normalized_parquet", "report_artifact"]
    source: str | None = None
    table: str | None = None
    path: Path
    row_count: int | None = None

    @field_validator("row_count")
    @classmethod
    def reject_negative_row_count(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("row_count cannot be negative")
        return value


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: datetime
    command: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    config_path: Path
    config_hash: str
    cache_root: Path
    run_root: Path
    sources: list[SourceMetadata] = Field(default_factory=list)
    outputs: list[OutputMetadata] = Field(default_factory=list)

    @field_validator("parameters")
    @classmethod
    def reject_secret_parameter_keys(cls, value: dict[str, Any]) -> dict[str, Any]:
        _reject_secret_keys(value, "run parameters")
        return value

    @field_validator("created_at")
    @classmethod
    def require_aware_created_at(cls, value: datetime) -> datetime:
        return _require_aware_datetime(value)

    @field_validator("config_hash")
    @classmethod
    def require_config_hash_shape(cls, value: str) -> str:
        if not re.fullmatch(r"^[a-f0-9]{64}$", value):
            raise ValueError("config_hash must be a 64-character lowercase hex digest")
        return value


def write_manifest(manifest: RunManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "OutputMetadata",
    "RunManifest",
    "SourceMetadata",
    "config_hash_from_mapping",
    "write_manifest",
]
