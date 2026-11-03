"""Typed Parquet tables plus JSON run manifests."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


def table_path(tables_dir: Path, name: str) -> Path:
    return tables_dir / f"{name}.parquet"


def write_table(
    tables_dir: Path,
    name: str,
    frame: pl.DataFrame,
    *,
    key: list[str] | None = None,
    merge: bool = True,
) -> Path:
    """Write ``frame`` to ``tables/<name>.parquet``.

    With ``merge`` and a ``key``, rows are upserted into the existing table: new rows win on
    key conflicts, so re-ingesting a window refreshes it without dropping other windows.
    """
    path = table_path(tables_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if merge and key and path.exists():
        existing = pl.read_parquet(path)
        frame = pl.concat([existing, frame], how="diagonal_relaxed").unique(
            subset=key, keep="last", maintain_order=True
        )
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)
    return path


def read_table(tables_dir: Path, name: str) -> pl.DataFrame:
    path = table_path(tables_dir, name)
    if not path.exists():
        raise FileNotFoundError(f"table {name!r} not found at {path}; run the ingest step first")
    return pl.read_parquet(path)


def write_manifest(data_dir: Path, command: str, params: dict[str, Any], outputs: dict[str, Any],
                   raw_files: list[Path]) -> Path:
    created = datetime.now(UTC)
    run_id = f"{created.strftime('%Y%m%dT%H%M%SZ')}-{command.replace(' ', '-')}"
    path = data_dir / "runs" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "run_id": run_id,
        "created_at": created.isoformat(),
        "command": command,
        "params": params,
        "outputs": outputs,
        "raw_files": sorted({str(p) for p in raw_files}),
    }, indent=2, default=str))
    return path
