"""Markdown + Parquet report writer (data/reports/<name>-<timestamp>/)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl


def md_table(frame: pl.DataFrame, max_rows: int = 30, digits: int = 4) -> str:
    if frame.is_empty():
        return "_(no rows)_"
    f = frame.head(max_rows)
    def cell(v):
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return "" if v is None else str(v)
    head = "| " + " | ".join(f.columns) + " |"
    sep = "| " + " | ".join("---" for _ in f.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in f.iter_rows()]
    more = [f"\n_{frame.height - max_rows} more rows in the Parquet file._"] if frame.height > max_rows else []
    return "\n".join([head, sep, *body, *more])


class Report:
    def __init__(self, reports_dir: Path, name: str) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.dir = reports_dir / f"{name}-{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.parts: list[str] = [f"# {name}\n\n_generated {stamp}_\n"]

    def text(self, title: str, body: str) -> None:
        self.parts.append(f"## {title}\n\n{body}\n")

    def table(self, title: str, frame: pl.DataFrame, *, save_as: str | None = None,
              max_rows: int = 30) -> None:
        self.parts.append(f"## {title}\n\n{md_table(frame, max_rows)}\n")
        if save_as:
            frame.write_parquet(self.dir / f"{save_as}.parquet")

    def json(self, title: str, data: dict) -> None:
        self.parts.append(f"## {title}\n\n```json\n{json.dumps(data, indent=2, default=str)}\n```\n")

    def image(self, title: str, filename: str) -> None:
        self.parts.append(f"## {title}\n\n![{title}]({filename})\n")

    def save(self) -> Path:
        path = self.dir / "report.md"
        path.write_text("\n".join(self.parts))
        return path
