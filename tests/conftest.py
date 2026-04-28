from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def frozen_utc() -> datetime:
    return datetime(2026, 4, 28, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def deterministic_run_id() -> str:
    return "test-run"


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def example_config_path(tmp_path: Path) -> Path:
    config_path = tmp_path / "albert.toml"
    config_path.write_text(
        "\n".join(
            [
                f'cache_dir = "{tmp_path / "cache"}"',
                f'run_dir = "{tmp_path / "runs"}"',
                f'config_dir = "{tmp_path / "config"}"',
                'timezone = "UTC"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path
