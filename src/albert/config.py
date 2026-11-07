"""Runtime settings. Everything is overridable through ALBERT_* environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: _env_path("ALBERT_DATA_DIR", "data"))
    kalshi_base: str = "https://api.elections.kalshi.com/trade-api/v2"
    pm_gamma_base: str = "https://gamma-api.polymarket.com"
    pm_data_base: str = "https://data-api.polymarket.com/v2"
    pmus_base: str = "https://gateway.polymarket.us/v1"
    iem_base: str = "https://mesonet.agron.iastate.edu"
    pandascore_base: str = "https://api.pandascore.co"
    # Save live (uncached) responses too. Off by default: a long-running paper/live loop would
    # write hundreds of MB a day, and its decisions are already in the ledger.
    record_live: bool = field(default_factory=lambda: os.environ.get("ALBERT_RECORD_LIVE", "0") == "1")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def tables_dir(self) -> Path:
        return self.data_dir / "tables"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def paper_dir(self) -> Path:
        return self.data_dir / "paper"


def load_env_file(path: Path = Path(".env")) -> None:
    """Fill os.environ from a KEY=VALUE .env file without overriding existing variables.

    Secrets stay in the process environment only: they are never put on ``Settings``,
    written to manifests or raw files, or logged.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def secret(name: str) -> str | None:
    load_env_file()
    return os.environ.get(name) or None


def settings() -> Settings:
    return Settings()
