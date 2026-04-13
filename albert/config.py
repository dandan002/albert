# albert/config.py
import json
import os
from pathlib import Path

_DEFAULTS = {
    "max_total_notional_usd": 10000.0,
    "daily_loss_limit_usd": -500.0,
    "order_debounce_seconds": 10,
    "orderbook_ttl_days": 7,
    "strategy_reload_interval": 30.0,
    "circuit_breaker_violations": 2,
    "health_check_interval_seconds": 60,
    "shutdown_timeout_seconds": 30,
}


def load_global_config() -> dict:
    config_path = Path("config.json")
    config = dict(_DEFAULTS)
    if config_path.exists():
        overrides = json.loads(config_path.read_text())
        config.update(overrides)
    return config


def load_project_env() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", "\""}:
            value = value[1:-1]

        os.environ[key] = value
