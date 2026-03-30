# albert/config.py
import json
from pathlib import Path

_DEFAULTS = {
    "max_total_notional_usd": 10000.0,
    "daily_loss_limit_usd": -500.0,
    "order_debounce_seconds": 10,
    "orderbook_ttl_days": 7,
    "strategy_reload_interval": 30.0,
}


def load_global_config() -> dict:
    config_path = Path("config.json")
    config = dict(_DEFAULTS)
    if config_path.exists():
        overrides = json.loads(config_path.read_text())
        config.update(overrides)
    return config
