# tests/test_main.py
import sys
import pytest
from unittest.mock import patch, MagicMock
from albert.config import load_global_config


def test_load_global_config_returns_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_global_config()
    assert "max_total_notional_usd" in config
    assert "daily_loss_limit_usd" in config
    assert "order_debounce_seconds" in config
    assert "orderbook_ttl_days" in config
    assert "strategy_reload_interval" in config


def test_load_global_config_reads_json_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text('{"max_total_notional_usd": 99999}')
    config = load_global_config()
    assert config["max_total_notional_usd"] == 99999
    # defaults still present for unspecified keys
    assert "daily_loss_limit_usd" in config
