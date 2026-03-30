# tests/test_main.py
import os

from albert.config import load_global_config, load_project_env


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


def test_load_project_env_reads_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
    monkeypatch.delenv("POLYMARKET_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "KALSHI_API_KEY_ID=test-key\n"
        "POLYMARKET_API_KEY='poly-key'\n"
        "# ignored comment\n"
        "\n"
    )

    load_project_env()

    assert os.environ["KALSHI_API_KEY_ID"] == "test-key"
    assert os.environ["POLYMARKET_API_KEY"] == "poly-key"


def test_load_project_env_does_not_override_existing_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KALSHI_API_KEY_ID", "shell-value")
    (tmp_path / ".env").write_text("KALSHI_API_KEY_ID=file-value\n")

    load_project_env()

    assert os.environ["KALSHI_API_KEY_ID"] == "shell-value"
