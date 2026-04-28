import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError


def _config_api():
    try:
        module = importlib.import_module("albert.config")
    except ModuleNotFoundError as exc:
        pytest.fail(f"albert.config is not importable: {exc}")
    return module.default_config, module.load_config


def test_default_config_uses_platform_user_roots() -> None:
    default_config, _ = _config_api()

    config = default_config()

    assert config.cache_dir.name == "albert"
    assert config.run_dir == config.cache_dir / "runs"
    assert config.config_dir.name == "albert"
    assert config.timezone == "UTC"


def test_load_config_reads_toml_over_defaults(tmp_path: Path) -> None:
    _, load_config = _config_api()

    config_path = tmp_path / "albert.toml"
    config_path.write_text(
        "\n".join(
            [
                f'cache_dir = "{tmp_path / "cache"}"',
                f'run_dir = "{tmp_path / "runs"}"',
                f'config_dir = "{tmp_path / "config"}"',
                'timezone = "America/New_York"',
            ]
        )
    )

    config = load_config(config_path)

    assert config.cache_dir == tmp_path / "cache"
    assert config.run_dir == tmp_path / "runs"
    assert config.config_dir == tmp_path / "config"
    assert config.timezone == "America/New_York"


def test_albert_environment_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, load_config = _config_api()

    config_path = tmp_path / "albert.toml"
    config_path.write_text(
        "\n".join(
            [
                f'cache_dir = "{tmp_path / "toml-cache"}"',
                f'run_dir = "{tmp_path / "toml-runs"}"',
                f'config_dir = "{tmp_path / "toml-config"}"',
            ]
        )
    )
    monkeypatch.setenv("ALBERT_CACHE_DIR", str(tmp_path / "env-cache"))
    monkeypatch.setenv("ALBERT_RUN_DIR", str(tmp_path / "env-runs"))

    config = load_config(config_path)

    assert config.cache_dir == tmp_path / "env-cache"
    assert config.run_dir == tmp_path / "env-runs"
    assert config.config_dir == tmp_path / "toml-config"


def test_rejects_parent_traversal_paths(tmp_path: Path) -> None:
    _, load_config = _config_api()

    config_path = tmp_path / "albert.toml"
    config_path.write_text('cache_dir = "../outside"\n')

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_rejects_secret_like_config_keys(tmp_path: Path) -> None:
    _, load_config = _config_api()

    config_path = tmp_path / "albert.toml"
    config_path.write_text('api_key = "secret"\n')

    with pytest.raises(ValidationError):
        load_config(config_path)
