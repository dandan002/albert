import importlib

import pytest
from typer.testing import CliRunner


runner = CliRunner()


def _load_app():
    try:
        module = importlib.import_module("albert.cli")
    except ModuleNotFoundError as exc:
        pytest.fail(f"albert.cli is not importable: {exc}")
    return module.app


def test_root_help_lists_phase_one_command_groups() -> None:
    app = _load_app()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command_name in ("ingest", "model", "backtest", "report"):
        assert command_name in result.stdout


def test_placeholder_status_commands_are_wired() -> None:
    app = _load_app()

    for command_name in ("ingest", "model", "backtest", "report"):
        result = runner.invoke(app, [command_name, "status"])

        assert result.exit_code == 0
        assert "Phase 1 only wires command structure" in result.stdout
