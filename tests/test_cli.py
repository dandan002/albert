from typer.testing import CliRunner

from albert import __version__
from albert.cli import app


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_live_kill_and_resume_write_the_global_switch(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ALBERT_DATA_DIR", str(tmp_path))
    r = CliRunner().invoke(app, ["live", "kill", "--reason", "test"])
    assert r.exit_code == 0 and (tmp_path / "live" / "KILL").exists()
    assert "manual: test" in (tmp_path / "live" / "KILL").read_text()
    r = CliRunner().invoke(app, ["live", "resume"])
    assert r.exit_code == 0 and "KILL" in r.stdout and not (tmp_path / "live" / "KILL").exists()
    r = CliRunner().invoke(app, ["live", "audit", "--env", "prod"])
    assert r.exit_code == 0 and "no order requests audited" in r.stdout and "global kill: off" in r.stdout
