from datetime import date
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from albert.cli import app


runner = CliRunner()


def test_discover_requires_explicit_date_range_and_writes_summary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_discovery(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            run_id=kwargs["run_id"],
            manifest_path=tmp_path / "runs" / kwargs["run_id"] / "manifest.json",
            output_paths={},
            market_count=2,
            eligible_market_count=1,
            skipped_market_count=1,
            missing_outcome_count=0,
            missing_price_history_count=0,
            missing_orderbook_count=0,
        )

    import albert.commands.ingest as ingest

    monkeypatch.setattr(ingest, "run_kalshi_discovery", fake_discovery, raising=False)
    try:
        import albert.kalshi.pipeline as pipeline
    except ModuleNotFoundError:
        pipeline = None
    if pipeline is not None:
        monkeypatch.setattr(pipeline, "run_kalshi_discovery", fake_discovery)

    missing_dates = runner.invoke(
        app,
        ["ingest", "kalshi", "discover", "--series-ticker", "KXBTC15M", "--run-id", "test-run"],
    )
    assert missing_dates.exit_code != 0
    assert "--start-date" in missing_dates.stdout
    assert "--end-date" in missing_dates.stdout

    result = runner.invoke(
        app,
        [
            "ingest",
            "kalshi",
            "discover",
            "--start-date",
            "2026-04-01",
            "--end-date",
            "2026-04-02",
            "--series-ticker",
            "KXBTC15M",
            "--run-id",
            "test-run",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--run-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Kalshi discovery complete" in result.stdout
    assert "eligible_market_count=1" in result.stdout
    assert calls[0]["start_date"] == date(2026, 4, 1)
    assert calls[0]["end_date"] == date(2026, 4, 2)
    assert calls[0]["series_ticker"] == "KXBTC15M"
    assert calls[0]["run_id"] == "test-run"
    assert calls[0]["cache_root"] == tmp_path / "cache"
    assert calls[0]["run_root"] == tmp_path / "runs"
