from datetime import date
from pathlib import Path

import typer

from albert.config import load_config
from albert.kalshi.pipeline import run_kalshi_discovery
from albert.manifest import config_hash_from_mapping


app = typer.Typer(help="Ingest market data command group.")
kalshi_app = typer.Typer(help="Ingest Kalshi market data.")
app.add_typer(kalshi_app, name="kalshi")


@app.command("status")
def status() -> None:
    typer.echo(
        "Albert ingest commands are wired; Phase 1 only wires command structure. "
        "Data clients are implemented in later phases."
    )


@kalshi_app.command("discover")
def kalshi_discover(
    start_date: str | None = typer.Option(None, "--start-date"),
    end_date: str | None = typer.Option(None, "--end-date"),
    series_ticker: str = typer.Option("KXBTC15M", "--series-ticker"),
    run_id: str = typer.Option(..., "--run-id"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    run_dir: Path | None = typer.Option(None, "--run-dir"),
    config_path: Path | None = typer.Option(None, "--config"),
) -> None:
    if start_date is None or end_date is None:
        typer.echo("Missing required options: --start-date, --end-date")
        raise typer.Exit(code=2)

    parsed_start_date = _parse_cli_date(start_date, "start-date")
    parsed_end_date = _parse_cli_date(end_date, "end-date")
    if parsed_end_date <= parsed_start_date:
        raise typer.BadParameter("end-date must be after start-date")

    config = load_config(config_path)
    cache_root = cache_dir or config.cache_dir
    run_root = run_dir or config.run_dir
    effective_config_path = config_path or config.config_dir / "config.toml"
    result = run_kalshi_discovery(
        start_date=parsed_start_date,
        end_date=parsed_end_date,
        cache_root=cache_root,
        run_root=run_root,
        config_path=effective_config_path,
        config_hash=config_hash_from_mapping(config.model_dump(mode="json")),
        series_ticker=series_ticker,
        run_id=run_id,
    )

    typer.echo("Kalshi discovery complete")
    typer.echo(f"market_count={result.market_count}")
    typer.echo(f"eligible_market_count={result.eligible_market_count}")
    typer.echo(f"skipped_market_count={result.skipped_market_count}")
    typer.echo(f"missing_outcome_count={result.missing_outcome_count}")
    typer.echo(f"missing_price_history_count={result.missing_price_history_count}")
    typer.echo(f"missing_orderbook_count={result.missing_orderbook_count}")
    typer.echo(f"manifest_path={result.manifest_path}")


def _parse_cli_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise typer.BadParameter(
            f"{field_name} must use YYYY-MM-DD format"
        ) from error
