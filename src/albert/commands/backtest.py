import typer


app = typer.Typer(help="Backtest strategy command group.")


@app.command("status")
def status() -> None:
    typer.echo(
        "Albert backtest commands are wired; Phase 1 only wires command structure. "
        "Backtests are implemented in later phases."
    )
