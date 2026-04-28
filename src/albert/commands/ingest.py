import typer


app = typer.Typer(help="Ingest market data command group.")


@app.command("status")
def status() -> None:
    typer.echo(
        "Albert ingest commands are wired; Phase 1 only wires command structure. "
        "Data clients are implemented in later phases."
    )
