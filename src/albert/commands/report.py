import typer


app = typer.Typer(help="Report generation command group.")


@app.command("status")
def status() -> None:
    typer.echo(
        "Albert report commands are wired; Phase 1 only wires command structure. "
        "Reports are implemented in later phases."
    )
