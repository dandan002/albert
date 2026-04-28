import typer


app = typer.Typer(help="Model probability command group.")


@app.command("status")
def status() -> None:
    typer.echo(
        "Albert model commands are wired; Phase 1 only wires command structure. "
        "Probability models are implemented in later phases."
    )
