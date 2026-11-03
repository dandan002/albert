import typer

app = typer.Typer(help="Albert: prediction-market research and systematic trading.")


@app.callback()
def main() -> None:
    """Albert research CLI."""


@app.command()
def version() -> None:
    from albert import __version__

    typer.echo(__version__)
