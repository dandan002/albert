import typer

from albert.commands import backtest, ingest, model, report


app = typer.Typer(help="Albert research CLI")
app.add_typer(ingest.app, name="ingest")
app.add_typer(model.app, name="model")
app.add_typer(backtest.app, name="backtest")
app.add_typer(report.app, name="report")
