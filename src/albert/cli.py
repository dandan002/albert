from datetime import UTC, date, datetime, time

import typer

from albert import __version__
from albert.config import settings

app = typer.Typer(help="Albert: prediction-market research and systematic trading.",
                  no_args_is_help=True)
ingest_app = typer.Typer(help="Fetch venue data into data/tables.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")


def _day(value: str) -> datetime:
    return datetime.combine(date.fromisoformat(value), time.min, tzinfo=UTC)


def _clients(*, live_loop: bool = False):
    from albert.ingest import Clients

    s = settings()
    return s, Clients.build(s, record_live=s.record_live if live_loop else True)


@app.command()
def version() -> None:
    typer.echo(__version__)


@ingest_app.command("kalshi")
def ingest_kalshi(
    start: str = typer.Option(..., help="YYYY-MM-DD, inclusive (market close date)"),
    end: str = typer.Option(..., help="YYYY-MM-DD, exclusive"),
    category: list[str] = typer.Option(None, help="Kalshi category, repeatable (e.g. Sports)"),
    min_volume: float = typer.Option(0.0, help="Minimum contracts traded"),
    trades: bool = typer.Option(False, help="Also fetch trades per market"),
    candles: bool = typer.Option(False, help="Also fetch bid/ask candlesticks per market"),
    candle_minutes: int = typer.Option(60, help="Candle period: 1, 60, or 1440"),
    max_markets: int = typer.Option(None, help="Fetch trades/candles for the top-N by volume"),
    per_category: bool = typer.Option(False, help="Apply --max-markets within each category"),
) -> None:
    """Settled non-financial Kalshi markets (plus optional trades and candles)."""
    from albert.ingest import ingest_kalshi as run

    s, c = _clients()
    result = run(s, c, _day(start), _day(end), categories=category or None,
                 min_volume=min_volume, trades=trades, candles=candles,
                 candle_minutes=candle_minutes, max_markets=max_markets,
                 per_category=per_category)
    typer.echo(result.summary())


@ingest_app.command("kalshi-detail")
def ingest_kalshi_detail(
    sample_per_category: int = typer.Option(0, help="Random markets per category (trade study)"),
    min_volume: float = typer.Option(500.0, help="Volume band low end for the sample"),
    max_volume: float = typer.Option(200_000.0, help="Volume band high end for the sample"),
    multi_events: int = typer.Option(0, help="Complete mutually-exclusive events to fetch"),
    max_outcomes: int = typer.Option(12, help="Max legs per multi-outcome event"),
    seed: int = typer.Option(7),
    trades: bool = typer.Option(True),
    candles: bool = typer.Option(True),
    max_trade_pages: int = typer.Option(10, help="1000-trade pages per market (newest first)"),
    series: list[str] = typer.Option(None, help="Every market of this series (one leg per event, no volume filter); repeatable"),
    start: str = typer.Option(None, help="YYYY-MM-DD, with --series"),
    end: str = typer.Option(None, help="YYYY-MM-DD, with --series"),
    category: list[str] = typer.Option(None, help="Restrict --sample-per-category to these categories"),
    all_legs: bool = typer.Option(False, help="With --series: every leg of each event, not one"),
    events_per_series: int = typer.Option(None, help="With --series: random (volume-blind) events per series"),
) -> None:
    """Trades/candles for a seeded sample of already-ingested settled Kalshi markets."""
    from albert.ingest import ingest_kalshi_detail as run

    s, c = _clients()
    result = run(s, c, sample_per_category=sample_per_category, min_volume=min_volume,
                 max_volume=max_volume, multi_events=multi_events, max_outcomes=max_outcomes,
                 seed=seed, trades=trades, candles=candles, max_trade_pages=max_trade_pages,
                 series=series or None, start=_day(start) if start else None,
                 end=_day(end) if end else None, categories=category or None,
                 all_legs=all_legs, events_per_series=events_per_series)
    typer.echo(result.summary())


@ingest_app.command("kalshi-trade-windows")
def ingest_kalshi_trade_windows(
    series: list[str] = typer.Option(..., help="Game series (one leg per game, all games); repeatable"),
    start: str = typer.Option(..., help="YYYY-MM-DD (market close date)"),
    end: str = typer.Option(..., help="YYYY-MM-DD, exclusive"),
    horizon: float = typer.Option(24.0, help="Maker order posted this many hours before scheduled expiry"),
    ttl: float = typer.Option(20.0, help="Maker order lifetime in hours"),
    anchor: str = typer.Option("scheduled", help="scheduled (expiry) or start (ticker game start)"),
    legs: str = typer.Option("population", help="population (one leg of every event) or sampled (legs with candles)"),
) -> None:
    """All trades inside each game's maker-order window (complete coverage, no volume bias)."""
    from albert.ingest import ingest_kalshi_trade_windows as run

    s, c = _clients()
    typer.echo(run(s, c, series=series, start=_day(start), end=_day(end), horizon_hours=horizon,
                   ttl_hours=ttl, anchor=anchor, legs=legs).summary())


@ingest_app.command("weather")
def ingest_weather(
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(..., help="YYYY-MM-DD, exclusive"),
    code: list[str] = typer.Option(None, help="CLI station code (NYC, MDW, ...); default: all Kalshi ones"),
) -> None:
    """NWS CLI reports, ASOS observations and NBM forecasts (IEM archive) for Kalshi cities."""
    from albert.ingest import ingest_weather as run

    s, c = _clients()
    typer.echo(run(s, c, _day(start), _day(end), codes=code or None).summary())


@ingest_app.command("kalshi-events")
def ingest_kalshi_events(min_legs: int = typer.Option(3, help="Only events with >= N legs")) -> None:
    """Event metadata (mutually_exclusive, leg count) for multi-leg events with candles."""
    from albert.ingest import ingest_kalshi_events as run

    s, c = _clients()
    typer.echo(run(s, c, min_legs=min_legs).summary())


@ingest_app.command("kalshi-open")
def ingest_kalshi_open() -> None:
    """Snapshot currently open non-financial Kalshi markets."""
    from albert.ingest import ingest_kalshi_open as run

    s, c = _clients()
    typer.echo(f"kalshi_open_markets: {run(s, c).height} rows")


@ingest_app.command("pm-markets")
def ingest_pm_markets(
    start: str = typer.Option(..., help="YYYY-MM-DD (event end date)"),
    end: str = typer.Option(..., help="YYYY-MM-DD, exclusive"),
    open_: bool = typer.Option(False, "--open", help="Open markets instead of resolved ones"),
    min_volume: float = typer.Option(100.0, help="Minimum USDC volume"),
) -> None:
    """Polymarket intl (data only) market metadata and resolutions."""
    from albert.ingest import ingest_pm_markets as run

    s, c = _clients()
    typer.echo(run(s, c, _day(start), _day(end), closed=not open_,
                   min_volume=min_volume).summary())


@ingest_app.command("pm-wallets")
def ingest_pm_wallets(
    start: str = typer.Option(..., help="YYYY-MM-DD, trade window start"),
    end: str = typer.Option(..., help="YYYY-MM-DD, trade window end (exclusive)"),
    top: int = typer.Option(200, help="Wallets per leaderboard category"),
    category: list[str] = typer.Option(None, help="Leaderboard category (overall, sports, politics...)"),
    time_period: str = typer.Option("all", help="Leaderboard window: day, week, month, all"),
    wallet: list[str] = typer.Option(None, help="Extra wallet address, repeatable"),
    max_pages: int = typer.Option(20, help="Max 1000-trade pages per wallet"),
) -> None:
    """Leaderboard wallets and their full trade history, then backfill market resolutions."""
    from albert.ingest import ingest_pm_resolutions_for_trades, ingest_pm_wallets as run

    s, c = _clients()
    result = run(s, c, _day(start), _day(end), top=top,
                 categories=tuple(category or ("overall",)), time_period=time_period,
                 extra_wallets=wallet, max_pages=max_pages)
    typer.echo(result.summary())
    typer.echo(ingest_pm_resolutions_for_trades(s, c).summary())


@ingest_app.command("pmus")
def ingest_pmus(
    open_: bool = typer.Option(False, "--open", help="Open markets instead of resolved ones"),
    start: str = typer.Option(None, help="YYYY-MM-DD: market end date from (needed for closed)"),
    end: str = typer.Option(None, help="YYYY-MM-DD: market end date to (exclusive)"),
    max_pages: int = typer.Option(None, help="Stop after N pages of 500 markets"),
) -> None:
    """Polymarket US market metadata (tradeable venue), non-financial categories only."""
    from albert.ingest import ingest_pmus as run

    if not open_ and not (start and end):
        typer.echo("Closed Polymarket US ingests need --start and --end (2M+ closed markets).")
        raise typer.Exit(2)
    s, c = _clients()
    typer.echo(run(s, c, closed=not open_, start=_day(start) if start else None,
                   end=_day(end) if end else None, max_pages=max_pages).summary())


# ---- matching / research / backtest ------------------------------------------------------------
research_app = typer.Typer(help="Research reports (written to data/reports).", no_args_is_help=True)
backtest_app = typer.Typer(help="Backtests over strategy signals.", no_args_is_help=True)
app.add_typer(research_app, name="research")
app.add_typer(backtest_app, name="backtest")


@app.command()
def match(fuzzy: bool = typer.Option(False, help="Also write fuzzy title candidates for review")) -> None:
    """Match markets across Kalshi, Polymarket intl, and Polymarket US."""
    from albert.matching import match_all

    for name, rows in match_all(settings().tables_dir, fuzzy=fuzzy).items():
        typer.echo(f"{name}: {rows} rows")


@research_app.command("calibration")
def research_calibration(
    horizon: float = typer.Option(2.0, help="Hours before close to take the decision price"),
    width: float = typer.Option(0.05, help="Price bucket width"),
) -> None:
    """Favorite-longshot / calibration tables with after-fee edges."""
    from albert.pipeline import research_calibration as run

    typer.echo(run(settings(), horizon_hours=horizon, width=width))


@research_app.command("wallets")
def research_wallets(
    split: str = typer.Option(..., help="YYYY-MM-DD: score before, evaluate after"),
    top_k: int = typer.Option(20, help="Wallets in the out-of-sample test"),
) -> None:
    """Wallet skill scores and out-of-sample persistence."""
    from albert.pipeline import research_wallets as run

    typer.echo(run(settings(), split=_day(split), top_k=top_k))


@research_app.command("gaps")
def research_gaps(half_spread: float = typer.Option(0.01, help="Assumed Polymarket half spread")) -> None:
    """Kalshi vs Polymarket price gaps on exactly matched markets."""
    from albert.pipeline import research_gaps as run

    s, c = _clients()
    typer.echo(run(s, c, half_spread=half_spread))


@research_app.command("sports")
def research_sports(
    series: list[str] = typer.Option(..., help="Kalshi series (with start times in tickers); repeatable"),
    lead: list[float] = typer.Option([1.0, 0.25], help="Hours before the scheduled start"),
    split: str = typer.Option("2026-10-01", help="Fit before, trade from"),
    anchor: str = typer.Option("start", help="start (ticker time) or scheduled (expected expiry; lead counts from it)"),
) -> None:
    """Pre-game calibration and a per-series recalibration strategy, volume-blind sample."""
    from albert.pipeline import research_sports as run

    typer.echo(run(settings(), series=series, leads=tuple(lead), split=_day(split), anchor=anchor))


@research_app.command("esports-reference")
def research_esports_reference() -> None:
    """Polymarket intl esports prices as a reference for Kalshi esports match winners."""
    from albert.pipeline import research_esports_reference as run

    s, c = _clients()
    typer.echo(run(s, c))


@research_app.command("literature")
def research_literature() -> None:
    """Re-test published effects: maker/taker, leg-sum arb, category slopes, reversion, moves."""
    from albert.pipeline import research_literature as run

    typer.echo(run(settings()))


@backtest_app.command("weather")
def backtest_weather(
    split: str = typer.Option("2026-10-01", help="YYYY-MM-DD: fit before, trade from this day"),
    slot: list[str] = typer.Option(None, help="Decision slots: 'd-1 15h', 'd 10h', 'd 13h', 'd 16h'"),
    cash: float = typer.Option(50.0, help="Kalshi bankroll"),
    min_edge: float = typer.Option(0.05, help="Model prob - price - fee needed to trade"),
    kelly: float = typer.Option(0.25),
    max_bet: float = typer.Option(10.0),
) -> None:
    """NWS (NBM forecast + ASOS observations) model vs Kalshi daily-high brackets."""
    from albert.pipeline import backtest_weather as run
    from albert.risk import RiskConfig

    typer.echo(run(settings(), split=_day(split), slots=tuple(slot) if slot else None, cash=cash,
                   cfg=RiskConfig(kelly_fraction=kelly, min_edge=min_edge, max_bet_dollars=max_bet)))


@backtest_app.command("maker")
def backtest_maker(
    horizon: list[float] = typer.Option([24.0, 6.0], help="Hours before scheduled expiry to post"),
    size: int = typer.Option(5, help="Contracts per order in the flat test"),
    cash: float = typer.Option(50.0, help="Kalshi bankroll for the compounding simulation"),
    category: list[str] = typer.Option(["Climate and Weather", "Sports"], help="Kalshi categories"),
    lo: float = typer.Option(0.50, help="Lowest favorite bid"),
    hi: float = typer.Option(0.95, help="Highest favorite bid"),
    ttl: float = typer.Option(None, help="Cancel after N hours (default: at market close)"),
    exclude_tickers: str = typer.Option(None, help="Parquet with a `ticker` column to exclude (holdout test)"),
    queue: str = typer.Option("any", help="any, join (1-tick markets only), improve (wider only)"),
    universe: str = typer.Option("sample", help="sample (random volume band; biased) or population (all games, windowed trades)"),
    anchor: str = typer.Option("scheduled", help="scheduled (expiry - horizon) or start (ticker start - horizon)"),
    series: list[str] = typer.Option(None, help="Restrict to these series"),
) -> None:
    """Resting bids on Kalshi favorites, filled only by later real trades into our price."""
    from pathlib import Path

    from albert.pipeline import backtest_maker as run

    typer.echo(run(settings(), horizons=tuple(horizon), size=size, cash=cash,
                   categories=tuple(category), lo=lo, hi=hi, ttl_hours=ttl,
                   exclude_tickers=Path(exclude_tickers) if exclude_tickers else None,
                   queue=queue, universe=universe, anchor=anchor, series=series or None))


@backtest_app.command("run")
def backtest_run(
    strategy: list[str] = typer.Option(["longshot"], help="longshot, slope, and/or wallet_follow"),
    split: str = typer.Option(..., help="YYYY-MM-DD: train before, trade after"),
    kalshi_cash: float = typer.Option(50.0),
    pmus_cash: float = typer.Option(50.0),
    kelly: float = typer.Option(0.25, help="Fraction of full Kelly"),
    min_edge: float = typer.Option(0.03, help="Required edge per contract after fees"),
    max_bet: float = typer.Option(10.0, help="Max dollars per order"),
    min_price: float = typer.Option(0.03, help="Skip contracts cheaper than this"),
    max_price: float = typer.Option(0.97, help="Skip contracts dearer than this"),
    horizon: float = typer.Option(2.0, help="longshot: hours before close"),
    prior: float = typer.Option(200.0, help="longshot: pseudo-observations at the market price"),
    anchor: str = typer.Option("scheduled", help="Decision anchor: scheduled (no leak) or close (leaky, for comparison)"),
    universe: str = typer.Option("all", help="Kalshi markets: all, or population (volume-blind game sample)"),
    delay: float = typer.Option(5.0, help="wallet_follow: minutes after the wallet's trade"),
    top_k: int = typer.Option(20, help="wallet_follow: wallets to copy"),
) -> None:
    """Simulate strategies with real fees, bid/ask fills, fractional Kelly, and risk caps."""
    from albert.pipeline import backtest
    from albert.risk import RiskConfig

    s, c = _clients()
    cfg = RiskConfig(kelly_fraction=kelly, min_edge=min_edge, max_bet_dollars=max_bet,
                     min_price=min_price, max_price=max_price)
    typer.echo(backtest(s, c, strategies=strategy, split=_day(split),
                        cash={"kalshi": kalshi_cash, "pmus": pmus_cash}, cfg=cfg,
                        horizon_hours=horizon, delay_minutes=delay, top_k=top_k, prior=prior,
                        anchor=anchor, universe=universe))


# ---- paper trading -------------------------------------------------------------------------------
paper_app = typer.Typer(help="Paper trading on live quotes (no real orders).", no_args_is_help=True)
app.add_typer(paper_app, name="paper")


@paper_app.command("reset")
def paper_reset(kalshi_cash: float = typer.Option(50.0), pmus_cash: float = typer.Option(50.0),
                yes: bool = typer.Option(False, "--yes", help="Confirm wiping the paper account")) -> None:
    """Start a fresh paper account (the ledger is kept and appended to)."""
    from albert.paper import PaperTrader

    s = settings()
    if (s.paper_dir / "state.json").exists() and not yes:
        typer.echo("A paper account exists. Re-run with --yes to replace it.")
        raise typer.Exit(1)
    PaperTrader.reset(s, {"kalshi": kalshi_cash, "pmus": pmus_cash})
    typer.echo(f"paper account reset: kalshi={kalshi_cash} pmus={pmus_cash}")


@paper_app.command("run")
def paper_run(
    strategy: list[str] = typer.Option(["maker"], help="maker, longshot, wallet_follow, xvenue"),
    interval: float = typer.Option(60.0, help="Seconds between ticks"),
    iterations: int = typer.Option(0, help="Number of ticks (0 = run until stopped)"),
    horizon: float = typer.Option(2.0, help="longshot: hours before close"),
    prior: float = typer.Option(200.0, help="longshot: pseudo-observations at the market price"),
    kelly: float = typer.Option(0.25),
    min_edge: float = typer.Option(0.03),
    max_bet: float = typer.Option(10.0),
    top_k: int = typer.Option(20, help="wallet_follow: wallets to follow"),
    maker_horizon: float = typer.Option(24.0, help="maker: post this many hours before scheduled expiry"),
    maker_ttl: float = typer.Option(20.0, help="maker: cancel unfilled orders after N hours"),
    maker_size: float = typer.Option(10.0, help="maker: max dollars per order"),
) -> None:
    """Poll live quotes, simulate fills, and settle positions as markets resolve."""
    from albert.paper import PaperTrader, run_loop
    from albert.risk import RiskConfig

    s, c = _clients(live_loop=True)
    trader = PaperTrader(s, c, cfg=RiskConfig(kelly_fraction=kelly, min_edge=min_edge,
                                              max_bet_dollars=max_bet),
                         strategies=tuple(strategy), horizon_hours=horizon, top_k=top_k,
                         prior=prior, maker_horizon=maker_horizon, maker_ttl=maker_ttl,
                         maker_size_dollars=maker_size)
    run_loop(trader, interval_seconds=interval, iterations=iterations)


@paper_app.command("status")
def paper_status() -> None:
    """Paper account equity, cash, and open positions."""
    from albert.paper import status

    typer.echo(status(settings()))


@app.command("compact-raw")
def compact_raw_cmd(min_age: float = typer.Option(60.0, help="Skip files younger than N seconds")) -> None:
    """Compress legacy plain-JSON raw files to .json.zst (lossless, verified, in place)."""
    from albert.http import compact_raw

    files, before, after = compact_raw(settings().raw_dir, min_age_seconds=min_age)
    typer.echo(f"compacted {files} files: {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB")


@app.command()
def capture(out: str = typer.Option("tests/fixtures", help="Fixture root directory")) -> None:
    """Record trimmed real API responses as test fixtures."""
    from pathlib import Path

    from albert.capture import capture as run

    s, c = _clients()
    for path in run(s, c, Path(out)):
        typer.echo(path)
