import json
from pathlib import Path
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
    tickers_file: str = typer.Option(None, help="Parquet with a `ticker` column: fetch exactly these legs"),
) -> None:
    """All trades inside each game's maker-order window (complete coverage, no volume bias)."""
    import polars as pl

    from albert.ingest import ingest_kalshi_trade_windows as run

    s, c = _clients()
    typer.echo(run(s, c, series=series, start=_day(start), end=_day(end), horizon_hours=horizon,
                   ttl_hours=ttl, anchor=anchor, legs=legs,
                   tickers=pl.read_parquet(tickers_file)["ticker"].to_list() if tickers_file else None).summary())


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


@ingest_app.command("pandascore")
def ingest_pandascore(
    start: str = typer.Option(..., help="YYYY-MM-DD"), end: str = typer.Option(..., help="YYYY-MM-DD"),
    game: list[str] = typer.Option(None, help="csgo, lol, dota2, valorant (default: all)"),
) -> None:
    """Esports match + map results from PandaScore (needs PANDASCORE_TOKEN in .env)."""
    from albert.ingest import ingest_pandascore as run

    s, c = _clients()
    typer.echo(run(s, c, _day(start), _day(end), games=tuple(game) if game else
                   ("csgo", "lol", "dota2", "valorant")).summary())


@ingest_app.command("elo-history")
def ingest_elo_history() -> None:
    """All settled markets (historical + live) for the Elo series."""
    from albert.ingest import ingest_elo_history as run

    s, c = _clients()
    typer.echo(run(s, c).summary())


@ingest_app.command("elo-candles")
def ingest_elo_candles(
    start: str = typer.Option(..., help="YYYY-MM-DD"), end: str = typer.Option(..., help="YYYY-MM-DD"),
    per_series_month: int = typer.Option(250), seed: int = typer.Option(41),
) -> None:
    """Candles for a volume-blind sample of Elo-series events (per series and month)."""
    from albert.ingest import ingest_elo_candles as run

    s, c = _clients()
    typer.echo(run(s, c, start=_day(start), end=_day(end), per_series_month=per_series_month,
                   seed=seed).summary())


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


@research_app.command("elo")
def research_elo(
    series: list[str] = typer.Option(..., help="Two-sided series with ticker start times; repeatable"),
    lead: float = typer.Option(1.0, help="Hours before the start"),
    split: str = typer.Option("2026-10-01", help="Fit the blend before, test from"),
    cash: float = typer.Option(50.0),
) -> None:
    """Market + Elo blend for minor sports, fit before the split and tested after it."""
    from albert.pipeline import research_elo as run

    typer.echo(run(settings(), series=series, lead=lead, split=_day(split), cash=cash))


@research_app.command("elo-walk")
def research_elo_walk(
    lead: float = typer.Option(1.0), k: float = typer.Option(32.0), min_games: int = typer.Option(10),
    exclude: list[str] = typer.Option(None, help="Series to leave out; repeatable"),
    first_test: str = typer.Option("2026-04", help="First test month (YYYY-MM)"),
) -> None:
    """Monthly walk-forward of the Elo + market blend (fit on earlier months only)."""
    from albert.pipeline import research_elo_walk as run

    typer.echo(run(settings(), lead=lead, k=k, min_games=min_games, exclude=tuple(exclude or ()),
                   first_test=first_test))


@research_app.command("esports-ratings")
def research_esports_ratings(
    lead: float = typer.Option(1.0), k: float = typer.Option(24.0),
    first_test: str = typer.Option("2026-04"), min_games: int = typer.Option(10),
) -> None:
    """PandaScore Elo + Kalshi price blend for esports, monthly walk-forward."""
    from albert.pipeline import research_esports_ratings as run

    s, c = _clients()
    typer.echo(run(s, c, lead=lead, k=k, first_test=first_test, min_games=min_games))


@research_app.command("literature")
def research_literature() -> None:
    """Re-test published effects: maker/taker, leg-sum arb, category slopes, reversion, moves."""
    from albert.pipeline import research_literature as run

    typer.echo(run(settings()))


@research_app.command("lead-lag")
def research_lead_lag(
    start: str = typer.Option(None, help="First hour, YYYY-MM-DDTHH (UTC)"),
    end: str = typer.Option(None, help="End hour (exclusive), YYYY-MM-DDTHH (UTC)"),
    move: float = typer.Option(0.02, help="Mid change that counts as a move"),
    window_s: float = typer.Option(5.0, help="...within this many seconds"),
    gap_s: float = typer.Option(30.0, help="At most one move per market per this many seconds"),
) -> None:
    """Paired game moneylines: does Kalshi follow Polymarket intl moves, or the reverse?"""
    from albert.research.leadlag import lead_lag

    typer.echo(lead_lag(settings(), start=start, end=end, move=move, window_s=window_s, gap_s=gap_s))


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


@paper_app.command("l2")
def paper_l2(
    minutes: float = typer.Option(1440.0, help="How long to run"),
    kalshi: int = typer.Option(150, help="Kalshi markets (sampled like `albert record`)"),
    days: float = typer.Option(7.0, help="Kalshi: markets closing within N days"),
    paired_days: float = typer.Option(3.0, help="Also game moneylines played within N days, with "
                                      "their Polymarket intl price as a signal (0 = off)"),
    seed: int = typer.Option(1, help="Seed for the Kalshi market sample"),
    latency_ms: float = typer.Option(100.0),
    writes_per_s: float = typer.Option(10.0),
    holdback_ms: float = typer.Option(500.0, help="Bookkeeping delay to pair trades with deltas"),
    min_spread: float = typer.Option(0.05),
    improve: bool = typer.Option(True),
    cooldown_s: float = typer.Option(300.0),
    max_pos: float = typer.Option(3.0, help="Max net contracts per market"),
    max_gross: float = typer.Option(30.0, help="Stop adding exposure above this many contracts"),
    max_loss: float = typer.Option(5.0, help="Halt when run PnL (marked to mid) < -$X"),
    pm_band: float = typer.Option(None, help="Keep bids this far inside the Polymarket fair value"),
    pm_pull: float = typer.Option(0.0, help="Pull quotes when Polymarket moved this much"),
) -> None:
    """Live paper maker: the backtest's Quoter and queue model on live WebSocket books.
    No orders are sent. Stop with `touch data/paper_l2/STOP` or SIGTERM."""
    import asyncio
    import signal

    from albert.backtest.l2 import Quoter, SimConfig
    from albert.ingest import kalshi_series
    from albert.paper_l2 import Guards, PaperMaker
    from albert.stream import select_kalshi, select_paired
    from albert.venues.kalshi_auth import KalshiSigner

    s, c = _clients()
    series = kalshi_series(s, c)
    k_rows = select_kalshi(series, kalshi, days=days, seed=seed) if kalshi else []
    pairs = select_paired(paired_days) if paired_days else []
    tickers = [m["ticker"] for m in k_rows]
    tickers += [p["kalshi_ticker"] for p in pairs if p["kalshi_ticker"] not in set(tickers)]
    tokens = sorted({p["token_id"] for p in pairs})
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = s.data_dir / "paper_l2"
    run_dir = root / run_id
    (run_dir / "stream" / "pairs").mkdir(parents=True)
    (run_dir / "stream" / "pairs" / f"{run_id}.json").write_text(json.dumps(pairs, indent=1))
    params = {"min_spread": min_spread, "improve": improve, "cooldown_s": cooldown_s,
              "max_pos": max_pos, "pm_band": pm_band, "pm_pull": pm_pull}
    strat = Quoter(category_of={k: v.get("category") for k, v in series.items()}, **params)
    cfg = SimConfig(latency_ms=latency_ms, fee_info=series, writes_per_s=writes_per_s or None)
    guards = Guards(max_gross=max_gross, max_loss=max_loss)
    pm = PaperMaker(run_dir, KalshiSigner.from_env(), strategy=strat, cfg=cfg, pairs=pairs,
                    guards=guards, holdback_ms=holdback_ms, stop_file=root / "STOP")
    pm.write("start", {"run_id": run_id, "minutes": minutes, "params": params,
                       "latency_ms": latency_ms, "writes_per_s": writes_per_s,
                       "holdback_ms": holdback_ms, "guards": vars(guards), "seed": seed,
                       "kalshi_markets": tickers, "pm_tokens": tokens, "pairs": len(pairs)})
    typer.echo(f"paper l2 {run_id}: {len(tickers)} Kalshi markets, {len(tokens)} Polymarket tokens")

    async def main():
        loop = asyncio.get_running_loop()
        def on_signal(name: str) -> None:
            pm.rec.stopped_reason = f"signal {name}"
            pm.rec.stop.set()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, on_signal, sig.name)
        await pm.run(minutes=minutes, kalshi=tickers, pm_tokens=tokens)

    asyncio.run(main())
    end = pm.close()
    typer.echo(f"{end['stopped_reason']}: {end['fills']} fills, {end['orders']} orders, "
               f"equity {end['equity']:+.2f}, halted={end['halted']}  ({run_dir})")


@paper_app.command("l2-report")
def paper_l2_report(run: str = typer.Option(None, help="Run id under data/paper_l2 (default: all)")) -> None:
    """Score live paper maker runs: markouts, mark-to-mid and settlement PnL."""
    from albert.paper_l2 import report

    s, c = _clients()
    typer.echo(report(s, c, run))


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
    strategy: list[str] = typer.Option(["maker"], help="maker, elo, longshot, wallet_follow, xvenue"),
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


@paper_app.command("elo-bootstrap")
def paper_elo_bootstrap(since: str = typer.Option("2026-08-01", help="Fetch settled results from this date")) -> None:
    """Fetch settled results for the Elo series (needed once on a fresh machine/VM)."""
    from albert.ingest import refresh_elo_markets

    s, c = _clients()
    typer.echo(f"elo_markets: {refresh_elo_markets(s, c, since=_day(since))} settled markets upserted")


@paper_app.command("status")
def paper_status() -> None:
    """Paper account equity, cash, and open positions."""
    from albert.paper import status

    typer.echo(status(settings()))


live_app = typer.Typer(help="Real Kalshi orders (demo or prod). Prod needs KALSHI_LIVE_ENABLED=1 "
                       "in .env and --live on every run.", no_args_is_help=True)
app.add_typer(live_app, name="live")
ENV_OPT = typer.Option("demo", "--env", help="demo | prod")


@live_app.command("check")
def live_check(env: str = ENV_OPT) -> None:
    """Read-only account check: key environment, balance, limits, bot vs manual orders, positions."""
    import asyncio

    from albert.live_l2 import BOT_PREFIX
    from albert.venues.kalshi_trade import KalshiTrader, signer_for

    async def main():
        t = KalshiTrader(env, signer_for(env))
        try:
            limits = await t.get("/account/limits")
            bal = await t.balance()
            resting = await t.orders("resting")
            pos = await t.positions()
        finally:
            await t.aclose()
        bot = [o for o in resting if o.client_order_id.startswith(BOT_PREFIX)]
        typer.echo(f"{env}: balance ${bal:.2f}  tier {limits.get('usage_tier')}  "
                   f"write {limits.get('write')}  writable={t.writable}")
        typer.echo(f"resting orders: {len(resting)} ({len(bot)} bot)")
        for o in resting:
            who = "BOT" if o in bot else "manual"
            typer.echo(f"  {who:6} {o.ticker} {o.side} @{o.price:.4f} x{o.remaining:g}")
        typer.echo(f"positions: {pos or 'none'}")

    asyncio.run(main())


@live_app.command("cancel-own")
def live_cancel_own(env: str = ENV_OPT, live: bool = typer.Option(False, "--live")) -> None:
    """Cancel every resting bot order (client_order_id prefix) and nothing else."""
    import asyncio

    from albert.live_l2 import BOT_PREFIX
    from albert.venues.kalshi_trade import KalshiTrader, signer_for

    async def main():
        t = KalshiTrader(env, signer_for(env), allow_prod=live)
        try:
            return await t.cancel_own(BOT_PREFIX)
        finally:
            await t.aclose()

    typer.echo(f"cancelled {asyncio.run(main())}")


@live_app.command("plumbing")
def live_plumbing(market: str = typer.Option(None, help="Demo market (default: pick one)")) -> None:
    """Demo plumbing test: resting order, queue position, cancel, post-only reject, expiry, one
    taker fill, order-group trigger, cancel-own sweep. Saves every response for fixtures."""
    import asyncio

    from albert.live_l2 import plumbing
    from albert.venues.kalshi_trade import KalshiTrader, signer_for

    s = settings()
    out = s.data_dir / "live" / "plumbing" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    async def main():
        signer = signer_for("demo")
        t = KalshiTrader("demo", signer)
        try:
            return await plumbing(t, signer, out, market=market)
        finally:
            await t.aclose()

    steps = asyncio.run(main())
    for st in steps:
        typer.echo(f"{'PASS' if st.ok else 'FAIL'}  {st.name}  {json.dumps(st.detail, default=str)[:200]}")
    typer.echo(f"{sum(st.ok for st in steps)}/{len(steps)} passed  ({out})")
    if not all(st.ok for st in steps):
        raise typer.Exit(1)


@live_app.command("mentions")
def live_mentions(
    env: str = ENV_OPT,
    live: bool = typer.Option(False, "--live", help="Required for prod (with KALSHI_LIVE_ENABLED=1)"),
    minutes: float = typer.Option(1440.0),
    margin: float = typer.Option(0.12, help="NO bid at floor(1 - fair - margin)"),
    max_collateral: float = typer.Option(20.0, help="$ cap: resting NO bids + today's fill cost"),
    max_daily_loss: float = typer.Option(10.0, help="$: no new events once today's fill cost reaches this"),
    max_events: int = typer.Option(3),
    group_limit: int = typer.Option(5, help="Exchange-side: contracts matched per rolling 15 s per event"),
    poll_s: float = typer.Option(60.0),
    force_event: str = typer.Option(None, help="Demo only: post this event now (orders expire in 15 min)"),
) -> None:
    """Pre-speech NO bids on Kalshi mention markets (research/mentions_history). Stop with
    `touch data/live/mentions/STOP` or SIGTERM; every bot order carries an expiration_time."""
    import asyncio
    import signal

    from albert.live_mentions import MentionLimits, MentionsBot
    from albert.venues.kalshi_trade import ENVS, KalshiTrader, signer_for

    if force_event and env != "demo":
        raise typer.BadParameter("--force-event is demo-only")
    s = settings()
    trader = KalshiTrader(env, signer_for(env), allow_prod=live)
    if not trader.writable:
        raise typer.BadParameter(f"{env} is not enabled for orders (prod needs KALSHI_LIVE_ENABLED=1 and --live)")
    root = s.data_dir / "live" / "mentions"
    run_dir = root / env / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    limits = MentionLimits(margin=margin, max_collateral=max_collateral, max_daily_loss=max_daily_loss,
                           max_events=max_events, group_limit=group_limit)
    bot = MentionsBot(s, env, trader, run_dir, limits=limits, base=ENVS[env]["rest"], stop_file=root / "STOP",
                      log=typer.echo)

    async def main():
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, task.cancel)
        bot.write("start", {"env": env, "limits": vars(limits), "minutes": minutes, "force_event": force_event})
        typer.echo(f"live mentions {env}: margin {margin:.2f}, collateral cap ${max_collateral:.2f} ({run_dir})")
        try:
            await bot.run(minutes, poll_s, force_events=[force_event] if force_event else None)
        except asyncio.CancelledError:
            pass
        finally:
            await trader.aclose()

    asyncio.run(main())


@live_app.command("l2")
def live_l2(
    env: str = ENV_OPT,
    live: bool = typer.Option(False, "--live", help="Required for prod (with KALSHI_LIVE_ENABLED=1)"),
    minutes: float = typer.Option(60.0),
    kalshi: int = typer.Option(20, help="Markets to quote (sampled like `albert record`)"),
    markets: str = typer.Option(None, help="Comma-separated tickers instead of sampling"),
    days: float = typer.Option(7.0),
    seed: int = typer.Option(1),
    min_spread: float = typer.Option(0.05),
    improve: bool = typer.Option(True),
    cooldown_s: float = typer.Option(300.0),
    max_pos: float = typer.Option(1.0, help="Max net contracts per market"),
    max_gross: float = typer.Option(10.0),
    max_loss: float = typer.Option(10.0, help="Halt when run PnL (marked to mid) < -$X"),
    group_limit: int = typer.Option(5, help="Exchange-side: contracts matched per rolling 15 s"),
    expiry_s: float = typer.Option(900.0, help="expiration_time on every order"),
) -> None:
    """Live maker with real orders: the paper maker's Quoter, own orders netted from the book,
    guards and exchange-side backstops. Stop with `touch data/live/STOP` or SIGTERM."""
    import asyncio
    import signal

    from albert.backtest.l2 import Quoter, SimConfig
    from albert.ingest import kalshi_series
    from albert.live_l2 import LiveGuards, LiveMaker
    from albert.stream import select_kalshi
    from albert.venues.kalshi_trade import ENVS, KalshiTrader, signer_for

    s, c = _clients()
    series = kalshi_series(s, c)
    signer = signer_for(env)
    trader = KalshiTrader(env, signer, allow_prod=live)
    if not trader.writable:
        raise typer.BadParameter(f"{env} is not enabled for orders (prod needs KALSHI_LIVE_ENABLED=1 and --live)")
    tickers = ([t.strip() for t in markets.split(",") if t.strip()] if markets else
               [m["ticker"] for m in select_kalshi(series, kalshi, days=days, seed=seed,
                                                   rest=ENVS[env]["rest"])])
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = s.data_dir / "live"
    run_dir = root / env / run_id
    run_dir.mkdir(parents=True)
    params = {"min_spread": min_spread, "improve": improve, "cooldown_s": cooldown_s, "max_pos": max_pos}
    strat = Quoter(category_of={k: v.get("category") for k, v in series.items()}, **params)
    guards = LiveGuards(max_gross=max_gross, max_loss=max_loss, group_contracts_limit=group_limit,
                        order_expiry_s=expiry_s)
    lm = LiveMaker(run_dir, env, signer, trader, run_id=run_id, strategy=strat,
                   cfg=SimConfig(fee_info=series, writes_per_s=10.0), guards=guards, pairs=[],
                   stop_file=root / "STOP")

    async def main():
        loop = asyncio.get_running_loop()

        def on_signal(name: str) -> None:
            lm.rec.stopped_reason = f"signal {name}"
            lm.rec.stop.set()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, on_signal, sig.name)
        try:
            ok = await lm.prepare(tickers)
            lm.write("start", {"run_id": run_id, "env": env, "minutes": minutes, "params": params,
                               "guards": vars(guards), "markets": ok, "seed": seed})
            typer.echo(f"live l2 {env} {run_id}: {len(ok)} markets ({len(tickers) - len(ok)} blocked)")
            await lm.run(minutes=minutes, kalshi=ok, pm_tokens=[])
        finally:
            end = await lm.shutdown()
            await trader.aclose()
        return end

    end = asyncio.run(main())
    typer.echo(f"{end['stopped_reason']}: {end['fills']} fills, {end['orders']} orders, "
               f"equity {end['equity']:+.2f}, halted={end['halted']}  ({run_dir})")


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


@app.command("record")
def record_cmd(
    minutes: float = typer.Option(60.0, help="How long to record"),
    kalshi: int = typer.Option(300, help="Kalshi markets to record (0 = none)"),
    pm: int = typer.Option(60, help="Polymarket intl YES tokens to record (0 = none)"),
    days: float = typer.Option(7.0, help="Kalshi: only markets closing within N days"),
    chunk: int = typer.Option(100, help="Markets per WebSocket connection"),
    check_every: float = typer.Option(600.0, help="Seconds between REST book cross-checks"),
    check_sample: int = typer.Option(5, help="Markets per REST cross-check (0 = off)"),
    seed: int = typer.Option(0, help="Seed for the Kalshi market sample"),
    min_free_gb: float = typer.Option(10.0, help="Stop when free disk space drops below this"),
    paired_days: float = typer.Option(0.0, help="Also record Kalshi game moneylines played within "
                                      "N days and their Polymarket intl moneyline (0 = off)"),
) -> None:
    """Record live order books and trades over WebSocket into data/raw/stream (append-only)."""
    import asyncio
    import signal
    from dataclasses import asdict

    from albert.ingest import kalshi_series
    from albert.store import write_manifest
    from pathlib import Path

    from albert.stream import Recorder, record, select_kalshi, select_paired, select_pm
    from albert.venues.kalshi_auth import KalshiSigner

    s, c = _clients()
    k_rows = select_kalshi(kalshi_series(s, c), kalshi, days=days, seed=seed) if kalshi else []
    pm_rows = select_pm(pm) if pm else []
    pairs = select_paired(paired_days) if paired_days else []
    have_k, have_pm = {m["ticker"] for m in k_rows}, {m["token_id"] for m in pm_rows}
    k_rows += [{"ticker": p["kalshi_ticker"], "event_ticker": p["kalshi_ticker"].rsplit("-", 1)[0],
                "category": "Sports"} for p in pairs if p["kalshi_ticker"] not in have_k]
    for p in pairs:
        if p["token_id"] not in have_pm:
            have_pm.add(p["token_id"])
            pm_rows.append({"token_id": p["token_id"], "question": p["question"], "pm_slug": p["pm_slug"]})
    typer.echo(f"recording {len(k_rows)} Kalshi markets and {len(pm_rows)} Polymarket tokens "
               f"for {minutes:g} min")
    if kalshi and not k_rows:
        raise typer.Exit("no eligible Kalshi markets")

    started = datetime.now(UTC)
    if pairs:   # written up front: replays of a still-running recording need the pairing
        pairs_dir = s.raw_dir / "stream" / "pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)
        (pairs_dir / f"{started:%Y%m%dT%H%M%SZ}.json").write_text(json.dumps(pairs, indent=1))

    rec = Recorder(s.raw_dir / "stream", KalshiSigner.from_env() if k_rows else None)

    async def main():
        loop = asyncio.get_running_loop()
        def on_signal(name: str) -> None:
            rec.stopped_reason = f"signal {name}"
            rec.stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, on_signal, sig.name)
        await record(rec, minutes=minutes, kalshi_markets=[m["ticker"] for m in k_rows],
                     pm_tokens=[m["token_id"] for m in pm_rows], chunk=chunk,
                     check_every=check_every, check_sample=check_sample)

    asyncio.run(main())
    files = rec.close()
    stats = {conn: st.as_dict() for conn, st in rec.stats.items()}
    checks = rec.checks
    path = write_manifest(s.data_dir, "record", {
        "minutes": minutes, "days": days, "min_free_gb": min_free_gb, "chunk": chunk, "seed": seed, "started_at": started,
        "kalshi_markets": [{k: m[k] for k in ("ticker", "event_ticker", "category")} for m in k_rows],
        "pm_tokens": pm_rows, "pairs": pairs,
    }, {"files": [asdict(f) for f in files], "stats": stats, "book_checks": checks,
        "stopped_reason": rec.stopped_reason},
        [Path(f.path) for f in files])
    total = sum(f.bytes for f in files)
    gaps = sum(len(st["gaps"]) for st in stats.values())
    matched = sum(ch["match"] for ch in checks)
    typer.echo(f"{len(files)} files, {total / 1e6:.1f} MB compressed, "
               f"{sum(st['frames'] for st in stats.values())} frames, {gaps} seq gaps, "
               f"book checks {matched}/{len(checks)} matched")
    typer.echo(f"manifest: {path}")


@backtest_app.command("l2")
def backtest_l2_cmd(
    start: str = typer.Option(None, help="First hour, YYYY-MM-DDTHH (UTC)"),
    end: str = typer.Option(None, help="End hour (exclusive), YYYY-MM-DDTHH (UTC)"),
    strategy: str = typer.Option("quote", help="Strategy name: quote | join"),
    latency_ms: float = typer.Option(100.0, help="Decision-to-exchange latency for place and cancel"),
    queue: str = typer.Option("risk_averse", help="risk_averse or optimistic cancel attribution"),
    writes_per_s: float = typer.Option(10.0, help="Order write budget per second (0 = unlimited)"),
    min_spread: float = typer.Option(0.05, help="Quote only when the spread is at least this"),
    max_spread: float = typer.Option(1.0, help="quote: and at most this"),
    max_pos: float = typer.Option(3.0, help="Max net contracts per market"),
    improve: bool = typer.Option(True, help="quote: one tick inside the best instead of joining"),
    cooldown_s: float = typer.Option(0.0, help="quote: pause a market this long after a fill"),
    pull_frac: float = typer.Option(0.0, help="quote: cancel when the queue ahead shrinks by this"),
    category: list[str] = typer.Option(None, help="quote: series categories to trade (repeatable)"),
    pm_band: float = typer.Option(None, help="quote: keep bids this far inside the Polymarket "
                                  "intl fair value on paired markets"),
    pm_pull: float = typer.Option(0.0, help="quote: pull when Polymarket moved this much"),
    pm_window_s: float = typer.Option(10.0, help="quote: ...within this many seconds"),
    pm_only: bool = typer.Option(False, help="quote: trade only markets with a Polymarket price"),
    root: str = typer.Option(None, help="Stream root to replay (default data/raw/stream; a paper "
                             "run's data/paper_l2/<run>/stream reconciles that run)"),
) -> None:
    """Replay recorded Kalshi order books and simulate a maker strategy with queue position."""
    from albert.backtest.l2 import backtest_l2

    params: dict = {"min_spread": min_spread, "max_pos": max_pos}
    if strategy == "quote":
        params |= {"max_spread": max_spread, "improve": improve, "cooldown_s": cooldown_s,
                   "pull_frac": pull_frac, "categories": set(category) if category else None,
                   "pm_band": pm_band, "pm_pull": pm_pull, "pm_window_s": pm_window_s,
                   "pm_only": pm_only}
    s, c = _clients()
    typer.echo(backtest_l2(s, c, start=start, end=end, strategy=strategy, latency_ms=latency_ms,
                           queue=queue, writes_per_s=writes_per_s or None, params=params,
                           root=Path(root) if root else None))
