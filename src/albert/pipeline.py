"""Research and backtest pipelines used by the CLI. Each writes a report under data/reports."""

from dataclasses import replace

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from albert.backtest.engine import BacktestResult, monte_carlo, run_backtest
from albert.backtest.strategies import longshot_signals, split_by_time
from albert.config import Settings
from albert.fees import kalshi_fee
from albert.ingest import Clients
from albert.reports import Report
from albert.research import calibration as cal
from albert.research import follow, gaps, wallets
from albert.research import literature as lit
from albert.research.follow import SIGNAL_SCHEMA
from albert.risk import RiskConfig
from albert.store import read_table, write_table
from albert.venues import polymarket as pm


# ---- data helpers -------------------------------------------------------------------------------
def kalshi_obs(s: Settings, horizon_hours: float, anchor: str = "scheduled",
               universe: str = "all") -> pl.DataFrame:
    """``universe="population"`` keeps only the volume-blind game sample (see
    research/literature.full_population_legs); ``"all"`` also includes volume-ranked markets."""
    markets = read_table(s.tables_dir, "kalshi_markets")
    candles = read_table(s.tables_dir, "kalshi_candles")
    obs = cal.kalshi_observations(markets, candles, horizon_hours=horizon_hours, anchor=anchor)
    if universe == "population":
        obs = obs.filter(pl.col("market_id").is_in(lit.full_population_legs(markets).implode()))
    elif universe != "all":
        raise ValueError(f"unknown universe {universe!r}")
    return obs


def fetch_pm_history(s: Settings, c: Clients, windows: pl.DataFrame, bucket_seconds: int = 300,
                     max_requests: int = 5000) -> pl.DataFrame:
    """Price history for (token_id, start, end) windows. Results are cached and appended."""
    rows: list[dict] = []
    for n, w in enumerate(windows.unique(["token_id", "start"]).head(max_requests).iter_rows(named=True)):
        for page in c.pm.price_history(w["token_id"], w["start"], w["end"], bucket_seconds):
            rows.extend(pm.price_history_rows(page.payload, w["token_id"]))
        if n and n % 250 == 0:
            print(f"  price history {n}/{min(windows.height, max_requests)}")
    frame = pl.DataFrame(rows, schema={"token_id": pl.String, "ts": pl.Datetime("us", "UTC"),
                                       "price": pl.Float64})
    write_table(s.tables_dir, "pm_price_history", frame, key=["token_id", "ts"])
    return read_table(s.tables_dir, "pm_price_history")


# ---- research -----------------------------------------------------------------------------------
def research_calibration(s: Settings, *, horizon_hours: float = 2.0, width: float = 0.05) -> Path:
    obs = kalshi_obs(s, horizon_hours)
    if obs.is_empty():
        raise RuntimeError("no Kalshi observations; run `albert ingest kalshi --candles` first")
    r = Report(s.reports_dir, "calibration")
    r.text("Setup", f"Kalshi, decision time = close - {horizon_hours}h, executable YES ask/bid "
                    f"from hourly candles, spread <= 10c. {obs.height} markets, "
                    f"{obs['category'].n_unique()} categories, "
                    f"{obs['decision_ts'].min():%Y-%m-%d} to {obs['decision_ts'].max():%Y-%m-%d}.\n\n"
                    "`edge_yes` = average realized profit per contract from buying YES at the ask, "
                    "after fees. `edge_no` = the same for buying NO at 1 - bid. |t| > 2 is worth a "
                    "look; with many buckets, expect some false positives.")
    overall = cal.calibration_table(obs, width=width, by_category=False)
    r.table("All categories by price bucket", overall, save_as="overall")
    by_cat = cal.calibration_table(obs, width=width, by_category=True)
    r.table("By category (n >= 30 and |t| >= 2 only)", by_cat.filter(
        (pl.col("n") >= 30) & ((pl.col("t_yes").abs() >= 2) | (pl.col("t_no").abs() >= 2))),
        save_as="by_category", max_rows=60)
    r.text("Caveat", "Markets from the same event (strike ladders, player props) are correlated, so "
                     "these standard errors are too small. Treat |t| < 3 as noise.")
    r.text("Market efficiency", f"Brier score of the market mid: {cal.brier(obs):.4f} "
                                f"(0.25 = coin flip).")
    return r.save()


def research_wallets(s: Settings, *, split: datetime, top_k: int = 20) -> Path:
    t = read_table(s.tables_dir, "pm_trades")
    m = read_table(s.tables_dir, "pm_markets")
    rt = wallets.resolved_trades(t, m)
    scored = wallets.score_wallets(rt)
    write_table(s.tables_dir, "wallet_scores", scored, merge=False)
    res = wallets.persistence(rt, split, top_k=top_k)
    r = Report(s.reports_dir, "wallets")
    r.text("Data", f"{t['wallet'].n_unique()} wallets, {t.height} trades, {rt.height} on resolved "
                   f"markets ({rt.height / max(1, t.height):.0%}). Window {t['ts'].min():%Y-%m-%d} to "
                   f"{t['ts'].max():%Y-%m-%d}. Sampling frame: Polymarket leaderboards, which are "
                   "biased toward past winners.")
    r.text("Method", "Every trade is valued as held to resolution, against the price the wallet "
                     "paid. Metrics are per market, not per fill. `shrunk_roi` is the mean per-market "
                     "ROI shrunk toward 0 with 30 pseudo-markets. `mm_like` flags two-sided "
                     "(market-making) wallets, which are excluded from the follow test.")
    r.table("Top wallets (full window)", scored.select(
        "wallet", "markets", "fills", "pnl", "risk", "roi", "shrunk_roi", "t_stat",
        "market_win_rate", "median_entry_price", "mm_like"), save_as="wallet_scores", max_rows=40)
    r.text("Out-of-sample persistence", res.summary() + "\n\nIf top-k ROI in B is not clearly above "
           "all-wallet ROI in B, past performance does not predict future performance and "
           "copy-trading has no basis.")
    r.table("Window A score vs window B result", res.evaluated.sort("score_a", descending=True),
            save_as="persistence", max_rows=40)
    return r.save()


def build_follow_signals(s: Settings, c: Clients, *, split: datetime, top_k: int = 20,
                         delay_minutes: float = 5.0, half_spread: float = 0.01,
                         min_usd: float = 50.0, max_requests: int = 3000) -> pl.DataFrame:
    t = read_table(s.tables_dir, "pm_trades")
    m = read_table(s.tables_dir, "pm_markets")
    rt = wallets.resolved_trades(t, m)
    first = rt.group_by("wallet", "condition_id").agg(pl.col("ts").min().alias("first_ts"))
    rt = rt.join(first, on=["wallet", "condition_id"])
    a = wallets.score_wallets(rt.filter(pl.col("first_ts") < split))
    a = a.filter(~pl.col("mm_like") & (pl.col("shrunk_roi") > 0)).head(top_k)
    chosen = a["wallet"].to_list()
    edges = follow.wallet_edges(rt.filter(pl.col("first_ts") < split), chosen)
    rt_b = rt.filter(pl.col("first_ts") >= split)
    cands = follow.follow_candidates(rt_b, edges, min_usd=min_usd).join(
        m.select("condition_id", "event_slug"), on="condition_id", how="left", suffix="_m"
    ).with_columns(pl.coalesce("event_slug", "event_slug_m").alias("event_slug"))
    windows = cands.select("token_id", (pl.col("ts") - timedelta(minutes=10)).alias("start"),
                           (pl.col("ts") + timedelta(hours=3)).alias("end"))
    history = fetch_pm_history(s, c, windows, bucket_seconds=300, max_requests=max_requests)
    sig = follow.follow_signals(cands, history, delay_minutes=delay_minutes, half_spread=half_spread)
    write_table(s.tables_dir, "signals_wallet_follow", sig, merge=False)
    return sig


def research_gaps(s: Settings, c: Clients, *, half_spread: float = 0.01) -> Path:
    matches = read_table(s.tables_dir, "match_kalshi_pm")
    candles = read_table(s.tables_dir, "kalshi_candles")
    pm_markets = read_table(s.tables_dir, "pm_markets")
    span = candles.join(matches.select("kalshi_ticker", "pm_condition_id", "pm_outcome_index"),
                        left_on="ticker", right_on="kalshi_ticker").group_by(
        "pm_condition_id", "pm_outcome_index").agg(pl.col("end_ts").min().alias("start"),
                                                   pl.col("end_ts").max().alias("end"))
    windows = span.join(pm_markets.select("condition_id", "token_ids"), left_on="pm_condition_id",
                        right_on="condition_id").select(
        pl.col("token_ids").list.get(pl.col("pm_outcome_index")).alias("token_id"), "start", "end")
    history = fetch_pm_history(s, c, windows, bucket_seconds=3600)
    g = gaps.gap_table(matches, candles, history, pm_markets,
                       read_table(s.tables_dir, "kalshi_markets"), half_spread=half_spread)
    r = Report(s.reports_dir, "gaps")
    r.text("Setup", "Exactly matched Kalshi and Polymarket game markets. Kalshi executable prices "
                    "come from hourly candles; Polymarket intl prices stand in for Polymarket US with "
                    f"a {half_spread:.2f} half spread and Polymarket US fees. Only observations at least "
                    "4h before the Kalshi close, with unresolved Polymarket prices (2c-98c), are "
                    "kept, because in-play and post-resolution prints create fake arbitrage. "
                    "Indicative only.")
    r.json("Summary", gaps.gap_summary(g))
    r.table("Largest locked-in opportunities", g.sort("best_arb", descending=True), save_as="gaps")
    return r.save()


# ---- backtests ----------------------------------------------------------------------------------
def _plot_equity(result: BacktestResult, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not result.equity:
        return False
    xs, ys = zip(*result.equity)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, ys, color="#2a6f97", linewidth=1.4)
    ax.axhline(result.start_equity, color="#999", linewidth=0.8, linestyle="--")
    ax.set_title("Equity (all venues)")
    ax.set_ylabel("$")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def research_literature(s: Settings, *, reversion_grid: tuple = ((24, 6), (24, 24), (72, 24),
                                                                (72, 72))) -> Path:
    """Re-test published prediction-market effects on our data (see research/literature.py)."""
    r = Report(s.reports_dir, "research-literature")
    markets = read_table(s.tables_dir, "kalshi_markets")
    candles = read_table(s.tables_dir, "kalshi_candles")
    trades = read_table(s.tables_dir, "kalshi_trades")
    summary: dict[str, object] = {}

    # Only the seeded random sample (`ingest kalshi-detail`, volume band 500-200k). Markets
    # picked as the top-N by volume are excluded: ranking on final volume selects upsets.
    band = markets.filter(pl.col("volume").is_between(500, 200_000))["ticker"].implode()
    trades = trades.filter(pl.col("ticker").is_in(band))
    legs = lit.trade_legs(trades, markets)
    mt = lit.maker_taker_returns(legs)
    r.text("1. Makers vs takers (Bürgi, Deng & Whelan 2025)",
           f"{trades['ticker'].n_unique()} randomly sampled markets (volume 500-200k contracts), "
           f"{trades.height} trades (newest 5,000 per market), each counted once for the taker and "
           "once for the maker, held to settlement, after fees. ROI = PnL / dollars staked; SE "
           "clustered by event.")
    r.table("ROI by role", mt["by_role"])
    r.table("ROI by role and price band", mt["by_role_band"], save_as="maker_taker_bands")
    r.table("Maker purchases at >= 50c, by category", mt["maker_favorites_by_category"])
    r.table("Purchases at >= 50c by role and month", mt["favorites_by_role_month"])
    r.table("Maker purchases at >= 50c by category and month",
            mt["maker_favorites_by_category_month"], max_rows=40)
    summary["maker_taker"] = mt["by_role"].to_dicts()

    try:
        events = read_table(s.tables_dir, "kalshi_events")
    except FileNotFoundError:
        events = None
    if events is not None:
        hours = lit.leg_sum_arbitrage(candles, markets, events)
        ls = lit.leg_sum_summary(hours)
        summary["leg_sum"] = ls
        r.text("2. Sum-of-legs arbitrage (Saguillo et al. 2025)",
               "Mutually exclusive, complete events only; every leg quoted in the same hourly "
               "candle; taker fees on a 100-lot per leg.")
        r.json("Summary", ls)
        if not hours.is_empty():
            per_event = hours.group_by("event_ticker").agg(
                pl.len().alias("hours"), pl.first("legs"), pl.first("n_yes"),
                pl.col("long_profit").max().alias("best_long"),
                pl.col("short_profit").max().alias("best_short"),
                pl.col("long_realized").max().alias("best_long_realized"),
            ).sort("best_long", descending=True)
            r.table("Events by best indicative long profit", per_event, max_rows=15)
            hours.write_parquet(r.dir / "leg_sum_hours.parquet")

    r.text("3. Calibration slope by category (arXiv:2602.19520)",
           "Logistic fit y ~ a + b*logit(mid). b > 1 means prices are too close to 50% "
           "(under-confident; favorites cheap), b < 1 means over-confident. Event bootstrap SE.")
    obs_by_h = {h: kalshi_obs(s, h) for h in (2.0, 6.0, 24.0, 26.0, 72.0)}
    slopes = []
    for h in (2.0, 24.0):
        sl = lit.calibration_slopes(obs_by_h[h]).with_columns(pl.lit(h).alias("horizon_h"))
        slopes.append(sl)
        r.table(f"Kalshi, {h:g}h before close", sl)
    try:
        pm_obs = cal.pm_observations(read_table(s.tables_dir, "pm_markets"),
                                     read_table(s.tables_dir, "pm_price_history"), horizon_hours=2.0)
    except FileNotFoundError:
        pm_obs = None
    if pm_obs is not None and pm_obs.height:
        sl = lit.calibration_slopes(pm_obs, min_n=150).with_columns(pl.lit(2.0).alias("horizon_h"))
        r.table("Polymarket intl (first tag), 2h before resolution", sl)
        slopes.append(sl.with_columns(pl.concat_str(pl.lit("pm:"), "category").alias("category")))
    summary["slopes"] = pl.concat(slopes).to_dicts()

    pop = lit.full_population_legs(markets).implode()
    r.text("3b. Sports moneylines: full population vs volume-selected sample",
           "The volume-ranked candle sample over-represents upsets (they draw in-game volume), "
           "which makes favorites look overpriced. Full population = first leg of every "
           f"{'/'.join(lit.GAME_SERIES)} game, chosen without volume. Decision at scheduled "
           "expiry - horizon. Slope test, then the slope strategy fit/tested on each sample "
           "(flat 1-contract bets, edge >= 3c after fees, split 2026-10-01).")
    split_dt = datetime(2026, 10, 1, tzinfo=UTC)
    pop_rows = []
    for h in (24.0, 6.0):
        o = obs_by_h[h] if h in obs_by_h else kalshi_obs(s, h)
        games = o.filter(pl.col("market_id").str.contains(r"^(" + "|".join(lit.GAME_SERIES) + r")-"))
        for label, g in (("full population", games.filter(pl.col("market_id").is_in(pop))),
                         ("volume-selected", games.filter(~pl.col("market_id").is_in(pop)))):
            if g.height < 100:
                continue
            sl = lit.calibration_slopes(g, min_n=100).filter(pl.col("category") == "ALL").row(0, named=True)
            fav = g.filter(pl.col("mid") > 0.55)
            train, test = split_by_time(g, split_dt)
            flat = lit.flat_signal_returns(lit.slope_signals(train.with_columns(pl.lit("G").alias("category")),
                                                             test.with_columns(pl.lit("G").alias("category")),
                                                             min_n=100))
            fr = flat.row(0, named=True) if not flat.is_empty() else {}
            pop_rows.append({"horizon_h": h, "sample": label, "markets": g.height,
                             "slope": sl["slope"], "slope_se": sl["slope_se"],
                             "favorite_residual": float((fav["y"] - fav["mid"]).mean()),
                             "strategy_trades": fr.get("n", 0), "strategy_roi": fr.get("roi"),
                             "strategy_t": fr.get("t")})
    pop_table = pl.DataFrame(pop_rows)
    r.table("Sports moneylines", pop_table)
    summary["sports_population"] = pop_rows

    rows = []
    for lookback, hold in reversion_grid:
        t = lit.rolling_min_reversion(candles, markets, lookback_hours=lookback, hold_hours=hold)
        if t.is_empty():
            continue
        st = lit.cluster_ratio(t.with_columns(pl.lit(1).alias("_all")), ["_all"]).row(0, named=True)
        rows.append({"lookback_h": lookback, "hold_h": hold, "trades": t.height,
                     "markets": t["ticker"].n_unique(), "gross_per_contract": float(t["gross"].mean()),
                     "net_roi": st["roi"], "se": st["se"], "t": st["t"],
                     "win_rate": float((t["pnl"] > 0).mean()),
                     "settled_share": float(t["settled"].mean())})
    rev = pl.DataFrame(rows)
    summary["reversion"] = rows
    r.text("4. Rolling-minimum mean reversion (Quantpedia, Polymarket)",
           "Buy YES at the ask when the hourly mid is at its trailing minimum; sell at the bid "
           "after the hold (or settle). Taker fees both ways. One position per market at a time.")
    r.table("Grid (hours)", rev)

    r.text("5. Pre-close move reversal (Moskowitz 2021)",
           "move = mid(late) - mid(early) for the same market; residual = outcome - mid(late). "
           "Slope < 0: moves over-shoot and partly reverse; > 0: they under-shoot (momentum).")
    mv_rows = []
    for late, early in ((2.0, 26.0), (24.0, 72.0)):
        mv = lit.move_reversal_slopes(lit.move_reversal(obs_by_h[late], obs_by_h[early]))
        mv = mv.with_columns(pl.lit(f"{early:g}h->{late:g}h").alias("window"))
        mv_rows.append(mv)
        r.table(f"Move from {early:g}h to {late:g}h before close", mv)
    summary["move_reversal"] = pl.concat(mv_rows).to_dicts()
    r.json("Summary", summary)
    return r.save()


def backtest_maker(s: Settings, *, horizons: tuple[float, ...] = (24.0, 6.0), size: int = 5,
                   cash: float = 50.0, categories: tuple[str, ...] = ("Climate and Weather", "Sports"),
                   lo: float = 0.50, hi: float = 0.95, ttl_hours: float | None = None,
                   exclude_tickers: Path | None = None, queue: str = "any",
                   universe: str = "sample", anchor: str = "scheduled",
                   series: list[str] | None = None) -> Path:
    """Resting-bid backtest on Kalshi favorites with a trade-driven fill model (backtest/maker.py)."""
    from albert.backtest import maker

    markets = read_table(s.tables_dir, "kalshi_markets")
    candles = read_table(s.tables_dir, "kalshi_candles")
    windows = None
    if universe == "population":
        # One leg of every game, with complete trades in each order window: no volume selection.
        windows = read_table(s.tables_dir, "kalshi_trade_windows")
        trades = read_table(s.tables_dir, "kalshi_trades").filter(
            pl.col("ticker").is_in(windows["ticker"].implode()))
    elif universe == "sample":
        # The random-sample volume band (500-200k). NOTE: biased toward favorites winning,
        # because the cap drops the busiest games and upsets draw volume (README section 5).
        band = markets.filter(pl.col("volume").is_between(500, 200_000))["ticker"].implode()
        trades = read_table(s.tables_dir, "kalshi_trades").filter(pl.col("ticker").is_in(band))
    else:
        raise ValueError(f"unknown universe {universe!r}")
    if series:
        markets = markets.filter(pl.col("series_ticker").is_in(series))
        trades = trades.filter(pl.col("ticker").is_in(markets["ticker"].implode()))
    holdout_note = ""
    if exclude_tickers is not None:
        # Holdout: drop every market an earlier (in-sample) run could have seen.
        seen = pl.read_parquet(exclude_tickers)["ticker"].implode()
        trades = trades.filter(~pl.col("ticker").is_in(seen))
        holdout_note = f"\n- HOLDOUT: excluding {pl.read_parquet(exclude_tickers).height} in-sample tickers ({exclude_tickers})"
    r = Report(s.reports_dir, "backtest-maker")
    r.text("Setup", f"- categories {list(categories)}, favorite bid in [{lo}, {hi}], {size} contracts "
                    f"per order for the flat test, ttl {ttl_hours or 'until close'}, queue mode {queue}, "
                    f"universe {universe}, anchor {anchor}, series {series or 'all'}\n"
                    f"- {trades['ticker'].n_unique()} randomly sampled markets with trades "
                    "(volume 500-200k)\n- fill = later real taker trades selling into our price "
                    "(at it if we improved the best bid, through it if we joined); held to settlement; "
                    "maker fee.\n- taker benchmark = same market and side bought at the ask at the "
                    "decision time, taker fee." + holdout_note)
    rows, all_orders = [], []
    for h in horizons:
        orders = maker.maker_orders(markets, candles, trades, horizon_hours=h, categories=categories,
                                    lo=lo, hi=hi, ttl_hours=ttl_hours, queue=queue,
                                    windows=windows, anchor=anchor)
        if orders.is_empty():
            continue
        fills = maker.with_pnl(maker.simulate_fills(orders, trades, size=size)).with_columns(
            pl.lit(h).alias("horizon_h"), pl.col("decision_ts").dt.strftime("%Y-%m").alias("month"))
        all_orders.append(fills)
        taker = maker.taker_benchmark(orders).with_columns(pl.lit("all").alias("_k"))
        flat = maker.summarize(fills.with_columns(pl.lit("all").alias("_k")), ["_k"])
        tk = maker.summarize(taker, ["_k"]).row(0, named=True)
        fr = flat.row(0, named=True) if not flat.is_empty() else {}
        rows.append({"horizon_h": h, "orders": orders.height,
                     "fill_rate": float((fills["filled"] > 0).mean()),
                     "full_fill_rate": float((fills["filled"] == size).mean()),
                     "improved_share": float(orders["improved"].mean()),
                     "maker_roi": fr.get("roi"), "maker_t": fr.get("t"), "maker_events": fr.get("clusters"),
                     "maker_pnl_per_order": float(fills["pnl"].sum() / orders.height),
                     "taker_roi": tk["roi"], "taker_t": tk["t"]})
        r.table(f"{h:g}h: maker ROI by category", maker.summarize(fills, ["category"]))
        r.table(f"{h:g}h: maker ROI by month", maker.summarize(fills, ["month"]))
        r.table(f"{h:g}h: maker ROI by queue position", maker.summarize(fills, ["improved"]))
        r.table(f"{h:g}h: maker ROI by bid band", maker.summarize(
            fills.with_columns(pl.col("bid").cut([0.6, 0.7, 0.8, 0.9]).cast(pl.String).alias("bid_band")),
            ["bid_band"]))
        r.table(f"{h:g}h: taker benchmark by category", maker.summarize(taker, ["category"]))
    r.table("Summary (flat size)", pl.DataFrame(rows))
    if all_orders:
        pl.concat(all_orders).write_parquet(r.dir / "orders.parquet")
    bank = {}
    for h in horizons:
        orders = maker.maker_orders(markets, candles, trades, horizon_hours=h, categories=categories,
                                    lo=lo, hi=hi, ttl_hours=ttl_hours, queue=queue,
                                    windows=windows, anchor=anchor)
        if orders.is_empty():
            continue
        b = maker.run_maker_bankroll(orders, trades, cash=cash)
        bank[f"{h:g}h"] = {"start": b.start_cash, "end": round(b.end_cash, 2),
                           "orders_placed": b.orders_placed, "orders_filled": b.orders_filled,
                           "skipped_no_cash": b.skipped_no_cash, "max_drawdown": round(b.max_drawdown, 4)}
    r.json(f"Bankroll simulation (${cash:g} Kalshi, 5% / $10 per order, 10% per event)", bank)
    return r.save()


def research_sports(s: Settings, *, series: list[str], leads: tuple[float, ...] = (1.0, 0.25),
                    split: datetime = datetime(2026, 10, 1, tzinfo=UTC), min_edge: float = 0.03,
                    anchor: str = "start") -> Path:
    """Pre-game pricing of (minor) sports series on a volume-blind sample.

    Decision = ticker start time - lead. Only markets with a two-sided quote at that moment
    are used, which is known at decision time (unlike final volume). Per series: calibration
    slope, favourite residual (taker at the ask), and the per-series slope strategy fit before
    ``split`` and traded after it (flat 1-contract bets, SEs clustered by date).
    """
    markets = read_table(s.tables_dir, "kalshi_markets").filter(pl.col("series_ticker").is_in(series))
    candles = read_table(s.tables_dir, "kalshi_candles")
    r = Report(s.reports_dir, "research-sports")
    r.text("Setup", f"- series: {', '.join(series)}\n- decision = {anchor} anchor - lead; quote must be "
                    f"two-sided with spread <= 10c\n- strategy: per-series logistic recalibration fit "
                    f"before {split:%Y-%m-%d}, flat bets after it, edge >= {min_edge} after taker fee")
    out = []
    for lead in leads:
        obs = cal.kalshi_observations(markets, candles, horizon_hours=lead, anchor=anchor,
                                      staleness_hours=2.0).with_columns(
            pl.col("market_id").str.split("-").list.first().alias("category"))
        if obs.is_empty():
            continue
        slopes = lit.calibration_slopes(obs, min_n=80, boot=100)
        fav = obs.with_columns(
            pl.when(pl.col("mid") >= 0.5).then(pl.col("yes_ask")).otherwise(1 - pl.col("yes_bid")).alias("fav_price"),
            pl.when(pl.col("mid") >= 0.5).then(pl.col("y")).otherwise(1 - pl.col("y")).alias("fav_win"),
            pl.col("event_id").str.split("-").list.get(1).str.slice(0, 7).alias("date"))
        favs = fav.group_by("category").agg(
            pl.len().alias("n"), pl.col("fav_price").mean().alias("avg_fav_ask"),
            (pl.col("fav_win") - pl.col("fav_price")).mean().alias("fav_edge_gross"),
            ((pl.col("fav_win") - pl.col("fav_price")).std() / pl.len().sqrt()).alias("se"))
        train, test = split_by_time(obs, split)
        sig = lit.slope_signals(train, test, min_n=80)
        rows = []
        for (ser,), g in sig.group_by(pl.col("market_id").str.split("-").list.first()):
            f = lit.flat_signal_returns(g, min_edge=min_edge,
                                        cluster=pl.col("event_id").str.split("-").list.get(1).str.slice(0, 7))
            if not f.is_empty():
                rows.append({"category": ser, **{k_: f.row(0, named=True)[k_] for k_ in ("n", "clusters", "roi", "t")}})
        strat = pl.DataFrame(rows) if rows else pl.DataFrame(schema={"category": pl.String})
        table = slopes.join(favs, on="category", how="left").join(
            strat.rename({c_: f"strat_{c_}" for c_ in strat.columns if c_ != "category"}), on="category", how="left")
        r.table(f"Lead {lead:g}h before start", table.sort("n", descending=True, nulls_last=True), max_rows=40)
        out.append(table.with_columns(pl.lit(lead).alias("lead_h")))
    if out:
        pl.concat(out, how="diagonal_relaxed").write_parquet(r.dir / "sports.parquet")
    return r.save()


def research_esports_reference(s: Settings, c: Clients, *, leads: tuple[float, ...] = (6.0, 2.0, 1.0, 0.25),
                               margins: tuple[float, ...] = (0.0, 0.02, 0.05)) -> Path:
    """Polymarket intl (data only) as a reference price for Kalshi esports match winners.

    For each matched leg and lead: Kalshi quote (last hourly candle <= t) and Polymarket price
    (last history point <= t). Tests whether the gap predicts the Kalshi outcome, and a fixed
    rule: buy on Kalshi when the Polymarket price beats Kalshi's ask (+ fee) by ``margin``.
    No parameters are fitted, so every month is out of sample.
    """
    from albert.ingest import _kalshi_candles, IngestResult
    from albert.matching import match_kalshi_pm_esports

    markets = read_table(s.tables_dir, "kalshi_markets")
    pm_markets = read_table(s.tables_dir, "pm_markets")
    matches = match_kalshi_pm_esports(markets, pm_markets)
    write_table(s.tables_dir, "match_kalshi_pm_esports", matches, merge=False)
    have = read_table(s.tables_dir, "kalshi_candles")["ticker"].unique().implode()
    missing = markets.filter(pl.col("ticker").is_in(matches["kalshi_ticker"].implode())
                             & ~pl.col("ticker").is_in(have))
    if missing.height:
        _kalshi_candles(s, c, missing, c.kalshi.cutoff(), 60, IngestResult())
    tokens = matches.join(pm_markets.select("condition_id", "token_ids"), left_on="pm_condition_id",
                          right_on="condition_id").with_columns(
        pl.col("token_ids").list.get(pl.col("pm_outcome_index")).alias("token_id"))
    windows = tokens.select("token_id", (pl.col("pm_start") - timedelta(hours=max(leads) + 2)).alias("start"),
                            pl.col("pm_start").alias("end"))
    history = fetch_pm_history(s, c, windows, bucket_seconds=300)
    candles = read_table(s.tables_dir, "kalshi_candles").filter(
        pl.col("ticker").is_in(matches["kalshi_ticker"].implode()))
    km = markets.select("ticker", "fee_type", "fee_multiplier")
    r = Report(s.reports_dir, "research-esports-reference")
    r.text("Setup", f"- {matches['kalshi_event'].n_unique()} Kalshi esports matches matched to Polymarket "
                    f"intl (winner agreement checked), {matches.height} legs\n- Polymarket is a data "
                    "signal only; trades are on Kalshi at the hourly candle's ask/bid, taker fee\n"
                    "- SEs clustered by date")
    rows, all_obs = [], []
    for lead in leads:
        base = tokens.with_columns((pl.col("start") - timedelta(hours=lead)).alias("t")).sort("t")
        q = base.join_asof(candles.sort("end_ts").select(pl.col("ticker").alias("kalshi_ticker"), "end_ts",
                                                         "yes_bid_close", "yes_ask_close"),
                           left_on="t", right_on="end_ts", by="kalshi_ticker",
                           strategy="backward", tolerance=timedelta(hours=2))
        q = q.join_asof(history.sort("ts").rename({"price": "pm"}), left_on="t", right_on="ts", by="token_id",
                        strategy="backward", tolerance=timedelta(hours=1))
        q = q.filter(pl.col("yes_bid_close").is_not_null() & pl.col("yes_ask_close").is_not_null()
                     & pl.col("pm").is_between(0.02, 0.98)
                     & (pl.col("yes_ask_close") - pl.col("yes_bid_close")).is_between(0.001, 0.15))
        q = q.join(km, left_on="kalshi_ticker", right_on="ticker").with_columns(
            ((pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2).alias("mid"),
            (pl.col("kalshi_result") == "yes").cast(pl.Float64).alias("y"),
            pl.col("t").dt.strftime("%Y-%m-%d").alias("date"),
            pl.col("t").dt.strftime("%Y-%m").alias("month"), pl.lit(lead).alias("lead_h"))
        if q.is_empty():
            continue
        fy = [kalshi_fee(100, a, fee_type=ft or "quadratic", multiplier=fm) / 100
              for a, ft, fm in q.select("yes_ask_close", "fee_type", "fee_multiplier").iter_rows()]
        fn = [kalshi_fee(100, 1 - b, fee_type=ft or "quadratic", multiplier=fm) / 100
              for b, ft, fm in q.select("yes_bid_close", "fee_type", "fee_multiplier").iter_rows()]
        q = q.with_columns(pl.Series("fee_yes", fy), pl.Series("fee_no", fn)).with_columns(
            (pl.col("pm") - pl.col("yes_ask_close") - pl.col("fee_yes")).alias("edge_yes"),
            ((1 - pl.col("pm")) - (1 - pl.col("yes_bid_close")) - pl.col("fee_no")).alias("edge_no"))
        all_obs.append(q)
        gap = (q["pm"] - q["mid"]).to_numpy()
        res = (q["y"] - q["mid"]).to_numpy()
        gc = gap - gap.mean()
        beta = float((gc * (res - res.mean())).sum() / (gc ** 2).sum()) if (gc ** 2).sum() else float("nan")
        row = {"lead_h": lead, "legs": q.height, "events": q["kalshi_event"].n_unique(),
               "mean_abs_gap": float(abs(gap).mean()), "gap_beta": beta,
               "brier_kalshi": float(((q["mid"] - q["y"]) ** 2).mean()),
               "brier_pm": float(((q["pm"] - q["y"]) ** 2).mean())}
        for mg in margins:
            yes = q.filter(pl.col("edge_yes") >= mg).select(
                "kalshi_event", "date", "month", (pl.col("y") - pl.col("yes_ask_close") - pl.col("fee_yes")).alias("pnl"),
                (pl.col("yes_ask_close") + pl.col("fee_yes")).alias("stake"))
            no = q.filter(pl.col("edge_no") >= mg).select(
                "kalshi_event", "date", "month", ((1 - pl.col("y")) - (1 - pl.col("yes_bid_close")) - pl.col("fee_no")).alias("pnl"),
                ((1 - pl.col("yes_bid_close")) + pl.col("fee_no")).alias("stake"))
            # One bet per match: legs mirror each other.
            bets = pl.concat([yes, no]).unique("kalshi_event", keep="first").rename({"date": "event_ticker"})
            if bets.height:
                st = lit.cluster_ratio(bets.with_columns(pl.lit(1).alias("k")), ["k"]).row(0, named=True)
                row[f"m{mg:g}_bets"], row[f"m{mg:g}_roi"], row[f"m{mg:g}_t"] = st["n"], st["roi"], st["t"]
                if mg == margins[1]:
                    r.table(f"Lead {lead:g}h, margin {mg:g}: by month",
                            lit.cluster_ratio(bets, ["month"]).select("month", "n", "clusters", "roi", "se", "t"))
        rows.append(row)
    r.table("Summary", pl.DataFrame(rows))
    if all_obs:
        pl.concat(all_obs, how="diagonal_relaxed").write_parquet(r.dir / "observations.parquet")
    return r.save()


def research_elo(s: Settings, *, series: list[str], lead: float = 1.0,
                 split: datetime = datetime(2026, 10, 1, tzinfo=UTC),
                 margins: tuple[float, ...] = (0.03, 0.06), cash: float = 50.0,
                 cfg: RiskConfig = RiskConfig()) -> Path:
    """Market + Elo blend on minor sports (research/ratings.py).

    Ratings come from every settled event (no look-ahead). The blend
    P = sigmoid(a + b*logit(mid) + c*logit(elo)) is fit on games before ``split`` and
    evaluated on games after it: Brier vs the market, and flat bets where the blend beats
    the Kalshi ask/bid plus taker fee by ``margin`` (SEs clustered by date).
    """
    import numpy as np

    from albert.research.ratings import elo_predictions

    markets = read_table(s.tables_dir, "kalshi_markets").filter(pl.col("series_ticker").is_in(series))
    preds = elo_predictions(markets)
    legs = markets.filter(pl.col("ticker").is_in(preds["ticker"].implode()))
    candles = read_table(s.tables_dir, "kalshi_candles").filter(pl.col("ticker").is_in(legs["ticker"].implode()))
    obs = cal.kalshi_observations(legs, candles, horizon_hours=lead, anchor="start", staleness_hours=2.0)
    obs = obs.join(preds.select(pl.col("ticker").alias("market_id"), "elo_p", "n_min",
                                pl.col("series").alias("ser")), on="market_id").filter(pl.col("n_min") >= 10)
    train, test = obs.filter(pl.col("decision_ts") < split), obs.filter(pl.col("decision_ts") >= split)
    def lg(col: pl.Series) -> np.ndarray:
        x = np.clip(col.to_numpy(), 1e-3, 1 - 1e-3)
        return np.log(x / (1 - x))
    X = np.column_stack([np.ones(train.height), lg(train["mid"]), lg(train["elo_p"])])
    w = np.array([0.0, 1.0, 0.0])
    y = train["y"].to_numpy()
    for _ in range(60):
        pr = 1 / (1 + np.exp(-X @ w))
        w -= np.linalg.solve(-(X.T * (pr * (1 - pr))) @ X - 1e-6 * np.eye(3), X.T @ (y - pr))
    Xt = np.column_stack([np.ones(test.height), lg(test["mid"]), lg(test["elo_p"])])
    test = test.with_columns(pl.Series("p", 1 / (1 + np.exp(-Xt @ w))))
    fy = [kalshi_fee(100, a, fee_type=ft or "quadratic", multiplier=fm) / 100
          for a, ft, fm in test.select("yes_ask", "fee_type", "fee_multiplier").iter_rows()]
    fn = [kalshi_fee(100, 1 - b, fee_type=ft or "quadratic", multiplier=fm) / 100
          for b, ft, fm in test.select("yes_bid", "fee_type", "fee_multiplier").iter_rows()]
    test = test.with_columns(pl.Series("fee_yes", fy), pl.Series("fee_no", fn),
                             pl.col("decision_ts").dt.strftime("%Y-%m-%d").alias("date"),
                             pl.col("decision_ts").dt.strftime("%G-W%V").alias("week"))
    r = Report(s.reports_dir, "research-elo")
    r.text("Setup", f"- series: {', '.join(series)}\n- decision: ticker start - {lead:g}h; both sides >= 10 "
                    f"rated games\n- blend fit on {train.height} games before {split:%Y-%m-%d}: "
                    f"P = sigmoid({w[0]:+.3f} {w[1]:+.3f}*logit(mid) {w[2]:+.3f}*logit(elo))\n"
                    f"- test: {test.height} games")
    yt = test["y"].to_numpy()
    r.json("Test Brier", {"market": float(np.mean((test["mid"].to_numpy() - yt) ** 2)),
                          "elo": float(np.mean((test["elo_p"].to_numpy() - yt) ** 2)),
                          "blend": float(np.mean((test["p"].to_numpy() - yt) ** 2))})
    sig_rows = []
    for mg in margins:
        yes = test.filter(pl.col("p") - pl.col("yes_ask") - pl.col("fee_yes") >= mg).with_columns(
            pl.lit("yes").alias("side"), pl.col("yes_ask").alias("price"), pl.col("p").alias("prob"),
            pl.col("y").alias("payoff"), pl.col("fee_yes").alias("fee"))
        no = test.filter((1 - pl.col("p")) - (1 - pl.col("yes_bid")) - pl.col("fee_no") >= mg).with_columns(
            pl.lit("no").alias("side"), (1 - pl.col("yes_bid")).alias("price"), (1 - pl.col("p")).alias("prob"),
            (1 - pl.col("y")).alias("payoff"), pl.col("fee_no").alias("fee"))
        bets = pl.concat([yes, no]).with_columns(
            (pl.col("payoff") - pl.col("price") - pl.col("fee")).alias("pnl"),
            (pl.col("price") + pl.col("fee")).alias("stake"), pl.col("date").alias("event_ticker"))
        if bets.is_empty():
            continue
        r.table(f"Margin {mg:g}: all", lit.cluster_ratio(bets.with_columns(pl.lit("all").alias("k")), ["k"]))
        r.table(f"Margin {mg:g}: by series", lit.cluster_ratio(bets, ["ser"]), max_rows=25)
        wk = lit.cluster_ratio(bets, ["week"])
        r.table(f"Margin {mg:g}: by week ({int((wk['pnl'] > 0).sum())} of {wk.height} positive)", wk)
        if mg == margins[0]:
            sig_rows.append(bets)
    if sig_rows:
        b = sig_rows[0]
        signals = b.select(pl.lit("elo").alias("strategy"), "venue", "event_id", pl.col("decision_ts").alias("signal_ts"),
                           "resolve_ts", pl.lit(None, dtype=pl.Float64).alias("available"), "fee_type",
                           "fee_multiplier", "market_id", "side", "price", "prob", "payoff",
                           pl.col("ser").alias("note")).select(list(SIGNAL_SCHEMA)).cast(SIGNAL_SCHEMA)
        result = run_backtest(signals, cash={"kalshi": cash, "pmus": 0.0},
                              cfg=replace(cfg, min_edge=margins[0]))
        r.json(f"Bankroll (${cash:g}, quarter Kelly, margin {margins[0]:g})", result.metrics())
        r.json("Monte Carlo", monte_carlo(result))
        if _plot_equity(result, r.dir / "equity.png"):
            r.image("Equity curve", "equity.png")
    test.write_parquet(r.dir / "test.parquet")
    return r.save()


def research_elo_walk(s: Settings, *, lead: float = 1.0, margins: tuple[float, ...] = (0.03, 0.06),
                      k: float = 32.0, min_games: int = 10, exclude: tuple[str, ...] = (),
                      first_test: str = "2026-04", cash: float = 50.0,
                      cfg: RiskConfig = RiskConfig()) -> Path:
    """Monthly walk-forward of the Elo + market blend over all history with prices."""
    from albert.research import ratings as rt

    markets = read_table(s.tables_dir, "elo_markets").filter(~pl.col("series_ticker").is_in(list(exclude)))
    preds = rt.elo_predictions(markets, k=k)
    legs = markets.filter(pl.col("ticker").is_in(preds["ticker"].implode()))
    candles = read_table(s.tables_dir, "kalshi_candles").filter(pl.col("ticker").is_in(legs["ticker"].implode()))
    obs = cal.kalshi_observations(legs, candles, horizon_hours=lead, anchor="start", staleness_hours=2.0)
    obs = obs.join(preds.select(pl.col("ticker").alias("market_id"), "elo_p", "n_min",
                                pl.col("series").alias("ser")), on="market_id").filter(pl.col("n_min") >= min_games)
    r = Report(s.reports_dir, "research-elo-walk")
    r.text("Setup", f"- {obs.height} priced games ({obs['ser'].n_unique()} series), lead {lead:g}h, K {k:g}, "
                    f"min games {min_games}, excluded {list(exclude) or 'none'}\n- each month's blend is fit "
                    f"on all earlier months only; flat one-contract bets; SEs clustered by date")
    out: dict = {}
    for mg in margins:
        per_month, bets = rt.walk_forward(obs, margin=mg, first_test=first_test)
        r.table(f"Margin {mg:g}: by month", per_month)
        if bets.is_empty():
            continue
        tot = lit.cluster_ratio(bets.with_columns(pl.lit("all").alias("k"), pl.col("date").alias("event_ticker")),
                                ["k"]).row(0, named=True)
        wk = bets.group_by("week").agg(pl.col("pnl").sum())
        out[f"m{mg:g}"] = {"bets": tot["n"], "dates": tot["clusters"], "roi": tot["roi"], "t": tot["t"],
                           "months_positive": int((per_month["pnl"] > 0).sum()), "months": per_month.height,
                           "weeks_positive": int((wk["pnl"] > 0).sum()), "weeks": wk.height}
        r.table(f"Margin {mg:g}: by series", lit.cluster_ratio(bets.with_columns(pl.col("date").alias("event_ticker")),
                                                              ["ser"]), max_rows=25)
        if mg == margins[0]:
            signals = bets.select(pl.lit("elo").alias("strategy"), "venue", "event_id",
                                  pl.col("decision_ts").alias("signal_ts"), "resolve_ts",
                                  pl.lit(None, dtype=pl.Float64).alias("available"), "fee_type", "fee_multiplier",
                                  "market_id", "side", "price", "prob", "payoff",
                                  pl.col("ser").alias("note")).select(list(SIGNAL_SCHEMA)).cast(SIGNAL_SCHEMA)
            result = run_backtest(signals, cash={"kalshi": cash, "pmus": 0.0}, cfg=replace(cfg, min_edge=mg))
            out["bankroll"] = {k_: v for k_, v in result.metrics().items()
                               if k_ in ("start_equity", "end_equity", "max_drawdown", "trades", "roi_per_dollar",
                                         "halted_at")}
            if _plot_equity(result, r.dir / "equity.png"):
                r.image("Equity curve", "equity.png")
    r.json("Summary", out)
    return r.save()


def research_esports_ratings(s: Settings, c: Clients, *, lead: float = 1.0, k: float = 24.0,
                             margins: tuple[float, ...] = (0.03, 0.06), first_test: str = "2026-04",
                             min_games: int = 10) -> Path:
    """PandaScore map-level Elo (all pro matches) blended with Kalshi esports prices.

    Walk-forward by month; ratings are leak-free (research/esports.py).
    """
    from albert.research import esports as es
    from albert.research import ratings as rt

    kalshi = read_table(s.tables_dir, "elo_markets")
    psm = read_table(s.tables_dir, "esports_matches")
    psg = read_table(s.tables_dir, "esports_games")
    mapping = es.match_kalshi_pandascore(kalshi, psm)
    # Winner check: match markets vs PandaScore match winner; map markets vs that map's winner.
    chk = mapping.join(psm.select("match_id", "winner_id"), on="match_id", how="left").join(
        psg.select("match_id", pl.col("position").alias("map_no"), pl.col("winner_id").alias("map_winner")),
        on=["match_id", "map_no"], how="left").with_columns(
        pl.when(pl.col("map_no").is_null()).then(pl.col("winner_id")).otherwise(pl.col("map_winner")).alias("w"))
    chk = chk.drop_nulls("w").with_columns(((pl.col("w") == pl.col("first_team_id")) == pl.col("kalshi_first_won"))
                                           .alias("agree"))
    bad = set(chk.filter(~pl.col("agree"))["kalshi_ticker"].to_list())
    mapping = mapping.filter(~pl.col("kalshi_ticker").is_in(list(bad)))
    preds = es.predictions(mapping, psm, psg, lead_hours=lead, k=k)
    legs = kalshi.filter(pl.col("ticker").is_in(mapping["kalshi_ticker"].implode()))
    candles = read_table(s.tables_dir, "kalshi_candles").filter(pl.col("ticker").is_in(legs["ticker"].implode()))
    obs = cal.kalshi_observations(legs, candles, horizon_hours=lead, anchor="start", staleness_hours=2.0)
    obs = obs.join(preds.rename({"kalshi_ticker": "market_id"}), on="market_id").join(
        mapping.select(pl.col("kalshi_ticker").alias("market_id"), pl.col("series").alias("ser")), on="market_id"
    ).filter(pl.col("n_min") >= min_games)
    r = Report(s.reports_dir, "research-esports-ratings")
    r.text("Setup", f"- PandaScore: {psm.height} matches, {psg.height} maps (all tiers)\n"
                    f"- Kalshi esports events matched: {mapping.height + len(bad)}; winner agreement "
                    f"{float(chk['agree'].mean()):.4f} on {chk.height} checked ({len(bad)} disagreeing dropped)\n"
                    f"- priced games (lead {lead:g}h, both teams >= {min_games} rated maps): {obs.height}\n"
                    "- map-level Elo (K {k:g}) from all maps, applied at each map's end time; blend fit "
                    "on earlier months only".replace("{k:g}", f"{k:g}"))
    out = {}
    for mg in margins:
        per_month, bets = rt.walk_forward(obs, margin=mg, first_test=first_test, min_train=300)
        r.table(f"Margin {mg:g}: by month", per_month)
        if bets.is_empty():
            continue
        tot = lit.cluster_ratio(bets.with_columns(pl.lit("all").alias("k"), pl.col("date").alias("event_ticker")),
                                ["k"]).row(0, named=True)
        wk = bets.group_by("week").agg(pl.col("pnl").sum())
        out[f"m{mg:g}"] = {"bets": tot["n"], "dates": tot["clusters"], "roi": tot["roi"], "t": tot["t"],
                           "months_positive": int((per_month["pnl"] > 0).sum()), "months": per_month.height,
                           "weeks_positive": int((wk["pnl"] > 0).sum()), "weeks": wk.height}
        r.table(f"Margin {mg:g}: by series", lit.cluster_ratio(
            bets.with_columns(pl.col("date").alias("event_ticker")), ["ser"]), max_rows=12)
    r.json("Summary", out)
    obs.write_parquet(r.dir / "observations.parquet")
    return r.save()


def backtest_weather(s: Settings, *, split: datetime, slots: tuple[str, ...] | None = None,
                     cash: float = 50.0, cfg: RiskConfig = RiskConfig()) -> Path:
    """NWS-model bets on Kalshi daily-high brackets (research/weather.py)."""
    from albert.research import weather as w

    slots = slots or tuple(w.SLOTS)
    markets = read_table(s.tables_dir, "kalshi_markets")
    candles = read_table(s.tables_dir, "kalshi_candles")
    cli = read_table(s.tables_dir, "wx_cli").drop_nulls("high")
    obs = read_table(s.tables_dir, "wx_obs")
    nbm = read_table(s.tables_dir, "wx_nbm")
    legs = w.high_markets(markets)
    split_day = split.date()
    offsets = w.obs_offsets(cli.filter(pl.col("day") < split_day), obs)["u"]
    r = Report(s.reports_dir, "backtest-weather")
    r.text("Setup", f"- {legs['event_ticker'].n_unique()} KXHIGH events ({legs.height} legs), "
                    f"{legs['code'].n_unique()} stations, {legs['day'].min()} to {legs['day'].max()}\n"
                    f"- model fit on CLI days before {split_day}, evaluated after; NBM runs used "
                    f"only after runtime + {w.NBM_LAG}; observations only before the decision\n"
                    f"- fills at the hourly candle's ask/bid at the decision time, taker fee\n"
                    f"- risk: {cfg}")
    rows, sig_frames = [], []
    for slot in slots:
        feats = w.features(legs.select("code", "day").unique(), obs, nbm, slot)
        train = feats.filter(pl.col("day") < split_day)
        test = feats.filter(pl.col("day") >= split_day)
        if train.is_empty() or test.is_empty():
            continue
        model = w.HighModel.fit(train, cli, offsets).recalibrate(
            legs.join(train, on=["code", "day"]))
        q = w.with_model(w.leg_quotes(legs, candles, test), model)
        q = q.filter(pl.col("yes_ask_close") > pl.col("yes_bid_close"))
        if q.is_empty():
            continue
        sig = w.weather_signals(q, slot)
        sig_frames.append(sig)
        # Weather busts are regional: cities on the same date are not independent. Cluster SEs
        # on the date in the event ticker (KXHIGHNY-26OCT17 -> 26OCT17).
        flat = lit.flat_signal_returns(sig, min_edge=cfg.min_edge, lo=cfg.min_price, hi=cfg.max_price,
                                       cluster=pl.col("event_id").str.split("-").list.last())
        fr = flat.row(0, named=True) if not flat.is_empty() else {}
        rows.append({"slot": slot, "bias": round(model.bias, 2), "sigma": round(model.sigma, 2),
                     "tail": round(model.tail_weight, 3),
                     "platt": f"{model.platt[0]:+.2f},{model.platt[1]:.2f}",
                     "test_events": q["event_ticker"].n_unique(), "legs": q.height,
                     "brier_model": w.brier(q["prob"], q["y"]), "brier_market": w.brier(q["mid"], q["y"]),
                     "flat_trades": fr.get("n", 0), "flat_days": fr.get("clusters"),
                     "flat_roi": fr.get("roi"), "flat_t": fr.get("t")})
    r.table("Per decision slot (test period)", pl.DataFrame(rows))
    signals = pl.concat(sig_frames) if sig_frames else pl.DataFrame(schema=SIGNAL_SCHEMA)
    result = run_backtest(signals, cash={"kalshi": cash, "pmus": 0.0}, cfg=cfg)
    r.json("Bankroll (all slots, one position per market)", result.metrics())
    r.json("Monte Carlo (bootstrap of realized bets)", monte_carlo(result))
    fills = result.fills_frame()
    if not fills.is_empty():
        by = fills.with_columns(pl.col("note").str.split(" ").list.slice(0, 2).list.join(" ").alias("slot")
                                ).group_by("slot").agg(
            pl.len().alias("trades"), pl.col("cost").sum().alias("staked"), pl.col("pnl").sum(),
            (pl.col("pnl") > 0).mean().alias("win_rate")).with_columns(
            (pl.col("pnl") / pl.col("staked")).alias("roi")).sort("slot")
        r.table("Bankroll fills by slot", by)
        r.table("Fills", fills.sort("ts").select("ts", "market_id", "side", "price", "prob", "contracts",
                                                 "cost", "pnl", "note"), save_as="fills", max_rows=30)
    if _plot_equity(result, r.dir / "equity.png"):
        r.image("Equity curve", "equity.png")
    signals.write_parquet(r.dir / "signals.parquet")
    return r.save()


def backtest(s: Settings, c: Clients | None, *, strategies: list[str], split: datetime,
             cash: dict[str, float], cfg: RiskConfig, horizon_hours: float = 2.0,
             delay_minutes: float = 5.0, top_k: int = 20, prior: float = 200.0,
             anchor: str = "scheduled", universe: str = "all") -> Path:
    frames = []
    notes = []
    notes.append(f"decision anchor: {anchor} (scheduled = expected_expiration_time - horizon); "
                 f"universe: {universe}")
    if "longshot" in strategies:
        obs = kalshi_obs(s, horizon_hours, anchor, universe)
        train, test = split_by_time(obs, split)
        notes.append(f"longshot: trained on {train.height} markets resolved before {split:%Y-%m-%d}, "
                     f"tested on {test.height} decided after it; horizon {horizon_hours:g}h, "
                     f"prior {prior:g}.")
        if train.height and test.height:
            frames.append(longshot_signals(train, test, prior=prior))
    if "slope" in strategies:
        obs = kalshi_obs(s, horizon_hours, anchor, universe)
        train, test = split_by_time(obs, split)
        notes.append(f"slope: per-category logistic recalibration fit on {train.height} markets "
                     f"resolved before {split:%Y-%m-%d}, tested on {test.height}; "
                     f"horizon {horizon_hours:g}h.")
        if train.height and test.height:
            frames.append(lit.slope_signals(train, test))
    if "wallet_follow" in strategies:
        if c is None:
            raise RuntimeError("wallet_follow needs API clients for price history")
        sig = build_follow_signals(s, c, split=split, top_k=top_k, delay_minutes=delay_minutes)
        notes.append(f"wallet_follow: {sig.height} copy signals from the top {top_k} window-A "
                     f"wallets, entry at price {delay_minutes:g} min after their trade + 1c, "
                     "Polymarket US fees.")
        frames.append(sig)
    signals = pl.concat(frames) if frames else pl.DataFrame(schema=SIGNAL_SCHEMA)
    result = run_backtest(signals, cash=cash, cfg=cfg)
    r = Report(s.reports_dir, "backtest-" + "+".join(strategies))
    r.text("Setup", "\n".join(f"- {n}" for n in notes) + f"\n- cash per venue: {cash}\n- risk: {cfg}")
    r.json("Results", result.metrics())
    r.json("Monte Carlo (bootstrap of realized bets)", monte_carlo(result))
    fills = result.fills_frame()
    if not fills.is_empty():
        by = fills.group_by("strategy", "venue").agg(
            pl.len().alias("trades"), pl.col("cost").sum().alias("staked"),
            pl.col("pnl").sum(), pl.col("fee").sum().alias("fees"),
            (pl.col("pnl") > 0).mean().alias("win_rate"),
            (pl.col("prob") - pl.col("price")).mean().alias("expected_edge"),
        ).with_columns((pl.col("pnl") / pl.col("staked")).alias("roi"))
        r.table("By strategy and venue", by)
        r.table("Fills", fills.sort("ts").select(
            "ts", "strategy", "venue", "market_id", "side", "price", "prob", "contracts", "fee",
            "cost", "payoff", "pnl", "note"), save_as="fills", max_rows=40)
    if _plot_equity(result, r.dir / "equity.png"):
        r.image("Equity curve", "equity.png")
    signals.write_parquet(r.dir / "signals.parquet")
    return r.save()
