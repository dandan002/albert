"""Research and backtest pipelines used by the CLI. Each writes a report under data/reports."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from albert.backtest.engine import BacktestResult, monte_carlo, run_backtest
from albert.backtest.strategies import longshot_signals, split_by_time
from albert.config import Settings
from albert.ingest import Clients
from albert.reports import Report
from albert.research import calibration as cal
from albert.research import follow, gaps, wallets
from albert.research.follow import SIGNAL_SCHEMA
from albert.risk import RiskConfig
from albert.store import read_table, write_table
from albert.venues import polymarket as pm


# ---- data helpers -------------------------------------------------------------------------------
def kalshi_obs(s: Settings, horizon_hours: float) -> pl.DataFrame:
    markets = read_table(s.tables_dir, "kalshi_markets")
    candles = read_table(s.tables_dir, "kalshi_candles")
    return cal.kalshi_observations(markets, candles, horizon_hours=horizon_hours)


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


def backtest(s: Settings, c: Clients | None, *, strategies: list[str], split: datetime,
             cash: dict[str, float], cfg: RiskConfig, horizon_hours: float = 2.0,
             delay_minutes: float = 5.0, top_k: int = 20, prior: float = 200.0) -> Path:
    frames = []
    notes = []
    if "longshot" in strategies:
        obs = kalshi_obs(s, horizon_hours)
        train, test = split_by_time(obs, split)
        notes.append(f"longshot: trained on {train.height} markets resolved before {split:%Y-%m-%d}, "
                     f"tested on {test.height} decided after it; horizon {horizon_hours:g}h, "
                     f"prior {prior:g}.")
        if train.height and test.height:
            frames.append(longshot_signals(train, test, prior=prior))
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
