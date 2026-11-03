# Albert: Roadmap

Status 2026-11-03: Phases 1-5 built (see README "Results so far"). Phase 5's 2-week paper run is the next step, then Phase 6.

Each phase ends with a **live smoke run that produces non-empty, sanity-checked output.**

## Phase 1: Capture + data layers (Kalshi and Polymarket in parallel)

- [x] `albert capture` saves trimmed real responses to `tests/fixtures/<venue>/`.
- [x] Shared HTTP client: retries with backoff on 429 and 5xx, rate limiting, append-only raw
      cache keyed by path + params, resumable runs.
- [x] Kalshi: events/markets filtered to non-financial categories, trades per market,
      candlesticks, historical-endpoint routing, settlement results.
- [x] Polymarket intl (data only): Gamma markets/events, Data API v2
      trades/activity/positions/leaderboard.
- [x] Polymarket US: public gateway markets/events/series, order books, BBO, price history,
      settlements.
- [x] Typed Parquet tables (real decimal and timestamp columns, not strings) plus run manifests.
- [x] Decide the non-financial category/tag list from observed values.

## Phase 2: Cross-venue matching

- [x] Match markets across Kalshi, Polymarket US, and Polymarket intl on the same event and the
      same resolution criteria. Do Polymarket intl -> US first, since that is the wallet-signal
      path.
      Start with rule-based matching on title/date/entities, then do a manual review pass.
- [x] Measure match quality and price-gap distributions.

## Phase 3: Research

- [x] **Wallet scoring:**
  - Per-wallet PnL per $ risked and edge vs entry price.
  - Shrinkage for small samples.
  - Filters for market makers and wash traders.
  - Persistence test: score on window A, evaluate on window B.
- [x] **Follow-the-wallet sim:** enter N seconds/minutes after the wallet's trade, at the Kalshi
      ask on the matched market (or the Polymarket price as a proxy).
- [x] **Market inefficiencies:** calibration by category × price bucket (favorite-longshot bias)
      and cross-venue gaps.
- [ ] Late-resolution drift (not built yet).
- [x] Literature re-tests (`research literature`, 2026-11-03): maker/taker, sum-of-legs arb,
      category calibration slopes, rolling-min reversion, move reversal. Only the maker
      premium survives.
- [x] Fixed two biases: decisions anchored on scheduled expiry (not actual close), and
      volume-blind samples (`ingest kalshi-detail --series`, `--universe population`). The
      Phase 5 longshot edge is withdrawn.

## Phase 4: Backtester

- [x] Event-driven sim with the Kalshi fee schedule, bid/ask fills, displayed-size limits,
      fractional Kelly, and risk caps.
- [x] Reports: equity curve, drawdown, per-strategy attribution, fee drag, and
      skipped-trade reasons.

## Phase 5: Paper trading

- [x] Paper trader built (`albert paper run`): live quotes, simulated fills, settlement, ledger.
- [x] Maker backtest (`backtest maker`): trade-driven fills, one order per game, game-clustered
      SEs. 24h/20h: +4.6% per filled $ (t = 1.0), not yet significant.
- [x] Maker paper mode (`paper run --strategy maker`), service-ready (SIGTERM, heartbeat,
      backoff, no live raw recording), Dockerfile + deploy/gcp.md. Started locally 2026-11-03.
- [x] Holdout (2,978 new markets) and volume-blind population test (2,357 games, windowed
      trades): maker 24h/20h = −1.7% per filled $ (t = −0.8). No edge on major leagues.
- [x] Weather (NWS/IEM): model loses to the market at every slot; decided brackets priced
      correctly. No edge at hourly resolution.
- [ ] Volume-blind maker test for Weather and minor sports (needs candles + windowed trades
      for every leg of each event).
- [ ] Move the paper loop to a GCE VM (deploy/gcp.md).
- [ ] Run it for >= 2 weeks or >= 100 fills and compare to the historical maker returns.

## Phase 6: Live trading ($100, split about $50 / $50)

- [ ] Order clients: Kalshi (RSA-PSS) and Polymarket US (Ed25519). Both need an explicit enable
      flag, per-order and daily-loss caps per venue, a global kill switch, and a full order/fill
      audit log.
- [ ] Cross-venue trades (buying opposite sides on two venues) only after checking that the
      resolution criteria are identical.

- [x] Elo + market blend for minor sports: the apparent edge was a look-ahead (same-start
      match/map events). Leak-free walk-forward May-Oct: no edge. Paper `elo` removed.
