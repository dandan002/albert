# Albert: Roadmap

Each phase ends with a **live smoke run that produces non-empty, sanity-checked output.**

## Phase 1: Capture + data layers (Kalshi and Polymarket in parallel)

- [ ] `albert capture` saves trimmed real responses to `tests/fixtures/<venue>/`.
- [ ] Shared HTTP client: retries with backoff on 429 and 5xx, rate limiting, append-only raw
      cache keyed by path + params, resumable runs.
- [ ] Kalshi: events/markets filtered to non-financial categories, trades per market,
      candlesticks, historical-endpoint routing, settlement results.
- [ ] Polymarket intl (data only): Gamma markets/events, Data API v2
      trades/activity/positions/leaderboard.
- [ ] Polymarket US: public gateway markets/events/series, order books, BBO, price history,
      settlements.
- [ ] Typed Parquet tables (real decimal and timestamp columns, not strings) plus run manifests.
- [ ] Decide the non-financial category/tag list from observed values.

## Phase 2: Cross-venue matching

- [ ] Match markets across Kalshi, Polymarket US, and Polymarket intl on the same event and the
      same resolution criteria. Do Polymarket intl -> US first, since that is the wallet-signal
      path.
      Start with rule-based matching on title/date/entities, then do a manual review pass.
- [ ] Measure match quality and price-gap distributions.

## Phase 3: Research

- [ ] **Wallet scoring:**
  - Per-wallet PnL per $ risked and edge vs entry price.
  - Shrinkage for small samples.
  - Filters for market makers and wash traders.
  - Persistence test: score on window A, evaluate on window B.
- [ ] **Follow-the-wallet sim:** enter N seconds/minutes after the wallet's trade, at the Kalshi
      ask on the matched market (or the Polymarket price as a proxy).
- [ ] **Market inefficiencies:** calibration by category × price bucket (favorite-longshot bias),
      cross-venue gaps, and late-resolution drift.

## Phase 4: Backtester

- [ ] Event-driven sim with the Kalshi fee schedule, bid/ask fills, displayed-size limits,
      fractional Kelly, and risk caps.
- [ ] Reports: equity curve, drawdown, per-strategy attribution, fee drag, and
      skipped-trade reasons.

## Phase 5: Paper trading

- [ ] Run the strategies live against real Kalshi quotes with simulated orders for >= 2 weeks or
      >= 100 signals. Compare to the backtest.

## Phase 6: Live trading ($100, split about $50 / $50)

- [ ] Order clients: Kalshi (RSA-PSS) and Polymarket US (Ed25519). Both need an explicit enable
      flag, per-order and daily-loss caps per venue, a global kill switch, and a full order/fill
      audit log.
- [ ] Cross-venue trades (buying opposite sides on two venues) only after checking that the
      resolution criteria are identical.
