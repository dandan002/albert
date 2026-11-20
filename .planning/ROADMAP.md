# Albert: Roadmap

Status 2026-11-19: Phases 1-6 built. Live on Kalshi since 2026-11-07 (mentions bot); the Phase 6
safety layer (global kill switch, account-wide daily-loss halt, order/fill audit log) went live
2026-11-19. See `.planning/LIVE-PLAN.md`.

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
- [x] Hyperliquid (data only, 2026-11-20): HIP-4 outcome metadata and candles (`ingest hl`), and
      a continuous WebSocket recording of non-financial outcome books and trades with
      addresses (`record-hl`, unit `albert-record-hl`).
- [x] Raw stream archive (2026-11-20): closed days go to S3 (Intelligent-Tiering, SHA-256
      verified, the host role cannot delete) by the daily `albert-archive.timer`, keeping 4 days on
      disk; readers fail loudly on an archived day (`albert restore-stream <day> --venue <v>`).
- [x] Typed Parquet tables (real decimal and timestamp columns, not strings) plus run manifests.
- [x] Decide the non-financial category/tag list from observed values.

## Phase 2: Cross-venue matching

- [x] Match markets across Kalshi, Polymarket US, and Polymarket intl on the same event and the
      same resolution criteria. Do Polymarket intl -> US first, since that is the wallet-signal
      path.
      Start with rule-based matching on title/date/entities, then do a manual review pass.
- [x] Measure match quality and price-gap distributions.
- [ ] Hyperliquid vs Kalshi (about 2026-12-04, after 2 weeks of recording): normalize sport
      labels, match soccer/NBA/NFL markets, then the frozen lead-lag and wallet-flow checks
      (`.planning/research/SUMMARY.md`).

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

## Phase 6: Live trading ($100, Kalshi only)

Built 2026-11-19 (user: "Let's build phase 6 we're live trading already anyway"); live trading
itself started 2026-11-07 with the mentions bot.

- [x] Order client: Kalshi (RSA-PSS; the same account's Ed25519 key is preferred), with an explicit
      enable flag, per-order and daily-loss caps, a global kill switch, and a full order/fill
      audit log.
  - Client (`venues/kalshi_trade.py`): prod writes need `KALSHI_LIVE_ENABLED=1` and `--live`;
    write allowlist (no Cancel All, no batch cancel); cancels by own client_order_id prefix only;
    per-order cap 5 contracts and $2, post-only on prod (`ORDER_LIMITS`); refuses every new order
    while the global kill switch or the daily-loss halt is on; audits every write request, response
    and refusal to `data/live/audit/<env>/orders-<day>.jsonl` (never headers).
  - Bot caps (unchanged): mentions $15 expected fill cost / $40 hard cap / $15 daily fill cost;
    makers per-run loss halt and contract caps (`live_l2.HARD_LIMITS`).
  - Guard (`live_guard.py`, `albert live guard`, systemd `albert-guard@prod`, cancel-only client):
    exchange-sourced fill audit (`fills.jsonl`, `order_tags.jsonl`); bot book marked to the mid or
    settlement; account-wide day loss cap $15 (HALT until the next 00:00 UTC) and at-risk cap $60
    (KILL); sweeps every bot order while either is on.
  - CLI: `live kill [--live]`, `live resume [--halt]`, `live audit`; `live check` shows both switches
    and the guard's last pass.
  - Smoke 2026-11-19: the guard backfilled 129 account fills (100 bot fills tagged by bot, 29 manual,
    0 untagged); with the kill switch on, a prod create was refused before sending and
    `albert-mentions@prod` stopped within a minute; switch removed and bot restarted.
- Polymarket US trading is on hold (2026-11-07): Polymarket is data only, so no cross-venue
  trades.

- [x] Elo + market blend for minor sports: the apparent edge was a look-ahead (same-start
      match/map events). Leak-free walk-forward May-Oct: no edge. Paper `elo` removed.
- [x] PandaScore esports ratings (24k matches, 99.84% winner agreement with Kalshi):
      Elo worse than market, blend adds nothing. No edge.
