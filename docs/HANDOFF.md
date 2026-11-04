# Albert: handoff prompt

Paste everything below the line into a new agent session to continue this project.

---

You are taking over **Albert**, a Python research CLI for systematic, small-stake trading of
**non-financial prediction markets** (sports, weather, politics, esports) on **Kalshi** and
**Polymarket US**. Repo: `github.com/dandan002/albert` (default branch `master`). Read
`AGENTS.md` first (ground rules), then `README.md` ("Results so far", sections 1-9) and
`.planning/ROADMAP.md`.

## Goal and constraints

- Bankroll: $100 ($50 Kalshi / $50 Polymarket US), **not funded**. Since 2026-11-03 the
  objective is **consistent positive returns** (most weeks/months positive, small
  drawdowns), not exponential growth. Thin, low-volume markets are in scope.
- Polymarket international (polymarket.com) is **data only**. The user is a US resident:
  never place orders there and never build geoblock workarounds.
- **No live trading is built.** Phase 6 (order clients, kill switch, caps) only starts after
  a strategy survives backtest, holdout *and* forward paper testing, and the user enables it.
- Credentials live only in `.env`: `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY`, Polymarket
  intl keys (do not use them for trading), and `PANDASCORE_TOKEN`. `albert.config.secret()`
  reads them. Never log them or put them in manifests, raw files or fixtures.
- Commit only when asked. Work on a branch, small focused commits, and end commit messages
  with the attribution lines your harness specifies.

## Environment

- Python: `/Users/dandan/anaconda3/bin/python3` (3.12). Run as `python -m albert ...`.
- Checks: `python -m pytest -q` (83 tests, offline) and `python -m ruff check src tests`.
- Data (git-ignored) is under `data/`:
  - `tables/*.parquet`: analysis tables
  - `raw/<venue>/*.json.zst`: append-only, zstd-compressed raw responses
  - `runs/*.json`: manifests
  - `reports/<name>-<ts>/`: reports
  - `paper/`: paper account

  There are about 2M Kalshi candles, 1.3M settled Kalshi markets, NWS weather and
  PandaScore esports tables. Re-running an ingest reuses the cache.
- Kalshi rate limits: the batch candles endpoint (`/markets/candlesticks`) caps a request at
  10,000 candles and serves only post-cutoff markets (after 2026-09-04). Historical markets
  go one at a time, on 6 threads. 429s are retried.

## What exists

- **Data:**
  - Kalshi markets, trades, candles (single and batch), events, and windowed trades
  - Polymarket intl markets, wallets, price history
  - Polymarket US markets
  - NWS/IEM weather (CLI settlement reports, ASOS observations, NBM forecasts)
  - PandaScore esports matches and maps
- **Matching:** Kalshi ↔ Polymarket intl ↔ Polymarket US for game moneylines; Kalshi ↔
  Polymarket esports; Kalshi ↔ PandaScore (99.84% winner agreement).
- **Research and backtests** (`albert research ...`, `albert backtest ...`):
  - calibration and literature effects
  - maker fill model (trade-driven, queue-aware)
  - weather model
  - minor sports
  - Elo and PandaScore ratings with monthly walk-forward
- **Paper trader** (`albert paper run`): runs unattended, handles SIGTERM, writes a
  heartbeat file and logs JSON lines. A `Dockerfile` and `deploy/gcp.md` (Compute Engine
  VM) exist, but nothing is deployed.

## Results (all negative so far; details in README)

Every strategy testable with public data at hourly resolution found **no edge** once the
backtest was correct:

- **Wallets:** no persistent skill, and copy-trading loses money.
- **Arbitrage:** cross-venue and sum-of-legs arbitrage are priced out.
- **Calibration / longshot:** no edge on clean data.
- **Maker bids:** major leagues −1.7% per filled dollar. A minor-sports subgroup failed
  its holdout.
- **Weather:** the NWS model loses to the market, and already-decided brackets are priced
  at 0c/100c.
- **Minor sports and tennis:** calibrated, or not enough to pay the spread plus fee.
  Esports prices track Polymarket within about 1c.
- **Ratings:** Elo (Kalshi-only) and PandaScore Elo (all pro matches) add nothing to Kalshi
  prices.

## Lessons: check every big backtest edge for these bugs

Each of these produced a fake 2-4x bankroll result before it was caught:

1. **Kalshi `close_time` on a settled market is the actual close** (game over, word said).
   Anchor decisions on `expected_expiration_time` or the ticker start time
   (`venues/kalshi.ticker_start`, US Eastern), never on `close_time`.
2. **Never sample on final volume.** Top-N selects upsets. A 500-200k band, or a
   "newest N trades" coverage rule, selects favourites winning. Use volume-blind samples
   (`ingest kalshi-detail --series/--events-per-series`, `--universe population`, windowed
   trades).
3. **Same-start events leak.** Match and map markets of one match share a start time.
   Apply results at their close or end time and predict at start − lead.

Also cluster standard errors by date or game, not by event, and use holdouts for any
subgroup you find.

## Current state

- `master` == `restart` (latest work).
- The paper trader may be running locally: `data/paper/run.pid`, maker strategy only,
  started 2026-11-03. This is a fill-model check, not a money-maker. Check it with
  `python -m albert paper status`.
- The `elo` paper strategy is withdrawn (its coefficients came from a leaky fit).

## Open ideas (each needs information or speed the public data lacks)

- **Information not in public match results.** Esports roster and stand-in changes, map
  vetoes and patches (Liquipedia has an API; check its terms). Lineups and injuries in
  minor leagues.
- **Sub-minute trading.** Poll order books and trades every few seconds and react to live
  events. This can only be tested forward. Build the capture first, then a paper strategy.
- **Sharp-book reference prices.** The Odds API (Pinnacle) needs a paid key for history.
  The user chose not to pursue it yet.
- **Volume-blind maker test for weather brackets.** Not run; would need windowed trades for
  every leg.

## First steps for you

1. Run `pytest` and `ruff`, and read README "Results so far".
2. Ask the user which open idea to pursue (or a new one) before building anything large.
3. For any new data source: record real responses into `tests/fixtures/<venue>/` first
   (ground rule 1), and treat zero-row ingests as failures.
