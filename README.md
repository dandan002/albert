# Albert

Albert is a research and paper-trading system for **non-financial prediction markets** (sports,
politics, culture, weather, science). It is built to answer one question with evidence before
any real money moves:

> Is there a repeatable edge, after real fees and real execution, that can compound a $100
> bankroll split across Kalshi and Polymarket US?

Albert ingests data from three venues, matches the same event across them, researches three
edge hypotheses, backtests them with realistic execution, and paper-trades them on live quotes.
**It cannot place real orders.** There is no authenticated or order-signing code in this
repository. Live trading is Phase 6, and it has not been built.

**Status (2026-11-03):** Phases 1-5 are built. Current findings are in
[Results so far](#results-so-far).

---

## Contents

- [Venues](#venues)
- [Install](#install)
- [Quick start](#quick-start)
- [Commands](#commands)
- [How it works](#how-it-works)
- [Strategies](#strategies)
- [Fees](#fees)
- [Risk and sizing](#risk-and-sizing)
- [Data layout](#data-layout)
- [Results so far](#results-so-far)
- [Known limitations](#known-limitations)
- [Development](#development)
- [Roadmap](#roadmap)

---

## Venues

| Venue | Role | What Albert uses |
|---|---|---|
| **Kalshi** | trading venue (CFTC-regulated) | series/categories, settled + open markets, trades, bid/ask candlesticks, order books |
| **Polymarket US** (`gateway.polymarket.us`) | trading venue (CFTC-regulated) | markets, events, order books, BBO, price history, resolutions |
| **Polymarket international** (`polymarket.com`) | **data only** | market metadata/resolutions (Gamma), on-chain wallet trades, leaderboards, price history (Data API v2) |

US residents are geoblocked from trading on polymarket.com. Albert only reads its public data
and never places orders there. Polymarket intl is where the wallet signals come from, because
its trades are on-chain and attributable. Kalshi and Polymarket US trades are anonymous.

"Non-financial" means Albert excludes Kalshi categories `Financials`, `Economics`, `Crypto`,
`Commodities` and `Companies`; Polymarket intl tags such as `crypto`, `finance`, `fed` and
`stocks`; and Polymarket US categories `finance` and `crypto`. See
`src/albert/categories.py`.

---

## Install

```bash
python3 -m pip install -e ".[dev,plots]"   # Python 3.12+
python3 -m pytest -q                        # 37 tests, offline, against recorded API responses
```

No API keys are needed for Phases 1-5. All endpoints used are public. `.env` holds Kalshi
credentials for the future live phase and is git-ignored. Nothing in Phases 1-5 reads it.

All data goes to `./data` (git-ignored). Override with `ALBERT_DATA_DIR=/path`.

---

## Quick start

The full research loop, from an empty `data/` directory:

```bash
# 1. Data (each step is cached; re-runs are cheap)
albert ingest kalshi --start 2026-08-15 --end 2026-11-01 --min-volume 2000 \
    --candles --max-markets 600 --per-category          # settled markets + hourly bid/ask
albert ingest pm-markets --start 2026-10-25 --end 2026-11-02   # Polymarket intl resolutions
albert ingest pm-wallets --start 2026-05-01 --end 2026-11-02 --top 100 \
    --category overall --category sports --category politics --max-pages 5
albert ingest pmus --start 2026-10-20 --end 2026-11-02         # Polymarket US (windowed!)

# 2. Cross-venue matching
albert match

# 3. Research reports (markdown + parquet under data/reports/)
albert research calibration --horizon 2
albert research wallets --split 2026-08-15
albert research gaps

# 4. Backtests
albert backtest run --strategy longshot --split 2026-10-01
albert backtest run --strategy wallet_follow --split 2026-08-15 --delay 5

# 5. Paper trading on live quotes
albert paper reset --kalshi-cash 50 --pmus-cash 50
albert paper run --interval 300            # Ctrl-C to stop; state persists
albert paper status
```

`albert` is installed as a console script. Without installing, use `python3 -m albert`.

---

## Commands

| Command | What it does |
|---|---|
| `albert ingest kalshi` | Settled non-financial Kalshi markets that closed in `[start, end)`, routed across the live/historical cutoff. `--trades` and `--candles` fetch per-market history for the top `--max-markets` by volume (`--per-category` applies the limit within each category). |
| `albert ingest kalshi-open` | Snapshot of open non-financial Kalshi markets. |
| `albert ingest pm-markets` | Polymarket intl events and markets with resolutions (`--open` for live markets). |
| `albert ingest pm-wallets` | Leaderboard wallets (per category) and their trades, paged month by month, then backfills resolutions for every market they traded. |
| `albert ingest pmus` | Polymarket US markets: game moneylines plus politics/culture/science. Closed ingests need `--start/--end`, because the venue has more than 2M closed markets. |
| `albert match` | Builds `match_kalshi_pm` and `match_pmus_pm` (and, with `--fuzzy`, review-only title candidates). |
| `albert research calibration` | Favorite-longshot tables: hit rate vs price, after-fee edge of buying YES or NO, by category. |
| `albert research wallets` | Wallet skill scores and an out-of-sample persistence test. |
| `albert research gaps` | Kalshi vs Polymarket price gaps and locked-in opportunities on exact matches. |
| `albert backtest run` | Runs one or more strategies through the backtester. Writes a report with metrics, Monte Carlo, fills and an equity curve. |
| `albert paper reset/run/status` | Paper trading account (see [Paper trading](#5-paper-trading)). |
| `albert capture` | Records small real API responses into `tests/fixtures/` for parser tests. |

Every command prints where it wrote its outputs. Every ingest writes a run manifest
(`data/runs/*.json`) listing its parameters, output row counts, and every raw response file it
read or wrote.

---

## How it works

```
             ┌──────────── venues/ ────────────┐
 HTTP ──────▶│ kalshi.py  polymarket.py  polymarket_us.py │── normalizers ──▶ data/tables/*.parquet
 (http.py)   └─────────────────────────────────┘                              │
   │ append-only raw cache: data/raw/<venue>/<sha256(url+params)>.json        │
   ▼                                                                          ▼
 data/runs/*.json (manifests)               matching.py ──▶ match_kalshi_pm, match_pmus_pm
                                                                              │
                         research/  calibration.py  wallets.py  follow.py  gaps.py
                                                                              │
                         backtest/  strategies.py ──▶ signals ──▶ engine.py ──▶ data/reports/
                                                                              │
                         paper.py (live quotes, simulated fills) ──▶ data/paper/{state.json,ledger.jsonl}
```

### 1. Data layer (`http.py`, `venues/`, `ingest.py`, `store.py`)

- **Cached client.** Per-host rate limiting. Retries with exponential backoff on 429/5xx,
  honoring `Retry-After`. Each response is stored with the request that produced it. The
  cache key hashes the **full URL and params**, so different endpoints never collide.
  Immutable requests (settled data) are served from disk on re-runs. Live requests (books,
  open markets) always hit the network and are written to new timestamped files, so a raw file
  referenced by an older manifest is never overwritten.
- **Venue quirks handled** (all observed on live responses, 2026-11):
  - **Kalshi.** Prices are `*_dollars` strings and sizes are `*_fp` strings. Historical
    candlesticks drop both suffixes. Older data lives on `/historical/*`: the split point comes
    from `/historical/cutoff`, and the historical markets endpoint ignores time filters, so
    Albert pages newest-first and stops at the window start. Candlesticks live under
    `/series/{series}/markets/{ticker}/candlesticks`. Multi-leg parlay markets (`KXMVE*`) are
    excluded. The order book lists bids only, so YES ask = 1 − best NO bid.
  - **Polymarket intl.** Data API v1 is retired; v2 wraps results in `{"data": [...]}` with
    cursor pagination. Gamma `/events` caps offset at 2000, so `/events/keyset` is used.
    Gamma `/markets` returns only open markets unless `closed=true` is passed. `prices-history`
    allows at most a 15-day span per request.
  - **Polymarket US.** Each market has a long and a short side. Resolution is read from
    `outcomePrices` once status is `MARKET_STATUS_RESOLVED`, because `/settlement` is missing
    for many resolved markets. Pages sometimes return one row short, so paging only stops on
    an empty page. Sports props run to hundreds of thousands of markets per fortnight, so only
    game moneylines are ingested.
- **Typed tables.** Prices are `Float64`, timestamps are UTC `Datetime`, and lists are native
  lists. Tables are upserted by key, so re-ingesting a window refreshes it without losing other
  windows.
- **Zero rows is a failure.** An ingest that ends with an empty table raises an error instead
  of reporting success.

### 2. Cross-venue matching (`matching.py`)

Only **exact** matches are used for trading:

| Pair | Rule | Validation on settled markets |
|---|---|---|
| Kalshi ↔ Polymarket intl | Kalshi `KXNBAGAME-26OCT30LALGSW-LAL` ↔ intl `nba-lal-gsw-2026-10-30`: same team codes, same US-local date. The intl outcome order follows the slug. Two-way leagues only (NBA, NFL, NHL, MLB, NCAAF, NCAAMB, WNBA). | **76/76** agree on the winner |
| Polymarket US ↔ intl, by slug | US event slug minus its product prefix (`aec-`) equals the intl event slug, and the US long side is one of the two intl outcomes. | **270/270** agree |
| Polymarket US ↔ intl, by names | Same league and date, and the US {long, short} team names equal the intl outcome names. Needed because the two sites use different team codes (`ny` vs `nyk`). | **901/904** agree (3 disagreements, all minor hockey or esports) |

Kalshi ↔ Polymarket US is built *through* the intl market, which is the only side whose team
codes match Kalshi's. Fuzzy title matching (`--fuzzy`) writes candidates for manual review and
is never traded.

### 3. Research (`research/`)

- **Calibration** (`calibration.py`). One observation per settled market: the executable YES
  ask and bid from the **last hourly candle at or before `close − horizon`**. A test enforces
  that no later candle is ever used. Tables report hit rate vs price and the realized after-fee
  edge of buying YES at the ask or NO at 1 − bid, with standard errors, by category and 5c
  bucket. The `CalibrationModel` shrinks the market price toward the training bucket's hit
  rate: `prob = (n·hit + k·mid)/(n + k)`, with k = 200.
- **Wallets** (`wallets.py`). Each trade is valued as if held to resolution against the price
  the wallet paid (a BUY at p earns `won − p` per share; a SELL at p earns `p − won`).
  Metrics are aggregated **per market**, not per fill. Wallets are ranked by mean per-market
  ROI shrunk toward 0 (30 pseudo-markets), **never by win rate**. Two-sided wallets are flagged
  as market-maker-like and excluded. **Persistence test:** score on markets first traded
  before a split date and evaluate on markets first traded after it.
- **Follow signals** (`follow.py`). The first BUY of at least $50 per (wallet, market, outcome)
  in the test window, from wallets with positive window-A edge. Entry is the market price
  `delay` minutes later plus a 1c half spread, never the wallet's own price.
  `prob = wallet price + wallet's shrunk window-A edge`.
- **Gaps** (`gaps.py`). On exact Kalshi ↔ Polymarket matches, compares Kalshi hourly bid/ask
  with Polymarket price history and computes the locked-in profit of buying both sides after
  fees. This is an indicative screen; the paper trader checks real simultaneous books.

### 4. Backtester (`backtest/engine.py`)

An event-driven engine over *signals* (time, venue, market, side, executable price, model
probability, resolution time, payoff). It:

- processes signals in time order and settles positions as their resolution time passes,
- keeps **separate cash per venue**, since money cannot move between venues instantly,
- sizes with fractional Kelly and the shared risk caps (`risk.py`),
- charges each venue's fee on the **actual contract count**, so per-order rounding is
  modeled,
- never adds to a held market; applies a daily loss stop and a drawdown kill switch,
- never reads `payoff` before settlement (a test checks that sizing is identical whether a
  bet wins or loses).

Reports include metrics, per-strategy attribution, fee drag, skip reasons, every fill, an
equity curve, and a **Monte Carlo** bootstrap of the realized bets (p5/p50/p95 final bankroll,
probability of loss, probability of halving).

### 5. Paper trading (`paper.py`)

`albert paper run` ticks every `--interval` seconds:

1. **Settle** open positions whose markets have resolved (Kalshi `/markets/{ticker}`,
   Polymarket US market status).
2. **Risk checks**: drawdown kill switch and daily loss stop.
3. **Strategies**:
   - **`longshot`**: Kalshi markets closing about `--horizon` hours from now, priced by the
     calibration model fit on all ingested history. Order books are read live.
   - **`wallet_follow`**: polls the followed wallets' Polymarket intl trades since the last
     tick. A new BUY of at least $50 on a game moneyline is mapped to the Kalshi ticker and
     the matched Polymarket US market, and bought on whichever is cheaper after fees.
   - **`xvenue`**: open game moneylines matched Kalshi ↔ Polymarket US (through intl). Buys
     both sides when combined cost plus fees is below $1 by at least 1c per contract. Each
     tick logs a `scan` record with candidate counts and the best margin seen.
4. **Fills are simulated** at the live best ask, capped at the displayed size, using the same
   `size_order()` as the backtester.

State is in `data/paper/state.json`. Every fill, skip (with reason), scan and settlement is
appended to `data/paper/ledger.jsonl`.

---

## Strategies

| Strategy | Hypothesis | Venue | Status |
|---|---|---|---|
| `longshot` | Prices are miscalibrated in some category × price buckets (classic favorite-longshot bias). | Kalshi | research + backtest + paper |
| `wallet_follow` | Some wallets have persistent skill that survives a copy delay. | Kalshi or Polymarket US (cheaper) | research + backtest + paper |
| `xvenue` | The same game is priced differently on Kalshi and Polymarket US by more than fees. | both | research (indicative) + paper |

---

## Fees

As of 2026-11-03 (`src/albert/fees.py`). **Re-check the official pages before trading live;
both venues change fees.**

| Venue | Taker | Maker | Rounding |
|---|---|---|---|
| Kalshi | `0.07 × mult × C × P × (1−P)` | `0.0175 × …` only on `*_with_maker_fees` series, else 0 | Exact to $0.000001, with balance rounding and rebates. Albert defaults to the **conservative** round-up to $0.01 per order. |
| Polymarket US | `0.0695 × C × p × (1−p)` (0.10 for table tennis) | 0 at tier 1 (rebates start at tier 2 and are ignored) | Banker's rounding to $0.01 |

Examples: 100 contracts at 50c costs $1.75 on Kalshi and $1.74 on Polymarket US. One contract
at 50c costs $0.02 on Kalshi (conservative), and one contract at 95c costs $0.00 on Polymarket
US.

---

## Risk and sizing

`RiskConfig` defaults (`src/albert/risk.py`), shared by the backtester and the paper trader:

| Setting | Default | Meaning |
|---|---|---|
| `kelly_fraction` | 0.25 | Quarter Kelly on the after-fee edge |
| `max_bet_fraction` | 5% | Per-order cap as a share of the venue's bankroll |
| `max_bet_dollars` | $10 | Hard per-order cap |
| `max_event_fraction` | 10% | Cap on exposure to one event |
| `min_edge` | 3c | Required edge per contract after fees, re-checked with the actual rounded fee |
| `min_price` / `max_price` | 3c / 97c | Skip extreme prices, where fees and ticks dominate |
| `daily_loss_fraction` | 10% | Stop opening positions for the day |
| `drawdown_kill` | 50% | Halt if equity falls below half of its peak |

With a $50 venue bankroll, quarter Kelly and a 5% cap, typical orders are $1-2.50. At that
size, conservative fee rounding alone can remove a 1-2c edge. That is why the minimum edge is
re-checked after rounding.

---

## Data layout

```
data/
  raw/<venue>/<key>.json          every API response + the request that produced it
  tables/*.parquet                typed tables (see below)
  runs/*.json                     ingest manifests (params, row counts, raw files used)
  reports/<name>-<ts>/            report.md + parquet extracts + equity.png
  paper/state.json, ledger.jsonl  paper account and full audit log
```

| Table | Key | Contents |
|---|---|---|
| `kalshi_series` | series_ticker | category, fee_type, fee_multiplier |
| `kalshi_markets` | ticker | settled non-financial markets: times, result, last/bid/ask, volume, category, fee params |
| `kalshi_trades` | trade_id | price, size, taker side |
| `kalshi_candles` | ticker, end_ts, period | YES bid/ask/price close, volume, OI |
| `pm_markets` | condition_id | Polymarket intl markets: outcomes, token ids, tags, winner_index |
| `pm_trades` | tx, wallet, token, side, price, size | wallet trades |
| `pm_leaderboard` | wallet, period, category | PnL/volume ranks (the sampling frame) |
| `pm_price_history` | token_id, ts | price points |
| `pmus_markets` / `pmus_open_markets` | slug | Polymarket US: long/short sides, settlement, fee coefficient, min qty |
| `match_kalshi_pm`, `match_pmus_pm` | n/a | exact cross-venue matches |
| `wallet_scores`, `signals_wallet_follow` | n/a | research outputs |

---

## Results so far

RESULTS_PLACEHOLDER

---

## Known limitations

- **Hourly candles.** Kalshi calibration and gap research use hourly candle closes. That is
  fine for "is this price miscalibrated 2h out" but too coarse for latency-sensitive edges.
- **Polymarket price history has no bid/ask.** Intl prices are mids or last trades. Albert
  adds a 1c half spread, which can understate costs in thin markets.
- **The wallet sampling frame is biased.** Wallets come from *today's* leaderboards, which
  include their later results. Any "leaderboard wallets earn X%" number is biased by
  hindsight. Only the *within-sample ranking* test (window A vs window B) is a fair test.
  Very active wallets are capped at 5,000 trades per month.
- **The wallet-follow backtest prices fills on Polymarket intl.** It uses intl prices with
  Polymarket US fees as a proxy. Only game moneylines can actually be executed on Kalshi or
  Polymarket US, and the paper trader does that mapping live.
- **Cross-venue resolution risk.** 3 of 904 name-matched pairs resolved differently across
  venues. xvenue only trades the major leagues Kalshi lists, where exact matching was 100%.
- **College basketball** has no Polymarket intl tag, so live xvenue covers NBA, NFL, NHL, MLB,
  NCAAF and WNBA.
- **Soccer and other three-way markets** are not matched.

---

## Development

```bash
python3 -m pytest -q          # offline; parsers are tested against recorded real responses
python3 -m ruff check src tests
albert capture                # re-record fixtures when a venue changes its schema
```

Ground rules (from `AGENTS.md`):

1. Parsers are tested only against recorded real responses (`albert capture`).
2. A smoke run that returns zero rows is a failure.
3. Raw data is append-only and re-runs reuse the cache.
4. No look-ahead.
5. Fills happen at executable bid/ask with real fees.
6. Win rate is not edge.
7. Public information only.
8. Live trading is gated.

Code map:

| Path | Purpose |
|---|---|
| `src/albert/http.py` | cached, rate-limited HTTP client |
| `src/albert/venues/` | per-venue fetchers + normalizers |
| `src/albert/ingest.py` | ingest pipelines → tables + manifests |
| `src/albert/matching.py` | cross-venue matching |
| `src/albert/research/` | calibration, wallets, follow signals, gaps |
| `src/albert/backtest/` | strategies → signals, event-driven engine, Monte Carlo |
| `src/albert/fees.py`, `risk.py` | fee schedules, Kelly sizing, risk caps |
| `src/albert/paper.py` | paper trader |
| `src/albert/pipeline.py`, `reports.py`, `cli.py` | orchestration, report writer, CLI |
| `src/albert/capture.py` | fixture recorder |

---

## Roadmap

See `.planning/ROADMAP.md`. Phases 1-5 are built. **Phase 6 (live trading with $100) is not
started.** It should only begin after paper trading has run for at least 2 weeks or 100 signals
and the results agree with the backtest. It will need Kalshi RSA-PSS and Polymarket US Ed25519
order clients, an explicit enable flag, per-venue caps, a global kill switch and an
order/fill audit log.
