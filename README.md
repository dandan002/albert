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
python3 -m pytest -q                        # 41 tests, offline, against recorded API responses
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
albert research literature
albert backtest run --strategy longshot --split 2026-10-01 --prior 20 --universe population
albert backtest run --strategy wallet_follow --split 2026-08-15 --delay 5

# 5. Paper trading on live quotes
albert paper reset --kalshi-cash 50 --pmus-cash 50
albert paper run --interval 300 --prior 20 # Ctrl-C to stop; state persists
albert paper status
```

`albert` is installed as a console script. Without installing, use `python3 -m albert`.

---

## Commands

| Command | What it does |
|---|---|
| `albert ingest kalshi` | Settled non-financial Kalshi markets that closed in `[start, end)`, routed across the live/historical cutoff. `--trades` and `--candles` fetch per-market history for the top `--max-markets` by volume (`--per-category` applies the limit within each category). |
| `albert ingest kalshi-detail` | Trades/candles for a *seeded random* sample per category (`--sample-per-category`, volume band), every leg of `--multi-events` multi-outcome events, and/or *every* game of `--series` (one leg per event, no volume filter). Use this, not `--max-markets`, for research samples: ranking on final volume selects upsets. |
| `albert ingest kalshi-events` | Event metadata (`mutually_exclusive`, leg count) for multi-leg events with candles. |
| `albert ingest kalshi-open` | Snapshot of open non-financial Kalshi markets. |
| `albert ingest pm-markets` | Polymarket intl events and markets with resolutions (`--open` for live markets). |
| `albert ingest pm-wallets` | Leaderboard wallets (per category) and their trades, paged month by month, then backfills resolutions for every market they traded. |
| `albert ingest pmus` | Polymarket US markets: game moneylines plus politics/culture/science. Closed ingests need `--start/--end`, because the venue has more than 2M closed markets. |
| `albert match` | Builds `match_kalshi_pm` and `match_pmus_pm` (and, with `--fuzzy`, review-only title candidates). |
| `albert research calibration` | Favorite-longshot tables: hit rate vs price, after-fee edge of buying YES or NO, by category. |
| `albert research wallets` | Wallet skill scores and an out-of-sample persistence test. |
| `albert research gaps` | Kalshi vs Polymarket price gaps and locked-in opportunities on exact matches. |
| `albert research literature` | Re-tests five published effects on our data: maker vs taker returns, sum-of-legs arbitrage, calibration slope by category (plus a full-population sports check), rolling-minimum mean reversion, pre-close move reversal. |
| `albert backtest run` | Runs `longshot`, `slope` and/or `wallet_follow` through the backtester. `--anchor scheduled` (default) places Kalshi decisions at scheduled expiry − horizon; `--anchor close` reproduces the old leaky timing for comparison. `--universe population` restricts Kalshi to the volume-blind game sample. Writes a report with metrics, Monte Carlo, fills and an equity curve. |
| `albert ingest weather` | NWS CLI reports, ASOS observations and NBM guidance (IEM archive) for every Kalshi temperature city. |
| `albert backtest weather` | NWS model vs Kalshi daily-high brackets: calibration vs market per decision slot, model bets through the backtester. |
| `albert backtest maker` | Resting bids on Kalshi favorites, filled only by later real trades into our price (`--horizon`, `--ttl`, `--size`, `--cash`, `--category`). |
| `albert paper reset/run/status` | Paper trading account (see [Paper trading](#5-paper-trading-paperpy)). Default strategy: `maker`. |
| `albert compact-raw` | Compresses legacy plain-JSON raw cache files to `.json.zst` (lossless, verified). |
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
   │ append-only raw cache: data/raw/<venue>/<sha256(url+params)>.json.zst    │
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
   - **`maker`** (default): resting bids on Kalshi Weather/Sports favorites. Every
     `--interval` it applies new *real* trades to each resting order (the same fill rule as
     `backtest maker`) and cancels orders after `--maker-ttl` hours (default 20) or at close.
     Every 10 minutes it scans open markets whose scheduled expiry is `--maker-horizon` hours
     away (default 24), quotes the favorite at best bid + 1c (or joins the best bid when the
     spread is 1c), and reserves the collateral. Rules: one order per underlying game (the
     moneyline, spread and totals of a match are one bet), 5% of free Kalshi cash per order,
     capped at `--maker-size` dollars. If the drawdown or daily-loss stop trips, every resting
     order is cancelled.
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

State is in `data/paper/state.json`. Every order, fill, cancel, skip (with reason), scan and
settlement is appended to `data/paper/ledger.jsonl`.

**Running as a service.** `paper run` is built to run unattended: SIGTERM/SIGINT finish the
current tick and save state; an error in one tick is logged (`tick_error`) and retried with
backoff instead of crashing; each tick prints one JSON line to stdout and rewrites
`data/paper/heartbeat.json`. Live responses are *not* written to the raw cache unless
`ALBERT_RECORD_LIVE=1` (they would add hundreds of MB a day, and the ledger already records
every decision). A `Dockerfile` and a Compute Engine guide are in [`deploy/gcp.md`](deploy/gcp.md).

---

## Strategies

| Strategy | Hypothesis | Venue | Status |
|---|---|---|---|
| `longshot` | Prices are miscalibrated in some category × price buckets (classic favorite-longshot bias). | Kalshi | research + backtest + paper. **No edge** once look-ahead and volume selection are fixed (Results §3). |
| `slope` | Per-category logistic recalibration (prices too close to / too far from 50%, arXiv:2602.19520). | Kalshi | research + backtest. **No edge** on clean data (Results §3). |
| maker favorites | Resting bids on ≥ 50c contracts earn the maker premium (Bürgi et al. 2025). | Kalshi | research only (Results §3a). Next to build: maker paper mode. |
| `wallet_follow` | Some wallets have persistent skill that survives a copy delay. | Kalshi or Polymarket US (cheaper) | research + backtest + paper. **No edge** (Results §1-2). |
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
  raw/<venue>/<key>.json.zst      every API response + the request that produced it (zstd JSON)
  tables/*.parquet                typed tables (see below)
  runs/*.json                     ingest manifests (params, row counts, raw files used)
  reports/<name>-<ts>/            report.md + parquet extracts + equity.png
  paper/state.json, ledger.jsonl  paper account and full audit log
```

**Storage formats.** Analysis tables are Parquet (columnar, compressed, fast to scan). The raw
cache stays one file per API response so any run can be replayed byte-for-byte from its
manifest, but each file is zstd-compressed JSON: about 20x smaller than plain JSON (the
cache was 20 GB uncompressed) and no slower to read, because there is less disk I/O.
Caches written before this change are plain `.json`. They are still read, and
`albert compact-raw` converts them losslessly (each file is verified before the original is
removed). Old manifests that list `.json` paths still resolve.

| Table | Key | Contents |
|---|---|---|
| `kalshi_events` | event_ticker | mutually_exclusive flag and listed leg count (for sum-of-legs arbitrage) |
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

All numbers below come from the data ingested on 2026-11-03 and are reproducible with the
Quick start commands (reports are in `data/reports/`). **None of this is evidence of a
tradeable edge yet.** Read it as "what survived a first honest test".

### Data collected

| Source | Rows | Window |
|---|---|---|
| Kalshi settled non-financial markets | 1,298,805 (all volumes; 94%+ Sports) | 2026-08-15 to 2026-11-01 |
| Kalshi hourly candles | 2.01M for 9,523 markets: top 600 per category by volume (biased, see below), a seeded random sample of 300 per category, all legs of 400 multi-outcome events, and one leg of every NFL/NBA/NCAAF/MLB/NHL/WNBA game | same |
| Kalshi trades | 2.45M (newest 5,000 per market) for 3,615 markets: 3,515 in the random volume band (used) + 20 top-volume (excluded from the maker study) | 2026-07-30 to 2026-10-31 |
| Kalshi events | 993 multi-leg events (552 mutually exclusive) | same |
| Polymarket intl wallet trades | 743,781 from 134 leaderboard wallets (88% on resolved markets) | 2026-05-01 to 2026-11-02 |
| Polymarket intl markets | ~140k with resolutions | mostly 2026-05 to 2026-11 |
| Polymarket US markets | 17,429 (moneylines + politics/culture/science) | 2026-10-20 to 2026-11-02 |

### 1. Smart wallets: no persistent skill found

Wallets were scored on markets first traded before a split date and evaluated on markets first
traded after it (`albert research wallets`):

| Split | Wallets evaluated | Spearman(A score, B ROI) | Top-10 ROI in B | Top-20 ROI in B | All-wallet ROI in B |
|---|---|---|---|---|---|
| 2026-07-15 | 56 | −0.07 | +0.5% | +1.6% | +3.5% |
| 2026-08-15 | 56 | −0.12 | +2.7% | +2.4% | +3.8% |
| 2026-09-15 | 48 | +0.11 | +2.3% | +2.8% | +5.0% |

Within this sample, **past performance does not predict future performance**. The rank
correlation is about zero, and the top-ranked wallets did *worse* afterwards than the average
leaderboard wallet. The positive all-wallet ROI is biased by hindsight, because wallets were
picked from today's all-time leaderboard.

### 2. Copying wallets loses money after delay and fees

`albert backtest run --strategy wallet_follow --split 2026-08-15` copies the top-20 window-A
wallets' first BUY per market in window B, at the price N minutes later + 1c, with Polymarket US
fees, $100 split $50/$50, quarter Kelly:

| Copy delay | Trades | End equity | ROI per $ staked | Fees | Monte Carlo P(loss) |
|---|---|---|---|---|---|
| 1 min | 544 | $68.14 | −7.3% | $16.20 | 90% |
| 5 min | 416 | $84.95 | −2.2% (gross +1.6%) | $25.84 | 72% |
| 30 min | 283 | $51.03 | −31.8% | $7.43 | 100% |

The model expected a +14.6% edge per trade, and the realized edge was about zero before fees.
**Verdict: do not copy-trade.** The 1-minute and 5-minute rows both use 5-minute price buckets,
so their difference is noise.

### 3. Calibration / favorite-longshot bias: the earlier edge was two biases

The first version of this section reported a "small, fragile" favorite-longshot edge
(longshot strategy: $109-119 from $100 at 2h before close). **That result does not survive
two corrections found on 2026-11-03, and is withdrawn:**

1. **Look-ahead in the decision time.** Observations were taken at `close_time − horizon`.
   For a settled Kalshi market, `close_time` is when it *actually* closed: the game ended, the
   word was said, a winner was declared. 30-45% of Sports, Politics and Mentions markets close
   more than an hour away from schedule, so "2h before close" leaked the event's duration (a
   game still running 2h before its end is more often close, which favors the underdog). The
   default is now `expected_expiration_time − horizon`, which is fixed at listing, and markets
   that had already closed by the decision time are dropped. Elections and Mentions have no
   short-horizon scheduled time (their expected expiration is a backstop weeks out), so they
   drop out of short-horizon tests entirely.
2. **Selection on volume.** The candle sample was the top 600 markets per category by
   *final* volume. Upsets draw heavy in-game trading, so the sample was full of upsets: 92% of
   its sports favorites were above their series' median volume, and favorites in the top
   volume quartile under-performed their price by 14c versus 3-5c in the bottom half. A
   volume-blind sample (one leg of *every* game in six leagues, `ingest kalshi-detail
   --series`) fixes this.

On the clean sample, Kalshi game prices are calibrated (`albert research literature`,
section 3b; 2,286 games, Aug 15 to Oct 31):

| Horizon | Sample | Markets | Calibration slope (1 = calibrated) | Favorite win rate − price |
|---|---|---|---|---|
| 24h before scheduled expiry | full population | 2,286 | **1.006** ± 0.060 | +0.2c |
| 24h | volume-selected | 221 | 0.380 ± 0.153 | −11.2c |
| 6h | full population | 2,286 | **1.014** ± 0.055 | +1.2c |
| 6h | volume-selected | 224 | 0.412 ± 0.153 | −9.6c |

Backtests (train before 2026-10-01, trade October, $50 per venue, `--prior 20` for longshot):

| Strategy | Timing / universe | Trades | End equity ($100 total) | ROI per $ |
|---|---|---|---|---|
| longshot 2h | old leaky `close` anchor, biased sample | 206 | $172.94 | +32.9% |
| slope 24h | `scheduled` anchor, biased sample | 254 | $343.50 | +33.2% |
| longshot 2h | `scheduled`, population | 217 | **$88.14** | −4.4% |
| longshot 24h | `scheduled`, population | 244 | **$98.28** | −0.8% |
| slope 2h / 24h | `scheduled`, population | 4 / 0 | $97.28 / $100.00 | n/a |
| longshot 2h / 24h | `scheduled`, all markets | 140 / 145 | $79.85 / $81.32 | −16.0% / −15.6% |

The first two rows show how large the biases are: an apparent 3.4x on the bankroll in one
month, from nothing. **No calibration strategy has an edge once both are fixed.**

### 3a. Strategies from the literature, re-tested (`albert research literature`)

I searched arXiv and the web for published prediction-market strategies and re-tested the
five that our data can test, after fees, with standard errors clustered by event:

| # | Source | Claim | Our result |
|---|---|---|---|
| 1 | Bürgi, Deng & Whelan, *Makers and Takers: The Economics of the Kalshi Prediction Market* (CEPR DP20631, 2025) | Makers earn more than takers; makers buying at ≥ 50c earn about +2.6% | **Replicates.** Makers +1.0% per $ (t = 1.4), takers −2.7% (t = −3.5). Makers buying ≥ 50c: Weather +3.0% (t = 4.0; +1.6/+2.9/+3.4% in Aug/Sep/Oct), Sports +4.1% (t = 3.1; +6.6/+2.7/+4.2%). Elections −12%, Politics −1%, others noise. |
| 2 | Saguillo et al., *Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets* (arXiv:2508.03474) | In multi-outcome events the legs' prices should sum to $1; deviations were worth about $40M on Polymarket | **Not on Kalshi.** In 258 complete mutually-exclusive events (23,935 event-hours) the guaranteed side (buy every NO) cleared fees in 15 hours of 5 events, at most 2.8c. The "buy every YES below $1" cases are not riskless: Kalshi's `mutually_exclusive` means *at most* one leg wins, and one such event resolved with no winner (−60c). |
| 3 | *Calibration is conditional* (arXiv:2602.19520) | Political markets are under-confident (favorites cheap) | Leans that way, not significant: Politics slope 1.08 (24h) to 1.35 (2h) ± 0.12-0.25. Polymarket intl overall 0.97 ± 0.07. Sports is calibrated once sampling is fixed (above). |
| 4 | Quantpedia, *Exploiting Mean-Reversion in Decentralized Prediction Markets* (Polymarket) | Buy at the X-day rolling low, hold Y days | **Loses heavily as a taker.** −13.5% to −15.3% per $ for every lookback/hold tried (24-72h / 6-72h), t ≈ −26 to −33. Crossing the spread twice costs more than the bounce. The author also says it needs passive (maker) fills. |
| 5 | Moskowitz, *Asset Pricing and Sports Betting* (J. Finance 2021) | Pre-event moves over-react and are reversed by the outcome | Nothing robust. All-category slope −0.04 (t = −1.2) for moves from 26h to 2h, −0.05 (t = −1.1) from 72h to 24h. Isolated cells (Mentions, Elections) at |t| ≈ 2-3 among ~15 tested, which is what chance produces. |

**The one survivor is structural: be the maker, not the taker.** Makers who posted resting
bids that got filled on favorites (≥ 50c) in Weather and Sports earned 3-4% per dollar, every
month. This measures *other people's realized maker fills*, so it already includes their
adverse selection. It does not show that a new order would get filled at the same rate:
queue position, cancellations and fill probability are not modeled. Testing it needs a
**maker paper-trading mode** (post bids, count a fill only when later trades print through
our price) before any real money.

Other ideas from the search that we could not test with this data: weather-forecast models
against Kalshi temperature brackets (needs archived NWS/GFS forecasts; practitioners report
that the market already prices the forecast and the bracket edges are easy to get wrong), and
Polymarket on-chain combinatorial arbitrage (not tradable from the US).

### 4. Cross-venue gaps: venues agree; no free money

- Matching is accurate. Kalshi ↔ Polymarket intl: **882/882** settled pairs agree on the
  winner. Polymarket US ↔ intl: 270/270 by slug and 901/904 by names.
- Pre-game Kalshi vs Polymarket prices on 165 matched games: median |gap| **0.5c**, 90th
  percentile 1.5c. The few "arbitrage" prints left after filtering are stale placeholder prices
  (e.g. 0.50 two weeks before kickoff), not executable books.
- Live order-book scans (paper `xvenue`, 46 matched games quoted on both venues): the best
  combined cost of both sides after fees was **2.6-2.9c above $1** on every tick so far.

### 5. Maker backtest: positive, not yet significant (`albert backtest maker`)

Section 3a shows that *other people's* filled maker bids on favorites earned a premium. The
maker backtest asks whether *our* bids would have filled, and what the fills earned:

- **Order:** at scheduled expiry − h, bid for the favorite at best bid + 1c (we become the
  best bid, first in queue) or join the best bid if the spread is 1c. One order per game.
- **Fill:** only from later real Kalshi trades in which a taker *sold* into our side at our
  price (if we improved) or through it (if we joined, since the queue ahead must clear
  first). Held to settlement, maker fee. Unfilled orders cancel at the TTL or at close.
- **Sample:** 3,515 randomly sampled Weather and Sports markets with trades (volume
  500-200k), Aug-Oct 2026. Standard errors clustered by *game*.

| Post h before expiry / cancel after | Fill rate | Maker ROI on fills | t | Taker, same bets | $50 bankroll |
|---|---|---|---|---|---|
| 24h / until close (sports: in-game) | 91% | +4.5% | 1.9 | +2.7% | $91 |
| 6h / until close | 92% | −2.0% | −1.1 | −2.4% | $16 |
| 24h / 20h (cancel pre-game) | 66% | +7.6% | 2.8 | +2.7% | $119 |
| **24h / 20h, one order per game** | 60% | **+4.6%** | **1.0** | +0.6% | **$72** |

The first three rows count each event separately. A match's moneyline, spread and totals are
one bet, though, and counting them separately overstated both the sample and the t-stat.
With one order per game (last row, the rule the paper trader uses) the result is still
positive and still beats taking the same bets, but **it is not statistically distinguishable
from zero**, and it varies by month (+0.3%, +13.2%, −0.3% in Aug/Sep/Oct). Resting into
in-game trading is clearly bad (6h row): fills arrive exactly when the favorite is collapsing.

**Holdout (2026-11-03).** The 24h / 20h, one-order-per-game rule was fixed in advance and
run on 2,978 *new* randomly sampled markets (seed 11) that no earlier run had seen:

| | In-sample | Holdout |
|---|---|---|
| Maker ROI on fills | +4.6% (t = 1.0) | +5.1% (t = 1.2) |
| Taker, same bets | +0.6% | +4.4% (t = 1.6) |
| $50 bankroll | $72 | $69 |
| Joined a 1c-spread best bid (subgroup) | +12.5% | +17.7% (t = 2.6) |
| Improved the best bid by 1c (subgroup) | +0.7% | −1.3% |

**But the trade sample itself is biased.** Trades were sampled from markets with *final*
volume 500-200k. In the volume-blind game population (section 3), buying the favorite at
the ask 24h out earns **+4.5c in that band and −1.9c above 200k** (−0.4c overall), because
the busiest games, where upsets happen, are cut off. The "newest 5,000 trades" coverage
rule drops busy markets as well. So the positive taker benchmark, and probably most of the
maker gain (including the "joined" subgroup), come from selection. The clean test is
`ingest kalshi-trade-windows` (all trades in each order's window, for one leg of *every*
game) followed by `backtest maker --universe population`.

**Clean test: no maker edge on major-league games.** All 2,357 NFL/NBA/NCAAF/MLB/NHL/WNBA
games Aug 15 - Oct 31, 3.85M trades with complete coverage of every order window, 24h / 20h,
one order per game:

| Queue mode | Fills | Maker ROI on fills | t | Taker, same bets | $50 bankroll |
|---|---|---|---|---|---|
| any (paper-trader rule) | 1,179 | **−1.7%** | −0.8 | −2.4% | **$27.53** |
| join a 1c-spread best bid | 828 | −3.5% | −1.4 | −2.7% | $27.68 |
| improve the best bid by 1c | 351 | +2.5% | 0.7 | −0.9% | $54.00 |

By month (any): +2.1%, +4.2%, −8.7% (Aug/Sep/Oct). The "joined +17%" subgroup and the
positive holdout were the volume-band selection. Being a maker is still less bad than
taking, which matches the literature, but there is no positive edge to compound on
major-league favorites. Weather and minor sports (esports, KHL, cricket and others, which the
paper trader also quotes) have not been tested volume-blind.

Rule change from live testing: ticker start times (`ticker_start`, US Eastern, validated on
~80k games) now cap every order at 15 minutes before the game. A fixed 20h TTL let orders
rest into cricket and KBO games, whose scheduled expiry is 4-8h after the start.

### 6. Weather: NWS data vs Kalshi daily-high brackets (`albert backtest weather`)

Kalshi's `KXHIGH*` markets (24 cities, six 2°F brackets per city and day) settle on the
NWS Daily Climate Report (CLI). They trade until about 1am ET the next day, so most of the
day's temperature is observed while they are open. Data: CLI reports, ASOS observations
and NBM forecast guidance from the IEM archive of NWS products (`albert ingest weather`).
`api.weather.gov` keeps about a week of observations and no past forecasts, so it is the
live source only. There are hourly Kalshi prices for all 10,670 legs (1,783 events, Aug 16 -
Oct 30, `ingest kalshi-detail --all-legs`).

**Data checks.** Bracket outcomes recomputed from the CLI match Kalshi's settlement on
99.98% of legs (2 mismatches, one Miami day). The CLI day runs on local *standard* time.
The CLI high is at or above the max hourly observation 99.8% of the time (+0°F on 38% of
days, +1°F on 51%, +2°F on 9%). NBM's max-temperature error is about 2.4°F at 1-2 days,
with fat tails: regional busts of about 10°F, e.g. the whole Northeast on Oct 17.

**Model** (`research/weather.py`). The high is max(observed max so far + U, rest-of-day
max). U is the CLI-minus-observed offset. The rest-of-day max is centred on the NBM
forecast, floored at the latest observation, and uses a fat-tailed error with per-station
bias, all fit on days before Oct 1. Probabilities are Platt-recalibrated on training legs.
Only NBM runs available at the decision time (runtime + 2h) and earlier observations are
used. Decision slots are 3pm the day before, and 10am, 1pm and 4pm local on the day.

| Slot | Brier: model | Brier: market mid | Model bets (flat, edge ≥ 5c) | ROI per $ | t (by date) |
|---|---|---|---|---|---|
| d-1 15h | 0.127 | **0.097** | 676 | −16.8% | −5.1 |
| d 10h | 0.123 | **0.083** | 682 | −26.5% | −5.8 |
| d 13h | 0.112 | **0.068** | 620 | −20.2% | −3.2 |
| d 16h | 0.070 | **0.024** | 460 | −35.6% | −4.8 |

$50 bankroll: **$19.97**, with the drawdown stop hit on Oct 5. Adding the model's
probability to the market price in a logistic fit makes October predictions *worse* at
every slot, so the NWS model adds no information the market lacks.

**Already-decided brackets are priced correctly.** Over 34,539 leg-hours where the observed
max was already at least 1°F above a bracket's top, the average YES bid was 0.02c (0.06% of
hours ever bid ≥ 2c). "X or above" brackets already reached were offered at $1.00 every
time. At hourly resolution the market absorbs observations immediately.

**Conclusion:** Kalshi weather traders already use more than one NBM run plus live
observations. A slower, public NWS feed is not an edge. What remains untested: sub-minute
latency on new observations (needs a live 1-5 minute feed and order-book polling; this is
competitive and practitioners report bots already doing it), and a volume-blind maker test
on weather brackets.

### 7. Minor sports and esports (`research sports`, `backtest maker --anchor start`, `research esports-reference`)

The goal changed on 2026-11-03 to *consistent* returns, with thin markets in scope. Sample: 21
minor series whose tickers encode the start time (esports, table tennis, e-soccer, darts,
T20 cricket, KHL/KBO/NPB, NCAA volleyball). Up to 500 random events per series, chosen
without looking at volume; one leg per event; Aug 15 - Oct 31. Decisions are at the
scheduled start minus a lead, and a market needs a two-sided quote then (known at the
decision time, unlike final volume).

- **Pricing is under-confident before the game.** Pooled calibration slope 1.17 ± 0.05
  (t = 3.5) at 1h before the start, 1.13 at 15 min. Favourites win more often than priced.
  Major leagues are calibrated (slope 1.006).
- **But it does not pay as a taker.** Buying the favourite at the ask 1h before the start:
  −1.0% per $ after fees (t = −0.9, 3,834 bets, 78 dates), Aug/Sep/Oct −2.5% / +0.2% /
  −1.6%, positive in 5 of 12 weeks. The bias is smaller than the spread plus fee. Only 90c+
  favourites were positive (+2.3%, t = 1.9), one band of five.
- **Maker (bid for the favourite before the start, cancel 15 min before), complete trade
  windows:** posting 12h before +0.3% (t = 0.1), 6h before −0.8%, 2h before +1.6% (t = 0.8).
  Within the 2h version, *improving* the bid by 1c earned +4.7% (t = 1.9, 775 fills) and
  joining lost 4.2%. That subgroup is post hoc, so a holdout on a fresh random sample
  (seed 99) with the rule fixed in advance is running.
- **Polymarket as a reference price for Kalshi esports.** 872 CS2/LoL/Dota 2/Valorant
  matches were matched by start time and both team names; the venues agree on the winner
  in 100%. Before the start the two prices differ by about 1c on average, are equally
  accurate (Brier 0.22 each), and the gap's predictive slope flips sign across lead times.
  Buying on Kalshi when Polymarket's price beats Kalshi's ask plus fee produced 7-115 bets
  per cell with no consistent sign. The one strong cell (+42% on 20 bets) is 1 of 12 tried.
  Kalshi esports prices track Polymarket closely.

### What this means for the $100 goal

Nothing found so far supports "exponential growth". The honest summary:

- **The insider/high-win-rate thesis fails** on this data: skill does not persist out of
  sample, and copying it after a realistic delay loses money.
- **Cross-venue and sum-of-legs arbitrage** are priced out by fees.
- **Taker strategies on public prices have no edge.** The calibration "edge" from Phase 5 was
  look-ahead plus sample selection, and mean reversion and move-fading lose after costs.
- **Being a maker loses less than taking, but has no positive edge on major-league
  games** once the sample is volume-blind (−1.7% per filled dollar on 1,179 fills). The
  apparent +3-5% premium came from samples that cut out the busiest (upset-heavy) games.
- **NWS weather data does not beat Kalshi weather prices.** The market is better calibrated
  than a calibrated NBM + observations model at every decision time, and already-decided
  brackets are priced at 0c/100c.
- Do not fund anything. The maker paper trader keeps running as a live check of the fill
  model (live fill rates vs simulated), not as a money-maker.

---

## Known limitations

- **Research samples must be volume-blind.** `ingest kalshi --max-markets` ranks markets by
  *final* volume, which selects upsets (section 3). Use `ingest kalshi-detail` samples
  (`--series`, or the seeded random sample), and `backtest run --universe population`.
- **Kalshi `close_time` is when a settled market actually closed.** Never anchor a
  decision on it; use `expected_expiration_time` (the default `--anchor scheduled`).
- **Maker fill model.** Fills come from real prints at or through our price. It ignores
  other makers who might join ahead of us at a newly improved price in the same instant, and
  any reaction to our own order. Fill rates are therefore optimistic, and the paper trader
  measures the same quantity live.
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
