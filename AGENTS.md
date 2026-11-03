## Project

**Albert** is a Python research CLI and trading system for **non-financial prediction-market contracts**
(politics, sports, culture, weather, science/tech, and so on). It does quantitative research on market
signals and on high-performing accounts, then runs systematic small-stake bets.

- **Bankroll:** $100 to start. The objective (changed 2026-11-03) is **consistent, positive
  returns**, not maximum growth: prefer many small, repeatable bets with stable month-to-month
  results. Low-liquidity markets are fine (we trade small). Ruin avoidance is a hard constraint.
- **Execution venues:** **Kalshi** (RSA-PSS signed API) and **Polymarket US** (the CFTC-regulated
  US exchange at docs.polymarket.us; Ed25519-signed API). The bankroll is split across both.
- **Data-only venue:** **Polymarket international** (polymarket.com) supplies on-chain wallet trades,
  positions, the leaderboard, and prices. The user is a US resident, so never place orders there and
  never build geoblock workarounds (VPNs, proxies, third-party wallets).
- **Non-financial:** exclude crypto, financial-index, rates, and economics-release contracts. The
  exact category list is decided from real Kalshi `category` / Polymarket tag values, not guessed.

See `.planning/PROJECT.md` and `.planning/ROADMAP.md`.

## Ground rules

1. **Real payloads only.** Every parser is written and tested against responses recorded from the
   live API (`albert capture ...` -> `tests/fixtures/<venue>/`). Hand-written fixtures are allowed
   only as trimmed copies of recorded ones. The previous version of this project failed because its
   fixtures were invented.
2. **A smoke test that returns zero rows is a failure,** not a pass.
3. **Raw data is append-only.** The cache key includes the endpoint path and the params. Never
   overwrite a raw file that a manifest references. Re-runs reuse the cache.
4. **No look-ahead.** Research and backtests may only use data timestamped before the simulated
   decision time.
5. **Realistic execution.** Fill at the executable bid/ask, apply each venue's own fee schedule
   (Kalshi rounds fees up per order; Polymarket US uses banker's rounding to the cent and pays
   maker rebates by tier), and respect displayed size. Re-check the official fee pages before any
   live run, because both venues change their fees.
6. **Win rate is not edge.** Rank accounts and strategies by PnL per dollar risked and by edge
   versus entry price, with out-of-sample persistence. Never rank by hit rate alone.
7. **Public information only.** Following public wallet flow is fine. Never build anything that
   trades on material non-public information or that circumvents venue rules.
8. **Live trading is gated:** paper-trading results first, then explicit user enablement, hard
   per-order and daily-loss caps, and a kill switch. Credentials come only from `.env`, never from
   logs, manifests, or fixtures.

## Stack

Python 3.12+, Typer CLI, httpx, polars + pyarrow (Parquet), pydantic for row schemas, pytest, ruff.
Local Parquet + JSON manifests; no database until volume proves the need.

## Process

Lightweight by user choice (2026-11-02): this replaces the earlier GSD workflow enforcement. Keep
short planning docs in `.planning/`, make small focused commits, and verify every data-layer
change with a live smoke run that produces non-empty, sane output.

## Commands

- `python -m pytest -q`
- `python -m ruff check src tests`
