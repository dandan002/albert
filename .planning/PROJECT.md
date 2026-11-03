# Albert: Project

Restarted 2026-11-02. The previous BTC 15-minute / Deribit backtest was discarded. Its code is in
git history before this restart.

## Goal

Find and exploit repeatable edges in **non-financial** prediction markets with a **$100** starting
bankroll through systematic small bets.

**Changed 2026-11-03:** the target is *consistent* returns (positive in most weeks/months,
small drawdowns), not exponential growth. Market volume does not matter: thin, minor markets
are in scope because our size is small. Current focus is sports, after major leagues, weather,
wallets and cross-venue arbitrage all tested efficient (README "Results so far").

## Venues (decided 2026-11-03)

| Venue | Role | Auth | Notes |
|---|---|---|---|
| Kalshi | trade | RSA-PSS | broad non-financial coverage; `category` on events |
| Polymarket US | trade | Ed25519: `X-PM-Access-Key`, `X-PM-Timestamp`, `X-PM-Signature` over `{ts_ms}{METHOD}{path}` | public data at `gateway.polymarket.us/v1`, trading at `api.polymarket.us`; keys at polymarket.us/developer after KYC; 24/7 except Thu 06:00-08:00 ET maintenance |
| Polymarket intl | data only | none | on-chain wallet signals; US geoblocked for trading |

Bankroll: about $50 on each trading venue.

Polymarket US active markets sampled on 2026-11-03: 416 sports, 45 politics, 27 finance, 12
culture (out of 500). It is mostly sports, so wallet signals from Polymarket intl must be matched
to US markets. The shared slugs (e.g. NFL games) should make that easier than matching to Kalshi.

### Fees (as of 2026-11-03; re-check before live trading)

- **Polymarket US taker:** `0.0695 × C × p × (1 − p)`. Table tennis uses 0.10. Banker's rounding to
  $0.01. Maker rebates (tiers 2-4) and taker rebates are based on the prior month's volume.
  Source: docs.polymarket.us/fees.
- **Kalshi taker:** roughly `0.07 × C × p × (1 − p)`, rounded **up** per order. Verify against
  Kalshi's current schedule.

## Research questions

1. **Smart wallets:** do some Polymarket wallets have persistent, statistically significant edge
   (not luck, and not just buying favorites)? Does following their trades stay profitable after
   realistic delay and the Kalshi execution price?
2. **Market inefficiencies:**
   - Favorite-longshot bias by category and price bucket.
   - Stale or thin markets.
   - Kalshi vs Polymarket price gaps on the same event.
   - Late-resolution drift.
3. **Small-bankroll economics:** which strategies survive Kalshi's per-order fee rounding at $1-5
   sizes? Maker (resting) orders vs taker; few high-edge bets vs many small ones.

## Sizing principles

- Fractional Kelly (start at <= 0.25 Kelly) on an edge estimate that has been shrunk toward zero.
- Hard caps: per-bet stake, per-event exposure, daily loss, and drawdown kill switch.
- The first $100 mainly buys evidence. Scale only after live results confirm the backtested edge.

## Data available (verified against live APIs, 2026-11-02)

| Source | Endpoint | Gives |
|---|---|---|
| Kalshi | `/trade-api/v2/events`, `/markets` | `category`, `series_ticker`, market metadata, `result` |
| Kalshi | `/markets/trades` | anonymous trades: `count_fp`, `yes_price_dollars`, `taker_side`, `created_time` |
| Kalshi | `/markets/{t}/candlesticks`, `/historical/*` | bid/ask history; older data on historical endpoints |
| Polymarket | `gamma-api /markets`, `/events` | market metadata, tags, resolution |
| Polymarket | `data-api /v2/trades`, `/v2/activity` | **per-wallet** trades (`proxy_wallet`, `side`, `price`, `size`, `timestamp`, `condition_id`) |
| Polymarket | `data-api /v2/positions?user=` | per-wallet positions with realized and unrealized PnL |
| Polymarket | `data-api /v2/leaderboard` | ranked wallets by PnL / volume (cursor pagination) |

Note: Polymarket Data API v1 is retired. Use v2 (responses wrapped in `{"data": [...]}`).
