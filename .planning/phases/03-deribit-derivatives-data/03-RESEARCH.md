# Phase 3: Deribit Derivatives Data - Research

**Researched:** 2026-04-28
**Status:** Complete

## Deribit API Surface

### Public Endpoints (No Authentication Required)

All Deribit public endpoints use JSON-RPC 2.0 over HTTPS GET to `https://www.deribit.com/api/v2/public/{method}`.

1. **`public/get_instruments`** — Discover BTC futures and options instruments
   - Params: `currency=BTC`, `kind=future|option`, `expired=true|false`
   - Returns: Array of instrument objects with `instrument_name`, `kind`, `expiration_timestamp`, `strike` (options), `option_type` (options), `is_active`, `contract_size`, `tick_size`, `settlement_period`, `base_currency`, `quote_currency`, `state`, `creation_timestamp`
   - Rate limit: 1 req/s sustained
   - Key: For instrument discovery — filter by `currency=BTC`, `kind=option` and `kind=future`, then select those expiring within the near-dated window

2. **`public/get_tradingview_chart_data`** — OHLC candle data (1-minute resolution per D-04)
   - Params: `instrument_name`, `start_timestamp` (ms epoch), `end_timestamp` (ms epoch), `resolution=1` (1-minute)
   - Returns: `ticks[]` (ms epoch), `open[]`, `high[]`, `low[]`, `close[]`, `volume[]`, `cost[]`, `status` ("ok" or "no_data")
   - Resolution values: 1, 3, 5, 10, 15, 30, 60, 120, 180, 360, 720, 1D
   - Supports historical data with start/end timestamps — no separate live/historical routing needed (per D-07)

3. **`public/ticker`** — Current mark price, index price, IV, greeks for a single instrument
   - Params: `instrument_name`
   - Returns: `mark_price`, `index_price`, `best_bid_price`, `best_ask_price`, `open_interest`, `last_price`, `underlying_price`, `mark_iv` (options), `bid_iv` (options), `ask_iv` (options), `greeks` (options: delta, gamma, rho, theta, vega), `stats.volume`, `timestamp`, `state`, `estimated_delivery_price`
   - Note: Ticker returns current snapshot only — for historical mark/index prices, use OHLC on perpetual futures as a proxy

4. **`public/get_book_summary_by_currency`** — Summary for all instruments in a currency
   - Params: `currency=BTC`, `kind=future|option`
   - Returns: Array of `book_summary` objects with `instrument_name`, `mark_price`, `mid_price`, `bid_price`, `ask_price`, `volume`, `open_interest`, `mark_iv` (options), `underlying_price` (options), `underlying_index` (options), `interest_rate` (options), `creation_timestamp`
   - Use case: Batch snapshot of mark prices and IV across all BTC instruments — complements OHLC for current state

### Key API Differences from Kalshi

| Aspect | Kalshi | Deribit |
|--------|--------|---------|
| Authentication | Required for some endpoints | Not required for public data endpoints |
| Live vs Historical | Separate endpoint families with cutoff | Single API with time-range parameters (D-07) |
| Pagination | Cursor-based for trades | No pagination on chart data; instruments may need offset |
| Response format | REST JSON | JSON-RPC 2.0 wrapper (`jsonrpc`, `id`, `result`) |
| Timestamps | ISO strings | Milliseconds since Unix epoch |
| Rate limiting | Unspecified | 1 req/s for get_instruments; 20 req/s for most others |

### Instrument Naming Convention

- **Perpetual:** `BTC-PERPETUAL`
- **Quarterly futures:** `BTC-28MAR26` (expiry date)
- **Options:** `BTC-28MAR26-100000-C` (expiry-strike-type, C=call, P=put)

## Normalization Strategy

### Tables to Create

Per CONTEXT.md D-11 and following Kalshi patterns:

1. **`deribit_instruments`** — Instrument metadata
   - `instrument_name`, `kind` (future/option), `base_currency`, `quote_currency`, `settlement_period`, `expiration_timestamp`, `strike` (null for futures), `option_type` (null for futures), `contract_size`, `tick_size`, `is_active`, `state`, `creation_timestamp`, `source_endpoint`, `source_raw_path`

2. **`deribit_ohlc_observations`** — OHLC candle data
   - `instrument_name`, `observed_at_utc`, `resolution_minutes`, `open`, `high`, `low`, `close`, `volume`, `cost`, `source_endpoint`, `source_raw_path`

3. **`deribit_ticker_observations`** — Mark/index/IV snapshot data
   - `instrument_name`, `observed_at_utc`, `mark_price`, `index_price`, `best_bid_price`, `best_ask_price`, `open_interest`, `last_price`, `mark_iv` (null for futures), `underlying_price` (null for futures), `delta` (null for futures), `gamma` (null for futures), `theta` (null for futures), `vega` (null for futures), `rho` (null for futures), `source_endpoint`, `source_raw_path`

4. **`deribit_skipped_instruments`** — Instruments with no observations
   - `instrument_name`, `kind`, `expiration_timestamp`, `skip_reason`, `source_endpoint`, `source_raw_path`

5. **`deribit_quality_summary`** — Run quality metrics
   - `instruments_fetched`, `instruments_skipped`, `ohlc_observations_missing`, `ticker_observations_missing`, `source_endpoint`, `source_raw_path`

### Missing Data Handling (per D-11)

- Optional fields (implied volatility, greeks, volume, open interest) become null with quality flags
- Instruments with no OHLC or ticker data → recorded in skipped_instruments table with skip reason
- No imputation or forward-fill

## Architecture Following Kalshi Patterns

### Module Structure

```
src/albert/deribit/
    __init__.py
    client.py      # DeribitClient — mirrors KalshiClient pattern
    schemas.py     # Pydantic row models + TABLE_* constants — mirrors kalshi/schemas.py
    normalize.py   # Raw-to-normalized transforms — mirrors kalshi/normalize.py
    pipeline.py    # Orchestrate discover → fetch → normalize — mirrors kalshi/pipeline.py
    quality.py     # Quality assessment and summary — mirrors kalshi/quality.py
```

### Key Design Decisions

1. **No live/historical routing** — Deribit has a single API with time-range params (D-07). The client is simpler than Kalshi's.
2. **Instrument discovery via get_instruments** — Fetch both `kind=option` and `kind=future` for BTC, filter by expiration window. Requires two API calls.
3. **OHLC as primary observation** — 1-minute candles (D-04) provide timestamped prices for look-ahead guards.
4. **Ticker supplement for mark/IV** — Current snapshots for mark price, index price, and implied volatility (D-06).
5. **Book summary for batch snapshots** — Optional, provides mark/IV across all instruments in one call. Useful for discovery-time snapshots.

## Common Pitfalls

### Deribit-Specific

1. **JSON-RPC response wrapper** — All responses are wrapped in `{"jsonrpc": "2.0", "id": ..., "result": ...}`. Must unwrap `result` before processing.
2. **Millisecond timestamps** — Deribit uses ms since epoch throughout. Must convert to UTC datetime for normalization.
3. **Option instrument naming** — Parse the instrument name to extract expiry, strike, and option type. Naming follows pattern `BTC-{DDMMMYY}-{strike}-{C|P}`.
4. **Perpetual has no expiry** — `expiration_timestamp` for BTC-PERPETUAL is far future (year 2999). Must handle specially.
5. **Null/missing optional fields** — Options-specific fields (mark_iv, greeks, underlying_price) are absent on futures. Must normalize to null.
6. **Rate limiting** — get_instruments is 1 req/s. Other endpoints are 20 req/s. Must implement conservative retry (D-08).
7. **Resolution values** — Only specific resolution values accepted (1, 3, 5, 10, 15, 30, 60, 120, 180, 360, 720, 1D). Use "1" for 1-minute candles (D-04).
8. **Status "no_data"** — OHLC responses can return `status: "no_data"` with empty arrays. Must handle gracefully.

### Cross-Phase

1. **Look-ahead bias** (PITFALLS.md) — OHLC timestamps must carry explicit timestamps. Phase 4 will enforce `observation_timestamp <= decision_timestamp`.
2. **Source price mismatch** (PITFALLS.md) — Deribit BTC index vs Kalshi settlement reference may differ. Store the Deribit index_price explicitly for later comparison.
3. **No separate live/historical** — Unlike Kalshi, Deribit doesn't route to different endpoints based on age. Client is structurally simpler.

## Validation Architecture

### Dimension Coverage

| Dimension | Approach |
|-----------|----------|
| 1. Completeness | Every normalized table has defined schema with required fields; Pydantic StrictRow validates |
| 2. Correctness | Fixture payloads for futures, options, missing fields, no-data responses; schemas validated against fixtures |
| 3. Data Integrity | No forward-fill or imputation; null optional fields; all timestamps UTC-aware |
| 4. Reproducibility | Raw cache with stable parameter hashes; manifests record all source metadata |
| 5. Look-Ahead Prevention | OHLC observations carry timestamp; pipeline validates time ranges don't exceed request window |
| 6. Edge Cases | no_data responses, perpetual with no real expiry, null IV/greeks on futures, empty instrument lists |
| 7. Provenance | Source endpoint, raw path, request parameters, and fetch timestamps in every normalized row |
| 8. Negative Testing | Skipped instruments with reason; quality summary counts missing observations |

## Don't Hand-Roll

- HTTP client with retry/timeout: Use httpx (already in dependencies from Phase 2)
- JSON-RPC parsing: Parse `result` from response wrapper — simple dict extraction, no library needed
- Pydantic models: Use existing ProvenanceRow and StrictRow patterns from kalshi/schemas.py
- Polars schemas: Use TABLE_SCHEMAS dict pattern from kalshi/pipeline.py
- Cache paths: Reuse raw_cache_path() and normalized_cache_path() from cache.py
- Manifests: Reuse RunManifest, SourceMetadata, OutputMetadata from manifest.py

## Stack

- httpx (already in pyproject.toml from Phase 2)
- polars + pyarrow (already in pyproject.toml from Phase 2)
- pydantic (already in pyproject.toml from Phase 1)
- No new dependencies needed