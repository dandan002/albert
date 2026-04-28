# Phase 3: Deribit Derivatives Data - Context

**Gathered:** 2026-04-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 3 delivers Deribit BTC futures and options data ingestion for offline research: discover BTC derivative instruments for requested date ranges, fetch/cache OHLC and mark/index price data through Deribit public endpoints, normalize raw payloads into analysis-ready tables aligned with Kalshi contract windows, and record provenance for reproducibility. This phase does not add probability modeling, backtesting, live trading, or dashboards.

</domain>

<decisions>
## Implementation Decisions

### Instrument Selection
- **D-01:** Fetch near-dated BTC options (all options expiring within a configurable window of days around the requested date range) plus BTC-PERP and the nearest quarterly BTC future for the forward price benchmark.
- **D-02:** Cast a configurable near-dated window (e.g., 1-7 days) around each requested date range and fetch all BTC options whose expiry falls within that window. Phase 4 selects the relevant subset for probability estimation.
- **D-03:** Fetch BTC-PERP (perpetual) and the nearest quarterly BTC futures for mark/index price reference. This gives Phase 4 a spot reference and a forward curve anchor.

### Data Resolution & Types
- **D-04:** Fetch 1-minute OHLC candle data from Deribit's TradingView chart data endpoint. The 1-minute resolution provides precise pre-decision timestamps for Phase 4 look-ahead guards.
- **D-05:** OHLC-only for Phase 3. Trade-level data (mark price, IV per trade) can be added in a future phase if Phase 4 needs more granular observations, but is out of scope.
- **D-06:** Fetch mark price and index price from Deribit's ticker/book summary endpoints in addition to OHLC. These provide the spot reference needed for probability models.

### Historical Data Handling
- **D-07:** Use Deribit's single public API with time-range window parameters for historical data. Deribit does not have separate live/historical endpoints like Kalshi — one API with start/end timestamps handles both current and older data.
- **D-08:** Conservative retries with configurable delays, matching Phase 2's Kalshi client HTTP pattern. No proactive rate-limit tracking in Phase 3.

### CLI Invocation Pattern
- **D-09:** Standalone date-range command (`albert ingest deribit fetch`) with `--start-date`, `--end-date`, and `--near-dated-days` parameters. Optionally accept `--kalshi-run-id` to read a Kalshi discovery manifest and align instrument selection and windows with known Kalshi markets.
- **D-10:** Automatic instrument discovery by date range. Given start/end dates, the Deribit client discovers available BTC options instruments and selects those expiring within the near-dated window. No need for users to specify Deribit instrument names manually.

### Missing Data Handling
- **D-11:** Follow the same pattern as Phase 2: missing optional fields (implied volatility, volume, open interest) become null with quality flags. Instruments with no observations are recorded in a skipped-instruments table. No imputation or forward-fill.

### the agent's Discretion
- Exact module boundaries, class names, and helper function names — should follow the existing `src/albert/kalshi/` pattern.
- Exact retry counts, timeout values, and delay durations.
- Exact normalized column names, as long as required fields and provenance requirements are satisfied.
- Default value for `--near-dated-days`.
- Whether to create a `src/albert/deribit/` package mirroring the `kalshi/` structure (recommended for consistency).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project scope and requirements
- `.planning/ROADMAP.md` — Phase 3 goal, success criteria, dependencies, and planned plan split.
- `.planning/REQUIREMENTS.md` — DATA-03 requirement for Deribit futures/options observations.
- `.planning/PROJECT.md` — v1 constraints: offline CLI, Kalshi plus Deribit only, conservative backtest orientation, provenance requirements.
- `.planning/STATE.md` — Current phase position and Phase 1-2 decisions that constrain Phase 3.

### Prior implementation contracts
- `.planning/phases/02-kalshi-market-data/02-CONTEXT.md` — Decisions on CLI patterns, cache conventions, missing-data handling, endpoint routing, and normalization approach that Phase 3 should follow.
- `.planning/phases/01-research-cli-foundation/01-01-SUMMARY.md` — CLI, config, package, and placeholder command group contracts.
- `.planning/phases/01-research-cli-foundation/01-02-SUMMARY.md` — Cache path and run manifest provenance contracts.
- `.planning/phases/01-research-cli-foundation/01-03-SUMMARY.md` — Test harness, fixtures, and local verification command contracts.

### Research context
- `.planning/research/STACK.md` — Recommended stack and Deribit API surface references (OHLC chart data, instrument trades, public market data endpoints).
- `.planning/research/PITFALLS.md` — Look-ahead bias and source price mismatch pitfalls relevant to Deribit-Kalshi alignment.

### Deribit API documentation
- `https://docs.deribit.com/api-reference/market-data/public-get_tradingview_chart_data` — OHLC candle data endpoint.
- `https://docs.deribit.com/api-reference/market-data/public-get_last_trades_by_instrument_and_time` — Trade-level data (deferred, but useful reference).
- `https://docs.deribit.com/api-reference/market-data/public-ticker` — Ticker endpoint for mark/index price.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/albert/commands/ingest.py`: Existing Typer ingest command group with `kalshi` subcommand. The `deribit` subcommand should be added alongside it.
- `src/albert/config.py`: `AlbertConfig`, `default_config()`, `load_config()` — reuse for Deribit configuration (cache/run roots, API settings).
- `src/albert/cache.py`: `raw_cache_path()`, `normalized_cache_path()`, `run_manifest_path()`, `sanitize_segment()`, `stable_params_hash()` — reuse for all Deribit cache writes.
- `src/albert/manifest.py`: `SourceMetadata`, `OutputMetadata`, `RunManifest`, `write_manifest()` — reuse for Deribit run provenance.
- `src/albert/kalshi/client.py`: `KalshiClient`, `FetchedPage` — architectural reference for building `DeribitClient`, but Deribit has no live/historical routing, so the client can be simpler.
- `src/albert/kalshi/pipeline.py`: Pipeline orchestration pattern (fetch → normalize → write → manifest) — mirror for Deribit.
- `src/albert/kalshi/schemas.py`: Pydantic row models with `ProvenanceRow` base, `StrictRow`, `TABLE_*` constants — mirror for Deribit schemas.
- `tests/conftest.py` and `tests/fixtures/`: Fixture pattern for API payloads — extend for Deribit.

### Established Patterns
- Source module structure: `src/albert/{source}/` with `client.py`, `routing.py` (if needed), `schemas.py`, `normalize.py`, `pipeline.py`.
- Cache paths: `raw/{source}/{endpoint}/{hash}.json` and `normalized/{source}/{table}/{hash}.parquet`.
- Manifests: `RunManifest` records all source metadata (endpoint, params, fetch times, raw paths) and output metadata (table name, path, row count).
- CLI: Typer app with `--start-date`, `--end-date`, `--run-id`, `--cache-dir`, `--run-dir`, `--config` pattern.
- Missing data: null/missing fields with quality flags, skipped items in separate table, no imputation.
- Tests: deterministic fixtures, no live network, pytest with explicit fixture injection.

### Integration Points
- New CLI commands attach under `albert ingest deribit` (next to `albert ingest kalshi`).
- Deribit normalized Parquet files go through existing `normalized_cache_path()` with source=`"deribit"`.
- Deribit run manifests use existing `RunManifest` models via `write_manifest()`.
- Test coverage extends existing patterns with Deribit fixture payloads.

</code_context>

<specifics>
## Specific Ideas

- Deribit instrument naming convention: BTC options use format like `BTC-28APR26-85000-C` (expiry-strike-type), futures like `BTC28APR26`, perp is `BTC-PERP`. The client should discover available instruments for the requested date range automatically.
- The near-dated window should be configurable via `--near-dated-days` with a sensible default (planner decides exact default).
- OHLC candles are fetched per instrument, so the pipeline needs to enumerate instruments first, then fetch candles for each within the time range.
- Deribit's public API does not require authentication for public market data endpoints, simplifying the client compared to Kalshi.
- The quality summary for Deribit should parallel Kalshi's: counts of fetched instruments, skipped instruments, missing data flags.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within Phase 3 scope.

</deferred>

---

*Phase: 03-deribit-derivatives-data*
*Context gathered: 2026-04-28*