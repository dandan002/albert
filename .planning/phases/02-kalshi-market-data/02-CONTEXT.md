# Phase 2: Kalshi Market Data - Context

**Gathered:** 2026-04-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 2 delivers Kalshi 15-minute Bitcoin market ingestion for offline research: discover eligible markets for explicit date ranges, fetch/cache metadata and market data through current and historical endpoints, normalize raw payloads into analysis-ready tables, and link markets to settlement outcomes. This phase does not add Deribit data, probability modeling, backtesting, paper trading, live trading, or dashboards.

</domain>

<decisions>
## Implementation Decisions

### Market Discovery Scope
- **D-01:** Target only Kalshi 15-minute BTC range/event contracts that map cleanly to settlement windows.
- **D-02:** The CLI should accept explicit `--start-date` and `--end-date` inputs in UTC/date form for requested history windows.
- **D-03:** Ambiguous BTC mappings or interval mappings should be excluded from eligible outputs and recorded as skipped, not included speculatively.
- **D-04:** Discovery should produce a normalized market table and cache the raw source API pages.

### Endpoint Stitching
- **D-05:** Select current versus historical Kalshi endpoints automatically based on requested date/window and API availability.
- **D-06:** If a requested date range spans current and historical availability, split the request into endpoint-specific windows and stitch the normalized outputs.
- **D-07:** Cache every source request/page separately, with endpoint and request parameters represented in the run manifest.
- **D-08:** Use conservative HTTP behavior: bounded retries, explicit timeouts, and clear failures rather than indefinite retries.

### Price And Liquidity Data
- **D-09:** For each discovered market, fetch trades, bid/ask pricing data or candlesticks, and order-book snapshots where available.
- **D-10:** Represent thin or missing liquidity with explicit null/missing fields and data-quality flags; do not impute or forward-fill in Phase 2.
- **D-11:** Preserve normalized pricing observations at source-native granularity with timestamps and endpoint provenance intact.
- **D-12:** Later conservative backtests need bid, ask, trade price, trade size, timestamp, market ticker, and liquidity/depth fields where Kalshi provides them.

### Normalization And Data Quality
- **D-13:** Create separate normalized tables for markets, market price observations, order-book observations, trades, and outcomes.
- **D-14:** If settlement outcome is unavailable, keep the market row with an `outcome_missing` quality flag; later settled backtest inputs should exclude it.
- **D-15:** Enforce strict required-field validation for normalized schemas while preserving unknown source fields only in cached raw payloads.
- **D-16:** Expose data-quality results through a concise CLI summary plus manifest/output metadata, and cover missing outcomes/history behavior with tests.

### the agent's Discretion
- Exact module boundaries, class names, and helper function names are open, but should follow the existing `src/albert` package style.
- Exact retry counts, timeout values, and quality-summary wording are open to planning and implementation.
- Exact normalized column names may be chosen by the planner/implementer as long as the fields and provenance requirements above are satisfied.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project scope and requirements
- `.planning/ROADMAP.md` - Phase 2 goal, success criteria, dependencies, and planned plan split.
- `.planning/REQUIREMENTS.md` - DATA-01 and DATA-02 requirements for Kalshi discovery and retrieval.
- `.planning/PROJECT.md` - v1 constraints: offline CLI, Kalshi plus Deribit only, conservative backtest orientation, and provenance requirements.
- `.planning/STATE.md` - Current phase position and Phase 1 decisions that constrain Phase 2.

### Prior implementation contracts
- `.planning/phases/01-research-cli-foundation/01-01-SUMMARY.md` - CLI, config, package, and placeholder command group contracts.
- `.planning/phases/01-research-cli-foundation/01-02-SUMMARY.md` - Cache path and run manifest provenance contracts.
- `.planning/phases/01-research-cli-foundation/01-03-SUMMARY.md` - Test harness, fixtures, and local verification command contracts.

### Research context
- `.planning/research/STACK.md` - Recommended stack and Kalshi API surface references, including market discovery, historical data, candlesticks, trades, and order books.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/albert/commands/ingest.py`: Existing Typer ingest command group is the natural place to add Kalshi discovery/fetch commands.
- `src/albert/config.py`: `AlbertConfig`, `default_config()`, and `load_config()` provide cache/run/config roots and `ALBERT_` environment override behavior.
- `src/albert/cache.py`: `raw_cache_path()`, `normalized_cache_path()`, `run_manifest_path()`, `sanitize_segment()`, and `stable_params_hash()` provide deterministic local file locations.
- `src/albert/manifest.py`: `SourceMetadata`, `OutputMetadata`, `RunManifest`, `config_hash_from_mapping()`, and `write_manifest()` provide the provenance model for source requests and outputs.
- `tests/conftest.py` and `tests/fixtures/`: Existing deterministic fixtures should be extended for Kalshi API payloads.

### Established Patterns
- The project uses a `src/` layout, Typer command groups, Pydantic models, and pytest.
- Phase 1 deliberately avoided external API clients; Phase 2 is where a Kalshi client/module should be introduced.
- Cache path segments are strict lowercase safe values; unsafe input fails rather than being normalized ambiguously.
- Raw cache filenames are parameter hashes, while manifests retain original non-secret request parameters.
- Tests are local and deterministic, with fixture payloads instead of live API dependence.

### Integration Points
- New CLI commands should attach under `albert ingest`.
- New data clients and normalizers should write raw JSON and normalized Parquet paths through existing cache helpers.
- Every fetch/normalize run should be represented in a run manifest using the existing manifest models.
- Test coverage should extend existing CLI, cache, manifest, and baseline contracts without introducing live network requirements.

</code_context>

<specifics>
## Specific Ideas

- The requested history interface should be explicit date-range based, not an implicit rolling lookback in Phase 2.
- Missing or thin liquidity is itself important research evidence and should be preserved with flags rather than hidden by imputation.
- Unknown or changed Kalshi payload fields should remain available in raw cache while normalized schemas stay controlled.

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within Phase 2 scope.

</deferred>

---

*Phase: 02-kalshi-market-data*
*Context gathered: 2026-04-28*
