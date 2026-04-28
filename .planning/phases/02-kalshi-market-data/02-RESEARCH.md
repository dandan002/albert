# Phase 02: Kalshi Market Data - Research

**Researched:** 2026-04-28 [VERIFIED: system date]
**Domain:** Kalshi REST ingestion, live/historical endpoint routing, local raw/Parquet caching, fixture-driven normalization [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md]
**Confidence:** HIGH for Kalshi endpoint routing and project contracts; MEDIUM for exact BTC 15-minute series discovery heuristics because the implementation should validate current series metadata through API fixtures and live user runs. [CITED: https://docs.kalshi.com/getting_started/historical_data] [VERIFIED: codebase grep]

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

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

### Claude's Discretion
- Exact module boundaries, class names, and helper function names are open, but should follow the existing `src/albert` package style.
- Exact retry counts, timeout values, and quality-summary wording are open to planning and implementation.
- Exact normalized column names may be chosen by the planner/implementer as long as the fields and provenance requirements above are satisfied.

### Deferred Ideas (OUT OF SCOPE)

None - discussion stayed within Phase 2 scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | User can run a CLI command that discovers Kalshi 15-minute Bitcoin markets over a requested date range. [VERIFIED: .planning/REQUIREMENTS.md] | Use `albert ingest kalshi discover --start-date --end-date`, route `/markets` versus `/historical/markets`, cache each page, normalize eligible/skipped market tables, and test cursor/date filters with fixtures. [CITED: https://docs.kalshi.com/api-reference/market/get-markets] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-markets] |
| DATA-02 | User can retrieve Kalshi market metadata, order-book snapshots or bid/ask candles, trades, and settlement outcomes for discovered markets. [VERIFIED: .planning/REQUIREMENTS.md] | Use client methods for markets, candlesticks, trades, and current orderbooks; expose historical routing for markets/candlesticks/trades; represent unavailable historical orderbooks and missing outcomes through explicit quality flags. [CITED: https://docs.kalshi.com/api-reference/market/get-market-candlesticks] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks] [CITED: https://docs.kalshi.com/api-reference/market/get-trades] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-trades] [CITED: https://docs.kalshi.com/api-reference/market/get-multiple-market-orderbooks] |
</phase_requirements>

## Summary

Phase 2 should be planned as a data-ingestion slice with three boundaries: `kalshi/client.py` for HTTP, pagination, timeouts, retries, and raw cache writes; `kalshi/routing.py` for live-versus-historical endpoint selection from `/historical/cutoff`; and `kalshi/normalize.py` for strict fixture-driven normalized tables. [VERIFIED: src/albert/cache.py] [VERIFIED: src/albert/manifest.py] [CITED: https://docs.kalshi.com/getting_started/historical_data]

Kalshi partitions older markets, candlesticks, trades, and orders into historical endpoints, and the live window target is about three months, so a 12-month v1 backtest cannot rely on current endpoints only. [CITED: https://docs.kalshi.com/getting_started/historical_data] The planner should create explicit tasks for cutoff retrieval, request-window splitting, raw page caching, and normalized output stitching, because these are central behavioral requirements rather than implementation details. [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md]

**Primary recommendation:** implement a fixture-first Kalshi ingestion package under `src/albert/kalshi/`, attach commands under `albert ingest kalshi`, and make every API page and normalized table appear in a `RunManifest` before adding any live-network assumptions to tests. [VERIFIED: src/albert/commands/ingest.py] [VERIFIED: src/albert/manifest.py]

## Project Constraints (from AGENTS.md)

- Albert is a Python research CLI for offline evidence generation before any live or paper trading work. [VERIFIED: AGENTS.md]
- v1 scope is Kalshi plus Deribit only, with Polymarket deferred. [VERIFIED: AGENTS.md]
- Backtests must use conservative bid/ask fills, fees, and liquidity constraints; Phase 2 must preserve bid/ask, trade size, and liquidity fields for later use. [VERIFIED: AGENTS.md] [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md]
- Every run must be traceable to source endpoints, timestamps, and cached raw inputs. [VERIFIED: AGENTS.md]
- Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it. [VERIFIED: AGENTS.md]
- No project-local `CLAUDE.md` file or project-local `.claude/skills` / `.agents/skills` skill indexes were found. [VERIFIED: shell `test -f CLAUDE.md`; shell `find .claude/skills .agents/skills -maxdepth 2 -name SKILL.md`]

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.12.2 available; project requires `>=3.12` | CLI runtime and data pipeline language | Existing package and tests target Python 3.12. [VERIFIED: `python3 --version`] [VERIFIED: pyproject.toml] |
| Typer | 0.25.0 latest and installed | CLI command groups | Existing `albert` console script and command groups use Typer. [VERIFIED: `python3 -m pip index versions typer`] [VERIFIED: src/albert/cli.py] |
| httpx | 0.28.1 latest and installed | Synchronous Kalshi REST client with explicit timeout/retry wrapper | HTTPX supports sync/async APIs and strict timeouts, and Phase 2 needs HTTP without adding a Kalshi SDK. [VERIFIED: `python3 -m pip index versions httpx`] [CITED: https://www.python-httpx.org/] |
| Pydantic | 2.13.3 latest and installed | Strict API payload and normalized-row validation | Existing config and manifest models already use Pydantic v2 with `extra="forbid"`. [VERIFIED: `python3 -m pip index versions pydantic`] [VERIFIED: src/albert/manifest.py] |
| polars | 1.40.1 latest, not installed in current environment | Build normalized DataFrames and write Parquet tables | Stack research selected Polars plus PyArrow for local analytics and Parquet output. [VERIFIED: `python3 -m pip index versions polars`] [VERIFIED: `.planning/research/STACK.md`] |
| pyarrow | 24.0.0 latest; 15.0.0 installed | Parquet backend/interoperability | Phase 2 writes normalized Parquet outputs through cache helpers. [VERIFIED: `python3 -m pip index versions pyarrow`] [VERIFIED: `.planning/research/STACK.md`] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | 9.0.3 latest and installed | Fixture-based unit and CLI tests | Required for deterministic normalization, routing, and CLI behavior tests without live API dependence. [VERIFIED: `python3 -m pip index versions pytest`] [VERIFIED: Makefile] |
| Ruff | 0.15.12 latest and installed | Static linting | Existing `make lint` contract uses Ruff. [VERIFIED: `python3 -m pip index versions ruff`] [VERIFIED: Makefile] |
| `decimal.Decimal` | Python standard library | Exact fixed-point price/count parsing | Kalshi fixed-point prices and counts are strings, and Decimal avoids float rounding in normalized bid/ask/trade fields. [CITED: https://docs.kalshi.com/getting_started/fixed_point_migration] [VERIFIED: Python stdlib] |
| `zoneinfo` / UTC-aware `datetime` | Python standard library | Date-range parsing and manifest timestamps | Existing manifest models require timezone-aware datetimes normalized to UTC. [VERIFIED: src/albert/manifest.py] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| httpx | `requests` | `requests` is adequate for sync HTTP, but HTTPX is already in the recommended stack and gives strict timeouts and a typed modern API. [VERIFIED: `.planning/research/STACK.md`] [CITED: https://www.python-httpx.org/] |
| polars/pyarrow | pandas-only CSV/Parquet | pandas is useful later for modeling interop, but Polars plus Parquet better matches the existing storage recommendation and local analytics goal. [VERIFIED: `.planning/research/STACK.md`] |
| Direct CLI HTTP calls | Dedicated client/router modules | Direct command code would duplicate pagination, retry, cutoff, and cache/manifest logic across Phase 2 plans. [VERIFIED: src/albert/commands/ingest.py] [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md] |
| Kalshi SDK | Thin first-party REST wrapper in `src/albert/kalshi/client.py` | Phase 2 needs public data retrieval, provenance, fixtures, and raw-cache control; an SDK would still need cache/manifest wrapping. [VERIFIED: .planning/PROJECT.md] [ASSUMED] |

**Installation:**

```bash
python -m pip install "httpx>=0.28.1" "polars>=1.40.1" "pyarrow>=24.0.0"
```

**Version verification:** Versions were checked with `python3 -m pip index versions ...` on 2026-04-28; `pip index` verified latest versions but did not return publish dates. [VERIFIED: shell output]

## Architecture Patterns

### Recommended Project Structure

```text
src/albert/
├── commands/
│   └── ingest.py          # Register `kalshi` subcommands under the existing ingest group.
├── kalshi/
│   ├── __init__.py        # Package exports only stable public helper types.
│   ├── client.py          # HTTPX client, pagination, timeout/retry, raw JSON cache writes.
│   ├── routing.py         # Historical cutoff retrieval and live/historical request plans.
│   ├── schemas.py         # Pydantic source payload fragments and normalized row models.
│   ├── normalize.py       # Raw payload -> Polars DataFrame/row lists for each table.
│   └── quality.py         # Missing outcomes, missing price history, ambiguity, liquidity flags.
tests/
├── fixtures/kalshi/
│   ├── live_markets_page.json
│   ├── historical_markets_page.json
│   ├── live_candlesticks.json
│   ├── historical_candlesticks.json
│   ├── live_trades_page.json
│   ├── historical_trades_page.json
│   └── orderbooks.json
└── test_kalshi_*.py
```

This structure follows the existing `src/albert` package, keeps command functions thin, and isolates source-specific API complexity under `src/albert/kalshi`. [VERIFIED: repo file tree] [VERIFIED: src/albert/commands/ingest.py]

### Pattern 1: Thin HTTP Client With Page-Level Cache Writes

**What:** Each client method builds canonical endpoint parameters, sends one HTTP request, validates response status, writes the raw JSON body to `raw_cache_path(cache_root, "kalshi", endpoint_segment, params)`, and returns a `FetchedPage` containing endpoint, params, covered range, raw path, and parsed JSON. [VERIFIED: src/albert/cache.py] [VERIFIED: src/albert/manifest.py]

**When to use:** Use this for markets, historical markets, trades, historical trades, candlesticks, historical candlesticks, cutoff, and current orderbooks. [CITED: https://docs.kalshi.com/getting_started/historical_data] [CITED: https://docs.kalshi.com/api-reference/market/get-multiple-market-orderbooks]

**Example:**

```python
# Source: https://docs.kalshi.com/getting_started/pagination
# Source: src/albert/cache.py and src/albert/manifest.py
class KalshiClient:
    def __init__(self, base_url: str, http: httpx.Client, cache_root: Path) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = http
        self._cache_root = cache_root

    def get_json_page(
        self,
        endpoint_segment: str,
        path: str,
        params: dict[str, object],
    ) -> FetchedPage:
        response = self._http.get(f"{self._base_url}{path}", params=params)
        response.raise_for_status()
        raw = response.json()
        raw_path = raw_cache_path(self._cache_root, "kalshi", endpoint_segment, params)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8")
        return FetchedPage(endpoint=path, parameters=params, raw_path=raw_path, payload=raw)
```

### Pattern 2: Cursor Pagination As A Shared Iterator

**What:** Implement one `iter_pages(path, params, collection_key)` helper that keeps original filters stable, adds `cursor` only after the first page, stops on null or empty cursor, and yields each raw page for manifest source records. [CITED: https://docs.kalshi.com/getting_started/pagination]

**When to use:** Use the iterator for `/markets`, `/historical/markets`, `/markets/trades`, and `/historical/trades`; candlesticks and orderbooks are bounded by ticker/window requests and should not pretend to be cursor-paginated. [CITED: https://docs.kalshi.com/api-reference/market/get-markets] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-markets] [CITED: https://docs.kalshi.com/api-reference/market/get-trades] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-trades]

**Example:**

```python
# Source: https://docs.kalshi.com/getting_started/pagination
def iter_cursor_pages(client: KalshiClient, path: str, params: dict[str, object]) -> Iterator[FetchedPage]:
    cursor: str | None = None
    while True:
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor
        page = client.get_json_page(endpoint_segment=endpoint_segment(path), path=path, params=page_params)
        yield page
        cursor = page.payload.get("cursor")
        if not cursor:
            break
```

### Pattern 3: Cutoff-Aware Route Planning

**What:** Fetch `/historical/cutoff`, compare requested windows to `market_settled_ts` for markets/candlesticks and `trades_created_ts` for public trades, then emit one or two `EndpointWindow` records per requested range. [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-cutoff-timestamps] [CITED: https://docs.kalshi.com/getting_started/historical_data]

**When to use:** Use route planning before market discovery, candlestick retrieval, and trade retrieval; do not bury cutoff checks inside normalizers. [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md]

**Example:**

```python
# Source: https://docs.kalshi.com/api-reference/historical/get-historical-cutoff-timestamps
def route_by_cutoff(start: datetime, end: datetime, cutoff: datetime) -> list[EndpointWindow]:
    if end < cutoff:
        return [EndpointWindow(kind="historical", start=start, end=end)]
    if start >= cutoff:
        return [EndpointWindow(kind="live", start=start, end=end)]
    return [
        EndpointWindow(kind="historical", start=start, end=cutoff),
        EndpointWindow(kind="live", start=cutoff, end=end),
    ]
```

### Pattern 4: Strict Normalized Tables Plus Raw Payload Preservation

**What:** Validate only columns needed by downstream phases and preserve unknown Kalshi fields only in raw JSON cache files. [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md] [VERIFIED: src/albert/cache.py]

**When to use:** Use separate normalizers for `markets`, `price_observations`, `trades`, `orderbook_observations`, `outcomes`, `skipped_markets`, and `quality_summary`. [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md]

**Recommended normalized table shapes:**

| Table | Required Columns |
|-------|------------------|
| `kalshi_markets` | `market_ticker`, `event_ticker`, `series_ticker`, `title`, `subtitle`, `status`, `open_time_utc`, `close_time_utc`, `settlement_time_utc`, `expiration_time_utc`, `can_close_early`, `price_level_structure`, `fractional_trading_enabled`, `source_endpoint`, `source_raw_path` [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md] [CITED: https://docs.kalshi.com/getting_started/fixed_point_migration] |
| `kalshi_price_observations` | `market_ticker`, `series_ticker`, `observed_at_utc`, `period_interval_minutes`, `yes_bid_open`, `yes_bid_high`, `yes_bid_low`, `yes_bid_close`, `yes_ask_open`, `yes_ask_high`, `yes_ask_low`, `yes_ask_close`, `last_price_open`, `last_price_close`, `volume_contracts`, `open_interest_contracts`, `source_endpoint`, `source_raw_path` [CITED: https://docs.kalshi.com/api-reference/market/get-market-candlesticks] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks] |
| `kalshi_trades` | `trade_id`, `market_ticker`, `created_time_utc`, `count_contracts`, `yes_price`, `no_price`, `taker_side`, `source_endpoint`, `source_raw_path` [CITED: https://docs.kalshi.com/api-reference/market/get-trades] |
| `kalshi_orderbook_observations` | `market_ticker`, `snapshot_time_utc`, `side`, `level_index`, `bid_price`, `bid_count_contracts`, `implied_ask_price`, `source_endpoint`, `source_raw_path` [CITED: https://docs.kalshi.com/getting_started/orderbook_responses] |
| `kalshi_outcomes` | `market_ticker`, `settlement_time_utc`, `outcome`, `outcome_missing`, `source_endpoint`, `source_raw_path` [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md] |
| `kalshi_skipped_markets` | `market_ticker`, `event_ticker`, `series_ticker`, `skip_reason`, `raw_title`, `raw_subtitle`, `source_endpoint`, `source_raw_path` [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md] |
| `kalshi_quality_summary` | `run_id`, `market_count`, `eligible_market_count`, `skipped_market_count`, `missing_outcome_count`, `missing_price_history_count`, `missing_orderbook_count`, `historical_window_count`, `live_window_count` [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md] |

### Anti-Patterns to Avoid

- **Hard-coding only live endpoints:** older market/candlestick/trade data must be routed to historical endpoints based on cutoff timestamps. [CITED: https://docs.kalshi.com/getting_started/historical_data]
- **Flattening live and historical candlesticks with stringly field access:** live examples use `*_dollars` price keys while historical examples use unsuffixed keys, so fixtures must cover both shapes. [CITED: https://docs.kalshi.com/api-reference/market/get-market-candlesticks] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks]
- **Treating orderbook arrays as asks:** Kalshi orderbook responses return YES and NO bids, and asks must be derived by complement when needed. [CITED: https://docs.kalshi.com/getting_started/orderbook_responses]
- **Converting fixed-point strings through `float`:** fixed-point strings can carry subpenny prices and fractional contracts, so use `Decimal` or integer scaled units. [CITED: https://docs.kalshi.com/getting_started/fixed_point_migration]
- **Dropping markets with missing outcomes silently:** Phase 2 requires kept rows with `outcome_missing` flags. [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP client transport | Custom socket/urllib retry stack | `httpx.Client` with explicit `Timeout` and bounded retry loop | HTTPX already provides robust request handling and strict timeouts. [CITED: https://www.python-httpx.org/] |
| Cursor pagination | Per-endpoint while loops copied into commands | One shared `iter_cursor_pages` helper | Kalshi list endpoints use the same cursor pattern and page-level manifests need consistent source records. [CITED: https://docs.kalshi.com/getting_started/pagination] |
| Historical routing | Ad hoc if-statements inside commands | `EndpointRouter` / `EndpointWindow` data model | Cutoff semantics differ by data family, and spanning windows need deterministic splitting. [CITED: https://docs.kalshi.com/getting_started/historical_data] |
| Parquet writing | Manual binary writer or CSV fallback | Polars DataFrame `write_parquet` backed by PyArrow-compatible Parquet ecosystem | Project storage standard is local Parquet plus manifests. [VERIFIED: .planning/research/STACK.md] |
| Money/count math | `float` conversions | `Decimal` or scaled integers | Kalshi fixed-point dollar and contract-count strings are designed to preserve subpenny/fractional precision. [CITED: https://docs.kalshi.com/getting_started/fixed_point_migration] |
| Provenance tracking | New manifest format | Existing `SourceMetadata`, `OutputMetadata`, and `RunManifest` | DATA-04 is complete and Phase 2 should reuse those models. [VERIFIED: src/albert/manifest.py] [VERIFIED: .planning/REQUIREMENTS.md] |

**Key insight:** the deceptive complexity is not HTTP itself; it is preserving endpoint provenance while stitching live/historical data without hiding missing liquidity or missing outcomes. [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md] [CITED: https://docs.kalshi.com/getting_started/historical_data]

## Common Pitfalls

### Pitfall 1: Wrong Cutoff Field For Data Family

**What goes wrong:** markets and candlesticks get routed by trade timestamp or trades get routed by settlement timestamp. [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-cutoff-timestamps]

**Why it happens:** `/historical/cutoff` returns multiple cutoff fields with different meanings. [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-cutoff-timestamps]

**How to avoid:** encode the cutoff field in the route-plan function per data family: markets/candlesticks use `market_settled_ts`; public trades should use the public historical trade docs plus `trades_created_ts` as the split timestamp. [CITED: https://docs.kalshi.com/getting_started/historical_data] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-trades]

**Warning signs:** a 12-month run returns only about the live-window worth of settled markets or trades. [CITED: https://docs.kalshi.com/getting_started/historical_data]

### Pitfall 2: Historical Trades Documentation Inconsistency

**What goes wrong:** implementation routes old public trades to `/historical/fills` instead of `/historical/trades`, or tests assume only one doc wording exists. [CITED: https://docs.kalshi.com/getting_started/historical_data] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-cutoff-timestamps]

**Why it happens:** the historical overview says public trades before cutoff are available from `/historical/trades`, while the cutoff endpoint text describes `trades_created_ts` as fills in one place. [CITED: https://docs.kalshi.com/getting_started/historical_data] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-cutoff-timestamps]

**How to avoid:** implement a route abstraction where the public trade endpoint is configured as `/historical/trades`, cover it with fixtures, and leave a test name documenting the docs inconsistency. [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-trades]

**Warning signs:** old trade retrieval returns authentication/user-fill-shaped errors instead of public trade rows. [ASSUMED]

### Pitfall 3: Orderbooks Are Current Snapshots, Not Historical History

**What goes wrong:** backtest inputs expect historical orderbook snapshots for all prior markets from a first-party endpoint. [CITED: https://docs.kalshi.com/getting_started/historical_data]

**Why it happens:** first-party historical endpoints listed in docs cover cutoff, markets, market candlesticks, trades, fills, and orders; I did not find a historical orderbook-snapshot endpoint in the official docs checked. [CITED: https://docs.kalshi.com/getting_started/historical_data] [CITED: https://docs.kalshi.com/api-reference/market/get-multiple-market-orderbooks]

**How to avoid:** fetch current orderbooks where available, normalize them separately with snapshot timestamps, and rely on candlestick bid/ask history for historical pricing unless later evidence identifies a historical orderbook source. [CITED: https://docs.kalshi.com/api-reference/market/get-market-candlesticks] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks]

**Warning signs:** a Phase 2 task tries to synthesize historical depth from candlestick OHLC fields. [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md]

### Pitfall 4: Ambiguous BTC Market Mapping

**What goes wrong:** discovery includes non-15-minute BTC markets or crypto range markets that do not map cleanly to the intended settlement window. [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md]

**Why it happens:** Kalshi has multiple crypto market families, and help docs describe cryptocurrency price markets broadly rather than a single event shape. [CITED: https://help.kalshi.com/en/articles/13823838-crypto-markets]

**How to avoid:** seed discovery with likely series filters such as `KXBTC15M` when provided or discovered, then validate each market by series, title/rules shape, event timestamps, and 15-minute close/open interval; skipped markets must include `skip_reason`. [CITED: https://kalshi.com/markets/kxbtc15m/bitcoin-price-up-down/kxbtc15m-25dec242215] [VERIFIED: WebSearch] [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md]

**Warning signs:** normalized markets contain mixed BTC hourly/daily range tickers or rows without clear open/close/settlement windows. [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md]

### Pitfall 5: Fixture Coverage Mirrors Only Happy-Path Live Payloads

**What goes wrong:** historical payloads break normalization after implementation because fields differ from live examples. [CITED: https://docs.kalshi.com/api-reference/market/get-market-candlesticks] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks]

**Why it happens:** live candlestick examples show `open_dollars` style fields while historical examples show `open` style fields. [CITED: https://docs.kalshi.com/api-reference/market/get-market-candlesticks] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks]

**How to avoid:** create paired live/historical fixtures for every normalizer before implementing CLI commands. [VERIFIED: tests/conftest.py] [VERIFIED: .planning/phases/01-research-cli-foundation/01-03-SUMMARY.md]

**Warning signs:** tests only cover one fixture named `candlesticks.json`. [VERIFIED: tests/fixtures]

## Code Examples

Verified patterns from official and local sources:

### Orderbook Bid/Ask Derivation

```python
# Source: https://docs.kalshi.com/getting_started/orderbook_responses
from decimal import Decimal


def best_yes_prices(orderbook_fp: dict[str, list[list[str]]]) -> tuple[Decimal | None, Decimal | None]:
    yes_levels = orderbook_fp.get("yes_dollars") or []
    no_levels = orderbook_fp.get("no_dollars") or []
    best_yes_bid = Decimal(yes_levels[-1][0]) if yes_levels else None
    best_yes_ask = Decimal("1.00") - Decimal(no_levels[-1][0]) if no_levels else None
    return best_yes_bid, best_yes_ask
```

### Live/Historical Candlestick Field Normalization

```python
# Source: https://docs.kalshi.com/api-reference/market/get-market-candlesticks
# Source: https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks
from decimal import Decimal


def decimal_price(node: dict[str, str], name: str) -> Decimal | None:
    value = node.get(f"{name}_dollars", node.get(name))
    return Decimal(value) if value is not None else None
```

### Manifest Source Record Per Raw Page

```python
# Source: src/albert/manifest.py
source = SourceMetadata(
    source="kalshi",
    endpoint="/historical/markets",
    parameters={"series_ticker": "KXBTC15M", "min_close_ts": 1767225600, "limit": 1000},
    fetched_at=fetched_at,
    covered_start=window.start,
    covered_end=window.end,
    raw_path=page.raw_path,
)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Cent-only integer price fields | Fixed-point dollar strings with `_dollars` suffix and `_fp` contract counts | Kalshi fixed-point migration page last updated April 2, 2026 | Normalizers must parse strings and avoid float math. [CITED: https://docs.kalshi.com/getting_started/fixed_point_migration] |
| Querying live endpoints for all historical markets | Route older markets/candlesticks/trades through `/historical/...` endpoints using `/historical/cutoff` | Current docs checked 2026-04-28 | 12-month backtests need endpoint splitting and output stitching. [CITED: https://docs.kalshi.com/getting_started/historical_data] |
| Treating current orderbook as bid/ask arrays | Treat orderbook arrays as YES and NO bids and derive asks by complement | Current docs checked 2026-04-28 | Later conservative fills need derived ask fields with provenance. [CITED: https://docs.kalshi.com/getting_started/orderbook_responses] |

**Deprecated/outdated:**
- Assuming historical Kalshi data lives only in `/markets` and `/markets/trades` is outdated for older settled markets/trades. [CITED: https://docs.kalshi.com/getting_started/historical_data]
- Assuming cent-only integer prices is outdated for markets using fixed-point price structures. [CITED: https://docs.kalshi.com/getting_started/fixed_point_migration]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | A Kalshi SDK would still need cache/manifest wrapping and is not worth adding for this phase. | Standard Stack | If wrong, planner may miss a maintained SDK feature that reduces endpoint/auth burden. |
| A2 | Old trade retrieval errors from `/historical/fills` would look authentication/user-fill-shaped. | Common Pitfalls | If wrong, tests should still catch route mismatch through fixture shape and endpoint assertions. |
| A3 | The research should be treated as valid for 7 days for Kalshi API/package details and 30 days for project-local architecture contracts. | Metadata | If wrong, planner may either over-refresh stable facts or under-refresh fast-moving API/package details. |

## Open Questions (RESOLVED)

1. **RESOLVED: Exact eligible BTC series filter**
   - What we know: public market URLs and third-party indexed pages identify `KXBTC15M` as a Kalshi BTC 15-minute up/down series. [CITED: https://kalshi.com/markets/kxbtc15m/bitcoin-price-up-down/kxbtc15m-25dec242215] [VERIFIED: WebSearch]
   - Resolution: default the CLI `--series-ticker` to `KXBTC15M`, allow user override, and record excluded or ambiguous markets in `kalshi_skipped_markets` with explicit skip reasons. This implements D-01, D-03, and D-16 without assuming the series is immutable. [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md]

2. **RESOLVED: Settlement outcome source field**
   - What we know: Phase 2 must link markets to outcomes and keep rows with missing outcomes. [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md]
   - Resolution: add fixture-backed support for observed settlement outcome fields `result`, `settlement_value_dollars`, and `settlement_ts` before finalizing normalized `kalshi_outcomes`; when none of those fields is available, keep the market row and emit `outcome_missing=True`. This implements D-14 and D-16 while preserving raw unknown fields only in cached payloads. [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md]

3. **RESOLVED: Historical orderbook depth**
   - What we know: official historical endpoints checked here do not list historical orderbook snapshots. [CITED: https://docs.kalshi.com/getting_started/historical_data]
   - Resolution: do not synthesize historical orderbook depth. Normalize current orderbook snapshots when available and set missing-orderbook quality flags for unavailable historical depth. This implements D-09, D-10, and D-12 conservatively for downstream backtests. [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python | CLI and tests | yes | 3.12.2 | none needed. [VERIFIED: `python3 --version`] |
| Typer | Existing CLI | yes | 0.25.0 | none needed. [VERIFIED: Python import] |
| httpx | Kalshi HTTP client | yes | 0.28.1 | none needed. [VERIFIED: Python import] |
| polars | Normalized table construction | no | latest 1.40.1 | Add dependency before normalizer implementation. [VERIFIED: Python import failure] [VERIFIED: `python3 -m pip index versions polars`] |
| pyarrow | Parquet output | yes, outdated | installed 15.0.0; latest 24.0.0 | Upgrade with project dependency change or let Polars Parquet writer cover initial files if compatible. [VERIFIED: Python import] [VERIFIED: `python3 -m pip index versions pyarrow`] |
| pytest | Validation | yes | 9.0.3 | none needed. [VERIFIED: `python -m pytest --version`] |
| Ruff | Lint validation | yes | 0.15.12 | none needed. [VERIFIED: `python -m ruff --version`] |
| GNU Make | Existing verification commands | yes | 3.81 | Run commands directly if Make is unavailable elsewhere. [VERIFIED: `make --version`] |
| Kalshi API network access | User live data fetches | not probed during research | current docs reachable through browser tool | Keep automated tests fixture-only. [CITED: https://docs.kalshi.com/] [VERIFIED: tests/fixtures] |

**Missing dependencies with no fallback:**
- `polars` is missing in the current Python environment, and Phase 2 should add it to `pyproject.toml` before normalized Parquet tasks. [VERIFIED: Python import failure]

**Missing dependencies with fallback:**
- Live Kalshi API access was not exercised in this research; the implementation can still be planned and tested with fixtures, then user live runs can validate credentials/network separately if needed. [VERIFIED: .planning/phases/01-research-cli-foundation/01-03-SUMMARY.md]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 with Ruff 0.15.12 linting. [VERIFIED: `python -m pytest --version`] [VERIFIED: `python -m ruff --version`] |
| Config file | `pyproject.toml` contains pytest test paths, Python path, Ruff target version, and line length. [VERIFIED: pyproject.toml] |
| Quick run command | `python -m pytest tests/test_kalshi_routing.py tests/test_kalshi_normalize.py -q` after Wave 0 creates these tests. [VERIFIED: Makefile patterns] |
| Full suite command | `make check` runs quick tests, full tests, and lint. [VERIFIED: Makefile] |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| DATA-01 | Discovers eligible Kalshi BTC 15-minute markets for explicit UTC date ranges and records skipped ambiguous rows. [VERIFIED: .planning/REQUIREMENTS.md] | unit + CLI | `python -m pytest tests/test_kalshi_discovery_cli.py -q` | no - Wave 0 |
| DATA-01 | Splits discovery across live and historical markets when date range spans cutoff. [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md] | unit | `python -m pytest tests/test_kalshi_routing.py -q` | no - Wave 0 |
| DATA-02 | Normalizes live and historical candlesticks despite field-name shape differences. [CITED: https://docs.kalshi.com/api-reference/market/get-market-candlesticks] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks] | unit | `python -m pytest tests/test_kalshi_normalize.py::test_normalizes_live_and_historical_candlesticks -q` | no - Wave 0 |
| DATA-02 | Normalizes public trades from live and historical trade endpoints with endpoint provenance. [CITED: https://docs.kalshi.com/api-reference/market/get-trades] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-trades] | unit | `python -m pytest tests/test_kalshi_normalize.py::test_normalizes_trade_pages_with_source_endpoint -q` | no - Wave 0 |
| DATA-02 | Normalizes current orderbooks as bid levels and derived ask fields without inventing historical depth. [CITED: https://docs.kalshi.com/getting_started/orderbook_responses] | unit | `python -m pytest tests/test_kalshi_normalize.py::test_normalizes_orderbook_bid_levels_and_implied_asks -q` | no - Wave 0 |
| DATA-02 | Writes raw pages, normalized Parquet outputs, and a manifest with source/output metadata. [VERIFIED: src/albert/cache.py] [VERIFIED: src/albert/manifest.py] | integration | `python -m pytest tests/test_kalshi_cache_manifest.py -q` | no - Wave 0 |

### Sampling Rate

- **Per task commit:** run the focused test file for the touched module plus `python -m ruff check src tests`. [VERIFIED: Makefile]
- **Per wave merge:** run `make check`. [VERIFIED: Makefile]
- **Phase gate:** full `make check` green before `/gsd-verify-work`. [VERIFIED: .planning/config.json]

### Wave 0 Gaps

- [ ] `tests/fixtures/kalshi/*.json` - live/historical markets, candlesticks, trades, orderbooks, missing outcome, missing history, and ambiguous BTC market cases. [VERIFIED: current `tests/fixtures` lacks Kalshi API payloads]
- [ ] `tests/test_kalshi_routing.py` - cutoff field selection and spanning-window splits for markets/candlesticks/trades. [VERIFIED: current tests list]
- [ ] `tests/test_kalshi_normalize.py` - strict schema normalization for markets, price observations, trades, orderbooks, outcomes, skipped markets, and quality summaries. [VERIFIED: current tests list]
- [ ] `tests/test_kalshi_client.py` - pagination, timeout/retry status handling, and raw cache path writes with mocked HTTP transport. [VERIFIED: current tests list]
- [ ] `tests/test_kalshi_discovery_cli.py` - Typer command arguments, output paths, summary text, and manifest creation. [VERIFIED: current tests list]
- [ ] `pyproject.toml` dependency update for `httpx`, `polars`, and `pyarrow`. [VERIFIED: pyproject.toml]

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no for public Phase 2 endpoints by default; possibly yes if future authenticated orderbook access is required | Do not store API keys in config/manifests; existing manifest/config code rejects secret-like keys. [VERIFIED: src/albert/manifest.py] [CITED: https://docs.kalshi.com/getting_started/orderbook_responses] |
| V3 Session Management | no | No user sessions are introduced in Phase 2. [VERIFIED: .planning/PROJECT.md] |
| V4 Access Control | no | Local CLI only; no multi-user authorization boundary is introduced. [VERIFIED: .planning/PROJECT.md] |
| V5 Input Validation | yes | Validate date ranges, safe cache segments, known endpoint names, and Pydantic normalized row schemas. [VERIFIED: src/albert/cache.py] [VERIFIED: src/albert/manifest.py] |
| V6 Cryptography | yes, limited to HTTPS and hashes | Use HTTPS API URLs through HTTPX and existing SHA-256 parameter/config hashes; do not implement custom cryptography. [VERIFIED: src/albert/cache.py] [VERIFIED: src/albert/manifest.py] |
| V8 Data Protection | yes | Raw cached payloads and manifests must exclude secret-like keys and retain provenance for reproducibility. [VERIFIED: src/albert/manifest.py] [VERIFIED: AGENTS.md] |

### Known Threat Patterns for Python CLI Data Ingestion

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal through source/endpoint/table names | Tampering | Reuse `sanitize_segment()` and cache helpers for all raw/normalized paths. [VERIFIED: src/albert/cache.py] |
| Secret leakage into manifests | Information Disclosure | Reuse `RunManifest` and `SourceMetadata` secret-like key rejection; never place API keys in persisted params. [VERIFIED: src/albert/manifest.py] |
| Look-ahead from wrong time filters | Tampering | Preserve endpoint, params, fetched time, and covered ranges per page; test `min_ts`, `max_ts`, close/settled ranges. [VERIFIED: src/albert/manifest.py] [CITED: https://docs.kalshi.com/api-reference/market/get-trades] |
| Silent data corruption from schema drift | Tampering | Use Pydantic `extra="forbid"` for normalized contracts and keep unknown fields only in raw JSON. [VERIFIED: src/albert/manifest.py] [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md] |
| Rate-limit amplification | Denial of Service | Use bounded retries with exponential backoff for `429` and explicit max pages/window limits. [CITED: https://docs.kalshi.com/getting_started/rate_limits] [VERIFIED: .planning/phases/02-kalshi-market-data/02-CONTEXT.md] |

## Sources

### Primary (HIGH confidence)

- `AGENTS.md` - project scope, constraints, GSD workflow instruction, and no project-local skills declaration. [VERIFIED: file read]
- `.planning/phases/02-kalshi-market-data/02-CONTEXT.md` - locked Phase 2 decisions and reusable code integration points. [VERIFIED: file read]
- `.planning/REQUIREMENTS.md` - DATA-01 and DATA-02 requirement text. [VERIFIED: file read]
- `.planning/ROADMAP.md` - Phase 2 goal, success criteria, and planned split. [VERIFIED: file read]
- `src/albert/cache.py` - safe cache segment, raw JSON path, normalized Parquet path, stable parameter hash. [VERIFIED: codebase grep]
- `src/albert/manifest.py` - source/output/run manifest contracts and secret-like key rejection. [VERIFIED: codebase grep]
- Kalshi Historical Data docs - live/historical partitioning, cutoff semantics, target live window, and endpoint list. [CITED: https://docs.kalshi.com/getting_started/historical_data]
- Kalshi Historical Cutoff docs - cutoff fields and route meanings. [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-cutoff-timestamps]
- Kalshi Markets docs - filters and cursor pagination for `/markets`. [CITED: https://docs.kalshi.com/api-reference/market/get-markets]
- Kalshi Historical Markets docs - historical market endpoint. [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-markets]
- Kalshi Candlesticks docs - live/historical paths, required timestamps, periods, and response shape. [CITED: https://docs.kalshi.com/api-reference/market/get-market-candlesticks] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks]
- Kalshi Trades docs - live/historical trade endpoints and fields. [CITED: https://docs.kalshi.com/api-reference/market/get-trades] [CITED: https://docs.kalshi.com/api-reference/historical/get-historical-trades]
- Kalshi Orderbook docs - bid-only book structure and derived asks. [CITED: https://docs.kalshi.com/getting_started/orderbook_responses]
- Kalshi Fixed-Point Migration docs - `_dollars`, `_fp`, subpenny, fractional-contract handling. [CITED: https://docs.kalshi.com/getting_started/fixed_point_migration]
- Kalshi Pagination docs - cursor pagination pattern. [CITED: https://docs.kalshi.com/getting_started/pagination]
- Kalshi Rate Limits docs - 429 behavior and no `Retry-After`/rate-limit headers. [CITED: https://docs.kalshi.com/getting_started/rate_limits]

### Secondary (MEDIUM confidence)

- Kalshi public market URL path for `KXBTC15M` - used as evidence for a candidate BTC 15-minute series identifier. [CITED: https://kalshi.com/markets/kxbtc15m/bitcoin-price-up-down/kxbtc15m-25dec242215]
- Kalshi Help Center Crypto Markets - settlement methodology context for crypto contracts. [CITED: https://help.kalshi.com/en/articles/13823838-crypto-markets]
- `python3 -m pip index versions ...` - package latest-version checks for httpx, polars, pyarrow, Typer, Pydantic, pytest, and Ruff. [VERIFIED: shell output]

### Tertiary (LOW confidence)

- Third-party indexed pages showing recent `KXBTC15M-*` tickers were used only to corroborate the candidate series name, not as implementation authority. [VERIFIED: WebSearch]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - existing project stack plus current PyPI checks identify the needed libraries and versions. [VERIFIED: pyproject.toml] [VERIFIED: shell output]
- Architecture: HIGH - module boundaries map directly to existing CLI/cache/manifest contracts and official Kalshi endpoint families. [VERIFIED: src/albert/cache.py] [VERIFIED: src/albert/manifest.py] [CITED: https://docs.kalshi.com/getting_started/historical_data]
- Pitfalls: HIGH for endpoint routing, fixed-point parsing, pagination, orderbook complement logic, and rate-limit handling because they are in official docs; MEDIUM for BTC series discovery because final eligibility should be validated against API payloads. [CITED: https://docs.kalshi.com/getting_started/historical_data] [CITED: https://docs.kalshi.com/getting_started/fixed_point_migration] [CITED: https://docs.kalshi.com/getting_started/orderbook_responses] [CITED: https://kalshi.com/markets/kxbtc15m/bitcoin-price-up-down/kxbtc15m-25dec242215]

**Research date:** 2026-04-28 [VERIFIED: system date]
**Valid until:** 2026-05-05 for Kalshi API details and package versions because endpoint docs and PyPI versions can change quickly; 2026-05-28 for project-local architecture contracts unless Phase 1 contracts are refactored. [ASSUMED]
