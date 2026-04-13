# Phase 1: Polymarket Production Readiness - Context

**Gathered:** 2026-04-12
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase fixes critical authentication blockers in the Polymarket integration so orders can execute successfully. The scope includes:
1. Implementing official SDK authentication for PolymarketAdapter
2. Registering the adapter in the execution engine
3. Verifying WebSocket data ingestion works

This does NOT include: Web dashboard, strategy expansion, or multi-exchange routing.

</domain>

<decisions>
## Implementation Decisions

### Authentication
- **D-01:** Use official Polymarket Python SDK (`py-clob-client`) for authentication — handles L2 ECDSA signing automatically, well-maintained, handles all auth headers

### Error Handling
- **D-02:** Fail gracefully — retry with exponential backoff (3 attempts), then disable PolymarketAdapter if auth fails, continue trading with other exchanges (Kalshi)

### Adapter Registration
- **D-03:** Register both WebSocket ingestor and REST adapter at startup — instantiate PolymarketAdapter when env vars present, start PolymarketIngestor for market data streaming

### Testing
- **D-04:** Use staging CLOB API for testing — test orders execute but with fake accounts, no real money at risk

### Agent's Discretion
- The specific retry backoff timing (specific seconds) — agent can decide
- The exact threshold for "disabling" the adapter (consecutive failures vs. single auth failure) — agent can decide
- How to verify staging API works before production — agent can decide

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Specs
- `docs/superpowers/specs/2026-03-29-trading-design.md` — Design spec (approved 2026-03-29)
- `.planning/PROJECT.md` — Project context
- `.planning/REQUIREMENTS.md` — Requirements (PM-01, PM-02, PM-03)

### Existing Code
- `albert/execution/adapters/polymarket.py` — Current (broken) adapter implementation
- `albert/execution/adapters/kalshi.py` — Working adapter for reference patterns
- `albert/ingestor/polymarket.py` — WebSocket ingestor (verify it still works)
- `albert/__main__.py` — Entry point where adapters are registered (line ~62)

### Research
- `.planning/research/STACK.md` — SDK recommendations (py-clob-client)
- `.planning/research/PITFALLS.md` — Pitfall 1: "ECDSA authentication not implemented"

### External Docs
- Polymarket CLOB API docs: https://docs.polymarket.com
- py-clob-client: https://github.com/Polymarket/py-clob-client

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `KalshiAdapter` in `albert/execution/adapters/kalshi.py` — Can use as implementation pattern for SDK integration
- `BaseIngestor` in `albert/ingestor/base.py` — PolymarketIngestor already extends this

### Established Patterns
- Retry logic: `kalshi.py` uses 3 retries with exponential backoff (`2 ** attempt` seconds)
- Adapter interface: `ExchangeAdapter` ABC in `base.py` with `place_order()`, `cancel_order()`, `get_bankroll()`

### Integration Points
- `albert/__main__.py` line ~62: Adapters are instantiated and added to dict
- `albert/__main__.py`: `run()` starts all services via `asyncio.gather()`

</code_context>

<specifics>
## Specific Ideas

No specific "I want it like X" moments mentioned. Standard approaches from the SDK are acceptable.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-polymarket-production-readiness*
*Context gathered: 2026-04-12*