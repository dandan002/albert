---
phase: 01-research-cli-foundation
plan: 02
subsystem: data-foundation
tags: [python, pytest, pydantic, cache, manifests, provenance]
requires:
  - phase: 01-research-cli-foundation
    provides: Installable Python package, Typer CLI skeleton, and AlbertConfig contract
provides:
  - Deterministic raw JSON, normalized Parquet, and run manifest path helpers
  - Provenance-rich Pydantic manifest models for source metadata, outputs, and runs
  - Secret-like key rejection for persisted run and source parameters
affects: [kalshi-ingestion, deribit-ingestion, modeling, backtesting, reports]
tech-stack:
  added: []
  patterns:
    - Allowlisted filesystem path segments plus hashed request parameters
    - Explicit manifest provenance fields with deterministic config hashing
    - Injected timestamps and run IDs for reproducible tests
key-files:
  created:
    - src/albert/cache.py
    - src/albert/manifest.py
    - tests/test_cache.py
    - tests/test_manifest.py
  modified: []
key-decisions:
  - "Use strict lowercase cache path segments and reject whitespace, traversal, slashes, and empty values instead of normalizing unsafe input."
  - "Hash canonical request parameters into cache filenames while storing original non-secret parameters in manifests."
  - "Persist run provenance with explicit config_path, config_hash, cache_root, run_root, source metadata, and output metadata."
patterns-established:
  - "Cache helpers return Path objects under configured roots and never encode raw parameters as path segments."
  - "Manifest models reject secret-like keys before JSON serialization."
  - "Tests inject exact timezone-aware datetimes for deterministic manifest assertions."
requirements-completed: [DATA-04]
duration: 4min
completed: 2026-04-28
---

# Phase 01 Plan 02: Cache Paths and Run Manifests Summary

**Deterministic cache path contracts and provenance manifests for reproducible source-data runs**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-28T09:06:35Z
- **Completed:** 2026-04-28T09:10:49Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added safe cache path helpers for raw JSON responses, normalized Parquet outputs, and run manifest locations.
- Added Pydantic manifest models that require DATA-04 provenance fields and reject secret-like keys before serialization.
- Added focused tests for path determinism, root containment, manifest field validation, config hashing, JSON writing, and cache/manifest integration.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement safe deterministic cache path helpers** - `fec1970` (test), `c71b6b5` (feat)
2. **Task 2: Implement provenance-rich manifest models** - `1752b70` (test), `75183fe` (feat)
3. **Task 3: Connect cache paths and manifests in integrated tests** - `9f64132` (test)

**Plan metadata:** pending final docs commit.

_Note: TDD tasks produced paired test and feature commits where production code was needed. Task 3 was test-only because Tasks 1 and 2 already satisfied the integrated behavior._

## Files Created/Modified

- `src/albert/cache.py` - Safe path segment validation, stable parameter hashing, and root-contained raw/normalized/manifest path helpers.
- `src/albert/manifest.py` - Source, output, and run manifest models plus deterministic config hashing and JSON writing.
- `tests/test_cache.py` - Cache path determinism, suffix, root containment, manifest path, and unsafe segment tests.
- `tests/test_manifest.py` - Manifest provenance, validation, serialization, secret rejection, and cache path integration tests.

## Decisions Made

- Unsafe path segments fail fast instead of being normalized into ambiguous filesystem names.
- Cache filenames use a stable SHA-256 parameter hash, while manifests retain the original non-secret request parameters for auditability.
- Manifest timestamps are supplied by callers and validated as timezone-aware, which keeps tests deterministic and prevents hidden wall-clock state.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Task 3's integration test passed immediately because the behavior had already been implemented by Tasks 1 and 2. No production changes were needed for that task.

## Verification

- `python -m pytest tests/test_cache.py tests/test_manifest.py -q` - 20 passed
- `python -m pytest tests/test_cli.py tests/test_config.py tests/test_cache.py tests/test_manifest.py -q` - 27 passed
- `python -m ruff check src tests` - all checks passed
- `python -m pytest -q` - 29 passed

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 03 can add the baseline test harness and CI-style verification commands using the cache and manifest contracts created here.

## Self-Check: PASSED

- Verified created files exist: `src/albert/cache.py`, `src/albert/manifest.py`, `tests/test_cache.py`, `tests/test_manifest.py`, and this summary.
- Verified task commits exist: `fec1970`, `c71b6b5`, `1752b70`, `75183fe`, and `9f64132`.
- Verified command evidence is recorded for pytest and Ruff checks.

---
*Phase: 01-research-cli-foundation*
*Completed: 2026-04-28*
