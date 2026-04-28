---
phase: 01-research-cli-foundation
plan: 03
subsystem: testing
tags: [python, pytest, ruff, makefile, fixtures, provenance]
requires:
  - phase: 01-research-cli-foundation
    provides: Typer CLI, config loading, cache path helpers, and manifest models from Plans 01 and 02
provides:
  - Shared deterministic pytest fixtures for timestamps, run IDs, cache roots, run roots, and example config files
  - Non-secret config and source metadata fixtures for reusable provenance tests
  - Baseline contract tests for package entrypoint, imports, manifest fixture validity, fixture hygiene, and Phase 1 scope boundaries
  - CI-style local verification targets for quick tests, full tests, lint, and check
affects: [kalshi-ingestion, deribit-ingestion, modeling, backtesting, reports]
tech-stack:
  added: []
  patterns:
    - Shared pytest fixtures in tests/conftest.py without autouse behavior
    - Static fixture files validated through production Pydantic manifest models
    - Makefile targets mirror phase validation commands exactly
key-files:
  created:
    - tests/conftest.py
    - tests/fixtures/config/example.toml
    - tests/fixtures/manifest/source_metadata.json
    - tests/test_baseline.py
    - Makefile
  modified: []
key-decisions:
  - "Use explicit pytest fixtures instead of autouse fixtures so tests opt into deterministic paths and timestamps."
  - "Keep baseline fixtures free of secret-like keys and enforce that with a regression test."
  - "Keep Phase 1 verification local and deterministic through Makefile targets that call python -m pytest and python -m ruff."
patterns-established:
  - "Reusable test data lives under tests/fixtures and is validated through the same models production code will use."
  - "Phase boundary tests prevent external API client modules before the Kalshi and Deribit phases."
requirements-completed: [DATA-04]
duration: 3min
completed: 2026-04-28
---

# Phase 01 Plan 03: Test Harness, Fixtures, and Verification Commands Summary

**Deterministic pytest fixtures, baseline provenance contract tests, and local Makefile verification targets for the research CLI foundation**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-28T09:14:04Z
- **Completed:** 2026-04-28T09:17:28Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- Added shared pytest fixtures for a frozen UTC timestamp, deterministic run ID, cache root, run root, and example config path.
- Added reusable non-secret fixture files for config and Kalshi source metadata with endpoint, parameters, fetch time, covered range, and raw path.
- Added baseline tests proving script entrypoint, package import, manifest fixture validity, fixture hygiene, and the absence of Phase 1 external client modules.
- Added Makefile targets for `quick-test`, `test`, `lint`, and `check`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add shared deterministic pytest fixtures** - `c78a8c1` (test)
2. **Task 2: Add fixture files and baseline contract tests** - `73c951d` (test RED), `3556f56` (test GREEN)
3. **Task 3: Add CI-style verification commands and run the full validation contract** - `0bf61f5` (chore)

**Plan metadata:** pending final docs commit.

_Note: Task 2 used a RED/GREEN cycle. Task 1 was a test-harness artifact, so it was committed as a verified test fixture change._

## Files Created/Modified

- `tests/conftest.py` - Shared deterministic pytest fixtures without hidden autouse behavior.
- `tests/fixtures/config/example.toml` - Non-secret TOML config fixture.
- `tests/fixtures/manifest/source_metadata.json` - Source metadata fixture with endpoint, parameters, fetched time, covered range, and raw cache path.
- `tests/test_baseline.py` - Baseline repository contract tests for package metadata, fixture validity, fixture hygiene, and Phase 1 scope.
- `Makefile` - Local verification commands for quick tests, full tests, lint, and aggregate check.

## Decisions Made

- Used explicit opt-in pytest fixtures rather than autouse fixtures to avoid hidden global behavior.
- Validated static manifest fixture data through `SourceMetadata.model_validate_json` so fixture shape stays aligned with production provenance models.
- Made the Makefile mirror `01-VALIDATION.md` commands exactly, keeping local and CI-style verification surfaces consistent.

## Deviations from Plan

None - plan outputs and verification commands were executed as written.

## Issues Encountered

- Task 1 was marked TDD, but the planned artifact was pytest fixture code rather than production behavior. It was verified through the plan's quick pytest command and acceptance checks instead of a separate production GREEN commit.

## Verification

- `python -m pytest tests/test_cli.py tests/test_config.py tests/test_cache.py tests/test_manifest.py -q` - 27 passed
- `python -m pytest tests/test_baseline.py -q` - 5 passed
- `python -m pytest -q` - 34 passed
- `python -m ruff check src tests` - all checks passed

## Known Stubs

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 1 now has a repeatable local validation surface and reusable deterministic fixtures. Phase 2 can start Kalshi ingestion work with package, config, cache, manifest, and baseline test contracts in place.

## Self-Check: PASSED

- Verified created files exist: `tests/conftest.py`, `tests/fixtures/config/example.toml`, `tests/fixtures/manifest/source_metadata.json`, `tests/test_baseline.py`, `Makefile`, and this summary.
- Verified task commits exist: `c78a8c1`, `73c951d`, `3556f56`, and `0bf61f5`.
- Verified command evidence is recorded for quick pytest, baseline pytest, full pytest, and Ruff checks.

---
*Phase: 01-research-cli-foundation*
*Completed: 2026-04-28*
