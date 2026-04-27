---
phase: 06-complete-health-monitoring
plan: 02
subsystem: health-monitoring
tags: [health, cli, main-loop, engines, integration]
dependency-graph:
  requires: [health-monitor-core]
  provides: [full-health-pipeline]
  affects: [albert/__main__.py, albert/cli.py, albert/health.py, albert/strategies/engine.py, albert/execution/engine.py, albert/portfolio/tracker.py]
tech-stack:
  added: []
  patterns: [TDD, engine task monitoring, CLI health queries]
key-files:
  created: []
  modified: [albert/__main__.py, albert/cli.py, albert/health.py, albert/strategies/engine.py, albert/execution/engine.py, albert/portfolio/tracker.py, tests/test_cli.py, tests/test_integration.py, tests/test_strategy_engine.py, tests/test_execution_engine.py, tests/test_portfolio_tracker.py, tests/test_health.py]
decisions:
  - Instantiated ingestors and engines before creating tasks to pass references to HealthMonitor
  - Added _started_at to all engines for future health diagnostics (not yet exposed in CLI)
  - cmd_health reads health_status table dynamically and populates adapters/ingestors/engines keys
metrics:
  duration_seconds: 254
  completed_date: "2026-04-27"
---

# Phase 06 Plan 02: Wire HealthMonitor into main loop and update health CLI Summary

**One-liner:** HealthMonitor wired into the main async loop with engine task tracking, and `python -m albert health` reports full component status from SQLite.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add engine health tracking and wire HealthMonitor into main loop | b10034d | albert/__main__.py, albert/strategies/engine.py, albert/execution/engine.py, albert/portfolio/tracker.py, albert/health.py, tests/test_strategy_engine.py, tests/test_execution_engine.py, tests/test_portfolio_tracker.py, tests/test_health.py |
| 2 | Update cmd_health to read health_status table | 95a1836 | albert/cli.py, tests/test_cli.py |
| 3 | Add integration test for full health pipeline | 8cc0a7e | tests/test_integration.py |

## Deviations from Plan

### Auto-fixed Issues

None - plan executed exactly as written.

### Test Adjustments

None beyond those documented in Plan 01.

## Auth Gates

None.

## Known Stubs

None.

## Threat Flags

None — no new security-relevant surface introduced beyond Plan 01's threat model.

## Self-Check: PASSED

- [x] albert/__main__.py wires HealthMonitor with adapters, ingestors, and engine_tasks
- [x] albert/cli.py queries health_status table
- [x] All commits exist in git log
- [x] Full test suite (84 tests) passes with no regressions
