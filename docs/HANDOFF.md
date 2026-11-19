# Albert: handoff prompt (server edition, updated 2026-11-19)

Paste everything below the line into a new agent session started in `~/albert` on the AWS host
(`ssh albert` from the owner's Mac lands as `ubuntu` there). `docs/HANDOFF.txt` is the same text,
ready to paste.

---

You are taking over **Albert**, a Python research CLI and small-stake trading system for
**non-financial prediction markets** on **Kalshi**. **It trades real money.** You are running ON the
production host: AWS EC2 `i-01990679def3bfdf4` (us-east-2, same region as Kalshi's exchange),
Ubuntu 24.04, repo at `~/albert` (work on branch `restart`; PRs go to `master` on GitHub
`dandan002/albert`; venv `~/albert/.venv`, use `.venv/bin/python`). Read `AGENTS.md` first (ground
rules, binding), then `.planning/LIVE-PLAN.md` (everything live: the newest entry is on top),
`.planning/research/SUMMARY.md` (every study and result), `README.md` and `deploy/aws.md` (this host).

## Goal and hard constraints
- **Money:** bankroll about $100, all on Kalshi (cash $97.28 on 2026-11-19).
  - The objective is consistent, positive returns from many small bets.
  - Ruin avoidance is a hard constraint.
  - Polymarket US is on hold.
- **Polymarket international (polymarket.com) is DATA ONLY.** The owner is a US resident: never
  place orders there, never build geoblock workarounds.
- **Live trading is ON for one strategy only:** the mentions NO-bid bot.
  - The owner enabled it: `KALSHI_LIVE_ENABLED=1` is in `.env`, and the prod bot runs with `--live`.
  - Prod orders need both. Never set the flag, start another live strategy, or raise a limit
    unless the owner asks.
  - The rule from AGENTS.md: paper results first, then explicit owner enablement, hard per-order and
    daily-loss caps, a kill switch.
- **The account holds the owner's MANUAL position.** It is `KXIMPEACH-29-JAN20`: 67.22 contracts,
  plus a resting NO order at 8c.
  - Bots must never cancel or trade anything they did not create.
  - Never call Cancel All.
  - Cancel only orders whose `client_order_id` starts with the bot prefix: `alb…`; the mentions bot
    is `albm`, the makers `albi` and `albl`.
  - Skip markets with a position or a non-bot order.
  - The order client (`venues/kalshi_trade.py`) enforces a write allowlist for this.
- **Secrets** live only in `~/albert/.env` (mode 600). Never print, log or copy them anywhere else.
  - Prod: `KALSHI_API_KEY_FAST_ID` / `KALSHI_PRIVATE_KEY_FAST` (Ed25519, preferred), and
    `KALSHI_API_KEY_ID` / `KALSHI_PRIVATE_KEY` (RSA, same account).
  - Demo: `KALSHI_DEMO_API_KEY_ID` / `KALSHI_DEMO_PRIVATE_KEY`.
  - Also `PANDASCORE_TOKEN` and `KALSHI_LIVE_ENABLED`.
- **Fees:** re-check Kalshi fees before any live run, including restarting a prod bot.
  - kalshi.com's fee PDF returns 429 to scripts, so use `GET /series/fee_changes` (public).
  - Last check 2026-11-19: 107 rows, the latest 2026-09-03, none on a mention series.
  - Mention series are `quadratic`, so resting NO bids pay no maker fee.
- **Data-source policy (owner decision):** use a host only if its OWN robots.txt allows us, and
  never fetch blocked content through another host.
  - Allowed: Kalshi, the Polymarket APIs, the ESPN API host, the MLB Stats API, and IEM
    (mesonet.agron.iastate.edu; avoid /usage, /tmp, /archive/nexrad).
  - Removed and not to be used: rollcall.com/Factba.se, api.weather.gov, Open-Meteo,
    football-data.co.uk, nflverse/GitHub release downloads.
- **AGENTS.md rules:**
  - Real payloads only: fixtures are trimmed copies of recorded responses (`albert capture`).
  - A smoke test with zero rows is a failure.
  - Raw data is append-only. No look-ahead. Realistic execution. Win rate is not edge.
  - Every strategy reports Sharpe and volatility from `research/vol_metrics` ledgers on the $100
    bankroll.
- **Git:** commit, push or open PRs only when the owner asks. Make small, focused commits ending
  with the attribution lines your harness specifies.

## What is running here (systemd; `sudo systemctl status 'albert-*'`, logs via journalctl)
| Unit | What | Output | Stop |
|---|---|---|---|
| `albert-mentions@prod` | **REAL ORDERS.** Pre-speech NO bids on mention markets (`albert live mentions --env prod`, args `deploy/mentions-prod.conf`) | `data/live/mentions/prod/<run>/ledger.jsonl` | `touch data/live/mentions/STOP` or `sudo systemctl stop` (shutdown cancels every `albm` order) |
| `albert-mentions@paper` | The same bot on the same markets with simulated fills (`deploy/mentions-paper.conf`) | `data/live/mentions/paper/` | `touch data/live/mentions/paper/STOP` |
| `albert-guard@prod` | Account-wide guard. Cancel-only. Appends every fill to the audit log, halts all bots until 00:00 UTC if the bot book loses more than $15 in a day, and turns on the global kill switch if bot exposure passes $60 (`deploy/guard-prod.conf`) | `data/live/guard/prod/`, `data/live/audit/prod/` | `sudo systemctl stop albert-guard@prod` |
| `albert-record` | Kalshi + Polymarket WebSocket L2 recorder (`deploy/record-loop.sh --paired-days 3 --skip-vol`) | `data/raw/stream/` (`k<n>`, `p<n>`) | `touch data/record.stop` |
| `albert-record-vol` | Kalshi L2 of complete vol-study events (`RECORD_CMD=record-vol deploy/record-loop.sh --ladder-interval-ms 1000`). It exits and restarts often (about 2,000 restarts by 11-19), cause not investigated | `data/raw/stream/` (`v<n>`, `c<n>`) | `touch data/record-vol.stop` |

Stopped, with units kept:
- `albert-live-maker@{liquid,illiquid}`: the real-money makers. They lost −$5.17 on 75 fills
  between 11-08 and 11-11, and `data/live/STOP` keeps them off.
- `albert-paper` and `albert-paper-illiquid`: paper makers.
- `albert-forward`: the forward paper test.
- `albert-mentions@demo`.

`albert-archive.timer` is not installed: the S3 bucket and IAM policy are pending (`deploy/aws.md`).

The host has 2 vCPU and 8 GB. Run heavy research with `nice -n 10` and keep it under 3 GB of RAM,
so the recorders and bots never starve.

## Live trading: how it works and how to stop it
- **Mentions bot** (`src/albert/live_mentions.py`):
  - **Bid:** one post-only NO bid per word leg at floor(1 − fair − 12c), with fair =
    σ(−0.1156 + 1.0474·logit(mid)).
  - **Windows:** from start − 12 h to start − 6 h (early phase), then on unfilled legs until
    start − 10 min. Starts come from `GET /milestones`.
  - **Budget:**
    - Expected fill cost ≤ $15, counting fills plus each resting bid's p(fill) × price. Fills that
      push it over cancel resting bids, largest expected cost first.
    - $40 hard cap on resting bids plus unsettled fills.
    - No new events after $15 of fills in a day, and at most 6 events.
  - **Circuit breaker:** a cheap leg trading or bidding at ≥ 95c triggers the event's order
    group, which cancels all its orders.
  - **Filters:** finance speakers and earnings calls are skipped. Executives trade only at mostly
    political venues (DealBook).
- **Order client** (`src/albert/venues/kalshi_trade.py`):
  - Per-order cap of 5 contracts and $2, and prod orders must be post-only.
  - Refuses every new order while `data/live/KILL` or the guard's `data/live/HALT.json` is in force.
  - Audits every write request and response to `data/live/audit/prod/orders-<day>.jsonl`.
- **Guard** (`src/albert/live_guard.py`): see the table above.
- **Stopping:**
  - Global stop: `.venv/bin/python -m albert live kill --live --reason "..."`. It writes `KILL` and
    cancels every bot order. Bots exit and the guard sweeps every minute.
  - Undo: `albert live resume`, then `sudo systemctl start albert-mentions@prod`.
- **Restarting a prod bot cancels its resting orders.** Restart only when nothing is resting or the
  bids can be lost: check `albert live check --env prod` first. Re-check fees before the restart.

## Where the research stands (details in SUMMARY.md)
- **Studies:** about 70, in two groups.
  - Up to 11-07: microstructure makers, momentum, arbitrage, lead-lag, market-making models, and
    fundamental models.
  - Since: vol models, TA and calendar effects, the Kalshi liquidity incentive program, linked
    markets, Polymarket lead-lag, and behavioural mechanisms by market family.
  - Protocol: rules are frozen on data before 2026-10-01 and scored once on the holdout from
    Oct 1. t-stats are clustered by event and date.
- **Verdict:** no taker edge anywhere. Public prices are efficient. Everything else shrank or
  flipped on its holdout.
- **The one robust edge: pre-speech NO bids on mention markets.**
  - Evidence:
    - Frozen rule on 548 unseen events: +7.9c per fill (t 6.4) at a 12c margin.
    - Early phase: +7.27c per fill on 433 holdout fills.
    - Expected-fill budget: $7.12 → $9.82 per week on the holdout, Sharpe 8.21 → 9.66.
  - It is price pressure from YES buyers before speeches, not information.
- **Live result** (`albert paper mentions-report --env prod`, 2026-11-19): 85 bids on 10 events, 24
  fills, all settled: +$2.82 (+11.8c per fill). That is too few fills to judge.

## Next steps, in order
1. **Daily check:**
   - `albert live check --env prod` (switches, guard, resting orders).
   - `albert paper mentions-report --env prod` and `--env paper`. Compare real against simulated
     fill rates, because research assumes we are never ahead in the queue.
   - `albert live audit --env prod`: refusals or exchange rejects need an explanation.
2. **Early-phase review:** demote it (drop `--early-hours 12`) if its fills earn below +5c over about
   6 weeks from 2026-11-18.
3. **Scaling:** scale only on the go/no-go: event-clustered t ≥ 2 and at least 60% positive weeks,
   on settled PnL. Report it with Sharpe and volatility.
4. **Open items:**
   - The bots don't require a live guard heartbeat.
   - The cause of the `albert-record-vol` restarts is unknown.
   - The archive timer is pending.

## Useful commands
- Tests and lint: `.venv/bin/python -m pytest -q` (961 pass) and `.venv/bin/python -m ruff check src tests`.
- Account (read-only): `.venv/bin/python -m albert live check --env prod`.
- Bot orders only: `.venv/bin/python -m albert live cancel-own --env prod --live`.
- Kalshi hosts: signed REST and every WebSocket use the direct us-east-2 hosts (`external-api*`,
  `warp-ws` for fills and user_orders). Unsigned market-data REST stays on
  `api.elections.kalshi.com`. Latencies are in `deploy/aws.md`.
- Disk (96 GB):
  - Use: 6.5 GB on a weekday and 13-16 GB on a weekend day, with ladders conflated to 1 s since
    11-16.
  - Recorders stop on low disk: vol at 20 GB free, main at 10 GB.
  - The archive timer, once installed, keeps the last 4 days of raw stream on disk.
