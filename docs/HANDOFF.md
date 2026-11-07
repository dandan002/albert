# Albert: handoff prompt (server edition, 2026-11-07)

Paste everything below the line into a new agent session started in `~/albert` on the AWS host
(`ssh albert` from the owner's Mac lands as `ubuntu` there).

---

You are taking over **Albert**, a Python research CLI and small-stake trading system for
**non-financial prediction markets** on **Kalshi**. You are running ON the production host:
AWS EC2 `i-01990679def3bfdf4` (us-east-2, same region as Kalshi's exchange), Ubuntu 24.04, repo at
`~/albert` (branch `restart`, venv `~/albert/.venv`, use `.venv/bin/python`). Read `AGENTS.md`
first (ground rules, binding), then `.planning/research/SUMMARY.md` (every study and result),
`.planning/LIVE-PLAN.md` (order layer and live plan) and `deploy/aws.md` (this host).

## Goal and hard constraints
- Bankroll ~$100 (Kalshi prod balance ~$95.5). Objective: consistent, positive returns from many
  small bets; ruin avoidance is a hard constraint. Polymarket US is on hold.
- **Polymarket international (polymarket.com) is DATA ONLY.** The owner is a US resident: never
  place orders there, never build geoblock workarounds.
- **Live trading is gated:** paper results first, then explicit owner enablement, hard per-order
  and daily-loss caps, kill switch. Prod orders need BOTH `KALSHI_LIVE_ENABLED=1` in `.env` AND
  `--live` on the command. Neither is set. Never set the flag yourself.
- **The prod Kalshi account holds the owner's MANUAL position** (KXIMPEACH-29-JAN20, ~67 contracts,
  plus a resting order). Bots must never cancel or trade anything they did not create: never call
  Cancel-All; cancel only orders whose `client_order_id` starts with the bot prefix (`alb…`;
  mentions bot `albm…`) or that sit in the bot's own order group; skip markets with a position or
  non-bot order. The order layer (`venues/kalshi_trade.py`) enforces a write allowlist for this.
- **Secrets** live only in `~/albert/.env` (mode 600): prod `KALSHI_API_KEY_ID` /
  `KALSHI_PRIVATE_KEY` (RSA), demo `KALSHI_DEMO_API_KEY_ID` / `KALSHI_DEMO_PRIVATE_KEY` (Ed25519;
  the signer supports both), `PANDASCORE_TOKEN`. Never print, log or copy them anywhere else.
- **Data-source policy (owner decision):** use a host only if its OWN robots.txt allows us, and
  never fetch blocked content through another host. Allowed: Kalshi, Polymarket APIs, ESPN API host,
  MLB Stats API, IEM (mesonet.agron.iastate.edu; avoid /usage, /tmp, /archive/nexrad). Removed and
  not to be used: rollcall.com/Factba.se, api.weather.gov, Open-Meteo, football-data.co.uk,
  nflverse/GitHub release downloads.
- AGENTS.md rules: real payloads only (fixtures are trimmed copies of recorded responses), a smoke
  test with zero rows is a failure, raw data is append-only, no look-ahead, realistic execution,
  win rate is not edge. Commit only when asked; small focused commits ending with the attribution
  lines your harness specifies. Do not push unless the owner asks (this server copy is now the
  primary working copy; the owner's Mac copy is a backup of history up to 2026-11-07 01:45 UTC).

## What is running here (systemd; `sudo systemctl status 'albert-*'`, logs via journalctl)
| Unit | What | Output | Stop |
|---|---|---|---|
| `albert-record` | Kalshi + Polymarket WebSocket L2 recorder (`deploy/record-loop.sh --paired-days 3`) | `data/raw/stream/` | `touch data/record.stop` |
| `albert-paper` | paper maker on live books, no orders (`deploy/paper-loop.sh`) | `data/paper_l2/` | `touch data/paper_l2/STOP` |
| `albert-forward` | hourly forward paper test of 3 leads, no orders (`deploy/forward-loop.sh`, LEADS="mentions fcs rain") | `data/cache/forward/` | `touch data/cache/forward/STOP` |
| `albert-mentions@demo` | mentions NO-bid bot on Kalshi DEMO (args `deploy/mentions-demo.conf`) | `data/live/mentions/demo/` | `touch data/live/mentions/STOP` |
Prefer `sudo systemctl stop <unit>` (the stop files also halt the loops; a stopped unit with
`Restart=always` stays stopped only via systemctl). Host has 2 vCPU / 8 GB: run heavy research
with `nice -n 10` and keep < 3 GB RAM so the recorder never starves.

## Where the research stands (details in SUMMARY.md)
- ~45 studies: microstructure makers, momentum, arbitrage, lead-lag, intra-minute, market-making
  equations (AS/GLFT, Glosten–Milgrom/Das, Hawkes), curve fitting, volatile-window re-tests, and
  fundamental models (soccer, NCAAF, MLB, NBA props, college basketball, thin sports, hourly temp,
  rain, mentions, approval, culture). Harness: `src/albert/research/fundamental.py`.
- Verdict: no taker edge anywhere; public prices are efficient; fundamentals add information only
  in thin, wide-spread markets and passive fills are adversely selected enough to eat it.
- **The one robust lead: pre-speech NO bids on Kalshi mention markets** (`research/mentions_history`):
  rest one post-only NO bid per word leg at floor(1 − fair − m), fair = σ(−0.1156 + 1.0474·logit(mid)),
  from ~6 h before the speech to start − 10 min, hold fills to settlement. On 548 unseen events
  (Apr 2025 – Aug 2026): +5.8c/fill (t 5.4) at m = 8c, +7.9c (t 6.4) at 12c; 15/18 months positive,
  no decay; October +8–9c. It is a price pattern (the plain mid earns the same), no maker fee on
  mention series, capacity is collateral-bound (~$8 per event). Unknowns: real queue position.
- Order layer: `venues/kalshi_trade.py`, `live_l2.py`; mentions bot `live_mentions.py`
  (`albert live mentions`); demo plumbing `albert live plumbing` passes 11/11.

## Next steps, in order
1. Daily: `nice -n 10 .venv/bin/python -m albert.research.forward score all` and report the
   mentions lead (fills, PnL/fill, t, events) vs its plain-mid control; FCS after weekend games.
2. Watch the demo bot (`journalctl -u albert-mentions@demo`, ledger in `data/live/mentions/demo/`):
   events planned at ~6 h, closes at start − 10 min, breaker, no errors.
3. **Prod pilot only after the owner enables it** (they add `KALSHI_LIVE_ENABLED=1` to `.env`), ideally
   after 2–3 days / 15–20 forward events: create `deploy/mentions-prod.conf` with
   `MENTIONS_ARGS=--live --margin 0.12 --max-collateral 15 --max-daily-loss 10 --max-events 2`, run
   `.venv/bin/python -m albert live check --env prod` first (the manual KXIMPEACH position must show
   as blocked/untouched), then `sudo systemctl start albert-mentions@prod`, supervised. Report
   every fill; stop on any anomaly (`touch data/live/mentions/STOP`).
4. Later: measure live fill rate and queue position vs the backtest's trade-through rule; settle
   PnL on hundreds of contracts before any scaling (go/no-go: event-clustered t ≥ 2 and ≥ 60%
   positive weeks, on settlement).

## Useful commands
- Tests/lint: `.venv/bin/python -m pytest -q` (456 pass), `.venv/bin/python -m ruff check src tests`
- Accounts (read-only): `.venv/bin/python -m albert live check --env demo|prod`
- Bot orders only: `.venv/bin/python -m albert live cancel-own --env demo` (prod: add `--live`)
- Disk: recorder ~1.3 GB/day; 96 GB disk; the recorder stops itself on low disk.
