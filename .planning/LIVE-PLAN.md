# Live (cash) maker pilot: plan

**UPDATE, 2026-11-25 ~18:50 UTC: two more rules trade real money.** User: "Let's just run the
approval NO and culture maker live please". They went live after 4 days of paper (apr 11/11 settled
NO, +$0.53; cmn 1 fill, +$0.16). The paper pass bars were not met; the user chose to go live. The
paper service keeps running as the control.
- **Service:** `albert-live-forward@prod` (`albert.live_forward`, `albert live forward`,
  `deploy/forward-prod.conf`).
- **cmn** (tag `albc`): the paper culture maker-NO rule with real post-only 1-lot NO bids. Each bid
  expires at its leg's close. Worst case (resting bids + unsettled fills) ≤ $15.
- **apr** (tag `alba`): the paper approval 5–15c NO rule as 1-lot immediate-or-cancel takers at
  1 − bid. Worst case ≤ $5. One change from paper: legs where NO + fee ≥ 99c are skipped, since they
  cannot profit (a 1c bid → 99c + 1c fee; found in the dry smoke).
- **Order client:** `KalshiTrader(allow_taker=True)` now permits non-post-only prod orders only as
  IOC/FOK, and only on that one client. The per-order caps still apply.
- **Caps:** the bots' worst cases now sum to the guard's $60 at-risk KILL line (mentions $40 +
  cmn $15 + apr $5).
- **Fees re-checked from the API's series schedule:** 36 series quadratic ×1, KXLLM1 also charges
  maker fees. kalshi.com is robots-blocked for us.
- **Safety:** markets with any position or other order are skipped. KILL or STOP cancels everything
  and exits; HALT cancels resting bids and pauses; SIGTERM cancels and the next start reposts
  the cmn bids.
- **Ledger:** `data/live/forward/prod/ledger.jsonl` and `status.json`. The guard books the fills
  under `albc` / `alba` (`albert live check --env prod`).
- **Dry smoke** (real reads, orders only logged) at the next real decision times: cmn would post 5
  bids over 3 events (NO 0.83–0.85); apr would have taken only a 1c-bid leg, now skipped.
- **Decision times:** apr daily at 06:00 UTC (settle − 12 h); cmn at each event's close − 72 h (next:
  2026-11-26 14:59 UTC).

**UPDATE, 2026-11-23 05:45 UTC**: second research swarm finished (12 agents + 2 skeptics;
`.planning/research/SUMMARY.md`). No new live or paper strategy; no service was restarted or changed
(the live services logged no 429s during the swarm). Pending user decisions from the portfolio study
(`data/research/portfolio/report.md`): raise the cmn paper cash cap $25 → $40 (would restart
`albert-paper-forward`), replace the paper pass bars with fills-needed gates (cmn ≈ 164 fills), and
add per-bot guard caps before any second live bot (the $60 at-risk KILL would trip with mentions +
cmn). Mentions supply risk: Kalshi has listed no Trump mention events since Nov 11.

**UPDATE, 2026-11-21 01:50 UTC**: new paper service `albert-paper-forward` (`albert.paper_forward`;
never sends an order, Kalshi public data only), started 01:45 UTC: forward paper tests of the swarm's
culture maker NO (`cmn`), HITS → KXALBUMEQUIV (`hits`) and approval NO (`apr`) rules, $25 paper
bankroll each (user: "assuming $20-30 bankroll initially"). New recorder timers `albert-rec-music.timer`
and `albert-rec-approval.timer`. Kill file `data/paper_forward/STOP`; score with
`python -m albert.paper_forward report`. First decisions: cmn 2026-11-21 04:59 UTC (Netflix weekly),
apr 06:00 UTC, hits at the first HITS preview (Tue 2026-11-24). Existing bots untouched.

**STATUS, 2026-11-20 22:00 UTC** (read-only check; no bot restarted or changed):

- **Running:** `albert-mentions@prod` (live) and `@paper` since 2026-11-19 21:19 / 21:18 UTC, 0 restarts,
  no errors; `albert-guard@prod`; recorders `albert-record`, `albert-record-vol`, `albert-record-hl`;
  `albert-archive.timer` (daily 08:00 UTC). Both live makers are stopped.
- **No mentions event since the restart, by design:** the Mentions milestone calendar has nothing the
  bot trades. 145 of 150 upcoming milestones are earnings calls (excluded); the others are the Trump
  monthly window series (Nov 30, excluded), Jensen Huang's NVIDIA GTC DC keynote (Dec 1; an executive,
  traded only at a mostly political event) and the FOMC press conference (Dec 9, financial). Speech
  milestones usually appear a few days ahead. Last live trades: Nov 18 (Vance, Carney).
- **Account:** balance $97.28; the only resting order is the manual KXIMPEACH-29-JAN20 NO; kill and
  halt off; bot book unchanged from the table below (−$2.35 across bots). Paper mentions since
  2026-11-18: 34 orders, 1 fill.

**PHASE 6 SAFETY LAYER, 2026-11-19** (user: "Let's build phase 6 we're live trading already anyway"):
the roadmap's Phase 6 item (order client with enable flag, per-order and daily-loss caps, global kill
switch, full order/fill audit log) is complete. New on top of each bot's own limits:

- **Global kill switch** `data/live/KILL` (`albert live kill [--reason] [--live]`, or `touch`): the
  order client refuses every new prod order, `albert-mentions@prod` cancels its orders and exits
  (systemd leaves it down), a maker stops, and the guard cancels every `alb*` order each pass.
  Undo: `albert live resume`, then `sudo systemctl start albert-mentions@prod`. The paper bot is not
  affected (it sends no orders).
- **Guard** `albert-guard@prod` (`deploy/guard-prod.conf`, `albert.live_guard`), every 60 s:
  - appends every account fill from `GET /portfolio/fills` to `data/live/audit/prod/fills.jsonl`.
    REST fills carry no client_order_id, so each order is looked up once (`order_tags.jsonl`) to name
    its bot.
  - marks every bot fill: settled markets at the settlement value, open ones at the mid (two-sided
    book) or else the last price.
  - **day loss**: bot-book equity below its 00:00 UTC level by more than $15 writes
    `data/live/HALT.json` until the next 00:00 UTC. New orders are refused, bot orders are cancelled,
    and the mentions bot closes open events and keeps running. The halt lapses on its own.
  - **at risk**: resting bot collateral + cost of unsettled bot fills above $60 turns on KILL, which
    only a person removes. Mentions caps itself at $40, so crossing $60 means a bug.
  - its client is cancel-only and cannot create orders. Status: `albert live check --env prod`.
- **Order client** (`venues/kalshi_trade.py`): per-order cap of 5 contracts and $2; prod orders must be
  post-only; refusals raise `Refused` (a `TradeError`, so bots carry on as on an exchange reject).
  Every write request, response, error and refusal goes to `data/live/audit/<env>/orders-<day>.jsonl`
  with actor, pid and latency, never headers. Summary: `albert live audit`.
- **Limits chosen**: the $15 day loss matches the user's "$15 VaR" / `--max-daily-loss 15`. It is a
  mark-to-market loss, so it binds after the bot's own fill-cost cap only when several bots run or
  something is wrong. $60 at risk is above mentions' $40 hard cap plus the stopped makers' $15.
- **Bot book since inception** (guard, 2026-11-19, everything settled):

  | Bot | Fills | Cost | Profit/loss |
  |---|---|---|---|
  | `albm` mentions | 25 | $11.18 | +$2.82 |
  | `albi` illiquid maker | 42 | $7.76 | −$1.70 |
  | `albl` liquid maker | 33 | $10.47 | −$3.47 |

- **Smoke on prod** (2026-11-19 21:18 UTC, nothing resting or scheduled):
  - The kill switch refused a create on a settled market before anything was sent; the refusal is in
    the audit.
  - The mentions bot logged `halt` "global kill" and exited; the guard status showed the kill.
  - `live resume` lifted it and the bot was restarted.
  - Fee re-check the same hour: 107 series fee changes, latest 2026-09-03, none on a mention series.
- **Not built**: a guard heartbeat check in the bots (if the guard dies, bots keep their own limits
  and trade on; systemd restarts the guard).

**EXPECTED-FILL BUDGET, 2026-11-19** (user: "take the expected number of fills and limiting that to
$15 instead. Going a bit over is acceptable if more get filled" and "Have it dynamically manage s.t.
if stuff overfills it can cancel orders; we want as close to the limit deployed as possible"):
both bots now run `--max-expected 15 --max-collateral 40 --max-events 6`. Postings keep the
expected fill cost, max(today's, unsettled fill cost) + each resting bid's fill probability ×
price, within $15. When fills push it over, resting bids are cancelled, largest expected cost
first (ledger `rebalance` / `cancel` with reason "expected budget"). Realized fill cost therefore
passes $15 only by fills between two 60 s passes; $40 is the hard cap on resting bids + unsettled
fills (the worst case if every bid fills and loses). The fill model (`live_mentions.FILL_W`,
`FILL_CURVE`) is a logistic in phase, price, price² and log(spread), with fills front-loaded in the
window. It was fitted on dev for the study and refitted on all replayed bids for the bot.
Evidence (`research/mentions_budget`; the 12 h cell's hourly replay, $100-bankroll ledgers):

| Oct 1 – Nov 16 | $/week | Sharpe | Sharpe ex-top-5 | Max DD | Peak realized | Peak resting |
|---|---|---|---|---|---|---|
| Old rule ($15 collateral, 2 events) | 7.12 | 8.21 | 5.54 | −7.03 | 14.14 | 15.0 |
| Expected $15, 6 events, $40 cap (live) | 9.82 | 9.66 | 8.38 | −6.52 | 15.55 | 38.0 |
| Expected $15, 6 events, no cancels | 9.24 | 9.06 | 7.58 | −8.84 | 18.13 | 43.5 |

The dev pick was 3 events ($72.05 vs $68.71 for 6 on dev; holdout $63.84 vs $67.20). 6 was
deployed because the user asked for the budget, not the event count, to be the binding limit.
Top-ups (re-posting legs cut by the budget when room frees) added nothing and are not built. The
model under-predicted holdout fills by 9% (1,115 vs 1,022), which the cancel rule absorbs.
The same day the bot logs a `skip` (reason "max events") once per event and phase when the event
cap blocks a posting. Before this, DealBook's Mamdani and Bieber sessions were dropped silently
behind Vance and Carney. Correction: `mentions_ext.capped_book` counted out-of-range legs (NaN
price) as resting, which disabled the room test. Fixed, the 12 h cell's holdout $15 book is
$7.39/wk (Sharpe 8.84), not $8.13 (9.15). Fee re-check 2026-11-19 04:55 UTC: no fee change after
2026-09-03, none on a mention series, Mentions series `quadratic` ×1.

**EXECUTIVES AT MOSTLY POLITICAL EVENTS, 2026-11-18** (user: "At a mostly political event like nyt
dealbook include business execs and public figures like musk"): the finance filter still rejects
executives' Mentions series (Musk, Altman, Nadella, Ellison, Luckey, Tenev, Dimon, …;
`categories._EXEC_MENTIONS`), but the bot now trades one when its milestone venue (the text after
the last " at " in the milestone title, e.g. "NYT DealBook Summit") also hosts, within a day, a
speaker the filter passes (`live_mentions.political_venue`; ledger record
`executive_at_political_event`). Company calls and events, Fed / Treasury / economic officials and
crypto or investment events stay out, and an executive alone at their own company's event
(Jensen at GTC) is still skipped. Limits unchanged ($15 VaR, 2 events), so at a ten-speaker summit
the bot still takes two events. Evidence: the frozen rule (calibration and margin chosen on
politicians) replayed on hourly candles for 27 earlier executive events (Mar 2025 – Sep 2026,
all venues, 99% candle coverage). The 6 h order made +16.6c/fill on 80 fills (t 2.5 by event and
by date); with the 12 h phase it made +10.1c on 103 fills (t 1.95; early +8.2c on 58). No
executive event falls after October 1. Last year's DealBook executives (Fink, Karp, Amodei,
Armstrong, 2025-12-03) lost 19c/fill on 6 fills, too few to read.

**12 H EARLY PHASE LIVE, 2026-11-18** (user: "Deploy commit and merge", after being offered the
promotion ahead of the 6-week paper test): `deploy/mentions-prod.conf` adds `--early-hours 12` to
the live bot, inside the same $15 value at risk, 2 events and 12c margin, and the CLI no longer
refuses `--early-hours` for `--env prod`. Pre-run fee re-check (2026-11-18 08:05 UTC):
`GET /series/fee_changes` has 107 rows, the latest scheduled 2026-09-03 and none on a mention series;
`/events/fee_changes` is empty; all 465 Mentions series are `quadratic` ×1 (no maker fee). The
restart waits until today's main-phase bids (Vance, Carney; expiring 12:50 UTC) are off the book,
because shutdown cancels every `albm` order. The paper bot below keeps running as a
simulated-vs-real check. Demote (drop `--early-hours`) if early-phase fills run below +5c over
~6 weeks.

**PAPER FORWARD TEST OF THE 12 H EARLY PHASE, STARTED 2026-11-18 05:12 UTC** (user: "Add it in
then"): `albert-mentions@paper` (simulated orders on prod markets, `albert.paper_trader`; no order can
leave the process) runs the live limits plus `--early-hours 12` (`deploy/mentions-paper.conf`): NO
bids from start − 12 h to start − 6 h, then the frozen 6 h order on legs the early phase did not
fill. Evidence: `research/mentions_ext` Q2 cell `L12h/m12c/extend`, frozen from dev; a one-off
confirmatory holdout run (2026-11-18) gave $15-book $8.13/week vs live $6.34, Sharpe 9.15 vs 8.36
(ex-top-5 6.96 vs 5.93), early-phase fills +7.27c on 433 fills. `--early-hours` is refused for
`--env prod` until promotion. Promote after ~6 weeks if early-phase fills earn ≥ +5c (t ≥ 2) and the
paper book beats live; score with `python -m albert paper mentions-report` (splits by phase). Stop:
`touch data/live/mentions/paper/STOP`.

**MENTIONS BOT ONLY, RESTARTED 2026-11-17 16:45 UTC at the user's request ("Start just the
mentions bot and run", "Do $15 VAR"):** `albert-mentions@prod`, margin 12c, max 2 events,
`--max-collateral 15 --max-daily-loss 15` (`deploy/mentions-prod.conf`). The collateral cap is now a
strict value at risk: resting NO bids + max(today's fill cost, cost of every unsettled bot fill)
≤ $15. Fills stay counted until their market shows no position (15 min grace), and a restart
rebuilds unsettled fills from earlier prod ledgers (`live_mentions.py`, tests in
`test_live_mentions.py`). Both makers stay stopped: `data/live/STOP` is in place (services
inactive but still enabled at boot, so the kill file is what keeps them off). Only
`data/live/mentions/STOP` was removed. Pre-run checks: `GET /series/fee_changes` shows nothing
after 2026-09-03 and nothing on any mention series; every Mentions series is `quadratic` (no maker
fee); kalshi.com's fee PDF is still behind a Vercel challenge (429). Account: $96.57 cash, no bot
positions, one manual order and position (`KXIMPEACH-29-JAN20`, blocked). The finance filter skips
executive and company mention series (Altman, Musk, Nadella, …); political speakers trade. Stop:
`touch data/live/mentions/STOP` or `sudo systemctl stop albert-mentions@prod`.
(Between 11-11 12:48 UTC and this restart every live service was stopped by kill files.)

**RESTARTED 2026-11-09 ~14:30 UTC at the user's request ("run a few strats with $5 each again",
including a mentions trial):** `albert-mentions@prod` ($5 collateral, $5 daily fill cost),
`albert-live-maker@liquid` (back to $5: 5 contracts, -$5 halt) and `albert-live-maker@illiquid`
($5). `KALSHI_LIVE_ENABLED=1`, kill files removed. Paper bots and the forward test stay off. Fee
re-check: no series fee changes scheduled or effective since 2026-11-01 (`GET /series/fee_changes`);
docs.kalshi.com fee rounding matches `fees.py`; kalshi.com's fee PDF still returns 429. Open
issue, not fixed: a false-halt race. When an order expires, the private feed reports the cancel
before `GET /portfolio/orders` drops it, and the reconcile grace covers only orders touched in
the last 5 s, so about 2% of reconciles see a phantom mismatch; two in a row halt a maker (seen
twice on 11-08, no halt). Proposed fix in `live_l2.py`: give feed-reported cancels the same grace. Account before restart: $103.72 cash,
7 one-contract bot positions from 11-08 still open.
Contract caps raised the same day because positions carried from 11-08 blocked quoting while
under the $5 cash cap: liquid `--max-gross` 5 → 10; illiquid 5 → 10 → 30 (cheap positions
filled 10 at $2.39 deployed). The $5 cash cap and the −$5 halt are unchanged for both.

**STOPPED 2026-11-08 05:11 UTC at the user's request ("kill all live and paper trades").** Every
trading service (live makers, mentions prod/paper/demo, paper makers, forward test) is stopped and
disabled; `KALSHI_LIVE_ENABLED=0` in `.env`; kill files left in place. No bot orders rest on the
exchange. The bots' 1-contract positions (13 markets) ride to settlement. The L2 recorder (public
data only) still runs.

**Live since 2026-11-08 00:11-00:25 UTC (user request: $5 of real money per strategy, sports
included, fees from Kalshi's per-series API data because kalshi.com returns 429 to fetches).**
`KALSHI_LIVE_ENABLED=1` is set in `.env`. Three systemd services on the primary account
(subaccounts need the Advanced API tier):

| Service | Tag / client ids | Budget |
| --- | --- | --- |
| `albert-mentions@prod` (`deploy/mentions-prod.conf`) | `albm` | $5 collateral, $5 daily fill cost |
| `albert-live-maker@illiquid` (`deploy/live-maker-illiquid.conf`) | `albi` | $5 deployed, 5 contracts, -$5 halt |
| `albert-live-maker@liquid` (`deploy/live-maker-liquid.conf`) | `albl` | $10 deployed, 10 contracts, -$10 halt (raised 2026-11-08 01:30) |

Makers: deployed = cost of own fills in unsettled markets (also from earlier runs of the tag,
rebuilt from `data/live/prod/<tag>/*/ledger.jsonl`) + open orders' collateral. Settled markets are
detected each reconcile (`GET /markets/{t}`) and leave gross/deployed/reconciliation. Markets
off shard 0 are blocked (cash is on shard 0; shard 3 = NBA rejects with
`insufficient_shard_balance`). Liquid and illiquid universes are disjoint by 24 h volume; Mentions
is excluded from both. A maker halted by a guard is not restarted by `deploy/live-maker-loop.sh`.
Stop: `touch data/live/STOP` (makers), `touch data/live/mentions/STOP` (mentions, also demo).

Status (2026-11-07 01:00 UTC): **demo plumbing passes 11/11** (resting order, queue position,
cancel, post-only reject, expiry, 1-contract fill, positions, order-group trigger, cancel-own sweep;
private WS frames parse, fixture `ws_private_demo.json`). The demo key is Ed25519 (supported).

**First live candidate changed to the pre-speech mentions NO-bid rule** (`live_mentions.py`,
`albert live mentions`; evidence: `research/mentions_history`, +5.8c/fill t 5.4 on 548 unseen
events at 8c, +7.9c t 6.4 at 12c). Demo: a forced event posted 3 NO bids in an order group with
expiration and cancelled them at shutdown; a 24 h scheduled demo run started 2026-11-07 00:57 UTC
(`data/live/mentions/demo-run.log`, stop with `touch data/live/mentions/STOP`). The forward paper
test (`research/forward`, mentions + FCS) runs hourly. Prod remains gated
(`KALSHI_LIVE_ENABLED` not set). Proposed prod pilot: 1 contract per leg, margin 12c, collateral cap
$15, daily fill-cost cap $10, max 2 events at once, Trump/Vance/other speech series (no earnings
calls), supervised, after the user enables it.

Built (commit 8437d8a), see "Software to build" for the design:

- `venues/kalshi_trade.py`: demo/prod scoped client. Write allowlist: single create/cancel and
  order groups only (Cancel All and batch cancel unreachable); `cancel_own(prefix)`; client-side
  Write bucket (100 tokens/s, create 10, cancel 2, verified with `GET /account/endpoint_costs`).
- `live_l2.py`: `LiveBroker` (Simulator interface, own orders netted from the book),
  `run_actions` worker, `reconcile`, `LiveMaker` (feeds, guards, ledger, queue-position
  sampling), `plumbing` (demo-only end-to-end test that saves real responses for fixtures).
- CLI: `albert live check` (read-only; works on prod now), `albert live cancel-own`,
  `albert live plumbing` (demo), `albert live l2 --env demo|prod [--live]`.
- Tests: parsers on recorded read-only portfolio payloads; fake exchange for the broker. The
  private WebSocket shapes (`user_order`, `fill`) are built from the recorded REST objects and
  must be checked against the demo recordings.

Next, once demo keys exist: `albert live check --env demo`, `albert live plumbing`, trim its
recordings into fixtures and fix any parser drift, then a short `albert live l2 --env demo`.

Decisions taken (user, 2026-11-05): build the order layer now, test it on Kalshi demo, then run a
supervised 1-contract live pilot (about 20 markets, $10 daily loss cap) after about a week of
paper data. Kalshi only. Stay on the Mac for the supervised pilot (VM deferred). In-play sports
stay in for paper; decide separately for live. Polymarket US on hold.

## Account facts (checked 2026-11-05, read-only calls)

- **API tier: Advanced since 2026-11-09 ~15:47 UTC** (user request; self-serve
  `POST /account/api_usage_level/upgrade` with an empty JSON body; permanent grant). Write
  and read buckets refill 300 tokens/s with 900 capacity (3 s burst); Basic was 100 write /
  200 read. Subaccounts (`POST /portfolio/subaccounts`) need Advanced, so they are now available.
- The key in `.env` (`KALSHI_API_KEY_ID` / `KALSHI_PRIVATE_KEY`) is a **production** key:
  `GET /portfolio/balance` returns 200 on prod and 401 `NOT_FOUND` on demo. Kalshi does not share
  credentials between environments, so demo needs its own account and key.
- Prod balance: $95.54 cash, all on shard 0 (shards 1-3 are $0; NBA/MLB/tennis need shard 3).
- **Pre-existing manual position and order: `KXIMPEACH-29-JAN20`** (67.22 contracts, about $49
  exposure) with a resting order at 0.92 for 67.22. The bot must never cancel or trade anything
  it did not create: never call Cancel All Orders (it cancels across every shard and subaccount),
  only cancel orders whose `client_order_id` carries the bot's prefix or that sit in the bot's
  order group, and exclude markets where we hold a manual position.
- Consider a dedicated subaccount (1-63) for the bot so positions and orders are separated by the
  exchange itself.

## What the user needs to do

1. Create a demo account at demo.kalshi.co and an API key there.
2. Add to `.env`: `KALSHI_DEMO_API_KEY_ID` and `KALSHI_DEMO_PRIVATE_KEY` (same formats as prod).
3. Before the live pilot: confirm the prod key has trading permission, choose the money at risk
   and the loss caps, and decide on in-play sports for live.

## Kalshi API facts for the order layer (docs.kalshi.com, fetched 2026-11-05)

- Environments: prod REST `https://api.elections.kalshi.com/trade-api/v2` (or
  `external-api.kalshi.com`), WS `wss://api.elections.kalshi.com/trade-api/ws/v2`; demo REST
  `https://demo-api.kalshi.co/trade-api/v2`, WS `wss://demo-api.kalshi.co/trade-api/ws/v2`.
- **Create Order V2**: `POST /portfolio/events/orders`. Required: `ticker`, `side` (`bid` = buy
  YES at `price`; `ask` = sell YES, equivalent to buying NO at 1 - price), `count` (string, 0.01
  granularity), `price` (string dollars), `time_in_force` (`good_till_canceled` |
  `immediate_or_cancel` | `fill_or_kill`), `self_trade_prevention_type` (`taker_at_cross` |
  `maker`). Optional: `client_order_id`, `post_only`, `expiration_time` (Unix seconds, with
  good_till_canceled), `cancel_order_on_pause`, `reduce_only` (IOC only), `subaccount`,
  `order_group_id`, `exchange_index` (-1 = auto-route by ticker). Response 201:
  `order_id`, `client_order_id`, `fill_count`, `remaining_count`, `ts_ms`. The legacy
  `/portfolio/orders` endpoint is being deprecated.
- **Cancel Order V2**: `DELETE /portfolio/events/orders/{order_id}?market_ticker=...` (auto-route);
  returns `order_id`, `client_order_id`, `reduced_by`, `ts_ms`. Batch cancel exists; batches cost
  the same tokens per item.
- **Order groups**: `POST /portfolio/order_groups/create` with `contracts_limit` (matched
  contracts per rolling 15 s). When hit, every order in the group is cancelled and new orders are
  blocked until reset. Also delete (cancels all its orders), trigger, reset, update limit.
- **Private WS channels**: `user_orders` (order created/updated: `order_id`, `ticker`, `status`
  resting/canceled/executed, `book_side`, `yes_price_dollars`, `fill_count_fp`,
  `remaining_count_fp`, `maker_fees_dollars`, `client_order_id`, `order_group_id`), `fill`
  (`trade_id`, `order_id`, `client_order_id`, `market_ticker`, `is_taker`, `yes_price_dollars`,
  `count_fp`, `fee_cost`, `post_position_fp`, `purchased_side`, `ts_ms`), `market_positions`,
  `order_group_updates`.
- Queue position: `GET` queue positions for all resting orders (for measuring fill-model error).
- **Rate limits**: token buckets, most requests cost 10 tokens; separate Read and Write buckets;
  batch cancel is 2 tokens per order. Basic-tier Write bucket holds one second of budget (no
  burst). `GET /portfolio/balance` has its own bucket, once per second on Basic. 429 on limit, no
  Retry-After: back off. 200 concurrent WebSocket connections on Basic. An Advanced usage level
  can be granted via `POST` upgrade once one of the last 100 orders came from the API.
- Kalshi notes that REST state lags the exchange slightly; combine order responses with the WS
  channels for the current state.

## Software to build

1. `venues/kalshi_trade.py`: environment-scoped client (`demo` | `prod`), create (post-only, GTC
   with `expiration_time`, `cancel_order_on_pause`, bot `client_order_id` prefix, order group),
   cancel, list own resting orders, positions, balance, order-group management. Prod requires
   an explicit enable (a `.env` flag plus a CLI flag) on every run.
2. **Live exchange adapter** with the Simulator's interface (`place`, `cancel`, `live_orders`,
   `position`, `fills`, `markets`, `ext_at`), so `Quoter` runs unchanged. It must **subtract our
   own resting orders from the book the strategy sees**. Otherwise improve logic treats our quote
   as the external best and bids against itself.
3. **Shadow simulation**: the paper Simulator runs on the same feed. Log predicted versus actual
   fills, plus queue positions, to measure the fill model's error. Built so far: exchange queue
   position of every resting order vs the size the net book shows at its price (ledger `queue`
   records, every 30 s). Per-order shadow fills are still to do; until then compare live fills
   with the concurrent paper run.
4. **Guards**:
   - per-order, per-market and gross-contract caps, plus a daily loss cap;
   - kill switch file;
   - on startup and shutdown, cancel the bot's own orders only;
   - cancel all of the bot's orders on a feed gap, disconnect, stale book or exception;
   - reconcile positions and balance each minute and halt on any mismatch;
   - exchange-side backstops: an order group contracts limit, `expiration_time` on every order
     (dead-man's switch while on the Mac), and `cancel_order_on_pause`.
5. **Demo plumbing test**: place, cancel, partial fill, expiry, order-group trigger, reconnect, and
   the kill switch. Demo does not behave like the real market; it proves plumbing only.

## Rollout

1. Demo plumbing test passes.
2. Live, 1 contract, about 20 markets, $10 daily loss cap, supervised, after about a week of
   paper data.
3. After 1-2 weeks, widen if shadow fills match real fills and PnL tracks paper.
4. Scale only on the go/no-go: event-clustered t ≥ 2 and ≥ 60% positive weeks, on settlement.
