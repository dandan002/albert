# Maker / microstructure research: synthesis (2026-11-04)

Sources: `market-making.md`, `price-discovery-momentum.md`, `exchange-infra-latency.md`,
`data-and-backtesting.md` (all cited; claims there are tagged verified / secondary / inference).

## Bottom line

- Makers as a class out-earn takers on Kalshi (~+1.1%/trade, widest in thin categories), but a
  small new maker does not inherit that average. Our −1.7% maker test failed on adverse selection:
  it bid the favourite one tick above the best bid and rested through news. Its trade-through fill
  rule likely also overstated the loss.
- For a binary contract, total remaining variance is fixed at p(1−p); what varies is *when* it
  arrives. The main lever is **when to pull quotes** (game start, lineups, observations, speeches),
  not how wide to quote.
- Speed is not our edge (people watching the event live or on official feeds are seconds ahead). A
  Python asyncio bot in AWS us-east-2 is fast enough. The hard requirement is streaming (WebSocket),
  not the 60s poll.
- Order-book history cannot be bought for our venues before 2026, and Polymarket US has nothing
  affordable. **We must record our own L2 now**; every day not recorded is lost.
- Liquidity-reward programs (Kalshi LIP until 2027-01-01; Polymarket US liquidity programs, no
  application) may matter more than spread capture at $100 scale. Polymarket US maker rebates need
  over $10M of volume and do not apply to us.

## Facts that change the code

| Item | Action |
|---|---|
| Kalshi production in AWS us-east-2, demo in us-west-2; PM-US exchange in us-east-1 (retail API behind Cloudflare) | Deploy in us-east-2 (GCP us-east4 is an acceptable fallback) |
| Kalshi supports Ed25519 keys (about 20× faster signing than RSA) | Switch signing |
| Kalshi rate tiers: Basic 10 orders/s or 50 cancels/s; next tier available by one API call | Request the upgrade before live |
| Kalshi shards: NBA/MLB/tennis need funds moved to shard 3 | Account plumbing |
| Kalshi `/markets` cached up to 15s | Price from orderbook / WebSocket only |
| Kalshi sub-penny prices and fractional contracts (listed 2026-11-27 / 2026-12-17) | Parser update |
| Kalshi order groups (auto-cancel when fills exceed a limit within 15s), PM-US expiring orders (good-till-date) | Use as kill switch / dead-man's switch |
| `fees.py`: PM-US effective date is 2026-10-28, not 10-19; tiers switch to a rolling 30-day window on 11-06; `"maker_fee" in fee_type` also matches `combo_maker_fees` (possible over-charge); per-order rounding still disputed | Fix and re-verify against official pages (kalshi.com rate-limited the agents) |
| Kalshi weather settlement source may have moved from NWS to The Weather Company (~Aug 2026); sources conflict | Read each market's rules before weather work |

## Candidate strategies (merged ranking)

1. **Reward-aware two-sided quoting in slow markets.** Join the best price rather than beating it,
   in markets that pay liquidity rewards (PM-US futures, Kalshi LIP markets). PnL = spread +
   rewards − markouts.
2. **Defensive quote-pulling as an overlay** for any maker strategy: cancel when Polymarket intl
   moves, a game event arrives (Kalshi `live_data`), or order flow turns one-sided. Test it as an
   A/B against the same quotes without the overlay.
3. **Weather brackets overnight.** Quote around the forecast (NBM) and pull before the day's
   observations. Caveat: Kalshi weather is already more accurate than NBM, so make money on the
   spread, not on forecasting.
4. **Pre-game sports quotes anchored to the Polymarket intl price.** Cancel when the prices diverge
   or lineups are near.
5. **Taker: PM-US stale quotes after Kalshi/Polymarket intl moves** (weak prior: 16 bets, t = 1.4).
   Polymarket holds taker orders 1s on sports, so test only with honest latency.

Not to build: anything that watches the event in person, uses leaked or embargoed text, or races a
delayed broadcast.

## Pipeline

1. **Pilot capture (1 hour):** Kalshi orderbook_delta + trades, plus Polymarket intl and PM-US books
   for about 300 markets → `.jsonl.zst` hourly raw + manifest. Verify there are no sequence gaps
   and check the data volume (estimate 15–100 MB/day).
2. **Continuous recorder** on a us-east-2 VM: gap detection → snapshot resync, a 10-minute check
   of the rebuilt book against a fresh snapshot, a heartbeat.
3. **Simulator (build, borrow hftbacktest queue models):** L2 replay, price-time priority,
   cancellations ahead of us in the queue, a latency model, per-venue fees and rounding,
   settlement.
4. **Research:** 4 weeks of recorded data to develop on + 4 weeks of holdout; markouts at
   1s/10s/60s/5m; standard errors clustered by event/date.
5. **Forward paper** against the live book.
6. **Kalshi demo** for order plumbing only (it does not behave like the real market).
7. **1-contract live:** use Kalshi's queue-position endpoint and our own order ID on book deltas to
   measure the simulator's fill error. Go/no-go: event-clustered t ≥ 2 and ≥ 60% positive weeks.

Cost: about $15–80/month (VM + storage) plus a few dollars of live losses.

## Progress (2026-11-05)

- **Recorder built** (`albert record`, `src/albert/stream.py`; continuous via `deploy/record-loop.sh`,
  stop with `touch data/record.stop` or SIGTERM to the python process).
  - 1-hour pilot (300 Kalshi markets / 150 PM tokens): 0 seq gaps, 25/25 REST book checks matched.
  - Real volume, much higher than estimated because thin books churn: ~1,000 Kalshi deltas per
    market-hour. At zstd level 9: about 5–6 MB/h Kalshi (300 markets) and about 10 MB/h PM
    (60 tokens).
  - Hourly `get_snapshot`, so each hourly file replays on its own.
  - Running locally since 2026-11-05 04:20 UTC (Mac, `caffeinate`); move to a VM for durability.
- **Feed facts (verified on recorded data):** each resting order that trades emits its own
  negative delta with the trade's `ts_ms` and size. All 1,080 pilot trades match one delta;
  about 10% of trades arrive before their delta. So executions are separable from cancels, and
  order sizes at a level are visible after the snapshot (near-L3).
- **Simulator built** (`albert backtest l2`, `src/albert/backtest/l2.py`): FIFO chunk queue,
  latency, post-only, trade-through and crossing fills, risk-averse/optimistic cancel
  attribution, series maker fees, markouts, settlement PnL.
  - Pilot-hour join-best baseline: −2 to −3c per contract at 5 min; worse at 500 ms latency.
    One hour, not significant.
  - Its churn (about 28k orders/h) would exceed Kalshi Basic write limits, so real strategies
    must re-peg sparingly.
- **Maker-flow study (7 recorded hours, 2,218 executions, 2026-11-05 04–11 UTC):** executions that
  do *not* clear their price level (front-of-queue makers) earn; executions that clear the level
  (the only fills a back-of-queue 1-lot receives) lose 2–4c at 60s–30min when the spread is ≤4c
  (t ≈ −3 to −5, clustered by market). Joining the back of a tight queue is structurally adverse.
- **Quoter strategy** (`albert backtest l2`, default `--strategy quote`): one tick inside the best
  when the spread ≥ 5c (alone at the front of a new level), spread band, cooldown after fills,
  queue-depletion pull, 10 writes/s budget (Kalshi Basic).
  - Dev hours 04–08: improve@5c +1.6c/contract at 5 min, +2.4c marked to last mid (t ≈ 2.8);
    holdout 08–11 (overnight, few fills): same sign, not significant (+1.4c, t 1.4, 58 fills).
  - Latency 100 ms vs 500 ms makes no difference; joining instead of improving is ~0.
  - Settled so far: 10 markets / 29 fills, +$0.80. Caveats: mark-to-mid in wide markets
    overstates realizable value; other bots cannot react to our replayed quote (only live 1-lot
    can test that); some fills were in live NBA games (exclude in-play sports before live).
- **Paired recording** (`albert record --paired-days 3`, running since 11:28 UTC): Kalshi game
  moneylines + the matching Polymarket intl moneyline token, for the lead/pull-signal study.
- **Next:** accumulate ≥4 weeks; add a rate-limit model to the simulator; implement the ranked
  strategies (reward-aware quoting, quote-pull overlay with a PM-intl lead signal, weather
  overnight, pre-game sports). PM-intl ↔ Kalshi matching is needed for the lead signal (record PM
  tokens matched to recorded Kalshi markets).
