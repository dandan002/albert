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
- **Rerun 04–12 UTC (in-play sports included, Polymarket ext events merged):** improve@5c +2.95c/contract
  marked to mid (t 3.7, 311 fills, $6.90); settled so far 34 fills / 8 markets +1.5c (27 of them
  sports, +$0.23).
- **Lead-lag** (`albert research lead-lag`): first paired hour (11 UTC, pre-game morning) has 16
  Polymarket moves ≥2c and Kalshi barely followed (+0.1c); needs evening game hours.
- **Live paper maker** (`albert paper l2`, `src/albert/paper_l2.py`, loop `deploy/paper-loop.sh`,
  running since 2026-11-05 13:06 UTC): the same Quoter and Simulator on live WebSocket books (no
  orders). A 500 ms holdback lets trades label their deltas online (`OnlineClassifier`, same
  labels as the batch matcher in tests). Each run records its own input, so
  `albert backtest l2 --root data/paper_l2/<run>/stream` replays it (smoke run reconciled: 67/67
  orders). Guards: per-market cap 3, gross cap 30 contracts, $5 loss halt, kill switch
  `touch data/paper_l2/STOP`. Score with `albert paper l2-report`.
- **Next:** accumulate ≥4 weeks; add a rate-limit model to the simulator; implement the ranked
  strategies (reward-aware quoting, quote-pull overlay with a PM-intl lead signal, weather
  overnight, pre-game sports). PM-intl ↔ Kalshi matching is needed for the lead signal (record PM
  tokens matched to recorded Kalshi markets).
- **Exit-price markouts (2026-11-05 15:40 UTC).** `mx_*` marks each fill to the best bid on its
  side (what it could be sold for at once); `mo_*` marks to the mid. Improve@5c, cooldown 300 s:
  - Recorded 04–15 UTC: 359 fills / 326 contracts / 70 events. mo_300s +1.7c (t 4.1 by event),
    mx_300s −4.0c (t −7.7), marked to last mid / settlement +2.4c (t 4.5), $7.44.
  - Live paper run 13:06–15 (its own recording replayed): 40 fills, mo_300s +3.2c, mx_300s −2.9c.
  - Paper ledger itself (2.5 h): 35 fills / 21 events, markouts +3.2 to +3.7c (t 3.2 to 3.7).
  - Settled so far: 25.6 contracts / 11 events, −1.8c per contract (−$0.45), where the same fills
    marked +1.1c at 5 min. At ±50c of settlement noise per contract this is uninformative.
  - Reading: the edge is real against the mid, but smaller than half the spread. It pays only if
    the mid is a fair price (earlier calibration work: roughly yes) and positions are held to
    settlement or exited passively. Crossing the spread to exit loses about 4c. **Settlement PnL
    over hundreds of contracts decides;** daily weather (KXHIGH/KXLOWT) settles fastest.
    Concentration: KXTRUMPSAY is 28% of contracts.
- **Fees:** the account's balance precision is $0.0001 (direct member), so real fees are about
  exact (maker 0.0175·P·(1−P) on maker-fee series only). The simulator rounds every fill up to
  1c, which is pessimistic (a 0.01-contract partial fill pays 1c in the sim).
- **Order layer built** (commits 8437d8a, 0a6a5f7; see `.planning/LIVE-PLAN.md`); waiting for
  demo keys.
- **Strategy sweep, four parallel studies (2026-11-05 ~16–17 UTC).** Code: `src/albert/research/
  {momentum_l2,momentum_hist,maker_variants,weather_nowcast}.py` (no CLI hooks yet).
  - *Short-horizon momentum / order flow (L2, dev 03–10, holdout 10–16 UTC):* mid jumps ≥2c in
    30 s revert ~2c (t −6 both windows) but only in wide markets (zero at spread ≤3c): quote
    flicker, not tradable. Same-side trade runs continue +0.5–0.9c; book/OFI imbalance ~0. Every
    taker version loses (−4 to −23c after spread and fees). As Quoter overlays: all within ±0.4c.
  - *Long-horizon momentum (38.8k settled markets, Aug 15–Nov 1, volume-blind, holdout from
    Oct 1):* 6h time-series momentum is real gross (+1.2/+1.4c, t ~2.8 dev/holdout) but −1.9c net
    as a taker (−0.8c at spread ≤2c). 1h/24h/72h, reversal, cross-sectional within event, drift
    into close (favourite-longshot, +1.5c gross), PM→Kalshi: all negative net. Caught artefacts:
    PM-vs-Kalshi gap +25c from resolved/placeholder PM prices vs stale Kalshi quotes; candles at or
    after actual close.
  - *Weather nowcast:* the earlier model loses to the market (Brier); settlement matches NWS CLI on
    21,255/21,259 legs; observation-decided brackets (5,783, 99.96% correct) are repriced within
    0–3 min by faster bots: no window on public hourly data. The Quoter already avoids them.
  - *Maker variants (dev 04–10, holdout 10–15):* nothing gets the exit markout to ≥0 out of sample.
    Only holdout-confirmed lever: require ≥20 contracts behind our quote ("behind20"), mx −3.7 →
    −2.7c, about half the fills, lower event concentration. Inventory skew, two-tick improve,
    toxicity filters, per-market history, price band, excluding KXTRUMPSAY: no gain. 6h momentum
    tilt +0.2–0.5c on mx (noise). micro-price 3c + behind20 overfit (dev mx −0.5, holdout −3.4).
  - **Net:** no taker edge anywhere; the maker's spread capture remains the only candidate and is
    decided by settlement PnL. Candidate change for paper: behind20 (+ optional mom3).
- **Intra-minute sweep (2026-11-05 ~17–18 UTC; 04–16 UTC data, all pre-game for sports).**
  Measured latency from the Mac: Kalshi event→send ~6 ms, send→our receive ~20 ms (Mac clock
  44 ± 18 ms behind; raw recv_ns precede ts_ms), REST RTT ~25 ms keep-alive → ~50 ms event to
  order at the exchange (the simulator's 100 ms default is conservative).
  From the AWS us-east-2 host on Kalshi's direct hosts (2026-11-07): WS send→receive ~1.4 ms,
  signed REST ~7 ms keep-alive, so event→order at the exchange is ~15 ms (`deploy/aws.md`).
  - *Structural arbitrage* (`research/arb_l2.py`): 95 complete mutually-exclusive sets, 929
    threshold-ladder pairs. Zero set arbs after fees (best gross gap +2c vs ~3.5c of fees on a
    50/50 pair). One ladder dislocation (KXTOKENUSE, +5.9c/lot, 88 lots) lasted 7 ms and was
    taken; at 25–500 ms only one leg fills (−3.9c legging loss).
  - *Fast lead-lag* (`research/leadlag_fast.py`): PM intl ↔ Kalshi cross-correlation ≤0.013 at
    any lag ±30 s; 50 PM moves ≥1c, Kalshi follows +0.18c; stale-quote taker −4.6c; team A/B
    sibling books requote together (median 55 ms), taker −4.4c. **In-play untested** — rerun:
    `fast_lead_lag(settings(), start='2026-11-05T23', end='2026-11-06T06')`.
  - *Intra-minute maker* (`research/intraminute_maker.py`): flicker provision, refill, scalps
    and leaning all negative on holdout (−1.3 to −3c at 10–60 s); fills are adversely selected
    (we fill exactly when the snap-back fails). Flat across 25–250 ms latency. Quote lifetime:
    improved quotes are alone at the front for a median 3.2 s; 40% of fills come after 5 min;
    short cancel timers would discard fills.
  - **Net:** no intra-minute edge at our speed in pre-game data. Remaining open question:
    in-play sports (tonight), via the lead-lag and arb reruns.
- **Research swarm: volatile data, curve fitting, market-making equations (2026-11-05 ~19–22 UTC).**
  Shared volatile-window definition `research/volatile.py`: 5-min buckets, mid path at spread ≤6c in
  the top 20% or in-play; consecutive-bucket correlation 0.54, so strategies condition on
  `volatile_prev`. Recorded hours have no in-play sports (1,445 volatile buckets: KXTRUMPSAY,
  KXRAIN, NBA pre-game, weather). Ten studies, all `research/*.py` with tests; reports under
  `data/reports/research-*`. Baseline Quoter mx_300 −3.8c dev / −3.3c holdout.
  - *AS / GLFT / logit-AS* (`mm_models.py`): worse than the Quoter on dev (mx −5 to −6c), mixed on
    holdout. Fill intensity depends on distance from the best, not the mid; the models quote too
    far inside wide spreads, and their inventory skew worsens mx. A "model gate" result is a
    price-band selection effect (near 0/1).
  - *Fair value* (`fair_value.py`): ridge/Kalman predict the mid out of sample (R² 0.1–0.3) only in
    ≥5c books (flicker snap-back); micro-price adds nothing (≤0.22c). Taker −0.8 to −5.9c; maker
    overlays ±0.2c except Kalman-centred quotes in volatile windows (+0.4–1.1c, ~60 fills, n.s.).
  - *Hawkes / VPIN* (`hawkes.py`): strong clustering, but no pull/widen filter improves holdout
    mx; the toxicity signal exists only at spreads ≤2c where the Quoter does not trade. Bursts
    predict move size, not direction.
  - *Glosten–Milgrom / PIN / Das* (`info_mm.py`): PIN-based market selection fails holdout. **Das
    Bayesian quoting improves mx by ~1c on dev and holdout at 50 and 100 ms** (−2.9 / −2.1c vs
    −3.8 / −3.3c); still negative. Candidate with behind20.
  - *In-play dynamics* (`inplay_dynamics.py`, 581 games fetched from public trades; the trades
    table is pre-game only): σ_logit ∝ (T−t)^−0.46, total variance matches p(1−p); jumps carry
    2–48% of variance by sport; under-reaction +0.6c at 60 s (t 5) but taker round trip ~3c; no
    in-play favourite-longshot bias; every rule negative on October holdout.
  - *Cross-strike curves* (`strike_curves.py`): residuals 3–10c are fit error, no convergence;
    taker −7 to −10c; historical KXHIGH −0.4c.
  - *YES over-buy* (`yes_bias.py`): buying YES passively is ~3c worse than selling it (holds), but
    sell-YES alone ~0 at settlement and −2 to −3c at exit; Bartlett & O'Hara single-name surplus
    reversed in October. Dropping the Quoter's YES bid: holdout mx −3.3 → −2.4c.
  - *Volatile re-test of every earlier strategy* (`vol_retest.py`): no sign change; volatile
    windows look better only through tighter spreads; the maker's mid edge is in calm windows; no
    volatility gate.
  - *Literature* (`.planning/research/MM-EQUATIONS.md`): equations, parameter recipes, ranked
    specs (event-level order-group kill switch, jump-hazard gate).
  - **Net:** no edge anywhere. The three ~+1c mx improvements (behind20, Das quoting, no YES bid)
    are the only levers; next test their combination, then rerun everything on tonight's in-play
    hours (each module's docstring/report gives the call with `start='2026-11-05T23'`).
- **Fundamental models, round 1 (2026-11-05 ~22–23 UTC).** Shared harness `research/fundamental.py`
  (quote = last hourly candle closed by t; blend `logit P = a + b·logit(mid) + c·logit(model)` with
  c's t by event; taker at ask/bid with fee; dev < Oct 1 ≤ holdout). Eight models of the underlying
  events; none pays as a taker on October holdout:
  - *Soccer* (`soccer.py`): Dixon–Coles loses to Kalshi (log loss 0.984 vs 0.952); de-vigged
    Polymarket 1X2 equals Kalshi (≤1c on 92% of legs); ties fairly priced. football-data.co.uk is
    blocked for AI agents by robots.txt, not used.
  - *College football* (`ncaaf.py`): no information in FBS; **FCS spreads: model adds information
    on both periods (c 0.67, t 3.8 dev; 1.00, t 5.0 Oct refit)** but taker +4.7c t 1.5 (5–14c
    spreads). Kalshi implied margins ≈ DraftKings close.
  - *MLB sabermetrics* (`mlb.py`): no information in moneyline/totals/F5/run line; inning totals
    c 0.63 only because books are 20–30c wide. Model saved for spring.
  - *Hourly temperature* (`hourly_temp.py`): markets open 60 min before the hour; settle on a
    5-station Kalshi weather index; market beats the physics nowcast at every lead (taker −3.4 to
    −4.8c).
  - *Rain* (`rain.py`): **adds information on both periods (c 0.61 t 8.1; Oct refit 0.39 t 2.7)**
    but taker +0.4c in October; Aug–Sep YES was overpriced, October calibrated.
  - *Mentions* (`mentions.py`): base rates lose; transcript rates c ≈ 0.45 (t 1.4–1.7, n.s.);
    sub-5c YES slightly overpriced but NO nets −0.65c after cent rounding.
  - *Approval* (`approval.py`): dev profit was overpriced early-September ladders; market sees the
    live RCP page; holdout taker −4.7c.
  - *Culture/tech* (`culture.py`): most series lack as-of public data; Vercel share "information"
    came from >25c spreads; Rotten Tomatoes 25 events, one film drives the holdout.
  - **Pattern:** a real model adds information only in thin, wide-spread markets (FCS spreads,
    rain), and that information sits inside the spread. Next: maker quoting around a model+price
    blend with trade-print fills, and more thin-market sports (college basketball).
- **Fundamental models, round 2: maker quoting around model+price blends (2026-11-06 ~00–02 UTC).**
  Harness gained series fee multipliers and a trade-through maker test (bids at fair ± margin,
  filled only when a later print trades through; held to settlement). The shared public rate
  limit saturated (~10 req/min), so most studies used hourly-candle trade ranges as the fill proxy,
  validated against real prints (97–99.6% agreement).
  - *Spread-width check (decisive):* model "information" mostly lives in books > 15c wide, where
    the mid is not a price — thin sports (volleyball, USL, college hockey, Argentina B, English NL:
    c ≈ 0 in ≤ 15c books), college basketball (same; wide books vanish by February), hourly temps.
    **Survives in priced books: FCS spreads (6–15c books: c ≈ 1.0, t 3–4 both periods) and rain
    (≤ 3c books: c 0.59 t 5.9 dev, 0.36 t 2.1 Oct; morning decision only).**
  - *FCS maker* (`ncaaf_maker.py`): −2.3c/fill Oct (1,585 fills), model-favoured bids are the ones
    traded through (adverse selection eats the information); books drying up (median spread 4c →
    17c → 23–26c). Forward snapshot of this weekend: `python -m albert.research.ncaaf_maker
    snapshot` (re-run Fri/Sat mornings), `... score` after settlement.
  - *Weather maker* (`weather_maker.py`): rain +1.2c/fill Oct (t 1.1, 489 fills, ~$1.3/week);
    ≤ 3c books +1.5c vs +0.4c around the plain mid; Aug–Sep profit was YES overpricing (rolling
    recalibration removes it). Hourly temps −1.3c.
  - *College basketball* (`cbb.py`, 2025-26 season): no information in priced books; holdout taker
    −8.2c, maker +2.3c t 0.6.
  - *Reference-price maker* (`refprice_maker.py`): quotes anchored to Polymarket lose in October
    (−8 to −11c/fill); Polymarket moves first (85–95% of fills come after a 1c adverse move);
    pull rule helps 0.3–0.4c.
  - *Mentions maker* (`mentions_maker.py`): **wide NO bids before Trump/Vance speeches earned in
    October** (pre-specified 8c margin: +4.8c/fill t 2.8 at 6 h; 12c: +8.4c t 3.4), but quoting
    around the YES-bias-corrected mid alone earns the same: a price pattern (selling YES into
    pre-speech spikes), not transcript skill. Best forward-paper candidate. Factba.se pages live
    on rollcall.com, whose robots.txt blocks the `anthropic-ai` crawler: all Factba.se data
    (cached pages, transcripts, word counts, derived predictions, test fixtures) and the fetcher
    were deleted on 2026-11-06 at the user's request. The transcript results above stand as
    recorded but cannot be rerun without a permitted transcript source.
  - *Series age* (`series_age.py`): no age effect; "young series overprice YES" is an empty-book
    artefact (spreads > 30c).
  - **Net:** information from fundamentals exists in a few thin markets (FCS spreads, rain) but
    passive fills are adversely selected enough to consume it; no strategy is significant on
    holdout except the mentions NO-bid price pattern.
- **Round 3 and data-source policy (2026-11-06 ~02–05 UTC).**
  - **Policy (user decision):** use a data host only if its *own* robots.txt allows us, and never
    fetch blocked content through another host. Removed with code, caches and fixtures:
    rollcall.com/Factba.se (blocks AI crawlers), api.weather.gov and Open-Meteo (`Disallow: /`),
    nflverse release files fetched via api.github.com (github.com blocks the download path for AI
    agents; the NFL study went with it). Kept: Kalshi, Polymarket APIs, ESPN API host (no
    robots.txt; www.espn.com blocks AI crawlers), MLB Stats API, IEM. The rain model is being
    rebuilt on IEM inputs only.
  - **Mentions NO-bid pattern is stable** (`mentions_history.py`): the frozen round-2 rule (NO bid
    at bias-corrected mid + 8c/12c from 6 h before the speech to start − 10 min, trade-through
    fills on real prints, circuit breaker) on 548 events round 2 never saw (Apr 2025 – Aug 16
    2026): **+5.8c/fill (t 5.4) at 8c, +7.9c (t 6.4) at 12c**; 15/18 (16/18) months positive, no
    month significantly negative, no decay (−1.3c/yr, t −0.6); Trump +5.7c, Vance +8.6c, other
    speakers +6.1c; earnings calls flat (exclude). Holds in ≤ 5c books (+3.0c, t 2.2). Plain-mid
    control earns the same and taking NO at t loses: it is selling YES into pre-speech price
    spikes, not information. Median 295 contracts trade through our price per fill (capacity is
    collateral-bound, not liquidity-bound). Mention series have no maker fee. Unknowns: real queue
    position (the rule assumes we are never ahead), start-time edits in old milestones.
  - *NFL EPA model* (removed with nflverse): no information in ≤ 5c books (2026 ladders ~1c wide);
    2025 signal came from 5–34c books.
  - *NBA props* (`nba_props.py`): c ≈ 0 in ≤ 5c books; no version pays on holdout or 2026-27
    smoke test; no pre-tip injury data.
  - *Forward paper test* (`forward.py`, `deploy/forward-loop.sh`, running for mentions + FCS):
    records the frozen rules' hypothetical bids each hour; `python3 -m albert.research.forward
    score all`.
- **Paper maker: settlement, cooldown and inventory skew (2026-11-07 ~19–23 UTC).**
  - The gross cap counted settled markets, so each 24 h run stopped quoting after ~1 h
    (fixed de25b48: held markets polled every 5 min, settled ones leave gross).
  - Only 8–14% of paper contracts were in YES+NO pairs (a pair pays exactly $1); pairs earned
    +2.8c/pair liquid, +16c illiquid, the unpaired rest was noise.
  - Follow-on fills (replay without cooldown): not worse than first fills, except same side
    within 60 s (−1.5c at 300 s). The market-wide 300 s cooldown mostly blocked pairing fills.
  - Replays of the 02:26–16:56 liquid paper stream (no gross cap, unsettled marked at mid):
    cooldown 300 s both sides: 51% paired, $1.20; none: 68%, $6.15; 5 s one-sided: 66%, $4.72.
    Skew 1c / 2c at 5 s: $2.95 / $1.70, per-event std $0.52 → $0.44 / $0.33 (worse PnL per unit
    of std). Illiquid 3 h: 5 s ≈ no cooldown ($7.66); skew roughly neutral per unit of std.
  - Decision (user): cooldown capped at 5 s, one-sided (8de5bd7). Skew off; `--skew` stays for
    replays. One day of data; settled contracts per config ≈ 100 liquid, ≈ 30 illiquid.
- **In-play rerun of the fast, risk-neutral studies (2026-11-08 ~06–08 UTC).** Window 2026-11-07
  17 UTC → 11-08 06 UTC (Saturday CFB, NBA, NHL, CBB; 466 markets, 166 paired). Runner and outputs
  in the session scratchpad (`inplay_hft.py`); reports `research-lead-lag-fast-inplay-20261108T073825Z`.
  The stock loaders (`read_stream`, `arb_l2.iter_stream`, `backtest.l2.iter_events`) hold a whole
  hour in memory (>4 GB on in-play hours); the runner streams line by line and filters markets
  first (checked event-for-event against the originals). Each recorded hour opens with fresh
  snapshots, so hour-aligned chunks are exact.
  - *Structural arbitrage:* 223 in-play episodes with ≥1 contract on every leg (24 games, ~10×
    pre-game), median life 5–7 ms, p90 13–19 ms, 72% taken by a trade. At 10 ms both legs fill on
    36 (+0.45c), one leg on 167 (−2.9c after unwind): −2.1c per attempt; −2.4c at 25 ms. Episodes
    lasting seconds to minutes (max "edge" 63c) are float dust in `arb_l2.Books` (levels of
    ~4e-9 contracts left after deltas sum to zero), not liquidity.
  - *Polymarket → Kalshi:* 1 s mid changes correlate 0.41–0.65 at lag 0 in play (≤0.013 pre-game);
    Kalshi leading by 1 s 0.17–0.29, Polymarket leading 0.04–0.08. Kalshi follows a ≥1c PM move by
    +0.29c at 60 s (2,630 moves, 32 games); stale Kalshi quote taken at 50 ms: −3.1 to −3.4c
    (t −16 to −23); every edge/move/exit cell negative.
  - *Sibling (team A book moves → take team B's stale quote):* responds in ~0.4–0.6 s, follow
    +0.6 to +1.9c; taker −2.4 to −3.1c at edge ≥0. Best cell (move ≥3c, edge ≥2c) +0.3 to +1.1c,
    t ≤0.7, 41 trades: noise (it was +2.8c before the 02–04 UTC hours were added).
  - *Maker defence:* Kalshi trades preceded by a PM move with the taker (5 s) mark +0.19c for the
    maker at 1 s vs +0.53c otherwise; still positive.
  - **Net:** no fast, risk-neutral edge in play either. Arbitrage appears ~10× as often but lives
    ~5 ms and is taken by faster traders; at our ~10–15 ms only one leg usually fills.
- **Parlay (multivariate "combo") pricing (2026-11-09).** Public data only: settled combos, their
  trades, and each leg's last complete 1-min candle before the trade (≤15 min old). Runner
  `combos.py` in the session scratchpad; report `research-combos-20261109T153822Z`. Samples: NBA
  single-game 60 games (of 340, Jan–Oct 2026; 2,703 combos), NFL single-game 5 playoff games
  (the 250-page listing cap reached only Jan 13–19), multi-game 2,000, recent cross-category
  2,000 (Aug–Nov 2026); legs must all be Sports series. Makers pay no fee (quadratic series).
  - *Markup over independence:* takers pay the product of the legs' mids plus 0.6–0.9c per
    contract (t 2.5–5.3 by game), in every group.
  - *Maker PnL vs outcome (per contract, count-weighted, t by game):* NBA +2.0c (t 0.75;
    pre-game +5.9c t 2.0, in play −3.2c t −0.6); cross-category +0.7c (t 0.6); multi-game +0.6c
    (t 1.3); NFL +3.3c on 5 games (unreliable). Per-trade cross-category +3.3c (t 3.1) does not
    survive contract weighting. Combo prices predict no better than the legs' product (Brier
    equal), so correlation pricing adds little.
  - **Net:** no robust overpricing to harvest. Hedging the legs costs ~half their summed spread
    (~5c) against a <1c markup; quoting needs the RFQ flow, and combos sit on shard 1 (cash on
    shard 0). The ~7c "edge" in the first preseason look was noise.
