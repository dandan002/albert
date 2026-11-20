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
    rebuilt on IEM inputs only. Added 2026-11-20 (owner): `api.hyperliquid.xyz` (no robots.txt;
    `hyperliquid.xyz` allows all), data only.
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
  17 UTC → 11-08 06 UTC (Saturday CFB, NBA, NHL, CBB; 466 markets, 166 paired). Runner
  `src/albert/research/inplay_hft.py` (outputs in `data/research/inplay_hft/`); reports `research-lead-lag-fast-inplay-20261108T073825Z`.
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
  `src/albert/research/parlays.py` (`collect OUTDIR` / `report OUTDIR`); report `research-combos-20261109T153822Z`. Samples: NBA
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
- **"Buy $1 of everything above 80%" (2026-11-09).** Hourly candles (`kalshi_candles`, the
  top-volume markets per category, 45,784 with two-sided quotes, closes Aug 15–Oct 31): the first
  hour a side's mid is above the threshold (before close), buy $1 of it at the ask (taker fee
  exact to $0.0001), hold to settlement. Runner `src/albert/research/favourites.py`.
  - 80%: 27,325 markets / 15,316 events, $27,325 staked, **−$1,011 (−3.7%, t −16 by event)**; win
    88.5% at an average ask of 91.4c; fees $164 (0.6%). One bet per event −4.0%. Every ask band
    (−2.6 to −4.5%), category (weather −1.4% … science −11%) and month (−2.9 to −4.4%) negative.
  - Higher thresholds lose less but still lose: 85% −2.7%, 90% −1.7%, 95% −1.1% (t −6.6).
  - A "held < 1 h" slice (+1.5%, t 3.5) is look-ahead (actual close of early-closing markets);
    by hours to the *scheduled* expiration, known at entry, <1 h is −2.1% (t −2.0) and every
    bucket is negative. Favourites are overpriced by about 2–3c plus the taker spread.
- **In-play "vol ladder" and LMSR market making (2026-11-09).** Replay of both recorded in-play
  nights (Sat 11-07 17 → 11-08 06 UTC, Sun 11-08 17 → 11-09 06; 56 games, game-winner markets
  only while in play) through `backtest.l2.Simulator` (20 ms latency, risk-averse queue,
  post-only, write budget 10/s Basic or 30/s Advanced), positions held to settlement, exact maker
  fees (0.0175·P(1−P); every in-play game series has maker fees except NCAAWB). Runner
  `src/albert/research/vol_ladder.py` (`run START END TAG [ladder|lmsr|pyramid]`; streams events, shared top-of-book history, cached
  best price; checked fill-for-fill against the plain version); report
  `research-vol-ladder-inplay-20261109T175400Z`.
  - *Ladders* (several resting levels per side around a reservation price; offsets scaled by
    σ of recent mid changes; inventory skew; optional trend lean and jump guard; 1-s and
    sub-second (0.25 s samples, 10 s σ, 2 s trend, 1 s guard, 0.1 s requote) versions): gross
    −0.29 to −0.34c per contract (t −2.3 to −7.9), net −0.63 to −0.65c (t −4.5 to −13.7), every
    variant negative on both nights, 0–22% of games positive. Join-the-best 1 lot: −0.19c gross,
    −0.52c net. Sub-second adjustment, trend lean, jump guard and the Advanced write budget change
    nothing (all ≈ −0.33c gross). Mid 10 s after a fill −0.2 to −0.3c, exit 60 s −0.75 to −1.2c:
    fills are adversely selected. ~99.8% of contracts paired, so losses are adverse selection plus
    fees, not outcome risk. Ladders need ~$160–180 of resting collateral.
  - *LMSR* (binary LMSR cost curve, 3 one-contract units per side, Advanced budget): pure LMSR
    anchored at the first in-play mid (b=10) −0.12c gross (t −1.0), −0.46c net (t −3.8), pairs
    +$246 but inventory −$286 (it fills to the 10-contract cap per market as games move);
    re-centred on the live mid (b=10 / b=25) −0.19 / −0.17c gross, −0.46 / −0.48c net (t −9 /
    −11); on a 60-s EWMA −0.20c gross, −0.50c net. Negative on both nights.
  - **Net:** no in-play passive quoting design earns its spread at our latency; the maker fee
    alone (~0.33c average) exceeds any gross result.
  - *Pyramid ladders (same replay, both nights, 56 games):* 5 levels per side at 0.5/1.5/3/5/8 σ
    units with sizes 1/2/3/5/8 (small near the mid, larger toward the extremes), max 20 net per
    market, inventory skew on price and size; spaced in cents or in log-odds (σ on logit mid, so
    the ladder compresses near 0/1); sizes × clip(√(60-s / 15-min traded rate), 0.5, 2); a
    sub-second Advanced-budget version; and a far-only pyramid (3/5/8/12 σ, sizes 2/4/8/16).
    Gross −0.29 to −0.30c per contract (t −4.2 to −5.9), net −0.61 to −0.62c (t −8.3 to −11.3),
    2–7% of games positive; far-only −0.18c gross (t −0.7), −0.53c net (t −1.9). Every variant
    negative on both nights; peak resting collateral $440–610. Log-odds spacing and volume
    scaling change nothing; the larger far orders fill into moves that continue (exit 60 s −1.0c).
- **Black-Scholes / SABR-style model of prediction-market prices (2026-11-09, academic; no edge sought).**
  Full write-up: `.planning/research/PM-VOL-MODEL.md`. Code in `src/albert/research/`:
  `pm_vol.py` (single-contract dynamics, hourly candles plus in-game L2),
  `pm_smile.py` (strike ladders as digital strips) and `pm_hazard.py` (deadline contracts as
  hazard curves). Reports:
  - `research-pm-vol-20261109T215533Z`
  - `research-pm-vol-hf-20261109T221912Z`
  - `research-pm-smile-20261109T220959Z`
  - `research-pm-hazard-20261109T222119Z`

  Findings:
  - **Martingale.** Prices are martingales with total variance p₀(1−p₀).
  - **Information clock.** It plays the role of implied vol: uniform in-game (Stern), back-loaded
    for weather, flat pre-game, a hazard rising into the deadline for "by date T" contracts.
  - **Jumps.** 50–85% of variance arrives in jumps, so a Gaussian probit diffusion fits the clock
    but not the shape.
  - **Backbone.** β ≈ 0.3–0.8, Jacobi-like; logit-normal is rejected.
  - **Smile.** Ladders move sticky-moneyness.
  - **Stochastic vol.** Half-life 7–21 h within market (weather about 40 h).
  - **Microstructure.** The mid is sticky with no bid-ask bounce. The probit-drift shortfall is
    explained by the tick floor outside sports and by jumps in sports.
  - **Tails and time are overpriced.**
    - Implied tails 1.2–1.4 against ≈ 1.05 realized.
    - Deadline hazards at k 0.6–0.8 of realized.
    - Theta decays at only 0.3–0.7 of the model rate.
    - Variance premium in-game 0.65–0.92.

    All are capped by the 1c tick, and none survives fees out of sample.
- **Vol trading strategies from the model (2026-11-10/11; backtest only, no orders).** Two studies
  test the "tails and time are overpriced" finding as trades, with resting (maker) orders and
  multi-leg structures that the earlier taker notes did not try. Every strategy is now also
  measured by **Sharpe ratio and volatility** with `src/albert/research/vol_metrics.py`:
  - Each study writes a per-leg ledger from a capped $100 book (≤ $5 per event, ≤ $20 open).
  - Positions are marked daily at the hourly-candle mid.
  - Sharpe = mean / sd of daily MTM PnL × √365.
  - The table includes a 90% block-bootstrap CI (7-day blocks), Sortino, drawdown, skew, and the
    Sharpe without the top-5 events.

  Output: `data/research/vol_metrics/metrics.{csv,md}`.
  - **Deadline contracts** (`vol_theta.py`, outputs `data/research/vol_theta/`). Walk-forward by
    decision month, Feb–Nov 2026: each month is priced only from contracts settled before it. 2.78M
    public prints back the maker fills (fill only on a print strictly through our level).

    | Strategy (walk-forward unless noted) | $/wk | Ann. vol | Sharpe (90% CI) | Ex top-5 events | Max DD |
    |---|---|---|---|---|---|
    | S1 maker NO bid at hazard-fair + 12c, T−30d..T−24h | 3.71 | 64% | 3.0 [1.2, 5.1] | 1.7 | −$33 |
    | S2 short forward hazard (calendar, taker) | −0.11 | 48% | −0.1 | −2.6 | −$29 |
    | S3t final hours, taker NO | 0.46 | 30% | 0.8 [−0.9, 3.2] | 0.0 | −$30 |
    | S3m final hours, pegged maker NO | 0.61 | 29% | 1.1 [−0.9, 3.2] | 0.1 | −$25 |
    | S3m L12_gap0 (picked ex post) | 1.65 | 31% | 2.8 [1.1, 5.1] | 1.9 | −$14 |
    | S3t L12_all (picked ex post) | 1.54 | 34% | 2.4 [0.4, 4.3] | 1.6 | −$15 |

    Reading the table:
    - S1 clears the bar but is fragile. The top 5 of 392 events carry 52% of PnL. Per contract
      it is negative in Jul, Sep and Oct. Skew is −0.9 (a premium seller).
    - Its edge sits in Politics one-off events, a split found ex post.
    - Final-hours decay (S3) is robust as a family: every variant is positive and its Sharpe is
      2.1–2.8 at 3–12 h. But the pre-declared selection rule picked thin variants, so only a
      forward run can validate it.
    - Calendar spreads are dead: two spreads, two fees and about 20 days of lock-up eat the
      0.27-vs-0.21 forward premium.

    Forward paper candidates:
    - S1 on Politics one-offs, with the mentions ≥ 95c breaker.
    - S3 frozen as NO at T−12 h on mention-window and elimination contracts priced 2–50c.
  - **Strike ladders** (`vol_ladders.py`, outputs `data/research/vol_ladders/`). **Nothing is
    robust.**
    - *Weather short-vol.* Buying the body and selling the wings when the implied width exceeds
      a forecast width passes Aug–Oct only at 16:00 local time. That is when the near-certain
      bracket at about 95c is underpriced, earning +1.8c holdout (t 2.2). It loses in November
      (−7c), apparently a seasonal effect.
    - *Maker version.* Its ledger makes about $0/week once resting orders count against the
      cap.
    - *Weather median drift.* The drift flips sign in the holdout.
    - *Moneyline versus the spread ladder's S(0).* The ladder adds nothing (gap sd 1.2–2.2c).
    - *Totals "over" bias.* Worth at most 2.4c, below costs.
    - *Sharpe.* Every ladder Sharpe CI includes 0 (`data/research/vol_metrics/ladders/`).
  - **Capacity correction and scaling** (`vol_theta.py run scale`; outputs `scale.csv`,
    `ledgers_scale/`, metrics in `data/research/vol_metrics/scale/`).
    - **The rule.** Maker fills are now capped at the contracts that printed strictly through
      our level, from the first fill to the end of the order's hour.
    - **S1 drops** from Sharpe 3.0 / $3.71 per week to **Sharpe 1.7 [−0.3, 3.9] / $1.96 per
      week** at the $5/$20 caps; 10% of fills had at most 3 contracts of capacity. **S1 no longer
      clears the bar.**
    - **At $100 open per strategy:**
      - S1 makes $2.6 / $7.7 / −$0.4 per week at $5 / $10 / $25 per event, with drawdowns of
        $120–274. It does not scale.
      - S3 holds about a day and has median capacity of 180–300 contracts. It scales linearly in
        the per-event cap, and the $100 open cap never binds.
      - The ex-post S3m L12_gap0 makes $1.6 / $3.5 / $8.2 per week at Sharpe 2.8–3.2, with
        drawdowns of $13 / $24 / $61.
      - The walk-forward S3 makes $0.6–2.7 per week at Sharpe 0.9–1.2, with drawdowns up to
        $139–156.
    - Sharpe is invariant to scale until capacity binds; $/week and drawdown scale together.
      S3t taker fills are not depth-capped, because historical depth is unobservable.
- **Greek trades on strike ladders (2026-11-11; `vol_greeks.py`, outputs `data/research/vol_greeks/`,
  read `verdict.md`).** Delta-hedged positions use model greeks from the hourly-fitted ladder
  (F, σ). Execution is taker at the candle bid/ask with fees, and each order is ≤ 25% of the leg's
  trailing-3 h volume. Dev runs to Sep 30, the holdout is October, and weather is also tested on
  Nov 1–9. **Nothing clears the bar, and hedging does not help.**
  - *Short strangles* (sell both wings) lose after costs in every group. Holdout: weather −1.9c,
    sports totals −2.2c, spreads −2.7c, team totals −2.9c per contract. In sports the wing
    premium is paid back almost exactly in tail payouts.
  - *Delta-hedging* earns +0.25 to +0.7c at mid on dev and about 0 on holdout. It costs 0.1–1.8c in
    spreads and fees and lowers Sharpe in every group: weather from −0.9 to −1.5, spreads from −3.3
    to −5.4. At $5 per event most hedges round to zero contracts.
  - *Vega* (long when the NBM or forecast σ ≥ 1.05–1.15 × σ_ATM). Sports flips sign in the
    holdout (t −2.1 to −2.5). Weather 16 h is positive in all three splits but has t 0.7 / 1.2 / 0.1
    and Sharpe ex top-5 −0.9.
  - *Gamma scalping*. Weather 12→8 h realized/implied variance is 1.36–1.57. The long strangle
    gains about +1c at mid, but taker round trips cost 2–3.5c, and two variants lost the whole
    $100 in dev.
  - *Capacity*: only 0–31% of sports entries can trade at all, because the thin wing leg's median
    volume is 0.
  - Forward-monitor ideas only: weather long vega at 16 h; weather 12→8 h long gamma as an
    unhedged maker (untested).
- **Structural arbitrage at the measured latency (2026-11-11; `inplay_hft arb START END OUT
  LATS`, outputs `data/research/arb_latency/`).**
  - **Latency.** Live timing records give event → engine p50 of 4.6 ms (p90 6.3 ms, 10.6k
    creates). The earlier replays started at 10 ms.
  - **Rerun.** Four recorded nights (11-07..11-10): 1,691 episodes, 1,578 of them in play.
  - **All episodes.** −2.2c per attempt at 2 ms, −2.5c at 5 ms, −3.0c at 10 ms. One-leg fills
    dominate: +0.9c per lot when both legs fill, −3.5c when only one does.
  - **Filtered.** With the outcome-blind filter "every leg shows ≥ 1 contract" (56% of
    episodes are sub-contract dust): −0.01c at 2 ms, −0.3c at 3 ms, −0.8c at 4 ms, −1.1c at
    5 ms. The total 1-lot opportunity is a few dollars over four nights.
  - **Verdict.** Faster execution narrows the loss but never makes it positive at our latency.
- **Favourite–longshot bias and retail flow: academic study (2026-11-11; paper `paper_flb/`,
  code `flb_{common,xs,flow,wallets,structural}.py`, outputs `data/research/flb_*/`).** This is a
  descriptive study, not a strategy test.
  - *Cross-section* (19,007 Kalshi markets, 4,770 events, 2021-07..2026-10, quotes at 1 h–30 d
    leads).
    - Calibration slope 1.13 (t 6.5 vs 1); volume-weighted 1.48.
    - Longshots (≤10c) return −55% per $ at the mid and −75% net of spread and taker fee.
      Favourites +1.1% at the mid, −0.7% net.
    - Present in every group and in both halves. Flat across liquidity terciles. Steepest in
      multi-outcome (1.28) and deadline (1.22) contracts. Does not fade at 1 h (1.18). Pre-game
      sports with an exact start are near calibrated.
  - *Kalshi trades* (2.69M prints, 18 strata, Aug–Oct 2026).
    - Small orders hold the longshots: 39% of contracts in ≤$10 taker orders are longshots,
      against 1.2% in >$1k orders (18/18 strata).
    - Taker longshot positions lose in 14/18 strata, and takers do worse than makers in 13/18.
      The immediacy cost (0.51c half-spread + 0.33c fee) is 14% of the stake.
    - No overall taker–maker gap, and no size class predicts outcomes.
    - Pooled per-$ longshot means are heavy-tailed: one MLB HR-prop event is 40% of the weighted
      profit. Use per-stratum estimates and sign tests.
  - *Polymarket wallets* (2.57M fills, 237,569 wallets).
    - Retail wallets make 86% of net longshot buying. No professional seller class.
    - The top 1% of wallets take 85% of gains, with no persistence (rank correlation 0.02).
    - Kalshi and Polymarket game prices match within 0.40c.
    - `gateway.polymarket.us` robots.txt disallows ClaudeBot/anthropic-ai/Claude-Web, so no
      Polymarket US prices were used. The older `albert ingest pmus` command uses that host;
      this is flagged to the user. *Re-read 2026-11-16:* the file is now Cloudflare's managed
      robots.txt: `User-agent: *` / `Allow: /` with `Content-Signal: search=yes, ai-train=no,
      use=reference`, plus `Disallow: /` for a "Training crawlers" list that includes ClaudeBot
      (anthropic-ai and Claude-Web are no longer listed; api.polymarket.us is the same). Whether a
      ClaudeBot block makes the host off-limits under the 2026-11-06 data-source policy (rollcall.com
      was dropped for blocking anthropic-ai) is **pending a user decision**; the 2026-11-16
      cross-venue study below used it.
  - *Structural.*
    - Market-weighted, the longshot gap is inside the taker no-arbitrage band; volume-weighted it
      exceeds the band by 0.79c (t 2.7).
    - The tick floor is ~half of the 1–2c loss.
    - LLO γ = 0.89, below 1 in every group; fitted on H1 it removes 61% of the H2 longshot gap.
    - Shin z is mostly the spread. Belief-dispersion proxies are not supported.
    - Verdict: frictions bound the bias; lottery demand by small traders is the best-supported
      source.
  - Fee note: Kalshi charges fees per fill to $0.000001 and rebates rounding over-payment
    (docs.kalshi.com/getting_started/fee_rounding), so exact fees are the fees paid.
- **Passive longshot seller on recorded L2 (2026-11-16; `research/longshot_maker.py`, outputs
  `data/research/longshot_maker/`, report `research-longshot-maker-20261116T190332Z`).** Does the
  FLB maker margin survive the queue? Rest a 1-lot favourite bid at 1 - q whenever a side's ask is
  q <= 10c (`join` the longshot ask, or `improve` it by 1c), hold to settlement, max 3 per market.
  - *Setup.* Stream days 11-12 00:00 to 11-16 14:00 UTC, both recorders. Replay `Simulator`, 20 ms
    latency, risk-averse queue, exact maker fees per `fee_type`. Only in-zone market-hours replayed.
    Ladders sampled at 10% of events, one leg per game.
  - *Result.* 22.5k fills, 1.4k events. `join` +0.13c per contract (t 0.3), `improve` -0.07c
    (t -0.1); upper 95% bounds about +1.0c, under the 2.2-2.4c maker bound. Fees rounded up per
    order make both negative (-0.27c / -0.49c). The 60 s mid moves -0.4c / -0.7c against the fill.
    Back-of-queue fills earn +0.67c; trade-through and cross fills lose 0.5-1.0c. No price band,
    family or phase is significant (best: halves +1.4-1.5c, t 1.2-1.4).
  - *Sharpe.* $100 capped book: Sharpe 7.9 / 7.1, but the 90% CIs span about -35 to +28 over
    5 days. Ex top-5 events: 3.5 / 4.4.
  - *Verdict.* No measurable edge after queue and adverse selection, so not a strategy as specified.
    Caveats: 5 days; ladders conflated to 1 s from 11-16 14:30 (outside the sample); resting-order
    collateral is ignored.
- **Polymarket US as a signal for Kalshi game moneylines (2026-11-16; `research/pmus_xvenue.py`,
  outputs `data/research/pmus_xvenue/`, report `research-pmus-xvenue-20261116T190743Z`).**
  Kalshi execution only; Polymarket US public gateway data (robots.txt question: see the
  2026-11-11 note above, pending a user decision).
  - *Data.* 770 Polymarket US moneylines (NBA, NHL, NFL, NCAAF, NCAAMB, NCAAWB) starting
    11-11T17..11-16T06; `/price-history` at `fidelity=1` returns one book snapshot per minute
    (ask = `longPrice`, bid = 1 − `shortPrice`) and reaches back ≥ 41 days (games of 2026-10-06
    still return full minute series). Matched 687 of the 690 recorded Kalshi games (99.6%); the
    winner agrees on 686 (one PM US market resolved at 0.14/0.86).
  - *Stamp.* A point stamped minute t is the book at **t + 60 s** (in play: mean |PM − Kalshi|
    0.37c at +60 s vs 1.59c at 0 s; minute-change correlation 0.97 vs 0.02). Using the point at t
    would be 60 s of look-ahead.
  - *Recorder lag (new, affects other studies).* Kalshi frames carry exchange timestamps; our
    receive time lags them by p50 6–12 ms but p99 0.5–8 s on 11-11..11-14 and **p90 6.6 s, p99
    24 s on 11-14/15 (Saturday; 20% of quote rows > 1 s late)**. Quote tables keyed on receive
    time (`pm_vol` HF quotes and anything else replaying `record-vol`) show Kalshi stale exactly
    when it is busy. This study rebuilt every leg's book on the exchange clock
    (`server_quotes`, 9.6M rows).
  - *Gaps.* Median |Kalshi − PM US| 0–0.5c, p90 ≤ 1.5c in every league, pre-game and in play.
    Buying both sides (Kalshi taker + PM US taker, Θ = 0.095) costs a median 3–5c over $1; < 0.3%
    of minute points dip below $1, on display prices with unknown size. No arbitrage even if PM
    US trading were allowed.
  - *Lead-lag (exchange clock).* In play, after a ≥ 1c PM US minute move Kalshi follows +0.15c at
    60 s and +0.22c at 120 s (t 5–6), like the PM intl result (+0.29c). Both venues close the
    gap: Kalshi 33% and PM US 27% within 60 s (t 11 and 9); neither leads. Pre-game Brier is
    identical (0.1674 vs 0.1674 at −15 min, t 0.1), so a PM US-anchored maker has nothing to
    anchor on (not simulated).
  - *Kalshi taker on divergence* (24 cells: pre-game / in play × edge 0/1/3c × exit 60 s/5 min/
    15 min/settle; one position per market; ask/bid at ≥ 1 contract, exact taker fee; displayed
    size caps the $5 stake): every exit-at-time cell negative or noise (−3.5% per $ at edge 0,
    t −17 in play; in play ≥ 3c: +7.6% t 1.3 on 38 trades, dev −4.0% on 11). The one t > 2 cell
    (in play ≥ 3c, hold to settlement, +49% t 2.3) is 27 binary bets with 81% of PnL in 5
    games (Sharpe ex-top-5 −7). Ledger Sharpe/vol are in the report; with 6 calendar days the
    bootstrap CI is not defined.
  - *Artifact caught.* On the receive clock the in-play ≥ 3c cells looked like +8.7% (t 3.9) to
    +18%, 81 of 106 trades on Saturday. Re-priced on the exchange-clock L2 book at the same
    instants, only 28–33% enter at the simulated price and the result goes to −1.7% … +5.7%
    (t ≤ 0.5): Kalshi had already moved; the recorder showed it late.
  - **Verdict:** Polymarket US is a second copy of the Kalshi price (agrees within ~0.4c,
    converges both ways within a minute, equally accurate). Nothing to trade on Kalshi from it at
    minute resolution; sub-second data would need a KYC'd API key for the markets WebSocket.
- **Naive technical-analysis factors and day-of-week effects (2026-11-17; `research/ta_calendar.py`,
  outputs `data/research/ta_calendar/`, run `python -m albert.research.ta_calendar all`).** Taker
  at the hourly candle's closing quote, ~$1 order (`n = floor(1/price)`), taker fee rounded up per
  order, held to settlement; dev = scheduled expiration before 10-01, holdout = October; t
  clustered by match, and calendar effects also by ET date (a weekday effect is a common shock).
  - *Data.* Volume-blind sports + weather candles (38,785 markets / 23,684 matches, closes
    Aug 15–Oct 31) on a regular hourly grid (Kalshi omits unchanged hours; quote carried forward);
    the `pm_hazard` deadline contracts (5,754 markets, 305 ET dates, Jan–Oct); the trade tape
    (7.3M trades / 14,760 markets, volume-selected: flagged).
  - *TA, 17 signals × trend/fade:* RSI 6/14 at 70/30 and 80/20 (needs ≥ 3c of movement in the
    window), EMA 3/12 and 6/24 crosses, Bollinger 12/24 at 2σ, Donchian 12/24/48 breakouts by 2c,
    MACD histogram cross, 6 h ≥ 5c moves split by volume (≥ 2× vs ≤ 0.5× trailing) and by OI
    change. |Gross| ≤ 1.7c in every rule against ~3.1c of taker cost; trend net −1.6 to −5.9c per
    contract (t −2 to −8 where n ≥ 1,000) and every fade also negative. Capped-book ledgers
    ($100, $1 per position, $3 per event): Sharpe −4.5 to −6.9 (CIs below 0), max DD −$146 to
    −$386 (ruin). The 6,044 volume-selected politics / entertainment / mentions markets look the
    same.
  - Only consistent cell: RSI trend 3–12 h before scheduled expiration, gross +2.3 / +4.0c
    (t 2.0 / 2.8) dev / holdout. This is the known near-event momentum (`momentum_hist` tsmom).
    Net −0.6 / +1.0c; ledger Sharpe −2.0 overall, +0.6 [−3.9, 4.0] holdout. One of ~270 cells.
  - *Calibration by ET weekday* (noon and 8 pm snapshots, y − mid): weekend − weekday +0.2c dev
    (t 0.6 by date), +1.1c holdout (t 2.3). Cells do not replicate (weather t 2.8 → 0; 12–48 h
    t 0.7 → 4.7). The favourite-longshot slope flips sign Aug–Sep vs October and is the same on
    weekends. Weekend favourites (buy YES 50–97c at noon ET): gross +4.7 / +4.0c vs weekday
    +1.9 / +0.1c, but ledger Sharpe 0.6 [−1.7, 2.5], holdout −0.3 (weekday control −3.9).
  - *Deadline-contract drift by weekday:* YES falls 0.7–0.9c/day Tue–Fri vs 0.1–0.7c Sat–Mon;
    weekend gap +0.2c/day (t 1.9; dev 1.3, holdout 1.5). A day on which the market closes must be
    valued at settlement: dropping it censors the early-YES jumps, which cluster on weekdays, and
    inflated the gap to t 4.1. Only Mentions replicates (+0.7 / +1.3c/day, t 2.2 / 2.7): speeches
    happen on weekdays, so the weekend hazard is lower and priced. Buying NO at noon ET every day,
    held to settlement, with lead-time and price fixed effects: weekend − weekday +0.3% per $ dev
    (t 0.2), +1.8% holdout (t 1.2). Ledgers: weekends-only Sharpe 2.2 [0.6, 4.1] ≈ weekdays-only
    2.3 [0.5, 4.0], both ~0.6 ex top-5, holdout 1.2 / 0.3. That is the `vol_theta` deadline
    theta; the weekday is not a lever on it.
  - *Taker tape:* taker PnL per $ by weekday has no pattern (Tue −16%, Wed −18%, other days +2 to
    +5%; 12 dates per weekday); the taker YES share is ~80% every day, weekends included.
  - **Verdict:** no naive TA or day-of-week rule pays the taker spread and fee; nothing to paper
    trade. Weekday effects are noise or rational (weekday-concentrated hazard). Ledgers are in
    `data/research/ta_calendar/ledgers/` (picked up by `vol_metrics run`).
- **Momentum outside sports and resolution spikes (2026-11-17; `research/xcat.py` sampler,
  `research/spikes.py` study, outputs `data/research/{xcat,spikes}/`, run
  `python -m albert.research.xcat sample|fetch hourly|fetch minute` then
  `python -m albert.research.spikes all`).** Taker at the candle's closing quote, fee rounded up per
  order, held to settlement; $1 orders for momentum, $5 for spike trades; dev = decision before
  10-01, holdout = October; t by event and by ET date.
  - *Data (volume-blind, seeded, nothing selected on volume or result).* Hourly candles over the
    whole life of every Politics / Science-Tech / Elections / Mentions market plus 450 random
    Entertainment events closing Aug 15–Oct 31 (23,506 markets, 2,180 events, 8.4M grid hours).
    Minute candles with full OHLC over each market's **first 3 days after open** (a window fixed
    by the open time, so no spike is chosen because the market then resolved) for random events:
    sports winners / spreads-totals / props, weather, mentions, entertainment, politics (16,183
    markets, 5.1M candles; the window holds the close for 99% of weather, 77–79% of game winners
    and props, 60% of spreads and mentions, 30–38% of politics / entertainment). API facts: the
    historical cutoff was 2026-09-18 (markets settled minutes after it are still live-only); a
    candle request may span at most 5,000 periods, a batch 10,000 summed over tickers; ~4 req/s
    sustained from one IP.
  - *H1 momentum (6 h ≥ 5c trend; also 1/24/72 h, 3/10c, fades, 1 h jumps).* Gross +1–2c
    overall, net −2 to −6c in both splits for every rule. Politics dev +4–6c gross (t 2.5–3.5)
    → holdout +0.6–2.4c; entertainment +3–4.5c gross in both splits but net ≈ 0; mentions moves
    reverse (buying 1 h jumps −8c net, t −6 / −7). Ledgers: Sharpe −2.3 overall, −0.4 to −4.4 by
    category (holdout −0.8 to −10.4).
  - *H2 spike taker (leader ≥ 90c after +10c in 60 min).* Net −1.9c dev / −1.5c holdout; ledger
    Sharpe −0.6 overall, −4.0 holdout. Across 78 variants (70/80/90/95c, 10/20c, 5/15/60 min,
    slow-approach control, fade, +1 min delay) gross stays within ±1c and every cell is net
    negative in both splits; the win-rate lower bound is under breakeven everywhere. First touch
    of 97c wins 97.7% vs 98.8% breakeven (−1.1c, t −3.2). The dev-only lead in game winners
    (26 of 26 cells +1.3 to +5.8c net) is negative in all 26 in October (−0.5 to −8.1c).
  - *H3 spike limit (leader bid 1/3/5c below, 60 min or to window end; laggard bids).* Adverse
    selection eats the discount: after a ≥ 90c spike the leader wins 93.6%, but a bid 3c below
    fills only on a dip and the filled ones win 88% vs 89% breakeven. Every holdout leader cell
    −0.6 to −3.9c; laggard bids −1.8 to +0.3c; ledger Sharpe −2.4 overall, −2.9 holdout. Same
    mechanism as the L2 `longshot_maker` result, now over 2.5 months and all categories.
  - *Screens.* Intraday momentum at 20–80c mids: trend −3.4 to −8.2c net; 15-min moves revert
    (fade +1.4 to +3.8c gross) but net −0.7 to −3.5c (quote flicker, as in the L2 studies). New
    listings (first quote in a market's first 6 h): −3 to −6c net either side; the consistent
    cells are longshots (−6 to −9c, the favourite-longshot bias, not a listing effect). Hour of
    day: YES overpriced ~1.5c dev / ~0.7c holdout at every ET hour, no hour stands out.
  - **Verdict:** public price paths (momentum at 15 min to 72 h, spikes into resolution, resting
    bids on dips, listing quotes, hour of day) carry no tradable information on Kalshi outside
    sports either; nothing to paper trade. Ledgers in `data/research/spikes/ledgers/`.
- **Three new directions, run in parallel (2026-11-17; reports in `data/research/{lip,pre_event,linked}/`).**
  - *Kalshi Liquidity Incentive Program* (`research/lip.py`): terms verified on help.kalshi.com and
    docs.kalshi.com (`get-incentives`; 1 s snapshots, both sides ≥ Target Size 1,000, DF 0.5, $1
    minimum payout per market program, ends 2027-01-01; ~$66k/day of non-financial pools on ~3,270
    markets). A $100 book can only reach the payout minimum on the 1–3c longshot side, where fills
    are informed (0.4% of filled contracts won). Replay of 10.7 recorded days, 1,209 program
    markets: −$10 to −$155/week across variants (rewards are an upper bound), Sharpe −0.6 to
    −12.5. Not worth a paper test.
  - *Pre-event NO bids beyond Mentions* (`research/pre_event.py`): the harness reproduces Mentions
    (October +9.1c/fill t 5.8, book Sharpe 6.6 [3.1, 11.1], 3.7 ex top-5) but no other scheduled
    event type passes the frozen rule: props +3.1 → −3.9c, game winners −4.5 → −2.0c, esports
    −3.8 → −3.1c (ruin), broadcasts +3.1 → +1.4c on 33 fills, earnings flat. A post-hoc "≤ 15c
    sports book" lead (+8–10c) fell to +2.4c (t 0.5) on a frozen November check. The pre-event
    YES premium is specific to speeches.
  - *Linked markets across events/series* (`research/linked.py`): ~406k links found by rules text
    (spread ⇒ winner, team ⇒ game total, player stats across series, protected/unprotected props,
    speech word ⇒ "says before date", rain day ⇒ rain weekend, Billboard duplicates). Prices
    agree almost everywhere (0.06% of quoted link-hours beat both legs' costs, 0.002% for sports
    games). A one-minute-delayed taker lock in the thin series makes +6.4c dev / +5.1c holdout per
    pair (t 3.2 / 2.4, ~4c locked floor), $1.7–2.7/week, Sharpe 3.8 / 2.6 but 0.9 / 0.3 ex top-5,
    ~30 trades per period; basis risk (same-text Trump series meaning different days; 1.7% of
    protected/unprotected pairs settle apart). Maker on the rich leg loses (−4 to −16c Oct).
    At most a live scanner for those few series later.
- **Four more directions, run in parallel (2026-11-17 evening; reports in
  `data/research/{mentions_ext,counts,recurring,xvenue_leadlag}/report.md`).**
  - *Mentions extensions* (`research/mentions_ext.py`): the live rule (6 h, 12c) holds: +8.6c/fill
    dev → +9.0c (t 4.6) Oct–Nov 16, book Sharpe 8.4 [6.1, 15.0], 5.9 ex top-5, $6.3/week at the
    $15 VaR. Wikimedia-pageview salience filter: trending words were the *best* fills (no filter
    passes dev); GDELT returned 429 on every call. Earlier/longer windows and other margins
    dilute; the dev-chosen "also trade late-listed events" cell fell +6.5 → +1.9c. Post-speech NO
    sweep loses 5–29c in all 32 cells (a ≥ 2c YES usually means the speech is still running).
    Polymarket mention matches: 37 fills, too thin. Open lead (dev only, not holdout-tested): an
    extra NO bid 12 h → 6 h before start at 12c (+31% $ on the $15 book in dev; 3,987 fills at
    +9.7c vs 3,053 at +8.8c over Mar 2025–Aug 2026). Paper test only.
  - *Accumulating-quantity ladders* (`research/counts.py`): monthly rain (IEM CLI, 997/997
    settlement match), weekly hot-day streaks, weekly executive orders (Federal Register lags the
    White House by ~4 days). Kalshi fully prices the published partial sum (asks $1.00 on 928/935
    passed strikes) and beats the nowcast where open. Frozen maker +4.9c dev → +1.6c (t 0.6) on 15
    events, no better than quoting the mid. TSA weekly is category Economics (excluded); Truth
    Social, x.com, spc.noaa.gov and congress.gov block us in robots.txt.
  - *Recurring-series persistence* (`research/recurring.py`): 143 series, 9.4k events (charts, AI
    leaderboards, post counts, approval, strait traffic). Incumbents repeat 44% after one win, 88%
    after five, and own history adds information only at open + 1 h inside 35–65c books. Frozen
    taker −0.8c dev → −4.3c holdout; walk-forward −0.7c with a −$190 drawdown; positive $ is one
    2c longshot. All 71 other cells lose.
  - *Polymarket intl leading Kalshi outside sports* (`research/xvenue_leadlag.py`): 2,415 exactly
    matched pairs (culture, Nov 3 midterms, mentions, world; 11 families rejected on rules
    differences). Moves are simultaneous at one minute; the "leader" gives back 30–40% of an
    unmatched move. Frozen taker −2.1c (t −4.0) dev → −2.0c (t −4.9) holdout; zero latency is no
    better; every one of 72 cells is negative in both periods.
  - **Net:** nothing new to trade; the mentions NO bid stays the only live strategy.
- **Behavioural mechanisms by market family, run in parallel (2026-11-18; reports in
  `data/research/behav_{culture,sports,politics,micro}/report.md`).** Each agent wrote ≥ 10
  psychology / market-structure reasons someone would trade at a bad price *before* opening data,
  screened them on dev, froze rules, then scored the Oct 1 – Nov 17 holdout once.
  - *Culture, entertainment, weather, science* (`research/behav_culture.py`, 19 hypotheses):
    fandom NO at 2–35c +2.8c (t 3.7) dev → +1.4c (t 1.1) holdout, book Sharpe 2.6 → 0.75 (−0.9 ex
    top-5); fails the |t| ≥ 2.5 Bonferroni bar. Attention buying runs backwards (heavy volume for
    the price = informed). Dread tails, round-10°F brackets, anchoring on yesterday's high, the
    headline-forecast bracket, pageview spikes and Musk/xAI tribalism all rejected. Post-hoc
    lead: fandom longshots at close − 24 h settled YES 0 vs 3.7 / 2.7 expected (paper only,
    ~$1–2/week, −$4.7 per miss against +$0.1–0.2 wins).
  - *Sports* (`research/behav_sports.py`, 19 hypotheses, 6,858 dev / 1,948 holdout games): popular
    teams, home teams, recency after blowouts, primetime flows, star props, late chasing,
    American tennis players, payday and spread key numbers all null or wrong-signed. 1 of 97 dev
    cells passed (chance level); all 3 frozen rules lose on the holdout (best t −0.31). Spreads 1c,
    depth ample: the books are made by pros anchored to the sportsbook consensus.
  - *Politics, elections, news* (`research/behav_politics.py`, 17 hypotheses): partisan bias
    follows each cycle's polling miss (Dems +8.5c rich 2024, −10c 2025, −4.3c 2026 midterms);
    drama premium is just the known deadline theta; status-quo YES is rich, not cheap; early
    election-night moves continue. Election-night leader 60–95c +7.8c (t 2.0) → +0.2c; day-after
    leader 80–98.5c +2.5c (t 5.9) → −3.1c. Post-hoc: 90–96c election-night leaders 53/53, but
    three nights and the 95% lower bound (93.2%) is below breakeven (94.1%).
  - *Cross-market microstructure* (`research/behav_micro.py`, 16 hypotheses): no round-number
    clustering, no fee-rounding avoidance, no lock-up premium (APY only above $250), Thursday
    pause reversal not executable, multi-outcome lag and post-settlement recycling null. Robust
    fact: single sweeps ≤ $50 are uninformed and revert, > $200 informed (dev, Oct and Nov). Stub
    15c past a small sweep +6.1 → +5.7c on prints (Sharpe 5.7), but the exact-book L2 replay is
    +7.4 → −3.5c (176 fills) and prints overstate it by 1–1.4c. Refill after a small sweep +7.5c
    both periods (Sharpe 6.5, 4.6 ex top-5) but 56% Mentions, exact-book ≈ 0 on 49 fills and the
    October holdout was partly seen. Forward paper candidate only; it needs a WebSocket bot on
    Mentions/non-sports books.
  - **Net:** no new live strategy. The mentions NO bid (plus its 12 h early phase, now on paper)
    remains the only edge. About 1,500 dev cells across the four studies; every confirmatory rule
    that looked good on dev shrank or flipped on the holdout except the microstructure refill,
    which is unconfirmed on the exact book.
  - *Mentions budget* (`research/mentions_budget.py`, 2026-11-19): capping the expected fill cost
    (fills held + Σ p(fill) × price over resting bids) at $15 instead of resting collateral, and
    cancelling resting bids when fills push it over, lifts the 12 h cell's holdout book from
    $7.12/wk (Sharpe 8.21) to $9.82/wk (Sharpe 9.66, ex-top-5 8.38) at 6 events with a $40 hard cap;
    realized fills peak at $15.55. Live on both bots since 2026-11-19 04:58 UTC. The research
    `capped_book` NaN-price bug is fixed (12 h cell holdout $8.13 → $7.39/wk).
- **Hyperliquid HIP-4 outcome markets: data collection started (2026-11-20; data only).**
  Hyperliquid's Terms (section 1.6) bar US persons, so it is never traded. Kalshi stays the only venue.
  - *Universe (2026-11-20).* 313 live outcomes, about 160 non-financial: European soccer match
    results (EPL, La Liga, Serie A, Bundesliga, Ligue 1), NBA/NFL/esports match winners, and
    EPL/UCL/NFL/NBA season winners. The rest are perp/crypto price binaries, IPO, Fed-rate and
    recurring crypto questions (excluded). No politics, mentions, weather or culture.
  - *Data.* `albert-record-hl` records every non-financial outcome's full book and trades
    (side 0; side 1 is the same book mirrored at 1 − p: 12 of 12 markets had identical trade
    hashes and sizes). Every trade carries buyer and seller addresses. `albert ingest hl` writes
    `hl_outcomes` and `hl_candles` (1 h back to listing, 1 m for the last 3.5 days).
  - *Planned check (after about 2 weeks of recording; frozen rules, dev/holdout):* (a) price
    agreement and lead-lag against Kalshi on matched soccer/NBA/NFL markets (prior: PM intl and
    PM US both agree with Kalshi within about 0.4c and neither leads); (b) whether identifiable
    Hyperliquid wallets' flow predicts Kalshi moves or outcomes (prior: PM wallets show no
    persistence, rank correlation 0.02). Any trade would be on Kalshi only and needs
    event-clustered t ≥ 2, ≥ 60% positive weeks, and Sharpe/vol from `vol_metrics`.
