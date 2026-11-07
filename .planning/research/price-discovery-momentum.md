# Price discovery, lead-lag, momentum and short-horizon strategies in prediction markets

Research note, 2026-11-04. Web research only; no code was run against Albert data.
Every factual claim has a URL. **[Fact]** marks something a cited source states. **[Inference]** marks
my own reasoning or extrapolation. **[Unverified]** marks a claim I saw only in a summary, snippet or
vendor/blog post that I could not check against the primary document.

---

## 0. Bottom line

1. **Who leads depends on the category and on where the volume is.** For 2024 election contracts,
   Polymarket led Kalshi. Across about 2,000 matched Kalshi/Polymarket pairs from 2023-26, Polymarket
   led on politics and entertainment and Kalshi led on sports, but only where Kalshi had more volume
   ([CRUCIAL summary of Ng, Peng, Tao & Zhou](https://crucialab.substack.com/p/prediction-markets-and-price-discovery);
   [SSRN 5331995](https://papers.ssrn.com/abstract=5331995)). Nobody has published a lead-lag study
   for the newer, thinner **Polymarket US** book. One open-source practitioner log found that "Kalshi
   moves first, and Polymarket US makes half of Kalshi's move within 15 minutes 74% of the time". The
   author marked that as **not proven** (16 graded bets, +0.95c, t = 1.4)
   ([line-reaper PR #282](https://github.com/lkshanahan565656/line-reaper-backend/pull/282)).
2. **Big venues absorb clean public news in seconds.** In-play prices "update swiftly and fully" after
   goals ([Croxson & Reade, EJ 2014](https://ideas.repec.org/p/bir/birmec/11-01.html)). The mispricings
   that remain after news are measured in **minutes** and are small: about 2.8% after Betfair commission
   when betting 2 minutes after a goal, falling to 0.8% at 6 minutes
   ([CBS on Choi & Hui 2014](https://www.cbsnews.com/news/what-sports-betting-can-teach-investors)).
   That is about the size of one Kalshi taker fee at mid prices.
3. **The short-horizon patterns that are documented mostly do not survive costs.** Momentum in betting
   lines is real but "fail[s] to overcome transactions costs"
   ([Moskowitz, JF 2021, abstract summary](https://econpapers.repec.org/RePEc:bla:jfinan:v:76:y:2021:i:6:p:3153-3209)).
   Albert's own re-tests agree (README §3a). The exceptions are **(a)** surprise-driven over-reaction in
   very liquid in-play soccer and **(b)** slow repricing in thin or rule-sensitive markets, such as the
   Kalshi Netflix "Warner Bros" mention market, where repricing took **40 minutes**
   ([Bartlett & O'Hara 2026](https://conference.nber.org/conf_papers/f243677.pdf)).
4. **Pure speed (latency arbitrage on live events) is a different business.** It needs sub-second
   engine-level feeds, and venues are actively adding taker delays. Polymarket has a 1-second matching
   delay on sports markets ([docs](https://docs.polymarket.com/concepts/order-lifecycle)). Crypto.com
   has a 3-second retail delay
   ([Yogonet](https://www.yogonet.com/international/news/2025/12/11/116729-cryptocom-implements-a-threesecond-delay-for-retail-users-in-its-sports-event-betting-market)).
   Kalshi filed for order delays ([Gambling911](https://www.gambling911.com/node/78397)). **[Inference]**
   A $100 account cannot win the race against firms buying ≤5 s esports feeds and official data. Albert
   *can* use the same public signals defensively, to avoid being picked off as a maker. It can also
   look for slow repricing in thin venues (Polymarket US, small Kalshi markets) where the lag is
   minutes, not milliseconds.

---

## 1. Cost hurdle (the bar every effect below must clear)

| Venue | Taker fee per contract | At 50c | At 80c | Maker | Source |
|---|---|---|---|---|---|
| Kalshi | 0.07·C·P·(1−P) (fee-schedule PDF writes round_up per order; Albert's `fees.py`, checked 2026-11-03 against docs.kalshi.com/getting_started/fee_rounding, says the fee is exact to $0.000001 with over-payment rebated) | 1.75c (2c worst case if rounded up on 1 lot) | 1.12c | 0, or 0.0175·C·P·(1−P) on listed series | [Kalshi fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf); [fee rounding](https://docs.kalshi.com/getting_started/fee_rounding); [secondary](https://www.predictionhunt.com/blog/kalshi-fees-complete-guide-2026) |
| Polymarket US | 0.0695·C·p·(1−p), banker's rounding to $0.01 | 1.74c | 1.11c | 0; rebates only from $10M/month maker volume | [docs.polymarket.us/fees](https://docs.polymarket.us/fees) (effective 2026-10-28) |

**[Fact]** Polymarket US moves fee tiers to a rolling 30-day window from 2026-11-06
([docs.polymarket.us/fees](https://docs.polymarket.us/fees)).

**[Inference]**
- A taker round trip at 50c costs about **3.5c plus two half-spreads**.
- A one-way taker trade held to settlement costs **1.75c plus a half-spread**.
- Any short-horizon signal that you exit before settlement therefore needs about 4-6c of expected
  move. A signal held to settlement needs about 2-3c of mispricing.
- If Kalshi rounds fees up per order (the conservative reading), 1-lot orders pay 2c at mid
  prices, and Albert's $100 bankroll means small orders. Keep the conservative model in backtests
  until a live fill confirms which rounding applies.

---

## 2. Q1: Price discovery. Which venue leads?

### 2.1 Prediction markets vs each other

| Study | Sample | Method | Finding | Status |
|---|---|---|---|---|
| Ng, Peng, Tao & Zhou, *Price Discovery and Trading in Modern Prediction Markets* (SSRN 5331995, rev. Apr 2026) | 2024 election contracts on Polymarket, Kalshi, PredictIt, Robinhood. Broader set: about 2,000 matched Kalshi/Polymarket pairs, Sep 2023 to Mar 2026 | Lead-lag; method not given in the summaries | Polymarket leads Kalshi, more so when liquidity is high. "Large-trade order imbalance strongly predicts subsequent returns". The venue with more large directional flow leads. Polymarket leads politics/entertainment; **Kalshi leads sports where it has higher volume** | [Fact] from abstract ([SSRN](https://papers.ssrn.com/abstract=5331995); [SZU seminar abstract](https://swift.szu.edu.cn/info/1051/5704.htm)). The category split is [Unverified]: it comes from a secondary summary ([CRUCIAL](https://crucialab.substack.com/p/prediction-markets-and-price-discovery)) |
| Aktuğ & Torul, *Price Discovery Across Political Prediction Markets* (WP Apr 2026) | 9 platforms, 5-minute probabilities, Jan to Nov 2024 | Hasbrouck IS, Gonzalo-Granger CS, Putniņš ILS | Search-engine summaries conflict: one says "Polymarket and Betfair jointly ≈85%", another says "Polymarket 44% / Kalshi 41%". Kalshi reached a non-trivial share within about 32 days of launch | **[Unverified]**: the PDF ([bogazici.edu.tr](https://web.bogazici.edu.tr/torul/pridis.pdf)) was unreachable. Do not quote the numbers until the PDF is read |
| Tsang & Yang, *Political Shocks and Price Discovery* (arXiv 2603.03152) | 3.65M Polymarket trades (Polygon), Jan to Nov 2024, 5-minute bins | Glosten-Harris, Kyle λ, variance ratios | Debate move (+11c) **largely reversed** (ended +2c). Assassination-attempt move (+11c) **persisted**. Dropout: little net move with two-sided flow | [Fact] ([arXiv](https://arxiv.org/html/2603.03152v2)) |
| Rothschild & Sethi (J. Prediction Markets 2016) | Intrade 2012 presidential market, 6,300 accounts | Trader-type taxonomy | Arbitrageurs held for a median under 10 minutes. A persistent Intrade/Betfair gap existed in one direction only (Romney overpriced on Intrade) | [Fact] from abstract/summaries ([SSRN](https://papers.ssrn.com/abstract=2322420)) |
| Practitioner log (line-reaper) | Sep 24 to Oct 14, 2026, 1-minute recorder | Move detection | Kalshi moves first; PM US closes half the gap within 15 minutes 74% of the time. Betting PM US at the stale price: +0.95c vs close, t = 1.4, n = 16. All Pinnacle-led and sportsbook-laggard pairs **lost** to their own close | [Fact] that the log says this ([PR #282](https://github.com/lkshanahan565656/line-reaper-backend/pull/282)). It is anecdotal evidence and not peer reviewed |

**[Fact]** Albert's own data: pre-game Kalshi vs Polymarket intl median |gap| is 0.5c, and esports
tracks Polymarket intl within about 1c at hourly resolution (README §4).

**[Inference]** At hourly resolution the venues are one market. Any lead-lag edge lives inside the
hour, and probably only around discrete news (scores, injuries, rule-relevant utterances). The best
candidate laggard is **Polymarket US**. It is new and thin, and the one available log (above) suggests
lags of minutes against Kalshi.

### 2.2 Betting exchanges vs bookmakers

- **[Fact]** A Euro 2008 in-play study found Betfair offered the best odds almost 90% of the time
  against two big bookmakers, with "tentative indications" that Betfair leads price discovery
  ([ResearchGate listing, *Exchange vs. Dealers*](https://www.researchgate.net/publication/228720836_Exchange_vs_Dealers_A_High-Frequency_Analysis_of_In-Play_Betting_Prices);
  full text not accessible).
- **[Fact]** Betfair prices forecast outcomes better than bookmaker odds
  ([Franck, Verbeek & Nüesch, IJF 2010](https://www.unifr.ch/tim/en/assets/public/uploads/Publication%20list/2010/IJoF_Franck_Verbeek_Nuesch.pdf)).
- **[Fact]** Using the consensus of many bookmakers' odds as "fair" value and betting the outliers was
  profitable over a 10-year simulation, a 6-month minute-by-minute simulation and 5 months of real
  money. Bookmakers then limited the authors' accounts
  ([Kaunitz, Zhong & Kreiner, arXiv 1710.02824](https://arxiv.org/abs/1710.02824)).
- **[Unverified]** On Pinnacle vs bet365, Pinnacle moved first in 60% of matched move pairs, with
  bet365 following a median 6.5 minutes later. This comes from a commercial blog with an unverified
  method ([pinnacleterminal.io](https://pinnacleterminal.io/blog/bet365-dropping-odds)).
- **[Unverified]** One vendor measured Kalshi's median winner-market margin at about 1.0% vs Pinnacle
  at about 4.0% on 88 fixtures in one day
  ([OddsPapi](https://oddspapi.io/blog/polymarket-api-kalshi-api-vs-sportsbooks-the-developers-guide/)).

**[Inference]**
- The Kaunitz-style "consensus vs outlier" logic is the template for "use the sharp venue to price the
  slow venue".
- Its success came from *soft bookmakers that re-price slowly*. Kalshi and PM intl are not that.
  Polymarket US may be.
- I found no peer-reviewed Betfair-vs-Pinnacle information-share paper. Treat the claim that
  "Pinnacle/Betfair lead" as practitioner consensus, not measured fact.

### 2.3 How fast prices converge after a scoring event

| Study | Venue / sample | Speed / pattern |
|---|---|---|
| Croxson & Reade (EJ 2014) | Betfair, soccer goals just before half-time | "prices update swiftly and fully" ([IDEAS abstract](https://ideas.repec.org/p/bir/birmec/11-01.html)) |
| Debnath, Pennock, Giles & Lawrence (ACM EC 2003) | World Sports Exchange, 34 World Cup 2002 games + 18 NBA playoff games, polled every 10 s | Average thresholded delay of **31.6 s** across 74 goals. This includes their data/website delays, so true market reaction was faster ([PDF](https://clgiles.ist.psu.edu/papers/ACM-2003-Sports-Betting.pdf)) |
| Gil & Levitt (J. Prediction Markets 2007) | Tradesports, 2002 World Cup | Prices jump at goals but **keep drifting for 10-15 minutes**. Trades were few. Market makers earned slightly negative returns ([IDEAS](https://ideas.repec.org/a/buc/jpredm/v1y2007i3p255-270.html); summary via search) |
| Choi & Hui (JEBO 2014) | Betfair, second-by-second, 2,017 matches | **Under-reaction** to ordinary goals, **over-reaction** to surprising (underdog) goals, corrected "within minutes". +2.8% after commission betting at +2 minutes, +0.8% at +6 minutes ([HKUST record](https://researchportal.hkust.edu.hk/en/publications/the-role-of-surprise-understanding-overreaction-and-underreaction/); [CBS](https://www.cbsnews.com/news/what-sports-betting-can-teach-investors)) |
| Angelini, De Angelis & Singleton (IJF 2022) | Betfair, 1,004 EPL matches | Reverse favourite-longshot bias. Prices respond correctly to expected goals but **over-react to surprises** (late underdog goals) ([Reading summary](https://research.reading.ac.uk/football-research/new-journal-article-informational-efficiency-and-behaviour-within-in-play-prediction-markets/); [paper PDF](https://cris.unibo.it/retrieve/8c5e3d3d-d36f-4279-b566-da91767d1a5c/information_efficiency_angelini_de_angelis_singleton.pdf)) |
| Easton & Uylangco (IJF 2010) | Betfair tennis, point by point | Very high correlation with a point-based model. Breaks are anticipated up to 4 points early, but the post-break loss tendency is "not captured instantaneously" ([RePEc](https://econpapers.repec.org/RePEc:eee:intfor:v:26:y::i:3:p:564-575)) |
| Winkelmann & Deutscher (arXiv 2505.21275) | Bundesliga 2018/19, bookmaker at 1 Hz | Odds and stakes **do not anticipate** goals. They respond to red cards and xG ([arXiv](https://arxiv.org/html/2505.21275v1)) |
| Cheng, Yang & Zou (arXiv 2605.00864) | Polymarket NBA, 173 games, Feb to Mar 2026, L1 polled every 3.6-5.5 s | Moneyline-vs-spread combinatorial dislocations: 290 executable episodes, median 101 bp, median **16 s** in-game, clustered in the final minutes. 76.9% of episodes could not absorb $100. Pre-fee: NBA fees were 0 then ([arXiv](https://arxiv.org/html/2605.00864v1)) |
| Moshrefi (arXiv 2607.14430) | 23M Kalshi moneyline trades | Calibration is near-perfect for most of a contract's life but becomes **step-like in the final 10 minutes**, which the author attributes to hedging of likely losers ([arXiv](https://arxiv.org/abs/2607.14430)). Related: Page (2012), InTrade's "Yogi Berra" bias, where losing teams were overpriced in the final 15 minutes (cited in [Bürgi, Deng & Whelan](https://www.karlwhelan.com/Papers/Kalshi.pdf)) |

**[Inference]**
- Liquid exchanges get the first-order jump right within seconds.
- The documented residuals are second-order:
  - under-reaction to *expected* news, over-reaction to *surprising* news, lasting minutes;
  - end-of-game distortions (final 10-15 minutes).
- On Betfair those residuals are a few percent, about the size of Kalshi taker costs. They are
  worth testing on Kalshi because Kalshi's in-game books are thinner and more retail than Betfair's,
  but the prior should be "small or nothing".

---

## 3. Q2: Momentum, reversal and drift at short horizons

| Effect | Evidence | Venue / period | Survives fees? |
|---|---|---|---|
| Momentum in betting lines (open to close), from delayed over-reaction | Moskowitz, JF 2021 ([RePEc](https://econpapers.repec.org/RePEc:bla:jfinan:v:76:y:2021:i:6:p:3153-3209); [Yale summary](https://insights.som.yale.edu/insights/what-sports-betting-teaches-us-about-financial-markets)) | US sportsbooks, multi-sport | **No**: "fail to overcome transactions costs". Albert re-test: nothing robust (README §3a #5) |
| Mean reversion (buy rolling low) | Quantpedia on Polymarket ([link](https://quantpedia.com/exploiting-mean-reversion-in-decentralized-prediction-markets-evidence-from-polymarket-binary-contracts/)) | Polymarket | **No** as a taker. Albert: −13 to −15% per $ (README §3a #4). The author says it needs maker fills |
| Post-news under- and over-reaction in-play | Choi & Hui 2014; Angelini et al. 2022; Gil & Levitt 2007 (table §2.3) | Betfair, Tradesports | **Marginal**: +2.8% after Betfair commission at 2 minutes. Betfair commission is charged on net market winnings, so it is not directly comparable with Kalshi's per-trade fee. **[Inference]** It probably does not clear a ~1.75c fee plus spread unless the effect is larger on Kalshi |
| Political shock transitory vs permanent | Tsang & Yang (arXiv 2603.03152) | Polymarket 2024 | Debate +11c → +2c (a reversal of about 9c, which would clear fees *ex post*). But n = 3 events, and classifying "transitory" vs "permanent" in advance is the hard part. **Not a strategy** |
| Order-flow imbalance predicts returns | Ng et al. (large-trade imbalance; [SSRN](https://papers.ssrn.com/abstract=5331995)); Bartlett & O'Hara (VPIN at 30-70c predicts maker losses on single-name Kalshi markets, t = −2.42; YES-directed high-VPIN buckets settle YES more often, t = 12.44; [NBER](https://conference.nber.org/conf_papers/f243677.pdf)) | Polymarket/Kalshi 2024-26 | **Unknown for takers.** **[Inference]** Its most usable form is *defensive*: a maker pulls quotes when one-sided flow appears. That needs no fee hurdle, only lower adverse selection |
| Weather: market leads official forecasts | Crosier (arXiv 2609.23969): Kalshi high-temperature market RMSE about 10% below NBM, 7 cities, 7,590 city-days, 2022 to Aug 2026. NBM moves about 4× more toward the market than the reverse. No jump at NBM publication ([abstract](https://arxiv.org/abs/2609.23969); [HTML](https://arxiv.org/html/2609.23969)) | Kalshi weather | Says nothing about profitability. **Implies the opposite of a forecast edge**: the market is ahead of the models. Consistent with Albert's NWS result (README/ROADMAP) |
| Favourite-longshot / YES optimism | Bürgi, Deng & Whelan ([PDF](https://www.karlwhelan.com/Papers/Kalshi.pdf)); Bartlett & O'Hara (single-name makers earn 1.91c/contract, broad-based 0.82c) | Kalshi 2021-26 | Maker-side only; takers lose. Albert: volume-blind maker test is −1.7% (ROADMAP) |
| Rule-literal repricing (mention markets) | Netflix "Warner Bros" market: the CEO said "Warner Brothers" about 3 minutes into the call. YES went from about 90c to below 40c over **40 minutes** as NO takers read the exact-phrase rule. Makers lost $85k ([Bartlett & O'Hara §3](https://conference.nber.org/conf_papers/f243677.pdf)) | Kalshi, Jan 2026 | **[Inference]** Yes, when it happens: a 50c move over 40 minutes dwarfs fees. Frequency is unknown (one anecdote) |
| Price dislocations ≥5pp with one-sided flow | Ibrahim & Zaki: 44,976 dislocations on Polymarket 2024-25. Citation rate rises 33% after them ([arXiv 2609.06005](https://arxiv.org/abs/2609.06005)) | Polymarket | Not tested for returns. It is a dataset definition that could be reused |

---

## 4. Q3: Live-event data feeds, latency, and the rules

### 4.1 Typical latencies (most figures are vendor or blog claims)

| Source | Claimed latency | Quality |
|---|---|---|
| Official league data (Genius, Sportradar) | "Sub-second" / 0.5-1 s; scouted data 2-3 s | [Unverified] blogs ([theplayoffs.news](https://theplayoffs.news/en/the-data-behind-in-play-in-the-uk-latency-feeds/)). Kalshi and Polymarket US both signed Genius official-data deals ([DeFi Rate](https://defirate.com/news/genius-sports-prediction-market-kalshi-polymarket-deals/)) |
| Stadium-scouted betting feed vs UK TV | Feed ≥4-6 s ahead of Freeview TV; HLS streams about 30 s behind live | **Measured**, one match, 2018 ([benjojo](https://blog.benjojo.co.uk/post/beating-the-broadcast-delay-world-cup)) |
| TV / OTT streaming | Satellite 7-15 s; OTT (YouTube TV, DAZN) up to about 60 s | [Unverified] industry blogs ([Dolby OptiView](https://optiview.dolby.com/resources/blog/sports/low-latency-sports-streaming/)) |
| **Kalshi `live_data` API** (game state and play-by-play, Sportradar-mapped) | Not stated; public, no auth per SDK docs | Exists ([docs](https://docs.kalshi.com/api-reference/live-data/get-live-data-with-type); [SDK](https://texascoding.github.io/kalshi-python-sdk/resources/live-data/)). **Useful as a free, venue-aligned event clock** |
| MLB Stats API `feed/live` | Fetch p50 about 187 ms; lag behind real play not measured | [GitHub bench](https://github.com/fatehaszaman/mlb-edge-bench) |
| PandaScore Low Latency Feed (LoL, CS2, Dota 2) | "≤5 s of the game server"; claims broadcast is 3-4 minutes behind; separate product/plan | Vendor ([docs](https://developers.pandascore.co/docs/low-latency-feed-overview)). Standard websockets send frames every 2 s ([docs](https://developers.pandascore.co/docs/websockets-overview)) |
| GRID (esports) | From game server; online-event broadcasts delayed about 2-5 minutes | Vendor ([GRID help](https://grid.helpjuice.com/3759126)) |
| Esports streams | 30-120 s | [Unverified], no evidence given ([Substack](https://sharpbetsclub.substack.com/p/latency-arbitrage-in-streaming-vs)) |
| Weather: METAR hourly | Issued about :51-:54 past the hour. 5-minute ASOS values are double-rounded (°F→°C→°F) and should be read as a range. SPECI/6-hour max groups/DSM give exact values | Third-party guide ([wethr.net](https://wethr.net/edu/nws-data-guide)) |
| Weather: ASOS 1-minute (NCEI → IEM) | **Not real time**: about 1-2 days lag | [IEM news](https://www.mesonet.agron.iastate.edu/onsite/news.phtml?id=1407) |

**[Fact]** There is a public esports case study. A Polymarket bot that traded live LoL/Dota 2 in-game
markets reportedly turned about $900 into about $208k over a quarter by reacting to kills/objectives
before prices moved
([esports.net](https://www.esports.net/news/polymarket-bot-makes-over-200k-in-3-months-with-lol-dota-2-esports/)).
PandaScore now markets low-latency feeds explicitly to prediction-market bots
([PandaScore blog](https://www.pandascore.co/blog/low-latency-esports-data-feed-prediction-bot)).

**[Inference]**
- The esports in-game edge was real for someone, but it is a speed race that is now crowded.
- Albert's PandaScore token gives the standard feed, not necessarily the ≤5 s product.
- The Kalshi-intl tracking within about 1c at hourly resolution says nothing about the seconds after
  an in-game event.

### 4.2 Weather settlement: the source changed

- **[Fact]** Kalshi and The Weather Company announced a partnership on 2026-08-27
  ([Kalshi news](https://news.kalshi.com/p/kalshi-weather-company-partnership)). The Crosier paper says
  Kalshi switched daily temperature settlement to TWC after 2026-08-12
  ([arXiv HTML](https://arxiv.org/html/2609.23969)).
- **[Fact]** Kalshi's help page still says daily markets settle on the final NWS Daily Climate Report
  (the high is measured in local *standard* time, so 1:00 AM to 12:59 AM during DST) and hourly
  markets settle on TWC ([help](https://help.kalshi.com/en/articles/13823837-weather-markets)).
- **[Inference]** These sources conflict. Before any weather work, read `rules_primary` /
  `settlement_sources` from recorded market payloads for each series and date. A model that targets
  NWS CLI while the contract settles on TWC is a silent bug.

### 4.3 Rules and legality: what Albert must not do

- **[Fact]** Kalshi bars trading on material non-public information and bars insiders. Rule 5.13(s)
  bars source-agency employees, holders of MNPI, and anyone able to influence the outcome
  ([Kalshi insider page](https://kalshi.com/market-integrity/insider-trading);
  [source-agency prohibitions](https://kalshi-public-docs.s3.amazonaws.com/kalshi-source-agency-trading-prohibitions.pdf);
  [rulebook on CFTC](https://www.cftc.gov/filings/orgrules/rules07012525155.pdf)). Kalshi has
  sanctioned insiders, and the CFTC issued its first prediction-market insider-trading advisory in
  Feb 2026 ([Bartlett & O'Hara fn.](https://conference.nber.org/conf_papers/f243677.pdf)).
- **[Fact]** Courtsiding (trading from inside the venue ahead of feeds) is the stated reason for the
  Crypto.com 3-second delay and the Kalshi delay filing
  ([Gambling911](https://www.gambling911.com/node/78397)). I found **no** Kalshi rule that names
  courtsiding or feed latency.
- **[Fact]** Polymarket sports markets hold marketable orders for a per-market delay (docs: 1 s), and
  those orders cannot be cancelled during the delay
  ([order lifecycle](https://docs.polymarket.com/concepts/order-lifecycle)).
- **Red flags for Albert. Do not build these:**
  1. In-venue observation or paid "scouts". This is courtsiding: legally grey and contrary to the
     purpose of venue delays.
  2. Anything using leaked or embargoed material, such as "prepared remarks leaked ahead of time" for
     mention markets, which one trading guide recommends
     ([recapio digest](https://recapio.com/digest/1-mentions-market-trader-0xtyrael-shares-his-prediction-market-strategies-by-degenz-live)).
     That is MNPI.
  3. Ripping or decoding a delayed broadcast, or using game-client spectator exploits, to get ahead of
     the official public state.
  4. Any order placement on polymarket.com.
- **Acceptable (public information):**
  - public APIs (Kalshi `live_data`, MLB/NBA/ESPN public endpoints, METAR/NWS);
  - public order books of other venues, including polymarket.com read-only websockets;
  - licensed commercial feeds used under their terms;
  - reading the contract rules more carefully than other traders (the Netflix case).
- **[Inference]** A commercial low-latency feed is "public" in the legal sense but sits close to the
  courtsiding line in spirit. If Albert ever uses one, check the venue rules for that specific use
  first.

---

## 5. Q4: Strategies a small account can run

Ranked from most to least plausible for a $100 account. All of them are **[Inference]** from the
evidence above.

1. **Lead-venue-guarded maker quoting (defensive use of lead-lag).**
   - Post maker bids on Kalshi or Polymarket US (0 maker fee on PM US; 0 on most Kalshi series).
   - Cancel or re-price the moment the lead venue (PM intl websocket, which is public and unauthed:
     [docs](https://docs.polymarket.com/market-data/websocket/overview)), Kalshi itself, or a
     `live_data` event moves.
   - Why it could work: Albert's maker backtest already shows that resting into in-game trading
     loses (README §5, 6h row) and the population maker test is about −1.7%. Bartlett & O'Hara show
     the maker's problem is the *informed* fill.
   - Cutting the worst fills (those right after a lead-venue move or a score) needs no fee hurdle.
     It only needs the cancel to beat the taker.
   - Constraint: Kalshi and Polymarket taker delays help a maker here, because they give cancels
     time to land. Polymarket documents that resting-order cancels are not delayed; only marketable
     orders are.
2. **Stale-quote taking on the laggard (Polymarket US) after the leader moves.**
   - The one log (§2.1) says the PM US lag is minutes.
   - Albert already matches PM US ↔ intl/Kalshi.
   - Hurdle: 1.74c taker plus half-spread, held to settlement.
3. **Post-score drift / over-reaction on Kalshi in-game markets.**
   - Uses `live_data` as the event clock and replicates Choi & Hui on Kalshi.
   - Lowest prior: on Betfair it was only a few percent.
4. **Weather: observation-release event study.**
   - Kalshi weather markets are already ahead of NBM (Crosier).
   - The remaining question is whether they lag the *observations* (METAR :51-:54, SPECI, DSM) by
     minutes on the target day, e.g. brackets that the running max has already made impossible.
   - Albert's hourly check says "decided brackets priced correctly". Sub-minute is untested.
   - The settlement-source change (§4.2) must be resolved first.
5. **Mention / culture markets: rule-literal repricing.**
   - Monitor live public transcripts or captions against the exact strike wording.
   - The Netflix case shows 40-minute repricing, but frequency is unknown, history is hard to
     reconstruct, and it is human-judgment heavy.

---

## 6. Evidence-quality summary

| Claim | Sample / period / venue | Peer-reviewed? | Survives Kalshi/PM-US fees? |
|---|---|---|---|
| Betfair prices update fully at goals | Soccer, Betfair, ~2000s | Yes (EJ 2014) | n/a (efficiency) |
| Under- and over-reaction to goals, +2.8% at 2 minutes | 2,017 matches, Betfair | Yes (JEBO 2014) | Doubtful (≈ fee size) |
| 10-15 minute post-goal drift | 2002 World Cup, Tradesports, few trades | Yes (JPM 2007) | Unknown; thin data |
| Betting-line momentum | US books, multi-sport | Yes (JF 2021) | No (authors) |
| Polymarket leads Kalshi on politics, Kalshi leads sports | 2023-26, about 2,000 pairs | Working paper | Not a trading result |
| Kalshi leads PM US by minutes | 3 weeks, 1-minute data, 16 bets | No (GitHub log) | +0.95c, t = 1.4: **not proven** |
| VPIN predicts maker losses (single-name) | 41.6M Kalshi trades, to 2026 | Working paper (NBER conf.) | Defensive use only |
| Weather market beats NBM by about 10% RMSE | 7,590 city-days, Kalshi 2022-26 | arXiv preprint | Implies no forecast edge |
| Last-10-minutes calibration distortion | 23M Kalshi moneyline trades | arXiv preprint | Untested |
| NBA ML/spread dislocations, about 101 bp for 16 s | 173 games, PM intl, Feb-Mar 2026 | arXiv preprint | No: about 1c gross vs 1.75c fee, and 77% too thin for $100 |
| Esports bot $900 → $208k | One PM intl wallet, 3 months | Press | Speed race; not reproducible at standard-feed latency |

---

## 7. Ranked, testable hypotheses for Albert

All five need a **new recorder**: sub-second, timestamped on receipt with a local monotonic clock
plus NTP-synced wall time, append-only raw files, and cache keys per AGENTS.md. Historical Kalshi
hourly candles cannot test any of them. The no-look-ahead rule for all of them: **a signal is usable
only at its local receive time plus a fixed latency budget (e.g. +500 ms for our own order to reach
the venue), never at the event's source timestamp.** Every fill must come from a later recorded
trade that prints through our price, as in `backtest maker`.

### H1 (rank 1). Polymarket US lags Kalshi and PM intl by enough to take stale quotes after fees

- **Claim:** after a ≥3c move in the Kalshi (or PM intl) mid that holds ≥60 s, the matched PM US ask
  or bid is still executable at a price whose settlement value beats 1.74c fee + spread.
- **Data:**
  - Kalshi websocket `ticker`/`trade` (authenticated connection; `orderbook_delta` is a private
    channel, [docs](https://docs.kalshi.com/getting_started/quick_start_websockets));
  - PM intl market websocket (no auth);
  - PM US order book via REST or websocket;
  - Albert's existing cross-venue matches.
  - 2-4 weeks of sports + politics.
- **Test:**
  - Define moves using only leader data up to time t.
  - At t + latency, take the PM US *displayed* ask/bid with its size.
  - Mark to settlement and to the PM US mid at +1, +5 and +15 minutes.
  - Cluster SEs by event.
  - Pre-register the threshold grid. Report the lag distribution (time for PM US to close half the
    gap) even if trading P&L is null.
- **Kill criterion:** mean edge after fees ≤ 0, or t < 2 at ≥100 events.

### H2 (rank 2). Lead-signal cancellation raises maker markout on Kalshi / PM US

- **Claim:** Albert's maker fills are adversely selected mostly in the seconds after (a) a lead-venue
  move or (b) a `live_data` game event. A rule that cancels within X s of such a signal raises
  per-fill markout enough to turn the maker result (currently about −1.7% to +4.6%, not
  significant) positive.
- **Data:**
  - the H1 recorder;
  - Kalshi `live_data` polled at ≤2 s (its own receive time);
  - Albert's maker paper trader in shadow mode, with queue position modelled as in `backtest maker`.
- **Test:**
  - For every simulated fill, compute markout at +10 s, +60 s, +5 minutes and at settlement.
  - Compare fills that a cancel-on-signal rule would have avoided with fills it keeps. Avoided fills
    only count if our cancel would have reached the venue before the fill: use the recorded fill time
    vs signal receive time + latency budget.
  - Also test VPIN-style one-sided flow at 30-70c as the trigger
    ([Bartlett & O'Hara](https://conference.nber.org/conf_papers/f243677.pdf)).
- **Kill criterion:** the difference in markout between avoided and kept fills is not significant,
  or the signal costs more in lost good fills than it saves.

### H3 (rank 3). Kalshi in-game prices under-react to expected events and over-react to surprising ones

- **Claim:** this is the Choi & Hui / Angelini pattern on Kalshi soccer/NBA/NHL in-game moneylines.
  In the minutes after a score, prices drift in the direction of expected-team scores and reverse
  after underdog scores.
- **Data:**
  - Kalshi `live_data` (event times as *received*);
  - Kalshi trades and order book;
  - pre-game Kalshi price as the measure of surprise (no outside model needed).
  - Recorded forward. Check whether `live_data` play-by-play has usable historical timestamps first.
    If it does, check them against recorded receive times before trusting them.
- **Test:**
  - Event study of mid and executable prices from +5 s to +10 minutes after each event, split by
    surprise.
  - Simulated taker entry at +N s at the recorded ask, held to settlement and to +10 minutes.
  - Exclude the final 10 minutes, or analyse them separately: the end-game distortion is a separate
    hypothesis ([Moshrefi](https://arxiv.org/abs/2607.14430)).
  - Cluster by game.
- **Kill criterion:** no drift/reversal larger than 1.75c + half-spread at any N.

### H4 (rank 4). Kalshi temperature brackets lag public METAR/SPECI/DSM releases on the target day

- **Claim:** in the minutes after an hourly METAR (:51-:54) or SPECI that raises the running maximum,
  brackets made impossible (or near-certain) are still quoted at executable prices for longer than
  our latency budget.
- **Data:**
  - METAR/SPECI with *issuance* time (Aviation Weather Center / IEM archive records the receive
    time; the observation time is not enough);
  - NWS DSM products;
  - the Kalshi order book recorded at ≤1 s for the weather series;
  - each market's `rules_primary` to confirm whether it settles on NWS CLI or TWC.
- **Test:**
  - For each release, compute the implied bracket bounds from the reported value range, honouring
    the °F→°C→°F rounding of 5-minute data ([wethr.net](https://wethr.net/edu/nws-data-guide)).
  - Measure how long the inconsistent quotes survive, and what size is executable.
  - Paper-trade only quotes still live at release receive time + latency budget.
- **Kill criterion:** median survival shorter than our latency, or executable size × edge below
  1 order's fee.
- **Prior:** low. The hourly check already found decided brackets priced correctly, and Crosier
  shows the market is ahead of the models.

### H5 (rank 5). Mention markets reprice slowly when wording diverges from the exact strike text

- **Claim:** during live events (earnings calls, speeches, broadcasts), Kalshi mention markets
  misprice near-miss utterances ("Warner Brothers" vs "Warner Bros") or already-satisfied strikes
  for minutes.
- **Data:**
  - Kalshi mention-series market rules (exact strike text);
  - recorded order books and trades;
  - a public live caption/transcript source with receive timestamps (the official webcast's live
    captions, or public speech-to-text run on the public stream). No pre-released or embargoed text.
- **Test:**
  - Forward-only event log of utterance receive times.
  - Measure price path and executable size for 60 minutes after each relevant utterance.
  - Compare against a simple rule-matching classifier that is frozen before the test window.
- **Kill criterion:** fewer than about 20 qualifying events per month, or no repricing lag beyond
  the latency budget.
- **Note:** this is labour- and judgment-heavy and hard to automate safely.

---

## 8. Open items I could not verify

- Aktuğ & Torul information-share numbers (PDF host unreachable). Summaries conflict.
- Whether Kalshi actually implemented a sports order delay after its Dec 2025 filing, and its length.
  Its spokesperson said it "is only a filing"
  ([Gambling911](https://www.gambling911.com/prediction-markets/kalshi-delay-implementation-on-order-executions-120925)).
- Whether Polymarket US applies a sports marketable-order delay like polymarket.com.
- The current Kalshi fee schedule PDF (HTTP 429 during this session). The formula above matches
  `src/albert/fees.py` and secondary sources. Re-check before live trading (AGENTS.md rule 5).
- Measured (not vendor-claimed) latency of Kalshi `live_data`, ESPN, NBA CDN and MLB `feed/live`
  against real play. **[Inference]** Measure it ourselves: record each feed's receive time for the
  same event and compare with the Kalshi trade burst that follows.
