# Market making on non-financial prediction markets (Kalshi, Polymarket US)

Research date: 2026-11-04. Scope: what makes prediction-market makers profitable, and which
maker strategies a $100 account split across Kalshi and Polymarket US could test.

**How to read this document.** Every factual claim has a URL. Tags:

- **[V]** Verified: I read it today on an official page or primary document.
- **[S]** Secondary: a blog, news story, or summary I could not confirm against a primary source.
- **[L]** A standard academic reference, cited from the literature. I did not re-read it today.
- **[I]** My inference. These are not facts. Treat them as hypotheses to test.

---

## 0. Executive summary

1. **Makers do earn more than takers on Kalshi, but the edge is small and is a structural
   average.** Over 72.1M Kalshi trades, makers' mean excess return was +1.12% per trade and
   takers' was −1.12%. The gap is largest in thin, retail-heavy categories (Media +3.64%, World
   Events +3.66%, Entertainment +2.40%, Weather +1.29%) and smallest in Finance (+0.08%)
   [S, from the author's own page: https://www.jbecker.dev/research/prediction-market-microstructure].
   Before 2024, before professional makers arrived, the sign was reversed: makers −2.0%,
   takers +2.0% (same source). Albert's own replication found realized maker fills on favorites
   (≥ 50c) earn +3-4% in Weather and Sports (README §3a).
2. **A new maker does not inherit that average.** The average covers all makers, including
   fast and well-informed ones, each with their own queue position, cancel speed and inventory
   control. The −1.7% Albert measured on "bid the favorite 24h out" is consistent with the
   textbook result: a slow one-sided quoter that improves the bid becomes the first target for
   informed flow (Glosten–Milgrom) [I]. The only practitioner report I found from someone who
   ran a cloned Polymarket MM bot says net PnL was negative and that "directional risk is the
   #1 P&L killer" [S: https://dev.to/fatherson/i-cloned-a-polymarket-market-making-bot-and-ran-it-heres-what-i-learned-46ag].
3. **For a small account, liquidity rewards are the most concrete source of non-directional
   income, and they are open to retail.** Kalshi's Liquidity Incentive Program (LIP) is open to
   "most regular U.S. Kalshi members" and pays $1-$1,000 per market per day for resting size
   near the best price. It ends **2027-01-01** [V: https://help.kalshi.com/en/articles/13823851-liquidity-incentive-program].
   Polymarket US's Liquidity Incentive Program is "Open: no signup or application needed"
   [V: https://docs.polymarket.us/incentives/overview]. Its live schedule lists 188 programs
   [V: https://polymarket.us/rewards]. The designated-MM programs on both venues require a
   signed Market Maker Agreement, so they are **not** available to us
   [V: Kalshi DLP filing https://www.cftc.gov/filings/orgrules/rules10182634499.pdf; Polymarket US
   https://www.cftc.gov/filings/orgrules/rules09302630733.pdf].
4. **Binary-contract theory [I, derived from L]:** for a contract at price p, the total remaining
   variance to resolution is always p(1−p), because the price is a martingale on {0,1}. The only
   question is *when* that variance arrives. A maker wants markets and windows where little of
   it arrives while the quote rests (overnight weather, long-dated futures between news) and
   must be out before it arrives in bulk (game start, observation releases, speeches, vote
   counts).
5. **Fees.** Polymarket US: fees.py matches the official page, but the page has new details.
   The current schedule took effect **2026-10-28 22:00 ET** (fees.py says 2026-10-19). Tiers move
   to a rolling 30-day window on **2026-11-06**. There are taker rebate tiers and a separate combo
   fee [V: https://docs.polymarket.us/fees]. Kalshi: the formulas in fees.py agree with every
   source I could reach, but kalshi.com returned HTTP 429 to the fetcher, so I could **not**
   re-read the official fee schedule today. See §3.

---

## 1. Theory

### 1.1 Classical models [L]

- **Glosten & Milgrom (1985)**: the spread exists because some takers are informed. A competitive
  maker sets ask = E[V | buy] and bid = E[V | sell], so the spread is the break-even cost of
  adverse selection. https://doi.org/10.1016/0304-405X(85)90044-3
- **Kyle (1985)**: an informed trader hides in noise flow, and price impact λ rises with the
  share of informed flow. https://doi.org/10.2307/1913210
- **Avellaneda & Stoikov (2008)**: an inventory-averse maker quotes around a reservation price
  r = s − qγσ²(T−t), with spread γσ²(T−t) + (2/γ)ln(1+γ/k). Here k is the decay of fill
  intensity with distance from mid. https://doi.org/10.1080/14697680701381228
- **Guéant, Lehalle & Fernandez-Tapia (2013)**: closed-form asymptotic quotes with inventory
  limits, so quotes on the side that would breach the limit are switched off.
  https://arxiv.org/abs/1105.3115
- **Toxicity measurement**: VPIN (Easley, López de Prado & O'Hara 2012),
  https://doi.org/10.1093/rfs/hhs053. Practitioners mostly use **markouts**: the mid at t+Δ minus
  the fill price, signed by side.
- **Automated market makers (LMSR)**: Hanson's logarithmic market scoring rule has a bounded
  worst-case loss of b·ln(n), so the house subsidizes liquidity.
  https://mason.gmu.edu/~rhanson/mktscore.pdf. Liquidity-sensitive variants are in Othman,
  Pennock, Reeves & Sandholm (2013), https://doi.org/10.1145/2509413.2509414. Kalshi and
  Polymarket US are both order-book (CLOB) venues, not LMSRs. The relevant lesson is that
  **someone has to pay for liquidity in thin event markets**. In an LMSR the house pays the
  bounded loss. On a CLOB it is either takers (through the spread) or the exchange (through
  liquidity rewards). The reward programs in §3 are the CLOB version of the LMSR subsidy [I].

### 1.2 Prediction-market-specific papers

- **Feil & Nendel, "Optimal Market Making in Prediction Markets"** (arXiv:2607.17991, July 2026).
  The price is modeled as the conditional probability generated by a latent belief diffusion.
  The maker maximizes terminal wealth with both mark-to-market inventory risk and settlement
  risk on positions still open at resolution. They prove a unique optimal bid and ask exists,
  and the optimal strategy "improves downside protection while keeping most of the expected
  profit" of a myopic benchmark [V abstract: https://arxiv.org/abs/2607.17991].
- **Dalen, "Toward Black–Scholes for Prediction Markets"** (arXiv:2510.15205). This uses a
  logit jump-diffusion, x = logit(p), and quotes Avellaneda–Stoikov in logit units, then maps
  back. Probability half-spreads scale as roughly p(1−p) × the logit half-spread, so a display
  floor (the 1c tick) is needed. The inventory cap tightens as p → 0 or 1. The cross-contract
  hedge ratio is β ≈ (S′ᵢ/S′ⱼ)·ρᵢⱼ, shrunk by α ∈ [0.5, 1], and the paper recommends
  over-hedging around known jump windows [V: https://arxiv.org/html/2510.15205v1]. The same
  author's other paper (arXiv:2604.10005) was **withdrawn** and should not be cited as evidence
  [V: https://arxiv.org/abs/2604.10005].
- **Xi, Moallemi, Pai & Wang, "Volatility in Prediction Markets: A Structural Approach"**
  (arXiv:2607.08199, v3 2026-10-13). This combines a Wright–Fisher deadline-resolution component
  with a Glosten–Milgrom order-flow component on a large Kalshi panel. Volatility is highest
  near 50c and rises toward resolution. **Sports contracts are event-concentrated and jump-like;
  economics contracts are smoother** [V abstract: https://arxiv.org/abs/2607.08199].
- **Bürgi, Deng & Whelan, "Makers and Takers: The Economics of the Kalshi Prediction Market"**
  (GWU WP 2026-001 / CEPR DP20631). Kalshi data from 2021 to April 2025, covering 46,282
  contracts with ≥ $1k volume and a final spread ≤ 20c. Per-contract average post-fee return was
  **−9.64% for makers vs −31.46% for takers**. These are unweighted averages dominated by cheap
  contracts. Makers buying at ≥ 50c earned **+2.6%**. Makers buying at ≤ 10c had significantly
  negative returns on 5 of 6 horizons. On the final day, makers' losses on cheap contracts look
  like takers' losses. In the paper's model, makers have less extreme beliefs than takers and
  spreads arise endogenously. Kalshi charged makers no fees during their sample. "Kalshi began
  to charge fees on Makers after April 2025" [V: https://www2.gwu.edu/~forcpgm/2026-001.pdf].
- **Becker, "Microstructure of wealth transfer"** (72.1M Kalshi trades through 2025-11-25).
  Makers +1.12%, takers −1.12% per trade. At 50c, takers −2.65% and makers +2.66%. Makers buying
  NO earned +1.25 pp vs +0.77 pp for makers buying YES. NO beats YES at 69 of 99 price levels.
  Effect sizes are tiny (Cohen's d ≈ 0.02-0.03), and the author argues makers profit "from market
  structure rather than from forecasting." The spread is not observable in his data, so spread
  capture and "exploiting biased flow" cannot be separated
  [S: https://www.jbecker.dev/research/prediction-market-microstructure].
- **Charjan, Polymarket microstructure note** (on-chain fills through March 2026, 327M fills,
  916k maker addresses). Volume-weighted markouts are positive for every maker-volume decile at
  10s, 1m and 10m. They rise steeply with maker size: D10 is about 84× D1 at 10 minutes. The
  note cannot see quotes or cancels, and its PnL split is unreliable (51% of rows have price ×
  amount ≠ USD by more than 1%) [S: https://avdheshcharjan.github.io/notes/polymarket-mm.html].
  Inference [I]: larger, presumably more automated makers get better markouts. That is the
  opposite of what a small, slow maker can expect.

### 1.3 Adapting the models to binaries: what matters for Albert [I]

1. **Inventory risk is jump risk, with a known total.** For one contract held to resolution,
   the payoff variance is p(1−p). That is at most 0.25 at p = 0.5, so a $1 standard deviation for
   4 contracts at 50c. The variance is not spread evenly over time. For a game it arrives mostly
   after kickoff. For a daily-high temperature bracket it arrives as the afternoon observations
   come in. For a nominee market it arrives in a few news jumps. In A–S terms, replace σ²(T−t)
   with "expected variance released before I can cancel or exit". Model that per market type
   and per window, not as a constant.
2. **Rule from (1): the cancel time is the main control, not the spread.** Adverse selection is
   concentrated in the windows where variance is released: lineups and injuries, game start,
   observation times, live speeches for Mentions. A maker who rests through those windows is
   short an option that informed traders exercise. Fix cancel windows per market type first,
   and only then tune the spread.
3. **The YES bid and the NO bid form one two-sided quote.** On Kalshi, buying both YES and NO in
   one market nets automatically [S: https://news.kalshi.com/p/collateral-return]. A NO bid at
   (1−a) is economically an ask at a. If both fill, the maker has locked a − b minus fees,
   **unless a jump happens between the two fills**. In that case the maker holds one leg at the
   worst time, which is the normal outcome under adverse selection.
4. **Bounded prices compress risk near 0 and 1, but the tick does not.** At 95c the payoff
   variance is 0.0475 and a 1c tick is about 21% of the 5c risk-to-zero on the NO side. Kalshi's
   subpenny grid (§5) partly fixes this on enabled markets. The Bürgi result (makers buying ≤ 10c
   lose) says **do not be the maker who buys longshots**. Becker's NO-side result says that if
   you rest orders near the extremes, rest the NO (favorite-side) bid.
5. **One-sided quoting is directional betting.** Albert's existing maker test only bid on the
   favorite. It had no offsetting NO bid and no inventory skew, so every fill was a long that
   was held or cancelled. That is a directional bet with a small price improvement, and the A–S
   framework predicts it inherits the full adverse selection without the round-trip spread.

---

## 2. Empirical: who makes markets, and what they report

### 2.1 Kalshi

- **Susquehanna (SIG)**: Kalshi's first committed institutional market maker, announced in April
  2024. Kalshi said SIG would provide depth of 100,000+ contracts and spreads of 2-3c or less on
  average [S, press release: https://www.businesswire.com/news/home/20240403664852/en/Kalshi-Onboards-Its-First-Dedicated-Institutional-Market-Maker].
  SIG was named alongside Kalshi in 2025 state suits that argue makers act as sportsbooks
  [S: https://straighttothepoint.substack.com/p/a-sportsbook-in-sheeps-clothing].
- Kalshi also has an affiliated market maker, "Kalshi Trading". A practitioner noted in
  August 2025 that Kalshi had not said whether SIG's and Kalshi Trading's quotes count in
  liquidity-reward scoring [S: https://ufoholdings.substack.com/p/kalshis-new-liquidity-incentives].
  The current LIP page excludes "Kalshi affiliates and employees" and MMs with existing
  agreements from rewards [V/S: help page above and search snippet].
- Kalshi's help center says market makers on Kalshi are "mostly individuals and small
  institutions" [V: https://help.kalshi.com/trading/who-are-you-trading-with].
- The maker-taker reversal around 2024 Q2 (Becker, above) is the best evidence that
  **professionalization moved the edge to makers**. It also means the marginal maker now
  competes with SIG-class firms in liquid series [I].

### 2.2 Polymarket (international = data only; US = tradable)

- Open-source bots: `warproxxx/poly-maker` describes itself as "a reference implementation and
  research harness, not a guaranteed-profitable product" and warns "Market making on
  Polymarket is competitive and can lose money" [S: https://github.com/warproxxx/poly-maker].
  Also `LeapTech-Lab/poly-maker` [S: https://github.com/LeapTech-Lab/poly-maker] and an A–S bot
  with a logit mode that clamps the reservation price into [0,1]
  [S: https://github.com/nexosdata/polymarket-market-maker-bot]. All three target polymarket.com,
  which is not tradable for us. They are useful only for design ideas.
- The one detailed practitioner report (above) had **negative** net PnL. The causes were bugs,
  directional hits, and stale inventory snapshots that kept adding to losing sides. Its lessons:
  "low-volatility markets are easier to hedge", and "volatile markets can wipe out weeks of
  rewards in minutes" [S].
- Earnings claims in guide/affiliate blogs ($150-300/day per market, 5-20% annualized on
  rewarded capital) are **unverified marketing** and should not be relied on
  [S: https://startpolymarket.com/strategies/market-making/, https://laikalabs.ai/prediction-markets/polymarket-liquidity-rewards].
- Polymarket international's reward formula, for reference: quadratic in distance from the
  size-adjusted mid, S = ((v−s)/v)²·b. For mids in [0.10, 0.90], one-sided quotes score at 1/c
  with c = 3. Outside that range, two-sided quoting is required. Samples are about once per
  minute, and the minimum payout is $1/day [V: https://docs.polymarket.com/market-makers/liquidity-rewards].
  Polymarket US uses a **different** formula (§3.2).

### 2.3 What we still don't know (evidence gaps)

- I found **no credible, audited P&L of a small retail maker** on either venue. Every profitable
  claim was anecdotal.
- I found no academic measurement of quoted spreads and depth by Kalshi category. Bürgi et al.
  drop spreads > 20c, and Becker cannot see spreads. **Albert can measure this itself** from
  order-book snapshots, which is a gap worth filling before any strategy.

---

## 3. Incentive programs and fees: exact current terms

### 3.1 Kalshi

**Liquidity Incentive Program (open to retail)** [V: https://help.kalshi.com/en/articles/13823851-liquidity-incentive-program]

- Eligibility: "Most regular U.S. Kalshi members". Kalshi affiliates and employees, IBs, FCMs
  and their customers, and non-U.S. users are excluded. Rewards above IRS reporting thresholds
  need a verified SSN.
- Ends **2027-01-01**, and can be ended or modified at any time.
- Snapshots are taken once per second, at a random moment within each second.
- **Reference price**: the first level, counting down from the best bid, at which cumulative
  resting size reaches one fifth of the Target Size. Orders at or better than the reference
  price score 1.0×. Orders worse than it score DiscountFactor^ticks.
- Your score is size × multiplier divided by the side's total, with YES and NO scored
  separately. A snapshot is worth at most 2.0 across all participants.
- **Exclusions**: a snapshot pays nobody if the market is not open, or if **total** resting
  orders (all participants) do not reach Target Size on both YES and NO. Target Size is between
  100 and 20,000 contracts.
- Reward = your share × period pool × (valid snapshots / total). Rounded **down** to the cent.
  **Minimum payout is $1.00 per market program**, and sub-dollar amounts do not combine.
- Pools are $1-$1,000 per market per day. Periods last up to 31 days and may overlap.
- Discovery: kalshi.com/incentives, a badge on the market, and a public API endpoint,
  `GET /incentive_programs`, which returns market, type, dates, pool, Target Size and Discount
  Factor [V: https://help.kalshi.com/en/articles/16076644-where-to-find-rewards-and-incentive-info].
  Status/type filters come from a third-party SDK [S: https://texascoding.github.io/kalshi-python-sdk/resources/incentive-programs/].
- Worked example (practitioner, Aug 2025): target 1,000, DF 0.9, $100 pool. 500 at best, 300
  one tick back and 400 two ticks back score 500 / 270 / 324 [S: ufoholdings link above].

**Designated Liquidity Provider (DLP) Market Making Program (not for us)**
[V: https://www.cftc.gov/filings/orgrules/rules10182634499.pdf, filed 2026-10-19]

- "Eligible Participants" are members **who have executed a Market Maker Agreement**. DLPs are
  chosen in a publicly noticed Bidding Period of at least 3 days, in which they bid the minimum
  reward they would accept.
- Obligations: Maximum Spread, Minimum Size within Spread, Minimum Uptime (no reward at ≤ 50%
  uptime), and quoting all markets in the series. Periods are 30-90 days. The cap is $50,000 per
  series per week. Categories are Weather, Culture, Politics, Mentions, Sports, Sci/Tech, and
  others. Termination date is 2027-12-01.
- Why this matters to us [I]: DLP-incentivized series will have a contracted maker with tight
  spreads. Those are the **worst** places for a small maker to compete on price. Kalshi says it
  selects series by "nascency and observed execution costs relative to the mid-spread."

**Volume Incentive Program**: being ended early, no earlier than 2026-10-13, after
wash-trading scrutiny around ether perpetuals [S: https://www.gambling.com/us/news/kalshi-end-volume-incentive-program-october-13,
https://sbcamericas.com/2026/10/02/kalshi-end-volume-rewards-scrutiny/]. It is a taker program
and not relevant to us.

**Kalshi fees (re-verification status)**

- I could **not** fetch kalshi.com/fee-schedule or the PDF today: HTTP 429, and the PDF URL
  returns a bot-check HTML page. The secondary sources below are consistent with fees.py.
- Taker 0.07·C·P·(1−P). Maker 0.0175·C·P·(1−P), only on designated series. As of 2026-11-04 those
  are **107 of 4,086 Sports series, 10 of 873 Economics, 2 of 294 Crypto, and 0 of 501
  Climate/Weather**. 31 series carry a half or zero multiplier
  [S: https://www.oddsshopper.com/articles/prediction-markets/how-does-kalshi-make-money].
  The fee-schedule PDF was titled "Fee Schedule for July 2026 - 7.7.26 Update"
  [S: search result for https://kalshi.com/docs/kalshi-fee-schedule.pdf].
- **Fee types**: `quadratic`, `quadratic_with_maker_fees`, `flat`, and reportedly
  `quadratic_with_combo_maker_fees` [S: https://cdn.jsdelivr.net/npm/kalshi-typescript@3.13.0/models/series.ts,
  https://docs.allium.so/historical-data/predictions/kalshi/series-fee-changes].
  - **Flag for fees.py [I]**: `"maker_fee" in fee_type` also matches
    `quadratic_with_combo_maker_fees`. If that type charges makers only on combo fills, Albert
    over-charges maker fees on single-market fills in those series. Check against the official
    schedule.
  - Kalshi publishes scheduled changes at `GET /series/fee_changes` [S, same Allium link]. Poll
    it before each live session.
- **Rounding** [V: https://docs.kalshi.com/getting_started/fee_rounding]: the trade fee is the
  model fee rounded up to $0.000001. A rounding fee then aligns the balance to **$0.0001 for
  direct members, $0.01 for non-direct (FCM-cleared)**. An accumulator rebates the overpayment,
  and the net fee is never negative. Albert's "round each order up to the cent" mode is
  conservative.
- **Maker fees are charged only on execution. Cancelling is free**
  [S: fee-schedule PDF quoted in search results].

### 3.2 Polymarket US

**Fees** [V: https://docs.polymarket.us/fees]

- Taker Θ·C·p·(1−p) with Θ = 0.0695 (0.10 for table tennis). This matches fees.py.
- Maker rebate tiers by qualifying maker volume: tier 1 (< $10M) gets nothing. Tier 2
  ($10M-50M) is Θ = −0.0035, tier 3 is −0.0070, tier 4 ($200M+) is −0.0104. **A $100 account is
  permanently tier 1, so maker fills are fee-free with no rebate.**
- **New since fees.py was written:** taker rebates of 3-30% by taker volume, starting at
  $250k. Combo taker fee C·p·[0.0695(1−p) + 0.085(1−p)⁴]. Combo fills earn no maker rebate.
- Volume counts the premium paid (C·p for a buy), not face value. Banker's rounding to $0.01.
  On multi-fill orders, the adjustment can only reduce a fill's charge. Maker rebates are
  computed per fill.
- Dates: this schedule took effect **2026-10-28 22:00 ET**. **Update the fees.py docstring,
  which says 2026-10-19.** Tiers move to a rolling 30-day window, recomputed daily, on
  **2026-11-06 13:00 ET**.

**Liquidity Incentive Program (open)** [V: https://docs.polymarket.us/incentives/liquidity, https://docs.polymarket.us/incentives/overview]

- "No signup or application needed". It rewards resting orders "whether or not they fill".
- Score = DiscountFactor^(ticks from best) × size, snapshotted every second. Each side is
  normalized to 1.0, and orders beyond cumulative Target Size from the best price do not score.
- Optional **Max Spread** requires the size-adjusted bid and ask to be within MaxSpread of the
  mid. Otherwise nobody is paid that second, and the share is not redistributed. **You do not
  have to quote both sides yourself.**
- Time periods: Early/pre-game (listing to −6h), Day-of (−6h to start), Live, and Daily for
  events without a fixed start. Each has its own pool.
- Payout within about 7 business days. **No per-trader cap. Rewards under $1.00 are not paid.**
  Cancelled or postponed games pay nothing.
- API: `GET /v1/incentives`. The `timePeriods[].maxSpread` field is in dollars.

**Live schedule snapshot (2026-11-04)** [V: https://polymarket.us/rewards, partial page]. It
lists 188 programs over 98,014 markets. Each pool is shared across the program's markets.

| Program | Period | Pool/day | DF | Target | Max spread | Markets |
|---|---|---|---|---|---|---|
| NFL Spreads Live | Live | $7,000 | 0.25 | 25,000 | none | 1,189 |
| NFL Player Props Live | Live | $3,250 | 0.35 | 550 | none | 1,898 |
| Presidential Election | Daily | $500 | 0.20 | 20,000 | none | 113 |
| Midterms Margins A / B | Daily | $300 / $80 | 0.30 | 1,000 | 3c | 111 / 403 |
| ATP WTA Futures | Daily | $200 | **0.90** | 8,000 | none | 123 |
| Climate (Weather) | Live | $185 | 0.30 | 7,500 | none | 90 |
| NBA Winner Futures | Daily | $100 | **0.90** | 10,000 | none | 509 |
| Dancing With the Stars | Live | $250 | 0.30 | 500 | none | 21 |
| Culture LOW | Daily | $40 | 0.25 | 500 | 6c | 854 |
| UTR Moneyline Day-of | Day-of | $10 | 0.35 | 250 | 7c | 122 |
| Politics T4 Coverage | Daily | $2 | 0.40 | 2,000 | 6c | 3,154 |

Observations [I]:
- The **largest pools are in Live periods**, which is exactly when variance and toxicity peak.
  The pools are large because the risk is large.
- **Daily futures programs with DF 0.9** (ATP/WTA futures, NBA Winner futures) pay for size
  several ticks behind the best price with little penalty, in slow, long-dated markets. That is
  the best fit for a slow small maker, **if** the book is not already deeper than Target Size
  near the touch. Orders beyond Target Size do not score.
- ATP/WTA futures pay roughly $200/123 ≈ $1.6 per market per day on average, if the pool is
  split evenly, which is unverified. A $25 slice of the account cannot hold a meaningful share
  of an 8,000-contract target in many markets. Expect cents per market per day, and the **$1
  minimum payout per period** can make a share worthless. This must be computed from live books
  before committing.
- The **Midterms** programs refer to markets resolving around the 2026-11-03 election and will
  shrink or roll off. Do not plan around them.

**Market Maker Program (not for us)**: requires an approved Market Maker Program Participant
Agreement and ≥ 5% of a series' monthly maker notional to earn any rebate. Rebates are a share
of taker fees on passive fills [V: https://www.cftc.gov/filings/orgrules/rules09302630733.pdf].

**Volume Incentive Program** (taker-side, open): pro-rata share of a contract pool, with only
trades at 3-97c counting [V: https://docs.polymarket.us/incentives/volume]. Not relevant to
makers.

### 3.3 What a $100 account could earn from rewards: a framework, not a forecast [I]

For a program with pool P per period, your expected reward ≈
P × (your time-averaged qualifying size share) × (fraction of valid snapshots), paid only if
it is ≥ $1 per market program.

Example: on Kalshi, a weather market with a $10/day pool where competitors rest 2,000
qualifying contracts per side. Resting 40 contracts per side at the reference price (about $20
of collateral per side at 50c) gives a share of about 40/2,040 ≈ 2% of each side, or about
$0.20/day. Over a 31-day period that is about $6, which clears the $1 minimum. **If** none of the
resting size is adversely filled, that is roughly a 15%/month yield on $40. The whole question
is whether adverse fills cost more than that. Albert's −1.7%-per-filled-dollar result suggests
they easily could if quotes rest through information windows.

---

## 4. Practical strategy design for $100

### 4.1 Where adverse selection should be lowest [mostly I, with supporting evidence]

| Market type | Evidence | Toxicity profile | Maker fit |
|---|---|---|---|
| **Kalshi weather (daily high/low brackets)** | Maker excess +1.29% (Becker) [S]; Albert replication: maker ≥ 50c +3.0%, t=4.0, positive each month (README §3a); **0 of 501 climate series have maker fees** [S] | Information arrives on a known clock (model runs, METAR/ASOS obs during the day). Before the day starts, the main information is NWP guidance, which Albert already ingests (NBM). | **Good**, if quotes are pulled before the intraday observation window and anchored to NBM. |
| **Long-dated futures/awards/culture** (season winners, tennis futures, DWTS, awards) | Entertainment +2.40%, Media +3.64% maker gaps (Becker) [S]; DF 0.9 reward programs on PM US [V] | Slow diffusion plus rare jumps (injury, elimination, leak). Long capital lockup. | **Good for rewards**, poor for capital turnover. Use small size and wide quotes. |
| **Pre-game sports (> 6h out)** | Sports maker +1.12% (Becker); Albert's "bid favorite 24h out" lost 1.7% per filled $ | Sharp money moves lines on injury/lineup news; sportsbooks lead. | Only with a live reference price (sportsbook or Polymarket intl mid) and cancel-on-divergence. |
| **Live sports** | Largest PM US pools [V]; jump-like volatility (Xi et al.) [V] | Extreme: fast traders with TV/data feeds. | **Avoid.** |
| **Mentions / live speeches** | High maker gaps, but small N (Becker); Elections −12% for makers in Albert's replication | Resolves in real time during the event. | Only well before the event, then cancel. |
| **Election/politics near votes** | Albert: makers −12% (Elections), −1% (Politics) | Dominated by information. | **Avoid.** |

### 4.2 Design elements

1. **Two-sided via YES bid + NO bid** (Kalshi), or BUY_LONG plus a short intent on Polymarket US
   [V: intent naming in https://docs.polymarket.us/api-reference/orders/create-order]. Skew both
   quotes against inventory (A–S reservation price). Turn off the side that would breach the
   inventory cap (GLFT).
2. **Post-only always.** Kalshi `post_only` rejects an order that would cross
   [S: https://docs.kalshi.com/python-sdk/models/CreateOrderRequest]. On Polymarket US the
   closest documented field is `participateDontInitiate` [S, verify with a capture].
3. **Join, don't improve.** Improving the bid by 1c (Albert's current maker rule) makes you
   the first fill for any informed seller [I, Glosten–Milgrom]. For reward scoring on both
   venues, size at or better than the reference price scores 1.0×, so there is no need to be
   alone at the top.
4. **Reference price**: Polymarket intl mid (data-only use is allowed), sportsbook consensus,
   or NBM guidance for weather. If the venue mid moves more than X from the reference, or the
   reference moves, cancel first and requote later.
5. **Cancel windows** (hard rules): before scheduled start, lineup and injury report times,
   observation windows, live speeches and results. Kalshi `cancel_order_on_pause` cancels on
   exchange pauses [S, SDK docs above].
6. **Toxicity signals to monitor and act on** [I, standard practice]:
   - trade-through of more than one level;
   - fills larger than the median size;
   - several related markets moving together (moneyline, spread and total of one game; all
     legs of one event);
   - order-book imbalance flips;
   - divergence from the cross-venue mid;
   - time-to-event under a threshold;
   - your own markouts at 1m, 10m and 1h turning negative over a rolling window. If they do,
     widen quotes or stop.
7. **Inventory and risk caps for $100**:
   - at most $5 exposure per market;
   - at most one open directional leg per underlying event;
   - daily loss stop;
   - a kill switch that cancels everything.

   This fits the existing RiskConfig [I].
8. **Cross-venue hedging**: quote maker on one venue and hedge as taker on the other. This only
   works when the fill price beats the other venue's executable price by more than the taker
   fee (about 1.75c at 50c on either venue) plus slippage. Fills are adversely selected exactly
   when the other venue has moved, so this is mostly a **filter** ("only rest quotes that are
   hedgeable at a profit now") rather than a guaranteed hedge. Resolution rules may differ
   between venues: check each contract's terms [I].

---

## 5. Key risks and venue mechanics

- **Adverse selection near events** is the main risk. See the evidence above (Albert −1.7%,
  Elections −12% for makers, the practitioner report).
- **Resolution jump risk**: inventory held through resolution has variance p(1−p) per contract.
  With a $100 bankroll, one 20-contract inventory at 50c that goes to 0 is a $10 loss (10%).
  Cap inventory in dollars at risk, not in contracts [I].
- **Stale quotes**: Kalshi Basic API tier allows 20 reads and 10 writes per second. Advanced is
  30/30 and needs at least 1 of the last 100 orders placed via API. Only order creates and
  cancels count as writes [S: https://docs.kalshi.com/getting_started/rate_limits via search
  summary]. Use WebSocket feeds. A sweep over many markets must fit the write budget, since
  cancelling 50 markets takes about 5s on Basic. Use batch cancel [I].
- **Self-trade**: Kalshi orders take `self_trade_prevention_type` (`taker_at_cross` or `maker`)
  [S: Kalshi SDK/spec links above]. Since YES and NO in one market net, resting a YES bid and a
  NO bid at crossing prices is impossible anyway [I].
- **Position limits**: I could not find current per-member limits for non-financial contracts in
  a primary source. They do not bind at $100 [I].
- **Capital lockup**: each resting buy reserves its cost. Long-dated futures tie up collateral
  for months, so turnover, not edge per fill, can be the binding constraint. Kalshi nets
  NO positions across mutually exclusive legs ("collateral return"), which frees capital in
  multi-leg events [S: https://news.kalshi.com/p/collateral-return].
- **Tick size and subpenny**: Kalshi publishes `price_ranges` per market (start, end, step), and
  off-grid prices are rejected. Some structures have finer ticks near 0 and 1 (e.g.,
  `center_deci_edge_centi_cent`). Prices are `*_dollars` strings with up to 4 decimals.
  Fractional contracts go down to 0.01 on enabled markets. The doc lists rollout dates of
  2026-11-27 (single-stock perps) and 2026-12-17 (remaining markets), which may refer to the
  fractional/subpenny rollout [V: https://docs.kalshi.com/getting_started/fixed_point_migration].
  **Implication**: a 1c quote improvement may no longer be the minimum. Competitors can step
  ahead by 0.1c, which erodes queue priority for slow makers. Parsers must read `*_dollars` and
  `*_fp` fields [I].
- **Reward-program risk**: Kalshi LIP ends 2027-01-01 and either venue can change terms at any
  time. Do not build a strategy whose only edge is a program with two months left without
  re-checking for a successor [I].
- **Rules risk**: both venues prohibit spoofing and "quotes with no bona fide intent to trade"
  in their incentive programs [V: Polymarket US filing §7]. Reward-harvesting quotes must be
  genuinely fillable and honored. This fits AGENTS.md rule 7.

---

## 6. Ranked testable maker strategies

Every strategy below needs the same infrastructure first:

- **A recorded order-book feed**: WebSocket snapshots and deltas from both venues, saved
  append-only.
- **A queue-aware fill model**: your order fills only after the size ahead of it at that price
  has traded, not only when a trade prints *through* your price. The current trade-through
  rule is conservative and selects adversely, because it records fills only when price went
  past you [I].
- **A markout report**: 1m, 10m and 1h markouts, plus PnL at resolution, per fill.

### 1. Reward-harvesting on slow, rewarded markets (Kalshi LIP + Polymarket US LIP), two-sided, join-not-improve

- **Hypothesis**: in long-dated or slow rewarded markets (weather on Kalshi; tennis and NBA
  futures, culture and DWTS pre-event on Polymarket US), rewards plus spread exceed adverse
  selection for small, two-sided, inventory-skewed quotes at the reference price, with hard
  cancel windows.
- **Data needed**:
  - `GET /incentive_programs` (Kalshi) and `GET /v1/incentives` (Polymarket US), polled daily
    and recorded;
  - 1-second book snapshots for candidate markets, to compute the share we would have held and
    the valid-snapshot fraction;
  - trades with taker side, for queue-aware fills and markouts.
- **Test**: run a 2-4 week paper simulation. The pass criterion is reward share − adverse
  markout − fees > 0 per day, after the $1 minimum payout per program.
- **Main failure mode**: our size share is too small to clear $1 per program, or crowded books
  push our size beyond Target Size. A news jump fills one side at the worst time and wipes out
  weeks of rewards.

### 2. Kalshi weather brackets: overnight two-sided quoting anchored to NBM guidance

- **Hypothesis**: before the observation day, the main information is model guidance, which
  Albert already ingests. Quoting YES and NO around an NBM-implied fair value, with skew, earns
  the weather maker premium. That premium is +1.29% (Becker) and +3.0% for ≥ 50c fills
  (Albert), with zero maker fees on weather series and possible LIP rewards on top. Cancel
  everything before the intraday observation window.
- **Data needed**: archived NBM/NWS guidance timestamped before each quote (no look-ahead),
  Kalshi weather books and trades, CLI settlement.
- **Test**: markouts by hour of day, to identify the toxic window empirically, and fill
  probability at the reference price.
- **Main failure mode**: bracket-edge model error, where the market already prices the forecast
  better than NBM. Informed observation-watchers trade early, so overnight is not actually
  quiet.

### 3. Pre-game sports maker quotes anchored to Polymarket intl mid, with cancel-on-divergence

- **Hypothesis**: Albert's −1.7% came from resting stale, improving, one-sided bids. If quotes are
  two-sided, join the best price rather than improve it, are pulled whenever the Polymarket
  intl mid (or a sportsbook line) moves more than k ticks or diverges from our quote, and are
  cancelled at T−60 to T−90 minutes (before lineups), then markouts turn non-negative.
  Polymarket US maker fees are 0 at tier 1. On Kalshi, use only series where `fee_type` has no
  maker fee.
- **Data needed**: Polymarket intl price history at ≤ 1-minute resolution, which Albert already
  has partially. Kalshi and Polymarket US books and trades on matched games. A lead-lag
  estimate showing which venue moves first.
- **Main failure mode**: the reference lags the informed flow (all venues move together on
  news), so the filter cancels after the fill. Fill rates too low to matter. Kalshi
  maker-fee series (107 sports series) erase the edge.

### 4. Cross-venue maker-then-hedge filter (Polymarket US maker, Kalshi taker hedge, or the reverse)

- **Hypothesis**: rest a maker quote on venue A only at prices where an immediate taker hedge on
  venue B locks in ≥ 1c after B's taker fee. A fill is then either riskless (hedge immediately)
  or not taken.
- **Data needed**: simultaneous books on matched contracts (Albert's `xvenue` scanner), plus
  latency measurements.
- **Main failure mode**: fills happen exactly when B has already moved, so the hedge is
  unavailable (legging risk). Contract wording or resolution differs across venues. The
  opportunity count is too small, given the earlier finding that cross-venue gaps rarely clear
  fees.

### 5. Passive NO-side (favorite-side) bids in multi-leg long-dated events

- **Hypothesis**: NO beats YES at 69 of 99 price levels, and NO makers beat YES makers (Becker).
  Makers buying ≤ 10c lose (Bürgi). Resting NO bids, which economically sell overpriced YES
  longshots at 3-15c, in multi-leg culture/awards/politics-nominee events earns the longshot
  premium as a maker, with Kalshi collateral netting across mutually exclusive legs.
- **Data needed**: Kalshi multi-leg event trades and books (Albert has 552 mutually exclusive
  events), with fills at our price, resolved outcomes, and timing of news jumps.
- **Main failure mode**: this is short tail risk. One leg hitting costs 85-97c per contract,
  so returns are lumpy rather than "consistent". The earlier longshot edge was withdrawn for
  look-ahead and volume bias, so this needs a volume-blind, scheduled-time-anchored test before
  anything else. Ranked last for that reason.

---

## 7. Changes suggested for the codebase (not made)

- `src/albert/fees.py` docstring: Polymarket US effective date is 2026-10-28 22:00 ET, not
  2026-10-19. Add the combo taker fee formula. Note that tiers are a rolling 30-day window from
  2026-11-06.
- `kalshi_fee`: confirm whether `quadratic_with_combo_maker_fees` charges makers on single-market
  fills. The current substring test treats it as a maker-fee series.
- Kalshi parsers: handle `*_dollars`, `*_fp` and `price_ranges` before the 2026-12-17 rollout.
- Maker backtest/paper: replace the trade-through fill rule with a queue-aware rule, and add a
  markout report.
