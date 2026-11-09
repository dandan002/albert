# A Black-Scholes / SABR-style model for prediction-market contracts

Started 2026-11-09. Theory first (§1–§5), then calibration on Kalshi data (§6), then what it is
good for (§7). Code: `src/albert/research/pm_vol.py` (single-contract dynamics),
`pm_smile.py` (strike ladders), `pm_hazard.py` (deadline contracts).

## 1. What plays the role of the underlying

A Kalshi binary pays 1 if an event happens. Its fair price p_t ∈ [0,1] is a conditional
probability, so (ignoring fees, interest and risk premia) it is a **martingale that must end at 0
or 1**. That single fact fixes most of the model:

- total variance is fixed: E Σ (Δp)² over the contract's life = p₀(1 − p₀);
- the only free choices are *how the variance is spread over time* (the clock), *how it depends
  on the price level* (the backbone), whether its scale is itself random (stochastic vol), and
  whether it arrives in jumps.

Three contract shapes need three versions of the same idea:

| Contract | Latent "underlying" | Analogue |
|---|---|---|
| Single binary (game winner, election, "will it rain") | a Gaussian latent score Y; YES iff Y_T > 0 | Bachelier digital option |
| Ladder of "above K" / bracket legs on one number (temperature, totals, scores) | the number X itself; leg K pays 1{X_T > K} | strip of digitals → implied density and smile |
| "Will X happen by date T" | an event time with hazard h_t | reduced-form credit (default by T) |

## 2. M0: the Bachelier digital ("Black-Scholes for binaries")

Latent score with deterministic information flow s(t):

    dY_t = s(t) dW_t,      outcome = 1{Y_T > 0},      V_t = ∫_t^T s(u)² du   (remaining information variance)

    p_t = P(Y_T > 0 | Y_t) = Φ( Y_t / √V_t )

**Pricing PDE.** p(t, y) solves the backward heat equation with a digital payoff,

    ∂p/∂t + ½ s(t)² ∂²p/∂y² = 0,      p(T, y) = 1{y > 0},

i.e. Black-Scholes with r = 0, normal (Bachelier) dynamics and a cash-or-nothing payoff.

**Price dynamics.** In probit units Z = Φ⁻¹(p) = Y/√V, with the *information rate*
λ_t = s_t² / V_t (fraction of the remaining uncertainty resolved per unit time):

    dZ = √λ_t dW + ½ λ_t Z dt                       (drifts away from 0: toward resolution)
    dp = φ(Z) √λ_t dW                               (martingale; price vol σ_p = φ(Φ⁻¹(p)) √λ_t)

With a uniform clock (s constant) λ = 1/τ, τ = time to expiry: price vol grows like 1/√τ near
expiry for prices away from 0/1. This is the volatility formula of Archak & Ipeirotis (2009) and
the in-play model of Stern (1994); Taleb (2018) uses the same construction for elections.

**Exact transition density.** With R = V_t / V_{t+h} ≥ 1 (uniform clock: R = τ/(τ − h)):

    Z_{t+h} | Z_t ~ N( Z_t √R ,  R − 1 )
    P(p_{t+h} ≤ x | p_t) = Φ( (Φ⁻¹(x) − Φ⁻¹(p_t) √R) / √(R − 1) )

so u = (Z_{t+h} − Z_t√R)/√(R − 1) is standard normal under the model. That is the main test
statistic in §6.

**Greeks.** Sensitivities with respect to the latent score and the remaining variance:

| Greek | Formula | Meaning |
|---|---|---|
| delta | ∂p/∂Y = φ(Z)/√V | largest at p = ½ |
| "vega" | ∂p/∂V = −½ Z φ(Z)/V | more uncertainty pulls the price toward ½ |
| theta | ∂p/∂t at fixed Y = ½ Z φ(Z) λ | with no news a favourite drifts up and a longshot down; diffusion offsets it in expectation |

**Backbone.** M0 implies price vol ∝ φ(Φ⁻¹(p)), which near the edges behaves like
p·√(2 ln(1/p)). The logit-normal model, dp = σ p(1−p) dW, behaves like p there instead. The two
are nested as special cases of the power family below.

## 3. M1: "bounded SABR" (stochastic volatility on a bounded martingale)

    dp = α_t · c(τ) · g_β(p) dW,      g_β(p) = [p(1 − p)]^β
    dα = ν α dB,      d⟨W, B⟩ = ρ · sgn(Z) dt

- **β (backbone).**
  - β = 1 is logit-normal (the lognormal analogue).
  - β = ½ is the Jacobi / Wright-Fisher diffusion (the square-root analogue).
  - The probit backbone φ(Φ⁻¹(p)) behaves like β = 1 with a log correction.
  - β = 0 (pure Bachelier on p) is not admissible, because it does not keep p in [0,1].
- **c(τ) (clock).** The deterministic information schedule: uniform 1/√τ, a power law τ^−γ, or a
  category schedule (games, weather hours, speech windows).
- **ν (vol of vol).** How much the information rate itself fluctuates (news bursts, game
  states).
- **ρ (correlation).**
  - The sgn(Z) factor makes it symmetric under relabelling YES/NO.
  - ρ < 0 means vol rises as the price moves toward ½ (rising uncertainty); ρ > 0 means vol rises
    as the price moves toward resolution.
- **M2 (jumps).** Add compensated jumps in Y: scores, news, a speech starting. The jump share of
  variance is reported separately.

The variance identity still holds, so stochastic vol redistributes variance across time and
paths without adding to it.

## 4. Strike ladders: implied density and smile

A ladder of legs "X_T > K_i" on one number X is a strip of digitals, S_t(K) = P(X_T > K). The
Bachelier digital S(K) = Φ((F − K)/σ_K) gives:

- **F:** the implied median (the strike where S = ½);
- **σ_K:** an implied vol per strike, σ_K = (F − K)/Φ⁻¹(S(K));
- **m:** moneyness, m = (K − F)/σ_ATM;
- **smile:** σ(m) = σ₀(1 + s·m + c·m²). Here s is the skew; c > 0 means fat implied tails.

**Link to M0.** For a single leg, Y = X − K and V_t = σ_ATM(t)². The ladder's ATM implied vol
term structure is therefore an **implied information clock**, λ_t = −(d/dt) ln σ_ATM(t)². It
predicts each leg's own price volatility, which gives an implied-vs-realized volatility test
(§6.4).

## 5. Deadline contracts: hazard (credit) model

For "X happens by T", with hazard h_t:

    p_t = 1 − E_t[ exp(−∫_t^T h_u du) ]
    deterministic h:  p_t = 1 − exp(−H(t,T)),   implied average hazard  h̄_t = −ln(1 − p_t)/(T − t)
    no event:  dp/dt = −h_t (1 − p_t)    (theta decay toward 0)
    event:     p jumps to 1              (the compensated jump makes p a martingale)
    strip T₁ < T₂:  forward hazard  h(T₁,T₂) = −ln[(1 − p(T₂))/(1 − p(T₁))]/(T₂ − T₁),   p(T₁) ≤ p(T₂)

A Cox (stochastic-intensity) version adds diffusion from news between events.

## 6. Calibration on Kalshi data

### 6.1 Single-contract dynamics (`pm_vol.py`, report `research-pm-vol-20261109T215533Z`)

**Sample.**
- 4.83M hourly candle steps on 47.5k markets, February–October 2026.
- A step is used only when its mid has a spread of 3–4c or less and lies in [0.03, 0.97].
- Parameters are fit on data before October 1; October is the holdout.

**Time to expiry.**
- `close_time` is look-ahead for markets that close early, so it is not used.
- `expected_expiration_time` is the settlement date, not when information stops arriving: it is
  11–14 h late for weather and about 13 days late for Mentions.
- Time to expiry is measured to a per-series scheduled end, estimated on dev markets only.

| Category | Clock | β | SV half-life; ν per √day | ρ | Jump share (dev / holdout) | var(u) holdout, uniform clock | Excess kurtosis (holdout) |
|---|---|---|---|---|---|---|---|
| Sports in-play | uniform 1/τ (a 0.89, b 0.86) | 0.47 | too short to fit | −0.11 in probit units (t −7.0 / −4.2) | 0.66 / 0.65 | 1.03 | 3 |
| Sports pre-game | buckets + hour of day; ≈ 4% of uniform rate | 0.39 | 9 h; 0.48 | t −2.2 (price units) | 0.74 / 0.72 | 1.30 | 618 |
| Weather | λ = 169·τ^−2.8 + hour of day | 0.58 | 7 h; 0.54 | t −5.7 / −5.7 (price units) | 0.62 / 0.55 | 1.03 | 22 |
| Mentions | buckets + hour of day | 0.63 | 7 h; 0.44 | ≈ 0 | 0.85 / 0.82 | 1.07 | 582 |
| Politics | power law + hour of day | 0.39 | 12 h; 0.33 | t −2.4 / −2.1 (price units) | 0.78 / 0.85 | 1.28 | 1281 |
| Elections | power law + hour of day | 0.81 | 21 h; 0.29 | ≈ 0 | 0.81 / – | dev only | – |
| Entertainment | buckets + hour of day | 0.65 | 13 h; 0.33 | t +3.2 / +2.4 (probit units) | 0.70 / 0.73 | 1.26 | 542 |
| Science and Technology | buckets + hour of day | 0.31 | 12 h; 0.40 | t +3.5 / −0.2 (probit units) | 0.76 / 0.80 | 2.31 | 59 |

**Stylized facts**

1. **Martingale and fixed total variance hold** where the samples are broad.
   - Σ(Δp)² plus the final move to settlement, divided by p₀(1−p₀), is 0.97 / 0.93 for weather
     (dev / holdout) and 0.98 / 1.01 for sports.
   - Over 6 hours, the mean move given p is within ±0.3c, with |t| < 2 in almost every price
     bin.
   - The ratios of 1.2–1.6 in the small categories come from candle samples ranked by volume,
     which selects markets after the fact.
2. **In-game sports follow the Brownian game model** (Stern).
   - With the uniform clock to the scheduled end, var(u) = 1.02 on both dev and holdout, with
     excess kurtosis of about 3.
   - In-game is the only regime where M0's Gaussian shape holds.
3. **Other categories run on their own clocks.**
   - Pre-game sports resolve information at about 4% of the uniform rate, with an intraday
     pattern.
   - Weather is front-loaded into the last ~16 h (λ ∝ τ^−2.8 plus hour of day).
   - Mentions, Politics and Entertainment are mostly hour-of-day profiles.
   - In Politics, Entertainment and Science/Tech, October volatility ran 1.3–2.3× the dev clock.
4. **Moves are lumpy, not Gaussian,** outside in-game sports.
   - Excess kurtosis runs from 20 (weather) into the hundreds.
   - 22–85% of hours show no change in the mid.
   - 55–85% of Σ(Δp)² comes from the jump component.
   - The Gaussian M0 overstates the typical one-hour |Δp| by 2–3×; the M2 mixture fits.
5. **Backbone between Jacobi and probit.**
   - Logit-normal (β = 1) is rejected everywhere: it fits worse than probit with t +3.5 to +7.8
     on dev and loses on holdout in 5 of 7 categories.
   - Fitted β is 0.3–0.8. Out of sample it is better than probit in pre-game sports, in-game
     sports and weather, and worse in Entertainment.
   - Part of the excess volatility near the edges comes from the 1c tick floor.
6. **Stochastic volatility is real but short-lived.**
   - corr(|u_t|, |u_t+1|) within a market is 0.09–0.14, with t 3.5–6.4, and holds on holdout.
   - Half-life is 6–20 h and ν is about 0.3–0.5 per √day.
   - The filtered vol state forecasts next-period variance better than M0 and better than
     recent realized variance (QLIKE, holdout: pre-game t −30 at 6 h, weather −6.1, in-game
     −3.0, Politics −2.2).
   - On MSE the result is the opposite: filtering helps in quiet states and hurts in jumps.
7. **Leverage (ρ) is clear only in-game.** There, volatility rises after moves toward ½
   (probit-unit ρ ≈ −0.11). Elsewhere the sign depends on the units and is unreliable.
8. **Probit drift is 15–75% of what M0 predicts.** An instrumental-variable fit gives the same
   number, so it is not quote noise. The tick floor is the leading suspect; the in-game L2
   follow-up tests this (§6.2).

**Economic note.**
- Quoting at the best bid and ask each hour, with fills inferred from trades through the quote
  and positions held to settlement, loses about 1.3–1.6c per fill.
- Model-based quote gating adds nothing over simply requiring a wide spread.
- Pre-game spreads are 4–38× the expected hourly move, yet makers still lose there: the cost is
  adverse selection on the fills, not diffusion risk.

### 6.2 In-game high-frequency dynamics (L2; report `research-pm-vol-hf-20261109T221912Z`)

**Sample.** Recorded order books for two nights, 50 game-winner legs (NBA 14, NCAAF 14, NFL 10,
NHL 12), about 6,970 in-game minutes.

**Control.** Every statistic is computed twice: on the data and on a matched control. For each
game the control:
- simulates a uniform-clock M0 path from the game's first mid;
- quotes it on the 1c grid with that game's own spreads;
- applies the same cuts.

The control shows what ticks and spreads alone produce.

1. **Variance scales like Brownian motion; the shape does not.**

   | Horizon | var(u), data | Excess kurtosis, data | var(u), control |
   |---|---|---|---|
   | 1 s | 0.71 | 151 | 2.6 |
   | 5 s | 0.77 | 51 | 1.4 |
   | 15 s | 0.77 | 24 | 1.16 |
   | 60 s | 0.73 | 9 | 1.02 |
   | 300 s | 0.77 | 2 | 1.08 |

   - **By league** the level differs but stays flat across horizons: NBA 1.17–1.33, NCAAF
     0.47–0.54, NFL 0.57–0.64, NHL 0.57–0.67.
   - **Different clock from §6.1.** Here τ runs to the scheduled settlement, not to the game
     clock; §6.1's hourly 1.02 used a per-series end. This probably explains why the levels
     differ by league.
2. **No microstructure noise.**
   - Realized variance of the mid is flat at 453–524 c² per in-game hour from 1 s to 600 s,
     and the fitted noise is about 0.
   - The control shows 0.27c of bounce: 1,148 c²/h at 1 s against 608 at 60 s.
   - The Kalshi mid is *sticky*, not bouncing across the tick.
   - The only structure is a small rise from 1 s to 10 s: quotes take a few seconds to adjust.
3. **Jumps dominate in-game variance.**
   - Bipower jump share is 59% / 53% / 48% at 5 / 10 / 15 s, against 16% / 9% / 8% in the
     control: about 40–45 points of genuine jump variance.
   - Threshold jumps (|u| > 4 local sd and |Δp| ≥ 2c) carry 51–68% of realized variance,
     against about 1% in the control.
   - There are about 7 jumps per game-hour, with a median size of 3.5c.
   - **Stern scaling.** The slope of log|Δp| on log φ(Z)/√τ, where 1 is predicted for a fixed
     score increment:
     - NBA 1.08 ± 0.09;
     - NFL 0.50, NCAAF 0.56 and NHL 0.65, flatter;
     - pooled 0.69 ± 0.07.

     So basketball's frequent small scores are Brownian. In the lower-scoring sports a score's
     size in probability depends less on the game state than Stern's model predicts.
4. **The probit-drift puzzle of §6.1 is resolved.**

   | Sample | Data slope | Tick-grid control slope |
   |---|---|---|
   | Weather (hourly) | 0.57 | 0.32 |
   | Mentions | 0.42 | 0.48 |
   | Elections | 0.35 | 0.37 |
   | Science and Technology | 0.47 | 0.53 |
   | Entertainment | 0.35 | 0.71 |
   | Politics | 0.74 | 1.03 |
   | Sports pre-game | 0.11 ± 0.08 | 0.93 |
   | In-game, 60 s | 0.46 ± 0.21 | 1.37 |

   - **Ticks plus clipping at the edges explain it** for weather, Mentions, Elections and
     Science/Tech, and partly for Entertainment and Politics.
   - **They do not explain sports.** There the cause is non-Gaussian increments:
     - in the diffusive core (|u| ≤ 2), squared Z-moves are only 8% of the clock's variance
       pre-game and 34% in-game, and the rest arrives in jumps;
     - within that core the outward drift is 1.0–2.3× what a martingale needs given the core's
       own variance.
   - **The price itself does not drift.** E[sgn(Z)·Δp] is +0.003c per hour pre-game (t 1.3)
     and −0.004c per minute in-game (t −0.1).
   - **Conclusion:** M0's Gaussian drift overstates the compensation when the variance comes
     in jumps. A jump-diffusion version (M2) has the right drift.
5. **No leverage at minute scale.**
   - In probit units, ρ at 1, 2 and 5 minutes is +0.016 / +0.003 / +0.045, all with |t| ≤ 1.4.
   - The price-unit negative values are reproduced by the control: φ(Z) rises toward ½.
   - So the hourly in-game ρ < 0 from §6.1 is an hour-scale effect.

### 6.3 Strike ladders: smile, its dynamics, calibration (`pm_smile.py`, report `research-pm-smile-20261109T220959Z`)

**Sample and method.**
- 24,192 snapshots of 13.2k ladder events, at fixed leads before `expected_expiration_time`:

  | Group | Events |
  |---|---|
  | Weather highs | 1,531 |
  | Weather lows | 836 |
  | Sports totals | 3,778 |
  | Sports team totals | 1,040 |
  | Sports spreads | 1,760 |
  | Politics | 115 |
  | Entertainment | 82 |
  | Science/tech | 33 |

- Quotes come from the last hourly candle closed before the snapshot.
- Only legs with spreads of 5c or less are used.
- The survival curve is forced to be monotone.
- σ_ATM comes from the interquartile range, because football key numbers distort the local slope.
- The realized X_T is the interval implied by which legs settled YES.

**Smile shape.** The fit is σ_K/σ_ATM − 1 = a + s·m + c·m², pooled, with 95% intervals.

| Group | Skew s | Curvature c | Tail heaviness, implied vs realized (skew-t; normal = 1.0) |
|---|---|---|---|
| Sports totals | +0.06 to +0.075 (right skew) | 0.014–0.034 | 1.18–1.20 vs 1.03–1.08 |
| Team totals | ≈ +0.045 | 0.02–0.05 | 1.25–1.40 vs 1.06 |
| Spreads | +0.01 to +0.02 | 0.035–0.055 | – |
| Weather highs | −0.017 (slight left skew) | 0.054 | 1.23–1.29 vs 1.04 |
| Politics | ≈ 0 | ≈ 0.07 | – |

- Every ladder has a **smile** (c > 0).
- The implied **skewness matches realized**: totals imply 0.10–0.12 and realize 0.07–0.12.
- **Implied tails are fatter than realized ones.** This is the cross-strike form of the
  favourite-longshot bias.

**Smile dynamics: the SABR backbone question.**
- *Fixed strike.* When the implied median F moves, the vol at a fixed strike moves with it. The
  sticky-moneyness coefficient is:
  - spreads 1.0;
  - totals 1.3;
  - weather highs 1.17;
  - team totals 2.1.

  All have t 9–25. Sticky strike, a coefficient of 0, is rejected everywhere.
- *ATM vol.* σ_ATM does not slide along a fixed smile. Its response to ΔF is about 0 (totals
  +0.017, t 1.4), whereas sticky strike predicts +0.065.
- *Backbone exponent.* Regressing log σ_ATM on log F gives:
  - within a game: 0.28 for totals (t 3.2) and −0.12 for team totals;
  - across games: 0.22–0.29.

  Both are below the Poisson value of 0.5 for a count of points.

**ATM term structure.** The exponent of σ_ATM against lead; Brownian is 0.5.

| Group | Exponent |
|---|---|
| Weather highs | 0.77 (information concentrated late) |
| Weather lows | 0.48 |
| Politics | 0.35 |
| Science/tech | 0.24 |
| Entertainment | 0.19 |
| Sports, pre-game | ≈ 0 (nothing is learned before the game) |

**Calibration against realized X_T.**
- *Width ratio* (realized dispersion / σ_ATM, where 1 is calibrated):

  | Group | Dev | Holdout | Reading |
  |---|---|---|---|
  | Sports | 1.02–1.14 | 1.02–1.03 | calibrated |
  | Weather highs | 0.94 (t −2.4) | 0.89 (t −3.8) | implied too wide |

- *PIT.* Uniform for sports. Weather puts too much mass in the tails.
- *Tail mass at |m| ≥ 2.* Weather implies 0.9–1.6c against 0–0.4c realized, with every z
  negative. Entertainment and politics wings are overpriced in small samples. Sports is mixed.
- *Location.* Realized sports totals land above the implied median by +0.12σ on dev (t 5.1) and
  +0.06σ on holdout (t 2.4): an "under" bias.

**Economic note.**
- Selling tail legs at |m| ≥ 2.5 earns +1.1c per contract on holdout (t_risk 2.0) but only +0.2c
  on dev.
- The gain is capped near 0.93c per contract by the 1c tick.
- Adding moneyness to a price-only model barely changes holdout log loss for weather.
- Arbitrage net of fees appears at 0–0.3% of snapshots, mostly stale quotes.

### 6.4 Implied vs realized variance (variance-swap analogue)

**What is compared.** Between two snapshots of the same ladder:
- the model says the implied median F is a martingale;
- its realized variance, E[(F_{t+h} − F_t)²], should therefore equal the implied variance drop
  E[σ_t² − σ_{t+h}²].

**Ratio realized / implied:**

| Group and window | Ratio | Reading |
|---|---|---|
| Sports totals, in-game, 6 h → 1 h before expiry | 0.90 [0.81, 0.98] | slight variance premium |
| Sports spreads, same window | 0.92 [0.84, 1.04] | slight variance premium |
| Team totals, same window | 0.65 [0.59, 0.70] | strong variance premium |
| Weather highs, overnight (24 h → 16 h) | 0.66 | implied variance released ahead of the news |
| Weather highs, afternoon (12 h → 8 h) | 1.55 [1.37, 1.74] | realized exceeds implied |

**Weather medians are not martingales.** They drift down overnight and up in the daytime
(t 3–6): the market under-forecasts the daytime high until the observations arrive.

**Link to §6.1.**
- The ladder's term-structure exponent of 0.77 for weather highs gives V ∝ τ^1.54, an implied
  information rate of about 1.5/τ.
- The single-contract realized clock is steeper still (λ ∝ τ^−2.8).
- Both say weather information is back-loaded, but they measure time to expiry differently:
  §6.1 uses a per-series end of information, §6.3 the settlement time.

### 6.5 Deadline contracts: hazard term structure and theta (`pm_hazard.py`, report `research-pm-hazard-20261109T222119Z`)

**Sample.**
- 5,872 "by date T" contracts with deadlines from 2026-01-01 to 11-02, giving 26,290 priced
  observations in 709 events.
- T is parsed from the rules text, never from close times. 98% of NO markets close within 26 h
  of the parsed deadline.
- Classes, with an example of each:

  | Class | Markets | Events | Example |
  |---|---|---|---|
  | One-off events | 2,180 | 548 | "Gemini Pro before Oct 16" |
  | Mentions | 1,940 | 71 | "Trump says 'Epstein' before Nov 2" |
  | Eliminations | 1,449 | 88 | "eliminated from DWTS before Oct 28" |
  | Counts | 303 | 52 | "endorse ≥ 5 this week" |

- Competing-leg, measured-at-T and scheduled-vote markets are excluded.

**Term structure.** 3,019 daily snapshots of 194 strips. The shape is the last forward hazard
divided by the hazard to the first deadline.
- Overall: 48% rising, 35% front-loaded, 15% flat, 2.4% inverted (median ratio 1.20).
- By category:

  | Category | Ratio | Shape |
  |---|---|---|
  | Science and Technology (release dates) | ≈ 2.0 | rising |
  | Climate | 1.7 | rising |
  | Politics | 1.10 | mixed |
  | Entertainment, Sports, Elections | 0.70–0.89 | front-loaded |

- **Along one contract with no event, the implied hazard climbs toward the deadline.** Relative
  to 7 days out:

  | Time left | Implied hazard |
  |---|---|
  | 72 h | 1.4× |
  | 24 h | 2.3× |
  | 6 h | 7× |
  | 1 h | about 30× |

  Mentions climb less: 1.2× / 1.4× / 3.8× / 10×.

**Calibration.** k is realized hazard divided by implied hazard; k < 1 means the price is too
high.

| Lead | < 6 h | 6–24 h | 1–3 d | 3–7 d | 7–14 d | 14–30 d | 30–60 d |
|---|---|---|---|---|---|---|---|
| k | 0.60 | 0.60 | 0.73 | 0.82 | 0.73 | 0.74 | 0.65 |

- Every 90% interval is below 1. Realized minus mid is −1.3 to −7.2c, with t −3.5 to −8.
- Longshots priced 5–15c have k 0.34–0.82. Below 5c the 1c tick floor dominates.

**Theta: the price is sticky.**
- With no event, prices fall at only 0.28–0.69 of the flat-hazard rate (gap t 3.6–11.5).
- Contracts at 15–50c a week out still sit at:

  | Time left | Market | Model |
  |---|---|---|
  | 24 h | 10.4c | 4.2c |
  | 6 h | 6.5c | 1.1c |
  | 1 h | 3.2c | 0.2c |

- Jumps to YES are rarer than implied: in the last 6 h, 18 actual against 74 expected.
- Net drift is −0.86c per hour in the last 6 h (t −7.2) and about zero before that. This is the
  one place a deadline price is not a martingale.

**Event timing.**
- YES resolutions arrive later in the window than a flat hazard predicts: mean PIT 0.61, t 7.7.
  15.7% fall in the last 10% of the window against 6.4% predicted, and 20% in the final day.
- Eliminations are scheduled reveals: PIT 0.95.
- Events do bunch toward the deadline, but less than prices assume.

**Variance decomposition.** Hourly price variance up to settlement, 4,032 markets:

| Component | Share |
|---|---|
| Diffusion | 34% |
| Pre-resolution news jumps (≥ 20c per hour, about 0.5 per market) | 42% |
| YES resolution jump | 23% |
| NO deadline step | 1% |

- Total variance is 1.19 × p₀(1−p₀).
- Implied-hazard vol is about 0.41–0.57 per √day beyond 3 days and 1.3–2.0 within 3 days.
- A mean-reverting (CIR) intensity does not fit: the fitted mean reversion is negative because
  the implied hazard trends up. The model needs a deadline-shaped deterministic hazard plus
  diffusion.

**Static arbitrage.**
- Near deadlines are priced above far ones in 3.4% of pair-hours (Climate 15%).
- That is executable after fees in only 0.04–0.12%.
- Forward segments are overpriced: 0.27 implied against 0.21 realized.

**Economic note.**
- The hazard model adds nothing to the price on holdout (blend coefficient 0.006, t 0.04).
- Its dev rule, +19c, falls to −5.3c on 49 holdout trades.
- Buying NO on every YES at 50c or less earns about +1.2–1.3c (holdout t 1.65).

## 7. Synthesis: the stylized facts of prediction-market prices

1. **Level and total variance are fair.**
   - Prices behave as martingales, and the variance over a contract's life matches p₀(1−p₀).
     The ratio is 0.93–1.01 for single contracts in the broad categories and 1.19 for
     deadline contracts, partly from quote bounce.
   - Sports ladder medians are unbiased within 0.06–0.12σ.
   - The exceptions are local:
     - weather ladder medians cycle with the day (down overnight, up in the daytime);
     - deadline contracts drift down only in their last ~6 h.
2. **The information clock is the prediction-market analogue of implied volatility.**
   - A single price says nothing about volatility. Its dynamics are set by *when* the remaining
     p(1−p) of variance arrives.
   - The clock can be read two ways:
     - from price paths (realized, §6.1);
     - from a ladder's ATM term structure (implied, §6.3) or a deadline strip's hazard curve
       (§6.5).
   - Clocks by market type:

     | Market type | Clock |
     |---|---|
     | In-game sports | uniform in time (Stern's Brownian game) |
     | Weather | back-loaded into the last ~16 h, with a time-of-day profile |
     | Pre-game sports | nearly flat (nothing is learned before kickoff) |
     | Deadline contracts | implied hazard rising steeply into the deadline |

   - Implied and realized agree in shape. In level, implied is mostly higher:
     - in-game ladder variance premium, realized/implied 0.65–0.92;
     - weather 0.66 overnight against 1.55 in the afternoon;
     - deadline hazards k ≈ 0.6–0.8.
3. **Variance arrives in jumps.**
   - Outside in-game sports at the hourly scale, 50–85% of variance is jumps (scores, news,
     resolutions), and excess kurtosis runs from 20 into the hundreds.
   - In-game, minute-scale variance is still about half jumps, about 7 per game-hour.
   - The Gaussian M0 gets the clock right and the shape wrong. M2, a jump-diffusion on the
     latent score, is the minimal adequate model. It also explains why the probit drift falls
     short of M0.
4. **Backbone.** Price vol as a function of level is ∝ [p(1−p)]^β with β ≈ 0.3–0.8. That is
   closer to Wright-Fisher/Jacobi (½) than to logit-normal (1, rejected everywhere). Across
   strikes, the smile moves with the implied median (sticky moneyness, coefficient 1.0–2.1)
   rather than staying fixed in strike.
5. **Stochastic volatility is real but short-lived.**
   - Half-life is 6–20 h, and vol of vol ν is about 0.3–0.5 per √day.
   - The filtered vol state improves variance forecasts.
   - Leverage (vol rising as the price nears ½) appears only in-game at the hourly scale.
6. **The one systematic mispricing is a single phenomenon seen three ways: tails and time are
   overpriced.**
   - Fat implied tails across strikes: skew-t tail index 1.2–1.4 implied against about 1.05
     realized. Weather ladders are too wide overall.
   - Implied deadline hazards too high (k 0.6–0.8), with prices decaying at only 0.3–0.7 of the
     model's theta, and forward segments overpriced.
   - The favourite-longshot bias found in earlier single-contract work.

   All are bounded by the 1c tick floor. Each is a few cents at most, and none survives fees
   as a trading rule out of sample.
7. **Microstructure.**
   - The Kalshi mid is sticky: it does not bounce, and its variance signature is flat from
     1 s to 10 min.
   - Quotes adjust over a few seconds.
   - Near 0 and 1, the 1c grid explains most of the departure from continuous-model drift
     outside sports.

**The model that fits** ("bounded jump-SABR with an information clock"):

    single binary / ladder leg:   p_t = P(Y_T > K | 𝓕_t),   dY = α_t s(t) dW + J dN_t,
                                  dα = ν α dB,   s(t) category clock,   J and N_t category jump law
    deadline contract:            p_t = 1 − E_t[exp(−∫_t^T h_u du)],   h_u = h_det(T − u) · ξ_u,
                                  h_det rising into the deadline,   ξ a log-diffusion (vol ≈ 0.4–0.6 per √day)

**Market price versus fair price.** The market price departs from the model's fair price in two
places:
- a tail/time premium, which the tick floor caps;
- in weather, a time-of-day bias in the median.

## 8. Open questions

- **Game clock.** Use the game clock and score (ESPN API, which is permitted) instead of time to
  scheduled settlement. This should explain the league differences in §6.2 and test the
  Stern/Skellam jump laws directly.
- **More in-game data.** Two nights and 50 games is small. Rerun the high-frequency study as
  recording continues.
- **Why theta is sticky.** Who holds overpriced deadline longshots into the last day? Possible
  explanations are lottery preference, inattention, and Kalshi fees making the NO side
  unattractive at 97–99c. Testable with trade-side data (`taker_side`).
- **Weather median cycle.** Is it a forecast-model artefact, such as NBM under-forecasting the
  daily maximum, or market behaviour?
- **Multi-maturity digitals on one underlying,** e.g. first half vs full game: a true calendar
  spread for the information clock.
