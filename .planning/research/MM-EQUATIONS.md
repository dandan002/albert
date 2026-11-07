# Market-making equations and short-horizon strategies for binary contracts in volatile regimes

Research date: 2026-11-05. Scope: the equations behind inventory-based and information-based
market making, how they change when prices are bounded in [0,1] and settle at 0 or 1, what the
empirical literature on prediction markets and in-play betting exchanges says, and how
professional makers handle volatile (in-play, news) windows. It ends with five testable
strategy specifications for Albert's Kalshi setup.

It builds on `.planning/research/market-making.md` (2026-11-04), which covers incentive programs,
fees and venue mechanics in detail. This file does not repeat that material; it adds the
equations, the estimators for our data, and the volatile-regime evidence.

**Tags** (same convention as market-making.md, plus two):

- **[V]** Verified today against the primary source (paper PDF or official page).
- **[S]** Secondary source, not confirmed against a primary.
- **[L]** Standard reference, cited from the literature. Not re-read today.
- **[D]** Derived here from the cited model. The algebra is mine; check it before relying on it.
- **[M]** Measured today on our own recorded data. A quick look, not a study.
- **[I]** My inference, a hypothesis to test.

**Our setup** (the constants every formula below is evaluated at):

- Kalshi, 1c ticks, 1-lot orders. YES bids and NO bids form one book: a NO bid at 1−a is a YES
  ask at a.
- Event to order at the exchange takes about 50 ms (measured, SUMMARY.md).
- Fees per contract: taker 0.07·P(1−P), which is 1.75c at 50c and 0.63c at 90c. Maker
  0.0175·P(1−P), only on maker-fee series, which is 0.44c at 50c and 0.16c at 90c.
- Write budget [V: https://docs.kalshi.com/getting_started/rate_limits]:
  - Basic tier: 100 tokens/s. A create or cancel costs 10 tokens (10 writes/s); a batch cancel
    costs 2 tokens per order (50 cancels/s). The simulator's `writes_per_s=10, cancel_cost=0.2`
    matches this.
  - **Advanced tier is 300 tokens/s, self-service through the "Upgrade Account API Usage Level"
    endpoint.** The page lists no other requirement.
- **Order groups** [V: https://docs.kalshi.com/getting_started/order_groups]: a contracts limit
  over a rolling 15-second window. When the limit is exceeded, every resting order in the group is
  cancelled on the exchange side and the group locks until reset. A manual "trigger" cancels the
  whole group in one call. This is the exchange-side kill switch professional makers use (§5.4).

---

## 0. Summary of findings

1. **The inventory models give a sensible scale for our books, and that scale is what the
   Quoter already does.** With intensities fitted to four recorded dev hours [M], A ≈ 1e-3
   fills/s per market at 0.5c from mid, k ≈ 0.33–0.50 per cent, and σ ≈ 0.04 c/√s. The GLFT
   closed form then gives a half-spread of about 2.6c for any risk aversion γ between 0.01 and
   0.3. That is about 1/k: the fill-rate decay sets it, not inventory risk. Improving inside a
   5c spread is already near the risk-neutral optimum, so a full AS/GLFT build can only change
   the inventory skew (0.25–3.5c per contract). §2.2.
2. **Those models assume fills are uninformed; our data say they are not.** Cartea–Jaimungal–
   Ricci's adverse-selection term moves each quote out by the expected post-fill drift in the
   fair value. Ambiguity aversion (Cartea–Donnelly–Jaimungal) replaces 1/κ with
   (1/ϕ)·ln(1+ϕ/κ) and adds an inventory drift penalty, which has the same form as risk
   aversion. §2.3.
3. **For a binary contract, Glosten–Milgrom has a closed form and it is a constant half-spread in
   log-odds** [D]: logit(ask) − logit(p) = ln((1+μ)/(1−μ)), where μ is the informed share. At 50c,
   a 5c spread is break-even for about 5% informed takers. Near 0 or 1 the 1c tick, not
   information, sets the spread. §3.1.
4. **The strongest new empirical paper says Kalshi makers profit by holding to settlement, not
   from the spread** (Bartlett & O'Hara 2026, 41.6M trades) [V]. Spreads cover at most about 6%
   informed takers at 50c. Single-name markets have the most informed trading *and* the highest
   maker profit (+1.91c per contract), because takers systematically over-buy YES. Trailing VPIN
   at 30–70c predicts maker losses in single-name markets only. This matches our own pattern:
   positive markout to mid, negative markout to exit price. §4.1.
5. **In-play, prices absorb goals within seconds.** On Betfair, the reaction to a goal is complete
   and efficient (Croxson–Reade) [V]. Mispricing remains around *surprise* events (an underdog
   scores) and **late in games, where longshots lose heavily even for makers and takers on strong
   favourites earn small significant profits** (Whelan 2025; Page 2012 "Yogi Berra") [V]. §4.3.
6. **Venues protect makers in-play with speed bumps.**
   - Betfair delays every in-play order 5–9 s in soccer, 5 s in tennis and 1 s in racing.
     Cancels are not delayed, and on some markets passive orders bypass the delay. Lengthening
     the soccer delay narrowed spreads and deepened books (Brown–Yang) [V].
   - Polymarket intl cancels all resting orders at game start and delays marketable sports orders
     by 1–3 s [V/S].
   - Crypto.com applies a 3 s delay with makers exempt [S]. Kalshi reportedly proposed a delay in
     Dec 2025; I found no confirmation that it was implemented [S].
   - Without a speed bump, a resting Kalshi quote is exposed for our whole *information* latency,
     which is seconds, not 50 ms. §5.
7. **At our speed, the latency that matters is information latency.**
   - TV is at least 4 s behind live play, satellite and HD add more, and streams lag tens of
     seconds.
   - Official data feeds beat cable TV.
   - Professional makers on Kalshi (SIG, Jump) come from sports betting and keep spreads tight
     in-game but cut displayed depth by about 76% (Blockworks, March Madness) [S].
   - Our 50 ms order path is irrelevant if we learn of a score seconds late. The controls that
     work for us are exchange-side: order groups, cheap batch cancels, and not quoting in
     high-hazard states. §5.3–5.4.
8. **A quick check of the Chakraborty–Kearns ladder identity on our 5-minute buckets** [M]:
   - Their result is that a ladder maker's profit equals (path length − net move²)/2.
   - Volatile buckets have far more path (median 16c against 1.5c) but less mean reversion per
     unit of path. Median path/range² is 1.9 in volatile buckets against 4.0 in calm ones.
     path > range² holds in 58% of volatile buckets against 74% of calm ones.
   - Volatile windows offer more spread to capture but are trendier. §3.6.

---

## 1. Notation

| Symbol | Meaning (our units) |
|---|---|
| p, s | Fair value (mid or micro-price) of YES, in dollars or cents |
| x = logit(p) = ln(p/(1−p)) | Log-odds. S(x) = 1/(1+e^(−x)) maps back; S′(x) = p(1−p) |
| q | Our net YES inventory (contracts) |
| δ_b, δ_a | Distance of our bid and ask from the fair value (cents, or logit units with subscript x) |
| λ(δ) = A·e^(−kδ) | Fill intensity of a quote at distance δ (fills/s) |
| σ | Volatility of the fair value (c/√s; σ_x in logit units) |
| γ | Risk aversion (CARA: 1/(c·contract)); φ running inventory penalty; γ_T terminal penalty |
| h, J | Jump hazard (jumps/s) and jump size of the fair value |
| μ (α in some papers) | Share of takers who are informed |
| f(P) | Fee per contract at price P |

---

## 2. Inventory-based market making

### 2.1 Avellaneda–Stoikov (2008) [L]

The mid follows dS = σ dW. The maker has CARA utility with risk aversion γ and horizon T. Fill
intensity is λ(δ) = A·e^(−kδ).

- Reservation price: r(s,q,t) = s − q·γ·σ²·(T−t)
- Optimal total spread: δ_a + δ_b = γσ²(T−t) + (2/γ)·ln(1 + γ/k)
- Quotes are r ± half of that spread.

Source: https://doi.org/10.1080/14697680701381228 (preprint:
https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf).

What it says for binaries: the γσ²(T−t) term is "variance still to arrive before I can get out".
For a contract held to settlement, the remaining variance is exactly p(1−p) (market-making.md
§1.3). Replace σ²(T−t) with the variance expected to arrive *before we can cancel or exit*, not a
constant.

### 2.2 Guéant–Lehalle–Fernandez-Tapia (2013): closed forms with bounded inventory [V]

GLFT solve the AS problem with an inventory cap |q| ≤ Q, where the side that would breach the
cap is switched off. They give asymptotic closed forms (their eqs. 4.6–4.7), written as in the
hftbacktest implementation [V: https://hftbacktest.readthedocs.io/en/latest/tutorials/GLFT%20Market%20Making%20Model%20and%20Grid%20Trading.html]:

```
c1 = (1/(ξΔ)) · ln(1 + ξΔ/k)
c2 = sqrt( (γ / (2·A·Δ·k)) · (1 + ξΔ/k)^(k/(ξΔ) + 1) )
δ_b(q) = c1 + (Δ/2)·σ·c2 + q·σ·c2          (bid distance below fair value)
δ_a(q) = c1 + (Δ/2)·σ·c2 − q·σ·c2          (ask distance above fair value)
half-spread = c1 + (Δ/2)·σ·c2 ;   skew per contract = σ·c2
```

Here Δ is the order size (1), ξ = γ for the CARA model, and ξ = 0 gives the risk-neutral model
with a running quadratic penalty. In that case c1 → 1/k and (1+ξΔ/k)^(k/ξΔ) → e.

Source: https://arxiv.org/abs/1105.3115 (Math. Fin. Econ. 7(4), 2013). Guéant generalizes the
results to any intensity shape and to multiple assets in https://arxiv.org/abs/1605.01862 [L].

**Estimating every parameter from our data** (the hftbacktest recipe, adapted):

- **A, k.** For each trade print, take the depth δ_i = |trade price − pre-trade mid|, signed
  positive when the taker crossed the mid. Use the mid *before* that trade's execution deltas: our
  stream labels them `is_exec`, and about 90% arrive before the trade print. Count the trades
  with δ_i ≥ δ on a grid of δ = 0.5, 1, …, 4c, divided by market-seconds. Fit
  ln λ(δ) = ln A − kδ by OLS over the shallow range only. hftbacktest found the full-range fit
  understates intensity near the mid. Do this per series, calm and volatile separately.
- **σ.** 1-second mid increments, kept only while the spread is ≤ 6c at both ends, so quote
  flicker in wide books does not count. Use bipower variation, (π/2)·mean(|r_t||r_{t−1}|), for
  the diffusive part. The difference from the plain variance is the jump part (§2.5).
- **γ.** Choose it so the skew per contract (σ·c2) is a sensible fraction of the spread. Not
  estimable; sweep it.
- **Quick calibration [M]**, 2026-11-05 05–09 UTC (dev), 1,346 trades over 112 markets that
  traded, 92 markets with enough tight-spread seconds:
  - All windows: A ≈ 8.7e-4 /s, k ≈ 0.40 /c.
  - Calm buckets: A ≈ 1.2e-3 /s, k ≈ 0.33 /c.
  - Volatile buckets: A ≈ 5.7e-3 /s, k ≈ 0.50 /c, so about 5× the fill rate.
  - Median σ ≈ 0.039 c/√s (std) and 0.013 c/√s (bipower). Jumps of 3c or more in one second:
    about 0.13 per market-hour overnight.
  - 69% of prints trade exactly at the touch.
  - These numbers are ±50%: four night hours, pooled.
- **Worked GLFT quotes** with A = 8.7e-4, k = 0.40, σ = 0.039:

  | γ | c1 | half-spread | skew per contract held |
  |---|---|---|---|
  | 0.01 | 2.47c | 2.59c | 0.25c |
  | 0.1 | 2.23c | 2.64c | 0.82c |
  | 0.3 | 1.87c | 2.64c | 1.56c |
  | 1.0 | 1.25c | 3.03c | 3.55c |

  The half-spread is about 2.6c for any reasonable γ, so a 5c quoted spread is what the model
  wants. That is what `Quoter(improve=True, min_spread=0.05)` already does. A GLFT build adds the
  inventory skew and switching off the capped side, and nothing else in calm hours [I].
- **What GLFT cannot see.** It assumes fills are independent of the fair value's next move. Our
  improve@5c fills are filled about 2.5c from mid, yet mark only +1.7c to mid after 5 minutes.
  So about 0.8c of adverse drift per fill [M, from SUMMARY.md]. §2.3 adds that.

### 2.3 Cartea–Jaimungal–(Penalva/Ricci/Donnelly) extensions

**Running inventory penalty, risk-neutral maker** [V: Cartea, Donnelly & Jaimungal 2017,
Prop. 2.1, https://ora.ox.ac.uk/objects/uuid:32248d98-3e1a-499a-ae2f-8cddd64beb4c]. The maker
maximizes E[X_T + q_T(S_T − ℓ(q_T)) − φ∫q²dt], with fill probability e^(−κδ):

```
δ+*(t,q) = ( 1/κ+ − h(t,q−1) + h(t,q) )+        (sell depth)
δ−*(t,q) = ( 1/κ− − h(t,q+1) + h(t,q) )+        (buy depth)
```

h solves a system of ODEs. For symmetric κ, h = (1/κ)·ln ω with ω(t) = e^(A(T−t))·ω(T), where A
is a tridiagonal matrix in q. Read 1/κ as the risk-neutral optimum: maximize δ·e^(−κδ). The h
differences are the inventory skew.

**Adverse selection, short-term alpha, self-exciting order flow** [V: Cartea, Jaimungal & Ricci
2014, SIAM J. Fin. Math., "Buy low, sell high", text read from
https://c.mql5.com/forextsd/forum/169/a_high_frequency_trading_perspective.pdf; republished in
SIAM Review 2018, https://epubs.siam.org/doi/10.1137/18N974601]:

```
dS_t  = (υ + α_t) dt + σ dW_t
dα_t  = −ζ α_t dt + σ_α dB_t + ε+ dM+_t − ε− dM−_t + (news jumps)          (short-term alpha)
dλ±_t = β(θ − λ±_t) dt + η dM±_t + ν dM∓_t + (news terms)                  (Hawkes activity)
δ±*   = δ0± + α·δα± + ε·δε± + φ·δφ± + o(ς),   with δ0 solving δ h′(δ) + h(δ) = 0
```

For exponential fill probabilities, δ0 = 1/κ. In this model the market orders themselves move
the drift. The adverse-selection terms push the quote on the side that is about to be picked off
*further* out and pull the other side *in*, so the maker can take the first leg of a
directional trade.

The two terms that matter, under symmetric assumptions (their eq. 17):

```
E[∫_t^T α_u du] = ε(λ+ − λ−)·[ (1−e^(−β̂τ))/β̂ − (e^(−ζτ)−e^(−β̂τ))/(β̂−ζ) ]/ζ + α_t(1−e^(−ζτ))/ζ
b_φ             = 2h(λ+ − λ−)·[ τ − (1−e^(−β̂τ))/β̂ ]/β̂² ,   β̂ = β − ρ(ν−η),  τ = T−t
```

Both terms scale with **λ+ − λ−, the imbalance in buy versus sell market-order activity**.
In a one-period sketch [D]: if the fair value moves by an expected ε after a fill on our side,
maximizing (δ − ε)·e^(−κδ) gives δ* = 1/κ + ε. Every cent of expected post-fill drift moves the
optimal quote out by one cent.

Their simulation finding [V]: makers who cannot forecast short-term alpha are driven out. They
survive only with a much higher inventory penalty, which pushes quotes away from the mid.

**Market making with alpha signals** (Cartea & Wang 2020, IJTAF, https://doi.org/10.1142/S0219024920500168) [S].
A latent alpha is pushed up by buy market orders and down by sell market orders. The maker
skews quotes with the signal, and expected profit from the signal rises with inventory
tolerance.

**Ambiguity aversion** [V: CDJ 2017, Prop. 3.2 eq. 24]:

```
δ±*(t,q) = ( (1/ϕ_κ)·ln(1 + ϕ_κ/κ±) − h(t,q∓1) + h(t,q) )+
η*(t,q)  = α − ϕ_α·σ²·q        (worst-case drift the maker plans against)
```

- **ϕ_κ (doubt about fill probabilities)** turns the risk-neutral 1/κ into the same
  (1/γ)·ln(1+γ/κ) term as AS risk aversion.
- **ϕ_α (doubt about the drift)** acts as an extra inventory penalty that scales with σ².
- Closed form (Prop. 3.4): with ϕ_λ = ϕ_κ = ϕ, ω(t) = e^(A(T−t))·ω(T), with
  A = diag[ξ±, ακq − ½κϕ_ασ²q²] and ξ± = (1+ϕ/κ)^(−(1+κ/ϕ))·λ±.
- Reported: robustness raises the Sharpe ratio over some parameter ranges.
- **Practical reading [I]:** estimate κ in calm windows and treat volatile windows as κ
  uncertainty, so raise ϕ_κ. This widens quotes without a full jump model.

**Adverse selection with "price reading"** (Barzykin, Bergault, Guéant & Lemmel 2025,
https://arxiv.org/abs/2508.20225) [V]. Each fill moves the reference price by an impact
function ζ̃. The quadratic approximation with exponential intensities has a closed form:

```
δ̂ = 1/κ + σ·sqrt( γ·e / (2·Σ Δ·Λ0·κ) ) · (±q + Δ/2)
```

This is GLFT with ξ = 0. Price readers (skew sniffers) get a reduced skew.

### 2.4 Bounded prices: logit diffusions, settlement, jumps

**Logit jump-diffusion** (Dalen, https://arxiv.org/abs/2510.15205) [V].

The model:

```
p = S(x),  dx = μ(t,x)dt + σ_b dW + ∫ z Ñ(dt,dz)
μ(t,x) = −[ ½ S″(x) σ_b² + ∫ (S(x+z) − S(x) − S′(x) z 1{|z|<1}) ν(dz) ] / S′(x)   (makes p a martingale)
```

AS in logit units:

```
r_x = x − q·γ·σ̄_b²·(T−t);   2δ_x ≈ γσ̄_b²(T−t) + (2/k)·ln(1+γ/k)
p_bid = S(r_x − δ_x),  p_ask = S(r_x + δ_x);   δ_p ≈ p(1−p)·δ_x
|q| ≤ q_max ∝ 1/max(p(1−p), ε)        (inventory cap widens near 0 and 1; needs a tick floor)
```

Estimating σ_b, h and J (Dalen §calibration):

1. Clamp p to [ε, 1−ε], use a uniform grid, drop crossed books, and Kalman-filter
   y = logit(p̃) = x + η, where the noise variance depends on spread, depth, trade rate and
   imbalance.
2. Run **EM on the increments** as a mixture of N(μΔ, σ_b²Δ) and jumps with probability hΔ.
   - E-step: posterior jump probability γ_t = hΔψ_t / (hΔψ_t + (1−hΔ)φ_t).
   - M-step: σ̂_b² = Σ(1−γ)(Δx−μΔ)²/Σ(1−γ)Δ; ĥ = mean(γ)/Δ; ŝ_J² = Σγ(Δx)²/Σγ.
3. Quoting rules: widen or pull on a toxicity spike, pre-widen before scheduled news, pause when
   the jump detector fires, and add a jump premium proportional to recent jump variance.

Caveat: the paper is validated on synthetic paths only.

**Optimal MM in prediction markets** (Feil & Nendel, https://arxiv.org/abs/2607.17991) [V].

The model:

- Latent belief L with dL = μdt + σdW and p = f(L), where f is logistic. In the numerics,
  dp = p(1−p)·σ̃·dW. Settlement Y ~ Bernoulli(p_T).
- The maker maximizes E[X_T + q_T·Y − γ∫q²ς²ds − γ_T·q_T²·p_T(1−p_T)]. The terminal penalty is
  the conditional variance of the settlement payoff.
- Intensities scale with √(p(1−p)) and with factors that vanish at the boundaries, e.g. the bid
  multiplier (2π/(π+p))^ν·e^(−k(p−π)).
- No closed form; HJB solved numerically, with V(t,p,q) after removing cash and the mark-to-model
  value.

Results:

- Spreads are largest near p = ½ and widen near settlement.
- At zero inventory quotes skew *toward ½*, because flow vanishes at the boundaries.
- Inventory risk fades as p(1−p) → 0.
- γ_T matters most near settlement.
- Against a myopic benchmark: same mean PnL (12.39 vs 12.47) at about ⅓ the standard deviation
  and ⅛ the 5% VaR.

**Jump-to-default analogy** [D]. In reduced-form credit, spread ≈ hazard × loss given default.
For a binary maker, the analogue is a **quote-or-not inequality** per side:

```
λ_u(δ)·(δ − f(p))  >  h · E[(|J| − δ)+ · 1{J against this side}] · π_pick  +  inventory penalty
```

- λ_u is the fill rate from *uninformed* takers at distance δ from fair.
- h and J are the hazard and size of jumps in the fair value.
- π_pick is the probability that someone hits our stale quote before we cancel. π_pick ≈ 1
  whenever the fastest informed taker learns of the event earlier than our cancel lands.
- With symmetric jumps and two-sided quotes, each side carries half the hazard.
- In-play, h is the scoring rate and J comes from the score model below. It is the cost of
  standing in front of a scoring event for one information-latency.

**In-play score models give J(t, state)** [L].

- **Brownian (basketball-like)**, Stern 1994 (https://doi.org/10.1080/01621459.1994.10476851) and
  Polson & Stern 2015 (https://doi.org/10.1515/jqas-2014-0095). The lead is X(t) = μt + σB(t),
  t ∈ [0,1]:

  ```
  p(t, ℓ) = Φ( (ℓ + μ(1−t)) / (σ·sqrt(1−t)) )
  J(Δℓ) ≈ φ(z)·Δℓ / (σ·sqrt(1−t))            [D]
  σ from the pre-game line: σ = μ / Φ⁻¹(p_0)
  ```

  The same score is worth more in probability as time runs out, and most when the game is close
  (φ(z) is largest). Late close games are the high-hazard state. Blowouts are low-hazard but sit
  near the tick floor.
- **Skellam (soccer/hockey-like)**, Feng, Polson & Xu (https://arxiv.org/abs/1604.03614) [V]:
  N(t) = N_A(t) − N_B(t) ~ Skellam(λ_A·t, λ_B·t), and P(win | ℓ) = Σ_{x > −ℓ} Skellam(x;
  λ_A(1−t), λ_B(1−t)). The rates are calibrated from pre-game win/draw/loss prices and
  re-estimated in play. Goals are rare and huge jumps.

### 2.5 Reduction to our action set [D]

With 1c ticks, 1-lot size, and wide books, the continuous δ of every model above collapses to
a few actions per side: improve by 1c, join (only with ≥ 20 contracts behind, the
holdout-confirmed "behind20"), step back 1c, or no quote.

The operational rule evaluates, for each candidate price π:

```
EV(π) = P_fill(π, T_rest) · [ |π − fair| − E[adverse move | fill at π] − f(π) ] − γ·q·(skew term) − jump term (§2.4)
```

All terms come from our data:

- P_fill from the simulator's queue model.
- The adverse move from our own fills' mo markouts by spread, price and regime.
- The fee exactly.

Quote the best positive action; otherwise pull. GLFT and CJR say how these terms scale. The
replayed data says what they are.

---

## 3. Information models

### 3.1 Glosten–Milgrom (1985) on a binary contract [L, D]

Source: https://doi.org/10.1016/0304-405X(85)90044-3.

Setup: V ∈ {0,1} with prior p. A share μ of takers know V. The rest buy or sell with probability
½ each. Zero-profit quotes are ask = E[V | buy] and bid = E[V | sell]:

```
ask = p(1+μ) / (1 + μ(2p−1))          bid = p(1−μ) / (1 − μ(2p−1))
logit(ask) − logit(p) = logit(p) − logit(bid) = ln((1+μ)/(1−μ)) = 2·artanh(μ)
```

**Binary GM is a constant log-odds half-spread** [D]. In price terms:

- ask − p = 2μp(1−p)/(1+μ(2p−1)); at 50c the spread equals μ.
- Implied informed share from a quoted spread: μ̂ = tanh(h_x/2), where h_x is the logit
  half-spread.

Half-spreads from these formulas, in cents:

| μ | p = 50c | p = 80c (ask / bid side) | p = 95c (ask / bid side) |
|---|---|---|---|
| 2% | 1.00 | 0.63 / 0.65 | 0.19 / 0.19 |
| 5% | 2.50 | 1.55 / 1.65 | 0.45 / 0.50 |
| 10% | 5.00 | 3.02 / 3.40 | 0.87 / 1.04 |
| 20% | 10.00 | 5.71 / 7.27 | 1.61 / 2.32 |

Readings:

- A 5c spread at 50c is break-even for about 5% informed takers. Near the extremes, a 1-tick
  half-spread covers 10% or more.
- So at extreme prices the **tick, not information, sets the spread**. That is a rent for makers,
  unless the informed share there is large, as with late-game outcomes or decided weather
  brackets.
- Adding fees: ask − p ≥ 2μp(1−p)/(1+μ(2p−1)) + f_maker(ask).

**Bartlett & O'Hara's binary break-even** [V, https://conference.nber.org/conf_papers/f243677.pdf].
A maker who sells YES and holds to settlement breaks even only if the half-spread s ≥ α·(100 − P)
cents. Observed Kalshi effective half-spreads average about 3c, which covers at most 6% informed
at 50c. At extremes the grid caps the half-spread at min(P, 100−P) while the loss is
max(P, 100−P). This agrees with binary GM at 50c.

### 3.2 Kyle (1985) and log-odds price impact

Kyle's single-period result is λ = σ_v/(2σ_u), with σ_v = √(p(1−p)) for a binary
[L, https://doi.org/10.2307/1913210].

Empirical version for prediction markets [V, Bartlett–O'Hara §4.1; Tsang & Yang 2026,
https://arxiv.org/abs/2603.03152]:

```
Δx_{j,h} = α_j + λ_j · SOF_{j,h} + ε          (5-minute intervals h; x = log-odds of the price;
SOF = Σ taker_sign·size; ≥ 50 intervals per market)
Δx_{j,h} = γ0 + γ1·SOF + γ2·ΔSOF + ε           (Glosten–Harris: γ1 permanent, γ2 transitory)
```

On Kalshi the price impact is almost entirely permanent (information, not inventory). Single-name
markets have the highest λ. Weather's high λ comes from *uninformed* flow that moves prices but is
wrong about the outcome [V].

Mapping to GM [D]: with unit trades, each buy moves the logit by 2·artanh(μ). A per-trade logit λ
(regress on the signed trade *count*) gives μ̂ ≈ tanh(λ̂/2).

Our trade data already carry the taker side, so no trade-classification algorithm is needed.

### 3.3 PIN and VPIN

**PIN** (Easley, Kiefer, O'Hara & Paperman 1996, https://doi.org/10.1111/j.1540-6261.1996.tb04074.x)
[L]. Per bucket with B buys and S sells:

```
L = (1−α)·Pois(B;ε_b)·Pois(S;ε_s) + α·δ·Pois(B;ε_b)·Pois(S;ε_s+μ) + α(1−δ)·Pois(B;ε_b+μ)·Pois(S;ε_s)
PIN = αμ / (αμ + ε_b + ε_s)
```

For us a "day" becomes an information window: a 5-minute bucket, a game quarter, or a speech.

**VPIN** (Easley, López de Prado & O'Hara 2012, https://doi.org/10.1093/rfs/hhs053) [L]:
VPIN = Σ_τ |V_B,τ − V_S,τ| / (n·V).

The prediction-market version (Bartlett–O'Hara §6) [V]:

- Taker side is exact.
- Buckets are **fixed $500 of dollar volume**. The standard ADV/50 rule gives 7-contract buckets
  and VPIN ≈ 1.
- Trailing VPIN is the mean over the last 20 buckets.
- **Only 30–70c bucket VWAP.** Near 0 and 1, flow is mechanically one-sided as the outcome
  becomes clear.
- Results: VPIN predicts maker losses in single-name markets (OLS t = −2.42; logit z = 3.58),
  not in broad-based ones. High-VPIN buckets with YES-directed flow settle YES much more often
  (t = 12.44). The effect is concentrated in NO-settling markets.

**PIN and VPIN disagree on Polymarket** (Le, SSRN 6933527,
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6933527) [S]. Across 5,518 resolved
contracts, linked negative-risk markets have higher PIN but lower VPIN. Do not treat the two as
interchangeable.

### 3.4 Das (2005) and the Bayesian market maker (Brahma et al. 2012)

Sources: Das, Quant. Finance 5(2), 2005, https://doi.org/10.1080/14697680500148067 [L];
BMM: EC 2012 and arXiv 1009.1446, https://arxiv.org/abs/1009.1446 [V].

The zero-profit learning maker:

- Holds a Gaussian belief V ~ N(μ_t, σ_t²). Traders see signals s ~ N(V, σ_ε²).
- Quotes are the fixed points ask = E[V | s > ask] and bid = E[V | s < bid].
- In closed form: ask = μ_t + σ_ε·Q(ρ_t)·√(1+ρ_t²), where ρ_t = σ_t/σ_ε is the maker's
  information disadvantage and Q is a universal function.
- After each action it does a range update: a trade or no-trade tells it s ∈ (z−, z+), and μ and
  σ² are updated by truncated-normal moments.

Uncertainty shrinks with every trade, so the maker becomes overconfident and adapts *exponentially
slowly* to a second shock.

**BMM's fix is a consistency index**:

```
L(μ,σ) = ∫ N(v; μ, σ) Π_i [Φ(z_i+; v, σ_ε) − Φ(z_i−; v, σ_ε)] dv
C = L(μ_t, 2σ_t) − L(μ_t, σ_t)
```

C is computed over a window of W recent trades. If C > 0 (one-sided recent flow is more likely
under doubled uncertainty), double σ_t, which widens the spread. W becomes the main parameter: it
trades stability in calm periods against speed in shocks.

In live trading against LMSR at equal average spreads, BMM had lower expected loss, but it has
**no loss bound**.

For us [I]: the consistency index is a principled *regime switch*, triggered when the recent trade
sequence becomes too one-sided for the current belief. That is a likelihood-ratio version of
"pull after a one-sided burst".

### 3.5 LMSR and cost-function market makers

Hanson's LMSR [V via the BMM paper's restatement; L:
https://mason.gmu.edu/~rhanson/mktscore.pdf]:

```
C(q) = b·ln Σ_i e^(q_i/b);   p_i = e^(q_i/b)/Σ_j e^(q_j/b);   worst-case loss b·ln n  (b·ln 2 for a binary)
binary: p = S((q_yes − q_no)/b)  →  logit(p) moves by Δq/b per contract
spread for size Q at inventory q:  δ(Q) = (b/Q)·ln[ (cosh(q/b) + cosh(Q/b)) / (2cosh²(q/2b)) ]
```

Readings:

- **LMSR is pure inventory skew in log-odds** [D]. It matches the logit-AS reservation price with
  γσ²(T−t) ↔ 1/b, and the GM logit half-spread with 2artanh(μ) ↔ 1/b for unit trades.
- It never learns, so it subsidizes informed traders up to b·ln 2 per binary market.
- The liquidity-sensitive LS-LMSR (Othman, Pennock, Reeves & Sandholm 2013,
  https://doi.org/10.1145/2509413.2509414) [L] sets b = α·Σq_i. Prices then sum to more than 1,
  a built-in vig, and the maker can profit.
- The cost-function framework (Abernethy, Chen & Wortman Vaughan 2013,
  https://doi.org/10.1145/2465769.2465777) [L] gives the general convex-duality view.
- Lesson for a CLOB [I]: in thin binary books someone must pay the informed. On an LMSR it is the
  sponsor. On Kalshi and Polymarket US it is takers through the spread, YES-optimists through the
  "behavioral surplus" (§4.1), or the exchange through liquidity rewards. A maker who quotes only
  on inventory, like an LMSR, will lose to informed flow.

### 3.6 Chakraborty–Kearns: when does a ladder maker profit? [V]

Source: https://www.cis.upenn.edu/~mkearns/papers/marketmaking.pdf.

A maker with 1-lot orders at every tick of a price ladder earns **exactly (K − z²)/2** on any
price path, where K = Σ|P_t − P_{t−1}| (path length, in ticks) and z = P_T − P_0 (net move).
Market making is profitable if and only if the path is long relative to its net displacement.
Any mean reversion gives positive expected profit (their Thm. 3.1).

Quick look on our shared volatile buckets, 2026-11-05 03–18 UTC [M]:

- Two-sided buckets: 5,411 calm and 1,445 volatile.
- Path length is `path_c`, measured at spreads ≤ 6c. The cache has no net move, so `range_c ≥ |z|`
  stands in for it, which makes the result below a *lower bound* on how often K > z².

| | median path (c) | median range (c) | median path/range² | share with path > range² |
|---|---|---|---|---|
| calm | 1.5 | 0.5 | 4.0 | 74% |
| volatile | 16.0 | 3.0 | 1.9 | 58% |

Volatile windows have about 10× the path to harvest but are trendier per unit of path. The
identity ignores queues, spreads and fees, so it is a regime diagnostic, not a P&L estimate.
Predicting next-bucket K/z² from current features is a cleaner target for a regime filter than
"volatile or not".

### 3.7 What each information model implies for quoting thin binary books

| Model | Quoting rule it implies | Parameter to estimate from our data |
|---|---|---|
| GM (binary) | Constant logit half-spread 2artanh(μ); tick binds near 0/1 | μ from spreads (tanh(h_x/2)), from logit λ, or from settlement-markouts of our fills |
| Bartlett–O'Hara | Spread cannot cover informed losses when held to settlement; profit comes from the YES-optimism surplus | Per class: informed λ, calibration gap (YES price minus YES settle rate) |
| Kyle / GH | Impact is permanent: do not expect fills to revert | λ in log-odds per 5-min interval; permanent share |
| PIN / VPIN | Pull or widen when flow is one-sided at interior prices | $500-bucket trailing VPIN (or a contract-bucket analogue in thin markets), 30–70c only |
| Das / BMM | Widen when the recent sequence is too one-sided for the current belief | Window W, signal noise σ_ε (from trade dispersion) |
| LMSR | Inventory-only skew in logit, b ↔ 1/(γσ²τ); loses to the informed | b equivalent of our skew |
| Chakraborty–Kearns | Quote when the path is mean-reverting | K/z² by regime, predicted ex ante |

---

## 4. Empirical evidence on prediction-market and betting-exchange microstructure

### 4.1 Kalshi

- **Bartlett & O'Hara, "Adverse Selection in Prediction Markets: Evidence from Kalshi"**
  (NBER conference paper, v. July 2, 2026; 41.6M trades, 478k markets) [V]:
  - Single-name markets have significantly higher informed price impact than broad-based ones
    (log-odds λ t = 10.88). The impact is almost entirely permanent.
  - Effective spreads are only modestly wider, measured at fresh quotes.
  - **Makers earn more in single-name markets: +1.91c per contract**, against +0.82c in
    broad-based markets. Makers win 63.4% of contracts but lose more when they lose: 23.13c
    against 16.34c won.
  - **Makers mostly hold to settlement.** The profit source is YES optimism: in every price
    bucket, YES prices exceed realized YES rates, even at mid-range prices.
  - About 5% of bid matches come from big institutional makers; the rest come from over 2,000
    smaller makers.
  - The Market Maker Program gives designated firms lower fees and adjusted position limits.
  - Large equity market-makers stayed away; the ones who entered (SIG) came from sports betting.
- **Bürgi, Deng & Whelan** (GWU WP 2026-001) and **Becker** (72.1M trades): makers beat takers,
  makers buying ≤ 10c lose, and NO beats YES at 69 of 99 price levels [V/S, in market-making.md
  §1.2].
- **Xi, Moallemi, Pai & Wang** (https://arxiv.org/abs/2607.08199) [V abstract/S details]:
  - Volatility splits into a deadline-resolution component (Wright–Fisher) and an
    adverse-selection component, on 880k hourly Kalshi observations.
  - Economics markets are explained by deadline resolution alone.
  - **Sports show abrupt event-driven spikes.**
- **In-game depth** [S, Blockworks Research, March Madness 2025 data,
  https://app.blockworksresearch.com/unlocked/from-betting-to-trading-how-kalshi-is-reshaping-sports-markets]:
  - Depth at best rises 19× over the 48 h before games.
  - **In-game depth falls 76% despite more trading, while spreads stay tight.** The authors read
    this as makers cutting displayed size rather than widening.
  - The best execution window is 48 h to 1 h pre-game.

### 4.2 Polymarket

- **Price discovery**: Polymarket leads Kalshi, especially when liquidity and activity are high.
  The venue with more directional large-trade flow leads (Ng, Peng, Tao & Zhou, SSRN 5331995,
  https://papers.ssrn.com/abstract=5331995) [S]. Our own fast lead-lag in pre-game hours found
  near-zero cross-correlation (SUMMARY.md).
- **Informed trading**: Mitts & Ofir flag over 210k suspicious Polymarket bets in Feb 2024 to
  Feb 2026, with about a 70% win rate (https://www.law.columbia.edu/news/archive/professor-joshua-mitts-prediction-markets-and-informed-trading)
  [S]. A critique (https://arxiv.org/abs/2605.00493) notes that the screen uses information only
  available after the fact [S].
- **Maker markouts rise with maker size** (Charjan note) [S, market-making.md §1.2].

### 4.3 In-play betting exchanges (the closest analogue to volatile windows)

- **Croxson & Reade** (Economic Journal 124(575), 2014; https://ideas.repec.org/p/bir/birmec/11-01.html)
  [V abstract]. Betfair prices "update swiftly and fully" after goals. On goals just before half
  time there is no profit from reacting during the break.
- **Choi & Hui** (JEBO 107, 2014, https://econpapers.repec.org/RePEc:eee:jeborg:v:107:y:2014:i:pb:p:614-629)
  [S]. Reactions depend on surprise: underreaction or overreaction depending on how unexpected
  the goal was.
- **Angelini, De Angelis & Singleton** (IJF 38(1), 2022, https://econpapers.repec.org/RePEc:eee:intfor:v:38:y:2022:i:1:p:282-299)
  [S]:
  - There is a reverse favourite-longshot bias pre-match and in-play.
  - Mispricing grows after surprises, such as a late underdog goal.
  - Prices adjust correctly to expected news but overreact to surprises (working-paper version).
- **Whelan, "Agreeing to Disagree: The Economics of Betting Exchanges"** (MPRA 126351, Sept 2025,
  https://mpra.ub.uni-muenchen.de/126351/1/MPRA_paper_126351.pdf) [V]. Uses Betfair soccer with
  maker and taker identified per trade:
  - Pre-match and early in-play, makers out-earn takers as the model predicts.
  - **As matches progress, longshot bets lose heavily even for makers.** At 105 minutes on the
    data clock (about 87 match minutes), bottom-decile takers lose about 70%, and makers who bet
    longshots lose almost as much.
  - **Takers backing strong favourites earn small significant profits in the second half.**
- **Page, "It ain't over till it's over: Yogi Berra bias on prediction markets"** (Applied
  Economics 44(1), 2012, https://ideas.repec.org/a/taf/applec/44y2012i1p81-92.html) [S]. Traders
  overestimate the trailing team's comeback chance in the final minutes. Whelan cites it with
  InTrade data on 1,100 US sports markets.
- **Brown, in-play insider trading** (Applied Economics 44(9), 2012,
  https://ueaeprints.uea.ac.uk/id/eprint/45783/) [S]. Courtsiders in tennis trade on point
  outcomes before broadcast viewers.
- **"Do betting markets sense a goal coming?"** (Bundesliga, https://arxiv.org/abs/2505.21275)
  [V]. No pre-goal anticipation by the bookmaker or bettors. Goals are unpredictable jumps, so
  hazard models (§2.4) are the right frame, not drift forecasts.

### 4.4 Liquidity-incentive programs: current terms

Re-checked today; details are in market-making.md §3.

- **Kalshi LIP** [V: https://help.kalshi.com/en/articles/13823851-liquidity-incentive-program]:
  - Ends 2027-01-01.
  - Snapshots once per second at a random moment.
  - Score = size × DiscountFactor^(ticks below the Reference Price), normalized per side.
  - The Reference Price is the first level where cumulative resting size reaches ⅕ of Target
    Size.
  - A snapshot is void if either side is below Target Size (100–20,000).
  - $1–$1,000 per market per day; payouts below $1.00 per market program are not paid.
- **Polymarket US LIP** [V: https://docs.polymarket.us/incentives/liquidity]:
  - Score = DF^(ticks from best) × size, snapshotted every second.
  - Only size inside Target Size from the best price counts.
  - Optional Max Spread: if it is breached, nobody is paid that second.
  - Periods: Early (listing to −6 h), Day-of (−6 h to start), Live, and Daily.
  - Rewards under $1 are not paid, and none are paid for cancelled or postponed games.
- Bartlett & O'Hara call Kalshi LIP rewards "negligible" against the behavioral surplus at the
  aggregate level [V]. For a $100 account the $1 minimum per program is the binding constraint
  (market-making.md §3.3).

---

## 5. Volatility-regime practice

### 5.1 Venue mechanisms that protect makers in volatile windows

| Venue | Mechanism | Evidence |
|---|---|---|
| Betfair | **Speed bump** on every in-play order: soccer 5–9 s, tennis 5 s, horse racing 1 s. **Cancels are not delayed.** `betDelay` per market. On darts, basketball, tennis and selected football markets, **passive (non-crossing) orders bypass the delay** (`betDelayModels`). | Brown & Yang [V: https://ueaeco.github.io/working-papers/papers/afe/UEA-AFE-070.pdf]. Betfair developer support [V: https://support.developer.betfair.com/hc/en-us/articles/26791968420636] |
| Betfair (effect) | Over five EPL seasons, raising the soccer delay to 8–9 s cut in-play quoted spreads 7.3% and effective spreads 3.8%, raised depth at the best three prices 19% and order frequency 87%. No change in tennis, where the delay stayed at 5 s (placebo). | Brown & Yang [V] |
| Smarkets (2015) | Taker trades pended 8 s and voided on a material event (goal, red card) | [S: https://sbcnews.co.uk/technology/2015/05/22/smarkets-rolls-out-instant-match-feature-following-successful-beta/] |
| Polymarket intl | **All resting orders on a sports market are cancelled at the official game start.** Marketable sports orders get a "delayed" status for a configured delay: the docs say 1 s in one place and 3 s in another; a third party saw 1 s on EPL and none on NBA (Oct 2026). | [V: https://docs.polymarket.com/concepts/markets-events.md; V/S: https://docs.polymarket.com/concepts/order-lifecycle.md, https://help.polymarket.com/en/articles/13364444-limit-orders] |
| Polymarket US | **Mass Quote Protection** (FIX): if the account's executed quantity in a link-ID bucket reaches X within a rolling Y seconds (e.g. 10 in 3 s), all its remaining resting orders in that bucket are cancelled. | [V: https://docs.polymarket.us/institutional/fix-api/fix-mass-quote-protection.md] |
| Crypto.com | About 3 s delay on sports orders for non-makers. **Makers exempt.** | [S: https://www.finnewsnetwork.com.au/archives/finance_news_network3392236.html, Bloomberg via secondary] |
| Kalshi | **Order groups**: rolling 15 s contracts limit, auto-cancels the whole group, plus a manual trigger. `cancel_order_on_pause`. A sports order delay was reportedly filed with the CFTC in Dec 2025; **I found no confirmation it went live**, and the API changelog has no speed-bump entry. | [V: order groups doc; S: https://www.gambling911.com/node/78397] |

**Check this in our own data [I]:** if Kalshi in-play prints show taker fills arriving a fixed
interval after the book's own reaction, a delay exists. Tonight's in-play recording can show it.

### 5.2 What professional makers do in volatile windows

From the papers and venue designs above; Kalshi-specific practice is not documented publicly.

1. **Pull, then requote.** Speed bumps exist because a maker's defence is the cancel, and the
   cancel only works if it lands before the informed order (Brown–Yang).
2. **Cut size, not spread** (Kalshi in-game depth −76% with tight spreads) [S]. With 1-lot
   orders we have no size to cut, so our lever is *whether* to quote.
3. **Quote deeper on the side that is about to be picked off and closer on the other**
   (CJR adverse-selection term). Widen in proportion to order-arrival intensity (Hawkes λ±) and
   to the buy/sell imbalance λ+ − λ−.
4. **Widen or pull on a toxicity spike or jump detector; pre-widen before scheduled news**
   (Dalen). Treat κ as uncertain in volatile windows (CDJ ambiguity aversion).
5. **Use exchange-side kill switches**: Kalshi order groups and Polymarket US MQP. The fill burst
   itself cancels the rest of the book before any further client message.
6. **Hold the edge to settlement in markets with a behavioral surplus** (Bartlett–O'Hara). Do not
   rely on round trips.

### 5.3 What latency is competitive

- **The informed in-play trader's advantage is seconds of *information* latency.**
  - The minimum TV delay is about 4 s. Satellite adds about 2 s, and HD and international
    relays add more (Kooij 2014, via Brown–Yang) [V].
  - A sports data feed was faster than cable TV for NBA games, with small variance
    (https://networks.cs.northwestern.edu/website/publications/gi18.pdf) [S].
  - A World Cup data API ran 4–6 s ahead of TV
    (https://blog.benjojo.co.uk/post/beating-the-broadcast-delay-world-cup) [S].
  - Streaming can lag tens of seconds [S].
  - Betfair's 8–9 s soccer delay was sized to that gap (Brown–Yang) [V].
- **Kalshi's own live data comes from Sportradar but is REST-polled and may be cached.** A
  `live_data_delayed` flag appears only once data is significantly stale [V:
  https://docs.kalshi.com/api-reference/live-data/get-live-data]. Kalshi recommends
  `GET /live_data/milestone/{id}` or `/live_data/batch` over milestone status, and
  `/game_stats` gives play-by-play for the main US sports and soccer. This is a *state* feed
  (score, clock, period) to drive hazard models. It is not fast enough to win the race to a
  score, and its latency against price jumps should be measured before use [I].
- **Our 50 ms order path** is competitive for anything triggered by the order book itself:
  sibling-book moves (median 55 ms requote across team A/B books) and our own fills. It is
  hopeless against a score observed seconds earlier.
- Gao & Wang (https://arxiv.org/abs/1806.05849) [S] show latency is a risk that lowers market-
  making value and give profitability criteria under latency.
- Our intraminute study found results flat over 25–250 ms latency in pre-game data
  (SUMMARY.md). The binding latency is information, not network [I].

### 5.4 Write-budget arithmetic for pulling quotes [V rate limits, D]

- Basic tier, 100 tokens/s:
  - one create or cancel is 10 tokens;
  - a batch cancel is 2 tokens per order;
  - pulling 40 resting orders costs 80 tokens (under 1 s);
  - re-placing them costs 400 tokens (4 s).
- **Advanced (300 tokens/s) is a self-service API call**, which cuts re-placing to about 1.3 s.
- **Order groups make the pull itself cost one write**, or zero when the group trips on a fill.
- Practice: one order group per Kalshi event (game, weather event, mention event), so that a
  burst in one leg pulls the siblings exchange-side. Requote slowly; the cooldown is a feature.

---

## 6. Parameter-estimation cookbook (our data → each parameter)

| Parameter | Estimator | Data | Window / split |
|---|---|---|---|
| A, k (fill intensity) | OLS of ln(#trades with depth ≥ δ / market-s) on δ ∈ [0.5, 4]c, pre-trade mid without exec deltas | Recorded L2 trade prints | Per series × regime (`volatile_prev`), trailing 1–4 h; fit on dev only |
| Queue-aware P_fill(π) | Simulator replay with FIFO queues (`risk_averse`) | Recorded L2 | As above |
| σ (diffusive) | Bipower variation of 1-s mid (or logit) increments while spread ≤ 6c | Recorded L2 | Trailing 10 min (hftbacktest) to 1 h |
| h, J (jumps) | Dalen EM mixture on 1-s logit increments, or threshold \|Δmid\| ≥ 3c/1 s; Hawkes MLE on jump times for clustering | Recorded L2; historical trades for in-play (1-s VWAP) | Per series × hour-of-day (weather obs clocks, speeches, game phase) |
| In-play J(t, state) | Stern/Polson–Stern Φ((ℓ+μτ)/(σ√τ)) with σ = μ/Φ⁻¹(p0); Skellam for soccer/hockey | Pre-game Kalshi moneyline and spread prices; Kalshi `live_data` state | Per game, causal onset (inplay_dynamics.py) |
| ε (adverse drift after our fill) | Mean signed mid move 30–300 s after our simulated fills (mo) | Simulator fills | By spread band, price band, regime |
| μ (informed share) | tanh(h_x/2) from quoted logit half-spreads; tanh(λ̂/2) from per-trade logit Kyle λ; PIN MLE on 5-min B/S counts | L2 quotes; trades with taker side | Per series; compare classes (single-name vs broad) |
| Kyle λ, GH split | Δlogit per 5-min on SOF and ΔSOF, ≥ 50 intervals per market | Historical trades + candles or L2 mids | Dev < Oct 1, holdout Oct |
| VPIN | $500 dollar buckets (try $100 and contract buckets in thin markets), trailing 20, VWAP 30–70c | Trades with taker side | Trailing; gate uses only past buckets |
| BMM σ_ε, W | Dispersion of trade prices around the filtered fair value; W swept 5–40 trades | Trades | Dev |
| K/z² (mean reversion) | path_c and net mid change per 5-min bucket | Volatile-bucket cache (add net change) | Predict next bucket from current |
| γ, φ, γ_T, ϕ | Not estimable: sweep on dev, pick by holdout settlement P&L and mx | n/a | n/a |

---

## 7. Ranked strategy specifications

Ranking is by expected value to the current maker candidate (improve@5c Quoter: mo positive,
mx negative, settlement undecided), then by cost to test. Every test reports all / volatile /
calm × dev / holdout. Fills are bucketed by their own bucket's `volatile` flag, and the
decisions use only `volatile_prev`. Each test also reports fills, events, mo/mx at 60 s and 300 s,
and settlement P&L per contract net of exact fees, with t-statistics clustered by event (game).
Latency 50 and 100 ms. Split: L2 dev 03–11, holdout 11–18 UTC; tonight's in-play hours are a
second holdout. Historical data: dev before 2026-10-01, holdout October.

### S1. Event-level fill-burst kill and sibling pull (order-group overlay). NEW, not in the swarm list.

- **Why.** Our fills are adversely selected exactly when the snap-back fails (intraminute study).
  CJR says the side about to be picked off must move out, and venues use fill-burst kill switches
  for this reason (Polymarket US MQP, Kalshi order groups). A 1-lot maker quoting several legs of
  one event is picked off leg by leg. The first fill is the cheapest warning. The sibling books
  requote within about 55 ms, so the event, not the market, is the unit of risk.
- **Rules.**
  1. Every resting order belongs to an order group per `event_ticker`, with contracts_limit L
     per rolling 15 s.
  2. When a fill makes the event's rolling 15-second total exceed L, cancel every other resting
     order of ours in that event *exchange-side*. In the simulator: at the fill's timestamp, no
     latency, no write tokens. Any order hit by the same aggressor sweep still fills. Lock the
     group for C seconds, then reset (1 write) and let the base Quoter requote.
  3. Manual trigger (1 write, normal latency): if any sibling market in the event moves its mid
     by ≥ m cents within w seconds while its spread is ≤ 6c, or the paired Polymarket mid moves
     (existing `pm_pull`).
  4. Base: `Quoter(improve=True, min_spread=0.05, cooldown_s=300)`, and the behind20 variant.
- **Parameters (dev only).** L ∈ {1, 2}; C ∈ {30, 120, 300} s; m ∈ {2, 3}c; w ∈ {0.5, 2, 5} s.
- **Test.**
  - Wrap the Simulator in a new module (do not edit l2.py).
  - Diagnostic first: in the baseline, compare markouts of *follow-on fills* (a second fill in the
    same event within 15 s or 120 s of a first) with first fills. If follow-on fills are not
    worse, stop.
  - Then the full grid.
  - **Pass:** holdout mx_300s improves by ≥ 0.5c per contract (t ≥ 2 by event), fills fall
    ≤ 30%, and settlement P&L is not worse.
  - Rerun on 2026-11-05T23..2026-11-06T07 (in-play).
- **Risk.** Too few multi-leg events with our quotes in them. Count them first: KXHIGH/KXLOWT
  brackets, KXTRUMPSAY strikes, and team A/B books should give enough.

### S2. Settlement-held bias-side maker at interior prices with a directional VPIN gate (Bartlett–O'Hara). PARTLY OVERLAPS the Hawkes/VPIN and GM/PIN/Das studies and maker_variants.

The new parts are: one-sided, the YES-optimism side, held to settlement, single-name classes,
and a *directional* VPIN gate.

- **Why.** The only documented source of maker profit large enough to cover informed losses in
  binaries is the YES-optimism surplus, captured by holding to settlement: +1.91c per contract in
  single-name markets [V]. Our mo > 0 > mx pattern says round trips do not pay. Becker's
  NO-beats-YES result and Bürgi's "makers buying ≤ 10c lose" point the same way.
- **Rules.**
  1. Universe: single-name classes (mentions such as KXTRUMPSAY, person-specific politics,
     culture, entertainment). Define the class list from `category`/series on dev, fixed before
     testing.
  2. Only when the mid is 30–70c.
  3. Rest a NO bid (that is, sell YES) one tick better than the best NO bid when the spread is
     ≥ 4c. Never bid YES. Hold to settlement.
  4. Caps: 3 contracts per market, 5 per event, $5 at risk per event.
  5. **Gate:** pull when trailing VPIN over 20 buckets is ≥ v* *and* the net taker flow over those
     buckets is YES-directed. Informed YES buyers are the takers who hit a YES seller, and
     YES-directed high-VPIN buckets settle YES (t = 12.4).
- **Parameters.** v* (dev quantiles 70, 80, 90%); bucket size ($100, $500, or 50 contracts); price
  band.
- **Test.**
  - Historical: `kalshi_trades` + `kalshi_markets.result`. A maker fill only when a later print
    trades strictly *through* our price.
  - **Volume-blind eligibility**: decide at the decision time from data before it. The withdrawn
    longshot edge failed on volume and look-ahead bias.
  - First verify that the trades table is complete for non-game series. The in-play study found
    game markets hold only pre-game windows.
  - Then check fill realism on the recorded L2 hours.
  - **Pass:** holdout ≥ +1c per contract after maker fees, t ≥ 2 by event, positive in each month.
- **Risk.** Lumpy: each informed loss is 30–70c. Needs hundreds of settled contracts to decide.

### S3. Jump-hazard quote gate (credit-style inequality). PARTLY OVERLAPS the in-play dynamics study (logit dynamics, phase) and the volatile re-tests (which gate on `volatile_prev`).

The new part is the quantitative hazard × overshoot against edge × fill-rate test.

- **Why.** GLFT/CJR in calm hours reduce to "quote about 2.6c from fair". What kills us is jumps.
  §2.4 gives the per-side break-even.
- **Rules.**
  1. Quote side j at the candidate price π (distance δ from fair, where fair is the mid, or the
     micro-price if the fair-value study finds it better) only if

     λ_u(δ)·(δ − f) > κ_s · ½ · ĥ · E[(|J| − δ)+]

     plus the skew term.
  2. Non-sports: ĥ and J come from the Dalen EM on 1-s logit increments, per series × hour-of-day,
     fitted on dev. Add a Hawkes-style excitation, ĥ_t = ĥ_0 + Σ η·e^(−β(t−t_i)) over recent jumps
     t_i, so that hazard stays elevated after a jump.
  3. In-play: J from Stern (basketball, NFL) or Skellam (soccer, hockey), with state from
     `live_data` (score, clock). Scoring hazard per league comes from historical play-by-play or
     from price-jump counts.
  4. λ_u: fills in calm buckets not followed by a jump within 10 s.
- **Parameters.** Jump threshold; EM window; β, η; safety multiplier κ_s ∈ {1, 2, 4}.
- **Test.** Same replay as S1, against the `volatile_prev` gate and against no gate. **Pass:**
  holdout mx improves with ≥ 50% of calm fills kept. In-play: tonight.
- **Risk.** Too few jumps per series to fit. Pool by category and say so.

### S4. Late-game favourite-side maker and taker in play ("Yogi Berra"). TAKER HALF ALREADY IN the in-play dynamics study.

`inplay_dynamics.py` already has late-game snapshots that buy the favourite or the underdog as a
taker at phase ≥ 0.5, 0.67 and 0.8. Read its result first. The new parts here are the maker
variant, the price band, and the per-league split.

- **Why.** Late in games, takers overrate comebacks. Longshots lose heavily even for makers, and
  takers backing strong favourites earn small significant profits (Whelan, Betfair 2022–24 [V];
  Page, InTrade [S]). Kalshi has the same retail-heavy in-play flow.
- **Rules.**
  - Game-winner markets only.
  - phase = elapsed since causal onset / dev median game length.
  - When phase ≥ φ0 and the favourite's price is in [0.80, 0.95]:
    - Maker: bid the favourite (YES on the favourite, or NO on the underdog), improving 1c if
      the spread is ≥ 2c, held to settlement.
    - Taker variant: buy the favourite at the ask if ask ≤ 0.95. The taker fee is
      0.07·p(1−p), which is 0.33–1.1c in this band.
  - Never bid the underdog after phase ≥ 0.5.
  - Stop at the first print in the decided zone (≥ 98c).
- **Parameters.** φ0 ∈ {0.5, 0.75, 0.9}; band.
- **Test.**
  1. Calibration first: realized win rate against price by phase × price decile, on
     `inplay_dynamics.fetch_inplay` data (`kalshi_trades` lacks in-play prints for games). Look
     for a favourite-underpricing gap above fees.
  2. Then P&L with trade-through fills for the maker and next-same-side print for the taker,
     clustered by game, monthly.
  - **Pass:** holdout ≥ +1c per contract net, t ≥ 2.
- **Risk.** Each loss is 80–95c, so monthly consistency is the real hurdle. Sports with
  sudden-death dynamics (MLB late innings, NFL) differ: report by league.

### S5. Binary GLFT in log-odds with an adverse-selection shift and a settlement penalty. ALREADY BEING TESTED by the AS/GLFT study.

This spec is here so its results can be compared on the same terms.

- **Rules.**
  1. x = logit(fair).
  2. Half-spread δ_x = c1 + σ_x·c2/2 + ε_x, where ε_x is the mean post-fill logit drift
     (CJR/one-period δ* = 1/κ + ε). Skew is q·σ_x·c2.
  3. γ rises toward settlement so that γσ_x²τ tracks the remaining settlement variance:
     γ_T·p(1−p) (Feil–Nendel).
  4. Map back: p_b = S(x − δ_b), p_a = S(x + δ_a). Round the bid down and the ask up to the 1c
     grid. Improve if the result is ≥ 1c inside the best; join only with ≥ 20 contracts behind;
     otherwise do not quote.
  5. Inventory cap: |q| ≤ ⌊Q0 / (4p(1−p))⌋, capped at 3.
- **Parameters.**
  - A and k from the trade-depth fit, per series, calm windows. Expect roughly A ≈ 1e-3/s and
    k ≈ 0.35–0.5/c [M].
  - σ_x from 1-s bipower variation.
  - γ ∈ {0.01, 0.1, 1}.
  - ε from dev fills.
- **Expected.** With calm-hour parameters the half-spread is about 2.6c, so S5 reproduces the
  Quoter, plus a skew that matters only after fills. Any gain must come from ε and the
  inventory terms in volatile windows. If S5 does not beat the Quoter on holdout mx, stop
  developing continuous-δ models for 1c, 1-lot books.

### Also considered, not ranked

- **Reward-aware calm quoting** (Kalshi LIP, Polymarket US LIP). New to the swarm, and already
  market-making.md strategy #1. It is testable on recorded books once `GET /incentive_programs`
  is recorded: compute our size share at or above the Reference Price each second. Ranked out
  only because of the $1 minimum per program at our size.
- **A Chakraborty–Kearns regime predictor**: forecast next-bucket K/z² from current features, and
  quote when it is > 1. This could be a feature inside the volatile re-tests.
- **Polymarket intl speed-bump lead (hypothesis for tonight's lead-lag rerun).** Polymarket intl
  delays sports takers 1–3 s but not maker cancels or reposts. Its *quotes* may therefore move
  before Kalshi's *trades* in play. Test PM mid changes against Kalshi mid changes at 0.1–3 s
  lags on in-play hours only: `fast_lead_lag(settings(), start='2026-11-05T23',
  end='2026-11-06T06')`.

---

## 8. Sources

Primary papers and pages (read today unless tagged otherwise in the text):

- Avellaneda & Stoikov (2008): https://doi.org/10.1080/14697680701381228 · https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf
- Guéant, Lehalle & Fernandez-Tapia (2013): https://arxiv.org/abs/1105.3115 · Guéant (2017): https://arxiv.org/abs/1605.01862
- hftbacktest GLFT tutorial: https://hftbacktest.readthedocs.io/en/latest/tutorials/GLFT%20Market%20Making%20Model%20and%20Grid%20Trading.html
- Cartea, Jaimungal & Ricci (2014): https://c.mql5.com/forextsd/forum/169/a_high_frequency_trading_perspective.pdf · SIAM Review version: https://epubs.siam.org/doi/10.1137/18N974601
- Cartea, Donnelly & Jaimungal (2017): https://ora.ox.ac.uk/objects/uuid:32248d98-3e1a-499a-ae2f-8cddd64beb4c
- Cartea & Wang (2020): https://doi.org/10.1142/S0219024920500168
- Barzykin, Bergault, Guéant & Lemmel (2025): https://arxiv.org/abs/2508.20225
- Feil & Nendel (2026): https://arxiv.org/abs/2607.17991
- Dalen (2025): https://arxiv.org/abs/2510.15205
- Xi, Moallemi, Pai & Wang (2026): https://arxiv.org/abs/2607.08199
- Stern (1994): https://doi.org/10.1080/01621459.1994.10476851 · Polson & Stern (2015): https://doi.org/10.1515/jqas-2014-0095 · Feng, Polson & Xu: https://arxiv.org/abs/1604.03614
- Glosten & Milgrom (1985): https://doi.org/10.1016/0304-405X(85)90044-3 · Kyle (1985): https://doi.org/10.2307/1913210
- Easley, Kiefer, O'Hara & Paperman (1996): https://doi.org/10.1111/j.1540-6261.1996.tb04074.x · Easley, López de Prado & O'Hara (2012): https://doi.org/10.1093/rfs/hhs053
- Bartlett & O'Hara (2026): https://conference.nber.org/conf_papers/f243677.pdf · https://law.stanford.edu/2026/04/21/adverse-selection-in-prediction-markets-evidence-from-kalshi/
- Tsang & Yang (2026): https://arxiv.org/abs/2603.03152 · Le (PIN vs VPIN): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6933527
- Das (2005): https://doi.org/10.1080/14697680500148067 · Brahma et al. (BMM): https://arxiv.org/abs/1009.1446
- Hanson LMSR: https://mason.gmu.edu/~rhanson/mktscore.pdf · Othman et al. (2013): https://doi.org/10.1145/2509413.2509414 · Abernethy, Chen & Wortman Vaughan (2013): https://doi.org/10.1145/2465769.2465777
- Chakraborty & Kearns (2011): https://www.cis.upenn.edu/~mkearns/papers/marketmaking.pdf
- Gao & Wang (latency): https://arxiv.org/abs/1806.05849
- Ng, Peng, Tao & Zhou: https://papers.ssrn.com/abstract=5331995 · Mitts & Ofir coverage: https://www.law.columbia.edu/news/archive/professor-joshua-mitts-prediction-markets-and-informed-trading · critique: https://arxiv.org/abs/2605.00493
- Croxson & Reade: https://ideas.repec.org/p/bir/birmec/11-01.html · Choi & Hui: https://econpapers.repec.org/RePEc:eee:jeborg:v:107:y:2014:i:pb:p:614-629 · Angelini, De Angelis & Singleton: https://econpapers.repec.org/RePEc:eee:intfor:v:38:y:2022:i:1:p:282-299
- Whelan (2025): https://mpra.ub.uni-muenchen.de/126351/1/MPRA_paper_126351.pdf · Page (2012): https://ideas.repec.org/a/taf/applec/44y2012i1p81-92.html · Brown (2012): https://ueaeprints.uea.ac.uk/id/eprint/45783/ · Bundesliga anticipation: https://arxiv.org/abs/2505.21275
- Brown & Yang (speed bumps): https://ueaeco.github.io/working-papers/papers/afe/UEA-AFE-070.pdf · Betfair passive orders: https://support.developer.betfair.com/hc/en-us/articles/26791968420636-How-can-I-identify-markets-that-accept-passive-orders-via-the-Betfair-API
- Blockworks Research (Kalshi sports depth): https://app.blockworksresearch.com/unlocked/from-betting-to-trading-how-kalshi-is-reshaping-sports-markets
- Data-feed latency: https://networks.cs.northwestern.edu/website/publications/gi18.pdf · https://blog.benjojo.co.uk/post/beating-the-broadcast-delay-world-cup
- Venue docs: Kalshi rate limits https://docs.kalshi.com/getting_started/rate_limits · order groups https://docs.kalshi.com/getting_started/order_groups · live data https://docs.kalshi.com/api-reference/live-data/get-live-data · changelog https://docs.kalshi.com/changelog · LIP https://help.kalshi.com/en/articles/13823851-liquidity-incentive-program · Polymarket intl markets/events https://docs.polymarket.com/concepts/markets-events.md · order lifecycle https://docs.polymarket.com/concepts/order-lifecycle.md · help https://help.polymarket.com/en/articles/13364444-limit-orders · Polymarket US LIP https://docs.polymarket.us/incentives/liquidity · MQP https://docs.polymarket.us/institutional/fix-api/fix-mass-quote-protection.md
- Order-delay reports: https://www.gambling911.com/node/78397 · https://www.finnewsnetwork.com.au/archives/finance_news_network3392236.html
