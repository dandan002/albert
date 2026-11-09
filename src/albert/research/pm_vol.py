"""Time-series dynamics of a single Kalshi binary: backbone, information clock, stochastic vol, jumps.

Model (shared notation). ``p_t`` is the fair price, a martingale on [0, 1] with ``p_T`` in {0, 1}.

* **M0, Bachelier digital / information clock** (Stern 1994; Archak & Ipeirotis 2009; Taleb 2018).
  Outcome = 1{Y_T > 0}, dY = s(t) dW, p_t = Phi(Y_t / sqrt(V_t)), V_t = int_t^T s^2. In probit units
  Z = Phi^-1(p): dZ = sqrt(lambda) dW + lambda Z / 2 dt with information rate lambda = s^2 / V.
  Over a step h write L = ln(V_t / V_{t+h}) = int lambda (the *clock increment*) and
  rho = 1 - exp(-L) (the fraction of the remaining information released in the step). Then
  exactly Z_{t+h} | Z_t ~ N(Z e^{L/2}, e^L - 1), u = (Z_{t+h} - Z e^{L/2}) / sqrt(e^L - 1) ~ N(0, 1),
  and the one-step price variance has the closed form
  Var(p_{t+h} | p_t) = Phi2(Z, Z; rho) - p^2 = p(1 - p) - 2 T(Z, sqrt((1 - rho) / (1 + rho)))
  (T = Owen's T). A uniform clock is lambda = 1/tau, i.e. L = ln(tau / (tau - h)).
* **M1, bounded SABR as a stochastic time change.** Price local vol sigma_p = alpha c(tau) g(p) with
  backbone g = [p(1 - p)]^beta (beta = 1 logit-normal, 1/2 Jacobi) or g = phi(Phi^-1(p)) (probit,
  which is M0). We implement stochastic vol as a random *clock speed*: the step's clock increment
  is alpha^2 L, so rho_alpha = 1 - exp(-alpha^2 L). Any rho keeps p an exact martingale on [0, 1],
  so M1 (and M2) stay arbitrage-free whatever alpha does. rho_SV (the vol/price correlation) is
  defined as in the brief: d<W, B> = rho sgn(Z) dt (rho < 0: vol rises when p moves toward 1/2).
* **M2, jumps** = clock bursts: with probability pi the step's clock increment is multiplied by
  m_J > 1 (a scale mixture of M0 transitions), again an exact martingale.

Data (``prep``). Hourly candles (``kalshi_candles``) for every candle market with metadata and a
yes/no result (7 non-financial categories; Sports is split per row into pre-game and in-play).
A missing hourly candle means the quote did not change (Kalshi omits empty periods), so bid and
ask are forward-filled onto a regular hourly grid between a market's first candle and its close;
candles that end after ``close_time`` (post-close book) are dropped.

Look-ahead guards.
* **Time to expiry.** ``close_time`` of an early-closing market is look-ahead, and
  ``expected_expiration_time`` is the *settlement* schedule, not the end of information
  (weather: close + 11-14 h; Mentions: event end + ~13 days; Elections: months). The scheduled
  information end is ``T = expected_expiration_time - offset(series)``, where the offset is the
  modal (expiration - close) of the series' markets that closed before the holdout (if at least
  half share the mode), else their median over NO-settled markets (which run to the scheduled
  end), else the category median. Offsets use dev markets only. Sports use
  ``expected_expiration_time`` (about game start + 3 h) and the scheduled start from the ticker
  (``kalshi.ticker_start``) or expiration - 3 h. Rows past T (overruns) are dropped from clock work.
* Quotes at t are the candle *close* at t; an increment is (t, t + h]. Every filter on the
  decision row uses only the quote at t; the end quote is only required to exist (spread <= 10c),
  and the share dropped by that measurement filter is reported.
* Dev = increments that end before 2026-10-01; holdout = increments that start on or after it.
  Every parameter (clock, backbone, noise, EWMA, thresholds) is fitted on dev only.

Run: ``python -m albert.research.pm_vol prep|fit|vi|fc|maker|report`` (each a separate transient unit).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import optimize, special, stats

HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
OUT = Path("data/research/pm_vol")
CATS = ("Sports pre-game", "Sports in-play", "Climate and Weather", "Mentions", "Politics",
        "Elections", "Entertainment", "Science and Technology")
BASE_CATS = ("Sports", "Climate and Weather", "Mentions", "Politics", "Elections", "Entertainment",
             "Science and Technology")
START_SPREAD = 0.04          # decision-row quote filter (spread <= 4c)
END_SPREAD = 0.10            # end quote must exist with spread <= 10c (measurement only)
P_LO, P_HI = 0.03, 0.97      # decision-row mid band (1c ticks distort probit/logit beyond it)
P_CLIP = 0.005               # mids are clipped here before Phi^-1 (|Z| <= 2.58)
MIN_TAU_AFTER = 0.5          # the step must end at least 30 min before the scheduled end
TAU_EDGES = (0.0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1e9)       # hours, at the step start
P_EDGES = (0.03, 0.05, 0.08, 0.12, 0.18, 0.26, 0.36, 0.5)              # bins of min(p, 1 - p)
SPORTS_START_OFFSET = timedelta(hours=3)
TAKER, MAKER = 0.07, 0.0175


# ---- fees ----------------------------------------------------------------------------------------
def fee(price: float | np.ndarray, rate: float, mult: float | np.ndarray = 1.0) -> np.ndarray:
    """Kalshi fee per contract, rate * mult * P(1 - P), rounded up to $0.0001 (account precision)."""
    raw = rate * np.asarray(mult) * np.asarray(price) * (1 - np.asarray(price))
    return np.ceil(raw * 1e4 - 1e-9) / 1e4


# ---- M0 math ---------------------------------------------------------------------------------------
def probit(p: np.ndarray | float) -> np.ndarray:
    return special.ndtri(np.clip(p, P_CLIP, 1 - P_CLIP))


def npdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(z) ** 2) / math.sqrt(2 * math.pi)


def clock_power(tau: np.ndarray, h: float | np.ndarray, a: float, b: float) -> np.ndarray:
    """Clock increment L = int_{tau-h}^{tau} a s^-b ds for lambda(tau) = a tau^-b (tau in hours).
    b = 1, a = 1 is the uniform clock L = ln(tau / (tau - h))."""
    tau = np.asarray(tau, dtype=float)
    lo = np.maximum(tau - h, 1e-9)
    e = 1.0 - b
    if abs(e) < 1e-12:
        return a * np.log(tau / lo)
    # expm1 form: accurate through b = 1 (finite-difference gradients cross it)
    return a * (np.expm1(e * np.log(tau)) - np.expm1(e * np.log(lo))) / e


def info_fraction(L: np.ndarray) -> np.ndarray:
    """rho = 1 - V_{t+h}/V_t = 1 - exp(-L)."""
    return -np.expm1(-np.asarray(L, dtype=float))


def m0_moments(z: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of Z_{t+h} | Z_t under M0: N(Z e^{L/2}, e^L - 1)."""
    L = np.asarray(L, dtype=float)
    return np.asarray(z) * np.exp(L / 2), np.expm1(L)


def m0_var_p(p: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Exact Var(p_{t+h} | p_t) under M0 for information fraction rho in [0, 1)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    z = special.ndtri(p)
    rho = np.clip(np.asarray(rho, dtype=float), 0.0, 1 - 1e-15)
    a = np.sqrt((1 - rho) / (1 + rho))
    return np.maximum(p * (1 - p) - 2 * special.owens_t(z, a), 0.0)


def bvn_cdf(h: np.ndarray, k: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Standard bivariate normal CDF Phi2(h, k; r) via Owen's T (vectorised, |r| < 1)."""
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    r = np.clip(np.asarray(r, dtype=float), -1 + 1e-15, 1 - 1e-15)
    h = np.where(h == 0, 1e-12, h)
    k = np.where(k == 0, 1e-12, k)
    q = np.sqrt(1 - r * r)
    c = np.where((h * k > 0) | ((h * k == 0) & (h + k >= 0)), 0.0, 0.5)
    return (0.5 * (special.ndtr(h) + special.ndtr(k)) - special.owens_t(h, (k - r * h) / (h * q))
            - special.owens_t(k, (h - r * k) / (k * q)) - c)


def m0_abs_dp(p: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Exact E|p_{t+h} - p_t| under M0: 2 [Phi2(Z, k; sqrt(rho)) - p Phi(k)],
    k = Z sqrt(rho) / (1 + sqrt(1 - rho))."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    rho = np.clip(np.broadcast_to(np.asarray(rho, dtype=float), p.shape), 1e-15, 1 - 1e-15)
    z = special.ndtri(p)
    k = z * np.sqrt(rho) / (1 + np.sqrt(1 - rho))
    return np.maximum(2 * (bvn_cdf(z, k, np.sqrt(rho)) - p * special.ndtr(k)), 0.0)


def u_stat(z0: np.ndarray, z1: np.ndarray, L: np.ndarray, noise_z: np.ndarray | float = 0.0) -> np.ndarray:
    """Standardised M0 innovation (N(0, 1) under the model; ``noise_z`` adds measurement variance)."""
    mu, var = m0_moments(z0, L)
    return (np.asarray(z1) - mu) / np.sqrt(var + noise_z)


def mixture_rho(rho: np.ndarray, alpha2: float | np.ndarray = 1.0) -> np.ndarray:
    """Information fraction after a stochastic time change of speed alpha^2 (M1): 1 - (1-rho)^alpha^2."""
    L = -np.log1p(-np.clip(rho, 0, 1 - 1e-15))
    return info_fraction(np.asarray(alpha2) * L)


# ---- fitted parameters and the reusable transition density -----------------------------------------
PARAMS_PATH = OUT / "params.json"
_PARAMS: dict[str, Any] | None = None


def load_params(path: Path | None = None) -> dict[str, Any]:
    global _PARAMS
    if path is not None:
        return json.loads(Path(path).read_text())
    if _PARAMS is None:
        _PARAMS = json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else {}
    return _PARAMS


def clock_L(cat_params: dict[str, Any] | None, tau: np.ndarray, h: float,
            hour: np.ndarray | int | None = None) -> np.ndarray:
    """Clock increment for a category's fitted clock (uniform 1/tau if no parameters).

    ``cat_params['clock']`` = {"form": "power", "a", "b", "hour_mult": [24] or None} or
    {"form": "buckets", "edges", "lam": per tau-bucket lambda per hour, "hour_mult"}. The hour
    multiplier (UTC hour of the step start) is only applied for h = 1 steps; longer steps use the
    daily mean (1.0), which is exact for 24 h multiples."""
    tau = np.asarray(tau, dtype=float)
    c = (cat_params or {}).get("clock") or {"form": "power", "a": 1.0, "b": 1.0}
    if c["form"] == "power":
        L = clock_power(tau, h, c["a"], c["b"])
    else:
        edges = np.asarray(c["edges"], dtype=float)
        lam = np.asarray(c["lam"], dtype=float)
        # integrate the piecewise-constant lambda over [tau - h, tau]
        lo = np.maximum(tau - h, 0.0)
        L = np.zeros_like(tau)
        for j in range(len(lam)):
            seg = np.clip(np.minimum(tau, edges[j + 1]) - np.maximum(lo, edges[j]), 0, None)
            L += lam[j] * seg
    hm = c.get("hour_mult")
    if hm is not None and hour is not None and h <= 1.0:
        L = L * np.asarray(hm, dtype=float)[np.asarray(hour, dtype=int) % 24]
    return L


@dataclass
class Transition:
    """Calibrated one-step transition of p over h hours (M0 clock x alpha^2 time change, optional
    M2 jump mixture). Every component is an exact martingale on [0, 1]."""
    p: float
    rho: np.ndarray        # information fraction per mixture component
    weights: np.ndarray    # mixture weights (sum 1)

    @property
    def z(self) -> float:
        return float(special.ndtri(min(max(self.p, 1e-12), 1 - 1e-12)))

    def z_moments(self) -> list[tuple[float, float]]:
        """(mean, sd) of Z_{t+h} per mixture component."""
        return [(self.z / math.sqrt(1 - r), math.sqrt(r / (1 - r))) for r in self.rho]

    @property
    def mean(self) -> float:
        return self.p

    @property
    def var(self) -> float:
        return float(np.sum(self.weights * m0_var_p(np.full(len(self.rho), self.p), self.rho)))

    @property
    def sd(self) -> float:
        return math.sqrt(self.var)

    @property
    def e_abs(self) -> float:
        return float(np.sum(self.weights * m0_abs_dp(np.full(len(self.rho), self.p), self.rho)))

    def cdf(self, x: np.ndarray | float) -> np.ndarray:
        zx = special.ndtri(np.clip(x, 1e-15, 1 - 1e-15))
        return sum(w * special.ndtr((zx - m) / s) for w, (m, s) in zip(self.weights, self.z_moments()))

    def pdf(self, x: np.ndarray | float) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=float), 1e-15, 1 - 1e-15)
        zx = special.ndtri(x)
        dens = sum(w * stats.norm.pdf(zx, m, s) for w, (m, s) in zip(self.weights, self.z_moments()))
        return dens / npdf(zx)

    def quantile(self, q: float) -> float:
        return float(optimize.brentq(lambda x: float(self.cdf(x)) - q, 1e-12, 1 - 1e-12))


def transition(p: float, tau: float, h: float, category: str, alpha_state: float = 1.0, *,
               hour: int | None = None, jumps: bool = False, params: dict[str, Any] | None = None) -> Transition:
    """Calibrated transition density of a Kalshi binary's fair price from p (now, tau hours before
    the scheduled information end) to t + h, for ``category`` (one of ``CATS``).

    ``alpha_state`` is the filtered stochastic-vol state alpha (1 = the clock's average speed);
    ``jumps=True`` mixes in the category's fitted clock-burst component (M2). Parameters come from
    ``data/research/pm_vol/params.json`` (fitted on dev) unless ``params`` is given. Without fitted
    parameters the uniform clock lambda = 1/tau is used."""
    allp = params if params is not None else load_params()
    cp = (allp.get("categories") or {}).get(category)
    if tau <= h:
        raise ValueError("tau must exceed h (the step must end before the scheduled end)")
    L = float(clock_L(cp, np.array([tau]), h, None if hour is None else np.array([hour]))[0])
    L *= alpha_state ** 2 * float((cp or {}).get("scale", 1.0))
    if jumps and cp and cp.get("jump"):
        j = cp["jump"]
        Ls = np.array([j["mult_d"] * L, j["mult_j"] * L])
        w = np.array([1 - j["pi"], j["pi"]])
    else:
        Ls, w = np.array([L]), np.array([1.0])
    return Transition(p=float(p), rho=np.minimum(info_fraction(Ls), 1 - 1e-12), weights=w)


# ---- statistics helpers -----------------------------------------------------------------------------
def clustered(x: np.ndarray, g: np.ndarray) -> dict[str, Any]:
    from albert.research.fundamental import clustered_mean

    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    mu, t, k = clustered_mean(x[ok], np.asarray(g)[ok])
    return {"mean": mu, "t": None if t is None else round(float(t), 2), "n": int(ok.sum()), "clusters": k}


def em_scale_mixture(u: np.ndarray, iters: int = 300, tol: float = 1e-8) -> dict[str, float]:
    """Two-component zero-mean Gaussian scale mixture u ~ (1-pi) N(0, s_d^2) + pi N(0, s_d^2 + s_j^2)
    (Dalen-style jump EM on standardised increments). Returns pi, s_d2, s_j2, the jump variance
    share pi s_j^2 / (s_d^2 + pi s_j^2), and the per-observation posterior jump probability."""
    u2 = np.asarray(u, dtype=float) ** 2
    v = float(u2.mean())
    pi, sd2, sj2 = 0.05, 0.5 * v, 10.0 * v
    ll_old = -np.inf
    for _ in range(iters):
        v1, v2 = sd2, sd2 + sj2
        l1 = (1 - pi) * np.exp(-0.5 * u2 / v1) / np.sqrt(v1)
        l2 = pi * np.exp(-0.5 * u2 / v2) / np.sqrt(v2)
        tot = l1 + l2 + 1e-300
        g = l2 / tot
        ll = float(np.log(tot).sum())
        pi = float(np.clip(g.mean(), 1e-6, 0.5))
        sd2 = float(((1 - g) * u2).sum() / max((1 - g).sum(), 1e-12))
        v2n = float((g * u2).sum() / max(g.sum(), 1e-12))
        sj2 = max(v2n - sd2, 1e-9)
        if abs(ll - ll_old) < tol * max(1.0, abs(ll)):
            break
        ll_old = ll
    return {"pi": pi, "s_d2": sd2, "s_j2": sj2, "jump_share": pi * sj2 / (sd2 + pi * sj2), "gamma": g}


def sv_from_acf(lags: np.ndarray, acf: np.ndarray) -> dict[str, float | None]:
    """Fit corr(|u_t|, |u_{t+k}|) = A phi^k on the given lags; map to log-normal SV (Taylor 1986):
    A = (x - 1)/((pi/2) x - 1), x = exp(sigma_h^2 / 4), sigma_h^2 = Var(log alpha^2). Returns the
    half-life (hours), sigma_h^2, and nu (vol of log alpha) per sqrt(hour) and per sqrt(day)."""
    lags, acf = np.asarray(lags, float), np.asarray(acf, float)
    ok = np.isfinite(acf) & (acf > 0)
    if ok.sum() < 3:
        return {"A": None, "phi": None, "half_life_h": None, "sigma_h2": None, "nu_per_sqrt_h": None,
                "nu_per_sqrt_day": None}

    def loss(th):
        A, phi = th
        return float(((A * phi ** lags[ok] - acf[ok]) ** 2).sum())

    res = optimize.minimize(loss, x0=[max(acf[ok][0], 1e-3), 0.9], bounds=[(1e-6, 0.6), (1e-4, 0.99999)],
                            method="L-BFGS-B")
    A, phi = res.x
    hl = math.log(0.5) / math.log(phi)
    x = (1 - A) / (1 - A * math.pi / 2) if A < 2 / math.pi - 1e-9 else float("nan")
    s2 = 4 * math.log(x) if math.isfinite(x) and x > 1 else float("nan")
    nu = math.sqrt(s2 * (1 - phi ** 2)) / 2 if math.isfinite(s2) else float("nan")
    return {"A": A, "phi": phi, "half_life_h": hl, "sigma_h2": s2, "nu_per_sqrt_h": nu,
            "nu_per_sqrt_day": nu * math.sqrt(24) if math.isfinite(nu) else float("nan")}


def qlike(target: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.log(f) + target / f


# ---- data: series offsets and the hourly panel ------------------------------------------------------
def series_offsets(markets: pl.DataFrame) -> pl.DataFrame:
    """Scheduled information end per series: offset (minutes) = expected_expiration - close, from
    markets that closed before the holdout only. Mode if >= half share it, else NO-market median,
    else the category median."""
    dev = (markets.filter(pl.col("close_time") < HOLDOUT_FROM)
           .with_columns(((pl.col("expected_expiration_time") - pl.col("close_time")).dt.total_minutes()).alias("off")))
    per = dev.group_by("series_ticker", "category").agg(
        pl.col("off").drop_nulls().mode().first().alias("mode"),
        (pl.col("off") == pl.col("off").drop_nulls().mode().first()).mean().alias("mode_share"),
        pl.col("off").filter(pl.col("result") == "no").median().alias("med_no"),
        pl.col("off").median().alias("med_all"), pl.len().alias("n_dev"))
    per = per.with_columns(pl.when(pl.col("mode_share") >= 0.5).then(pl.col("mode").cast(pl.Float64))
                           .otherwise(pl.coalesce("med_no", "med_all")).alias("offset_min"),
                           pl.when(pl.col("mode_share") >= 0.5).then(pl.lit("mode")).otherwise(pl.lit("median")).alias("offset_src"))
    cat_med = per.group_by("category").agg(pl.col("offset_min").median().alias("cat_offset"))
    allser = markets.select("series_ticker", "category").unique("series_ticker")
    out = (allser.join(per.drop("category"), on="series_ticker", how="left").join(cat_med, on="category", how="left")
           .with_columns(pl.when(pl.col("offset_min").is_null()).then(pl.lit("category")).otherwise(pl.col("offset_src")).alias("offset_src"),
                         pl.coalesce("offset_min", "cat_offset").fill_null(0.0).alias("offset_min")))
    # Sports: expected_expiration_time is game start + ~3 h, used as is.
    sports = pl.col("category") == "Sports"
    return out.with_columns(pl.when(sports).then(0.0).otherwise(pl.col("offset_min")).alias("offset_min"),
                            pl.when(sports).then(pl.lit("sports")).otherwise(pl.col("offset_src")).alias("offset_src"))


def market_meta(s) -> pl.DataFrame:
    from albert.venues.kalshi import ticker_start

    cols = ["ticker", "event_ticker", "series_ticker", "category", "open_time", "close_time",
            "expected_expiration_time", "result", "fee_type", "fee_multiplier"]
    allm = (pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet").select(cols)
            .filter(pl.col("category").is_in(BASE_CATS) & pl.col("result").is_in(["yes", "no"])
                    & pl.col("expected_expiration_time").is_not_null()).collect())
    off = series_offsets(allm)
    have = pl.scan_parquet(s.tables_dir / "kalshi_candles.parquet").select("ticker").unique().collect()
    m = allm.join(have, on="ticker", how="inner").join(off.select("series_ticker", "offset_min", "offset_src"),
                                                       on="series_ticker", how="left")
    m = m.with_columns((pl.col("expected_expiration_time") - pl.duration(minutes=pl.col("offset_min").round(0).cast(pl.Int64))).alias("T_sched"),
                       (pl.col("result") == "yes").cast(pl.Int8).alias("y"),
                       pl.col("fee_type").fill_null("").str.contains("maker").alias("maker_fee"),
                       pl.col("fee_multiplier").fill_null(1.0))
    starts = [ticker_start(e) for e in m["event_ticker"].to_list()]
    m = m.with_columns(pl.Series("start_ticker", starts, dtype=pl.Datetime("us", "UTC")))
    m = m.with_columns(pl.when(pl.col("category") == "Sports")
                       .then(pl.coalesce("start_ticker", pl.col("expected_expiration_time") - SPORTS_START_OFFSET))
                       .alias("start"))
    return m


def build_panel(s) -> Path:
    """Regular hourly grid per market (numpy, to stay well under 1 GB): candle quotes forward-filled
    from the first candle to the last candle at or before the close."""
    OUT.mkdir(parents=True, exist_ok=True)
    meta = market_meta(s).sort("ticker")
    meta.write_parquet(OUT / "markets.parquet")
    c = (pl.scan_parquet(s.tables_dir / "kalshi_candles.parquet")
         .select("ticker", "end_ts", "yes_bid_close", "yes_ask_close")
         .join(meta.lazy().select("ticker", "close_time"), on="ticker", how="inner")
         .filter(pl.col("end_ts") <= pl.col("close_time"))
         .select(pl.col("ticker"), (pl.col("end_ts").dt.epoch("s") // 3600).alias("h"),
                 pl.col("yes_bid_close").cast(pl.Float32).alias("bid"),
                 pl.col("yes_ask_close").cast(pl.Float32).alias("ask"))
         .sort("ticker", "h").collect())
    print(f"candles kept {c.height:,} rows, {c['ticker'].n_unique():,} markets", flush=True)
    tickers = meta["ticker"].to_list()
    tix = {t: i for i, t in enumerate(tickers)}
    ci = np.array([tix[t] for t in c["ticker"].to_list()], dtype=np.int32)
    ch = c["h"].to_numpy().astype(np.int64)
    cb, ca = c["bid"].to_numpy(), c["ask"].to_numpy()
    del c
    first = np.full(len(tickers), -1, dtype=np.int64)
    last = np.full(len(tickers), -1, dtype=np.int64)
    bnd = np.r_[0, np.flatnonzero(np.diff(ci)) + 1]
    first[ci[bnd]] = ch[bnd]
    last[ci[np.r_[bnd[1:] - 1, len(ci) - 1]]] = ch[np.r_[bnd[1:] - 1, len(ci) - 1]]
    has = first >= 0
    span = np.where(has, last - first + 1, 0)
    off = np.r_[0, np.cumsum(span)[:-1]]
    n = int(span.sum())
    tk = np.repeat(np.arange(len(tickers), dtype=np.int32), span)
    hh = first[tk] + (np.arange(n, dtype=np.int64) - off[tk])
    pos = off[ci] + (ch - first[ci])
    bid = np.full(n, np.nan, dtype=np.float32)
    ask = np.full(n, np.nan, dtype=np.float32)
    candle = np.zeros(n, dtype=bool)
    bid[pos], ask[pos], candle[pos] = cb, ca, True
    last_idx = np.maximum.accumulate(np.where(candle, np.arange(n), 0))
    bid, ask = bid[last_idx], ask[last_idx]
    del cb, ca, ci, ch, pos, last_idx
    print(f"grid {n:,} rows", flush=True)
    m = meta.with_columns(
        (pl.col("T_sched").dt.epoch("s") / 3600).alias("T_h"),
        (pl.col("start").dt.epoch("s") / 3600).alias("start_h"),
        (pl.col("offset_src") != "category").alias("sched_ok"))
    cat_code = {c: i for i, c in enumerate(CATS)}
    base = m["category"].to_numpy()
    T_h = m["T_h"].to_numpy()
    start_h = m["start_h"].fill_null(np.inf).to_numpy()
    tau = (T_h[tk] - hh).astype(np.float32)
    sports = (base == "Sports")[tk]
    pre = (hh + 1) <= start_h[tk]
    cat = np.array([cat_code.get(b, -1) for b in base], dtype=np.int8)[tk]
    cat = np.where(sports, np.where(pre, cat_code["Sports pre-game"], cat_code["Sports in-play"]), cat).astype(np.int8)
    ev_codes, ev_idx = np.unique(m["event_ticker"].to_numpy(), return_inverse=True)
    frame = pl.DataFrame({
        "tk": tk, "h": hh, "bid": bid, "ask": ask, "candle": candle, "tau": tau,
        "hour": (hh % 24).astype(np.int8), "cat": cat,
        "ev": ev_idx.astype(np.int32)[tk], "y": m["y"].to_numpy().astype(np.int8)[tk],
        "maker_fee": m["maker_fee"].to_numpy()[tk],
        "fee_mult": m["fee_multiplier"].to_numpy().astype(np.float32)[tk],
        "sched_ok": m["sched_ok"].to_numpy()[tk],
    })
    path = OUT / "panel.parquet"
    frame.write_parquet(path)
    pl.DataFrame({"tk": np.arange(len(tickers), dtype=np.int32), "ticker": tickers}).write_parquet(OUT / "tickers.parquet")
    pl.DataFrame({"ev": np.arange(len(ev_codes), dtype=np.int32), "event": ev_codes}).write_parquet(OUT / "events.parquet")
    print(f"panel {frame.height:,} rows -> {path}", flush=True)
    print(frame.group_by("cat").agg(pl.len(), pl.col("tk").n_unique().alias("markets"),
                                    pl.col("candle").mean().alias("candle_share"),
                                    pl.col("sched_ok").mean().alias("sched_ok")).sort("cat"), flush=True)
    return path


# ---- panel arrays and increments ------------------------------------------------------------------
HOLDOUT_H = int(HOLDOUT_FROM.timestamp() // 3600)


def load_panel() -> dict[str, np.ndarray]:
    path = OUT / "panel.parquet"
    a = {}
    for c in pl.read_parquet_schema(path):      # one column at a time keeps the peak low
        a[c] = pl.read_parquet(path, columns=[c])[c].to_numpy()
    a["mid"] = ((a["bid"].astype(np.float64) + a["ask"]) / 2)
    a["spr"] = (a["ask"].astype(np.float64) - a["bid"])
    q = a["spr"]
    a["end_ok"] = (q > 0) & (q <= END_SPREAD + 1e-9) & (a["bid"] >= 0) & (a["ask"] <= 1)
    a["start_ok"] = (a["end_ok"] & (q <= START_SPREAD + 1e-9) & (a["bid"] >= 0.01) & (a["ask"] <= 0.99)
                     & (a["mid"] >= P_LO) & (a["mid"] <= P_HI))
    return a


def lead_index(a: dict[str, np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Index of the row k hours later on the same market, and whether it exists."""
    n = len(a["tk"])
    j = np.minimum(np.arange(n) + k, n - 1)
    ok = (np.arange(n) + k < n) & (a["tk"][j] == a["tk"]) & (a["h"][j] - a["h"] == k)
    return j, ok


def increments(a: dict[str, np.ndarray], cat: int, h: int = 1, *, need_sched: bool = True) -> dict[str, np.ndarray]:
    """h-hour increments of one category from valid decision rows; split by step end/start."""
    j, ok = lead_index(a, h)
    sel = (a["cat"] == cat) & a["start_ok"] & ok & a["end_ok"][j]
    if need_sched:
        sel &= a["sched_ok"] & (a["tau"] - h >= MIN_TAU_AFTER)
    base = (a["cat"] == cat) & a["start_ok"] & ok & (a["sched_ok"] if need_sched else True)
    if need_sched:
        base &= a["tau"] - h >= MIN_TAU_AFTER
    i = np.flatnonzero(sel)
    jj = j[i]
    hh = a["h"][i]
    split = np.where(hh + h <= HOLDOUT_H, 0, np.where(hh >= HOLDOUT_H, 1, -1)).astype(np.int8)
    return {"i": i, "j": jj, "p0": a["mid"][i], "p1": a["mid"][jj], "spr": a["spr"][i],
            "tau": a["tau"][i].astype(np.float64), "hour": a["hour"][i], "ev": a["ev"][i], "tk": a["tk"][i],
            "split": split, "candle1": a["candle"][jj], "dropped_end": int(base.sum() - sel.sum()),
            "base_n": int(base.sum())}


def tau_bucket(tau: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(np.asarray(TAU_EDGES), tau, side="right") - 1, 0, len(TAU_EDGES) - 2)


def _sub(n: int, k: int, seed: int = 11) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


# ---- clock fits (M0 likelihood in probit units) ------------------------------------------------------
CLOCK_FORMS = ("uniform", "scaled", "power", "power_hour", "buckets", "buckets_hour")


def _clock_from_theta(form: str, th: np.ndarray) -> dict[str, Any]:
    hm = None
    if form.endswith("_hour"):
        eta = np.r_[0.0, th[-23:]]
        w = np.exp(eta)
        hm = (w / w.mean()).tolist()
        th = th[:-23]
    if form == "uniform":
        c = {"form": "power", "a": 1.0, "b": 1.0}
    elif form == "scaled":
        c = {"form": "power", "a": float(np.exp(th[0])), "b": 1.0}
    elif form.startswith("power"):
        c = {"form": "power", "a": float(np.exp(th[0])), "b": float(th[1])}
    else:
        c = {"form": "buckets", "edges": list(TAU_EDGES), "lam": np.exp(th).tolist()}
    c["hour_mult"] = hm
    return c


def _theta0(form: str) -> np.ndarray:
    core = {"uniform": [], "scaled": [0.0], "power": [0.0, 1.0], "power_hour": [0.0, 1.0],
            "buckets": [-3.0] * (len(TAU_EDGES) - 1), "buckets_hour": [-3.0] * (len(TAU_EDGES) - 1)}[form]
    return np.r_[core, np.zeros(23) if form.endswith("_hour") else []]


def m0_nll(L: np.ndarray, z0: np.ndarray, z1: np.ndarray, noise_z: np.ndarray) -> np.ndarray:
    L = np.clip(L, 1e-10, 30)
    mu, var = m0_moments(z0, L)
    v = var + noise_z
    return 0.5 * (np.log(2 * np.pi * v) + (z1 - mu) ** 2 / v)


def fit_clock(inc: dict[str, np.ndarray], form: str, idx: np.ndarray, h: int = 1,
              starts: list[np.ndarray] | None = None) -> dict[str, Any]:
    """Maximum likelihood of the exact M0 transition (plus mid-quote noise k * spread^2 in p,
    mapped to Z by 1/phi(Z)^2) on the rows ``idx``. Multi-start; returns the best clock, k, NLL."""
    z0, z1 = probit(inc["p0"][idx]), probit(inc["p1"][idx])
    tau, hour = inc["tau"][idx], inc["hour"][idx]
    nz = inc["spr"][idx] ** 2 / npdf(z0) ** 2

    def obj(th):
        c = _clock_from_theta(form, th[:-1])
        L = clock_L({"clock": c}, tau, h, hour)
        v = float(m0_nll(L, z0, z1, np.exp(th[-1]) * nz).mean())
        return v if np.isfinite(v) else 1e6

    if starts is None:
        base = _theta0(form)
        starts = [np.r_[base, np.log(0.1)]]
        if form.startswith("power"):
            starts = [np.r_[np.log(av), bv, np.zeros(23) if form.endswith("_hour") else [], np.log(0.1)]
                      for av, bv in ((0.5, 0.7), (1.0, 1.3), (5.0, 2.0))]
    best = None
    for th0 in starts:
        res = optimize.minimize(obj, th0, method="L-BFGS-B", options={"maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res
    c = _clock_from_theta(form, best.x[:-1])
    return {"form": form, "clock": c, "k_noise": float(np.exp(best.x[-1])), "nll_fit": float(best.fun),
            "n_params": len(best.x), "converged": bool(best.success), "theta": best.x.tolist()}


def clock_eval(inc: dict[str, np.ndarray], fit: dict[str, Any], idx: np.ndarray, h: int = 1) -> dict[str, np.ndarray]:
    z0, z1 = probit(inc["p0"][idx]), probit(inc["p1"][idx])
    L = clock_L({"clock": fit["clock"]}, inc["tau"][idx], h, inc["hour"][idx])
    nz = fit["k_noise"] * inc["spr"][idx] ** 2 / npdf(z0) ** 2
    return {"nll": m0_nll(L, z0, z1, nz), "u": u_stat(z0, z1, np.clip(L, 1e-10, 30), nz),
            "u_raw": u_stat(z0, z1, np.clip(L, 1e-10, 30)), "L": L, "z0": z0, "z1": z1}


# ---- backbone fits (price space) ---------------------------------------------------------------------
BACKBONES = ("power", "logit_b1", "jacobi_b05", "probit")


def backbone_g2(kind: str, p: np.ndarray, beta: float | None = None) -> np.ndarray:
    q = p * (1 - p)
    if kind == "power":
        return q ** (2 * beta)
    if kind == "logit_b1":
        return q ** 2
    if kind == "jacobi_b05":
        return q
    return npdf(special.ndtri(p)) ** 2


def fit_backbone(inc: dict[str, np.ndarray], kind: str, idx: np.ndarray) -> dict[str, Any]:
    """QLIKE (Gaussian ML) of dp^2 = c(tau bucket)^2 g(p)^2 + k spread^2 on rows ``idx``."""
    p, dp2, s2 = inc["p0"][idx], (inc["p1"][idx] - inc["p0"][idx]) ** 2, inc["spr"][idx] ** 2
    b = tau_bucket(inc["tau"][idx])
    nb = len(TAU_EDGES) - 1
    used = np.unique(b)
    th0 = np.r_[np.full(nb, np.log(1e-4)), np.log(0.05), [1.0] if kind == "power" else []]

    def var(th):
        g2 = backbone_g2(kind, p, th[-1] if kind == "power" else None)
        q0 = np.mean(backbone_g2(kind, np.array([0.5]), th[-1] if kind == "power" else None))
        return np.exp(th[b]) * g2 / q0 + np.exp(th[nb]) * s2

    def obj(th):
        v = var(th)
        return float(np.mean(np.log(v) + dp2 / v))

    res = optimize.minimize(obj, th0, method="L-BFGS-B", options={"maxiter": 500})
    th = res.x
    return {"kind": kind, "beta": float(th[-1]) if kind == "power" else None,
            "c2_at_half": {str(TAU_EDGES[j]): float(np.exp(th[j])) for j in used},
            "k_noise": float(np.exp(th[nb])), "theta": th.tolist(), "qlike_fit": float(res.fun)}


def backbone_var(fit: dict[str, Any], inc: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    th = np.asarray(fit["theta"])
    nb = len(TAU_EDGES) - 1
    kind = fit["kind"]
    p = inc["p0"][idx]
    g2 = backbone_g2(kind, p, fit["beta"])
    q0 = float(np.mean(backbone_g2(kind, np.array([0.5]), fit["beta"])))
    return np.exp(th[tau_bucket(inc["tau"][idx])]) * g2 / q0 + np.exp(th[nb]) * inc["spr"][idx] ** 2


def beta_cells(inc: dict[str, np.ndarray], idx: np.ndarray, a: dict[str, np.ndarray],
               boots: int = 100, seed: int = 3) -> dict[str, Any]:
    """Cell regression log E[dp^2 | tau bucket, p bin] = theta_tau + 2 beta log(p(1-p)), raw and
    Roll-corrected (E dp_t^2 + 2 E dp_t dp_{t+1}, which removes i.i.d. quote noise); event
    multiplier bootstrap (Poisson weights) for the standard error."""
    i = inc["i"][idx]
    j1, ok1 = lead_index(a, 2)
    nxt_ok = ok1[i] & a["end_ok"][j1[i]]
    dp = inc["p1"][idx] - inc["p0"][idx]
    dp_next = np.where(nxt_ok, a["mid"][j1[i]] - inc["p1"][idx], np.nan)
    m = np.minimum(inc["p0"][idx], 1 - inc["p0"][idx])
    pb = np.clip(np.searchsorted(np.asarray(P_EDGES), m, side="right") - 1, 0, len(P_EDGES) - 2)
    tb = tau_bucket(inc["tau"][idx])
    cell = tb * 16 + pb
    ev = inc["ev"][idx]
    _, evi = np.unique(ev, return_inverse=True)
    x_mid = np.array([(P_EDGES[k] + P_EDGES[k + 1]) / 2 for k in range(len(P_EDGES) - 1)])
    roll_ok = np.isfinite(dp_next)

    def est(w):
        out = {}
        for name, y, wm in (("raw", dp ** 2, np.ones_like(dp)), ("roll", np.where(roll_ok, dp ** 2 + 2 * dp * np.nan_to_num(dp_next), 0.0), roll_ok.astype(float))):
            ww = w * wm
            num = np.bincount(cell, weights=ww * y, minlength=16 * 16)
            den = np.bincount(cell, weights=ww, minlength=16 * 16)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = num / den
            c = np.flatnonzero((den >= 30) & (mean > 0))
            if len(c) < 4:
                out[name] = np.nan
                continue
            tb_c, pb_c = c // 16, c % 16
            X = np.zeros((len(c), 16 + 1))
            X[np.arange(len(c)), tb_c] = 1
            X[:, 16] = np.log(x_mid[pb_c] * (1 - x_mid[pb_c]))
            X = X[:, np.r_[np.unique(tb_c), 16]]
            sw = np.sqrt(den[c])
            coef = np.linalg.lstsq(X * sw[:, None], np.log(mean[c]) * sw, rcond=None)[0]
            out[name] = coef[-1] / 2
        return out

    point = est(np.ones(len(dp)))
    rng = np.random.default_rng(seed)
    bs = [est(rng.poisson(1.0, evi.max() + 1)[evi].astype(float)) for _ in range(boots)]
    return {k: {"beta": float(point[k]), "se": float(np.nanstd([b[k] for b in bs]))} for k in point}


# ---- per-category analysis ---------------------------------------------------------------------------
FIT_ROWS = 200_000
WITHIN_MIN_OBS = 48
Z_EDGES = (0.0, 0.5, 1.0, 1.5, 3.0)


def _u_stats(u: np.ndarray) -> dict[str, Any]:
    u = u[np.isfinite(u)]
    if len(u) < 10:
        return {"n": len(u)}
    return {"n": len(u), "var": float(u.var()), "mean": float(u.mean()),
            "exkurt": float(stats.kurtosis(u)), "p_gt2": float((np.abs(u) > 2).mean()),
            "p_gt3": float((np.abs(u) > 3).mean()), "p_gt4": float((np.abs(u) > 4).mean()),
            "zero_share": float((np.abs(u) < 1e-6).mean())}


def drift_test(z0: np.ndarray, z1: np.ndarray, L: np.ndarray, ev: np.ndarray,
               zlag: np.ndarray | None = None, lmax: float = 2.0) -> dict[str, Any]:
    """Slope of (Z1 - Z0) on x = Z0 (e^{L/2} - 1): M0 predicts 1 (the probit drift that keeps p a
    martingale). Steps with L > ``lmax`` (near-certain resolution) are excluded. Quote noise in Z0
    biases OLS towards 0 (errors in variables plus bid-ask bounce), so an IV slope instruments x
    with the previous hour's Z (when given). t-stats of (slope - 1) use event-clustered influence
    functions."""
    keep = np.isfinite(z0) & np.isfinite(z1) & (L <= lmax)
    if zlag is not None:
        keep &= np.isfinite(zlag)
    g = np.expm1(L[keep] / 2)
    x = z0[keep] * g
    y = z1[keep] - z0[keep]
    _, inv = np.unique(ev[keep], return_inverse=True)
    k = inv.max() + 1 if len(inv) else 0

    def slope(w):
        sxw = float((x * w).mean())
        if sxw == 0 or k < 2:
            return None, None
        b = float((w * y).mean() / sxw)
        psi = w * (y - b * x) / sxw
        gs = np.bincount(inv, weights=psi)
        se = math.sqrt((gs ** 2).sum() * k / (k - 1)) / len(x)
        return b, se

    b, se = slope(x)
    out = {"slope": b, "se": se, "t_vs_1": (b - 1) / se if b is not None and se else None, "n": len(x)}
    if zlag is not None:
        bi, sei = slope(zlag[keep] * g)
        out.update({"slope_iv": bi, "se_iv": sei, "t_iv_vs_1": (bi - 1) / sei if bi is not None and sei else None})
    return out


def acf_abs(U: np.ndarray, a: dict[str, np.ndarray], lags: list[int], rows: np.ndarray,
            ev: np.ndarray | None = None, within: bool = False) -> dict[int, dict[str, Any]]:
    """Pooled corr(|u_t|, |u_{t+k}|) within markets; rows = starting rows (positions in U).
    ``within``: demean |u| by each market's own mean first, which removes static differences in
    vol between markets (a market-level alpha) and leaves time variation only (biased down by
    about 1/T per market)."""
    absu = np.abs(U)
    if within:
        f = np.isfinite(absu)
        tk = a["tk"]
        s_ = np.bincount(tk[f], weights=absu[f], minlength=tk.max() + 1)
        c_ = np.bincount(tk[f], minlength=tk.max() + 1)
        absu = absu - (s_ / np.maximum(c_, 1))[tk]
        absu[c_[tk] < WITHIN_MIN_OBS] = np.nan     # short markets: demeaning bias ~ -1/T
    x = absu[rows]
    mu = 0.0 if within else float(np.nanmean(x))
    var = float(np.nanmean((x - mu) ** 2))
    out = {}
    for k in lags:
        j, ok = lead_index(a, k)
        jj = j[rows]
        y = np.where(ok[rows], absu[jj], np.nan)
        prod = (x - mu) * (y - mu) / var
        f = np.isfinite(prod)
        d = {"acf": float(prod[f].mean()) if f.any() else None, "pairs": int(f.sum())}
        if ev is not None and k in (1, 24) and f.sum() > 100:
            d.update({"t": clustered(prod[f], ev[f])["t"]})
        out[k] = d
    return out


def leverage(U: np.ndarray, Z0: np.ndarray, a: dict[str, np.ndarray], rows: np.ndarray,
             ev: np.ndarray, horizon: int = 6, zmax: float = 1.2816, cap: float = 8.0) -> dict[str, Any]:
    """SV/price correlation in the brief's sign convention. x_t = sgn(Z_t) u_t, m_t = mean of
    |u_{t+1..t+6}| (u winsorised at +-8, |Z| <= 1.28 i.e. p in [0.1, 0.9]); theta = Cov(x, m).
    For log-normal SV with d<W,B> = rho sgn(Z) dt and small vol-of-vol,
    Cov(z_t, |u_{t+k}|) ~ (1/2) E|u| rho sigma_w phi^(k-1), so rho ~ 2 theta / (E|u| sigma_w mean phi^(k-1)).
    rho < 0 means vol rises after moves toward 1/2."""
    u = np.clip(U[rows], -cap, cap)
    z = Z0[rows]
    keep = np.isfinite(u) & (np.abs(z) <= zmax)
    fut = np.zeros(len(rows))
    cnt = np.zeros(len(rows))
    for k in range(1, horizon + 1):
        j, ok = lead_index(a, k)
        v = np.abs(np.clip(np.where(ok[rows], U[j[rows]], np.nan), -cap, cap))
        f = np.isfinite(v)
        fut[f] += v[f]
        cnt[f] += 1
    keep &= cnt > 0
    x = np.sign(z[keep]) * u[keep]
    m = fut[keep] / cnt[keep]
    prod = (x - x.mean()) * (m - m.mean())
    r = clustered(prod, ev[keep])
    corr = float(np.corrcoef(x, m)[0, 1]) if keep.sum() > 2 else None
    return {"theta": r["mean"], "t": r["t"], "n": r["n"], "clusters": r["clusters"], "corr": corr,
            "e_abs_u": float(np.abs(u[keep]).mean())}


def analyze_category(a: dict[str, np.ndarray], c: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    name = CATS[c]
    inc = increments(a, c, 1)
    n = len(inc["i"])
    res: dict[str, Any] = {"category": name, "n_steps": n, "dropped_end_quote": inc["dropped_end"],
                           "base_steps": inc["base_n"]}
    if n < 500:
        return {}, res, {}
    dev = np.flatnonzero(inc["split"] == 0)
    ho = np.flatnonzero(inc["split"] == 1)
    res["n_dev"], res["n_holdout"] = len(dev), len(ho)
    res["events_dev"] = len(np.unique(inc["ev"][dev]))
    res["events_holdout"] = len(np.unique(inc["ev"][ho]))
    fit_idx = dev[_sub(len(dev), FIT_ROWS)]
    # -- clock
    fits = {}
    for f in ("uniform", "scaled", "power", "buckets"):
        fits[f] = fit_clock(inc, f, fit_idx)
    for f in ("power", "buckets"):   # hour-of-day variants warm-start from the fitted core
        th = np.asarray(fits[f]["theta"])
        fits[f + "_hour"] = fit_clock(inc, f + "_hour", fit_idx, starts=[np.r_[th[:-1], np.zeros(23), th[-1]]])
    fits = {f: fits[f] for f in CLOCK_FORMS}
    nf = len(fit_idx)
    for f in fits.values():
        f["bic"] = 2 * nf * f["nll_fit"] + f["n_params"] * math.log(nf)
    chosen = min(fits, key=lambda f: fits[f]["bic"])
    res["clock_chosen"] = chosen
    ctab = []
    ev_dev, ev_ho = inc["ev"][dev], inc["ev"][ho]
    evals = {}
    for split, idx, ev in (("dev", dev, ev_dev), ("holdout", ho, ev_ho)):
        if not len(idx):
            continue
        evals[split] = {f: clock_eval(inc, fits[f], idx) for f in CLOCK_FORMS}
        for f in CLOCK_FORMS:
            e = evals[split][f]
            d_uni = clustered(evals[split]["uniform"]["nll"] - e["nll"], ev)
            d_ch = clustered(e["nll"] - evals[split][chosen]["nll"], ev)
            ctab.append({"form": f, "split": split, "nll": float(e["nll"].mean()), "gain_vs_uniform": d_uni["mean"],
                         "t_vs_uniform": d_uni["t"], "loss_vs_chosen": d_ch["mean"], "t_vs_chosen": d_ch["t"],
                         "var_u": float(np.var(e["u"])), "exkurt_u": float(stats.kurtosis(e["u"])),
                         "n": len(idx), "events": len(np.unique(ev))})
    res["clock_table"] = ctab
    res["clock_fits"] = {f: {k: v for k, v in fits[f].items()} for f in CLOCK_FORMS}
    # -- u diagnostics (uniform and chosen), by tau bucket and |Z|
    udiag = []
    for split, idx in (("dev", dev), ("holdout", ho)):
        if split not in evals:
            continue
        for f in ("uniform", chosen):
            e = evals[split][f]
            udiag.append({"split": split, "clock": f, "group": "all", **_u_stats(e["u"]),
                          "var_u_raw": float(np.var(e["u_raw"]))})
            tb = tau_bucket(inc["tau"][idx])
            for b in np.unique(tb):
                m = tb == b
                udiag.append({"split": split, "clock": f, "group": f"tau>={TAU_EDGES[b]:g}h", **_u_stats(e["u"][m])})
            zb = np.clip(np.searchsorted(np.asarray(Z_EDGES), np.abs(e["z0"]), side="right") - 1, 0, len(Z_EDGES) - 2)
            for b in np.unique(zb):
                m = zb == b
                udiag.append({"split": split, "clock": f, "group": f"|Z|>={Z_EDGES[b]:g}", **_u_stats(e["u"][m])})
        e = evals[split][chosen]
        ii = inc["i"][idx]
        prev = np.maximum(ii - 1, 0)
        okp = (ii > 0) & (a["tk"][prev] == a["tk"][ii]) & a["end_ok"][prev]
        zlag = np.where(okp, probit(a["mid"][prev]), np.nan)
        res.setdefault("drift", {})[split] = drift_test(e["z0"], e["z1"], np.clip(e["L"], 1e-10, 30), inc["ev"][idx], zlag)
        eu = evals[split]["uniform"]
        res["drift"][split + "_uniform"] = drift_test(eu["z0"], eu["z1"], np.clip(eu["L"], 1e-10, 30), inc["ev"][idx], zlag)
    res["u_diag"] = udiag
    # -- backbone
    bfits = {k: fit_backbone(inc, k, fit_idx) for k in BACKBONES}
    btab = []
    for split, idx, ev in (("dev", dev, ev_dev), ("holdout", ho, ev_ho)):
        if not len(idx):
            continue
        dp2 = (inc["p1"][idx] - inc["p0"][idx]) ** 2
        losses = {k: qlike(dp2, backbone_var(bfits[k], inc, idx)) for k in BACKBONES}
        for k in BACKBONES:
            d = clustered(losses[k] - losses["probit"], ev)
            btab.append({"backbone": k, "split": split, "beta": bfits[k]["beta"], "qlike": float(losses[k].mean()),
                         "minus_probit": d["mean"], "t_vs_probit": d["t"], "n": len(idx)})
    res["backbone_table"] = btab
    res["beta_cells"] = {split: beta_cells(inc, idx, a) for split, idx in (("dev", dev), ("holdout", ho)) if len(idx) > 1000}
    # -- full-grid u (chosen clock, noise-adjusted) for SV / jumps
    N = len(a["tk"])
    U = np.full(N, np.nan)
    Z0 = np.full(N, np.nan)
    allidx = np.arange(n)
    e_all = clock_eval(inc, fits[chosen], allidx)
    U[inc["i"]] = e_all["u"]
    Z0[inc["i"]] = e_all["z0"]
    Uu = np.full(N, np.nan)
    Uu[inc["i"]] = clock_eval(inc, fits["uniform"], allidx)["u"]
    Ep = np.full(N, np.nan)       # price-space residual standardised by the fitted power backbone
    Ep[inc["i"]] = (inc["p1"] - inc["p0"]) / np.sqrt(backbone_var(bfits["power"], inc, allidx))
    lags = list(range(1, 25)) + [36, 48, 72, 168]
    sv = {}
    for split, idx in (("dev", dev), ("holdout", ho)):
        if len(idx) < 1000:
            continue
        rows = inc["i"][idx]
        evr = inc["ev"][idx]
        acf_c = acf_abs(U, a, lags, rows, evr)
        acf_u = acf_abs(Uu, a, [1, 2, 3, 6, 12, 24], rows)
        acf_w = acf_abs(U, a, list(range(1, 25)), rows, evr, within=True)
        lw = np.arange(1, 25)
        fit_w = sv_from_acf(lw, np.array([acf_w[k]["acf"] if acf_w[k]["acf"] is not None else np.nan for k in lw]))
        act = idx[inc["candle1"][idx]]
        Ua = np.full(N, np.nan)
        Ua[inc["i"][act]] = U[inc["i"][act]]
        acf_a = acf_abs(Ua, a, [1, 2, 3, 6, 12, 24], inc["i"][act])
        lk = np.array([k for k in lags if k <= 24])
        fit_sv = sv_from_acf(lk, np.array([acf_c[k]["acf"] if acf_c[k]["acf"] is not None else np.nan for k in lk]))
        lev = leverage(U, Z0, a, rows, evr)
        lev_p = leverage(Ep, Z0, a, rows, evr)
        phi = fit_sv.get("phi")
        s2 = fit_sv.get("sigma_h2")
        rho = None
        if phi and s2 and math.isfinite(s2) and math.isfinite(lev["theta"]):
            sw = math.sqrt(max(s2 * (1 - phi ** 2), 1e-12))
            rho = float(2 * lev["theta"] / (lev["e_abs_u"] * sw * np.mean(phi ** np.arange(6))))
        if rho is not None and (abs(rho) > 1.5 or (phi or 0) >= 0.999):
            rho = None            # SV fit degenerate: the small-vol-of-vol mapping does not apply
        sv[split] = {"acf_chosen": {k: v for k, v in acf_c.items()}, "acf_uniform": acf_u, "acf_active": acf_a,
                     "acf_within": acf_w, "fit_within": fit_w,
                     "fit": fit_sv, "leverage": lev, "leverage_price_space": lev_p, "rho_implied": rho}
    res["sv"] = sv
    # -- jumps
    jumps = {}
    for split, idx in (("dev", dev), ("holdout", ho)):
        if len(idx) < 1000:
            continue
        u = e_all["u"][idx]
        dp2 = (inc["p1"][idx] - inc["p0"][idx]) ** 2
        moved = dp2 > 0
        em = em_scale_mixture(u[moved])
        tot = (1 - em["pi"]) * em["s_d2"] + em["pi"] * (em["s_d2"] + em["s_j2"])
        jumps[split] = {"pi": em["pi"], "s_d2": em["s_d2"], "s_j2": em["s_j2"], "jump_share_u": em["jump_share"],
                        "mult_d": em["s_d2"] / tot, "mult_j": (em["s_d2"] + em["s_j2"]) / tot,
                        "jump_share_dp2": float((em["gamma"] * dp2[moved]).sum() / dp2.sum()) if dp2.sum() > 0 else None,
                        "jump_rate_per_h": float(em["pi"] * moved.mean()),
                        "share_dp2_ge5c": float(dp2[np.abs(inc["p1"][idx] - inc["p0"][idx]) >= 0.05 - 1e-9].sum() / dp2.sum()) if dp2.sum() > 0 else None,
                        "zero_move_share": float((dp2 == 0).mean()),
                        "big_after_big": float(np.mean(np.abs(U[lead_index(a, 1)[0][inc["i"][idx]]])[np.abs(u) > 3] > 3)) if (np.abs(u) > 3).any() else None,
                        "big_uncond": float(np.mean(np.abs(u) > 3))}
    res["jumps"] = jumps
    jd = jumps.get("dev") or {}
    params = {"clock": fits[chosen]["clock"], "clock_form": chosen, "k_noise": fits[chosen]["k_noise"], "scale": 1.0,
              "backbone_beta": bfits["power"]["beta"], "backbone_theta": bfits["power"]["theta"],
              "jump": {"pi": jd.get("pi"), "mult_d": jd.get("mult_d"), "mult_j": jd.get("mult_j"),
                       "zero_share": jd.get("zero_move_share")} if jd else None,
              "sv": {k: (sv.get("dev") or {}).get("fit", {}).get(k) for k in ("half_life_h", "nu_per_sqrt_h", "phi", "sigma_h2")},
              "sv_within": {k: (sv.get("dev") or {}).get("fit_within", {}).get(k) for k in ("half_life_h", "nu_per_sqrt_h", "phi")},
              "rho": (sv.get("dev") or {}).get("rho_implied")}
    return params, res, {"U": U}


# ---- variance identity -------------------------------------------------------------------------------
def variance_identity(a: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Per market: sum of squared mid changes (sampled every k hours, two-sided quotes with spread
    <= 4c) plus the terminal (y - last mid)^2, against p0(1 - p0). For a martingale both have the
    same expectation; (y - p0)^2 vs p0(1 - p0) checks calibration alone. Split by close date."""
    meta = pl.read_parquet(OUT / "markets.parquet").select("ticker", "category", "close_time")
    tick = pl.read_parquet(OUT / "tickers.parquet")
    meta = tick.join(meta, on="ticker").with_columns((pl.col("close_time") >= HOLDOUT_FROM).alias("ho"))
    good = (a["spr"] <= START_SPREAD + 1e-9) & (a["bid"] >= 0.01) & (a["ask"] <= 0.99)
    f = pl.DataFrame({"tk": a["tk"][good], "h": a["h"][good], "mid": a["mid"][good], "y": a["y"][good],
                      "ev": a["ev"][good], "inplay": (a["cat"][good] == 1)})
    f = f.with_columns((pl.col("h") - pl.col("h").min().over("tk")).alias("age"))
    out = []
    per = []
    for k in (1, 6, 24):
        g = f.filter(pl.col("age") % k == 0).with_columns((pl.col("mid").diff().over("tk") ** 2).alias("d2"))
        m = g.group_by("tk").agg(pl.col("mid").first().alias("p0"), pl.col("mid").last().alias("pl"),
                                 pl.col("y").first(), pl.col("ev").first(), pl.col("d2").sum().alias("S"),
                                 pl.col("d2").filter(pl.col("inplay")).sum().alias("S_inplay"), pl.len().alias("obs"))
        m = m.with_columns(pl.lit(k).alias("k"))
        per.append(m)
    allm = pl.concat(per).join(meta.select("tk", "category", "ho"), on="tk")
    allm = allm.with_columns(((pl.col("y") - pl.col("pl")) ** 2).alias("term"), (pl.col("p0") * (1 - pl.col("p0"))).alias("v0"),
                             ((pl.col("y") - pl.col("p0")) ** 2).alias("calib"))
    allm = allm.with_columns((pl.col("S") + pl.col("term")).alias("total"))
    for (cat, ho), part in allm.partition_by(["category", "ho"], as_dict=True).items():
        for k in (1, 6, 24):
            q = part.filter(pl.col("k") == k)
            if q.height < 20:
                continue
            ev = q["ev"].to_numpy()
            v0 = q["v0"].to_numpy()
            tot = q["total"].to_numpy()
            row = {"category": cat, "split": "holdout" if ho else "dev", "sample_h": k, "markets": q.height,
                   "events": len(np.unique(ev)), "mean_p0q0": float(v0.mean()),
                   "calib_ratio": float(q["calib"].mean() / v0.mean()),
                   "t_calib": clustered(q["calib"].to_numpy() - v0, ev)["t"],
                   "total_ratio": float(tot.mean() / v0.mean()), "t_total": clustered(tot - v0, ev)["t"],
                   "path_ratio": float(q["S"].mean() / v0.mean()), "terminal_share": float(q["term"].mean() / tot.mean()),
                   "inplay_share": float(q["S_inplay"].mean() / tot.mean()) if cat == "Sports" else None,
                   "median_obs": float(q["obs"].median())}
            out.append(row)
    return sorted(out, key=lambda r: (r["category"], r["split"], r["sample_h"]))


# ---- forecasting next-period realised variance ---------------------------------------------------------
FC_MODELS = ("HIST", "M0", "BB", "M1", "M1BB")
EWMA_HL = (3, 12, 48)
SV_W = (0.25, 0.5, 0.75, 1.0)
HIST_W = (6, 24, 72)


def _row_L(a: dict[str, np.ndarray], params: dict[str, Any], rows: np.ndarray, H: int) -> np.ndarray:
    """Sum of hourly clock increments over the next H hours for each row (its category's clock)."""
    L = np.zeros(len(rows))
    cat = a["cat"][rows]
    for c in np.unique(cat):
        cp = params.get(CATS[c])
        m = cat == c
        r = rows[m]
        for j in range(H):
            L[m] += clock_L(cp, a["tau"][r].astype(float) - j, 1, (a["hour"][r].astype(int) + j) % 24)
    return L


def _bb_base(a: dict[str, np.ndarray], params: dict[str, Any], rows: np.ndarray, H: int) -> np.ndarray:
    out = np.zeros(len(rows))
    cat = a["cat"][rows]
    nb = len(TAU_EDGES) - 1
    for c in np.unique(cat):
        cp = params.get(CATS[c])
        m = cat == c
        r = rows[m]
        th = np.asarray(cp["backbone_theta"])
        beta = cp["backbone_beta"]
        g2 = backbone_g2("power", a["mid"][r], beta) / backbone_g2("power", np.array([0.5]), beta)[0]
        for j in range(H):
            out[m] += np.exp(th[np.clip(tau_bucket(a["tau"][r].astype(float) - j), 0, nb - 1)]) * g2
    return out


def _calibrate(base: np.ndarray, s2H: np.ndarray, rv: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """f = e^c1 base + e^c0 spread^2 H + e^c2 1e-6, QLIKE-optimal on rows idx."""
    def f(th, i):
        return np.exp(th[1]) * base[i] + np.exp(th[0]) * s2H[i] + np.exp(th[2]) * 1e-6

    def obj(th):
        v = f(th, idx)
        return float(np.mean(np.log(v) + rv[idx] / v))

    res = optimize.minimize(obj, np.array([np.log(0.1), 0.0, 0.0]), method="L-BFGS-B")
    return res.x


def group_bounds(tk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: index of its market's first row and one past its last row (rows sorted by market)."""
    n = len(tk)
    starts = np.r_[0, np.flatnonzero(np.diff(tk)) + 1]
    ends = np.r_[starts[1:], n]
    reps = ends - starts
    return np.repeat(starts, reps), np.repeat(ends, reps)


def trailing_sum(x: np.ndarray, gs: np.ndarray, w: int) -> np.ndarray:
    """Sum of x over the previous w rows of the same market (rows t-w .. t-1)."""
    cs = np.r_[0.0, np.cumsum(x)]
    i = np.arange(len(x))
    lo = np.maximum(i - w, gs)
    return cs[i] - cs[lo]


def forward_sum(x: np.ndarray, ge: np.ndarray, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum of x over rows t .. t+H-1 of the same market, and whether all H rows exist."""
    cs = np.r_[0.0, np.cumsum(x)]
    i = np.arange(len(x))
    hi = np.minimum(i + H, ge)
    return cs[hi] - cs[i], i + H <= ge


def ewma_past(x: np.ndarray, tk: np.ndarray, half_lives: tuple[int, ...]) -> dict[int, np.ndarray]:
    """Causal EWMA per market of the non-NaN x *before* each row (weights decay per observation,
    normalised like polars ``ewm_mean(adjust=True, ignore_nulls=True)``); NaN until the first one.
    Vectorised: one IIR pass (lfilter) over the observations, minus the carry-over from earlier
    markets, which is exact because the filter is linear."""
    from scipy.signal import lfilter

    pos = np.flatnonzero(np.isfinite(x))
    xo = x[pos].astype(np.float64)
    to = tk[pos]
    n = len(x)
    out = {}
    if not len(pos):
        return {hl: np.full(n, np.nan, dtype=np.float32) for hl in half_lives}
    first = np.r_[0, np.flatnonzero(np.diff(to)) + 1]
    gfirst = np.repeat(first, np.diff(np.r_[first, len(to)]))
    j = np.arange(len(to))
    # last observation strictly before each row, if it belongs to the same market
    last = np.searchsorted(pos, np.arange(n), side="left") - 1
    ok = last >= 0
    ok[ok] = to[last[ok]] == tk[ok]
    for hl in half_lives:
        d = math.exp(-math.log(2) / hl)
        num = lfilter([1.0], [1.0, -d], xo)
        den = lfilter([1.0], [1.0, -d], np.ones_like(xo))
        carry = gfirst > 0
        decay = np.where(carry, d ** (j - gfirst + 1).astype(np.float64), 0.0)
        prev = np.maximum(gfirst - 1, 0)
        num = num - decay * num[prev]
        den = den - decay * den[prev]
        v = np.full(n, np.nan, dtype=np.float32)
        v[ok] = (num[last[ok]] / den[last[ok]]).astype(np.float32)
        out[hl] = v
    return out


def forecast_eval(a: dict[str, np.ndarray], params: dict[str, Any], Hs: tuple[int, ...] = (1, 6)) -> dict[str, Any]:
    N = len(a["tk"])
    j1, ok1 = lead_index(a, 1)
    has_clock = np.zeros(N, dtype=bool)
    for c, name in enumerate(CATS):
        if params.get(name):
            has_clock |= a["cat"] == c
    step_ok = ok1 & a["end_ok"] & a["end_ok"][j1] & (a["mid"] - a["spr"] / 2 >= 0.01 - 1e-9) & (a["mid"] + a["spr"] / 2 <= 0.99 + 1e-9)
    dp2 = np.where(step_ok, (a["mid"][j1] - a["mid"]) ** 2, np.nan)
    del j1, ok1
    # one-step M0 variance (+ noise) on every row that has a clock, for the alpha filter
    v1 = np.full(N, np.nan)
    rr = np.flatnonzero(step_ok & has_clock & a["sched_ok"] & (a["tau"] - 1 >= MIN_TAU_AFTER))
    del step_ok
    L1 = _row_L(a, params, rr, 1)
    kn = np.array([params[CATS[c]]["k_noise"] if params.get(CATS[c]) else 0.0 for c in range(len(CATS))])
    v1[rr] = m0_var_p(np.clip(a["mid"][rr], 1e-4, 1 - 1e-4), info_fraction(L1)) + kn[a["cat"][rr]] * a["spr"][rr] ** 2
    ratio = dp2 / v1
    del v1, L1
    gs, ge = group_bounds(a["tk"])
    age = a["h"] - a["h"][gs]
    ew_e = ewma_past(ratio, a["tk"], EWMA_HL)
    del ratio
    fin = np.isfinite(dp2)
    x0 = np.where(fin, dp2, 0.0)
    ew_r = {w: trailing_sum(x0, gs, w) / trailing_sum(fin.astype(float), gs, w) for w in HIST_W}
    out: dict[str, Any] = {"tables": [], "choices": {}}
    for H in Hs:
        rv_all, full = forward_sum(x0, ge, H)
        cnt, _ = forward_sum(fin.astype(float), ge, H)
        rows = np.flatnonzero(a["start_ok"] & a["sched_ok"] & has_clock & (a["tau"] - H >= MIN_TAU_AFTER)
                              & full & (cnt == H) & (age >= 24))
        del cnt
        e = {hl: ew_e[hl][rows].astype(np.float64) for hl in EWMA_HL}
        r = {w: ew_r[w][rows].astype(np.float64) for w in HIST_W}
        good = np.all([np.isfinite(v) for v in e.values()], axis=0) & np.all([np.isfinite(v) for v in r.values()], axis=0)
        rows = rows[good]
        e = {k: np.clip(v[good], 0.02, 50) for k, v in e.items()}
        r = {k: v[good] for k, v in r.items()}
        rv = rv_all[rows] * 1e4          # cents^2
        s2H = a["spr"][rows] ** 2 * H * 1e4
        LH = _row_L(a, params, rows, H)
        m0 = m0_var_p(a["mid"][rows], info_fraction(LH)) * 1e4
        bb = _bb_base(a, params, rows, H) * 1e4
        hh = a["h"][rows]
        split = np.where(hh + H <= HOLDOUT_H, 0, np.where(hh >= HOLDOUT_H, 1, -1))
        cat = a["cat"][rows]
        ev = a["ev"][rows]
        for c in np.unique(cat):
            ic = np.flatnonzero(cat == c)
            sp_c, ev_c, rv_c, s2_c = split[ic], ev[ic], rv[ic], s2H[ic]
            m0_c, bb_c = m0[ic], bb[ic]
            e_c = {hl: e[hl][ic] for hl in EWMA_HL}
            dev = np.flatnonzero(sp_c == 0)
            ho = np.flatnonzero(sp_c == 1)
            if len(dev) < 2000:
                continue
            fit_i = dev[_sub(len(dev), 300_000)]

            def cands(mname):    # candidate base forecasts, generated lazily (memory)
                if mname == "HIST":
                    for w in HIST_W:
                        yield f"W={w}", r[w][ic] * H * 1e4
                elif mname in ("M0", "BB"):
                    yield "", (m0_c if mname == "M0" else bb_c)
                else:
                    base = m0_c if mname == "M1" else bb_c
                    for hl in EWMA_HL:
                        for w in SV_W:
                            yield f"hl={hl},w={w}", base * ((1 - w) + w * e_c[hl])

            preds = {}
            for mname in FC_MODELS:
                best = None
                for tag, base in cands(mname):
                    th = _calibrate(base, s2_c, rv_c, fit_i)
                    f = np.exp(th[1]) * base + np.exp(th[0]) * s2_c + np.exp(th[2]) * 1e-6
                    lv = float(np.mean(qlike(rv_c[fit_i], f[fit_i])))
                    if best is None or lv < best[0]:
                        best = (lv, tag, f, th)
                preds[mname] = best[2]
                out["choices"][f"{CATS[c]}|H={H}|{mname}"] = {"choice": best[1], "coef": np.exp(best[3]).tolist()}
            for split_name, idx in (("dev", dev), ("holdout", ho)):
                if len(idx) < 200:
                    continue
                ql = {m: qlike(rv_c[idx], preds[m][idx]) for m in FC_MODELS}
                se = {m: (rv_c[idx] - preds[m][idx]) ** 2 for m in FC_MODELS}
                row = {"category": CATS[c], "H": H, "split": split_name, "n": len(idx),
                       "events": len(np.unique(ev_c[idx])), "mean_rv_c2": float(rv_c[idx].mean())}
                for m in FC_MODELS:
                    row[f"qlike_{m}"] = float(ql[m].mean())
                    row[f"mse_rel_{m}"] = float(se[m].mean() / se["HIST"].mean())
                for m1, m2 in (("M1", "HIST"), ("M1", "M0"), ("M0", "HIST"), ("M1BB", "M1"), ("BB", "M0")):
                    dq = clustered(ql[m1] - ql[m2], ev_c[idx])
                    dm = clustered(se[m1] - se[m2], ev_c[idx])
                    row[f"dQ_{m1}-{m2}"] = dq["mean"]
                    row[f"tQ_{m1}-{m2}"] = dq["t"]
                    row[f"tMSE_{m1}-{m2}"] = dm["t"]
                out["tables"].append(row)
            del preds
    return out


# ---- economic check: model move vs quoted half-spread ------------------------------------------------------
TAU_GROUPS = ((1.5, 6, "<6h"), (6, 24, "6-24h"), (24, 168, "1-7d"), (168, 1e9, ">7d"))
P_GROUPS = ((0.03, 0.10, "tail"), (0.10, 0.30, "mid-tail"), (0.30, 0.50001, "centre"))


def expected_move(a: dict[str, np.ndarray], params: dict[str, Any], rows: np.ndarray,
                  mixture: bool = True) -> np.ndarray:
    """Model E|p_{t+1h} - p_t| of the quoted mid. ``mixture``: the category's dev-fitted M2 scale
    mixture (zero-move mass z, then diffusive / jump components of the moved hours), whose total
    variance equals the clock's; otherwise the Gaussian M0 value."""
    L = _row_L(a, params, rows, 1)
    p = a["mid"][rows]
    if not mixture:
        return m0_abs_dp(p, info_fraction(L))
    out = np.zeros(len(rows))
    cat = a["cat"][rows]
    for c in np.unique(cat):
        m = cat == c
        j = (params.get(CATS[c]) or {}).get("jump") or {}
        if not j.get("pi"):
            out[m] = m0_abs_dp(p[m], info_fraction(L[m]))
            continue
        z = j.get("zero_share") or 0.0
        sc = 1.0 / max(1 - z, 1e-3)
        out[m] = (1 - z) * ((1 - j["pi"]) * m0_abs_dp(p[m], info_fraction(L[m] * j["mult_d"] * sc))
                            + j["pi"] * m0_abs_dp(p[m], info_fraction(L[m] * j["mult_j"] * sc)))
    return out


def economic_table(a: dict[str, np.ndarray], params: dict[str, Any]) -> list[dict[str, Any]]:
    j1, ok1 = lead_index(a, 1)
    has = np.zeros(len(a["tk"]), dtype=bool)
    for c, name in enumerate(CATS):
        if params.get(name):
            has |= a["cat"] == c
    rows = np.flatnonzero(has & a["sched_ok"] & (a["bid"] >= 0.01) & (a["ask"] <= 0.99) & (a["ask"] > a["bid"])
                          & (a["mid"] >= P_LO) & (a["mid"] <= P_HI) & (a["tau"] >= 1.5))
    em = expected_move(a, params, rows)
    em0 = expected_move(a, params, rows, mixture=False)
    realized = np.where(ok1[rows] & a["end_ok"][j1[rows]], np.abs(a["mid"][j1[rows]] - a["mid"][rows]), np.nan)
    hs = a["spr"][rows] / 2
    m = np.minimum(a["mid"][rows], 1 - a["mid"][rows])
    tau = a["tau"][rows]
    cat = a["cat"][rows]
    out = []
    for c in np.unique(cat):
        for lo, hi, tn in TAU_GROUPS:
            for plo, phi, pn in P_GROUPS:
                sel = (cat == c) & (tau >= lo) & (tau < hi) & (m >= plo) & (m < phi)
                if sel.sum() < 200:
                    continue
                rz = realized[sel]
                out.append({"category": CATS[c], "tau": tn, "p": pn, "n": int(sel.sum()),
                            "median_half_spread_c": float(np.median(hs[sel]) * 100),
                            "mean_model_abs_move_c": float(em[sel].mean() * 100),
                            "mean_m0_gauss_abs_move_c": float(em0[sel].mean() * 100),
                            "mean_realized_abs_move_c": float(np.nanmean(rz) * 100) if np.isfinite(rz).any() else None,
                            "median_ratio_hs_to_model": float(np.median(hs[sel] / np.maximum(em[sel], 1e-6))),
                            "share_hs_gt_2x_model": float(np.mean(hs[sel] > 2 * em[sel]))})
    return out


# ---- price martingale check ----------------------------------------------------------------------------------
MG_BINS = (0.03, 0.06, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.94, 0.97 + 1e-9)


def martingale_table(a: dict[str, np.ndarray], h: int = 6) -> list[dict[str, Any]]:
    """E[mid_{t+h} - mid_t | mid_t bin] in cents (M0 and M1 both say 0), and the settlement
    analogue E[y - mid_t], event-clustered. Decision rows as for the clock fits."""
    out = []
    for c in range(len(CATS)):
        inc = increments(a, c, h, need_sched=False)
        if len(inc["i"]) < 1000:
            continue
        dp = (inc["p1"] - inc["p0"]) * 100
        ys = (a["y"][inc["i"]] - inc["p0"]) * 100
        b = np.clip(np.searchsorted(np.asarray(MG_BINS), inc["p0"], side="right") - 1, 0, len(MG_BINS) - 2)
        for sp, sv in (("dev", 0), ("holdout", 1)):
            for k in range(len(MG_BINS) - 1):
                m = (inc["split"] == sv) & (b == k)
                if m.sum() < 200:
                    continue
                d = clustered(dp[m], inc["ev"][m])
                yy = clustered(ys[m], inc["ev"][m])
                out.append({"category": CATS[c], "split": sp, "h": h, "p_from": MG_BINS[k], "p_to": round(MG_BINS[k + 1], 2),
                            "n": d["n"], "events": d["clusters"], "mean_dp_c": d["mean"], "t_dp": d["t"],
                            "mean_y_minus_p_c": yy["mean"], "t_y": yy["t"]})
    return out


# ---- maker check: touch quotes where the half-spread dwarfs the model move ----------------------------------
K_GRID = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
HS_GRID = (0.005, 0.01, 0.015, 0.025, 0.05, 0.1)


def trade_extremes(s, tickers: list[str]) -> pl.DataFrame:
    """Per (ticker, decision hour t): lowest YES price paid by NO-takers and highest YES price paid by
    YES-takers in (t, t + 1 h]."""
    tr = (pl.scan_parquet(s.tables_dir / "kalshi_trades.parquet")
          .select("ticker", "created_time", "yes_price", "taker_side")
          .filter(pl.col("ticker").is_in(tickers))
          .with_columns(((pl.col("created_time").dt.epoch("us") + 3_600_000_000 - 1) // 3_600_000_000 - 1).alias("h"))
          .group_by("ticker", "h").agg(pl.col("yes_price").filter(pl.col("taker_side") == "no").min().alias("min_no_px"),
                                       pl.col("yes_price").filter(pl.col("taker_side") == "yes").max().alias("max_yes_px"))
          .collect())
    return tr


def trade_coverage(s, tickers: list[str]) -> pl.DataFrame:
    """Hours for which the trades table is complete: the ingest window if one exists, else from
    the first recorded trade to the close."""
    tr = (pl.scan_parquet(s.tables_dir / "kalshi_trades.parquet").filter(pl.col("ticker").is_in(tickers))
          .group_by("ticker").agg(pl.col("created_time").min().alias("first")).collect())
    w = pl.read_parquet(s.tables_dir / "kalshi_trade_windows.parquet").select("ticker", "window_start", "window_end")
    mk = pl.read_parquet(OUT / "markets.parquet").select("ticker", "close_time")
    cov = tr.join(w, on="ticker", how="left").join(mk, on="ticker", how="left")
    return cov.select("ticker", pl.coalesce("window_start", "first").alias("cov_start"),
                      pl.coalesce("window_end", "close_time").alias("cov_end"))


def maker_test(s, a: dict[str, np.ndarray], params: dict[str, Any]) -> dict[str, Any]:
    tick = pl.read_parquet(OUT / "tickers.parquet")
    names = tick["ticker"].to_list()
    trade_tk = (pl.scan_parquet(s.tables_dir / "kalshi_trades.parquet").select("ticker").unique().collect()
                .join(tick, on="ticker", how="inner"))
    cov = trade_coverage(s, trade_tk["ticker"].to_list()).join(tick, on="ticker")
    cs = np.full(len(names), np.iinfo(np.int64).max)
    ce = np.full(len(names), np.iinfo(np.int64).min)
    cs[cov["tk"].to_numpy()] = np.ceil(cov["cov_start"].dt.epoch("s").to_numpy() / 3600).astype(np.int64)
    ce[cov["tk"].to_numpy()] = (cov["cov_end"].dt.epoch("s").to_numpy() // 3600).astype(np.int64)
    has = np.zeros(len(a["tk"]), dtype=bool)
    for c, name in enumerate(CATS):
        if params.get(name):
            has |= a["cat"] == c
    rows = np.flatnonzero(has & a["sched_ok"] & (a["bid"] >= 0.01) & (a["ask"] <= 0.99) & (a["ask"] > a["bid"])
                          & (a["mid"] >= P_LO) & (a["mid"] <= P_HI) & (a["tau"] - 1 >= MIN_TAU_AFTER)
                          & (a["h"] >= cs[a["tk"]]) & (a["h"] + 1 <= ce[a["tk"]]))
    print(f"maker universe: {len(rows):,} decision hours on {len(np.unique(a['tk'][rows])):,} markets", flush=True)
    ext = trade_extremes(s, sorted({names[t] for t in np.unique(a["tk"][rows])})).join(tick, on="ticker")
    key = pl.DataFrame({"tk": a["tk"][rows], "h": a["h"][rows], "r": np.arange(len(rows))})
    j = key.join(ext.select("tk", "h", "min_no_px", "max_yes_px"), on=["tk", "h"], how="left").sort("r")
    min_no = j["min_no_px"].fill_null(9.0).to_numpy()
    max_yes = j["max_yes_px"].fill_null(-9.0).to_numpy()
    bid, ask = a["bid"][rows].astype(float), a["ask"][rows].astype(float)
    bid, ask = np.round(bid, 2), np.round(ask, 2)
    y = a["y"][rows].astype(float)
    mf, mult = a["maker_fee"][rows], a["fee_mult"][rows].astype(float)
    fy = np.where(mf, fee(bid, MAKER, mult), 0.0)
    fn = np.where(mf, fee(1 - ask, MAKER, mult), 0.0)
    fill_y = min_no < bid - 1e-9
    fill_n = max_yes > ask + 1e-9
    pnl_y = np.where(fill_y, y - bid - fy, 0.0)
    pnl_n = np.where(fill_n, (1 - y) - (1 - ask) - fn, 0.0)
    em = expected_move(a, params, rows)
    ratio = (ask - bid) / 2 / np.maximum(em, 1e-6)
    hs = (ask - bid) / 2
    split = np.where(a["h"][rows] + 1 <= HOLDOUT_H, 0, np.where(a["h"][rows] >= HOLDOUT_H, 1, -1))
    ev = a["ev"][rows]
    cat = a["cat"][rows]

    def summarize(mask: np.ndarray, label: str, extra: dict[str, Any]) -> list[dict[str, Any]]:
        res = []
        for sn, sv in (("dev", 0), ("holdout", 1)):
            m = mask & (split == sv)
            orders = 2 * int(m.sum())
            fy_, fn_ = m & fill_y, m & fill_n
            fills = int(fy_.sum() + fn_.sum())
            pnl = np.r_[pnl_y[fy_], pnl_n[fn_]]
            g = np.r_[ev[fy_], ev[fn_]]
            row = {"rule": label, **extra, "split": sn, "decision_hours": int(m.sum()),
                   "share_selected": float(m.sum() / max((split == sv).sum(), 1)), "orders": orders, "fills": fills,
                   "fill_rate": fills / orders if orders else None}
            if fills:
                cm = clustered(pnl * 100, g)
                per_order = np.r_[pnl_y[m] + pnl_n[m]] * 100 / 2
                co = clustered(per_order, ev[m])
                row.update({"pnl_per_fill_c": cm["mean"], "t_fill": cm["t"], "events": cm["clusters"],
                            "pnl_per_order_c": co["mean"], "t_order": co["t"],
                            "yes_fill_share": float(fy_.sum() / fills)})
            res.append(row)
        return res

    table = summarize(np.ones(len(rows), dtype=bool), "touch, unconditional", {"k": None})
    best = None
    for k in K_GRID:
        part = summarize(ratio >= k, "touch, half-spread >= k x model E|dp|", {"k": k})
        table += part
        d = part[0]
        if d["fills"] >= 200 and d.get("pnl_per_fill_c") is not None and (best is None or d["pnl_per_fill_c"] > best[1]):
            best = (k, d["pnl_per_fill_c"])
    # control: the same rule family thresholded on the raw half-spread (no model), chosen on dev by
    # the same criterion
    ctrl, best_c = [], None
    for thr in HS_GRID:
        part = summarize(hs >= thr - 1e-9, "touch, half-spread >= threshold (no model)", {"k": None, "hs_thr_c": thr * 100})
        ctrl += part
        d = part[0]
        if d["fills"] >= 200 and d.get("pnl_per_fill_c") is not None and (best_c is None or d["pnl_per_fill_c"] > best_c[1]):
            best_c = (thr, d["pnl_per_fill_c"])
    bycat = []
    for c in np.unique(cat):
        bycat += summarize(cat == c, f"unconditional | {CATS[c]}", {"k": None})
        if best is not None:
            bycat += summarize((cat == c) & (ratio >= best[0]), f"model k={best[0]} | {CATS[c]}", {"k": best[0]})
    return {"table": table + ctrl, "by_category": bycat, "k_chosen_on_dev": None if best is None else best[0],
            "hs_threshold_chosen_on_dev": None if best_c is None else best_c[0],
            "universe_hours": len(rows)}


# ---- drivers -----------------------------------------------------------------------------------------------
def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_fit(s) -> None:
    """Per-category fits (dev) and diagnostics -> params.json, results_cat.json."""
    a = load_panel()
    params: dict[str, Any] = {"categories": {}, "fitted_on": f"increments ending before {HOLDOUT_FROM.date()}",
                              "units": "tau, h in hours; clock L = int lambda dt; k_noise multiplies spread^2 (price units)"}
    results: dict[str, Any] = {"categories": {}}
    for c, name in enumerate(CATS):
        print(f"== {name}", flush=True)
        pc, rc, _ = analyze_category(a, c)
        if pc:
            params["categories"][name] = pc
        results["categories"][name] = rc
        (OUT / "results_cat.json").write_text(json.dumps(_jsonable(results)))
        print(f"   steps {rc.get('n_steps')} chosen {rc.get('clock_chosen')}", flush=True)
    PARAMS_PATH.write_text(json.dumps(_jsonable(params), indent=1))


def run_vi(s) -> None:
    """Variance identity per market -> results_vi.json."""
    a = load_panel()
    (OUT / "results_vi.json").write_text(json.dumps(_jsonable({"variance_identity": variance_identity(a)})))
    print("done", flush=True)


def run_fc(s) -> None:
    """Variance forecasts with the dev-fitted parameters -> results_fc.json."""
    a = load_panel()
    for k in ("bid", "ask", "candle", "y", "maker_fee", "fee_mult"):
        a.pop(k)
    params = load_params(PARAMS_PATH)["categories"]
    out = {"forecast": forecast_eval(a, params)}
    (OUT / "results_fc.json").write_text(json.dumps(_jsonable(out)))
    print("done", flush=True)


def run_maker(s) -> None:
    a = load_panel()
    params = load_params(PARAMS_PATH)["categories"]
    out = maker_test(s, a, params)
    out["economic"] = economic_table(a, params)
    out["martingale"] = martingale_table(a)
    (OUT / "maker.json").write_text(json.dumps(_jsonable(out)))
    print(pl.DataFrame(out["table"], infer_schema_length=None))


# ---- report ------------------------------------------------------------------------------------------------
def _r(v: Any, d: int = 4) -> Any:
    return round(v, d) if isinstance(v, float) else v


def _frame(rows: list[dict[str, Any]], cols: list[str] | None = None, d: int = 4) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    rows = [{k: _r(v, d) for k, v in r.items() if cols is None or k in cols} for r in rows]
    f = pl.DataFrame(rows, infer_schema_length=None)
    return f.select([c for c in (cols or f.columns) if c in f.columns])


def schedule_diagnostic() -> pl.DataFrame:
    m = pl.read_parquet(OUT / "markets.parquet")
    d = m.filter((pl.col("result") == "no") & (pl.col("offset_src") != "category")).with_columns(
        ((pl.col("close_time") - pl.col("T_sched")).dt.total_minutes() / 60).alias("err_h"),
        pl.when(pl.col("close_time") >= HOLDOUT_FROM).then(pl.lit("holdout")).otherwise(pl.lit("dev")).alias("split"))
    return (d.group_by("category", "split").agg(pl.len().alias("no_markets"),
                                                *[pl.col("err_h").quantile(q).round(2).alias(f"q{int(q * 100)}") for q in (0.1, 0.25, 0.5, 0.75, 0.9)])
            .sort("category", "split"))


def write_report(s) -> Path:
    from albert.reports import Report

    res = json.loads((OUT / "results_cat.json").read_text())
    for extra in ("results_fc.json", "results_vi.json"):
        if (OUT / extra).exists():
            res.update(json.loads((OUT / extra).read_text()))
    mk = json.loads((OUT / "maker.json").read_text()) if (OUT / "maker.json").exists() else {}
    params = load_params(PARAMS_PATH)["categories"]
    cats = res["categories"]
    rep = Report(s.reports_dir, "research-pm-vol")
    if (OUT / "verdict.md").exists():
        rep.text("Verdict", (OUT / "verdict.md").read_text())
    rep.text("Scope and guards", __doc__.split("Run:")[0].strip().replace("\n", " \n"))
    rep.table("Scheduled information end: close_time - T_sched (h) for NO-settled markets (0 = schedule exact)",
              schedule_diagnostic())
    steps = [{"category": c, "steps": r.get("n_steps"), "dev": r.get("n_dev"), "holdout": r.get("n_holdout"),
              "events_dev": r.get("events_dev"), "events_holdout": r.get("events_holdout"),
              "end_quote_dropped_share": (r.get("dropped_end_quote") or 0) / max(r.get("base_steps") or 1, 1),
              "clock_chosen_on_dev": r.get("clock_chosen")} for c, r in cats.items()]
    rep.table("Hourly steps (decision quote spread <= 4c, mid in [3c, 97c], scheduled end known)", _frame(steps))

    # fitted parameters
    prow = []
    for c, r in cats.items():
        if not r.get("clock_fits"):
            continue
        pf = r["clock_fits"]["power"]["clock"]
        ud = {(x["split"], x["clock"], x["group"]): x for x in r.get("u_diag", [])}
        ch = r["clock_chosen"]
        sv = r.get("sv", {})
        jd, jh = r.get("jumps", {}).get("dev", {}), r.get("jumps", {}).get("holdout", {})
        bc = r.get("beta_cells", {})
        prow.append({
            "category": c, "clock": ch, "power_a": pf["a"], "power_b": pf["b"],
            "beta_qlike_dev": params.get(c, {}).get("backbone_beta"),
            "beta_cells_dev_roll": (bc.get("dev") or {}).get("roll", {}).get("beta"),
            "beta_cells_ho_roll": (bc.get("holdout") or {}).get("roll", {}).get("beta"),
            "nu_per_sqrt_day": (sv.get("dev") or {}).get("fit", {}).get("nu_per_sqrt_day"),
            "sv_half_life_h": (sv.get("dev") or {}).get("fit", {}).get("half_life_h"),
            "nu_within_per_sqrt_day": (sv.get("dev") or {}).get("fit_within", {}).get("nu_per_sqrt_day"),
            "half_life_within_h": (sv.get("dev") or {}).get("fit_within", {}).get("half_life_h"),
            "acf1_within_dev": ((sv.get("dev") or {}).get("acf_within", {}).get("1") or {}).get("acf"),
            "acf1_within_ho": ((sv.get("holdout") or {}).get("acf_within", {}).get("1") or {}).get("acf"),
            "rho_implied": (sv.get("dev") or {}).get("rho_implied"),
            "lev_t_dev_u": (sv.get("dev") or {}).get("leverage", {}).get("t"),
            "lev_t_dev_p": (sv.get("dev") or {}).get("leverage_price_space", {}).get("t"),
            "lev_t_ho_u": (sv.get("holdout") or {}).get("leverage", {}).get("t"),
            "jump_share_dev": jd.get("jump_share_dp2"), "jump_share_ho": jh.get("jump_share_dp2"),
            "zero_move_share": jd.get("zero_move_share"),
            "var_u_uniform_ho": (ud.get(("holdout", "uniform", "all")) or {}).get("var"),
            "var_u_ho": (ud.get(("holdout", ch, "all")) or {}).get("var"),
            "exkurt_u_ho": (ud.get(("holdout", ch, "all")) or {}).get("exkurt"),
            "var_u_uniform_dev": (ud.get(("dev", "uniform", "all")) or {}).get("var"),
            "exkurt_u_dev": (ud.get(("dev", ch, "all")) or {}).get("exkurt")})
    ptab = _frame(prow, d=3)
    rep.table("Fitted parameters by category (dev fit; holdout diagnostics)", ptab, save_as="params")

    rows = []
    for c, r in cats.items():
        for x in r.get("clock_table", []):
            rows.append({"category": c, **x})
    rep.table("Q2 clock: mean NLL per step of the exact M0 transition (lower is better); gain vs uniform 1/tau, t by event",
              _frame(rows), save_as="clock", max_rows=120)
    lam = []
    for c, r in cats.items():
        cf = (r.get("clock_fits") or {}).get("buckets")
        if not cf:
            continue
        e = cf["clock"]["edges"]
        for j, v in enumerate(cf["clock"]["lam"]):
            mid = math.sqrt(max(e[j], 1.0) * min(e[j + 1], 4096))
            lam.append({"category": c, "tau_from_h": e[j], "lambda_per_h": v, "uniform_1_over_tau": 1 / mid,
                        "ratio": v * mid})
    rep.table("Q2 clock: piecewise lambda(tau) per hour vs uniform 1/tau at the bucket's geometric mid",
              _frame(lam), save_as="lambda", max_rows=100)
    rep.table("Q2 variance identity: E[sum dp^2 + (y - p_last)^2] vs E[p0(1-p0)] (ratio 1 for a calibrated martingale)",
              _frame(res.get("variance_identity", [])), save_as="variance_identity", max_rows=60)

    ud = []
    for c, r in cats.items():
        for x in r.get("u_diag", []):
            ud.append({"category": c, **x})
    rep.table("Q3 u = (Z' - Z e^{L/2}) / sqrt(e^L - 1 + noise): moments and tails (N(0,1): P(|u|>2)=4.55%, >3=0.27%, >4=0.0063%)",
              _frame(ud, ["category", "split", "clock", "group", "n", "var", "exkurt", "p_gt2", "p_gt3", "p_gt4", "zero_share", "var_u_raw"]),
              save_as="u_diag", max_rows=60)
    dr = []
    for c, r in cats.items():
        for k, v in (r.get("drift") or {}).items():
            dr.append({"category": c, "split_clock": k, **v})
    rep.table("Q3 probit drift: slope of dZ on Z(e^{L/2}-1) (M0 predicts 1)", _frame(dr))
    bb = []
    for c, r in cats.items():
        for x in r.get("backbone_table", []):
            bb.append({"category": c, **x})
        for sp, v in (r.get("beta_cells") or {}).items():
            bb.append({"category": c, "backbone": "cells", "split": sp, "beta": v["raw"]["beta"],
                       "qlike": None, "minus_probit": v["roll"]["beta"], "t_vs_probit": v["roll"]["se"]})
    rep.table("Q1 backbone: QLIKE of dp^2 = c(tau)^2 g(p)^2 + k s^2 (fit on dev). 'cells' rows: beta raw | "
              "minus_probit = Roll-corrected beta | t_vs_probit = its bootstrap SE", _frame(bb), save_as="backbone", max_rows=80)
    svr = []
    for c, r in cats.items():
        for sp, v in (r.get("sv") or {}).items():
            ac = v["acf_chosen"]
            row = {"category": c, "split": sp}
            for k in ("1", "2", "3", "6", "12", "24", "48", "168"):
                row[f"acf{k}"] = (ac.get(k) or {}).get("acf")
            row["t_acf1"] = (ac.get("1") or {}).get("t")
            row["t_acf24"] = (ac.get("24") or {}).get("t")
            row["acf1_uniform_clock"] = (v["acf_uniform"].get("1") or {}).get("acf")
            row["acf24_uniform_clock"] = (v["acf_uniform"].get("24") or {}).get("acf")
            row["acf1_active_only"] = (v["acf_active"].get("1") or {}).get("acf")
            aw = v.get("acf_within") or {}
            row["acf1_within_mkt"] = (aw.get("1") or {}).get("acf")
            row["t_acf1_within"] = (aw.get("1") or {}).get("t")
            row["acf24_within_mkt"] = (aw.get("24") or {}).get("acf")
            row["acf6_within_mkt"] = (aw.get("6") or {}).get("acf")
            fw = v.get("fit_within") or {}
            row["within_half_life_h"] = fw.get("half_life_h")
            row["within_nu_per_sqrt_day"] = fw.get("nu_per_sqrt_day")
            row.update({f"fit_{k}": v["fit"].get(k) for k in ("phi", "half_life_h", "nu_per_sqrt_h", "nu_per_sqrt_day")})
            row.update({"lev_corr_u": v["leverage"].get("corr"), "lev_t_u": v["leverage"].get("t"),
                        "lev_corr_p": v["leverage_price_space"].get("corr"), "lev_t_p": v["leverage_price_space"].get("t"),
                        "rho_implied": v.get("rho_implied")})
            svr.append(row)
    rep.table("Q4 stochastic vol: corr(|u_t|, |u_t+k|) within markets; SV fit A phi^k; leverage (rho>0: vol rises "
              "after moves away from 1/2; u = probit units, p = price units standardised by the beta backbone)",
              _frame(svr), save_as="sv")
    jr = []
    for c, r in cats.items():
        for sp, v in (r.get("jumps") or {}).items():
            jr.append({"category": c, "split": sp, **v})
    rep.table("Q5 jumps: EM scale mixture on moved hours; jump_share_dp2 = posterior-weighted share of sum dp^2; "
              "big_after_big = P(|u_t+1|>3 | |u_t|>3)", _frame(jr), save_as="jumps")
    fc = res.get("forecast", {})
    fcols = ["category", "H", "split", "n", "events", "mean_rv_c2", "qlike_HIST", "qlike_M0", "qlike_BB", "qlike_M1",
             "qlike_M1BB", "dQ_M1-HIST", "tQ_M1-HIST", "dQ_M1-M0", "tQ_M1-M0", "dQ_M0-HIST", "tQ_M0-HIST",
             "dQ_M1BB-M1", "tQ_M1BB-M1", "mse_rel_M0", "mse_rel_M1", "mse_rel_M1BB", "tMSE_M1-HIST", "tMSE_M1-M0"]
    rep.table("Q6 forecasting next-H-hour realised variance (sum of hourly dp^2, cents^2); QLIKE (lower better), "
              "loss differences with event-clustered t; MSE relative to HIST", _frame(fc.get("tables", []), fcols),
              save_as="forecast", max_rows=60)
    rep.json("Q6 dev-chosen forecast settings and calibration coefficients [spread^2 H, base, floor]", fc.get("choices", {}))
    eco = mk.get("economic") or []
    rep.table("Q7 one-hour expected |dp| (model: M2 mixture incl. zero-move mass; m0_gauss: Gaussian M0) vs quoted half-spread",
              _frame(eco, d=3), save_as="economic", max_rows=80)
    rep.table("Q3 price martingale: E[mid_t+6h - mid_t | mid_t] and E[y - mid_t] in cents (both 0 under M0/M1)",
              _frame(mk.get("martingale") or [], d=3), save_as="martingale", max_rows=120)
    if mk:
        rep.json("Q7 maker: thresholds chosen on dev", {"k_model": mk.get("k_chosen_on_dev"),
                                                         "half_spread_no_model": mk.get("hs_threshold_chosen_on_dev"),
                                                         "universe_hours": mk.get("universe_hours")})
        rep.table("Q7 touch quotes (YES bid at bid, NO bid at 1-ask, 1 h life), trade-through fills, held to settlement, "
                  "maker fees; cents per contract, t by event", _frame(mk.get("table", []), d=3), save_as="maker")
        rep.table("Q7 maker by category", _frame(mk.get("by_category", []), d=3), save_as="maker_by_cat", max_rows=60)
    path = rep.save()
    summary = {"report": str(path), "params_table": ptab.to_dicts(), "k_chosen_on_dev": mk.get("k_chosen_on_dev"),
               "maker": mk.get("table"), "forecast": [r for r in fc.get("tables", []) if r["split"] == "holdout"],
               "variance_identity": [r for r in res.get("variance_identity", []) if r["sample_h"] == 6]}
    (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1))
    print(path)
    return path


# =============================================================================================================
# High-frequency follow-up: recorded L2 for two in-play nights (descriptive / modelling only, no trading tests)
# =============================================================================================================
HF_NIGHTS = (("2026-11-07T17", "2026-11-08T06", "sat"), ("2026-11-08T17", "2026-11-09T06", "sun"))
HF_DIR = OUT / "hf"
HF_H = (1, 5, 15, 60, 300)                     # seconds, M0 test horizons
SIG_H = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600)
HF_MIN_TAU_S = 600.0                           # step must end >= 10 min before the scheduled end
HF_START_SPREAD = (0.02, 0.03)
_ESC_TICKER = __import__("re").compile(r'market_ticker\\":\\"([^"\\]+)')


def hf_extract(s, start: str, end: str, tag: str) -> dict[str, Any]:
    """Replay the recorded Kalshi books of game-winner series and keep every top-of-book change
    (recv_ns, ticker, yes bid, yes ask) and every trade. Books persist across hours; a snapshot
    resets a market's book. Only lines whose (escaped) ticker belongs to a game series are parsed."""
    import io

    import zstandard

    from albert.backtest.l2 import stream_files
    from albert.research.inplay_hft import _by_hour
    from albert.research.volatile import GAME_SERIES

    HF_DIR.mkdir(parents=True, exist_ok=True)
    files = stream_files(s.raw_dir / "stream", start, end)
    books: dict[str, dict[str, dict[float, float]]] = {}
    last: dict[str, tuple[float | None, float | None]] = {}
    q_t, q_m, q_b, q_a = [], [], [], []
    t_t, t_m, t_p, t_c, t_s = [], [], [], [], []
    stats = {"files": len(files), "lines": 0, "parsed": 0, "pre_snapshot": 0}

    def best(m: str) -> tuple[float | None, float | None]:
        b = books[m]
        yb = max(b["yes"]) if b["yes"] else None
        nb = max(b["no"]) if b["no"] else None
        return yb, (round(1 - nb, 4) if nb is not None else None)

    for hour_files in _by_hour(files):
        for path in hour_files:
            with open(path, "rb") as f:
                for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8"):
                    stats["lines"] += 1
                    mm = _ESC_TICKER.search(line)
                    if mm is None or mm.group(1).split("-", 1)[0] not in GAME_SERIES:
                        continue
                    row = json.loads(line)
                    d = json.loads(row["frame"])
                    typ = d.get("type")
                    msg = d.get("msg") or {}
                    m = msg.get("market_ticker")
                    if m is None:
                        continue
                    stats["parsed"] += 1
                    ns = row["recv_ns"]
                    if typ == "trade":
                        t_t.append(ns)
                        t_m.append(m)
                        t_p.append(float(msg["yes_price_dollars"]))
                        t_c.append(float(msg.get("count_fp") or 0))
                        t_s.append(msg.get("taker_side"))
                        continue
                    if typ == "orderbook_snapshot":
                        books[m] = {side: {float(px): float(q) for px, q in (msg.get(f"{side}_dollars_fp") or []) if float(q) > 0}
                                    for side in ("yes", "no")}
                    elif typ == "orderbook_delta":
                        if m not in books:
                            stats["pre_snapshot"] += 1
                            continue
                        lv = books[m][msg["side"]]
                        px = float(msg["price_dollars"])
                        q = lv.get(px, 0.0) + float(msg["delta_fp"])
                        if q > 1e-9:
                            lv[px] = q
                        else:
                            lv.pop(px, None)
                    else:
                        continue
                    ba = best(m)
                    if last.get(m) != ba:
                        last[m] = ba
                        q_t.append(ns)
                        q_m.append(m)
                        q_b.append(np.nan if ba[0] is None else ba[0])
                        q_a.append(np.nan if ba[1] is None else ba[1])
        print(f"  {tag} {hour_files[0].parent.name}T{hour_files[0].name[:2]}: {len(q_t):,} quote changes, "
              f"{len(t_t):,} trades, {stats['parsed']:,} parsed / {stats['lines']:,} lines", flush=True)
    pl.DataFrame({"ns": np.array(q_t, dtype=np.int64), "ticker": q_m, "bid": np.array(q_b, dtype=np.float32),
                  "ask": np.array(q_a, dtype=np.float32)}).write_parquet(HF_DIR / f"quotes_{tag}.parquet")
    pl.DataFrame({"ns": np.array(t_t, dtype=np.int64), "ticker": t_m, "yes_price": np.array(t_p, dtype=np.float32),
                  "count": np.array(t_c, dtype=np.float32), "taker_side": t_s}).write_parquet(HF_DIR / f"trades_{tag}.parquet")
    stats.update({"quote_changes": len(q_t), "trades": len(t_t), "markets": len(set(q_m))})
    (HF_DIR / f"extract_{tag}.json").write_text(json.dumps(stats))
    return stats


def hf_games(s) -> pl.DataFrame:
    """One leg per game (first alphabetical non-TIE leg) with its scheduled in-play window
    (``volatile._in_play_bounds``) and scheduled end T = expected_expiration_time."""
    from albert.research.volatile import _in_play_bounds, market_meta

    tick = set()
    for _, _, tag in HF_NIGHTS:
        tick |= set(pl.read_parquet(HF_DIR / f"quotes_{tag}.parquet", columns=["ticker"])["ticker"].unique().to_list())
    cache = HF_DIR / "markets_meta.json"
    shared = s.data_dir / "cache" / "volatile_markets.json"
    if not cache.exists() and shared.exists():           # seed from the shared cache, never write it
        cache.write_text(shared.read_text())
    meta = market_meta(sorted(tick), cache)
    bounds = _in_play_bounds(meta)
    rows = []
    for t, (lo, hi) in bounds.items():
        exp = datetime.fromisoformat(meta[t]["expected_expiration_time"])
        rows.append({"ticker": t, "event": t.rsplit("-", 1)[0], "series": t.split("-", 1)[0], "lo": lo, "hi": hi,
                     "T": int(exp.timestamp() * 1e9), "result": meta[t].get("result")})
    g = pl.DataFrame(rows).filter(~pl.col("ticker").str.ends_with("-TIE")).sort("ticker")
    return g.unique("event", keep="first", maintain_order=True)


def hf_paths(s) -> list[dict[str, Any]]:
    """Per game leg and night: quote change times and bid/ask inside the in-play window, cut at the
    first decided quote (mid <= 2c or >= 98c with spread <= 10c, observable at that time) and at
    T - 10 min."""
    games = hf_games(s)
    out = []
    for start, end, tag in HF_NIGHTS:
        q = pl.read_parquet(HF_DIR / f"quotes_{tag}.parquet")
        w0 = int(datetime.fromisoformat(start + ":00:00+00:00").timestamp() * 1e9)
        w1 = int(datetime.fromisoformat(end + ":00:00+00:00").timestamp() * 1e9)
        q = q.join(games, on="ticker", how="inner").filter((pl.col("ns") >= pl.max_horizontal("lo", pl.lit(w0)))
                                                           & (pl.col("ns") <= pl.min_horizontal("hi", pl.lit(w1))))
        for (t,), part in q.sort("ns").partition_by(["ticker"], as_dict=True, maintain_order=True).items():
            ns = part["ns"].to_numpy()
            bid = part["bid"].to_numpy().astype(float)
            ask = part["ask"].to_numpy().astype(float)
            mid, spr = (bid + ask) / 2, ask - bid
            dec = np.flatnonzero(np.isfinite(mid) & (spr <= 0.10) & ((mid <= 0.02) | (mid >= 0.98)))
            T = int(part["T"][0])
            stop = min(int(ns[dec[0]]) if len(dec) else int(part["hi"][0]), T - int(HF_MIN_TAU_S * 1e9))
            lo = max(int(part["lo"][0]), w0)
            keep = ns <= stop
            if keep.sum() < 10 or stop - lo < 1800 * 1e9:
                continue
            out.append({"ticker": t, "event": part["event"][0], "series": part["series"][0], "night": tag,
                        "ns": ns[keep], "bid": bid[keep], "ask": ask[keep], "lo": lo, "stop": stop, "T": T})
    return out


def hf_sample(path: dict[str, Any], h: int) -> dict[str, np.ndarray]:
    """Quote as of each point of an h-second grid over [lo, stop] (last change at or before)."""
    grid = np.arange(path["lo"], path["stop"] + 1, int(h * 1e9), dtype=np.int64)
    j = np.searchsorted(path["ns"], grid, side="right") - 1
    ok = j >= 0
    jj = np.maximum(j, 0)
    bid = np.where(ok, path["bid"][jj], np.nan)
    ask = np.where(ok, path["ask"][jj], np.nan)
    return {"t": grid, "bid": bid, "ask": ask, "mid": (bid + ask) / 2, "spr": ask - bid,
            "tau": (path["T"] - grid) / 1e9}


def hf_steps(paths: list[dict[str, Any]], h: int, start_spread: float = 0.03) -> dict[str, np.ndarray]:
    """Non-overlapping h-second steps with a valid start quote (spread <= start_spread, mid in
    [3c, 97c]) and a valid end quote (spread <= 10c); uniform clock L = ln(tau / (tau - h))."""
    cols: dict[str, list] = {k: [] for k in ("p0", "p1", "s0", "s1", "tau", "game", "series", "k", "night")}
    for gi, pth in enumerate(paths):
        g = hf_sample(pth, h)
        p0, p1, s0, s1 = g["mid"][:-1], g["mid"][1:], g["spr"][:-1], g["spr"][1:]
        ok = (np.isfinite(p0) & np.isfinite(p1) & (s0 <= start_spread + 1e-9) & (s0 > 0) & (p0 >= P_LO) & (p0 <= P_HI)
              & (s1 > 0) & (s1 <= END_SPREAD + 1e-9) & (g["tau"][:-1] - h >= HF_MIN_TAU_S))
        idx = np.flatnonzero(ok)
        cols["p0"].append(p0[idx]); cols["p1"].append(p1[idx]); cols["s0"].append(s0[idx]); cols["s1"].append(s1[idx])
        cols["tau"].append(g["tau"][:-1][idx]); cols["game"].append(np.full(len(idx), gi)); cols["k"].append(idx)
        cols["series"].append(np.full(len(idx), pth["series"], dtype=object)); cols["night"].append(np.full(len(idx), pth["night"], dtype=object))
    out = {k: np.concatenate(v) if v else np.array([]) for k, v in cols.items()}
    out["L"] = np.log(out["tau"] / (out["tau"] - h))
    out["z0"], out["z1"] = probit(out["p0"]), probit(out["p1"])
    out["u"] = u_stat(out["z0"], out["z1"], out["L"])
    return out


def hf_m0_table(paths: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict]:
    rows, cache = [], {}
    for sp in HF_START_SPREAD:
        for h in HF_H:
            st = hf_steps(paths, h, sp)
            cache[(sp, h)] = st
            u = st["u"]
            if len(u) < 50:
                continue
            x = clustered(u ** 2 - 1, st["game"])
            r = {"start_spread_max_c": sp * 100, "h_s": h, "games": len(np.unique(st["game"])), **_u_stats(u),
                 "mean_u2_minus_1": x["mean"], "t_u2": x["t"], "median_abs_dp_c": float(np.median(np.abs(st["p1"] - st["p0"])) * 100)}
            rows.append(r)
            if sp == 0.03:
                for ser in np.unique(st["series"]):
                    m = st["series"] == ser
                    if m.sum() >= 200:
                        rows.append({"start_spread_max_c": sp * 100, "h_s": h, "series": ser,
                                     "games": len(np.unique(st["game"][m])), **_u_stats(u[m])})
    return rows, cache


def hf_signature(paths: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Realised variance of the mid (spread <= 10c at both ends) per hour of in-play time against the
    sampling interval, on the same windows; and mean u^2 under the uniform clock. Fit
    RV(h)/T = sigma^2 + 2 eta^2 / h (i.i.d. noise of std eta in the mid)."""
    rows = []
    for h in SIG_H:
        ss = tt = su = nu = 0.0
        n = 0
        for pth in paths:
            g = hf_sample(pth, h)
            d = np.diff(g["mid"])
            ok = np.isfinite(d) & (g["spr"][:-1] <= END_SPREAD) & (g["spr"][1:] <= END_SPREAD) & (g["tau"][:-1] - h >= HF_MIN_TAU_S)
            ss += float((d[ok] ** 2).sum())
            tt += h * float(ok.sum())
            n += int(ok.sum())
            L = np.log(g["tau"][:-1] / (g["tau"][:-1] - h))
            okz = ok & (g["mid"][:-1] >= P_LO) & (g["mid"][:-1] <= P_HI)
            u = u_stat(probit(g["mid"][:-1][okz]), probit(g["mid"][1:][okz]), L[okz])
            su += float((u ** 2).sum())
            nu += len(u)
        rows.append({"h_s": h, "steps": n, "rv_c2_per_hour": ss / tt * 3600 * 1e4, "mean_u2": su / max(nu, 1)})
    hs = np.array([r["h_s"] for r in rows], float)
    rv = np.array([r["rv_c2_per_hour"] for r in rows]) / 3600   # c^2 per second
    X = np.c_[np.ones_like(hs), 2 / hs]
    coef = np.linalg.lstsq(X, rv, rcond=None)[0]
    fit = {"sigma2_c2_per_hour": float(coef[0] * 3600), "eta_c": float(math.sqrt(max(coef[1], 0))),
           "noise_share_of_rv_at_1s": float(2 * coef[1] / rv[0]) if rv[0] > 0 else None,
           "noise_share_of_rv_at_60s": float(2 * coef[1] / 60 / rv[SIG_H.index(60)])}
    return rows, fit


# ---- probit-drift shortfall: does the 1c grid explain it? -----------------------------------------------------
ZB = (0.0, 0.5, 1.0, 1.2816, 1.89)


def round_quote(p: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quote a true price p on the 1c grid with an integer-cent spread s, centred as closely as the
    grid allows: bid = round(100 p - s/2) cents, ask = bid + s, kept inside [1c, 99c]."""
    sc = np.maximum(np.round(np.asarray(s) * 100), 1)
    b = np.round(np.asarray(p) * 100 - sc / 2)
    b = np.clip(b, 1, 99 - sc)
    return b / 100, (b + sc) / 100


def drift_by_z(z0: np.ndarray, z1: np.ndarray, L: np.ndarray, g: np.ndarray, zlag: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Probit drift slope by |Z0| bin, with the direct price check: measured E[sgn(Z0) dp] (cents,
    t by cluster) against the value the slope shortfall implies if the price itself drifted,
    -(1 - slope) E[|Z| phi(Z) (e^{L/2} - 1)]. If the measured price drift is ~0 while the implied one
    is not, the shortfall is about the shape of the Z-variance (or measurement), not about p."""
    out = []
    az = np.abs(z0)
    dp = (special.ndtr(z1) - special.ndtr(z0)) * 100
    scale = az * npdf(z0) * np.expm1(np.minimum(L, 2.0) / 2) * 100

    def row(m, lo, hi, label):
        d = drift_test(z0[m], z1[m], L[m], g[m], None if zlag is None else zlag[m])
        keep = m & (L <= 2.0)
        x = clustered(np.sign(z0[keep]) * dp[keep], g[keep])
        imp = None if d.get("slope") is None else -(1 - d["slope"]) * float(scale[keep].mean())
        return {"absZ_from": lo, "absZ_to": hi, "p_range": label,
                **{kk: d.get(kk) for kk in ("slope", "se", "t_vs_1", "n", "slope_iv", "t_iv_vs_1")},
                "e_sgn_dp_c": x["mean"], "t_sgn_dp": x["t"], "implied_sgn_dp_c": imp}

    for k in range(len(ZB) - 1):
        m = (az >= ZB[k]) & (az < ZB[k + 1])
        if m.sum() < 500:
            continue
        out.append(row(m, ZB[k], ZB[k + 1], f"{special.ndtr(-ZB[k + 1]):.2f}-{special.ndtr(-ZB[k]):.2f}"))
    out.append(row(np.ones(len(z0), dtype=bool), 0.0, 1.89, "all"))
    return out


def m0_paths(z0: np.ndarray, L: np.ndarray, eps: np.ndarray, gs: np.ndarray) -> np.ndarray:
    """Exact M0 paths, vectorised over many markets laid end to end (``gs`` = each row's market
    start). Z at a market's first row is z0; Z_{t+1} = Z_t e^{L_t/2} + sqrt(e^{L_t} - 1) eps_t, i.e.
    Z_t = A_t (z0 + sum_{k<t} sqrt(e^{L_k} - 1) eps_k / A_{k+1}) with A_t = exp(sum_{k<t} L_k / 2)."""
    cs = np.cumsum(L)
    S = np.minimum(cs - L - (cs - L)[gs], 60.0)
    A = np.exp(S / 2)
    A_next = np.exp(np.minimum(S + L, 60.0) / 2)
    incr = np.sqrt(np.expm1(L)) * eps / A_next
    ci = np.cumsum(incr)
    before = ci - incr - (ci - incr)[gs]
    return A * (z0 + before)


def simulate_panel_quotes(a: dict[str, np.ndarray], c: int, cp: dict[str, Any], variant: str, seed: int = 1) -> dict[str, np.ndarray]:
    """M0 paths on the category's real hourly grids (each market starts at its first valid mid; the
    dev-fitted clock gives L per hour), then quoted like the data:
    * ``true``: the true price itself (spread 1e-6, no grid),
    * ``tick``: re-quoted every hour on the 1c grid with that hour's real spread,
    * ``stale``: re-quoted only in hours where the real market had a candle (activity); otherwise
      the previous quote stands.
    Returns a copy of the panel arrays restricted to the category's markets, with simulated quotes."""
    rng = np.random.default_rng(seed)
    sel = np.flatnonzero(np.isin(a["tk"], np.unique(a["tk"][(a["cat"] == c) & a["sched_ok"]])))
    b = {k: v[sel] for k, v in a.items()}
    n = len(sel)
    tau = b["tau"].astype(float)
    L = np.where(tau - 1 >= MIN_TAU_AFTER, clock_L(cp, np.maximum(tau, 1.0 + 1e-6), 1, b["hour"]), 30.0)
    L = np.clip(np.nan_to_num(L, nan=30.0), 0, 30)
    gs, _ = group_bounds(b["tk"])
    f = np.flatnonzero(b["start_ok"] | b["end_ok"])
    first = np.full(b["tk"].max() + 1, -1)
    utk, ui = np.unique(b["tk"][f], return_index=True)
    first[utk] = f[ui]                                 # first two-sided quote of each market
    z0 = probit(np.where(first[b["tk"]] >= 0, b["mid"][np.maximum(first[b["tk"]], 0)], 0.5))
    Z = m0_paths(z0, L, rng.standard_normal(n), gs)
    p = special.ndtr(Z)
    if variant == "true":
        bid, ask = p - 5e-7, p + 5e-7
    else:
        sb, sa = round_quote(p, b["spr"])           # quotes centre on the true price
        if variant == "stale":
            upd = b["candle"] | (np.arange(n) == gs)
            last = np.maximum.accumulate(np.where(upd, np.arange(n), 0))
            sb, sa = sb[last], sa[last]
        bid, ask = sb, sa
    b["bid"], b["ask"] = bid.astype(np.float32), ask.astype(np.float32)
    b["mid"] = (bid + ask) / 2
    b["spr"] = ask - bid
    q = b["spr"]
    b["end_ok"] = (q > 0) & (q <= END_SPREAD + 1e-9) & (bid >= 0) & (ask <= 1)
    lim = 0.0 if variant == "true" else 0.01
    b["start_ok"] = (b["end_ok"] & (q <= START_SPREAD + 1e-9) & (bid >= lim) & (ask <= 1 - lim)
                     & (b["mid"] >= P_LO) & (b["mid"] <= P_HI))
    return b


def drift_shortfall_panel(a: dict[str, np.ndarray], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Data vs simulated (true / tick / stale) probit drift slopes by |Z| on the hourly panel, dev rows,
    the category's dev-fitted clock."""
    out = []
    for c, name in enumerate(CATS):
        cp = params.get(name)
        if not cp or name == "Sports in-play":
            continue
        variants = {"data": None, "sim_true": "true", "sim_tick": "tick", "sim_stale": "stale"}
        for vname, var in variants.items():
            b = a if var is None else simulate_panel_quotes(a, c, cp, var)
            inc = increments(b, c, 1)
            dev = np.flatnonzero(inc["split"] == 0)
            if len(dev) < 1000:
                continue
            z0, z1 = probit(inc["p0"][dev]), probit(inc["p1"][dev])
            L = clock_L(cp, inc["tau"][dev], 1, inc["hour"][dev])
            for r in drift_by_z(z0, z1, np.clip(L, 1e-10, 30), inc["ev"][dev]):
                out.append({"category": name, "variant": vname, **r})
            if var is not None:
                del b
    return out


def hf_drift(cache: dict, seed: int = 4) -> list[dict[str, Any]]:
    """In-play probit drift slope by |Z| at 60 s and 300 s: data, and the same rows simulated under M0
    from the observed start mid, re-quoted on the 1c grid with the observed spreads."""
    rng = np.random.default_rng(seed)
    out = []
    for h in (60, 300):
        st = cache.get((0.03, h))
        if st is None or len(st["u"]) < 500:
            continue
        for r in drift_by_z(st["z0"], st["z1"], st["L"], st["game"]):
            out.append({"h_s": h, "variant": "data", **r})
        ptrue0 = np.clip(st["p0"] + rng.uniform(-0.005, 0.005, len(st["p0"])), 0.001, 0.999)
        zt0 = probit(ptrue0)
        mu, var = m0_moments(zt0, st["L"])
        ptrue1 = special.ndtr(mu + np.sqrt(var) * rng.standard_normal(len(mu)))
        b0, a0 = round_quote(ptrue0, st["s0"])
        b1, a1 = round_quote(ptrue1, st["s1"])
        m0q, m1q = (b0 + a0) / 2, (b1 + a1) / 2
        ok = (m0q >= P_LO) & (m0q <= P_HI)
        for vname, z0, z1 in (("sim_true", zt0, probit(ptrue1)), ("sim_tick", probit(m0q), probit(m1q))):
            for r in drift_by_z(z0[ok], z1[ok], st["L"][ok], st["game"][ok]):
                out.append({"h_s": h, "variant": vname, **r})
    return out


# ---- jumps and leverage at high frequency -----------------------------------------------------------------
def hf_jumps(paths: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bipower vs realised variance of the mid at 5/10/15 s (per game, then pooled) and threshold jumps:
    |u| > 4 local bipower sd (preceding 30 min, floor 0.25) and |dp| >= 2c. Jump sizes are compared with
    Stern's fixed-score prediction |dp| ~ phi(Z) / sqrt(tau) (log-log slope 1, by sport)."""
    rows, jumps = [], []
    mu1 = math.sqrt(2 / math.pi)
    for h in (5, 10, 15):
        RV = BV = 0.0
        per_game = []
        jump_var = 0.0
        for gi, pth in enumerate(paths):
            g = hf_sample(pth, h)
            mid, spr, tau = g["mid"], g["spr"], g["tau"]
            d = np.diff(mid)
            ok = np.isfinite(d) & (spr[:-1] <= 0.03 + 1e-9) & (spr[1:] <= END_SPREAD) & (tau[:-1] - h >= HF_MIN_TAU_S) \
                & (mid[:-1] >= P_LO) & (mid[:-1] <= P_HI)
            dd = np.where(ok, d, np.nan)
            rv = float(np.nansum(dd ** 2))
            pair = dd[1:] * 0 + np.abs(dd[1:]) * np.abs(dd[:-1])
            bv = float(np.nansum(pair)) / mu1 ** 2
            if rv <= 0:
                continue
            RV += rv
            BV += min(bv, rv)
            per_game.append(max(rv - bv, 0) / rv)
            # threshold jumps in clock-standardised units
            L = np.log(tau[:-1] / np.maximum(tau[:-1] - h, 1e-9))
            u = np.where(ok, u_stat(probit(mid[:-1]), probit(mid[1:]), L), np.nan)
            win = max(int(1800 / h), 10)
            au = np.abs(np.nan_to_num(u))
            prod = np.r_[0.0, au[1:] * au[:-1]]
            cnt = np.r_[0.0, (np.isfinite(u[1:]) & np.isfinite(u[:-1])).astype(float)]
            cs, cc = np.cumsum(prod), np.cumsum(cnt)
            i = np.arange(len(u))
            lo = np.maximum(i - win, 0)
            sig = np.sqrt(np.maximum((cs[np.maximum(i - 1, 0)] - cs[lo]) / np.maximum(cc[np.maximum(i - 1, 0)] - cc[lo], 1), 0) / mu1 ** 2)
            sig = np.maximum(sig, 0.25)
            flag = ok & (np.abs(np.nan_to_num(u)) > 4 * sig) & (np.abs(np.nan_to_num(d)) >= 0.02 - 1e-9)
            for k in np.flatnonzero(flag):
                jump_var += d[k] ** 2
                if h == 10:
                    z = float(probit(mid[k]))
                    jumps.append({"game": gi, "series": pth["series"], "night": pth["night"], "p0": float(mid[k]),
                                  "dp": float(d[k]), "dz": float(probit(mid[k + 1]) - z), "tau_s": float(tau[k]),
                                  "stern_scale": float(npdf(z) / math.sqrt(tau[k] / 3600)), "u_over_sig": float(u[k] / sig[k])})
        pg = np.array(per_game)
        rows.append({"h_s": h, "games": len(per_game), "rv_c2": RV * 1e4, "bv_c2": BV * 1e4,
                     "jump_share_bipower": (RV - BV) / RV if RV > 0 else None,
                     "jump_share_bipower_median_game": float(np.median(pg)) if len(pg) else None,
                     "jump_share_threshold": jump_var / RV if RV > 0 else None})
    stern = []
    if jumps:
        jf = pl.DataFrame(jumps)
        for ser, part in [("all", jf)] + [(k[0], v) for k, v in jf.partition_by(["series"], as_dict=True).items()]:
            if part.height < 15:
                continue
            y = np.log(np.abs(part["dp"].to_numpy()))
            x = np.log(part["stern_scale"].to_numpy())
            yz = np.log(np.abs(part["dz"].to_numpy()))
            xz = np.log(1 / np.sqrt(part["tau_s"].to_numpy() / 3600))
            g = part["game"].to_numpy()
            if ser == "all":    # sport fixed effects
                sers = part["series"].to_numpy()
                for v in (y, x, yz, xz):
                    for u_ in np.unique(sers):
                        mm = sers == u_
                        v[mm] -= v[mm].mean()
            b1, se1 = _ols_slope(x, y, g)
            b2, se2 = _ols_slope(xz, yz, g)
            stern.append({"series": ser, "jumps": part.height, "games": len(np.unique(g)),
                          "median_abs_dp_c": float(np.median(np.abs(part["dp"].to_numpy())) * 100),
                          "slope_log_dp_on_log_phi_over_sqrt_tau": b1, "se": se1,
                          "slope_log_dz_on_log_inv_sqrt_tau": b2, "se_z": se2})
    return rows + [{"stern": stern}], jumps


def _ols_slope(x: np.ndarray, y: np.ndarray, g: np.ndarray) -> tuple[float | None, float | None]:
    x = x - x.mean()
    y = y - y.mean()
    sxx = float((x * x).sum())
    if sxx <= 0 or len(x) < 5:
        return None, None
    b = float((x * y).sum() / sxx)
    e = y - b * x
    _, inv = np.unique(g, return_inverse=True)
    k = inv.max() + 1
    sc = np.bincount(inv, weights=x * e)
    se = math.sqrt((sc ** 2).sum() * k / max(k - 1, 1)) / sxx if k > 1 else None
    return b, se


def _future_mean_abs(x: np.ndarray, K: int) -> np.ndarray:
    """Mean of |x| over the next K steps (NaNs skipped; NaN if none)."""
    n = len(x)
    s_ = np.zeros(n)
    c_ = np.zeros(n)
    ax = np.abs(x)
    for k in range(1, K + 1):
        if k >= n:
            break
        v = ax[k:]
        f = np.isfinite(v)
        s_[: n - k] += np.where(f, v, 0.0)
        c_[: n - k] += f
    return np.where(c_ > 0, s_ / np.maximum(c_, 1), np.nan)


def hf_leverage(paths: list[dict[str, Any]], horizons: tuple[int, ...] = (60, 120, 300), K: int = 5) -> list[dict[str, Any]]:
    """corr(sgn(Z_t) x_t, mean of the next K |x|) at 1-5 min, x = u (uniform clock, probit units) or
    dp (cents), p in [0.1, 0.9]. Negative: vol rises after moves toward 1/2 (the brief's rho < 0). In
    price units M0 itself predicts a negative value (phi(Z) grows toward 1/2); in u units it predicts 0."""
    out = []
    for h in horizons:
        acc = {"u": ([], [], []), "dp": ([], [], [])}
        for gi, pth in enumerate(paths):
            g = hf_sample(pth, h)
            mid, spr, tau = g["mid"], g["spr"], g["tau"]
            if len(mid) < 3:
                continue
            ok = np.isfinite(mid[:-1]) & np.isfinite(mid[1:]) & (spr[:-1] <= 0.03 + 1e-9) & (spr[1:] <= END_SPREAD) \
                & (mid[:-1] >= 0.1) & (mid[:-1] <= 0.9) & (tau[:-1] - h >= HF_MIN_TAU_S)
            L = np.log(tau[:-1] / np.maximum(tau[:-1] - h, 1e-9))
            z = probit(mid[:-1])
            with np.errstate(invalid="ignore"):
                u = np.where(ok, np.clip(u_stat(z, probit(mid[1:]), L), -8, 8), np.nan)
            dp = np.where(ok, np.diff(mid) * 100, np.nan)
            for key, x in (("u", u), ("dp", dp)):
                fut = _future_mean_abs(x, K)
                keep = np.isfinite(x) & np.isfinite(fut)
                acc[key][0].append(np.sign(z[keep]) * x[keep])
                acc[key][1].append(fut[keep])
                acc[key][2].append(np.full(int(keep.sum()), gi))
        for key, label in (("u", "u (probit, uniform clock)"), ("dp", "dp (cents)")):
            if not acc[key][0]:
                continue
            x, m, g = (np.concatenate(v) for v in acc[key])
            if len(x) < 20:
                continue
            prod = (x - x.mean()) * (m - m.mean()) / (x.std() * m.std())
            r = clustered(prod, g)
            out.append({"h_s": h, "unit": label, "next_K": K, "n": r["n"], "games": r["clusters"], "corr": r["mean"], "t": r["t"]})
    return out


def hf_sim_paths(paths: list[dict[str, Any]], seed: int = 12) -> list[dict[str, Any]]:
    """Matched control: for each real game leg, an M0 path with the uniform clock to its scheduled
    end, simulated per second from the leg's first two-sided mid, quoted on the 1c grid with the leg's
    own observed spread path (as of each second; 1c where none), cut at the same window and at the
    first decided quote. Any HF statistic of these paths is what ticks + spreads + M0 alone produce."""
    rng = np.random.default_rng(seed)
    out = []
    for pth in paths:
        g = hf_sample(pth, 1)
        spr = np.where(np.isfinite(g["spr"]) & (g["spr"] > 0), g["spr"], 0.01)
        ok0 = np.flatnonzero(np.isfinite(g["mid"]) & (g["spr"] <= END_SPREAD))
        if not len(ok0):
            continue
        n = len(g["t"])
        tau = g["tau"]
        L = np.log(tau / np.maximum(tau - 1, 1e-3))
        Z = m0_paths(np.full(n, float(probit(g["mid"][ok0[0]]))), L, rng.standard_normal(n), np.zeros(n, dtype=int))
        b, a = round_quote(special.ndtr(Z), spr)
        mid = (b + a) / 2
        dec = np.flatnonzero((mid <= 0.02) | (mid >= 0.98))
        stop = int(g["t"][dec[0]]) if len(dec) else pth["stop"]
        ch = np.r_[True, (np.diff(b) != 0) | (np.diff(a) != 0)]
        keep = ch & (g["t"] <= stop)
        if keep.sum() < 10:
            continue
        out.append({**{k: pth[k] for k in ("ticker", "event", "series", "night", "lo", "T")}, "stop": stop,
                    "ns": g["t"][keep], "bid": b[keep], "ask": a[keep]})
    return out


def run_hf_extract(s) -> None:
    for start, end, tag in HF_NIGHTS:
        if (HF_DIR / f"quotes_{tag}.parquet").exists():
            print(f"{tag}: already extracted", flush=True)
            continue
        print(f"== extract {tag} {start} -> {end}", flush=True)
        print(hf_extract(s, start, end, tag), flush=True)


def run_hf(s) -> None:
    """All HF analyses -> data/research/pm_vol/hf_results.json."""
    paths = hf_paths(s)
    print(f"{len(paths)} in-play game legs", flush=True)
    res: dict[str, Any] = {"games": [{"ticker": p["ticker"], "series": p["series"], "night": p["night"],
                                      "minutes": (p["stop"] - p["lo"]) / 6e10, "quote_changes": len(p["ns"])} for p in paths]}
    sims = hf_sim_paths(paths)
    print(f"{len(sims)} simulated control legs", flush=True)
    print("== M0 by horizon", flush=True)
    res["m0"], cache = hf_m0_table(paths)
    res["m0_sim"], _ = hf_m0_table(sims)
    print("== signature", flush=True)
    res["signature"], res["signature_fit"] = hf_signature(paths)
    res["signature_sim"], res["signature_fit_sim"] = hf_signature(sims)
    print("== drift", flush=True)
    res["drift_hf"] = hf_drift(cache)
    del cache
    print("== jumps", flush=True)
    jrows, jumps = hf_jumps(paths)
    res["jumps"] = jrows[:-1]
    res["stern"] = jrows[-1]["stern"]
    if jumps:
        pl.DataFrame(jumps).write_parquet(HF_DIR / "jumps.parquet")
    jsim, jumps_sim = hf_jumps(sims)
    res["jumps_sim"] = jsim[:-1]
    res["n_jumps_sim_10s"] = len(jumps_sim)
    print("== leverage", flush=True)
    res["leverage"] = hf_leverage(paths)
    res["leverage_sim"] = hf_leverage(sims)
    (OUT / "hf_results.json").write_text(json.dumps(_jsonable(res)))
    print("done", flush=True)


def drift_tail_rows(z0: np.ndarray, z1: np.ndarray, L: np.ndarray, g: np.ndarray, label: dict[str, Any]) -> list[dict[str, Any]]:
    """Drift slope on all steps, after trimming the tails with two windows that bias in opposite
    directions (|u| <= c around M0's own mean -> towards 1; |Z1 - Z0| <= c sd around no drift ->
    towards 0), and without steps whose end mid sits at the 0.5c / 99.5c probit clip. Also the local
    second-order martingale ratio E[sgn(Z) dZ] / E[|Z| dZ^2 / 2] on |Z| >= 0.5 (1 if p is a
    martingale, whatever the clock), and E[dZ^2] / E[L] (how much of the clock the sample carries)."""
    mu, var = m0_moments(z0, L)
    sd = np.sqrt(var)
    u = (z1 - mu) / sd
    w0 = np.abs(z1 - z0) / sd
    clip = np.abs(z1) >= abs(float(probit(np.array([P_CLIP]))[0])) - 1e-9
    out = []
    for name, m in (("all", np.ones(len(z0), dtype=bool)), ("|u|<=4 (M0-centred)", np.abs(u) <= 4),
                    ("|u|<=2 (M0-centred)", np.abs(u) <= 2), ("|dZ|<=4sd (zero-centred)", w0 <= 4),
                    ("|dZ|<=2sd (zero-centred)", w0 <= 2), ("end mid off the clip", ~clip)):
        d = drift_test(z0[m], z1[m], L[m], g[m])
        dz = z1[m] - z0[m]
        az = np.abs(z0[m]) >= 0.5
        num = clustered(np.sign(z0[m][az]) * dz[az], g[m][az])
        den = float(np.mean(0.5 * np.abs(z0[m][az]) * dz[az] ** 2))
        out.append({**label, "sample": name, "share_kept": float(m.mean()), "slope": d.get("slope"), "se": d.get("se"),
                    "t_vs_1": d.get("t_vs_1"), "n": d.get("n"),
                    "var_dz_over_L": float(np.mean(dz ** 2) / np.mean(L[m])),
                    "local_ratio_|Z|>=0.5": num["mean"] / den if den > 0 else None,
                    "t_mean_sgn_dz": num["t"]})
    return out


def run_drift_tail(s) -> None:
    """Where in the distribution is the probit-drift shortfall? Hourly dev steps (dev-fitted clock)
    and in-play HF steps (uniform clock) -> drift_tail.json."""
    a = load_panel()
    params = load_params(PARAMS_PATH)["categories"]
    out = []
    for name in ("Sports pre-game", "Entertainment", "Politics", "Climate and Weather", "Mentions"):
        c = CATS.index(name)
        inc = increments(a, c, 1)
        dev = np.flatnonzero(inc["split"] == 0)
        z0, z1 = probit(inc["p0"][dev]), probit(inc["p1"][dev])
        L = np.clip(clock_L(params[name], inc["tau"][dev], 1, inc["hour"][dev]), 1e-10, 30)
        out += drift_tail_rows(z0, z1, L, inc["ev"][dev], {"data": f"hourly {name}", "h": "1h"})
    del a
    paths = hf_paths(s)
    for h in (60, 300):
        st = hf_steps(paths, h, 0.03)
        out += drift_tail_rows(st["z0"], st["z1"], st["L"], st["game"], {"data": "in-play L2", "h": f"{h}s"})
    (OUT / "drift_tail.json").write_text(json.dumps(_jsonable(out)))
    print("done", flush=True)


def run_drift_sim(s) -> None:
    """Probit-drift shortfall on the hourly panel: data vs M0 simulations -> drift_sim.json."""
    a = load_panel()
    params = load_params(PARAMS_PATH)["categories"]
    out = drift_shortfall_panel(a, params)
    (OUT / "drift_sim.json").write_text(json.dumps(_jsonable(out)))
    print("done", flush=True)


def write_hf_report(s) -> Path:
    from albert.reports import Report

    res = json.loads((OUT / "hf_results.json").read_text())
    dsim = json.loads((OUT / "drift_sim.json").read_text()) if (OUT / "drift_sim.json").exists() else []
    rep = Report(s.reports_dir, "research-pm-vol-hf")
    if (OUT / "hf_verdict.md").exists():
        rep.text("Summary", (OUT / "hf_verdict.md").read_text())
    rep.text("Data and definitions", (
        "Recorded Kalshi L2 (public market-data websocket) replayed to top of book, nights Sat 2026-11-07 17 -> "
        "11-08 06 UTC and Sun 11-08 17 -> 11-09 06 UTC. Game-winner series of `volatile.GAME_SERIES`, one leg per "
        "game (first alphabetical non-TIE leg), inside the scheduled in-play window of `volatile._in_play_bounds` "
        "(expected_expiration - league game length .. expected_expiration + 1 h), cut at the first decided quote "
        "(mid <= 2c or >= 98c, spread <= 10c) and 10 min before the scheduled end T = expected_expiration_time. "
        "Mid sampled as of each grid point (last top-of-book change at or before it). Uniform clock: "
        "L = ln(tau / (tau - h)), tau = T - t, u = (Z' - Z e^{L/2}) / sqrt(e^L - 1). Steps start with spread <= 3c "
        "(or 2c) and mid in [3c, 97c] and end with spread <= 10c. t-stats cluster by game. Descriptive only."))
    games = pl.DataFrame(res["games"])
    if games.height:
        rep.table("Games (in-play legs)", games.group_by("night", "series").agg(
            pl.len().alias("games"), pl.col("minutes").sum().round(0).alias("in_play_minutes"),
            pl.col("quote_changes").sum().alias("quote_changes")).sort("night", "series"), max_rows=60)
    m0 = [r for r in res["m0"] if "series" not in r]
    rep.table("Q1 M0 at high frequency, uniform clock: u moments and tails by horizon (N(0,1): P(|u|>2)=4.55%, "
              ">3=0.27%, >4=0.0063%); mean_u2_minus_1 with t by game", _frame(m0), save_as="hf_m0")
    rep.table("Q1 matched control: the same statistics on M0 paths simulated per game and quoted on the 1c grid "
              "with the game's own spreads (what ticks + spreads + M0 alone produce)",
              _frame([r for r in res.get("m0_sim", []) if "series" not in r]), save_as="hf_m0_sim")
    rep.table("Q1 by league (start spread <= 3c)", _frame([r for r in res["m0"] if "series" in r]), save_as="hf_m0_series", max_rows=80)
    rep.table("Q1 variance signature: realised variance of the mid (c^2 per in-play hour) and mean u^2 against "
              "the sampling interval, same windows", _frame(res["signature"]), save_as="hf_signature")
    rep.json("Q1 signature fit RV(h)/T = sigma^2 + 2 eta^2 / h (data | matched M0 control)",
             {"data": res["signature_fit"], "control": res.get("signature_fit_sim")})
    rep.table("Q1 signature of the matched M0 control", _frame(res.get("signature_sim", [])), save_as="hf_signature_sim")
    rep.table("Q2 probit drift at 60 s / 300 s by |Z|: data, M0 simulated on the same rows (true), and re-quoted "
              "on the 1c grid with the observed spreads (tick); e_sgn_dp_c = measured E[sgn(Z) dp] (cents), "
              "implied_sgn_dp_c = what the slope shortfall would imply for the price", _frame(res["drift_hf"]), save_as="hf_drift", max_rows=80)
    if dsim:
        rep.table("Q2 hourly panel (dev): probit drift slope by |Z|, data vs M0 paths on the real hourly grids "
                  "(sim_true: no quoting; sim_tick: 1c grid + real spreads; sim_stale: + re-quoted only in hours "
                  "with real activity)", _frame(dsim), save_as="panel_drift_sim", max_rows=200)
    if (OUT / "drift_tail.json").exists():
        rep.table("Q2 where is the shortfall? Drift slope on all steps and after trimming tails (M0-centred trims "
                  "bias toward 1, zero-centred trims toward 0), and without end mids at the probit clip",
                  _frame(json.loads((OUT / "drift_tail.json").read_text())), save_as="drift_tail", max_rows=80)
    rep.table("Q3 jumps: bipower vs realised variance of the mid and threshold jumps (|u| > 4 local bipower sd, "
              "|dp| >= 2c)", _frame(res["jumps"]), save_as="hf_jumps")
    rep.table("Q3 matched M0 control (no jumps by construction): what ticks alone do to the jump statistics",
              _frame(res.get("jumps_sim", [])), save_as="hf_jumps_sim")
    rep.table("Q3 jump size vs Stern's fixed-score prediction |dp| ~ phi(Z)/sqrt(tau) (slope 1); 'all' has "
              "league fixed effects; SEs cluster by game", _frame(res["stern"]), save_as="hf_stern")
    rep.table("Q4 leverage at 1-5 min: corr(sgn(Z_t) x_t, mean |x| over the next 5 steps), p in [0.1, 0.9]; "
              "negative = vol rises after moves toward 1/2", _frame(res["leverage"]), save_as="hf_leverage")
    rep.table("Q4 matched M0 control (rho = 0 by construction)", _frame(res.get("leverage_sim", [])), save_as="hf_leverage_sim")
    path = rep.save()
    summ = {"report": str(path), "games": games.height, "m0": m0, "signature": res["signature"],
            "signature_fit": res["signature_fit"], "drift_hf": res["drift_hf"],
            "drift_panel_all_rows": [r for r in dsim if r.get("p_range") == "all"],
            "jumps": res["jumps"], "stern": res["stern"], "leverage": res["leverage"],
            "control": {"m0": [r for r in res.get("m0_sim", []) if "series" not in r], "signature_fit": res.get("signature_fit_sim"),
                        "jumps": res.get("jumps_sim"), "leverage": res.get("leverage_sim")}}
    (OUT / "hf_summary.json").write_text(json.dumps(_jsonable(summ), indent=1))
    print(path)
    return path


# ---- entry point ------------------------------------------------------------------------------------
def main(argv: list[str]) -> None:
    from albert.config import settings

    s = settings()
    step = argv[0] if argv else "all"
    if step in ("prep", "all"):
        build_panel(s)
    if step in ("fit", "all"):
        run_fit(s)
    if step in ("vi", "all"):
        run_vi(s)
    if step in ("fc", "all"):
        run_fc(s)
    if step in ("maker", "all"):
        run_maker(s)
    if step in ("report", "all"):
        write_report(s)
    if step == "hf_extract":
        run_hf_extract(s)
    if step == "hf":
        run_hf(s)
    if step == "drift_sim":
        run_drift_sim(s)
    if step == "drift_tail":
        run_drift_tail(s)
    if step == "hf_report":
        write_hf_report(s)


if __name__ == "__main__":
    main(sys.argv[1:])

