"""NWS-based model for Kalshi daily-high temperature brackets (``KXHIGH*``).

Settlement value H = the CLI daily max (whole degrees F) for one station and *standard-time*
day. At decision time t:

    H = max(M + U, R)

* M = max ASOS temperature observed so far in the CLI day (obs scale, known at t);
* U = CLI-minus-observed offset (hourly METARs miss the true peak: U is 0 on 38% of days,
  +1F on 51%, +2F on 9%); its empirical distribution is fit on training days;
* R = max temperature over the rest of the day (CLI scale), Normal(mu + bias, sigma), where
  mu is the NBM forecast for the remaining hours (``txn`` before the day starts; the max of
  the hourly ``tmp`` guidance over the remaining hours during the day), and bias/sigma are
  fit per decision slot on training days.

P(H <= x) = P(M + U <= x) * P(R <= x + 0.5). A bracket [lo, hi] (inclusive integers) has
P(H <= hi) - P(H <= lo - 1). Only NBM runs available at t (runtime + ``NBM_LAG``) and
observations before t are used.
"""

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from albert.fees import kalshi_fee
from albert.research.follow import SIGNAL_SCHEMA
from albert.venues import weather as wx

NBM_LAG = timedelta(hours=2)          # conservative: NBM text is out ~1h after the run time
SLOTS = {"d-1 15h": (-1, 15), "d 10h": (0, 10), "d 13h": (0, 13), "d 16h": (0, 16)}

_SUB_RX = [
    (re.compile(r"^(-?\d+)° or above"), lambda a: (int(a), math.inf)),
    (re.compile(r"^(-?\d+)° or below"), lambda a: (-math.inf, int(a))),
]
_BETWEEN = re.compile(r"^(-?\d+)° to (-?\d+)°")


def bracket(yes_sub_title: str | None) -> tuple[float, float] | None:
    """``"72° to 73°"`` -> (72, 73); ``"83° or above"`` -> (83, inf); ``"65° or below"`` -> (-inf, 65)."""
    s = (yes_sub_title or "").strip()
    m = _BETWEEN.match(s)
    if m:
        return float(m.group(1)), float(m.group(2))
    for rx, f in _SUB_RX:
        m = rx.match(s)
        if m:
            return f(m.group(1))
    return None


def high_markets(markets: pl.DataFrame) -> pl.DataFrame:
    """Settled KXHIGH legs with station code, CLI day and bracket bounds."""
    m = markets.filter(pl.col("series_ticker").str.starts_with("KXHIGH")
                       & pl.col("result").is_in(["yes", "no"]))
    rows = []
    for r in m.iter_rows(named=True):
        code, b = wx.cli_code(r["rules_primary"]), bracket(r["yes_sub_title"])
        day = _event_day(r["event_ticker"])
        if code not in wx.STATION_TZ or b is None or day is None:
            continue
        rows.append({"ticker": r["ticker"], "event_ticker": r["event_ticker"], "code": code,
                     "day": day, "lo": b[0], "hi": b[1], "y": float(r["result"] == "yes"),
                     "close_time": r["close_time"], "fee_type": r["fee_type"],
                     "fee_multiplier": r["fee_multiplier"]})
    return pl.DataFrame(rows, schema={
        "ticker": pl.String, "event_ticker": pl.String, "code": pl.String, "day": pl.Date,
        "lo": pl.Float64, "hi": pl.Float64, "y": pl.Float64,
        "close_time": pl.Datetime("us", "UTC"), "fee_type": pl.String, "fee_multiplier": pl.Float64})


_DAY_RX = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})$")
_MON = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


def _event_day(event_ticker: str) -> date | None:
    m = _DAY_RX.search(event_ticker or "")
    if not m or m.group(2) not in _MON:
        return None
    return date(2000 + int(m.group(1)), _MON[m.group(2)], int(m.group(3)))


def decision_time(code: str, day: date, slot: str) -> datetime:
    offset, hour = SLOTS[slot]
    d = day + timedelta(days=offset)
    return datetime(d.year, d.month, d.day, hour, tzinfo=ZoneInfo(wx.STATION_TZ[code])).astimezone(
        ZoneInfo("UTC"))


# ---- features at a decision time ---------------------------------------------------------------
def features(events: pl.DataFrame, obs: pl.DataFrame, nbm: pl.DataFrame, slot: str) -> pl.DataFrame:
    """Per (code, day): decision time, observed max so far M, and the NBM centre mu.

    ``events`` has columns code, day. Everything used is timestamped before the decision.
    """
    rows = []
    obs_by = obs.partition_by("code", as_dict=True)
    nbm_by = nbm.partition_by("code", as_dict=True)
    for code, day in events.select("code", "day").unique().iter_rows():
        t = decision_time(code, day, slot)
        lo, hi = wx.lst_day_bounds(code, day)
        o = obs_by.get((code,))
        n = nbm_by.get((code,))
        if n is None:
            continue
        m_obs = r_obs = last = None
        if o is not None:
            if t > lo:
                seen = o.filter((pl.col("ts") >= lo) & (pl.col("ts") < t)).sort("ts")["tmpf"]
                m_obs = float(seen.max()) if seen.len() else None
                last = float(seen[-1]) if seen.len() else None
            # Training label only (never a feature): the observed max after t.
            later = o.filter((pl.col("ts") >= max(t, lo)) & (pl.col("ts") < hi))["tmpf"]
            r_obs = float(later.max()) if later.len() else None
        runs = n.filter(pl.col("runtime") + NBM_LAG <= t)
        if runs.is_empty():
            continue
        run = runs["runtime"].max()
        latest = runs.filter(pl.col("runtime") == run)
        if t <= lo:
            # Before the CLI day: NBM's own max-temperature product for that day.
            row = latest.filter(pl.col("txn").is_not_null() & (pl.col("ftime") == datetime(
                day.year, day.month, day.day, tzinfo=ZoneInfo("UTC")) + timedelta(days=1)))
            if row.is_empty():
                continue
            mu, spread = float(row["txn"][0]), float(row["xnd"][0] or 2.5)
        else:
            rest = latest.filter((pl.col("ftime") > t) & (pl.col("ftime") <= hi)
                                 & pl.col("tmp").is_not_null())
            if rest.is_empty():
                mu, spread = -math.inf, 0.0  # nothing left to forecast: the day is (nearly) over
            else:
                # NBM text guidance is 3-hourly, so the next step can be hours away; the latest
                # observation is the floor for the temperature right after t.
                mu, spread = float(rest["tmp"].max()), float("nan")
                if last is not None:
                    mu = max(mu, last)
        rows.append({"code": code, "day": day, "t": t, "m_obs": m_obs, "mu": mu, "spread": spread,
                     "run": run, "hours_left": max((hi - t).total_seconds() / 3600, 0.0),
                     "r_obs": r_obs})
    return pl.DataFrame(rows, schema={
        "code": pl.String, "day": pl.Date, "t": pl.Datetime("us", "UTC"), "m_obs": pl.Float64,
        "r_obs": pl.Float64,
        "mu": pl.Float64, "spread": pl.Float64, "run": pl.Datetime("us", "UTC"),
        "hours_left": pl.Float64})


# ---- model -----------------------------------------------------------------------------------------
def _phi(x: np.ndarray | float) -> np.ndarray:
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2)))


@dataclass
class HighModel:
    """Calibrated distribution of the CLI high at one decision slot."""

    bias: float
    sigma: float
    u_values: np.ndarray            # CLI - obs offsets (F)
    u_probs: np.ndarray
    station_bias: dict[str, float] = field(default_factory=dict)
    tail_weight: float = 0.0        # share of forecast busts (fat tails), fit on training days
    tail_scale: float = 3.0         # bust sd = tail_scale * sigma
    platt: tuple[float, float] = (0.0, 1.0)   # logistic recalibration, fit on training legs

    @classmethod
    def fit(cls, feats: pl.DataFrame, cli: pl.DataFrame, obs_offsets: pl.Series,
            *, min_station_days: int = 15, shrink: float = 15.0) -> "HighModel":
        """Fit bias/sigma of R (the rest-of-day max, CLI scale) on training days.

        Before the CLI day starts, R is the whole-day high: residual = CLI high - mu. During
        the day, R is not directly observed, so it is built from the observed max after t
        (``r_obs``, a training label) plus the CLI-minus-observed offset U: bias = mean(r_obs -
        mu) + E[U], sigma^2 = var(r_obs - mu) + var(U). No selection on the outcome. Station
        biases are shrunk toward 0 with ``shrink`` pseudo-days.
        """
        vals, counts = np.unique(obs_offsets.drop_nulls().to_numpy(), return_counts=True)
        probs = counts / counts.sum()
        eu = float((vals * probs).sum())
        d = feats.join(cli.select("code", "day", "high"), on=["code", "day"]).drop_nulls("high")
        d = d.filter(pl.col("mu").is_finite())
        pre = d.filter(pl.col("m_obs").is_null())
        if pre.height >= intra_min(d):
            bias = float((pre["high"] - pre["mu"]).mean())
            resid = pre.with_columns((pl.col("high") - pl.col("mu") - bias).alias("r"))
        else:
            ok = d.drop_nulls("r_obs")
            bias = float((ok["r_obs"] - ok["mu"]).mean()) + eu
            resid = ok.with_columns((pl.col("r_obs") + eu - pl.col("mu") - bias).alias("r"))
        sb = {}
        for (code,), g in resid.group_by("code"):
            if g.height >= min_station_days:
                sb[code] = float(g["r"].sum() / (g.height + shrink))
        # Fat tails: a robust core sigma (MAD) plus a wide component for forecast busts.
        rr = resid["r"].to_numpy()
        core = max(1.4826 * float(np.median(np.abs(rr - np.median(rr)))), 0.8)
        tail = float(np.mean(np.abs(rr) > 3 * core))
        return cls(bias, core, vals.astype(float), probs, sb, tail_weight=min(max(tail * 2, 0.02), 0.3))

    def cdf(self, x: float, code: str, m_obs: float | None, mu: float) -> float:
        """P(H <= x) for integer x."""
        if not math.isfinite(mu):
            p_rest = 1.0
        else:
            z = x + 0.5 - (mu + self.bias + self.station_bias.get(code, 0.0))
            p_rest = float((1 - self.tail_weight) * _phi(z / self.sigma)
                           + self.tail_weight * _phi(z / (self.sigma * self.tail_scale)))
        if m_obs is None:
            return p_rest
        p_seen = float(self.u_probs[(m_obs + self.u_values) <= x + 1e-9].sum())
        return p_seen * p_rest

    def raw_prob(self, lo: float, hi: float, code: str, m_obs: float | None, mu: float) -> float:
        top = 1.0 if math.isinf(hi) else self.cdf(hi, code, m_obs, mu)
        bottom = 0.0 if math.isinf(lo) else self.cdf(lo - 1, code, m_obs, mu)
        return min(max(top - bottom, 0.0), 1.0)

    def prob(self, lo: float, hi: float, code: str, m_obs: float | None, mu: float) -> float:
        p = self.raw_prob(lo, hi, code, m_obs, mu)
        a, b = self.platt
        if (a, b) == (0.0, 1.0) or p in (0.0, 1.0):
            return p  # exactly 0/1 are logical certainties (e.g. below the observed max)
        x = math.log(min(max(p, 1e-4), 1 - 1e-4) / (1 - min(max(p, 1e-4), 1 - 1e-4)))
        return 1 / (1 + math.exp(-(a + b * x)))

    def recalibrate(self, legs: pl.DataFrame) -> "HighModel":
        """Fit ``platt`` on training legs (columns lo, hi, code, m_obs, mu, y)."""
        from albert.research.literature import logistic_fit

        p = np.array([self.raw_prob(lo, hi, c, m, mu) for lo, hi, c, m, mu in
                      legs.select("lo", "hi", "code", "m_obs", "mu").iter_rows()])
        y = legs["y"].to_numpy()
        keep = (p > 0) & (p < 1)
        if keep.sum() < 200:
            return self
        x = np.log(np.clip(p[keep], 1e-4, 1 - 1e-4) / (1 - np.clip(p[keep], 1e-4, 1 - 1e-4)))
        self.platt = logistic_fit(x, y[keep])
        return self


def intra_min(d: pl.DataFrame) -> int:
    """Use the pre-day fit when most training rows have no observations yet."""
    return max(1, d.height // 2)


def obs_offsets(cli: pl.DataFrame, obs: pl.DataFrame) -> pl.DataFrame:
    """CLI high minus the max hourly observation, per (code, day)."""
    rows = []
    for code, day in cli.select("code", "day").iter_rows():
        lo, hi = wx.lst_day_bounds(code, day)
        rows.append({"code": code, "day": day, "lo": lo, "hi": hi})
    b = pl.DataFrame(rows)
    mx = (obs.join(b, on="code").filter((pl.col("ts") >= pl.col("lo")) & (pl.col("ts") < pl.col("hi")))
          .group_by("code", "day").agg(pl.col("tmpf").max().alias("obs_max")))
    return cli.join(mx, on=["code", "day"]).with_columns((pl.col("high") - pl.col("obs_max")).alias("u"))


# ---- evaluation and signals -------------------------------------------------------------------------
def leg_quotes(legs: pl.DataFrame, candles: pl.DataFrame, feats: pl.DataFrame,
               staleness_hours: float = 2.0) -> pl.DataFrame:
    """Each leg's executable quote at its event's decision time (last candle at or before t)."""
    j = legs.join(feats, on=["code", "day"]).filter(pl.col("t") < pl.col("close_time")).sort("t")
    c = candles.filter(pl.col("yes_bid_close").is_not_null() & pl.col("yes_ask_close").is_not_null()).sort("end_ts")
    return j.join_asof(c.select("ticker", "end_ts", "yes_bid_close", "yes_ask_close"), left_on="t",
                       right_on="end_ts", by="ticker", strategy="backward",
                       tolerance=timedelta(hours=staleness_hours)).filter(pl.col("end_ts").is_not_null())


def with_model(quotes: pl.DataFrame, model: HighModel) -> pl.DataFrame:
    probs = [model.prob(lo, hi, code, m, mu) for lo, hi, code, m, mu in
             quotes.select("lo", "hi", "code", "m_obs", "mu").iter_rows()]
    return quotes.with_columns(pl.Series("prob", probs, dtype=pl.Float64),
                               ((pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2).alias("mid"))


def weather_signals(q: pl.DataFrame, slot: str) -> pl.DataFrame:
    """Best leg/side per event by model edge after the taker fee (the engine applies thresholds)."""
    fee_yes = [kalshi_fee(100, a, fee_type=ft or "quadratic", multiplier=fm) / 100
               for a, ft, fm in q.select("yes_ask_close", "fee_type", "fee_multiplier").iter_rows()]
    fee_no = [kalshi_fee(100, 1 - b, fee_type=ft or "quadratic", multiplier=fm) / 100
              for b, ft, fm in q.select("yes_bid_close", "fee_type", "fee_multiplier").iter_rows()]
    q = q.with_columns(pl.Series("fee_yes", fee_yes), pl.Series("fee_no", fee_no))
    common = [pl.lit("weather").alias("strategy"), pl.lit("kalshi").alias("venue"),
              pl.col("event_ticker").alias("event_id"), pl.col("t").alias("signal_ts"),
              pl.col("close_time").alias("resolve_ts"), pl.lit(None, dtype=pl.Float64).alias("available"),
              "fee_type", "fee_multiplier", pl.col("ticker").alias("market_id"),
              pl.format("{} {} M={} mu={} p={}", pl.lit(slot), "code", "m_obs", pl.col("mu").round(1),
                        pl.col("prob").round(3)).alias("note")]
    yes = q.select(*common, pl.lit("yes").alias("side"), pl.col("yes_ask_close").alias("price"),
                   "prob", pl.col("y").alias("payoff"), (pl.col("prob") - pl.col("yes_ask_close")
                                                         - pl.col("fee_yes")).alias("_e"))
    no = q.select(*common, pl.lit("no").alias("side"), (1 - pl.col("yes_bid_close")).alias("price"),
                  (1 - pl.col("prob")).alias("prob"), (1 - pl.col("y")).alias("payoff"),
                  ((1 - pl.col("prob")) - (1 - pl.col("yes_bid_close")) - pl.col("fee_no")).alias("_e"))
    both = pl.concat([yes, no]).filter(pl.col("price").is_between(0.01, 0.99))
    best = (both.sort(["_e", "market_id", "side"], descending=[True, False, False])
            .unique("event_id", keep="first", maintain_order=True))
    return best.select(list(SIGNAL_SCHEMA)).cast(SIGNAL_SCHEMA)  # type: ignore[arg-type]


def brier(p: pl.Series, y: pl.Series) -> float:
    return float(((p - y) ** 2).mean())
