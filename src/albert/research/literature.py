"""Strategies taken from the literature, re-tested on our own non-financial data.

Each function is a screen: it measures whether the published effect shows up in our sample
after fees, with standard errors clustered by event (legs of one event are not independent).

1. ``maker_taker_returns`` - Bürgi, Deng & Whelan, "Makers and Takers: The Economics of the
   Kalshi Prediction Market" (CEPR DP20631, 2025): makers out-earn takers and makers buying
   contracts >= 50c earned about +2.6%.
2. ``leg_sum_arbitrage`` - Saguillo et al., "Unravelling the Probabilistic Forest: Arbitrage in
   Prediction Markets" (arXiv:2508.03474): in mutually-exclusive multi-outcome events the legs'
   YES prices should sum to 1; deviations are riskless profit.
3. ``calibration_slopes`` / ``slope_signals`` - "calibration is conditional" (arXiv:2602.19520):
   political markets are under-confident (prices pulled toward 50%).
4. ``rolling_min_reversion`` - Quantpedia, "Exploiting Mean-Reversion in Decentralized Prediction
   Markets" (Polymarket): buy at the X-day rolling minimum, hold Y days.
5. ``move_reversal`` - Moskowitz, "Asset Pricing and Sports Betting" (JF 2021): price moves
   before the event over-react and are reversed by the outcome.
"""

from datetime import timedelta

import numpy as np
import polars as pl

from albert.fees import kalshi_fee
from albert.research.follow import SIGNAL_SCHEMA

BANDS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]


def band_expr(col: str = "price") -> pl.Expr:
    labels = [f"{int(lo * 100):02d}-{int(hi * 100):02d}c" for lo, hi in zip(BANDS, BANDS[1:])]
    return pl.col(col).cut(BANDS[1:-1], labels=labels, left_closed=True).cast(pl.String).alias("band")


def cluster_ratio(frame: pl.DataFrame, keys: list[str], *, pnl: str = "pnl", stake: str = "stake",
                  cluster: str = "event_ticker") -> pl.DataFrame:
    """PnL per dollar staked by ``keys`` with a delta-method SE clustered on ``cluster``."""
    totals = frame.group_by(keys).agg(pl.col(pnl).sum().alias("_p"), pl.col(stake).sum().alias("_s"))
    f = frame.join(totals, on=keys).with_columns(
        (pl.col(pnl) - pl.col("_p") / pl.col("_s") * pl.col(stake)).alias("_e"))
    per_cluster = f.group_by([*keys, cluster]).agg(pl.col("_e").sum())
    se = per_cluster.group_by(keys).agg(
        (pl.col("_e") ** 2).sum().sqrt().alias("_se_num"), pl.len().alias("clusters"))
    out = totals.join(se, on=keys).join(frame.group_by(keys).agg(pl.len().alias("n")), on=keys)
    return out.select(
        *keys, "n", "clusters", pl.col("_s").alias("staked"), pl.col("_p").alias("pnl"),
        (pl.col("_p") / pl.col("_s")).alias("roi"),
        (pl.col("_se_num") / pl.col("_s")).alias("se"),
    ).with_columns((pl.col("roi") / pl.col("se")).alias("t")).sort(keys)


def _fee_lookup(frame: pl.DataFrame, *, maker: bool) -> pl.Series:
    """Per-contract Kalshi fee for a 100-contract order (rounded up), memoised on unique inputs."""
    cache: dict[tuple, float] = {}
    out = []
    for p, ft, fm in frame.select("price", "fee_type", "fee_multiplier").iter_rows():
        key = (p, ft, fm)
        if key not in cache:
            cache[key] = kalshi_fee(100, p, maker=maker, fee_type=ft or "quadratic",
                                    multiplier=fm) / 100
        out.append(cache[key])
    return pl.Series(out, dtype=pl.Float64)


# ---- 1. makers vs takers --------------------------------------------------------------------------
def trade_legs(trades: pl.DataFrame, markets: pl.DataFrame) -> pl.DataFrame:
    """Every trade as two purchases: the taker's side and the maker's (opposite) side.

    ``price`` is what that party paid per contract; ``stake`` = contracts x (price + fee) and
    ``pnl`` = contracts x (payoff - price - fee), both in dollars, held to settlement.
    """
    m = markets.filter(pl.col("result").is_in(["yes", "no"])).select(
        "ticker", "event_ticker", "category", "result", "close_time", "fee_type", "fee_multiplier")
    t = trades.join(m, on="ticker").filter(pl.col("yes_price").is_between(0.01, 0.99))
    taker = t.with_columns(pl.lit("taker").alias("role"), pl.col("taker_side").alias("side"))
    maker = t.with_columns(pl.lit("maker").alias("role"),
                           pl.when(pl.col("taker_side") == "yes").then(pl.lit("no"))
                           .otherwise(pl.lit("yes")).alias("side"))
    legs = pl.concat([taker, maker]).with_columns(
        pl.when(pl.col("side") == "yes").then(pl.col("yes_price"))
        .otherwise(1 - pl.col("yes_price")).round(4).alias("price"),
        (pl.col("side") == pl.col("result")).cast(pl.Float64).alias("payoff"),
    )
    parts = []
    for role in ("taker", "maker"):
        part = legs.filter(pl.col("role") == role)
        parts.append(part.with_columns(_fee_lookup(part, maker=role == "maker").alias("fee")))
    legs = pl.concat(parts)
    return legs.with_columns(
        (pl.col("count") * (pl.col("price") + pl.col("fee"))).alias("stake"),
        (pl.col("count") * (pl.col("payoff") - pl.col("price") - pl.col("fee"))).alias("pnl"),
        band_expr(),
        pl.col("created_time").dt.strftime("%Y-%m").alias("month"),
    )


def maker_taker_returns(legs: pl.DataFrame) -> dict[str, pl.DataFrame]:
    fav = legs.filter(pl.col("price") >= 0.5)
    return {
        "by_role": cluster_ratio(legs, ["role"]),
        "by_role_band": cluster_ratio(legs, ["role", "band"]),
        "maker_favorites_by_category": cluster_ratio(fav.filter(pl.col("role") == "maker"),
                                                     ["category"]),
        "favorites_by_role_month": cluster_ratio(fav, ["role", "month"]),
        "maker_favorites_by_category_month": cluster_ratio(fav.filter(pl.col("role") == "maker"),
                                                           ["category", "month"]),
    }


# ---- 2. sum-of-legs arbitrage ---------------------------------------------------------------------
def leg_sum_arbitrage(candles: pl.DataFrame, markets: pl.DataFrame, events: pl.DataFrame
                      ) -> pl.DataFrame:
    """Per (event, hour): cost to buy every YES leg and every NO leg, after taker fees.

    Only mutually-exclusive events whose every listed leg is in our data are used, and only
    hours where every leg has a two-sided quote.

    Kalshi's ``mutually_exclusive`` means *at most* one leg resolves YES, not exactly one.
    So ``short_profit`` = (N - 1) - sum(1 - bid + fee) is a guaranteed floor, while
    ``long_profit`` = 1 - sum(ask + fee) is only riskless if the listed legs are exhaustive,
    which the API does not say. ``*_realized`` uses the actual settlement (``n_yes`` legs won).
    Hourly candle closes are snapshots of the top of book without depth, so positive values
    are *indicative* opportunities.
    """
    ev = events.filter(pl.col("mutually_exclusive"))
    m = markets.select("ticker", "event_ticker", "fee_type", "fee_multiplier",
                       (pl.col("result") == "yes").cast(pl.Int64).alias("won")).join(
        ev.select("event_ticker", "n_markets"), on="event_ticker")
    have = m.group_by("event_ticker").agg(pl.len().alias("legs"), pl.col("won").sum().alias("n_yes"))
    m = m.join(have, on="event_ticker").filter(pl.col("legs") == pl.col("n_markets"))
    c = candles.filter(pl.col("yes_bid_close").is_between(0.001, 0.999)
                       & pl.col("yes_ask_close").is_between(0.001, 0.999)).join(m, on="ticker")
    if c.is_empty():
        return pl.DataFrame()
    c = c.with_columns(pl.col("yes_ask_close").alias("price"))
    c = c.with_columns(_fee_lookup(c, maker=False).alias("fee_yes"))
    c = c.with_columns((1 - pl.col("yes_bid_close")).round(4).alias("price"))
    c = c.with_columns(_fee_lookup(c, maker=False).alias("fee_no")).drop("price")
    hours = c.group_by("event_ticker", "end_ts").agg(
        pl.len().alias("quoted"), pl.first("legs"), pl.first("n_yes"),
        (pl.col("yes_ask_close") + pl.col("fee_yes")).sum().alias("long_cost"),
        (1 - pl.col("yes_bid_close") + pl.col("fee_no")).sum().alias("short_cost"),
        pl.col("yes_ask_close").sum().alias("sum_ask"),
        pl.col("yes_bid_close").sum().alias("sum_bid"),
    ).filter(pl.col("quoted") == pl.col("legs"))
    return hours.with_columns(
        (1 - pl.col("long_cost")).alias("long_profit"),
        (pl.col("legs") - 1 - pl.col("short_cost")).alias("short_profit"),
        (pl.col("n_yes") - pl.col("long_cost")).alias("long_realized"),
        (pl.col("legs") - pl.col("n_yes") - pl.col("short_cost")).alias("short_realized"),
    ).with_columns(pl.max_horizontal("long_profit", "short_profit").alias("best_profit")
                   ).sort("event_ticker", "end_ts")


def leg_sum_summary(hours: pl.DataFrame) -> dict[str, float]:
    if hours.is_empty():
        return {"event_hours": 0}
    short = hours.filter(pl.col("short_profit") > 0)
    long_ = hours.filter(pl.col("long_profit") > 0)
    return {
        "events": hours["event_ticker"].n_unique(),
        "event_hours": hours.height,
        "events_with_zero_winners": int(hours.unique("event_ticker").filter(pl.col("n_yes") == 0).height),
        "median_sum_ask": float(hours["sum_ask"].median()),
        "median_sum_bid": float(hours["sum_bid"].median()),
        # Guaranteed side (needs only mutual exclusivity).
        "short_hours_positive": short.height,
        "short_events_positive": short["event_ticker"].n_unique(),
        "short_max_profit": float(hours["short_profit"].max()),
        # Long side: looks riskless only if the legs are exhaustive.
        "long_hours_positive": long_.height,
        "long_events_positive": long_["event_ticker"].n_unique(),
        "long_events_positive_with_no_winner": long_.filter(pl.col("n_yes") == 0)["event_ticker"].n_unique(),
        "long_mean_realized_when_positive": float(long_["long_realized"].mean() or 0) if long_.height else 0.0,
        "long_events_positive_and_realized_profit": long_.filter(pl.col("long_realized") > 0)["event_ticker"].n_unique(),
    }


# ---- 3. calibration slope by category -------------------------------------------------------------
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def logistic_fit(x: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """MLE of P(y=1) = sigmoid(a + b x) by Newton's method (tiny ridge for stability)."""
    X = np.column_stack([np.ones_like(x), x])
    w = np.array([0.0, 1.0])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (y - p) - 1e-6 * w
        H = -(X.T * (p * (1 - p))) @ X - 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w - step
        if np.abs(step).max() < 1e-9:
            break
    return float(w[0]), float(w[1])


def calibration_slopes(obs: pl.DataFrame, *, min_n: int = 100, boot: int = 200,
                       seed: int = 0) -> pl.DataFrame:
    """Slope b of y ~ sigmoid(a + b logit(mid)) per category. b > 1: prices too close to 50%.

    The SE comes from an event-level bootstrap.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cat, g in [("ALL", obs), *obs.group_by("category")]:
        cat = cat if isinstance(cat, str) else cat[0]
        if g.height < min_n:
            continue
        x, y = _logit(g["mid"].to_numpy()), g["y"].to_numpy()
        a, b = logistic_fit(x, y)
        events = g["event_id"].to_numpy()
        uniq, inv = np.unique(events, return_inverse=True)
        idx_by_event = [np.flatnonzero(inv == i) for i in range(len(uniq))]
        bs = []
        for _ in range(boot):
            pick = np.concatenate([idx_by_event[i] for i in rng.integers(0, len(uniq), len(uniq))])
            bs.append(logistic_fit(x[pick], y[pick])[1])
        se = float(np.std(bs, ddof=1))
        rows.append({"category": cat, "n": g.height, "events": len(uniq), "intercept": a,
                     "slope": b, "slope_se": se, "t_vs_1": (b - 1) / se if se else None})
    return pl.DataFrame(rows).sort("category")


def slope_signals(train: pl.DataFrame, test: pl.DataFrame, *, min_n: int = 200) -> pl.DataFrame:
    """Re-map each test price through its category's train-fit logistic curve, then emit the
    better side (the engine's after-fee edge threshold decides what trades).

    Categories with fewer than ``min_n`` training markets use the identity (no edge).
    """
    params: dict[str, tuple[float, float]] = {}
    for (cat,), g in train.group_by("category"):
        if g.height >= min_n:
            params[cat] = logistic_fit(_logit(g["mid"].to_numpy()), g["y"].to_numpy())
    a = test["category"].replace_strict({k: v[0] for k, v in params.items()}, default=0.0,
                                        return_dtype=pl.Float64)
    b = test["category"].replace_strict({k: v[1] for k, v in params.items()}, default=1.0,
                                        return_dtype=pl.Float64)
    x = _logit(test["mid"].to_numpy())
    prob = 1 / (1 + np.exp(-(a.to_numpy() + b.to_numpy() * x)))
    p = test.with_columns(pl.Series("prob", prob))
    common = [pl.lit("slope").alias("strategy"), "venue", "event_id",
              pl.col("decision_ts").alias("signal_ts"), "resolve_ts",
              pl.lit(None, dtype=pl.Float64).alias("available"), "fee_type", "fee_multiplier",
              "market_id", pl.format("{} p={}", "category", pl.col("prob").round(3)).alias("note")]
    yes = p.select(*common, pl.lit("yes").alias("side"), pl.col("yes_ask").alias("price"),
                   "prob", pl.col("y").alias("payoff"))
    no = p.select(*common, pl.lit("no").alias("side"), (1 - pl.col("yes_bid")).alias("price"),
                  (1 - pl.col("prob")).alias("prob"), (1 - pl.col("y")).alias("payoff"))
    both = pl.concat([yes, no]).with_columns((pl.col("prob") - pl.col("price")).alias("_e"))
    best = (both.sort(["_e", "market_id", "side"], descending=[True, False, False])
            .unique("market_id", keep="first", maintain_order=True))
    return best.select(list(SIGNAL_SCHEMA)).cast(SIGNAL_SCHEMA)  # type: ignore[arg-type]


# ---- 4. rolling-minimum mean reversion -----------------------------------------------------------
def rolling_min_reversion(candles: pl.DataFrame, markets: pl.DataFrame, *, lookback_hours: int,
                          hold_hours: int, max_spread: float = 0.10, lo: float = 0.05,
                          hi: float = 0.95) -> pl.DataFrame:
    """Buy YES at the ask when the mid is at its trailing ``lookback_hours`` minimum; sell at the
    bid ``hold_hours`` later (taker fees both ways), or hold to settlement if it closes first.

    Positions in one market never overlap. The trailing window excludes the current candle.
    """
    m = markets.filter(pl.col("result").is_in(["yes", "no"])).select(
        "ticker", "event_ticker", "category", "close_time", "fee_type", "fee_multiplier",
        (pl.col("result") == "yes").cast(pl.Float64).alias("y"))
    c = (candles.filter(pl.col("yes_bid_close").is_between(0.001, 0.999)
                        & pl.col("yes_ask_close").is_between(0.001, 0.999))
         .join(m, on="ticker").sort("ticker", "end_ts")
         .with_columns(((pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2).alias("mid")))
    c = c.with_columns(
        pl.col("mid").shift(1).rolling_min_by("end_ts", window_size=f"{lookback_hours}h")
        .over("ticker").alias("trail_min"),
        (pl.col("end_ts") - pl.col("end_ts").min().over("ticker")).alias("age"),
    )
    entries = c.filter(
        (pl.col("age") >= timedelta(hours=lookback_hours))
        & (pl.col("mid") <= pl.col("trail_min"))
        & ((pl.col("yes_ask_close") - pl.col("yes_bid_close")) <= max_spread)
        & pl.col("mid").is_between(lo, hi)
        & (pl.col("end_ts") < pl.col("close_time") - timedelta(hours=1))
    )
    exits = c.select("ticker", pl.col("end_ts").alias("exit_ts"),
                     pl.col("yes_bid_close").alias("exit_bid")).sort("exit_ts")
    trades = []
    exits_by = exits.partition_by("ticker", as_dict=True)
    for (ticker,), g in entries.group_by("ticker"):
        busy_until = None
        ex = exits_by[(ticker,)]
        for r in g.sort("end_ts").iter_rows(named=True):
            if busy_until is not None and r["end_ts"] < busy_until:
                continue
            target = r["end_ts"] + timedelta(hours=hold_hours)
            if target >= r["close_time"]:
                exit_ts, exit_px, settled = r["close_time"], r["y"], True
            else:
                later = ex.filter(pl.col("exit_ts") >= target).head(1)
                if later.is_empty():
                    exit_ts, exit_px, settled = r["close_time"], r["y"], True
                else:
                    exit_ts, exit_px, settled = later["exit_ts"][0], later["exit_bid"][0], False
            ask = r["yes_ask_close"]
            fee_in = kalshi_fee(100, ask, fee_type=r["fee_type"] or "quadratic",
                                multiplier=r["fee_multiplier"]) / 100
            fee_out = 0.0 if settled else kalshi_fee(
                100, exit_px, fee_type=r["fee_type"] or "quadratic",
                multiplier=r["fee_multiplier"]) / 100
            trades.append({
                "ticker": ticker, "event_ticker": r["event_ticker"], "category": r["category"],
                "entry_ts": r["end_ts"], "exit_ts": exit_ts, "entry_ask": ask, "mid": r["mid"],
                "exit_price": exit_px, "settled": settled,
                "stake": ask + fee_in, "pnl": exit_px - fee_out - ask - fee_in,
                "gross": exit_px - ask,
            })
            busy_until = exit_ts
    schema = {"ticker": pl.String, "event_ticker": pl.String, "category": pl.String,
              "entry_ts": pl.Datetime("us", "UTC"), "exit_ts": pl.Datetime("us", "UTC"),
              "entry_ask": pl.Float64, "mid": pl.Float64, "exit_price": pl.Float64,
              "settled": pl.Boolean, "stake": pl.Float64, "pnl": pl.Float64, "gross": pl.Float64}
    return pl.DataFrame(trades, schema=schema)


# ---- 5. pre-event move reversal -------------------------------------------------------------------
def move_reversal(obs_late: pl.DataFrame, obs_early: pl.DataFrame) -> pl.DataFrame:
    """Join the same market at two decision times; ``move`` = late mid - early mid and
    ``residual`` = outcome - late mid. A negative move->residual slope means over-reaction.
    """
    e = obs_early.select("market_id", pl.col("mid").alias("mid_early"),
                         pl.col("decision_ts").alias("ts_early"))
    j = obs_late.join(e, on="market_id").filter(pl.col("ts_early") < pl.col("decision_ts"))
    return j.with_columns((pl.col("mid") - pl.col("mid_early")).alias("move"),
                          (pl.col("y") - pl.col("mid")).alias("residual"))


def move_reversal_slopes(moves: pl.DataFrame, *, min_n: int = 100) -> pl.DataFrame:
    """OLS slope of residual on move per category, SE clustered by event."""
    rows = []
    for cat, g in [("ALL", moves), *moves.group_by("category")]:
        cat = cat if isinstance(cat, str) else cat[0]
        g = g.filter(pl.col("move").abs() > 0)
        if g.height < min_n:
            continue
        x, y = g["move"].to_numpy(), g["residual"].to_numpy()
        xc = x - x.mean()
        beta = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
        resid = (y - y.mean()) - beta * xc
        score = pl.DataFrame({"ev": g["event_id"], "s": xc * resid}).group_by("ev").agg(pl.col("s").sum())
        se = float(np.sqrt((score["s"].to_numpy() ** 2).sum()) / (xc ** 2).sum())
        big = g.filter(pl.col("move").abs() >= 0.10)
        rows.append({"category": cat, "n": g.height, "events": g["event_id"].n_unique(),
                     "slope": beta, "se": se, "t": beta / se if se else None,
                     "n_big_moves": big.height,
                     "mean_residual_after_big_up": float(big.filter(pl.col("move") > 0)["residual"].mean() or 0),
                     "mean_residual_after_big_down": float(big.filter(pl.col("move") < 0)["residual"].mean() or 0)})
    return pl.DataFrame(rows).sort("category")


# ---- sample hygiene --------------------------------------------------------------------------------
GAME_SERIES = ("KXNFLGAME", "KXNBAGAME", "KXNCAAFGAME", "KXMLBGAME", "KXNHLGAME", "KXWNBAGAME")


def full_population_legs(markets: pl.DataFrame, series: tuple[str, ...] = GAME_SERIES) -> pl.Series:
    """The first leg (alphabetically) of *every* settled event in ``series``.

    This is the selection ``ingest kalshi-detail --series`` fetches. It does not depend on
    volume or outcome, unlike a top-N-by-volume sample (upsets attract volume).
    """
    return (markets.filter(pl.col("series_ticker").is_in(list(series))).sort("ticker")
            .unique("event_ticker", keep="first", maintain_order=True)["ticker"])


def flat_signal_returns(signals: pl.DataFrame, *, min_edge: float = 0.03, lo: float = 0.03,
                        hi: float = 0.97) -> pl.DataFrame:
    """Flat one-contract returns of signals whose after-fee model edge clears ``min_edge``.

    Compounding equity curves are dominated by a few large late bets; this is the per-dollar
    edge of the signal itself, clustered by event.
    """
    fees = [kalshi_fee(100, p, fee_type=ft or "quadratic", multiplier=fm) / 100
            for p, ft, fm in signals.select("price", "fee_type", "fee_multiplier").iter_rows()]
    f = signals.with_columns(pl.Series("fee", fees, dtype=pl.Float64)).filter(
        ((pl.col("prob") - pl.col("price") - pl.col("fee")) >= min_edge)
        & pl.col("price").is_between(lo, hi))
    f = f.with_columns((pl.col("payoff") - pl.col("price") - pl.col("fee")).alias("pnl"),
                       (pl.col("price") + pl.col("fee")).alias("stake"),
                       pl.col("event_id").alias("event_ticker"), pl.lit("all").alias("_k"))
    if f.is_empty():
        return pl.DataFrame()
    return cluster_ratio(f, ["_k"]).drop("_k")
