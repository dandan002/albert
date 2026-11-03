"""Maker (resting limit order) backtest with a conservative, trade-driven fill model.

Hypothesis (Bürgi, Deng & Whelan 2025; replicated in ``research literature``): makers who buy
favorites earn a small premium that takers pay. The historical result measures *other
people's* fills, so this module asks the real question: if *we* had posted a bid, would it
have filled, and what would the fills have earned?

Order rule, per market, at ``decision_ts = expected_expiration_time - horizon``:
  * favorite side = YES if the candle mid >= 0.5, else NO;
  * our bid = favorite best bid + 1 tick when the spread is >= 2 ticks (we create a new best
    price, so we are first in its queue), otherwise the best bid (we join an existing queue
    of unknown size);
  * the order rests until the market stops trading (``close_time``), ``ttl_hours``, or
    ``start_buffer`` before the game start encoded in the ticker, whichever is first; an
    order that would only rest after that point is never posted (resting into a live game
    is filled mostly when the favorite is collapsing).

Fill rule, using real Kalshi trades after ``decision_ts`` (``taker_side`` is the taker's side):
  * a YES bid at q is hit by takers *selling* YES, i.e. trades with ``taker_side == "no"``;
    a NO bid at q is hit by trades with ``taker_side == "yes"``, at YES price 1 - q;
  * when we improved the price, a trade *at* our price fills us (we are first in queue);
    when we joined, only a trade *through* our price counts (everyone ahead must be filled);
  * fills accumulate trade by trade until the order size is reached.

This ignores the price impact of our own order (tiny at a few contracts) and assumes other
makers did not jump ahead of us in the same tick. Fills are held to settlement and pay the
Kalshi maker fee (zero except on ``*_with_maker_fees`` series).
"""

import heapq
import math
from dataclasses import dataclass
from datetime import timedelta

import polars as pl

from albert.fees import kalshi_fee
from albert.research.literature import cluster_ratio
from albert.venues.kalshi import ticker_start

ORDER_SCHEMA = {
    "ticker": pl.String, "event_ticker": pl.String, "category": pl.String,
    "decision_ts": pl.Datetime("us", "UTC"), "cancel_ts": pl.Datetime("us", "UTC"),
    "resolve_ts": pl.Datetime("us", "UTC"), "side": pl.String, "bid": pl.Float64,
    "ask": pl.Float64, "improved": pl.Boolean, "payoff": pl.Float64, "fee_type": pl.String,
    "fee_multiplier": pl.Float64,
}


def game_key_expr(col: str = "event_ticker") -> pl.Expr:
    """``KXCS2SPREAD-<key>`` / ``KXCS2GAME-<key>``: the part after the series identifies the game."""
    return pl.col(col).str.split_exact("-", 1).struct.field("field_1").fill_null(pl.col(col))


def quote_order(yes_bid: float, yes_ask: float, *, tick: float = 0.01, lo: float = 0.50,
                hi: float = 0.95, max_spread: float = 0.10) -> dict | None:
    """The live version of the order rule in ``maker_orders`` (same favorite/price logic)."""
    if yes_bid is None or yes_ask is None or not (0 < yes_bid < yes_ask < 1):
        return None
    if not (tick - 1e-9 <= yes_ask - yes_bid <= max_spread):
        return None
    yes_fav = (yes_bid + yes_ask) / 2 >= 0.5
    fav_bid, fav_ask = (yes_bid, yes_ask) if yes_fav else (1 - yes_ask, 1 - yes_bid)
    improved = fav_ask - fav_bid >= 2 * tick - 1e-9
    bid = round(fav_bid + tick if improved else fav_bid, 4)
    if not (lo <= bid <= hi):
        return None
    return {"side": "yes" if yes_fav else "no", "bid": bid, "ask": round(fav_ask, 4),
            "improved": improved}


def maker_orders(markets: pl.DataFrame, candles: pl.DataFrame, trades: pl.DataFrame, *,
                 horizon_hours: float, categories: tuple[str, ...] = ("Climate and Weather", "Sports"),
                 lo: float = 0.50, hi: float = 0.95, tick: float = 0.01, ttl_hours: float | None = None,
                 max_spread: float = 0.10, staleness_hours: float = 6.0,
                 start_buffer_minutes: float = 15.0, queue: str = "any",
                 windows: pl.DataFrame | None = None) -> pl.DataFrame:
    """One candidate maker order per market (see module docstring).

    Only markets with trades covering the whole order life are kept. With ``windows``
    (``kalshi_trade_windows``: complete trades in [window_start, window_end]) coverage is
    exact. Without it, the trade sample is the newest N per market, so the decision must be
    after the first recorded trade, which quietly drops busy markets.
    """
    m = markets.filter(pl.col("result").is_in(["yes", "no"]) & pl.col("category").is_in(list(categories))
                       ).with_columns(
        (pl.coalesce("expected_expiration_time", "close_time") - timedelta(hours=horizon_hours)
         ).alias("decision_ts"))
    m = m.filter(pl.col("decision_ts") < pl.col("close_time"))
    if windows is not None:
        m = m.join(windows.select("ticker", "window_start", "window_end"), on="ticker").filter(
            pl.col("window_start") <= pl.col("decision_ts"))
    else:
        first_trade = trades.group_by("ticker").agg(pl.col("created_time").min().alias("first_trade"))
        m = m.join(first_trade, on="ticker").filter(pl.col("first_trade") <= pl.col("decision_ts"))
    c = candles.filter(pl.col("yes_bid_close").is_between(0.001, 0.999)
                       & pl.col("yes_ask_close").is_between(0.001, 0.999)).sort("end_ts")
    j = m.sort("decision_ts").join_asof(
        c.select("ticker", "end_ts", "yes_bid_close", "yes_ask_close"), left_on="decision_ts",
        right_on="end_ts", by="ticker", strategy="backward",
        tolerance=timedelta(hours=staleness_hours)).filter(pl.col("end_ts").is_not_null())
    j = j.filter((pl.col("yes_ask_close") - pl.col("yes_bid_close")).is_between(tick - 1e-9, max_spread))
    yes_fav = ((pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2) >= 0.5
    fav_bid = pl.when(yes_fav).then(pl.col("yes_bid_close")).otherwise(1 - pl.col("yes_ask_close"))
    fav_ask = pl.when(yes_fav).then(pl.col("yes_ask_close")).otherwise(1 - pl.col("yes_bid_close"))
    spread = fav_ask - fav_bid
    improved = spread >= 2 * tick - 1e-9
    bid = pl.when(improved).then(fav_bid + tick).otherwise(fav_bid).round(4)
    j = j.with_columns(pl.col("event_ticker").map_elements(
        ticker_start, return_dtype=pl.Datetime("us", "UTC")).alias("_start"))
    cancel = pl.min_horizontal(pl.col("close_time"),
                               pl.col("_start") - timedelta(minutes=start_buffer_minutes))
    if ttl_hours is not None:
        cancel = pl.min_horizontal(cancel, pl.col("decision_ts") + timedelta(hours=ttl_hours))
    out = j.select(
        "ticker", "event_ticker", "category", "decision_ts", cancel.alias("cancel_ts"),
        pl.coalesce("settlement_ts", "close_time").alias("resolve_ts"),
        pl.when(yes_fav).then(pl.lit("yes")).otherwise(pl.lit("no")).alias("side"),
        bid.alias("bid"), fav_ask.round(4).alias("ask"), improved.alias("improved"),
        pl.when(yes_fav == (pl.col("result") == "yes")).then(1.0).otherwise(0.0).alias("payoff"),
        "fee_type", "fee_multiplier",
        *([pl.col("window_end")] if windows is not None else []),
    ).filter(pl.col("bid").is_between(lo, hi) & (pl.col("cancel_ts") > pl.col("decision_ts")))
    if windows is not None:  # the whole resting period must be inside the fetched window
        out = out.filter(pl.col("cancel_ts") <= pl.col("window_end")).drop("window_end")
    if queue == "join":      # only 1-tick markets: rest at the existing best bid
        out = out.filter(~pl.col("improved"))
    elif queue == "improve":  # only wider markets: become the new best bid
        out = out.filter(pl.col("improved"))
    elif queue != "any":
        raise ValueError(f"unknown queue mode {queue!r}")
    # One order per underlying game (legs, spreads and totals of one match are one bet), as
    # in the paper trader: KXCS2GAME-<key> and KXCS2SPREAD-<key> share <key>.
    out = (out.with_columns(game_key_expr().alias("_game"))
           .sort("decision_ts", "ticker").unique("_game", keep="first", maintain_order=True)
           .drop("_game"))
    return out.cast(ORDER_SCHEMA)  # type: ignore[arg-type]


def simulate_fills(orders: pl.DataFrame, trades: pl.DataFrame, *, size: int) -> pl.DataFrame:
    """Contracts filled per order (0..size) and the fill time, from real trades only."""
    t = trades.join(orders.select("ticker", "decision_ts", "cancel_ts", "side", "bid", "improved"),
                    on="ticker").filter(
        (pl.col("created_time") > pl.col("decision_ts")) & (pl.col("created_time") < pl.col("cancel_ts")))
    # Price of the trade on our side, and whether the taker sold into our side.
    our_px = pl.when(pl.col("side") == "yes").then(pl.col("yes_price")).otherwise(1 - pl.col("yes_price"))
    hits_us = pl.col("taker_side") != pl.col("side")
    at_or_through = pl.when(pl.col("improved")).then(our_px <= pl.col("bid") + 1e-9).otherwise(
        our_px < pl.col("bid") - 1e-9)
    t = t.filter(hits_us & at_or_through).sort("ticker", "created_time").with_columns(
        pl.col("count").cum_sum().over("ticker").alias("cum"))
    first_full = t.filter(pl.col("cum") >= size).group_by("ticker").agg(
        pl.col("created_time").min().alias("full_ts"))
    total = t.group_by("ticker").agg(pl.col("count").sum().alias("hit"),
                                     pl.col("created_time").min().alias("first_fill_ts"))
    out = orders.join(total, on="ticker", how="left").join(first_full, on="ticker", how="left")
    return out.with_columns(
        pl.min_horizontal(pl.col("hit").fill_null(0), pl.lit(size)).floor().cast(pl.Int64).alias("filled"),
        pl.coalesce("full_ts", "first_fill_ts").alias("fill_ts"),
    ).drop("hit", "full_ts", "first_fill_ts")


def with_pnl(fills: pl.DataFrame) -> pl.DataFrame:
    """Per-order cost and PnL held to settlement, maker fee rounded up per order."""
    fees = [kalshi_fee(n, b, maker=True, fee_type=ft or "quadratic", multiplier=fm) if n else 0.0
            for n, b, ft, fm in fills.select("filled", "bid", "fee_type", "fee_multiplier").iter_rows()]
    return fills.with_columns(pl.Series("fee", fees, dtype=pl.Float64)).with_columns(
        (pl.col("filled") * pl.col("bid") + pl.col("fee")).alias("stake"),
        (pl.col("filled") * (pl.col("payoff") - pl.col("bid")) - pl.col("fee")).alias("pnl"),
        # The taker alternative: lift the ask at the decision time, taker fee.
        (pl.col("payoff") - pl.col("ask")).alias("taker_gross"),
    )


def taker_benchmark(orders: pl.DataFrame) -> pl.DataFrame:
    """Same markets and side, bought at the ask at the decision time (1 contract, taker fee)."""
    fees = [kalshi_fee(100, a, fee_type=ft or "quadratic", multiplier=fm) / 100
            for a, ft, fm in orders.select("ask", "fee_type", "fee_multiplier").iter_rows()]
    return orders.with_columns(pl.Series("fee", fees, dtype=pl.Float64)).with_columns(
        (pl.col("ask") + pl.col("fee")).alias("stake"),
        (pl.col("payoff") - pl.col("ask") - pl.col("fee")).alias("pnl"))


def summarize(frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """ROI per dollar with SEs clustered by *game*, not event: one match has several events
    (moneyline, spread, totals) whose outcomes move together."""
    filled = frame.filter(pl.col("stake") > 0).with_columns(game_key_expr().alias("game"))
    if filled.is_empty():
        return pl.DataFrame()
    return cluster_ratio(filled, keys, cluster="game")


@dataclass
class MakerBankroll:
    start_cash: float
    end_cash: float
    orders_placed: int
    orders_filled: int
    skipped_no_cash: int
    max_drawdown: float
    curve: pl.DataFrame


def run_maker_bankroll(orders: pl.DataFrame, trades: pl.DataFrame, *, cash: float,
                       max_bet_fraction: float = 0.05, max_bet_dollars: float = 10.0,
                       max_event_fraction: float = 0.10) -> MakerBankroll:
    """Chronological bankroll simulation.

    Each order reserves its full collateral (contracts x bid + fee) when posted, as Kalshi
    does for resting orders; unfilled contracts release it at cancel time, filled ones pay out
    at settlement. Size = min(max_bet_fraction x free cash, max_bet_dollars) / bid.
    """
    # Heap items: (ts, cash returned, event, collateral locked until ts). Equity counts locked
    # collateral at cost, so it moves only when cash comes back.
    free = cash
    heap: list[tuple] = []
    event_locked: dict[str, float] = {}
    placed = filled_n = skipped = 0
    curve: list[dict] = []
    peak, max_dd = cash, 0.0

    def release(until) -> None:
        nonlocal free, peak, max_dd
        while heap and (until is None or heap[0][0] <= until):
            ts, back, ev, locked = heapq.heappop(heap)
            free += back
            event_locked[ev] = event_locked.get(ev, 0.0) - locked
            eq = free + sum(h[3] for h in heap)
            peak = max(peak, eq)
            max_dd = min(max_dd, eq / peak - 1)
            curve.append({"ts": ts, "equity": eq})

    by_ticker = trades.filter(pl.col("ticker").is_in(orders["ticker"].implode())).partition_by(
        "ticker", as_dict=True)
    empty = trades.clear()
    for r in orders.sort("decision_ts", "ticker").iter_rows(named=True):
        release(r["decision_ts"])
        locked_total = sum(h[3] for h in heap)
        budget = min(max_bet_fraction * free, max_bet_dollars,
                     max_event_fraction * (free + locked_total) - event_locked.get(r["event_ticker"], 0.0))
        n = math.floor(budget / (r["bid"] * 1.01)) if r["bid"] > 0 else 0
        if n < 1:
            skipped += 1
            continue
        own = by_ticker.get((r["ticker"],), empty)
        o = with_pnl(simulate_fills(pl.DataFrame([r], schema=ORDER_SCHEMA), own, size=n)).row(0, named=True)
        reserve = n * r["bid"] + kalshi_fee(n, r["bid"], maker=True, fee_type=r["fee_type"] or "quadratic",
                                            multiplier=r["fee_multiplier"])
        free -= reserve
        placed += 1
        event_locked[r["event_ticker"]] = event_locked.get(r["event_ticker"], 0.0) + reserve
        if o["filled"]:
            filled_n += 1
            unfilled = reserve - o["stake"]
            if unfilled > 1e-9:
                heapq.heappush(heap, (r["cancel_ts"], unfilled, r["event_ticker"], unfilled))
            heapq.heappush(heap, (r["resolve_ts"], o["filled"] * r["payoff"], r["event_ticker"], o["stake"]))
        else:
            heapq.heappush(heap, (r["cancel_ts"], reserve, r["event_ticker"], reserve))
    release(None)
    return MakerBankroll(cash, free, placed, filled_n, skipped, max_dd,
                         pl.DataFrame(curve, schema={"ts": pl.Datetime("us", "UTC"), "equity": pl.Float64}))
