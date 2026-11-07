"""Event-driven backtester over strategy signals.

A signal is a priced opportunity at a time: buy ``side`` of ``market_id`` at ``price`` with model
probability ``prob``, resolving at ``resolve_ts`` with ``payoff`` (1 win, 0 loss, 0.5 void).
The engine:

* processes signals in time order and settles positions as their resolution time passes,
* sizes each order with fractional Kelly and the shared risk caps (see albert.risk),
* charges each venue's real fee schedule on the actual contract count (per-order rounding),
* keeps separate cash per venue, since money cannot move between venues instantly,
* never adds to a market it already holds.

``payoff`` is only read at settlement and never during sizing, so the engine cannot leak it.
"""

import heapq
import math
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime

import polars as pl

from albert.risk import RiskConfig, size_order

VENUE_WALLET = {"kalshi": "kalshi", "pmus": "pmus", "pm_proxy": "pmus"}


@dataclass
class Fill:
    strategy: str
    venue: str
    market_id: str
    event_id: str
    side: str
    ts: datetime
    resolve_ts: datetime
    price: float
    prob: float
    contracts: int
    fee: float
    cost: float
    payoff: float
    pnl: float = 0.0
    note: str = ""


@dataclass
class BacktestResult:
    start_cash: dict[str, float]
    end_cash: dict[str, float]
    fills: list[Fill]
    skipped: dict[str, int]
    equity: list[tuple[datetime, float]]
    halted_at: datetime | None = None
    open_positions: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def start_equity(self) -> float:
        return sum(self.start_cash.values())

    @property
    def end_equity(self) -> float:
        return sum(self.end_cash.values())

    def fills_frame(self) -> pl.DataFrame:
        return pl.DataFrame([f.__dict__ for f in self.fills]) if self.fills else pl.DataFrame()

    def max_drawdown(self) -> float:
        peak, worst = -math.inf, 0.0
        for _, e in self.equity:
            peak = max(peak, e)
            worst = min(worst, e / peak - 1 if peak > 0 else 0)
        return worst

    def metrics(self) -> dict:
        n = len(self.fills)
        wins = sum(1 for f in self.fills if f.pnl > 0)
        fees = sum(f.fee for f in self.fills)
        staked = sum(f.cost for f in self.fills)
        gross = sum(f.pnl + f.fee for f in self.fills)
        return {
            "start_equity": round(self.start_equity, 2),
            "end_equity": round(self.end_equity, 2),
            "return": round(self.end_equity / self.start_equity - 1, 4) if self.start_equity else 0,
            "max_drawdown": round(self.max_drawdown(), 4),
            "trades": n,
            "win_rate": round(wins / n, 4) if n else None,
            "staked": round(staked, 2),
            "pnl": round(sum(f.pnl for f in self.fills), 2),
            "fees": round(fees, 2),
            "fee_drag_of_gross": round(fees / gross, 4) if gross > 0 else None,
            "roi_per_dollar": round(sum(f.pnl for f in self.fills) / staked, 4) if staked else None,
            "mean_expected_edge": round(sum(f.prob - f.price for f in self.fills) / n, 4) if n else None,
            "halted_at": self.halted_at.isoformat() if self.halted_at else None,
            "open_positions": self.open_positions,
            "skipped": dict(sorted(self.skipped.items(), key=lambda kv: -kv[1])),
        }


def run_backtest(signals: pl.DataFrame, *, cash: dict[str, float], cfg: RiskConfig = RiskConfig(),
                 settle_open: bool = True) -> BacktestResult:
    sig = signals.filter(pl.col("signal_ts").is_not_null() & pl.col("resolve_ts").is_not_null()
                         & pl.col("payoff").is_not_null()).sort(["signal_ts", "market_id", "side"])
    wallet_cash = dict(cash)
    start_cash = dict(cash)
    open_heap: list[tuple[datetime, int, Fill]] = []
    held: set[str] = set()
    event_exposure: dict[str, float] = {}
    fills: list[Fill] = []
    skipped: dict[str, int] = {}
    equity: list[tuple[datetime, float]] = []
    peak = sum(cash.values())
    day = None
    day_start = peak
    halted_at = None
    seq = 0

    def open_cost() -> float:
        return sum(f.cost for _, _, f in open_heap)

    def settle_until(ts: datetime) -> None:
        nonlocal peak
        while open_heap and open_heap[0][0] <= ts:
            _, _, f = heapq.heappop(open_heap)
            proceeds = f.payoff * f.contracts
            f.pnl = proceeds - f.cost
            wallet_cash[VENUE_WALLET[f.venue]] += proceeds
            held.discard(f.market_id)
            event_exposure[f.event_id] = event_exposure.get(f.event_id, 0) - f.cost
            eq = sum(wallet_cash.values()) + open_cost()
            peak = max(peak, eq)
            equity.append((f.resolve_ts, eq))

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for s in sig.iter_rows(named=True):
        ts = s["signal_ts"]
        settle_until(ts)
        eq = sum(wallet_cash.values()) + open_cost()
        if day != ts.date():
            day, day_start = ts.date(), eq
        if halted_at:
            skip("halted_drawdown")
            continue
        if eq < peak * (1 - cfg.drawdown_kill):
            halted_at = ts
            skip("halted_drawdown")
            continue
        if eq < day_start * (1 - cfg.daily_loss_fraction):
            skip("daily_loss_limit")
            continue
        if s["market_id"] in held:
            skip("already_held")
            continue
        if s["resolve_ts"] <= ts:
            skip("already_resolved")
            continue
        wallet = VENUE_WALLET[s["venue"]]
        sz = size_order(venue=s["venue"], prob=s["prob"], price=s["price"], available=s["available"],
                        bankroll=wallet_cash[wallet], event_exposure=event_exposure.get(s["event_id"], 0),
                        cfg=cfg, fee_type=s["fee_type"], multiplier=s["fee_multiplier"])
        if sz.contracts == 0:
            skip(sz.reason)
            continue
        wallet_cash[wallet] -= sz.cost
        f = Fill(strategy=s["strategy"], venue=s["venue"], market_id=s["market_id"],
                 event_id=s["event_id"], side=s["side"], ts=ts, resolve_ts=s["resolve_ts"],
                 price=s["price"], prob=s["prob"], contracts=sz.contracts, fee=sz.fee,
                 cost=sz.cost, payoff=s["payoff"], note=s.get("note") or "")
        fills.append(f)
        held.add(f.market_id)
        event_exposure[f.event_id] = event_exposure.get(f.event_id, 0) + f.cost
        seq += 1
        heapq.heappush(open_heap, (f.resolve_ts, seq, f))

    if settle_open:
        settle_until(datetime(9999, 1, 1, tzinfo=UTC))
    remaining = len(open_heap)
    return BacktestResult(start_cash, wallet_cash, fills, skipped, equity, halted_at, remaining)


def monte_carlo(result: BacktestResult, *, paths: int = 2000, seed: int = 7) -> dict:
    """Resample realized per-dollar returns at their realized bankroll fractions.

    This answers how lucky the realized path was: the spread of outcomes if the same bets had
    arrived in a different order or mix.
    """
    if not result.fills:
        return {}
    rng = random.Random(seed)
    start = result.start_equity
    # Fraction of total equity staked per bet, approximated from realized cost / start equity.
    bets = [(f.cost / start, f.pnl / f.cost) for f in result.fills if f.cost > 0]
    finals = []
    for _ in range(paths):
        b = 1.0
        for _ in range(len(bets)):
            frac, r = bets[rng.randrange(len(bets))]
            b *= max(0.0, 1 + frac * r)
        finals.append(b * start)
    finals.sort()
    q = lambda p: round(finals[min(len(finals) - 1, int(p * len(finals)))], 2)
    return {"paths": paths, "p05": q(0.05), "p50": q(0.5), "p95": q(0.95),
            "prob_loss": round(sum(1 for x in finals if x < start) / paths, 3),
            "prob_halved": round(sum(1 for x in finals if x < start / 2) / paths, 3)}
