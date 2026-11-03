"""Position sizing and risk limits, shared by the backtester and the paper trader."""

import math
from dataclasses import dataclass, field
from datetime import date

from albert.fees import fee


@dataclass(frozen=True)
class RiskConfig:
    kelly_fraction: float = 0.25        # fraction of full Kelly
    max_bet_fraction: float = 0.05      # cap per order, as a fraction of venue bankroll
    max_bet_dollars: float = 10.0       # hard cap per order
    max_event_fraction: float = 0.10    # cap on total exposure to one event
    daily_loss_fraction: float = 0.10   # stop opening positions after this daily loss
    min_edge: float = 0.03              # required edge per contract after fees
    min_price: float = 0.03             # avoid sub-3c contracts (fee and tick noise)
    max_price: float = 0.97
    drawdown_kill: float = 0.5          # halt the strategy if equity falls below 50% of peak


@dataclass
class Sizing:
    contracts: int
    cost: float
    fee: float
    edge_per_contract: float
    reason: str


@dataclass
class RiskState:
    cash: dict[str, float]
    peak_equity: float
    event_exposure: dict[str, float] = field(default_factory=dict)
    day: date | None = None
    day_start_equity: float = 0.0
    halted: bool = False


def kelly_fraction(prob: float, price: float) -> float:
    """Full-Kelly bankroll fraction for buying a $1 binary contract at ``price``."""
    if price <= 0 or price >= 1:
        return 0.0
    return max(0.0, (prob - price) / (1 - price))


def size_order(*, venue: str, prob: float, price: float, available: float | None,
               bankroll: float, event_exposure: float, cfg: RiskConfig,
               fee_type: str | None = None, multiplier: float | None = None,
               theta: float | None = None, min_qty: int = 1) -> Sizing:
    if not (cfg.min_price <= price <= cfg.max_price):
        return Sizing(0, 0, 0, 0, "price_out_of_range")
    one_fee = fee(venue, 100, price, fee_type=fee_type, multiplier=multiplier, theta=theta) / 100
    edge = prob - price - one_fee
    if edge < cfg.min_edge:
        return Sizing(0, 0, 0, edge, "edge_below_threshold")
    f = cfg.kelly_fraction * kelly_fraction(prob - one_fee, price)
    stake = min(f * bankroll, cfg.max_bet_fraction * bankroll, cfg.max_bet_dollars,
                max(0.0, cfg.max_event_fraction * bankroll - event_exposure))
    contracts = math.floor(stake / price)
    if available is not None:
        contracts = min(contracts, math.floor(available))
    if contracts < min_qty:
        return Sizing(0, 0, 0, edge, "below_min_size")
    # Fees are charged per order, so re-check the edge with the actual (rounded) fee.
    order_fee = fee(venue, contracts, price, fee_type=fee_type, multiplier=multiplier, theta=theta)
    if (prob - price) * contracts - order_fee < cfg.min_edge * contracts:
        return Sizing(0, 0, 0, edge, "edge_lost_to_fee_rounding")
    cost = contracts * price + order_fee
    if cost > bankroll:
        return Sizing(0, 0, 0, edge, "insufficient_cash")
    return Sizing(contracts, cost, order_fee, edge, "ok")
