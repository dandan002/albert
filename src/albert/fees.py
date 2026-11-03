"""Venue fee models (checked 2026-11-03; re-check the official pages before trading live).

Kalshi (kalshi.com/docs/kalshi-fee-schedule.pdf, docs.kalshi.com/getting_started/fee_rounding):
    taker fee  = 0.07   x multiplier x C x P x (1 - P)
    maker fee  = 0.0175 x multiplier x C x P x (1 - P), only on ``*_with_maker_fees`` series
    The fee itself is exact to $0.000001. Balances settle to the cent and over-payment is
    rebated later. ``conservative=True`` rounds each order up to the cent instead, which is
    the worst case for small orders.

Polymarket US (docs.polymarket.us/fees, effective 2026-10-19):
    taker fee  = theta x C x p x (1 - p), theta = 0.0695 (0.10 for table tennis),
                 banker's rounding to $0.01
    maker      = 0 at tier 1. Rebates only start at tier 2, so they are ignored (conservative).
"""

import math
from decimal import ROUND_HALF_EVEN, Decimal

KALSHI_TAKER = 0.07
KALSHI_MAKER = 0.0175
PMUS_THETA = 0.0695


def kalshi_fee(contracts: float, price: float, *, maker: bool = False,
               fee_type: str | None = "quadratic", multiplier: float | None = 1.0,
               conservative: bool = True) -> float:
    if contracts <= 0:
        return 0.0
    mult = 1.0 if multiplier is None else multiplier
    if maker:
        coef = KALSHI_MAKER if fee_type and "maker_fee" in fee_type else 0.0
    else:
        coef = KALSHI_TAKER
    raw = coef * mult * contracts * price * (1 - price)
    if conservative:
        return math.ceil(round(raw * 100, 9)) / 100
    return math.ceil(raw * 1e6) / 1e6


def pmus_fee(contracts: float, price: float, *, maker: bool = False,
             theta: float | None = PMUS_THETA) -> float:
    if contracts <= 0 or maker:
        return 0.0
    raw = Decimal(str((theta or PMUS_THETA) * contracts * price * (1 - price)))
    return float(raw.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))


def fee(venue: str, contracts: float, price: float, *, maker: bool = False,
        fee_type: str | None = None, multiplier: float | None = None,
        theta: float | None = None) -> float:
    if venue == "kalshi":
        return kalshi_fee(contracts, price, maker=maker, fee_type=fee_type or "quadratic",
                          multiplier=multiplier)
    if venue in ("pmus", "pm_proxy"):
        # Polymarket intl signals are executed on (or proxied by) Polymarket US pricing.
        return pmus_fee(contracts, price, maker=maker, theta=theta)
    raise ValueError(f"unknown venue {venue!r}")
