"""Shared definitions for the favourite-longshot-bias / retail-flow study (``paper_flb/``).

Every workstream (``flb_xsection``, ``flb_flow``, ``flb_wallets``, ``flb_structural``) uses these so
that categories, price buckets, returns and fees mean the same thing in every table.

Conventions
-----------
* **Side price** ``q``: the price paid for the side actually bought (YES at ``p`` -> ``q = p``; NO at
  YES price ``p`` -> ``q = 1 - p``), in dollars on (0, 1). ``won`` is 1 if that side paid out.
  A **longshot** is a side bought at ``q <= LONGSHOT`` (10c); a **favourite** at ``q >= FAVOURITE``.
* **Returns** per contract: ``won - q - fee``; per dollar: ``(won - q - fee) / (q + fee)``.
  Gross returns set ``fee = 0``.
* **Fees** (Kalshi, checked 2026-11-03): taker ``0.07 x mult x q (1 - q)`` per contract; maker
  ``0.0175 x mult x q (1 - q)`` only on maker-fee series. Fees are symmetric in ``q`` and ``1 - q``,
  so the YES and NO side of a trade pay the same fee. Per-order rounding up to the cent is a
  separate, conservative variant (``albert.fees.kalshi_fee(conservative=True)``).
* **Categories**: Kalshi series categories mapped to the coarse groups in ``COARSE``; financial
  categories and money-related series (``albert.categories.kalshi_series_is_financial``) are out of
  scope for this project and are excluded. Polymarket tags are mapped with ``pm_coarse_category``.
* **Inference**: means with standard errors clustered by event (``clustered``).
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import polars as pl

from albert.categories import PM_FINANCIAL_TAGS, kalshi_series_is_financial

LONGSHOT = 0.10
FAVOURITE = 0.90
TAKER_RATE = 0.07
MAKER_RATE = 0.0175

COARSE: dict[str, str] = {
    "Sports": "Sports",
    "Climate and Weather": "Weather",
    "Entertainment": "Entertainment", "Culture": "Entertainment", "Social": "Entertainment",
    "Mentions": "Mentions",
    "Science and Technology": "Science & Tech", "Health": "Science & Tech",
    "Politics": "Politics", "Elections": "Politics", "World": "Politics",
    "Transportation": "Other", "Exotics": "Other", "Education": "Other",
}
GROUPS = ("Sports", "Weather", "Politics", "Entertainment", "Mentions", "Science & Tech", "Other")

# Price buckets on the side price, in cents: (0,2], (2,5], (5,10], ..., (95,98], (98,100)
EDGES_C = (0, 2, 5, 10, 20, 35, 50, 65, 80, 90, 95, 98, 100)
BUCKET_LABELS = (*(f"{a + 1}-{b}c" for a, b in itertools.pairwise(EDGES_C[:-1])), "99c")

# Trade-size buckets by dollar notional (count x price paid): a retail proxy where identities are hidden
SIZE_EDGES_USD = (0.0, 10.0, 100.0, 1000.0, math.inf)
SIZE_LABELS = ("<=$10", "$10-100", "$100-1k", ">$1k")


def size_bucket_expr(notional: str | pl.Expr = "notional") -> pl.Expr:
    c = pl.col(notional) if isinstance(notional, str) else notional
    return (pl.when(c <= 10).then(pl.lit(SIZE_LABELS[0])).when(c <= 100).then(pl.lit(SIZE_LABELS[1]))
            .when(c <= 1000).then(pl.lit(SIZE_LABELS[2])).otherwise(pl.lit(SIZE_LABELS[3])).alias("size_bucket"))


# Polymarket tag slug -> coarse group (first match in this order wins; financial tags excluded)
PM_TAG_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    ("Sports", frozenset({"sports", "nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball", "baseball",
                          "hockey", "tennis", "golf", "mma", "ufc", "boxing", "cricket", "f1", "formula-1",
                          "esports", "ncaa", "college-football", "college-basketball", "epl", "champions-league"})),
    ("Weather", frozenset({"weather", "climate", "temperature", "hurricanes", "climate-science"})),
    ("Mentions", frozenset({"mentions", "mention-markets", "say", "what-will-they-say"})),
    ("Politics", frozenset({"politics", "elections", "us-politics", "world", "geopolitics", "global-elections",
                            "trump", "congress", "us-elections", "world-elections", "middle-east", "ukraine"})),
    ("Entertainment", frozenset({"pop-culture", "culture", "entertainment", "movies", "music", "awards", "tv",
                                 "celebrities", "oscars", "grammys", "box-office", "youtube", "tweets-markets"})),
    ("Science & Tech", frozenset({"science", "tech", "ai", "space", "technology", "health", "openai", "spacex"})),
)


def coarse_category(category: str | None, series_ticker: str | None = None, title: str | None = None) -> str | None:
    """Coarse group of a Kalshi series, or None if it is financial / unknown (out of scope)."""
    if kalshi_series_is_financial(series_ticker, category, title):
        return None
    return COARSE.get(category or "", "Other")


def pm_coarse_category(tags: list[str] | None) -> str | None:
    """Coarse group of a Polymarket market from its event tag slugs, or None if financial."""
    t = {x.lower() for x in (tags or [])}
    if t & PM_FINANCIAL_TAGS:
        return None
    for group, keys in PM_TAG_GROUPS:
        if t & keys:
            return group
    return "Other"


def bucket_expr(q: str | pl.Expr = "q") -> pl.Expr:
    """Polars expression: the price-bucket label of side price ``q`` (dollars)."""
    c = (pl.col(q) if isinstance(q, str) else q) * 100
    e = pl.when(c <= EDGES_C[1] + 1e-9).then(pl.lit(BUCKET_LABELS[0]))
    for hi, lab in zip(EDGES_C[2:], BUCKET_LABELS[1:], strict=True):
        e = e.when(c <= hi + 1e-9).then(pl.lit(lab))
    return e.otherwise(pl.lit(None)).alias("bucket")


def bucket_of(q: float) -> str | None:
    c = q * 100
    for hi, lab in zip(EDGES_C[1:], BUCKET_LABELS, strict=True):
        if c <= hi + 1e-9:
            return lab
    return None


def side_price(yes_price: Any, side: Any) -> Any:
    """Price of the side bought: ``p`` for YES, ``1 - p`` for NO (numpy-friendly)."""
    p = np.asarray(yes_price, dtype=float)
    return np.where(np.asarray(side) == "yes", p, 1 - p)


def side_won(yes_result: Any, side: Any) -> Any:
    y = np.asarray(yes_result, dtype=float)
    return np.where(np.asarray(side) == "yes", y, 1 - y)


def fee_pc(q: Any, *, maker: bool = False, mult: Any = 1.0, maker_fee_series: Any = True) -> Any:
    """Exact Kalshi fee per contract at side price ``q`` (no per-order rounding)."""
    q = np.asarray(q, dtype=float)
    rate = MAKER_RATE if maker else TAKER_RATE
    f = rate * np.asarray(mult, dtype=float) * q * (1 - q)
    return np.where(np.asarray(maker_fee_series), f, 0.0) if maker else f


def returns(won: Any, q: Any, fee: Any = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(PnL per contract, return per dollar) of buying a side at ``q`` with fee ``fee`` per contract."""
    won, q, fee = (np.asarray(x, dtype=float) for x in (won, q, fee))
    pnl = won - q - fee
    return pnl, pnl / (q + fee)


def clustered(x: Any, g: Any, w: Any | None = None) -> dict[str, Any]:
    """Mean (optionally weighted), cluster-robust SE and t by ``g``, number of obs and clusters."""
    x = np.asarray(x, dtype=float)
    g = np.asarray(g)
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    wt = np.ones_like(x) if w is None else np.asarray(w, dtype=float)[ok]
    n = len(x)
    if n == 0 or wt.sum() <= 0:
        return {"n": 0, "clusters": 0, "mean": None, "se": None, "t": None}
    mu = float((wt * x).sum() / wt.sum())
    _, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=wt * (x - mu))
    k = len(s)
    se = math.sqrt((s ** 2).sum() * k / (k - 1)) / wt.sum() if k > 1 else None
    return {"n": n, "clusters": k, "mean": mu, "se": se, "t": (mu / se) if se else None}
