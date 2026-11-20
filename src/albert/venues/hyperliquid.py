"""Hyperliquid HIP-4 outcome markets: public market data ONLY.

**Never trade here.** Hyperliquid's Terms of Use (section 1.6, last updated 2026-06-15) make anyone who
resides or is located in the United States a "Restricted Person" barred from the Interface, and
section 3.1.5 forbids circumventing that (VPNs, proxies, misrepresentation, "any other activity designed to
evade the restrictions"). The owner is a US resident, so this project has no Hyperliquid order
client (owner decision 2026-11-20, AGENTS.md). Reading the public info API is fine:
``api.hyperliquid.xyz`` serves no robots.txt (404) and the owner allowed it on 2026-11-20.

Outcomes (docs: HIP-4 outcome markets; API asset ids):

- Outcome ``n``, side ``s`` in {0, 1}: encoding ``10 n + s``. Spot coin ``#<enc>``, token ``+<enc>``,
  asset id ``100_000_000 + enc``. ``sideSpecs`` name the sides (side 0 is "Yes" for binary
  templates, participant A for ``sportsContestWinner``).
- The two sides share one merged book. The side-1 book and trades are the side-0 ones mirrored
  at ``1 - p`` (checked 2026-11-20 on 12 markets: same trade hashes and sizes, opposite
  aggressor), so we read side 0 only.
- A *question* groups outcomes of which exactly one settles Yes (``namedOutcomes`` plus a
  ``fallbackOutcome`` "other").
- Descriptions are ``key:value|key:value`` specs filled from validator-approved templates. Times
  are ``YYYYMMDD-HHMM`` UTC.

Non-financial universe (AGENTS.md): the sports templates and every outcome of a sports question.
Everything else seen on 2026-11-20 is financial (price binaries and touches on perps and crypto,
IPO market caps, policy-rate decisions, recurring crypto price questions) and is excluded. A
template not seen before is excluded until reviewed; :func:`unknown_templates` lists it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import polars as pl

from albert.http import CachedClient, Fetched

WS_URL = "wss://api.hyperliquid.xyz/ws"
OUTCOME_ASSET_BASE = 100_000_000

SPORTS_OUTCOME_TEMPLATES = frozenset({
    "sportsContestWinner", "sportsContestParticipant2", "sportsContestDraw2",
    "sportsTournamentParticipant",
})
SPORTS_QUESTION_TEMPLATES = frozenset({"sportsContestResult", "sportsTournamentWinner"})
# Seen on 2026-11-20 and excluded as financial. "template fallback", "Recurring Named Outcome" and
# "Recurring Fallback" are question members: they take the category of their question.
FINANCIAL_TEMPLATES = frozenset({
    "binaryPrice", "binaryPriceExternal", "priceTouch", "companyIpoFirstDayMarketCap",
    "companyIpoConfirmed", "policyRateDecrease", "policyRateIncrease", "policyRateNoChange",
    "policyRateDecision", "Recurring",
})
MEMBER_NAMES = frozenset({"template fallback", "Recurring Named Outcome", "Recurring Fallback"})

# REST weights (docs: rate limits and user limits): 1200 per minute per IP. Most info requests
# weigh 20; l2Book and allMids 2; recentTrades +1 per 20 items, candleSnapshot +1 per 60 items.
WEIGHT_BUDGET_PER_MIN = 900          # leave headroom under 1200

OUTCOME_SCHEMA = {
    "outcome": pl.Int64,
    "coin": pl.String,
    "template": pl.String,
    "deployer": pl.String,
    "description": pl.String,
    "side0": pl.String,
    "side1": pl.String,
    "quote": pl.String,
    "question": pl.Int64,
    "question_template": pl.String,
    "question_role": pl.String,
    "settled": pl.Boolean,
    "participant": pl.String,
    "participant_a": pl.String,
    "participant_b": pl.String,
    "short_a": pl.String,
    "short_b": pl.String,
    "sport": pl.String,
    "competition": pl.String,
    "contest_type": pl.String,
    "stage": pl.String,
    "season": pl.String,
    "scheduled_start": pl.Datetime("us", "UTC"),
    "resolution_deadline": pl.Datetime("us", "UTC"),
    "non_financial": pl.Boolean,
}

TRADE_SCHEMA = {
    "coin": pl.String,
    "outcome": pl.Int64,
    "tid": pl.Int64,
    "hash": pl.String,
    "time": pl.Datetime("ms", "UTC"),
    "side": pl.String,
    "px": pl.Float64,
    "sz": pl.Float64,
    "buyer": pl.String,
    "seller": pl.String,
}

CANDLE_SCHEMA = {
    "coin": pl.String,
    "outcome": pl.Int64,
    "interval": pl.String,
    "t": pl.Datetime("ms", "UTC"),
    "close_time": pl.Datetime("ms", "UTC"),
    "o": pl.Float64,
    "h": pl.Float64,
    "l": pl.Float64,
    "c": pl.Float64,
    "v": pl.Float64,
    "n": pl.Int64,
}


def coin(outcome: int, side: int = 0) -> str:
    """Spot coin of one outcome side (``#<10 n + s>``)."""
    if side not in (0, 1):
        raise ValueError("outcome side must be 0 or 1")
    return f"#{10 * outcome + side}"


def outcome_of(coin_name: str) -> tuple[int, int]:
    """``#372860`` -> (37286, 0)."""
    enc = int(coin_name.lstrip("#+"))
    return enc // 10, enc % 10


def parse_spec(description: str | None) -> dict[str, str]:
    """``key:value|key:value`` -> dict. Values may contain ``:`` (``perp:xyz:CL``)."""
    out: dict[str, str] = {}
    for part in (description or "").split("|"):
        key, sep, value = part.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def parse_time(value: str | None) -> datetime | None:
    """``20261122-1515`` (UTC) -> datetime."""
    if not value:
        return None
    return datetime.strptime(value, "%Y%m%d-%H%M").replace(tzinfo=UTC)


def template_of(name: str | None) -> str:
    return (name or "").removeprefix("template:")


def unknown_templates(meta: dict[str, Any]) -> list[str]:
    """Outcome or question templates not classified yet (excluded until reviewed)."""
    known = SPORTS_OUTCOME_TEMPLATES | SPORTS_QUESTION_TEMPLATES | FINANCIAL_TEMPLATES | MEMBER_NAMES
    names = {template_of(o.get("name")) for o in meta.get("outcomes", [])}
    names |= {template_of(q.get("name")) for q in meta.get("questions", [])}
    return sorted(n for n in names if n not in known)


def outcome_rows(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per outcome of an ``outcomeMeta`` response, joined to its question."""
    member: dict[int, tuple[dict[str, Any], str]] = {}
    for q in meta.get("questions", []):
        for o in q.get("namedOutcomes", []):
            member[o] = (q, "named")
        if q.get("fallbackOutcome") is not None:
            member[q["fallbackOutcome"]] = (q, "fallback")
    rows = []
    for o in meta.get("outcomes", []):
        n = int(o["outcome"])
        template = template_of(o.get("name"))
        q, role = member.get(n, (None, None))
        q_template = template_of(q.get("name")) if q else None
        spec = {**(parse_spec(q.get("description")) if q else {}), **parse_spec(o.get("description"))}
        sides = [s.get("name") for s in o.get("sideSpecs", [])] + [None, None]
        if q is not None:
            non_financial = q_template in SPORTS_QUESTION_TEMPLATES
        else:
            non_financial = template in SPORTS_OUTCOME_TEMPLATES
        rows.append({
            "outcome": n,
            "coin": coin(n),
            "template": template,
            "deployer": o.get("venue"),
            "description": o.get("description"),
            "side0": sides[0],
            "side1": sides[1],
            "quote": o.get("quoteToken"),
            "question": q.get("question") if q else None,
            "question_template": q_template,
            "question_role": role,
            "settled": (n in q.get("settledNamedOutcomes", [])) if q else None,
            "participant": spec.get("participant"),
            "participant_a": spec.get("participantA"),
            "participant_b": spec.get("participantB"),
            "short_a": spec.get("shortNameA"),
            "short_b": spec.get("shortNameB"),
            "sport": spec.get("sport"),
            "competition": spec.get("competition"),
            "contest_type": spec.get("contestType"),
            "stage": spec.get("stage"),
            "season": spec.get("season"),
            "scheduled_start": parse_time(spec.get("scheduledStart")),
            "resolution_deadline": parse_time(spec.get("resolutionDeadline")),
            "non_financial": non_financial,
        })
    return rows


def outcome_frame(meta: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(outcome_rows(meta), schema=OUTCOME_SCHEMA)


def non_financial_coins(meta: dict[str, Any]) -> list[str]:
    """Side-0 coins of every non-financial outcome, soonest scheduled start first (undated
    futures last), so a subscription cap drops the far-off ones."""
    far = datetime.max.replace(tzinfo=UTC)
    rows = [r for r in outcome_rows(meta) if r["non_financial"] and not r["settled"]]
    rows.sort(key=lambda r: (r["scheduled_start"] or far, r["outcome"]))
    return [r["coin"] for r in rows]


def trade_rows(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``recentTrades`` items or a WebSocket ``trades`` message's data. ``users`` is
    [buyer, seller]; ``side`` is the aggressor (B = buy)."""
    out = []
    for t in payload:
        users = list(t.get("users") or []) + [None, None]
        out.append({
            "coin": t["coin"], "outcome": outcome_of(t["coin"])[0] if t["coin"].startswith("#") else None,
            "tid": int(t["tid"]), "hash": t.get("hash"),
            "time": datetime.fromtimestamp(t["time"] / 1000, UTC), "side": t.get("side"),
            "px": float(t["px"]), "sz": float(t["sz"]), "buyer": users[0], "seller": users[1],
        })
    return out


def candle_rows(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "coin": k["s"], "outcome": outcome_of(k["s"])[0] if k["s"].startswith("#") else None,
        "interval": k["i"], "t": datetime.fromtimestamp(k["t"] / 1000, UTC),
        "close_time": datetime.fromtimestamp(k["T"] / 1000, UTC),
        "o": float(k["o"]), "h": float(k["h"]), "l": float(k["l"]), "c": float(k["c"]),
        "v": float(k["v"]), "n": int(k["n"]),
    } for k in payload]


def book_levels(payload: dict[str, Any]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """``l2Book`` (REST, or a WebSocket message's data) -> (bids best first, asks best first)."""
    bids, asks = payload["levels"]
    return ([(float(x["px"]), float(x["sz"])) for x in bids],
            [(float(x["px"]), float(x["sz"])) for x in asks])


class Hyperliquid:
    """Info API client. Live calls pause in proportion to their documented weight, so a long
    ingest stays under the per-IP budget that the recorder's hourly ``outcomeMeta`` shares."""

    def __init__(self, client: CachedClient, *, budget_per_min: float = WEIGHT_BUDGET_PER_MIN,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.client = client
        self.budget = budget_per_min
        self._sleep = sleep

    def info(self, body: dict[str, Any], *, cache: bool = False, base_weight: int = 20,
             per_items: int | None = None) -> Fetched:
        f = self.client.post("/info", body, cache=cache)
        if not f.from_cache:
            n = len(f.payload) if isinstance(f.payload, list) else 0
            self._sleep((base_weight + (n // per_items if per_items else 0)) * 60.0 / self.budget)
        return f

    def outcome_meta(self) -> Fetched:
        return self.info({"type": "outcomeMeta"})

    def l2_book(self, coin_name: str) -> Fetched:
        return self.info({"type": "l2Book", "coin": coin_name}, base_weight=2)

    def recent_trades(self, coin_name: str) -> Fetched:
        return self.info({"type": "recentTrades", "coin": coin_name}, per_items=20)

    def candles(self, coin_name: str, interval: str, start: datetime, end: datetime) -> Fetched:
        """At most the 5000 most recent candles exist per (coin, interval)."""
        return self.info({"type": "candleSnapshot", "req": {
            "coin": coin_name, "interval": interval,
            "startTime": int(start.timestamp() * 1000), "endTime": int(end.timestamp() * 1000)}},
            per_items=60)
