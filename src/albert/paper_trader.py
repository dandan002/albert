"""Paper version of ``KalshiTrader`` for the live bots: real reads, simulated writes.

Market-data reads (``/milestones``, ``/markets``, books, ``/series``) go to the real exchange through a
read-only ``KalshiTrader`` (``allow_prod=False``, so no write can ever leave this process). Orders,
cancels and order groups are kept here; ``/portfolio/*`` reads are answered from that state, in the
exchange's own shapes, so a bot (``live_mentions.MentionsBot``) runs unchanged.

Fill rule (the research harness's, ``research.forward.first_fills``): a resting bid fills in full at
its own price at the first public print after it was posted, and before it ended (cancel, group
trigger, expiry), that trades strictly THROUGH it: for a NO bid at p, a YES taker paying more than
1 - p; for a YES bid at p, a NO taker selling YES below p. The queue at our own price is never
assumed to clear, so this is conservative on fill count. Prints are polled from
``GET /markets/trades`` (``min_ts`` = last poll) whenever the bot reads ``/portfolio/fills``. Fees are
the series maker fee (0.0175 x multiplier x P(1-P), rounded up to $0.0001, ``*maker_fee*`` series only).
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from albert.venues.kalshi_trade import OrderState, TradeError, fmt_count, fmt_price, parse_order

PAPER_ENV = "paper"


def maker_fee(price: float, qty: float, fee_type: str | None, mult: float) -> float:
    if not fee_type or "maker_fee" not in fee_type:
        return 0.0
    return math.ceil(0.0175 * mult * qty * price * (1 - price) * 1e4 - 1e-9) / 1e4


def through(side: str, price: float, yes_price: float, taker_side: str) -> bool:
    """Does this print trade strictly through a ``side`` bid at ``price``?"""
    if side == "no":
        return taker_side == "yes" and yes_price > 1 - price + 1e-9
    return taker_side == "no" and yes_price < price - 1e-9


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


class PaperTrader:
    writable = True          # writes are simulated; nothing is sent
    env = PAPER_ENV

    def __init__(self, reader, *, clock: Callable[[], datetime] | None = None) -> None:
        if getattr(reader, "writable", False):
            raise ValueError("PaperTrader needs a read-only trader (allow_prod=False)")
        self.reader = reader
        self.clock = clock or (lambda: datetime.now(UTC))
        self.orders_: dict[str, dict[str, Any]] = {}     # order_id -> order (exchange shape + internals)
        self.groups: dict[str, set[str]] = {}
        self.fills_: list[dict[str, Any]] = []
        self.fees: dict[str, tuple[str | None, float]] = {}
        self.polled: dict[str, float] = {}               # ticker -> last trades poll (unix s)
        self.seen: set[str] = set()                      # trade ids already applied

    async def aclose(self) -> None:
        await self.reader.aclose()

    def _now(self) -> float:
        return self.clock().timestamp()

    # -- reads --
    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if path == "/portfolio/fills":
            await self.match()
            return {"fills": list(reversed(self.fills_)), "cursor": ""}
        if path.startswith("/portfolio"):
            raise TradeError(400, f"paper trader does not serve {path}", f"GET {path}")
        return await self.reader.get(path, params)

    async def shards(self, tickers: list[str]) -> dict[str, int]:
        return await self.reader.shards(tickers)

    async def orders(self, status: str = "resting", ticker: str | None = None) -> list[OrderState]:
        self._expire()
        return [parse_order(o) for o in self.orders_.values()
                if o["status"] == status and (ticker is None or o["ticker"] == ticker)]

    async def positions(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for f in self.fills_:
            q = float(f["count_fp"]) * (1 if f["outcome_side"] == "yes" else -1)
            out[f["market_ticker"]] = out.get(f["market_ticker"], 0.0) + q
        return {t: q for t, q in out.items() if abs(q) > 1e-9}

    # -- writes (simulated) --
    async def create(self, body: dict[str, Any]) -> dict[str, Any]:
        oid = str(uuid.uuid4())
        side = "yes" if body["side"] == "bid" else "no"
        yes_price = float(body["price"])
        count = float(body["count"])
        now = self._now()
        group = body.get("order_group_id")
        if group is not None and group not in self.groups:
            raise TradeError(404, "order group not found", f"create {body.get('client_order_id')}")
        o = {"order_id": oid, "client_order_id": body.get("client_order_id") or "", "ticker": body["ticker"],
             "outcome_side": side, "yes_price_dollars": fmt_price(yes_price), "status": "resting",
             "initial_count_fp": fmt_count(count), "fill_count_fp": "0.00", "remaining_count_fp": fmt_count(count),
             "maker_fees_dollars": "0.0000", "order_group_id": group,
             "_side": side, "_price": yes_price if side == "yes" else round(1 - yes_price, 6), "_count": count,
             "_t0": now, "_end": float(body["expiration_time"]) if body.get("expiration_time") else math.inf}
        self.orders_[oid] = o
        if group is not None:
            self.groups[group].add(oid)
        self.polled.setdefault(o["ticker"], now)
        return {"order_id": oid, "client_order_id": o["client_order_id"], "fill_count": "0.00",
                "remaining_count": fmt_count(count), "ts_ms": int(now * 1000)}

    def _end(self, o: dict[str, Any], status: str) -> None:
        if o["status"] == "resting":
            o["status"] = status
            o["_end"] = min(o["_end"], self._now())

    def _expire(self) -> None:
        now = self._now()
        for o in self.orders_.values():
            if o["status"] == "resting" and o["_end"] <= now:
                o["status"] = "canceled"

    async def cancel(self, order_id: str, ticker: str, exchange_index: int | None = None) -> dict[str, Any]:
        o = self.orders_.get(order_id)
        if o is None or o["status"] != "resting":
            raise TradeError(404, "order not found", f"cancel {order_id}")
        self._end(o, "canceled")
        return {"order_id": order_id, "client_order_id": o["client_order_id"], "reduced_by": o["remaining_count_fp"],
                "ts_ms": int(self._now() * 1000)}

    async def cancel_own(self, prefix: str) -> list[str]:
        done = []
        for o in await self.orders("resting"):
            if o.client_order_id.startswith(prefix):
                await self.cancel(o.order_id, o.ticker)
                done.append(o.order_id)
        return done

    async def create_order_group(self, contracts_limit: int) -> str:
        gid = str(uuid.uuid4())
        self.groups[gid] = set()
        return gid

    async def trigger_order_group(self, group_id: str) -> Any:
        for oid in self.groups.get(group_id, ()):
            self._end(self.orders_[oid], "canceled")
        return {}

    async def delete_order_group(self, group_id: str) -> None:
        await self.trigger_order_group(group_id)
        self.groups.pop(group_id, None)

    # -- fills --
    async def _fee(self, ticker: str) -> tuple[str | None, float]:
        series = ticker.split("-")[0]
        if series not in self.fees:
            info = ((await self.reader.get(f"/series/{series}")) or {}).get("series") or {}
            self.fees[series] = (info.get("fee_type"), float(info.get("fee_multiplier") or 1.0))
        return self.fees[series]

    async def match(self) -> list[dict[str, Any]]:
        """Poll new public prints for every market with an order that could still fill, and fill
        orders the prints trade through. Returns the new fills."""
        now = self._now()
        live = [o for o in self.orders_.values() if float(o["remaining_count_fp"]) > 0
                and (o["status"] == "resting" or o["_end"] > self.polled.get(o["ticker"], 0.0))]
        new: list[dict[str, Any]] = []
        for ticker in sorted({o["ticker"] for o in live}):
            since = self.polled.get(ticker, now)
            prints, cursor = [], None
            while True:
                params: dict[str, Any] = {"ticker": ticker, "limit": 1000, "min_ts": int(since) - 1}
                if cursor:
                    params["cursor"] = cursor
                page = await self.reader.get("/markets/trades", params)
                prints += page.get("trades") or []
                cursor = page.get("cursor")
                if not cursor:
                    break
            self.polled[ticker] = now
            prints.sort(key=lambda t: t["created_time"])
            for t in prints:
                if t["trade_id"] in self.seen:
                    continue
                self.seen.add(t["trade_id"])
                ts = _ts(t["created_time"])
                yes = float(t["yes_price_dollars"])
                for o in live:
                    if (o["ticker"] != ticker or float(o["remaining_count_fp"]) <= 0
                            or not (o["_t0"] < ts <= o["_end"]) or not through(o["_side"], o["_price"], yes, t["taker_side"])):
                        continue
                    new.append(await self._fill(o, t, ts))
        self._expire()
        return new

    async def _fill(self, o: dict[str, Any], t: dict[str, Any], ts: float) -> dict[str, Any]:
        qty = float(o["remaining_count_fp"])
        fee = maker_fee(o["_price"], qty, *(await self._fee(o["ticker"])))
        o["fill_count_fp"] = fmt_count(float(o["fill_count_fp"]) + qty)
        o["remaining_count_fp"] = "0.00"
        o["maker_fees_dollars"] = f"{float(o['maker_fees_dollars']) + fee:.4f}"
        o["status"] = "executed"
        f = {"trade_id": f"paper-{t['trade_id']}-{o['order_id'][:8]}", "order_id": o["order_id"],
             "client_order_id": o["client_order_id"], "market_ticker": o["ticker"], "outcome_side": o["_side"],
             "yes_price_dollars": o["yes_price_dollars"], "count_fp": fmt_count(qty), "fee_cost": f"{fee:.4f}",
             "is_taker": False, "ts_ms": int(ts * 1000), "print_trade_id": t["trade_id"],
             "print_yes_price": t["yes_price_dollars"]}
        self.fills_.append(f)
        return f


def ledger_report(s, ledgers: list, *, transport=None):
    """Orders and fills from bot ledgers (``order`` / ``fill`` records), each fill held to the
    settled result (``research.forward.fetch_results``). One row per order: ticker, event, side,
    price, filled, fee, y (null until settled) and pnl ($, null until settled)."""
    import json

    import polars as pl

    from albert.research import forward as F

    orders: dict[str, dict[str, Any]] = {}
    for path in ledgers:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if r.get("kind") == "order":
                orders[r["order_id"]] = {"run": path.parent.name, "ts": r["ts"], "event": r["event"], "ticker": r["ticker"],
                                         "side": r["side"], "price": float(r["price"]), "filled": False, "fee": 0.0}
            elif r.get("kind") == "fill" and r.get("order_id") in orders:
                orders[r["order_id"]].update(filled=True, fee=float(r.get("fee") or 0.0))
    schema = {"run": pl.String, "ts": pl.String, "event": pl.String, "ticker": pl.String, "side": pl.String,
              "price": pl.Float64, "filled": pl.Boolean, "fee": pl.Float64}
    df = pl.DataFrame(list(orders.values()), schema=schema)
    if not df.height:
        return df
    res = F.fetch_results(s, df["ticker"].unique().to_list(), transport=transport).select("ticker", "y")
    df = df.join(res, on="ticker", how="left")
    won = pl.when(pl.col("side") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y")).cast(pl.Float64)
    pnl = pl.when(~pl.col("filled")).then(0.0).otherwise(won - pl.col("price") - pl.col("fee"))
    return df.with_columns(pl.when(pl.col("y").is_null() & pl.col("filled")).then(None).otherwise(pnl).alias("pnl"))
