"""Kalshi order client (V2 order endpoints), scoped to one environment: ``demo`` or ``prod``.

Safety rules enforced here, not left to callers:

* **Environment-scoped keys.** demo reads ``KALSHI_DEMO_API_KEY_ID`` / ``KALSHI_DEMO_PRIVATE_KEY``;
  prod reads ``KALSHI_API_KEY_ID`` / ``KALSHI_PRIVATE_KEY``. Kalshi does not share keys between
  environments, so a demo run can never touch the real account.
* **Prod is gated.** Writing to prod needs ``KALSHI_LIVE_ENABLED=1`` in ``.env`` *and*
  ``allow_prod=True`` from the caller (a CLI flag on every run). Reads are always allowed.
* **Write allowlist.** Only single-order create/cancel and order-group endpoints can be written.
  Cancel All Orders and batch cancel are unreachable: the account also holds the user's manual
  orders, and the bot must never cancel or trade anything it did not create.
* **Own orders only.** Every order carries ``client_order_id`` = ``<prefix><run>-<n>``;
  ``cancel_own`` lists resting orders and cancels only those with the bot's prefix.
* **Write budget.** A client-side token bucket mirrors the account's Write bucket (Basic tier:
  100 tokens/s, capacity 100; create costs 10, cancel 2; ``GET /account/endpoint_costs``), and a
  429 backs off and retries.

Direction (docs: Order direction): V2 orders quote the YES leg. A bid for YES at ``p`` is
``side=bid, price=p``; a bid for NO at ``q`` is ``side=ask, price=1-q``. Responses carry
``outcome_side`` (yes|no) and ``yes_price_dollars``; positions are net YES (``position_fp``).
"""

import asyncio
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

from albert.config import secret
from albert.venues.kalshi_auth import KalshiSigner, load_private_key

ENVS = {
    "prod": {"rest": "https://api.elections.kalshi.com/trade-api/v2",
             "ws": "wss://api.elections.kalshi.com/trade-api/ws/v2",
             "key_id": "KALSHI_API_KEY_ID", "key": "KALSHI_PRIVATE_KEY"},
    "demo": {"rest": "https://demo-api.kalshi.co/trade-api/v2",
             "ws": "wss://demo-api.kalshi.co/trade-api/ws/v2",
             "key_id": "KALSHI_DEMO_API_KEY_ID", "key": "KALSHI_DEMO_PRIVATE_KEY"},
}
API_PREFIX = "/trade-api/v2"
WS_PATH = "/trade-api/ws/v2"
LIVE_FLAG = "KALSHI_LIVE_ENABLED"

# The only write requests this client will send (method, path regex).
WRITE_ALLOW = [
    ("POST", re.compile(r"^/portfolio/events/orders$")),
    ("DELETE", re.compile(r"^/portfolio/events/orders/[0-9a-fA-F-]{8,}$")),
    ("POST", re.compile(r"^/portfolio/order_groups/create$")),
    ("DELETE", re.compile(r"^/portfolio/order_groups/[0-9a-fA-F-]{8,}$")),
    ("PUT", re.compile(r"^/portfolio/order_groups/[0-9a-fA-F-]{8,}/(reset|trigger)$")),
]
WRITE_COST = {"POST": 10.0, "PUT": 10.0, "DELETE": 2.0}   # tokens (DELETE order: 2)


class TradeError(RuntimeError):
    def __init__(self, status: int, body: Any, what: str) -> None:
        super().__init__(f"{what}: HTTP {status}: {str(body)[:300]}")
        self.status, self.body = status, body


class NotAllowed(RuntimeError):
    pass


def signer_for(env: str) -> KalshiSigner:
    e = ENVS[env]
    key_id, pem = secret(e["key_id"]), secret(e["key"])
    if not key_id or not pem:
        raise RuntimeError(f"{e['key_id']} and {e['key']} must be set in .env for the {env} environment")
    return KalshiSigner(key_id, load_private_key(pem))


def check_writable(env: str, allow_prod: bool) -> None:
    if env not in ENVS:
        raise NotAllowed(f"unknown environment {env!r}")
    if env == "prod" and not (allow_prod and secret(LIVE_FLAG) == "1"):
        raise NotAllowed(f"prod orders need {LIVE_FLAG}=1 in .env and an explicit --live flag")


def _write_allowed(method: str, path: str) -> bool:
    return any(m == method and rx.match(path) for m, rx in WRITE_ALLOW)


# ---- request bodies and parsers ---------------------------------------------------------------
def fmt_price(p: float) -> str:
    return f"{round(p, 4):.4f}"


def fmt_count(q: float) -> str:
    return f"{round(q, 2):.2f}"


def order_body(ticker: str, side: str, price: float, count: float, *, client_order_id: str,
               order_group_id: str | None = None, expiration_ts: int | None = None,
               subaccount: int | None = None, post_only: bool = True) -> dict[str, Any]:
    """A resting maker bid on ``side`` (yes|no) at ``price`` (that side's price)."""
    if side not in ("yes", "no"):
        raise ValueError(f"side must be yes or no, not {side!r}")
    if not 0 < price < 1:
        raise ValueError(f"price {price} outside (0, 1)")
    body: dict[str, Any] = {
        "ticker": ticker, "client_order_id": client_order_id,
        "side": "bid" if side == "yes" else "ask",
        "price": fmt_price(price if side == "yes" else 1 - price),
        "count": fmt_count(count), "time_in_force": "good_till_canceled",
        "self_trade_prevention_type": "taker_at_cross", "post_only": post_only,
        "cancel_order_on_pause": True, "exchange_index": -1,
    }
    if expiration_ts is not None:
        body["expiration_time"] = int(expiration_ts)
    if order_group_id:
        body["order_group_id"] = order_group_id
    if subaccount is not None:
        body["subaccount"] = subaccount
    return body


def _f(x: Any) -> float:
    return float(x) if x not in (None, "") else 0.0


def _outcome(node: dict[str, Any]) -> str:
    side = node.get("outcome_side")
    if side in ("yes", "no"):
        return side
    book = node.get("book_side")
    if book in ("bid", "ask"):
        return "yes" if book == "bid" else "no"
    raise ValueError(f"order/fill without a direction: {sorted(node)}")


@dataclass
class OrderState:
    order_id: str
    client_order_id: str
    ticker: str
    side: str              # yes | no: the outcome we are buying
    price: float           # in that side's terms
    status: str            # resting | canceled | executed | ...
    initial: float
    filled: float
    remaining: float
    maker_fees: float
    order_group_id: str | None


def parse_order(o: dict[str, Any]) -> OrderState:
    """An order from ``GET /portfolio/orders`` or a ``user_order`` WebSocket message."""
    side = _outcome(o)
    yes_price = _f(o.get("yes_price_dollars"))
    return OrderState(
        order_id=o["order_id"], client_order_id=o.get("client_order_id") or "",
        ticker=o.get("ticker") or o.get("market_ticker"), side=side,
        price=round(yes_price if side == "yes" else 1 - yes_price, 6), status=o.get("status", ""),
        initial=_f(o.get("initial_count_fp")), filled=_f(o.get("fill_count_fp")),
        remaining=_f(o.get("remaining_count_fp")), maker_fees=_f(o.get("maker_fees_dollars")),
        order_group_id=o.get("order_group_id") or None)


@dataclass
class FillRow:
    trade_id: str
    order_id: str
    client_order_id: str
    ticker: str
    side: str              # outcome bought
    price: float           # in that side's terms
    qty: float
    fee: float
    is_taker: bool
    ts_ms: int | None
    post_position: float | None   # net YES after the fill (WebSocket only)


def parse_fill(f: dict[str, Any]) -> FillRow:
    """A fill from ``GET /portfolio/fills`` or the ``fill`` WebSocket channel."""
    side = _outcome(f)
    yes_price = _f(f.get("yes_price_dollars"))
    ts_ms = f.get("ts_ms")
    if ts_ms is None and f.get("ts") is not None:
        ts_ms = int(f["ts"]) * 1000
    post = f.get("post_position_fp")
    return FillRow(
        trade_id=f.get("trade_id") or f.get("fill_id"), order_id=f["order_id"],
        client_order_id=f.get("client_order_id") or "",
        ticker=f.get("market_ticker") or f.get("ticker"), side=side,
        price=round(yes_price if side == "yes" else 1 - yes_price, 6), qty=_f(f.get("count_fp")),
        fee=_f(f.get("fee_cost")), is_taker=bool(f.get("is_taker")), ts_ms=ts_ms,
        post_position=None if post is None else _f(post))


def positions_by_ticker(payload: dict[str, Any]) -> dict[str, float]:
    """Net YES contracts per market (non-zero only) from ``GET /portfolio/positions``."""
    out = {}
    for p in payload.get("market_positions") or []:
        q = _f(p.get("position_fp"))
        if abs(q) > 1e-9:
            out[p["ticker"]] = q
    return out


def balance_dollars(payload: dict[str, Any]) -> float:
    if payload.get("balance_dollars") is not None:
        return float(payload["balance_dollars"])
    return payload["balance"] / 100.0


# ---- clients -----------------------------------------------------------------------------------
class TokenBucket:
    def __init__(self, rate: float, capacity: float) -> None:
        self.rate, self.capacity = rate, capacity
        self.tokens, self.t = capacity, time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
        self.t = now

    def try_spend(self, cost: float) -> bool:
        self._refill()
        if self.tokens < cost:
            return False
        self.tokens -= cost
        return True

    async def spend(self, cost: float) -> None:
        while not self.try_spend(cost):
            await asyncio.sleep(max((cost - self.tokens) / self.rate, 0.005))


class KalshiTrader:
    """Async REST client for one environment. ``transport`` is for tests (httpx.MockTransport)."""

    def __init__(self, env: str, signer: KalshiSigner, *, allow_prod: bool = False,
                 subaccount: int | None = None, write_rate: float = 100.0,
                 write_capacity: float = 100.0, transport: httpx.AsyncBaseTransport | None = None,
                 timeout: float = 10.0) -> None:
        if env not in ENVS:
            raise NotAllowed(f"unknown environment {env!r}")
        self.env, self.signer, self.subaccount = env, signer, subaccount
        self.writable = True
        try:
            check_writable(env, allow_prod)
        except NotAllowed:
            self.writable = False
        self.bucket = TokenBucket(write_rate, write_capacity)
        self.http = httpx.AsyncClient(base_url=ENVS[env]["rest"], transport=transport, timeout=timeout)
        self.requests: list[tuple[str, str]] = []   # (method, path) log, for tests and audits

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _send(self, method: str, path: str, *, params: dict[str, Any] | None = None,
                    body: dict[str, Any] | None = None, what: str = "") -> Any:
        if method != "GET":
            if not self.writable:
                raise NotAllowed(f"{self.env} is read-only for this client")
            if not _write_allowed(method, path):
                raise NotAllowed(f"write not allowed: {method} {path}")
        backoff = 0.05
        for _ in range(6):
            if method != "GET":
                await self.bucket.spend(WRITE_COST[method])
            headers = self.signer.headers(method, API_PREFIX + path)
            self.requests.append((method, path))
            r = await self.http.request(method, path, params=params, json=body, headers=headers)
            if r.status_code == 429:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 2.0)
                continue
            try:
                data = r.json() if r.content else {}
            except ValueError:
                data = r.text
            if r.status_code >= 400:
                raise TradeError(r.status_code, data, what or f"{method} {path}")
            return data
        raise TradeError(429, "rate limited after retries", what or f"{method} {path}")

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return await self._send("GET", path, params=params)

    # -- reads --
    async def orders(self, status: str = "resting", ticker: str | None = None) -> list[OrderState]:
        out, cursor = [], None
        while True:
            params: dict[str, Any] = {"status": status, "limit": 200}
            if ticker:
                params["ticker"] = ticker
            if self.subaccount is not None:
                params["subaccount"] = self.subaccount
            if cursor:
                params["cursor"] = cursor
            page = await self.get("/portfolio/orders", params)
            out += [parse_order(o) for o in page.get("orders") or []]
            cursor = page.get("cursor")
            if not cursor:
                return out

    async def positions(self) -> dict[str, float]:
        out: dict[str, float] = {}
        cursor = None
        while True:
            params: dict[str, Any] = {"limit": 200}
            if self.subaccount is not None:
                params["subaccount"] = self.subaccount
            if cursor:
                params["cursor"] = cursor
            page = await self.get("/portfolio/positions", params)
            out.update(positions_by_ticker(page))
            cursor = page.get("cursor")
            if not cursor:
                return out

    async def balance(self) -> float:
        return balance_dollars(await self.get("/portfolio/balance"))

    async def queue_positions(self, tickers: list[str]) -> dict[str, float]:
        if not tickers:
            return {}
        page = await self.get("/portfolio/orders/queue_positions", {"market_tickers": ",".join(tickers)})
        return {q["order_id"]: _f(q.get("queue_position_fp")) for q in page.get("queue_positions") or []}

    # -- writes --
    async def create(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._send("POST", "/portfolio/events/orders", body=body,
                                what=f"create {body.get('client_order_id')}")

    async def cancel(self, order_id: str, ticker: str) -> dict[str, Any]:
        params: dict[str, Any] = {"market_ticker": ticker}
        if self.subaccount is not None:
            params["subaccount"] = self.subaccount
        return await self._send("DELETE", f"/portfolio/events/orders/{order_id}", params=params,
                                what=f"cancel {order_id}")

    async def cancel_own(self, prefix: str) -> list[str]:
        """Cancel every resting order whose client_order_id starts with ``prefix`` (and nothing
        else). Returns the cancelled order ids."""
        if not prefix:
            raise NotAllowed("cancel_own needs a non-empty prefix")
        done = []
        for o in await self.orders("resting"):
            if o.client_order_id.startswith(prefix):
                try:
                    await self.cancel(o.order_id, o.ticker)
                    done.append(o.order_id)
                except TradeError as e:
                    if e.status != 404:      # already gone
                        raise
        return done

    async def create_order_group(self, contracts_limit: int) -> str:
        body: dict[str, Any] = {"contracts_limit": int(contracts_limit)}
        if self.subaccount is not None:
            body["subaccount"] = self.subaccount
        r = await self._send("POST", "/portfolio/order_groups/create", body=body, what="order group")
        return r["order_group_id"]

    async def delete_order_group(self, group_id: str) -> None:
        await self._send("DELETE", f"/portfolio/order_groups/{group_id}", what="delete order group")

    async def trigger_order_group(self, group_id: str) -> Any:
        """Cancel every order in the group and block new ones until reset. Kalshi rejects a body-less
        PUT with ``invalid_content_type``, so an empty JSON object is sent."""
        return await self._send("PUT", f"/portfolio/order_groups/{group_id}/trigger", body={},
                                what="trigger order group")

    async def reset_order_group(self, group_id: str) -> Any:
        return await self._send("PUT", f"/portfolio/order_groups/{group_id}/reset", body={},
                                what="reset order group")

    # -- read-only sync access (fixtures, status commands) --
    @staticmethod
    @contextmanager
    def sync(env: str) -> Iterator["_SyncReader"]:
        r = _SyncReader(env, signer_for(env))
        try:
            yield r
        finally:
            r.http.close()


class _SyncReader:
    def __init__(self, env: str, signer: KalshiSigner) -> None:
        self.signer = signer
        self.http = httpx.Client(base_url=ENVS[env]["rest"], timeout=20)

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        r = self.http.get(path, params=params, headers=self.signer.headers("GET", API_PREFIX + path))
        if r.status_code >= 400:
            raise TradeError(r.status_code, r.text, f"GET {path}")
        return r.json()
