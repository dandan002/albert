"""Phase 6: account-level safety for real Kalshi orders. Every bot has its own caps and kill file;
this module adds what sits above them, shared by every bot on the account.

* **Global kill switch** ``data/live/KILL`` (``albert live kill``, or ``touch``). While it exists the
  order client refuses every new order (``KalshiTrader(switches=...)``), each prod bot cancels its
  own orders and exits, and the guard sweeps every ``alb*`` order each pass. Only a person removes
  it (``albert live resume``).
* **Daily-loss halt** ``data/live/HALT.json``, written by the guard with ``until`` = the next 00:00
  UTC. New orders are refused until then; the mentions bot closes its open events and keeps
  watching fills; a maker stops. It lapses by itself.
* **Audit log** ``data/live/audit/<env>/`` (append-only):
    - ``orders-YYYY-MM-DD.jsonl``: every write request any process sends through ``KalshiTrader``
      (method, path, params, body, HTTP status, response or error, latency, actor, pid), and every
      order the client refused before sending. Headers (credentials, signatures) are never logged.
    - ``fills.jsonl``: every account fill from ``GET /portfolio/fills`` as the exchange returned it,
      deduplicated by trade_id: the exchange's record, not the bots' own ledgers.
    - ``order_tags.jsonl``: order id -> client_order_id (REST fills carry no client_order_id), from
      ``GET /portfolio/orders/{id}``. The prefix names the bot (``albm`` mentions, ``albi`` / ``albl``
      makers); anything else is the user's manual trading and is left out of the bot book.
* **Guard** (``albert live guard``; systemd ``albert-guard@prod``), every ``poll_s``:
    1. syncs fills and their order tags;
    2. marks the bot book, every ``alb*`` fill: a settled market at its settlement value, an open
       one at the mid (two-sided book) or else the last price, else at cost;
    3. day PnL = bot-book equity now minus equity at the guard's first pass of the UTC day;
       below ``-max_day_loss`` -> HALT until the next 00:00 UTC and cancel every bot order;
    4. at risk = resting bot orders' collateral + cost of bot fills in markets not yet settled;
       above ``max_at_risk`` -> KILL (each bot caps itself below this, so it means a bug) and cancel
       every bot order;
    5. writes ``data/live/guard/<env>/status.json`` each pass and a ledger record on every change.
  The guard's client is cancel-only: it cannot create an order. Bots keep their own limits; the
  guard is the backstop across them.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from albert.venues.kalshi_trade import TradeError, parse_fill

BOT_PREFIX = "alb"
KILL_NAME, HALT_NAME = "KILL", "HALT.json"
EPS = 1e-9


def _now() -> datetime:
    return datetime.now(UTC)


def next_midnight(now: datetime) -> datetime:
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def tag_of(client_order_id: str | None) -> str | None:
    """The bot prefix (``alb`` + one letter) of a client_order_id; None for manual orders."""
    coid = client_order_id or ""
    return coid[:4] if coid.startswith(BOT_PREFIX) and len(coid) >= 4 else None


# ---- switches ----------------------------------------------------------------------------------
class Switches:
    """The global kill file and the guard's daily-loss halt, under ``root`` (``data/live``)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.kill_path, self.halt_path = root / KILL_NAME, root / HALT_NAME

    def killed(self) -> str | None:
        if not self.kill_path.exists():
            return None
        try:
            d = json.loads(self.kill_path.read_text() or "{}")
        except (OSError, ValueError):
            return "kill file"
        return (d.get("reason") if isinstance(d, dict) else None) or "kill file"

    def halted(self, now: datetime | None = None) -> str | None:
        if not self.halt_path.exists():
            return None
        try:
            d = json.loads(self.halt_path.read_text())
            until = datetime.fromisoformat(d["until"])
        except (OSError, ValueError, KeyError, TypeError):
            return "halt file (unreadable)"          # fail closed
        if (now or _now()) >= until:
            return None
        return f"{d.get('reason') or 'halt'} (until {until.isoformat()})"

    def blocked(self, now: datetime | None = None) -> str | None:
        k = self.killed()
        return f"kill: {k}" if k else (f"halt: {h}" if (h := self.halted(now)) else None)

    def kill(self, reason: str, now: datetime | None = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.kill_path.write_text(json.dumps({"reason": reason, "set_at": (now or _now()).isoformat()}) + "\n")

    def halt(self, reason: str, until: datetime, now: datetime | None = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.halt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"reason": reason, "until": until.isoformat(),
                                   "set_at": (now or _now()).isoformat()}) + "\n")
        tmp.replace(self.halt_path)

    def resume(self, *, halt: bool = False) -> list[str]:
        gone = []
        for p in [self.kill_path, *([self.halt_path] if halt else [])]:
            if p.exists():
                p.unlink()
                gone.append(p.name)
        return gone


# ---- audit log ---------------------------------------------------------------------------------
class Audit:
    """Append-only JSON lines under ``data/live/audit/<env>``. One ``write`` call is one line written
    with a single ``write()``, so several processes can append to the same day file."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def append(self, name: str, rec: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / name).open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def order(self, rec: dict[str, Any]) -> None:
        ts = datetime.fromisoformat(rec["ts"]) if "ts" in rec else _now()
        self.append(f"orders-{ts:%Y-%m-%d}.jsonl", rec)

    def read(self, name: str) -> list[dict[str, Any]]:
        path = self.root / name
        if not path.exists():
            return []
        out = []
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue                      # a torn last line after a crash
        return out


# ---- marks -------------------------------------------------------------------------------------
def yes_value(m: dict[str, Any]) -> tuple[float | None, bool]:
    """(YES value per contract, final?) from a ``GET /markets`` row. Final once the market has a
    result; otherwise the mid of a two-sided book, else the last price, else None (value at cost)."""
    result = m.get("result") or ""
    if result in ("yes", "no"):
        return (1.0 if result == "yes" else 0.0), True
    if (result or m.get("status") in ("finalized", "settled")) and m.get("settlement_value_dollars") not in (None, ""):
        return float(m["settlement_value_dollars"]), True        # scalar (result "scalar")
    bid = float(m.get("yes_bid_dollars") or 0)
    ask = float(m.get("yes_ask_dollars") or 0)
    if bid > 0 and 0 < ask < 1:
        return (bid + ask) / 2, False
    last = float(m.get("last_price_dollars") or 0)
    return (last if last > 0 else None), False


@dataclass(frozen=True)
class GuardLimits:
    max_day_loss: float = 15.0     # $ bot-book loss since 00:00 UTC -> HALT to the next 00:00 UTC
    max_at_risk: float = 60.0      # $ resting bot collateral + unsettled bot fill cost -> KILL


class Guard:
    def __init__(self, env: str, trader, root: Path, *, limits: GuardLimits,
                 clock: Callable[[], datetime] | None = None, log: Callable[[str], None] = print,
                 lookup_pause_s: float = 0.05) -> None:
        self.env, self.trader, self.limits = env, trader, limits
        self.clock = clock or _now
        self.log = log
        self.switches = Switches(root)
        self.audit = Audit(root / "audit" / env)
        self.dir = root / "guard" / env
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lookup_pause_s = lookup_pause_s
        self.fills: dict[str, dict[str, Any]] = {}
        for r in self.audit.read("fills.jsonl"):
            f = r.get("fill") or {}
            if f.get("trade_id"):
                self.fills[f["trade_id"]] = f
        self.tags: dict[str, str | None] = {r["order_id"]: r.get("client_order_id")
                                            for r in self.audit.read("order_tags.jsonl") if r.get("order_id")}
        self.marks: dict[str, tuple[float | None, bool]] = {}
        self.state: dict[str, Any] = self._load_state()
        self.last: dict[str, Any] = {}
        self._last_ledger: datetime | None = None

    # -- state --
    def _load_state(self) -> dict[str, Any]:
        p = self.dir / "state.json"
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self, name: str, data: dict[str, Any]) -> None:
        tmp = self.dir / f".{name}.tmp"
        tmp.write_text(json.dumps(data, default=str, indent=1) + "\n")
        tmp.replace(self.dir / name)

    def write(self, kind: str, data: dict[str, Any]) -> None:
        with (self.dir / "ledger.jsonl").open("a") as fh:
            fh.write(json.dumps({"ts": self.clock().isoformat(), "kind": kind, **data}, default=str) + "\n")

    # -- exchange sync --
    async def sync_fills(self) -> int:
        """Append fills not seen before (all account fills; overlap one hour on min_ts)."""
        params: dict[str, Any] = {"limit": 200}
        if self.state.get("max_ts"):
            params["min_ts"] = int(self.state["max_ts"]) - 3600
        got: list[dict[str, Any]] = []
        while True:
            page = await self.trader.get("/portfolio/fills", params)
            got += page.get("fills") or []
            if not page.get("cursor"):
                break
            params = {**params, "cursor": page["cursor"]}
        new = [f for f in got if (f.get("trade_id") or f.get("fill_id")) not in self.fills]
        seen = self.clock().isoformat()
        for f in sorted(new, key=lambda f: (f.get("ts") or 0, f.get("trade_id") or "")):
            tid = f.get("trade_id") or f.get("fill_id")
            self.fills[tid] = f
            self.audit.append("fills.jsonl", {"seen": seen, "fill": f})
        if got:
            self.state["max_ts"] = max(int(self.state.get("max_ts") or 0), *(int(f.get("ts") or 0) for f in got))
        return len(new)

    async def tag_orders(self) -> int:
        """Look up the client_order_id of every filled order not tagged yet (one GET per order)."""
        todo = sorted({f["order_id"] for f in self.fills.values() if f.get("order_id") not in self.tags})
        for oid in todo:
            try:
                o = (await self.trader.get(f"/portfolio/orders/{oid}")).get("order") or {}
                rec = {"order_id": oid, "client_order_id": o.get("client_order_id") or "",
                       "ticker": o.get("ticker"), "created_time": o.get("created_time")}
            except TradeError as err:
                if err.status != 404:
                    raise
                rec = {"order_id": oid, "client_order_id": None, "error": "404"}
            self.tags[oid] = rec["client_order_id"]
            self.audit.append("order_tags.jsonl", rec)
            if self.lookup_pause_s:
                await asyncio.sleep(self.lookup_pause_s)
        return len(todo)

    def bot_fills(self) -> list[dict[str, Any]]:
        out = []
        for f in self.fills.values():
            tag = tag_of(self.tags.get(f.get("order_id")))
            if tag:
                fr = parse_fill(f)
                out.append({"tag": tag, **asdict(fr)})
        return out

    async def refresh_marks(self, tickers: list[str]) -> None:
        todo = sorted({t for t in tickers if not self.marks.get(t, (None, False))[1]})
        for i in range(0, len(todo), 100):
            chunk = todo[i:i + 100]
            page = await self.trader.get("/markets", {"tickers": ",".join(chunk), "limit": 100})
            for m in page.get("markets") or []:
                self.marks[m["ticker"]] = yes_value(m)

    def book(self) -> dict[str, dict[str, Any]]:
        """Per bot tag: fills, cost (price x qty + fees), value, equity, unsettled cost."""
        out: dict[str, dict[str, Any]] = {}
        for f in self.bot_fills():
            b = out.setdefault(f["tag"], {"fills": 0, "cost": 0.0, "value": 0.0, "unsettled_cost": 0.0,
                                          "open_markets": set(), "unmarked": 0})
            cost = f["qty"] * f["price"] + f["fee"]
            yv, final = self.marks.get(f["ticker"], (None, False))
            if yv is None:
                value = f["qty"] * f["price"]          # no mark yet: carried at cost (fees still count)
                b["unmarked"] += 1
            else:
                value = f["qty"] * (yv if f["side"] == "yes" else 1 - yv)
            b["fills"] += 1
            b["cost"] += cost
            b["value"] += value
            if not final:
                b["unsettled_cost"] += cost
                b["open_markets"].add(f["ticker"])
        for b in out.values():
            b["equity"] = b["value"] - b["cost"]
            b["open_markets"] = len(b["open_markets"])
            for k in ("cost", "value", "equity", "unsettled_cost"):
                b[k] = round(b[k], 4)
        return out

    async def resting(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for o in await self.trader.orders("resting"):
            tag = tag_of(o.client_order_id)
            if tag:
                r = out.setdefault(tag, {"orders": 0, "collateral": 0.0})
                r["orders"] += 1
                r["collateral"] = round(r["collateral"] + o.price * o.remaining, 4)
        return out

    async def sweep(self, why: str) -> list[str]:
        if not getattr(self.trader, "writable", False):
            self.write("sweep_skipped", {"why": why, "reason": "read-only client"})
            return []
        done = await self.trader.cancel_own(BOT_PREFIX)
        self.write("sweep", {"why": why, "cancelled": done})
        if done:
            self.log(f"guard: cancelled {len(done)} bot orders ({why})")
        return done

    # -- one pass --
    async def tick(self) -> dict[str, Any]:
        now = self.clock()
        new = await self.sync_fills()
        await self.tag_orders()
        bf = self.bot_fills()
        await self.refresh_marks([f["ticker"] for f in bf])
        book = self.book()
        equity = round(sum(b["equity"] for b in book.values()), 4)
        day = now.strftime("%Y-%m-%d")
        if self.state.get("day") != day:
            self.state.update(day=day, start_equity=equity)
            self.write("day_start", {"day": day, "start_equity": equity})
        day_pnl = round(equity - float(self.state.get("start_equity", equity)), 4)
        resting = await self.resting()
        at_risk = round(sum(r["collateral"] for r in resting.values())
                        + sum(b["unsettled_cost"] for b in book.values()), 4)
        lim = self.limits
        if not self.switches.killed() and at_risk > lim.max_at_risk + EPS:
            reason = f"guard: bot at risk ${at_risk:.2f} > ${lim.max_at_risk:.2f}"
            self.switches.kill(reason, now)
            self.write("kill", {"reason": reason, "at_risk": at_risk})
            self.log(reason)
        if not self.switches.blocked(now) and day_pnl < -lim.max_day_loss - EPS:
            until = next_midnight(now)
            reason = f"guard: bot day PnL ${day_pnl:.2f} < -${lim.max_day_loss:.2f}"
            self.switches.halt(reason, until, now)
            self.write("halt", {"reason": reason, "until": until, "day_pnl": day_pnl})
            self.log(reason)
        blocked = self.switches.blocked(now)
        if blocked and resting:
            await self.sweep(blocked)
            resting = await self.resting()
        status = {"ts": now.isoformat(), "env": self.env, "day": day, "start_equity": self.state.get("start_equity"),
                  "equity": equity, "day_pnl": day_pnl, "at_risk": at_risk, "blocked": blocked,
                  "limits": asdict(lim), "book": book, "resting": resting, "new_fills": new,
                  "fills_total": len(self.fills),
                  "untagged": sum(1 for f in self.fills.values() if self.tags.get(f.get("order_id")) is None)}
        self._save("state.json", self.state)
        self._save("status.json", status)
        changed = (new or blocked != self.last.get("blocked")
                   or resting != self.last.get("resting") or day != self.last.get("day"))
        if changed or self._last_ledger is None or now - self._last_ledger >= timedelta(minutes=10):
            self.write("status", {k: v for k, v in status.items() if k not in ("ts", "limits")})
            self._last_ledger = now
        self.last = status
        return status

    async def run(self, minutes: float, poll_s: float = 60.0) -> None:
        stop_at = self.clock() + timedelta(minutes=minutes)
        self.write("start", {"env": self.env, "limits": asdict(self.limits), "pid": os.getpid(),
                             "writable": getattr(self.trader, "writable", False)})
        errors = 0
        while self.clock() < stop_at:
            try:
                await self.tick()
                errors = 0
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 - a bad pass must not stop the guard
                errors += 1
                self.write("error", {"error": repr(err)[:500], "consecutive": errors})
                self.log(f"guard error: {err!r}"[:300])
            await asyncio.sleep(poll_s)


def read_status(root: Path, env: str) -> dict[str, Any] | None:
    try:
        return json.loads((root / "guard" / env / "status.json").read_text())
    except (OSError, ValueError):
        return None
