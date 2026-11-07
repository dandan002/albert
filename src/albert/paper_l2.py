"""Live paper maker: the replay simulator fed by the live WebSocket books (no real orders).

The strategy (``backtest.l2.Quoter``) and the fill model (``backtest.l2.Simulator``: queue
position, latency, post-only, write budget) are the same objects the backtest uses, so a paper run
differs from a replay only in *when* it runs. This module contains no order-placing code; the
Kalshi key is used only to authenticate the market-data WebSocket.

A run lives in ``data/paper_l2/<run_id>/``:

* ``stream/``: the raw frames the run traded on (same format as ``albert record``), so the run can
  be replayed with ``albert backtest l2 --root`` and the fills reconciled.
* ``ledger.jsonl``: start (config and universe), every fill, markouts at 10s/60s/300s (to the mid,
  ``mo``, and to the exit price, ``mx``: the best bid on our side), a status
  line every minute (equity marked to mid, positions, open orders, diagnostics), halts, and end.

Risk guards (the same ones a live run will need): per-market position cap (``max_pos``), a cap on
gross open contracts, a loss limit on mark-to-mid PnL for the run, and a kill switch
(``touch data/paper_l2/STOP`` or SIGTERM). A halt cancels every order and stops quoting.
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from albert.backtest.l2 import Event, OnlineClassifier, Quoter, SimConfig, Simulator, _pm_tops
from albert.stream import Recorder, disk_guard

MARKOUT_S = (10, 60, 300)


@dataclass
class Guards:
    max_gross: float = 30.0          # stop adding exposure above this many open contracts
    max_loss: float = 5.0            # halt when mark-to-mid PnL for the run falls below -$X
    status_every_s: float = 60.0


@dataclass
class PaperState:
    halted: str | None = None
    fills_logged: int = 0
    pending_markouts: list[tuple[int, int, int]] = field(default_factory=list)  # (due_ns, fill_idx, h)
    peak: float = 0.0


class PaperMaker:
    def __init__(self, run_dir: Path, signer, *, strategy: Quoter, cfg: SimConfig,
                 pairs: list[dict[str, Any]], guards: Guards, holdback_ms: float = 500.0,
                 stop_file: Path | None = None) -> None:
        self.dir = run_dir
        self.sim = Simulator(cfg)
        self.strategy = strategy
        self.guards = guards
        self.clf = OnlineClassifier(holdback_ms)
        self.state = PaperState()
        self.stop_file = stop_file
        self.by_token: dict[str, list[tuple[str, bool]]] = {}
        for p in pairs:
            self.by_token.setdefault(p["token_id"], []).append((p["kalshi_ticker"], bool(p.get("complement"))))
        self.rec = Recorder(run_dir / "stream", signer, on_frame=self.on_frame)
        self.ledger = open(run_dir / "ledger.jsonl", "a", buffering=1)  # noqa: SIM115 - closed at the end of run

    # -- input --
    def on_frame(self, venue: str, recv_ns: int, frame: str) -> None:
        if venue == "kalshi":
            d = json.loads(frame)
            kind = {"orderbook_snapshot": "snapshot", "orderbook_delta": "delta",
                    "trade": "trade"}.get(d.get("type"))
            if kind:
                self.clf.push(Event(recv_ns, kind, d["msg"]["market_ticker"], d["msg"]))
        elif venue == "pm":
            for asset, bid, ask in _pm_tops({"frame": frame}):
                for ticker, comp in self.by_token.get(asset, []):
                    b, a = ((None if ask is None else 1 - ask, None if bid is None else 1 - bid)
                            if comp else (bid, ask))
                    mid = (a + b) / 2 if a is not None and b is not None else None
                    self.clf.push(Event(recv_ns, "ext", ticker, {"bid": b, "ask": a, "mid": mid}))
        self.step(recv_ns)

    def step(self, now_ns: int) -> None:
        for ev in self.clf.release(now_ns):
            self.sim.apply(ev)
            if self.state.halted:
                continue
            if self._gross() >= self.guards.max_gross:
                self._cancel_market(ev.market)    # stop adding exposure; existing positions ride
                continue
            self.strategy.on_event(self.sim, ev.market)
        self._log_fills()

    # -- guards and accounting --
    def _gross(self) -> float:
        return sum(abs(q) for q in self.sim.position.values())

    def _cancel_market(self, market: str) -> None:
        for o in self.sim.live_orders(market):
            self.sim.cancel(o.oid)

    def last_mid(self, market: str) -> float | None:
        _, vals = self.sim.quotes.get(market, ([], []))
        for b, a in reversed(vals):
            if b is not None and a is not None:
                return (b + a) / 2
        return None

    def equity(self) -> float:
        """Run PnL with every bought contract marked to the last two-sided Kalshi mid. Computed per
        fill (YES and NO bought in one market net to $1, as Kalshi settles them), fees included."""
        held: dict[str, list[float]] = {}
        pnl = 0.0
        for f in self.sim.fills:
            h = held.setdefault(f.market, [0.0, 0.0])
            h[0 if f.side == "yes" else 1] += f.qty
            pnl -= f.qty * f.price + f.fee
        for m, (yes, no) in held.items():
            mid = self.last_mid(m)
            if mid is None:
                mid = 0.5   # never quoted two-sided: cannot happen for a fill, kept for safety
            pnl += yes * mid + no * (1 - mid)
        return pnl

    def halt(self, reason: str) -> None:
        if self.state.halted:
            return
        self.state.halted = reason
        for m in list(self.sim.by_market):
            self._cancel_market(m)
        self.write("halt", {"reason": reason, "equity": self.equity()})

    def _log_fills(self) -> None:
        fills = self.sim.fills
        for i in range(self.state.fills_logged, len(fills)):
            f = fills[i]
            self.write("fill", {**asdict(f), "quote": self.sim.markets[f.market].quote(),
                                "pm_mid": self.sim.ext_at(f.market, f.t_ns),
                                "position": self.sim.position[f.market]})
            for h in MARKOUT_S:
                self.state.pending_markouts.append((f.t_ns + h * 10**9, i, h))
        self.state.fills_logged = len(fills)
        due = [x for x in self.state.pending_markouts if x[0] <= self.sim.now]
        if due:
            self.state.pending_markouts = [x for x in self.state.pending_markouts if x[0] > self.sim.now]
            for t, i, h in due:
                f = fills[i]
                mid = self.sim.mid_at(f.market, t)
                mo = None if mid is None else (mid if f.side == "yes" else 1 - mid) - f.price
                ex = self.sim.exit_at(f.market, f.side, t)
                self.write("markout", {"fill": i, "oid": f.oid, "market": f.market, "h": h, "mo": mo,
                                       "mx": None if ex is None else ex - f.price})

    def write(self, kind: str, data: dict[str, Any]) -> None:
        self.ledger.write(json.dumps({"kind": kind, "wall": datetime.now(UTC).isoformat(),
                                      "sim_ns": self.sim.now, **data}, default=str) + "\n")

    async def ticker(self) -> None:
        """Release held-back events on quiet feeds, write status, enforce guards."""
        next_status = 0.0
        while not self.rec.stop.is_set():
            await self.rec._sleep_or_stop(0.2)
            self.step(time.time_ns())
            if self.stop_file is not None and self.stop_file.exists():
                self.halt("kill switch file")
                self.rec.stopped_reason = "kill switch"
                self.rec.stop.set()
            eq = self.equity()
            if eq < -self.guards.max_loss and not self.state.halted:
                self.halt(f"loss limit: equity {eq:.2f} < -{self.guards.max_loss}")
            if time.monotonic() >= next_status:
                next_status = time.monotonic() + self.guards.status_every_s
                self.write("status", {
                    "equity": round(eq, 4), "cash": round(self.sim.cash, 4),
                    "gross": self._gross(), "markets_held": sum(1 for q in self.sim.position.values() if abs(q) > 1e-9),
                    "open_orders": sum(len(self.sim.live_orders(m)) for m in list(self.sim.by_market)),
                    "orders": len(self.sim.orders), "fills": len(self.sim.fills),
                    "synced": len(self.sim.synced), "halted": self.state.halted,
                    "classifier": dict(self.clf.stats), "diag": dict(self.sim.diag)})

    async def run(self, *, minutes: float, kalshi: list[str], pm_tokens: list[str],
                  chunk: int = 100, min_free_gb: float = 10.0) -> None:
        tasks = [disk_guard(self.rec, min_free_gb), self.ticker()]
        for i in range(0, len(kalshi), chunk):
            tasks.append(self.rec.kalshi_conn(f"k{i // chunk}", kalshi[i:i + chunk]))
        for i in range(0, len(pm_tokens), chunk):
            tasks.append(self.rec.pm_conn(f"p{i // chunk}", pm_tokens[i:i + chunk]))

        async def timer() -> None:
            await self.rec._sleep_or_stop(minutes * 60)
            self.rec.stop.set()

        await asyncio.gather(timer(), *tasks)

    def close(self) -> dict[str, Any]:
        self.step(time.time_ns() + 10**12)   # release everything still held back
        files = self.rec.close()
        positions = {m: q for m, q in self.sim.position.items() if abs(q) > 1e-9}
        end = {"stopped_reason": self.rec.stopped_reason, "halted": self.state.halted,
               "equity": self.equity(), "cash": self.sim.cash, "fills": len(self.sim.fills),
               "orders": len(self.sim.orders), "positions": positions,
               "final_mids": {m: self.last_mid(m) for m in {f.market for f in self.sim.fills}},
               "classifier": dict(self.clf.stats), "diag": dict(self.sim.diag),
               "files": [asdict(f) for f in files],
               "conn_stats": {c: st.as_dict() for c, st in self.rec.stats.items()}}
        self.write("end", end)
        self.ledger.close()
        return end


def report(s, c, run: str | None = None) -> Path:
    """Fills from paper run ledgers, scored like ``backtest l2``: markouts, PnL marked to the last
    mid of the run, and settlement PnL where the market has resolved."""
    import math

    import polars as pl

    from albert.reports import Report

    root = s.data_dir / "paper_l2"
    runs = [root / run] if run else sorted(p for p in root.iterdir() if (p / "ledger.jsonl").exists())
    fills, marks, info = [], {}, []
    for rd in runs:
        rows = [json.loads(line) for line in (rd / "ledger.jsonl").read_text().splitlines() if line.strip()]
        end = next((r for r in rows if r["kind"] == "end"), None)
        last_status = next((r for r in reversed(rows) if r["kind"] == "status"), None)
        info.append({"run": rd.name, "ended": end is not None,
                     "stopped": end and end["stopped_reason"], "halted": end and end["halted"],
                     "equity": (end or last_status or {}).get("equity"),
                     "fills": sum(r["kind"] == "fill" for r in rows),
                     "classifier": (end or last_status or {}).get("classifier")})
        final_mids = (end or {}).get("final_mids", {})
        n = 0
        for r in rows:
            if r["kind"] == "fill":
                fills.append({"run": rd.name, "idx": n, "market": r["market"], "side": r["side"],
                              "price": r["price"], "qty": r["qty"], "fee": r["fee"], "how": r["how"],
                              "last_mid": final_mids.get(r["market"])})
                n += 1
            elif r["kind"] == "markout":
                marks[(rd.name, r["fill"], r["h"])] = (r["mo"], r.get("mx"))
    rep = Report(s.reports_dir, "paper-l2")
    rep.table("Runs", pl.DataFrame(info, infer_schema_length=None).with_columns(
        pl.col("classifier").cast(pl.String)))
    if not fills:
        rep.text("Fills", "No fills yet.")
        return rep.save()
    for f in fills:
        for h in MARKOUT_S:
            mo, mx = marks.get((f["run"], f["idx"], h), (None, None))
            f[f"mo_{h}s"], f[f"mx_{h}s"] = mo, mx
    results = {}
    for t in {f["market"] for f in fills}:
        m = c.kalshi.client.get(f"/markets/{t}", cache=False).payload.get("market", {})
        results[t] = 1.0 if m.get("result") == "yes" else 0.0 if m.get("result") == "no" else None
    for f in fills:
        settle = results.get(f["market"])
        final = settle if settle is not None else f["last_mid"]
        f["settled"] = settle is not None
        f["pnl_per_contract"] = (None if final is None else
                                 (final if f["side"] == "yes" else 1 - final) - f["price"] - f["fee"] / f["qty"])
    frame = pl.DataFrame(fills, infer_schema_length=None).with_columns(
        pl.col("market").str.replace(r"-[^-]+$", "").alias("event"))
    summary = []
    for col in [f"mo_{h}s" for h in MARKOUT_S] + [f"mx_{h}s" for h in MARKOUT_S] + ["pnl_per_contract"]:
        for sub, name in ((frame, "all"), (frame.filter(pl.col("settled")), "settled")):
            g = sub.filter(pl.col(col).is_not_null()).group_by("event").agg(
                (pl.col(col) * pl.col("qty")).sum().alias("s"), pl.col("qty").sum().alias("n"))
            if g.is_empty():
                continue
            n, k = g["n"].sum(), g.height
            mean = g["s"].sum() / n
            se = math.sqrt(k / (k - 1) * ((g["s"] - mean * g["n"]) ** 2).sum()) / n if k > 1 else 0
            summary.append({"metric": col, "fills": name, "contracts": n, "mean_cents": mean * 100,
                            "t_by_event": mean / se if se else None, "events": k,
                            "pnl": g["s"].sum()})
    rep.table("Per contract, clustered by event", pl.DataFrame(summary))
    rep.table("Fills", frame, save_as="fills", max_rows=40)
    return rep.save()
