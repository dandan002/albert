"""Latency instrumentation for the live order path (wall-clock ns; Kalshi ``ts_ms`` for the engine).

Each order action (create or cancel) is followed through its legs::

    event_ns   receive time of the market-data frame whose processing queued the action
               (None when the action was not driven by market data: halts, pauses, re-cancels)
    decide_ns  the strategy queued the action
    send_ns    the worker started the request (includes signing and the write-budget wait)
    resp_ns    the REST response arrived
    engine_ms  the response's ``ts_ms``: the matching engine's timestamp
    ws_ns      our private WebSocket delivered the matching ``user_order`` update

Derived legs (ms): ``event_to_decide`` (classifier holdback + strategy), ``decide_to_send``
(worker queue), ``rest`` (request round trip), ``send_to_engine`` (one way plus clock offset;
``ts_ms`` has 1 ms resolution), ``send_to_ws``, and the end-to-end ``event_to_engine`` /
``event_to_ws``. ``LoopLag`` measures how late the asyncio loop runs a short sleep, i.e. how long
other work blocks the loop.
"""

import asyncio
import time
from collections import deque
from collections.abc import Callable
from typing import Any

LEGS = ("event_to_decide", "decide_to_send", "rest", "send_to_engine", "send_to_ws",
        "event_to_engine", "event_to_ws")


def _ms(a: float | None, b: float | None, scale: float = 1e6) -> float | None:
    return None if a is None or b is None else round((b - a) / scale, 3)


def percentiles(xs: list[float]) -> dict[str, float] | None:
    if not xs:
        return None
    s = sorted(xs)

    def q(p: float) -> float:
        return round(s[min(len(s) - 1, int(p * len(s)))], 3)
    return {"n": len(s), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": round(s[-1], 3)}


class _Samples:
    """Per-leg samples for the current status window and (bounded) for the whole run."""

    def __init__(self, keep: int) -> None:
        self.window: dict[str, list[float]] = {}
        self.run: dict[str, deque[float]] = {}
        self.keep = keep

    def add(self, leg: str, v: float) -> None:
        self.window.setdefault(leg, []).append(v)
        self.run.setdefault(leg, deque(maxlen=self.keep)).append(v)

    def summary(self, whole_run: bool = False) -> dict[str, Any]:
        src = self.run if whole_run else self.window
        out = {leg: percentiles(list(xs)) for leg, xs in src.items() if xs}
        if not whole_run:
            self.window = {}
        return out


class OrderTiming:
    def __init__(self, clock: Callable[[], int] = time.time_ns, keep: int = 20000,
                 max_wait_s: float = 5.0) -> None:
        self.clock = clock
        self.open: dict[tuple[str, int], dict[str, Any]] = {}
        self.samples = _Samples(keep)
        self.max_wait_ns = int(max_wait_s * 1e9)

    def queued(self, kind: str, oid: int, market: str, event_ns: int | None) -> None:
        self.open[(kind, oid)] = {"kind": kind, "oid": oid, "market": market, "event_ns": event_ns,
                                  "decide_ns": self.clock()}

    def sent(self, kind: str, oid: int) -> None:
        r = self.open.get((kind, oid))
        if r is not None:
            r["send_ns"] = self.clock()

    def acked(self, kind: str, oid: int, resp: Any) -> None:
        r = self.open.get((kind, oid))
        if r is not None:
            r["resp_ns"] = self.clock()
            ts = resp.get("ts_ms") if isinstance(resp, dict) else None
            r["engine_ms"] = int(ts) if ts else None

    def failed(self, kind: str, oid: int, why: str) -> None:
        r = self.open.get((kind, oid))
        if r is not None:
            r["resp_ns"] = r.get("resp_ns") or self.clock()
            r["error"] = why

    def drop(self, kind: str, oid: int) -> None:
        self.open.pop((kind, oid), None)

    def ws(self, kind: str, oid: int, now_ns: int) -> None:
        r = self.open.get((kind, oid))
        if r is not None and "ws_ns" not in r and "send_ns" in r:
            r["ws_ns"] = now_ns

    def flush(self, now_ns: int | None = None) -> list[dict[str, Any]]:
        """Finished records (REST answered and WS update seen, or REST answered more than
        ``max_wait_s`` ago), with their derived legs. Errors are returned but not sampled."""
        now_ns = self.clock() if now_ns is None else now_ns
        out = []
        for key, r in list(self.open.items()):
            if "resp_ns" not in r:
                continue
            if "ws_ns" not in r and now_ns - r["resp_ns"] < self.max_wait_ns:
                continue
            del self.open[key]
            eng_ns = None if r.get("engine_ms") is None else r["engine_ms"] * 10**6
            legs = {"event_to_decide": _ms(r["event_ns"], r["decide_ns"]),
                    "decide_to_send": _ms(r["decide_ns"], r.get("send_ns")),
                    "rest": _ms(r.get("send_ns"), r["resp_ns"]),
                    "send_to_engine": _ms(r.get("send_ns"), eng_ns),
                    "send_to_ws": _ms(r.get("send_ns"), r.get("ws_ns")),
                    "event_to_engine": _ms(r["event_ns"], eng_ns),
                    "event_to_ws": _ms(r["event_ns"], r.get("ws_ns"))}
            rec = {"action": r["kind"], **{k: r[k] for k in ("oid", "market", "error") if r.get(k) is not None}}
            rec.update({k: v for k, v in legs.items() if v is not None})
            if "error" not in r:
                for k, v in legs.items():
                    if v is not None:
                        self.samples.add(f"{r['kind']}.{k}", v)
            out.append(rec)
        return out

    def summary(self, whole_run: bool = False) -> dict[str, Any]:
        return self.samples.summary(whole_run)


class LoopLag:
    """Lateness of a periodic short sleep: how long the event loop was blocked by other work."""

    def __init__(self, period_s: float = 0.05, keep: int = 20000) -> None:
        self.period = period_s
        self.samples = _Samples(keep)

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            t0 = time.perf_counter()
            await asyncio.sleep(self.period)
            self.samples.add("loop_lag", (time.perf_counter() - t0 - self.period) * 1e3)

    def summary(self, whole_run: bool = False) -> dict[str, Any] | None:
        return self.samples.summary(whole_run).get("loop_lag")
