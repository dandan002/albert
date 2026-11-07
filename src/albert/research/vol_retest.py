"""Re-test of every recorded-L2 strategy in volatile vs calm windows.

Thin orchestration over the existing studies, which are imported and not modified:

* makers (``backtest.l2``): ``Quoter`` baseline (improve one tick at spread >= 5c, 300 s cooldown)
  and ``JoinBest``; ``maker_variants.Variant`` behind20 (>= 20 contracts behind our quote), mom3
  (6 h momentum tilt, 3c) and both; ``momentum_l2.SignalQuoter`` overlays (flow 60 s/10, flow
  30 s/3, mid move 2c/30 s); ``intraminute_maker.VacatedLevel`` flicker provision (pull, 30 s),
  refill (exec, 30 s), scalp (pull, 10 s, exit +1 tick) and leaning (pull, 30 s). Parameters are
  the ones those studies chose on their dev windows. Every lane has its own ``Simulator`` (FIFO
  queue, risk-averse cancels, post-only, 10 writes/s, series maker fees) at 50 ms and 100 ms;
  all lanes of a pass share one parse of the recorded events.
* takers (``momentum_l2``): mid jump >= 2c in 30 s faded (the documented reversal), same-side
  trade runs and big trades followed; book imbalance and 10 s OFI in their dev top quintile
  followed. Entry at the ask after the latency with displayed size, taker fee both ways.
* ``arb_l2``: mutually exclusive sets and threshold ladders (episodes after fees, and how close
  each set gets to an arbitrage in each 5-minute bucket).
* ``leadlag_fast``: Polymarket move >= 1c -> take the stale Kalshi quote, and the sibling-team
  version (paired markets only, recorded from 11:28 UTC).

**Regimes.** The shared definition (``research.volatile``): a (market, 5-min bucket) is volatile
when its tight-spread mid path is in the top quintile, or the game is in play.

* *Evaluation split*: each fill / trigger is labelled with its **own** bucket's ``volatile``
  flag (look-ahead allowed, it only selects what to report). Arbitrage sets are volatile when any
  leg is. Volatile windows are also split by their nature (``nature``: KXTRUMPSAY, KXRAIN*,
  weather highs/lows, game moneylines (pre-game today), other sports (spreads, totals, player
  props), other).
* *Decision-time conditioning* (makers only): a lane that quotes a market only while the market's
  **previous** 5-minute bucket was volatile (``|vol``) or calm (``|calm``). The cut is fixed from
  the dev part of the window only (shared rule: top quintile of ``path_c`` > 0, or in play), and
  a bucket without top-of-book changes counts as calm, so no future data enters a decision.
  When the gate closes, live orders in that market are cancelled at its next event.

**Metrics** per fill (makers) or trade (takers), in cents per contract, t-stats clustered by
event (game for lead-lag): ``mo`` mid markout at 5 min (takers: signed mid change), ``mx`` exit
markout at 5 min (sell at the best bid on our side; takers: executable change without fees),
``net`` 5-minute PnL net of all fees (makers: ``mx`` minus the maker fee minus the taker fee to
exit; scalp entries use the realized round trip when the exit filled; takers: entry after the
latency, crossing out, both fees), ``hold`` held to settlement where known, else marked to the
last recorded mid, net of the entry fee. Split: dev before ``split``, holdout after.
"""

import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import polars as pl

from albert.backtest.l2 import (EPS, Event, JoinBest, Market, Quoter, SimConfig, Simulator,
                                iter_events, markouts, stream_files)
from albert.research.intraminute_maker import VacatedLevel, fill_rows
from albert.research.maker_variants import Variant
from albert.research.momentum_l2 import (SignalQuoter, collect, grid_samples, outcomes, taker_fee,
                                         trigger_samples)
from albert.research.volatile import GAME_SERIES, series_of

BUCKET_NS = 300 * 10**9
API = "https://api.elections.kalshi.com/trade-api/v2"
METRICS = ("mo", "mx", "net", "hold")
BASE = dict(min_spread=0.05, improve=True, cooldown_s=300.0)
MOM3 = dict(mom_window_s=21600.0, mom_move=0.03)


# ---- regimes ---------------------------------------------------------------------------------
def nature(series: str, category: str | None = None) -> str:
    """Which kind of window a series' volatility comes from (see module doc)."""
    if series == "KXTRUMPSAY":
        return "trumpsay"
    if series.startswith("KXRAIN"):
        return "rain"
    if series.startswith(("KXHIGH", "KXLOWT", "KXTEMP")) and category in (None, "Climate and Weather"):
        return "weather_hilo"
    if series in GAME_SERIES or (series.endswith("GAME") and category in (None, "Sports")):
        return "game"
    if category == "Sports":
        return "sports_props"
    return "other"


def event_of(market: str) -> str:
    return market.rsplit("-", 1)[0]


def _ns(hour: str) -> int:
    return int(datetime.fromisoformat(hour + ":00:00+00:00").timestamp() * 1e9)


def dev_cut(buckets: pl.DataFrame, split_ns: int, q: float = 0.8) -> float:
    """The shared top-quintile cut on ``path_c``, from dev buckets only."""
    dev = buckets.filter((pl.col("bucket_ns") < split_ns) & (pl.col("path_c") > 0))
    if dev.is_empty():
        dev = buckets.filter(pl.col("path_c") > 0)
    return float(dev["path_c"].quantile(q)) if dev.height else math.inf


def gate_flags(buckets: pl.DataFrame, cut_c: float) -> dict[tuple[str, int], bool]:
    """(market, bucket index) -> volatile under a fixed cut (each bucket's own data only)."""
    f = buckets.select("market", (pl.col("bucket_ns") // BUCKET_NS).alias("b"),
                       (((pl.col("path_c") >= cut_c) & (pl.col("path_c") > 0)) | pl.col("in_play")).alias("v"))
    return {(m, b): v for m, b, v in f.iter_rows()}


def persistence(flags: dict[tuple[str, int], bool]) -> dict[str, float | None]:
    """P(bucket volatile | previous bucket volatile / calm) under the gate's flags: whether a
    previous-bucket gate can select volatile windows at all."""
    acc = {True: [0, 0], False: [0, 0]}
    for (m, b), v in flags.items():
        a = acc[flags.get((m, b - 1), False)]
        a[0] += v
        a[1] += 1
    return {"p_vol_after_vol": round(acc[True][0] / acc[True][1], 3) if acc[True][1] else None,
            "p_vol_after_calm": round(acc[False][0] / acc[False][1], 3) if acc[False][1] else None}


class Gate:
    """Open for a market while its previous 5-minute bucket's flag equals ``want``."""

    def __init__(self, flags: dict[tuple[str, int], bool], want: bool) -> None:
        self.flags, self.want = flags, want

    def open(self, market: str, t_ns: int) -> bool:
        return self.flags.get((market, t_ns // BUCKET_NS - 1), False) is self.want


def label(frame: pl.DataFrame, buckets: pl.DataFrame, split_ns: int,
          category_of: dict[str, str | None]) -> pl.DataFrame:
    """Add ``volatile`` (the row's own bucket, shared definition; no row = calm), ``spread_c``
    (the bucket's median quoted spread), ``group`` (``nature`` of the series) and ``split`` (dev /
    hold) to rows with ``market`` and ``t_ns``. Rows that already carry ``volatile`` keep it."""
    if frame.is_empty():
        return frame
    lab = buckets.select("market", "bucket_ns", pl.col("volatile").alias("_v"),
                         pl.col("spread_med_c").alias("spread_c"))
    frame = (frame.drop("spread_c", strict=False)
             .with_columns((pl.col("t_ns") // BUCKET_NS * BUCKET_NS).alias("bucket_ns"))
             .join(lab, on=["market", "bucket_ns"], how="left"))
    if "volatile" not in frame.columns:
        frame = frame.with_columns(pl.col("_v").fill_null(False).alias("volatile"))
    frame = frame.drop("_v", "bucket_ns")
    groups = {m: nature(series_of(m), category_of.get(series_of(m))) for m in frame["market"].unique()}
    return frame.with_columns(
        pl.col("market").replace_strict(groups, return_dtype=pl.Utf8).alias("group"),
        pl.when(pl.col("t_ns") < split_ns).then(pl.lit("dev")).otherwise(pl.lit("hold")).alias("split"))


# ---- maker lanes -----------------------------------------------------------------------------
class FastMarket(Market):
    """``Market`` whose ``best`` walks prices from the top and stops at the first level with
    external size: the same (price, size) as the base class, which sums every level first."""

    def best(self, side: str) -> tuple[float, float] | None:
        lv = self.levels[side]
        for p in sorted(lv, reverse=True):
            s = lv[p].ext_total()
            if s > EPS:
                return p, s
        return None


class FastSimulator(Simulator):
    """``Simulator`` on ``FastMarket`` books, dropping finished orders (filled, cancelled,
    rejected) from the per-market order lists that ``live_orders`` and ``_ours_at`` scan on every
    event (neither ever returns them). Speed only; the model is unchanged. ``orders`` keeps all."""

    LIVE = ("pending", "active", "cancelling", "cancel_pending")

    def __init__(self, cfg: SimConfig) -> None:
        super().__init__(cfg)
        self.markets = defaultdict(FastMarket)

    def _prune(self, market: str) -> None:
        ids = self.by_market[market]
        if len(ids) > 16:
            self.by_market[market] = [i for i in ids if self.orders[i].status in self.LIVE]

    def live_orders(self, market: str):
        self._prune(market)
        return super().live_orders(market)

    def _ours_at(self, market: str, side: str):
        self._prune(market)
        return super()._ours_at(market, side)


def _specs(cat: dict[str, str | None]) -> dict[str, tuple[Callable[[], Any], str]]:
    """name -> (strategy factory, call style). Parameters as chosen in the original studies."""
    return {
        "quoter_base": (lambda: Quoter(category_of=cat, **BASE), "market"),
        "join_best": (lambda: JoinBest(), "market"),
        "behind20": (lambda: Variant(category_of=cat, min_behind=20, **BASE), "market"),
        "mom3": (lambda: Variant(category_of=cat, **MOM3, **BASE), "market"),
        "behind20_mom3": (lambda: Variant(category_of=cat, min_behind=20, **MOM3, **BASE), "market"),
        "ovl_flow60": (lambda: SignalQuoter(signal="flow", window_s=60, flow_min=10, hold_s=60,
                                            category_of=cat, **BASE), "market"),
        "ovl_flow30": (lambda: SignalQuoter(signal="flow", window_s=30, flow_min=3, hold_s=120,
                                            category_of=cat, **BASE), "market"),
        "ovl_mom30": (lambda: SignalQuoter(signal="mom", window_s=30, move=0.02, hold_s=60,
                                           category_of=cat, **BASE), "market"),
        "im_flicker": (lambda: VacatedLevel(cause="pull", offset=0, ttl_s=30), "event"),
        "im_refill": (lambda: VacatedLevel(cause="exec", offset=0, ttl_s=30), "event"),
        "im_scalp": (lambda: VacatedLevel(cause="pull", offset=0, ttl_s=10, scalp_k=1), "event"),
        "im_lean": (lambda: VacatedLevel(cause="pull", ttl_s=30, direction="lean"), "event"),
    }


GATED = ("quoter_base", "join_best", "behind20", "mom3")


def lane_names(latencies: Iterable[float], gated: Iterable[str] = GATED,
               specs: Iterable[str] | None = None) -> list[str]:
    """``<spec>[|vol|calm]@<latency>`` for every spec, plus gated versions of ``gated``."""
    specs = list(specs) if specs is not None else list(_specs({}))
    out = []
    for lat in latencies:
        for sp in specs:
            out.append(f"{sp}@{lat:g}")
            if sp in gated:
                out += [f"{sp}|vol@{lat:g}", f"{sp}|calm@{lat:g}"]
    return out


def parse_lane(name: str) -> tuple[str, str | None, float]:
    head, lat = name.rsplit("@", 1)
    spec, _, gate = head.partition("|")
    return spec, gate or None, float(lat)


@dataclass
class Lane:
    name: str
    strategy: Any
    style: str                      # market: on_event(sim, market); event: on_event(sim, ev)
    sim: Simulator
    gate: Gate | None = None
    stats: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def step(self, ev: Event) -> None:
        sim = self.sim
        sim.apply(ev)
        observe = getattr(self.strategy, "observe", None)
        if observe is not None:
            observe(sim, ev)
        if self.gate is not None and not self.gate.open(ev.market, ev.recv_ns):
            self.stats["gated_events"] += 1
            for o in sim.live_orders(ev.market):
                sim.cancel(o.oid)
            return
        if self.style == "event":
            self.strategy.on_event(sim, ev)
        else:
            self.strategy.on_event(sim, ev.market)


def make_lane(name: str, fee_info: dict[str, dict], category_of: dict[str, str | None],
              flags: dict[tuple[str, int], bool] | None) -> Lane:
    spec, gate, lat = parse_lane(name)
    factory, style = _specs(category_of)[spec]
    g = None if gate is None else Gate(flags or {}, gate == "vol")
    sim = FastSimulator(SimConfig(latency_ms=lat, queue="risk_averse", fee_info=fee_info, writes_per_s=10.0))
    return Lane(name, factory(), style, sim, g)


def run_lanes(events: Iterable[Event], lanes: list[Lane], log: Callable[[str], None] | None = None) -> None:
    """One pass over the events, every lane stepping on each event in turn."""
    hour, t0, n = None, time.time(), 0
    for ev in events:
        n += 1
        h = ev.recv_ns // 3_600_000_000_000
        if log is not None and h != hour:
            if hour is not None:
                log(f"  hour done ({n} events, {time.time() - t0:.0f}s, fills "
                    + ", ".join(f"{ln.name}:{len(ln.sim.fills)}" for ln in lanes[:4]) + " ...)")
            hour = h
        for ln in lanes:
            ln.step(ev)


FILL_SCHEMA = {"lane": pl.Utf8, "t_ns": pl.Int64, "market": pl.Utf8, "side": pl.Utf8, "price": pl.Float64,
               "qty": pl.Float64, "fee": pl.Float64, "how": pl.Utf8, "role": pl.Utf8,
               "mo_300s": pl.Float64, "mx_300s": pl.Float64, "rt_pnl": pl.Float64, "last_mid": pl.Float64,
               "age_s": pl.Float64}


def lane_fills(lane: Lane) -> pl.DataFrame:
    """Per fill: markouts (``backtest.l2.markouts``), role and round trip (VacatedLevel lanes),
    and the last recorded mid of the market (the fallback mark for ``hold``)."""
    sim = lane.sim
    mo = markouts(sim)
    extra = fill_rows(sim, lane.strategy, {}, horizons=(300,)) if lane.style == "event" else None
    rows = []
    for i, (f, r) in enumerate(zip(sim.fills, mo)):
        e = extra[i] if extra is not None else {}
        rows.append({"lane": lane.name, "t_ns": f.t_ns, "market": f.market, "side": f.side,
                     "price": f.price, "qty": f.qty, "fee": f.fee, "how": f.how,
                     "role": e.get("role", "entry"), "mo_300s": r["mo_300s"], "mx_300s": r["mx_300s"],
                     "rt_pnl": e.get("rt_pnl"), "last_mid": sim.mid_at(f.market, 2**62),
                     "age_s": (f.t_ns - sim.orders[f.oid].active_ns) / 1e9})
    return pl.DataFrame(rows, schema=FILL_SCHEMA)


def maker_metrics(fills: pl.DataFrame, results: dict[str, float | None],
                  fee_mult: dict[str, float]) -> pl.DataFrame:
    """Unified rows (strategy, latency, market, t_ns, qty, cluster, mo, mx, net, hold) from
    cached fills: entries only (scalp exits are scored through ``rt_pnl``)."""
    rows = []
    for r in fills.filter(pl.col("role") == "entry").iter_rows(named=True):
        spec, gate, lat = parse_lane(r["lane"])
        fee_pc = r["fee"] / r["qty"]
        net = None
        if r["rt_pnl"] is not None:
            net = r["rt_pnl"]
        elif r["mx_300s"] is not None:
            ex = r["mx_300s"] + r["price"]
            net = r["mx_300s"] - fee_pc - taker_fee(ex, fee_mult.get(series_of(r["market"]), 1.0))
        res = results.get(r["market"])
        final = res if res is not None else r["last_mid"]
        hold = None if final is None else (final if r["side"] == "yes" else 1 - final) - r["price"] - fee_pc
        rows.append({"strategy": spec + (f"|{gate}" if gate else ""), "latency": lat, "market": r["market"],
                     "t_ns": r["t_ns"], "qty": r["qty"], "cluster": event_of(r["market"]),
                     "mo": r["mo_300s"], "mx": r["mx_300s"], "net": net, "hold": hold,
                     "settled": res is not None})
    return pl.DataFrame(rows, schema=UNIFIED)


UNIFIED = {"strategy": pl.Utf8, "latency": pl.Float64, "market": pl.Utf8, "t_ns": pl.Int64,
           "qty": pl.Float64, "cluster": pl.Utf8, "mo": pl.Float64, "mx": pl.Float64,
           "net": pl.Float64, "hold": pl.Float64, "settled": pl.Boolean}


# ---- takers: momentum_l2 -----------------------------------------------------------------------
def momentum_rows(tapes: dict, end: int, split_ns: int, latencies: Iterable[float],
                  fee_info: dict[str, dict]) -> pl.DataFrame:
    """Trigger and strong-feature taker trades at each latency (the settle leg is filled in
    later from ``entry``/``dir``, so the cache does not freeze unsettled results)."""
    out: list[dict[str, Any]] = []

    def add(name: str, market: str, t: int, d: float) -> None:
        tp = tapes[market]
        mult = ((fee_info.get(series_of(market)) or {}).get("fee_multiplier")) or 1.0
        for lat in latencies:
            o = outcomes(tp, t, d, end, latency_ns=int(lat * 1e6), fee_mult=mult)
            if o is None:
                continue
            out.append({"strategy": name, "latency": float(lat), "market": market, "t_ns": t, "dir": d,
                        "entry": o["entry"], "mult": float(mult), "mo": o["mid_300s"], "mx": o["exit_300s"],
                        "net": o["taker_300s"]})

    for r in trigger_samples(tapes, end, move=0.02, move_s=30.0, gap_s=60.0, fee_info=fee_info):
        kind = r["kind"]
        name = {"move": "tk_jump_fade", "run": "tk_run_follow", "big": "tk_big_follow"}[kind]
        add(name, r["market"], r["t_ns"], -r["dir"] if kind == "move" else r["dir"])
    # Grid features, one market at a time (bounded memory): dev quintile cut, then the trades.
    feats = ("imb", "ofi_10")
    dev_abs: dict[str, list[float]] = {f: [] for f in feats}
    for mk in tapes:
        for r in grid_samples({mk: tapes[mk]}, end, fee_info=fee_info):
            if r["t_ns"] < split_ns:
                for f in feats:
                    if abs(r[f]) > 1e-12:
                        dev_abs[f].append(abs(r[f]))
    thr = {}
    for f in feats:
        nz = sorted(dev_abs[f])
        thr[f] = nz[int(0.8 * len(nz))] if nz else math.inf
    for mk in tapes:
        for r in grid_samples({mk: tapes[mk]}, end, fee_info=fee_info):
            for f in feats:
                if abs(r[f]) >= thr[f] and abs(r[f]) > 1e-12:
                    add(f"tk_{f}_follow", r["market"], r["t_ns"], math.copysign(1.0, r[f]))
    schema = {"strategy": pl.Utf8, "latency": pl.Float64, "market": pl.Utf8, "t_ns": pl.Int64, "dir": pl.Float64,
              "entry": pl.Float64, "mult": pl.Float64, "mo": pl.Float64, "mx": pl.Float64, "net": pl.Float64}
    return pl.DataFrame(out, schema=schema)


def momentum_metrics(rows: pl.DataFrame, results: dict[str, float | None]) -> pl.DataFrame:
    def hold(r: dict[str, Any]) -> float | None:
        res = results.get(r["market"])
        if res is None or r["entry"] is None:
            return None
        return (res if r["dir"] > 0 else 1 - res) - r["entry"] - taker_fee(r["entry"], r["mult"])

    if rows.is_empty():
        return pl.DataFrame(schema=UNIFIED)
    holds = [hold(r) for r in rows.iter_rows(named=True)]
    return rows.with_columns(
        pl.Series("hold", holds, dtype=pl.Float64), pl.lit(1.0).alias("qty"),
        pl.col("market").str.replace(r"-[^-]+$", "").alias("cluster"),
        pl.col("market").map_elements(lambda m: results.get(m) is not None, return_dtype=pl.Boolean).alias("settled"),
    ).select(list(UNIFIED))


# ---- arbitrage: arb_l2 -------------------------------------------------------------------------
def parse_event(payload: dict[str, Any]) -> dict[str, Any]:
    """``arb_l2.build_sets`` input from a recorded ``GET /events/{ticker}`` payload."""
    ev = payload.get("event") or {}
    legs = [{"t": m["ticker"], "strike_type": m.get("strike_type"), "floor": m.get("floor_strike"),
             "close_time": m.get("close_time")} for m in payload.get("markets") or []]
    return {"me": bool(ev.get("mutually_exclusive")), "legs": legs}


def event_meta(markets: Iterable[str], cache: Path, *, fetch: bool = True) -> dict[str, dict[str, Any]]:
    """Event structure for the events of ``markets`` (public REST, cached as JSON)."""
    have = json.loads(cache.read_text()) if cache.exists() else {}
    todo = sorted({event_of(m) for m in markets} - set(have))
    if fetch and todo:
        import httpx

        with httpx.Client(timeout=20) as http:
            for e in todo:
                r = http.get(f"{API}/events/{e}")
                have[e] = parse_event(r.json()) if r.status_code == 200 else None
                time.sleep(0.05)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(have))
    return {e: have[e] for e in {event_of(m) for m in markets} if have.get(e)}


def arb_scan(stream: Callable[[], Iterable[Any]], sets: list, latencies: Iterable[float]
             ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(episodes, profile): ``arb_l2.replay`` episodes with their latency outcomes, and per (set,
    5-min bucket) the best edge after fees seen at any book update (how close to an arbitrage).
    ``stream()`` returns a fresh ``arb_l2.Event`` iterator (two passes)."""
    from albert.research.arb_l2 import Books, replay, set_edge

    lats = tuple(int(x) for x in latencies)
    eps, _ = replay(iter(stream()), sets, latencies_ms=lats)
    ep_rows = [{"key": e.key, "kind": e.kind, "event": e.event, "t_ns": e.start_ns, "edge0": e.edge0,
                "legs": [m for m, _ in next(s for s in sets if s.key == e.key).legs],
                **{f"pnl_{lat}": e.latency.get(lat, {}).get("pnl") for lat in lats}} for e in eps]
    books = Books()
    by_market: dict[str, list] = defaultdict(list)
    for s in sets:
        for m, _ in s.legs:
            by_market[m].append(s)
    best: dict[tuple[str, int], float] = {}
    for ev in stream():
        if ev.kind == "trade" or ev.market not in by_market:
            continue
        books.apply(ev.kind, ev.market, ev.msg)
        if books.crossed(ev.market):
            continue
        b = ev.recv_ns // BUCKET_NS
        for s in by_market[ev.market]:
            r = set_edge(s, books, ev.recv_ns)
            if r is not None and r[0] > best.get((s.key, b), -math.inf):
                best[(s.key, b)] = r[0]
    meta = {s.key: s for s in sets}
    prof = [{"key": k, "kind": meta[k].kind, "event": meta[k].event, "bucket": b, "best_edge": v,
             "legs": [m for m, _ in meta[k].legs]} for (k, b), v in best.items()]
    ep_schema = {"key": pl.Utf8, "kind": pl.Utf8, "event": pl.Utf8, "t_ns": pl.Int64, "edge0": pl.Float64,
                 "legs": pl.List(pl.Utf8), **{f"pnl_{lat}": pl.Float64 for lat in lats}}
    pr_schema = {"key": pl.Utf8, "kind": pl.Utf8, "event": pl.Utf8, "bucket": pl.Int64,
                 "best_edge": pl.Float64, "legs": pl.List(pl.Utf8)}
    return pl.DataFrame(ep_rows, schema=ep_schema), pl.DataFrame(prof, schema=pr_schema)


def _any_leg_volatile(legs: list[str], b: int, vol: dict[tuple[str, int], bool]) -> bool:
    return any(vol.get((m, b), False) for m in legs)


def arb_metrics(eps: pl.DataFrame, vol: dict[tuple[str, int], bool], latencies: Iterable[float]) -> pl.DataFrame:
    rows = []
    for r in eps.iter_rows(named=True):
        v = _any_leg_volatile(r["legs"], r["t_ns"] // BUCKET_NS, vol)
        for lat in latencies:
            pnl = r.get(f"pnl_{int(lat)}")
            rows.append({"strategy": f"arb_{r['kind']}", "latency": float(lat), "market": r["legs"][0],
                         "t_ns": r["t_ns"], "qty": 1.0, "cluster": r["event"], "mo": r["edge0"], "mx": None,
                         "net": pnl, "hold": pnl, "settled": False, "volatile": v})
    return pl.DataFrame(rows, schema=UNIFIED | {"volatile": pl.Boolean})


def arb_profile(prof: pl.DataFrame, vol: dict[tuple[str, int], bool], split_ns: int) -> pl.DataFrame:
    """Share of (set, bucket) cells within 1c / at an arbitrage after fees, by regime and split."""
    if prof.is_empty():
        return pl.DataFrame()
    f = prof.with_columns(
        pl.struct("legs", "bucket").map_elements(lambda x: _any_leg_volatile(x["legs"], x["bucket"], vol),
                                                 return_dtype=pl.Boolean).alias("volatile"),
        pl.when(pl.col("bucket") * BUCKET_NS < split_ns).then(pl.lit("dev")).otherwise(pl.lit("hold")).alias("split"))
    out = []
    for kind in sorted(f["kind"].unique()):
        for regime, cond in (("all", None), ("volatile", True), ("calm", False)):
            for split in ("dev", "hold"):
                g = f.filter((pl.col("kind") == kind) & (pl.col("split") == split))
                if cond is not None:
                    g = g.filter(pl.col("volatile") == cond)
                out.append({"kind": kind, "regime": regime, "split": split, "set_buckets": g.height,
                            "best_edge_c_median": None if g.is_empty() else round(100 * g["best_edge"].median(), 2),
                            "best_edge_c_max": None if g.is_empty() else round(100 * g["best_edge"].max(), 2),
                            "within_1c_share": None if g.is_empty() else round((g["best_edge"] >= -0.01).mean(), 4),
                            "arb_share": None if g.is_empty() else round((g["best_edge"] > 1e-9).mean(), 4)})
    return pl.DataFrame(out)


# ---- lead-lag: leadlag_fast --------------------------------------------------------------------
def leadlag_metrics(moves: dict[str, pl.DataFrame], latencies: Iterable[float], exit_s: int = 10) -> pl.DataFrame:
    """PM->Kalshi and sibling stale-quote takers (move >= 1c, edge >= 0 at the latency)."""
    rows = []
    for leader, f in moves.items():
        if f.is_empty():
            continue
        for r in f.iter_rows(named=True):
            for lat in latencies:
                edge, pnl = r.get(f"edge_{int(lat)}ms"), r.get(f"pnl_{int(lat)}ms_{exit_s}s")
                if edge is None or edge < -EPS or pnl is None:
                    continue
                rows.append({"strategy": f"ll_{leader}_taker", "latency": float(lat), "market": r["market"],
                             "t_ns": r["t_ns"], "qty": 1.0, "cluster": r["game"], "mo": r.get(f"resp_{exit_s}s"),
                             "mx": None, "net": pnl, "hold": None, "settled": False})
    return pl.DataFrame(rows, schema=UNIFIED)


# ---- results -----------------------------------------------------------------------------------
def market_results(tickers: Iterable[str], cache: Path, *, fetch: bool = True) -> dict[str, float | None]:
    """Settled YES result (1.0 / 0.0) per market, public REST, cached; unsettled ones are re-asked."""
    tickers = sorted(set(tickers))
    have = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [t for t in tickers if have.get(t) is None]
    if fetch and todo:
        import httpx

        with httpx.Client(timeout=20) as http:
            for t in todo:
                r = http.get(f"{API}/markets/{t}")
                if r.status_code == 200:
                    have[t] = {"yes": 1.0, "no": 0.0}.get(r.json().get("market", {}).get("result"))
                time.sleep(0.05)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(have))
    return {t: have.get(t) for t in tickers}


# ---- summary -----------------------------------------------------------------------------------
def clustered_t(x: pl.DataFrame, col: str, cluster: str = "cluster") -> tuple[float | None, float | None, int]:
    """Contract-weighted mean of ``col`` and its t-stat with standard errors clustered by
    ``cluster`` (as ``backtest.l2._cluster_t``); t is None with < 2 clusters or no variation."""
    f = x.filter(pl.col(col).is_not_null())
    if f.is_empty():
        return None, None, 0
    g = f.group_by(cluster).agg((pl.col(col) * pl.col("qty")).sum().alias("s"), pl.col("qty").sum().alias("n"))
    n = float(g["n"].sum())
    mean = float(g["s"].sum()) / n
    k = g.height
    if k < 2:
        return mean, None, k
    se = math.sqrt(k / (k - 1) * float(((g["s"] - mean * g["n"]) ** 2).sum())) / n
    return mean, (mean / se if se > 1e-9 else None), k


def _stats(x: pl.DataFrame) -> dict[str, Any]:
    r: dict[str, Any] = {"n": x.height, "contracts": float(x["qty"].sum()) if x.height else 0.0,
                         "events": x["cluster"].n_unique() if x.height else 0}
    for m in METRICS:
        mean, t, _ = clustered_t(x, m)
        r[m] = None if mean is None else round(100 * mean, 2)
        r[f"t_{m}"] = None if t is None else round(t, 2)
    r["net_usd"] = round(float((x["qty"] * x["net"]).sum()), 3) if x.height else 0.0
    r["hold_usd"] = round(float((x["qty"] * x["hold"]).sum()), 3) if x.height else 0.0
    r["settled_share"] = round(float(x["settled"].mean()), 3) if x.height else None
    sp = x["spread_c"].drop_nulls() if "spread_c" in x.columns else pl.Series([], dtype=pl.Float64)
    r["spread_c_med"] = round(float(sp.median()), 2) if sp.len() else None
    return r


def spread_band(col: str = "spread_c") -> pl.Expr:
    """The bucket's median quoted spread in bands (cents)."""
    c = pl.col(col)
    return (pl.when(c.is_null()).then(pl.lit("spread ?")).when(c <= 3).then(pl.lit("spread<=3c"))
            .when(c <= 6).then(pl.lit("spread 4-6c")).otherwise(pl.lit("spread>6c")))


def summarize(df: pl.DataFrame, *, by_group: bool = False, by: str | None = None) -> pl.DataFrame:
    """Long table: strategy x latency x regime x split. Regimes: all / volatile / calm, or with
    ``by_group`` (or ``by=<column>``) volatile:<value> and calm:<value>."""
    by = by or ("group" if by_group else None)
    out = []
    for (strategy, latency), g in df.group_by(["strategy", "latency"], maintain_order=True):
        if by:
            regimes = [(f"{'volatile' if v else 'calm'}:{grp}", (pl.col("volatile") == v) & (pl.col(by) == grp))
                       for grp in sorted(g[by].unique().drop_nulls()) for v in (True, False)]
        else:
            regimes = [("all", None), ("volatile", pl.col("volatile")), ("calm", ~pl.col("volatile"))]
        for regime, cond in regimes:
            gg = g if cond is None else g.filter(cond)
            for split in ("dev", "hold"):
                x = gg.filter(pl.col("split") == split)
                out.append({"strategy": strategy, "latency": latency, "regime": regime, "split": split, **_stats(x)})
    return pl.DataFrame(out, infer_schema_length=None)


def flags(long: pl.DataFrame) -> pl.DataFrame:
    """Per (strategy, latency, regime): metrics with t >= 2 on both dev and holdout (``+``), and
    with t <= -2 on both (``-``)."""
    out = []
    for (st, lat, reg), g in long.group_by(["strategy", "latency", "regime"], maintain_order=True):
        d = {r["split"]: r for r in g.iter_rows(named=True)}
        dev, hold = d.get("dev", {}), d.get("hold", {})
        pos = [m for m in METRICS if (dev.get(f"t_{m}") or 0) >= 2 and (hold.get(f"t_{m}") or 0) >= 2]
        neg = [m for m in METRICS if (dev.get(f"t_{m}") or 0) <= -2 and (hold.get(f"t_{m}") or 0) <= -2]
        out.append({"strategy": st, "latency": lat, "regime": reg, "pos": ",".join(pos), "neg": ",".join(neg)})
    return pl.DataFrame(out)


def wide(long: pl.DataFrame, latency: float, metrics: tuple[str, ...] = METRICS) -> pl.DataFrame:
    """Readable table: one row per strategy x regime; cells 'mean (t)', ``**`` when t >= 2 in both
    dev and holdout."""
    fl = {(r["strategy"], r["regime"]): r for r in flags(long).filter(pl.col("latency") == latency).iter_rows(named=True)}
    rows = []
    sub = long.filter(pl.col("latency") == latency)
    for (st, reg), g in sub.group_by(["strategy", "regime"], maintain_order=True):
        d = {r["split"]: r for r in g.iter_rows(named=True)}
        dev, hold = d.get("dev", {}), d.get("hold", {})
        pos = (fl.get((st, reg)) or {}).get("pos", "").split(",")
        row = {"strategy": st, "regime": reg, "n dev/hold": f"{dev.get('n', 0)}/{hold.get('n', 0)}",
               "ev dev/hold": f"{dev.get('events', 0)}/{hold.get('events', 0)}"}
        for m in metrics:
            for nm, x in (("dev", dev), ("hold", hold)):
                v, t = x.get(m), x.get(f"t_{m}")
                cell = "" if v is None else f"{v:+.2f} ({'-' if t is None else f'{t:+.1f}'})"
                row[f"{m} {nm}"] = cell + (" **" if m in pos and cell else "")
        row["net$ dev/hold"] = f"{dev.get('net_usd', 0):+.2f}/{hold.get('net_usd', 0):+.2f}"
        row["hold$ dev/hold"] = f"{dev.get('hold_usd', 0):+.2f}/{hold.get('hold_usd', 0):+.2f}"
        row["spread c dev/hold"] = f"{dev.get('spread_c_med')}/{hold.get('spread_c_med')}"
        rows.append(row)
    return pl.DataFrame(rows)


def sign_changes(long: pl.DataFrame, latency: float, metrics: tuple[str, ...] = ("mo", "mx", "net", "hold")) -> pl.DataFrame:
    """Volatile vs calm means per strategy; ``flips`` when the sign differs in both dev and
    holdout (and both cells have at least 10 observations)."""
    sub = long.filter(pl.col("latency") == latency)
    out = []
    for st in sub["strategy"].unique(maintain_order=True):
        g = {(r["regime"], r["split"]): r for r in sub.filter(pl.col("strategy") == st).iter_rows(named=True)}
        for m in metrics:
            cells = {k: g.get(k, {}) for k in [("volatile", "dev"), ("calm", "dev"), ("volatile", "hold"), ("calm", "hold")]}
            vals = {k: c.get(m) for k, c in cells.items()}
            enough = all((c.get("n") or 0) >= 10 for c in cells.values())

            def flip(split: str) -> bool:
                a, b = vals[("volatile", split)], vals[("calm", split)]
                return a is not None and b is not None and (a > 0) != (b > 0) and a != 0 and b != 0

            out.append({"strategy": st, "metric": m,
                        "vol dev": vals[("volatile", "dev")], "calm dev": vals[("calm", "dev")],
                        "vol hold": vals[("volatile", "hold")], "calm hold": vals[("calm", "hold")],
                        "t vol hold": cells[("volatile", "hold")].get(f"t_{m}"),
                        "t calm hold": cells[("calm", "hold")].get(f"t_{m}"),
                        "spread vol/calm hold": f"{cells[('volatile', 'hold')].get('spread_c_med')}/"
                                                f"{cells[('calm', 'hold')].get('spread_c_med')}",
                        "flips": enough and flip("dev") and flip("hold"),
                        "vol better": (enough and None not in vals.values()
                                       and vals[("volatile", "dev")] > vals[("calm", "dev")]
                                       and vals[("volatile", "hold")] > vals[("calm", "hold")])})
    return pl.DataFrame(out)


def gate_table(long: pl.DataFrame) -> pl.DataFrame:
    """Unconditioned vs quote-only-after-volatile vs quote-only-after-calm, regime ``all``."""
    keep = [s for s in long["strategy"].unique() if s.endswith(("|vol", "|calm"))]
    bases = sorted({s.split("|")[0] for s in keep})
    f = long.filter(pl.col("regime").eq("all") & pl.col("strategy").is_in(bases + keep))
    return f.select("strategy", "latency", "split", "n", "events", "mo", "t_mo", "mx", "t_mx", "net", "t_net",
                    "hold", "t_hold", "net_usd", "hold_usd").sort("strategy", "latency", "split")


def conclusion(long: pl.DataFrame, latency: float, bands: pl.DataFrame | None = None) -> str:
    """Rule-based reading of the tables (regenerated on every run). ``bands``: ``summarize`` by
    spread band, to check whether a volatile/calm gap is only a spread-composition effect."""
    lines = []
    sc = sign_changes(long, latency, ("mo", "mx", "net"))
    flips = sc.filter(pl.col("flips"))
    if flips.height:
        lines.append("Sign changes volatile vs calm (both dev and holdout): "
                     + "; ".join(f"{r['strategy']} {r['metric']} (vol {r['vol hold']:+.2f}c vs calm "
                                 f"{r['calm hold']:+.2f}c holdout)" for r in flips.iter_rows(named=True)) + ".")
    else:
        lines.append("No strategy changes the sign of mo/mx/net between volatile and calm windows on both "
                     "dev and holdout (cells with >= 10 observations).")
    better = sc.filter(pl.col("vol better") & pl.col("metric").is_in(["mx", "net"]))
    if better.height:
        lines.append("Volatile better than calm on dev and holdout: "
                     + ", ".join(f"{r['strategy']} {r['metric']}" for r in better.iter_rows(named=True)) + ".")
    fl = flags(long).filter(pl.col("latency") == latency)
    pos = fl.filter(pl.col("pos").str.len_chars() > 0)
    lines.append("Cells with t >= 2 on both dev and holdout: "
                 + ("; ".join(f"{r['strategy']} [{r['regime']}] {r['pos']}" for r in pos.iter_rows(named=True))
                    if pos.height else "none") + ".")
    g = gate_table(long).filter(pl.col("latency") == latency)
    for base in ("quoter_base",):
        d = {(r["strategy"], r["split"]): r for r in g.iter_rows(named=True)}
        parts = []
        for v in (base, f"{base}|vol", f"{base}|calm"):
            h = d.get((v, "hold"))
            if h:
                parts.append(f"{v}: {h['n']} fills, net {h['net']}c (t {h['t_net']}), hold {h['hold']}c, "
                             f"net ${h['net_usd']}, hold ${h['hold_usd']}")
        if parts:
            lines.append("Baseline gating, holdout: " + " | ".join(parts) + ".")
    if bands is not None and bands.height:
        b = bands.filter((pl.col("strategy") == "quoter_base") & (pl.col("latency") == latency))
        d = {(r["regime"], r["split"]): r for r in b.iter_rows(named=True)}
        cmp = []
        for band in ("spread<=3c", "spread 4-6c", "spread>6c"):
            for split in ("dev", "hold"):
                v, c = d.get((f"volatile:{band}", split)), d.get((f"calm:{band}", split))
                if v and c and v["n"] >= 5 and c["n"] >= 5:
                    cmp.append(f"{band} {split}: vol mx {v['mx']} / net {v['net']} (n {v['n']}) vs calm mx "
                               f"{c['mx']} / net {c['net']} (n {c['n']})")
        if cmp:
            lines.append("Baseline within spread bands (is the volatile/calm gap only tighter spreads?): "
                         + "; ".join(cmp) + ".")
    return "\n\n".join(lines)


# ---- orchestration -----------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, file=sys.stderr, flush=True)


def run(start: str = "2026-11-05T03", end: str = "2026-11-05T18", *, split: str = "2026-11-05T11",
        latencies: tuple[float, ...] = (50.0, 100.0), lanes: list[str] | None = None,
        parts: tuple[str, ...] = ("maker", "momentum", "arb", "leadlag"), max_lanes: int = 14,
        gate_cut_c: float | None = None, root: Path | None = None, s=None, fetch: bool = True) -> Path:
    """Run every re-test over recorded hours ``start <= hour < end`` (``YYYY-MM-DDTHH``), split
    dev/holdout at ``split``, and write report ``research-vol-retest``. Derived results are cached
    under ``data/cache/vol_retest_<start>_<end>_*``; a rerun only computes what is missing.
    Tonight's in-play hours: ``run('2026-11-05T23', '2026-11-06T07', split='2026-11-06T03')``."""
    from albert.config import settings
    from albert.reports import Report
    from albert.research.volatile import cached_buckets

    s = s or settings()
    root = root or s.raw_dir / "stream"
    cache = s.data_dir / "cache"
    tag = f"vol_retest_{start}_{end}".replace(":", "")
    split_ns = _ns(split)
    files = stream_files(root, start, end)
    if not files:
        raise RuntimeError(f"no recorded Kalshi stream files in {start}..{end}")
    ser = pl.read_parquet(s.tables_dir / "kalshi_series.parquet")
    fee_info = {r["series_ticker"]: r for r in ser.iter_rows(named=True)}
    category_of = dict(zip(ser["series_ticker"], ser["category"]))
    fee_mult = {k: float(v.get("fee_multiplier") or 1.0) for k, v in fee_info.items()}
    buckets = cached_buckets(s, start, end, root=root)
    cut = gate_cut_c if gate_cut_c is not None else dev_cut(buckets, split_ns)
    gflags = gate_flags(buckets, cut)
    vol = {(m, b // BUCKET_NS): v for m, b, v in buckets.select("market", "bucket_ns", "volatile").iter_rows()}
    t_start = time.time()
    unified: list[pl.DataFrame] = []
    info: dict[str, Any] = {"start": start, "end": end, "split": split, "files": len(files),
                            "latencies_ms": list(latencies), "gate_cut_c (dev-only)": cut,
                            "shared_cut_c (whole window)": float(buckets.filter(pl.col("path_c") > 0)["path_c"].quantile(0.8)),
                            "buckets": buckets.height, "volatile_buckets": int(buckets["volatile"].sum()),
                            "in_play_buckets": int(buckets["in_play"].sum()),
                            "gate_persistence": persistence(gflags)}

    # makers
    if "maker" in parts:
        names = lanes or lane_names(latencies)
        todo = [n for n in names if not (cache / f"{tag}_lane_{n.replace('|', '-')}.parquet").exists()]
        for i in range(0, len(todo), max_lanes):
            chunk = todo[i:i + max_lanes]
            _log(f"maker pass {i // max_lanes + 1}: {chunk}")
            ls = [make_lane(n, fee_info, category_of, gflags) for n in chunk]
            run_lanes(iter_events(files, defaultdict(int)), ls, _log)
            for ln in ls:
                lane_fills(ln).write_parquet(cache / f"{tag}_lane_{ln.name.replace('|', '-')}.parquet")
                _log(f"  {ln.name}: {len(ln.sim.fills)} fills, {len(ln.sim.orders)} orders, {dict(ln.stats)}")
            del ls
        fills = pl.concat([pl.read_parquet(cache / f"{tag}_lane_{n.replace('|', '-')}.parquet") for n in names])
        results = market_results(fills["market"].unique().to_list(), cache / "vol_retest_results.json", fetch=fetch)
        unified.append(maker_metrics(fills, results, fee_mult))
        info["maker_lanes"] = names
        info["maker_fills"] = fills.height

    # momentum takers
    if "momentum" in parts:
        p = cache / f"{tag}_momentum.parquet"
        if not p.exists():
            _log("momentum: collecting tapes")
            tapes, last = collect(iter_events(files, defaultdict(int)))
            momentum_rows(tapes, last, split_ns, latencies, fee_info).write_parquet(p)
            del tapes
        mrows = pl.read_parquet(p)
        results = market_results(mrows["market"].unique().to_list(), cache / "vol_retest_results.json", fetch=fetch)
        unified.append(momentum_metrics(mrows, results))

    # arbitrage
    arb_prof = pl.DataFrame()
    if "arb" in parts:
        pe, pp = cache / f"{tag}_arb_episodes.parquet", cache / f"{tag}_arb_profile.parquet"
        if not (pe.exists() and pp.exists()):
            from albert.research.arb_l2 import build_sets, iter_stream

            _log("arb: event metadata and replay")
            recorded = set(buckets["market"].unique().to_list())
            events = event_meta(recorded, cache / "vol_retest_events.json", fetch=fetch)
            sets = build_sets(events, recorded, fee_mult)
            info["arb_sets"] = dict(pl.Series([x.kind for x in sets]).value_counts().iter_rows()) if sets else {}
            eps, prof = arb_scan(lambda: iter_stream(files), sets, latencies)
            eps.write_parquet(pe)
            prof.write_parquet(pp)
        eps, prof = pl.read_parquet(pe), pl.read_parquet(pp)
        unified.append(arb_metrics(eps, vol, latencies))
        arb_prof = arb_profile(prof, vol, split_ns)

    # lead-lag
    if "leadlag" in parts:
        p = cache / f"{tag}_leadlag.parquet"
        if not p.exists():
            from albert.backtest.l2 import load_pairs
            from albert.research.leadlag_fast import event_study, load, market_info, scheduled_starts

            pairs = load_pairs(root)
            frames = []
            if pairs:
                _log("lead-lag: loading paired books")
                info_m = market_info(sorted(pairs), cache / "vol_retest_markets.json")
                d = load([root], start, end, scheduled_starts(pairs, info_m))
                for leader in ("pm", "sibling"):
                    ev = event_study(d, 0.01, leader=leader)
                    if not ev.is_empty():
                        frames.append(ev.with_columns(pl.lit(leader).alias("leader")))
            (pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame({"leader": []}, schema={"leader": pl.Utf8})).write_parquet(p)
        lf = pl.read_parquet(p)
        moves = {ld: lf.filter(pl.col("leader") == ld) for ld in ("pm", "sibling")} if lf.height else {}
        unified.append(leadlag_metrics(moves, latencies))

    df = pl.concat([label(u, buckets, split_ns, category_of) for u in unified if u.height], how="diagonal_relaxed")
    long = summarize(df)
    grp = summarize(df, by_group=True)
    bands = summarize(df.with_columns(spread_band().alias("band")), by="band")
    info["minutes"] = round((time.time() - t_start) / 60, 1)
    info["rows"] = df.height

    rep = Report(s.reports_dir, "research-vol-retest")
    rep.json("Run", info)
    rep.table("Volatile buckets by nature (shared definition)",
              buckets.with_columns(pl.col("market").map_elements(
                  lambda m: nature(series_of(m), category_of.get(series_of(m))), return_dtype=pl.Utf8).alias("group"))
              .group_by("group").agg(pl.len().alias("buckets"), pl.col("volatile").sum().alias("volatile"),
                                     pl.col("market").n_unique().alias("markets"))
              .sort("volatile", descending=True), max_rows=20)
    for lat in latencies:
        rep.table(f"Summary at {lat:g} ms: cents per contract, mean (t by event); ** = t >= 2 on dev and holdout",
                  wide(long, lat), save_as=f"summary_{int(lat)}", max_rows=500)
    rep.table("All cells (long)", long, save_as="summary_long", max_rows=0)
    lat0 = latencies[0]
    rep.table(f"Volatile vs calm sign check at {lat0:g} ms (cents)", sign_changes(long, lat0), max_rows=500)
    rep.table("Decision-time gating (previous bucket volatile / calm), regime all", gate_table(long), max_rows=500)
    key = ["quoter_base", "join_best", "behind20", "mom3", "im_flicker", "tk_jump_fade", "tk_run_follow"]
    rep.table(f"By nature at {lat0:g} ms (key strategies)",
              wide(grp.filter(pl.col("strategy").is_in(key)), lat0, ("mo", "mx", "net")),
              save_as="by_nature", max_rows=500)
    rep.table("All by-nature cells (long)", grp, save_as="by_nature_long", max_rows=0)
    rep.table(f"Volatile vs calm within spread bands at {lat0:g} ms (makers; bucket median spread)",
              wide(bands.filter(pl.col("strategy").is_in(key[:5] + ["quoter_base|vol", "quoter_base|calm"])),
                   lat0, ("mo", "mx", "net")), save_as="by_spread", max_rows=500)
    rep.table("All by-spread cells (long)", bands, save_as="by_spread_long", max_rows=0)
    if arb_prof.height:
        rep.table("Arbitrage proximity per (set, 5-min bucket): best edge after fees", arb_prof, max_rows=100)
    rep.text("Conclusion (rule-based, regenerated each run)", conclusion(long, lat0, bands))
    return rep.save()
