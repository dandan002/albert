"""Who moves first: Polymarket intl vs Kalshi on the same game moneyline (recorded streams).

For each paired market (``albert record --paired-days``), a *move* on one venue is a mid change of
at least ``move`` within ``window_s`` seconds (de-clustered: one move per ``gap_s``). For every
move we measure the other venue's mid change, signed in the move's direction, over the
``window_s`` before the move (did it already move?) and at each horizon after it (does it
follow?). Times are our receive times on both feeds, so a lead that is shorter than the
difference in our network paths cannot be told apart from noise; leads of seconds can.
Standard errors are clustered by game.
"""

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl

from albert.backtest.l2 import (
    SimConfig,
    Simulator,
    iter_events,
    load_pairs,
    pm_events,
    pm_files,
    stream_files,
)

HORIZONS_S = (1, 5, 30, 120, 600)


class _Null:
    def on_event(self, sim, market) -> None:
        pass


def _moves(times: list[int], mids: list[float], move: float, window_ns: int,
           gap_ns: int) -> list[tuple[int, float]]:
    """(t, direction) for each move of at least ``move`` over the trailing window."""
    out: list[tuple[int, float]] = []
    j = 0
    last = -(2**62)
    for i, t in enumerate(times):
        while times[j] < t - window_ns:
            j += 1
        lo, hi = min(mids[j:i + 1]), max(mids[j:i + 1])
        if t - last < gap_ns:
            continue
        if mids[i] - lo >= move - 1e-9 and mids[i] > mids[j]:
            out.append((t, 1.0))
            last = t
        elif hi - mids[i] >= move - 1e-9 and mids[i] < mids[j]:
            out.append((t, -1.0))
            last = t
    return out


def _clustered(frame: pl.DataFrame, col: str) -> tuple[float | None, float | None, int]:
    f = frame.filter(pl.col(col).is_not_null())
    if f.is_empty():
        return None, None, 0
    g = f.group_by("game").agg(pl.col(col).sum().alias("s"), pl.len().alias("n"))
    n = g["n"].sum()
    mean = g["s"].sum() / n
    k = g.height
    if k < 2:
        return mean, None, k
    se = math.sqrt(k / (k - 1) * ((g["s"] - mean * g["n"]) ** 2).sum()) / n
    return mean, (mean / se if se > 0 else None), k


def lead_lag_frame(sim: Simulator, pairs: dict[str, dict[str, Any]], *, move: float,
                   window_s: float, gap_s: float, end_ns: int) -> pl.DataFrame:
    win, gap = int(window_s * 1e9), int(gap_s * 1e9)
    rows = []
    for ticker, pair in pairs.items():
        k_times, k_vals = sim.quotes.get(ticker, ([], []))
        k_mid = [((b + a) / 2 if b is not None and a is not None else None) for b, a in k_vals]
        kt = [t for t, m in zip(k_times, k_mid) if m is not None]
        km = [m for m in k_mid if m is not None]
        pt, pm = sim.ext.get(ticker, ([], []))
        if not kt or not pt:
            continue
        k_at = lambda t: sim.mid_at(ticker, t)
        p_at = lambda t: sim.ext_at(ticker, t)
        for leader, (times, mids), other in (("pm", (pt, pm), k_at), ("kalshi", (kt, km), p_at)):
            for t, d in _moves(times, mids, move, win, gap):
                base = other(t)
                before = other(t - win)
                if base is None:
                    continue
                row = {"leader": leader, "market": ticker, "game": pair.get("pm_slug", ticker),
                       "t_ns": t, "dir": d,
                       "other_before": None if before is None else d * (base - before)}
                for h in HORIZONS_S:
                    th = t + h * 10**9
                    v = other(th) if th <= end_ns else None
                    row[f"other_{h}s"] = None if v is None else d * (v - base)
                rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def lead_lag(s, *, start: str | None, end: str | None, move: float = 0.02, window_s: float = 5.0,
             gap_s: float = 30.0) -> Path:
    from albert.reports import Report

    root = s.raw_dir / "stream"
    pairs = load_pairs(root)
    if not pairs:
        raise RuntimeError("no pairs recorded; run `albert record --paired-days N` first")
    kfiles, pfiles = stream_files(root, start, end), pm_files(root, start, end)
    if not kfiles or not pfiles:
        raise RuntimeError("no recorded Kalshi and Polymarket stream files in range")
    sim = Simulator(SimConfig(writes_per_s=None))
    stats: dict[str, int] = defaultdict(int)
    import heapq

    events = heapq.merge((e for e in iter_events(kfiles, stats) if e.market in pairs),
                         pm_events(pfiles, pairs), key=lambda e: e.recv_ns)
    end_ns = 0
    for ev in events:
        sim.apply(ev)
        end_ns = ev.recv_ns
    frame = lead_lag_frame(sim, pairs, move=move, window_s=window_s, gap_s=gap_s, end_ns=end_ns)
    rep = Report(s.reports_dir, "research-lead-lag")
    covered = [t for t in pairs if sim.quotes.get(t) and sim.ext.get(t, ([], []))[0]]
    rep.json("Run", {"kalshi_files": len(kfiles), "pm_files": len(pfiles), "pairs": len(pairs),
                     "pairs_with_both_feeds": len(covered), "move": move, "window_s": window_s,
                     "gap_s": gap_s, "moves": frame.height})
    if frame.is_empty():
        rep.text("Moves", "No moves of that size on either venue.")
        return rep.save()
    summary = []
    for leader in ("pm", "kalshi"):
        f = frame.filter(pl.col("leader") == leader)
        for col in ["other_before"] + [f"other_{h}s" for h in HORIZONS_S]:
            mean, t, k = _clustered(f, col)
            summary.append({"leader": leader, "other_venue_change": col, "moves": f.height,
                            "mean_cents": None if mean is None else round(mean * 100, 2),
                            "t_by_game": t, "games": k})
    rep.table("Other venue's mid change in the move's direction (cents)", pl.DataFrame(summary))
    rep.table("Moves", frame.drop("t_ns"), save_as="moves", max_rows=30)
    return rep.save()
