"""In-play rerun of the two fast, risk-neutral studies on last night's recording.

    python -m albert.research.inplay_hft arb      -> arb_l2 episodes + proximity, split in-play / pre-game
    python -m albert.research.inplay_hft leadlag  -> leadlag_fast.fast_lead_lag report (splits by regime itself)

Window START <= hour < END (UTC). Uses the research modules unchanged; writes to data/research/inplay_hft/.
"""
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import polars as pl

START, END = "2026-11-07T17", "2026-11-08T06"
OUT = Path("data/research/inplay_hft")
LATS = (10, 25, 50, 100, 250, 500)
BUCKET_NS = 300 * 10**9


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


# ---- streaming readers (same order as the originals, without holding an hour in memory) -------
import heapq
import io
import re

import zstandard

MKT = re.compile(r'"market_ticker":"([^"]+)"')
KINDS = {"orderbook_snapshot": "snapshot", "orderbook_delta": "delta", "trade": "trade"}


def lazy_rows(path):
    """``albert.stream.read_stream`` one line at a time."""
    with open(path, "rb") as f:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8"):
            if line.strip():
                yield json.loads(line)


def _by_hour(files):
    by_hour = defaultdict(list)
    for p in files:
        by_hour[f"{p.parent.name}T{p.name[:2]}"].append(p)
    return [by_hour[h] for h in sorted(by_hour)]


def _kalshi(path, keep):
    """(recv_ns, kind, market, msg) for ``keep`` markets (None = all) from one Kalshi file."""
    for row in lazy_rows(path):
        fr = row["frame"]
        m = MKT.search(fr)
        if m is None or (keep is not None and m.group(1) not in keep):
            continue
        d = json.loads(fr)
        kind = KINDS.get(d.get("type"))
        if kind:
            yield row["recv_ns"], kind, d["msg"]["market_ticker"], d["msg"]


def arb_stream(files, keep):
    """``arb_l2.iter_stream`` restricted to ``keep`` markets, merged lazily."""
    from albert.research.arb_l2 import Event

    def one(p):
        conn = p.name.split(".")[0][3:].split("-")[0]
        for t, k, m, msg in _kalshi(p, keep):
            yield Event(t, k, m, msg, conn)

    for hour_files in _by_hour(files):
        yield from heapq.merge(*[one(p) for p in hour_files], key=lambda e: e.recv_ns)


def markets_in(files):
    out = set()
    for p in files:
        with open(p, "rb") as f:
            for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8"):
                m = re.search(r'market_ticker\\":\\"([^"\\]+)', line)
                if m:
                    out.add(m.group(1))
    return out


def patch_leadlag(keep):
    """Make ``leadlag_fast.load`` stream: Polymarket rows lazily, Kalshi only ``keep`` markets
    (execution labelling is per market, so filtering first gives the same labels)."""
    import albert.research.leadlag_fast as lf
    from albert.backtest import l2

    def iter_events(files, stats):
        for hour_files in _by_hour(files):
            streams = []
            for p in hour_files:
                evs = [l2.Event(t, k, m, msg) for t, k, m, msg in _kalshi(p, keep)]
                t, mm = l2._mark_executions(evs)
                stats["trades"] += t
                stats["trades_matched"] += mm
                streams.append(evs)
            yield from heapq.merge(*streams, key=lambda e: e.recv_ns)

    lf.iter_events = iter_events
    lf.read_stream = lazy_rows


def arb():
    from albert.backtest.l2 import stream_files
    from albert.config import settings
    from albert.research.arb_l2 import Books, build_sets, replay, set_edge
    from albert.research.vol_retest import event_meta
    from albert.research.volatile import _in_play_bounds, market_meta

    s = settings()
    root = s.raw_dir / "stream"
    files = stream_files(root, START, END)
    log(f"{len(files)} Kalshi files")
    recorded = markets_in(files)
    log(f"{len(recorded)} recorded markets")
    ser = pl.read_parquet(s.tables_dir / "kalshi_series.parquet")
    fee_mult = {r["series_ticker"]: float(r.get("fee_multiplier") or 1.0) for r in ser.iter_rows(named=True)}
    events = event_meta(recorded, s.data_dir / "cache" / "vol_retest_events.json")
    sets = build_sets(events, recorded, fee_mult)
    kinds = defaultdict(int)
    for x in sets:
        kinds[x.kind] += 1
    log(f"{len(sets)} sets {dict(kinds)}")
    bounds = _in_play_bounds(market_meta(sorted(recorded), s.data_dir / "cache" / "volatile_markets.json"))

    def in_play(legs, t):
        return any(lo <= t <= hi for m in legs for lo, hi in [bounds.get(m, (1, 0))])

    legs_of = {x.key: [m for m, _ in x.legs] for x in sets}
    keep = {m for legs in legs_of.values() for m in legs}
    log(f"{len(keep)} markets in sets")
    eps, diag = replay(arb_stream(files, keep), sets, latencies_ms=LATS)
    log(f"{len(eps)} episodes, diag {dict(diag)}")
    rows = []
    for e in eps:
        r = {"key": e.key, "kind": e.kind, "event": e.event, "start_ns": e.start_ns,
             "dur_ms": None if e.end_ns is None else (e.end_ns - e.start_ns) / 1e6,
             "edge0_c": 100 * e.edge0, "best_edge_c": 100 * e.best_edge, "size0": e.size0,
             "best_dollars": e.best_dollars, "trades_on_legs": e.trades_on_legs, "same_conn": e.same_conn,
             "in_play": in_play(legs_of[e.key], e.start_ns)}
        for lat in LATS:
            x = e.latency.get(lat, {})
            r[f"pnl_{lat}"] = x.get("pnl")
            r[f"all_{lat}"] = bool(x.get("filled")) and all(x["filled"])
            r[f"part_{lat}"] = bool(x.get("filled")) and any(x["filled"]) and not all(x["filled"])
        rows.append(r)
    ep = pl.DataFrame(rows, infer_schema_length=None)
    ep.write_parquet(OUT / "inplay_arb_episodes.parquet")

    # proximity: best edge after fees per (set, 5-min bucket)
    books = Books()
    by_market = defaultdict(list)
    for x in sets:
        for m, _ in x.legs:
            by_market[m].append(x)
    best: dict[tuple[str, int], float] = {}
    for ev in arb_stream(files, keep):
        if ev.kind == "trade" or ev.market not in by_market:
            continue
        books.apply(ev.kind, ev.market, ev.msg)
        if books.crossed(ev.market):
            continue
        b = ev.recv_ns // BUCKET_NS
        for x in by_market[ev.market]:
            r = set_edge(x, books, ev.recv_ns)
            if r is not None and r[0] > best.get((x.key, b), -math.inf):
                best[(x.key, b)] = r[0]
    kind_of = {x.key: x.kind for x in sets}
    prof = pl.DataFrame([{"key": k, "kind": kind_of[k], "bucket": b, "best_edge_c": 100 * v,
                          "in_play": in_play(legs_of[k], b * BUCKET_NS + BUCKET_NS // 2)}
                         for (k, b), v in best.items()])
    prof.write_parquet(OUT / "inplay_arb_profile.parquet")

    out = {"window": [START, END], "files": len(files), "recorded": len(recorded), "sets": dict(kinds),
           "in_play_markets": sum(1 for m in recorded if m in bounds), "diag": dict(diag), "episodes": {}, "proximity": {}}
    for kind in sorted(set(kind_of.values())):
        for reg in (True, False):
            g = ep.filter((pl.col("kind") == kind) & (pl.col("in_play") == reg)) if ep.height else ep
            key = f"{kind}|{'in_play' if reg else 'pre_game'}"
            if g.height:
                d = g["dur_ms"].drop_nulls()
                row = {"episodes": g.height, "events": g["event"].n_unique(),
                       "dur_ms_median": d.median(), "dur_ms_p90": d.quantile(0.9),
                       "edge0_c_median": g["edge0_c"].median(), "best_edge_c_max": g["best_edge_c"].max(),
                       "best_dollars_sum": g["best_dollars"].sum(), "taken_by_trade_share": (g["trades_on_legs"] > 0).mean(),
                       "same_conn_share": g["same_conn"].mean()}
                for lat in LATS:
                    row[f"L{lat}"] = {"all_filled": int(g[f"all_{lat}"].sum()), "partial": int(g[f"part_{lat}"].sum()),
                                      "pnl_1lot_c": round(100 * g[f"pnl_{lat}"].drop_nulls().sum(), 2)}
                out["episodes"][key] = row
            p = prof.filter((pl.col("kind") == kind) & (pl.col("in_play") == reg))
            if p.height:
                out["proximity"][key] = {"set_buckets": p.height, "sets": p["key"].n_unique(),
                                         "best_edge_c_median": round(p["best_edge_c"].median(), 2),
                                         "best_edge_c_p99": round(p["best_edge_c"].quantile(0.99), 2),
                                         "best_edge_c_max": round(p["best_edge_c"].max(), 2),
                                         "within_1c_share": round((p["best_edge_c"] >= -1).mean(), 4),
                                         "arb_share": round((p["best_edge_c"] > 1e-7).mean(), 4)}
    (OUT / "inplay_arb.json").write_text(json.dumps(out, indent=1, default=str))
    log("arb done")
    print(json.dumps(out, indent=1, default=str))


def leadlag():
    from albert.backtest.l2 import load_pairs
    from albert.config import settings
    from albert.research.leadlag_fast import fast_lead_lag

    s = settings()
    patch_leadlag(set(load_pairs(s.raw_dir / "stream")))
    p = fast_lead_lag(s, start=START, end=END)
    log(f"lead-lag report {p}")
    (OUT / "inplay_leadlag.txt").write_text(str(p))


LL_MOVES = (0.01, 0.02, 0.03)


def ll_chunk(start, end):
    """One hour-aligned chunk (every hour opens with fresh snapshots): leadlag_fast's own book
    builders and analyses, minus the feed-path lists and the second copy ``load`` makes."""
    import albert.research.leadlag_fast as lf
    from albert.backtest.l2 import load_pairs, pm_files, stream_files
    from albert.config import settings

    s = settings()
    root = s.raw_dir / "stream"
    pairs = load_pairs(root)
    patch_leadlag(set(pairs))
    info = lf.market_info(sorted(pairs), s.data_dir / "cache" / "leadlag_fast_markets.json")
    kf = stream_files(root, start, end)
    hours = {f"{f.parent.name}T{f.name[:2]}" for f in kf}
    pf = [f for f in pm_files(root, start, end) if f"{f.parent.name}T{f.name[:2]}" in hours]
    log(f"chunk {start}..{end}: {len(kf)} Kalshi, {len(pf)} PM files, {len(pairs)} pairs")
    kb, tr = lf.kalshi_books(lf.iter_events(kf, defaultdict(int)), set(pairs), None)
    log(f"  kalshi books {sum(len(b.t) for b in kb.values())} tops, {len(tr)} trades")
    pb = lf.pm_books(heapq.merge(*[lazy_rows(f) for f in pf], key=lambda x: x["recv_ns"]), pairs, None)
    log(f"  pm books {sum(len(b.t) for b in pb.values())} tops")
    end_ns = max((b.t[-1] for b in kb.values() if b.t), default=0)
    d = lf.Data(pairs, dict(kb), dict(pb), tr, [], [], lf.scheduled_starts(pairs, info), end_ns)
    tag = f"{start}_{end}"
    for k in LL_MOVES:
        for leader in ("pm", "sibling"):
            lf.event_study(d, k, leader=leader).write_parquet(OUT / f"ll_{tag}_{leader}_{int(k * 100)}.parquet")
    lf.maker_defence(d).write_parquet(OUT / f"ll_{tag}_defence.parquet")
    lf.xcorr(d).write_parquet(OUT / f"ll_{tag}_xcorr.parquet")
    log("  chunk done")


def ll_agg():
    """Pool the chunks' per-move rows and summarize with leadlag_fast's own functions."""
    import albert.research.leadlag_fast as lf
    from albert.config import settings
    from albert.reports import Report

    # move tables: every chunk that wrote them, minus the 02-06 run superseded by 02-04 + 04-06
    tags = sorted({p.name[3:].split("_pm_1")[0] for p in OUT.glob("ll_*_pm_1.parquet")}
                  - {"2026-11-08T02_2026-11-08T06"})
    xtags = [t for t in tags if (OUT / f"ll_{t}_xcorr.parquet").exists()]
    rep = Report(settings().reports_dir, "research-lead-lag-fast-inplay")
    rep.json("Run", {"window": [START, END], "move_chunks": tags, "defence_xcorr_chunks": xtags,
                     "note": "leadlag_fast analyses on hour-aligned chunks (each hour opens with fresh "
                             "snapshots); per-move rows pooled; feed-path stats skipped; chunks missing "
                             "from defence_xcorr_chunks ran out of memory after writing their move tables"})
    out = {"chunks": tags, "defence_xcorr_chunks": xtags}
    for k in LL_MOVES:
        for leader in ("pm", "sibling"):
            fs = [pl.read_parquet(p) for t in tags if (p := OUT / f"ll_{t}_{leader}_{int(k * 100)}.parquet").exists()]
            fs = [f for f in fs if f.height]
            f = pl.concat(fs, how="diagonal_relaxed") if fs else pl.DataFrame()
            name = f"{leader} move >= {int(k * 100)}c"
            if f.is_empty():
                out[name] = "no moves"
                continue
            ev, tk = lf.summarize_events(f), lf.summarize_taker(f)
            out[name] = {"events": ev, "taker": tk}
            rep.table(f"{name}: Kalshi response (c, t by game)", pl.DataFrame(ev, infer_schema_length=None))
            rep.table(f"{name}: taker on the stale Kalshi quote (net c/contract)", pl.DataFrame(tk, infer_schema_length=None))
            rep.table(f"Moves {name}", f.drop("t_ns", strict=False), save_as=f"moves_{leader}_{int(k * 100)}c", max_rows=10)
    md = [pl.read_parquet(p) for t in tags if (p := OUT / f"ll_{t}_defence.parquet").exists()]
    md = [f for f in md if f.height]
    if md:
        dfc = lf.summarize_defence(pl.concat(md, how="diagonal_relaxed"))
        out["maker_defence"] = dfc
        rep.table("Maker markout of Kalshi trades by preceding Polymarket move (c, t by game)",
                  pl.DataFrame(dfc, infer_schema_length=None))
    xs = []
    for t in xtags:
        x = pl.read_parquet(OUT / f"ll_{t}_xcorr.parquet")
        if x.height:
            xs.append(x.with_columns(pl.lit(t).alias("chunk")))
    if xs:
        x = pl.concat(xs)
        rep.table("Cross-correlation of 1 s mid changes per chunk (positive lag = Polymarket leads)",
                  x.filter(pl.col("lag_s").is_between(-5, 10)), max_rows=200)
        out["xcorr_max_abs"] = x.group_by("chunk").agg(pl.col("corr").abs().max()).to_dicts()
        out["xcorr_lag0_to_5"] = x.filter(pl.col("lag_s").is_between(0, 5)).to_dicts()
    path = rep.save()
    (OUT / "inplay_leadlag.json").write_text(json.dumps(out, indent=1, default=str))
    log(f"report {path}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    # optional window/output/latency overrides: arb START END [OUT] [LAT,LAT,...]
    if cmd == "arb" and len(sys.argv) >= 4:
        START, END = sys.argv[2], sys.argv[3]
        if len(sys.argv) > 4:
            OUT = Path(sys.argv[4])
        if len(sys.argv) > 5:
            LATS = tuple(int(x) for x in sys.argv[5].split(","))
        OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "llchunk":
        ll_chunk(sys.argv[2], sys.argv[3])
    else:
        {"arb": arb, "leadlag": leadlag, "llagg": ll_agg}[cmd]()
