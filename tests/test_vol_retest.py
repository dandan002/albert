"""Volatile-vs-calm re-test on trimmed recorded streams: regime labels, the decision-time gate,
lanes over every strategy spec, unified metrics, takers, arbitrage and the summary tables."""

import json
from pathlib import Path

import polars as pl
import pytest

from albert.backtest.l2 import Quoter, SimConfig, Simulator, _mark_executions, events_from_rows
from albert.research import vol_retest as vr
from albert.research.momentum_l2 import collect, taker_fee
from albert.research.volatile import from_events

FIX = Path(__file__).parent / "fixtures" / "kalshi"
B = vr.BUCKET_NS


def _events(name: str = "ws_stream_wide.jsonl"):
    rows = [json.loads(line) for line in (FIX / name).read_text().splitlines()]
    evs = events_from_rows(rows)
    _mark_executions(evs)
    return evs


def _lane(gate=None, latency_ms=50.0):
    q = Quoter(min_spread=0.0, lo=0.0, improve=True, min_requote_s=0)
    return vr.Lane("q", q, "market", Simulator(SimConfig(latency_ms=latency_ms)), gate)


def _replay(lane, evs):
    for ev in evs:
        lane.step(ev)
    return lane


def test_nature_groups_from_real_series():
    assert vr.nature("KXTRUMPSAY", "Mentions") == "trumpsay"
    assert vr.nature("KXRAINWKND", "Climate and Weather") == "rain"
    assert vr.nature("KXHIGHNY", "Climate and Weather") == "weather_hilo"
    assert vr.nature("KXLOWTDC", "Climate and Weather") == "weather_hilo"
    assert vr.nature("KXNBAGAME", "Sports") == "game"
    assert vr.nature("KXNBA3PT", "Sports") == "sports_props"
    assert vr.nature("KXGOOGVREQ", "Science and Technology") == "other"
    assert vr.event_of("KXHIGHDEN-26NOV05-B60.5") == "KXHIGHDEN-26NOV05"


def test_gate_reads_only_the_previous_bucket():
    flags = {("m", 10): True, ("m", 11): False}
    vol, calm = vr.Gate(flags, True), vr.Gate(flags, False)
    assert vol.open("m", 11 * B) and vol.open("m", 12 * B - 1)       # bucket 11 sees bucket 10
    assert not vol.open("m", 12 * B) and calm.open("m", 12 * B)      # bucket 12 sees bucket 11
    assert calm.open("m", 20 * B) and not vol.open("m", 20 * B)      # no row: calm
    assert calm.open("other", 11 * B)


def test_gate_flags_use_a_fixed_cut_and_in_play():
    df = pl.DataFrame({"market": ["a", "a", "a", "b"], "bucket_ns": [0, B, 2 * B, 0],
                       "path_c": [0.0, 5.0, 12.0, 0.0], "in_play": [False, False, False, True]})
    f = vr.gate_flags(df, 9.0)
    assert f == {("a", 0): False, ("a", 1): False, ("a", 2): True, ("b", 0): True}
    assert vr.dev_cut(df, split_ns=2 * B, q=0.8) == 5.0             # dev buckets only
    assert vr.persistence(f) == {"p_vol_after_vol": None, "p_vol_after_calm": 0.5}


def test_gated_lanes_on_the_recorded_stream():
    evs = _events()
    base = _replay(_lane(), evs)
    assert base.sim.orders and base.sim.fills
    # an always-open gate (no flags: every previous bucket is calm) changes nothing
    calm = _replay(_lane(vr.Gate({}, False)), evs)
    assert [(o.side, o.price, o.placed_ns) for o in calm.sim.orders.values()] == \
           [(o.side, o.price, o.placed_ns) for o in base.sim.orders.values()]
    # a volatile-only gate with no volatile bucket never quotes
    vol = _replay(_lane(vr.Gate({}, True)), evs)
    assert not vol.sim.orders and vol.stats["gated_events"] > 0
    # flag one bucket volatile: orders are placed only in the bucket after it
    mkt = evs[0].market
    b0 = sorted({e.recv_ns // B for e in evs})[1]
    lane = _replay(_lane(vr.Gate({(mkt, b0): True}, True)), evs)
    assert lane.sim.orders
    assert all(o.placed_ns // B == b0 + 1 for o in lane.sim.orders.values())
    # once the gate closes, live orders are pulled at the market's next event
    later = [e for e in evs if e.recv_ns // B > b0 + 1]
    if later:
        assert not [o for o in lane.sim.orders.values() if o.status in ("pending", "active")
                    and o.placed_ns < later[0].recv_ns - 10**9]


def test_fast_simulator_matches_the_simulator():
    evs = _events() + _events("ws_stream_trades.jsonl") + _events("ws_stream_ladder_arb.jsonl")
    evs.sort(key=lambda e: e.recv_ns)
    a = _lane()
    b = vr.Lane("q", Quoter(min_spread=0.0, lo=0.0, improve=True, min_requote_s=0), "market",
                vr.FastSimulator(SimConfig(latency_ms=50.0)))
    _replay(a, evs)
    _replay(b, evs)
    assert a.sim.fills and dict(a.sim.quotes) == dict(b.sim.quotes)
    assert [(o.market, o.side, o.price, o.placed_ns, o.status) for o in a.sim.orders.values()] == \
           [(o.market, o.side, o.price, o.placed_ns, o.status) for o in b.sim.orders.values()]
    assert [(f.t_ns, f.side, f.price, f.qty, f.how) for f in a.sim.fills] == \
           [(f.t_ns, f.side, f.price, f.qty, f.how) for f in b.sim.fills]
    for m in a.sim.markets:
        for side in ("yes", "no"):
            assert a.sim.markets[m].best(side) == b.sim.markets[m].best(side)


def test_every_spec_runs_on_the_recorded_stream():
    evs = _events()
    names = vr.lane_names((50.0,))
    assert len(names) == 12 + 2 * len(vr.GATED)
    lanes = [vr.make_lane(n, {}, {}, {}) for n in names]
    vr.run_lanes(evs, lanes)
    assert all(ln.sim.synced for ln in lanes)
    for ln in lanes:            # the trimmed stream is too short for fills at the studies' settings
        assert (len(ln.sim.orders) == 0) if "|vol" in ln.name else (len(ln.sim.orders) > 0), ln.name
        assert vr.lane_fills(ln).schema == pl.Schema(vr.FILL_SCHEMA)


def test_maker_metrics_net_and_hold():
    lane = _replay(_lane(), _events())
    fills = vr.lane_fills(lane).with_columns(pl.lit("quoter_base@50").alias("lane"))
    assert fills.height > 0
    mkt = fills["market"][0]
    u = vr.maker_metrics(fills, {mkt: 1.0}, {})
    assert u.height == fills.height and (u["strategy"] == "quoter_base").all()
    for f, r in zip(fills.iter_rows(named=True), u.iter_rows(named=True)):
        fee_pc = f["fee"] / f["qty"]
        if f["mx_300s"] is not None:
            assert r["net"] == pytest.approx(f["mx_300s"] - fee_pc - taker_fee(f["mx_300s"] + f["price"]))
        settle = 1.0 if f["side"] == "yes" else 0.0
        assert r["hold"] == pytest.approx(settle - f["price"] - fee_pc) and r["settled"]
    # unsettled: marked to the last recorded mid
    u2 = vr.maker_metrics(fills, {}, {})
    f = fills.row(0, named=True)
    last = f["last_mid"] if f["side"] == "yes" else 1 - f["last_mid"]
    assert u2["hold"][0] == pytest.approx(last - f["price"] - f["fee"] / f["qty"])


def test_label_uses_each_rows_own_bucket():
    evs = _events()
    buckets = from_events(evs, bucket_s=300)
    assert buckets.height > 1 and buckets["volatile"].any()
    mkt = evs[0].market
    rows = pl.DataFrame({"market": [mkt] * buckets.height + [mkt],
                         "t_ns": buckets["bucket_ns"].to_list() + [10**18]}).with_columns(pl.col("t_ns").cast(pl.Int64))
    rows = rows.with_columns((pl.col("t_ns") + 1).alias("t_ns"))
    out = vr.label(rows, buckets, split_ns=int(buckets["bucket_ns"].median()), category_of={})
    assert out["volatile"].to_list()[:-1] == buckets["volatile"].to_list()
    assert out["volatile"][-1] is False                                  # no bucket row: calm
    assert out["spread_c"].to_list()[:-1] == buckets["spread_med_c"].to_list() and out["spread_c"][-1] is None
    assert set(out["split"]) == {"dev", "hold"} and set(out["group"]) == {"weather_hilo"}


def test_momentum_takers_on_recorded_tapes():
    evs = _events() + _events("ws_stream_trades.jsonl")
    evs.sort(key=lambda e: e.recv_ns)
    tapes, end = collect(evs)
    split = sorted(e.recv_ns for e in evs)[len(evs) // 2]
    rows = vr.momentum_rows(tapes, end, split, (50.0, 100.0), {})
    assert rows.height > 0 and set(rows["latency"]) == {50.0, 100.0}
    assert rows["strategy"].str.starts_with("tk_").all()
    assert (rows.filter(pl.col("net").is_not_null())["net"] < rows.filter(pl.col("net").is_not_null())["mx"] + 1e-9).all()
    res = {m: 1.0 for m in rows["market"].unique()}
    u = vr.momentum_metrics(rows, res)
    r = u.filter(pl.col("hold").is_not_null()).row(0, named=True)
    raw = rows.filter((pl.col("market") == r["market"]) & (pl.col("t_ns") == r["t_ns"])
                      & (pl.col("strategy") == r["strategy"]) & (pl.col("latency") == r["latency"])).row(0, named=True)
    side = 1.0 if raw["dir"] > 0 else 0.0
    assert r["hold"] == pytest.approx(side - raw["entry"] - taker_fee(raw["entry"]))


def test_parse_event_from_recorded_payload():
    e = vr.parse_event(json.loads((FIX / "event.json").read_text()))
    assert e["me"] is True and len(e["legs"]) == 2
    assert all(leg["t"].startswith("KXNBAGAME-26NOV02MIALAL") and leg["close_time"] for leg in e["legs"])


def test_arb_scan_profile_on_recorded_ladder():
    from albert.research.arb_l2 import Event, build_sets

    kinds = {"orderbook_snapshot": "snapshot", "orderbook_delta": "delta", "trade": "trade"}

    def stream():
        for line in (FIX / "ws_stream_ladder_arb.jsonl").read_text().splitlines():
            r = json.loads(line)
            d = json.loads(r["frame"])
            yield Event(r["recv_ns"], kinds[d["type"]], d["msg"]["market_ticker"], d["msg"], "k0")

    ladder = {"KXTOKENUSE-26NOV09": {"me": False, "legs": [
        {"t": "KXTOKENUSE-26NOV09-T207", "strike_type": "greater", "floor": 207},
        {"t": "KXTOKENUSE-26NOV09-T210", "strike_type": "greater", "floor": 210}]}}
    sets = build_sets(ladder, {leg["t"] for leg in ladder["KXTOKENUSE-26NOV09"]["legs"]})
    eps, prof = vr.arb_scan(stream, sets, (50, 100))
    assert eps.height == 1 and eps["pnl_50"][0] < 0
    assert prof.height > 0 and prof["best_edge"].max() > 0
    t0 = eps["t_ns"][0]
    vol = {("KXTOKENUSE-26NOV09-T210", t0 // B): True}
    u = vr.arb_metrics(eps, vol, (50.0, 100.0))
    assert u.height == 2 and u["volatile"].all() and (u["net"] < 0).all()
    p = vr.arb_profile(prof, vol, split_ns=10**19)
    hit = p.filter((pl.col("regime") == "volatile") & (pl.col("split") == "dev"))
    assert hit["set_buckets"][0] == 1 and hit["arb_share"][0] == 1.0


def test_leadlag_rows_keep_takeable_moves():
    f = pl.DataFrame({"game": ["g1", "g2"], "market": ["KXNBAGAME-X-A", "KXNBAGAME-Y-B"], "t_ns": [1, 2],
                      "resp_10s": [0.01, -0.01], "edge_50ms": [0.0, -0.02], "pnl_50ms_10s": [-0.04, -0.05],
                      "edge_100ms": [0.01, 0.0], "pnl_100ms_10s": [-0.03, None]})
    u = vr.leadlag_metrics({"pm": f}, (50.0, 100.0))
    assert u.select("latency", "net").rows() == [(50.0, -0.04), (100.0, -0.03)]


def test_summary_flags_and_tables():
    rng = list(range(40))
    df = pl.DataFrame({
        "strategy": ["s"] * 40, "latency": [50.0] * 40, "market": [f"E{i % 8}-M" for i in rng],
        "t_ns": rng, "qty": [1.0] * 40, "cluster": [f"E{i % 8}" for i in rng],
        "mo": [0.02 + 0.001 * (i % 5) for i in rng], "mx": [-0.03 + 0.001 * (i % 3) for i in rng],
        "net": [0.01 * ((-1) ** i) for i in rng], "hold": [None] * 40, "settled": [False] * 40,
        "volatile": [i % 4 == 0 for i in rng], "group": ["rain"] * 40,
        "split": ["dev" if i < 20 else "hold" for i in rng]}, schema_overrides={"hold": pl.Float64})
    long = vr.summarize(df)
    assert long.height == 6 and set(long["regime"]) == {"all", "volatile", "calm"}
    fl = vr.flags(long).filter(pl.col("regime") == "all").row(0, named=True)
    assert "mo" in fl["pos"] and "mx" in fl["neg"]
    w = vr.wide(long, 50.0)
    assert w.height == 3 and w.filter(pl.col("regime") == "all")["mo dev"][0].endswith("**")
    assert vr.summarize(df, by_group=True)["regime"].unique().sort().to_list() == ["calm:rain", "volatile:rain"]
    sc = vr.sign_changes(long, 50.0)
    assert set(sc["metric"]) == {"mo", "mx", "net", "hold"}
    banded = df.with_columns(pl.Series("spread_c", [2.0, 5.0, 9.0, None] * 10)).with_columns(
        vr.spread_band().alias("band"))
    assert set(banded["band"]) == {"spread<=3c", "spread 4-6c", "spread>6c", "spread ?"}
    sb = vr.summarize(banded, by="band")
    assert "volatile:spread<=3c" in set(sb["regime"]) and sb["spread_c_med"].drop_nulls().len() > 0
    assert "Baseline within spread bands" not in vr.conclusion(long, 50.0, sb)    # no quoter_base rows
    assert "t >= 2" in vr.conclusion(long, 50.0)
    mean, t, k = vr.clustered_t(df.with_columns(pl.lit(0.01).alias("mo")), "mo")
    assert mean == pytest.approx(0.01) and t is None and k == 8         # no variation: no t
