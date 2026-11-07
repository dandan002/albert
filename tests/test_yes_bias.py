import json
from pathlib import Path

import polars as pl
import pytest

from albert.backtest.l2 import SimConfig, _mark_executions, events_from_rows
from albert.research.yes_bias import (
    LaneSim, OneSided, PrintParams, aggregate, band_of, cell_stats, maker_fee, parse_event,
    print_maker_market, structure_of)
from albert.venues import kalshi as k

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _prints():
    """Recorded in-play MLB trades page (newest first in the payload) in time order."""
    rows = k.trade_rows(json.loads((FIX / "trades_inplay_mlb.json").read_text()))
    rows.sort(key=lambda r: r["created_time"])
    t = [r["created_time"].timestamp() for r in rows]
    return rows, t, [r["yes_price"] for r in rows], [r["taker_side"] == "yes" for r in rows], [r["count"] for r in rows]


def _events(name: str):
    rows = [json.loads(line) for line in (FIX / name).read_text().splitlines()]
    evs = events_from_rows(rows)
    _mark_executions(evs)
    return evs


PERMISSIVE = dict(min_spread=0.01, lo=0.0, hi=1.0, ttl_s=120.0, cooldown_s=30.0, cap_market=50.0)


@pytest.mark.parametrize("side", ["sell_yes", "buy_yes"])
def test_print_maker_fills_only_on_later_prints_through_or_crossing(side):
    _, t, p, yes, q = _prints()
    prm = PrintParams(side=side, **PERMISSIVE)
    fills = print_maker_market(t, p, yes, q, prm)
    assert len(fills) >= 5
    for f in fills:
        assert f["t"] > f["t_place"] + prm.latency_s
        hits = [i for i in range(len(t)) if t[i] == f["t"]]
        ok = False
        for i in hits:
            if side == "sell_yes":
                ok |= (yes[i] and p[i] > f["px"]) or (not yes[i] and p[i] >= f["px"])
            else:
                ok |= (not yes[i] and p[i] < f["px"]) or (yes[i] and p[i] <= f["px"])
        assert ok
        # the quote was one tick inside the print-implied spread
        assert f["spread"] >= prm.min_spread - 1e-9
    assert sum(f["qty"] for f in fills) <= prm.cap_market + 1e-9


def test_print_maker_decisions_do_not_depend_on_future_prints():
    _, t, p, yes, q = _prints()
    prm = PrintParams(**PERMISSIVE)
    full = print_maker_market(t, p, yes, q, prm)
    assert full
    cut = len(t) // 2
    part = print_maker_market(t[:cut], p[:cut], yes[:cut], q[:cut], prm)
    assert part == [f for f in full if f["t"] <= t[cut - 1]]


def test_flow_gate_only_removes_quotes():
    _, t, p, yes, q = _prints()
    base = print_maker_market(t, p, yes, q, PrintParams(**PERMISSIVE))
    gated = print_maker_market(t, p, yes, q, PrintParams(gate_window_s=300, gate_share=0.6, gate_min_q=1,
                                                         **PERMISSIVE))
    assert base and len(gated) < len(base)
    # every gated decision saw a YES share below the threshold
    assert all(f["flow_share"] is not None and f["flow_share"] < 0.6 for f in gated)


def _tiny(tmp_path: Path, rows, y: float, fee_type: str = "quadratic"):
    prints = pl.DataFrame({
        "mid": [0] * len(rows), "t": [r["created_time"] for r in rows], "q": [r["count"] for r in rows],
        "p": [r["yes_price"] for r in rows], "yes": [r["taker_side"] == "yes" for r in rows],
        "source": ["window"] * len(rows), "phase": ["inplay"] * len(rows),
        "h_exp": [1.0] * len(rows), "cum_ok": [True] * len(rows)},
        schema_overrides={"mid": pl.UInt32, "h_exp": pl.Float32})
    path = tmp_path / "prints.parquet"
    prints.write_parquet(path)
    markets = pl.DataFrame({"mid": [0], "ticker": [rows[0]["ticker"]], "event": ["E"], "series": ["KXMLBGAME"],
                            "category": ["Sports"], "cls": ["sports"], "split": ["dev"], "y": [y],
                            "fee_type": [fee_type], "fee_multiplier": [1.0]},
                           schema_overrides={"mid": pl.UInt32})
    return path, markets


@pytest.mark.parametrize("y", [0.0, 1.0])
def test_aggregate_maker_pnl_matches_direct_computation(tmp_path, y):
    rows, *_ = _prints()
    path, markets = _tiny(tmp_path, rows, y)
    a = aggregate(path, markets, ["band"])
    assert a.height > 0 and int(a["n"].sum()) == len(rows)
    st = cell_stats(a)
    yq = sum(r["count"] for r in rows if r["taker_side"] == "yes")
    ys = sum(r["count"] * (r["yes_price"] - y) for r in rows if r["taker_side"] == "yes")
    nq = sum(r["count"] for r in rows if r["taker_side"] == "no")
    ns = sum(r["count"] * (y - r["yes_price"]) for r in rows if r["taker_side"] == "no")
    assert st["sellY_c"] == pytest.approx(round(100 * ys / yq, 2), abs=0.011)
    assert st["buyY_c"] == pytest.approx(round(100 * ns / nq, 2), abs=0.011)
    assert st["asym_c"] == pytest.approx(round(100 * (ys / yq - ns / nq), 2), abs=0.011)
    assert st["contracts"] == pytest.approx(yq + nq, abs=0.1)


def test_maker_fees_only_on_maker_fee_series(tmp_path):
    rows, *_ = _prints()
    path, plain = _tiny(tmp_path, rows, 1.0)
    _, mf = _tiny(tmp_path, rows, 1.0, fee_type="quadratic_with_maker_fees")
    a0, a1 = aggregate(path, plain, ["band"]), aggregate(path, mf, ["band"])
    fee = sum(r["count"] * 0.0175 * r["yes_price"] * (1 - r["yes_price"]) for r in rows)
    assert float(a0["sm"].sum() - a1["sm"].sum()) == pytest.approx(fee, rel=1e-9)
    assert maker_fee(0.5, "quadratic", 1.0) == 0.0
    assert maker_fee(0.5, "quadratic_with_maker_fees", 1.0) == pytest.approx(0.004375)


def test_labels_and_structure():
    df = pl.DataFrame({"p": [0.0, 0.01, 0.05, 0.499, 0.5, 0.95, 0.99, 1.0]})
    bands = df.select(band_of(pl.col("p"))).to_series().to_list()
    assert bands == ["00-05", "00-05", "05-10", "45-50", "50-55", "95-100", "95-100", "95-100"]
    ev = parse_event(json.loads((FIX / "event.json").read_text()))
    assert ev["me"] is True and ev["n"] == 2
    assert structure_of(ev["n"], ev["me"], "KXNBAGAME") == "2-way"
    assert structure_of(1, None, "KXFOO") == "single"
    assert structure_of(8, False, "KXTRUMPSAY") == "multi-indep"
    assert structure_of(6, True, "KXHIGHNY") == "multi-ME"


@pytest.mark.parametrize("side", ["no", "yes"])
def test_one_sided_quoter_never_quotes_the_other_side(side):
    evs = _events("ws_stream_ladder_arb.jsonl")
    st = OneSided(side=side, improve=True, min_spread=0.02, cooldown_s=0.0, lo=0.0, hi=1.0)
    sim = LaneSim(SimConfig(latency_ms=50.0, writes_per_s=10.0), keep_quotes=True)
    for ev in evs:
        sim.apply(ev)
        st.observe(sim, ev)
        st.on_event(sim, ev.market)
    assert sim.orders and {o.side for o in sim.orders.values()} == {side}


def test_one_sided_flow_gate_pulls_on_adverse_flow():
    evs = _events("ws_stream_wide.jsonl")
    base = OneSided(side="no", improve=True, min_spread=0.05, cooldown_s=300.0, lo=0.05, hi=0.95)
    gate = OneSided(side="no", improve=True, min_spread=0.05, cooldown_s=300.0, lo=0.05, hi=0.95,
                    gate_window_s=600.0, gate_share=0.5, gate_min_q=1.0)
    sims = [LaneSim(SimConfig(latency_ms=50.0, writes_per_s=10.0)) for _ in range(2)]
    for ev in evs:
        for st, sim in zip((base, gate), sims):
            sim.apply(ev)
            st.observe(sim, ev)
            st.on_event(sim, ev.market)
    assert len(sims[0].orders) >= 1
    assert gate.gated > 0 and len(sims[1].orders) <= len(sims[0].orders)


def test_one_sided_sell_yes_fills_on_recorded_book_and_scores():
    """Trimmed recording (hour snapshot .. a sell-YES fill + 120 s) of one KXTRUMPSAY strike."""
    from albert.backtest.l2 import markouts
    from albert.research.yes_bias import BASE, l2_metrics, l2_summary

    evs = _events("ws_stream_yes_bias.jsonl")
    st = OneSided(side="no", lo=0.30, hi=0.70, **BASE)
    sim = LaneSim(SimConfig(latency_ms=50.0, writes_per_s=10.0), keep_quotes=True)
    placed = []
    for ev in evs:
        sim.apply(ev)
        st.observe(sim, ev)
        n = len(sim.orders)
        q = sim.markets[ev.market].quote() if ev.market in sim.synced else None
        st.on_event(sim, ev.market)
        placed += [(oid, q) for oid in list(sim.orders)[n:]]
    assert placed and sim.fills and all(f.side == "no" for f in sim.fills)
    for oid, q in placed:     # one tick better than the best NO bid, strictly inside the spread
        o = sim.orders[oid]
        assert o.price == pytest.approx(round(1 - q["yes_ask"] + 0.01, 6))
        assert 1 - q["yes_ask"] < o.price < 1 - q["yes_bid"]
    mo = markouts(sim)
    fills = pl.DataFrame([{"lane": "sellY", "latency": 50.0, "t_ns": f.t_ns, "market": f.market, "side": f.side,
                           "price": f.price, "qty": f.qty, "fee": f.fee, "how": f.how, "mo_60s": r["mo_60s"],
                           "mo_300s": r["mo_300s"], "mx_60s": r["mx_60s"], "mx_300s": r["mx_300s"],
                           "last_mid": sim.mid_at(f.market, 2**62)} for f, r in zip(sim.fills, mo)])
    m = l2_metrics(fills, {fills["market"][0]: 0.0}, None, split_ns=0)
    f0 = m.row(0, named=True)
    assert f0["settled"] and f0["hold"] == pytest.approx(1.0 - f0["price"] - f0["fee"] / f0["qty"])
    summ = l2_summary(m)
    assert summ.filter((pl.col("regime") == "all") & (pl.col("split") == "holdout"))["fills"][0] == len(sim.fills)
