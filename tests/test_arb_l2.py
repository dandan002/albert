"""Structural-arbitrage scanner on real recordings: the one ladder episode found in the 2026-11-05
stream (KXTOKENUSE-26NOV09 T207 vs T210, 05:16:45 UTC: yes_ask(207) 0.61 < yes_bid(210) 0.70),
trimmed to the two markets' frames, and the recorded two-team NBA event metadata."""

import json
from pathlib import Path

import pytest

from albert.research.arb_l2 import (Books, Event, build_sets, replay, set_edge, summarize,
                                    taker_fee)

FIX = Path(__file__).parent / "fixtures" / "kalshi"
KINDS = {"orderbook_snapshot": "snapshot", "orderbook_delta": "delta", "trade": "trade"}


def _events(name: str = "ws_stream_ladder_arb.jsonl") -> list[Event]:
    out = []
    for line in (FIX / name).read_text().splitlines():
        r = json.loads(line)
        d = json.loads(r["frame"])
        out.append(Event(r["recv_ns"], KINDS[d["type"]], d["msg"]["market_ticker"], d["msg"], "k0"))
    return out


LADDER = {"KXTOKENUSE-26NOV09": {"me": False, "legs": [
    {"t": "KXTOKENUSE-26NOV09-T207", "strike_type": "greater", "floor": 207},
    {"t": "KXTOKENUSE-26NOV09-T210", "strike_type": "greater", "floor": 210}]}}


def test_taker_fee_rounds_up_to_a_hundredth_of_a_cent():
    assert taker_fee(1, 0.5) == pytest.approx(0.0175)
    assert taker_fee(1, 0.61) == pytest.approx(0.0167)       # 0.016653 -> 0.0167
    assert taker_fee(1, 0.0) == 0.0 and taker_fee(0, 0.5) == 0.0


def test_build_sets_from_recorded_event_metadata():
    e = json.loads((FIX / "event.json").read_text())
    legs = [{"t": m["ticker"], "strike_type": m.get("strike_type"), "floor": m.get("floor_strike"),
             "close_time": m.get("close_time")} for m in e["markets"]]
    events = {e["event"]["event_ticker"]: {"me": e["event"]["mutually_exclusive"], "legs": legs}}
    sets = build_sets(events, {x["t"] for x in legs})
    assert sorted(s.kind for s in sets) == ["me_no", "me_yes"]
    yes = next(s for s in sets if s.kind == "me_yes")
    assert yes.payoff == 1.0 and len(yes.legs) == 2 and all(o == "yes" for _, o in yes.legs)
    assert build_sets(events, {legs[0]["t"]}) == []            # incomplete set: not counted
    # a settled game's close_time is its actual end: books after it are ignored
    books = Books()
    assert set_edge(yes, books, 0) is None


def test_ladder_arb_found_in_the_recording_and_lost_to_latency():
    sets = build_sets(LADDER, {leg["t"] for leg in LADDER["KXTOKENUSE-26NOV09"]["legs"]})
    assert [s.kind for s in sets] == ["ladder"]
    eps, diag = replay(iter(_events()), sets)
    assert len(eps) == 1
    ep = eps[0]
    assert ep.limits == [pytest.approx(0.61), pytest.approx(0.30)]
    assert ep.edge0 == pytest.approx(1 - 0.61 - 0.30 - taker_fee(1, 0.61) - taker_fee(1, 0.30))
    assert (ep.end_ns - ep.start_ns) / 1e6 < 10                  # gone within 10 ms
    assert ep.trades_on_legs > 0                                 # someone else took it
    for lat in (50, 100):                                        # at our latency: one leg only
        r = ep.latency[lat]
        assert r["filled"] == [False, True] and r["pnl"] < 0
    s = summarize(eps, hours=1.0)["ladder"]
    assert s["episodes"] == 1 and s["L50"]["all_filled"] == 0


def test_books_follow_snapshot_and_deltas():
    evs = _events()
    books = Books()
    for ev in evs:
        if ev.kind != "trade":
            books.apply(ev.kind, ev.market, ev.msg)
    for m in ("KXTOKENUSE-26NOV09-T207", "KXTOKENUSE-26NOV09-T210"):
        y, n = books.best(m, "yes"), books.best(m, "no")
        assert y and n and y[0] + n[0] <= 1 + 1e-9                # never crossed
