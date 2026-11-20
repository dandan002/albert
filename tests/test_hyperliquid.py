"""Hyperliquid HIP-4 outcome data (public, never traded) against recorded responses
(``albert capture --only hl``)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from albert import archive
from albert.http import CachedClient, read_raw
from albert.stream import HlUniverse
from albert.venues import hyperliquid as hl

FIX = Path(__file__).parent / "fixtures" / "hyperliquid"


def _load(name: str):
    return json.loads((FIX / name).read_text())


def test_coin_encoding_round_trip():
    assert hl.coin(37286) == "#372860"
    assert hl.coin(37286, 1) == "#372861"
    assert hl.outcome_of("#372861") == (37286, 1)
    assert hl.outcome_of("+10") == (1, 0)
    with pytest.raises(ValueError):
        hl.coin(1, 2)


def test_spec_values_keep_colons_and_times_are_utc():
    spec = hl.parse_spec("perp:xyz:CL|priceDescription:xyz:CL-USDC perp mark|stage: Stage 1")
    assert spec == {"perp": "xyz:CL", "priceDescription": "xyz:CL-USDC perp mark", "stage": "Stage 1"}
    assert hl.parse_spec("") == {} and hl.parse_spec("other") == {}
    assert hl.parse_time("20261122-1515") == datetime(2026, 11, 22, 15, 15, tzinfo=UTC)


def test_outcomes_classify_sports_and_exclude_financial():
    meta = _load("outcome_meta.json")
    f = hl.outcome_frame(meta)
    assert f.height == len(meta["outcomes"]) > 0
    rows = {r["outcome"]: r for r in f.to_dicts()}
    match = next(q for q in meta["questions"] if q["name"] == "template:sportsContestResult")
    spec = hl.parse_spec(match["description"])
    for o in [*match["namedOutcomes"], match["fallbackOutcome"]]:
        r = rows[o]
        assert r["non_financial"] and r["question"] == match["question"]
        assert r["sport"] == spec["sport"] and r["participant_a"] == spec["participantA"]
        assert r["scheduled_start"] == hl.parse_time(spec["scheduledStart"])
    assert rows[match["fallbackOutcome"]]["question_role"] == "fallback"
    assert any(r["participant"] for o, r in rows.items() if o in match["namedOutcomes"])
    for r in rows.values():                              # every financial template stays out
        if r["template"] in hl.FINANCIAL_TEMPLATES or r["question_template"] in hl.FINANCIAL_TEMPLATES:
            assert not r["non_financial"], r
    winner = next(r for r in rows.values() if r["template"] == "sportsContestWinner")
    assert winner["non_financial"] and winner["side0"].startswith("template:")
    coins = hl.non_financial_coins(meta)
    assert coins and all(rows[hl.outcome_of(c)[0]]["non_financial"] for c in coins)
    dated = [rows[hl.outcome_of(c)[0]]["scheduled_start"] for c in coins]
    assert dated.index(None) > 0 if None in dated else True      # undated futures sort last
    assert hl.unknown_templates(meta) == []


def test_unknown_template_is_excluded_and_reported():
    meta = {"outcomes": [{"outcome": 9, "name": "template:weatherHigh", "description": "city:NYC",
                          "sideSpecs": [{"name": "Yes"}, {"name": "No"}]}], "questions": []}
    assert hl.unknown_templates(meta) == ["weatherHigh"]
    assert not hl.outcome_rows(meta)[0]["non_financial"]
    assert hl.non_financial_coins(meta) == []


def test_trades_candles_and_book_parse():
    trades = hl.trade_rows(_load("recent_trades.json"))
    assert trades and all(t["buyer"].startswith("0x") and t["seller"].startswith("0x") for t in trades)
    assert all(0 < t["px"] < 1 and t["sz"] > 0 and t["side"] in ("A", "B") for t in trades)
    candles = hl.candle_rows(_load("candle_snapshot.json"))
    assert candles and all(c["l"] <= min(c["o"], c["c"]) <= max(c["o"], c["c"]) <= c["h"] for c in candles)
    assert all(c["interval"] == "1h" and c["outcome"] == trades[0]["outcome"] for c in candles)
    bids, asks = hl.book_levels(_load("l2_book.json"))
    assert bids and bids == sorted(bids, reverse=True)
    assert asks == sorted(asks)


def test_ws_frames_have_the_channels_the_recorder_counts():
    frames = [json.loads(x) for x in (FIX / "ws_frames.jsonl").read_text().splitlines()]
    by = {f["channel"]: f for f in frames}
    assert {"l2Book", "trades", "subscriptionResponse", "pong"} <= set(by)
    assert hl.book_levels(by["l2Book"]["data"])[0]
    assert hl.trade_rows(by["trades"]["data"])[0]["buyer"]


def test_post_is_cached_by_body_and_keeps_it(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=_load("recent_trades.json"))

    slept = []
    client = CachedClient("hyperliquid", "https://api.hyperliquid.xyz", tmp_path,
                          transport=httpx.MockTransport(handler), sleep=lambda s: None)
    api = hl.Hyperliquid(client, budget_per_min=600, sleep=slept.append)
    live = api.recent_trades("#372860")
    assert not live.from_cache and calls == [{"type": "recentTrades", "coin": "#372860"}]
    assert read_raw(live.raw_path)["body"] == calls[0]
    assert slept == [pytest.approx((20 + len(live.payload) // 20) * 60 / 600)]
    a = client.post("/info", {"type": "x", "coin": "#1"})
    b = client.post("/info", {"coin": "#1", "type": "x"})          # same body, other key order
    assert b.from_cache and a.raw_path == b.raw_path and len(calls) == 2
    assert client.post("/info", {"type": "x", "coin": "#2"}).raw_path != a.raw_path


def test_universe_caps_and_adds_to_running_connections():
    at = datetime(2026, 11, 20, tzinfo=UTC)
    uni = HlUniverse(chunk=2, max_coins=4)
    opened = uni.assign(["#10", "#20", "#30"], at)
    assert [c["conn"] for c in opened] == ["h0", "h1"]
    assert opened[0]["queue"].empty() and opened[1]["coins"] == ["#30"]
    assert uni.assign(["#10", "#40", "#50", "#60"], at) == []        # #40 joins h1 live
    assert uni.conns[1]["queue"].get_nowait() == "#40"
    assert uni.conns[1]["coins"] == ["#30", "#40"] and len(uni.conns) == 2
    assert uni.skipped == {"#50", "#60"} and len(uni.added) == 4
    m = uni.manifest()
    assert m["conns"][0]["coins"] == 2 and m["skipped"] == ["#50", "#60"]


def test_archive_moves_hl_days_too():
    assert "hl" in archive.VENUES
