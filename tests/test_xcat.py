"""Cross-category candle sampler: OHLC parsing on recorded payloads and batch planning within API caps."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from albert.research import xcat

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def test_ohlc_rows_parse_live_batch_payload():
    payload = json.loads((FIX / "candles_batch.json").read_text())
    got = xcat.batch_rows(payload, 60)
    assert got and all(rows for rows in got.values())
    r = next(iter(got.values()))[0]
    raw = payload["markets"][0]["candlesticks"][0]
    assert r["p_high"] == pytest.approx(float(raw["price"]["high_dollars"]))
    assert r["bid_low"] == pytest.approx(float(raw["yes_bid"]["low_dollars"]))
    assert r["ask_open"] == pytest.approx(float(raw["yes_ask"]["open_dollars"]))
    assert r["volume"] == pytest.approx(float(raw["volume_fp"]))
    assert r["end_ts"].tzinfo is not None
    pl.DataFrame([r], schema=xcat.OHLC_SCHEMA)      # matches the stored schema


def test_ohlc_rows_parse_historical_payload():
    payload = json.loads((FIX / "candles_historical.json").read_text())
    rows = xcat.ohlc_rows(payload, payload["ticker"], 60)
    raw = payload["candlesticks"][0]
    assert len(rows) == len(payload["candlesticks"])
    assert rows[0]["p_low"] == pytest.approx(float(raw["price"]["low"]))
    assert rows[0]["ask_high"] == pytest.approx(float(raw["yes_ask"]["high"]))
    assert rows[0]["oi"] == pytest.approx(float(raw["open_interest"]))


def test_jobs_respect_batch_caps_and_route_historical():
    cutoff = datetime(2026, 9, 18, tzinfo=UTC)
    t0 = datetime(2026, 10, 1, tzinfo=UTC)
    rows = [{"ticker": f"EV-{i}", "event_ticker": "EV", "period": 1, "lo": t0, "hi": t0 + timedelta(days=2),
             "settlement_ts": t0 + timedelta(days=3)} for i in range(7)]
    rows.append({"ticker": "OLD-1", "event_ticker": "OLD", "period": 1, "lo": t0 - timedelta(days=40),
                 "hi": t0 - timedelta(days=39), "settlement_ts": cutoff - timedelta(days=5)})
    jobs = xcat._jobs(pl.DataFrame(rows), cutoff)
    hist = [j for j in jobs if j["kind"] == "hist"]
    assert [j["rows"][0]["ticker"] for j in hist] == ["OLD-1"]
    for j in jobs:
        if j["kind"] == "batch":
            span = 2 * 24 * 60 + 1
            assert span * len(j["rows"]) <= xcat.MAX_BATCH_CANDLES
    assert sum(len(j["rows"]) for j in jobs) == 8
