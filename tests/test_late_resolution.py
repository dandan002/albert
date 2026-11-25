"""late_resolution: parsers on trimmed recorded Kalshi payloads and the H1 accounting."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from albert.research import late_resolution as lr

FX = Path(__file__).parent / "fixtures" / "late_resolution"


def _load(name: str):
    return json.loads((FX / name).read_text())


def test_trade_rows_live_payload():
    rows = lr.trade_rows(_load("trades_live_trimmed.json"))
    assert len(rows) == 4
    r = rows[0]
    assert r["ticker"] == "KXJOHNSONMENTION-26OCT29-HEAL"
    assert r["yes_price"] == pytest.approx(0.01)
    assert r["count"] == pytest.approx(70.13)
    assert r["taker_side"] == "yes"
    assert r["ts"] == datetime(2026, 10, 29, 12, 53, 10, 918790, tzinfo=UTC)
    assert r["trade_id"]
    assert pl.DataFrame(rows, schema=lr.TRADE_SCHEMA).height == 4


def test_candle_rows_batch_and_historical():
    b = _load("candles_batch_trimmed.json")
    rows = lr.candle_rows(b["markets"][0], b["markets"][0]["market_ticker"])
    assert len(rows) == 3 and rows[0]["ticker"] == "KXHIGHNY-26OCT21-B61.5"
    assert rows[0]["bid"] == pytest.approx(0.57) and rows[0]["ask"] == pytest.approx(0.58)
    h = _load("candles_historical_trimmed.json")
    hr = lr.candle_rows(h, h["ticker"])
    assert hr[0]["ask"] == pytest.approx(0.75) and hr[0]["bid"] == pytest.approx(0.25)
    assert pl.DataFrame(hr, schema=lr.CANDLE_SCHEMA).height == 3


def test_fee_rounds_up_per_order():
    assert lr.fee(1, 0.99) == pytest.approx(0.01)          # 0.07 * 0.0099 -> 1c
    assert lr.fee(60, 0.99) == pytest.approx(0.05)         # 0.0416 -> 5c
    assert lr.fee(10, 0.97, maker=True) == 0.0             # quadratic series: no maker fee


def test_h1_tables_winner_accounting():
    occ = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)
    s = pl.DataFrame({"ticker": ["A", "B"], "event_ticker": ["E", "E"], "series_ticker": ["S", "S"],
                      "category": ["Mentions"] * 2, "wclass": ["short"] * 2, "split": ["holdout"] * 2,
                      "occurrence_time": [occ, occ], "close_time": [occ + timedelta(minutes=20)] * 2,
                      "win_min": [20.0, 20.0], "result": ["yes", "no"], "fee_type": ["quadratic"] * 2,
                      "fee_multiplier": [1.0, 1.0]})
    t = pl.DataFrame({"trade_id": ["1", "2", "3", "4"], "ticker": ["A", "A", "B", "B"],
                      "ts": [occ - timedelta(minutes=5), occ + timedelta(minutes=2), occ + timedelta(minutes=3),
                             occ + timedelta(minutes=4)],
                      "count": [5.0, 10.0, 4.0, 2.0], "yes_price": [0.80, 0.95, 0.06, 0.02],
                      "taker_side": ["yes", "yes", "yes", "no"]}, schema=lr.TRADE_SCHEMA)
    c = pl.DataFrame({"ticker": ["A", "B"], "ts": [occ + timedelta(minutes=1)] * 2, "bid": [0.94, 0.01],
                      "ask": [0.96, 0.05], "p_close": [0.95, 0.02], "volume": [1.0, 1.0]}, schema=lr.CANDLE_SCHEMA)
    out = lr.h1_tables(s, t, c)["h1_cells"].row(0, named=True)
    # A: taker bought the winner (YES) at 95c after occurrence; B: a YES taker bought the dead leg at 6c
    assert out["post_contracts"] == 16
    assert out["post_taker_bought_winner"] == 12          # 10 on A + 2 NO-takers on B
    # taker pool d1: A 10 x 5c - fee(10, .95) = 4c -> 0.46; B 2 x 2c - fee(2, .98) = 1c -> 0.03
    assert out["taker_pool_usd_d1"] == pytest.approx(0.49)
    # resting NO-winner bid at 97c on B: the 6c YES print trades through (winner price 94c < 97c)
    assert out["maker97_usd"] == pytest.approx(4 * 0.03)
    # ask-based taker at +1 min: A's YES ask 96c, B's NO ask 1 - 0.01 = 99c
    assert out["ask_offer_markets_d1"] == 2


def test_l_anchor_from_ticker_date():
    df = pl.DataFrame({"t": ["KXYTVIEWSD-TAY26SEP19-15.9M", "KXYTVIEWSW-FUT26OCT04-7.0M", "BAD"]})
    a = df.select(lr.l_anchor(pl.col("t")).alias("a"))["a"].to_list()
    assert a[0] == datetime(2026, 9, 20, 14, 0, tzinfo=UTC)
    assert a[1] == datetime(2026, 10, 5, 14, 0, tzinfo=UTC)
    assert a[2] is None


def test_capped_book_caps_open_and_event():
    t0 = datetime(2026, 10, 1, tzinfo=UTC)
    f = pl.DataFrame({"ticker": ["a", "b", "c", "d"], "event_ticker": ["E1", "E1", "E2", "E3"],
                      "t_in": [t0, t0 + timedelta(hours=1), t0 + timedelta(hours=2), t0 + timedelta(hours=5)],
                      "settle_ts": [t0 + timedelta(hours=4)] * 3 + [t0 + timedelta(hours=6)],
                      "cost_usd": [4.0, 4.0, 4.0, 4.0]})
    got = lr.capped_book(f, max_open=8.0, per_event=5.0)["ticker"].to_list()
    assert got == ["a", "c", "d"]      # b breaks E1's $5 cap; d enters after a and c settled


def test_holm_and_choose():
    adj = lr.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    dev = pl.DataFrame({"cell": ["x", "y"], "events": [20, 20], "t_event": [4.0, 1.0], "c_per_contract": [1.0, 5.0]})
    pick, _ = lr.choose(dev)
    assert pick == "x"


def test_parsers_on_this_study_recorded_windows():
    h = lr.trade_rows(_load("trades_historical_trimmed.json"))
    assert h and h[0]["ticker"] == "KXVOTEPRIMARY-MAPRIMARY08D26SLYN-58"
    assert h[0]["yes_price"] == pytest.approx(0.01) and h[0]["count"] == pytest.approx(1000.0)
    b = _load("candles_batch_minute_trimmed.json")
    rows = [r for m in b["markets"] for r in lr.candle_rows(m, m["market_ticker"])]
    assert len(rows) == 6
    r0 = rows[0]
    assert r0["bid"] == pytest.approx(0.95) and r0["ask"] == pytest.approx(1.0)
    assert r0["p_close"] is None          # no trade in the hour: only price.previous is reported
