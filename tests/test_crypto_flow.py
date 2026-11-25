"""crypto_flow parsers and rules on trimmed copies of recorded responses (tests/fixtures/crypto_flow)."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from albert.research import crypto_flow as cf

FX = Path(__file__).parent / "fixtures" / "crypto_flow"


def _load(name: str) -> dict:
    return json.loads((FX / name).read_text())["response"]


def test_series_rows_keep_crypto_only():
    rows = cf.series_rows(_load("series.json"))
    tickers = {r["series_ticker"] for r in rows}
    assert {"KXBTCD", "KXBTCMAXMON", "KXAIRDROPMONAD"} <= tickers
    assert all(r["fee_type"] for r in rows)
    btcd = next(r for r in rows if r["series_ticker"] == "KXBTCD")
    assert btcd["frequency"] == "hourly" and btcd["fee_multiplier"] == 1.0


def test_fee_changes_filtered_to_crypto():
    rows = cf.fee_change_rows(_load("fee_changes.json"), {"KXBTCY", "KXETHY"})
    assert {r["series_ticker"] for r in rows} == {"KXBTCY", "KXETHY"}
    assert all(float(r["fee_multiplier"]) == 0 for r in rows)


def test_spot_rows_parse_coinbase_candles():
    rows = cf.spot_rows(_load("coinbase_btc_candles.json"), "BTC")
    assert len(rows) == 5
    r = rows[0]
    assert r["coin"] == "BTC" and r["ts"].tzinfo is not None
    assert r["low"] <= min(r["open"], r["close"]) <= max(r["open"], r["close"]) <= r["high"]
    assert 10_000 < r["close"] < 500_000


def test_daily_event_tickers_format():
    got = cf.daily_event_tickers("KXBTCD", date(2026, 6, 1), date(2026, 6, 3))
    assert [e for e, _ in got] == ["KXBTCD-26JUN0117", "KXBTCD-26JUN0217"]


def test_holm_and_p():
    assert cf.p_two_sided(None) == 1.0
    assert cf.p_two_sided(1.96) == pytest.approx(0.05, abs=1e-3)
    adj = cf.holm([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)


def test_family_and_coin():
    assert cf.family_of("KXBTCD", "KXBTCD-26NOV2717", 169) == "weekly"
    assert cf.family_of("KXBTCD", "KXBTCD-26NOV2317", 25) == "daily"
    assert cf.family_of("KXBTCMAXMON", "KXBTCMAXMON-26NOV", 720) == "onetouch"
    assert cf.family_of("KXAIRDROPMONAD", "x", 2000) == "other"
    assert cf.coin_of("KXETHMINMON") == "ETH" and cf.coin_of("KXBTXMINW") == "BTC"
    assert cf.coin_of("KXAIRDROPMONAD") is None


def test_capped_book_respects_cash():
    t0 = datetime(2026, 10, 1, tzinfo=UTC)
    rows = []
    for i in range(4):
        rows.append({"ticker": f"T{i}", "event_ticker": "E", "t": t0, "c": 0.02 + i / 100, "n": 1, "price": 0.85,
                     "fee_c": 0.0, "filled": i % 2 == 0, "settlement_ts": datetime(2026, 10, 3, tzinfo=UTC),
                     "close_time": datetime(2026, 10, 2, tzinfo=UTC), "t_fill": t0, "pnl_c": 15.0})
    df = pl.DataFrame(rows)
    # $2 cap: only two 85c reservations fit; the accepted orders are T0 (filled) and T1 (not filled)
    out = cf.capped(df, 2.0)
    assert out["ticker"].to_list() == ["T0"]
    assert cf.capped(df, 10.0).height == 2


def _markets(event: str) -> pl.DataFrame:
    payload = _load(f"markets_{event}.json")
    series = event.split("-")[0]
    return pl.DataFrame([cf.market_row(m, series, "hist") for m in payload["markets"]], schema=cf.MARKET_SCHEMA)


def test_market_rows_and_doge_custom_strike():
    btc = _markets("KXBTCD-26JUN0117")
    assert btc["strike_type"].to_list() == ["greater"] * 3
    assert btc["floor_strike"][0] == pytest.approx(63499.99)
    assert set(btc["result"]) <= {"yes", "no"}
    doge = cf.normalize(_markets("KXDOGED-26JUL2217"))
    assert doge["strike_type"].to_list() == ["greater"] * 3
    assert doge["floor_strike"].to_list() == pytest.approx([0.0049999, 0.0099999, 0.0149999])


def test_family_tags_daily_ladder_and_sched_close():
    mk = cf.sched_close(cf.with_family(_markets("KXBTCD-26JUN0117")))
    assert set(mk["family"]) == {"daily"} and set(mk["coin"]) == {"BTC"}
    # 5 pm EDT close; the scheduled close equals the leg close for these ladders
    assert (mk["sched_close"] == mk["close_time"]).all()
    assert mk["sched_close"][0] == datetime(2026, 6, 1, 21, tzinfo=UTC)


def _candles() -> pl.DataFrame:
    from albert.research import xcat

    payload = _load("candles_hist.json")
    rows = xcat.ohlc_rows(payload, "KXBTCD-26JUN1017-T65749.99", 60)
    return pl.DataFrame(rows, schema=xcat.OHLC_SCHEMA)


def test_tape_first_through_is_strict_and_after_t():
    c = _candles()
    assert c.height == 12
    traded = c.filter(pl.col("p_high").is_not_null()).sort("end_ts")
    assert traded.height >= 1
    tape = cf.Tape(c)
    tk = "KXBTCD-26JUN1017-T65749.99"
    first = traded.row(0, named=True)
    t0 = first["end_ts"] - timedelta(hours=1)
    end = c["end_ts"].max()
    # a price just below the candle's high is traded through; the high itself is not (strict)
    assert tape.first_through(tk, t0, end, first["p_high"] - 0.01, above=True) == first["end_ts"]
    later = tape.first_through(tk, t0, end, first["p_high"], above=True)
    assert later is None or later > first["end_ts"]
    # orders placed at or after a candle's end cannot be filled by that candle
    assert tape.first_through(tk, first["end_ts"], end, 0.0, above=True) != first["end_ts"]


def test_quotes_at_two_sided_flag():
    c = _candles()
    dec = pl.DataFrame({"ticker": ["KXBTCD-26JUN1017-T65749.99"], "t": [c["end_ts"].max()]})
    q = cf.quotes_at(dec, c)
    assert q.height == 1
    r = q.row(0, named=True)
    assert r["two_sided"] == (0 < r["bid"] < r["ask"] < 1)


def test_exact_p_handles_all_wins():
    t0 = datetime(2026, 10, 1, tzinfo=UTC)
    # 10 events, every fill won at an 80c break-even: P(10 wins) = 0.8^10 = 0.107, two-sided 0.215
    df = pl.DataFrame({"event_ticker": [f"E{i}" for i in range(10)], "t_fill": [t0] * 10, "c": [0.03] * 10,
                       "price": [0.8] * 10, "fee_c": [0.0] * 10, "won": [1] * 10})
    assert cf.p_exact_events(df) == pytest.approx(2 * 0.8 ** 10)
    assert cf.cell_p(df, 50.0) == pytest.approx(2 * 0.8 ** 10)
