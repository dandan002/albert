"""Resolution-spike helpers: leader side and as-of rises on sparse minute candles, trigger choice,
trade-through limit fills, fees, and no use of candles at or after the close."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from albert.fees import kalshi_fee
from albert.research import spikes as sp
from albert.research import xcat

T0 = datetime(2026, 10, 5, 12, tzinfo=UTC)


def _meta(result: str = "yes", close_after_min: int = 300, fee_type: str = "quadratic_with_maker_fees") -> pl.DataFrame:
    return pl.DataFrame({
        "ticker": ["M"], "event_ticker": ["EV"], "series_ticker": ["KXTEST"], "category": ["Mentions"],
        "result": [result], "open_time": [T0 - timedelta(hours=1)], "close_time": [T0 + timedelta(minutes=close_after_min)],
        "settlement_ts": [T0 + timedelta(minutes=close_after_min + 5)], "expected_expiration_time": [T0 + timedelta(days=9)],
        "fee_type": [fee_type], "fee_multiplier": [1.0], "stratum": ["mentions"]})


def _candles(path: list[tuple[int, float, float, float, float]]) -> pl.DataFrame:
    """(minute, bid, ask, trade_low, trade_high) -> sparse minute candles."""
    rows = []
    for mnt, b, a, lo, hi in path:
        rows.append({"ticker": "M", "end_ts": T0 + timedelta(minutes=mnt), "period": 1,
                     "p_open": lo, "p_high": hi, "p_low": lo, "p_close": hi, "p_mean": (lo + hi) / 2,
                     "bid_open": b, "bid_high": b, "bid_low": b, "bid_close": b,
                     "ask_open": a, "ask_high": a, "ask_low": a, "ask_close": a, "volume": 50.0, "oi": 100.0})
    return pl.DataFrame(rows, schema=xcat.OHLC_SCHEMA)


PATH = [(0, 0.20, 0.22, 0.21, 0.21),      # quiet at 21c
        (30, 0.24, 0.26, 0.25, 0.25),
        (50, 0.86, 0.88, 0.30, 0.87),      # the spike: leader YES jumps to 87c
        (55, 0.91, 0.93, 0.88, 0.92),      # >= 90c, +71c over 60 min: the primary trigger
        (70, 0.92, 0.94, 0.86, 0.93),      # a print at 86c: through an 88c YES bid
        (400, 0.99, 1.00, 0.99, 0.99)]     # after the close: must be dropped


def test_minute_panel_rises_use_the_quote_standing_w_minutes_ago_and_drop_post_close_rows():
    p = sp.minute_panel(_meta(), _candles(PATH))
    assert p["t"].max() < T0 + timedelta(minutes=300)
    r = p.filter(pl.col("t") == T0 + timedelta(minutes=55)).row(0, named=True)
    assert r["lead_yes"] and r["q"] == pytest.approx(0.92)
    assert r["q5"] == pytest.approx(0.87)          # the 50-min candle stands at minute 50
    assert r["q15"] == pytest.approx(0.25)         # minute 40 -> the 30-min candle (sparse: no change since)
    assert r["q60"] is None                        # nothing before minute -5
    assert r["rise15"] == pytest.approx(0.67)
    assert r["ask_1m"] == pytest.approx(0.93)      # nothing new within a minute: same quote


def test_primary_trigger_fires_once_at_the_first_qualifying_minute():
    p = sp.minute_panel(_meta(), _candles(PATH))
    trig = sp.trigger_rows(p)
    prim = trig.filter(pl.col("trigger") == "q>=90 rise>=10c/15m")
    assert prim.height == 1 and prim["t"][0] == T0 + timedelta(minutes=55)
    assert trig.filter(pl.col("trigger") == "q>=97 first touch").is_empty()


def test_spike_taker_follow_and_fade_prices_and_fees():
    p = sp.minute_panel(_meta(), _candles(PATH))
    trig = sp.trigger_rows(p).filter(pl.col("trigger") == "q>=90 rise>=10c/15m")
    tk = sp.spike_taker(trig)
    fol = tk.filter(pl.col("mode") == "follow").row(0, named=True)
    assert fol["price"] == pytest.approx(0.93) and fol["n"] == 5 and fol["payoff"] == 1.0
    assert fol["fee_order"] == pytest.approx(kalshi_fee(5, 0.93))
    fade = tk.filter(pl.col("mode") == "fade").row(0, named=True)
    assert fade["price"] == pytest.approx(1 - 0.91) and fade["payoff"] == 0.0


def test_maker_bid_fills_only_on_a_print_strictly_through_the_level():
    p = sp.minute_panel(_meta(), _candles(PATH))
    trig = sp.trigger_rows(p).filter(pl.col("trigger") == "q>=90 rise>=10c/15m")
    mk = sp.spike_maker(trig, p, ("q>=90 rise>=10c/15m",))
    lead = mk.filter((pl.col("side") == "leader") & (pl.col("horizon") == "60m"))
    by_off = {r["offset"]: r for r in lead.iter_rows(named=True)}
    assert by_off["-3c"]["L"] == pytest.approx(0.89) and by_off["-3c"]["filled"]   # 86c print < 89c
    assert by_off["-3c"]["t_fill"] == T0 + timedelta(minutes=70)
    assert by_off["-5c"]["L"] == pytest.approx(0.87) and by_off["-5c"]["filled"]   # 86 < 87
    m = by_off["-3c"]
    assert m["fee_order"] == pytest.approx(kalshi_fee(m["n"], 0.89, maker=True, fee_type="quadratic_with_maker_fees"))
    lag = mk.filter((pl.col("side") == "laggard") & (pl.col("offset") == "-1c") & (pl.col("horizon") == "60m")).row(0, named=True)
    # NO bid at floor(0.08 - 0.01) = 7c fills only if YES prints above 93c: the 93c high is not through
    assert lag["L"] == pytest.approx(0.07) and not lag["filled"]


def test_maker_fee_is_zero_without_maker_fee_series():
    p = sp.minute_panel(_meta(fee_type="quadratic"), _candles(PATH))
    trig = sp.trigger_rows(p).filter(pl.col("trigger") == "q>=90 rise>=10c/15m")
    mk = sp.spike_maker(trig, p, ("q>=90 rise>=10c/15m",))
    assert (mk["fee_order"] == 0).all()
