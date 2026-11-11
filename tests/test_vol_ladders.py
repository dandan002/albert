"""Tests for the ladder vol-trading study: candle OHLC parsing (trimmed copy of a recorded
Kalshi candlestick response), per-order fee rounding, strictly-through maker fills, S(0)
interpolation, the cap scaling and the interval-censored width fit on simulated data."""

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from scipy.special import ndtr

from albert.research import vol_ladders as vl

# trimmed from data/raw/kalshi (historical candlesticks of a KXHIGH leg, recorded 2026-11-03)
RECORDED = {"candlesticks": [{"end_period_ts": 1786806000, "open_interest": "157.73",
                              "price": {"close": "0.5900", "high": "0.6000", "low": "0.5000", "mean": "0.5638",
                                        "open": "0.5300", "previous": None},
                              "volume": "157.73",
                              "yes_ask": {"close": "0.6000", "high": "1.0000", "low": "0.5000", "open": "1.0000"},
                              "yes_bid": {"close": "0.5400", "high": "0.5700", "low": "0.0100", "open": "0.0100"}}],
            "ticker": "KXHIGHAUS-26AUG15-B95.5"}


# trimmed from a recorded batch (/markets/candlesticks) response: "_dollars" / "_fp" field names
RECORDED_BATCH = [{"end_period_ts": 1789416000, "open_interest_fp": "194.98",
                   "price": {"close_dollars": "0.0400", "high_dollars": "0.0400", "low_dollars": "0.0400",
                             "mean_dollars": "0.0400", "open_dollars": "0.0400", "previous_dollars": "0.0300"},
                   "volume_fp": "1.00",
                   "yes_ask": {"close_dollars": "0.0400", "high_dollars": "0.0400", "low_dollars": "0.0400",
                               "open_dollars": "0.0400"},
                   "yes_bid": {"close_dollars": "0.0300", "high_dollars": "0.0300", "low_dollars": "0.0200",
                               "open_dollars": "0.0300"}}]


def test_parse_candle_payload():
    url = "https://api.elections.kalshi.com/trade-api/v2/historical/markets/KXHIGHAUS-26AUG15-B95.5/candlesticks"
    rows = vl.parse_candle_payload(url, {"period_interval": 60}, RECORDED)
    assert len(rows) == 1
    r = rows[0]
    assert r["ticker"] == "KXHIGHAUS-26AUG15-B95.5"
    assert (r["p_low"], r["p_high"], r["bid"], r["ask"]) == (0.50, 0.60, 0.54, 0.60)
    assert r["end_ts"] == datetime.fromtimestamp(1786806000, tz=UTC)
    assert vl.parse_candle_payload(url, {"period_interval": 1}, RECORDED) == []
    batch = {"markets": [{"market_ticker": "KXNFLGAME-X", "candlesticks": RECORDED["candlesticks"]},
                         {"market_ticker": "KXHIGHAUS-26OCT08-B86.5", "candlesticks": RECORDED_BATCH}]}
    got = vl.parse_candle_payload("https://x/markets/candlesticks", {"period_interval": 60}, batch)
    assert [g["ticker"] for g in got] == ["KXHIGHAUS-26OCT08-B86.5"]
    assert (got[0]["p_low"], got[0]["p_high"], got[0]["volume"], got[0]["bid"]) == (0.04, 0.04, 1.0, 0.03)


def test_fee_per_contract_rounds_per_order():
    # 0.07 * 5 * 0.25 = 8.75c -> 9c per 5-lot order
    assert vl.fee_pc(0.50) == pytest.approx(0.018)
    # weather series charge no maker fee; maker-fee series 0.0175 * 5 * 0.25 = 2.19c -> 3c
    assert vl.fee_pc(0.50, maker=True, fee_type="quadratic") == 0.0
    assert vl.fee_pc(0.50, maker=True, fee_type="quadratic_with_maker_fees") == pytest.approx(0.006)
    assert vl.fee_pc(0.95) == pytest.approx(0.004)        # 1.66c -> 2c per order


def test_maker_fill_strictly_through():
    t0 = datetime(2026, 10, 5, 12, tzinfo=UTC)
    o = pl.DataFrame({"ticker": ["A", "A", "B"], "side": ["yes", "yes", "no"], "lim": [0.40, 0.41, 0.10],
                      "snapshot_ts": [t0] * 3, "t_cancel": [t0 + timedelta(hours=3)] * 3})
    ohlc = pl.DataFrame({"ticker": ["A", "A", "A", "B", "B"],
                         "end_ts": [t0, t0 + timedelta(hours=1), t0 + timedelta(hours=4), t0 + timedelta(hours=2),
                                    t0 + timedelta(hours=3)],
                         "p_low": [0.10, 0.40, 0.30, 0.09, 0.09], "p_high": [0.5, 0.45, 0.5, 0.10, 0.11],
                         "volume": [5.0, 3.0, 9.0, 1.0, 2.0]})
    f = vl.maker_fills(o, ohlc).sort("oid")
    # A @ 0.40: the print at 0.40 is not through; the 0.30 print is after the cancel
    # A @ 0.41: filled by the 0.40 print in the candle ending t0 + 1 h (the t0 candle is before posting)
    # B NO bid at YES 0.10: filled by the 0.11 print (strictly above), not by the 0.10 one
    assert f["filled"].to_list() == [False, True, True]
    assert f["fill_ts"].to_list()[1:] == [t0 + timedelta(hours=1), t0 + timedelta(hours=3)]


def test_s_at_zero_normal_strip():
    F, sig = 3.0, 12.0
    K = np.array([-7.5, -3.5, -1.5, 1.5, 3.5, 7.5])
    pt = pl.DataFrame({"snap_id": [1] * len(K), "K": K, "S": ndtr((F - K) / sig)})
    s0 = vl.s_at_zero(pt)
    assert s0["S0"][0] == pytest.approx(float(ndtr(F / sig)), abs=1e-9)   # exact for a normal strip
    assert (s0["K_lo"][0], s0["K_hi"][0]) == (-1.5, 1.5)


def test_at_caps_scales_to_event_and_open_caps():
    t = datetime(2026, 10, 1, tzinfo=UTC)
    # two events, each one contract at 50c that wins 50c, held one day
    df = pl.DataFrame({"event_ticker": ["e1", "e2"], "pnl": [0.5, 0.5], "cost": [0.5, 0.5], "t_dec": [t, t],
                       "t_end": [t + timedelta(days=1)] * 2, "t_ref": [t, t]})
    usd, mo = vl.at_caps(df, weeks=1.0)
    # scaled to $5 each -> $10 PnL; capital need 2 x $5 x 1 d = 10 $d < $20 x 7 d
    assert usd == pytest.approx(10.0)
    assert mo["usd_at_caps"].sum() == pytest.approx(10.0)
    usd_short, _ = vl.at_caps(df, weeks=1 / 14)       # half a day of calendar: 20 x 0.5 = 10 of 10 $d
    assert usd_short == pytest.approx(10.0 * 14)
    usd_tight, _ = vl.at_caps(df, weeks=1 / 28)       # a quarter day: only half the capital fits
    assert usd_tight == pytest.approx(5.0 * 28)


def test_fit_sigma_recovers_shrink():
    rng = np.random.default_rng(0)
    n = 3000
    lead = rng.choice(vl.WX_LEADS, n)
    sig = rng.uniform(0.8, 2.5, n)
    F = rng.uniform(60, 90, n)
    X = F + 0.8 * sig * rng.standard_normal(n)
    lo = np.floor(X - 0.5) + 0.5            # 1-degree interval censoring around the integer high
    sn = pl.DataFrame({"lead_h": lead, "F": F, "sigma_atm": sig, "x_lo": lo, "x_hi": lo + 1.0})
    fit = vl.fit_sigma(sn, "shrink")
    lam = np.exp(np.array(fit["theta"][len(vl.WX_LEADS):]))
    assert np.allclose(lam, 0.8, atol=0.06)
    assert fit["converged"]
    assert math.isfinite(fit["dev_ll"])


def _legs(rows):
    base = {"maker": False, "fee_type": "quadratic", "fee_mult": 1.0, "won": True, "split": "dev"}
    return pl.DataFrame([{**base, **r} for r in rows])


def test_book_caps_and_sizing():
    t = datetime(2026, 9, 1, tzinfo=UTC)
    end = t + timedelta(days=1)
    rows = [{"struct_id": f"s{i}", "event_ticker": f"e{i}", "ticker": f"T{i}", "side": "yes", "price": 0.50,
             "t_post": t, "t_fill": t, "t_cancel": t, "settle_ts": end} for i in range(5)]
    led = vl.book(_legs(rows), "x", "v")
    # $5 per event: 9 contracts at 50c cost 4.50 + 0.16 fee; the fifth event gets what is left of $20
    assert led["n"].to_list() == [9, 9, 9, 9, 2]
    assert led["cost_usd"][0] == pytest.approx(4.66)
    assert led["cost_usd"].sum() <= vl.OPEN_CAP + 1e-9
    assert led["payoff_usd"].to_list() == [9.0, 9.0, 9.0, 9.0, 2.0]
    # after settlement the capital is free again
    later = [{**rows[0], "struct_id": "s9", "event_ticker": "e9", "t_post": end, "t_fill": end, "t_cancel": end,
              "settle_ts": end + timedelta(days=1)}]
    led2 = vl.book(_legs(rows + later), "x", "v")
    assert led2.filter(pl.col("struct_id") == "s9")["n"][0] == 9


def test_book_resting_orders_reserve_until_cancel_and_structures_share_n():
    t = datetime(2026, 9, 1, tzinfo=UTC)
    rows = [  # a two-leg maker structure on one event: equal n on both legs, unfilled leg not in the ledger
        {"struct_id": "a", "event_ticker": "e1", "ticker": "A1", "side": "yes", "price": 0.94, "maker": True,
         "t_post": t, "t_fill": t + timedelta(hours=1), "t_cancel": t + timedelta(hours=4), "settle_ts": t + timedelta(hours=8)},
        {"struct_id": "a", "event_ticker": "e1", "ticker": "A2", "side": "no", "price": 0.96, "maker": True,
         "t_post": t, "t_fill": None, "t_cancel": t + timedelta(hours=4), "settle_ts": t + timedelta(hours=8)},
        # same event two hours later: the unfilled order still reserves its collateral
        {"struct_id": "b", "event_ticker": "e1", "ticker": "A3", "side": "yes", "price": 0.50,
         "t_post": t + timedelta(hours=2), "t_fill": t + timedelta(hours=2), "t_cancel": t + timedelta(hours=2),
         "settle_ts": t + timedelta(hours=8)},
    ]
    led = vl.book(_legs(rows), "x", "v")
    a = led.filter(pl.col("struct_id") == "a")
    assert a.height == 1 and a["n"][0] == 2          # 2 x (0.94 + 0.96) = 3.80 <= 5 (no maker fee)
    b = led.filter(pl.col("struct_id") == "b")
    assert b["n"][0] == 2                           # 5 - 3.80 = 1.20 left on the event: 2 x 0.50 + 0.04 fee
