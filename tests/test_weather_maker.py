import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import rain
from albert.research import weather_maker as wm
from albert.venues.kalshi import trade_rows

FIX = Path(__file__).parent / "fixtures"


def _trades() -> pl.DataFrame:
    # Trimmed copy of a recorded GET /markets/trades page (KXRAIN-26OCT04-CHI, 45 trades).
    d = json.loads((FIX / "kalshi" / "weather_maker_rain_trades.json").read_text())
    return pl.DataFrame(trade_rows(d), schema=wm.TRADE_SCHEMA)


def _minutes() -> pl.DataFrame:
    # Recorded GET /markets/candlesticks (period 1) page from the hourly-temperature study.
    d = json.loads((FIX / "kalshi" / "hourly_temp_candles_1m.json").read_text())
    return pl.DataFrame(wm.minute_rows(d), schema=wm.MINUTE_SCHEMA)


def _row(t: datetime, end: datetime, fair: float, bid: float, ask: float, y: int, ticker: str) -> pl.DataFrame:
    return pl.DataFrame({"ticker": [ticker], "event": ["e"], "cluster": ["e"], "t": [t], "end": [end],
                         "fair": [fair], "bid": [bid], "ask": [ask], "y": [y], "holdout": [True],
                         "fee_type": ["quadratic"], "fee_multiplier": [1.0]})


def test_minute_rows_keep_trade_extremes_only_when_something_traded():
    m = _minutes()
    assert m.height > 0 and m["ticker"].n_unique() >= 1
    traded = m.filter(pl.col("volume") > 0)
    idle = m.filter(pl.col("volume") == 0)
    assert traded.height and idle.height
    assert traded["p_low"].is_not_null().all() and (traded["p_low"] <= traded["p_high"]).all()
    assert idle["p_low"].is_null().all()            # "previous" price only: no print that minute
    p = wm.candle_prints(m)
    assert p.height == traded.height and p["lo"].is_not_null().all()


def test_trade_prints_follow_the_taker_side_rule():
    tr = _trades()
    assert tr.height == 45 and set(tr["taker_side"].unique().to_list()) == {"yes", "no"}
    p = wm.trade_prints(tr)
    # A YES-seller's print (taker no) can fill a YES bid; a YES-buyer's print (taker yes) a NO bid.
    no_rows = p.filter(pl.col("lo").is_not_null())
    assert no_rows.height == tr.filter(pl.col("taker_side") == "no").height
    assert p.filter(pl.col("lo").is_not_null() & pl.col("hi").is_not_null()).height == 0


def test_fills_only_strictly_through_and_inside_the_window():
    tr = _trades()
    tk = tr["ticker"][0]
    t0 = tr["created_time"].min() - timedelta(seconds=1)
    t1 = tr["created_time"].max()
    pr = wm.trade_prints(tr)
    # YES sellers printed at 1-4c; YES buyers at 1-8c.
    lo_min = tr.filter(pl.col("taker_side") == "no")["yes_price"].min()
    hi_max = tr.filter(pl.col("taker_side") == "yes")["yes_price"].max()
    assert lo_min == 0.01 and hi_max == 0.08
    rows = wm.window_extremes(_row(t0, t1, 0.03, 0.02, 0.10, 0, tk), pr)
    assert rows["lo_min"][0] == lo_min and rows["hi_max"][0] == hi_max
    # fair 3c, margin 0: YES bid 3c (a 1c sell went through it), NO bid 97c <=> YES ask 3c (8c buy above).
    o = wm.orders(rows, "fair", 0.0).sort("side")
    assert o["filled"].to_list() == [True, True]
    assert abs(o.filter(pl.col("side") == "yes")["pnl"][0] - (0 - 0.03)) < 1e-12
    assert abs(o.filter(pl.col("side") == "no")["pnl"][0] - (1 - 0.97)) < 1e-12
    # A YES bid at 1c is never filled: nothing printed strictly below 1c (the queue is not assumed to clear).
    o1 = wm.orders(rows, "fair", 0.02)
    assert not o1.filter(pl.col("side") == "yes")["filled"].any()
    # A window that ends before the first trade sees no prints.
    early = wm.window_extremes(_row(t0 - timedelta(hours=2), t0, 0.03, 0.02, 0.10, 0, tk), pr)
    assert np.isinf(early["lo_min"][0]) and not wm.orders(early, "fair", 0.0)["filled"].any()


def test_post_only_cap_gate_and_one_sided():
    r = _row(datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 10, 1, 1, tzinfo=UTC), 0.60, 0.40, 0.50, 1, "x")
    r = r.with_columns(pl.lit(0.30).alias("lo_min"), pl.lit(0.70).alias("hi_max"))
    o = wm.orders(r, "fair", 0.0).sort("side")
    # YES bid would be 60c but the ask is 50c: capped one tick inside at 49c.
    assert o.filter(pl.col("side") == "yes")["price"][0] == 0.49
    assert o.filter(pl.col("side") == "no")["price"][0] == 0.40
    assert wm.orders(r, "fair", 0.0, gate=0.20).height == 0          # |fair - mid| = 15c < 20c
    one = wm.orders(r, "fair", 0.0, one_sided=True)
    assert one["side"].to_list() == ["yes"]                           # fair above the mid
    assert wm.orders(r, "fair", 0.0, no_only=True)["side"].to_list() == ["no"]
    # Lenient: fill at our own price when we improved the best bid.
    r2 = r.with_columns(pl.lit(0.49).alias("lo_min"))
    assert not wm.orders(r2, "fair", 0.0).filter(pl.col("side") == "yes")["filled"][0]
    assert wm.orders(r2, "fair", 0.0, lenient=True).filter(pl.col("side") == "yes")["filled"][0]
    # Hypothetical maker fee reduces the PnL of a fill.
    fee = wm.orders(r, "fair", 0.0, maker_fee_type="quadratic_with_maker_fees").filter(pl.col("side") == "yes")
    assert fee["pnl"][0] < 1 - 0.49


def test_first_fill_per_side_drops_later_quotes_on_a_filled_side():
    t = [datetime(2026, 10, 1, h, tzinfo=UTC) for h in (0, 12)]
    o = pl.DataFrame({"ticker": ["a"] * 4, "side": ["yes", "yes", "no", "no"], "t": [t[0], t[1], t[0], t[1]],
                      "filled": [True, True, False, True], "pnl": [0.1, 0.2, 0.0, 0.3]})
    f = wm.first_fill_per_side(o).sort("side", "t")
    assert f.filter(pl.col("side") == "yes").height == 1                # second YES quote never posted
    assert f.filter(pl.col("side") == "no").height == 2


def test_first_rain_uses_reports_inside_the_day_on_recorded_asos():
    obs = pl.DataFrame(rain.asos_precip_rows((FIX / "iem" / "asos_precip.csv").read_text(), "NYC"))
    assert obs.height > 0
    lo, hi = rain.lst_bounds("NYC", obs["ts"].min().date())
    w = pl.DataFrame({"code": ["NYC"], "key": ["d"], "lo": [lo], "hi": [hi]})
    got = wm.first_rain(obs, w)["wet_ts"][0]
    # NYC's LST day starts 05:00 UTC; the 04:51 / 04:58 reports belong to the day before, the
    # 05:08 special (0.07") is the first rain report inside the day.
    assert got == datetime(2026, 9, 1, 5, 8, tzinfo=UTC)
    dry = w.with_columns(pl.col("lo") + timedelta(days=2), pl.col("hi") + timedelta(days=2))
    assert wm.first_rain(obs, dry)["wet_ts"][0] is None


def test_rolling_blend_uses_only_outcomes_known_before_the_day():
    rng = np.random.default_rng(1)
    n = 3000
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    t = [t0 + timedelta(hours=i // 10) for i in range(n)]
    mid = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < mid).astype(int)
    df = pl.DataFrame({"t": t, "mid": mid, "p": mid, "y": y}).with_columns(
        (pl.col("t") + timedelta(hours=2)).alias("known_ts"))
    base = wm.rolling_blend(df, window=timedelta(days=5), min_rows=50)
    # Flip every label from day 8 on: predictions for days <= 8 must not change.
    cut = t0 + timedelta(days=8)
    flipped = df.with_columns(pl.when(pl.col("t") >= cut).then(1 - pl.col("y")).otherwise(pl.col("y")).alias("y"))
    alt = wm.rolling_blend(flipped, window=timedelta(days=5), min_rows=50)
    before = (df["t"] < cut + timedelta(days=1)).to_numpy()
    ok = before & np.isfinite(base)
    assert ok.sum() > 100 and np.allclose(base[ok], alt[ok])
    assert np.isnan(base[: 10 * 24]).all()            # first day: nothing settled yet


def test_static_blend_dev_predictions_are_out_of_fold():
    rng = np.random.default_rng(2)
    n = 2000
    mid = rng.uniform(0.05, 0.95, n)
    df = pl.DataFrame({"mid": mid, "p": np.clip(mid + rng.normal(0, 0.1, n), 0.01, 0.99),
                       "y": (rng.uniform(size=n) < mid).astype(int), "fold": np.arange(n) % 4,
                       "holdout": np.arange(n) >= 1600})
    a, coefs = wm.static_blend(df, model="p")
    assert np.isfinite(a).all() and len(coefs[str(None)]) == 3
    alt = df.with_columns(pl.when(pl.col("fold") == 0).then(1 - pl.col("y")).otherwise(pl.col("y")).alias("y"))
    b, _ = wm.static_blend(alt, model="p")
    f0 = (df["fold"] == 0).to_numpy() & ~df["holdout"].to_numpy()
    assert np.allclose(a[f0], b[f0])                    # fold 0's own labels never fit its predictions


def test_vectorised_maker_fee_matches_the_harness():
    from albert.research.fundamental import maker_fee

    r = _row(datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 10, 1, 1, tzinfo=UTC), 0.37, 0.30, 0.45, 1, "x")
    r = r.with_columns(pl.lit(0.10).alias("lo_min"), pl.lit(0.90).alias("hi_max"), pl.lit(1.5).alias("fee_multiplier"))
    o = wm.orders(r, "fair", 0.0, maker_fee_type="quadratic_with_maker_fees").sort("side")
    for side, won in (("no", 0), ("yes", 1)):
        x = o.filter(pl.col("side") == side)
        price = x["price"][0]
        assert abs(x["pnl"][0] - (won - price - maker_fee(price, "quadratic_with_maker_fees", 1.5))) < 1e-12
    assert wm.orders(r, "fair", 0.0).filter(pl.col("side") == "yes")["pnl"][0] == 1 - 0.37   # quadratic: no fee


def test_asos_interval_totals_do_not_double_count_specials():
    obs = pl.DataFrame(rain.asos_precip_rows((FIX / "iem" / "asos_precip.csv").read_text(), "NYC"))
    h = wm.asos_hourly_totals(obs)
    assert h.height > 0
    # Routine reports at :51; the specials at 04:58 .. 05:41 are running partials of the 04:51-05:51
    # interval whose routine total is 0.11".
    b = h.filter(pl.col("bucket") == datetime(2026, 9, 1, 4, tzinfo=UTC))
    assert b["inch"][0] == 0.11 and b["ts_end"][0] == datetime(2026, 9, 1, 5, 51, tzinfo=UTC)
    naive = obs.filter(pl.col("p01i") >= 0.01)["p01i"].sum()
    assert h["inch"].sum() < naive


def test_climo_remaining_sums_the_rest_of_the_month():
    days = pl.date_range(datetime(2016, 1, 1).date(), datetime(2025, 12, 31).date(), "1d", eager=True)
    cli = pl.DataFrame({"code": "NYC", "day": days, "precip": 0.1})
    rem = wm.climo_remaining(cli, "NYC", datetime(2026, 10, 25).date(), 0.5)
    assert len(rem) >= 90 and np.allclose(rem, 0.5 * 0.1 + 6 * 0.1)
    assert wm.month_end(datetime(2026, 2, 3).date()) == datetime(2026, 2, 28).date()


def test_spread_buckets_and_blend_by_segment():
    df = pl.DataFrame({"bid": [0.40, 0.40, 0.40, 0.40], "ask": [0.43, 0.46, 0.55, 0.56]})
    b = df.select(wm.spread_bucket(pl.col("bid"), pl.col("ask")).alias("b"))["b"].to_list()
    assert b == ["s1 <=3c", "s2 4-6c", "s3 7-15c", "s4 >15c"]
    rng = np.random.default_rng(3)
    n = 4000
    truth = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < truth).astype(int)
    noisy = np.clip(truth + rng.normal(0, 0.2, n), 0.02, 0.98)
    # In "wide" rows the mid is noise and the model is the truth; in "tight" rows the mid is the truth.
    wide = np.arange(n) % 2 == 0
    rows = pl.DataFrame({"mid": np.where(wide, noisy, truth), "p": np.where(wide, truth, noisy), "y": y,
                         "seg": np.where(wide, "wide", "tight"), "holdout": np.arange(n) >= 3000,
                         "cluster": np.arange(n) // 4})
    t = wm.blend_by(rows, "seg").filter(pl.col("split") == "dev")
    c = dict(zip(t["seg"].to_list(), t["c_model"].to_list()))
    assert c["wide"] > 0.6 and abs(c["tight"]) < 0.25
