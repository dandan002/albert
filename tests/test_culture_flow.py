"""culture_flow: family mapping, fill tests on a recorded Kalshi candle payload, fees, clustering,
event exclusivity, Polymarket edge rows and the twin taker rule."""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import culture_flow as cf
from albert.research import xcat

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _tape() -> tuple[cf.Tape, pl.DataFrame]:
    payload = json.loads((FIX / "recurring_candles_incumbent.json").read_text())
    rows = xcat.ohlc_rows(payload, "KXTEST-26AUG17-A", 60)
    c = pl.DataFrame(rows, schema=xcat.OHLC_SCHEMA)
    return cf.Tape(c), c


def test_family_mapping():
    assert cf.family_of("KXNETFLIXRANKSHOWGLOBAL2") == "netflix"
    assert cf.family_of("KXNETFLIXTOPVIEWSTV") == "netflix"
    assert cf.family_of("KXBILLBOARDRUNNERUPSONG") == "billboard"
    assert cf.family_of("KXTOPSONG") == "billboard"
    assert cf.family_of("KXTOPSONGRECORDPLAY") is None
    assert cf.family_of("KXTRUTHSOCIALD") == "truth"
    assert cf.family_of("KXRTZOOTOPIA2") is None
    assert cf.family_of("KXHORMUZWEEKLY") == "hormuz"
    df = pl.DataFrame({"series_ticker": ["KXEOWEEK", "KXVHGCB", "KXFOO"]})
    assert df.select(cf.family_expr())["literal"].to_list() == ["exec", "approval", None]


def test_tape_first_through_is_strict_and_after_order_time():
    tape, c = _tape()
    tk = "KXTEST-26AUG17-A"
    t0 = datetime(2026, 8, 14, 18, 0, tzinfo=UTC)          # the 18:00 candle (high 0.62) is not after t0
    t1 = datetime(2026, 8, 20, tzinfo=UTC)
    # first candle ending after t0 with a print above 0.64 is 19:00 (high 0.65)
    assert tape.first_through(tk, t0, t1, 0.64, "sell_yes") == datetime(2026, 8, 14, 19, 0, tzinfo=UTC)
    # strictly through: a print AT our price does not fill
    assert tape.first_through(tk, t0, datetime(2026, 8, 14, 19, 0, tzinfo=UTC), 0.65, "sell_yes") is None
    hi = c.filter(pl.col("end_ts") > t0)["p_high"].max()
    assert tape.first_through(tk, t0, t1, hi, "sell_yes") is None
    # buy side: a print below our YES bid
    assert tape.first_through(tk, t0, t1, 0.64, "buy_yes") == datetime(2026, 8, 14, 19, 0, tzinfo=UTC)
    assert tape.first_through("UNKNOWN", t0, t1, 0.5, "sell_yes") is None


def test_maker_no_prices_and_pnl():
    tape, _ = _tape()
    t = datetime(2026, 8, 14, 18, 0, tzinfo=UTC)
    o = pl.DataFrame([{"ticker": "KXTEST-26AUG17-A", "event_ticker": "KXTEST-26AUG17", "family": "netflix",
                       "series_ticker": "KXTEST", "split": "dev", "h": 72, "t": t, "close_time": t + timedelta(days=5),
                       "bid": 0.55, "ask": 0.59, "y": 0, "fee_type": "quadratic", "fee_multiplier": 1.0,
                       "settlement_ts": t + timedelta(days=6)}])
    f = cf.maker_no(o, tape, 6)
    r = f.row(0, named=True)
    assert r["sell_at"] == 0.63                       # ceil(0.57 + 0.06)
    assert math.isclose(r["price"], 0.37)
    assert r["t"] == datetime(2026, 8, 14, 19, 0, tzinfo=UTC)
    assert math.isclose(r["pnl_c"], 63.0)              # NO won: 1 - 0.37
    # a quote never traded through does not fill
    o2 = o.with_columns(pl.lit(0.95).alias("bid"), pl.lit(0.97).alias("ask"))
    assert cf.maker_no(o2, tape, 6).is_empty()


def test_taker_fee_rounds_up_per_order():
    assert cf.taker_fee_order(5, 0.9) == 0.04          # 0.0315 -> 4c
    assert cf.taker_fee_order(1, 0.5) == 0.02          # 0.0175 -> 2c
    assert cf.taker_fee_order(100, 0.5) == 1.75


def test_taker_rows_no_side_uses_bid():
    t = datetime(2026, 9, 1, tzinfo=UTC)
    o = pl.DataFrame([{"ticker": "A", "event_ticker": "E", "family": "rt", "series_ticker": "KXRT", "split": "dev",
                       "h": 24, "t": t, "bid": 0.04, "ask": 0.07, "y": 0, "fee_multiplier": 1.0,
                       "settlement_ts": t + timedelta(days=1)}])
    r = cf.taker_rows(o, "no").row(0, named=True)
    assert math.isclose(r["price"], 0.96) and r["n"] == 4
    assert math.isclose(r["fee_order"], math.ceil(0.07 * 4 * 0.96 * 0.04 * 100) / 100)
    assert math.isclose(r["pnl_c"], 100 * (1 - 0.96 - r["fee_order"] / 4))


def test_two_way_t_reduces_to_one_way():
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 1, 200)
    g = np.repeat(np.arange(40), 5)
    _, t1, _ = cf.cl_t(x, g)
    t2 = cf.two_way_t(x, g, np.arange(200))
    assert math.isclose(t1, t2, rel_tol=0.05)


def test_events_table_exclusivity():
    t = datetime(2026, 9, 1, tzinfo=UTC)
    rows = []
    for ev, ser, st, n in (("N-1", "KXNETFLIXRANKSHOW", None, 4), ("T-1", "KXTRUTHSOCIAL", "between", 3),
                           ("G-1", "KXTRUMPAPPROVE", "greater", 5), ("R-1", "KXRANKLISTSONGTOP10", None, 5)):
        for i in range(n):
            rows.append({"event_ticker": ev, "series_ticker": ser, "family": cf.family_of(ser), "close_time": t,
                         "open_time": t - timedelta(days=7), "y": int(i == 0), "strike_type": st})
    ev = cf.events_table(pl.DataFrame(rows)).sort("event_ticker")
    assert dict(zip(ev["event_ticker"], ev["exclusive"])) == {"G-1": False, "N-1": True, "R-1": False, "T-1": True}


def test_pm_edge_rows():
    df = pl.DataFrame({"buy": [True, False, True], "outcome_index": [0, 0, 1], "winner_index": [0, 0, 0],
                       "price": [0.3, 0.8, 0.1], "size": [10.0, 5.0, 20.0]})
    out = df.with_columns(*cf.edge_exprs())
    assert np.allclose(out["edge"].to_numpy(), [7.0, -1.0, -2.0])    # buy won; sold the winner at 0.8; buy lost
    assert np.allclose(out["risk"].to_numpy(), [3.0, 1.0, 2.0])


def test_twin_taker_direction_and_fee():
    t = datetime(2026, 9, 1, tzinfo=UTC)
    g = pl.DataFrame([{"pair_id": "p", "fam": "netflix", "ticker": "A", "event_ticker": "E", "h": 6, "T": t,
                       "split": "dev", "day": t.date(), "k_settle": t, "pm": 0.6, "eask": 0.5, "ebid": 0.46,
                       "y": 1, "fee_multiplier": 1.0}])
    tk = cf.twin_taker(g, 0.05)
    r = tk.row(0, named=True)
    assert r["dir"] == 1 and r["n"] == 9
    fee = math.ceil(0.07 * 9 * 0.25 * 100) / 100
    assert math.isclose(r["pnl_c"], 100 * (1 - 0.5 - fee / 9))
    assert cf.twin_taker(g.with_columns(pl.lit(0.52).alias("pm")), 0.05).is_empty()


def test_pick_requires_size_and_positive_mean():
    cells = pl.DataFrame({"cell": ["a", "b", "c", "d"], "n": [100, 100, 30, 200], "events": [30, 30, 30, 10],
                          "mean_c": [1.0, 2.0, 9.0, 5.0], "t_2way": [3.0, 2.5, 9.0, 9.0]})
    assert cf.pick(cells, 60, 20)["cell"] == "a"
    assert cf.pick(cells.with_columns(pl.lit(-1.0).alias("mean_c")), 60, 20) is None


def test_fam_filter_counts_group():
    df = pl.DataFrame({"family": ["truth", "exec", "netflix"]})
    assert cf.fam_filter(df, "counts").height == 2
    assert cf.fam_filter(df, "all").height == 3
    assert cf.fam_filter(df, "netflix").height == 1


def test_weeks_positive():
    t0 = datetime(2026, 9, 7, tzinfo=UTC)
    df = pl.DataFrame({"t": [t0, t0 + timedelta(days=1), t0 + timedelta(days=8)], "pnl_c": [5.0, -1.0, -3.0]})
    assert cf.weeks_positive(df) == 50.0
