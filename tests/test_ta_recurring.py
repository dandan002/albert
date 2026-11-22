import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import ta_recurring as tr
from albert.research.xcat import OHLC_SCHEMA, ohlc_rows

FIXTURE = Path(__file__).parent / "fixtures" / "kalshi" / "candles_hourly_KXTRUMPACT-26MAY24-T4.json"
TICKER = "KXTRUMPACT-26MAY24-T4"


def _meta() -> pl.DataFrame:
    # the market's real listing fields (recorded /historical/markets row)
    return pl.DataFrame({
        "ticker": [TICKER], "event_ticker": ["KXTRUMPACT-26MAY24"], "series_ticker": ["KXTRUMPACT"], "family": ["exec"],
        "split": ["dev"], "strike_type": ["greater_or_equal"], "result": ["no"],
        "open_time": [datetime(2026, 5, 24, 14, tzinfo=UTC)], "close_time": [datetime(2026, 5, 31, 14, tzinfo=UTC)],
        "expected_expiration_time": [datetime(2026, 5, 31, 14, tzinfo=UTC)],
        "settlement_ts": [datetime(2026, 5, 31, 18, 46, tzinfo=UTC)], "fee_type": ["quadratic"], "fee_multiplier": [1.0]})


def _candles() -> pl.DataFrame:
    rec = json.loads(FIXTURE.read_text())
    rows = ohlc_rows(rec["response"], TICKER, 60)
    return pl.DataFrame(rows, schema=OHLC_SCHEMA).select(tr.OHLC_COLS)


def test_fixture_panel_is_regular_and_causal():
    c = _candles()
    assert c.height == 72 and c["p_low"].null_count() > 0          # quiet hours have no prints
    p = tr.build_panel(c, _meta())
    assert p.height > c.height                                     # Kalshi omits unchanged hours: upsampled
    steps = p["t"].diff().drop_nulls().dt.total_minutes().unique().to_list()
    assert steps == [60]
    assert (p["t"] < datetime(2026, 5, 31, 14, tzinfo=UTC)).all()
    # indicators use hours <= t only: r6 is the carried signal mid minus its value 6 grid hours back
    x = p["x"].to_numpy()
    r6 = p["r6"].to_numpy()
    for i in range(6, len(x)):
        if not (np.isnan(x[i]) or np.isnan(x[i - 6])):
            assert abs(r6[i] - (x[i] - x[i - 6])) < 1e-12
    assert p["bid"].null_count() == 0                              # quotes are carried forward


def test_maker_fills_only_on_trade_through_with_size():
    t0 = datetime(2026, 6, 1, tzinfo=UTC)
    base = {"ticker": "M", "event_ticker": "E", "series_ticker": "S", "family": "netflix", "cluster": "E", "hb": ">48h",
            "et_date": t0.date(), "et_hour": 20, "holdout": False, "y": 1.0, "settlement_ts": t0 + timedelta(days=3),
            "fee_type": "quadratic", "fee_multiplier": 1.0, "h_to_exp": 60.0}
    orders = pl.DataFrame([{**base, "t": t0, "bid": 0.29, "ask": 0.32, "mid": 0.305, "spread": 0.03, "dir": 1},
                           {**base, "t": t0, "bid": 0.29, "ask": 0.32, "mid": 0.305, "spread": 0.03, "dir": -1}])
    hours = [t0 + timedelta(hours=k) for k in range(1, 5)]
    panel = pl.DataFrame({"ticker": ["M"] * 4, "t": hours,
                          "plo": [0.30, 0.29, 0.28, 0.25], "phi": [0.31, 0.31, 0.31, 0.31],   # 0.30 = our bid: not through
                          "volume": [50.0, 5.0, 50.0, 50.0]})                                # 5 < our 10 contracts
    f = tr.maker(orders, panel)
    yes = f.filter(pl.col("dir") == 1).row(0, named=True)
    assert yes["price"] == 0.30 and yes["n"] == 10
    assert yes["filled"] and yes["t_fill"] == hours[2]          # first print strictly below 30c with >= 10 traded
    assert abs(yes["pnl"] - (1 - 0.30)) < 1e-12                  # quadratic series: no maker fee
    no = f.filter(pl.col("dir") == -1).row(0, named=True)
    assert abs(no["price"] - 0.69) < 1e-12                        # NO bid 1c inside: YES offer at 31c
    assert not no["filled"]                                       # no YES print above 31c


def test_taker_fee_rounded_up_per_order():
    f = pl.DataFrame({"dir": [1], "bid": [0.28], "ask": [0.30], "mid": [0.29], "y": [0.0], "t": [datetime(2026, 6, 1, tzinfo=UTC)],
                      "fee_multiplier": [1.0]})
    r = tr.taker(f).row(0, named=True)
    assert r["n"] == 10
    assert abs(r["fee_order"] - 0.15) < 1e-12                    # ceil(0.07 * 10 * 0.3 * 0.7 * 100) / 100
    assert abs(r["pnl"] - (-0.30 - 0.015)) < 1e-12


def test_cross_sectional_top_riser():
    t0 = datetime(2026, 6, 1, 0, tzinfo=UTC)
    rows = []
    for leg, (x24, x) in {"A": (0.50, 0.40), "B": (0.30, 0.45), "C": (0.20, 0.15)}.items():
        for t, v in ((t0, x24), (t0 + timedelta(hours=24), x)):
            rows.append({"ticker": f"E-{leg}", "event_ticker": "E", "series_ticker": "KXNETFLIXRANKSHOW", "family": "netflix",
                         "strike_type": "custom", "cluster": "E", "hb": ">48h", "t": t, "et_date": t.date(), "et_hour": 20,
                         "holdout": False, "bid": v - 0.01, "ask": v + 0.01, "mid": v, "spread": 0.02, "smid": v, "x": v,
                         "x24": x24 if t > t0 else None, "y": 1.0 if leg == "B" else 0.0, "settlement_ts": t0 + timedelta(days=3),
                         "fee_type": "quadratic", "fee_multiplier": 1.0, "h_to_exp": 60.0})
    o = tr.xs_orders(pl.DataFrame(rows))
    riser = o.filter((pl.col("signal") == "xs top riser (24h share)") & (pl.col("mode") == "trend"))
    assert riser["ticker"].to_list() == ["E-B"] and riser["dir"].to_list() == [1]
    leader = o.filter((pl.col("signal") == "xs new leader") & (pl.col("mode") == "fade"))
    assert leader["ticker"].to_list() == ["E-B"] and leader["dir"].to_list() == [-1]


def test_cluster_t_and_holm():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(50), 4)
    num = rng.normal(0.05, 0.1, 200)
    mu, t = tr.cluster_t(num, np.ones(200), g)
    assert abs(mu - num.mean()) < 1e-12 and t > 2
    _, t2 = tr.cluster_t(num, np.ones(200), g, np.repeat(np.arange(20), 10))
    assert t2 is not None
    ok = tr.holm(np.array([5.0, 0.5, 3.5]))
    assert ok.tolist() == [True, False, True]


def test_family_and_publication_window():
    s = pl.DataFrame({"series_ticker": ["KXNETFLIXRANKSHOW", "KXTOPSONG", "KXTRUMPAPPROVE", "KXANTHVREQ", "KXNFLGAME"]})
    fam = s.select(tr.family_expr(pl.col("series_ticker")).alias("f"))["f"].to_list()
    assert fam == ["netflix", "billboard", "approval", "ai_share", None]
    w = pl.DataFrame({"fam": ["netflix", "netflix", "billboard", "approval"], "dow": [2, 4, 7, 6], "h": [16, 16, 13, 15]})
    assert w.select(tr.pub_window(pl.col("fam"), pl.col("dow"), pl.col("h")).alias("p"))["p"].to_list() == [True, False, True, False]
