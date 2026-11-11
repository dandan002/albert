import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from albert.research import flb_xs as X
from albert.venues import kalshi as k

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _hist(name: str) -> dict:
    return json.loads((FIX / f"flb_xs_historical_markets_{name}.json").read_text())


# ---- parsers on recorded payloads ---------------------------------------------------------------
def test_frame_rows_on_recorded_weather_ladder():
    rows = X.frame_rows(_hist("HIGHNY-23JAN05"), "Climate and Weather", "historical")
    df = pl.DataFrame(rows, schema=X.FRAME_SCHEMA)
    assert df.height >= 6
    assert set(df["result"]) <= {"yes", "no"} and df["y" if "y" in df.columns else "result"].len() == df.height
    assert (df["result"] == "yes").sum() == 1                      # one bucket of the ladder wins
    assert set(df["strike_type"]) >= {"between", "greater", "less"}
    assert df["volume"].min() > 0 and df["dl_kind"].null_count() == df.height
    t59 = df.filter(pl.col("ticker") == "HIGHNY-23JAN05-T59").row(0, named=True)
    assert t59["floor_strike"] == 59 and t59["close_time"] == datetime(2023, 1, 6, 4, 59, tzinfo=UTC)
    assert t59["occurrence_time"] is None


def test_frame_rows_on_recorded_election():
    df = pl.DataFrame(X.frame_rows(_hist("PRES"), "Elections", "historical"), schema=X.FRAME_SCHEMA)
    win = df.filter(pl.col("result") == "yes")
    assert win["ticker"].to_list() == ["PRES-2024-DJT"]
    # the market stayed open until the inauguration: the scheduled close is not a pre-outcome anchor
    assert win["close_time"][0] > datetime(2025, 1, 1, tzinfo=UTC)


def test_candles_and_quotes_from_recorded_payload():
    payload = json.loads((FIX / "flb_xs_candles_HIGHNY-23JAN05-T59.json").read_text())
    tk = payload["ticker"]
    c = pl.DataFrame([{kk: r[kk] for kk in X.CANDLE_COLS} for r in k.candle_rows(payload, tk, 60)],
                     schema=X.CANDLE_COLS)
    assert c.height == 12 and c["yes_ask_close"].min() >= 0.01
    T = datetime(2023, 1, 5, 5, 0, tzinfo=UTC)     # midnight ET of the measurement day
    smp = pl.DataFrame({"ticker": [tk], "group": ["Weather"], "sched_src": ["date"], "T": [T],
                        "open_time": [datetime(2023, 1, 4, 15, 0, tzinfo=UTC)],
                        "close_time": [datetime(2023, 1, 6, 4, 59, tzinfo=UTC)]})
    g = X.lead_rows(smp)
    assert sorted(g["lead_h"].to_list()) == [1.0, 6.0]          # the market opened 14 h before T
    q = X.attach_quotes(g, c)
    assert q.height == 2
    assert (q["quote_ts"] <= q["ts"]).all()                       # no look-ahead
    one = q.filter(pl.col("lead_h") == 1.0).row(0, named=True)
    last = c.filter(pl.col("end_ts") <= one["ts"]).sort("end_ts").row(-1, named=True)
    assert one["yes_ask_close"] == last["yes_ask_close"] and one["yes_bid_close"] == last["yes_bid_close"]
    assert one["vol_7d"] == pytest.approx(c.filter(pl.col("end_ts") <= one["ts"])["volume"].sum())


# ---- scheduled time, strikes ----------------------------------------------------------------------
def test_ticker_date_and_leg_strike():
    assert X.ticker_date("KXVOTESWEDEN-26SEP13SDP") == datetime(2026, 9, 13)
    assert X.ticker_date("KXYTVIEWSD-TAY26SEP21") == datetime(2026, 9, 21)
    assert X.ticker_date("KXMLBGAME-26SEP142040SDCOL") == datetime(2026, 9, 14)
    assert X.ticker_date("KXNBA-26") is None and X.ticker_date("PRES-2024") is None
    assert X.leg_strike("HIGHNY-23JAN05-T59", "HIGHNY-23JAN05") == ("", 59.0)
    assert X.leg_strike("HIGHNY-23JAN05-B58.5", "HIGHNY-23JAN05") == ("", 58.5)
    assert X.leg_strike("KXNFLSPREAD-26OCT29CARGB-CAR8", "KXNFLSPREAD-26OCT29CARGB") == ("CAR", 8.0)
    assert X.leg_strike("KXNBAPTS-26OCT31PHICHA-PHITMAXEY0-20", "KXNBAPTS-26OCT31PHICHA") == ("PHITMAXEY0-", 20.0)
    assert X.leg_strike("PRES-2024-DJT", "PRES-2024") == ("DJT", None)


def test_scheduled_time_rules():
    close = datetime(2026, 9, 10, 5, 0, tzinfo=UTC)
    T, src = X.scheduled_time("Weather", "daily", "KXHIGHNY-26SEP09", None, close, None)
    assert src == "date" and T == datetime(2026, 9, 9, 4, 0, tzinfo=UTC)       # midnight EDT
    # Tokyo: scheduled close at local midnight (15:00 UTC) -> the measurement day starts 24 h earlier
    T, _ = X.scheduled_time("Weather", "daily", "KXHIGHTRJTT-26SEP09", None, datetime(2026, 9, 9, 15, tzinfo=UTC), None)
    assert T == datetime(2026, 9, 8, 15, tzinfo=UTC)
    st = datetime(2026, 9, 15, 0, 40, tzinfo=UTC)
    assert X.scheduled_time("Sports", None, "KXMLBGAME-26SEP142040SDCOL", st, None, None) == (st, "start")
    T, src = X.scheduled_time("Sports", None, "KXNBAGAME-26NOV06SASSAC", None, None, None)
    assert src == "date" and T == datetime(2026, 11, 5, 10, 0, tzinfo=UTC)     # midnight at UTC+14
    # an election: the ticker date (election day) comes before the scheduled close
    T, src = X.scheduled_time("Politics", "custom", "KXVOTESWEDEN-26SEP13SDP", None,
                              datetime(2026, 9, 19, 15, tzinfo=UTC), None)
    assert src == "date" and T == datetime(2026, 9, 13, 4, 0, tzinfo=UTC)
    # a chart dated after its close: the close comes first
    cl = datetime(2026, 9, 14, 3, 59, tzinfo=UTC)
    assert X.scheduled_time("Entertainment", "weekly", "KXBILLBOARD-26SEP19", None, cl, None) == (cl, "close")
    dl = datetime(2026, 10, 1, 4, tzinfo=UTC)
    assert X.scheduled_time("Politics", None, "KXFOO-26SEP01", None, cl, dl) == (dl, "deadline")


def test_lead_rows_guard_and_inplay():
    T = datetime(2026, 9, 15, 0, 0, tzinfo=UTC)
    smp = pl.DataFrame({"ticker": ["A", "B"], "group": ["Sports", "Politics"], "sched_src": ["start", "date"],
                        "T": [T, T], "open_time": [T - timedelta(days=2), T - timedelta(days=40)],
                        "close_time": [T + timedelta(hours=1, minutes=10), T - timedelta(hours=6, minutes=10)]})
    g = X.lead_rows(smp)
    a = sorted(g.filter(pl.col("ticker") == "A")["lead_h"].to_list())
    assert a == [-0.5, 1.0, 6.0, 24.0]          # -1 h fails the 15-min guard, -2 h is after the close
    b = sorted(g.filter(pl.col("ticker") == "B")["lead_h"].to_list())
    assert b == [24.0, 72.0, 168.0, 720.0]      # 6 h is 10 min before the close: dropped by the guard


# ---- statistics ---------------------------------------------------------------------------------
def test_logit_fit_recovers_slope():
    rng = np.random.default_rng(0)
    n = 40_000
    q = rng.uniform(0.02, 0.98, n)
    g = np.arange(n) // 4
    for b_true in (1.0, 1.3):
        p = 1 / (1 + np.exp(-b_true * np.log(q / (1 - q))))
        y = (rng.uniform(size=n) < p).astype(float)
        f = X.logit_fit(y, X.logit(q), g)
        assert abs(f["b"] - b_true) < 4 * f["se_b"] and abs(f["a"]) < 4 * f["se_a"]
    assert f["t_b1"] > 5


def test_ols_fit_difference_of_means():
    rng = np.random.default_rng(1)
    d = rng.integers(0, 2, 2000).astype(float)
    yv = 0.3 * d + rng.normal(size=2000)
    r = X.ols_fit(yv, np.column_stack([np.ones(2000), d]), np.arange(2000) // 2)
    assert r["coef"][1] == pytest.approx(yv[d == 1].mean() - yv[d == 0].mean())


def test_shin_z_inverts_shin_prices():
    pi = np.array([0.55, 0.25, 0.12, 0.05, 0.03])
    for z in (0.0, 0.03, 0.1):
        r = np.sqrt(z * pi + (1 - z) * pi ** 2)
        prices = r * r.sum()                      # Shin (1993) quotes, sum > 1 when z > 0
        zh, ph = X.shin_z(prices)
        if z == 0:
            assert zh == 0.0 and np.allclose(ph, pi)
        else:
            assert zh == pytest.approx(z, abs=1e-8) and np.allclose(ph, pi, atol=1e-8)
    # Shin moves probability from longshots to favourites relative to normalising
    zh, ph = X.shin_z(np.array([0.6, 0.3, 0.2]))
    norm = np.array([0.6, 0.3, 0.2]) / 1.1
    assert ph[0] > norm[0] and ph[-1] < norm[-1]


def test_fee_per_dollar_rounding():
    assert X.fee_per_dollar(np.array([0.01]))[0] == pytest.approx(0.07 * 0.99)
    assert X.fee_per_dollar(np.array([0.01]), 1)[0] == pytest.approx(1.0)      # 1c fee on a 1c contract
    assert X.fee_per_dollar(np.array([0.5]), 100)[0] == pytest.approx(1.75 / 50)          # $1.75 on $50


def test_sides_returns_and_fees():
    p = pl.DataFrame({"ticker": ["A"], "event_ticker": ["E"], "series_ticker": ["S"], "group": ["Politics"],
                      "structure": ["binary"], "lead_h": [24.0], "phase": ["pre"], "ts": [datetime(2026, 1, 1, tzinfo=UTC)],
                      "month": ["2026-01"], "vol_7d": [10.0], "open_interest": [5.0], "spread": [0.02],
                      "fee_multiplier": [1.0], "maker_fee": [False], "sched_src": ["date"], "yes_bid": [0.04],
                      "yes_ask": [0.06], "mid": [0.05], "y": [0], "n_legs": [1]})
    s = X.sides(X.with_weights(p)).sort("side")
    no, yes = s.row(0, named=True), s.row(1, named=True)
    assert yes["q_mid"] == pytest.approx(0.05) and no["q_mid"] == pytest.approx(0.95)
    assert yes["q_ask"] == pytest.approx(0.06) and no["q_ask"] == pytest.approx(0.96)
    assert yes["r_mid"] == pytest.approx(-1.0) and no["r_mid"] == pytest.approx(0.05 / 0.95)
    assert no["r_net"] < no["r_ask"] < no["r_mid"]
    assert no["fee"] == pytest.approx(0.07 * 0.96 * 0.04)
    assert no["mfee"] == 0.0                     # quadratic series: no maker fee
    assert yes["bucket"] == "3-5c" and no["bucket"] == "91-95c"   # upper edges inclusive


def test_water_fill():
    q = X.water_fill({"ladder": 1000, "binary": 50, "multi": 0, "deadline": 10}, 400)
    assert q["binary"] == 50 and q["deadline"] == 10 and q["multi"] == 0
    assert q["ladder"] == pytest.approx(340)
    q = X.water_fill({"a": 1000, "b": 1000}, 400)
    assert q == {"a": 200, "b": 200}


def test_vectorised_rules_match_scalar():
    cases = [  # group, frequency, event, ticker, sports_start, t_close_sched, dl_T, strike_type, floor, cap
        ("Weather", "daily", "KXHIGHNY-26SEP09", "KXHIGHNY-26SEP09-T91", None, datetime(2026, 9, 10, 5, tzinfo=UTC), None, None, None, None),
        ("Weather", "daily", "KXHIGHTRJTT-26SEP09", "KXHIGHTRJTT-26SEP09-B30.5", None, datetime(2026, 9, 9, 15, tzinfo=UTC), None, "between", 30.0, 31.0),
        ("Sports", None, "KXMLBGAME-26SEP142040SDCOL", "KXMLBGAME-26SEP142040SDCOL-SD",
         datetime(2026, 9, 15, 0, 40, tzinfo=UTC), datetime(2026, 9, 15, 4, tzinfo=UTC), None, "structured", None, None),
        ("Sports", None, "KXNBAGAME-26NOV06SASSAC", "KXNBAGAME-26NOV06SASSAC-SAS", None, None, None, None, None, None),
        ("Sports", None, "KXNBA-26", "KXNBA-26-LAL", None, datetime(2026, 6, 20, tzinfo=UTC), None, None, None, None),
        ("Politics", "custom", "KXVOTESWEDEN-26SEP13SDP", "KXVOTESWEDEN-26SEP13SDP-M", None, datetime(2026, 9, 19, 15, tzinfo=UTC), None, None, None, None),
        ("Entertainment", "weekly", "KXBILLBOARD-26SEP19", "KXBILLBOARD-26SEP19-X", None, datetime(2026, 9, 14, 3, 59, tzinfo=UTC), None, None, None, None),
        ("Politics", None, "KXFOO-26SEP01", "KXFOO-26SEP01", None, datetime(2026, 9, 14, tzinfo=UTC), datetime(2026, 10, 1, 4, tzinfo=UTC), None, None, None),
        ("Sports", None, "KXNFLSPREAD-26OCT29CARGB", "KXNFLSPREAD-26OCT29CARGB-CAR8", None, None, None, None, None, None),
        ("Elections", None, "PRES-2024", "PRES-2024-DJT", None, datetime(2025, 1, 20, tzinfo=UTC), None, None, None, None),
    ]
    cols = ["group", "frequency", "event_ticker", "ticker", "sports_start", "t_close_sched", "dl_T", "strike_type",
            "floor_strike", "cap_strike"]
    df = pl.DataFrame([dict(zip(cols, c, strict=True)) for c in cases],
                      schema={"group": pl.String, "frequency": pl.String, "event_ticker": pl.String, "ticker": pl.String,
                              "sports_start": pl.Datetime("us", "UTC"), "t_close_sched": pl.Datetime("us", "UTC"),
                              "dl_T": pl.Datetime("us", "UTC"), "strike_type": pl.String, "floor_strike": pl.Float64,
                              "cap_strike": pl.Float64})
    out = df.with_columns(*X.scheduled_time_exprs(), *X.leg_strike_exprs())
    for c, r in zip(cases, out.iter_rows(named=True), strict=True):
        assert (r["T"], r["sched_src"]) == X.scheduled_time(c[0], c[1], c[2], c[4], c[5], c[6])
        base, k = X.leg_strike(c[3], c[2])
        assert r["ladder_base"] == base
        if c[7] is None:
            assert r["strike"] == k
    assert out["strike"][1] == 30.5 and out["strike"][2] is None


def _panel_rows(rows: list[dict]) -> pl.DataFrame:
    base = {"series_ticker": "S", "group": "Politics", "lead_h": 24.0, "phase": "pre",
            "ts": datetime(2026, 1, 1, tzinfo=UTC), "month": "2026-01", "vol_7d": 10.0, "open_interest": 5.0,
            "fee_multiplier": 1.0, "maker_fee": False, "sched_src": "date", "subcent": False, "p_event": 1.0}
    df = pl.DataFrame([{**base, **r} for r in rows])
    return df.with_columns(((pl.col("yes_bid") + pl.col("yes_ask")) / 2).alias("mid"),
                           (pl.col("yes_ask") - pl.col("yes_bid")).alias("spread"))


def test_multi_outcome_overround_and_field(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "OUT", tmp_path)
    rows = []
    for e in range(30):                       # 3-leg exclusive events: favourite always wins
        for leg, (b, a) in enumerate(((0.60, 0.64), (0.25, 0.28), (0.08, 0.10))):
            rows.append({"ticker": f"E{e}-{leg}", "event_ticker": f"E{e}", "structure": "multi", "n_legs": 3,
                         "mutually_exclusive": True, "yes_bid": b, "yes_ask": a, "y": int(leg == 0)})
    out = X.multi_outcome(X.with_weights(_panel_rows(rows)))
    m = out["multi"]
    assert m["events"] == 30
    assert m["over_ask"]["mean"] == pytest.approx(0.64 + 0.28 + 0.10 - 1)
    assert m["over_mid"]["mean"] == pytest.approx(0.62 + 0.265 + 0.09 - 1)
    assert 0 < m["shin_z_median"] < 0.1
    fav = next(r for r in m["by_rank"] if r["rank_class"] == "1 (favourite)")
    assert fav["e_mid"]["mean"] == pytest.approx(1 - 0.62)
    assert m["field"]["price"] == pytest.approx(0.09) and m["field"]["win"] == 0


def test_floor_decomposition_mechanical_share():
    rows = []
    for i in range(200):                     # pinned at the floor (mid 0.5c), never wins
        rows.append({"ticker": f"P{i}", "event_ticker": f"P{i}", "structure": "binary", "n_legs": 1,
                     "yes_bid": 0.0, "yes_ask": 0.01, "y": 0})
    for i in range(200):                     # above the floor (mid 1.5c), wins 1 in 200
        rows.append({"ticker": f"A{i}", "event_ticker": f"A{i}", "structure": "binary", "n_legs": 1,
                     "yes_bid": 0.01, "yes_ask": 0.02, "y": int(i == 0)})
    pre = X.with_weights(_panel_rows(rows))
    fl = X.floor_analysis(pre, X.sides(pre))
    d = fl["decomposition"]
    assert d["pinned_at_floor"]["mechanical_share"] == pytest.approx(1.0)
    assert d["above_floor"]["mechanical_pc"] == 0.0
    # all 1-2c sides: loss = mean price - mean win; mechanical = half the weight x 0.5c
    allr = d["all_1_2c"]
    assert allr["loss_pc"] == pytest.approx((0.005 + 0.015) / 2 - 0.5 / 200)
    assert allr["mechanical_pc"] == pytest.approx(0.5 * 0.005)
    assert fl["share_quotes_pinned"] == pytest.approx(0.5)


def test_flb_measures_detects_bias():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(4000):
        mid = rng.choice([0.03, 0.10, 0.30, 0.50, 0.70, 0.90, 0.97])
        p_true = 1 / (1 + np.exp(-1.4 * np.log(mid / (1 - mid))))       # FLB: true odds more extreme
        rows.append({"ticker": f"M{i}", "event_ticker": f"E{i // 2}", "structure": "binary", "n_legs": 1,
                     "yes_bid": mid - 0.005, "yes_ask": mid + 0.005, "y": int(rng.uniform() < p_true)})
    p = X.with_weights(_panel_rows(rows))
    m = X.flb_measures(p, None, "w_m")
    assert m["logit"]["b"] > 1.2 and m["logit"]["t_b1"] > 3
    assert m["ls_r_mid"]["mean"] < 0 < m["fav_r_mid"]["mean"]
    assert m["spread_mid"]["diff"] < 0
    rows_cal = X.bucket_table(X.sides(p))
    assert {r["bucket"] for r in rows_cal} >= {"3-5c", "6-10c", "81-90c", "96-98c"}
