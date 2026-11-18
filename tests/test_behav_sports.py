"""Behavioural sports study: parsers on recorded ESPN / Kalshi payloads, Kalshi <-> ESPN matching,
as-of quotes and team history without look-ahead, statistics, fees and fills."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from albert.research import behav_sports as bs
from albert.research import xcat

FIX = Path(__file__).parent / "fixtures"


def _espn() -> pl.DataFrame:
    payload = json.loads((FIX / "espn" / "behav_sports_nfl_202510.json").read_text())
    return pl.DataFrame(bs.espn_game_rows(payload, "NFL"), schema=bs.ESPN_SCHEMA)


def _markets() -> pl.DataFrame:
    payload = json.loads((FIX / "kalshi" / "behav_sports_hist_markets_KXNFLGAME.json").read_text())
    rows = bs._market_rows(payload, "hist", {"KXNFLGAME": ("quadratic_with_maker_fees", 1.0)})
    m = pl.DataFrame(rows, infer_schema_length=None)
    return m.with_columns(pl.lit("NFL").alias("league"), pl.col("ticker").str.split("-").list.last().alias("code"))


def _candles() -> pl.DataFrame:
    payload = json.loads((FIX / "kalshi" / "behav_sports_candles_hist_SFLA.json").read_text())
    return pl.DataFrame(xcat.ohlc_rows(payload, payload["ticker"], 60), schema=xcat.OHLC_SCHEMA)


# ---- parsers on recorded payloads -----------------------------------------------------------------
def test_espn_rows_parse_recorded_scoreboard():
    e = _espn()
    assert e.height == 3
    sf = e.filter(pl.col("away_abbr") == "SF").row(0, named=True)
    assert sf["home_abbr"] == "LAR" and sf["start"] == datetime(2025, 10, 3, 0, 15, tzinfo=UTC)
    assert sf["national_tv"] and sf["big_tv"] and not sf["neutral"]          # Thursday night, Prime Video
    assert isinstance(sf["home_score"], float) and sf["home_score"] != sf["away_score"]
    london = e.filter(pl.col("home_abbr").is_in(["CLE", "MIN"])).row(0, named=True)
    assert london["neutral"]


def test_market_rows_and_games_from_recorded_listing():
    m = _markets()
    assert m.height == 6 and set(m["result"]) <= {"yes", "no"}
    g = bs.kalshi_games(m)
    assert g.height == 3
    sf = g.filter(pl.col("event_ticker") == "KXNFLGAME-25OCT02SFLA").row(0, named=True)
    assert (sf["first"], sf["second"]) == ("SF", "LA")                       # US tickers: away first
    assert sf["et_date"] == date(2025, 10, 2)
    assert (g["y_first"] + g["y_second"]).to_list() == [1, 1, 1]


def test_match_espn_on_recorded_payloads():
    g = bs.kalshi_games(_markets())
    x = bs.match_espn(g, _espn())
    assert x.height == 3 and x["espn_id"].n_unique() == 3
    j = g.join(x, on="event_ticker")
    sf = j.filter(pl.col("event_ticker") == "KXNFLGAME-25OCT02SFLA").row(0, named=True)
    assert not sf["first_home"] and sf["first_name"] == "San Francisco 49ers"
    # Kalshi results agree with ESPN's final scores
    agree = j.select(((pl.col("first_score") > pl.col("second_score")).cast(pl.Int64) == pl.col("y_first")).all()).item()
    assert agree


def test_parse_event_variants():
    p = bs.parse_event("KXMLBGAME-26SEP031310TORCLE", ["CLE", "TOR"])
    assert p == {"et_date": date(2026, 9, 3), "first": "TOR", "second": "CLE"}
    p = bs.parse_event("KXNFLGAME-26OCT29CARGB", ["GB", "CAR"])
    assert (p["first"], p["second"]) == ("CAR", "GB")
    p = bs.parse_event("KXEPLGAME-26OCT31AVLFUL", ["TIE", "FUL", "AVL"])
    assert (p["first"], p["second"]) == ("AVL", "FUL")                       # soccer: home first
    assert bs.parse_event("KXNFLGAME-26OCT29CARGB", ["GB"]) is None
    assert bs.parse_event("KXNFLGAME-XXOCT29CARGB", ["GB", "CAR"]) is None


def test_batch_rows_parse_recorded_batch():
    payload = json.loads((FIX / "kalshi" / "behav_sports_candles_batch.json").read_text())
    got = xcat.batch_rows(payload, 60)
    assert len(got) == 2 and all(len(v) == 12 for v in got.values())
    r = next(iter(got.values()))[-1]
    assert r["end_ts"] is not None and (r["bid_close"] is None or 0 <= r["bid_close"] <= 1)


# ---- as-of quotes ----------------------------------------------------------------------------------
def test_leg_quotes_use_only_candles_at_or_before_decision():
    c = _candles()
    ok = c.filter((pl.col("bid_close") > 0) & (pl.col("ask_close") < 1) & (pl.col("ask_close") > pl.col("bid_close"))).sort("end_ts")
    assert ok.height >= 5
    ref = ok.row(ok.height // 2, named=True)
    t = ref["end_ts"] + timedelta(minutes=30)                                # before the next hourly candle ends
    req = pl.DataFrame({"ticker": [ref["ticker"], ref["ticker"]], "t": [t, c["end_ts"].min() - timedelta(hours=1)]}).with_row_index("rid")
    q = bs.leg_quotes(c.lazy(), req).sort("rid")
    first = q.row(0, named=True)
    nxt = ok.filter(pl.col("end_ts") > ref["end_ts"]).row(0, named=True)
    assert nxt["end_ts"] > t
    assert (first["bid"], first["ask"], first["q_ts"]) == (ref["bid_close"], ref["ask_close"], ref["end_ts"])
    vol = c.filter(pl.col("end_ts") <= t)["volume"].fill_null(0).sum()
    assert first["cumvol"] == pytest.approx(vol)
    assert q.row(1, named=True)["bid"] is None and q.row(1, named=True)["cumvol"] == 0.0


# ---- team history ----------------------------------------------------------------------------------
def _hist_panel() -> pl.DataFrame:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    starts = [t0, t0 + timedelta(days=2), t0 + timedelta(days=4), t0 + timedelta(days=6), t0 + timedelta(days=80)]
    ys = [1, 1, 0, 1, 0]
    ps = [0.30, 0.50, 0.70, 0.40, 0.55]
    rows = []
    for i, (s, y, p) in enumerate(zip(starts, ys, ps, strict=True)):
        ev = f"EV{i}"
        rows.append({"league": "NBA", "team": "AAA", "opp": f"O{i}", "event_ticker": ev, "start": s,
                     "settlement_ts": s + timedelta(hours=3), "mid1": p, "y": y, "pts": 100.0 + 10 * y, "opp_pts": 100.0})
        rows.append({"league": "NBA", "team": f"O{i}", "opp": "AAA", "event_ticker": ev, "start": s,
                     "settlement_ts": s + timedelta(hours=3), "mid1": 1 - p, "y": 1 - y, "pts": 100.0, "opp_pts": 100.0 + 10 * y})
    return pl.DataFrame(rows)


def test_add_history_uses_previous_games_only():
    h = bs.add_history(_hist_panel()).filter(pl.col("team") == "AAA").sort("start")
    assert h["prev_surprise"][0] is None
    assert h["prev_surprise"][1] == pytest.approx(1 - 0.30)                  # game 1: won at 30c
    assert h["prev_surprise"][2] == pytest.approx(1 - 0.50)
    assert h["prev_streak"].to_list()[1:4] == [1, 2, -1]
    assert h["prev_p"][3] == pytest.approx(0.70) and h["prev_win"][3] == 0
    assert h["prev_surprise"][4] is None                                     # 74-day gap: new season
    o = bs.add_history(_hist_panel()).filter(pl.col("opp") == "AAA").sort("start")
    assert o["opp_prev_streak"].to_list()[1:4] == [1, 2, -1]                 # attached to the opponent row


# ---- statistics ------------------------------------------------------------------------------------
def test_cmean_clusters_and_cslope():
    rng = np.random.default_rng(3)
    n = 400
    d = pl.DataFrame({"r": rng.normal(0.02, 0.4, n), "event_ticker": [f"g{i}" for i in range(n)],
                      "start_date": [date(2026, 1, 1 + i % 20) for i in range(n)]})
    out = bs.cmean(d)
    assert out["n"] == n and out["games"] == n
    assert out["mean_c"] == pytest.approx(100 * d["r"].mean(), abs=0.01)
    se_iid = d["r"].std() / np.sqrt(n)
    assert out["se_c"] >= 0.5 * 100 * se_iid
    x = rng.normal(0, 1, n)
    d2 = d.with_columns(pl.Series("x", x), pl.Series("yv", 0.3 * x + rng.normal(0, 0.01, n)))
    s = bs.cslope(d2, "yv", "x")
    assert s["slope"] == pytest.approx(0.3, abs=0.01) and s["t"] > 50


# ---- fees, fills, ledger ---------------------------------------------------------------------------
def test_fee_rounding_and_maker_series():
    f = pl.DataFrame({"n": [10.0, 10.0, 10.0, 10.0], "p": [0.5, 0.5, 0.5, 0.5], "fee_multiplier": [1.0, 0.5, 1.0, 1.0],
                      "fee_type": ["quadratic", "quadratic_with_maker_fees", "quadratic", "quadratic_with_maker_fees"]})
    out = f.with_columns(bs.fee_order(pl.col("n"), pl.col("p")).alias("taker"),
                         bs.fee_order(pl.col("n"), pl.col("p"), maker=True).alias("maker"))
    assert out["taker"].to_list() == [0.18, 0.09, 0.18, 0.18]                # 0.175 / 0.0875 rounded up
    assert out["maker"].to_list() == [0.0, 0.03, 0.0, 0.05]                  # 0.021875 / 0.04375 rounded up


def _rows(**kw) -> pl.DataFrame:
    base = {"event_ticker": ["E1", "E2"], "team": ["A", "B"], "ticker": ["T1", "T2"], "side": ["yes", "no"],
            "start": [datetime(2026, 9, 1, 20, tzinfo=UTC)] * 2, "start_date": [date(2026, 9, 1)] * 2,
            "settlement_ts": [datetime(2026, 9, 1, 23, tzinfo=UTC)] * 2, "payoff": [1, 0],
            "px1": [0.52, 0.40], "mid1": [0.515, 0.39], "spr1": [0.01, 0.08],
            "fee_type": ["quadratic_with_maker_fees"] * 2, "fee_multiplier": [1.0, 1.0]}
    base.update(kw)
    return pl.DataFrame(base)


def test_taker_trades_pnl_and_spread_filter():
    tr = bs.taker_trades(_rows(), 1)
    assert tr.height == 1                                                    # the 8c-wide quote is not crossed
    r = tr.row(0, named=True)
    assert r["n"] == 9 and r["t"] == datetime(2026, 9, 1, 19, tzinfo=UTC)
    fee = r["fee_order"]
    assert fee == pytest.approx(0.16)                                        # 0.07 x 9 x .52 x .48 = 0.157 -> 0.16
    assert r["pnl"] == pytest.approx(1 - 0.52 - fee / 9)


def test_maker_fill_needs_a_strict_trade_through_after_posting():
    rows = _rows(px1=[0.52, 0.40], spr1=[0.02, 0.02])                        # bids: YES 0.50 on T1, NO 0.38 on T2
    t = datetime(2026, 9, 1, 19, tzinfo=UTC)
    c = pl.DataFrame({"ticker": ["T1", "T1", "T2", "T2"],
                      "end_ts": [t + timedelta(minutes=30), t + timedelta(hours=1), t + timedelta(hours=1), t + timedelta(minutes=30)],
                      "p_low": [0.40, 0.50, 0.55, 0.55], "p_high": [0.60, 0.55, 0.62, 0.70]})
    tr = bs.maker_trades(rows, 1, candles=c)
    assert tr.is_empty()      # T1: only an at-the-bid print after posting; T2: through-print only in the posting hour
    c2 = c.with_columns(pl.when((pl.col("ticker") == "T2") & (pl.col("end_ts") == t + timedelta(hours=1)))
                        .then(0.63).otherwise(pl.col("p_high")).alias("p_high"))
    tr2 = bs.maker_trades(rows, 1, candles=c2)
    assert tr2["ticker"].to_list() == ["T2"] and tr2["price"][0] == pytest.approx(0.38)


def test_capped_ledger_respects_game_cap():
    t0 = datetime(2026, 9, 1, 18, tzinfo=UTC)
    tr = pl.DataFrame({"t": [t0, t0 + timedelta(minutes=1), t0 + timedelta(minutes=2)], "n": [9.0, 9.0, 9.0],
                       "price": [0.52, 0.52, 0.52], "fee_order": [0.08] * 3, "event_ticker": ["G1", "G1", "G1"],
                       "settlement_ts": [t0 + timedelta(hours=4)] * 3, "ticker": ["T1"] * 3, "dir": [1, 1, 1], "payoff": [1, 1, 1]})
    led = bs.capped_ledger(tr, "s", "v", "dev")
    assert led.height == 2                                                   # $4.76 each: the third breaks $10 per game


# ---- props attention -------------------------------------------------------------------------------
def test_props_attention_is_as_of():
    start = datetime(2026, 9, 20, 17, tzinfo=UTC)
    legs = pl.DataFrame({"ticker": ["KXNFLTD-26SEP20AB-AP1-1", "KXNFLTD-26SEP20AB-BP2-1", "KXNFLTD-26SEP20AB-CP3-1"],
                         "event_ticker": ["KXNFLTD-26SEP20AB"] * 3, "series_ticker": ["KXNFLTD"] * 3,
                         "group": ["sports prop"] * 3, "start": [start] * 3, "y": [1, 0, 0], "holdout": [False] * 3})
    rows = []
    for tk, v_before, v_after in zip(legs["ticker"], [5.0, 50.0, 500.0], [10_000.0, 0.0, 0.0], strict=True):
        rows.append({"ticker": tk, "end_ts": start - timedelta(hours=3), "bid_close": 0.30, "ask_close": 0.32, "volume": v_before})
        rows.append({"ticker": tk, "end_ts": start, "bid_close": 0.30, "ask_close": 0.32, "volume": v_after})
    p = bs.props_panel(legs, pl.DataFrame(rows).lazy(), lead_h=1).sort("ticker")
    assert p["cumvol"].to_list() == [5.0, 50.0, 500.0]                       # the 10k traded after T-1h is unseen
    assert p["att_pct"].to_list() == pytest.approx([1 / 6, 0.5, 5 / 6])
    assert p["r"][0] == pytest.approx(1 - 0.31)


# ---- key numbers ------------------------------------------------------------------------------------
def test_keynum_pairs_adjacent_strikes_both_teams():
    start = datetime(2026, 9, 13, 17, tzinfo=UTC)
    lad = "KXNFLSPREAD-26SEP13AAABBB"
    # AAA won by exactly 3: "AAA by > 2.5" yes, "> 3.5" no; BBB legs all no
    legs = pl.DataFrame({"ticker": [f"{lad}-A3", f"{lad}-A4", f"{lad}-A5", f"{lad}-B3", f"{lad}-B4"], "ladder": [lad] * 5,
                         "series_ticker": ["KXNFLSPREAD"] * 5, "lo": [2.5, 3.5, 4.5, -np.inf, -np.inf],
                         "hi": [np.inf, np.inf, np.inf, -2.5, -3.5], "y": [1, 0, 0, 0, 0]})
    q = pl.DataFrame({"ladder": [lad] * 5, "t": [start - timedelta(hours=2)] * 5, "ticker": legs["ticker"],
                      "bid": [0.40, 0.33, 0.30, 0.25, 0.19], "ask": [0.41, 0.34, 0.31, 0.26, 0.20]})
    games = pl.DataFrame({"event_ticker": ["KXNFLGAME-26SEP13AAABBB"], "start": [start], "start_date": [date(2026, 9, 13)],
                          "league": ["NFL"]})
    k = bs.keynum_rows(1, games=games, legs=legs, quotes=q).sort("team", "m")
    first = k.filter(pl.col("team") == "first")
    assert first["m"].to_list() == [3, 4]
    assert first["implied"].to_list() == pytest.approx([0.07, 0.03])        # .405 - .335, .335 - .305
    assert first["realized"].to_list() == [1.0, 0.0] and first["key"].to_list() == [True, False]
    second = k.filter(pl.col("team") == "second")
    assert second["m"].to_list() == [3] and second["implied"][0] == pytest.approx(0.06) and second["realized"][0] == 0.0
