"""Polymarket-leads-Kalshi study on recorded payloads (trimmed copies of this study's raw responses):

* ``polymarket/xvenue_events.json``: Gamma closed events (Netflix top US movie, Billboard Hot 100 #1 for
  two chart weeks, a Trump rally mention event, the AZ-01 House race);
* ``kalshi/xvenue_markets.json``: the Kalshi markets of the same contracts (``/markets?series_ticker=``);
* ``polymarket/xvenue_ws_stream.jsonl.zst`` + ``xvenue_prices_history_stamp.json``: one in-play token's
  recorded WebSocket book (best-quote changes, 2026-11-12 03:10-03:30 UTC) and its CLOB minute history;
* ``polymarket/xvenue_prices_history_bb.json`` + ``kalshi/xvenue_candles_bb.json``: a Big Brother
  eviction pair's CLOB history and Kalshi minute candles around its first Polymarket-led signal.
"""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import xvenue_leadlag as X
from albert.research.xcat import ohlc_rows

FIX = Path(__file__).parent / "fixtures"
EVENTS = json.loads((FIX / "polymarket" / "xvenue_events.json").read_text())
MARKETS = json.loads((FIX / "kalshi" / "xvenue_markets.json").read_text())
STAMP = json.loads((FIX / "polymarket" / "xvenue_prices_history_stamp.json").read_text())
BB_PM = json.loads((FIX / "polymarket" / "xvenue_prices_history_bb.json").read_text())
BB_K = json.loads((FIX / "kalshi" / "xvenue_candles_bb.json").read_text())


def _pm() -> pl.DataFrame:
    return pl.DataFrame(X.pm_market_rows(EVENTS["events"]), schema=X.PM_MARKET_SCHEMA)


def _kalshi() -> pl.DataFrame:
    info = {"KXNETFLIXRANKMOVIE": {"category": "Entertainment", "fee_multiplier": 1.0},
            "KXTOPSONG": {"category": "Entertainment", "fee_multiplier": 1.0},
            "KXTRUMPMENTION": {"category": "Mentions", "fee_multiplier": 1.0},
            "HOUSEAZ1": {"category": "Elections", "fee_multiplier": 1.0}}
    rows = X.kalshi_market_rows(MARKETS, {}, info, "series")
    return pl.DataFrame(rows, schema=X.KALSHI_MARKET_SCHEMA)


def _fam(name: str) -> dict:
    return next(f for f in X.FAMILIES if f["name"] == name)


def _pairs(name: str) -> dict[str, str]:
    p = _pm()
    by_cid = dict(zip(p["condition_id"], p["question"], strict=True))
    return {m["ticker"]: by_cid[m["condition_id"]] for m in X.match_family(_fam(name), _kalshi(), p)}


def test_parsers_on_recorded_payloads():
    p = _pm()
    assert p.height == 15 and p["condition_id"].n_unique() == 15
    assert set(p["outcomes"].list.get(0)) == {"Yes"} and (p["token_ids"].list.len() == 2).all()
    nf = p.filter(pl.col("group_title") == "Animals").row(0, named=True)
    assert "Tuesday, November 3, 2026" in nf["description"]
    k = _kalshi()
    assert k.height == 15 and set(k["result"]) <= {"yes", "no"}
    ani = k.filter(pl.col("ticker") == "KXNETFLIXRANKMOVIE-26NOV02-ANI").row(0, named=True)
    assert ani["yes_sub_title"] == "Animals" and "chart published on Nov 3, 2026" in ani["rules_primary"]


def test_netflix_and_billboard_pairs_need_the_same_list_and_date():
    nf = _pairs("netflix_rank")
    assert set(nf) == {"KXNETFLIXRANKMOVIE-26NOV02-ANI", "KXNETFLIXRANKMOVIE-26NOV02-GLA",
                       "KXNETFLIXRANKMOVIE-26NOV02-SAC"}     # UNA has no PM outcome here; 'Movie A' never matches
    bb = X.match_family(_fam("billboard_hot100_1"), _kalshi(), _pm())
    slugs = dict(zip(_pm()["condition_id"], _pm()["event_slug"], strict=True))
    assert {slugs[m["condition_id"]] for m in bb} == {"billboard-hot-100-1-song-week-of-november-7"}
    assert {m["ticker"] for m in bb} == {"KXTOPSONG-26NOV07-PAT", "KXTOPSONG-26NOV07-CHO", "KXTOPSONG-26NOV07-BOS",
                                         "KXTOPSONG-26NOV07-HOU"}   # 'HOUSEWIFE' = 'Housewife - Tate McRae'
    assert all(m["name_score"] == 1.0 for m in bb)
    # the Nov 14 chart lists the same songs: a date mismatch, never a pair; 'stupid song' never matches 'Song A'
    assert X.name_match("stupid song", "Song A") == 0.0


def test_house_party_and_mentions_pairs():
    hs = X.match_family(_fam("house_party"), _kalshi(), _pm())
    p = _pm()
    q = dict(zip(p["condition_id"], p["question"], strict=True))
    got = {m["ticker"]: q[m["condition_id"]] for m in hs}
    assert got == {"HOUSEAZ1-26-R": "Will the Republican Party win the AZ-01 House seat?",
                   "HOUSEAZ1-26-D": "Will the Democratic Party win the AZ-01 House seat?"}
    mn = X.match_family(_fam("mentions"), _kalshi(), p)
    assert [m["ticker"] for m in mn] == ["KXTRUMPMENTION-26NOV01-AI"]
    k = _kalshi().filter(pl.col("ticker") == "KXTRUMPMENTION-26NOV01-AI").row(0, named=True)
    pm = p.filter(pl.col("condition_id") == mn[0]["condition_id"]).row(0, named=True)
    assert X.same_speech(k, pm)
    assert X.mention_terms(k["yes_sub_title"]) == X.mention_terms(pm["question"]) == frozenset({"ai", "artificial intelligence"})
    # a different speech (Kalshi title rewritten to another city) is rejected on the event words
    assert not X.same_speech({**k, "title": k["title"].replace("Iowa", "Ohio").replace("Dubuque", "Toledo")}, pm)


def test_clob_point_describes_the_book_about_14_s_earlier():
    q = X.pm_stream_quotes(FIX / "polymarket" / "xvenue_ws_stream.jsonl.zst", assets={STAMP["token"]})
    assert q.height > 1000
    h = pl.DataFrame(X.pm_history_rows(STAMP, STAMP["token"]))
    res = X.stamp_offsets(q, h, tuple(range(-40, 21, 2)))
    best = res.sort("mae_c").row(0, named=True)
    assert -20 <= best["offset_s"] <= -10 and best["match_mid"] > 0.85
    # using the point at its own stamp is conservative: worse fit than the true (earlier) instant
    at0 = res.filter(pl.col("offset_s") == 0).row(0, named=True)
    assert at0["mae_c"] > best["mae_c"]


def _bb_pair() -> dict:
    return {"pair_id": "KXBIGBROTHERELIMINATION-26AUG27-YAS|0x1d47e24f", "ticker": BB_K["ticker"],
            "event_ticker": "KXBIGBROTHERELIMINATION-26AUG27", "pm_event_slug": "who-will-be-evicted-from-big-brother-week-7",
            "family": "bigbrother_evict", "group": "culture", "exact": "near", "k_result": "no", "fee_multiplier": 1.0,
            "t0": datetime(2026, 8, 21, 21, 28, 57, tzinfo=UTC), "t1": datetime(2026, 8, 22, 1, 0, tzinfo=UTC),
            "k_end": datetime(2026, 8, 22, 2, 0, tzinfo=UTC), "k_settle": datetime(2026, 8, 28, 14, 34, tzinfo=UTC)}


def _bb_grid():
    pm = pl.DataFrame(X.pm_history_rows(BB_PM, BB_PM["token"])).sort("t")
    k = pl.DataFrame(ohlc_rows(BB_K, BB_K["ticker"], 1)).sort("end_ts")
    g = X.grid(_bb_pair(), pm["t"].to_numpy(), pm["p"].to_numpy(), k["end_ts"].dt.epoch("s").to_numpy(),
               k["bid_close"].to_numpy(), k["ask_close"].to_numpy())
    return g, pm, k


def test_grid_uses_only_past_points():
    g, pm, k = _bb_grid()
    t, p = pm["t"].to_numpy(), pm["p"].to_numpy()
    for i in range(0, len(g["T"]), 7):
        T = g["T"][i]
        past = t[t <= T]
        if len(past) and T - past[-1] <= X.PM_STALE_S:
            assert g["pm"][i] == p[t <= T][-1]               # last point stamped at or before T_i
        ends = k["end_ts"].dt.epoch("s").to_numpy()
        j = np.flatnonzero(ends <= T)
        if len(j) and math.isfinite(g["ask"][i]):
            assert g["ask"][i] == k["ask_close"][int(j[-1])]  # forward-filled close, never a later candle
    assert not g["live"][g["T"] > int(_bb_pair()["t1"].timestamp())].any()


def test_signal_trades_next_minute_at_the_ask_with_rounded_fee():
    g, _, _ = _bb_grid()
    ev = X.events_for(g, 0.05, 15, "pm")
    assert ev, "the recorded window holds a >= 5c Polymarket move"
    i, d = ev[0]
    assert abs(g["pm"][i] - g["pm"][i - 15]) >= 0.05 - 1e-9
    assert np.sign(g["pm"][i] - g["pm"][i - 15]) * (g["mid"][i] - g["mid"][i - 15]) < 0.025
    tr = X.trades_for(_bb_pair(), g, ev[:1], lag=1)
    assert len(tr) == 1
    t = tr[0]
    e = i + 1
    assert t["t_in"] == g["T"][e] == t["t_sig"] + 60
    price = g["ask"][e] if d > 0 else 1 - g["bid"][e]
    assert t["price"] == price
    n = max(1.0, math.floor(X.STAKE / price))
    assert t["n"] == n and t["fee_in"] == math.ceil(round(0.07 * n * price * (1 - price) * 100, 9)) / 100
    x = e + 15
    exit_ = g["bid"][x] if d > 0 else 1 - g["ask"][x]
    fee_out = X.taker_fee(n, exit_, 1.0)
    assert math.isclose(t["pnl_15m"], exit_ - price - (t["fee_in"] + fee_out) / n)
    # beyond the fetched Kalshi window the position is valued at settlement (no exit fee)
    y = 0.0 if d > 0 else 1.0
    assert math.isclose(t["pnl_settle"], y - price - t["fee_in"] / n)
    assert t["t_out_settle"] == int(_bb_pair()["k_settle"].timestamp())


def test_kalshi_jobs_respect_api_caps():
    pairs = pl.DataFrame([{**_bb_pair(), "t0": datetime(2026, 9, 20, tzinfo=UTC), "t1": datetime(2026, 10, 30, tzinfo=UTC),
                           "k_close": datetime(2026, 10, 30, 7, tzinfo=UTC),
                           "k_end": datetime(2026, 10, 30, 7, tzinfo=UTC), "k_settle": datetime(2026, 10, 31, tzinfo=UTC),
                           "ticker": f"KXTEST-26OCT30-{c}", "event_ticker": "KXTEST-26OCT30"} for c in "ABC"])
    jobs = X.kalshi_jobs(pairs, datetime(2026, 9, 18, tzinfo=UTC))
    assert {j["kind"] for j in jobs} == {"batch"}
    for j in jobs:
        minutes = (j["hi"] - j["lo"]).total_seconds() / 60 + 1
        assert minutes <= X.KALSHI_MAX_PERIODS and minutes * len(j["tickers"]) <= X.KALSHI_MAX_BATCH
    covered = {t: sum((j["hi"] - j["lo"]).total_seconds() / 60 + 1 for j in jobs if t in j["tickers"]) for t in pairs["ticker"]}
    want = (datetime(2026, 10, 30, 7, tzinfo=UTC) - (datetime(2026, 9, 20, tzinfo=UTC) - X.WARMUP)).total_seconds() / 60
    assert all(v >= want for v in covered.values())


def test_fee_rounding_and_clustered_t():
    assert X.taker_fee(10, 0.5, 1.0) == 0.18 and X.taker_fee(1, 0.5, 1.0) == 0.02 and X.taker_fee(100, 0.5, 1.0) == 1.75
    df = pl.DataFrame({"v": [1.0, 1.0, -1.0, 3.0], "pm_event": ["a", "a", "b", "c"], "date": ["d1", "d2", "d1", "d3"]})
    c = X.cluster_t(df, "v")
    assert c["n"] == 4 and math.isclose(c["mean"], 1.0) and c["events"] == 3 and c["dates"] == 3 and c["t"] > 0
