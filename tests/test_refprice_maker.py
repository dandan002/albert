from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from albert.research import refprice_maker as R
from albert.venues import kalshi as k
from albert.venues.polymarket import price_history_rows

KICKOFF = datetime(2026, 9, 3, 17, 10, tzinfo=UTC)          # Polymarket gameStartTime, TOR @ CLE
TOK0 = "76707183707928870842544799008737154929784240389156762187233897121006449222052"
# Polymarket keeps 5-minute history ~60 days, so the path tests use an October game:
VANNJ_KICKOFF = datetime(2026, 10, 10, 19, 30, tzinfo=UTC)   # nhl-van-nj-2026-10-10
VANNJ_TOK0 = "97136069805191878553875524650799385477169935978501248631122016636089002512690"


def _pm_games(fixture) -> pl.DataFrame:
    """Real Gamma page (trimmed to the TOR @ CLE event: moneyline + one prop market)."""
    return pl.DataFrame(R.pm_us_game_rows(fixture("polymarket/refprice_gamma_mlb_torcle.json")["events"]))


def _kalshi(fixture) -> pl.DataFrame:
    rows = k.market_rows(fixture("kalshi/mlb_markets_torcle_2026-09-03.json"), {}, "test")
    return pl.DataFrame(rows, schema=k.MARKET_SCHEMA)


def _history(fixture) -> pl.DataFrame:
    h = fixture("polymarket/refprice_prices_history_vannj.json")
    return pl.DataFrame(price_history_rows(h["response"], h["token_id"]))


def test_pm_us_game_rows_reads_the_moneyline(fixture):
    g = _pm_games(fixture)
    assert g.height == 1
    r = g.row(0, named=True)
    assert (r["pm_slug"], r["league"], r["pm_code0"], r["pm_code1"]) == ("mlb-tor-cle-2026-09-03", "mlb", "tor", "cle")
    assert (r["pm_out0"], r["pm_out1"]) == ("Toronto Blue Jays", "Cleveland Guardians")
    assert r["pm_tok0"] == TOK0
    assert r["pm_start"] == KICKOFF


def test_kalshi_us_games_and_match(fixture):
    kg = R.kalshi_us_games(_kalshi(fixture).filter(pl.col("series_ticker") == "KXMLBGAME"))
    assert kg.height == 1
    g = kg.row(0, named=True)
    assert g["kdate"].isoformat() == "2026-09-03" and {g["k_code0"], g["k_code1"]} == {"tor", "cle"}
    # this trimmed fixture has no occurrence time: both team codes agreeing is enough
    assert g["occurrence_time"] is None
    m = R.match_us(kg, _pm_games(fixture))
    assert m.height == 2
    legs = dict(zip(m["ticker"], m["pm_outcome"]))
    assert legs == {"KXMLBGAME-26SEP031310TORCLE-TOR": 0, "KXMLBGAME-26SEP031310TORCLE-CLE": 1}
    assert set(m["kickoff"]) == {KICKOFF}
    # Toronto won: the TOR leg settles YES
    assert dict(zip(m["ticker"], m["y"]))["KXMLBGAME-26SEP031310TORCLE-TOR"] == 1


def test_leg_score_name_fallbacks():
    assert R.leg_score("vgk", "Vegas", "las", ("Golden Knights", "Vegas Golden Knights")) >= 0.75
    assert R.leg_score("kenn", "Kennesaw St.", "kensaw", ("Kennesaw State", "Owls")) >= 0.75
    assert R.leg_score("tor", "Toronto", "tor", ("x",)) == 1.0
    assert R.leg_score("bos", "Boston", "nyy", ("New York Yankees",)) < 0.75


def test_fair_path_uses_only_ended_buckets_and_complements(fixture):
    h = _history(fixture)
    assert h.height > 200
    # the bucket that starts at kickoff is the cleared book (price 0); it must never be used
    assert h.filter(pl.col("ts") == VANNJ_KICKOFF)["price"].to_list() == [0.0]
    legs = pl.DataFrame({"event": ["E", "E"], "ticker": ["VAN", "NJ"], "tokens": [[VANNJ_TOK0], [VANNJ_TOK0]], "leg_index": [0, 1]})
    paths = R.fair_paths(legs, h.filter((pl.col("price") > 0) & (pl.col("price") < 1)))
    t = (VANNJ_KICKOFF - timedelta(hours=6)).replace(minute=0)
    p_van, p_nj = R.fair_at(paths["VAN"], t.timestamp()), R.fair_at(paths["NJ"], t.timestamp())
    assert p_van + p_nj == pytest.approx(1.0)
    # the value at t is the bucket that ENDED by t, i.e. the one that started 5 minutes earlier
    want = h.filter(pl.col("ts") == t - timedelta(seconds=R.PM_BUCKET))["price"].item()
    assert p_van == pytest.approx(want)
    # at kickoff the last usable bucket is the one before the cleared book
    assert R.fair_at(paths["VAN"], VANNJ_KICKOFF.timestamp()) > 0.05
    assert R.fair_at(paths["VAN"], (VANNJ_KICKOFF - timedelta(hours=40)).timestamp()) is None


def test_pull_time_first_move_of_k(fixture):
    h = _history(fixture).filter((pl.col("price") > 0) & (pl.col("price") < 1))
    legs = pl.DataFrame({"event": ["E"], "ticker": ["VAN"], "tokens": [[VANNJ_TOK0]], "leg_index": [0]})
    path = R.fair_paths(legs, h)["VAN"]
    t0, t1 = (VANNJ_KICKOFF - timedelta(hours=24)).replace(minute=0).timestamp(), VANNJ_KICKOFF.timestamp()
    p0 = R.fair_at(path, t0)
    assert R.pull_time(path, t0, t1, p0, None) == t1
    for kk in (0.01, 0.02):
        tau = R.pull_time(path, t0, t1, p0, kk)
        ts, fv = path
        seen = fv[(ts > t0) & (ts <= tau)]
        if tau < t1:
            assert abs(seen[-1] - p0) >= kk - 1e-9 and (np.abs(seen[:-1] - p0) < kk - 1e-9).all()
        else:
            assert (np.abs(seen - p0) < kk - 1e-9).all()
    # a bigger threshold never pulls earlier; this game's fair did move 1c within the day
    assert R.pull_time(path, t0, t1, p0, 0.01) < t1
    assert R.pull_time(path, t0, t1, p0, 0.05) >= R.pull_time(path, t0, t1, p0, 0.01)


def test_first_through_on_real_prints(fixture):
    tr = pl.DataFrame(k.trade_rows(fixture("kalshi/refprice_trades_vannj_nj.json")))
    assert tr.height == 80
    arr = R.trade_arrays(tr)["KXNHLGAME-26OCT10VANNJ-NJ"]
    lo, hi = arr[0][0] - 1, arr[0][-1]
    sells = tr.filter(pl.col("taker_side") == "no").sort("created_time")
    px = float(sells["yes_price"].max())
    # strictly through: the first YES-seller print below our bid
    want = sells.filter(pl.col("yes_price") < px - 1e-9)["created_time"].min()
    got = R.first_through(*arr, px, "yes", lo, hi)
    assert got == pytest.approx(want.timestamp(), abs=1e-3)
    # lenient also counts the print AT our price, so it is never later
    assert R.first_through(*arr, px, "yes", lo, hi, lenient=True) <= got
    # a bid below every print never fills
    assert R.first_through(*arr, 0.01, "yes", lo, hi) == float("inf")


def test_bid_prices_post_only_cap():
    assert R._bid_prices(0.555, 0.02, 0.50, 0.52) == (0.51, 0.42)   # YES capped one tick under the ask
    assert R._bid_prices(0.40, 0.0, 0.39, 0.41) == (0.40, 0.60)
    yb, nb = R._bid_prices(0.40, 0.01, 0.30, 0.45)
    assert (yb, nb) == (0.39, 0.59)


def test_devig_power_matches_soccer():
    from albert.research.soccer import devig

    q = np.array([[0.52, 0.27, 0.25], [0.18, 0.24, 0.62]])
    assert np.allclose(R.devig_power(q), devig(q, "power"), atol=1e-6)
    assert np.allclose(R.devig_power(q).sum(1), 1.0)


def test_requote_times_reset_reference():
    ts = np.array([100.0, 200, 300, 400, 500])
    fv = np.array([0.50, 0.505, 0.52, 0.525, 0.50])
    assert R.requote_times((ts, fv), 150, 500, 0.02) == [150, 300, 500]
    assert R.pull_time((ts, fv), 150, 500, 0.50, 0.02) == 300


def test_simulate_one_row_real_prints_and_path(fixture):
    """One NJ-leg row: real prints and the real Polymarket path; fills obey the pull."""
    h = _history(fixture).filter((pl.col("price") > 0) & (pl.col("price") < 1))
    legs = pl.DataFrame({"event": ["E"], "ticker": ["KXNHLGAME-26OCT10VANNJ-NJ"], "tokens": [[VANNJ_TOK0]], "leg_index": [1]})
    paths = R.fair_paths(legs, h)
    tr = R.trade_arrays(pl.DataFrame(k.trade_rows(fixture("kalshi/refprice_trades_vannj_nj.json"))))
    arr = tr["KXNHLGAME-26OCT10VANNJ-NJ"]
    t = datetime.fromtimestamp(arr[0][0] - 3600, UTC).replace(minute=0, second=0, microsecond=0)
    end = datetime.fromtimestamp(arr[0][-1], UTC)
    p = R.fair_at(paths["KXNHLGAME-26OCT10VANNJ-NJ"], t.timestamp())
    row = pl.DataFrame([{"row": 0, "ticker": "KXNHLGAME-26OCT10VANNJ-NJ", "t": t, "end": end, "p_ref": p, "p_end": p,
                         "mid": p, "bid": round(p - 0.015, 2), "ask": round(p + 0.015, 2), "y": 1,
                         "fee_type": "quadratic_with_maker_fees", "fee_multiplier": 1.0}])
    o = R.simulate(row, tr, paths, margins=(0.0,), pulls=(None, 0.01))
    assert o.height == 4                      # (ref, mid) × (yes, no)
    for r in o.iter_rows(named=True):
        want = R.first_through(*arr, r["price"], r["side"], t.timestamp(), end.timestamp())
        assert r["hit"] == want
        assert r["tau_1c"] <= r["tau_none"] == end.timestamp()
    lo = R.long_orders(o, row.with_columns(pl.lit("E").alias("event"), pl.lit("long").alias("lead"), pl.lit("nhl").alias("group"),
                                           pl.lit("0.6-0.8").alias("band"), pl.lit(0.0).alias("gap")),
                       datetime(2026, 10, 1, tzinfo=UTC), pulls=(None, 0.01))
    # a pull can only remove fills
    assert lo.filter(pl.col("pull") == "1c")["filled"].sum() <= lo.filter(pl.col("pull") == "none")["filled"].sum()
    assert lo.filter(pl.col("pull") == "none")["filled"].sum() >= 1


def test_minute_candles_become_side_agnostic_prints(fixture):
    """Real 1-minute batch candles: only minutes with prints are kept; a YES bid fills at the
    first minute whose low trades strictly below it, a NO bid at the first minute whose high
    trades strictly above 1 − bid."""
    page = fixture("kalshi/refprice_minute_candles.json")
    m = page["markets"][0]
    rows = R.minute_rows(m, m["market_ticker"])
    n_traded = sum(1 for c in m["candlesticks"] if (c.get("price") or {}).get("low_dollars"))
    assert len(rows) == n_traded >= 3
    mins = pl.DataFrame(rows, schema=R.MINUTE_SCHEMA)
    assert (mins["price_low"] <= mins["price_high"]).all()
    arr = R.trade_arrays(R.minute_prints(mins))[m["market_ticker"]]
    lo_t, hi_t = arr[0].min() - 60, arr[0].max()
    b = float(mins["price_low"].max())
    want = mins.filter(pl.col("price_low") < b - 1e-9)["end_ts"].min()
    got = R.first_through(*arr, b, "yes", lo_t, hi_t)
    assert (want is None and got == float("inf")) or got == pytest.approx(want.timestamp())
    nb = round(1 - float(mins["price_high"].min()), 2)
    want = mins.filter(pl.col("price_high") > 1 - nb + 1e-9)["end_ts"].min()
    got = R.first_through(*arr, nb, "no", lo_t, hi_t)
    assert (want is None and got == float("inf")) or got == pytest.approx(want.timestamp())


def test_side_agnostic_counts_both_takers(fixture):
    tr = pl.DataFrame(k.trade_rows(fixture("kalshi/refprice_trades_vannj_nj.json")))
    both = R.side_agnostic(tr)
    assert both.height == 2 * tr.height
    a1 = R.trade_arrays(tr)["KXNHLGAME-26OCT10VANNJ-NJ"]
    a2 = R.trade_arrays(both)["KXNHLGAME-26OCT10VANNJ-NJ"]
    lo, hi = a1[0][0] - 1, a1[0][-1]
    for px in (0.70, 0.72, 0.74):
        # the side-agnostic rule never fills later than the taker-side rule
        assert R.first_through(*a2, px, "yes", lo, hi) <= R.first_through(*a1, px, "yes", lo, hi)
