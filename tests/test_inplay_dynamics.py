"""In-play dynamics helpers on a recorded in-play MLB trades page (trimmed copy of a raw page)."""

import itertools
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import zstandard

from albert.research import inplay_dynamics as d
from albert.venues import kalshi as k
from albert.venues.kalshi import ticker_start

FIX = Path(__file__).parent / "fixtures" / "kalshi"
TICKER = "KXMLBGAME-26AUG211910WSHMIA-MIA"


def _trades() -> pl.DataFrame:
    payload = json.loads((FIX / "trades_inplay_mlb.json").read_text())
    return pl.DataFrame(k.trade_rows(payload), schema=k.TRADE_SCHEMA)


def _path() -> tuple[d.Path_, pl.DataFrame]:
    pr = d.prints_frame(_trades())
    t = pr["ts"].dt.epoch("us").to_numpy() / 1e6
    return d.Path_(t, pr["side"].to_numpy(), pr["pmin"].to_numpy(), pr["pmax"].to_numpy()), pr


def test_prints_frame_groups_one_sweep_into_one_taker_group():
    raw = _trades()
    pr = d.prints_frame(raw)
    assert raw.height == 489 and 0 < pr.height < raw.height
    assert pr["ts"].is_sorted()
    sweep = pr.filter(pl.col("ts") == datetime(2026, 8, 22, 1, 37, 40, 577114, tzinfo=UTC))
    assert sweep.height == 1
    r = sweep.row(0, named=True)
    # one YES taker bought 1,195.91 contracts at 0.73 .. 0.83 in 38 prints at the same microsecond
    assert (r["side"], r["pmin"], r["pmax"], r["k"]) == (1, 0.73, 0.83, 38)
    assert r["n"] == pytest.approx(1195.91)


def test_mid_proxy_uses_only_past_prints_and_the_post_sweep_book():
    path, pr = _path()
    assert np.isfinite(path.mid).sum() > 50
    for i in (10, 60, 120, len(path.t) - 1):
        past = pr.head(i + 1)
        ask = past.filter((pl.col("side") == 1) & (pl.col("ts").dt.epoch("us") / 1e6 >= path.t[i] - d.MID_MAX_AGE_S))
        bid = past.filter((pl.col("side") == -1) & (pl.col("ts").dt.epoch("us") / 1e6 >= path.t[i] - d.MID_MAX_AGE_S))
        if ask.height and bid.height:
            want = 0.5 * (ask["pmax"][-1] + bid["pmin"][-1])
            assert path.mid[i] == pytest.approx(want)
        else:
            assert math.isnan(path.mid[i])


def test_scheduled_start_matches_the_ticker_and_ignores_rewritten_occurrence():
    start = ticker_start("KXMLBGAME-26AUG211910WSHMIA")
    assert start == datetime(2026, 8, 21, 23, 10, tzinfo=UTC)
    occ = start + timedelta(hours=3)
    assert d.scheduled_start(occ, occ) == start
    # NFL style: expected expiration = occurrence + 3 h
    assert d.scheduled_start(occ, occ + timedelta(hours=3)) == start
    # occurrence rewritten to the actual end (non-round seconds): fall back to the expiration
    rewritten = occ + timedelta(hours=20, minutes=14, seconds=49)
    assert d.scheduled_start(rewritten, occ) == start
    market = json.loads((FIX / "settled_markets.json").read_text())["markets"][0]
    o, e = k.parse_ts(market.get("occurrence_datetime")), k.parse_ts(market.get("expected_expiration_time"))
    assert d.scheduled_start(o, e) == min(o, e) - timedelta(hours=3)


def test_taker_round_trip_never_uses_the_trigger_print_and_pays_fees():
    path, _ = _path()
    close = path.t[-1] + 3600
    i = int(np.flatnonzero(path.side > 0)[5])          # a YES-taker group as the trigger
    rt = d.taker_round_trip(path, True, path.t[i], (60.0, math.inf), y=1.0, mult=0.5, close=close)
    assert rt is not None
    j = path.next_print(1, path.t[i] + d.ENTRY_DELAY_S)
    assert j > i and path.t[j] > path.t[i] + d.ENTRY_DELAY_S
    assert rt["t_entry"] == path.t[j] and rt["entry"] == path.best[j] == path.pmin[j]
    assert rt["fee_in"] == pytest.approx(0.07 * 0.5 * rt["entry"] * (1 - rt["entry"]))
    x = path.next_print(-1, rt["t_entry"] + 60.0, strict=False)
    if x >= 0:
        val = path.best[x]
        want = val - rt["entry"] - rt["fee_in"] - 0.07 * 0.5 * val * (1 - val)
    else:
        want = 1.0 - rt["entry"] - rt["fee_in"]
    assert rt["pnl_60"] == pytest.approx(want)
    assert rt["pnl_settle"] == pytest.approx(1.0 - rt["entry"] - rt["fee_in"])
    # buying NO pays 1 - (the highest YES price of the next NO-taker group)
    rn = d.taker_round_trip(path, False, path.t[i], (math.inf,), y=1.0, mult=0.5, close=close)
    jn = path.next_print(-1, path.t[i] + d.ENTRY_DELAY_S)
    assert rn["entry"] == pytest.approx(1 - path.pmax[jn])
    assert rn["pnl_settle"] == pytest.approx(-rn["entry"] - rn["fee_in"])


def test_exit_after_the_last_print_settles():
    path, _ = _path()
    rt = d.taker_round_trip(path, True, path.t[0], (3600.0,), y=0.0, mult=1.0, close=path.t[-1] + 7200)
    assert rt["pnl_3600"] == pytest.approx(0.0 - rt["entry"] - rt["fee_in"])


def test_maker_fills_only_when_a_later_print_trades_through():
    path, _ = _path()
    close = path.t[-1] + 3600
    t0 = path.t[0]
    seen = path.pmin[(path.t > t0 + d.ENTRY_DELAY_S)].min()
    # a bid exactly at the lowest later price is touched, never traded through: no fill
    assert d.maker_round_trip(path, True, float(seen), t0, 1e6, (math.inf,), 1.0, 1.0, True, close) is None
    lvl = float(seen) + 0.01
    rt = d.maker_round_trip(path, True, lvl, t0, 1e6, (60.0, math.inf), 1.0, 1.0, True, close)
    assert rt is not None and rt["entry"] == lvl and rt["t_fill"] > t0 + d.ENTRY_DELAY_S
    f = int(np.searchsorted(path.t, rt["t_fill"]))
    assert path.pmin[f] < lvl and not (path.pmin[(path.t > t0 + d.ENTRY_DELAY_S) & (path.t < rt["t_fill"])] < lvl).any()
    assert rt["fee_in"] == pytest.approx(0.0175 * lvl * (1 - lvl))
    assert rt["pnl_settle"] == pytest.approx(1.0 - lvl - rt["fee_in"])


def test_jump_events_are_causal_and_respect_the_refractory_period():
    path, pr = _path()
    ev = d.jump_events(path, 0.05, path.t[0], path.t[-1] + 1)
    assert ev, "the fixture holds two runs of >= 5c moves"
    times = [path.t[i] for i, _, _ in ev]
    assert all(b - a >= 60.0 for a, b in itertools.pairwise(times))
    for i, ref, sgn in ev:
        assert abs(path.mid[i] - ref) >= 0.05 - 1e-9 and sgn == (1 if path.mid[i] > ref else -1)
        # the same event is found when every later print is removed (no look-ahead)
        cut = pr.head(i + 1)
        tc = cut["ts"].dt.epoch("us").to_numpy() / 1e6
        pc = d.Path_(tc, cut["side"].to_numpy(), cut["pmin"].to_numpy(), cut["pmax"].to_numpy())
        again = d.jump_events(pc, 0.05, path.t[0], path.t[-1] + 1)
        assert again and again[-1][0] == i and again[-1][1] == pytest.approx(ref)


def test_analyze_game_produces_grid_events_and_trades():
    _, pr = _path()
    sched = ticker_start("KXMLBGAME-26AUG211910WSHMIA")
    meta = {"ticker": TICKER, "sport": "MLB", "split": "dev", "sched": sched, "y": 1.0, "mult": 0.5,
            "maker_fee": True, "L": 177.0, "on_schedule": True,
            "close_time": pr["ts"].max() + timedelta(minutes=30)}
    out = d.analyze_game(meta, pr)
    assert out["grid"] and out["game"] and out["events"] and out["maker"]
    g = pl.DataFrame(out["grid"], infer_schema_length=None)
    assert g["e"].min() == -d.PRE_MIN and g.filter(pl.col("mid").is_not_null()).height > 5
    ev = pl.DataFrame(out["events"], infer_schema_length=None)
    assert set(ev["window"]) == {"inplay"} and set(ev["dir"]) == {"follow", "fade"}
    assert (ev["phase"] == "late").all()   # 138-148 min after a 23:10 UTC start: >= 2/3 of L = 177
    filled = ev.filter(pl.col("filled"))
    assert filled.height and (filled["pnl_settle"] <= 1).all()
    tbl = d.taker_table(ev)
    assert tbl.filter(pl.col("split") == "dev")["trades"].sum() == filled.height


def test_clustered_mean_uses_games_not_rows():
    f = pl.DataFrame({"ticker": ["a", "a", "b", "c"], "x": [0.1, 0.1, -0.05, 0.2]})
    r = d.cl(f, "x")
    assert r["games"] == 3 and r["n"] == 4 and r["mean"] == pytest.approx(0.0875) and r["t"] is not None


def test_stream_trades_parses_recorded_websocket_trades(tmp_path):
    day = tmp_path / "kalshi" / "2026-11-05"
    day.mkdir(parents=True)
    raw = (FIX / "ws_stream_trades.jsonl").read_bytes()
    (day / "04-k1.jsonl.zst").write_bytes(zstandard.ZstdCompressor().compress(raw))
    f = d.stream_trades(tmp_path, "2026-11-05T04", "2026-11-05T05", series=("KXRAIN",))
    n = sum(1 for line in raw.decode().splitlines() if '\\"type\\":\\"trade\\"' in line)
    assert n > 0 and 0 < f.height <= n
    assert set(f["taker_side"]) <= {"yes", "no"} and f["yes_price"].is_between(0.01, 0.99).all()
    assert f["ticker"].n_unique() == 1 or f["ticker"].str.replace(r"-[^-]+$", "").n_unique() == f["ticker"].n_unique()
