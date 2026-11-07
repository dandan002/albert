import json
from pathlib import Path

import polars as pl

from albert.backtest.l2 import _mark_executions, events_from_rows
from albert.research.volatile import flag, from_events

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _events(name: str):
    rows = [json.loads(line) for line in (FIX / name).read_text().splitlines()]
    evs = events_from_rows(rows)
    _mark_executions(evs)
    return evs


def test_buckets_from_recorded_stream():
    evs = _events("ws_stream_wide.jsonl")
    df = from_events(evs, bucket_s=60)
    assert df.height > 0
    assert df["updates"].sum() > 0
    assert (df["path_all_c"] >= df["path_c"]).all()
    assert df.filter(pl.col("range_c").is_not_null())["range_c"].min() >= 0
    assert not df["in_play"].any()          # no schedule given


def test_in_play_from_schedule_and_prev_flag_has_no_look_ahead():
    evs = _events("ws_stream_trades.jsonl")
    t = evs[len(evs) // 2].recv_ns
    from datetime import UTC, datetime, timedelta
    mkt = "KXNBAGAME-26NOV05AAABBB-AAA"
    exp = datetime.fromtimestamp(t / 1e9, UTC) + timedelta(hours=1)
    for e in evs:
        e.market = mkt
        e.msg = {**e.msg, "market_ticker": mkt}
    df = from_events(evs, bucket_s=60, meta={mkt: {"expected_expiration_time": exp.isoformat()}})
    assert df["in_play"].any()
    df = df.sort("bucket_ns")
    assert not df["volatile_prev"][0]
    assert df["volatile_prev"].to_list()[1:] == df["volatile"].to_list()[:-1]


def test_flag_cut_ignores_flat_buckets():
    df = pl.DataFrame({"market": ["a"] * 5, "bucket_ns": [0, 1, 2, 3, 4],
                       "path_c": [0.0, 0.0, 1.0, 2.0, 10.0], "in_play": [False] * 5})
    out = flag(df, q=0.5).sort("bucket_ns")
    assert out["volatile"].to_list() == [False, False, False, True, True]
