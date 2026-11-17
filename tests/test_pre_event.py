"""Pre-event NO bids: milestone and candle parsing on recorded payloads; classification, batch
caps, as-of quotes and the early-start exclusion, order prices, trade-through fills with the
circuit breaker, per-order fee rounding and the candidate screen on synthetic frames."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from albert.research import pre_event as P
from albert.research import xcat

FIX = Path(__file__).parent / "fixtures" / "kalshi"
T0 = datetime(2026, 10, 5, 12, tzinfo=UTC)


def _j(name: str) -> dict:
    return json.loads((FIX / name).read_text())


# ---- recorded payloads -----------------------------------------------------------------------------
def test_milestone_rows_recorded():
    page = _j("pre_event_milestones.json")
    rows = P.milestone_rows(page)
    assert rows, "zero milestone rows"
    ms = page["milestones"]
    want = {e for m in ms for e in (m.get("primary_event_tickers") or []) + (m.get("related_event_tickers") or [])}
    assert {r["event"] for r in rows} == want
    r = next(x for x in rows if x["event"] == ms[0]["primary_event_tickers"][0])
    assert r["start"] == datetime.fromisoformat(ms[0]["start_date"])
    assert r["ms_type"] == ms[0]["type"] and r["ms_category"] == "Sports" and r["league"]
    df = pl.DataFrame(rows, schema=P.MS_SCHEMA)
    st = P.event_starts(pl.concat([df, df.with_columns((pl.col("start") + timedelta(hours=1)), pl.lit("x").alias("ms_id"))]))
    assert st.height == df["event"].n_unique()
    assert st.join(df.group_by("event").agg(pl.col("start").min().alias("m")), on="event").filter(pl.col("start") != pl.col("m")).height == 0


def test_batch_candles_recorded():
    page = _j("pre_event_candles_batch.json")
    got = xcat.batch_rows(page, 60)
    assert got and all(rows for rows in got.values())
    rows = [r for rs in got.values() for r in rs]
    assert all(r["end_ts"].tzinfo is not None for r in rows)
    q = [r for r in rows if r["bid_close"] is not None and r["ask_close"] is not None]
    assert q and all(0 <= r["bid_close"] <= r["ask_close"] <= 1 for r in q)
    tr = [r for r in rows if r["p_high"] is not None]
    assert tr and all(r["p_low"] <= r["p_high"] for r in tr)


# ---- classification and batching -------------------------------------------------------------------
@pytest.mark.parametrize("series,kind", [
    ("KXMLBHIT", "prop"), ("KXNFLRECYDSPROT", "prop"), ("KXNHLGOAL", "prop"), ("KXNBAPTS", "prop"),
    ("KXEPLGOAL", "prop"), ("KXMLBGAME", "winner"), ("KXCS2MAP", "winner"), ("KXNFLSPREAD", "line"),
    ("KXNBATEAMTOTAL", "line"), ("KXEPLSCORE", "line"), ("KXMLBF5", "line")])
def test_sports_kind(series, kind):
    assert P.sports_kind(series) == kind


def test_batch_jobs_respect_caps():
    rows = [(f"T{i}", T0 + timedelta(minutes=i), T0 + timedelta(hours=6, minutes=i)) for i in range(120)]
    jobs = P._batch_jobs(rows, period_min=1, max_span=timedelta(hours=8))
    assert sum(len(j["tickers"]) for j in jobs) == 120
    for j in jobs:
        per = (j["hi"] - j["lo"]).total_seconds() / 60 + 1
        assert per * len(j["tickers"]) <= P.MAX_BATCH_CANDLES and len(j["tickers"]) <= P.MAX_BATCH_TICKERS
    far = [("A", T0, T0 + timedelta(hours=1)), ("B", T0 + timedelta(hours=30), T0 + timedelta(hours=31))]
    assert len(P._batch_jobs(far, period_min=60, max_span=timedelta(hours=24))) == 2


# ---- decision rows ------------------------------------------------------------------------------------
def _legs(n: int = 4, start: datetime = T0 + timedelta(hours=6)) -> pl.DataFrame:
    return pl.DataFrame({
        "ticker": [f"E-{i}" for i in range(n)], "event_ticker": ["E"] * n, "start": [start] * n,
        "t": [start - P.LEAD] * n, "close_time": [start + timedelta(hours=3)] * n, "y": [0] * n})


def _q(rows: list[tuple[str, datetime, float, float]]) -> pl.DataFrame:
    return pl.DataFrame({"ticker": [r[0] for r in rows], "end_ts": [r[1] for r in rows],
                         "bid_close": [r[2] for r in rows], "ask_close": [r[3] for r in rows]})


def test_quote_asof_age_and_no_lookahead():
    L = _legs(3)
    C = _q([("E-0", T0 - timedelta(hours=2), 0.20, 0.24),        # carried forward to t
            ("E-0", T0 + timedelta(minutes=1), 0.50, 0.55),     # after t: never used
            ("E-1", T0 - timedelta(hours=30), 0.10, 0.12),      # older than 24 h
            ("E-2", T0, 0.30, 0.30)])                           # locked book: not a quote
    q = P.quote_asof(L, C, "t")
    assert q["ticker"].to_list() == ["E-0"] and q["bid"][0] == 0.20


def test_early_start_exclusion():
    L = _legs(4)
    t24 = T0 + timedelta(hours=6) - P.H24
    at_t = [("E-0", T0, 0.97, 0.99), ("E-1", T0, 0.97, 0.99), ("E-2", T0, 0.10, 0.12), ("E-3", T0, 0.40, 0.45)]
    before = [(tk, t24 - timedelta(minutes=5), 0.20, 0.25) for tk, *_ in at_t]
    Q, info = P.decision_rows(L, _q(at_t + before))
    assert Q.height == 0 and info["events_excluded_early_start"] == 1          # 50% said at t vs 0% a day before
    Q, info = P.decision_rows(L, _q(at_t + [(tk, t24 - timedelta(minutes=5), b, a) for tk, _, b, a in at_t]))
    assert Q.height == 4 and info["events_excluded_early_start"] == 0          # already said a day before
    assert (Q["end"] == T0 + timedelta(hours=6) - P.END_MARGIN).all()


def test_order_prices():
    d = pl.DataFrame({"fair": [0.30, 0.30, 0.95], "bid": [0.28, 0.20, 0.94], "ask": [0.32, 0.21, 0.96]})
    yb, nb = P.order_prices(pl.col("fair"), pl.col("bid"), pl.col("ask"), 0.08)
    o = d.select(yb.alias("yb"), nb.alias("nb"))
    assert o["yb"].to_list() == [pytest.approx(0.22), pytest.approx(0.20), pytest.approx(0.87)]   # capped at ask - 1c
    assert o["nb"].to_list()[:2] == [pytest.approx(0.62), pytest.approx(0.62)]
    assert o["nb"][2] is None                                                                      # 1 - .95 - .08 < 1c


# ---- fills ----------------------------------------------------------------------------------------------
def _rows(asks: list[float], fair: float = 0.30, y: int = 0, fee_type: str = "quadratic") -> pl.DataFrame:
    n = len(asks)
    return pl.DataFrame({
        "row": list(range(n)), "ticker": [f"E-{i}" for i in range(n)], "event_ticker": ["E"] * n,
        "t": [T0] * n, "end": [T0 + timedelta(hours=5, minutes=50)] * n, "bid": [a - 0.04 for a in asks], "ask": asks,
        "fv_cal": [fair] * n, "fv_mid": [fair] * n, "y": [y] * n, "fee_type": [fee_type] * n, "fee_multiplier": [1.0] * n},
        schema_overrides={"row": pl.UInt32})


def _tape(rows: list[tuple[str, int, float, float, bool | None]]) -> pl.DataFrame:
    return pl.DataFrame({"ticker": [r[0] for r in rows], "ts": [T0 + timedelta(minutes=r[1]) for r in rows],
                         "hi": [r[2] for r in rows], "lo": [r[3] for r in rows], "yes_taker": [r[4] for r in rows]},
                        schema=P.TAPE_SCHEMA)


def _get(F: pl.DataFrame, row: int, side: str = "no", cfg: tuple = ("fv_cal", "end_cb", 0.08)) -> dict:
    return F.filter((pl.col("row") == row) & (pl.col("side") == side) & (pl.col("fair") == cfg[0])
                    & (pl.col("endc") == cfg[1]) & (pl.col("margin") == cfg[2])).row(0, named=True)


def test_trade_through_and_breaker():
    # fair 30c: NO bid 62c = sell YES at 38c. E-0: a print AT 38c does not fill, 39c does (minute 20).
    # E-1 (ask 30c <= 80c): a 96c print at minute 60 trips the breaker, and fills E-1 itself.
    # E-2 (ask 85c, fair 83c: sell YES at 91c): its 93c print at minute 90 fills only without the
    # breaker; E-3's 40c print in the breaker's minute still fills (reaction time).
    Q = _rows([0.32, 0.30, 0.85, 0.32]).with_columns(
        pl.when(pl.col("row") == 2).then(0.83).otherwise(pl.col("fv_cal")).alias("fv_cal"))
    tape = _tape([("E-0", 0, 0.38, 0.38, None), ("E-0", 20, 0.39, 0.36, None), ("E-0", 400, 0.99, 0.99, None),
                  ("E-1", 60, 0.96, 0.31, None), ("E-2", 60, 0.90, 0.90, None), ("E-2", 90, 0.93, 0.93, None),
                  ("E-3", 60, 0.40, 0.40, None), ("E-0", -5, 0.90, 0.10, None)])     # before t: ignored
    q = P.with_kill(Q, tape)
    assert q["kill_ts"][0] == T0 + timedelta(minutes=60)
    F = P.fills(q, tape)
    f0 = _get(F, 0)
    assert f0["filled"] and f0["fill_ts"] == T0 + timedelta(minutes=20) and f0["price"] == pytest.approx(0.62)
    assert f0["pnl1"] == pytest.approx(1 - 0.62)                         # NO won, no maker fee on this series
    assert _get(F, 1)["filled"] and _get(F, 3)["fill_ts"] == T0 + timedelta(minutes=60)
    assert _get(F, 2)["price"] == pytest.approx(0.09) and not _get(F, 2)["filled"]
    assert _get(F, 2, cfg=("fv_cal", "end", 0.08))["fill_ts"] == T0 + timedelta(minutes=90)
    # YES bid at 22c fills only on a print below it
    assert not _get(F, 0, "yes")["filled"]
    assert _get(P.fills(q, _tape([("E-0", 30, 0.30, 0.21, None)])), 0, "yes")["filled"]


def test_prints_need_the_right_taker():
    Q = P.with_kill(_rows([0.32]), _tape([]))
    no_taker = _tape([("E-0", 10, 0.40, 0.40, False)])                   # a NO taker at 40c hit a YES bid
    assert not _get(P.fills(Q, no_taker), 0)["filled"]
    assert _get(P.fills(Q, _tape([("E-0", 10, 0.40, 0.40, True)])), 0)["filled"]


def test_fee_rounded_up_per_order():
    Q = P.with_kill(_rows([0.32], y=1, fee_type="quadratic_with_maker_fees"), _tape([]))
    f = _get(P.fills(Q, _tape([("E-0", 10, 0.40, 0.40, None)])), 0)
    assert f["fee1"] == pytest.approx(0.01)                              # 0.0175 x .62 x .38 = 0.41c -> 1c
    assert f["n5"] == 8 and f["fee5"] == pytest.approx(0.04)             # 8 x 0.41c = 3.3c -> 4c
    assert f["pnl1"] == pytest.approx(-0.63) and f["pnl5"] == pytest.approx(-0.62 - 0.005)


def test_candidates_screen():
    Q = _rows([0.32, 0.32, 0.32]).with_columns(pl.col("fv_cal").alias("fv_mid"))
    H = pl.DataFrame({"ticker": ["E-0", "E-1", "E-2"],
                      "end_ts": [T0 + timedelta(hours=2), T0 + timedelta(hours=2), T0 + timedelta(hours=7)],
                      "p_high": [0.39, 0.37, 0.90], "p_low": [0.30, 0.30, 0.10]})
    c = P.candidates(Q, H)
    assert c["ticker"].to_list() == ["E-0"]            # E-1 never traded through 38c; E-2's hour starts after end
    # the screen must use the most aggressive level of the two fair values: fair 28c sells YES at 36c
    c = P.candidates(Q.with_columns(pl.lit(0.28).alias("fv_mid")), H)
    assert c["ticker"].to_list() == ["E-0", "E-1"]
