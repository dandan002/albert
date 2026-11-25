import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import approval2 as A

FIX = Path(__file__).parent / "fixtures"


def _markets() -> pl.DataFrame:
    payload = json.loads((FIX / "kalshi" / "approval2_trumpapprove_26nov19_trimmed.json").read_text())
    rows = [A.market_row(m, "KXTRUMPAPPROVE", "settled") for m in payload["markets"]]
    return pl.DataFrame(rows, schema=A.MARKET_COLS)


def test_bracket_titles():
    assert A.bracket("38.5%") == (385, 385)
    assert A.bracket("Above 38.5%") == (386, 1000)
    assert A.bracket("Below 37.9%") == (0, 378)
    assert A.bracket("37.5-37.7%") == (375, 377)
    assert A.bracket("38.8 to 39.0") == (388, 390)
    assert A.bracket("") is None


def test_settle_instant_rounds_up_59():
    assert A.settle_instant(datetime(2026, 11, 19, 17, 59, tzinfo=UTC)) == datetime(2026, 11, 19, 18, tzinfo=UTC)
    assert A.settle_instant(datetime(2026, 11, 20, 16, 0, tzinfo=UTC)) == datetime(2026, 11, 20, 16, tzinfo=UTC)


def test_order_fee_rounds_up_per_order():
    assert A.order_fee(1, 0.5) == 0.02          # 0.0175 -> 2c
    assert A.order_fee(2, 0.95) == 0.01         # 0.00665 -> 1c
    assert A.order_fee(10, 0.5) == 0.18         # 0.175 -> 18c


def test_legs_and_value_from_real_event():
    lg = A.legs(_markets(), ["KXTRUMPAPPROVE"])
    assert lg.height == 9
    assert lg["T"].unique().to_list() == [datetime(2026, 11, 19, 18, tzinfo=UTC)]
    v = A.event_values(lg)
    assert v.height == 1
    assert v["value"][0] == 383 and v["win_lo"][0] == 383 and v["n_yes"][0] == 1
    # the legs partition the outcome space: tails below 37.9 and above 38.5
    assert min(zip(lg["lo"], lg["hi"], strict=True)) == (0, 378)
    assert max(zip(lg["lo"], lg["hi"], strict=True)) == (386, 1000)


def test_parse_ras_tracker_and_home_link():
    html = (FIX / "rasmussen" / "prez_track_nov20_trimmed.html").read_text()
    r = A.parse_ras_tracker(html)
    assert r == {"weekday": "Friday", "approve": 41, "disapprove": 58, "csv_last_date": "2026-11-20",
                 "csv_last_approve": 41}
    home = (FIX / "rasmussen" / "home_20261120_trimmed.html").read_text()
    assert A.home_links_tracker(home, date(2026, 11, 20)) is True
    assert A.home_links_tracker(home, date(2026, 11, 19)) is False
    assert A.home_links_tracker("<p>nothing</p>", date(2026, 11, 19)) is None
    assert A.track_slug(date(2026, 10, 7)) == "prez_track_oct07"


def test_quotes_at_forward_fills_sparse_candles():
    t0 = datetime(2026, 11, 19, 12, tzinfo=UTC)
    cand = pl.DataFrame({"ticker": ["X", "X"], "end_ts": [t0, t0 + timedelta(minutes=5)],
                         "bid": [0.40, None], "ask": [0.45, 0.47]})
    frame = pl.DataFrame({"ticker": ["X", "X"], "t": [t0 + timedelta(minutes=2), t0 + timedelta(minutes=9)]})
    q = A.quotes_at(cand, frame).sort("t")
    assert q["bid"].to_list() == [0.40, 0.40]
    assert q["ask"].to_list() == [0.45, 0.47]


def test_fills_respect_event_budget():
    T = datetime(2026, 11, 19, 18, tzinfo=UTC)
    sel = pl.DataFrame({"event": ["E"] * 4, "ticker": [f"E-{i}" for i in range(4)], "series": ["S"] * 4,
                        "T": [T] * 4, "t": [T - timedelta(hours=1)] * 4, "lead_h": [1.0] * 4,
                        "bid": [0.02] * 4, "ask": [0.04] * 4, "y": [0, 0, 1, 0]})
    f = A.fills_from(sel, "no", budget=2.0, per_event=5.0)
    assert f.height == 2                         # each NO order costs ~$1.96 + fee; third would exceed $5
    assert (f["price"] == 0.98).all()
    s = A.summarize(f)
    assert s["fills"] == 2 and s["events"] == 1


def test_two_way_t_matches_one_way_when_dates_are_events():
    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 1, 200)
    g = np.repeat(np.arange(40), 5)
    t2 = A.two_way_t(x, g, g)
    _mu, t1, k = A.cluster_t(x, g)
    assert k == 40 and t2 is not None and abs(t2 - t1) < 0.5


def test_ladder_frame_and_cells_on_real_event():
    lg = A.legs(_markets(), ["KXTRUMPAPPROVE"])
    vals = A.event_values(lg)
    T = lg["T"][0]
    # one quote row per leg two hours before settlement: the settled leg (38.3) is the favourite
    cand = pl.DataFrame({"ticker": lg["ticker"], "end_ts": [T - timedelta(hours=3)] * lg.height,
                         "bid": [0.80 if lo == 383 else 0.02 for lo in lg["lo"]],
                         "ask": [0.84 if lo == 383 else 0.04 for lo in lg["lo"]]})
    dec = A.ladder_frame(lg, vals, cand, [2, 0.5])
    assert dec.height == 2 * lg.height
    top = dec.filter((pl.col("mid_rank") == 1) & (pl.col("lead_h") == 2.0))
    assert top["lo"].to_list() == [383]
    cells = {n: (side, sel) for n, side, sel in A.rule_cells(dec)}
    side, sel = cells["H4 modal YES L2h"]
    f = A.fills_from(sel, side)
    assert f.height == 1 and f["payoff"][0] == f["n"][0]
    side, sel = cells["H3 longshot NO ask<=5c L2h"]
    f = A.fills_from(sel, side)
    assert f.height >= 1 and (f["payoff"] == f["n"].cast(pl.Float64)).all()


def test_ladder_mean_path_weights_mids():
    lg = A.legs(_markets(), ["KXTRUMPAPPROVE"])
    T = lg["T"][0]
    cand = pl.DataFrame({"ticker": lg["ticker"], "end_ts": [T - timedelta(hours=3)] * lg.height,
                         "bid": [0.98 if lo == 383 else 0.0 for lo in lg["lo"]],
                         "ask": [1.0 if lo == 383 else 0.0 for lo in lg["lo"]]})
    m = A.ladder_mean_path(lg, cand, lg["event"][0], [T - timedelta(hours=1)])
    assert abs(m[0] - 383) < 1e-6


def test_paper_fills_from_recorded_books(tmp_path):
    mk = _markets()
    mk.write_parquet(tmp_path / "markets.parquet")
    T = datetime(2026, 11, 19, 18, tzinfo=UTC)
    snap = (T - timedelta(hours=12) + timedelta(minutes=3)).isoformat()
    rows = [{"kind": "book", "fetched_at": snap, "ticker": "KXTRUMPAPPROVE-26NOV19-E38.5", "yes_bid": 0.08,
             "yes_bid_size": 5.0, "yes_ask": 0.12},
            {"kind": "book", "fetched_at": snap, "ticker": "KXTRUMPAPPROVE-26NOV19-E38.3", "yes_bid": 0.40,
             "yes_bid_size": 5.0, "yes_ask": 0.45},
            {"kind": "book", "fetched_at": snap, "ticker": "KXTRUMPAPPROVE-26NOV19-E38.0", "yes_bid": 0.06,
             "yes_bid_size": 1.0, "yes_ask": 0.09}]
    (tmp_path / "forward").mkdir()
    (tmp_path / "forward" / "2026-11-19.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    f = A.paper_fills(tmp_path).sort("ticker")
    assert f["ticker"].to_list() == ["KXTRUMPAPPROVE-26NOV19-E38.0", "KXTRUMPAPPROVE-26NOV19-E38.5"]
    assert f["n"].to_list() == [1, 2]                  # displayed size caps the first order
    assert f["payoff"].to_list() == [1.0, 2.0]         # both legs settled NO (value 38.3)


def test_capped_book_limits_open_cost():
    T = datetime(2026, 11, 19, 18, tzinfo=UTC)
    f = pl.DataFrame({"event": [f"E{i}" for i in range(6)], "t": [T - timedelta(hours=1)] * 6, "T": [T] * 6,
                      "cost": [4.0] * 6, "payoff": [4.1] * 6})
    assert A.capped_book(f, max_open=20.0).height == 5
