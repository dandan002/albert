"""Polymarket US cross-venue study on recorded payloads: the gateway's moneyline listing and minute
price history (``tests/fixtures/pmus``), Kalshi's settled game legs, and one Kalshi game leg's
WebSocket book (``ws_stream_pmus_xvenue.jsonl.zst``: TXST-UTSA from its 01:00 UTC snapshot through
five best-price changes, 2026-11-16, at the start of play)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import zstandard

from albert.config import Settings
from albert.research import pmus_xvenue as px
from albert.venues import kalshi as k

FIX = Path(__file__).parent / "fixtures"
LISTING = json.loads((FIX / "pmus" / "markets_moneyline_closed.json").read_text())
HISTORY = json.loads((FIX / "pmus" / "price_history_txst_utsa.json").read_text())
SETTLED = json.loads((FIX / "kalshi" / "settled_game_legs.json").read_text())
STREAM = zstandard.ZstdDecompressor().decompress((FIX / "kalshi" / "ws_stream_pmus_xvenue.jsonl.zst").read_bytes())
LEG = "KXNCAAMBGAME-26NOV15TXSTUTSA-TXST"
OTHER = "KXNCAAMBGAME-26NOV15TXSTUTSA-UTSA"
SLUG = "aec-cbb-txst-utsa-2026-11-15"


def _games() -> pl.DataFrame:
    return pl.DataFrame(px.game_rows(LISTING), schema=px.GAME_SCHEMA)


def _kalshi() -> pl.DataFrame:
    rows = [r for page in SETTLED["pages"] for r in k.market_rows(page, {}, "settled")]
    return k.markets_frame(rows, non_financial_only=False)


def _matched() -> pl.DataFrame:
    return px.match(px.kalshi_games(_kalshi()), _games())


def test_game_rows_on_recorded_listing():
    g = _games()
    assert g.height == 3
    tx = g.filter(pl.col("slug") == SLUG).row(0, named=True)
    assert (tx["league"], tx["code_a"], tx["code_b"]) == ("cbb", "txst", "utsa")
    assert tx["long_name"] == "Texas State" and tx["short_name"] == "UTSA"
    assert tx["game_start"] == datetime(2026, 11, 16, 1, tzinfo=UTC)
    assert set(g["settlement"].to_list()) <= {0.0, 1.0}
    # women's teams carry " (W)" on Polymarket US only; pm_frame strips it for matching
    pf = px.pm_frame(g.filter(pl.col("league") == "wcbb"))
    assert not any("(W)" in x for x in pf["pm_name0"].to_list() + pf["pm_name1"].to_list())


def test_match_maps_long_side_and_settlements_agree():
    m = _matched()
    # both recorded Kalshi games match, every leg once; Bob Jones (no Kalshi market) does not
    assert m.height == 4 and m["ticker"].n_unique() == 4
    assert "aec-cbb-bobj-scup-2026-11-11" not in m["slug"].to_list()
    by = {r["ticker"]: r for r in m.iter_rows(named=True)}
    assert by[LEG]["long"] and not by[OTHER]["long"]
    assert all(r["pm_y"] == r["y"] for r in by.values())


def test_history_rows_are_minute_books():
    rows = px.history_rows(HISTORY, SLUG)
    assert len(rows) == len(HISTORY["history"]) > 40
    assert all(r["ts"] % 60 == 0 for r in rows)
    raw = {p["timestamp"]: p for p in HISTORY["history"]}
    for r in rows:
        assert r["ask"] == raw[r["ts"]]["longPrice"]
        assert r["bid"] == pytest.approx(1 - raw[r["ts"]]["shortPrice"])
        assert r["bid"] <= r["ask"] + 1e-9


@pytest.fixture
def stream_settings(tmp_path) -> Settings:
    """A data dir holding the fixture as hour 01 of connection v5, with a record-vol manifest."""
    s = Settings(data_dir=tmp_path / "data")
    day = s.raw_dir / "stream" / "kalshi" / "2026-11-16"
    day.mkdir(parents=True)
    (day / "01-v17.jsonl.zst").write_bytes(zstandard.ZstdCompressor().compress(STREAM))
    runs = s.data_dir / "runs"
    runs.mkdir(parents=True)
    (runs / "20261116T060000Z-record-vol.json").write_text(json.dumps({
        "created_at": "2026-11-16T06:00:00+00:00",
        "params": {"started_at": "2026-11-15T17:00:00+00:00", "kalshi_markets": [
            {"ticker": LEG, "series_ticker": "KXNCAAMBGAME", "conn": "v17"},
            {"ticker": OTHER, "series_ticker": "KXNCAAMBGAME", "conn": "v17"}]}}))
    return s


def _frames():
    rows = [json.loads(x) for x in STREAM.decode().splitlines()]
    return [(r, json.loads(r["frame"])) for r in rows if LEG in r["frame"]]


def test_server_quotes_use_the_exchange_clock(stream_settings):
    q = px.server_quotes(stream_settings, [LEG], "2026-11-16T01", "2026-11-16T02", log=None)
    assert q.height > 5 and q["ticker"].unique().to_list() == [LEG]
    assert (q["srv_ns"] <= q["recv_ns"]).all()          # the recorder only ever receives late
    two = q.filter(pl.col("bid").is_finite() & pl.col("ask").is_finite())
    assert two.height and (two["bid"] < two["ask"]).all()
    # the first row is the snapshot's book, stamped with the frame's sending_ts_ms
    snap_r, snap_f = next((r, f) for r, f in _frames() if f["type"] == "orderbook_snapshot")
    assert q["recv_ns"][0] == snap_r["recv_ns"]
    assert q["srv_ns"][0] == snap_f["sending_ts_ms"] * 1_000_000
    assert q["bid"][0] == max(float(p) for p, n in snap_f["msg"]["yes_dollars_fp"] if float(n) >= 1)


def test_l2_at_clocks_and_book_tops(stream_settings):
    q = px.server_quotes(stream_settings, [LEG], "2026-11-16T01", "2026-11-16T02", log=None)
    frames = _frames()
    snap_i = next(i for i, (_, f) in enumerate(frames) if f["type"] == "orderbook_snapshot")
    # the delta received longest after its exchange time: between the two, the clocks disagree
    snap_recv = frames[snap_i][0]["recv_ns"]
    deltas = [(r, f) for r, f in frames[snap_i + 1:]
              if f["type"] == "orderbook_delta" and f["msg"]["ts_ms"] * 1_000_000 > snap_recv]
    lagged = max(deltas, key=lambda rf: rf[0]["recv_ns"] - rf[1]["msg"]["ts_ms"] * 1_000_000)
    assert lagged[0]["recv_ns"] > lagged[1]["msg"]["ts_ms"] * 1_000_000
    mid_ns = (lagged[0]["recv_ns"] + lagged[1]["msg"]["ts_ms"] * 1_000_000) // 2
    qs = pl.DataFrame({"ticker": [LEG] * 3, "ns": [int(q["recv_ns"][3]), mid_ns, int(q["recv_ns"][-1]) + 1]})
    l2 = px.l2_at(stream_settings, qs, log=None)
    assert l2.height == 6 and set(l2["clock"]) == {"recv", "server"}
    srv = l2.filter(pl.col("clock") == "server").sort("ns")
    # the server-clock answer equals the server-quote table as of the same instant
    kb = px.KBook(q["srv_ns"].to_numpy(), q["bid"].to_numpy(), q["ask"].to_numpy())
    b, a = kb.at(srv["ns"].to_numpy())
    assert np.allclose(b, srv["bid1"].to_numpy(), equal_nan=True) and np.allclose(a, srv["ask1"].to_numpy(), equal_nan=True)
    # book_tops: a size threshold above the best level's size moves to the next level
    snap = frames[snap_i][1]["msg"]
    book = {sd: {float(p): float(n) for p, n in snap[f"{sd}_dollars_fp"] if float(n) > 0} for sd in ("yes", "no")}
    best = max(book["yes"])
    top = px.book_tops(book, min_size=book["yes"][best] + 1)
    assert top["bid_raw"] == best and top["bid_raw_sz"] == book["yes"][best]
    assert top["bid1"] < best or np.isnan(top["bid1"])


def test_taker_decides_only_on_known_points(stream_settings):
    """Every decision uses a Polymarket US point only once its minute has ended (stamp + 60 s)
    plus latency, and pays the Kalshi ask plus the exact taker fee."""
    q = px.server_quotes(stream_settings, [LEG], "2026-11-16T01", "2026-11-16T02", log=None)
    kb = px.KBook(q["srv_ns"].to_numpy(), q["bid"].to_numpy(), q["ask"].to_numpy(),
                  q["bid_sz"].to_numpy(), q["ask_sz"].to_numpy())
    hist = pl.DataFrame(px.history_rows(HISTORY, SLUG), schema=px.HIST_SCHEMA)
    lg = px.legs(_matched()).filter(pl.col("ticker") == LEG)
    al = px.aligned(hist, lg, {LEG: kb}, offsets_s=(0, 60))
    assert al.height == hist.height
    d = al.filter(pl.col("k_ask_d").is_finite())
    assert d.height >= 1                                    # the fixture's minute with a Kalshi book
    for r in d.iter_rows(named=True):
        b, a = kb.at(r["ts"] * 1_000_000_000 + int((px.ASOF_DELAY + px.LATENCY) * 1e9))
        assert (r["k_bid_d"], r["k_ask_d"]) == (b[0], a[0])
    tr = px.taker_trades(al, -1.0, 60, "inplay")             # edge -1: every eligible point trades
    assert tr.height >= 1
    for r in tr.iter_rows(named=True):
        src = al.filter(pl.col("ts") * 1_000_000_000 + int((px.ASOF_DELAY + px.LATENCY) * 1e9) == r["t_in_ns"])
        assert src.height == 1
        ka = src["k_ask_d"][0]
        assert r["side"] == "yes" and r["price"] == ka
        assert r["cost"] == pytest.approx(ka + 0.07 * ka * (1 - ka))
        if r["how"] == "exit":                              # sold at the bid a minute later, fee paid
            b, _ = kb.at(r["t_out_ns"])
            assert r["t_out_ns"] == r["t_in_ns"] + 60_000_000_000
            assert r["value"] == pytest.approx(b[0] - 0.07 * b[0] * (1 - b[0]))
        else:                                               # past the fixture's book: held; TXST won
            assert r["value"] == 1.0
    # displayed size at entry comes from the exchange-clock book, floored to its size bucket
    assert tr["entry_size"].is_finite().all() and tr["entry_size"].is_in([0.0, *px.SIZE_BUCKETS]).all()
    led = px.ledger(tr, "pmus_taker", "test")
    assert led.height == tr.filter(pl.col("entry_size") >= 1).height
    assert (led["cost_usd"] <= px.STAKE + 1).all()
    sizes = dict(zip(tr["t_in_ns"].to_list(), tr["entry_size"].to_list(), strict=True))
    assert all(r["n"] <= sizes[int(r["struct_id"].rsplit("@", 1)[1])] for r in led.iter_rows(named=True))
    assert px.per_dollar(tr)["n"] == tr.height
