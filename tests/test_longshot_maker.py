"""Passive longshot seller (research.longshot_maker) on recorded books: two weather ladder strikes
and one NBA game leg, trimmed from `albert record-vol` 2026-11-14 hours 21-22 UTC, with their
manifest rows (family, expected expiration)."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from albert.backtest.l2 import SimConfig, Simulator, _mark_executions, events_from_rows
from albert.research.longshot_maker import (
    LONGSHOT,
    LongshotSeller,
    _prune,
    _snap_top,
    active_hours,
    attach_settlement,
    bucket_of,
    capped_ledger,
    fair_t,
    fast_simulator,
    file_events,
    game_legs,
    inplay_windows,
    line_event,
    longshot_quote,
    sampled,
    scan_lines,
    settle_fields,
)

FIX = Path(__file__).parent / "fixtures" / "kalshi"
LINES = (FIX / "ws_stream_longshot.jsonl").read_text().splitlines()
META = json.loads((FIX / "ws_stream_longshot_markets.json").read_text())["markets"]
CTX = {t: {"family": r["family"], "event": r["event_ticker"], "category": r["category"],
           "exp": r["expected_expiration_time"]} for t, r in META.items()}


def _replay(prune_every: int | None = None, fast: bool = True):
    evs = events_from_rows([json.loads(x) for x in LINES])
    _mark_executions(evs)
    sim = (fast_simulator if fast else Simulator)(SimConfig(writes_per_s=None))
    st = LongshotSeller(CTX, inplay_windows(CTX))
    for i, ev in enumerate(evs):
        sim.apply(ev)
        st.on_event(sim, ev.market)
        if prune_every and i % prune_every == 0:
            _prune(sim, keep_s=0)
    return sim, st


def test_longshot_side_and_target_prices_on_recorded_snapshots():
    snaps = [json.loads(json.loads(x)["frame"])["msg"] for x in LINES if "orderbook_snapshot" in x]
    assert len(snaps) == 3
    st = LongshotSeller({}, {})
    for msg in snaps:
        yb, ya = _snap_top(msg)
        side, ask = longshot_quote(yb, ya)
        assert ask <= LONGSHOT + 1e-9          # every fixture market opens as a longshot
        q = {"yes_bid": yb, "yes_ask": ya}
        bid_side, bid, offered = st.target("join", q)
        assert bid_side == ("no" if side == "yes" else "yes") and abs(bid - (1 - ask)) < 1e-9 and offered == ask
        imp = st.target("improve", q)
        other = yb if side == "yes" else 1 - ya
        if (other is None or ask - 0.01 > other + 1e-9) and ask >= 0.02 - 1e-9:
            assert abs(imp[1] - (bid + 0.01)) < 1e-9 and abs(imp[2] - (ask - 0.01)) < 1e-9
        else:
            assert imp is None


def test_replay_fills_both_variants_only_as_favourite_bids():
    sim, st = _replay()
    by_variant = {st.meta[f.oid]["variant"] for f in sim.fills}
    assert by_variant == {"join", "improve"}
    for f in sim.fills:
        m = st.meta[f.oid]
        assert f.price >= 1 - LONGSHOT - 1e-9 and abs(f.price - (1 - m["q"])) < 1e-9
        assert m["bucket"] == bucket_of(m["q"]) and m["family"] == CTX[f.market]["family"]
        assert f.how in ("queue", "through", "cross")
    counted: dict = {}
    for f in sim.fills:
        k = (st.meta[f.oid]["variant"], f.market)
        counted[k] = counted.get(k, 0) + f.qty
    assert {k: v for k, v in st.filled.items() if v} == counted
    assert all(n <= 3 for n in counted.values())
    # the NBA leg is a day before its scheduled tip-off: pre-game; weather has no sports phase
    phases = {(CTX[f.market]["family"], st.meta[f.oid]["phase"]) for f in sim.fills}
    assert ("game", "pre") in phases and phases <= {("game", "pre"), ("weather", "n/a")}


def test_fast_market_gives_the_simulators_own_fills():
    a, _ = _replay(fast=False)
    b, _ = _replay()
    assert a.fills and [vars(f) for f in a.fills] == [vars(f) for f in b.fills]
    assert {m: v for m, v in a.quotes.items()} == {m: v for m, v in b.quotes.items()}


def test_pruning_finished_orders_and_old_quotes_changes_no_fill():
    a, _ = _replay()
    b, _ = _replay(prune_every=7)
    assert [(f.t_ns, f.market, f.side, f.price, f.qty, f.how) for f in a.fills] == \
           [(f.t_ns, f.market, f.side, f.price, f.qty, f.how) for f in b.fills]
    assert len(b.orders) < len(a.orders)


def test_scan_and_candidate_start_hours():
    rows = scan_lines(LINES, "2026-11-14T21")
    got = {r["ticker"]: r for r in rows}
    assert set(got) == set(META)
    assert all(r["snap_ls"] <= LONGSHOT + 1e-9 for r in rows)
    assert sum(r["trades"] for r in rows) == sum('type\\":\\"trade' in x for x in LINES)
    df = pl.DataFrame(rows + [
        {"hour": "2026-11-14T20", "ticker": t, "snap_ls": 0.5, "trade_ls": None, "trades": 0} for t in META] + [
        {"hour": "2026-11-14T19", "ticker": t, "snap_ls": 0.5, "trade_ls": 0.3, "trades": 1} for t in META] + [
        {"hour": "2026-11-14T22", "ticker": t, "snap_ls": 0.4, "trade_ls": 0.4, "trades": 1} for t in META])
    act = active_hours(df)
    # in the zone at 21:00: 21 is replayed, and 20 too (the entry happened during 20); 19 and 22 are not
    assert all(act[t] == {"2026-11-14T20", "2026-11-14T21"} for t in META)
    one = pl.DataFrame([{"hour": "2026-11-14T22", "ticker": "X", "snap_ls": 0.5, "trade_ls": 0.07, "trades": 2}])
    assert active_hours(one) == {"X": {"2026-11-14T22"}}     # a print at <= 10c makes its hour active


def test_capped_ledger_keeps_open_cost_within_the_bankroll():
    sim, st = _replay()
    t0 = datetime(2026, 11, 14, 21, tzinfo=UTC)
    rows = [{"oid": f.oid, "market": f.market, "event": f.market.rsplit("-", 1)[0], "side": f.side,
             "price": f.price, "qty": f.qty, "fee": f.fee, "fee_exact": f.fee, "variant": st.meta[f.oid]["variant"],
             "t_in": datetime.fromtimestamp(f.t_ns / 1e9, UTC), "settle_ts": t0 + timedelta(days=1),
             "final_value": 1.0} for f in sim.fills]
    f = pl.DataFrame(rows)
    full = capped_ledger(f, "all", bankroll=100.0)
    assert full.height == f.height
    one = capped_ledger(f, "all", bankroll=1.0)   # each position costs ~0.9-0.99: only one fits
    assert one.height == 1 and one["cost_usd"].sum() <= 1.0
    assert set(full.columns) >= {"struct_id", "event_ticker", "ticker", "side", "n", "t_in", "cost_usd",
                                 "settle_ts", "payoff_usd"}


def test_settlement_from_recorded_market_pages():
    pages = json.loads((FIX / "longshot_settle_markets.json").read_text())["pages"]
    info = {t: v for p in pages.values() for t, v in settle_fields(p).items()}
    assert set(META) <= set(info)
    sim, _ = _replay()
    rows = [{"oid": f.oid, "market": f.market, "side": f.side, "price": f.price, "qty": f.qty, "fee": f.fee,
             "fee_exact": f.fee} for f in sim.fills]
    f = attach_settlement(pl.DataFrame(rows), info, {}, datetime(2026, 11, 16, tzinfo=UTC))
    assert f["settled"].all()
    for r in f.iter_rows(named=True):
        assert r["won"] == (info[r["market"]]["result"] == r["side"])
        assert r["final_value"] == float(r["won"]) and abs(r["gross_pc"] - (r["final_value"] - r["price"])) < 1e-12
        assert r["settle_ts"] == datetime.fromisoformat(info[r["market"]]["settlement_ts"])
    # San Antonio was the favourite (NO at <= 10c was the longshot): the sold longshot expired worthless
    sas = f.filter(pl.col("market") == "KXNBAGAME-26NOV15SASUTA-SAS")
    assert sas.height and sas["won"].all() and (sas["side"] == "yes").all() and (sas["price"] >= 0.9).all()


def test_streaming_labels_match_the_offline_classifier():
    offline = events_from_rows([json.loads(x) for x in LINES])
    _mark_executions(offline)
    stream = list(file_events(LINES))
    assert [(e.recv_ns, e.kind, e.market) for e in stream] == [(e.recv_ns, e.kind, e.market) for e in offline]
    assert [e.is_exec for e in stream] == [e.is_exec for e in offline] and any(e.is_exec for e in stream)


def test_ladder_sampling_keeps_whole_events_and_every_other_family():
    strikes = [f"KXNBATOTAL-26NOV14LALBOS-{k}" for k in range(200, 240)]
    assert len({sampled(t, "ladder", 0.1) for t in strikes}) == 1
    events = [f"KXNBATOTAL-26NOV{d:02d}X{i}-220" for d in range(1, 29) for i in range(40)]
    share = sum(sampled(t, "ladder", 0.1) for t in events) / len(events)
    assert 0.06 < share < 0.14
    assert all(sampled(t, fam, 0.1) for t in events for fam in ("game", "half", "weather", "other"))


def test_single_decode_parser_matches_events_from_rows():
    fast = [line_event(x) for x in LINES]
    slow = events_from_rows([json.loads(x) for x in LINES])
    assert [(e.recv_ns, e.kind, e.market, e.msg) for e in fast if e] == [(e.recv_ns, e.kind, e.market, e.msg) for e in slow]


def test_fair_t_uses_the_binomial_error_of_the_sold_longshot():
    # 100 one-lot sales at 2c in 100 markets, none lost: +2c each, binomial SE sqrt(100 * .02 * .98) / 100
    f = pl.DataFrame({"market": [f"M{i}" for i in range(100)], "qty": [1.0] * 100, "price": [0.98] * 100,
                      "net_exact_pc": [0.02] * 100})
    assert abs(fair_t(f) - 0.02 / 0.014) < 1e-9
    # the same 100 contracts in one market are one outcome: the error is 10x larger
    one = f.with_columns(pl.lit("M").alias("market"))
    assert abs(fair_t(one) - fair_t(f) / 10) < 1e-9


def test_one_leg_per_game_event():
    ctx = {"KXNBAGAME-26NOV15SASUTA-SAS": {"family": "game", "event": "KXNBAGAME-26NOV15SASUTA"},
           "KXNBAGAME-26NOV15SASUTA-UTA": {"family": "game", "event": "KXNBAGAME-26NOV15SASUTA"},
           "KXEPLGAME-26NOV22ARSCHE-TIE": {"family": "game", "event": "KXEPLGAME-26NOV22ARSCHE"},
           "KXEPLGAME-26NOV22ARSCHE-CHE": {"family": "game", "event": "KXEPLGAME-26NOV22ARSCHE"},
           "KXEPLGAME-26NOV22ARSCHE-ARS": {"family": "game", "event": "KXEPLGAME-26NOV22ARSCHE"},
           **{t: {"family": c["family"], "event": c["event"]} for t, c in CTX.items() if c["family"] != "game"}}
    assert game_legs(ctx) == {"KXNBAGAME-26NOV15SASUTA-SAS", "KXEPLGAME-26NOV22ARSCHE-ARS"}


def test_settlement_refuses_fills_without_metadata():
    import pytest

    sim, _ = _replay()
    rows = [{"oid": f.oid, "market": f.market, "side": f.side, "price": f.price, "qty": f.qty, "fee": f.fee,
             "fee_exact": f.fee} for f in sim.fills]
    pages = json.loads((FIX / "longshot_settle_markets.json").read_text())["pages"]
    info = settle_fields(pages["KXLOWTSEA-26NOV14"])          # the other two events "failed to fetch"
    with pytest.raises(RuntimeError, match="no settlement metadata"):
        attach_settlement(pl.DataFrame(rows), info, {}, datetime(2026, 11, 16, tzinfo=UTC))
