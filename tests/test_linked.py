"""Cross-series linked contracts: link parsing on recorded market payloads, gap / fee math, the taker
and maker fill rules, settlement payoffs and the capped book (synthetic frames)."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from albert.fees import kalshi_fee
from albert.research import linked as lk
from albert.venues import kalshi as k

FIX = Path(__file__).parent / "fixtures" / "kalshi"
T0 = datetime(2026, 10, 5, 12, tzinfo=UTC)


# ---- recorded payloads ---------------------------------------------------------------------------
def _recorded_markets() -> pl.DataFrame:
    rec = json.loads((FIX / "linked_markets.json").read_text())
    info = {s: {"category": "Sports", "fee_type": "quadratic", "fee_multiplier": 1.0}
            for s in {k.series_of(m["event_ticker"]) for m in rec["markets"]}}
    m = k.markets_frame(k.market_rows({"markets": rec["markets"]}, info, "live"), non_financial_only=False)
    return lk.with_keys(m.select(lk.MARKET_COLS))


@pytest.fixture(scope="module")
def rec_links() -> pl.DataFrame:
    m = _recorded_markets()
    return lk.build_links(loader=lambda series, **_: m.filter(pl.col("series_ticker").is_in(series)),
                          names=set(m["series_ticker"]))


def _has(links: pl.DataFrame, family: str, a: str, b: str, side: str = "yes") -> bool:
    return links.filter((pl.col("family") == family) & (pl.col("a") == a) & (pl.col("b").list.first() == b)
                        & (pl.col("b_side") == side)).height == 1


def test_recorded_fixture_holds_both_legs_of_every_pair():
    m = _recorded_markets()
    assert m.height >= 24 and m["rules_primary"].null_count() == 0
    assert set(m["result"].unique()) <= {"yes", "no"}


def test_links_from_recorded_rules_text(rec_links):
    L = rec_links
    assert _has(L, "spread_win", "KXNFLSPREAD-26OCT29CARGB-CAR7", "KXNFLGAME-26OCT29CARGB-CAR")
    assert not _has(L, "spread_win", "KXNFLSPREAD-26OCT29CARGB-CAR7", "KXNFLGAME-26OCT29CARGB-GB")   # other team
    assert _has(L, "tt_total", "KXNFLTEAMTOTAL-26OCT25LARLV-LAR49", "KXNFLTOTAL-26OCT25LARLV-49")
    assert _has(L, "period_total", "KXNFL1HTEAMTOTAL-26OCT04TENBAL-BAL21", "KXNFLTEAMTOTAL-26OCT04TENBAL-BAL21")
    assert _has(L, "first_any", "KXEPLFIRSTGOAL-26OCT18NFOARS-ARSBGUIMA39", "KXEPLGOAL-26OCT18NFOARS-ARSBGUIMA39-1")
    assert _has(L, "prop_cross", "KXMLBRBI-26OCT071600CLECWS-CLEJRAMREZ11-2", "KXMLBHRR-26OCT071600CLECWS-CLEJRAMREZ11-2")
    assert _has(L, "protected", "KXNFLRECYDS-26OCT25TBCAR-CARJMETCHIE13-30", "KXNFLRECYDSPROT-26OCT25TBCAR-CARJMETCHIE13-30")
    assert _has(L, "protected", "KXNFLRECYDSPROT-26OCT25TBCAR-CARJMETCHIE13-30", "KXNFLRECYDS-26OCT25TBCAR-CARJMETCHIE13-30")
    assert _has(L, "finish_ladder", "KXNASCARRACE-NATI26-WISA", "KXNASCARTOP3-NATI26-WISA")
    assert _has(L, "exact_win", "KXATPEXACTMATCH-26OCT12TIERUU-RUU20", "KXATPMATCH-26OCT12TIERUU-RUU")
    assert _has(L, "chart_dup", "KXTOPSONG-26SEP26-MAN", "KXBBCHARTPOSITIONSONG-26SEP26MAN-1")
    assert _has(L, "chart_dup", "KXBBCHARTPOSITIONSONG-26SEP26MAN-1", "KXTOPSONG-26SEP26-MAN")
    assert _has(L, "rain_weekend", "KXRAIN-26OCT25-SFO", "KXRAINWKND-26OCT24-SFO")
    assert _has(L, "mention_say", "KXTRUMPMENTION-26OCT09-DIVI", "KXTRUMPSAY-26OCT12-DIVI")
    assert _has(L, "mention_say", "KXTRUMPMENTION-26OCT09-DIVI", "KXTRUMPSAYMONTH-26NOV01-DIVI")
    assert not _has(L, "say_nested", "KXTRUMPSAYMONTH-26NOV01-DIVI", "KXTRUMPSAY-26OCT12-DIVI")
    assert _has(L, "wx_hilo", "KXLOWTCHI-26OCT15-T59", "KXHIGHCHI-26OCT15-T59", side="no")
    # every link is an implication between two different markets
    assert (L["a"] != L["b"].list.first()).all() and L["link_id"].n_unique() == L.height


def test_links_hold_at_settlement_on_recorded_results(rec_links):
    m = _recorded_markets()
    sv = dict(zip(m["ticker"], m["settlement_value"], strict=True))
    for r in rec_links.iter_rows(named=True):
        va, vb = sv[r["a"]], sum(sv[b] for b in r["b"])
        vb = vb if r["b_side"] == "yes" else 1 - vb
        assert va <= vb + 1e-9, r["link_id"]


def test_small_parsers_on_recorded_strings():
    m = _recorded_markets()
    row = m.filter(pl.col("ticker") == "KXNFLSPREAD-26OCT29CARGB-CAR7").row(0, named=True)
    assert lk.game_key(row["event_ticker"]) == "26OCT29CARGB"
    assert lk.suffix(row["ticker"], row["event_ticker"]) == "CAR7" and lk.team_code("CAR7") == "CAR"
    assert lk.over_strike(row["rules_primary"]) == 6.5
    tt = m.filter(pl.col("ticker") == "KXNFLTEAMTOTAL-26OCT25LARLV-LAR49").row(0, named=True)
    assert lk.over_strike(tt["yes_sub_title"]) == 48.5
    assert lk.ladder_k("CLEJRAMREZ11-2") == ("CLEJRAMREZ11", 2)
    assert lk.score_winner("CFC0TOT1") == "TOT" and lk.score_winner("PAR2NIC2") == "TIE"
    assert lk.wx_range("78° to 79°") == (78.0, 79.0)
    assert lk.wx_range("83° or above")[0] == 83.0 and lk.wx_range("68° or below")[1] == 68.0
    assert lk.code_date("26SEP051840LAAPIT") == date(2026, 9, 5)
    assert lk.text_date("originally scheduled for October 27, 2026") == date(2026, 10, 27)
    say = m.filter(pl.col("ticker") == "KXTRUMPSAYMONTH-26NOV01-DIVI").row(0, named=True)
    st, en = lk.say_window(say["rules_primary"], say["open_time"])
    assert st == datetime(2026, 10, 1, 14, tzinfo=UTC) and en == datetime(2026, 11, 1, 4, tzinfo=UTC)
    assert lk.word_set("Deport / Deported (5+ times)") == frozenset({"deport", "deported"})


# ---- fees and gap math -----------------------------------------------------------------------------
@pytest.mark.parametrize(("n", "p"), [(1, 0.5), (5, 0.37), (5, 0.99), (10, 0.03), (3, 0.5)])
def test_fee_matches_venue_model(n, p):
    assert lk.fee_order(n, p) == pytest.approx(kalshi_fee(n, p, conservative=True))
    got = pl.DataFrame({"p": [p]}).select(lk.fee_expr(n, pl.col("p"), lk.TAKER)).item()
    assert got == pytest.approx(kalshi_fee(n, p, conservative=True))


def _grid(rows):
    return pl.DataFrame(rows, orient="row", schema={"ticker": pl.String, "t": pl.Datetime("us", "UTC"), "bid": pl.Float64,
                                      "ask": pl.Float64, "high": pl.Float64, "low": pl.Float64, "volume": pl.Float64,
                                      "rank": pl.Int32, "ohlc": pl.Boolean, "real": pl.Boolean})


def _mk(tickers, maker=False):
    return pl.DataFrame({"ticker": tickers, "fee_multiplier": [1.0] * len(tickers),
                         "fee_type": ["quadratic_with_maker_fees" if maker else "quadratic"] * len(tickers)})


LINK = pl.DataFrame({"link_id": ["A>B:yes"], "family": ["x"], "kind": ["implies"], "a": ["A"], "b": [["B"]],
                     "b_side": ["yes"], "cluster": ["S|G1"], "detail": [""]})


def test_link_panel_gap_net_and_validity():
    h = timedelta(hours=1)
    g = _grid([
        ("A", T0, 0.60, 0.62, 0.62, 0.6, 10, 0, True, True), ("B", T0, 0.55, 0.57, None, None, 0, 0, True, True),
        ("A", T0 + h, 0.0, 0.62, None, None, 0, 0, True, False), ("B", T0 + h, 0.55, 0.57, None, None, 0, 0, True, False),
        ("A", T0 + 2 * h, 0.50, 0.52, None, None, 0, 0, True, False),                     # B missing this hour
    ])
    p = lk.link_panel(LINK, g, _mk(["A", "B"]), n=5)
    assert p.height == 1                                    # bid 0 (no buyer of A) and the missing B hour drop out
    r = p.row(0, named=True)
    assert r["gap"] == pytest.approx(0.60 - 0.57)           # sell A at its bid, buy B at its ask
    fees = kalshi_fee(5, 0.40) + kalshi_fee(5, 0.57)        # NO on A costs 1 - 0.60
    assert r["fee"] == pytest.approx(fees) and r["net"] == pytest.approx(0.03 - fees / 5)


def test_link_panel_no_side_leg_costs_one_minus_bid():
    link = LINK.with_columns(pl.lit("no").alias("b_side"))
    g = _grid([("A", T0, 0.30, 0.32, None, None, 0, 0, True, True), ("B", T0, 0.75, 0.77, None, None, 0, 0, True, True)])
    r = lk.link_panel(link, g, _mk(["A", "B"])).row(0, named=True)
    assert r["b_cost"] == pytest.approx(0.25) and r["gap"] == pytest.approx(0.30 - 0.25)


def _panel(rows):
    """Minimal panel for the entry rules: (t, a_bid, a_ask, b_cost, net, a_high_next, a_real_next, b_cost_next)."""
    out = []
    for i, (a_bid, a_ask, b_cost, net, hi_n, real_n, b_n) in enumerate(rows):
        out.append({"link_id": "A>B:yes", "t": T0 + timedelta(hours=i), "a_bid": a_bid, "a_ask": a_ask, "b_cost": b_cost,
                    "b_fee": 0.05, "net": net, "a_px": 1 - a_bid, "a_ohlc": True, "has_next": True, "a_high_next": hi_n,
                    "a_real_next": real_n, "b_cost_next": b_n, "a_mk": False, "a_mult": 1.0,
                    "has_prev": i > 0, "net_prev": rows[i - 1][3] if i else None})
    return pl.DataFrame(out)


def test_taker_entries_snapshot_and_persistent():
    p = _panel([(0.5, 0.52, 0.45, 0.01, None, False, None), (0.5, 0.52, 0.49, -0.02, None, False, None),
                (0.6, 0.62, 0.55, 0.02, None, False, None), (0.6, 0.62, 0.55, 0.02, None, False, None)])
    snap = lk.taker_entries(p, 0.0)
    assert snap["t"].to_list() == [T0]                      # first hour with a net gap
    pers = lk.taker_entries(p, 0.0, persist=True)
    assert pers["t"].to_list() == [T0 + timedelta(hours=3)]  # needs the gap an hour earlier too


def test_maker_fills_only_on_prints_strictly_through_the_level():
    # B costs 0.50 + 1c fee per contract -> L = ceil(0.51 + 0.01) = 0.52; A quoted 0.48 / 0.53
    base = (0.48, 0.53, 0.50, -0.05)
    p = _panel([(*base, 0.52, True, 0.50)])                  # high == L: not through
    assert lk.maker_entries(p, 0.01, 0.0).is_empty()
    p = _panel([(*base, 0.53, False, 0.50)])                 # no candle next hour: no trade range known
    assert lk.maker_entries(p, 0.01, 0.0).is_empty()
    p = _panel([(*base, 0.53, True, 0.51)])
    e = lk.maker_entries(p, 0.01, 0.0).row(0, named=True)
    assert e["a_price"] == pytest.approx(1 - 0.52) and e["b_cost"] == pytest.approx(0.51)   # hedge at next hour's B
    assert e["t"] == T0 + timedelta(hours=1)
    p = _panel([(0.53, 0.55, 0.50, 0.0, 0.60, True, 0.5)])   # L <= bid would cross: that is the taker case
    assert lk.maker_entries(p, 0.01, 0.0).is_empty()
    p = _panel([(0.40, 0.45, 0.50, -0.1, 0.60, True, 0.5)])  # L above the ask by 7c: beyond reach
    assert lk.maker_entries(p, 0.01, 0.0).is_empty() and lk.maker_entries(p, 0.01, 0.07).height == 1


# ---- payoffs, book, settlement, episodes ---------------------------------------------------------
def _markets(sv_a, sv_b):
    return pl.DataFrame({"ticker": ["A", "B"], "settlement_ts": [T0 + timedelta(days=1)] * 2,
                         "settlement_value": [sv_a, sv_b], "fee_type": ["quadratic"] * 2, "fee_multiplier": [1.0, 1.0]})


@pytest.mark.parametrize(("sv_a", "sv_b", "payoff"), [(1.0, 1.0, 1.0), (0.0, 0.0, 1.0), (0.0, 1.0, 2.0), (1.0, 0.0, 0.0)])
def test_lock_payoff_by_outcome(sv_a, sv_b, payoff):
    e = pl.DataFrame({"link_id": ["A>B:yes"], "t": [T0], "a_price": [0.40], "b_cost": [0.57], "how": ["taker"]})
    pos = lk.positions(e, LINK, _markets(sv_a, sv_b), n=5)
    s = lk.pos_pnl(pos, LINK).row(0, named=True)
    fees = kalshi_fee(5, 0.40) + kalshi_fee(5, 0.57)
    assert s["payoff_usd"] == pytest.approx(5 * payoff)
    assert s["pnl"] == pytest.approx(payoff - 0.97 - fees / 5)
    assert s["broke"] == (payoff == 0.0)                     # A yes but B no: the link failed (basis risk)


def test_capped_book_enforces_event_and_bankroll_caps():
    links = pl.DataFrame({"link_id": [f"L{i}" for i in range(4)], "cluster": ["S|G1", "S|G1", "S|G2", "S|G3"]})
    pos = pl.DataFrame({"struct_id": [f"s{i}" for i in range(4)], "link_id": [f"L{i}" for i in range(4)],
                        "t_in": [T0 + timedelta(minutes=i) for i in range(4)], "cost_usd": [6.0, 6.0, 60.0, 50.0],
                        "settle_ts": [T0 + timedelta(days=1)] * 4})
    led = lk.capped_book(pos, links, bankroll=100.0, event_cap=10.0)
    assert led["struct_id"].to_list() == ["s0"]             # s1 breaks the G1 cap, s2/s3 the per-event cap
    led = lk.capped_book(pos, links, bankroll=100.0, event_cap=70.0)
    assert led["struct_id"].to_list() == ["s0", "s1", "s2"]  # s3 would take the book past $100
    assert set(led["event_ticker"]) == {"G1", "G2"}


def test_episodes_split_on_gaps_and_missing_hours():
    h = timedelta(hours=1)
    p = pl.DataFrame({"link_id": ["x"] * 6, "t": [T0, T0 + h, T0 + 2 * h, T0 + 4 * h, T0 + 5 * h, T0 + 6 * h],
                      "net": [0.01, 0.0, -0.01, 0.02, 0.03, 0.01]})
    ep = lk.episodes(p).sort("start")
    assert ep["hours"].to_list() == [2, 3] and ep["best"].to_list() == [0.01, 0.03]


def test_settle_check_flags_only_failed_implications(tmp_path, monkeypatch):
    monkeypatch.setattr(lk, "OUT", tmp_path)
    links = pl.DataFrame({"link_id": ["1", "2", "3"], "family": ["f"] * 3, "kind": ["implies"] * 3,
                          "a": ["A", "A", "C"], "b": [["B"], ["C"], ["B"]], "b_side": ["yes", "no", "yes"],
                          "cluster": ["S|G"] * 3, "detail": [""] * 3})
    sv = {"A": 1.0, "B": 1.0, "C": 0.0}
    monkeypatch.setattr(lk, "leg_markets", lambda _: pl.DataFrame({"ticker": list(sv), "settlement_value": list(sv.values())}))
    out = lk.settle_check(links)
    assert out["broken"].to_list() == [0]
    sv["B"] = 0.0
    out = lk.settle_check(links)
    assert out["broken"].to_list() == [1]                    # A yes, B no: link 1 failed; C no => B anything is fine
