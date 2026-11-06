"""Forward paper test (research/forward.py) on trimmed copies of recorded live payloads
(``tests/fixtures/*/forward_*``, recorded 2026-11-06 ~02:45 UTC) plus the round-2 mention trade
pages; quotes in the scoring test are a synthetic grid (pure order-price math), as in
test_mentions_maker."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import polars as pl
import pytest

from albert.config import Settings
from albert.research import forward as F
from albert.research import mentions as M
from albert.research import mentions_maker as MM
from albert.research import ncaaf_maker as nm
from albert.research import rain
from albert.venues import kalshi as k

FIX = Path(__file__).parent / "fixtures"
EV = "KXTRUMPMENTION-26NOV06"
START = datetime(2026, 11, 6, 22, 30, tzinfo=UTC)          # milestone start of EV


def _j(name: str):
    return json.loads((FIX / name).read_text())


class Router:
    """httpx transport serving recorded payloads by URL path suffix; records every request."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((path, dict(request.url.params.multi_items())))
        for suffix, body in self.routes.items():
            if path.endswith(suffix):
                body = body(request) if callable(body) else body
                if isinstance(body, str):
                    return httpx.Response(200, text=body, headers={"content-type": "text/csv"})
                return httpx.Response(200, json=body)
        return httpx.Response(404, json={"error": f"no route for {path}"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


def _mention_routes() -> dict:
    return {"/milestones": _j("kalshi/forward_milestones_mentions.json"),
            "/markets/orderbooks": _j(f"kalshi/forward_orderbooks_{EV}.json"),
            "/markets/candlesticks": _j(f"kalshi/forward_candles_{EV}.json"),
            "/series/KXTRUMPMENTION": _j("kalshi/forward_series_KXTRUMPMENTION.json"),
            "/markets": _j(f"kalshi/forward_markets_{EV}.json")}


# ---- order book and prices -----------------------------------------------------------------------
def test_book_rows_keep_the_best_levels_and_match_orderbook_top():
    page = _j(f"kalshi/forward_orderbooks_{EV}.json")
    rows = F.book_rows(page, datetime(2026, 11, 6, 2, 44, tzinfo=UTC))
    assert len(rows) == len(page["orderbooks"]) == 6
    for r, ob in zip(rows, page["orderbooks"], strict=True):
        top = k.orderbook_top(ob)
        assert r["ticker"] == ob["ticker"]
        assert (r["book_bid"], r["book_ask"], r["book_bid_size"], r["book_ask_size"]) == (
            top["yes_bid"], top["yes_ask"], top["yes_bid_size"], top["yes_ask_size"])
        assert r["yes_levels"][0][0] == r["book_bid"] and r["book_ask"] == pytest.approx(1 - r["no_levels"][0][0])
        assert all(a[0] > b[0] for a, b in zip(r["yes_levels"], r["yes_levels"][1:])) and len(r["yes_levels"]) <= 10


def test_bid_prices_follow_the_harness_quote():
    fair = np.array([0.30, 0.30, 0.05, np.nan])
    bid, ask = np.array([0.28, 0.20, 0.00, 0.4]), np.array([0.32, 0.25, 0.02, 0.5])
    yes, no = F.bid_prices(fair, bid, ask, 0.08)
    assert yes[0] == 0.22 and no[0] == 0.62                 # floor(0.30 - 0.08), floor(0.70 - 0.08)
    assert yes[1] == 0.22 and no[1] == 0.62
    assert np.isnan(yes[2]) and no[2] == 0.87               # YES below 1c: no order
    assert np.isnan(yes[3]) and np.isnan(no[3])
    # post-only cap: one tick inside the opposite quote
    y2, n2 = F.bid_prices(np.array([0.5]), np.array([0.45]), np.array([0.47]), 0.0)
    assert y2[0] == 0.46 and n2[0] == 0.50                 # YES capped at ask - 1c
    y3, n3 = F.bid_prices(np.array([0.5]), np.array([0.53]), np.array([0.56]), 0.0)
    assert y3[0] == 0.50 and n3[0] == 0.46                 # NO capped at 1 - bid - 1c


# ---- mentions --------------------------------------------------------------------------------------
def test_mention_due_window_excludes_window_series_and_ledger():
    page = _j("kalshi/forward_milestones_mentions.json")
    now = START - timedelta(hours=6, minutes=30)
    due = F.mention_due(page, now, F.DUE[0], F.DUE[1], set())
    assert due["event"].to_list() == [EV] and due["start"][0] == START
    assert F.mention_due(page, now, F.DUE[0], F.DUE[1], {EV}).height == 0
    assert F.mention_due(page, START - timedelta(hours=8), F.DUE[0], F.DUE[1], set()).height == 0   # too early
    assert F.mention_due(page, START - timedelta(hours=4), F.DUE[0], F.DUE[1], set()).height == 0   # too late
    wide = F.mention_due(page, datetime(2026, 11, 6, 2, 48, tzinfo=UTC), timedelta(0), timedelta(days=30), set())
    assert "KXTRUMPSAY-26NOV09" not in wide["event"].to_list()     # "say X before <date>": no start
    assert {"KXLASTWORDMENTION-26NOV05", "KXEARNINGSMENTIONDKNG-26NOV06", EV} <= set(wide["event"].to_list())


def test_said24_reads_the_last_candle_closed_by_start_minus_24h():
    page = _j(f"kalshi/forward_candles_{EV}.json")
    at = START - timedelta(hours=24)
    assert F.said24_from_candles(page, at) == 0.0                   # no leg was bid >= 97c a day before
    # the same candles with one leg's bid at 98c -> 1 of the quoted legs
    hot = json.loads(json.dumps(page))
    c = max(hot["markets"][0]["candlesticks"], key=lambda c: c["end_period_ts"])
    c["yes_bid"]["close_dollars"], c["yes_ask"]["close_dollars"] = "0.9800", "0.9900"
    quoted = sum(1 for m in page["markets"] if any(k.parse_ts(x["end_period_ts"]) <= at for x in m["candlesticks"]))
    assert F.said24_from_candles(hot, at) == pytest.approx(1 / quoted)
    assert F.said24_from_candles(page, at - timedelta(hours=5)) == 0.0   # nothing within 2 h: no legs


def test_mentions_snapshot_records_frozen_no_bids_once(tmp_path):
    s = Settings(data_dir=tmp_path)
    r = Router(_mention_routes())
    now = START - timedelta(hours=6, minutes=30)
    path = F.snapshot("mentions", s, now=now, transport=r.transport(), log=lambda *_: None)
    assert path is not None and path.parent == tmp_path / "cache" / "forward" / "mentions"
    df = pl.read_parquet(path)
    assert df.height == 6 and set(df["event"]) == {EV} and df["scheduled"].all() and df["eligible"].all()
    assert df["group"].unique().to_list() == ["trump"] and (df["end"] == START - F.MENTIONS_END).all()
    assert df["fee_type"].unique().to_list() == ["quadratic"] and df["lead_h"][0] == 6.5
    row = df.filter(pl.col("ticker") == f"{EV}-AUTO").row(0, named=True)
    # recorded book: bid 0.28 x 7, ask 0.32 x 2 -> mid 0.30; fv_cal = σ(-0.1156 + 1.0474 logit 0.30) = 0.2683
    assert (row["bid"], row["ask"], row["book_bid_size"], row["book_ask_size"]) == (0.28, 0.32, 7.0, 2.0)
    assert row["fair_cal"] == pytest.approx(0.26834, abs=1e-5)
    assert (row["o_cal8_no"], row["o_cal12_no"], row["o_mid8_no"], row["o_mid12_no"]) == (0.65, 0.61, 0.62, 0.58)
    assert not any(c.endswith("_yes") for c in df.columns if c.startswith("o_"))   # the rule rests NO bids only
    assert row["said"] == 0.0 and row["said24"] == 0.0 and not row["excluded"]
    ledger = json.loads((path.parent / "ledger.json").read_text())
    assert list(ledger) == [EV]
    # idempotent: the next hourly run sees the event in the ledger and makes one request
    n = len(r.calls)
    assert F.snapshot("mentions", s, now=now + timedelta(hours=1), transport=r.transport(), log=lambda *_: None) is None
    assert len(r.calls) == n + 1 and r.calls[-1][0].endswith("/milestones")
    # a forced run records off-schedule rows and leaves the ledger alone
    p2 = F.snapshot("mentions", s, now=now + timedelta(minutes=1), force=True, transport=r.transport(), log=lambda *_: None)
    assert p2 is not None and not pl.read_parquet(p2)["scheduled"].any()
    assert json.loads((path.parent / "ledger.json").read_text()) == ledger


def test_mentions_snapshot_excludes_an_event_that_already_started():
    rows = pl.DataFrame({"quoted": [True] * 10, "bid": [0.98] * 2 + [0.30] * 8})
    assert F.said_share(rows) == pytest.approx(0.2)
    assert F.said_share(rows) > 0.0 + F.SAID_JUMP                  # vs 0 a day earlier -> excluded


# ---- fills and scoring -------------------------------------------------------------------------------
def _oct5_trades() -> pl.DataFrame:
    rows = []
    for leg in ("CORN", "TRAN"):
        rows.extend(k.trade_rows(_j(f"kalshi/mentions_maker_trades_KXTRUMPMENTION-26OCT05-{leg}.json")))
    return pl.DataFrame(rows, schema=k.TRADE_SCHEMA)


def _oct5_snapshot() -> pl.DataFrame:
    """Snapshot rows for two real legs at start - 2 h with a grid of quotes (synthetic numbers:
    only the order-price math depends on them)."""
    start = datetime(2026, 10, 5, 23, tzinfo=UTC)
    rows = []
    for tk in ("KXTRUMPMENTION-26OCT05-CORN", "KXTRUMPMENTION-26OCT05-TRAN"):
        for i, (bid, ask) in enumerate(((0.10, 0.14), (0.40, 0.46), (0.64, 0.70), (0.70, 0.74))):
            t = start - timedelta(hours=2, minutes=i)
            rows.append({"ticker": tk, "event": "KXTRUMPMENTION-26OCT05", "series": "KXTRUMPMENTION", "group": "trump",
                         "snapshot_ts": t, "t": t, "end": start - F.MENTIONS_END, "bid": bid, "ask": ask,
                         "mid": (bid + ask) / 2, "quoted": True, "eligible": True, "scheduled": True,
                         "fee_type": "quadratic", "fee_multiplier": 1.0})
    return F.mention_orders(pl.DataFrame(rows))


def test_first_fills_equal_mentions_maker_fast_fills_on_real_prints():
    trades = _oct5_trades()
    snap = _oct5_snapshot().with_columns(pl.col("ticker").str.ends_with("TRAN").cast(pl.Int8).alias("y"))
    for m in F.MENTIONS_MARGINS:
        tag = int(round(m * 100))
        o = F.order_frame(snap.drop([c for c in snap.columns if c.startswith("o_") and c != f"o_cal{tag}_no"]))
        o = o.join(snap.select("ticker", "t", "y"), on=["ticker", "t"])
        mine = F.settle(F.first_fills(o, trades)).sort("ticker", "t")
        ref = MM.fast_fills(snap.with_columns(pl.col("fair_cal").alias("p")), trades, m)
        ref = ref.filter(pl.col("side") == "no").join(snap.with_row_index("row").select("row", "ticker", "t"), on="row")
        ref = ref.sort("ticker", "t")
        assert mine.height == ref.height == 8
        assert mine["price"].to_list() == ref["price"].to_list()
        assert mine["filled"].to_list() == ref["filled"].to_list()
        assert np.allclose(mine["pnl"].to_numpy(), ref["pnl"].to_numpy())
        assert 0 < mine["filled"].sum() < mine.height
    # a window that has ended before t never fills
    o = F.order_frame(snap).join(snap.select("ticker", "t", "y"), on=["ticker", "t"])
    assert not F.first_fills(o.with_columns((pl.col("t") - timedelta(seconds=1)).alias("end")), trades)["filled"].any()


def test_kill_times_match_the_study_circuit_breaker():
    trades = _oct5_trades()
    snap = _oct5_snapshot()
    kt = F.kill_times(snap.with_columns(pl.lit(0.78).alias("ask")), trades)
    q = snap.select("event", pl.col("snapshot_ts").alias("when"), "ticker", "t").with_columns(pl.lit(0.78).alias("ask"))
    ref = MM.kill_times(q, trades)
    assert kt.height == ref.height > 0
    assert sorted(kt["kill_ts"].to_list()) == sorted(ref["kill_ts"].to_list())


def test_score_mentions_end_to_end(tmp_path):
    s = Settings(data_dir=tmp_path)
    F.write_parquet(_oct5_snapshot(), F.lead_dir(s, "mentions") / "20261005T205600Z.parquet")
    pages = {leg: _j(f"kalshi/mentions_maker_trades_KXTRUMPMENTION-26OCT05-{leg}.json") for leg in ("CORN", "TRAN")}

    def trades(request):
        return pages[request.url.params["ticker"].rsplit("-", 1)[1]]

    r = Router({"/historical/cutoff": _j("kalshi/cutoff.json"), "/trades": trades,
                "/markets": _j("kalshi/forward_results_KXTRUMPMENTION-26OCT05.json")})
    table = F.score("mentions", s, now=datetime(2026, 11, 6, tzinfo=UTC), transport=r.transport(), log=lambda *_: None)
    assert table.height and {"cfg", "cancel", "scope", "fills", "pnl_per_fill_c", "t", "events", "days"} <= set(table.columns)
    allrows = table.filter(pl.col("scope") == "all")
    assert set(allrows["cfg"]) == {"cal8", "cal12", "mid8", "mid12"} and set(allrows["cancel"]) == {"end", "cb"}
    scored = pl.read_parquet(F.lead_dir(s, "mentions") / "scored.parquet")
    end = scored.filter(pl.col("cancel") == "end")
    assert end.height == 4 * 8 and end["filled"].sum() > 0
    # TRAN settled YES: every filled NO bid on it lost its price; CORN settled NO: won 1 - price
    tr = end.filter(pl.col("filled") & pl.col("ticker").str.ends_with("TRAN"))
    co = end.filter(pl.col("filled") & pl.col("ticker").str.ends_with("CORN"))
    assert np.allclose(tr["pnl"].to_numpy(), -tr["price"].to_numpy())
    assert np.allclose(co["pnl"].to_numpy(), 1 - co["price"].to_numpy())
    # the breaker can only remove fills
    cb = scored.filter(pl.col("cancel") == "cb")
    assert cb["filled"].sum() <= end["filled"].sum()
    # results are kept and not requested again
    n = sum(1 for p, _ in r.calls if p.endswith("/markets"))
    F.score("mentions", s, now=datetime(2026, 11, 6, tzinfo=UTC), transport=r.transport(), log=lambda *_: None)
    assert sum(1 for p, _ in r.calls if p.endswith("/markets")) == n


def test_score_waits_for_the_order_window_to_close(tmp_path):
    s = Settings(data_dir=tmp_path)
    F.write_parquet(_oct5_snapshot(), F.lead_dir(s, "mentions") / "20261005T205600Z.parquet")
    r = Router({})
    assert F.score("mentions", s, now=datetime(2026, 10, 5, 22, tzinfo=UTC), transport=r.transport(),
                   log=lambda *_: None).height == 0
    assert r.calls == []


# ---- rain ------------------------------------------------------------------------------------------
def test_asos_multi_maps_stations_to_codes():
    obs = F.asos_multi((FIX / "iem" / "forward_asos_multi.csv").read_text())
    assert set(obs["code"]) == {"NYC", "DJT", "SEA"} and obs.height > 20
    one = pl.DataFrame(rain.asos_precip_rows((FIX / "iem" / "forward_asos_multi.csv").read_text(), "X"))
    assert one.height == obs.height                       # same parser, split by station
    assert obs.filter(pl.col("code") == "DJT")["p01i"].max() == rain.TRACE      # "T" = trace
    assert obs.filter(pl.col("code") == "SEA")["p01i"].max() == 0.01
    assert F.asos_multi("station,valid,lon,lat,p01i\n").height == 0


def test_om_multi_splits_locations_in_request_order():
    page = _j("openmeteo/forward_previous_runs_multi.json")
    om = F.om_multi(page, ["NYC", "PIT"])
    assert set(om["code"]) == {"NYC", "PIT"} and set(om["n"]) == {1, 2}
    assert set(om["model"]) <= set(rain.OM_MODELS) and len(set(om["model"])) >= 9
    assert om.filter(pl.col("code") == "NYC").height == rain.om_frame(page[0], "NYC").height
    with pytest.raises(ValueError):
        F.om_multi(page, ["NYC"])


def test_rain_inputs_build_round1_features_from_recorded_payloads(tmp_path):
    s = Settings(data_dir=tmp_path)
    (tmp_path / "cache").mkdir()
    asos = (FIX / "iem" / "forward_asos_multi.csv").read_text()
    F.write_parquet(F.asos_multi(asos), rain._table(s, "obs"))
    cli = pl.DataFrame(rain.cli_precip_rows(_j("iem/cli_precip.json"), "NYC"), infer_schema_length=None)
    F.write_parquet(cli, rain._table(s, "cli"))
    mos = {"NBS": (FIX / "iem" / "forward_mos_nbs_knyc.csv").read_text(), "GFS": (FIX / "iem" / "forward_mos_gfs_knyc.csv").read_text()}
    om = _j("openmeteo/forward_previous_runs_multi.json")[:1]
    r = Router({"/asos.py": "station,valid,lon,lat,p01i\n", "/mos.py": lambda req: mos[req.url.params["model"]],
                "/v1/forecast": om})
    t = datetime(2026, 11, 6, 2, 45, tzinfo=UTC)
    f = F.rain_inputs(s, ["NYC"], date(2026, 11, 6), t, transport=r.transport())
    assert f.height == 1 and f["code"][0] == "NYC" and not f["decided"][0]
    row = f.row(0, named=True)
    assert row["left"] == 1.0 and row["seen"] == 0.0                    # t is before the CLI day starts
    assert row["om_n"] >= 9 and row["nbs_pany"] is not None and row["gfs_pany"] is not None
    # MOS runs used were public at t (NBS 2 h, GFS 5 h lag): rebuild with t one hour earlier differs only via lags
    nbs = pl.DataFrame(rain.mos_rows(mos["NBS"], "NYC"))
    assert nbs.filter(pl.col("runtime") <= t - rain.MOS_LAG["NBS"]).height > 0
    hosts = {p for p, _ in r.calls}
    assert any(p.endswith("/asos.py") for p in hosts) and any(p.endswith("/v1/forecast") for p in hosts)
    p = F.rain_predict(_toy_model(f), f)
    assert 0 < p[0] < 1


def _toy_model(f: pl.DataFrame):
    """A fitted Logit of the right width (pure math; the frozen model needs the cached features)."""
    X, _ = rain.design(f, rain.VARIANTS[F.RAIN_VARIANT], [])
    m = rain.Logit(lam=1.0)
    m.beta, m.mu, m.sd = np.zeros(X.shape[1] + 1), np.zeros(X.shape[1]), np.ones(X.shape[1])
    return m


def test_rain_orders_use_the_frozen_d12z_blend():
    df = pl.DataFrame({"mid": [0.52, 0.015, 0.52], "bid": [0.51, 0.01, 0.51], "ask": [0.53, 0.02, 0.53],
                       "p_model": [0.30, 0.01, 0.30], "eligible": [True, True, False]})
    out = F.rain_orders(df)
    lg = lambda x: np.log(x / (1 - x))  # noqa: E731
    want = 1 / (1 + np.exp(-(-0.308 + 0.706 * lg(0.52) + 0.587 * lg(0.30))))
    assert out["fair_blend"][0] == pytest.approx(want)
    assert out["fair_midcal"][0] == pytest.approx(1 / (1 + np.exp(-(-0.362 + 1.292 * lg(0.52)))))
    assert out["o_blend8_yes"][0] == np.floor((want - 0.08) * 100) / 100
    assert out["o_blend8_no"][0] == min(np.floor((1 - want - 0.08) * 100) / 100, 0.48)   # cap: 1 - bid - 1c
    assert out["o_mid8_yes"][0] == 0.44 and out["o_mid8_no"][0] == 0.40
    assert out["o_blend8_yes"][1] is None and out["o_blend8_no"][1] == 0.91     # YES below 1c; NO at 91c
    assert all(out[c][2] is None for c in out.columns if c.startswith("o_"))    # ineligible: no bids


def test_rain_ends_cancel_at_the_first_rain_report(tmp_path):
    s = Settings(data_dir=tmp_path)
    asos = (FIX / "iem" / "forward_asos_multi.csv").read_text()
    day = date(2026, 10, 30)
    lo, hi = rain.lst_bounds("SEA", day)                    # 08:00 UTC: the 0.01" report at 07:53 is the day before
    snaps = pl.DataFrame({"code": ["SEA", "SEA", "DJT"], "day": [day, day + timedelta(days=1), day],
                          "lo": [lo, datetime(2026, 10, 30, 7, tzinfo=UTC), rain.lst_bounds("DJT", day)[0]],
                          "hi": [hi, datetime(2026, 10, 30, 12, tzinfo=UTC), rain.lst_bounds("DJT", day)[1]],
                          "close_time": [hi + timedelta(days=1)] * 3})
    r = Router({"/asos.py": asos})
    out = F.rain_ends(s, snaps, datetime(2026, 11, 6, tzinfo=UTC), transport=r.transport())
    ends = dict(zip(zip(out["code"], out["day"]), out["end"], strict=True))
    assert ends[("SEA", day)] == hi                                                   # no 0.01" inside the day
    assert ends[("SEA", day + timedelta(days=1))] == datetime(2026, 10, 30, 8, 3, tzinfo=UTC)   # 07:53 + 10 min
    assert ends[("DJT", day)] == rain.lst_bounds("DJT", day)[1]                       # a trace does not cancel
    assert len(r.calls) == 1


# ---- FCS ------------------------------------------------------------------------------------------
def test_football_due_finds_ncaaf_games_and_their_spread_events():
    page = _j("kalshi/forward_milestones_football.json")
    kick = datetime(2026, 11, 7, 17, tzinfo=UTC)
    due = F.football_due(page, kick - timedelta(hours=6, minutes=30), F.DUE[0], F.DUE[1], set())
    assert set(due["key"]) == {"26NOV07BRWNYALE", "26NOV07NCATMONM"} and set(due["division"]) == {"FCS"}
    assert due.filter(pl.col("key") == "26NOV07BRWNYALE")["spread_events"][0].to_list() == ["KXNCAAFSPREAD-26NOV07BRWNYALE"]
    assert F.football_due(page, kick - timedelta(hours=6, minutes=30), F.DUE[0], F.DUE[1], {"26NOV07BRWNYALE"}).height == 1
    wide = F.football_due(page, datetime(2026, 11, 6, 3, tzinfo=UTC), timedelta(0), timedelta(days=3), set())
    assert "26NOV06VTSMU" in wide["key"].to_list() and not any(x.startswith("26NOV07WPG") for x in wide["key"])  # CFL ignored


def test_fcs_orders_wrap_ncaaf_maker_prices():
    page = _j("kalshi/ncaaf_open_markets.json")
    legs = pl.DataFrame(nm.open_rows(page, datetime(2026, 11, 5, 23, 30, tzinfo=UTC)), infer_schema_length=None)
    legs = legs.filter(pl.col("series_ticker") == "KXNCAAFSPREAD").select(
        "ticker", pl.col("yes_bid").alias("bid"), pl.col("yes_ask").alias("ask"), pl.lit("spread").alias("kind"),
        pl.lit("FCS").alias("tier"), pl.lit("FCS").alias("tgroup"), pl.lit("quadratic_with_maker_fees").alias("fee_type"),
        pl.lit(1.0).alias("fee_multiplier"), pl.lit(0.5).alias("p"))
    frozen = {"blends": {"spread|FCS": {"blend": [0.2083, 0.5015, 0.5935], "mid_only": [0.0800, 1.0342]}},
              "maker": {"source": "blend", "margin": 0.03, "x": 0.02, "sides": "both", "tiers": ["FCS"]}, "taker_theta": 0.03}
    out = nm.forward_prices(legs, frozen).with_columns(((pl.col("ask") - pl.col("bid")) * 100).round(0).alias("spread_c"))
    out = out.with_columns((pl.col("fair").is_not_null() & pl.col("spread_c").is_between(*F.FCS_SPREAD_C)).alias("eligible"))
    o = F.fcs_orders(out)
    assert o["eligible"].sum() > 0
    el = o.filter("eligible")
    # blend3 = ncaaf_maker's bid prices; rule = the same where the frozen gate posts
    assert el["o_blend3_yes"].to_list() == el["px_yes"].to_list()
    assert (el.filter(pl.col("post_yes"))["o_rule_yes"] == el.filter(pl.col("post_yes"))["px_yes"]).all()
    assert el.filter(~pl.col("post_yes"))["o_rule_yes"].null_count() == el.filter(~pl.col("post_yes")).height
    y, n = F.bid_prices(el["fair_mid"].to_numpy(), el["bid"].to_numpy(), el["ask"].to_numpy(), 0.03)
    assert np.allclose(el["o_midcal3_yes"].fill_null(np.nan).to_numpy(), y, equal_nan=True)
    assert np.allclose(el["o_midcal3_no"].fill_null(np.nan).to_numpy(), n, equal_nan=True)
    assert all(o.filter(~pl.col("eligible"))[c].null_count() == o.filter(~pl.col("eligible")).height
               for c in o.columns if c.startswith("o_"))
    # depth read after the model ran: a bid that would cross that book is dropped (post-only)
    first = el["ticker"][0]
    yes0 = el["o_blend3_yes"][0]
    late = out.with_columns(pl.lit(datetime(2026, 11, 5, 23, 37, tzinfo=UTC)).alias("book_ts"),
                            pl.col("bid").alias("book_bid"),
                            pl.when(pl.col("ticker") == first).then(pl.lit(yes0)).otherwise(pl.col("ask")).alias("book_ask"))
    o2 = F.fcs_orders(late).filter("eligible")
    assert o2.filter(pl.col("ticker") == first)["o_blend3_yes"][0] is None
    assert o2.filter(pl.col("ticker") == first)["o_blend3_no"][0] == el["o_blend3_no"][0]
    assert o2.filter(pl.col("ticker") != first)["o_blend3_yes"].to_list() == el.filter(pl.col("ticker") != first)["o_blend3_yes"].to_list()


def test_ncaaf_snapshot_records_raw_responses_under_the_forward_cache(tmp_path):
    s = Settings(data_dir=tmp_path)
    view = F._OwnRaw(s)
    assert view.raw_dir == tmp_path / "cache" / "forward" / "raw" / "ncaaf"
    assert view.data_dir == s.data_dir and view.kalshi_base == s.kalshi_base


def test_speech_groups_follow_the_study():
    assert F.speech_group("KXTRUMPMENTION") == "trump" and F.speech_group("KXTRUMPMENTIONB") == "trump"
    assert F.speech_group("KXVANCEMENTION") == "vance" and F.speech_group("KXEARNINGSMENTIONDKNG") == "earnings"
    assert F.speech_group("KXLASTWORDMENTION") == "other"
    assert M.speaker_of("KXLASTWORDMENTION") == "lastword"
