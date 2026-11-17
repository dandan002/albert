"""albert record-vol: family classification, window selection and connection assignment, on
pages of open prod markets recorded by ``albert capture --only vol``."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from albert.research.volatile import GAME_SERIES
from albert.stream import VOL_LEAGUES, VolUniverse, select_kalshi, select_vol, vol_family

FIX = Path(__file__).parent / "fixtures" / "kalshi"
VOL = json.loads((FIX / "vol_markets.json").read_text())
SERIES = VOL["_series_trimmed"]
RECORDED = datetime.fromisoformat(VOL["_recorded"])


def _transport(requested: list[str]):
    def handler(req: httpx.Request) -> httpx.Response:
        st = req.url.params["series_ticker"]
        requested.append(st)
        if req.url.params.get("cursor"):          # second page: empty, ends the listing
            return httpx.Response(200, json={"markets": [], "cursor": ""})
        return httpx.Response(200, json=VOL["pages"].get(st, {"markets": [], "cursor": ""}))
    return httpx.MockTransport(handler)


def _exp(m) -> datetime:
    return datetime.fromisoformat(m["expected_expiration_time"])


def _want(hours: float, families=None) -> set[str]:
    hi = RECORDED + timedelta(hours=hours)
    return {m["ticker"] for st, page in VOL["pages"].items() for m in page["markets"]
            if RECORDED <= _exp(m) <= hi
            and (families is None or vol_family(st, SERIES[st]["category"]) in families)}


def test_vol_family_on_recorded_series_rows():
    fam = {st: vol_family(st, row["category"]) for st, row in SERIES.items()}
    assert fam == {"KXNBAGAME": "game", "KXEPLGAME": "game", "KXNBA1H": "half", "KXNHL1P": "half",
                   "KXNBATOTAL": "ladder", "KXNBATEAMTOTAL": "ladder", "KXHIGHNY": "weather",
                   # look-alikes: an inflation and an election series share the prefix; quarter ladders
                   # are not in the study
                   "KXHIGHINFLATION": None, "KXHIGHMOVKH": None, "KXNBA3QSPREAD": None}


def test_every_vol_league_has_an_in_play_duration_for_the_hf_study():
    assert all(f"KX{lg}GAME" in GAME_SERIES for lg in VOL_LEAGUES)


def test_select_vol_keeps_whole_series_inside_the_expiration_window():
    for hours in (36, 72):
        requested: list[str] = []
        got = select_vol(SERIES, hours, now=RECORDED, pause=0, transport=_transport(requested))
        assert {r["ticker"] for r in got} == _want(hours) and got
        # only vol series are listed, each paged to the end
        assert set(requested) == {st for st, row in SERIES.items() if vol_family(st, row["category"])}
        assert all(r["family"] == vol_family(r["series_ticker"], r["category"]) for r in got)
    assert _want(36) < _want(72)   # the window matters on this page


def test_select_vol_family_filter():
    got = select_vol(SERIES, 72, families=("weather",), now=RECORDED, pause=0, transport=_transport([]))
    assert {r["ticker"] for r in got} == _want(72, ("weather",)) and got
    assert {r["family"] for r in got} == {"weather"}


def test_universe_never_subscribes_a_market_twice_and_conflates_only_ladders():
    rows = select_vol(SERIES, 72, now=RECORDED, pause=0, transport=_transport([]))
    uni = VolUniverse(chunk=4, ladder_interval_ms=100)
    first = uni.assign(rows[:7], RECORDED)
    again = uni.assign(rows, RECORDED + timedelta(hours=1))   # a refresh returns old + new rows
    conns = first + again
    tickers = [t for _, ts, _ in conns for t in ts]
    assert sorted(tickers) == sorted(r["ticker"] for r in rows) and len(set(tickers)) == len(tickers)
    fam = {r["ticker"]: r["family"] for r in rows}
    for conn, ts, interval in conns:
        assert len(ts) <= 4
        assert (interval == 100) == conn.startswith("c") == all(fam[t] == "ladder" for t in ts)
    assert len({c for c, _, _ in conns}) == len(conns) == len(uni.conns)
    assert {m["ticker"]: m["conn"] for m in uni.markets} == {t: c for c, ts, _ in conns for t in ts}
    # without conflation everything is real-time
    plain = VolUniverse(chunk=100).assign(rows, RECORDED)
    assert all(c.startswith("v") and i == 0 for c, _, i in plain)


def test_main_recorder_skip_vol_drops_vol_series_only():
    page = json.loads((FIX / "open_markets_7d.json").read_text())
    series = page["_series_trimmed"]
    transport = httpx.MockTransport(lambda req: httpx.Response(
        200, json={k: v for k, v in page.items() if not k.startswith("_")}))
    every = select_kalshi(series, 1000, transport=transport)
    kept = select_kalshi(series, 1000, skip_vol=True, transport=transport)
    dropped = {m["ticker"] for m in every} - {m["ticker"] for m in kept}
    assert dropped and all(m["ticker"].startswith("KXNHLGAME-") for m in every if m["ticker"] in dropped)
    assert not any(vol_family(m["series_ticker"], m["category"]) for m in kept)
    assert kept


def test_record_vol_subscribes_refreshed_markets_on_new_connections(tmp_path):
    import asyncio

    from albert.stream import Recorder, record_vol

    rows = select_vol(SERIES, 72, now=RECORDED, pause=0, transport=_transport([]))
    calls: list[tuple[str, list[str], int]] = []

    class Rec(Recorder):
        async def kalshi_conn(self, conn, tickers, update_interval_ms=0):
            calls.append((conn, tickers, update_interval_ms))
            await self.stop.wait()

    rec = Rec(tmp_path, None)
    uni = VolUniverse(chunk=100, ladder_interval_ms=100)
    # every refresh lists the whole window: the first adds the rest, later ones add nothing
    asyncio.run(record_vol(rec, uni, minutes=0.7 / 60, initial=rows[:5], select=lambda: rows,
                           refresh_every=0.2, check_every=60, check_sample=0, min_free_gb=0))
    subscribed = [t for _, ts, _ in calls for t in ts]
    assert sorted(subscribed) == sorted(r["ticker"] for r in rows)
    added = [r["added"] for r in uni.refreshes]
    assert len(added) >= 2 and added[0] == len(rows) - 5 and not any(added[1:])
    assert len(calls) == len(uni.conns) > 1
