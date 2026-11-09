"""select_kalshi on a recorded page of open prod markets (closing within 7 days)."""

import json
from pathlib import Path

import httpx

from albert.stream import select_kalshi

PAGE = json.loads((Path(__file__).parent / "fixtures" / "kalshi" / "open_markets_7d.json").read_text())
SERIES = PAGE["_series_trimmed"]


def _transport():
    return httpx.MockTransport(lambda req: httpx.Response(200, json={k: v for k, v in PAGE.items() if not k.startswith("_")}))


def _f(m, k):
    return float(m.get(k) or 0)


def _expected(lo_spread, hi_spread, vol):
    out = set()
    for m in PAGE["markets"]:
        b, a = _f(m, "yes_bid_dollars"), _f(m, "yes_ask_dollars")
        if b > 0 and a > 0 and lo_spread - 1e-9 <= a - b <= hi_spread + 1e-9 and a > b \
                and (vol is None or vol[0] <= _f(m, "volume_24h_fp") <= vol[1]):
            out.add(m["ticker"])
    return out


def test_default_universe_is_every_two_sided_market():
    got = select_kalshi(SERIES, 1000, transport=_transport())
    assert {m["ticker"] for m in got} == _expected(0.0, 0.25, None) and len(got) > 10


def test_illiquid_universe_filters_on_trailing_volume_and_spread():
    got = select_kalshi(SERIES, 1000, min_spread=0.05, max_spread=0.40, vol24h=(1, 100), transport=_transport())
    want = _expected(0.05, 0.40, (1, 100))
    assert {m["ticker"] for m in got} == want and want
    assert all(1 <= m["volume_24h"] <= 100 and m["yes_ask"] - m["yes_bid"] >= 0.05 - 1e-9 for m in got)
    # the liquid side of the same page is excluded
    busy = {m["ticker"] for m in PAGE["markets"] if _f(m, "volume_24h_fp") > 100}
    assert busy and not busy & {m["ticker"] for m in got}
