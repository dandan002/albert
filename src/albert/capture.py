"""Record small *real* API responses into tests/fixtures/<venue>/ for parser tests.

Fixtures are trimmed copies of live responses (lists cut to a few items), never hand-written.
Re-run ``albert capture`` whenever a venue changes its schema, then fix the parsers until
``pytest`` passes again.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from albert.config import Settings
from albert.ingest import Clients


def _trim(payload: Any, n: int) -> Any:
    if isinstance(payload, list):
        return [_trim(x, n) for x in payload[:n]]
    if isinstance(payload, dict):
        return {k: (_trim(v, n) if isinstance(v, list | dict) else v) for k, v in payload.items()}
    return payload


def _write(root: Path, venue: str, name: str, payload: Any, n: int = 3) -> Path:
    path = root / venue / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_trim(payload, n), indent=1, sort_keys=True) + "\n")
    return path


def _write_text(root: Path, venue: str, name: str, text: str, lines: int = 40) -> Path:
    path = root / venue / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text.splitlines()[:lines]) + "\n")
    return path


def capture_weather(s: Settings, c: Clients, root: Path) -> list[Path]:
    """IEM (CLI, ASOS, NBM) samples for the weather parsers."""
    now = datetime.now(UTC)
    day = (now - timedelta(days=3)).date()
    out = [
        _write(root, "iem", "cli", c.iem.cli("NYC", day.year, cache=False).payload, 5),
        _write_text(root, "iem", "asos.csv", c.iem.asos("NYC", day, day + timedelta(days=1), cache=False).payload),
        _write_text(root, "iem", "nbm.csv", c.iem.nbm(
            "NYC", now - timedelta(days=2), now - timedelta(days=2) + timedelta(hours=7), cache=False).payload, 60),
    ]
    return out


def capture(s: Settings, c: Clients, root: Path) -> list[Path]:
    out: list[Path] = []
    now = datetime.now(UTC)
    kc = c.kalshi.client
    settled = kc.get("/markets", {"status": "settled", "mve_filter": "exclude", "limit": 20,
                                  "series_ticker": "KXNBAGAME"}, cache=False).payload
    out.append(_write(root, "kalshi", "settled_markets", settled, 4))
    ticker = settled["markets"][0]["ticker"]
    out.append(_write(root, "kalshi", "trades", kc.get("/markets/trades", {"ticker": ticker, "limit": 5},
                                                       cache=False).payload))
    out.append(_write(root, "kalshi", "candles_live", kc.get(
        f"/series/KXNBAGAME/markets/{ticker}/candlesticks",
        {"start_ts": int((now - timedelta(days=20)).timestamp()), "end_ts": int(now.timestamp()),
         "period_interval": 60}, cache=False).payload, 4))
    hist = kc.get("/historical/markets", {"limit": 3, "series_ticker": "KXNBAGAME"}, cache=False).payload
    out.append(_write(root, "kalshi", "historical_markets", hist))
    hticker = hist["markets"][0]["ticker"]
    out.append(_write(root, "kalshi", "candles_historical", kc.get(
        f"/historical/markets/{hticker}/candlesticks",
        {"start_ts": 1700000000, "end_ts": int(now.timestamp()), "period_interval": 1440},
        cache=False).payload, 4))
    out.append(_write(root, "kalshi", "candles_batch", kc.get("/markets/candlesticks", {
        "market_tickers": ",".join(m["ticker"] for m in settled["markets"][:2]),
        "start_ts": int((now - timedelta(days=20)).timestamp()), "end_ts": int(now.timestamp()),
        "period_interval": 60}, cache=False).payload, 4))
    out.append(_write(root, "kalshi", "event", kc.get(
        f"/events/{settled['markets'][0]['event_ticker']}", cache=False).payload))
    out.append(_write(root, "kalshi", "cutoff", kc.get("/historical/cutoff", cache=False).payload))
    open_m = kc.get("/markets", {"status": "open", "mve_filter": "exclude", "limit": 50,
                                 "series_ticker": "KXNBAGAME"}, cache=False).payload
    if open_m.get("markets"):
        out.append(_write(root, "kalshi", "orderbook", kc.get(
            f"/markets/{open_m['markets'][0]['ticker']}/orderbook", {"depth": 5}, cache=False).payload, 5))
    series = kc.get("/series", cache=False).payload
    series["series"] = [x for x in series["series"] if x.get("ticker") in
                        ("KXNBAGAME", "KXBTC15M", "KXNFLGAME", "FEMA")]
    out.append(_write(root, "kalshi", "series", series, 10))

    g, d = c.pm.gamma, c.pm.data
    events = g.get("/events/keyset", {"closed": "true", "limit": 3, "tag_slug": "nba"}, cache=False).payload
    out.append(_write(root, "polymarket", "events_closed", events, 3))
    trades = d.get("/trades", {"limit": 3}, cache=False).payload
    out.append(_write(root, "polymarket", "trades", trades))
    out.append(_write(root, "polymarket", "leaderboard", d.get(
        "/leaderboard", {"limit": 3, "time_period": "all"}, cache=False).payload))
    token = trades["data"][0]["token_id"]
    out.append(_write(root, "polymarket", "price_history", d.get("/prices-history", {
        "token_id": token, "start": int((now - timedelta(days=3)).timestamp()),
        "end": int(now.timestamp()), "bucket_seconds": 3600}, cache=False).payload, 5))

    u = c.pmus.client
    closed = u.get("/markets", {"limit": 5, "closed": "true", "orderBy": "id",
                                "orderDirection": "desc"}, cache=False).payload
    out.append(_write(root, "pmus", "markets_closed", closed, 5))
    open_us = u.get("/markets", {"limit": 3, "closed": "false", "active": "true"}, cache=False).payload
    out.append(_write(root, "pmus", "markets_open", open_us))
    slug = open_us["markets"][0]["slug"]
    out.append(_write(root, "pmus", "book", u.get(f"/markets/{slug}/book", cache=False).payload, 5))
    out.append(_write(root, "pmus", "bbo", u.get(f"/markets/{slug}/bbo", cache=False).payload))
    out.append(_write(root, "pmus", "events", u.get("/events", {"limit": 2, "closed": "true"},
                                                    cache=False).payload, 2))
    return out


VOL_SERIES = ("KXNBAGAME", "KXNBA1H", "KXNBATOTAL", "KXNHL1P", "KXHIGHNY")
VOL_DECOYS = ("KXHIGHINFLATION", "KXHIGHMOVKH", "KXNBA3QSPREAD", "KXNBATEAMTOTAL", "KXEPLGAME")


def capture_vol(s: Settings, c: Clients, root: Path) -> list[Path]:
    """Open-market pages of one series per vol family (``stream.select_vol``) plus the series
    rows that ``stream.vol_family`` classifies, including look-alikes that are not vol series."""
    from albert.ingest import kalshi_series

    info = kalshi_series(s, c)
    kc = c.kalshi.client
    pages = {st: _trim(kc.get("/markets", {"series_ticker": st, "status": "open", "limit": 1000},
                              cache=False).payload, 6) for st in VOL_SERIES}
    payload = {"_recorded": datetime.now(UTC).isoformat(), "pages": pages,
               "_series_trimmed": {st: info[st] for st in (*VOL_SERIES, *VOL_DECOYS) if st in info}}
    return [_write(root, "kalshi", "vol_markets", payload, 1000)]


def _redact(payload: Any) -> Any:
    """Drop the account's user id from portfolio fixtures (the rest is kept verbatim)."""
    if isinstance(payload, list):
        return [_redact(x) for x in payload]
    if isinstance(payload, dict):
        return {k: ("REDACTED" if k == "user_id" else _redact(v)) for k, v in payload.items()}
    return payload


def capture_portfolio(root: Path, env: str = "prod") -> list[Path]:
    """Read-only Kalshi portfolio endpoints (orders, fills, positions, balance, queue positions)
    for the order-layer parsers. GET requests only: nothing is placed or cancelled."""
    from albert.venues.kalshi_trade import KalshiTrader

    out: list[Path] = []
    with KalshiTrader.sync(env) as t:
        resting = t.get("/portfolio/orders", {"status": "resting", "limit": 5})
        out.append(_write(root, "kalshi", "portfolio_orders_resting", _redact(resting), 3))
        for status in ("canceled", "executed"):
            out.append(_write(root, "kalshi", f"portfolio_orders_{status}", _redact(t.get(
                "/portfolio/orders", {"status": status, "limit": 3})), 3))
        out.append(_write(root, "kalshi", "portfolio_fills", _redact(t.get("/portfolio/fills", {"limit": 3})), 3))
        out.append(_write(root, "kalshi", "portfolio_positions", _redact(t.get("/portfolio/positions")), 3))
        out.append(_write(root, "kalshi", "portfolio_balance", t.get("/portfolio/balance"), 5))
        if resting.get("orders"):
            out.append(_write(root, "kalshi", "portfolio_queue_positions", t.get(
                "/portfolio/orders/queue_positions",
                {"market_tickers": resting["orders"][0]["ticker"]}), 3))
        out.append(_write(root, "kalshi", "account_limits", t.get("/account/limits"), 5))
    return out


def capture_guard(root: Path, env: str = "prod") -> list[Path]:
    """Read-only responses the live guard parses (``albert.live_guard``): one bot order looked up by
    id (fills carry no client_order_id) and markets by ticker (a settled bot market plus an open
    one). GET requests only."""
    from albert.venues.kalshi_trade import KalshiTrader

    out: list[Path] = []
    with KalshiTrader.sync(env) as t:
        bot = [o for o in t.get("/portfolio/orders", {"status": "executed", "limit": 200}).get("orders") or []
               if (o.get("client_order_id") or "").startswith("alb")]
        if not bot:
            raise RuntimeError("no executed bot order to record")
        out.append(_write(root, "kalshi", "portfolio_order_single",
                          _redact(t.get(f"/portfolio/orders/{bot[0]['order_id']}")), 3))
        live = t.get("/markets", {"status": "open", "limit": 1}).get("markets") or []
        tickers = [bot[0]["ticker"], *(m["ticker"] for m in live)]
        out.append(_write(root, "kalshi", "markets_by_tickers",
                          t.get("/markets", {"tickers": ",".join(tickers), "limit": 100}), 3))
    return out


def capture_hl(c: Clients, root: Path) -> list[Path]:
    """Hyperliquid HIP-4 outcome data (public, never traded): ``outcomeMeta`` trimmed to one
    question per question template (3 named outcomes plus the fallback) and one standalone
    outcome per template, then ``l2Book``, ``recentTrades``, ``candleSnapshot`` and a few
    WebSocket frames for one live non-financial outcome that has traded."""
    import asyncio

    from websockets.asyncio.client import connect

    from albert.venues import hyperliquid as hl

    assert c.hl is not None
    meta = c.hl.outcome_meta().payload
    questions, members = [], set()
    for q in meta["questions"]:
        if q["name"] in {x["name"] for x in questions}:
            continue
        q = {**q, "namedOutcomes": q["namedOutcomes"][:3]}
        questions.append(q)
        members |= set(q["namedOutcomes"]) | {q["fallbackOutcome"]}
    in_any = {o for q in meta["questions"] for o in [*q["namedOutcomes"], q["fallbackOutcome"]]}
    standalone: dict[str, dict] = {}
    for o in meta["outcomes"]:
        if o["outcome"] not in in_any:
            standalone.setdefault(o["name"], o)
    trimmed = {
        "outcomes": [o for o in meta["outcomes"] if o["outcome"] in members] + list(standalone.values()),
        "questions": questions,
        "deployers": [{**d, "subDeployers": [[name, addrs[:1]] for name, addrs in d["subDeployers"][:2]]}
                      for d in meta["deployers"][:2]],
        "feeScale": meta["feeScale"],
    }
    paths = [_write(root, "hyperliquid", "outcome_meta", trimmed, n=100)]
    coin = next(x for x in hl.non_financial_coins(meta) if c.hl.recent_trades(x).payload)
    now = datetime.now(UTC)
    paths.append(_write(root, "hyperliquid", "l2_book", c.hl.l2_book(coin).payload))
    paths.append(_write(root, "hyperliquid", "recent_trades", c.hl.recent_trades(coin).payload))
    paths.append(_write(root, "hyperliquid", "candle_snapshot",
                        c.hl.candles(coin, "1h", now - timedelta(days=3), now).payload))

    async def frames() -> list[str]:
        got: dict[str, str] = {}
        async with connect(hl.WS_URL, open_timeout=20, ping_interval=None) as ws:
            for typ in ("l2Book", "trades"):
                await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": typ, "coin": coin}}))
            await ws.send(json.dumps({"method": "ping"}))
            async with asyncio.timeout(60):
                while len(got) < 4:
                    d = json.loads(await ws.recv())
                    got.setdefault(d["channel"], json.dumps(_trim(d, 3), separators=(",", ":")))
        return [got[k] for k in sorted(got)]

    paths.append(_write_text(root, "hyperliquid", "ws_frames.jsonl", "\n".join(asyncio.run(frames()))))
    return paths
