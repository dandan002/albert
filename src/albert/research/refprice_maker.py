"""Reference-anchored pre-game maker: quote Kalshi game markets around the de-vigged Polymarket
international price, with a PULL rule, and ask whether it earns after fees out of sample.

Round 1 (``soccer.py``) found the de-vigged Polymarket 1X2 price equal to Kalshi's (within 1c on
92% of legs), so a taker has nothing to trade. A pre-game *maker* is different: it needs a fair
value it can trust while it rests, and a signal to step out of the way when the price moves. This
study gives it both from Polymarket (data only; never traded) and measures whether that beats
the same maker anchored to Kalshi's own mid.

Universe (volume-blind: every settled Kalshi game with a Polymarket match):

* **Soccer** 1X2 (25 leagues, three legs per game): Kalshi ``*GAME`` series, matched to
  Polymarket 1X2 events in round 1 (``fundamental_soccer_pm_match.parquet``); kickoff is
  ``expected_expiration_time − 3h`` (checked in round 1: 98.9% within 15 min of Polymarket's
  ``gameStartTime``). Polymarket fair = power de-vig of the three "Yes" prices (chosen on dev in
  round 1; mult/power/Shin are equal to 4 decimals).
* **US moneylines** (NFL, NCAAF, MLB, NBA, NHL, WNBA; two legs per game): every closed
  Polymarket game event of the sport's tag (Gamma ``/events/keyset``), matched to Kalshi events
  by start time and team code / name. Kickoff = Polymarket ``gameStartTime`` (the scheduled
  start, public in advance). The two outcome tokens of one Polymarket market are complementary,
  so the fair value of a leg is that outcome's price (normalised by the pair's sum when both are
  available).

Windows and data (2026-09-06 → 2026-10-31; dev = Sep 6–30, holdout = October):

* Polymarket keeps 5-minute ``/prices-history`` buckets for ~60 days (every request for a game
  before 2026-09-06 came back empty on 2026-11-05), so August is out of reach at this
  resolution and the dev period starts on Sep 6.
* **Soccer:** post at kickoff − 24h and − 6h (truncated to the hour), rest until kickoff; every
  Kalshi print of each leg over that window fetched here (``fetch_trades``, cached).
* **US:** the Kalshi API is shared by many agents (~20 trade requests/min for this study), so
  US games use the recorded pre-game trade table (``kalshi_trades`` / ``kalshi_trade_windows``:
  every print of one leg per game from scheduled start − 22h to start − 1h). Long lead = the
  first full hour of that window (≈ 21–22h before kickoff), short lead = kickoff − 6h; orders
  rest until the window ends (≈ 1h before kickoff).

As-of rules (AGENTS.md rule 4):

* Polymarket buckets (300 s) are used only once they have *ended* (bucket start + 300 s ≤ t),
  so every fair value and every pull decision uses data timestamped before the moment it is
  applied. The bucket that starts at kickoff is the cleared book (price 0) and is never used.
* Decision times are on the hour, so the Kalshi hourly candle that closes at ``t`` is the book
  at ``t`` (bid/ask for the post-only cap, the Kalshi-mid baseline and the taker test).
* Fills: from ``t`` to ``end``, trade-through — a later Kalshi print strictly through our bid
  (YES bid b: a print below b; NO bid n: a YES print above 1 − n), whichever side took it,
  since such a print means an order that would have crossed our resting one-lot. Soccer reads
  prints from 1-minute candles (the minute's low/high, stamped at the minute's end; the batch
  endpoint makes this ~8× cheaper than per-leg trade pages); US from the recorded prints. The
  harness's stricter taker-side rule is reported for US as a variant. Orders are held to
  settlement; maker fee by series; markout = the Polymarket fair at expiry minus our price
  minus the fee (same expectation as settlement PnL if that fair is calibrated, far less noise).
* PULL rule: cancel both bids at the first moment ``τ`` when the Polymarket fair (from buckets
  ended by ``τ``) has moved ≥ k from its value at posting; a Kalshi trade at ``s > τ`` cannot
  fill us. The same pull is applied to the Kalshi-mid maker (Polymarket as a pure "get out of
  the way" signal), which separates the reference's two roles: fair value and pull signal.
  A third variant pulls AND re-posts around the new fair (:func:`simulate_requote`).
"""

import math
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from albert.research import fundamental as F

NAME = "refprice_maker"
HOLDOUT = datetime(2026, 10, 1, tzinfo=UTC)
LEADS = (24, 6)                          # hours before kickoff (soccer; US: see us_decisions)
LEAD_CLASSES = ("long", "short")
WINDOW = timedelta(hours=27)             # history / candle window before kickoff
PM_BUCKET = 300
PM_RANGE = (datetime(2026, 8, 14, tzinfo=UTC), datetime(2026, 11, 3, tzinfo=UTC))
# Polymarket keeps 5-minute price buckets for ~60 days: on 2026-11-05 every history ending before
# 2026-09-06 came back empty at 300 s and 900 s (hourly buckets still exist). The study therefore
# starts at 2026-09-06: dev = Sep 6–30, holdout = October.
PM_FIVE_MIN_FROM = datetime(2026, 9, 6, tzinfo=UTC)
US = {"KXNFLGAME": "nfl", "KXNCAAFGAME": "cfb", "KXMLBGAME": "mlb", "KXNBAGAME": "nba",
      "KXNHLGAME": "nhl", "KXWNBAGAME": "wnba"}
MARGINS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08)
PULLS = (None, 0.01, 0.02, 0.03, 0.05)   # None = never pull
BANDS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))


def cache_dir(s):
    return s.data_dir / "cache" / NAME


def _write(df: pl.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


# ---- Polymarket intl US game events (data only) -----------------------------------------------
_PM_US_RX = re.compile(r"^(?P<lg>[a-z0-9]+)-(?P<a>[a-z0-9]+)-(?P<b>[a-z0-9]+)-(?P<d>\d{4}-\d{2}-\d{2})(?:-(?P<n>\d))?$")


def pm_us_game_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Moneyline market of each two-team game event: outcome names, tokens, team names and
    codes in slug order (outcome 0 = first code of the slug), and the scheduled start."""
    from albert.venues.polymarket import _json_list, _parse_loose

    rows = []
    for e in events:
        slug = e.get("slug") or ""
        g = _PM_US_RX.match(slug)
        if not g:
            continue
        ml = [m for m in e.get("markets") or [] if m.get("slug") == slug]
        if len(ml) != 1:
            continue
        m = ml[0]
        outs = [str(o) for o in _json_list(m.get("outcomes"))]
        toks = [str(t) for t in _json_list(m.get("clobTokenIds"))]
        start = _parse_loose(m.get("gameStartTime")) or _parse_loose(e.get("startTime"))
        if len(outs) != 2 or len(toks) != 2 or start is None:
            continue
        teams = {(t.get("abbreviation") or "").lower(): t for t in e.get("teams") or []}
        alias = {(t.get("alias") or "").lower(): t for t in e.get("teams") or []}
        names = []
        for code, out in ((g["a"], outs[0]), (g["b"], outs[1])):
            t = teams.get(code) or alias.get(out.lower()) or {}
            names.append(t.get("name") or out)
        rows.append({"pm_slug": slug, "league": g["lg"], "pm_start": start, "pm_code0": g["a"], "pm_code1": g["b"],
                     "pm_out0": outs[0], "pm_out1": outs[1], "pm_name0": names[0], "pm_name1": names[1],
                     "pm_tok0": toks[0], "pm_tok1": toks[1], "pm_volume": float(m.get("volumeNum") or 0)})
    return rows


def pm_us_games(s, leagues: tuple[str, ...] = tuple(US.values())) -> pl.DataFrame:
    """Every closed Polymarket game event of each league tag that ended in ``PM_RANGE``."""
    from albert.http import CachedClient

    path = s.data_dir / "cache" / f"fundamental_{NAME}_pm_games.parquet"
    if path.exists():
        return pl.read_parquet(path)
    gamma = CachedClient("pm_gamma", s.pm_gamma_base, cache_dir(s) / "raw", min_interval=0.5)
    rows: list[dict[str, Any]] = []
    for lg in leagues:
        cursor, pages = None, 0
        while True:
            page = gamma.get("/events/keyset", {
                "closed": "true", "limit": 100, "tag_slug": lg, "after_cursor": cursor,
                "end_date_min": PM_RANGE[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_date_max": PM_RANGE[1].strftime("%Y-%m-%dT%H:%M:%SZ")})
            evs = page.payload.get("events") or []
            rows += [r for r in pm_us_game_rows(evs) if r["league"] == lg]
            cursor, pages = page.payload.get("next_cursor"), pages + 1
            if not cursor or not evs:
                break
        print(f"  pm {lg}: {pages} pages, {sum(r['league'] == lg for r in rows)} games", flush=True)
    out = pl.DataFrame(rows).unique("pm_slug") if rows else pl.DataFrame()
    _write(out, path)
    return out


# ---- Kalshi US games and the match -------------------------------------------------------------
_MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
_K_US_RX = re.compile(r"^(?P<series>KX[A-Z]+)-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?P<hhmm>\d{4})?(?P<teams>[A-Z0-9]+)$")


def kalshi_us_games(markets: pl.DataFrame) -> pl.DataFrame:
    """One row per settled two-leg Kalshi game: both legs (ticker, team code, team name, y)."""
    rows = []
    for (ev,), g in markets.group_by(["event_ticker"]):
        m = _K_US_RX.match(ev or "")
        if not m or g.height != 2 or m["series"] not in US or g["result"].to_list().count("yes") != 1:
            continue
        legs = g.sort("ticker").to_dicts()
        rows.append({"event": ev, "series": m["series"], "league": US[m["series"]],
                     "kdate": date(2000 + int(m["yy"]), _MON[m["mon"]], int(m["dd"])),
                     "occurrence_time": legs[0]["occurrence_time"], "open_time": min(x["open_time"] for x in legs),
                     "close_time": max(x["close_time"] for x in legs),
                     **{f"k_{k}{i}": v for i, x in enumerate(legs) for k, v in (
                         ("ticker", x["ticker"]), ("code", x["ticker"].rsplit("-", 1)[1].lower()),
                         ("name", x["yes_sub_title"] or ""), ("y", int(x["result"] == "yes")))}})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _clean(name: str) -> str:
    """Kalshi abbreviations to Polymarket spellings: trailing "St." = State, leading = Saint."""
    name = re.sub(r"\bSt\.?$", "State", (name or "").strip())
    return re.sub(r"^St\.?\s", "Saint ", name)


def leg_score(code: str, name: str, pm_code: str, pm_names: tuple[str, ...]) -> float:
    """1.0 when the team codes agree, else the best club-name similarity (round 1's soccer
    ``name_score``) against any Polymarket spelling of the team (outcome label, team name)."""
    from albert.research.soccer import name_score

    if code == pm_code:
        return 1.0
    return max(0.0, *(name_score(_clean(name), x) for x in pm_names if x))


def match_us(kg: pl.DataFrame, pm: pl.DataFrame, *, threshold: float = 0.75) -> pl.DataFrame:
    """Kalshi game -> Polymarket game of the same league whose slug date is within a day of the
    Kalshi ticker date, with both teams agreeing (code or name, :func:`leg_score`). When one team
    matches exactly and the other weakly, the game is still accepted (one team cannot play twice
    at the same time; doubleheaders are split by the start time). Returns one row per Kalshi
    leg with the Polymarket outcome index it backs."""
    by: dict[tuple[str, date], list[dict[str, Any]]] = defaultdict(list)
    for r in pm.iter_rows(named=True):
        d = date.fromisoformat(r["pm_slug"].split("-", 3)[3][:10])
        by[(r["league"], d)].append(r)
    out = []
    for g in kg.iter_rows(named=True):
        best = None
        for dd in (0, -1, 1):
            for r in by.get((g["league"], g["kdate"] + timedelta(days=dd)), []):
                for swap in (False, True):
                    i0, i1 = (1, 0) if swap else (0, 1)
                    s0 = leg_score(g["k_code0"], g["k_name0"], r[f"pm_code{i0}"], (r[f"pm_out{i0}"], r[f"pm_name{i0}"]))
                    s1 = leg_score(g["k_code1"], g["k_name1"], r[f"pm_code{i1}"], (r[f"pm_out{i1}"], r[f"pm_name{i1}"]))
                    # Kalshi's occurrence time is the scheduled start + 3h for every US league
                    gap = abs((r["pm_start"] - (g["occurrence_time"] - timedelta(hours=3))).total_seconds()) \
                        if g["occurrence_time"] is not None else 1e9
                    sc = min(s0, s1)
                    if sc < threshold and max(s0, s1) >= 0.99 and gap <= 1800:
                        sc = threshold   # one team agrees exactly and the start time agrees: same game
                    if sc < threshold:
                        continue
                    key = (sc, -abs(dd), -gap)
                    if best is None or key > best[0]:
                        best = (key, r, swap)
        if best is None:
            continue
        (sc, _, _), r, swap = best
        for leg in (0, 1):
            idx = (1 - leg) if swap else leg
            out.append({"event": g["event"], "ticker": g[f"k_ticker{leg}"], "league": g["league"], "series": g["series"],
                        "y": g[f"k_y{leg}"], "kickoff": r["pm_start"], "pm_slug": r["pm_slug"], "pm_outcome": idx,
                        "pm_tok": r[f"pm_tok{idx}"], "pm_tok_other": r[f"pm_tok{1 - idx}"], "match_score": sc,
                        "k_name": g[f"k_name{leg}"], "pm_name": r[f"pm_name{idx}"],
                        "occurrence_time": g["occurrence_time"], "open_time": g["open_time"], "close_time": g["close_time"]})
    m = pl.DataFrame(out)
    if not m.height:
        return m
    # one Polymarket game per Kalshi game and vice versa
    best_ev = (m.group_by("event", "pm_slug").agg(pl.col("match_score").first()).sort("match_score", descending=True)
               .unique("pm_slug", keep="first").unique("event", keep="first"))
    return m.join(best_ev.select("event", "pm_slug"), on=["event", "pm_slug"])


# ---- Polymarket price history ----------------------------------------------------------------
def fetch_pm_history(s, jobs: pl.DataFrame) -> pl.DataFrame:
    """``/prices-history`` (300 s buckets) of each token over [kickoff − ``WINDOW``, kickoff],
    cached in ``data/cache/fundamental_refprice_maker_pm_history.parquet``. ``jobs`` needs
    token_id and kickoff. Two workers, each throttled to one request per second."""
    from albert.http import CachedClient
    from albert.venues.polymarket import price_history_rows

    path = s.data_dir / "cache" / f"fundamental_{NAME}_pm_history.parquet"
    have = pl.read_parquet(path) if path.exists() else None
    done_path = path.with_name(path.stem + "_tokens.parquet")
    done = set(pl.read_parquet(done_path)["token_id"]) if done_path.exists() else set()
    todo = [r for r in jobs.select("token_id", "kickoff").unique("token_id").iter_rows(named=True) if r["token_id"] not in done]
    print(f"pm history: {len(todo)} tokens to fetch", flush=True)
    if not todo:
        return have if have is not None else pl.DataFrame()
    clients = [CachedClient("pm_data", s.pm_data_base, cache_dir(s) / "raw", min_interval=1.0) for _ in range(2)]
    local, lock, counter = threading.local(), threading.Lock(), iter(range(2))

    def one(r: dict[str, Any]) -> list[dict[str, Any]]:
        if not hasattr(local, "client"):
            with lock:
                local.client = clients[next(counter)]
        lo, hi = r["kickoff"] - WINDOW, r["kickoff"]
        try:
            page = local.client.get("/prices-history", {"token_id": r["token_id"], "start": int(lo.timestamp()),
                                                        "end": int(hi.timestamp()), "bucket_seconds": PM_BUCKET})
        except RuntimeError as error:
            print(f"skip pm {r['token_id'][:12]}: {str(error)[:160]}", flush=True)
            return []
        return price_history_rows(page.payload, r["token_id"])

    rows: list[dict[str, Any]] = []
    schema = {"token_id": pl.String, "ts": pl.Datetime("us", "UTC"), "price": pl.Float64}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, got in enumerate(pool.map(one, todo), 1):
            rows += got
            if i % 300 == 0:
                print(f"  pm tokens {i}/{len(todo)}", flush=True)
    new = pl.DataFrame(rows, schema=schema) if rows else None
    frames = [f for f in (have, new) if f is not None and f.height]
    out = pl.concat(frames).unique(["token_id", "ts"]) if frames else pl.DataFrame(schema=schema)
    _write(out, path)
    _write(pl.DataFrame({"token_id": sorted(done | {r["token_id"] for r in todo})}), done_path)
    return out


# ---- Kalshi hourly candles over the pre-game window ------------------------------------------
def fetch_candles(s, legs: pl.DataFrame, *, workers: int = 2, max_tickers: int = 99) -> pl.DataFrame:
    """Hourly candles of each leg over [kickoff − ``WINDOW``, kickoff], cached in
    ``data/cache/fundamental_refprice_maker_candles.parquet`` (harness schema). Legs settled
    after the historical cutoff go through the batch endpoint (games kicking off the same UTC
    day share a request); older ones one market at a time. ``legs`` needs ticker, kickoff,
    close_time."""
    import time

    from albert.http import CachedClient
    from albert.venues import kalshi as k

    path = s.data_dir / "cache" / f"fundamental_{NAME}_candles.parquet"
    have = pl.read_parquet(path) if path.exists() else None
    done = set(have["ticker"].unique()) if have is not None and have.height else set()
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.1))
    cutoff = api.cutoff().get("market_settled_ts")
    jobs: list[tuple[str, list[str], datetime, datetime]] = []
    live: dict[date, list[tuple[str, datetime, datetime]]] = defaultdict(list)
    for r in legs.unique("ticker").iter_rows(named=True):
        if r["ticker"] in done or r["kickoff"] is None:
            continue
        lo, hi = r["kickoff"] - WINDOW, r["kickoff"]
        if cutoff and r["close_time"] < cutoff:
            jobs.append(("hist", [r["ticker"]], lo, hi))
        else:
            live[r["kickoff"].date()].append((r["ticker"], lo, hi))
    for items in live.values():
        for i in range(0, len(items), max_tickers):
            chunk = items[i:i + max_tickers]
            jobs.append(("live", [x[0] for x in chunk], min(x[1] for x in chunk), max(x[2] for x in chunk)))

    def one(job: tuple[str, list[str], datetime, datetime]) -> list[dict[str, Any]]:
        kind, tickers, lo, hi = job
        for attempt in range(6):
            try:
                if kind == "hist":
                    return k.candle_rows(api.candles(tickers[0], lo, hi, 60, True).payload, tickers[0], 60)
                page = api.candles_batch(tickers, lo, hi, 60)
                return [r for mk in page.payload.get("markets") or [] for r in k.candle_rows(mk, mk.get("market_ticker"), 60)]
            except RuntimeError as error:
                if "429" in str(error):
                    time.sleep(5 * (attempt + 1))
                    continue
                print(f"skip candles {tickers[0]} (+{len(tickers) - 1}): {str(error)[:160]}", flush=True)
                return []
        return []

    jobs.sort(key=lambda j: j[0] == "hist")
    print(f"candle jobs: {len(jobs)} ({sum(j[0] == 'hist' for j in jobs)} historical)", flush=True)
    rows: list[dict[str, Any]] = []

    def save() -> pl.DataFrame:
        new = pl.DataFrame(rows, schema=k.CANDLE_SCHEMA) if rows else None
        frames = [f for f in (have, new) if f is not None and f.height]
        out = pl.concat(frames, how="diagonal_relaxed").unique(["ticker", "end_ts", "period_minutes"]) if frames \
            else pl.DataFrame(schema=k.CANDLE_SCHEMA)
        _write(out, path)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, got in enumerate(pool.map(one, jobs), 1):
            rows.extend(got)
            if i % 200 == 0:
                print(f"  candle jobs {i}/{len(jobs)}", flush=True)
                save()
    return save()


# ---- Kalshi 1-minute candles (soccer prints, at minute resolution) ---------------------------
MINUTE_SCHEMA = {"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "price_low": pl.Float64,
                 "price_high": pl.Float64, "volume": pl.Float64}


def minute_rows(payload: dict[str, Any], ticker: str) -> list[dict[str, Any]]:
    """Minutes with at least one print: lowest and highest traded YES price of the minute."""
    from albert.venues.kalshi import _field, num, parse_ts

    rows = []
    for c in payload.get("candlesticks") or []:
        price = c.get("price") or {}
        lo, hi = num(_field(price, "low")), num(_field(price, "high"))
        if lo is None or hi is None:
            continue
        rows.append({"ticker": ticker, "end_ts": parse_ts(c.get("end_period_ts")), "price_low": lo, "price_high": hi,
                     "volume": num(_field(c, "volume"))})
    return rows


def fetch_minute_candles(s, windows: pl.DataFrame, *, per_request: int = 8, workers: int = 2) -> pl.DataFrame:
    """1-minute candles of each leg over its window (ticker, t_from, t_to), from the batch
    endpoint (≤ 10,000 candles per response; legs grouped by kickoff, a response that hits the
    cap is split and re-requested). Cached in ``fundamental_refprice_maker_minute.parquet`` with
    a ``_windows`` file of completed windows. Live (post-cutoff) markets only."""
    import time

    from albert.http import CachedClient
    from albert.venues import kalshi as k

    path = s.data_dir / "cache" / f"fundamental_{NAME}_minute.parquet"
    done_path = path.with_name(path.stem + "_windows.parquet")
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=MINUTE_SCHEMA)
    done = pl.read_parquet(done_path) if done_path.exists() else windows.clear().select("ticker", "t_from", "t_to")
    todo = windows.select("ticker", "t_from", "t_to").unique().join(done, on=["ticker", "t_from", "t_to"], how="anti").sort("t_to")
    if not todo.height:
        return have
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.1))
    # the endpoint refuses a request whose tickers × minutes in [start, end] exceed 10,000
    def span(rs: list[dict[str, Any]]) -> float:
        return (max(r["t_to"] for r in rs) - min(r["t_from"] for r in rs)).total_seconds() / 60 + 1

    chunks, cur = [], []
    for r in todo.iter_rows(named=True):
        if cur and (len(cur) >= per_request or (len(cur) + 1) * span(cur + [r]) > 9900):
            chunks.append(cur)
            cur = []
        cur.append(r)
    if cur:
        chunks.append(cur)

    def one(chunk: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        lo, hi = min(r["t_from"] for r in chunk), max(r["t_to"] for r in chunk)
        for attempt in range(8):
            try:
                page = api.candles_batch([r["ticker"] for r in chunk], lo, hi, 1)
                break
            except RuntimeError as error:
                if "-> 429" in str(error):
                    time.sleep(3 * (attempt + 1))
                    continue
                print(f"skip minute candles {chunk[0]['ticker']} (+{len(chunk) - 1}): {str(error)[-200:]}", flush=True)
                return [], []
        else:
            return [], []
        ms = page.payload.get("markets") or []
        if sum(len(m.get("candlesticks") or []) for m in ms) >= 9990:      # capped: split
            if len(chunk) == 1:
                print(f"capped single leg {chunk[0]['ticker']}", flush=True)
                return [], []
            a, b = one(chunk[: len(chunk) // 2]), one(chunk[len(chunk) // 2:])
            return a[0] + b[0], a[1] + b[1]
        rows = [x for m in ms for x in minute_rows(m, m.get("market_ticker"))]
        win = {r["ticker"]: r for r in chunk}
        rows = [x for x in rows if win[x["ticker"]]["t_from"] < x["end_ts"] <= win[x["ticker"]]["t_to"] + timedelta(minutes=1)]
        return rows, chunk

    print(f"minute candles: {todo.height} legs in {len(chunks)} requests", flush=True)
    rows: list[dict[str, Any]] = []
    fin: list[dict[str, Any]] = []

    def save() -> pl.DataFrame:
        nonlocal have, done
        if rows:
            have = pl.concat([have, pl.DataFrame(rows, schema=MINUTE_SCHEMA)]).unique(["ticker", "end_ts"])
            rows.clear()
        if fin:
            done = pl.concat([done, pl.DataFrame(fin).select("ticker", "t_from", "t_to")], how="vertical_relaxed").unique()
            fin.clear()
        _write(have, path)
        _write(done, done_path)
        return have

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (got, ok) in enumerate(pool.map(one, chunks), 1):
            rows.extend(got)
            fin.extend(ok)
            if i % 25 == 0:
                print(f"  minute requests {i}/{len(chunks)}", flush=True)
                save()
    return save()


def minute_prints(minutes: pl.DataFrame) -> pl.DataFrame:
    """Pseudo-prints from 1-minute candles: the minute's low as a YES-seller print and its high
    as a YES-buyer print, both stamped at the minute's end. With these, the harness rule
    (:func:`first_through`) becomes "some print in that minute traded strictly through our bid"
    — side-agnostic, which is the right counterfactual for a resting one-lot (a print below our
    YES bid, whichever side took it, means an order that would have crossed ours)."""
    lo = minutes.select("ticker", pl.col("end_ts").alias("created_time"), pl.col("price_low").alias("yes_price"),
                        pl.lit("no").alias("taker_side"), pl.lit(1.0).alias("count"))
    hi = minutes.select("ticker", pl.col("end_ts").alias("created_time"), pl.col("price_high").alias("yes_price"),
                        pl.lit("yes").alias("taker_side"), pl.lit(1.0).alias("count"))
    return pl.concat([lo, hi])


def side_agnostic(trades: pl.DataFrame) -> pl.DataFrame:
    """Every print counted for both bids (side-agnostic trade-through), as :func:`minute_prints`."""
    return pl.concat([trades.with_columns(pl.lit("no").alias("taker_side")), trades.with_columns(pl.lit("yes").alias("taker_side"))])


# ---- fair-value paths --------------------------------------------------------------------------
def devig_power(q: np.ndarray, iters: int = 60) -> np.ndarray:
    """Power de-vig, vectorised over rows: ``p_i = q_i^k`` with ``k`` such that each row sums
    to 1 (same method as ``soccer.devig(q, "power")``, which loops in Python)."""
    lo, hi = np.full(len(q), 0.2), np.full(len(q), 5.0)
    for _ in range(iters):
        k = (lo + hi) / 2
        over = (q ** k[:, None]).sum(1) > 1
        lo, hi = np.where(over, k, lo), np.where(over, hi, k)
    p = q ** ((lo + hi) / 2)[:, None]
    return p / p.sum(1, keepdims=True)


def fair_paths(legs: pl.DataFrame, hist: pl.DataFrame, *, devig_method: str = "power") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Polymarket fair value of each Kalshi leg through time: ``{ticker: (avail_ts, fair)}`` with
    ``avail_ts`` = bucket start + ``PM_BUCKET`` (epoch seconds) — the moment the bucket is known.

    ``legs`` has one row per Kalshi leg: event, ticker, ``tokens`` (the game's Polymarket tokens,
    same order for every leg of the game) and ``leg_index`` (which token's outcome the leg backs).
    Each token's last price is carried forward; a bucket counts once every token has a price.
    Three tokens (soccer home/draw/away "Yes"): de-vigged (``devig_method``). Two tokens (the two
    outcomes of one market): normalised by their sum. One token: ``p`` and ``1 − p``."""
    h = hist.filter(pl.col("price") > 0).with_columns((pl.col("ts").dt.epoch("s") + PM_BUCKET).alias("avail"))
    by_tok = {k[0]: (v["avail"].to_numpy(), v["price"].to_numpy()) for k, v in h.sort("avail").partition_by("token_id", as_dict=True).items()}
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (_,), g in legs.group_by(["event"]):
        toks = g["tokens"][0].to_list()
        series = [by_tok.get(t) for t in toks]
        if any(x is None for x in series):
            continue
        grid = np.unique(np.concatenate([x[0] for x in series]))
        q = np.full((len(grid), len(toks)), np.nan)
        for j, (ts, px) in enumerate(series):
            idx = np.searchsorted(ts, grid, side="right") - 1
            ok = idx >= 0
            q[ok, j] = px[idx[ok]]
        keep = ~np.isnan(q).any(1)
        grid, q = grid[keep], q[keep]
        if not len(grid):
            continue
        if q.shape[1] == 3:
            fair = devig_power(q) if devig_method == "power" else q / q.sum(1, keepdims=True)
        elif q.shape[1] == 2:
            fair = q / q.sum(1, keepdims=True)
        else:
            fair = np.column_stack([q[:, 0], 1 - q[:, 0]])
        for r in g.iter_rows(named=True):
            out[r["ticker"]] = (grid, fair[:, r["leg_index"]])
    return out


def fair_at(path: tuple[np.ndarray, np.ndarray], t: float, max_age: float = 7200.0) -> float | None:
    """Fair value from the last bucket that ended at or before ``t`` (epoch s), if fresh."""
    ts, fv = path
    i = int(np.searchsorted(ts, t, side="right")) - 1
    if i < 0 or t - ts[i] > max_age:
        return None
    return float(fv[i])


def pull_time(path: tuple[np.ndarray, np.ndarray], t0: float, t_end: float, p0: float, k: float | None) -> float:
    """First moment in (t0, t_end] at which the fair value (buckets ended by then) is at least
    ``k`` away from ``p0``; ``t_end`` if it never is (or ``k`` is None)."""
    if k is None:
        return t_end
    ts, fv = path
    lo, hi = np.searchsorted(ts, t0, side="right"), np.searchsorted(ts, t_end, side="right")
    moved = np.nonzero(np.abs(fv[lo:hi] - p0) >= k - 1e-9)[0]
    return float(ts[lo + moved[0]]) if len(moved) else t_end


# ---- universe ------------------------------------------------------------------------------------
def us_legs(s) -> pl.DataFrame:
    """Every matched leg of the six US leagues, with fee schedule and Polymarket token(s).

    Kalshi's occurrence time is the scheduled start + 3h in every US league except college
    football (where it is often a placeholder; Polymarket's start agrees with ESPN's kickoff
    within 15 min on 98% of games there). Outside college football, a game whose Polymarket
    start disagrees with Kalshi's by more than 1h is dropped (doubleheaders, postponements:
    the two venues may be describing different games)."""
    mk = F.settled_markets(s, list(US))
    m = match_us(kalshi_us_games(mk), pm_us_games(s))
    gap = ((pl.col("occurrence_time") - timedelta(hours=3) - pl.col("kickoff")).dt.total_minutes().abs())
    m = m.filter((pl.col("league") == "cfb") | (gap <= 60))
    fee = mk.select("ticker", "fee_type", "fee_multiplier")
    return (m.join(fee, on="ticker")
            # outcome 0's token only: the two outcome prices of one market sum to 1 (checked:
            # 11 games, every bucket within ±0.005), so leg fair = p0 or 1 − p0
            .with_columns(pl.when(pl.col("pm_outcome") == 0).then(pl.concat_list("pm_tok")).otherwise(pl.concat_list("pm_tok_other"))
                          .alias("tokens"), pl.col("pm_outcome").alias("leg_index"), pl.lit("us").alias("kind")))


def soccer_legs(s) -> pl.DataFrame:
    """Every leg of round 1's matched soccer games (home / tie / away), Polymarket tokens in
    (home, draw, away) order; kickoff = Kalshi expected expiration − 3h. Games whose Polymarket
    start is more than 15 min earlier are dropped (moved earlier; 4 games since Sep 6)."""
    from albert.research import soccer

    mk = F.settled_markets(s, list(soccer.LEAGUES))
    games = soccer.game_frame(mk)
    matched = pl.read_parquet(s.data_dir / "cache" / "fundamental_soccer_pm_match.parquet")
    g = games.join(matched.select("event", "pm_home", "pm_draw", "pm_away", "pm_start"), on="event")
    # a game Polymarket started (book cleared, then in-play prices) well before Kalshi's
    # estimated kickoff was rescheduled earlier: Kalshi's window would include in-play prints
    g = g.filter(pl.col("pm_start") >= pl.col("kickoff") - timedelta(minutes=15))
    parts = []
    for i, (col, o) in enumerate((("home_ticker", "H"), ("tie_ticker", "D"), ("away_ticker", "A"))):
        parts.append(g.select("event", pl.col(col).alias("ticker"), "league", pl.col("series"), "kickoff", "open_time", "close_time",
                              (pl.col("outcome") == o).cast(pl.Int64).alias("y"),
                              pl.concat_list("pm_home", "pm_draw", "pm_away").alias("tokens"), pl.lit(i).alias("leg_index")))
    out = pl.concat(parts).join(mk.select("ticker", "fee_type", "fee_multiplier"), on="ticker")
    return out.with_columns(pl.lit("soccer").alias("kind"))


# ---- decision rows ------------------------------------------------------------------------------
def soccer_decisions(legs: pl.DataFrame, leads: tuple[int, ...] = LEADS) -> pl.DataFrame:
    """Soccer: post at ``t = kickoff − lead`` (truncated to the hour), rest until kickoff."""
    return pl.concat([legs.with_columns((pl.col("kickoff") - timedelta(hours=h)).dt.truncate("1h").alias("t"),
                                        pl.col("kickoff").alias("end"), pl.lit("long" if h == max(leads) else "short").alias("lead"))
                      for h in leads])


def us_decisions(legs: pl.DataFrame, windows: pl.DataFrame) -> pl.DataFrame:
    """US leagues: one leg per game, the one in the recorded pre-game trade table
    (``kalshi_trade_windows``: every print of that leg from scheduled start − 22h to start − 1h).
    Long lead: post at the first full hour of that window (≈ 21–22h before kickoff); short lead:
    ``kickoff − 6h`` truncated to the hour. Orders rest until the window ends (≈ 1h before the
    Polymarket kickoff) or kickoff, whichever is first."""
    w = windows.select("ticker", "window_start", "window_end")
    base = legs.join(w, on="ticker").with_columns(pl.min_horizontal("kickoff", "window_end").alias("end"))
    long_t = pl.col("window_start").dt.truncate("1h") + pl.when(pl.col("window_start").dt.truncate("1h") < pl.col("window_start")) \
        .then(pl.duration(hours=1)).otherwise(pl.duration(hours=0))
    out = pl.concat([base.with_columns(long_t.alias("t"), pl.lit("long").alias("lead")),
                     base.with_columns((pl.col("kickoff") - timedelta(hours=6)).dt.truncate("1h").alias("t"), pl.lit("short").alias("lead"))])
    return out.filter((pl.col("t") >= pl.col("window_start")) & (pl.col("t") < pl.col("end"))).drop("window_start", "window_end")


def decision_rows(base: pl.DataFrame, candles: pl.DataFrame, paths: dict[str, tuple[np.ndarray, np.ndarray]]) -> pl.DataFrame:
    """Decision rows (``base`` carries t, end, lead per leg) while the market was open, with the
    Kalshi bid/ask from the hourly candle that closed by ``t`` (two-sided books only: 1c ≤ bid <
    ask ≤ 99c, so the Kalshi mid is a real price) and the Polymarket fair ``p_ref`` from buckets
    that ended by ``t`` (at most 2h old); ``p_end`` is the same fair when the order expires (the
    reference "closing" price, used only to mark fills, never to decide)."""
    base = base.filter(pl.col("open_time") <= pl.col("t"))
    q = F.attach_quotes(base, candles).filter((pl.col("bid") >= 0.01) & (pl.col("ask") <= 0.99))
    ts = q["t"].dt.epoch("s").to_numpy()
    te = q["end"].dt.epoch("s").to_numpy()
    p_ref = [fair_at(paths[tk], float(t)) if tk in paths else None for tk, t in zip(q["ticker"], ts)]
    p_end = [fair_at(paths[tk], float(t)) if tk in paths else None for tk, t in zip(q["ticker"], te)]
    q = q.with_columns(pl.Series("p_ref", p_ref, dtype=pl.Float64), pl.Series("p_end", p_end, dtype=pl.Float64),
                       ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                       ((pl.col("kickoff") - pl.col("t")).dt.total_minutes() / 60).alias("lead_h"))
    return q.filter(pl.col("p_ref").is_not_null() & (pl.col("p_ref") > 0) & (pl.col("p_ref") < 1))


def group_of(league: str, kind: str) -> str:
    """Segment label: big European soccer (EPL, top-4 leagues, UCL), other soccer, or the US league."""
    if kind == "soccer":
        return "soccer_big" if league in ("EPL", "ESP1", "GER1", "ITA1", "FRA1", "UCL") else "soccer_other"
    return league


# ---- maker simulation ---------------------------------------------------------------------------
def _bid_prices(p: float, margin: float, bid: float, ask: float) -> tuple[float, float]:
    """YES bid ``floor(p − margin)`` and NO bid ``floor(1 − p − margin)`` in whole cents, each
    capped one tick inside the opposite Kalshi quote (post-only), as in ``F.maker_fills``."""
    yb = min(math.floor((p - margin) * 100 + 1e-9) / 100, round(ask - 0.01, 2))
    nb = min(math.floor((1 - p - margin) * 100 + 1e-9) / 100, round(1 - bid - 0.01, 2))
    return yb, nb


def first_through(ts: np.ndarray, px: np.ndarray, side: np.ndarray, price: float, which: str,
                  lo: float, hi: float, *, lenient: bool = False) -> float:
    """First trade time in (lo, hi] that fills a resting bid at ``price``: for a YES bid, a
    YES-seller (taker_side "no") at a YES price strictly below it; for a NO bid, a YES-buyer at a
    YES price strictly above ``1 − price``. ``lenient`` also counts prints AT our price (the
    queue ahead of us assumed empty). ``inf`` if none. ``ts`` must be sorted."""
    a, b = np.searchsorted(ts, lo, side="right"), np.searchsorted(ts, hi, side="right")
    if a >= b:
        return math.inf
    eps = -1e-9 if lenient else 1e-9
    if which == "yes":
        ok = (side[a:b] == 0) & (px[a:b] < price - eps)
    else:
        ok = (side[a:b] == 1) & (px[a:b] > 1 - price + eps)
    i = np.flatnonzero(ok)
    return float(ts[a + i[0]]) if len(i) else math.inf


def trade_arrays(trades: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``{ticker: (epoch seconds, yes_price, taker side 1 = yes / 0 = no)}``, time-sorted."""
    t = trades.select("ticker", pl.col("created_time").dt.epoch("us").alias("us"), "yes_price",
                      (pl.col("taker_side") == "yes").cast(pl.Int8).alias("s")).sort("ticker", "us")
    return {k[0]: (v["us"].to_numpy() / 1e6, v["yes_price"].to_numpy(), v["s"].to_numpy())
            for k, v in t.partition_by("ticker", as_dict=True).items()}


def simulate(rows: pl.DataFrame, trades: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
             paths: dict[str, tuple[np.ndarray, np.ndarray]], *, margins: tuple[float, ...] = MARGINS,
             pulls: tuple[float | None, ...] = PULLS) -> pl.DataFrame:
    """Every order of the grid fair ∈ {ref, mid} × margin × side, posted at ``t`` and resting
    until ``end``, with the first trade-through time (strict and lenient) and, per pull ``k``,
    whether the fill came before the pull. One row per order; ``pnl_if_filled`` is the PnL of
    one contract held to settlement after the series maker fee, ``mark_if_filled`` the same
    against the Polymarket fair at expiry (a low-variance markout)."""
    out: list[dict[str, Any]] = []
    for r in rows.iter_rows(named=True):
        t0, t1 = r["t"].timestamp(), r["end"].timestamp()
        tr = trades.get(r["ticker"])
        path = paths[r["ticker"]]
        taus = {k: pull_time(path, t0, t1, r["p_ref"], k) for k in pulls}
        ft, mult = r.get("fee_type"), r.get("fee_multiplier") or 1.0
        for fair, p in (("ref", r["p_ref"]), ("mid", r["mid"])):
            for m in margins:
                yb, nb = _bid_prices(p, m, r["bid"], r["ask"])
                for side, price in (("yes", yb), ("no", nb)):
                    if price < 0.01:
                        continue
                    if tr is None:
                        hs = hl = math.inf
                    else:
                        hs = first_through(*tr, price, side, t0, t1)
                        hl = first_through(*tr, price, side, t0, t1, lenient=True)
                    won = r["y"] if side == "yes" else 1 - r["y"]
                    fee = F.maker_fee(price, ft, mult)
                    close = None if r["p_end"] is None else (r["p_end"] if side == "yes" else 1 - r["p_end"])
                    rec = {"row": r["row"], "fair": fair, "margin": m, "side": side, "price": price,
                           "hit": hs, "hit_lenient": hl, "pnl_if_filled": won - price - fee,
                           "mark_if_filled": None if close is None else close - price - fee}
                    for k, tau in taus.items():
                        rec[f"tau_{_kname(k)}"] = tau
                    out.append(rec)
    return pl.DataFrame(out)


def _kname(k: float | None) -> str:
    return "none" if k is None else f"{round(k * 100):d}c"


# ---- requote variant ---------------------------------------------------------------------------
def requote_times(path: tuple[np.ndarray, np.ndarray], t0: float, t1: float, k: float) -> list[float]:
    """Re-posting moments for a maker that cancels and re-posts whenever the fair has moved ≥ k
    since its last post: [t0, τ1, τ2, ...] with τ_{i+1} the first bucket end after τ_i whose fair
    is ≥ k from the fair at τ_i."""
    ts, fv = path
    lo, hi = np.searchsorted(ts, t0, side="right"), np.searchsorted(ts, t1, side="right")
    i0 = np.searchsorted(ts, t0, side="right") - 1
    ref = float(fv[i0]) if i0 >= 0 else float(fv[lo]) if lo < len(fv) else 0.0
    out = [t0]
    for tt, f in zip(ts[lo:hi].tolist(), fv[lo:hi].tolist()):
        if abs(f - ref) >= k - 1e-9:
            out.append(tt)
            ref = f
    return out


def candle_arrays(candles: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    c = (candles.filter(pl.col("yes_bid_close").is_not_null() & pl.col("yes_ask_close").is_not_null())
         .select("ticker", pl.col("end_ts").dt.epoch("s").alias("e"), "yes_bid_close", "yes_ask_close").sort("ticker", "e"))
    return {k[0]: (v["e"].to_numpy().astype(float), v["yes_bid_close"].to_numpy(), v["yes_ask_close"].to_numpy())
            for k, v in c.partition_by("ticker", as_dict=True).items()}


def simulate_requote(rows: pl.DataFrame, trades: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                     paths: dict[str, tuple[np.ndarray, np.ndarray]], candles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                     *, margins: tuple[float, ...] = MARGINS, pulls: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05)) -> pl.DataFrame:
    """Pull AND re-post: at every requote time (:func:`requote_times`) both bids are cancelled
    and re-posted around the new fair (ref: the Polymarket fair then; mid: the Kalshi mid of the
    last hourly candle closed by then), capped inside the Kalshi quote of that candle. At most
    one fill per side per row (the first segment whose bid a later print trades through)."""
    out: list[dict[str, Any]] = []
    for r in rows.iter_rows(named=True):
        t0, t1 = r["t"].timestamp(), r["end"].timestamp()
        tr, path, cd = trades.get(r["ticker"]), paths[r["ticker"]], candles.get(r["ticker"])
        ft, mult = r.get("fee_type"), r.get("fee_multiplier") or 1.0
        for k in pulls:
            b = requote_times(path, t0, t1, k)
            ends = b[1:] + [t1]
            # quotes and fairs at each post
            segs = []
            for s0, s1 in zip(b, ends):
                if s0 == t0:
                    pr, bid, ask = r["p_ref"], r["bid"], r["ask"]
                else:
                    pr = fair_at(path, s0)
                    j = int(np.searchsorted(cd[0], s0, side="right")) - 1 if cd is not None else -1
                    if pr is None or j < 0 or s0 - cd[0][j] > 7200:
                        continue
                    bid, ask = float(cd[1][j]), float(cd[2][j])
                    if not (0.01 <= bid < ask <= 0.99):
                        continue
                if tr is not None:
                    a, z = np.searchsorted(tr[0], s0, side="right"), np.searchsorted(tr[0], s1, side="right")
                    sells, buys = tr[1][a:z][tr[2][a:z] == 0], tr[1][a:z][tr[2][a:z] == 1]
                    lo_sell = float(sells.min()) if len(sells) else math.inf
                    hi_buy = float(buys.max()) if len(buys) else -math.inf
                else:
                    lo_sell, hi_buy = math.inf, -math.inf
                segs.append((pr, (bid + ask) / 2, bid, ask, lo_sell, hi_buy))
            for fair in ("ref", "mid"):
                for m in margins:
                    for side in ("yes", "no"):
                        price, n_posts = None, 0
                        for pr, mid, bid, ask, lo_sell, hi_buy in segs:
                            yb, nb = _bid_prices(pr if fair == "ref" else mid, m, bid, ask)
                            px = yb if side == "yes" else nb
                            if px < 0.01:
                                continue
                            n_posts += 1
                            if (side == "yes" and lo_sell < px - 1e-9) or (side == "no" and hi_buy > 1 - px + 1e-9):
                                price = px
                                break
                        if not n_posts:
                            continue
                        won = r["y"] if side == "yes" else 1 - r["y"]
                        fee = F.maker_fee(price, ft, mult) if price is not None else 0.0
                        close = None if r["p_end"] is None else (r["p_end"] if side == "yes" else 1 - r["p_end"])
                        out.append({"row": r["row"], "fair": fair, "margin": m, "side": side, "pull": _kname(k),
                                    "posts": n_posts, "filled": price is not None, "price": price,
                                    "pnl": (won - price - fee) if price is not None else 0.0,
                                    "mark": (close - price - fee) if price is not None and close is not None else None})
    return pl.DataFrame(out)


# ---- evaluation -----------------------------------------------------------------------------------
def long_orders(orders: pl.DataFrame, rows: pl.DataFrame, holdout_from: datetime, *, lenient: bool = False,
                pulls: tuple[float | None, ...] = PULLS) -> pl.DataFrame:
    """One row per (order, pull k): filled if the first through-print came before the pull."""
    info = rows.select("row", "event", "ticker", "t", "lead", "group", "band", "gap", (pl.col("t") >= holdout_from).alias("holdout"))
    hit = "hit_lenient" if lenient else "hit"
    parts = []
    for k in pulls:
        tau = pl.col(f"tau_{_kname(k)}")
        filled = pl.col(hit).is_finite() & (pl.col(hit) <= tau)
        parts.append(orders.select("row", "fair", "margin", "side", "price", pl.lit(_kname(k)).alias("pull"), filled.alias("filled"),
                                   pl.when(filled).then(pl.col("pnl_if_filled")).otherwise(0.0).alias("pnl"),
                                   pl.when(filled).then(pl.col("mark_if_filled")).otherwise(None).alias("mark")))
    return pl.concat(parts).join(info, on="row")


def summarize(lo: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Orders, fills, fill rate, PnL per fill (cents, t clustered by event) and per order, and
    the markout per fill against the Polymarket fair at expiry (cents, t by event)."""
    out = []
    for key, g in lo.group_by(keys):
        f = g.filter("filled")
        row = dict(zip(keys, key)) | {"orders": g.height, "fills": f.height,
                                       "fill_rate": round(f.height / g.height, 3) if g.height else None}
        if f.height:
            mu, tt, k = F.clustered_mean(f["pnl"].to_numpy(), f["event"].to_numpy())
            row |= {"events": k, "pnl_per_fill_c": round(mu * 100, 3), "t": None if tt is None else round(tt, 2),
                    "pnl_per_order_c": round(float(g["pnl"].sum()) / g.height * 100, 3),
                    "total_pnl": round(float(f["pnl"].sum()), 2), "yes_share": round(float((f["side"] == "yes").mean()), 3)}
            if "mark" in f.columns:
                fm = f.filter(pl.col("mark").is_not_null())
                if fm.height:
                    m2, t2, _ = F.clustered_mean(fm["mark"].to_numpy(), fm["event"].to_numpy())
                    row |= {"markout_c": round(m2 * 100, 3), "t_markout": None if t2 is None else round(t2, 2)}
        out.append(row)
    return pl.DataFrame(out, infer_schema_length=None).sort(keys) if out else pl.DataFrame()


def choose(grid: pl.DataFrame, *, by: str = "pnl_per_fill_c", min_fills: int = 30) -> dict[str, Any] | None:
    """Dev choice of (margin, pull): best ``by`` among dev cells with at least ``min_fills``."""
    d = grid.filter((pl.col("split") == "dev") & (pl.col("fills") >= min_fills) & pl.col(by).is_not_null())
    if not d.height:
        return None
    best = d.sort(by, descending=True).row(0, named=True)
    return {"margin": best["margin"], "pull": best["pull"]}


def split_col(lo: pl.DataFrame) -> pl.DataFrame:
    return lo.with_columns(pl.when(pl.col("holdout")).then(pl.lit("holdout")).otherwise(pl.lit("dev")).alias("split"))


def grid_and_choice(lo: pl.DataFrame, *, by: str = "pnl_per_fill_c", min_fills: int = 30) -> dict[str, Any]:
    """Full (fair, pull, margin, split) grid, the dev choice per fair, and its dev/holdout rows."""
    lo = split_col(lo)
    grid = summarize(lo, ["fair", "pull", "margin", "split"])
    chosen, rows = {}, []
    for fair in ("ref", "mid"):
        c = choose(grid.filter(pl.col("fair") == fair), by=by, min_fills=min_fills)
        chosen[fair] = c
        if c:
            rows += grid.filter((pl.col("fair") == fair) & (pl.col("margin") == c["margin"]) & (pl.col("pull") == c["pull"])).to_dicts()
    return {"grid": grid, "chosen": chosen, "chosen_rows": pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()}


# ---- the study ----------------------------------------------------------------------------------
LEG_COLS = ["event", "ticker", "league", "kind", "kickoff", "open_time", "close_time", "y", "tokens", "leg_index",
            "fee_type", "fee_multiplier"]


def band_of(p: pl.Expr) -> pl.Expr:
    expr = pl.lit(None, dtype=pl.String)
    for lo, hi in reversed(BANDS):
        expr = pl.when((p >= lo) & (p < hi)).then(pl.lit(f"{lo:.1f}-{min(hi, 1.0):.1f}")).otherwise(expr)
    return expr


def build_rows(s) -> tuple[pl.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]], pl.DataFrame, pl.DataFrame]:
    """Legs of both universes, their fair-value paths, the decision rows (with ``row`` id,
    ``group``, price ``band`` of the leg's Polymarket fair, ``gap`` = p_ref − Kalshi mid) and the
    candles. Only decision times from ``PM_FIVE_MIN_FROM`` on (5-minute Polymarket history)."""
    sl = soccer_legs(s).select(LEG_COLS)
    ul = us_legs(s).with_columns(pl.col("leg_index").cast(pl.Int32)).select(LEG_COLS)
    legs = pl.concat([sl, ul], how="vertical_relaxed")
    hist = pl.concat([pl.read_parquet(s.data_dir / "cache" / "fundamental_soccer_pm_history.parquet"),
                      pl.read_parquet(s.data_dir / "cache" / f"fundamental_{NAME}_pm_history.parquet")])
    hist = hist.filter((pl.col("price") > 0) & (pl.col("price") < 1))
    paths = fair_paths(legs, hist)
    candles = pl.concat([pl.read_parquet(s.data_dir / "cache" / "fundamental_soccer_candles.parquet"),
                         pl.read_parquet(s.data_dir / "cache" / f"fundamental_{NAME}_candles.parquet")], how="diagonal_relaxed")
    windows = pl.read_parquet(s.tables_dir / "kalshi_trade_windows.parquet")
    base = pl.concat([soccer_decisions(sl), us_decisions(ul, windows)], how="diagonal_relaxed")
    rows = decision_rows(base, candles, paths).filter(pl.col("t") >= PM_FIVE_MIN_FROM)
    rows = (rows.sort("t", "ticker", "lead").with_row_index("row")
            .with_columns(pl.struct("league", "kind").map_elements(lambda x: group_of(x["league"], x["kind"]), return_dtype=pl.String).alias("group"),
                          band_of(pl.col("p_ref")).alias("band"), (pl.col("p_ref") - pl.col("mid")).alias("gap")))
    return legs, paths, rows.with_columns(pl.col("row").cast(pl.Int64)), candles


def soccer_trade_windows(rows: pl.DataFrame) -> pl.DataFrame:
    """Trade windows of the soccer legs: earliest decision time to kickoff."""
    return (rows.filter(pl.col("kind") == "soccer").group_by("ticker")
            .agg(pl.col("t").min().alias("t_from"), pl.col("kickoff").max().alias("t_to")))


def load_trades(s, rows: pl.DataFrame, *, fetch: bool = False) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Kalshi prints for every row's window, as (prints for the main fill rule, the US prints
    as recorded with their taker side, tickers whose window is complete).

    Main rule, both universes: a resting bid fills when a later print trades strictly through
    it, whichever side took it (see :func:`minute_prints`). Soccer: 1-minute candles (each
    minute's low and high print) from this study's own fetch (``fetch=True`` fetches what is
    missing; otherwise only legs whose window is complete are kept). US: the recorded pre-game
    trade table, complete inside each leg's window (also kept with taker sides, for the
    harness's stricter taker-side rule)."""
    w = soccer_trade_windows(rows)
    if fetch:
        fetch_minute_candles(s, w)
    path = s.data_dir / "cache" / f"fundamental_{NAME}_minute.parquet"
    done_path = path.with_name(path.stem + "_windows.parquet")
    if path.exists() and done_path.exists():
        sw = w.join(pl.read_parquet(done_path), on=["ticker", "t_from", "t_to"], how="semi").select("ticker")
        sm = pl.scan_parquet(path).join(sw.lazy(), on="ticker", how="semi").collect()
    else:
        sw, sm = w.clear().select("ticker"), pl.DataFrame(schema=MINUTE_SCHEMA)
    us = rows.filter(pl.col("kind") == "us").select("ticker").unique()
    win = pl.read_parquet(s.tables_dir / "kalshi_trade_windows.parquet").join(us, on="ticker", how="semi")
    cols = ["ticker", "created_time", "yes_price", "taker_side", "count"]
    ut = (pl.scan_parquet(s.tables_dir / "kalshi_trades.parquet").join(win.lazy(), on="ticker", how="inner")
          .filter((pl.col("created_time") >= pl.col("window_start")) & (pl.col("created_time") <= pl.col("window_end")))
          .select(cols).collect())
    main = pl.concat([minute_prints(sm).select(cols), side_agnostic(ut).select(cols)], how="vertical_relaxed")
    covered = pl.concat([sw, win.select("ticker")]).unique()
    return main, ut, covered


def fill_diagnostics(orders: pl.DataFrame, rows: pl.DataFrame, paths: dict[str, tuple[np.ndarray, np.ndarray]],
                     holdout_from: datetime) -> pl.DataFrame:
    """For every filled order without a pull: the Polymarket fair known at the fill (buckets
    ended by then) and when the order would have expired (``end``), as edge in cents for the side we bought, and whether the
    fair had already moved ≥ 1c against us before the print (a pull could have saved it)."""
    f = orders.filter(pl.col("hit").is_finite()).join(rows.select("row", "ticker", "t", "end", "lead", "p_ref", "event"), on="row")
    out = []
    for r in f.iter_rows(named=True):
        path = paths[r["ticker"]]
        pf = fair_at(path, r["hit"], max_age=1e9)
        pk = fair_at(path, r["end"].timestamp(), max_age=1e9)
        sgn = 1 if r["side"] == "yes" else -1
        v_post, v_fill, v_kick = ((p if sgn > 0 else 1 - p) for p in (r["p_ref"], pf, pk))
        out.append({"fair": r["fair"], "margin": r["margin"], "lead": r["lead"], "split": "holdout" if r["t"] >= holdout_from else "dev",
                    "edge_at_post_c": (v_post - r["price"]) * 100, "edge_at_fill_c": (v_fill - r["price"]) * 100,
                    "edge_at_end_c": (v_kick - r["price"]) * 100, "moved_against_1c": sgn * (pf - r["p_ref"]) <= -0.01 + 1e-9,
                    "pnl_c": r["pnl_if_filled"] * 100})
    d = pl.DataFrame(out)
    return (d.group_by("fair", "margin", "lead", "split")
            .agg(pl.len().alias("fills"), pl.col("edge_at_post_c").mean().round(2), pl.col("edge_at_fill_c").mean().round(2),
                 pl.col("edge_at_end_c").mean().round(2), pl.col("moved_against_1c").mean().round(3).alias("share_pm_moved_against_first"),
                 pl.col("pnl_c").mean().round(2).alias("pnl_per_fill_c"))
            .sort("fair", "lead", "margin", "split"))


def run(s, *, holdout_from: datetime = HOLDOUT, fetch: bool = False) -> dict[str, Any]:
    """The whole study from cached inputs (fetching whatever is missing)."""
    legs, paths, rows, candles = build_rows(s)
    all_rows = rows.height
    trades, us_prints, covered = load_trades(s, rows, fetch=fetch)
    rows = rows.join(covered, on="ticker", how="semi")
    arr = trade_arrays(trades)
    extra: dict[str, Any] = {
        "legs": legs.height, "legs_with_pm_path": len(paths), "rows_with_quotes_and_fair": all_rows,
        "rows_with_complete_trades": rows.height, "events": rows["event"].n_unique(),
        "rows_by_lead_split": rows.group_by("lead", (pl.col("t") >= holdout_from).alias("holdout")).len().sort("lead", "holdout").to_dicts(),
        "trades": trades.height, "dev": f"{PM_FIVE_MIN_FROM.date()} to {holdout_from.date()}", "holdout_from": str(holdout_from.date()),
        "median_abs_gap_c": round(float(rows["gap"].abs().median()) * 100, 3),
        "median_spread_c": round(float((rows["ask"] - rows["bid"]).median()) * 100, 3)}
    universe = (rows.group_by("kind", "group", "lead", (pl.col("t") >= holdout_from).alias("holdout"))
                .agg(pl.len().alias("rows"), pl.col("event").n_unique().alias("games"),
                     ((pl.col("ask") - pl.col("bid")) * 100).mean().round(2).alias("spread_c"),
                     (pl.col("gap").abs() * 100).mean().round(2).alias("abs_gap_c"))
                .sort("kind", "group", "lead", "holdout"))

    # taker (harness) on the Polymarket fair, per lead
    taker = {}
    for kind in ("soccer", "us"):
        for lead in LEAD_CLASSES:
            part = rows.filter((pl.col("lead") == lead) & (pl.col("kind") == kind)).with_columns(pl.col("p_ref").alias("p"))
            if part.height:
                taker[f"{kind}, {lead}"] = F.evaluate(part, holdout_from)

    orders = simulate(rows, arr, paths)
    out: dict[str, Any] = {"extra": extra, "universe": universe, "taker": taker, "rows": rows}
    makers, chosen_tab, pull_tables, segs, variants = {}, [], [], [], []
    for scope in ("all", "soccer", "us"):
        for lead in LEAD_CLASSES:
            rl = rows.filter((pl.col("lead") == lead) & ((pl.col("kind") == scope) if scope != "all" else pl.lit(True)))
            if not rl.height:
                continue
            ol = orders.join(rl.select("row"), on="row")
            lo = long_orders(ol, rl, holdout_from)
            gc = grid_and_choice(lo)
            makers[(scope, lead)] = gc
            chosen_tab += [{"scope": scope, "lead": lead, **r} for r in gc["chosen_rows"].to_dicts()]
            lo = split_col(lo)
            for fair, c in gc["chosen"].items():
                if not c:
                    continue
                # does the pull matter? same margin, every pull threshold
                pt = summarize(lo.filter((pl.col("fair") == fair) & (pl.col("margin") == c["margin"])), ["pull", "split"])
                pull_tables += [{"scope": scope, "lead": lead, "fair": fair, "margin": c["margin"], **r} for r in pt.to_dicts()]
                if scope != "all":
                    continue
                sel = lo.filter((pl.col("fair") == fair) & (pl.col("margin") == c["margin"]) & (pl.col("pull") == c["pull"]))
                sel = sel.with_columns(pl.col("t").dt.truncate("1w").dt.date().cast(pl.String).alias("week"))
                for seg in ("group", "band", "week"):
                    for r in summarize(sel, [seg, "split"]).to_dicts():
                        segs.append({"lead": lead, "fair": fair, "margin": c["margin"], "pull": c["pull"], "segment": seg,
                                     "value": r.pop(seg), **r})
            if scope != "all":
                continue
            # variants, each re-chosen on dev: only rest when |fair − mid| ≥ x; lenient fills; per-order choice
            for x in (0.01, 0.02, 0.03):
                keep = rl.filter(pl.col("gap").abs() >= x - 1e-9).select("row")
                v = grid_and_choice(lo.drop("split").join(keep, on="row"))
                variants += [{"lead": lead, "variant": f"|p_ref-mid|>={x:.2f}", **r} for r in v["chosen_rows"].to_dicts()]
            v = grid_and_choice(long_orders(ol, rl, holdout_from, lenient=True))
            variants += [{"lead": lead, "variant": "lenient fills (prints at our price count)", **r} for r in v["chosen_rows"].to_dicts()]
            v = grid_and_choice(lo.drop("split"), by="pnl_per_order_c")
            variants += [{"lead": lead, "variant": "margin chosen by PnL per order", **r} for r in v["chosen_rows"].to_dicts()]
            v = grid_and_choice(lo.drop("split"), by="markout_c")
            variants += [{"lead": lead, "variant": "margin chosen by markout per fill", **r} for r in v["chosen_rows"].to_dicts()]
    # US prints with the harness's taker-side rule (a YES bid fills only on a YES-seller print)
    us_rows = rows.filter(pl.col("kind") == "us")
    if us_rows.height:
        oh = simulate(us_rows, trade_arrays(us_prints), paths)
        for lead in LEAD_CLASSES:
            rl = us_rows.filter(pl.col("lead") == lead)
            v = grid_and_choice(long_orders(oh.join(rl.select("row"), on="row"), rl, holdout_from))
            variants += [{"lead": lead, "variant": "US only, harness taker-side rule", **r} for r in v["chosen_rows"].to_dicts()]
            vv = grid_and_choice(long_orders(orders.join(rl.select("row"), on="row"), rl, holdout_from))
            variants += [{"lead": lead, "variant": "US only, main side-agnostic rule", **r} for r in vv["chosen_rows"].to_dicts()]
    out |= {"makers": makers, "chosen": pl.DataFrame(chosen_tab, infer_schema_length=None),
            "pull": pl.DataFrame(pull_tables, infer_schema_length=None),
            "segments": pl.DataFrame(segs, infer_schema_length=None), "variants": pl.DataFrame(variants, infer_schema_length=None)}

    # pull AND re-post at the new fair
    rq = simulate_requote(rows, arr, paths, candle_arrays(candles))
    rq = rq.join(rows.select("row", "event", "lead", "kind", (pl.col("t") >= holdout_from).alias("holdout")), on="row")
    rq_rows = []
    for scope in ("all", "soccer", "us"):
        for lead in LEAD_CLASSES:
            part = rq.filter((pl.col("lead") == lead) & ((pl.col("kind") == scope) if scope != "all" else pl.lit(True)))
            if not part.height:
                continue
            gc = grid_and_choice(part)
            rq_rows += [{"scope": scope, "lead": lead, **r} for r in gc["chosen_rows"].to_dicts()]
            out.setdefault("requote_grid", {})[(scope, lead)] = gc["grid"]
    out["requote"] = pl.DataFrame(rq_rows, infer_schema_length=None)
    out["diagnostics"] = fill_diagnostics(orders.filter(pl.col("margin").is_in([0.0, 0.01, 0.02, 0.03])), rows, paths, holdout_from)

    # October prediction rows for cross-study comparison
    hold = rows.filter(pl.col("t") >= holdout_from).select("ticker", "event", "t", "lead", pl.col("p_ref").alias("p_fair"), "mid", "bid", "ask",
                                                         "y", "group")
    _write(hold, s.data_dir / "cache" / f"fundamental_{NAME}_holdout_preds.parquet")
    return out


def write_report(s, out: dict[str, Any]):
    from albert.reports import Report

    rep = Report(s.reports_dir, f"research-fundamental-{NAME}")
    rep.json("Run", out["extra"])
    rep.table("Universe: decision rows with a Polymarket fair, a two-sided Kalshi book and complete prints", out["universe"])
    for key, r in out["taker"].items():
        if r.get("scores"):
            rep.table(f"[taker, {key}] scores: Polymarket fair vs Kalshi mid", pl.DataFrame(r["scores"]))
        if r.get("blend"):
            rep.json(f"[taker, {key}] blend logit P = a + b·logit(mid) + c·logit(p_ref)", r["blend"])
        rep.table(f"[taker, {key}] taker at the ask/bid with fees (theta chosen on dev = {r.get('theta_chosen_on_dev')})",
                  pl.DataFrame(r["taker"], infer_schema_length=None))
    rep.table("Maker: (margin, pull) chosen on dev per scope, lead and fair; dev vs October holdout", out["chosen"],
              save_as="maker_chosen")
    rep.table("Does the pull matter? Chosen margin, every pull threshold", out["pull"], save_as="pull")
    rep.table("Variants (each re-chosen on dev; scope all)", out["variants"], save_as="variants")
    rep.table("Pull AND re-post at the new fair (chosen on dev)", out["requote"], save_as="requote")
    rep.table("Segments at the chosen configuration, scope all (group, Polymarket price band, week)", out["segments"],
              save_as="segments")
    rep.table("Fill diagnostics (no pull): Polymarket edge of the side bought, at post / fill / expiry", out["diagnostics"])
    for (scope, lead), gc in out["makers"].items():
        rep.table(f"[maker, {scope}, {lead}] full grid", gc["grid"], save_as=f"grid_{scope}_{lead}")
    for (scope, lead), g in out.get("requote_grid", {}).items():
        rep.table(f"[requote, {scope}, {lead}] full grid", g, save_as=f"requote_grid_{scope}_{lead}")
    return rep.save()
