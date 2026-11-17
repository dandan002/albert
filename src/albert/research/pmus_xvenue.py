"""Polymarket US (data only) as a signal for Kalshi game moneylines.

Polymarket US is a separate, CFTC-regulated order book with its own (mostly US retail) flow; it is
never traded here (AGENTS.md: Kalshi is the only execution venue). This study asks whether its
quotes tell a Kalshi trader anything the Kalshi book does not already show.

Data
* **Polymarket US**: every sports moneyline on the public gateway (``gateway.polymarket.us/v1``,
  no key) whose game starts inside the study window, and its ``/price-history`` at
  ``fidelity=1``: one point per whole minute with ``longPrice`` (YES display price, the best ask)
  and ``shortPrice`` (NO display price, 1 - best bid). These are book snapshots, not trades.
  Window per market: game start - 6 h to the market's end (custom ranges of at most 24 h).
* **Kalshi**: every top-of-book change of every GAME market from the ``albert record-vol``
  recordings (``pm_vol`` HF extraction, ``data/research/pm_vol/hf/quotes_<night>.parquet``), and
  the settled market list per series for team names and results.

Matching (``match_us`` of the reference-price study, reused): Kalshi game -> Polymarket US game of
the same league within a day of the ticker date, both teams agreeing by code or name; the long
side of the Polymarket US market is mapped to the Kalshi leg of the same team.

As-of rule (AGENTS.md rule 4): a Polymarket US point stamped at minute ``t`` is used only from
``t + ASOF_DELAY`` (default 60 s, i.e. as if it summarised the minute that starts at ``t``),
plus ``LATENCY`` before a Kalshi order reaches the book. :func:`stamp_check` measures what the
stamp means against Kalshi's book.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

NAME = "pmus_xvenue"
LEAGUES = {"KXNBAGAME": "nba", "KXNCAAMBGAME": "cbb", "KXNCAAWBGAME": "wcbb", "KXNCAAFGAME": "cfb",
           "KXNFLGAME": "nfl", "KXNHLGAME": "nhl"}
# record-vol nights with extracted Kalshi game quotes (pm_vol HF_NIGHTS)
NIGHTS = {"wed": ("2026-11-11T17", "2026-11-12T06"), "thu": ("2026-11-12T17", "2026-11-13T06"),
          "fri": ("2026-11-13T17", "2026-11-14T06"), "sat2": ("2026-11-14T17", "2026-11-15T06"),
          "sun2": ("2026-11-15T17", "2026-11-16T06")}
DEV = ("wed", "thu", "fri")
HOLDOUT = ("sat2", "sun2")
PRE = timedelta(hours=6)
ASOF_DELAY = 60.0
LATENCY = 0.05


def out_dir(s) -> Path:
    return s.data_dir / "research" / NAME


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _ts(x: str | None) -> datetime | None:
    if not x:
        return None
    return datetime.fromisoformat(x)


# ---- Polymarket US: games and minute books --------------------------------------------------------
def client(s, *, min_interval: float = 0.25):
    """Public gateway client (no key), at most 4 requests/s (the public limit is 20/s/IP)."""
    from albert.http import CachedClient

    return CachedClient("pmus", s.pmus_base, s.raw_dir, min_interval=min_interval)


def game_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Two-team moneylines (``aec-<league>-<a>-<b>-<date>``): league, codes in slug order, the long
    side's team (code, name, nickname), scheduled start, end, and the long side's settlement."""
    from albert.venues.polymarket import _json_list

    rows = []
    for m in payload.get("markets") or []:
        slug = m.get("slug") or ""
        parts = slug.split("-")
        if m.get("marketType") != "moneyline" or len(parts) < 7 or parts[0] != "aec":
            continue
        sides = m.get("marketSides") or []
        long_side = next((x for x in sides if x.get("long")), None)
        short_side = next((x for x in sides if not x.get("long")), None)
        if not long_side or not short_side:
            continue
        lt, st = long_side.get("team") or {}, short_side.get("team") or {}
        settle = None
        if m.get("status") == "MARKET_STATUS_RESOLVED":
            prices = _json_list(m.get("outcomePrices"))
            settle = float(prices[0]) if prices else None
        rows.append({
            "slug": slug, "league": parts[1], "code_a": parts[2], "code_b": parts[3],
            "slug_date": "-".join(parts[4:7]),
            "long_code": (lt.get("abbreviation") or "").lower(), "long_name": lt.get("name") or "",
            "long_desc": long_side.get("description") or "",
            "short_code": (st.get("abbreviation") or "").lower(), "short_name": st.get("name") or "",
            "short_desc": short_side.get("description") or "",
            "game_start": _ts(m.get("gameStartTime")), "end_date": _ts(m.get("endDate")),
            "status": m.get("status"), "settlement": settle, "fee_coefficient": m.get("feeCoefficient"),
            "tick": m.get("orderPriceMinTickSize"), "sports_type": m.get("sportsMarketType"),
        })
    return rows


GAME_SCHEMA = {"slug": pl.String, "league": pl.String, "code_a": pl.String, "code_b": pl.String,
               "slug_date": pl.String, "long_code": pl.String, "long_name": pl.String, "long_desc": pl.String,
               "short_code": pl.String, "short_name": pl.String, "short_desc": pl.String,
               "game_start": pl.Datetime("us", "UTC"), "end_date": pl.Datetime("us", "UTC"),
               "status": pl.String, "settlement": pl.Float64, "fee_coefficient": pl.Float64,
               "tick": pl.Float64, "sports_type": pl.String}


def list_games(c, start: datetime, end: datetime) -> pl.DataFrame:
    """Every Polymarket US sports moneyline whose game starts in ``[start, end)``. Pages are capped
    at 500 and can come back short, so paging stops only on an empty page."""
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    rows: list[dict[str, Any]] = []
    for closed in (True, False):
        offset = 0
        while True:
            page = c.get("/markets", {
                "limit": 500, "offset": offset, "closed": str(closed).lower(), "orderBy": "id",
                "orderDirection": "asc", "categories": ["sports"], "marketTypes": ["moneyline"],
                "endDateMin": start.strftime(fmt), "endDateMax": (end + timedelta(hours=18)).strftime(fmt),
            }, cache=closed)
            got = page.payload.get("markets") or []
            if not got:
                break
            rows += game_rows(page.payload)
            offset += 500
    df = pl.DataFrame(rows, schema=GAME_SCHEMA).unique("slug", keep="first")
    return df.filter(pl.col("game_start").is_between(start, end, closed="left")).sort("game_start", "slug")


def history_rows(payload: dict[str, Any], slug: str) -> list[dict[str, Any]]:
    """Minute book snapshots: ``ask`` = longPrice, ``bid`` = 1 - shortPrice (YES side)."""
    out = []
    for p in payload.get("history") or []:
        ts, lp, sp = p.get("timestamp"), p.get("longPrice"), p.get("shortPrice")
        if ts is None or lp is None or sp is None:
            continue
        out.append({"slug": slug, "ts": int(ts), "ask": float(lp), "bid": round(1.0 - float(sp), 6)})
    return out


HIST_SCHEMA = {"slug": pl.String, "ts": pl.Int64, "ask": pl.Float64, "bid": pl.Float64}


def fetch_history(c, games: pl.DataFrame, *, pre: timedelta = PRE, log=print) -> pl.DataFrame:
    """Minute history of each game from ``game_start - pre`` to its end (closed markets only, so
    the responses are immutable and cached)."""
    rows: list[dict[str, Any]] = []
    for i, g in enumerate(games.iter_rows(named=True)):
        t0 = int((g["game_start"] - pre).timestamp())
        t1 = int(((g["end_date"] or g["game_start"] + timedelta(hours=6)) + timedelta(minutes=30)).timestamp())
        t1 = min(t1, int((g["game_start"] + timedelta(hours=18)).timestamp()))
        a = t0
        while a < t1:
            b = min(a + 86_400, t1)
            page = c.get("/price-history", {"symbol": g["slug"], "timestamp.startTimestamp": a,
                                            "timestamp.endTimestamp": b, "fidelity": 1})
            rows += history_rows(page.payload, g["slug"])
            a = b
        if log and (i + 1) % 100 == 0:
            log(f"  history {i + 1}/{games.height}, {len(rows)} points")
    return pl.DataFrame(rows, schema=HIST_SCHEMA).unique(["slug", "ts"]).sort("slug", "ts")


# ---- Kalshi games and the match -----------------------------------------------------------------
def kalshi_markets(s, start: datetime, end: datetime) -> pl.DataFrame:
    """Settled markets of each game series that closed in ``[start, end]`` (live listing, recorded)."""
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    kc = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.2))
    rows: list[dict[str, Any]] = []
    for series in LEAGUES:
        for page in kc.series_markets(series, "settled", min_close=start, max_close=end):
            rows += k.market_rows(page.payload, {}, "settled")
    return k.markets_frame(rows, non_financial_only=False)


def kalshi_games(markets: pl.DataFrame) -> pl.DataFrame:
    """One row per settled two-leg Kalshi game (the reference-price study's shape, all leagues of
    :data:`LEAGUES`)."""
    from albert.research.refprice_maker import _K_US_RX, _MON

    rows = []
    for (ev,), g in markets.group_by(["event_ticker"]):
        m = _K_US_RX.match(ev or "")
        if not m or g.height != 2 or m["series"] not in LEAGUES or g["result"].to_list().count("yes") != 1:
            continue
        legs = g.sort("ticker").to_dicts()
        rows.append({"event": ev, "series": m["series"], "league": LEAGUES[m["series"]],
                     "kdate": date(2000 + int(m["yy"]), _MON[m["mon"]], int(m["dd"])),
                     "occurrence_time": legs[0]["occurrence_time"], "open_time": min(x["open_time"] for x in legs),
                     "close_time": max(x["close_time"] for x in legs),
                     "expected_expiration_time": legs[0]["expected_expiration_time"],
                     **{f"k_{k}{i}": v for i, x in enumerate(legs) for k, v in (
                         ("ticker", x["ticker"]), ("code", x["ticker"].rsplit("-", 1)[1].lower()),
                         ("name", x["yes_sub_title"] or ""), ("y", int(x["result"] == "yes")))}})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _w(name: str | None) -> str:
    """Polymarket US women's teams carry a " (W)" suffix that Kalshi names do not."""
    return (name or "").removesuffix(" (W)")


def pm_frame(games: pl.DataFrame) -> pl.DataFrame:
    """Polymarket US games in the shape ``refprice_maker.match_us`` expects. Outcome 0 is the first
    code of the slug; ``pm_tok<i>`` is "L" when outcome i is the market's long (YES) side."""
    rows = []
    for g in games.iter_rows(named=True):
        long_is_a = g["long_code"] == g["code_a"] or (g["long_code"] != g["code_b"] and g["short_code"] == g["code_b"])
        sides = [("L", _w(g["long_desc"]), _w(g["long_name"])), ("S", _w(g["short_desc"]), _w(g["short_name"]))]
        if not long_is_a:
            sides.reverse()
        rows.append({"pm_slug": g["slug"].removeprefix("aec-"), "slug": g["slug"], "league": g["league"],
                     "pm_start": g["game_start"], "pm_code0": g["code_a"], "pm_code1": g["code_b"],
                     "pm_out0": sides[0][1], "pm_out1": sides[1][1], "pm_name0": sides[0][2], "pm_name1": sides[1][2],
                     "pm_tok0": sides[0][0], "pm_tok1": sides[1][0], "pm_settle": g["settlement"]})
    return pl.DataFrame(rows)


def match(kg: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """One row per matched Kalshi leg: its Polymarket US market and whether the Kalshi YES team is
    the long side (``long``; the leg's PM US YES price is the long price, else 1 minus it)."""
    from albert.research.refprice_maker import match_us

    m = match_us(kg, pm_frame(games))
    if not m.height:
        return m
    slug = pm_frame(games).select("pm_slug", "slug", "pm_settle")
    m = m.join(slug, on="pm_slug").with_columns((pl.col("pm_tok") == "L").alias("long"))
    # the Kalshi leg's own Polymarket US settlement (1 = that team won on Polymarket US)
    return m.with_columns(pl.when(pl.col("long")).then(pl.col("pm_settle")).otherwise(1 - pl.col("pm_settle"))
                          .alias("pm_y")).drop("pm_tok", "pm_tok_other")


def collect(s, start: datetime | None = None, end: datetime | None = None) -> dict[str, Any]:
    """Step 1: Polymarket US games and minute books for the record-vol nights (cached raw)."""
    start = start or datetime.fromisoformat(NIGHTS["wed"][0] + ":00+00:00")
    end = end or datetime.fromisoformat(NIGHTS["sun2"][1] + ":00+00:00")
    c = client(s)
    games = list_games(c, start, end).filter(pl.col("league").is_in(list(LEAGUES.values())))
    if not games.height:
        raise RuntimeError("no Polymarket US moneylines in the window")
    od = out_dir(s)
    _write(games, od / "pmus_games.parquet")
    print(f"pmus games: {games.height} ({dict(games.group_by('league').len().iter_rows())})", flush=True)
    hist = fetch_history(c, games)
    if not hist.height:
        raise RuntimeError("Polymarket US price history came back empty")
    _write(hist, od / "pmus_history.parquet")
    return {"games": games.height, "points": hist.height, "with_history": hist["slug"].n_unique()}


# ---- aligned series ---------------------------------------------------------------------------------
K_FEE = 0.07          # Kalshi taker, exact per fill (Kalshi rebates rounding over-payment)
MAX_PM_SPREAD = 0.05  # a Polymarket US point is a signal only when its book is this tight
MAX_K_SPREAD = 0.05


def k_fee(p):
    return K_FEE * p * (1 - p)


def night_of(ns: int) -> str | None:
    for tag, (a, b) in NIGHTS.items():
        if datetime.fromisoformat(a + ":00+00:00").timestamp() * 1e9 <= ns < datetime.fromisoformat(b + ":00+00:00").timestamp() * 1e9:
            return tag
    return None


def _night_bounds() -> list[tuple[int, int]]:
    return [(int(datetime.fromisoformat(a + ":00+00:00").timestamp() * 1e9),
             int(datetime.fromisoformat(b + ":00+00:00").timestamp() * 1e9)) for a, b in NIGHTS.values()]


class KBook:
    """Kalshi top of book of one market from the recorded quote changes, read as of a time."""

    def __init__(self, ns: np.ndarray, bid: np.ndarray, ask: np.ndarray,
                 bid_sz: np.ndarray | None = None, ask_sz: np.ndarray | None = None) -> None:
        o = np.argsort(ns, kind="stable")
        self.ns, self.bid, self.ask = ns[o], bid[o].astype(float), ask[o].astype(float)
        self.bid_sz = bid_sz[o].astype(float) if bid_sz is not None else None
        self.ask_sz = ask_sz[o].astype(float) if ask_sz is not None else None

    def sizes_at(self, t_ns: np.ndarray | int) -> tuple[np.ndarray, np.ndarray]:
        """Displayed size at the best >= 1-contract levels at ``t_ns``, floored to the size bucket
        it was last recorded in (sizes are stored when the bucket changes); NaN without sizes."""
        t = np.atleast_1d(np.asarray(t_ns, dtype=np.int64))
        if self.bid_sz is None:
            return np.full(t.shape, np.nan), np.full(t.shape, np.nan)
        i = np.maximum(np.searchsorted(self.ns, t, side="right") - 1, 0)
        floor = np.array((0.0, *SIZE_BUCKETS))
        fb = floor[np.searchsorted(floor, self.bid_sz[i], side="right") - 1]
        fa = floor[np.searchsorted(floor, self.ask_sz[i], side="right") - 1]
        b, a = self.at(t)
        return np.where(np.isfinite(b), fb, np.nan), np.where(np.isfinite(a), fa, np.nan)

    def at(self, t_ns: np.ndarray | int) -> tuple[np.ndarray, np.ndarray]:
        """(bid, ask) in force at ``t_ns`` (NaN before the first quote of the night or for an empty
        side). Quotes from an earlier night are not carried across the unrecorded day."""
        t = np.atleast_1d(np.asarray(t_ns, dtype=np.int64))
        i = np.searchsorted(self.ns, t, side="right") - 1
        ok = i >= 0
        b = np.full(t.shape, np.nan)
        a = np.full(t.shape, np.nan)
        b[ok], a[ok] = self.bid[i[ok]], self.ask[i[ok]]
        same = np.zeros(t.shape, dtype=bool)
        for lo, hi in _night_bounds():
            same |= (t >= lo) & (t < hi) & ok & (self.ns[np.maximum(i, 0)] >= lo)
        b[~same], a[~same] = np.nan, np.nan
        return b, a

    @property
    def last_ns(self) -> int:
        return int(self.ns[-1])


def load_kbooks(tickers: list[str], nights=tuple(NIGHTS)) -> dict[str, KBook]:
    from albert.research.pm_vol import HF_DIR

    want = set(tickers)
    frames = [pl.scan_parquet(HF_DIR / f"quotes_{n}.parquet").filter(pl.col("ticker").is_in(want)).collect()
              for n in nights]
    q = pl.concat(frames).sort("ticker", "ns")
    out = {}
    for (t,), g in q.group_by(["ticker"], maintain_order=True):
        out[t] = KBook(g["ns"].to_numpy(), g["bid"].to_numpy(), g["ask"].to_numpy())
    return out


def legs(matched: pl.DataFrame) -> pl.DataFrame:
    """One leg per game (the alphabetically first Kalshi ticker), so a game's two complementary
    books are never counted twice."""
    return matched.sort("ticker").unique("event", keep="first").sort("kickoff", "ticker")


def pm_leg_series(hist: pl.DataFrame, leg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(stamp_s, bid, ask) of the Kalshi leg's team on Polymarket US (complement for the short side)."""
    h = hist.filter(pl.col("slug") == leg["slug"]).sort("ts")
    ts, b, a = h["ts"].to_numpy(), h["bid"].to_numpy(), h["ask"].to_numpy()
    if not leg["long"]:
        b, a = 1 - a, 1 - b
    return ts, b, a


def aligned(hist: pl.DataFrame, lg: pl.DataFrame, books: dict[str, KBook], *, offsets_s=(0,),
            asof_s: float = ASOF_DELAY, latency_s: float = LATENCY, horizons_s=(60, 120, 300, 900)) -> pl.DataFrame:
    """One row per (leg, Polymarket US point): PM bid/ask at the stamp, its change over the
    previous minute, Kalshi bid/ask at stamp + each offset, at the decision time
    (stamp + asof + latency) and at decision + each horizon."""
    by_slug = {s: g for (s,), g in hist.group_by(["slug"])}
    rows = []
    for leg in lg.iter_rows(named=True):
        kb = books.get(leg["ticker"])
        h = by_slug.get(leg["slug"])
        if kb is None or h is None:
            continue
        h = h.sort("ts")
        ts, b, a = h["ts"].to_numpy(), h["bid"].to_numpy(), h["ask"].to_numpy()
        if not leg["long"]:
            b, a = 1 - a, 1 - b
        t_ns = ts.astype(np.int64) * 1_000_000_000
        d_ns = t_ns + int((asof_s + latency_s) * 1e9)
        rec: dict[str, Any] = {"ticker": leg["ticker"], "event": leg["event"], "league": leg["league"],
                               "kickoff": leg["kickoff"], "y": leg["y"], "ts": ts, "pm_bid": b, "pm_ask": a}
        prev = np.searchsorted(ts, ts - 60)
        has_prev = (prev < len(ts)) & (ts[np.minimum(prev, len(ts) - 1)] == ts - 60)
        pm_mid = (a + b) / 2
        rec["pm_dmid"] = np.where(has_prev, pm_mid - pm_mid[np.minimum(prev, len(ts) - 1)], np.nan)
        for off in offsets_s:
            kb_, ka_ = kb.at(t_ns + int(off * 1e9))
            rec[f"k_bid_o{off}"], rec[f"k_ask_o{off}"] = kb_, ka_
        kb_, ka_ = kb.at(d_ns)
        rec["k_bid_d"], rec["k_ask_d"] = kb_, ka_
        rec["k_bid_sz_d"], rec["k_ask_sz_d"] = kb.sizes_at(d_ns)
        for hz in horizons_s:
            kb_, ka_ = kb.at(d_ns + int(hz * 1e9))
            rec[f"k_bid_h{hz}"], rec[f"k_ask_h{hz}"] = kb_, ka_
            j = np.searchsorted(ts, ts + hz)
            ok = (j < len(ts)) & (ts[np.minimum(j, len(ts) - 1)] == ts + hz)
            rec[f"pm_mid_h{hz}"] = np.where(ok, pm_mid[np.minimum(j, len(ts) - 1)], np.nan)
        rec["k_last_ns"] = np.full(len(ts), kb.last_ns)
        rows.append(pl.DataFrame({k: (v if isinstance(v, np.ndarray) else [v] * len(ts)) for k, v in rec.items()}))
    df = pl.concat(rows, how="vertical_relaxed")
    return df.with_columns(
        pl.from_epoch("ts", time_unit="s").dt.replace_time_zone("UTC").alias("t"),
        ((pl.col("pm_bid") + pl.col("pm_ask")) / 2).alias("pm_mid"),
        (pl.col("pm_ask") - pl.col("pm_bid")).alias("pm_spread"),
    ).with_columns(
        pl.when(pl.col("t") >= pl.col("kickoff")).then(pl.lit("inplay")).otherwise(pl.lit("pregame")).alias("regime"),
        pl.col("ts").map_elements(lambda x: night_of(int(x) * 1_000_000_000), return_dtype=pl.String).alias("night"),
    )


def clustered(x: np.ndarray, groups: np.ndarray) -> tuple[float, float, int]:
    """Mean and its game-clustered standard error."""
    df = pl.DataFrame({"x": x, "g": groups}).filter(pl.col("x").is_finite())
    n = df.height
    if n < 2:
        return float("nan"), float("nan"), n
    mu = float(df["x"].mean())
    s = df.group_by("g").agg((pl.col("x") - mu).sum().alias("e"))["e"].to_numpy()
    G = len(s)
    se = math.sqrt((s ** 2).sum() * G / max(G - 1, 1)) / n
    return mu, se, n


def ols_clustered(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Slope of y on x (with intercept), game-clustered SE, and the correlation."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], groups[ok]
    n = len(x)
    if n < 10 or np.std(x) == 0:
        return {"n": n, "slope": float("nan"), "se": float("nan"), "t": float("nan"), "corr": float("nan")}
    xc, yc = x - x.mean(), y - y.mean()
    sxx = float((xc ** 2).sum())
    beta = float((xc * yc).sum()) / sxx
    e = yc - beta * xc
    sc = pl.DataFrame({"s": xc * e, "g": g}).group_by("g").agg(pl.col("s").sum())["s"].to_numpy()
    G = len(sc)
    se = math.sqrt((sc ** 2).sum() * G / max(G - 1, 1)) / sxx
    return {"n": n, "games": G, "slope": beta, "se": se, "t": beta / se if se > 0 else float("nan"),
            "corr": float(np.corrcoef(x, y)[0, 1])}


# ---- step 4: what a minute stamp means -------------------------------------------------------------
STAMP_OFFSETS = (-120, -90, -60, -45, -30, -20, -10, -5, 0, 5, 10, 20, 30, 45, 60, 90, 120)


def stamp_check(al: pl.DataFrame, offsets=STAMP_OFFSETS) -> pl.DataFrame:
    """For tight books on both venues, the mean |PM mid(t) - Kalshi mid(t + offset)| and the
    correlation of PM's minute change with Kalshi's minute change ending at t + offset. A point
    that is a snapshot at t fits best at offset 0; a bucket-end (last-in-minute) value near +60."""
    rows = []
    for regime in ("inplay", "pregame"):
        d = al.filter((pl.col("regime") == regime) & (pl.col("pm_spread") <= 0.03))
        for off in offsets:
            km = (d[f"k_bid_o{off}"] + d[f"k_ask_o{off}"]) / 2
            ok = (d[f"k_ask_o{off}"] - d[f"k_bid_o{off}"]) <= 0.03
            gap = (d["pm_mid"] - km).filter(ok & km.is_finite())
            # Kalshi minute change ending at t + off vs PM change over (t - 60, t]
            row = {"regime": regime, "offset_s": off, "n": gap.len(), "mean_abs_gap_c": 100 * float(gap.abs().mean()),
                   "rmse_c": 100 * math.sqrt(float((gap ** 2).mean()))}
            if off - 60 in offsets:
                km0 = (d[f"k_bid_o{off - 60}"] + d[f"k_ask_o{off - 60}"]) / 2
                dk = (km - km0).to_numpy()
                dp = d["pm_dmid"].to_numpy()
                ok2 = np.isfinite(dk) & np.isfinite(dp)
                row["corr_dmid"] = float(np.corrcoef(dp[ok2], dk[ok2])[0, 1]) if ok2.sum() > 10 else None
            rows.append(row)
    return pl.DataFrame(rows)


# ---- step 5a: gaps and two-venue crossing (informational) -----------------------------------------
def gap_stats(al: pl.DataFrame, off: int, pm_fee: float = 0.095) -> pl.DataFrame:
    """Kalshi mid - PM US mid at PM stamp + ``off`` (the stamp's measured meaning), and the best
    two-venue cost of a $1 payoff (YES on one venue + NO on the other, both taker fees)."""
    kb, ka = pl.col(f"k_bid_o{off}"), pl.col(f"k_ask_o{off}")
    d = al.filter(kb.is_finite() & ka.is_finite() & (pl.col("pm_spread") >= 0) & (ka > kb)).with_columns(
        ((kb + ka) / 2 - pl.col("pm_mid")).alias("gap"),
        (ka + K_FEE * ka * (1 - ka) + (1 - pl.col("pm_bid")) + pm_fee * pl.col("pm_bid") * (1 - pl.col("pm_bid"))).alias("c1"),
        ((1 - kb) + K_FEE * kb * (1 - kb) + pl.col("pm_ask") + pm_fee * pl.col("pm_ask") * (1 - pl.col("pm_ask"))).alias("c2"),
    ).with_columns(pl.min_horizontal("c1", "c2").alias("cost"))
    d = d.filter(((ka - kb) <= MAX_K_SPREAD) & (pl.col("pm_spread") <= MAX_PM_SPREAD))
    return d.group_by("regime", "league").agg(
        pl.len().alias("points"), pl.col("event").n_unique().alias("games"),
        (100 * pl.col("gap").abs().median()).alias("med_abs_gap_c"),
        (100 * pl.col("gap").abs().quantile(0.9)).alias("p90_abs_gap_c"),
        (100 * pl.col("gap").mean()).alias("mean_gap_c"),
        (100 * (pl.col("cost") - 1).median()).alias("med_two_venue_cost_over_1_c"),
        (pl.col("cost") < 1).mean().alias("share_cost_below_1"),
        (100 * (1 - pl.col("cost")).clip(0, None).max()).alias("max_gross_arb_c"),
    ).sort("regime", "league")


# ---- step 5b: lead-lag at minute resolution ---------------------------------------------------------
STAMP_S = 60   # measured (stamp_check): a point stamped t is the book at t + 60 s


def lead_lag(al: pl.DataFrame, horizons=(60, 120, 300, 900), stamp_s: int = STAMP_S) -> list[dict[str, Any]]:
    """At the decision time (stamp + as-of delay + latency, i.e. when the point's book is known):

    * ``pm_move->kalshi``: PM's last minute change vs Kalshi's mid change over the next ``h``;
    * ``kalshi_move->kalshi``: the same with Kalshi's own last minute change (control);
    * ``gap->kalshi``: PM mid - Kalshi mid vs Kalshi's next change (convergence toward PM);
    * ``kalshi_move->pm``: Kalshi's last minute change vs PM's next change (the reverse lead);
    * ``gap->pm``: Kalshi mid - PM mid vs PM's next change (PM converging toward Kalshi);
    * PM-move event studies: Kalshi's follow in cents, signed by the PM move.
    Requires Kalshi books at stamp offsets ``stamp_s - 60`` and ``stamp_s`` in ``al``."""
    out = []
    base = al.filter((pl.col("pm_spread") <= MAX_PM_SPREAD)
                     & ((pl.col("k_ask_d") - pl.col("k_bid_d")) <= MAX_K_SPREAD))
    for regime in ("pregame", "inplay"):
        d = base.filter(pl.col("regime") == regime)
        g = d["event"].to_numpy()
        kmd = ((d["k_bid_d"] + d["k_ask_d"]) / 2).to_numpy()
        dp = d["pm_dmid"].to_numpy()
        gap = d["pm_mid"].to_numpy() - kmd
        k_now = ((d[f"k_bid_o{stamp_s}"] + d[f"k_ask_o{stamp_s}"]) / 2).to_numpy()
        k_prev = ((d[f"k_bid_o{stamp_s - 60}"] + d[f"k_ask_o{stamp_s - 60}"]) / 2).to_numpy()
        dk_past = k_now - k_prev
        for hz in horizons:
            dk = ((d[f"k_bid_h{hz}"] + d[f"k_ask_h{hz}"]) / 2).to_numpy() - kmd
            out.append({"regime": regime, "test": "pm_move->kalshi", "h_s": hz, **ols_clustered(dp, dk, g)})
            out.append({"regime": regime, "test": "kalshi_move->kalshi", "h_s": hz, **ols_clustered(dk_past, dk, g)})
            out.append({"regime": regime, "test": "gap->kalshi", "h_s": hz, **ols_clustered(gap, dk, g)})
            dpm = d[f"pm_mid_h{hz}"].to_numpy() - d["pm_mid"].to_numpy()
            out.append({"regime": regime, "test": "kalshi_move->pm", "h_s": hz, **ols_clustered(dk_past, dpm, g)})
            out.append({"regime": regime, "test": "gap->pm (Kalshi - PM)", "h_s": hz, **ols_clustered(-gap, dpm, g)})
            for k in (0.01, 0.02, 0.03):
                big = np.isfinite(dp) & (np.abs(dp) >= k - 1e-9)
                mu, se, n = clustered(100 * np.sign(dp[big]) * dk[big], g[big])
                out.append({"regime": regime, "test": f"pm_move>={int(k * 100)}c follow_c", "h_s": hz,
                            "n": n, "games": len(set(g[big])), "mean": mu, "se": se,
                            "t": mu / se if se else float("nan")})
    return out


# ---- step 5c: Kalshi taker on divergence -----------------------------------------------------------
EDGES = (0.0, 0.01, 0.03)
EXITS = (60, 300, 900, None)   # None = hold to settlement
STAKE = 5.0                    # $ per position in the capped-book ledger
GROSS_CAP = 50.0               # $ open at once


def taker_trades(al: pl.DataFrame, edge: float, exit_s: int | None, regime: str) -> pl.DataFrame:
    """Take Kalshi when the as-of PM US mid is beyond Kalshi's touch by fee + ``edge``: buy YES at
    the ask if pm_mid - ask - fee >= edge, buy NO at 1 - bid if bid - pm_mid - fee >= edge. Exit
    by selling at the touch after ``exit_s`` (taker fee again) or hold to settlement; one position
    per market at a time. Per-contract results; sizing happens in :func:`ledger`."""
    d = al.filter((pl.col("regime") == regime) & (pl.col("pm_spread") <= MAX_PM_SPREAD)
                  & pl.col("k_bid_d").is_finite() & pl.col("k_ask_d").is_finite()
                  & (pl.col("k_ask_d") > pl.col("k_bid_d"))).sort("ticker", "ts")
    rows = []
    for (tk,), g in d.group_by(["ticker"], maintain_order=True):
        free_at = -1
        for r in g.iter_rows(named=True):
            dec = int(r["ts"]) * 1_000_000_000 + int((ASOF_DELAY + LATENCY) * 1e9)
            if dec < free_at or dec >= r["k_last_ns"]:
                continue
            kb, ka, pm = r["k_bid_d"], r["k_ask_d"], r["pm_mid"]
            if pm - ka - k_fee(ka) >= edge and ka < 1:
                side, px = "yes", ka
            elif kb - pm - k_fee(kb) >= edge and kb > 0:
                side, px = "no", 1 - kb
            else:
                continue
            cost = px + k_fee(px)
            won = r["y"] if side == "yes" else 1 - r["y"]
            if exit_s is None:
                out_ns, value, how = r["k_last_ns"], float(won), "settle"
            else:
                out_ns = dec + exit_s * 1_000_000_000
                eb, ea = r[f"k_bid_h{exit_s}"], r[f"k_ask_h{exit_s}"]
                sell = eb if side == "yes" else (1 - ea if math.isfinite(ea) else float("nan"))
                if out_ns >= r["k_last_ns"] or not math.isfinite(sell) or sell <= 0:
                    value, how = float(won), "settle"      # market gone or no bid: held to settlement
                    out_ns = max(out_ns, r["k_last_ns"])
                else:
                    value, how = sell - k_fee(sell), "exit"
            rows.append({"ticker": tk, "event": r["event"], "league": r["league"], "night": r["night"],
                         "t_in_ns": dec, "t_out_ns": out_ns, "side": side, "price": px, "cost": cost,
                         "value": value, "pnl": value - cost, "how": how, "pm_mid": pm,
                         "edge_at_entry": (pm - ka - k_fee(ka)) if side == "yes" else (kb - pm - k_fee(kb)),
                         # buying NO at 1 - bid lifts the YES bid's size
                         "entry_size": r["k_ask_sz_d"] if side == "yes" else r["k_bid_sz_d"]})
            free_at = out_ns
    return pl.DataFrame(rows, schema={"ticker": pl.String, "event": pl.String, "league": pl.String,
                                      "night": pl.String, "t_in_ns": pl.Int64, "t_out_ns": pl.Int64,
                                      "side": pl.String, "price": pl.Float64, "cost": pl.Float64,
                                      "value": pl.Float64, "pnl": pl.Float64, "how": pl.String,
                                      "pm_mid": pl.Float64, "edge_at_entry": pl.Float64, "entry_size": pl.Float64})


def per_dollar(tr: pl.DataFrame) -> dict[str, Any]:
    """PnL per $ risked (ratio of sums) with a game-clustered delta-method t."""
    if not tr.height:
        return {"n": 0, "games": 0, "pnl_per_usd_pct": None, "t": None, "c_per_contract": None}
    gsum = tr.group_by("event").agg(pl.col("pnl").sum(), pl.col("cost").sum())
    R = float(gsum["pnl"].sum() / gsum["cost"].sum())
    e = (gsum["pnl"] - R * gsum["cost"]).to_numpy()
    G = len(e)
    se = math.sqrt((e ** 2).sum() * G / max(G - 1, 1)) / float(gsum["cost"].sum()) if G > 1 else float("nan")
    return {"n": tr.height, "games": G, "pnl_per_usd_pct": 100 * R, "t": R / se if se else None,
            "c_per_contract": 100 * float(tr["pnl"].mean())}


def ledger(tr: pl.DataFrame, strategy: str, variant: str, *, stake: float = STAKE, cap: float = GROSS_CAP) -> pl.DataFrame:
    """Capped book on the $100 bankroll: ``stake`` dollars per position (whole contracts, at least
    one, at most the displayed size at entry when it is known), skipped while ``cap`` dollars
    are already open."""
    open_: list[tuple[int, float]] = []
    rows = []
    for r in tr.sort("t_in_ns").iter_rows(named=True):
        open_ = [(t, c) for t, c in open_ if t > r["t_in_ns"]]
        n = max(1, int(stake // r["cost"]))
        size = r.get("entry_size")
        if size is not None and math.isfinite(size):
            if size < 1:
                continue
            n = min(n, int(size))
        c = n * r["cost"]
        if sum(x for _, x in open_) + c > cap:
            continue
        open_.append((r["t_out_ns"], c))
        rows.append({"strategy": strategy, "variant": variant,
                     "split": "holdout" if r["night"] in HOLDOUT else "dev",
                     "struct_id": f"{r['ticker']}@{r['t_in_ns']}", "event_ticker": r["event"], "ticker": r["ticker"],
                     "side": r["side"], "n": n,
                     "t_in": datetime.fromtimestamp(r["t_in_ns"] / 1e9, UTC),
                     "cost_usd": c, "settle_ts": datetime.fromtimestamp(r["t_out_ns"] / 1e9, UTC),
                     "payoff_usd": n * r["value"]})
    return pl.DataFrame(rows)


def kalshi_marks(books: dict[str, KBook], tickers: list[str]) -> pl.DataFrame:
    """Hourly YES mids from the recorded Kalshi books (for vol_metrics' end-of-day marks)."""
    rows = []
    for t in tickers:
        kb = books.get(t)
        if kb is None:
            continue
        hours = np.arange((kb.ns[0] // 3_600_000_000_000 + 1) * 3_600_000_000_000, kb.ns[-1] + 1, 3_600_000_000_000)
        if not len(hours):
            continue
        b, a = kb.at(hours)
        for h, bb, aa in zip(hours, b, a, strict=True):
            if math.isfinite(bb) and math.isfinite(aa) and aa > bb:
                rows.append({"ticker": t, "ts": datetime.fromtimestamp(h / 1e9, UTC), "yes_mid": (aa + bb) / 2})
    return pl.DataFrame(rows, schema={"ticker": pl.String, "ts": pl.Datetime("us", "UTC"), "yes_mid": pl.Float64})


# ---- displayed size at the trade instants (targeted L2 replay) --------------------------------------
L2_SCHEMA = {"ticker": pl.String, "ns": pl.Int64, "clock": pl.String, "upd_ns": pl.Int64,
             "bid_raw": pl.Float64, "bid_raw_sz": pl.Float64, "ask_raw": pl.Float64, "ask_raw_sz": pl.Float64,
             "bid1": pl.Float64, "bid1_sz": pl.Float64, "ask1": pl.Float64, "ask1_sz": pl.Float64}


def _vol_runs(s) -> list[tuple[float, float, dict[str, str]]]:
    runs = []
    for mp in sorted((s.data_dir / "runs").glob("*-record-vol.json")):
        m = json.loads(mp.read_text())
        runs.append((datetime.fromisoformat(m["params"]["started_at"]).timestamp(),
                     datetime.fromisoformat(m["created_at"]).timestamp(),
                     {r["ticker"]: r["conn"] for r in m["params"]["kalshi_markets"]}))
    return runs


def book_tops(book: dict[str, dict[float, float]], min_size: float = 1.0) -> dict[str, float]:
    """Raw best levels (any quantity > 1e-9, what the quote table shows) and the best levels
    holding at least ``min_size`` contracts, with their sizes. YES ask = 1 - best NO bid."""
    out: dict[str, float] = {}
    for side, key in (("yes", "bid"), ("no", "ask")):
        lv = book.get(side) or {}
        raw = max(lv, default=None)
        real = max((p for p, q in lv.items() if q >= min_size), default=None)
        conv = (lambda p: p) if side == "yes" else (lambda p: round(1 - p, 4))
        out[f"{key}_raw"] = conv(raw) if raw is not None else float("nan")
        out[f"{key}_raw_sz"] = lv[raw] if raw is not None else 0.0
        out[f"{key}1"] = conv(real) if real is not None else float("nan")
        out[f"{key}1_sz"] = lv[real] if real is not None else 0.0
    return out


def l2_at(s, queries: pl.DataFrame, *, log=print) -> pl.DataFrame:
    """Kalshi book of ``queries`` (ticker, ns) from the raw ``record-vol`` stream, replaying only the
    connection-hours that held those tickers, on two clocks:

    * ``clock="recv"``: every frame received by our recorder before ``ns`` (what the quote table
      and the recorder saw; it lags the exchange when the recorder falls behind);
    * ``clock="server"``: every delta whose exchange timestamp (``ts_ms``, else the frame's
      ``sending_ts_ms``) is before ``ns``: the book an unlagged client would see and trade against.

    Each recorded hour opens with fresh snapshots; answers before a ticker's snapshot are left out."""
    import io
    import re

    import zstandard

    from albert.research.pm_vol import _ESC_TICKER

    ts_rx = re.compile(r'"ts_ms\\?":(\d+)')
    send_rx = re.compile(r'sending_ts_ms\\?":(\d+)')
    runs = _vol_runs(s)
    q = queries.select("ticker", "ns").unique().with_columns((pl.col("ns") // 3_600_000_000_000).alias("hr"))
    want: dict[tuple[int, str], set[str]] = defaultdict(set)
    for t, hr in q.select("ticker", "hr").unique().iter_rows():
        for t0, t1, mp in runs:
            if t in mp and t0 < (hr + 1) * 3600 and t1 >= hr * 3600:
                want[(hr, mp[t])].add(t)
    by_t: dict[tuple[int, str], list[int]] = defaultdict(list)
    for t, ns, hr in q.iter_rows():
        by_t[(hr, t)].append(ns)
    root = s.raw_dir / "stream" / "kalshi"
    rows: list[dict[str, Any]] = []
    for i, ((hr, conn), tickers) in enumerate(sorted(want.items())):
        h = datetime.fromtimestamp(hr * 3600, UTC)
        day = root / h.strftime("%Y-%m-%d")
        files = sorted(day.glob(f"{h:%H}-{conn}.jsonl.zst")) + sorted(day.glob(f"{h:%H}-{conn}-*.jsonl.zst"))
        for path in files:
            books: dict[str, dict[str, dict[float, float]]] = {}
            upd: dict[str, int] = {}
            pending = {(t, c): sorted(by_t[(hr, t)]) for t in tickers for c in ("recv", "server")}
            pat = re.compile("|".join(re.escape(t) for t in tickers))

            def answer(t: str, clock: str, upto: int) -> None:
                pend = pending[(t, clock)]
                while pend and pend[0] < upto:
                    ns = pend.pop(0)
                    if t in books:
                        rows.append({"ticker": t, "ns": ns, "clock": clock, "upd_ns": upd[t], **book_tops(books[t])})

            with open(path, "rb") as f:
                for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8"):
                    if "GAME-" not in line or not pat.search(line):
                        continue
                    mm = _ESC_TICKER.search(line)
                    if mm is None or mm.group(1) not in tickers:
                        continue
                    t = mm.group(1)
                    row = json.loads(line)
                    recv = row["recv_ns"]
                    sm = ts_rx.search(line) or send_rx.search(line)
                    srv = int(sm.group(1)) * 1_000_000 if sm else None
                    answer(t, "recv", recv)
                    if srv is not None:
                        answer(t, "server", srv)
                    d = json.loads(row["frame"])
                    msg = d.get("msg") or {}
                    if d.get("type") == "orderbook_snapshot":
                        books[t] = {sd: {float(px): float(qq) for px, qq in (msg.get(f"{sd}_dollars_fp") or []) if float(qq) > 0}
                                    for sd in ("yes", "no")}
                    elif d.get("type") == "orderbook_delta" and t in books:
                        lv = books[t][msg["side"]]
                        px = float(msg["price_dollars"])
                        qq = lv.get(px, 0.0) + float(msg["delta_fp"])
                        if qq > 1e-9:
                            lv[px] = qq
                        else:
                            lv.pop(px, None)
                    else:
                        continue
                    upd[t] = recv
            for t in tickers:
                for c in ("recv", "server"):
                    answer(t, c, 1 << 62)
        if log and (i + 1) % 10 == 0:
            log(f"  l2_at {i + 1}/{len(want)} conn-hours, {len(rows)} answers", flush=True)
    out = pl.DataFrame(rows, schema=L2_SCHEMA)
    # a ticker can sit on two files of one hour (a recorder restart): keep the latest book
    return out.sort("upd_ns").unique(["ticker", "ns", "clock"], keep="last").sort("ticker", "ns", "clock")


SQ_SCHEMA = {"srv_ns": pl.Int64, "recv_ns": pl.Int64, "ticker": pl.String, "bid": pl.Float64, "ask": pl.Float64,
             "bid_sz": pl.Float64, "ask_sz": pl.Float64, "bid_raw": pl.Float64, "ask_raw": pl.Float64}


SIZE_BUCKETS = (1.0, 5.0, 20.0, 100.0)


def size_bucket(q: float) -> int:
    """Index of the largest of :data:`SIZE_BUCKETS` that ``q`` reaches (-1 below one contract)."""
    return sum(q >= b for b in SIZE_BUCKETS) - 1


def _fast_grep() -> bool:
    import shutil

    return bool(shutil.which("zstd") and shutil.which("grep"))


def _lines(path: Path, patfile: Path | None):
    """Lines of a zstd JSONL file; with ``patfile``, only those containing one of its fixed strings
    (``zstd -dc | grep -F -f``, many times faster than scanning every line in Python)."""
    import io
    import subprocess

    import zstandard

    if patfile is None:
        with open(path, "rb") as f:
            yield from io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8")
        return
    zs = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE)
    gr = subprocess.Popen(["grep", "-F", "-f", str(patfile)], stdin=zs.stdout, stdout=subprocess.PIPE, text=True)
    zs.stdout.close()
    try:
        yield from gr.stdout
    finally:
        gr.stdout.close()
        gr.wait()
        zs.wait()


def server_quotes(s, tickers: list[str], start: str, end: str, *, log=print) -> pl.DataFrame:
    """Top of book of ``tickers`` on the exchange clock: one row per change of the best levels
    holding >= 1 contract (``bid``/``ask``), of the raw best levels, or of the displayed size's
    bucket (:data:`SIZE_BUCKETS`; the exact size at that moment is stored), stamped with
    the frame's exchange time (delta ``ts_ms``, snapshot ``sending_ts_ms``) and our receive time.
    Replays only the connection-hours of ``[start, end)`` (``YYYY-MM-DDTHH``) that held a ticker;
    books carry across hours and a snapshot resets one."""
    import re
    import tempfile

    from albert.research.pm_vol import _ESC_TICKER

    patfile = None
    if _fast_grep():
        patfile = Path(tempfile.mkstemp(prefix="pmus_xvenue_", suffix=".txt")[1])
        patfile.write_text("\n".join(sorted(set(tickers))) + "\n")
    ts_rx = re.compile(r'"ts_ms\\?":(\d+)')
    send_rx = re.compile(r'sending_ts_ms\\?":(\d+)')
    want = set(tickers)
    runs = _vol_runs(s)
    h0 = int(datetime.fromisoformat(start + ":00:00+00:00").timestamp()) // 3600
    h1 = int(datetime.fromisoformat(end + ":00:00+00:00").timestamp()) // 3600
    root = s.raw_dir / "stream" / "kalshi"
    books: dict[str, dict[str, dict[float, float]]] = {}
    last: dict[str, tuple] = {}
    cols: dict[str, list] = {k: [] for k in SQ_SCHEMA}
    parts: list[pl.DataFrame] = []
    total = 0
    for hr in range(h0, h1):
        conns: dict[str, set[str]] = defaultdict(set)
        for t0, t1, mp in runs:
            if t0 < (hr + 1) * 3600 and t1 >= hr * 3600:
                for t, c in mp.items():
                    if t in want:
                        conns[c].add(t)
        h = datetime.fromtimestamp(hr * 3600, UTC)
        day = root / h.strftime("%Y-%m-%d")
        for conn, held in sorted(conns.items()):
            files = sorted(day.glob(f"{h:%H}-{conn}.jsonl.zst")) + sorted(day.glob(f"{h:%H}-{conn}-*.jsonl.zst"))
            for path in files:
                for line in _lines(path, patfile):
                    if "GAME-" not in line:
                        continue
                    mm = _ESC_TICKER.search(line)
                    if mm is None or mm.group(1) not in held:
                        continue
                    t = mm.group(1)
                    row = json.loads(line)
                    d = json.loads(row["frame"])
                    msg = d.get("msg") or {}
                    typ = d.get("type")
                    if typ == "orderbook_snapshot":
                        books[t] = {sd: {float(px): float(q) for px, q in (msg.get(f"{sd}_dollars_fp") or []) if float(q) > 0}
                                    for sd in ("yes", "no")}
                    elif typ == "orderbook_delta" and t in books:
                        lv = books[t][msg["side"]]
                        px = float(msg["price_dollars"])
                        q = lv.get(px, 0.0) + float(msg["delta_fp"])
                        if q > 1e-9:
                            lv[px] = q
                        else:
                            lv.pop(px, None)
                    else:
                        continue
                    sm = ts_rx.search(line) or send_rx.search(line)
                    top = book_tops(books[t])
                    key = (top["bid1"], top["ask1"], top["bid_raw"], top["ask_raw"],
                           size_bucket(top["bid1_sz"]), size_bucket(top["ask1_sz"]))
                    if last.get(t) == key:
                        continue
                    last[t] = key
                    cols["srv_ns"].append(int(sm.group(1)) * 1_000_000 if sm else row["recv_ns"])
                    cols["recv_ns"].append(row["recv_ns"])
                    cols["ticker"].append(t)
                    for k, v in (("bid", "bid1"), ("ask", "ask1"), ("bid_sz", "bid1_sz"), ("ask_sz", "ask1_sz"),
                                 ("bid_raw", "bid_raw"), ("ask_raw", "ask_raw")):
                        cols[k].append(top[v])
        parts.append(pl.DataFrame(cols, schema=SQ_SCHEMA))   # flush hourly: columnar is ~10x smaller
        total += parts[-1].height
        cols = {k: [] for k in SQ_SCHEMA}
        if log:
            log(f"  server_quotes {h:%Y-%m-%dT%H}: {total:,} rows", flush=True)
    if patfile is not None:
        patfile.unlink()
    return pl.concat(parts).sort("ticker", "srv_ns")


def load_server_books(s, tickers: list[str], nights=tuple(NIGHTS)) -> dict[str, KBook]:
    """KBooks on the exchange clock from :func:`server_quotes` output (executable >= 1-contract
    levels), plus the displayed sizes (``KBook.bid_sz``/``ask_sz``)."""
    want = set(tickers)
    q = pl.concat([pl.scan_parquet(out_dir(s) / f"kq_server_{n}.parquet").filter(pl.col("ticker").is_in(want)).collect()
                   for n in nights]).sort("ticker", "srv_ns")
    out = {}
    for (t,), g in q.group_by(["ticker"], maintain_order=True):
        out[t] = KBook(g["srv_ns"].to_numpy(), g["bid"].to_numpy(), g["ask"].to_numpy(),
                       g["bid_sz"].to_numpy(), g["ask_sz"].to_numpy())
    return out


def size_checked(tr: pl.DataFrame, l2: pl.DataFrame, *, min_size: float = 1.0) -> pl.DataFrame:
    """Re-price trades on the L2 book: entry at the best level holding >= ``min_size`` contracts
    (YES at ask1, NO at 1 - bid1), exit likewise; the signal must still clear its edge at the
    real touch. Adds the displayed size at entry."""
    e = l2.select("ticker", pl.col("ns").alias("t_in_ns"), pl.col("bid1").alias("e_bid"), pl.col("ask1").alias("e_ask"),
                  pl.col("bid1_sz").alias("e_bid_sz"), pl.col("ask1_sz").alias("e_ask_sz"),
                  pl.col("bid_raw").alias("e_bid_raw"), pl.col("ask_raw").alias("e_ask_raw"))
    x = l2.select("ticker", pl.col("ns").alias("t_out_ns"), pl.col("bid1").alias("x_bid"), pl.col("ask1").alias("x_ask"))
    d = tr.join(e, on=["ticker", "t_in_ns"], how="left").join(x, on=["ticker", "t_out_ns"], how="left")
    px = pl.when(pl.col("side") == "yes").then(pl.col("e_ask")).otherwise(1 - pl.col("e_bid"))
    sz = pl.when(pl.col("side") == "yes").then(pl.col("e_ask_sz")).otherwise(pl.col("e_bid_sz"))
    sell = pl.when(pl.col("side") == "yes").then(pl.col("x_bid")).otherwise(1 - pl.col("x_ask"))
    d = d.with_columns(px.alias("px_l2"), sz.alias("entry_size"), sell.alias("sell_l2"))
    edge_l2 = pl.when(pl.col("side") == "yes").then(pl.col("pm_mid") - pl.col("px_l2") - K_FEE * pl.col("px_l2") * (1 - pl.col("px_l2"))) \
        .otherwise((1 - pl.col("px_l2")) - pl.col("pm_mid") - K_FEE * pl.col("px_l2") * (1 - pl.col("px_l2")))
    value = pl.when(pl.col("how") == "settle").then(pl.col("value")) \
        .otherwise(pl.col("sell_l2") - K_FEE * pl.col("sell_l2") * (1 - pl.col("sell_l2")))
    cost = pl.col("px_l2") + K_FEE * pl.col("px_l2") * (1 - pl.col("px_l2"))
    return d.with_columns(edge_l2.alias("edge_l2"), cost.alias("cost_l2"), value.alias("value_l2")).with_columns(
        (pl.col("value_l2") - pl.col("cost_l2")).alias("pnl_l2"),
        (pl.col("px_l2") - pl.col("price")).abs().gt(1e-6).alias("dust_at_entry"))


# ---- step 5d (lite): who is more accurate before the start ------------------------------------------
def pregame_accuracy(al: pl.DataFrame, leads_min=(60, 15, 1), stamp_s: int = STAMP_S) -> list[dict[str, Any]]:
    """Brier of the PM US mid vs the Kalshi mid at the same instant (the point's book time), at
    the last point at least ``lead`` minutes before the start (paired by game; a positive diff
    means PM US is less accurate)."""
    out = []
    kb, ka = f"k_bid_o{stamp_s}", f"k_ask_o{stamp_s}"
    d = al.filter((pl.col("pm_spread") <= MAX_PM_SPREAD) & pl.col(kb).is_finite() & pl.col(ka).is_finite())
    for lead in leads_min:
        x = (d.filter(pl.col("t") + pl.duration(seconds=stamp_s) <= pl.col("kickoff") - pl.duration(minutes=lead))
             .sort("t").group_by("event").last()
             .with_columns(((pl.col(kb) + pl.col(ka)) / 2).alias("km")))
        if not x.height:
            continue
        bp = ((x["pm_mid"] - x["y"]) ** 2).to_numpy()
        bk = ((x["km"] - x["y"]) ** 2).to_numpy()
        diff = bp - bk
        se = float(np.std(diff, ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
        out.append({"lead_min": lead, "games": x.height, "brier_pmus": float(bp.mean()), "brier_kalshi": float(bk.mean()),
                    "diff": float(diff.mean()), "t": float(diff.mean()) / se if se else None,
                    "mean_abs_gap_c": 100 * float((x["pm_mid"] - x["km"]).abs().mean())})
    return out


# ---- run and report ------------------------------------------------------------------------------
ALIGN_OFFSETS = (-120, -90, -60, -45, -30, -20, -10, -5, 0, 5, 10, 20, 30, 45, 50, 55, 58, 60, 62, 65, 70, 90, 120)


def _fmt(v: Any, k: int = 2) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return f"{v:.{k}f}" if isinstance(v, float) else str(v)


def _md(df: pl.DataFrame, k: int = 2) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        lines.append("| " + " | ".join(_fmt(v, k) for v in r) + " |")
    return "\n".join(lines)


def analyse(s, books: dict[str, KBook], lg: pl.DataFrame, hist: pl.DataFrame, clock: str) -> dict[str, Any]:
    """Every step-4/5 analysis on one Kalshi clock; trades and ledgers written under ``out_dir``."""
    from albert.research.vol_metrics import ledger_metrics

    od = out_dir(s)
    al = aligned(hist, lg, books, offsets_s=ALIGN_OFFSETS).filter(pl.col("night").is_not_null())
    _write(al, od / f"aligned_{clock}.parquet")
    res: dict[str, Any] = {"points": al.height, "legs": al["ticker"].n_unique(),
                           "stamp": stamp_check(al, offsets=ALIGN_OFFSETS), "gaps": gap_stats(al, STAMP_S),
                           "lead_lag": pl.DataFrame(lead_lag(al)), "accuracy": pl.DataFrame(pregame_accuracy(al))}
    marks = kalshi_marks(books, lg["ticker"].to_list())
    grid, metrics = [], []
    for regime in ("pregame", "inplay"):
        for edge in EDGES:
            for ex in EXITS:
                variant = f"{regime}_e{int(edge * 100)}_x{ex or 'settle'}"
                tr = taker_trades(al, edge, ex, regime)
                _write(tr, od / "trades" / f"{clock}_{variant}.parquet")
                for split, nights in (("all", tuple(NIGHTS)), ("dev", DEV), ("holdout", HOLDOUT)):
                    grid.append({"regime": regime, "edge_c": int(edge * 100), "exit": str(ex or "settle"), "split": split,
                                 **per_dollar(tr.filter(pl.col("night").is_in(list(nights))))})
                led = ledger(tr, f"pmus_taker_{clock}", variant)
                if led.height:
                    _write(led, od / "ledgers" / f"ledger_{clock}_{variant}.parquet")
                    m = ledger_metrics(led, marks)
                    metrics.append({"variant": variant, "positions": m["positions"], "usd_per_week": m["usd_per_week"],
                                    "ann_vol_pct": m["ann_vol_pct"], "sharpe": m["sharpe"],
                                    "sharpe_ci90": str(m["sharpe_ci90"]) if m["sharpe_ci90"] else "n/a (<14 days)",
                                    "sharpe_ex_top5": m["sharpe_ex_top5"], "max_dd_usd": m["max_dd_usd"],
                                    "days": m["days"]})
    res["grid"] = pl.DataFrame(grid)
    res["metrics"] = pl.DataFrame(metrics)
    return res


def run(s) -> dict[str, Any]:
    """The study on both Kalshi clocks (exchange time = primary; recorder receive time = what the
    quote table shows, kept to measure the lag bias); writes the report."""
    od = out_dir(s)
    matched = pl.read_parquet(od / "matched.parquet")
    lg = legs(matched)
    hist = pl.read_parquet(od / "pmus_history.parquet")
    out = {"server": analyse(s, load_server_books(s, lg["ticker"].to_list()), lg, hist, "server"),
           "recv": analyse(s, load_kbooks(lg["ticker"].to_list()), lg, hist, "recv")}
    for clock, r in out.items():
        for k in ("stamp", "gaps", "lead_lag", "accuracy", "grid", "metrics"):
            _write(r[k], od / f"{k}_{clock}.parquet")
    return out


def write_report(s, out: dict[str, Any], extra: str = "") -> Path:
    from albert.store import write_manifest

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rdir = s.data_dir / "reports" / f"research-pmus-xvenue-{stamp}"
    rdir.mkdir(parents=True, exist_ok=True)
    sv, rc = out["server"], out["recv"]
    ll_keep = ["pm_move->kalshi", "kalshi_move->kalshi", "gap->kalshi", "gap->pm (Kalshi - PM)", "kalshi_move->pm"]

    def ll_table(r):
        d = r["lead_lag"].filter(pl.col("test").is_in(ll_keep)).select("regime", "test", "h_s", "n", "slope", "t", "corr")
        e = r["lead_lag"].filter(pl.col("test").str.contains("follow")).select("regime", "test", "h_s", "n", "games", "mean", "t")
        return _md(d, 3), _md(e, 3)

    sv_ll, sv_ev = ll_table(sv)
    rc_ll, rc_ev = ll_table(rc)
    lines = [
        "# Polymarket US as a signal for Kalshi game moneylines", "",
        f"Generated {stamp}. Kalshi execution only; Polymarket US is data (public gateway, no key).", "",
        extra, "",
        "## Timestamp semantics (exchange clock)", "",
        _md(sv["stamp"].filter(pl.col("regime") == "inplay").select("offset_s", "n", "mean_abs_gap_c", "rmse_c", "corr_dmid"), 3), "",
        "## Gaps and two-venue crossing (exchange clock, at the point's book time)", "",
        _md(sv["gaps"], 3), "",
        "## Lead-lag at minute resolution (exchange clock)", "", sv_ll, "", sv_ev, "",
        "## Lead-lag on the recorder's receive clock (biased by recorder lag; for comparison)", "", rc_ll, "", rc_ev, "",
        "## Pre-game accuracy (exchange clock)", "", _md(sv["accuracy"], 4), "",
        "## Kalshi taker on divergence: exchange clock (primary)", "",
        _md(sv["grid"].select("regime", "edge_c", "exit", "split", "n", "games", "pnl_per_usd_pct", "t", "c_per_contract")), "",
        "### Capped-book ledgers ($5/position, $50 gross cap, displayed size; vol_metrics MTM daily)", "",
        _md(sv["metrics"]), "",
        "## Same grid on the receive clock (biased; for comparison)", "",
        _md(rc["grid"].filter(pl.col("split") == "all").select("regime", "edge_c", "exit", "n", "games", "pnl_per_usd_pct", "t")), "",
    ]
    (rdir / "report.md").write_text("\n".join(lines) + "\n")
    write_manifest(s.data_dir, "research pmus-xvenue", {"nights": NIGHTS, "asof_s": ASOF_DELAY, "latency_s": LATENCY,
                                                        "edges": EDGES, "exits": EXITS, "stake": STAKE, "cap": GROSS_CAP},
                   {"points_server": sv["points"], "legs": sv["legs"]}, [])
    return rdir / "report.md"


def recorder_lag(s) -> pl.DataFrame:
    """Recorder receive time minus exchange time of every server-clock quote row, by night and
    regime-agnostic hour bucket (the bias that makes Kalshi look stale on the receive clock)."""
    rows = []
    for n in NIGHTS:
        q = pl.read_parquet(out_dir(s) / f"kq_server_{n}.parquet", columns=["srv_ns", "recv_ns"])
        lag = ((q["recv_ns"] - q["srv_ns"]) / 1e6).to_numpy()
        rows.append({"night": n, "rows": len(lag), **{f"p{p}_ms": float(np.percentile(lag, p)) for p in (50, 90, 99)},
                     "share_over_1s": float((lag > 1000).mean())})
    return pl.DataFrame(rows)


def reprice_table(s) -> pl.DataFrame:
    """The receive-clock in-play taker trades (edge >= 1c) re-priced on the L2 book at the same
    instants on both clocks (best level holding >= 1 contract): share entering at the same price,
    PnL per $ as simulated vs re-priced, and how many still clear their edge at the real touch."""
    od = out_dir(s)
    l2 = pl.read_parquet(od / "l2_at_inplay_edge.parquet")
    rows = []
    for edge in (1, 3):
        for ex in ("60", "300", "900", "settle"):
            p = od / "trades" / f"recv_inplay_e{edge}_x{ex}.parquet"
            if not p.exists():
                continue
            tr = pl.read_parquet(p)
            for clock in ("recv", "server"):
                sc = size_checked(tr, l2.filter(pl.col("clock") == clock))
                ok = sc.filter(pl.col("px_l2").is_finite() & pl.col("pnl_l2").is_finite())
                still = ok.filter(pl.col("edge_l2") >= edge / 100)
                a = per_dollar(ok.select("event", "pnl", "cost"))
                b = per_dollar(ok.select("event", pl.col("pnl_l2").alias("pnl"), pl.col("cost_l2").alias("cost")))
                c = per_dollar(still.select("event", pl.col("pnl_l2").alias("pnl"), pl.col("cost_l2").alias("cost")))
                rows.append({"edge_c": edge, "exit": ex, "book_clock": clock, "trades": ok.height,
                             "same_entry_px": float((~ok["dust_at_entry"]).mean()) if ok.height else None,
                             "sim_pct": a["pnl_per_usd_pct"], "sim_t": a["t"], "l2_pct": b["pnl_per_usd_pct"], "l2_t": b["t"],
                             "still_signal": still.height, "still_pct": c["pnl_per_usd_pct"], "still_t": c["t"]})
    return pl.DataFrame(rows)


def extract_server(s, nights=tuple(NIGHTS)) -> dict[str, int]:
    """Server-clock quotes of every matched leg-0 ticker, one Parquet per night."""
    lg = legs(pl.read_parquet(out_dir(s) / "matched.parquet"))
    tickers = lg["ticker"].to_list()
    out = {}
    for n in nights:
        a, b = NIGHTS[n]
        q = server_quotes(s, tickers, a, b)
        if not q.height:
            raise RuntimeError(f"no server-clock quotes for {n}")
        _write(q, out_dir(s) / f"kq_server_{n}.parquet")
        out[n] = q.height
        print(f"{n}: {q.height:,} rows, {q['ticker'].n_unique()} tickers", flush=True)
    return out


if __name__ == "__main__":
    import sys

    from albert.config import settings

    if sys.argv[1:2] == ["collect"]:
        print(json.dumps(collect(settings()), default=str))
    elif sys.argv[1:2] == ["extract-server"]:
        print(json.dumps(extract_server(settings(), tuple(sys.argv[2:]) or tuple(NIGHTS))))
    elif sys.argv[1:2] == ["run"]:
        s = settings()
        res = run(s)
        extra = "\n".join([
            "## Data", "", _md(pl.read_parquet(out_dir(s) / "coverage.parquet"), 3), "",
            "## Recorder receive lag (Kalshi frames: our receive time - exchange timestamp)", "",
            _md(recorder_lag(s), 1), "",
            "## Receive-clock in-play trades re-priced on the L2 book (>= 1 contract)", "",
            _md(reprice_table(s), 2)])
        print(write_report(s, res, extra))
