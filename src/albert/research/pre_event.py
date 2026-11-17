"""Pre-event NO bids beyond Mentions: does selling YES above fair before a scheduled start pay in
other scheduled-event markets?

Question (user, 2026-11-17). The one robust result so far (``mentions_history``): resting NO bids on
Mentions speech legs at a bias-corrected fair value + 8-12c, from 6 h before the scheduled speech
to start - 10 min, filled only by public prints that trade through the bid, held to settlement:
+5.8c per fill (t 5.4), a price pattern (retail buys YES into the run-up of a scheduled event).
Does the same rule pay on sports player props, game lines and winners, esports, entertainment
broadcasts, politics and elections with scheduled times? The mirror (YES bids below fair) is the
control.

Frozen rule (``mentions_history``; every input as of the decision time ``t``):

* **Start** = the scheduled start. Sports: the time in the event ticker (``kalshi.ticker_start``)
  when the ticker carries one, else the earliest Kalshi ``/milestones`` ``start_date`` listing the
  event (types football_game, basketball_game, hockey_match, baseball_game,
  soccer_tournament_multi_leg). Mentions, Entertainment, Politics, Elections, Science and
  Technology: the earliest milestone of the category listing the event. Never close or settlement
  time. (Milestones are read today; ``ms_known`` flags a start last edited before ``t``.)
* ``t = start - 6 h``; the leg must be listed and open at ``t``. Quote = the last hourly candle
  ending at or before ``t`` (Kalshi omits a candle when nothing changed, so the quote is carried for
  up to 24 h). Outcome-blind early-start exclusion: drop an event whose share of quoted legs bid
  >= 97c at ``t`` exceeds the share at ``start - 24 h`` by > 12 points.
* **Fair value**: per calibration group ``logit fair = a + b logit(mid)`` fitted on the group's dev
  rows (``t`` before 2026-10-01); Mentions keep the frozen round-2 fit (a -0.1156, b 1.0474).
  Control: the raw mid.
* **Orders**: NO bid ``floor(1 - fair - m)`` capped one tick inside the bid, ``m`` in {8c, 12c};
  mirror: YES bid ``floor(fair - m)`` capped one tick under the ask; bids under 1c are not posted.
  One contract per order, and a $5 version (``n = floor(5 / price)``); the maker fee (0.0175 x
  multiplier x n p (1 - p), maker-fee series only) is rounded up to the cent per order in both.
  Orders rest from ``t`` to ``end = min(start - 10 min, close)``.
* **Circuit breaker**: cancel the event's orders after the first print >= 95c on a leg asked <= 80c
  at ``t``; fills in that print's minute still count (about a minute of reaction time).
* **Fills**: a bid fills only if a later public print trades strictly through it. Source: one-minute
  candles' trade high / low (a print above our ask implies the book could not hold a bid above it
  with our offer resting, so we would have been hit), and, for Mentions, the public prints that
  ``mentions_history`` cached (taker side checked) - the reproduction and the validation of the
  candle rule. Held to settlement.

Data (public Kalshi API through ``pm_hazard._api``: cached, append-only raw store; ~2,550 requests
in all): settled markets from ``data/tables/kalshi_markets.parquet`` (closes 2026-08-15..10-31);
milestones; hourly candles over ``[start - 30 h, start]`` (live batch endpoint, <= 100 tickers and
<= 10,000 candles per request; the ``xcat`` whole-life hourly layer is reused for Mentions /
Elections / Science / its Entertainment events); one-minute candles over ``[t, end]`` only for legs
whose hourly trade range could hold a fill or a breaker print at the most aggressive level (every
other leg provably did not fill: the hourly high / low bound every print). Sports legs settled
before the historical cutoff (2026-09-18) would need one request per leg, so sports use the live
era only (Sep 18 - Oct 31), sampled volume- and result-blind: games in seeded-hash order up to a leg
target per group and split, at most 30 (props) / 20 (lines) legs per game. Politics has 3 timed
events (debates and pressers live in Mentions).

Splits: dev = ``t`` before 2026-10-01, holdout = October; t-stats clustered by event, by game
(several sports events share one) and by ET date of the start. Capped-book ledgers ($100, $5 orders,
$10 per game) go to ``ledgers/`` and are scored with ``vol_metrics.ledger_metrics`` (daily marks at
the hourly mid).

Added after the dev / holdout tables (labelled as such in the outputs):

* ``sports_tight``: sports legs whose quoted spread at ``t`` is <= 15c (the spread band is the
  ``mentions_history`` robustness cut; the pooled cut was looked at on both splits at once).
* ``--forward``: a November check (games starting 11-01..11-16; a seeded half of the main-league
  and top-soccer series, every Mentions / Entertainment event with a November milestone) with the
  Aug-Sep calibrations frozen, in ``OUT/forward``.

    python -m albert.research.pre_event milestones|universe|hourly|rows|minute|score [--forward]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.research import xcat

OUT = Path("data/research/pre_event")
TABLES = Path("data/tables")
TAG = "pre_event"                    # mixed into the cache key of listing requests (stable re-runs)
ET = "America/New_York"

LEAD, H24, END_MARGIN = timedelta(hours=6), timedelta(hours=24), timedelta(minutes=10)
QUOTE_AGE = timedelta(hours=24)
HOURLY_SPAN = timedelta(hours=30)
MARGINS = (0.08, 0.12)
CB_PRINT, CB_ASK = 0.95, 0.80
SAID_BID, SAID_JUMP = 0.97, 0.12
FROZEN_A, FROZEN_B = -0.1156, 1.0474
STAKE = 5.0
MAKER = 0.0175
EVENT_CAP = 10.0
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
DATA_FROM, DATA_TO = datetime(2026, 8, 15, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC)
MS_FROM = datetime(2026, 8, 13, tzinfo=UTC)
MAX_BATCH_CANDLES, MAX_BATCH_TICKERS = 10_000, 100
LEG_CAP = 40
FROZEN_CAL = OUT / "rows_info.json"
# forward check (added after the Aug-Oct results): games starting 2026-11-01..11-16, sports series of
# the main leagues and top soccer (a seeded-hash half of them), Mentions and Entertainment events
FWD_FROM, FWD_TO = datetime(2026, 11, 1, tzinfo=UTC), datetime(2026, 11, 16, 12, tzinfo=UTC)
FWD_PREFIXES = ("KXNFL", "KXNCAAF", "KXNBA", "KXNHL", "KXEPL", "KXLALIGA", "KXSERIEA", "KXBUNDESLIGA", "KXLIGUE1",
                "KXUCL", "KXUEL", "KXMLS")
FWD_SERIES_SHARE = 0.5

SPORT_TYPES = ("football_game", "basketball_game", "hockey_match", "baseball_game", "soccer_tournament_multi_leg")
CATEGORY_QUERIES = ("Mentions", "Entertainment", "Politics", "Elections", "Science and Technology")


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _h(s: str) -> float:
    """Seeded uniform in [0, 1) from a string (volume- and result-blind sampling)."""
    return int(hashlib.sha256(f"pre_event:{s}".encode()).hexdigest()[:12], 16) / 16 ** 12


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


# ---- API -----------------------------------------------------------------------------------------
class Api:
    """``pm_hazard._api`` (cached, append-only raw store, 0.5 s between requests) with a count of
    network requests, logged to ``requests.jsonl``."""

    def __init__(self, min_interval: float = 0.5) -> None:
        from albert.config import settings
        from albert.research.pm_hazard import _api

        self.k = _api(settings(), min_interval=min_interval, nap_cap=10.0, retries=30)
        self.n = 0

    def get(self, path: str, params: dict[str, Any], *, tag: str | None = None):
        page = self.k.client.get(path, params, tag=tag)
        if not page.from_cache:
            self.n += 1
        return page

    def log(self, step: str) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "requests.jsonl", "a") as f:
            f.write(json.dumps({"step": step, "requests": self.n, "at": _iso(datetime.now(UTC))}) + "\n")


# ---- milestones ----------------------------------------------------------------------------------
MS_SCHEMA = {"event": pl.String, "start": pl.Datetime("us", "UTC"), "ms_end": pl.Datetime("us", "UTC"),
             "ms_updated": pl.Datetime("us", "UTC"), "ms_type": pl.String, "ms_title": pl.String,
             "ms_category": pl.String, "league": pl.String, "ms_id": pl.String}


def milestone_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (event ticker, milestone) of a ``/milestones`` page: every primary or related
    event the milestone lists, with its start, end, last edit, type, title, category and league."""
    from albert.venues.kalshi import parse_ts

    out = []
    for ms in payload.get("milestones") or []:
        evs = set((ms.get("primary_event_tickers") or []) + (ms.get("related_event_tickers") or []))
        for ev in sorted(evs):
            out.append({"event": ev, "start": parse_ts(ms.get("start_date")), "ms_end": parse_ts(ms.get("end_date")),
                        "ms_updated": parse_ts(ms.get("last_updated_ts")), "ms_type": ms.get("type"),
                        "ms_title": ms.get("title"), "ms_category": ms.get("category"),
                        "league": (ms.get("details") or {}).get("league"), "ms_id": ms.get("id")})
    return out


def fetch_milestones(api: Api, category: str, mtype: str | None, *, until: datetime = DATA_TO + timedelta(days=1),
                     max_pages: int = 60) -> list[dict[str, Any]]:
    """Every milestone of a category (and type) starting from ``MS_FROM``, paged until a page
    starts after ``until`` or the listing ends."""
    from albert.venues.kalshi import parse_ts

    rows: list[dict[str, Any]] = []
    cursor = None
    for _ in range(max_pages):
        prm = {"limit": 500, "minimum_start_date": _iso(MS_FROM), "category": category, "type": mtype, "cursor": cursor}
        page = api.get("/milestones", prm, tag=TAG)
        ms = page.payload.get("milestones") or []
        rows.extend(milestone_rows(page.payload))
        cursor = page.payload.get("cursor")
        starts = [s for s in (parse_ts(m.get("start_date")) for m in ms) if s is not None]
        if not cursor or not ms or (starts and min(starts) > until):
            break
    return rows


def run_milestones() -> pl.DataFrame:
    api = Api()
    rows: list[dict[str, Any]] = []
    for ty in SPORT_TYPES:
        rows += fetch_milestones(api, "Sports", ty)
        print(f"Sports/{ty}: {len(rows)} rows, {api.n} requests", flush=True)
    for cat in CATEGORY_QUERIES:
        rows += fetch_milestones(api, cat, None)
        print(f"{cat}: {len(rows)} rows, {api.n} requests", flush=True)
    api.log("milestones")
    df = pl.DataFrame(rows, schema=MS_SCHEMA).unique(["event", "ms_id"])
    _save(df, OUT / "milestones.parquet")
    return df


def event_starts(ms: pl.DataFrame) -> pl.DataFrame:
    """Earliest milestone start per event (the mentions rule) and that milestone's attributes."""
    return (ms.filter(pl.col("start").is_not_null()).sort("start", "ms_id")
            .unique("event", keep="first", maintain_order=True))


# ---- classification ------------------------------------------------------------------------------
LEAGUES = (("MLB", "KXMLB"), ("NFL", "KXNFL"), ("NCAAF", "KXNCAAF"), ("WNBA", "KXWNBA"), ("NBA", "KXNBA"),
           ("NHL", "KXNHL"))
ESPORTS_RX = re.compile(r"^KX(CS2|LOL|VALORANT|DOTA2|R6|COD|OW|RL|SC2)")
VIRTUAL_RX = re.compile(r"^KX(ESOCCER|EBASKETBALL)")
WINNER_RX = re.compile(r"(GAME|MATCH|MAP|FIGHT|BOUT)$")
PROP_SUFFIX = {
    # MLB batter / pitcher props
    "HIT", "HRR", "TB", "RBI", "HR", "SB", "KS", "HA", "ERA", "WA", "OUTS", "SINGLES", "DOUBLES", "RUNS", "BB",
    # NFL player props
    "RECYDS", "REC", "RSHYDS", "TD", "PASSYDS", "PASSTDS", "RSHATT", "PASSATT", "PASSCOMP", "RRYDS", "FFPTS",
    "LONGREC", "LONGRSH", "FIRSTTD", "TEAMFIRSTTD", "MOSTRECYDS", "MOSTRSHYDS", "INT", "PASSYDSPRECISE", "SACKS",
    # NBA / WNBA / NHL player props
    "PTS", "REB", "AST", "3PT", "PRA", "PR", "PA", "RA", "STL", "BLK", "STOCK", "PTSLEADER", "DD", "TRIPLEDOUBLE",
    "GOAL", "FIRSTGOAL", "SOG", "SAVES",
}


def sports_league(series: pl.Expr) -> pl.Expr:
    e = pl.when(series.str.contains(ESPORTS_RX.pattern)).then(pl.lit("esports"))
    e = e.when(series.str.contains(VIRTUAL_RX.pattern)).then(pl.lit("virtual"))
    for lg, pre in LEAGUES:
        e = e.when(series.str.starts_with(pre)).then(pl.lit(lg))
    return e.otherwise(pl.lit("other"))


def sports_kind(series: str) -> str:
    """``prop`` (player props, soccer goalscorers), ``winner`` (game / match / map winner) or
    ``line`` (spreads, totals, team totals, halves, quarters, innings, correct score, ...)."""
    base = series.removesuffix("PROT")
    for _, pre in LEAGUES:
        if base.startswith(pre):
            if base[len(pre):] in PROP_SUFFIX:
                return "prop"
            break
    else:
        if re.search(r"(FIRSTGOAL|GOAL|SCORER)$", base):
            return "prop"
    if WINNER_RX.search(base):
        return "winner"
    return "line"


def group_cols(m: pl.DataFrame) -> pl.DataFrame:
    """``league``, ``kind`` and the reporting / calibration ``group`` of each market."""
    from albert.research.mentions_history import speaker_group

    kinds = {s: sports_kind(s) for s in m["series_ticker"].unique().to_list()}
    m = m.with_columns(sports_league(pl.col("series_ticker")).alias("league"),
                       pl.col("series_ticker").replace_strict(kinds, default="line").alias("kind"),
                       speaker_group(pl.col("series_ticker")).alias("sgroup"))
    cat = pl.col("category")
    grp = (pl.when(cat == "Mentions").then(pl.when(pl.col("sgroup") == "earnings").then(pl.lit("mentions earnings"))
                                           .otherwise(pl.lit("mentions")))
           .when((cat == "Sports") & pl.col("league").is_in(["esports", "virtual"])).then(pl.col("league"))
           .when(cat == "Sports").then(pl.lit("sports ") + pl.col("kind"))
           .when(cat == "Science and Technology").then(pl.lit("scitech"))
           .otherwise(cat.str.to_lowercase()))
    return m.with_columns(grp.alias("group"))




# ---- universe --------------------------------------------------------------------------------------
LEG_COLS = ["ticker", "event_ticker", "open_time", "close_time", "settlement_ts", "result", "fee_type", "fee_multiplier"]
CUTOFF = datetime(2026, 9, 18, tzinfo=UTC)   # historical cutoff (``/historical/cutoff``, 2026-11-17)


def _markets(tables: Path = TABLES) -> pl.LazyFrame:
    return (pl.scan_parquet(tables / "kalshi_markets.parquet")
            .filter(pl.col("result").is_in(["yes", "no"]) & (pl.col("market_type") == "binary")
                    & pl.col("open_time").is_not_null()))


def event_universe(starts: pl.DataFrame, tables: Path = TABLES) -> pl.DataFrame:
    """One row per event with a scheduled start in the data window: the ticker time for Sports
    events that carry one, else the earliest milestone of the matching category (a sports event's
    start only from a sports-type milestone); group, league, ``game`` (cluster key: the shared
    game code of a sports event, else the event), ``t``, split and month."""
    from albert.venues.kalshi import ticker_start

    ev = (_markets(tables).group_by("event_ticker")
          .agg(pl.col("series_ticker").first(), pl.col("category").first(), pl.len().alias("n_legs")).collect())
    ev = group_cols(ev).filter(pl.col("group") != "virtual")
    sp = ev.filter(pl.col("category") == "Sports")["event_ticker"].to_list()
    tk = pl.DataFrame({"event_ticker": sp, "t_start": [ticker_start(e) for e in sp]},
                      schema={"event_ticker": pl.String, "t_start": pl.Datetime("us", "UTC")})
    st = starts.select(pl.col("event").alias("event_ticker"), pl.col("start").alias("ms_start"), "ms_updated", "ms_type",
                       "ms_title", "ms_category")
    ev = ev.join(tk, on="event_ticker", how="left").join(st, on="event_ticker", how="left")
    ok = (pl.col("ms_category") == pl.col("category")).fill_null(False)
    ms_start = pl.when(ok).then(pl.col("ms_start"))
    ev = ev.with_columns(pl.coalesce("t_start", ms_start).alias("start"),
                         pl.when(pl.col("t_start").is_not_null()).then(pl.lit("ticker"))
                         .when(ms_start.is_not_null()).then(pl.lit("milestone")).alias("start_src"))
    ev = ev.filter(pl.col("start").is_not_null() & (pl.col("start") >= DATA_FROM) & (pl.col("start") < DATA_TO))
    code = pl.col("event_ticker").str.split("-").list.get(1, null_on_oob=True)
    game = pl.when(pl.col("category") == "Sports").then(pl.col("league") + ":" + code).otherwise(pl.col("event_ticker"))
    return ev.with_columns((pl.col("start") - LEAD).alias("t"), game.alias("game"),
                           ((pl.col("start") - LEAD) >= HOLDOUT_FROM).alias("holdout"),
                           pl.col("start").dt.strftime("%Y-%m").alias("month"),
                           (pl.col("ms_updated") <= pl.col("start") - LEAD).alias("ms_known"))


def leg_universe(evu: pl.DataFrame, tables: Path = TABLES) -> pl.DataFrame:
    """Legs of those events open at ``t`` (Mentions: speech legs only), with the fetch route:
    ``xcat`` (whole-life hourly layer exists), ``live`` (settled after the historical cutoff:
    batch endpoint) or ``hist`` (one request per leg)."""
    from albert.research import mentions as M

    keep = ["event_ticker", "series_ticker", "category", "group", "league", "kind", "sgroup", "start", "start_src", "t",
            "game", "holdout", "month", "ms_known"]
    L = (_markets(tables).select(LEG_COLS).join(evu.select(keep).lazy(), on="event_ticker")
         .filter((pl.col("open_time") <= pl.col("t")) & (pl.col("close_time") > pl.col("t"))).collect())
    mm = (_markets(tables).filter(pl.col("category") == "Mentions")
          .select("ticker", "event_ticker", "series_ticker", "title", "yes_sub_title", "open_time", "close_time",
                  "settlement_ts", "result").collect().with_columns(pl.lit(None, dtype=pl.Float64).alias("volume")))
    speech = M.legs(mm).filter(~pl.col("series").is_in(M.WINDOW_SERIES))["ticker"]
    L = L.filter((pl.col("category") != "Mentions") | pl.col("ticker").is_in(speech.implode()))
    xh = pl.read_parquet(xcat.OUT / "hourly" / "done.parquet")["ticker"]
    route = (pl.when(pl.col("ticker").is_in(xh.implode())).then(pl.lit("xcat"))
             .when(pl.col("settlement_ts") >= CUTOFF).then(pl.lit("live")).otherwise(pl.lit("hist")))
    return L.with_columns(route.alias("route"), (pl.col("settlement_ts") >= CUTOFF).alias("live_api"),
                          (pl.col("result") == "yes").cast(pl.Int8).alias("y"))


# Sampled groups: (legs per game, dev legs, holdout legs), live route only (the batch endpoint).
SAMPLE = {"sports prop": (30, 9000, 12000), "sports line": (20, 7000, 9000), "sports winner": (99, 4000, 5000),
          "esports": (99, 4000, 5000)}


def sample(L: pl.DataFrame, spec: dict[str, tuple[int, int, int]] = SAMPLE) -> pl.DataFrame:
    """Every leg of the unsampled groups; for the sampled groups, games in seeded-hash order until
    the leg target is reached, at most ``cap`` legs per game (seeded hash of the ticker). Never on
    volume or result."""
    out = [L.filter(~pl.col("group").is_in(list(spec)))]
    for g, (cap, n_dev, n_hold) in spec.items():
        for hold, n in ((False, n_dev), (True, n_hold)):
            x = L.filter((pl.col("group") == g) & (pl.col("holdout") == hold) & (pl.col("route") == "live"))
            if not x.height:
                continue
            gm = x.group_by("game").len().with_columns(
                pl.col("game").map_elements(_h, return_dtype=pl.Float64).alias("u"),
                pl.min_horizontal(pl.col("len"), pl.lit(cap)).alias("take")).sort("u")
            gm = gm.with_columns(pl.col("take").cum_sum().alias("cum")).filter(pl.col("cum") - pl.col("take") < n)
            x = x.filter(pl.col("game").is_in(gm["game"].implode()))
            x = x.with_columns(pl.col("ticker").map_elements(_h, return_dtype=pl.Float64).alias("u_leg"))
            out.append(x.filter(pl.col("u_leg").rank("ordinal").over("game") <= cap).drop("u_leg"))
    return pl.concat(out, how="vertical_relaxed")


def run_universe() -> pl.DataFrame:
    ms = pl.read_parquet(OUT / "milestones.parquet")
    evu = event_universe(event_starts(ms))
    L = leg_universe(evu)
    _save(L.group_by("group", "league", "route", "holdout").agg(pl.len().alias("legs"), pl.col("game").n_unique().alias("games"))
          .sort("group", "league", "route", "holdout"), OUT / "universe_counts.parquet")
    S = sample(L)
    _save(S, OUT / "legs.parquet")
    print(S.group_by("group", "route", "holdout").agg(pl.len().alias("legs"), pl.col("game").n_unique().alias("games"),
                                                      pl.col("event_ticker").n_unique().alias("events"))
          .sort("group", "route", "holdout"))
    return S


# ---- candles ----------------------------------------------------------------------------------------
def _batch_jobs(rows: list[tuple[str, datetime, datetime]], *, period_min: int, max_span: timedelta) -> list[dict[str, Any]]:
    """Group (ticker, lo, hi) windows, sorted by ``lo``, into batch requests sharing one window
    [min lo, max hi] within the API caps (tickers, candles summed over tickers) and ``max_span``."""
    jobs: list[dict[str, Any]] = []
    cur: list[tuple[str, datetime, datetime]] = []

    def periods(lo: datetime, hi: datetime) -> int:
        return math.ceil((hi - lo).total_seconds() / 60 / period_min) + 1

    for r in sorted(rows, key=lambda x: (x[1], x[0])):
        if cur:
            lo, hi = cur[0][1], max(max(x[2] for x in cur), r[2])
            if (len(cur) >= MAX_BATCH_TICKERS or periods(lo, hi) * (len(cur) + 1) > MAX_BATCH_CANDLES
                    or hi - lo > max_span):
                jobs.append({"tickers": [x[0] for x in cur], "lo": cur[0][1], "hi": max(x[2] for x in cur)})
                cur = []
        cur.append(r)
    if cur:
        jobs.append({"tickers": [x[0] for x in cur], "lo": cur[0][1], "hi": max(x[2] for x in cur)})
    return jobs


def _fetch_batches(api: Api, jobs: list[dict[str, Any]], period: int, label: str) -> pl.DataFrame:
    frames, buf = [], []
    for i, j in enumerate(jobs, 1):
        prm = {"market_tickers": ",".join(sorted(j["tickers"])), "start_ts": int(j["lo"].timestamp()),
               "end_ts": int(j["hi"].timestamp()), "period_interval": period}
        try:
            got = xcat.batch_rows(api.get("/markets/candlesticks", prm).payload, period)
        except RuntimeError as error:
            print(f"batch failed: {str(error)[-200:]}", flush=True)
            continue
        for rows in got.values():
            buf.extend(rows)
        if len(buf) > 20_000:
            frames.append(pl.DataFrame(buf, schema=xcat.OHLC_SCHEMA))
            buf = []
        if i % 50 == 0:
            print(f"  {label} {i}/{len(jobs)} requests, {api.n} new", flush=True)
    frames.append(pl.DataFrame(buf, schema=xcat.OHLC_SCHEMA))
    return pl.concat(frames)


def _fetch_single(api: Api, rows: list[tuple[str, datetime, datetime, bool]], period: int, label: str) -> pl.DataFrame:
    """One request per market: historical endpoint before the cutoff, else the live one (a 404
    tries the other, as in ``xcat``)."""
    from albert.venues.kalshi import series_of

    out: list[dict[str, Any]] = []
    for i, (tk, lo, hi, live) in enumerate(rows, 1):
        prm = {"start_ts": int(lo.timestamp()), "end_ts": int(hi.timestamp()), "period_interval": period}
        paths = [f"/series/{series_of(tk)}/markets/{tk}/candlesticks", f"/historical/markets/{tk}/candlesticks"]
        for path in (paths if live else paths[::-1]):
            try:
                out.extend(xcat.ohlc_rows(api.get(path, prm).payload, tk, period))
                break
            except RuntimeError as error:
                if "404" not in str(error):
                    print(f"skip {tk}: {str(error)[-160:]}", flush=True)
                    break
        if i % 100 == 0:
            print(f"  {label} {i}/{len(rows)} markets, {api.n} new", flush=True)
    return pl.DataFrame(out, schema=xcat.OHLC_SCHEMA)


def _xcat_hourly(L: pl.DataFrame) -> pl.DataFrame:
    """The ``xcat`` whole-life hourly candles of these legs within [start - 30 h, start]."""
    parts = sorted((xcat.OUT / "hourly").glob("part-*.parquet"))
    w = L.select("ticker", (pl.col("start") - HOURLY_SPAN).alias("lo"), pl.col("start").alias("hi"))
    frames = [pl.DataFrame(schema=xcat.OHLC_SCHEMA)]
    tk = w["ticker"].unique().sort().to_list()
    for i in range(0, len(tk), 2000):
        c = pl.scan_parquet(parts).filter(pl.col("ticker").is_in(tk[i:i + 2000])).collect()
        c = c.join(w, on="ticker").filter((pl.col("end_ts") >= pl.col("lo")) & (pl.col("end_ts") <= pl.col("hi")))
        frames.append(c.select(list(xcat.OHLC_SCHEMA)))
    return pl.concat(frames).unique(["ticker", "end_ts"])


def run_hourly() -> pl.DataFrame:
    """Stage 1: hourly candles over [start - 30 h, start] for every sampled leg (``xcat`` layer or
    live batch requests). ``hist`` legs get one-minute candles per leg over [start - 25 h, start]
    instead (one request gives the quotes and the tape)."""
    L = pl.read_parquet(OUT / "legs.parquet")
    api = Api()
    xc = _xcat_hourly(L.filter(pl.col("route") == "xcat"))
    live = L.filter(pl.col("route") == "live")
    jobs = _batch_jobs([(r[0], r[1] - HOURLY_SPAN, r[1]) for r in live.select("ticker", "start").iter_rows()],
                       period_min=60, max_span=timedelta(hours=60))
    print(f"hourly: {xc.height} xcat candles, {len(jobs)} batch requests for {live.height} live legs", flush=True)
    hb = _fetch_batches(api, jobs, 60, "hourly")
    api.log("hourly")
    H = pl.concat([xc, hb]).unique(["ticker", "end_ts"]).sort("ticker", "end_ts")
    _save(H, OUT / "hourly.parquet")
    hist = L.filter(pl.col("route") == "hist")
    if hist.height:
        rows = [(r[0], r[1] - H24 - timedelta(hours=1), r[1], False) for r in hist.select("ticker", "start").iter_rows()]
        print(f"hist legs: {len(rows)} single requests", flush=True)
        _save(_fetch_single(api, rows, 1, "hist minute"), OUT / "minute_hist.parquet")
        api.log("hist minute")
    return H


# ---- decision rows ------------------------------------------------------------------------------------
def quote_asof(L: pl.DataFrame, C: pl.DataFrame, at: str, *, max_age: timedelta = QUOTE_AGE) -> pl.DataFrame:
    """Last candle ending at or before ``L[at]`` with both sides quoted (ask > bid), within
    ``max_age``: bid, ask (Kalshi omits candles in which nothing changed, so the quote stands)."""
    c = (C.select("ticker", pl.col("end_ts").alias("q_ts"), pl.col("bid_close").alias("bid"), pl.col("ask_close").alias("ask"))
         .filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("ask") > pl.col("bid")))
         .sort("ticker", "q_ts"))
    out = L.sort("ticker", at).join_asof(c, left_on=at, right_on="q_ts", by="ticker", strategy="backward",
                                        check_sortedness=False)
    return out.filter(pl.col("q_ts").is_not_null() & ((pl.col(at) - pl.col("q_ts")) <= max_age))


def decision_rows(L: pl.DataFrame, C: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Quoted legs at ``t`` with the early-start exclusion (share bid >= 97c at ``t`` vs at
    ``start - 24 h``), ``end``, mid, spread."""
    q6 = quote_asof(L, C, "t")
    q24 = quote_asof(L.with_columns((pl.col("start") - H24).alias("t24")), C, "t24")
    s6 = q6.group_by("event_ticker").agg((pl.col("bid") >= SAID_BID).mean().alias("said"))
    s24 = q24.group_by("event_ticker").agg((pl.col("bid") >= SAID_BID).mean().alias("said24"))
    sh = s6.join(s24, on="event_ticker", how="left").with_columns(pl.col("said24").fill_null(0.0))
    bad = sh.filter(pl.col("said") > pl.col("said24") + SAID_JUMP)["event_ticker"]
    Q = q6.filter(~pl.col("event_ticker").is_in(bad.implode()))
    Q = Q.with_columns(pl.min_horizontal(pl.col("start") - END_MARGIN, pl.col("close_time")).alias("end"),
                       ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("ask") - pl.col("bid")).alias("spread"),
                       pl.col("start").dt.convert_time_zone(ET).dt.date().alias("et_date"))
    Q = Q.filter(pl.col("end") > pl.col("t"))
    info = {"legs": L.height, "quoted_at_t": q6.height, "events_quoted": q6["event_ticker"].n_unique(),
            "events_excluded_early_start": int(bad.len()), "rows": Q.height, "events": Q["event_ticker"].n_unique()}
    return Q, info


def logit(p: np.ndarray, eps: float = 0.005) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def calibrate(Q: pl.DataFrame) -> dict[str, tuple[float, float, int]]:
    """Per group ``logit P(yes) = a + b logit(mid)`` on dev rows (Newton logistic); Mentions keep the
    frozen round-2 fit; a group without 200 dev rows uses the raw mid (a 0, b 1)."""
    from albert.research.fundamental import fit_logistic

    out: dict[str, tuple[float, float, int]] = {}
    for g in sorted(Q["group"].unique().to_list()):
        d = Q.filter((pl.col("group") == g) & ~pl.col("holdout"))
        if g == "mentions":
            out[g] = (FROZEN_A, FROZEN_B, d.height)
        elif d.height < 200:
            out[g] = (0.0, 1.0, d.height)
        else:
            X = np.column_stack([np.ones(d.height), logit(d["mid"].to_numpy())])
            beta, _ = fit_logistic(X, d["y"].to_numpy().astype(float), d["game"].to_numpy())
            out[g] = (round(float(beta[0]), 4), round(float(beta[1]), 4), d.height)
    return out


def with_fair(Q: pl.DataFrame, cal: dict[str, tuple[float, float, int]]) -> pl.DataFrame:
    a = Q["group"].replace_strict({g: v[0] for g, v in cal.items()}, return_dtype=pl.Float64).to_numpy()
    b = Q["group"].replace_strict({g: v[1] for g, v in cal.items()}, return_dtype=pl.Float64).to_numpy()
    fv = np.clip(1 / (1 + np.exp(-(a + b * logit(Q["mid"].to_numpy())))), 0.003, 0.997)
    return Q.with_columns(pl.Series("fv_cal", fv), pl.col("mid").alias("fv_mid"))


def order_prices(fair: pl.Expr, bid: pl.Expr, ask: pl.Expr, margin: float) -> tuple[pl.Expr, pl.Expr]:
    """(YES bid, NO bid): ``floor(fair - m)`` capped at ask - 1c, ``floor(1 - fair - m)`` capped at
    1 - bid - 1c, on the cent grid (null when under 1c: not posted)."""
    yb = pl.min_horizontal(((fair - margin) * 100 + 1e-9).floor() / 100, (ask * 100 - 1).round(0) / 100)
    nb = pl.min_horizontal(((1 - fair - margin) * 100 + 1e-9).floor() / 100, ((1 - bid) * 100 - 1).round(0) / 100)
    return (pl.when(yb >= 0.01 - 1e-9).then(yb), pl.when(nb >= 0.01 - 1e-9).then(nb))


def candidates(Q: pl.DataFrame, H: pl.DataFrame) -> pl.DataFrame:
    """Rows whose hourly trade range over hours overlapping (t, end] could hold a fill of any
    configuration (8c, either fair value, either side) or a breaker print (>= 95c on a leg asked
    <= 80c at t). Every other row provably did not fill (no print beyond its levels)."""
    lv = []
    for fv in ("fv_cal", "fv_mid"):
        yb, nb = order_prices(pl.col(fv), pl.col("bid"), pl.col("ask"), min(MARGINS))
        lv += [yb.alias(f"yb_{fv}"), nb.alias(f"nb_{fv}")]
    q = Q.select("ticker", "t", "end", "ask", *lv).with_columns(
        (1 - pl.max_horizontal("nb_fv_cal", "nb_fv_mid")).fill_null(9.0).alias("ask_lvl"),   # most aggressive
        pl.max_horizontal("yb_fv_cal", "yb_fv_mid").fill_null(-9.0).alias("yes_lvl"))
    h = H.select("ticker", "end_ts", "p_low", "p_high").join(q.select("ticker", "t", "end", "ask", "ask_lvl", "yes_lvl"), on="ticker")
    h = h.filter((pl.col("end_ts") > pl.col("t")) & (pl.col("end_ts") - timedelta(hours=1) < pl.col("end")))
    hit = h.filter((pl.col("p_high") > pl.col("ask_lvl") + 1e-9) | (pl.col("p_low") < pl.col("yes_lvl") - 1e-9)
                   | ((pl.col("ask") <= CB_ASK) & (pl.col("p_high") >= CB_PRINT - 1e-9)))
    return Q.filter(pl.col("ticker").is_in(hit["ticker"].unique().implode()))


def run_rows() -> tuple[pl.DataFrame, dict[str, Any]]:
    L = pl.read_parquet(OUT / "legs.parquet")
    H = pl.read_parquet(OUT / "hourly.parquet")
    mh = OUT / "minute_hist.parquet"
    C = pl.concat([H, pl.read_parquet(mh)]) if mh.exists() else H
    Q, info = decision_rows(L, C)
    frozen = FROZEN_CAL if FROZEN_CAL.exists() and OUT != FROZEN_CAL.parent else None
    if frozen:     # the forward period applies the Aug-Sep dev fits unchanged
        cal = {g: (v["a"], v["b"], v["dev_rows"]) for g, v in json.loads(frozen.read_text())["calibration"].items()}
        cal = {g: cal.get(g, (0.0, 1.0, 0)) for g in Q["group"].unique().to_list()}
    else:
        cal = calibrate(Q)
    Q = with_fair(Q, cal)
    _save(Q, OUT / "rows.parquet")
    hist = set(L.filter(pl.col("route") == "hist")["ticker"].to_list())
    cand = candidates(Q.filter(~pl.col("ticker").is_in(list(hist))), H)
    _save(cand.select("ticker", "event_ticker", "group", "t", "end", "start", "live_api"), OUT / "candidates.parquet")
    info |= {"calibration": {g: {"a": v[0], "b": v[1], "dev_rows": v[2]} for g, v in cal.items()},
             "candidates": cand.height, "candidates_live": int(cand["live_api"].sum())}
    (OUT / "rows_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info, indent=1, default=str))
    print(Q.group_by("group", "holdout").agg(pl.len().alias("rows"), pl.col("event_ticker").n_unique().alias("events"),
                                             pl.col("spread").median().alias("spread_med")).sort("group", "holdout"))
    print(cand.group_by("group", "live_api").len().sort("group", "live_api"))
    return Q, info


# ---- stage 2: minute candles over (t, end] for candidates ----------------------------------------------
def run_minute(groups: list[str] | None = None, *, max_requests: int = 2500) -> pl.DataFrame:
    """One-minute candles over [t, end] for candidate rows: live batch requests (legs sharing a
    window of <= 8 h, <= 10,000 candles each) and one historical request per pre-cutoff leg
    (Mentions excluded: their pre-cutoff fills come from the cached public prints)."""
    cand = pl.read_parquet(OUT / "candidates.parquet")
    if groups:
        cand = cand.filter(pl.col("group").is_in(groups))
    path = OUT / "minute.parquet"
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=xcat.OHLC_SCHEMA)
    done = set(have["ticker"].unique().to_list())
    cand = cand.filter(~pl.col("ticker").is_in(list(done)))
    api = Api()
    live = cand.filter(pl.col("live_api"))
    jobs = _batch_jobs([(r[0], r[1], r[2]) for r in live.select("ticker", "t", "end").iter_rows()],
                       period_min=1, max_span=timedelta(hours=8))
    hist = cand.filter(~pl.col("live_api") & ~pl.col("group").str.starts_with("mentions"))
    print(f"minute: {len(jobs)} batch requests for {live.height} live candidates, {hist.height} single", flush=True)
    if len(jobs) + hist.height > max_requests:
        raise RuntimeError(f"over the request cap ({len(jobs) + hist.height} > {max_requests})")
    parts = [have, _fetch_batches(api, jobs, 1, "minute")]
    if hist.height:
        parts.append(_fetch_single(api, [(r[0], r[1], r[2], False) for r in hist.select("ticker", "t", "end").iter_rows()],
                                   1, "minute hist"))
    api.log("minute")
    M = pl.concat(parts).unique(["ticker", "end_ts"]).sort("ticker", "end_ts")
    _save(M, path)
    return M



# ---- fills ---------------------------------------------------------------------------------------------
TAPE_SCHEMA = {"ticker": pl.String, "ts": pl.Datetime("us", "UTC"), "hi": pl.Float64, "lo": pl.Float64,
               "yes_taker": pl.Boolean}
CONFIGS = tuple((fv, endc, m) for fv in ("fv_cal", "fv_mid") for endc in ("end_cb", "end") for m in MARGINS)


def tape_from_candles(M: pl.DataFrame) -> pl.DataFrame:
    """One-minute candles as a tape: ``ts`` = candle end, ``hi`` / ``lo`` = the minute's trade range
    (taker side unknown: a print above our ask is a fill whoever the taker was)."""
    return (M.filter(pl.col("p_high").is_not_null())
            .select("ticker", pl.col("end_ts").alias("ts"), pl.col("p_high").alias("hi"), pl.col("p_low").alias("lo"),
                    pl.lit(None, dtype=pl.Boolean).alias("yes_taker")))


def tape_from_prints(tr: pl.DataFrame) -> pl.DataFrame:
    """Public prints as a tape (taker side kept: a NO bid fills only on a YES-taker print above our
    ask, a YES bid only on a NO-taker print below our bid, as in ``mentions_history``)."""
    return tr.select("ticker", pl.col("created_time").alias("ts"), pl.col("yes_price").alias("hi"),
                     pl.col("yes_price").alias("lo"), (pl.col("taker_side") == "yes").alias("yes_taker"))


def kill_times(Q: pl.DataFrame, tape: pl.DataFrame) -> pl.DataFrame:
    """Circuit breaker per event: the first tape time in (t, end] with a print >= 95c on a leg of
    the event asked <= 80c at t."""
    legs = Q.filter(pl.col("ask") <= CB_ASK).select("event_ticker", "ticker", "t", "end")
    k = (legs.join(tape.filter(pl.col("hi") >= CB_PRINT - 1e-9).select("ticker", "ts"), on="ticker")
         .filter((pl.col("ts") > pl.col("t")) & (pl.col("ts") <= pl.col("end"))))
    return k.group_by("event_ticker").agg(pl.col("ts").min().alias("kill_ts"))


def with_kill(Q: pl.DataFrame, tape: pl.DataFrame) -> pl.DataFrame:
    kt = kill_times(Q, tape)
    return (Q.drop("kill_ts", "end_cb", strict=False).join(kt, on="event_ticker", how="left")
            .with_columns(pl.min_horizontal("end", pl.col("kill_ts").fill_null(pl.col("end"))).alias("end_cb")))


def fee_order(n: pl.Expr, price: pl.Expr) -> pl.Expr:
    """Kalshi maker fee for one order, rounded up to the cent (maker-fee series only)."""
    coef = pl.when(pl.col("fee_type").fill_null("").str.contains("maker_fee")).then(MAKER).otherwise(0.0)
    raw = coef * pl.col("fee_multiplier").fill_null(1.0) * n * price * (1 - price)
    return (raw * 100 - 1e-9).ceil().clip(lower_bound=0) / 100


def orders(Q: pl.DataFrame, fv: str, margin: float) -> pl.DataFrame:
    """The rule's two bids per row: side, price, and the trade level a print must pass."""
    yb, nb = order_prices(pl.col(fv), pl.col("bid"), pl.col("ask"), margin)
    o = Q.select("row", yb.alias("yes"), nb.alias("no")).unpivot(index="row", variable_name="side", value_name="price")
    return o.filter(pl.col("price").is_not_null())


def fills(Q: pl.DataFrame, tape: pl.DataFrame) -> pl.DataFrame:
    """Every configuration's orders on rows ``Q`` (needs ``row``, ``end_cb``): filled when a later
    tape entry in (t, end_c] trades strictly through the level; PnL per contract held to settlement
    for a 1-contract order and a $5 order (``n = floor(5 / price)``), maker fee rounded up per order."""
    W = Q.select("row", "ticker", "t", "end", "end_cb")
    T = (tape.join(W, on="ticker").filter((pl.col("ts") > pl.col("t")) & (pl.col("ts") <= pl.col("end")))
         .select("row", "ts", "hi", "lo", "yes_taker", "end_cb"))
    base = Q.select("row", "y", "fee_type", "fee_multiplier")
    out = []
    for fv, endc, m in CONFIGS:
        o = orders(Q, fv, m)
        lvl = pl.when(pl.col("side") == "no").then(1 - pl.col("price")).otherwise(pl.col("price"))
        thru = (pl.when(pl.col("side") == "no").then((pl.col("hi") > lvl + 1e-9) & pl.col("yes_taker").fill_null(True))
                .otherwise((pl.col("lo") < lvl - 1e-9) & (~pl.col("yes_taker")).fill_null(True)))
        h = o.join(T, on="row").filter(thru)
        if endc == "end_cb":
            h = h.filter(pl.col("ts") <= pl.col("end_cb"))
        h = h.group_by("row", "side").agg(pl.col("ts").min().alias("fill_ts"))
        f = o.join(h, on=["row", "side"], how="left").join(base, on="row")
        n5 = pl.max_horizontal(pl.lit(1.0), (STAKE / pl.col("price") + 1e-9).floor())
        payoff = pl.when(pl.col("side") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y")).cast(pl.Float64)
        f = f.with_columns(pl.col("fill_ts").is_not_null().alias("filled"), payoff.alias("payoff"), n5.alias("n5"))
        f = f.with_columns(fee_order(pl.lit(1.0), pl.col("price")).alias("fee1"),
                           fee_order(pl.col("n5"), pl.col("price")).alias("fee5"))
        f = f.with_columns((pl.col("payoff") - pl.col("price") - pl.col("fee1")).alias("pnl1"),
                           (pl.col("payoff") - pl.col("price") - pl.col("fee5") / pl.col("n5")).alias("pnl5"))
        out.append(f.with_columns(pl.lit(fv).alias("fair"), pl.lit(endc).alias("endc"), pl.lit(m).alias("margin"))
                   .drop("y", "fee_type", "fee_multiplier"))
    return pl.concat(out).with_columns([pl.col(c).cast(pl.Categorical) for c in ("side", "fair", "endc")])


def source_of(Q: pl.DataFrame, cand: set[str], minute: set[str], covered: pl.Series | None) -> pl.Series:
    """Fill source per row's event: ``prints`` if every candidate leg's window has cached prints,
    else ``minute`` if every candidate leg has minute candles, else ``none`` (not evaluable).
    Non-candidate legs provably did not fill under either source."""
    c = Q["ticker"].is_in(list(cand))
    pr = covered if covered is not None else pl.Series([False] * Q.height)
    mi = Q["ticker"].is_in(list(minute))
    ev = Q.select("event_ticker").with_columns(c.alias("c"), pr.alias("pr"), mi.alias("mi")).group_by("event_ticker").agg(
        (~pl.col("c") | pl.col("pr")).all().alias("all_pr"), (~pl.col("c") | pl.col("mi")).all().alias("all_mi"),
        pl.col("pr").any().alias("any_pr"))
    lab = (pl.when(pl.col("all_pr") & pl.col("any_pr")).then(pl.lit("prints"))
           .when(pl.col("all_mi")).then(pl.lit("minute")).otherwise(pl.lit("none")))
    m = ev.with_columns(lab.alias("src")).select("event_ticker", "src")
    return Q.select("event_ticker").join(m, on="event_ticker", how="left")["src"]


# ---- scoring -------------------------------------------------------------------------------------------
def cl(f: pl.DataFrame, col: str, cluster: str) -> tuple[float | None, float | None, int]:
    from albert.research.ta_calendar import cl_mean

    return cl_mean(f, col, cluster)


def _r(v: Any, k: int = 2) -> Any:
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), k)


def summary(g: pl.DataFrame) -> dict[str, Any]:
    """Orders, fills, fill rate, net c per fill and per order (1 contract), t by event / game / ET
    date, the $5 version, events, and the fill PnL's skew and worst decile."""
    f = g.filter("filled")
    row: dict[str, Any] = {"orders": g.height, "fills": f.height, "fill_rate": _r(f.height / g.height, 3) if g.height else None,
                           "events": g["event_ticker"].n_unique()}
    if not f.height:
        return row
    mu, te, ke = cl(f, "pnl1", "event_ticker")
    _, tg, kg = cl(f, "pnl1", "game")
    _, td, kd = cl(f, "pnl1", "et_date")
    x = f["pnl1"].to_numpy()
    sd = float(x.std()) if len(x) > 1 else 0.0
    row.update({"events_filled": ke, "games_filled": kg, "dates": kd, "net_c_fill": _r(100 * mu), "t_event": _r(te),
                "t_game": _r(tg), "t_date": _r(td), "net_c_order": _r(100 * f["pnl1"].sum() / g.height, 3),
                "avg_price": _r(f["price"].mean(), 3), "win": _r(f["payoff"].mean(), 3),
                "net_c_fill_5usd": _r(100 * float((f["pnl5"] * f["n5"]).sum() / f["n5"].sum())),
                "usd_5": _r(float((f["pnl5"] * f["n5"]).sum())),
                "skew": _r(float(((x - x.mean()) ** 3).mean() / sd ** 3)) if sd > 0 else None,
                "p10_c": _r(100 * float(np.quantile(x, 0.10)))})
    return row


def table(F: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    rows = [{**dict(zip(keys, k)), **summary(g)} for k, g in F.group_by(keys)]
    return pl.DataFrame(rows, infer_schema_length=None).sort(keys)


def months_positive(F: pl.DataFrame, min_fills: int = 10, min_events: int = 3) -> dict[str, Any]:
    T = table(F, ["month"])
    if "events_filled" not in T.columns:
        return {"months": 0}
    T = T.filter((pl.col("fills") >= min_fills) & (pl.col("events_filled") >= min_events))
    if not T.height:
        return {"months": 0}
    return {"months": T.height, "positive": int((T["net_c_fill"] > 0).sum()),
            "detail": " ".join(f"{m}:{v:+.1f}" for m, v in T.select("month", "net_c_fill").iter_rows())}


def taker_reference(Q: pl.DataFrame) -> pl.DataFrame:
    """Buy NO at ``1 - bid`` at t on every quoted leg (taker fee rounded up per 1-contract order)."""
    from albert.research.ta_calendar import taker_fee_expr

    d = Q.filter(pl.col("bid") > 0).with_columns((1 - pl.col("bid")).alias("px"))
    d = d.with_columns(taker_fee_expr(pl.lit(1.0), pl.col("px")).alias("fee"))
    d = d.with_columns((1 - pl.col("y") - pl.col("px") - pl.col("fee")).alias("pnl"))
    rows = []
    for (g, ho), x in d.group_by(["group", "holdout"]):
        mu, te, k = cl(x, "pnl", "event_ticker")
        _, td, _ = cl(x, "pnl", "et_date")
        rows.append({"group": g, "split": "holdout" if ho else "dev", "legs": x.height, "events": k,
                     "net_c": _r(None if mu is None else 100 * mu), "t_event": _r(te), "t_date": _r(td)})
    return pl.DataFrame(rows).sort("group", "split")


def marks_for(tickers: list[str]) -> pl.DataFrame:
    """YES mids (ticker, ts, yes_mid) of these tickers from this study's hourly candles
    (the bid/ask midpoint when both sides are quoted), plus whatever ``vol_metrics.load_marks``
    finds. Read lazily, filtered by ticker (memory)."""
    from albert.research.vol_metrics import load_marks

    paths = [p for p in (OUT / "hourly.parquet", OUT / "minute_hist.parquet") if p.exists()]   # hourly suffices for daily marks
    c = pl.concat([pl.scan_parquet(p).filter(pl.col("ticker").is_in(tickers)).select("ticker", "end_ts", "bid_close", "ask_close")
                   for p in paths]).collect()
    c = c.filter(pl.col("bid_close").is_not_null() & pl.col("ask_close").is_not_null() & (pl.col("ask_close") > pl.col("bid_close")))
    mine = c.select("ticker", pl.col("end_ts").alias("ts"), ((pl.col("bid_close") + pl.col("ask_close")) / 2).alias("yes_mid"))
    return pl.concat([mine, load_marks(tickers)]).unique(["ticker", "ts"]).sort("ticker", "ts")


TIGHT = 0.15 + 1e-9    # exploratory cut (chosen after the dev/holdout tables): quoted spread at t <= 15c


def sports_tight(F: pl.DataFrame) -> pl.DataFrame:
    return F.filter(pl.col("group").cast(pl.String).str.starts_with("sports") & (pl.col("spread") <= TIGHT))


def ledger_specs(F: pl.DataFrame) -> list[tuple[str, str, pl.DataFrame]]:
    prim = F.filter((pl.col("fair") == "fv_cal") & (pl.col("endc") == "end_cb") & pl.col("filled"))
    specs = []
    for m in MARGINS:
        x = sports_tight(prim).filter((pl.col("margin") == m) & (pl.col("side") == "no"))
        if x.height:
            specs.append(("pre_event sports <=15c books (exploratory)", f"NO bid fv_cal+{int(m * 100)}c cb $5", x))
    for g in sorted(prim["group"].unique().to_list()):
        for m in MARGINS:
            for side in ("no", "yes"):
                x = prim.filter((pl.col("group") == g) & (pl.col("margin") == m) & (pl.col("side") == side))
                if x.height:
                    specs.append((f"pre_event {g}", f"{side.upper()} bid fv_cal+{int(m * 100)}c cb $5", x))
    return specs


def run_ledgers(F: pl.DataFrame) -> pl.DataFrame:
    from albert.research.ta_calendar import capped_ledger
    from albert.research.vol_metrics import ledger_metrics

    d = OUT / "ledgers"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("ledger_*.parquet"):
        old.unlink()
    rows = []
    specs = ledger_specs(F)
    marks = marks_for(sorted({t for _, _, f in specs for t in f["ticker"].unique().to_list()}))
    for i, (strategy, variant, f) in enumerate(specs):
        # the $10 cap and the top-5 concentration check apply per game (several sports events share one)
        f = f.with_columns(pl.col("fill_ts").alias("t"), pl.col("n5").alias("n"), pl.col("fee5").alias("fee_order"),
                           pl.when(pl.col("side") == "yes").then(1).otherwise(-1).alias("dir"),
                           pl.col("game").alias("event_ticker"))
        for split, sub in (("all", f), ("holdout", f.filter(pl.col("holdout")))):
            led = capped_ledger(sub, strategy, variant, split, event_cap=EVENT_CAP)
            if not led.height:
                continue
            if split == "all":
                led.write_parquet(d / f"ledger_{i:02d}.parquet")
            met = ledger_metrics(led, marks)
            ci = met.get("sharpe_ci90")
            rows.append({"strategy": strategy, "variant": variant, "split": split, "fills": sub.height,
                         **{k: met.get(k) for k in ("positions", "events", "pnl_usd", "usd_per_week", "ann_vol_pct",
                                                     "sharpe", "p_sharpe_pos", "sharpe_ex_top5", "max_dd_usd",
                                                     "weeks_positive_pct", "skew", "first_day", "last_day")},
                         "sharpe_ci90": f"[{ci[0]}, {ci[1]}]" if ci else ""})
    return pl.DataFrame(rows, infer_schema_length=None)


def cached_prints(tickers: list[str], data_dir: Path = Path("data")) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The public prints ``mentions_history`` (and its round 2) cached, for these tickers only, and
    the windows each fetch covered (round-2 tickers whose fetch failed are not covered); the same
    files as ``mentions_history.load_all_trades``, read lazily to keep memory low."""
    from albert.research.mentions_history import R2_NAME, TRADE_SCHEMA

    c = data_dir / "cache"
    tp = [c / "mentions_history" / "trades.parquet"] + [c / f"fundamental_{n}_trades.parquet" for n in (R2_NAME, f"{R2_NAME}_retry")]
    wp = [c / "mentions_history" / "trade_windows.parquet"] + [c / f"fundamental_{n}_trades_windows.parquet"
                                                             for n in (R2_NAME, f"{R2_NAME}_retry")]
    failed = c / f"fundamental_{R2_NAME}_failed.parquet"
    bad = pl.read_parquet(failed)["ticker"].to_list() if failed.exists() else []
    tr = pl.concat([pl.scan_parquet(p).filter(pl.col("ticker").is_in(tickers)).select(list(TRADE_SCHEMA)).collect()
                    for p in tp if p.exists()]).unique("trade_id")
    ws = []
    for i, p in enumerate(w for w in wp if w.exists()):
        w = pl.scan_parquet(p).filter(pl.col("ticker").is_in(tickers)).select("ticker", "t_from", "t_to").collect()
        ws.append(w if i == 0 else w.filter(~pl.col("ticker").is_in(bad)))
    return tr, pl.concat(ws).unique()


def load_inputs() -> dict[str, Any]:
    from albert.research import mentions_history as MH

    Q = pl.read_parquet(OUT / "rows.parquet").sort("ticker", "t").with_row_index("row")
    cand = set(pl.read_parquet(OUT / "candidates.parquet")["ticker"].to_list())
    L = pl.read_parquet(OUT / "legs.parquet")
    hist_legs = set(L.filter(pl.col("route") == "hist")["ticker"].to_list())
    cand |= hist_legs & set(Q["ticker"].to_list())
    cols = ["ticker", "end_ts", "p_high", "p_low"]
    M = pl.concat([pl.scan_parquet(p).select(cols).filter(pl.col("p_high").is_not_null()).collect()
                   for p in (OUT / "minute.parquet", OUT / "minute_hist.parquet") if p.exists()])
    mq = Q.filter(pl.col("group").str.starts_with("mentions"))
    trades, windows = cached_prints(mq["ticker"].unique().to_list())
    cov = MH.covered_rows(mq.select("ticker", "t", "end"), windows) if windows.height else pl.Series([False] * mq.height)
    covered = Q.select("row").join(mq.select("row").with_columns(cov.alias("cov")), on="row", how="left")["cov"].fill_null(False)
    return {"Q": Q, "cand": cand, "M": M, "trades": trades, "covered": covered}


def run_score() -> dict[str, pl.DataFrame]:
    inp = load_inputs()
    Q, M = inp["Q"], inp["M"]
    mt = tape_from_candles(M)
    Q = Q.with_columns(source_of(Q, inp["cand"], set(mt["ticker"].unique().to_list()), inp["covered"]).alias("src"))
    pt = tape_from_prints(inp["trades"])
    attrs = ["row", "ticker", "event_ticker", "group", "league", "sgroup", "game", "et_date", "month", "holdout", "t",
             "settlement_ts", "mid", "spread"]
    cat = [pl.col(c).cast(pl.Categorical) for c in ("group", "league", "sgroup", "month")]
    parts = []
    for src, tape in (("minute", mt), ("prints", pt)):
        q = with_kill(Q.filter(pl.col("src") == src), tape)
        if q.height:
            parts.append(fills(q, tape).join(q.select(attrs).with_columns(cat), on="row")
                         .with_columns(pl.lit(src).alias("fill_src")))
    # validation: Mentions events evaluable both ways, filled from the minute candles
    both = Q.filter(pl.col("src") == "prints")
    mi_ok = source_of(both, inp["cand"], set(mt["ticker"].unique().to_list()), None) == "minute"
    vq = with_kill(both.filter(mi_ok), mt)
    V = fills(vq, mt).join(vq.select(attrs), on="row").with_columns(pl.lit("minute (validation)").alias("fill_src")) if vq.height else None
    F = pl.concat(parts, how="vertical_relaxed").with_columns(
        pl.when(pl.col("t") >= FWD_FROM).then(pl.lit("forward Nov")).when(pl.col("holdout")).then(pl.lit("holdout Oct"))
        .otherwise(pl.lit("dev")).alias("split"),
        pl.concat_str([pl.col("side").cast(pl.String).str.to_uppercase(), pl.col("fair").cast(pl.String), pl.col("endc").cast(pl.String),
                       (pl.col("margin") * 100).round(0).cast(pl.Int64).cast(pl.String) + "c"], separator=" ").alias("config"))
    F.write_parquet(OUT / "orders.parquet")
    prim = F.filter((pl.col("fair") == "fv_cal") & (pl.col("endc") == "end_cb"))
    T: dict[str, pl.DataFrame] = {}
    T["coverage"] = Q.group_by("group", "holdout", "src").agg(pl.len().alias("rows"), pl.col("event_ticker").n_unique().alias("events")).sort("group", "holdout", "src")
    T["headline"] = table(F, ["group", "config", "split"])
    T["headline_pooled"] = table(F, ["group", "config"])
    T["by_league"] = table(prim.filter(pl.col("group").cast(pl.String).str.starts_with("sports")), ["group", "league", "config", "split"])
    T["mentions_by_speaker"] = table(prim.filter(pl.col("group") == "mentions"), ["sgroup", "config", "split"])
    T["monthly"] = table(prim, ["group", "config", "month"])
    T["sports_tight"] = pl.concat([table(sports_tight(prim), ["config", "split"]).with_columns(pl.lit("all").alias("group")),
                                   table(sports_tight(prim), ["group", "config", "split"])], how="diagonal_relaxed")
    T["spread_band"] = table(prim.with_columns(pl.when(pl.col("spread") <= 0.05 + 1e-9).then(pl.lit("<=5c"))
                                               .when(pl.col("spread") <= 0.15 + 1e-9).then(pl.lit("6-15c"))
                                               .otherwise(pl.lit(">15c")).alias("sband")), ["group", "config", "sband", "split"])
    mp = []
    for (g, c), x in prim.group_by(["group", "config"]):
        mp.append({"group": g, "config": c, **months_positive(x)})
    T["months_positive"] = pl.DataFrame(mp, infer_schema_length=None).sort("group", "config")
    T["taker_no_at_t"] = taker_reference(Q)
    if V is not None:
        V = V.with_columns(pl.concat_str([pl.col("side").cast(pl.String).str.to_uppercase(), pl.col("fair").cast(pl.String), pl.col("endc").cast(pl.String),
                                          (pl.col("margin") * 100).round(0).cast(pl.Int64).cast(pl.String) + "c"],
                                         separator=" ").alias("config"))
        a = F.filter(pl.col("fill_src") == "prints").join(V.select("row", "config", pl.col("filled").alias("filled_m"),
                                                                   pl.col("pnl1").alias("pnl1_m")), on=["row", "config"])
        vr = []
        for (c,), x in a.group_by(["config"]):
            fp, fm = x.filter("filled"), x.filter("filled_m")
            vr.append({"config": c, "orders": x.height, "events": x["event_ticker"].n_unique(), "fills_prints": fp.height,
                       "fills_minute": fm.height, "agree": _r(float((x["filled"] == x["filled_m"]).mean()), 4),
                       "net_c_fill_prints": _r(100 * fp["pnl1"].mean()) if fp.height else None,
                       "net_c_fill_minute": _r(100 * fm["pnl1_m"].mean()) if fm.height else None})
        T["validation_minute_vs_prints"] = pl.DataFrame(vr).sort("config")
    del M, mt, pt, inp
    T["ledgers"] = run_ledgers(F)
    for k, v in T.items():
        v.write_csv(OUT / f"{k}.csv")
    return T


# ---- forward period (November) -----------------------------------------------------------------------
def use_forward() -> None:
    """Point every stage at ``OUT/forward`` and the November window (milestones from Oct 31)."""
    global OUT, DATA_FROM, DATA_TO, MS_FROM
    OUT, DATA_FROM, DATA_TO, MS_FROM = OUT / "forward", FWD_FROM, FWD_TO, FWD_FROM - timedelta(days=1)
    OUT.mkdir(parents=True, exist_ok=True)


def run_forward_markets() -> pl.DataFrame:
    """Settled November markets: per-series listings (``/markets?series_ticker&status=settled``,
    closes from 11-01) for a seeded-hash half of the main-league and top-soccer series that traded
    after 10-15, and per-event listings for the Mentions / Entertainment events with a November
    milestone. Also the November milestones (sports types and those two categories)."""
    from albert.venues import kalshi as k

    api = Api()
    rows: list[dict[str, Any]] = []
    for ty in SPORT_TYPES[:3] + SPORT_TYPES[4:]:
        rows += fetch_milestones(api, "Sports", ty, until=FWD_TO + timedelta(days=1))
    for cat in ("Mentions", "Entertainment"):
        rows += fetch_milestones(api, cat, None, until=FWD_TO + timedelta(days=1))
    ms = pl.DataFrame(rows, schema=MS_SCHEMA).unique(["event", "ms_id"])
    _save(ms, OUT / "milestones.parquet")
    print(f"milestones: {ms.height} rows, {api.n} requests", flush=True)
    info = {r["series_ticker"]: r for r in pl.read_parquet(TABLES / "kalshi_series.parquet").iter_rows(named=True)}
    old = (pl.scan_parquet(TABLES / "kalshi_markets.parquet").filter(pl.col("category") == "Sports")
           .group_by("series_ticker").agg(pl.col("close_time").max().alias("last")).collect())
    ser = old.filter(pl.any_horizontal([pl.col("series_ticker").str.starts_with(x) for x in FWD_PREFIXES])
                     & (pl.col("last") >= datetime(2026, 10, 15, tzinfo=UTC)))["series_ticker"].to_list()
    ser = sorted(x for x in ser if _h(f"fwd:{x}") < FWD_SERIES_SHARE)
    print(f"series: {len(ser)}", flush=True)
    mk: list[dict[str, Any]] = []
    for i, sr in enumerate(ser, 1):
        cursor = None
        while True:
            prm = {"series_ticker": sr, "status": "settled", "min_close_ts": int(FWD_FROM.timestamp()),
                   "max_close_ts": int(datetime(2026, 11, 17, tzinfo=UTC).timestamp()), "limit": 1000,
                   "mve_filter": "exclude", "cursor": cursor}
            page = api.get("/markets", prm, tag=TAG)
            mk += k.market_rows(page.payload, info, "live")
            cursor = page.payload.get("cursor")
            if not cursor or not page.payload.get("markets"):
                break
        if i % 25 == 0:
            print(f"  series {i}/{len(ser)}: {len(mk)} markets, {api.n} requests", flush=True)
    st = event_starts(ms)
    evs = st.filter(pl.col("ms_category").is_in(["Mentions", "Entertainment"]) & (pl.col("start") >= FWD_FROM)
                    & (pl.col("start") < FWD_TO))["event"].to_list()
    print(f"mention / entertainment events: {len(evs)}", flush=True)
    for e in evs:
        try:
            mk += k.market_rows(api.get("/markets", {"event_ticker": e, "limit": 1000}, tag=TAG).payload, info, "live")
        except RuntimeError as error:
            print(f"skip {e}: {str(error)[-120:]}", flush=True)
    api.log("forward markets")
    df = k.markets_frame(mk, non_financial_only=False).unique("ticker", keep="first")
    df = df.with_columns(pl.col("category").fill_null(pl.when(pl.col("series_ticker").str.contains("MENTION")).then(pl.lit("Mentions"))))
    _save(df, OUT / "kalshi_markets.parquet")
    print(f"forward markets: {df.height}, settled yes/no {df.filter(pl.col('result').is_in(['yes', 'no'])).height}, "
          f"{api.n} requests", flush=True)
    return df


def run_forward_universe() -> pl.DataFrame:
    ms = pl.read_parquet(OUT / "milestones.parquet")
    evu = event_universe(event_starts(ms), tables=OUT)
    L = leg_universe(evu, tables=OUT)
    spec = {g: v for g, v in SAMPLE.items() if g != "esports"}
    S = sample(L, spec)
    S = S.filter(pl.col("group").is_in(["mentions", "mentions earnings", "entertainment", *spec]))
    _save(S, OUT / "legs.parquet")
    print(S.group_by("group", "route", "holdout").agg(pl.len().alias("legs"), pl.col("game").n_unique().alias("games"))
          .sort("group"))
    return S


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.pre_event")
    ap.add_argument("step", choices=["milestones", "universe", "hourly", "rows", "minute", "score", "all"])
    ap.add_argument("--forward", action="store_true", help="the November forward check (OUT/forward)")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.forward:
        use_forward()
    if a.step in ("milestones", "all"):
        run_forward_markets() if a.forward else run_milestones()
    if a.step in ("universe", "all"):
        run_forward_universe() if a.forward else run_universe()
    if a.step in ("hourly", "all"):
        run_hourly()
    if a.step in ("rows", "all"):
        run_rows()
    if a.step in ("minute", "all"):
        run_minute()
    if a.step in ("score", "all"):
        for k, v in run_score().items():
            print(k)
            print(v)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
