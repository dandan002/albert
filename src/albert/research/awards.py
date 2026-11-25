"""Awards and annual culture / sport-award events on Kalshi (execution venue; Polymarket data only).

Universe: every winner-take-all award series on Kalshi (Oscars, Golden Globes, Critics Choice, Actor
Awards/SAG, BAFTA, DGA/PGA, festivals, Emmys, Grammys, Latin Grammys, VMAs/AMAs/CMAs/BET, The Game Awards,
Steam/App Store awards, Time POY, Sexiest Man Alive, Word of the Year, Booker/National Book/Tony, Eurovision,
Nobel prizes, Ballon d'Or, voted sports awards). Nomination, count, attendance, performer, host, viewership
and mention series are excluded (``FAMILIES``).

Hypotheses and the 24 pre-registered cells are in ``data/research/awards/hypotheses.md`` (written before
prices or outcomes were opened). Cells:

* A maker NO on longshots (the cmn mechanism), B taker NO on longshots, C taker YES on the favourite,
  D taker NO on fandom legs, all at event close - h;
* E precursor: after a precursor ceremony's Kalshi market closes (e.g. the Golden Globe for Best Actor),
  trade the same nominee's leg in the downstream award (the Oscar for Best Actor) at close + lag.

Execution: takers pay the hourly candle's closing ask (YES) or 1 - bid (NO), ~$2 per order, <= 20
contracts, fee 0.07 x mult x n x P(1-P) rounded up per order; makers fill only when a later hourly candle
trades strictly through the quote (validated against prints by the culture_flow review), 1 contract,
maker fee where the series charges one. Held to settlement. Sharpe / vol from ``vol_metrics`` ledgers.

Raw API responses: read-through of the shared ``data/raw/kalshi`` cache (read only); new requests are
cached append-only under ``data/cache/awards/raw``.

    python -m albert.research.awards robots | universe | candles | panel | dev | freeze | holdout | forward
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import time
import unicodedata
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

OUT = Path("data/research/awards")
CACHE = Path("data/cache/awards")
SHARED_RAW = Path("data/raw")
DEV_END = datetime(2026, 10, 1, tzinfo=UTC)
HOLDOUT_END = datetime(2026, 11, 23, tzinfo=UTC)
UA = "albert-research/0.2"

# ---- universe ------------------------------------------------------------------------------------
FAMILIES: tuple[tuple[str, str], ...] = (
    ("oscar", (r"^(KX)?OSCAR(PIC|DIR|ACTO|ACTR|SUPACTO|SUPACTR|SPLAY|ASPLAY|ANIMATED|INTLFILM|CINE|COSTUME|DOCU|DS|"
              r"EDIT|LAS|AS|MAH|PROD|SCORE|SONG|SOUND|VIS|S)$")),
    ("sag", r"^(KX)?(SAGAWARD(ACTO|ACTR|SUPACTO|SUPACTR|ENSEMBLE|CASTCOMEDY|CASTDRAMA)|OSCARAWARDACTR)$"),
    ("bafta", r"^(KX)?BAFTA[A-Z]+$"),
    ("gg", r"^KXGG(?!G?NOM)[A-Z]+$"),
    ("critics", r"^KXCRITICS(?!CHOICENOM|NOMPIC)[A-Z]+$"),
    ("guild", r"^KX(DGAAWARDS|PGAAWARDS|FISPIRITAWARDS)$"),
    ("festival", r"^KX(CANNES[A-Z]+|VENICEFILMFESTIVAL[A-Z]*|VFILMFESTIVAL|TIFF)$"),
    ("emmy", r"^(KX)?EMMY(?!NOMS|COUNT)[A-Z]+$|^KX(KX)?EMMY$|^KXDTEMMYS$|^KXSPORTSEMMY$"),
    ("grammy", r"^(KX)?GRAM(AOTY|BNA|BRA|BRS|RAOTY|ROTY|SOTY|BCA|BCS|BPDGP|BPS|BPVA|BRBS|POTY|SWOTY)$|^KXGRAMMY$"),
    ("latin_grammy", r"^KX(LATINGRAMMY(AOTY|BNA|ROTY|SOTY)|LGRAM)$"),
    ("music", (r"^KX(VMA|VMABNA|VMASOTY|MTVVOFTY|AMA|AMAALBOTY|AMASOTY|AMASSOTY|CMA|AWARDSCMAAOTY|AWARDSCMASOOTY|"
              r"ACMAAOTY|ACMASOTY|BET|BETAOTY|BETAWARDS|BETBNA|BETDBJBGIA|BETSMOTYA|BETSWOTYA|BETVCA|BETYSA|IHEART)$")),
    ("games", (r"^(KX)?GAMEAWARDS[A-Z]*$|^KX(STEAM(BGOSD|BGYSA|BS|BWF|GOTY|LOL|MIG|OSRG|OVS|SBAR|VRGOTY)|"
              r"APPSTORE(ARCADE|CULTURE|IPADGAME|IPHONEGAME|MACGAME|VISIONGAME)|STRMRGOTY|STRMRTSA|STRMRSOTY|"
              r"STREAMERAWARDS)$")),
    ("person", r"^(KX)?TIME$|^KX(FT|SEXYMAN|WORDOFTHEYEAR|MISSUNIVERSE)$"),
    ("books_stage", r"^KX(BOOKERPRIZE|BOOKAWARDS(FIC|NONF|POETRY|TLIT|YPLIT)|NATBOOKAWARDS|JAMESBEARD|TONYAWARDS)$"),
    ("eurovision", r"^KXEUROVISION(JURY|TELEVOTE|ASIA)?$"),
    ("nobel", r"^(KX)?NOBEL(PEACE|CHEM|LIT|MED|PHYSICS|ECON)$"),
    ("ballon", r"^KX(SUPER)?BALLONDOR(AWARD)?$"),
    ("sports", (r"^KX(MLB(AL|NL)(MVP|CY|ROTY|MOTY|CPOTY)|ALMVP|NLMVP|NBA(MVP|ROY|DPOY|COY|MIMP|SIXTH|CLUTCH)|"
               r"NFL(MVP|DPOY|DPOTY|OROTY|DROTY|DROY|CPOTY|COTY|COACH|ASSCOACH|WPMOTY)|NHL(ADAMS|CALDER|NORRIS|"
               r"VEZINA|HART)|CONNSMYTHE|WNBA(MVP|ROY|ROTY|COY|DPOY|MIMP)|HEISMAN|CFBCOTY|CFBDOBRIEN|NCAAFAWARD|"
               r"NCAAFCOTY|NCAAMBNAISMITH|NCAAMBCOTY|HOBEYBAKER|LAXTEWAARATON|NCAABBGS)$")),
)
GROUP = {"oscar": "film", "sag": "film", "bafta": "film", "gg": "film", "critics": "film", "guild": "film",
         "festival": "film", "emmy": "tv", "grammy": "music", "latin_grammy": "music", "music": "music",
         "games": "games", "person": "person", "books_stage": "other", "eurovision": "other", "nobel": "nobel",
         "ballon": "sports", "sports": "sports"}
# announcement time not public in advance -> no "close - h" decision is implementable
UNSCHEDULED = r"^KX(SEXYMAN|WORDOFTHEYEAR|FT)$|^(KX)?TIME$"
# decided days-weeks before a low-security announcement (juries / ballots; leaks are on record)
LEAKY_FAMILIES = ("nobel", "ballon", "person", "books_stage", "sports")


def family_of(series: str | None) -> str | None:
    for fam, rx in FAMILIES:
        if re.match(rx, series or ""):
            return fam
    return None


def family_expr(col: str = "series_ticker") -> pl.Expr:
    return pl.col(col).map_elements(family_of, return_dtype=pl.String)


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


class ReadThroughClient:
    """``CachedClient`` whose cached GETs are first looked up (read only) in the shared raw cache, so
    requests other studies already made cost nothing; new responses go to ``data/cache/awards/raw``."""

    def __init__(self, venue: str, base: str, *, min_interval: float = 0.6) -> None:
        from albert.http import CachedClient

        self.inner = CachedClient(venue, base, CACHE / "raw", min_interval=min_interval, max_retries=12, user_agent=UA)
        self.shared = CachedClient(venue, base, SHARED_RAW, min_interval=min_interval, user_agent=UA)

    def get(self, path: str, params: dict[str, Any] | None = None, *, cache: bool = True, tag: str | None = None):
        from albert.http import RAW_SUFFIX, Fetched, read_raw

        if cache:
            p = {k: v for k, v in (params or {}).items() if v is not None}
            url = f"{self.shared.base_url}{path}"
            key = self.shared.cache_key(url, p if tag is None else {**p, "__tag": tag})
            for f in (self.shared.raw_dir / f"{key}{RAW_SUFFIX}", self.shared.raw_dir / f"{key}.json"):
                if f.exists():
                    rec = read_raw(f)
                    return Fetched(url, p, datetime.fromisoformat(rec["fetched_at"]), f, rec["response"], True)
        return self.inner.get(path, params, cache=cache, tag=tag)


def _api(min_interval: float = 0.6):
    from albert.config import Settings
    from albert.venues import kalshi as k

    return k.Kalshi(ReadThroughClient("kalshi", Settings().kalshi_base, min_interval=min_interval))


def _pages(api, path: str, params: dict[str, Any], cache: bool = True):
    cursor = None
    while True:
        page = api.client.get(path, {**params, "cursor": cursor}, cache=cache)
        yield page
        cursor = page.payload.get("cursor")
        if not cursor or not page.payload.get("markets"):
            return


def award_series() -> pl.DataFrame:
    """Award series from today's ``kalshi_series`` table, minus anything ``albert.categories`` calls
    financial (drops the Nobel economics prize, category Economics)."""
    from albert.categories import kalshi_series_is_financial

    s = pl.read_parquet("data/tables/kalshi_series.parquet")
    s = s.with_columns(family_expr().alias("family")).filter(pl.col("family").is_not_null())
    fin = [kalshi_series_is_financial(a, b, c) for a, b, c in s.select("series_ticker", "category", "title").iter_rows()]
    return (s.filter(~pl.Series(fin)).with_columns(pl.col("family").replace_strict(GROUP).alias("group"))
            .sort("series_ticker"))


def universe(*, log=print) -> pl.DataFrame:
    """Every settled market of every award series (``/historical/markets`` + ``/markets?status=settled``)."""
    from albert.research.counts import market_row

    ser = award_series()
    api = _api()
    rows: list[dict[str, Any]] = []
    for i, s in enumerate(ser["series_ticker"].to_list()):
        for page in _pages(api, "/historical/markets", {"series_ticker": s, "limit": 1000}):
            rows.extend(market_row(m, s, "hist") for m in page.payload.get("markets") or [])
        for page in _pages(api, "/markets", {"series_ticker": s, "status": "settled", "limit": 1000}):
            rows.extend(market_row(m, s, "live") for m in page.payload.get("markets") or [])
        if i % 25 == 0:
            log(f"  universe {i + 1}/{ser.height} {s}: {len(rows)} markets", flush=True)
    from albert.research.counts import MARKET_COLS

    mk = pl.DataFrame(rows, schema=MARKET_COLS).unique("ticker", keep="last")
    mk = (mk.filter(pl.col("result").is_in(["yes", "no"]))
          .join(ser.select("series_ticker", "family", "group", "category", "fee_type", "fee_multiplier"),
                on="series_ticker", how="left")
          .with_columns(pl.col("fee_multiplier").fill_null(1.0), (pl.col("result") == "yes").cast(pl.Int8).alias("y")))
    _save(ser, OUT / "series.parquet")
    _save(mk, OUT / "markets.parquet")
    log(f"universe: {ser.height} series, {mk.height} settled legs, {mk['event_ticker'].n_unique()} events, "
        f"close {mk['close_time'].min()} .. {mk['close_time'].max()}")
    return mk


# ---- events and precursor pairs ------------------------------------------------------------------
def split_of(col: str = "ev_close") -> pl.Expr:
    return (pl.when(pl.col(col) < DEV_END).then(pl.lit("dev"))
            .when(pl.col(col) < HOLDOUT_END).then(pl.lit("holdout")).otherwise(pl.lit("later")).alias("split"))


def events_table(mk: pl.DataFrame) -> pl.DataFrame:
    """One row per event: close = latest leg close (awards close at the announcement), first listing,
    legs, winners, whether the announcement time is public in advance, and whether it is leak-prone."""
    ev = mk.group_by("event_ticker").agg(
        pl.col("series_ticker").first(), pl.col("family").first(), pl.col("group").first(),
        pl.col("close_time").max().alias("ev_close"), pl.col("open_time").min().alias("ev_open"),
        pl.len().alias("legs"), pl.col("y").sum().alias("n_yes"), pl.col("volume").sum().alias("ev_volume"))
    still_open = open_event_tickers()
    return ev.with_columns(
        (~pl.col("series_ticker").str.contains(UNSCHEDULED)).alias("scheduled"),
        pl.col("family").is_in(LEAKY_FAMILIES).alias("leaky"), split_of(),
        (~pl.col("event_ticker").is_in(still_open)).alias("complete")).sort("ev_close")


def open_event_tickers() -> list[str]:
    """Events that still had open legs at the ``forward`` snapshot: their settled legs are nominees
    eliminated early (e.g. at the nominations), so the event is not over and enters no test."""
    f = OUT / "open_events.parquet"
    return pl.read_parquet(f)["event_ticker"].to_list() if f.exists() else []


# category keys: (family, series suffix) -> keys shared by precursor and downstream awards
_FILM = {"PIC": ["picture"], "DIR": ["director"], "ACTO": ["actor"], "ACTR": ["actress"], "SUPACTO": ["sup_actor"],
         "SUPACTR": ["sup_actress"], "SPLAY": ["orig_screenplay"], "OSPLAY": ["orig_screenplay"],
         "ASPLAY": ["adapt_screenplay"], "ANIMATED": ["animated"], "ANIM": ["animated"], "AFILM": ["animated"],
         "INTLFILM": ["intl"], "FOREIGN": ["intl"], "FOREIGNFILM": ["intl"], "CINE": ["cine"], "COSTUME": ["costume"],
         "COST": ["costume"], "EDIT": ["edit"], "MAH": ["makeup"], "HAIR": ["makeup"], "PROD": ["prod_design"],
         "PRODDES": ["prod_design"], "SCORE": ["score"], "SONG": ["song"], "SOUND": ["sound"], "VIS": ["vfx"],
         "DOCU": ["doc"], "FILM": ["picture"], "ENSEMBLE": ["picture"], "DRAMAFILM": ["picture"],
         "MCOMFILM": ["picture"], "DRAMAFILMACTO": ["actor"], "MCOMFILMACTO": ["actor"], "DRAMAFILMACTR": ["actress"],
         "MCOMFILMACTR": ["actress"], "FILMSUPACTO": ["sup_actor"], "FILMSUPACTR": ["sup_actress"],
         "SCREEN": ["orig_screenplay", "adapt_screenplay"]}
_TV = {"CSERIES": ["tv_comedy"], "DSERIES": ["tv_drama"], "LSERIES": ["tv_limited"], "CACTO": ["tv_comedy_actor"],
       "CACTR": ["tv_comedy_actress"], "DACTO": ["tv_drama_actor"], "DACTR": ["tv_drama_actress"],
       "LIMITEDACTO": ["tv_limited_actor"], "LIMITEDACTR": ["tv_limited_actress"], "DRAMATV": ["tv_drama"],
       "MCOMTV": ["tv_comedy"], "DRAMA": ["tv_drama"], "COMEDY": ["tv_comedy"], "DRAMATVACTO": ["tv_drama_actor"],
       "DRAMATVACTR": ["tv_drama_actress"], "MCOMTVACTO": ["tv_comedy_actor"], "MCOMTVACTR": ["tv_comedy_actress"],
       "DRAMAACTO": ["tv_drama_actor"], "DRAMAACTR": ["tv_drama_actress"], "COMEDYACTO": ["tv_comedy_actor"],
       "COMEDYACTR": ["tv_comedy_actress"], "LSERIESACTO": ["tv_limited_actor"], "LSERIESACTR": ["tv_limited_actress"],
       "CASTCOMEDY": ["tv_comedy"], "CASTDRAMA": ["tv_drama"]}
_PREFIX = {"oscar": r"^(KX)?OSCAR", "sag": r"^(KX)?(SAGAWARD|OSCARAWARD)", "bafta": r"^(KX)?BAFTA", "gg": r"^KXGG",
           "critics": r"^KXCRITICS", "emmy": r"^(KX)?EMMY"}
_TV_ONLY = {"gg": {"LSERIES"}, "critics": {"DRAMA", "COMEDY", "LSERIES"}}   # suffixes that are TV in these families


def category_keys(series: str, family: str | None) -> list[str]:
    if family == "guild":
        return {"KXDGAAWARDS": ["director"], "KXPGAAWARDS": ["picture"]}.get(series, [])
    rx = _PREFIX.get(family or "")
    if rx is None:
        return []
    suf = re.sub(rx, "", series)
    if family == "emmy" or suf in _TV_ONLY.get(family or "", set()) or suf.startswith("CAST"):
        return _TV.get(suf, [])
    if family == "oscar" and suf == "S":
        return []
    return _FILM.get(suf) or _TV.get(suf, [])


def norm_name(s: str | None) -> str:
    """Nominee name for matching across ceremonies: accents, case, punctuation and a trailing
    parenthetical (film / show) stripped."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"\(.*?\)", " ", s.lower()).replace("&", " and ")
    s = re.sub(r"^the\s+", "", re.sub(r"[^a-z0-9 ]+", " ", s).strip())
    return re.sub(r"\s+", " ", s).strip()


def names_match(a: str, b: str) -> bool:
    if not a or not b or a in ("tie", "no winner") or b in ("tie", "no winner"):
        return False
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    return len(ta) >= 2 and len(tb) >= 2 and (ta <= tb or tb <= ta)


MAX_PAIR_GAP = timedelta(days=120)


def precursor_pairs(mk: pl.DataFrame, ev: pl.DataFrame) -> pl.DataFrame:
    """(precursor event P, downstream event F, a P leg, the F leg of the same nominee): same category key,
    P closes before F and at most ``MAX_PAIR_GAP`` earlier, different ceremonies. ``p_y`` is the P leg's
    result (known at P's close); events with a tie or no single winner are skipped. Uses no prices."""
    e = ev.filter(pl.col("family").is_in(["oscar", "sag", "bafta", "gg", "critics", "guild", "emmy"])
                  & pl.col("complete"))
    keys = [(r["event_ticker"], k, r["ev_close"], r["family"]) for r in e.iter_rows(named=True)
            for k in category_keys(r["series_ticker"], r["family"])]
    kdf = pl.DataFrame(keys, schema=["event_ticker", "key", "ev_close", "family"], orient="row")
    legs = mk.select("event_ticker", "ticker", "yes_sub_title", "y", "open_time", "close_time").with_columns(
        pl.col("yes_sub_title").map_elements(norm_name, return_dtype=pl.String).alias("name"))
    pr = kdf.join(kdf, on="key", suffix="_f").filter(
        (pl.col("event_ticker") != pl.col("event_ticker_f")) & (pl.col("ev_close") < pl.col("ev_close_f"))
        & (pl.col("ev_close_f") - pl.col("ev_close") <= MAX_PAIR_GAP) & (pl.col("family") != pl.col("family_f")))
    pr = pr.rename({"event_ticker": "p_event", "ev_close": "p_close", "family": "p_family",
                    "event_ticker_f": "f_event", "ev_close_f": "f_close", "family_f": "f_family"})
    one_winner = ev.filter(pl.col("n_yes") == 1)["event_ticker"]       # P's result must identify one winner
    pr = pr.unique(["p_event", "f_event"]).filter(pl.col("p_event").is_in(one_winner.implode()))
    pl_ = legs.rename({"event_ticker": "p_event", "ticker": "p_ticker", "name": "p_name", "y": "p_y"}).select(
        "p_event", "p_ticker", "p_name", "p_y")
    fl = legs.rename({"event_ticker": "f_event", "ticker": "f_ticker", "name": "f_name", "y": "f_y",
                      "open_time": "f_open", "close_time": "f_leg_close"}).drop("yes_sub_title")
    j = pr.join(pl_, on="p_event", how="inner").join(fl, on="f_event", how="inner")
    j = j.filter(pl.struct("p_name", "f_name").map_elements(lambda r: names_match(r["p_name"], r["f_name"]),
                                                           return_dtype=pl.Boolean))
    return j.sort("p_close", "f_event", "p_ticker")


# ---- hourly candles ----------------------------------------------------------------------------
CANDLE_DIR = CACHE / "candles"
BASE_LOOKBACK = timedelta(days=4)


def candle_windows(mk: pl.DataFrame, ev: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """Per leg: [max(open, event close - 4 d, or earliest precursor close - 3 h for downstream legs), close]."""
    w = mk.join(ev.select("event_ticker", "ev_close"), on="event_ticker").with_columns(
        (pl.col("ev_close") - BASE_LOOKBACK).alias("lo0"))
    if pairs.height:
        early = (pairs.filter(pl.col("p_y") == 1).group_by("f_event").agg(pl.col("p_close").min().alias("p0"))
                 .rename({"f_event": "event_ticker"}))
        w = w.join(early, on="event_ticker", how="left").with_columns(
            pl.min_horizontal("lo0", pl.col("p0") - timedelta(hours=3)).alias("lo0"))
    return w.select("ticker", "event_ticker", "settlement_ts", pl.lit(60).alias("period"),
                    pl.max_horizontal("open_time", "lo0").alias("lo"), pl.col("close_time").alias("hi"))


def _have() -> set[str]:
    done = CACHE / "candles_done.parquet"
    return set(pl.read_parquet(done)["ticker"].to_list()) if done.exists() else set()


# Dev sample for cells A-D (volume-blind: seeded hash of the event ticker). Historical candles cost one
# request per leg, so dev uses every event the precursor study touches plus at most this many events per group.
DEV_SAMPLE_CAP = {"film": 0, "music": 15, "tv": 15, "games": 15, "other": 15, "sports": 6, "nobel": 40,
                  "person": 40}


def alive_legs(mk: pl.DataFrame, ev: pl.DataFrame) -> pl.DataFrame:
    """Legs of scheduled events with >= 2 legs that can enter a cell: listed before close - 24 h and still
    open at close - 72 h (legs eliminated earlier, e.g. at the nominations, settle before any decision)."""
    e = ev.filter(pl.col("scheduled") & (pl.col("legs") >= 2) & pl.col("complete"))
    return mk.join(e.select("event_ticker", "ev_close", "split"), on="event_ticker").filter(
        (pl.col("close_time") > pl.col("ev_close") - timedelta(hours=72))
        & (pl.col("open_time") < pl.col("ev_close") - timedelta(hours=24)))


def layer_tickers(layer: str, mk: pl.DataFrame, ev: pl.DataFrame, pairs: pl.DataFrame) -> list[str]:
    """holdout: every alive leg of holdout events; precursor: every leg in a precursor pair plus every alive
    leg of the events those legs belong to (so those events are complete); devsample: alive legs of a seeded
    sample of the other dev events (``DEV_SAMPLE_CAP`` per group)."""
    al = alive_legs(mk, ev)
    if layer == "holdout":
        return al.filter(pl.col("split") == "holdout")["ticker"].to_list()
    pe = set(pairs["p_event"].to_list()) | set(pairs["f_event"].to_list())
    if layer == "precursor":
        return sorted(set(pairs["f_ticker"].to_list()) | set(pairs["p_ticker"].to_list())
                      | set(al.filter(pl.col("event_ticker").is_in(list(pe)))["ticker"].to_list()))
    if layer == "devsample":
        return al.filter(pl.col("event_ticker").is_in(dev_sample_events(ev, pe)))["ticker"].to_list()
    raise ValueError(layer)


def _event_hash(t: str) -> int:
    import hashlib

    return int(hashlib.sha256(f"awards-2026-11-23|{t}".encode()).hexdigest()[:12], 16)


def dev_sample_events(ev: pl.DataFrame, exclude: set[str]) -> list[str]:
    e = ev.filter((pl.col("split") == "dev") & pl.col("scheduled") & (pl.col("legs") >= 2) & pl.col("complete")
                  & ~pl.col("event_ticker").is_in(list(exclude)))
    e = e.with_columns(pl.col("event_ticker").map_elements(_event_hash, return_dtype=pl.Int64).alias("h"))
    out: list[str] = []
    for (g,), x in e.sort("h").group_by("group", maintain_order=True):
        out += x["event_ticker"].head(DEV_SAMPLE_CAP.get(g, 40)).to_list()
    return sorted(out)


def fetch_candles(layer: str, *, workers: int = 2, min_interval: float = 0.6, log=print) -> None:
    """Hourly OHLC (trade price, bid, ask) per leg window for one layer; historical legs one request each,
    live legs batched per event (``xcat`` helpers). Written as ``part-*.parquet`` under
    ``data/cache/awards/candles`` (``vol_metrics.load_marks`` reads that glob)."""
    import threading
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    from albert.research import xcat

    mk = pl.read_parquet(OUT / "markets.parquet")
    ev = events_table(mk)
    pairs = precursor_pairs(mk, ev)
    want = layer_tickers(layer, mk, ev, pairs)
    win = candle_windows(mk, ev, pairs).filter((pl.col("hi") > pl.col("lo")) & pl.col("ticker").is_in(want))
    have = _have()
    todo = win.filter(~pl.col("ticker").is_in(list(have)))
    # one event at a time (seeded hash order), so an interrupted fetch leaves whole events, not scattered legs
    todo = todo.with_columns(pl.col("event_ticker").map_elements(_event_hash, return_dtype=pl.Int64).alias("eh")
                             ).sort("eh", "ticker").drop("eh")
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    api = _api(min_interval)
    cutoff = api.cutoff()["market_settled_ts"]
    jobs = xcat._jobs(todo, cutoff)
    log(f"candles: {win.height} legs, {len(have)} done, {todo.height} to fetch in {len(jobs)} requests", flush=True)
    lock = threading.Lock()
    buf: list[dict[str, Any]] = []
    fin: list[str] = []
    failed: list[str] = []
    part = len(list(CANDLE_DIR.glob("part-*.parquet")))
    t0 = time.time()

    def flush() -> None:
        nonlocal buf, part
        if buf:
            pl.DataFrame(buf, schema=xcat.OHLC_SCHEMA).write_parquet(CANDLE_DIR / f"part-{part:05d}.parquet")
            part += 1
        buf = []
        pl.DataFrame({"ticker": sorted(have | set(fin))}, schema={"ticker": pl.String}).write_parquet(
            CACHE / "candles_done.parquet")
        if failed:
            (CACHE / "candles_failed.json").write_text(json.dumps(sorted(set(failed))))

    n = 0
    # results are consumed as they complete (an ordered map would pile up finished results behind one slow,
    # retrying request) and flushed every 50 requests or 40k candles, so memory stays small
    todo_jobs = iter(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        live = {pool.submit(xcat._run_job, api, j) for j in itertools.islice(todo_jobs, 2 * workers)}
        while live:
            done, live = wait(live, return_when=FIRST_COMPLETED)
            for j in itertools.islice(todo_jobs, len(done)):
                live.add(pool.submit(xcat._run_job, api, j))
            res = [x for fut in done for x in fut.result()]
            k = len(done)
            del done
            with lock:
                for tk, got in res:
                    if got is None:
                        failed.append(tk)
                    else:
                        buf.extend(got)
                        fin.append(tk)
                if (n + k) // 50 > n // 50 or len(buf) > 15_000:
                    flush()
                if (n + k) // 100 > n // 100:
                    log(f"  candles {n + k}/{len(jobs)} requests, {(n + k) / (time.time() - t0):.2f} req/s", flush=True)
                n += k
    flush()
    log(f"candles done: {n} requests, {len(set(failed))} failed")


def load_candles(tickers: list[str] | None = None) -> pl.DataFrame:
    lf = pl.scan_parquet(sorted(CANDLE_DIR.glob("part-*.parquet")))
    if tickers is not None:
        lf = lf.filter(pl.col("ticker").is_in(tickers))
    return (lf.select("ticker", "end_ts", "p_high", "p_low", "p_close", "bid_close", "ask_close", "volume")
            .collect().unique(["ticker", "end_ts"], keep="first").sort("ticker", "end_ts"))


# ---- quotes and cells ---------------------------------------------------------------------------
MAX_QUOTE_AGE = timedelta(days=7)
MAX_SPREAD = 0.10
TAKER_BUDGET = 2.0
TAKER_MAX_N = 20
HOURS = (72, 24)
MAKER_BANDS = ((0.0, 0.05), (0.05, 0.15))
MAKER_MARGINS = (0.06, 0.12)
NO_BANDS = ((0.05, 0.15), (0.15, 0.35))
FAV_BANDS = ((0.50, 0.80), (0.80, 0.95))
FANDOM_BAND = (0.02, 0.35)
MAX_LEGS = 3
LAGS = (1, 6)
PRIMARY = "A_h72_b0-5_m12"


def quotes_at(points: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    """(ticker, t) -> last hourly candle ending at or before t (<= 7 days old): bid, ask, mid (two-sided
    0 < bid < ask < 1 only), spread."""
    c = candles.select("ticker", "end_ts", pl.col("bid_close").alias("bid"), pl.col("ask_close").alias("ask"),
                       pl.col("p_close").alias("last"))
    q = points.sort("t").join_asof(c.sort("end_ts"), left_on="t", right_on="end_ts", by="ticker",
                                   strategy="backward", check_sortedness=False)
    fresh = pl.col("end_ts").is_not_null() & (pl.col("t") - pl.col("end_ts") <= MAX_QUOTE_AGE)
    two = fresh & (pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("ask") > pl.col("bid"))
    return q.with_columns(
        pl.when(fresh).then(pl.col("bid")).alias("bid"), pl.when(fresh).then(pl.col("ask")).alias("ask"),
        pl.when(two).then((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
        pl.when(two).then(pl.col("ask") - pl.col("bid")).alias("spread"), two.alias("two_sided"))


def ceremony_expr(family: str, close: str) -> pl.Expr:
    """Cluster key for one ceremony: award family + the US Eastern date of the event close."""
    return (pl.col(family) + "|" + pl.col(close).dt.convert_time_zone("America/New_York").dt.date().cast(pl.String)
            ).alias("ceremony")


def decision_panel(mk: pl.DataFrame, ev: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    """Every leg listed before ``t = event close - h`` (h in HOURS) and still open, of scheduled events with
    >= 2 legs, with its quote at t and the mid rank within the event (1 = favourite)."""
    from albert.research.behav_culture import fandom_expr

    e = ev.filter(pl.col("scheduled") & (pl.col("legs") >= 2) & pl.col("complete"))
    frames = []
    for h in HOURS:
        x = (mk.join(e.select("event_ticker", "ev_close", "split", "leaky"), on="event_ticker")
             .with_columns((pl.col("ev_close") - timedelta(hours=h)).alias("t"), pl.lit(h).alias("h"))
             .filter((pl.col("open_time") < pl.col("t")) & (pl.col("close_time") > pl.col("t"))))
        frames.append(x)
    x = pl.concat(frames)
    x = x.select("event_ticker", "series_ticker", "family", "group", "ticker", "yes_sub_title", "y", "h", "t",
                 "close_time", "settlement_ts", "ev_close", "split", "leaky", "fee_type", "fee_multiplier",
                 ceremony_expr("family", "ev_close"))
    q = quotes_at(x, candles)
    q = q.with_columns(
        pl.col("mid").rank("ordinal", descending=True).over("event_ticker", "h").alias("mid_rank"),
        pl.col("mid").is_not_null().sum().over("event_ticker", "h").alias("n_quoted"),
        fandom_expr(pl.col("yes_sub_title")).alias("fandom"))
    return q.sort("event_ticker", "h", "ticker")


def _fee(n: int, price: float, *, maker: bool, fee_type: str | None, mult: float | None) -> float:
    from albert.fees import kalshi_fee

    return kalshi_fee(n, price, maker=maker, fee_type=fee_type, multiplier=mult)


def taker_rows(df: pl.DataFrame, side: str) -> pl.DataFrame:
    """Taker orders at t: YES at the ask / NO at 1 - bid, ~$2 (<= 20 contracts), fee rounded up per order,
    held to settlement. Adds price, n, fee, cost_usd, payoff_usd, pnl_c (cents per contract, net)."""
    if not df.height:
        return df
    price = pl.col("ask") if side == "yes" else 1 - pl.col("bid")
    win = pl.col("y") if side == "yes" else 1 - pl.col("y")
    d = df.with_columns(price.round(4).alias("price"), win.alias("payoff"), pl.lit(side).alias("side"),
                        pl.col("t").alias("t_in"))
    n = [max(1, min(TAKER_MAX_N, math.floor(TAKER_BUDGET / max(p_, 0.01) + 1e-9))) for p_ in d["price"].to_list()]
    fee = [_fee(k, p_, maker=False, fee_type=ft, mult=m) for k, p_, ft, m in
           zip(n, d["price"].to_list(), d["fee_type"].to_list(), d["fee_multiplier"].to_list(), strict=True)]
    d = d.with_columns(pl.Series("n", n, dtype=pl.Int64), pl.Series("fee", fee, dtype=pl.Float64))
    return d.with_columns(
        (pl.col("n") * pl.col("price") + pl.col("fee")).alias("cost_usd"),
        (pl.col("n") * pl.col("payoff")).cast(pl.Float64).alias("payoff_usd"),
        (100 * (pl.col("payoff") - pl.col("price") - pl.col("fee") / pl.col("n"))).alias("pnl_c"))


def maker_rows(panel: pl.DataFrame, candles: pl.DataFrame, h: int, band: tuple[float, float],
               margin: float) -> pl.DataFrame:
    """cmn-style maker NO: legs with a two-sided mid in (lo, hi], sell YES at ceil(mid + margin) (post-only:
    above the YES bid, below 99c), <= 3 legs per event (cheapest NO first). 1 contract; filled when a later
    hourly candle (ending >= t + 1 h) trades strictly above the sell price before the leg closes."""
    lo, hi = band
    o = panel.filter((pl.col("h") == h) & pl.col("two_sided") & (pl.col("mid") > lo + 1e-9)
                     & (pl.col("mid") <= hi + 1e-9))
    o = o.with_columns((((pl.col("mid") + margin) * 100 - 1e-6).ceil() / 100).round(2).alias("sell_at"))
    o = o.filter((pl.col("sell_at") < 0.99) & (pl.col("sell_at") > pl.col("bid") + 1e-9))
    o = (o.with_columns((1 - pl.col("sell_at")).round(2).alias("price"))
         .sort("event_ticker", "price", "ticker")
         .with_columns(pl.int_range(pl.len()).over("event_ticker").alias("leg_rank"))
         .filter(pl.col("leg_rank") < MAX_LEGS))
    if not o.height:
        return o
    c = candles.filter(pl.col("ticker").is_in(o["ticker"].unique().implode())).select("ticker", "end_ts", "p_high")
    f = (o.select("ticker", "t", "sell_at", "close_time").join(c, on="ticker")
         .filter((pl.col("end_ts") >= pl.col("t") + timedelta(hours=1))
                 & (pl.col("end_ts") <= pl.col("close_time") + timedelta(hours=1))
                 & (pl.col("p_high") > pl.col("sell_at") + 1e-9))
         .group_by("ticker", "t").agg(pl.col("end_ts").min().alias("fill_ts")))
    o = o.join(f, on=["ticker", "t"], how="left").with_columns(pl.lit("no").alias("side"), pl.lit(1, pl.Int64).alias("n"),
                                                                (1 - pl.col("y")).alias("payoff"))
    fee = [_fee(1, p_, maker=True, fee_type=ft, mult=m) for p_, ft, m in
           zip(o["price"].to_list(), o["fee_type"].to_list(), o["fee_multiplier"].to_list(), strict=True)]
    o = o.with_columns(pl.Series("fee", fee, dtype=pl.Float64), pl.col("fill_ts").alias("t_in"))
    return o.with_columns(
        (pl.col("price") + pl.col("fee")).alias("cost_usd"), pl.col("payoff").cast(pl.Float64).alias("payoff_usd"),
        (100 * (pl.col("payoff") - pl.col("price") - pl.col("fee"))).alias("pnl_c"))


def cell_orders(panel: pl.DataFrame, candles: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Every pre-registered A-D cell -> its orders (maker cells include unfilled orders; fill_ts null)."""
    out: dict[str, pl.DataFrame] = {}
    for h in HOURS:
        for b in MAKER_BANDS:
            for m in MAKER_MARGINS:
                out[f"A_h{h}_b{round(b[0] * 100)}-{round(b[1] * 100)}_m{round(m * 100)}"] = maker_rows(panel, candles, h, b, m)
        ok = (pl.col("h") == h) & pl.col("two_sided") & (pl.col("spread") <= MAX_SPREAD + 1e-9)
        for lo, hi in NO_BANDS:
            out[f"B_h{h}_no{round(lo * 100)}-{round(hi * 100)}"] = taker_rows(
                panel.filter(ok & pl.col("mid").is_between(lo, hi, closed="left")), "no")
        for lo, hi in FAV_BANDS:
            out[f"C_h{h}_fav{round(lo * 100)}-{round(hi * 100)}"] = taker_rows(
                panel.filter(ok & (pl.col("mid_rank") == 1) & pl.col("mid").is_between(lo, hi, closed="left")), "yes")
        lo, hi = FANDOM_BAND
        out[f"D_h{h}_fandom"] = taker_rows(
            panel.filter(ok & pl.col("fandom") & pl.col("mid").is_between(lo, hi, closed="both")), "no")
    return out


def precursor_orders(pairs: pl.DataFrame, candles: pl.DataFrame, mk: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """E cells. For each (P, F) pair: quotes of every P leg at P close - 1 h give the precursor favourite and
    the winner's surprise. The result counts as public at P close if P's winner last traded >= 90c by then,
    else at the winner's settlement time (P closed before the announcement). At public + lag, F's leg of P's
    winner is bought YES at the ask (E_win) and F's leg of P's favourite who lost is bought NO at 1 - bid
    (E_lose). Also returns the drift panel (F mids at public - 1 h, + 1 h, + 6 h, + 24 h)."""
    pev = pairs.select("p_event", "p_close").unique()
    plegs = mk.select(pl.col("event_ticker").alias("p_event"), "ticker", "y", "settlement_ts").join(pev, on="p_event")
    pq = quotes_at(plegs.with_columns((pl.col("p_close") - timedelta(hours=1)).alias("t")), candles)
    pq = pq.select("p_event", pl.col("ticker").alias("p_ticker"), pl.col("mid").alias("p_mid")).with_columns(
        pl.col("p_mid").rank("ordinal", descending=True).over("p_event").alias("p_rank"))
    last = quotes_at(plegs.filter(pl.col("y") == 1).with_columns(pl.col("p_close").alias("t")), candles)
    known = last.select("p_event", pl.when(pl.col("last") >= 0.9).then(pl.col("p_close"))
                        .otherwise(pl.max_horizontal("p_close", "settlement_ts")).alias("p_known"),
                        pl.col("last").alias("p_win_last"))
    j = pairs.join(pq, on=["p_event", "p_ticker"], how="left").join(known, on="p_event", how="left")
    meta = mk.select(pl.col("ticker").alias("f_ticker"), "settlement_ts", "fee_type", "fee_multiplier",
                     pl.col("series_ticker").alias("f_series"), pl.col("group"))
    j = j.join(meta, on="f_ticker", how="left").with_columns(
        pl.col("p_close").dt.year().alias("season"), pl.col("f_y").alias("y"), pl.col("f_event").alias("event_ticker"),
        pl.col("f_ticker").alias("ticker"), split_of("f_close").alias("split"), pl.lit(False).alias("leaky"),
        pl.col("f_leg_close").alias("close_time"), ceremony_expr("f_family", "f_close"),
        pl.coalesce("p_known", "p_close").alias("p_known"))
    out: dict[str, pl.DataFrame] = {}
    drift_pts = []
    for lag in LAGS:
        x = j.with_columns((pl.col("p_known") + timedelta(hours=lag)).alias("t")).filter(
            (pl.col("f_open") < pl.col("t")) & (pl.col("f_leg_close") > pl.col("t") + timedelta(hours=1)))
        q = quotes_at(x, candles).filter(pl.col("two_sided") & (pl.col("spread") <= MAX_SPREAD + 1e-9))
        win = q.filter(pl.col("p_y") == 1)
        out[f"E_win_lag{lag}"] = taker_rows(win, "yes")
        out[f"E_win_surprise_lag{lag}"] = taker_rows(win.filter(pl.col("p_mid") < 0.5), "yes")
        out[f"E_lose_lag{lag}"] = taker_rows(q.filter((pl.col("p_y") == 0) & (pl.col("p_rank") == 1)
                                                      & (pl.col("p_mid") >= 0.5)), "no")
    for off in (-1, 1, 6, 24):
        drift_pts.append(j.with_columns((pl.col("p_known") + timedelta(hours=off)).alias("t"), pl.lit(off).alias("off")))
    d = quotes_at(pl.concat(drift_pts).filter(pl.col("f_open") < pl.col("t")), candles)
    out["_drift"] = d.select("p_event", "p_close", "f_event", "f_ticker", "p_y", "p_rank", "p_mid", "p_win_last",
                             "season", "off", "mid", "y")
    return out


# ---- statistics, books, ledgers ------------------------------------------------------------------
MIN_CLUSTERS = 8


def summarize(o: pl.DataFrame) -> dict[str, Any]:
    """Filled orders: c/contract (order-level mean), t clustered by event and by ET date, P&L per $ risked,
    YES (loss) rate for NO cells."""
    from albert.research.counts import cluster_se

    f = o.filter(pl.col("t_in").is_not_null()) if "t_in" in o.columns and o.height else o
    if not f.height:
        return {"orders": o.height, "fills": 0}
    x = f["pnl_c"].to_numpy().astype(float)
    ev = f["event_ticker"].to_numpy().astype(str)
    day = f["t_in"].dt.convert_time_zone("America/New_York").dt.date().cast(pl.String).to_numpy().astype(str)
    m = float(x.mean())
    # a clustered t needs enough clusters to mean anything; below MIN_CLUSTERS it is not reported
    se_e = cluster_se(x, ev) if len(np.unique(ev)) >= MIN_CLUSTERS else float("nan")
    se_d = cluster_se(x, day) if len(np.unique(day)) >= MIN_CLUSTERS else float("nan")
    cer = f["ceremony"].to_numpy().astype(str) if "ceremony" in f.columns else ev
    se_c = cluster_se(x, cer) if len(np.unique(cer)) >= MIN_CLUSTERS else float("nan")
    cost, pay = float(f["cost_usd"].sum()), float(f["payoff_usd"].sum())
    return {"orders": o.height, "fills": f.height, "events": len(np.unique(ev)), "days": len(np.unique(day)),
            "c_per_contract": round(m, 2),
            "t_event": round(m / se_e, 2) if se_e and math.isfinite(se_e) and se_e > 0 else None,
            "t_date": round(m / se_d, 2) if se_d and math.isfinite(se_d) and se_d > 0 else None,
            "ceremonies": len(np.unique(cer)),
            "t_ceremony": round(m / se_c, 2) if se_c and math.isfinite(se_c) and se_c > 0 else None,
            "pnl_usd": round(pay - cost, 2), "risked_usd": round(cost, 2),
            "pnl_per_usd": round((pay - cost) / cost, 4) if cost > 0 else None,
            "side_won_pct": round(100 * float(f["payoff"].mean()), 1), "mean_price": round(float(f["price"].mean()), 3),
            "losses": int((f["payoff"] == 0).sum()),
            "losses_breakeven": round(float((1 - f["price"] - f["fee"] / f["n"]).sum()), 2),
            "p_breakeven": _r4(breakeven_p(f))}


def _r4(v: float | None) -> float | None:
    return None if v is None else round(v, 5)


def breakeven_p(f: pl.DataFrame) -> float | None:
    """One-sided p that the held side wins no more often than its all-in price implies: under H0 fill i
    loses with probability q_i = 1 - (price + fee per contract), independently; p = P(losses <= observed)
    (Poisson-binomial, exact). Unlike a clustered t it stays honest when a sample has no losses."""
    q = np.clip((1 - f["price"] - f["fee"] / f["n"]).to_numpy().astype(float), 0.0, 1.0)
    k = int((f["payoff"] == 0).sum())
    dist = np.zeros(len(q) + 1)
    dist[0] = 1.0
    for qi in q:
        dist[1:] = dist[1:] * (1 - qi) + dist[:-1] * qi
        dist[0] *= 1 - qi
    return float(dist[: k + 1].sum()) if len(q) else None


BOOK_CASH = 50.0        # mentions uses ~$40 of the $100
BOOK_EVENT_CAP = 5.0


def capped_book(o: pl.DataFrame, *, cash: float = BOOK_CASH, event_cap: float = BOOK_EVENT_CAP) -> pl.DataFrame:
    """Chronological cash cap: an order reserves its collateral (cost for takers; 1 - sell price for resting
    makers) from its decision time until settlement (filled) or the leg's close (unfilled maker). Orders that
    would push the total over ``cash`` or the event over ``event_cap`` are skipped. Returns filled, kept orders."""
    if not o.height:
        return o
    x = o.with_columns(pl.coalesce("settlement_ts", "close_time").alias("settle_ts"))
    x = x.with_columns(pl.when(pl.col("t_in").is_null()).then(pl.col("close_time")).otherwise(pl.col("settle_ts"))
                       .alias("release"), pl.coalesce("cost_usd", "price").alias("reserve")).sort("t", "ticker")
    open_: list[tuple[datetime, float, str]] = []
    keep = []
    for r in x.iter_rows(named=True):
        open_ = [z for z in open_ if z[0] > r["t"]]
        tot = sum(z[1] for z in open_)
        evt = sum(z[1] for z in open_ if z[2] == r["event_ticker"])
        if tot + r["reserve"] <= cash + 1e-9 and evt + r["reserve"] <= event_cap + 1e-9:
            open_.append((r["release"], r["reserve"], r["event_ticker"]))
            keep.append(True)
        else:
            keep.append(False)
    return x.filter(pl.Series(keep) & pl.col("t_in").is_not_null())


def ledger(o: pl.DataFrame, strategy: str, variant: str, split: str) -> pl.DataFrame:
    return o.select(pl.lit(strategy).alias("strategy"), pl.lit(variant).alias("variant"), pl.lit(split).alias("split"),
                    (pl.col("ticker") + "|" + pl.col("t").cast(pl.String)).alias("struct_id"), "event_ticker", "ticker",
                    "side", "n", "t_in", "cost_usd", "settle_ts", "payoff_usd")


def book_metrics(o: pl.DataFrame, name: str, split: str) -> dict[str, Any]:
    from albert.research import vol_metrics as vm

    b = capped_book(o)
    if b.height < 2:
        return {"positions": b.height}
    led = ledger(b, "awards", name, split)
    d = OUT / "ledgers"
    d.mkdir(parents=True, exist_ok=True)
    led.write_parquet(d / f"ledger_{name}_{split}.parquet")
    m = vm.ledger_metrics(led, vm.load_marks(led["ticker"].unique().to_list()))
    keep = ("positions", "events", "usd_per_week", "ann_return_pct", "ann_vol_pct", "sharpe", "sharpe_ci90",
            "sharpe_ex_top5", "max_dd_usd", "weeks_positive_pct", "pnl_usd", "first_day", "last_day", "marked_share")
    return {k: m.get(k) for k in keep}


# ---- drivers -----------------------------------------------------------------------------------------
PREDICTED = "pnl > 0 for every cell (each cell is a trade in the hypothesised direction)"


def _norm_sf(t: float) -> float:
    return 0.5 * math.erfc(t / math.sqrt(2))


def holm(pvals: dict[str, float | None], alpha: float = 0.05) -> dict[str, bool]:
    items = sorted(((p_, k) for k, p_ in pvals.items() if p_ is not None))
    m, out, alive = len(pvals), dict.fromkeys(pvals, False), True
    for i, (p_, k) in enumerate(items):
        alive = alive and p_ <= alpha / (m - i)
        out[k] = alive
    return out


def build_all(log=print) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    mk = pl.read_parquet(OUT / "markets.parquet")
    ev = events_table(mk)
    pairs = precursor_pairs(mk, ev)
    tick = list(set(mk["ticker"].to_list()))
    candles = load_candles(tick)
    # dev cells use only the designed sample (every event the precursor study touches + the seeded sample);
    # any other dev event that happens to have candles is left out so no ladder enters half-fetched
    pe = set(pairs["p_event"].to_list()) | set(pairs["f_event"].to_list())
    keep = pe | set(dev_sample_events(ev, pe))
    # ... and only events whose every alive leg has been fetched (the fetch order is the listing order of the
    # series, unrelated to prices or outcomes, so an unfinished fetch drops events without selecting on them)
    al = alive_legs(mk, ev).with_columns(pl.col("ticker").is_in(list(_have())).alias("got"))
    complete_fetch = set(al.group_by("event_ticker").agg(pl.col("got").all()).filter(pl.col("got"))["event_ticker"])
    keep &= complete_fetch
    panel = decision_panel(mk, ev, candles).filter((pl.col("split") != "dev") | pl.col("event_ticker").is_in(list(keep)))
    _save(panel, OUT / "panel.parquet")
    orders = cell_orders(panel, candles)
    e = precursor_orders(pairs, candles, mk)
    drift = e.pop("_drift")
    orders.update(e)
    _save(drift, OUT / "precursor_drift.parquet")
    _save(pairs, OUT / "precursor_pairs.parquet")
    log(f"panel: {panel.height} leg-decisions, {panel['event_ticker'].n_unique()} events; pairs {pairs.height} "
        f"({pairs.filter(pl.col('p_y') == 1)['f_event'].n_unique()} downstream events)")
    return orders, ev


def _sub(o: pl.DataFrame, split: str) -> pl.DataFrame:
    return o.filter(pl.col("split") == split) if o.height else o


def run_split(split: str, cells: list[str] | None = None, *, only: set[str] | None = None,
              exclude: set[str] | None = None, log=print) -> dict[str, Any]:
    """Every cell's orders in one split; ``only`` / ``exclude`` restrict A-D orders to (or away from) a set of
    events. A dev run records which events it used in ``dev_events.json``."""
    orders, _ev = build_all(log=log)
    if split == "dev" and only is None and exclude is None:
        used = {e for k, o in orders.items() if o.height and not k.startswith("E_")
                for e in _sub(o, "dev")["event_ticker"].to_list()}
        (OUT / "dev_events.json").write_text(json.dumps(sorted(used)))
    names = cells or sorted(orders)
    res: dict[str, Any] = {}
    for k in names:
        o = _sub(orders[k], split)
        if only is not None and o.height:
            o = o.filter(pl.col("event_ticker").is_in(list(only)))
        if exclude is not None and o.height:
            o = o.filter(~pl.col("event_ticker").is_in(list(exclude)))
        if k.startswith("E_") and split == "dev":
            o = o.filter(pl.col("p_close") < DEV_END)
        r = summarize(o)
        if o.height and r.get("fills"):
            f = o.filter(pl.col("t_in").is_not_null())
            r["by_group"] = {g[0]: summarize(x) for g, x in f.group_by("group")}
            r["by_leaky"] = {str(g[0]): summarize(x) for g, x in f.group_by("leaky")} if "leaky" in f.columns else None
            if k.startswith("E_"):
                r["by_season"] = {str(g[0]): summarize(x) for g, x in f.group_by("season")}
            r["book"] = book_metrics(o, k, split if only is None and exclude is None else f"{split}_extra")
        res[k] = r
        log(f"{split} {k}: fills {r.get('fills')} events {r.get('events')} c {r.get('c_per_contract')} "
            f"t {r.get('t_event')} P&L/$ {r.get('pnl_per_usd')}", flush=True)
    return res


def dev(log=print) -> dict[str, Any]:
    res = run_split("dev", log=log)
    # Holm on the most conservative of: t clustered by event, t clustered by ceremony, and the break-even
    # (loss count) p. A cell with < MIN_CLUSTERS ceremonies has no p and cannot pass.
    def pval(v: dict[str, Any]) -> float | None:
        if v.get("t_event") is None or v.get("t_ceremony") is None:
            return None
        return max(_norm_sf(min(v["t_event"], v["t_ceremony"])), v.get("p_breakeven") or 0.0)
    p = {k: pval(v) for k, v in res.items()}
    passed = holm(p)
    for k in res:
        res[k]["p_one_sided"] = None if p[k] is None else round(p[k], 5)
        res[k]["holm_pass"] = passed[k]
    (OUT / "results_dev.json").write_text(json.dumps(res, indent=1, default=str))
    return res


MIN_DEV_FILLS, MIN_DEV_EVENTS = 30, 15


def freeze(log=print) -> dict[str, Any]:
    """Frozen rules: the pre-named primary, plus every dev cell that passes Holm (24 cells, one-sided
    alpha 0.05) with >= 30 fills over >= 15 events. Written before the holdout is opened."""
    res = json.loads((OUT / "results_dev.json").read_text())
    picks = [PRIMARY] + [k for k, v in res.items() if k != PRIMARY and v.get("holm_pass")
                         and (v.get("fills") or 0) >= MIN_DEV_FILLS and (v.get("events") or 0) >= MIN_DEV_EVENTS
                         and not k.startswith("E_")]
    frozen = {"frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"), "cells": picks,
              "rule": "dev Holm over 24 one-sided cells; E cells have no holdout events (film season Jan-Mar)",
              "dev": {k: {x: res[k].get(x) for x in ("fills", "events", "c_per_contract", "t_event", "pnl_per_usd")}
                      for k in picks}}
    (OUT / "frozen.json").write_text(json.dumps(frozen, indent=1))
    log(f"frozen: {picks}")
    return frozen


def holdout(log=print) -> dict[str, Any]:
    frozen = json.loads((OUT / "frozen.json").read_text())
    res = run_split("holdout", log=log)
    out = {"frozen": frozen, "confirmatory": {k: res[k] for k in frozen["cells"]},
           "secondary_all_cells": res}
    (OUT / "results_holdout.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def extra(log=print) -> dict[str, Any]:
    """Frozen cells on dev-period award events that were NOT in the dev run (the non-film sample fetched after
    the freeze): unseen data from before October, reported next to the holdout."""
    frozen = json.loads((OUT / "frozen.json").read_text())
    seen = set(json.loads((OUT / "dev_events.json").read_text()))
    res = run_split("dev", [c for c in frozen["cells"] if not c.startswith("E_")], exclude=seen, log=log)
    out = {"frozen": frozen, "unseen_dev_period": res}
    (OUT / "results_extra.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def tables(log=print) -> str:
    """Markdown tables of every cell (dev, and holdout when scored) for the report."""
    dev_ = json.loads((OUT / "results_dev.json").read_text())
    ho = json.loads((OUT / "results_holdout.json").read_text())["secondary_all_cells"] \
        if (OUT / "results_holdout.json").exists() else {}
    hdr = ("| cell | dev fills | dev events | dev c/ct | dev t | dev P&L/$ | Holm | ho fills | ho events | ho c/ct "
           "| ho t | ho P&L/$ |")
    lines = [hdr, "|" + "---|" * 12]
    for k in sorted(dev_):
        d, h = dev_[k], ho.get(k, {})
        lines.append(f"| {k} | {d.get('fills')} | {d.get('events')} | {d.get('c_per_contract')} | {d.get('t_event')} | "
                     f"{d.get('pnl_per_usd')} | {'pass' if d.get('holm_pass') else ''} | {h.get('fills', '')} | "
                     f"{h.get('events', '')} | {h.get('c_per_contract', '')} | {h.get('t_event', '')} | "
                     f"{h.get('pnl_per_usd', '')} |")
    books = ["| cell | split | positions | events | $/wk | ann vol % | Sharpe | CI90 | ex top-5 | max DD | % wk + |",
             "|" + "---|" * 11]
    for split, res in (("dev", dev_), ("holdout", ho)):
        for k in sorted(res):
            b = res[k].get("book") or {}
            if b.get("positions", 0) >= 2:
                books.append(f"| {k} | {split} | {b.get('positions')} | {b.get('events')} | {b.get('usd_per_week')} | "
                             f"{b.get('ann_vol_pct')} | {b.get('sharpe')} | {b.get('sharpe_ci90')} | "
                             f"{b.get('sharpe_ex_top5')} | {b.get('max_dd_usd')} | {b.get('weeks_positive_pct')} |")
    txt = "\n".join(lines) + "\n\n" + "\n".join(books) + "\n"
    (OUT / "tables.md").write_text(txt)
    log(txt)
    return txt


def drift_table() -> pl.DataFrame:
    """Downstream (F) mid of the precursor winner / losing favourite at public - 1 h, + 1 h, + 6 h, + 24 h:
    mean change from - 1 h (c), with t clustered by F event, and the mean outcome - mid (c)."""
    from albert.research.counts import cluster_se

    d = pl.read_parquet(OUT / "precursor_drift.parquet").filter(pl.col("p_close") < DEV_END)
    d = d.with_columns(pl.when(pl.col("p_y") == 1).then(pl.lit("winner"))
                       .when((pl.col("p_rank") == 1) & (pl.col("p_mid") >= 0.5)).then(pl.lit("losing_favourite"))
                       .otherwise(None).alias("role")).filter(pl.col("role").is_not_null())
    base = d.filter(pl.col("off") == -1).select("p_event", "f_ticker", "role", pl.col("mid").alias("m0"))
    rows = []
    for off in (1, 6, 24):
        x = (d.filter(pl.col("off") == off).join(base, on=["p_event", "f_ticker", "role"])
             .filter(pl.col("mid").is_not_null() & pl.col("m0").is_not_null()))
        for (role,), g in x.group_by("role", maintain_order=True):
            ch = (100 * (g["mid"] - g["m0"])).to_numpy()
            out = (100 * (g["y"] - g["mid"])).to_numpy()
            fe = g["f_event"].to_numpy().astype(str)
            se = cluster_se(ch, fe) if len(np.unique(fe)) >= MIN_CLUSTERS else float("nan")
            so = cluster_se(out, fe) if len(np.unique(fe)) >= MIN_CLUSTERS else float("nan")
            rows.append({"role": role, "hours_after": off, "n": g.height, "f_events": len(np.unique(fe)),
                         "mid_before_c": round(100 * float(g["m0"].mean()), 1),
                         "change_c": round(float(ch.mean()), 2),
                         "t_change": round(float(ch.mean()) / se, 2) if se > 0 and math.isfinite(se) else None,
                         "y_minus_mid_c": round(float(out.mean()), 2),
                         "t_y_minus_mid": round(float(out.mean()) / so, 2) if so > 0 and math.isfinite(so) else None})
    return pl.DataFrame(rows)


# ---- forward calendar (open award markets now) ---------------------------------------------------
def forward(series: list[str] | None = None, *, log=print) -> pl.DataFrame:
    """Open markets of every award series right now (live ``/markets?status=open``, recorded with a
    timestamp): one row per event with earliest/latest leg close, legs, volume, open interest and the
    number of legs whose listing quote is a <= 5c / <= 15c longshot."""
    from albert.research.counts import market_row
    from albert.venues.kalshi import _field, num

    ser = award_series()
    if series is not None:
        ser = ser.filter(pl.col("series_ticker").is_in(series))
    api = _api()
    rows: list[dict[str, Any]] = []
    for s in ser["series_ticker"].to_list():
        for page in _pages(api, "/markets", {"series_ticker": s, "status": "open", "limit": 1000}, cache=False):
            for m in page.payload.get("markets") or []:
                r = market_row(m, s, "open")
                r |= {"yes_bid": num(_field(m, "yes_bid")), "yes_ask": num(_field(m, "yes_ask")),
                      "open_interest": num(_field(m, "open_interest"))}
                rows.append(r)
    if not rows:
        raise RuntimeError("forward: no open award markets returned (a zero-row smoke run is a failure)")
    mk = pl.DataFrame(rows, infer_schema_length=None).join(ser.select("series_ticker", "family", "group", "fee_type"),
                                                          on="series_ticker", how="left")
    for c in ("yes_bid", "yes_ask"):        # legacy integer-cent fields
        mk = mk.with_columns(pl.when(pl.col(c) > 1).then(pl.col(c) / 100).otherwise(pl.col(c)).alias(c))
    mid = (pl.col("yes_bid") + pl.col("yes_ask")) / 2
    two = (pl.col("yes_bid") > 0) & (pl.col("yes_ask") < 1) & (pl.col("yes_ask") > pl.col("yes_bid"))
    ev = mk.group_by("event_ticker").agg(
        pl.col("series_ticker").first(), pl.col("family").first(), pl.col("group").first(),
        pl.col("title").first(), pl.col("close_time").min().alias("first_close"),
        pl.col("close_time").max().alias("last_close"), pl.len().alias("legs"),
        pl.col("volume").sum().alias("volume"), pl.col("open_interest").sum().alias("oi"),
        (two & (mid <= 0.05)).sum().alias("legs_le5c"), (two & (mid <= 0.15)).sum().alias("legs_le15c"),
        two.sum().alias("legs_two_sided"), pl.col("fee_type").first()).sort("last_close")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    _save(mk, OUT / f"open_markets_{stamp}.parquet")
    if series is None:
        _save(ev, OUT / "open_events.parquet")
    log(f"forward: {mk.height} open legs in {ev.height} events of {mk['series_ticker'].n_unique()} series")
    return ev


# ---- forward recorder ---------------------------------------------------------------------------
# Ceremonies in the forward window. Times are the broadcast start in UTC where an allowed official page gives
# the date (goldenglobes.com, grammy.com, actorawards.org, heisman.com, fetched 2026-11-23); None = not
# confirmed from an allowed source (record from the Kalshi listing until the date is known).
CALENDAR: tuple[dict[str, Any], ...] = (
    {"ceremony": "Heisman Trophy", "series": ("KXHEISMAN",), "start": datetime(2026, 12, 13, 1, 0, tzinfo=UTC),
     "source": "heisman.com: finalists and voting deadline Dec 7, ceremony Dec 12 2026"},
    {"ceremony": "The Game Awards 2026", "series": ("KXGAMEAWARDS", "KXGAMEAWARDSBGD"), "start": None,
     "source": "thegameawards.com robots allows but the page answered 403; usually the 2nd Thursday of December"},
    {"ceremony": "Golden Globes 2027", "series": ("KXGGWINNER",), "start": datetime(2027, 1, 11, 1, 0, tzinfo=UTC),
     "source": "goldenglobes.com: live Sunday Jan 10 2027 on CBS"},
    {"ceremony": "Critics Choice 2027 (32nd)", "series": ("KXCRITICS",), "start": None,
     "source": "criticschoice.com shows no date yet (January 2027)"},
    {"ceremony": "Grammys 2027 (69th)", "series": ("KXGRAMMY", "KXGRAMAOTY", "KXGRAMBNA", "KXGRAMROTY", "KXGRAMSOTY"),
     "start": datetime(2027, 2, 8, 1, 0, tzinfo=UTC), "source": "grammy.com: Grammy night February 7 2027"},
    {"ceremony": "Actor Awards 2027", "series": ("KXSAGAWARD",), "start": datetime(2027, 3, 1, 1, 0, tzinfo=UTC),
     "source": "actorawards.org: live on Netflix Sunday Feb 28 2027 8 PM ET"},
    {"ceremony": "Oscars 2027", "series": ("KXOSCAR",), "start": None, "source": "oscars.org page shows no date"},
)
RECORD_DIR = OUT / "record"
# "major televised ceremony" for the post-hoc forward rule (fixed 2026-11-23, before the 2027 season)
MAJOR_PREFIXES = ("KXOSCAR", "KXGG", "KXCRITICS", "KXSAGAWARD", "KXBAFTA", "KXEMMY", "KXGRAM", "KXGRAMMY",
                  "KXLATINGRAMMY", "KXLGRAM", "KXCMA")


def due_events(open_ev: pl.DataFrame, now: datetime, horizon: timedelta) -> pl.DataFrame:
    """Open events of a calendar ceremony that starts within ``horizon`` of ``now`` (dated ceremonies only)."""
    rows = []
    for c in CALENDAR:
        if c["start"] is None or not (now - timedelta(hours=6) <= c["start"] <= now + horizon):
            continue
        hit = open_ev.filter(pl.any_horizontal([pl.col("series_ticker").str.starts_with(x) for x in c["series"]]))
        rows.append(hit.with_columns(pl.lit(c["ceremony"]).alias("ceremony"), pl.lit(c["start"]).alias("start")))
    return pl.concat(rows) if rows else open_ev.head(0)


def record(days: float = 4.0, *, log=print) -> int:
    """One recording pass for the forward test: for every open leg of a dated ceremony starting within
    ``days``, the live REST book (best bid/ask and displayed size, raw response stored append-only) is
    appended to ``record/books.jsonl``. Run hourly from T - 96 h, every 10 min in the last 6 h."""
    from albert.venues.kalshi import orderbook_top

    now = datetime.now(UTC)
    want = [c for c in CALENDAR if c["start"] is not None and now - timedelta(hours=6) <= c["start"] <= now + timedelta(days=days)]
    ser = award_series()["series_ticker"].to_list()
    series = [x for x in ser if any(x.startswith(p_) for c in want for p_ in c["series"])]
    if not series:
        raise RuntimeError(f"record: no dated ceremony within {days} days (a zero-row run is a failure)")
    ev = forward(series, log=log)
    due = due_events(ev, now, timedelta(days=days))
    mk = pl.read_parquet(max(OUT.glob("open_markets_*.parquet")))
    legs = mk.filter(pl.col("event_ticker").is_in(due["event_ticker"].implode()))
    api = _api()
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with (RECORD_DIR / "books.jsonl").open("a") as fh:
        for r in legs.iter_rows(named=True):
            page = api.orderbook(r["ticker"])
            fh.write(json.dumps({"ts": page.fetched_at.isoformat(), "event_ticker": r["event_ticker"],
                                 "ticker": r["ticker"], "raw": str(page.raw_path), **orderbook_top(page.payload)}) + "\n")
            n += 1
    log(f"record: {due.height} due events, {n} books")
    if n == 0:
        raise RuntimeError("record: zero books recorded (no dated ceremony inside the horizon?)")
    return n


def forward_plan(log=print) -> dict[str, Any]:
    """Dated forward-test plan (written to ``forward_plan.json``): the calendar with each ceremony's open
    Kalshi events and cheap two-sided legs at the 2026-11-23 snapshot, the decision/record times, and what to
    record. Rules are the frozen ones from ``frozen.json``."""
    ev = pl.read_parquet(OUT / "open_events.parquet")
    frozen = json.loads((OUT / "frozen.json").read_text()) if (OUT / "frozen.json").exists() else {}
    cal = []
    for c in CALENDAR:
        hit = ev.filter(pl.any_horizontal([pl.col("series_ticker").str.starts_with(x) for x in c["series"]]))
        st = c["start"]
        cal.append({"ceremony": c["ceremony"], "start_utc": st.isoformat() if st else None, "source": c["source"],
                    "open_events": hit.height, "open_legs": int(hit["legs"].sum() or 0),
                    "legs_two_sided_le5c_now": int(hit["legs_le5c"].sum() or 0),
                    "volume_usd": round(float(hit["volume"].sum() or 0), 0),
                    "decide_maker_no_utc": (st - timedelta(hours=72)).isoformat() if st else "ceremony - 72 h",
                    "record_books_from_utc": (st - timedelta(hours=96)).isoformat() if st else "ceremony - 96 h"})
    plan = {"written_utc": datetime.now(UTC).isoformat(timespec="seconds"), "frozen_rules": frozen.get("cells"),
            "holdout_verdict": "frozen rules lost on the 2026-10-01..11-22 holdout (no edge); see results_holdout.json",
            "forward_rule_posthoc": {
                "name": "major-ceremony maker NO (post-hoc restriction, named 2026-11-23 before any of these events)",
                "series": list(MAJOR_PREFIXES),
                "rule": "at ceremony start - 72 h, legs with a real two-sided mid <= 5c: one NO bid selling YES at "
                        "ceil(mid + 12c), post-only, <= 3 legs per event (cheapest NO first), $5 per event, "
                        "$20 resting collateral cap, held to settlement; fill only on a later YES print strictly "
                        "through the price; cancel the event's other bids 60 s after its first fill",
                "why_posthoc": "holdout losses came from Daytime Emmys (5 of 7 fills lost) and Streamer Awards "
                               "(2 of 5); the 14 holdout fills in CMA and Nobel events won (+16c); dev was all "
                               "major film/TV ceremonies"},
            "calendar": cal,
            "unscheduled_not_tradable_by_rule": ["KXTIME-26 (Time POY)", "KXWORDOFTHEYEAR-26DEC",
                                                 "KXMISSUNIVERSE-26 (date not confirmed)"],
            "record": [("books of every open leg of a dated ceremony: hourly from T - 96 h, every 10 min in the "
                       "last 6 h (`python -m albert.research.awards record --days 4`)"),
                       ("trade prints of every leg with a resting paper order, from placement to close "
                       "(/markets/trades), to apply the strictly-through fill rule"),
                       ("for precursor ceremonies (Golden Globes Jan 10, Critics Choice, Actor Awards Feb 28, "
                       "BAFTA): books of the downstream Oscar/BAFTA/Actor Awards legs at the precursor close "
                       "- 1 h, + 1 h, + 6 h, + 24 h")],
            "pass_bar": "event-clustered t >= 2 and ceremony-clustered t >= 2 on fills placed after 2026-11-23, "
                        ">= 60 fills over >= 6 ceremonies, losses below the break-even count (p < 0.05), "
                        "Sharpe ex top-5 > 1 from vol_metrics"}
    (OUT / "forward_plan.json").write_text(json.dumps(plan, indent=1, default=str))
    log(json.dumps(plan, indent=1, default=str))
    return plan


# ---- robots (data-source policy) ----------------------------------------------------------------
ROBOTS_CHECKS: dict[str, list[str]] = {   # host -> paths a precursor / consensus / settlement reader would use
    "api.elections.kalshi.com": ["/trade-api/v2/markets", "/trade-api/v2/series"],
    "gamma-api.polymarket.com": ["/events", "/markets"],
    "clob.polymarket.com": ["/prices-history"],
    "www.goldderby.com": ["/odds/", "/film/", "/tv/", "/music/"],
    "www.nextbestpicture.com": ["/predictions/"],
    "www.awardswatch.com": ["/forums/"],
    "www.awardsdaily.com": ["/"],
    "www.oscars.org": ["/oscars/ceremonies/"],
    "awardsdatabase.oscars.org": ["/"],
    "goldenglobes.com": ["/winners-nominees/"],
    "www.goldenglobes.com": ["/winners-nominees/"],
    "www.criticschoice.com": ["/"],
    "www.sagawards.org": ["/awards/nominees-and-recipients/"],
    "www.actorawards.org": ["/"],
    "www.bafta.org": ["/film/awards/"],
    "www.dga.org": ["/Awards/"],
    "producersguild.org": ["/pga-awards/"],
    "www.wga.org": ["/awards/"],
    "www.grammy.com": ["/awards/"],
    "www.latingrammy.com": ["/"],
    "www.cmaawards.com": ["/"],
    "thegameawards.com": ["/nominees", "/winners"],
    "time.com": ["/person-of-the-year/"],
    "people.com": ["/sexiest-man-alive/"],
    "www.merriam-webster.com": ["/wordplay/word-of-the-year"],
    "www.nobelprize.org": ["/prizes/", "/all-nobel-prizes-2025/"],
    "www.thebookerprizes.com": ["/"],
    "www.nationalbook.org": ["/"],
    "eurovision.tv": ["/event/"],
    "www.francefootball.fr": ["/"],
    "www.heisman.com": ["/"],
    "bbwaa.com": ["/"],
    "www.mlb.com": ["/awards/"],
    "www.nfl.com": ["/honors/"],
    "www.nba.com": ["/news/"],
    "en.wikipedia.org": ["/wiki/Academy_Awards", "/w/api.php"],
    "www.metacritic.com": ["/game/"],
    "store.steampowered.com": ["/steamawards"],
}


def check_robots(out: Path = OUT, *, pause: float = 1.0) -> list[dict[str, Any]]:
    """Each host's own robots.txt, fetched now (no cache), evaluated for ``counts.AGENTS``. A 403 or an
    error counts as blocked; 404/410 (no robots.txt) as allowed. Bodies saved under ``robots/``."""
    import httpx

    from albert.research.counts import AGENTS, robots_verdict

    rows = []
    d = out / "robots"
    d.mkdir(parents=True, exist_ok=True)
    for host, paths in ROBOTS_CHECKS.items():
        try:
            r = httpx.get(f"https://{host}/robots.txt", timeout=20, follow_redirects=True, headers={"User-Agent": UA})
            status, text = r.status_code, (r.text if r.status_code == 200 else "")
        except httpx.HTTPError as error:
            status, text = None, ""
            print(f"robots {host}: {error}", flush=True)
        (d / f"{host}.txt").write_text(text)
        for path in paths:
            v = robots_verdict(text, path) if status in (200, 404, 410) else dict.fromkeys(AGENTS, False)
            rows.append({"host": host, "path": path, "http_status": status,
                         "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                         **{f"allow_{a}": v[a] for a in AGENTS}, "allowed": all(v.values())})
        print(f"robots {host}: {status} {'allowed' if all(r['allowed'] for r in rows if r['host'] == host) else 'BLOCKED'}",
              flush=True)
        time.sleep(pause)
    (out / "robots.json").write_text(json.dumps(rows, indent=1))
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.awards")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("robots", "universe", "panel", "dev", "freeze", "holdout", "extra", "forward", "report"):
        sub.add_parser(c)
    sub.add_parser("record").add_argument("--days", type=float, default=4.0)
    sub.add_parser("candles").add_argument("layer", choices=("holdout", "precursor", "devsample", "all"))
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "robots":
        check_robots()
    elif a.cmd == "universe":
        universe()
    elif a.cmd == "candles":
        for layer in (("holdout", "precursor", "devsample") if a.layer == "all" else (a.layer,)):
            fetch_candles(layer)
    elif a.cmd == "dev":
        dev()
    elif a.cmd == "freeze":
        freeze()
    elif a.cmd == "holdout":
        holdout()
    elif a.cmd == "extra":
        extra()
    elif a.cmd == "forward":
        forward()
    elif a.cmd == "report":
        tables()
        print(drift_table())
        forward_plan()
    elif a.cmd == "record":
        record(a.days)


if __name__ == "__main__":
    main()
