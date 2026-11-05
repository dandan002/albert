"""Mention markets: do base rates or transcript word rates beat Kalshi mention prices?

Kalshi "Mentions" events are one speech or broadcast (a Trump rally, a press briefing, a TV show,
an earnings call) with many word legs: "Will <speaker> say <word>?" and some "<word> (N+ times)".
``rules_secondary`` (read on the live API, e.g. KXTRUMPMENTION-26OCT05): the exact word or phrase,
or a plural or possessive form, counts; other inflections do not; "A / B" means either word; the
speaker must say it during the named event (live broadcast, then official transcripts).

Timing (all as-of guarantees are enforced in code, see :func:`decision_frame`,
:func:`asof_counts`, :func:`transcript_predict`, :func:`start_check`):

* **Event start** = Kalshi milestone ``start_date`` (``/milestones?related_event_ticker=``),
  a public schedule. Verified per event against the market itself (:func:`start_check`): the
  share of eventually-YES legs whose hourly bid is ≥ 0.97 is ~0 at ``start − 24 h`` and
  ``start − 1 h`` and ~1 at ``start + 2 h`` (a said word goes to 0.99 at once); events where
  it jumps before ``start − 1 h`` are dropped from the 1-hour decision.
* **Decision times** ``t = start − 24 h`` and ``start − 1 h`` (window events "Will Trump say X
  before <date>": window open + 1 h / + 24 h). Legs listed after ``t`` are dropped. Quotes: the
  last hourly candle that ended at or before ``t`` (shared harness).
* **Base-rate model**: a leg at ``t`` learns only from legs whose ``settlement_ts < t``.
* **Transcript model**: only Factba.se transcripts dated ≤ event day − 1 (1-hour decision) or
  − 2 (24-hour decision), US Eastern.

Entry point: :func:`run` (``holdout=False`` for the inner dev split used to pick variants),
then :func:`write_report`. Data caches: ``data/cache/mentions/`` (milestones, Factba.se index,
pages, segments, word counts) and ``data/cache/fundamental_mentions_*.parquet`` (candles,
predictions).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

NAME = "mentions"
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

_THRESH = re.compile(r"\((\d+)\+\s*times?\)", re.I)


# ---- leg parsing ----------------------------------------------------------------------------
def parse_word(sub_title: str | None) -> tuple[str, tuple[str, ...], int]:
    """``"Iran (5+ times)"`` -> ("iran", ("iran",), 5); ``"Farm / Farmer"`` -> alternatives."""
    s = (sub_title or "").strip()
    k = 1
    m = _THRESH.search(s)
    if m:
        k = int(m.group(1))
        s = _THRESH.sub("", s).strip()
    alts = tuple(sorted({a.strip().lower() for a in s.split("/") if a.strip()}))
    return " / ".join(alts), alts, k


SPEAKERS = {  # series -> speaker group; anything else falls back to the series itself
    "KXTRUMPMENTION": "trump", "KXTRUMPMENTIONB": "trump", "KXTRUMPSAY": "trump",
    "KXTRUMPSAYCOMPANY": "trump", "KXTRUMPSAYCOUNTRY": "trump", "KXTRUMPSAYMONTH": "trump_month",
    "KXTRUMPSAYNICKNAME": "trump",
    "KXVANCEMENTION": "vance", "KXSECPRESSMENTION": "press_sec", "KXLEAVITTMENTION": "press_sec",
}


def speaker_of(series: str) -> str:
    if series in SPEAKERS:
        return SPEAKERS[series]
    if series.startswith("KXEARNINGSMENTION") or series.startswith("KXMENTIONEARN"):
        return "earnings"
    return series.removeprefix("KX").removesuffix("MENTION").lower()


_TYPES = (  # first match wins; applied to the lower-cased event title
    ("earnings", ("earnings",)),
    ("rally", ("rally", "rallies")),
    ("press_conf", ("press conference", "press briefing", "briefing", "presser", "press gaggle", "gaggle")),
    ("interview", ("interview", "60 minutes", "podcast", "town hall", "townhall")),
    ("address", ("address", "speech", "remarks", "un general", "united nations", "joint session")),
    ("meeting", ("meeting", "bilateral", "summit", "dinner", "roundtable", "cabinet")),
    ("debate", ("debate",)),
    ("show", ("show", "tonight", "news", "meet the press", "face the nation", "sunday", "hannity",
              "ingraham", "watters", "special report", "last word", "all in", "weeknight", "beat")),
)


def event_type(title: str | None, speaker: str) -> str:
    if speaker == "earnings":
        return "earnings"
    t = (title or "").lower()
    for name, keys in _TYPES:
        if any(k in t for k in keys):
            return name
    return "other"


def legs(markets: pl.DataFrame) -> pl.DataFrame:
    """Word legs of every settled Mentions market (``Event does not qualify`` legs dropped)."""
    m = markets.filter(pl.col("result").is_in(["yes", "no"]))
    rows = []
    for r in m.iter_rows(named=True):
        word, alts, k = parse_word(r["yes_sub_title"])
        if not word or word.startswith("event does not qualify") or word == "event does not qualify":
            continue
        sp = speaker_of(r["series_ticker"])
        rows.append({"ticker": r["ticker"], "event": r["event_ticker"], "series": r["series_ticker"],
                     "speaker": sp, "etype": event_type(r["title"], sp), "word": word,
                     "n_alts": len(alts), "n_tokens": max(len(a.split()) for a in alts), "k": k,
                     "title": r["title"], "open_time": r["open_time"], "close_time": r["close_time"],
                     "settlement_ts": r["settlement_ts"], "volume": r["volume"],
                     "y": int(r["result"] == "yes")})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- event start times (Kalshi milestones) ---------------------------------------------------
def _client(s, venue: str = "kalshi", base: str = KALSHI, min_interval: float = 0.5, **kw):
    from albert.http import CachedClient

    return CachedClient(venue, base, s.data_dir / "cache" / NAME / "raw", min_interval=min_interval, **kw)


def milestone_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (event ticker, milestone) from a ``/milestones`` page."""
    from albert.venues.kalshi import parse_ts

    out = []
    for ms in payload.get("milestones") or []:
        for ev in set((ms.get("primary_event_tickers") or []) + (ms.get("related_event_tickers") or [])):
            out.append({"event": ev, "start": parse_ts(ms.get("start_date")), "end": parse_ts(ms.get("end_date")),
                        "ms_title": ms.get("title"), "ms_type": ms.get("type"),
                        "ms_updated": parse_ts(ms.get("last_updated_ts"))})
    return out


def fetch_starts(s, events: list[str]) -> pl.DataFrame:
    """Milestone start per event (one cached request per event; 2 req/s)."""
    c = _client(s)
    rows = []
    for i, ev in enumerate(events, 1):
        try:
            page = c.get("/milestones", {"related_event_ticker": ev, "limit": 10})
        except RuntimeError as error:
            print(f"milestone {ev}: {str(error)[:100]}")
            continue
        rows.extend(r for r in milestone_rows(page.payload) if r["event"] == ev)
        if i % 100 == 0:
            print(f"  milestones {i}/{len(events)}", flush=True)
    df = pl.DataFrame(rows, infer_schema_length=None)
    # Several milestones per event are rare; keep the earliest start (conservative for t = start - h).
    return df.sort("start").unique("event", keep="first") if df.height else df


# ---- transcripts: Roll Call Factba.se (public JSON search + public transcript pages) ---------
FACTBASE = "https://rollcall.com"
_SEG_HEAD = re.compile(r'<h2 class="text-md inline">(.*?)</h2>\s*(?:<span[^>]*>(\d\d:\d\d:\d\d)-(\d\d:\d\d:\d\d)[^<]*</span>)?', re.S)
_SEG_TEXT = re.compile(r'<div class=" flex-auto text-md text-gray-600 leading-loose">(.*?)</div>', re.S)


def factbase_index(s, since: str, max_pages: int = 200) -> pl.DataFrame:
    """Factbase records newest-first until ``date < since`` (``YYYY-MM-DD``). Cached per page:
    page contents shift as new records arrive, so the cache key carries today's date."""
    c = _client(s, "factbase", FACTBASE, min_interval=1.0)
    tag = datetime.now(UTC).strftime("%Y-%m-%d")
    rows = []
    for page in range(1, max_pages + 1):
        d = c.get("/wp-json/factbase/v1/search", {"q": "", "media": "", "type": "", "sort": "date",
                                                   "location": "all", "place": "all", "page": page,
                                                   "format": "json"}, tag=tag).payload
        data = d.get("data") or []
        rows.extend(index_rows(data))
        if not data or min(r["date"] for r in data) < since:
            break
    df = pl.DataFrame(rows, infer_schema_length=None).unique("document_id", keep="first")
    return df.filter(pl.col("date") >= since)


def index_rows(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in data:
        doc = r.get("document") or {}
        dur = doc.get("duration")
        secs = None
        if dur and re.fullmatch(r"\d+:\d\d:\d\d", dur):
            h, mi, se = (int(x) for x in dur.split(":"))
            secs = h * 3600 + mi * 60 + se
        out.append({"document_id": r.get("document_id"), "date": r.get("date"), "type": r.get("type"),
                    "media": r.get("media_type"), "title": r.get("record_title"),
                    "url": r.get("factbase_url"), "duration_s": secs})
    return out


def parse_transcript(page_html: str) -> list[dict[str, Any]]:
    """Speaker segments of a Factbase transcript page: speaker, start/end seconds, text."""
    import html as _html

    out = []
    heads = list(_SEG_HEAD.finditer(page_html))
    for i, h in enumerate(heads):
        stop = heads[i + 1].start() if i + 1 < len(heads) else len(page_html)
        m = _SEG_TEXT.search(page_html, h.end(), stop)
        if not m:
            continue
        text = _html.unescape(re.sub(r"<[^>]+>", " ", m.group(1)))
        text = re.sub(r"\s+", " ", text).strip()

        def secs(x: str | None) -> int | None:
            if not x:
                return None
            a, b, c = (int(v) for v in x.split(":"))
            return a * 3600 + b * 60 + c
        out.append({"speaker": _html.unescape(h.group(1)).strip(), "t0": secs(h.group(2)),
                    "t1": secs(h.group(3)), "text": text})
    return out


def fetch_transcripts(s, index: pl.DataFrame) -> pl.DataFrame:
    """One row per (document, speaker segment); pages are cached zstd-compressed."""
    c = _client(s, "factbase", FACTBASE, min_interval=1.0)
    rows = []
    for i, r in enumerate(index.iter_rows(named=True), 1):
        path = r["url"].removeprefix(FACTBASE)
        try:
            page = c.get(path)
        except RuntimeError as error:
            print(f"transcript {r['document_id']}: {str(error)[:100]}")
            continue
        text = page.payload if isinstance(page.payload, str) else ""
        for j, seg in enumerate(parse_transcript(text)):
            rows.append({"document_id": r["document_id"], "seg": j, **seg})
        if i % 50 == 0:
            print(f"  transcripts {i}/{index.height}", flush=True)
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- candles ---------------------------------------------------------------------------------
WINDOW_SERIES = ("KXTRUMPSAY", "KXTRUMPSAYCOMPANY", "KXTRUMPSAYCOUNTRY", "KXTRUMPSAYMONTH", "KXTRUMPSAYNICKNAME")


def fetch_candles_batch(s, windows: pl.DataFrame, *, chunk: int = 60) -> pl.DataFrame:
    """Hourly candles for live (post-cutoff) markets, one ``/markets/candlesticks`` request per
    event window (``event, tickers, lo, hi``). Cached under data/cache/mentions/raw and
    data/cache/fundamental_mentions_candles_batch.parquet."""
    from albert.venues import kalshi as k

    path = s.data_dir / "cache" / f"fundamental_{NAME}_candles_batch.parquet"
    have = pl.read_parquet(path) if path.exists() else None
    done = set(have["ticker"].unique()) if have is not None and have.height else set()
    api = k.Kalshi(_client(s, min_interval=0.5))
    rows: list[dict[str, Any]] = []
    todo = [w for w in windows.iter_rows(named=True) if not set(w["tickers"]) <= done]
    for i, w in enumerate(todo, 1):
        ticks = sorted(w["tickers"])
        for j in range(0, len(ticks), chunk):
            try:
                page = api.candles_batch(ticks[j:j + chunk], w["lo"], w["hi"], 60)
            except RuntimeError as error:
                print(f"skip batch {w['event']}: {str(error)[-200:]}")
                continue
            for m in page.payload.get("markets") or []:
                rows.extend(k.candle_rows({"candlesticks": m.get("candlesticks") or []}, m["market_ticker"], 60))
        if i % 25 == 0:
            print(f"  batch {i}/{len(todo)}", flush=True)
    new = pl.DataFrame(rows, infer_schema_length=None) if rows else None
    frames = [f for f in (have, new) if f is not None and f.height]
    out = pl.concat(frames, how="diagonal_relaxed").unique(["ticker", "end_ts", "period_minutes"]) if frames else pl.DataFrame()
    tmp = path.with_suffix(".tmp")
    out.write_parquet(tmp)
    tmp.replace(path)
    return out


# ---- decision times --------------------------------------------------------------------------
def window_end(event: str) -> datetime | None:
    """``KXTRUMPSAY-26OCT19`` -> 2026-10-19 04:00 UTC (the window ends at 12:00am ET)."""
    m = re.search(r"-(\d\d)([A-Z]{3})(\d\d)", event)
    if not m:
        return None
    months = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
    yy, mon, dd = m.groups()
    if mon not in months:
        return None
    return datetime(2000 + int(yy), months.index(mon) + 1, int(dd), 4, tzinfo=UTC)


def decision_frame(lg: pl.DataFrame, starts: pl.DataFrame, data_end: datetime) -> pl.DataFrame:
    """One row per (leg, decision time). Speech events: ``start − 24 h`` and ``start − 1 h``
    (labels ``h24`` / ``h1``). Window events (``Will Trump say X before <date>``): ``open + 1 h``
    and ``open + 24 h`` (labels ``w1`` / ``w24``), keeping only legs still open at ``t`` (a leg
    closes early when the word is said, which is public at its close) and dropping windows that
    end after the data end (their NO legs had not settled yet, so only early YES legs exist)."""
    d = lg.join(starts.select("event", "start"), on="event", how="left")
    win = pl.col("series").is_in(WINDOW_SERIES)
    sp = d.filter(~win & pl.col("start").is_not_null())
    out = [sp.with_columns((pl.col("start") - timedelta(hours=h)).alias("t"), pl.lit(f"h{h}").alias("when"))
           for h in (24, 1)]
    w = d.filter(win).with_columns(pl.col("open_time").min().over("event").alias("w_open"),
                                   pl.col("event").map_elements(window_end, return_dtype=pl.Datetime("us", "UTC")).alias("w_end"))
    w = w.filter(pl.col("w_end").is_not_null() & (pl.col("w_end") <= data_end))
    for h in (1, 24):
        x = w.with_columns((pl.col("w_open") + timedelta(hours=h)).alias("t"), pl.lit(f"w{h}").alias("when"))
        out.append(x.filter(pl.col("close_time") > pl.col("t")).with_columns(pl.col("w_open").alias("start")))
    cols = lg.columns + ["start", "t", "when"]
    # a leg listed after t did not exist at t (Kalshi adds words during the day)
    return pl.concat([o.select(cols) for o in out], how="vertical_relaxed").filter(pl.col("open_time") <= pl.col("t"))


# ---- model 1: hierarchical beta-binomial base rates (as of t) ------------------------------
LEVELS = ("g", "s", "se", "x", "sw")


def _kband(k: int) -> str:
    return "1" if k <= 1 else "2+"


def level_keys(r: dict[str, Any]) -> dict[str, tuple]:
    kind = "win" if r["series"] in WINDOW_SERIES else "speech"
    return {"g": (kind,), "s": (kind, r["speaker"]), "se": (kind, r["speaker"], r["etype"], _kband(r["k"])),
            "x": (kind, r["word"], r["k"]), "sw": (kind, r["speaker"], r["word"], r["k"])}


def asof_counts(train: pl.DataFrame, queries: pl.DataFrame, half_lives: tuple[float, ...] = (0.0,)) -> pl.DataFrame:
    """Weighted (hits, trials) per hierarchy level for each query row, using only training legs
    whose ``settlement_ts`` is strictly before the query's ``t``. Weight ``0.5^(age/half_life)``
    (``half_life = 0``: no decay). Level ``x`` (the same word for *other* speakers) excludes the
    query's own speaker. Columns ``h_<level>_<hl>``, ``n_<level>_<hl>``."""
    tr = train.sort("settlement_ts")
    ts = tr["settlement_ts"].dt.epoch("s").to_numpy().astype(float)
    ys = tr["y"].to_numpy().astype(float)
    spk = tr["speaker"].to_numpy()
    idx: dict[str, dict[tuple, list[int]]] = {lv: defaultdict(list) for lv in LEVELS}
    for i, r in enumerate(tr.iter_rows(named=True)):
        for lv, key in level_keys(r).items():
            idx[lv][key].append(i)
    idx_np = {lv: {k: np.asarray(v) for k, v in d.items()} for lv, d in idx.items()}
    out: dict[str, list[float]] = defaultdict(list)
    for r in queries.iter_rows(named=True):
        tq = r["t"].timestamp()
        keys = level_keys(r)
        for lv in LEVELS:
            ii = idx_np[lv].get(keys[lv])
            if ii is not None:
                ii = ii[ts[ii] < tq]
                if lv == "x" and len(ii):
                    ii = ii[spk[ii] != r["speaker"]]
            for hl in half_lives:
                if ii is None or not len(ii):
                    h = n = 0.0
                else:
                    w = np.ones(len(ii)) if hl <= 0 else 0.5 ** ((tq - ts[ii]) / 86400 / hl)
                    h, n = float((w * ys[ii]).sum()), float(w.sum())
                out[f"h_{lv}_{hl:g}"].append(h)
                out[f"n_{lv}_{hl:g}"].append(n)
    return queries.with_columns(**{k: pl.Series(v, dtype=pl.Float64) for k, v in out.items()})


@dataclass(frozen=True)
class BB:
    """Shrinkage strengths (pseudo-trials) of each level toward its parent, and the decay."""
    k_s: float = 20.0
    k_se: float = 10.0
    k_x: float = 10.0
    k_w: float = 3.0
    half_life: float = 0.0


def bb_predict(df: pl.DataFrame, prm: BB) -> np.ndarray:
    hl = f"{prm.half_life:g}"

    def hn(lv: str) -> tuple[np.ndarray, np.ndarray]:
        return df[f"h_{lv}_{hl}"].to_numpy(), df[f"n_{lv}_{hl}"].to_numpy()
    h, n = hn("g")
    m = (h + 1) / (n + 2)
    for lv, kap in (("s", prm.k_s), ("se", prm.k_se), ("x", prm.k_x), ("sw", prm.k_w)):
        h, n = hn(lv)
        m = (h + kap * m) / (n + kap)
    return np.clip(m, 0.003, 0.997)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    q = np.clip(p, 0.005, 0.995)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def tune_bb(df: pl.DataFrame, half_lives: tuple[float, ...]) -> tuple[BB, list[dict[str, Any]]]:
    """Grid search of the shrinkage strengths on ``df`` (dev rows only) by log loss."""
    y = df["y"].to_numpy().astype(float)
    grid = []
    for hl in half_lives:
        for ks in (5, 20, 80, 300):
            for kse in (3, 10, 30, 100):
                for kx in (3, 10, 30, 100):
                    for kw in (0.5, 1, 2, 4, 8, 16, 32):
                        prm = BB(ks, kse, kx, kw, hl)
                        grid.append({**prm.__dict__, "logloss": log_loss(bb_predict(df, prm), y)})
    grid.sort(key=lambda g: g["logloss"])
    best = grid[0]
    return BB(best["k_s"], best["k_se"], best["k_x"], best["k_w"], best["half_life"]), grid


# ---- model 2: transcript word rates (Factbase, Trump and Vance) -----------------------------
FB_SPEAKER = {"trump": "Donald Trump", "vance": "JD Vance"}
ET = "America/New_York"


def word_pattern(alts: tuple[str, ...] | list[str]) -> re.Pattern:
    """Kalshi counting rule: the exact word or phrase, or its plural or possessive (any of the
    slash alternatives). Case-insensitive; spaces in phrases also match hyphens."""
    suffix = r"(?:'s|’s|s'|s’|es|s)?"

    def token(w: str) -> str:
        # long single words are often transcribed split ("health care", "cease-fire")
        if len(w) >= 8 and w.isalpha():
            return r"[\s\-]?".join(re.escape(ch) for ch in w)
        return re.escape(w)
    parts = []
    for a in alts:
        a = a.strip().lower()
        if not a:
            continue
        # phrases: tokens separated by spaces, hyphens or punctuation ("drill, baby, drill");
        # plurals inside a noun phrase count ("taxes on tips")
        parts.append(r"[\s\-,\.;:!…]+".join(token(w) + suffix for w in a.split()))
    return re.compile(r"(?<![\w])(?:" + "|".join(parts) + r")(?![\w])", re.I)


def fb_type(doc_type: str | None, title: str | None) -> str:
    t = (title or "").lower()
    if "tele-rally" in t or "telerally" in t:
        return "telerally"
    if "rally" in t:
        return "rally"
    if doc_type == "Interview":
        return "interview"
    if doc_type in ("Press Gaggle", "Press Briefing", "Press Conference"):
        return "press"
    return "remarks"


def kalshi_ftype(title: str | None, etype: str) -> str:
    t = (title or "").lower()
    if "tele-rally" in t or "telerally" in t:
        return "telerally"
    if etype == "rally":
        return "rally"
    if etype == "press_conf":
        return "press"
    if etype in ("interview", "show") or "fox news" in t or "podcast" in t or "town hall" in t:
        return "interview"
    return "remarks"


def speaker_docs(segments: pl.DataFrame, index: pl.DataFrame) -> pl.DataFrame:
    """One row per (document, Kalshi speaker): the speaker's own text, word count and speaking
    seconds, the document date and Factbase type."""
    rev = {v: k for k, v in FB_SPEAKER.items()}
    seg = segments.filter(pl.col("speaker").is_in(list(rev)))
    seg = seg.with_columns(pl.col("speaker").replace_strict(rev, default=None).alias("who"),
                           (pl.col("t1") - pl.col("t0") + 1).clip(0, None).alias("secs"))
    docs = (seg.sort("document_id", "seg").group_by("document_id", "who")
            .agg(pl.col("text").str.join(" "), pl.col("secs").sum()))
    docs = docs.with_columns(pl.col("text").str.count_matches(r"\S+").alias("words"))
    meta = index.select("document_id", "date", "type", "title", "duration_s")
    docs = docs.join(meta, on="document_id", how="inner")
    return docs.with_columns(pl.struct("type", "title").map_elements(lambda r: fb_type(r["type"], r["title"]),
                                                                       return_dtype=pl.String).alias("ftype"),
                             pl.col("date").str.to_date())


def doc_counts(docs: pl.DataFrame, words: dict[str, tuple[str, ...]]) -> pl.DataFrame:
    """Occurrences of every word key in every speaker document: ``document_id, who, word, c``."""
    pats = {w: word_pattern(a) for w, a in words.items()}
    rows = []
    for d in docs.iter_rows(named=True):
        for w, pat in pats.items():
            rows.append({"document_id": d["document_id"], "who": d["who"], "word": w,
                         "c": len(pat.findall(d["text"]))})
    return pl.DataFrame(rows, schema={"document_id": pl.String, "who": pl.String, "word": pl.String, "c": pl.Int64})


def nb_tail(mu: np.ndarray, r: float, k: np.ndarray) -> np.ndarray:
    """P(N >= k) for N ~ negative binomial with mean ``mu`` and size ``r`` (Poisson as r→∞)."""
    from scipy import stats

    mu = np.maximum(mu, 1e-9)
    if not math.isfinite(r):
        return stats.poisson.sf(k - 1, mu)
    return stats.nbinom.sf(k - 1, r, r / (r + mu))


@dataclass(frozen=True)
class TX:
    lookback_days: int = 45
    other_type_weight: float = 0.3   # weight of other-type documents in the rate
    prior_words: float = 20000.0     # pseudo-exposure (words) of the speaker-wide prior rate
    r: float = 1.0                   # NB size (between-speech overdispersion)
    min_lag_days: int = 1            # transcripts dated at least this many days before the event


def transcript_predict(q: pl.DataFrame, docs: pl.DataFrame, counts: pl.DataFrame, prm: TX) -> pl.DataFrame:
    """Transcript-rate probability for each query leg (needs speaker in FB_SPEAKER, word, k,
    ftype, event_date). Uses only documents dated within [event_date − lookback,
    event_date − min_lag]. Adds ``p_tx`` (NB rate model), ``p_df`` (shrunk share of same-type
    documents with ≥ k mentions), ``mu_tx`` (expected count), ``n_docs``."""
    dmeta = docs.select("document_id", "who", "date", "ftype", "words")
    cc = counts.join(dmeta, on=["document_id", "who"], how="inner")
    by_who = {w: cc.filter(pl.col("who") == w) for w in FB_SPEAKER}
    meta_who = {w: dmeta.filter(pl.col("who") == w) for w in FB_SPEAKER}
    out = []
    for (who, date, ftype), grp in q.group_by(["speaker", "event_date", "ftype"]):
        lo, hi = date - timedelta(days=prm.lookback_days), date - timedelta(days=prm.min_lag_days)
        dm = meta_who[who].filter((pl.col("date") >= lo) & (pl.col("date") <= hi))
        same = dm.filter(pl.col("ftype") == ftype)
        x_e = float(same["words"].median()) if same.height >= 2 else (float(dm["words"].median()) if dm.height else float("nan"))
        wts = dm.with_columns(pl.when(pl.col("ftype") == ftype).then(1.0).otherwise(prm.other_type_weight).alias("wt"))
        expo = float((wts["wt"] * wts["words"]).sum())
        c = by_who[who].filter((pl.col("date") >= lo) & (pl.col("date") <= hi))
        c = c.with_columns(pl.when(pl.col("ftype") == ftype).then(1.0).otherwise(prm.other_type_weight).alias("wt"))
        agg = c.group_by("word").agg((pl.col("wt") * pl.col("c")).sum().alias("wc"))
        sm = c.filter(pl.col("ftype") == ftype)
        g = grp.join(agg, on="word", how="left").with_columns(pl.col("wc").fill_null(0.0))
        # prior rate: the mean rate of this event's own listed words (all known at t)
        gw = g.unique("word")
        tot_rate = float(gw["wc"].sum()) / max(expo, 1.0) / max(gw.height, 1)
        lam = (g["wc"].to_numpy() + prm.prior_words * tot_rate) / (expo + prm.prior_words)
        mu = lam * x_e
        k = g["k"].to_numpy()
        p_tx = nb_tail(mu, prm.r, k)
        # document frequency among same-type documents (k-threshold), beta(1, 1) + shrink to p_tx
        if sm.height:
            hits = sm.join(g.select("word", "k").unique(), on="word").with_columns((pl.col("c") >= pl.col("k")).alias("hit"))
            dfreq = hits.group_by("word", "k").agg(pl.col("hit").sum().alias("dh"))
            g = g.join(dfreq, on=["word", "k"], how="left").with_columns(pl.col("dh").fill_null(0))
            n_same = same.height
        else:
            g = g.with_columns(pl.lit(0).alias("dh"))
            n_same = 0
        p_df = (g["dh"].to_numpy() + 2 * p_tx) / (n_same + 2)
        out.append(g.with_columns(pl.Series("p_tx", np.clip(p_tx, 0.003, 0.997)), pl.Series("p_df", np.clip(p_df, 0.003, 0.997)),
                                  pl.Series("mu_tx", mu), pl.lit(dm.height).alias("n_docs"),
                                  pl.lit(n_same).alias("n_same"), pl.lit(x_e).alias("x_e")))
    return pl.concat(out, how="diagonal_relaxed") if out else q


# ---- calibration layer (model-only features, fitted on dev) ---------------------------------
def features(df: pl.DataFrame, p_col: str, hl: float) -> np.ndarray:
    """Design for the calibrated base-rate variant: logit(p), new word for this speaker,
    log(1 + past listings), extra slash alternatives, threshold leg, log legs in the event."""
    from albert.research.fundamental import logit

    n_sw = df[f"n_sw_{hl:g}"].to_numpy()
    legs_ev = df.group_by("event", "when").agg(pl.len().alias("_n"))
    n_ev = df.join(legs_ev, on=["event", "when"], how="left")["_n"].to_numpy().astype(float)
    return np.column_stack([np.ones(df.height), logit(df[p_col].to_numpy()), (n_sw <= 0).astype(float),
                            np.log1p(n_sw), df["n_alts"].to_numpy() - 1.0, (df["k"].to_numpy() > 1).astype(float),
                            np.log(n_ev)])


def fit_calibration(df: pl.DataFrame, p_col: str, hl: float) -> np.ndarray:
    from albert.research.fundamental import fit_logistic

    X = features(df, p_col, hl)
    keep = (X.std(axis=0) > 1e-9)
    keep[0] = True                     # constant features (e.g. no threshold legs) get 0
    beta = np.zeros(X.shape[1])
    beta[keep], _ = fit_logistic(X[:, keep], df["y"].to_numpy().astype(float), df["event"].to_numpy())
    return beta


def apply_calibration(df: pl.DataFrame, p_col: str, hl: float, beta: np.ndarray) -> np.ndarray:
    return np.clip(1 / (1 + np.exp(-features(df, p_col, hl) @ beta)), 0.003, 0.997)


# ---- quotes, start verification, segments ---------------------------------------------------
def load_candles(s) -> pl.DataFrame:
    frames = []
    for name in (f"fundamental_{NAME}_candles.parquet", f"fundamental_{NAME}_candles_batch.parquet"):
        p = s.data_dir / "cache" / name
        if p.exists():
            frames.append(pl.read_parquet(p))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").unique(["ticker", "end_ts", "period_minutes"])


def start_check(lg: pl.DataFrame, starts: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    """Per speech event: share of legs whose hourly bid is ≥ 0.97 at the last candle by
    ``start − 1 h`` (``said_h1``), by ``start − 24 h`` and at ``start + 2 h``. A said word goes
    to 0.99 at once, so ``said_h1`` well above ``said_h24`` means the speech (or the words) came
    before the milestone start; such events are excluded from ``h1``. Outcome-blind (all legs
    count), so the exclusion uses only what is visible at ``t``."""
    from albert.research.fundamental import attach_quotes

    sp = lg.filter(~pl.col("series").is_in(WINDOW_SERIES)).join(starts.select("event", "start"), on="event")
    out = {}
    for lab, dh in (("said_h24", -24), ("said_h1", -1), ("said_p2", 2)):
        q = attach_quotes(sp.with_columns((pl.col("start") + timedelta(hours=dh)).alias("t"), pl.lit(0.5).alias("p")),
                          candles, max_age=timedelta(hours=3))
        q = q.group_by("event").agg((pl.col("bid") >= 0.97).mean().alias(lab), pl.len().alias(f"n_{lab}"))
        out[lab] = q
    ev = sp.select("event").unique()
    for q in out.values():
        ev = ev.join(q, on="event", how="left")
    return ev


def price_band(p: pl.Expr) -> pl.Expr:
    return (pl.when(p < 0.05).then(pl.lit("a <5c")).when(p < 0.15).then(pl.lit("b 5-15c"))
            .when(p < 0.35).then(pl.lit("c 15-35c")).when(p < 0.65).then(pl.lit("d 35-65c"))
            .when(p < 0.85).then(pl.lit("e 65-85c")).when(p < 0.95).then(pl.lit("f 85-95c"))
            .otherwise(pl.lit("g >=95c")))


def fee_exact(p: np.ndarray) -> np.ndarray:
    return np.ceil(np.round(0.07 * p * (1 - p) * 1e4, 9)) / 1e4


def fee_cent(p: np.ndarray, contracts: int = 1) -> np.ndarray:
    """Per-contract fee when each order of ``contracts`` is rounded up to the cent."""
    return np.ceil(np.round(0.07 * contracts * p * (1 - p) * 100, 9)) / 100 / contracts


def side_pnl(df: pl.DataFrame) -> pl.DataFrame:
    """Hold-to-settlement PnL of buying YES at the ask and of buying NO at 1 − bid, per
    contract, with the exact fee and with a 1-contract order rounded up to the cent."""
    ask, bid, y = df["ask"].to_numpy(), df["bid"].to_numpy(), df["y"].to_numpy().astype(float)
    no = 1 - bid
    return df.with_columns(pl.Series("yes_pnl", y - ask - fee_exact(ask)), pl.Series("no_pnl", (1 - y) - no - fee_exact(no)),
                           pl.Series("yes_pnl_1c", y - ask - fee_cent(ask)), pl.Series("no_pnl_1c", (1 - y) - no - fee_cent(no)))


def band_table(df: pl.DataFrame, by: str | list[str], model_col: str | None = None) -> pl.DataFrame:
    """Calibration and naive-side PnL by segment: realised YES rate vs mid (and model), PnL of
    buying every NO / every YES (cents per contract, t clustered by event)."""
    from albert.research.fundamental import clustered_mean

    by = [by] if isinstance(by, str) else by
    d = side_pnl(df.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid")))
    rows = []
    for key, g in d.group_by(by):
        ev = g["event"].to_numpy()
        r = {**dict(zip(by, key)), "n": g.height, "events": g["event"].n_unique(),
             "mid": round(float(g["mid"].mean()), 4), "y_rate": round(float(g["y"].mean()), 4),
             "spread_c": round(float((g["ask"] - g["bid"]).mean()) * 100, 2)}
        if model_col:
            r["model"] = round(float(g[model_col].mean()), 4)
        for col in ("no_pnl", "no_pnl_1c", "yes_pnl"):
            mu, t, _ = clustered_mean(g[col].to_numpy(), ev)
            r[f"{col}_c"] = round(mu * 100, 2)
            r[f"t_{col}"] = None if t is None else round(t, 2)
        rows.append(r)
    return pl.DataFrame(rows).sort(by)


# ---- event-level latent factor and the per-event portfolio ----------------------------------
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(21)
_GH_W = _GH_W / _GH_W.sum()


def event_loglik(p: np.ndarray, y: np.ndarray, ev: np.ndarray, sigma: float) -> float:
    """Marginal log likelihood with one shared N(0, σ²) shift of logit p per event (the
    latent "how long / how wide-ranging was the speech" factor)."""
    from albert.research.fundamental import logit

    lp = logit(p)
    total = 0.0
    for e in np.unique(ev):
        m = ev == e
        z = lp[m][None, :] + sigma * _GH_X[:, None]
        q = 1 / (1 + np.exp(-z))
        ll = (y[m][None, :] * np.log(q) + (1 - y[m][None, :]) * np.log(1 - q)).sum(axis=1)
        mx = ll.max()
        total += mx + math.log(float((_GH_W * np.exp(ll - mx)).sum()))
    return total


def fit_sigma(p: np.ndarray, y: np.ndarray, ev: np.ndarray, grid=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5)) -> tuple[float, list]:
    ll = [(s_, event_loglik(p, y, ev, s_)) for s_ in grid]
    return max(ll, key=lambda x: x[1])[0], ll


def portfolio(p: np.ndarray, bid: np.ndarray, ask: np.ndarray, sigma: float, *, wealth: float = 100.0,
              cap: float = 0.10, nmax: int = 20, draws: int = 4000, seed: int = 0,
              independent: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Integer contracts per leg (side +1 = YES at the ask, −1 = NO at 1 − bid) that greedily
    maximise E[log(wealth + PnL)] under the model with one latent event factor (σ on the
    logit scale; ``independent=True`` ignores it), after exact taker fees; total cost ≤ cap·wealth."""
    from albert.research.fundamental import logit

    rng = np.random.default_rng(seed)
    q = 1 / (1 + np.exp(-(logit(p)[None, :] + sigma * rng.standard_normal(draws)[:, None])))
    pm = q.mean(axis=0)                                   # marginal P(YES) per leg
    if independent:                                       # same marginals, no shared factor
        q = np.broadcast_to(pm, q.shape)
    yy = (rng.uniform(size=q.shape) < q).astype(float)
    cost_y, cost_n = ask + fee_exact(ask), (1 - bid) + fee_exact(1 - bid)
    side = np.where(pm - cost_y >= (1 - pm) - cost_n, 1, -1)
    cost = np.where(side == 1, cost_y, cost_n)
    pay = np.where(side[None, :] == 1, yy, 1 - yy) - cost[None, :]          # per-contract PnL draws
    edge = np.where(side == 1, pm, 1 - pm) - cost                           # expected PnL per contract
    n = np.zeros(len(p), dtype=int)
    pnl = np.zeros(draws)
    cur = float(np.log(wealth + pnl).mean())
    while True:
        ok = (n < nmax) & ((n * cost).sum() + cost <= cap * wealth) & (edge > 0)
        if not ok.any():
            break
        gains = np.full(len(p), -np.inf)
        for i in np.flatnonzero(ok):
            gains[i] = float(np.log(np.maximum(wealth + pnl + pay[:, i], 1e-9)).mean())
        i = int(np.argmax(gains))
        if gains[i] <= cur + 1e-12:
            break
        n[i] += 1
        pnl += pay[:, i]
        cur = gains[i]
    return n, side


# ---- the study -------------------------------------------------------------------------------
DATA_END = datetime(2026, 11, 1, tzinfo=UTC)      # kalshi_markets ingest end (settled before)
HALF_LIVES = (0.0, 14.0, 45.0)
DEV_SPLIT = datetime(2026, 9, 15, tzinfo=UTC)     # inner split of dev for choosing variants
TX_PARAMS = TX(lookback_days=45, other_type_weight=0.3, prior_words=30000.0, r=1.0)  # tuned on dev


def nb_mean_from_p1(p1: np.ndarray, r: float) -> np.ndarray:
    """Mean count μ with P(N ≥ 1) = p1 under NB(size r) (Poisson for r = ∞)."""
    p1 = np.clip(p1, 1e-6, 1 - 1e-6)
    if not math.isfinite(r):
        return -np.log(1 - p1)
    return r * ((1 - p1) ** (-1 / r) - 1)


def threshold_nb(lg: pl.DataFrame, part: pl.DataFrame, prm: BB, fit_until: datetime,
                 sizes: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, float("inf"))) -> tuple[pl.DataFrame, float | None]:
    """``p_bbn``: threshold legs ("N+ times") get their prior from a count model. The same
    word's 1+ base rate p1 (as of t, :func:`bb_predict` with k = 1) gives the NB mean, the tail
    P(N ≥ k) is the prior, then the (word, k) history updates it with the word-level pseudo
    count. NB size chosen on dev threshold legs by log loss. Other legs: ``p_bbn = p_bb``."""
    thr = part.filter(pl.col("k") > 1)
    part = part.with_columns(pl.col("p_bb").alias("p_bbn"))
    if not thr.height:
        return part, None
    base = thr.select([c for c in thr.columns if not c.startswith(("h_", "n_"))])
    c1 = asof_counts(lg, base.with_columns(pl.lit(1).alias("k")), (prm.half_life,))
    p1 = bb_predict(c1, prm)
    hl = f"{prm.half_life:g}"
    h, n, k = thr[f"h_sw_{hl}"].to_numpy(), thr[f"n_sw_{hl}"].to_numpy(), thr["k"].to_numpy()
    dev = (thr["t"] < fit_until).to_numpy()
    y = thr["y"].to_numpy().astype(float)
    best = None
    for r in sizes:
        pn = nb_tail(nb_mean_from_p1(p1, r), r, k)
        pk = np.clip((h + prm.k_w * pn) / (n + prm.k_w), 0.003, 0.997)
        if dev.any():
            ll = log_loss(pk[dev], y[dev])
            if best is None or ll < best[1]:
                best = (r, ll, pk)
    if best is None:
        return part, None
    upd = thr.select("ticker", "when").with_columns(pl.Series("_pn", best[2]))
    part = part.join(upd, on=["ticker", "when"], how="left").with_columns(
        pl.coalesce("_pn", "p_bbn").alias("p_bbn")).drop("_pn")
    return part, best[0]


# chosen on the inner dev split (fit < Sep 15, test Sep 15–30) before the October run
PRIMARY = ("h24/bb/all", "h1/bb/all", "h24/txc/tv", "h1/txc/tv", "window/bbc/all")


def build_predictions(s, lg: pl.DataFrame, starts: pl.DataFrame,
                      fit_until: datetime = HOLDOUT_FROM) -> tuple[pl.DataFrame, dict[str, Any]]:
    """All model variants for every (leg, decision time). Hyper-parameters and calibration
    coefficients are fitted on dev rows only (``t`` < 2026-10-01); counts and transcripts
    are always as of ``t``."""
    from albert.research.fundamental import fit_logistic, logit

    info: dict[str, Any] = {}
    D = decision_frame(lg, starts, DATA_END)
    C = asof_counts(lg, D, HALF_LIVES).with_columns(pl.col("series").is_in(WINDOW_SERIES).alias("window"))
    dev = C.filter(pl.col("t") < fit_until)
    parts = []
    for kind, part in C.group_by("window"):
        dv = dev.filter(pl.col("window") == kind[0])
        prm, _ = tune_bb(dv, HALF_LIVES)
        info[f"bb_params_{'window' if kind[0] else 'speech'}"] = prm.__dict__
        part = part.with_columns(pl.Series("p_bb", bb_predict(part, prm)), pl.lit(prm.half_life).alias("hl"))
        part, r_nb = threshold_nb(lg, part, prm, fit_until)
        info[f"nb_size_{'window' if kind[0] else 'speech'}"] = r_nb
        dvp = part.filter(pl.col("t") < fit_until)
        beta = fit_calibration(dvp, "p_bb", prm.half_life)
        info[f"bbc_coef_{'window' if kind[0] else 'speech'}"] = [round(float(b), 4) for b in beta]
        part = part.with_columns(pl.Series("p_bbc", apply_calibration(part, "p_bb", prm.half_life, beta)))
        parts.append(part)
    P = pl.concat(parts, how="diagonal_relaxed")
    # transcript model for Trump / Vance speech legs (documents dated ≥ 1 day before the event)
    C_ = s.data_dir / "cache" / NAME
    docs = pl.read_parquet(C_ / "factbase_docs.parquet")
    cnt = pl.read_parquet(C_ / "factbase_word_counts.parquet")
    q = P.filter(pl.col("speaker").is_in(list(FB_SPEAKER)) & ~pl.col("window"))
    q = q.with_columns(pl.col("start").dt.convert_time_zone(ET).dt.date().alias("event_date"),
                       pl.struct("title", "etype").map_elements(lambda r: kalshi_ftype(r["title"], r["etype"]),
                                                                return_dtype=pl.String).alias("ftype"))
    # h24: documents dated ≤ event day − 2 (a doc dated the day before could postdate start − 24 h);
    # h1: ≤ event day − 1 (Factbase posts within hours; see caveats in the report)
    from dataclasses import replace
    tx = pl.concat([transcript_predict(q.filter(pl.col("when") == w), docs, cnt, replace(TX_PARAMS, min_lag_days=lag))
                    for w, lag in (("h24", 2), ("h1", 1))], how="diagonal_relaxed")
    tx = tx.select("ticker", "when", "p_tx", "p_df", "mu_tx", "n_docs", "n_same", "ftype")
    P = P.join(tx, on=["ticker", "when"], how="left")
    dv = P.filter((pl.col("t") < fit_until) & pl.col("p_tx").is_not_null())
    X = np.column_stack([np.ones(dv.height), logit(dv["p_tx"].to_numpy())])
    b_tx, t_tx = fit_logistic(X, dv["y"].to_numpy().astype(float), dv["event"].to_numpy())
    X2 = np.column_stack([np.ones(dv.height), logit(dv["p_tx"].to_numpy()), logit(dv["p_bbc"].to_numpy())])
    b_mix, t_mix = fit_logistic(X2, dv["y"].to_numpy().astype(float), dv["event"].to_numpy())
    info["txc_coef"] = {"a": round(float(b_tx[0]), 4), "b": round(float(b_tx[1]), 4), "t_b": round(float(t_tx[1]), 2)}
    info["mix_coef"] = {"a": round(float(b_mix[0]), 4), "b_tx": round(float(b_mix[1]), 4), "b_bbc": round(float(b_mix[2]), 4),
                        "t_tx": round(float(t_mix[1]), 2), "t_bbc": round(float(t_mix[2]), 2)}
    has = pl.col("p_tx").is_not_null()
    ltx = pl.col("p_tx").clip(0.005, 0.995)
    ltx = (ltx / (1 - ltx)).log()
    lbc = pl.col("p_bbc").clip(0.005, 0.995)
    lbc = (lbc / (1 - lbc)).log()
    P = P.with_columns(
        pl.when(has).then(1 / (1 + (-(b_tx[0] + b_tx[1] * ltx)).exp())).alias("p_txc"),
        pl.when(has).then(1 / (1 + (-(b_mix[0] + b_mix[1] * ltx + b_mix[2] * lbc)).exp())).alias("p_mix"))
    return P, info


def blend_prob(df: pl.DataFrame, blend: dict[str, Any], col: str) -> np.ndarray:
    from albert.research.fundamental import logit

    mid = ((df["bid"] + df["ask"]) / 2).to_numpy()
    z = blend["a"] + blend["b_mid"] * logit(mid) + blend["c_model"] * logit(df[col].to_numpy())
    return 1 / (1 + np.exp(-z))


def word_band(n: pl.Expr) -> pl.Expr:
    return pl.when(n <= 0).then(pl.lit("new word")).when(n < 4).then(pl.lit("seen 1-3x")).otherwise(pl.lit("seen 4+x"))


def series_group(speaker: pl.Expr) -> pl.Expr:
    return (pl.when(speaker.is_in(["trump", "vance", "earnings", "worldnews", "bernie"])).then(speaker)
            .otherwise(pl.lit("other")))


def taker_by(tr: pl.DataFrame, by: str) -> pl.DataFrame:
    from albert.research.fundamental import clustered_mean

    rows = []
    for key, g in tr.group_by(by):
        mu, t, k = clustered_mean(g["pnl"].to_numpy(), g["event"].to_numpy())
        mu1, t1, _ = clustered_mean(g["pnl_1c"].to_numpy(), g["event"].to_numpy())
        rows.append({by: key[0], "trades": g.height, "events": k, "pnl_c": round(mu * 100, 2),
                     "t": None if t is None else round(t, 2), "pnl_1c_floor_c": round(mu1 * 100, 2),
                     "t_1c": None if t1 is None else round(t1, 2), "no_share": round(float((g["side"] == "no").mean()), 3)})
    return pl.DataFrame(rows).sort(by) if rows else pl.DataFrame()


def with_1c(tr: pl.DataFrame) -> pl.DataFrame:
    """Add the PnL of the same trades if each were a 1-contract order rounded up to the cent."""
    if not tr.height:
        return tr
    px = np.where(tr["side"].to_numpy() == "yes", tr["ask"].to_numpy(), 1 - tr["bid"].to_numpy())
    return tr.with_columns(pl.Series("pnl_1c", tr["pnl"].to_numpy() + fee_exact(px) - fee_cent(px)))


def run_portfolio(df: pl.DataFrame, col: str, sigma: float, **kw) -> pl.DataFrame:
    """Per event (one decision time): correlated optimiser vs independent optimiser vs the
    single best leg (same Kelly objective, only that leg allowed). Realised PnL in dollars."""
    rows = []
    for (ev,), g in df.group_by(["event"]):
        p, bid, ask, y = g[col].to_numpy(), g["bid"].to_numpy(), g["ask"].to_numpy(), g["y"].to_numpy().astype(float)
        res = {"event": ev, "t": g["t"][0], "legs": g.height}
        for name, indep in (("joint", False), ("indep", True)):
            n, side = portfolio(p, bid, ask, sigma, independent=indep, **kw)
            cost = np.where(side == 1, ask + fee_exact(ask), 1 - bid + fee_exact(1 - bid))
            pay = np.where(side == 1, y, 1 - y)
            res[f"{name}_contracts"] = int(n.sum())
            res[f"{name}_cost"] = float((n * cost).sum())
            res[f"{name}_pnl"] = float((n * (pay - cost)).sum())
        # single best leg: largest model edge after fees, sized by the same objective
        ey = p - ask - fee_exact(ask)
        en = (1 - p) - (1 - bid) - fee_exact(1 - bid)
        e = np.maximum(ey, en)
        i = int(np.argmax(e))
        if e[i] > 0:
            n1, s1 = portfolio(p[i:i + 1], bid[i:i + 1], ask[i:i + 1], 0.0, **kw)
            c1 = ask[i] + fee_exact(ask[i]) if s1[0] == 1 else 1 - bid[i] + fee_exact(1 - bid[i])
            pay1 = y[i] if s1[0] == 1 else 1 - y[i]
            res["single_contracts"], res["single_pnl"] = int(n1[0]), float(n1[0] * (pay1 - c1))
        else:
            res["single_contracts"], res["single_pnl"] = 0, 0.0
        rows.append(res)
    return pl.DataFrame(rows)


def summarize_portfolio(pf: pl.DataFrame) -> list[dict[str, Any]]:
    out = []
    for name in ("joint", "indep", "single"):
        x = pf[f"{name}_pnl"].to_numpy()
        traded = pf[f"{name}_contracts"].to_numpy() > 0
        sd = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
        out.append({"strategy": name, "events": pf.height, "events_traded": int(traded.sum()),
                    "contracts": int(pf[f"{name}_contracts"].sum()), "pnl_$": round(float(x.sum()), 2),
                    "mean_$_per_event": round(float(x.mean()), 3),
                    "t": round(float(x.mean()) / (sd / math.sqrt(len(x))), 2) if sd and sd > 0 else None})
    return out


def run(s, *, holdout: bool = True) -> dict[str, Any]:
    """The whole study (``holdout=False`` reports dev only, for model selection)."""
    from albert.research.fundamental import attach_quotes, evaluate, taker_pnl

    C_ = s.data_dir / "cache" / NAME
    mk = pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet").filter(pl.col("category") == "Mentions").collect()
    lg = legs(mk)
    starts = pl.read_parquet(C_ / "starts.parquet")
    P, info = build_predictions(s, lg, starts, HOLDOUT_FROM if holdout else DEV_SPLIT)
    candles = load_candles(s)
    sc = start_check(lg, starts, candles)
    bad = sc.filter(pl.col("said_h1").fill_null(0) > pl.col("said_h24").fill_null(0) + 0.12)["event"].to_list()
    info["start_check"] = {"events": sc.height, "with_h1_quotes": int(sc["said_h1"].is_not_null().sum()),
                           "median_said_h1": sc["said_h1"].median(), "median_said_p2": sc["said_p2"].median(),
                           "excluded_started_early": len(bad)}
    Q = attach_quotes(P.with_columns(pl.col("p_bb").alias("p")), candles, max_age=timedelta(hours=2))
    bad24 = sc.filter(pl.col("said_h24").fill_null(0) > 0.12)["event"].to_list()
    info["start_check"]["excluded_h24"] = len(bad24)
    Q = Q.filter(~(pl.col("event").is_in(bad) & (pl.col("when") == "h1")) & ~(pl.col("event").is_in(bad24) & (pl.col("when") == "h24")))
    Q = Q.with_columns(price_band((pl.col("bid") + pl.col("ask")) / 2).alias("band"),
                       word_band(pl.col("n_sw_0")).alias("wband"), series_group(pl.col("speaker")).alias("sgroup"),
                       pl.when(pl.col("window")).then(pl.lit("window")).otherwise(pl.col("when")).alias("dt"))
    info["quoted_rows"] = Q.group_by("dt").agg(pl.len().alias("rows"), pl.col("event").n_unique().alias("events")).sort("dt").to_dicts()
    split_from = HOLDOUT_FROM if holdout else DEV_SPLIT
    Q = Q if holdout else Q.filter(pl.col("t") < HOLDOUT_FROM)
    results: dict[str, Any] = {}
    tables: dict[str, pl.DataFrame] = {}
    combos = [("bb", "all"), ("bbn", "all"), ("bbc", "all"), ("bb", "tv"), ("bbc", "tv"), ("txc", "tv"), ("mix", "tv"),
              ("bb", "thr"), ("bbn", "thr")]
    for dt in ("h24", "h1", "window"):
        for var, scope in combos:
            if dt == "window" and scope != "all":
                continue
            d = Q.filter(pl.col("dt") == dt)
            if scope == "tv":
                d = d.filter(pl.col("p_tx").is_not_null())
            elif scope == "thr":
                d = d.filter(pl.col("k") > 1)
            d = d.with_columns(pl.col(f"p_{var}").alias("p"))
            if not d.height:
                continue
            r = evaluate(d, split_from)
            results[f"{dt}/{var}/{scope}"] = r
            primary = f"{dt}/{var}/{scope}" in PRIMARY
            if r.get("blend") and r.get("theta_chosen_on_dev") is not None:
                th = r["theta_chosen_on_dev"]
                ho = d.filter(pl.col("t") >= split_from)
                tr = with_1c(taker_pnl(ho, th))
                if tr.height and primary:
                    for by in ("band", "wband", "sgroup", "series"):
                        tables[f"{dt}/{var}/{scope} taker θ={th} by {by} (holdout)"] = taker_by(tr, by)
                # the dev-fitted blend as the fair value
                bl = d.with_columns(pl.Series("p", blend_prob(d, r["blend"], f"p_{var}")))
                rb = evaluate(bl, split_from)
                results[f"{dt}/{var}/{scope}/blend-as-p"] = {"taker": rb["taker"], "theta_chosen_on_dev": rb["theta_chosen_on_dev"]}
    # longshot / band calibration of the price itself (no model), both splits
    for dt in ("h24", "h1", "window"):
        d = Q.filter(pl.col("dt") == dt)
        for split, part in (("dev", d.filter(pl.col("t") < split_from)), ("holdout", d.filter(pl.col("t") >= split_from))):
            if part.height:
                tables[f"price bands {dt} {split}: realised YES vs mid, buy-every-NO / YES PnL"] = band_table(part, "band", "p_bb")
    # pre-registered on dev (2026-11-05, before the October run): KXWORLDNEWSMENTION YES legs
    # looked overpriced at h1 (see NOTES)
    for dt in ("h24", "h1"):
        d = Q.filter(pl.col("dt") == dt)
        for split, part in (("dev", d.filter(pl.col("t") < split_from)), ("holdout", d.filter(pl.col("t") >= split_from))):
            if part.height:
                tables[f"series groups {dt} {split}: buy-every-NO / YES PnL (no model)"] = band_table(part, "sgroup", "p_bb")
    hb = Q.filter((pl.col("dt") == "h1") & (pl.col("t") >= split_from))
    if hb.height:
        tables["h1 holdout by series group x band (buy every NO)"] = band_table(hb, ["sgroup", "band"])
    # per-event portfolio (h1, speech) with the latent event factor, fair value = dev blend
    for key, var in (("h1/bb/all", "bb"), ("h1/txc/tv", "txc")):
        if key not in results or not results[key].get("blend"):
            continue
        d = Q.filter(pl.col("dt") == "h1")
        if key.endswith("/tv"):
            d = d.filter(pl.col("p_tx").is_not_null())
        d = d.with_columns(pl.Series("pf", blend_prob(d, results[key]["blend"], f"p_{var}")))
        dv = d.filter(pl.col("t") < split_from)
        sig, grid = fit_sigma(dv["pf"].to_numpy(), dv["y"].to_numpy().astype(float), dv["event"].to_numpy())
        info[f"latent_sigma_{key}"] = [(g, round(v, 1)) for g, v in grid]
        if key == "h1/bb/all":
            info["latent_sigma_mid"] = fit_sigma(((dv["bid"] + dv["ask"]) / 2).to_numpy(), dv["y"].to_numpy().astype(float),
                                                 dv["event"].to_numpy())[0]
        for split, part in (("dev", dv), ("holdout", d.filter(pl.col("t") >= split_from))):
            if part.height:
                pf = run_portfolio(part, "pf", sig)
                tables[f"per-event portfolio {key} ({split}), fair value = dev blend(model, mid), σ={sig}"] = \
                    pl.DataFrame(summarize_portfolio(pf))
    info["holdout"] = holdout
    if holdout:
        out = s.data_dir / "cache" / f"fundamental_{NAME}_predictions.parquet"
        Q.drop([c for c in Q.columns if c.startswith(("h_", "n_")) and c not in ("n_sw_0",)]).write_parquet(out)
    return {"results": results, "tables": tables, "info": info}


def summary_table(results: dict[str, Any]) -> pl.DataFrame:
    """One row per (decision time, variant, scope): Brier model/mid per split, blend c and t,
    dev-chosen θ and the taker result at that θ on each split."""
    rows = []
    for key, r in results.items():
        if key.endswith("/blend-as-p") or "scores" not in r:
            continue
        row: dict[str, Any] = {"run": key, "primary": key in PRIMARY}
        for sc in r["scores"]:
            row[f"brier_model_{sc['split']}"] = sc["brier_model"]
            row[f"brier_mid_{sc['split']}"] = sc["brier_mid"]
            row[f"n_{sc['split']}"] = sc["n"]
        b = r.get("blend") or {}
        row.update({"c": b.get("c_model"), "t_c": b.get("t_c"), "brier_blend_ho": b.get("brier_blend_holdout"),
                    "c_ho_refit": b.get("c_model_holdout_refit"), "t_c_ho_refit": b.get("t_c_holdout_refit")})
        th = r.get("theta_chosen_on_dev")
        row["theta"] = th
        for rr in [x for x in r["taker"] if x["theta"] == th]:
            row[f"taker_{rr['split']}_c"] = rr.get("pnl_c")
            row[f"taker_{rr['split']}_t"] = rr.get("t")
            row[f"taker_{rr['split']}_n"] = rr.get("trades")
        bp = results.get(key + "/blend-as-p")
        if bp:
            th2 = bp["theta_chosen_on_dev"]
            row["blendp_theta"] = th2
            for rr in [x for x in bp["taker"] if x["theta"] == th2]:
                row[f"blendp_{rr['split']}_c"] = rr.get("pnl_c")
                row[f"blendp_{rr['split']}_t"] = rr.get("t")
                row[f"blendp_{rr['split']}_n"] = rr.get("trades")
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def write_report(s, out: dict[str, Any], main_key: str, notes: str = "") -> Path:
    from albert.reports import Report

    rep = Report(s.reports_dir, f"research-fundamental-{NAME}")
    if notes:
        rep.text("Notes", notes)
    rep.json("Run info", out["info"])
    rep.table("Summary (all variants; holdout = October 2026 unless noted)", summary_table(out["results"]),
              save_as="summary", max_rows=60)
    r = out["results"][main_key]
    rep.table(f"Scores {main_key}", pl.DataFrame(r["scores"]))
    if r.get("blend"):
        rep.json(f"Blend {main_key}: logit P = a + b·logit(mid) + c·logit(model), fitted on dev", r["blend"])
    rep.table(f"Taker {main_key} (cents per contract, exact fee)", pl.DataFrame(r["taker"], infer_schema_length=None))
    for title, tab in out["tables"].items():
        if tab.height:
            rep.table(title, tab, max_rows=60)
    return rep.save()


NOTES = """\
**Question.** Do base rates (past Kalshi mention results) or transcript word rates (Roll Call
Factba.se) beat Kalshi mention prices after fees on the October 2026 holdout, and in which segment?

**Data.** Every settled Mentions-category market Aug 17–Oct 31 2026 (179 series, ~9.1k word legs,
~470 events; `Event does not qualify` legs dropped). Hourly candles (yes bid/ask close) per leg
from Kalshi (historical endpoint per ticker before the 2026-09-06 cutoff, batch endpoint after).
Factba.se: public search JSON + public transcript pages for Trump and Vance, 2026-06-01 onward
(~400 videos with speaker-segmented text).

**Timing / no look-ahead.**
* Event start = Kalshi milestone `start_date` (public schedule). Verified per event: share of
  legs with bid ≥ 0.97 is ~0 at start−24 h and start−1 h (median 0) and high at start+2 h;
  events where it jumps by > 0.12 before start−1 h are dropped from the 1-hour decision (the
  speech came earlier than the milestone), and from `h24` if > 0.12 already then. The rule is
  outcome-blind (all legs; only quotes visible at `t`).
* Decisions at start−24 h (`h24`) and start−1 h (`h1`); legs listed after `t` are dropped.
  Window events ("Will Trump say X before <date>", KXTRUMPSAY*): `t` = window open + 1 h / + 24 h
  (`window`), legs already closed (said) at `t` dropped, windows ending after the data end
  (Nov 1) dropped (only their early-YES legs had settled).
* Quotes: last hourly candle that ended at or before `t` (≤ 2 h old).
* Base rates: only legs with `settlement_ts < t`.
* Transcripts: documents dated ≤ event day − 1 (`h1`) or − 2 (`h24`), ET. Factba.se usually
  posts within hours, but first-publication times are not recorded (caveat).
* All hyper-parameters, calibration coefficients, the blend, θ and σ are fitted on dev
  (t < 2026-10-01). Variants were chosen on an inner dev split (fit < Sep 15, test Sep 15–30)
  before the October run.

**Models.**
* `bb`: hierarchical beta-binomial: global → speaker → speaker×event type×(threshold or not)
  → same word for other speakers → same word (and threshold) for this speaker, pseudo-count
  shrinkage at each level and exponential decay of old events, tuned by dev log loss.
* `bbn`: `bb` with a negative-binomial prior for "N+ times" legs (mean from the same word's
  1+ base rate, size chosen on dev); worse than `bb` on dev threshold legs, so not primary.
* `bbc`: `bb` + logistic calibration with model-only features (new word, log listings,
  alternatives, threshold leg, legs in the event).
* `txc`: Factba.se rate model: speaker word rate (same-type documents weight 1, others 0.3,
  45-day window, gamma prior), expected count = rate × median speaker words of the event type,
  P(count ≥ k) from a negative binomial (size 1), then a 2-parameter logistic calibration (dev).
  Counting rule = Kalshi's (exact word/phrase, plural or possessive, slash alternatives);
  checked post hoc on 2,151 legs whose own event transcript exists: 97.4% agreement (residual
  mostly event↔document mismatches).
* `mix`: logistic on logit `txc`-raw and logit `bbc` (dev).
* Primary variants (chosen on the inner dev split): speech base rate = `bb`, window base rate
  = `bbc`, transcript = `txc`.
* Pre-registered segment (seen on dev before the October run, no model): KXWORLDNEWSMENTION
  (ABC World News Tonight) YES legs overpriced at `h1` (buy every YES −8.9c / −15.5c), buy
  every NO +0.3c (Aug 17–Sep 14, t 0.1, 20 events) and +7.4c (Sep 15–30, t 2.4, 12 events)
  per contract with the 1c floor.

**Fees.** Taker 0.07·P·(1−P) exact (Kalshi charges to $0.0001 and rebates the cent rounding);
`1c floor` columns charge a 1-contract order rounded up to the cent (worst case, e.g. 1c on a
95c NO vs 0.33c exact).

**Portfolio.** Per event at `h1`: greedy integer contracts maximising E[log(100 + PnL)] with a
10%-of-bankroll cost cap and ≤ 20 contracts per leg, under the dev blend with one latent
N(0, σ²) logit shift per event (σ by dev marginal likelihood: the "speech length/range" factor);
compared with the same optimiser ignoring the correlation and with the single best leg.
"""
