"""Presidential-action count markets on Kalshi: executive orders, "presidential actions", nominations.

Question (2026-11-20): is there a repeatable small-stake edge in the weekly / daily presidential-action
series, using same-day public sources (the White House "Presidential Actions" feed, the Federal
Register, senate.gov, nyc.gov) or the series' own price structure?

Series (rules read 2026-11-20, ``data/research/exec_actions/rules.json``):

* ``KXEOWEEK``: EOs signed in a Sun-Sat week, by the **Federal Register signing date**; trades until
  the FR has published (up to two weeks after the week).
* ``KXTRUMPEOD`` (daily): an EO signed that day whose full text is on whitehouse.gov / pool by 10:00 ET
  the next day (numbered EOs only).
* ``KXTRUMPACT`` (weekly ladder) / ``KXTRUMPACTD`` (daily): items on whitehouse.gov "Presidential
  Actions" by the **displayed publication date** (checked 10:00 ET next day).
* ``KXTRUMPNOMNUM``: nominations sent to the Senate, from the same page's publication dates.
* ``KXMAMDANIEO``: NYC Mayor's non-emergency EOs signed in the week (expiry three months later).
* ``KXTRUMPENDORSEMENTS`` (Truth Social) and ``KXTRUMPPHOTO`` / ``KXTRUMPPHOTOD`` (Getty): sources block
  us (robots), so price and outcome only. Monthly ``KXBILLSSIGNED`` / ``KXPARDONSTRUMP``: few events.

Sources (robots verdicts in ``robots.json``): the WH Presidential Actions RSS feed (``/presidential-
actions/feed/?paged=N``; ``pubDate`` equals the listing's displayed ``datetime``; wp-json answers 403 and
is not used), the Federal Register API (EO signing dates), senate.gov, nyc.gov; Kalshi public API.
Raw non-Kalshi responses are cached append-only under ``data/cache/exec_actions/<host>/`` keyed by
URL + params + fetch date.

    python -m albert.research.exec_actions robots | discover | wh | fr | candles | build
    python -m albert.research.exec_actions screen | freeze | holdout | report
    python -m albert.research.exec_actions record      # one forward snapshot (WH feed, nyc.gov, Kalshi books)
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from albert.research.counts import AGENTS, market_row, robots_verdict
from albert.venues.kalshi import num

NAME = "exec_actions"
OUT = Path("data/research/exec_actions")
CACHE = Path("data/cache/exec_actions")
ET = ZoneInfo("America/New_York")
SNAP = "2026-11-20"                                   # cache tag for listings fetched in this study
DEV_END = date(2026, 9, 30)                           # dev: period's last day <= this
HOLD_START = date(2026, 10, 1)
FORWARD_START = date(2026, 11, 21)                   # periods ending on/after this were not open when the rule froze
KALSHI_MIN_INTERVAL = 1.0                             # sequential, <= 1 request/s (shared IP)
UA = "albert-research/0.2 (public-data research)"

WEEKLY = ("KXEOWEEK", "KXTRUMPACT", "KXTRUMPNOMNUM", "KXMAMDANIEO", "KXTRUMPENDORSEMENTS", "KXTRUMPPHOTO")
DAILY = ("KXTRUMPACTD", "KXTRUMPEOD", "KXTRUMPPHOTOD")
MONTHLY = ("KXBILLSSIGNED", "KXBILLSCOUNT", "KXPARDONSTRUMP", "KXPARDONANYONE", "KXBILLSCOUNTWEEKLY")
SERIES = WEEKLY + DAILY + MONTHLY
WH_SERIES = ("KXTRUMPACT", "KXTRUMPACTD", "KXTRUMPNOMNUM", "KXEOWEEK", "KXTRUMPEOD")   # nowcast from the WH feed
BLOCKED_SOURCE = {"KXTRUMPENDORSEMENTS": "truthsocial.com (robots 403)", "KXTRUMPPHOTO": "gettyimages.com (robots 403)",
                  "KXTRUMPPHOTOD": "gettyimages.com (robots 403)"}

ROBOTS_CHECKS: dict[str, list[str]] = {
    "www.whitehouse.gov": ["/presidential-actions/", "/presidential-actions/feed/", "/wp-json/wp/v2/posts"],
    "www.federalregister.gov": ["/api/v1/documents.json", "/api/v1/public-inspection-documents.json"],
    "www.senate.gov": ["/legislative/2026_schedule.htm", "/legislative/nominations_new.htm"],
    "www.nyc.gov": ["/mayors-office/news/executive-orders.page", "/office-of-the-mayor/news.page"],
    "www.gettyimages.com": ["/photos/donald-trump", "/search/2/image"],
    "truthsocial.com": ["/@realDonaldTrump"],
    "www.presidency.ucsb.edu": ["/documents/app-categories/presidential/executive-orders"],
    "www.govinfo.gov": ["/app/collection/dcpd"],
    "www.congress.gov": ["/nominations"],
    "rollcall.com": ["/factbase/trump/topic/calendar/"],
    "api.elections.kalshi.com": ["/trade-api/v2/markets"],
}


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _kalshi(min_interval: float = KALSHI_MIN_INTERVAL):
    from albert.config import Settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    s = Settings()
    return k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=min_interval, max_retries=12))


def _web(host: str, min_interval: float = 2.0):
    """Cached client for a public host (append-only raw records under ``data/cache/exec_actions/<host>``)."""
    from albert.http import CachedClient

    return CachedClient(host, f"https://{host}", CACHE, min_interval=min_interval, max_retries=4, user_agent=UA)


# ---- robots ----------------------------------------------------------------------------------------
def check_robots(out: Path = OUT) -> list[dict[str, Any]]:
    import httpx

    rows = []
    (out / "robots").mkdir(parents=True, exist_ok=True)
    for host, paths in ROBOTS_CHECKS.items():
        try:
            r = httpx.get(f"https://{host}/robots.txt", timeout=20, follow_redirects=True, headers={"User-Agent": UA})
            status, text = r.status_code, (r.text if r.status_code == 200 else "")
        except httpx.HTTPError as error:
            status, text = None, ""
            print(f"robots {host}: {error}", flush=True)
        (out / "robots" / f"{host}.txt").write_text(text)
        for p in paths:
            v = robots_verdict(text, p) if status in (200, 404, 410) else dict.fromkeys(AGENTS, False)
            rows.append({"host": host, "path": p, "http_status": status,
                         "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                         **{f"allow_{a}": v[a] for a in AGENTS}, "allowed": all(v.values())})
    (out / "robots.json").write_text(json.dumps(rows, indent=1))
    return rows


# ---- Kalshi markets ----------------------------------------------------------------------------------
MARKET_COLS = {"ticker": pl.String, "event_ticker": pl.String, "series_ticker": pl.String,
               "title": pl.String, "yes_sub_title": pl.String, "strike_type": pl.String,
               "floor_strike": pl.Float64, "cap_strike": pl.Float64, "open_time": pl.Datetime("us", "UTC"),
               "close_time": pl.Datetime("us", "UTC"), "expected_expiration_time": pl.Datetime("us", "UTC"),
               "settlement_ts": pl.Datetime("us", "UTC"), "result": pl.String, "expiration_value": pl.String,
               "can_close_early": pl.Boolean, "volume": pl.Float64, "rules_primary": pl.String,
               "status": pl.String, "src": pl.String}


def _pages(api, path: str, params: dict[str, Any], key: str = "markets", snap: str = SNAP):
    cursor = None
    while True:
        page = api.client.get(path, {**params, "cursor": cursor}, cache=True, tag=snap)
        yield page
        cursor = page.payload.get("cursor")
        if not cursor or not page.payload.get(key):
            return


def discover(out: Path = OUT, *, snap: str = SNAP, log=print) -> pl.DataFrame:
    """Every settled market of the slice series (``/historical/markets`` + ``/markets?status=settled``),
    the series' fee terms from today's ``/series``, and the current rules of each series."""
    api = _kalshi()
    ser = {t: (api.client.get(f"/series/{t}", cache=True, tag=snap).payload.get("series") or {}) for t in SERIES}
    srows = [{"series_ticker": t, "title": s.get("title"), "category": s.get("category"), "frequency": s.get("frequency"),
              "fee_type": s.get("fee_type"), "fee_multiplier": num(s.get("fee_multiplier")),
              "source": ((s.get("settlement_sources") or [{}])[0] or {}).get("name")} for t, s in ser.items() if s]
    _save(pl.DataFrame(srows, infer_schema_length=None), out / "series.parquet")
    rows: list[dict[str, Any]] = []
    for t in SERIES:
        for page in _pages(api, "/historical/markets", {"series_ticker": t, "limit": 1000}, snap=snap):
            rows.extend(market_row(m, t, "hist") for m in page.payload.get("markets") or [])
        for page in _pages(api, "/markets", {"series_ticker": t, "status": "settled", "limit": 1000}, snap=snap):
            rows.extend(market_row(m, t, "live") for m in page.payload.get("markets") or [])
        log(f"  discover {t}: {len(rows)} markets so far", flush=True)
    mk = (pl.DataFrame(rows, schema=MARKET_COLS).unique("ticker", keep="last")
          .filter(pl.col("result").is_in(["yes", "no"])))
    mk = mk.with_columns(pl.Series("period", [json.dumps(period_of(r)) for r in mk.iter_rows(named=True)]))
    _save(mk, out / "markets.parquet")
    log(f"discover: {mk.height} settled markets, {mk['event_ticker'].n_unique()} events")
    return mk


_MON = {m: i + 1 for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
DATE_RX = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? (\d{1,2}),? (\d{4})")
_D = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4}"
RANGE_RX = re.compile(rf"\b(?:from|between|during)\s+({_D})\s+(?:through|and|to)\s+({_D})")
MONTH_RX = re.compile(r"\bin (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (\d{4})")


def _dates(text: str | None) -> list[date]:
    return [date(int(y), _MON[m], int(d)) for m, d, y in DATE_RX.findall(text or "")]


def period_of(m: dict[str, Any]) -> tuple[str, str] | None:
    """(first, last) local day of the period counted, from the rules (daily: the first date named;
    weekly: the first two; monthly: "in Mon YYYY")."""
    s, rules = m.get("series_ticker"), (m.get("rules_primary") or "") + " " + (m.get("title") or "")
    ds = _dates(rules)
    if s in DAILY or s in WEEKLY:
        span = RANGE_RX.search(rules)
        if span:                                     # "from X through Y", "between X and Y", "during X to Y"
            a, b = _dates(span.group(1))[0], _dates(span.group(2))[0]
            return (a.isoformat(), b.isoformat()) if 0 <= (b - a).days <= 10 else None
        return (ds[0].isoformat(), ds[0].isoformat()) if ds else None     # "on X (as checked ... X+1)"
    mm = MONTH_RX.search(rules)
    if mm:
        y, mo = int(mm.group(2)), _MON[mm.group(1)]
        last = (date(y + (mo == 12), mo % 12 + 1, 1) - timedelta(days=1))
        return date(y, mo, 1).isoformat(), last.isoformat()
    return None


def strike_yes(count: float | None, strike_type: str | None, lo: float | None, hi: float | None) -> int | None:
    """YES label of a ladder leg for a final count (custom/daily legs: count > 0)."""
    if count is None:
        return None
    if strike_type == "greater":
        return int(count > lo)
    if strike_type == "greater_or_equal":
        return int(count >= lo)
    if strike_type == "less":
        return int(count < hi)
    if strike_type == "less_or_equal":
        return int(count <= hi)
    if strike_type == "between":
        return int(lo <= count <= hi)
    return int(count > 0)


def need_for_yes(strike_type: str | None, lo: float | None, hi: float | None) -> tuple[float, float]:
    """Counts c with YES: lo_c <= c <= hi_c (inf for open ladders)."""
    if strike_type == "greater":
        return math.floor(lo) + 1, math.inf
    if strike_type == "greater_or_equal":
        return math.ceil(lo), math.inf
    if strike_type == "less":
        return 0, math.ceil(hi) - 1
    if strike_type == "less_or_equal":
        return 0, math.floor(hi)
    if strike_type == "between":
        return math.ceil(lo), math.floor(hi)
    return 1, math.inf


# ---- White House Presidential Actions feed ---------------------------------------------------------
WH_HOST = "www.whitehouse.gov"
WH_COLS = {"guid": pl.String, "title": pl.String, "link": pl.String, "pub_ts": pl.Datetime("us", "UTC"),
           "pub_date_et": pl.Date, "categories": pl.String, "kind": pl.String, "n_nom": pl.Int64,
           "n_withdraw": pl.Int64, "round_time": pl.Boolean, "fetched_at": pl.String}
KIND_ORDER = (("Executive Orders", "eo"), ("Proclamations", "proclamation"), ("Presidential Memoranda", "memo"),
              ("Nominations & Appointments", "nomination"))


def _rss_time(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%a, %d %b %Y %H:%M:%S %z").astimezone(UTC)


def count_nominations(content: str) -> tuple[int, int]:
    """(nominations, withdrawals) in a "Nominations ... Sent to the Senate" post: one per paragraph that
    names a person "to be" an office, split at the WITHDRAWAL(S) header."""
    ps = [html.unescape(re.sub(r"<[^>]+>", "", p)).strip() for p in re.findall(r"<p[^>]*>(.*?)</p>", content, re.DOTALL)]
    nom = wd = 0
    in_wd = False
    for p in ps:
        if re.match(r"^WITHDRAWALS? SENT TO THE SENATE", p, re.IGNORECASE):
            in_wd = True
            continue
        if re.match(r"^NOMINATIONS? SENT TO THE SENATE", p, re.IGNORECASE):
            in_wd = False
            rest = re.sub(r"^NOMINATIONS? SENT TO THE SENATE:?", "", p, flags=re.IGNORECASE).strip()
            if not re.search(r"\bto be\b", rest):
                continue
            p = rest
        if re.search(r"\bto be\b", p) and not p.startswith("The post "):
            if in_wd:
                wd += 1
            else:
                nom += 1
    return nom, wd


def wh_items(xml: str, fetched_at: str) -> list[dict[str, Any]]:
    rows = []
    for it in re.findall(r"<item>(.*?)</item>", xml, re.DOTALL):
        def tag(name: str) -> str | None:
            m = re.search(rf"<{name}(?:\s[^>]*)?>(.*?)</{name}>", it, re.DOTALL)
            return html.unescape(m.group(1).strip()) if m else None
        pub = _rss_time(tag("pubDate") or "")
        cats = re.findall(r"<category><!\[CDATA\[(.*?)\]\]></category>", it)
        kind = next((k for c, k in KIND_ORDER if c in cats), "other")
        cm = re.search(r"<content:encoded><!\[CDATA\[(.*?)\]\]></content:encoded>", it, re.DOTALL)
        nom, wd = count_nominations(cm.group(1)) if (cm and kind == "nomination") else (0, 0)
        rows.append({"guid": tag("guid"), "title": tag("title"), "link": tag("link"), "pub_ts": pub,
                     "pub_date_et": pub.astimezone(ET).date(), "categories": "|".join(cats), "kind": kind,
                     "n_nom": nom, "n_withdraw": wd, "round_time": pub.minute == 0 and pub.second == 0,
                     "fetched_at": fetched_at})
    return rows


def fetch_wh(out: Path = OUT, *, tag: str | None = None, max_pages: int = 40, log=print) -> pl.DataFrame:
    """Every page of the Presidential Actions feed (30 items per page, newest first; a 404 ends it)."""
    from albert.http import HttpError

    cl = _web(WH_HOST)
    tag = tag or datetime.now(UTC).date().isoformat()
    rows: list[dict[str, Any]] = []
    for n in range(1, max_pages + 1):
        try:
            p = cl.get("/presidential-actions/feed/", {"paged": n} if n > 1 else {}, cache=True, tag=tag)
        except HttpError as error:
            if "-> 404" in str(error):
                break
            raise
        got = wh_items(p.payload if isinstance(p.payload, str) else "", p.fetched_at.isoformat())
        if not got:
            break
        rows.extend(got)
        log(f"  wh page {n}: {len(got)} items, oldest {got[-1]['pub_ts']}", flush=True)
    df = pl.DataFrame(rows, schema=WH_COLS).unique("guid", keep="first").sort("pub_ts")
    _save(df, out / "wh_actions.parquet")
    log(f"wh: {df.height} items {df['pub_ts'].min()} .. {df['pub_ts'].max()}")
    return df


# ---- Federal Register: executive orders ---------------------------------------------------------------
FR_COLS = {"document_number": pl.String, "eo_number": pl.Int64, "signing_date": pl.Date, "publication_date": pl.Date,
           "title": pl.String}


def fetch_fr(out: Path = OUT, since: date = date(2025, 1, 20), *, tag: str | None = None) -> pl.DataFrame:
    cl = _web("www.federalregister.gov", min_interval=1.0)
    tag = tag or datetime.now(UTC).date().isoformat()
    rows, page = [], 1
    while True:
        p = cl.get("/api/v1/documents.json", {
            "conditions[presidential_document_type][]": "executive_order",
            "conditions[signing_date][gte]": since.isoformat(), "per_page": 1000, "page": page, "order": "oldest",
            "fields[]": ["signing_date", "publication_date", "executive_order_number", "title", "document_number"]},
            cache=True, tag=tag)
        for d in p.payload.get("results") or []:
            try:
                rows.append({"document_number": d.get("document_number"),
                             "eo_number": int(d["executive_order_number"]) if d.get("executive_order_number") else None,
                             "signing_date": date.fromisoformat(d["signing_date"]),
                             "publication_date": date.fromisoformat(d["publication_date"]), "title": d.get("title")})
            except (KeyError, TypeError, ValueError):
                continue
        if page >= int(p.payload.get("total_pages") or 1):
            break
        page += 1
    df = pl.DataFrame(rows, schema=FR_COLS).unique("document_number").sort("signing_date")
    _save(df, out / "fr_eos.parquet")
    return df


# ---- Senate schedule (senate.gov, published before the year: 2026 "Updated on Nov 21, 2025") -----------
SEN_COLS = {"day": pl.Date, "no_session": pl.Boolean, "label": pl.String, "page_updated": pl.String}


def senate_rows(page: str, year: int) -> list[dict[str, Any]]:
    """Days listed in the schedule table (state work periods and holidays = no legislative session)."""
    upd = re.search(r"Updated on\s*(?:<[^>]+>\s*)*([A-Z][a-z]{2})(?:&nbsp;|\s)+(\d{1,2}), (\d{4})", page)
    updated = (date(int(upd.group(3)), _MON[upd.group(1)], int(upd.group(2))).isoformat() if upd else None)
    rows = []
    for tr in re.findall(r"<tr>(.*?)</tr>", page, re.DOTALL):
        tds = [html.unescape(re.sub(r"<[^>]+>", "", td)).replace("\xa0", " ").strip()
               for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)]
        if not tds:
            continue
        ds = re.findall(r"([A-Z][a-z]{2}) (\d{1,2})", tds[0])
        if not ds:
            continue
        d0 = date(year, _MON[ds[0][0]], int(ds[0][1]))
        d1 = date(year, _MON[ds[-1][0]], int(ds[-1][1]))
        label = " ".join(x for x in tds[1:] if x) or "listed"
        d = d0
        while d <= d1:
            rows.append({"day": d, "no_session": True, "label": label, "page_updated": updated})
            d += timedelta(days=1)
    return rows


def fetch_senate(out: Path = OUT, years: tuple[int, ...] = (2025, 2026)) -> pl.DataFrame:
    cl = _web("www.senate.gov")
    rows = []
    for y in years:
        p = cl.get(f"/legislative/{y}_schedule.htm", cache=True, tag=SNAP)
        rows.extend(senate_rows(p.payload if isinstance(p.payload, str) else "", y))
    df = pl.DataFrame(rows, schema=SEN_COLS).unique("day").sort("day")
    _save(df, out / "senate_schedule.parquet")
    return df


def recess_week(days: list[date], sched: set[date]) -> bool:
    """A week is a recess week when none of its weekdays is a scheduled session day."""
    wk = [d for d in days if d.weekday() < 5]
    return bool(wk) and all(d in sched for d in wk)


# ---- settlement rebuild --------------------------------------------------------------------------------
def _days(p0: date, p1: date) -> list[date]:
    return [p0 + timedelta(days=i) for i in range((p1 - p0).days + 1)]


def final_counts(mk: pl.DataFrame, wh: pl.DataFrame, fr: pl.DataFrame) -> pl.DataFrame:
    """Per market: the period, the final count rebuilt from the source as published today, and its label."""
    rows = []
    for r in mk.iter_rows(named=True):
        per = json.loads(r["period"]) if r["period"] and r["period"] != "null" else None
        if per is None:
            continue
        p0, p1 = date.fromisoformat(per[0]), date.fromisoformat(per[1])
        s = r["series_ticker"]
        in_p = (pl.col("pub_date_et") >= p0) & (pl.col("pub_date_et") <= p1)
        cnt = wh_eo = None
        if s in ("KXTRUMPACT", "KXTRUMPACTD"):
            cnt = float(wh.filter(in_p).height)
        elif s == "KXTRUMPNOMNUM":
            cnt = float(wh.filter(in_p & (pl.col("kind") == "nomination"))["n_nom"].sum())
        elif s in ("KXEOWEEK", "KXTRUMPEOD"):
            cnt = float(fr.filter((pl.col("signing_date") >= p0) & (pl.col("signing_date") <= p1)).height)
            wh_eo = float(wh.filter(in_p & (pl.col("kind") == "eo")).height)
        y = int(r["result"] == "yes")
        lab = strike_yes(cnt, r["strike_type"], r["floor_strike"], r["cap_strike"])
        rows.append({"ticker": r["ticker"], "event_ticker": r["event_ticker"], "series_ticker": s, "p0": p0, "p1": p1,
                     "split": "dev" if p1 <= DEV_END else "holdout", "y": y, "count_final": cnt, "wh_eo_final": wh_eo,
                     "label_rebuilt": lab, "agrees": None if lab is None else lab == y})
    return pl.DataFrame(rows, infer_schema_length=None)


def settle_check(out: Path = OUT) -> pl.DataFrame:
    fc = final_counts(pl.read_parquet(out / "markets.parquet"), pl.read_parquet(out / "wh_actions.parquet"),
                      pl.read_parquet(out / "fr_eos.parquet"))
    _save(fc, out / "final_counts.parquet")
    summ = (fc.filter(pl.col("agrees").is_not_null()).group_by("series_ticker", "split")
            .agg(pl.len().alias("markets"), pl.col("agrees").mean().alias("agree_rate")).sort("series_ticker", "split"))
    summ.write_csv(out / "settle_check.csv")
    print(summ)
    return fc


# ---- as-of nowcast from the White House feed ---------------------------------------------------------
N_ANALOG = 12                 # previous same-hour-of-week windows used for the remainder
EO_LAG_WEEKS = 3              # KXEOWEEK analogs: weeks that ended >= 14 days before t (FR complete)


def _et(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=ET).astimezone(UTC)


class Feed:
    """As-of sums over the WH feed: ``metric`` = all items, nominations (n_nom) or EO-category items."""

    def __init__(self, wh: pl.DataFrame, metric: str):
        if metric == "nom":
            w, val = wh.filter(pl.col("kind") == "nomination"), "n_nom"
        elif metric == "eo":
            w, val = wh.filter(pl.col("kind") == "eo"), None
        else:
            w, val = wh, None
        w = w.sort("pub_ts")
        self.ts = w["pub_ts"].dt.epoch("s").to_numpy().astype(float)
        v = w[val].to_numpy().astype(float) if val else np.ones(w.height)
        self.cum = np.concatenate([[0.0], np.cumsum(v)])

    def window(self, t0: float, t1: float) -> float:
        """Sum over items with t0 <= pub_ts < t1 (epoch seconds)."""
        i0, i1 = np.searchsorted(self.ts, t0, side="left"), np.searchsorted(self.ts, t1, side="left")
        return float(self.cum[i1] - self.cum[i0]) if i1 > i0 else 0.0


def session_weekdays(d0: date, d1: date, sched: set[date]) -> int:
    return sum(1 for d in _days(d0, d1) if d.weekday() < 5 and d not in sched)


def p_from_samples(k: float, rem: np.ndarray, lo: float, hi: float) -> float:
    """P(lo <= k + R <= hi) from analog remainder samples, shrunk toward 1/2 (Jeffreys)."""
    if len(rem) == 0:
        return float("nan")
    hit = ((k + rem >= lo) & (k + rem <= hi)).sum()
    return float((hit + 0.5) / (len(rem) + 1.0))


def nowcast(series: str, p0: date, p1: date, t: datetime, lo: float, hi: float, feeds: dict[str, Feed],
            fr: pl.DataFrame, sched: set[date], senate: bool = False) -> dict[str, Any]:
    """Partial count published by ``t`` plus the remainder from analog windows (the same offsets in
    earlier weeks, all published before ``t``). KXEOWEEK settles on FR signing dates: its remainder is
    D = FR count - WH EO items visible at the same offset, from weeks whose FR is complete."""
    S, E, ts = _et(p0).timestamp(), _et(p1 + timedelta(days=1)).timestamp(), t.timestamp()
    wk = 7 * 86400
    metric = {"KXTRUMPNOMNUM": "nom", "KXEOWEEK": "eo", "KXTRUMPEOD": "eo"}.get(series, "all")
    f = feeds[metric]
    k = f.window(S, min(ts, E) + 1e-6)
    if series in ("KXEOWEEK", "KXTRUMPEOD"):
        frc = fr.filter(pl.col("publication_date") < t.astimezone(ET).date())
        sd = frc["signing_date"].to_numpy()
        rem = []
        for w in range(EO_LAG_WEEKS, EO_LAG_WEEKS + N_ANALOG):
            a0, a1 = p0 - timedelta(days=7 * w), p1 - timedelta(days=7 * w)
            fr_w = float(((sd >= np.datetime64(a0)) & (sd <= np.datetime64(a1))).sum())
            rem.append(fr_w - f.window(S - w * wk, min(ts, E) - w * wk + 1e-6))
        rem = np.array(rem)
    else:
        rem, stat = [], []
        need_session = session_weekdays(max(p0, t.astimezone(ET).date()), p1, sched) > 0
        w = 1
        while len(rem) < N_ANALOG and w <= 52:
            if senate:
                a0 = max(p0, t.astimezone(ET).date()) - timedelta(days=7 * w)
                if (session_weekdays(a0, p1 - timedelta(days=7 * w), sched) > 0) != need_session:
                    w += 1
                    continue
            rem.append(f.window(max(ts, S) - w * wk, E - w * wk))
            stat.append(w)
            w += 1
        rem = np.array(rem)
    decided = (k >= lo and hi == math.inf) or k > hi
    if decided:
        p = 1.0 if hi == math.inf else 0.0
    else:
        p = p_from_samples(k, rem, lo, hi)
    return {"partial": k, "p_model": p, "decided": bool(decided), "rem_mean": float(rem.mean()) if len(rem) else None,
            "n_analog": len(rem)}


# ---- decision rows -------------------------------------------------------------------------------------
WEEK_PHASES = (("f10", 0.10), ("f25", 0.25), ("f50", 0.50), ("f75", 0.75), ("f90", 0.90))
DAY_HOURS_ET = (("h09", 9), ("h12", 12), ("h15", 15), ("h18", 18), ("h20", 20), ("h22", 22))
EO_POST_H = (("post06", 6), ("post30", 30), ("post78", 78), ("post150", 150))


def decision_times(series: str, p0: date, p1: date) -> list[tuple[str, datetime]]:
    S, E = _et(p0), _et(p1 + timedelta(days=1))
    out: list[tuple[str, datetime]] = []
    if p0 == p1:
        out = [(lab, datetime(p0.year, p0.month, p0.day, h, tzinfo=ET).astimezone(UTC)) for lab, h in DAY_HOURS_ET]
        out.append(("post08", datetime(p1.year, p1.month, p1.day, 8, tzinfo=ET).astimezone(UTC) + timedelta(days=1)))
        return out
    span = (E - S).total_seconds()
    for lab, fr_ in WEEK_PHASES:
        t = S + timedelta(seconds=span * fr_)
        out.append((lab, t.replace(minute=0, second=0, microsecond=0)))
    out.append(("final", E - timedelta(hours=4)))
    if series == "KXEOWEEK":
        out += [(lab, E + timedelta(hours=h)) for lab, h in EO_POST_H]
    return out


def _split(p1: date) -> str:
    return "dev" if p1 <= DEV_END else "holdout" if p1 < FORWARD_START else "forward"


def build(out: Path = OUT, *, log=print) -> pl.DataFrame:
    from albert.research.counts import _candle_arrays, quote_at, window_prints

    mk = pl.read_parquet(out / "markets.parquet")
    ser = pl.read_parquet(out / "series.parquet").select("series_ticker", "fee_type", "fee_multiplier")
    mk = mk.join(ser, on="series_ticker", how="left").filter(pl.col("series_ticker").is_in(list(WEEKLY + DAILY)))
    wh = pl.read_parquet(out / "wh_actions.parquet")
    fr = pl.read_parquet(out / "fr_eos.parquet")
    sched = set(pl.read_parquet(out / "senate_schedule.parquet")["day"].to_list())
    feeds = {m: Feed(wh, m) for m in ("all", "nom", "eo")}
    cand = load_candles(out)
    arrays = {tk: _candle_arrays(g) for (tk,), g in cand.group_by("ticker")}
    rows = []
    for r in mk.iter_rows(named=True):
        per = json.loads(r["period"]) if r["period"] and r["period"] != "null" else None
        if per is None:
            continue
        p0, p1 = date.fromisoformat(per[0]), date.fromisoformat(per[1])
        s = r["series_ticker"]
        lo, hi = need_for_yes(r["strike_type"], r["floor_strike"], r["cap_strike"])
        ca = arrays.get(r["ticker"])
        dts = decision_times(s, p0, p1)
        for i, (lab, t) in enumerate(dts):
            if r["open_time"] is None or not (r["open_time"] <= t < r["close_time"]):
                continue
            t_next = min(dts[i + 1][1] if i + 1 < len(dts) else r["close_time"], r["close_time"])
            q = quote_at(ca, t)
            wp = window_prints(ca, t, t_next)
            row = {"ticker": r["ticker"], "event_ticker": r["event_ticker"], "series_ticker": s, "p0": p0, "p1": p1,
                   "split": _split(p1), "decision": lab, "t": t, "t_next": t_next,
                   "date": t.astimezone(ET).date(), "lo": lo, "hi": hi, "strike_type": r["strike_type"],
                   "y": int(r["result"] == "yes"), "fee_type": r["fee_type"], "fee_multiplier": r["fee_multiplier"],
                   "settle_ts": r["settlement_ts"] or r["close_time"], **q, **wp}
            if s in WH_SERIES:
                nc = nowcast(s, p0, p1, t, lo, hi, feeds, fr, sched)
                row.update(nc)
                if s == "KXTRUMPNOMNUM":
                    row["p_senate"] = nowcast(s, p0, p1, t, lo, hi, feeds, fr, sched, senate=True)["p_model"]
            rows.append(row)
        if len(rows) % 2000 < 10:
            log(f"  build: {len(rows)} rows", flush=True)
    df = pl.DataFrame(rows, infer_schema_length=None)
    df = df.with_columns(
        pl.when(pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("ask") > pl.col("bid")))
        .then((pl.col("bid") + pl.col("ask")) / 2).otherwise(None).alias("mid"),
        pl.col("p1").map_elements(lambda d: recess_week(_days(d - timedelta(days=6), d), sched), return_dtype=pl.Boolean)
        .alias("recess_week"))
    _save(df, out / "rows.parquet")
    log(f"build: {df.height} rows, {df['event_ticker'].n_unique()} events")
    return df


# ---- trading rules ------------------------------------------------------------------------------------
STAKE_USD = 2.0               # ~$2 per order (fee rounding is per order) ...
MAX_CONTRACTS = 20            # ... at most 20 contracts
MAX_SPREAD = 0.15             # takers cross quotes at most this wide
EARLY = ("f10", "f25", "f50")
LATE = ("f75", "f90", "final")
POST = tuple(lab for lab, _ in EO_POST_H)
EVENING = ("h18", "h20", "h22")
PHASES = {"early": EARLY, "late": LATE, "post": POST, "evening": EVENING, "all": None}
GROUPS = {"act": ("KXTRUMPACT",), "nom": ("KXTRUMPNOMNUM",), "eo": ("KXEOWEEK",), "actd": ("KXTRUMPACTD",),
          "wh": ("KXTRUMPACT", "KXTRUMPNOMNUM", "KXEOWEEK", "KXTRUMPACTD"),
          "photo": ("KXTRUMPPHOTO",), "endorse": ("KXTRUMPENDORSEMENTS",), "mamdani": ("KXMAMDANIEO",),
          "daily": ("KXTRUMPACT", "KXTRUMPACTD"), "all": WEEKLY + ("KXTRUMPACTD",)}


def _n_for(price: float) -> int:
    return min(MAX_CONTRACTS, max(1, int(STAKE_USD // max(price, 0.01))))


def taker(rows: pl.DataFrame, sig: str, theta: float, sides: tuple[str, ...] = ("yes", "no")) -> pl.DataFrame:
    """Buy YES at the ask when sig - ask - fee >= theta, NO at 1 - bid when (1 - sig) - (1 - bid) - fee >=
    theta; ~$2 order, taker fee 0.07 x mult x n x P(1-P) rounded up to the cent per order; held to settlement."""
    from albert.research.counts import taker_fee

    r = rows.filter(pl.col(sig).is_not_null() & pl.col(sig).is_not_nan() & pl.col("bid").is_not_null()
                    & pl.col("ask").is_not_null() & (pl.col("bid") > 0) & (pl.col("ask") < 1)
                    & (pl.col("ask") - pl.col("bid") <= MAX_SPREAD + 1e-9))
    out = []
    for x in r.iter_rows(named=True):
        for side in sides:
            price = x["ask"] if side == "yes" else round(1 - x["bid"], 4)
            fair = x[sig] if side == "yes" else 1 - x[sig]
            n = _n_for(price)
            fee = taker_fee(n, price, x["fee_multiplier"])
            if fair - price - fee / n >= theta - 1e-9:
                pay = x["y"] if side == "yes" else 1 - x["y"]
                out.append({**x, "side": side, "price": price, "n": n, "fee": fee, "t_fill": x["t"],
                            "pnl_c": 100 * (pay - price - fee / n)})
    return pl.DataFrame(out, infer_schema_length=None) if out else pl.DataFrame()


def maker(rows: pl.DataFrame, sig: str, margin: float, sides: tuple[str, ...] = ("yes", "no"),
          band: tuple[float, float] | None = None) -> pl.DataFrame:
    """Rest a YES bid at floor(sig - m) and/or a NO bid at floor(1 - sig - m) from the decision to the next
    one (post-only: at most one tick inside the opposite quote). Filled only if a print in a later whole
    hour trades strictly through it (candle trade low < YES bid; high > 1 - NO bid); held to settlement.
    All series here are fee type ``quadratic`` (no maker fee). ``band`` keeps rows with mid in it."""
    from albert.research.counts import maker_fee

    r = rows.filter(pl.col(sig).is_not_null() & pl.col(sig).is_not_nan())
    if band is not None:
        r = r.filter(pl.col("mid").is_not_null() & pl.col("mid").is_between(*band))
    out = []
    for x in r.iter_rows(named=True):
        s = x[sig]
        if "yes" in sides:
            yb = math.floor(round((s - margin) * 100, 6)) / 100
            if x["ask"] is not None:
                yb = min(yb, round(x["ask"] - 0.01, 2))
            if yb >= 0.01 and x["w_lo"] is not None and x["w_lo"] < yb - 1e-9:
                fee = maker_fee(yb, x["fee_type"])
                out.append({**x, "side": "yes", "price": yb, "n": _n_for(yb), "fee": fee * _n_for(yb),
                            "pnl_c": 100 * (x["y"] - yb - fee)})
        if "no" in sides:
            nb = math.floor(round((1 - s - margin) * 100, 6)) / 100
            if x["bid"] is not None:
                nb = min(nb, round(1 - x["bid"] - 0.01, 2))
            if nb >= 0.01 and x["w_hi"] is not None and x["w_hi"] > 1 - nb + 1e-9:
                fee = maker_fee(nb, x["fee_type"])
                out.append({**x, "side": "no", "price": nb, "n": _n_for(nb), "fee": fee * _n_for(nb),
                            "pnl_c": 100 * (1 - x["y"] - nb - fee)})
    return pl.DataFrame(out, infer_schema_length=None) if out else pl.DataFrame()


def subset(rows: pl.DataFrame, group: str, phase: str) -> pl.DataFrame:
    r = rows.filter(pl.col("series_ticker").is_in(list(GROUPS[group])))
    ph = PHASES[phase]
    return r if ph is None else r.filter(pl.col("decision").is_in(list(ph)))


def summarize(f: pl.DataFrame) -> dict[str, Any]:
    from albert.research.counts import summarize as cs

    if not f.height:
        return {"n": 0, "events": 0, "mean_c": None, "t_event": None, "t_date": None, "t_2way": None,
                "win_pct": None, "pnl_per_usd": None}
    s = cs(f.with_columns(pl.col("date").cast(pl.String)))
    s["pnl_per_usd"] = round(float(f["pnl_c"].sum() / 100 / (f["price"].sum())), 3)
    return s


# The dev screen: (name, kind, signal, param, sides, band, groups, phases). Every cell is counted.
def cells() -> list[dict[str, Any]]:
    out = []
    for g in ("act", "nom", "eo", "wh"):
        for ph in ("early", "late", "all") + (("post",) if g in ("eo", "wh") else ()):
            for th in (0.03, 0.05, 0.08, 0.12):
                out.append({"hyp": "H1", "kind": "taker", "sig": "p_model", "param": th, "sides": "both", "band": None,
                            "group": g, "phase": ph})
            for m in (0.03, 0.05, 0.08, 0.12):
                out.append({"hyp": "H1", "kind": "maker", "sig": "p_model", "param": m, "sides": "both", "band": None,
                            "group": g, "phase": ph})
    for ph in ("early", "late", "all"):              # H3: Senate-conditioned nominations nowcast
        for th in (0.03, 0.05, 0.08, 0.12):
            out.append({"hyp": "H3", "kind": "taker", "sig": "p_senate", "param": th, "sides": "both", "band": None,
                        "group": "nom", "phase": ph})
        for m in (0.03, 0.05, 0.08, 0.12):
            out.append({"hyp": "H3", "kind": "maker", "sig": "p_senate", "param": m, "sides": "both", "band": None,
                        "group": "nom", "phase": ph})
    for th in (0.03, 0.05, 0.08, 0.12):               # H4: daily "any action" markets in the evening (ET)
        out.append({"hyp": "H4", "kind": "taker", "sig": "p_model", "param": th, "sides": "both", "band": None,
                    "group": "daily", "phase": "evening"})
    for m in (0.03, 0.05, 0.08, 0.12):
        out.append({"hyp": "H4", "kind": "maker", "sig": "p_model", "param": m, "sides": "both", "band": None,
                    "group": "daily", "phase": "evening"})
    for g in ("act", "nom", "eo", "photo", "endorse", "mamdani", "all"):   # H2 / H11: longshot NO bid at the mid
        for ph in ("early", "late", "all"):
            for m in (0.03, 0.05, 0.08, 0.12):
                out.append({"hyp": "H2" if g in ("act", "nom", "eo", "all") else "H11", "kind": "maker", "sig": "mid",
                            "param": m, "sides": "no", "band": (0.02, 0.40), "group": g, "phase": ph})
    return out


def run_cell(rows: pl.DataFrame, c: dict[str, Any]) -> pl.DataFrame:
    r = subset(rows, c["group"], c["phase"])
    sides = ("yes", "no") if c["sides"] == "both" else (c["sides"],)
    if c["kind"] == "taker":
        return taker(r, c["sig"], c["param"], sides)
    return maker(r, c["sig"], c["param"], sides, c["band"])


def cell_name(c: dict[str, Any]) -> str:
    b = "" if c["band"] is None else f"/mid{c['band'][0]:.2f}-{c['band'][1]:.2f}"
    return f"{c['hyp']}:{c['kind']}/{c['sig']}/{c['param']}/{c['sides']}{b}/{c['group']}/{c['phase']}"


def grid(rows: pl.DataFrame, split: str) -> pl.DataFrame:
    r = rows.filter(pl.col("split") == split)
    res = []
    for c in cells():
        f = run_cell(r, c)
        res.append({"cell": cell_name(c), **{k: (json.dumps(v) if k == "band" else v) for k, v in c.items()},
                    **summarize(f)})
    return pl.DataFrame(res, infer_schema_length=None)


def info_table(rows: pl.DataFrame) -> pl.DataFrame:
    """Undecided rows with a quote <= 15c wide and a mid in 5-95c: Brier of the mid vs the nowcast, and
    the slope of (y - mid) on (p - mid) (> 0: the price under-reacts to what the nowcast knows)."""
    from albert.research.counts import cluster_se

    r = rows.filter(pl.col("p_model").is_not_null() & pl.col("p_model").is_not_nan() & ~pl.col("decided")
                    & pl.col("mid").is_not_null() & (pl.col("ask") - pl.col("bid") <= MAX_SPREAD + 1e-9)
                    & pl.col("mid").is_between(0.05, 0.95))
    out = []
    for (s, sp, ph), g in r.with_columns(
            pl.when(pl.col("decision").is_in(list(EARLY))).then(pl.lit("early"))
            .when(pl.col("decision").is_in(list(LATE + EVENING))).then(pl.lit("late"))
            .when(pl.col("decision").is_in(list(POST + ("post08",)))).then(pl.lit("post")).otherwise(pl.lit("other"))
            .alias("ph")).group_by("series_ticker", "split", "ph"):
        y, mid, p = (g[c].to_numpy().astype(float) for c in ("y", "mid", "p_model"))
        x, z = p - mid, y - mid
        slope = float((x * z).sum() / (x * x).sum()) if (x * x).sum() > 0 else None
        t = None
        if slope is not None and g.height >= 30 and g["event_ticker"].n_unique() > 2:
            se = cluster_se(x * (z - slope * x) / max((x * x).mean(), 1e-12), g["event_ticker"].to_numpy().astype(str),
                            g["date"].cast(pl.String).to_numpy().astype(str))
            t = round(slope / se, 2) if se and se > 0 else None
        out.append({"series_ticker": s, "split": sp, "phase": ph, "rows": g.height, "events": g["event_ticker"].n_unique(),
                    "brier_mid": round(float(((y - mid) ** 2).mean()), 4), "brier_model": round(float(((y - p) ** 2).mean()), 4),
                    "slope": None if slope is None else round(slope, 3), "t_slope_2way": t})
    return pl.DataFrame(out, infer_schema_length=None).sort("series_ticker", "split", "phase")


def decided_table(rows: pl.DataFrame) -> pl.DataFrame:
    """Rows whose published partial already decides the leg: does the quote say so? (A bid far below
    $1 on a 'decided YES' row means the market had not seen it, or the item was backdated.)"""
    d = rows.filter(pl.col("decided") & pl.col("bid").is_not_null())
    d = d.with_columns(pl.when(pl.col("p_model") >= 0.5).then(pl.col("ask")).otherwise(1 - pl.col("bid")).alias("cost"),
                       pl.when(pl.col("p_model") >= 0.5).then(pl.col("y")).otherwise(1 - pl.col("y")).alias("wins"))
    return (d.group_by("series_ticker", "split").agg(
        pl.len().alias("rows"), pl.col("wins").mean().alias("label_agrees"), pl.col("cost").median().alias("median_cost"),
        (pl.col("cost") <= 0.97).sum().alias("cost_le_97c"), pl.col("quote_age_h").median().alias("median_quote_age_h"))
        .sort("series_ticker", "split"))


# ---- capped-book ledgers and risk metrics ---------------------------------------------------------------
EVENT_CAP = 5.0               # <= $5 cost per event
OPEN_CAP = 40.0               # <= $40 open on the $100 book


def ledger(fills: pl.DataFrame, kind: str, name: str, split: str, arrays: dict | None = None) -> pl.DataFrame:
    """Per-leg ledger of a $100 book: ~$2 per order, <= $5 per event, <= $40 open; orders that do not fit
    are shrunk or skipped. Maker size is capped by the fill window's traded volume; fills are dated at the
    end of the first hour that traded through."""
    from albert.research.counts import _fill_time, taker_fee

    if not fills.height:
        return pl.DataFrame()
    legs = []
    for x in fills.iter_rows(named=True):
        price, n = float(x["price"]), int(x["n"])
        if kind == "taker":
            t_in, fee_pc = x["t"], float(x["fee"]) / n
        else:
            t_in = _fill_time((arrays or {}).get(x["ticker"]), x["t"], x["t_next"], x["side"], price)
            if x.get("w_vol") and x["w_vol"] >= 1:
                n = min(n, int(x["w_vol"]))
            fee_pc = float(x["fee"]) / max(int(x["n"]), 1)
        pay = x["y"] if x["side"] == "yes" else 1 - x["y"]
        legs.append({"t_in": t_in, "n": n, "unit": price + fee_pc, "price": price, "payoff": pay,
                     **{k: x[k] for k in ("ticker", "event_ticker", "side", "settle_ts", "decision", "series_ticker",
                                          "fee_multiplier")}})
    legs.sort(key=lambda z: z["t_in"])
    open_pos: list[dict[str, Any]] = []
    ev_cost: dict[str, float] = {}
    rows = []
    for i, z in enumerate(legs):
        open_pos = [o for o in open_pos if o["settle_ts"] > z["t_in"]]
        room = min(OPEN_CAP - sum(o["cost"] for o in open_pos), EVENT_CAP - ev_cost.get(z["event_ticker"], 0.0))
        n = min(z["n"], int(room // z["unit"])) if room > 0 else 0
        if kind == "taker" and 1 <= n < z["n"]:            # a smaller taker order pays its own rounded fee
            fee = taker_fee(n, z["price"], z["fee_multiplier"]) / n
            n = min(n, int(room // (z["price"] + fee)))
            z = {**z, "unit": z["price"] + fee}
        if n < 1:
            continue
        cost = n * z["unit"]
        open_pos.append({"settle_ts": z["settle_ts"], "cost": cost})
        ev_cost[z["event_ticker"]] = ev_cost.get(z["event_ticker"], 0.0) + cost
        rows.append({"strategy": f"{NAME}_{kind}", "variant": name, "split": split, "struct_id": f"{z['ticker']}#{i}",
                     "event_ticker": z["event_ticker"], "ticker": z["ticker"], "side": z["side"], "n": n,
                     "t_in": z["t_in"], "cost_usd": cost, "settle_ts": z["settle_ts"],
                     "payoff_usd": n * float(z["payoff"]), "decision": z["decision"], "series_ticker": z["series_ticker"]})
    return pl.DataFrame(rows, infer_schema_length=None)


def candle_marks(out: Path = OUT) -> pl.DataFrame:
    c = load_candles(out)
    two = pl.col("bid_close").is_not_null() & pl.col("ask_close").is_not_null() & (pl.col("ask_close") > pl.col("bid_close"))
    mid = pl.when(two).then((pl.col("bid_close") + pl.col("ask_close")) / 2).otherwise(pl.col("p_close"))
    return (c.with_columns(mid.alias("yes_mid")).filter(pl.col("yes_mid").is_not_null())
            .select("ticker", pl.col("end_ts").alias("ts"), "yes_mid").sort("ticker", "ts"))


METRIC_KEYS = ("positions", "events", "pnl_usd", "usd_per_week", "ann_vol_pct", "sharpe", "sharpe_ci90", "p_sharpe_pos",
               "sharpe_ex_top5", "usd_per_week_ex_top5", "max_dd_usd", "weeks_positive_pct", "first_day", "last_day", "days")


def book_metrics(led: pl.DataFrame, marks: pl.DataFrame) -> dict[str, Any]:
    from albert.research import vol_metrics as vm

    if not led.height:
        return {}
    m = vm.ledger_metrics(led, marks.filter(pl.col("ticker").is_in(led["ticker"].unique().implode())))
    return {k: m.get(k) for k in METRIC_KEYS}


# ---- dev screen, freeze, holdout ---------------------------------------------------------------------
MIN_FILLS, MIN_EVENTS = 40, 10


def screen(out: Path = OUT, *, log=print) -> pl.DataFrame:
    rows = pl.read_parquet(out / "rows.parquet")
    g = grid(rows, "dev")
    g.write_csv(out / "grid_dev.csv")
    info = info_table(rows.filter(pl.col("split") == "dev"))
    info.write_csv(out / "info_dev.csv")
    decided_table(rows.filter(pl.col("split") == "dev")).write_csv(out / "decided_dev.csv")
    log(f"screen: {g.height} dev cells")
    ok = g.filter((pl.col("n") >= MIN_FILLS) & (pl.col("events") >= MIN_EVENTS) & pl.col("t_2way").is_not_null())
    log(ok.sort("t_2way", descending=True).select("cell", "n", "events", "mean_c", "pnl_per_usd", "t_event", "t_2way").head(25))
    return g


def freeze(out: Path = OUT, primary: str | None = None, secondaries: list[str] | None = None, note: str = "") -> dict[str, Any]:
    """Freeze the primary (default: highest dev two-way t among cells with >= 40 fills / 10 events and a
    positive mean) and up to two secondaries (default: the best such cell of two other hypotheses)."""
    if (out / "primary.json").exists():
        raise SystemExit("primary.json exists: the rule is frozen (delete it only to restart the study)")
    g = pl.read_csv(out / "grid_dev.csv")
    ok = g.filter((pl.col("n") >= MIN_FILLS) & (pl.col("events") >= MIN_EVENTS) & pl.col("t_2way").is_not_null()
                  & (pl.col("mean_c") > 0)).sort("t_2way", descending=True)
    prim = primary or (ok["cell"][0] if ok.height else None)
    if secondaries is None:
        hyp0 = prim.split(":")[0] if prim else None
        secondaries, seen = [], {hyp0}
        for r in ok.iter_rows(named=True):
            if r["hyp"] not in seen and len(secondaries) < 2:
                secondaries.append(r["cell"])
                seen.add(r["hyp"])
    spec = {c_name: next(c for c in cells() if cell_name(c) == c_name) for c_name in [prim, *secondaries] if c_name}
    res = {"primary": prim, "secondaries": secondaries, "cells": spec,
           "dev": {c: g.filter(pl.col("cell") == c).to_dicts()[0] for c in spec},
           "rule": "highest dev t_2way, >= 40 fills / 10 events, mean > 0", "note": note,
           "frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"),
           "dev_cells_tried": g.height}
    (out / "primary.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def holdout(out: Path = OUT, *, log=print) -> dict[str, Any]:
    from albert.research.counts import _candle_arrays

    fz = json.loads((out / "primary.json").read_text())
    rows = pl.read_parquet(out / "rows.parquet")
    cand = load_candles(out)
    arrays = {tk: _candle_arrays(g) for (tk,), g in cand.group_by("ticker")}
    marks = candle_marks(out)
    (out / "ledgers").mkdir(exist_ok=True)
    res: dict[str, Any] = {}
    for name, c in fz["cells"].items():
        c = {**c, "band": tuple(c["band"]) if c["band"] else None}
        r = {}
        for split in ("dev", "holdout", "forward"):
            if split == "forward" and not rows.filter(pl.col("split") == "forward").height:
                continue
            f = run_cell(rows.filter(pl.col("split") == split), c)
            led = ledger(f, c["kind"], name, split, arrays)
            if led.height:
                led.write_parquet(out / "ledgers" / f"ledger_{_slug(name)}_{split}.parquet")
            by_series = ({s: summarize(g) for (s,), g in f.group_by("series_ticker")} if f.height else {})
            r[split] = {**summarize(f), "book": book_metrics(led, marks), "by_series": by_series}
            log(f"{split} {name}: {r[split]['n']} fills, {r[split]['mean_c']}c, t2 {r[split]['t_2way']}, "
                f"Sharpe {r[split]['book'].get('sharpe')}")
        res[name] = r
    (out / "holdout.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:80]


# ---- forward recorder ------------------------------------------------------------------------------------
FWD = OUT / "forward"


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")


def signals(quotes: list[dict[str, Any]], items: list[dict[str, Any]], now: datetime, out: Path = OUT) -> list[dict[str, Any]]:
    """The as-of nowcast for every open WH-sourced market in a snapshot (stored feed history + the items
    just seen), with the frozen rule's would-be taker side. Paper only: nothing is sent anywhere."""
    if not (out / "wh_actions.parquet").exists():
        return []
    hist = pl.read_parquet(out / "wh_actions.parquet")
    new = pl.DataFrame(items, schema=WH_COLS) if items else hist.head(0)
    wh = pl.concat([hist, new.select(hist.columns)]).unique("guid", keep="first").filter(pl.col("pub_ts") <= now)
    fr = pl.read_parquet(out / "fr_eos.parquet")
    sched = set(pl.read_parquet(out / "senate_schedule.parquet")["day"].to_list()) if (out / "senate_schedule.parquet").exists() else set()
    feeds = {m: Feed(wh, m) for m in ("all", "nom", "eo")}
    fz = json.loads((out / "primary.json").read_text()) if (out / "primary.json").exists() else None
    theta = fz["cells"][fz["primary"]]["param"] if fz else None
    res = []
    for q in quotes:
        s = q["series_ticker"]
        if s not in ("KXTRUMPACT", "KXTRUMPACTD", "KXTRUMPNOMNUM", "KXEOWEEK"):
            continue
        lo, hi = need_for_yes(q["strike_type"], num(q["floor_strike"]), num(q["cap_strike"]))
        try:
            bid, ask = float(q["yes_bid_dollars"]), float(q["yes_ask_dollars"])
        except (TypeError, ValueError):
            bid = ask = None
        p0, p1 = _period_from_event(s, q["event_ticker"])
        if p0 is None:
            continue
        nc = nowcast(s, p0, p1, now, lo, hi, feeds, fr, sched)
        S, E = _et(p0), _et(p1 + timedelta(days=1))
        frac = (now - S).total_seconds() / (E - S).total_seconds()
        side = None
        if theta is not None and s == "KXTRUMPACT" and frac >= 0.75 and bid and ask and 0 < bid and ask < 1:
            from albert.research.counts import taker_fee
            p = nc["p_model"]
            if p - ask - taker_fee(_n_for(ask), ask, 1.0) / _n_for(ask) >= theta:
                side = "yes"
            elif (1 - p) - (1 - bid) - taker_fee(_n_for(1 - bid), 1 - bid, 1.0) / _n_for(1 - bid) >= theta:
                side = "no"
        res.append({"snapshot_utc": now.isoformat(timespec="seconds"), "ticker": q["ticker"], "series_ticker": s,
                    "p0": p0.isoformat(), "p1": p1.isoformat(), "period_frac": round(frac, 3), "partial": nc["partial"],
                    "p_model": nc["p_model"], "decided": nc["decided"], "yes_bid": bid, "yes_ask": ask,
                    "paper_side_primary": side})
    return res


def _period_from_event(series: str, event_ticker: str | None) -> tuple[date | None, date | None]:
    """Period of an open market from its event ticker: weekly events are named by their Sunday start
    (KXTRUMPACT/KXTRUMPNOMNUM ``-26NOV22``) or Saturday end (KXEOWEEK ``-26NOV28``); daily by the day."""
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", event_ticker or "")
    if not m:
        return None, None
    d = date(2000 + int(m.group(1)), _MON[m.group(2).title()], int(m.group(3)))
    if series in ("KXTRUMPACTD", "KXTRUMPEOD"):
        return d, d
    if series == "KXEOWEEK":
        return d - timedelta(days=6), d
    return d, d + timedelta(days=6)


def record(out: Path = FWD, *, log=print) -> dict[str, int]:
    """One forward snapshot (append-only JSONL under ``forward/<UTC day>/``): the WH Presidential Actions
    feed page 1 (every item with the time we first saw it: measures backdating and posting lag), the
    Federal Register public-inspection desk (presidential documents filed before publication), and the
    Kalshi quotes + displayed sizes of every open market of the slice series. Nothing is traded."""
    from albert.config import Settings
    from albert.http import CachedClient

    now = datetime.now(UTC)
    day = out / now.date().isoformat()
    snap = now.isoformat(timespec="seconds")
    n: dict[str, int] = {}
    wh = CachedClient(WH_HOST, f"https://{WH_HOST}", CACHE, min_interval=1.0, max_retries=3, user_agent=UA, record_live=False)
    items = wh_items(wh.get("/presidential-actions/feed/", cache=False).payload, snap)
    _append_jsonl(day / "wh_feed.jsonl", [{**i, "snapshot_utc": snap} for i in items])
    n["wh_items"] = len(items)
    fr = CachedClient("www.federalregister.gov", "https://www.federalregister.gov", CACHE, min_interval=1.0,
                      max_retries=3, user_agent=UA, record_live=False)
    pi = fr.get("/api/v1/public-inspection-documents/current.json", cache=False).payload
    docs = [{"snapshot_utc": snap, **{k: d.get(k) for k in ("document_number", "type", "subtype", "title", "filed_at",
                                                              "publication_date", "pdf_url")}}
            for d in (pi.get("results") or []) if (d.get("type") or "").startswith("Presidential")]
    _append_jsonl(day / "fr_public_inspection.jsonl", docs)
    n["fr_presidential_docs"] = len(docs)
    s = Settings()
    k = CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=KALSHI_MIN_INTERVAL, max_retries=6, record_live=False)
    q = []
    for t in WEEKLY + DAILY + ("KXBILLSSIGNED", "KXPARDONSTRUMP"):
        page = k.get("/markets", {"series_ticker": t, "status": "open", "limit": 1000, "mve_filter": "exclude"}, cache=False)
        for m in page.payload.get("markets") or []:
            q.append({"snapshot_utc": snap, "series_ticker": t, **{f: m.get(f) for f in (
                "ticker", "event_ticker", "yes_sub_title", "strike_type", "floor_strike", "cap_strike", "close_time",
                "yes_bid_dollars", "yes_ask_dollars", "yes_bid_size_fp", "yes_ask_size_fp", "last_price_dollars",
                "volume_fp", "volume_24h_fp", "open_interest_fp", "updated_time")}})
    _append_jsonl(day / "kalshi_quotes.jsonl", q)
    n["kalshi_markets"] = len(q)
    sig = signals(q, items, now)
    _append_jsonl(day / "signals.jsonl", sig)
    n["signals"] = len(sig)
    log(f"record {snap}: {n}")
    return n


# ---- Kalshi hourly candles (whole life) -------------------------------------------------------------
def fetch_candles(out: Path = OUT, *, log=print) -> pl.DataFrame:
    """Hourly OHLC (trade price, bid, ask) over each market's life (<= 4,900 h before the close):
    historical endpoint per market before the cutoff, live batch per event after. Sequential, 1 req/s."""
    from albert.research import xcat

    mk = pl.read_parquet(out / "markets.parquet").filter(pl.col("volume") > 0)
    lo = pl.max_horizontal(pl.col("open_time"), pl.col("close_time") - timedelta(hours=4900))
    sample = mk.select("ticker", "event_ticker", "settlement_ts", pl.lit(60).alias("period"), lo.alias("lo"),
                       pl.col("close_time").alias("hi"))
    d = out / "candles"
    d.mkdir(parents=True, exist_ok=True)
    done_path = d / "done.parquet"
    done = set(pl.read_parquet(done_path)["ticker"]) if done_path.exists() else set()
    todo = sample.filter(~pl.col("ticker").is_in(list(done)))
    api = _kalshi()
    cutoff = api.cutoff()["market_settled_ts"]
    jobs = xcat._jobs(todo, cutoff)
    log(f"candles: {sample.height} markets, {len(done)} done, {len(jobs)} requests", flush=True)
    part = len(list(d.glob("part-*.parquet")))
    buf: list[dict[str, Any]] = []
    fin: list[str] = []
    for i, job in enumerate(jobs):
        for tk, got in xcat._run_job(api, job):
            if got is not None:
                buf.extend(got)
                fin.append(tk)
        if (i + 1) % 100 == 0 or i + 1 == len(jobs):
            if buf:
                pl.DataFrame(buf, schema=xcat.OHLC_SCHEMA).write_parquet(d / f"part-{part:05d}.parquet")
                part += 1
            buf = []
            pl.DataFrame({"ticker": sorted(done | set(fin))}).write_parquet(done_path)
            log(f"  candles {i + 1}/{len(jobs)} requests", flush=True)
    return load_candles(out)


def load_candles(out: Path = OUT) -> pl.DataFrame:
    parts = sorted((out / "candles").glob("part-*.parquet"))
    if not parts:
        return pl.DataFrame(schema={"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC")})
    return pl.concat([pl.read_parquet(p) for p in parts]).unique(["ticker", "end_ts"]).sort("ticker", "end_ts")


# ---- latency: White House post -> Kalshi repricing (trade prints) --------------------------------------
LAT_WINDOW = timedelta(hours=6)


def decided_by_item(mk: pl.DataFrame, feeds: dict[str, Feed], wh: pl.DataFrame) -> pl.DataFrame:
    """(item, market) pairs where the item moves the published count into the market's YES set."""
    rows = []
    for it in wh.filter(pl.col("pub_ts") >= datetime(2025, 9, 1, tzinfo=UTC)).iter_rows(named=True):
        d, ts = it["pub_date_et"], it["pub_ts"].timestamp()
        for r in mk.iter_rows(named=True):
            if not (r["p0"] <= d <= r["p1"]):
                continue
            s = r["series_ticker"]
            if s == "KXTRUMPNOMNUM" and it["kind"] != "nomination":
                continue
            if s == "KXEOWEEK" and it["kind"] != "eo":
                continue
            metric = {"KXTRUMPNOMNUM": "nom", "KXEOWEEK": "eo"}.get(s, "all")
            S = _et(r["p0"]).timestamp()
            before, after = feeds[metric].window(S, ts), feeds[metric].window(S, ts + 1)
            lo, hi = r["lo"], r["hi"]
            if not (lo <= before <= hi) and lo <= after <= hi and hi == math.inf:
                rows.append({"guid": it["guid"], "kind": it["kind"], "pub_ts": it["pub_ts"], "round_time": it["round_time"],
                             "ticker": r["ticker"], "event_ticker": r["event_ticker"], "series_ticker": s,
                             "split": "dev" if r["p1"] <= DEV_END else "holdout", "y": r["y"]})
    return pl.DataFrame(rows, infer_schema_length=None)


def latency(out: Path = OUT, *, log=print) -> pl.DataFrame:
    """For every (WH item, market it decides): trade prints from 6 h before to 6 h after the post (cached
    per window); last print before the post, first print >= 90c, and the lag (minutes; < 0 = the market
    had it before the post, e.g. a televised signing)."""
    from albert.venues import kalshi as k

    mk = pl.read_parquet(out / "markets.parquet").filter(pl.col("series_ticker").is_in(list(WH_SERIES)))
    per = [json.loads(p) if p and p != "null" else None for p in mk["period"].to_list()]
    mk = mk.with_columns(pl.Series("p0", [date.fromisoformat(p[0]) if p else None for p in per]),
                         pl.Series("p1", [date.fromisoformat(p[1]) if p else None for p in per]),
                         (pl.col("result") == "yes").cast(pl.Int8).alias("y")).filter(pl.col("p0").is_not_null())
    lohi = [need_for_yes(a, b, c) for a, b, c in zip(mk["strike_type"], mk["floor_strike"], mk["cap_strike"], strict=True)]
    mk = mk.with_columns(pl.Series("lo", [x[0] for x in lohi]), pl.Series("hi", [x[1] for x in lohi]))
    wh = pl.read_parquet(out / "wh_actions.parquet")
    feeds = {m: Feed(wh, m) for m in ("all", "nom", "eo")}
    pairs = decided_by_item(mk, feeds, wh)
    api = _kalshi()
    cutoff = api.cutoff()["market_settled_ts"]
    sett = dict(zip(mk["ticker"], mk["settlement_ts"], strict=True))
    res = []
    for i, p in enumerate(pairs.iter_rows(named=True)):
        t0, t1 = p["pub_ts"] - LAT_WINDOW, p["pub_ts"] + LAT_WINDOW
        hist = bool(cutoff and sett.get(p["ticker"]) and sett[p["ticker"]] < cutoff)
        tr = []
        try:
            for page in api.trades(p["ticker"], hist, min_ts=t0, max_ts=t1):
                tr.extend(k.trade_rows(page.payload))
        except RuntimeError as error:
            log(f"  skip {p['ticker']}: {str(error)[:120]}")
            continue
        tr = sorted(tr, key=lambda x: x["created_time"])
        pre = [x for x in tr if x["created_time"] < p["pub_ts"]]
        hit = [x for x in tr if x["yes_price"] is not None and x["yes_price"] >= 0.90]
        first90 = hit[0]["created_time"] if hit else None
        post = [x for x in tr if x["created_time"] >= p["pub_ts"]]
        res.append({**p, "n_trades": len(tr), "pre_price": pre[-1]["yes_price"] if pre else None,
                    "pre_time": pre[-1]["created_time"] if pre else None, "first90": first90,
                    "lag_min": (first90 - p["pub_ts"]).total_seconds() / 60 if first90 else None,
                    "first_post_price": post[0]["yes_price"] if post else None,
                    "first_post_lag_min": (post[0]["created_time"] - p["pub_ts"]).total_seconds() / 60 if post else None})
        if (i + 1) % 25 == 0:
            log(f"  latency {i + 1}/{pairs.height}", flush=True)
    df = pl.DataFrame(res, infer_schema_length=None)
    _save(df, out / "latency.parquet")
    log(f"latency: {df.height} item-market pairs")
    return df


# ---- descriptive dev checks (H2 calibration, H6 holidays, H7 anchoring, H10 ladder consistency) -----------
HOLIDAY_WEEKS = {date(2025, 11, 23), date(2025, 12, 21), date(2025, 12, 28), date(2026, 5, 24), date(2026, 6, 28),
                 date(2026, 8, 30), date(2026, 11, 22), date(2026, 12, 20), date(2026, 12, 27)}   # Sunday starts, fixed in advance


def _ph(dec: pl.Expr) -> pl.Expr:
    return (pl.when(dec.is_in(list(EARLY))).then(pl.lit("early")).when(dec.is_in(list(LATE + EVENING))).then(pl.lit("late"))
            .when(dec.is_in(list(POST + ("post08",)))).then(pl.lit("post")).otherwise(pl.lit("day")))


def explore(out: Path = OUT, split: str = "dev") -> dict[str, pl.DataFrame]:
    from albert.research.counts import cluster_se

    rows = pl.read_parquet(out / "rows.parquet").filter(pl.col("split") == split)
    q = rows.filter(pl.col("mid").is_not_null() & (pl.col("ask") - pl.col("bid") <= MAX_SPREAD + 1e-9)).with_columns(
        _ph(pl.col("decision")).alias("phase"),
        pl.col("mid").cut([0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]).alias("bucket"))
    res: dict[str, pl.DataFrame] = {}

    def tstat(g: pl.DataFrame) -> float | None:
        x = (g["y"] - g["mid"]).to_numpy().astype(float)
        if len(x) < 10 or g["event_ticker"].n_unique() < 3:
            return None
        se = cluster_se(x, g["event_ticker"].to_numpy().astype(str))
        return round(float(x.mean()) / se, 2) if se > 0 else None
    cal = []
    for (s, ph, b), g in q.group_by("series_ticker", "phase", "bucket"):
        cal.append({"series_ticker": s, "phase": ph, "bucket": str(b), "rows": g.height, "events": g["event_ticker"].n_unique(),
                    "mean_mid": round(float(g["mid"].mean()), 3), "yes_rate": round(float(g["y"].mean()), 3),
                    "y_minus_mid_c": round(100 * float((g["y"] - g["mid"]).mean()), 2), "t_event": tstat(g)})
    res["calibration"] = pl.DataFrame(cal).sort("series_ticker", "phase", "bucket")
    # H6: holiday weeks (fixed list) vs other weeks, first decisions, legs priced 5-95c
    wk = q.filter((pl.col("phase") == "early") & pl.col("mid").is_between(0.05, 0.95)).with_columns(
        pl.col("p0").is_in(list(HOLIDAY_WEEKS)).alias("holiday"))
    res["holiday"] = pl.DataFrame([{"holiday": h, "rows": g.height, "events": g["event_ticker"].n_unique(),
                                    "y_minus_mid_c": round(100 * float((g["y"] - g["mid"]).mean()), 2), "t_event": tstat(g)}
                                   for (h,), g in wk.group_by("holiday")])
    # H7: last week's final count vs its 12-week mean (both public before the week), early legs 5-95c
    fc = pl.read_parquet(out / "final_counts.parquet").group_by("series_ticker", "p0").agg(pl.col("count_final").first())
    an = []
    for s in ("KXTRUMPACT", "KXEOWEEK", "KXTRUMPNOMNUM"):
        h = fc.filter(pl.col("series_ticker") == s).sort("p0")
        cnt = dict(zip(h["p0"].to_list(), h["count_final"].to_list(), strict=True))
        g = wk.filter(pl.col("series_ticker") == s)
        dev_ = []
        for p0 in g["p0"].to_list():
            prev = [cnt.get(p0 - timedelta(days=7 * w)) for w in range(1, 13)]
            prev = [v for v in prev if v is not None]
            dev_.append(None if len(prev) < 4 else float(prev[0] - np.mean(prev[1:])) if prev[0] is not None else None)
        g = g.with_columns(pl.Series("last_minus_mean", dev_, dtype=pl.Float64)).filter(pl.col("last_minus_mean").is_not_null())
        for lab, sub in (("last week above its mean", g.filter(pl.col("last_minus_mean") > 0)),
                         ("last week at/below its mean", g.filter(pl.col("last_minus_mean") <= 0))):
            an.append({"series_ticker": s, "state": lab, "rows": sub.height, "events": sub["event_ticker"].n_unique(),
                       "y_minus_mid_c": round(100 * float((sub["y"] - sub["mid"]).mean()), 2) if sub.height else None,
                       "t_event": tstat(sub) if sub.height else None})
    res["anchoring"] = pl.DataFrame(an)
    # H10: ladder monotonicity on simultaneous hourly quotes (">= k" legs of one event)
    c = load_candles(out).select("ticker", "end_ts", "bid_close", "ask_close")
    mk = rows.select("ticker", "event_ticker", "series_ticker", "lo", "hi").unique("ticker").filter(pl.col("hi").is_infinite())
    c = c.join(mk, on="ticker").filter(pl.col("bid_close").is_not_null() & pl.col("ask_close").is_not_null())
    v = (c.join(c, on=["event_ticker", "end_ts"], suffix="_h").filter(pl.col("lo_h") > pl.col("lo"))
         .with_columns((pl.col("bid_close_h") - pl.col("ask_close")).alias("gap")))
    res["ladder"] = (v.group_by("series_ticker").agg(pl.len().alias("pair_hours"), (pl.col("gap") > 0.02).sum().alias("gap_gt_2c"),
                                                     pl.col("gap").max().alias("max_gap")).sort("series_ticker"))
    for k, df in res.items():
        df.write_csv(out / f"explore_{k}_{split}.csv")
    return res


# ---- report ----------------------------------------------------------------------------------------------
def _md(df: pl.DataFrame, cols: list[str] | None = None, nd: int = 3) -> str:
    if df is None or not df.height:
        return "(none)\n"
    cols = [c for c in (cols or df.columns) if c in df.columns]

    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.{nd}g}" if abs(v) < 1e5 else f"{v:.3e}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.select(cols).iter_rows()]
    return "\n".join(lines) + "\n"


def _read(path: Path) -> Any:
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=None)
    return pl.read_parquet(path)


def write_report(out: Path = OUT, verdict: str = "", extra: str = "") -> Path:
    """``report.md`` from the saved artefacts; ``verdict`` / ``extra`` are the written conclusions."""
    head = (f"# Presidential-action markets (executive orders, actions, nominations): exec_actions\n\n"
            f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `python -m albert.research.{NAME} report` "
            f"(module `src/albert/research/{NAME}.py`, tests `tests/test_{NAME}.py`).\n\n## Verdict\n\n{verdict}\n")
    parts = [head]
    hyp = out / "hypotheses.md"
    if hyp.exists():
        parts.append("## 1. Hypotheses (written before any outcome was opened)\n\n"
                     + hyp.read_text().split("\n", 1)[1].strip() + "\n")
    parts.append(extra)
    (out / "report.md").write_text("\n".join(parts))
    return out / "report.md"


VERDICT = """**No edge (verdict `no_edge`).** The White House "Presidential Actions" feed is a clean, robots-allowed,
time-stamped source. Rebuilt as of each decision, it reproduces every KXTRUMPACT / KXTRUMPACTD /
KXTRUMPNOMNUM result (448/448), and the Federal Register reproduces KXEOWEEK (125/125). On dev (Mar-Sep 2026)
the as-of count plus a 12-week analog remainder beat the KXTRUMPACT price late in the week. The frozen
primary (taker, 5c threshold, Fri 06:00 / Sat 07:00 / Sat 20:00 ET) made **+20.0c per contract (t 4.5, 71
fills, 23 weeks)**. On the holdout (7 weeks, Oct 3 - Nov 14) it made **+1.4c (t 0.16, 16 fills)**, and the
$100 book lost $5.35 (Sharpe -2.4, CI [-7.3, 2.2]). Both secondaries also went flat or negative in the
holdout. The dev book was lottery-shaped too: Sharpe 3.5 but 0.47 without the top-5 events, and 46%
positive weeks.

Why it failed: White House activity fell in Oct-Nov 2026 (13 items in October, 9 in November, against
16-30 a month earlier in 2026). The 12-week analog then over-forecast the rest of the week, and every holdout loss
is a YES buy. In dev the market had been the one lagging (the NO side made the money). Late in the
holdout, the mid beat the model (Brier 0.012 vs 0.015).

Nothing else survives. Post-week KXEOWEEK (FR vs WH) and EO nowcasts lose. Senate-recess nominations
(H3) and the evening decay of daily markets (H4) have 4-8 dev events. Resting NO bids at the mid (H2 / H11,
including the price-only Getty and Truth Social ladders) are flat. Ladders are consistent (H10). On
latency (H8), the market reprices a decided strike 1-16 s after the White House post, and EO and action
ladders are often priced before the post (televised signings).

The only lead is post hoc: the NO side alone of the primary (dev +19.9c, t 4.0; holdout +15.6c on 7
fills / 4 weeks). It is too thin and too lumpy for paper status (dev book Sharpe -1.6 without the top 5).
The forward recorder (`record`) logs everything needed to test it, and the frozen primary, on weeks after
2026-11-20, with no further work."""
FINDINGS = """## 9. Forward recorder (built and smoked, not scheduled)

    cd /home/ubuntu/albert && .venv/bin/python -m albert.research.exec_actions record

One snapshot per run, appended to `data/research/exec_actions/forward/<UTC day>/`:
* `wh_feed.jsonl`: page 1 of the Presidential Actions feed with the time we saw each item (measures
  posting lag / backdating going forward);
* `fr_public_inspection.jsonl`: presidential documents on the FR public-inspection desk;
* `kalshi_quotes.jsonl`: bid / ask / displayed size / volume of every open market of the slice series;
* `signals.jsonl`: the as-of nowcast for every open KXTRUMPACT / KXTRUMPACTD / KXTRUMPNOMNUM / KXEOWEEK leg
  and the frozen primary's would-be taker side (paper only).

Suggested cadence: every 15 minutes (13 light requests, about 15 s and about 60 KB per snapshot, about
6 MB a day). Hourly is enough for the frozen rule's decision times; 15 minutes also times the posts.
Smoke 2026-11-20 23:24 UTC: 30 feed items, 73 Kalshi markets, 39 signals, 0 public-inspection
documents (none on the desk at the time). To score forward weeks later:
`discover --snap <date>`, `wh`, `fr`, `candles`, `build`, `holdout`. Periods ending on or after 2026-11-21
form a separate `forward` split, scored with the frozen rules unchanged.

## 10. Caveats

* The holdout has 7 KXTRUMPACT weeks: it cannot reject a 5-10c edge; it only shows the dev effect did not
  carry over.
* Execution realism: decisions use hourly candle closes, not the live book. The recorder's top-of-book sizes
  (2026-11-20) are tens to thousands of contracts on KXTRUMPACT, KXEOWEEK, NOMNUM, PHOTO and ENDORSEMENTS.
  KXTRUMPACTD and KXTRUMPEOD are thin and wide.
* The analog remainder model is crude, and its failure mode is regime change in the posting rate.
* Feed history: an item removed from whitehouse.gov would be missing today. The 100% settlement match says
  it did not matter here.
* KXMAMDANIEO is price-only: nyc.gov allows us, but its EO list is rendered client-side.

## 11. What would change the verdict

1. About 12 forward KXTRUMPACT weeks from the recorder in which the NO-only late-week rule (or the frozen
   primary) holds: per-contract t >= 2 clustered by week, >= 60% positive weeks, book Sharpe without the
   top 5 > 1.
2. A remainder model that tracks the posting rate faster than a 12-week window, with calendar inputs fixed
   in advance (Senate schedule, holidays, presidential travel announced ahead). It must be tested on
   forward weeks, because the holdout has been seen.
3. A public, time-stamped schedule of signings (whitehouse.gov posts none, and Factba.se / Roll Call block
   us). It would move the EO and action nowcasts ahead of the post, which is where those markets already
   are.
4. A static or JSON list of NYC executive orders with signing and posting dates, for KXMAMDANIEO.
"""


def report_body(out: Path = OUT) -> str:
    """Sections 2-9 of the report from the saved artefacts."""
    s = []
    mk = _read(out / "markets.parquet")
    if mk is not None:
        per = [json.loads(p)[1] if p and p != "null" else None for p in mk["period"].to_list()]
        mk = mk.with_columns(pl.Series("p1", per).str.to_date())
        t = (mk.with_columns(pl.when(pl.col("p1") <= DEV_END).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("split"))
             .group_by("series_ticker").agg(
                 (pl.col("split") == "dev").sum().alias("markets_dev"), (pl.col("split") == "holdout").sum().alias("markets_holdout"),
                 pl.col("event_ticker").filter(pl.col("split") == "dev").n_unique().alias("events_dev"),
                 pl.col("event_ticker").filter(pl.col("split") == "holdout").n_unique().alias("events_holdout"),
                 pl.col("volume").sum().round(0).alias("volume_contracts"), pl.col("p1").min().alias("first_period_end"),
                 pl.col("p1").max().alias("last_period_end")).sort("volume_contracts", descending=True))
        src = _read(out / "series.parquet")
        if src is not None:
            t = t.join(src.select("series_ticker", "frequency", "source"), on="series_ticker", how="left")
        s.append("## 2. Series, settlement rules and data\n\n" + FINDINGS_RULES + "\n" + _md(t) + "\n")
    rb = _read(out / "robots.json")
    if rb is not None:
        s.append("## 3. Sources and robots.txt (checked " + rb[0]["checked_utc"] + ")\n\n"
                 + _md(pl.DataFrame(rb), ["host", "path", "http_status", "allow_*", "allow_anthropic-ai", "allow_ClaudeBot",
                                          "allow_Claude-Web", "allowed"]) + "\n" + SOURCES_NOTE + "\n")
    sc = _read(out / "settle_check.csv")
    if sc is not None:
        s.append("## 4. Settlement rebuilt from the sources (as published 2026-11-20)\n\n" + _md(sc) + "\n" + SETTLE_NOTE + "\n")
    lat = _read(out / "latency.parquet")
    if lat is not None and lat.height:
        lat = lat.with_columns(
            pl.when(pl.col("pre_price") >= 0.9).then(pl.lit("a: >= 90c before the post"))
            .when(pl.col("lag_min").is_null()).then(pl.lit("e: not >= 90c within 6 h"))
            .when(pl.col("lag_min") <= 5).then(pl.lit("b: <= 5 min after")).when(pl.col("lag_min") <= 30)
            .then(pl.lit("c: 5-30 min after")).otherwise(pl.lit("d: > 30 min after")).alias("first_print_ge_90c"))
        t1 = lat.group_by("series_ticker", "first_print_ge_90c").agg(pl.len().alias("pairs")).sort("series_ticker", "first_print_ge_90c")
        t2 = (lat.filter(pl.col("n_trades") > 0).group_by("series_ticker").agg(
            pl.len().alias("pairs"), pl.col("pre_price").median().alias("median_last_print_before"),
            pl.col("first_post_price").median().alias("median_first_print_after"),
            (pl.col("first_post_lag_min") * 60).median().alias("median_s_to_first_print_after"),
            pl.col("lag_min").median().alias("median_min_to_90c")).sort("series_ticker"))
        s.append("## 5. As-of check and latency: White House post -> Kalshi prints (H8)\n\n"
                 "For each feed item and each market it moves into YES (the strike its count crosses), the trade "
                 "prints from 6 h before to 6 h after `pubDate` (`latency.parquet`).\n\n" + _md(t1) + "\n" + _md(t2)
                 + "\n" + LATENCY_NOTE + "\n")
    g = _read(out / "grid_dev.csv")
    if g is not None:
        top = (g.filter(pl.col("n") >= 15).sort("t_2way", descending=True, nulls_last=True)
               .select("cell", "n", "events", "mean_c", "pnl_per_usd", "t_event", "t_date", "t_2way", "win_pct").head(25))
        byh = g.group_by("hyp").agg(pl.len().alias("cells"), (pl.col("n") >= MIN_FILLS).sum().alias("cells_ge_40_fills"),
                                    ((pl.col("t_2way") >= 2) & (pl.col("n") >= MIN_FILLS) & (pl.col("events") >= MIN_EVENTS))
                                    .sum().alias("cells_t_ge_2_eligible"),
                                    pl.col("t_2way").max().alias("best_t")).sort("hyp")
        parts6 = ["## 6. Dev screen (periods ending <= 2026-09-30)\n\n" + SCREEN_NOTE + "\n", _md(byh), "\nTop dev cells "
                  "(>= 15 fills; full grid in `grid_dev.csv`):\n\n" + _md(top)]
        for name, title in (("info_dev.csv", "Does the nowcast add information to the mid? (undecided rows, mid 5-95c)"),
                            ("decided_dev.csv", "Rows already decided by the published count"),
                            ("explore_calibration_dev.csv", "Calibration of the mid (H2 / H11), buckets with >= 15 rows"),
                            ("explore_holiday_dev.csv", "H6: holiday weeks, early legs 5-95c"),
                            ("explore_anchoring_dev.csv", "H7: last week's count vs its 12-week mean, early legs 5-95c"),
                            ("explore_ladder_dev.csv", "H10: ladder monotonicity, simultaneous hourly quotes (higher strike bid - lower strike ask)")):
            d = _read(out / name)
            if d is not None:
                if "bucket" in d.columns:
                    d = d.filter(pl.col("rows") >= 15)
                parts6.append(f"\n{title}:\n\n" + _md(d))
        s.append("\n".join(parts6) + "\n" + DEV_NOTE + "\n")
    fz = _read(out / "primary.json")
    if fz is not None:
        s.append("## 7. Frozen rule (`primary.json`, frozen " + fz["frozen_utc"] + ", before any holdout number)\n\n"
                 f"Primary: `{fz['primary']}`; secondaries: " + ", ".join(f"`{c}`" for c in fz["secondaries"])
                 + f". Dev cells tried: {fz['dev_cells_tried']} in the grid (+ the dev robustness splits and descriptive "
                 "tables above).\n\n" + fz["note"] + "\n\n" + RULE_NOTE + "\n")
    ho = _read(out / "holdout.json")
    if ho is not None:
        rows = []
        for name, r in ho.items():
            for split in ("dev", "holdout"):
                x = r[split]
                b = x.get("book") or {}
                rows.append({"rule": ("primary " if name == fz["primary"] else "secondary ") + name, "split": split,
                             "fills": x["n"], "events": x["events"], "c_per_contract": x["mean_c"],
                             "pnl_per_usd": x["pnl_per_usd"], "t_event": x["t_event"], "t_2way": x["t_2way"],
                             "win_pct": x["win_pct"], "book_positions": b.get("positions"), "usd_per_week": b.get("usd_per_week"),
                             "sharpe": b.get("sharpe"), "sharpe_ci90": str(b.get("sharpe_ci90")),
                             "sharpe_ex_top5": b.get("sharpe_ex_top5"), "ann_vol_pct": b.get("ann_vol_pct"),
                             "max_dd_usd": b.get("max_dd_usd"), "weeks_positive_pct": b.get("weeks_positive_pct"),
                             "days": b.get("days")})
        s.append("## 8. Holdout (periods ending 2026-10-01 .. 2026-11-14) and the book\n\n"
                 "Per contract after the taker fee; t clustered by event (weekly events do not overlap within a "
                 "series) and two-way by event and date. Book: $100, ~$2 per order, <= $5 per event, <= $40 open, "
                 "daily MTM at the hourly-candle mid (`albert.research.vol_metrics.ledger_metrics`; Sharpe x sqrt(365), "
                 "90% block-bootstrap CI, Sharpe without the top-5 events).\n\n" + _md(pl.DataFrame(rows, infer_schema_length=None))
                 + "\n" + HOLDOUT_NOTE + "\n")
    s.append(FINDINGS)
    (out / "report_body.md").write_text("\n".join(s))
    return "\n".join(s)


LATENCY_NOTE = """Reading: the WH post *is* the news for nominations and the daily "any action" market. The first
print comes a median 1-3 s after `pubDate` (EOs: 16 s), and 18 of 20 nomination strikes reach 90c within 5 minutes.
For EOs and the action ladder, 35 of 59 and 85 of 182 decided strikes already traded at >= 90c before
the post (televised signings, scheduled proclamations). A poller at our latency gets nothing (H8 rejected),
and the stamp is a sound as-of time: markets move right after it, not hours later. Exactly-on-the-hour
stamps (round_time, 2% of items) were also followed within 0.1-2 minutes; one (2026-10-08 01:00:00 UTC)
was never priced >= 90c within 6 h and may be backdated."""
SCREEN_NOTE = """Decision times: weekly 10/25/50/75/90% of the Sun-Sat week (ET, on the hour) and 4 h before its end;
KXEOWEEK also 6/30/78/150 h after the week (it trades until the FR publishes); daily markets at 09, 12,
15, 18, 20 and 22 ET and 08 ET next day. Quote = the last hourly candle that closed at or before the
decision (median age 0 h). Nowcast = the published count at `t` (WH `pubDate` <= t) plus the remainder
from the same window in the 12 previous weeks (KXEOWEEK: FR count minus the visible WH EO count, from weeks
whose FR was complete 3-14 weeks back; H3 restricts the analog weeks to the same Senate status). The
probability is shrunk toward 1/2: (hits + 0.5) / (12 + 1). Taker: ~$2 orders at the ask / 1 - bid, spread
<= 15c, fee rounded up per order, threshold includes the fee. Maker: bid at floor(sig -/+ m), filled only
by a later whole hour that traded strictly through, held to settlement (no maker fee on these series).
H2 / H11: a NO bid at floor(1 - mid - m) on legs with mid 2-40c. Deviation from `hypotheses.md`: H1 says
"weekday-rate Poisson remainder"; the screen used the empirical 12-week analog instead (decided while
writing the code, before any outcome or price was scored), because nomination counts arrive in batches
that a Poisson model cannot fit."""
DEV_NOTE = """Dev reading. The KXTRUMPACT nowcast adds information to the mid: slope 0.77 early (t 2.6) and late
(t 4.0), and a lower Brier late (0.078 vs 0.101). The nominations and EO nowcasts do not (EO late slope
-0.21, t -2.3). Late-week action legs priced 30-50c settle 15% of the time and 5-15c legs 1.5%: a
YES premium on high strikes. The mid-based NO bid (H2) rarely fills, so those cells are thin. H6: early
legs in the holiday weeks listed in advance (Thanksgiving, Christmas, New Year, Memorial Day, July 4,
Labor Day; 15 events across series in dev) were 17c rich (t -2.65), descriptive only.
H7 (anchoring on last week) is null; H10 finds 9 of 104k ladder pair-hours more than 2c inconsistent.
H9 had no as-of source (the nyc.gov list is client-rendered), so KXMAMDANIEO was tested only on price, in
the H11 cells. A NO bid at the mid filled 66 times across 12 cells and lost (best -21c, 6 fills): the
single "above 0" leg is not YES-rich, and its dev calibration buckets are within 9c of the price.
Extra dev checks on the leading cell before freezing (also counted as cells): by month (6/7 positive),
by decision time, spreads <= 3c (+20.2c), prices 10-90c (+25.5c), side split. Of 117 grid cells with
>= 40 fills and >= 10 events, 12 reach t >= 2, all of them correlated KXTRUMPACT / WH nowcast variants,
and 73 have a negative mean."""
RULE_NOTE = """Rule: at Fri 06:00 ET (75% of the week), Sat 07:00 ET (90%) and Sat 20:00 ET, for every open KXTRUMPACT leg,
buy YES at the ask if p - ask - fee/n >= 5c, or NO at 1 - bid if (1 - p) - (1 - bid) - fee/n >= 5c,
~$2 per order, held to settlement; p = the as-of WH count + 12-week analog remainder above."""
HOLDOUT_NOTE = """Post hoc, after the holdout was scored (not a confirmatory test; `posthoc.json`): the primary's NO side
alone made +19.9c (t 4.0, 59 fills / 18 weeks) on dev and +15.6c (7 fills / 4 weeks) on the holdout; its
YES side made +20.6c (12) on dev and -9.5c (9) on the holdout. Dev book for NO-only: Sharpe 3.1 [0.4, 4.9],
-1.6 without the top 5 events, 39% positive weeks, $1.56/week. Holdout NO-only: $0.46/week over 45 days."""


FINDINGS_RULES = """Rules read 2026-11-20 (`rules.json` copies from `/markets`); every series is fee type `quadratic`,
multiplier 1 (taker fee 0.07 x P(1-P) rounded up per order, no maker fee).

* **KXEOWEEK** (Sun-Sat): numbered Executive Orders by the **Federal Register signing date**; memoranda and
  proclamations do not count. Trading continues until the FR has published (expiry up to two weeks after
  the week). Ladder "above k" (k = 0..4 in 2026).
* **KXTRUMPEOD** (daily, since Nov 2026): an EO signed that day with full text on whitehouse.gov or in a
  pool report by 10:00 ET next day. 4 settled markets, zero volume: not testable.
* **KXTRUMPACT** (Sun-Sat; daily "anything on <day>" markets in Feb 2026) and **KXTRUMPACTD** (daily, since
  2026-10-11): items on whitehouse.gov "Presidential Actions" by the **displayed publication date**, checked
  10:00 ET the next day; any item counts (EO, proclamation, memorandum, nominations post, ...); an item
  released later with a backdated date does not count for the later day.
* **KXTRUMPNOMNUM** (Sun-Sat): nominations sent to the Senate, from the same page's publication dates; each
  nomination counts (one person twice = 2).
* **KXMAMDANIEO** (Sun-Sat): non-emergency NYC mayoral EOs; expiry three months after the week.
* **KXTRUMPENDORSEMENTS** (Truth Social) and **KXTRUMPPHOTO / KXTRUMPPHOTOD** (Getty Images): sources block
  us, so price and outcome only. Monthly KXBILLSSIGNED / KXBILLSCOUNT / KXPARDONSTRUMP: 2-12 events each,
  not tested.
"""
SOURCES_NOTE = """Used: the White House Presidential Actions RSS feed (`/presidential-actions/feed/?paged=N`, 22 pages of 30,
2025-01-20 .. 2026-11-16; `pubDate` equals the listing's displayed `datetime` to the second), the Federal
Register documents API (EOs signed since 2025-01-20), senate.gov's 2025 and 2026 schedules (published in
advance: "Updated on Dec 05, 2024" and "Updated on Nov 21, 2025", so as-of safe), and the Kalshi public
API. Not used: `/wp-json/` (robots allows it but the server answers 403; not worked around), Getty Images and
Truth Social (robots 403), presidency.ucsb.edu (blocks ClaudeBot), congress.gov and rollcall.com (block
these agents), the public-inspection PDF host (403). nyc.gov allows us, but its EO list is rendered in the
browser (no static list or JSON found), so KXMAMDANIEO is price-only.

**As-of status of the WH feed.** `pubDate` is the time the item became public: (1) trade prints on the
strike each item decides move within minutes *after* it (section 5), and (2) the feed as published today
rebuilds every Kalshi result (section 4). One of 273 item-market pairs looks backdated (an exact 01:00:00
UTC stamp, never priced above 90c within 6 h). History is therefore rebuilt as of each decision.
"""
SETTLE_NOTE = """KXTRUMPACT / KXTRUMPACTD: count of feed items with displayed (ET) date in the period; KXTRUMPNOMNUM: one
nomination per "<name>, of <state>, to be ..." paragraph before the WITHDRAWAL header; KXEOWEEK / KXTRUMPEOD:
FR EOs by signing date. The White House EO category matches the FR signing date for 90 of 93 matched
dev EOs (same day) and the weekly counts agree in 46 of 54 dev weeks (off by one in 8).
"""


# ---- CLI -----------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog=f"albert.research.{NAME}")
    ap.add_argument("cmd")
    ap.add_argument("--snap", default=SNAP, help="discover: cache tag for the market listings (use today's date to refresh)")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "discover":
        discover(snap=a.snap)
        return
    fn = COMMANDS.get(a.cmd)
    if fn is None:
        raise SystemExit(f"unknown command {a.cmd}; one of {sorted(COMMANDS)}")
    fn()


COMMANDS: dict[str, Any] = {
    "robots": lambda: print(json.dumps([r for r in check_robots() if True], indent=0)[:4000]),
    "discover": lambda: discover(),
    "wh": lambda: fetch_wh(),
    "fr": lambda: print(fetch_fr().tail(5)),
    "candles": lambda: fetch_candles(),
    "senate": lambda: print(fetch_senate().filter(pl.col("day") >= date(2026, 7, 1))),
    "settle": lambda: settle_check(),
    "build": lambda: build(),
    "screen": lambda: screen(),
    "freeze": lambda: print(json.dumps(freeze(), indent=1, default=str)[:3000]),
    "holdout": lambda: holdout(),
    "record": lambda: record(),
    "latency": lambda: latency(),
    "explore": lambda: [print(k, v, sep="\n") for k, v in explore().items()],
    "report": lambda: write_report(verdict=VERDICT, extra=report_body()),
}


if __name__ == "__main__":
    main()
