"""Accumulating-quantity ladders on Kalshi: does a partial-sum nowcast beat the price late in the period?

Hypothesis (2026-11-17): ladders that settle on a total, average or count over a week / month, whose
components are published while the period runs, under-react to the partial sum already published, so
"known partial + simple forecast of the remainder" beats the price late in the period.

**Discovery and triage** (:func:`discover`, :func:`triage`). Candidate series are ticker families in
today's ``/series`` (see ``FAMILIES``); a series is excluded when its Kalshi ``category`` or title is
financial (``albert.categories.kalshi_series_is_financial``: Financials, Economics, Crypto, Commodities,
Companies; TSA weekly/daily are filed under Economics and are reported only). Every settled market comes
from ``/historical/markets`` + ``/markets?status=settled`` per series. Each family's component source,
its robots.txt verdict (:func:`check_robots`) and whether it can be rebuilt *as of* a past decision are in
``FEASIBILITY``. Split by the period's last local day: dev <= 2026-09-30, holdout 2026-10-01..11-16.

**Tested families.**

* ``rain_month`` (KXRAIN*M): monthly precipitation at one CLI station. Components: IEM ``json/cli.py``
  daily precipitation; a day is known at ``t`` only if its CLI product (issuance time = product-id
  prefix) was issued by ``t``. Remainder: 225 analog sums of the same unknown calendar days in the 15
  previous years shifted -7..+7 days.
* ``temp_avg_week`` (KXAVGTK*): "at least N consecutive days with daily mean > X F" in a Mon-Sun week.
  Components: CLI (high + low) / 2; remainder: analog paths plus the last known anomaly decayed 0.7/day.
* ``eo_week`` (KXEOWEEK): executive orders signed in a week. Components: Federal Register API signing
  date, known from its public-inspection time (median 4 days after signing; the White House posts the
  same day, so this nowcast is structurally late). Remainder: Poisson at the recent signing rate over the
  week's not-yet-visible days.

**Test** (:func:`build`, :func:`evaluate`). Decisions at 25/50/75/90% of the period and the final day;
quote = last hourly candle closed at or before the decision. Taker at the ask / 1 - bid (<= 10c wide),
~$2 orders capped at 20 contracts, taker fee rounded up per order; maker bids at nowcast +- margin,
filled only on later prints strictly through (hourly candle trade range), held to settlement. The
primary rule is frozen on dev (``primary.json``) before the holdout is evaluated; t-stats are clustered
by event and by date (two-way). Ledgers: $100 book, <= $5/event, <= $20 open, daily MTM at the candle mid
(``albert.research.vol_metrics``).

**Result (2026-11-17, report ``data/research/counts/report.md``): no tradable edge; the hypothesis is
rejected.** The CLI rebuild matches Kalshi's result on every rain market (997/997) and 97-100% of streak
markets. When the issued CLI days already pass a strike, the market is at 99c bid / $1.00 ask (935 dev
rows, 7 takeable). On undecided priced rows the mid beats the nowcast late in the period (rain f90 Brier
0.111 vs 0.151, final 0.066 vs 0.111). The dev-frozen primary (maker, bids 12c around the nowcast,
rain, 75/90%/final decisions) makes +4.9c per fill (t 3.4, 524 fills / 112 events) dev and +1.6c
(t 0.6, 69 / 15) holdout, no better than the same maker around the recalibrated mid (+2.7c dev, +6.1c
holdout); book Sharpe 0.54 [-0.39, 1.50] dev, ex top-5 -1.06, $0.27/week. Executive orders: the
Federal Register lags the White House by ~4 days, and that nowcast loses everywhere. Nothing to paper
trade.

    python -m albert.research.counts robots | discover | triage | cli | candles | build | evaluate | report | all
"""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.robotparser
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.venues.kalshi import _field, num, parse_ts

OUT = Path("data/research/counts")
DEV_END = datetime(2026, 10, 1, tzinfo=UTC)          # dev: period ends before this
HOLDOUT_END = datetime(2026, 11, 17, tzinfo=UTC)     # holdout: period ends in [DEV_END, HOLDOUT_END)
KALSHI_MIN_INTERVAL = 1.0                            # <= 1 request/s, sequential (shared IP)
IEM_MIN_INTERVAL = 0.5

# ---- candidate families: ticker regex -> family (Kalshi /series, reviewed 2026-11-17) ----------------
FAMILIES: list[tuple[str, str]] = [
    ("rain_month", r"^(KX)?RAIN[A-Z]+M$"),
    ("rain_week", r"^KXRAIN[A-Z]+W$"),
    ("snow_month", r"^(KX)?(?!.*MENTION)[A-Z]*SNOW[A-Z]*M[B]?$|^KXALTASNOW$|^KXSNOWNYM?$|^SNOWNYM?$"),
    ("temp_avg_week", r"^KXAVGT(K[A-Z]{3}|ATL)$"),
    ("temp_high_week", r"^KXWEEKHT[A-Z]+$"),
    ("heat_week", r"^KXHEATA[A-Z]+$"),
    ("temp_low_month", r"^KX(MONTHLOWMSN|MSNLOWT|SJCLOWT|MINNYC)$"),
    ("us_temp_month", r"^KX(AVGTEMP|TEMPMON)$|^(AVGTEMP|TEMPMON)$"),
    ("tornado_month", r"^KXTORNADO$"),
    ("storms_season", r"^KX(HURCTOT|HURCTOTMAJ|HURCMAJ|TROPSTORM)$"),
    ("truth_posts", r"^KXTRUTHSOCIALD?$"),
    ("tweets_week", r"^KX(POTUSTWEETS|ELONTWEETS|POPETWEETS)$"),
    ("eo_week", r"^KXEOWEEK$"),
    ("trump_counts_week", r"^KX(TRUMPENDORSEMENTS|TRUMPNOMNUM|BILLSCOUNTWEEKLY)$"),
    ("trump_counts_month", (r"^KX(BILLSSIGNED|PRESSBRIEFINGCOUNT|LAGOVISITCOUNT|DCGOLFVISITCOUNT|"
                            r"NJGOLFVISITCOUNT|TRUMPDELETE|TRUMPNUMSTATES|BILLCOUNTM)$")),
    ("tsa", r"^(KX)?TSA[WD]?$|^KXTRUFTSA$"),
    ("transit_week", r"^KX(MTA|PORTLA)$"),
    ("streams_week", r"^KX(ARTISTSTREAMS|DRAKE|BUNNY|KLAMAR|SCARPENTER|BMARS|ODEAN|PMALONE|KWEST|BEYONCE)$"),
    ("spotify_days_month", r"^KX(SPOTIFYNUM1M|SPOTIFYNUM1USA|KXSPOTIFYNUM1GLB)$"),
    ("launches_month", r"^KX(LAUNCHCOUNTM|ROCKET)$"),
    ("disease_year", r"^KX(MEASLES|PERTUSSIS|FLDENGUE)$"),
    ("ships_month", r"^KXMAXSHIPSHORMUZ$"),
    ("season_wins", r"^KX(NBAWINS|NFLWINS|WNBAWINS|NHLWINS|NHLSEASONPTS|NCAAFWINS)$"),
]


def family_of(series: str) -> str | None:
    for fam, rx in FAMILIES:
        if re.match(rx, series or ""):
            return fam
    return None


def _api(min_interval: float = KALSHI_MIN_INTERVAL):
    from albert.config import Settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    s = Settings()
    return k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=min_interval, max_retries=12))


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


# ---- robots ----------------------------------------------------------------------------------------
AGENTS = ("*", "anthropic-ai", "ClaudeBot", "Claude-Web")
ROBOTS_CHECKS: dict[str, list[str]] = {   # host -> paths we would use (or considered using)
    "mesonet.agron.iastate.edu": ["/json/cli.py", "/cgi-bin/request/asos.py", "/cgi-bin/request/mos.py"],
    "api.elections.kalshi.com": ["/trade-api/v2/markets", "/trade-api/v2/series"],
    "www.tsa.gov": ["/travel/passenger-volumes", "/coronavirus/passenger-throughput"],
    "rollcall.com": ["/factbase/trump/topic/social/"],
    "truthsocial.com": ["/@realDonaldTrump", "/api/v1/accounts/107780257626128497/statuses"],
    "x.com": ["/elonmusk"],
    "weather.com": ["/kalshi"],
    "www.spc.noaa.gov": ["/climo/reports/", "/wcm/data/"],
    "data.ny.gov": ["/resource/"],
    "www.federalregister.gov": ["/api/v1/documents.json"],
    "www.whitehouse.gov": ["/presidential-actions/"],
    "www.congress.gov": ["/nominations"],
}


def robots_verdict(text: str, path: str, agents: tuple[str, ...] = AGENTS) -> dict[str, bool]:
    """``can_fetch`` per agent for one robots.txt body (an empty or missing file allows all)."""
    rp = urllib.robotparser.RobotFileParser()
    # RFC 9309: indentation and blank lines do not end a group (urllib's parser treats a blank line
    # as the end of one, which would drop rules that follow it); strip both, which can only add rules
    rp.parse([ln.strip() for ln in text.splitlines() if ln.strip()])
    return {a: rp.can_fetch(a, path) for a in agents}


def check_robots(out: Path = OUT) -> list[dict[str, Any]]:
    """Fetch each host's own robots.txt (no cache, so the check is current) and evaluate our paths."""
    import httpx

    rows = []
    d = out / "robots"
    d.mkdir(parents=True, exist_ok=True)
    for host, paths in ROBOTS_CHECKS.items():
        url = f"https://{host}/robots.txt"
        try:
            r = httpx.get(url, timeout=20, follow_redirects=True, headers={"User-Agent": "albert-research/0.2"})
            status, text = r.status_code, (r.text if r.status_code == 200 else "")
        except httpx.HTTPError as error:
            status, text = None, ""
            print(f"robots {host}: {error}", flush=True)
        (d / f"{host}.txt").write_text(text)
        for p in paths:
            v = robots_verdict(text, p) if status in (200, 404, 410) else dict.fromkeys(AGENTS, False)
            rows.append({"host": host, "path": p, "http_status": status, "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                         **{f"allow_{a}": v[a] for a in AGENTS}, "allowed": all(v.values())})
    (out / "robots.json").write_text(json.dumps(rows, indent=1))
    return rows


# ---- discovery -------------------------------------------------------------------------------------
SERIES_COLS = {"series_ticker": pl.String, "family": pl.String, "category": pl.String, "title": pl.String,
               "frequency": pl.String, "fee_type": pl.String, "fee_multiplier": pl.Float64,
               "source_name": pl.String, "source_url": pl.String}
MARKET_COLS = {"ticker": pl.String, "event_ticker": pl.String, "series_ticker": pl.String,
               "title": pl.String, "yes_sub_title": pl.String, "strike_type": pl.String,
               "floor_strike": pl.Float64, "cap_strike": pl.Float64, "open_time": pl.Datetime("us", "UTC"),
               "close_time": pl.Datetime("us", "UTC"), "expected_expiration_time": pl.Datetime("us", "UTC"),
               "settlement_ts": pl.Datetime("us", "UTC"), "result": pl.String, "expiration_value": pl.String,
               "can_close_early": pl.Boolean, "volume": pl.Float64, "rules_primary": pl.String,
               "status": pl.String, "src": pl.String}


def series_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Candidate series from a ``/series`` payload: a family match and a non-financial category/title."""
    from albert.categories import kalshi_series_is_financial

    rows = []
    for s in payload.get("series") or []:
        fam = family_of(s.get("ticker"))
        if fam is None:
            continue
        src = (s.get("settlement_sources") or [{}])[0] or {}
        rows.append({"series_ticker": s.get("ticker"), "family": fam, "category": s.get("category"),
                     "title": s.get("title"), "frequency": s.get("frequency"), "fee_type": s.get("fee_type"),
                     "fee_multiplier": num(s.get("fee_multiplier")), "source_name": src.get("name"),
                     "source_url": src.get("url"),
                     "financial": kalshi_series_is_financial(s.get("ticker"), s.get("category"), s.get("title"))})
    return rows


def market_row(m: dict[str, Any], series: str, src: str) -> dict[str, Any]:
    return {"ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"), "series_ticker": series,
            "title": m.get("title"), "yes_sub_title": m.get("yes_sub_title"), "strike_type": m.get("strike_type"),
            "floor_strike": num(m.get("floor_strike")), "cap_strike": num(m.get("cap_strike")),
            "open_time": parse_ts(m.get("open_time")), "close_time": parse_ts(m.get("close_time")),
            "expected_expiration_time": parse_ts(m.get("expected_expiration_time")),
            "settlement_ts": parse_ts(m.get("settlement_ts")), "result": m.get("result") or None,
            "expiration_value": (str(m.get("expiration_value")) if m.get("expiration_value") not in (None, "") else None),
            "can_close_early": m.get("can_close_early"), "volume": num(_field(m, "volume")),
            "rules_primary": m.get("rules_primary"), "status": m.get("status"), "src": src}


def _pages(api, path: str, params: dict[str, Any]):
    cursor = None
    while True:
        page = api.client.get(path, {**params, "cursor": cursor}, cache=True)
        yield page
        cursor = page.payload.get("cursor")
        if not cursor or not page.payload.get("markets"):
            return


def discover(out: Path = OUT, *, log=print) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Candidate series (today's ``/series``) and every settled market of each (historical + live)."""
    api = _api()
    ser = pl.DataFrame(series_rows(api.series().payload), infer_schema_length=None)
    _save(ser, out / "series.parquet")
    rows: list[dict[str, Any]] = []
    todo = ser.filter(~pl.col("financial"))
    for i, s in enumerate(todo.iter_rows(named=True)):
        t = s["series_ticker"]
        for page in _pages(api, "/historical/markets", {"series_ticker": t, "limit": 1000}):
            rows.extend(market_row(m, t, "hist") for m in page.payload.get("markets") or [])
        for page in _pages(api, "/markets", {"series_ticker": t, "status": "settled", "limit": 1000}):
            rows.extend(market_row(m, t, "live") for m in page.payload.get("markets") or [])
        if i % 20 == 0:
            log(f"  discover {i + 1}/{todo.height} {t}: {len(rows)} markets so far", flush=True)
    mk = (pl.DataFrame(rows, schema=MARKET_COLS).unique("ticker", keep="last")
          .filter(pl.col("result").is_in(["yes", "no"])))
    _save(mk, out / "markets.parquet")
    log(f"discover: {ser.height} candidate series ({todo.height} non-financial), {mk.height} settled markets")
    return ser, mk




# ---- triage ------------------------------------------------------------------------------------------
# Family -> (component source and whether it can be rebuilt as of a past decision time, under the
# data-source policy). Robots verdicts come from ``check_robots`` (data/research/counts/robots.json).
FEASIBILITY: dict[str, tuple[str, bool]] = {
    "rain_month": ("IEM CLI daily precipitation, stamped with the CLI issuance time; allowed", True),
    "rain_week": ("IEM CLI daily precipitation; allowed", True),
    "snow_month": (("IEM CLI daily snowfall; allowed (OnTheSnow resort series: no source); winter "
                    "only, no event settles in the holdout, so not tested"), True),
    "temp_avg_week": (("IEM CLI daily high/low (series settle on The Weather Company; proxy checked "
                      "against Kalshi expiration_value)"), True),
    "temp_high_week": ("IEM CLI daily high (TWC settlement; proxy checked)", True),
    "heat_week": ("IEM CLI daily high (NWS CLI Phoenix)", True),
    "temp_low_month": ("IEM CLI daily low (TWC/NWS; proxy checked)", True),
    "us_temp_month": ("NCEI monthly national/global analysis: no published daily components", False),
    "tornado_month": ("SPC storm reports: www.spc.noaa.gov robots.txt is 'Disallow: /' for *", False),
    "storms_season": ("NHC advisories; one season per year, nothing settles Oct 1-Nov 17 in most", False),
    "truth_posts": (("settles on Roll Call Factba.se: rollcall.com blocks anthropic-ai; truthsocial.com "
                    "robots.txt 403"), False),
    "tweets_week": ("x.com robots.txt disallows every path for *", False),
    "eo_week": (("Federal Register API (robots allows /api/v1): signing date + public-inspection "
                "timestamp; lags the White House posting by 1-6 days"), True),
    "trump_counts_week": (("Truth Social endorsements (truthsocial.com robots 403) / White House "
                          "nominations: no issuance-stamped archive of what was public when"), False),
    "trump_counts_month": (("White House / Roll Call / press pool: no issuance-stamped archive; Roll "
                           "Call blocked"), False),
    "tsa": (("tsa.gov daily throughput (robots allows); KXTSAW is category Economics -> excluded; "
            "legacy TSA/TSAW only"), False),
    "transit_week": ("MTA ridership (data.ny.gov allows) / Port of LA; see event counts", False),
    "streams_week": ("Luminate (paywalled, no public daily components)", False),
    "spotify_days_month": ("Spotify charts (login wall)", False),
    "launches_month": ("FAA licensed-launch list: no as-of archive; few events", False),
    "disease_year": ("CDC weekly counts; annual events (1 per year)", False),
    "ships_month": ("Hormuz transit counts (no public as-of daily source found)", False),
    "season_wins": ("game results (ESPN API allowed) but seasons settle outside the holdout", False),
}


def event_table(mk: pl.DataFrame) -> pl.DataFrame:
    """One row per event: its scheduled end (latest expected expiration, else close), markets, volume."""
    end = pl.coalesce(pl.col("expected_expiration_time"), pl.col("close_time"))
    return (mk.group_by("series_ticker", "event_ticker")
            .agg(end.max().alias("end"), pl.len().alias("markets"), pl.col("volume").sum().alias("volume"),
                 pl.col("open_time").min().alias("open"))
            .with_columns(pl.when(pl.col("end") < DEV_END).then(pl.lit("dev"))
                          .when(pl.col("end") < HOLDOUT_END).then(pl.lit("holdout")).otherwise(pl.lit("later"))
                          .alias("split")))


def triage(out: Path = OUT) -> pl.DataFrame:
    ser = pl.read_parquet(out / "series.parquet").with_columns(
        pl.col("series_ticker").map_elements(family_of, return_dtype=pl.String).alias("family"))
    mk = pl.read_parquet(out / "markets.parquet")
    ev = event_table(mk)
    per = (ev.group_by("series_ticker")
           .agg((pl.col("split") == "dev").sum().alias("events_dev"),
                (pl.col("split") == "holdout").sum().alias("events_holdout"),
                pl.col("markets").sum().alias("markets"), pl.col("volume").sum().alias("volume"),
                pl.col("end").min().alias("first_end"), pl.col("end").max().alias("last_end")))
    t = (ser.join(per, on="series_ticker", how="left")
         .with_columns(pl.col("events_dev", "events_holdout", "markets").fill_null(0),
                       pl.col("family").replace_strict({k: v[0] for k, v in FEASIBILITY.items()}, default=None)
                       .alias("component_source"),
                       pl.col("family").replace_strict({k: v[1] for k, v in FEASIBILITY.items()}, default=False)
                       .alias("as_of_feasible"))
         .sort("family", "series_ticker"))
    u = universe(out)            # tested families: split by the period's last local day
    if u.height:
        tested = u.group_by("series_ticker").agg(
            pl.col("event_ticker").filter(pl.col("split") == "dev").n_unique().alias("tested_events_dev"),
            pl.col("event_ticker").filter(pl.col("split") == "holdout").n_unique().alias("tested_events_holdout"))
        t = t.join(tested, on="series_ticker", how="left")
    _save(t, out / "triage.parquet")
    t.drop("financial").write_csv(out / "triage.csv")
    return t



# ---- stations and periods (parsed from each market's rules) ------------------------------------------
_CITY_STATION = {"Central Park": "KNYC", "Chicago Midway": "KMDW"}


def rules_station(rules: str | None) -> str | None:
    """ICAO-style station the market settles on: ``CLIAUS`` -> KAUS, ``at KDEN`` -> KDEN, Central Park
    -> KNYC (the CLI site). None when the rules name only a city."""
    r = rules or ""
    m = re.search(r"\bCLI([A-Z]{3})\b", r)
    if m:
        return "K" + m.group(1)
    m = re.search(r"\bat (K[A-Z]{3})\b", r)
    if m:
        return m.group(1)
    for k, v in _CITY_STATION.items():
        if k in r:
            return v
    return None


_MONTHS = {m: i + 1 for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
                                           "NOV", "DEC"))}
_MONTH_NAMES = {n: i + 1 for i, n in enumerate(("January", "February", "March", "April", "May", "June", "July",
                                                "August", "September", "October", "November", "December"))}
_MONTH_NAMES.update({n[:3]: i for n, i in list(_MONTH_NAMES.items())})


def month_period(event_ticker: str) -> tuple[date, date] | None:
    """``KXRAINNYCM-26OCT`` -> (2026-10-01, 2026-10-31): the local calendar days summed."""
    m = re.search(r"-(\d{2})([A-Z]{3})$", event_ticker or "")
    if not m or m.group(2) not in _MONTHS:
        return None
    y, mo = 2000 + int(m.group(1)), _MONTHS[m.group(2)]
    nxt = date(y + (mo == 12), mo % 12 + 1, 1)
    return date(y, mo, 1), nxt - timedelta(days=1)


def rules_period(rules: str | None) -> tuple[date, date] | None:
    """``within October 5, 2026 through October 11, 2026`` / ``during Oct 25, 2026 to Oct 31, 2026`` ->
    (first day, last day)."""
    rx = r"([A-Z][a-z]{2,8})\.? (\d{1,2}),? (\d{4})"
    m = re.search(rx + r"\s*(?:through|to|-|and)\s*" + rx, rules or "")
    if not m:
        return None
    a, b = m.groups()[:3], m.groups()[3:]
    try:
        return (date(int(a[2]), _MONTH_NAMES[a[0]], int(a[1])), date(int(b[2]), _MONTH_NAMES[b[0]], int(b[1])))
    except (KeyError, ValueError):
        return None



# ---- the tested universe ---------------------------------------------------------------------------
TESTED = ("rain_month", "temp_avg_week", "eo_week")
DEV_LAST_DAY = date(2026, 9, 30)        # a period whose last local day is on/before this is dev
HOLDOUT_LAST_DAY = date(2026, 11, 16)   # ... on/before this (and after DEV_LAST_DAY) is holdout


def universe(out: Path = OUT, families: tuple[str, ...] = TESTED) -> pl.DataFrame:
    """Settled markets of the tested families with station, period (local days) and split."""
    mk = pl.read_parquet(out / "markets.parquet").with_columns(
        pl.col("series_ticker").map_elements(family_of, return_dtype=pl.String).alias("family"))
    mk = mk.filter(pl.col("family").is_in(list(families)))
    rows = []
    for r in mk.iter_rows(named=True):
        per = month_period(r["event_ticker"]) if r["family"].endswith("_month") else rules_period(r["rules_primary"])
        st = "FR" if r["family"] == "eo_week" else rules_station(r["rules_primary"])
        if per is None or st is None:
            continue
        if r["family"] == "eo_week" and r["strike_type"] is None:
            m = re.search(r"above (\d+)|signs (any) ", r["rules_primary"] or "")
            if not m:
                continue
            r = {**r, "strike_type": "greater", "floor_strike": float(m.group(1) or 0)}
        split = "dev" if per[1] <= DEV_LAST_DAY else "holdout" if per[1] <= HOLDOUT_LAST_DAY else "later"
        rows.append({**r, "station": st, "p_first": per[0], "p_last": per[1], "split": split})
    u = pl.DataFrame(rows, infer_schema_length=None)
    ser = pl.read_parquet(out / "series.parquet").select("series_ticker", "fee_type", "fee_multiplier")
    return u.join(ser, on="series_ticker", how="left").filter(pl.col("split") != "later")


# ---- Kalshi hourly candles (OHLC of price, bid, ask) -----------------------------------------------
def fetch_candles(out: Path = OUT, *, families: tuple[str, ...] = TESTED, log=print) -> pl.DataFrame:
    """Hourly candles over each market's life from 3 days before its period (historical endpoint per
    market before the cutoff, live batch per event after). Sequential at <= 1 request/s."""
    from albert.research import xcat

    u = universe(out, families)
    lo = pl.max_horizontal(pl.col("open_time"),
                           pl.col("p_first").cast(pl.Datetime("us")).dt.replace_time_zone("UTC") - timedelta(days=3))
    sample = u.select("ticker", "event_ticker", "settlement_ts", pl.lit(60).alias("period"), lo.alias("lo"),
                      pl.col("close_time").alias("hi"))
    d = out / "candles"
    d.mkdir(parents=True, exist_ok=True)
    done_path = d / "done.parquet"
    done = set(pl.read_parquet(done_path)["ticker"]) if done_path.exists() else set()
    todo = sample.filter(~pl.col("ticker").is_in(list(done)))
    api = _api()
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
        return pl.DataFrame(schema={"ticker": pl.String})
    return pl.concat([pl.read_parquet(p) for p in parts]).unique(["ticker", "end_ts"]).sort("ticker", "end_ts")


# ---- IEM CLI (daily climate reports): the components, with their issuance time --------------------
CLI_COLS = {"station": pl.String, "day": pl.Date, "issued": pl.Datetime("us", "UTC"), "precip": pl.Float64,
            "snow": pl.Float64, "high": pl.Float64, "low": pl.Float64, "precip_month": pl.Float64,
            "snow_month": pl.Float64}
TRACE = 0.0001       # "T" (trace) counts as > 0 but adds nothing measurable


def _cli_val(v: Any) -> float | None:
    if v == "T":
        return TRACE
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def product_time(product: str | None) -> datetime | None:
    """``"202608190617-KOKX-CDUS41-CLINYC"`` -> 2026-08-19 06:17 UTC (when that CLI was issued)."""
    m = re.match(r"^(\d{12})-", product or "")
    return datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=UTC) if m else None


def cli_rows(payload: dict[str, Any], station: str) -> list[dict[str, Any]]:
    """IEM ``json/cli.py``: one row per CLI day, values as floats (trace = 0.0001, missing = None)."""
    rows = []
    for r in payload.get("results") or []:
        try:
            day = date.fromisoformat(r["valid"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({"station": station, "day": day, "issued": product_time(r.get("product")),
                     **{k: _cli_val(r.get(k)) for k in ("precip", "snow", "high", "low", "precip_month", "snow_month")}})
    return rows


def _iem():
    from albert.config import Settings
    from albert.http import CachedClient

    s = Settings()
    return CachedClient("iem", s.iem_base, s.raw_dir, min_interval=IEM_MIN_INTERVAL, max_retries=6, timeout=90)


def fetch_cli(stations: list[str], years: range, out: Path = OUT, *, log=print) -> pl.DataFrame:
    """CLI history per station-year. Past years are immutable (cached); the current year is fetched
    under today's tag so new days arrive without overwriting an earlier raw file."""
    client = _iem()
    today = datetime.now(UTC).date()
    rows: list[dict[str, Any]] = []
    for st in stations:
        for y in years:
            tag = today.isoformat() if y >= today.year else None
            try:
                page = client.get("/json/cli.py", {"station": st, "year": y}, cache=True, tag=tag)
            except RuntimeError as error:
                log(f"  cli {st} {y}: {str(error)[:120]}")
                continue
            rows.extend(cli_rows(page.payload, st))
        log(f"  cli {st}: {len(rows)} rows so far", flush=True)
    df = pl.DataFrame(rows, schema=CLI_COLS).unique(["station", "day"], keep="last").sort("station", "day")
    _save(df, out / "cli.parquet")
    return df



# ---- the nowcast: known partial sum + analog-climatology remainder -----------------------------------
STATION_TZ = {
    "KNYC": "America/New_York", "KMIA": "America/New_York", "KATL": "America/New_York",
    "KCMH": "America/New_York", "KLEX": "America/New_York", "KPVD": "America/New_York",
    "KSPG": "America/New_York", "KAUS": "America/Chicago", "KDFW": "America/Chicago",
    "KHOU": "America/Chicago", "KIAH": "America/Chicago", "KCLL": "America/Chicago",
    "KMDW": "America/Chicago", "KORD": "America/Chicago", "KMKE": "America/Chicago",
    "KMSY": "America/Chicago", "KOKC": "America/Chicago", "KSAT": "America/Chicago",
    "KDEN": "America/Denver", "KPHX": "America/Phoenix", "KLAX": "America/Los_Angeles",
    "KSFO": "America/Los_Angeles", "KSEA": "America/Los_Angeles", "KLAS": "America/Los_Angeles",
    "KSAN": "America/Los_Angeles", "KAPC": "America/Los_Angeles", "FR": "America/New_York",
}
FRACTIONS = (0.25, 0.5, 0.75, 0.9)
FINAL_LEAD = timedelta(hours=10)      # "final day" decision: 10 h before the period's last LST midnight
ANALOG_YEARS = 15                     # climatology: the same calendar days in the 15 previous years
ANALOG_SHIFT = 7                      # ... shifted by -7..+7 days (15 x 15 = 225 analog paths)


def lst_bounds(station: str, first: date, last: date) -> tuple[datetime, datetime]:
    """UTC instants of local-standard midnight before ``first`` and after ``last`` (the CLI day)."""
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(STATION_TZ[station])
    probe = datetime(first.year, 1, 15, 12, tzinfo=tz)       # January: standard time everywhere
    std = probe.utcoffset()
    start = datetime(first.year, first.month, first.day, tzinfo=UTC) - std
    end = datetime(last.year, last.month, last.day, tzinfo=UTC) - std + timedelta(days=1)
    return start, end


def decision_times(p0: datetime, p1: datetime) -> list[tuple[str, datetime]]:
    """25/50/75/90% through the period and the final day, each floored to the hour."""
    def floor_h(x: datetime) -> datetime:
        return x.replace(minute=0, second=0, microsecond=0)
    out = [(f"f{int(f * 100)}", floor_h(p0 + (p1 - p0) * f)) for f in FRACTIONS]
    out.append(("final", floor_h(p1 - FINAL_LEAD)))
    return out


class StationHistory:
    """Daily CLI values on a dense calendar (NaN = missing) and each day's issuance time (epoch s)."""

    def __init__(self, cli: pl.DataFrame) -> None:
        cli = cli.sort("day")
        self.base: date = cli["day"][0]
        n = (cli["day"][-1] - self.base).days + 1
        idx = np.array([(d - self.base).days for d in cli["day"].to_list()])
        self.precip = np.full(n, np.nan)
        self.tavg = np.full(n, np.nan)
        self.issued = np.full(n, np.inf)
        pr = cli["precip"].to_numpy().astype(float)
        self.precip[idx] = np.where(pr == TRACE, 0.0, pr)      # trace adds nothing measurable
        hi, lo = cli["high"].to_numpy().astype(float), cli["low"].to_numpy().astype(float)
        self.tavg[idx] = (hi + lo) / 2
        iss = cli["issued"].dt.epoch("s").to_numpy().astype(float)
        self.issued[idx] = np.where(np.isnan(iss), np.inf, iss)

    def ix(self, d: date) -> int:
        return (d - self.base).days

    def known(self, days: list[date], t: datetime, field: str) -> np.ndarray:
        """Mask of period days whose CLI was issued at or before ``t`` with a value."""
        ts = t.timestamp()
        arr = getattr(self, field)
        out = np.zeros(len(days), dtype=bool)
        for i, d in enumerate(days):
            j = self.ix(d)
            out[i] = 0 <= j < len(arr) and self.issued[j] <= ts and not np.isnan(arr[j])
        return out

    def analogs(self, days: list[date], field: str, *, years: int = ANALOG_YEARS,
                shift: int = ANALOG_SHIFT) -> np.ndarray:
        """Matrix (paths x len(days)): the same calendar days in each of the ``years`` years before the
        period, shifted by -shift..+shift days; paths with a missing day are dropped. Every value is
        from a year before the period, so it was public long before any decision."""
        arr = getattr(self, field)
        y0 = days[0].year
        paths = []
        for y in range(y0 - years, y0):
            for dlt in range(-shift, shift + 1):
                js = []
                for d in days:
                    try:
                        dd = d.replace(year=y - (d.year - y0))
                    except ValueError:                 # Feb 29
                        dd = date(y - (d.year - y0), 2, 28)
                    js.append(self.ix(dd + timedelta(days=dlt)))
                js = np.array(js)
                if js.min() < 0 or js.max() >= len(arr):
                    continue
                v = arr[js]
                if not np.isnan(v).any():
                    paths.append(v)
        return np.array(paths) if paths else np.empty((0, len(days)))


def smooth_p(hits: float, n: int) -> float:
    return (hits + 0.5) / (n + 1)


def rain_nowcast(h: StationHistory, days: list[date], t: datetime, strike: float) -> dict[str, Any]:
    """P(total > strike) = P(known + remainder > strike); remainder from analog sums of the unknown days."""
    kn = h.known(days, t, "precip")
    k_sum = float(sum(h.precip[h.ix(d)] for d, m in zip(days, kn, strict=True) if m))
    unknown = [d for d, m in zip(days, kn, strict=True) if not m]
    base = {"known_days": int(kn.sum()), "unknown_days": len(unknown), "partial": k_sum}
    if k_sum > strike + 1e-9:
        return {**base, "p_model": 1.0, "decided": True, "n_paths": 0, "clim_mean_rem": 0.0}
    if not unknown:
        return {**base, "p_model": 0.0, "decided": True, "n_paths": 0, "clim_mean_rem": 0.0}
    rem = h.analogs(unknown, "precip").sum(axis=1)
    if not len(rem):
        return {**base, "p_model": None, "decided": False, "n_paths": 0, "clim_mean_rem": None}
    hits = float((k_sum + rem > strike + 1e-9).sum())
    return {**base, "p_model": smooth_p(hits, len(rem)), "decided": False, "n_paths": len(rem),
            "clim_mean_rem": float(rem.mean())}


def max_run(hot: np.ndarray) -> np.ndarray:
    """Longest run of True along the last axis."""
    hot = np.atleast_2d(hot)
    best = np.zeros(hot.shape[0], dtype=int)
    cur = np.zeros(hot.shape[0], dtype=int)
    for j in range(hot.shape[1]):
        cur = np.where(hot[:, j], cur + 1, 0)
        best = np.maximum(best, cur)
    return best


ANOM_RHO = 0.7       # daily-mean anomaly persistence per day (checked on station history: lag-1 r ~0.6-0.8)


def streak_nowcast(h: StationHistory, days: list[date], t: datetime, threshold: float, need: int,
                   rho: float = ANOM_RHO) -> dict[str, Any]:
    """P(at least ``need`` consecutive days with daily mean > threshold). Known days are fixed; unknown
    days are analog paths plus the latest known anomaly decayed by ``rho`` per day ahead."""
    kn = h.known(days, t, "tavg")
    vals = np.array([h.tavg[h.ix(d)] if m else np.nan for d, m in zip(days, kn, strict=True)])
    paths = h.analogs(days, "tavg")
    base = {"known_days": int(kn.sum()), "unknown_days": int((~kn).sum())}
    known_run = int(max_run(np.where(kn, vals > threshold, False))[0])
    base["partial"] = float(known_run)
    if known_run >= need:
        return {**base, "p_model": 1.0, "decided": True, "n_paths": 0, "clim_mean_rem": None}
    if not len(paths):
        return {**base, "p_model": None, "decided": False, "n_paths": 0, "clim_mean_rem": None}
    sim = paths.copy()
    if kn.any():
        last = int(np.flatnonzero(kn).max())
        anom = vals[last] - paths[:, last].mean()
        for j in range(len(days)):
            if not kn[j] and j > last:
                sim[:, j] += anom * rho ** (j - last)
    sim[:, kn] = vals[kn]
    runs = max_run(sim > threshold)
    if (~kn).sum() == 0:
        return {**base, "p_model": float(runs[0] >= need), "decided": True, "n_paths": 0, "clim_mean_rem": None}
    hits = float((runs >= need).sum())
    return {**base, "p_model": smooth_p(hits, len(runs)), "decided": False, "n_paths": len(runs),
            "clim_mean_rem": float(sim[:, ~kn].mean())}


def streak_terms(rules: str | None) -> tuple[float, int] | None:
    m = re.search(r"at least (\d+) or more consecutive days.*strictly greater than (\d+(?:\.\d+)?) degrees", rules or "")
    return (float(m.group(2)), int(m.group(1))) if m else None




# ---- executive orders (Federal Register): signing date and public-inspection time --------------------
FR_BASE = "https://www.federalregister.gov"
EO_COLS = {"document_number": pl.String, "eo_number": pl.String, "signing_date": pl.Date, "publication_date": pl.Date,
           "pi_ts": pl.Datetime("us", "UTC"), "title": pl.String}


def pi_time(url: str | None) -> datetime | None:
    """``https://public-inspection.federalregister.gov/2026-22051.pdf?1793114109`` -> the epoch in the
    query (when the document went on public inspection; 15:15 UTC on every checked EO)."""
    m = re.search(r"\?(\d{9,11})$", url or "")
    return datetime.fromtimestamp(int(m.group(1)), UTC) if m else None


def eo_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for d in payload.get("results") or []:
        try:
            sd = date.fromisoformat(d["signing_date"])
            pd_ = date.fromisoformat(d["publication_date"])
        except (KeyError, TypeError, ValueError):
            continue
        pi = pi_time(d.get("public_inspection_pdf_url"))
        # no PI record: available at publication (FR issues post at 06:00 ET = 11:00 UTC)
        pi = pi or datetime(pd_.year, pd_.month, pd_.day, 11, tzinfo=UTC)
        rows.append({"document_number": d.get("document_number"), "eo_number": d.get("executive_order_number"), "signing_date": sd, "publication_date": pd_,
                     "pi_ts": pi, "title": d.get("title")})
    return rows


def fetch_eos(out: Path = OUT, since: date = date(2024, 6, 1)) -> pl.DataFrame:
    from albert.config import Settings
    from albert.http import CachedClient

    s = Settings()
    cl = CachedClient("federalregister", FR_BASE, s.raw_dir, min_interval=1.0, max_retries=6)
    rows, page = [], 1
    while True:
        p = cl.get("/api/v1/documents.json", {
            "conditions[presidential_document_type][]": "executive_order",
            "conditions[signing_date][gte]": since.isoformat(), "per_page": 1000, "page": page, "order": "oldest",
            "fields[]": ["signing_date", "publication_date", "executive_order_number", "title",
                         "public_inspection_pdf_url", "document_number"]},
            cache=True, tag=datetime.now(UTC).date().isoformat())
        rows.extend(eo_rows(p.payload))
        if page >= int(p.payload.get("total_pages") or 1):
            break
        page += 1
    df = pl.DataFrame(rows, schema=EO_COLS).unique("document_number").sort("signing_date")
    _save(df, out / "eos.parquet")
    return df


EO_RATE_WINDOW = (63, 7)      # signing rate from EOs signed 63..7 days before t (whose PI is out by t)


def eo_nowcast(eos: pl.DataFrame, first: date, last: date, t: datetime, strike: float) -> dict[str, Any]:
    """P(count > strike): EOs signed in the week and on public inspection by t, plus a Poisson remainder
    at the recent signing rate over the week's days not yet covered (future days count fully; past days
    by the share of EOs whose PI came later than t - day, from EOs filed before t)."""
    vis = eos.filter(pl.col("pi_ts") <= t)
    k = vis.filter((pl.col("signing_date") >= first) & (pl.col("signing_date") <= last)).height
    base = {"known_days": None, "unknown_days": None, "partial": float(k)}
    if k > strike:
        return {**base, "p_model": 1.0, "decided": True, "n_paths": 0, "clim_mean_rem": 0.0}
    td = t.date()
    lo, hi = td - timedelta(days=EO_RATE_WINDOW[0]), td - timedelta(days=EO_RATE_WINDOW[1])
    rate = vis.filter((pl.col("signing_date") >= lo) & (pl.col("signing_date") < hi)).height / (hi - lo).days
    hist = vis.filter(pl.col("signing_date") < hi)
    lag_days = ((hist["pi_ts"].dt.date() - hist["signing_date"]).dt.total_days()).to_numpy() if hist.height else np.array([3])
    expo = 0.0
    d = first
    while d <= last:
        age = (td - d).days
        expo += 1.0 if age < 0 else float((lag_days > age).mean()) if age < 30 else 0.0
        d += timedelta(days=1)
    lam = rate * expo
    need = math.floor(strike) + 1 - k                 # more EOs needed
    from math import exp, factorial
    p_less = sum(exp(-lam) * lam ** j / factorial(j) for j in range(max(need, 0)))
    elapsed = max(0, min((td - first).days, (last - first).days + 1))
    return {**base, "p_model": float(min(max(1 - p_less, 0.0), 1.0)), "decided": False, "n_paths": 0,
            "clim_mean_rem": lam, "known_days": elapsed, "unknown_days": (last - first).days + 1 - elapsed}


# ---- decision rows: nowcast + the quote of the last candle closed before the decision ----------------
def _candle_arrays(c: pl.DataFrame | None) -> dict[str, np.ndarray] | None:
    if c is None or not c.height:
        return None
    c = c.sort("end_ts")
    return {"end": c["end_ts"].dt.epoch("s").to_numpy(), "bid": c["bid_close"].to_numpy().astype(float),
            "ask": c["ask_close"].to_numpy().astype(float), "lo": c["p_low"].to_numpy().astype(float),
            "hi": c["p_high"].to_numpy().astype(float), "vol": c["volume"].to_numpy().astype(float),
            "pc": c["p_close"].to_numpy().astype(float)}


def quote_at(ca: dict[str, np.ndarray] | None, t: datetime) -> dict[str, Any]:
    """Bid/ask of the last candle that closed at or before ``t`` (Kalshi omits unchanged hours, so the
    last one carries forward), and its age."""
    if ca is None:
        return {"bid": None, "ask": None, "quote_age_h": None}
    j = int(np.searchsorted(ca["end"], t.timestamp(), side="right")) - 1
    if j < 0:
        return {"bid": None, "ask": None, "quote_age_h": None}
    b, a = ca["bid"][j], ca["ask"][j]
    return {"bid": None if np.isnan(b) else float(b), "ask": None if np.isnan(a) else float(a),
            "quote_age_h": (t.timestamp() - ca["end"][j]) / 3600}


def window_prints(ca: dict[str, np.ndarray] | None, t0: datetime, t1: datetime) -> dict[str, Any]:
    """Trade-price extremes and volume of the candles that start at/after ``t0`` and end by ``t1``
    (whole hours after the decision only: no print from before the order counts)."""
    if ca is None or t1 <= t0:
        return {"w_lo": None, "w_hi": None, "w_vol": 0.0}
    a, b = t0.timestamp() + 3600, t1.timestamp()
    m = (ca["end"] >= a) & (ca["end"] <= b)
    lo, hi = ca["lo"][m], ca["hi"][m]
    lo, hi = lo[~np.isnan(lo)], hi[~np.isnan(hi)]
    return {"w_lo": float(lo.min()) if len(lo) else None, "w_hi": float(hi.max()) if len(hi) else None,
            "w_vol": float(np.nansum(ca["vol"][m]))}


def cli_final(h: StationHistory, days: list[date], family: str, strike: float | None,
              rules: str | None) -> tuple[float | None, int | None]:
    """The settlement value rebuilt from the CLI (all days, as finally published) and its YES label."""
    if family == "rain_month":
        v = np.array([h.precip[h.ix(d)] if 0 <= h.ix(d) < len(h.precip) else np.nan for d in days])
        if np.isnan(v).any() or strike is None:
            return None, None
        tot = float(v.sum())
        return tot, int(tot > strike + 1e-9)
    terms = streak_terms(rules)
    v = np.array([h.tavg[h.ix(d)] if 0 <= h.ix(d) < len(h.tavg) else np.nan for d in days])
    if terms is None or np.isnan(v).any():
        return None, None
    run = int(max_run(v > terms[0])[0])
    return float(run), int(run >= terms[1])


def build(out: Path = OUT, *, log=print) -> pl.DataFrame:
    u = universe(out).filter(pl.col("strike_type").is_not_null())
    cli = pl.read_parquet(out / "cli.parquet")
    hist: dict[str, Any] = {st: StationHistory(g) for (st,), g in cli.group_by("station")}
    eos = pl.read_parquet(out / "eos.parquet") if (out / "eos.parquet").exists() else None
    if eos is not None:
        hist["FR"] = eos
    cand = load_candles(out)
    arrays = {tk: _candle_arrays(g) for (tk,), g in cand.group_by("ticker")} if cand.height else {}
    rows, checks = [], []
    for r in u.iter_rows(named=True):
        st, fam = r["station"], r["family"]
        h = hist.get(st)
        if h is None or st not in STATION_TZ:
            continue
        days = [r["p_first"] + timedelta(days=i) for i in range((r["p_last"] - r["p_first"]).days + 1)]
        p0, p1 = lst_bounds(st, r["p_first"], r["p_last"])
        y = int(r["result"] == "yes")
        if fam == "eo_week":
            n_eo = h.filter((pl.col("signing_date") >= days[0]) & (pl.col("signing_date") <= days[-1])).height
            val, y_cli = float(n_eo), int(n_eo > r["floor_strike"])
        else:
            val, y_cli = cli_final(h, days, fam, r["floor_strike"], r["rules_primary"])
        checks.append({"ticker": r["ticker"], "event_ticker": r["event_ticker"], "family": fam, "station": st,
                       "split": r["split"], "y": y, "y_cli": y_cli, "cli_value": val,
                       "expiration_value": r["expiration_value"]})
        ca = arrays.get(r["ticker"])
        dts = decision_times(p0, p1)
        settle = r["settlement_ts"] or r["close_time"]
        for i, (lab, t) in enumerate(dts):
            if not (r["open_time"] <= t < r["close_time"]):
                continue
            t_next = min(dts[i + 1][1] if i + 1 < len(dts) else p1, r["close_time"])
            if fam == "rain_month":
                nc = rain_nowcast(h, days, t, r["floor_strike"])
            elif fam == "eo_week":
                nc = eo_nowcast(h, days[0], days[-1], t, r["floor_strike"])
            else:
                terms = streak_terms(r["rules_primary"])
                if terms is None:
                    continue
                nc = streak_nowcast(h, days, t, terms[0], terms[1])
            rows.append({"ticker": r["ticker"], "event_ticker": r["event_ticker"], "series_ticker": r["series_ticker"],
                         "family": fam, "station": st, "split": r["split"], "decision": lab, "t": t,
                         "t_next": t_next, "date": t.date(), "strike": r["floor_strike"], "y": y,
                         "settle_ts": settle, "fee_type": r["fee_type"], "fee_multiplier": r["fee_multiplier"],
                         **nc, **quote_at(ca, t), **window_prints(ca, t, t_next),
                         "has_candles": ca is not None})
    df = pl.DataFrame(rows, infer_schema_length=None)
    ck = pl.DataFrame(checks, infer_schema_length=None)
    _save(df, out / "rows.parquet")
    _save(ck, out / "settle_check.parquet")
    log(f"build: {df.height} decision rows, {ck.height} markets checked", flush=True)
    return df



# ---- statistics ----------------------------------------------------------------------------------------
def cluster_se(x: np.ndarray, *groups: np.ndarray) -> float:
    """Standard error of mean(x) clustered by one key, or two-way (Cameron-Gelbach-Miller) by two:
    V = V_a + V_b - V_ab (falls back to max(V_a, V_b) if that is not positive)."""
    n = len(x)
    if n < 2:
        return float("nan")
    e = x - x.mean()

    def v(keys: np.ndarray) -> float:
        _, inv = np.unique(keys, return_inverse=True)
        sums = np.bincount(inv, weights=e)
        return float((sums ** 2).sum()) / n ** 2
    if len(groups) == 1:
        return math.sqrt(v(groups[0]))
    a, b = groups
    ab = np.array([f"{p}|{q}" for p, q in zip(a, b, strict=True)])
    va, vb, vab = v(a), v(b), v(ab)
    tot = va + vb - vab
    return math.sqrt(tot if tot > 0 else max(va, vb))


def summarize(df: pl.DataFrame, col: str = "pnl_c") -> dict[str, Any]:
    """Mean per contract (cents) with t by event, by date and two-way."""
    if not df.height:
        return {"n": 0, "events": 0, "mean_c": None, "t_event": None, "t_date": None, "t_2way": None}
    x = df[col].to_numpy().astype(float)
    ev = df["event_ticker"].to_numpy().astype(str)
    dt = df["date"].cast(pl.String).to_numpy().astype(str)
    m = float(x.mean())

    def t(se: float, *keys: np.ndarray) -> float | None:
        if any(len(np.unique(k)) < 2 for k in keys):          # one cluster: no clustered error
            return None
        return round(m / se, 2) if se and se > 1e-9 and math.isfinite(se) else None
    return {"n": len(x), "events": len(np.unique(ev)), "mean_c": round(m, 2),
            "t_event": t(cluster_se(x, ev), ev), "t_date": t(cluster_se(x, dt), dt),
            "t_2way": t(cluster_se(x, ev, dt), ev, dt),
            "win_pct": round(100 * float((x > 0).mean()), 1)}


def taker_fee(n: int, price: float, mult: float | None) -> float:
    """0.07 x mult x n x P(1-P), rounded up to the cent per order (dollars)."""
    raw = 0.07 * (mult or 1.0) * n * price * (1 - price)
    return math.ceil(round(raw * 100, 9)) / 100


STAKE_USD = 2.0          # each order buys ~$2 of contracts (fee rounding is per order) ...
MAX_CONTRACTS = 20       # ... and at most 20 contracts (displayed size is not in the candles)
MAX_TAKE_SPREAD = 0.10   # only cross quotes at most this wide
LATE = ("f75", "f90", "final")


def taker_trades(rows: pl.DataFrame, theta: float, *, use: str = "p_model") -> pl.DataFrame:
    """Buy YES at the ask when p - ask >= theta, NO at 1 - bid when bid - p >= theta; ~$2 order, taker
    fee rounded up per order, held to settlement. pnl_c is cents per contract net of the fee."""
    r = rows.filter(pl.col(use).is_not_null() & pl.col("bid").is_not_null() & pl.col("ask").is_not_null()
                    & (pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("ask") - pl.col("bid") <= MAX_TAKE_SPREAD + 1e-9))
    r = r.with_columns(
        pl.when(pl.col(use) - pl.col("ask") >= theta - 1e-9).then(pl.lit("yes"))
        .when(pl.col("bid") - pl.col(use) >= theta - 1e-9).then(pl.lit("no")).otherwise(None).alias("side"))
    r = r.filter(pl.col("side").is_not_null()).with_columns(
        pl.when(pl.col("side") == "yes").then(pl.col("ask")).otherwise(1 - pl.col("bid")).alias("price"),
        pl.when(pl.col("side") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y")).alias("payoff"))
    if not r.height:
        return r.with_columns(pl.lit(None, pl.Int64).alias("n"), pl.lit(None, pl.Float64).alias("fee"),
                              pl.lit(None, pl.Float64).alias("pnl_c"))
    n = [min(MAX_CONTRACTS, max(1, int(STAKE_USD // p))) for p in r["price"].to_list()]
    fee = [taker_fee(k, p, m) for k, p, m in zip(n, r["price"].to_list(), r["fee_multiplier"].to_list(), strict=True)]
    r = r.with_columns(pl.Series("n", n), pl.Series("fee", fee))
    return r.with_columns((100 * (pl.col("payoff") - pl.col("price") - pl.col("fee") / pl.col("n"))).alias("pnl_c"))


def maker_fee(price: float, fee_type: str | None) -> float:
    """Per contract; only series whose fee type carries maker fees pay one."""
    return 0.0175 * price * (1 - price) if fee_type and fee_type.endswith("_with_maker_fees") else 0.0


def maker_fills(rows: pl.DataFrame, margin: float, *, use: str = "p_model") -> pl.DataFrame:
    """Rest a YES bid at floor(p - m) and a NO bid at floor(1 - p - m) from the decision to the next one
    (post-only: at most one tick inside the opposite quote). A bid fills only if a later print trades
    strictly through it (hourly candle trade low < YES bid; high > 1 - NO bid), in hours that start
    after the decision. Filled at our price, held to settlement; pnl_c is cents per contract."""
    r = rows.filter(pl.col(use).is_not_null())
    out = []
    for x in r.iter_rows(named=True):
        p = x[use]
        yb = math.floor(round((p - margin) * 100, 6)) / 100
        if x["ask"] is not None:
            yb = min(yb, round(x["ask"] - 0.01, 2))
        nb = math.floor(round((1 - p - margin) * 100, 6)) / 100
        if x["bid"] is not None:
            nb = min(nb, round(1 - x["bid"] - 0.01, 2))
        if yb >= 0.01 and x["w_lo"] is not None and x["w_lo"] < yb - 1e-9:
            out.append({**x, "side": "yes", "price": yb, "payoff": x["y"]})
        if nb >= 0.01 and x["w_hi"] is not None and x["w_hi"] > 1 - nb + 1e-9:
            out.append({**x, "side": "no", "price": nb, "payoff": 1 - x["y"]})
    if not out:
        return pl.DataFrame()
    f = pl.DataFrame(out, infer_schema_length=None)
    fee = [maker_fee(pr, ft) for pr, ft in zip(f["price"].to_list(), f["fee_type"].to_list(), strict=True)]
    return f.with_columns(pl.Series("fee_pc", fee), pl.lit(1).alias("n")).with_columns(
        (100 * (pl.col("payoff") - pl.col("price") - pl.col("fee_pc"))).alias("pnl_c"))



# ---- does the nowcast add information to the price? ---------------------------------------------------
def info_table(rows: pl.DataFrame) -> pl.DataFrame:
    """Per family x split x decision, undecided rows with a quote <= 10c wide and a mid in 5-95c (deep
    tails are left out: there the mid is not a price and the favourite-longshot bias alone makes a
    slope): Brier of mid vs model, and the slope of (y - mid) on (p - mid) (> 0: the price under-reacts
    to what the nowcast knows), t two-way clustered by event and date."""
    r = rows.filter(pl.col("p_model").is_not_null() & ~pl.col("decided") & pl.col("bid").is_not_null()
                    & pl.col("ask").is_not_null() & (pl.col("ask") > pl.col("bid"))
                    & (pl.col("ask") - pl.col("bid") <= MAX_TAKE_SPREAD + 1e-9)
                    ).with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid")).filter(
                        pl.col("mid").is_between(0.05, 0.95))
    out = []
    keys = [("family", "split", "decision"), ("family", "split")]
    for ks in keys:
        for k, g in r.group_by(list(ks)):
            x = (g["p_model"] - g["mid"]).to_numpy()
            yy = (g["y"] - g["mid"]).to_numpy()
            ev, dt = g["event_ticker"].to_numpy().astype(str), g["date"].cast(pl.String).to_numpy().astype(str)
            sxx = float(((x - x.mean()) ** 2).sum())
            beta = float(((x - x.mean()) * (yy - yy.mean())).sum() / sxx) if sxx > 0 else None
            se = None
            if beta is not None and len(x) >= 30 and len(np.unique(ev)) > 1 and len(np.unique(dt)) > 1:
                resid = yy - yy.mean() - beta * (x - x.mean())
                score = (x - x.mean()) * resid             # sandwich: var(beta) = var(sum score) / sxx^2
                se = cluster_se(score, ev, dt) * len(x) / sxx
            out.append({**dict(zip(ks, k, strict=True)), "decision": k[2] if len(ks) == 3 else "all", "rows": g.height,
                        "events": g["event_ticker"].n_unique(),
                        "brier_mid": round(float(((g["y"] - g["mid"]) ** 2).mean()), 4),
                        "brier_model": round(float(((g["y"] - g["p_model"]) ** 2).mean()), 4),
                        "mean_abs_gap_c": round(100 * float(np.abs(x).mean()), 2),
                        "slope": None if beta is None else round(beta, 3),
                        "t_slope_2way": None if not se else round(beta / se, 2)})
    return pl.DataFrame(out, infer_schema_length=None).sort("family", "split", "decision")



# ---- price-only control: the dev-recalibrated mid (no partial-sum information) ----------------------
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


def _logistic_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    a, b = 0.0, 1.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-(a + b * x)))
        w = p * (1 - p) + 1e-9
        g = np.array([(y - p).sum(), ((y - p) * x).sum()])
        hm = np.array([[w.sum(), (w * x).sum()], [(w * x).sum(), (w * x * x).sum()]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(hm, g)
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


def fit_mid_calibration(rows: pl.DataFrame) -> tuple[float, float]:
    """Logistic fit y ~ a + b logit(mid) on quoted rows (Newton steps); used on dev rows only."""
    r = rows.filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("ask") > pl.col("bid")))
    return _logistic_fit(_logit(((r["bid"] + r["ask"]) / 2).to_numpy()), r["y"].to_numpy().astype(float))


CAL_FOLDS = 5


def with_calibrated_nowcast(rows: pl.DataFrame) -> pl.DataFrame:
    """``p_cal``: the nowcast's probability recalibrated per family by a logistic fit y ~ a + b logit(p)
    on dev rows (the "residual distribution fit on dev"). Dev rows get cross-fitted values (5 folds by
    event, each fitted on the other folds); holdout rows use the fit on all dev rows. Decided rows keep
    p = 0 / 1. The fit needs no market data."""
    out = []
    for (fam,), g in rows.group_by("family"):
        g = g.with_row_index("_i")
        und = g.filter(pl.col("p_model").is_not_null() & ~pl.col("decided"))
        dev = und.filter(pl.col("split") == "dev")
        cal = np.full(g.height, np.nan)
        if dev.height >= 50:
            evs = sorted(dev["event_ticker"].unique().to_list())
            fold = {e: i % CAL_FOLDS for i, e in enumerate(evs)}
            for k in range(CAL_FOLDS):
                fk = pl.col("event_ticker").replace_strict(fold, return_dtype=pl.Int64)
                tr, te = dev.filter(fk != k), dev.filter(fk == k)
                if not te.height or tr.height < 30:
                    continue
                a, b = _logistic_fit(_logit(tr["p_model"].to_numpy()), tr["y"].to_numpy().astype(float))
                cal[te["_i"].to_numpy()] = 1 / (1 + np.exp(-(a + b * _logit(te["p_model"].to_numpy()))))
            a, b = _logistic_fit(_logit(dev["p_model"].to_numpy()), dev["y"].to_numpy().astype(float))
            ho = und.filter(pl.col("split") != "dev")
            if ho.height:
                cal[ho["_i"].to_numpy()] = 1 / (1 + np.exp(-(a + b * _logit(ho["p_model"].to_numpy()))))
        g = g.with_columns(pl.Series("_cal", cal, nan_to_null=True)).with_columns(
            pl.when(pl.col("decided")).then(pl.col("p_model")).otherwise(pl.col("_cal")).alias("p_cal"))
        out.append(g.drop("_i", "_cal"))
    return pl.concat(out, how="diagonal_relaxed")


def with_control(rows: pl.DataFrame, ab: tuple[float, float]) -> pl.DataFrame:
    a, b = ab
    mid = (pl.col("bid") + pl.col("ask")) / 2
    lg = mid.clip(0.005, 0.995).log() - (1 - mid.clip(0.005, 0.995)).log()
    return rows.with_columns(pl.when(pl.col("bid").is_not_null() & pl.col("ask").is_not_null())
                             .then(1 / (1 + (-(a + b * lg)).exp())).otherwise(None).alias("p_ctrl"))


# ---- pre-specified primary rule (dev only), then the holdout ------------------------------------------
TAKER_THETAS = (0.03, 0.05, 0.08, 0.12, 0.16)
MAKER_MARGINS = (0.03, 0.05, 0.08, 0.12)
FAMILY_SETS = {"rain": ("rain_month",), "temp": ("temp_avg_week",), "eo": ("eo_week",), "all": TESTED}
DECISION_SETS = {"all": ("f25", "f50", "f75", "f90", "final"), "late": LATE}
MIN_DEV_TRADES, MIN_DEV_EVENTS = 40, 10
FALLBACK_PRIMARY = {"nowcast": "p_model", "kind": "taker", "param": 0.08, "families": "all", "decisions": "late"}


def _subset(rows: pl.DataFrame, fams: str, decs: str) -> pl.DataFrame:
    return rows.filter(pl.col("family").is_in(list(FAMILY_SETS[fams])) & pl.col("decision").is_in(list(DECISION_SETS[decs])))


def run_cell(rows: pl.DataFrame, kind: str, param: float, fams: str, decs: str, use: str = "p_model") -> pl.DataFrame:
    sub = _subset(rows, fams, decs)
    return taker_trades(sub, param, use=use) if kind == "taker" else maker_fills(sub, param, use=use)


NOWCASTS = ("p_model", "p_cal")


def grid(rows: pl.DataFrame) -> pl.DataFrame:
    out = []
    for use in NOWCASTS:
        for fams in FAMILY_SETS:
            for decs in DECISION_SETS:
                for kind, params in (("taker", TAKER_THETAS), ("maker", MAKER_MARGINS)):
                    for prm in params:
                        tr = run_cell(rows, kind, prm, fams, decs, use=use)
                        out.append({"nowcast": use, "kind": kind, "param": prm, "families": fams, "decisions": decs,
                                    **(summarize(tr) if tr.height else summarize(pl.DataFrame()))})
    return pl.DataFrame(out, infer_schema_length=None)


def select_primary(rows: pl.DataFrame, out: Path = OUT) -> dict[str, Any]:
    """Freeze the primary rule from dev rows only: the highest two-way t among cells with at least
    MIN_DEV_TRADES trades on MIN_DEV_EVENTS events; FALLBACK_PRIMARY (declared in code) if none has t > 0."""
    dev = rows.filter(pl.col("split") == "dev")
    g = grid(dev)
    _save(g, out / "grid_dev.parquet")
    ok = g.filter((pl.col("n") >= MIN_DEV_TRADES) & (pl.col("events") >= MIN_DEV_EVENTS)
                  & pl.col("t_2way").is_not_null()).sort("t_2way", descending=True)
    if ok.height and ok["t_2way"][0] > 0:
        best = ok.row(0, named=True)
        prim = {k: best[k] for k in ("nowcast", "kind", "param", "families", "decisions")}
        prim["dev"] = {k: best[k] for k in ("n", "events", "mean_c", "t_event", "t_date", "t_2way")}
    else:
        prim = dict(FALLBACK_PRIMARY, dev=None)
    prim["frozen_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    prim["rule"] = f"highest dev t_2way, >= {MIN_DEV_TRADES} trades / {MIN_DEV_EVENTS} events"
    (out / "primary.json").write_text(json.dumps(prim, indent=1))
    return prim



# ---- capped-book ledgers and Sharpe / vol (albert.research.vol_metrics) --------------------------------
BANKROLL, EVENT_CAP, OPEN_CAP = 100.0, 5.0, 20.0


def _fill_time(ca: dict[str, np.ndarray] | None, t0: datetime, t1: datetime, side: str, price: float) -> datetime:
    """End of the first whole hour after ``t0`` whose prints trade through our bid."""
    if ca is not None:
        a, b = t0.timestamp() + 3600, t1.timestamp()
        for j in np.flatnonzero((ca["end"] >= a) & (ca["end"] <= b)):
            if (side == "yes" and ca["lo"][j] < price - 1e-9) or (side == "no" and ca["hi"][j] > 1 - price + 1e-9):
                return datetime.fromtimestamp(float(ca["end"][j]), UTC)
    return t0 + timedelta(hours=1)


def ledger(trades: pl.DataFrame, kind: str, name: str, split: str, arrays: dict | None = None) -> pl.DataFrame:
    """Per-leg ledger of a $100 book: ~$2 per order, <= $5 cost per event, <= $20 open; trades that do
    not fit are shrunk to what fits or skipped. Maker size is capped by the fill hours' volume."""
    if not trades.height:
        return pl.DataFrame()
    legs = []
    for x in trades.iter_rows(named=True):
        price = float(x["price"])
        if kind == "taker":
            t_in, n, fee_pc = x["t"], int(x["n"]), float(x["fee"]) / int(x["n"])
        else:
            t_in = _fill_time((arrays or {}).get(x["ticker"]), x["t"], x["t_next"], x["side"], price)
            n = min(MAX_CONTRACTS, max(1, int(STAKE_USD // price)))
            if x.get("w_vol") and x["w_vol"] >= 1:
                n = min(n, int(x["w_vol"]))
            fee_pc = float(x["fee_pc"])
        legs.append({"t_in": t_in, "n": n, "unit": price + fee_pc, "fee_pc": fee_pc, **{k: x[k] for k in (
            "ticker", "event_ticker", "side", "settle_ts", "payoff", "decision", "family")}, "price": price})
    legs.sort(key=lambda z: z["t_in"])
    open_pos: list[dict[str, Any]] = []
    ev_cost: dict[str, float] = {}
    rows = []
    for i, z in enumerate(legs):
        open_pos = [o for o in open_pos if o["settle_ts"] > z["t_in"]]
        room = min(OPEN_CAP - sum(o["cost"] for o in open_pos), EVENT_CAP - ev_cost.get(z["event_ticker"], 0.0))
        n = min(z["n"], int(room // z["unit"])) if room > 0 else 0
        if kind == "taker" and n < z["n"] and n >= 1:      # a smaller taker order pays its own rounded fee
            fee = taker_fee(n, z["price"], None) / n
            n = min(n, int(room // (z["price"] + fee)))
            z = {**z, "unit": z["price"] + fee}
        if n < 1:
            continue
        cost = n * z["unit"]
        open_pos.append({"settle_ts": z["settle_ts"], "cost": cost})
        ev_cost[z["event_ticker"]] = ev_cost.get(z["event_ticker"], 0.0) + cost
        rows.append({"strategy": f"counts_{kind}", "variant": name, "split": split, "struct_id": f"{z['ticker']}#{i}",
                     "event_ticker": z["event_ticker"], "ticker": z["ticker"], "side": z["side"], "n": n,
                     "t_in": z["t_in"], "cost_usd": cost, "settle_ts": z["settle_ts"],
                     "payoff_usd": n * float(z["payoff"]), "decision": z["decision"], "family": z["family"]})
    return pl.DataFrame(rows, infer_schema_length=None)


def candle_marks(out: Path = OUT) -> pl.DataFrame:
    c = load_candles(out)
    two = c["bid_close"].is_not_null() & c["ask_close"].is_not_null() & (c["ask_close"] > c["bid_close"])
    mid = pl.when(two).then((pl.col("bid_close") + pl.col("ask_close")) / 2).otherwise(pl.col("p_close"))
    return (c.with_columns(mid.alias("yes_mid")).filter(pl.col("yes_mid").is_not_null())
            .select("ticker", pl.col("end_ts").alias("ts"), "yes_mid").sort("ticker", "ts"))


def ledger_metrics(led: pl.DataFrame, marks: pl.DataFrame) -> dict[str, Any]:
    from albert.research import vol_metrics as vm

    if not led.height:
        return {}
    m = vm.ledger_metrics(led, marks.filter(pl.col("ticker").is_in(led["ticker"].unique().implode())))
    return {k: m.get(k) for k in ("positions", "events", "pnl_usd", "usd_per_week", "ann_vol_pct", "sharpe",
                                  "sharpe_ci90", "p_sharpe_pos", "sharpe_ex_top5", "usd_per_week_ex_top5",
                                  "max_dd_usd", "weeks_positive_pct", "first_day", "last_day", "days")}



# ---- the study: info, frozen primary on dev, holdout, ledgers -----------------------------------------
def _arrays(out: Path = OUT) -> dict[str, dict[str, np.ndarray] | None]:
    cand = load_candles(out)
    return {tk: _candle_arrays(g) for (tk,), g in cand.group_by("ticker")} if cand.height else {}


def decided_table(rows: pl.DataFrame) -> pl.DataFrame:
    """Rows whose published partial sum already fixes the outcome while the market is still open: how
    far is the quote from 0/1, and what does taking it earn?"""
    d = rows.filter(pl.col("decided") & pl.col("bid").is_not_null() & pl.col("ask").is_not_null())
    if not d.height:
        return pl.DataFrame()
    tr = taker_trades(d, 0.01)
    out = []
    for (fam, split), g in d.group_by("family", "split"):
        t = tr.filter((pl.col("family") == fam) & (pl.col("split") == split))
        yes = g.filter(pl.col("p_model") == 1.0)
        out.append({"family": fam, "split": split, "rows": g.height, "decided_yes": yes.height,
                    "label_agrees": round(float((g["y"] == g["p_model"]).mean()), 4),
                    "median_ask_if_yes": yes["ask"].median(), "median_quote_age_h": g["quote_age_h"].median(),
                    **{f"take_{k}": v for k, v in summarize(t).items()}})
    return pl.DataFrame(out, infer_schema_length=None)


def evaluate(out: Path = OUT, *, log=print) -> dict[str, Any]:
    rows = pl.read_parquet(out / "rows.parquet")
    ab = fit_mid_calibration(rows.filter(pl.col("split") == "dev"))
    rows = with_calibrated_nowcast(with_control(rows, ab))
    info = info_table(rows)
    info.write_csv(out / "info.csv")
    dec = decided_table(rows)
    if dec.height:
        dec.write_csv(out / "decided.csv")
    prim_path = out / "primary.json"
    if not prim_path.exists():                       # frozen once, from dev rows only
        select_primary(rows, out)
    prim = json.loads(prim_path.read_text())
    pl.read_parquet(out / "grid_dev.parquet").write_csv(out / "grid_dev.csv")
    hold = grid(rows.filter(pl.col("split") == "holdout"))   # every cell, for transparency (not for choosing)
    hold.write_csv(out / "grid_holdout.csv")
    arrays = _arrays(out)
    marks = candle_marks(out)
    led_dir = out / "ledgers"
    led_dir.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {"primary": prim, "mid_calibration_dev": {"a": ab[0], "b": ab[1]}, "splits": {}}
    by_dec = []
    for split in ("dev", "holdout"):
        sub = rows.filter(pl.col("split") == split)
        tr = run_cell(sub, prim["kind"], prim["param"], prim["families"], prim["decisions"], use=prim["nowcast"])
        stats = summarize(tr) if tr.height else summarize(pl.DataFrame())
        led = ledger(tr, prim["kind"], "primary", split, arrays)
        if led.height:
            led.write_parquet(led_dir / f"ledger_primary_{split}.parquet")
        res["splits"][split] = {"trades": stats, "book": ledger_metrics(led, marks) if led.height else {}}
        for (d,), g in (tr.group_by("decision") if tr.height else []):
            by_dec.append({"split": split, "decision": d, **summarize(g)})
        for (f,), g in (tr.group_by("family") if tr.height else []):
            by_dec.append({"split": split, "decision": f"family={f}", **summarize(g)})
        for (dd,), g in (tr.group_by("decided") if tr.height else []):
            by_dec.append({"split": split, "decision": f"decided={dd}", **summarize(g)})
        ctl = run_cell(sub, prim["kind"], prim["param"], prim["families"], prim["decisions"], use="p_ctrl")
        res["splits"][split]["control_recalibrated_mid"] = summarize(ctl) if ctl.height else summarize(pl.DataFrame())
    pl.DataFrame(by_dec, infer_schema_length=None).sort("split", "decision").write_csv(out / "primary_breakdown.csv")
    # both splits in one book (continuous calendar)
    allr = run_cell(rows, prim["kind"], prim["param"], prim["families"], prim["decisions"], use=prim["nowcast"])
    led = ledger(allr, prim["kind"], "primary", "all", arrays)
    if led.height:
        led.write_parquet(led_dir / "ledger_primary_all.parquet")
        res["splits"]["all"] = {"trades": summarize(allr), "book": ledger_metrics(led, marks)}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    met = [{"strategy": f"counts_{prim['kind']}_primary", "split": sp, **{k: (str(v) if isinstance(v, list) else v)
                                                                       for k, v in (val.get("book") or {}).items()}}
           for sp, val in res["splits"].items()]
    pl.DataFrame(met, infer_schema_length=None).write_csv(out / "metrics.csv")
    log(json.dumps(res, indent=1, default=str))
    return res



# ---- report ----------------------------------------------------------------------------------------
VERDICT = """**No tradable edge; the hypothesis is rejected.** Kalshi prices the published partial sum fully:

* *Decided brackets.* 935 dev decision rows (rain) had a strike already passed by the issued CLI days
  while the market was still open; the YES ask was $1.00 on all but 7 (median and p10 bid 99c).
  Nothing is left to take. On the weekly streak series the CLI proxy disagrees with The Weather Company's daily
  mean on 8.5% of "decided" rows, and taking them loses (-11.5c, 20 trades).
* *Undecided brackets* (quotes <= 10c, mid 5-95c): for rain and EOs the mid's Brier beats the nowcast
  at every late decision (rain f90 0.111 vs 0.151, final 0.066 vs 0.111). The slope of (y - mid) on
  (nowcast - mid) is 0.13 (t 1.6) dev and 0.11 (t 0.4) holdout for rain: the market already holds the
  partial sum plus forecasts and same-day observations. The streak nowcast scores better than the mid
  on the few priced rows (32 in holdout), but those are near-certain NOs 2-5c from the edge, which
  0-3 holdout taker trades can clear after the fee.
* *Primary rule* (frozen on dev 2026-11-17 19:48 UTC: maker, NO/YES bids 12c around the raw nowcast,
  rain, decisions at 75/90%/final): +4.9c per fill (t 3.4, 524 fills, 112 events) dev -> +1.6c (t 0.6,
  69 fills, 15 events, 3 decision dates) holdout. The same maker around the dev-recalibrated mid (no
  partial-sum information) earns +2.7c (t 1.9) dev and +6.1c (t 1.9) holdout, so whatever the maker
  earns is the generic wide-quote/YES-overpricing pattern already found in `weather_maker`, not the
  nowcast. Book ($100, <= $5/event, <= $20 open): dev Sharpe 0.54 [-0.39, 1.50], ex top-5 -1.06,
  $0.27/week; holdout 9 days of positions, Sharpe 1.41 (no CI), ex top-5 -14.6, $1.85/week.
* *Takers:* no dev cell is significant (best rain-late +2.1c, t 1.05; every streak, EO and pooled
  cell negative). Rain-late taker cells show +3.5-3.9c (t 2.3, ~30 trades) in the holdout; they were
  not the frozen rule, sit on three decision dates, and four settlements carry them.
* *Executive orders:* the Federal Register's public-inspection time is a median 4 days after the
  signing the market sees on whitehouse.gov, so the as-of nowcast is structurally late; it loses
  everywhere (taker -3 to -10c, maker -1 to -10c per contract in both splits).

Nothing here is worth forward paper trading. Holdout power is low by construction (monthly rain:
one month; weekly streak series stopped after 2026-10-12)."""


def _md(df: pl.DataFrame, cols: list[str] | None = None) -> str:
    if df is None or not df.height:
        return "(none)\n"
    df = df.select(cols) if cols else df
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for r in df.iter_rows():
        lines.append("| " + " | ".join("" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines) + "\n"


def write_report(out: Path = OUT) -> Path:
    tri = pl.read_parquet(out / "triage.parquet").filter(~pl.col("financial"))
    fam = (tri.group_by("family").agg(pl.len().alias("series"), pl.col("events_dev").sum(), pl.col("events_holdout").sum(),
                                      pl.col("tested_events_dev").sum().alias("tested_dev"),
                                      pl.col("tested_events_holdout").sum().alias("tested_holdout"),
                                      pl.col("volume").sum().round(0).alias("volume_usd"),
                                      pl.col("component_source").first(), pl.col("as_of_feasible").first())
           .sort("as_of_feasible", "events_dev", descending=True))
    robots = pl.DataFrame(json.loads((out / "robots.json").read_text()))
    ck = pl.read_parquet(out / "settle_check.parquet")
    agree = ck.group_by("family", "split").agg(pl.len().alias("markets"), pl.col("y_cli").is_null().sum().alias("no_cli"),
                                               (pl.col("y") == pl.col("y_cli")).mean().round(4).alias("cli_label_agrees"))
    info = pl.read_csv(out / "info.csv")
    dec = pl.read_csv(out / "decided.csv") if (out / "decided.csv").exists() else pl.DataFrame()
    gdev = pl.read_csv(out / "grid_dev.csv")
    ghold = pl.read_csv(out / "grid_holdout.csv")
    res = json.loads((out / "results.json").read_text())
    brk = pl.read_csv(out / "primary_breakdown.csv")
    prim = res["primary"]
    book = []
    for sp, v in res["splits"].items():
        b = v.get("book") or {}
        tr = v.get("trades") or {}
        ctl = v.get("control_recalibrated_mid") or {}
        book.append({"split": sp, "trades": tr.get("n"), "events": tr.get("events"), "c_per_contract": tr.get("mean_c"),
                     "t_2way": tr.get("t_2way"), "t_event": tr.get("t_event"), "t_date": tr.get("t_date"),
                     "control_c": ctl.get("mean_c"), "control_t2": ctl.get("t_2way"),
                     "book_positions": b.get("positions"), "usd_per_week": b.get("usd_per_week"),
                     "sharpe": b.get("sharpe"), "sharpe_ci90": str(b.get("sharpe_ci90")),
                     "sharpe_ex_top5": b.get("sharpe_ex_top5"), "ann_vol_pct": b.get("ann_vol_pct"),
                     "max_dd_usd": b.get("max_dd_usd"), "days": b.get("days")})
    key = ["nowcast", "kind", "param", "families", "decisions", "n", "events", "mean_c", "t_event", "t_date", "t_2way"]
    txt = f"""# Accumulating-quantity ladders: partial-sum nowcast vs the Kalshi price

Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `python -m albert.research.counts report`
(module `src/albert/research/counts.py`; tests `tests/test_counts.py`).

## Verdict

{VERDICT}

## 1. Candidate series and triage

Candidates come from today's Kalshi `/series` (16,485 series): ticker families that settle on a total,
average or count over a week / month / season. Exclusion is decided by the series' Kalshi `category`
and title (`albert.categories.kalshi_series_is_financial`: Financials, Economics, Crypto, Commodities,
Companies, plus money words) and by the settlement source named in `/series` and the market rules.
KXTSAW / KXTSAD (TSA) are category *Economics*: excluded and reported only. Every settled market of each
remaining series was listed from `/historical/markets` and `/markets?status=settled` (1 request/s).
Split by the period's last local day: dev <= 2026-09-30, holdout 2026-10-01..11-16.

{_md(fam)}
Per series: `data/research/counts/triage.csv`.

## 2. Robots checks (fetched {robots['checked_utc'][0] if robots.height else ''})

{_md(robots.select("host", "path", "http_status", "allow_*", "allow_anthropic-ai", "allow_ClaudeBot", "allow_Claude-Web", "allowed"))}
Used: IEM (`/json/cli.py`), the Federal Register API (`/api/v1/documents.json`) and the Kalshi public
API only. Not used: rollcall.com (blocks anthropic-ai), truthsocial.com (robots.txt 403), x.com, SPC and
congress.gov (block these agents); TSA, data.ny.gov, whitehouse.gov and weather.com allow us but feed
no tested series (TSA weekly is Economics; TWC's archive needs an API key).

## 3. Components rebuilt as of each decision

IEM `json/cli.py` gives every NWS CLI day with its product id, whose prefix is the issuance time
(median 29-33 h after the day's UTC midnight). A day counts as known at decision `t` only if its CLI was
issued at or before `t`. The settlement value rebuilt from the CLI agrees with Kalshi's result:

{_md(agree)}
(Rain series name The Weather Company as source but match the NWS CLI on every market; the weekly
streak series use TWC's daily average, which (high+low)/2 from the CLI misses on 8 of 305 dev markets.)

Nowcast: known partial (rain: sum of issued days; streak: the run so far) plus the remainder from 225
analog paths: the same calendar days in each of the 15 previous years, shifted -7..+7 days (all public
long before the decision); streak paths also carry the latest known daily-mean anomaly forward at
{ANOM_RHO}/day. A strike already passed by the published partial sum is decided (p = 1). Executive
orders: EOs signed in the week whose Federal Register public-inspection time is at or before `t`, plus a
Poisson remainder at the signing rate of the previous 63..7 days over the week's not-yet-visible days.
`p_cal` = the nowcast recalibrated per family on dev (logistic y ~ a + b logit p; dev rows cross-fitted
over 5 event folds, holdout rows use the all-dev fit): the "residual distribution fit on dev".

Decisions: 25/50/75/90% through the period and the final day (10 h before the period's end), on the
hour; quote = the last hourly candle that closed at or before `t` (market open at `t`).

## 4. Does the nowcast add information to the price? (undecided rows, quote <= 10c wide, mid 5-95c)

{_md(info)}
`slope` = OLS of (y - mid) on (p_model - mid); t two-way clustered by event and date, shown only with
>= 30 rows on more than one event and date. The streak rows are mostly near-certain NOs (the large holdout t
comes from a residual variance near zero), so they say little about a tradable gap.

Partial sum already decides the market while it is still open:

{_md(dec)}

## 5. Primary rule, frozen on dev before the holdout was evaluated

`primary.json`: {json.dumps({k: prim.get(k) for k in ("nowcast", "kind", "param", "families", "decisions", "rule", "frozen_utc")})}

Taker: buy YES at the ask if p - ask >= theta, NO at 1 - bid if bid - p >= theta, quotes <= 10c wide,
~$2 orders, fee 0.07 x mult x n x P(1-P) rounded up per order, held to settlement. Maker: YES bid at
floor(p - m), NO bid at floor(1 - p - m), post-only, from the decision to the next one; filled only if a
print in a later whole hour trades strictly through (hourly candle trade low/high), held to settlement;
these series are fee type `quadratic` (no maker fee).

Top dev cells (selection set):

{_md(gdev.filter(pl.col("n") >= MIN_DEV_TRADES).sort("t_2way", descending=True, nulls_last=True).head(12), key)}

**Primary, dev -> holdout** (per contract; book = $100, ~$2 per order, <= $5 per event, <= $20 open,
daily MTM at the hourly-candle mid, `vol_metrics.ledger_metrics`; control = same rule on the
dev-recalibrated mid, i.e. no partial-sum information):

{_md(pl.DataFrame(book, infer_schema_length=None))}
Breakdown:

{_md(brk)}

Holdout, every cell (for transparency only; not used to choose):

{_md(ghold.sort("t_2way", descending=True, nulls_last=True), key)}

Files: `rows.parquet` (decision rows), `grid_dev.csv`, `grid_holdout.csv`, `results.json`,
`ledgers/ledger_primary_{{dev,holdout,all}}.parquet`, `candles/`, `cli.parquet`, `markets.parquet`.
"""
    path = out / "report.md"
    path.write_text(txt)
    return path



CLI_STATIONS = ("KNYC", "KAUS", "KORD", "KMDW", "KCLL", "KCMH", "KDFW", "KDEN", "KHOU", "KLAX", "KLEX", "KMIA",
                "KMKE", "KPVD", "KSEA", "KSFO", "KSPG", "KAPC", "KIAH", "KLAS", "KMSY", "KOKC", "KPHX", "KSAN",
                "KSAT", "KATL")
CLI_YEARS = range(2008, 2027)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.counts")
    ap.add_argument("step", choices=["robots", "discover", "triage", "cli", "candles", "build", "evaluate", "report",
                                     "all"])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    steps = ["robots", "discover", "triage", "cli", "candles", "build", "evaluate", "report"] if a.step == "all" else [a.step]
    for st in steps:
        if st == "robots":
            for r in check_robots():
                print(r, flush=True)
        elif st == "discover":
            discover()
        elif st == "triage":
            triage()
        elif st == "cli":
            fetch_cli(list(CLI_STATIONS), CLI_YEARS)
        elif st == "candles":
            fetch_candles()
        elif st == "build":
            build()
        elif st == "evaluate":
            evaluate()
        elif st == "report":
            write_report()


if __name__ == "__main__":
    main()
