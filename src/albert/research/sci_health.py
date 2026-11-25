"""Kalshi science, health, space and natural-event markets that have public data feeds.

Swarm 2026-11-23, slug ``sci_health``. Outputs in ``data/research/sci_health/`` (``tables.md`` is
generated; hypotheses in ``hypotheses.md``, sources and robots verdicts in ``sources.md``).

Families tested (the only ones with enough events): the daily world-earthquake ladder KXBIGGESTQUAKE
(USGS catalog base rate + as-published magnitudes rebuilt from superseded origin products; 82 days) and
KXHURCAT named-storm category ladders (HURDAT2 climatology + 2026 ATCF b-decks; 93 storms).

Result (2026-11-23): **no edge.** The quake ladder carries a 3-5c YES premium over a 10-year base rate in
both splits, but it is pay for clustering (the active Oct-Nov holdout was calibrated at the mid); the
dev-frozen taker NO (bid >= p_cond + 3c, strikes 5.2-5.8, 12-23 h UTC) went +29.5c t 3.55 -> +8.8c t
0.99, book Sharpe 1.3, -4.0 ex top-5. The USGS release race is mostly won before USGS publishes
(Kalshi moves first in 67% of Oct-Nov crossings); frozen race rule +5.7c t 0.72. Hurricanes: active
storms are priced off NHC forecasts, unformed names at climatology; frozen rule -7.7c on the holdout.

    python -m albert.research.sci_health robots | discover | triage | usgs | versions | tracks
    python -m albert.research.sci_health fetch quake KXBIGGESTQUAKE ... | fetch hur KXHURCAT
    python -m albert.research.sci_health quake-rows-rt | quake-eval-rt | race | race-eval
    python -m albert.research.sci_health hur-rows | hur-eval | tables
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.research.counts import AGENTS, market_row, robots_verdict
from albert.venues.kalshi import num

OUT = Path("data/research/sci_health")
CACHE = Path("data/cache/sci_health")
DEV_END = datetime(2026, 10, 1, tzinfo=UTC)          # dev: event ends before this
HOLDOUT_END = datetime(2026, 11, 23, tzinfo=UTC)     # holdout: event ends in [DEV_END, HOLDOUT_END)
KALSHI_MIN_INTERVAL = 0.5
UA = "albert-research/0.2"

# ---- candidate families: series-ticker regex -> family (Kalshi /series, reviewed 2026-11-23) ---------
FAMILIES: list[tuple[str, str]] = [
    ("quake_day", r"^KXBIGGESTQUAKE$"),
    ("quake_hour", r"^KXBIGGESTQUAKEH$"),
    ("quake_month", r"^KX(BIGGESTQUAKEMONTH|EARTHQUAKEM|EARTHQUAKEM7)$"),
    ("quake_region", r"^(KX)?EARTHQUAKE(LA|JAPAN|CALIFORNIA|JAPANTATSUKI)?$"),
    ("tsunami_volcano", r"^(KX)?(TSUNAMI|KILAUEA|ERUPT[A-Z]+|SPURRERUPT|VEI4)$"),
    ("hurricane_storm", r"^(KX)?HURCAT[A-Z]*$|^KX(NAMEDSTORM|HURRICANE|HURRICANENAMES|FIRSTHURRICANE)$"),
    ("hurricane_landfall", r"^(KX)?HUR(PATH|CITY|COAST|CROSS|RLAND|CLAND)[A-Z0-9]*$|^(KX)?HUR(CHARL|SAV|TB|JACKFL|NYC|ORL|HAT|MYR|NOR|NJ|NO|MIA|CAL)$"),
    ("hurricane_dates", r"^KX(NEXTHURDATE|NEXTCAT5HURDATE)$"),
    ("storms_season", r"^(KX)?(HURCTOT|HURCTOTMAJ|HURCMAJ|TROPSTORM)$"),
    ("space_weather", r"^(KX)?(GSTORM|XFLARE)$"),
    ("sea_ice", r"^(KX)?(ARTICICE|SEAICE)$"),
    ("global_temp", r"^(KX)?(HMONTH|GTEMP|HOTYEAR)$"),
    ("enso", r"^KX(LANINA|ELNINO|RONI)$"),
    ("co2", r"^KXCO2LEVEL$"),
    ("lake_mead", r"^(KX)?(MEAD|CORIVER)$"),
    ("tornado", r"^(KX)?TORNADO$"),
    ("aqi_wind", r"^KX(AQICITY|WINDSPEEDNYC|HEATWARNING|DROUGHTLEVEL)$|^(KX)?PMLA$"),
    ("disaster_decl", r"^(KX)?EMERGENCY[A-Z]+$|^(KX)?FEMA$"),
    ("launch_count", r"^(KX)?ROCKET$|^KX(LAUNCHCOUNTM|SPACEXCOUNT|LAUNCHES|RKLBCOUNT|OTHERLAUNCH)$"),
    ("launch_date", (r"^KX(SPCXLAUNCH|RKLBLAUNCH|BOLAUNCH|NEWGLENN|NEXTVULCAN|LANDSPACELAUNCH|NEXTLAUNCHSMALL|"
                     r"SPACEX|SPACEXSTARSHIP|STARSHIP[A-Z0-9]*|ARTEMISII|DRAGONISS|NEUTRONORBIT|RKLBORBIT|"
                     r"SPACEXREUSE|SPACEXLAND|CHOPSTICKS|SPACEXORBIT|ASTRONAUTS)$|^SPACEXLAND$")),
    ("disease_count", (r"^(KX)?(MEASLES|PERTUSSIS|FLDENGUE|CHIKCOUNT|SCREWWORMCOUNT|H5N1COUNT|H5N1CASES|MPOXCOUNT|"
                       r"MPOXCASES|TBCOUNT|GONORRHEACOUNT|HANTACOUNT|CYCLOCOUNT|PLAGUECOUNT|POLIO|GUINEAWORM|"
                       r"AVGMEASLESDJT|CDCSTATFLU)$")),
    ("disease_country", r"^KX(EBOLACOUNTRY|HANTACOUNTRY|USAMEASLESELIMINATE)$"),
    ("covid_legacy", r"^(KX)?(CASE[A-Z0-9]*|SCASE-MARCH2020|VAXX|BOOSTER|BOOSTERM|VAXXDEP)$"),
    ("science_misc", r"^KX(EXOPLAN|NEURALINKCOUNT|DISTANCE3IATLAS|SIZE3IATLAS|INTERSTELLAR)$|^EXOPLAN$"),
    ("ski_open", r"^KX[A-Z]+(OPEN|CLOSE)$"),
]
TOPIC_CATEGORIES = ("Science and Technology", "Health", "World", "Climate and Weather")


def family_of(series: str | None) -> str | None:
    for fam, rx in FAMILIES:
        if re.match(rx, series or ""):
            return fam
    return None


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _api(min_interval: float = KALSHI_MIN_INTERVAL):
    from albert.config import Settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    s = Settings()
    return k.Kalshi(CachedClient("kalshi", s.kalshi_base, CACHE, min_interval=min_interval, max_retries=7,
                                 user_agent=UA))


# ---- robots ----------------------------------------------------------------------------------------
ROBOTS_CHECKS: dict[str, list[str]] = {   # host -> paths we would use (or considered using)
    "api.elections.kalshi.com": ["/trade-api/v2/markets", "/trade-api/v2/series"],
    "earthquake.usgs.gov": ["/fdsnws/event/1/query", "/earthquakes/feed/v1.0/summary/all_hour.geojson",
                            "/fdsnws/event/1/count"],
    "www.nhc.noaa.gov": ["/archive/2026/", "/gis/", "/data/"],
    "ftp.nhc.noaa.gov": ["/atcf/"],
    "services.swpc.noaa.gov": ["/json/goes/primary/xray-flares-7-day.json", "/products/noaa-planetary-k-index.json"],
    "www.swpc.noaa.gov": ["/products/", "/content/warehouse/"],
    "kp.gfz-potsdam.de": ["/app/json/", "/app/files/"],
    "www.ngdc.noaa.gov": ["/stp/space-weather/"],
    "www.ncei.noaa.gov": ["/access/monitoring/", "/data/goes-space-environment-monitor/"],
    "noaadata.apps.nsidc.org": ["/NOAA/G02135/"],
    "nsidc.org": ["/arcticseaicenews/", "/data/seaice_index/"],
    "data.giss.nasa.gov": ["/gistemp/"],
    "climate.copernicus.eu": ["/surface-air-temperature-maps"],
    "cds.climate.copernicus.eu": ["/"],
    "gml.noaa.gov": ["/ccgg/trends/", "/webdata/ccgg/trends/"],
    "www.cpc.ncep.noaa.gov": ["/products/analysis_monitoring/enso_advisory/"],
    "www.usbr.gov": ["/lc/region/g4000/hourly/levels.html", "/uc/water/crsp/studies/24Month.html"],
    "ll.thespacedevs.com": ["/2.3.0/launches/previous/", "/2.2.0/launch/previous/"],
    "lldev.thespacedevs.com": ["/2.3.0/launches/previous/"],
    "planet4589.org": ["/space/gcat/data/", "/space/log/launch.html"],
    "www.faa.gov": ["/data_research/commercial_space_data/launches"],
    "www.cdc.gov": ["/measles/data-research/index.html", "/fluview/", "/pertussis/php/surveillance/"],
    "data.cdc.gov": ["/resource/", "/api/"],
    "wonder.cdc.gov": ["/nndss/"],
    "www.who.int": ["/emergencies/disease-outbreak-news"],
    "www.spc.noaa.gov": ["/climo/reports/"],
    "www.fema.gov": ["/api/open/v2/DisasterDeclarationsSummaries"],
    "exoplanetarchive.ipac.caltech.edu": ["/TAP/sync"],
    "www.airnow.gov": ["/"],
}


def check_robots(out: Path = OUT) -> list[dict[str, Any]]:
    """Fetch each host's own robots.txt (uncached, so the check is current) and evaluate our paths.
    200/404/410 are parsed (404/410 = no file = allowed); anything else (403, 5xx, errors) = blocked."""
    import httpx

    rows = []
    d = out / "robots"
    d.mkdir(parents=True, exist_ok=True)
    for host, paths in ROBOTS_CHECKS.items():
        url = f"https://{host}/robots.txt"
        try:
            r = httpx.get(url, timeout=20, follow_redirects=True, headers={"User-Agent": UA})
            status, text = r.status_code, (r.text if r.status_code == 200 else "")
        except httpx.HTTPError as error:
            status, text = None, ""
            print(f"robots {host}: {type(error).__name__}: {error}", flush=True)
        (d / f"{host}.txt").write_text(text)
        for p in paths:
            v = robots_verdict(text, p) if status in (200, 404, 410) else dict.fromkeys(AGENTS, False)
            rows.append({"host": host, "path": p, "http_status": status,
                         "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                         **{f"allow_{a}": v[a] for a in AGENTS}, "allowed": all(v.values())})
    (out / "robots.json").write_text(json.dumps(rows, indent=1))
    return rows


def robots_allowed(host: str, out: Path = OUT) -> bool:
    """True only if every logged path of ``host`` is allowed for all four agents."""
    rows = [r for r in json.loads((out / "robots.json").read_text()) if r["host"] == host]
    return bool(rows) and all(r["allowed"] for r in rows)


# ---- discovery -------------------------------------------------------------------------------------
MARKET_COLS = {"ticker": pl.String, "event_ticker": pl.String, "series_ticker": pl.String,
               "title": pl.String, "yes_sub_title": pl.String, "strike_type": pl.String,
               "floor_strike": pl.Float64, "cap_strike": pl.Float64, "open_time": pl.Datetime("us", "UTC"),
               "close_time": pl.Datetime("us", "UTC"), "expected_expiration_time": pl.Datetime("us", "UTC"),
               "settlement_ts": pl.Datetime("us", "UTC"), "result": pl.String, "expiration_value": pl.String,
               "can_close_early": pl.Boolean, "volume": pl.Float64, "rules_primary": pl.String,
               "status": pl.String, "src": pl.String}


def series_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Topic series from a ``/series`` payload: a family match plus the financial flag."""
    from albert.categories import kalshi_series_is_financial

    rows = []
    for s in payload.get("series") or []:
        fam = family_of(s.get("ticker"))
        if fam is None or s.get("category") not in TOPIC_CATEGORIES:
            continue
        src = (s.get("settlement_sources") or [{}])[0] or {}
        rows.append({"series_ticker": s.get("ticker"), "family": fam, "category": s.get("category"),
                     "title": s.get("title"), "frequency": s.get("frequency"), "fee_type": s.get("fee_type"),
                     "fee_multiplier": num(s.get("fee_multiplier")), "source_name": src.get("name"),
                     "source_url": src.get("url"),
                     "financial": kalshi_series_is_financial(s.get("ticker"), s.get("category"), s.get("title"))})
    return rows


def _pages(api, path: str, params: dict[str, Any], key: str = "markets"):
    cursor = None
    while True:
        page = api.client.get(path, {**params, "cursor": cursor}, cache=True)
        yield page
        cursor = page.payload.get("cursor")
        if not cursor or not page.payload.get(key):
            return


def discover(out: Path = OUT, *, log=print) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Topic series (today's ``/series``) and every settled market of each (historical + live), plus
    the count of currently open markets (uncached) so the triage knows what is still listed."""
    api = _api()
    ser = pl.DataFrame(series_rows(api.series().payload), infer_schema_length=None)
    rows: list[dict[str, Any]] = []
    open_n: dict[str, int] = {}
    todo = ser.filter(~pl.col("financial"))
    import time

    for i, s in enumerate(todo.iter_rows(named=True)):
        t = s["series_ticker"]
        for attempt in range(3):
            try:
                got: list[dict[str, Any]] = []
                for page in _pages(api, "/historical/markets", {"series_ticker": t, "limit": 1000}):
                    got.extend(market_row(m, t, "hist") for m in page.payload.get("markets") or [])
                for page in _pages(api, "/markets", {"series_ticker": t, "status": "settled", "limit": 1000}):
                    got.extend(market_row(m, t, "live") for m in page.payload.get("markets") or [])
                n_open = 0
                for page in api.series_markets(t, "open"):
                    n_open += len(page.payload.get("markets") or [])
                rows.extend(got)
                open_n[t] = n_open
                break
            except RuntimeError as error:     # shared IP: back off on 429 and retry the series
                log(f"  {t}: {str(error)[-80:]}; retry in 60 s", flush=True)
                time.sleep(60)
        if i % 20 == 0:
            log(f"  discover {i + 1}/{todo.height} {t}: {len(rows)} markets so far", flush=True)
    ser = ser.with_columns(pl.col("series_ticker").replace_strict(open_n, default=None, return_dtype=pl.Int64)
                           .alias("open_markets_now"))
    _save(ser, out / "series.parquet")
    mk = (pl.DataFrame(rows, schema=MARKET_COLS).unique("ticker", keep="last")
          .filter(pl.col("result").is_in(["yes", "no"])))
    _save(mk, out / "markets.parquet")
    log(f"discover: {ser.height} topic series ({todo.height} non-financial), {mk.height} settled markets")
    return ser, mk


def event_table(mk: pl.DataFrame) -> pl.DataFrame:
    """One row per event: its end (latest close), markets, volume, split."""
    return (mk.group_by("series_ticker", "event_ticker")
            .agg(pl.col("close_time").max().alias("end"), pl.len().alias("markets"),
                 pl.col("volume").sum().alias("volume"), pl.col("open_time").min().alias("open"))
            .with_columns(pl.when(pl.col("end") < DEV_END).then(pl.lit("dev"))
                          .when(pl.col("end") < HOLDOUT_END).then(pl.lit("holdout")).otherwise(pl.lit("later"))
                          .alias("split")))


def triage(out: Path = OUT) -> pl.DataFrame:
    ser = pl.read_parquet(out / "series.parquet")
    mk = pl.read_parquet(out / "markets.parquet")
    ev = event_table(mk)
    recent = ev.filter(pl.col("end") >= datetime(2025, 10, 1, tzinfo=UTC))
    per = (ev.group_by("series_ticker").agg(
        pl.len().alias("events"), (pl.col("split") == "dev").sum().alias("events_dev"),
        (pl.col("split") == "holdout").sum().alias("events_holdout"),
        pl.col("markets").sum().alias("markets"), pl.col("volume").sum().alias("volume"),
        pl.col("end").min().alias("first_end"), pl.col("end").max().alias("last_end"))
        .join(recent.group_by("series_ticker").agg(pl.len().alias("events_last_14m"),
                                                   pl.col("volume").sum().alias("volume_last_14m")),
              on="series_ticker", how="left"))
    tri = (ser.join(per, on="series_ticker", how="left")
           .with_columns(pl.col("events", "events_dev", "events_holdout", "markets", "events_last_14m").fill_null(0))
           .sort(["family", "volume"], descending=[False, True], nulls_last=True))
    _save(tri, out / "triage.parquet")
    tri.drop("source_url").write_csv(out / "triage.csv")
    fam = (tri.group_by("family").agg(pl.len().alias("series"), pl.col("events").sum(), pl.col("events_dev").sum(),
                                      pl.col("events_holdout").sum(), pl.col("events_last_14m").sum(),
                                      pl.col("volume").sum(), pl.col("volume_last_14m").sum(),
                                      pl.col("open_markets_now").sum().alias("open_now"))
           .sort("volume_last_14m", descending=True, nulls_last=True))
    fam.write_csv(out / "triage_family.csv")
    return fam


# ---- Kalshi trades and candles for a market set ---------------------------------------------------------
TRADE_COLS = {"trade_id": pl.String, "ticker": pl.String, "created_time": pl.Datetime("us", "UTC"),
              "count": pl.Float64, "yes_price": pl.Float64, "taker_side": pl.String}
CANDLE_COLS = {"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "period_minutes": pl.Int64,
               "yes_bid_close": pl.Float64, "yes_ask_close": pl.Float64, "price_close": pl.Float64,
               "price_mean": pl.Float64, "volume": pl.Float64, "open_interest": pl.Float64}


def fetch_trades(markets: pl.DataFrame, name: str, out: Path = OUT, *, log=print) -> pl.DataFrame:
    """Every trade of each market (needs ticker, close_time); historical endpoint before the cutoff.
    Pages are cached (append-only raw), so re-runs only fetch what is missing."""
    from albert.venues.kalshi import trade_rows

    api = _api()
    cutoff = api.cutoff().get("trades_created_ts")
    rows: list[dict[str, Any]] = []
    todo = markets.select("ticker", "close_time").unique("ticker").sort("close_time").iter_rows(named=True)
    for i, m in enumerate(todo, 1):
        hist = bool(cutoff and m["close_time"] and m["close_time"] < cutoff)
        path = "/historical/trades" if hist else "/markets/trades"
        try:
            for page in _pages(api, path, {"ticker": m["ticker"], "limit": 1000}, key="trades"):
                rows.extend(trade_rows(page.payload))
        except RuntimeError as error:      # retries exhausted: skip this market, keep the rest
            log(f"skip trades {m['ticker']}: {str(error)[:120]}")
        if i % 100 == 0:
            log(f"  trades {name} {i}: {len(rows)} trades so far", flush=True)
    tr = pl.DataFrame(rows, schema=TRADE_COLS).unique("trade_id").sort("ticker", "created_time")
    _save(tr, out / f"{name}_trades.parquet")
    log(f"trades {name}: {tr.height} trades on {tr['ticker'].n_unique()} markets")
    return tr


def fetch_candles(markets: pl.DataFrame, name: str, out: Path = OUT, *, period: int = 60, log=print) -> pl.DataFrame:
    """Candles over each market's whole life (open -> close). Live markets in batches of <= 9,000
    summed periods; markets that closed before the cutoff one by one from the historical endpoint."""
    from albert.venues.kalshi import candle_rows

    api = _api()
    cutoff = api.cutoff().get("market_settled_ts")
    rows: list[dict[str, Any]] = []
    ms = markets.select("ticker", "open_time", "close_time").unique("ticker").sort("close_time").to_dicts()
    hist = [m for m in ms if cutoff and m["close_time"] < cutoff]
    live = [m for m in ms if not (cutoff and m["close_time"] < cutoff)]
    for m in hist:
        lo = max(m["open_time"], m["close_time"] - timedelta(minutes=period * 4900))
        try:
            page = api.candles(m["ticker"], lo, m["close_time"] + timedelta(minutes=period), period, True)
        except RuntimeError as error:
            log(f"skip candles {m['ticker']}: {str(error)[:120]}")
            continue
        rows.extend(candle_rows(page.payload, m["ticker"], period))
    batch: list[dict[str, Any]] = []

    def flush() -> None:
        if not batch:
            return
        lo = min(m["open_time"] for m in batch)
        hi = max(m["close_time"] for m in batch) + timedelta(minutes=period)
        page = api.candles_batch([m["ticker"] for m in batch], lo, hi, period)
        for node in page.payload.get("markets") or []:
            rows.extend(candle_rows(node, node.get("market_ticker"), period))
        batch.clear()
    span = 0
    for m in live:
        n = int((m["close_time"] - m["open_time"]).total_seconds() // (60 * period)) + 2
        if n > 4900:
            m = {**m, "open_time": m["close_time"] - timedelta(minutes=period * 4800)}
            n = 4802
        if batch:
            lo = min(min(b["open_time"] for b in batch), m["open_time"])
            hi = max(max(b["close_time"] for b in batch), m["close_time"])
            span = int((hi - lo).total_seconds() // (60 * period)) + 2
            if span * (len(batch) + 1) > 9000 or len(batch) >= 100:
                flush()
        batch.append(m)
    flush()
    c = pl.DataFrame(rows, schema=CANDLE_COLS).unique(["ticker", "end_ts"]).sort("ticker", "end_ts")
    _save(c, out / "candles" / f"{name}_{period}m.parquet")
    log(f"candles {name}: {c.height} candles on {c['ticker'].n_unique()} of {len(ms)} markets")
    return c


# ---- USGS earthquake catalog (earthquake.usgs.gov: robots.txt 404 = allowed) ----------------------------
USGS_BASE = "https://earthquake.usgs.gov"
USGS_MIN_INTERVAL = 1.0
QUAKE_COLS = {"id": pl.String, "time": pl.Datetime("ms", "UTC"), "mag": pl.Float64, "mag_type": pl.String,
              "updated": pl.Datetime("ms", "UTC"), "place": pl.String, "net": pl.String, "status": pl.String,
              "lat": pl.Float64, "lon": pl.Float64, "depth": pl.Float64}


def _usgs():
    from albert.http import CachedClient

    return CachedClient("usgs", USGS_BASE, CACHE, min_interval=USGS_MIN_INTERVAL, max_retries=8, user_agent=UA)


def usgs_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows from an FDSN ``format=geojson`` query (earthquakes only; other event types dropped)."""
    rows = []
    for f in payload.get("features") or []:
        p = f.get("properties") or {}
        if p.get("type") not in (None, "earthquake") or p.get("mag") is None or p.get("time") is None:
            continue
        c = (f.get("geometry") or {}).get("coordinates") or [None, None, None]
        rows.append({"id": f.get("id"), "time": datetime.fromtimestamp(p["time"] / 1000, UTC),
                     "mag": float(p["mag"]), "mag_type": p.get("magType"),
                     "updated": datetime.fromtimestamp(p["updated"] / 1000, UTC) if p.get("updated") else None,
                     "place": p.get("place"), "net": p.get("net"), "status": p.get("status"),
                     "lat": c[1], "lon": c[0], "depth": c[2]})
    return rows


def fetch_quakes(start: datetime = datetime(2012, 1, 1, tzinfo=UTC), end: datetime = HOLDOUT_END,
                 min_mag: float = 4.0, out: Path = OUT, *, log=print) -> pl.DataFrame:
    """Global catalog M >= ``min_mag`` in monthly windows (each well under the 20,000-row cap).
    Windows that end in the past 40 days are fetched uncached (the catalog is still being reviewed)."""
    c = _usgs()
    rows: list[dict[str, Any]] = []
    a = start
    while a < end:
        b = datetime(a.year + (a.month == 12), a.month % 12 + 1, 1, tzinfo=UTC)
        b = min(b, end)
        fresh = (datetime.now(UTC) - b).days < 40
        r = c.get("/fdsnws/event/1/query", {"format": "geojson", "starttime": a.strftime("%Y-%m-%dT%H:%M:%S"),
                                             "endtime": b.strftime("%Y-%m-%dT%H:%M:%S"), "minmagnitude": min_mag,
                                             "orderby": "time-asc", "eventtype": "earthquake"},
                  tag="fresh-" + datetime.now(UTC).strftime("%Y-%m-%d") if fresh else None)
        rows.extend(usgs_rows(r.payload))
        if a.month == 1:
            log(f"  usgs {a:%Y-%m}: {len(rows)} quakes so far", flush=True)
        a = b
    q = pl.DataFrame(rows, schema=QUAKE_COLS).unique("id", keep="last").sort("time")
    _save(q, out / "quakes.parquet")
    log(f"usgs: {q.height} quakes M>={min_mag} {start:%Y-%m-%d}..{end:%Y-%m-%d}")
    return q


# ---- quake daily ladder (KXBIGGESTQUAKE): empirical as-of model ---------------------------------------
QUAKE_SERIES = "KXBIGGESTQUAKE"
SLOT_MIN = 15                         # intraday resolution of the empirical model
N_SLOTS = 24 * 60 // SLOT_MIN
PUB_LAG = timedelta(minutes=30)       # a quake counts as public 30 min after origin (USGS posts M5+ in ~10-25 min)
TRAIN_DAYS = 3650                     # base-rate window: the 10 years before the event day
RECENT_BINS = (6.5, 7.0)              # H3: max magnitude in the 3 days before -> bin 0 / 1 / 2
DECISION_HOURS = (-3, 0, 3, 6, 9, 12, 15, 18, 21, 23)   # hours after the event day's 00:00 UTC
_MON = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
                                        "NOV", "DEC"])}


def quake_day(event_ticker: str) -> date | None:
    """``KXBIGGESTQUAKE-13NOV26`` -> 2026-11-13 (the UTC day the ladder settles on)."""
    m = re.match(r"^KXBIGGESTQUAKE-(\d{2})([A-Z]{3})(\d{2})$", event_ticker or "")
    if not m or m.group(2) not in _MON:
        return None
    return date(2000 + int(m.group(3)), _MON[m.group(2)], int(m.group(1)))


class QuakeClimate:
    """Per-day, per-15-minute maximum magnitude from the USGS catalog, so that
    P(max over the rest of the day >= m | nothing >= m so far) can be counted over past days."""

    def __init__(self, quakes: pl.DataFrame, first: date = date(2012, 1, 1), last: date | None = None) -> None:
        last = last or quakes["time"].max().date()
        self.first = first
        n = (last - first).days + 1
        grid = np.full((n, N_SLOTS), -np.inf)
        q = quakes.filter(pl.col("time") >= datetime(first.year, first.month, first.day, tzinfo=UTC))
        t = q["time"].to_numpy().astype("datetime64[ms]").astype(np.int64)
        day0 = np.datetime64(first, "D").astype("datetime64[ms]").astype(np.int64)
        d = (t - day0) // 86_400_000
        sl = ((t - day0) % 86_400_000) // (SLOT_MIN * 60_000)
        ok = (d >= 0) & (d < n)
        np.maximum.at(grid, (d[ok], sl[ok]), q["mag"].to_numpy()[ok])
        self.grid = grid
        self.pre = np.concatenate([np.full((n, 1), -np.inf), np.maximum.accumulate(grid, axis=1)], axis=1)  # max over slots < s
        self.suf = np.concatenate([np.maximum.accumulate(grid[:, ::-1], axis=1)[:, ::-1], np.full((n, 1), -np.inf)],
                                  axis=1)                                                                   # max over slots >= s
        daymax = grid.max(axis=1)
        prev3 = np.full(n, -np.inf)
        for k in (1, 2, 3):
            prev3[k:] = np.maximum(prev3[k:], daymax[:-k])
        self.daymax = daymax
        self.recent = np.digitize(prev3, RECENT_BINS)
        self.times = q["time"]
        self.mags = q["mag"].to_numpy()
        self._t = t

    def ix(self, d: date) -> int:
        return (d - self.first).days

    def known_max(self, d: date, t: datetime, lag: timedelta = PUB_LAG) -> float:
        """Largest magnitude on day ``d`` with origin before ``t - lag`` (public by ``t``)."""
        lo = int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)
        hi = int((t - lag).timestamp() * 1000)
        if hi <= lo:
            return -np.inf
        a, b = np.searchsorted(self._t, [lo, hi])
        return float(self.mags[a:b].max()) if b > a else -np.inf

    def recent_bin(self, d: date, t: datetime, lag: timedelta = PUB_LAG) -> int:
        """H3 state: bin of the max magnitude in the 72 h before the day (public by ``t``)."""
        d0 = datetime(d.year, d.month, d.day, tzinfo=UTC)
        lo = int((d0 - timedelta(days=3)).timestamp() * 1000)
        hi = int(min(d0, t - lag).timestamp() * 1000)
        a, b = np.searchsorted(self._t, [lo, hi])
        mx = float(self.mags[a:b].max()) if b > a else -np.inf
        return int(np.digitize([mx], RECENT_BINS)[0])

    def p(self, d: date, t: datetime, m: float, *, lag: timedelta = PUB_LAG, cond: bool = False,
          train_days: int = TRAIN_DAYS) -> tuple[float, int]:
        """P(day max >= m | nothing >= m public by t) counted over the ``train_days`` days before ``d``
        (all complete and public long before ``t``). The unknown window is [t - lag, end of day)."""
        d0 = datetime(d.year, d.month, d.day, tzinfo=UTC)
        s = int(np.clip(np.floor((t - lag - d0).total_seconds() / (SLOT_MIN * 60)), 0, N_SLOTS))
        i = self.ix(d)
        lo = max(0, i - train_days)
        if i - lo < 365:
            return float("nan"), 0
        pre, suf = self.pre[lo:i, s], self.suf[lo:i, s]
        mask = pre < m - 1e-9
        if cond:
            mask &= self.recent[lo:i] == self.recent_bin(d, t, lag)
        n = int(mask.sum())
        hits = int((suf[mask] >= m - 1e-9).sum())
        return (hits + 0.5) / (n + 1.0), n


def _quote_arrays(c: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    for (tk,), g in c.sort("ticker", "end_ts").group_by(["ticker"], maintain_order=True):
        out[tk] = (g["end_ts"].to_numpy().astype("datetime64[us]").astype(np.int64), g["yes_bid_close"].to_numpy(),
                   g["yes_ask_close"].to_numpy())
    return out


def _quote(qa: dict, tk: str, t: datetime) -> tuple[float | None, float | None]:
    a = qa.get(tk)
    if a is None:
        return None, None
    i = int(np.searchsorted(a[0], int(t.timestamp() * 1e6), side="right")) - 1
    if i < 0:
        return None, None
    b, k = a[1][i], a[2][i]
    return (None if b is None or np.isnan(b) else float(b)), (None if k is None or np.isnan(k) else float(k))


def quake_decisions(name: str = "quake", out: Path = OUT, *, realtime: bool = True, tag: str = "",
                    log=print) -> pl.DataFrame:
    """One row per (market, decision time) while the market is open: as-of model (static and H3
    conditional), the known running max, the hourly quote, and the outcome."""
    mk = (pl.read_parquet(out / f"{name}_markets.parquet").filter(pl.col("series_ticker") == QUAKE_SERIES)
          .with_columns(pl.col("event_ticker").map_elements(quake_day, return_dtype=pl.Date).alias("day"))
          .filter(pl.col("day").is_not_null()))
    qk = pl.read_parquet(out / "quakes.parquet")
    clim = QuakeClimate(qk)
    qa = _quote_arrays(pl.read_parquet(out / "candles" / f"{name}_60m.parquet"))
    vp = out / "quake_versions.parquet"
    rt = Realtime(pl.read_parquet(vp)) if realtime and vp.exists() else None
    rows = []
    for r in mk.iter_rows(named=True):
        d, m = r["day"], r["floor_strike"]
        d0 = datetime(d.year, d.month, d.day, tzinfo=UTC)
        cross = rt.first_cross(d, m) if rt is not None else first_cross(clim, d, m)
        for h in DECISION_HOURS:
            t = d0 + timedelta(hours=h)
            if not (r["open_time"] <= t < r["close_time"]):
                continue
            km = rt.known_max(d, t) if rt is not None else clim.known_max(d, t)
            ps, ns = clim.p(d, t, m)
            pc, nc = clim.p(d, t, m, cond=True)
            bid, ask = _quote(qa, r["ticker"], t)
            rows.append({"ticker": r["ticker"], "event_ticker": r["event_ticker"], "day": d, "strike": m, "h": h,
                         "cross_time": cross,
                         "t": t, "close_time": r["close_time"], "settlement_ts": r["settlement_ts"],
                         "y": int(r["result"] == "yes"), "known_max": km, "decided": km >= m - 1e-9,
                         "p_static": ps, "n_static": ns, "p_cond": pc, "n_cond": nc,
                         "recent_bin": clim.recent_bin(d, t), "bid": bid, "ask": ask,
                         "fee_type": r.get("fee_type"), "split": "dev" if r["close_time"] < DEV_END else
                         ("holdout" if r["close_time"] < HOLDOUT_END else "later")})
    df = pl.DataFrame(rows, infer_schema_length=None)
    # settlement check: catalog day max vs Kalshi's result
    chk = (mk.select("ticker", "day", "floor_strike", "result")
           .with_columns(pl.col("day").map_elements(lambda d: float(clim.daymax[clim.ix(d)]) if 0 <= clim.ix(d) < len(clim.daymax) else None,
                                                    return_dtype=pl.Float64).alias("cat_max"))
           .with_columns(((pl.col("cat_max") >= pl.col("floor_strike") - 1e-9) == (pl.col("result") == "yes")).alias("agree")))
    if rt is not None:   # the magnitude USGS displayed at the market's expiration (or early close)
        exp = mk.select("ticker", pl.min_horizontal(pl.col("expected_expiration_time"), pl.col("close_time")).alias("exp"))
        rtmax = {tk: rt.known_max(day, ex, timedelta(0)) for tk, day, ex in
                 zip(mk["ticker"], mk["day"], exp["exp"], strict=True)}
        chk = chk.with_columns(pl.col("ticker").replace_strict(rtmax, return_dtype=pl.Float64).alias("rt_max_at_exp")).with_columns(
            ((pl.col("rt_max_at_exp") >= pl.col("floor_strike") - 1e-9) == (pl.col("result") == "yes")).alias("agree_rt"))
    _save(chk, out / f"{name}{tag}_settle_check.parquet")
    log(f"quake rows: {df.height} decision rows on {df['ticker'].n_unique()} markets; final catalog agrees with "
        f"Kalshi on {int(chk['agree'].sum())}/{chk.height} markets"
        + (f"; displayed-at-expiration magnitude on {int(chk['agree_rt'].sum())}/{chk.height}" if rt is not None else ""))
    _save(df, out / f"{name}{tag}_rows.parquet")
    return df


# ---- quake ladder: taker / maker cells, ledgers ------------------------------------------------------------
STAKE_USD = 3.0           # each order risks ~$3 (fee rounding is per order) ...
MAX_CONTRACTS = 20        # ... and at most 20 contracts
MAX_SPREAD = 0.15         # takers only cross books at most this wide
CANCEL_LAG = timedelta(minutes=20)   # a resting quote is pulled 20 min after a strike-crossing quake's origin
TAKER_THETAS = (0.03, 0.05, 0.08)
MAKER_MARGINS = (0.02, 0.04, 0.08, 0.12)
QUAKE_BUCKETS = {"pre": pl.col("h") < 0, "early": (pl.col("h") >= 0) & (pl.col("h") < 12), "late": pl.col("h") >= 12,
                 "all": pl.lit(True)}
QUAKE_BANDS = {"low": pl.col("strike") <= 5.85, "high": pl.col("strike") >= 5.95, "all": pl.lit(True)}
HUR_BUCKETS = {"unformed": pl.col("state") == "unformed", "active": pl.col("state") == "active",
               "dissipated": pl.col("state") == "dissipated", "all": pl.lit(True)}
HUR_BANDS = {"cat12": pl.col("strike") <= 2.5, "cat345": pl.col("strike") >= 2.5, "all": pl.lit(True)}


def order_size(price: float) -> int:
    return int(min(MAX_CONTRACTS, max(1, int(STAKE_USD // max(price, 0.01)))))


def taker_fee(n: int, price: float, mult: float | None = 1.0) -> float:
    """0.07 x mult x n x P(1-P), rounded up to the cent per order (dollars)."""
    import math

    raw = 0.07 * (mult or 1.0) * n * price * (1 - price)
    return math.ceil(round(raw * 100, 9)) / 100


def taker_trades(rows: pl.DataFrame, side: str, theta: float, model: str) -> pl.DataFrame:
    """Buy NO at 1 - bid when bid - p >= theta, or YES at the ask when p - ask >= theta (undecided
    strikes, two-sided book <= 15c wide), ~$3 order, taker fee rounded up, held to settlement."""
    r = rows.filter(~pl.col("decided") & pl.col(model).is_not_null() & pl.col("bid").is_not_null()
                    & pl.col("ask").is_not_null() & (pl.col("bid") > 0) & (pl.col("ask") < 1)
                    & (pl.col("ask") - pl.col("bid") <= MAX_SPREAD + 1e-9))
    if side == "no":
        r = r.filter(pl.col("bid") - pl.col(model) >= theta - 1e-9).with_columns(
            (1 - pl.col("bid")).alias("price"), (1 - pl.col("y")).alias("payoff"))
    else:
        r = r.filter(pl.col(model) - pl.col("ask") >= theta - 1e-9).with_columns(
            pl.col("ask").alias("price"), pl.col("y").alias("payoff"))
    if r.is_empty():
        return r.with_columns(pl.lit(None, pl.Int64).alias("n"), pl.lit(None, pl.Float64).alias("fee_order"),
                              pl.lit(None, pl.Float64).alias("pnl_c"), pl.lit(side).alias("side"))
    n = [order_size(x) for x in r["price"].to_list()]
    fee = [taker_fee(k, x) for k, x in zip(n, r["price"].to_list(), strict=True)]
    return r.with_columns(pl.Series("n", n, pl.Int64), pl.Series("fee_order", fee), pl.lit(side).alias("side"),
                          pl.col("t").alias("t_fill")).with_columns(
        (100 * (pl.col("payoff") - pl.col("price") - pl.col("fee_order") / pl.col("n"))).alias("pnl_c"))


class Tape:
    """Kalshi prints per ticker for maker fills (fill only on a later print strictly through)."""

    def __init__(self, trades: pl.DataFrame) -> None:
        self.by: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for (tk,), g in trades.sort("ticker", "created_time").group_by(["ticker"], maintain_order=True):
            self.by[tk] = (g["created_time"].to_numpy().astype("datetime64[us]").astype(np.int64),
                           g["yes_price"].to_numpy(), g["count"].to_numpy(),
                           (g["taker_side"] == "yes").to_numpy())

    def fill(self, tk: str, t0: datetime, t1: datetime, side: str, q: float, n: int) -> tuple[int, datetime | None]:
        """Contracts filled (<= n) for a resting quote at YES price ``q`` between t0 and t1: a NO bid
        (side 'no', i.e. selling YES at q) fills on YES-taker prints above q; a YES bid on NO-taker
        prints below q. Returns (filled contracts, time of the completing print)."""
        a = self.by.get(tk)
        if a is None or t1 <= t0:
            return 0, None
        lo, hi = np.searchsorted(a[0], [int(t0.timestamp() * 1e6), int(t1.timestamp() * 1e6)], side="right")
        ts, px, ct, ty = a[0][lo:hi], a[1][lo:hi], a[2][lo:hi], a[3][lo:hi]
        hit = (ty & (px > q + 1e-9)) if side == "no" else (~ty & (px < q - 1e-9))
        if not hit.any():
            return 0, None
        cum = np.cumsum(np.where(hit, ct, 0.0))
        k = int(np.searchsorted(cum, n - 1e-9))
        k = min(k, len(cum) - 1)
        filled = int(min(n, cum[-1]))
        return filled, datetime.fromtimestamp(int(ts[k]) / 1e6, UTC)


def first_cross(clim: QuakeClimate, d: date, m: float) -> datetime | None:
    """Origin time of the first quake on day d with magnitude >= m (final catalog)."""
    lo = int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)
    a, b = np.searchsorted(clim._t, [lo, lo + 86_400_000])
    idx = np.nonzero(clim.mags[a:b] >= m - 1e-9)[0]
    return datetime.fromtimestamp(int(clim._t[a + idx[0]]) / 1000, UTC) if len(idx) else None


def maker_trades(rows: pl.DataFrame, tape: Tape, side: str, margin: float, model: str,
                 cancel_lag: timedelta | None = CANCEL_LAG) -> pl.DataFrame:
    """A resting quote at each decision time, valid until the market's next decision time (or close):
    side 'no' = NO bid selling YES at ceil(p + margin); side 'yes' = YES bid at floor(p - margin).
    Only non-crossing quotes; pulled ``cancel_lag`` after ``cross_time`` (the strike-crossing quake's
    origin, or the hurricane fix that reached the threshold; None = never pulled). Fill on later prints strictly through; held to settlement; no maker fee here."""
    import math

    r = rows.filter(~pl.col("decided") & pl.col("bid").is_not_null() & pl.col("ask").is_not_null())
    if model == "mid":
        r = r.filter((pl.col("bid") > 0) & (pl.col("ask") < 1)).with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    r = r.filter(pl.col(model).is_not_null()).sort("ticker", "t")
    r = r.with_columns(pl.col("t").shift(-1).over("ticker").alias("t_next"))
    out = []
    for x in r.iter_rows(named=True):
        p = x[model]
        if side == "no":
            q = math.ceil(round((p + margin) * 100, 6)) / 100
            if q >= 0.99 or q <= (x["bid"] or 0.0):
                continue
            price = round(1 - q, 2)
        else:
            q = math.floor(round((p - margin) * 100, 6)) / 100
            if q < 0.01 or q >= (x["ask"] if x["ask"] is not None else 1.0):
                continue
            price = q
        end = min(x["t_next"] or x["close_time"], x["close_time"])
        if cancel_lag is not None:
            fc = x.get("cross_time")
            if fc is not None:
                end = min(end, fc + cancel_lag)
        n = order_size(price)
        got, ts = tape.fill(x["ticker"], x["t"], end, side, q, n)
        if got <= 0:
            continue
        payoff = (1 - x["y"]) if side == "no" else x["y"]
        out.append({**{k: x.get(k) for k in ("ticker", "event_ticker", "day", "strike", "h", "t", "settlement_ts", "y",
                                              "split", "bid", "ask", "state")},
                    "side": side, "q": q, "price": price, "n": got, "t_fill": ts, "fee_order": 0.0, "payoff": payoff,
                    "pnl_c": 100 * (payoff - price)})
    return pl.DataFrame(out, infer_schema_length=None)


def _bucket(df: pl.DataFrame, tb: pl.Expr, sb: pl.Expr) -> pl.DataFrame:
    return df if df.is_empty() else df.filter(tb & sb)


def cell_stats(df: pl.DataFrame) -> dict[str, Any]:
    from albert.research.counts import cluster_se

    if df.is_empty():
        return {"fills": 0, "events": 0, "contracts": 0, "c_per_contract": None, "t_event": None, "pnl_usd": 0.0,
                "pnl_per_usd": None, "win_pct": None}
    w = df["n"].to_numpy().astype(float)
    x = df["pnl_c"].to_numpy().astype(float)
    ev = df["event_ticker"].to_numpy().astype(str)
    usd = (df["pnl_c"] * df["n"] / 100).to_numpy()
    risk = (df["price"] * df["n"] + df["fee_order"]).to_numpy()
    # event-clustered t on per-order dollars, the unit that is risked
    m = float(usd.mean())
    se = cluster_se(usd, ev) if len(np.unique(ev)) > 1 else float("nan")
    return {"fills": len(x), "events": len(np.unique(ev)), "contracts": int(w.sum()),
            "c_per_contract": round(float((x * w).sum() / w.sum()), 2),
            "t_event": round(m / se, 2) if se and np.isfinite(se) and se > 0 else None,
            "pnl_usd": round(float(usd.sum()), 2), "pnl_per_usd": round(float(usd.sum() / risk.sum()), 4),
            "win_pct": round(100 * float((x > 0).mean()), 1)}


def quake_cells(rows: pl.DataFrame, tape: Tape, split: str, models: tuple[str, ...] = ("p_static", "p_cond"),
                buckets: dict[str, pl.Expr] | None = None, bands: dict[str, pl.Expr] | None = None,
                ) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    buckets = buckets or QUAKE_BUCKETS
    bands = bands or QUAKE_BANDS
    r = rows.filter(pl.col("split") == split)
    res, fills = [], {}
    base: list[tuple[str, str, float, str, pl.DataFrame]] = []
    for side in ("no", "yes"):
        for model in models:
            for th in TAKER_THETAS:
                base.append(("taker", side, th, model, taker_trades(r, side, th, model)))
        for model in (*models, "mid"):
            for mg in MAKER_MARGINS:
                base.append(("maker", side, mg, model, maker_trades(r, tape, side, mg, model)))
    for kind, side, prm, model, f in base:
        for tb, te in buckets.items():
            for sb, se in bands.items():
                cid = f"{kind}|{side}|{prm:.2f}|{model}|{tb}|{sb}"
                g = _bucket(f, te, se)
                fills[cid] = g
                res.append({"cell": cid, "kind": kind, "side": side, "param": prm, "model": model, "time": tb,
                            "strikes": sb, "split": split, **cell_stats(g)})
    return pl.DataFrame(res, infer_schema_length=None), fills


def ledger(fills: pl.DataFrame, variant: str, split: str, bankroll: float = 100.0, event_cap: float = 5.0) -> pl.DataFrame:
    """Capped book: positions taken in fill order while open cost <= bankroll and <= event_cap per
    event (UTC day); capital returns at settlement. vol_metrics ledger schema."""
    import heapq

    heap: list[tuple[datetime, float, str]] = []
    used, per_ev, out = 0.0, {}, []
    for r in fills.sort("t_fill").iter_rows(named=True):
        while heap and heap[0][0] <= r["t_fill"]:
            _, c, e = heapq.heappop(heap)
            used -= c
            per_ev[e] -= c
        cost = r["n"] * r["price"] + r["fee_order"]
        if used + cost > bankroll + 1e-9 or per_ev.get(r["event_ticker"], 0.0) + cost > event_cap + 1e-9:
            continue
        settle = max(r["settlement_ts"], r["t_fill"])
        used += cost
        per_ev[r["event_ticker"]] = per_ev.get(r["event_ticker"], 0.0) + cost
        heapq.heappush(heap, (settle, cost, r["event_ticker"]))
        out.append({"strategy": "sci_health", "variant": variant, "split": split,
                    "struct_id": f"{r['ticker']}|{r['t'].isoformat()}|{r['side']}", "event_ticker": r["event_ticker"],
                    "ticker": r["ticker"], "side": r["side"], "n": int(r["n"]), "t_in": r["t_fill"], "cost_usd": cost,
                    "settle_ts": settle, "payoff_usd": float(r["n"] * r["payoff"])})
    return pl.DataFrame(out)


def book_metrics(led: pl.DataFrame) -> dict[str, Any]:
    from albert.research.vol_metrics import ledger_metrics, load_marks

    if led.is_empty():
        return {"positions": 0}
    m = ledger_metrics(led, load_marks(led["ticker"].unique().to_list()))
    keep = ("positions", "events", "pnl_usd", "usd_per_week", "ann_return_pct", "ann_vol_pct", "sharpe", "sharpe_ci90",
            "p_sharpe_pos", "sharpe_ex_top5", "usd_per_week_ex_top5", "max_dd_usd", "weeks_positive_pct", "skew",
            "first_day", "last_day")
    return {k: m.get(k) for k in keep}


# ---- real-time (as-published) magnitudes: USGS origin-product versions ----------------------------------
RT_FIRST, RT_LAST = date(2026, 8, 31), date(2026, 11, 23)
RT_MIN_MAG = 4.8          # initial magnitudes can sit ~0.3-0.4 above the final one
POLL_LAG = timedelta(minutes=2)   # our own detection delay after a USGS product update


def origin_versions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Every origin-product version of one event (``includesuperseded=true``): source, update time,
    magnitude, preferred weight, status."""
    props = payload.get("properties") or {}
    rows = []
    for x in (props.get("products") or {}).get("origin") or []:
        rows.append({"event_id": payload.get("id"), "source": x.get("source"),
                     "update_time": datetime.fromtimestamp(x["updateTime"] / 1000, UTC),
                     "mag": num((x.get("properties") or {}).get("magnitude")),
                     "weight": int(x.get("preferredWeight") or 0), "status": x.get("status")})
    return rows


def fetch_versions(out: Path = OUT, *, log=print) -> pl.DataFrame:
    qk = pl.read_parquet(out / "quakes.parquet").filter(
        (pl.col("time") >= datetime(RT_FIRST.year, RT_FIRST.month, RT_FIRST.day, tzinfo=UTC))
        & (pl.col("time") < datetime(RT_LAST.year, RT_LAST.month, RT_LAST.day, tzinfo=UTC)) & (pl.col("mag") >= RT_MIN_MAG))
    c = _usgs()
    rows: list[dict[str, Any]] = []
    for i, q in enumerate(qk.iter_rows(named=True), 1):
        try:
            det = c.get("/fdsnws/event/1/query", {"format": "geojson", "eventid": q["id"], "includesuperseded": "true"})
        except RuntimeError as error:
            log(f"skip {q['id']}: {str(error)[:100]}")
            continue
        for v in origin_versions(det.payload):
            rows.append({**v, "event_id": q["id"], "origin": q["time"], "final_mag": q["mag"]})
        if i % 100 == 0:
            log(f"  versions {i}/{qk.height}", flush=True)
    vs = pl.DataFrame(rows, infer_schema_length=None).sort("event_id", "update_time")
    _save(vs, out / "quake_versions.parquet")
    log(f"versions: {vs.height} origin versions for {vs['event_id'].n_unique()} of {qk.height} events")
    return vs


class Realtime:
    """Magnitude USGS displayed for each event at time t: per source the latest origin version at or
    before t (a DELETE drops the source), then the source with the highest preferred weight."""

    def __init__(self, vs: pl.DataFrame) -> None:
        self.by_day: dict[date, list[tuple[datetime, list[tuple[datetime, str, float | None, int, str]]]]] = {}
        for (eid,), g in vs.sort("update_time").group_by(["event_id"], maintain_order=True):
            org = g["origin"][0]
            vers = list(zip(g["update_time"].to_list(), g["source"].to_list(), g["mag"].to_list(),
                            g["weight"].to_list(), g["status"].to_list(), strict=True))
            self.by_day.setdefault(org.date(), []).append((org, vers))

    @staticmethod
    def displayed(vers: list, t: datetime) -> float | None:
        latest: dict[str, tuple] = {}
        for v in vers:
            if v[0] > t:
                break
            latest[v[1]] = v
        live = [v for v in latest.values() if v[4] != "DELETE" and v[2] is not None]
        if not live:
            return None
        return float(max(live, key=lambda v: (v[3], v[0]))[2])

    def known_max(self, d: date, t: datetime, lag: timedelta = POLL_LAG) -> float:
        mags = [self.displayed(vers, t - lag) for _, vers in self.by_day.get(d, [])]
        mags = [m for m in mags if m is not None]
        return max(mags) if mags else -np.inf

    def first_cross(self, d: date, m: float) -> datetime | None:
        """First time a displayed magnitude on day d reached m (when a quote would be pulled)."""
        best = None
        for _, vers in self.by_day.get(d, []):
            for v in vers:
                if v[2] is not None and v[2] >= m - 1e-9 and self.displayed(vers, v[0]) is not None \
                        and self.displayed(vers, v[0]) >= m - 1e-9:
                    best = v[0] if best is None or v[0] < best else best
                    break
        return best


# ---- H4: release race on the quake ladder -----------------------------------------------------------------
def first_cross_id(qk: pl.DataFrame, d: date, m: float) -> dict[str, Any] | None:
    d0 = datetime(d.year, d.month, d.day, tzinfo=UTC)
    x = qk.filter((pl.col("time") >= d0) & (pl.col("time") < d0 + timedelta(days=1)) & (pl.col("mag") >= m - 1e-9)).sort("time")
    return x.row(0, named=True) if x.height else None


def usgs_first_publication(payload: dict[str, Any]) -> dict[str, Any]:
    """Earliest product update time (any product, and origin products only) and the magnitude carried
    by the earliest origin product, from a detail query with ``includesuperseded=true``."""
    props = payload.get("properties") or {}
    first_any = first_origin = None
    mag0 = None
    for typ, prods in (props.get("products") or {}).items():
        for x in prods:
            ut = datetime.fromtimestamp(x["updateTime"] / 1000, UTC)
            if first_any is None or ut < first_any:
                first_any = ut
            if typ == "origin" and (first_origin is None or ut < first_origin):
                first_origin = ut
                mag0 = num((x.get("properties") or {}).get("magnitude"))
    return {"pub_any": first_any, "pub_origin": first_origin, "mag_first": mag0}


def release_race(name: str = "quake", out: Path = OUT, *, log=print) -> pl.DataFrame:
    """For every YES strike: the first crossing quake, its first USGS publication, Kalshi's 1-minute
    asks around it and the prints after publication (stale asks someone took)."""
    from albert.venues.kalshi import candle_rows

    mk = (pl.read_parquet(out / f"{name}_markets.parquet").filter((pl.col("series_ticker") == QUAKE_SERIES)
                                                                 & (pl.col("result") == "yes"))
          .with_columns(pl.col("event_ticker").map_elements(quake_day, return_dtype=pl.Date).alias("day"))
          .filter(pl.col("day").is_not_null()))
    qk = pl.read_parquet(out / "quakes.parquet")
    tr = pl.read_parquet(out / f"{name}_trades.parquet")
    c, api = _usgs(), _api()
    cutoff = api.cutoff().get("market_settled_ts")
    rows = []
    for r in mk.sort("close_time").iter_rows(named=True):
        q = first_cross_id(qk, r["day"], r["floor_strike"])
        if q is None:
            rows.append({"ticker": r["ticker"], "event_ticker": r["event_ticker"], "strike": r["floor_strike"],
                         "quake_id": None})
            continue
        det = c.get("/fdsnws/event/1/query", {"format": "geojson", "eventid": q["id"], "includesuperseded": "true"})
        pub = usgs_first_publication(det.payload)
        t0 = q["time"]
        lo, hi = t0 - timedelta(minutes=10), t0 + timedelta(minutes=120)
        hist = bool(cutoff and r["close_time"] < cutoff)
        try:
            if hist:
                page = api.candles(r["ticker"], lo, hi, 1, True)
                cr = candle_rows(page.payload, r["ticker"], 1)
            else:
                page = api.candles_batch([r["ticker"]], lo, hi, 1)
                cr = [x for node in page.payload.get("markets") or [] for x in candle_rows(node, node.get("market_ticker"), 1)]
        except RuntimeError as error:
            log(f"skip minute candles {r['ticker']}: {str(error)[:100]}")
            cr = []
        cm = pl.DataFrame(cr, schema=CANDLE_COLS) if cr else pl.DataFrame(schema=CANDLE_COLS)
        pa = pub["pub_origin"] or pub["pub_any"]

        def ask_at(t: datetime | None, cm: pl.DataFrame = cm) -> float | None:
            if t is None or cm.is_empty():
                return None
            x = cm.filter(pl.col("end_ts") <= t).sort("end_ts")
            return float(x["yes_ask_close"][-1]) if x.height and x["yes_ask_close"][-1] is not None else None
        p = tr.filter(pl.col("ticker") == r["ticker"])
        after = p.filter(pl.col("created_time") > t0)
        hi90 = after.filter(pl.col("yes_price") >= 0.90)
        late = p.filter((pl.col("created_time") > (pa or t0) + timedelta(seconds=60)) & (pl.col("taker_side") == "yes")
                        & (pl.col("yes_price") <= 0.95))
        gain = float(((1 - late["yes_price"]) * late["count"]).sum()) if late.height else 0.0
        rows.append({"ticker": r["ticker"], "event_ticker": r["event_ticker"], "strike": r["floor_strike"],
                     "quake_id": q["id"], "quake_mag": q["mag"], "origin": t0, **pub, "close_time": r["close_time"],
                     "pub_lag_min": (pa - t0).total_seconds() / 60 if pa else None,
                     "first_print_ge90_min": (hi90["created_time"].min() - t0).total_seconds() / 60 if hi90.height else None,
                     "close_lag_min": (r["close_time"] - t0).total_seconds() / 60,
                     "ask_at_origin": ask_at(t0), "ask_pub_plus1": ask_at(pa + timedelta(minutes=1) if pa else None),
                     "ask_pub_plus2": ask_at(pa + timedelta(minutes=2) if pa else None),
                     "ask_pub_plus5": ask_at(pa + timedelta(minutes=5) if pa else None),
                     "stale_prints_after_pub60": late.height, "stale_contracts": float(late["count"].sum()) if late.height else 0.0,
                     "stale_gross_usd": round(gain, 2)})
    df = pl.DataFrame(rows, infer_schema_length=None)
    _save(df, out / f"{name}_race.parquet")
    log(f"race: {df.height} YES strikes, {df['quake_id'].n_unique()} crossing quakes")
    return df


# ---- H4 as a frozen rule: take stale YES asks after a USGS publication -----------------------------------
RACE_MARGINS = (0.0, 0.2, 0.4)        # strike <= displayed magnitude - margin (revisions can go down)
RACE_LIMITS = (0.90, 0.95, 0.97)      # highest YES price we pay
RACE_DELAYS = (timedelta(seconds=90), timedelta(seconds=180))   # poll the 1-min feed + act
RACE_WINDOW = timedelta(minutes=30)
RACE_MAX_N = 10


def race_triggers(vs: pl.DataFrame) -> pl.DataFrame:
    """Every time the day's displayed maximum magnitude rises (a new origin version anywhere), as
    USGS showed it: (day, time, displayed max)."""
    rt = Realtime(vs)
    out = []
    for d, evs in rt.by_day.items():
        times = sorted({v[0] for _, vers in evs for v in vers if v[0].date() <= d + timedelta(days=1)})
        best = -np.inf
        for t in times:
            mx = rt.known_max(d, t, timedelta(0))
            if mx > best + 1e-9:
                out.append({"day": d, "t_pub": t, "disp_max": mx})
                best = mx
    return pl.DataFrame(out, infer_schema_length=None).sort("t_pub")


def race_fills(mk: pl.DataFrame, trades: pl.DataFrame, trig: pl.DataFrame, margin: float, limit: float,
               delay: timedelta) -> pl.DataFrame:
    """Buy YES (taker) on each strike that the displayed maximum passed by ``margin``: our fills are the
    YES-taker prints at <= ``limit`` from trigger + delay for 30 min (liquidity someone took after we
    could have acted), capped at 10 contracts, held to settlement, fee rounded up per order."""
    tape = trades.filter(pl.col("taker_side") == "yes").sort("ticker", "created_time")
    by = {tk: g for (tk,), g in tape.group_by(["ticker"], maintain_order=True)}
    out = []
    for m in mk.iter_rows(named=True):
        tg = trig.filter((pl.col("day") == m["day"]) & (pl.col("disp_max") >= m["floor_strike"] + margin - 1e-9))
        if tg.is_empty():
            continue
        t0 = tg["t_pub"].min() + delay
        if t0 >= m["close_time"]:
            continue
        g = by.get(m["ticker"])
        if g is None:
            continue
        w = g.filter((pl.col("created_time") >= t0) & (pl.col("created_time") < t0 + RACE_WINDOW)
                     & (pl.col("yes_price") <= limit + 1e-9))
        if w.is_empty():
            continue
        n = int(min(RACE_MAX_N, w["count"].sum()))
        if n < 1:
            continue
        price = float((w["yes_price"] * w["count"]).sum() / w["count"].sum())
        price = round(min(limit, price), 2)
        fee = taker_fee(n, price)
        y = int(m["result"] == "yes")
        out.append({"ticker": m["ticker"], "event_ticker": m["event_ticker"], "day": m["day"], "strike": m["floor_strike"],
                    "h": 0, "t": t0, "t_fill": w["created_time"][0], "settlement_ts": m["settlement_ts"], "y": y,
                    "side": "yes", "price": price, "n": n, "fee_order": fee, "payoff": y,
                    "pnl_c": 100 * (y - price - fee / n),
                    "split": "dev" if m["close_time"] < DEV_END else ("holdout" if m["close_time"] < HOLDOUT_END else "later")})
    return pl.DataFrame(out, infer_schema_length=None)


def race_eval(name: str = "quake", out: Path = OUT, *, log=print) -> dict[str, Any]:
    vs = pl.read_parquet(out / "quake_versions.parquet")
    trig = race_triggers(vs)
    _save(trig, out / "quake_race_triggers.parquet")
    mk = (pl.read_parquet(out / f"{name}_markets.parquet").filter((pl.col("series_ticker") == QUAKE_SERIES)
                                                                 & pl.col("result").is_in(["yes", "no"]))
          .with_columns(pl.col("event_ticker").map_elements(quake_day, return_dtype=pl.Date).alias("day"))
          .filter(pl.col("day").is_not_null()))
    trades = pl.read_parquet(out / f"{name}_trades.parquet")
    res, fills = [], {}
    for mg in RACE_MARGINS:
        for lim in RACE_LIMITS:
            for dl in RACE_DELAYS:
                f = race_fills(mk, trades, trig, mg, lim, dl)
                cid = f"race|{mg:.1f}|{lim:.2f}|{int(dl.total_seconds())}s"
                for split in ("dev", "holdout"):
                    g = f.filter(pl.col("split") == split) if not f.is_empty() else f
                    fills[(cid, split)] = g
                    res.append({"cell": cid, "split": split, **cell_stats(g),
                                "losses": int((g["y"] == 0).sum()) if not g.is_empty() else 0})
    grid = pl.DataFrame(res, infer_schema_length=None)
    grid.write_csv(out / "race_grid.csv")
    dev = grid.filter(pl.col("split") == "dev")
    prim_path = out / "race_primary.json"
    if prim_path.exists():
        prim = json.loads(prim_path.read_text())
    else:
        prim = freeze_primary(dev.with_columns(pl.lit(0).alias("dummy")), prim_path)
    result: dict[str, Any] = {"cells": dev.height, "primary": prim.get("cell"), "triggers": trig.height}
    if prim.get("cell"):
        for split in ("dev", "holdout"):
            f = fills[(prim["cell"], split)]
            led = ledger(f, prim["cell"], split, event_cap=20.0) if not f.is_empty() else pl.DataFrame()
            if not led.is_empty():
                _save(led, out / "ledgers" / f"ledger_race_{split}.parquet")
            row = grid.filter((pl.col("cell") == prim["cell"]) & (pl.col("split") == split)).row(0, named=True)
            result[split] = {**row, "book": book_metrics(led) if not led.is_empty() else {"positions": 0}}
    (out / "race_results.json").write_text(json.dumps(result, indent=1, default=str))
    log(json.dumps(result, indent=1, default=str))
    return result


# ---- hurricanes (KXHURCAT): HURDAT2 climatology + 2026 b-decks (www/ftp.nhc.noaa.gov allow us) ----------
HURDAT_FILES = {"AL": "/data/hurdat/hurdat2-1851-2025-02272026.txt",          # released 2026-02-27: before
                "EP": "/data/hurdat/hurdat2-nepac-1949-2025-02272026.txt"}    # every 2026 decision
CAT_KT = {1: 64, 2: 83, 3: 96, 4: 113, 5: 137}    # Saffir-Simpson thresholds (74/96/111/130/157 mph)
TROPICAL = {"TD", "TS", "HU", "SS", "SD", "TY", "ST"}
NAMED = {"TS", "HU", "SS", "TY", "ST"}
ADVISORY_LAG = timedelta(hours=3)                 # a synoptic fix is public with the advisory ~3 h later
NAME_LISTS = {
    (2022, "AL"): "Alex Bonnie Colin Danielle Earl Fiona Gaston Hermine Ian Julia Karl Lisa Martin Nicole Owen Paula "
                  "Richard Shary Tobias Virginie Walter",
    (2023, "AL"): "Arlene Bret Cindy Don Emily Franklin Gert Harold Idalia Jose Katia Lee Margot Nigel Ophelia "
                  "Philippe Rina Sean Tammy Vince Whitney",
    (2023, "EP"): "Adrian Beatriz Calvin Dora Eugene Fernanda Greg Hilary Irwin Jova Kenneth Lidia Max Norma Otis "
                  "Pilar Ramon Selma Todd Veronica Wiley Xina York Zelda",
    (2024, "AL"): "Alberto Beryl Chris Debby Ernesto Francine Gordon Helene Isaac Joyce Kirk Leslie Milton Nadine "
                  "Oscar Patty Rafael Sara Tony Valerie William",
    (2025, "AL"): "Andrea Barry Chantal Dexter Erin Fernand Gabrielle Humberto Imelda Jerry Karen Lorenzo Melissa "
                  "Nestor Olga Pablo Rebekah Sebastien Tanya Van Wendy",
    (2026, "AL"): "Arthur Bertha Cristobal Dolly Edouard Fay Gonzalo Hanna Isaias Josephine Kyle Leah Marco Nana Omar "
                  "Paulette Rene Sally Teddy Vicky Wilfred",
    (2026, "EP"): "Amanda Boris Cristina Douglas Elida Fausto Genevieve Hernan Iselle Julio Karina Lowell Marie Norbert "
                  "Odalys Polo Rachel Simon Trudy Vance Winnie Xavier Yolanda Zeke",
    (2026, "CP"): "Lala Moke Nolo Olana Pena Ulana Wale",     # Hawaiian list continues after Iona/Keli (2025)
}
HUR_DECISION_HOUR = 15                              # daily decision at 15:00 UTC (after the 15Z advisory)


def _nhc(host: str = "https://www.nhc.noaa.gov"):
    from albert.http import CachedClient

    return CachedClient("nhc" if "www" in host else "nhcftp", host, CACHE, min_interval=1.0, max_retries=6, user_agent=UA)


def hurdat_rows(text: str) -> list[dict[str, Any]]:
    """HURDAT2 text -> one row per fix (storm id, name, time, status, max wind kt)."""
    rows, sid, name = [], None, None
    for ln in text.splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) < 4:
            continue
        if re.match(r"^(AL|EP|CP)\d{6}$", f[0]):
            sid, name = f[0], f[1]
            continue
        if sid is None or not re.match(r"^\d{8}$", f[0]):
            continue
        t = datetime.strptime(f[0] + f[1].zfill(4), "%Y%m%d%H%M").replace(tzinfo=UTC)
        rows.append({"storm_id": sid, "basin": sid[:2], "year": int(sid[4:8]), "name": name.title(), "time": t,
                     "status": f[3], "vmax": float(f[6]) if f[6] not in ("", "-99", "-999") else None, "src": "hurdat"})
    return rows


def bdeck_rows(text: str) -> list[dict[str, Any]]:
    """ATCF b-deck text -> one row per (fix time), keeping the storm's name as carried on that line."""
    rows = []
    for ln in text.splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) < 28 or f[4] != "BEST":
            continue
        t = datetime.strptime(f[2], "%Y%m%d%H").replace(tzinfo=UTC)
        rows.append({"storm_id": f"{f[0]}{f[1]}{f[2][:4]}", "basin": f[0], "year": int(f[2][:4]),
                     "name": f[27].title(), "time": t, "status": f[10],
                     "vmax": float(f[8]) if f[8] not in ("", "-99") else None, "src": "bdeck"})
    return rows


def fetch_tracks(out: Path = OUT, *, log=print) -> pl.DataFrame:
    """HURDAT2 (AL + NE/CP, through 2025) and every 2026 b-deck (AL/EP/CP) -> tracks.parquet."""
    www, ftp = _nhc(), _nhc("https://ftp.nhc.noaa.gov")
    rows: list[dict[str, Any]] = []
    for path in HURDAT_FILES.values():
        rows.extend(hurdat_rows(www.get(path).payload))
    listing = ftp.get("/atcf/btk/", tag=datetime.now(UTC).strftime("%Y-%m-%d")).payload
    for fn in sorted(set(re.findall(r'href="(b(?:al|ep|cp)\d{2}2026\.dat)"', listing))):
        rows.extend(bdeck_rows(ftp.get(f"/atcf/btk/{fn}", tag=datetime.now(UTC).strftime("%Y-%m-%d")).payload))
    tr = (pl.DataFrame(rows, infer_schema_length=None).unique(["storm_id", "time", "src"], keep="last")
          .sort("storm_id", "time"))
    _save(tr, out / "tracks.parquet")
    log(f"tracks: {tr.height} fixes, {tr['storm_id'].n_unique()} storms")
    return tr


def storm_table(tr: pl.DataFrame) -> pl.DataFrame:
    """One row per storm: formation (first named fix), the name it carried when named, lifetime max
    wind over tropical fixes, last fix. 2026 comes from b-decks only, earlier years from HURDAT2.
    CP-formed storms carry basin CP (HURDAT2 ids CP.., b-decks bcp)."""
    tr = tr.filter(((pl.col("year") >= 2026) & (pl.col("src") == "bdeck")) | ((pl.col("year") < 2026) & (pl.col("src") == "hurdat")))
    named = tr.filter(pl.col("status").is_in(list(NAMED)) & (pl.col("vmax") >= 34))
    first = named.sort("time").group_by("storm_id").agg(pl.col("time").first().alias("formed"),
                                                        pl.col("name").last().alias("name"))
    agg = (tr.filter(pl.col("status").is_in(list(TROPICAL))).group_by("storm_id")
           .agg(pl.col("basin").first(), pl.col("year").first(), pl.col("vmax").max().alias("vmax_life"),
                pl.col("time").max().alias("last_fix")))
    return first.join(agg, on="storm_id", how="inner").sort("formed")


class HurClimate:
    """Season climatology from HURDAT2 seasons before the decision's season (30 years)."""

    def __init__(self, storms: pl.DataFrame, tracks: pl.DataFrame) -> None:
        self.st = storms
        self.seasons: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        for (b, y), g in storms.group_by(["basin", "year"]):
            g = g.sort("formed")
            doy = np.array([t.timetuple().tm_yday + t.hour / 24 for t in g["formed"].to_list()])
            self.seasons[(b, y)] = (doy, g["vmax_life"].fill_null(0).to_numpy().astype(float))
        self.tr = tracks.filter(pl.col("src") == "hurdat").join(
            storms.select("storm_id", "vmax_life"), on="storm_id", how="inner").sort("storm_id", "time")
        self.tr = self.tr.with_columns(pl.col("vmax").cum_max().over("storm_id").alias("vmax_sofar"))

    @staticmethod
    def _basin_set(basin: str) -> list[str]:
        return ["EP"] if basin == "EP" else (["CP"] if basin == "CP" else ["AL"])

    def p_unformed(self, basin: str, season: int, t: datetime, j: int, thr: float) -> tuple[float, int]:
        """P(the j-th storm named after the day-of-year of ``t`` exists and reaches ``thr`` kt)."""
        doy = t.timetuple().tm_yday + t.hour / 24
        hits = n = 0
        for y in range(season - 30, season):
            n += 1
            doys, vmax = self.seasons.get((basin, y), (np.array([]), np.array([])))
            after = vmax[doys > doy]
            if len(after) >= j and after[j - 1] >= thr:
                hits += 1
        return (hits + 0.5) / (n + 1.0), n

    def p_active(self, basin: str, season: int, v: float, thr: float) -> tuple[float, int]:
        """P(lifetime max >= thr | current wind within 5 kt of v, max so far < thr), past 30 seasons."""
        b = self._basin_set(basin) if basin != "CP" else ["EP", "CP"]
        r = self.tr.filter(pl.col("basin").is_in(b) & pl.col("year").is_between(season - 30, season - 1)
                           & pl.col("status").is_in(list(TROPICAL)) & ((pl.col("vmax") - v).abs() <= 5)
                           & (pl.col("vmax_sofar") < thr))
        n = r["storm_id"].n_unique()
        if n == 0:
            return float("nan"), 0
        g = r.group_by("storm_id").agg(pl.col("vmax_life").first())
        hits = int((g["vmax_life"] >= thr).sum())
        return (hits + 0.5) / (n + 1.0), n


def hur_event_name(event_ticker: str, open_year: int | None = None) -> tuple[int, str] | None:
    """``KXHURCAT-26FAY`` -> (2026, 'Fay'); ``KXHURCAT-JERRY`` -> (2025, 'Jerry'); legacy
    ``HURCAT-EARL`` -> (year the market opened, 'Earl'); letter-coded ``HURCAT-21-I`` -> None."""
    m = re.match(r"^(KX)?HURCAT-(\d{2})?([A-Z]{2,})$", event_ticker or "")
    if not m:
        return None
    if m.group(2):
        season = 2000 + int(m.group(2))
    elif m.group(1):
        season = 2025
    elif open_year is not None:
        season = open_year
    else:
        return None
    return season, m.group(3).title()


def name_basin(season: int, name: str) -> tuple[str, int] | None:
    for (y, b), lst in NAME_LISTS.items():
        if y == season and name in lst.split():
            return b, lst.split().index(name) + 1
    return None


def hur_decisions(name: str = "hur", out: Path = OUT, *, log=print) -> pl.DataFrame:
    """One row per (market, day at 15:00 UTC) while open: storm state as of t - 3 h and the model."""
    mk = pl.read_parquet(out / f"{name}_markets.parquet").filter(pl.col("series_ticker").is_in(["KXHURCAT", "HURCAT"]))
    tr = pl.read_parquet(out / "tracks.parquet")
    st = storm_table(tr)
    clim = HurClimate(st, tr)
    qa = _quote_arrays(pl.read_parquet(out / "candles" / f"{name}_60m.parquet"))
    cur = tr.filter(((pl.col("year") >= 2026) & (pl.col("src") == "bdeck")) | ((pl.col("year") < 2026) & (pl.col("src") == "hurdat")))
    cache: dict[tuple, tuple[float, int]] = {}
    rows = []
    for r in mk.iter_rows(named=True):
        en = hur_event_name(r["event_ticker"], r["open_time"].year)
        if en is None:
            continue
        season, nm = en
        nb = name_basin(season, nm)
        if nb is None:
            continue
        basin, k = nb
        if r["floor_strike"] is not None:
            cat = round(r["floor_strike"])
        else:
            mt = re.search(r"-T(\d)$", r["ticker"])
            if not mt:
                continue
            cat = int(mt.group(1))
        thr = CAT_KT.get(cat)
        if thr is None:
            continue
        storm = st.filter((pl.col("year") == season) & (pl.col("name") == nm))   # CP names can carry EP ids
        names_in_list = NAME_LISTS[(season, basin)].split()
        srow = storm.row(0, named=True) if storm.height else None
        fx = cur.filter(pl.col("storm_id") == srow["storm_id"]).sort("time") if srow else None
        cross = None
        if fx is not None:
            hit = fx.filter(pl.col("status").is_in(list(TROPICAL)) & (pl.col("vmax") >= thr))
            cross = hit["time"].min() + ADVISORY_LAG if hit.height else None
        t = r["open_time"].replace(hour=HUR_DECISION_HOUR, minute=0, second=0, microsecond=0)
        if t < r["open_time"]:
            t += timedelta(days=1)
        while t < r["close_time"]:
            known = t - ADVISORY_LAG
            state, p, n, v = "unformed", None, 0, None
            if srow is not None and srow["formed"] <= known:
                seen = fx.filter(pl.col("time") <= known)
                sofar = seen.filter(pl.col("status").is_in(list(TROPICAL)))["vmax"].max() or 0
                v = float(seen["vmax"][-1])
                if sofar >= thr:
                    state, p = "decided_yes", 1.0
                elif srow["last_fix"] <= known - timedelta(hours=6):
                    state, p = "dissipated", 0.0
                else:
                    state = "active"
                    key = ("a", basin, season, v, thr)
                    if key not in cache:
                        cache[key] = clim.p_active(basin, season, v, thr)
                    p, n = cache[key]
            else:
                nsofar = st.filter((pl.col("year") == season) & pl.col("name").is_in(names_in_list)
                                   & (pl.col("formed") <= known)).height
                j = max(k - nsofar, 1)    # CP: the 2026 list starts at the season's first CP name
                key = ("u", basin, season, t.date(), j, thr)
                if key not in cache:
                    cache[key] = clim.p_unformed(basin, season, t, j, thr)
                p, n = cache[key]
            bid, ask = _quote(qa, r["ticker"], t)
            rows.append({"ticker": r["ticker"], "event_ticker": r["event_ticker"], "storm": nm, "basin": basin,
                         "season": season, "strike": float(cat), "h": 0, "day": t.date(), "t": t,
                         "close_time": r["close_time"], "settlement_ts": r["settlement_ts"],
                         "y": int(r["result"] == "yes"), "state": state, "decided": state in ("decided_yes",),
                         "p_clim": p, "n_clim": n, "v_now": v, "cross_time": cross, "bid": bid, "ask": ask,
                         "split": "dev" if r["close_time"] < DEV_END else ("holdout" if r["close_time"] < HOLDOUT_END else "later")})
            t += timedelta(days=1)
    df = pl.DataFrame(rows, infer_schema_length=None)
    _save(df, out / f"{name}_rows.parquet")
    log(f"hurricane rows: {df.height} on {df['ticker'].n_unique()} markets / {df['event_ticker'].n_unique()} storms")
    return df


# ---- diagnostics: does the public-data model add information to the price? ------------------------------
def diagnostics(rows: pl.DataFrame, model: str, by: list[str]) -> pl.DataFrame:
    """Undecided rows with a real two-sided book (0 < bid < ask < 1, <= 15c wide): Brier of the mid vs the
    model, the market premium (mid - model) and the slope of (y - mid) on (model - mid), event-clustered."""
    from albert.research.counts import cluster_se

    r = rows.filter(~pl.col("decided") & pl.col(model).is_not_null() & (pl.col("bid") > 0) & (pl.col("ask") < 1)
                    & (pl.col("ask") > pl.col("bid")) & (pl.col("ask") - pl.col("bid") <= MAX_SPREAD + 1e-9))
    r = r.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    out = []
    for key, g in r.group_by(by, maintain_order=True):
        y, mid, p = (g[c].to_numpy().astype(float) for c in ("y", "mid", model))
        ev = g["event_ticker"].to_numpy().astype(str)
        gap = mid - p
        x, e = p - mid, y - mid
        b = float((x * e).sum() / (x * x).sum()) if (x * x).sum() > 0 else float("nan")
        se_gap = cluster_se(gap, ev) if len(np.unique(ev)) > 1 else float("nan")
        out.append({**dict(zip(by, key, strict=True)), "rows": g.height, "events": len(np.unique(ev)),
                    "mean_mid": round(float(mid.mean()), 4), "mean_model": round(float(p.mean()), 4),
                    "yes_rate": round(float(y.mean()), 4), "brier_mid": round(float(((y - mid) ** 2).mean()), 4),
                    "brier_model": round(float(((y - p) ** 2).mean()), 4),
                    "premium_c": round(100 * float(gap.mean()), 2),
                    "t_premium": round(float(gap.mean()) / se_gap, 2) if se_gap and np.isfinite(se_gap) and se_gap > 0 else None,
                    "slope": round(b, 3)})
    return pl.DataFrame(out, infer_schema_length=None)


# ---- evaluation driver: dev grid -> frozen primary -> holdout once ------------------------------------------
MIN_FILLS, MIN_EVENTS = 30, 10


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (n - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def _pval(t: float | None) -> float:
    import math

    if t is None or not np.isfinite(t):
        return 1.0
    return float(math.erfc(abs(t) / math.sqrt(2)))   # two-sided normal


def freeze_primary(grid: pl.DataFrame, path: Path) -> dict[str, Any]:
    """Dev-only choice, written before any holdout number exists: among cells with >= 30 fills and
    >= 10 events, the highest event-clustered t on $ P&L (ties -> c/contract)."""
    ok = grid.filter((pl.col("fills") >= MIN_FILLS) & (pl.col("events") >= MIN_EVENTS) & pl.col("t_event").is_not_null())
    if ok.is_empty():
        pick = {"cell": None}
    else:
        pick = ok.sort(["t_event", "c_per_contract"], descending=True).row(0, named=True)
    pick = {**pick, "frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "rule": "max dev t_event among cells with >= 30 fills and >= 10 events"}
    path.write_text(json.dumps(pick, indent=1, default=str))
    return pick


def evaluate_family(rows: pl.DataFrame, tape: Tape, fam: str, models: tuple[str, ...], out: Path = OUT,
                    *, buckets: dict[str, pl.Expr] | None = None, bands: dict[str, pl.Expr] | None = None,
                    log=print) -> dict[str, Any]:
    """Dev grid (all cells, Holm), frozen primary, then the holdout of the primary (and the grid for
    the record). Ledgers for the primary in both splits go to ``ledgers/``."""
    dev, dev_fills = quake_cells(rows, tape, "dev", models, buckets, bands)
    # a clustered t needs clusters: cells with < 10 events (or < 30 fills) get p = 1 (counted, never pass)
    dev = dev.with_columns(pl.Series("p_raw", [_pval(t) if (e or 0) >= MIN_EVENTS and (f or 0) >= MIN_FILLS else 1.0
                                               for t, e, f in zip(dev["t_event"].to_list(), dev["events"].to_list(),
                                                                  dev["fills"].to_list(), strict=True)]))
    dev = dev.with_columns(pl.Series("p_holm", holm(dev["p_raw"].to_numpy())))
    dev.write_csv(out / f"{fam}_grid_dev.csv")
    prim_path = out / f"{fam}_primary.json"
    if prim_path.exists():
        prim = json.loads(prim_path.read_text())
        log(f"{fam}: primary already frozen at {prim.get('frozen_utc')}: {prim.get('cell')}")
    else:
        prim = freeze_primary(dev, prim_path)
        log(f"{fam}: froze primary {prim.get('cell')}")
    hold, hold_fills = quake_cells(rows, tape, "holdout", models, buckets, bands)
    hold.write_csv(out / f"{fam}_grid_holdout.csv")
    res: dict[str, Any] = {"family": fam, "cells": dev.height, "dev_cells_holm_005": int((dev["p_holm"] < 0.05).sum()),
                           "primary": prim.get("cell")}
    if prim.get("cell"):
        for split, fills in (("dev", dev_fills), ("holdout", hold_fills)):
            f = fills[prim["cell"]]
            led = ledger(f, prim["cell"], split) if not f.is_empty() else pl.DataFrame()
            if not led.is_empty():
                _save(led, out / "ledgers" / f"ledger_{fam}_{split}.parquet")
            g = (dev if split == "dev" else hold).filter(pl.col("cell") == prim["cell"]).row(0, named=True)
            wk = None
            if not f.is_empty():
                w = (f.with_columns((pl.col("pnl_c") * pl.col("n") / 100).alias("usd"))
                     .group_by(pl.col("t_fill").dt.truncate("1w")).agg(pl.col("usd").sum()))
                wk = round(100 * float((w["usd"] > 0).mean()), 0)
            res[split] = {**{k: g[k] for k in ("fills", "events", "contracts", "c_per_contract", "t_event", "pnl_usd",
                                               "pnl_per_usd", "win_pct")}, "weeks_positive_pct_fills": wk,
                          "book": book_metrics(led) if not led.is_empty() else {"positions": 0}}
    (out / f"{fam}_results.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ---- report tables -----------------------------------------------------------------------------------------
def md_table(df: pl.DataFrame, cols: list[str] | None = None, n: int | None = None) -> str:
    d = df.select(cols) if cols else df
    if n is not None:
        d = d.head(n)
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    for r in d.iter_rows():
        lines.append("| " + " | ".join("" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def write_tables(out: Path = OUT) -> Path:
    """Auto-generated appendix (``tables.md``): triage, settlement check, diagnostics, grids, race."""
    parts = ["# sci_health: generated tables\n", (f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by "
             "`python -m albert.research.sci_health tables`.\n")]
    if (out / "triage_family.csv").exists():
        parts += ["## Triage by family (settled events; dev <= 2026-09-30, holdout 2026-10-01..11-22)\n",
                  md_table(pl.read_csv(out / "triage_family.csv")), ""]
    for fam, model, by in (("quake", "p_static", ["split", "hb"]), ("quake_rt", "p_static", ["split", "hb"]),
                           ("hur", "p_clim", ["split", "state"])):
        rp = out / f"{fam}_rows.parquet"
        if not rp.exists():
            continue
        rows = pl.read_parquet(rp)
        if fam.startswith("quake"):
            rows = rows.with_columns(pl.when(pl.col("h") < 0).then(pl.lit("pre")).when(pl.col("h") < 12)
                                     .then(pl.lit("early")).otherwise(pl.lit("late")).alias("hb"))
            band = rows.with_columns(pl.when(pl.col("strike") <= 5.85).then(pl.lit("5.2-5.8")).when(pl.col("strike") <= 6.45)
                                     .then(pl.lit("6.0-6.4")).otherwise(pl.lit("6.6-7.0")).alias("band"))
            parts += [f"## {fam}: model vs mid by split and band ({model})\n",
                      md_table(diagnostics(band, model, ["split", "band"]).sort("split", "band")), ""]
        parts += [f"## {fam}: model vs mid by split and time bucket ({model})\n",
                  md_table(diagnostics(rows, model, by).sort(by)), ""]
        g = out / f"{fam}_grid_dev.csv"
        if g.exists():
            grid = pl.read_csv(g)
            top = grid.filter(pl.col("fills") >= MIN_FILLS).sort("t_event", descending=True, nulls_last=True)
            parts += [(f"## {fam}: dev grid, top 15 cells by event-clustered t ({grid.height} cells; "
                      f"{int((grid['p_holm'] < 0.05).sum())} pass Holm at 5%)\n"),
                      md_table(top, ["cell", "fills", "events", "c_per_contract", "t_event", "pnl_usd", "pnl_per_usd",
                                     "p_holm"], 15), ""]
            h = out / f"{fam}_grid_holdout.csv"
            if h.exists():
                hold = pl.read_csv(h)
                parts += [f"## {fam}: the same 15 cells on the holdout (for the record; only the frozen primary is confirmatory)\n",
                          md_table(top.head(15).select("cell").join(hold, on="cell", how="left"),
                                   ["cell", "fills", "events", "c_per_contract", "t_event", "pnl_usd"]), ""]
    if (out / "race_grid.csv").exists():
        parts += ["## H4 race rule grid (taker YES on prints after a displayed-magnitude trigger; 18 cells)\n",
                  md_table(pl.read_csv(out / "race_grid.csv")), ""]
    if (out / "quake_race.parquet").exists():
        rc = pl.read_parquet(out / "quake_race.parquet").filter(pl.col("quake_id").is_not_null())
        parts += ["## quake release race (YES strikes)\n", md_table(rc.select(
            pl.len().alias("strikes"), pl.col("quake_id").n_unique().alias("quakes"),
            pl.col("pub_lag_min").median().alias("median_pub_lag_min"),
            pl.col("first_print_ge90_min").median().alias("median_first_print_ge90_min"),
            pl.col("close_lag_min").median().alias("median_close_lag_min"),
            (pl.col("ask_pub_plus2") <= 0.95).sum().alias("ask_le95_at_pub_plus2"),
            pl.col("stale_prints_after_pub60").sum().alias("stale_prints"),
            pl.col("stale_contracts").sum().alias("stale_contracts"),
            pl.col("stale_gross_usd").sum().alias("stale_gross_usd"))), ""]
    path = out / "tables.md"
    path.write_text("\n".join(parts) + "\n")
    return path


# ---- CLI ---------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.sci_health")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("robots", "discover", "triage", "usgs", "tracks", "quake-rows", "hur-rows", "race", "quake-eval",
              "versions", "quake-rows-rt", "quake-eval-rt", "race-eval",
              "hur-eval", "tables"):
        sub.add_parser(c)
    f = sub.add_parser("fetch", help="trades + hourly candles for every settled market of some series")
    f.add_argument("name")
    f.add_argument("series", nargs="+")
    f.add_argument("--markets", default=str(OUT / "markets.parquet"))
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "robots":
        for r in check_robots():
            print(r["host"], r["path"], r["http_status"], r["allowed"])
    elif a.cmd == "discover":
        discover()
    elif a.cmd == "fetch":
        mk = pl.read_parquet(a.markets).filter(pl.col("series_ticker").is_in(a.series)
                                               & pl.col("result").is_in(["yes", "no"]))
        _save(mk, OUT / f"{a.name}_markets.parquet")
        fetch_candles(mk, a.name)
        fetch_trades(mk, a.name)
    elif a.cmd == "usgs":
        fetch_quakes()
    elif a.cmd == "tracks":
        fetch_tracks()
    elif a.cmd == "quake-rows":
        quake_decisions(realtime=False)
    elif a.cmd == "versions":
        fetch_versions()
    elif a.cmd == "race-eval":
        race_eval()
    elif a.cmd == "quake-rows-rt":
        quake_decisions(realtime=True, tag="_rt")
    elif a.cmd == "quake-eval-rt":
        rows = pl.read_parquet(OUT / "quake_rt_rows.parquet")
        print(json.dumps(evaluate_family(rows, Tape(pl.read_parquet(OUT / "quake_trades.parquet")), "quake_rt",
                                         ("p_static", "p_cond")), indent=1, default=str))
    elif a.cmd == "hur-rows":
        hur_decisions()
    elif a.cmd == "race":
        release_race()
    elif a.cmd == "tables":
        print(write_tables())
    elif a.cmd == "quake-eval":
        rows = pl.read_parquet(OUT / "quake_rows.parquet")
        print(json.dumps(evaluate_family(rows, Tape(pl.read_parquet(OUT / "quake_trades.parquet")), "quake",
                                         ("p_static", "p_cond")), indent=1, default=str))
    elif a.cmd == "hur-eval":
        rows = pl.read_parquet(OUT / "hur_rows.parquet").with_columns(pl.lit(0).alias("h"))
        print(json.dumps(evaluate_family(rows, Tape(pl.read_parquet(OUT / "hur_trades.parquet")), "hur",
                                         ("p_clim",), buckets=HUR_BUCKETS, bands=HUR_BANDS), indent=1, default=str))
    elif a.cmd == "triage":
        with pl.Config(tbl_rows=60, tbl_width_chars=220):
            print(triage())


if __name__ == "__main__":
    main()
