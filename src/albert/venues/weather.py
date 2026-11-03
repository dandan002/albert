"""NWS weather data for Kalshi temperature markets.

Kalshi's daily high/low markets (``KXHIGH*``, ``KXLOWT*``) settle on the NWS Daily Climate
Report (CLI) for one station, e.g. ``CLINYC`` = Central Park (KNYC). Three NWS products matter:

* **CLI**: the official daily max/min (the settlement value).
* **ASOS observations**: temperatures through the day (a running lower bound on the max).
* **NBM** (National Blend of Models) text guidance: the forecast NWS forecasters start from,
  including ``txn`` (max/min temperature) and ``xnd`` (its standard deviation).

``api.weather.gov`` serves these live but keeps only about a week of observations and no
past forecasts, so history comes from the Iowa Environmental Mesonet (IEM), which archives
the same NWS products. Live trading uses ``api.weather.gov`` plus IEM's real-time NBM feed.

The CLI "day" is midnight to midnight *local standard time* (1am-1am during daylight time).
"""

import csv
import io
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from albert.http import CachedClient, Fetched

IEM_BASE = "https://mesonet.agron.iastate.edu"
NWS_BASE = "https://api.weather.gov"

# CLI site code (from Kalshi rules, e.g. "CLINYC" -> "NYC") -> IANA time zone.
STATION_TZ = {
    "NYC": "America/New_York", "MIA": "America/New_York", "ATL": "America/New_York",
    "PHL": "America/New_York", "BOS": "America/New_York", "DCA": "America/New_York",
    "EWR": "America/New_York", "TTN": "America/New_York", "SDF": "America/Kentucky/Louisville",
    "MDW": "America/Chicago", "AUS": "America/Chicago", "DFW": "America/Chicago",
    "HOU": "America/Chicago", "OKC": "America/Chicago", "MSP": "America/Chicago",
    "MSY": "America/Chicago", "SAT": "America/Chicago", "DEN": "America/Denver",
    "PHX": "America/Phoenix", "LAX": "America/Los_Angeles", "SFO": "America/Los_Angeles",
    "SEA": "America/Los_Angeles", "LAS": "America/Los_Angeles", "SAN": "America/Los_Angeles",
}

_CLI_RX = re.compile(r"\(CLI([A-Z]{3})\)")


def cli_code(rules: str | None) -> str | None:
    """``"... recorded at New York City (CLINYC) for ..."`` -> ``"NYC"``."""
    m = _CLI_RX.search(rules or "")
    return m.group(1) if m else None


def lst_day_bounds(code: str, day: date) -> tuple[datetime, datetime]:
    """UTC bounds of the CLI day: local *standard* midnight to midnight."""
    tz = ZoneInfo(STATION_TZ[code])
    probe = datetime(day.year, day.month, day.day, 12, tzinfo=tz)
    std = probe.utcoffset() - (probe.dst() or timedelta(0))
    start = datetime(day.year, day.month, day.day, tzinfo=UTC) - std
    return start, start + timedelta(days=1)


class IEM:
    def __init__(self, client: CachedClient) -> None:
        self.client = client

    def cli(self, code: str, year: int, *, cache: bool = True, tag: str | None = None) -> Fetched:
        return self.client.get("/json/cli.py", {"station": f"K{code}", "year": year},
                               cache=cache, tag=tag)

    def asos(self, code: str, start: date, end: date, *, cache: bool = True) -> Fetched:
        """Routine + special METARs, temperatures in F, timestamps in UTC, [start, end)."""
        return self.client.get("/cgi-bin/request/asos.py", {
            "station": code, "data": "tmpf", "tz": "Etc/UTC", "format": "onlycomma",
            "latlon": "no", "missing": "empty", "report_type": "3,4",
            "year1": start.year, "month1": start.month, "day1": start.day,
            "year2": end.year, "month2": end.month, "day2": end.day}, cache=cache)

    def nbm(self, code: str, start: datetime, end: datetime, *, cache: bool = True) -> Fetched:
        """NBM short-range text guidance (NBS) runs in [start, end)."""
        return self.client.get("/cgi-bin/request/mos.py", {
            "station": f"K{code}", "model": "NBS", "format": "csv",
            "sts": start.strftime("%Y-%m-%dT%H:%MZ"), "ets": end.strftime("%Y-%m-%dT%H:%MZ")},
            cache=cache)


class NWS:
    """api.weather.gov (live only). Needs a User-Agent with a contact address."""

    def __init__(self, client: CachedClient) -> None:
        self.client = client

    def latest_observations(self, code: str, start: datetime) -> Fetched:
        return self.client.get(f"/stations/K{code}/observations",
                               {"start": start.strftime("%Y-%m-%dT%H:%M:%SZ")}, cache=False)

    def point(self, lat: float, lon: float) -> Fetched:
        return self.client.get(f"/points/{lat:.4f},{lon:.4f}")

    def station(self, code: str) -> Fetched:
        return self.client.get(f"/stations/K{code}")


# ---- normalizers ------------------------------------------------------------------------------
CLI_SCHEMA = {"code": pl.String, "day": pl.Date, "high": pl.Int64, "low": pl.Int64,
              "product": pl.String}
OBS_SCHEMA = {"code": pl.String, "ts": pl.Datetime("us", "UTC"), "tmpf": pl.Float64}
NBM_SCHEMA = {"code": pl.String, "runtime": pl.Datetime("us", "UTC"), "ftime": pl.Datetime("us", "UTC"),
              "tmp": pl.Float64, "txn": pl.Float64, "xnd": pl.Float64}


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def cli_rows(payload: dict[str, Any], code: str) -> list[dict]:
    """IEM ``json/cli.py``: one row per CLI report day ("M" = missing -> None)."""
    rows = []
    for r in payload.get("results") or []:
        try:
            day = date.fromisoformat(r["valid"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({"code": code, "day": day, "high": _int(r.get("high")), "low": _int(r.get("low")),
                     "product": r.get("product")})
    return rows


def _parse_utc(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.strip()).replace(tzinfo=UTC)
    except (AttributeError, ValueError):
        return None


def asos_rows(text: str, code: str) -> list[dict]:
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        ts, t = _parse_utc(r.get("valid", "")), _float(r.get("tmpf"))
        if ts is not None and t is not None:
            rows.append({"code": code, "ts": ts, "tmpf": t})
    return rows


def nbm_rows(text: str, code: str) -> list[dict]:
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        rt, ft = _parse_utc(r.get("runtime", "")), _parse_utc(r.get("ftime", ""))
        if rt is None or ft is None:
            continue
        rows.append({"code": code, "runtime": rt, "ftime": ft, "tmp": _float(r.get("tmp")),
                     "txn": _float(r.get("txn")), "xnd": _float(r.get("xnd"))})
    return rows


def frame(rows: list[dict], schema: dict) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema)


def nws_obs_rows(payload: dict[str, Any], code: str) -> list[dict]:
    """api.weather.gov observations (degC, sometimes whole degrees) -> F. Missing values skipped."""
    rows = []
    for f in payload.get("features") or []:
        p = f.get("properties") or {}
        c = (p.get("temperature") or {}).get("value")
        ts = p.get("timestamp")
        if c is None or ts is None:
            continue
        rows.append({"code": code, "ts": datetime.fromisoformat(ts).astimezone(UTC),
                     "tmpf": round(c * 9 / 5 + 32, 1)})
    return rows
