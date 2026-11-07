"""Precipitation model vs Kalshi daily rain markets (``KXRAIN``, ``KXRAINWKND``).

Question: does a calibrated multi-model precipitation forecast beat, or add information to, the
Kalshi price after fees on the October 2026 holdout?

**Settlement (rules_primary, verified against the CLI below).** ``KXRAIN-26OCT05-NYC`` resolves
YES if "the total precipitation at CLINYC ... is strictly greater than 0 inches": the NWS Daily
Climate Report (CLI) for one station and one *local standard time* day (midnight to midnight LST,
1am-1am during daylight time). ``KXRAINWKND`` is the same for Saturday through Sunday. Whether a
trace ("T", < 0.01") counts as YES is decided from data (:func:`trace_rule`), not assumed.

**Inputs and when each was knowable** (every feature is built only from data available at the
decision time ``t``):

* *IEM MOS archive*: NBM text (NBS) and GFS MOS (MAV) station guidance, 6/12-h probability of
  precipitation (PoP, >= 0.01") and QPF. A run is used only if runtime + lag <= t (NBS 2 h, GFS
  MOS 5 h; MOS text is out ~1 h / ~4 h after the cycle).
* *IEM ASOS* METARs (routine + special), hourly precipitation ``p01i`` (trace = 0.0001), from
  reports valid at least ``OBS_LAG`` before t: "rain so far" in the CLI day (a row where rain was
  already reported before t is *decided* and is excluded from model fitting and scoring, reported
  separately; a trace so far is a feature) and recent rain whatever the day (the wettest hour in
  the 6 h before the last report, :func:`recent_rain`).
* *IEM CLI archive* (json/cli.py): settlement labels (training) and climatology (2016-2025).

Decision slots (UTC): ``d-1 12z`` (the market opens ~09:10 UTC the day before), ``d 00z`` (previous
evening), ``d 12z`` (morning) and ``d 18z`` (midday). The forecast model is fit on station-days
from 2025-09-01 to 2026-09-30 (labels from the CLI, no market data needed); the market comparison
uses dev = markets with t before 2026-10-01 and holdout = October 2026. Fits are expanding by
month: month M is predicted by a model trained on station-days before M - 2 days (CLI labels are
public by the month's first decision).

Result (2026-11-05, ``run()``; report data/reports/research-fundamental-rain-20261105T230143Z):
trace = NO (203/203 trace days settled NO; CLI label = Kalshi result on all 2,044 KXRAIN legs).
Chosen on dev: "ens" (NBM + GFS MOS PoP + a 10-model Open-Meteo ensemble summary; that source
and its feature groups were removed on 2026-11-06 because the Open-Meteo API hosts' robots.txt is
``Disallow: /``), ridge 30. Dev: the model
adds information (blend c 0.61, t 8.1) and the taker earns +8.1c at theta 0.12 (t 2.9). October
holdout: market Brier 0.0490 vs model 0.0517, dev blend 0.0504 (worse than the mid), taker
+0.4c (t 0.1, 186 trades), blend taker and maker negative. About half the dev edge was the
market over-pricing YES in Aug-Sep (a mid-only recalibration made +1.4c on dev, -1.4c in
October); the market was calibrated in October. No tradable edge.

IEM-only rebuild (2026-11-07, :func:`run_iem`; report
data/reports/research-fundamental-rain-iem-20261107T010544Z). Same splits, as-of rules and
expanding fits. Dev choice: "mos_obs" (NBS + GFS MOS + recent ASOS rain), ridge 1, station-day
log loss 0.2636 (Open-Meteo "ens" was 0.2651, "mos" 0.2657). Blend c (dev fit / October refit,
t by event): <= 3c books, all slots 0.55 (5.8) / 0.32 (2.0) (Open-Meteo: 0.59 / 0.36 (2.1));
morning d 12z <= 3c 0.59 (3.5) / 0.58 (2.6) (Open-Meteo 0.67 / 0.68 (3.6)); evening d 00z
October 0.07 (0.3). The dev-fitted blend does not beat the mid on October Brier (d 12z <= 3c
0.0451 vs 0.0445); taker at the dev theta 0.12 +4.1c (t 0.6, 49 trades). Harness maker (exact trades, trade-through, d 12z until rain / day end, <= 3c,
dev-chosen 8c): blend +4.9c dev (t 5.7, 510 fills), October +0.4c (t 0.2, 300 fills); plain-mid
control -1.6c (t -1.3), mid-only recalibration -0.9c, model alone +1.0c (t 0.7). Round 2's
candle requote check (d 00z -> d 12z, 8c, <= 3c): +1.6c (t 1.1) vs mid +0.4c (Open-Meteo +1.5c,
t 1.3). The information survives without Open-Meteo; a significant maker edge does not.

Rerun: ``rain.fetch_iem`` (cached), ``fundamental.fetch_candles`` for pre-cutoff markets +
:func:`fetch_candles_batch`, then ``rain.run(settings())`` (round-1 report) and
``rain.run_iem(settings())`` (the IEM-only harness check below).
"""

import csv
import io
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

NAME = "rain"
SERIES = ("KXRAIN", "KXRAINWKND")
IEM_BASE = "https://mesonet.agron.iastate.edu"
MOS_LAG = {"NBS": timedelta(hours=2), "GFS": timedelta(hours=5)}
OBS_LAG = timedelta(minutes=10)
TRACE = 0.0001
SLOTS = {"d-2 12z": (-2, 12), "d-1 12z": (-1, 12), "d 00z": (0, 0), "d 12z": (0, 12),
         "d 18z": (0, 18)}
MARKET_SLOTS = ("d-1 12z", "d 00z", "d 12z", "d 18z")
WX_START, WX_END = date(2025, 9, 1), date(2026, 11, 2)
CLIMO_YEARS = tuple(range(2016, 2026))
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)

# CLI site code (from Kalshi rules: "CLINYC" or "KNYC") -> IANA zone. The CLI day is LST.
STATION_TZ = {
    **{c: "America/New_York" for c in ("NYC", "MIA", "DCA", "PHL", "ATL", "BOS", "TTN", "EWR",
                                       "CMH", "PVD", "PIT", "TPA", "DTW", "DJT", "LEX")},
    "IND": "America/Indiana/Indianapolis",
    **{c: "America/Chicago" for c in ("AUS", "MSP", "DFW", "HOU", "OKC", "MSY", "ORD", "SAT",
                                      "CLL", "MKE", "SGF", "STL")},
    "DEN": "America/Denver", "ABQ": "America/Denver", "PHX": "America/Phoenix",
    **{c: "America/Los_Angeles" for c in ("LAX", "LAS", "SEA", "SFO")},
}
# West Palm Beach (Palm Beach Intl) was renamed in 2026: CLIDJT now, CLIPBI / KPBI before.
OLD_ID = {"DJT": "PBI"}

_CODE_RX = re.compile(r"\bat (?:CLI|K)([A-Z]{3}) in ")
_DAY_RX = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})$")
_MON = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


def rule_code(rules: str | None) -> str | None:
    """``"... precipitation at CLINYC in New York City ..."`` (or ``"at KMKE in"``) -> ``"NYC"``."""
    m = _CODE_RX.search(rules or "")
    return m.group(1) if m else None


def event_day(event_ticker: str) -> date | None:
    m = _DAY_RX.search(event_ticker or "")
    if not m or m.group(2) not in _MON:
        return None
    return date(2000 + int(m.group(1)), _MON[m.group(2)], int(m.group(3)))


def lst_bounds(code: str, day: date) -> tuple[datetime, datetime]:
    """UTC bounds of the CLI day: local *standard* midnight to midnight."""
    tz = ZoneInfo(STATION_TZ[code])
    probe = datetime(day.year, day.month, day.day, 12, tzinfo=tz)
    std = probe.utcoffset() - (probe.dst() or timedelta(0))
    start = datetime(day.year, day.month, day.day, tzinfo=UTC) - std
    return start, start + timedelta(days=1)


def slot_time(day: date, slot: str) -> datetime:
    off, hour = SLOTS[slot]
    d = day + timedelta(days=off)
    return datetime(d.year, d.month, d.day, hour, tzinfo=UTC)


# ---- parsers (tested on recorded IEM payloads) ----------------------------------------------
def _precip(v: Any) -> float | None:
    if v in (None, "", "M"):
        return None
    if v == "T":
        return TRACE
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def cli_precip_rows(payload: dict[str, Any], code: str) -> list[dict[str, Any]]:
    """IEM json/cli.py -> one row per CLI day: precip in inches, trace = 0.0001, missing = None."""
    rows = []
    for r in payload.get("results") or []:
        try:
            day = date.fromisoformat(r["valid"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({"code": code, "day": day, "precip": _precip(r.get("precip")),
                     "product": r.get("product")})
    return rows


def asos_precip_rows(text: str, code: str) -> list[dict[str, Any]]:
    """IEM asos.py (data=p01i, latlon=yes): precip since the last routine METAR, trace = 0.0001."""
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        try:
            ts = datetime.fromisoformat(r["valid"].strip()).replace(tzinfo=UTC)
        except (KeyError, AttributeError, ValueError):
            continue
        rows.append({"code": code, "ts": ts, "p01i": _precip(r.get("p01i")),
                     "lat": _float(r.get("lat")), "lon": _float(r.get("lon"))})
    return rows


def _float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mos_rows(text: str, code: str) -> list[dict[str, Any]]:
    """IEM mos.py CSV (NBS or GFS MOS): PoP (%) and QPF for 6-h / 12-h periods ending at ftime.
    NBS QPF is in inches; GFS MOS QPF is a category (0 none, 1 0.01-0.09", 2 0.10-0.24", ...)."""
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        try:
            rt = datetime.fromisoformat(r["runtime"].strip()).replace(tzinfo=UTC)
            ft = datetime.fromisoformat(r["ftime"].strip()).replace(tzinfo=UTC)
        except (KeyError, AttributeError, ValueError):
            continue
        rows.append({"code": code, "model": r.get("model"), "runtime": rt, "ftime": ft,
                     "p06": _float(r.get("p06")), "p12": _float(r.get("p12")),
                     "q06": _float(r.get("q06")), "q12": _float(r.get("q12"))})
    return rows


# ---- fetching (cached under data/cache/rain/) ------------------------------------------------
def _cache_dir(s) -> Path:
    return s.data_dir / "cache" / NAME


def _table(s, name: str) -> Path:
    return s.data_dir / "cache" / f"fundamental_{NAME}_{name}.parquet"


def _write(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def stations_in_markets(markets: pl.DataFrame) -> list[str]:
    codes = {rule_code(r) for r in markets["rules_primary"].to_list()}
    return sorted(c for c in codes if c in STATION_TZ)


def fetch_iem(s, codes: list[str], *, log=print) -> None:
    """CLI (2016-2026), ASOS precip and NBS / GFS MOS runs (WX_START..WX_END) for ``codes``."""
    from albert.http import CachedClient, HttpError

    iem = CachedClient("iem", IEM_BASE, _cache_dir(s), min_interval=1.5, max_retries=8)

    def ids(code: str) -> list[str]:
        return [code] + ([OLD_ID[code]] if code in OLD_ID else [])

    obs: list[dict[str, Any]] = []
    for code in codes:
        for sid in ids(code):
            text = iem.get("/cgi-bin/request/asos.py", {
                "station": sid, "data": "p01i", "tz": "Etc/UTC", "format": "onlycomma",
                "latlon": "yes", "missing": "M", "trace": "T", "report_type": "3,4",
                "year1": WX_START.year, "month1": WX_START.month, "day1": WX_START.day,
                "year2": WX_END.year, "month2": WX_END.month, "day2": WX_END.day}).payload
            obs.extend(asos_precip_rows(text, code) if isinstance(text, str) else [])
    log(f"asos rows {len(obs)}")
    _write(pl.DataFrame(obs, infer_schema_length=None).unique(["code", "ts"], keep="first").sort("code", "ts"),
           _table(s, "obs"))
    cli: list[dict[str, Any]] = []
    for code in codes:
        for year in (*CLIMO_YEARS, 2025, 2026):
            for sid in ids(code):
                try:
                    page = iem.get("/json/cli.py", {"station": f"K{sid}", "year": year})
                except HttpError as e:
                    log(f"cli {sid} {year}: {str(e)[:100]}")
                    continue
                cli.extend(cli_precip_rows(page.payload, code))
    log(f"cli rows {len(cli)}")
    _write(pl.DataFrame(cli, infer_schema_length=None).unique(["code", "day"], keep="last").sort("code", "day"),
           _table(s, "cli"))
    mos: list[dict[str, Any]] = []
    for code in codes:
        for model in ("NBS", "GFS"):
            for sid in ids(code):
                lo = datetime(WX_START.year, WX_START.month, 1, tzinfo=UTC)
                while lo.date() < WX_END:
                    hi = min(datetime(lo.year + (lo.month + 2) // 12, (lo.month + 2) % 12 + 1, 1, tzinfo=UTC),
                             datetime(WX_END.year, WX_END.month, WX_END.day, tzinfo=UTC))
                    try:
                        text = iem.get("/cgi-bin/request/mos.py", {
                            "station": f"K{sid}", "model": model, "format": "csv",
                            "sts": lo.strftime("%Y-%m-%dT%H:%MZ"),
                            "ets": hi.strftime("%Y-%m-%dT%H:%MZ")}).payload
                    except HttpError as e:
                        log(f"mos {model} {sid} {lo:%Y-%m}: {str(e)[:100]}")
                        text = ""
                    if isinstance(text, str):
                        mos.extend(mos_rows(text, code))
                    lo = hi
        log(f"mos {code}: {len(mos)}")
    _write(pl.DataFrame(mos, infer_schema_length=None).unique(["code", "model", "runtime", "ftime"]),
           _table(s, "mos"))


def load(s, name: str) -> pl.DataFrame:
    return pl.read_parquet(_table(s, name))


# ---- labels, climatology, rain so far --------------------------------------------------------
def trace_rule(markets: pl.DataFrame, cli: pl.DataFrame) -> pl.DataFrame:
    """Kalshi result by CLI precipitation class (none / trace / measurable) for KXRAIN legs:
    decides whether a trace counts as rain."""
    legs = market_legs(markets.filter(pl.col("series_ticker") == "KXRAIN"))
    d = legs.join(cli.select("code", "day", "precip"), on=["code", "day"], how="left")
    cls = (pl.when(pl.col("precip").is_null()).then(pl.lit("missing"))
           .when(pl.col("precip") == 0).then(pl.lit("0"))
           .when(pl.col("precip") < 0.01).then(pl.lit("trace"))
           .otherwise(pl.lit(">=0.01")))
    return (d.with_columns(cls.alias("cli")).group_by("cli")
            .agg(pl.len().alias("legs"), pl.col("y").mean().alias("yes_rate")).sort("cli"))


def label(precip: pl.Expr, trace_counts: bool) -> pl.Expr:
    return (precip > 0) if trace_counts else (precip >= 0.01)


def climatology(cli: pl.DataFrame, trace_counts: bool, years=CLIMO_YEARS, prior: float = 20.0) -> pl.DataFrame:
    """P(rain) by station and calendar month from CLI days in ``years`` (shrunk to the station's
    all-month rate with ``prior`` pseudo-days)."""
    c = cli.filter(pl.col("day").dt.year().is_in(list(years)) & pl.col("precip").is_not_null())
    c = c.with_columns(label(pl.col("precip"), trace_counts).cast(pl.Float64).alias("wet"),
                       pl.col("day").dt.month().alias("month"))
    base = c.group_by("code").agg(pl.col("wet").mean().alias("base"))
    m = c.group_by("code", "month").agg(pl.col("wet").sum().alias("k"), pl.len().alias("n"))
    return (m.join(base, on="code")
            .with_columns(((pl.col("k") + prior * pl.col("base")) / (pl.col("n") + prior)).alias("clim"))
            .select("code", "month", "clim"))


def market_legs(markets: pl.DataFrame) -> pl.DataFrame:
    """Settled rain legs with station code, first CLI day and number of days (1 or 2)."""
    rows = []
    for r in markets.iter_rows(named=True):
        code, day = rule_code(r["rules_primary"]), event_day(r["event_ticker"])
        if code not in STATION_TZ or day is None:
            continue
        ndays = 2 if r["series_ticker"] == "KXRAINWKND" else 1
        rows.append({"ticker": r["ticker"], "event": r["event_ticker"], "series": r["series_ticker"],
                     "code": code, "day": day, "ndays": ndays, "y": int(r["result"] == "yes"),
                     "open_time": r["open_time"], "close_time": r["close_time"]})
    return pl.DataFrame(rows, schema={"ticker": pl.String, "event": pl.String, "series": pl.String,
                                      "code": pl.String, "day": pl.Date, "ndays": pl.Int64, "y": pl.Int64,
                                      "open_time": pl.Datetime("us", "UTC"),
                                      "close_time": pl.Datetime("us", "UTC")})


def windows(codes_days: pl.DataFrame, slot: str) -> pl.DataFrame:
    """(code, day) -> decision time t and the CLI-day window [lo, hi) in UTC."""
    rows = []
    for code, day in codes_days.select("code", "day").unique().iter_rows():
        lo, hi = lst_bounds(code, day)
        rows.append({"code": code, "day": day, "slot": slot, "t": slot_time(day, slot), "lo": lo, "hi": hi})
    return pl.DataFrame(rows, schema={"code": pl.String, "day": pl.Date, "slot": pl.String,
                                      "t": pl.Datetime("us", "UTC"), "lo": pl.Datetime("us", "UTC"),
                                      "hi": pl.Datetime("us", "UTC")})


def rain_so_far(win: pl.DataFrame, obs: pl.DataFrame) -> pl.DataFrame:
    """Adds ``seen`` (max precip reported in the CLI day before t - OBS_LAG; trace = 0.0001) and
    ``seen_n`` (reports in that interval). Only reports whose hour lies inside the day count (a
    METAR at lo+51min sums the hour that started before lo, so reports before lo+1h are skipped)."""
    o = obs.filter(pl.col("p01i").is_not_null()).select("code", "ts", "p01i")
    j = (win.select("code", "day", "slot", "t", "lo").join(o, on="code", how="inner")
         .filter((pl.col("ts") >= pl.col("lo") + timedelta(hours=1)) & (pl.col("ts") <= pl.col("t") - OBS_LAG))
         .group_by("code", "day", "slot").agg(pl.col("p01i").max().alias("seen"), pl.len().alias("seen_n")))
    return win.join(j, on=["code", "day", "slot"], how="left").with_columns(
        pl.col("seen").fill_null(0.0), pl.col("seen_n").fill_null(0))


# ---- forecast features (as of t) ---------------------------------------------------------------
def mos_features(win: pl.DataFrame, mos: pl.DataFrame, model: str) -> pl.DataFrame:
    """Latest MOS run public at t (runtime + lag <= t); the probability of no precipitation over
    the rest of the day (ws, hi] from 6-h PoPs, each block weighted by its overlap with the window
    (independence across blocks), the max block PoP, and the overlap-weighted QPF."""
    lag = MOS_LAG[model]
    m = mos.filter(pl.col("model") == model)
    runs = m.select("code", "runtime").unique().sort("code", "runtime")
    w = (win.select("code", "day", "slot", "t", "lo", "hi")
         .with_columns((pl.col("t") - lag).alias("tl")).sort("code", "tl"))
    w = w.join_asof(runs, left_on="tl", right_on="runtime", by="code", strategy="backward",
                    check_sortedness=False).drop_nulls("runtime")
    w = w.with_columns(pl.max_horizontal("lo", "t").alias("ws"))
    j = w.join(m.filter(pl.col("p06").is_not_null()).select("code", "runtime", "ftime", "p06", "q06"),
               on=["code", "runtime"], how="inner")
    six = timedelta(hours=6)
    j = j.with_columns(((pl.min_horizontal("ftime", "hi") - pl.max_horizontal(pl.col("ftime") - six, "ws"))
                        .dt.total_seconds() / 21600.0).clip(0.0, 1.0).alias("ov"))
    j = j.filter(pl.col("ov") > 0)
    need = w.with_columns(((pl.col("hi") - pl.col("ws")).dt.total_seconds() / 21600.0).alias("need"))
    pre = model.lower()
    agg = (j.group_by("code", "day", "slot")
           .agg((1 - (1 - (pl.col("ov") * pl.col("p06") / 100).clip(0, 0.999)).product()).alias(f"{pre}_pany"),
                (pl.col("p06").max() / 100).alias(f"{pre}_pmax"),
                (pl.col("ov") * pl.col("q06").fill_null(0)).sum().alias(f"{pre}_q"),
                pl.col("ov").sum().alias(f"{pre}_cov")))
    return (need.select("code", "day", "slot", "need").join(agg, on=["code", "day", "slot"], how="inner")
            .with_columns((pl.col(f"{pre}_cov") / pl.col("need")).clip(0, 1).alias(f"{pre}_cov")).drop("need"))


RECENT = timedelta(hours=6)
RECENT_STALE = timedelta(hours=2)


def recent_rain(win: pl.DataFrame, obs: pl.DataFrame) -> pl.DataFrame:
    """ASOS precipitation in the ``RECENT`` hours before t - OBS_LAG, whatever the CLI day (rain
    that is falling or just fell, e.g. the evening before): ``pre_max`` = the largest ``p01i``
    among reports in (u - 6 h, u] where u is the last report valid by t - OBS_LAG (``p01i`` sums
    since the last routine METAR, so this is the wettest hour), ``pre_n`` = reports in that span.
    Null when the last report is more than ``RECENT_STALE`` older than t - OBS_LAG."""
    span = f"{int(RECENT.total_seconds())}s"
    o = (obs.filter(pl.col("p01i").is_not_null()).select("code", "ts", "p01i").sort("code", "ts")
         .with_columns(pl.col("p01i").rolling_max_by("ts", window_size=span).over("code").alias("pre_max"),
                       pl.col("p01i").is_not_null().cast(pl.Int64).rolling_sum_by("ts", window_size=span)
                       .over("code").alias("pre_n")))
    w = (win.select("code", "day", "slot", "t").with_columns((pl.col("t") - OBS_LAG).alias("u"))
         .sort("code", "u"))
    j = w.join_asof(o.select("code", "ts", "pre_max", "pre_n"), left_on="u", right_on="ts", by="code",
                    strategy="backward", check_sortedness=False)
    fresh = pl.col("ts").is_not_null() & (pl.col("ts") >= pl.col("u") - RECENT_STALE)
    return j.select("code", "day", "slot", pl.when(fresh).then(pl.col("pre_max")).alias("pre_max"),
                    pl.when(fresh).then(pl.col("pre_n")).otherwise(0).alias("pre_n"))


def build_features(win: pl.DataFrame, obs: pl.DataFrame, mos: pl.DataFrame, clim: pl.DataFrame) -> pl.DataFrame:
    """IEM-only station-day features at t: rain so far in the CLI day, recent ASOS rain, NBS and
    GFS MOS PoP / QPF from the latest run public at t, climatology and the share of the day left."""
    f = rain_so_far(win, obs)
    f = f.join(recent_rain(win, obs), on=["code", "day", "slot"], how="left")
    for model in ("NBS", "GFS"):
        f = f.join(mos_features(win, mos, model), on=["code", "day", "slot"], how="left")
    f = f.with_columns(pl.col("day").dt.month().alias("month")).join(clim, on=["code", "month"], how="left")
    left = (pl.col("hi") - pl.max_horizontal("lo", "t")).dt.total_seconds() / 86400.0
    return f.with_columns(left.clip(0, 1).alias("left"))


# ---- model -----------------------------------------------------------------------------------
IEM_VARIANT, IEM_LAMBDA = "mos_obs", 1.0      # select_variant on dev station-days, 2026-11-06 (see run_iem)


def _lg(x: np.ndarray, eps: float = 0.005) -> np.ndarray:
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


# Feature groups (IEM only since 2026-11-06): clim = logit climatology + share of the day left;
# nbs / gfs = MOS PoP (any rain in the rest of the day, max block PoP, QPF, missing flag); obs =
# recent ASOS rain (wettest hour in the last 6 h, any-rain flag, missing flag) and a trace already
# reported in the CLI day; city = station intercepts (ridge-shrunk). Round 1's "ens" / "om" groups
# (Open-Meteo NWP runs) were removed with that data source (robots.txt ``Disallow: /``).
VARIANTS: dict[str, tuple[str, ...]] = {
    "climo": ("clim",),
    "nbm": ("clim", "nbs"),
    "mos": ("clim", "nbs", "gfs"),
    "mos_obs": ("clim", "nbs", "gfs", "obs"),
    "full": ("clim", "nbs", "gfs", "obs", "city"),
}
LAMBDAS = (1.0, 30.0, 300.0)


def design(f: pl.DataFrame, parts: tuple[str, ...], cities: list[str]) -> tuple[np.ndarray, list[str]]:
    """Feature matrix (no intercept). Missing guidance is filled with climatology / zero QPF plus
    a missing indicator, so every row gets a prediction."""
    cols, names = [], []
    clim = f["clim"].fill_null(0.3).to_numpy()
    lc = _lg(clim)
    left = f["left"].to_numpy()

    def add(x: np.ndarray, name: str) -> None:
        cols.append(np.asarray(x, dtype=float))
        names.append(name)

    if "clim" in parts:
        add(lc, "logit_clim")
        add(left, "left")
    for src in ("nbs", "gfs"):
        if src in parts:
            pa = f[f"{src}_pany"].to_numpy().astype(float)
            miss = np.isnan(pa)
            add(np.where(miss, lc, _lg(np.nan_to_num(pa, nan=0.5), 0.01)), f"logit_{src}_pany")
            add(np.where(miss, 0.0, _lg(np.nan_to_num(f[f"{src}_pmax"].to_numpy().astype(float), nan=0.5), 0.01)),
                f"logit_{src}_pmax")
            add(np.log1p(np.nan_to_num(f[f"{src}_q"].to_numpy().astype(float), nan=0.0)), f"log_{src}_q")
            add(miss.astype(float), f"{src}_missing")
    if "obs" in parts:
        pm = f["pre_max"].to_numpy().astype(float)
        miss = np.isnan(pm)
        pm = np.nan_to_num(pm, nan=0.0)
        add(np.log1p(100 * pm), "log_pre_max")
        add((pm >= 0.01).astype(float), "pre_wet")
        add(miss.astype(float), "pre_missing")
        add((f["seen"].fill_null(0.0).to_numpy() > 0).astype(float), "trace_today")
    if "city" in parts:
        dummies = (f["code"].to_numpy()[:, None] == np.array(cities)[None, :]).astype(float)
        for i, c in enumerate(cities):
            add(dummies[:, i], f"city_{c}")
    return np.column_stack(cols), names


@dataclass
class Logit:
    """Ridge logistic regression on standardised features (intercept unpenalised)."""

    lam: float = 1.0
    beta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sd: np.ndarray = field(default_factory=lambda: np.ones(0))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Logit":
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        pen = np.full(Z.shape[1], self.lam)
        pen[0] = 0.0
        b = np.zeros(Z.shape[1])
        for _ in range(100):
            p = 1 / (1 + np.exp(-Z @ b))
            g = Z.T @ (y - p) - pen * b
            H = Z.T @ (Z * (p * (1 - p))[:, None]) + np.diag(pen) + 1e-9 * np.eye(len(b))
            step = np.linalg.solve(H, g)
            b += step
            if np.abs(step).max() < 1e-8:
                break
        self.beta = b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        return 1 / (1 + np.exp(-Z @ self.beta))


def brier_logloss(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    q = np.clip(p, 0.005, 0.995)
    return float(((p - y) ** 2).mean()), float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def expanding_predictions(feats: pl.DataFrame, variant: str, months: list[date], *,
                          lam: float = 1.0) -> pl.DataFrame:
    """Out-of-sample p for each target month: per slot, fit on all undecided station-days from
    WX_START up to three days before the month starts (labels public by then), predict that month."""
    parts = VARIANTS[variant]
    cities = sorted(feats["code"].unique().to_list())
    out = []
    for slot in SLOTS:
        fs = feats.filter((pl.col("slot") == slot) & ~pl.col("decided") & pl.col("wet").is_not_null())
        for m0 in months:
            m1 = (m0 + timedelta(days=32)).replace(day=1)
            # Labels must be public before the month's first decision (d-2 12z of its first day):
            # the CLI for day d is issued the next morning, so train on days before m0 - 2 days.
            tr = fs.filter(pl.col("day") < m0 - timedelta(days=2))
            te = fs.filter((pl.col("day") >= m0) & (pl.col("day") < m1))
            if tr.height < 200 or te.is_empty():
                continue
            Xtr, _ = design(tr, parts, cities)
            model = Logit(lam=lam).fit(Xtr, tr["wet"].to_numpy().astype(float))
            Xte, _ = design(te, parts, cities)
            out.append(te.select("code", "day", "slot", "t", "wet").with_columns(
                pl.Series("p", model.predict(Xte)), pl.lit(variant).alias("variant")))
    return pl.concat(out) if out else pl.DataFrame()


# ---- market evaluation -------------------------------------------------------------------------
WKND_SLOTS = {"fri 12z": (-1, 12), "sat 12z": (0, 12), "sun 12z": (1, 12)}  # offsets from Saturday


def all_windows(codes: list[str], start: date, end: date, slots=SLOTS) -> pl.DataFrame:
    days = pl.DataFrame({"day": pl.date_range(start, end, "1d", eager=True)})
    cd = pl.DataFrame({"code": codes}).join(days, how="cross")
    return pl.concat([windows(cd, slot) for slot in slots])


def label_frame(feats: pl.DataFrame, cli: pl.DataFrame, trace_counts: bool) -> pl.DataFrame:
    thr = TRACE if trace_counts else 0.01
    return (feats.join(cli.select("code", "day", "precip"), on=["code", "day"], how="left")
            .with_columns(pl.when(pl.col("precip").is_null()).then(None)
                          .otherwise(label(pl.col("precip"), trace_counts).cast(pl.Int8)).alias("wet"),
                          (pl.col("seen") >= thr).alias("decided")))


def candle_extremes(s, legs: pl.DataFrame, lookback: dict[str, timedelta]) -> pl.DataFrame:
    """Hourly trade-price high/low per market from the same (cached) candle requests that
    ``fundamental.fetch_candles`` made; used only by the maker check."""
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.1))
    cutoff = api.cutoff().get("market_settled_ts")
    rows = []
    for m in legs.select("ticker", "series", "open_time", "close_time").iter_rows(named=True):
        hi = m["close_time"]
        lb = lookback[m["series"]]
        lo = max(m["open_time"] or hi - lb, hi - lb, hi - timedelta(minutes=60 * 4900))
        try:
            page = api.candles(m["ticker"], lo, hi, 60, bool(cutoff and hi < cutoff))
        except RuntimeError:
            continue
        for c in page.payload.get("candlesticks", []):
            price = c.get("price") or {}
            rows.append({"ticker": m["ticker"], "end_ts": k.parse_ts(c.get("end_period_ts")),
                         "p_low": k.num(k._field(price, "low")), "p_high": k.num(k._field(price, "high"))})
    return pl.DataFrame(rows, schema={"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"),
                                      "p_low": pl.Float64, "p_high": pl.Float64})


def _candle_rows_ext(payload: dict[str, Any], ticker: str) -> list[dict[str, Any]]:
    """``kalshi.candle_rows`` plus the hour's trade-price low/high (for the maker check)."""
    from albert.venues import kalshi as k

    rows = k.candle_rows(payload, ticker, 60)
    for r, c in zip(rows, payload.get("candlesticks", []), strict=True):
        price = c.get("price") or {}
        r["p_low"] = k.num(k._field(price, "low"))
        r["p_high"] = k.num(k._field(price, "high"))
    return rows


def fetch_candles_batch(s, legs: pl.DataFrame, lookback: dict[str, timedelta], *,
                        max_candles: int = 9000) -> pl.DataFrame:
    """Hourly candles for post-cutoff (live-endpoint) markets, one ``/markets/candlesticks``
    request per event (chunked to <= ``max_candles``). Raw responses are cached under
    data/cache/rain/kalshi; rows in data/cache/fundamental_rain_candles_batch.parquet."""
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    path = _table(s, "candles_batch")
    have = pl.read_parquet(path) if path.exists() else None
    done = set(have["ticker"].unique()) if have is not None and have.height else set()
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, _cache_dir(s), min_interval=0.5))
    cutoff = api.cutoff().get("market_settled_ts")
    live = legs.filter(pl.col("close_time") >= cutoff) if cutoff else legs
    rows: list[dict[str, Any]] = []
    for (event,), g in live.group_by("event"):
        todo = [m for m in g.iter_rows(named=True) if m["ticker"] not in done]
        if not todo:
            continue
        lb = lookback[todo[0]["series"]]
        hi = max(m["close_time"] for m in todo)
        lo = min(max(m["open_time"], m["close_time"] - lb) for m in todo)
        per = max(1, int(max_candles // ((hi - lo).total_seconds() / 3600 + 1)))
        ticks = sorted(m["ticker"] for m in todo)
        for j in range(0, len(ticks), per):
            try:
                page = api.candles_batch(ticks[j:j + per], lo, hi, 60)
            except RuntimeError as error:
                print(f"skip batch {event}: {str(error)[:160]}")
                continue
            for mk in page.payload.get("markets") or []:
                rows.extend(_candle_rows_ext({"candlesticks": mk.get("candlesticks") or []}, mk["market_ticker"]))
    new = pl.DataFrame(rows, infer_schema_length=None) if rows else None
    frames = [f for f in (have, new) if f is not None and f.height]
    out = pl.concat(frames, how="diagonal_relaxed").unique(["ticker", "end_ts"]) if frames else pl.DataFrame()
    if out.height:
        _write(out, path)
    return out


def maker_sim(rows: pl.DataFrame, ext: pl.DataFrame, margin: float, *,
              horizon: timedelta = timedelta(hours=6), maker_fee: float = 0.0) -> pl.DataFrame:
    """Rest a YES bid at floor(p - margin) and a YES ask at ceil(p + margin) (never crossing the
    quote at t) for ``horizon`` (or until close). Filled only if a trade prints *through* the
    price in a later hourly candle (at the price if we improved the best quote), held to
    settlement. One contract per side."""
    e = ext.drop_nulls("p_low").sort("ticker", "end_ts")
    j = (rows.select("ticker", "event", "t", "p", "bid", "ask", "y", "close_time")
         .join(e, on="ticker", how="inner")
         .filter((pl.col("end_ts") > pl.col("t")) & (pl.col("end_ts") <= pl.min_horizontal(
             pl.col("t") + horizon, "close_time")))
         .group_by("ticker", "t").agg(pl.col("p_low").min(), pl.col("p_high").max()))
    d = rows.join(j, on=["ticker", "t"], how="left")
    out = []
    for r in d.iter_rows(named=True):
        p, bid, ask, y = r["p"], r["bid"], r["ask"], r["y"]
        b = math.floor((p - margin) * 100 + 1e-9) / 100
        a = math.ceil((p + margin) * 100 - 1e-9) / 100
        lo_, hi_ = r["p_low"], r["p_high"]
        if 0.01 <= b < ask and lo_ is not None and (lo_ < b or (b > bid and lo_ <= b)):
            out.append({"ticker": r["ticker"], "event": r["event"], "t": r["t"], "slot": r.get("slot"),
                        "side": "yes", "price": b, "pnl": y - b - maker_fee * b * (1 - b)})
        if a <= 0.99 and a > bid and hi_ is not None and (hi_ > a or (a < ask and hi_ >= a)):
            out.append({"ticker": r["ticker"], "event": r["event"], "t": r["t"], "slot": r.get("slot"),
                        "side": "no", "price": a, "pnl": a - y - maker_fee * a * (1 - a)})
    return pl.DataFrame(out, schema={"ticker": pl.String, "event": pl.String, "t": pl.Datetime("us", "UTC"),
                                     "slot": pl.String, "side": pl.String, "price": pl.Float64,
                                     "pnl": pl.Float64})


def segment_scores(df: pl.DataFrame, by: str) -> pl.DataFrame:
    """Brier of model vs mid per split and segment (needs p, mid, y, holdout)."""
    out = []
    for (split, seg), g in df.group_by("holdout", by):
        y = g["y"].to_numpy().astype(float)
        bm, lm = brier_logloss(g["p"].to_numpy(), y)
        bk, lk = brier_logloss(g["mid"].to_numpy(), y)
        out.append({"split": "holdout" if split else "dev", by: seg, "n": g.height,
                    "events": g["event"].n_unique(), "mean_y": round(float(y.mean()), 3),
                    "mean_p": round(float(g["p"].mean()), 3), "mean_mid": round(float(g["mid"].mean()), 3),
                    "brier_model": round(bm, 4), "brier_mid": round(bk, 4),
                    "ll_model": round(lm, 4), "ll_mid": round(lk, 4)})
    return pl.DataFrame(out).sort("split", by)


def price_band(mid: pl.Expr) -> pl.Expr:
    return (pl.when(mid < 0.1).then(pl.lit("a <10c")).when(mid < 0.3).then(pl.lit("b 10-30c"))
            .when(mid < 0.7).then(pl.lit("c 30-70c")).when(mid < 0.9).then(pl.lit("d 70-90c"))
            .otherwise(pl.lit("e >=90c")))


def calibration_table(df: pl.DataFrame, col: str) -> pl.DataFrame:
    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0001]
    d = df.with_columns(pl.col(col).cut(bins[1:-1], labels=[f"{a:.2f}-{b:.2f}" for a, b in zip(bins, bins[1:])]
                                        ).alias("bin"))
    return (d.group_by("holdout", "bin").agg(pl.len().alias("n"), pl.col(col).mean().alias("mean_pred"),
                                             pl.col("y").mean().alias("freq"))
            .with_columns(pl.when(pl.col("holdout")).then(pl.lit("holdout")).otherwise(pl.lit("dev")).alias("split"))
            .drop("holdout").sort("split", "bin"))


# ---- driver ------------------------------------------------------------------------------------
def station_day_features(s, codes: list[str], trace_counts: bool) -> pl.DataFrame:
    """Features + labels for every station, day (WX_START+1 .. 2026-10-31) and slot; cached."""
    path = _table(s, f"features_iem_{'T' if trace_counts else '001'}")
    if path.exists():
        return pl.read_parquet(path)
    cli, obs = load(s, "cli"), load(s, "obs")
    clim = climatology(cli, trace_counts)
    parts = []
    for code in codes:  # one station at a time keeps memory well under 1 GB
        mos = pl.scan_parquet(_table(s, "mos")).filter(pl.col("code") == code).collect()
        o = obs.filter(pl.col("code") == code)
        win = all_windows([code], WX_START + timedelta(days=3), date(2026, 10, 31))
        parts.append(build_features(win, o, mos, clim))
    f = label_frame(pl.concat(parts, how="diagonal_relaxed"), cli, trace_counts)
    _write(f, path)
    return f


DEV_MONTHS = tuple(date(2026, m, 1) for m in range(3, 10))


def select_variant(feats: pl.DataFrame, *, slots: tuple[str, ...] = MARKET_SLOTS
                   ) -> tuple[pl.DataFrame, pl.DataFrame, tuple[str, float]]:
    """Round 1's model selection, on station-days only (CLI labels, no market data): expanding
    out-of-sample fits for March-September 2026 for every variant and ridge lambda, scored on the
    market slots; the lowest log loss wins. Returns the overall table, the per-slot table (each
    variant at its best lambda) and (variant, lambda)."""
    sel = pl.concat([expanding_predictions(feats, v, list(DEV_MONTHS), lam=lam).with_columns(pl.lit(lam).alias("lam"))
                     for v in VARIANTS for lam in LAMBDAS])
    sel = sel.filter(pl.col("slot").is_in(list(slots)))
    overall = (sel.group_by("variant", "lam")
               .agg(pl.len().alias("n"), ((pl.col("p") - pl.col("wet")) ** 2).mean().alias("brier"),
                    (-(pl.col("wet") * pl.col("p").clip(0.005, 0.995).log()
                       + (1 - pl.col("wet")) * (1 - pl.col("p").clip(0.005, 0.995)).log())).mean().alias("logloss"),
                    pl.col("p").mean().alias("mean_p"), pl.col("wet").mean().alias("mean_y"))
               .sort("logloss", "variant", "lam"))
    best_lam = overall.unique("variant", keep="first", maintain_order=True)
    vt = variant_table(sel.join(best_lam.select("variant", "lam"), on=["variant", "lam"]))
    return overall, vt, (overall["variant"][0], float(overall["lam"][0]))


def variant_table(preds: pl.DataFrame) -> pl.DataFrame:
    out = []
    for (variant, slot), g in preds.group_by("variant", "slot"):
        y = g["wet"].to_numpy().astype(float)
        b, ll = brier_logloss(g["p"].to_numpy(), y)
        out.append({"variant": variant, "slot": slot, "n": g.height, "brier": round(b, 4),
                    "logloss": round(ll, 4), "mean_p": round(float(g["p"].mean()), 3),
                    "mean_y": round(float(y.mean()), 3)})
    return pl.DataFrame(out).sort("slot", "logloss")


def weekend_rows(legs: pl.DataFrame, preds: pl.DataFrame, feats: pl.DataFrame) -> pl.DataFrame:
    """KXRAINWKND: P(rain Sat or Sun) = 1 - (1 - P_sat)(1 - P_sun) from the daily model at the
    per-day slot matching t (independence across the two days). Days already observed dry at t
    contribute 0; rain already seen at t marks the row decided."""
    pr = preds.select("code", "day", "slot", "p")
    seen = feats.select("code", "day", "slot", "decided")
    rows = []
    for r in legs.iter_rows(named=True):
        sat = r["day"]
        sun = sat + timedelta(days=1)
        for ws, (off, hour) in WKND_SLOTS.items():
            d = sat + timedelta(days=off)
            t = datetime(d.year, d.month, d.day, hour, tzinfo=UTC)
            if ws == "fri 12z":
                parts = [(sat, "d-1 12z"), (sun, "d-2 12z")]
            elif ws == "sat 12z":
                parts = [(sat, "d 12z"), (sun, "d-1 12z")]
            else:
                parts = [(sat, "past"), (sun, "d 12z")]
            q, decided, ok = 1.0, False, True
            for day, slot in parts:
                if slot == "past":
                    # Saturday is over at Sunday 12 UTC: its 18z "rain so far" is a lower bound;
                    # use the whole-day ASOS check from the d 12z row of Sunday's frame instead.
                    hit = seen.filter((pl.col("code") == r["code"]) & (pl.col("day") == day)
                                      & (pl.col("slot") == "sat_full"))
                    decided = decided or (hit.height > 0 and bool(hit["decided"][0]))
                    continue
                hit = seen.filter((pl.col("code") == r["code"]) & (pl.col("day") == day) & (pl.col("slot") == slot))
                if hit.height and bool(hit["decided"][0]):
                    decided = True
                    continue
                pp = pr.filter((pl.col("code") == r["code"]) & (pl.col("day") == day) & (pl.col("slot") == slot))
                if pp.is_empty():
                    ok = False
                    continue
                q *= 1 - float(pp["p"][0])
            rows.append({"ticker": r["ticker"], "event": r["event"], "code": r["code"], "slot": ws, "t": t,
                         "p": None if (not ok or decided) else 1 - q, "decided": decided, "y": r["y"],
                         "close_time": r["close_time"], "open_time": r["open_time"]})
    return pl.DataFrame(rows, schema={"ticker": pl.String, "event": pl.String, "code": pl.String,
                                      "slot": pl.String, "t": pl.Datetime("us", "UTC"), "p": pl.Float64,
                                      "decided": pl.Boolean, "y": pl.Int64,
                                      "close_time": pl.Datetime("us", "UTC"),
                                      "open_time": pl.Datetime("us", "UTC")})


def saturday_full(feats_codes: list[str], days: list[date], obs: pl.DataFrame, trace_counts: bool) -> pl.DataFrame:
    """Whole-Saturday 'rain seen' (all ASOS reports in the LST day) as slot ``sat_full``."""
    rows = []
    for code in feats_codes:
        for day in days:
            lo, hi = lst_bounds(code, day)
            rows.append({"code": code, "day": day, "slot": "sat_full", "t": hi + OBS_LAG + timedelta(hours=1),
                         "lo": lo, "hi": hi})
    w = pl.DataFrame(rows, schema={"code": pl.String, "day": pl.Date, "slot": pl.String,
                                   "t": pl.Datetime("us", "UTC"), "lo": pl.Datetime("us", "UTC"),
                                   "hi": pl.Datetime("us", "UTC")})
    f = rain_so_far(w, obs)
    # Count reports up to the end of the day only.
    return f.with_columns((pl.col("seen") >= (TRACE if trace_counts else 0.01)).alias("decided"))


def run(s=None, *, variant: str | None = None, dev_only: bool = False, log=print) -> Path:
    """Full study. ``dev_only`` drops every October row (debugging before the one holdout run)."""
    from albert.config import settings
    from albert.research import fundamental as fu

    s = s or settings()
    markets = fu.settled_markets(s, list(SERIES))
    legs = market_legs(markets)
    codes = sorted(set(legs["code"].to_list()))
    cli = load(s, "cli")
    tr = trace_rule(markets, cli)
    log(tr)
    trow = tr.filter(pl.col("cli") == "trace")
    trace_counts = bool(trow.height and trow["yes_rate"][0] > 0.5)
    feats = station_day_features(s, codes, trace_counts)
    log(f"features {feats.shape}")
    overall, vt, (chosen, lam) = select_variant(feats)
    log(overall)
    if variant:
        chosen, lam = variant, LAMBDAS[1]
    preds = expanding_predictions(feats, chosen, [date(2026, m, 1) for m in (8, 9, 10)], lam=lam)
    # Station-day scores of the frozen model on October (all stations, not only market legs).
    oct_sd = preds.filter(pl.col("day") >= HOLDOUT_FROM.date())
    if dev_only:
        preds = preds.filter(pl.col("day") < HOLDOUT_FROM.date() - timedelta(days=1))
        oct_sd = oct_sd.clear()
    # ---- KXRAIN rows -------------------------------------------------------------------------
    daily = legs.filter(pl.col("series") == "KXRAIN")
    rows = []
    for slot in MARKET_SLOTS:
        rows.append(daily.with_columns(pl.lit(slot).alias("slot"),
                                       pl.col("day").map_elements(lambda d, sl=slot: slot_time(d, sl),
                                                                  return_dtype=pl.Datetime("us", "UTC")).alias("t")))
    mk = pl.concat(rows).filter((pl.col("t") >= pl.col("open_time") + timedelta(minutes=50))
                                & (pl.col("t") < pl.col("close_time")))
    mk = (mk.join(feats.select("code", "day", "slot", "decided", "wet"), on=["code", "day", "slot"], how="left")
          .join(preds.select("code", "day", "slot", "p"), on=["code", "day", "slot"], how="left"))
    harness = pl.read_parquet(s.data_dir / "cache" / f"fundamental_{NAME}_candles.parquet")
    batch = load(s, "candles_batch")
    candles = pl.concat([harness.select("ticker", "end_ts", "yes_bid_close", "yes_ask_close"),
                         batch.filter(~pl.col("ticker").is_in(harness["ticker"].unique().implode()))
                         .select("ticker", "end_ts", "yes_bid_close", "yes_ask_close")])
    extra_cov = {"markets_with_candles": candles["ticker"].n_unique(), "markets": legs.height}
    q = fu.attach_quotes(mk, candles)
    q = q.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("t") >= HOLDOUT_FROM).alias("holdout"))
    if dev_only:
        q = q.filter(~pl.col("holdout"))
    decided = q.filter(pl.col("decided").fill_null(False))
    live = q.filter(~pl.col("decided").fill_null(False) & pl.col("p").is_not_null())
    label_check = (q.filter(pl.col("wet").is_not_null())
                   .select((pl.col("wet") == pl.col("y")).mean().alias("cli_label_matches_kalshi"), pl.len().alias("n")))
    res = fu.evaluate(live.select("ticker", "event", "t", "p", "y", "bid", "ask", "slot", "code", "holdout",
                                  "close_time"), HOLDOUT_FROM)
    extra: dict[str, Any] = {"variant": chosen, "lambda": lam, "trace_counts_as_rain": trace_counts,
                             "rows_decided_excluded": decided.height,
                             "decided_mean_mid": None if decided.is_empty() else round(float(decided["mid"].mean()), 3),
                             "decided_yes_rate": None if decided.is_empty() else round(float(decided["y"].mean()), 4),
                             **extra_cov}
    # ---- per-slot evaluation (lead time) ------------------------------------------------------
    per_slot = []
    for slot in MARKET_SLOTS:
        g = live.filter(pl.col("slot") == slot)
        if g.height < 50:
            continue
        r = fu.evaluate(g.select("ticker", "event", "t", "p", "y", "bid", "ask"), HOLDOUT_FROM)
        b = r.get("blend", {})
        th = r.get("theta_chosen_on_dev")
        tk = {x["split"]: x for x in r["taker"] if x["theta"] == th} if th is not None else {}
        for sc in r["scores"]:
            per_slot.append({"slot": slot, **{k: sc[k] for k in ("split", "n", "brier_model", "brier_mid")},
                             "c_model": b.get("c_model"), "t_c": b.get("t_c"), "theta": th,
                             "taker_trades": tk.get(sc["split"], {}).get("trades"),
                             "taker_pnl_c": tk.get(sc["split"], {}).get("pnl_c"),
                             "taker_t": tk.get(sc["split"], {}).get("t")})
    # ---- maker check --------------------------------------------------------------------------
    old_legs = legs.filter(pl.col("ticker").is_in(harness["ticker"].unique().implode()))
    ext = pl.concat([candle_extremes(s, old_legs, {"KXRAIN": timedelta(days=3), "KXRAINWKND": timedelta(days=8)}),
                     batch.select("ticker", "end_ts", "p_low", "p_high")]).unique(["ticker", "end_ts"])
    mrows = []
    best = None
    for margin in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
        for split in (False, True):
            fills = maker_sim(live.filter(pl.col("holdout") == split), ext, margin)
            if fills.is_empty():
                mrows.append({"margin": margin, "split": "holdout" if split else "dev", "fills": 0})
                continue
            mu, tt, k = fu.clustered_mean(fills["pnl"].to_numpy(), fills["event"].to_numpy())
            mrows.append({"margin": margin, "split": "holdout" if split else "dev", "fills": fills.height,
                          "events": k, "pnl_c": round(mu * 100, 3), "t": None if tt is None else round(tt, 2),
                          "yes_share": round(float((fills["side"] == "yes").mean()), 3)})
            if not split and fills.height >= 30 and (best is None or mu > best[1]):
                best = (margin, mu)
    extra["maker_margin_chosen_on_dev"] = None if best is None else best[0]
    # ---- weekend --------------------------------------------------------------------------------
    wk_legs = legs.filter(pl.col("series") == "KXRAINWKND")
    wk_res = None
    if wk_legs.height:
        sats = sorted(set(wk_legs["day"].to_list()))
        satf = saturday_full(codes, sats, load(s, "obs"), trace_counts)
        seen = pl.concat([feats.select("code", "day", "slot", "decided"),
                          satf.select("code", "day", "slot", "decided")])
        wk = weekend_rows(wk_legs, preds, seen)
        wk = wk.filter((pl.col("t") >= pl.col("open_time") + timedelta(minutes=50)) & (pl.col("t") < pl.col("close_time")))
        wq = fu.attach_quotes(wk, candles)
        wlive = wq.filter(~pl.col("decided") & pl.col("p").is_not_null())
        if dev_only:
            wlive = wlive.filter(pl.col("t") < HOLDOUT_FROM)
        extra["wknd_rows"] = wlive.height
        if wlive.height >= 40:
            wk_res = fu.evaluate(wlive.select("ticker", "event", "t", "p", "y", "bid", "ask"), HOLDOUT_FROM)
    # ---- report -------------------------------------------------------------------------------
    from albert.reports import Report  # noqa: F401  (fu.report uses it)

    live_seg = live.with_columns(price_band(pl.col("mid")).alias("band"))
    path = fu.report(s, NAME + ("-dev" if dev_only else ""), res, extra)
    with open(path, "a") as fh:
        from albert.reports import md_table

        def sec(title: str, frame: pl.DataFrame, n: int = 80) -> None:
            fh.write(f"\n## {title}\n\n{md_table(frame, n)}\n")

        sec("Settlement rule: Kalshi YES rate by CLI precipitation class", tr)
        sec("CLI label vs Kalshi result (rows with a CLI value)", label_check)
        sec("Variant selection on dev (expanding OOS fits, station-days Mar-Sep 2026, market slots)", overall)
        sec("Variant selection by slot (each variant at its best lambda)", vt)
        if oct_sd.height:
            sec("Frozen model on October station-days (all stations, market slots)",
                variant_table(oct_sd.filter(pl.col("slot").is_in(list(MARKET_SLOTS)))))
        sec("Per decision slot (lead time): Brier, blend c, taker at dev-chosen theta", pl.DataFrame(per_slot))
        sec("By price band of the mid", segment_scores(live_seg, "band"))
        blend = res.get("blend")
        if blend:
            # Trade on the dev-fitted blend probability (market + model) instead of the model alone.
            lp = blend["a"] + blend["b_mid"] * fu.logit(live_seg["mid"].to_numpy()) + blend["c_model"] * fu.logit(
                live_seg["p"].to_numpy())
            bl = live_seg.with_columns(pl.Series("p", 1 / (1 + np.exp(-lp))))
            brows, bbest = [], None
            for theta in fu.THETAS:
                for h in (False, True):
                    tr = fu.taker_pnl(bl.filter(pl.col("holdout") == h).select("ticker", "event", "t", "p", "y", "bid", "ask"), theta)
                    if not tr.height:
                        brows.append({"theta": theta, "split": "holdout" if h else "dev", "trades": 0})
                        continue
                    mu, tt, k = fu.clustered_mean(tr["pnl"].to_numpy(), tr["event"].to_numpy())
                    brows.append({"theta": theta, "split": "holdout" if h else "dev", "trades": tr.height, "events": k,
                                  "pnl_c": round(mu * 100, 3), "t": None if tt is None else round(tt, 2)})
                    if not h and tr.height >= 30 and (bbest is None or mu > bbest[1]):
                        bbest = (theta, mu)
            sec(f"Taker on the dev-fitted blend probability (dev theta choice: {None if bbest is None else bbest[0]}; "
                "blend coefficients are in-sample on dev)", pl.DataFrame(brows, infer_schema_length=None))
        th = res.get("theta_chosen_on_dev")
        if th is not None:
            tk = fu.taker_pnl(live_seg.select("ticker", "event", "t", "p", "y", "bid", "ask", "slot", "band",
                                              "holdout", "code"), th)
            if tk.height:
                seg = []
                for (h, slot, band), g in tk.group_by("holdout", "slot", "band"):
                    mu, tt, k = fu.clustered_mean(g["pnl"].to_numpy(), g["event"].to_numpy())
                    seg.append({"split": "holdout" if h else "dev", "slot": slot, "band": band,
                                "trades": g.height, "events": k, "pnl_c": round(mu * 100, 2),
                                "t": None if tt is None else round(tt, 2),
                                "yes_share": round(float((g["side"] == "yes").mean()), 2)})
                sec(f"Taker at dev-chosen theta={th} by slot and price band", pl.DataFrame(seg).sort("split", "slot", "band"))
                conc = []
                for h in (False, True):
                    g = tk.filter(pl.col("holdout") == h)
                    if not g.height:
                        continue
                    by_ev = g.group_by("event").agg(pl.col("pnl").sum()).sort("pnl", descending=True)
                    top3 = by_ev.head(3)["event"].to_list()
                    rest = g.filter(~pl.col("event").is_in(top3))
                    mu, tt, k = fu.clustered_mean(rest["pnl"].to_numpy(), rest["event"].to_numpy())
                    conc.append({"split": "holdout" if h else "dev", "trades": g.height, "total_$": round(float(g["pnl"].sum()), 2),
                                 "top3_events": ", ".join(top3), "top3_$": round(float(by_ev.head(3)["pnl"].sum()), 2),
                                 "ex_top3_pnl_c": round(mu * 100, 2), "ex_top3_t": None if tt is None else round(tt, 2),
                                 "positive_events_share": round(float((by_ev["pnl"] > 0).mean()), 2)})
                sec("Taker concentration (one contract per trade)", pl.DataFrame(conc))
                cty = []
                for (h, code), g in tk.group_by("holdout", "code" if "code" in tk.columns else "ticker"):
                    cty.append({"split": "holdout" if h else "dev", "code": code, "trades": g.height,
                                "pnl_c": round(float(g["pnl"].mean()) * 100, 2)})
                sec("Taker by city", pl.DataFrame(cty).sort("split", "code"), 90)
        sec("By city", segment_scores(live_seg, "code"), 90)
        sec("Calibration: model", calibration_table(live_seg, "p"))
        sec("Calibration: market mid", calibration_table(live_seg, "mid"))
        sec("Maker check: rest at floor(p-m) / ceil(p+m) for 6 h, trade-through fills, held to settlement "
            "(cents per contract, no maker fee on this series)", pl.DataFrame(mrows, infer_schema_length=None))
        if wk_res:
            sec("KXRAINWKND scores", pl.DataFrame(wk_res["scores"]))
            fh.write(f"\nKXRAINWKND blend: {wk_res.get('blend')}\n")
            sec("KXRAINWKND taker", pl.DataFrame(wk_res["taker"], infer_schema_length=None))
    live.write_parquet(path.parent / "rows.parquet")
    log(f"report: {path}")
    return path


# ---- IEM-only rebuild (2026-11-06): the harness at the morning decision, by quoted spread --------
IEM_SLOT = "d 12z"
SPREADS = ("<=3c", "4-6c", ">6c")


def spread3(bid: pl.Expr, ask: pl.Expr) -> pl.Expr:
    """Quoted spread at t in whole cents: <= 3c, 4-6c, > 6c."""
    sp = ((ask - bid) * 100).round(0)
    return pl.when(sp <= 3).then(pl.lit(SPREADS[0])).when(sp <= 6).then(pl.lit(SPREADS[1])).otherwise(pl.lit(SPREADS[2]))


def morning_trades(s, rows: pl.DataFrame, *, log=print) -> pl.DataFrame:
    """Every public print of each row's market in [t, end]: complete trade-page chains that round 2
    cached for whole market lives (no requests) plus ``weather_maker.fetch_trades`` (2 threads,
    raw pages under data/cache/weather_maker/kalshi) for the other windows."""
    from albert.research import weather_maker as wm

    have = wm.cached_trades(s, "KXRAIN")
    todo = rows.filter(~pl.col("ticker").is_in(have["ticker"].unique().implode()))
    log(f"trades: {rows['ticker'].n_unique() - todo['ticker'].n_unique()} markets cached, {todo.height} windows to fetch")
    new = wm.fetch_trades(s, todo.select("ticker", pl.col("t").alias("t_from"), pl.col("end").alias("t_to")),
                          "rain_iem_d12z", workers=2, log=log)
    return pl.concat([have, new.select(have.columns)]).unique("trade_id")


def run_iem(s=None, *, log=print) -> Path:
    """IEM-only rebuild of the rain model (Open-Meteo removed): dev variant selection on
    station-days, the expanding monthly fits, round 2's market rows and fair values
    (``weather_maker.prepare_rain``: dev out-of-fold blend per slot, mid-only recalibration), then
    the harness on the morning decision (d 12z, quote until rain is reported / the CLI day ends /
    close): ``fundamental.evaluate`` and ``fundamental.evaluate_maker`` (exact trades,
    trade-through fills, margin chosen on dev) overall and by quoted spread (<= 3c, 4-6c, > 6c),
    with the plain mid and the mid-only recalibration as maker controls. Also round 2's blend test
    by spread for every slot and its candle-fill requote check (d 00z -> d 12z, 8c) for
    comparison with the Open-Meteo numbers. Writes a report; returns its path."""
    from albert.config import settings
    from albert.reports import Report
    from albert.research import fundamental as fu
    from albert.research import weather_maker as wm

    s = s or settings()
    legs = market_legs(fu.settled_markets(s, list(SERIES)))
    feats = station_day_features(s, sorted(set(legs["code"].to_list())), False)
    overall, vt, (variant, lam) = select_variant(feats)
    log(f"dev choice: {variant} lambda {lam}")
    q, meta = wm.prepare_rain(s, variant=variant, lam=lam, log=log)
    q = q.with_columns(spread3(pl.col("bid"), pl.col("ask")).alias("spread"))
    daily = q.filter(pl.col("series") == "KXRAIN")
    morning = daily.filter(pl.col("slot") == IEM_SLOT)
    # information: round 2's blend test (dev fit, October refit) per spread bucket and slot
    info = pl.concat([wm.blend_by(daily, "spread").with_columns(pl.lit("all slots").alias("slot"))]
                     + [wm.blend_by(daily.filter(pl.col("slot") == sl), "spread").with_columns(pl.lit(sl).alias("slot"))
                        for sl in MARKET_SLOTS], how="diagonal_relaxed")
    mix = daily.group_by("holdout", "slot", "spread").agg(pl.len().alias("rows")).sort("holdout", "slot", "spread")
    # harness: evaluate (scores, dev blend, taker) on the morning rows, overall and per bucket
    cols = ["ticker", "event", "t", "p", "y", "bid", "ask", "fee_multiplier"]
    parts = {"all": morning, **{b: morning.filter(pl.col("spread") == b) for b in SPREADS}}
    ev = {b: fu.evaluate(g.select(cols), HOLDOUT_FROM) for b, g in parts.items() if g.height}
    ev_scores = pl.concat([pl.DataFrame(r["scores"]).with_columns(pl.lit(b).alias("spread")) for b, r in ev.items()],
                          how="diagonal_relaxed")
    ev_blend = pl.DataFrame([{"spread": b, **r.get("blend", {})} for b, r in ev.items()], infer_schema_length=None)
    ev_taker = pl.concat([pl.DataFrame(r["taker"], infer_schema_length=None).with_columns(
        pl.lit(b).alias("spread"), pl.lit(r["theta_chosen_on_dev"]).alias("theta_dev")) for b, r in ev.items()],
        how="diagonal_relaxed")
    # harness maker on exact trades: the blend (forward rule's fair), the model alone, and controls
    trades = morning_trades(s, morning, log=log)
    mk, chosen = [], []
    for fair in ("f_blend", "f_mid", "f_midrecal", "p"):
        for b, g in parts.items():
            g = g.filter(pl.col(fair).is_not_null() & pl.col(fair).is_not_nan())
            if not g.height:
                continue
            r = fu.evaluate_maker(g.select("ticker", "event", "t", "end", pl.col(fair).alias("p"), "bid", "ask", "y",
                                           "fee_type", "fee_multiplier"), trades, HOLDOUT_FROM)
            m0 = r["margin_chosen_on_dev"]
            for row in r["maker"]:
                mk.append({"fair": fair, "spread": b, **row})
                if row["margin"] in (m0, 0.08):
                    chosen.append({"fair": fair, "spread": b, "which": "dev choice" if row["margin"] == m0 else "8c rule",
                                   **row})
            log(f"  maker {fair} {b} done")
    mk_df = pl.DataFrame(mk, infer_schema_length=None)
    ch_df = pl.DataFrame(chosen, infer_schema_length=None).sort("which", "spread", "fair", "split")
    # round 2's candle-fill requote check (d 00z -> d 12z, 8c, pessimistic) by spread, for comparison
    rq = wm.rain_requote_end(q, ("d 00z", "d 12z")).select("ticker", "t", pl.col("end_rq").alias("end_rq2"))
    two = daily.filter(pl.col("slot").is_in(["d 00z", IEM_SLOT])).join(rq, on=["ticker", "t"], how="left")
    cp = wm._shift_pessimistic(wm.rain_candle_prints(s, q))
    weeks = wm.split_weeks(daily)
    # (weather_maker.orders keeps a fixed column set: the 3-way bucket travels as ``sbucket``)
    two, m3 = (x.with_columns(pl.col("spread").alias("sbucket")) for x in (two, morning))
    cand = pl.concat([wm.segments(two, cp, pl.col("end_rq2"), f, 0.08, weeks, ["sbucket"], replace=True)
                      .with_columns(pl.lit(f).alias("fair"), pl.lit("requote d 00z -> d 12z").alias("test"))
                      for f in ("f_blend", "f_mid")]
                     + [wm.segments(m3, cp, pl.col("end"), f, 0.08, weeks, ["sbucket"])
                        .with_columns(pl.lit(f).alias("fair"), pl.lit("d 12z -> observable").alias("test"))
                        for f in ("f_blend", "f_mid")], how="diagonal_relaxed").rename({"sbucket": "spread"})
    rep = Report(s.reports_dir, f"research-fundamental-{NAME}-iem")
    rep.json("Run", {"variant": variant, "lambda": lam, "inputs": "IEM only: CLI, ASOS p01i, NBS + GFS MOS",
                     "morning_rows": morning.height, "trade_markets": trades["ticker"].n_unique(),
                     "weeks": weeks, **meta})
    rep.table("Variant selection on dev (expanding OOS fits, station-days Mar-Sep 2026, market slots)", overall)
    rep.table("Variant selection by slot (each variant at its best lambda)", vt)
    rep.table("Blend logit P = a + b·logit(mid) + c·logit(model) by quoted spread at t and slot (dev fit; "
              "October refit)", info, max_rows=60)
    rep.table("Rows by split, slot and spread", mix, max_rows=40)
    rep.table(f"Harness evaluate, {IEM_SLOT}: scores by spread", ev_scores)
    rep.table(f"Harness evaluate, {IEM_SLOT}: dev blend and October refit by spread", ev_blend)
    rep.table(f"Harness evaluate, {IEM_SLOT}: taker by spread (theta chosen on dev per bucket)", ev_taker, max_rows=80)
    rep.table(f"Harness evaluate_maker, {IEM_SLOT} -> observable (exact trades, trade-through): dev-chosen margin "
              "and the forward rule's 8c", ch_df, max_rows=80)
    rep.table("Round 2's candle check (pessimistic hourly fills, 8c) by spread", cand, max_rows=60)
    rep.table(f"Harness evaluate_maker, {IEM_SLOT}: full margin grid", mk_df, save_as="maker_grid", max_rows=200)
    path = rep.save()
    morning.write_parquet(path.parent / "morning_rows.parquet")
    log(f"report: {path}")
    return path
