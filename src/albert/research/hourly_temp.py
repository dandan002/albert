"""Physics / statistical nowcast of Kalshi hourly temperature ladders (``KXTEMP*``).

**What settles them.** ``KXTEMPMIAH``, ``KXTEMPNYCHS``, ``KXTEMPCHIHS`` and ``KXTEMPLAXHS`` ("will
the temperature in <city> at <hour> ET be above X.99") settle on the *Kalshi Weather Index* (contract
terms MIAWINDEX / NYCWINDEX / CHIMETROWINDEX / LAWINDEX, Appendix B): an equal-weight mean of the
one-minute HF-ASOS readings of 5 airports (8 for NYC), delivered by Synoptic Data in whole deg C,
offset-adjusted only when a member is missing, published in deg F to 0.01 F for every UTC minute
5 minutes after the minute. The settlement value is the index at the settlement minute S (the
top of the hour); Kalshi's ``expiration_value`` equals the published index at S for 3,760 / 3,760
events. With all members reporting, the index lives on the lattice 32 + 1.8 j / n F, so "above
X.99" is a comparison with a lattice midpoint (:func:`threshold`). (The older ``KXTEMP*H`` series,
Aug 15 - Sep 10, settled on The Weather Company's reading of one station's hourly METAR.)

**What is knowable when.** Each market lists 60 minutes before S, so decision times are
S - 55/40/25/10 min (leads of 1-6 h do not exist). The public endpoint ``GET /live_data/weather/{city}``
(no auth) serves the canonical index and, with ``detailed=true``, every member reading as soon as
Kalshi receives it (144 s median, <= 240 s for all members in 100% of minutes), so the model uses
index minutes u <= t - 4 (:data:`DELAY_MIN`). NBM text guidance (4 runs/day, IEM archive) is used
from run time + 2 h. The minute archive starts 2026-09-21 22:22 UTC: dev = Sep 22-30, holdout =
October 2026.

**Model** (damped anomaly / Ornstein-Uhlenbeck, per city and lead, :data:`VARIANTS`): the change
I(S) - I0 from the newest known index value is regressed on the climatological diurnal drift over
(u0, S] (a 3-harmonic Fourier slope in local time, fitted on an IEM one-minute proxy of the index
for Sep-Oct 2025 and Aug 2026 plus the dev index), the recent changes over 15/30/60 min, the NBM
forecast change over (u0, S] and the anomaly NBM(u0) - I0; log sigma on log recent volatility and
|drift|; the predictive CDF is the empirical distribution of standardised residuals. Variants
(persistence, drift only, + trend, + NBM, + OU anomaly, + HRRR 15-minute) were compared by
leave-one-day-out CV on dev minutes; OU won in every city and lead, HRRR added nothing.

**Result** (``run(s, holdout=True)``, report ``research-fundamental-hourly_temp-*``): the market mid
beats the nowcast at every lead and city (holdout Brier 0.123 vs 0.088); the model adds a little
information only at lead 55 (holdout refit c 0.07, t 2.8) and none is tradable: taker -3.4 to
-4.8c per contract, ladder optimisation -2.3c, blend taker +1.9c (t 1.0) which the market-only
recalibration reproduces. The market's implied 1-hour centre is more accurate than the model's.
"""

import csv
import io
import math
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from albert.http import CachedClient

NAME = "hourly_temp"
IEM_BASE = "https://mesonet.agron.iastate.edu"

# Kalshi Weather Index configurations (contract terms MIAWINDEX / NYCWINDEX / CHIMETROWINDEX /
# LAWINDEX, Appendix B): equal-weight member stations and their initial offsets b_i (deg C).
INDEX = {
    "KXTEMPMIAH": {"MIA": 0.0, "OPF": -0.5, "FLL": 0.0, "FXE": 0.0, "PMP": 0.0},
    "KXTEMPNYCHS": {"LGA": 1.0, "EWR": 1.0, "TEB": 1.0, "CDW": 0.0, "HPN": -1.0, "FRG": 0.0,
                    "SMQ": -1.0, "ISP": 0.0},
    "KXTEMPCHIHS": {"MDW": 1.0, "ORD": 0.5, "PWK": 0.0, "DPA": -0.5, "ARR": -1.0},
    "KXTEMPLAXHS": {"HHR": 0.0, "LAX": -1.0, "SMO": -1.0, "LGB": 0.5, "SNA": 1.0},
}
CITY = {"KXTEMPMIAH": "MIA", "KXTEMPNYCHS": "NYC", "KXTEMPCHIHS": "CHI", "KXTEMPLAXHS": "LAX"}
# Kalshi live-data city ids (GET /live_data/weather/{city}, public, no auth).
INDEX_CITY = {"MIA": "miami", "NYC": "nyc", "CHI": "chicago", "LAX": "la-coastal"}
# Information delay. The canonical point for minute u is published at u + 5 min (D = 300 s), but the
# same public endpoint already serves the member stations' readings as "pending" once Kalshi receives
# them: receipt is 144 s after the minute at the median and <= 240 s for every member in 100% of
# minutes (detailed history, 2026-09-25, all four cities). A public poller therefore knows minute
# u by u + 4 min.
PUBLISH_DELAY = timedelta(minutes=4)
DELAY_MIN = 4                          # module default; diagnostics may change it
TZ = {"MIA": "America/New_York", "NYC": "America/New_York", "CHI": "America/Chicago",
      "LAX": "America/Los_Angeles"}

# METAR temperature: remarks T-group (0.1 C) and the body group "25/23" or "M02/M05" (whole C).
_TGRP = re.compile(r"\bT([01])(\d{3})([01])(\d{3})\b")
_BODY = re.compile(r"\s(M?\d{2})/(M?\d{2})?\s")


def cache_dir(s) -> Path:
    return s.data_dir / "cache" / NAME


def iem_client(s) -> CachedClient:
    return CachedClient("iem", IEM_BASE, cache_dir(s), min_interval=1.0)


def fetch_iem_hf(client: CachedClient, station: str, start: date, end: date) -> str:
    """IEM ASOS archive, every report type (5-minute HFMETAR, routine, special) with the raw METAR
    text, [start, end) in UTC. Cached per (station, window)."""
    return client.get("/cgi-bin/request/asos.py", {
        "station": station, "data": ["metar", "tmpf", "dwpf", "sknt", "skyc1"], "tz": "Etc/UTC",
        "format": "onlycomma", "latlon": "no", "missing": "empty", "report_type": [1, 3, 4],
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day}).payload


def metar_temp_c(metar: str) -> tuple[float | None, bool]:
    """(temperature in C, precise?) from a METAR: the T-group if present (0.1 C), else the body."""
    m = _TGRP.search(metar or "")
    if m:
        v = int(m.group(2)) / 10
        return (-v if m.group(1) == "1" else v), True
    m = _BODY.search(" " + (metar or "") + " ")
    if m:
        g = m.group(1)
        return (-int(g[1:]) if g.startswith("M") else int(g)), False
    return None, False


def obs_rows(text: str, station: str) -> list[dict[str, Any]]:
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        try:
            ts = datetime.fromisoformat(r["valid"].strip()).replace(tzinfo=UTC)
        except (KeyError, ValueError):
            continue
        metar = r.get("metar") or ""
        c, precise = metar_temp_c(metar)
        if c is None:
            continue
        hf = "MADISHF" in metar
        rows.append({"station": station, "ts": ts, "tc": c, "tf": c * 9 / 5 + 32,
                     "precise": precise and not hf, "hf": hf,
                     "dwpf": _float(r.get("dwpf")), "sknt": _float(r.get("sknt")),
                     "sky": (r.get("skyc1") or "").strip() or None})
    return rows


def _float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_obs(s, stations: list[str], start: date, end: date) -> pl.DataFrame:
    """5-minute + routine + special observations for ``stations`` in [start, end), month chunks."""
    cl = iem_client(s)
    rows: list[dict[str, Any]] = []
    for st in stations:
        a = start
        while a < end:
            b = min(end, (a.replace(day=1) + timedelta(days=32)).replace(day=1))
            rows.extend(obs_rows(fetch_iem_hf(cl, st, a, b), st))
            a = b
    return (pl.DataFrame(rows, infer_schema_length=None)
            .unique(["station", "ts"], keep="last").sort("station", "ts"))


def fetch_iem_1min(client: CachedClient, station: str, start: date, end: date) -> str:
    """IEM archive of the NCEI one-minute ASOS record (whole degrees F), [start, end) UTC."""
    return client.get("/cgi-bin/request/asos1min.py", {
        "station": station, "vars": "tmpf", "sts": f"{start.isoformat()}T00:00Z",
        "ets": f"{end.isoformat()}T00:00Z", "sample": "1min", "what": "download",
        "delim": "comma", "tz": "UTC"}).payload


def one_min_rows(text: str, station: str) -> list[dict[str, Any]]:
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        v = _float(r.get("tmpf"))
        try:
            ts = datetime.fromisoformat((r.get("valid(UTC)") or "").strip()).replace(tzinfo=UTC)
        except ValueError:
            continue
        if v is not None:
            rows.append({"station": station, "ts": ts, "tf": v})
    return rows


def load_1min(s, stations: list[str], start: date, end: date) -> pl.DataFrame:
    cl = iem_client(s)
    frames = []
    for st in stations:
        a = start
        while a < end:
            b = min(end, (a.replace(day=1) + timedelta(days=32)).replace(day=1))
            rows = one_min_rows(fetch_iem_1min(cl, st, a, b), st)
            if rows:
                frames.append(pl.DataFrame(rows, schema={"station": pl.String, "ts": pl.Datetime("us", "UTC"),
                                                         "tf": pl.Float64}))
            a = b
    return pl.concat(frames).unique(["station", "ts"], keep="last").sort("station", "ts")


# ---- Kalshi Weather Index (minute series) and markets --------------------------------------------
def kalshi_client(s) -> CachedClient:
    return CachedClient("kalshi", s.kalshi_base, cache_dir(s), min_interval=0.5)


def fetch_index_day(client: CachedClient, city: str, day: date, detailed: bool = False) -> dict[str, Any]:
    """One UTC day of the published minute index (``GET /live_data/weather/{city}``)."""
    a = datetime(day.year, day.month, day.day, tzinfo=UTC)
    params = {"from": int(a.timestamp() * 1000), "to": int((a + timedelta(days=1)).timestamp() * 1000) - 1}
    if detailed:
        params["detailed"] = "true"
    return client.get(f"/live_data/weather/{city}", params).payload


def index_rows(payload: dict[str, Any], city: str) -> list[dict[str, Any]]:
    rows = []
    for x in payload.get("timeseries") or []:
        v = x.get("v")
        rows.append({"city": city, "t": datetime.fromtimestamp(x["t"] / 1000, UTC),
                     "v": None if v is None else float(v), "status": x.get("status"),
                     "contributors": x.get("contributors")})
    return rows


INDEX_SCHEMA = {"city": pl.String, "t": pl.Datetime("us", "UTC"), "v": pl.Float64, "status": pl.String,
                "contributors": pl.Int64}


def load_index(s, start: date, end: date, cities: list[str] | None = None) -> pl.DataFrame:
    """Minute index for each city in [start, end); days before the archive starts come back empty."""
    cl = kalshi_client(s)
    rows: list[dict[str, Any]] = []
    for c in cities or list(INDEX_CITY):
        d = start
        while d < end:
            rows.extend(index_rows(fetch_index_day(cl, INDEX_CITY[c], d), c))
            d += timedelta(days=1)
    return pl.DataFrame(rows, schema=INDEX_SCHEMA).unique(["city", "t"], keep="last").sort("city", "t")


def fetch_market_list(s, series: list[str], start: datetime, end: datetime) -> pl.DataFrame:
    """Settled markets of the hourly series with Kalshi's ``expiration_value`` (the index value
    that settled the event), from ``GET /markets?series_ticker=...&status=settled``."""
    from albert.venues import kalshi as k

    api = k.Kalshi(kalshi_client(s))
    rows = []
    for ser in series:
        for page in api.series_markets(ser, "settled", min_close=start, max_close=end, cache=True):
            for m in page.payload.get("markets", []):
                rows.append({"ticker": m["ticker"], "event": m["event_ticker"], "series": ser,
                             "exp_value": _float(m.get("expiration_value")), "result": m.get("result"),
                             "floor_strike": _float(m.get("floor_strike")), "strike_type": m.get("strike_type"),
                             "open_time": k.parse_ts(m.get("open_time")), "close_time": k.parse_ts(m.get("close_time"))})
    return pl.DataFrame(rows, infer_schema_length=None)


def _patient(call: Any, tries: int = 8, wait: float = 5.0) -> Any:
    """Retry a cached GET through shared-rate-limit 429s (the client gives up after 5)."""
    import time

    from albert.http import HttpError

    for i in range(tries):
        try:
            return call()
        except HttpError as error:
            if "429" not in str(error) or i == tries - 1:
                raise
            time.sleep(wait * (i + 1))
    raise RuntimeError("unreachable")


def fetch_minute_candles(s, markets: pl.DataFrame, *, hours_per_call: int = 2) -> pl.DataFrame:
    """1-minute candles for every market over its life (``open_time`` .. ``close_time``), via the
    batch endpoint (at most 10,000 requested candles per call), grouped by close hour. Cached."""
    from albert.venues import kalshi as k

    api = k.Kalshi(kalshi_client(s))
    m = markets.select("ticker", "close_time").sort("close_time")
    closes = sorted(set(m["close_time"].to_list()))
    rows: list[dict[str, Any]] = []
    for i in range(0, len(closes), hours_per_call):
        grp = closes[i:i + hours_per_call]
        tk = m.filter(pl.col("close_time").is_in(grp))["ticker"].to_list()
        lo, hi = min(grp) - timedelta(minutes=61), max(grp)
        per = max(1, 10_000 // max(1, int((hi - lo).total_seconds() // 60)))
        for j in range(0, len(tk), per):
            page = _patient(lambda: api.candles_batch(tk[j:j + per], lo, hi, 1))
            for mk in page.payload.get("markets", []):
                rows.extend(k.candle_rows(mk, mk["market_ticker"], 1))
        if (i // hours_per_call) % 25 == 0:
            print(f"  candles {i}/{len(closes)} hours {datetime.now(UTC):%H:%M:%S}", flush=True)
    return pl.DataFrame(rows, schema=k.CANDLE_SCHEMA) if rows else pl.DataFrame(schema=k.CANDLE_SCHEMA)


def strike(ticker: str) -> float:
    """``KXTEMPMIAH-26SEP2308-T76.99`` -> 77.0 (YES iff the reading is above 76.99)."""
    return math.floor(float(ticker.rsplit("-T", 1)[1])) + 1.0


# ---- events, lattice, minute grid -----------------------------------------------------------------
def events(markets: pl.DataFrame) -> pl.DataFrame:
    """One row per (market): series, city, event, settlement minute ``S`` (= close time), the
    strike ``k`` (YES iff index > k - 0.01), result ``y`` and the settled index ``exp_value``."""
    m = markets.filter(pl.col("result").is_in(["yes", "no"]) & pl.col("series").is_in(list(INDEX)))
    return m.select(
        "ticker", "event", "series", pl.col("series").replace_strict(CITY).alias("city"),
        pl.col("close_time").alias("S"), "exp_value",
        pl.col("ticker").map_elements(strike, return_dtype=pl.Float64).alias("k"),
        (pl.col("result") == "yes").cast(pl.Int8).alias("y"))


def members(city: str) -> int:
    return len(INDEX[next(s for s, c in CITY.items() if c == city)])


def threshold(k: float, n: int) -> float:
    """With all ``n`` members reporting whole degrees C, the index lives on the lattice
    32 + 1.8 j / n (deg F). YES (index > k - 0.01) <=> index >= the first lattice point above
    k - 0.01; return the midpoint between that point and the one below it, so a continuous
    latent value can be compared against it."""
    step = 1.8 / n
    j = math.ceil(((k - 0.01) - 32) / step + 1e-9)
    hi = 32 + j * step
    return hi - step / 2


def minute_grid(index: pl.DataFrame) -> dict[str, tuple[Any, Any]]:
    """city -> (minute numbers since epoch, values with NaN where no canonical point)."""
    import numpy as np

    out = {}
    for c in index["city"].unique().to_list():
        g = index.filter((pl.col("city") == c) & pl.col("v").is_not_null()).sort("t")
        mins = (g["t"].dt.epoch("s").to_numpy() // 60).astype(np.int64)
        lo, hi = mins.min(), mins.max()
        v = np.full(hi - lo + 1, np.nan)
        v[mins - lo] = g["v"].to_numpy()
        out[c] = (lo, v)
    return out


def proxy_index(one_min: pl.DataFrame, city: str, *, round_c: bool = True, min_members: int | None = None
                ) -> pl.DataFrame:
    """Index proxy from the IEM/NCEI one-minute ASOS record (whole deg F per station), following
    the methodology: x_i in deg C (optionally rounded to whole C, as Synoptic delivers it),
    z_i = x_i - b_i + B_c, equal-weight mean over the members reporting that minute."""
    series = next(s for s, c in CITY.items() if c == city)
    mem = INDEX[series]
    bc = sum(mem.values()) / len(mem)
    need = min_members or max(1, len(mem) - 1)
    c = (pl.col("tf") - 32) * 5 / 9
    w = one_min.filter(pl.col("station").is_in(list(mem))).with_columns((c.round(0) if round_c else c).alias("c"))
    w = w.with_columns((pl.col("c") - pl.col("station").replace_strict(mem, return_dtype=pl.Float64) + bc).alias("z"))
    g = w.group_by("ts").agg(pl.col("z").mean().alias("zc"), pl.len().alias("k")).filter(pl.col("k") >= need)
    return g.select(pl.lit(city).alias("city"), pl.col("ts").alias("t"), (pl.col("zc") * 9 / 5 + 32).alias("v"),
                    pl.lit("proxy").alias("status"), pl.col("k").cast(pl.Int64).alias("contributors")).sort("t")


# ---- climatological diurnal drift --------------------------------------------------------------
HARMONICS = 3


def local_hour(t_min: Any, city: str) -> Any:
    """Fractional local clock hour for minute numbers since the epoch (vectorised)."""
    import numpy as np
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(TZ[city])
    t_min = np.asarray(t_min, dtype=np.int64)
    # Offsets change only at DST transitions: evaluate per unique hour to keep it cheap.
    hrs = t_min // 60
    uh, inv = np.unique(hrs, return_inverse=True)
    off = np.array([datetime.fromtimestamp(int(x) * 3600, UTC).astimezone(tz).utcoffset().total_seconds() / 60
                    for x in uh])
    return ((t_min + off[inv]) % 1440) / 60.0


def fourier(hour: Any, k: int = HARMONICS) -> Any:
    import numpy as np

    w = 2 * np.pi * np.asarray(hour) / 24
    cols = [np.ones_like(w)]
    for j in range(1, k + 1):
        cols += [np.cos(j * w), np.sin(j * w)]
    return np.column_stack(cols)


def fit_drift(grids: list[dict[str, tuple[Any, Any]]], half: int = 30) -> dict[str, Any]:
    """Per city, the mean index slope (deg F per minute) as a Fourier series in local hour,
    fitted by least squares on centred differences (v[t+half] - v[t-half]) / (2 half)."""
    import numpy as np

    out = {}
    cities = sorted({c for g in grids for c in g})
    for c in cities:
        X, Y = [], []
        for g in grids:
            if c not in g:
                continue
            lo, v = g[c]
            n = len(v)
            i = np.arange(half, n - half)
            y = (v[i + half] - v[i - half]) / (2 * half)
            ok = np.isfinite(y)
            X.append(fourier(local_hour(lo + i[ok], c)))
            Y.append(y[ok])
        X, Y = np.vstack(X), np.concatenate(Y)
        out[c] = np.linalg.lstsq(X, Y, rcond=None)[0]
    return out


def drift_sum(coef: Any, city: str, a_min: Any, b_min: Any) -> Any:
    """Integral of the climatological slope over minutes (a, b] for each pair (vectorised)."""
    import numpy as np

    a_min, b_min = np.asarray(a_min, dtype=np.int64), np.asarray(b_min, dtype=np.int64)
    out = np.zeros(len(a_min))
    span = int((b_min - a_min).max()) if len(a_min) else 0
    for k in range(1, span + 1):
        m = (a_min + k) <= b_min
        if m.any():
            out[m] += fourier(local_hour(a_min[m] + k, city)) @ coef
    return out


# ---- decision-time features --------------------------------------------------------------------
LEADS = (55, 40, 25, 10)      # minutes before the settlement minute S (the market lists at S - 60)
MAX_STALE = 15                # drop a decision if the newest published point is older than this


def _last_valid(v: Any) -> Any:
    import numpy as np

    return np.maximum.accumulate(np.where(np.isfinite(v), np.arange(len(v)), -1))


def features(grid: dict[str, tuple[Any, Any]], city: str, S: Any, lead: int, coef: dict[str, Any] | None = None,
             delay: int = int(PUBLISH_DELAY.total_seconds() // 60)) -> dict[str, Any]:
    """Information set at t = S - lead: only index minutes u <= t - delay (received by then).
    Returns arrays: u0 (newest usable minute), I0, d15/d30/d60 (I0 minus the value 15/30/60 min
    earlier), vol (sd of 5-min changes over the past hour), h = S - u0, clim (climatological
    drift over (u0, S]) and ok (enough history and not stale)."""
    import numpy as np

    lo, v = grid[city]
    lv = _last_valid(v)
    S = np.asarray(S, dtype=np.int64)
    n = len(v)

    def at(m: Any) -> Any:
        i = np.clip(m - lo, 0, n - 1)
        j = lv[i]
        good = (m - lo >= 0) & (m - lo < n) & (j >= 0) & ((i - j) <= 3)
        return np.where(good, v[np.maximum(j, 0)], np.nan)

    t = S - lead
    i = np.clip(t - delay - lo, 0, n - 1)
    j = lv[i]
    inside = (t - delay - lo >= 0) & (t - delay - lo < n) & (j >= 0)
    u0 = np.where(inside, lo + np.maximum(j, 0), t - delay)
    I0 = np.where(inside, v[np.maximum(j, 0)], np.nan)
    d = {f"d{k}": I0 - at(u0 - k) for k in (15, 30, 60)}
    steps = np.column_stack([at(u0 - 5 * k) for k in range(13)])
    vol = np.nanstd(np.diff(steps[:, ::-1], axis=1), axis=1)
    ok = inside & np.isfinite(I0) & ((t - delay - u0) <= MAX_STALE) & np.isfinite(d["d60"]) & np.isfinite(vol)
    out = {"u0": u0, "I0": I0, "vol": vol, "h": (S - u0).astype(float), "ok": ok, **d}
    out["clim"] = drift_sum(coef[city], city, u0, S) if coef is not None else np.zeros(len(S))
    return out


# ---- NBM forecast change -----------------------------------------------------------------------
NBM_LAG = timedelta(hours=2)          # NBM text guidance is public ~1 h after the run time
NBM_CODE = {"MIA": "MIA", "NYC": "EWR", "CHI": "MDW", "LAX": "LAX"}


def nbm_change(nbm: pl.DataFrame, city: str, t_dec: Any, a: Any, b: Any, *, level: bool = False) -> Any:
    """Forecast change F(b) - F(a) (deg F) of the newest NBM run public at each decision minute
    ``t_dec`` (runtime + NBM_LAG <= t), linear in time between the 3-hourly guidance points.
    ``level=True`` returns F(a) instead (the forecast level, for the anomaly I0 - F)."""
    import numpy as np

    g = nbm.filter((pl.col("code") == NBM_CODE[city]) & pl.col("tmp").is_not_null()).sort("runtime", "ftime")
    runs = (g["runtime"].dt.epoch("s").to_numpy() // 60).astype(np.int64)
    ft = (g["ftime"].dt.epoch("s").to_numpy() // 60).astype(np.int64)
    tmp = g["tmp"].to_numpy()
    uniq = np.unique(runs)
    lag = int(NBM_LAG.total_seconds() // 60)
    t_dec, a, b = (np.asarray(x, dtype=np.int64) for x in (t_dec, a, b))
    newest = np.searchsorted(uniq, t_dec - lag, side="right") - 1
    out = np.full(len(t_dec), np.nan)
    # The newest public run starts its guidance ~6 h after its run time; fall back to older runs
    # (still public at t) until one covers [a, b].
    for back in range(4):
        pick = newest - back
        for r in np.unique(pick[pick >= 0]):
            sel = runs == uniq[r]
            x, y = ft[sel], tmp[sel]
            if len(x) < 2:
                continue
            ok = (pick == r) & np.isnan(out) & (a >= x[0]) & (b <= x[-1])
            out[ok] = np.interp(a[ok], x, y) if level else np.interp(b[ok], x, y) - np.interp(a[ok], x, y)
    return out


# ---- predictive model --------------------------------------------------------------------------
VARIANTS = {
    "persist": None,                                   # mean change 0
    "clim": "fixed",                                   # mean change = climatological drift
    "trend": ["clim", "d15", "d30", "d60"],            # fitted damped-anomaly regression
    "nbm": ["clim", "d15", "d30", "d60", "nbm"],       # + NBM forecast change
    "ou": ["clim", "d15", "d30", "d60", "nbm", "anom"],  # + anomaly vs NBM level (damped toward F)
    "hrrr": ["clim", "d15", "d30", "d60", "nbm", "anom", "hc", "ha"],  # + HRRR 15-min change and anomaly
}
LOG_ABS_Z = -0.6352                                    # E[log |z|] for z ~ N(0, 1)


def _design(f: dict[str, Any], cols: list[str]) -> Any:
    import numpy as np

    return np.column_stack([np.ones(len(f["I0"]))] + [np.nan_to_num(f[c]) for c in cols])


def _mean_change(f: dict[str, Any], variant: str, beta: Any) -> Any:
    import numpy as np

    spec = VARIANTS[variant]
    if spec is None:
        return np.zeros(len(f["I0"]))
    if spec == "fixed":
        return f["clim"]
    return _design(f, spec) @ beta


def _sigma_design(f: dict[str, Any]) -> Any:
    import numpy as np

    return np.column_stack([np.ones(len(f["I0"])), np.log(f["vol"] + 0.1), np.abs(f["clim"])])


def fit_variant(f: dict[str, Any], dy: Any, variant: str) -> dict[str, Any]:
    """Mean by OLS (or none / fixed), log-sigma by regressing log|resid| on log(vol), |clim|;
    keeps the sorted standardized residuals for an empirical predictive CDF."""
    import numpy as np

    spec = VARIANTS[variant]
    beta = None
    if isinstance(spec, list):
        beta = np.linalg.lstsq(_design(f, spec), dy, rcond=None)[0]
    r = dy - _mean_change(f, variant, beta)
    Z = _sigma_design(f)
    gamma = np.linalg.lstsq(Z, np.log(np.abs(r) + 0.02) - LOG_ABS_Z, rcond=None)[0]
    sig = np.exp(Z @ gamma)
    z = np.sort(r / sig)
    scale = float(np.std(z))                   # recalibrate so standardized residuals have sd 1
    return {"variant": variant, "beta": beta, "gamma": gamma, "scale": scale, "z": z / scale}


def predict_variant(fit: dict[str, Any], f: dict[str, Any]) -> tuple[Any, Any]:
    import numpy as np

    mu = f["I0"] + _mean_change(f, fit["variant"], fit["beta"])
    sig = np.exp(_sigma_design(f) @ fit["gamma"]) * fit["scale"]
    return mu, sig


def prob_above(mu: Any, sig: Any, thr: Any, z_sorted: Any | None = None) -> Any:
    """P(latent > thr) under N(mu, sig) or the empirical standardized-residual distribution."""
    import numpy as np
    from scipy.stats import norm

    x = (np.asarray(thr) - mu) / sig
    if z_sorted is None:
        return norm.sf(x)
    n = len(z_sorted)
    return 1 - (np.searchsorted(z_sorted, x, side="right") + 0.5) / (n + 1)


# ---- training samples and event predictions -----------------------------------------------------
CITY_VARIANT = {"MIA": "ou", "NYC": "ou", "CHI": "ou", "LAX": "ou"}   # best in every city/lead, dev day-CV
DEV_FROM = datetime(2026, 9, 22, tzinfo=UTC)   # minute index archive starts 2026-09-21 22:22 UTC
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
HOLDOUT_TO = datetime(2026, 11, 1, tzinfo=UTC)


def full_features(grid: dict[str, tuple[Any, Any]], fc: dict[str, pl.DataFrame], city: str, S: Any, lead: int,
                  coef: dict[str, Any], delay: int | None = None) -> dict[str, Any]:
    """Index features plus forecast features from ``fc``: NBM (``nbm`` change, ``anom`` = NBM
    level - I0) and, when present, HRRR (``hc`` change, ``ha`` = HRRR level - I0)."""
    import numpy as np

    f = features(grid, city, S, lead, coef, delay if delay is not None else DELAY_MIN)
    S = np.asarray(S, dtype=np.int64)
    nbm = fc["nbm"]
    f["nbm"] = nbm_change(nbm, city, S - lead, f["u0"], S)
    f["anom"] = nbm_change(nbm, city, S - lead, f["u0"], S, level=True) - f["I0"]
    f["ok"] = f["ok"] & np.isfinite(f["nbm"]) & np.isfinite(f["anom"])
    if fc.get("hrrr") is not None:
        f["hc"] = hrrr_feature(fc["hrrr"], city, S - lead, f["u0"], S)
        f["ha"] = hrrr_feature(fc["hrrr"], city, S - lead, f["u0"], S, level=True) - f["I0"]
        f["ok"] = f["ok"] & np.isfinite(f["hc"]) & np.isfinite(f["ha"])
    return f


def training_samples(grid: dict[str, tuple[Any, Any]], fc: dict[str, pl.DataFrame], city: str, lead: int,
                     coef: dict[str, Any], start: datetime, end: datetime, every: int = 5
                     ) -> tuple[dict[str, Any], Any, Any]:
    """Every ``every``-th minute S in [start, end) with a published index value: features at
    S - lead, the realised change I(S) - I0, and the UTC day (for day-blocked CV)."""
    import numpy as np

    lo, v = grid[city]
    a, b = int(start.timestamp() // 60), int(end.timestamp() // 60)
    S = np.arange(max(a, lo + 120), min(b, lo + len(v)), every)
    S = S[np.isfinite(v[S - lo])]
    f = full_features(grid, fc, city, S, lead, coef)
    ok = f["ok"]
    sel = {k: val[ok] for k, val in f.items()}
    return sel, (v[S - lo] - f["I0"])[ok], (S // 1440)[ok]


def event_frame(markets: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    ev = events(markets).filter((pl.col("S") >= start) & (pl.col("S") < end))
    return ev.with_columns((pl.col("S").dt.epoch("s") // 60).alias("S_min"))


def predict_events(ev: pl.DataFrame, grid: dict[str, tuple[Any, Any]], fc: dict[str, pl.DataFrame],
                   coef: dict[str, Any], fits: dict[tuple[str, int, int | None], dict[str, Any]],
                   leads: tuple[int, ...] = LEADS) -> pl.DataFrame:
    """Model probability for every market and lead. ``fits[(city, lead, day)]`` is the model to
    use for events on UTC ``day`` (None = the one frozen model)."""
    import numpy as np

    out = []
    for city in ev["city"].unique().to_list():
        e = ev.filter(pl.col("city") == city)
        n = members(city)
        S = e["S_min"].to_numpy().astype(np.int64)
        thr = np.array([threshold(k, n) for k in e["k"].to_list()])
        for lead in leads:
            f = full_features(grid, fc, city, S, lead, coef)
            p = np.full(len(S), np.nan)
            mu_all, sg_all = np.full(len(S), np.nan), np.full(len(S), np.nan)
            days = S // 1440
            for day in np.unique(days):
                fit = fits.get((city, lead, int(day))) or fits.get((city, lead, None))
                if fit is None:
                    continue
                m = (days == day) & f["ok"]
                if not m.any():
                    continue
                mu, sg = predict_variant(fit, {k: val[m] for k, val in f.items()})
                p[m] = prob_above(mu, sg, thr[m], fit["z"])
                mu_all[m], sg_all[m] = mu, sg
            out.append(e.with_columns(
                pl.lit(lead).alias("lead"),
                (pl.col("S") - pl.duration(minutes=lead)).alias("t"),
                pl.Series("p", p), pl.Series("mu", mu_all), pl.Series("sigma", sg_all),
                pl.Series("I0", f["I0"]), pl.Series("h", f["h"])))
    return pl.concat(out).filter(pl.col("p").is_not_nan())


def quotes_at(pred: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    """Executable yes bid/ask at t: the last 1-minute candle that ended at or before t (the book
    at that minute's end; a minute without a candle had no change). Two-sided books only."""
    c = (candles.select("ticker", pl.col("end_ts").alias("q_ts"), pl.col("yes_bid_close").alias("bid"),
                        pl.col("yes_ask_close").alias("ask"))
         .filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null()).sort("ticker", "q_ts"))
    j = pred.sort("ticker", "t").join_asof(c, left_on="t", right_on="q_ts", by="ticker", strategy="backward",
                                           check_sortedness=False)
    return j.filter(pl.col("q_ts").is_not_null() & (pl.col("bid") > 0) & (pl.col("ask") < 1)
                    & (pl.col("ask") > pl.col("bid")))


# ---- ladder optimisation -----------------------------------------------------------------------
def _fee(p: float) -> float:
    from albert.research.fundamental import taker_fee

    return taker_fee(p)


def ladder_trades(rows: list[dict[str, Any]], theta: float, lam: float | None) -> list[dict[str, Any]]:
    """One event at one decision time. ``rows``: the event's strikes with model p, bid, ask, y.
    Candidates: buy YES at the ask or NO at 1 - bid where the model edge after the taker fee is
    >= theta. lam None = take every candidate; lam = -1 = only the single best; otherwise the
    subset maximising E[PnL] - lam * Var[PnL] under the model's distribution over the ladder's
    outcome bins (brute force over at most 10 candidates)."""
    import itertools

    import numpy as np

    rows = sorted(rows, key=lambda r: r["k"])
    pk = np.minimum.accumulate(np.array([r["p"] for r in rows]))    # enforce P(>k) non-increasing
    bins = np.concatenate([[1 - pk[0]], pk[:-1] - pk[1:], [pk[-1]]])  # bin j: index between k_j, k_j+1
    bins = np.clip(bins, 0, None)
    bins = bins / bins.sum()
    cands = []
    for j, r in enumerate(rows):
        ask, bid = r["ask"], r["bid"]
        win = np.arange(len(bins)) > j                                 # YES on strike j pays in bins > j
        ey = pk[j] - ask - _fee(ask)
        en = (1 - pk[j]) - (1 - bid) - _fee(1 - bid)
        if ey >= theta:
            cands.append({**r, "side": "yes", "edge": ey, "pay": win.astype(float) - ask - _fee(ask),
                          "pnl": r["y"] - ask - _fee(ask), "cost": ask})
        if en >= theta:
            cands.append({**r, "side": "no", "edge": en, "pay": (~win).astype(float) - (1 - bid) - _fee(1 - bid),
                          "pnl": (1 - r["y"]) - (1 - bid) - _fee(1 - bid), "cost": 1 - bid})
    if not cands:
        return []
    if lam is None:
        return cands
    if lam < 0:
        return [max(cands, key=lambda c: c["edge"])]
    cands = sorted(cands, key=lambda c: -c["edge"])[:10]
    best, best_val = [], -np.inf
    for m in range(1, len(cands) + 1):
        for combo in itertools.combinations(range(len(cands)), m):
            pay = sum(cands[i]["pay"] for i in combo)
            mean = float(bins @ pay)
            var = float(bins @ (pay - mean) ** 2)
            val = mean - lam * var
            if val > best_val:
                best, best_val = [cands[i] for i in combo], val
    return best


def ladder_sim(df: pl.DataFrame, theta: float, lam: float | None) -> pl.DataFrame:
    """Apply :func:`ladder_trades` to every (event, lead) group; one row per contract bought."""
    out = []
    for _, g in df.group_by(["event", "lead"]):
        out.extend(ladder_trades(g.select("ticker", "event", "lead", "k", "p", "bid", "ask", "y", "t")
                                 .to_dicts(), theta, lam))
    if not out:
        return pl.DataFrame()
    return pl.DataFrame([{k: v for k, v in r.items() if k != "pay"} for r in out], infer_schema_length=None)


# ---- fitting -----------------------------------------------------------------------------------
def load_inputs(s, *, through: datetime) -> dict[str, Any]:
    """Everything the model may see for decisions before ``through``: the published minute index,
    the IEM one-minute proxy history (Sep-Oct 2025, Aug 1 - Sep 21 2026) for the diurnal drift
    climatology, NBM guidance, and the market list."""
    d = cache_dir(s)
    idx = pl.read_parquet(d / "index_minutes.parquet").filter(pl.col("t") < through)
    o25 = pl.read_parquet(d / "one_min_2025.parquet")
    o26 = pl.read_parquet(d / "one_min_2026.parquet").filter(pl.col("ts") < datetime(2026, 9, 21, tzinfo=UTC))
    prox = [minute_grid(pl.concat([proxy_index(o, c) for c in TZ])) for o in (o25, o26)]
    nbm = pl.read_parquet(s.tables_dir / "wx_nbm.parquet")
    nbm = nbm.filter(pl.col("runtime") + NBM_LAG < through)
    hrrr = None
    if (d / "hrrr.parquet").exists():
        hrrr = pl.read_parquet(d / "hrrr.parquet").filter(pl.col("run") + HRRR_LAG < through)
    return {"index": idx, "grid": minute_grid(idx), "proxy": prox, "fc": {"nbm": nbm, "hrrr": hrrr},
            "markets": pl.read_parquet(d / "markets.parquet")}


def fit_all(inp: dict[str, Any], fit_end: datetime, *, cv_days: bool,
            variants: dict[str, str] | None = None) -> tuple[dict[str, Any], dict[tuple, dict[str, Any]]]:
    """Drift climatology on proxy history + the real index before ``fit_end``; per (city, lead)
    model on real-index minutes in [DEV_FROM, fit_end). With ``cv_days`` also one model per
    left-out UTC day (dev predictions are then out-of-day)."""
    import numpy as np

    grid = {c: g for c, g in inp["grid"].items()}
    coef = fit_drift(inp["proxy"] + [grid])
    variants = variants or CITY_VARIANT
    fits: dict[tuple, dict[str, Any]] = {}
    for city in TZ:
        for lead in LEADS:
            f, dy, day = training_samples(grid, inp["fc"], city, lead, coef, DEV_FROM, fit_end)
            fits[(city, lead, None)] = fit_variant(f, dy, variants[city])
            if cv_days:
                for dd in np.unique(day):
                    m = day != dd
                    fits[(city, lead, int(dd))] = fit_variant({k: v[m] for k, v in f.items()}, dy[m],
                                                              variants[city])
    return coef, fits


# ---- study driver ------------------------------------------------------------------------------
LAMBDAS = (None, -1.0, 0.5, 2.0, 8.0)       # None = every candidate, -1 = single best trade


def load_candles(s) -> pl.DataFrame:
    d = cache_dir(s)
    files = sorted(d.glob("candles_1m_*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files]).unique(["ticker", "end_ts"])


def build_predictions(s, *, holdout: bool) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Dev predictions are out-of-day (leave-one-UTC-day-out fits on dev); holdout predictions come
    from one model frozen on all data before HOLDOUT_FROM. Decision-time features at holdout
    events use the index published before each decision (never after it)."""
    inp_fit = load_inputs(s, through=HOLDOUT_FROM)
    coef, fits = fit_all(inp_fit, HOLDOUT_FROM, cv_days=True)
    dev_ev = event_frame(inp_fit["markets"], DEV_FROM + timedelta(hours=1), HOLDOUT_FROM)
    dev_fits = {key: f for key, f in fits.items() if key[2] is not None}
    parts = [predict_events(dev_ev, inp_fit["grid"], inp_fit["fc"], coef, dev_fits).with_columns(
        pl.lit("dev").alias("split"))]
    if holdout:
        inp_all = load_inputs(s, through=HOLDOUT_TO)
        frozen = {key: f for key, f in fits.items() if key[2] is None}
        # S from 01:00 so every decision time (S - 55 min at the earliest) is inside October
        hold_ev = event_frame(inp_all["markets"], HOLDOUT_FROM + timedelta(hours=1), HOLDOUT_TO)
        parts.append(predict_events(hold_ev, inp_all["grid"], inp_all["fc"], coef, frozen).with_columns(
            pl.lit("holdout").alias("split")))
    pred = pl.concat(parts, how="diagonal_relaxed")
    q = quotes_at(pred, load_candles(s)).with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    meta = {"fits": {f"{c}/{ld}": {"variant": f["variant"],
                                   "beta": None if f["beta"] is None else [round(float(b), 3) for b in f["beta"]],
                                   "gamma": [round(float(g), 3) for g in f["gamma"]], "scale": round(f["scale"], 3)}
                     for (c, ld, d), f in fits.items() if d is None},
            "drift_F_per_hour_at_local_0_3_6_9_12_15_18_21": {
                c: [round(float(x) * 60, 2) for x in fourier(range(0, 24, 3)) @ coef[c]] for c in coef}}
    return q, meta


def segment_scores(q: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    return (q.group_by(["split", *by])
            .agg(pl.len().alias("n"), pl.col("event").n_unique().alias("events"),
                 ((pl.col("p") - pl.col("y")) ** 2).mean().alias("brier_model"),
                 ((pl.col("mid") - pl.col("y")) ** 2).mean().alias("brier_mid"),
                 (pl.col("ask") - pl.col("bid")).mean().alias("spread"))
            .with_columns((pl.col("brier_model") - pl.col("brier_mid")).alias("model_minus_mid"))
            .sort(["split", *by]))


def calibration(q: pl.DataFrame, col: str) -> pl.DataFrame:
    edges = [0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0001]
    return (q.with_columns(pl.col(col).cut(edges[1:-1]).alias("bin"))
            .group_by("split", "bin").agg(pl.len().alias("n"), pl.col(col).mean().alias("mean_pred"),
                                          pl.col("y").mean().alias("freq"))
            .sort("split", "bin"))


def ladder_table(q: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    from albert.research.fundamental import THETAS, clustered_mean

    rows, best = [], None
    for theta in THETAS:
        for lam in LAMBDAS:
            for split in ("dev", "holdout"):
                part = q.filter(pl.col("split") == split)
                if not part.height:
                    continue
                tr = ladder_sim(part, theta, lam)
                name = {None: "all", -1.0: "single"}.get(lam, f"mv{lam}")
                if not tr.height:
                    rows.append({"theta": theta, "rule": name, "split": split, "contracts": 0})
                    continue
                per_ev = tr.group_by("event", "lead").agg(pl.col("pnl").sum())
                mu, tt, k = clustered_mean(tr["pnl"].to_numpy(), tr["event"].to_numpy())
                rows.append({"theta": theta, "rule": name, "split": split, "contracts": tr.height,
                             "event_leads": per_ev.height, "events": k, "pnl_c_per_contract": round(mu * 100, 3),
                             "t": None if tt is None else round(tt, 2), "total_usd": round(float(tr["pnl"].sum()), 2),
                             "pnl_per_dollar": round(float(tr["pnl"].sum() / tr["cost"].sum()), 4)})
                if split == "dev" and tr.height >= 30 and (best is None or tr["pnl"].sum() / tr["cost"].sum() > best[2]):
                    best = (theta, name, float(tr["pnl"].sum() / tr["cost"].sum()))
    return pl.DataFrame(rows, infer_schema_length=None), {"chosen_on_dev": best}


def run(s, *, holdout: bool = False) -> dict[str, Any]:
    """Whole study. ``holdout=False`` touches only dev (Sep 22-30) events; run ``holdout=True``
    once, after the model, variants and rules are frozen."""
    from albert.research.fundamental import evaluate, report

    q, meta = build_predictions(s, holdout=holdout)
    q.write_parquet(cache_dir(s) / f"predictions_{'holdout' if holdout else 'dev'}.parquet")
    results = {"pooled": evaluate(q, HOLDOUT_FROM)}
    for lead in LEADS:
        results[f"lead{lead}"] = evaluate(q.filter(pl.col("lead") == lead), HOLDOUT_FROM)
    seg_city = segment_scores(q, ["city", "lead"])
    seg_band = segment_scores(q.with_columns(pl.col("mid").cut([0.1, 0.3, 0.7, 0.9]).alias("mid_band")),
                              ["mid_band"])
    lad, lad_meta = ladder_table(q)
    extra = {"study": "hourly_temp", "leads_min": list(LEADS), "dev": f"{DEV_FROM:%Y-%m-%d}..2026-09-30",
             "holdout": "2026-10" if holdout else "not run", **meta, **lad_meta,
             "quote_age_min_median": float((q["t"] - q["q_ts"]).dt.total_minutes().median()),
             "rows_by_split": q.group_by("split").len().to_dicts()}
    path = report(s, NAME, results["pooled"], extra)
    _append_tables(path, results, seg_city, seg_band, lad, calibration(q, "p"), calibration(q, "mid"))
    return {"report": path, "results": results, "seg_city": seg_city, "seg_band": seg_band, "ladder": lad,
            "meta": extra}


def _append_tables(path: Path, results: dict[str, Any], seg_city: pl.DataFrame, seg_band: pl.DataFrame,
                   lad: pl.DataFrame, cal_p: pl.DataFrame, cal_mid: pl.DataFrame) -> None:
    import json as _json

    def md(df: pl.DataFrame) -> str:
        cols = df.columns
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in df.iter_rows():
            lines.append("| " + " | ".join("" if v is None else (f"{v:.5g}" if isinstance(v, float) else str(v))
                                           for v in r) + " |")
        return "\n".join(lines)

    out = ["", "## By lead (each lead evaluated separately; blend and theta fitted on dev)", ""]
    for lead in LEADS:
        r = results[f"lead{lead}"]
        out += [f"### Lead {lead} min", "", md(pl.DataFrame(r["scores"])), "",
                "Blend: `" + _json.dumps(r.get("blend")) + "`", "",
                f"Taker (theta chosen on dev = {r.get('theta_chosen_on_dev')}):", "",
                md(pl.DataFrame(r["taker"], infer_schema_length=None)), ""]
    out += ["## Brier by city and lead", "", md(seg_city), "", "## Brier by market-mid band", "", md(seg_band), "",
            "## Ladder optimisation (taker, held to settlement)", "", md(lad), "",
            "## Calibration: model p", "", md(cal_p.with_columns(pl.col("bin").cast(pl.String))), "",
            "## Calibration: market mid", "", md(cal_mid.with_columns(pl.col("bin").cast(pl.String))), ""]
    with path.open("a") as fh:
        fh.write("\n".join(out))


# ---- HRRR 15-minute forecasts (Open-Meteo single-runs archive) -------------------------------------
OM_SINGLE = "https://single-runs-api.open-meteo.com"
HRRR_LAG = timedelta(hours=2)          # run init + 2 h: HRRR posts ~1 h after init, plus archive delay
HRRR_EVERY = 3                         # the archive keeps the 00/03/.../21 UTC runs
# Member-station coordinates (IEM network geojson, FL/NY/NJ/IL/CA_ASOS); the HRRR point per city is
# the members' centroid (one location per city keeps the Open-Meteo call weight low).
COORDS = {"FLL": (26.0787, -80.1622), "FXE": (26.2, -80.17), "MIA": (25.788, -80.3169), "OPF": (25.9102, -80.2828),
          "PMP": (26.2464, -80.111), "FRG": (40.7288, -73.4134), "HPN": (41.0669, -73.7075),
          "ISP": (40.7939, -73.1017), "LGA": (40.7794, -73.8803), "CDW": (40.8764, -74.2831),
          "EWR": (40.6827, -74.1693), "SMQ": (40.6239, -74.6694), "TEB": (40.859, -74.0562),
          "ARR": (41.77, -88.48), "DPA": (41.9078, -88.2486), "MDW": (41.786, -87.7524), "ORD": (41.9602, -87.9316),
          "PWK": (42.1208, -87.9047), "HHR": (33.9228, -118.3352), "LAX": (33.9382, -118.3865),
          "LGB": (33.8118, -118.1472), "SMO": (34.021, -118.4471), "SNA": (33.6757, -117.8682)}


def city_point(city: str) -> tuple[float, float]:
    mem = INDEX[next(s for s, c in CITY.items() if c == city)]
    lat = sum(COORDS[m][0] for m in mem) / len(mem)
    lon = sum(COORDS[m][1] for m in mem) / len(mem)
    return round(lat, 4), round(lon, 4)


def fetch_hrrr_run(client: CachedClient, run: datetime, cities: list[str], hours: int = 9) -> Any:
    pts = [city_point(c) for c in cities]
    return client.get("/v1/forecast", {
        "latitude": ",".join(str(p[0]) for p in pts), "longitude": ",".join(str(p[1]) for p in pts),
        "models": "ncep_hrrr_conus", "run": run.strftime("%Y-%m-%dT%H:%M"), "minutely_15": "temperature_2m",
        "temperature_unit": "fahrenheit", "forecast_hours": hours}).payload


def hrrr_rows(payload: Any, run: datetime, cities: list[str]) -> list[dict[str, Any]]:
    rows = []
    for c, loc in zip(cities, payload if isinstance(payload, list) else [payload]):
        m = loc.get("minutely_15") or {}
        for ts, v in zip(m.get("time") or [], m.get("temperature_2m") or []):
            if v is not None:
                rows.append({"city": c, "run": run, "valid": datetime.fromisoformat(ts).replace(tzinfo=UTC),
                             "tf": float(v)})
    return rows


HRRR_SCHEMA = {"city": pl.String, "run": pl.Datetime("us", "UTC"), "valid": pl.Datetime("us", "UTC"),
               "tf": pl.Float64}


def load_hrrr(s, start: datetime, end: datetime) -> pl.DataFrame:
    """Every archived HRRR run initialised in [start, end), 15-minute 2 m temperature at each
    city's member centroid. Runs missing from the archive are skipped."""
    from albert.http import HttpError

    cl = CachedClient("openmeteo", OM_SINGLE, cache_dir(s), min_interval=1.0)
    cities = list(TZ)
    rows: list[dict[str, Any]] = []
    run = start
    while run < end:
        try:
            rows.extend(hrrr_rows(fetch_hrrr_run(cl, run, cities), run, cities))
        except HttpError as error:
            if "request limit" in str(error):        # Open-Meteo hourly/daily quota (shared IP)
                import time

                now = datetime.now(UTC)
                print(f"quota hit at {now:%H:%M}; waiting for the next hour", flush=True)
                time.sleep(3630 - now.minute * 60 - now.second)
                continue
            print(f"skip HRRR run {run:%Y-%m-%dT%H}: {str(error)[-120:]}", flush=True)
        run += timedelta(hours=HRRR_EVERY)
    return pl.DataFrame(rows, schema=HRRR_SCHEMA)


def hrrr_feature(hrrr: pl.DataFrame, city: str, t_dec: Any, a: Any, b: Any, *, level: bool = False) -> Any:
    """HRRR change F(b) - F(a) (or level F(a)) from the newest run with init + HRRR_LAG <= t."""
    import numpy as np

    g = hrrr.filter(pl.col("city") == city).sort("run", "valid")
    runs = (g["run"].dt.epoch("s").to_numpy() // 60).astype(np.int64)
    vt = (g["valid"].dt.epoch("s").to_numpy() // 60).astype(np.int64)
    tf = g["tf"].to_numpy()
    uniq = np.unique(runs)
    lag = int(HRRR_LAG.total_seconds() // 60)
    t_dec, a, b = (np.asarray(x, dtype=np.int64) for x in (t_dec, a, b))
    newest = np.searchsorted(uniq, t_dec - lag, side="right") - 1
    out = np.full(len(t_dec), np.nan)
    for back in range(3):
        pick = newest - back
        for r in np.unique(pick[pick >= 0]):
            sel = runs == uniq[r]
            x, y = vt[sel], tf[sel]
            if len(x) < 2:
                continue
            ok = (pick == r) & np.isnan(out) & (a >= x[0]) & (b <= x[-1])
            out[ok] = np.interp(a[ok], x, y) if level else np.interp(b[ok], x, y) - np.interp(a[ok], x, y)
    return out


def blend_taker(q: pl.DataFrame) -> pl.DataFrame:
    """Is the blend's extra information tradable? Per lead, fit on dev (i) the blend
    logit P = a + b logit(mid) + c logit(p) and (ii) the market-only recalibration
    a + b logit(mid); use each as the taker's probability (theta chosen on dev) on holdout.
    If (ii) earns as much as (i), the edge is the mid's miscalibration, not the model."""
    import numpy as np

    from albert.research.fundamental import evaluate, fit_logistic, logit

    rows = []
    for lead in LEADS:
        d = q.filter(pl.col("lead") == lead)
        dev = d.filter(pl.col("split") == "dev")
        specs = {"blend": lambda x: np.column_stack([np.ones(x.height), logit(x["mid"].to_numpy()),
                                                     logit(x["p"].to_numpy())]),
                 "mid_recalibrated": lambda x: np.column_stack([np.ones(x.height), logit(x["mid"].to_numpy())])}
        for name, design in specs.items():
            beta, _ = fit_logistic(design(dev), dev["y"].to_numpy().astype(float), dev["event"].to_numpy())
            r = evaluate(d.with_columns(pl.Series("p", 1 / (1 + np.exp(-design(d) @ beta)))), HOLDOUT_FROM)
            th = r["theta_chosen_on_dev"]
            for x in r["taker"]:
                if x["theta"] == th:
                    rows.append({"lead": lead, "prob": name, "theta_dev": th, **{k: v for k, v in x.items()
                                                                                  if k != "theta"}})
    return pl.DataFrame(rows, infer_schema_length=None)
