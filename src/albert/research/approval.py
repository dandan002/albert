"""Poll-average mechanics vs Kalshi presidential-approval markets.

Kalshi settles ``KXTRUMPAPPROVE`` (daily, 1:00 PM ET, exact 0.1 values plus two tails),
``KXAPRPOTUS`` (weekly, Friday 11:00 AM ET, 0.3-wide brackets) and ``KXAPRPOTUSD`` (daily,
11:00 AM ET, Aug 2026 only) on the RealClearPolitics (RCP) "President Trump Job Approval" average:
the *Approve* number as displayed (one decimal).

Data (public only). realclearpolling.com answers scripts with a DataDome captcha (HTTP 403), so
it is never scraped. Every Internet Archive (Wayback) capture of the page in 2026 is read instead
(``id_`` raw mode, cached under ``data/cache/approval/wayback``). Each capture embeds, in its
Next.js payload, the displayed average, the polls in it (``poll_rcp_avg``), the listed polls with
field dates and RCP's ``updated`` stamp (US Central), and the daily chart of the average. RCP later
deletes the daily Rasmussen tracker rows from its list; they are restored from Rasmussen's public
history table. Verified: in all 198 captures the displayed number is the plain mean of the
in-average polls rounded to 0.1. Membership is discretionary (one poll per pollster; old polls
are pruned in irregular batches), so no fixed rule reproduces it; the model tracks it instead.

As-of rules (no look-ahead), for a decision at ``t``:

* a poll is known from its ``updated`` stamp (from the earliest capture showing that version).
  Check on the dense October captures: 4 of 54 new rows carried a stamp 0.6-20 h earlier than a
  capture that did not yet show them (stale page or late publication), so a robustness run
  delays every entry by 6 h. Restored Rasmussen rows: release day 15:00 UTC.
* displayed values ("anchors"): a Wayback capture is public at capture time; a Kalshi settlement
  value at its ``settlement_ts``; a chart end-of-day value for ET date D at the end of D.
* the Kalman filter, pollster cadences and house effects use only polls known at ``t``; quotes
  come from the last hourly candle that closed by ``t`` (``fundamental.attach_quotes``).
"""

from __future__ import annotations

import gzip
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import polars as pl

PAGE = "https://www.realclearpolling.com/polls/approval/donald-trump/approval-rating"
CDX = "https://web.archive.org/cdx/search/cdx"
UA = "albert-research/0.2 (public-data research; github.com/dandan002/albert)"
SERIES = ["KXTRUMPAPPROVE", "KXAPRPOTUS", "KXAPRPOTUSD"]


# ---- fetching (Wayback only; every response cached, >= 1 s between requests) ------------------
class Fetcher:
    def __init__(self, cache_dir: Path, min_interval: float = 1.0) -> None:
        self.cache_dir = cache_dir
        self.min_interval = min_interval
        self._last = 0.0
        self._http = httpx.Client(timeout=60, headers={"User-Agent": UA}, follow_redirects=True)

    def _get(self, url: str, params: dict[str, Any] | None = None, tries: int = 6) -> httpx.Response:
        for attempt in range(tries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self._http.get(url, params=params)
            except httpx.HTTPError:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(10 * (attempt + 1))
                continue
            return r
        raise RuntimeError(f"giving up on {url}")

    def cdx(self, url: str, start: str, end: str) -> list[dict[str, str]]:
        """Wayback captures of ``url`` between ``start`` and ``end`` (YYYYMMDD). Cached per day of
        fetch, so a re-run on a later day sees new captures; older index files are kept."""
        path = self.cache_dir / "cdx" / f"{re.sub(r'[^a-z0-9]+', '_', url.lower())[-60:]}_{start}_{end}.json"
        if not path.exists():
            r = self._get(CDX, {"url": url, "from": start, "to": end, "output": "json",
                                "fl": "original,timestamp,statuscode,length,digest"})
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(r.text)
        rows = json.loads(path.read_text() or "[]")
        return [dict(zip(rows[0], r, strict=True)) for r in rows[1:]] if rows else []

    def snapshot(self, timestamp: str, original: str) -> bytes | None:
        """Raw (``id_``) bytes of one capture, cached forever (captures are immutable)."""
        path = self.cache_dir / "wayback" / f"{timestamp}.html.gz"
        if path.exists():
            return gzip.decompress(path.read_bytes())
        failed = path.with_suffix(".failed")
        if failed.exists():
            return None
        try:
            r = self._get(f"https://web.archive.org/web/{timestamp}id_/{original}", tries=3)
        except RuntimeError:
            r = None
        if r is None or r.status_code != 200:
            failed.parent.mkdir(parents=True, exist_ok=True)
            failed.write_text(str(r.status_code if r is not None else "error"))
            return None
        body = r.content
        if body[:2] == b"\x1f\x8b":          # Wayback replays the original gzip encoding
            body = gzip.decompress(body)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(body))
        tmp.replace(path)
        return body


# ---- parsing ---------------------------------------------------------------------------------
_PUSH = re.compile(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', re.DOTALL)
_ROW = re.compile(r"^([0-9a-f]+):(.*)$", re.DOTALL)


def rsc_rows(html: str) -> dict[str, str]:
    """The Next.js RSC payload of a page as {row id: row text}."""
    payload = "".join(json.loads('"' + c + '"') for c in _PUSH.findall(html))
    rows: dict[str, str] = {}
    for line in payload.split("\n"):
        m = _ROW.match(line)
        if m:
            rows[m.group(1)] = m.group(2)
    return rows


def _f(x: Any) -> float | None:
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return None


def _cand(p: dict[str, Any], name: str) -> float | None:
    for c in p.get("candidate") or []:
        if isinstance(c, dict) and c.get("name") == name:
            return _f(c.get("value"))
    return None


def _date(s: str | None):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y/%m/%d").date()
    except ValueError:
        return None


@dataclass
class Snapshot:
    ts: datetime                       # capture time (UTC)
    avg_approve: float | None
    avg_disapprove: float | None
    avg_window: str | None
    in_avg: list[dict[str, Any]]       # rows of the "Poll Data" table that are in the average
    polls: list[dict[str, Any]]        # every listed poll (in average or not)
    chart: list[dict[str, Any]]        # daily average: {date, approve}


def _poll_row(p: dict[str, Any]) -> dict[str, Any]:
    upd = p.get("updated")
    try:
        updated = parsedate_to_datetime(upd).astimezone(UTC) if upd else None
    except (TypeError, ValueError):
        updated = None
    size = str(p.get("sampleSize") or "")
    m = re.match(r"\s*([\d,]+)?\s*([A-Za-z]+)?", size)
    return {"id": str(p.get("id")), "pollster": p.get("pollster"), "group_id": p.get("pollster_group_id"),
            "updated": updated, "start": _date(p.get("data_start_date")), "end": _date(p.get("data_end_date")),
            "approve": _cand(p, "Approve"), "disapprove": _cand(p, "Disapprove"),
            "n": int(m.group(1).replace(",", "")) if m and m.group(1) else None,
            "pop": (m.group(2) or "").upper() if m else "", "in_avg": p.get("type") == "poll_rcp_avg",
            "pollster_type": p.get("pollster_type")}


def parse_snapshot(html: str, ts: datetime) -> Snapshot | None:
    """Parse one copy of the RCP approval page. Returns None when the page has no poll table."""
    rows = rsc_rows(html)
    table = None
    full: list[dict[str, Any]] = []
    chart: list[dict[str, Any]] = []
    for text in rows.values():
        head = text[:400]
        if '"tableTitle":"Poll Data"' in head and table is None and text.startswith('["$"'):
            try:
                table = json.loads(text)[3]
            except (json.JSONDecodeError, IndexError, KeyError, TypeError):
                table = None
        elif '"tableTitle":"Poll Data"' in head and text.startswith("[["):
            try:
                for el in json.loads(text):
                    if isinstance(el, list) and len(el) > 3 and isinstance(el[3], dict):
                        full = [p for p in el[3].get("polls", []) if isinstance(p, dict)]
            except json.JSONDecodeError:
                pass
        elif '"candidatesData"' in head and not chart:
            try:
                d = json.loads(text)[3]
            except (json.JSONDecodeError, IndexError, KeyError, TypeError):
                continue
            appr = next((c for c in d.get("candidatesData", []) if c.get("name") == "Approve"), None)
            if appr is None:
                continue
            for v in appr.get("votes", []):
                dt = str(v.get("date", "")).lstrip("$D")
                try:
                    stamp = datetime.fromisoformat(dt)
                except ValueError:
                    continue
                chart.append({"stamp": stamp, "approve": _f(v.get("value"))})
    if table is None:
        return None
    polls = table.get("polls", [])
    avg = next((p for p in polls if isinstance(p, dict) and p.get("type") == "rcp_average"), None)
    in_avg = [_poll_row(p) for p in polls if isinstance(p, dict) and p.get("type") == "poll_rcp_avg"]
    listed = {r["id"]: r for r in (_poll_row(p) for p in full if p.get("type") in ("poll", "poll_rcp_avg"))}
    for r in in_avg:
        listed[r["id"]] = r
    return Snapshot(ts=ts, avg_approve=_cand(avg, "Approve") if avg else None,
                    avg_disapprove=_cand(avg, "Disapprove") if avg else None,
                    avg_window=avg.get("date") if avg else None, in_avg=in_avg,
                    polls=list(listed.values()), chart=chart)


def load_snapshots(cache_dir: Path, start: str = "20260601", end: str = "20261105",
                   fetch: bool = True) -> list[Snapshot]:
    """Every parseable Wayback copy of the RCP approval page in [start, end]."""
    f = Fetcher(cache_dir)
    caps = [c for c in f.cdx(PAGE, start, end) if c["statuscode"] == "200"] if fetch else []
    if not fetch:
        caps = [{"timestamp": p.name.split(".")[0], "original": PAGE}
                for p in sorted((cache_dir / "wayback").glob("*.html.gz"))]
    out: list[Snapshot] = []
    seen: set[str] = set()
    for c in caps:
        if c["timestamp"] in seen:
            continue
        seen.add(c["timestamp"])
        try:
            body = f.snapshot(c["timestamp"], c["original"]) if fetch else gzip.decompress(
                (cache_dir / "wayback" / f"{c['timestamp']}.html.gz").read_bytes())
        except RuntimeError as error:          # archive.org hiccup: skip this copy, keep going
            print(f"skip capture {c['timestamp']}: {error}")
            continue
        if body is None:
            continue
        ts = datetime.strptime(c["timestamp"], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        snap = parse_snapshot(body.decode("utf-8", "replace"), ts)
        if snap is not None:
            out.append(snap)
    return sorted(out, key=lambda s: s.ts)


def rcp_mean(values: list[float] | np.ndarray) -> float:
    """RCP's displayed average: simple mean rounded half-up to one decimal."""
    v = float(np.mean(values))
    return float(np.floor(v * 10 + 0.5 + 1e-9) / 10)


# ---- Kalshi side -----------------------------------------------------------------------------
_TO = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)\s*%?\s*$", re.IGNORECASE)
_EX = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%?\s*$")
_ABOVE = re.compile(r"^\s*above\s*(\d+(?:\.\d+)?)\s*%?\s*$", re.IGNORECASE)
_BELOW = re.compile(r"^\s*below\s*(\d+(?:\.\d+)?)\s*%?\s*$", re.IGNORECASE)
LO_INF, HI_INF = 0, 1000                          # tenths of a point


def bracket_tenths(sub: str) -> tuple[int, int] | None:
    """Inclusive [lo, hi] in tenths of a point for a Kalshi approval leg's ``yes_sub_title``:
    "39.5%" / "39.5" -> [395, 395]; "38.8 to 39.0" -> [388, 390]; "Above 40.5" -> [406, inf];
    "Below 38.8" -> [-inf, 387]. RCP shows one decimal, so the outcome space is the 0.1 grid."""
    if m := _TO.match(sub):
        return round(float(m.group(1)) * 10), round(float(m.group(2)) * 10)
    if m := _ABOVE.match(sub):
        return round(float(m.group(1)) * 10) + 1, HI_INF
    if m := _BELOW.match(sub):
        return LO_INF, round(float(m.group(1)) * 10) - 1
    if m := _EX.match(sub):
        v = round(float(m.group(1)) * 10)
        return v, v
    return None


def kalshi_legs(markets: pl.DataFrame) -> pl.DataFrame:
    """Settled approval legs with bracket bounds (tenths), settlement time ``T`` and ``y``."""
    rows = []
    for r in markets.iter_rows(named=True):
        b = bracket_tenths(r["yes_sub_title"] or "")
        if b is None:
            continue
        # Settlement instant from the rules ("at exactly 1:00 PM ET" / "at 11:00 AM ET"):
        # it equals close_time for KXTRUMPAPPROVE (17:00 UTC, the 16:59 closes round up) and the
        # KXAPRPOTUS* close_time (15:00 UTC). Both windows are in EDT (UTC-4).
        hour = 17 if r["series_ticker"] == "KXTRUMPAPPROVE" else 15
        ct = r["close_time"]
        T = ct.replace(hour=hour, minute=0, second=0, microsecond=0)
        rows.append({"ticker": r["ticker"], "event": r["event_ticker"], "series": r["series_ticker"],
                     "lo": b[0], "hi": b[1], "T": T, "open_time": r["open_time"], "settled_at": r["settlement_ts"],
                     "y": int(r["result"] == "yes")})
    return pl.DataFrame(rows)


def settled_values(legs: pl.DataFrame) -> pl.DataFrame:
    """Per event: the winning bracket. ``value`` (tenths) is exact when the winner is a single
    0.1 value, else null (a tail or a 0.3-wide bracket won)."""
    w = legs.filter(pl.col("y") == 1).group_by("event", "series", "T").agg(
        pl.col("lo").first(), pl.col("hi").first(), pl.col("settled_at").max(), pl.len().alias("winners"))
    return w.with_columns(pl.when(pl.col("lo") == pl.col("hi")).then(pl.col("lo")).otherwise(None).alias("value"))


# ---- poll table with as-of times -------------------------------------------------------------
def family(pollster: str | None) -> str:
    """Pollster key for "one poll per pollster". RCP appends asterisks to some rows (footnotes;
    ``Reuters/Ipsos`` became ``Reuters/Ipsos**`` on an edit) and those rows carry no group id, so
    the key is the name without HTML entities and trailing asterisks. Verified on copies: the
    newest row of ``RMG Research`` / ``RMG Research*`` is the one in the average."""
    return (pollster or "").replace("&amp;", "&").strip().rstrip("*").strip()


def poll_versions(snaps: list[Snapshot]) -> pl.DataFrame:
    """One row per distinct version of each listed poll across all copies, with ``first_seen``
    (earliest copy showing that version) and ``known_at`` = the row's ``updated`` stamp when it is
    not later than ``first_seen`` (else ``first_seen``)."""
    rows = []
    for s in snaps:
        for p in s.polls:
            rows.append({**p, "snap": s.ts})
    df = pl.DataFrame(rows, infer_schema_length=None)
    key = ["id", "updated", "start", "end", "approve", "disapprove"]
    v = (df.group_by(key).agg(pl.col("snap").min().alias("first_seen"), pl.col("snap").max().alias("last_seen"),
                              pl.col("pollster").first(), pl.col("group_id").first(), pl.col("n").first(),
                              pl.col("pop").first(), pl.col("pollster_type").first()))
    v = v.with_columns(pl.col("pollster").str.replace_all("&amp;", "&", literal=True).str.strip_chars()
                       .str.strip_chars_end("*").str.strip_chars().alias("fam"),
                       pl.when(pl.col("updated").is_not_null() & (pl.col("updated") <= pl.col("first_seen")))
                       .then(pl.col("updated")).otherwise(pl.col("first_seen")).alias("known_at"))
    return v.sort("known_at")


def membership(snaps: list[Snapshot]) -> pl.DataFrame:
    """Which poll ids each copy shows *in* the average, with the displayed average."""
    rows = [{"snap": s.ts, "id": p["id"], "fam": family(p["pollster"]), "approve": p["approve"],
             "end": p["end"], "avg": s.avg_approve, "k": len(s.in_avg)} for s in snaps for p in s.in_avg]
    return pl.DataFrame(rows)


def asof(versions: pl.DataFrame, t: datetime) -> pl.DataFrame:
    """Polls as they stood at ``t``: per id the latest version known at or before ``t``."""
    return (versions.filter(pl.col("known_at") <= t).sort("known_at")
            .group_by("id").agg(pl.all().last()))


def latest_per_family(polls: pl.DataFrame) -> pl.DataFrame:
    return polls.sort("end", "known_at").group_by("fam").agg(pl.all().last())


# ---- state-space model: latent approval + pollster house effects ----------------------------
@dataclass
class KalmanFit:
    day0: Any                     # date of state index 0
    x: float                      # filtered latent approval on the last day
    var_x: float
    house: dict[str, float]       # posterior mean house effect per family (relative to the latent)
    var_house: dict[str, float]
    cov_xh: dict[str, float]
    q: float
    tau: float
    loglik: float


def kalman_house(polls: pl.DataFrame, last_day, *, q: float = 0.04, tau: float = 1.5,
                 prior_house: float = 3.0, days: int = 240) -> KalmanFit:
    """Local-level Kalman filter over days for the latent approval ``x_d`` with a static house
    effect per pollster family in the state. Poll i (family f, field midpoint d_i) observes
    ``x_{d_i} + h_f + e_i``, ``e_i ~ N(0, 100^2 p(1-p)/n_i + tau^2)``. Sum-to-zero is not imposed:
    a weak N(0, prior_house^2) prior per family plus a pseudo-observation that the mean house
    effect is 0 pins the level. Only polls in ``polls`` are used: the caller passes the as-of set.
    Polls are processed in field-midpoint order, so a late-released poll updates the day it
    describes; the result is the filtered state on ``last_day`` given exactly that set."""
    p = polls.filter(pl.col("approve").is_not_null() & pl.col("start").is_not_null() & pl.col("end").is_not_null())
    first = last_day - timedelta(days=days)
    p = p.with_columns((pl.col("start") + (pl.col("end") - pl.col("start")) / 2).dt.date().alias("mid"))
    p = p.filter((pl.col("mid") >= first) & (pl.col("mid") <= last_day)).sort("mid")
    fams = sorted(set(p["fam"].to_list()))
    idx = {f: i + 1 for i, f in enumerate(fams)}
    k = len(fams) + 1
    m = np.zeros(k)
    P = np.eye(k) * prior_house ** 2
    m[0] = float(p["approve"].mean()) if p.height else 40.0
    P[0, 0] = 25.0
    if len(fams) > 1:                                  # soft sum-to-zero of house effects
        H = np.zeros(k)
        H[1:] = 1.0 / len(fams)
        S = H @ P @ H + 0.25
        K = P @ H / S
        m = m + K * (0 - H @ m)
        P = P - np.outer(K, H @ P)
    ll = 0.0
    by_day: dict[Any, list[tuple[str, float, float]]] = {}
    for r in p.iter_rows(named=True):
        n = r["n"] or 1000
        pr = min(max(r["approve"] / 100, 0.05), 0.95)
        by_day.setdefault(r["mid"], []).append((r["fam"], r["approve"], 1e4 * pr * (1 - pr) / n + tau ** 2))
    d = first
    while d <= last_day:
        P[0, 0] += q
        for f, y, rv in by_day.get(d, []):
            H = np.zeros(k)
            H[0] = 1.0
            H[idx[f]] = 1.0
            S = H @ P @ H + rv
            e = y - H @ m
            ll += -0.5 * (np.log(2 * np.pi * S) + e * e / S)
            K = P @ H / S
            m = m + K * e
            P = P - np.outer(K, H @ P)
        d += timedelta(days=1)
    return KalmanFit(day0=first, x=float(m[0]), var_x=float(P[0, 0]),
                     house={f: float(m[i]) for f, i in idx.items()},
                     var_house={f: float(P[i, i]) for f, i in idx.items()},
                     cov_xh={f: float(P[0, i]) for f, i in idx.items()}, q=q, tau=tau, loglik=ll)


# ---- ladder optimisation ---------------------------------------------------------------------
def kalshi_taker_fee(p: float) -> float:
    from albert.research.fundamental import taker_fee
    return taker_fee(p)


def ladder_kelly(grid: np.ndarray, pmf: np.ndarray, legs: list[dict[str, Any]], *, theta: float = 0.0,
                 max_frac: float = 0.25) -> list[dict[str, Any]]:
    """Expected-log-growth portfolio over one event's ladder, taker only.

    ``grid``: outcome values (tenths), ``pmf``: model probabilities on the grid. Each leg has lo,
    hi (tenths, inclusive), bid, ask. Instruments: buy YES at ``ask`` or NO at ``1 - bid``, each
    plus the Kalshi taker fee. Only instruments whose own edge (model prob - price - fee) is at
    least ``theta`` are allowed. Maximises ``sum_v pmf_v log(1 + sum_j x_j (pay_vj - cost_j))``
    over stakes ``x_j >= 0`` (fractions of bankroll, total cost <= ``max_frac``) by projected
    Newton-free L-BFGS-B with a cost penalty. Returns the chosen instruments with their stake
    fraction (before any fractional-Kelly scaling)."""
    from scipy.optimize import minimize

    inst = []
    for leg in legs:
        cover = (grid >= leg["lo"]) & (grid <= leg["hi"])
        q = float(pmf[cover].sum())
        if leg.get("ask") is not None and 0 < leg["ask"] < 1:
            c = leg["ask"] + kalshi_taker_fee(leg["ask"])
            if q - c >= theta:
                inst.append({**leg, "side": "yes", "cost": c, "pay": cover.astype(float), "q": q})
        if leg.get("bid") is not None and 0 < leg["bid"] < 1:
            c = (1 - leg["bid"]) + kalshi_taker_fee(1 - leg["bid"])
            if (1 - q) - c >= theta:
                inst.append({**leg, "side": "no", "cost": c, "pay": (~cover).astype(float), "q": 1 - q})
    if not inst:
        return []
    R = np.column_stack([i["pay"] / i["cost"] - 1.0 for i in inst])     # return per $ staked
    keep = pmf > 0
    R, w = R[keep], pmf[keep]

    def f(x: np.ndarray) -> tuple[float, np.ndarray]:
        W = 1.0 + R @ x
        if np.any(W <= 1e-9):
            return 1e6, np.zeros_like(x)
        return float(-(w * np.log(W)).sum()), -(R.T @ (w / W))

    cons = {"type": "ineq", "fun": lambda x: max_frac - x.sum(), "jac": lambda x: -np.ones_like(x)}
    x0 = np.full(len(inst), 1e-4)
    res = minimize(f, x0, jac=True, bounds=[(0, max_frac)] * len(inst), constraints=[cons], method="SLSQP")
    out = []
    for i, x in zip(inst, res.x, strict=True):
        if x > 1e-4:
            out.append({k: v for k, v in i.items() if k != "pay"} | {"stake": float(x)})
    return out


# ---- the RCP average mechanics ---------------------------------------------------------------
ET = timedelta(hours=-4)          # all Kalshi approval settlements in the sample fall in EDT


def et_date(t: datetime):
    return (t + ET).date()


@dataclass
class Rule:
    """Reproduction rule for RCP's approval average: the latest poll of each eligible family whose
    field end date is no older than ``window`` days before the (ET) date of ``t``."""
    window: int = 21
    eligible_lookback: int = 150   # a family is eligible if a copy captured in the last N days
                                   # (before t) showed it in the average


def eligible_families(mem: pl.DataFrame, t: datetime, lookback: int) -> set[str]:
    m = mem.filter((pl.col("snap") <= t) & (pl.col("snap") > t - timedelta(days=lookback)))
    return set(m["fam"].unique().to_list())


def members_at(polls_t: pl.DataFrame, t: datetime, eligible: set[str], rule: Rule) -> pl.DataFrame:
    cutoff = et_date(t) - timedelta(days=rule.window)
    last = latest_per_family(polls_t.filter(pl.col("fam").is_in(list(eligible))))
    return last.filter(pl.col("end") >= cutoff)


# ---- Rasmussen daily tracker -----------------------------------------------------------------
RASMUSSEN_URL = ("https://www.rasmussenreports.com/public_content/politics/trump_administration_second_term/"
                 "trump_approval_index_history_second_term")
_RAS_ROW = re.compile(r'<tr id="history_data_(\d{8})">\s*<td[^>]*>[^<]+</td>\s*<td>\s*[-+\d]+\s*</td>\s*'
                      r"<td>(\d+)%</td>\s*<td>(\d+)%</td>\s*<td>(\d+)%</td>\s*<td>(\d+)%</td>")


def parse_rasmussen(html: str) -> pl.DataFrame:
    """Rasmussen's public daily history table: release date -> Total Approve / Disapprove.
    RCP lists the tracker as one row per release (field dates ending the day before); those rows
    are pruned from RCP's list later, so this table restores them."""
    rows = [{"date": datetime.strptime(d, "%Y%m%d").date(), "approve": float(a), "disapprove": float(b)}
            for d, _sa, _sd, a, b in _RAS_ROW.findall(html)]
    return pl.DataFrame(rows, schema={"date": pl.Date, "approve": pl.Float64, "disapprove": pl.Float64}).sort("date")


def rasmussen_history(cache_dir: Path) -> pl.DataFrame:
    """Latest cached copy of the Rasmussen history page (fetched with ``fetch_rasmussen``)."""
    files = sorted((cache_dir / "rasmussen").glob("trump_approval_index_history_second_term_fetched_*.html.gz"))
    if not files:
        return pl.DataFrame(schema={"date": pl.Date, "approve": pl.Float64, "disapprove": pl.Float64})
    return parse_rasmussen(gzip.decompress(files[-1].read_bytes()).decode("utf-8", "replace"))


def fetch_rasmussen(cache_dir: Path) -> Path:
    """Fetch the public history page once per day (append-only, date-stamped)."""
    path = cache_dir / "rasmussen" / f"trump_approval_index_history_second_term_fetched_{datetime.now(UTC):%Y%m%d}.html.gz"
    if not path.exists():
        r = Fetcher(cache_dir)._get(RASMUSSEN_URL)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(r.content))
    return path


def with_rasmussen(versions: pl.DataFrame, ras: pl.DataFrame, entry_utc_hour: float = 15.0) -> pl.DataFrame:
    """Add the Rasmussen tracker rows that RCP later pruned from its list. A release dated D is
    entered by RCP on D (observed entry times 13:00-15:10 UTC); a restored row is treated as known
    at D ``entry_utc_hour``:00 UTC unless a Wayback copy shows the real row (then that row wins).
    Field dates follow RCP's convention for the tracker: D-7 .. D-1."""
    seen = versions.filter(pl.col("fam") == "Rasmussen Reports")
    have = set(seen["end"].to_list())
    rows = []
    for r in ras.iter_rows(named=True):
        end = r["date"] - timedelta(days=1)
        if end in have:
            continue
        known = datetime(r["date"].year, r["date"].month, r["date"].day, tzinfo=UTC) + timedelta(hours=entry_utc_hour)
        rows.append({"id": f"ras-{r['date']:%Y%m%d}", "updated": None, "start": r["date"] - timedelta(days=7),
                     "end": end, "approve": r["approve"], "disapprove": r["disapprove"], "first_seen": known,
                     "last_seen": known, "pollster": "Rasmussen Reports", "group_id": "456", "n": 1500, "pop": "LV",
                     "pollster_type": "2", "fam": "Rasmussen Reports", "known_at": known})
    if not rows:
        return versions
    extra = pl.DataFrame(rows).cast({c: versions.schema[c] for c in versions.columns}).select(versions.columns)
    return pl.concat([versions, extra]).sort("known_at")


def release_times(versions: pl.DataFrame) -> pl.DataFrame:
    """First known time of every poll id (its RCP entry time as far as we can tell)."""
    return versions.group_by("id").agg(pl.col("fam").first(), pl.col("known_at").min().alias("rel"),
                                       pl.col("end").first())


def arrival_prob(rel: np.ndarray, t: datetime, T: datetime, weeks: int = 8) -> float:
    """P(a family enters a new poll in (t, T]) from the same window shifted back 1..``weeks`` weeks
    (captures day-of-week and time-of-day cadence). ``rel``: that family's entry times (all < t)."""
    hits = 0
    for k in range(1, weeks + 1):
        lo, hi = t - timedelta(days=7 * k), T - timedelta(days=7 * k)
        hits += bool(np.any((rel > np.datetime64(lo.replace(tzinfo=None))) & (rel <= np.datetime64(hi.replace(tzinfo=None)))))
    return (hits + 0.1) / (weeks + 0.2)


# ---- decision frame --------------------------------------------------------------------------
LEADS_H = {"KXTRUMPAPPROVE": (24, 4, 1), "KXAPRPOTUS": (162, 72, 24, 4), "KXAPRPOTUSD": (20, 4)}


def decision_frame(legs: pl.DataFrame) -> pl.DataFrame:
    """One row per (leg, decision time). Decision times are fixed offsets before the scheduled
    settlement instant ``T`` (known at listing), kept only if the market had been open >= 1 h."""
    rows = []
    for r in legs.iter_rows(named=True):
        for h in LEADS_H[r["series"]]:
            t = r["T"] - timedelta(hours=h)
            if t >= r["open_time"] + timedelta(hours=1):
                rows.append({**r, "t": t, "lead_h": h})
    return pl.DataFrame(rows)


# ---- what was public at t: anchors and the tracked membership --------------------------------
GRID = np.arange(300, 501)                     # outcome grid, tenths of a point


@dataclass
class Capture:
    ts: datetime                 # Wayback capture time (public at this time)
    content: datetime            # newest ``updated`` on the page (state time of what it shows)
    avg: float | None
    members: list[dict[str, Any]]


@dataclass
class Context:
    versions: pl.DataFrame       # poll versions incl. restored Rasmussen rows
    rel: pl.DataFrame            # first entry time per poll id
    captures: list[Capture]
    public: pl.DataFrame         # displayed values: state_ts, public_ts, value (tenths), source


def public_values(snaps: list[Snapshot], settled: pl.DataFrame) -> pl.DataFrame:
    """Every displayed RCP value with the time it described (``state_ts``) and the time it was
    public (``public_ts``): Wayback captures (public at capture), Kalshi settlements with an exact
    value (public at ``settlement_ts``) and the chart's end-of-day values (the chart point for ET
    date D is the value after D's last update, public at D's end; taken from the newest capture,
    dates whose end precedes that capture)."""
    rows = []
    for s in snaps:
        if s.avg_approve is None:
            continue
        ups = [p["updated"] for p in s.in_avg if p["updated"]]
        rows.append({"state_ts": max(ups) if ups else s.ts, "public_ts": s.ts, "value": round(s.avg_approve * 10),
                     "source": "wayback"})
    for r in settled.filter(pl.col("value").is_not_null()).iter_rows(named=True):
        rows.append({"state_ts": r["T"], "public_ts": r["settled_at"], "value": int(r["value"]), "source": "kalshi"})
    last = snaps[-1]
    for c in last.chart:
        d = c["stamp"].date()
        end = datetime(d.year, d.month, d.day, tzinfo=UTC) + timedelta(days=1, hours=5)   # ET midnight (EST-safe)
        if end <= last.ts and c["approve"] is not None:
            rows.append({"state_ts": end, "public_ts": end, "value": round(c["approve"] * 10), "source": "chart"})
    return pl.DataFrame(rows).sort("state_ts")


def build_context(snaps: list[Snapshot], ras: pl.DataFrame, settled: pl.DataFrame,
                  delay: timedelta = timedelta(0)) -> Context:
    """``delay`` shifts every poll's entry time later (robustness check of the as-of rule)."""
    v = with_rasmussen(poll_versions(snaps), ras)
    if delay:
        v = v.with_columns(pl.col("known_at") + delay)
    caps = []
    for s in snaps:
        if s.avg_approve is None or not s.in_avg:
            continue
        ups = [p["updated"] for p in s.in_avg if p["updated"]]
        caps.append(Capture(ts=s.ts, content=max(ups) if ups else s.ts, avg=s.avg_approve,
                            members=[{**p, "fam": family(p["pollster"])} for p in s.in_avg]))
    return Context(versions=v, rel=release_times(v), captures=sorted(caps, key=lambda c: c.ts),
                   public=public_values(snaps, settled))


def anchor(ctx: Context, t: datetime) -> dict[str, Any]:
    """The newest displayed value public at ``t`` (by the time it describes)."""
    p = ctx.public.filter(pl.col("public_ts") <= t).sort("state_ts", "public_ts")
    r = p.row(p.height - 1, named=True)
    return r


def track(ctx: Context, X: datetime, t: datetime, prune_window: int | None = None) -> dict[str, dict[str, Any]]:
    """Reconstructed membership at state time ``X`` using only what was public at ``t`` (X <= t):
    the newest Wayback capture taken by ``t`` whose content is not after ``X``, rolled forward with
    every poll entered in (content, X] (a newer poll replaces its family's member, a new family is
    added), then (optionally) polls whose field START is more than ``prune_window`` days before
    X's ET date are dropped (RCP prunes old polls in batches)."""
    caps = [c for c in ctx.captures if c.ts <= t and c.content <= X]
    base = caps[-1]
    mem = {m["fam"]: dict(m) for m in base.members}
    new = ctx.versions.filter((pl.col("known_at") > base.content) & (pl.col("known_at") <= X)).sort("known_at")
    for r in new.iter_rows(named=True):
        f = r["fam"]
        cur = mem.get(f)
        if cur is not None and cur["id"] != r["id"] and cur["end"] is not None and r["end"] is not None and r["end"] < cur["end"]:
            continue
        mem[f] = dict(r)
    if prune_window is not None:
        cut = et_date(X) - timedelta(days=prune_window)
        mem = {f: m for f, m in mem.items() if m["start"] is None or m["start"] >= cut}
    return mem


# ---- forecasting the average at settlement ---------------------------------------------------
@dataclass
class Params:
    prune_window: int = 19        # RCP keeps polls whose field start is within ~19 days ...
    prune_rate: float = 0.25      # ... but prunes only on some days (probability per day)
    q: float = 0.02               # latent daily variance (Kalman; dev MLE)
    tau: float = 1.0              # poll noise beyond sampling (points; dev MLE)
    sigma_res: float = 0.05       # residual noise on the mean (RCP discretion, page lag)
    weeks: int = 8                # arrival estimator lookback
    value_mode: str = "kalman"    # "kalman" (latent + house) or "last" (pollster's last value)
    sims: int = 4000


def _round_tenths(x: np.ndarray) -> np.ndarray:
    return np.floor(x * 10 + 0.5 + 1e-9).astype(int)


def _pmf(tenths: np.ndarray) -> np.ndarray:
    v = np.clip(tenths, GRID[0], GRID[-1])
    return np.bincount(v - GRID[0], minlength=len(GRID)).astype(float) / len(v)


def _start_level(a: dict[str, Any], mem: dict[str, dict[str, Any]], rng: np.random.Generator, S: int) -> np.ndarray:
    """Unrounded average at the anchor: exact when the reconstructed membership rounds to the
    displayed value, else uniform within the displayed value's rounding interval."""
    m = float(np.mean([x["approve"] for x in mem.values()]))
    if round(rcp_mean([x["approve"] for x in mem.values()]) * 10) == a["value"]:
        return np.full(S, m)
    return a["value"] / 10 + rng.uniform(-0.05, 0.05, S)


@dataclass
class Prepared:
    """Everything the Monte Carlo needs that does not depend on the tuned parameters."""
    anchor: dict[str, Any]
    mem_a: dict[str, dict[str, Any]]
    fams: list[str]
    cur_val: np.ndarray            # member value at t (nan if the family is not a member)
    cur_start: np.ndarray          # member field start (days since 2000-01-01; nan if none)
    ref_val: np.ndarray            # value used by the "last" mode
    p_arrive: np.ndarray
    n: np.ndarray
    is_tracker: np.ndarray
    fit: KalmanFit | None
    horizon_days: int
    days: list[int]                # ET days (since 2000-01-01) on which a prune may happen
    dT: int
    n_known_new: int


_PREP_CACHE: dict[tuple, Any] = {}
_EPOCH = datetime(2000, 1, 1).date()


def prepare(ctx: Context, t: datetime, T: datetime, *, prune_window: int = 19, weeks: int = 8,
            q: float = 0.02, tau: float = 1.0) -> Prepared:
    key = (id(ctx), t, T, prune_window, weeks, q, tau)
    if key in _PREP_CACHE:
        return _PREP_CACHE[key]
    a = anchor(ctx, t)
    mem_a = track(ctx, a["state_ts"], t, prune_window)
    mem = {f: dict(m) for f, m in mem_a.items()}
    new = ctx.versions.filter((pl.col("known_at") > a["state_ts"]) & (pl.col("known_at") <= t)).sort("known_at")
    for r in new.iter_rows(named=True):                  # known entries in (anchor, t]
        cur = mem.get(r["fam"])
        if cur is not None and cur["id"] != r["id"] and cur["end"] and r["end"] and r["end"] < cur["end"]:
            continue
        mem[r["fam"]] = dict(r)
    polls_t = asof(ctx.versions, t)
    last_day = et_date(t) - timedelta(days=1)
    fit = kalman_house(polls_t, last_day, q=q, tau=tau)
    rel = ctx.rel.filter(pl.col("rel") <= t)
    recent = set(rel.filter(pl.col("rel") > t - timedelta(days=60))["fam"].to_list())
    lastp = {r["fam"]: r for r in latest_per_family(polls_t).iter_rows(named=True)}
    fams = sorted(set(mem) | (recent & set(lastp)))
    by_fam = {f: g["rel"].dt.replace_time_zone(None).to_numpy() for (f,), g in rel.group_by(["fam"])}
    cur_val, cur_start, ref_val, p_arr, n, trk = [], [], [], [], [], []
    for f in fams:
        cur = mem.get(f)
        ref = cur if cur is not None else lastp[f]
        cur_val.append(cur["approve"] if cur is not None else np.nan)
        cur_start.append((cur["start"] - _EPOCH).days if cur is not None and cur["start"] else np.nan)
        ref_val.append(ref["approve"])
        times = by_fam.get(f, np.array([], dtype="datetime64[us]"))
        p_arr.append(arrival_prob(times, t, T, weeks) if len(times) else 0.0)
        n.append(ref["n"] or 1000)
        trk.append(f == "Rasmussen Reports")
    d0, dT = et_date(a["state_ts"]), et_date(T)
    out = Prepared(anchor=a, mem_a=mem_a, fams=fams, cur_val=np.array(cur_val, float), cur_start=np.array(cur_start, float),
                   ref_val=np.array(ref_val, float), p_arrive=np.array(p_arr), n=np.array(n, float), is_tracker=np.array(trk),
                   fit=fit, horizon_days=(dT - last_day).days,
                   days=[(d0 + timedelta(days=k) - _EPOCH).days for k in range(1, (dT - d0).days + 1)],
                   dT=(dT - _EPOCH).days, n_known_new=new.height)
    _PREP_CACHE[key] = out
    return out


def forecast_mechanical(ctx: Context, t: datetime, T: datetime, prm: Params,
                        rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Monte Carlo of the displayed RCP average at ``T`` from what was public at ``t``.

    1. Anchor: newest displayed value public at ``t``; its membership is reconstructed (exact
       when the anchor is a Wayback capture).
    2. Known entries: polls entered between the anchor and ``t`` replace their family's member.
    3. Future entries: each family enters a new poll in (t, T] with its weekly-cadence
       probability; the value is latent approval at T + house effect + noise (Kalman, polls known
       at t), or the family's last value + noise; Rasmussen's rolling tracker moves +-1.
    4. Pruning: on each ET day after the anchor, with probability ``prune_rate`` RCP drops
       members whose field start is more than ``prune_window`` days old.
    5. Mean, plus the anchor's level offset, residual noise, RCP rounding."""
    pp = prepare(ctx, t, T, prune_window=prm.prune_window, weeks=prm.weeks, q=prm.q, tau=prm.tau)
    S, F = prm.sims, len(pp.fams)
    level = _start_level(pp.anchor, pp.mem_a, rng, S)
    base_mean = float(np.mean([x["approve"] for x in pp.mem_a.values()]))
    arrive = rng.random((F, S)) < pp.p_arrive[:, None]
    fit = pp.fit
    if prm.value_mode == "kalman":
        xT = rng.normal(fit.x, np.sqrt(fit.var_x + prm.q * pp.horizon_days), S)
        h = np.array([fit.house.get(f, 0.0) for f in pp.fams])
        vh = np.array([fit.var_house.get(f, 9.0) for f in pp.fams])
        pr = np.clip((fit.x + h) / 100, 0.05, 0.95)
        sd = np.sqrt(1e4 * pr * (1 - pr) / pp.n + prm.tau ** 2 + vh)
        new_v = xT[None, :] + h[:, None] + rng.normal(0, 1, (F, S)) * sd[:, None]
    else:
        new_v = pp.ref_val[:, None] + rng.normal(0, 2.5, (F, S))
    new_v = np.where(pp.is_tracker[:, None], pp.ref_val[:, None] + rng.normal(0, 0.9, (F, S)), new_v)
    new_v = np.round(new_v)
    vals = np.where(arrive, new_v, pp.cur_val[:, None])
    start = np.where(arrive, float(pp.dT - 3), pp.cur_start[:, None])
    for day in pp.days:                                   # pruning, day by day
        drop = (rng.random(S) < prm.prune_rate)[None, :] & (start < day - prm.prune_window)
        vals = np.where(drop, np.nan, vals)
        start = np.where(drop, np.nan, start)
    with np.errstate(invalid="ignore"):
        mean_T = np.nanmean(vals, axis=0)
    out = level + (mean_T - base_mean) + rng.normal(0, prm.sigma_res, S)
    a = pp.anchor
    return _pmf(_round_tenths(out)), {"anchor_source": a["source"], "anchor_age_h": (t - a["state_ts"]).total_seconds() / 3600,
                                      "anchor_value": a["value"], "n_known_new": pp.n_known_new, "n_fams": F}


def forecast_rw(ctx: Context, t: datetime, T: datetime, sigma_day: float, rng: np.random.Generator,
                sims: int = 4000, known: bool = False, prm: Params | None = None) -> np.ndarray:
    """Baseline: the displayed value public at ``t`` (uniform within its rounding interval),
    plus a Gaussian random walk with variance ``sigma_day^2`` per day from the anchor's state time
    to ``T``. With ``known=True`` the walk starts at ``t`` from the anchor shifted by the
    deterministic effect of polls entered between the anchor and ``t`` (no other mechanics)."""
    a = anchor(ctx, t)
    level = a["value"] / 10 + rng.uniform(-0.05, 0.05, sims)
    t0 = a["state_ts"]
    if known:
        prm = prm or Params()
        key = ("rwk", id(ctx), t, prm.prune_window)
        if key not in _PREP_CACHE:
            _PREP_CACHE[key] = (track(ctx, a["state_ts"], t, prm.prune_window), track(ctx, t, t, prm.prune_window))
        mem_a, mem_t = _PREP_CACHE[key]
        level = _start_level(a, mem_a, rng, sims) + (np.mean([x["approve"] for x in mem_t.values()])
                                                      - np.mean([x["approve"] for x in mem_a.values()]))
        t0 = t
    h = max((T - t0).total_seconds() / 86400, 0.0)
    out = level + rng.normal(0, sigma_day * np.sqrt(h), sims)
    return _pmf(_round_tenths(out))


def leg_prob(pmf: np.ndarray, lo: int, hi: int) -> float:
    return float(pmf[(GRID >= lo) & (GRID <= hi)].sum())


# ---- evaluation ------------------------------------------------------------------------------
HOLDOUT = datetime(2026, 10, 1, tzinfo=UTC)
KELLY_FRACTION = 0.25
BANKROLL = 100.0


def event_loglik(pmfs: dict[tuple, np.ndarray], dec: pl.DataFrame) -> pl.DataFrame:
    """Log probability of the settled bracket per decision (event, t)."""
    rows = []
    for r in dec.iter_rows(named=True):
        q = leg_prob(pmfs[(r["event"], r["t"])], r["wlo"], r["whi"])
        rows.append({"event": r["event"], "t": r["t"], "series": r["series"], "lead_h": r["lead_h"],
                     "split": "holdout" if r["T"] >= HOLDOUT else "dev", "ll": float(np.log(max(q, 1e-3)))})
    return pl.DataFrame(rows)


def market_loglik(q: pl.DataFrame) -> pl.DataFrame:
    """Same for the market: mids normalised across the legs quoted at t."""
    ev = (q.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
          .group_by("event", "t", "series", "lead_h", "T")
          .agg(pl.col("mid").sum().alias("msum"), (pl.col("mid") * pl.col("y")).sum().alias("mwin"), pl.len().alias("legs_q")))
    return ev.with_columns(pl.when(pl.col("T") >= HOLDOUT).then(pl.lit("holdout")).otherwise(pl.lit("dev")).alias("split"),
                           (pl.col("mwin") / pl.col("msum")).clip(1e-3, 1).log().alias("ll"))


def ladder_backtest(q: pl.DataFrame, pmfs: dict[tuple, np.ndarray], theta: float) -> pl.DataFrame:
    """Per decision (event, t): (a) the expected-log-growth ladder (quarter Kelly on a $100
    bankroll, fractional contracts, taker at ask / 1-bid plus fees) and (b) the single best
    instrument (largest edge >= theta, one contract). Held to settlement."""
    from albert.research.fundamental import taker_fee
    rows = []
    for (ev, t), part in q.group_by(["event", "t"], maintain_order=True):
        legs = part.select("lo", "hi", "bid", "ask", "y", "ticker").to_dicts()
        pmf = pmfs[(ev, t)]
        picks = ladder_kelly(GRID, pmf, legs, theta=theta)
        staked = pnl = 0.0
        for pk in picks:
            dollars = KELLY_FRACTION * pk["stake"] * BANKROLL
            n = dollars / pk["cost"]
            win = pk["y"] if pk["side"] == "yes" else 1 - pk["y"]
            staked += dollars
            pnl += n * (win - pk["cost"])
        best = None
        for leg in legs:
            pr = leg_prob(pmf, leg["lo"], leg["hi"])
            for side, price, prob, win in (("yes", leg["ask"], pr, leg["y"]), ("no", 1 - leg["bid"], 1 - pr, 1 - leg["y"])):
                if not 0 < price < 1:
                    continue
                edge = prob - price - taker_fee(price)
                if edge >= theta and (best is None or edge > best[0]):
                    best = (edge, win - price - taker_fee(price))
        r0 = part.row(0, named=True)
        rows.append({"event": ev, "t": t, "series": r0["series"], "lead_h": r0["lead_h"],
                     "split": "holdout" if r0["T"] >= HOLDOUT else "dev", "ladder_n": len(picks),
                     "ladder_staked": staked, "ladder_pnl": pnl, "best_pnl": None if best is None else best[1]})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- reproduction checks -----------------------------------------------------------------------
def reproduction_checks(ctx: Context, snaps: list[Snapshot], sv: pl.DataFrame, prune_window: int = 19) -> pl.DataFrame:
    """(1) At every Wayback capture: is the displayed average the plain mean of the polls shown in
    it, rounded to 0.1? (2) Nowcast at each Kalshi settlement instant T from what was public at T
    (newest displayed value + polls entered since, membership tracked): is it in the settled
    bracket? (3) The same with a fixed rule (latest poll per family with field end in the last
    21 days, no anchor)."""
    caps = [s for s in snaps if s.in_avg and s.avg_approve is not None]
    exact = [rcp_mean([p["approve"] for p in s.in_avg]) == s.avg_approve for s in caps]
    rows = [{"check": "capture avg == mean(in-average polls)", "split": "all", "n": len(exact), "rate": float(np.mean(exact))}]
    elig = {f for c in ctx.captures for f in (m["fam"] for m in c.members)}
    for split, part in (("dev", sv.filter(pl.col("T") < HOLDOUT)), ("holdout", sv.filter(pl.col("T") >= HOLDOUT))):
        ok_t, ok_r = [], []
        for r in part.iter_rows(named=True):
            pp = prepare(ctx, r["T"], r["T"], prune_window=prune_window)
            m = np.nanmean(pp.cur_val) - np.mean([x["approve"] for x in pp.mem_a.values()])
            a = pp.anchor
            v = round(rcp_mean([x["approve"] for x in pp.mem_a.values()]) * 10)
            lvl = np.mean([x["approve"] for x in pp.mem_a.values()]) if v == a["value"] else a["value"] / 10
            now = int(_round_tenths(np.array([lvl + m]))[0])
            ok_t.append(r["lo"] <= now <= r["hi"])
            mem = members_at(asof(ctx.versions, r["T"]), r["T"], elig, Rule(window=21))
            rule = round(rcp_mean(mem["approve"].to_list()) * 10)
            ok_r.append(r["lo"] <= rule <= r["hi"])
        rows.append({"check": "nowcast at T in settled bracket (anchor + entries)", "split": split, "n": len(ok_t), "rate": float(np.mean(ok_t))})
        rows.append({"check": "fixed 21-day rule at T in settled bracket", "split": split, "n": len(ok_r), "rate": float(np.mean(ok_r))})
    return pl.DataFrame(rows)


# ---- driver ----------------------------------------------------------------------------------
RW_SIGMAS = (0.1, 0.15, 0.2, 0.3, 0.4)
MECH_GRID = [{"value_mode": vm, "prune_window": w, "prune_rate": pr, "sigma_res": sr}
             for vm in ("kalman", "last") for w in (17, 19, 21) for pr in (0.15, 0.35, 0.6) for sr in (0.03, 0.08)]


def _pmfs(fn, dec: pl.DataFrame) -> dict[tuple, np.ndarray]:
    return {(r["event"], r["t"]): fn(r) for r in dec.iter_rows(named=True)}


def _ll(pmfs: dict[tuple, np.ndarray], dec: pl.DataFrame) -> float:
    return float(event_loglik(pmfs, dec)["ll"].mean())


def _p_frame(q: pl.DataFrame, pmfs: dict[tuple, np.ndarray]) -> pl.DataFrame:
    return q.with_columns(pl.struct("event", "t", "lo", "hi").map_elements(
        lambda r: leg_prob(pmfs[(r["event"], r["t"])], r["lo"], r["hi"]), return_dtype=pl.Float64).alias("p"))


def run(fetch: bool = False, sims: int = 4000, seed: int = 7) -> Path:
    """Full study: data -> dev fit -> frozen models -> holdout once -> report."""
    from albert.config import Settings
    from albert.reports import md_table
    from albert.research.fundamental import (
        attach_quotes,
        evaluate,
        fetch_candles,
        report,
        settled_markets,
    )

    s = Settings()
    cache = s.data_dir / "cache" / "approval"
    if fetch:
        fetch_rasmussen(cache)
    snaps = load_snapshots(cache, "20260101", "20261105", fetch=fetch)
    markets = settled_markets(s, SERIES)
    legs = kalshi_legs(markets)
    sv = settled_values(legs)
    ctx = build_context(snaps, rasmussen_history(cache), sv)
    fr = decision_frame(legs)
    # Keep each decision's split equal to its event's split (an Oct 1 event decided on Sep 30
    # would otherwise put an October outcome into the dev fit).
    fr = fr.filter(~((pl.col("t") < HOLDOUT) & (pl.col("T") >= HOLDOUT)))
    dec = (fr.select("event", "series", "T", "t", "lead_h").unique()
           .join(sv.select("event", pl.col("lo").alias("wlo"), pl.col("hi").alias("whi")), on="event").sort("t"))
    weekly = markets.filter(pl.col("series_ticker") == "KXAPRPOTUS")
    fetch_candles(s, weekly, "approval", lookback=timedelta(days=8), workers=2)
    candles = fetch_candles(s, markets, "approval", lookback=timedelta(days=2), workers=2)
    q = attach_quotes(fr.with_columns(pl.lit(0.5).alias("p")), candles).drop("p")
    dev = dec.filter(pl.col("T") < HOLDOUT)
    rng = np.random.default_rng(seed)

    # -- fit every variant on dev only
    fits: list[dict[str, Any]] = []
    for sig in RW_SIGMAS:
        fits.append({"model": "rw", "sigma": sig, "dev_ll": _ll(_pmfs(lambda r, s_=sig: forecast_rw(
            ctx, r["t"], r["T"], s_, rng, sims), dev), dev)})
        fits.append({"model": "rw_known", "sigma": sig, "dev_ll": _ll(_pmfs(lambda r, s_=sig: forecast_rw(
            ctx, r["t"], r["T"], s_, rng, sims, known=True), dev), dev)})
    for g in MECH_GRID:
        prm = Params(**g, sims=min(sims, 2000))
        fits.append({"model": f"mech_{g['value_mode']}", **g, "dev_ll": _ll(_pmfs(
            lambda r, p_=prm: forecast_mechanical(ctx, r["t"], r["T"], p_, rng)[0], dev), dev)})
    fit_df = pl.DataFrame(fits, infer_schema_length=None).sort("dev_ll", descending=True)
    best = {m: fit_df.filter(pl.col("model") == m).row(0, named=True) for m in fit_df["model"].unique().to_list()}

    def make(m: str):
        b = best[m]
        if m.startswith("mech"):
            prm = Params(value_mode=b["value_mode"], prune_window=b["prune_window"], prune_rate=b["prune_rate"],
                         sigma_res=b["sigma_res"], sims=sims)
            return lambda r: forecast_mechanical(ctx, r["t"], r["T"], prm, rng)[0]
        return lambda r: forecast_rw(ctx, r["t"], r["T"], b["sigma"], rng, sims, known=(m == "rw_known"))

    chosen = fit_df.row(0, named=True)["model"]
    pmfs = {m: _pmfs(make(m), dec) for m in best}           # frozen; holdout scored once

    # -- scores
    mkt = market_loglik(q.join(dec.select("event", "t"), on=["event", "t"]))
    lls = [event_loglik(p, dec).with_columns(pl.lit(m).alias("model")) for m, p in pmfs.items()]
    lls.append(mkt.select("event", "t", "series", "lead_h", "split", "ll").with_columns(pl.lit("market").alias("model")))
    ll_all = pl.concat(lls, how="diagonal_relaxed")
    ll_tab = (ll_all.group_by("model", "split").agg(pl.len().alias("decisions"), pl.col("event").n_unique().alias("events"),
                                                    pl.col("ll").mean().round(4).alias("mean_loglik"))
              .sort("split", "mean_loglik", descending=[False, True]))
    ll_lead = (ll_all.group_by("model", "split", "series", "lead_h").agg(pl.len().alias("n"), pl.col("ll").mean().round(3).alias("ll"))
               .pivot(on="model", index=["split", "series", "lead_h", "n"], values="ll").sort("split", "series", "lead_h"))
    results = {m: evaluate(_p_frame(q, p).filter(pl.col("p").is_not_null()), HOLDOUT) for m, p in pmfs.items()}
    res = results[chosen]
    seg_rows = []
    for (series, lead), part in _p_frame(q, pmfs[chosen]).group_by(["series", "lead_h"], maintain_order=True):
        r = evaluate(part, HOLDOUT, min_trades=10)
        for sc in r["scores"]:
            seg_rows.append({"series": series, "lead_h": lead, **{k: sc[k] for k in ("split", "n", "events", "brier_model", "brier_mid")}})
    lad_by_theta = {th: ladder_backtest(q.join(dec.select("event", "t"), on=["event", "t"]), pmfs[chosen], th)
                    for th in (0.0, 0.02, 0.05, 0.08)}
    th_star = max(lad_by_theta, key=lambda th: float(lad_by_theta[th].filter(pl.col("split") == "dev")["ladder_pnl"].sum()))
    lad = lad_by_theta[th_star]
    from albert.research.fundamental import clustered_mean
    lad_rows = []
    for split in ("dev", "holdout"):
        part = lad.filter(pl.col("split") == split)
        traded = part.filter(pl.col("ladder_n") > 0)
        _mu, tt, k = clustered_mean(traded["ladder_pnl"].to_numpy(), traded["event"].to_numpy()) if traded.height else (0, None, 0)
        bp = part.filter(pl.col("best_pnl").is_not_null())
        mu2, tt2, _k2 = clustered_mean(bp["best_pnl"].to_numpy(), bp["event"].to_numpy()) if bp.height else (0, None, 0)
        lad_rows.append({"split": split, "theta": th_star, "decisions": part.height, "ladder_decisions_traded": traded.height,
                         "ladder_staked_$": round(float(traded["ladder_staked"].sum()), 2),
                         "ladder_pnl_$": round(float(traded["ladder_pnl"].sum()), 2),
                         "ladder_pnl_per_$": round(float(traded["ladder_pnl"].sum() / max(traded["ladder_staked"].sum(), 1e-9)), 4),
                         "ladder_t": None if tt is None else round(tt, 2), "ladder_events": k,
                         "best1_trades": bp.height, "best1_pnl_c": round(mu2 * 100, 2) if bp.height else None,
                         "best1_t": None if tt2 is None else round(tt2, 2)})

    extra = {"chosen_model": chosen, "chosen_params": best[chosen], "holdout_from": str(HOLDOUT),
             "events": {"dev": dev["event"].n_unique(), "holdout": dec.filter(pl.col("T") >= HOLDOUT)["event"].n_unique()},
             "decisions": dec.height, "leads_h": LEADS_H, "wayback_captures": len(snaps),
             "anchor_sources_at_decisions": dict(zip(*np.unique([anchor(ctx, r["t"])["source"] for r in dec.iter_rows(named=True)],
                                                                return_counts=True), strict=True)),
             "kelly_fraction": KELLY_FRACTION, "bankroll": BANKROLL}
    extra["anchor_sources_at_decisions"] = {str(k): int(v) for k, v in extra["anchor_sources_at_decisions"].items()}
    # -- as-of robustness: every poll treated as entered 6 h later than RCP's stamp
    ctx_late = build_context(snaps, rasmussen_history(cache), sv, delay=timedelta(hours=6))
    b = best[chosen]
    late_fn = ((lambda r: forecast_mechanical(ctx_late, r["t"], r["T"], Params(
        value_mode=b["value_mode"], prune_window=b["prune_window"], prune_rate=b["prune_rate"],
        sigma_res=b["sigma_res"], sims=sims), rng)[0]) if chosen.startswith("mech")
        else (lambda r: forecast_rw(ctx_late, r["t"], r["T"], b["sigma"], rng, sims, known=(chosen == "rw_known"))))
    late = event_loglik(_pmfs(late_fn, dec), dec).group_by("split").agg(pl.col("ll").mean().round(4).alias("mean_loglik_delay6h"))
    repro = reproduction_checks(ctx, snaps, sv, best[chosen].get("prune_window") or 19)
    path = report(s, "approval", res, extra)
    with path.open("a") as fh:
        fh.write("\n## Reproducing RCP's number\n\n" + md_table(repro, 10) + "\n")
        fh.write("\n## As-of robustness (chosen model, entries delayed 6 h)\n\n" + md_table(late, 5) + "\n")
        fh.write("\n## Dev fit of every variant (mean log prob of the settled bracket per decision)\n\n" + md_table(fit_df, 40) + "\n")
        fh.write("\n## Log score of the settled bracket: models vs market (mids normalised over the ladder)\n\n" + md_table(ll_tab, 40) + "\n")
        fh.write("\n## Log score by series and lead (hours before settlement)\n\n" + md_table(ll_lead, 40) + "\n")
        fh.write(f"\n## Chosen model ({chosen}) by series x lead: Brier per leg vs mid\n\n" + md_table(pl.DataFrame(seg_rows), 40) + "\n")
        for m, r in results.items():
            fh.write(f"\n## Variant {m}: blend and scores\n\n```json\n{json.dumps({'scores': r['scores'], 'blend': r.get('blend'), 'theta': r.get('theta_chosen_on_dev')}, default=str, indent=1)}\n```\n")
        fh.write(f"\n## Ladder (quarter-Kelly expected-log-growth portfolio over the ladder, theta {th_star} chosen on dev) vs single best leg (1 contract)\n\n"
                 + md_table(pl.DataFrame(lad_rows), 10) + "\n")
    ll_all.write_parquet(path.parent / "loglik.parquet")
    lad.write_parquet(path.parent / "ladder.parquet")
    _p_frame(q, pmfs[chosen]).write_parquet(path.parent / "predictions.parquet")
    return path


if __name__ == "__main__":
    import sys
    print(run(fetch="--fetch" in sys.argv))
