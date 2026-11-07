"""Culture / tech fundamental models: can a public-data model price Kalshi entertainment and
AI-usage markets better than the market, out of sample and after fees?

Feasibility triage (2026-11-05). A backtest is honest only if the data source's state *as of* a
past decision time can be rebuilt. Checked per candidate series (events = settled Kalshi events
Aug 15 - Nov 1 2026):

| Series (events) | Settlement source | As-of reconstruction | Verdict |
|---|---|---|---|
| KX{ANTH,OPEN,GOOG,DEEP}VREQ, KX{ANTH,OPEN,MOON}VSPEND (20 each) | Vercel AI Gateway labs leaderboard, daily share | the public page embeds every lab's daily request/token/cost share since 2025-10-01; a day's value appears the next day; Wayback snapshots check lag and revisions | **feasible** (study A) |
| KXRT (25), KXRTTV (1) | Rotten Tomatoes Tomatometer at a fixed time | Wayback snapshots of the movie page carry liked/not-liked counts (`criticsScore`) at the snapshot time; RT's review API needs a browser session (401), so timestamped review lists are not used | **feasible** (study B) |
| KXOPENSOURCESHARE (25) | Vercel models page, open-weights share | page embeds only the top-10 models per day; the open/closed split comes from a robots-disallowed `/api/` call | dropped (partial data) |
| KXYTVIEWSW/D/HIGH (137/119/32), KXYTTOPVIDEO* (72 x4) | YouTube Charts | the data API (`/youtubei/`) is disallowed by robots.txt; kworb.net keeps only ~12 days of per-video history and counts official channels only (a different quantity) | dropped (no as-of source) |
| KXALBUMEQUIV/STREAMS, KXPUREALBUMS (43/36/38), KXARTISTSTREAMSY | Luminate | tracking-week data is not public before the chart; mid-week industry projections are not archived reliably | dropped |
| KXNETFLIXRANK* (10-11 each) | Netflix Tudum weekly top 10 | weekly lists are archived, but the intra-week signal (daily top 10) is not on an as-of public archive we can use; 11 weeks | dropped (sample, as-of) |
| KXTOPSONG/KXTOPALBUM/Billboard runner-ups (10-11) | Billboard | 10 weeks; intra-week streaming data not archived | dropped (sample) |
| KX{ANTH,GOOG,DEEP,OPEN}SHARE, KXTOKENUSE (10 weekly each) | OpenRouter rankings | Wayback captures exist several times a week, but the chart data is fetched client-side, so captures hold no numbers; 10 weeks | dropped (no as-of data) |
| KXLLM1 (9) | LMArena leaderboard | 9 weeks, the leader rarely changes | dropped (sample) |
| KXVMA/KXEMMY/KXDTEMMYS (23/17/39), Big Brother (5), DWTS (7) | award shows / TV votes | one-off categories, no repeatable public model input | dropped |
| KXMC (14) | Metacritic Metascore | weighted (secret weights) average; Wayback-only | not built (sample) |

**Study A (Vercel AI Gateway daily shares).** Each event asks whether a lab's share of AI Gateway
requests (``VREQ``) or cost (``VSPEND``) on a UTC day ``d`` is above a ladder of strikes. The labs
page (https://vercel.com/ai-gateway/leaderboards/labs) embeds the full daily share history of every
lab. Decision times: ``d 16:00 UTC`` (the day being measured, ~12 h before close, latest known day
``d-1``) and ``d-1 16:00 UTC`` (latest known day ``d-2``). Only values for days ``<= latest known``
enter the model. The latest-known day is set from the publication lag measured on Wayback snapshots
of the same pages (a snapshot at time T shows day T-1 from ~05 UTC), and the values used are those
on today's page: revisions measured against snapshots are small (see report).

**Study B (Rotten Tomatoes).** Each event asks whether the Tomatometer ``round(100·liked/total)``
is above K at a fixed time T. State at a Wayback snapshot time t: liked and total critic reviews.
Model: the reviews still to come by T (growth fitted on dev movies) have a fresh rate drawn from a
Beta centred on the current rate (plus a drift), giving a beta-binomial distribution of the final
score.

**Results (2026-11-05, dev < 2026-10-01 <= holdout; reports research-fundamental-culture-*).**
No edge. Vercel at ``d 16:00 UTC``: the market beats the model wherever the spread is <= 25c
(holdout Brier 0.050 vs 0.121; blend c = 0.06, t 0.3) and the taker loses 12.8c/contract
(298 trades, 87 events, t -5.3); the blend's "information" (c 0.66, t 3.2) comes only from
>25c-spread rows where the mid is not a price. Traders evidently know more about day ``d`` by
16:00 UTC than the published ``d-1`` value. At ``(d-1) 20:00 UTC`` the taker is flat (+0.5c,
t 0.2). Rotten Tomatoes: dev taker -5.4c (17 events, t -3.2), holdout +3.8c (11 events, t 2.0)
with one event (Wildwood at 100%) giving 63% of the holdout PnL; pooled over all 25 events
+1.2c (t 0.7), without Wildwood -0.6c. 25 events cannot separate skill from noise.
"""

import gzip
import json
import math
import re
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import polars as pl
from scipy import stats
from scipy.special import betaln, gammaln

NAME = "culture"
UA = "albert-research/0.2 (culture study; polite, cached)"
VERCEL_LABS = "https://vercel.com/ai-gateway/leaderboards/labs"
VERCEL_LAB_CODES = {"ANTH": "anthropic", "OPEN": "openai", "GOOG": "google", "DEEP": "deepseek",
                    "MOON": "moonshotai", "ZAI": "zai"}
VERCEL_SERIES = ["KXANTHVREQ", "KXOPENVREQ", "KXGOOGVREQ", "KXDEEPVREQ", "KXZAIVREQ",
                 "KXANTHVSPEND", "KXOPENVSPEND", "KXMOONVSPEND", "KXGOOGVSPEND", "KXZAIVSPEND"]
RT_SERIES = ["KXRT"]


# ---- polite cached fetching ------------------------------------------------------------------
class Fetcher:
    """GET with an on-disk cache (one file per URL, never overwritten) and a minimum interval."""

    def __init__(self, cache_dir: Path, min_interval: float = 1.0) -> None:
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self._last = 0.0
        self._http = httpx.Client(timeout=60, headers={"User-Agent": UA}, follow_redirects=True)

    def path(self, url: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", url.split("://", 1)[-1])
        if len(safe) > 180:
            import hashlib
            safe = safe[:120] + "_" + hashlib.sha1(url.encode()).hexdigest()[:16]
        return self.dir / f"{safe}.bin"

    def get(self, url: str, *, retries: int = 4) -> bytes | None:
        p = self.path(url)
        if p.exists():
            return p.read_bytes()
        for attempt in range(retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self._http.get(url)
            except httpx.HTTPError:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 200:
                tmp = p.with_suffix(".tmp")
                tmp.write_bytes(r.content)
                tmp.replace(p)
                return r.content
            if r.status_code in (404, 403, 401):
                return None
            time.sleep(5 * (attempt + 1))
        return None


def decode(raw: bytes | None) -> str:
    if raw is None:
        return ""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace")


def wayback_cdx(f: Fetcher, url: str, start: str, end: str, *, prefix: bool = False) -> list[dict[str, str]]:
    """Wayback CDX rows (timestamp, original, length) of 200 captures of ``url`` in [start, end]."""
    q = (f"https://web.archive.org/cdx/search/cdx?url={url}&from={start}&to={end}&output=json"
         f"&fl=timestamp,original,length&filter=statuscode:200" + ("&matchType=prefix" if prefix else ""))
    txt = decode(f.get(q))
    try:
        rows = json.loads(txt) if txt else []
    except json.JSONDecodeError:
        return []
    return [dict(zip(rows[0], r, strict=False)) for r in rows[1:]] if rows else []


def wayback_page(f: Fetcher, timestamp: str, url: str) -> str:
    return decode(f.get(f"https://web.archive.org/web/{timestamp}id_/{url}"))


def wb_time(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=UTC)


# ---- Vercel AI Gateway -----------------------------------------------------------------------
def next_payload(html: str) -> str:
    """Concatenate the Next.js flight chunks (``self.__next_f.push([1, "..."])``) of a page."""
    chunks = re.findall(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)', html)
    return "".join(json.loads('"' + c + '"') for c in chunks)


def _bracket_array(s: str, start: int) -> Any:
    depth = 0
    for k in range(start, len(s)):
        ch = s[k]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return json.loads(s[start:k + 1])
    raise ValueError("unbalanced array")


def vercel_rawdata(html: str) -> pl.DataFrame:
    """Daily shares embedded in a Vercel leaderboard page: day (date), metric, name, v (percent)."""
    payload = next_payload(html)
    m = re.search(r'"rawData":\[', payload)
    if not m:
        return pl.DataFrame(schema={"day": pl.Date, "metric": pl.Utf8, "name": pl.Utf8, "v": pl.Float64})
    arr = _bracket_array(payload, m.end() - 1)
    rows = [{"day": date.fromisoformat(a["day"][:10]), "metric": a["metric"], "name": n, "v": float(v)}
            for a in arr for n, v in a["chef_values"]]
    return pl.DataFrame(rows)


def vercel_markets(s) -> pl.DataFrame:
    """Settled Vercel lab-share markets with lab, metric, measured day and strike."""
    from albert.research.fundamental import settled_markets

    m = settled_markets(s, VERCEL_SERIES)
    code = pl.col("series_ticker").str.extract(r"^KX([A-Z]+?)V(?:REQ|SPEND)$", 1)
    return (m.with_columns(
        code.replace_strict(VERCEL_LAB_CODES, default=None).alias("lab"),
        pl.when(pl.col("series_ticker").str.ends_with("VREQ")).then(pl.lit("requests")).otherwise(pl.lit("cost")).alias("metric"),
        pl.col("event_ticker").str.extract(r"-(\d{2}[A-Z]{3}\d{2})$").str.to_date("%d%b%y").alias("day"),
        pl.col("yes_sub_title").str.extract(r"Above ([\d.]+)%").cast(pl.Float64).alias("K"))
        .filter(pl.col("lab").is_not_null() & pl.col("K").is_not_null()))


def fetch_candles_batched(s, markets: pl.DataFrame, name: str, *, lookback: timedelta,
                          max_candles: int = 9000, max_batch: int = 100) -> pl.DataFrame:
    """Hourly candles like :func:`albert.research.fundamental.fetch_candles` (same cache file and
    row format), but markets settled after Kalshi's historical cutoff are fetched with the batch
    endpoint (one request per chunk of markets whose joint window stays under the 10,000-candle
    cap); older markets fall back to the per-market harness. Fewer requests on a shared limit."""
    from albert.http import CachedClient
    from albert.research.fundamental import fetch_candles
    from albert.venues import kalshi as k

    path = s.data_dir / "cache" / f"fundamental_{name}_candles.parquet"
    have = pl.read_parquet(path) if path.exists() else None
    done = set(have["ticker"].unique()) if have is not None and have.height else set()
    todo = markets.filter(~pl.col("ticker").is_in(list(done)))
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.2))
    cutoff = api.cutoff().get("market_settled_ts")
    hist = todo.filter(pl.col("close_time") < cutoff) if cutoff else todo.head(0)
    live = todo.filter(pl.col("close_time") >= cutoff) if cutoff else todo

    def window(m: dict[str, Any]) -> tuple[datetime, datetime]:
        hi = m["close_time"]
        return max(m["open_time"] or hi - lookback, hi - lookback), hi

    ms = sorted(live.select("ticker", "open_time", "close_time").iter_rows(named=True), key=lambda m: (window(m)[0], m["ticker"]))
    chunks, cur = [], []
    for m in ms:
        trial = cur + [m]
        lo, hi = min(window(x)[0] for x in trial), max(window(x)[1] for x in trial)
        if cur and (len(trial) > max_batch or ((hi - lo).total_seconds() / 3600 + 1) * len(trial) > max_candles):
            chunks.append(cur)
            cur = [m]
        else:
            cur = trial
    if cur:
        chunks.append(cur)
    rows: list[dict[str, Any]] = []
    for i, ch in enumerate(chunks, 1):
        lo, hi = min(window(x)[0] for x in ch), max(window(x)[1] for x in ch)
        for attempt in range(6):
            try:
                page = api.candles_batch([x["ticker"] for x in ch], lo, hi, 60)
                rows.extend(r for mk in page.payload.get("markets") or []
                            for r in k.candle_rows(mk, mk.get("market_ticker"), 60))
                break
            except RuntimeError as error:
                if "429" in str(error):
                    time.sleep(5 * (attempt + 1))
                    continue
                print(f"skip batch at {ch[0]['ticker']}: {str(error)[:160]}", flush=True)
                break
        if i % 5 == 0:
            print(f"  batch {i}/{len(chunks)}", flush=True)
    new = pl.DataFrame(rows, schema=k.CANDLE_SCHEMA) if rows else None
    frames = [f for f in (have, new) if f is not None and f.height]
    out = pl.concat(frames, how="diagonal_relaxed").unique(["ticker", "end_ts", "period_minutes"]) if frames else pl.DataFrame()
    if out.height:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        out.write_parquet(tmp)
        tmp.replace(path)
    if hist.height:
        out = fetch_candles(s, hist, name, period_minutes=60, lookback=lookback, workers=2)
    return out


# ---- Rotten Tomatoes -------------------------------------------------------------------------
def rt_scorecard(html: str) -> dict[str, Any] | None:
    """Critic counts from a Rotten Tomatoes movie page: liked, not liked, total, displayed score."""
    m = re.search(r'"criticsScore":(\{[^{}]*"likedCount"[^{}]*\})', html)
    if not m:
        return None
    d = json.loads(m.group(1))
    liked, nl = d.get("likedCount"), d.get("notLikedCount")
    if liked is None or nl is None:
        return None
    score = d.get("score")
    return {"liked": int(liked), "not_liked": int(nl), "total": int(liked) + int(nl),
            "score": int(score) if score not in (None, "") else None,
            "review_count": d.get("reviewCount")}


def rt_round(liked: np.ndarray | int, total: np.ndarray | int) -> np.ndarray:
    """Tomatometer as RT displays it: 100·liked/total rounded half up (43/128 = 33.6 -> 34)."""
    return np.floor(100.0 * np.asarray(liked) / np.maximum(np.asarray(total), 1) + 0.5)


def now_utc() -> datetime:
    return datetime.now(UTC)



RT_SLUGS = {"KXRT-BUD": "buddy_2026", "KXRT-COY": "coyote_vs_acme", "KXRT-DOG": "the_dog_stars",
            "KXRT-ITE": "it_ends", "KXRT-BYA": "by_any_means_2026", "KXRT-ONS": "onslaught",
            "KXRT-PRA": "practical_magic_2", "KXRT-UPR": "the_uprising_2026", "KXRT-RUN": "runner_2026",
            "KXRT-RES": "resident_evil_2026", "KXRT-WEI": "the_weight_2026", "KXRT-PRI": "primetime",
            "KXRT-FOR": "forgotten_island", "KXRT-HEA": "heart_of_the_beast", "KXRT-VER": "verity",
            "KXRT-DIG": "digger_2026", "KXRT-SOC": "the_social_reckoning", "KXRT-OTH": "other_mommy",
            "KXRT-WHA": "whalefall", "KXRT-STRA": "street_fighter_2026",
            "KXRT-SENS": "sense_and_sensibility_2026", "KXRT-YOUC": "you_can_see_everything",
            "KXRT-WILDW": "wildwood", "KXRT-CLA": "clayface", "KXRT-KLA": "klara_and_the_sun"}


def rt_snapshots(s, f: "Fetcher", *, days_before: int = 30) -> pl.DataFrame:
    """Every Wayback capture of each KXRT movie page from ``days_before`` before close to one day
    after close, parsed to (event, slug, snap_ts, liked, total, score). ``snap_ts`` is the
    capture time: the counts are what the page showed at that moment."""
    from albert.research.fundamental import settled_markets

    m = settled_markets(s, RT_SERIES)
    ev = m.group_by("event_ticker").agg(pl.col("close_time").max().alias("close"))
    rows = []
    for event, close in ev.sort("close").iter_rows():
        slug = RT_SLUGS.get(event)
        if slug is None:
            continue
        url = f"https://www.rottentomatoes.com/m/{slug}"
        caps = wayback_cdx(f, f"rottentomatoes.com/m/{slug}", (close - timedelta(days=days_before)).strftime("%Y%m%d"),
                           (close + timedelta(days=1)).strftime("%Y%m%d"))
        for c in caps:
            card = rt_scorecard(wayback_page(f, c["timestamp"], url))
            if card is None:
                continue
            rows.append({"event": event, "slug": slug, "close": close, "snap_ts": wb_time(c["timestamp"]), **card})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- study A: Vercel daily share forecasts --------------------------------------------------
VARIANTS = ("last", "dow", "mean7_dow")
NU = 4.0


def _logit_pct(v: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(v, dtype=float) / 100.0, 1e-4, 1 - 1e-4)
    return np.log(q / (1 - q))


def daily_panel(hist: pl.DataFrame) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """(lab, metric) -> (days as int ordinal, logit share), one value per calendar day (missing
    days, i.e. the lab absent from that day's list, count as 0.01%)."""
    out = {}
    for (name, metric), g in hist.group_by(["name", "metric"]):
        g = g.sort("day")
        d0, d1 = g["day"].min(), g["day"].max()
        days = np.arange(d0.toordinal(), d1.toordinal() + 1)
        v = np.full(len(days), 0.01)
        idx = np.array([d.toordinal() for d in g["day"]]) - d0.toordinal()
        v[idx] = g["v"].to_numpy()
        out[(name, metric)] = (days, _logit_pct(v))
    return out


def point_forecast(x: np.ndarray, i_target: int, h: int, variant: str, weeks: int = 8) -> float | None:
    """Forecast x[i_target] from x[: i_target - h + 1] (index i_target - h is the latest known)."""
    L = i_target - h
    if L < 7 * weeks + 7:
        return None
    if variant == "last":
        return float(x[L])
    if variant == "dow":
        ks = np.arange(1, weeks + 1)
        return float(x[L] + np.mean(x[i_target - 7 * ks] - x[L - 7 * ks]))
    if variant == "mean7_dow":
        level = x[L - 6:L + 1].mean()
        # weekday effect of the target day: same weekday minus its trailing 7-day mean, as of
        # weeks that ended by L
        effs = [x[i_target - 7 * k] - x[i_target - 7 * k - 6 - h:i_target - 7 * k - h + 1].mean()
                for k in range(1, weeks + 1)]
        return float(level + np.mean(effs))
    raise ValueError(variant)


def predictive(x: np.ndarray, i_target: int, h: int, variant: str, window: int = 56) -> tuple[float, float] | None:
    """(mu, scale) of a Student-t predictive for x[i_target]; the scale comes from the same rule's
    errors on the ``window`` days ending at the latest known day (all strictly known by then)."""
    mu = point_forecast(x, i_target, h, variant)
    if mu is None:
        return None
    L = i_target - h
    errs = []
    for j in range(L - window + 1, L + 1):
        f = point_forecast(x, j, h, variant)
        if f is not None:
            errs.append(x[j] - f)
    if len(errs) < 20:
        return None
    sd = float(np.std(errs, ddof=1))
    return mu, max(sd, 1e-3) * math.sqrt((NU - 2) / NU)


def prob_above(mu: float, scale: float, K: float) -> float:
    return float(stats.t.sf((_logit_pct(np.array([K]))[0] - mu) / scale, NU))


def vercel_history_scores(panel, labs_metrics, start: date, end: date, h: int) -> pl.DataFrame:
    """Out-of-sample log score of each variant's predictive density on the Vercel history alone
    (no Kalshi data): used to pick the variant before looking at any market."""
    rows = []
    for key in labs_metrics:
        days, x = panel[key]
        for i, dd in enumerate(days):
            if not (start.toordinal() <= dd < end.toordinal()):
                continue
            for var in VARIANTS:
                pr = predictive(x, i, h, var)
                if pr is None:
                    continue
                mu, sc = pr
                rows.append({"lab": key[0], "metric": key[1], "variant": var, "h": h,
                             "logscore": float(stats.t.logpdf((x[i] - mu) / sc, NU) - math.log(sc)),
                             "abs_err": abs(x[i] - mu)})
    return pl.DataFrame(rows)


def vercel_predictions(markets: pl.DataFrame, panel, variant: str) -> pl.DataFrame:
    """One row per (market, decision time). h = 1: t = d 16:00 UTC, latest known day d-1;
    h = 2: t = (d-1) 20:00 UTC, latest known day d-2 (a day's value is on the page by ~10 UTC of
    the next day, measured on Wayback snapshots)."""
    rows = []
    for r in markets.iter_rows(named=True):
        key = (r["lab"], r["metric"])
        if key not in panel:
            continue
        days, x = panel[key]
        i = r["day"].toordinal() - int(days[0])
        if i <= 0 or i >= len(days) + 2:
            continue
        for h, t in ((1, datetime.combine(r["day"], datetime.min.time(), UTC) + timedelta(hours=16)),
                     (2, datetime.combine(r["day"], datetime.min.time(), UTC) - timedelta(hours=4))):
            pr = predictive(x[:i - h + 1], i, h, variant) if i - h < len(x) else None
            if pr is None:
                continue
            mu, sc = pr
            rows.append({"ticker": r["ticker"], "event": r["event_ticker"], "series": r["series_ticker"],
                         "lab": r["lab"], "metric": r["metric"], "day": r["day"], "K": r["K"], "h": h, "t": t,
                         "mu_pct": 100 / (1 + math.exp(-mu)), "scale": sc, "p": prob_above(mu, sc, r["K"]),
                         "last_known_pct": 100 / (1 + math.exp(-x[i - h])), "y": r["y"]})
    return pl.DataFrame(rows)


# ---- study B: Rotten Tomatoes beta-binomial ------------------------------------------------
def rt_final_counts(snaps: pl.DataFrame) -> pl.DataFrame:
    """Counts at each event's settlement time T (= close): linear interpolation between the last
    capture before T and the first after it (or the nearest one within 36 h on one side)."""
    out = []
    for (event,), g in snaps.sort("snap_ts").group_by(["event"], maintain_order=True):
        T = g["close"][0]
        before = g.filter(pl.col("snap_ts") <= T).tail(1)
        after = g.filter(pl.col("snap_ts") > T).head(1)
        if before.height and after.height:
            t0, t1 = before["snap_ts"][0], after["snap_ts"][0]
            w = (T - t0) / (t1 - t0)
            liked = before["liked"][0] + w * (after["liked"][0] - before["liked"][0])
            total = before["total"][0] + w * (after["total"][0] - before["total"][0])
            gap = min(T - t0, t1 - T)
        else:
            one = before if before.height else after
            liked, total, gap = one["liked"][0], one["total"][0], abs(one["snap_ts"][0] - T)
        if gap > timedelta(hours=36):
            continue
        out.append({"event": event, "close": T, "liked_T": round(liked), "total_T": round(total),
                    "final_gap_h": gap.total_seconds() / 3600})
    return pl.DataFrame(out)


def rt_pairs(snaps: pl.DataFrame, finals: pl.DataFrame, *, min_total: int = 5,
             min_lead: timedelta = timedelta(hours=1)) -> pl.DataFrame:
    """One row per (event, capture before T - min_lead): the state then and at T."""
    p = (snaps.join(finals, on=["event", "close"], how="inner")
         .filter((pl.col("snap_ts") <= pl.col("close") - min_lead) & (pl.col("total") >= min_total))
         .with_columns(((pl.col("close") - pl.col("snap_ts")).dt.total_seconds() / 3600).alias("hours_left"),
                       (pl.col("total_T") - pl.col("total")).alias("dN"),
                       (pl.col("liked_T") - pl.col("liked")).alias("dL")))
    return p.sort("event", "snap_ts")


def fit_growth(pairs: pl.DataFrame) -> dict[str, float]:
    """OLS: log(1 + dN) = a + b·log(1 + hours_left) + c·log(1 + total) + e."""
    d = pairs.filter(pl.col("dN") >= 0)
    X = np.column_stack([np.ones(d.height), np.log1p(d["hours_left"].to_numpy()), np.log1p(d["total"].to_numpy())])
    yv = np.log1p(d["dN"].to_numpy().astype(float))
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    return {"a": float(beta[0]), "b": float(beta[1]), "c": float(beta[2]), "sd": float(resid.std(ddof=3))}


def fit_freshrate(pairs: pl.DataFrame, drift: bool = True) -> dict[str, float]:
    """MLE of the later reviews' fresh rate: dL ~ BetaBinomial(dN, m·k, (1-m)·k) with
    logit m = logit(current rate) + delta (delta = 0 when ``drift`` is False)."""
    from scipy.optimize import minimize

    d = pairs.filter((pl.col("dN") > 0) & (pl.col("dL") >= 0) & (pl.col("dL") <= pl.col("dN")))
    n, k = d["dN"].to_numpy().astype(float), d["dL"].to_numpy().astype(float)
    r = np.clip((d["liked"].to_numpy() + 0.5) / (d["total"].to_numpy() + 1.0), 0.005, 0.995)
    lr = np.log(r / (1 - r))

    def nll(th):
        delta = th[0] if drift else 0.0
        kappa = math.exp(th[-1])
        m = 1 / (1 + np.exp(-(lr + delta)))
        a, b = m * kappa, (1 - m) * kappa
        ll = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + betaln(k + a, n - k + b) - betaln(a, b)
        return -ll.sum()

    x0 = [0.0, math.log(20.0)] if drift else [math.log(20.0)]
    res = minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    return {"delta": float(res.x[0]) if drift else 0.0, "kappa": float(math.exp(res.x[-1])), "nll": float(res.fun),
            "n_pairs": int(d.height)}


def rt_score_pmf(liked: int, total: int, hours_left: float, growth: dict[str, float],
                 fresh: dict[str, float], n_quant: int = 9) -> dict[int, float]:
    """Distribution of the displayed score round(100·(liked + F)/(total + dN)) at T: dN lognormal
    (growth fit, integrated over ``n_quant`` quantiles), F | dN beta-binomial (fresh-rate fit)."""
    mu = growth["a"] + growth["b"] * math.log1p(hours_left) + growth["c"] * math.log1p(total)
    qs = stats.norm.ppf((np.arange(n_quant) + 0.5) / n_quant)
    r = min(max((liked + 0.5) / (total + 1.0), 0.005), 0.995)
    m = 1 / (1 + math.exp(-(math.log(r / (1 - r)) + fresh["delta"])))
    a, b = m * fresh["kappa"], (1 - m) * fresh["kappa"]
    pmf: dict[int, float] = {}
    for z in qs:
        dn = max(round(math.expm1(mu + growth["sd"] * z)), 0)
        f = np.arange(dn + 1)
        pm = stats.betabinom.pmf(f, dn, a, b) if dn > 0 else np.array([1.0])
        for sc, w in zip(rt_round(liked + f, total + dn).astype(int), pm, strict=True):
            pmf[int(sc)] = pmf.get(int(sc), 0.0) + float(w) / n_quant
    return pmf


def rt_prob_above(liked: int, total: int, hours_left: float, K: float, growth: dict[str, float],
                  fresh: dict[str, float], n_quant: int = 9) -> float:
    """P(final displayed score > K) under :func:`rt_score_pmf`."""
    pmf = rt_score_pmf(liked, total, hours_left, growth, fresh, n_quant)
    return min(max(sum(w for sc, w in pmf.items() if sc > K), 0.0), 1.0)


# ---- shared analysis --------------------------------------------------------------------------
def calibration_table(df: pl.DataFrame, col: str, bins=(0, .05, .2, .4, .6, .8, .95, 1.0001)) -> pl.DataFrame:
    b = np.digitize(df[col].to_numpy(), bins) - 1
    return (df.with_columns(pl.Series("bin", b)).group_by("bin")
            .agg(pl.len().alias("n"), pl.col(col).mean().round(3).alias(f"mean_{col}"),
                 pl.col("y").mean().round(3).alias("freq_yes"), pl.col("event").n_unique().alias("events"))
            .sort("bin").with_columns(pl.col("bin").map_elements(lambda i: f"{bins[i]:.2f}-{min(bins[i + 1], 1):.2f}",
                                                                 return_dtype=pl.Utf8)))


def segment_scores(df: pl.DataFrame, by: str) -> pl.DataFrame:
    """Brier of model vs mid per segment (rows at a quote), with event counts."""
    return (df.group_by(by).agg(pl.len().alias("n"), pl.col("event").n_unique().alias("events"),
                                ((pl.col("p") - pl.col("y")) ** 2).mean().round(5).alias("brier_model"),
                                ((pl.col("mid") - pl.col("y")) ** 2).mean().round(5).alias("brier_mid"))
            .sort(by))


def with_mid(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))


def ladder_trades(group: pl.DataFrame, scen: np.ndarray, lam: float, theta: float) -> list[dict[str, Any]]:
    """Mean-variance ladder choice for one (event, decision time).

    ``scen``: payoff matrix, scenarios x strikes (1 if the strike's YES wins in that scenario),
    equally weighted draws from the model distribution of the underlying. Candidate legs: buy YES
    at the ask or NO at 1 - bid, one contract each, taker fee included; a leg is a candidate only if
    its model edge after fee is >= theta. Greedily add the leg that most raises
    E[PnL] - lam·Var[PnL] across scenarios; stop when nothing raises it."""
    from albert.research.fundamental import taker_fee

    legs = []
    for j, r in enumerate(group.iter_rows(named=True)):
        fy, fn = taker_fee(r["ask"]), taker_fee(1 - r["bid"])
        legs.append((j, "yes", scen[:, j] - r["ask"] - fy, r["y"] - r["ask"] - fy, r))
        legs.append((j, "no", (1 - scen[:, j]) - (1 - r["bid"]) - fn, (1 - r["y"]) - (1 - r["bid"]) - fn, r))
    legs = [lg for lg in legs if lg[2].mean() >= theta]
    chosen: list[tuple] = []
    tot = np.zeros(scen.shape[0])
    obj = 0.0
    while legs:
        best = max(legs, key=lambda lg: (tot + lg[2]).mean() - lam * (tot + lg[2]).var())
        val = (tot + best[2]).mean() - lam * (tot + best[2]).var()
        if val <= obj + 1e-9:
            break
        chosen.append(best)
        tot += best[2]
        obj = val
        legs = [lg for lg in legs if lg[0] != best[0]]  # one side per strike
    return [{"ticker": lg[4]["ticker"], "event": lg[4]["event"], "t": lg[4]["t"], "side": lg[1],
             "edge": float(lg[2].mean()), "pnl": float(lg[3])} for lg in chosen]


def ladder_backtest(df: pl.DataFrame, scen_fn, holdout_from: datetime, lams=(0.0, 0.5, 2.0),
                    thetas=(0.0, 0.02, 0.05)) -> pl.DataFrame:
    """Compare per (event, t): the mean-variance ladder vs the single best-edge leg (same theta).
    ``scen_fn(group) -> scenarios x strikes`` payoff matrix from the model distribution."""
    from albert.research.fundamental import clustered_mean

    out = []
    groups = [(k, g.sort("K")) for k, g in df.group_by(["event", "t"])]
    scens = {k: scen_fn(g) for k, g in groups}
    for theta in thetas:
        for lam in (*lams, "single"):
            rows = []
            for k, g in groups:
                tr = ladder_trades(g, scens[k], 0.0 if lam == "single" else lam, theta)
                if lam == "single" and tr:
                    tr = [max(tr, key=lambda r: r["edge"])]
                rows.extend(tr)
            if not rows:
                continue
            tr = pl.DataFrame(rows).with_columns((pl.col("t") >= holdout_from).alias("holdout"))
            for split in (False, True):
                part = tr.filter(pl.col("holdout") == split)
                if not part.height:
                    continue
                per_pos = part.group_by("event", "t").agg(pl.col("pnl").sum())
                mu, t, k_ = clustered_mean(part["pnl"].to_numpy(), part["event"].to_numpy())
                out.append({"theta": theta, "rule": f"ladder_lam{lam}" if lam != "single" else "single_best",
                            "split": "holdout" if split else "dev", "contracts": part.height,
                            "positions": per_pos.height, "events": k_, "pnl_c_per_contract": round(mu * 100, 3),
                            "t": None if t is None else round(t, 2), "total_pnl_usd": round(float(part["pnl"].sum()), 2)})
    return pl.DataFrame(out)


def taker_segments(df: pl.DataFrame, by: str, holdout_from: datetime, theta: float = 0.0) -> pl.DataFrame:
    """Taker PnL (cents per contract, t clustered by event) per segment and split."""
    from albert.research.fundamental import clustered_mean, taker_pnl

    tr = taker_pnl(df, theta)
    if not tr.height:
        return pl.DataFrame()
    tr = tr.with_columns(pl.when(pl.col("t") >= holdout_from).then(pl.lit("holdout")).otherwise(pl.lit("dev")).alias("split"))
    rows = []
    for (seg, split), g in tr.group_by([by, "split"]):
        mu, t, k = clustered_mean(g["pnl"].to_numpy(), g["event"].to_numpy())
        rows.append({by: seg, "split": split, "trades": g.height, "events": k, "pnl_c": round(mu * 100, 2),
                     "t": None if t is None else round(t, 2)})
    return pl.DataFrame(rows).sort(by, "split")


# ---- runners ----------------------------------------------------------------------------------
def run_vercel(s, holdout_from: datetime = datetime(2026, 10, 1, tzinfo=UTC), variant: str = "dow") -> Path:
    from albert.research.fundamental import attach_quotes, evaluate, report

    labs_page = max((s.data_dir / "cache" / NAME / "vercel").glob("labs_*.html"))
    hist = vercel_rawdata(labs_page.read_text())
    panel = daily_panel(hist)
    mk = vercel_markets(s)
    pred = vercel_predictions(mk, panel, variant)
    candles = pl.read_parquet(s.data_dir / "cache" / "fundamental_culture_vercel_candles.parquet")
    q = with_mid(attach_quotes(pred, candles))
    q.write_parquet(s.data_dir / "cache" / "fundamental_culture_vercel_pred_quotes.parquet")
    # the variant was picked on the seven series with 20 events each (ZAI/GOOG spend have 2)
    main = mk.group_by("lab", "metric").agg(pl.col("event_ticker").n_unique().alias("ev")).filter(pl.col("ev") >= 10)
    keys = [(lab, met) for lab, met in main.select("lab", "metric").iter_rows()]
    hs = pl.concat([vercel_history_scores(panel, keys, date(2026, 6, 1), date(2026, 9, 21), h) for h in (1, 2)])
    results = {h: evaluate(q.filter(pl.col("h") == h), holdout_from) for h in (1, 2)}
    actual = hist.rename({"name": "lab", "v": "actual_pct"})
    err = (pred.filter(pl.col("h") == 1).unique(["event", "h"])
           .join(actual, on=["lab", "metric", "day"], how="left")
           .with_columns((pl.col("mu_pct") - pl.col("actual_pct")).abs().alias("abs_err_model"),
                         (pl.col("last_known_pct") - pl.col("actual_pct")).abs().alias("abs_err_last")))

    def scen_fn(g: pl.DataFrame) -> np.ndarray:
        r = g.row(0, named=True)
        z = stats.t.ppf((np.arange(400) + 0.5) / 400, NU)
        v = 100 / (1 + np.exp(-(math.log(r["mu_pct"] / (100 - r["mu_pct"])) + r["scale"] * z)))
        return (v[:, None] > g["K"].to_numpy()[None, :]).astype(float)

    ladders = {h: ladder_backtest(q.filter(pl.col("h") == h), scen_fn, holdout_from) for h in (1, 2)}
    extra = {"study": "A: Vercel AI Gateway daily lab shares (VREQ = requests, VSPEND = cost)",
             "variant": variant, "holdout_from": holdout_from.isoformat(), "labs_page": labs_page.name,
             "decision_times": "h=1: d 16:00 UTC (latest known day d-1); h=2: (d-1) 20:00 UTC (latest known d-2)",
             "rows_by_h": {h: results[h]["rows"] for h in (1, 2)}}
    from albert.reports import Report
    path = report(s, f"{NAME}-vercel-h1", results[1], extra)
    path2 = report(s, f"{NAME}-vercel-h2", results[2], extra)
    rep = Report(s.reports_dir, f"research-fundamental-{NAME}-vercel-extra")
    rep.json("Main reports", {"h1": str(path), "h2": str(path2)})
    rep.table("Variant choice on Vercel history only (2026-06-01..09-20, no Kalshi data): mean log score",
              hs.group_by("h", "variant").agg(pl.col("logscore").mean().round(4), pl.col("abs_err").mean().round(4),
                                              pl.len()).sort("h", "logscore", descending=[False, True]))
    rep.table("Point forecast error (share points) for h=1, model vs last known value",
              err.with_columns((pl.col("t") >= holdout_from).alias("holdout")).group_by("holdout", "series")
              .agg(pl.len().alias("events"), pl.col("abs_err_model").mean().round(3), pl.col("abs_err_last").mean().round(3))
              .sort("holdout", "series"), max_rows=40)
    for h in (1, 2):
        part = q.filter(pl.col("h") == h).with_columns((pl.col("t") >= holdout_from).alias("holdout"),
                                                       (pl.col("p") - pl.col("mid")).abs().alias("dev_abs"))
        rep.table(f"h={h}: Brier by series and split", segment_scores(part.with_columns(
            (pl.col("series") + pl.when(pl.col("holdout")).then(pl.lit(" holdout")).otherwise(pl.lit(" dev"))).alias("seg")), "seg"),
            max_rows=40)
        rep.table(f"h={h}: Brier by mid band (all splits)", segment_scores(part.with_columns(
            (pl.col("mid") * 10).floor().cast(pl.Int32).alias("mid_decile")), "mid_decile"))
        rep.table(f"h={h}: calibration of model p (holdout)", calibration_table(part.filter(pl.col("holdout")), "p"))
        rep.table(f"h={h}: calibration of mid (holdout)", calibration_table(part.filter(pl.col("holdout")), "mid"))
        rep.table(f"h={h}: ladder optimisation vs single best leg (one contract per leg, taker)", ladders[h], max_rows=60)
        spr = part.with_columns((pl.col("ask") - pl.col("bid")).cut([0.035, 0.105, 0.255], labels=["<=3c", "4-10c", "11-25c", ">25c"])
                                .cast(pl.Utf8).alias("spread"))
        rep.table(f"h={h}: taker theta=0 by spread at decision time", taker_segments(spr, "spread", holdout_from))
        rep.table(f"h={h}: Brier by spread (all splits)", segment_scores(spr, "spread"))
        rep.table(f"h={h}: taker theta=0 by metric", taker_segments(part, "metric", holdout_from))
    rep.save()
    return path


def rt_events(s) -> pl.DataFrame:
    from albert.research.fundamental import settled_markets

    m = settled_markets(s, RT_SERIES)
    return m.with_columns(pl.col("yes_sub_title").str.extract(r"Above (\d+)").cast(pl.Float64).alias("K"))


def rt_predictions(markets: pl.DataFrame, snaps: pl.DataFrame, growth: dict[str, float],
                   fresh: dict[str, float], lead: timedelta = timedelta(minutes=2)) -> pl.DataFrame:
    """One row per (market, Wayback capture before close - 1 h). t = capture time + ``lead``: the
    counts were public on the page at the capture time."""
    rows = []
    caps = snaps.filter(pl.col("total") >= 5).sort("snap_ts")
    for (event,), g in caps.group_by(["event"], maintain_order=True):
        mk = markets.filter(pl.col("event_ticker") == event)
        if not mk.height:
            continue
        T = mk["close_time"].max()
        for c in g.filter(pl.col("snap_ts") <= T - timedelta(hours=1)).iter_rows(named=True):
            hours_left = (T - c["snap_ts"]).total_seconds() / 3600
            pmf = rt_score_pmf(c["liked"], c["total"], hours_left, growth, fresh)
            for r in mk.iter_rows(named=True):
                p = min(max(sum(w for sc, w in pmf.items() if sc > r["K"]), 0.0), 1.0)
                rows.append({"ticker": r["ticker"], "event": event, "t": c["snap_ts"] + lead, "K": r["K"], "p": p,
                             "liked": c["liked"], "total": c["total"], "score_now": int(rt_round(c["liked"], c["total"])),
                             "hours_left": hours_left, "y": r["y"]})
    return pl.DataFrame(rows)


def rt_check_finals(markets: pl.DataFrame, finals: pl.DataFrame) -> pl.DataFrame:
    """Does the interpolated final score fall in the interval Kalshi's settled ladder implies?"""
    iv = markets.group_by("event_ticker").agg(pl.col("K").filter(pl.col("y") == 1).max().alias("lo"),
                                              pl.col("K").filter(pl.col("y") == 0).min().alias("hi"))
    j = finals.join(iv.rename({"event_ticker": "event"}), on="event", how="inner").with_columns(
        pl.struct("liked_T", "total_T").map_elements(lambda r: int(rt_round(r["liked_T"], r["total_T"])),
                                                     return_dtype=pl.Int64).alias("score_T"))
    return j.with_columns((((pl.col("lo").is_null()) | (pl.col("score_T") > pl.col("lo")))
                           & ((pl.col("hi").is_null()) | (pl.col("score_T") <= pl.col("hi")))).alias("consistent"))


def run_rt(s, holdout_from: datetime = datetime(2026, 10, 1, tzinfo=UTC)) -> Path:
    from albert.reports import Report
    from albert.research.fundamental import attach_quotes, evaluate, report

    snaps = pl.read_parquet(s.data_dir / "cache" / "fundamental_culture_rt_snapshots.parquet")
    mk = rt_events(s)
    finals = rt_final_counts(snaps)
    check = rt_check_finals(mk, finals)
    pairs = rt_pairs(snaps, finals)
    dev_pairs = pairs.filter(pl.col("close") < holdout_from)
    growth = fit_growth(dev_pairs)
    variants = {"drift": fit_freshrate(dev_pairs, drift=True), "nodrift": fit_freshrate(dev_pairs, drift=False)}
    candles = pl.read_parquet(s.data_dir / "cache" / "fundamental_culture_rt_candles.parquet")
    evals, preds = {}, {}
    for name, fresh in variants.items():
        pred = rt_predictions(mk, snaps, growth, fresh)
        preds[name] = with_mid(attach_quotes(pred, candles))
        dev = preds[name].filter(pl.col("t") < holdout_from)
        evals[name] = {"dev_brier": float(((dev["p"] - dev["y"]) ** 2).mean()), "dev_rows": dev.height}
    chosen = min(evals, key=lambda k: evals[k]["dev_brier"])
    q = preds[chosen]
    q.write_parquet(s.data_dir / "cache" / "fundamental_culture_rt_pred_quotes.parquet")
    res = evaluate(q, holdout_from, min_trades=15)
    extra = {"study": "B: Rotten Tomatoes Tomatometer at T from Wayback captures (beta-binomial)",
             "growth_fit_dev": growth, "freshrate_fits_dev": variants, "variant_dev_brier": evals,
             "variant_chosen": chosen, "holdout_from": holdout_from.isoformat(),
             "captures": snaps.height, "events_with_captures": snaps["event"].n_unique(),
             "dev_pairs": dev_pairs.height}
    path = report(s, f"{NAME}-rt", res, extra)

    def scen_fn(g: pl.DataFrame) -> np.ndarray:
        r = g.row(0, named=True)
        pmf = rt_score_pmf(r["liked"], r["total"], r["hours_left"], growth, variants[chosen])
        sc = np.array(sorted(pmf))
        cdf = np.cumsum([pmf[k] for k in sc])
        draws = sc[np.minimum(np.searchsorted(cdf, (np.arange(400) + 0.5) / 400 * cdf[-1]), len(sc) - 1)]
        return (draws[:, None] > g["K"].to_numpy()[None, :]).astype(float)

    rep = Report(s.reports_dir, f"research-fundamental-{NAME}-rt-extra")
    rep.json("Main report", {"path": str(path)})
    rep.table("Interpolated final counts vs Kalshi's settled ladder (data check)", check.sort("close"), max_rows=40)
    part = q.with_columns((pl.col("t") >= holdout_from).alias("holdout"),
                          (pl.col("score_now") - pl.col("K")).alias("gap"))
    rep.table("Brier by hours left (all splits)", segment_scores(part.with_columns(
        pl.col("hours_left").cut([24, 72, 168], labels=["<1d", "1-3d", "3-7d", ">7d"]).cast(pl.Utf8).alias("lead")), "lead"))
    rep.table("Brier by current score minus strike", segment_scores(part.with_columns(
        pl.col("gap").cut([-10, -3, 3, 10], labels=["<-10", "-10..-3", "-3..3", "3..10", ">10"]).cast(pl.Utf8).alias("gap_band")),
        "gap_band"))
    rep.table("Brier by event (holdout flag)", segment_scores(part.with_columns(
        (pl.col("event") + pl.when(pl.col("holdout")).then(pl.lit(" H")).otherwise(pl.lit(" D"))).alias("ev")), "ev"), max_rows=40)
    rep.table("Calibration of model p (all)", calibration_table(part, "p"))
    rep.table("Calibration of mid (all)", calibration_table(part, "mid"))
    rep.table("Ladder optimisation vs single best leg", ladder_backtest(q, scen_fn, holdout_from), max_rows=60)
    rep.table("Taker theta=0 by event (one event dominates the holdout: check)", taker_segments(q, "event", holdout_from),
              max_rows=40)
    rep.save()
    return path


def main(which: str = "all") -> None:
    from albert.config import settings

    s = settings()
    if which in ("vercel", "all"):
        print(run_vercel(s))
    if which in ("rt", "all"):
        print(run_rt(s))


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "all")
