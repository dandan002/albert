"""Adversarial review of ``music_charts``' paper candidate (HITS Tuesday preview -> Luminate ladders).

Reviewed claim (``data/research/music_charts/primary.json``): at the first whole hour >= 5 min after
HITS Daily Double's Tuesday chart preview, price every KXALBUMEQUIV / KXPUREALBUMS / KXALBUMSTREAMS
strike from the HITS number (Student-t on log(Luminate / HITS), fit on dev) and take YES at the ask /
NO at 1 - bid when the edge after fee is >= 5c; $4 orders, <= $5 per event, held to settlement.
Dev +20.8c (t 3.75, 60 fills), holdout +16.7c (t 1.83, 15 fills).

This module never edits the reviewed module or its outputs. It reads the agent's panel and caches,
adds its own Wayback / Kalshi fetches (append-only caches, polite single-threaded), and writes to
``data/research/music_charts/review/``. Notes: ``data/research/music_charts/review.md``.

    python -m albert.research.music_charts_review wayback    # CDX + copies of the traded posts / listing pages
    python -m albert.research.music_charts_review timing     # check 1
    python -m albert.research.music_charts_review candles    # check 2 (re-fetches a few markets' candles + trades)
    python -m albert.research.music_charts_review settle     # check 3 (re-fetches the holdout markets)
    python -m albert.research.music_charts_review l2         # check 4 (displayed depth from the local L2 recorder)
    python -m albert.research.music_charts_review score      # checks 4 + 5 (re-scoring, clustered t, corrections)
    python -m albert.research.music_charts_review drift      # check 6 (price drift toward the HITS model)
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.research import music_charts as mc

SRC = mc.OUT
OUT = SRC / "review"
WB = Path("data/cache/music_charts_review/wayback")
STREAM = Path("data/raw/stream/kalshi")
THETA = 0.05
DECISION = "preview_fast"
CELLS_TRIED = 83
SEP22_SLUG = "chart-preview-going-green-2026-09-22"


def _dump(obj: Any, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=str))


def _ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=UTC)


# ---- Wayback (robots: web.archive.org has no robots file -> allowed; copies of HITS only) ----------
_last = [0.0]


def _get(url: str, params: dict[str, Any] | None = None, *, gap: float = 6.0):
    import httpx

    for attempt in range(5):
        wait = gap - (time.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
        try:
            r = httpx.get(url, params=params, headers={"User-Agent": mc.UA}, timeout=120, follow_redirects=True)
        except httpx.HTTPError as error:
            print(f"  {url}: {error}", flush=True)
            time.sleep(60 * (attempt + 1))
            continue
        if r.status_code in (429, 502, 503, 504):
            time.sleep(20 * (attempt + 1))
            continue
        return r
    return None


def cdx(url: str, *, prefix: bool = False, frm: str = "20260401", to: str = "20261120") -> list[dict[str, Any]]:
    """Every capture of ``url`` (cached once per query; never overwritten)."""
    WB.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9]+", "_", f"{url}_{prefix}_{frm}_{to}").strip("_")
    p = WB / f"cdx_{key}.json"
    if not p.exists():
        params = {"url": url, "from": frm, "to": to, "output": "json", "fl": "timestamp,original,statuscode,digest",
                  "limit": 50000}
        if prefix:
            params["matchType"] = "prefix"
        r = _get("https://web.archive.org/cdx/search/cdx", params)
        if r is None or r.status_code != 200:
            raise RuntimeError(f"cdx {url}: {None if r is None else r.status_code}")
        tmp = p.with_suffix(".tmp")
        tmp.write_text(r.text or "[]")
        tmp.replace(p)
    rows = json.loads(p.read_text() or "[]")
    return [{"ts": _ts(a), "original": b, "status": c, "digest": d} for a, b, c, d in rows[1:]]


def wb_copy(ts: datetime, original: str) -> str | None:
    """The archived bytes of one capture (``id_``: no Wayback rewriting), cached."""
    stamp = ts.strftime("%Y%m%d%H%M%S")
    name = re.sub(r"[^A-Za-z0-9]+", "_", original.split("://", 1)[-1]).strip("_")[:150]
    p = WB / f"{stamp}_{name}.html.gz"
    if not p.exists():
        r = _get(f"https://web.archive.org/web/{stamp}id_/{original}")
        if r is None or r.status_code != 200:
            print(f"  copy {stamp} {original}: {None if r is None else r.status_code}", flush=True)
            return None
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(r.content))
        tmp.replace(p)
    return gzip.decompress(p.read_bytes()).decode("utf-8", "ignore")


# ---- the reviewed rule, reproduced -------------------------------------------------------------------
def panel() -> pl.DataFrame:
    return pl.read_parquet(SRC / "panel_ladder.parquet")


def base_fills(p: pl.DataFrame, split: str, *, theta: float = THETA, col: str = "p_model") -> pl.DataFrame:
    rows = p.filter((pl.col("split") == split) & (pl.col("decision") == DECISION))
    return mc.cap_event(mc.taker_fills(rows, col, theta))


def traded_posts(p: pl.DataFrame) -> pl.DataFrame:
    """One row per traded week: the HITS post behind the decision, its stamp and the decision time."""
    f = pl.concat([base_fills(p, s).with_columns(pl.lit(s).alias("split_")) for s in ("dev", "holdout")], how="diagonal_relaxed")
    wk = f.group_by("t").agg(pl.col("hits_pub").min(), pl.col("split_").first().alias("split"), pl.len().alias("fills"))
    posts = pl.read_parquet(SRC / "hits_posts.parquet").select("slug", "published_at", "first_capture", "kind")
    return wk.join(posts, left_on="hits_pub", right_on="published_at", how="left").sort("t")


# ---- check 1: is the HITS table public, and unchanged, at the decision? ------------------------------
_EMBED_RX = re.compile(r'"_type":"chartsEmbed","chart":\{"chart_data":"[^"]*","chart_type":\{"slug":\{"_type":"slug",'
                       r'"current":"([^"]+)"\},"title":"[^"]*"\},"date":"([^"]+)"\}')


def post_body(slug: str) -> dict[str, Any]:
    """Asset upload stamps inside the post's own sections (before its chart embed) and the chart
    document the post embeds (the table is a separate CMS document rendered at fetch time)."""
    raw = gzip.decompress((mc.HITS_CACHE / f"{slug}.html.gz").read_bytes()).decode("utf-8", "ignore")
    t = raw.replace('\\"', '"')
    m = mc._MAIN_PUB_RX.search(t)
    if not m:
        return {"assets": [], "embed": None}
    ci = t.find('"chartItems":[', m.start())
    seg = t[m.start(): ci if ci > 0 else m.start() + 20000]
    assets = sorted(datetime.fromisoformat(x) for x in re.findall(r'"created_at":"([^"]+)"', seg))
    emb = _EMBED_RX.search(t, m.start())
    return {"assets": assets, "embed": f"{emb.group(1)}/{emb.group(2)}" if emb else None}


def table_of(raw: str, slug: str) -> pl.DataFrame:
    items = mc.parse_hits_post(raw, slug)["items"]
    return pl.DataFrame(items, infer_schema_length=None) if items else pl.DataFrame()


def compare_tables(a: pl.DataFrame, b: pl.DataFrame) -> dict[str, Any]:
    """Rows identical in (album, artist, activity, albums) between two parses of the same post."""
    if a.is_empty() or b.is_empty():
        return {"rows_a": a.height, "rows_b": b.height, "identical": None}
    cols = ["pos", "album", "artist", "activity", "albums"]
    j = a.select(cols).join(b.select(cols), on="pos", suffix="_b")
    same = ((j["album"] == j["album_b"]) & (j["artist"] == j["artist_b"]) & (j["activity"] == j["activity_b"])
            & (j["albums"].fill_null(-1) == j["albums_b"].fill_null(-1)))
    diff = j.filter(~same).select("pos", "album", "activity", "activity_b", "albums", "albums_b").to_dicts()
    return {"rows_a": a.height, "rows_b": b.height, "identical": int(same.sum()), "diff": diff}


_MIDWEEK_RX = re.compile(r"/charts/midweek-20(\?_rsc=[a-z0-9]+)?$")


def _captures() -> dict[str, list[dict[str, Any]]]:
    """CDX lists (cached): chart posts, the homepage, and the midweek-20 chart page (latest midweek table)."""
    ok = lambda cs: [c for c in cs if c["status"] == "200"]
    return {"posts": ok(cdx("www.hitsdailydouble.com/news/charts/", prefix=True)),
            "home": ok(cdx("www.hitsdailydouble.com/")),
            "midweek": [c for c in ok(cdx("www.hitsdailydouble.com/charts/", prefix=True)) if _MIDWEEK_RX.search(c["original"])]}


def evidence_plan(w: dict[str, Any], caps: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """The captures that bear on one traded post: its first capture; homepage captures between the stamp
    and the decision plus the last one in the 3 h before the stamp; midweek-chart captures from 24 h
    before the stamp to 48 h after the decision."""
    slug, pub, t = w["slug"], w["hits_pub"], w["t"]
    mine = sorted((c for c in caps["posts"] if c["original"].rstrip("/").endswith(slug)), key=lambda c: c["ts"])
    home_in = [c for c in caps["home"] if pub < c["ts"] <= t]
    home_pre = [c for c in caps["home"] if pub - timedelta(hours=3) <= c["ts"] <= pub][-1:]
    mid = [c for c in caps["midweek"] if pub - timedelta(hours=24) <= c["ts"] <= t + timedelta(hours=48)]
    return {"post_first": mine[:1], "post_all": mine, "home_in": home_in, "home_pre": home_pre, "midweek": mid}


def wayback(log=print) -> None:
    """Fetch the planned captures once (6 s apart; Wayback refuses faster clients)."""
    tp = traded_posts(panel())
    caps = _captures()
    log(f"cdx: {({k: len(v) for k, v in caps.items()})}")
    for w in tp.iter_rows(named=True):
        plan = evidence_plan(w, caps)
        for role in ("post_first", "home_in", "home_pre", "midweek"):
            for c in plan[role]:
                wb_copy(c["ts"], c["original"])
        log(f"  {w['slug']}: " + ", ".join(f"{k} {len(v)}" for k, v in plan.items() if k != "post_all"), flush=True)


def _cached(c: dict[str, Any]) -> str | None:
    stamp = c["ts"].strftime("%Y%m%d%H%M%S")
    name = re.sub(r"[^A-Za-z0-9]+", "_", c["original"].split("://", 1)[-1]).strip("_")[:150]
    path = WB / f"{stamp}_{name}.html.gz"
    return gzip.decompress(path.read_bytes()).decode("utf-8", "ignore") if path.exists() else None


def timing(log=print) -> dict[str, Any]:
    tp = traded_posts(panel())
    caps = _captures()
    items = pl.read_parquet(SRC / "hits_items.parquet")
    rows = []
    for w in tp.iter_rows(named=True):
        slug, pub, t = w["slug"], w["hits_pub"], w["t"]
        plan = evidence_plan(w, caps)
        body = post_body(slug)
        late = [a for a in body["assets"] if a > pub]
        first = plan["post_first"][0] if plan["post_first"] else None
        raw = _cached(first) if first else None
        cmp = compare_tables(table_of(raw, slug), items.filter(pl.col("slug") == slug)) if raw else None
        home = [(c["ts"], (_cached(c) or "")) for c in plan["home_in"] + plan["home_pre"]]
        home_in = [(ts, slug in h) for ts, h in home if ts > pub and h]
        home_pre = [(ts, slug in h) for ts, h in home if ts <= pub and h]
        mids = []
        for c in plan["midweek"]:
            h = _cached(c)
            if not h:
                continue
            tab = table_of(h, "midweek")
            cur = items.filter(pl.col("slug") == slug)
            mids.append({"ts": c["ts"], "url": c["original"].rsplit("/", 1)[-1], "rows": tab.height,
                         "same_as_post_today": (compare_tables(tab, cur) or {}).get("identical")})
        rows.append({
            "split": w["split"], "slug": slug, "fills": w["fills"], "published_at": pub, "decision_t": t,
            "min_pub_to_t": round((t - pub).total_seconds() / 60, 1),
            "round_stamp": pub.second == 0 and pub.microsecond == 0,
            "chart_doc": body["embed"], "body_assets": len(body["assets"]),
            "max_body_asset": max(body["assets"]) if body["assets"] else None,
            "assets_after_pub": len(late), "assets_after_t": sum(a > t for a in late),
            "first_post_capture": first["ts"] if first else None,
            "post_captures_before_t": sum(c["ts"] <= t for c in plan["post_all"]),
            "home_in_window": home_in, "home_before_pub": home_pre, "midweek_captures": mids,
            "first_capture_vs_today": cmp,
        })
    docs: dict[str, list[str]] = {}
    posts = pl.read_parquet(SRC / "hits_posts.parquet").filter(pl.col("n_items") > 0)
    for s in posts["slug"].to_list():
        e = post_body(s)["embed"]
        if e:
            docs.setdefault(e, []).append(s)
    shared = {k: v for k, v in docs.items() if len(v) > 1}
    sep22 = None
    sep = [r for r in rows if r["slug"] == SEP22_SLUG]
    if sep and sep[0]["first_post_capture"]:
        first = evidence_plan(tp.filter(pl.col("slug") == SEP22_SLUG).row(0, named=True), caps)["post_first"][0]
        raw = _cached(first)
        tues = table_of(raw, SEP22_SLUG) if raw else pl.DataFrame()
        if not tues.is_empty():
            OUT.mkdir(parents=True, exist_ok=True)
            tues.write_parquet(OUT / "sep22_tuesday_table.parquet")
        sep22 = {"capture": first["ts"], **compare_tables(tues, items.filter(pl.col("slug") == SEP22_SLUG))}
    res = {"weeks": rows, "shared_chart_docs": shared, "sep22_tuesday_vs_today": sep22}
    _dump(res, "timing.json")
    with pl.Config(tbl_rows=40, tbl_width_chars=260, fmt_str_lengths=36):
        log(pl.DataFrame([{k: v for k, v in r.items() if k in ("split", "slug", "published_at", "decision_t", "min_pub_to_t",
                                                                 "round_stamp", "assets_after_pub", "assets_after_t",
                                                                 "first_post_capture", "post_captures_before_t")}
                          | {"cap_identical": (r["first_capture_vs_today"] or {}).get("identical"),
                             "home_in": str([(f"{a:%H:%M}", b) for a, b in r["home_in_window"]]),
                             "home_pre": str([(f"{a:%H:%M}", b) for a, b in r["home_before_pub"]]),
                             "midweek": str([(f"{m['ts']:%m-%d %H:%M}", m["rows"], m["same_as_post_today"]) for m in r["midweek_captures"]])}
                          for r in rows]))
    log(json.dumps({"shared": shared, "sep22": sep22}, default=str, indent=1)[:4000])
    return res


# ---- check 2: candle timing against the API ---------------------------------------------------------
def candles_check(n_markets: int = 6, log=print) -> dict[str, Any]:
    """Re-fetch hourly candles and trades around the decision for a few traded markets; check that the
    candle used (end_ts <= t) is end-labelled (it holds the trades of the hour before end_ts) and that
    the cached close quotes match the API."""
    from albert.research.counts import _api
    from albert.research.xcat import ohlc_rows
    from albert.venues.kalshi import parse_ts

    p = panel()
    f = pl.concat([base_fills(p, "holdout"), base_fills(p, "dev")], how="diagonal_relaxed")
    pick = f.sort("pnl_c", descending=True).unique("event_ticker", keep="first").sort("pnl_c", descending=True).head(n_markets)
    api = _api(min_interval=1.0)
    cutoff = api.cutoff()["market_settled_ts"]
    out = []
    for r in pick.iter_rows(named=True):
        tk, t = r["ticker"], r["t"]
        hist = r["settlement_ts"] < cutoff
        lo, hi = t - timedelta(hours=4), t + timedelta(hours=2)
        try:
            page = api.candles(tk, lo, hi, 60, hist)
        except RuntimeError:
            page = api.candles(tk, lo, hi, 60, not hist)
            hist = not hist
        cs = ohlc_rows(page.payload, tk, 60)
        trades = []
        for pg in api.trades(tk, hist, min_ts=lo, max_ts=hi):
            trades.extend(pg.payload.get("trades") or [])
        tts = [parse_ts(x.get("created_time")) for x in trades]
        # end-labelling: the candle ending at E holds the trades in (E - 1h, E]
        check = []
        for c in cs:
            e = c["end_ts"]
            n_in = sum(1 for x in tts if e - timedelta(hours=1) < x <= e)
            n_next = sum(1 for x in tts if e < x <= e + timedelta(hours=1))
            check.append({"end_ts": e, "volume": c["volume"], "trades_prev_hour": n_in, "trades_next_hour": n_next,
                          "bid_close": c["bid_close"], "ask_close": c["ask_close"]})
        used = [c for c in cs if c["end_ts"] <= t]
        api_q = used[-1] if used else None
        out.append({"ticker": tk, "t": t, "q_ts_panel": r["q_ts"], "bid_panel": r["bid"], "ask_panel": r["ask"],
                    "q_ts_api": api_q["end_ts"] if api_q else None,
                    "bid_api": api_q["bid_close"] if api_q else None, "ask_api": api_q["ask_close"] if api_q else None,
                    "trades": len(trades), "candles": check})
        log(f"{tk} t={t:%Y-%m-%d %H:%M} panel q={r['q_ts']:%H:%M} {r['bid']}/{r['ask']}  api q="
            f"{api_q['end_ts']:%H:%M} {api_q['bid_close']}/{api_q['ask_close']}  trades {len(trades)}" if api_q else f"{tk}: no api candle")
        for c in check:
            log(f"    end {c['end_ts']:%m-%d %H:%M} vol {c['volume']} trades(prev h) {c['trades_prev_hour']} "
                f"(next h) {c['trades_next_hour']} bid {c['bid_close']} ask {c['ask_close']}")
    # every traded row: the quote's candle end vs the decision
    age = f.with_columns(((pl.col("t") - pl.col("q_ts")).dt.total_minutes()).alias("age_min"))
    summ = {"rows": f.height, "q_ts_after_t": int((age["age_min"] < 0).sum()),
            "q_ts_eq_t": int((age["age_min"] == 0).sum()),
            "age_min_quantiles": [float(age["age_min"].quantile(q)) for q in (0.5, 0.9, 1.0)]}
    log(json.dumps(summ))
    res = {"markets": out, "all_traded": summ}
    _dump(res, "candles_check.json")
    return res


# ---- check 3: settlement and the thousands scale ------------------------------------------------------
def settle(log=print) -> dict[str, Any]:
    from albert.http import HttpError
    from albert.research.counts import _api
    from albert.venues.kalshi import num

    p = panel()
    hf = base_fills(p, "holdout")
    df = base_fills(p, "dev")
    api = _api(min_interval=1.0)
    rows = []
    for r in hf.iter_rows(named=True):
        m = None
        for path in (f"/markets/{r['ticker']}", f"/historical/markets/{r['ticker']}"):
            try:
                m = api.client.get(path).payload.get("market")
                break
            except HttpError as error:          # 404 on the other endpoint
                print(f"  {path}: {str(error)[-80:]}")
        if m is None:
            rows.append({"ticker": r["ticker"], "api": None})
            continue
        ev = m.get("expiration_value")
        v = mc.luminate_value(ev)
        k = num(m.get("floor_strike"))
        scales = {s: (v * s > k) if v is not None and k is not None else None for s in (1, 1e3, 1e6)}
        rows.append({"ticker": r["ticker"], "side": r["side"], "y_panel": r["y"], "result_api": m.get("result"),
                     "status": m.get("status"), "strike_type": m.get("strike_type"), "floor_strike": k,
                     "expiration_value": ev, "yes_if_scale_1": scales[1], "yes_if_scale_1e3": scales[1e3],
                     "yes_if_scale_1e6": scales[1e6],
                     "agree": (m.get("result") == "yes") == bool(r["y"])})
    # scale inference over every ladder event: ambiguous / impossible scales and any traded event affected
    mk = mc.load_markets(SRC)
    ev = mc.ladder_events(mk)
    amb = []
    for e in ev.iter_rows(named=True):
        v, lo, hi = e["lum_raw"], e["lo"], e["hi"]
        ok = [s for s in (1.0, 1e3, 1e6) if v is not None and (lo is None or v * s >= lo) and (hi is None or v * s <= hi)]
        if len(ok) != 1:
            amb.append({"event_ticker": e["event_ticker"], "lum_raw": v, "lo": lo, "hi": hi, "ok_scales": ok, "split": e["split"]})
    traded = set(hf["event_ticker"].to_list()) | set(df["event_ticker"].to_list())
    # result vs expiration value on every strike of every event (with the inferred scale)
    lad = mk.filter(pl.col("series").is_in(mc.LADDER_SERIES)).join(ev.select("event_ticker", "lum"), on="event_ticker")
    bad = lad.filter(pl.col("lum").is_not_null() & ((pl.col("lum") > pl.col("floor_strike")).cast(pl.Int8) != pl.col("y")))
    res = {"holdout_rows": rows, "holdout_all_agree": all(r.get("agree") for r in rows),
           "events": ev.height, "scale_not_unique": amb, "scale_not_unique_traded": [a for a in amb if a["event_ticker"] in traded],
           "strikes_result_contradicts_value": bad.select("ticker", "floor_strike", "lum", "y", "expiration_value").to_dicts()}
    _dump(res, "settle.json")
    with pl.Config(tbl_rows=30, tbl_width_chars=250):
        log(pl.DataFrame(rows))
    log(f"scale not unique: {len(amb)} events ({len(res['scale_not_unique_traded'])} traded); "
        f"strikes whose result contradicts the inferred value: {bad.height}")
    return res


# ---- check 4: displayed depth (local L2 recorder) ----------------------------------------------------
def _books_at_hour(day: str, hour: int, tickers: set[str]) -> dict[str, dict[str, Any]]:
    """The first snapshot of each ticker in the hour's files (the recorder re-snapshots at each hour)."""
    import io

    import zstandard

    out: dict[str, dict[str, Any]] = {}
    for f in sorted((STREAM / day).glob(f"{hour:02d}-k*.jsonl.zst")):
        with open(f, "rb") as fh:
            rd = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
            for line in rd:
                if "orderbook_snapshot" not in line or not any(t in line for t in tickers):
                    continue
                fr = json.loads(json.loads(line)["frame"])
                m = fr.get("msg") or {}
                tk = m.get("market_ticker")
                if tk not in tickers or tk in out:
                    continue
                yes = sorted(((float(p), float(q)) for p, q in m.get("yes_dollars_fp") or []), reverse=True)
                no = sorted(((float(p), float(q)) for p, q in m.get("no_dollars_fp") or []), reverse=True)
                asks = [(round(1 - p, 4), q) for p, q in no]          # YES asks, best first
                out[tk] = {"bid": yes[0][0] if yes else None, "bid_size": yes[0][1] if yes else None,
                           "ask": asks[0][0] if asks else None, "ask_size": asks[0][1] if asks else None,
                           "ask_depth_5c": sum(q for p, q in asks if asks and p <= asks[0][0] + 0.05 + 1e-9),
                           "bid_depth_5c": sum(q for p, q in yes if yes and p >= yes[0][0] - 0.05 - 1e-9)}
    return out


def l2(log=print) -> dict[str, Any]:
    days = sorted(d.name for d in STREAM.iterdir() if d.is_dir())
    local = [d for d in days if not (STREAM / d / "_archived.json").exists()]
    archived = [d for d in days if (STREAM / d / "_archived.json").exists()]
    # which ladder markets each sample-recorder run covered
    runs = []
    for f in sorted(Path("data/runs").glob("*-record.json")):
        try:
            prm = json.loads(f.read_text())["params"]
        except (ValueError, KeyError):
            continue
        lad = [m["ticker"] for m in prm.get("kalshi_markets") or [] if m["ticker"].split("-")[0] in mc.LADDER_SERIES]
        if lad:
            runs.append({"run": f.stem, "start": prm.get("started_at"), "minutes": prm.get("minutes"), "ladder_markets": lad})
    tues = "2026-11-17"
    targets = {t for r in runs if r["start"] and r["start"][:10] in ("2026-11-16", "2026-11-17") for t in r["ladder_markets"]}
    books = {h: _books_at_hour(tues, h, targets) for h in (20, 21, 22)} if tues in local else {}
    rows = [{"hour_utc": h, "ticker": tk, **b} for h, bs in books.items() for tk, b in bs.items()]
    # the agent's forward snapshots (Friday Nov 20): top-of-book sizes by series
    fw = pl.concat([pl.read_parquet(f) for f in sorted((SRC / "forward" / "kalshi").glob("*.parquet"))], how="diagonal_relaxed")
    fw = fw.with_columns(pl.col("ticker").str.extract(r"^([A-Z0-9]+)-", 1).alias("series")).filter(pl.col("series").is_in(mc.LADDER_SERIES))
    fws = (fw.filter(pl.col("yes_ask").is_between(0.02, 0.98)).group_by("series")
           .agg(pl.len(), pl.col("yes_ask_size").median().alias("ask_size_med"), pl.col("yes_bid_size").median().alias("bid_size_med"),
                (pl.col("yes_ask_size") < 10).mean().alias("ask_size_lt10"),
                (pl.col("yes_ask") - pl.col("yes_bid")).median().alias("spread_med")).sort("series"))
    held = {"KXALBUMEQUIV-FRE26NOV12-65K", "KXALBUMEQUIV-FRE26NOV12-70K"}
    res = {"local_days": local, "archived_days": archived, "ladder_runs": runs, "nov17_books": rows,
           "holdout_fills_recorded": [{"ticker": t, "run": r["run"]} for r in runs for t in r["ladder_markets"] if t in held],
           "forward_snapshot_sizes": fws.to_dicts()}
    _dump(res, "l2.json")
    with pl.Config(tbl_rows=60, tbl_width_chars=250):
        log(pl.DataFrame(rows) if rows else "no Nov 17 books")
        log(fws)
    log(f"local days {local}; archived {archived[:3]}..{archived[-1:]}; holdout fills on the recorder: {res['holdout_fills_recorded']}")
    return res


# ---- check 5: statistics ----------------------------------------------------------------------------
def _t(x: np.ndarray, *groups: np.ndarray) -> tuple[float | None, int]:
    from albert.research.counts import cluster_se

    g = min(len(np.unique(k)) for k in groups)
    if len(x) < 2 or g < 2:
        return None, g
    se = cluster_se(x, *groups)
    return (round(float(x.mean()) / se, 2) if se and se > 1e-12 else None), g


def stats(f: pl.DataFrame) -> dict[str, Any]:
    from scipy import stats as st

    if f.is_empty():
        return {"fills": 0}
    x = f["pnl_c"].to_numpy().astype(float)
    ev = f["event_ticker"].to_numpy().astype(str)
    dt = f["t"].dt.date().cast(pl.String).to_numpy().astype(str)
    alb = np.array([e.split("-", 1)[1] for e in ev])
    t_ev, _g_ev = _t(x, ev)
    t_2, _ = _t(x, ev, dt)
    t_wk, g_wk = _t(x, dt)
    t_alb, _g_alb = _t(x, alb)
    usd = (f["payoff_usd"] - f["cost_usd"]).to_numpy()
    p_wk = float(st.t.sf(t_wk, g_wk - 1)) if t_wk is not None else None
    return {"fills": len(x), "events": len(np.unique(ev)), "album_weeks": len(np.unique(alb)),
            "weeks": len(np.unique(dt)), "c_per_contract": round(100 * float(x.mean()), 2),
            "c_contract_weighted": round(100 * float(usd.sum() / f["n"].sum()), 2),
            "usd_pnl": round(float(usd.sum()), 2), "pnl_per_usd": round(float(usd.sum() / f["cost_usd"].sum()), 3),
            "t_event": t_ev, "t_2way_event_date": t_2, "t_week": t_wk, "t_album_week": t_alb,
            "p_one_sided_week_t_df": None if p_wk is None else round(p_wk, 4),
            "win_rate": round(float((usd > 0).mean()), 3)}


def drop_best(f: pl.DataFrame, k: int = 1) -> pl.DataFrame:
    if f.is_empty():
        return f
    best = (f.group_by("event_ticker").agg((pl.col("payoff_usd") - pl.col("cost_usd")).sum().alias("u"))
            .sort("u", descending=True).head(k)["event_ticker"].to_list())
    return f.filter(~pl.col("event_ticker").is_in(best))


# ---- execution variants ----------------------------------------------------------------------------------
def capped_fills(rows: pl.DataFrame, theta: float, cap: int | dict[str, int]) -> pl.DataFrame:
    """The reviewed taker rule with the order size capped (per series when ``cap`` is a dict); the edge
    test uses the capped order's fee per contract. Re-runs the per-event $5 cap, so more strikes can fit."""
    out = []
    for r in rows.iter_rows(named=True):
        p, bid, ask = r["p_model"], r["bid"], r["ask"]
        if p is None or bid is None or ask is None:
            continue
        c = cap if isinstance(cap, int) else cap.get(r["series"], 25)
        cands = []
        if 0 < ask < 1:
            n = min(mc.order_size(ask), c)
            fee = mc.order_fee(n, ask)
            cands.append(("yes", ask, n, fee, p - ask - fee / n))
        if 0 < bid < 1:
            q = 1 - bid
            n = min(mc.order_size(q), c)
            fee = mc.order_fee(n, q)
            cands.append(("no", q, n, fee, (1 - p) - q - fee / n))
        best = max(cands, key=lambda z: z[4], default=None)
        if best is None or best[4] < theta:
            continue
        side, price, n, fee, edge = best
        win = r["y"] if side == "yes" else 1 - r["y"]
        out.append({**r, "side": side, "price": price, "n": n, "fee": fee, "edge": edge, "pnl_c": win - price - fee / n,
                    "cost_usd": n * price + fee, "payoff_usd": float(n * win)})
    return mc.cap_event(pl.DataFrame(out, infer_schema_length=None)) if out else pl.DataFrame()


def rescale_same_fills(f: pl.DataFrame, cap: int) -> pl.DataFrame:
    """The same fills, each order shrunk to at most ``cap`` contracts (fee re-rounded per order)."""
    if f.is_empty():
        return f
    n = f["n"].clip(upper_bound=cap)
    fee = [mc.order_fee(int(k), float(pr)) for k, pr in zip(n.to_list(), f["price"].to_list())]
    win = (f["payoff_usd"] / f["n"]).to_list()
    return f.with_columns(n.alias("n"), pl.Series("fee", fee),
                          (pl.Series(win) - f["price"] - pl.Series(fee) / n).alias("pnl_c"),
                          (n * f["price"] + pl.Series(fee)).alias("cost_usd"), (n * pl.Series(win)).alias("payoff_usd"))


# ---- corrections: the table as first archived, the visibility lag, a leave-one-week-out fit ---------------
def first_archived_tables() -> dict[str, pl.DataFrame]:
    """For every traded post whose earliest Wayback copy differs from the table the post renders today,
    that earliest copy's table (the closest record of what was public around the decision)."""
    tp = traded_posts(panel())
    caps = _captures()
    items = pl.read_parquet(SRC / "hits_items.parquet")
    out = {}
    for w in tp.iter_rows(named=True):
        first = evidence_plan(w, caps)["post_first"]
        raw = _cached(first[0]) if first else None
        if not raw:
            continue
        tab = table_of(raw, w["slug"])
        cmp = compare_tables(tab, items.filter(pl.col("slug") == w["slug"]))
        if cmp["identical"] is not None and cmp["identical"] < min(cmp["rows_a"], cmp["rows_b"]):
            out[w["slug"]] = tab
    return out


def reprice(p: pl.DataFrame, slug: str, table: pl.DataFrame) -> pl.DataFrame:
    """Every panel row priced from ``slug`` re-priced from ``table``; an event the table does not list is
    dropped (no HITS number for it at the time)."""
    pub = pl.read_parquet(SRC / "hits_posts.parquet").filter(pl.col("slug") == slug)["published_at"][0]
    we = mc.week_end_for_post(pub, "preview")
    hits = table.with_columns(pl.lit(pub).alias("published_at"), pl.lit("preview").alias("kind"), pl.lit(we).alias("week_end"))
    ev = mc.ladder_events(mc.load_markets(SRC)).filter(pl.col("week_end") == we)
    ev = ev.join(mc.match_hits(ev.select("week_end", "album", "artist"), hits), on=["week_end", "album", "artist"], how="left")
    hv = {}
    for e in ev.iter_rows(named=True):
        if e["h_album"] is None:
            continue
        row = hits.filter((pl.col("album") == e["h_album"]) & (pl.col("artist") == e["h_artist"])).row(0, named=True)
        v = mc.hits_value(row, e["measure"])
        if v:
            hv[e["event_ticker"]] = v
    sel = pl.col("hits_pub") == pub
    wk, others = p.filter(sel), p.filter(~sel)
    wk = (wk.with_columns(pl.col("event_ticker").replace_strict(hv, default=None, return_dtype=pl.Float64).alias("hval"))
          .filter(pl.col("hval").is_not_null()))
    if not wk.is_empty():
        wk = wk.with_columns(pl.struct("floor_strike", "hval", "mu", "sd").map_elements(
            lambda r: mc.p_above(r["floor_strike"], r["hval"], r["mu"], r["sd"]), return_dtype=pl.Float64).alias("p_model"))
    return pl.concat([others, wk.select(others.columns)])


def corrected_panel(p: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    tabs = first_archived_tables()
    for slug, tab in tabs.items():
        p = reprice(p, slug, tab)
    return p, sorted(tabs)


def lag_rows(p: pl.DataFrame, min_lag_min: float) -> pl.DataFrame:
    """Act at the first whole hour at least ``min_lag_min`` after the stamp: the preview_fast row when its
    lag qualifies, else the next hour (preview_first). Same signal, same theta: an execution change."""
    fast = p.filter(pl.col("decision") == DECISION)
    ok = fast.filter((pl.col("t") - pl.col("hits_pub")).dt.total_minutes() >= min_lag_min)["event_ticker"].unique().implode()
    return pl.concat([fast.filter(pl.col("event_ticker").is_in(ok)),
                      p.filter((pl.col("decision") == "preview_first") & ~pl.col("event_ticker").is_in(ok))])


def lowo(rows: pl.DataFrame) -> pl.DataFrame:
    """Dev rows re-priced with mu / sd refit on the other dev weeks (the reviewed dev fit is in-sample)."""
    pairs = pl.read_parquet(SRC / "calibration_pairs.parquet").filter(pl.col("split") == "dev")
    out = []
    for (we,), g in rows.with_columns(pl.col("hits_pub").map_elements(lambda d: mc.week_end_for_post(d, "preview"),
                                                                        return_dtype=pl.Date).alias("we")).group_by("we"):
        fit = mc.fit_ratio(pairs.filter(pl.col("week_end") != we))
        g = g.with_columns(pl.struct("floor_strike", "hval", "measure").map_elements(
            lambda s, fit=fit: mc.p_above(s["floor_strike"], s["hval"], fit[f"{s['measure']}|preview"]["mu"],
                                          fit[f"{s['measure']}|preview"]["sd"]) if f"{s['measure']}|preview" in fit else None,
            return_dtype=pl.Float64).alias("p_model"))
        out.append(g.drop("we"))
    return pl.concat(out)


def run(rows: pl.DataFrame, split: str, *, cap: int = 25, series: str | None = None, theta: float = THETA) -> pl.DataFrame:
    r = rows.filter(pl.col("split") == split)
    if series:
        r = r.filter(pl.col("series") == series)
    return capped_fills(r, theta, cap)


def score(log=print) -> dict[str, Any]:
    from scipy import stats as st

    p = panel()
    pc, fixed = corrected_panel(p)
    res: dict[str, Any] = {"weeks_repriced_from_wayback": fixed}
    fast = lambda q: q.filter(pl.col("decision") == DECISION)
    rows = {"reviewed": fast(p), "A_tables": fast(pc), "A_lag45": lag_rows(pc, 45), "A_lag60": pc.filter(pl.col("decision") == "preview_first")}
    fills: dict[str, dict[str, pl.DataFrame]] = {}
    for name, r in rows.items():
        fills[name] = {s: run(r, s) for s in ("dev", "holdout")}
        fills[f"{name}+equiv"] = {s: run(r, s, series="KXALBUMEQUIV") for s in ("dev", "holdout")}
        fills[f"{name}+cap10"] = {s: run(r, s, cap=10) for s in ("dev", "holdout")}
        fills[f"{name}+equiv+cap10"] = {s: run(r, s, cap=10, series="KXALBUMEQUIV") for s in ("dev", "holdout")}
    for name in ("A_tables", "A_lag45"):
        fills[f"{name}+spread<=10c"] = {s: f if f.is_empty() else f.filter((pl.col("ask") - pl.col("bid")) <= 0.10 + 1e-9)
                                         for s, f in fills[name].items()}
        fills[f"{name}+lowo_fit"] = {"dev": run(lowo(rows[name].filter(pl.col("split") == "dev")), "dev")}
    # sanity: the reviewed numbers reproduce
    assert stats(fills["reviewed"]["dev"])["fills"] == 60 and stats(fills["reviewed"]["holdout"])["fills"] == 15
    res["variants"] = {}
    for name, d in fills.items():
        res["variants"][name] = {s: stats(f) for s, f in d.items()}
        if set(d) == {"dev", "holdout"}:
            res["variants"][name]["pooled"] = stats(pl.concat([d["dev"], d["holdout"]], how="diagonal_relaxed"))
    # what changed in the re-priced weeks
    res["repriced_fills"] = {}
    for s in ("dev", "holdout"):
        a, b = fills["reviewed"][s], fills["A_tables"][s]
        cols = ["ticker", "side", "price", "n", "p_model", "pnl_c"]
        res["repriced_fills"][s] = {"removed": a.join(b.select("ticker"), on="ticker", how="anti").select(cols).to_dicts(),
                                    "added": b.join(a.select("ticker"), on="ticker", how="anti").select(cols).to_dicts()}
    # dependence and fragility
    for name in ("reviewed", "A_tables", "A_lag45"):
        h = fills[name]["holdout"].with_columns(pl.col("event_ticker").str.split("-").list.get(1).alias("album_week"))
        dep = h.group_by("album_week").agg(pl.len().alias("fills"), pl.col("ticker")).filter(pl.col("fills") > 1)
        res.setdefault("holdout_dependence", {})[name] = {
            "fills": h.height, "album_weeks": h["album_week"].n_unique(), "decision_hours": h["t"].n_unique(),
            "fills_sharing_album_week": int(dep["fills"].sum()) if dep.height else 0, "groups": dep.to_dicts()}
        res.setdefault("drop_best_event", {})[name] = {
            s: {f"minus_{k}": stats(drop_best(f, k)) for k in (1, 2, 3)} for s, f in fills[name].items()}
    # multiple testing on the selected dev cell
    mt = {}
    for name in ("reviewed", "A_tables", "A_lag45"):
        sd = res["variants"][name]["dev"]
        for tname, tv, df in (("t_2way_normal", sd["t_2way_event_date"], None), ("t_week_tdf", sd["t_week"], sd["weeks"] - 1)):
            if tv is None:
                continue
            p1 = float(st.norm.sf(tv)) if df is None else float(st.t.sf(tv, df))
            mt[f"{name}|{tname}"] = {"t": tv, "p_one_sided": round(p1, 6), "holm_bonferroni_83": round(min(1.0, p1 * CELLS_TRIED), 4),
                                     "holm_bonferroni_72": round(min(1.0, p1 * 72), 4)}
    mt["t_needed_bonferroni_83_one_sided_normal"] = round(float(st.norm.isf(0.05 / CELLS_TRIED)), 2)
    mt["t_needed_bonferroni_83_one_sided_t18"] = round(float(st.t.isf(0.05 / CELLS_TRIED, 18)), 2)
    res["multiple_testing"] = mt
    # book metrics (vol_metrics: capped $100 book, <= $5 per event, daily marks at the hourly-candle mid)
    book_set = ["reviewed", "A_tables", "A_lag45", "A_tables+equiv", "A_lag45+equiv", "A_lag45+equiv+cap10"]
    tick = sorted({t for n in book_set for f in fills[n].values() if not f.is_empty() for t in f["ticker"].to_list()})
    cand = mc.load_candles(tick)
    res["book"] = {}
    for n in book_set:
        for s, f in fills[n].items():
            led = mc.to_ledger(f, "music_charts_review", n, s)
            res["book"][f"{n}|{s}"] = mc.book_metrics(led, cand) if not led.is_empty() else {}
    _dump(res, "score.json")
    keys = ["fills", "events", "weeks", "c_per_contract", "usd_pnl", "t_2way_event_date", "t_week", "t_album_week"]
    for name, d in res["variants"].items():
        log(name.ljust(26), " | ".join(f"{s}: " + " ".join(str(v.get(k)) for k in keys) for s, v in d.items()))
    log(json.dumps({k: res[k] for k in ("weeks_repriced_from_wayback", "repriced_fills", "holdout_dependence", "multiple_testing")}, default=str, indent=1))
    for k, v in res["book"].items():
        log(k.ljust(34), {kk: v.get(kk) for kk in ("sharpe", "sharpe_ci90", "sharpe_ex_top5", "ann_vol_pct", "max_dd_usd", "usd_per_week")})
    return res


# ---- check 6: does the market move toward the HITS model after the decision? -------------------------
def _mid_at(c: pl.DataFrame, req: pl.DataFrame, col: str) -> pl.DataFrame:
    q = mc.quotes_at(c, req.rename({col: "t"}))
    two = q["bid"].is_not_null() & q["ask"].is_not_null() & (q["ask"] > q["bid"])
    q = q.with_columns(pl.when(two).then((pl.col("bid") + pl.col("ask")) / 2).otherwise(None).alias(f"mid_{col}"),
                       (pl.col("ask") - pl.col("bid")).alias(f"spr_{col}"))
    return q.select("row_id", f"mid_{col}", f"spr_{col}")


def reaction(p: pl.DataFrame, c: pl.DataFrame) -> dict[str, Any]:
    """When does the market move toward the HITS model? Per traded week, the mean signed mid move in each
    hour around the stamp (sign: model vs mid at the stamp's floor hour), and the move in the hour ending
    at the decision vs the hour after, split by stamp-to-decision lag. A post that goes live later than its
    stamp shows up as a move after t in the short-lag weeks."""
    q = p.with_columns(pl.col("hits_pub").dt.truncate("1h").alias("h0"))
    base = q.select("row_id", "split", "t", "hits_pub", "h0", "p_model")
    offs = list(range(-2, 5))
    for k in offs:
        base = base.join(_mid_at(c, q.select("row_id", "ticker", (pl.col("h0") + timedelta(hours=k)).alias(f"h{k}")), f"h{k}"),
                         on="row_id", how="left")
    gap = pl.col("p_model") - pl.col("mid_h0")
    b = base.filter(pl.col("mid_h0").is_not_null() & (gap.abs() >= 0.05) & (pl.col("spr_h0") <= 0.25)).with_columns(gap.sign().alias("s"))
    weeks = []
    for (d,), g in sorted(b.group_by(pl.col("t").dt.date()), key=lambda x: x[0]):
        t0, h0, pub = g["t"][0], g["h0"][0], g["hits_pub"][0]
        moves = {}
        for k in offs[:-1]:
            x = g.filter(pl.col(f"mid_h{k}").is_not_null() & pl.col(f"mid_h{k + 1}").is_not_null())
            v = (x["s"] * (x[f"mid_h{k + 1}"] - x[f"mid_h{k}"])).mean()
            rel = int((h0 + timedelta(hours=k + 1) - t0).total_seconds() // 3600)     # 0 = hour ending at t
            moves[rel] = None if v is None else round(100 * float(v), 2)
        weeks.append({"date": str(d), "split": g["split"][0], "lag_min": round((t0 - pub).total_seconds() / 60, 1),
                      "rows": g.height, "move_c_by_hour_end_rel_t": moves})
    groups = {}
    for name, cond in (("lag<40min", lambda w: w["lag_min"] < 40), ("lag>=40min", lambda w: w["lag_min"] >= 40)):
        ws = [w for w in weeks if cond(w)]
        groups[name] = {"weeks": len(ws), **{f"hour_end_t{h:+d}": round(float(np.mean([w["move_c_by_hour_end_rel_t"].get(h) or 0 for w in ws])), 2)
                                             for h in (-1, 0, 1, 2)}}
    return {"weeks": weeks, "by_lag": groups}


def drift(log=print) -> dict[str, Any]:
    from albert.research.counts import cluster_se

    p = corrected_panel(panel())[0].filter(pl.col("decision") == DECISION).with_row_index("row_id")
    c = mc.load_candles(p["ticker"].unique().to_list())
    p = p.with_columns(pl.col("hits_pub").dt.truncate("1h").alias("pub_floor"))
    anchors = {"pre2": pl.col("pub_floor") - timedelta(hours=2), "pre1": pl.col("pub_floor") - timedelta(hours=1),
               "pre0": pl.col("pub_floor"), "dec": pl.col("t")}
    delays = (1, 2, 3, 6, 12, 24, 36)
    anchors |= {f"d{h}": pl.col("t") + timedelta(hours=h) for h in delays}
    base = p.select("row_id", "event_ticker", "ticker", "split", "t", "p_model", "close_time", "y")
    for name, expr in anchors.items():
        req = p.select("row_id", "ticker", expr.alias(name))
        base = base.join(_mid_at(c, req, name), on="row_id", how="left")
    res: dict[str, Any] = {}
    for split in ("dev", "holdout", "all"):
        b = base if split == "all" else base.filter(pl.col("split") == split)
        for sign_at in ("dec", "pre0", "pre2"):
            gap = pl.col("p_model") - pl.col(f"mid_{sign_at}")
            s = b.filter(pl.col(f"mid_{sign_at}").is_not_null() & (gap.abs() >= 0.05) & (pl.col(f"spr_{sign_at}") <= 0.25))
            s = s.with_columns(gap.sign().alias("sgn"), gap.abs().alias("gap"))
            out = {"rows": s.height, "events": s["event_ticker"].n_unique(), "gap_c": round(100 * float(s["gap"].mean()), 2) if s.height else None}
            legs = [("pre2->pre1", "pre2", "pre1"), ("pre1->pre0", "pre1", "pre0"), ("pre2->pre0", "pre2", "pre0"),
                    ("pre0->dec", "pre0", "dec")]
            legs += [(f"dec->+{h}h", "dec", f"d{h}") for h in delays]
            for lab, a, z in legs:
                ahead = int(z[1:]) if re.fullmatch(r"d\d+", z) else 0
                x = s.filter(pl.col(f"mid_{a}").is_not_null() & pl.col(f"mid_{z}").is_not_null()
                             & (pl.col("close_time") > pl.col("t") + timedelta(hours=ahead)))
                if x.height < 5:
                    continue
                v = (x["sgn"] * (x[f"mid_{z}"] - x[f"mid_{a}"])).to_numpy()
                ev = x["event_ticker"].to_numpy()
                dt = x["t"].dt.date().cast(pl.String).to_numpy()
                se = cluster_se(v, ev, dt)
                out[lab] = {"n": len(v), "move_c": round(100 * float(v.mean()), 2),
                            "t_2way": round(float(v.mean()) / se, 2) if se and se > 0 else None,
                            "share_of_gap": round(float(v.sum() / x["gap"].sum()), 3)}
            res[f"{split}|sign@{sign_at}"] = out
    res["reaction_by_week"] = reaction(p, c)
    _dump(res, "drift.json")
    log(json.dumps(res["reaction_by_week"]["by_lag"]))
    for k, v in res.items():
        if k == "reaction_by_week":
            continue
        log(k, {kk: (vv if not isinstance(vv, dict) else f"{vv['move_c']:+.1f}c t{vv['t_2way']} n{vv['n']} {100 * vv['share_of_gap']:.0f}%")
                for kk, vv in v.items()})
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.music_charts_review")
    ap.add_argument("cmd", choices=["wayback", "timing", "candles", "settle", "l2", "score", "drift"])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    {"wayback": wayback, "timing": timing, "candles": candles_check, "settle": settle, "l2": l2, "score": score,
     "drift": drift}[a.cmd]()


if __name__ == "__main__":
    main()
