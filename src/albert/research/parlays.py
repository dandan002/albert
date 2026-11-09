"""Kalshi parlay (multivariate "combo") pricing study. Public REST data only; no orders.

Questions:
  1. Do combo takers overpay? Per trade, the maker's PnL per contract = p - y if the taker
     bought YES at p (maker holds NO), y - p if the taker bought NO. Clustered by game and day.
  2. Is the premium mispriced correlation or markup? q = product of the legs' side-adjusted
     mids from the last complete 1-min candle before the trade (no look-ahead). Then
     p - y = (p - q) + (q - y): (q - y) < 0 means the legs are positively correlated.

Usage: python -m albert.research.parlays collect OUTDIR   (writes trades.parquet, meta.json, log)
       python -m albert.research.parlays report OUTDIR    (writes a Report and summary.json)
"""
import collections
import datetime as dt
import json
import math
import random
import sys
import time
from pathlib import Path

import httpx
import polars as pl

R = "https://api.elections.kalshi.com/trade-api/v2"
SEED = 0
GROUPS = {
    # name: (hist series, recent-settled series, max pages per listing, clustered by game?)
    "nba_single_game": (["KXMVENBASINGLEGAME"], ["KXMVENBASINGLEGAME"], 400, True),
    "nfl_single_game": (["KXMVENFLSINGLEGAME"], ["KXMVENFLSINGLEGAME"], 250, True),
    "multi_game": (["KXMVENFLMULTIGAME", "KXMVENFLMULTIGAMEEXTENDED", "KXMVENBAMULTIGAMEEXTENDED",
                    "KXMVESPORTSMULTIGAMEEXTENDED"],
                   ["KXMVENFLMULTIGAMEEXTENDED", "KXMVENBAMULTIGAMEEXTENDED", "KXMVESPORTSMULTIGAMEEXTENDED"],
                   60, False),
    "cross_category_sports": (["KXMVECROSSCATEGORY", "KXMVECROSSCATEGORY-SHARD1"],
                              ["KXMVECROSSCATEGORY", "KXMVECROSSCATEGORY-SHARD1"], 100, False),
}
GAMES_PER_GROUP = 60          # single-game groups: random games
COMBOS_PER_GAME = 80          # random traded combos per sampled game
COMBOS_OTHER = 2000           # multi-game / cross-category: random traded combos
LEAGUE_HOURS = [("KXNFL", 3.4), ("KXNCAAF", 3.6), ("KXNBA", 2.6), ("KXWNBA", 2.4), ("KXNHL", 2.6),
                ("KXMLB", 3.2), ("KXNCAAMB", 2.2), ("KXNCAAWB", 2.2)]
MAX_STALE_S = 900
if __import__("os").environ.get("SMOKE"):
    GROUPS = {k: (h, r, 2, b) for k, (h, r, _, b) in GROUPS.items()}
    GAMES_PER_GROUP, COMBOS_PER_GAME, COMBOS_OTHER = 2, 4, 8


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ts(s: str) -> float:
    return dt.datetime.fromisoformat(s).timestamp()


class Api:
    def __init__(self, rps: float = 10.0):
        self.c = httpx.Client(timeout=30)
        self.gap = 1.0 / rps
        self.last = 0.0
        self.calls = 0

    def get(self, path, **q):
        for a in range(8):
            w = self.last + self.gap - time.monotonic()
            if w > 0:
                time.sleep(w)
            self.last = time.monotonic()
            self.calls += 1
            try:
                r = self.c.get(R + path, params=q)
            except httpx.HTTPError:
                time.sleep(2 + a)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(1 + a * 2)
                continue
            if r.status_code >= 400:
                return None
            return r.json()
        return None

    def pages(self, path, key, max_pages, **q):
        cur = None
        for _ in range(max_pages):
            if cur:
                q["cursor"] = cur
            j = self.get(path, **q)
            if not j:
                return
            items = j.get(key) or []
            yield from items
            cur = j.get("cursor")
            if not cur or not items:
                return


def compact(m, hist):
    legs = m.get("mve_selected_legs") or []
    sv = m.get("settlement_value_dollars")
    y = float(sv) if sv not in (None, "") else (1.0 if m.get("result") == "yes" else 0.0 if m.get("result") == "no" else None)
    return {"ticker": m["ticker"], "coll": m.get("mve_collection_ticker") or "",
            "legs": [(x["market_ticker"], x["event_ticker"], x["side"]) for x in legs],
            "y": y, "volume": float(m.get("volume_fp") or 0), "open": m.get("open_time"),
            "close": m.get("close_time"), "hist": hist}


def collect(out: Path):
    api = Api()
    rng = random.Random(SEED)
    cut = api.get("/historical/cutoff")
    ts(cut["trades_created_ts"])
    sports = set(pl.read_parquet("data/tables/kalshi_series.parquet")
                 .filter(pl.col("category") == "Sports")["series_ticker"].to_list())
    meta = {"cutoff": cut, "groups": {}}
    sampled: dict[str, list[dict]] = {}
    for g, (hist_s, recent_s, maxp, by_game) in GROUPS.items():
        rows = []
        for s in hist_s:
            n0 = len(rows)
            rows += [compact(m, True) for m in api.pages("/historical/markets", "markets", maxp,
                                                         series_ticker=s, limit=1000)]
            log(g, s, "historical", len(rows) - n0)
        for s in recent_s:
            n0 = len(rows)
            rows += [compact(m, False) for m in api.pages("/markets", "markets", maxp, series_ticker=s,
                                                          status="settled", limit=1000)]
            log(g, s, "recent settled", len(rows) - n0)
        rows = [r for r in rows if r["volume"] > 0 and r["y"] is not None and len(r["legs"]) >= 2
                and all(t.split("-")[0] in sports for t, _, _ in r["legs"])]
        if by_game:
            games = collections.defaultdict(list)
            for r in rows:
                games[r["coll"]].append(r)
            keys = sorted(games)
            pick = rng.sample(keys, min(GAMES_PER_GROUP, len(keys)))
            samp = []
            for k in pick:
                cs = games[k]
                samp += rng.sample(cs, min(COMBOS_PER_GAME, len(cs)))
        else:
            samp = rng.sample(rows, min(COMBOS_OTHER, len(rows)))
        closes = sorted(r["close"] for r in rows if r["close"])
        meta["groups"][g] = {"population_traded_sports_combos": len(rows),
                             "population_games": len({r["coll"] for r in rows}),
                             "close_range": [closes[0], closes[-1]] if closes else None,
                             "sampled_combos": len(samp)}
        log(g, meta["groups"][g])
        sampled[g] = samp

    # trades
    trades = []
    for g, samp in sampled.items():
        for i, r in enumerate(samp):
            path = "/historical/trades" if r["hist"] else "/markets/trades"
            for t in api.pages(path, "trades", 10, ticker=r["ticker"], limit=1000):
                trades.append({"group": g, "ticker": r["ticker"], "t": ts(t["created_time"]),
                               "p": float(t["yes_price_dollars"]), "count": float(t["count_fp"]),
                               "taker_side": t.get("taker_side") or t.get("taker_outcome_side"),
                               "block": bool(t.get("is_block_trade"))})
            if i % 200 == 0:
                log(g, "trades", i, len(samp), "rows", len(trades), "calls", api.calls)
    combo = {r["ticker"]: r for s in sampled.values() for r in s}

    # leg windows and leg markets
    win: dict[str, list[float]] = {}
    leg_hist: dict[str, bool] = {}
    events: dict[str, bool] = {}
    for tr in trades:
        for lt, le, _ in combo[tr["ticker"]]["legs"]:
            w = win.setdefault(lt, [tr["t"], tr["t"]])
            w[0], w[1] = min(w[0], tr["t"]), max(w[1], tr["t"])
            leg_hist[lt] = combo[tr["ticker"]]["hist"]
            events[le] = combo[tr["ticker"]]["hist"]
    log("legs", len(win), "leg events", len(events))
    leg_info = {}
    for i, (e, hist) in enumerate(sorted(events.items())):
        path = "/historical/markets" if hist else "/markets"
        for m in api.pages(path, "markets", 5, event_ticker=e, limit=1000):
            leg_info[m["ticker"]] = {"exp": m.get("expected_expiration_time"), "result": m.get("result")}
        if hist and not any(t.startswith(e + "-") for t in leg_info):   # moved after the cutoff?
            for m in api.pages("/markets", "markets", 5, event_ticker=e, limit=1000):
                leg_info[m["ticker"]] = {"exp": m.get("expected_expiration_time"), "result": m.get("result")}
        if i % 200 == 0:
            log("leg events", i, len(events), "calls", api.calls)

    # leg candles: last complete 1-min candle before each trade
    candles: dict[str, list[tuple[int, float, float]]] = {}
    for i, (lt, (a, b)) in enumerate(sorted(win.items())):
        lo = int(max(a - 1800, b - 7 * 86400)) // 60 * 60
        hi = int(b) // 60 * 60 + 60
        got = []
        s = lo
        while s < hi:
            e = min(hi, s + 4000 * 60)
            paths = ([f"/historical/markets/{lt}/candlesticks"] if leg_hist[lt] else []) + \
                    [f"/series/{lt.split('-')[0]}/markets/{lt}/candlesticks"]
            for path in paths:
                j = api.get(path, start_ts=s, end_ts=e, period_interval=1)
                if j and j.get("candlesticks") is not None:
                    for c in j["candlesticks"]:
                        yb, ya = c.get("yes_bid") or {}, c.get("yes_ask") or {}
                        bid = yb.get("close") or yb.get("close_dollars")
                        ask = ya.get("close") or ya.get("close_dollars")
                        if bid is not None and ask is not None:
                            got.append((int(c["end_period_ts"]), float(bid), float(ask)))
                    break
            s = e
        candles[lt] = sorted(got)
        if i % 300 == 0:
            log("candles", i, len(win), "calls", api.calls)

    import bisect
    rows = []
    for tr in trades:
        r = combo[tr["ticker"]]
        probs, stale, spreads, starts, ok = [], 0.0, 0.0, [], True
        for lt, le, side in r["legs"]:
            cs = candles.get(lt) or []
            k = bisect.bisect_right(cs, (int(tr["t"]), 9.0, 9.0)) - 1
            if k < 0:
                ok = False
                break
            end, bid, ask = cs[k]
            if not (0 < bid <= ask < 1) or tr["t"] - end > MAX_STALE_S:
                ok = False
                break
            mid = (bid + ask) / 2
            probs.append(mid if side == "yes" else 1 - mid)
            stale = max(stale, tr["t"] - end)
            spreads += ask - bid
            info = leg_info.get(lt) or {}
            h = next((h for pre, h in LEAGUE_HOURS if lt.startswith(pre)), None)
            if info.get("exp") and h:
                starts.append(ts(info["exp"]) - h * 3600)
        q = math.prod(probs) if ok else None
        maker_pnl = (tr["p"] - r["y"]) if tr["taker_side"] == "yes" else (r["y"] - tr["p"])
        games = sorted({le.split("-")[1] for _, le, _ in r["legs"] if "-" in le})
        rows.append({**tr, "y": r["y"], "n_legs": len(r["legs"]), "coll": r["coll"],
                     "game": r["coll"].split("-", 1)[1] if GROUPS[tr["group"]][3] else "|".join(games),
                     "day": dt.datetime.fromtimestamp(tr["t"], dt.UTC).strftime("%Y-%m-%d"),
                     "q": q, "leg_spread_sum": spreads if ok else None, "max_stale_s": stale if ok else None,
                     "in_play": (tr["t"] >= min(starts)) if starts else None,
                     "maker_pnl": maker_pnl, "hist": r["hist"]})
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_parquet(out / "trades.parquet")
    meta["calls"] = api.calls
    meta["trades"] = len(df)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log("done", len(df), "trades;", api.calls, "calls")


def cl_ratio(num: list[float], den: list[float], cl: list[str]):
    """Count-weighted mean sum(num)/sum(den) with a cluster-robust SE (ratio estimator)."""
    S, C = collections.defaultdict(float), collections.defaultdict(float)
    for a, b, k in zip(num, den, cl):
        S[k] += a
        C[k] += b
    G = len(S)
    tot_c = sum(C.values())
    if G < 2 or tot_c <= 0:
        return None, None, G
    r = sum(S.values()) / tot_c
    v = sum((S[k] - r * C[k]) ** 2 for k in S) * G / (G - 1)
    se = math.sqrt(v) / tot_c
    return r, (r / se if se > 0 else None), G


def stats(d: pl.DataFrame, cols=("maker_pnl",)) -> dict:
    out = {"trades": d.height, "contracts": round(d["count"].sum(), 1),
           "combos": d["ticker"].n_unique(), "games": d["game"].n_unique(), "days": d["day"].n_unique()}
    if d.height == 0:
        return out
    w = d["count"].to_list()
    for c in cols:
        x = d[c].to_list()
        num = [a * b for a, b in zip(x, w)]
        for k in ("game", "day"):
            m, t, _ = cl_ratio(num, w, d[k].to_list())
            out[f"{c}_per_contract"] = None if m is None else round(m, 4)
            out[f"{c}_t_{k}"] = None if t is None else round(t, 2)
        # per-trade (unweighted) too, clustered by game
        m, t, _g = cl_ratio(x, [1.0] * len(x), d["game"].to_list())
        out[f"{c}_per_trade"] = None if m is None else round(m, 4)
        out[f"{c}_per_trade_t_game"] = None if t is None else round(t, 2)
    out["taker_yes_share"] = round(d.filter(pl.col("taker_side") == "yes")["count"].sum() / max(1e-9, d["count"].sum()), 3)
    out["mean_price"] = round((d["p"] * d["count"]).sum() / d["count"].sum(), 4)
    out["hit_rate"] = round((d["y"] * d["count"]).sum() / d["count"].sum(), 4)
    return out


def report(out: Path):
    from albert.config import settings
    from albert.reports import Report
    df = pl.read_parquet(out / "trades.parquet")
    meta = json.loads((out / "meta.json").read_text())
    df = df.with_columns(
        taker_fee=0.07 * pl.col("p") * (1 - pl.col("p")),
        pq=pl.when(pl.col("taker_side") == "yes").then(pl.col("p") - pl.col("q")).otherwise(pl.col("q") - pl.col("p")),
        qy=pl.when(pl.col("taker_side") == "yes").then(pl.col("q") - pl.col("y")).otherwise(pl.col("y") - pl.col("q")),
        pbucket=(pl.col("p") * 10).floor().clip(0, 9) / 10,
        legs_b=pl.when(pl.col("n_legs") >= 4).then(pl.lit("4+")).otherwise(pl.col("n_legs").cast(pl.Utf8)),
    )
    summ = {"meta": meta, "groups": {}}
    rep = Report(settings().reports_dir, "research-combos")
    rep.json("Run", {"note": "Kalshi multivariate (parlay) trades, public data. maker_pnl: PnL per contract of "
                             "the side opposite the taker, before fees (makers pay 0 on these quadratic series). "
                             "q: product of side-adjusted leg mids from the last complete 1-min candle before "
                             "the trade (<=15 min old). pq = maker's markup over independence; qy = independence "
                             "minus outcome, both signed to the maker. t-stats are cluster-robust by game and by day.",
                     "meta": meta})
    for g in df["group"].unique().sort().to_list():
        d = df.filter(pl.col("group") == g)
        dq = d.filter(pl.col("q").is_not_null())
        s = {"all": stats(d), "with_leg_prices": stats(dq, ("maker_pnl", "pq", "qy")),
             "pregame": stats(dq.filter(pl.col("in_play") == False)),
             "in_play": stats(dq.filter(pl.col("in_play") == True)),
             "by_legs": {k: stats(d.filter(pl.col("legs_b") == k)) for k in d["legs_b"].unique().sort().to_list()},
             "by_price": {f"{k:.1f}": stats(d.filter(pl.col("pbucket") == k)) for k in d["pbucket"].unique().sort().to_list()},
             "taker_yes": stats(d.filter(pl.col("taker_side") == "yes")),
             "taker_no": stats(d.filter(pl.col("taker_side") == "no"))}
        if dq.height:
            s["brier_price"] = round(((dq["p"] - dq["y"]) ** 2).mean(), 4)
            s["brier_independence"] = round(((dq["q"] - dq["y"]) ** 2).mean(), 4)
            s["mean_leg_spread_sum"] = round(dq["leg_spread_sum"].mean(), 4)
        summ["groups"][g] = s
        rep.json(g, s)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(rep.save())


if __name__ == "__main__":
    cmd, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    {"collect": collect, "report": report}[cmd](out)
