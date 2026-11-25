"""Literature sweep (research swarm 2026-11-23, slug ``lit_sweep``).

Mine 2024-2026 academic and practitioner work on prediction-market anomalies, tabulate the
claims, and test the most promising ones that this repo has not tested yet on Kalshi data.

Commands (``python -m albert.research.lit_sweep <cmd>``):

- ``robots``: fetch each literature host's own robots.txt and log a verdict for the paths read
  (``albert.research.counts.robots_verdict``; 403/error = blocked, 404 = allowed).
"""

from __future__ import annotations

import glob
import itertools
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.research.counts import AGENTS, robots_verdict

OUT = Path("data/research/lit_sweep")
CACHE = Path("data/cache/lit_sweep")
UA = "albert-research/0.2"

# host -> paths read (or considered) for the survey
LIT_HOSTS: dict[str, list[str]] = {
    "arxiv.org": ["/abs/2602.19520", "/pdf/2607.14430", "/abs/2607.14430", "/html/2605.31431v1",
                  "/abs/2607.08199", "/abs/2609.12878", "/abs/2109.09871"],
    "www2.gwu.edu": ["/~forcpgm/2026-001.pdf"],
    "www.karlwhelan.com": ["/Papers/Kalshi.pdf"],
    "www.ucd.ie": ["/economics/t4media/WP2025_19.pdf"],
    "www.ifo.de": ["/DocDL/cesifo1_wp12122.pdf"],
    "cepr.org": ["/voxeu/columns/economics-kalshi-prediction-market", "/publications/dp21615"],
    "kalshi.com": ["/research/kalshi-research-calibration.pdf", "/research/publications/calibration",
                   "/research/kalshi-research-who-calibrates-markets.pdf"],
    "law.stanford.edu": ["/publications/adverse-selection-in-prediction-markets-evidence-from-kalshi/"],
    "conference.nber.org": ["/conf_papers/f243677.pdf"],
    "www.nber.org": ["/papers/w35846", "/papers/w35845"],
    "papers.ssrn.com": ["/sol3/papers.cfm"],
    "ideas.repec.org": ["/p/cpr/ceprdp/21615.html"],
    "econpapers.repec.org": ["/RePEc:cpr:ceprdp:20631"],
    "quantpedia.com": ["/exploiting-mean-reversion-in-decentralized-prediction-markets-evidence-from-polymarket-binary-contracts/"],
    "whirligigbear.substack.com": ["/p/so-called-bonds-in-prediction-markets"],
    "decrypt.co": ["/364381/polymarket-bot-bets-no-has-a-point"],
    "dev.to": ["/edgelab/the-polymarket-paradox-what-2490-sports-prediction-markets-reveal-about-bookmaker-edge-17k8"],
    "rooseveltinstitute.org": ["/blog/since-kalshis-launch-ordinary-users-have-lost-half-a-billion-dollars/"],
    "www.alphaxiv.org": ["/abs/2607.08199"],
    "osf.io": ["/preprints/socarxiv/d5yx2"],
    "insights.som.yale.edu": ["/insights/wisdom-of-the-few-prediction-markets-are-driven-by-small-number-of-skilled-traders"],
    "finpolicy.georgetown.edu": ["/wp-content/uploads/2026/07/Wisdom-of-Crowds-Rise-of-Prediction-Markets-.docx.pdf"],
    "mpra.ub.uni-muenchen.de": ["/126350/"],
    "www.kucoin.com": ["/news/flash/analysis-reveals-systematic-wealth-transfer-in-prediction-markets"],
    "news.kalshi.com": ["/p/trading-the-weather"],
    "www.cs.utexas.edu": ["/~kavish/blog/kalshi-weather-calibration.html"],
    "ppc.land": ["/prediction-markets-show-no-partisan-bias-in-data-since-1880-nber-paper-finds/"],
    "drops.dagstuhl.de": ["/storage/00lipics/lipics-vol354-aft2025/html/LIPIcs.AFT.2025.27/LIPIcs.AFT.2025.27.html"],
}


def check_hosts(hosts: dict[str, list[str]] | None = None, out: Path = OUT) -> list[dict[str, Any]]:
    """Fetch each host's own robots.txt now (no cache) and evaluate every path for every agent.

    Policy (swarm rules): 200 -> parse; 404/410 -> no file, allowed; 403, other status or a network
    error -> blocked. One request per host, 1 s apart.
    """
    import httpx

    hosts = hosts or LIT_HOSTS
    rows: list[dict[str, Any]] = []
    d = out / "robots"
    d.mkdir(parents=True, exist_ok=True)
    for host, paths in hosts.items():
        url = f"https://{host}/robots.txt"
        try:
            r = httpx.get(url, timeout=20, follow_redirects=True, headers={"User-Agent": UA})
            status, text = r.status_code, (r.text if r.status_code == 200 else "")
        except httpx.HTTPError as error:
            status, text = None, ""
            print(f"robots {host}: {error}", flush=True)
        (d / f"{host}.txt").write_text(text)
        for p in paths:
            v = robots_verdict(text, p) if status in (200, 404, 410) else dict.fromkeys(AGENTS, False)
            rows.append({"host": host, "path": p, "http_status": status,
                         "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                         **{f"allow_{a}": v[a] for a in AGENTS}, "allowed": all(v.values())})
        time.sleep(1.0)
    (out / "robots.json").write_text(json.dumps(rows, indent=1))
    return rows


# ---- shared stats ----------------------------------------------------------------------------------
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
HOLDOUT_TO = datetime(2026, 11, 23, tzinfo=UTC)


def logit(p: np.ndarray, eps: float = 0.005) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def weighted_logistic(x: np.ndarray, y: np.ndarray, w: np.ndarray, iters: int = 60) -> tuple[float, float]:
    """Platt recalibration y ~ sigmoid(a + b * x) with case weights ``w`` (Newton steps).

    Returns (a, b). b > 1: prices too close to 50c (under-confident); b < 1: too extreme."""
    beta = np.zeros(2)
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        g = X.T @ (w * (y - mu))
        H = (X * (w * mu * (1 - mu))[:, None]).T @ X
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    return float(beta[0]), float(beta[1])


def platt_by_cluster(cells: pl.DataFrame, *, weight: str = "w", reps: int = 200, seed: int = 11) -> dict[str, Any]:
    """Platt slope on pre-aggregated rows (cluster, p, y, w) with a cluster bootstrap SE.

    Every row of one cluster (one game / one market) is resampled together."""
    if cells.height == 0:
        return {"rows": 0}
    x = logit(cells["p"].to_numpy())
    y = cells["y"].to_numpy().astype(float)
    w = cells[weight].to_numpy().astype(float)
    a, b = weighted_logistic(x, y, w)
    _, inv = np.unique(cells["cluster"].to_numpy(), return_inverse=True)
    k = inv.max() + 1
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(reps):
        draw = np.bincount(rng.integers(0, k, size=k), minlength=k)
        ww = w * draw[inv]
        if ww.sum() <= 0:
            continue
        bs.append(weighted_logistic(x, y, ww)[1])
    se = float(np.std(bs, ddof=1)) if len(bs) > 1 else float("nan")
    return {"clusters": int(k), "obs": float(w.sum()), "a": round(a, 3), "b": round(b, 3),
            "b_se": round(se, 3), "t_b_vs_1": round((b - 1) / se, 2) if se and se > 0 else None}


def band_rates(cells: pl.DataFrame, bands: list[tuple[float, float]], weight: str = "w") -> list[dict[str, Any]]:
    """Weighted mean price and win rate per price band (the 'step' picture)."""
    out = []
    for lo, hi in bands:
        c = cells.filter((pl.col("p") >= lo) & (pl.col("p") < hi))
        if c.height == 0:
            out.append({"band": f"{lo:.2f}-{hi:.2f}", "obs": 0})
            continue
        w = c[weight].to_numpy()
        out.append({"band": f"{lo:.2f}-{hi:.2f}", "obs": float(w.sum()), "clusters": c["cluster"].n_unique(),
                    "mean_p": round(float((c["p"].to_numpy() * w).sum() / w.sum()), 3),
                    "win_rate": round(float((c["y"].to_numpy() * w).sum() / w.sum()), 3)})
    return out


# ---- T1: Moshrefi near-expiry calibration in sports moneylines --------------------------------------
INPLAY_TRADES = "data/cache/inplay_dynamics_trades/*.parquet"
GAME_SERIES = ("KXNFLGAME", "KXNCAAFGAME", "KXMLBGAME", "KXNBAGAME", "KXNHLGAME", "KXWNBAGAME")
START_OFFSET_MIN = {"KXNBAGAME": 165}          # scheduled start = expected_expiration - offset
DEFAULT_START_OFFSET_MIN = 180
TTE_EDGES = [0, 10, 30, 90, 240]               # Moshrefi's buckets (minutes)
STEP_BANDS = [(0.01, 0.10), (0.10, 0.40), (0.40, 0.60), (0.60, 0.90), (0.90, 1.00)]


def game_markets(tickers: list[str] | None = None) -> pl.DataFrame:
    m = (pl.scan_parquet("data/tables/kalshi_markets.parquet")
         .filter(pl.col("series_ticker").is_in(list(GAME_SERIES)) & pl.col("result").is_in(["yes", "no"])
                 & pl.col("expected_expiration_time").is_not_null())
         .select("ticker", "event_ticker", "series_ticker", "close_time", "expected_expiration_time", "result",
                 "fee_type", "fee_multiplier"))
    if tickers is not None:
        m = m.filter(pl.col("ticker").is_in(tickers))
    off = pl.col("series_ticker").replace_strict(START_OFFSET_MIN, default=DEFAULT_START_OFFSET_MIN,
                                                 return_dtype=pl.Int64)
    return m.collect().with_columns(
        (pl.col("expected_expiration_time") - pl.duration(minutes=off)).alias("sched_start"),
        (pl.col("result") == "yes").cast(pl.Int8).alias("y"),
        pl.when(pl.col("expected_expiration_time") >= HOLDOUT_FROM).then(pl.lit("holdout"))
        .otherwise(pl.lit("dev")).alias("split"))


def inplay_tickers() -> list[str]:
    return sorted(Path(p).name[:-8] for p in glob.glob(INPLAY_TRADES))


def t1_trades(markets: pl.DataFrame, files: list[str] | None = None) -> pl.LazyFrame:
    """In-play sample prints (window end .. close + 2 min) with every clock attached, before close."""
    m = markets.lazy().select("ticker", "series_ticker", "split", "close_time", "expected_expiration_time",
                              "sched_start", "y", "fee_type", "fee_multiplier")
    src = files if files is not None else INPLAY_TRADES
    t = (pl.scan_parquet(src).select("ticker", "created_time", "count", "yes_price", "taker_side")
         .join(m, on="ticker", how="inner")
         .filter((pl.col("created_time") < pl.col("close_time")) & pl.col("yes_price").is_between(0.01, 0.99)))
    mins = 60_000_000.0
    return t.with_columns(
        ((pl.col("close_time") - pl.col("created_time")).dt.total_microseconds() / mins).alias("tte_actual"),
        ((pl.col("expected_expiration_time") - pl.col("created_time")).dt.total_microseconds() / mins).alias("tte_sched"),
        ((pl.col("created_time") - pl.col("sched_start")).dt.total_microseconds() / mins).alias("elapsed"),
        ((pl.col("close_time") - pl.col("sched_start")).dt.total_microseconds() / mins).alias("game_len"),
    )


def _bucket(col: str, edges: list[float], name: str) -> pl.Expr:
    labels = [f"<{edges[0]}"] + [f"[{a},{b})" for a, b in itertools.pairwise(edges)] + [f">={edges[-1]}"]
    return pl.col(col).cut(edges, labels=labels, left_closed=True).cast(pl.String).alias(name)


def t1_cells(markets: pl.DataFrame, lengths: dict[str, tuple[float, int]],
             files: list[str] | None = None) -> pl.LazyFrame:
    """Prints aggregated to (game, clock bucket, price cent) with trade and contract weights.

    ``late`` is a pre-known clock: minutes since the scheduled start >= the league's dev median game
    length - 30 (``lengths`` from ``league_lengths('dev')``)."""
    L = pl.col("series_ticker").replace_strict({k: v[0] for k, v in lengths.items()}, default=None,
                                               return_dtype=pl.Float64)
    t = t1_trades(markets, files).with_columns(
        _bucket("tte_actual", TTE_EDGES, "b_actual"),
        _bucket("tte_sched", TTE_EDGES, "b_sched"),
        _bucket("elapsed", [0, 60, 120, 150, 180, 210], "b_elapsed"),
        (pl.col("yes_price") * 100).round(0).cast(pl.Int32).alias("pc"),
        (pl.col("tte_actual") < 10).alias("ends_10"),
        (pl.col("elapsed") >= L - 30).fill_null(False).alias("late"))
    keys = ["ticker", "series_ticker", "split", "y", "pc", "b_actual", "b_sched", "b_elapsed", "ends_10", "late"]
    return (t.group_by(keys).agg(pl.len().alias("trades"), pl.col("count").sum().alias("contracts"))
            .with_columns((pl.col("pc") / 100).alias("p"), pl.col("ticker").alias("cluster")))


def _chunks(files: list[str], k: int = 40) -> list[list[str]]:
    return [files[i:i + k] for i in range(0, len(files), k)]


def t1_cells_chunked(markets: pl.DataFrame, lengths: dict[str, tuple[float, int]]) -> pl.DataFrame:
    """``t1_cells`` over the per-game files in chunks (keeps memory under the throttle cap)."""
    files = sorted(glob.glob(INPLAY_TRADES))
    parts = []
    for ch in _chunks(files):
        parts.append(t1_cells(markets, lengths, ch).collect())
    return pl.concat(parts)


# The market table has no settled NBA game before 2026-10-01, so the NBA length is set a priori
# (declared before any NBA holdout print was opened): 138 min from the scheduled tip
# (= expected_expiration - 165 min), a typical 2 h 15 min NBA game.
PRIOR_LENGTHS = {"KXNBAGAME": (138.0, 0)}


def league_lengths(split: str = "dev") -> dict[str, tuple[float, int]]:
    """Median game length (actual close - scheduled start, minutes) per series from the whole
    market table, settled games with scheduled expiration before the holdout only."""
    m = game_markets().filter(pl.col("split") == split)
    g = (m.unique("event_ticker").with_columns(((pl.col("close_time") - pl.col("sched_start")).dt.total_seconds() / 60)
                                              .alias("len_min"))
         .filter(pl.col("len_min").is_between(60, 400))
         .group_by("series_ticker").agg(pl.col("len_min").median(), pl.len().alias("games")))
    out = {r["series_ticker"]: (round(r["len_min"], 1), r["games"]) for r in g.iter_rows(named=True)}
    return {**PRIOR_LENGTHS, **out}


def t1_calibration(cells: pl.DataFrame) -> dict[str, Any]:
    """Platt slope and step bands by clock bucket, trade- and contract-weighted, dev / holdout."""
    out: dict[str, Any] = {}
    for split in ("dev", "holdout", "all"):
        c0 = cells if split == "all" else cells.filter(pl.col("split") == split)
        for clock in ("b_actual", "b_sched", "b_elapsed"):
            agg = c0.group_by(["cluster", clock, "p", "y"]).agg(pl.col("trades").sum().alias("w"),
                                                                pl.col("contracts").sum().alias("wc"))
            for b in sorted(agg[clock].drop_nulls().unique().to_list()):
                cb = agg.filter(pl.col(clock) == b)
                out[f"{split}|{clock}|{b}"] = {
                    "trade_w": platt_by_cluster(cb, weight="w"),
                    "contract_w": platt_by_cluster(cb, weight="wc", reps=100),
                    "bands": band_rates(cb, STEP_BANDS, weight="w")}
    return out


def t1_artefact(cells: pl.DataFrame) -> dict[str, Any]:
    """Within the same pre-known late-game window, split prints by whether the game actually ended
    within 10 minutes. If the step is a selection effect it appears only in the 'ends within 10'
    half, which a trader cannot know at the time."""
    late = cells.filter(pl.col("late"))
    out = {}
    for split in ("dev", "holdout", "all"):
        c0 = late if split == "all" else late.filter(pl.col("split") == split)
        for flag in (True, False, None):
            cf = c0 if flag is None else c0.filter(pl.col("ends_10") == flag)
            agg = cf.group_by(["cluster", "p", "y"]).agg(pl.col("trades").sum().alias("w"))
            out[f"{split}|late(L-30)|ends_within_10={flag}"] = {
                "trade_w": platt_by_cluster(agg, weight="w"), "bands": band_rates(agg, STEP_BANDS)}
    return out


def t1_first_prints(markets: pl.DataFrame, lengths: dict[str, tuple[float, int]]) -> pl.DataFrame:
    """Equal-weight version: per game and clock bucket, the price of the *first* print in the bucket
    (one observation per game). For a clock known at the time this is a stopping time, so an
    efficient market is calibrated on it; for the actual-close clock it is not (the bucket's start
    depends on when the game will end)."""
    files = sorted(glob.glob(INPLAY_TRADES))
    L = pl.col("series_ticker").replace_strict({k: v[0] for k, v in lengths.items()}, default=None,
                                               return_dtype=pl.Float64)
    out = []
    for ch in _chunks(files):
        t = (t1_trades(markets, ch).with_columns(_bucket("tte_actual", TTE_EDGES, "b_actual"),
                                                 _bucket("tte_sched", TTE_EDGES, "b_sched"),
                                                 (pl.col("tte_actual") < 10).alias("ends_10"),
                                                 (pl.col("elapsed") >= L - 30).fill_null(False).alias("late"))
             .sort("ticker", "created_time").collect())
        for clock in ("b_actual", "b_sched"):
            out.append(t.group_by("ticker", clock, maintain_order=True)
                       .agg(pl.col("yes_price").first().alias("p"), pl.col("y").first(), pl.col("split").first())
                       .rename({clock: "bucket"}).with_columns(pl.lit(clock).alias("clock")))
        lt = t.filter(pl.col("late"))
        out.append(lt.group_by("ticker", maintain_order=True)
                   .agg(pl.col("yes_price").first().alias("p"), pl.col("y").first(), pl.col("split").first(),
                        pl.col("ends_10").first().cast(pl.String).alias("bucket"))
                   .with_columns(pl.lit("late_first(ends_10 at that print)").alias("clock")))
    return pl.concat(out, how="diagonal_relaxed")


def t1_first_stats(fp: pl.DataFrame) -> dict[str, Any]:
    out = {}
    fp = fp.with_columns(pl.col("ticker").alias("cluster"), pl.lit(1.0).alias("w"))
    for split in ("dev", "holdout", "all"):
        c0 = fp if split == "all" else fp.filter(pl.col("split") == split)
        for (clock, b), g in sorted(c0.group_by(["clock", "bucket"]), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            out[f"{split}|{clock}|{b}"] = {"fit": platt_by_cluster(g.select("cluster", "p", "y", "w"), reps=300),
                                           "bands": band_rates(g, STEP_BANDS)}
    return out


def t1_game_level(cells: pl.DataFrame, lo: float = 0.10, hi: float = 0.90) -> dict[str, Any]:
    """One observation per game and bucket (its trade-weighted mean price over prints at lo..hi;
    outcome y), so a few busy games cannot dominate and decided games (prices at 1-9c / 91-99c, which
    make the fit separate) do not enter: Platt slope with a game bootstrap, for the actual-close
    clock, the two pre-known clocks and the pre-known late window split by the look-ahead flag."""
    out: dict[str, Any] = {}
    cells = cells.filter(pl.col("p").is_between(lo, hi))
    specs = [("b_actual", None), ("b_sched", None), ("b_elapsed", None)]
    for split in ("dev", "holdout", "all"):
        c0 = cells if split == "all" else cells.filter(pl.col("split") == split)
        for clock, _ in specs:
            g = (c0.group_by("cluster", clock, "y").agg(((pl.col("p") * pl.col("trades")).sum()
                                                         / pl.col("trades").sum()).alias("p"))
                 .with_columns(pl.lit(1.0).alias("w")))
            for b in sorted(g[clock].drop_nulls().unique().to_list()):
                out[f"{split}|{clock}|{b}"] = platt_by_cluster(g.filter(pl.col(clock) == b), reps=300)
        late = c0.filter(pl.col("late"))
        for flag in (True, False, None):
            cf = late if flag is None else late.filter(pl.col("ends_10") == flag)
            g = (cf.group_by("cluster", "y").agg(((pl.col("p") * pl.col("trades")).sum() / pl.col("trades").sum()).alias("p"))
                 .with_columns(pl.lit(1.0).alias("w")))
            out[f"{split}|late(L-30)|ends_within_10={flag}"] = platt_by_cluster(g, reps=300)
    return out


# ---- shared execution / book helpers ----------------------------------------------------------------
ORDER_USD = 5.0
FAV_BANDS = [(0.50, 0.70), (0.70, 0.85), (0.85, 0.95), (0.95, 0.995)]


def taker_fill(price: float, avail: float, fee_type: str | None, mult: float | None,
               usd: float = ORDER_USD) -> tuple[int, float]:
    """Contracts (<= ``usd`` / price and <= the size the print shows) and the taker fee, rounded up
    per order (``albert.fees.kalshi_fee``, conservative)."""
    from albert.fees import kalshi_fee

    n = int(min(usd // price, avail))
    if n < 1:
        return 0, 0.0
    return n, kalshi_fee(n, price, maker=False, fee_type=fee_type or "quadratic", multiplier=mult)


def capped_book(f: pl.DataFrame, *, bankroll: float = 100.0, max_open: float = 20.0) -> pl.DataFrame:
    """Copied from ``approval2.capped_book`` (columns t, T, cost, payoff): keep fills in time order
    while cash covers them and the open cost stays <= ``max_open``."""
    keep, open_pos, cash = [], [], bankroll
    for r in f.sort("t").iter_rows(named=True):
        still = []
        for T, c, pay in open_pos:
            if T <= r["t"]:
                cash += pay
            else:
                still.append((T, c, pay))
        open_pos = still
        if r["cost"] > cash or sum(c for _, c, _ in open_pos) + r["cost"] > max_open:
            continue
        cash -= r["cost"]
        open_pos.append((r["T"], r["cost"], r["payoff"]))
        keep.append(r)
    return pl.DataFrame(keep, schema=f.schema)


def to_ledger(f: pl.DataFrame, strategy: str, variant: str, split: str) -> pl.DataFrame:
    return f.select(pl.lit(strategy).alias("strategy"), pl.lit(variant).alias("variant"), pl.lit(split).alias("split"),
                    pl.concat_str("ticker", pl.col("t").cast(pl.String), separator="@").alias("struct_id"),
                    pl.col("event").alias("event_ticker"), "ticker", "side", "n", pl.col("t").alias("t_in"),
                    pl.col("cost").alias("cost_usd"), pl.col("T").alias("settle_ts"), pl.col("payoff").alias("payoff_usd"))


def fill_stats(f: pl.DataFrame) -> dict[str, Any]:
    """c/contract and P&L per $ with event-clustered t (and two-way event x ET date)."""
    from albert.research.counts import cluster_se

    if f.height == 0:
        return {"fills": 0}
    pc = (f["pnl"] / f["n"] * 100).to_numpy()
    ev = f["event"].to_numpy()
    day = f["t"].dt.convert_time_zone("America/New_York").dt.date().cast(pl.String).to_numpy()
    # contract-weighted mean per contract
    w = f["n"].to_numpy().astype(float)
    x = f["pnl"].to_numpy() / w * 100
    mu = float((x * w).sum() / w.sum())
    se1 = cluster_se(x, ev)
    se2 = cluster_se(x, ev, day)
    return {"fills": f.height, "events": int(np.unique(ev).size), "contracts": int(w.sum()),
            "c_per_contract": round(mu, 2), "c_per_fill_mean": round(float(pc.mean()), 2),
            "t_event": round(float(pc.mean()) / se1, 2) if se1 and se1 > 0 else None,
            "t_2way": round(float(pc.mean()) / se2, 2) if se2 and se2 > 0 else None,
            "pnl_usd": round(float(f["pnl"].sum()), 2), "staked_usd": round(float(f["cost"].sum()), 2),
            "pnl_per_usd": round(float(f["pnl"].sum() / f["cost"].sum()), 4),
            "win_rate": round(float(f["win"].mean()), 3)}


def vol_block(led: pl.DataFrame, marks: pl.DataFrame | None = None) -> dict[str, Any]:
    from albert.research.vol_metrics import ledger_metrics, load_marks

    if led.height == 0:
        return {}
    if marks is None:
        marks = load_marks(led["ticker"].unique().to_list())
    m = ledger_metrics(led, marks)
    keep = ["positions", "events", "usd_per_week", "ann_vol_pct", "sharpe", "sharpe_ci90", "sharpe_ex_top5",
            "max_dd_usd", "weeks_positive_pct", "min_cum_pnl_usd", "marked_share"]
    return {k: m.get(k) for k in keep}


# ---- T1b: late-game favourite taker on a pre-known clock ---------------------------------------------
def t1_taker(markets: pl.DataFrame, lengths: dict[str, tuple[float, int]], offsets: tuple[int, ...] = (30, 15),
             delay_s: float = 2.0, max_wait_min: float = 5.0, stale_min: float = 10.0,
             trades: pl.DataFrame | None = None) -> pl.DataFrame:
    """One decision per game and offset k at sched_start + L - k (L = dev median length): if the market
    is still open, the favourite is the side of the last print before the decision (>= 50c); buy it at
    the first later taker print on that side (>= delay_s after the decision, within max_wait_min),
    <= $5 and <= that print's size, taker fee rounded up per order, held to settlement."""
    cols = ["ticker", "created_time", "count", "yes_price", "taker_side"]
    if trades is None:                       # one chunk of game files at a time (memory cap)
        files = [f for f in sorted(glob.glob(INPLAY_TRADES)) if Path(f).name[:-8] in set(markets["ticker"])]
        parts = [t1_taker(markets, lengths, offsets, delay_s, max_wait_min, stale_min,
                          trades=pl.read_parquet(ch, columns=cols)) for ch in _chunks(files)]
        parts = [x for x in parts if x.height]
        return pl.concat(parts) if parts else pl.DataFrame()
    else:
        t = (trades.select(cols).join(markets.select("ticker", "close_time"), on="ticker")
             .filter((pl.col("created_time") < pl.col("close_time")) & pl.col("yes_price").is_between(0.01, 0.99))
             .select(cols))
    t = t.sort("ticker", "created_time")
    info = {r["ticker"]: r for r in markets.iter_rows(named=True)}
    rows = []
    for (tk,), g in t.group_by(["ticker"], maintain_order=True):
        r = info[tk]
        L = lengths.get(r["series_ticker"], (None, 0))[0]
        if L is None:
            continue
        ts = g["created_time"].to_list()
        ps = g["yes_price"].to_numpy()
        qs = g["count"].to_numpy()
        sides = g["taker_side"].to_list()
        for k in offsets:
            d = r["sched_start"] + _td(minutes=L - k)
            if r["close_time"] <= d:
                continue                       # already closed at the decision: nothing to trade
            i = _bisect_left(ts, d) - 1
            if i < 0 or (d - ts[i]).total_seconds() > stale_min * 60:
                continue
            fav = "yes" if ps[i] >= 0.5 else "no"
            j = _bisect_left(ts, d + _td(seconds=delay_s))
            while j < len(ts) and sides[j] != fav:
                j += 1
            if j >= len(ts) or (ts[j] - d).total_seconds() > max_wait_min * 60:
                continue
            e = float(ps[j]) if fav == "yes" else round(1 - float(ps[j]), 4)
            if e < 0.5 or e > 0.995:
                continue
            n, fee = taker_fill(e, float(qs[j]), r["fee_type"], r["fee_multiplier"])
            if n < 1:
                continue
            win = int(r["y"] == 1) if fav == "yes" else int(r["y"] == 0)
            rows.append({"ticker": tk, "event": r["event_ticker"], "series": r["series_ticker"], "split": r["split"],
                         "offset": k, "side": fav, "t": ts[j], "T": r["close_time"], "price": e, "n": n, "fee": fee,
                         "cost": n * e + fee, "payoff": float(n * win), "win": win,
                         "pnl": n * win - n * e - fee, "gross_c": (win - e) * 100})
    f = pl.DataFrame(rows)
    if f.height:
        f = f.with_columns(_band_expr("price"))
    return f


def _band_expr(col: str) -> pl.Expr:
    e = pl.lit(None, dtype=pl.String)
    for lo, hi in reversed(FAV_BANDS):
        e = pl.when((pl.col(col) >= lo) & (pl.col(col) < hi)).then(pl.lit(f"[{lo:.2f},{hi:.2f})")).otherwise(e)
    return e.alias("band")


def _td(**kw: float):
    from datetime import timedelta

    return timedelta(**kw)


def _bisect_left(a: list, x: Any) -> int:
    import bisect

    return bisect.bisect_left(a, x)


# ---- T2: Le, large political trades and compression ---------------------------------------------------
POLITICS = ("Politics", "Elections")
SWEEP_USD = (100.0, 500.0)
SMALL_USD = 20.0


def politics_markets() -> pl.DataFrame:
    """Settled markets with the integer id ``mid`` exactly as ``yes_bias.market_frame`` assigns it
    (row number of settled markets sorted by ticker), so ``yes_bias_prints`` can be joined."""
    m = pl.read_parquet("data/tables/kalshi_markets.parquet",
                        columns=["ticker", "event_ticker", "series_ticker", "category", "result", "close_time",
                                 "expected_expiration_time", "fee_type", "fee_multiplier"])
    m = m.filter(pl.col("result").is_in(["yes", "no"])).sort("ticker").with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("mid"))
    exp = pl.coalesce("expected_expiration_time", "close_time")
    return m.filter(pl.col("category").is_in(list(POLITICS))).with_columns(
        (pl.col("result") == "yes").cast(pl.Int8).alias("y"),
        pl.when(exp >= HOLDOUT_FROM).then(pl.lit("holdout")).otherwise(pl.lit("dev")).alias("split"))


def politics_prints(markets: pl.DataFrame) -> pl.DataFrame:
    """Volume-blind 'band' prints of Politics/Elections markets (yes_bias source 'band', cum_ok), with
    same-instant same-side prints merged into one sweep (one taker order)."""
    p = (pl.scan_parquet("data/cache/yes_bias_prints.parquet")
         .filter((pl.col("source") == "band") & pl.col("cum_ok"))
         .join(markets.lazy().select("mid"), on="mid", how="inner")
         .select("mid", "t", "q", "p", "yes").collect())
    usd = pl.when(pl.col("yes")).then(pl.col("p")).otherwise(1 - pl.col("p")) * pl.col("q")
    return (p.with_columns(usd.alias("usd"))
            .group_by("mid", "t", "yes", maintain_order=True)
            .agg(pl.col("q").sum(), pl.col("usd").sum(), pl.col("p").first().alias("p_first"),
                 pl.col("p").last().alias("p_last"))
            .sort("mid", "t"))


def t2_entries(sweeps: pl.DataFrame, markets: pl.DataFrame, *, thresholds: tuple[float, ...] = SWEEP_USD,
               delay_s: float = 2.0, max_wait_min: float = 60.0) -> pl.DataFrame:
    """After a longshot-side sweep (taker buys the side priced < 50c) of >= S dollars, buy the
    favourite at the next taker print on the favourite side (>= delay_s later, within max_wait_min),
    <= $5 and <= that sweep's size, fee rounded up, held to settlement; one entry per market, ET day
    and trigger. Trigger 'small' (< SMALL_USD) is the control."""
    info = {r["mid"]: r for r in markets.iter_rows(named=True)}
    rows = []
    for (mid,), g in sweeps.group_by(["mid"], maintain_order=True):
        if mid not in info:
            continue                                      # other split
        r = info[mid]
        ts = g["t"].to_list()
        yes = g["yes"].to_list()
        pf = g["p_first"].to_list()
        pl_ = g["p_last"].to_list()
        usd = g["usd"].to_list()
        qq = g["q"].to_list()
        used: set[tuple[str, Any]] = set()
        for i in range(len(ts)):
            side_price = pf[i] if yes[i] else 1 - pf[i]
            if side_price >= 0.5:
                continue                                  # not a longshot-side sweep
            trig = [f"S{int(s)}" for s in thresholds if usd[i] >= s] + (["small"] if usd[i] < SMALL_USD else [])
            if not trig:
                continue
            fav_yes = not yes[i]
            j = i + 1
            while j < len(ts) and (yes[j] != fav_yes or (ts[j] - ts[i]).total_seconds() < delay_s):
                j += 1
            if j >= len(ts) or (ts[j] - ts[i]).total_seconds() > max_wait_min * 60:
                continue
            e = pf[j] if fav_yes else round(1 - pf[j], 4)
            if e < 0.5 or e > 0.995 or ts[j] >= r["close_time"]:
                continue
            day = ts[i].astimezone(_ET).date()
            for tg in trig:
                if (tg, day) in used:
                    continue
                used.add((tg, day))
                n, fee = taker_fill(e, qq[j], r["fee_type"], r["fee_multiplier"])
                if n < 1:
                    continue
                win = int(r["y"] == 1) if fav_yes else int(r["y"] == 0)
                rows.append({"mid": mid, "ticker": r["ticker"], "event": r["event_ticker"], "category": r["category"],
                             "split": r["split"], "trigger": tg, "sweep_usd": usd[i], "side": "yes" if fav_yes else "no",
                             "t0": ts[i], "t": ts[j], "T": r["close_time"], "price": e, "move_c": (pl_[i] - pf[i]) * 100,
                             "n": n, "fee": fee, "cost": n * e + fee, "payoff": float(n * win), "win": win,
                             "pnl": n * win - n * e - fee, "gross_c": (win - e) * 100})
    f = pl.DataFrame(rows)
    return f.with_columns(_band_expr("price")) if f.height else f


def t2_calibration(sweeps: pl.DataFrame, markets: pl.DataFrame, splits: tuple[str, ...] = ("dev",)) -> dict[str, Any]:
    """Le's descriptive claim: Platt slope of trade prices at large vs small sweeps (one row per sweep,
    weight 1, clustered by market)."""
    s = sweeps.join(markets.select("mid", "y", "split"), on="mid").with_columns(
        pl.col("p_first").alias("p"), pl.col("mid").alias("cluster"), pl.lit(1.0).alias("w"),
        pl.when(pl.col("usd") >= 500).then(pl.lit(">=500")).when(pl.col("usd") >= 100).then(pl.lit("100-500"))
        .when(pl.col("usd") >= 20).then(pl.lit("20-100")).otherwise(pl.lit("<20")).alias("size"))
    out = {}
    for split in splits:
        for size in ("<20", "20-100", "100-500", ">=500"):
            c = s.filter((pl.col("split") == split) & (pl.col("size") == size)).select("cluster", "p", "y", "w")
            out[f"{split}|{size}"] = platt_by_cluster(c, reps=200)
    return out


def _et():
    from zoneinfo import ZoneInfo

    return ZoneInfo("America/New_York")


_ET = _et()


# ---- T3: Cardozo & Rivero-Wildemauwe, longshots per contract vs per event ----------------------------
LONGSHOT = 0.10


def t3_frame() -> pl.DataFrame:
    """flb_xs panel quotes of complete mutually-exclusive multi-outcome events (every leg quoted at
    the lead, usable two-sided quote: ask > bid, spread <= 10c)."""
    p = pl.read_parquet("data/research/flb_xs/panel.parquet")
    p = p.filter((pl.col("structure") == "multi") & (pl.col("yes_ask") > pl.col("yes_bid")) & (pl.col("spread") <= 0.10)
                 & pl.col("y").is_not_null())
    cnt = p.group_by("event_ticker", "lead_h").agg(pl.col("ticker").n_unique().alias("in_panel"),
                                                   pl.col("n_legs").first().alias("legs"),
                                                   pl.col("y").sum().alias("winners"))
    p = p.join(cnt, on=["event_ticker", "lead_h"]).filter((pl.col("in_panel") == pl.col("legs")) & (pl.col("winners") == 1))
    return p.with_columns(pl.when(pl.col("scheduled_ts") >= HOLDOUT_FROM).then(pl.lit("holdout"))
                          .otherwise(pl.lit("dev")).alias("split"))


def t3_stats(p: pl.DataFrame) -> list[dict[str, Any]]:
    """Per lead and split: longshot (mid <= 10c) YES return per $ at the mid, contract-weighted vs
    event-grouped, and the taker short-field (NO on every <= 10c leg at 1 - bid, fee per leg, 1 lot)."""
    from albert.fees import kalshi_fee

    out = []
    ls = p.filter(pl.col("mid") <= LONGSHOT)
    fee_no = [kalshi_fee(1, 1 - b, fee_type=ft or "quadratic", multiplier=fm)
              for b, ft, fm in ls.select("yes_bid", "fee_type", "fee_multiplier").iter_rows()]
    ls = ls.with_columns(pl.Series("fee_no", fee_no),
                         (pl.col("y") - pl.col("mid")).alias("g_mid"))
    for (lead, split), g in ls.group_by(["lead_h", "split"]):
        ev = g.group_by("event_ticker").agg(pl.col("g_mid").sum(), pl.col("mid").sum(), pl.len().alias("k"),
                                            ((1 - pl.col("y")) - (1 - pl.col("yes_bid")) - pl.col("fee_no")).sum().alias("short"),
                                            (1 - pl.col("yes_bid") + pl.col("fee_no")).sum().alias("short_cost"))
        r_ev = (ev["g_mid"] / ev["mid"]).to_numpy()
        sh = (ev["short"] / ev["short_cost"]).to_numpy()
        out.append({
            "lead_h": lead, "split": split, "legs": g.height, "events": ev.height,
            "contract_w_ret_per_usd": round(float(g["g_mid"].sum() / g["mid"].sum()), 3),
            "event_grouped_ret_per_usd": round(float(r_ev.mean()), 3),
            "event_grouped_t": round(float(r_ev.mean() / (r_ev.std(ddof=1) / np.sqrt(len(r_ev)))), 2) if len(r_ev) > 2 else None,
            "field_won_share": round(float((ev["g_mid"] + ev["mid"] > 0.5).mean()), 3),
            "field_price_mean": round(float(ev["mid"].mean()), 3),
            "short_field_ret_per_usd": round(float(sh.mean()), 4),
            "short_field_t": round(float(sh.mean() / (sh.std(ddof=1) / np.sqrt(len(sh)))), 2) if len(sh) > 2 else None,
            "short_field_c_per_leg": round(float(ev["short"].sum() / ev["k"].sum() * 100), 2)})
    return sorted(out, key=lambda d: (d["split"], d["lead_h"]))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "help"
    if cmd == "robots":
        for r in check_hosts():
            flag = "ALLOWED" if r["allowed"] else "BLOCKED"
            print(f"{flag:8s} {r['http_status']} {r['host']}{r['path']}")
    elif cmd == "t1-cal":
        OUT.mkdir(parents=True, exist_ok=True)
        markets = game_markets(inplay_tickers())
        lengths = league_lengths("dev")
        print("dev median game length (min), games:", lengths, flush=True)
        cells = t1_cells_chunked(markets, lengths)
        cells.write_parquet(OUT / "t1_cells.parquet")
        res = {"lengths_dev": lengths, "games": markets.group_by("series_ticker", "split").len().to_dicts(),
               "calibration": t1_calibration(cells), "artefact": t1_artefact(cells)}
        (OUT / "t1_calibration.json").write_text(json.dumps(res, indent=1, default=str))
        print("wrote", OUT / "t1_calibration.json")
    elif cmd == "ledgers":
        # capped $100 book ($5 orders, <= $20 open) for the pre-declared primaries, dev and holdout
        (OUT / "ledgers").mkdir(parents=True, exist_ok=True)
        specs = [("t1_late_favourite", "L-30_all", "t1_taker_{}.parquet", pl.col("offset") == 30),
                 ("t1_late_favourite", "L-15_all", "t1_taker_{}.parquet", pl.col("offset") == 15),
                 ("t2_after_large_longshot_sweep", "S100_all", "t2_entries_{}.parquet", pl.col("trigger") == "S100"),
                 ("t2_after_large_longshot_sweep", "small_control", "t2_entries_{}.parquet", pl.col("trigger") == "small")]
        res = {}
        for strat, var, pat, flt in specs:
            for split in ("dev", "holdout"):
                f = pl.read_parquet(OUT / pat.format(split)).filter(flt)
                book = capped_book(f.select("ticker", "event", "side", "n", "t", "T", "cost", "payoff", "pnl"))
                led = to_ledger(book, strat, var, split)
                led.write_parquet(OUT / "ledgers" / f"ledger_{strat}_{var}_{split}.parquet")
                res[f"{strat}|{var}|{split}"] = {"book_fills": book.height, "book_pnl": round(float(book["pnl"].sum()), 2),
                                                 **vol_block(led)}
        (OUT / "ledgers.json").write_text(json.dumps(res, indent=1, default=str))
        for k, v in res.items():
            print(k, v)
    elif cmd == "t1-first":
        markets = game_markets(inplay_tickers())
        fp = t1_first_prints(markets, league_lengths("dev"))
        fp.write_parquet(OUT / "t1_first_prints.parquet")
        res = t1_first_stats(fp)
        (OUT / "t1_first.json").write_text(json.dumps(res, indent=1, default=str))
        for k, v in res.items():
            print(k, v["fit"], [(b["band"], b.get("mean_p"), b.get("win_rate"), b.get("clusters")) for b in v["bands"]])
    elif cmd == "t1-game":
        res = t1_game_level(pl.read_parquet(OUT / "t1_cells.parquet"))
        (OUT / "t1_game_level.json").write_text(json.dumps(res, indent=1, default=str))
        for k, v in res.items():
            print(k, v)
    elif cmd == "t1-taker":
        split = argv[1] if len(argv) > 1 else "dev"
        markets = game_markets(inplay_tickers())
        lengths = league_lengths("dev")
        f = t1_taker(markets.filter(pl.col("split") == split), lengths)
        f.write_parquet(OUT / f"t1_taker_{split}.parquet")
        res = {}
        for (k, band), g in f.group_by(["offset", "band"]):
            res[f"L-{k}|{band}"] = fill_stats(g)
        for k in (30, 15):
            res[f"L-{k}|all"] = fill_stats(f.filter(pl.col("offset") == k))
        (OUT / f"t1_taker_{split}.json").write_text(json.dumps(dict(sorted(res.items())), indent=1, default=str))
        print(json.dumps(dict(sorted(res.items())), indent=1, default=str))
    elif cmd == "t2":
        split = argv[1] if len(argv) > 1 else "dev"
        markets = politics_markets()
        sw = politics_prints(markets)
        print("politics sweeps:", sw.height, "markets:", sw["mid"].n_unique(), flush=True)
        f = t2_entries(sw, markets.filter(pl.col("split") == split))
        f.write_parquet(OUT / f"t2_entries_{split}.parquet")
        res = {"calibration": t2_calibration(sw, markets, (split,))}
        for (tg, band), g in f.group_by(["trigger", "band"]):
            res[f"{tg}|{band}"] = fill_stats(g)
        for tg in ("S100", "S500", "small"):
            res[f"{tg}|all"] = fill_stats(f.filter(pl.col("trigger") == tg))
        (OUT / f"t2_{split}.json").write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(res, indent=1, default=str))
    elif cmd == "t3-ledger":
        from albert.fees import kalshi_fee

        (OUT / "ledgers").mkdir(parents=True, exist_ok=True)
        fr = t3_frame().filter(pl.col("mid") <= LONGSHOT)
        res = {}
        for lead in (24.0, 6.0):
            for split in ("dev", "holdout"):
                g = fr.filter((pl.col("lead_h") == lead) & (pl.col("split") == split))
                fees = [kalshi_fee(1, 1 - b, fee_type=ft or "quadratic", multiplier=fm)
                        for b, ft, fm in g.select("yes_bid", "fee_type", "fee_multiplier").iter_rows()]
                f = g.select(pl.col("ticker"), pl.col("event_ticker").alias("event"), pl.lit("no").alias("side"),
                             pl.lit(1).alias("n"), pl.col("ts").alias("t"), pl.col("close_time").alias("T"),
                             (1 - pl.col("yes_bid")).alias("px"), (1 - pl.col("y")).cast(pl.Float64).alias("payoff")
                             ).with_columns(pl.Series("fee", fees)).with_columns(
                    (pl.col("px") + pl.col("fee")).alias("cost")).with_columns((pl.col("payoff") - pl.col("cost")).alias("pnl"))
                book = capped_book(f.select("ticker", "event", "side", "n", "t", "T", "cost", "payoff", "pnl"))
                led = to_ledger(book, "t3_short_field", f"L{int(lead)}h", split)
                led.write_parquet(OUT / "ledgers" / f"ledger_t3_short_field_L{int(lead)}h_{split}.parquet")
                stats = fill_stats(book.with_columns((pl.col("payoff") > 0).cast(pl.Int8).alias("win")))
                res[f"L{int(lead)}h|{split}"] = {**stats, **vol_block(led)}
        (OUT / "t3_ledger.json").write_text(json.dumps(res, indent=1, default=str))
        for k, v in res.items():
            print(k, v)
    elif cmd == "t3":
        rows = t3_stats(t3_frame())
        (OUT / "t3.json").write_text(json.dumps(rows, indent=1, default=str))
        print(pl.DataFrame(rows))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
