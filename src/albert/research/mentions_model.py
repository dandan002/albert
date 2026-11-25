"""Word-level probability model for the live pre-speech NO bids on Kalshi mention markets.

Question (research swarm 2026-11-23, slug ``mentions_model``): the live rule rests a NO bid on
every word leg of a speech event at ``floor(1 - fair - 12c)`` (``fair`` = frozen calibrated mid,
``research/mentions_ext``), from start - 12 h (early phase) and start - 6 h (main phase) to start -
10 min. Does a per-word estimate of p(word said), built as of the bid time from Kalshi's own history
of settled mention markets, improve it: skip legs the model calls likely, set the margin per word,
or add YES bids on words the model calls underpriced?

Hypotheses (written before any outcome data of this study was opened):

* **H1 habit words (skip).** Speakers repeat themselves; a word this speaker said in most past
  Kalshi events but quoted at t below that rate is under-priced, and the YES buyer who lifts our
  offer later is informed by the same history, so NO fills on such legs are adversely selected.
  Test: drop the NO bid when the model's p >= (our YES sell price) - theta.
* **H2 novel / rare words (margin).** The pre-speech YES premium (the edge) is retail buying topical
  words; it should be largest where the speaker rarely or never said the word. Test: anchor the
  sell price on the model (``p_model + m``) or tighten to 8c where p_model is well below fair.
* **H3 YES bids on under-priced words.** If the model finds words priced well below their history,
  a resting YES bid below the model fair could earn the mirror premium. Prior: weak (the documented
  flow is YES buying, so YES bids fill mainly on bad news for the word).
* **H0 (expected from earlier work, ``research/mentions``).** Kalshi history alone is much worse than
  the market (Brier 0.24 vs 0.17) and adds little to it (blend coefficient 0.14-0.26, t 1.3-1.8);
  the NO-bid edge is a price pattern (selling YES into pre-speech spikes), and the salience study
  found the most hyped words were the best fills. So the overlay may well do nothing.

Data: ``research/mentions_ext`` universe (settled Mentions speech legs Mar 2025 - Nov 16 2026, window
series, earnings calls and finance-filtered series excluded) and its cached hourly candles, plus
events that started 2026-11-16 12:00 - 11-23 fetched here from the public Kalshi API (milestones,
markets, hourly candles) into ``data/cache/mentions_model/``. Model and policies: see :func:`run`.

    python -m albert.research.mentions_model fetch     # Nov 15 - 22 events (public Kalshi, cached)
    python -m albert.research.mentions_model build     # decision rows + as-of history features
    python -m albert.research.mentions_model run       # walk-forward model, dev grid, frozen holdout
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

NAME = "mentions_model"
OUT = Path("data/research/mentions_model")
CACHE = Path("data/cache/mentions_model")
EXT_UNIVERSE = Path("data/research/mentions_ext/universe.parquet")
TABLES = Path("data/tables")

DEV_END = datetime(2026, 10, 1, tzinfo=UTC)          # dev: starts before this
SEEN_END = datetime(2026, 11, 16, 12, tzinfo=UTC)    # Oct 1 - Nov 16: live family scored by earlier studies
NEW_FROM = datetime(2026, 11, 16, 12, tzinfo=UTC)    # events fetched here (the ext universe ends here)
FETCH_UNTIL = datetime(2026, 11, 23, 6, tzinfo=UTC)
CONFIRM_FROM = datetime(2026, 11, 17, 5, tzinfo=UTC)  # Nov 17 00:00 ET: clean confirmatory slice
MS_FROM = datetime(2026, 11, 14, tzinfo=UTC)
EPS = 1e-9


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _iso(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def split_of(start: pl.Expr) -> pl.Expr:
    """dev (< Oct 1), seen (Oct 1 - Nov 16 12:00 UTC: the live family's holdout, scored before) and
    new (Nov 16 12:00 -: never scored by any study; ``confirm`` = Nov 17 ET onward)."""
    return (pl.when(start < DEV_END).then(pl.lit("dev"))
            .when(start < SEEN_END).then(pl.lit("seen"))
            .when(start < CONFIRM_FROM).then(pl.lit("gap"))
            .otherwise(pl.lit("confirm")))


# ---- fetch (public Kalshi, cached append-only under data/cache/mentions_model/raw) -----------------
def client(min_interval: float = 0.6):
    import random
    import time

    from albert.config import Settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    s = Settings()

    def nap(x: float) -> None:
        time.sleep(min(max(x, 2.0), 30.0) * (0.75 + 0.5 * random.random()))
    return k.Kalshi(CachedClient("kalshi", s.kalshi_base, CACHE / "raw", min_interval=min_interval,
                                 max_retries=12, sleep=nap))


def milestone_frame(pages: list[dict[str, Any]]) -> pl.DataFrame:
    from albert.research.pre_event import MS_SCHEMA, milestone_rows

    rows = [r for p in pages for r in milestone_rows(p)]
    return pl.DataFrame(rows, schema=MS_SCHEMA).unique(["event", "ms_id"])


def fetch_milestones(api, tag: str) -> pl.DataFrame:
    pages, cursor = [], None
    for _ in range(40):
        page = api.client.get("/milestones", {"limit": 500, "minimum_start_date": _iso(MS_FROM), "category": "Mentions",
                                              "cursor": cursor}, tag=tag)
        pages.append(page.payload)
        cursor = page.payload.get("cursor")
        if not cursor or not page.payload.get("milestones"):
            break
    return milestone_frame(pages)


def market_frame(payloads: list[dict[str, Any]]) -> pl.DataFrame:
    """Settled-or-not market rows of ``/markets?event_ticker=`` pages with series fee info."""
    from albert.venues import kalshi as k

    ser = pl.read_parquet(TABLES / "kalshi_series.parquet")
    info = {r["series_ticker"]: r for r in ser.select("series_ticker", "category", "fee_type", "fee_multiplier").iter_rows(named=True)}
    rows = [r for p in payloads for r in k.market_rows(p, info, "live")]
    return pl.DataFrame(rows, schema=k.MARKET_SCHEMA) if rows else pl.DataFrame(schema=k.MARKET_SCHEMA)


def fetch(until: datetime = FETCH_UNTIL) -> dict[str, Any]:
    """Milestones (Mentions, start >= Nov 14), markets of every event starting in [NEW_FROM, until)
    and hourly candles from max(open, start - 51 h) to start + 8 h (one batch request per event)."""
    from albert.research import xcat

    tag = f"{NAME}-{datetime.now(UTC):%Y%m%d}"
    api = client()
    ms = fetch_milestones(api, tag)
    _save(ms, CACHE / "milestones.parquet")
    starts = ms.filter(pl.col("start").is_not_null()).sort("start").unique("event", keep="first")
    new = starts.filter((pl.col("start") >= NEW_FROM) & (pl.col("start") < until))
    print(f"milestones: {ms.height} rows, {starts.height} events, {new.height} new events", flush=True)
    payloads = []
    for i, ev in enumerate(new["event"].to_list(), 1):
        try:
            payloads.append(api.client.get("/markets", {"event_ticker": ev, "limit": 1000}, tag=tag).payload)
        except RuntimeError as error:
            print(f"markets {ev}: {str(error)[-160:]}", flush=True)
        if i % 50 == 0:
            print(f"  markets {i}/{new.height}", flush=True)
    mk = market_frame(payloads)
    _save(mk, CACHE / "markets.parquet")
    mk = mk.filter(pl.col("category") == "Mentions")
    jobs = mk.join(new.select("event", "start"), left_on="event_ticker", right_on="event").with_columns(
        pl.max_horizontal(pl.col("open_time"), pl.col("start") - timedelta(hours=51)).alias("lo"),
        (pl.col("start") + timedelta(hours=8)).alias("hi"))
    buf: list[dict[str, Any]] = []
    for i, (ev, g) in enumerate(jobs.group_by("event_ticker", maintain_order=True), 1):
        tks = g["ticker"].to_list()
        lo, hi = g["lo"].min(), min(g["hi"].max(), datetime.now(UTC))
        for j in range(0, len(tks), 50):
            part = tks[j:j + 50]
            span_h = math.ceil((hi - lo).total_seconds() / 3600) + 1
            if span_h * len(part) > 9_500:
                print(f"  {ev}: span too long ({span_h} h)", flush=True)
            try:
                got = xcat.batch_rows(api.candles_batch(part, lo, hi, 60).payload, 60)
            except RuntimeError as error:
                print(f"candles {ev}: {str(error)[-160:]}", flush=True)
                continue
            for rows in got.values():
                buf.extend(rows)
        if i % 50 == 0:
            print(f"  candles {i} events", flush=True)
    H = pl.DataFrame(buf, schema=xcat.OHLC_SCHEMA)
    _save(H, CACHE / "hourly.parquet")
    info = {"milestone_rows": ms.height, "new_events": new.height, "markets": mk.height,
            "settled": int(mk["result"].is_in(["yes", "no"]).sum()), "candles": H.height,
            "candle_tickers": H["ticker"].n_unique(), "fetched_at": _iso(datetime.now(UTC))}
    (CACHE / "fetch_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    return info


# ---- universe ---------------------------------------------------------------------------------------
def new_legs() -> pl.DataFrame:
    """Settled speech legs of the fetched events, built exactly like ``mentions_ext.universe``."""
    from albert.research import mentions as M
    from albert.research import mentions_ext as X
    from albert.research import mentions_history as MH

    if not (CACHE / "markets.parquet").exists():
        return pl.DataFrame()
    mk = pl.read_parquet(CACHE / "markets.parquet").filter(pl.col("category") == "Mentions")
    if not mk.height:
        return pl.DataFrame()
    ms = pl.read_parquet(CACHE / "milestones.parquet")
    starts = ms.filter(pl.col("start").is_not_null()).sort("start").unique("event", keep="first").select(
        "event", "start", "ms_updated")
    lg = M.legs(mk).filter(~pl.col("series").is_in(M.WINDOW_SERIES))
    lg = lg.join(starts, on="event").join(mk.select("ticker", "yes_sub_title", "fee_type", "fee_multiplier"), on="ticker")
    lg = lg.with_columns(MH.speaker_group(pl.col("series")).alias("sgroup"))
    fin = X.financial_series(lg["series"].unique().to_list())
    lg = lg.filter((pl.col("sgroup") != "earnings") & ~pl.col("series").is_in(pl.Series(sorted(fin) or [""]).implode()))
    lg = lg.filter((pl.col("start") >= NEW_FROM) & (pl.col("start") < FETCH_UNTIL) & pl.col("close_time").is_not_null())
    return lg.with_columns(pl.lit("holdout").alias("split"), X.et_date("start").alias("et_date"),
                           pl.col("yes_sub_title").map_elements(lambda x: "|".join(X.word_titles(x)), return_dtype=pl.String
                                                                ).alias("wiki")).sort("start", "event", "ticker")


def universe() -> pl.DataFrame:
    U = pl.read_parquet(EXT_UNIVERSE)
    N = new_legs()
    if N.height:
        U = pl.concat([U, N.select(U.columns).filter(~pl.col("ticker").is_in(U["ticker"].implode()))], how="vertical_relaxed")
    return U.with_columns(split_of(pl.col("start")).alias("split")).sort("start", "event", "ticker")


def load_hourly(tickers: list[str]) -> pl.DataFrame:
    """``mentions_ext.load_hourly`` plus the candles fetched here (same Kalshi candles)."""
    from albert.research import mentions_ext as X

    H = X.load_hourly(tickers)
    p = CACHE / "hourly.parquet"
    if p.exists():
        mine = X._ohlc(pl.scan_parquet(p)).filter(pl.col("ticker").is_in(pl.Series(tickers).implode())).collect()
        mine = mine.with_columns(pl.lit(NAME).alias("src")).filter(~pl.col("ticker").is_in(H["ticker"].unique().implode()))
        H = pl.concat([H, mine.select(H.columns)], how="vertical_relaxed").sort("ticker", "end_ts")
    return H


# ---- as-of history features -------------------------------------------------------------------------
PRIOR_SW, PRIOR_W, PRIOR_SP = 2.0, 2.0, 20.0


def wkey() -> pl.Expr:
    return pl.when(pl.col("k") > 1).then(pl.col("word") + pl.lit("#") + pl.col("k").cast(pl.String)).otherwise(pl.col("word"))


def _cum(U: pl.DataFrame, by: list[str], name: str) -> pl.DataFrame:
    """Per key, cumulative listings ``n_<name>`` and YES results ``h_<name>`` by settlement time."""
    g = U.group_by([*by, "settlement_ts"]).agg(pl.len().alias("n"), pl.col("y").sum().alias("h")).sort([*by, "settlement_ts"])
    return g.with_columns(pl.col("n").cum_sum().over(by).alias(f"n_{name}"), pl.col("h").cum_sum().over(by).alias(f"h_{name}"),
                          ).select(*by, pl.col("settlement_ts").alias(f"ts_{name}"), f"n_{name}", f"h_{name}")


def _asof(D: pl.DataFrame, C: pl.DataFrame, by: list[str], name: str) -> pl.DataFrame:
    """Join the last cumulative row settled strictly before ``t``."""
    C = C.with_columns(pl.col(f"ts_{name}").alias("_ts")).sort("_ts")
    out = D.sort("t").join_asof(C, left_on="t", right_on="_ts", by=by or None, strategy="backward",
                                allow_exact_matches=False, check_sortedness=False)
    return out.drop("_ts").with_columns(pl.col(f"n_{name}").fill_null(0), pl.col(f"h_{name}").fill_null(0))


def history_features(D: pl.DataFrame, U: pl.DataFrame) -> pl.DataFrame:
    """As-of (settled strictly before ``t``) counts from every universe leg: same speaker + word,
    same word any speaker, speaker (single-mention legs), speaker x event type, all; plus the last
    result and days since the last listing of the same speaker + word."""
    U = U.filter(pl.col("settlement_ts").is_not_null()).with_columns(wkey().alias("wkey"),
                                                                     (pl.col("k") > 1).alias("thr"))
    D = D.with_columns(wkey().alias("wkey"), (pl.col("k") > 1).alias("thr"))
    D = _asof(D, _cum(U, ["speaker", "wkey"], "sw"), ["speaker", "wkey"], "sw")
    D = _asof(D, _cum(U, ["wkey"], "w"), ["wkey"], "w")
    D = _asof(D, _cum(U, ["speaker", "thr"], "sp"), ["speaker", "thr"], "sp")
    D = _asof(D, _cum(U, ["speaker", "etype", "thr"], "se"), ["speaker", "etype", "thr"], "se")
    D = _asof(D, _cum(U, ["thr"], "g"), ["thr"], "g")
    last = (U.sort("settlement_ts").select("speaker", "wkey", pl.col("settlement_ts").alias("_ts"), pl.col("y").alias("last_y"),
                                            pl.col("start").alias("last_start")))
    D = D.sort("t").join_asof(last, left_on="t", right_on="_ts", by=["speaker", "wkey"], strategy="backward",
                              allow_exact_matches=False, check_sortedness=False).drop("_ts")
    g = (pl.col("h_g") + 1) / (pl.col("n_g") + 2)
    sp = (pl.col("h_sp") + PRIOR_SP * g) / (pl.col("n_sp") + PRIOR_SP)
    se = (pl.col("h_se") + PRIOR_SP * sp) / (pl.col("n_se") + PRIOR_SP)
    return D.with_columns(g.alias("r_g"), sp.alias("r_sp"), se.alias("r_se")).with_columns(
        ((pl.col("h_sw") + PRIOR_SW * pl.col("r_se")) / (pl.col("n_sw") + PRIOR_SW)).alias("r_sw"),
        ((pl.col("h_w") + PRIOR_W * pl.col("r_se")) / (pl.col("n_w") + PRIOR_W)).alias("r_w"),
        ((pl.col("t") - pl.col("last_start")).dt.total_hours() / 24).alias("days_since"),
    )


# ---- decision rows ----------------------------------------------------------------------------------
DROWS_COLS = ["event", "ticker", "series", "speaker", "sgroup", "etype", "word", "k", "n_alts", "n_tokens", "split",
              "et_date", "start", "settlement_ts", "close_time", "y", "t", "bid", "ask", "mid", "spread", "fair", "end",
              "kill", "end_cb", "lead_h"]


def build() -> dict[str, Any]:
    """Decision rows at start - 12 h and start - 6 h over the whole universe with candles, as-of
    history features, and the hourly candles of those legs (saved for the run step)."""
    from albert.research import mentions_ext as X

    U = universe()
    _save(U, OUT / "universe.parquet")
    H = load_hourly(U["ticker"].to_list())
    print(f"universe {U.height} legs / {U['event'].n_unique()} events; hourly {H.height} rows, "
          f"{H['ticker'].n_unique()} tickers", flush=True)
    Uc = U.filter(pl.col("ticker").is_in(H["ticker"].unique().implode()))
    frames = []
    for L in (12, 6):
        Q = X.with_breaker(X.decision_rows(Uc, H, L), H)
        frames.append(Q.select([c for c in DROWS_COLS if c in Q.columns]))
        print(f"  L{L}h: {Q.height} rows, {Q['event'].n_unique()} events", flush=True)
    D = pl.concat(frames, how="vertical_relaxed")
    D = D.join(Uc.group_by("event").agg(pl.len().alias("legs_event")), on="event", how="left")
    D = history_features(D, U)
    _save(D.sort("event", "ticker", "lead_h"), OUT / "rows.parquet")
    keep = H.filter(pl.col("ticker").is_in(D["ticker"].unique().implode()))
    _save(keep, OUT / "hourly.parquet")
    cov = D.group_by("split", "lead_h").agg(pl.col("event").n_unique().alias("events"), pl.len().alias("legs")).sort("split", "lead_h")
    cov.write_csv(OUT / "coverage.csv")
    print(cov, flush=True)
    return {"rows": D.height, "events": D["event"].n_unique()}


# ---- word model (walk-forward logistic, monthly refits) ---------------------------------------------
FIRST_FIT = datetime(2025, 9, 1, tzinfo=UTC)
MIN_TRAIN = 400
L2 = 1.0
MODEL_FEATURES = ("lm", "lspread", "r_sw", "has_sw", "log_n_sw", "last_y", "r_w", "r_se", "thr", "lead12")
HIST_FEATURES = ("r_sw", "has_sw", "log_n_sw", "last_y", "r_w", "r_se", "thr", "lead12")


def logit(p: np.ndarray, lo: float = 0.005) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), lo, 1 - lo)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def design(D: pl.DataFrame, names: tuple[str, ...] = MODEL_FEATURES) -> np.ndarray:
    cols = {
        "lm": lambda: logit(D["mid"].to_numpy()),
        "lspread": lambda: np.log(np.clip(D["spread"].to_numpy(), 0.01, 0.5)),
        "r_sw": lambda: logit(D["r_sw"].to_numpy(), 0.02),
        "has_sw": lambda: (D["n_sw"].to_numpy() > 0).astype(float),
        "log_n_sw": lambda: np.log1p(D["n_sw"].to_numpy().astype(float)),
        "last_y": lambda: np.where(D["last_y"].is_null().to_numpy(), 0.0, D["last_y"].fill_null(0).to_numpy() - 0.5),
        "r_w": lambda: logit(D["r_w"].to_numpy(), 0.02),
        "r_se": lambda: logit(D["r_se"].to_numpy(), 0.02),
        "thr": lambda: D["thr"].cast(pl.Float64).to_numpy(),
        "lead12": lambda: (D["lead_h"].to_numpy() == 12).astype(float),
    }
    return np.column_stack([cols[n]() for n in names])


def fit_logit(X: np.ndarray, y: np.ndarray, l2: float = L2) -> dict[str, np.ndarray]:
    """L2 logistic regression on standardized features (Newton); intercept unpenalized."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    pen = np.full(Z.shape[1], l2)
    pen[0] = 0.0
    for _ in range(100):
        p = sigmoid(Z @ w)
        g = Z.T @ (p - y) + pen * w
        Hm = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(pen) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return {"w": w, "mu": mu, "sd": sd}


def predict(m: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    return sigmoid(np.column_stack([np.ones(len(X)), (X - m["mu"]) / m["sd"]]) @ m["w"])


def months(D: pl.DataFrame) -> list[datetime]:
    lo = max(FIRST_FIT, D["start"].min().replace(day=1, hour=0, minute=0, second=0, microsecond=0))
    hi = D["start"].max()
    out, m = [], lo
    while m <= hi:
        out.append(m)
        m = (m + timedelta(days=32)).replace(day=1)
    return out


def walk_forward(D: pl.DataFrame) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """For each month M from ``FIRST_FIT``: fit on rows whose event started before M - 1 day (all
    settled), predict rows starting in M. ``p_model`` uses the market; ``p_hist`` history only."""
    D = D.with_row_index("rid")
    preds, coefs = [], []
    for m in months(D):
        nxt = (m + timedelta(days=32)).replace(day=1)
        tr = D.filter(pl.col("start") < m - timedelta(days=1))
        te = D.filter((pl.col("start") >= m) & (pl.col("start") < nxt))
        if tr.height < MIN_TRAIN or not te.height:
            continue
        y = tr["y"].to_numpy().astype(float)
        mm = fit_logit(design(tr), y)
        mh = fit_logit(design(tr, HIST_FEATURES), y)
        preds.append(te.select("rid").with_columns(pl.Series("p_model", predict(mm, design(te))),
                                                   pl.Series("p_hist", predict(mh, design(te, HIST_FEATURES))),
                                                   pl.lit(m.strftime("%Y-%m")).alias("fit_month")))
        coefs.append({"month": m.strftime("%Y-%m"), "n_train": tr.height,
                      **{f"b_{n}": round(float(b), 4) for n, b in zip(("const", *MODEL_FEATURES), mm["w"], strict=True)}})
    P = pl.concat(preds) if preds else pl.DataFrame(schema={"rid": pl.UInt32, "p_model": pl.Float64, "p_hist": pl.Float64,
                                                            "fit_month": pl.String})
    return D.join(P, on="rid", how="left").drop("rid"), coefs


def scores(D: pl.DataFrame) -> pl.DataFrame:
    """Log loss and Brier of the frozen fair value (calibrated mid), the history-only model and the
    blend, by split and lead, on rows with a walk-forward prediction."""
    rows = []
    d = D.filter(pl.col("p_model").is_not_null())
    for (sp, L), g in d.group_by(["split", "lead_h"]):
        y = g["y"].to_numpy().astype(float)
        r: dict[str, Any] = {"split": sp, "lead_h": L, "legs": g.height, "events": g["event"].n_unique(), "ybar": round(y.mean(), 3)}
        for name in ("fair", "p_hist", "p_model"):
            p = np.clip(g[name].to_numpy(), 0.003, 0.997)
            r[f"logloss_{name}"] = round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 4)
            r[f"brier_{name}"] = round(float(((p - y) ** 2).mean()), 4)
        # paired per-event log-loss gain of the blend over the fair value, t clustered by event
        pf, pm = np.clip(g["fair"].to_numpy(), 0.003, 0.997), np.clip(g["p_model"].to_numpy(), 0.003, 0.997)
        gain = (y * np.log(pm) + (1 - y) * np.log(1 - pm)) - (y * np.log(pf) + (1 - y) * np.log(1 - pf))
        mu, t, _ = clustered_t(gain, g["event"].to_numpy())
        r["ll_gain_x1000"], r["ll_gain_t"] = round(mu * 1000, 2), None if t is None else round(t, 2)
        rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None).sort("split", "lead_h")


def clustered_t(x: np.ndarray, g: np.ndarray) -> tuple[float, float | None, int]:
    from albert.research import mentions_ext as X

    return X.clustered_t(np.asarray(x, dtype=float), np.asarray(g))


# ---- policies (orders, fills, PnL) ------------------------------------------------------------------
LIVE_M = 0.12
SIX = timedelta(hours=6)


def phases(D: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The live structure: early phase from start - 12 h to min(start - 6 h, breaker); main phase
    from start - 6 h to start - 10 min (or breaker), skipped for events whose early breaker fired
    before start - 6 h."""
    e = D.filter(pl.col("lead_h") == 12).with_columns(
        pl.min_horizontal(pl.col("end_cb"), pl.col("start") - SIX).alias("end_o"), pl.lit("early").alias("phase"))
    killed = e.filter(pl.col("kill").is_not_null() & (pl.col("kill") <= pl.col("start") - SIX))["event"].unique()
    m = D.filter((pl.col("lead_h") == 6) & ~pl.col("event").is_in(killed.implode())).with_columns(
        pl.col("end_cb").alias("end_o"), pl.lit("late").alias("phase"))
    return e.filter(pl.col("end_o") > pl.col("t")), m.filter(pl.col("end_o") > pl.col("t"))


def yes_fills(O: pl.DataFrame, H: pl.DataFrame, period: timedelta = timedelta(hours=1)) -> pl.DataFrame:
    """Mirror of ``mentions_ext.hourly_fills`` for a YES bid at ``price``: filled if a candle
    overlapping (t, end_o] printed strictly below it."""
    o = O.select("row", "ticker", "t", "end_o", "price")
    h = H.select("ticker", "end_ts", "lo").filter(pl.col("lo").is_not_null()).join(o, on="ticker")
    h = h.filter((pl.col("end_ts") > pl.col("t")) & (pl.col("end_ts") - period < pl.col("end_o"))
                 & (pl.col("lo") < pl.col("price") - EPS))
    f = h.group_by("row").agg(pl.col("end_ts").min().alias("hit_end"), pl.col("t").first(), pl.col("end_o").first())
    f = f.with_columns(pl.min_horizontal(pl.max_horizontal(pl.col("hit_end") - period / 2, pl.col("t") + timedelta(minutes=1)),
                                         pl.col("end_o")).alias("fill_ts"))
    return O.join(f.select("row", "fill_ts"), on="row", how="left").with_columns(pl.col("fill_ts").is_not_null().alias("filled"))


def price_cols(Q: pl.DataFrame, pol: dict[str, Any]) -> pl.DataFrame:
    """NO bid price (``price``; NaN = no order) and YES bid price (``yes_price``) for one policy."""
    from albert.research import mentions_ext as X

    fair, bid, ask = Q["fair"].to_numpy(), Q["bid"].to_numpy(), Q["ask"].to_numpy()
    pm = Q["p_model"].to_numpy()
    kind = pol["kind"]
    if kind == "live" or kind == "skip" or kind == "yes":
        px = X.no_bid(fair, bid, LIVE_M)
    elif kind == "mfair":
        px = X.no_bid(pm, bid, pol["m"])
    elif kind == "widen":
        px = X.no_bid(np.maximum(fair, pm), bid, LIVE_M)
    elif kind == "tight":
        px = X.no_bid(fair, bid, np.where(pm <= fair - pol["g"], 0.08, LIVE_M))
    else:
        raise ValueError(kind)
    if kind == "skip":
        px = np.where(pm >= (1 - px) - pol["theta"], np.nan, px)
    yp = np.full(len(px), np.nan)
    if kind == "yes":
        with np.errstate(invalid="ignore"):
            raw = np.minimum(np.floor((pm - pol["m"]) * 100 + 1e-9) / 100, np.round(ask - 0.01, 2))
        yp = np.where((pm - fair >= pol["g"]) & (raw >= 0.01 - EPS), np.round(raw, 2), np.nan)
    return Q.with_columns(pl.Series("price", px), pl.Series("yes_price", yp))


def policy_orders(D: pl.DataFrame, H: pl.DataFrame, pol: dict[str, Any]) -> dict[str, pl.DataFrame]:
    """Uncapped orders of one policy: early NO bids, main NO bids on legs the early phase did not
    fill, and (``yes`` policies) the same two phases for YES bids. ``book`` = NO orders in the shape
    ``mentions_budget.replay`` takes (the main order on every leg; the replay drops filled legs)."""
    from albert.research import mentions_ext as X

    e, m = phases(D)
    e, m = price_cols(e, pol), price_cols(m, pol)

    def no_side(Q: pl.DataFrame) -> pl.DataFrame:
        o = Q.filter(pl.col("price").is_not_nan()).with_row_index("row")
        return X.hourly_fills(o, H, "end_o").drop("row").with_columns(
            pl.when(pl.col("filled")).then((1 - pl.col("y")) - pl.col("price")).otherwise(0.0).alias("pnl"),
            pl.lit("no").alias("side"))

    ne, nm_all = no_side(e), no_side(m)
    early_filled = ne.filter("filled")["ticker"].unique()
    nm = nm_all.filter(~pl.col("ticker").is_in(early_filled.implode()))
    out = {"no": pl.concat([ne, nm], how="vertical_relaxed"), "book": pl.concat([ne, nm_all], how="vertical_relaxed")}
    if pol["kind"] == "yes":
        def yes_side(Q: pl.DataFrame) -> pl.DataFrame:
            o = Q.filter(pl.col("yes_price").is_not_nan()).drop("price").rename({"yes_price": "price"}).with_row_index("row")
            return yes_fills(o, H).drop("row").with_columns(
                pl.when(pl.col("filled")).then(pl.col("y") - pl.col("price")).otherwise(0.0).alias("pnl"),
                pl.lit("yes").alias("side"))
        ye = yes_side(e)
        ym = yes_side(m).filter(~pl.col("ticker").is_in(ye.filter("filled")["ticker"].unique().implode()))
        out["yes"] = pl.concat([ye, ym], how="vertical_relaxed")
    return out


def event_pnl(o: dict[str, pl.DataFrame]) -> pl.DataFrame:
    parts = [o["no"].select("event", "et_date", "split", "pnl")]
    if "yes" in o:
        parts.append(o["yes"].select("event", "et_date", "split", "pnl"))
    return pl.concat(parts).group_by("event", "et_date", "split").agg(pl.col("pnl").sum())


def fill_stats(f: pl.DataFrame) -> dict[str, Any]:
    if not f.height:
        return {"fills": 0}
    x = f["pnl"].to_numpy()
    mu, te, _ = clustered_t(x, f["event"].to_numpy())
    _, td, _ = clustered_t(x, f["et_date"].cast(pl.String).to_numpy())
    return {"fills": f.height, "events_filled": f["event"].n_unique(), "c_per_fill": round(mu * 100, 2),
            "t_event": None if te is None else round(te, 2), "t_date": None if td is None else round(td, 2),
            "pnl_usd": round(float(x.sum()), 2), "avg_price": round(float(f["price"].mean()), 3)}


# ---- grid, selection, holdout -----------------------------------------------------------------------
# Frozen before any outcome of this study was looked at (2026-11-23 01:10 UTC).
CELLS: tuple[dict[str, Any], ...] = (
    {"name": "live 12c", "kind": "live"},
    {"name": "skip p>=sell-0c", "kind": "skip", "theta": 0.0},
    {"name": "skip p>=sell-4c", "kind": "skip", "theta": 0.04},
    {"name": "skip p>=sell-8c", "kind": "skip", "theta": 0.08},
    {"name": "model fair +8c", "kind": "mfair", "m": 0.08},
    {"name": "model fair +12c", "kind": "mfair", "m": 0.12},
    {"name": "model fair +16c", "kind": "mfair", "m": 0.16},
    {"name": "max(fair,p)+12c", "kind": "widen"},
    {"name": "8c if p<=fair-4c", "kind": "tight", "g": 0.04},
    {"name": "8c if p<=fair-8c", "kind": "tight", "g": 0.08},
    {"name": "+YES p-4c if p>=fair+5c", "kind": "yes", "g": 0.05, "m": 0.04},
    {"name": "+YES p-8c if p>=fair+5c", "kind": "yes", "g": 0.05, "m": 0.08},
    {"name": "+YES p-4c if p>=fair+10c", "kind": "yes", "g": 0.10, "m": 0.04},
    {"name": "+YES p-8c if p>=fair+10c", "kind": "yes", "g": 0.10, "m": 0.08},
)
N_ALT = len(CELLS) - 1
T_BAR = 2.66            # one-sided Bonferroni z at 0.05 / 13
BOOK_POLICY = {"name": "E$15, 6 events, ceiling $40", "max_events": 6, "ceiling": 40.0}


def paired(ev: pl.DataFrame, base: pl.DataFrame) -> dict[str, Any]:
    """Policy minus live PnL per event (events of either side; a missing side is 0), mean per event
    and t clustered by ET date."""
    j = base.rename({"pnl": "b"}).join(ev.rename({"pnl": "a"}).select("event", "a"), on="event", how="full", coalesce=True)
    j = j.with_columns(pl.col("a").fill_null(0.0), pl.col("b").fill_null(0.0)).filter(pl.col("et_date").is_not_null())
    d = (j["a"] - j["b"]).to_numpy()
    if not len(d):
        return {}
    mu, t, _ = clustered_t(d, j["et_date"].cast(pl.String).to_numpy())
    return {"diff_usd": round(float(d.sum()), 2), "diff_c_per_event": round(mu * 100, 2),
            "diff_t_date": None if t is None else round(t, 2), "events_changed": int((np.abs(d) > EPS).sum())}


def book(o: pl.DataFrame, marks: pl.DataFrame, split: str, name: str) -> dict[str, Any]:
    """The live budget (expected fill cost $15, 6 events, $40 ceiling; ``mentions_budget.replay``
    with its dev fill model) on one policy's NO orders, scored with ``vol_metrics``."""
    from albert.research import mentions_budget as B
    from albert.research import mentions_ext as X

    res = json.loads((B.OUT / "results.json").read_text())
    pol = B.Policy(BOOK_POLICY["name"], max_events=BOOK_POLICY["max_events"], ceiling=BOOK_POLICY["ceiling"])
    oo = o.with_columns(pl.col("fill_ts").cast(pl.Datetime("us", "UTC")))
    bk, st = B.replay(oo, pol, np.array(res["w_dev"]), res["curve_dev"])
    if not bk.height:
        return {"book_bids": 0}
    led = X.ledger(bk.with_columns(pl.col("fill_ts").cast(pl.Datetime("us", "UTC"))), NAME, name, split)
    f = bk.filter("filled")
    out = {"book_bids": bk.height, "book_fills": f.height,
           "book_c_per_fill": round(float(f["pnl"].mean()) * 100, 2) if f.height else None,
           "book_pnl_usd": round(float(f["pnl"].sum()), 2)}
    out.update({f"book_{k}": v for k, v in X.book_metrics(led, marks).items()})
    out["book_peak_realized"] = st.get("peak_realized")
    if led.height:
        _save(led, OUT / "ledgers" / f"ledger_{name.replace(' ', '_').replace('/', '_')}_{split}.parquet")
    return out


def score_cell(D: pl.DataFrame, H: pl.DataFrame, marks: pl.DataFrame, cell: dict[str, Any], split: str,
               base_ev: pl.DataFrame | None, *, with_book: bool = True) -> tuple[dict[str, Any], pl.DataFrame]:
    s = D.filter(pl.col("split").is_in(SPLITS[split]))
    o = policy_orders(s, H, cell)
    ev = event_pnl(o)
    row: dict[str, Any] = {"cell": cell["name"], "split": split, "events": s["event"].n_unique(),
                           "no_orders": o["no"].height, **{f"no_{k}": v for k, v in fill_stats(o["no"].filter("filled")).items()}}
    if "yes" in o:
        row.update({"yes_orders": o["yes"].height, **{f"yes_{k}": v for k, v in fill_stats(o["yes"].filter("filled")).items()}})
    row["pnl_usd"] = round(float(ev["pnl"].sum()), 2)
    if base_ev is not None:
        row.update(paired(ev, base_ev))
    if with_book and cell["kind"] != "yes":
        row.update(book(o["book"], marks, split, cell["name"]))
    return row, ev


SPLITS = {"dev": ("dev",), "seen": ("seen",), "confirm": ("confirm",), "holdout": ("seen", "gap", "confirm")}


def choose(T: pl.DataFrame) -> str | None:
    """Pre-specified: among dev alternatives with paired gain t (by ET date) >= ``T_BAR`` and a dev
    capped book at least as large as live's (YES cells: uncapped gain only), the largest dev gain."""
    d = T.filter(pl.col("split") == "dev")
    live_book = d.filter(pl.col("cell") == CELLS[0]["name"])["book_pnl_usd"][0]
    ok = d.filter((pl.col("cell") != CELLS[0]["name"]) & (pl.col("diff_t_date").fill_null(-99) >= T_BAR)
                  & (pl.col("book_pnl_usd").is_null() | (pl.col("book_pnl_usd").fill_null(0) >= live_book)))
    return ok.sort("diff_usd", descending=True)["cell"][0] if ok.height else None


def edge_table(D: pl.DataFrame, H: pl.DataFrame, split: str = "dev") -> pl.DataFrame:
    """Descriptive (dev only): live-rule NO fills by the model's edge at our sell price (1 - price -
    p_model) and by the model-vs-fair gap."""
    o = policy_orders(D.filter(pl.col("split") == split), H, CELLS[0])["no"].filter("filled")
    o = o.with_columns((1 - pl.col("price") - pl.col("p_model")).alias("edge"), (pl.col("p_model") - pl.col("fair")).alias("gap"))
    rows = []
    for col, cuts in (("edge", [-1, 0.0, 0.04, 0.08, 0.12, 0.16, 2]), ("gap", [-1, -0.08, -0.04, 0.0, 0.04, 0.08, 2])):
        for lo, hi in itertools.pairwise(cuts):
            g = o.filter((pl.col(col) >= lo) & (pl.col(col) < hi))
            rows.append({"by": col, "lo": lo, "hi": hi, **fill_stats(g),
                         "p_model_mean": round(float(g["p_model"].mean()), 3) if g.height else None,
                         "y_mean": round(float(g["y"].mean()), 3) if g.height else None,
                         "sell_mean": round(float((1 - g["price"]).mean()), 3) if g.height else None})
    return pl.DataFrame(rows, infer_schema_length=None)


BOOK_FROM = datetime(2026, 8, 17, tzinfo=UTC)   # dev book window (whole-life candles; as mentions_budget)


def run(cells: tuple[dict[str, Any], ...] = CELLS) -> dict[str, Any]:
    from albert.research import mentions_ext as X

    D = pl.read_parquet(OUT / "rows.parquet")
    H = pl.read_parquet(OUT / "hourly.parquet")
    marks = X.marks_from(H)
    D, coefs = walk_forward(D)
    _save(D, OUT / "predictions.parquet")
    pl.DataFrame(coefs).write_csv(OUT / "coefs.csv")
    S = scores(D)
    S.write_csv(OUT / "model_scores.csv")
    print(S, flush=True)
    Dp = D.filter(pl.col("p_model").is_not_null())
    E = edge_table(Dp, H)
    E.write_csv(OUT / "edge_dev.csv")
    # dev grid: uncapped over every dev row with a walk-forward prediction; book from BOOK_FROM
    rows = []
    _, base = score_cell(Dp, H, marks, cells[0], "dev", None, with_book=False)
    Db = Dp.filter((pl.col("split") != "dev") | (pl.col("start") >= BOOK_FROM))
    for c in cells:
        r, _ = score_cell(Dp, H, marks, c, "dev", base, with_book=False)
        if c["kind"] != "yes":
            o = policy_orders(Db.filter(pl.col("split") == "dev"), H, c)
            r.update(book(o["book"], marks, "dev", c["name"]))
        rows.append(r)
        print(json.dumps(r, default=str), flush=True)
    T = pl.DataFrame(rows, infer_schema_length=None)
    T.write_csv(OUT / "grid_dev.csv")
    chosen = choose(T)
    best = (T.filter((pl.col("split") == "dev") & (pl.col("cell") != cells[0]["name"]))
            .sort("diff_usd", descending=True)["cell"][0])
    carry = [cells[0]] + [c for c in cells if c["name"] in {chosen, best} - {None}]
    hold = []
    for sp in ("seen", "confirm", "holdout"):
        _, b = score_cell(Dp, H, marks, cells[0], sp, None, with_book=False)
        for c in carry:
            r, _ = score_cell(Dp, H, marks, c, sp, b)
            r["role"] = "live" if c is cells[0] else ("chosen" if c["name"] == chosen else "best dev (descriptive)")
            hold.append(r)
            print(json.dumps(r, default=str), flush=True)
    TH = pl.DataFrame(hold, infer_schema_length=None)
    TH.write_csv(OUT / "holdout.csv")
    info = {"chosen": chosen, "best_dev": best, "cells": len(cells), "t_bar": T_BAR, "book_policy": BOOK_POLICY,
            "first_fit": FIRST_FIT.isoformat(), "rows": D.height, "rows_pred": Dp.height,
            "splits": {k: list(v) for k, v in SPLITS.items()}}
    (OUT / "results.json").write_text(json.dumps(info, indent=1, default=str))
    return {"scores": S, "grid": T, "holdout": TH, "edge": E, **info}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m albert.research.{NAME}")
    ap.add_argument("cmd", choices=["fetch", "build", "run", "report"])
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch()
    elif a.cmd == "build":
        print(build())
    elif a.cmd == "run":
        run()
    elif a.cmd == "report":
        write_report()



# ---- report -----------------------------------------------------------------------------------------
def md_table(df: pl.DataFrame, cols: list[str] | None = None) -> str:
    if cols:
        df = df.select([c for c in cols if c in df.columns])
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join("" if v is None else str(v) for v in r) + " |\n" for r in df.iter_rows())


GRID_COLS = ["cell", "split", "role", "events", "no_orders", "no_fills", "no_c_per_fill", "no_t_date", "yes_orders", "yes_fills",
             "yes_c_per_fill", "yes_t_date", "pnl_usd", "diff_usd", "diff_c_per_event", "diff_t_date", "events_changed",
             "book_fills", "book_c_per_fill", "book_pnl_usd", "book_usd_per_week", "book_ann_vol_pct", "book_sharpe",
             "book_sharpe_ci90", "book_sharpe_ex_top5", "book_max_dd_usd", "book_weeks_positive_pct"]


def write_report() -> Path:
    """``report.md`` = the hand-written narrative (``narrative.md``: verdict, hypotheses, sources,
    interpretation) followed by the generated tables of the last run."""
    parts = [(OUT / "narrative.md").read_text() if (OUT / "narrative.md").exists() else "# mentions_model\n"]
    for title, name, cols in (("Model scores (walk-forward, monthly refits)", "model_scores.csv", None),
                              ("Dev grid (14 cells; uncapped on every dev row with a prediction, book from 2026-08-17)",
                               "grid_dev.csv", GRID_COLS),
                              ("Holdout (frozen): seen = Oct 1 - Nov 16, confirm = Nov 17 - 22, holdout = all from Oct 1",
                               "holdout.csv", GRID_COLS),
                              ("Dev only: live-rule fills by model edge at our sell price and by model - fair",
                               "edge_dev.csv", None),
                              ("Walk-forward coefficients (standardized features)", "coefs.csv", None),
                              ("Coverage", "coverage.csv", None)):
        p = OUT / name
        if p.exists():
            parts.append(f"\n## {title}\n\n" + md_table(pl.read_csv(p, infer_schema_length=None), cols))
    if (OUT / "results.json").exists():
        parts.append("\n## Run info\n\n```\n" + (OUT / "results.json").read_text() + "\n```\n")
    out = OUT / "report.md"
    out.write_text("\n".join(parts))
    return out


if __name__ == "__main__":
    main()
