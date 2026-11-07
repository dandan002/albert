"""Mentions, round 2: passive (maker) quoting around a transcript + price fair value.

(The Factba.se transcript data used below was removed on 2026-11-06 because rollcall.com's
robots.txt blocks AI crawlers; the transcript paths need a permitted source to run again.)

Round 1 (:mod:`albert.research.mentions`) found that Factba.se transcript word rates for Trump
and Vance add a little information to the Kalshi price (blend c ≈ 0.45, t 1.4–1.7) but that
taking loses, because the legs are 2–7c wide before the event. This study asks whether a
*maker* earns: rest a YES bid and a NO bid around a dev-fitted fair value from a decision time
before the event until just before the scheduled start, filled only when a later public trade
prints strictly through the bid (the harness rule, :func:`albert.research.fundamental.maker_fills`),
held to settlement.

Design (every input as of the decision time ``t``):

* **Events / legs.** Every settled Mentions speech leg (window series "say X before <date>"
  excluded: they have no start). Event start = Kalshi milestone ``start_date`` (round 1).
* **Decision times** ``t = start − 24 h / 6 h / 1 h`` (labels ``h24``, ``h6``, ``h1``); a leg
  must be listed and still open at ``t``; quote = last hourly candle closed by ``t`` (≤ 2 h old).
  Outcome-blind exclusion (as round 1): an (event, t) whose share of legs bid ≥ 0.97 at ``t`` is
  > 0.12 above the share at ``start − 24 h`` (or > 0.12 at ``h24``) is dropped: the speech
  probably came earlier than the milestone.
* **Orders rest from ``t`` until ``end = start − 10 min``** (never during the speech), or the
  leg's close if earlier. Variant ``cb`` (circuit breaker, live-implementable from the public
  trade feed): cancel every order of the event at the first print ≥ 95c on a leg whose ask at
  ``t`` was ≤ 80c (a word was just said: the speech started early).
* **Base rates** ``p_bb``: round 1's hierarchical beta-binomial (speech parameters tuned on dev
  in round 1), counts as of ``t`` (legs settled before ``t``).
* **Transcript model** (Trump, Vance): round 1's negative-binomial word-rate model on Factba.se
  transcripts, with a stricter per-row availability rule: a document dated ``d`` (ET) is used
  only if ``d + 1 day, 10:00 ET ≤ t``. Improvements tried here (selected on an inner dev split):
  speech type from the Kalshi title + milestone title (rally / tele-rally / press / interview /
  address / remarks), scheduled duration (only when the milestone was last updated before
  ``t``), topical salience (the word's rate in the speaker's last 7 days of transcripts vs the
  45-day rate), document frequency, multi-word / alternative / threshold legs.
* **Fair value** = logistic blend ``logit P = a + b·logit(mid) + c·logit(model) [+ d·logit(p_bb)]``
  fitted on dev (t < 2026-10-01), pooled over the three decision times. Controls: the
  dev-calibrated mid alone (``a + b·logit(mid)``) and the raw mid.
* **Maker test**: harness rule (bids at floor(fair − margin) and floor(1 − fair − margin), capped
  one tick inside the opposite quote, trade-through fills, maker fee by series — 0 on every
  Mentions series, all ``quadratic`` taker-only). Margin chosen on dev. Variants: both sides,
  the side the fair value favours only, and that side only when |fair − mid| ≥ x.

Entry point: :func:`run` then :func:`write_report`. Caches: ``data/cache/fundamental_mentions_maker_*``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from albert.research import mentions as M

NAME = "mentions_maker"
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
DEV_SPLIT = datetime(2026, 9, 15, tzinfo=UTC)          # inner split of dev for model selection
LEADS = {"h24": 24, "h6": 6, "h1": 1}
END_MARGIN = timedelta(minutes=10)
BB_SPEECH = M.BB(k_s=300, k_se=100, k_x=10, k_w=8, half_life=45.0)   # round 1, tuned on dev
TX_PUBLIC_AFTER = timedelta(hours=10)                    # doc dated d is public by d+1 10:00 ET
MARGINS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
GAPS = (0.01, 0.02, 0.03, 0.05)
CB_PRINT, CB_ASK = 0.95, 0.80


# ---- speech type, duration, availability ------------------------------------------------------
def speech_type(title: str | None, ms_title: str | None) -> str:
    """Speech type from the Kalshi event title and the milestone title (round-2 classifier)."""
    t = f"{title or ''} | {ms_title or ''}".lower()
    if any(k in t for k in ("tele-rally", "telerally", "tele rally")):
        return "telerally"
    if "rally" in t or "speaks in " in t:
        return "rally"
    if any(k in t for k in ("press conference", "press briefing", "briefing", "gaggle", "presser",
                            "news conference", "media availability")):
        return "press"
    if any(k in t for k in ("interview", "fox news", "hannity", "ingraham", "podcast", "town hall", "townhall",
                            "60 minutes", "sunday night", "meet the press", "face the nation", "fox & friends",
                            "cnbc", "joins ", "appears on")):
        return "interview"
    if any(k in t for k in ("address", "convention", "united nations", "general assembly", "joint session",
                            "commencement", "keynote", "summit")):
        return "address"
    return "remarks"


def fb_type2(doc_type: str | None, title: str | None) -> str:
    """Factba.se document type on the same scale: round 1's type, with non-rally ``Speech``
    documents split out as ``address``."""
    base = M.fb_type(doc_type, title)
    if base == "remarks" and doc_type == "Speech":
        return "address"
    return base


def tx_cutoff(t: datetime) -> date:
    """Latest Factba.se document date usable at ``t``: a document dated ``d`` (US Eastern) is
    assumed public by ``d + 1 day, 10:00 ET``."""
    from zoneinfo import ZoneInfo

    return (t - TX_PUBLIC_AFTER).astimezone(ZoneInfo(M.ET)).date() - timedelta(days=1)


# ---- decision rows -----------------------------------------------------------------------------
def decision_rows(lg: pl.DataFrame, starts: pl.DataFrame, candles: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    """One row per (speech leg, lead) with the quote at ``t``, ``end`` and the outcome-blind
    early-start exclusion applied."""
    from albert.research.fundamental import attach_quotes

    sp = lg.filter(~pl.col("series").is_in(M.WINDOW_SERIES)).join(
        starts.select("event", "start", pl.col("end").alias("ms_end"), "ms_title", "ms_updated"), on="event")
    parts = [sp.with_columns((pl.col("start") - timedelta(hours=h)).alias("t"), pl.lit(lab).alias("when"))
             for lab, h in LEADS.items()]
    D = pl.concat(parts).filter((pl.col("open_time") <= pl.col("t")) & (pl.col("close_time") > pl.col("t")))
    Q = attach_quotes(D, candles, max_age=timedelta(hours=2))
    # share of the event's quoted legs already bid >= 0.97 at t (a said word goes to 0.99 at once)
    sh = Q.group_by("event", "when").agg((pl.col("bid") >= 0.97).mean().alias("said"))
    base = sh.filter(pl.col("when") == "h24").select("event", pl.col("said").alias("said24"))
    sh = sh.join(base, on="event", how="left").with_columns(pl.col("said24").fill_null(0.0))
    bad = sh.filter(pl.when(pl.col("when") == "h24").then(pl.col("said") > 0.12)
                    .otherwise(pl.col("said") > pl.col("said24") + 0.12)).select("event", "when")
    Q = Q.join(bad, on=["event", "when"], how="anti")
    Q = Q.with_columns(pl.min_horizontal(pl.col("start") - END_MARGIN, pl.col("close_time")).alias("end"),
                       ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                       (pl.col("ms_updated") <= pl.col("start")).alias("ms_pre"))
    info = {"rows_by_lead": Q.group_by("when").agg(pl.len().alias("rows"), pl.col("event").n_unique().alias("events"))
            .sort("when").to_dicts(), "excluded_event_leads": bad.height}
    return Q, info


def add_base_rates(Q: pl.DataFrame, lg: pl.DataFrame) -> pl.DataFrame:
    """Round-1 base rate ``p_bb`` as of ``t`` and the speaker's past listings of the word."""
    C = M.asof_counts(lg, Q, (0.0, BB_SPEECH.half_life))
    C = C.with_columns(pl.Series("p_bb", M.bb_predict(C, BB_SPEECH)))
    return C.select(*Q.columns, "n_sw_0", "p_bb")


# ---- transcript features ------------------------------------------------------------------------
def salience(q: pl.DataFrame, docs: pl.DataFrame, counts: pl.DataFrame, *, days: int = 7, base_days: int = 45,
             m_words: float = 3000.0) -> pl.DataFrame:
    """Topical salience per row (needs speaker, word, tx_hi): the word's rate in the speaker's
    documents of the last ``days`` days up to ``tx_hi``, shrunk toward the ``base_days`` rate with
    ``m_words`` pseudo-words, as ``sal = log(rate_7 / rate_45)``; also raw counts ``c7``, ``c45``."""
    dmeta = docs.select("document_id", "who", "date", "words")
    cc = counts.join(dmeta, on=["document_id", "who"], how="inner")
    out = []
    for who, hi in q.select("speaker", "tx_hi").unique().iter_rows():
        lo45, lo7 = hi - timedelta(days=base_days - 1), hi - timedelta(days=days - 1)
        dm = dmeta.filter((pl.col("who") == who) & pl.col("date").is_between(lo45, hi))
        w45 = float(dm["words"].sum())
        w7 = float(dm.filter(pl.col("date") >= lo7)["words"].sum())
        c = cc.filter((pl.col("who") == who) & pl.col("date").is_between(lo45, hi))
        agg = c.group_by("word").agg(pl.col("c").sum().alias("c45"),
                                     pl.col("c").filter(pl.col("date") >= lo7).sum().alias("c7"))
        out.append(agg.with_columns(pl.lit(who).alias("speaker"), pl.lit(hi).alias("tx_hi"),
                                    pl.lit(w7).alias("w7"), pl.lit(w45).alias("w45")))
    S = pl.concat(out, how="vertical_relaxed") if out else pl.DataFrame()
    keys = S.select("speaker", "tx_hi", "w7", "w45").unique(["speaker", "tx_hi"])
    q = q.join(S.select("speaker", "tx_hi", "word", "c7", "c45"), on=["speaker", "tx_hi", "word"], how="left")
    q = q.join(keys, on=["speaker", "tx_hi"], how="left").with_columns(
        pl.col("c7").fill_null(0).cast(pl.Float64), pl.col("c45").fill_null(0).cast(pl.Float64),
        pl.col("w7").fill_null(0.0), pl.col("w45").fill_null(0.0))
    r45 = (pl.col("c45") + 0.5) / (pl.col("w45") + 1000.0)
    r7 = (pl.col("c7") + m_words * r45) / (pl.col("w7") + m_words)
    return q.with_columns((r7 / r45).log().alias("sal"))


def add_transcripts(Q: pl.DataFrame, docs: pl.DataFrame, counts: pl.DataFrame) -> pl.DataFrame:
    """Transcript features for Trump / Vance rows: ``p_tx1`` (round-1 types), ``p_tx2`` (round-2
    types), ``p_df2``, ``mu_tx2``, ``sal``, ``ldur``, ``phrase``, ``alts``, ``thr``."""
    q = Q.filter(pl.col("speaker").is_in(list(M.FB_SPEAKER)))
    q = q.with_columns(pl.col("t").map_elements(tx_cutoff, return_dtype=pl.Date).alias("tx_hi"),
                       pl.struct("title", "etype").map_elements(lambda r: M.kalshi_ftype(r["title"], r["etype"]),
                                                                return_dtype=pl.String).alias("ftype1"),
                       pl.struct("title", "ms_title").map_elements(lambda r: speech_type(r["title"], r["ms_title"]),
                                                                   return_dtype=pl.String).alias("ftype2"))
    # transcript_predict uses documents dated in [event_date − lookback, event_date − min_lag]
    q = q.with_columns((pl.col("tx_hi") + timedelta(days=1)).alias("event_date"))
    prm = replace(M.TX_PARAMS, min_lag_days=1)
    keep = ["ticker", "when"]
    t1 = M.transcript_predict(q.with_columns(pl.col("ftype1").alias("ftype")), docs, counts, prm)
    t1 = t1.select(*keep, pl.col("p_tx").alias("p_tx1"), pl.col("mu_tx").alias("mu_tx1"), "n_docs")
    docs2 = docs.with_columns(pl.struct("type", "title").map_elements(lambda r: fb_type2(r["type"], r["title"]),
                                                                      return_dtype=pl.String).alias("ftype"))
    t2 = M.transcript_predict(q.with_columns(pl.col("ftype2").alias("ftype")), docs2, counts, prm)
    t2 = t2.select(*keep, pl.col("p_tx").alias("p_tx2"), pl.col("p_df").alias("p_df2"), pl.col("mu_tx").alias("mu_tx2"),
                   pl.col("x_e").alias("x_e2"))
    q = q.join(t1, on=keep, how="left").join(t2, on=keep, how="left")
    q = salience(q, docs, counts)
    # scheduled duration, only if the milestone was last edited before t (else the type median)
    dur = ((pl.col("ms_end") - pl.col("start")).dt.total_minutes() / 60).clip(0.25, 6.0)
    q = q.with_columns(pl.when(pl.col("ms_updated") <= pl.col("t")).then(dur).otherwise(None).alias("dur_h"))
    med = q.filter(pl.col("t") < HOLDOUT_FROM).group_by("ftype2").agg(pl.col("dur_h").median().alias("dur_med"))  # dev only
    q = q.join(med, on="ftype2", how="left").with_columns(
        (pl.col("dur_h").fill_null(pl.col("dur_med")) / pl.col("dur_med")).log().fill_null(0.0).alias("ldur"),
        (pl.col("n_tokens") > 1).cast(pl.Float64).alias("phrase"), (pl.col("n_alts") > 1).cast(pl.Float64).alias("alts"),
        (pl.col("k") > 1).cast(pl.Float64).alias("thr"))
    cols = ["p_tx1", "mu_tx1", "n_docs", "p_tx2", "p_df2", "mu_tx2", "x_e2", "sal", "c7", "c45", "ldur", "phrase",
            "alts", "thr", "ftype1", "ftype2", "tx_hi"]
    return Q.join(q.select(*keep, *cols), on=keep, how="left")


# ---- calibration / blends (dev fits) -----------------------------------------------------------
TX_VARIANTS: dict[str, list[str]] = {
    # name: design columns (``L:<col>`` = logit of a probability column, else the raw column)
    "tx0": ["L:p_tx1"],                                      # round 1 txc, stricter availability
    "tx1": ["L:p_tx2"],                                      # round-2 speech types
    "tx2": ["L:p_tx2", "sal"],                               # + topical salience
    "tx3": ["L:p_tx2", "sal", "phrase", "alts", "thr"],      # + leg form
    "tx4": ["L:p_tx2", "L:p_df2", "sal", "phrase", "alts", "thr", "ldur"],   # + doc frequency, duration
}


def design(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    from albert.research.fundamental import logit

    X = [np.ones(df.height)]
    for c in cols:
        x = df[c[2:]].to_numpy().astype(float) if c.startswith("L:") else df[c].to_numpy().astype(float)
        X.append(logit(x) if c.startswith("L:") else x)
    return np.column_stack(X)


def fit(df: pl.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    from albert.research.fundamental import fit_logistic

    return fit_logistic(design(df, cols), df["y"].to_numpy().astype(float), df["event"].to_numpy())


def predict(df: pl.DataFrame, cols: list[str], beta: np.ndarray) -> np.ndarray:
    return np.clip(1 / (1 + np.exp(-design(df, cols) @ beta)), 0.003, 0.997)


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    q = np.clip(p, 0.005, 0.995)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def select_tx(tv: pl.DataFrame) -> tuple[str, list[dict[str, Any]]]:
    """Inner dev split: fit each transcript variant and its blend with the mid and ``p_bb`` on
    t < Sep 15, score the blend on Sep 15–30 by log loss. Returns the best variant and the table."""
    fit_part = tv.filter(pl.col("t") < DEV_SPLIT)
    test = tv.filter((pl.col("t") >= DEV_SPLIT) & (pl.col("t") < HOLDOUT_FROM))
    y = test["y"].to_numpy().astype(float)
    rows = [{"variant": "mid", "logloss_inner": round(logloss(test["mid"].to_numpy(), y), 5)}]
    for name, cols in (("mid_cal", ["L:mid"]), ("mid+bb", ["L:mid", "L:p_bb"])):
        b, _ = fit(fit_part, cols)
        rows.append({"variant": name, "logloss_inner": round(logloss(predict(test, cols, b), y), 5)})
    best = None
    for name, cols in TX_VARIANTS.items():
        b, t = fit(fit_part, cols)
        f2 = fit_part.with_columns(pl.Series("p_m", predict(fit_part, cols, b)))
        te = test.with_columns(pl.Series("p_m", predict(test, cols, b)))
        bcols = ["L:mid", "L:p_m", "L:p_bb"]
        bb, bt = fit(f2, bcols)
        ll = logloss(predict(te, bcols, bb), y)
        rows.append({"variant": name, "logloss_model_inner": round(logloss(te["p_m"].to_numpy(), y), 5),
                     "logloss_inner": round(ll, 5), "c_model": round(float(bb[2]), 3), "t_c": round(float(bt[2]), 2),
                     "coef": [round(float(x), 3) for x in b], "t": [round(float(x), 2) for x in t]})
        if best is None or ll < best[1]:
            best = (name, ll)
    return best[0], rows


def fair_values(Q: pl.DataFrame, tx_variant: str) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Dev-fitted (t < Oct 1) fair values on every row:

    * ``fv_tv``  (Trump/Vance): blend of mid, calibrated transcript model and ``p_bb``;
    * ``fv_bb``  (every speech leg): blend of mid and ``p_bb``;
    * ``fv_cal`` (every leg): the calibrated mid ``a + b·logit(mid)`` (control);
    * ``fv_mid``: the raw mid (control)."""
    info: dict[str, Any] = {}
    dev = Q.filter(pl.col("t") < HOLDOUT_FROM)
    tvmask = pl.col("p_tx2").is_not_null() & pl.col("p_tx1").is_not_null()
    tv_dev = dev.filter(tvmask)
    cols = TX_VARIANTS[tx_variant]
    b_tx, t_tx = fit(tv_dev, cols)
    info["tx_calibration"] = {"variant": tx_variant, "cols": cols, "coef": [round(float(x), 4) for x in b_tx],
                              "t": [round(float(x), 2) for x in t_tx]}
    tv = Q.filter(tvmask)
    tv = tv.with_columns(pl.Series("p_txc", predict(tv, cols, b_tx)))
    Q = Q.join(tv.select("ticker", "when", "p_txc"), on=["ticker", "when"], how="left")
    dev = Q.filter(pl.col("t") < HOLDOUT_FROM)
    hold = Q.filter(pl.col("t") >= HOLDOUT_FROM)
    specs = {"fv_tv": (["L:mid", "L:p_txc", "L:p_bb"], pl.col("p_txc").is_not_null()),
             "fv_bb": (["L:mid", "L:p_bb"], pl.lit(True)),
             "fv_cal": (["L:mid"], pl.lit(True))}
    for name, (bcols, mask) in specs.items():
        d = dev.filter(mask)
        b, t = fit(d, bcols)
        rec = {"cols": bcols, "coef": [round(float(x), 4) for x in b], "t": [round(float(x), 2) for x in t],
               "n_dev": d.height, "events_dev": d["event"].n_unique()}
        h = hold.filter(mask)
        if h.height >= 50:
            bh, th = fit(h, bcols)
            rec.update({"coef_holdout_refit": [round(float(x), 4) for x in bh], "t_holdout_refit": [round(float(x), 2) for x in th]})
            yh = h["y"].to_numpy().astype(float)
            rec.update({"logloss_holdout": round(logloss(predict(h, bcols, b), yh), 5),
                        "logloss_mid_holdout": round(logloss(h["mid"].to_numpy(), yh), 5),
                        "brier_holdout": round(float(((predict(h, bcols, b) - yh) ** 2).mean()), 5),
                        "brier_mid_holdout": round(float(((h["mid"].to_numpy() - yh) ** 2).mean()), 5)})
        info[name] = rec
        sub = Q.filter(mask)
        Q = Q.join(sub.select("ticker", "when").with_columns(pl.Series(name, predict(sub, bcols, b))),
                   on=["ticker", "when"], how="left")
    Q = Q.with_columns(pl.col("mid").alias("fv_mid"))
    return Q, info


# ---- maker fills (fast equivalent of the harness rule) -----------------------------------------
def kill_times(Q: pl.DataFrame, trades: pl.DataFrame, *, cb_print: float = CB_PRINT,
               cb_ask: float = CB_ASK) -> pl.DataFrame:
    """Circuit breaker per (event, lead): the first print after ``t`` at ≥ ``cb_print`` on any leg
    of the event whose ask at ``t`` was ≤ ``cb_ask`` (a word was just said). Uses only the public
    trade feed, so a live quoter can do the same."""
    legs = Q.filter(pl.col("ask") <= cb_ask).select("event", "when", "ticker", "t")
    tr = trades.filter(pl.col("yes_price") >= cb_print).select("ticker", pl.col("created_time").alias("ts"))
    k = legs.join(tr, on="ticker").filter(pl.col("ts") > pl.col("t"))
    return k.group_by("event", "when").agg(pl.col("ts").min().alias("kill_ts"))


def fast_fills(pred: pl.DataFrame, trades: pl.DataFrame, margin: float, p_col: str = "p",
               end_col: str = "end", rule: str = "through") -> pl.DataFrame:
    """Same rule and output as :func:`albert.research.fundamental.maker_fills` (one YES and one NO
    bid per row, trade-through fills, held to settlement, maker fee by series), vectorised per
    ticker with numpy. Adds ``row`` (the row index in ``pred``).

    ``rule="improve"`` (lenient variant): when our bid is strictly better than the same-side best
    bid at ``t`` (YES bid above the yes bid; NO bid above 1 − yes ask), a later print *at* our
    price also fills it: any other order at that price arrived after ours, so we are first in the
    queue. Otherwise the harness trade-through rule applies."""
    from albert.research.fundamental import maker_fee

    tr = trades.select("ticker", pl.col("created_time").alias("ts"), "yes_price", "taker_side").sort("ticker", "ts")
    by: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for (tk,), g in tr.group_by(["ticker"]):
        by[tk] = (g["ts"].dt.epoch("us").to_numpy(), g["yes_price"].to_numpy().astype(float),
                  (g["taker_side"] == "yes").to_numpy())
    P = pred.with_row_index("row")
    tick = P["ticker"].to_list()
    t0 = P["t"].dt.epoch("us").to_numpy()
    t1 = P[end_col].dt.epoch("us").to_numpy()
    p = P[p_col].to_numpy().astype(float)
    bid, ask, y = P["bid"].to_numpy(), P["ask"].to_numpy(), P["y"].to_numpy().astype(float)
    ft = P["fee_type"].to_list() if "fee_type" in P.columns else [None] * P.height
    fm = P["fee_multiplier"].to_list() if "fee_multiplier" in P.columns else [1.0] * P.height
    rows = []
    for i in range(P.height):
        yb = min(math.floor((p[i] - margin) * 100 + 1e-9) / 100, round(ask[i] - 0.01, 2))
        nb = min(math.floor((1 - p[i] - margin) * 100 + 1e-9) / 100, round(1 - bid[i] - 0.01, 2))
        g = by.get(tick[i])
        if g is not None:
            lo, hi = np.searchsorted(g[0], t0[i], "right"), np.searchsorted(g[0], t1[i], "right")
            ts, yp, ys = g[0][lo:hi], g[1][lo:hi], g[2][lo:hi]
        else:
            ts = yp = ys = np.empty(0)
        mult = fm[i] or 1.0
        for side, price in (("yes", yb), ("no", nb)):
            if price < 0.01:
                continue
            hit = None
            if len(ts):
                at = 1e-9
                if rule == "improve" and ((side == "yes" and price > bid[i] + 1e-9) or (side == "no" and price > 1 - ask[i] + 1e-9)):
                    at = -1e-9
                m = (~ys & (yp < price - at)) if side == "yes" else (ys & (yp > 1 - price + at))
                if m.any():
                    hit = int(ts[np.argmax(m)])
            won = y[i] if side == "yes" else 1 - y[i]
            rows.append({"row": i, "side": side, "price": price, "filled": hit is not None,
                         "fill_us": hit, "pnl": (won - price - maker_fee(price, ft[i], mult)) if hit is not None else 0.0})
    return pl.DataFrame(rows, schema={"row": pl.UInt32, "side": pl.String, "price": pl.Float64, "filled": pl.Boolean,
                                      "fill_us": pl.Int64, "pnl": pl.Float64})


def summarize(f: pl.DataFrame) -> dict[str, Any]:
    """Orders, fills, fill rate, PnL per fill and per order (cents), t by event."""
    from albert.research.fundamental import clustered_mean

    filled = f.filter("filled")
    row: dict[str, Any] = {"orders": f.height, "fills": filled.height,
                           "fill_rate": round(filled.height / f.height, 3) if f.height else None}
    if filled.height:
        mu, tt, k = clustered_mean(filled["pnl"].to_numpy(), filled["event"].to_numpy())
        row.update({"events": k, "pnl_per_fill_c": round(mu * 100, 3), "t": None if tt is None else round(tt, 2),
                    "pnl_per_order_c": round(float(f["pnl"].sum()) / f.height * 100, 3),
                    "pnl_total_$": round(float(filled["pnl"].sum()), 2),
                    "yes_share": round(float((filled["side"] == "yes").mean()), 3)})
    return row


def side_variant(f: pl.DataFrame, variant: str) -> pl.DataFrame:
    """``both``; ``fav`` = only the side the fair value favours (YES bid if fair > mid, NO bid if
    fair < mid; no order if equal); ``gapX`` = ``fav`` and |fair − mid| ≥ X."""
    if variant == "both":
        return f
    fav = f.filter(((pl.col("side") == "yes") & (pl.col("gap") > 1e-9)) | ((pl.col("side") == "no") & (pl.col("gap") < -1e-9)))
    if variant == "fav":
        return fav
    x = float(variant.removeprefix("gap"))
    return fav.filter(pl.col("gap").abs() >= x - 1e-9)


def maker_grid(Q: pl.DataFrame, trades: pl.DataFrame, fv: str, *, end_col: str = "end", rule: str = "through",
               variants: tuple[str, ...] = ("both", "fav", *(f"gap{g}" for g in GAPS)),
               margins: tuple[float, ...] = MARGINS) -> pl.DataFrame:
    """Fills of every row for every margin, with the row attributes needed for segmenting."""
    attrs = ["ticker", "event", "series", "sgroup", "when", "t", "split", "band", "wband", "fband", "ms_pre", "mid", fv]
    attrs = [a for a in dict.fromkeys(attrs) if a in Q.columns]
    base = Q.with_row_index("row").select("row", *attrs).with_columns((pl.col(fv) - pl.col("mid")).alias("gap"))
    out = []
    for m in margins:
        f = fast_fills(Q, trades, m, p_col=fv, end_col=end_col, rule=rule).join(base, on="row")
        out.append(f.with_columns(pl.lit(m).alias("margin")))
    g = pl.concat(out)
    return pl.concat([side_variant(g, v).with_columns(pl.lit(v).alias("variant")) for v in variants])


def maker_table(G: pl.DataFrame, by: tuple[str, ...] = ("when", "variant"), *, min_fills: int = 30) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per group in ``by``: every margin's dev and holdout result, and the margin chosen on dev
    (best PnL per fill with ≥ ``min_fills`` dev fills) with its holdout result."""
    rows = []
    for key, g in G.group_by([*by, "margin", "split"]):
        rows.append({**dict(zip([*by, "margin", "split"], key)), **summarize(g)})
    T = pl.DataFrame(rows, infer_schema_length=None).sort([*by, "margin", "split"])
    chosen = []
    for key, g in T.group_by(list(by)):
        dv = g.filter((pl.col("split") == "dev") & (pl.col("fills") >= min_fills))
        if not dv.height:
            continue
        m = dv.sort("pnl_per_fill_c", descending=True)["margin"][0]
        for r in g.filter(pl.col("margin") == m).iter_rows(named=True):
            chosen.append(r)
    C = pl.DataFrame(chosen, infer_schema_length=None).sort([*by, "split"]) if chosen else pl.DataFrame()
    return T, C


def chosen_margins(C: pl.DataFrame, by: tuple[str, ...]) -> dict[tuple, float]:
    return {tuple(r[b] for b in by): r["margin"] for r in C.filter(pl.col("split") == "dev").iter_rows(named=True)}


def segment(G: pl.DataFrame, margins: dict[tuple, float], by: tuple[str, ...], seg: str, split: str = "holdout") -> pl.DataFrame:
    """Results at the dev-chosen margin of each ``by`` group, split by ``seg``."""
    rows = []
    for key, m in margins.items():
        g = G.filter(pl.all_horizontal([pl.col(b) == v for b, v in zip(by, key)]) & (pl.col("margin") == m)
                     & (pl.col("split") == split))
        for sk, gg in g.group_by([seg]):
            rows.append({**dict(zip(by, key)), "margin": m, seg: sk[0], **summarize(gg)})
    return pl.DataFrame(rows, infer_schema_length=None).sort([*by, seg]) if rows else pl.DataFrame()


def robustness(G: pl.DataFrame, margins: dict[tuple, float], by: tuple[str, ...], split: str = "holdout") -> pl.DataFrame:
    """At the dev-chosen margin: PnL per fill without the 3 most profitable events, and the share
    of calendar weeks (by ``t``) with positive PnL, per ``by`` group."""
    from albert.research.fundamental import clustered_mean

    rows = []
    for key, m in margins.items():
        g = G.filter(pl.all_horizontal([pl.col(b) == v for b, v in zip(by, key)]) & (pl.col("margin") == m)
                     & (pl.col("split") == split) & pl.col("filled"))
        if not g.height:
            continue
        ev = g.group_by("event").agg(pl.col("pnl").sum().alias("p")).sort("p", descending=True)
        top = ev.head(3)["event"].to_list()
        x = g.filter(~pl.col("event").is_in(top))
        mu, tt, _ = clustered_mean(x["pnl"].to_numpy(), x["event"].to_numpy()) if x.height else (float("nan"), None, 0)
        wk = g.group_by(pl.col("t").dt.truncate("1w").alias("wk")).agg(pl.col("pnl").sum().alias("p"))
        rows.append({**dict(zip(by, key)), "margin": m, "fills": g.height, "events": ev.height,
                     "pnl_total_$": round(float(g["pnl"].sum()), 2), "top3_events_$": round(float(top and ev.head(3)["p"].sum()), 2),
                     "pnl_per_fill_ex_top3_c": round(mu * 100, 3), "t_ex_top3": None if tt is None else round(tt, 2),
                     "events_positive": int((ev["p"] > 0).sum()), "weeks": wk.height, "weeks_positive": int((wk["p"] > 0).sum())})
    return pl.DataFrame(rows, infer_schema_length=None).sort(list(by)) if rows else pl.DataFrame()


# ---- bands -------------------------------------------------------------------------------------
def freq_band(mu: pl.Expr) -> pl.Expr:
    """Transcript expected count of the word in this speech (Trump/Vance only)."""
    return (pl.when(mu.is_null()).then(pl.lit("n/a")).when(mu < 0.3).then(pl.lit("a mu<0.3"))
            .when(mu < 1).then(pl.lit("b 0.3-1")).when(mu < 3).then(pl.lit("c 1-3")).otherwise(pl.lit("d 3+")))


# ---- the study ---------------------------------------------------------------------------------
def load_trades(s) -> tuple[pl.DataFrame, set[str]]:
    """Cached prints (main fetch + retry of transient failures) and the tickers whose window was
    fetched without error (failures listed in ``fundamental_mentions_maker_failed.parquet``)."""
    frames, covered = [], set()
    for name in (NAME, f"{NAME}_retry"):
        p = s.data_dir / "cache" / f"fundamental_{name}_trades.parquet"
        w = p.with_name(p.stem + "_windows.parquet")
        if p.exists():
            frames.append(pl.read_parquet(p))
        if w.exists():
            covered |= set(pl.read_parquet(w)["ticker"])
    failed = s.data_dir / "cache" / f"fundamental_{NAME}_failed.parquet"
    if failed.exists():
        covered -= set(pl.read_parquet(failed)["ticker"])
    trades = pl.concat(frames, how="diagonal_relaxed").unique("trade_id") if frames else pl.DataFrame()
    return trades, covered


def build(s) -> tuple[pl.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Decision rows with base rates, transcript features, the inner-split model selection and
    the dev-fitted fair values."""
    C_ = s.data_dir / "cache" / M.NAME
    mk = pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet").filter(pl.col("category") == "Mentions").collect()
    lg = M.legs(mk)
    fees = mk.select("ticker", "fee_type", "fee_multiplier")
    starts = pl.read_parquet(C_ / "starts.parquet")
    candles = M.load_candles(s)
    Q, info = decision_rows(lg, starts, candles)
    Q = add_base_rates(Q, lg).join(fees, on="ticker", how="left")
    if not (C_ / "transcript_docs.parquet").exists():
        raise RuntimeError("no transcript source: the Factba.se data was removed (robots.txt); add a "
                           "permitted source as transcript_docs/transcript_word_counts.parquet")
    docs = pl.read_parquet(C_ / "transcript_docs.parquet")
    counts = pl.read_parquet(C_ / "transcript_word_counts.parquet")
    Q = add_transcripts(Q, docs, counts)
    tv = Q.filter(pl.col("p_tx2").is_not_null() & pl.col("p_tx1").is_not_null() & (pl.col("t") < HOLDOUT_FROM))
    best, sel = select_tx(tv)
    info["tx_variant_chosen_inner"] = best
    Q, fv_info = fair_values(Q, best)
    info.update(fv_info)
    Q = Q.with_columns(pl.when(pl.col("t") >= HOLDOUT_FROM).then(pl.lit("holdout")).otherwise(pl.lit("dev")).alias("split"),
                       M.price_band(pl.col("mid")).alias("band"), M.word_band(pl.col("n_sw_0")).alias("wband"),
                       M.series_group(pl.col("speaker")).alias("sgroup"), freq_band(pl.col("mu_tx2")).alias("fband"))
    return Q, info, sel


# (scope, fair value, end column, fill rule): the maker configurations reported
CONFIGS = (
    ("tv", "fv_tv", "end", "through"), ("tv", "fv_cal", "end", "through"), ("tv", "fv_bb", "end", "through"),
    ("tv", "fv_mid", "end", "through"), ("tv", "fv_tv", "end_cb", "through"), ("tv", "fv_tv", "end", "improve"),
    ("tv", "fv_cal", "end", "improve"),
    ("all", "fv_bb", "end", "through"), ("all", "fv_cal", "end", "through"), ("all", "fv_mid", "end", "through"),
    ("all", "fv_bb", "end_cb", "through"), ("all", "fv_bb", "end", "improve"), ("all", "fv_cal", "end", "improve"),
    ("other", "fv_bb", "end", "through"), ("other", "fv_cal", "end", "through"),
)
SEGMENT_CONFIGS = ("tv/fv_tv/end/through", "tv/fv_cal/end/through", "all/fv_bb/end/through", "all/fv_cal/end/through",
                   "tv/fv_tv/end/improve", "all/fv_bb/end/improve")


def run(s, *, harness_check: bool = True) -> dict[str, Any]:
    from albert.research.fundamental import evaluate, evaluate_maker

    Q, info, sel = build(s)
    trades, covered = load_trades(s)
    Q = Q.filter(pl.col("ticker").is_in(covered))      # rows whose trade window was fetched
    info["trades"] = {"tickers_covered": len(covered), "prints": trades.height}
    info["rows_with_trades"] = (Q.group_by("when", "split").agg(pl.len().alias("rows"), pl.col("event").n_unique().alias("events"),
                                                               pl.col("fv_tv").is_not_null().sum().alias("tv_rows"))
                                .sort("when", "split").to_dicts())
    kt = kill_times(Q, trades)
    Q = Q.join(kt, on=["event", "when"], how="left").with_columns(
        pl.min_horizontal("end", pl.col("kill_ts").fill_null(pl.col("end"))).alias("end_cb"))
    info["circuit_breaker"] = (Q.unique(["event", "when"]).group_by("when", "split")
                               .agg(pl.len().alias("event_leads"), (pl.col("kill_ts") < pl.col("end")).sum().alias("tripped_before_end"))
                               .sort("when", "split").to_dicts())
    tables: dict[str, pl.DataFrame] = {
        "Inner dev split (fit < Sep 15, score Sep 15–30): transcript variants, blend log loss": pl.DataFrame(sel, infer_schema_length=None)}
    scope_rows = {"tv": Q.filter(pl.col("fv_tv").is_not_null()), "all": Q, "other": Q.filter(pl.col("fv_tv").is_null())}
    grids: dict[str, pl.DataFrame] = {}
    chosen = []
    for scope, fv, endc, rule in CONFIGS:
        key = f"{scope}/{fv}/{endc}/{rule}"
        if not scope_rows[scope].height:
            continue
        G = maker_grid(scope_rows[scope], trades, fv, end_col=endc, rule=rule)
        grids[key] = G
        T, C = maker_table(G)
        if C.height:
            chosen.append(C.with_columns(pl.lit(scope).alias("scope"), pl.lit(fv).alias("fair"), pl.lit(endc).alias("end"),
                                         pl.lit(rule).alias("rule")))
        if key in ("tv/fv_tv/end/through", "all/fv_bb/end/through"):
            tables[f"maker {key}: every margin (both sides)"] = T.filter(pl.col("variant") == "both")
    CH = pl.concat(chosen, how="diagonal_relaxed").select(
        "scope", "fair", "end", "rule", "when", "variant", "margin", "split", "orders", "fills", "fill_rate", "events",
        "pnl_per_fill_c", "t", "pnl_per_order_c", "pnl_total_$", "yes_share")
    tables["maker: dev-chosen margin per (scope, fair value, end, fill rule, lead, variant): dev vs holdout"] = CH
    for key in SEGMENT_CONFIGS:
        G = grids.get(key)
        if G is None or not G.height:
            continue
        _, C = maker_table(G)
        mm = {k_: v for k_, v in chosen_margins(C, ("when", "variant")).items() if k_[1] in ("both", "fav")}
        Gs = G.filter(pl.col("variant").is_in(["both", "fav"]))
        segs = ("sgroup", "series", "band", "wband", "fband", "ms_pre", "side") if key.endswith("through") else ("band", "side")
        for seg in segs:
            if seg == "fband" and not key.startswith("tv"):
                continue
            tables[f"maker {key} holdout by {seg} (dev-chosen margin)"] = segment(Gs, mm, ("when", "variant"), seg)
            if seg in ("band", "sgroup", "series", "fband"):
                tables[f"maker {key} DEV by {seg} (dev-chosen margin)"] = segment(Gs, mm, ("when", "variant"), seg, split="dev")
        tables[f"maker {key} holdout robustness (dev-chosen margin)"] = robustness(Gs, mm, ("when", "variant"))
        both = {k_: v for k_, v in mm.items() if k_[1] == "both"}
        bs = Gs.filter(pl.col("variant") == "both").with_columns((pl.col("band") + " / " + pl.col("side")).alias("band_side"))
        for split in ("dev", "holdout"):
            tables[f"maker {key} {split} by price band x side (both sides, dev-chosen margin)"] = segment(
                bs, both, ("when", "variant"), "band_side", split=split)
    # taker with the same fair values (harness)
    tk = []
    for scope, fv in (("tv", "fv_tv"), ("tv", "fv_cal"), ("all", "fv_bb"), ("all", "fv_cal")):
        for lead in LEADS:
            d = scope_rows[scope].filter(pl.col("when") == lead).with_columns(pl.col(fv).alias("p"))
            if not d.height:
                continue
            r = evaluate(d, HOLDOUT_FROM)
            th = r["theta_chosen_on_dev"]
            for x in r["taker"]:
                if x["theta"] == th:
                    tk.append({"scope": scope, "fair": fv, "when": lead, **x})
    tables["taker with the same fair value (harness evaluate; θ chosen on dev)"] = pl.DataFrame(tk, infer_schema_length=None)
    # harness cross-check: evaluate_maker on the headline spec must equal the fast implementation
    if harness_check:
        chk = []
        for lead in LEADS:
            d = scope_rows["tv"].filter(pl.col("when") == lead).with_columns(pl.col("fv_tv").alias("p"))
            r = evaluate_maker(d, trades, HOLDOUT_FROM)
            for x in r["maker"]:
                chk.append({"when": lead, **x, "margin_chosen_on_dev": r["margin_chosen_on_dev"]})
        tables["harness evaluate_maker, tv/fv_tv (both sides, trade-through): cross-check of the fast fills"] = pl.DataFrame(
            chk, infer_schema_length=None)
    # holdout predictions for the coordinator
    out = s.data_dir / "cache" / f"fundamental_{NAME}_holdout_preds.parquet"
    ho = Q.filter(pl.col("split") == "holdout").with_columns(
        pl.coalesce("fv_tv", "fv_bb").alias("p_fair"),
        pl.when(pl.col("fv_tv").is_not_null()).then(pl.lit("fv_tv")).otherwise(pl.lit("fv_bb")).alias("fair"))
    ho.select("ticker", "t", "p_fair", "mid", "fair", "when", "event", "series", "bid", "ask", "y").write_parquet(out)
    info["holdout_preds"] = str(out)
    return {"Q": Q, "info": info, "tables": tables, "grids": grids}


NOTES = __doc__


def write_report(s, out: dict[str, Any], verdict: str = "") -> Any:
    from albert.reports import Report

    rep = Report(s.reports_dir, f"research-fundamental-{NAME}")
    if verdict:
        rep.text("Verdict", verdict)
    rep.text("Design", NOTES)
    rep.json("Run info", out["info"])
    for i, (title, tab) in enumerate(out["tables"].items()):
        if tab.height:
            rep.table(title, tab, max_rows=80, save_as=f"t{i:02d}")
    return rep.save()
