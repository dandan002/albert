"""Maker quoting around a model + price blend in Kalshi college-football ladders (round 2 of
``research/ncaaf.py``).

Round 1 found that the frozen ratings model (ridge margin + robust offense/defense totals,
Student-t law) adds information to FCS spread ladders on both periods, but the information sits
inside 5-14c spreads, so a taker cannot capture it. Here we ask whether *resting* bids around a
fair value can.

Fair value
----------
``logit fair = a + b·logit(mid) + c·logit(model)``, fitted on Aug-Sep 2026 rows only (decision
times 24 h and 6 h before ESPN's scheduled kickoff pooled, clustered by game), separately for each
contract kind (moneyline, spread, total, team total) and tier group (FCS = any FCS team in the
game; FBS = both teams FBS). The October holdout never enters the fit. A mid-only control
(``logit fair = a + b·logit(mid)``, same groups) shows what quoting around the price alone earns.

Maker test
----------
From the decision time ``t`` to the scheduled kickoff (in-game fills excluded), rest a YES bid at
``floor(fair − m)`` and a NO bid at ``floor(1 − fair − m)``, post-only (capped one tick inside the
opposite quote at ``t``), one contract each, held to settlement with the series maker fee
(``0.0175·P(1−P)`` on spread / total / moneyline; none on team totals). Fill rules:

* ``strict`` (the harness rule): a later trade prints strictly through our price;
* ``improve``: as strict, plus a trade AT our price when our bid improved the quote at ``t``
  (a new best price is first in the queue at that level, so the first seller there hits us).
  The quote at ``t`` comes from an hourly candle up to 2 h old, so this rule is optimistic.

Variants (all chosen on dev): margin ``m``; rest only where ``|fair − mid| ≥ x``; both sides vs
only the side the model favours; and fill rate / PnL by the size of ``|fair − mid|``.

Forward test (no orders)
------------------------
:func:`snapshot_open` records, for open Kalshi FCS/FBS legs kicking off in the next 7 days, the
public quote (``/markets`` listing: best bid / ask and their sizes), the model probability, the
frozen fair value and the hypothetical maker prices, to
``data/cache/ncaaf_maker/forward/<UTC stamp>.parquet``. :func:`score_forward` scores the rows
whose markets have settled: taker at the snapshot quote, and the hypothetical maker filled by
trades printed after the snapshot and before kickoff.

    nice -n 10 python -m albert.research.ncaaf_maker snapshot   # record open legs now
    nice -n 10 python -m albert.research.ncaaf_maker score      # score settled snapshot rows

``study`` reruns the dev/holdout study and report; ``freeze`` rewrites only the forward config
(dev rows only). Fill data: Kalshi's public trade prints, read as hourly trade-price high/low
from the batch candlestick endpoint (one request per game instead of one per leg; the per-leg
trades endpoint was rate-limited to < 1 request/s on the shared IP) — see :func:`hl_pseudo_trades`.
"""

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.research import ncaaf as nc
from albert.research.fundamental import clustered_mean, fit_logistic, logit

STUDY = "ncaaf_maker"
LEADS_H = (24, 6)
DEV_END = nc.DEV_END
KINDS = nc.KINDS
TIER_GROUP = {"FCS": "FCS", "G5": "FBS", "P4": "FBS"}
MARGINS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08)
EDGE_X = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08)
MIN_FILLS = 30


def cache_dir(s) -> Path:
    d = s.data_dir / "cache" / STUDY
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- model frames -------------------------------------------------------------------------------
def chosen_run(inp: dict[str, Any], leads: tuple[int, ...] = LEADS_H) -> dict[str, Any]:
    """Round 1's frozen model (``nc.CHOSEN``) answering every Kalshi-listed game at ``leads``
    hours before kickoff (plus kickoff − 2 h for every completed game: the residual law)."""
    margin = nc.model_run(inp, nc.MODELS[nc.CHOSEN["margin"]], leads=leads)
    totals = nc.model_run(inp, nc.MODELS[nc.CHOSEN["totals"]], leads=leads)
    return nc.combine_runs(margin, totals)


def contract_frames(s, *, leads: tuple[int, ...] = LEADS_H, refresh: bool = False) -> pl.DataFrame:
    """Every settled leg at each lead with model ``p``, the candle quote at ``t``, ``y``, tier,
    series fee columns and ``end`` = scheduled kickoff. Cached in data/cache/ncaaf_maker/."""
    path = cache_dir(s) / "contracts.parquet"
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    inp = nc.load_inputs(s)
    run = chosen_run(inp, leads)
    dists = {k: f[nc.CHOSEN["law"]] for k, f in nc.residual_laws(run).items()}
    tier = nc.game_tier(inp["games"], inp["tiers"])
    df = pl.concat([nc.contract_frame(inp, run, dists, lead, tier) for lead in leads], how="diagonal_relaxed")
    fees = inp["markets"].select("ticker", "fee_type", "fee_multiplier").unique("ticker")
    df = (df.join(fees, on="ticker", how="left")
          .with_columns(pl.col("kickoff").alias("end"), pl.col("tier").replace_strict(TIER_GROUP, default="FBS").alias("tgroup")))
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)
    return df


# ---- trades -------------------------------------------------------------------------------------
def game_sample(game_ids: pl.Series, keep_every: int) -> pl.Series:
    """Volume-blind deterministic game sample: keep a game when its numeric ESPN id is divisible
    by ``keep_every`` (1 keeps every game)."""
    return (game_ids.cast(pl.Int64) % keep_every) == 0


def trade_windows(df: pl.DataFrame, max_lead: int = max(LEADS_H)) -> pl.DataFrame:
    """One window per leg: [kickoff − ``max_lead`` h, kickoff] covers every decision time."""
    return (df.group_by("ticker").agg(pl.col("kickoff").first())
            .select("ticker", (pl.col("kickoff") - pl.duration(hours=max_lead)).alias("t_from"),
                    pl.col("kickoff").alias("t_to")))


def fetch_study_trades(s, df: pl.DataFrame, *, chunk: int = 1000, workers: int = 2) -> pl.DataFrame:
    """Every public trade of each leg in ``df`` from 24 h before kickoff to kickoff, cached in
    ``data/cache/fundamental_ncaaf_maker_trades.parquet`` (saved after each chunk of legs)."""
    from albert.research.fundamental import fetch_trades

    w = trade_windows(df).sort("t_to", "ticker")
    out = pl.DataFrame()
    for i in range(0, w.height, chunk):
        out = fetch_trades(s, w.slice(i, chunk), STUDY, workers=workers)
        print(f"trade windows {min(i + chunk, w.height)}/{w.height}", flush=True)
    return out


# ---- fair value ---------------------------------------------------------------------------------
def _design(part: pl.DataFrame, with_model: bool) -> np.ndarray:
    cols = [np.ones(part.height), logit(part["mid"].to_numpy())]
    if with_model:
        cols.append(logit(part["p"].to_numpy()))
    return np.column_stack(cols)


def fit_blends(df: pl.DataFrame, *, min_games: int = 20) -> dict[str, dict[str, Any]]:
    """Per ``kind|tgroup``: the dev-fitted blend ``logit fair = a + b·logit(mid) + c·logit(p)``
    and the mid-only control ``a + b·logit(mid)`` (both leads pooled, t clustered by game). A
    group with fewer than ``min_games`` dev games uses its tier group's all-kinds fit."""
    dev = df.filter(pl.col("t") < DEV_END)
    out: dict[str, dict[str, Any]] = {}

    def fit(part: pl.DataFrame) -> dict[str, Any]:
        y, g = part["y"].to_numpy().astype(float), part["game_id"].to_numpy()
        bb, tb = fit_logistic(_design(part, True), y, g)
        bm, _ = fit_logistic(_design(part, False), y, g)
        return {"blend": [float(v) for v in bb], "t_blend": [round(float(v), 2) for v in tb],
                "mid_only": [float(v) for v in bm], "n": part.height, "games": part["game_id"].n_unique()}

    for tg in ("FCS", "FBS"):
        if dev.filter(pl.col("tgroup") == tg)["game_id"].n_unique() < min_games:
            continue
        pooled = fit(dev.filter(pl.col("tgroup") == tg))
        for kind in KINDS:
            part = dev.filter((pl.col("tgroup") == tg) & (pl.col("kind") == kind))
            key = f"{kind}|{tg}"
            out[key] = ({**fit(part), "source": key} if part["game_id"].n_unique() >= min_games
                        else {**pooled, "source": f"all|{tg} (only {part['game_id'].n_unique()} dev games)"})
    return out


def apply_fair(df: pl.DataFrame, blends: dict[str, dict[str, Any]]) -> pl.DataFrame:
    """Adds ``p_model`` (round-1 model), ``fair`` (blend) and ``fair_mid`` (mid-only control)."""
    parts = []
    for key, b in blends.items():
        kind, tg = key.split("|")
        part = df.filter((pl.col("kind") == kind) & (pl.col("tgroup") == tg))
        if not part.height:
            continue
        fb = 1 / (1 + np.exp(-_design(part, True) @ np.array(b["blend"])))
        fm = 1 / (1 + np.exp(-_design(part, False) @ np.array(b["mid_only"])))
        parts.append(part.with_columns(pl.col("p").alias("p_model"), pl.Series("fair", fb), pl.Series("fair_mid", fm)))
    return pl.concat(parts, how="diagonal_relaxed") if parts else df.clear()


# ---- maker fills (vectorised; same rules as fundamental.maker_fills, plus the "improve" rule) ----
def maker_orders(pred: pl.DataFrame, margin: float) -> pl.DataFrame:
    """One YES bid and one NO bid per row around ``p`` (see ``fundamental.maker_fills``)."""
    base = pred.with_row_index("row")
    yes = base.with_columns(pl.lit("yes").alias("side"), pl.min_horizontal(
        ((pl.col("p") - margin) * 100 + 1e-9).floor() / 100, (pl.col("ask") - 0.01).round(2)).alias("price"))
    no = base.with_columns(pl.lit("no").alias("side"), pl.min_horizontal(
        ((1 - pl.col("p") - margin) * 100 + 1e-9).floor() / 100, (1 - pl.col("bid") - 0.01).round(2)).alias("price"))
    o = pl.concat([yes, no]).filter(pl.col("price") >= 0.01 - 1e-9)
    # Our bid improves the quote at t: a new best price is first in its queue.
    improves = pl.when(pl.col("side") == "yes").then(pl.col("price") > pl.col("bid") + 1e-9).otherwise(
        pl.col("price") > 1 - pl.col("ask") + 1e-9)
    return o.with_columns(improves.alias("improves"))


def _first_fill(orders: pl.DataFrame, trades: pl.DataFrame, rule: str) -> pl.DataFrame:
    tr = trades.select("ticker", pl.col("created_time").alias("ts"), "yes_price", "taker_side")
    j = orders.select("row", "side", "ticker", "t", "end", "price", "improves").join(tr, on="ticker")
    j = j.filter((pl.col("ts") > pl.col("t")) & (pl.col("ts") <= pl.col("end")))
    through_yes = (pl.col("taker_side") == "no") & (pl.col("yes_price") < pl.col("price") - 1e-9)
    through_no = (pl.col("taker_side") == "yes") & (pl.col("yes_price") > 1 - pl.col("price") + 1e-9)
    if rule == "improve":
        through_yes = through_yes | (pl.col("improves") & (pl.col("taker_side") == "no")
                                     & ((pl.col("yes_price") - pl.col("price")).abs() < 1e-9))
        through_no = through_no | (pl.col("improves") & (pl.col("taker_side") == "yes")
                                   & ((pl.col("yes_price") - (1 - pl.col("price"))).abs() < 1e-9))
    hit = pl.when(pl.col("side") == "yes").then(through_yes).otherwise(through_no)
    return j.filter(hit).group_by("row", "side").agg(pl.col("ts").min().alias("fill_ts"))


def maker_fee_expr(price: pl.Expr) -> pl.Expr:
    mult = pl.col("fee_multiplier").fill_null(1.0)
    fee = ((0.0175 * mult * price * (1 - price) * 1e4 - 1e-9).ceil() / 1e4)
    return pl.when(pl.col("fee_type").fill_null("").str.contains("maker_fee")).then(fee).otherwise(0.0)


def fills(pred: pl.DataFrame, trades: pl.DataFrame, margin: float, rule: str = "strict",
          chunk: int = 3000) -> pl.DataFrame:
    """Every order (row x side) with ``filled``, ``fill_ts`` and settlement ``pnl`` (maker fee
    by series). ``trades`` are public prints (exact) or :func:`hl_pseudo_trades` (hourly).
    ``pred`` needs ticker, event, t, end, p, bid, ask, y, fee_type, fee_multiplier; extra columns
    are kept. Processed in chunks of tickers to bound the join's memory."""
    o = maker_orders(pred, margin)
    tickers = o["ticker"].unique().sort()
    hits = []
    for i in range(0, tickers.len(), chunk):
        tk = tickers.slice(i, chunk).implode()
        hits.append(_first_fill(o.filter(pl.col("ticker").is_in(tk)), trades.filter(pl.col("ticker").is_in(tk)), rule))
    h = pl.concat(hits) if hits else pl.DataFrame(schema={"row": pl.UInt32, "side": pl.String,
                                                          "fill_ts": pl.Datetime("us", "UTC")})
    o = o.join(h, on=["row", "side"], how="left").with_columns(pl.col("fill_ts").is_not_null().alias("filled"))
    won = pl.when(pl.col("side") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y")).cast(pl.Float64)
    pnl = won - pl.col("price") - maker_fee_expr(pl.col("price"))
    return o.with_columns(pl.when(pl.col("filled")).then(pnl).otherwise(0.0).alias("pnl"))


def hl_pseudo_trades(hl: pl.DataFrame) -> pl.DataFrame:
    """Hourly trade low/high as two prints stamped 1 µs after the hour STARTS: the low as a
    YES-seller, the high as a YES-buyer. With the harness rule (a print strictly after t and at
    or before ``end``) an hour counts only if it starts at or after t, and the candles end at or
    before kickoff, so no print before t or after kickoff is used. Any print below a resting YES
    bid means the bid was hit (a seller sells to the best bid first; an ask below it would have
    crossed it), so the taker side of the real prints is not needed."""
    c = hl.filter(pl.col("price_low").is_not_null())
    ts = (pl.col("end_ts") - pl.duration(hours=1) + pl.duration(microseconds=1)).alias("created_time")
    return pl.concat([c.select("ticker", ts, pl.col("price_low").alias("yes_price"), pl.lit("no").alias("taker_side")),
                      c.select("ticker", ts, pl.col("price_high").alias("yes_price"), pl.lit("yes").alias("taker_side"))])


def favoured(o: pl.DataFrame, fair: str = "p") -> pl.Expr:
    """The side the fair value favours against the mid: YES if fair > mid, NO if fair < mid."""
    return (pl.when(pl.col(fair) > pl.col("mid")).then(pl.col("side") == "yes")
            .when(pl.col(fair) < pl.col("mid")).then(pl.col("side") == "no").otherwise(False))


def summarise(o: pl.DataFrame, **keys: Any) -> dict[str, Any]:
    """Orders, fills, fill rate, PnL per fill (t clustered by game), PnL per order (cents)."""
    f = o.filter("filled")
    row = {**keys, "orders": o.height, "fills": f.height,
           "fill_rate": round(f.height / o.height, 3) if o.height else None}
    if f.height:
        mu, t, k = clustered_mean(f["pnl"].to_numpy(), f["event"].to_numpy())
        row.update({"games": k, "pnl_per_fill_c": round(mu * 100, 2), "t": None if t is None else round(t, 2),
                    "pnl_per_order_c": round(float(f["pnl"].sum()) / o.height * 100, 3),
                    "total_usd": round(float(f["pnl"].sum()), 2), "yes_share": round(float((f["side"] == "yes").mean()), 2),
                    "avg_price": round(float(f["price"].mean()), 3)})
    return row


# ---- study --------------------------------------------------------------------------------------
TIERS = ("FCS", "G5", "P4")


def with_fair(df: pl.DataFrame, source: str) -> pl.DataFrame:
    """The frame with ``p`` set to the chosen fair value: ``blend``, ``mid_only`` or ``model``."""
    col = {"blend": "fair", "mid_only": "fair_mid", "model": "p_model"}[source]
    return df.with_columns(pl.col(col).alias("p"))


def variant_grid(fv: pl.DataFrame, trades: pl.DataFrame, *, sources: tuple[str, ...] = ("blend", "mid_only"),
                 rules: tuple[str, ...] = ("strict", "improve")) -> pl.DataFrame:
    """Per fair source x rule x margin x edge filter x sides x tier x lead x split: summaries."""
    rows = []
    for source in sources:
        pred = with_fair(fv, source)
        for rule in rules:
            for m in MARGINS:
                o = fills(pred, trades, m, rule).with_columns((pl.col("p") - pl.col("mid")).abs().alias("edge"),
                                                             (pl.col("t") >= DEV_END).alias("holdout"))
                o = o.with_columns(favoured(o).alias("fav"))
                for (tier, lead, hold), part in o.group_by("tier", "lead", "holdout"):
                    for x in EDGE_X:
                        px = part.filter(pl.col("edge") >= x - 1e-9)
                        for sides, pp in (("both", px), ("favoured", px.filter("fav"))):
                            rows.append(summarise(pp, source=source, rule=rule, margin=m, x=x, sides=sides,
                                                  tier=tier, lead=lead, split="holdout" if hold else "dev"))
    return pl.DataFrame(rows, infer_schema_length=None)


def choose_on_dev(grid: pl.DataFrame, *, min_fills: int = MIN_FILLS) -> pl.DataFrame:
    """Per source x rule x tier x lead: the (margin, x, sides) with the highest total dev PnL
    among cells with at least ``min_fills`` dev fills; returns that config's dev and holdout rows."""
    keys = ["source", "rule", "tier", "lead"]
    cfg = ["margin", "x", "sides"]
    dev = grid.filter((pl.col("split") == "dev") & (pl.col("fills") >= min_fills))
    best = dev.sort("total_usd", descending=True).group_by(keys).first().select(*keys, *cfg)
    return (grid.join(best, on=[*keys, *cfg]).sort(*keys, "split"))


def edge_bands(fv: pl.DataFrame, trades: pl.DataFrame, source: str = "blend", margin: float = 0.0,
               rule: str = "strict") -> pl.DataFrame:
    """Fill rate and PnL per fill by the size of |fair − mid|, favoured vs other side."""
    o = fills(with_fair(fv, source), trades, margin, rule).with_columns(
        (pl.col("p") - pl.col("mid")).abs().alias("edge"), (pl.col("t") >= DEV_END).alias("holdout"))
    o = o.with_columns(favoured(o).alias("fav"), pl.col("edge").cut([0.01, 0.02, 0.03, 0.05, 0.08], left_closed=True)
                       .cast(pl.String).alias("band"))
    rows = []
    for (tier, hold, band, fav), part in o.group_by("tier", "holdout", "band", "fav"):
        rows.append(summarise(part, tier=tier, split="holdout" if hold else "dev", band=band, favoured=fav))
    return pl.DataFrame(rows, infer_schema_length=None).sort("tier", "split", "band", "favoured")


def per_week(o: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Fills and PnL per ESPN week (consistency check)."""
    w = o.join(games.select("game_id", "week"), on="game_id", how="left")
    rows = []
    for (wk, hold), part in w.group_by("week", "holdout"):
        rows.append(summarise(part, week=wk, split="holdout" if hold else "dev"))
    return pl.DataFrame(rows, infer_schema_length=None).sort("week")


def taker_table(fv: pl.DataFrame, source: str) -> pl.DataFrame:
    """Harness ``evaluate`` with p = fair value, per tier x lead: scores and taker at dev θ."""
    from albert.research.fundamental import evaluate

    rows = []
    for tier in TIERS:
        for lead in LEADS_H:
            part = with_fair(fv, source).filter((pl.col("tier") == tier) & (pl.col("lead") == lead))
            if part.filter(pl.col("t") < DEV_END).height < 50:
                continue
            r = evaluate(part, DEV_END)
            sc = {x["split"]: x for x in r["scores"]}
            th = r["theta_chosen_on_dev"]
            tk = {(x["theta"], x["split"]): x for x in r["taker"]}
            for split in ("dev", "holdout"):
                t = tk.get((th, split), {})
                rows.append({"source": source, "tier": tier, "lead": lead, "split": split,
                             "brier_fair": sc.get(split, {}).get("brier_model"), "brier_mid": sc.get(split, {}).get("brier_mid"),
                             "theta": th, "trades": t.get("trades"), "games": t.get("events"),
                             "pnl_c": t.get("pnl_c"), "t": t.get("t")})
    return pl.DataFrame(rows, infer_schema_length=None)


def harness_maker(fv: pl.DataFrame, trades: pl.DataFrame, source: str = "blend") -> pl.DataFrame:
    """``fundamental.evaluate_maker`` as shipped (both sides, margin chosen on dev, strict rule),
    per tier x lead."""
    from albert.research.fundamental import evaluate_maker

    rows = []
    pred = with_fair(fv, source)
    for tier in TIERS:
        for lead in LEADS_H:
            part = pred.filter((pl.col("tier") == tier) & (pl.col("lead") == lead))
            tk = trades.filter(pl.col("ticker").is_in(part["ticker"].unique().implode()))
            r = evaluate_maker(part, tk, DEV_END)
            for x in r["maker"]:
                rows.append({"source": source, "tier": tier, "lead": lead, "chosen": x["margin"] == r["margin_chosen_on_dev"], **x})
    return pl.DataFrame(rows, infer_schema_length=None)


def study_frame(s) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Contracts with fair values, restricted to games whose hourly candles were all fetched,
    plus the pseudo-trades and the dev-fitted blends."""
    df = contract_frames(s)
    blends = fit_blends(df)
    fv = apply_fair(df, blends)
    done = pl.read_parquet(cache_dir(s) / "candles_hl_jobs.parquet")["job"]
    hl = pl.read_parquet(cache_dir(s) / "candles_hl.parquet")
    fv = fv.filter(pl.col("game_id").is_in(done.implode()) & pl.col("ticker").is_in(hl["ticker"].unique().implode()))
    return fv, hl_pseudo_trades(hl), blends


def spread_bucket(spread: pl.Expr) -> pl.Expr:
    """Quoted spread at t in whole cents: ≤ 5c, 6-10c, 11-15c, > 15c."""
    c = (spread * 100).round(0)
    return (pl.when(c <= 5).then(pl.lit("a ≤5c")).when(c <= 10).then(pl.lit("b 6-10c"))
            .when(c <= 15).then(pl.lit("c 11-15c")).otherwise(pl.lit("d >15c")))


def blend_by_spread(fv: pl.DataFrame, *, min_games: int = 15) -> pl.DataFrame:
    """Does the model add information where the book has a real price? Per tier x kind (and all
    kinds) x quoted-spread bucket: ``logit P = a + b·logit(mid) + c·logit(model)`` fitted on dev
    and refitted on the October holdout (both leads pooled, t clustered by game)."""
    df = fv.with_columns(spread_bucket(pl.col("ask") - pl.col("bid")).alias("bucket"), (pl.col("t") >= DEV_END).alias("holdout"))
    rows = []
    for tier in TIERS:
        for kind in (*KINDS, "all"):
            for bucket in ("a ≤5c", "b 6-10c", "c 11-15c", "d >15c", "all"):
                part = df.filter(pl.col("tier") == tier)
                if kind != "all":
                    part = part.filter(pl.col("kind") == kind)
                if bucket != "all":
                    part = part.filter(pl.col("bucket") == bucket)
                row: dict[str, Any] = {"tier": tier, "kind": kind, "bucket": bucket}
                for split, q in (("dev", part.filter(~pl.col("holdout"))), ("hold", part.filter(pl.col("holdout")))):
                    row[f"n_{split}"], row[f"games_{split}"] = q.height, q["game_id"].n_unique()
                    if q["game_id"].n_unique() < min_games or q["y"].n_unique() < 2:
                        continue
                    X = np.column_stack([np.ones(q.height), logit(q["mid"].to_numpy()), logit(q["p_model"].to_numpy())])
                    beta, t = fit_logistic(X, q["y"].to_numpy().astype(float), q["game_id"].to_numpy())
                    row.update({f"b_{split}": round(float(beta[1]), 3), f"c_{split}": round(float(beta[2]), 3),
                                f"t_c_{split}": round(float(t[2]), 2),
                                f"brier_mid_{split}": round(float(((q["mid"] - q["y"]) ** 2).mean()), 4),
                                f"brier_model_{split}": round(float(((q["p_model"] - q["y"]) ** 2).mean()), 4)})
                if row["n_dev"] or row["n_hold"]:
                    rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def maker_by_spread(fv: pl.DataFrame, trades: pl.DataFrame, configs: dict[str, dict[str, Any]]) -> pl.DataFrame:
    """Maker results per quoted-spread bucket for named configs (source, margin, x, sides, rule)."""
    rows = []
    for name, c in configs.items():
        o = fills(with_fair(fv, c["source"]), trades, c["margin"], c["rule"]).with_columns(
            (pl.col("p") - pl.col("mid")).abs().alias("edge"), (pl.col("t") >= DEV_END).alias("holdout"),
            spread_bucket(pl.col("ask") - pl.col("bid")).alias("bucket"))
        o = o.with_columns(favoured(o).alias("fav"))
        o = o.filter((pl.col("edge") >= c["x"] - 1e-9) & (pl.lit(c["sides"] == "both") | pl.col("fav")))
        for (tier, lead, hold, bucket), part in o.group_by("tier", "lead", "holdout", "bucket"):
            rows.append(summarise(part, config=name, tier=tier, lead=lead, bucket=bucket, split="holdout" if hold else "dev"))
    return pl.DataFrame(rows, infer_schema_length=None).sort("config", "tier", "lead", "bucket", "split")


def blend_table(blends: dict[str, dict[str, Any]]) -> pl.DataFrame:
    rows = []
    for key, b in blends.items():
        kind, tg = key.split("|")
        rows.append({"kind": kind, "tgroup": tg, "source": b["source"], "n_dev": b["n"], "games_dev": b["games"],
                     "a": round(b["blend"][0], 3), "b_mid": round(b["blend"][1], 3), "c_model": round(b["blend"][2], 3),
                     "t_b": b["t_blend"][1], "t_c": b["t_blend"][2],
                     "mid_only_a": round(b["mid_only"][0], 3), "mid_only_b": round(b["mid_only"][1], 3)})
    return pl.DataFrame(rows)


def forward_config(fv: pl.DataFrame, trades: pl.DataFrame) -> tuple[dict[str, Any], float]:
    """The maker config (blend, strict, chosen on FCS dev rows at 24 h) and the taker θ (FCS,
    24 h, dev) frozen for the forward test. Uses dev rows only."""
    from albert.research.fundamental import evaluate

    dev = fv.filter((pl.col("tier") == "FCS") & (pl.col("lead") == 24) & (pl.col("t") < DEV_END))
    pick = choose_on_dev(variant_grid(dev, trades, sources=("blend",), rules=("strict",))).filter(pl.col("split") == "dev")
    cfg = ({"source": "blend", "margin": float(pick["margin"][0]), "x": float(pick["x"][0]), "sides": pick["sides"][0],
            "tiers": ["FCS"], "chosen_on": "FCS, 24 h, strict fills, Aug-Sep dev, max total dev PnL"} if pick.height else
           {"source": "blend", "margin": 0.0, "x": 0.0, "sides": "both", "tiers": ["FCS"], "chosen_on": "default (no dev pick)"})
    theta = evaluate(with_fair(dev, "blend"), DEV_END)["theta_chosen_on_dev"]
    return cfg, float(theta or 0.0)


def residual_dists(s) -> dict[str, nc.Dist]:
    """Round 1's residual laws (fitted on dev games), recomputed from the cached inputs."""
    return {k: f[nc.CHOSEN["law"]] for k, f in nc.residual_laws(chosen_run(nc.load_inputs(s))).items()}


def run_study(s, *, out_preds: bool = True) -> Path:
    """Dev-chosen maker variants on Aug-Sep, October holdout scored once; writes the report,
    the frozen forward config and the holdout prediction rows."""
    from albert.reports import Report

    fv, trades, blends = study_frame(s)
    games = pl.read_parquet(s.data_dir / "cache" / "fundamental_ncaaf_games.parquet").select("game_id", "week")
    fv = fv.join(games, on="game_id", how="left")
    rep = Report(s.reports_dir, f"research-fundamental-{STUDY}")
    cover = (fv.group_by("tier", "lead", pl.col("t") >= DEV_END).agg(
        pl.len().alias("rows"), pl.col("game_id").n_unique().alias("games"), pl.col("week").n_unique().alias("weeks"),
        (pl.col("ask") - pl.col("bid")).median().alias("median_spread"), (pl.col("fair") - pl.col("mid")).abs().mean().alias("mean_abs_fair_minus_mid"))
        .rename({"t": "holdout"}).sort("tier", "lead", "holdout"))
    rep.json("Run", {
        "fair_value": "logit fair = a + b·logit(mid) + c·logit(model), per kind × tier group (FCS: any FCS team; FBS: both FBS), "
                      "fitted on Aug-Sep 2026 rows (24 h and 6 h pooled, t clustered by game); mid = candle quote at t",
        "model": f"round-1 frozen ratings: {nc.CHOSEN} (ncaaf.MODELS)",
        "decision_times": "ESPN scheduled kickoff − 24 h and − 6 h; orders rest from t to kickoff (in-game excluded)",
        "fills": "hourly batch candles (trade-price low/high) over [kickoff − 25 h, kickoff]; an hour counts only if it starts at or "
                 "after t and ends by kickoff; strict = a print strictly through our price (any taker side); improve = also a print AT "
                 "our price when our bid improved the candle quote at t",
        "fees": "maker 0.0175·P(1−P) rounded up to $0.0001 on KXNCAAFGAME/SPREAD/TOTAL (quadratic_with_maker_fees); none on "
                "KXNCAAFTEAMTOTAL; taker 0.07·P(1−P)",
        "selection": f"per source × rule × tier × lead: (margin, x, sides) with the highest total dev PnL among cells with ≥ {MIN_FILLS} dev fills",
        "excluded": "week-1 legs that settled before Kalshi's 2026-09-06 historical cutoff (batch candles are live-only); games whose "
                    "candle requests did not all succeed"})
    rep.table("Coverage (rows = leg × decision time)", cover)
    rep.table("Fair value: dev-fitted blend coefficients", blend_table(blends), save_as="blends")
    taker = pl.concat([taker_table(fv, src) for src in ("blend", "model")], how="diagonal_relaxed")
    rep.table("Taker with the fair value as p (θ chosen on dev; cents per contract, t by game)", taker, save_as="taker", max_rows=60)
    bs = blend_by_spread(fv)
    rep.table("Model information by quoted spread at t: logit P = a + b·logit(mid) + c·logit(model), dev fit and October refit",
              bs, save_as="blend_by_spread", max_rows=200)
    hm = pl.concat([harness_maker(fv, trades, src) for src in ("blend", "mid_only")], how="diagonal_relaxed")
    rep.table("fundamental.evaluate_maker as shipped (both sides, margin chosen on dev, strict)", hm, save_as="harness_maker", max_rows=150)
    grid = variant_grid(fv, trades)
    grid.write_parquet(rep.dir / "variant_grid.parquet")
    chosen = choose_on_dev(grid)
    rep.table("Variants chosen on dev (margin, |fair − mid| ≥ x, sides) and their October holdout", chosen, save_as="chosen", max_rows=120)
    fixed = grid.filter((pl.col("rule") == "strict") & (pl.col("source") == "blend") & (pl.col("sides") == "favoured")
                        & (pl.col("margin") == 0.0)).sort("tier", "lead", "x", "split")
    rep.table("Blend, strict, favoured side only, margin 0: by edge filter x", fixed, max_rows=80)
    for rule in ("strict", "improve"):
        rep.table(f"Fill rate and PnL by |fair − mid| (blend, margin 0, both sides, {rule})",
                  edge_bands(fv, trades, "blend", 0.0, rule), max_rows=120)
    # The FCS 24 h configuration chosen on dev (strict, blend): weekly results and the forward freeze.
    cfg, theta = forward_config(fv, trades)
    configs = {"frozen": {**cfg, "rule": "strict"}, "frozen_improve": {**cfg, "rule": "improve"},
               "blend_m0_both": {"source": "blend", "margin": 0.0, "x": 0.0, "sides": "both", "rule": "strict"},
               "mid_only_m0_both": {"source": "mid_only", "margin": 0.0, "x": 0.0, "sides": "both", "rule": "strict"},
               "mid_only_at_frozen": {**cfg, "source": "mid_only", "rule": "strict"}}
    rep.table("Maker by quoted spread at t (≤5c / 6-10c / 11-15c / >15c)", maker_by_spread(fv, trades, configs),
              save_as="maker_by_spread", max_rows=250)
    weekly, segs = [], []
    for rule in ("strict", "improve"):
        for lead in LEADS_H:
            part = with_fair(fv.filter((pl.col("tier") == "FCS") & (pl.col("lead") == lead)), "blend")
            o = fills(part, trades, cfg["margin"], rule).with_columns((pl.col("t") >= DEV_END).alias("holdout"),
                                                                      (pl.col("p") - pl.col("mid")).abs().alias("edge"))
            o = o.with_columns(favoured(o).alias("fav"),
                               spread_bucket(pl.col("ask") - pl.col("bid")).alias("quoted_spread"))
            o = o.filter((pl.col("edge") >= cfg["x"] - 1e-9) & (pl.lit(cfg["sides"] == "both") | pl.col("fav")))
            weekly.append(per_week(o, games).with_columns(pl.lit(rule).alias("rule"), pl.lit(lead).alias("lead")))
            for by in ("kind", "quoted_spread", "side"):
                for (h, seg), q in o.group_by("holdout", by):
                    segs.append(summarise(q, rule=rule, lead=lead, split="holdout" if h else "dev", by=by, segment=seg))
    rep.table(f"FCS per ESPN week at the frozen config {cfg}", pl.concat(weekly, how="diagonal_relaxed"), save_as="weekly", max_rows=60)
    rep.table("FCS at the frozen config: by contract kind, quoted spread at t and side",
              pl.DataFrame(segs, infer_schema_length=None).sort("rule", "lead", "by", "segment", "split"), save_as="segments", max_rows=150)
    path = cache_dir(s) / "frozen.json"
    if path.exists():  # forward snapshots already use it: never overwrite, only compare
        prior = json.loads(path.read_text())
        rep.json("Frozen for the forward test (kept; this run's dev choice shown for comparison)",
                 {"path": str(path), "frozen_at": prior["frozen_at"], "maker": prior["maker"],
                  "taker_theta": prior["taker_theta"], "this_run_dev_choice": {"maker": cfg, "taker_theta": theta}})
    else:
        freeze(s, blends, residual_dists(s), cfg, theta)
        rep.json("Frozen for the forward test", {"path": str(path), "maker": cfg, "taker_theta": theta})
    if out_preds:
        hold = fv.filter(pl.col("t") >= DEV_END).select("ticker", "t", pl.col("fair").alias("p_fair"), "mid", "p_model", "fair_mid",
                                                        "bid", "ask", "lead", "kind", "tier", "game_id", "y")
        hold.write_parquet(s.data_dir / "cache" / f"fundamental_{STUDY}_holdout_preds.parquet")
    return rep.save()


# ---- frozen configuration -----------------------------------------------------------------------
def freeze(s, blends: dict[str, Any], dists: dict[str, Any], config: dict[str, Any], theta: float) -> Path:
    """Write the dev-fitted fair value, residual laws, maker config and taker θ used by the
    forward test (``data/cache/ncaaf_maker/frozen.json``)."""
    path = cache_dir(s) / "frozen.json"
    path.write_text(json.dumps({"frozen_at": datetime.now(UTC).isoformat(), "blends": blends,
                                "dists": {k: {"scale": d.scale, "df": d.df} for k, d in dists.items()},
                                "maker": config, "taker_theta": theta}, indent=1, default=str))
    return path


def load_frozen(s) -> dict[str, Any]:
    f = json.loads((cache_dir(s) / "frozen.json").read_text())
    f["dists"] = {k: nc.Dist(v["scale"], v["df"]) for k, v in f["dists"].items()}
    return f


# ---- forward test (no orders) -------------------------------------------------------------------
def open_rows(payload: dict[str, Any], fetched_at: datetime) -> list[dict[str, Any]]:
    """Open legs from a ``/markets`` page: the shared parser's fields plus displayed sizes."""
    from albert.venues import kalshi as k

    size = {m["ticker"]: (k.num(k._field(m, "yes_bid_size")), k.num(k._field(m, "yes_ask_size")))
            for m in payload.get("markets", [])}
    rows = []
    for r in k.market_rows(payload, {}, "ncaaf-maker-forward"):
        bs, as_ = size.get(r["ticker"], (None, None))
        rows.append({**r, "yes_bid_size": bs, "yes_ask_size": as_, "quote_ts": fetched_at})
    return rows


def _current_games(s, now: datetime, stamp: str) -> pl.DataFrame:
    """The cached game table with every 2026 week that overlaps [now − 9 d, now + 9 d] re-read
    fresh from ESPN (new files under data/cache/ncaaf_maker/espn/<stamp>/; nothing overwritten)."""
    cached = pl.read_parquet(s.data_dir / "cache" / "fundamental_ncaaf_games.parquet")
    near = (cached.filter((pl.col("season") == now.year) & (pl.col("season_type") == 2)
                          & pl.col("kickoff").is_between(now - timedelta(days=9), now + timedelta(days=9)))
            ["week"].unique().to_list())
    rows: list[dict[str, Any]] = []
    for week in sorted(near):
        for group in ("80", "81"):
            payload = nc.espn_get(s, "scoreboard", {"dates": str(now.year), "seasontype": "2", "week": str(week),
                                                    "groups": group, "limit": "500"},
                                  cache_dir=cache_dir(s) / "espn" / stamp)
            rows.extend(nc.game_rows(payload, group))
    fresh = pl.DataFrame(rows, schema=nc.GAME_SCHEMA).sort("group").unique("game_id", keep="first")
    games = pl.concat([cached.filter(~pl.col("game_id").is_in(fresh["game_id"].implode())), fresh])
    return games.filter(~pl.col("status").is_in(["STATUS_CANCELED", "STATUS_POSTPONED"])).sort("kickoff", "game_id")


def _patient(pull, attempts: int = 12):
    """Call ``pull`` (one request), retrying after rate-limit errors with a growing pause."""
    import time

    for attempt in range(attempts):
        try:
            return pull()
        except RuntimeError as error:
            if attempt == attempts - 1:
                raise
            print(f"rate-limited ({str(error)[-60:]}); retrying in {5 * (attempt + 1)} s", flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def _all_pages(client, series: str, status: str, min_close: datetime | None = None):
    """Every live ``/markets`` page of one series and status (each page retried patiently, so
    a snapshot is never partial)."""
    params: dict[str, Any] = {"series_ticker": series, "status": status, "limit": 1000, "mve_filter": "exclude"}
    if min_close is not None:
        params["min_close_ts"] = int(min_close.timestamp())
    cursor = None
    while True:
        page = _patient(lambda: client.get("/markets", {**params, "cursor": cursor}, cache=False))
        yield page
        cursor = page.payload.get("cursor")
        if not cursor or not page.payload.get("markets"):
            return


def forward_prices(legs: pl.DataFrame, frozen: dict[str, Any]) -> pl.DataFrame:
    """Fair value, the hypothetical maker bids and the taker decision (frozen config) for
    snapshot legs that carry p (model), bid, ask, kind, tier, tgroup and the fee columns. Legs
    without a two-sided quote keep a null fair value and no orders."""
    from albert.research.fundamental import taker_fee

    cfg = frozen["maker"]
    theta = float(frozen["taker_theta"])
    q = legs.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    ok = q.filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("bid") > 0)
                  & (pl.col("ask") < 1) & (pl.col("ask") > pl.col("bid")) & pl.col("p").is_not_null())
    fv = apply_fair(ok, frozen["blends"])
    o = maker_orders(with_fair(fv, cfg["source"]), cfg["margin"])
    o = o.with_columns(favoured(o).alias("fav"), (pl.col("p") - pl.col("mid")).abs().alias("edge"))
    o = o.with_columns(((pl.col("edge") >= cfg["x"] - 1e-9) & (pl.lit(cfg["sides"] == "both") | pl.col("fav"))
                        & pl.col("tier").is_in(cfg["tiers"])).alias("post"))
    wide = o.pivot(on="side", index="ticker", values=["price", "post", "improves"])
    wide = wide.rename({c: c.replace("price_", "px_") for c in wide.columns})
    # Taker: buy YES at the ask or NO at 1 − bid when the fair value's edge after the fee ≥ θ.
    tk = []
    for r in fv.select("ticker", "fair", "bid", "ask", "fee_multiplier").iter_rows(named=True):
        mult = r["fee_multiplier"] or 1.0
        ey = r["fair"] - r["ask"] - taker_fee(r["ask"], mult)
        en = (1 - r["fair"]) - (1 - r["bid"]) - taker_fee(1 - r["bid"], mult)
        side, price, edge = ("yes", r["ask"], ey) if ey >= en else ("no", round(1 - r["bid"], 4), en)
        tk.append({"ticker": r["ticker"], "taker_side": side, "taker_price": price, "taker_edge": edge,
                   "taker_go": bool(edge >= theta - 1e-12)})
    taker = pl.DataFrame(tk, schema={"ticker": pl.String, "taker_side": pl.String, "taker_price": pl.Float64,
                                     "taker_edge": pl.Float64, "taker_go": pl.Boolean})
    out = (q.rename({"p": "p_model"}).join(fv.select("ticker", "fair", "fair_mid"), on="ticker", how="left")
           .join(wide, on="ticker", how="left").join(taker, on="ticker", how="left"))
    return out.with_columns(pl.lit(cfg["source"]).alias("cfg_source"), pl.lit(float(cfg["margin"])).alias("cfg_margin"),
                            pl.lit(float(cfg["x"])).alias("cfg_x"), pl.lit(cfg["sides"]).alias("cfg_sides"),
                            pl.lit(theta).alias("cfg_theta"))


def snapshot_open(s, *, horizon: timedelta = timedelta(days=7), now: datetime | None = None,
                  frozen: dict[str, Any] | None = None, out_dir: Path | None = None) -> Path:
    """Record open FCS/FBS legs kicking off within ``horizon``: the public quote, the frozen fair
    value and the maker bids we would rest, to data/cache/ncaaf_maker/forward/<stamp>.parquet."""
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    now = now or datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    frozen = frozen or load_frozen(s)
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.2))
    fees = k.series_frame(_patient(api.series).payload).filter(pl.col("series_ticker").is_in(list(nc.SERIES)))
    rows = [r for series in nc.SERIES for page in _all_pages(api.client, series, "open")
            for r in open_rows(page.payload, page.fetched_at)]
    mk = (pl.DataFrame(rows, infer_schema_length=None)
          .drop("fee_type", "fee_multiplier").join(fees.select("series_ticker", "fee_type", "fee_multiplier"), on="series_ticker", how="left")
          .with_columns(pl.lit(0, dtype=pl.Int8).alias("y")))
    games = _current_games(s, now, stamp)
    tiers = nc.team_tiers(games.filter(pl.col("completed")))
    match = nc.match_events(mk.filter(pl.col("series_ticker") == "KXNCAAFGAME"), games)
    # Moneylines keep the home leg only (the away leg is its mirror), as in the study.
    legs = (nc.ladder_legs(mk, match, games).drop("y", "y_espn")
            .filter((pl.col("kickoff") > now) & (pl.col("kickoff") <= now + horizon)
                    & ((pl.col("kind") != "ml") | (pl.col("team_side") == "home"))))
    legs = legs.join(mk.select("ticker", pl.col("yes_bid").alias("bid"), pl.col("yes_ask").alias("ask"), "yes_bid_size",
                               "yes_ask_size", "last_price", "volume", "quote_ts", "fee_type", "fee_multiplier"), on="ticker")
    # Model at the snapshot time: results enter at kickoff + 4.5 h (strictly before now).
    q = games.filter(pl.col("game_id").is_in(legs["game_id"].unique().implode())).select(
        "game_id", "season", "home_id", "away_id", "neutral").with_columns(pl.lit(now).alias("t"))
    hist = games.filter(pl.col("kickoff") < now)
    margin = nc.run_kf(hist, q, nc.MODELS[nc.CHOSEN["margin"]], tiers)
    totals = nc.run_kf(hist, q, nc.MODELS[nc.CHOSEN["totals"]], tiers)
    pred = margin.select("game_id", "t", "mu_margin", "sd_margin").join(
        totals.select("game_id", "t", "mu_total", "sd_total", "mu_home", "mu_away", "sd_home", "sd_away"), on=["game_id", "t"])
    tier = nc.game_tier(games, tiers)
    lp = nc.leg_probabilities(legs.with_columns(pl.lit(now).alias("t")), pred, frozen["dists"])
    lp = lp.join(tier, on="game_id", how="left").with_columns(
        pl.col("tier").replace_strict(TIER_GROUP, default="FBS").alias("tgroup"))
    out = forward_prices(lp, frozen).with_columns(pl.lit(now).alias("snapshot_ts"), pl.col("kickoff").alias("end"))
    path = (out_dir or cache_dir(s) / "forward") / f"{stamp}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    return path


def _settled_results(s, tickers: list[str], since: datetime) -> pl.DataFrame:
    """Settled result (y) of these legs from live ``/markets?status=settled`` pages."""
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.2))
    rows = [r for series in nc.SERIES for page in _all_pages(api.client, series, "settled", since)
            for r in k.market_rows(page.payload, {}, "ncaaf-maker-score")]
    if not rows:
        return pl.DataFrame(schema={"ticker": pl.String, "y": pl.Int8})
    m = pl.DataFrame(rows, infer_schema_length=None).filter(pl.col("ticker").is_in(tickers) & pl.col("result").is_in(["yes", "no"]))
    return m.select("ticker", (pl.col("result") == "yes").cast(pl.Int8).alias("y")).unique("ticker")


def forward_orders(snaps: pl.DataFrame) -> pl.DataFrame:
    """Long frame of the maker bids recorded in snapshots: one row per leg x side."""
    keep = ["ticker", "game_id", "tier", "kind", "snapshot_ts", "kickoff", "bid", "ask", "mid", "fair",
            "fee_type", "fee_multiplier"]
    parts = [snaps.select(*keep, pl.lit(side).alias("side"), pl.col(f"px_{side}").alias("price"),
                          pl.col(f"improves_{side}").alias("improves"), pl.col(f"post_{side}").alias("post"))
             for side in ("yes", "no")]
    return (pl.concat(parts).filter(pl.col("price").is_not_null() & (pl.col("price") >= 0.01 - 1e-9))
            .with_row_index("row").with_columns(pl.col("snapshot_ts").alias("t"), pl.col("kickoff").alias("end"),
                                                pl.col("game_id").alias("event")))


def score_forward(s, *, now: datetime | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Score every snapshot row whose game kicked off (≥ 10 min ago) and whose market settled.
    Maker: the bids recorded in the snapshot, filled by public trade prints (hourly high/low
    candles, hours starting at or after the snapshot and ending by kickoff; strict and improve
    rules), held to settlement with the series maker fee.
    Taker: the recorded decision at the snapshot bid/ask with the taker fee. Writes
    data/cache/ncaaf_maker/forward_{maker,taker}.parquet; returns (maker orders, taker trades)."""
    from albert.research.fundamental import taker_fee

    now = now or datetime.now(UTC)
    files = sorted((cache_dir(s) / "forward").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("no forward snapshots yet: run `python -m albert.research.ncaaf_maker snapshot`")
    snaps = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    snaps = snaps.filter(pl.col("fair").is_not_null() & (pl.col("kickoff") < now - timedelta(minutes=10)))
    if not snaps.height:
        return pl.DataFrame(), pl.DataFrame()
    res = _settled_results(s, snaps["ticker"].unique().to_list(), snaps["kickoff"].min() - timedelta(days=1))
    snaps = snaps.join(res, on="ticker")
    if not snaps.height:
        return pl.DataFrame(), pl.DataFrame()
    jobs = [{"job": f"{r['game_id']}|{r['snapshot_ts']:%Y%m%dT%H%M%S}", "tickers": sorted(r["ticker"]),
             "lo": r["snapshot_ts"], "hi": r["kickoff"]}
            for r in snaps.group_by("game_id", "snapshot_ts").agg(pl.col("ticker").unique(), pl.col("kickoff").first())
            .iter_rows(named=True)]
    trades = hl_pseudo_trades(fetch_hl_jobs(s, jobs, "forward_candles_hl"))
    orders = forward_orders(snaps).join(snaps.select("ticker", "snapshot_ts", "y"), on=["ticker", "snapshot_ts"])
    won = pl.when(pl.col("side") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y")).cast(pl.Float64)
    out = []
    for rule in ("strict", "improve"):
        h = (_first_fill(orders, trades, rule) if trades.height else
             pl.DataFrame(schema={"row": pl.UInt32, "side": pl.String, "fill_ts": pl.Datetime("us", "UTC")}))
        o = orders.join(h, on=["row", "side"], how="left").with_columns(
            pl.col("fill_ts").is_not_null().alias("filled"), pl.lit(rule).alias("rule"))
        out.append(o.with_columns(pl.when(pl.col("filled")).then(won - pl.col("price") - maker_fee_expr(pl.col("price")))
                                  .otherwise(0.0).alias("pnl")))
    maker = pl.concat(out)
    taker = snaps.filter(pl.col("taker_go").fill_null(False)).with_columns(pl.col("game_id").alias("event"))
    if taker.height:
        fee = pl.Series([taker_fee(p, m or 1.0) for p, m in zip(taker["taker_price"], taker["fee_multiplier"])])
        taker = taker.with_columns(
            (pl.when(pl.col("taker_side") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y")).cast(pl.Float64)
             - pl.col("taker_price") - fee).alias("pnl"))
    maker.write_parquet(cache_dir(s) / "forward_maker.parquet")
    taker.write_parquet(cache_dir(s) / "forward_taker.parquet")
    return maker, taker


def forward_summary(maker: pl.DataFrame) -> pl.DataFrame:
    """Per snapshot x rule x tier: the frozen-config bids (``posted``) and every bid (``all``)."""
    rows = []
    for (snap, rule, tier), part in maker.group_by("snapshot_ts", "rule", "tier"):
        rows.append(summarise(part.filter("post"), snapshot_ts=snap, rule=rule, tier=tier, orders_set="posted"))
        rows.append(summarise(part, snapshot_ts=snap, rule=rule, tier=tier, orders_set="all"))
    return pl.DataFrame(rows, infer_schema_length=None).sort("snapshot_ts", "rule", "tier", "orders_set")


def main(argv: list[str]) -> None:
    from albert.config import Settings

    s = Settings()
    cmd = argv[1] if len(argv) > 1 else "snapshot"
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(250)
    if cmd == "snapshot":
        path = snapshot_open(s)
        snap = pl.read_parquet(path)
        print(f"wrote {path} ({snap.height} legs, {snap['game_id'].n_unique()} games)")
        print(snap.group_by("tier").agg(pl.len().alias("legs"), pl.col("game_id").n_unique().alias("games"),
                                        pl.col("fair").is_not_null().sum().alias("two_sided"),
                                        pl.col("post_yes").fill_null(False).sum().alias("yes_bids"),
                                        pl.col("post_no").fill_null(False).sum().alias("no_bids")).sort("tier"))
    elif cmd == "freeze":
        fv, trades, blends = study_frame(s)
        cfg, theta = forward_config(fv, trades)
        print(freeze(s, blends, residual_dists(s), cfg, theta), cfg, theta)
    elif cmd == "study":
        print(run_study(s))
    elif cmd == "score":
        maker, taker = score_forward(s)
        if not maker.height:
            print("nothing to score yet (no snapshot game has kicked off and settled)")
            return
        print(forward_summary(maker))
        if taker.height:
            for (tier,), part in taker.group_by("tier"):
                mu, t, k = clustered_mean(part["pnl"].to_numpy(), part["event"].to_numpy())
                print(f"taker {tier}: {part.height} trades, {k} games, {mu * 100:.2f}c per contract, t {t}")
    else:
        raise SystemExit(f"unknown command {cmd!r}: use snapshot, score, freeze or study")


if __name__ == "__main__":
    main(sys.argv)


# ---- hourly trade high/low (batch candles) -------------------------------------------------------
HL_SCHEMA = {"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "price_low": pl.Float64,
             "price_high": pl.Float64, "price_close": pl.Float64, "volume": pl.Float64,
             "yes_bid_close": pl.Float64, "yes_ask_close": pl.Float64}


def hl_rows(market: dict[str, Any]) -> list[dict[str, Any]]:
    """Hourly candles of one market from a batch-candles payload, with the trade-price low and
    high of each hour (null when nothing traded)."""
    from albert.venues import kalshi as k

    rows = []
    for c in market.get("candlesticks") or []:
        price, bid, ask = c.get("price") or {}, c.get("yes_bid") or {}, c.get("yes_ask") or {}
        rows.append({"ticker": market.get("market_ticker"), "end_ts": k.parse_ts(c.get("end_period_ts")),
                     "price_low": k.num(k._field(price, "low")), "price_high": k.num(k._field(price, "high")),
                     "price_close": k.num(k._field(price, "close")), "volume": k.num(k._field(c, "volume")),
                     "yes_bid_close": k.num(k._field(bid, "close")), "yes_ask_close": k.num(k._field(ask, "close"))})
    return rows


def fetch_hl_jobs(s, jobs: list[dict[str, Any]], name: str, *, workers: int = 2,
                  save_every: int = 50) -> pl.DataFrame:
    """Hourly candles for batch jobs (``job`` key, ``tickers``, ``lo``, ``hi``): live batch
    requests of at most 100 legs and 10,000 candles each. Cached in
    data/cache/ncaaf_maker/<name>.parquet; a job is marked done (<name>_jobs.parquet) only when
    all its requests succeeded, so rate-limited jobs are retried on the next call."""
    from concurrent.futures import ThreadPoolExecutor

    from albert.http import CachedClient
    from albert.venues import kalshi as k

    path, done_path = cache_dir(s) / f"{name}.parquet", cache_dir(s) / f"{name}_jobs.parquet"
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=HL_SCHEMA)
    done = set(pl.read_parquet(done_path)["job"]) if done_path.exists() else set()
    todo = [j for j in jobs if j["job"] not in done]
    if not todo:
        return have
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.2))

    def one(j: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None]:
        hours = int((j["hi"] - j["lo"]).total_seconds() // 3600) + 2
        per = max(1, min(100, 10_000 // hours))
        out: list[dict[str, Any]] = []
        for i in range(0, len(j["tickers"]), per):
            try:
                page = api.candles_batch(j["tickers"][i:i + per], j["lo"], j["hi"], 60)
            except RuntimeError as error:
                print(f"retry later {j['job']}: {str(error)[-80:]}", flush=True)
                return j["job"], None
            for mk in page.payload.get("markets") or []:
                out.extend(hl_rows(mk))
        return j["job"], out

    def save(rows: list[dict[str, Any]], ok: list[str]) -> None:
        nonlocal have
        if rows:
            have = pl.concat([have, pl.DataFrame(rows, schema=HL_SCHEMA)]).unique(["ticker", "end_ts"])
        done.update(ok)
        for frame, target in ((have, path), (pl.DataFrame({"job": sorted(done)}, schema={"job": pl.String}), done_path)):
            tmp = target.with_suffix(".tmp")
            frame.write_parquet(tmp)
            tmp.replace(target)

    rows: list[dict[str, Any]] = []
    ok: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (key, got) in enumerate(pool.map(one, todo), 1):
            if got is not None:
                rows.extend(got)
                ok.append(key)
            if i % save_every == 0:
                save(rows, ok)
                rows, ok = [], []
                print(f"  candle jobs {i}/{len(todo)}", flush=True)
    save(rows, ok)
    return have


def fetch_hl_candles(s, df: pl.DataFrame, *, workers: int = 2) -> pl.DataFrame:
    """Hourly candles (trade high/low) of every leg over [kickoff − 25 h, kickoff], one batch job
    per game. Only legs that settled after Kalshi's historical cutoff (the batch endpoint is
    live-only); cached in data/cache/ncaaf_maker/candles_hl.parquet."""
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    cutoff = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir)).cutoff().get("market_settled_ts")
    legs = (df.filter(pl.col("close_time") >= cutoff)
            .group_by("game_id").agg(pl.col("ticker").unique().sort(), pl.col("kickoff").first()))
    jobs = [{"job": r["game_id"], "tickers": r["ticker"], "lo": r["kickoff"] - timedelta(hours=25), "hi": r["kickoff"]}
            for r in legs.iter_rows(named=True)]
    return fetch_hl_jobs(s, jobs, "candles_hl", workers=workers)
