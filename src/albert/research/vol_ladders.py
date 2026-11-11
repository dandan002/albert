"""Volatility trades on Kalshi strike ladders: implied width against a forecast width (study
``vol_ladders``; research only, no orders).

Builds on ``pm_smile`` (its ladders, snapshot quotes, implied median F / sigma_ATM and dev/holdout
split are reused unchanged) and asks whether the ladder mispricings it found are *tradeable* with
structures whose edge is not capped by the 1c tick.

**W1 Weather short-vol (KXHIGH*).** A forecast width sigma_f for the CLI daily high is fitted on dev
by interval-censored MLE, X_T ~ N(F + mu_lead, sigma_f^2) with (a) ``shrink``: sigma_f =
lambda_lead x sigma_ATM, or (b) ``nbm``: log sigma_f = a_lead + b log sigma_ATM + c log xnd +
d log rmse + e log(1 + gap) (NBM TXN spread ``xnd`` of the latest run available 2 h after its
runtime, the station's trailing 30-day RMSE of CLI high - NBM TXN over days whose CLI was out 6 h
before t, and for 08/12/16 h local the NBM-forecast remaining heating above the observed max so
far). Each leg's vol-only fair value keeps the market median: P(lo < X <= hi) under N(F,
sigma_f^2). When r = sigma_f / sigma_ATM <= r*, the short-vol legs are traded: YES on legs whose
value rises when the distribution narrows (``body``), NO on legs whose value falls (``wings``);
``fly`` = both. Taker: at the snapshot ask / 1 - bid, only legs whose fair beats price + fee.
Maker: a YES bid (body) or NO bid (wings) at the best bid (``join``) or one tick better when the
spread is >= 2c (``improve``), resting until the next snapshot lead (8 h lead: 4 h), filled only
if a later hourly candle prints a trade strictly through the price (trade low < our YES bid /
trade high > our YES-equivalent ask; trade low/high re-parsed from the recorded raw candle
payloads, no new requests); held to settlement with the series maker fee.

**W2 Weather median drift.** Buy the bracket adjacent to F on the side of the dev-sign drift
between consecutive leads, exit at the next lead's bid or hold to settlement.

**S1 Sports moneyline vs spread ladder.** The game-winner is the K = 0 digital of the spread
ladder: S(0) from the isotonic survival curve (z = Phi^-1(S) linear between the two strikes that
straddle 0) against the favourite's moneyline quote at the same snapshot; trade the moneyline
toward S(0) or toward a dev-fitted logistic blend of both.

**S2 Sports totals over bias.** Buy "Over K" legs near the implied median at the ask.

Execution: taker at the snapshot candle's ask / 1 - bid with ``kalshi_fee(ORDER_N, p,
conservative=True) / ORDER_N`` (orders of ``ORDER_N`` contracts, fee rounded up to the cent per
order, series multiplier / maker-fee flag from the markets table). Every leg of a structure pays
its own spread and fee. Displayed sizes are not in the hourly candles (see verdict.md).

Statistics: PnL per contract (cents), t clustered by event, PnL per dollar risked (sum PnL / sum
cost incl. fees), and $/week at our caps: every event is scaled to ``EVENT_CAP`` of collateral,
then scaled down so collateral x holding time never exceeds ``OPEN_CAP`` x calendar time.
Dev = t_ref before 2026-10-01, holdout = October 2026. Parameters come from small pre-declared
grids chosen on dev only; the full grids are in ``grid_*.parquet``.

Ledgers: ``book`` replays each reported variant as a chronological capped book ($100, <= $5 per
event, <= $20 open including resting orders, most whole contracts per leg) and writes one row per
filled leg to ``ledger_<strategy>_<variant>.parquet`` for the shared ``vol_metrics`` module.

Results (2026-11-11, data/research/vol_ladders/verdict.md): nothing clears the bar robustly.
Late-day (16:00 LST) weather short vol works Aug-Oct (taker body +2.8c dev / +1.8c holdout, t 2.2;
maker +4.4c / +3.9c per fill, t 2.1, fills favourably selected, candle fills = exact trade fills on
4,025 orders) but loses on the Nov 1-9 second holdout (-7c / -10c, ~15 events) and, with resting
orders inside the $20 cap, the maker ledger makes ~$1/week dev and ~$0 holdout. The NBM-conditioned
pre-day body buy has dev t 0.5. Drift, moneyline-vs-spread and totals-over fail out of sample.

    python -m albert.research.vol_ladders run [ohlc|panel|wx_taker|wx_maker|wx_drift|sp_ml|sp_totals|report|all]
    python -m albert.research.vol_ladders run [nov|wx_trades|wx_late_exact|l2_tob|l2_size]   (network / L2 steps)
"""

import itertools
import json
import math
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri

from albert.fees import kalshi_fee
from albert.research.pm_smile import HOLDOUT, cmean

OUT = Path("data/research/vol_ladders")
NOV = OUT / "nov"                 # second holdout: events settled 2026-11-01..11-09
SPLITS = ("dev", "holdout", "holdout2")
SMILE = Path("data/research/pm_smile")
ORDER_N = 5                     # contracts per leg order: the fee is rounded up per order
EVENT_CAP = 5.0                 # $ collateral per event
OPEN_CAP = 20.0                 # $ total open
WX_LEADS = (36, 24, 16, 12, 8)
NEXT_LEAD = {36: 24, 24: 16, 16: 12, 12: 8, 8: 4}
LEAD_GROUPS = {"pre-day 36/24": (36, 24), "day 16/12": (16, 12), "late 8": (8,)}
NBM_LAG = timedelta(hours=2)
CLI_LAG = timedelta(hours=6)
RMSE_DAYS, RMSE_MIN = 30, 7
EPS = 1e-9


# ---- shared helpers ------------------------------------------------------------------------------
_FEE: dict[tuple, float] = {}


def fee_pc(price: float, mult: float = 1.0, fee_type: str | None = "quadratic", maker: bool = False) -> float:
    """Kalshi fee per contract for an order of ``ORDER_N`` contracts (rounded up per order)."""
    key = (round(price, 4), mult, fee_type, maker)
    if key not in _FEE:
        _FEE[key] = kalshi_fee(ORDER_N, price, maker=maker, fee_type=fee_type or "quadratic",
                               multiplier=mult, conservative=True) / ORDER_N
    return _FEE[key]


def fees(price: np.ndarray, mult: np.ndarray, fee_type: list | np.ndarray, maker: bool = False) -> np.ndarray:
    return np.array([fee_pc(float(p), float(m), ft, maker) if np.isfinite(p) else np.nan
                     for p, m, ft in zip(price, mult, fee_type)])


def split_expr(col: str = "t_ref") -> pl.Expr:
    return pl.when(pl.col(col) < HOLDOUT).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("split")


def stat(df: pl.DataFrame) -> dict[str, Any]:
    """n contracts, events, mean PnL (c per contract), t clustered by event, PnL per $ risked."""
    if df.is_empty():
        return {"n": 0, "events": 0}
    mu, t, n, k = cmean(df["pnl"].to_numpy(), df["event_ticker"].to_numpy())
    cost = float(df["cost"].sum())
    return {"n": n, "events": k, "c": None if mu is None else round(mu * 100, 3),
            "t": None if t is None else round(t, 2),
            "per_dollar": round(float(df["pnl"].sum()) / cost, 4) if cost > 0 else None}


def at_caps(df: pl.DataFrame, weeks: float) -> tuple[float | None, pl.DataFrame]:
    """$/week at our caps and per-month $: each event's contracts are scaled to EVENT_CAP of
    collateral, then all scaled down so sum(collateral x hold) <= OPEN_CAP x calendar time.
    ``df`` needs event_ticker, pnl, cost, t_dec (entry), t_end (settlement or exit), t_ref."""
    if df.is_empty() or weeks <= 0:
        return None, pl.DataFrame()
    ev = (df.group_by("event_ticker").agg(pl.col("pnl").sum(), pl.col("cost").sum(), pl.col("t_dec").min(),
                                          pl.col("t_end").max(), pl.col("t_ref").first())
          .with_columns((EVENT_CAP / pl.col("cost")).alias("k"))
          .with_columns((pl.col("pnl") * pl.col("k")).alias("pnl_usd"),
                        ((pl.col("t_end") - pl.col("t_dec")).dt.total_seconds() / 86400).clip(1 / 24).alias("hold_d")))
    need = float((ev["hold_d"] * EVENT_CAP).sum())
    have = OPEN_CAP * weeks * 7
    f = min(1.0, have / need) if need > 0 else 1.0
    ev = ev.with_columns((pl.col("pnl_usd") * f).alias("pnl_usd"))
    mo = (ev.group_by(pl.col("t_ref").dt.strftime("%Y-%m").alias("month"))
          .agg(pl.len().alias("events"), pl.col("pnl_usd").sum().round(2).alias("usd_at_caps"))
          .sort("month"))
    return round(float(ev["pnl_usd"].sum()) / weeks, 2), mo


def weekly_usd(df: pl.DataFrame, weeks: dict[str, float]) -> dict[str, Any]:
    """Share of calendar weeks (by t_ref week) with positive $ PnL at our caps, per split."""
    out = {}
    for sp in SPLITS:
        g = df.filter(pl.col("split") == sp)
        if g.is_empty():
            continue
        ev = (g.group_by("event_ticker").agg(pl.col("pnl").sum(), pl.col("cost").sum(), pl.col("t_ref").first())
              .with_columns((pl.col("pnl") * EVENT_CAP / pl.col("cost")).alias("usd"),
                            pl.col("t_ref").dt.truncate("1w").alias("wk")))
        wk = ev.group_by("wk").agg(pl.col("usd").sum()).sort("wk")
        out[sp] = {"weeks": wk.height, "positive": int((wk["usd"] > 0).sum()),
                   "median_usd": round(float(wk["usd"].median()), 2), "worst_usd": round(float(wk["usd"].min()), 2)}
    return out


def monthly(df: pl.DataFrame) -> pl.DataFrame:
    """Per month: contracts, events, c per contract, t, per $ risked."""
    rows = []
    for (mo,), g in df.group_by(pl.col("t_ref").dt.strftime("%Y-%m"), maintain_order=True):
        rows.append({"month": mo, **stat(g)})
    return pl.DataFrame(rows).sort("month") if rows else pl.DataFrame()


def weeks_of(df: pl.DataFrame, col: str = "t_ref") -> dict[str, float]:
    out = {}
    for sp in SPLITS:
        g = df.filter(pl.col("split") == sp)
        out[sp] = ((g[col].max() - g[col].min()).total_seconds() / 604800 + 1 / 7) if g.height else 0.0
    return out


def run_grid(name: str, trades: pl.DataFrame, variants: list[dict[str, Any]], weeks: dict[str, float],
             key: str = "variant", min_dev_events: int = 30) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Grid table (dev and holdout stats per variant) and the dev-chosen variant (highest dev PnL
    per $ risked with >= ``min_dev_events`` events), with its holdout, months and $/week."""
    rows = []
    for v in variants:
        g = trades.filter(pl.col(key) == v[key])
        row = dict(v)
        for sp in SPLITS:
            s = g.filter(pl.col("split") == sp)
            row.update({f"{sp}_{k}": x for k, x in stat(s).items()})
            row[f"{sp}_usd_week"] = at_caps(s, weeks[sp])[0]
        rows.append(row)
    grid = pl.DataFrame(rows, infer_schema_length=None)
    ok = grid.filter((pl.col("dev_events") >= min_dev_events) & pl.col("dev_per_dollar").is_not_null())
    chosen: dict[str, Any] = {"strategy": name}
    if ok.height:
        best = ok.sort("dev_per_dollar", descending=True).row(0, named=True)
        chosen.update(best)
        g = trades.filter(pl.col(key) == best[key])
        chosen["months"] = monthly(g).to_dicts()
        chosen["months_usd"] = [r for sp in SPLITS for r in at_caps(g.filter(pl.col("split") == sp), weeks[sp])[1].to_dicts()]
        chosen["weeks_usd"] = weekly_usd(g, weeks)
    grid.write_parquet(OUT / f"grid_{name}.parquet")
    return grid, chosen


# ---- chronological capped book and trade ledger (for the shared vol_metrics module) --------------
BANKROLL = 100.0
LEDGER_SPLIT = {"dev": "dev", "holdout": "holdout", "holdout2": "nov"}
LEDGER_SCHEMA = {"strategy": pl.String, "variant": pl.String, "split": pl.String, "struct_id": pl.String,
                 "event_ticker": pl.String, "ticker": pl.String, "side": pl.String, "n": pl.Int64,
                 "t_in": pl.Datetime("us", "UTC"), "cost_usd": pl.Float64, "settle_ts": pl.Datetime("us", "UTC"),
                 "payoff_usd": pl.Float64}


def _leg_cost(n: int, price: float, maker: bool, fee_type: str | None, mult: float | None) -> float:
    return n * price + kalshi_fee(n, price, maker=maker, fee_type=fee_type or "quadratic", multiplier=mult,
                                  conservative=True)


def book(legs: pl.DataFrame, strategy: str, variant: str) -> pl.DataFrame:
    """Chronological capped book: $100 bankroll, <= EVENT_CAP collateral per event and <= OPEN_CAP
    open (positions and resting orders), each structure sized to the most whole contracts per leg
    (equal n on every leg) that the caps allow, fees per leg order via ``kalshi_fee(conservative)``.

    ``legs``: struct_id, event_ticker, ticker, side (held), price (paid for the held side), maker,
    fee_type, fee_mult, t_post, t_fill (null = never filled), t_cancel, settle_ts, won, split, and
    optionally exit_ts / exit_price (sold before settlement at exit_price, taker fee). A leg holds
    collateral from t_post until settlement (filled) or t_cancel (unfilled). Returns one ledger row
    per filled leg (``LEDGER_SCHEMA``)."""
    cols = legs.columns
    has_exit = "exit_ts" in cols
    by = {}
    for r in legs.sort("t_post", "struct_id", "ticker", "side").iter_rows(named=True):
        by.setdefault(r["struct_id"], []).append(r)
    active: list[tuple[datetime, str, float, float]] = []      # (release, event, amount, pnl at release)
    realized = 0.0
    rows: list[dict[str, Any]] = []
    for sid, lg in sorted(by.items(), key=lambda kv: (kv[1][0]["t_post"], kv[0])):
        t = lg[0]["t_post"]
        keep = []
        for a in active:
            if a[0] <= t:
                realized += a[3]
            else:
                keep.append(a)
        active = keep
        open_all = sum(a[2] for a in active)
        open_ev = sum(a[2] for a in active if a[1] == lg[0]["event_ticker"])
        cap = min(EVENT_CAP - open_ev, OPEN_CAP - open_all, BANKROLL + realized - open_all)
        unit = sum(r["price"] for r in lg)
        if cap <= 0 or unit <= 0:
            continue
        n = int(cap / unit)
        while n > 0 and sum(_leg_cost(n, r["price"], r["maker"], r["fee_type"], r["fee_mult"]) for r in lg) > cap + 1e-9:
            n -= 1
        if n <= 0:
            continue
        for r in lg:
            amt = _leg_cost(n, r["price"], r["maker"], r["fee_type"], r["fee_mult"])
            if r["t_fill"] is None:
                active.append((r["t_cancel"], r["event_ticker"], amt, 0.0))
                continue
            if has_exit and r.get("exit_ts") is not None and r.get("exit_price") is not None:
                pay = n * r["exit_price"] - kalshi_fee(n, r["exit_price"], fee_type=r["fee_type"] or "quadratic",
                                                       multiplier=r["fee_mult"], conservative=True)
                end = r["exit_ts"]
            else:
                pay, end = (float(n) if r["won"] else 0.0), r["settle_ts"]
            active.append((end, r["event_ticker"], amt, pay - amt))
            rows.append({"strategy": strategy, "variant": variant, "split": LEDGER_SPLIT.get(r["split"], r["split"]),
                         "struct_id": str(sid), "event_ticker": r["event_ticker"], "ticker": r["ticker"],
                         "side": r["side"], "n": n, "t_in": r["t_fill"], "cost_usd": round(amt, 6),
                         "settle_ts": end, "payoff_usd": round(pay, 6)})
    return pl.DataFrame(rows, schema=LEDGER_SCHEMA)


def _settle_map() -> pl.DataFrame:
    from albert.config import settings

    mk = pl.scan_parquet(settings().tables_dir / "kalshi_markets.parquet").select(
        "ticker", pl.coalesce("settlement_ts", "close_time").alias("settle_ts")).collect()
    nov = NOV / "tables" / "kalshi_markets.parquet"
    if nov.exists():
        mk = pl.concat([mk, pl.read_parquet(nov).select("ticker", pl.coalesce("settlement_ts", "close_time").alias("settle_ts"))])
    return mk.unique("ticker", keep="last")


def write_ledger(strategy: str, variant: str, legs: pl.DataFrame) -> dict[str, Any]:
    """Run the capped book on ``legs`` (see ``book``; settle_ts filled from the markets tables when
    absent) and write data/research/vol_ladders/ledger_<strategy>_<variant>.parquet."""
    import re

    if "settle_ts" not in legs.columns:
        legs = legs.join(_settle_map(), on="ticker", how="left")
    led = book(legs, strategy, variant)
    safe = re.sub(r"[^A-Za-z0-9.=-]+", "_", variant).strip("_")
    path = OUT / f"ledger_{strategy}_{safe}.parquet"
    led.write_parquet(path)
    summ: dict[str, Any] = {"path": str(path), "legs": led.height}
    for sp, g in led.group_by("split"):
        days = max((g["t_in"].max() - g["t_in"].min()).total_seconds() / 86400, 1.0)
        summ[sp[0]] = {"legs": g.height, "structs": g["struct_id"].n_unique(), "events": g["event_ticker"].n_unique(),
                       "pnl_usd": round(float((g["payoff_usd"] - g["cost_usd"]).sum()), 2),
                       "cost_usd": round(float(g["cost_usd"].sum()), 2),
                       "usd_week": round(float((g["payoff_usd"] - g["cost_usd"]).sum()) / days * 7, 2),
                       "mean_n": round(float(g["n"].mean()), 2)}
    return summ


# ---- weather: raw candle OHLC ---------------------------------------------------------------------
def parse_candle_payload(url: str, params: dict[str, Any], resp: dict[str, Any],
                         prefixes: tuple[str, ...] = ("KXHIGH", "KXLOWT")) -> list[dict[str, Any]]:
    """Hourly (ticker, end_ts, trade low/high, volume, bid/ask close) from one recorded candlestick
    response (batch ``{"markets": [...]}`` or per-market ``{"candlesticks": [...]}``)."""
    if int(params.get("period_interval") or 0) != 60:
        return []
    if "markets" in resp:
        mk = [(m.get("market_ticker"), m.get("candlesticks") or []) for m in resp.get("markets") or []]
    else:
        tk = resp.get("ticker") or url.rstrip("/").split("/")[-2]
        mk = [(tk, resp.get("candlesticks") or [])]
    rows = []
    for tk, cs in mk:
        if not tk or not tk.startswith(prefixes):
            continue
        for c in cs:
            pr, b, a = c.get("price") or {}, c.get("yes_bid") or {}, c.get("yes_ask") or {}

            def f(d: dict, k: str) -> float | None:
                v = d.get(k) if d.get(k) is not None else d.get(f"{k}_dollars")
                return None if v in (None, "") else float(v)

            rows.append({"ticker": tk, "end_ts": datetime.fromtimestamp(int(c["end_period_ts"]), tz=UTC),
                         "p_low": f(pr, "low"), "p_high": f(pr, "high"),
                         "volume": f(c, "volume") if f(c, "volume") is not None else f(c, "volume_fp"),
                         "bid": f(b, "close"), "ask": f(a, "close")})
    return rows


OHLC_SCHEMA = {"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "p_low": pl.Float64, "p_high": pl.Float64,
               "volume": pl.Float64, "bid": pl.Float64, "ask": pl.Float64}


def step_ohlc(s) -> pl.DataFrame:
    """Re-parse every recorded weather candlestick response listed in the ingest manifests."""
    from albert.http import read_raw

    rows: list[dict[str, Any]] = []
    for man in sorted((s.data_dir / "runs").glob("*ingest-kalshi-detail.json")):
        m = json.loads(man.read_text())
        ser = (m.get("params") or {}).get("series") or []
        if not any(x.startswith(("KXHIGH", "KXLOWT")) for x in ser):
            continue
        for p in m["raw_files"]:
            path = Path(p)
            if not (path.exists() or path.with_name(path.name + ".zst").exists()):
                continue
            r = read_raw(path)
            if "candlesticks" not in r.get("url", ""):
                continue
            rows.extend(parse_candle_payload(r["url"], r.get("params") or {}, r.get("response") or {}))
        print(f"{man.name}: {len(rows)} candle rows", flush=True)
    df = pl.DataFrame(rows, schema=OHLC_SCHEMA).unique(["ticker", "end_ts"], keep="first").sort("ticker", "end_ts")
    df.write_parquet(OUT / "wx_ohlc.parquet")
    print(f"wx_ohlc: {df.height} rows, {df['ticker'].n_unique()} markets", flush=True)
    return df


# ---- weather: snapshot panel with forecast features ----------------------------------------------
def event_codes(tables: Path) -> pl.DataFrame:
    """event_ticker -> CLI station code and measurement day for KXHIGH / KXLOWT events."""
    from albert.research.weather import _event_day
    from albert.venues.weather import cli_code

    mk = (pl.scan_parquet(tables / "kalshi_markets.parquet")
          .filter(pl.col("series_ticker").str.contains(r"^KX(HIGH|LOWT)"))
          .select("event_ticker", "rules_primary").unique("event_ticker").collect())
    return pl.DataFrame({"event_ticker": mk["event_ticker"],
                         "code": [cli_code(r) for r in mk["rules_primary"]],
                         "day": [_event_day(e) for e in mk["event_ticker"]]},
                        schema={"event_ticker": pl.String, "code": pl.String, "day": pl.Date})


def nbm_txn(nbm: pl.DataFrame, kind: str = "high") -> pl.DataFrame:
    """NBM max (ftime 00Z of D+1) or min (12Z of D) rows with their availability time."""
    hour = 0 if kind == "high" else 12
    return (nbm.filter(pl.col("txn").is_not_null() & (pl.col("ftime").dt.hour() == hour))
            .select("code", "ftime", (pl.col("runtime") + NBM_LAG).alias("avail"), "txn", "xnd")
            .sort("avail"))


def target_ftime(day: pl.Expr, kind: str = "high") -> pl.Expr:
    base = day.cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    return base + pl.duration(days=1) if kind == "high" else base + pl.duration(hours=12)


def nbm_at(df: pl.DataFrame, txn: pl.DataFrame, t_col: str, kind: str = "high") -> pl.DataFrame:
    """Latest NBM txn/xnd for (code, day) available at ``t_col`` (runtime + NBM_LAG <= t)."""
    d = df.with_columns(target_ftime(pl.col("day"), kind).alias("ftime")).sort(t_col)
    return d.join_asof(txn, left_on=t_col, right_on="avail", by=["code", "ftime"], strategy="backward").drop("ftime")


def lst_bounds(code: str, day: date) -> tuple[datetime, datetime]:
    from albert.venues.weather import lst_day_bounds
    return lst_day_bounds(code, day)


def station_errors(cli: pl.DataFrame, txn: pl.DataFrame) -> pl.DataFrame:
    """Per (code, day): CLI high minus the NBM TXN of the run available at the LST day start, and
    the time the CLI value is usable (day end + CLI_LAG)."""
    rows = []
    for code, day in cli.select("code", "day").iter_rows():
        if code is None or day is None:
            continue
        try:
            lo, hi = lst_bounds(code, day)
        except KeyError:
            continue
        rows.append({"code": code, "day": day, "t0": lo, "known": hi + CLI_LAG})
    b = pl.DataFrame(rows)
    b = nbm_at(b, txn, "t0").join(cli.select("code", "day", "high"), on=["code", "day"])
    return b.filter(pl.col("txn").is_not_null() & pl.col("high").is_not_null()).with_columns(
        (pl.col("high") - pl.col("txn")).alias("err")).select("code", "day", "known", "err").sort("code", "known")


def rolling_rmse(panel: pl.DataFrame, err: pl.DataFrame) -> pl.Series:
    """Trailing RMSE of the station's NBM error over the last RMSE_DAYS days known before t_snap."""
    by = {c: (g.sort("known")["known"].to_numpy(), g.sort("known")["err"].to_numpy().astype(float))
          for (c,), g in err.group_by("code")}
    out = np.full(panel.height, np.nan)
    for i, (code, t) in enumerate(zip(panel["code"], panel["snapshot_ts"].to_numpy())):
        if code not in by:
            continue
        kn, e = by[code]
        j = np.searchsorted(kn, t, side="right")
        w = e[max(0, j - RMSE_DAYS):j]
        if len(w) >= RMSE_MIN:
            out[i] = math.sqrt(float(np.mean(w**2)))
    return pl.Series("rmse", out)


def obs_features(panel: pl.DataFrame, obs: pl.DataFrame, nbm: pl.DataFrame) -> pl.DataFrame:
    """Intraday (t after the LST day start): observed max so far M and the NBM-forecast remaining
    heating gap = max(max NBM tmp over (t, day end] of the latest run - M, 0)."""
    ob = {c: (g.sort("ts")["ts"].to_numpy(), g.sort("ts")["tmpf"].to_numpy()) for (c,), g in obs.group_by("code")}
    nb = {c: g for (c,), g in nbm.filter(pl.col("tmp").is_not_null()).with_columns(
        (pl.col("runtime") + NBM_LAG).alias("avail")).group_by("code")}
    M, gap = np.full(panel.height, np.nan), np.full(panel.height, np.nan)
    for i, r in enumerate(panel.select("code", "day", "snapshot_ts").iter_rows()):
        code, day, t = r
        if code not in ob or day is None:
            continue
        lo, hi = lst_bounds(code, day)
        if t <= lo:
            continue
        ts, tf = ob[code]
        sel = (ts >= np.datetime64(lo.replace(tzinfo=None))) & (ts < np.datetime64(t.replace(tzinfo=None)))
        if not sel.any():
            continue
        M[i] = float(tf[sel].max())
        g = nb.get(code)
        if g is None:
            continue
        runs = g.filter(pl.col("avail") <= t)
        if runs.is_empty():
            continue
        rest = runs.filter((pl.col("runtime") == runs["runtime"].max()) & (pl.col("ftime") > t) & (pl.col("ftime") <= hi))
        gap[i] = max(float(rest["tmp"].max()) - M[i], 0.0) if rest.height else 0.0
    return panel.with_columns(pl.Series("obs_max", M), pl.Series("gap", gap))


def panel_from(sn: pl.DataFrame, lg: pl.DataFrame, lad: pl.DataFrame, codes: pl.DataFrame, nbm: pl.DataFrame,
               cli: pl.DataFrame, obs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(snapshots, legs) for weather-high ladders with forecast features (pm_smile frames in)."""
    sn = (sn.filter(pl.col("group") == "wx_high")
          .select("snap_id", "ladder", "event_ticker", "series_ticker", "snapshot_ts", "t_ref", "lead_h", "F",
                  "sigma_atm", "x_lo", "x_hi", "split", "n_legs", "n_legs_used"))
    sn = sn.join(codes, on="event_ticker", how="left").filter(pl.col("code").is_not_null())
    txn = nbm_txn(nbm, "high")
    sn = nbm_at(sn, txn, "snapshot_ts").rename({"avail": "nbm_avail"})
    err = station_errors(cli, txn)
    sn = sn.sort("snap_id")
    sn = sn.with_columns(rolling_rmse(sn, err))
    sn = obs_features(sn, obs, nbm)
    lg = lg.select("snap_id", "ticker", "lo", "hi", "bid", "ask", "ok", "y").join(sn.select("snap_id"), on="snap_id")
    lg = lg.join(lad.select("ticker", "fee_mult", "fee_type", "close_time").unique("ticker"), on="ticker", how="left")
    return sn, lg


def build_panel(s) -> tuple[pl.DataFrame, pl.DataFrame]:
    return panel_from(pl.read_parquet(SMILE / "snapshots.parquet"), pl.read_parquet(SMILE / "legs.parquet"),
                      pl.read_parquet(SMILE / "ladder_legs.parquet"), event_codes(s.tables_dir),
                      pl.read_parquet(s.tables_dir / "wx_nbm.parquet"), pl.read_parquet(s.tables_dir / "wx_cli.parquet"),
                      pl.read_parquet(s.tables_dir / "wx_obs.parquet"))


# ---- weather: second holdout (events settled 2026-11-01..11-09) ---------------------------------
NOV_FROM = datetime(2026, 11, 1, 12, tzinfo=UTC)       # first close of a Nov 1 event is ~Nov 2 05:00 UTC
NOV_TO = datetime(2026, 11, 10, 12, tzinfo=UTC)


def step_nov(s) -> dict[str, Any]:
    """Fetch the settled KXHIGH markets closing in [NOV_FROM, NOV_TO), their hourly candles (batch,
    raw pages append-only) and the IEM CLI / ASOS / NBM data around them, then rebuild pm_smile's
    snapshots and this module's panel with the dev-fitted sigma models (no refit)."""
    from albert.http import CachedClient
    from albert.research import pm_smile as sm
    from albert.venues import kalshi as k
    from albert.venues import weather as wx

    NOV.mkdir(parents=True, exist_ok=True)
    tables = NOV / "tables"
    tables.mkdir(exist_ok=True)
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.25, max_retries=8))
    ser = pl.read_parquet(s.tables_dir / "kalshi_series.parquet")
    info = {r["series_ticker"]: r for r in ser.to_dicts()}
    series = sorted(x for x in ser["series_ticker"] if x.startswith("KXHIGH"))
    rows: list[dict[str, Any]] = []
    for sr in series:
        for page in api.series_markets(sr, "settled", min_close=NOV_FROM, max_close=NOV_TO, cache=True):
            rows.extend(k.market_rows(page.payload, info, "vol_ladders_nov"))
    mk = k.markets_frame(rows, non_financial_only=False)
    print(f"nov markets {mk.height} in {mk['event_ticker'].n_unique()} events", flush=True)
    mk.write_parquet(tables / "kalshi_markets.parquet")
    ser.write_parquet(tables / "kalshi_series.parquet")
    # hourly candles, batch endpoint: <= 100 tickers and <= 10,000 candles per request
    mk = mk.with_columns(pl.col("close_time").dt.date().alias("cday"))
    ohlc: list[dict[str, Any]] = []
    for (_,), g in mk.group_by("cday"):
        lo, hi = g["open_time"].min(), g["close_time"].max()
        per = max(1, min(100, 10_000 // (int((hi - lo).total_seconds() // 3600) + 2)))
        tk = sorted(g["ticker"].to_list())
        for i in range(0, len(tk), per):
            page = api.candles_batch(tk[i:i + per], lo, hi, 60)
            ohlc.extend(parse_candle_payload("/markets/candlesticks", {"period_interval": 60}, page.payload))
    oh = pl.DataFrame(ohlc, schema=OHLC_SCHEMA).unique(["ticker", "end_ts"]).sort("ticker", "end_ts")
    oh.write_parquet(NOV / "wx_ohlc.parquet")
    print(f"nov candles {oh.height} for {oh['ticker'].n_unique()} markets", flush=True)
    # IEM: CLI (refreshed with a date tag), ASOS and NBM from Oct 1 (for the trailing error window)
    iem = wx.IEM(CachedClient("iem", s.iem_base, s.raw_dir, min_interval=0.5, timeout=120.0))
    codes = event_codes(tables)
    cli_r, obs_r, nbm_r = [], [], []
    w0 = datetime(2026, 10, 1, tzinfo=UTC)
    for code in sorted(c for c in codes["code"].drop_nulls().unique().to_list() if c in wx.STATION_TZ):
        cli_r.extend(wx.cli_rows(iem.cli(code, 2026, tag="2026-11-10").payload, code))
        obs_r.extend(wx.asos_rows(iem.asos(code, w0.date(), NOV_TO.date() + timedelta(days=1)).payload, code))
        nbm_r.extend(wx.nbm_rows(iem.nbm(code, w0, NOV_TO).payload, code))
    tab = s.tables_dir
    nbm = pl.concat([pl.read_parquet(tab / "wx_nbm.parquet"), wx.frame(nbm_r, wx.NBM_SCHEMA)]).unique(
        ["code", "runtime", "ftime"], keep="last")
    cli = pl.concat([pl.read_parquet(tab / "wx_cli.parquet"), wx.frame(cli_r, wx.CLI_SCHEMA)]).unique(
        ["code", "day"], keep="last")
    obs = pl.concat([pl.read_parquet(tab / "wx_obs.parquet"), wx.frame(obs_r, wx.OBS_SCHEMA)]).unique(
        ["code", "ts"], keep="last")
    for nm, f in (("wx_nbm", nbm), ("wx_cli", cli), ("wx_obs", obs)):
        f.write_parquet(NOV / f"{nm}.parquet")
    # pm_smile ladders / snapshots on the new markets, quotes from the new candles
    lad, lstats = sm.build_ladders(tables)
    lad = lad.filter(pl.col("group") == "wx_high")
    snaps = sm.snapshot_times(lad)
    c = oh.select("ticker", "end_ts", "bid", "ask").filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null())
    q = sm.quotes_at(lad, snaps, c)
    sn, _, lg, why = sm.build_snapshots(lad, q)
    off = 10_000_000
    sn = sn.with_columns(pl.col("snap_id") + off, pl.lit("holdout2").alias("split"))
    lg = lg.with_columns(pl.col("snap_id") + off)
    sn, lg = panel_from(sn, lg, lad, codes, nbm, cli, obs)
    fits = json.loads((OUT / "wx_sigma.json").read_text())["fits"]
    sn = sn.filter(pl.col("xnd").is_not_null() & pl.col("rmse").is_not_null() & pl.col("F").is_not_null())
    for model in MODELS:
        sn = sn.with_columns(pl.Series(f"sigma_{model}", apply_sigma(sn, fits[model])))
    sn = sn.with_columns(*(pl.col(f"sigma_{m}").truediv(pl.col("sigma_atm")).alias(f"r_{m}") for m in ("shrink", "nbm")))
    ll = {m: round(mean_ll(sn, fits[m], sn[f"sigma_{m}"].to_numpy(), use_mu=False), 4) for m in MODELS}
    sn.write_parquet(NOV / "wx_snapshots.parquet")
    lg.write_parquet(NOV / "wx_legs.parquet")
    out = {"markets": mk.height, "ladder_stats": lstats, "build": why, "snapshots": sn.height,
           "events": sn["event_ticker"].n_unique(), "ll_mu0": ll,
           "t_ref_range": [str(sn["t_ref"].min()), str(sn["t_ref"].max())]}
    print(out, flush=True)
    return out


# ---- weather: forecast width model ---------------------------------------------------------------
MODELS = ("market", "shrink", "nbm", "nbm_only")


def _design(sn: pl.DataFrame, model: str) -> tuple[np.ndarray, list[str]]:
    lead = sn["lead_h"].to_numpy()
    cols = [(lead == h).astype(float) for h in WX_LEADS]
    names = [f"a_{h}" for h in WX_LEADS]
    if model in ("nbm", "nbm_only"):
        intra = np.isin(lead, (16, 12, 8)).astype(float)
        extra = [np.log(np.clip(sn["xnd"].to_numpy().astype(float), 0.5, None)),
                 np.log(np.clip(sn["rmse"].to_numpy().astype(float), 0.5, None)),
                 intra * np.log1p(np.nan_to_num(sn["gap"].to_numpy().astype(float), nan=0.0))]
        names += ["c_log_xnd", "d_log_rmse", "e_log1p_gap"]
        if model == "nbm":
            extra = [np.log(sn["sigma_atm"].to_numpy())] + extra
            names.insert(len(WX_LEADS), "b_log_sigma_atm")
        cols += extra
    return np.column_stack(cols), names


def _ll(theta: np.ndarray, X: np.ndarray, lead_ix: np.ndarray, F: np.ndarray, sig: np.ndarray, L: np.ndarray,
        U: np.ndarray, model: str) -> np.ndarray:
    k = len(WX_LEADS)
    mu = theta[:k][lead_ix]
    beta = theta[k:]
    if model == "market":
        s = sig
    elif model == "shrink":
        s = sig * np.exp(X @ beta)
    else:
        s = np.exp(X @ beta)
    m = F + mu * sig
    p = ndtr((U - m) / s) - ndtr((L - m) / s)
    return np.log(np.clip(p, 1e-9, None))


def fit_sigma(sn: pl.DataFrame, model: str) -> dict[str, Any]:
    """Interval-censored MLE on ``sn`` (dev): X ~ N(F + mu_lead sigma_ATM, sigma_f^2)."""
    X, names = _design(sn, model)
    lead_ix = np.array([WX_LEADS.index(h) for h in sn["lead_h"].to_list()])
    F, sig = sn["F"].to_numpy(), sn["sigma_atm"].to_numpy()
    L, U = sn["x_lo"].to_numpy(), sn["x_hi"].to_numpy()
    k = len(WX_LEADS)
    nb = 0 if model == "market" else X.shape[1]
    th0 = np.zeros(k + nb)
    if model in ("nbm", "nbm_only"):
        th0[k:k + k] = np.log(np.median(sig))
        if model == "nbm":
            th0[k:k + k] = 0.0
            th0[k + k] = 1.0
    res = minimize(lambda t: -_ll(t, X, lead_ix, F, sig, L, U, model).sum(), th0, method="L-BFGS-B")
    return {"model": model, "theta": res.x.tolist(), "names": [f"mu_{h}" for h in WX_LEADS] + (names if nb else []),
            "converged": bool(res.success), "dev_ll": float(-res.fun / len(F)), "n": len(F)}


def apply_sigma(sn: pl.DataFrame, fit: dict[str, Any]) -> np.ndarray:
    model = fit["model"]
    if model == "market":
        return sn["sigma_atm"].to_numpy()
    X, _ = _design(sn, model)
    beta = np.array(fit["theta"][len(WX_LEADS):])
    return sn["sigma_atm"].to_numpy() * np.exp(X @ beta) if model == "shrink" else np.exp(X @ beta)


def mean_ll(sn: pl.DataFrame, fit: dict[str, Any], sigma_f: np.ndarray, use_mu: bool = True) -> float:
    k = len(WX_LEADS)
    lead_ix = np.array([WX_LEADS.index(h) for h in sn["lead_h"].to_list()])
    mu = np.array(fit["theta"][:k])[lead_ix] if use_mu else 0.0
    m = sn["F"].to_numpy() + mu * sn["sigma_atm"].to_numpy()
    p = ndtr((sn["x_hi"].to_numpy() - m) / sigma_f) - ndtr((sn["x_lo"].to_numpy() - m) / sigma_f)
    return float(np.mean(np.log(np.clip(p, 1e-9, None))))


def step_panel(s) -> dict[str, Any]:
    sn, lg = build_panel(s)
    feat_ok = sn["xnd"].is_not_null() & sn["rmse"].is_not_null() & sn["F"].is_not_null()
    sn = sn.filter(feat_ok)
    dev = sn.filter(pl.col("split") == "dev")
    fits, table = {}, []
    for model in MODELS:
        fit = fit_sigma(dev, model)
        fits[model] = fit
        sf = apply_sigma(sn, fit)
        sn = sn.with_columns(pl.Series(f"sigma_{model}", sf))
        row = {"model": model, "converged": fit["converged"]}
        for sp in ("dev", "holdout"):
            m = (sn["split"] == sp).to_numpy()
            row[f"{sp}_ll"] = round(mean_ll(sn.filter(pl.col("split") == sp), fit, sf[m]), 4)
            row[f"{sp}_ll_mu0"] = round(mean_ll(sn.filter(pl.col("split") == sp), fit, sf[m], use_mu=False), 4)
        row.update({n: round(v, 3) for n, v in zip(fit["names"], fit["theta"])})
        table.append(row)
        print(row, flush=True)
    # per-lead shrink factor and NBM model ratio sigma_f / sigma_ATM
    sn = sn.with_columns(*(pl.col(f"sigma_{m}").truediv(pl.col("sigma_atm")).alias(f"r_{m}") for m in ("shrink", "nbm")))
    # calibration of the ratio: realized width vs r (dev and holdout), by r tercile
    sn.write_parquet(OUT / "wx_snapshots.parquet")
    lg.join(sn.select("snap_id"), on="snap_id").write_parquet(OUT / "wx_legs.parquet")
    summ = {"fits": fits, "ll_table": table, "snapshots": sn.height,
            "by_split": sn.group_by("split").agg(pl.len(), pl.col("event_ticker").n_unique().alias("events")).to_dicts()}
    (OUT / "wx_sigma.json").write_text(json.dumps(summ, indent=1, default=str))
    return summ


# ---- weather: leg-level fair values and trade construction ---------------------------------------
def wx_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Weather panel (dev + holdout) plus the second holdout (Nov 1-9) when it has been built."""
    sn, lg = pl.read_parquet(OUT / "wx_snapshots.parquet"), pl.read_parquet(OUT / "wx_legs.parquet")
    if (NOV / "wx_snapshots.parquet").exists():
        sn = pl.concat([sn, pl.read_parquet(NOV / "wx_snapshots.parquet")], how="diagonal_relaxed")
        lg = pl.concat([lg, pl.read_parquet(NOV / "wx_legs.parquet")], how="diagonal_relaxed")
    return sn, lg


def wx_ohlc() -> pl.DataFrame:
    o = pl.read_parquet(OUT / "wx_ohlc.parquet")
    if (NOV / "wx_ohlc.parquet").exists():
        o = pl.concat([o, pl.read_parquet(NOV / "wx_ohlc.parquet")]).unique(["ticker", "end_ts"], keep="first")
    return o


def leg_frame(sigma_col: str) -> pl.DataFrame:
    """Per (snapshot, leg): quotes, settlement, fee data, vol-only fair under sigma_f and the
    short-vol direction (sign of the fair change from sigma_ATM to sigma_f)."""
    sn, lg = wx_frames()
    x = lg.join(sn.select("snap_id", "event_ticker", "series_ticker", "snapshot_ts", "t_ref", "lead_h", "F", "sigma_atm",
                          "split", pl.col(sigma_col).alias("sigma_f")), on="snap_id")
    F, sf, sa = x["F"].to_numpy(), x["sigma_f"].to_numpy(), x["sigma_atm"].to_numpy()
    lo, hi = x["lo"].to_numpy(), x["hi"].to_numpy()
    fair = ndtr((hi - F) / sf) - ndtr((lo - F) / sf)
    impl = ndtr((hi - F) / sa) - ndtr((lo - F) / sa)
    return x.with_columns(pl.Series("fair", fair), pl.Series("fair_impl", impl),
                          pl.Series("dir", np.sign(fair - impl)), (pl.col("sigma_f") / pl.col("sigma_atm")).alias("r"),
                          pl.when((pl.col("lo") <= pl.col("F")) & (pl.col("F") < pl.col("hi"))).then(pl.lit("body"))
                          .otherwise(pl.lit("wing")).alias("pos"))


def taker_legs(x: pl.DataFrame) -> pl.DataFrame:
    """Both taker sides of every quoted leg: YES at the ask, NO at 1 - bid, fee per contract."""
    mult = x["fee_mult"].fill_null(1.0).to_numpy()
    ft = x["fee_type"].fill_null("quadratic").to_list()
    ask, bid = x["ask"].fill_null(np.nan).to_numpy(), x["bid"].fill_null(np.nan).to_numpy()
    fy = fees(np.where((ask >= 0.01) & (ask <= 0.99), ask, np.nan), mult, ft)
    fn = fees(np.where((bid >= 0.01) & (bid <= 0.99), 1 - bid, np.nan), mult, ft)
    y = x["y"].to_numpy().astype(float)
    fair = x["fair"].to_numpy()
    base = x.select("snap_id", "ticker", "event_ticker", "t_ref", "snapshot_ts", "lead_h", "split", "dir", "pos", "r",
                    "close_time", "fair", "bid", "ask", "y", "fee_type", "fee_mult")
    yes = base.with_columns(pl.lit("yes").alias("side"), pl.Series("price", ask), pl.Series("fee", fy),
                            pl.Series("edge", fair - ask - fy), pl.Series("pnl", y - ask - fy))
    no = base.with_columns(pl.lit("no").alias("side"), pl.Series("price", 1 - bid), pl.Series("fee", fn),
                           pl.Series("edge", bid - fair - fn), pl.Series("pnl", (1 - y) - (1 - bid) - fn))
    out = pl.concat([yes, no]).filter(pl.col("fee").is_not_nan() & pl.col("fee").is_not_null())
    return out.with_columns((pl.col("price") + pl.col("fee")).alias("cost"), pl.col("snapshot_ts").alias("t_dec"),
                            pl.col("t_ref").alias("t_end"))


def short_vol_select(rstar: float, structure: str, leads: tuple[int, ...]) -> pl.Expr:
    side_ok = (((pl.col("side") == "yes") & (pl.col("dir") > 0)) if structure == "body" else
               ((pl.col("side") == "no") & (pl.col("dir") < 0)) if structure == "wings" else
               (((pl.col("side") == "yes") & (pl.col("dir") > 0)) | ((pl.col("side") == "no") & (pl.col("dir") < 0))))
    return side_ok & (pl.col("r") <= rstar) & pl.col("lead_h").is_in(list(leads))


TAKER_GRID = [{"leads": lg, "rstar": r, "structure": st} for lg in LEAD_GROUPS for r in (0.95, 0.85)
              for st in ("body", "fly")]


def step_wx_taker(sigma_model: str) -> dict[str, Any]:
    x = leg_frame(f"sigma_{sigma_model}")
    tl = taker_legs(x)
    weeks = weeks_of(x)
    parts, variants = [], []
    for v in TAKER_GRID:
        name = f"{v['leads']}|r<={v['rstar']}|{v['structure']}"
        sel = tl.filter(short_vol_select(v["rstar"], v["structure"], LEAD_GROUPS[v["leads"]]) & (pl.col("edge") > 0))
        parts.append(sel.with_columns(pl.lit(name).alias("variant")))
        variants.append({"variant": name, **v})
    trades = pl.concat(parts)
    _, chosen = run_grid(f"wx_taker_{sigma_model}", trades, variants, weeks)
    if "variant" in chosen:
        best = trades.filter(pl.col("variant") == chosen["variant"])
        chosen["by_pos_side"] = (best.group_by("split", "pos", "side").agg(
            pl.len().alias("n"), (pl.col("pnl").mean() * 100).round(2).alias("c"),
            (pl.col("price").mean() * 100).round(1).alias("price_c")).sort("split", "pos", "side").to_dicts())
    # unconditional controls: every body YES / every wing NO at each lead (no forecast filter)
    ctrl = []
    for lead in WX_LEADS:
        for nm, cond in (("body_yes_all", (pl.col("pos") == "body") & (pl.col("side") == "yes")),
                         ("wing_no_all", (pl.col("pos") == "wing") & (pl.col("side") == "no"))):
            g = tl.filter(cond & (pl.col("lead_h") == lead))
            row = {"control": nm, "lead_h": lead}
            for sp in SPLITS:
                row.update({f"{sp}_{k}": v for k, v in stat(g.filter(pl.col("split") == sp)).items()})
            ctrl.append(row)
    chosen["controls"] = ctrl
    chosen["ledgers"] = {v: write_ledger(f"wx_taker_{sigma_model}", v, taker_ledger_legs(trades.filter(pl.col("variant") == v)))
                         for v in [chosen.get("variant"), *FLAGGED.get(f"wx_taker_{sigma_model}", [])] if v}
    return chosen


def won_expr() -> pl.Expr:
    return pl.when(pl.col("side") == "yes").then(pl.col("y") == 1).otherwise(pl.col("y") == 0).alias("won")


def taker_ledger_legs(t: pl.DataFrame, struct: str = "snap_id") -> pl.DataFrame:
    """Taker trades -> ``book`` legs: filled at the decision time, one structure per snapshot."""
    return t.select(pl.col(struct).cast(pl.String).alias("struct_id"), "event_ticker", "ticker", "side", "price",
                    pl.lit(False).alias("maker"), "fee_type", "fee_mult", pl.col("snapshot_ts").alias("t_post"),
                    pl.col("snapshot_ts").alias("t_fill"), pl.col("snapshot_ts").alias("t_cancel"), won_expr(), "split")


# variants flagged for a forward paper test (ledgers written next to the dev-chosen one)
FLAGGED = {"wx_taker_shrink": ["late 8|r<=0.95|body"], "wx_taker_nbm": ["pre-day 36/24|r<=0.85|fly"],
           "wx_maker_shrink": ["late 8|improve|r<=0.95"]}


# ---- weather: maker -------------------------------------------------------------------------------
MAKER_GRID = [{"leads": lg, "price": pr, "rstar": r} for lg in LEAD_GROUPS for pr in ("join", "improve")
              for r in (0.95, 0.85)]
MAKER_EDGE = 0.01


def maker_orders(x: pl.DataFrame, price_rule: str) -> pl.DataFrame:
    """One resting order per short-vol leg: a YES bid on legs whose value rises when the
    distribution narrows (dir > 0), a NO bid on the others. ``lim`` is the YES-price limit (bid
    for YES orders, YES-equivalent ask for NO orders); the order lives until the next lead."""
    imp = price_rule == "improve"
    spr = pl.col("ask") - pl.col("bid")
    o = x.filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("ask") > pl.col("bid")))
    o = o.with_columns(
        pl.when(pl.col("dir") > 0).then(pl.lit("yes")).otherwise(pl.lit("no")).alias("side"),
        pl.when(pl.col("dir") > 0)
        .then(pl.col("bid") + pl.when(spr >= 0.02 - EPS).then(0.01 if imp else 0.0).otherwise(0.0))
        .otherwise(pl.col("ask") - pl.when(spr >= 0.02 - EPS).then(0.01 if imp else 0.0).otherwise(0.0))
        .round(2).alias("lim"),
        (pl.col("snapshot_ts") + pl.duration(hours=pl.col("lead_h").replace_strict(
            {k: k - v for k, v in NEXT_LEAD.items()}, return_dtype=pl.Int64))).alias("t_cancel"))
    o = o.with_columns(pl.min_horizontal("t_cancel", "close_time").alias("t_cancel"))
    o = o.with_columns(pl.when(pl.col("side") == "yes").then(pl.col("fair") - pl.col("lim"))
                       .otherwise(pl.col("lim") - pl.col("fair")).alias("edge_m"),
                       pl.when(pl.col("side") == "yes").then(pl.col("lim")).otherwise(1 - pl.col("lim")).alias("price"))
    return o.filter((pl.col("lim") >= 0.01) & (pl.col("lim") <= 0.99) & (pl.col("dir") != 0))


def maker_fills(o: pl.DataFrame, ohlc: pl.DataFrame) -> pl.DataFrame:
    """First hourly candle in (t_snap, t_cancel] that printed strictly through the limit."""
    oid = o.with_row_index("oid").select("oid", "ticker", "side", "lim", "snapshot_ts", "t_cancel")
    c = ohlc.select("ticker", "end_ts", "p_low", "p_high", "volume")
    # a candle ending at e covers (e - 1 h, e]: those ending in (t_snap, t_cancel] cover the order life
    j = oid.join(c, on="ticker").filter((pl.col("end_ts") > pl.col("snapshot_ts"))
                                        & (pl.col("end_ts") <= pl.col("t_cancel")))
    hit = j.filter(pl.when(pl.col("side") == "yes").then(pl.col("p_low") < pl.col("lim") - EPS)
                   .otherwise(pl.col("p_high") > pl.col("lim") + EPS))
    first = hit.sort("end_ts").group_by("oid").agg(pl.col("end_ts").first().alias("fill_ts"),
                                                   pl.col("volume").first().alias("fill_hour_volume"))
    covered = j.group_by("oid").agg(pl.len().alias("hours_seen"))
    return (o.with_row_index("oid").join(first, on="oid", how="left").join(covered, on="oid", how="left")
            .with_columns(pl.col("fill_ts").is_not_null().alias("filled")))


def markouts(f: pl.DataFrame, ohlc: pl.DataFrame, hours: tuple[int, ...] = (1, 4)) -> pl.DataFrame:
    """Mid at fill + h (last candle close at or before) minus the fill price, from our side."""
    q = ohlc.filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null()).select(
        "ticker", "end_ts", ((pl.col("bid") + pl.col("ask")) / 2).alias("mid")).sort("end_ts")
    out = f
    for h in hours:
        t = f.with_columns((pl.col("fill_ts") + pl.duration(hours=h)).alias("_t")).sort("_t")
        t = t.join_asof(q.rename({"end_ts": "_q"}), left_on="_t", right_on="_q", by="ticker", strategy="backward")
        out = out.join(t.select("oid", pl.col("mid").alias(f"mid_{h}h")), on="oid", how="left")
    sgn = pl.when(pl.col("side") == "yes").then(1.0).otherwise(-1.0)
    mid0 = (pl.col("bid") + pl.col("ask")) / 2
    return out.with_columns((sgn * (mid0 - pl.col("lim"))).alias("mk_post"),
                            *((sgn * (pl.col(f"mid_{h}h") - pl.col("lim"))).alias(f"mk_{h}h") for h in hours),
                            (sgn * (pl.col("y") - pl.col("lim"))).alias("mk_settle"))


def maker_ledger_legs(o: pl.DataFrame) -> pl.DataFrame:
    """Maker orders (filled or not) -> ``book`` legs: collateral reserved from posting until the
    fill (then held to settlement) or the cancel."""
    return o.select(pl.col("snap_id").cast(pl.String).alias("struct_id"), "event_ticker", "ticker", "side", "price",
                    pl.lit(True).alias("maker"), "fee_type", "fee_mult", pl.col("snapshot_ts").alias("t_post"),
                    pl.when(pl.col("filled")).then(pl.col("fill_ts")).otherwise(None).alias("t_fill"), "t_cancel",
                    won_expr(), "split")


def step_wx_maker(sigma_model: str) -> dict[str, Any]:
    x = leg_frame(f"sigma_{sigma_model}")
    ohlc = wx_ohlc()
    weeks = weeks_of(x)
    parts, variants, fill_tabs = [], [], []
    for pr in ("join", "improve"):
        o = maker_orders(x, pr)
        f = maker_fills(o, ohlc)
        mult = f["fee_mult"].fill_null(1.0).to_numpy()
        ft = f["fee_type"].fill_null("quadratic").to_list()
        fe = fees(f["price"].to_numpy(), mult, ft, maker=True)
        f = f.with_columns(pl.Series("fee", fe)).with_columns(
            (pl.col("price") + pl.col("fee")).alias("cost"),
            pl.when(pl.col("side") == "yes").then(pl.col("y") - pl.col("price"))
            .otherwise((1 - pl.col("y")) - pl.col("price")).sub(pl.col("fee")).alias("pnl"),
            pl.col("fill_ts").alias("t_dec"), pl.col("t_ref").alias("t_end"))
        f = pl.concat([markouts(f.filter("filled"), ohlc), f.filter(~pl.col("filled"))], how="diagonal_relaxed")
        for v in [m for m in MAKER_GRID if m["price"] == pr]:
            name = f"{v['leads']}|{pr}|r<={v['rstar']}"
            sel = f.filter(pl.col("lead_h").is_in(list(LEAD_GROUPS[v["leads"]])) & (pl.col("r") <= v["rstar"])
                           & (pl.col("edge_m") - pl.col("fee") >= MAKER_EDGE))
            fill_tabs.append(sel.with_columns(pl.lit(name).alias("variant")))
            parts.append(sel.filter("filled").with_columns(pl.lit(name).alias("variant")))
            variants.append({"variant": name, **v})
    trades = pl.concat(parts)
    orders = pl.concat(fill_tabs)
    grid, chosen = run_grid(f"wx_maker_{sigma_model}", trades, variants, weeks)
    fr = orders.group_by("variant", "split").agg(pl.len().alias("orders"), pl.col("filled").mean().round(3).alias("fill_rate"),
                                                 pl.col("hours_seen").is_null().mean().round(3).alias("no_candles"))
    grid = grid.join(fr.pivot("split", index="variant", values=["orders", "fill_rate"]), on="variant", how="left")
    grid.write_parquet(OUT / f"grid_wx_maker_{sigma_model}.parquet")
    chosen["ledgers"] = {v: write_ledger(f"wx_maker_{sigma_model}", v, maker_ledger_legs(orders.filter(pl.col("variant") == v)))
                         for v in [chosen.get("variant"), *FLAGGED.get(f"wx_maker_{sigma_model}", [])] if v}
    if "variant" in chosen:
        v = chosen["variant"]
        best = trades.filter(pl.col("variant") == v)
        chosen["fill_rate"] = fr.filter(pl.col("variant") == v).to_dicts()
        chosen["markouts_c"] = (best.group_by("split", "pos", "side").agg(
            pl.len().alias("fills"), *((pl.col(c).mean() * 100).round(2).alias(c) for c in
                                       ("mk_post", "mk_1h", "mk_4h", "mk_settle", "pnl")),
            (pl.col("fill_hour_volume") >= 10).mean().round(3).alias("share_hour_vol_ge10"))
            .sort("split", "pos", "side").to_dicts())
        best.write_parquet(OUT / f"wx_maker_fills_{sigma_model}.parquet")
        chosen["fill_check"] = fill_check(orders.filter(pl.col("variant") == v), ohlc)
    return chosen


def fill_check(orders: pl.DataFrame, ohlc: pl.DataFrame) -> dict[str, Any]:
    """Candle fills against exact trade fills on the orders whose life lies inside a recorded
    kalshi_trades window (strictly through: a YES-seller below our bid / YES-buyer above our ask)."""
    from albert.config import settings

    s = settings()
    w = pl.read_parquet(s.tables_dir / "kalshi_trade_windows.parquet")
    o = orders.join(w, on="ticker").filter((pl.col("window_start") <= pl.col("snapshot_ts"))
                                           & (pl.col("window_end") >= pl.col("t_cancel")))
    if o.is_empty():
        return {"orders": 0}
    tk = o["ticker"].unique().implode()
    tr = (pl.scan_parquet(s.tables_dir / "kalshi_trades.parquet").filter(pl.col("ticker").is_in(tk))
          .select("ticker", "created_time", "yes_price").collect())
    j = o.select("oid", "ticker", "side", "lim", "snapshot_ts", "t_cancel", "filled").join(tr, on="ticker").filter(
        (pl.col("created_time") > pl.col("snapshot_ts")) & (pl.col("created_time") <= pl.col("t_cancel")))
    hit = j.filter(pl.when(pl.col("side") == "yes").then(pl.col("yes_price") < pl.col("lim") - EPS)
                   .otherwise(pl.col("yes_price") > pl.col("lim") + EPS))["oid"].unique()
    o = o.unique("oid").with_columns(pl.col("oid").is_in(hit.implode()).alias("trade_filled"))
    return {"orders": o.height, "agree": round(float((o["filled"] == o["trade_filled"]).mean()), 4),
            "candle_fills": int(o["filled"].sum()), "trade_fills": int(o["trade_filled"].sum())}


TRADE_SCHEMA = {"trade_id": pl.String, "ticker": pl.String, "created_time": pl.Datetime("us", "UTC"),
                "count": pl.Float64, "yes_price": pl.Float64, "taker_side": pl.String}


def fetch_trades(s, windows: pl.DataFrame, path: Path, *, min_interval: float = 0.25) -> pl.DataFrame:
    """Every public trade of each (ticker, t_from, t_to) window through the repo's cached client
    (raw pages append-only under data/raw/kalshi, <= 4 requests/s, one worker). Rows are appended
    to ``path``; windows already fetched are skipped."""
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    done_path = path.with_name(path.stem + "_windows.parquet")
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=TRADE_SCHEMA)
    done = pl.read_parquet(done_path) if done_path.exists() else windows.head(0).select("ticker", "t_from", "t_to")
    todo = windows.select("ticker", "t_from", "t_to").unique().join(done, on=["ticker", "t_from", "t_to"], how="anti")
    if todo.is_empty():
        return have
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=min_interval, max_retries=8))
    cut = api.cutoff()
    cutoff = cut.get("trades_created_ts") or cut.get("market_settled_ts")
    rows: list[dict[str, Any]] = []
    fin: list[dict[str, Any]] = []
    for i, r in enumerate(todo.sort("t_to", descending=True).iter_rows(named=True), 1):
        hist = bool(cutoff and r["t_to"] < cutoff)
        try:
            for page in api.trades(r["ticker"], hist, min_ts=r["t_from"], max_ts=r["t_to"]):
                rows.extend(k.trade_rows(page.payload))
            fin.append(r)
        except Exception as error:  # noqa: BLE001  (a failed window is retried on the next run)
            print(f"skip {r['ticker']}: {str(error)[:120]}", flush=True)
        if i % 250 == 0 or i == todo.height:
            have = pl.concat([have, pl.DataFrame(rows, schema=TRADE_SCHEMA)]).unique("trade_id")
            done = pl.concat([done, pl.DataFrame(fin, schema=done.schema)])
            rows, fin = [], []
            have.write_parquet(path)
            done.write_parquet(done_path)
            print(f"  trades {i}/{todo.height} windows, {have.height} trades {datetime.now(UTC):%H:%M:%S}", flush=True)
    return have


def late_orders() -> pl.DataFrame:
    """Every late (8 h lead) maker order of both price rules (same orders for both sigma models)."""
    x = leg_frame("sigma_shrink")
    return pl.concat([maker_orders(x, pr).with_columns(pl.lit(pr).alias("price_rule")) for pr in ("join", "improve")]
                     ).filter(pl.col("lead_h") == 8)


def step_wx_trades(s) -> pl.DataFrame:
    o = late_orders()
    w = o.select("ticker", pl.col("snapshot_ts").alias("t_from"), pl.col("t_cancel").alias("t_to")).unique()
    print(f"late orders {o.height}, windows {w.height}", flush=True)
    return fetch_trades(s, w, OUT / "wx_trades_late.parquet")


def exact_fills(o: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    """Trade-level fills: first print strictly through the limit in (t_snap, t_cancel] and the
    contracts printed strictly through over the order life (an upper bound on our fill size)."""
    oid = o.select("oid", "ticker", "side", "lim", "snapshot_ts", "t_cancel")
    j = oid.join(trades.select("ticker", "created_time", "yes_price", "count"), on="ticker").filter(
        (pl.col("created_time") > pl.col("snapshot_ts")) & (pl.col("created_time") <= pl.col("t_cancel")))
    hit = j.filter(pl.when(pl.col("side") == "yes").then(pl.col("yes_price") < pl.col("lim") - EPS)
                   .otherwise(pl.col("yes_price") > pl.col("lim") + EPS))
    agg = hit.sort("created_time").group_by("oid").agg(pl.col("created_time").first().alias("fill_ts_x"),
                                                       pl.col("count").sum().alias("through_qty"))
    return o.join(agg, on="oid", how="left").with_columns(pl.col("fill_ts_x").is_not_null().alias("filled_x"))


def qty_usd_week(g: pl.DataFrame, weeks: float) -> float | None:
    """$/week when each filled order gets min(ORDER_N, contracts printed through our price) and an
    event's collateral is capped at EVENT_CAP (never scaled up)."""
    if g.is_empty() or weeks <= 0:
        return None
    e = (g.with_columns(pl.min_horizontal(pl.lit(float(ORDER_N)), pl.col("through_qty")).alias("q"))
         .group_by("event_ticker").agg((pl.col("q") * pl.col("pnl")).sum().alias("pnl"),
                                       (pl.col("q") * pl.col("cost")).sum().alias("cost"))
         .with_columns((pl.col("pnl") * pl.min_horizontal(pl.lit(1.0), EVENT_CAP / pl.col("cost"))).alias("usd")))
    return round(float(e["usd"].sum()) / weeks, 2)


def adverse(o: pl.DataFrame) -> dict[str, Any] | None:
    """Adverse selection in cents: settlement value minus limit (our side) for filled orders vs
    all posted orders at the same limits."""
    if o.is_empty():
        return None
    v = pl.when(pl.col("side") == "yes").then(pl.col("y") - pl.col("lim")).otherwise(pl.col("lim") - pl.col("y"))
    a = o.select((v.mean() * 100).alias("all"), (v.filter(pl.col("filled_x")).mean() * 100).alias("filled"),
                 (v.filter(~pl.col("filled_x")).mean() * 100).alias("unfilled")).row(0, named=True)
    return {k: None if x is None else round(x, 2) for k, x in a.items()}


def step_wx_late_exact(sigma_model: str = "shrink") -> dict[str, Any]:
    """The late maker grid rows re-scored with exact trade fills, and the candle-vs-trade agreement."""
    path = OUT / "wx_trades_late.parquet"
    tr = pl.read_parquet(path)
    done = pl.read_parquet(path.with_name(path.stem + "_windows.parquet"))
    x = leg_frame(f"sigma_{sigma_model}")
    ohlc = wx_ohlc()
    weeks = weeks_of(x)
    out: dict[str, Any] = {}
    for pr in ("join", "improve"):
        o = maker_orders(x, pr).filter(pl.col("lead_h") == 8)
        o = o.join(done.rename({"t_from": "snapshot_ts", "t_to": "t_cancel"}), on=["ticker", "snapshot_ts", "t_cancel"])
        f = maker_fills(o, ohlc)
        f = exact_fills(f, tr)
        mult = f["fee_mult"].fill_null(1.0).to_numpy()
        fe = fees(f["price"].to_numpy(), mult, f["fee_type"].fill_null("quadratic").to_list(), maker=True)
        f = f.with_columns(pl.Series("fee", fe)).with_columns(
            (pl.col("price") + pl.col("fee")).alias("cost"),
            pl.when(pl.col("side") == "yes").then(pl.col("y") - pl.col("price"))
            .otherwise((1 - pl.col("y")) - pl.col("price")).sub(pl.col("fee")).alias("pnl"),
            pl.col("fill_ts_x").alias("t_dec"), pl.col("t_ref").alias("t_end"))
        sel = f.filter(pl.col("edge_m") - pl.col("fee") >= MAKER_EDGE)
        filled = sel.filter("filled_x").with_columns(pl.col("fill_ts_x").alias("fill_ts"))
        filled = markouts(filled, ohlc)
        row: dict[str, Any] = {"orders_with_trades": sel.height,
                               "agree_candle_vs_trade": round(float((sel["filled"] == sel["filled_x"]).mean()), 4),
                               "candle_fills": int(sel["filled"].sum()), "trade_fills": int(sel["filled_x"].sum())}
        for sp in SPLITS:
            g = filled.filter(pl.col("split") == sp)
            if sel.filter(pl.col("split") == sp).is_empty():
                continue
            row[sp] = {**stat(g), "fill_rate": round(float(sel.filter(pl.col("split") == sp)["filled_x"].mean()), 4),
                       "usd_week": at_caps(g, weeks[sp])[0],
                       "usd_week_through_qty": qty_usd_week(g, weeks[sp]),
                       "adverse_c": adverse(sel.filter(pl.col("split") == sp)),
                       "median_through_qty": float(g["through_qty"].median()) if g.height else None,
                       "share_through_ge5": round(float((g["through_qty"] >= 5).mean()), 3) if g.height else None,
                       "markouts_c": {c: None if g[c].mean() is None else round(float(g[c].mean()) * 100, 2)
                                      for c in ("mk_post", "mk_1h", "mk_4h", "mk_settle")} if g.height else None}
        row["months"] = monthly(filled).to_dicts()
        row["by_side"] = (filled.group_by("split", "side").agg(pl.len().alias("fills"), (pl.col("pnl").mean() * 100).round(2).alias("c"),
                                                              (pl.col("price").mean() * 100).round(1).alias("price_c"))
                          .sort("split", "side").to_dicts())
        out[pr] = row
        filled.write_parquet(OUT / f"wx_late_exact_fills_{pr}.parquet")
    return out


# ---- weather: median drift ------------------------------------------------------------------------
def step_wx_drift() -> dict[str, Any]:
    """Lean on the dev-sign drift of F between consecutive leads: buy the bracket adjacent to the
    one containing F on the drift side at the ask; exit at the next lead's bid or at settlement."""
    sn, lg = wx_frames()
    pairs = list(itertools.pairwise(WX_LEADS))
    drift = {}
    for a, b in pairs:
        j = (sn.filter((pl.col("lead_h") == a) & (pl.col("split") == "dev"))
             .join(sn.filter(pl.col("lead_h") == b).select("ladder", pl.col("F").alias("Fb")), on="ladder"))
        drift[f"{a}->{b}"] = float(((j["Fb"] - j["F"]) / j["sigma_atm"]).mean())
    x = lg.join(sn.select("snap_id", "ladder", "event_ticker", "t_ref", "snapshot_ts", "lead_h", "F", "split"), on="snap_id")
    rows, variants = [], []
    weeks = weeks_of(sn)
    for a, b in pairs:
        d = 1 if drift[f"{a}->{b}"] > 0 else -1
        e = x.filter(pl.col("lead_h") == a)
        # adjacent bracket on the drift side: smallest lo >= F (up) / largest hi <= F (down)
        e = e.filter((pl.col("lo") >= pl.col("F")) if d > 0 else (pl.col("hi") <= pl.col("F")))
        e = e.sort("lo", descending=d < 0).group_by("snap_id").first()
        e = e.filter(pl.col("ask").is_between(0.01, 0.99))
        nxt = x.filter(pl.col("lead_h") == b).select("ladder", "ticker", pl.col("bid").alias("bid_next"),
                                                     pl.col("snapshot_ts").alias("t_next"))
        e = e.join(nxt, on=["ladder", "ticker"], how="left")
        mult = e["fee_mult"].fill_null(1.0).to_numpy()
        ft = e["fee_type"].fill_null("quadratic").to_list()
        fin = fees(e["ask"].to_numpy(), mult, ft)
        bn = e["bid_next"].fill_null(np.nan).to_numpy()
        fout = fees(np.where(bn >= 0.01, bn, np.nan), mult, ft)
        e = e.with_columns(pl.Series("fee", fin), (pl.col("ask")).alias("price"))
        for exit_ in ("next_lead", "settle"):
            name = f"{a}->{b}|{'up' if d > 0 else 'down'}|{exit_}"
            if exit_ == "settle":
                t = e.with_columns((pl.col("y") - pl.col("ask") - pl.col("fee")).alias("pnl"), pl.col("t_ref").alias("t_end"))
            else:                 # no bid at the next lead: the leg is held to settlement
                sell = pl.col("fee_out").is_not_nan() & pl.col("fee_out").is_not_null()
                t = e.with_columns(pl.Series("fee_out", fout)).with_columns(
                    pl.when(sell).then(pl.col("bid_next") - pl.col("fee_out")).otherwise(pl.col("y"))
                    .sub(pl.col("ask") + pl.col("fee")).alias("pnl"),
                    pl.when(sell).then(pl.col("t_next")).otherwise(pl.col("t_ref")).alias("t_end"))
            rows.append(t.with_columns(pl.lit(name).alias("variant"), (pl.col("ask") + pl.col("fee")).alias("cost"),
                                       pl.col("snapshot_ts").alias("t_dec"),
                                       pl.lit(exit_ == "next_lead").alias("exits"))
                        .select("variant", "event_ticker", "split", "t_ref", "t_dec", "t_end", "pnl", "cost", "snap_id",
                                "ticker", "ask", "y", "fee_type", "fee_mult", "snapshot_ts", "t_next", "bid_next", "exits"))
            variants.append({"variant": name, "pair": f"{a}->{b}", "dev_drift_sigma": round(drift[f"{a}->{b}"], 3),
                             "exit": exit_})
    trades = pl.concat(rows, how="diagonal_relaxed")
    _, chosen = run_grid("wx_drift", trades, variants, weeks)
    if "variant" in chosen:
        t = trades.filter(pl.col("variant") == chosen["variant"]).with_columns(
            pl.lit("yes").alias("side"), pl.col("ask").alias("price"))
        legs = taker_ledger_legs(t).with_columns(
            pl.when(t["exits"] & (t["bid_next"] >= 0.01)).then(t["t_next"]).otherwise(None).alias("exit_ts"),
            pl.when(t["exits"] & (t["bid_next"] >= 0.01)).then(t["bid_next"]).otherwise(None).alias("exit_price"))
        chosen["ledgers"] = {chosen["variant"]: write_ledger("wx_drift", chosen["variant"], legs)}
    hold = {}
    for a, b in pairs:
        j = (sn.filter(pl.col("lead_h") == a)
             .join(sn.filter(pl.col("lead_h") == b).select("ladder", pl.col("F").alias("Fb")), on="ladder"))
        for (sp,), g in j.group_by("split"):
            mu, t, n, _ = cmean(((g["Fb"] - g["F"]) / g["sigma_atm"]).to_numpy(), g["event_ticker"].to_numpy())
            hold[f"{a}->{b}|{sp}"] = {"drift_sigma": round(mu, 3), "t": round(t, 2), "n": n}
    chosen["drift_by_split"] = hold
    return chosen


# ---- sports: moneyline vs spread ladder ----------------------------------------------------------
LEAGUES = ("NFL", "NBA", "MLB", "NHL", "NCAAF", "WNBA")


def s_at_zero(pt: pl.DataFrame) -> pl.DataFrame:
    """Per snapshot S(0) from the isotonic survival points: z linear between the strikes straddling 0."""
    rows = []
    for (sid,), g in pt.sort("snap_id", "K").group_by("snap_id", maintain_order=True):
        K, S = g["K"].to_numpy(), g["S"].to_numpy()
        z = ndtri(np.clip(S, 0.005, 0.995))
        i = np.flatnonzero((K[:-1] < 0) & (K[1:] > 0))
        if i.size == 0:
            continue
        i = int(i[0])
        w = (0 - K[i]) / (K[i + 1] - K[i])
        rows.append({"snap_id": sid, "S0": float(ndtr(z[i] + w * (z[i + 1] - z[i]))),
                     "K_lo": float(K[i]), "K_hi": float(K[i + 1])})
    return pl.DataFrame(rows)


def ml_panel(s) -> pl.DataFrame:
    """Spread-ladder snapshots joined with the favourite's moneyline quote at the snapshot."""
    from albert.research.pm_smile import load_candles, parse_leg

    sn = pl.read_parquet(SMILE / "snapshots.parquet").filter(
        (pl.col("group") == "sp_spread") & pl.col("series_ticker").str.contains(rf"^KX({'|'.join(LEAGUES)})SPREAD$"))
    pt = pl.read_parquet(SMILE / "points.parquet", columns=["snap_id", "K", "S"]).join(sn.select("snap_id"), on="snap_id")
    sn = sn.join(s_at_zero(pt), on="snap_id")
    mk = pl.read_parquet(s.tables_dir / "kalshi_markets.parquet",
                         columns=["ticker", "event_ticker", "series_ticker", "yes_sub_title", "result", "fee_type",
                                  "fee_multiplier", "close_time"])
    sp = mk.filter(pl.col("event_ticker").is_in(sn["event_ticker"].unique().implode()))
    names = {}
    for ev, sub in zip(sp["event_ticker"], sp["yes_sub_title"]):
        p = parse_leg(sub, "sp_spread")
        if p and p[0] == "spread":
            names.setdefault(ev, set()).add(p[3])
    sn = sn.with_columns(pl.col("event_ticker").map_elements(lambda e: "|".join(sorted(names.get(e, set()))),
                                                             return_dtype=pl.String).alias("teams"))
    sn = sn.filter(pl.col("teams").str.count_matches(r"\|") == 1).with_columns(
        pl.when(pl.col("orient") == 1).then(pl.col("teams").str.split("|").list.get(0))
        .otherwise(pl.col("teams").str.split("|").list.get(1)).alias("fav"),
        pl.col("event_ticker").str.replace("SPREAD-", "GAME-").alias("ml_event"))
    ml = mk.filter(pl.col("series_ticker").str.contains(rf"^KX({'|'.join(LEAGUES)})GAME$")
                   & pl.col("result").is_in(["yes", "no"]))
    sn = sn.join(ml.select(pl.col("event_ticker").alias("ml_event"), pl.col("yes_sub_title").alias("fav"),
                           pl.col("ticker").alias("ml_ticker"), pl.col("result").alias("ml_result"),
                           pl.col("fee_type").alias("ml_fee_type"), pl.col("fee_multiplier").alias("ml_fee_mult"),
                           pl.col("close_time").alias("ml_close")),
                 on=["ml_event", "fav"])
    c = load_candles(s.data_dir, sn["ml_ticker"].unique().to_list())
    q = sn.sort("snapshot_ts").join_asof(c.rename({"end_ts": "q_ts", "ticker": "ml_ticker"}).sort("q_ts"),
                                         left_on="snapshot_ts", right_on="q_ts", by="ml_ticker",
                                         strategy="backward", tolerance=timedelta(hours=2))
    q = q.filter(pl.col("q_ts").is_not_null() & (pl.col("ml_close") > pl.col("snapshot_ts"))
                 & (pl.col("ask") > pl.col("bid")) & (pl.col("ask") - pl.col("bid") <= 0.05 + EPS))
    return q.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("ml_mid"),
                          (pl.col("ml_result") == "yes").cast(pl.Float64).alias("y"),
                          pl.col("series_ticker").str.extract(r"^KX([A-Z]+)SPREAD$").alias("league"))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def step_sp_ml(s) -> dict[str, Any]:
    from albert.research.fundamental import fit_logistic

    q = ml_panel(s)
    q = q.filter(pl.col("lead_h").is_in([48, 24, 6]))
    dev = q.filter(pl.col("split") == "dev")

    def X(d: pl.DataFrame) -> np.ndarray:
        return np.column_stack([np.ones(d.height), _logit(d["ml_mid"].to_numpy()), _logit(d["S0"].to_numpy())])

    beta, tb = fit_logistic(X(dev), dev["y"].to_numpy(), dev["event_ticker"].to_numpy())
    q = q.with_columns(pl.Series("blend", 1 / (1 + np.exp(-X(q) @ beta))))
    diag = {"blend_beta": [round(float(b), 3) for b in beta], "blend_t": [round(float(t), 2) for t in tb],
            "n_dev": dev.height, "n_hold": q.height - dev.height}
    for sp in ("dev", "holdout"):
        g = q.filter(pl.col("split") == sp)
        y = g["y"].to_numpy()
        for nm in ("ml_mid", "S0", "blend"):
            p = np.clip(g[nm].to_numpy(), 0.01, 0.99)
            diag[f"{sp}_brier_{nm}"] = round(float(np.mean((p - y) ** 2)), 5)
        diag[f"{sp}_mean_gap_c"] = round(float((g["S0"] - g["ml_mid"]).mean()) * 100, 2)
        diag[f"{sp}_sd_gap_c"] = round(float((g["S0"] - g["ml_mid"]).std()) * 100, 2)
    mult = q["ml_fee_mult"].fill_null(1.0).to_numpy()
    ft = q["ml_fee_type"].fill_null("quadratic").to_list()
    ask, bid = q["ask"].to_numpy(), q["bid"].to_numpy()
    fy = fees(np.where((ask >= 0.01) & (ask <= 0.99), ask, np.nan), mult, ft)
    fn = fees(np.where((bid >= 0.01) & (bid <= 0.99), 1 - bid, np.nan), mult, ft)
    q = q.with_columns(pl.Series("fee_yes", fy), pl.Series("fee_no", fn), pl.col("snapshot_ts").alias("t_dec"),
                       pl.col("t_ref").alias("t_end"))
    parts, variants = [], []
    weeks = weeks_of(q)
    for fair in ("S0", "blend"):
        for th in (0.0, 0.02, 0.04):
            for lead in ("first", 6):
                name = f"{fair}|theta={th}|lead={lead}"
                g = q if lead == "first" else q.filter(pl.col("lead_h") == lead)
                e_y = g[fair] - g["ask"] - g["fee_yes"]
                e_n = g["bid"] - g[fair] - g["fee_no"]
                g = g.with_columns(e_y.alias("e_y"), e_n.alias("e_n"))
                yes = g.filter((pl.col("e_y") > th) & (pl.col("e_y") >= pl.col("e_n"))).with_columns(
                    (pl.col("ask") + pl.col("fee_yes")).alias("cost"), (pl.col("y") - pl.col("ask") - pl.col("fee_yes")).alias("pnl"),
                    pl.lit("yes").alias("side"), pl.col("ask").alias("price"))
                no = g.filter((pl.col("e_n") > th) & (pl.col("e_n") > pl.col("e_y"))).with_columns(
                    (1 - pl.col("bid") + pl.col("fee_no")).alias("cost"),
                    ((1 - pl.col("y")) - (1 - pl.col("bid")) - pl.col("fee_no")).alias("pnl"),
                    pl.lit("no").alias("side"), (1 - pl.col("bid")).alias("price"))
                t = pl.concat([yes, no]).filter(pl.col("cost").is_not_nan())
                if lead == "first":       # one trade per game: the earliest qualifying lead
                    t = t.sort("lead_h", descending=True).unique("event_ticker", keep="first")
                parts.append(t.select("event_ticker", "split", "t_ref", "t_dec", "t_end", "pnl", "cost", "league",
                                      pl.col("ml_ticker").alias("ticker"), "side", "price", "y", "snapshot_ts",
                                      pl.col("ml_fee_type").alias("fee_type"), pl.col("ml_fee_mult").alias("fee_mult"),
                                      pl.format("{}|{}", "event_ticker", "lead_h").alias("snap_id"))
                             .with_columns(pl.lit(name).alias("variant")))
                variants.append({"variant": name, "fair": fair, "theta": th, "lead": str(lead)})
    trades = pl.concat(parts)
    _, chosen = run_grid("sp_ml", trades, variants, weeks)
    chosen["diag"] = diag
    if "variant" in chosen:
        chosen["ledgers"] = {chosen["variant"]: write_ledger(
            "sp_ml", chosen["variant"], taker_ledger_legs(trades.filter(pl.col("variant") == chosen["variant"])))}
        best = trades.filter(pl.col("variant") == chosen["variant"])
        chosen["by_league"] = [{"league": lg_, "split": sp, **stat(g)} for (lg_, sp), g in
                               best.group_by("league", "split", maintain_order=True)]
    return chosen


# ---- sports: totals over bias --------------------------------------------------------------------
def step_sp_totals() -> dict[str, Any]:
    """Buy "Over K" legs whose strike is within |m| <= m* of the implied median at the ask."""
    lg = pl.read_parquet(SMILE / "legs.parquet").filter((pl.col("group") == "sp_total") & pl.col("hi").is_infinite()
                                                        & pl.col("lo").is_finite() & pl.col("ok"))
    sn = pl.read_parquet(SMILE / "snapshots.parquet", columns=["snap_id", "t_ref"])
    lad = pl.read_parquet(SMILE / "ladder_legs.parquet", columns=["ticker", "fee_mult", "fee_type"]).unique("ticker")
    x = lg.join(sn, on="snap_id").join(lad, on="ticker", how="left").filter(pl.col("ask").is_between(0.01, 0.99))
    mult = x["fee_mult"].fill_null(1.0).to_numpy()
    ft = x["fee_type"].fill_null("quadratic").to_list()
    x = x.with_columns(pl.Series("fee", fees(x["ask"].to_numpy(), mult, ft))).with_columns(
        (pl.col("ask") + pl.col("fee")).alias("cost"), (pl.col("y") - pl.col("ask") - pl.col("fee")).alias("pnl"),
        pl.col("snapshot_ts").alias("t_dec"), pl.col("t_ref").alias("t_end"))
    weeks = weeks_of(x)
    parts, variants = [], []
    for mstar in (0.25, 0.5, 1.0):
        for lead in (48, 24, 6, "pre"):
            name = f"|m|<={mstar}|lead={lead}"
            g = x.filter(pl.col("m_lo").abs() <= mstar)
            if lead == "pre":     # one leg per ladder: the strike nearest F at the earliest pre-game lead
                g = (g.filter(pl.col("lead_h") >= 6).with_columns(pl.col("m_lo").abs().alias("_am"))
                     .sort(["lead_h", "_am"], descending=[True, False]).unique("event_ticker", keep="first"))
            else:
                g = g.filter(pl.col("lead_h") == lead)
            parts.append(g.select("event_ticker", "split", "t_ref", "t_dec", "t_end", "pnl", "cost", "snap_id", "ticker",
                                  "y", "fee_type", "fee_mult", "snapshot_ts", pl.lit("yes").alias("side"),
                                  pl.col("ask").alias("price"))
                         .with_columns(pl.lit(name).alias("variant")))
            variants.append({"variant": name, "mstar": mstar, "lead": str(lead)})
    trades = pl.concat(parts)
    chosen = run_grid("sp_totals", trades, variants, weeks)[1]
    if "variant" in chosen:
        chosen["ledgers"] = {chosen["variant"]: write_ledger(
            "sp_totals", chosen["variant"], taker_ledger_legs(trades.filter(pl.col("variant") == chosen["variant"])))}
    return chosen


# ---- displayed size at the touch (L2 recorder, 2026-11-07..10) ----------------------------------
def _stream_rows(path: Path, prefix: str = "KXHIGH") -> list[tuple[int, str, dict]]:
    """(recv_ns, kind, msg) of the book/trade frames of ``prefix`` markets in one recorded file,
    read line by line (the whole-file reader needs GBs for the busiest hours)."""
    import io

    import zstandard

    from albert.research.strike_curves import _KINDS

    out = []
    with open(path, "rb") as f:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8"):
            if prefix not in line:
                continue
            d = json.loads(json.loads(line)["frame"]) if line.strip() else {}
            kind = _KINDS.get(d.get("type"))
            msg = d.get("msg") or {}
            if kind and str(msg.get("market_ticker", "")).startswith(prefix):
                out.append((json.loads(line)["recv_ns"], kind, msg))
    return out


def step_l2_tob(s, days: tuple[str, ...] = ("2026-11-07", "2026-11-08", "2026-11-09", "2026-11-10")) -> pl.DataFrame:
    """Top of book (price and displayed size) of every KXHIGH market in the recorded streams, hour
    by hour (``strike_curves.tob_from_events``), cached per hour under OUT/l2_tob."""
    import heapq

    from albert.research.strike_curves import _with_end, tob_from_events

    out = []
    (OUT / "l2_tob").mkdir(exist_ok=True)
    for day in days:
        root = s.raw_dir / "stream" / "kalshi" / day
        for h in range(24):
            path = OUT / "l2_tob" / f"{day}T{h:02d}.parquet"
            if not path.exists():
                files = sorted(root.glob(f"{h:02d}-*.jsonl.zst"))
                if not files:
                    continue
                streams = [_with_end(_stream_rows(p)) for p in files]
                tob_from_events(heapq.merge(*streams, key=lambda e: e.recv_ns)).write_parquet(path)
                print(f"{day} {h:02d}", flush=True)
            out.append(pl.read_parquet(path))
    tob = pl.concat(out)
    tob.write_parquet(OUT / "l2_tob_kxhigh.parquet")
    return tob


def step_l2_size() -> dict[str, Any]:
    """Displayed size at the touch at the snapshot times of the recorded KXHIGH legs, by lead and
    ask-price bucket."""
    from albert.venues.weather import STATION_TZ

    tob = pl.read_parquet(OUT / "l2_tob_kxhigh.parquet").with_columns(
        pl.from_epoch("t_ns", time_unit="ns").dt.cast_time_unit("us").dt.replace_time_zone("UTC").alias("ts")).sort("ts")
    # the hourly replay writes an empty quote when a file ends; keep real quotes only
    tob = tob.filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null())
    mk = pl.read_parquet(NOV / "tables" / "kalshi_markets.parquet", columns=["ticker", "event_ticker", "close_time"])
    codes = event_codes(NOV / "tables")
    tk = tob.select(pl.col("market").alias("ticker")).unique().join(mk, on="ticker").join(codes, on="event_ticker")
    rows = []
    for r in tk.iter_rows(named=True):
        if r["code"] not in STATION_TZ or r["day"] is None:
            continue
        lo, _ = lst_bounds(r["code"], r["day"])
        for lead, hrs in ((36, -12), (24, 0), (16, 8), (12, 12), (8, 16)):
            rows.append({"ticker": r["ticker"], "event_ticker": r["event_ticker"], "lead_h": lead,
                         "t": lo + timedelta(hours=hrs)})
    snaps = pl.DataFrame(rows).with_columns(pl.col("t").cast(pl.Datetime("us", "UTC"))).sort("t")
    q = snaps.join_asof(tob.rename({"market": "ticker"}), left_on="t", right_on="ts", by="ticker", strategy="backward",
                        tolerance=timedelta(hours=2)).filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null())
    q = q.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    # the recorder only streams the legs the live bots watch (mostly cheap wings): bucket by ask price
    q = q.with_columns(pl.col("ask").cut([0.10, 0.50, 0.90], labels=["<10c", "10-50c", "50-90c", ">=90c"]).alias("ask_bucket"))
    tab = (q.group_by("lead_h", "ask_bucket").agg(pl.len().alias("quotes"), pl.col("event_ticker").n_unique().alias("events"),
                                            (pl.col("ask") * 100).median().alias("ask_c_med"),
                                            pl.col("ask_sz").median().alias("ask_size_med"),
                                            (pl.col("ask_sz") >= ORDER_N).mean().round(3).alias("share_ask_ge_order"),
                                            pl.col("bid_sz").median().alias("bid_size_med"))
           .sort("lead_h", "ask_bucket", descending=[True, False]))
    print(tab, flush=True)
    return {"table": tab.to_dicts(), "markets": tk.height, "events": tk["event_ticker"].n_unique()}


# ---- summary -------------------------------------------------------------------------------------
def _fmt(c: dict[str, Any], sp: str) -> str:
    if not c.get(f"{sp}_n"):
        return "-"
    return f"{c[f'{sp}_n']} / {c[f'{sp}_events']} / {c[f'{sp}_c']:+.2f} / {c[f'{sp}_t']}"


def step_report() -> str:
    """Markdown table of every dev-chosen variant (from the step JSONs) into summary.md."""
    rows = []
    srcs = [("wx_taker", "W1 weather short-vol", "taker"), ("wx_maker", "W1 weather short-vol", "maker"),
            ("wx_drift", "W2 weather median drift", "taker"), ("sp_ml", "S1 moneyline vs spread", "taker"),
            ("sp_totals", "S2 totals over", "taker")]
    for name, strat, ex in srcs:
        path = OUT / f"{name}.json"
        if not path.exists():
            continue
        js = json.loads(path.read_text())
        items = js.items() if name in ("wx_taker", "wx_maker") else [("", js)]
        for model, c in items:
            if "variant" not in c:
                continue
            fr = ""
            if c.get("fill_rate"):
                fr = ", ".join(f"{r['split']} {r['fill_rate']}" for r in c["fill_rate"])
            led = (c.get("ledgers") or {}).get(c["variant"]) or {}
            lw = " / ".join(str((led.get(k) or {}).get("usd_week", "-")) for k in ("dev", "holdout", "nov"))
            rows.append(f"| {strat}{' (' + model + ')' if model else ''} | {ex} | {c['variant']} | {_fmt(c, 'dev')} | "
                        f"{_fmt(c, 'holdout')} | {_fmt(c, 'holdout2')} | {c.get('dev_per_dollar')} / "
                        f"{c.get('holdout_per_dollar')} / {c.get('holdout2_per_dollar')} | {fr or '-'} | "
                        f"{c.get('dev_usd_week')} / {c.get('holdout_usd_week')} / {c.get('holdout2_usd_week')} | {lw} |")
    head = ("| strategy | exec | dev-chosen variant | dev n / events / c / t | holdout n / events / c / t | "
            "holdout2 n / events / c / t | per $ dev / hold / hold2 | fill rate | $/week dev / hold / hold2 | "
            "ledger $/week dev / hold / nov |\n|---|---|---|---|---|---|---|---|---|---|")
    md = head + "\n" + "\n".join(rows) + "\n"
    (OUT / "summary.md").write_text(md)
    return md


# ---- driver --------------------------------------------------------------------------------------
def _dump(name: str, obj: Any) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str))


def main(argv: list[str]) -> None:
    from albert.config import settings

    s = settings()
    OUT.mkdir(parents=True, exist_ok=True)
    if not argv or argv[0] != "run":
        print(__doc__)
        return
    step = argv[1] if len(argv) > 1 else "all"
    if step in ("ohlc", "all"):
        step_ohlc(s)
    if step in ("panel", "all"):
        step_panel(s)
    if step == "l2_tob":
        step_l2_tob(s)
    if step == "l2_size":
        _dump("l2_size", step_l2_size())
    if step == "nov":
        _dump("nov", step_nov(s))
    if step in ("wx_taker", "all"):
        _dump("wx_taker", {m: step_wx_taker(m) for m in ("shrink", "nbm")})
    if step in ("wx_maker", "all"):
        _dump("wx_maker", {m: step_wx_maker(m) for m in ("shrink", "nbm")})
    if step in ("wx_trades", "all"):
        step_wx_trades(s)
    if step in ("wx_late_exact", "all"):
        _dump("wx_late_exact", step_wx_late_exact())
    if step in ("wx_drift", "all"):
        _dump("wx_drift", step_wx_drift())
    if step in ("sp_ml", "all"):
        _dump("sp_ml", step_sp_ml(s))
    if step in ("sp_totals", "all"):
        _dump("sp_totals", step_sp_totals())
    if step in ("report", "all"):
        print(step_report())


if __name__ == "__main__":
    main(sys.argv[1:])
