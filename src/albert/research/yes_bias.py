"""Asymmetric "sell the over-bought YES" maker held to settlement (Bartlett & O'Hara 2026, spec S2).

Claim under test: Kalshi takers over-buy YES, so makers who *sell* YES (rest NO bids) at
mid-range prices earn when held to settlement, most of all in single-name markets.

**1. Historical taker/maker split** (``kalshi_trades`` + ``kalshi_markets.result``). Per print
(price p = yes_price, y = 1 if the market settled YES, q = contracts):

* YES taker gross = y - p, NO taker gross = p - y (per contract, to settlement).
* The maker on the other side earns the negative, minus the maker fee 0.0175*mult*p*(1-p) on
  ``*_with_maker_fees`` series only (exact, the account's precision is $0.0001): the maker who
  **sells YES** (fills against YES takers) earns ``p - y - fee``; the maker who buys YES (fills
  against NO takers) earns ``y - p - fee``. Taker net subtracts 0.07*mult*p*(1-p).

Cells: 5c price band, category / series, market structure, hours to the scheduled expiration,
in-play vs pre-game (major leagues). Every cell is reported contract-weighted *and* per-market
equal-weighted, t-stats clustered by event, split dev (scheduled expiration before 2026-10-01) /
holdout (October), so each market sits in one split.

**Which prints are used (selection and look-ahead).** The trades table mixes several ingests;
only the volume-blind or completely covered parts enter, decided per ticker:

* ``window``: one leg of *every* game in the game/esports series (``kalshi_trade_windows``,
  population, no volume filter); only prints inside the recorded window (pre-game:
  expected_expiration - 25 h .. - 4 h for the major leagues). Prints of the same tickers outside
  the window come from other ingests (newest pages of a volume-ranked sample) and are dropped.
* ``inplay``: the in-play study's seeded volume-blind game sample (``data/cache/
  inplay_dynamics_trades``, read only), prints from the window end to the close.
* ``band``: the seeded random per-category sample with final volume in [500, 200k] contracts
  (``ingest kalshi-detail``), only tickers whose prints cover their whole volume (>= 98%); the
  275 truncated ones hold only their newest prints (close to resolution) and are dropped. The
  volume band is a selection on *final* volume; the lower bound is made look-ahead-free by the
  ``cum_ok`` filter (cumulative volume before the print >= 500, known at the print), and the upper
  bound removes < 1% of markets in the non-sports categories (reported).
* The top-20-by-volume ingest of Oct 30-31 (selected by final volume) is dropped outside windows.

In-play timing (verified on the in-play sample): the per-game trade rate jumps at
expected_expiration - 3 h (MLB, NFL, NCAAF; NBA at -2.75 h), so a print is ``inplay`` from
expected_expiration - ``INPLAY_OFFSET_H`` on; the scheduled time is known in advance.

**2. Trade-print maker** (``print_maker``): rest one contract one tick inside the print-implied
spread (last YES-taker price = ask proxy, last NO-taker price = bid proxy, both <= ``fresh_s``
old) on one side only. Sell-YES fills only when a later YES-taker print (after ``latency_s``)
trades strictly *through* our price (above it); the order lives ``ttl_s``. Held to settlement,
maker fee where charged. Optional directional-flow gate: skip/pull while the YES share of taker
contracts in the last ``gate_window_s`` is >= ``gate_share`` (prints strictly before the
decision). The mirror buy-YES maker is the control.

**3. Recorded L2** (``run_l2``): ``OneSided``, the baseline ``Quoter(improve=True,
min_spread=0.05, cooldown_s=300)`` restricted to the NO bid (sell YES) inside a price band, with
the optional flow pull, in the ``backtest.l2`` Simulator (FIFO queues, latency 50/100 ms,
post-only, 10 writes/s, exact series maker fees). Markouts mo/mx (``backtest.l2.markouts``),
settlement where settled else last mid, all / volatile / calm (each fill's own 5-min bucket,
shared definition) x dev / holdout.
"""

import bisect
import json
import math
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
INPLAY_OFFSET_H = 3.0
MAJOR_GAMES = ("KXNFLGAME", "KXNCAAFGAME", "KXMLBGAME", "KXNBAGAME", "KXNHLGAME", "KXWNBAGAME")
SINGLE_NAME = ("Mentions", "Politics", "Elections", "Entertainment", "Science and Technology")
BAND_MIN_VOLUME = 500.0
COMPLETE_COV = 0.98
TAKER, MAKER = 0.07, 0.0175
API = "https://api.elections.kalshi.com/trade-api/v2"
EPS = 1e-9


# ---- fees -------------------------------------------------------------------------------------
def maker_fee(p: float, fee_type: str | None, mult: float | None) -> float:
    """Exact Kalshi maker fee per contract (0 on series without maker fees)."""
    if not fee_type or "maker_fee" not in fee_type:
        return 0.0
    return MAKER * (1.0 if mult is None else mult) * p * (1 - p)


def taker_fee(p: float, mult: float | None) -> float:
    return TAKER * (1.0 if mult is None else mult) * p * (1 - p)


def _maker_fee_expr(p: pl.Expr) -> pl.Expr:
    return (pl.when(pl.col("fee_type").fill_null("").str.contains("maker_fee"))
            .then(MAKER * pl.col("fee_multiplier").fill_null(1.0) * p * (1 - p)).otherwise(0.0))


# ---- classification ---------------------------------------------------------------------------
def band_of(p: pl.Expr) -> pl.Expr:
    """5c band label of a YES price: '00-05', ..., '95-100'."""
    lo = (p * 100 + 1e-6).floor().clip(0, 99).floordiv(5) * 5
    return pl.format("{}-{}", lo.cast(pl.Int64).cast(pl.Utf8).str.zfill(2),
                     (lo + 5).cast(pl.Int64).cast(pl.Utf8).str.zfill(2))


def h_bucket(h: pl.Expr) -> pl.Expr:
    return (pl.when(h.is_null()).then(pl.lit("unknown")).when(h < 0).then(pl.lit("past exp"))
            .when(h < 1).then(pl.lit("0-1h")).when(h < 6).then(pl.lit("1-6h"))
            .when(h < 24).then(pl.lit("6-24h")).when(h < 168).then(pl.lit("1-7d"))
            .otherwise(pl.lit(">7d")))


def cls_of(category: pl.Expr) -> pl.Expr:
    return (pl.when(category.is_in(list(SINGLE_NAME))).then(pl.lit("single-name"))
            .when(category == "Sports").then(pl.lit("sports"))
            .when(category == "Climate and Weather").then(pl.lit("weather"))
            .otherwise(pl.lit("other")))


def top_volume_tickers(markets: pl.DataFrame) -> set[str]:
    """The 20 markets the first ``ingest kalshi --max-markets 20`` run took by final volume
    (closing 2026-10-30 .. 11-01, volume >= 1000): a volume-ranked selection to exclude."""
    m = markets.filter(pl.col("close_time").is_between(datetime(2026, 10, 30, tzinfo=UTC),
                                                       datetime(2026, 11, 1, tzinfo=UTC), closed="left")
                       & (pl.col("volume") >= 1000) & pl.col("result").is_in(["yes", "no"]))
    return set(m.sort("volume", descending=True).head(20)["ticker"].to_list())


def parse_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Structure of a recorded ``GET /events/{ticker}`` payload."""
    ev = payload.get("event") or {}
    mk = payload.get("markets") or []
    return {"me": ev.get("mutually_exclusive"), "n": len(mk),
            "strike_type": (mk[0].get("strike_type") if mk else None)}


def event_structure(events: Iterable[str], cache: Path, *, fetch: bool = True,
                    workers: int = 4) -> dict[str, dict[str, Any]]:
    """``mutually_exclusive`` and leg count per event (public REST, cached JSON)."""
    have = json.loads(cache.read_text()) if cache.exists() else {}
    todo = sorted(set(events) - set(have))
    if fetch and todo:
        import httpx
        from concurrent.futures import ThreadPoolExecutor

        def one(e: str) -> tuple[str, Any]:
            for _ in range(3):
                try:
                    r = http.get(f"{API}/events/{e}")
                except httpx.HTTPError:
                    time.sleep(1.0)
                    continue
                if r.status_code == 200:
                    return e, parse_event(r.json())
                if r.status_code == 404:
                    return e, None
                time.sleep(2.0)
            return e, "error"

        with httpx.Client(timeout=20) as http, ThreadPoolExecutor(max_workers=workers) as ex:
            for i, (e, v) in enumerate(ex.map(one, todo)):
                if v != "error":
                    have[e] = v
                if i % 500 == 499:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(have))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(have))
    return {e: have.get(e) for e in set(events)}


def structure_of(legs: int | None, me: bool | None, series: str) -> str:
    """single (one market), 2-way (two exclusive legs, e.g. a game), multi-ME (one leg of >= 3
    exclusive outcomes), multi-indep (one strike of non-exclusive legs, e.g. mention words)."""
    if legs is None:
        return "unknown"
    if legs <= 1:
        return "single"
    if me is None:
        me = series.endswith(("GAME", "MATCH", "MAP"))
    if me:
        return "2-way" if legs == 2 else "multi-ME"
    return "multi-indep"


# ---- panel ------------------------------------------------------------------------------------
TRADE_COLS = ["ticker", "created_time", "count", "yes_price", "taker_side"]
SOURCES = pl.Enum(["window", "inplay", "band"])
PHASES = pl.Enum(["pregame", "inplay", "n/a"])


def market_frame(s) -> pl.DataFrame:
    """Settled markets with an integer id ``mid``, outcome ``y``, split (scheduled expiration
    before / from 2026-10-01), class and leg count of the event."""
    m = pl.read_parquet(s.tables_dir / "kalshi_markets.parquet",
                        columns=["ticker", "event_ticker", "series_ticker", "category", "result", "volume",
                                 "close_time", "expected_expiration_time", "fee_type", "fee_multiplier"])
    m = m.filter(pl.col("result").is_in(["yes", "no"])).sort("ticker")
    legs = m.group_by("event_ticker").agg(pl.len().alias("legs"))
    exp = pl.coalesce("expected_expiration_time", "close_time")
    return m.join(legs, on="event_ticker", how="left").with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("mid"),
        pl.col("event_ticker").alias("event"), pl.col("series_ticker").alias("series"),
        (pl.col("result") == "yes").cast(pl.Float64).alias("y"),
        pl.when(exp >= HOLDOUT_FROM).then(pl.lit("holdout")).otherwise(pl.lit("dev")).alias("split"),
        exp.alias("exp"), cls_of(pl.col("category")).alias("cls"))


def coverage(trades: pl.LazyFrame) -> pl.DataFrame:
    return trades.group_by("ticker").agg(pl.len().alias("n"), pl.col("count").sum().alias("q")).collect()


def build_prints(s, markets: pl.DataFrame | None = None, *, inplay_dir: Path | None = None,
                 force: bool = False) -> Path:
    """Usable prints (module doc) as lean Parquet ``data/cache/yes_bias_prints.parquet``:
    mid, t, q, p, yes (YES taker), source, phase, h_exp (hours to the scheduled expiration),
    cum_ok (volume-band lower bound already reached at the print). Derived; ``force`` rebuilds."""
    path = s.data_dir / "cache" / "yes_bias_prints.parquet"
    if path.exists() and not force:
        return path
    markets = market_frame(s) if markets is None else markets
    windows = pl.read_parquet(s.tables_dir / "kalshi_trade_windows.parquet")
    lf = pl.scan_parquet(s.tables_dir / "kalshi_trades.parquet").select(TRADE_COLS)
    cov = coverage(lf).join(markets.select("ticker", "volume"), on="ticker", how="inner").with_columns(
        (pl.col("q") / pl.col("volume")).alias("cov"))
    top = top_volume_tickers(markets)
    win_t = set(windows["ticker"].to_list())
    band_t = cov.filter((pl.col("cov") >= COMPLETE_COV) & ~pl.col("ticker").is_in(list(win_t | top)))["ticker"]
    parts = [
        lf.join(windows.lazy().select("ticker", "window_start", "window_end"), on="ticker", how="inner")
        .filter(pl.col("created_time").is_between(pl.col("window_start"), pl.col("window_end")))
        .select(TRADE_COLS).with_columns(pl.lit("window").alias("source")),
        lf.filter(pl.col("ticker").is_in(band_t.implode())).with_columns(pl.lit("band").alias("source")),
    ]
    inplay_dir = inplay_dir or s.data_dir / "cache" / "inplay_dynamics_trades"
    files = sorted(inplay_dir.glob("*.parquet")) if inplay_dir.exists() else []
    if files:
        parts.append(pl.scan_parquet(files).select(TRADE_COLS)
                     .join(windows.lazy().select("ticker", "window_end"), on="ticker", how="inner")
                     .filter(pl.col("created_time") > pl.col("window_end")).select(TRADE_COLS)
                     .with_columns(pl.lit("inplay").alias("source")))
    # Cumulative volume before each print, for the volume-band lower bound (``band`` has
    # complete coverage, so this is the market's true cumulative volume at the print).
    parts[1] = parts[1].sort("ticker", "created_time").with_columns(
        ((pl.col("count").cum_sum().over("ticker") - pl.col("count")) >= BAND_MIN_VOLUME).alias("cum_ok"))
    parts[0] = parts[0].with_columns(pl.lit(True).alias("cum_ok"))
    if len(parts) > 2:
        parts[2] = parts[2].with_columns(pl.lit(True).alias("cum_ok"))
    attrs = markets.lazy().select("ticker", "mid", "series", "exp", "expected_expiration_time")
    major = pl.col("series").is_in(list(MAJOR_GAMES))
    start = pl.col("expected_expiration_time") - pl.duration(minutes=int(INPLAY_OFFSET_H * 60))
    out = pl.concat(parts).join(attrs, on="ticker", how="inner").select(
        "mid", pl.col("created_time").alias("t"), pl.col("count").alias("q"), pl.col("yes_price").alias("p"),
        (pl.col("taker_side") == "yes").alias("yes"), pl.col("source").cast(SOURCES),
        pl.when(major).then(pl.when(pl.col("created_time") >= start).then(pl.lit("inplay"))
                            .otherwise(pl.lit("pregame"))).otherwise(pl.lit("n/a")).cast(PHASES).alias("phase"),
        ((pl.col("exp") - pl.col("created_time")).dt.total_seconds() / 3600).cast(pl.Float32).alias("h_exp"),
        "cum_ok")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.sink_parquet(path)
    meta = {"built": datetime.now(UTC).isoformat(), "inplay_files": len(files), "band_tickers": band_t.len(),
            "window_tickers": len(win_t), "top_volume_excluded": sorted(top)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return path


def add_structure(markets: pl.DataFrame, events: Iterable[str], s, *, fetch: bool = True) -> pl.DataFrame:
    """Add ``structure`` (``structure_of``) to ``markets``; ``mutually_exclusive`` comes from
    public REST for ``events`` (cached), else from the series name (games are exclusive)."""
    struct = event_structure(events, s.data_dir / "cache" / "yes_bias_events.json", fetch=fetch)
    me = {e: (v or {}).get("me") for e, v in struct.items()}
    m = markets.with_columns(pl.col("event").replace_strict(me, default=None, return_dtype=pl.Boolean).alias("me"))
    keys = m.select("event", "legs", "me", "series").unique()
    keys = keys.with_columns(pl.struct("legs", "me", "series").map_elements(
        lambda r: structure_of(r["legs"], r["me"], r["series"]), return_dtype=pl.Utf8).alias("structure"))
    return m.join(keys.select("event", "series", "structure"), on=["event", "series"], how="left")


def maker_net_expr() -> pl.Expr:
    """Per contract, to settlement: the maker who sold YES to a YES taker earns p - y, the one who
    bought YES from a NO taker y - p; minus the maker fee where charged."""
    gross = pl.when(pl.col("yes")).then(pl.col("p") - pl.col("y")).otherwise(pl.col("y") - pl.col("p"))
    return gross - _maker_fee_expr(pl.col("p"))


def aggregate(prints: Path, markets: pl.DataFrame, keys: list[str], *,
              where: pl.Expr | None = None) -> pl.DataFrame:
    """Per (market, keys..., taker side): contracts ``q``, prints ``n``, and sums of maker PnL
    ``sm`` and of YES-taker-minus-settlement ``sg`` (= sum q * (p - y)), joined to market
    attributes. Derived keys: ``band`` (5c), ``h_bucket``."""
    attrs = markets.select("mid", "y", "fee_type", "fee_multiplier")
    lf = pl.scan_parquet(prints)
    if where is not None:
        lf = lf.filter(where)
    lf = lf.join(attrs.lazy(), on="mid", how="inner").with_columns(
        band_of(pl.col("p")).alias("band"), h_bucket(pl.col("h_exp")).alias("h_bucket"),
        maker_net_expr().alias("mn"))
    g = lf.group_by(["mid", *keys, "yes"]).agg(
        pl.col("q").sum().alias("q"), pl.len().alias("n"), (pl.col("q") * pl.col("mn")).sum().alias("sm"),
        (pl.col("q") * (pl.col("p") - pl.col("y"))).sum().alias("sg")).collect()
    cols = ["mid", "ticker", "event", "series", "category", "cls", "split"] + (
        ["structure"] if "structure" in markets.columns else [])
    return g.join(markets.select(cols), on="mid", how="inner")


# ---- statistics ---------------------------------------------------------------------------------
def _ct(g: pl.DataFrame) -> tuple[float | None, float | None, int]:
    """Mean sum(s)/sum(n) with a CR1 t-stat; one row per cluster with columns s, n."""
    if g.is_empty():
        return None, None, 0
    n = float(g["n"].sum())
    if n <= 0:
        return None, None, 0
    mean = float(g["s"].sum()) / n
    k = g.height
    if k < 2:
        return mean, None, k
    se = math.sqrt(k / (k - 1) * float(((g["s"] - mean * g["n"]) ** 2).sum())) / n
    return mean, (mean / se if se > 1e-12 else None), k


def clustered(df: pl.DataFrame, col: str, *, weight: str | None = "q",
              cluster: str = "event") -> tuple[float | None, float | None, int]:
    """Weighted mean of ``col`` with a t-stat clustered by ``cluster`` (CR1 sandwich)."""
    f = df.filter(pl.col(col).is_not_null())
    if f.is_empty():
        return None, None, 0
    w = pl.col(weight) if weight else pl.lit(1.0)
    return _ct(f.group_by(cluster).agg((pl.col(col) * w).sum().alias("s"), w.sum().alias("n")))


def market_equal(df: pl.DataFrame, col: str) -> tuple[float | None, float | None, int]:
    """Per-market contract-weighted mean of ``col``, then every market weighs 1; t by event."""
    if df.is_empty():
        return None, None, 0
    g = df.group_by("ticker", "event").agg(((pl.col(col) * pl.col("q")).sum() / pl.col("q").sum()).alias("m"))
    return clustered(g, "m", weight=None)


def _r(x: float | None, k: int = 2, scale: float = 1.0) -> float | None:
    return None if x is None else round(scale * x, k)


def cell_stats(a: pl.DataFrame) -> dict[str, Any]:
    """From ``aggregate`` rows of one cell: maker PnL per contract to settlement in cents for the
    maker selling YES (``sellY``: counterparty of YES takers) and buying YES (``buyY``),
    contract-weighted and market-equal-weighted (``_ew``), t clustered by event; the paired
    within-market asymmetry ``asym`` = sellY - buyY (markets with both, equal weight), about
    2 x (price - settlement rate) plus any extra informedness of YES over NO takers; and the
    calibration gap ``gap`` = contract-weighted p - y over all prints."""
    out: dict[str, Any] = {"prints": int(a["n"].sum()) if a.height else 0,
                           "contracts": round(float(a["q"].sum()), 1) if a.height else 0.0,
                           "markets": a["mid"].n_unique() if a.height else 0,
                           "events": a["event"].n_unique() if a.height else 0}
    per_m = a.group_by("mid", "event", "yes").agg(pl.col("q").sum(), pl.col("sm").sum(), pl.col("sg").sum())
    for side, name in ((True, "sellY"), (False, "buyY")):
        x = per_m.filter(pl.col("yes") == side)
        m, t, _ = _ct(x.group_by("event").agg(pl.col("sm").sum().alias("s"), pl.col("q").sum().alias("n")))
        me, te, _ = _ct(x.with_columns((pl.col("sm") / pl.col("q")).alias("m")).group_by("event")
                        .agg(pl.col("m").sum().alias("s"), pl.len().alias("n")))
        out[f"{name}_q%"] = round(100 * float(x["q"].sum()) / max(float(a["q"].sum()), EPS), 1)
        out[f"{name}_c"], out[f"{name}_t"] = _r(m, 2, 100), _r(t)
        out[f"{name}_ew_c"], out[f"{name}_ew_t"] = _r(me, 2, 100), _r(te)
    piv = per_m.with_columns((pl.col("sm") / pl.col("q")).alias("m")).pivot(
        on="yes", index=["mid", "event"], values="m") if per_m.height else per_m
    if piv.height and {"true", "false"} <= set(piv.columns):
        d = piv.drop_nulls().with_columns((pl.col("true") - pl.col("false")).alias("d"))
        md, td, _ = _ct(d.group_by("event").agg(pl.col("d").sum().alias("s"), pl.len().alias("n")))
        out["asym_c"], out["asym_t"], out["asym_mkts"] = _r(md, 2, 100), _r(td), d.height
    else:
        out["asym_c"], out["asym_t"], out["asym_mkts"] = None, None, 0
    gm, gt, _ = _ct(per_m.group_by("event").agg(pl.col("sg").sum().alias("s"), pl.col("q").sum().alias("n")))
    out["gap_c"], out["gap_t"] = _r(gm, 2, 100), _r(gt)
    return out


def split_table(a: pl.DataFrame, by: list[str], *, min_contracts: float = 0.0) -> pl.DataFrame:
    """``cell_stats`` per value of ``by`` x split (dev / holdout)."""
    rows = []
    if a.is_empty():
        return pl.DataFrame()
    for key, sub in sorted(a.group_by(by), key=lambda kv: tuple(str(x) for x in kv[0])):
        if float(sub["q"].sum()) < min_contracts:
            continue
        for split in ("dev", "holdout"):
            rows.append({**dict(zip(by, key)), "split": split, **cell_stats(sub.filter(pl.col("split") == split))})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- 2. trade-print maker (historical) ---------------------------------------------------------
@dataclass
class PrintParams:
    side: str = "sell_yes"         # sell_yes (rest a NO bid) | buy_yes (rest a YES bid): control
    min_spread: float = 0.02       # print-implied spread (last YES-taker - last NO-taker price)
    lo: float = 0.30               # mid band
    hi: float = 0.70
    fresh_s: float = 120.0         # both proxies at most this old
    ttl_s: float = 300.0           # order lifetime
    latency_s: float = 0.1         # prints after placement + latency can fill
    cooldown_s: float = 300.0      # after a fill
    cap_market: float = 3.0        # contracts per market
    cap_event: float = 5.0         # contracts per event (applied in time order)
    gate_window_s: float = 0.0     # 0 = no flow gate
    gate_share: float = 1.01       # pull while the adverse taker side's share >= this
    gate_min_q: float = 0.0        # ... and at least this many taker contracts in the window

    def name(self) -> str:
        g = (f"|gate{self.gate_window_s:g}s>={self.gate_share:g}" if self.gate_window_s else "")
        return (f"{self.side} sp>={self.min_spread:g} mid{self.lo:g}-{self.hi:g} ttl{self.ttl_s:g}{g}")


def print_maker_market(t: list[float], p: list[float], yes: list[bool], q: list[float],
                       prm: PrintParams) -> list[dict[str, Any]]:
    """Fills of the one-sided print maker in one market (prints in time order, times in
    seconds). Decisions after a print use only that print and earlier ones; an order placed at
    t fills only on prints later than t + latency:

    * sell_yes at a: a YES-taker print above a (strictly through), or a NO-taker print at or
      above a (a YES bid that would have crossed our offer: ``cross``);
    * buy_yes at b: a NO-taker print below b, or a YES-taker print at or below b (cross).
    """
    sell = prm.side == "sell_yes"
    fills: list[dict[str, Any]] = []
    ask = bid = None
    t_ask = t_bid = -math.inf
    flow: deque[tuple[float, bool, float]] = deque()
    flow_yes = flow_tot = 0.0
    order: dict[str, Any] | None = None
    cool_until = -math.inf
    pos = 0.0
    for i in range(len(t)):
        ti, pi, yi, qi = t[i], p[i], yes[i], q[i]
        # 1. the print against our resting order
        if order is not None and ti > order["t_exp"]:
            order = None
        if order is not None and ti > order["t_act"]:
            px = order["px"]
            how = None
            if sell:
                if yi and pi > px + EPS:
                    how = "through"
                elif not yi and pi >= px - EPS:
                    how = "cross"
            else:
                if not yi and pi < px - EPS:
                    how = "through"
                elif yi and pi <= px + EPS:
                    how = "cross"
            if how is not None:
                x = min(order["rem"], qi)
                order["rem"] -= x
                pos += x
                fills.append({"t": ti, "px": px, "qty": x, "how": how, "t_place": order["t_place"],
                              "spread": order["spread"], "mid": order["mid"], "flow_share": order["share"]})
                if order["rem"] <= EPS or pos >= prm.cap_market - EPS:
                    order = None
                    cool_until = ti + prm.cooldown_s
        # 2. state after this print
        if yi:
            ask, t_ask = pi, ti
        else:
            bid, t_bid = pi, ti
        if prm.gate_window_s:
            flow.append((ti, yi, qi))
            flow_tot += qi
            flow_yes += qi if yi else 0.0
            while flow and flow[0][0] <= ti - prm.gate_window_s:
                _, y0, q0 = flow.popleft()
                flow_tot -= q0
                flow_yes -= q0 if y0 else 0.0
        # 3. decision
        ok = (ask is not None and bid is not None and ti - t_ask <= prm.fresh_s and ti - t_bid <= prm.fresh_s
              and ask - bid >= prm.min_spread - EPS and prm.lo - EPS <= (ask + bid) / 2 <= prm.hi + EPS)
        share = None
        if prm.gate_window_s and flow_tot > EPS:
            share = (flow_yes if sell else flow_tot - flow_yes) / flow_tot
            if flow_tot >= prm.gate_min_q - EPS and share >= prm.gate_share - EPS:
                ok = False
        if order is not None:
            if not ok:
                order = None              # pull (band left, spread closed or gate)
            continue
        if not ok or ti < cool_until or pos >= prm.cap_market - EPS:
            continue
        px = round(ask - 0.01, 6) if sell else round(bid + 0.01, 6)
        order = {"px": px, "t_place": ti, "t_act": ti + prm.latency_s, "t_exp": ti + prm.ttl_s,
                 "rem": min(1.0, prm.cap_market - pos), "spread": ask - bid, "mid": (ask + bid) / 2,
                 "share": share}
    return fills


@dataclass
class Sequences:
    """Prints of every market as flat numpy arrays sorted by (market, time); ``bounds[i]`` ..
    ``bounds[i+1]`` is market ``mids[i]``."""
    mids: Any
    bounds: Any
    t: Any
    p: Any
    yes: Any
    q: Any
    phase: Any
    source: Any


def load_sequences(prints: Path, sources: tuple[str, ...] = ("window", "inplay", "band")) -> Sequences:
    import numpy as np

    f = (pl.scan_parquet(prints).filter(pl.col("source").cast(pl.Utf8).is_in(list(sources)))
         .select("mid", (pl.col("t").dt.epoch("us") / 1e6).alias("t"), "p", "yes", "q",
                 pl.col("phase").to_physical().alias("phase"), pl.col("source").to_physical().alias("source"))
         .sort("mid", "t").collect())
    mid = f["mid"].to_numpy()
    cut = np.flatnonzero(np.diff(mid)) + 1
    bounds = np.concatenate([[0], cut, [len(mid)]])
    return Sequences(mids=mid[bounds[:-1]], bounds=bounds, t=f["t"].to_numpy(), p=f["p"].to_numpy(),
                     yes=f["yes"].to_numpy(), q=f["q"].to_numpy(), phase=f["phase"].to_numpy(),
                     source=f["source"].to_numpy())


FILL_SCHEMA = {"mid": pl.UInt32, "t": pl.Float64, "px": pl.Float64, "q": pl.Float64, "how": pl.Utf8,
               "spread": pl.Float64, "fmid": pl.Float64, "flow_share": pl.Float64, "phase": pl.Utf8,
               "source": pl.Utf8}


def print_maker(seq: Sequences, markets: pl.DataFrame, prm: PrintParams) -> pl.DataFrame:
    """Run ``print_maker_market`` over every market and score each fill to settlement net of the
    maker fee (``pnl`` per contract). The event cap is applied in time order."""
    phases, sources = PHASES.categories.to_list(), SOURCES.categories.to_list()
    rows: list[dict[str, Any]] = []
    for i, m in enumerate(seq.mids.tolist()):
        a, b = int(seq.bounds[i]), int(seq.bounds[i + 1])
        t = seq.t[a:b].tolist()
        fills = print_maker_market(t, seq.p[a:b].tolist(), seq.yes[a:b].tolist(), seq.q[a:b].tolist(), prm)
        for fl in fills:
            j = min(bisect.bisect_left(t, fl["t"]), b - a - 1)
            rows.append({"mid": m, "t": fl["t"], "px": fl["px"], "q": fl["qty"], "how": fl["how"],
                         "spread": fl["spread"], "fmid": fl["mid"], "flow_share": fl["flow_share"],
                         "phase": phases[int(seq.phase[a + j])], "source": sources[int(seq.source[a + j])]})
    out = pl.DataFrame(rows, schema=FILL_SCHEMA)
    cols = ["mid", "ticker", "event", "series", "category", "cls", "split", "y", "fee_type", "fee_multiplier"] + (
        ["structure"] if "structure" in markets.columns else [])
    out = out.join(markets.select(cols), on="mid", how="inner")
    if out.is_empty():
        return out.with_columns(pl.lit(None, dtype=pl.Float64).alias("fee"), pl.lit(None, dtype=pl.Float64).alias("pnl"))
    out = out.sort("t").with_columns(pl.col("q").cum_sum().over("event").alias("_cum"))
    out = out.filter(pl.col("_cum") - pl.col("q") < prm.cap_event - EPS).with_columns(
        pl.min_horizontal(pl.col("q"), prm.cap_event - (pl.col("_cum") - pl.col("q"))).alias("q")).drop("_cum")
    fee = _maker_fee_expr(pl.col("px"))
    gross = (pl.col("px") - pl.col("y")) if prm.side == "sell_yes" else (pl.col("y") - pl.col("px"))
    return out.with_columns(fee.alias("fee"), (gross - fee).alias("pnl"))


def fill_stats(f: pl.DataFrame) -> dict[str, Any]:
    m, t, k = clustered(f, "pnl") if f.height else (None, None, 0)
    me, te, _ = market_equal(f, "pnl") if f.height else (None, None, 0)
    return {"fills": f.height, "contracts": round(float(f["q"].sum()), 2) if f.height else 0.0,
            "markets": f["ticker"].n_unique() if f.height else 0, "events": k,
            "pnl_c": None if m is None else round(100 * m, 2), "t": None if t is None else round(t, 2),
            "ew_c": None if me is None else round(100 * me, 2), "ew_t": None if te is None else round(te, 2),
            "usd": round(float((f["q"] * f["pnl"]).sum()), 2) if f.height else 0.0,
            "avg_px": round(float(f["px"].mean()), 3) if f.height else None,
            "yes_rate": round(float(f["y"].mean()), 3) if f.height else None}


def fills_table(f: pl.DataFrame, by: list[str] | None = None) -> pl.DataFrame:
    rows = []
    if not by:
        groups = [((), f)]
    else:
        groups = [(k, g) for k, g in f.group_by(by, maintain_order=True)] if f.height else []
    for key, g in groups:
        for split in ("dev", "holdout"):
            x = g.filter(pl.col("split") == split)
            rows.append({**dict(zip(by or [], key)), "split": split, **fill_stats(x)})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- 3. recorded L2 ----------------------------------------------------------------------------
from albert.backtest.l2 import (Event, Market, Quoter, SimConfig, Simulator, iter_events,  # noqa: E402
                                markouts, stream_files)
from albert.fees import kalshi_fee  # noqa: E402

BASE = dict(improve=True, min_spread=0.05, cooldown_s=300.0)
BUCKET_NS = 300 * 10**9


class _FastMarket(Market):
    """Same (price, size) as ``Market.best``; walks levels from the top and stops early."""

    def best(self, side: str) -> tuple[float, float] | None:
        lv = self.levels[side]
        for p in sorted(lv, reverse=True):
            s = lv[p].ext_total()
            if s > EPS:
                return p, s
        return None


class _NoList(list):
    def append(self, x: Any) -> None:   # noqa: D401 - a list that never stores
        return None


class LaneSim(Simulator):
    """``Simulator`` with fast books, exact maker fees (the account's precision is $0.0001,
    the base class rounds every fill up to a cent) and, unless ``keep_quotes``, no top-of-book
    history (markouts come from one shared reference simulator: the book strategies see is the
    recorded book, identical in every lane)."""

    def __init__(self, cfg: SimConfig, keep_quotes: bool = False) -> None:
        super().__init__(cfg)
        self.markets = defaultdict(_FastMarket)
        if not keep_quotes:
            self.quotes = defaultdict(lambda: (_NoList(), _NoList()))

    def _fee(self, market: str, qty: float, price: float) -> float:
        info = self.cfg.fee_info.get(market.split("-", 1)[0], {})
        return kalshi_fee(qty, price, maker=True, fee_type=info.get("fee_type") or "quadratic",
                          multiplier=info.get("fee_multiplier") or 1.0, conservative=False)


class OneSided(Quoter):
    """The baseline Quoter on one side only. ``side='no'`` rests the NO bid (sells YES);
    ``side='yes'`` (the control) the YES bid. The mid band is ``lo``..``hi``. Optional flow
    pull: no quote while, over the last ``gate_window_s`` of this market's taker prints (all
    before the decision), the adverse side (YES takers when selling YES) has at least
    ``gate_share`` of at least ``gate_min_q`` contracts."""

    def __init__(self, side: str = "no", gate_window_s: float = 0.0, gate_share: float = 1.01,
                 gate_min_q: float = 0.0, **kw: Any) -> None:
        super().__init__(**kw)
        self.side = side
        self.gate_ns = int(gate_window_s * 1e9)
        self.gate_share, self.gate_min_q = gate_share, gate_min_q
        self.flow: dict[str, deque] = defaultdict(deque)
        self.gated = 0

    def observe(self, sim: Simulator, ev: Event) -> None:
        if self.gate_ns and ev.kind == "trade":
            m = ev.msg
            self.flow[ev.market].append((ev.recv_ns, m.get("taker_side") == "yes", float(m.get("count_fp") or 0)))

    def adverse_share(self, market: str, now: int) -> tuple[float, float]:
        dq = self.flow[market]
        while dq and dq[0][0] <= now - self.gate_ns:
            dq.popleft()
        tot = sum(x[2] for x in dq)
        yes = sum(x[2] for x in dq if x[1])
        adv = yes if self.side == "no" else tot - yes
        return (adv / tot if tot > EPS else 0.0), tot

    def _wanted(self, sim: Simulator, market: str) -> dict[str, float | None]:
        want = super()._wanted(sim, market)
        want["yes" if self.side == "no" else "no"] = None
        if self.gate_ns and want[self.side] is not None:
            share, tot = self.adverse_share(market, sim.now)
            if tot >= self.gate_min_q - EPS and share >= self.gate_share - EPS:
                want[self.side] = None
                self.gated += 1
        return want


def l2_lanes(category_of: dict[str, str | None], lo: float, hi: float,
             gate: tuple[float, float, float]) -> dict[str, Any]:
    gw, gs, gq = gate
    return {
        "sellY": lambda: OneSided(side="no", lo=lo, hi=hi, category_of=category_of, **BASE),
        "sellY_gate": lambda: OneSided(side="no", lo=lo, hi=hi, gate_window_s=gw, gate_share=gs, gate_min_q=gq,
                                       category_of=category_of, **BASE),
        "sellY_all": lambda: OneSided(side="no", category_of=category_of, **BASE),
        "buyY": lambda: OneSided(side="yes", lo=lo, hi=hi, category_of=category_of, **BASE),
        "two_sided": lambda: Quoter(category_of=category_of, **BASE),
    }


def market_results(tickers: Iterable[str], cache: Path, *, fetch: bool = True) -> dict[str, float | None]:
    """Settled YES result (1.0 / 0.0) per market (public REST, cached JSON; unsettled re-asked)."""
    tickers = sorted(set(tickers))
    have = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [t for t in tickers if have.get(t) is None]
    if fetch and todo:
        import httpx

        with httpx.Client(timeout=20) as http:
            for t in todo:
                try:
                    r = http.get(f"{API}/markets/{t}")
                except httpx.HTTPError:
                    continue
                if r.status_code == 200:
                    have[t] = {"yes": 1.0, "no": 0.0}.get((r.json().get("market") or {}).get("result"))
                time.sleep(0.06)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(have))
    return {t: have.get(t) for t in tickers}


L2_SCHEMA = {"lane": pl.Utf8, "latency": pl.Float64, "t_ns": pl.Int64, "market": pl.Utf8, "side": pl.Utf8,
             "price": pl.Float64, "qty": pl.Float64, "fee": pl.Float64, "how": pl.Utf8,
             "mo_60s": pl.Float64, "mo_300s": pl.Float64, "mx_60s": pl.Float64, "mx_300s": pl.Float64,
             "last_mid": pl.Float64}


def replay_l2(s, start: str, end: str, *, lo: float, hi: float, gate: tuple[float, float, float],
              latencies: Iterable[float] = (50.0, 100.0), root: Path | None = None,
              log: Any = print) -> pl.DataFrame:
    """One pass over the recorded hours; every lane has its own simulator. Returns the fills
    with markouts (from a shared reference book) and the last recorded mid."""
    series = pl.read_parquet(s.tables_dir / "kalshi_series.parquet")
    fee_info = {r["series_ticker"]: r for r in series.iter_rows(named=True)}
    category_of = {k: v.get("category") for k, v in fee_info.items()}
    files = stream_files(root or s.raw_dir / "stream", start, end)
    if not files:
        raise RuntimeError(f"no recorded Kalshi stream files in [{start}, {end})")
    specs = l2_lanes(category_of, lo, hi, gate)
    lanes = []
    for lat in latencies:
        for name, make in specs.items():
            cfg = SimConfig(latency_ms=lat, queue="risk_averse", fee_info=fee_info, writes_per_s=10.0)
            lanes.append((name, float(lat), make(), LaneSim(cfg)))
    ref = LaneSim(SimConfig(writes_per_s=None), keep_quotes=True)
    stats: dict[str, int] = defaultdict(int)
    hour, t0, n = None, time.time(), 0
    for ev in iter_events(files, stats):
        n += 1
        h = ev.recv_ns // 3_600_000_000_000
        if h != hour:
            if hour is not None and log:
                log(f"  hour {hour % 24:02d} done: {n} events, {time.time() - t0:.0f}s, fills "
                    + " ".join(f"{nm}@{lat:g}:{len(sm.fills)}" for nm, lat, _, sm in lanes))
            hour = h
        ref.apply(ev)
        for _, _, strat, sim in lanes:
            sim.apply(ev)
            obs = getattr(strat, "observe", None)
            if obs is not None:
                obs(sim, ev)
            strat.on_event(sim, ev.market)
    rows = []
    real = ref.fills
    for name, lat, _, sim in lanes:
        ref.fills = sim.fills
        for f, mo in zip(sim.fills, markouts(ref)):
            rows.append({"lane": name, "latency": lat, "t_ns": f.t_ns, "market": f.market, "side": f.side,
                         "price": f.price, "qty": f.qty, "fee": f.fee, "how": f.how,
                         "mo_60s": mo["mo_60s"], "mo_300s": mo["mo_300s"], "mx_60s": mo["mx_60s"],
                         "mx_300s": mo["mx_300s"], "last_mid": ref.mid_at(f.market, 2**62)})
    ref.fills = real
    if log:
        log(f"  replay done: {n} events, {time.time() - t0:.0f}s; trades {stats['trades']}")
    return pl.DataFrame(rows, schema=L2_SCHEMA)


# ---- 2b. grid on dev ---------------------------------------------------------------------------
GRID_BANDS = ((0.30, 0.70), (0.10, 0.90), (0.70, 0.95), (0.05, 0.30))
GRID_GATES = ((0.0, 1.01, 0.0), (300.0, 0.70, 10.0), (300.0, 0.85, 10.0), (60.0, 0.80, 5.0))


def grid_params(sides: tuple[str, ...] = ("sell_yes", "buy_yes")) -> list[PrintParams]:
    out = []
    for side in sides:
        for ms in (0.02, 0.04):
            for lo, hi in GRID_BANDS:
                for ttl in (60.0, 300.0):
                    for gw, gs, gq in GRID_GATES:
                        out.append(PrintParams(side=side, min_spread=ms, lo=lo, hi=hi, ttl_s=ttl,
                                               gate_window_s=gw, gate_share=gs, gate_min_q=gq))
    return out


def group_expr() -> pl.Expr:
    """Reporting group of a fill: the sample (window = pre-game game/esports population,
    inplay) or, for the volume-band sample, the class (single-name / sports / weather)."""
    return (pl.when(pl.col("source") == "band").then(pl.concat_str(pl.lit("band:"), pl.col("cls")))
            .otherwise(pl.col("source"))).alias("grp")


def run_grid(seq: Sequences, markets: pl.DataFrame, params: list[PrintParams],
             log: Any = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(summary, fills): ``fills_table`` per run for all fills and per group, and every fill."""
    summ, allf = [], []
    for i, prm in enumerate(params):
        f = print_maker(seq, markets, prm).with_columns(group_expr())
        f = f.with_columns(pl.lit(prm.name()).alias("run"))
        allf.append(f.select("run", "mid", "ticker", "event", "grp", "split", "phase", "t", "px", "q", "how",
                             "flow_share", "fee", "pnl", "y"))
        for grp in ["all", *sorted(f["grp"].unique().to_list())]:
            sub = f if grp == "all" else f.filter(pl.col("grp") == grp)
            tab = fills_table(sub)
            summ.append(tab.with_columns(pl.lit(prm.name()).alias("run"), pl.lit(grp).alias("grp"),
                                         pl.lit(json.dumps(asdict(prm))).alias("params")))
        if log and i % 8 == 7:
            log(f"  grid {i + 1}/{len(params)}")
    return pl.concat(summ, how="diagonal_relaxed"), pl.concat(allf, how="diagonal_relaxed")


def _hour_ns(hour: str) -> int:
    return int(datetime.fromisoformat(hour + ":00:00+00:00").timestamp() * 1e9)


def l2_metrics(fills: pl.DataFrame, results: dict[str, float | None], buckets: pl.DataFrame | None,
               split_ns: int) -> pl.DataFrame:
    """Per fill: markouts per contract (mo to mid, mx to the exit price on our side), ``hold``
    = settlement where settled else last recorded mid, minus price and the maker fee, ``settle``
    = the same only where settled; ``volatile`` = the fill's own 5-minute bucket (shared
    definition; evaluation split, not a decision input); ``split`` dev / holdout."""
    if fills.is_empty():
        return fills
    res = {m: results.get(m) for m in fills["market"].unique().to_list()}
    f = fills.with_columns(
        pl.col("market").replace_strict(res, default=None, return_dtype=pl.Float64).alias("settle_yes"),
        pl.col("market").str.replace(r"-[^-]+$", "").alias("cluster"),
        (pl.col("fee") / pl.col("qty")).alias("fee_pc"),
        pl.when(pl.col("t_ns") < split_ns).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("split"),
        (pl.col("t_ns") // BUCKET_NS * BUCKET_NS).alias("bucket_ns"))
    final = pl.coalesce("settle_yes", "last_mid")
    val = pl.when(pl.col("side") == "yes").then(final).otherwise(1 - final)
    sval = pl.when(pl.col("side") == "yes").then(pl.col("settle_yes")).otherwise(1 - pl.col("settle_yes"))
    f = f.with_columns((val - pl.col("price") - pl.col("fee_pc")).alias("hold"),
                       (sval - pl.col("price") - pl.col("fee_pc")).alias("settle"),
                       pl.col("settle_yes").is_not_null().alias("settled"))
    if buckets is not None and buckets.height:
        lab = buckets.select("market", "bucket_ns", pl.col("volatile").alias("_v"))
        f = f.join(lab, on=["market", "bucket_ns"], how="left").with_columns(
            pl.col("_v").fill_null(False).alias("volatile")).drop("_v")
    else:
        f = f.with_columns(pl.lit(False).alias("volatile"))
    return f


L2_METRICS = ("mo_60s", "mo_300s", "mx_60s", "mx_300s", "hold", "settle")


def l2_summary(m: pl.DataFrame) -> pl.DataFrame:
    """lane x latency x regime (all / volatile / calm) x split: fills, contracts, events and each
    metric in cents per contract with its t clustered by event."""
    rows = []
    if m.is_empty():
        return pl.DataFrame()
    for (lane, lat), g in m.group_by(["lane", "latency"], maintain_order=True):
        for regime, cond in (("all", None), ("volatile", pl.col("volatile")), ("calm", ~pl.col("volatile"))):
            gg = g if cond is None else g.filter(cond)
            for split in ("dev", "holdout"):
                x = gg.filter(pl.col("split") == split)
                r: dict[str, Any] = {"lane": lane, "latency": lat, "regime": regime, "split": split,
                                     "fills": x.height, "contracts": round(float(x["qty"].sum()), 2) if x.height else 0.0,
                                     "events": x["cluster"].n_unique() if x.height else 0,
                                     "settled_q": round(float(x.filter("settled")["qty"].sum()), 2) if x.height else 0.0}
                for col in L2_METRICS:
                    mean, t, _ = clustered(x.rename({"qty": "q"}), col, cluster="cluster") if x.height else (None, None, 0)
                    r[col] = _r(mean, 2, 100)
                    r[f"t_{col}"] = _r(t)
                r["hold_usd"] = round(float((x["qty"] * x["hold"]).sum()), 2) if x.height else 0.0
                rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None)


def run_l2(s, start: str = "2026-11-05T03", end: str = "2026-11-05T18", split: str | None = "2026-11-05T11", *,
           lo: float = 0.30, hi: float = 0.70, gate: tuple[float, float, float] = (300.0, 0.85, 10.0),
           latencies: Iterable[float] = (50.0, 100.0), fetch: bool = True, force: bool = False,
           log: Any = print) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Replay the recorded hours [start, end) (cached fills under ``data/cache``), score them and
    return (per-fill metrics, summary). ``split`` = first holdout hour (None: all holdout)."""
    from albert.research.volatile import cached_buckets

    tag = f"{start}_{end}_{lo:g}-{hi:g}_g{gate[0]:g}-{gate[1]:g}-{gate[2]:g}".replace(":", "")
    path = s.data_dir / "cache" / f"yes_bias_l2_{tag}.parquet"
    if path.exists() and not force:
        fills = pl.read_parquet(path)
    else:
        fills = replay_l2(s, start, end, lo=lo, hi=hi, gate=gate, latencies=latencies, log=log)
        path.parent.mkdir(parents=True, exist_ok=True)
        fills.write_parquet(path)
    results = market_results(fills["market"].unique().to_list(), s.data_dir / "cache" / "yes_bias_results.json",
                             fetch=fetch) if fills.height else {}
    buckets = cached_buckets(s, start, end)
    split_ns = _hour_ns(split) if split else _hour_ns(start)
    m = l2_metrics(fills, results, buckets, split_ns)
    return m, l2_summary(m)


# ---- report ------------------------------------------------------------------------------------
MID_BANDS = ("30-35", "35-40", "40-45", "45-50", "50-55", "55-60", "60-65", "65-70")
P1_COLS = ["split", "contracts", "markets", "events", "sellY_q%", "sellY_c", "sellY_t", "sellY_ew_c", "sellY_ew_t",
           "buyY_c", "buyY_t", "buyY_ew_c", "buyY_ew_t", "asym_c", "asym_t", "gap_c", "gap_t"]
SPEC = PrintParams()          # spec S2 as written: sell YES, mid 30-70c, 1 tick inside >= 2c, 300 s


def _grp_col() -> pl.Expr:
    return (pl.when(pl.col("source") == "band").then(pl.concat_str(pl.lit("band:"), pl.col("cls")))
            .otherwise(pl.col("source").cast(pl.Utf8))).alias("grp")


def _sel(t: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return t.select([c for c in keys + P1_COLS if c in t.columns]) if t.height else t


def inplay_onset(prints: Path, markets: pl.DataFrame) -> pl.DataFrame:
    """Per major league: prints per game per 15 minutes around expected_expiration - 3 h
    (the in-play sample, which covers exp - 4 h .. close)."""
    lf = (pl.scan_parquet(prints).filter(pl.col("source") == "inplay")
          .join(markets.lazy().select("mid", "series"), on="mid")
          .with_columns(((-pl.col("h_exp").cast(pl.Float64) + INPLAY_OFFSET_H) * 4).floor().alias("qh")))
    g = lf.group_by("series", "qh").agg(pl.len().alias("n")).collect()
    games = lf.group_by("series").agg(pl.col("mid").n_unique().alias("games")).collect()
    g = g.join(games, on="series").filter(pl.col("qh").is_between(-4, 3)).with_columns(
        (pl.col("n") / pl.col("games")).round(0).alias("per_game"), (pl.col("qh") / 4).alias("h_from_start"))
    return g.pivot(on="h_from_start", index=["series", "games"], values="per_game", sort_columns=True)


def p2_select(summary: pl.DataFrame, group: str, min_fills: int = 150) -> dict[str, Any] | None:
    """The sell-YES grid run with the highest dev PnL per contract in ``group`` (dev only,
    at least ``min_fills`` dev fills), with its holdout row."""
    d = summary.filter((pl.col("grp") == group) & (pl.col("split") == "dev") & (pl.col("fills") >= min_fills)
                       & pl.col("run").str.starts_with("sell_yes") & pl.col("pnl_c").is_not_null())
    if d.is_empty():
        return None
    best = d.sort("pnl_c", descending=True).row(0, named=True)
    h = summary.filter((pl.col("grp") == group) & (pl.col("run") == best["run"]) & (pl.col("split") == "holdout"))
    return {"group": group, "run": best["run"], "dev": best, "holdout": h.row(0, named=True) if h.height else None}


def report(s, *, l2: tuple[str, str, str | None] = ("2026-11-05T03", "2026-11-05T18", "2026-11-05T11"),
           rebuild: bool = False, fetch: bool = True, l2_gate: tuple[float, float, float] | None = None,
           log: Any = print) -> Path:
    """Run all three parts and write ``research-yes-bias``."""
    from albert.reports import Report

    rep = Report(s.reports_dir, "research-yes-bias")
    markets = market_frame(s)
    prints = build_prints(s, markets, force=rebuild)
    meta = json.loads(prints.with_suffix(".json").read_text())
    band_ids = pl.scan_parquet(prints).filter(pl.col("source") == "band").select("mid").unique().collect()["mid"]
    markets = add_structure(markets, markets.filter(pl.col("mid").is_in(band_ids.implode()))["event"].unique().to_list(),
                            s, fetch=fetch)
    # -- data
    src = (pl.scan_parquet(prints).join(markets.lazy().select("mid", "split", "event"), on="mid")
           .group_by("source", "phase", "split").agg(pl.len().alias("prints"), pl.col("q").sum().round(0).alias("contracts"),
                                                     pl.col("mid").n_unique().alias("markets"),
                                                     pl.col("event").n_unique().alias("events"),
                                                     pl.col("cum_ok").mean().round(3).alias("cum_ok_share"))
           .collect().sort("source", "phase", "split"))
    rep.table("Usable prints by sample (module doc: window = pre-game population, inplay = seeded game sample, "
              "band = volume-band sample with complete coverage)", src, max_rows=40)
    vol = markets.group_by("category").agg(pl.len().alias("settled_markets"),
                                           (pl.col("volume") > 200_000).mean().round(4).alias("share_gt_200k"))
    rep.json("Selection notes", {**meta, "top_volume_excluded": len(meta.get("top_volume_excluded", [])),
                                 "share of settled markets above the 200k band cap": {
                                     r["category"]: r["share_gt_200k"] for r in vol.iter_rows(named=True)}})
    rep.table("In-play onset check: prints per game per 15 min, hours from expected_expiration - 3 h",
              inplay_onset(prints, markets))
    # -- part 1
    a = aggregate(prints, markets, ["band", "source", "phase", "h_bucket"], where=pl.col("cum_ok")).with_columns(
        _grp_col(), pl.col("band").is_in(list(MID_BANDS)).alias("midband"))
    rep.text("Part 1 legend", "Cents per contract to settlement. sellY = maker selling YES (counterparty of YES "
             "takers), buyY = maker buying YES; _c contract-weighted, _ew every market weighs 1; t clustered by "
             "event. asym = sellY - buyY paired within market. gap = contract-weighted price - settlement. "
             "Maker fees (maker-fee series only) are subtracted; takers' net = -(maker) - fees.")
    rep.table("P1a. By sample/class, all prices", _sel(split_table(a, ["grp"]), ["grp"]), max_rows=40)
    rep.table("P1b. By sample/class, mid-range 30-70c", _sel(split_table(a.filter("midband"), ["grp"]), ["grp"]), max_rows=40)
    rep.table("P1c. By 5c band, all usable prints", _sel(split_table(a, ["band"]), ["band"]), max_rows=40)
    rep.table("P1d. By 5c band, single-name class (volume-band sample)",
              _sel(split_table(a.filter(pl.col("grp") == "band:single-name"), ["band"]), ["band"]), max_rows=40)
    a2 = aggregate(prints, markets, ["band", "source"], where=pl.col("cum_ok")).with_columns(
        _grp_col(), pl.col("band").is_in(list(MID_BANDS)).alias("midband"))
    rep.table("P1e. By sample/class x market structure, all prices",
              _sel(split_table(a2, ["grp", "structure"]), ["grp", "structure"]), max_rows=60)
    rep.table("P1f. By sample/class x market structure, mid-range 30-70c",
              _sel(split_table(a2.filter("midband"), ["grp", "structure"]), ["grp", "structure"]), max_rows=60)
    rep.table("P1g. By 5c band, sports multi-outcome legs (volume-band sample)",
              _sel(split_table(a2.filter((pl.col("grp") == "band:sports") & (pl.col("structure") == "multi-ME")),
                               ["band"]), ["band"]), max_rows=40)
    rep.table("P1h. By category, volume-band sample, mid-range 30-70c",
              _sel(split_table(a.filter((pl.col("source") == "band") & pl.col("midband")), ["category"]),
                   ["category"]), max_rows=40)
    rep.table("P1i. By hours to scheduled expiration, mid-range 30-70c",
              _sel(split_table(a.filter("midband"), ["grp", "h_bucket"], min_contracts=1000), ["grp", "h_bucket"]),
              max_rows=60)
    rep.table("P1j. Major-league game winners: in-play vs pre-game (expected_expiration - 3 h), all prices / mid",
              _sel(pl.concat([split_table(a.filter(pl.col("phase") != "n/a"), ["phase"]).with_columns(pl.lit("all").alias("prices")),
                              split_table(a.filter((pl.col("phase") != "n/a") & pl.col("midband")), ["phase"]).with_columns(
                                  pl.lit("30-70").alias("prices"))], how="diagonal_relaxed"), ["phase", "prices"]))
    b = aggregate(prints, markets, ["band", "cum_ok"], where=pl.col("source") == "band").with_columns(
        pl.col("band").is_in(list(MID_BANDS)).alias("midband"))
    rep.table("P1k. Volume-band look-ahead check: prints before vs after cumulative volume 500, mid-range",
              _sel(split_table(b.filter("midband"), ["cls", "cum_ok"]), ["cls", "cum_ok"]), max_rows=40)
    a3 = aggregate(prints, markets, ["source"], where=pl.col("cum_ok") & (pl.col("source") == "band"))
    big = a3.group_by("series").agg(pl.col("mid").n_unique().alias("n")).filter(pl.col("n") >= 25)["series"]
    rep.table("P1l. By series (volume-band sample, >= 25 markets), all prices",
              _sel(split_table(a3.filter(pl.col("series").is_in(big.implode())), ["series"]), ["series"]), max_rows=80)
    # -- part 2
    summ_path = s.data_dir / "cache" / "yes_bias_grid_summary.parquet"
    fills_path = s.data_dir / "cache" / "yes_bias_grid_fills.parquet"
    if summ_path.exists() and fills_path.exists() and not rebuild:
        summ, gfills = pl.read_parquet(summ_path), pl.read_parquet(fills_path)
    else:
        summ, gfills = run_grid(load_sequences(prints), markets, grid_params(), log=log)
        summ.write_parquet(summ_path)
        gfills.write_parquet(fills_path)
    keep = ["grp", "run", "split", "fills", "contracts", "markets", "events", "pnl_c", "t", "ew_c", "ew_t", "usd",
            "avg_px", "yes_rate"]
    spec = summ.filter(pl.col("run").is_in([SPEC.name(), PrintParams(side="buy_yes").name()]))
    rep.table("P2a. Trade-print maker, spec S2 as written (sell YES 1 tick inside >= 2c, mid 30-70c, ttl 300 s) and its "
              "buy-YES mirror (control)", spec.select(keep).sort("grp", "run", "split"), max_rows=40)
    chosen = [x for x in (p2_select(summ, g) for g in ["all", "band:sports", "band:weather", "band:single-name",
                                                        "window", "inplay"]) if x]
    rows = []
    for c in chosen:
        for split in ("dev", "holdout"):
            r = c[split] if split == "dev" else c["holdout"]
            if r:
                rows.append({"group": c["group"], "chosen_on_dev": c["run"], **{k: r.get(k) for k in keep[2:]}})
    rep.table("P2b. Best sell-YES grid run per group, chosen on dev (>= 150 dev fills); holdout untouched",
              pl.DataFrame(rows, infer_schema_length=None), max_rows=40)
    gf = gfills.join(markets.select("mid", "structure"), on="mid", how="left")
    st = gf.filter(pl.col("run").is_in([SPEC.name(), PrintParams(side="buy_yes").name()]))
    rows = []
    for (run, grp, stc), g in st.group_by(["run", "grp", "structure"], maintain_order=True):
        for split in ("dev", "holdout"):
            x = g.filter(pl.col("split") == split)
            rows.append({"run": run, "grp": grp, "structure": stc, "split": split, **fill_stats(x)})
    rep.table("P2c. Spec run and mirror by sample x market structure",
              pl.DataFrame(rows, infer_schema_length=None).sort("grp", "structure", "run", "split"), max_rows=80)
    sel = summ.filter((pl.col("grp") == "all") & pl.col("run").str.starts_with("sell_yes"))
    piv = sel.pivot(on="split", index="run", values=["pnl_c", "t", "fills"]).sort("pnl_c_dev", descending=True)
    rep.table("P2d. Whole sell-YES grid, all groups (dev-sorted)", piv, save_as="p2_grid", max_rows=70)
    # -- part 3
    gate = l2_gate or _l2_gate(summ)
    m, l2s = run_l2(s, l2[0], l2[1], l2[2], gate=gate, fetch=fetch, log=log)
    rep.json("P3 setup", {"hours": list(l2), "band": [0.30, 0.70], "gate (window_s, yes_share, min_contracts)": gate,
                          "lanes": {"sellY": "baseline Quoter (improve, min_spread 5c, cooldown 300 s) on the NO bid "
                                             "only, mid 30-70c", "sellY_gate": "sellY + pull while YES-taker share "
                                             "of the last window >= share", "sellY_all": "NO bid only, mid 5-95c",
                                    "buyY": "YES bid only, mid 30-70c (control)", "two_sided": "baseline Quoter"}})
    if l2s.height:
        for lat in sorted(l2s["latency"].unique().to_list()):
            rep.table(f"P3. Recorded L2 at {lat:g} ms: cents per contract (t by event); volatile = fill's own bucket",
                      l2s.filter(pl.col("latency") == lat).drop("latency"), max_rows=80)
        rep.table("P3b. Recorded L2 at 50 ms by market structure (events from public REST)",
                  l2_by_structure(m.filter(pl.col("latency") == 50.0), s, fetch=fetch), max_rows=60)
        m.write_parquet(rep.dir / "l2_fills.parquet")
    else:
        rep.text("P3", "No fills.")
    return rep.save()


def l2_by_structure(m: pl.DataFrame, s, *, fetch: bool = True) -> pl.DataFrame:
    """``l2_summary`` (regime all) per market structure (``structure_of`` from the event's
    public metadata; leg count from the event payload)."""
    if m.is_empty():
        return pl.DataFrame()
    evs = m["cluster"].unique().to_list()
    st = event_structure(evs, s.data_dir / "cache" / "yes_bias_events.json", fetch=fetch)
    lab = {e: structure_of((v or {}).get("n"), (v or {}).get("me"), e.split("-", 1)[0]) for e, v in st.items()}
    x = m.with_columns(pl.col("cluster").replace_strict(lab, default="unknown", return_dtype=pl.Utf8).alias("structure"))
    out = []
    for stc in sorted(x["structure"].unique().to_list()):
        t = l2_summary(x.filter(pl.col("structure") == stc))
        out.append(t.filter(pl.col("regime") == "all").with_columns(pl.lit(stc).alias("structure")))
    keep = ["structure", "lane", "split", "fills", "events", "settled_q", "mo_300s", "t_mo_300s", "mx_300s",
            "t_mx_300s", "hold", "t_hold", "settle", "t_settle", "hold_usd"]
    return pl.concat(out, how="diagonal_relaxed").select(keep).sort("structure", "lane", "split")


def _l2_gate(summary: pl.DataFrame) -> tuple[float, float, float]:
    """The flow gate of the best sell-YES mid-band (30-70c) grid run on dev (all groups)."""
    d = summary.filter((pl.col("grp") == "all") & (pl.col("split") == "dev") & pl.col("run").str.starts_with("sell_yes")
                       & pl.col("run").str.contains("mid0.3-0.7") & pl.col("run").str.contains("gate"))
    if d.is_empty():
        return GRID_GATES[2]
    p = json.loads(d.sort("pnl_c", descending=True).row(0, named=True)["params"])
    return (p["gate_window_s"], p["gate_share"], p["gate_min_q"])


def report_l2(s, start: str, end: str, split: str | None = None, *,
              gate: tuple[float, float, float] | None = None, fetch: bool = True, log: Any = print) -> Path:
    """Part 3 alone on other recorded hours (e.g. tonight's in-play games:
    ``report_l2(settings(), '2026-11-05T23', '2026-11-06T07')``, all holdout). The gate defaults
    to the one chosen on the historical dev grid."""
    from albert.reports import Report

    summ_path = s.data_dir / "cache" / "yes_bias_grid_summary.parquet"
    gate = gate or (_l2_gate(pl.read_parquet(summ_path)) if summ_path.exists() else GRID_GATES[2])
    m, l2s = run_l2(s, start, end, split, gate=gate, fetch=fetch, log=log)
    rep = Report(s.reports_dir, "research-yes-bias-l2")
    rep.json("Setup", {"hours": [start, end, split], "gate": gate})
    for lat in sorted(l2s["latency"].unique().to_list()) if l2s.height else []:
        rep.table(f"Recorded L2 at {lat:g} ms (cents per contract, t by event)",
                  l2s.filter(pl.col("latency") == lat).drop("latency"), max_rows=80)
    if m.height:
        m.write_parquet(rep.dir / "l2_fills.parquet")
    return rep.save()
