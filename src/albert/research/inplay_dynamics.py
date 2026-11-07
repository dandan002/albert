"""In-play price dynamics of game-winner markets (Kalshi trade prints), fitted in logit space.

Data. ``kalshi_trades`` holds, for one leg of every game, only the *pre-game* window
[expected_expiration - 25 h, expected_expiration - 4 h] (``kalshi_trade_windows``). Kalshi's
``expected_expiration_time`` is about kickoff + 3 h for most games (so that window ends about an
hour before the start) and kickoff + 6 h for some NFL games, so the table has almost no in-play
trades. ``fetch_inplay`` completes a seeded, volume-blind sample of games (the same first-
alphabetical leg the window ingest used) from the window end to the market close with the public
trades endpoint; pages are cached append-only under ``data/cache/inplay_dynamics_raw`` and parsed
with the venue parser (``kalshi.trade_rows``).

Game start. ``sched`` = occurrence_time - 3 h (``scheduled_start``; for MLB, whose tickers
carry the scheduled start, it matches exactly for 597 of 599 games; a rewritten occurrence with
non-round seconds is ignored). Two causal detectors are reported against it (``onsets``): the
print-rate onset (first minute whose trailing 10-minute count is >= 3x the median of the previous
two hours) is poor (NFL pre-game hours trade as much as the game), while the volatility onset
(first 3c mid move within 5 minutes) lands a few minutes after ``sched``. Trading rules use games
whose volatility onset is at most 90 minutes after ``sched`` (known by then; drops postponements)
and phase = (t - sched) / L, L = dev median of close - sched per sport (a clock scale only).
Descriptive fits use time remaining to the close (ex post) and the ``on_schedule`` filter
(0.6 L < close - sched < 1.6 L). Every game stops being traded at its first print in the decided
zone (mid <= 2c or >= 98c), which is observable at that moment; post-final 99c/1c prints are
therefore never traded on.

Prices. Prints with the same timestamp and taker side are one sweep (their order inside it is
not recoverable) and are grouped (``prints_frame``). ``mid`` at a group is the average of the last
YES-taker level (an ask) and the last NO-taker level (a bid) when both are at most 120 s old;
otherwise null. Model fits use ``mid`` and drop the decided zone.

Execution (taker). A decision at time t buys YES at the first later YES-taker group at time
> t + 2 s (never the trigger), at its best level; NO symmetrically. Exit at time X: the first
opposite-side group at or after X (the bid we would hit); if none before the close, the contract
settles. Fees: taker 0.07 * mult * P(1 - P) per contract on entry and on a taker exit (exact,
account precision $0.0001; mult from the market, 0.5 for MLB); maker 0.0175 * mult * P(1 - P) on
maker-fee series. Trade-print maker: rest one contract X cents from the last print; it fills only
when a later print trades strictly through our price within the order lifetime.

Split: dev = games with expected_expiration_time before 2026-10-01, holdout = October. t-stats are
clustered by game.
"""

import itertools
import math
import random
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

GAME_SERIES = ("KXNFLGAME", "KXNCAAFGAME", "KXMLBGAME", "KXNBAGAME", "KXNHLGAME", "KXWNBAGAME")
SPORT = {"KXNFLGAME": "NFL", "KXNCAAFGAME": "NCAAF", "KXMLBGAME": "MLB", "KXNBAGAME": "NBA",
         "KXNHLGAME": "NHL", "KXWNBAGAME": "WNBA"}
HOLDOUT_FROM = datetime.fromisoformat("2026-10-01T00:00:00+00:00")
# Games sampled per (series, split), volume-blind and seeded. NBA has no games before October.
SAMPLE = {("KXNFLGAME", "dev"): 30, ("KXNFLGAME", "holdout"): 30,
          ("KXNBAGAME", "dev"): 0, ("KXNBAGAME", "holdout"): 60,
          ("KXMLBGAME", "dev"): 60, ("KXMLBGAME", "holdout"): 35,
          ("KXNCAAFGAME", "dev"): 100, ("KXNCAAFGAME", "holdout"): 100,
          ("KXNHLGAME", "dev"): 40, ("KXNHLGAME", "holdout"): 60,
          ("KXWNBAGAME", "dev"): 50, ("KXWNBAGAME", "holdout"): 16}
SCHED_OFFSET = timedelta(hours=3)     # sched = expected_expiration_time - 3 h
ONSET_RATIO = 3.0
ONSET_MIN = 30
DECIDED_LO, DECIDED_HI = 0.02, 0.98
MID_MAX_AGE_S = 120.0
ENTRY_DELAY_S = 2.0
TAKER, MAKER = 0.07, 0.0175

CACHE_NAME = "inplay_dynamics"


# ---- sample and fetch ---------------------------------------------------------------------------
def select_games(markets: pl.DataFrame, windows: pl.DataFrame, sample: dict | None = None,
                 seed: int = 7) -> pl.DataFrame:
    """One leg per settled game (first alphabetically, the leg the window ingest fetched),
    seeded random sample per (series, split). Volume and outcome are never looked at."""
    sample = SAMPLE if sample is None else sample
    m = (markets.filter(pl.col("series_ticker").is_in(list(GAME_SERIES))
                        & pl.col("result").is_in(["yes", "no"])
                        & pl.col("expected_expiration_time").is_not_null())
         .sort("ticker").unique("event_ticker", keep="first", maintain_order=True)
         .with_columns(pl.when(pl.col("expected_expiration_time") >= HOLDOUT_FROM)
                       .then(pl.lit("holdout")).otherwise(pl.lit("dev")).alias("split")))
    m = m.join(windows.select("ticker", "window_start", "window_end"), on="ticker", how="left")
    picked = []
    for (series, split), n in sorted(sample.items()):
        pool = m.filter((pl.col("series_ticker") == series) & (pl.col("split") == split)
                        & pl.col("window_end").is_not_null())["ticker"].sort().to_list()
        rng = random.Random(f"{seed}-{series}-{split}")
        picked += rng.sample(pool, min(n, len(pool)))
    return m.filter(pl.col("ticker").is_in(picked)).sort("ticker")


def fetch_window(row: dict[str, Any]) -> tuple[datetime, datetime] | None:
    """[window_end, close + 2 min]: the part of the game the pre-game window ingest missed."""
    lo, hi = row["window_end"], row["close_time"] + timedelta(minutes=2)
    return (lo, hi) if lo is not None and hi > lo else None


def fetch_inplay(s, games: pl.DataFrame, *, workers: int = 3, min_interval: float = 0.25,
                 progress: bool = True) -> Path:
    """Fetch trades in ``fetch_window`` for each game into ``data/cache/inplay_dynamics_trades/
    <ticker>.parquet`` (skipped when present, so the fetch resumes). Raw pages are cached
    append-only in ``data/cache/inplay_dynamics_raw``; ``data/raw`` is not touched."""
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    out = s.data_dir / "cache" / f"{CACHE_NAME}_trades"
    out.mkdir(parents=True, exist_ok=True)
    raw = s.data_dir / "cache" / f"{CACHE_NAME}_raw"
    cutoff = k.Kalshi(CachedClient("kalshi", s.kalshi_base, raw, record_live=False)).cutoff()
    todo = [r for r in games.iter_rows(named=True) if not (out / f"{r['ticker']}.parquet").exists()]

    def work(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        client = k.Kalshi(CachedClient("kalshi", s.kalshi_base, raw, min_interval=min_interval,
                                       record_live=False, max_retries=8))
        failed = []
        for n, r in enumerate(rows, 1):
            win = fetch_window(r)
            got: list[dict] = []
            try:
                if win is not None:
                    hist = bool(cutoff["trades_created_ts"] and r["close_time"] < cutoff["trades_created_ts"])
                    for page in client.trades(r["ticker"], hist, min_ts=win[0], max_ts=win[1]):
                        got.extend(k.trade_rows(page.payload))
            except RuntimeError as error:      # rate limited past the retries: try again later
                print(f"  retry later {r['ticker']}: {str(error)[:120]}", flush=True)
                failed.append(r)
                continue
            df = pl.DataFrame(got, schema=k.TRADE_SCHEMA)
            tmp = out / f"{r['ticker']}.parquet.tmp"
            df.write_parquet(tmp)
            tmp.replace(out / f"{r['ticker']}.parquet")
            if progress and n % 10 == 0:
                print(f"  fetched {n}/{len(rows)} in one worker", flush=True)
        return failed

    for attempt in range(3):
        chunks = [todo[i::workers] for i in range(workers)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            todo = [r for part in ex.map(work, chunks) for r in part]
        if not todo:
            break
    if progress:
        print(f"fetch finished into {out}; {len(todo)} games still missing", flush=True)
    return out


# ---- per-game print files -------------------------------------------------------------------------
def prints_frame(trades: pl.DataFrame) -> pl.DataFrame:
    """Raw trade rows -> one row per taker group: all prints with the same timestamp and taker
    side (one order sweeping several levels; their order inside the group is not recoverable).
    ts, side (+1 YES taker, -1 NO taker), pmin/pmax (YES prices), n (contracts), k (prints).
    Deduplicated by trade_id and sorted by time."""
    return (trades.filter(pl.col("yes_price").is_not_null() & pl.col("taker_side").is_in(["yes", "no"]))
            .unique("trade_id")
            .with_columns(pl.when(pl.col("taker_side") == "yes").then(1).otherwise(-1).cast(pl.Int8).alias("side"))
            .group_by(pl.col("created_time").alias("ts"), "side")
            .agg(pl.col("yes_price").min().alias("pmin"), pl.col("yes_price").max().alias("pmax"),
                 pl.col("count").sum().alias("n"), pl.len().alias("k"))
            .sort("ts", "side"))


def build_game_files(s, games: pl.DataFrame, batch: int = 80) -> Path:
    """``data/cache/inplay_dynamics_games/<ticker>.parquet``: the table's pre-game window plus the
    fetched rest of the game, for every game whose fetch has finished."""
    out = s.data_dir / "cache" / f"{CACHE_NAME}_games"
    fetched = s.data_dir / "cache" / f"{CACHE_NAME}_trades"
    out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in games["ticker"].to_list()
            if (fetched / f"{t}.parquet").exists() and not (out / f"{t}.parquet").exists()]
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        pre = (pl.scan_parquet(s.tables_dir / "kalshi_trades.parquet")
               .filter(pl.col("ticker").is_in(part)).collect())
        groups = {k[0]: v for k, v in pre.partition_by("ticker", as_dict=True).items()}
        for t in part:
            got = pl.read_parquet(fetched / f"{t}.parquet")
            both = pl.concat([groups.get(t, got.clear()), got], how="vertical_relaxed")
            tmp = out / f"{t}.parquet.tmp"
            prints_frame(both).write_parquet(tmp)
            tmp.replace(out / f"{t}.parquet")
    return out


# ---- game metadata ----------------------------------------------------------------------------------
def scheduled_start(occurrence: datetime | None, expiration: datetime | None) -> datetime | None:
    """Scheduled start = occurrence_time - 3 h (MLB: equals the start in the ticker for 597 of 599
    games). Kalshi sometimes rewrites occurrence_time to the actual end (non-round seconds) or sets
    expected_expiration = occurrence + 3 h; a rewritten occurrence is not used (look-ahead)."""
    occ_ok = (occurrence is not None and occurrence.second == 0 and occurrence.microsecond == 0
              and (expiration is None or occurrence <= expiration))
    base = occurrence if occ_ok else expiration
    return None if base is None else base - SCHED_OFFSET


def game_meta(games: pl.DataFrame) -> pl.DataFrame:
    """sched, sport, outcome, fee terms and the nominal game length L (dev median of close - sched,
    used only as a clock scale for the causal phase)."""
    g = games.with_columns(
        pl.struct("occurrence_time", "expected_expiration_time").map_elements(
            lambda r: scheduled_start(r["occurrence_time"], r["expected_expiration_time"]),
            return_dtype=pl.Datetime("us", "UTC")).alias("sched"),
        pl.col("series_ticker").replace_strict(SPORT, default=None).alias("sport"),
        (pl.col("result") == "yes").cast(pl.Float64).alias("y"),
        pl.col("fee_multiplier").fill_null(1.0).alias("mult"),
        pl.col("fee_type").fill_null("quadratic").str.contains("maker").alias("maker_fee"),
    ).with_columns(((pl.col("close_time") - pl.col("sched")).dt.total_seconds() / 60).alias("len_min"))
    lengths = (g.filter(pl.col("split") == "dev").group_by("sport").agg(pl.col("len_min").median().alias("L")))
    # NBA has no dev games: its clock scale comes from all of its games (a length, not an outcome).
    missing = g.filter(~pl.col("sport").is_in(lengths["sport"].implode())).group_by("sport").agg(
        pl.col("len_min").median().alias("L"))
    g = g.join(pl.concat([lengths, missing]), on="sport", how="left")
    # Postponed or abandoned games: the market closed far from the scheduled clock.
    return g.with_columns(((pl.col("len_min") > 0.6 * pl.col("L")) & (pl.col("len_min") < 1.6 * pl.col("L")))
                          .alias("on_schedule"))


# ---- print-level primitives ---------------------------------------------------------------------------
def mid_proxy(t: np.ndarray, p: np.ndarray, side: np.ndarray, max_age: float = MID_MAX_AGE_S) -> np.ndarray:
    """Average of the last YES-taker price (an ask) and the last NO-taker price (a bid), both at
    most ``max_age`` seconds old at each print; NaN otherwise. Uses prints up to and including i."""
    i = np.arange(len(t))
    ia = np.maximum.accumulate(np.where(side > 0, i, -1)) if len(t) else i
    ib = np.maximum.accumulate(np.where(side < 0, i, -1)) if len(t) else i
    a0, b0 = np.maximum(ia, 0), np.maximum(ib, 0)
    ok = (ia >= 0) & (ib >= 0) & (t - t[a0] <= max_age) & (t - t[b0] <= max_age) if len(t) else ia > 0
    return np.where(ok, 0.5 * (p[a0] + p[b0]), np.nan) if len(t) else np.zeros(0)


class Path_:
    """Taker groups of one game with as-of lookups (only groups at or before the query time).

    ``best`` is the group's first fill (the best price the taker got: lowest YES price for a YES
    taker, highest for a NO taker): the price a 1-lot taker arriving before that group pays.
    ``p`` is the group's last level (the book after the sweep), used for the mid proxy and as the
    "last print"."""

    def __init__(self, t: np.ndarray, side: np.ndarray, pmin: np.ndarray, pmax: np.ndarray,
                 max_age: float = MID_MAX_AGE_S):
        self.t, self.side, self.pmin, self.pmax = t, side, pmin, pmax
        self.best = np.where(side > 0, pmin, pmax)
        self.p = np.where(side > 0, pmax, pmin)
        self.mid = mid_proxy(t, self.p, side, max_age)
        i = np.arange(len(t))
        self.lv = np.maximum.accumulate(np.where(~np.isnan(self.mid), i, -1)) if len(t) else i
        self.t_yes, self.i_yes = t[side > 0], np.flatnonzero(side > 0)
        self.t_no, self.i_no = t[side < 0], np.flatnonzero(side < 0)
        self.ia = np.maximum.accumulate(np.where(side > 0, i, -1)) if len(t) else i
        self.ib = np.maximum.accumulate(np.where(side < 0, i, -1)) if len(t) else i

    def fresh_mid_at(self, when: float, max_age: float = 15.0) -> float:
        """Mid from the last YES-taker and NO-taker prints at or before ``when`` only if both are at
        most ``max_age`` seconds old (guards against a stale side faking a continuation)."""
        j = int(np.searchsorted(self.t, when, side="right")) - 1
        if j < 0:
            return math.nan
        a, b = self.ia[j], self.ib[j]
        if a < 0 or b < 0 or when - self.t[a] > max_age or when - self.t[b] > max_age:
            return math.nan
        return 0.5 * (self.p[a] + self.p[b])

    def mid_at(self, when: np.ndarray, max_stale: float = 300.0) -> np.ndarray:
        """Last valid mid at or before each time (NaN if none within ``max_stale`` seconds)."""
        when = np.asarray(when, dtype=float)
        j = np.searchsorted(self.t, when, side="right") - 1
        k = np.where(j >= 0, self.lv[np.maximum(j, 0)], -1)
        ok = (k >= 0) & (when - self.t[np.maximum(k, 0)] <= max_stale)
        return np.where(ok, self.mid[np.maximum(k, 0)], np.nan)

    def next_print(self, taker: int, after: float, *, strict: bool = True, until: float = math.inf) -> int:
        """Index of the first print with this taker side at time > after (>= when not strict)."""
        ts, ix = (self.t_yes, self.i_yes) if taker > 0 else (self.t_no, self.i_no)
        k = np.searchsorted(ts, after, side="right" if strict else "left")
        if k >= len(ts) or ts[k] > until:
            return -1
        return int(ix[k])


def taker_fee(price: float, mult: float) -> float:
    return TAKER * mult * price * (1 - price)


def maker_fee(price: float, mult: float, on: bool) -> float:
    return MAKER * mult * price * (1 - price) if on else 0.0


def taker_round_trip(path: Path_, buy_yes: bool, t_dec: float, horizons: Iterable[float], y: float,
                     mult: float, close: float, max_wait: float = 30.0) -> dict[str, Any] | None:
    """Buy one contract at the first same-direction taker print after t_dec + 2 s (within
    ``max_wait`` s), then for each horizon h sell at the first opposite-direction taker print at or
    after entry + h, or settle. Returns entry price/fee and per-horizon net PnL; h=inf = settlement."""
    j = path.next_print(1 if buy_yes else -1, t_dec + ENTRY_DELAY_S, until=t_dec + ENTRY_DELAY_S + max_wait)
    if j < 0:
        return None
    px = path.best[j] if buy_yes else 1 - path.best[j]
    if not 0 < px < 1:
        return None
    fee_in = taker_fee(px, mult)
    payoff = y if buy_yes else 1 - y
    out: dict[str, Any] = {"t_entry": path.t[j], "entry": px, "fee_in": fee_in}
    for h in horizons:
        if math.isinf(h):
            out["pnl_settle"] = payoff - px - fee_in
            out["gross_settle"] = payoff - px
            continue
        k = path.next_print(-1 if buy_yes else 1, path.t[j] + h, strict=False, until=close)
        if k < 0:
            val, fee_out = payoff, 0.0
        else:
            val = path.best[k] if buy_yes else 1 - path.best[k]
            fee_out = taker_fee(val, mult)
        out[f"pnl_{int(h)}"] = val - px - fee_in - fee_out
        out[f"gross_{int(h)}"] = val - px
    return out


def maker_round_trip(path: Path_, buy_yes: bool, level: float, t0: float, life: float,
                     horizons: Iterable[float], y: float, mult: float, maker_on: bool,
                     close: float) -> dict[str, Any] | None:
    """A resting order to buy YES at ``level`` (or to sell YES at ``level`` = buy NO at 1 - level),
    live from t0 + 2 s for ``life`` seconds. It fills only if a later print trades strictly through
    the level. Exit as a taker after each horizon, or settle. None = not filled."""
    lo = np.searchsorted(path.t, t0 + ENTRY_DELAY_S, side="right")
    hi = np.searchsorted(path.t, min(t0 + ENTRY_DELAY_S + life, close), side="right")
    if hi <= lo:
        return None
    hit = path.pmin[lo:hi] < level - 1e-9 if buy_yes else path.pmax[lo:hi] > level + 1e-9
    if not hit.any():
        return None
    f = lo + int(np.argmax(hit))
    px = level if buy_yes else 1 - level
    fee_in = maker_fee(px, mult, maker_on)
    payoff = y if buy_yes else 1 - y
    out: dict[str, Any] = {"t_fill": path.t[f], "entry": px, "fee_in": fee_in}
    for h in horizons:
        if math.isinf(h):
            out["pnl_settle"] = payoff - px - fee_in
            continue
        th = path.t[f] + h
        m = payoff if th >= close else float(path.mid_at(np.array([th]), math.inf)[0])
        if not math.isnan(m):   # markout to the mid (before the cost of exiting)
            out[f"mo_{int(h)}"] = (m if buy_yes or th >= close else 1 - m) - px - fee_in
        k = path.next_print(-1 if buy_yes else 1, path.t[f] + h, strict=False, until=close)
        if k < 0:
            val, fee_out = payoff, 0.0
        else:
            val = path.best[k] if buy_yes else 1 - path.best[k]
            fee_out = taker_fee(val, mult)
        out[f"pnl_{int(h)}"] = val - px - fee_in - fee_out
    return out


def jump_events(path: Path_, k: float, t_from: float, t_to: float, window: float = 60.0,
                refractory: float = 60.0, lo: float = 0.05, hi: float = 0.95) -> list[tuple[int, float, int]]:
    """Prints where the mid moved >= k within ``window`` seconds: (index, mid ``window`` s earlier,
    sign). The first crossing starts an event; the next one needs ``refractory`` seconds. Both the
    earlier mid and the current mid must be inside [lo, hi]. Uses only prints up to the event."""
    t, mid = path.t, path.mid
    a, b = np.searchsorted(t, t_from, side="left"), np.searchsorted(t, t_to, side="left")
    if b <= a:
        return []
    ref = path.mid_at(t[a:b] - window, max_stale=window + 120.0)
    d = mid[a:b] - ref
    ok = (np.abs(d) >= k - 1e-9) & (ref >= lo) & (ref <= hi) & (mid[a:b] >= lo) & (mid[a:b] <= hi)
    out, last = [], -math.inf
    for c in np.flatnonzero(ok):
        i = a + int(c)
        if t[i] - last >= refractory:
            out.append((i, float(ref[c]), 1 if d[c] > 0 else -1))
            last = t[i]
    return out


# ---- per-game analysis ----------------------------------------------------------------------------------
KS = (0.03, 0.05, 0.10)
HORIZONS = (60.0, 300.0, 900.0, math.inf)
MAKER_X = (0.01, 0.02, 0.03, 0.05)
MAKER_LIFE = (60.0, 300.0)
LATE_PHASES = (0.5, 0.67, 0.8)
PRE_MIN = 120          # grid starts this many minutes before the scheduled start


def phase_name(e_min: float, length: float) -> str:
    if e_min < 0:
        return "pre"
    f = e_min / length
    return "early" if f < 1 / 3 else "mid" if f < 2 / 3 else "late" if f < 1 else "over"


def onsets(counts: np.ndarray, mids: np.ndarray, m0: int, ratio: float = ONSET_RATIO,
           min_n: int = ONSET_MIN, move: float = 0.03) -> tuple[float | None, float | None]:
    """Causal start detectors on the 1-minute grid (index 0 = minute ``m0`` relative to sched).

    rate: first minute >= -120 whose trailing 10-minute print count >= max(min_n, ratio x the
    median 10-minute count over the two hours before that window). vol: first minute >= -120 whose
    mid moved >= ``move`` over the last 5 minutes. Returns minutes relative to sched."""
    c10 = np.convolve(counts, np.ones(10), mode="full")[:len(counts)]
    rate = vol = None
    for i in range(len(counts)):
        m = m0 + i
        if m < -120:
            continue
        if rate is None and i >= 130:
            base = np.median([c10[i - 10 - 10 * b] for b in range(12)])
            if c10[i] >= max(min_n, ratio * base):
                rate = float(m)
        if (vol is None and i >= 5 and not (np.isnan(mids[i]) or np.isnan(mids[i - 5]))
                and abs(mids[i] - mids[i - 5]) >= move - 1e-9):
            vol = float(m)
        if rate is not None and vol is not None:
            break
    return rate, vol


def analyze_game(meta: dict[str, Any], prints: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Every per-game product: 1-minute grid, start detectors, jump events with reactions and taker
    round trips (in play and a pre-game control), trade-print maker fills, late-game snapshots."""
    out: dict[str, list[dict[str, Any]]] = {"grid": [], "events": [], "maker": [], "maker_n": [],
                                            "late": [], "game": []}
    # no prints, or the market closed before the scheduled start (cancelled / rescheduled early)
    if prints.is_empty() or meta["sched"] is None or meta["close_time"] <= meta["sched"]:
        return out
    t = (prints["ts"].dt.epoch("us").to_numpy() / 1e6).astype(float)
    path = Path_(t, prints["side"].to_numpy(), prints["pmin"].to_numpy().astype(float),
                 prints["pmax"].to_numpy().astype(float))
    sched = meta["sched"].timestamp()
    close = meta["close_time"].timestamp()
    y, mult, mk, length = meta["y"], meta["mult"], bool(meta["maker_fee"]), float(meta["L"])
    base = {"ticker": meta["ticker"], "sport": meta["sport"], "split": meta["split"]}
    # decided: first print at/after the scheduled start whose mid is in the decided zone (causal)
    a = np.searchsorted(t, sched)
    dz = np.flatnonzero((path.mid[a:] <= DECIDED_LO) | (path.mid[a:] >= DECIDED_HI))
    t_dec = float(t[a + dz[0]]) if len(dz) else math.inf
    stop = min(t_dec, close)

    # 1-minute grid
    m0 = -PRE_MIN - 140
    mins = np.arange(m0, math.ceil((close - sched) / 60) + 1)
    g = sched + 60.0 * mins
    mid_g = path.mid_at(g, 300.0)
    mid_ff = path.mid_at(g, math.inf)
    cnt = np.diff(np.searchsorted(t, np.concatenate([[g[0] - 60.0], g]), side="right"))
    last_i = np.searchsorted(t, g, side="right") - 1
    rate_on, vol_on = onsets(cnt, mid_ff, int(m0))
    for m, gg, md, mf, c in zip(mins, g, mid_g, mid_ff, cnt):
        if m < -PRE_MIN:
            continue
        out["grid"].append({**base, "e": int(m), "tau": (close - gg) / 60, "mid": None if np.isnan(md) else float(md),
                            "mid_ff": None if np.isnan(mf) else float(mf), "prints": int(c),
                            "decided": bool(gg >= t_dec), "phase": phase_name(float(m), length)})
    out["game"].append({**base, "y": y, "L": length, "len_min": (close - sched) / 60, "prints": len(t),
                        "prints_inplay": int(np.searchsorted(t, close) - a), "decided_e": (t_dec - sched) / 60
                        if math.isfinite(t_dec) else None, "rate_onset": rate_on, "vol_onset": vol_on,
                        "first_print_e": (t[0] - sched) / 60, "on_schedule": bool(meta["on_schedule"])})

    # jump events: in play [sched, stop) and a pre-game control [sched - 6 h, sched - 10 min)
    for window, (t0, t1) in (("inplay", (sched, stop)), ("pregame", (sched - 6 * 3600, sched - 600))):
        for k in KS:
            for i, ref, sgn in jump_events(path, k, t0, t1):
                e_min = (t[i] - sched) / 60
                mid = float(path.mid[i])
                ev = {**base, "window": window, "k": k, "e": e_min, "phase": phase_name(e_min, length),
                      "ref": ref, "mid": mid, "sign": sgn, "last": float(path.p[i])}
                for h in HORIZONS[:-1]:
                    th = t[i] + h
                    v = y if th >= close else float(path.mid_at(np.array([th]), math.inf)[0])
                    ev[f"r_{int(h)}"] = sgn * (v - mid)
                ev["r_settle"] = sgn * (y - mid)
                f0 = path.fresh_mid_at(float(t[i]))
                for h in HORIZONS[:-1]:
                    th = t[i] + h
                    v = y if th >= close else path.fresh_mid_at(th)
                    ev[f"rf_{int(h)}"] = sgn * (v - f0) if not (math.isnan(f0) or math.isnan(v)) else None
                for name, buy_yes in (("follow", sgn > 0), ("fade", sgn < 0)):
                    rt = taker_round_trip(path, buy_yes, float(t[i]), HORIZONS, y, mult, close)
                    row = {**ev, "dir": name, "filled": rt is not None}
                    if rt is not None:
                        rt.pop("t_entry")
                        row.update(rt)
                    out["events"].append(row)
                if window == "inplay" and k == 0.05:
                    _maker(out, path, base, "jump", float(t[i]), float(path.p[i]), sgn, e_min, length,
                           y, mult, mk, close)
    # unconditional in-play maker every 5 minutes
    for m, gg, li in zip(mins, g, last_i):
        if m >= 0 and m % 5 == 0 and gg < stop and li >= 0:
            _maker(out, path, base, "grid5", float(gg), float(path.p[li]), 0, float(m), length, y, mult,
                   mk, close)
    # late-game snapshots: first minute at phase >= f (not decided), buy favourite / underdog
    for f in LATE_PHASES:
        idx = np.flatnonzero((mins >= f * length) & (g < stop) & ~np.isnan(mid_g))
        if not len(idx):
            continue
        j = idx[0]
        md = float(mid_g[j])
        fav_yes = md >= 0.5
        row = {**base, "f": f, "e": float(mins[j]), "mid": md, "q": max(md, 1 - md), "fav_win": y if fav_yes else 1 - y}
        for name, buy_yes in (("favourite", fav_yes), ("underdog", not fav_yes)):
            rt = taker_round_trip(path, buy_yes, float(g[j]), (math.inf,), y, mult, close)
            if rt is not None:
                out["late"].append({**row, "dir": name, "entry": rt["entry"], "fee_in": rt["fee_in"],
                                    "pnl_settle": rt["pnl_settle"], "gross_settle": rt["gross_settle"]})
    return out


def _maker(out, path: Path_, base: dict, trigger: str, t0: float, last: float, sgn: int, e_min: float,
           length: float, y: float, mult: float, mk: bool, close: float) -> None:
    if not 0.05 <= last <= 0.95:
        return
    for x in MAKER_X:
        for life in MAKER_LIFE:
            for side, buy_yes, level in (("bid", True, round(last - x, 2)), ("ask", False, round(last + x, 2))):
                if not 0.01 <= level <= 0.99:
                    continue
                # relative to the jump: "with" = the order that fills if the move continues
                rel = "none" if sgn == 0 else ("with" if (sgn > 0) != buy_yes else "against")
                key = {**base, "trigger": trigger, "x": x, "life": life, "side": side, "rel": rel,
                       "phase": phase_name(e_min, length)}
                out["maker_n"].append(key)
                rt = maker_round_trip(path, buy_yes, level, t0, life, HORIZONS, y, mult, mk, close)
                if rt is not None:
                    rt.pop("t_fill")
                    out["maker"].append({**key, "e": e_min, **rt})


def run_games(s, meta: pl.DataFrame, *, limit: int | None = None, progress: bool = True) -> dict[str, pl.DataFrame]:
    """Analyse every game with a print file; returns (and caches) the stacked per-game products."""
    src = s.data_dir / "cache" / f"{CACHE_NAME}_games"
    acc: dict[str, list[dict[str, Any]]] = {}
    rows = [r for r in meta.iter_rows(named=True) if (src / f"{r['ticker']}.parquet").exists()]
    for n, r in enumerate(rows[:limit] if limit else rows, 1):
        res = analyze_game(r, pl.read_parquet(src / f"{r['ticker']}.parquet"))
        for key, val in res.items():
            acc.setdefault(key, []).extend(val)
        if progress and n % 50 == 0:
            print(f"  analysed {n}/{len(rows)} games", flush=True)
    frames = {k: pl.DataFrame(v, infer_schema_length=None) for k, v in acc.items()}
    for k, f in frames.items():
        if limit is None:
            f.write_parquet(s.data_dir / "cache" / f"{CACHE_NAME}_{k}.parquet")
    return frames


# ---- statistics ---------------------------------------------------------------------------------------
def cl(f: pl.DataFrame, col: str, cluster: str = "ticker") -> dict[str, Any]:
    """Mean of ``col`` with a t-stat clustered by game (null rows dropped)."""
    f = f.filter(pl.col(col).is_not_null() & pl.col(col).is_not_nan())
    if f.is_empty():
        return {"n": 0, "games": 0, "mean": None, "t": None}
    g = f.group_by(cluster).agg(pl.col(col).sum().alias("s"), pl.len().alias("w"))
    n, k = int(g["w"].sum()), g.height
    mean = float(g["s"].sum()) / n
    t = None
    if k > 1:
        se = math.sqrt(k / (k - 1) * float(((g["s"] - mean * g["w"]) ** 2).sum())) / n
        t = mean / se if se > 0 else None
    return {"n": n, "games": k, "mean": mean, "t": t}


def _c(d: dict[str, Any], name: str) -> dict[str, Any]:
    """Cents + t, prefixed."""
    return {f"{name}_c": None if d["mean"] is None else round(100 * d["mean"], 3),
            f"t_{name}": None if d["t"] is None else round(d["t"], 2)}


def z_null(f: pl.DataFrame, col: str, pcol: str, cluster: str = "ticker") -> float | None:
    """z-stat of sum(col) against the null that prices are fair, for settlement outcomes: each
    contract bought at p pays 1 with probability p (variance p(1 - p)); contracts in one game are
    taken as perfectly correlated (conservative). Unlike the clustered t, it does not explode when
    every sampled favourite happened to win."""
    f = f.filter(pl.col(col).is_not_null() & pl.col(pcol).is_not_null())
    if f.is_empty():
        return None
    g = f.group_by(cluster).agg(pl.col(col).sum().alias("s"),
                                (pl.col(pcol) * (1 - pl.col(pcol))).sqrt().sum().alias("sd"))
    v = float((g["sd"] ** 2).sum())
    return float(g["s"].sum()) / math.sqrt(v) if v > 0 else None


def splits(f: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    """all, dev, holdout (always, possibly empty) and any other split label present (stream)."""
    names = ["dev", "holdout"] + sorted(set(f["split"].unique().to_list()) - {"dev", "holdout"}) if f.height else []
    return [("all", f)] + [(n, f.filter(pl.col("split") == n)) for n in names]


def _logit(p: pl.Expr) -> pl.Expr:
    return (p / (1 - p)).log()


def _probit(p: np.ndarray) -> np.ndarray:
    from statistics import NormalDist
    nd = NormalDist()
    return np.array([nd.inv_cdf(v) if 0 < v < 1 else np.nan for v in p])


def increments(grid: pl.DataFrame, games: pl.DataFrame, step: int = 1, lo: float = 0.05,
               hi: float = 0.95) -> pl.DataFrame:
    """In-play increments of the mid on a ``step``-minute grid (on-schedule games, not decided,
    both ends inside [lo, hi]): dp, dlogit, dprobit, tau (minutes to close at the start), e, phase."""
    ok = games.filter(pl.col("on_schedule"))["ticker"].implode()
    g = (grid.filter(pl.col("ticker").is_in(ok) & (pl.col("e") >= 0) & (pl.col("e") % step == 0))
         .sort("ticker", "e")
         .with_columns(pl.col("mid").shift(-1).over("ticker").alias("mid1"),
                       pl.col("e").shift(-1).over("ticker").alias("e1"),
                       pl.col("decided").shift(-1).over("ticker").alias("dec1")))
    g = g.filter((pl.col("e1") == pl.col("e") + step) & ~pl.col("decided") & ~pl.col("dec1")
                 & pl.col("mid").is_between(lo, hi) & pl.col("mid1").is_between(lo, hi))
    pr0, pr1 = _probit(g["mid"].to_numpy()), _probit(g["mid1"].to_numpy())
    return g.with_columns((pl.col("mid1") - pl.col("mid")).alias("dp"),
                          (_logit(pl.col("mid1")) - _logit(pl.col("mid"))).alias("dlogit"),
                          pl.Series("dprobit", pr1 - pr0))


TAU_EDGES = (5, 10, 20, 40, 80, 160, 320)


def tau_bin(tau: pl.Expr) -> pl.Expr:
    e = TAU_EDGES
    expr = pl.when(tau < e[0]).then(pl.lit(e[0] / 2))
    for a, b in itertools.pairwise(e):
        expr = expr.when(tau < b).then(pl.lit(math.sqrt(a * b)))
    return expr.otherwise(pl.lit(e[-1] * 1.4))


def sigma_fit(inc: pl.DataFrame, col: str, step: int, boots: int = 200, seed: int = 1) -> dict[str, Any]:
    """Fit mean(d^2)/step = c * tau^beta across tau bins (WLS on logs, bins weighted by count);
    standard errors from a bootstrap over games. Constant sigma: beta = 0; a fair win probability
    of a Brownian score in probit space: c = 1, beta = -1 (variance 1/tau per minute)."""
    f = inc.with_columns(tau_bin(pl.col("tau")).alias("tb"), (pl.col(col) ** 2).alias("d2"))
    per = f.group_by("ticker", "tb").agg(pl.col("d2").sum().alias("s"), pl.len().alias("n"))
    tick = per["ticker"].unique().sort().to_list()
    if len(tick) < 3:
        return {"games": len(tick)}
    bins = sorted(per["tb"].unique().to_list())
    idx = {t: i for i, t in enumerate(tick)}
    bpos = {b: j for j, b in enumerate(bins)}
    S = np.zeros((len(tick), len(bins)))
    N = np.zeros_like(S)
    for t, b, s_, n_ in per.iter_rows():
        S[idx[t], bpos[b]] += s_
        N[idx[t], bpos[b]] += n_
    lt = np.log(np.array(bins))

    def fit(s_: np.ndarray, n_: np.ndarray) -> tuple[float, float]:
        ok = (n_ >= 30) & (s_ > 0)
        if ok.sum() < 3:
            return float("nan"), float("nan")
        v = np.log(s_[ok] / n_[ok] / step)
        w = n_[ok]
        X = np.column_stack([np.ones(ok.sum()), lt[ok]])
        beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], v * np.sqrt(w), rcond=None)[0]
        return float(math.exp(beta[0])), float(beta[1])

    c0, b0 = fit(S.sum(0), N.sum(0))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boots):
        pick = rng.integers(0, len(tick), len(tick))
        bs.append(fit(S[pick].sum(0), N[pick].sum(0)))
    bs = np.array(bs)
    tot_s, tot_n = S.sum(0), N.sum(0)
    out = {"games": len(tick), "obs": int(tot_n.sum()), "c": c0, "beta": b0,
           "se_beta": float(np.nanstd(bs[:, 1])), "se_logc": float(np.nanstd(np.log(bs[:, 0])))}
    for b, s_, n_ in zip(bins, tot_s, tot_n):
        out[f"var@{b:.0f}m"] = (s_ / n_ / step) if n_ else None
    return out


def variance_bound(grid: pl.DataFrame, games: pl.DataFrame, step: int = 1) -> pl.DataFrame:
    """Model-free martingale test. For a fair price that settles at 0/1, the expected quadratic
    variation from t0 to settlement equals p0 (1 - p0). One row per game and starting phase: QV
    on the ``step``-minute grid (forward-filled mid, decided stretches included) plus the final
    step to the outcome, and p0 (1 - p0)."""
    ok = games.filter(pl.col("on_schedule")).select("ticker", "y")
    g = (grid.join(ok, on="ticker").filter((pl.col("e") >= 0) & (pl.col("e") % step == 0)
                                          & pl.col("mid_ff").is_not_null()).sort("ticker", "e"))
    rows = []
    for (tk,), sub in g.group_by(["ticker"], maintain_order=True):
        p = sub["mid_ff"].to_numpy()
        ph = sub["phase"].to_list()
        y = sub["y"][0]
        for start in ("early", "mid", "late"):
            if start not in ph:
                continue
            i0 = ph.index(start)
            if not 0.05 <= p[i0] <= 0.95:
                continue
            seg = np.concatenate([p[i0:], [y]])
            rows.append({"ticker": tk, "sport": sub["sport"][0], "split": sub["split"][0], "start": start,
                         "p0": float(p[i0]), "bound": float(p[i0] * (1 - p[i0])),
                         "qv": float(np.sum(np.diff(seg) ** 2)),
                         "qv_path": float(np.sum(np.diff(p[i0:]) ** 2))})
    return pl.DataFrame(rows)


# ---- summary tables -------------------------------------------------------------------------------------
def coverage_table(games: pl.DataFrame) -> pl.DataFrame:
    return (games.group_by("sport", "split").agg(
        pl.len().alias("games"), pl.col("on_schedule").sum().alias("on_sched"),
        pl.col("prints").sum().alias("prints"), pl.col("prints_inplay").median().alias("inplay_prints_med"),
        pl.col("first_print_e").max().alias("latest_first_print_e"),
        pl.col("len_min").median().alias("close_minus_sched_med"),
        pl.col("vol_onset").median().alias("vol_onset_med"),
        pl.col("vol_onset").quantile(0.25).alias("vol_onset_q25"),
        pl.col("vol_onset").quantile(0.75).alias("vol_onset_q75"),
        pl.col("rate_onset").median().alias("rate_onset_med"),
        pl.col("rate_onset").quantile(0.25).alias("rate_onset_q25"),
        pl.col("rate_onset").quantile(0.75).alias("rate_onset_q75"),
        pl.col("decided_e").median().alias("decided_e_med"))
        .sort("sport", "split"))


def start_profile(grid: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Prints per minute and mean |1-minute mid change| in 10-minute bins around the scheduled
    start: the volatility steps up at the scheduled start; the print rate often does not."""
    ok = games.filter(pl.col("on_schedule"))["ticker"].implode()
    g = (grid.filter(pl.col("ticker").is_in(ok) & pl.col("e").is_between(-60, 59)).sort("ticker", "e")
         .with_columns((pl.col("mid_ff") - pl.col("mid_ff").shift(1).over("ticker")).abs().alias("adm"),
                       (pl.col("e") // 10 * 10).alias("bin")))
    return (g.group_by("sport", "bin").agg(pl.col("prints").mean().round(1).alias("prints_per_min"),
                                           (100 * pl.col("adm").mean()).round(3).alias("abs_dmid_c"))
            .sort("sport", "bin"))


def sigma_table(grid: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for step in (1, 5):
        inc = increments(grid, games, step)
        for sport in ["all", *sorted(inc["sport"].unique().to_list())]:
            sub = inc if sport == "all" else inc.filter(pl.col("sport") == sport)
            for split, f in splits(sub):
                for col in ("dlogit", "dprobit"):
                    rows.append({"step_min": step, "space": col[1:], "sport": sport, "split": split,
                                 **sigma_fit(f, col, step)})
    return pl.DataFrame(rows, infer_schema_length=None)


def jump_table(grid: pl.DataFrame, games: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Jump-diffusion summary per sport and split. z = 1-minute probit change x sqrt(tau): N(0,1)
    for a fair Brownian-score win probability; jumps are |z| > 4. BV/RV: bipower vs realised
    variance of 1-minute logit changes (jump share = 1 - BV/RV). Event rates: 60-second mid moves
    >= k per in-play hour (events table, 60 s refractory)."""
    inc = increments(grid, games, 1).with_columns(
        (pl.col("dprobit") * pl.col("tau").sqrt()).alias("z"),
        (pl.col("dlogit").abs() * pl.col("dlogit").abs().shift(1).over("ticker")).alias("bp"),
        (pl.col("e") - pl.col("e").shift(1).over("ticker")).alias("gap"))
    hours = (grid.filter(pl.col("ticker").is_in(games.filter(pl.col("on_schedule"))["ticker"].implode())
                         & (pl.col("e") >= 0) & ~pl.col("decided"))
             .group_by("ticker").agg((pl.len() / 60).alias("h")))
    ev = events.filter((pl.col("window") == "inplay") & (pl.col("dir") == "follow"))
    rows = []
    for sport in ["all", *sorted(inc["sport"].unique().to_list())]:
        sub = inc if sport == "all" else inc.filter(pl.col("sport") == sport)
        for split, f in splits(sub):
            if f.is_empty():
                continue
            z = f["z"].to_numpy()
            jmp = np.abs(z) > 4
            h = hours.filter(pl.col("ticker").is_in(f["ticker"].unique().implode()))["h"].sum()
            bp = f.filter(pl.col("gap") == 1)["bp"].drop_nulls()
            rv = float((f["dlogit"] ** 2).sum())
            r = {"sport": sport, "split": split, "games": f["ticker"].n_unique(), "minutes": f.height,
                 "var_z": float(np.var(z)), "var_z_diffusion": float(np.var(z[~jmp])),
                 "kurt_z": float(np.mean(z ** 4) / np.var(z) ** 2), "jumps_per_h": float(jmp.sum() / h) if h else None,
                 "jump_share_var": float(np.sum(z[jmp] ** 2) / np.sum(z ** 2)),
                 "bv_rv_jump_share": float(1 - (math.pi / 2) * bp.sum() / rv) if rv else None,
                 "jump_abs_dp_c_med": float(100 * np.median(np.abs(f["dp"].to_numpy()[jmp]))) if jmp.any() else None,
                 "jump_abs_dp_c_q90": float(100 * np.quantile(np.abs(f["dp"].to_numpy()[jmp]), 0.9)) if jmp.any() else None,
                 "jump_abs_dlogit_med": float(np.median(np.abs(f["dlogit"].to_numpy()[jmp]))) if jmp.any() else None,
                 "jump_up_frac": float(np.mean(z[jmp] > 0)) if jmp.any() else None}
            e = ev.filter(pl.col("ticker").is_in(f["ticker"].unique().implode()))
            for k in KS:
                ek = e.filter(pl.col("k") == k)
                r[f"moves{int(100 * k)}c_per_h"] = ek.height / h if h else None
                r[f"moves{int(100 * k)}c_size_med_c"] = (100 * (ek["mid"] - ek["ref"]).abs().median()
                                                        if ek.height else None)
            rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None)


def phase_jump_rates(grid: pl.DataFrame, games: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    ok = games.filter(pl.col("on_schedule"))["ticker"].implode()
    hours = (grid.filter(pl.col("ticker").is_in(ok) & (pl.col("e") >= 0) & ~pl.col("decided"))
             .group_by("sport", "phase").agg((pl.len() / 60).alias("hours")))
    ev = (events.filter((pl.col("window") == "inplay") & (pl.col("dir") == "follow") & pl.col("ticker").is_in(ok))
          .group_by("sport", "phase", "k").agg(pl.len().alias("n")))
    return (ev.join(hours, on=["sport", "phase"]).with_columns((pl.col("n") / pl.col("hours")).round(2).alias("per_h"))
            .pivot(on="k", index=["sport", "phase", "hours"], values="per_h").sort("sport", "phase"))


def calibration_table(grid: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Favourite price q = max(mid, 1 - mid) every 5 minutes vs the favourite winning, by phase and
    q band (on-schedule games, not decided, mid in [0.02, 0.98]); t clustered by game."""
    ok = games.filter(pl.col("on_schedule")).select("ticker", "y")
    g = (grid.join(ok, on="ticker").filter((pl.col("e") % 5 == 0) & ~pl.col("decided")
                                          & pl.col("mid").is_between(0.02, 0.98))
         .with_columns(pl.max_horizontal("mid", 1 - pl.col("mid")).alias("q"),
                       pl.when(pl.col("mid") >= 0.5).then(pl.col("y")).otherwise(1 - pl.col("y")).alias("win"))
         .with_columns((pl.col("win") - pl.col("q")).alias("err"),
                       pl.col("q").cut([0.6, 0.7, 0.8, 0.9, 0.95], left_closed=True).cast(pl.String).alias("qb")))
    rows = []
    for (phase, qb), sub in g.group_by(["phase", "qb"]):
        for split, f in splits(sub):
            r = cl(f, "err")
            # one observation per game for the null z (snapshots of one game share its outcome)
            pg = f.group_by("ticker").agg(pl.col("err").mean(), pl.col("q").mean())
            rows.append({"phase": phase, "q_band": qb, "split": split, "obs": r["n"], "games": r["games"],
                         "q_mean": f["q"].mean(), "win_rate": f["win"].mean(), **_c(r, "err"),
                         "err_per_game_c": None if pg.is_empty() else round(100 * pg["err"].mean(), 3),
                         "z_null": z_null(pg, "err", "q")})
    order = {"pre": 0, "early": 1, "mid": 2, "late": 3, "over": 4}
    return pl.DataFrame(rows).with_columns(pl.col("phase").replace_strict(order).alias("o")).sort(
        "o", "q_band", "split").drop("o")


def _bucket(f: pl.DataFrame) -> pl.DataFrame:
    return f.with_columns(
        pl.when(((pl.col("ref") >= 0.5) & (pl.col("sign") > 0)) | ((pl.col("ref") < 0.5) & (pl.col("sign") < 0)))
        .then(pl.lit("toward fav")).otherwise(pl.lit("toward dog")).alias("towards"),
        pl.when((pl.col("ref") - 0.5).abs() < 0.15).then(pl.lit("ref 35-65"))
        .when((pl.col("ref") - 0.5).abs() < 0.3).then(pl.lit("ref 20-35/65-80"))
        .otherwise(pl.lit("ref <20/>80")).alias("ref_band"))


def reaction_table(events: pl.DataFrame, by: tuple[str, ...] = ()) -> pl.DataFrame:
    """Signed mid move after a jump (positive = continuation / under-reaction), in cents."""
    ev = _bucket(events.filter(pl.col("dir") == "follow"))
    rows = []
    keys = ["window", "k", *by]
    for key, sub in ev.group_by(keys):
        for split, f in splits(sub):
            r = {**dict(zip(keys, key)), "split": split}
            r.update({"events": f.height, "games": f["ticker"].n_unique()})
            for col in ("r_60", "r_300", "r_900", "r_settle", "rf_60", "rf_300", "rf_900"):
                if col in f.columns:
                    r.update(_c(cl(f, col), col))
            rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None).sort(*keys, "split")


def taker_table(events: pl.DataFrame, by: tuple[str, ...] = ()) -> pl.DataFrame:
    """Net PnL per contract (cents) of following / fading jumps as a taker, per exit horizon."""
    ev = _bucket(events.filter(pl.col("filled")))
    rows = []
    keys = ["window", "k", "dir", *by]
    for key, sub in ev.group_by(keys):
        for split, f in splits(sub):
            r = {**dict(zip(keys, key)), "split": split, "trades": f.height,
                 "games": f["ticker"].n_unique(), "entry": f["entry"].mean()}
            for h in ("60", "300", "900", "settle"):
                r.update(_c(cl(f, f"pnl_{h}"), f"pnl_{h}"))
                r[f"gross_{h}_c"] = None if f.is_empty() else round(100 * f[f"gross_{h}"].mean(), 3)
            r["z0_settle"] = z_null(f, "pnl_settle", "entry")
            rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None).sort(*keys, "split")


def maker_table(maker: pl.DataFrame, maker_n: pl.DataFrame) -> pl.DataFrame:
    keys = ["trigger", "x", "life", "rel", "side"]
    att = maker_n.group_by([*keys, "split"]).agg(pl.len().alias("orders"))
    att = pl.concat([att, maker_n.group_by(keys).agg(pl.len().alias("orders")).with_columns(pl.lit("all").alias("split"))],
                    how="diagonal")
    rows = []
    for key, sub in maker.group_by(keys):
        for split, f in splits(sub):
            r = {**dict(zip(keys, key)), "split": split, "fills": f.height, "games": f["ticker"].n_unique()}
            for h in ("60", "300", "900"):
                if f"mo_{h}" in f.columns:
                    r.update(_c(cl(f, f"mo_{h}"), f"mo_{h}"))
            for h in ("60", "300", "900", "settle"):
                r.update(_c(cl(f, f"pnl_{h}"), f"pnl_{h}"))
            rows.append(r)
    out = pl.DataFrame(rows, infer_schema_length=None).join(att, on=[*keys, "split"], how="left")
    return out.with_columns((pl.col("fills") / pl.col("orders")).round(3).alias("fill_rate")).sort(*keys, "split")


LATE_BANDS = [0.7, 0.85, 0.95]


def late_table(late: pl.DataFrame) -> pl.DataFrame:
    """One taker trade per game at the first minute of phase >= f, held to settlement. z0 = null
    z (fair prices), robust when every sampled favourite won."""
    f0 = late.with_columns(pl.col("q").cut(LATE_BANDS, left_closed=True).cast(pl.String).alias("qb"))
    rows = []
    for key, sub in f0.group_by(["f", "dir", "qb"]):
        for split, f in splits(sub):
            c = cl(f, "pnl_settle")
            rows.append({"phase_from": key[0], "dir": key[1], "q_band": key[2], "split": split, "trades": c["n"],
                         "games": c["games"], "entry": f["entry"].mean() if f.height else None,
                         "gross_c": None if f.is_empty() else round(100 * f["gross_settle"].mean(), 3),
                         **_c(c, "pnl"), "z0": z_null(f, "pnl_settle", "entry")})
    return pl.DataFrame(rows, infer_schema_length=None).sort("phase_from", "dir", "q_band", "split")


def pick_on_dev(table: pl.DataFrame, keys: list[str], pnl_cols: list[str], min_games: int = 20) -> pl.DataFrame:
    """For every config, the best exit column on dev (by mean net PnL, with >= min_games games),
    then the same config and column on holdout. One row per config, sorted by dev PnL."""
    rows = []
    dev = table.filter(pl.col("split") == "dev")
    hold = table.filter(pl.col("split") == "holdout")
    for r in dev.iter_rows(named=True):
        if (r.get("games") or 0) < min_games:
            continue
        best = max(((c, r[c]) for c in pnl_cols if r.get(c) is not None), key=lambda x: x[1], default=None)
        if best is None:
            continue
        h = hold.filter(pl.all_horizontal([pl.col(k) == r[k] for k in keys]))
        hr = h.row(0, named=True) if h.height else {}
        tcol = "t_" + best[0][:-2]
        rows.append({**{k: r[k] for k in keys}, "exit": best[0], "dev_c": best[1], "dev_t": r.get(tcol),
                     "dev_games": r.get("games"), "hold_c": hr.get(best[0]), "hold_t": hr.get(tcol),
                     "hold_games": hr.get("games")})
    return pl.DataFrame(rows, infer_schema_length=None).sort("dev_c", descending=True) if rows else pl.DataFrame()


# ---- recorded WebSocket stream (tonight's in-play games) ----------------------------------------------
def stream_trades(root: Path, start: str, end: str, series: tuple[str, ...] = GAME_SERIES) -> pl.DataFrame:
    """Trade prints of game-winner markets from the recorded Kalshi stream (``start``/``end`` are
    ``YYYY-MM-DDTHH``, end exclusive), in the trades-table schema, one leg per game."""
    import json

    from albert.backtest.l2 import stream_files
    from albert.stream import read_stream
    from albert.venues import kalshi as k

    rows = []
    for p in stream_files(root, start, end):
        for row in read_stream(p):
            d = json.loads(row["frame"])
            if d.get("type") != "trade":
                continue
            m = d["msg"]
            tk = m.get("market_ticker", "")
            if not tk.startswith(series):
                continue
            rows.append({"trade_id": m.get("trade_id") or f"{tk}-{m.get('ts_ms')}-{row['recv_ns']}",
                         "ticker": tk, "created_time": k.parse_ts((m.get("ts_ms") or 0) / 1000),
                         "count": float(m["count_fp"]), "yes_price": float(m["yes_price_dollars"]),
                         "taker_side": m.get("taker_side")})
    f = pl.DataFrame(rows, schema=k.TRADE_SCHEMA)
    if f.is_empty():
        return f
    legs = (f.select("ticker").unique().with_columns(pl.col("ticker").str.replace(r"-[^-]+$", "").alias("ev"))
            .sort("ticker").unique("ev", keep="first")["ticker"])
    return f.filter(pl.col("ticker").is_in(legs.implode()))


def stream_meta(s, tickers: list[str]) -> pl.DataFrame:
    """Market metadata (result, occurrence, close, fees) for recorded tickers via the public REST
    market endpoint; only settled markets are kept. Responses are cached under data/cache."""
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    client = CachedClient("kalshi", s.kalshi_base, s.data_dir / "cache" / f"{CACHE_NAME}_raw",
                          min_interval=0.25, record_live=False)
    info = {r["series_ticker"]: r for r in pl.read_parquet(s.tables_dir / "kalshi_series.parquet").iter_rows(named=True)}
    rows = []
    for tk in tickers:
        page = client.get(f"/markets/{tk}", cache=False)
        rows += k.market_rows({"markets": [page.payload.get("market", {})]}, info, "live")
    m = pl.DataFrame(rows, schema=k.MARKET_SCHEMA)
    return m.filter(pl.col("result").is_in(["yes", "no"])).with_columns(pl.lit("stream").alias("split"))


def run_stream(s, start: str, end: str, root: Path | None = None) -> dict[str, pl.DataFrame]:
    """Analyse tonight's recorded in-play games with the historical pipeline (clock scale L from
    the historical dev games). Rerun after the games have settled."""
    root = root or (s.data_dir / "raw" / "stream")
    tr = stream_trades(root, start, end)
    if tr.is_empty():
        return {}
    meta = stream_meta(s, tr["ticker"].unique().sort().to_list())
    hist = game_meta(pl.read_parquet(s.data_dir / "cache" / f"{CACHE_NAME}_games.parquet"))
    lengths = hist.select("sport", "L").unique("sport")
    g = (game_meta(meta.with_columns(pl.lit("dev").alias("split"))).drop("L", "on_schedule")
         .join(lengths, on="sport", how="left").with_columns(pl.lit("stream").alias("split"), pl.lit(True).alias("on_schedule")))
    acc: dict[str, list[dict[str, Any]]] = {}
    for r in g.iter_rows(named=True):
        res = analyze_game(r, prints_frame(tr.filter(pl.col("ticker") == r["ticker"])))
        for key, val in res.items():
            acc.setdefault(key, []).extend(val)
    return {k: pl.DataFrame(v, infer_schema_length=None) for k, v in acc.items()}


# ---- driver ------------------------------------------------------------------------------------------------
def mlb_start_check(markets: pl.DataFrame) -> dict[str, Any]:
    """occurrence_time - 3 h vs the start time encoded in MLB tickers (all settled MLB games)."""
    from albert.venues.kalshi import ticker_start

    m = markets.filter(pl.col("series_ticker") == "KXMLBGAME").unique("event_ticker")
    diffs = []
    for ev, occ, exp in m.select("event_ticker", "occurrence_time", "expected_expiration_time").iter_rows():
        st, sc = ticker_start(ev), scheduled_start(occ, exp)
        if st is not None and sc is not None:
            diffs.append((sc - st).total_seconds() / 60)
    d = np.array(diffs)
    return {"games": len(d), "exact": int((d == 0).sum()), "within_15min": int((np.abs(d) <= 15).sum()),
            "median_min": float(np.median(d)) if len(d) else None}


TAKER_EXITS = ["pnl_60_c", "pnl_300_c", "pnl_900_c", "pnl_settle_c"]


def inplay_dynamics(s, *, fetch: bool = True) -> Path:
    """Fetch (resumable), analyse and report. Returns the report path."""
    from albert.reports import Report

    cache = s.data_dir / "cache"
    gpath = cache / f"{CACHE_NAME}_games.parquet"
    markets = pl.read_parquet(s.tables_dir / "kalshi_markets.parquet")
    if gpath.exists():
        games = pl.read_parquet(gpath)
    else:
        games = select_games(markets, pl.read_parquet(s.tables_dir / "kalshi_trade_windows.parquet"))
        games.write_parquet(gpath)
    if fetch:
        fetch_inplay(s, games, workers=2, min_interval=0.2)
    build_game_files(s, games)
    meta = game_meta(games)
    fr = run_games(s, meta)
    gm, grid = fr["game"], fr["grid"]
    # Trading rules use games whose play started near the scheduled time, judged causally: a >= 3c
    # move within 5 minutes happened between sched - 2 h and sched + 90 min (postponed games
    # otherwise put their real play in the wrong phase). Known by sched + 90 min at the latest.
    ok = gm.filter(pl.col("vol_onset").is_not_null() & (pl.col("vol_onset") <= 90))["ticker"].implode()
    ev = fr["events"].filter(pl.col("ticker").is_in(ok))
    fr = {**fr, "maker": fr["maker"].filter(pl.col("ticker").is_in(ok)),
          "maker_n": fr["maker_n"].filter(pl.col("ticker").is_in(ok)),
          "late": fr["late"].filter(pl.col("ticker").is_in(ok))}

    rep = Report(s.reports_dir, "research-inplay-dynamics")
    rep.text("Data", (
        "kalshi_trades holds only the pre-game window [expected_expiration - 25 h, - 4 h] for the "
        "population legs (in-play for a handful of top-volume games only). A seeded, volume-blind "
        "sample of games (one leg each) was completed from the window end to the close with the "
        "public trades endpoint (cache: data/cache/inplay_dynamics_*). Dev = expected expiration "
        "before 2026-10-01, holdout = October; NBA has October games only (holdout-only). "
        "t-stats are clustered by game."))
    rep.json("Start rule check (MLB tickers carry the scheduled start)", mlb_start_check(markets))
    rep.table("Coverage, start detectors (minutes from sched = occurrence - 3 h) and game length",
              coverage_table(gm), save_as="coverage", max_rows=40)
    rep.table("Prints per minute and mean |1-min mid change| (cents) around the scheduled start",
              start_profile(grid, gm).pivot(on="sport", index="bin", values="abs_dmid_c"), max_rows=20)
    rep.table("... prints per minute", start_profile(grid, gm).pivot(on="sport", index="bin", values="prints_per_min"),
              save_as="start_profile", max_rows=20)
    rep.table("(a) sigma vs time remaining: mean(d^2)/min = c * tau^beta (tau = minutes to close)",
              sigma_table(grid, gm), save_as="sigma", max_rows=80)
    vb = variance_bound(grid, gm)
    vrows = []
    for start in ("early", "mid", "late"):
        for sport in ["all", *sorted(vb["sport"].unique().to_list())]:
            sub = vb.filter(pl.col("start") == start)
            sub = sub if sport == "all" else sub.filter(pl.col("sport") == sport)
            for split, f in splits(sub.with_columns((pl.col("qv") - pl.col("bound")).alias("excess"))):
                c = cl(f, "excess")
                vrows.append({"start": start, "sport": sport, "split": split, "games": c["n"],
                              "qv": f["qv"].mean(), "bound_p(1-p)": f["bound"].mean(),
                              "ratio": f["qv"].sum() / f["bound"].sum() if f.height else None,
                              "excess": c["mean"], "t_excess": c["t"]})
    rep.table("(a') variance bound: E[QV to settlement] vs E[p0(1-p0)] (1-min grid; ratio > 1 = excess "
              "volatility)", pl.DataFrame(vrows, infer_schema_length=None), save_as="variance_bound", max_rows=80)
    rep.table("(b) jump-diffusion (z = 1-min probit change x sqrt(tau), jumps |z| > 4)",
              jump_table(grid, gm, ev), save_as="jumps", max_rows=40)
    rep.table("(b) 60-second moves >= k per in-play hour by phase", phase_jump_rates(grid, gm, ev),
              save_as="jump_phase", max_rows=40)
    cal = calibration_table(grid, gm)
    rep.table("(c) calibration: favourite win rate - favourite price, every 5 min, by phase",
              cal, save_as="calibration", max_rows=120)
    rep.table("(3) reaction after a 60-s move >= k (cents, + = continuation); rf_* use fresh mids "
              "(both sides printed within 15 s)", reaction_table(ev), save_as="reaction", max_rows=40)
    rep.table("(3) reaction by sport", reaction_table(ev, ("sport",)), save_as="reaction_sport", max_rows=120)
    rep.table("(3) reaction by phase", reaction_table(ev, ("phase",)), save_as="reaction_phase", max_rows=120)
    rep.table("(3) reaction by pre-jump price", reaction_table(ev, ("towards", "ref_band")),
              save_as="reaction_price", max_rows=120)
    tt = taker_table(ev)
    rep.table("(4) taker: follow / fade jumps, net cents per contract", tt, save_as="taker", max_rows=40)
    tts = taker_table(ev, ("sport",))
    ttp = taker_table(ev, ("phase",))
    rep.table("(4) taker by sport", tts, save_as="taker_sport", max_rows=200)
    rep.table("(4) taker by phase", ttp, save_as="taker_phase", max_rows=200)
    rep.table("(4) taker configs chosen on dev (best exit per config), with holdout",
              pl.concat([pick_on_dev(tt, ["window", "k", "dir"], TAKER_EXITS),
                         pick_on_dev(tts, ["window", "k", "dir", "sport"], TAKER_EXITS),
                         pick_on_dev(ttp, ["window", "k", "dir", "phase"], TAKER_EXITS)], how="diagonal"),
              save_as="taker_pick", max_rows=60)
    lt = late_table(fr["late"])
    rep.table("(4) late game: buy favourite / underdog at the first minute of phase >= f, hold to settlement",
              lt, save_as="late", max_rows=200)
    rep.table("(4) late game chosen on dev", pick_on_dev(lt, ["phase_from", "dir", "q_band"], ["pnl_c"], min_games=20),
              max_rows=40)
    mt = maker_table(fr["maker"], fr["maker_n"])
    rep.table("(4) trade-print maker (fills only when a later print trades through; mo_* = to the mid, "
              "pnl_* = taker exit)", mt, save_as="maker", max_rows=300)
    rep.table("(4) maker configs chosen on dev", pick_on_dev(mt, ["trigger", "x", "life", "rel", "side"],
                                                             ["mo_60_c", "mo_300_c", "mo_900_c"] + TAKER_EXITS),
              save_as="maker_pick", max_rows=60)
    return rep.save()


def report_stream(s, start: str, end: str, root: Path | None = None) -> Path | None:
    """Report for recorded in-play games (e.g. start='2026-11-05T23', end='2026-11-06T07'): the
    same reaction, taker, maker and calibration tables on the stream's trade prints. Run after the
    games have settled (unsettled markets are skipped)."""
    from albert.reports import Report

    fr = run_stream(s, start, end, root)
    if not fr or "game" not in fr:
        return None
    rep = Report(s.reports_dir, "research-inplay-dynamics-stream")
    rep.text("Data", f"Recorded Kalshi stream {start}..{end} (end exclusive), game-winner markets, one "
                     "leg per game, settled markets only. Clock scale L from the historical dev games.")
    rep.table("Games", fr["game"], save_as="games", max_rows=60)
    ev = fr.get("events", pl.DataFrame())
    if ev.height:
        rep.table("Reaction after a 60-s move >= k (cents)", reaction_table(ev), save_as="reaction")
        rep.table("Taker follow / fade (net cents)", taker_table(ev), save_as="taker")
    if fr.get("maker") is not None and fr["maker"].height:
        rep.table("Trade-print maker", maker_table(fr["maker"], fr["maker_n"]), save_as="maker", max_rows=200)
    return rep.save()
