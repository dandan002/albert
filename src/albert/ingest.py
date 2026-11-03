"""Ingest pipelines: fetch -> raw cache -> typed Parquet tables -> run manifest."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from albert.categories import pm_is_non_financial, pmus_is_non_financial
from albert.config import Settings
from albert.http import CachedClient
from albert.store import read_table, write_manifest, write_table
from albert.venues import kalshi as k
from albert.venues import polymarket as pm
from albert.venues import polymarket_us as pmus
from albert.venues import weather as wx


@dataclass
class Clients:
    kalshi: k.Kalshi
    pm: pm.Polymarket
    pmus: pmus.PolymarketUS
    iem: wx.IEM | None = None
    nws: wx.NWS | None = None

    @classmethod
    def build(cls, s: Settings, *, record_live: bool = True) -> "Clients":
        """``record_live=False`` skips saving uncached (live) responses; use it for long-running
        loops. Research ingests keep the default so their manifests point at real files."""
        return cls(
            kalshi=k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.08,
                                           record_live=record_live)),
            pm=pm.Polymarket(
                CachedClient("pm_gamma", s.pm_gamma_base, s.raw_dir, min_interval=0.12,
                             record_live=record_live),
                CachedClient("pm_data", s.pm_data_base, s.raw_dir, min_interval=0.12,
                             record_live=record_live),
            ),
            pmus=pmus.PolymarketUS(CachedClient("pmus", s.pmus_base, s.raw_dir, min_interval=0.1,
                                               record_live=record_live)),
            iem=wx.IEM(CachedClient("iem", s.iem_base, s.raw_dir, min_interval=0.5,
                                    record_live=record_live, timeout=120.0)),
            nws=wx.NWS(CachedClient("nws", s.nws_base, s.raw_dir, min_interval=0.5,
                                    record_live=record_live, user_agent=s.nws_user_agent)),
        )


@dataclass
class IngestResult:
    tables: dict[str, int] = field(default_factory=dict)
    raw_files: list[Path] = field(default_factory=list)
    manifest: Path | None = None

    def summary(self) -> str:
        lines = [f"{name}: {rows} rows" for name, rows in self.tables.items()]
        if self.manifest:
            lines.append(f"manifest: {self.manifest}")
        return "\n".join(lines)


def _require_rows(name: str, frame: pl.DataFrame) -> None:
    if frame.is_empty():
        raise RuntimeError(f"{name}: 0 rows. Treating as a failure (check filters and dates).")


# ---- Kalshi ---------------------------------------------------------------------------------
def kalshi_series(s: Settings, c: Clients) -> dict[str, dict]:
    page = c.kalshi.series()
    frame = k.series_frame(page.payload)
    write_table(s.tables_dir, "kalshi_series", frame, merge=False)
    return {row["series_ticker"]: row for row in frame.iter_rows(named=True)}


def ingest_kalshi(s: Settings, c: Clients, start: datetime, end: datetime, *,
                  categories: list[str] | None = None, min_volume: float = 0.0,
                  trades: bool = False, candles: bool = False, candle_minutes: int = 60,
                  max_markets: int | None = None, per_category: bool = False) -> IngestResult:
    result = IngestResult()
    series_info = kalshi_series(s, c)
    rows: list[dict] = []
    for page in c.kalshi.settled_markets(start, end):
        result.raw_files.append(page.raw_path)
        source = "historical" if "/historical/" in page.url else "live"
        rows.extend(k.market_rows(page.payload, series_info, source))
    markets = k.markets_frame(rows).filter(
        pl.col("close_time").is_between(start, end, closed="left")
        & (pl.col("volume").fill_null(0) >= min_volume)
        & pl.col("result").is_in(["yes", "no"])
    )
    if categories:
        markets = markets.filter(pl.col("category").is_in(categories))
    _require_rows("kalshi_markets", markets)
    write_table(s.tables_dir, "kalshi_markets", markets, key=["ticker"])
    result.tables["kalshi_markets"] = markets.height

    selected = markets.sort("volume", descending=True)
    if max_markets:
        # per_category: top-N within each category, so Sports (~95% of volume) does not crowd
        # out everything else.
        selected = (selected.group_by("category", maintain_order=True).head(max_markets)
                    if per_category else selected.head(max_markets))
    cutoff = c.kalshi.cutoff()

    if trades:
        _kalshi_trades(s, c, selected, cutoff, result)
    if candles:
        _kalshi_candles(s, c, selected, cutoff, candle_minutes, result)

    result.manifest = write_manifest(s.data_dir, "ingest kalshi", {
        "start": start, "end": end, "categories": categories, "min_volume": min_volume,
        "trades": trades, "candles": candles, "candle_minutes": candle_minutes,
        "max_markets": max_markets, "per_category": per_category,
    }, result.tables, result.raw_files)
    return result


def _kalshi_trades(s: Settings, c: Clients, selected: pl.DataFrame, cutoff: dict,
                   result: IngestResult, max_pages: int | None = None) -> None:
    trade_rows: list[dict] = []
    for i, m in enumerate(selected.iter_rows(named=True), 1):
        historical = bool(cutoff["trades_created_ts"] and m["close_time"] < cutoff["trades_created_ts"])
        try:
            for page in c.kalshi.trades(m["ticker"], historical, max_pages=max_pages):
                result.raw_files.append(page.raw_path)
                trade_rows.extend(k.trade_rows(page.payload))
        except RuntimeError as error:
            print(f"skip trades {m['ticker']}: {error}")
        if i % 250 == 0:
            print(f"  trades: {i}/{selected.height} markets, {len(trade_rows)} rows")
    frame = pl.DataFrame(trade_rows, schema=k.TRADE_SCHEMA)
    _require_rows("kalshi_trades", frame)
    write_table(s.tables_dir, "kalshi_trades", frame, key=["trade_id"])
    result.tables["kalshi_trades"] = frame.height


def _kalshi_candles(s: Settings, c: Clients, selected: pl.DataFrame, cutoff: dict,
                    candle_minutes: int, result: IngestResult, batch_size: int = 60) -> None:
    """Candles per market. Post-cutoff markets go through the batch endpoint, grouped by a
    shared time window, which cuts the request count by ~``batch_size``x."""
    candle_rows: list[dict] = []

    def window(m: dict) -> tuple[datetime, datetime]:
        open_time = m["open_time"] or (m["close_time"] - timedelta(days=30))
        # Kalshi caps candle requests at ~5000 periods; a 60-minute candle covers ~200 days.
        return max(open_time, m["close_time"] - timedelta(minutes=candle_minutes * 4900)), m["close_time"]

    historical, live = [], []
    for m in selected.iter_rows(named=True):
        old = bool(cutoff["market_settled_ts"] and m["close_time"] < cutoff["market_settled_ts"])
        (historical if old else live).append(m)
    for i, m in enumerate(historical, 1):
        lo, hi = window(m)
        try:
            page = c.kalshi.candles(m["ticker"], lo, hi, candle_minutes, True)
        except RuntimeError as error:  # one bad market should not kill a long run
            print(f"skip candles {m['ticker']}: {error}")
            continue
        result.raw_files.append(page.raw_path)
        candle_rows.extend(k.candle_rows(page.payload, m["ticker"], candle_minutes))
        if i % 250 == 0:
            print(f"  candles (historical): {i}/{len(historical)} markets, {len(candle_rows)} rows", flush=True)
    # Batch live markets: sort by window, then chunk; each request spans its chunk's window.
    live.sort(key=lambda m: (window(m)[0], m["ticker"]))
    for i in range(0, len(live), batch_size):
        chunk = live[i:i + batch_size]
        lo = min(window(m)[0] for m in chunk)
        hi = max(window(m)[1] for m in chunk)
        if (hi - lo) > timedelta(minutes=candle_minutes * 4900):
            lo = hi - timedelta(minutes=candle_minutes * 4900)
        try:
            page = c.kalshi.candles_batch([m["ticker"] for m in chunk], lo, hi, candle_minutes)
        except RuntimeError as error:
            print(f"skip candle batch at {chunk[0]['ticker']}: {error}")
            continue
        result.raw_files.append(page.raw_path)
        for mk in page.payload.get("markets") or []:
            candle_rows.extend(k.candle_rows(mk, mk.get("market_ticker"), candle_minutes))
        done = min(i + batch_size, len(live))
        if (i // batch_size) % 10 == 0:
            print(f"  candles (batch): {done}/{len(live)} markets, {len(candle_rows)} rows", flush=True)
    frame = pl.DataFrame(candle_rows, schema=k.CANDLE_SCHEMA)
    _require_rows("kalshi_candles", frame)
    write_table(s.tables_dir, "kalshi_candles", frame, key=["ticker", "end_ts", "period_minutes"])
    result.tables["kalshi_candles"] = frame.height


def ingest_kalshi_detail(s: Settings, c: Clients, *, sample_per_category: int = 0,
                         min_volume: float = 500.0, max_volume: float = 200_000.0,
                         multi_events: int = 0, max_outcomes: int = 12, seed: int = 7,
                         trades: bool = True, candles: bool = True, max_trade_pages: int = 10,
                         candle_minutes: int = 60, series: list[str] | None = None,
                         start: datetime | None = None, end: datetime | None = None,
                         categories: list[str] | None = None, all_legs: bool = False) -> IngestResult:
    """Trades/candles for a reproducible *sample* of already-ingested settled markets.

    ``ingest kalshi --max-markets`` takes the top-N by volume, which over-represents a few huge
    markets. This draws a seeded random sample per category within a volume band (for
    maker/taker return studies), plus every leg of ``multi_events`` mutually-exclusive events
    (exactly one YES) so that sum-of-legs arbitrage can be checked across complete events.

    ``series`` takes *every* settled market of those series closing in [start, end), one leg
    per event (the first ticker alphabetically), with no volume filter. Choosing markets by
    final volume is look-ahead: upsets draw heavy in-game trading, so a volume-ranked sample
    over-represents them.
    """
    result = IngestResult()
    markets = read_table(s.tables_dir, "kalshi_markets")
    parts = []
    if sample_per_category:
        band = markets.filter(pl.col("volume").is_between(min_volume, max_volume))
        if categories:
            band = band.filter(pl.col("category").is_in(categories))
        parts.append(band.sample(fraction=1.0, shuffle=True, seed=seed)
                     .group_by("category", maintain_order=True).head(sample_per_category)
                     .select(markets.columns))
    if multi_events:
        ev = markets.group_by("event_ticker").agg(
            n=pl.len(), yes=(pl.col("result") == "yes").sum(), vol=pl.col("volume").sum())
        ev = ev.filter(pl.col("n").is_between(3, max_outcomes) & (pl.col("yes") == 1))
        chosen = ev.sample(n=min(multi_events, ev.height), shuffle=True, seed=seed)["event_ticker"]
        parts.append(markets.filter(pl.col("event_ticker").is_in(chosen.implode())))
    if series:
        full = markets.filter(pl.col("series_ticker").is_in(series))
        if start:
            full = full.filter(pl.col("close_time") >= start)
        if end:
            full = full.filter(pl.col("close_time") < end)
        # One leg per event (games: both legs mirror each other) or every leg (bracket events
        # such as daily temperature, where each leg is a different outcome).
        parts.append(full if all_legs else
                     full.sort("ticker").unique("event_ticker", keep="first", maintain_order=True))
    if not parts:
        raise ValueError("nothing selected: pass sample_per_category, multi_events or series")
    selected = pl.concat(parts).unique("ticker", maintain_order=True)
    print(f"detail ingest: {selected.height} markets")
    cutoff = c.kalshi.cutoff()
    if trades:
        _kalshi_trades(s, c, selected, cutoff, result, max_pages=max_trade_pages)
    if candles:
        _kalshi_candles(s, c, selected, cutoff, candle_minutes, result)
    result.manifest = write_manifest(s.data_dir, "ingest kalshi-detail", {
        "sample_per_category": sample_per_category, "min_volume": min_volume,
        "max_volume": max_volume, "multi_events": multi_events, "max_outcomes": max_outcomes,
        "seed": seed, "trades": trades, "candles": candles, "max_trade_pages": max_trade_pages,
        "series": series, "start": start, "end": end, "categories": categories,
        "all_legs": all_legs,
    }, result.tables, result.raw_files)
    return result


def ingest_kalshi_events(s: Settings, c: Clients, *, min_legs: int = 3) -> IngestResult:
    """Event metadata (``mutually_exclusive``, leg count) for multi-leg events that have candles.

    Sum-of-legs arbitrage is only riskless when the legs are mutually exclusive *and* complete,
    which the market table alone cannot tell us.
    """
    result = IngestResult()
    markets = read_table(s.tables_dir, "kalshi_markets")
    with_candles = read_table(s.tables_dir, "kalshi_candles")["ticker"].unique().implode()
    ev = (markets.filter(pl.col("ticker").is_in(with_candles))
          .group_by("event_ticker").agg(pl.len().alias("n")).filter(pl.col("n") >= min_legs))
    rows = []
    for e in ev["event_ticker"].to_list():
        try:
            page = c.kalshi.event(e)
        except RuntimeError as error:
            print(f"skip event {e}: {error}")
            continue
        result.raw_files.append(page.raw_path)
        rows.append(k.event_row(page.payload))
    frame = pl.DataFrame(rows, schema=k.EVENT_SCHEMA)
    _require_rows("kalshi_events", frame)
    write_table(s.tables_dir, "kalshi_events", frame, key=["event_ticker"])
    result.tables["kalshi_events"] = frame.height
    result.manifest = write_manifest(s.data_dir, "ingest kalshi-events", {"min_legs": min_legs},
                                     result.tables, result.raw_files)
    return result


WINDOW_SCHEMA = {"ticker": pl.String, "window_start": pl.Datetime("us", "UTC"),
                 "window_end": pl.Datetime("us", "UTC"), "trades": pl.Int64}


def ingest_kalshi_trade_windows(s: Settings, c: Clients, *, series: list[str], start: datetime,
                                end: datetime, horizon_hours: float = 24.0,
                                ttl_hours: float = 20.0, lead_hours: float = 1.0) -> IngestResult:
    """*All* trades in each game's maker-order window, for one leg of every game in ``series``.

    Window = [scheduled expiry - horizon - lead, scheduled expiry - horizon + ttl]. Unlike the
    newest-N sample this has full coverage however busy the market is, so the maker backtest
    is not restricted (and biased) to quiet markets. Coverage is recorded in
    ``kalshi_trade_windows``.
    """
    from albert.research.literature import full_population_legs

    result = IngestResult()
    markets = read_table(s.tables_dir, "kalshi_markets")
    legs = full_population_legs(markets, tuple(series)).implode()
    sel = markets.filter(pl.col("ticker").is_in(legs) & (pl.col("close_time") >= start)
                         & (pl.col("close_time") < end))
    cutoff = c.kalshi.cutoff()
    trade_rows: list[dict] = []
    windows: list[dict] = []
    for i, m in enumerate(sel.iter_rows(named=True), 1):
        anchor = m["expected_expiration_time"] or m["close_time"]
        lo = anchor - timedelta(hours=horizon_hours + lead_hours)
        hi = anchor - timedelta(hours=horizon_hours) + timedelta(hours=ttl_hours)
        historical = bool(cutoff["trades_created_ts"] and m["close_time"] < cutoff["trades_created_ts"])
        n = 0
        try:
            for page in c.kalshi.trades(m["ticker"], historical, min_ts=lo, max_ts=hi):
                result.raw_files.append(page.raw_path)
                rows = k.trade_rows(page.payload)
                trade_rows.extend(rows)
                n += len(rows)
        except RuntimeError as error:
            print(f"skip trades {m['ticker']}: {error}")
            continue
        windows.append({"ticker": m["ticker"], "window_start": lo, "window_end": hi, "trades": n})
        if i % 250 == 0:
            print(f"  windows: {i}/{sel.height} markets, {len(trade_rows)} trades", flush=True)
    frame = pl.DataFrame(trade_rows, schema=k.TRADE_SCHEMA)
    _require_rows("kalshi_trades", frame)
    write_table(s.tables_dir, "kalshi_trades", frame, key=["trade_id"])
    write_table(s.tables_dir, "kalshi_trade_windows", pl.DataFrame(windows, schema=WINDOW_SCHEMA),
                key=["ticker"])
    result.tables["kalshi_trades"] = frame.height
    result.tables["kalshi_trade_windows"] = len(windows)
    result.manifest = write_manifest(s.data_dir, "ingest kalshi-trade-windows", {
        "series": series, "start": start, "end": end, "horizon_hours": horizon_hours,
        "ttl_hours": ttl_hours, "lead_hours": lead_hours}, result.tables, result.raw_files)
    return result


def weather_codes(markets: pl.DataFrame) -> list[str]:
    """CLI station codes used by Kalshi daily high/low markets (from their rules text)."""
    rules = markets.filter(pl.col("series_ticker").str.contains(r"^KX(HIGH|LOWT)"))
    codes = {wx.cli_code(r) for r in rules["rules_primary"].unique().to_list()}
    return sorted(c for c in codes if c and c in wx.STATION_TZ)


def ingest_weather(s: Settings, c: Clients, start: datetime, end: datetime,
                   codes: list[str] | None = None) -> IngestResult:
    """NWS CLI (settlement), ASOS observations and NBM guidance from the IEM archive."""
    result = IngestResult()
    if codes is None:
        codes = weather_codes(read_table(s.tables_dir, "kalshi_markets"))
    cli, obs, nbm = [], [], []
    for code in codes:
        for year in range(start.year, end.year + 1):
            page = c.iem.cli(code, year)
            result.raw_files.append(page.raw_path)
            cli.extend(wx.cli_rows(page.payload, code))
        page = c.iem.asos(code, (start - timedelta(days=2)).date(), (end + timedelta(days=2)).date())
        result.raw_files.append(page.raw_path)
        obs.extend(wx.asos_rows(page.payload, code))
        page = c.iem.nbm(code, start - timedelta(days=3), end + timedelta(days=1))
        result.raw_files.append(page.raw_path)
        nbm.extend(wx.nbm_rows(page.payload, code))
        print(f"  {code}: cli {len(cli)} obs {len(obs)} nbm {len(nbm)}", flush=True)
    for name, rows, schema, key in (("wx_cli", cli, wx.CLI_SCHEMA, ["code", "day"]),
                                    ("wx_obs", obs, wx.OBS_SCHEMA, ["code", "ts"]),
                                    ("wx_nbm", nbm, wx.NBM_SCHEMA, ["code", "runtime", "ftime"])):
        f = wx.frame(rows, schema).unique(key, keep="last")
        _require_rows(name, f)
        write_table(s.tables_dir, name, f, key=key)
        result.tables[name] = f.height
    result.manifest = write_manifest(s.data_dir, "ingest weather", {
        "start": start, "end": end, "codes": codes}, result.tables, result.raw_files)
    return result


def ingest_kalshi_open(s: Settings, c: Clients) -> pl.DataFrame:
    """Snapshot open markets (for matching and paper trading). Not written as the settled table."""
    series_info = kalshi_series(s, c)
    rows: list[dict] = []
    for page in c.kalshi.open_markets():
        rows.extend(k.market_rows(page.payload, series_info, "live"))
    frame = k.markets_frame(rows)
    write_table(s.tables_dir, "kalshi_open_markets", frame, merge=False)
    return frame


# ---- Polymarket intl (data only) --------------------------------------------------------------
def ingest_pm_markets(s: Settings, c: Clients, start: datetime, end: datetime, *,
                      closed: bool = True, min_volume: float = 100.0) -> IngestResult:
    result = IngestResult()
    rows: list[dict] = []
    for page in c.pm.events(closed=closed, end_min=start, end_max=end, cache=closed):
        result.raw_files.append(page.raw_path)
        rows.extend(pm.market_rows(page.payload["events"]))
    frame = pl.DataFrame(rows, schema=pm.MARKET_SCHEMA)
    frame = frame.filter(pl.Series([pm_is_non_financial(t) for t in frame["tags"].to_list()],
                                   dtype=pl.Boolean))
    frame = frame.filter(pl.col("volume") >= min_volume)
    if closed:
        # Event end dates are loose; keep markets whose own close falls in the window.
        when = pl.coalesce("closed_time", "end_date")
        frame = frame.filter(pl.col("winner_index").is_not_null()
                             & when.is_between(start, end, closed="left"))
    name = "pm_markets" if closed else "pm_open_markets"
    _require_rows(name, frame)
    write_table(s.tables_dir, name, frame, key=["condition_id"], merge=closed)
    result.tables[name] = frame.height
    result.manifest = write_manifest(s.data_dir, "ingest pm-markets",
                                     {"start": start, "end": end, "closed": closed,
                                      "min_volume": min_volume},
                                     result.tables, result.raw_files)
    return result


def ingest_pm_wallets(s: Settings, c: Clients, start: datetime, end: datetime, *,
                      top: int = 200, categories: tuple[str, ...] = ("overall",),
                      time_period: str = "all", extra_wallets: list[str] | None = None,
                      max_pages: int = 20) -> IngestResult:
    """Leaderboard wallets (plus any extras) and all their trades in [start, end).

    The leaderboard is only a sampling frame. It is biased toward winners, which is why
    research/wallets.py tests persistence out of sample instead of trusting its rankings.
    """
    result = IngestResult()
    board: list[dict] = []
    for category in categories:
        for page in c.pm.leaderboard(time_period=time_period, category=category, limit=top):
            result.raw_files.append(page.raw_path)
            board.extend(pm.leaderboard_rows(page.payload, time_period, category))
    board_frame = pl.DataFrame(board, schema=pm.LEADERBOARD_SCHEMA)
    write_table(s.tables_dir, "pm_leaderboard", board_frame,
                key=["wallet", "time_period", "category"])
    result.tables["pm_leaderboard"] = board_frame.height

    wallets = list(dict.fromkeys(board_frame["wallet"].to_list() + [w.lower() for w in extra_wallets or []]))
    trade_rows: list[dict] = []
    # Page month by month: the feed is newest-first, so one capped pass over a long window would
    # leave very active wallets with only their last few days of history.
    chunks = []
    lo = start
    while lo < end:
        hi = min(end, lo + timedelta(days=30))
        chunks.append((lo, hi))
        lo = hi
    for i, wallet in enumerate(wallets, 1):
        for lo, hi in chunks:
            for page in c.pm.wallet_trades(wallet, lo, hi, max_pages=max_pages):
                result.raw_files.append(page.raw_path)
                trade_rows.extend(pm.trade_rows(page.payload))
        if i % 25 == 0:
            print(f"  wallets {i}/{len(wallets)}, trades so far {len(trade_rows)}")
    trades = pl.DataFrame(trade_rows, schema=pm.TRADE_SCHEMA)
    _require_rows("pm_trades", trades)
    write_table(s.tables_dir, "pm_trades", trades,
                key=["transaction_hash", "wallet", "token_id", "side", "price", "size"])
    result.tables["pm_trades"] = trades.height
    result.manifest = write_manifest(s.data_dir, "ingest pm-wallets", {
        "start": start, "end": end, "top": top, "categories": categories,
        "time_period": time_period, "extra_wallets": extra_wallets,
    }, result.tables, result.raw_files)
    return result


def ingest_pm_resolutions_for_trades(s: Settings, c: Clients) -> IngestResult:
    """Make sure every market a tracked wallet traded has metadata and a resolution.

    The wallet trade feed spans markets outside any date window we ingested, so we fetch the
    missing ones by condition id from Gamma.
    """
    result = IngestResult()
    trades = read_table(s.tables_dir, "pm_trades")
    try:
        known = set(read_table(s.tables_dir, "pm_markets")["condition_id"].to_list())
    except FileNotFoundError:
        known = set()
    missing = sorted(set(trades["condition_id"].drop_nulls().to_list()) - known)
    rows: list[dict] = []
    for i in range(0, len(missing), 50):
        batch = missing[i:i + 50]
        # Gamma /markets defaults to open markets only, so ask for closed ones explicitly.
        page = c.pm.gamma.get("/markets", {"condition_ids": batch, "limit": 100, "closed": "true"})
        result.raw_files.append(page.raw_path)
        # /markets returns bare markets; wrap them so market_rows can reuse the event parser.
        for m in page.payload:
            event = (m.get("events") or [{}])[0]
            rows.extend(pm.market_rows([{**event, "markets": [m]}]))
        if (i // 50) % 100 == 99:
            print(f"  resolutions {i + 50}/{len(missing)}")
    if rows:
        frame = pl.DataFrame(rows, schema=pm.MARKET_SCHEMA)
        write_table(s.tables_dir, "pm_markets", frame, key=["condition_id"])
        result.tables["pm_markets_added"] = frame.height
    result.manifest = write_manifest(s.data_dir, "ingest pm-resolutions", {"missing": len(missing)},
                                     result.tables, result.raw_files)
    return result


# ---- Polymarket US ------------------------------------------------------------------------------
def ingest_pmus(s: Settings, c: Clients, *, closed: bool, start: datetime | None = None,
                end: datetime | None = None, max_pages: int | None = None) -> IngestResult:
    """Polymarket US markets in non-financial categories, optionally limited by end date.

    The venue has over 2M closed markets (mostly short crypto contracts), so closed ingests
    should always pass a window.
    """
    result = IngestResult()
    passes = pmus.window_params(start, end)
    rows: list[dict] = []
    for extra in passes:
        for n, page in enumerate(c.pmus.markets(closed=closed, cache=closed, extra=extra), 1):
            result.raw_files.append(page.raw_path)
            rows.extend(pmus.market_rows(page.payload))
            if max_pages and n >= max_pages:
                break
    event_map: dict[str, str] = {}
    event_filter = {k_: v for k_, v in passes[0].items() if k_ != "marketTypes"}
    event_filter["categories"] = list(pmus.NON_FINANCIAL_CATEGORIES)
    for n, page in enumerate(c.pmus.events(closed=closed, cache=closed, extra=event_filter), 1):
        result.raw_files.append(page.raw_path)
        event_map.update(pmus.event_slug_map(page.payload))
        if max_pages and n >= max_pages * 5:
            break
    frame = pl.DataFrame(rows, schema=pmus.MARKET_SCHEMA).with_columns(
        pl.col("slug").replace_strict(event_map, default=None, return_dtype=pl.String)
        .alias("event_slug")
    )
    frame = frame.filter(pl.Series([pmus_is_non_financial(x) for x in frame["category"].to_list()],
                                   dtype=pl.Boolean))
    if closed:
        frame = frame.filter(pl.col("settlement").is_not_null())
    name = "pmus_markets" if closed else "pmus_open_markets"
    _require_rows(name, frame)
    write_table(s.tables_dir, name, frame, key=["slug"], merge=closed)
    result.tables[name] = frame.height
    result.manifest = write_manifest(s.data_dir, f"ingest pmus {'closed' if closed else 'open'}",
                                     {"closed": closed, "start": start, "end": end,
                                      "max_pages": max_pages},
                                     result.tables, result.raw_files)
    return result
