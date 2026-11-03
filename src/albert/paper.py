"""Paper trading: live quotes, simulated orders, the same sizing and risk caps as the backtester.

No real orders are placed. This module contains no authenticated client and no order-signing
code. Fills are simulated at the current best ask, limited to the displayed size.

State lives in ``data/paper/state.json``. Every decision (fill, skip with reason, settlement) is
appended to ``data/paper/ledger.jsonl``, so a run can be audited and compared to the backtest.

Strategies:
* ``longshot``: Kalshi markets closing ~``horizon`` hours from now, priced by the calibration
  model fit on all settled history in data/tables.
* ``wallet_follow``: poll scored wallets' Polymarket intl trades. When one buys a game
  moneyline, buy the same outcome on the cheaper of Kalshi or Polymarket US.
* ``xvenue``: matched Kalshi / Polymarket US moneylines. Buy both sides when the combined
  cost plus fees is below $1.
* ``maker``: resting bids on Kalshi Weather/Sports favorites posted ``maker_horizon`` hours
  before scheduled expiry and cancelled ``maker_ttl`` hours later. Uses the same order and fill
  rules as ``backtest maker``: a paper order fills only when a later *real* trade sells into
  our price (at it if we improved the best bid, through it if we joined the queue).
  Collateral is reserved while an order rests, as on Kalshi.
"""

import json
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from albert.backtest import maker as mk
from albert.categories import kalshi_is_non_financial
from albert.config import Settings
from albert.fees import fee, kalshi_fee
from albert.http import HttpError
from albert.ingest import Clients, kalshi_series
from albert.matching import kalshi_ticker_for, match_pmus_pm
from albert.research import calibration as cal
from albert.research import follow, wallets
from albert.risk import RiskConfig, size_order
from albert.store import read_table
from albert.venues import kalshi as k
from albert.venues import polymarket as pm
from albert.venues import polymarket_us as pmus

# Polymarket intl event tags for the leagues Kalshi lists (college basketball has no tag).
INTL_LEAGUE_TAGS = ("nba", "nfl", "nhl", "mlb", "cfb", "wnba")


@dataclass
class Position:
    strategy: str
    venue: str
    market_id: str        # Kalshi ticker or Polymarket US slug
    side: str             # "yes"/"no" (Kalshi) or "long"/"short" (Polymarket US)
    event_id: str
    contracts: int
    price: float
    fee: float
    cost: float
    prob: float
    opened_at: str
    note: str = ""


@dataclass
class RestingOrder:
    ticker: str
    event_id: str
    category: str
    side: str
    bid: float
    contracts: int          # order size
    improved: bool          # True: we set a new best bid (first in queue)
    reserved: float         # collateral still held for this order
    placed_at: str
    cancel_at: str
    last_trade_ts: str      # fills are counted from trades strictly after this time
    fee_type: str | None = None
    fee_multiplier: float | None = None
    filled: int = 0
    fee_paid: float = 0.0


@dataclass
class PaperState:
    cash: dict[str, float]
    positions: list[Position] = field(default_factory=list)
    peak_equity: float = 0.0
    day: str = ""
    day_start_equity: float = 0.0
    halted: bool = False
    wallet_poll_ts: str | None = None
    seen_trades: list[str] = field(default_factory=list)
    realized_pnl: float = 0.0
    settled: int = 0
    orders: list[RestingOrder] = field(default_factory=list)
    maker_scan_ts: str | None = None

    @classmethod
    def load(cls, path: Path) -> "PaperState":
        raw = json.loads(path.read_text())
        raw["positions"] = [Position(**p) for p in raw.get("positions", [])]
        raw["orders"] = [RestingOrder(**o) for o in raw.get("orders", [])]
        return cls(**raw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["seen_trades"] = self.seen_trades[-5000:]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    def equity(self) -> float:
        """Cash + open positions at cost + collateral reserved by resting orders."""
        return (sum(self.cash.values()) + sum(p.cost for p in self.positions)
                + sum(o.reserved for o in self.orders))


class PaperTrader:
    def __init__(self, s: Settings, c: Clients, *, cfg: RiskConfig = RiskConfig(),
                 strategies: tuple[str, ...] = ("longshot", "wallet_follow", "xvenue"),
                 horizon_hours: float = 2.0, top_k: int = 20, arb_min_profit: float = 0.01,
                 prior: float = 200.0, maker_horizon: float = 24.0, maker_ttl: float = 20.0,
                 maker_size_dollars: float = 10.0,
                 maker_categories: tuple[str, ...] = ("Climate and Weather", "Sports"),
                 maker_scan_minutes: float = 10.0) -> None:
        self.prior = prior
        self.maker_horizon, self.maker_ttl = maker_horizon, maker_ttl
        self.maker_size_dollars = maker_size_dollars
        self.maker_categories = maker_categories
        self.maker_scan_minutes = maker_scan_minutes
        self.s, self.c, self.cfg = s, c, cfg
        self.strategies = strategies
        self.horizon = horizon_hours
        self.top_k = top_k
        self.arb_min_profit = arb_min_profit
        self.state_path = s.paper_dir / "state.json"
        self.ledger_path = s.paper_dir / "ledger.jsonl"
        if not self.state_path.exists():
            raise RuntimeError("no paper account; run `albert paper reset` first")
        self.state = PaperState.load(self.state_path)
        self._model: cal.CalibrationModel | None = None
        self._series: dict[str, dict] | None = None
        self._wallet_edges: pl.DataFrame | None = None
        self._pmus_open: pl.DataFrame | None = None
        self._pmus_open_at: datetime | None = None

    # ---- bookkeeping --------------------------------------------------------------------------
    @staticmethod
    def reset(s: Settings, cash: dict[str, float]) -> PaperState:
        state = PaperState(cash=dict(cash), peak_equity=sum(cash.values()))
        state.save(s.paper_dir / "state.json")
        (s.paper_dir / "ledger.jsonl").open("a").write(json.dumps(
            {"ts": _now().isoformat(), "event": "reset", "cash": cash}) + "\n")
        return state

    def log(self, event: str, **data: Any) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a") as f:
            f.write(json.dumps({"ts": _now().isoformat(), "event": event, **data}, default=str) + "\n")

    def tick(self) -> dict[str, int]:
        counts = {"settled": self.settle()}
        now = _now()
        eq = self.state.equity()
        self.state.peak_equity = max(self.state.peak_equity, eq)
        if self.state.day != now.strftime("%Y-%m-%d"):
            self.state.day, self.state.day_start_equity = now.strftime("%Y-%m-%d"), eq
        if eq < self.state.peak_equity * (1 - self.cfg.drawdown_kill):
            if not self.state.halted:
                self.log("halted", equity=eq, peak=self.state.peak_equity)
            self.state.halted = True
        daily_stop = eq < self.state.day_start_equity * (1 - self.cfg.daily_loss_fraction)
        if self.state.halted or daily_stop:
            counts["cancelled"] = self.cancel_all_orders("halted" if self.state.halted else "daily_loss")
        if not self.state.halted and not daily_stop:
            for name in self.strategies:
                try:
                    counts[name] = getattr(self, f"run_{name}")()
                except (HttpError, RuntimeError, FileNotFoundError) as error:
                    self.log("strategy_error", strategy=name, error=str(error))
                    counts[name] = -1
        self.state.save(self.state_path)
        return counts

    def _held(self) -> set[str]:
        return {p.market_id for p in self.state.positions} | {o.ticker for o in self.state.orders}

    def _event_exposure(self, event_id: str) -> float:
        return (sum(p.cost for p in self.state.positions if p.event_id == event_id)
                + sum(o.reserved for o in self.state.orders if o.event_id == event_id))

    def _try_fill(self, *, strategy: str, venue: str, market_id: str, side: str, event_id: str,
                  price: float, prob: float, available: float | None, note: str = "",
                  fee_type: str | None = None, multiplier: float | None = None) -> bool:
        if market_id in self._held():
            return False
        wallet = "kalshi" if venue == "kalshi" else "pmus"
        sz = size_order(venue=venue, prob=prob, price=price, available=available,
                        bankroll=self.state.cash[wallet], event_exposure=self._event_exposure(event_id),
                        cfg=self.cfg, fee_type=fee_type, multiplier=multiplier)
        if sz.contracts == 0:
            if sz.reason != "edge_below_threshold":
                self.log("skip", strategy=strategy, venue=venue, market_id=market_id, side=side,
                         price=price, prob=prob, reason=sz.reason)
            return False
        self.state.cash[wallet] -= sz.cost
        pos = Position(strategy, venue, market_id, side, event_id, sz.contracts, price, sz.fee,
                       sz.cost, prob, _now().isoformat(), note)
        self.state.positions.append(pos)
        self.log("fill", **asdict(pos))
        return True

    # ---- settlement ---------------------------------------------------------------------------
    def settle(self) -> int:
        keep, done = [], 0
        for p in self.state.positions:
            payoff = self._resolution(p)
            if payoff is None:
                keep.append(p)
                continue
            proceeds = payoff * p.contracts
            self.state.cash["kalshi" if p.venue == "kalshi" else "pmus"] += proceeds
            self.state.realized_pnl += proceeds - p.cost
            self.state.settled += 1
            done += 1
            self.log("settle", market_id=p.market_id, venue=p.venue, side=p.side, payoff=payoff,
                     contracts=p.contracts, pnl=proceeds - p.cost, strategy=p.strategy)
        self.state.positions = keep
        return done

    def _resolution(self, p: Position) -> float | None:
        try:
            if p.venue == "kalshi":
                m = self.c.kalshi.client.get(f"/markets/{p.market_id}", cache=False).payload["market"]
                if m.get("status") not in ("settled", "finalized") or m.get("result") not in ("yes", "no"):
                    return None
                yes = 1.0 if m["result"] == "yes" else 0.0
                return yes if p.side == "yes" else 1 - yes
            page = self.c.pmus.client.get("/markets", {"slug": p.market_id}, cache=False)
            rows = pmus.market_rows(page.payload)
            if not rows or rows[0]["settlement"] is None:
                return None
            long_payoff = rows[0]["settlement"]
            return long_payoff if p.side == "long" else 1 - long_payoff
        except (HttpError, KeyError):
            return None

    # ---- quotes -------------------------------------------------------------------------------
    def kalshi_quote(self, ticker: str) -> dict | None:
        try:
            return k.orderbook_top(self.c.kalshi.orderbook(ticker).payload)
        except HttpError:
            return None

    def pmus_quote(self, slug: str) -> dict | None:
        try:
            return pmus.book_top(self.c.pmus.book(slug).payload)
        except HttpError:
            return None

    def pmus_open(self) -> pl.DataFrame:
        """Open Polymarket US markets with event slugs, refreshed hourly."""
        now = _now()
        if self._pmus_open is None or self._pmus_open_at is None or now - self._pmus_open_at > timedelta(hours=1):
            rows, emap = [], {}
            # No date window: open game moneylines carry end dates well past the game itself.
            passes = pmus.window_params(None, None)
            for extra in passes:
                for page in self.c.pmus.markets(closed=False, cache=False, extra=extra):
                    rows.extend(pmus.market_rows(page.payload))
            ev_filter = {k_: v for k_, v in passes[0].items() if k_ != "marketTypes"}
            ev_filter["categories"] = list(pmus.NON_FINANCIAL_CATEGORIES)
            for page in self.c.pmus.events(closed=False, cache=False, extra=ev_filter):
                emap.update(pmus.event_slug_map(page.payload))
            frame = pl.DataFrame(rows, schema=pmus.MARKET_SCHEMA).with_columns(
                pl.col("slug").replace_strict(emap, default=None, return_dtype=pl.String).alias("event_slug"))
            self._pmus_open, self._pmus_open_at = frame.filter(pl.col("active")), now
        return self._pmus_open

    # ---- strategy: longshot -------------------------------------------------------------------
    def model(self) -> cal.CalibrationModel:
        if self._model is None:
            obs = cal.kalshi_observations(read_table(self.s.tables_dir, "kalshi_markets"),
                                          read_table(self.s.tables_dir, "kalshi_candles"),
                                          horizon_hours=self.horizon)
            if obs.height < 500:
                raise RuntimeError(f"only {obs.height} calibration observations; ingest more history")
            self._model = cal.CalibrationModel.fit(obs, prior=self.prior)
        return self._model

    def run_longshot(self, window_minutes: float = 30.0) -> int:
        if self._series is None:
            self._series = kalshi_series(self.s, self.c)
        now = _now()
        lo = now + timedelta(hours=self.horizon) - timedelta(minutes=window_minutes)
        hi = now + timedelta(hours=self.horizon) + timedelta(minutes=window_minutes)
        rows: list[dict] = []
        for page in self.c.kalshi._cursor_pages("/markets", {
                "status": "open", "mve_filter": "exclude", "limit": 1000,
                "min_close_ts": int(lo.timestamp()), "max_close_ts": int(hi.timestamp())},
                "markets", cache=False):
            rows.extend(k.market_rows(page.payload, self._series, "live"))
        markets = [r for r in rows if kalshi_is_non_financial(r["category"])
                   and (r["volume"] or 0) >= 1000 and r["yes_bid"] and r["yes_ask"]
                   and 0 < r["yes_ask"] - r["yes_bid"] <= 0.10]
        fills = 0
        scan = {"closing_in_window": len(rows), "eligible": len(markets), "quoted": 0,
                "best_edge": None}
        for m in markets:
            if m["ticker"] in self._held():
                continue
            q = self.kalshi_quote(m["ticker"])
            if not q or q["yes_ask"] is None or q["yes_bid"] is None:
                continue
            obs = pl.DataFrame([{"category": m["category"], "mid": (q["yes_ask"] + q["yes_bid"]) / 2}])
            prob = float(self.model().predict(obs)["prob"][0])
            yes_edge = prob - q["yes_ask"]
            no_edge = (1 - prob) - (1 - q["yes_bid"])
            scan["quoted"] += 1
            best = round(max(yes_edge, no_edge), 4)
            scan["best_edge"] = best if scan["best_edge"] is None else max(scan["best_edge"], best)
            if yes_edge >= no_edge:
                side, price, p, avail = "yes", q["yes_ask"], prob, q["yes_ask_size"]
            else:
                side, price, p, avail = "no", round(1 - q["yes_bid"], 4), 1 - prob, q["yes_bid_size"]
            fills += self._try_fill(strategy="longshot", venue="kalshi", market_id=m["ticker"],
                                    side=side, event_id=m["event_ticker"], price=price, prob=p,
                                    available=avail, fee_type=m["fee_type"],
                                    multiplier=m["fee_multiplier"], note=f"{m['category']} mid->{prob:.3f}")
        self.log("scan", strategy="longshot", **scan)
        return fills

    # ---- strategy: maker ----------------------------------------------------------------------
    def cancel_all_orders(self, reason: str) -> int:
        n = len(self.state.orders)
        for o in self.state.orders:
            self.state.cash["kalshi"] += o.reserved
            self.log("cancel", strategy="maker", market_id=o.ticker, reason=reason, filled=o.filled,
                     released=round(o.reserved, 4))
        self.state.orders = []
        return n

    def _new_trades(self, ticker: str, after: datetime) -> pl.DataFrame:
        rows: list[dict] = []
        for page in self.c.kalshi._cursor_pages(
                "/markets/trades", {"ticker": ticker, "limit": 1000, "min_ts": int(after.timestamp())},
                "trades", cache=False):
            rows.extend(k.trade_rows(page.payload))
        return pl.DataFrame(rows, schema=k.TRADE_SCHEMA).filter(pl.col("created_time") > after)

    def update_orders(self) -> int:
        """Apply real trades since the last check to each resting order; cancel expired ones."""
        now, fills, keep = _now(), 0, []
        for o in self.state.orders:
            after = datetime.fromisoformat(o.last_trade_ts)
            cancel_at = datetime.fromisoformat(o.cancel_at)
            start = k.ticker_start(o.event_id)
            if start is not None and start - timedelta(minutes=15) < cancel_at:
                cancel_at = start - timedelta(minutes=15)  # never rest into a live game
                o.cancel_at = cancel_at.isoformat()
            try:
                trades = self._new_trades(o.ticker, after)
            except HttpError as error:
                self.log("strategy_error", strategy="maker", market_id=o.ticker, error=str(error))
                keep.append(o)
                continue
            if not trades.is_empty():
                frame = pl.DataFrame([{
                    "ticker": o.ticker, "event_ticker": o.event_id, "category": o.category,
                    "decision_ts": after, "cancel_ts": min(cancel_at, now), "resolve_ts": cancel_at,
                    "side": o.side, "bid": o.bid, "ask": None, "improved": o.improved, "payoff": None,
                    "fee_type": o.fee_type, "fee_multiplier": o.fee_multiplier}], schema=mk.ORDER_SCHEMA)
                got = int(mk.simulate_fills(frame, trades, size=o.contracts - o.filled)["filled"][0])
                o.last_trade_ts = trades["created_time"].max().isoformat()
                if got:
                    total_fee = kalshi_fee(o.filled + got, o.bid, maker=True,
                                           fee_type=o.fee_type or "quadratic", multiplier=o.fee_multiplier)
                    fee_now = total_fee - o.fee_paid
                    cost = got * o.bid + fee_now
                    o.filled += got
                    o.fee_paid = total_fee
                    o.reserved = max(o.reserved - cost, 0.0)
                    pos = Position("maker", "kalshi", o.ticker, o.side, o.event_id, got, o.bid, fee_now,
                                   cost, o.bid, now.isoformat(),
                                   f"{o.category} {'improved' if o.improved else 'joined'} fill {o.filled}/{o.contracts}")
                    self.state.positions.append(pos)
                    self.log("fill", **asdict(pos))
                    fills += 1
            if o.filled >= o.contracts or now >= cancel_at:
                self.state.cash["kalshi"] += o.reserved
                self.log("cancel", strategy="maker", market_id=o.ticker,
                         reason="filled" if o.filled >= o.contracts else "ttl", filled=o.filled,
                         released=round(o.reserved, 4))
                continue
            keep.append(o)
        self.state.orders = keep
        return fills

    def place_orders(self) -> int:
        now = _now()
        if self._series is None:
            self._series = kalshi_series(self.s, self.c)
        target = now + timedelta(hours=self.maker_horizon)
        window = timedelta(minutes=max(self.maker_scan_minutes, 15.0))
        rows: list[dict] = []
        # The API filters on close_time only. Open game markets carry close_time = the backstop
        # latest_expiration_time (~2 days after the game) while weather closes before its
        # scheduled expiry, so query wide and filter on expected_expiration_time locally.
        for page in self.c.kalshi._cursor_pages("/markets", {
                "status": "open", "mve_filter": "exclude", "limit": 1000,
                "min_close_ts": int(now.timestamp()),
                "max_close_ts": int((target + window + timedelta(days=3)).timestamp())},
                "markets", cache=False):
            rows.extend(k.market_rows(page.payload, self._series, "live"))
        busy_events = ({_game_key(p.event_id) for p in self.state.positions}
                       | {_game_key(o.event_id) for o in self.state.orders})
        cands = [r for r in rows if r["category"] in self.maker_categories
                 and r["expected_expiration_time"] is not None
                 and abs(r["expected_expiration_time"] - target) <= window
                 and r["ticker"] not in self._held() and _game_key(r["event_ticker"]) not in busy_events]
        scan = {"open_in_window": len(rows), "candidates": len(cands), "quoted": 0, "placed": 0}
        for m in cands:
            q = self.kalshi_quote(m["ticker"])
            if not q:
                continue
            scan["quoted"] += 1
            if _game_key(m["event_ticker"]) in busy_events:
                continue  # one order per game: both legs, spreads and totals are correlated bets
            rule = mk.quote_order(q["yes_bid"], q["yes_ask"])
            if rule is None:
                continue
            # Same sizing as backtest maker: 5% of free Kalshi cash, capped per order.
            cash = self.state.cash["kalshi"]
            budget = min(self.cfg.max_bet_fraction * cash, self.maker_size_dollars)
            n = int(budget // (rule["bid"] * 1.02))
            reserve = n * rule["bid"] + kalshi_fee(n, rule["bid"], maker=True,
                                                   fee_type=m["fee_type"] or "quadratic",
                                                   multiplier=m["fee_multiplier"]) if n else 0.0
            if n < 1 or reserve > cash:
                self.log("skip", strategy="maker", market_id=m["ticker"], reason="insufficient_cash"
                         if n >= 1 else "below_min_size", bid=rule["bid"])
                continue
            cancel_at = min(now + timedelta(hours=self.maker_ttl), m["close_time"])
            start = k.ticker_start(m["event_ticker"])
            if start is not None:
                cancel_at = min(cancel_at, start - timedelta(minutes=15))
            if cancel_at <= now:
                continue  # game starts too soon: never rest into a live game
            o = RestingOrder(m["ticker"], m["event_ticker"], m["category"], rule["side"], rule["bid"], n,
                             rule["improved"], reserve, now.isoformat(), cancel_at.isoformat(),
                             now.isoformat(), m["fee_type"], m["fee_multiplier"])
            self.state.cash["kalshi"] -= reserve
            self.state.orders.append(o)
            busy_events.add(_game_key(m["event_ticker"]))
            self.log("order", **asdict(o))
            scan["placed"] += 1
        self.log("scan", strategy="maker", **scan)
        return scan["placed"]

    def run_maker(self) -> int:
        fills = self.update_orders()
        last = self.state.maker_scan_ts
        if last is None or _now() - datetime.fromisoformat(last) >= timedelta(minutes=self.maker_scan_minutes):
            self.place_orders()
            self.state.maker_scan_ts = _now().isoformat()
        return fills

    # ---- strategy: wallet_follow --------------------------------------------------------------
    def wallet_edges(self) -> pl.DataFrame:
        if self._wallet_edges is None:
            t = read_table(self.s.tables_dir, "pm_trades")
            rt = wallets.resolved_trades(t, read_table(self.s.tables_dir, "pm_markets"))
            scored = wallets.score_wallets(rt).filter(~pl.col("mm_like") & (pl.col("shrunk_roi") > 0))
            chosen = scored.head(self.top_k)["wallet"].to_list()
            self._wallet_edges = follow.wallet_edges(rt, chosen).filter(pl.col("shrunk_edge_pts") > 0)
        return self._wallet_edges

    def run_wallet_follow(self) -> int:
        edges = self.wallet_edges()
        since = (datetime.fromisoformat(self.state.wallet_poll_ts) if self.state.wallet_poll_ts
                 else _now() - timedelta(minutes=15))
        self.state.wallet_poll_ts = _now().isoformat()
        seen = set(self.state.seen_trades)
        fills = 0
        scan = {"wallets": edges.height, "new_buys": 0}
        for row in edges.iter_rows(named=True):
            for page in self.c.pm.recent_wallet_trades(row["wallet"], since):
                for t in pm.trade_rows(page.payload):
                    key = f"{t['transaction_hash']}:{t['token_id']}:{t['side']}"
                    if key in seen or t["side"] != "BUY" or (t["price"] or 0) * (t["size"] or 0) < 50:
                        continue
                    seen.add(key)
                    self.state.seen_trades.append(key)
                    scan["new_buys"] += 1
                    fills += self._follow_trade(t, row["shrunk_edge_pts"], row["wallet"])
        self.log("scan", strategy="wallet_follow", since=since.isoformat(), **scan)
        return fills

    def _kalshi_ticker(self, intl_event_slug: str, outcome_index: int) -> str | None:
        return kalshi_ticker_for(intl_event_slug, outcome_index)

    def _intl_market(self, condition_id: str) -> pl.DataFrame:
        page = self.c.pm.gamma.get("/markets", {"condition_ids": [condition_id]}, cache=False)
        rows = []
        for m in page.payload or []:
            event = (m.get("events") or [{}])[0]
            rows.extend(pm.market_rows([{**event, "markets": [m]}]))
        return pl.DataFrame(rows, schema=pm.MARKET_SCHEMA)

    def _follow_trade(self, t: dict, edge_pts: float, wallet: str) -> int:
        prob = min(0.99, t["price"] + edge_pts)
        options = []
        intl = self._intl_market(t["condition_id"])
        if not intl.is_empty():
            ticker = self._kalshi_ticker(intl["event_slug"][0], t["outcome_index"]) \
                if intl["slug"][0] == intl["event_slug"][0] else None
            if ticker:
                q = self.kalshi_quote(ticker)
                if q and q["yes_ask"]:
                    options.append(("kalshi", ticker, "yes", q["yes_ask"], q["yes_ask_size"]))
            for mt in match_pmus_pm(self.pmus_open(), intl).iter_rows(named=True):
                q = self.pmus_quote(mt["pmus_slug"])
                if not q:
                    continue
                if mt["pm_outcome_index"] == t["outcome_index"] and q["yes_ask"]:
                    options.append(("pmus", mt["pmus_slug"], "long", q["yes_ask"], q["yes_ask_size"]))
                elif mt["pm_outcome_index"] != t["outcome_index"] and q["yes_bid"]:
                    options.append(("pmus", mt["pmus_slug"], "short", round(1 - q["yes_bid"], 4),
                                    q["yes_bid_size"]))
        if not options:
            self.log("skip", strategy="wallet_follow", wallet=wallet, slug=t["slug"],
                     outcome=t["outcome"], reason="no_tradeable_match")
            return 0
        venue, mid, side, price, avail = min(options, key=lambda o: o[3] + fee(o[0], 100, o[3]) / 100)
        return int(self._try_fill(strategy="wallet_follow", venue=venue, market_id=mid, side=side,
                                  event_id=t["event_slug"] or mid, price=price, prob=prob,
                                  available=avail, note=f"copy {wallet[:10]} @ {t['price']:.3f}"))

    # ---- strategy: xvenue ---------------------------------------------------------------------
    def intl_open_games(self) -> pl.DataFrame:
        now = _now()
        rows: list[dict] = []
        for tag in INTL_LEAGUE_TAGS:
            cursor = None
            while True:
                page = self.c.pm.gamma.get("/events/keyset", {
                    "closed": "false", "limit": 100, "tag_slug": tag, "after_cursor": cursor,
                    "end_date_min": (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end_date_max": (now + timedelta(days=4)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }, cache=False)
                rows.extend(pm.market_rows(page.payload.get("events") or []))
                cursor = page.payload.get("next_cursor")
                if not cursor or not page.payload.get("events"):
                    break
        frame = pl.DataFrame(rows, schema=pm.MARKET_SCHEMA)
        return frame.filter(pl.col("slug") == pl.col("event_slug"))

    def run_xvenue(self) -> int:
        """Matched Kalshi / Polymarket US moneylines, matched through the intl market.

        Kalshi YES(team) == intl outcome i == Polymarket US long side (when the match says the
        US long side is outcome i). These are the same mappings validated on settled markets.
        """
        fills = 0
        scan = {"intl_games": 0, "us_matches": 0, "quoted": 0, "best_margin": None}
        intl = self.intl_open_games()
        scan["intl_games"] = intl.height
        matches = match_pmus_pm(self.pmus_open(), intl)
        scan["us_matches"] = matches.height
        for mt in matches.iter_rows(named=True):
            ticker = self._kalshi_ticker(mt["pm_event_slug"], mt["pm_outcome_index"])
            if not ticker:
                continue
            kq = self.kalshi_quote(ticker)
            uq = self.pmus_quote(mt["pmus_slug"])
            if not kq or not uq or None in (kq["yes_ask"], kq["yes_bid"], uq["yes_ask"], uq["yes_bid"]):
                continue
            scan["quoted"] += 1
            # Kalshi YES(team) and the PM US long side are the same team.
            # Leg 1: YES on Kalshi + short on PM US.  Leg 2: NO on Kalshi + long on PM US.
            legs = [
                (kq["yes_ask"], "yes", kq["yes_ask_size"], round(1 - uq["yes_bid"], 4), "short", uq["yes_bid_size"]),
                (round(1 - kq["yes_bid"], 4), "no", kq["yes_bid_size"], uq["yes_ask"], "long", uq["yes_ask_size"]),
            ]
            for kp, kside, ksize, up, uside, usize in legs:
                margin = round(1 - kp - up - fee("kalshi", 100, kp) / 100 - fee("pmus", 100, up) / 100, 4)
                scan["best_margin"] = margin if scan["best_margin"] is None else max(scan["best_margin"], margin)
                n = int(min(ksize or 0, usize or 0, self.cfg.max_bet_dollars // max(kp + up, 0.01)))
                if n < 1:
                    continue
                total = n * (kp + up) + fee("kalshi", n, kp) + fee("pmus", n, up)
                profit = n - total
                if profit / n < self.arb_min_profit:
                    continue
                if (self.state.cash["kalshi"] < n * kp + fee("kalshi", n, kp)
                        or self.state.cash["pmus"] < n * up + fee("pmus", n, up)):
                    self.log("skip", strategy="xvenue", market_id=ticker, reason="insufficient_cash")
                    continue
                for venue, mid, side, price in (("kalshi", ticker, kside, kp),
                                                ("pmus", mt["pmus_slug"], uside, up)):
                    f = fee(venue, n, price)
                    pos = Position("xvenue", venue, mid, side, ticker.rsplit("-", 1)[0], n, price, f,
                                   n * price + f, 1.0, _now().isoformat(), f"locked profit {profit:.2f}")
                    self.state.cash["kalshi" if venue == "kalshi" else "pmus"] -= pos.cost
                    self.state.positions.append(pos)
                    self.log("fill", **asdict(pos))
                fills += 1
                break
        self.log("scan", strategy="xvenue", **scan)
        return fills


def run_loop(trader: PaperTrader, *, interval_seconds: float, iterations: int) -> None:
    """Tick until stopped. Built to run as a service (e.g. a container on a VM):

    * SIGTERM/SIGINT finish the current tick, save state and exit cleanly;
    * an unexpected error in one tick is logged to the ledger and retried with backoff
      instead of killing the process;
    * ``data/paper/heartbeat.json`` is rewritten after every tick for health checks.
    """
    stop = {"flag": False}

    def _stop(signum, frame):  # noqa: ARG001
        stop["flag"] = True
        print(f"[{_now():%Y-%m-%d %H:%M:%S}] signal {signum}: stopping after this tick", flush=True)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)
    heartbeat = trader.s.paper_dir / "heartbeat.json"
    n, failures = 0, 0
    while not stop["flag"] and (iterations <= 0 or n < iterations):
        n += 1
        try:
            counts = trader.tick()
            failures = 0
        except Exception as error:  # keep the service alive; the ledger records what happened
            failures += 1
            trader.log("tick_error", error=repr(error), consecutive=failures)
            counts = {"error": repr(error)[:200]}
        st = trader.state
        line = {"ts": _now().isoformat(), "tick": n, "counts": counts, "equity": round(st.equity(), 4),
                "cash": {k_: round(v, 4) for k_, v in st.cash.items()}, "positions": len(st.positions),
                "orders": len(st.orders), "halted": st.halted}
        print(json.dumps(line, default=str), flush=True)
        heartbeat.parent.mkdir(parents=True, exist_ok=True)
        tmp = heartbeat.with_suffix(".tmp")
        tmp.write_text(json.dumps(line, default=str))
        tmp.replace(heartbeat)
        if stop["flag"] or (iterations > 0 and n >= iterations):
            break
        wait = interval_seconds * min(2 ** failures, 16)
        deadline = time.monotonic() + wait
        while not stop["flag"] and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))
    trader.state.save(trader.state_path)


def status(s: Settings) -> str:
    st = PaperState.load(s.paper_dir / "state.json")
    lines = [f"equity {st.equity():.2f} (peak {st.peak_equity:.2f}) realized_pnl {st.realized_pnl:+.2f} "
             f"settled {st.settled} halted={st.halted}",
             f"cash {st.cash}", f"open positions: {len(st.positions)}",
             f"resting orders: {len(st.orders)} (reserved {sum(o.reserved for o in st.orders):.2f})"]
    for o in st.orders:
        lines.append(f"  order {o.placed_at[:16]} {o.ticker} {o.side} bid {o.bid:.2f} "
                     f"{o.filled}/{o.contracts} until {o.cancel_at[:16]}")
    for p in st.positions:
        lines.append(f"  {p.opened_at[:16]} {p.strategy:13} {p.venue:6} {p.market_id} {p.side} "
                     f"x{p.contracts} @ {p.price:.3f} cost {p.cost:.2f} p={p.prob:.3f}")
    return "\n".join(lines)


def _game_key(event_ticker: str) -> str:
    """``KXCS2SPREAD-26NOV040430PRVMOUZ`` and ``KXCS2GAME-26NOV040430PRVMOUZ`` are one match:
    the part after the series prefix identifies the underlying game/date/place."""
    return event_ticker.split("-", 1)[1] if "-" in event_ticker else event_ticker


def _now() -> datetime:
    return datetime.now(UTC)
