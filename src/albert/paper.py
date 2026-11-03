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
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from albert.categories import kalshi_is_non_financial
from albert.config import Settings
from albert.fees import fee
from albert.http import HttpError
from albert.ingest import Clients, kalshi_series
from albert.matching import KALSHI_GAME_LEAGUES, MONTHS, parse_pm_game
from albert.research import calibration as cal
from albert.research import follow, wallets
from albert.risk import RiskConfig, size_order
from albert.store import read_table
from albert.venues import kalshi as k
from albert.venues import polymarket as pm
from albert.venues import polymarket_us as pmus

LEAGUE_TO_KALSHI = {v: kk for kk, v in KALSHI_GAME_LEAGUES.items()}
MONTH_ABBR = {v: kk for kk, v in MONTHS.items()}


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

    @classmethod
    def load(cls, path: Path) -> "PaperState":
        raw = json.loads(path.read_text())
        raw["positions"] = [Position(**p) for p in raw.get("positions", [])]
        return cls(**raw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["seen_trades"] = self.seen_trades[-5000:]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    def equity(self) -> float:
        return sum(self.cash.values()) + sum(p.cost for p in self.positions)


class PaperTrader:
    def __init__(self, s: Settings, c: Clients, *, cfg: RiskConfig = RiskConfig(),
                 strategies: tuple[str, ...] = ("longshot", "wallet_follow", "xvenue"),
                 horizon_hours: float = 2.0, top_k: int = 20, arb_min_profit: float = 0.01) -> None:
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
        if not self.state.halted and eq >= self.state.day_start_equity * (1 - self.cfg.daily_loss_fraction):
            for name in self.strategies:
                try:
                    counts[name] = getattr(self, f"run_{name}")()
                except (HttpError, RuntimeError, FileNotFoundError) as error:
                    self.log("strategy_error", strategy=name, error=str(error))
                    counts[name] = -1
        self.state.save(self.state_path)
        return counts

    def _held(self) -> set[str]:
        return {p.market_id for p in self.state.positions}

    def _event_exposure(self, event_id: str) -> float:
        return sum(p.cost for p in self.state.positions if p.event_id == event_id)

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
            passes = pmus.window_params(now - timedelta(days=1), now + timedelta(days=14))
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
            self._model = cal.CalibrationModel.fit(obs)
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
            if yes_edge >= no_edge:
                side, price, p, avail = "yes", q["yes_ask"], prob, q["yes_ask_size"]
            else:
                side, price, p, avail = "no", round(1 - q["yes_bid"], 4), 1 - prob, q["yes_bid_size"]
            fills += self._try_fill(strategy="longshot", venue="kalshi", market_id=m["ticker"],
                                    side=side, event_id=m["event_ticker"], price=price, prob=p,
                                    available=avail, fee_type=m["fee_type"],
                                    multiplier=m["fee_multiplier"], note=f"{m['category']} mid->{prob:.3f}")
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
        for row in edges.iter_rows(named=True):
            for page in self.c.pm.recent_wallet_trades(row["wallet"], since):
                for t in pm.trade_rows(page.payload):
                    key = f"{t['transaction_hash']}:{t['token_id']}:{t['side']}"
                    if key in seen or t["side"] != "BUY" or (t["price"] or 0) * (t["size"] or 0) < 50:
                        continue
                    seen.add(key)
                    self.state.seen_trades.append(key)
                    fills += self._follow_trade(t, row["shrunk_edge_pts"], row["wallet"])
        return fills

    def _follow_trade(self, t: dict, edge_pts: float, wallet: str) -> int:
        prob = min(0.99, t["price"] + edge_pts)
        g = parse_pm_game(t["event_slug"] or t["slug"] or "")
        options = []
        if g and g["league"] in LEAGUE_TO_KALSHI:
            pick = (g["a"] if t["outcome_index"] == 0 else g["b"]).upper()
            y, mo, d = g["date"].split("-")
            ticker = (f"{LEAGUE_TO_KALSHI[g['league']]}-{y[2:]}{MONTH_ABBR[int(mo)]}{d}"
                      f"{g['a'].upper()}{g['b'].upper()}-{pick}")
            q = self.kalshi_quote(ticker)
            if q and q["yes_ask"]:
                options.append(("kalshi", ticker, "yes", q["yes_ask"], q["yes_ask_size"]))
        us = self.pmus_open().filter(pl.col("event_slug").str.ends_with(t["event_slug"] or "~none~"))
        for m in us.iter_rows(named=True):
            if (m["long_side"] or "").lower() == (t["outcome"] or "").lower():
                q = self.pmus_quote(m["slug"])
                if q and q["yes_ask"]:
                    options.append(("pmus", m["slug"], "long", q["yes_ask"], q["yes_ask_size"]))
            elif (m["short_side"] or "").lower() == (t["outcome"] or "").lower():
                q = self.pmus_quote(m["slug"])
                if q and q["yes_bid"]:
                    options.append(("pmus", m["slug"], "short", round(1 - q["yes_bid"], 4), q["yes_bid_size"]))
        if not options:
            self.log("skip", strategy="wallet_follow", wallet=wallet, slug=t["slug"],
                     outcome=t["outcome"], reason="no_tradeable_match")
            return 0
        venue, mid, side, price, avail = min(options, key=lambda o: o[3] + fee(o[0], 100, o[3]) / 100)
        return int(self._try_fill(strategy="wallet_follow", venue=venue, market_id=mid, side=side,
                                  event_id=t["event_slug"] or mid, price=price, prob=prob,
                                  available=avail, note=f"copy {wallet[:10]} @ {t['price']:.3f}"))

    # ---- strategy: xvenue ---------------------------------------------------------------------
    def _pm_outcomes(self, event_slug: str) -> list[str]:
        if not hasattr(self, "_outcomes_cache"):
            self._outcomes_cache: dict[str, list[str]] = {}
        if event_slug not in self._outcomes_cache:
            try:
                page = self.c.pm.gamma.get("/markets", {"slug": event_slug})
                market = (page.payload or [{}])[0]
                self._outcomes_cache[event_slug] = [o.lower() for o in pm._json_list(market.get("outcomes"))]
            except (HttpError, IndexError):
                self._outcomes_cache[event_slug] = []
        return self._outcomes_cache[event_slug]

    def run_xvenue(self) -> int:
        fills = 0
        us = self.pmus_open().filter(pl.col("market_type") == "moneyline")
        for m in us.iter_rows(named=True):
            ev = m["event_slug"] or ""
            g = parse_pm_game(ev) or parse_pm_game(ev.split("-", 1)[1] if "-" in ev else "")
            if not g or g["league"] not in LEAGUE_TO_KALSHI:
                continue
            # Pair the US long side with a Kalshi team code by name, using the intl moneyline
            # whose outcomes are listed in slug order (validated by matching.match_kalshi_pm).
            # Skip if the name does not line up; a wrong pairing would be a double bet, not an arb.
            outcomes = self._pm_outcomes(ev if parse_pm_game(ev) else ev.split("-", 1)[1])
            long_name = (m["long_side"] or "").lower()
            if len(outcomes) != 2 or long_name not in outcomes:
                continue
            code = (g["a"] if outcomes.index(long_name) == 0 else g["b"]).upper()
            y, mo, d = g["date"].split("-")
            base = f"{LEAGUE_TO_KALSHI[g['league']]}-{y[2:]}{MONTH_ABBR[int(mo)]}{d}{g['a'].upper()}{g['b'].upper()}"
            kq = self.kalshi_quote(f"{base}-{code}")
            uq = self.pmus_quote(m["slug"])
            if not kq or not uq or None in (kq["yes_ask"], kq["yes_bid"], uq["yes_ask"], uq["yes_bid"]):
                continue
            # Kalshi YES(code) and PM US long are the same team.
            # Leg 1: YES on Kalshi + short on PM US.  Leg 2: NO on Kalshi + long on PM US.
            legs = [
                (kq["yes_ask"], "yes", kq["yes_ask_size"], round(1 - uq["yes_bid"], 4), "short", uq["yes_bid_size"]),
                (round(1 - kq["yes_bid"], 4), "no", kq["yes_bid_size"], uq["yes_ask"], "long", uq["yes_ask_size"]),
            ]
            for kp, kside, ksize, up, uside, usize in legs:
                n = int(min(ksize or 0, usize or 0, self.cfg.max_bet_dollars // max(kp + up, 0.01)))
                if n < 1:
                    continue
                total = n * (kp + up) + fee("kalshi", n, kp) + fee("pmus", n, up)
                profit = n - total
                if profit / n < self.arb_min_profit:
                    continue
                if self.state.cash["kalshi"] < n * kp + fee("kalshi", n, kp) or \
                        self.state.cash["pmus"] < n * up + fee("pmus", n, up):
                    self.log("skip", strategy="xvenue", market_id=base, reason="insufficient_cash")
                    continue
                for venue, mid, side, price in (("kalshi", f"{base}-{code}", kside, kp),
                                                ("pmus", m["slug"], uside, up)):
                    f = fee(venue, n, price)
                    pos = Position("xvenue", venue, mid, side, base, n, price, f, n * price + f, 1.0,
                                   _now().isoformat(), f"locked profit {profit:.2f}")
                    self.state.cash["kalshi" if venue == "kalshi" else "pmus"] -= pos.cost
                    self.state.positions.append(pos)
                    self.log("fill", **asdict(pos))
                fills += 1
                break
        return fills


def run_loop(trader: PaperTrader, *, interval_seconds: float, iterations: int) -> None:
    n = 0
    while iterations <= 0 or n < iterations:
        n += 1
        counts = trader.tick()
        st = trader.state
        print(f"[{_now():%Y-%m-%d %H:%M:%S}] tick {n}: {counts} equity={st.equity():.2f} "
              f"cash={ {k_: round(v, 2) for k_, v in st.cash.items()} } open={len(st.positions)}",
              flush=True)
        if iterations <= 0 or n < iterations:
            time.sleep(interval_seconds)


def status(s: Settings) -> str:
    st = PaperState.load(s.paper_dir / "state.json")
    lines = [f"equity {st.equity():.2f} (peak {st.peak_equity:.2f}) realized_pnl {st.realized_pnl:+.2f} "
             f"settled {st.settled} halted={st.halted}",
             f"cash {st.cash}", f"open positions: {len(st.positions)}"]
    for p in st.positions:
        lines.append(f"  {p.opened_at[:16]} {p.strategy:13} {p.venue:6} {p.market_id} {p.side} "
                     f"x{p.contracts} @ {p.price:.3f} cost {p.cost:.2f} p={p.prob:.3f}")
    return "\n".join(lines)


def _now() -> datetime:
    return datetime.now(UTC)
