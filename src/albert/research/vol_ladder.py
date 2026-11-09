"""In-play "vol ladder" market making, replayed on the recorded Kalshi books (research only).

Each variant rests bids on both sides (YES bids below the mid, NO bids = YES asks above it) at
several distances from a reservation price r and holds what fills to settlement:

    u      = max(min_unit, vol_k * sigma10)     sigma10: std of 1-s mid changes (last 30 s) x sqrt(10)
    r      = mid + dir_beta * (mid - EWMA_mid(dir_tau)) - inv_gamma * position
    bid_k  = floor(r - off_k * u)   ask_k = ceil(r + off_k * u)   (post-only: clamped inside the book)
    jump guard: |mid - mid 2 s ago| >= jump -> pull everything for cool_s

Quotes only while the game is in play (scheduled start .. expected expiration + 1 h) and the mid
is inside [lo, hi]; per-market net position is capped at max_pos. One pass over the events drives
every variant's own Simulator (same queue/latency/fee/write-budget model as `albert backtest l2`).

    python -m albert.research.vol_ladder run START END TAG     e.g. 2026-11-07T17 2026-11-08T06 sat
    python -m albert.research.vol_ladder report TAG [TAG...]
"""
import collections
import heapq
import itertools
import json
import math
import pickle
import sys
import time
from pathlib import Path

import polars as pl

from albert.research.inplay_hft import _by_hour, _kalshi, markets_in

OUT = Path("data/research/vol_ladder")
EPS = 1e-9
S = 10**9
LIVE = ("pending", "active", "cancelling", "cancel_pending")

LADDER = [(0.5, 1), (1.5, 2), (3.0, 3)]
FAST = {"sample_s": 0.25, "vol_n": 40, "dir_tau_s": 2.0, "jump_window_s": 1.0, "idle_s": 0.25,
        "min_requote_s": 0.1}
VARIANTS = {
    # name: params (see VolLadder); writes_per_s 10 = Basic tier (100 tokens/s), 30 = Advanced (300)
    "touch_1lot": {"levels": [(0.0, 1)], "vol_k": 0.0, "min_unit": 0.0, "inv_gamma": 0.01, "max_pos": 5, "touch": True},
    "grid_fixed": {"levels": [(1.0, 1), (3.0, 2), (6.0, 3)], "vol_k": 0.0, "min_unit": 0.01, "inv_gamma": 0.01,
                       "max_pos": 10},
    # 1-s family: sigma from 30 1-s changes, 10-s trend, 3c/2s jump guard
    "slow_vol": {"levels": LADDER, "vol_k": 1.0, "min_unit": 0.01, "inv_gamma": 0.01, "max_pos": 10},
    "slow_vol_dir_guard": {"levels": LADDER, "vol_k": 1.0, "min_unit": 0.01, "inv_gamma": 0.01, "max_pos": 10,
                               "dir_beta": 0.5, "jump": 0.03, "cool_s": 5.0},
    # sub-second family: sigma from 40 0.25-s changes (10 s), 2-s trend, 2c/1s guard, requote 0.1 s
    "fast_vol_dir_guard": dict(levels=LADDER, vol_k=1.0, min_unit=0.01, inv_gamma=0.01, max_pos=10,
                               dir_beta=0.5, jump=0.02, cool_s=5.0, **FAST),
    "fast_vol_dir_guard_adv": dict(levels=LADDER, vol_k=1.0, min_unit=0.01, inv_gamma=0.01, max_pos=10,
                                   dir_beta=0.5, jump=0.02, cool_s=5.0, writes_per_s=30.0, **FAST),
    "fast_deep": dict(levels=[(2.0, 2), (4.0, 3)], vol_k=1.0, min_unit=0.01, inv_gamma=0.01, max_pos=10,
                      **FAST),
}


LMSR_FAST = {"idle_s": 0.25, "min_requote_s": 0.1, "writes_per_s": 30.0}
LMSR_VARIANTS = {
    # b = LMSR liquidity (contracts); 3 one-contract units shown per side; Advanced write budget
    "lmsr_fixed_b10": dict(cls="lmsr", b=10.0, anchor="fixed", **LMSR_FAST),
    "lmsr_mid_b10": dict(cls="lmsr", b=10.0, anchor="mid", **LMSR_FAST),
    "lmsr_mid_b25": dict(cls="lmsr", b=25.0, anchor="mid", **LMSR_FAST),
    "lmsr_ewma60_b10": dict(cls="lmsr", b=10.0, anchor="ewma", anchor_tau_s=60.0, **LMSR_FAST),
}


PYR = [(0.5, 1), (1.5, 2), (3.0, 3), (5.0, 5), (8.0, 8)]
PYRAMID_VARIANTS = {
    # small near the mid, larger toward the extremes; max 20 net per market
    "pyr_price": {"cls": "pyr", "levels": PYR, "space": "price", "vol_size": False, "max_pos": 20},
    "pyr_logit": {"cls": "pyr", "levels": PYR, "space": "logit", "vol_size": False, "max_pos": 20},
    "pyr_logit_vol": {"cls": "pyr", "levels": PYR, "space": "logit", "vol_size": True, "max_pos": 20},
    "pyr_logit_vol_fast": dict(cls="pyr", levels=PYR, space="logit", vol_size=True, max_pos=20, **FAST,
                               writes_per_s=30.0),
    "pyr_far_logit_vol": {"cls": "pyr", "levels": [(3.0, 2), (5.0, 4), (8.0, 8), (12.0, 16)], "space": "logit",
                              "vol_size": True, "max_pos": 30},
}
# market -> [short-rate, long-rate, t_ns]: EWMA traded contracts per second (60 s and 15 min)
TRADE_RATE: dict[str, list] = {}
RATE_TAU = (60 * S, 900 * S)


def note_trade(m, t_ns, count):
    r = TRADE_RATE.get(m)
    if r is None:
        r = TRADE_RATE[m] = [0.0, 0.0, t_ns]
    dt = t_ns - r[2]
    for i, tau in enumerate(RATE_TAU):
        r[i] = r[i] * math.exp(-dt / tau) + count * S / tau
    r[2] = t_ns


def trade_rates(m, t_ns):
    r = TRADE_RATE.get(m)
    if r is None:
        return 0.0, 0.0
    dt = t_ns - r[2]
    return tuple(r[i] * math.exp(-dt / tau) for i, tau in enumerate(RATE_TAU))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class VolLadder:
    def __init__(self, bounds: dict[str, tuple[int, int]], levels, vol_k=1.0, min_unit=0.01,
                 inv_gamma=0.01, max_pos=10, dir_beta=0.0, dir_tau_s=10.0, jump=0.0, cool_s=5.0,
                 lo=0.05, hi=0.95, touch=False, min_requote_s=0.25, idle_s=1.0, sample_s=1.0,
                 vol_n=30, jump_window_s=2.0, writes_per_s=10.0):
        self.bounds, self.levels, self.vol_k, self.min_unit = bounds, levels, vol_k, min_unit
        self.inv_gamma, self.max_pos, self.dir_beta = inv_gamma, max_pos, dir_beta
        self.dir_tau, self.jump, self.cool = dir_tau_s * S, jump, int(cool_s * S)
        self.lo, self.hi, self.touch = lo, hi, touch
        self.requote, self.idle = int(min_requote_s * S), int(idle_s * S)
        self.sample, self.vol_n, self.jump_win = int(sample_s * S), vol_n, int(jump_window_s * S)
        self.st: dict[str, dict] = {}
        self.peak_collateral = 0.0
        self.peak_gross = 0.0
        self._last_cap_ns = 0

    def _state(self, m):
        s = self.st.get(m)
        if s is None:
            s = self.st[m] = {"samples": collections.deque(maxlen=self.vol_n + 1), "next_sample": 0, "ewma": None,
                              "ewma_ns": 0, "top": None, "last_decide": 0, "cool_until": 0,
                              "nfills": 0, "last_place": {}}
        return s

    def _cancel_all(self, sim, m):
        for o in sim.live_orders(m):
            sim.cancel(o.oid)

    def on_event(self, sim, m):
        b = self.bounds.get(m)
        if b is None or m not in sim.synced:
            return
        now = sim.now
        ids = sim.by_market[m]
        if len(ids) > 64:   # drop finished orders so per-market scans stay short
            sim.by_market[m] = [i for i in ids if sim.orders[i].status in LIVE]
        if now - self._last_cap_ns >= S:     # capital use, sampled once a second
            self._last_cap_ns = now
            col = sum(sim.orders[i].rem * sim.orders[i].price for mm in self.bounds
                      for i in sim.by_market.get(mm, ()) if sim.orders[i].status in ("pending", "active"))
            self.peak_collateral = max(self.peak_collateral, col)
            self.peak_gross = max(self.peak_gross, sum(abs(v) for v in sim.position.values()))
        live = sim.live_orders(m)
        if not (b[0] <= now < b[1]):
            if live:
                self._cancel_all(sim, m)
            return
        qv = sim.quotes[m][1]
        bid, ask = qv[-1] if qv else (None, None)
        if bid is None or ask is None or ask <= bid:
            if live:
                self._cancel_all(sim, m)
            return
        mid = (bid + ask) / 2
        s = self._state(m)
        # 1-s mid samples for sigma and the jump guard; EWMA for direction
        while now >= s["next_sample"]:
            s["samples"].append((now, mid))
            s["next_sample"] = (now // self.sample + 1) * self.sample
        if s["ewma"] is None:
            s["ewma"] = mid
        else:
            a = 1 - math.exp(-(now - s["ewma_ns"]) / self.dir_tau)
            s["ewma"] += a * (mid - s["ewma"])
        s["ewma_ns"] = now
        fills_changed = s["nfills"] != self._fills_in(sim, m)
        top = (bid, ask)
        if top == s["top"] and not fills_changed and now - s["last_decide"] < self.idle:
            return
        s["top"], s["last_decide"] = top, now
        s["nfills"] = self._fills_in(sim, m)
        if self.jump:
            old = [x for t, x in s["samples"] if now - t <= self.jump_win]
            if old and abs(mid - old[0]) >= self.jump - EPS:
                s["cool_until"] = now + self.cool
        if now < s["cool_until"] or not (self.lo <= mid <= self.hi):
            if live:
                self._cancel_all(sim, m)
            return
        pos = sim.position[m]
        want = self._want(s, bid, ask, mid, pos)
        self._reconcile(sim, m, live, want, s)

    def _fills_in(self, sim, m):
        return sim._nfills.get(m, 0)

    def _want(self, s, bid, ask, mid, pos):
        """{(side, price): qty} for YES bids and NO bids."""
        xs = [x for _, x in s["samples"]]
        d = [b - a for a, b in itertools.pairwise(xs)]
        sigma10 = math.sqrt(sum(v * v for v in d) / len(d) * 10 * S / self.sample) if d else 0.0
        u = max(self.min_unit, self.vol_k * sigma10)
        r = mid - self.inv_gamma * pos
        if self.dir_beta:
            r += self.dir_beta * (mid - s["ewma"])
        want: dict[tuple[str, float], float] = collections.defaultdict(float)
        cy = cn = 0.0
        for off, qty in self.levels:
            if self.touch:
                yb, ya = bid, ask
                if self.inv_gamma and abs(pos) > EPS:
                    shift = round(self.inv_gamma * abs(pos), 2)
                    if pos > 0:
                        yb, ya = bid - shift, max(bid + 0.01, ask - shift)
                    else:
                        yb, ya = min(ask - 0.01, bid + shift), ask + shift
            else:
                yb = math.floor((r - off * u) * 100 + 1e-6) / 100
                ya = math.ceil((r + off * u) * 100 - 1e-6) / 100
            yb = min(round(yb, 2), round(ask - 0.01, 2))     # post-only
            ya = max(round(ya, 2), round(bid + 0.01, 2))
            if pos + cy + qty <= self.max_pos + EPS and 0.01 <= yb <= 0.99:
                want[("yes", yb)] += qty
                cy += qty
            if pos - cn - qty >= -self.max_pos - EPS and 0.01 <= ya <= 0.99:
                want[("no", round(1 - ya, 2))] += qty
                cn += qty
        return want

    def _reconcile(self, sim, m, live, want, s):
        have: dict[tuple[str, float], list] = collections.defaultdict(list)
        for o in live:
            have[(o.side, round(o.price, 2))].append(o)
        for k, os_ in have.items():
            if k not in want:
                for o in os_:
                    sim.cancel(o.oid)
        for (side, price), qty in want.items():
            cur = sum(o.rem for o in have.get((side, price), []))
            if cur + EPS >= qty:
                continue
            if sim.now - s["last_place"].get((side, price), -(2**62)) < self.requote:
                continue
            oid = sim.place(m, side, price, qty - cur)
            if oid is not None:
                s["last_place"][(side, price)] = sim.now


def _softplus(z):
    return z + math.log1p(math.exp(-z)) if z > 0 else math.log1p(math.exp(z))


class LmsrMaker(VolLadder):
    """Quotes the binary LMSR cost curve. Traders' net YES holdings are x = -position, so the
    LMSR state is a = b*logit(p0) - position and the price of one more YES unit from state a is
    b*(softplus((a+1)/b) - softplus(a/b)). We show ``units`` one-contract asks (we sell YES) and
    bids (we buy YES) at successive units, asks rounded up and bids down to the cent, post-only.
    ``anchor``: fixed = p0 is the mid at the first in-play quote and never moves (pure LMSR: only
    our fills move the price); mid = re-centred on the live mid every decision; ewma = on an
    EWMA of the mid (``anchor_tau_s``)."""

    def __init__(self, bounds, b=10.0, units=3, anchor="mid", anchor_tau_s=60.0, max_pos=10, **kw):
        super().__init__(bounds, levels=[], vol_k=0.0, min_unit=0.0, inv_gamma=0.0, max_pos=max_pos,
                         dir_tau_s=anchor_tau_s, **kw)
        self.b, self.units, self.anchor = b, units, anchor

    def _want(self, s, bid, ask, mid, pos):
        if self.anchor == "fixed":
            p0 = s.setdefault("p0", mid)
        elif self.anchor == "ewma":
            p0 = s["ewma"]
        else:
            p0 = mid
        p0 = min(max(p0, 0.02), 0.98)
        b = self.b
        a = b * math.log(p0 / (1 - p0)) - pos
        want: dict[tuple[str, float], float] = collections.defaultdict(float)
        for k in range(self.units):
            if pos - (k + 1) >= -self.max_pos - EPS:      # sell YES (= NO bid) for unit k+1
                x = a + k
                ya = math.ceil(b * (_softplus((x + 1) / b) - _softplus(x / b)) * 100 - 1e-6) / 100
                ya = max(round(ya, 2), round(bid + 0.01, 2))
                if ya <= 0.99:
                    want[("no", round(1 - ya, 2))] += 1.0
            if pos + (k + 1) <= self.max_pos + EPS:       # buy YES for unit k+1
                x = a - k
                yb = math.floor(b * (_softplus(x / b) - _softplus((x - 1) / b)) * 100 + 1e-6) / 100
                yb = min(round(yb, 2), round(ask - 0.01, 2))
                if yb >= 0.01:
                    want[("yes", yb)] += 1.0
        return want


def _logit(p):
    return math.log(p / (1 - p))


def _expit(z):
    return 1 / (1 + math.exp(-z))


class PyramidLadder(VolLadder):
    """Small orders near the mid, progressively larger toward the extremes.

    * ``space="logit"``: levels are equal steps in log-odds around the reservation price, with
      sigma measured on logit(mid) changes, so the ladder compresses toward 0 and 1 (level- and
      skew-aware); ``"price"`` uses cents as `VolLadder`.
    * ``vol_size``: sizes x clip(sqrt(60-s traded rate / 15-min rate), 0.5, 2).
    * Inventory: reservation shifted by ``inv_gamma`` per contract, and the side that adds to the
      position is scaled by (1 - |pos|/max_pos) while the side that reduces it by (1 + |pos|/max_pos).
    """

    def __init__(self, bounds, levels, space="logit", vol_size=True, **kw):
        kw.setdefault("vol_k", 1.0)
        kw.setdefault("min_unit", 0.01)
        kw.setdefault("inv_gamma", 0.005)
        super().__init__(bounds, levels, **kw)
        self.space, self.vol_size = space, vol_size
        self._m = None

    def on_event(self, sim, m):
        self._m = m
        super().on_event(sim, m)

    def _want(self, s, bid, ask, mid, pos):
        xs = [x for _, x in s["samples"]]
        if self.space == "logit":
            xs = [_logit(min(max(x, 0.01), 0.99)) for x in xs]
        d = [b - a for a, b in itertools.pairwise(xs)]
        sigma10 = math.sqrt(sum(v * v for v in d) / len(d) * 10 * S / self.sample) if d else 0.0
        r = min(max(mid - self.inv_gamma * pos, 0.02), 0.98)
        if self.space == "logit":
            u = max(self.min_unit / (r * (1 - r)), self.vol_k * sigma10)
            base = _logit(r)
            price = lambda off: _expit(base + off * u)
        else:
            u = max(self.min_unit, self.vol_k * sigma10)
            price = lambda off: r + off * u
        mult = 1.0
        if self.vol_size:
            short, long_ = trade_rates(self._m, s["last_decide"])
            mult = min(max(math.sqrt(short / long_), 0.5), 2.0) if long_ > 0 else 0.5
        f = min(abs(pos) / self.max_pos, 1.0)
        add_long, cut_long = (1 - f, 1 + f) if pos > 0 else (1 + f, 1 - f) if pos < 0 else (1.0, 1.0)
        want: dict[tuple[str, float], float] = collections.defaultdict(float)
        cy = cn = 0.0
        for off, qty in self.levels:
            yb = min(round(math.floor(price(-off) * 100 + 1e-6) / 100, 2), round(ask - 0.01, 2))
            ya = max(round(math.ceil(price(off) * 100 - 1e-6) / 100, 2), round(bid + 0.01, 2))
            qy = round(qty * mult * add_long)
            qn = round(qty * mult * cut_long)
            if qy >= 1 and pos + cy + qy <= self.max_pos + EPS and 0.01 <= yb <= 0.99:
                want[("yes", yb)] += qy
                cy += qy
            if qn >= 1 and pos - cn - qn >= -self.max_pos - EPS and 0.01 <= ya <= 0.99:
                want[("no", round(1 - ya, 2))] += qn
                cn += qn
        return want


def patch_sim(sim):
    """Count fills per market (cheap fill-change detection for the strategy)."""
    sim._nfills = collections.defaultdict(int)
    orig = sim._fill

    def _fill(o, qty, how):
        n = len(sim.fills)
        orig(o, qty, how)
        if len(sim.fills) > n:
            sim._nfills[o.market] += 1
    sim._fill = _fill


def speedups():
    """Same results, less work: best price by early exit (not summing every level), and the
    online execution classifier on a deque with its stale-trade sweep once a second."""
    from albert.backtest import l2

    def best(self, side):
        lv = self.levels[side]
        for p in sorted(lv, reverse=True):
            t = lv[p].ext_total()
            if t > EPS:
                return (p, t)
        return None
    def cached_best(self, side):
        c = self.__dict__.setdefault("_bc", {})
        if side not in c:
            c[side] = best(self, side)
        return c[side]
    if __import__("os").environ.get("NOCACHE"):
        l2.Market.best = best
        return Clf_factory(l2)
    l2.Market.best = cached_best
    orig_apply = l2.Simulator.apply

    def apply(self, ev):
        self._run_schedule(ev.recv_ns)   # order activations may read (and cache) the book first
        if ev.kind == "snapshot" or (ev.kind == "delta" and ev.market in self.synced):
            m = self.markets[ev.market]
            c = m.__dict__.get("_bc")
            if c:
                if ev.kind == "snapshot":
                    c.clear()
                else:
                    side = ev.msg["side"]
                    b = c.get(side, 0)
                    if b is None or side not in c or float(ev.msg["price_dollars"]) >= b[0] - EPS:
                        c.pop(side, None)
        orig_apply(self, ev)
    l2.Simulator.apply = apply
    return Clf_factory(l2)


def Clf_factory(l2):

    class Clf(l2.OnlineClassifier):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.buf = collections.deque()
            self._sweep = 0

        def release(self, now_ns):
            cut = now_ns - self.hold
            out = []
            while self.buf and self.buf[0].recv_ns <= cut:
                out.append(self.buf.popleft())
            if now_ns >= self._sweep:
                self._sweep = now_ns + S
                old = now_ns - 2 * S
                for m, w in self.waiting.items():
                    if w and w[0][1] < old:
                        self.stats["trades_unmatched"] += sum(1 for _, t in w if t < old)
                        self.waiting[m] = [x for x in w if x[1] >= old]
            return out
    return Clf


def events(files, keep):
    from albert.backtest import l2
    Clf = speedups()
    for hour_files in _by_hour(files):
        def one(p):
            for t, k, m, msg in _kalshi(p, keep):
                yield l2.Event(t, k, m, msg)
        clf = Clf()
        for ev in heapq.merge(*[one(p) for p in hour_files], key=lambda e: e.recv_ns):
            clf.push(ev)
            yield from clf.release(ev.recv_ns)
        yield from clf.release(2**62)


def run(start, end, tag, family="ladder"):
    from albert.backtest import l2
    from albert.backtest.l2 import SimConfig, Simulator, stream_files
    from albert.config import settings
    from albert.research.volatile import GAME_SERIES, _in_play_bounds, market_meta, series_of

    OUT.mkdir(parents=True, exist_ok=True)
    s = settings()
    files = stream_files(s.raw_dir / "stream", start, end)
    log(tag, len(files), "files")
    tickers = markets_in(files)
    games = sorted(t for t in tickers if series_of(t) in GAME_SERIES)
    meta = market_meta(games, Path("data/cache/volatile_markets.json"))
    bounds = _in_play_bounds(meta)
    log(tag, len(tickers), "markets,", len(games), "game markets,", len(bounds), "with in-play bounds")
    fee_info = {r["series_ticker"]: {"fee_type": r["fee_type"], "fee_multiplier": r["fee_multiplier"]}
                for r in pl.read_parquet("data/tables/kalshi_series.parquet").iter_rows(named=True)}
    sims, strats = {}, {}
    variants = {"lmsr": LMSR_VARIANTS, "pyramid": PYRAMID_VARIANTS}.get(family, VARIANTS)
    for name, p in variants.items():
        sim = Simulator(SimConfig(latency_ms=20.0, queue="risk_averse", fee_info=fee_info,
                                  writes_per_s=p.get("writes_per_s", 10.0)))
        patch_sim(sim)
        kw = {k: v for k, v in p.items() if k != "cls"}
        strats[name] = {"lmsr": LmsrMaker, "pyr": PyramidLadder}.get(p.get("cls"), VolLadder)(bounds, **kw)
        sims[name] = sim
    # The recorded (external) book, and so its top-of-book history, is the same in every sim:
    # our orders never change external size. Share one history instead of one per variant.
    shared = next(iter(sims.values())).quotes
    for sim in sims.values():
        sim.quotes = shared
    n, t0 = 0, time.time()
    cap = int(__import__("os").environ.get("MAX_EVENTS", "0"))
    for ev in events(files, set(bounds)):
        if cap and n >= cap:
            break
        if ev.kind == "trade":
            note_trade(ev.market, ev.recv_ns, float(ev.msg.get("count_fp") or 0))
        for name, sim in sims.items():
            sim.apply(ev)
            strats[name].on_event(sim, ev.market)
        n += 1
        if n % 1_000_000 == 0:
            log(tag, f"{n:,} events", f"{n / (time.time() - t0):,.0f}/s",
                {k: len(v.fills) for k, v in sims.items()})
    log(tag, "done", f"{n:,} events")
    out = {"tag": tag, "window": [start, end], "events": n, "bounds": bounds, "variants": {}}
    for name, sim in sims.items():
        st = strats[name]
        rows = l2.markouts(sim)
        for r in rows:
            r["level"] = None
        out["variants"][name] = {"params": variants[name], "fills": rows, "diag": dict(sim.diag),
                                 "orders": len(sim.orders), "peak_collateral": st.peak_collateral,
                                 "peak_gross": st.peak_gross,
                                 "end_position": {k: v for k, v in sim.position.items() if abs(v) > EPS}}
    with open(OUT / f"{tag}.pkl", "wb") as f:
        pickle.dump(out, f)
    log(tag, "saved", {k: len(v["fills"]) for k, v in out["variants"].items()})


def results(markets, cache: Path):
    import httpx
    got = json.loads(cache.read_text()) if cache.exists() else {}
    c = httpx.Client(timeout=30)
    for t in markets:
        if t in got and got[t] is not None:
            continue
        for a in range(6):
            r = c.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{t}")
            if r.status_code == 429:
                time.sleep(1 + a)
                continue
            m = r.json().get("market", {}) if r.status_code == 200 else {}
            res = m.get("result")
            got[t] = 1.0 if res == "yes" else 0.0 if res == "no" else None
            break
        time.sleep(0.1)
    cache.write_text(json.dumps(got))
    return got


def report(tags):
    from albert.backtest.l2 import _cluster_t, pair_breakdown
    from albert.config import settings
    from albert.reports import Report

    runs = [pickle.loads((OUT / f"{t}.pkl").read_bytes()) for t in tags]
    frames = []
    for r in runs:
        for name, v in r["variants"].items():
            if v["fills"]:
                frames.append(pl.DataFrame(v["fills"], infer_schema_length=None)
                              .with_columns(variant=pl.lit(name), night=pl.lit(r["tag"])))
    fills = pl.concat(frames, how="diagonal_relaxed")
    res = results(sorted(fills["market"].unique().to_list()), OUT / "results.json")
    # Exact maker fees (the account settles to $0.0001); the simulator's per-fill rounding up to
    # 1c is kept as fee_cent for reference (it charges a 0.01-contract partial fill 1c).
    ser = pl.read_parquet("data/tables/kalshi_series.parquet").select(
        pl.col("series_ticker").alias("series"), "fee_type", "fee_multiplier")
    fills = (fills.rename({"fee": "fee_cent"})
             .with_columns(series=pl.col("market").str.extract(r"^([^-]+)"))
             .join(ser, on="series", how="left")
             .with_columns(fee=pl.when(pl.col("fee_type").fill_null("").str.contains("maker_fee"))
                           .then((0.0175 * pl.col("fee_multiplier").fill_null(1.0) * pl.col("qty")
                                  * pl.col("price") * (1 - pl.col("price")) * 1e4).ceil() / 1e4)
                           .otherwise(0.0)))
    fills = fills.with_columns(
        pl.col("market").replace_strict(res, default=None, return_dtype=pl.Float64).alias("final_yes"),
        pl.col("market").str.replace(r"-[^-]+$", "").alias("event"),
    ).with_columns(
        pl.col("final_yes").is_not_null().alias("settled"),
        (pl.when(pl.col("side") == "yes").then(pl.col("final_yes")).otherwise(1 - pl.col("final_yes"))
         - pl.col("price")).alias("gross_per_contract"),
    ).with_columns(pnl_per_contract=pl.col("gross_per_contract") - pl.col("fee") / pl.col("qty"))
    rep = Report(settings().reports_dir, "research-vol-ladder-inplay")
    rep.json("Run", {"nights": [{"tag": r["tag"], "window": r["window"], "events": r["events"],
                                 "in_play_markets": len(r["bounds"])} for r in runs],
                     "model": "albert.backtest.l2.Simulator: 20 ms latency, risk-averse queue, post-only, "
                              "Kalshi maker fees per series, 10 writes/s (cancel = 0.2). Positions held to "
                              "settlement. t-stats clustered by event (game).",
                     "variants": {k: v for r in runs for k, v in
                                  {n: x["params"] for n, x in r["variants"].items()}.items()}})
    rows, summ = [], {}
    names = list(dict.fromkeys(n for r in runs for n in r["variants"]))
    for name in names:
        f = fills.filter(pl.col("variant") == name)
        d = {"variant": name}
        if f.height:
            pb = pair_breakdown(f.filter(pl.col("settled")))
            mean, t, k = _cluster_t(f.filter(pl.col("settled")), "pnl_per_contract", "event")
            gm, gt, _ = _cluster_t(f.filter(pl.col("settled")), "gross_per_contract", "event")
            d.update({"contracts": round(f["qty"].sum(), 1), "games": k,
                      "gross_c_per_contract": round(100 * gm, 2), "t_gross": None if gt is None else round(gt, 2),
                      "fee_cent_rounding_$": round(f["fee_cent"].sum(), 2),
                      "pnl_$": round(pb.get("pnl", 0.0), 2), "pnl_c_per_contract": round(100 * mean, 2),
                      "t_game": None if t is None else round(t, 2),
                      "paired_share": round(pb.get("paired_share") or 0, 3),
                      "pair_pnl_$": round(pb.get("pair_pnl", 0.0), 2),
                      "directional_pnl_$": round(pb.get("directional_pnl", 0.0), 2),
                      "fees_$": round(pb.get("fees", 0.0), 2),
                      "games_positive": round(pb.get("events_positive") or 0, 2)})
            for col in ("mo_1s", "mo_10s", "mo_60s", "mx_60s"):
                m_, t_, _ = _cluster_t(f, col, "event")
                d[f"{col}_c"] = None if m_ is None else round(100 * m_, 2)
                d[f"{col}_t"] = None if t_ is None else round(t_, 2)
        peaks = [r["variants"][name] for r in runs if name in r["variants"]]
        d["peak_collateral_$"] = round(max(p["peak_collateral"] for p in peaks), 2)
        d["peak_gross_contracts"] = max(p["peak_gross"] for p in peaks)
        d["orders"] = sum(p["orders"] for p in peaks)
        d["throttled"] = sum(p["diag"].get("throttled_places", 0) + p["diag"].get("throttled_cancels", 0)
                             for p in peaks)
        rows.append(d)
        summ[name] = d
        # per night
        for r in runs:
            if name not in r["variants"]:
                continue
            fn = f.filter(pl.col("night") == r["tag"]) if f.height else f
            if fn.height:
                pbn = pair_breakdown(fn.filter(pl.col("settled")))
                summ[f"{name}@{r['tag']}"] = {"pnl_$": round(pbn.get("pnl", 0.0), 2),
                                              "contracts": round(fn["qty"].sum(), 1)}
    table = pl.DataFrame(rows, infer_schema_length=None)
    rep.table("Variants (settled PnL; markouts per contract in cents)", table, save_as="variants")
    rep.json("Per night", {k: v for k, v in summ.items() if "@" in k})
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    fills.write_parquet(OUT / "fills.parquet")
    print(rep.save())
    with pl.Config(tbl_cols=-1, tbl_width_chars=250):
        print(table)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*sys.argv[2:6])
    else:
        report(sys.argv[2:])
