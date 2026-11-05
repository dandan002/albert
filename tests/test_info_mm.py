import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from albert.backtest.l2 import SimConfig, _mark_executions, events_from_rows
from albert.research import info_mm as im

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIX / name).read_text().splitlines()]


def _events(name: str):
    evs = events_from_rows(_rows(name))
    _mark_executions(evs)
    return evs


def _tape_orders() -> pl.DataFrame:
    """Recorded trade prints (KXHIGHLAX, 2026-11-05 16 UTC) as taker orders."""
    return im.hist_orders(im.ws_trade_rows(_rows("ws_trades_tape.jsonl"))).collect()


# ---- PIN ----------------------------------------------------------------------------------------
def test_pin_likelihood_matches_direct_mixture_and_is_stable():
    B = np.array([0.0, 3, 12, 1, 7])
    S = np.array([2.0, 0, 1, 9, 4])
    a, d, mu, eb, es = 0.4, 0.3, 6.0, 1.5, 2.0
    x = np.array([math.log(a / (1 - a)), math.log(d / (1 - d)), math.log(mu), math.log(eb), math.log(es)])
    lgf = np.array([math.lgamma(b + 1) + math.lgamma(s + 1) for b, s in zip(B, S)])

    def pois(k, lam):
        return math.exp(-lam) * lam ** k / math.factorial(int(k))
    direct = sum(math.log((1 - a) * pois(b, eb) * pois(s, es) + a * d * pois(b, eb) * pois(s, es + mu)
                          + a * (1 - d) * pois(b, eb + mu) * pois(s, es)) for b, s in zip(B, S))
    assert -im.pin_nll(x, B, S, lgf) == pytest.approx(direct, rel=1e-10)
    # counts in the thousands overflow the direct form but not Lin & Ke's
    big = im.pin_nll(x, B * 1000, S * 1000, lgf * 0)
    assert math.isfinite(big)


def test_fit_pin_recovers_known_parameters_and_moment_estimate_is_unbiased_at_zero():
    rng = np.random.default_rng(7)
    n, a, mu, eb, es = 400, 0.35, 20.0, 4.0, 5.0
    ev, bad = rng.random(n) < a, rng.random(n) < 0.5
    B = rng.poisson(eb + mu * (ev & ~bad))
    S = rng.poisson(es + mu * (ev & bad))
    r = im.fit_pin(B, S)
    true = a * mu / (a * mu + eb + es)
    assert r is not None and abs(r["pin"] - true) < 0.05 and abs(r["alpha"] - a) < 0.08
    # no informed flow: the bias-adjusted moment estimate is ~0, the raw VPIN is not
    B0, S0 = rng.poisson(0.8, 2000), rng.poisson(0.8, 2000)
    assert im.pin_mom(B0, S0) < 0.03 and im.vpin(B0, S0) > 0.4
    assert im.fit_pin([1, 0], [0, 1]) is None          # too small


def test_bucket_counts_and_pin_on_recorded_tape():
    o = _tape_orders()
    assert o.height == 117 and 0 < o["buy"].sum() < o.height      # 222 prints -> 117 taker orders
    o1 = o.with_columns((pl.col("us") // 60_000_000).alias("bucket"))
    c = im.bucket_counts(o1, ["ticker"])
    span = o1["bucket"].max() - o1["bucket"].min() + 1
    assert c.height == span and c["B"].sum() + c["S"].sum() == o.height
    assert (c["B"] + c["S"] == 0).any()                             # empty minutes filled with zeros
    fits = im.fit_units(c, ["ticker"])
    assert fits.height == 1
    r = fits.row(0, named=True)
    assert 0 <= r["pin"] <= 1 and 0 <= r["pin_mom"] <= r["vpin"] <= 1


# ---- Glosten–Milgrom ---------------------------------------------------------------------------
def test_gm_binary_is_bayes_and_two_point_is_pi_delta():
    for p in (0.1, 0.5, 0.83):
        for pi in (0.0, 0.2, 0.7):
            bid, ask = im.gm_binary(p, pi)
            # Bayes: P(buy|V=1) = (1+pi)/2, P(buy|V=0) = (1-pi)/2
            assert ask == pytest.approx(p * (1 + pi) / 2 / (p * (1 + pi) / 2 + (1 - p) * (1 - pi) / 2))
            assert bid <= p <= ask
            assert im.gm_half_binary(p, pi) == pytest.approx((ask - bid) / 2)
    assert im.gm_binary(0.3, 1.0) == pytest.approx((0.0, 1.0))
    # two-point value p +- delta: E[V | buy] - p = pi * delta
    p, d, pi = 0.4, 0.05, 0.3
    e_buy = ((p + d) * (1 + pi) / 2 + (p - d) * (1 - pi) / 2) / 1.0
    assert e_buy - p == pytest.approx(im.gm_half_two_point(pi, d))


def test_kyle_lambda_and_sign_autocorr():
    rng = np.random.default_rng(3)
    q = rng.normal(0, 50, 500)
    dp = 0.0004 * q + rng.normal(0, 0.005, 500)
    k = im.kyle_lambda(dp, q)
    assert k["lam"] == pytest.approx(0.0004, rel=0.15) and k["t_lam"] > 10
    s = np.repeat([1.0, -1.0] * 50, 5)                               # persistent runs
    assert im.sign_autocorr(s) == pytest.approx(0.6, abs=0.05)    # 4 of 5 neighbours agree


# ---- Das market maker ----------------------------------------------------------------------------
def test_zero_profit_quotes_are_the_gm_fixed_point():
    das = im.DasMM(alpha=0.4, sigma=0.03)
    bel = das._prior(0.42)
    bid, ask = das.zero_profit(bel, 0.4, 0.0175)
    assert bid is not None and ask is not None and bid < 0.42 < ask

    def e_given(price, buy):
        up = im.ndtr((das.v - price) / das.sigma)
        like = 0.4 * up + 0.3 if buy else 0.4 * (1 - up) + 0.3
        return float((das.v * bel * like).sum() / (bel * like).sum())
    fee = lambda p: 0.0175 * p * (1 - p)   # noqa: E731
    assert ask - e_given(ask, True) >= fee(ask) - 1e-9                       # ask earns
    assert (ask - 0.01) - e_given(ask - 0.01, True) < fee(ask - 0.01)        # one tick tighter loses
    assert e_given(bid, False) - bid >= fee(bid) - 1e-9
    assert e_given(bid + 0.01, False) - (bid + 0.01) < fee(bid + 0.01)
    # informed buying moves the belief up
    for _ in range(5):
        lb = 0.4 * das._up[int(round(ask * 100)) - 1] + 0.3
        bel = bel * lb / (bel * lb).sum()
    assert float(das.v @ bel) > 0.42


def _quotes_at_place(evs, strat, latency_ms=50.0):
    sim = im.Simulator(SimConfig(latency_ms=latency_ms, writes_per_s=10.0))
    placed = []
    for ev in evs:
        sim.apply(ev)
        strat.observe(sim, ev)
        n = len(sim.orders)
        q = sim.markets[ev.market].quote() if ev.market in sim.synced else None
        strat.on_event(sim, ev.market)
        zp = strat.zp.get(ev.market)        # the zero-profit quote the decision used
        for oid in list(sim.orders)[n:]:
            placed.append((ev.recv_ns, oid, q, zp))
    return sim, placed


def test_das_quotes_improve_inside_the_zero_profit_quote_and_never_cross():
    evs = _events("ws_stream_ladder_arb.jsonl")
    das = im.DasMM(alpha=0.2, sigma=0.03, lo=0.0, hi=1.0, cooldown_s=0)
    sim, placed = _quotes_at_place(evs, das)
    assert placed and das.trades_seen == 10
    for _, oid, q, (bid_star, ask_star) in placed:
        o = sim.orders[oid]
        if o.side == "yes":
            assert o.price == pytest.approx(q["yes_bid"] + 0.01) and o.price <= bid_star + 1e-9
            assert o.price < q["yes_ask"]
        else:
            assert 1 - o.price == pytest.approx(q["yes_ask"] - 0.01) and 1 - o.price >= ask_star - 1e-9
            assert 1 - o.price > q["yes_bid"]
    # a very informed market (alpha -> 1) quotes far less often
    strict = im.DasMM(alpha=0.95, sigma=0.03, lo=0.0, hi=1.0, cooldown_s=0)
    _, placed_strict = _quotes_at_place(evs, strict)
    assert len(placed_strict) < len(placed)


def test_das_decisions_do_not_depend_on_future_events():
    evs = _events("ws_stream_ladder_arb.jsonl")
    kw = dict(alpha=0.2, sigma=0.03, lo=0.0, hi=1.0, cooldown_s=0)
    _, full = _quotes_at_place(evs, im.DasMM(**kw))
    assert full
    idx = {e.recv_ns: i for i, e in enumerate(evs)}
    for t, _, _, _ in full[:5] + full[-3:]:
        _, part = _quotes_at_place(evs[:idx[t] + 1], im.DasMM(**kw))
        assert [(a, b) for a, b, _, _ in part] == [(a, b) for a, b, _, _ in full if a <= t]


def test_recorder_tape_and_exec_rows():
    evs = _events("ws_stream_wide.jsonl")
    rec = im.Recorder()
    sim, st = im.replay(iter(evs), None, SimConfig(latency_ms=50), rec)
    n_trades = sum(1 for e in evs if e.kind == "trade")
    n_exec = sum(1 for e in evs if e.kind == "delta" and e.is_exec)
    assert st["events"] == len(evs) and len(rec.trades) == n_trades == 32 and len(rec.execs) == n_exec > 0
    orders = im.l2_orders(rec.trades_frame())
    assert orders.height == 1 and orders["qty"].item() == pytest.approx(sum(t[4] for t in rec.trades))
    ex = im.exec_frame(sim, rec, {"KXHIGHDEN-26NOV05-B60.5": 0.0}, {})
    assert ex.height == n_exec and ex["bid1"].is_not_null().all()
    # the sweep cleared levels; settlement uses the result on the maker's side, no fee off maker-fee series
    assert ex["cleared"].any()
    r = ex.row(0, named=True)
    assert r["settle"] == pytest.approx((1.0 if r["side"] == "no" else 0.0) - r["price"])


def test_rolling_alpha_uses_only_previous_buckets():
    o = _tape_orders()
    orders = (o.rename({"ticker": "market"}).with_columns((pl.col("us") * 1000).alias("t_ns"))
              .with_columns((pl.col("t_ns") // (im.BUCKET_S * 10**9)).alias("bucket")))
    bk = orders.select("market", "bucket").unique().with_columns(
        (pl.col("bucket") * im.BUCKET_S * 10**9).alias("bucket_ns"), pl.lit(1.0).alias("range_c"),
        pl.lit(False).alias("volatile"))
    roll = im.rolling_alpha(orders, bk, n_prev=12).sort("bucket")
    assert roll.height == bk.height and roll["pin_prev"][0] is None      # nothing before the first bucket
    # pin_prev of a bucket equals pin_mom over the buckets strictly before it
    c = orders.group_by("bucket").agg(pl.col("buy").sum().alias("B"), (~pl.col("buy")).sum().alias("S")).sort("bucket")
    for i in range(1, c.height):
        prev = c.head(i).tail(12)
        exp = im.pin_mom(prev["B"].to_numpy(), prev["S"].to_numpy()) if prev["B"].sum() + prev["S"].sum() >= 10 else None
        got = roll["pin_prev"][i]
        assert (got is None and exp is None) or got == pytest.approx(min(max(exp, 0.0), 1.0), abs=1e-9)


# ---- trade-only tape maker ---------------------------------------------------------------------------
def test_print_maker_fills_only_through_and_respects_cooldown():
    o = _tape_orders()
    fills = im.print_maker(o, min_spread=0.02, latency_s=1.0, cooldown_s=60)
    assert fills.height > 0
    us, buy, lo, hi = (o[c].to_list() for c in ("us", "buy", "p_lo", "p_hi"))
    for r in fills.iter_rows(named=True):
        i = us.index(r["us"])
        if r["side"] == "yes":            # our bid = last sell + 1c, hit by a sell printing below it
            assert not buy[i] and lo[i] < r["price"] - 1e-9 and r["price"] == pytest.approx(r["bid0"] + 0.01)
        else:                             # our ask = last buy - 1c, lifted by a buy printing above it
            assert buy[i] and hi[i] > 1 - r["price"] + 1e-9 and 1 - r["price"] == pytest.approx(r["ask0"] - 0.01)
    t = fills["us"].to_list()
    assert all(b - a >= 60_000_000 for a, b in zip(t, t[1:]))
    assert im.print_maker(o, min_spread=0.02, cooldown_s=60).height >= im.print_maker(o, min_spread=0.05, cooldown_s=60).height


def test_hist_markouts_and_attach_on_recorded_tape():
    o = _tape_orders()
    st = im.tape_state(o)
    fills = im.print_maker(o, min_spread=0.02, latency_s=1.0, cooldown_s=60)
    mk = pl.DataFrame({"ticker": ["KXHIGHLAX-26NOV05-B79.5"], "event_ticker": ["KXHIGHLAX-26NOV05"], "result": ["no"],
                       "fee_type": ["quadratic"], "fee_multiplier": [1.0]})
    m = im.hist_markouts(fills, st, mk)
    assert m.height == fills.height and m["mx_60s"].is_not_null().all()
    for r in m.iter_rows(named=True):
        assert r["settle"] == pytest.approx((1.0 if r["side"] == "no" else 0.0) - r["price"])
    pins = pl.DataFrame({"unit": ["ticker_day"], "ticker": ["x"], "day": [o["t"][0].date()], "pin": [0.3]})
    prior = im.hist_alpha_prior(o, pins)
    vol = pl.DataFrame({"ticker": ["x"], "bucket": [0], "volatile": [True], "in_play": [False]})
    a = im.hist_attach(m, prior, vol, pl.DataFrame({"series": ["KXHIGHLAX"], "pin_series_dev": [0.25]}))
    assert a.height == m.height and set(a["window"].unique()) == {"holdout"}
    # half-spread captured = half the print spread proxy minus our 1c improvement
    exp_h = 100 * ((a["ask0"] - a["bid0"]) / 2 - 0.01)
    assert (a["h_c"] - exp_h).abs().max() < 1e-6 and a["xb_series"].is_not_null().all()
    t = im.split_table(a, ["mx_60s", "settle"])
    assert t.filter((pl.col("window") == "holdout") & (pl.col("regime") == "all"))["fills"].item() == a.height


def test_pick_rules_chooses_on_dev_and_reports_holdout():
    o = _tape_orders()
    fills = im.print_maker(o, min_spread=0.02, latency_s=1.0, cooldown_s=60)
    mk = pl.DataFrame({"ticker": ["KXHIGHLAX-26NOV05-B79.5"], "event_ticker": ["KXHIGHLAX-26NOV05"], "result": ["no"],
                       "fee_type": ["quadratic"], "fee_multiplier": [1.0]})
    m = im.hist_markouts(fills, im.tape_state(o), mk).with_columns(
        pl.col("ticker").alias("event"), pl.lit(False).alias("volatile"),
        pl.when(pl.int_range(pl.len()) < pl.len() // 2).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("window"))
    fam = {"side": {"yes": pl.col("side") == "yes", "no": pl.col("side") == "no"}}
    r = im.pick_rules(m, fam, "mx_60s", min_fills=1)
    assert r.height == 1
    row = r.row(0, named=True)
    dev = m.filter(pl.col("window") == "dev")
    best = max(("yes", "no"), key=lambda sd: dev.filter(pl.col("side") == sd)["mx_60s"].mean() if dev.filter(pl.col("side") == sd).height else -9)
    assert row["chosen_on_dev"] == best
    hold = m.filter((pl.col("window") == "holdout") & (pl.col("side") == best))
    assert row["holdout_fills"] == hold.height


def test_adverse_tables_only_use_executions_whose_horizon_has_ended():
    b = im.BUCKET_S * 10**9
    # one execution per bucket in buckets 0..5; adverse move = bucket index (cents)
    rows = [{"market": "KXA-1-X", "t_ns": k * b + 10, "side": "yes", "price": 0.40, "bid1": 0.40, "ask1": 0.42,
             "mo_300s": 0.01 - k / 100} for k in range(6)]
    ex = pl.DataFrame(rows)
    roll, static = im.adverse_tables(ex, split_ns=3 * b, n_prev=12)
    # h = 1c, adverse = h - 100*mo = k; rolling value usable at bucket j uses executions in buckets <= j-2
    got = dict(roll.select("bucket", "adverse_prev_c").iter_rows())
    assert 0 not in got and 1 not in got
    # needs >= 5 executions: first available at bucket 6 (executions 0..4), mean 2
    assert got == {6: pytest.approx(2.0), 7: pytest.approx(2.5)}
    # dev static: executions whose 300 s horizon ends before the split (buckets 0, 1), fewer than
    # 10 -> no market/series value, pooled mean over them
    assert static["adverse_dev_c"].item() == pytest.approx(0.5)
