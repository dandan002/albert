import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import mentions as M

FIX = Path(__file__).parent / "fixtures"
EVENT = FIX / "kalshi" / "mention_event_KXTRUMPMENTION-26OCT05.json"


def _markets() -> pl.DataFrame:
    """The recorded event's legs in the kalshi_markets table layout."""
    ev = json.loads(EVENT.read_text())["event"]
    rows = []
    for m in ev["markets"]:
        rows.append({"ticker": m["ticker"], "event_ticker": m["event_ticker"], "series_ticker": ev["series_ticker"],
                     "title": m["title"], "yes_sub_title": m["yes_sub_title"], "result": m["result"],
                     "open_time": datetime.fromisoformat(m["open_time"]), "close_time": datetime.fromisoformat(m["close_time"]),
                     "settlement_ts": datetime.fromisoformat(m["settlement_ts"]), "volume": float(m["volume_fp"])})
    return pl.DataFrame(rows)


def test_parse_word_threshold_and_alternatives_from_real_legs():
    lg = M.legs(_markets())
    assert lg.height == 5                      # "Event does not qualify" is dropped
    by = {r["word"]: r for r in lg.iter_rows(named=True)}
    assert by["iran"]["k"] == 5 and by["iran"]["n_alts"] == 1
    assert by["farm / farmer"]["n_alts"] == 2 and by["farm / farmer"]["k"] == 1
    assert by["bitcoin / crypto"]["y"] == 0 and by["transgender"]["y"] == 1
    assert set(lg["speaker"]) == {"trump"} and set(lg["etype"]) == {"rally"}


def test_milestone_gives_the_scheduled_start_before_any_close():
    ms = json.loads((FIX / "kalshi" / "mention_milestones_KXTRUMPMENTION-26OCT05.json").read_text())
    rows = M.milestone_rows(ms)
    assert rows and rows[0]["event"] == "KXTRUMPMENTION-26OCT05"
    start = rows[0]["start"]
    lg = M.legs(_markets())
    assert start < lg["close_time"].min()      # every leg closes after the speech starts
    d = M.decision_frame(lg, pl.DataFrame(rows), datetime(2026, 11, 1, tzinfo=UTC))
    # listed ~21 h before the 23:00 UTC start: only the 1-hour decision exists (no leg at t − 24 h)
    assert d.height == lg.height and set(d["when"]) == {"h1"}
    assert set((d["start"] - d["t"]).dt.total_hours()) == {1}
    assert (d["open_time"] <= d["t"]).all()


def test_factbase_search_and_transcript_parsers_on_recorded_pages():
    srch = json.loads((FIX / "factbase" / "search_page1_trim.json").read_text())
    ix = M.index_rows(srch["data"])
    assert len(ix) == 4 and all(r["url"].startswith(M.FACTBASE) for r in ix)
    assert any(r["duration_s"] and r["duration_s"] > 600 for r in ix)
    html = (FIX / "factbase" / "transcript_dt47-26-ef2b7858_trim.html").read_text()
    segs = M.parse_transcript(html)
    assert len(segs) == 8
    assert segs[0]["speaker"] == "Donald Trump" and segs[0]["t0"] == 0 and segs[0]["t1"] == 7
    assert segs[1]["speaker"] == "Unidentified"
    trump = " ".join(s["text"] for s in segs if s["speaker"] == "Donald Trump")
    # "room" must not match inside "ballroom"; both are counted as whole words
    assert len(M.word_pattern(("ballroom",)).findall(trump)) == trump.lower().count("ballroom") == 6
    assert len(M.word_pattern(("room",)).findall(trump)) == trump.lower().count(" room") > 0


def test_word_pattern_counts_plural_and_possessive_but_not_inflections():
    pat = M.word_pattern(("farm", "farmer"))
    assert len(pat.findall("Farmers and farms; the farmer's land. Farming is not counted.")) == 3
    assert len(M.word_pattern(("made in america",)).findall("made-in-America and Made in America's")) == 2
    assert len(M.word_pattern(("ice",)).findall("ICE, policing, juice")) == 1


def test_asof_counts_never_use_legs_settled_after_t():
    lg = M.legs(_markets())
    q = lg.head(1).with_columns(pl.lit(lg["settlement_ts"].min() - timedelta(seconds=1)).alias("t"))
    c = M.asof_counts(lg, q, (0.0,))
    assert c["n_g_0"][0] == 0                  # nothing had settled yet
    q2 = q.with_columns(pl.lit(lg["settlement_ts"].max() + timedelta(seconds=1)).alias("t"))
    c2 = M.asof_counts(lg, q2, (0.0,))
    assert c2["n_g_0"][0] == lg.height and c2["h_g_0"][0] == lg["y"].sum()
    assert c2["n_sw_0"][0] == 1 and c2["n_x_0"][0] == 0     # its own word, no other speaker
    p = M.bb_predict(c2, M.BB())
    assert 0 < p[0] < 1


def test_nb_tail_matches_poisson_limit_and_is_monotone():
    mu = np.array([0.5, 2.0, 6.0])
    k = np.array([1, 3, 5])
    pois = M.nb_tail(mu, float("inf"), k)
    nb_big = M.nb_tail(mu, 1e6, k)
    assert np.allclose(pois, nb_big, atol=1e-4)
    assert np.isclose(pois[0], 1 - np.exp(-0.5))
    assert (M.nb_tail(mu, 1.0, k) < 1).all() and M.nb_tail(np.array([3.0]), 1.0, np.array([1]))[0] > M.nb_tail(
        np.array([3.0]), 1.0, np.array([4]))[0]


def test_latent_event_factor_is_recovered_and_portfolio_respects_caps():
    # pure math: events with a shared logit shift of sd 1.0
    rng = np.random.default_rng(1)
    ev, p, y = [], [], []
    for e in range(300):
        z = rng.normal(0, 1.0)
        base = rng.uniform(0.2, 0.8, 10)
        lp = np.log(base / (1 - base)) + z
        y.extend((rng.uniform(size=10) < 1 / (1 + np.exp(-lp))).astype(float))
        p.extend(base)
        ev.extend([e] * 10)
    sig, grid = M.fit_sigma(np.array(p), np.array(y), np.array(ev))
    assert sig in (0.75, 1.0, 1.5) and dict(grid)[sig] > dict(grid)[0.0]
    # one leg the model likes on the YES side, one on the NO side, one fairly priced
    pm = np.array([0.6, 0.1, 0.5])
    bid, ask = np.array([0.38, 0.30, 0.48]), np.array([0.40, 0.32, 0.52])
    n, side = M.portfolio(pm, bid, ask, 0.5, wealth=100.0, cap=0.10)
    assert side[0] == 1 and side[1] == -1 and n[0] > 0 and n[1] > 0 and n[2] == 0
    cost = np.where(side == 1, ask, 1 - bid) * n
    assert cost.sum() <= 10.0 + 1e-9
    ni, _ = M.portfolio(pm, bid, ask, 0.5, wealth=100.0, cap=0.10, independent=True)
    assert ni.sum() > 0


def test_fees_exact_and_one_cent_floor():
    p = np.array([0.95, 0.5])
    assert np.allclose(M.fee_exact(p), [0.0034, 0.0175])
    assert np.allclose(M.fee_cent(p), [0.01, 0.02])                 # 1-contract order rounds up
    assert np.allclose(M.fee_cent(p, contracts=10), [0.004, 0.018])


def test_nb_mean_from_p1_inverts_the_one_plus_probability():
    p1 = np.array([0.1, 0.5, 0.9])
    for r in (0.5, 1.0, 5.0, float("inf")):
        mu = M.nb_mean_from_p1(p1, r)
        assert np.allclose(M.nb_tail(mu, r, np.ones(3, dtype=int)), p1, atol=1e-6)
