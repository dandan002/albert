import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import mentions as M
from albert.research import mentions_maker as MM
from albert.research.fundamental import maker_fills
from albert.venues import kalshi as k

FIX = Path(__file__).parent / "fixtures"
EVENT = FIX / "kalshi" / "mention_event_KXTRUMPMENTION-26OCT05.json"
MILESTONES = FIX / "kalshi" / "mention_milestones_KXTRUMPMENTION-26OCT05.json"


def _start() -> datetime:
    return M.milestone_rows(json.loads(MILESTONES.read_text()))[0]["start"]


def _trades() -> pl.DataFrame:
    """Trimmed real /markets/trades pages (last 120 prints before the start) of two legs."""
    rows = []
    for leg in ("CORN", "TRAN"):
        page = json.loads((FIX / "kalshi" / f"mentions_maker_trades_KXTRUMPMENTION-26OCT05-{leg}.json").read_text())
        rows.extend(k.trade_rows(page))
    return pl.DataFrame(rows)


def _results() -> dict[str, int]:
    ev = json.loads(EVENT.read_text())["event"]
    return {m["ticker"]: int(m["result"] == "yes") for m in ev["markets"]}


def test_speech_type_on_recorded_titles():
    ms_title = json.loads(MILESTONES.read_text())["milestones"][0]["title"]
    ktitle = json.loads(EVENT.read_text())["event"]["markets"][0]["title"]
    assert MM.speech_type(ktitle, ms_title) == "rally"
    # Kalshi event titles seen in the settled Mentions table
    assert MM.speech_type("What will Donald Trump say during Tele-Rally with Darline Graham?", None) == "telerally"
    assert MM.speech_type("What will Donald Trump say during his GB News interview originally scheduled", None) == "interview"
    assert MM.speech_type("What will JD Vance say during press conference originally scheduled for", None) == "press"
    assert MM.speech_type("What will Donald Trump say during RNC Midterm Convention?", None) == "address"
    assert MM.speech_type("What will Donald Trump say during Remarks at Steel Across America?", None) == "remarks"
    assert MM.fb_type2("Speech", "Speech: Donald Trump Holds a Midterm Political Rally in Chillicothe, Ohio") == "rally"
    assert MM.fb_type2("Speech", "Speech: Donald Trump Tours a Peterbilt Plant in Denton, Texas") == "address"
    assert MM.fb_type2("Remarks", "Remarks: Donald Trump Holds a Virtual Tele-Rally for Kurt Alme") == "telerally"


def test_transcript_cutoff_respects_the_publication_lag():
    start = _start()                                   # 2026-10-05 23:00 UTC = 7 pm ET
    assert start == datetime(2026, 10, 5, 23, tzinfo=UTC)
    assert MM.tx_cutoff(start - timedelta(hours=1)) == date(2026, 10, 4)
    assert MM.tx_cutoff(start - timedelta(hours=6)) == date(2026, 10, 4)
    assert MM.tx_cutoff(start - timedelta(hours=24)) == date(2026, 10, 3)
    # a 10 am ET event: one hour before, yesterday's evening transcript is not yet assumed public
    assert MM.tx_cutoff(datetime(2026, 10, 5, 13, tzinfo=UTC)) == date(2026, 10, 3)


def test_salience_compares_last_week_with_the_45_day_rate():
    html = (FIX / "factbase" / "transcript_dt47-26-ef2b7858_trim.html").read_text()
    text = " ".join(s["text"] for s in M.parse_transcript(html) if s["speaker"] == "Donald Trump")
    assert "ballroom" in text.lower()
    hi = date(2026, 10, 4)
    docs = pl.DataFrame({"document_id": ["recent", "old"], "who": ["trump", "trump"], "text": [text, text],
                         "words": [len(text.split())] * 2, "date": [hi - timedelta(days=2), hi - timedelta(days=30)]})
    counts = M.doc_counts(docs, {"ballroom": ("ballroom",), "tariff": ("tariff",)})
    # the same word only in the recent document vs only in the old one
    recent = counts.with_columns(pl.when(pl.col("document_id") == "old").then(0).otherwise(pl.col("c")).alias("c"))
    old = counts.with_columns(pl.when(pl.col("document_id") == "recent").then(0).otherwise(pl.col("c")).alias("c"))
    q = pl.DataFrame({"speaker": ["trump"], "word": ["ballroom"], "tx_hi": [hi]})
    a = MM.salience(q, docs, recent)
    b = MM.salience(q, docs, old)
    assert a["c7"][0] == 6 and a["c45"][0] == 6 and b["c7"][0] == 0 and b["c45"][0] == 6
    assert a["sal"][0] > 0 > b["sal"][0]
    # documents dated after the cutoff are never used
    late = docs.with_columns(pl.lit(hi + timedelta(days=1)).alias("date"))
    assert MM.salience(q, late, recent)["c45"][0] == 0


def _pred(trades: pl.DataFrame) -> pl.DataFrame:
    start, res = _start(), _results()
    rows = []
    for tk in trades["ticker"].unique().sort():
        for h in (2, 1):
            for p in (0.55, 0.66, 0.70, 0.74, 0.80, 0.90):
                for bid, ask in ((0.64, 0.70), (0.70, 0.74), (0.85, 0.90)):
                    rows.append({"ticker": tk, "event": "KXTRUMPMENTION-26OCT05", "t": start - timedelta(hours=h),
                                 "end": start - MM.END_MARGIN, "p": p, "bid": bid, "ask": ask, "y": res[tk],
                                 "fee_type": "quadratic", "fee_multiplier": 1.0})
    return pl.DataFrame(rows)


def test_fast_fills_equal_the_harness_rule_on_real_prints():
    trades = _trades()
    assert trades.height == 240
    pred = _pred(trades)
    for margin in (0.0, 0.02):
        slow = maker_fills(pred, trades, margin)
        fast = MM.fast_fills(pred, trades, margin)
        assert slow.height == fast.height > 0
        assert slow["side"].to_list() == fast["side"].to_list()
        assert slow["filled"].to_list() == fast["filled"].to_list()
        assert np.allclose(slow["price"].to_numpy(), fast["price"].to_numpy())
        assert np.allclose(slow["pnl"].to_numpy(), fast["pnl"].to_numpy())
        assert 0 < fast["filled"].sum() < fast.height          # some bids fill, some do not
    # no maker fee on a quadratic (taker-only) series: a filled YES bid on CORN (settled NO) loses its price
    f = MM.fast_fills(pred, trades, 0.0).join(pred.with_row_index("row").select("row", "ticker", "y"), on="row")
    lost = f.filter(pl.col("filled") & (pl.col("side") == "yes") & (pl.col("y") == 0))
    assert lost.height and np.allclose(lost["pnl"].to_numpy(), -lost["price"].to_numpy())


def test_fills_only_after_t_and_before_end():
    trades = _trades()
    pred = _pred(trades)
    # a window that ends before it starts can never fill
    shut = MM.fast_fills(pred.with_columns((pl.col("t") - timedelta(minutes=1)).alias("end")), trades, 0.0)
    assert shut.height and not shut["filled"].any()
    f = MM.fast_fills(pred, trades, 0.0).filter("filled").join(pred.with_row_index("row").select("row", "t", "end"), on="row")
    ts = pl.from_epoch(f["fill_us"], time_unit="us").dt.replace_time_zone("UTC")
    assert f.height and ((ts > f["t"]) & (ts <= f["end"])).all()


def test_side_variants_keep_the_favoured_side_only():
    trades = _trades()
    pred = _pred(trades).with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    f = MM.fast_fills(pred, trades, 0.0).join(pred.with_row_index("row").select("row", "p", "mid"), on="row")
    f = f.with_columns((pl.col("p") - pl.col("mid")).alias("gap"))
    fav = MM.side_variant(f, "fav")
    assert fav.height and fav.group_by("row").len()["len"].max() == 1
    assert (fav.filter(pl.col("side") == "yes")["gap"] > 0).all() and (fav.filter(pl.col("side") == "no")["gap"] < 0).all()
    g = MM.side_variant(f, "gap0.05")
    assert g.height < fav.height and (g["gap"].abs() >= 0.05 - 1e-9).all()


def test_circuit_breaker_cancels_at_the_first_said_print():
    trades = _trades()
    start = _start()
    t = start - timedelta(hours=6)
    hi = trades.filter(pl.col("ticker").str.ends_with("TRAN") & (pl.col("yes_price") >= MM.CB_PRINT))
    assert hi.height                                   # TRAN printed 95c+ before the scheduled start
    q = pl.DataFrame({"event": ["E", "E"], "when": ["h6", "h6"], "ticker": ["KXTRUMPMENTION-26OCT05-TRAN",
                      "KXTRUMPMENTION-26OCT05-CORN"], "t": [t, t], "ask": [0.78, 0.70]})
    kt = MM.kill_times(q, trades)
    assert kt.height == 1 and kt["kill_ts"][0] == hi["created_time"].min()
    # a leg already priced near certainty at t does not arm the breaker
    assert MM.kill_times(q.with_columns(pl.lit(0.9).alias("ask")), trades).height == 0


def test_improve_rule_counts_prints_at_our_price_only_when_we_improved_the_bid():
    trades = _trades().filter(pl.col("ticker").str.ends_with("CORN"))
    start = _start()
    # a real NO-taker print (YES seller) at some price after t
    t = start - timedelta(hours=2)
    sell = trades.filter((pl.col("taker_side") == "no") & (pl.col("created_time") > t)).sort("created_time")
    assert sell.height
    px = sell["yes_price"][0]
    lower = trades.filter((pl.col("taker_side") == "no") & (pl.col("created_time") > t) & (pl.col("yes_price") < px - 1e-9))
    assert lower.height == 0 or lower["created_time"].min() > sell["created_time"][0]
    base = {"ticker": "KXTRUMPMENTION-26OCT05-CORN", "event": "E", "t": t, "end": start - MM.END_MARGIN, "y": 0,
            "fee_type": "quadratic", "fee_multiplier": 1.0, "p": px + 0.005, "ask": px + 0.05}
    improving = pl.DataFrame([{**base, "bid": px - 0.02}])      # our YES bid at px is above the bid at t
    joining = pl.DataFrame([{**base, "bid": px}])               # someone already bid px at t
    yes = pl.col("side") == "yes"
    fi = MM.fast_fills(improving, trades, 0.0, rule="improve").filter(yes)
    fj = MM.fast_fills(joining, trades, 0.0, rule="improve").filter(yes)
    ft = MM.fast_fills(improving, trades, 0.0, rule="through").filter(yes)
    assert fi["price"][0] == fj["price"][0] == round(px, 2)
    assert fi["filled"][0] and fi["fill_us"][0] == sell["created_time"].dt.epoch("us")[0]
    assert (not fj["filled"][0]) or fj["fill_us"][0] > fi["fill_us"][0]
    assert (not ft["filled"][0]) or ft["fill_us"][0] > fi["fill_us"][0]


def test_margin_is_chosen_on_dev_only_and_reported_for_both_splits():
    trades = _trades()
    pred = _pred(trades).with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), pl.lit("S").alias("series"),
                                    pl.lit("h1").alias("when"))
    # the same real rows twice: once labelled dev, once holdout with every outcome flipped
    dev = pred.with_columns(pl.lit("dev").alias("split"))
    ho = pred.with_columns(pl.lit("holdout").alias("split"), (1 - pl.col("y")).alias("y"),
                           (pl.col("t") + timedelta(seconds=0)).alias("t"))
    both = pl.concat([dev, ho]).with_columns(pl.col("p").alias("fv"))
    G = MM.maker_grid(both, trades, "fv", margins=(0.0, 0.02, 0.05), variants=("both",))
    assert G.height and set(G["split"]) == {"dev", "holdout"}
    T, C = MM.maker_table(G, min_fills=1)
    assert T.height == 3 * 2 and C.height == 2
    chosen = C.filter(pl.col("split") == "dev")
    best_dev = T.filter(pl.col("split") == "dev").sort("pnl_per_fill_c", descending=True)["margin"][0]
    assert chosen["margin"][0] == best_dev == C.filter(pl.col("split") == "holdout")["margin"][0]
