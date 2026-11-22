import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from albert.research import culture_recurring as cr

FIX = Path(__file__).parent / "fixtures"


def test_openrouter_request_share_matches_kalshi_settlement():
    payload = json.loads((FIX / "openrouter" / "author_market_share_req_text_trimmed.json").read_text())
    rows = cr.openrouter_rows(payload, datetime(2026, 11, 20, tzinfo=UTC), "metric=requests&modality=text")
    df = pl.DataFrame(rows)
    assert df.height > 0 and set(df["week"].to_list()) == {date(2026, 11, 2), date(2026, 11, 9)}
    wk = df.filter(pl.col("week") == date(2026, 11, 9))
    share = 100 * wk.filter(pl.col("author") == "openai")["value"][0] / wk["value"].sum()
    mk = json.loads((FIX / "kalshi" / "culture_recurring_kxopenshare_markets_trimmed.json").read_text())["markets"]
    # Kalshi settled KXOPENSHARE-26NOV16 at 19.0 (above 18.9 YES, above 19.1 NO)
    assert {m["ticker"]: m["expiration_value"] for m in mk}["KXOPENSHARE-26NOV16-18.9"] == "19.0"
    assert round(share, 1) == 19.0
    for m in mk:
        assert (round(share, 1) > m["floor_strike"]) == (m["result"] == "yes")


def test_market_row_and_week_parse():
    mk = json.loads((FIX / "kalshi" / "culture_recurring_kxopenshare_markets_trimmed.json").read_text())["markets"]
    r = cr.market_row(mk[0], "KXOPENSHARE", "live")
    assert r["floor_strike"] == 18.9 and r["result"] == "yes" and r["close_time"] == datetime(2026, 11, 16, 4, 59, tzinfo=UTC)
    assert cr.or_week(r["rules_primary"], r["event_ticker"]) == date(2026, 11, 9)
    tok = "If the OpenRouter token usage for Nov 9–15, 2026 is above 207T tokens, then the market resolves to Yes."
    assert cr.or_week(tok, "KXTOKENUSE-26NOV16") == date(2026, 11, 9)
    assert cr.or_week(None, "KXOPENSHARE-26NOV16") == date(2026, 11, 9)


def test_lmarena_commit_titles_give_publication_times():
    commits = json.loads((FIX / "huggingface" / "lmarena_commits_trimmed.json").read_text())
    df = pl.DataFrame([{"commit_ts": datetime.fromisoformat(c["date"]), "title": c["title"]} for c in commits])
    d = df.with_columns(pl.col("title").str.extract(r"^Update text for (\d{4}-\d{2}-\d{2})", 1).alias("d")).drop_nulls("d")
    assert d.height == 2
    lag = (d["commit_ts"][0] - datetime.fromisoformat(d["d"][0] + "T00:00:00+00:00")).total_seconds() / 3600
    assert 24 <= lag <= 30          # snapshot D is committed ~04:05 UTC on D+1


def test_fee_rounds_up_per_order():
    assert cr.taker_fee(25, 0.5) == 0.44          # 0.4375 -> 0.44
    assert cr.taker_fee(1, 0.5) == 0.02           # 0.0175 -> 0.02
    assert cr.taker_fee(10, 0.9) == 0.07          # 0.063 -> 0.07


def test_pava_and_implied_median():
    assert np.allclose(cr.pava_decreasing([0.9, 0.6, 0.7, 0.2]), [0.9, 0.65, 0.65, 0.2])
    m = cr.implied_median([50, 60, 70, 80], [0.95, 0.7, 0.3, 0.05])
    assert 64 < m < 66


def test_take_pnl_yes_and_no():
    df = pl.DataFrame({"event_ticker": ["E", "E"], "ticker": ["A", "B"], "bid": [0.40, 0.40], "ask": [0.44, 0.44],
                       "spread": [0.04, 0.04], "y": [1, 0], "fee_multiplier": [1.0, 1.0]})
    ty = cr.take(df, "yes")
    assert ty["n"][0] == 6 and abs(ty["pnl_usd"][0] - (6 - 6 * 0.44 - cr.taker_fee(6, 0.44))) < 1e-9
    tn = cr.take(df, "no")
    assert tn["px"][1] == 0.6 and tn["win"][1] == 1.0


def test_stay_model_monotone_in_margin():
    rng = np.random.default_rng(0)
    margin = rng.uniform(0, 40, 4000)
    h = rng.integers(1, 30, 4000)
    stay = (rng.uniform(size=4000) < 1 / (1 + np.exp(-(0.5 + 0.15 * margin - 0.5 * np.log(h))))).astype(int)
    w = cr.fit_stay(pl.DataFrame({"margin": margin, "h": h, "stay": stay}))
    assert cr.p_stay(w, 30, 7) > cr.p_stay(w, 2, 7) and cr.p_stay(w, 10, 2) > cr.p_stay(w, 10, 30)


def test_leg_keys():
    assert cr.leg_key("KXLLM1", "Gemini") == "google"
    assert cr.leg_key("KXLLM1", "Amazon Nova") == "amazon"
    assert cr.leg_key("KXTOPMODEL", "gemini-4-argon-high") == cr.norm_model("Gemini 4 Argon (High)")
    assert cr.norm_org("Moonshot AI") == "moonshot" and cr.norm_org("xAI") == "xai"


def test_ladder_arbs_detects_crossed_strikes():
    p = pl.DataFrame({"event_ticker": ["E"] * 2, "label": ["T-24h"] * 2, "K": [60.0, 70.0],
                      "bid": [0.50, 0.70], "ask": [0.55, 0.75]})
    a = cr.ladder_arbs(p)
    assert a.height == 1 and a["k_lo"][0] == 60.0


def test_kworb_rows_parse_trimmed_page():
    html = (FIX / "kworb" / "youtube_daily_trimmed.html").read_text()
    rows = cr.kworb_rows(html, datetime(2026, 11, 20, 22, 23, tzinfo=UTC))
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert rows[1]["video_id"] == "fcnDmrtj6Sk" and rows[1]["views"] == 4765724


def test_or_week_ending_form():
    r = "If OpenAI scores above 16% on OpenRouter Market Share by model author week ending Aug 16, 2026, then"
    assert cr.or_week(r, "KXOPENSHARE-26AUG17") == date(2026, 8, 10)


def test_p_at_least_exact():
    assert abs(cr.p_at_least(3, [0.9, 0.9, 0.9]) - 0.729) < 1e-12
    assert abs(cr.p_at_least(0, [0.5, 0.5]) - 1.0) < 1e-12
    assert abs(cr.p_at_least(1, [0.5, 0.5]) - 0.75) < 1e-12
