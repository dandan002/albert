"""Finance filter on real Kalshi series (trimmed from GET /series): finance series hide in Politics,
World, Social, Entertainment and Mentions, and look-alikes (James Bond, Outer Banks, state
Treasurer races, Glen Powell, ticker fragments such as ECB in J Cole CBA) must not be caught."""

import json
from pathlib import Path

import pytest

from albert.categories import kalshi_financial_series, kalshi_series_is_financial

ROWS = json.loads((Path(__file__).parent / "fixtures" / "kalshi" / "series_finance_review.json").read_text())["series"]


@pytest.mark.parametrize("row", ROWS, ids=[r["series_ticker"] for r in ROWS])
def test_series_finance_label(row):
    assert kalshi_series_is_financial(row["series_ticker"], row["category"], row["title"]) is row["financial"]


def test_financial_categories_and_unknown_category_are_financial():
    assert kalshi_series_is_financial("KXHIGHNY", "Financials", "Highest temperature in NYC")
    assert kalshi_series_is_financial("KXHIGHNY", None, "Highest temperature in NYC")
    assert not kalshi_series_is_financial("KXHIGHNY", "Climate and Weather", "Highest temperature in NYC")


def test_financial_series_set_from_series_rows():
    info = {r["series_ticker"]: {"category": r["category"], "title": r["title"]} for r in ROWS}
    assert kalshi_financial_series(info) == {r["series_ticker"] for r in ROWS if r["financial"]}


def test_executives_are_financial_but_flagged_as_executives():
    from albert.categories import kalshi_mention_is_executive

    for s, title in [("KXELONMENTION", "Elon Musk Mention"), ("KXALTMANMENTION", "Sam Altman Mention"),
                     ("KXNADELLAMENTION", "Satya Nadella Mention"), ("KXDIMONMENTION", "Jamie Dimon Mention")]:
        assert kalshi_series_is_financial(s, "Mentions", title)
        assert kalshi_mention_is_executive(s, "Mentions", title)
    for s, title in [("KXWALLERMENTION", "Waller Mention"), ("KXMSFTMENTION", "Microsoft Mention"),
                     ("KXTSLAMENTION", "Tesla Mention"), ("KXEARNINGSMENTIONTGT", "Target earnings call"),
                     ("KXELONMENTION", "What will Elon say at the Tesla shareholder meeting?"),
                     ("KXTRUMPMENTION", "What will Trump say?"), ("KXELONMENTION", "Elon Musk Mention")]:
        cat = "Companies" if title == "Elon Musk Mention" else "Mentions"
        assert not kalshi_mention_is_executive(s, cat, title)
