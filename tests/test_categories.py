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
