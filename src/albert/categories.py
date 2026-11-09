"""Non-financial filters, derived from category and tag values observed on each venue (2026-11)."""

import re

# Kalshi series categories (from /series). Everything else is treated as non-financial.
KALSHI_FINANCIAL_CATEGORIES = frozenset(
    {"Financials", "Economics", "Crypto", "Commodities", "Companies"}
)

# Polymarket intl event tag slugs that mark a market as financial.
PM_FINANCIAL_TAGS = frozenset(
    {
        "crypto", "crypto-prices", "bitcoin", "ethereum", "solana", "xrp", "altcoins", "stablecoins",
        "finance", "economy", "economics", "fed", "fed-rates", "fomc", "interest-rates", "inflation",
        "cpi", "gdp", "jobs", "stocks", "equities", "earnings", "ipos", "ipo", "commodities", "gold",
        "oil", "forex", "business", "macro-single", "macro-graph", "etf", "tariffs-economy",
    }
)

# Polymarket US market categories (from /v1/markets).
PMUS_FINANCIAL_CATEGORIES = frozenset({"finance", "crypto", "economics", "commodities"})


# Money-related series filed under other categories (Politics, World, Mentions, Social, ...):
# central banks and the Fed, Treasury, rates, economic releases, crypto, tariffs, public debt,
# securities and banking, company earnings and executives. Matched on the series title and
# ticker; reviewed against the full 2026-11 series list (tests/test_categories.py).
_FIN_TITLE = re.compile(
    r"\bfed\b|federal reserve|\bfomc\b|tapering|central bank|\becb\b|lagarde|\bimf\b|treasury|"
    r"bessent|(?<!glen )powell|warsh|crypto|bitcoin|\bbtc\b|ethereum|stablecoin|memecoin|blockchain|\bcbdc\b|"
    r"\bgdp\b|\bcpi\b|inflation|recession|unemployment|jobs report|payroll|interest rate|mortgage|"
    r"tariff|debt ceiling|national debt|deficit|\bstocks?\b|stockx|\bbonds\b|bond tax|"
    r"\bbank(ing)?\b|dollar|earnings|revenue|\bipo\b|market cap|share price|microstrategy|"
    r"berkshire|nvidia|tesla|\bjpm\b|investor|shareholder", re.IGNORECASE)
# Ticker fragments only where they cannot be part of another word (ECB, BTC, TSLA, FED and TREAS
# can: J Cole CBA, Best Traditional Country Album, state Treasurer races); titles catch the rest.
_FIN_TICKER = re.compile(r"EARNING|EARNIG|MENTIONEARN|FOMC|JPOW|WARSH|BESSENT|NVDA|BRKEM|CRYPTO|GDP|"
                         r"TARIFF|DOLLAR")

# Mentions series are named after a speaker or show: companies' calls and events, corporate and
# finance executives, Fed and Treasury officials, crypto and investment events (reviewed against
# the 2026-11 Mentions series list).
_FIN_MENTIONS = re.compile(
    r"ABNB|AIRBNB|ADBE|ADOBE|^KXBAMENTION|BOEING|^KXCATMENTION|CATERPILLAR|^KXCRMMENTION|SALESFORCE|"
    r"^KXFIGMENTION|FIGMA|GOOG|^KXMCDMENTION|MCDONALD|MRVL|MARVELL|MSFT|MICROSOFT|NBIS|NEBIUS|ORCL|"
    r"ORACLE|RDDT|REDDIT|^KXSNOWMENTION|SNOWFLAKE|^KXSPOTMENTION|SPOTIFY|UBER|^KXVZMENTION|VERIZON|"
    r"WDAY|WORKDAY|^KXWMTMENTION|WALMART|^KXZMMENTION|ZOOM|SPCX|SPACEX|APPLE|WWDC|NVDA|NVIDIA|TSLA|"
    r"TESLA|BRK|BERKSHIRE|JPM|ACKMAN|DIMON|FINK|GRIFFIN|ARMSTRONG|TENEV|BOWMAN|WALLER|MIRAN|LUTNICK|"
    r"HASSETT|ALTMAN|AMODEI|BENIOFF|NADELLA|ZUCK|JENSEN|KARP|ALTHOFF|LUCKEY|ELLISON|ELON|MUSK|"
    r"INVESTMENT|BUSINESS|DIGITAL ASSET|DJTDAS|FEDMENTION|BTCMENTION|^KXWEFMENTION|ECONOMIC FORUM|"
    r"THREADGUY|^KXBARRMENTION|TBPN",
    re.IGNORECASE)


def kalshi_is_non_financial(category: str | None) -> bool:
    return category is not None and category not in KALSHI_FINANCIAL_CATEGORIES


def kalshi_series_is_financial(series: str | None, category: str | None, title: str | None = None) -> bool:
    """True for a financial category, or a series whose title or ticker is about money (see above).
    An unknown category counts as financial."""
    if not kalshi_is_non_financial(category):
        return True
    if _FIN_TITLE.search(title or "") or _FIN_TICKER.search(series or ""):
        return True
    return category == "Mentions" and bool(_FIN_MENTIONS.search(series or "") or _FIN_MENTIONS.search(title or ""))


def kalshi_financial_series(series_info: dict[str, dict]) -> frozenset[str]:
    """Series tickers that ``kalshi_series_is_financial`` rejects, from ``kalshi_series`` rows."""
    return frozenset(s for s, info in series_info.items()
                     if kalshi_series_is_financial(s, info.get("category"), info.get("title")))


def pm_is_non_financial(tags: list[str] | None) -> bool:
    return not (set(tags or ()) & PM_FINANCIAL_TAGS)


def pmus_is_non_financial(category: str | None) -> bool:
    return category is not None and category.lower() not in PMUS_FINANCIAL_CATEGORIES
