"""Non-financial filters, derived from category and tag values observed on each venue (2026-11)."""

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


def kalshi_is_non_financial(category: str | None) -> bool:
    return category is not None and category not in KALSHI_FINANCIAL_CATEGORIES


def pm_is_non_financial(tags: list[str] | None) -> bool:
    return not (set(tags or ()) & PM_FINANCIAL_TAGS)


def pmus_is_non_financial(category: str | None) -> bool:
    return category is not None and category.lower() not in PMUS_FINANCIAL_CATEGORIES
