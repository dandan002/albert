def kelly_size(
    edge: float,
    ask_price: float,
    bankroll: float,
    kelly_fraction: float,
    confidence: float,
    max_position_usd: float,
) -> float:
    """
    Compute fractional Kelly position size in USD.

    Args:
        edge: Estimated probability edge above ask price (0.0–1.0).
        ask_price: Current ask price for the side being bought (0–1 exclusive).
        bankroll: Total available capital in USD.
        kelly_fraction: Fraction of full Kelly to use (e.g. 0.25 for quarter-Kelly).
        confidence: Strategy confidence scalar (0.0–1.0).
        max_position_usd: Hard cap on position size regardless of Kelly output.

    Returns:
        Position size in USD, >= 0.
    """
    if ask_price <= 0.0 or ask_price >= 1.0:
        return 0.0
    if edge <= 0.0:
        return 0.0

    # Net odds on a $1 bet
    b = (1.0 - ask_price) / ask_price

    # Only trade if net odds are at least 1:1
    if b < 1.0:
        return 0.0

    # True probability is market price plus our edge
    p = ask_price + edge
    if p > 1.0:
        return 0.0

    # Kelly criterion: f* = (p*b - (1-p)) / b
    q = 1.0 - p
    f_star = (p * b - q) / b
    if f_star <= 0.0:
        return 0.0

    size = bankroll * f_star * kelly_fraction * confidence
    return min(size, max_position_usd)
