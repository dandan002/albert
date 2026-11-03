"""Parsers against recorded real responses (tests/fixtures, written by `albert capture`)."""

from pathlib import Path

from albert.venues import kalshi as k
from albert.venues import polymarket as pm
from albert.venues import polymarket_us as pmus

RAW = Path("raw.json")


def series_info(fixture):
    return {r["series_ticker"]: r for r in k.series_frame(fixture("kalshi/series.json")).iter_rows(named=True)}


def test_kalshi_series_have_categories(fixture):
    info = series_info(fixture)
    assert info["KXNBAGAME"]["category"] == "Sports"
    assert info["KXBTC15M"]["category"] == "Crypto"
    assert info["KXNBAGAME"]["fee_type"].startswith("quadratic")


def test_kalshi_settled_markets_parse_with_results_and_prices(fixture):
    rows = k.market_rows(fixture("kalshi/settled_markets.json"), series_info(fixture), "live")
    assert rows, "fixture produced no rows"
    for r in rows:
        assert r["series_ticker"] == "KXNBAGAME"
        assert r["category"] == "Sports"
        assert r["result"] in ("yes", "no")
        assert r["close_time"] is not None and r["close_time"].tzinfo is not None
        assert r["volume"] is not None
    frame = k.markets_frame(rows)
    assert frame.height == len(rows)


def test_kalshi_financial_markets_are_filtered(fixture):
    rows = k.market_rows(fixture("kalshi/settled_markets.json"), series_info(fixture), "live")
    for r in rows:
        r["category"] = "Crypto"
    assert k.markets_frame(rows).height == 0


def test_kalshi_trades_read_fixed_point_fields(fixture):
    rows = k.trade_rows(fixture("kalshi/trades.json"))
    assert rows
    for r in rows:
        assert 0 < r["yes_price"] < 1
        assert r["count"] > 0
        assert r["taker_side"] in ("yes", "no")


def test_kalshi_candles_live_and_historical_spellings(fixture):
    live = k.candle_rows(fixture("kalshi/candles_live.json"), "T", 60)
    hist = k.candle_rows(fixture("kalshi/candles_historical.json"), "T", 1440)
    assert live and hist
    for rows in (live, hist):
        assert all(r["end_ts"] is not None for r in rows)
        assert any(r["yes_ask_close"] is not None and r["yes_bid_close"] is not None for r in rows)


def test_kalshi_orderbook_implied_ask(fixture):
    payload = fixture("kalshi/orderbook.json")
    top = k.orderbook_top(payload)
    book = payload["orderbook_fp"]
    if book["no_dollars"]:
        best_no = max(float(p) for p, _ in book["no_dollars"])
        assert top["yes_ask"] == round(1 - best_no, 4)
    if book["yes_dollars"]:
        assert top["yes_bid"] == max(float(p) for p, _ in book["yes_dollars"])


def test_kalshi_cutoff_parses(fixture):
    cutoff = fixture("kalshi/cutoff.json")
    assert k.parse_ts(cutoff["market_settled_ts"]).year >= 2026


def test_polymarket_closed_events_have_winners(fixture):
    rows = pm.market_rows(fixture("polymarket/events_closed.json")["events"])
    assert rows
    for r in rows:
        assert r["closed"] is True
        assert len(r["token_ids"]) == len(r["outcomes"]) == 2
        assert "nba" in r["tags"]
    assert any(r["winner_index"] in (0, 1) for r in rows)


def test_polymarket_trades_and_leaderboard(fixture):
    trades = pm.trade_rows(fixture("polymarket/trades.json"))
    assert trades
    for t in trades:
        assert t["wallet"].startswith("0x") and t["wallet"] == t["wallet"].lower()
        assert t["side"] in ("BUY", "SELL")
        assert 0 < t["price"] < 1 and t["size"] > 0 and t["ts"] is not None
    board = pm.leaderboard_rows(fixture("polymarket/leaderboard.json"), "all", "overall")
    assert board and board[0]["rank"] == 1 and board[0]["pnl"] is not None


def test_polymarket_price_history(fixture):
    rows = pm.price_history_rows(fixture("polymarket/price_history.json"), "tok")
    assert rows and all(0 <= r["price"] <= 1 for r in rows)


def test_winner_index_rules():
    assert pm.winner_index(["1", "0"], True) == 0
    assert pm.winner_index(["0", "1"], True) == 1
    assert pm.winner_index(["0.5", "0.5"], True) is None
    assert pm.winner_index(["1", "0"], False) is None


def test_pmus_closed_markets_settlement_from_outcome_prices(fixture):
    payload = fixture("pmus/markets_closed.json")
    rows = pmus.market_rows(payload)
    assert rows
    for raw, row in zip(payload["markets"], rows):
        if raw["status"] == "MARKET_STATUS_RESOLVED":
            assert row["settlement"] == float(pm._json_list(raw["outcomePrices"])[0])
        else:
            assert row["settlement"] is None


def test_pmus_open_markets_book_and_bbo(fixture):
    rows = pmus.market_rows(fixture("pmus/markets_open.json"))
    assert rows and all(r["long_side"] for r in rows)
    book = pmus.book_top(fixture("pmus/book.json"))
    bbo = pmus.bbo_top(fixture("pmus/bbo.json"))
    if book["yes_bid"] is not None and book["yes_ask"] is not None:
        assert book["yes_bid"] < book["yes_ask"]
    assert bbo["state"]


def test_pmus_event_slug_map(fixture):
    mapping = pmus.event_slug_map(fixture("pmus/events.json"))
    assert mapping and all(isinstance(v, str) for v in mapping.values())


def test_kalshi_event_row_reads_exclusivity_and_legs(fixture):
    row = k.event_row(fixture("kalshi/event.json"))
    assert row["event_ticker"].startswith("KXNBAGAME-")
    assert row["mutually_exclusive"] is True
    assert row["n_markets"] >= 2


def test_kalshi_markets_keep_scheduled_expiration(fixture):
    series = k.series_frame(fixture("kalshi/series.json"))
    info = {r["series_ticker"]: r for r in series.iter_rows(named=True)}
    rows = k.market_rows(fixture("kalshi/settled_markets.json"), info, "live")
    assert all(r["expected_expiration_time"] is not None for r in rows)
    assert any(r["can_close_early"] for r in rows)


def test_ticker_start_is_eastern_time():
    from datetime import UTC, datetime

    assert k.ticker_start("KXKBOGAME-26NOV040430SAMKTW") == datetime(2026, 11, 4, 9, 30, tzinfo=UTC)
    # Daylight time in October: 20:00 EDT = 00:00 UTC next day.
    assert k.ticker_start("KXMLBGAME-26OCT152000MILLAD") == datetime(2026, 10, 16, 0, 0, tzinfo=UTC)
    assert k.ticker_start("KXNBAGAME-26NOV06SASSAC") is None
    assert k.ticker_start("KXHIGHNY-26NOV04") is None
