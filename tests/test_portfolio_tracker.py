import asyncio
import pytest
from datetime import datetime, date, timezone
from albert.db import get_connection, migrate
from albert.events import EventBus, FillEvent, MarketDataEvent
from albert.portfolio.tracker import PortfolioTracker


def make_fill(contracts: float = 5.0, price: float = 0.40, side: str = "yes") -> FillEvent:
    return FillEvent(
        fill_id="f1",
        market_id="kalshi:X",
        strategy_id="s1",
        side=side,
        contracts=contracts,
        fill_price=price,
        fee=0.0,
        filled_at=datetime.now(timezone.utc),
    )


def make_market_data(yes_bid: float = 0.45) -> MarketDataEvent:
    return MarketDataEvent(
        market_id="kalshi:X",
        exchange="kalshi",
        timestamp=datetime.now(timezone.utc),
        yes_bid=yes_bid,
        yes_ask=yes_bid + 0.02,
        no_bid=0.0,
        no_ask=0.0,
        last_price=yes_bid,
        volume=0.0,
    )


async def run_tracker_briefly(bus: EventBus, conn) -> None:
    tracker = PortfolioTracker(bus, conn)
    task = asyncio.create_task(tracker.run())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_fill_creates_position():
    conn = get_connection(":memory:")
    migrate(conn)
    bus = EventBus()
    tracker = PortfolioTracker(bus, conn)
    task = asyncio.create_task(tracker.run())
    await asyncio.sleep(0.01)
    await bus.publish("fills", make_fill(contracts=5.0, price=0.40))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    row = conn.execute("SELECT * FROM positions WHERE market_id = 'kalshi:X'").fetchone()
    assert row is not None
    assert row["contracts"] == 5.0
    assert row["avg_entry_price"] == pytest.approx(0.40)


async def test_market_data_updates_unrealized_pnl():
    conn = get_connection(":memory:")
    migrate(conn)
    bus = EventBus()
    tracker = PortfolioTracker(bus, conn)
    task = asyncio.create_task(tracker.run())
    await asyncio.sleep(0.01)
    await bus.publish("fills", make_fill(contracts=5.0, price=0.40))
    await bus.publish("market_data", make_market_data(yes_bid=0.50))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    row = conn.execute("SELECT unrealized_pnl FROM positions WHERE market_id = 'kalshi:X'").fetchone()
    # (0.50 - 0.40) * 5 = 0.50
    assert row["unrealized_pnl"] == pytest.approx(0.50)


async def test_closing_fill_records_realized_pnl():
    conn = get_connection(":memory:")
    migrate(conn)
    bus = EventBus()
    tracker = PortfolioTracker(bus, conn)
    task = asyncio.create_task(tracker.run())
    await asyncio.sleep(0.01)

    # Open 5 contracts at 0.40
    await bus.publish("fills", make_fill(contracts=5.0, price=0.40))
    await asyncio.sleep(0.05)

    # Close 5 contracts at 0.50 (closing fill has negative contracts)
    close_fill = FillEvent(
        fill_id="f2",
        market_id="kalshi:X",
        strategy_id="s1",
        side="yes",
        contracts=-5.0,
        fill_price=0.50,
        fee=0.0,
        filled_at=datetime.now(timezone.utc),
    )
    await bus.publish("fills", close_fill)
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Position should be gone
    row = conn.execute("SELECT * FROM positions WHERE market_id = 'kalshi:X'").fetchone()
    assert row is None

    # Realized PnL recorded: (0.50 - 0.40) * 5 = 0.50
    today = date.today().isoformat()
    pnl_row = conn.execute(
        "SELECT realized_pnl FROM daily_pnl WHERE date = ? AND strategy_id = 's1'", (today,)
    ).fetchone()
    assert pnl_row is not None
    assert pnl_row["realized_pnl"] == pytest.approx(0.50)


async def test_partial_close_records_realized_pnl():
    conn = get_connection(":memory:")
    migrate(conn)
    bus = EventBus()
    tracker = PortfolioTracker(bus, conn)
    task = asyncio.create_task(tracker.run())
    await asyncio.sleep(0.01)

    # Open 10 contracts at 0.40
    await bus.publish("fills", make_fill(contracts=10.0, price=0.40))
    await asyncio.sleep(0.05)

    # Partial close: sell 3 contracts at 0.55
    partial_close = FillEvent(
        fill_id="f3",
        market_id="kalshi:X",
        strategy_id="s1",
        side="yes",
        contracts=-3.0,
        fill_price=0.55,
        fee=0.0,
        filled_at=datetime.now(timezone.utc),
    )
    await bus.publish("fills", partial_close)
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # 7 contracts should remain
    row = conn.execute("SELECT contracts FROM positions WHERE market_id = 'kalshi:X'").fetchone()
    assert row is not None
    assert row["contracts"] == pytest.approx(7.0)

    # Realized PnL: (0.55 - 0.40) * 3 = 0.45
    today = date.today().isoformat()
    pnl_row = conn.execute(
        "SELECT realized_pnl FROM daily_pnl WHERE date = ? AND strategy_id = 's1'", (today,)
    ).fetchone()
    assert pnl_row is not None
    assert pnl_row["realized_pnl"] == pytest.approx(0.45)
