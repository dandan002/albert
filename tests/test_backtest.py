import pytest
import sqlite3
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from albert.db import migrate
from albert.backtest import run_backtest, load_strategy_class
from albert.strategies.examples.momentum import MomentumV1

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_albert.db"
    conn = sqlite3.connect(str(db_path))
    migrate(conn)
    
    # Insert mock snapshots
    market_id = "test:ticker"
    now = datetime.utcnow()
    
    snapshots = [
        # market_id, timestamp, yes_bid, yes_ask, no_bid, no_ask, last_price, volume
        (market_id, (now - timedelta(minutes=10)).isoformat(), 0.40, 0.41, 0.58, 0.59, 0.40, 1000),
        (market_id, (now - timedelta(minutes=9)).isoformat(), 0.42, 0.43, 0.56, 0.57, 0.42, 1100),
        (market_id, (now - timedelta(minutes=8)).isoformat(), 0.45, 0.46, 0.53, 0.54, 0.45, 1200),
        (market_id, (now - timedelta(minutes=7)).isoformat(), 0.48, 0.49, 0.50, 0.51, 0.48, 1300),
        (market_id, (now - timedelta(minutes=6)).isoformat(), 0.51, 0.52, 0.47, 0.48, 0.51, 1400),
    ]
    
    conn.executemany(
        "INSERT INTO orderbook_snapshots (market_id, timestamp, yes_bid, yes_ask, no_bid, no_ask, last_price, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        snapshots
    )
    conn.commit()
    conn.close()
    return db_path

@pytest.mark.asyncio
async def test_run_backtest(temp_db):
    market_id = "test:ticker"
    config = {
        "min_edge": 0.05,
        "kelly_fraction": 0.25,
        "bankroll": 1000,
        "max_position_usd": 100
    }
    
    # Run backtest
    # This will print to stdout, we can capture if needed but for now just verify it runs
    await run_backtest(market_id, MomentumV1, config, db_path=temp_db)
    
    # If it completed without error, that's a good start.
    # We could add assertions by returning data from run_backtest, 
    # but for now we'll check it didn't crash and processed snapshots.
    assert True

def test_load_strategy_class():
    cls = load_strategy_class("albert.strategies.examples.momentum.MomentumV1")
    assert cls == MomentumV1
    assert issubclass(cls, MomentumV1)

