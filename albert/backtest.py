import asyncio
import json
import logging
import sys
import importlib
from datetime import datetime
from typing import Any, Type
from pathlib import Path

from albert.db import get_connection
from albert.events import MarketDataEvent, OrderIntent
from albert.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

async def run_backtest(market_id: str, strategy_class: Type[BaseStrategy], config: dict, db_path: str | Path = "albert.db"):
    """Runs a strategy against historical orderbook snapshots."""
    strategy = strategy_class("backtest_strategy", config)
    conn = get_connection(db_path)
    
    # Query snapshots
    cursor = conn.execute(
        "SELECT * FROM orderbook_snapshots WHERE market_id = ? ORDER BY timestamp ASC",
        (market_id,)
    )
    
    # Performance tracking
    balance = 0.0
    position = 0.0  # Net contracts (YES is positive, NO is negative)
    avg_price = 0.0
    realized_pnl = 0.0
    
    trades = []
    pnl_history = []
    
    print(f"Starting backtest for {market_id}...")
    print(f"{'Timestamp':<25} | {'Action':<10} | {'Price':<6} | {'Pos':<4} | {'P&L':<8}")
    print("-" * 65)

    for row in cursor:
        # row['timestamp'] might be a string or datetime depending on sqlite3 config
        ts = row['timestamp']
        if isinstance(ts, str):
             try:
                 ts = datetime.fromisoformat(ts)
             except ValueError:
                 # Handle formats like "2026-04-12 12:00:00"
                 ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")

        event = MarketDataEvent(
            market_id=row['market_id'],
            exchange="kalshi" if "kalshi" in row['market_id'].lower() else "polymarket",
            timestamp=ts,
            yes_bid=row['yes_bid'],
            yes_ask=row['yes_ask'],
            no_bid=row['no_bid'],
            no_ask=row['no_ask'],
            last_price=row['last_price'],
            volume=row['volume']
        )
        
        intents = await strategy.on_market_data(event)
        
        if intents:
            for intent in intents:
                # Simple backtest: assume immediate fill at current ask/bid
                # This is a naive assumption but sufficient for the requirement
                fill_price = 0.0
                if intent.side == "yes":
                    fill_price = event.yes_ask
                    # For simplicity, assume fixed size of 1 contract if not specified
                    # In a real bot we'd use Kelly sizing, but here we just want to verify logic
                    size = 1.0 
                    
                    # Update position and balance
                    # cost = size * fill_price
                    # In prediction markets, price is usually 0-1
                    balance -= size * fill_price
                    position += size
                    action = "BUY YES"
                else: # side == "no"
                    # Buying NO is equivalent to selling YES in some markets, 
                    # but here we treat them as separate contracts usually.
                    # For Albert, we'll assume "yes" or "no" contracts.
                    fill_price = event.no_ask
                    size = 1.0
                    balance -= size * fill_price
                    position -= size # Simplified: NO pos reduces YES pos or vice versa
                    action = "BUY NO"
                
                trades.append({
                    "timestamp": event.timestamp,
                    "action": action,
                    "price": fill_price,
                    "size": size
                })
                
                # Mark-to-market P&L (approximate)
                current_val = position * event.yes_bid if position > 0 else abs(position) * event.no_bid
                total_pnl = balance + current_val
                
                print(f"{str(event.timestamp):<25} | {action:<10} | {fill_price:<6.2f} | {position:<4.1f} | {total_pnl:<8.2f}")

        # Track P&L at every snapshot
        current_val = position * event.yes_bid if position > 0 else abs(position) * event.no_bid
        pnl_history.append(balance + current_val)

    # Final Summary
    if not pnl_history:
        print("No snapshots found for this market.")
        return

    final_pnl = pnl_history[-1]
    max_drawdown = 0.0
    peak = -float('inf')
    for p in pnl_history:
        if p > peak:
            peak = p
        drawdown = peak - p
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    win_rate = 0.0
    if trades:
        # Naive win rate: percentage of trades that ended in positive P&L 
        # (difficult to define for individual trades in a streaming context)
        # Let's just report total trades
        pass

    print("\n" + "="*30)
    print("BACKTEST SUMMARY")
    print("="*30)
    print(f"Total Snapshots: {len(pnl_history)}")
    print(f"Total Trades:    {len(trades)}")
    print(f"Final P&L:       {final_pnl:.2f}")
    print(f"Max Drawdown:    {max_drawdown:.2f}")
    print("="*30)

def load_strategy_class(class_path: str) -> Type[BaseStrategy]:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m albert.backtest <market_id> <strategy_class_path> [config_json]")
        sys.exit(1)
    
    m_id = sys.argv[1]
    s_path = sys.argv[2]
    cfg = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    
    s_class = load_strategy_class(s_path)
    asyncio.run(run_backtest(m_id, s_class, cfg))
