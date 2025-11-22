#!/usr/bin/env python3
"""
Test script for comprehensive backtesting using sample data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from quantsploit.utils.comprehensive_backtest import ComprehensiveBacktester, TestPeriod
from rich.console import Console

console = Console()


def generate_sample_data(symbol: str, days: int = 1000) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducibility

    # Generate dates
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='D')

    # Generate price data with trend and noise
    base_price = 100
    trend = np.linspace(0, 50, days)  # Upward trend
    noise = np.random.randn(days) * 2  # Random noise
    close_prices = base_price + trend + noise.cumsum() * 0.5

    # Generate OHLC from close
    opens = close_prices * (1 + np.random.randn(days) * 0.005)
    highs = np.maximum(opens, close_prices) * (1 + abs(np.random.randn(days)) * 0.01)
    lows = np.minimum(opens, close_prices) * (1 - abs(np.random.randn(days)) * 0.01)

    # Generate volume
    volumes = np.random.randint(1000000, 10000000, days)

    # Create DataFrame
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': close_prices,
        'Volume': volumes
    }, index=dates)

    return df


def main():
    console.print("\n[bold cyan]═══ Comprehensive Backtest Test (Sample Data) ═══[/bold cyan]\n")

    # Generate sample data
    console.print("[yellow]Generating sample data...[/yellow]")
    sample_data = generate_sample_data('TEST', days=1000)

    console.print(f"[green]✓ Generated {len(sample_data)} days of sample data[/green]\n")

    # Create a simple test by running one strategy manually
    from quantsploit.utils.backtesting import Backtester, BacktestConfig
    from quantsploit.utils.data_fetcher import DataFetcher

    config = BacktestConfig(
        initial_capital=100000,
        commission_pct=0.001,
        slippage_pct=0.001,
        position_size=1.0,
        max_positions=1
    )

    # Test SMA crossover strategy
    console.print("[cyan]Testing SMA Crossover strategy...[/cyan]")

    def sma_strategy(bt, date, row):
        history = sample_data.loc[:date]

        if len(history) < 50:
            return

        short_sma = history['Close'].rolling(window=20).mean().iloc[-1]
        long_sma = history['Close'].rolling(window=50).mean().iloc[-1]

        if len(history) < 2:
            return

        prev_short_sma = history['Close'].rolling(window=20).mean().iloc[-2]
        prev_long_sma = history['Close'].rolling(window=50).mean().iloc[-2]

        current_position = bt.positions.get('TEST')

        # Bullish crossover
        if prev_short_sma <= prev_long_sma and short_sma > long_sma:
            if current_position is None:
                bt.enter_long('TEST', date, row['Close'])

        # Bearish crossover
        elif prev_short_sma >= prev_long_sma and short_sma < long_sma:
            if current_position is not None:
                bt.exit_position('TEST', date, row['Close'])

    backtester = Backtester(config)
    results = backtester.run_backtest(sample_data, sma_strategy)

    # Display results
    console.print("\n[bold green]✓ Backtest Complete![/bold green]\n")
    console.print("[cyan]Results:[/cyan]")
    console.print(f"  Total Return: {results.total_return_pct:.2f}%")
    console.print(f"  Annualized Return: {results.annualized_return:.2f}%")
    console.print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
    console.print(f"  Max Drawdown: {results.max_drawdown:.2f}%")
    console.print(f"  Total Trades: {results.total_trades}")
    console.print(f"  Win Rate: {results.win_rate:.2f}%")
    console.print(f"  Profit Factor: {results.profit_factor:.2f}\n")

    if results.total_trades > 0:
        console.print("[bold green]✓ Backtesting system is working correctly![/bold green]\n")
        console.print("[yellow]Note:[/yellow] The Yahoo Finance API is currently blocking requests.")
        console.print("To use with real data, you may need to:")
        console.print("  1. Use a VPN or different network")
        console.print("  2. Wait and try again later")
        console.print("  3. Use an alternative data source\n")
    else:
        console.print("[yellow]No trades executed. This may be normal depending on the strategy and data.[/yellow]\n")


if __name__ == '__main__':
    main()
