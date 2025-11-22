#!/usr/bin/env python3
"""
Standalone script to run comprehensive strategy backtests

Usage:
    python run_comprehensive_backtest.py --symbols AAPL,MSFT,GOOGL --output ./results
    python run_comprehensive_backtest.py --symbols SPY,QQQ,IWM --capital 50000
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from quantsploit.utils.comprehensive_backtest import run_comprehensive_analysis
from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive backtests across multiple strategies and time periods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test major tech stocks with default settings
  python run_comprehensive_backtest.py --symbols AAPL,MSFT,GOOGL,TSLA

  # Test ETFs with custom capital
  python run_comprehensive_backtest.py --symbols SPY,QQQ,IWM --capital 50000

  # Full test with custom output directory
  python run_comprehensive_backtest.py --symbols AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA \\
      --capital 100000 --commission 0.001 --output ./my_backtest_results

  # Custom time periods: 4 periods of 6 months each over 2 years
  python run_comprehensive_backtest.py --symbols AAPL,MSFT --tspan 2y --bspan 6m --period 4

  # Custom time periods: 6 periods of 3 months each over 18 months
  python run_comprehensive_backtest.py --symbols SPY,QQQ --tspan 18m --bspan 3m --period 6
        """
    )

    parser.add_argument(
        '--symbols',
        type=str,
        default='AAPL,MSFT,GOOGL,TSLA,SPY',
        help='Comma-separated list of stock symbols (default: AAPL,MSFT,GOOGL,TSLA,SPY)'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='Initial capital for backtests (default: 100000)'
    )

    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission percentage per trade (default: 0.001 = 0.1%%)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./backtest_results',
        help='Output directory for results (default: ./backtest_results)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: test only 1 symbol with fewer strategies'
    )

    parser.add_argument(
        '--tspan',
        type=str,
        default=None,
        help='Total time span (e.g., 2y, 18m, 730d). If not specified, uses default periods'
    )

    parser.add_argument(
        '--bspan',
        type=str,
        default=None,
        help='Backtest span for each period (e.g., 6m, 180d, 1y). Required if --tspan is provided'
    )

    parser.add_argument(
        '--period',
        type=int,
        default=None,
        help='Number of separate backtest periods to run. Required if --tspan is provided'
    )

    args = parser.parse_args()

    # Validate custom period arguments
    if any([args.tspan, args.bspan, args.period]):
        if not all([args.tspan, args.bspan, args.period]):
            console.print("[bold red]Error:[/bold red] When using custom periods, all three arguments (--tspan, --bspan, --period) must be provided")
            return 1

    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]

    if args.quick:
        symbols = symbols[:1]
        console.print("[yellow]Quick mode: Testing only first symbol[/yellow]\n")

    # Display configuration
    console.print("\n[bold cyan]═══ Comprehensive Strategy Backtest ═══[/bold cyan]\n")
    console.print(f"[yellow]Symbols:[/yellow] {', '.join(symbols)}")
    console.print(f"[yellow]Initial Capital:[/yellow] ${args.capital:,.2f}")
    console.print(f"[yellow]Commission:[/yellow] {args.commission*100:.3f}%")
    console.print(f"[yellow]Output Directory:[/yellow] {args.output}")
    if args.tspan:
        console.print(f"[yellow]Custom Periods:[/yellow] {args.period} periods of {args.bspan} each over {args.tspan}")
    console.print()

    console.print("[cyan]Starting comprehensive backtest...[/cyan]\n")

    try:
        # Run comprehensive analysis
        results_df, summary = run_comprehensive_analysis(
            symbols=symbols,
            output_dir=args.output,
            tspan=args.tspan,
            bspan=args.bspan,
            num_periods=args.period
        )

        if results_df is None:
            console.print("[bold red]Failed to generate results[/bold red]")
            return 1

        # Display summary
        stats = summary['overall_stats']
        console.print("\n[bold green]✓ Backtest Complete![/bold green]\n")
        console.print("[bold cyan]Overall Results:[/bold cyan]")
        console.print(f"  • Total Backtests: {stats['total_backtests']}")
        console.print(f"  • Average Return: {stats['avg_return']:.2f}%")
        console.print(f"  • Average Sharpe Ratio: {stats['avg_sharpe']:.2f}")
        console.print(f"  • Average Win Rate: {stats['avg_win_rate']:.2f}%")
        console.print(f"  • Average Signal Accuracy: {stats['avg_signal_accuracy']:.2f}%")
        console.print(f"  • Beating Buy & Hold: {stats['strategies_beating_buy_hold']} "
                     f"({stats['percentage_beating_buy_hold']:.1f}%)\n")

        # Show top strategy
        if summary['best_by_total_return']:
            top = summary['best_by_total_return'][0]
            console.print("[bold cyan]🏆 Top Strategy by Return:[/bold cyan]")
            console.print(f"  • Strategy: {top['strategy_name']}")
            console.print(f"  • Symbol: {top['symbol']}")
            console.print(f"  • Period: {top['period_name']}")
            console.print(f"  • Return: {top['total_return']:.2f}%")
            console.print(f"  • Win Rate: {top['win_rate']:.1f}%")
            console.print(f"  • Signal Accuracy: {top['signal_accuracy']:.1f}%")
            console.print(f"  • Sharpe Ratio: {top['sharpe_ratio']:.2f}\n")

        console.print(f"[green]📁 Results saved to:[/green] {args.output}")
        console.print(f"  • View the markdown report for detailed analysis")
        console.print(f"  • Check the CSV for full data export\n")

        return 0

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
