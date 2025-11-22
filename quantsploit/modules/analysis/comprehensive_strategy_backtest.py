"""
Comprehensive Strategy Backtest Module

This module provides a command-line interface for running comprehensive backtests
across all available trading strategies over multiple time periods.
"""

from quantsploit.core.module import BaseModule
from quantsploit.utils.comprehensive_backtest import ComprehensiveBacktester, run_comprehensive_analysis
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import pandas as pd

console = Console()


class ComprehensiveStrategyBacktest(BaseModule):
    """Module for running comprehensive strategy backtests"""

    def __init__(self, framework):
        super().__init__(framework)

    def _init_options(self):
        """Initialize module options"""
        super()._init_options()
        self.options.update({
            'SYMBOLS': {
                'value': 'AAPL,MSFT,GOOGL,TSLA,SPY',
                'required': True,
                'description': 'Comma-separated list of symbols'
            },
            'OUTPUT_DIR': {
                'value': './backtest_results',
                'required': False,
                'description': 'Directory to save results'
            },
            'INITIAL_CAPITAL': {
                'value': '100000',
                'required': False,
                'description': 'Initial capital for backtests'
            },
            'COMMISSION': {
                'value': '0.001',
                'required': False,
                'description': 'Commission percentage (e.g., 0.001 = 0.1%)'
            }
        })

    @property
    def name(self) -> str:
        return "Comprehensive Strategy Backtest"

    @property
    def description(self) -> str:
        return "Run comprehensive backtests across all strategies and time periods"

    @property
    def author(self) -> str:
        return "Quantsploit Team"

    @property
    def category(self) -> str:
        return "analysis"

    def run(self):
        """Execute the comprehensive backtest"""
        # Parse options
        symbols_str = self.get_option('SYMBOLS')
        symbols = [s.strip().upper() for s in symbols_str.split(',')]
        output_dir = self.get_option('OUTPUT_DIR')
        initial_capital = float(self.get_option('INITIAL_CAPITAL'))
        commission = float(self.get_option('COMMISSION'))

        console.print("\n[bold cyan]═══ Comprehensive Strategy Backtest ═══[/bold cyan]\n")
        console.print(f"[yellow]Symbols:[/yellow] {', '.join(symbols)}")
        console.print(f"[yellow]Initial Capital:[/yellow] ${initial_capital:,.2f}")
        console.print(f"[yellow]Commission:[/yellow] {commission*100:.3f}%")
        console.print(f"[yellow]Output Directory:[/yellow] {output_dir}\n")

        # Create backtester
        backtester = ComprehensiveBacktester(
            symbols=symbols,
            initial_capital=initial_capital,
            commission_pct=commission,
            slippage_pct=commission  # Use same as commission
        )

        # Show test configuration
        periods = backtester.generate_test_periods()
        total_tests = len(backtester.strategies) * len(symbols) * len(periods)

        console.print(f"[green]Configuration:[/green]")
        console.print(f"  • Strategies: {len(backtester.strategies)}")
        console.print(f"  • Symbols: {len(symbols)}")
        console.print(f"  • Time Periods: {len(periods)}")
        console.print(f"  • Total Backtests: {total_tests}\n")

        # Run backtests with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Running backtests...", total=None)

            results_df = backtester.run_comprehensive_backtest(parallel=False)

            progress.update(task, completed=True)

        if results_df.empty:
            console.print("[bold red]No results generated. Check logs for errors.[/bold red]")
            return

        console.print(f"\n[green]✓ Completed {len(results_df)} backtests successfully[/green]\n")

        # Generate summary
        summary = backtester.generate_summary_report(results_df)

        # Display results
        self._display_summary(summary)
        self._display_top_strategies(summary)

        # Save results
        backtester.save_results(results_df, summary, output_dir)

        console.print(f"\n[green]✓ Results saved to {output_dir}[/green]")
        console.print(f"  • Detailed CSV: detailed_results_*.csv")
        console.print(f"  • Summary JSON: summary_*.json")
        console.print(f"  • Markdown Report: report_*.md\n")

    def _display_summary(self, summary: dict):
        """Display overall summary statistics"""
        stats = summary['overall_stats']

        table = Table(title="Overall Performance Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Backtests", str(stats['total_backtests']))
        table.add_row("Average Return", f"{stats['avg_return']:.2f}%")
        table.add_row("Average Sharpe Ratio", f"{stats['avg_sharpe']:.2f}")
        table.add_row("Average Win Rate", f"{stats['avg_win_rate']:.2f}%")
        table.add_row("Average Signal Accuracy", f"{stats['avg_signal_accuracy']:.2f}%")
        table.add_row("Beating Buy & Hold",
                     f"{stats['strategies_beating_buy_hold']} ({stats['percentage_beating_buy_hold']:.1f}%)")

        console.print(table)
        console.print()

    def _display_top_strategies(self, summary: dict):
        """Display top performing strategies"""
        # Top by Return
        console.print("[bold cyan]Top 5 Strategies by Total Return[/bold cyan]")
        table = Table(show_header=True)
        table.add_column("Strategy", style="cyan")
        table.add_column("Symbol", style="yellow")
        table.add_column("Period", style="magenta")
        table.add_column("Return", style="green")
        table.add_column("Win Rate", style="blue")
        table.add_column("Accuracy", style="red")

        for row in summary['best_by_total_return'][:5]:
            table.add_row(
                row['strategy_name'],
                row['symbol'],
                row['period_name'][:20],  # Truncate period name
                f"{row['total_return']:.2f}%",
                f"{row['win_rate']:.1f}%",
                f"{row['signal_accuracy']:.1f}%"
            )

        console.print(table)
        console.print()

        # Top by Sharpe Ratio
        console.print("[bold cyan]Top 5 Strategies by Sharpe Ratio[/bold cyan]")
        table = Table(show_header=True)
        table.add_column("Strategy", style="cyan")
        table.add_column("Symbol", style="yellow")
        table.add_column("Period", style="magenta")
        table.add_column("Sharpe", style="green")
        table.add_column("Return", style="blue")

        for row in summary['best_by_sharpe_ratio'][:5]:
            table.add_row(
                row['strategy_name'],
                row['symbol'],
                row['period_name'][:20],
                f"{row['sharpe_ratio']:.2f}",
                f"{row['total_return']:.2f}%"
            )

        console.print(table)
        console.print()

        # Top by Signal Accuracy
        console.print("[bold cyan]Top 5 Strategies by Signal Accuracy[/bold cyan]")
        table = Table(show_header=True)
        table.add_column("Strategy", style="cyan")
        table.add_column("Symbol", style="yellow")
        table.add_column("Period", style="magenta")
        table.add_column("Accuracy", style="green")
        table.add_column("Win Rate", style="blue")
        table.add_column("Return", style="red")

        for row in summary['best_by_signal_accuracy'][:5]:
            table.add_row(
                row['strategy_name'],
                row['symbol'],
                row['period_name'][:20],
                f"{row['signal_accuracy']:.1f}%",
                f"{row['win_rate']:.1f}%",
                f"{row['total_return']:.2f}%"
            )

        console.print(table)
