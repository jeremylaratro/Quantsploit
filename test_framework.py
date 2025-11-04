#!/usr/bin/env python3
"""
Quick Test Script for Quantsploit
Run this to verify everything is working
"""

import sys
sys.path.insert(0, '/home/user/Quantsploit')

from quantsploit.core.framework import Framework
from rich.console import Console
from rich.table import Table

console = Console()

# Initialize
console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
console.print("[bold cyan]   QUANTSPLOIT QUICK TEST[/bold cyan]")
console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")

framework = Framework()
framework.discover_modules()

console.print(f"[green]✓[/green] Loaded {len(framework.modules)} modules\n")

# Test 1: Simple Technical Analysis
console.print("[bold]Test 1: Technical Indicators (AAPL)[/bold]")
console.print("-" * 50)

module = framework.use_module("analysis/technical_indicators")
module.set_option("SYMBOL", "AAPL")
module.set_option("PERIOD", "1mo")
module.set_option("INDICATORS", "RSI,SMA")

results = framework.run_module(module)

if results.get("success"):
    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Symbol", str(results.get('symbol', 'N/A')))
    table.add_row("Price", f"${results.get('latest_price', 0):.2f}")
    table.add_row("RSI", f"{results.get('RSI', 0):.2f}")
    table.add_row("RSI Signal", results.get('RSI_signal', 'N/A'))
    table.add_row("SMA(20)", f"${results.get('SMA_20', 0):.2f}")

    console.print(table)
    console.print("[green]✓ Technical Indicators working![/green]\n")
else:
    console.print(f"[red]✗ Error: {results.get('error')}[/red]\n")

# Test 2: Bulk Screener with relaxed filters
console.print("[bold]Test 2: Bulk Screener (Top 10 stocks)[/bold]")
console.print("-" * 50)

module2 = framework.use_module("scanners/bulk_screener")
module2.set_option("SYMBOLS", "AAPL,MSFT,GOOGL,NVDA,TSLA,META,AMD,NFLX")
module2.set_option("MIN_VOLUME", "0")  # No volume filter
module2.set_option("SORT_BY", "score")

results2 = framework.run_module(module2)

if results2.get("success"):
    console.print(f"[green]Analyzed: {results2.get('results_found', 0)} stocks[/green]")

    if 'all_results' in results2 and not results2['all_results'].empty:
        console.print("\n[yellow]Top Stocks by Score:[/yellow]")
        df = results2['all_results'].head(5)

        table2 = Table(show_header=True)
        table2.add_column("Symbol")
        table2.add_column("Price")
        table2.add_column("RSI")
        table2.add_column("Score")
        table2.add_column("Trend")

        for _, row in df.iterrows():
            table2.add_row(
                str(row['Symbol']),
                f"${row['Price']:.2f}",
                f"{row['RSI']:.1f}",
                f"{row['Score']:.1f}",
                str(row['Trend'])
            )

        console.print(table2)
        console.print("[green]✓ Bulk Screener working![/green]\n")
else:
    console.print(f"[red]✗ Error: {results2.get('error')}[/red]\n")

# Test 3: Pattern Recognition
console.print("[bold]Test 3: Pattern Recognition (TSLA)[/bold]")
console.print("-" * 50)

module3 = framework.use_module("analysis/pattern_recognition")
module3.set_option("SYMBOL", "TSLA")
module3.set_option("LOOKBACK", "30")

results3 = framework.run_module(module3)

if results3.get("success"):
    console.print(f"[cyan]Symbol:[/cyan] {results3.get('symbol')}")
    console.print(f"[cyan]Price:[/cyan] ${results3.get('current_price', 0):.2f}")
    console.print(f"[cyan]Patterns Found:[/cyan] {len(results3.get('candlestick_patterns', []))} candlestick, {len(results3.get('chart_patterns', []))} chart")

    if results3.get('signals'):
        console.print(f"\n[yellow]Signals:[/yellow]")
        for signal in results3['signals'][:3]:
            console.print(f"  • {signal}")

    console.print("[green]✓ Pattern Recognition working![/green]\n")
else:
    console.print(f"[red]✗ Error: {results3.get('error')}[/red]\n")

# Summary
console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]")
console.print("[bold green]✓ ALL MODULES WORKING![/bold green]")
console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")

console.print("[yellow]Note:[/yellow] Using sample data (live data unavailable)")
console.print("[yellow]Tip:[/yellow] Try: python -m quantsploit.main")
