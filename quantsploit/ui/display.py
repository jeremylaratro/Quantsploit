"""
Display utilities for the TUI
"""

from rich.console import Console as RichConsole
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from typing import Dict, List, Any
import pandas as pd


class Display:
    """
    Handles formatted output to the console
    """

    def __init__(self):
        self.console = RichConsole()

    def print(self, text: str, style: str = ""):
        """Print text with optional style"""
        self.console.print(text, style=style)

    def print_banner(self):
        """Print the Quantsploit banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗██████╗║
║  ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔══██║
║  ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ███████╗██████╔╝
║  ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ╚════██║██╔═══╝║
║  ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ███████║██║    ║
║   ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝    ║
║                    EXPLOIT THE MARKET                         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        """
        self.console.print(banner, style="bold cyan")
        self.console.print("  Quantitative Analysis Trading Framework v0.1.0\n", style="bold white")

    def print_success(self, message: str):
        """Print success message"""
        self.console.print(f"[+] {message}", style="bold green")

    def print_error(self, message: str):
        """Print error message"""
        self.console.print(f"[-] {message}", style="bold red")

    def print_warning(self, message: str):
        """Print warning message"""
        self.console.print(f"[!] {message}", style="bold yellow")

    def print_info(self, message: str):
        """Print info message"""
        self.console.print(f"[*] {message}", style="bold blue")

    def print_module_info(self, module_info: Dict[str, Any]):
        """Display module information"""
        table = Table(title=f"Module: {module_info['name']}", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Name", module_info['name'])
        table.add_row("Category", module_info['category'])
        table.add_row("Description", module_info['description'])
        table.add_row("Author", module_info['author'])

        self.console.print(table)

        # Display detailed trading guide if available
        if 'trading_guide' in module_info and module_info['trading_guide']:
            self.console.print("\n")
            self.console.print("[bold cyan]═══ TRADING GUIDE ═══[/bold cyan]")
            self.console.print(module_info['trading_guide'])
            self.console.print("[bold cyan]═" * 40 + "[/bold cyan]")

    def print_options(self, options: Dict[str, Any]):
        """Display module options"""
        table = Table(title="Module Options", show_header=True, header_style="bold magenta")
        table.add_column("Option", style="cyan")
        table.add_column("Current Value", style="white")
        table.add_column("Required", style="yellow")
        table.add_column("Description", style="white")

        for key, opt in options.items():
            required = "yes" if opt.get("required", False) else "no"
            value = str(opt.get("value", "")) if opt.get("value") is not None else ""
            table.add_row(
                key,
                value,
                required,
                opt.get("description", "")
            )

        self.console.print(table)

    def print_modules(self, modules: List[Any], category: str = "All"):
        """Display list of modules"""
        table = Table(title=f"Available Modules - {category}", show_header=True, header_style="bold magenta")
        table.add_column("Path", style="cyan")
        table.add_column("Category", style="yellow")
        table.add_column("Description", style="white")

        for module in modules:
            table.add_row(
                module.path,
                module.category,
                module.description
            )

        self.console.print(table)

    def print_dataframe(self, df: pd.DataFrame, title: str = "Data", max_rows: int = 50):
        """Display a pandas DataFrame"""
        if df is None or df.empty:
            self.print_warning("No data to display")
            return

        # Limit rows
        if len(df) > max_rows:
            display_df = df.head(max_rows)
            self.print_warning(f"Showing first {max_rows} of {len(df)} rows")
        else:
            display_df = df

        table = Table(title=title, show_header=True, header_style="bold magenta")

        # Add columns
        for col in display_df.columns:
            table.add_column(str(col))

        # Add rows
        for idx, row in display_df.iterrows():
            table.add_row(*[str(val) for val in row])

        self.console.print(table)

    def print_results(self, results: Dict[str, Any]):
        """Display module execution results"""
        if not results.get("success", False):
            self.print_error(f"Execution failed: {results.get('error', 'Unknown error')}")
            return

        self.print_success("Module executed successfully\n")

        # Display results based on type
        for key, value in results.items():
            if key in ["success", "error"]:
                continue

            if isinstance(value, pd.DataFrame):
                self.print_dataframe(value, title=key.replace('_', ' ').title())
            elif isinstance(value, dict):
                self._print_dict(value, key.replace('_', ' ').title())
            elif isinstance(value, list):
                self._print_list(value, key.replace('_', ' ').title())
            else:
                self.print_info(f"{key}: {value}")

    def _print_dict(self, data: Dict[str, Any], title: str = "Data"):
        """Print dictionary as table"""
        table = Table(title=title, show_header=False)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="white")

        for key, value in data.items():
            table.add_row(str(key), str(value))

        self.console.print(table)

    def _print_list(self, data: List[Any], title: str = "Data"):
        """Print list as numbered items"""
        self.console.print(f"\n[bold]{title}:[/bold]")
        for i, item in enumerate(data, 1):
            self.console.print(f"  {i}. {item}")

    def print_help(self, commands: Dict[str, str]):
        """Display help information"""
        table = Table(title="Available Commands", show_header=True, header_style="bold magenta")
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="white")

        for cmd, desc in sorted(commands.items()):
            table.add_row(cmd, desc)

        self.console.print(table)

    def print_prompt(self, prompt: str):
        """Print the command prompt"""
        self.console.print(prompt, end="")
