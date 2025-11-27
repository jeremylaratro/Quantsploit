"""
Helper utility functions
"""

from typing import Any, Dict, List

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover - graceful fallback when optional dep missing
    def tabulate(data, headers=None, tablefmt="simple"):
        """Minimal tabulate fallback to avoid hard dependency during imports."""
        headers = headers or []
        lines = []
        if headers:
            lines.append("\t".join(map(str, headers)))
        for row in data:
            if isinstance(row, dict):
                values = [row.get(h, "") for h in headers] if headers else row.values()
            else:
                values = row
            lines.append("\t".join(map(str, values)))
        return "\n".join(lines)

from rich.console import Console
from rich.table import Table


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a number as currency"""
    if value is None:
        return "N/A"
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a number as percentage"""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with specified decimals"""
    if value is None:
        return "N/A"
    return f"{value:,.{decimals}f}"


def format_table(data: List[Dict[str, Any]], headers: List[str] = None,
                tablefmt: str = "simple") -> str:
    """
    Format data as a table

    Args:
        data: List of dictionaries with table data
        headers: List of header names (uses dict keys if None)
        tablefmt: Table format (simple, grid, fancy_grid, etc.)

    Returns:
        Formatted table string
    """
    if not data:
        return "No data to display"

    if headers is None and isinstance(data[0], dict):
        headers = list(data[0].keys())

    return tabulate(data, headers=headers, tablefmt=tablefmt)


def create_rich_table(title: str, columns: List[str], rows: List[List[Any]]) -> Table:
    """
    Create a Rich table for beautiful console output

    Args:
        title: Table title
        columns: Column headers
        rows: List of row data

    Returns:
        Rich Table object
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")

    for col in columns:
        table.add_column(col)

    for row in rows:
        table.add_row(*[str(item) for item in row])

    return table


def color_text(text: str, color: str = "green") -> str:
    """
    Add ANSI color codes to text

    Args:
        text: Text to color
        color: Color name (green, red, yellow, blue, etc.)

    Returns:
        Colored text string
    """
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }

    color_code = colors.get(color, colors["reset"])
    return f"{color_code}{text}{colors['reset']}"


def format_large_number(num: float) -> str:
    """
    Format large numbers with abbreviations (K, M, B, T)

    Args:
        num: Number to format

    Returns:
        Formatted string
    """
    if num is None:
        return "N/A"

    magnitude = 0
    suffixes = ['', 'K', 'M', 'B', 'T']

    while abs(num) >= 1000 and magnitude < len(suffixes) - 1:
        magnitude += 1
        num /= 1000.0

    return f"{num:.2f}{suffixes[magnitude]}"
