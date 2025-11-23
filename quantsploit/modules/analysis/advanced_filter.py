"""
Advanced Filtering System

Multi-dimensional filtering for backtest results enabling complex queries like:
- "Show tech stocks where Kalman strategy had >1.0 Sharpe over 1yr periods"
- "Find all stocks with >15% return and <20% volatility"
- "Conservative strategies in AI sector with >70% win rate"

Supports filtering by:
- Sector, stock symbols
- Strategy name, risk class
- Time period, period category
- Performance metrics (return, Sharpe, win rate, etc.)
- Reliability/quality thresholds
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.statistical_analyzer import (
    StatisticalAnalyzer,
    StrategyRiskClass,
    STRATEGY_RISK_MAP
)


class FilterOperator(Enum):
    """Comparison operators for filters"""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    BETWEEN = "between"


@dataclass
class FilterCriteria:
    """Single filter criterion"""
    field: str
    operator: FilterOperator
    value: Any
    description: str = ""


@dataclass
class FilterResult:
    """Result of applying filters"""
    original_count: int
    filtered_count: int
    reduction_pct: float

    filtered_df: pd.DataFrame
    filters_applied: List[FilterCriteria]

    # Summary statistics
    summary: Dict[str, Any]


class AdvancedFilter:
    """
    Multi-dimensional filtering system for backtest results
    """

    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize filter

        Args:
            results_df: DataFrame with backtest results
        """
        self.df = results_df.copy()
        self.stat_analyzer = StatisticalAnalyzer(min_sample_size=3)

        # Add computed columns
        self._enrich_data()

    @classmethod
    def from_csv(cls, csv_path: str) -> 'AdvancedFilter':
        """Load from CSV file"""
        df = pd.read_csv(csv_path)
        return cls(df)

    @classmethod
    def from_timestamp(cls, timestamp: str, results_dir: str = 'backtest_results') -> 'AdvancedFilter':
        """Load from timestamp"""
        csv_path = Path(results_dir) / f'detailed_results_{timestamp}.csv'
        return cls.from_csv(str(csv_path))

    def filter(
        self,
        criteria: List[FilterCriteria],
        min_trades: int = 5
    ) -> FilterResult:
        """
        Apply multiple filter criteria

        Args:
            criteria: List of filter criteria
            min_trades: Minimum trades threshold

        Returns:
            FilterResult with filtered data and statistics
        """
        original_count = len(self.df)
        filtered_df = self.df.copy()

        # Apply minimum trades filter first
        if 'total_trades' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['total_trades'] >= min_trades]

        # Apply each criterion
        for criterion in criteria:
            filtered_df = self._apply_criterion(filtered_df, criterion)

        filtered_count = len(filtered_df)
        reduction_pct = (1 - filtered_count / original_count) * 100 if original_count > 0 else 0

        # Calculate summary statistics
        summary = self._calculate_summary(filtered_df)

        return FilterResult(
            original_count=original_count,
            filtered_count=filtered_count,
            reduction_pct=reduction_pct,
            filtered_df=filtered_df,
            filters_applied=criteria,
            summary=summary
        )

    def quick_filter(
        self,
        sector: Optional[str] = None,
        symbol: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        strategy: Optional[str] = None,
        strategies: Optional[List[str]] = None,
        risk_class: Optional[StrategyRiskClass] = None,
        period: Optional[str] = None,
        period_category: Optional[str] = None,
        min_return: Optional[float] = None,
        max_return: Optional[float] = None,
        min_sharpe: Optional[float] = None,
        min_win_rate: Optional[float] = None,
        max_volatility: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        min_trades: int = 5
    ) -> pd.DataFrame:
        """
        Quick filtering with common parameters

        Args:
            sector: Filter by sector
            symbol: Filter by single symbol
            symbols: Filter by list of symbols
            strategy: Filter by single strategy
            strategies: Filter by list of strategies
            risk_class: Filter by strategy risk class
            period: Filter by specific period
            period_category: Filter by period category (short/medium/long)
            min_return: Minimum return threshold
            max_return: Maximum return threshold
            min_sharpe: Minimum Sharpe ratio
            min_win_rate: Minimum win rate
            max_volatility: Maximum volatility
            max_drawdown: Maximum drawdown (less negative)
            min_trades: Minimum trades

        Returns:
            Filtered DataFrame
        """
        criteria = []

        if sector:
            criteria.append(FilterCriteria('sector', FilterOperator.EQUALS, sector, f"Sector = {sector}"))

        if symbol:
            criteria.append(FilterCriteria('symbol', FilterOperator.EQUALS, symbol, f"Symbol = {symbol}"))

        if symbols:
            criteria.append(FilterCriteria('symbol', FilterOperator.IN, symbols, f"Symbol in {symbols}"))

        if strategy:
            criteria.append(FilterCriteria('strategy_name', FilterOperator.EQUALS, strategy, f"Strategy = {strategy}"))

        if strategies:
            criteria.append(FilterCriteria('strategy_name', FilterOperator.IN, strategies, f"Strategy in {strategies}"))

        if risk_class:
            criteria.append(FilterCriteria('risk_class', FilterOperator.EQUALS, risk_class, f"Risk Class = {risk_class.value}"))

        if period:
            criteria.append(FilterCriteria('period_name', FilterOperator.EQUALS, period, f"Period = {period}"))

        if period_category:
            criteria.append(FilterCriteria('period_category', FilterOperator.EQUALS, period_category, f"Period Category = {period_category}"))

        if min_return is not None:
            criteria.append(FilterCriteria('total_return', FilterOperator.GREATER_EQUAL, min_return, f"Return >= {min_return}%"))

        if max_return is not None:
            criteria.append(FilterCriteria('total_return', FilterOperator.LESS_EQUAL, max_return, f"Return <= {max_return}%"))

        if min_sharpe is not None:
            criteria.append(FilterCriteria('sharpe_ratio', FilterOperator.GREATER_EQUAL, min_sharpe, f"Sharpe >= {min_sharpe}"))

        if min_win_rate is not None:
            criteria.append(FilterCriteria('win_rate', FilterOperator.GREATER_EQUAL, min_win_rate, f"Win Rate >= {min_win_rate}%"))

        if max_volatility is not None:
            criteria.append(FilterCriteria('volatility', FilterOperator.LESS_EQUAL, max_volatility, f"Volatility <= {max_volatility}%"))

        if max_drawdown is not None:
            criteria.append(FilterCriteria('max_drawdown', FilterOperator.GREATER_EQUAL, max_drawdown, f"Max DD >= {max_drawdown}%"))

        result = self.filter(criteria, min_trades=min_trades)
        return result.filtered_df

    def top_n(
        self,
        n: int = 10,
        sort_by: str = 'total_return',
        ascending: bool = False,
        **filter_kwargs
    ) -> pd.DataFrame:
        """
        Get top N results after filtering

        Args:
            n: Number of results
            sort_by: Column to sort by
            ascending: Sort order
            **filter_kwargs: Arguments for quick_filter

        Returns:
            Top N DataFrame
        """
        filtered = self.quick_filter(**filter_kwargs)

        if len(filtered) == 0:
            return pd.DataFrame()

        sorted_df = filtered.sort_values(sort_by, ascending=ascending)
        return sorted_df.head(n)

    def rank_by(
        self,
        group_by: str,
        metric: str = 'total_return',
        top_n: int = 10,
        **filter_kwargs
    ) -> pd.DataFrame:
        """
        Rank groups by a metric after filtering

        Args:
            group_by: Column to group by (e.g., 'symbol', 'strategy_name')
            metric: Metric to rank by
            top_n: Number of top groups to return
            **filter_kwargs: Arguments for quick_filter

        Returns:
            Ranked DataFrame
        """
        filtered = self.quick_filter(**filter_kwargs)

        if len(filtered) == 0:
            return pd.DataFrame()

        # Aggregate by group
        grouped = filtered.groupby(group_by).agg({
            metric: ['mean', 'median', 'std', 'count'],
            'symbol': 'nunique' if group_by != 'symbol' else 'first',
            'strategy_name': 'nunique' if group_by != 'strategy_name' else 'first',
        }).round(2)

        grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]

        # Sort by mean
        grouped = grouped.sort_values(f'{metric}_mean', ascending=False)

        return grouped.head(top_n)

    def _enrich_data(self):
        """Add computed columns to data"""
        # Add sector
        try:
            from dashboard.ticker_universe import get_sector
            self.df['sector'] = self.df['symbol'].apply(get_sector)
        except ImportError:
            self.df['sector'] = 'Unknown'

        # Add risk class
        self.df['risk_class'] = self.df['strategy_name'].map(
            lambda x: STRATEGY_RISK_MAP.get(x, StrategyRiskClass.MODERATE)
        )

        # Add period category
        from modules.analysis.period_analyzer import PeriodAnalyzer
        temp_analyzer = PeriodAnalyzer(self.df)
        self.df['period_category'] = self.df['period_name'].apply(temp_analyzer._categorize_period)

        # Add excess return (vs buy and hold)
        if 'buy_and_hold_return' in self.df.columns:
            self.df['excess_return'] = self.df['total_return'] - self.df['buy_and_hold_return']
        elif 'benchmark_return' in self.df.columns:
            self.df['excess_return'] = self.df['total_return'] - self.df['benchmark_return']

        # Add return category
        self.df['return_category'] = pd.cut(
            self.df['total_return'],
            bins=[-np.inf, -10, 0, 10, 20, np.inf],
            labels=['large_loss', 'loss', 'small_gain', 'good_gain', 'excellent']
        )

        # Add sharpe category
        if 'sharpe_ratio' in self.df.columns:
            self.df['sharpe_category'] = pd.cut(
                self.df['sharpe_ratio'],
                bins=[-np.inf, 0, 0.5, 1.0, 2.0, np.inf],
                labels=['poor', 'below_avg', 'average', 'good', 'excellent']
            )

    def _apply_criterion(self, df: pd.DataFrame, criterion: FilterCriteria) -> pd.DataFrame:
        """Apply a single filter criterion"""
        field = criterion.field
        op = criterion.operator
        value = criterion.value

        if field not in df.columns:
            print(f"Warning: Field '{field}' not found in data. Skipping filter.")
            return df

        if op == FilterOperator.EQUALS:
            return df[df[field] == value]

        elif op == FilterOperator.NOT_EQUALS:
            return df[df[field] != value]

        elif op == FilterOperator.GREATER:
            return df[df[field] > value]

        elif op == FilterOperator.GREATER_EQUAL:
            return df[df[field] >= value]

        elif op == FilterOperator.LESS:
            return df[df[field] < value]

        elif op == FilterOperator.LESS_EQUAL:
            return df[df[field] <= value]

        elif op == FilterOperator.IN:
            return df[df[field].isin(value)]

        elif op == FilterOperator.NOT_IN:
            return df[~df[field].isin(value)]

        elif op == FilterOperator.CONTAINS:
            return df[df[field].str.contains(value, case=False, na=False)]

        elif op == FilterOperator.BETWEEN:
            if isinstance(value, tuple) and len(value) == 2:
                return df[(df[field] >= value[0]) & (df[field] <= value[1])]
            else:
                print(f"Warning: BETWEEN operator requires tuple of (min, max). Skipping.")
                return df

        else:
            print(f"Warning: Unknown operator {op}. Skipping filter.")
            return df

    def _calculate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for filtered results"""
        if len(df) == 0:
            return {}

        summary = {
            'count': len(df),
            'num_symbols': df['symbol'].nunique() if 'symbol' in df.columns else 0,
            'num_strategies': df['strategy_name'].nunique() if 'strategy_name' in df.columns else 0,
            'num_periods': df['period_name'].nunique() if 'period_name' in df.columns else 0,
        }

        # Performance metrics
        if 'total_return' in df.columns:
            summary['avg_return'] = df['total_return'].mean()
            summary['median_return'] = df['total_return'].median()
            summary['best_return'] = df['total_return'].max()
            summary['worst_return'] = df['total_return'].min()

        if 'sharpe_ratio' in df.columns:
            summary['avg_sharpe'] = df['sharpe_ratio'].mean()
            summary['median_sharpe'] = df['sharpe_ratio'].median()

        if 'win_rate' in df.columns:
            summary['avg_win_rate'] = df['win_rate'].mean()

        if 'volatility' in df.columns:
            summary['avg_volatility'] = df['volatility'].mean()

        # Best combination
        if len(df) > 0:
            best_idx = df['total_return'].idxmax()
            summary['best_combination'] = {
                'symbol': df.loc[best_idx, 'symbol'] if 'symbol' in df.columns else None,
                'strategy': df.loc[best_idx, 'strategy_name'] if 'strategy_name' in df.columns else None,
                'period': df.loc[best_idx, 'period_name'] if 'period_name' in df.columns else None,
                'return': df.loc[best_idx, 'total_return'] if 'total_return' in df.columns else None,
            }

        return summary

    def format_filter_result(self, result: FilterResult) -> str:
        """Format filter result as text"""
        report = f"""
{'='*70}
FILTER RESULTS
{'='*70}

FILTERS APPLIED:
"""

        for i, criterion in enumerate(result.filters_applied, 1):
            desc = criterion.description or f"{criterion.field} {criterion.operator.value} {criterion.value}"
            report += f"  {i}. {desc}\n"

        report += f"""
RESULTS:
  Original Count:  {result.original_count:,}
  Filtered Count:  {result.filtered_count:,}
  Reduction:       {result.reduction_pct:.1f}%

SUMMARY STATISTICS:
  Symbols:         {result.summary.get('num_symbols', 0)}
  Strategies:      {result.summary.get('num_strategies', 0)}
  Periods:         {result.summary.get('num_periods', 0)}

PERFORMANCE:
  Avg Return:      {result.summary.get('avg_return', 0):>8.2f}%
  Median Return:   {result.summary.get('median_return', 0):>8.2f}%
  Best Return:     {result.summary.get('best_return', 0):>8.2f}%
  Worst Return:    {result.summary.get('worst_return', 0):>8.2f}%
  Avg Sharpe:      {result.summary.get('avg_sharpe', 0):>8.2f}
  Avg Win Rate:    {result.summary.get('avg_win_rate', 0):>8.1f}%
"""

        if 'best_combination' in result.summary:
            best = result.summary['best_combination']
            report += f"""
BEST COMBINATION:
  Symbol:          {best.get('symbol', 'N/A')}
  Strategy:        {best.get('strategy', 'N/A')}
  Period:          {best.get('period', 'N/A')}
  Return:          {best.get('return', 0):>8.2f}%
"""

        report += f"\n{'='*70}\n"

        return report


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Advanced filtering of backtest results')
    parser.add_argument('--timestamp', type=str, help='Backtest timestamp')
    parser.add_argument('--csv', type=str, help='Path to CSV file')

    # Filter arguments
    parser.add_argument('--sector', type=str, help='Filter by sector')
    parser.add_argument('--symbol', type=str, help='Filter by symbol')
    parser.add_argument('--strategy', type=str, help='Filter by strategy')
    parser.add_argument('--period', type=str, help='Filter by period')
    parser.add_argument('--period-category', type=str, choices=['short', 'medium', 'long'], help='Filter by period category')
    parser.add_argument('--risk-class', type=str, choices=['conservative', 'moderate', 'aggressive'], help='Filter by strategy risk class')

    # Metric thresholds
    parser.add_argument('--min-return', type=float, help='Minimum return')
    parser.add_argument('--min-sharpe', type=float, help='Minimum Sharpe ratio')
    parser.add_argument('--min-win-rate', type=float, help='Minimum win rate')
    parser.add_argument('--max-volatility', type=float, help='Maximum volatility')
    parser.add_argument('--min-trades', type=int, default=5, help='Minimum trades')

    # Output options
    parser.add_argument('--top-n', type=int, help='Show only top N results')
    parser.add_argument('--sort-by', type=str, default='total_return', help='Sort by column')
    parser.add_argument('--export', type=str, help='Export to CSV file')

    args = parser.parse_args()

    # Load data
    if args.csv:
        filter_sys = AdvancedFilter.from_csv(args.csv)
    elif args.timestamp:
        filter_sys = AdvancedFilter.from_timestamp(args.timestamp)
    else:
        # Find latest
        results_dir = Path('backtest_results')
        csv_files = list(results_dir.glob('detailed_results_*.csv'))
        if not csv_files:
            print("No backtest results found!")
            return
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"Using latest results: {latest_csv.name}\n")
        filter_sys = AdvancedFilter.from_csv(str(latest_csv))

    # Build filter kwargs
    filter_kwargs = {
        'min_trades': args.min_trades
    }

    if args.sector:
        filter_kwargs['sector'] = args.sector
    if args.symbol:
        filter_kwargs['symbol'] = args.symbol
    if args.strategy:
        filter_kwargs['strategy'] = args.strategy
    if args.period:
        filter_kwargs['period'] = args.period
    if args.period_category:
        filter_kwargs['period_category'] = args.period_category
    if args.risk_class:
        risk_class_map = {
            'conservative': StrategyRiskClass.CONSERVATIVE,
            'moderate': StrategyRiskClass.MODERATE,
            'aggressive': StrategyRiskClass.AGGRESSIVE
        }
        filter_kwargs['risk_class'] = risk_class_map[args.risk_class]
    if args.min_return is not None:
        filter_kwargs['min_return'] = args.min_return
    if args.min_sharpe is not None:
        filter_kwargs['min_sharpe'] = args.min_sharpe
    if args.min_win_rate is not None:
        filter_kwargs['min_win_rate'] = args.min_win_rate
    if args.max_volatility is not None:
        filter_kwargs['max_volatility'] = args.max_volatility

    # Filter
    if args.top_n:
        results_df = filter_sys.top_n(n=args.top_n, sort_by=args.sort_by, **filter_kwargs)
        print(f"\nTop {args.top_n} results (sorted by {args.sort_by}):\n")
        print(results_df[['symbol', 'strategy_name', 'period_name', 'total_return', 'sharpe_ratio', 'win_rate']].to_string(index=False))
    else:
        results_df = filter_sys.quick_filter(**filter_kwargs)
        print(f"\nFiltered {len(results_df)} results\n")
        print(results_df[['symbol', 'strategy_name', 'period_name', 'total_return', 'sharpe_ratio', 'win_rate']].to_string(index=False))

    # Export if requested
    if args.export:
        results_df.to_csv(args.export, index=False)
        print(f"\nExported to {args.export}")


if __name__ == '__main__':
    main()
