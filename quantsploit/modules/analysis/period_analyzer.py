"""
Time Period Analysis Module

Analyzes strategy performance across different time horizons to identify:
- Which strategies work best for short vs long periods
- Whether performance is consistent across time
- Optimal time horizon for each strategy/stock
- Regime-dependent performance

Answers questions like:
- "Does SMA work better for 1-year or 2-year periods?"
- "Is Kalman consistent across all time frames?"
- "What's the optimal period for trading AAPL?"
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import re

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.statistical_analyzer import (
    StatisticalAnalyzer,
    RobustStatistics
)


@dataclass
class PeriodMetrics:
    """Performance metrics for a specific time period"""
    period_name: str
    period_length_days: int  # Estimated
    period_category: str  # "short" (<6mo), "medium" (6mo-1yr), "long" (>1yr)

    mean_return: float
    median_return: float
    annualized_return: float

    mean_sharpe: float
    mean_volatility: float
    mean_win_rate: float

    num_tests: int
    success_rate: float  # % profitable

    best_strategy: str
    best_strategy_return: float

    consistency_score: float
    reliability: str


@dataclass
class StrategyPeriodProfile:
    """How a strategy performs across different time periods"""
    strategy_name: str

    # Performance by period length
    short_term_return: float  # <6 months
    medium_term_return: float  # 6mo-1yr
    long_term_return: float   # >1yr

    # Optimal period
    optimal_period: str
    optimal_period_return: float

    # Consistency across periods
    period_consistency: float  # Low CV = consistent across periods
    is_period_sensitive: bool  # Large performance difference

    # Detailed period breakdown
    period_metrics: List[PeriodMetrics]


@dataclass
class StockPeriodProfile:
    """How a stock performs across different time periods"""
    symbol: str

    # Best period for this stock
    optimal_period: str
    optimal_period_return: float
    optimal_period_strategy: str

    # Consistency across periods
    period_consistency: float

    # Period breakdown
    period_metrics: List[PeriodMetrics]


class PeriodAnalyzer:
    """
    Analyze performance across different time periods
    """

    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize analyzer

        Args:
            results_df: DataFrame with backtest results (must include 'period_name' column)
        """
        self.df = results_df
        self.stat_analyzer = StatisticalAnalyzer(min_sample_size=3)

        # Categorize periods
        self.df['period_category'] = self.df['period_name'].apply(self._categorize_period)
        self.df['period_length'] = self.df['period_name'].apply(self._estimate_period_length)

    @classmethod
    def from_csv(cls, csv_path: str) -> 'PeriodAnalyzer':
        """Load from CSV file"""
        df = pd.read_csv(csv_path)
        return cls(df)

    @classmethod
    def from_timestamp(cls, timestamp: str, results_dir: str = 'backtest_results') -> 'PeriodAnalyzer':
        """Load from timestamp"""
        csv_path = Path(results_dir) / f'detailed_results_{timestamp}.csv'
        return cls.from_csv(str(csv_path))

    def analyze_strategy_by_period(
        self,
        strategy: str,
        min_trades: int = 5
    ) -> Optional[StrategyPeriodProfile]:
        """
        Analyze how a strategy performs across different time periods

        Args:
            strategy: Strategy name
            min_trades: Minimum trades per result

        Returns:
            StrategyPeriodProfile or None
        """
        # Filter for this strategy
        strategy_df = self.df[self.df['strategy_name'] == strategy].copy()

        if len(strategy_df) == 0:
            return None

        # Filter by minimum trades
        if 'total_trades' in strategy_df.columns:
            strategy_df = strategy_df[strategy_df['total_trades'] >= min_trades]

        if len(strategy_df) < 3:
            return None

        # Calculate average return by period category
        short_term = strategy_df[strategy_df['period_category'] == 'short']['total_return']
        medium_term = strategy_df[strategy_df['period_category'] == 'medium']['total_return']
        long_term = strategy_df[strategy_df['period_category'] == 'long']['total_return']

        short_term_return = short_term.mean() if len(short_term) > 0 else 0.0
        medium_term_return = medium_term.mean() if len(medium_term) > 0 else 0.0
        long_term_return = long_term.mean() if len(long_term) > 0 else 0.0

        # Find optimal period
        period_returns = strategy_df.groupby('period_name')['total_return'].mean()
        if len(period_returns) > 0:
            optimal_period = period_returns.idxmax()
            optimal_period_return = period_returns.max()
        else:
            optimal_period = "N/A"
            optimal_period_return = 0.0

        # Calculate consistency across periods
        period_means = strategy_df.groupby('period_name')['total_return'].mean()
        if len(period_means) > 1:
            period_consistency = 1 / (1 + (period_means.std() / abs(period_means.mean()))) if abs(period_means.mean()) > 0.001 else 0.0
        else:
            period_consistency = 1.0

        # Check if period-sensitive (large performance difference)
        if len(period_means) > 1:
            max_diff = period_means.max() - period_means.min()
            is_period_sensitive = max_diff > 10.0  # >10% difference
        else:
            is_period_sensitive = False

        # Detailed period metrics
        period_metrics = self._calculate_period_metrics(strategy_df)

        return StrategyPeriodProfile(
            strategy_name=strategy,
            short_term_return=short_term_return,
            medium_term_return=medium_term_return,
            long_term_return=long_term_return,
            optimal_period=optimal_period,
            optimal_period_return=optimal_period_return,
            period_consistency=period_consistency,
            is_period_sensitive=is_period_sensitive,
            period_metrics=period_metrics
        )

    def analyze_stock_by_period(
        self,
        symbol: str,
        min_trades: int = 5
    ) -> Optional[StockPeriodProfile]:
        """
        Analyze how a stock performs across different time periods

        Args:
            symbol: Stock ticker
            min_trades: Minimum trades

        Returns:
            StockPeriodProfile or None
        """
        # Filter for this stock
        stock_df = self.df[self.df['symbol'] == symbol].copy()

        if len(stock_df) == 0:
            return None

        # Filter by minimum trades
        if 'total_trades' in stock_df.columns:
            stock_df = stock_df[stock_df['total_trades'] >= min_trades]

        if len(stock_df) < 3:
            return None

        # Find optimal period (across all strategies)
        period_returns = stock_df.groupby('period_name')['total_return'].mean()
        if len(period_returns) > 0:
            optimal_period = period_returns.idxmax()
            optimal_period_return = period_returns.max()

            # Find best strategy for optimal period
            optimal_period_df = stock_df[stock_df['period_name'] == optimal_period]
            best_idx = optimal_period_df['total_return'].idxmax()
            optimal_period_strategy = optimal_period_df.loc[best_idx, 'strategy_name']
        else:
            optimal_period = "N/A"
            optimal_period_return = 0.0
            optimal_period_strategy = "N/A"

        # Calculate consistency across periods
        period_means = stock_df.groupby('period_name')['total_return'].mean()
        if len(period_means) > 1:
            period_consistency = 1 / (1 + (period_means.std() / abs(period_means.mean()))) if abs(period_means.mean()) > 0.001 else 0.0
        else:
            period_consistency = 1.0

        # Detailed period metrics
        period_metrics = self._calculate_period_metrics(stock_df)

        return StockPeriodProfile(
            symbol=symbol,
            optimal_period=optimal_period,
            optimal_period_return=optimal_period_return,
            optimal_period_strategy=optimal_period_strategy,
            period_consistency=period_consistency,
            period_metrics=period_metrics
        )

    def compare_periods(
        self,
        min_trades: int = 5
    ) -> List[PeriodMetrics]:
        """
        Compare all time periods

        Args:
            min_trades: Minimum trades

        Returns:
            List of PeriodMetrics sorted by return
        """
        # Filter by minimum trades
        df = self.df.copy()
        if 'total_trades' in df.columns:
            df = df[df['total_trades'] >= min_trades]

        period_metrics = self._calculate_period_metrics(df)

        # Sort by mean return
        period_metrics.sort(key=lambda x: x.mean_return, reverse=True)

        return period_metrics

    def find_optimal_period_by_strategy(
        self,
        min_trades: int = 5
    ) -> pd.DataFrame:
        """
        Find optimal period for each strategy

        Args:
            min_trades: Minimum trades

        Returns:
            DataFrame with strategy, optimal_period, return
        """
        results = []

        for strategy in self.df['strategy_name'].unique():
            profile = self.analyze_strategy_by_period(strategy, min_trades)

            if profile:
                results.append({
                    'strategy': strategy,
                    'optimal_period': profile.optimal_period,
                    'optimal_return': profile.optimal_period_return,
                    'short_term': profile.short_term_return,
                    'medium_term': profile.medium_term_return,
                    'long_term': profile.long_term_return,
                    'consistency': profile.period_consistency,
                    'period_sensitive': profile.is_period_sensitive
                })

        df = pd.DataFrame(results)
        df = df.sort_values('optimal_return', ascending=False)

        return df

    def _calculate_period_metrics(self, df: pd.DataFrame) -> List[PeriodMetrics]:
        """Calculate metrics for each period"""
        metrics = []

        for period_name, period_data in df.groupby('period_name'):
            if len(period_data) == 0:
                continue

            # Calculate statistics
            stats = self.stat_analyzer.calculate_robust_stats(period_data['total_return'])

            # Annualize return
            period_length = self._estimate_period_length(period_name)
            if period_length > 0:
                annualization_factor = 365.0 / period_length
                annualized_return = ((1 + stats.mean / 100) ** annualization_factor - 1) * 100
            else:
                annualized_return = stats.mean

            # Success rate
            success_rate = (period_data['total_return'] > 0).sum() / len(period_data) * 100

            # Find best strategy for this period
            best_idx = period_data['total_return'].idxmax()
            best_strategy = period_data.loc[best_idx, 'strategy_name']
            best_strategy_return = period_data.loc[best_idx, 'total_return']

            # Consistency
            consistency = 1 / (1 + stats.cv) if stats.cv > 0 else 1.0

            metrics.append(PeriodMetrics(
                period_name=period_name,
                period_length_days=period_length,
                period_category=self._categorize_period(period_name),

                mean_return=stats.mean,
                median_return=stats.median,
                annualized_return=annualized_return,

                mean_sharpe=period_data['sharpe_ratio'].mean() if 'sharpe_ratio' in period_data.columns else 0.0,
                mean_volatility=period_data['volatility'].mean() if 'volatility' in period_data.columns else 0.0,
                mean_win_rate=period_data['win_rate'].mean() if 'win_rate' in period_data.columns else 0.0,

                num_tests=len(period_data),
                success_rate=success_rate,

                best_strategy=best_strategy,
                best_strategy_return=best_strategy_return,

                consistency_score=consistency,
                reliability='High' if len(period_data) >= 20 else 'Medium' if len(period_data) >= 10 else 'Low'
            ))

        return metrics

    def _categorize_period(self, period_name: str) -> str:
        """Categorize period as short/medium/long"""
        period_lower = period_name.lower()

        # Check for explicit period lengths
        if '1yr' in period_lower or '1 yr' in period_lower or '12mo' in period_lower or '12 mo' in period_lower:
            return 'medium'
        elif '2yr' in period_lower or '2 yr' in period_lower or '24mo' in period_lower:
            return 'long'
        elif '3yr' in period_lower or '3 yr' in period_lower or '36mo' in period_lower:
            return 'long'
        elif '6mo' in period_lower or '6 mo' in period_lower:
            return 'medium'
        elif '3mo' in period_lower or '3 mo' in period_lower:
            return 'short'
        elif 'quarter' in period_lower or 'q1' in period_lower or 'q2' in period_lower or 'q3' in period_lower or 'q4' in period_lower:
            return 'short'
        elif 'rolling' in period_lower:
            # Try to extract month count
            match = re.search(r'(\d+)\s*mo', period_lower)
            if match:
                months = int(match.group(1))
                if months < 6:
                    return 'short'
                elif months <= 12:
                    return 'medium'
                else:
                    return 'long'
            return 'medium'  # Default for rolling
        else:
            return 'medium'  # Default

    def _estimate_period_length(self, period_name: str) -> int:
        """Estimate period length in days"""
        period_lower = period_name.lower()

        # Extract year/month information
        if '3yr' in period_lower:
            return 365 * 3
        elif '2yr' in period_lower:
            return 365 * 2
        elif '1yr' in period_lower:
            return 365
        elif '36mo' in period_lower:
            return 365 * 3
        elif '24mo' in period_lower:
            return 365 * 2
        elif '12mo' in period_lower:
            return 365
        elif '6mo' in period_lower:
            return 182
        elif '3mo' in period_lower or 'quarter' in period_lower:
            return 91
        elif 'rolling' in period_lower:
            match = re.search(r'(\d+)\s*mo', period_lower)
            if match:
                months = int(match.group(1))
                return int(months * 30.5)
            return 182  # Default 6 months
        else:
            return 365  # Default 1 year

    def format_strategy_period_profile(self, profile: StrategyPeriodProfile) -> str:
        """Format strategy period profile as text"""
        report = f"""
{'='*70}
STRATEGY PERIOD ANALYSIS: {profile.strategy_name}
{'='*70}

PERFORMANCE BY TIME HORIZON:
  Short-term (<6mo):   {profile.short_term_return:>8.2f}%
  Medium-term (6-12mo):{profile.medium_term_return:>8.2f}%
  Long-term (>1yr):    {profile.long_term_return:>8.2f}%

OPTIMAL PERIOD:
  Period:              {profile.optimal_period}
  Return:              {profile.optimal_period_return:>8.2f}%

CONSISTENCY:
  Across Periods:      {profile.period_consistency:>8.3f} (1.0 = perfect consistency)
  Period Sensitive:    {'Yes - >10% variation' if profile.is_period_sensitive else 'No - stable across periods'}

{'─'*70}
DETAILED PERIOD BREAKDOWN:
{'─'*70}
"""

        for period in sorted(profile.period_metrics, key=lambda x: x.mean_return, reverse=True):
            report += f"""
{period.period_name} ({period.period_category}, ~{period.period_length_days}d):
  Mean Return:       {period.mean_return:>8.2f}%
  Annualized:        {period.annualized_return:>8.2f}%
  Sharpe Ratio:      {period.mean_sharpe:>8.2f}
  Success Rate:      {period.success_rate:>8.1f}%
  Tests:             {period.num_tests}
  Reliability:       {period.reliability}
"""

        report += f"\n{'='*70}\n"

        return report

    def format_stock_period_profile(self, profile: StockPeriodProfile) -> str:
        """Format stock period profile as text"""
        report = f"""
{'='*70}
STOCK PERIOD ANALYSIS: {profile.symbol}
{'='*70}

OPTIMAL CONFIGURATION:
  Best Period:         {profile.optimal_period}
  Return:              {profile.optimal_period_return:>8.2f}%
  Best Strategy:       {profile.optimal_period_strategy}

CONSISTENCY:
  Across Periods:      {profile.period_consistency:>8.3f}

{'─'*70}
PERIOD BREAKDOWN:
{'─'*70}
"""

        for period in sorted(profile.period_metrics, key=lambda x: x.mean_return, reverse=True):
            report += f"""
{period.period_name} ({period.period_category}):
  Mean Return:       {period.mean_return:>8.2f}%
  Best Strategy:     {period.best_strategy} ({period.best_strategy_return:+.2f}%)
  Win Rate:          {period.mean_win_rate:>8.1f}%
  Success Rate:      {period.success_rate:>8.1f}%
  Consistency:       {period.consistency_score:>8.3f}
"""

        report += f"\n{'='*70}\n"

        return report


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze performance across time periods')
    parser.add_argument('--strategy', type=str, help='Analyze specific strategy')
    parser.add_argument('--stock', type=str, help='Analyze specific stock')
    parser.add_argument('--timestamp', type=str, help='Backtest timestamp')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--min-trades', type=int, default=5, help='Minimum trades (default: 5)')
    parser.add_argument('--compare-all', action='store_true', help='Compare all periods')
    parser.add_argument('--optimal', action='store_true', help='Show optimal period for each strategy')

    args = parser.parse_args()

    # Load data
    if args.csv:
        analyzer = PeriodAnalyzer.from_csv(args.csv)
    elif args.timestamp:
        analyzer = PeriodAnalyzer.from_timestamp(args.timestamp)
    else:
        # Find latest
        results_dir = Path('backtest_results')
        csv_files = list(results_dir.glob('detailed_results_*.csv'))
        if not csv_files:
            print("No backtest results found!")
            return
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"Using latest results: {latest_csv.name}\n")
        analyzer = PeriodAnalyzer.from_csv(str(latest_csv))

    # Analyze
    if args.strategy:
        profile = analyzer.analyze_strategy_by_period(args.strategy, min_trades=args.min_trades)
        if profile:
            print(analyzer.format_strategy_period_profile(profile))
        else:
            print(f"No data available for strategy: {args.strategy}")

    elif args.stock:
        profile = analyzer.analyze_stock_by_period(args.stock, min_trades=args.min_trades)
        if profile:
            print(analyzer.format_stock_period_profile(profile))
        else:
            print(f"No data available for stock: {args.stock}")

    elif args.optimal:
        df = analyzer.find_optimal_period_by_strategy(min_trades=args.min_trades)
        print("\nOptimal Periods by Strategy:\n")
        print(df.to_string(index=False))

    elif args.compare_all:
        periods = analyzer.compare_periods(min_trades=args.min_trades)
        print(f"\n{'='*70}")
        print("PERIOD COMPARISON")
        print(f"{'='*70}\n")
        for period in periods:
            print(f"{period.period_name} ({period.period_category}):")
            print(f"  Mean Return:     {period.mean_return:>8.2f}%")
            print(f"  Annualized:      {period.annualized_return:>8.2f}%")
            print(f"  Best Strategy:   {period.best_strategy} ({period.best_strategy_return:+.2f}%)")
            print(f"  Tests:           {period.num_tests}")
            print()

    else:
        print("Please specify --strategy, --stock, --compare-all, or --optimal")


if __name__ == '__main__':
    main()
