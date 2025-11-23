"""
Stock Deep Dive Analyzer

Provides comprehensive analysis of individual stocks across:
- All strategies (which works best?)
- All time periods (is performance consistent?)
- All metrics (return, risk, win rate, etc.)

Enables answering questions like:
- "How does AAPL perform with different strategies?"
- "Which strategy is most consistent for NVDA?"
- "Does MSFT work better with short or long time periods?"
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.statistical_analyzer import (
    StatisticalAnalyzer,
    RobustStatistics,
    StrategyRiskClass,
    STRATEGY_RISK_MAP
)


@dataclass
class StrategyPerformance:
    """Performance of a single strategy on a stock"""
    strategy_name: str
    risk_class: StrategyRiskClass

    # Aggregate metrics across all periods
    avg_return: float
    median_return: float
    best_return: float
    worst_return: float
    return_consistency: float  # 1/CV - higher is better

    avg_sharpe: float
    avg_win_rate: float
    avg_volatility: float
    avg_max_drawdown: float

    total_periods: int
    profitable_periods: int
    success_rate: float  # % of periods with positive return

    # Best period for this strategy
    best_period: str
    best_period_return: float
    best_period_sharpe: float

    # Confidence
    ci_lower: float
    ci_upper: float
    reliability: str


@dataclass
class PeriodPerformance:
    """Performance across all strategies for a specific period"""
    period_name: str

    # Best strategy for this period
    best_strategy: str
    best_return: float
    best_sharpe: float

    # Average across all strategies
    avg_return: float
    median_return: float
    return_range: float  # max - min

    total_strategies: int
    profitable_strategies: int


@dataclass
class StockAnalysis:
    """Complete analysis of a single stock"""
    symbol: str
    total_backtests: int
    total_strategies: int
    total_periods: int

    # Overall performance
    overall_avg_return: float
    overall_median_return: float
    overall_best_return: float
    overall_worst_return: float

    # Best combination
    best_strategy: str
    best_period: str
    best_combination_return: float
    best_combination_sharpe: float

    # Most consistent strategy
    most_consistent_strategy: str
    most_consistent_cv: float

    # Strategy rankings
    strategy_rankings: List[StrategyPerformance]

    # Period analysis
    period_analysis: List[PeriodPerformance]

    # Risk profile
    avg_volatility: float
    avg_sharpe: float
    avg_max_drawdown: float

    # Data quality
    data_quality_score: float
    reliability_rating: str


class StockAnalyzer:
    """
    Analyze individual stocks in detail
    """

    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize analyzer with backtest results

        Args:
            results_df: DataFrame with columns:
                - symbol, strategy_name, period_name
                - total_return, sharpe_ratio, win_rate, volatility, max_drawdown
                - total_trades, etc.
        """
        self.df = results_df
        self.stat_analyzer = StatisticalAnalyzer(min_sample_size=3)

    @classmethod
    def from_csv(cls, csv_path: str) -> 'StockAnalyzer':
        """Load from CSV file"""
        df = pd.read_csv(csv_path)
        return cls(df)

    @classmethod
    def from_timestamp(cls, timestamp: str, results_dir: str = 'backtest_results') -> 'StockAnalyzer':
        """Load from timestamp"""
        csv_path = Path(results_dir) / f'detailed_results_{timestamp}.csv'
        return cls.from_csv(str(csv_path))

    def analyze_stock(self, symbol: str, min_trades: int = 5) -> Optional[StockAnalysis]:
        """
        Perform deep dive analysis on a single stock

        Args:
            symbol: Stock ticker symbol
            min_trades: Minimum trades required for valid result

        Returns:
            StockAnalysis object or None if insufficient data
        """
        # Filter for this symbol
        stock_df = self.df[self.df['symbol'] == symbol].copy()

        if len(stock_df) == 0:
            return None

        # Filter by minimum trades
        if 'total_trades' in stock_df.columns:
            stock_df = stock_df[stock_df['total_trades'] >= min_trades]

        if len(stock_df) == 0:
            return None

        # Calculate overall statistics
        overall_stats = self.stat_analyzer.calculate_robust_stats(stock_df['total_return'])

        # Analyze by strategy
        strategy_rankings = self._analyze_by_strategy(stock_df)

        # Analyze by period
        period_analysis = self._analyze_by_period(stock_df)

        # Find best combination
        best_idx = stock_df['total_return'].idxmax()
        best_row = stock_df.loc[best_idx]

        # Find most consistent strategy
        strategy_cv = stock_df.groupby('strategy_name')['total_return'].apply(
            lambda x: x.std() / abs(x.mean()) if abs(x.mean()) > 0.001 and len(x) > 1 else 999
        )
        most_consistent = strategy_cv.idxmin() if len(strategy_cv) > 0 else "N/A"
        most_consistent_cv = strategy_cv.min() if len(strategy_cv) > 0 else 0.0

        # Calculate data quality
        quality_score = self._calculate_quality_score(stock_df)
        reliability = self._get_reliability_rating(quality_score, len(stock_df))

        return StockAnalysis(
            symbol=symbol,
            total_backtests=len(stock_df),
            total_strategies=stock_df['strategy_name'].nunique(),
            total_periods=stock_df['period_name'].nunique(),

            overall_avg_return=overall_stats.mean,
            overall_median_return=overall_stats.median,
            overall_best_return=overall_stats.max,
            overall_worst_return=overall_stats.min,

            best_strategy=best_row['strategy_name'],
            best_period=best_row['period_name'],
            best_combination_return=best_row['total_return'],
            best_combination_sharpe=best_row.get('sharpe_ratio', 0.0),

            most_consistent_strategy=most_consistent,
            most_consistent_cv=most_consistent_cv,

            strategy_rankings=strategy_rankings,
            period_analysis=period_analysis,

            avg_volatility=stock_df['volatility'].mean() if 'volatility' in stock_df.columns else 0.0,
            avg_sharpe=stock_df['sharpe_ratio'].mean() if 'sharpe_ratio' in stock_df.columns else 0.0,
            avg_max_drawdown=stock_df['max_drawdown'].mean() if 'max_drawdown' in stock_df.columns else 0.0,

            data_quality_score=quality_score,
            reliability_rating=reliability
        )

    def _analyze_by_strategy(self, stock_df: pd.DataFrame) -> List[StrategyPerformance]:
        """Analyze performance by strategy"""
        strategy_perfs = []

        for strategy_name, strategy_data in stock_df.groupby('strategy_name'):
            stats = self.stat_analyzer.calculate_robust_stats(strategy_data['total_return'])

            # Find best period for this strategy
            best_idx = strategy_data['total_return'].idxmax()
            best_period_data = strategy_data.loc[best_idx]

            # Calculate success rate
            profitable_periods = (strategy_data['total_return'] > 0).sum()
            success_rate = profitable_periods / len(strategy_data) * 100

            # Get risk class
            risk_class = STRATEGY_RISK_MAP.get(strategy_name, StrategyRiskClass.MODERATE)

            strategy_perfs.append(StrategyPerformance(
                strategy_name=strategy_name,
                risk_class=risk_class,

                avg_return=stats.mean,
                median_return=stats.median,
                best_return=stats.max,
                worst_return=stats.min,
                return_consistency=1 / (1 + stats.cv) if stats.cv > 0 else 1.0,

                avg_sharpe=strategy_data['sharpe_ratio'].mean() if 'sharpe_ratio' in strategy_data.columns else 0.0,
                avg_win_rate=strategy_data['win_rate'].mean() if 'win_rate' in strategy_data.columns else 0.0,
                avg_volatility=strategy_data['volatility'].mean() if 'volatility' in strategy_data.columns else 0.0,
                avg_max_drawdown=strategy_data['max_drawdown'].mean() if 'max_drawdown' in strategy_data.columns else 0.0,

                total_periods=len(strategy_data),
                profitable_periods=profitable_periods,
                success_rate=success_rate,

                best_period=best_period_data['period_name'],
                best_period_return=best_period_data['total_return'],
                best_period_sharpe=best_period_data.get('sharpe_ratio', 0.0),

                ci_lower=stats.ci_lower,
                ci_upper=stats.ci_upper,
                reliability='High' if len(strategy_data) >= 5 else 'Medium' if len(strategy_data) >= 3 else 'Low'
            ))

        # Sort by average return
        strategy_perfs.sort(key=lambda x: x.avg_return, reverse=True)

        return strategy_perfs

    def _analyze_by_period(self, stock_df: pd.DataFrame) -> List[PeriodPerformance]:
        """Analyze performance by time period"""
        period_perfs = []

        for period_name, period_data in stock_df.groupby('period_name'):
            # Find best strategy for this period
            best_idx = period_data['total_return'].idxmax()
            best_strategy_data = period_data.loc[best_idx]

            # Calculate statistics
            profitable_strategies = (period_data['total_return'] > 0).sum()

            period_perfs.append(PeriodPerformance(
                period_name=period_name,

                best_strategy=best_strategy_data['strategy_name'],
                best_return=best_strategy_data['total_return'],
                best_sharpe=best_strategy_data.get('sharpe_ratio', 0.0),

                avg_return=period_data['total_return'].mean(),
                median_return=period_data['total_return'].median(),
                return_range=period_data['total_return'].max() - period_data['total_return'].min(),

                total_strategies=len(period_data),
                profitable_strategies=profitable_strategies
            ))

        # Sort by average return
        period_perfs.sort(key=lambda x: x.avg_return, reverse=True)

        return period_perfs

    def _calculate_quality_score(self, stock_df: pd.DataFrame) -> float:
        """Calculate data quality score for this stock"""
        score = 0.0

        # Number of strategies tested (0-25 points)
        num_strategies = stock_df['strategy_name'].nunique()
        score += min(25, num_strategies * 2.5)

        # Number of periods tested (0-25 points)
        num_periods = stock_df['period_name'].nunique()
        score += min(25, num_periods * 5)

        # Total backtests (0-25 points)
        total_tests = len(stock_df)
        score += min(25, total_tests / 2)

        # Consistency across strategies (0-25 points)
        if 'total_return' in stock_df.columns:
            returns = stock_df['total_return']
            cv = returns.std() / abs(returns.mean()) if abs(returns.mean()) > 0.001 else 10
            consistency_score = max(0, 25 - cv * 5)
            score += consistency_score

        return min(100, score)

    def _get_reliability_rating(self, quality_score: float, num_tests: int) -> str:
        """Convert quality score to reliability rating"""
        if quality_score >= 70 and num_tests >= 15:
            return "High"
        elif quality_score >= 50 and num_tests >= 8:
            return "Medium"
        else:
            return "Low"

    def compare_stocks(
        self,
        symbols: List[str],
        metric: str = 'avg_return',
        min_trades: int = 5
    ) -> pd.DataFrame:
        """
        Compare multiple stocks

        Args:
            symbols: List of ticker symbols
            metric: Metric to compare (avg_return, avg_sharpe, etc.)
            min_trades: Minimum trades for validity

        Returns:
            DataFrame with comparison
        """
        results = []

        for symbol in symbols:
            analysis = self.analyze_stock(symbol, min_trades)
            if analysis:
                results.append({
                    'symbol': symbol,
                    'avg_return': analysis.overall_avg_return,
                    'median_return': analysis.overall_median_return,
                    'best_return': analysis.overall_best_return,
                    'avg_sharpe': analysis.avg_sharpe,
                    'avg_volatility': analysis.avg_volatility,
                    'best_strategy': analysis.best_strategy,
                    'most_consistent_strategy': analysis.most_consistent_strategy,
                    'total_backtests': analysis.total_backtests,
                    'reliability': analysis.reliability_rating
                })

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values(metric, ascending=False)

        return df

    def get_stock_summary(self, symbol: str, min_trades: int = 5) -> str:
        """
        Generate a text summary for a stock

        Args:
            symbol: Stock ticker
            min_trades: Minimum trades

        Returns:
            Formatted text summary
        """
        analysis = self.analyze_stock(symbol, min_trades)

        if not analysis:
            return f"No data available for {symbol} (or insufficient trades)"

        summary = f"""
{'='*70}
STOCK DEEP DIVE: {analysis.symbol}
{'='*70}

OVERVIEW:
  Reliability:     {analysis.reliability_rating} (Quality Score: {analysis.data_quality_score:.1f}/100)
  Total Tests:     {analysis.total_backtests} ({analysis.total_strategies} strategies × {analysis.total_periods} periods)

PERFORMANCE:
  Average Return:  {analysis.overall_avg_return:>8.2f}%
  Median Return:   {analysis.overall_median_return:>8.2f}%
  Best Return:     {analysis.overall_best_return:>8.2f}%
  Worst Return:    {analysis.overall_worst_return:>8.2f}%

RISK METRICS:
  Avg Sharpe Ratio:  {analysis.avg_sharpe:>6.2f}
  Avg Volatility:    {analysis.avg_volatility:>6.2f}%
  Avg Max Drawdown:  {analysis.avg_max_drawdown:>6.2f}%

BEST COMBINATION:
  Strategy:        {analysis.best_strategy}
  Period:          {analysis.best_period}
  Return:          {analysis.best_combination_return:.2f}%
  Sharpe Ratio:    {analysis.best_combination_sharpe:.2f}

MOST CONSISTENT:
  Strategy:        {analysis.most_consistent_strategy}
  CV (lower=better): {analysis.most_consistent_cv:.3f}

{'─'*70}
TOP 5 STRATEGIES (by Average Return):
{'─'*70}
"""

        for i, strat in enumerate(analysis.strategy_rankings[:5], 1):
            summary += f"""
{i}. {strat.strategy_name} ({strat.risk_class.value})
   Avg Return:    {strat.avg_return:>8.2f}% (Median: {strat.median_return:.2f}%)
   Sharpe Ratio:  {strat.avg_sharpe:>8.2f}
   Win Rate:      {strat.avg_win_rate:>8.1f}%
   Success Rate:  {strat.success_rate:>8.1f}% ({strat.profitable_periods}/{strat.total_periods} periods)
   Consistency:   {strat.return_consistency:>8.3f}
   Best Period:   {strat.best_period} ({strat.best_period_return:+.2f}%)
   95% CI:        [{strat.ci_lower:.2f}%, {strat.ci_upper:.2f}%]
   Reliability:   {strat.reliability}
"""

        summary += f"\n{'─'*70}\n"
        summary += "PERIOD ANALYSIS:\n"
        summary += f"{'─'*70}\n"

        for period in analysis.period_analysis:
            summary += f"""
{period.period_name}:
   Best Strategy:  {period.best_strategy} ({period.best_return:+.2f}%)
   Avg Return:     {period.avg_return:>8.2f}%
   Profitable:     {period.profitable_strategies}/{period.total_strategies} strategies
   Return Range:   {period.return_range:.2f}%
"""

        summary += f"\n{'='*70}\n"

        return summary


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze individual stocks')
    parser.add_argument('symbol', type=str, help='Stock ticker symbol')
    parser.add_argument('--timestamp', type=str, help='Backtest timestamp (e.g., 20251122_203908)')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--min-trades', type=int, default=5, help='Minimum trades (default: 5)')
    parser.add_argument('--compare', type=str, nargs='+', help='Compare multiple stocks')

    args = parser.parse_args()

    # Load data
    if args.csv:
        analyzer = StockAnalyzer.from_csv(args.csv)
    elif args.timestamp:
        analyzer = StockAnalyzer.from_timestamp(args.timestamp)
    else:
        # Find latest results
        results_dir = Path('backtest_results')
        csv_files = list(results_dir.glob('detailed_results_*.csv'))
        if not csv_files:
            print("No backtest results found!")
            return
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"Using latest results: {latest_csv.name}")
        analyzer = StockAnalyzer.from_csv(str(latest_csv))

    # Analyze
    if args.compare:
        print(f"\nComparing stocks: {', '.join(args.compare)}\n")
        comparison = analyzer.compare_stocks(args.compare, min_trades=args.min_trades)
        print(comparison.to_string(index=False))
    else:
        summary = analyzer.get_stock_summary(args.symbol, min_trades=args.min_trades)
        print(summary)


if __name__ == '__main__':
    main()
