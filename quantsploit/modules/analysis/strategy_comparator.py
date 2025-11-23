"""
Strategy Comparison Engine

Enables head-to-head comparison of trading strategies across:
- Specific stocks
- Specific sectors
- Specific time periods
- Multiple metrics simultaneously

Answers questions like:
- "Which strategy is better for AAPL: SMA or Kalman?"
- "Does Momentum beat Mean Reversion for tech stocks?"
- "Is Multi-Factor more consistent than HMM over 1-year periods?"
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.statistical_analyzer import (
    StatisticalAnalyzer,
    RobustStatistics,
    StrategyRiskClass,
    STRATEGY_RISK_MAP
)


@dataclass
class StrategyMetrics:
    """Comprehensive metrics for a strategy"""
    strategy_name: str
    risk_class: StrategyRiskClass

    # Performance
    mean_return: float
    median_return: float
    total_return_range: Tuple[float, float]  # (min, max)
    ci_95: Tuple[float, float]

    # Risk
    mean_sharpe: float
    mean_volatility: float
    mean_max_drawdown: float
    mean_sortino: float

    # Trading
    mean_win_rate: float
    mean_profit_factor: float
    avg_trades_per_period: float

    # Consistency
    return_std: float
    return_cv: float
    consistency_score: float  # 0-100

    # Success metrics
    num_tests: int
    profitable_tests: int
    success_rate: float

    # Statistical significance
    reliability: str


@dataclass
class ComparisonResult:
    """Result of comparing two strategies"""
    strategy1: str
    strategy2: str

    # Performance comparison
    return_diff: float  # strategy1 - strategy2
    sharpe_diff: float
    win_rate_diff: float

    # Winner determination
    winner: str  # Which strategy is better overall
    confidence: str  # "High", "Medium", "Low"

    # Statistical test
    test_statistic: float
    p_value: float
    is_significant: bool  # p < 0.05
    interpretation: str

    # Detailed breakdown
    strategy1_better_count: int
    strategy2_better_count: int
    tie_count: int

    # Risk-adjusted comparison
    risk_adjusted_winner: str  # Based on Sharpe ratio


@dataclass
class MultiStrategyComparison:
    """Comparison of 3+ strategies"""
    strategies: List[str]
    num_comparisons: int

    # Rankings by metric
    rankings_by_return: List[Tuple[str, float]]
    rankings_by_sharpe: List[Tuple[str, float]]
    rankings_by_consistency: List[Tuple[str, float]]
    rankings_by_win_rate: List[Tuple[str, float]]

    # Overall winner
    overall_winner: str
    overall_score: float

    # Pairwise comparisons
    pairwise_results: List[ComparisonResult]

    # Summary statistics
    strategy_metrics: Dict[str, StrategyMetrics]


class StrategyComparator:
    """
    Compare trading strategies head-to-head
    """

    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize comparator

        Args:
            results_df: DataFrame with backtest results
        """
        self.df = results_df
        self.stat_analyzer = StatisticalAnalyzer(min_sample_size=3)

    @classmethod
    def from_csv(cls, csv_path: str) -> 'StrategyComparator':
        """Load from CSV file"""
        df = pd.read_csv(csv_path)
        return cls(df)

    @classmethod
    def from_timestamp(cls, timestamp: str, results_dir: str = 'backtest_results') -> 'StrategyComparator':
        """Load from timestamp"""
        csv_path = Path(results_dir) / f'detailed_results_{timestamp}.csv'
        return cls.from_csv(str(csv_path))

    def compare_two_strategies(
        self,
        strategy1: str,
        strategy2: str,
        stock: Optional[str] = None,
        period: Optional[str] = None,
        sector: Optional[str] = None
    ) -> Optional[ComparisonResult]:
        """
        Compare two strategies head-to-head

        Args:
            strategy1: First strategy name
            strategy2: Second strategy name
            stock: Optional stock symbol to filter
            period: Optional period to filter
            sector: Optional sector to filter

        Returns:
            ComparisonResult or None if insufficient data
        """
        # Filter data
        df1 = self._filter_data(strategy1, stock, period, sector)
        df2 = self._filter_data(strategy2, stock, period, sector)

        if len(df1) < 3 or len(df2) < 3:
            return None

        # Get returns
        returns1 = df1['total_return']
        returns2 = df2['total_return']

        # Calculate metrics
        metrics1 = self._calculate_strategy_metrics(df1, strategy1)
        metrics2 = self._calculate_strategy_metrics(df2, strategy2)

        # Performance differences
        return_diff = metrics1.mean_return - metrics2.mean_return
        sharpe_diff = metrics1.mean_sharpe - metrics2.mean_sharpe
        win_rate_diff = metrics1.mean_win_rate - metrics2.mean_win_rate

        # Statistical test (Mann-Whitney U - non-parametric)
        test_stat, p_value, interp = self.stat_analyzer.compare_groups(
            returns1, returns2, test='mannwhitneyu'
        )

        is_significant = p_value < 0.05

        # Determine winner
        winner = strategy1 if return_diff > 0 else strategy2
        risk_adjusted_winner = strategy1 if sharpe_diff > 0 else strategy2

        # Confidence based on p-value and sample size
        min_samples = min(len(df1), len(df2))
        if is_significant and min_samples >= 10:
            confidence = "High"
        elif is_significant or min_samples >= 5:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Head-to-head breakdown (when both tested on same stock+period)
        common_tests = self._get_common_tests(df1, df2)
        s1_better = 0
        s2_better = 0
        ties = 0

        for _, row1 in common_tests['df1'].iterrows():
            key = (row1['symbol'], row1['period_name'])
            row2 = common_tests['df2'][
                (common_tests['df2']['symbol'] == key[0]) &
                (common_tests['df2']['period_name'] == key[1])
            ]
            if len(row2) > 0:
                r1 = row1['total_return']
                r2 = row2.iloc[0]['total_return']
                if abs(r1 - r2) < 0.01:  # Within 0.01% = tie
                    ties += 1
                elif r1 > r2:
                    s1_better += 1
                else:
                    s2_better += 1

        return ComparisonResult(
            strategy1=strategy1,
            strategy2=strategy2,
            return_diff=return_diff,
            sharpe_diff=sharpe_diff,
            win_rate_diff=win_rate_diff,
            winner=winner,
            confidence=confidence,
            test_statistic=test_stat,
            p_value=p_value,
            is_significant=is_significant,
            interpretation=interp,
            strategy1_better_count=s1_better,
            strategy2_better_count=s2_better,
            tie_count=ties,
            risk_adjusted_winner=risk_adjusted_winner
        )

    def compare_multiple_strategies(
        self,
        strategies: List[str],
        stock: Optional[str] = None,
        period: Optional[str] = None,
        sector: Optional[str] = None
    ) -> MultiStrategyComparison:
        """
        Compare 3+ strategies simultaneously

        Args:
            strategies: List of strategy names
            stock: Optional stock filter
            period: Optional period filter
            sector: Optional sector filter

        Returns:
            MultiStrategyComparison with rankings and pairwise comparisons
        """
        # Calculate metrics for each strategy
        strategy_metrics = {}
        for strategy in strategies:
            df = self._filter_data(strategy, stock, period, sector)
            if len(df) >= 3:
                strategy_metrics[strategy] = self._calculate_strategy_metrics(df, strategy)

        if len(strategy_metrics) < 2:
            # Not enough data
            return self._empty_multi_comparison(strategies)

        # Rank by different metrics
        rankings_return = sorted(
            [(s, m.mean_return) for s, m in strategy_metrics.items()],
            key=lambda x: x[1],
            reverse=True
        )

        rankings_sharpe = sorted(
            [(s, m.mean_sharpe) for s, m in strategy_metrics.items()],
            key=lambda x: x[1],
            reverse=True
        )

        rankings_consistency = sorted(
            [(s, m.consistency_score) for s, m in strategy_metrics.items()],
            key=lambda x: x[1],
            reverse=True
        )

        rankings_win_rate = sorted(
            [(s, m.mean_win_rate) for s, m in strategy_metrics.items()],
            key=lambda x: x[1],
            reverse=True
        )

        # Calculate overall score (weighted average of rankings)
        overall_scores = {}
        for strategy in strategy_metrics.keys():
            # Position in each ranking (1st = 1, 2nd = 2, etc.)
            return_rank = [s for s, _ in rankings_return].index(strategy) + 1
            sharpe_rank = [s for s, _ in rankings_sharpe].index(strategy) + 1
            consistency_rank = [s for s, _ in rankings_consistency].index(strategy) + 1
            win_rate_rank = [s for s, _ in rankings_win_rate].index(strategy) + 1

            # Weighted score (lower is better, like golf)
            # Return: 40%, Sharpe: 30%, Consistency: 20%, Win Rate: 10%
            score = (return_rank * 0.4 +
                    sharpe_rank * 0.3 +
                    consistency_rank * 0.2 +
                    win_rate_rank * 0.1)

            overall_scores[strategy] = score

        overall_winner = min(overall_scores.items(), key=lambda x: x[1])[0]
        overall_score = overall_scores[overall_winner]

        # Pairwise comparisons
        pairwise_results = []
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i+1:]:
                result = self.compare_two_strategies(s1, s2, stock, period, sector)
                if result:
                    pairwise_results.append(result)

        return MultiStrategyComparison(
            strategies=list(strategy_metrics.keys()),
            num_comparisons=len(pairwise_results),
            rankings_by_return=rankings_return,
            rankings_by_sharpe=rankings_sharpe,
            rankings_by_consistency=rankings_consistency,
            rankings_by_win_rate=rankings_win_rate,
            overall_winner=overall_winner,
            overall_score=overall_score,
            pairwise_results=pairwise_results,
            strategy_metrics=strategy_metrics
        )

    def compare_strategies_by_stock(
        self,
        strategies: List[str],
        min_stocks: int = 5
    ) -> pd.DataFrame:
        """
        Compare strategies across all stocks, showing which works best for each

        Args:
            strategies: List of strategies to compare
            min_stocks: Minimum number of stocks

        Returns:
            DataFrame with best strategy per stock
        """
        results = []

        for symbol in self.df['symbol'].unique():
            comparison = self.compare_multiple_strategies(strategies, stock=symbol)

            if len(comparison.strategies) >= 2:
                best_strategy = comparison.overall_winner
                best_return = comparison.strategy_metrics[best_strategy].mean_return
                best_sharpe = comparison.strategy_metrics[best_strategy].mean_sharpe

                results.append({
                    'symbol': symbol,
                    'best_strategy': best_strategy,
                    'return': best_return,
                    'sharpe': best_sharpe,
                    'num_strategies_tested': len(comparison.strategies)
                })

        if len(results) < min_stocks:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values('return', ascending=False)

        return df

    def _filter_data(
        self,
        strategy: str,
        stock: Optional[str] = None,
        period: Optional[str] = None,
        sector: Optional[str] = None
    ) -> pd.DataFrame:
        """Filter data by criteria"""
        df = self.df[self.df['strategy_name'] == strategy].copy()

        if stock:
            df = df[df['symbol'] == stock]

        if period:
            df = df[df['period_name'] == period]

        if sector:
            # Need to import sector classification
            from dashboard.ticker_universe import get_sector
            symbols_in_sector = [s for s in df['symbol'].unique() if get_sector(s) == sector]
            df = df[df['symbol'].isin(symbols_in_sector)]

        # Filter by minimum trades
        if 'total_trades' in df.columns:
            df = df[df['total_trades'] >= 3]

        return df

    def _calculate_strategy_metrics(self, df: pd.DataFrame, strategy_name: str) -> StrategyMetrics:
        """Calculate comprehensive metrics for a strategy"""
        if len(df) == 0:
            return self._empty_metrics(strategy_name)

        # Get statistics
        stats = self.stat_analyzer.calculate_robust_stats(df['total_return'])

        # Calculate consistency score (0-100, higher is better)
        # Based on CV, success rate, and reliability
        if stats.cv > 0:
            cv_score = max(0, 100 - stats.cv * 50)
        else:
            cv_score = 100

        success_rate = (df['total_return'] > 0).sum() / len(df) * 100
        success_score = success_rate

        reliability_score = min(100, len(df) * 5)

        consistency_score = (cv_score * 0.4 + success_score * 0.4 + reliability_score * 0.2)

        # Get risk class
        risk_class = STRATEGY_RISK_MAP.get(strategy_name, StrategyRiskClass.MODERATE)

        return StrategyMetrics(
            strategy_name=strategy_name,
            risk_class=risk_class,

            mean_return=stats.mean,
            median_return=stats.median,
            total_return_range=(stats.min, stats.max),
            ci_95=(stats.ci_lower, stats.ci_upper),

            mean_sharpe=df['sharpe_ratio'].mean() if 'sharpe_ratio' in df.columns else 0.0,
            mean_volatility=df['volatility'].mean() if 'volatility' in df.columns else 0.0,
            mean_max_drawdown=df['max_drawdown'].mean() if 'max_drawdown' in df.columns else 0.0,
            mean_sortino=df['sortino_ratio'].mean() if 'sortino_ratio' in df.columns else 0.0,

            mean_win_rate=df['win_rate'].mean() if 'win_rate' in df.columns else 0.0,
            mean_profit_factor=df['profit_factor'].mean() if 'profit_factor' in df.columns else 0.0,
            avg_trades_per_period=df['total_trades'].mean() if 'total_trades' in df.columns else 0.0,

            return_std=stats.std,
            return_cv=stats.cv,
            consistency_score=consistency_score,

            num_tests=len(df),
            profitable_tests=(df['total_return'] > 0).sum(),
            success_rate=success_rate,

            reliability='High' if len(df) >= 10 else 'Medium' if len(df) >= 5 else 'Low'
        )

    def _get_common_tests(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Get tests where both strategies were run on same stock+period"""
        # Find common stock-period combinations
        df1_keys = set(zip(df1['symbol'], df1['period_name']))
        df2_keys = set(zip(df2['symbol'], df2['period_name']))
        common_keys = df1_keys & df2_keys

        # Filter to common tests
        common_df1 = df1[
            df1.apply(lambda r: (r['symbol'], r['period_name']) in common_keys, axis=1)
        ]
        common_df2 = df2[
            df2.apply(lambda r: (r['symbol'], r['period_name']) in common_keys, axis=1)
        ]

        return {'df1': common_df1, 'df2': common_df2}

    def _empty_metrics(self, strategy_name: str) -> StrategyMetrics:
        """Return empty metrics"""
        return StrategyMetrics(
            strategy_name=strategy_name,
            risk_class=StrategyRiskClass.MODERATE,
            mean_return=0.0, median_return=0.0, total_return_range=(0.0, 0.0), ci_95=(0.0, 0.0),
            mean_sharpe=0.0, mean_volatility=0.0, mean_max_drawdown=0.0, mean_sortino=0.0,
            mean_win_rate=0.0, mean_profit_factor=0.0, avg_trades_per_period=0.0,
            return_std=0.0, return_cv=0.0, consistency_score=0.0,
            num_tests=0, profitable_tests=0, success_rate=0.0,
            reliability='Low'
        )

    def _empty_multi_comparison(self, strategies: List[str]) -> MultiStrategyComparison:
        """Return empty multi-comparison"""
        return MultiStrategyComparison(
            strategies=strategies,
            num_comparisons=0,
            rankings_by_return=[],
            rankings_by_sharpe=[],
            rankings_by_consistency=[],
            rankings_by_win_rate=[],
            overall_winner="N/A",
            overall_score=0.0,
            pairwise_results=[],
            strategy_metrics={}
        )

    def format_comparison(self, result: ComparisonResult) -> str:
        """Format comparison result as text"""
        report = f"""
{'='*70}
STRATEGY COMPARISON
{'='*70}

{result.strategy1}  vs  {result.strategy2}

PERFORMANCE:
  Return Difference:    {result.return_diff:>+8.2f}% ({result.strategy1} - {result.strategy2})
  Sharpe Difference:    {result.sharpe_diff:>+8.2f}
  Win Rate Difference:  {result.win_rate_diff:>+8.2f}%

WINNER:
  By Return:            {result.winner} ({result.confidence} confidence)
  By Risk-Adj Return:   {result.risk_adjusted_winner}

STATISTICAL TEST (Mann-Whitney U):
  Test Statistic:       {result.test_statistic:.2f}
  P-Value:              {result.p_value:.4f}
  Significant:          {'Yes (p < 0.05)' if result.is_significant else 'No'}
  Interpretation:       {result.interpretation}

HEAD-TO-HEAD (on same stocks & periods):
  {result.strategy1} wins:   {result.strategy1_better_count}
  {result.strategy2} wins:   {result.strategy2_better_count}
  Ties:                 {result.tie_count}

{'='*70}
"""
        return report

    def format_multi_comparison(self, result: MultiStrategyComparison) -> str:
        """Format multi-strategy comparison"""
        report = f"""
{'='*70}
MULTI-STRATEGY COMPARISON
{'='*70}

Comparing {len(result.strategies)} strategies:
{', '.join(result.strategies)}

OVERALL WINNER: {result.overall_winner} (Score: {result.overall_score:.2f})

{'─'*70}
RANKINGS BY RETURN:
{'─'*70}
"""
        for i, (strategy, value) in enumerate(result.rankings_by_return, 1):
            report += f"{i}. {strategy:<40} {value:>8.2f}%\n"

        report += f"\n{'─'*70}\n"
        report += "RANKINGS BY SHARPE RATIO:\n"
        report += f"{'─'*70}\n"

        for i, (strategy, value) in enumerate(result.rankings_by_sharpe, 1):
            report += f"{i}. {strategy:<40} {value:>8.2f}\n"

        report += f"\n{'─'*70}\n"
        report += "RANKINGS BY CONSISTENCY:\n"
        report += f"{'─'*70}\n"

        for i, (strategy, value) in enumerate(result.rankings_by_consistency, 1):
            report += f"{i}. {strategy:<40} {value:>8.1f}/100\n"

        report += f"\n{'─'*70}\n"
        report += "DETAILED METRICS:\n"
        report += f"{'─'*70}\n"

        for strategy, metrics in result.strategy_metrics.items():
            report += f"""
{strategy} ({metrics.risk_class.value}):
  Mean Return:      {metrics.mean_return:>8.2f}% (95% CI: [{metrics.ci_95[0]:.2f}%, {metrics.ci_95[1]:.2f}%])
  Sharpe Ratio:     {metrics.mean_sharpe:>8.2f}
  Win Rate:         {metrics.mean_win_rate:>8.1f}%
  Consistency:      {metrics.consistency_score:>8.1f}/100
  Success Rate:     {metrics.success_rate:>8.1f}% ({metrics.profitable_tests}/{metrics.num_tests})
  Reliability:      {metrics.reliability}
"""

        report += f"\n{'='*70}\n"

        return report


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Compare trading strategies')
    parser.add_argument('strategies', nargs='+', help='Strategies to compare (2 or more)')
    parser.add_argument('--timestamp', type=str, help='Backtest timestamp')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--stock', type=str, help='Filter by stock symbol')
    parser.add_argument('--period', type=str, help='Filter by period')
    parser.add_argument('--sector', type=str, help='Filter by sector')

    args = parser.parse_args()

    # Load data
    if args.csv:
        comparator = StrategyComparator.from_csv(args.csv)
    elif args.timestamp:
        comparator = StrategyComparator.from_timestamp(args.timestamp)
    else:
        # Find latest
        results_dir = Path('backtest_results')
        csv_files = list(results_dir.glob('detailed_results_*.csv'))
        if not csv_files:
            print("No backtest results found!")
            return
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"Using latest results: {latest_csv.name}\n")
        comparator = StrategyComparator.from_csv(str(latest_csv))

    # Compare
    if len(args.strategies) == 2:
        result = comparator.compare_two_strategies(
            args.strategies[0],
            args.strategies[1],
            stock=args.stock,
            period=args.period,
            sector=args.sector
        )
        if result:
            print(comparator.format_comparison(result))
        else:
            print("Insufficient data for comparison")
    else:
        result = comparator.compare_multiple_strategies(
            args.strategies,
            stock=args.stock,
            period=args.period,
            sector=args.sector
        )
        print(comparator.format_multi_comparison(result))


if __name__ == '__main__':
    main()
