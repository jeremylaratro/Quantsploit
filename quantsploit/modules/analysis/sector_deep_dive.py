"""
Enhanced Sector Analysis with Stratified Statistics

Provides robust sector-level analysis that fixes issues with naive aggregation:
- Stratifies by strategy risk class
- Uses robust statistics (median, MAD, IQR)
- Filters unreliable results
- Provides confidence intervals
- Ranks stocks within each sector

Enables answering:
- "Which sectors are performing best?"
- "Which stocks lead each sector?"
- "Is sector performance consistent across strategies?"
- "What's the best strategy for tech stocks?"
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.statistical_analyzer import (
    StatisticalAnalyzer,
    StratifiedStatistics,
    RobustStatistics,
    StrategyRiskClass,
    format_stratified_report
)


@dataclass
class StockRanking:
    """Ranking of a stock within its sector"""
    symbol: str
    rank: int
    total_in_sector: int

    mean_return: float
    median_return: float
    best_strategy: str
    best_strategy_return: float

    consistency_score: float
    num_tests: int
    reliability: str


@dataclass
class SectorPerformance:
    """Performance metrics for a sector"""
    sector_name: str
    num_stocks: int
    num_strategies: int
    total_tests: int

    # Robust statistics
    stratified_stats: StratifiedStatistics
    overall_return: RobustStatistics

    # Top performers
    top_stocks: List[StockRanking]
    best_stock: str
    best_stock_return: float

    # Best strategy for this sector
    best_strategy: str
    best_strategy_return: float
    best_strategy_consistency: float

    # Sector characteristics
    avg_volatility: float
    avg_sharpe: float
    avg_win_rate: float

    # Quality metrics
    data_quality_score: float
    reliability_rating: str


@dataclass
class SectorComparison:
    """Comparison across multiple sectors"""
    num_sectors: int
    total_stocks: int
    total_tests: int

    # Sector rankings
    by_return: List[Tuple[str, float]]
    by_sharpe: List[Tuple[str, float]]
    by_consistency: List[Tuple[str, float]]

    # Best overall sector
    best_sector: str
    best_sector_return: float
    best_sector_reliability: str

    # Detailed sector data
    sector_details: Dict[str, SectorPerformance]


class SectorAnalyzer:
    """
    Analyze performance by sector with robust statistics
    """

    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize analyzer

        Args:
            results_df: DataFrame with backtest results (must include 'symbol' column)
        """
        self.df = results_df
        self.stat_analyzer = StatisticalAnalyzer(min_sample_size=5)

        # Import sector taxonomy
        try:
            from dashboard.ticker_universe import get_sector
            self.get_sector = get_sector
        except ImportError:
            # Fallback if not available
            self.get_sector = lambda x: "Unknown"

    @classmethod
    def from_csv(cls, csv_path: str) -> 'SectorAnalyzer':
        """Load from CSV file"""
        df = pd.read_csv(csv_path)
        return cls(df)

    @classmethod
    def from_timestamp(cls, timestamp: str, results_dir: str = 'backtest_results') -> 'SectorAnalyzer':
        """Load from timestamp"""
        csv_path = Path(results_dir) / f'detailed_results_{timestamp}.csv'
        return cls.from_csv(str(csv_path))

    def analyze_sector(
        self,
        sector: str,
        min_trades: int = 5,
        top_n_stocks: int = 10
    ) -> Optional[SectorPerformance]:
        """
        Deep dive into a single sector

        Args:
            sector: Sector name
            min_trades: Minimum trades per result
            top_n_stocks: Number of top stocks to include

        Returns:
            SectorPerformance or None if insufficient data
        """
        # Add sector classification
        self.df['sector'] = self.df['symbol'].apply(self.get_sector)

        # Filter for this sector
        sector_df = self.df[self.df['sector'] == sector].copy()

        if len(sector_df) == 0:
            return None

        # Filter by minimum trades
        if 'total_trades' in sector_df.columns:
            sector_df = sector_df[sector_df['total_trades'] >= min_trades]

        if len(sector_df) < 5:  # Need at least 5 results
            return None

        # Calculate stratified statistics (fixes large stdev problem!)
        stratified_stats = self.stat_analyzer.calculate_stratified_stats(
            sector_df,
            value_col='total_return',
            strategy_col='strategy_name'
        )

        # Calculate overall robust statistics
        overall_stats = self.stat_analyzer.calculate_robust_stats(
            sector_df['total_return']
        )

        # Rank stocks in this sector
        stock_rankings = self._rank_stocks_in_sector(sector_df, top_n_stocks)

        # Find best stock
        if stock_rankings:
            best_stock = stock_rankings[0].symbol
            best_stock_return = stock_rankings[0].mean_return
        else:
            best_stock = "N/A"
            best_stock_return = 0.0

        # Find best strategy for this sector
        best_strategy_data = self._find_best_strategy(sector_df)

        # Calculate sector characteristics
        avg_volatility = sector_df['volatility'].mean() if 'volatility' in sector_df.columns else 0.0
        avg_sharpe = sector_df['sharpe_ratio'].mean() if 'sharpe_ratio' in sector_df.columns else 0.0
        avg_win_rate = sector_df['win_rate'].mean() if 'win_rate' in sector_df.columns else 0.0

        return SectorPerformance(
            sector_name=sector,
            num_stocks=sector_df['symbol'].nunique(),
            num_strategies=sector_df['strategy_name'].nunique(),
            total_tests=len(sector_df),

            stratified_stats=stratified_stats,
            overall_return=overall_stats,

            top_stocks=stock_rankings,
            best_stock=best_stock,
            best_stock_return=best_stock_return,

            best_strategy=best_strategy_data['name'],
            best_strategy_return=best_strategy_data['return'],
            best_strategy_consistency=best_strategy_data['consistency'],

            avg_volatility=avg_volatility,
            avg_sharpe=avg_sharpe,
            avg_win_rate=avg_win_rate,

            data_quality_score=stratified_stats.data_quality_score,
            reliability_rating=stratified_stats.reliability_rating
        )

    def compare_all_sectors(
        self,
        min_trades: int = 5,
        min_stocks_per_sector: int = 3
    ) -> SectorComparison:
        """
        Compare all sectors

        Args:
            min_trades: Minimum trades per result
            min_stocks_per_sector: Minimum stocks to qualify as a sector

        Returns:
            SectorComparison with rankings and details
        """
        # Add sector classification
        self.df['sector'] = self.df['symbol'].apply(self.get_sector)

        # Analyze each sector
        sector_details = {}
        for sector in self.df['sector'].unique():
            if sector and sector != "Unknown":
                perf = self.analyze_sector(sector, min_trades)
                if perf and perf.num_stocks >= min_stocks_per_sector:
                    sector_details[sector] = perf

        if len(sector_details) == 0:
            return self._empty_comparison()

        # Rank sectors by different metrics
        by_return = sorted(
            [(s, p.stratified_stats.overall.median) for s, p in sector_details.items()],
            key=lambda x: x[1],
            reverse=True
        )

        by_sharpe = sorted(
            [(s, p.avg_sharpe) for s, p in sector_details.items()],
            key=lambda x: x[1],
            reverse=True
        )

        by_consistency = sorted(
            [(s, 1/(1 + p.stratified_stats.overall.cv)) for s, p in sector_details.items()],
            key=lambda x: x[1],
            reverse=True
        )

        # Best overall sector (highest median return with good reliability)
        reliable_sectors = {s: p for s, p in sector_details.items()
                          if p.reliability_rating in ['High', 'Medium']}

        if reliable_sectors:
            best_sector = max(
                reliable_sectors.items(),
                key=lambda x: x[1].stratified_stats.overall.median
            )[0]
            best_sector_return = reliable_sectors[best_sector].stratified_stats.overall.median
            best_sector_reliability = reliable_sectors[best_sector].reliability_rating
        else:
            best_sector = by_return[0][0] if by_return else "N/A"
            best_sector_return = by_return[0][1] if by_return else 0.0
            best_sector_reliability = "Low"

        # Total statistics
        total_stocks = sum(p.num_stocks for p in sector_details.values())
        total_tests = sum(p.total_tests for p in sector_details.values())

        return SectorComparison(
            num_sectors=len(sector_details),
            total_stocks=total_stocks,
            total_tests=total_tests,

            by_return=by_return,
            by_sharpe=by_sharpe,
            by_consistency=by_consistency,

            best_sector=best_sector,
            best_sector_return=best_sector_return,
            best_sector_reliability=best_sector_reliability,

            sector_details=sector_details
        )

    def _rank_stocks_in_sector(
        self,
        sector_df: pd.DataFrame,
        top_n: int
    ) -> List[StockRanking]:
        """Rank stocks within a sector"""
        rankings = []

        for symbol in sector_df['symbol'].unique():
            stock_data = sector_df[sector_df['symbol'] == symbol]

            if len(stock_data) == 0:
                continue

            # Calculate statistics
            stats = self.stat_analyzer.calculate_robust_stats(stock_data['total_return'])

            # Find best strategy for this stock
            best_idx = stock_data['total_return'].idxmax()
            best_strategy = stock_data.loc[best_idx, 'strategy_name']
            best_strategy_return = stock_data.loc[best_idx, 'total_return']

            # Consistency score
            consistency = 1 / (1 + stats.cv) if stats.cv > 0 else 1.0

            rankings.append({
                'symbol': symbol,
                'mean_return': stats.mean,
                'median_return': stats.median,
                'best_strategy': best_strategy,
                'best_strategy_return': best_strategy_return,
                'consistency_score': consistency,
                'num_tests': len(stock_data),
                'reliability': 'High' if len(stock_data) >= 10 else 'Medium' if len(stock_data) >= 5 else 'Low'
            })

        # Sort by mean return
        rankings.sort(key=lambda x: x['mean_return'], reverse=True)

        # Convert to StockRanking objects with rank
        total_stocks = len(rankings)
        result = []
        for i, r in enumerate(rankings[:top_n], 1):
            result.append(StockRanking(
                symbol=r['symbol'],
                rank=i,
                total_in_sector=total_stocks,
                mean_return=r['mean_return'],
                median_return=r['median_return'],
                best_strategy=r['best_strategy'],
                best_strategy_return=r['best_strategy_return'],
                consistency_score=r['consistency_score'],
                num_tests=r['num_tests'],
                reliability=r['reliability']
            ))

        return result

    def _find_best_strategy(self, sector_df: pd.DataFrame) -> Dict:
        """Find best strategy for a sector"""
        strategy_stats = []

        for strategy in sector_df['strategy_name'].unique():
            strategy_data = sector_df[sector_df['strategy_name'] == strategy]

            if len(strategy_data) < 3:
                continue

            stats = self.stat_analyzer.calculate_robust_stats(strategy_data['total_return'])

            consistency = 1 / (1 + stats.cv) if stats.cv > 0 else 1.0

            strategy_stats.append({
                'name': strategy,
                'return': stats.mean,
                'consistency': consistency,
                'num_tests': len(strategy_data)
            })

        if not strategy_stats:
            return {'name': 'N/A', 'return': 0.0, 'consistency': 0.0}

        # Best = highest return among strategies with at least 5 tests
        reliable_strategies = [s for s in strategy_stats if s['num_tests'] >= 5]

        if reliable_strategies:
            best = max(reliable_strategies, key=lambda x: x['return'])
        else:
            best = max(strategy_stats, key=lambda x: x['return'])

        return best

    def _empty_comparison(self) -> SectorComparison:
        """Return empty comparison"""
        return SectorComparison(
            num_sectors=0,
            total_stocks=0,
            total_tests=0,
            by_return=[],
            by_sharpe=[],
            by_consistency=[],
            best_sector="N/A",
            best_sector_return=0.0,
            best_sector_reliability="Low",
            sector_details={}
        )

    def format_sector_report(self, sector_perf: SectorPerformance) -> str:
        """Format sector performance as text"""
        report = f"""
{'='*70}
SECTOR DEEP DIVE: {sector_perf.sector_name}
{'='*70}

OVERVIEW:
  Reliability:     {sector_perf.reliability_rating} (Quality Score: {sector_perf.data_quality_score:.1f}/100)
  Stocks Tested:   {sector_perf.num_stocks}
  Strategies:      {sector_perf.num_strategies}
  Total Tests:     {sector_perf.total_tests}

STRATIFIED PERFORMANCE (fixes large stdev problem!):
"""

        # Add stratified statistics
        for risk_class, class_stats in sector_perf.stratified_stats.by_risk_class.items():
            report += f"""
  {risk_class.value.upper()} Strategies:
    Median Return: {class_stats.median:>8.2f}%
    Mean Return:   {class_stats.mean:>8.2f}% ± {class_stats.sem:.2f}%
    Std Dev:       {class_stats.std:>8.2f}% (MAD: {class_stats.mad:.2f}%)
    95% CI:        [{class_stats.ci_lower:.2f}%, {class_stats.ci_upper:.2f}%]
    Tests:         {class_stats.count}
"""

        report += f"""
OVERALL STATISTICS:
  Median Return:   {sector_perf.overall_return.median:>8.2f}%
  Mean Return:     {sector_perf.overall_return.mean:>8.2f}%
  Std Dev:         {sector_perf.overall_return.std:>8.2f}%
  95% CI:          [{sector_perf.overall_return.ci_lower:.2f}%, {sector_perf.overall_return.ci_upper:.2f}%]

RISK METRICS:
  Avg Sharpe Ratio:  {sector_perf.avg_sharpe:>6.2f}
  Avg Volatility:    {sector_perf.avg_volatility:>6.2f}%
  Avg Win Rate:      {sector_perf.avg_win_rate:>6.1f}%

BEST PERFORMERS:
  Best Stock:      {sector_perf.best_stock} ({sector_perf.best_stock_return:+.2f}%)
  Best Strategy:   {sector_perf.best_strategy} ({sector_perf.best_strategy_return:+.2f}%)
  Strategy Consistency: {sector_perf.best_strategy_consistency:.3f}

{'─'*70}
TOP STOCKS IN {sector_perf.sector_name}:
{'─'*70}
"""

        for stock in sector_perf.top_stocks:
            report += f"""
{stock.rank}. {stock.symbol} (Rank {stock.rank}/{stock.total_in_sector})
   Mean Return:    {stock.mean_return:>8.2f}% (Median: {stock.median_return:.2f}%)
   Best Strategy:  {stock.best_strategy} ({stock.best_strategy_return:+.2f}%)
   Consistency:    {stock.consistency_score:>8.3f}
   Tests:          {stock.num_tests}
   Reliability:    {stock.reliability}
"""

        report += f"\n{'='*70}\n"

        return report

    def format_sector_comparison(self, comparison: SectorComparison) -> str:
        """Format sector comparison as text"""
        report = f"""
{'='*70}
SECTOR COMPARISON ({comparison.num_sectors} sectors)
{'='*70}

OVERVIEW:
  Total Stocks:    {comparison.total_stocks}
  Total Tests:     {comparison.total_tests}

BEST SECTOR: {comparison.best_sector}
  Median Return:   {comparison.best_sector_return:.2f}%
  Reliability:     {comparison.best_sector_reliability}

{'─'*70}
RANKINGS BY MEDIAN RETURN:
{'─'*70}
"""

        for i, (sector, value) in enumerate(comparison.by_return[:10], 1):
            perf = comparison.sector_details[sector]
            report += f"{i:2d}. {sector:<30} {value:>8.2f}% ({perf.reliability_rating}, {perf.num_stocks} stocks)\n"

        report += f"\n{'─'*70}\n"
        report += "RANKINGS BY SHARPE RATIO:\n"
        report += f"{'─'*70}\n"

        for i, (sector, value) in enumerate(comparison.by_sharpe[:10], 1):
            perf = comparison.sector_details[sector]
            report += f"{i:2d}. {sector:<30} {value:>8.2f} ({perf.num_stocks} stocks)\n"

        report += f"\n{'─'*70}\n"
        report += "RANKINGS BY CONSISTENCY:\n"
        report += f"{'─'*70}\n"

        for i, (sector, value) in enumerate(comparison.by_consistency[:10], 1):
            perf = comparison.sector_details[sector]
            report += f"{i:2d}. {sector:<30} {value:>8.3f} ({perf.num_stocks} stocks)\n"

        report += f"\n{'='*70}\n"

        return report


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze sectors with robust statistics')
    parser.add_argument('--sector', type=str, help='Specific sector to analyze')
    parser.add_argument('--timestamp', type=str, help='Backtest timestamp')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--min-trades', type=int, default=5, help='Minimum trades (default: 5)')
    parser.add_argument('--compare-all', action='store_true', help='Compare all sectors')

    args = parser.parse_args()

    # Load data
    if args.csv:
        analyzer = SectorAnalyzer.from_csv(args.csv)
    elif args.timestamp:
        analyzer = SectorAnalyzer.from_timestamp(args.timestamp)
    else:
        # Find latest
        results_dir = Path('backtest_results')
        csv_files = list(results_dir.glob('detailed_results_*.csv'))
        if not csv_files:
            print("No backtest results found!")
            return
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"Using latest results: {latest_csv.name}\n")
        analyzer = SectorAnalyzer.from_csv(str(latest_csv))

    # Analyze
    if args.compare_all:
        comparison = analyzer.compare_all_sectors(min_trades=args.min_trades)
        print(analyzer.format_sector_comparison(comparison))
    elif args.sector:
        perf = analyzer.analyze_sector(args.sector, min_trades=args.min_trades)
        if perf:
            print(analyzer.format_sector_report(perf))
        else:
            print(f"No data available for sector: {args.sector}")
    else:
        # Default: show comparison
        comparison = analyzer.compare_all_sectors(min_trades=args.min_trades)
        print(analyzer.format_sector_comparison(comparison))


if __name__ == '__main__':
    main()
