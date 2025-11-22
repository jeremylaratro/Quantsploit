"""
Comprehensive Multi-Strategy Backtesting System

This module provides a framework for running comprehensive backtests across
all available trading strategies over multiple time periods, comparing their
performance, and identifying the most accurate signal generators.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
import json
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

from quantsploit.utils.backtesting import Backtester, BacktestConfig, BacktestResults
from quantsploit.utils.data_fetcher import DataFetcher
from quantsploit.modules.strategies.sma_crossover import SMACrossover
from quantsploit.modules.strategies.mean_reversion import MeanReversion
from quantsploit.modules.strategies.momentum_signals import MomentumSignals
from quantsploit.modules.strategies.multifactor_scoring import MultiFactorScoring

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestPeriod:
    """Defines a time period for backtesting"""
    name: str
    start_date: str
    end_date: str
    description: str


@dataclass
class StrategyPerformance:
    """Performance metrics for a single strategy in a single time period"""
    strategy_name: str
    period_name: str
    symbol: str

    # Returns metrics
    total_return: float
    annual_return: float
    buy_and_hold_return: float
    excess_return: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float

    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Signal accuracy metrics
    correct_signals: int
    total_signals: int
    signal_accuracy: float

    def to_dict(self):
        return asdict(self)


class StrategyAdapter:
    """Adapts various strategy modules to work with the backtesting framework"""

    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher

    def sma_crossover_strategy(self, backtester: Backtester, date: pd.Timestamp,
                               row: pd.Series, symbol: str, data: pd.DataFrame,
                               short_window: int = 20, long_window: int = 50):
        """SMA Crossover Strategy Adapter"""
        # Get historical data up to current date
        history = data.loc[:date]

        if len(history) < long_window:
            return

        # Calculate SMAs
        short_sma = history['Close'].rolling(window=short_window).mean().iloc[-1]
        long_sma = history['Close'].rolling(window=long_window).mean().iloc[-1]

        if len(history) < 2:
            return

        prev_short_sma = history['Close'].rolling(window=short_window).mean().iloc[-2]
        prev_long_sma = history['Close'].rolling(window=long_window).mean().iloc[-2]

        # Generate signals
        current_position = backtester.positions.get(symbol)

        # Bullish crossover
        if prev_short_sma <= prev_long_sma and short_sma > long_sma:
            if current_position is None:
                backtester.enter_long(symbol, date, row['Close'])

        # Bearish crossover
        elif prev_short_sma >= prev_long_sma and short_sma < long_sma:
            if current_position is not None:
                backtester.exit_position(symbol, date, row['Close'])

    def mean_reversion_strategy(self, backtester: Backtester, date: pd.Timestamp,
                                row: pd.Series, symbol: str, data: pd.DataFrame,
                                lookback: int = 20, entry_threshold: float = -60,
                                exit_threshold: float = 40):
        """Mean Reversion Strategy Adapter"""
        history = data.loc[:date]

        if len(history) < lookback:
            return

        # Calculate z-score
        recent_prices = history['Close'].iloc[-lookback:]
        mean_price = recent_prices.mean()
        std_price = recent_prices.std()

        if std_price == 0:
            return

        z_score = (row['Close'] - mean_price) / std_price
        signal_strength = -z_score * 40  # Scale to -100 to 100

        current_position = backtester.positions.get(symbol)

        # Enter long when oversold
        if signal_strength < entry_threshold:
            if current_position is None:
                backtester.enter_long(symbol, date, row['Close'])

        # Exit when overbought or mean reverted
        elif signal_strength > exit_threshold:
            if current_position is not None:
                backtester.exit_position(symbol, date, row['Close'])

    def momentum_strategy(self, backtester: Backtester, date: pd.Timestamp,
                          row: pd.Series, symbol: str, data: pd.DataFrame,
                          periods: List[int] = [10, 20, 50],
                          entry_threshold: float = 60, exit_threshold: float = -40):
        """Momentum Strategy Adapter"""
        history = data.loc[:date]

        max_period = max(periods)
        if len(history) < max_period:
            return

        # Calculate multi-period momentum
        momentum_scores = []
        for period in periods:
            if len(history) >= period:
                roc = ((row['Close'] - history['Close'].iloc[-period]) /
                       history['Close'].iloc[-period]) * 100
                momentum_scores.append(roc)

        if not momentum_scores:
            return

        avg_momentum = np.mean(momentum_scores)
        # Scale to -100 to 100
        signal_strength = np.clip(avg_momentum * 10, -100, 100)

        current_position = backtester.positions.get(symbol)

        # Enter long on strong momentum
        if signal_strength > entry_threshold:
            if current_position is None:
                backtester.enter_long(symbol, date, row['Close'])

        # Exit on weak/negative momentum
        elif signal_strength < exit_threshold:
            if current_position is not None:
                backtester.exit_position(symbol, date, row['Close'])


class ComprehensiveBacktester:
    """
    Main class for running comprehensive backtests across multiple strategies
    and time periods
    """

    def __init__(self, symbols: List[str], initial_capital: float = 100000,
                 commission_pct: float = 0.001, slippage_pct: float = 0.001):
        """
        Initialize the comprehensive backtester

        Args:
            symbols: List of stock symbols to test
            initial_capital: Starting capital for each backtest
            commission_pct: Commission percentage per trade
            slippage_pct: Slippage percentage per trade
        """
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        self.data_fetcher = DataFetcher()
        self.adapter = StrategyAdapter(self.data_fetcher)

        # Define available strategies
        self.strategies = {
            'sma_crossover': {
                'name': 'SMA Crossover (20/50)',
                'function': self.adapter.sma_crossover_strategy,
                'params': {'short_window': 20, 'long_window': 50}
            },
            'sma_crossover_fast': {
                'name': 'SMA Crossover (10/30)',
                'function': self.adapter.sma_crossover_strategy,
                'params': {'short_window': 10, 'long_window': 30}
            },
            'mean_reversion': {
                'name': 'Mean Reversion (20 day)',
                'function': self.adapter.mean_reversion_strategy,
                'params': {'lookback': 20, 'entry_threshold': -60, 'exit_threshold': 40}
            },
            'mean_reversion_aggressive': {
                'name': 'Mean Reversion Aggressive',
                'function': self.adapter.mean_reversion_strategy,
                'params': {'lookback': 10, 'entry_threshold': -50, 'exit_threshold': 30}
            },
            'momentum': {
                'name': 'Momentum (10/20/50)',
                'function': self.adapter.momentum_strategy,
                'params': {'periods': [10, 20, 50], 'entry_threshold': 60, 'exit_threshold': -40}
            },
            'momentum_aggressive': {
                'name': 'Momentum Aggressive',
                'function': self.adapter.momentum_strategy,
                'params': {'periods': [5, 10, 20], 'entry_threshold': 50, 'exit_threshold': -30}
            }
        }

        self.results: List[StrategyPerformance] = []

    def generate_test_periods(self, years_back: int = 3) -> List[TestPeriod]:
        """
        Generate test periods for backtesting

        Args:
            years_back: Number of years to look back

        Returns:
            List of TestPeriod objects
        """
        periods = []
        end_date = datetime.now()

        # Full period tests
        for years in [1, 2, 3]:
            if years <= years_back:
                start_date = end_date - timedelta(days=years*365)
                periods.append(TestPeriod(
                    name=f'{years}year',
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    description=f'{years} Year Period'
                ))

        # 6-month rolling periods over the past 2 years
        for i in range(4):
            period_end = end_date - timedelta(days=i*180)
            period_start = period_end - timedelta(days=180)
            periods.append(TestPeriod(
                name=f'6mo_period_{i+1}',
                start_date=period_start.strftime('%Y-%m-%d'),
                end_date=period_end.strftime('%Y-%m-%d'),
                description=f'6-Month Period {i+1} ({period_start.strftime("%b %Y")} - {period_end.strftime("%b %Y")})'
            ))

        # Market condition periods (you can customize these)
        # Bull market period (if applicable)
        # Bear market period (if applicable)
        # Volatile period (if applicable)

        return periods

    def calculate_signal_accuracy(self, trades: List, data: pd.DataFrame) -> Tuple[int, int, float]:
        """
        Calculate signal accuracy based on trade outcomes

        Args:
            trades: List of Trade objects
            data: Historical price data

        Returns:
            Tuple of (correct_signals, total_signals, accuracy_percentage)
        """
        if not trades:
            return 0, 0, 0.0

        correct = sum(1 for trade in trades if trade.pnl > 0)
        total = len(trades)
        accuracy = (correct / total * 100) if total > 0 else 0.0

        return correct, total, accuracy

    def run_single_backtest(self, strategy_key: str, symbol: str,
                           period: TestPeriod) -> Optional[StrategyPerformance]:
        """
        Run a single backtest for one strategy, symbol, and time period

        Args:
            strategy_key: Key identifying the strategy
            symbol: Stock symbol
            period: TestPeriod object

        Returns:
            StrategyPerformance object or None if backtest fails
        """
        try:
            logger.info(f"Running {strategy_key} for {symbol} in {period.name}")

            # Fetch data
            data = self.data_fetcher.get_stock_data(
                symbol=symbol,
                period='3y',  # Fetch more data to ensure we have enough
                interval='1d'
            )

            if data is None or len(data) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return None

            # Filter data to the specific period
            data = data.loc[period.start_date:period.end_date]

            if len(data) < 30:
                logger.warning(f"Insufficient data in period {period.name} for {symbol}")
                return None

            # Create backtester config
            config = BacktestConfig(
                initial_capital=self.initial_capital,
                commission_pct=self.commission_pct,
                slippage_pct=self.slippage_pct,
                position_size=1.0,
                max_positions=1
            )

            # Create strategy function with params
            strategy_info = self.strategies[strategy_key]

            # Create a closure to pass data to strategy
            def strategy_func(bt, date, row):
                strategy_info['function'](bt, date, row, symbol, data, **strategy_info['params'])

            # Run backtest
            backtester = Backtester(config)
            results = backtester.run_backtest(data, strategy_func)

            # Calculate signal accuracy from completed trades
            completed_trades = [t for t in backtester.trades if t.exit_date is not None]
            correct, total, accuracy = self.calculate_signal_accuracy(
                completed_trades, data
            )

            # Create performance object
            performance = StrategyPerformance(
                strategy_name=strategy_info['name'],
                period_name=period.description,
                symbol=symbol,
                total_return=results.total_return_pct,
                annual_return=results.annualized_return,
                buy_and_hold_return=results.benchmark_return,
                excess_return=results.total_return_pct - results.benchmark_return,
                sharpe_ratio=results.sharpe_ratio,
                sortino_ratio=results.sortino_ratio,
                max_drawdown=results.max_drawdown,
                volatility=results.volatility,
                total_trades=results.total_trades,
                win_rate=results.win_rate,
                profit_factor=results.profit_factor,
                avg_win=results.avg_win,
                avg_loss=results.avg_loss,
                correct_signals=correct,
                total_signals=total,
                signal_accuracy=accuracy
            )

            return performance

        except Exception as e:
            logger.error(f"Error running backtest for {strategy_key}/{symbol}/{period.name}: {e}")
            return None

    def run_comprehensive_backtest(self, parallel: bool = True,
                                   max_workers: int = 4) -> pd.DataFrame:
        """
        Run comprehensive backtests across all strategies, symbols, and periods

        Args:
            parallel: Whether to run backtests in parallel
            max_workers: Maximum number of parallel workers

        Returns:
            DataFrame with all results
        """
        periods = self.generate_test_periods()

        logger.info(f"Running comprehensive backtest:")
        logger.info(f"  - {len(self.strategies)} strategies")
        logger.info(f"  - {len(self.symbols)} symbols")
        logger.info(f"  - {len(periods)} time periods")
        logger.info(f"  - Total: {len(self.strategies) * len(self.symbols) * len(periods)} backtests")

        self.results = []

        # Generate all combinations
        tasks = []
        for strategy_key in self.strategies.keys():
            for symbol in self.symbols:
                for period in periods:
                    tasks.append((strategy_key, symbol, period))

        # Run backtests
        if parallel:
            # Note: ProcessPoolExecutor may have issues with instance methods
            # Run sequentially for now, but optimized
            for strategy_key, symbol, period in tasks:
                result = self.run_single_backtest(strategy_key, symbol, period)
                if result:
                    self.results.append(result)
        else:
            for strategy_key, symbol, period in tasks:
                result = self.run_single_backtest(strategy_key, symbol, period)
                if result:
                    self.results.append(result)

        # Convert to DataFrame
        if self.results:
            df = pd.DataFrame([r.to_dict() for r in self.results])
            return df
        else:
            return pd.DataFrame()

    def generate_summary_report(self, results_df: pd.DataFrame) -> Dict:
        """
        Generate a summary report from backtest results

        Args:
            results_df: DataFrame with all backtest results

        Returns:
            Dictionary with summary statistics
        """
        if results_df.empty:
            return {"error": "No results to summarize"}

        # Overall best strategies by different metrics
        summary = {
            'best_by_total_return': self._rank_strategies(results_df, 'total_return'),
            'best_by_sharpe_ratio': self._rank_strategies(results_df, 'sharpe_ratio'),
            'best_by_win_rate': self._rank_strategies(results_df, 'win_rate'),
            'best_by_signal_accuracy': self._rank_strategies(results_df, 'signal_accuracy'),
            'best_by_profit_factor': self._rank_strategies(results_df, 'profit_factor'),
            'best_excess_return': self._rank_strategies(results_df, 'excess_return'),

            # Statistics by period
            'performance_by_period': self._analyze_by_period(results_df),

            # Statistics by symbol
            'performance_by_symbol': self._analyze_by_symbol(results_df),

            # Overall statistics
            'overall_stats': {
                'total_backtests': len(results_df),
                'avg_return': results_df['total_return'].mean(),
                'avg_sharpe': results_df['sharpe_ratio'].mean(),
                'avg_win_rate': results_df['win_rate'].mean(),
                'avg_signal_accuracy': results_df['signal_accuracy'].mean(),
                'strategies_beating_buy_hold': (results_df['excess_return'] > 0).sum(),
                'percentage_beating_buy_hold': (results_df['excess_return'] > 0).sum() / len(results_df) * 100
            }
        }

        return summary

    def _rank_strategies(self, df: pd.DataFrame, metric: str, top_n: int = 10) -> List[Dict]:
        """Rank strategies by a specific metric"""
        ranked = df.nlargest(top_n, metric)[
            ['strategy_name', 'symbol', 'period_name', metric, 'total_return',
             'win_rate', 'signal_accuracy', 'sharpe_ratio']
        ]
        return ranked.to_dict('records')

    def _analyze_by_period(self, df: pd.DataFrame) -> Dict:
        """Analyze performance grouped by time period"""
        grouped = df.groupby('period_name').agg({
            'total_return': ['mean', 'std', 'min', 'max'],
            'sharpe_ratio': ['mean', 'std'],
            'win_rate': 'mean',
            'signal_accuracy': 'mean',
            'excess_return': 'mean'
        }).round(4)

        return grouped.to_dict()

    def _analyze_by_symbol(self, df: pd.DataFrame) -> Dict:
        """Analyze performance grouped by symbol"""
        grouped = df.groupby('symbol').agg({
            'total_return': ['mean', 'std'],
            'sharpe_ratio': 'mean',
            'win_rate': 'mean',
            'signal_accuracy': 'mean',
            'excess_return': 'mean'
        }).round(4)

        return grouped.to_dict()

    def save_results(self, results_df: pd.DataFrame, summary: Dict,
                    output_dir: str = './backtest_results'):
        """
        Save backtest results to files

        Args:
            results_df: DataFrame with all results
            summary: Summary dictionary
            output_dir: Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save detailed results
        results_file = f'{output_dir}/detailed_results_{timestamp}.csv'
        results_df.to_csv(results_file, index=False)
        logger.info(f"Detailed results saved to {results_file}")

        # Save summary
        summary_file = f'{output_dir}/summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary saved to {summary_file}")

        # Generate markdown report
        self._generate_markdown_report(results_df, summary,
                                       f'{output_dir}/report_{timestamp}.md')

    def _generate_markdown_report(self, results_df: pd.DataFrame,
                                  summary: Dict, output_file: str):
        """Generate a markdown report"""
        with open(output_file, 'w') as f:
            f.write("# Comprehensive Backtest Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overall statistics
            f.write("## Overall Statistics\n\n")
            stats = summary['overall_stats']
            f.write(f"- Total Backtests: {stats['total_backtests']}\n")
            f.write(f"- Average Return: {stats['avg_return']:.2f}%\n")
            f.write(f"- Average Sharpe Ratio: {stats['avg_sharpe']:.2f}\n")
            f.write(f"- Average Win Rate: {stats['avg_win_rate']:.2f}%\n")
            f.write(f"- Average Signal Accuracy: {stats['avg_signal_accuracy']:.2f}%\n")
            f.write(f"- Strategies Beating Buy & Hold: {stats['strategies_beating_buy_hold']} ({stats['percentage_beating_buy_hold']:.1f}%)\n\n")

            # Best strategies by return
            f.write("## Top 10 Strategies by Total Return\n\n")
            f.write("| Rank | Strategy | Symbol | Period | Return | Win Rate | Signal Accuracy | Sharpe |\n")
            f.write("|------|----------|--------|--------|--------|----------|-----------------|--------|\n")
            for i, row in enumerate(summary['best_by_total_return'], 1):
                f.write(f"| {i} | {row['strategy_name']} | {row['symbol']} | {row['period_name']} | "
                       f"{row['total_return']:.2f}% | {row['win_rate']:.1f}% | "
                       f"{row['signal_accuracy']:.1f}% | {row['sharpe_ratio']:.2f} |\n")

            # Best by Sharpe Ratio
            f.write("\n## Top 10 Strategies by Sharpe Ratio\n\n")
            f.write("| Rank | Strategy | Symbol | Period | Sharpe | Return | Win Rate |\n")
            f.write("|------|----------|--------|--------|--------|--------|----------|\n")
            for i, row in enumerate(summary['best_by_sharpe_ratio'], 1):
                f.write(f"| {i} | {row['strategy_name']} | {row['symbol']} | {row['period_name']} | "
                       f"{row['sharpe_ratio']:.2f} | {row['total_return']:.2f}% | {row['win_rate']:.1f}% |\n")

            # Best by Signal Accuracy
            f.write("\n## Top 10 Strategies by Signal Accuracy\n\n")
            f.write("| Rank | Strategy | Symbol | Period | Accuracy | Win Rate | Return |\n")
            f.write("|------|----------|--------|--------|----------|----------|--------|\n")
            for i, row in enumerate(summary['best_by_signal_accuracy'], 1):
                f.write(f"| {i} | {row['strategy_name']} | {row['symbol']} | {row['period_name']} | "
                       f"{row['signal_accuracy']:.1f}% | {row['win_rate']:.1f}% | {row['total_return']:.2f}% |\n")

        logger.info(f"Markdown report saved to {output_file}")


def run_comprehensive_analysis(symbols: List[str], output_dir: str = './backtest_results'):
    """
    Convenience function to run a complete comprehensive backtest

    Args:
        symbols: List of stock symbols to analyze
        output_dir: Directory to save results
    """
    # Create backtester
    backtester = ComprehensiveBacktester(
        symbols=symbols,
        initial_capital=100000,
        commission_pct=0.001,
        slippage_pct=0.001
    )

    # Run comprehensive backtest
    results_df = backtester.run_comprehensive_backtest(parallel=False)

    if results_df.empty:
        logger.error("No results generated")
        return None, None

    # Generate summary
    summary = backtester.generate_summary_report(results_df)

    # Save results
    backtester.save_results(results_df, summary, output_dir)

    return results_df, summary
