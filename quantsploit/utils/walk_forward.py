"""
Walk-Forward Optimization Framework for Quantsploit

This module provides a robust walk-forward optimization and analysis framework
for validating trading strategies with proper out-of-sample testing. Walk-forward
analysis helps prevent overfitting by testing strategies on unseen data.

Key Features:
- Anchored walk-forward: Expanding training window with fixed test window
- Rolling walk-forward: Fixed rolling training window with fixed test window
- Configurable train/test window sizes and step sizes
- Walk-forward efficiency ratio calculation
- Parameter optimization support within each training window
- Comprehensive reporting and metrics aggregation

References:
- Pardo, R. (2008). The Evaluation and Optimization of Trading Strategies
- Walk-Forward Efficiency = Out-of-Sample Performance / In-Sample Performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from copy import deepcopy

from quantsploit.utils.backtesting import Backtester, BacktestConfig, BacktestResults

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardMode(Enum):
    """Walk-forward analysis modes"""
    ANCHORED = "anchored"  # Expanding training window
    ROLLING = "rolling"    # Fixed rolling training window


@dataclass
class WalkForwardWindow:
    """
    Represents a single walk-forward window with train and test periods.

    Attributes:
        window_id: Unique identifier for this window
        train_start: Start date of training period
        train_end: End date of training period
        test_start: Start date of test (out-of-sample) period
        test_end: End date of test period
        train_days: Number of days in training period
        test_days: Number of days in test period
    """
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_days: int = 0
    test_days: int = 0

    def __post_init__(self):
        """Calculate period lengths after initialization"""
        self.train_days = (self.train_end - self.train_start).days
        self.test_days = (self.test_end - self.test_start).days


@dataclass
class WalkForwardResult:
    """
    Results from a single walk-forward window.

    Attributes:
        window: The WalkForwardWindow configuration
        in_sample_results: BacktestResults from training period
        out_of_sample_results: BacktestResults from test period
        optimized_params: Best parameters found during optimization (if applicable)
        efficiency_ratio: Out-of-sample return / In-sample return
    """
    window: WalkForwardWindow
    in_sample_results: BacktestResults
    out_of_sample_results: BacktestResults
    optimized_params: Optional[Dict[str, Any]] = None
    efficiency_ratio: float = 0.0

    def __post_init__(self):
        """Calculate efficiency ratio after initialization"""
        if (self.in_sample_results.total_return_pct != 0 and
            not np.isnan(self.in_sample_results.total_return_pct) and
            not np.isnan(self.out_of_sample_results.total_return_pct)):
            # Handle case where in-sample return is very small
            if abs(self.in_sample_results.total_return_pct) > 0.01:
                self.efficiency_ratio = (
                    self.out_of_sample_results.total_return_pct /
                    self.in_sample_results.total_return_pct
                )
            else:
                # If in-sample return is essentially zero, efficiency is undefined
                self.efficiency_ratio = np.nan
        else:
            self.efficiency_ratio = np.nan

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for reporting"""
        return {
            'window_id': self.window.window_id,
            'train_start': self.window.train_start.strftime('%Y-%m-%d'),
            'train_end': self.window.train_end.strftime('%Y-%m-%d'),
            'test_start': self.window.test_start.strftime('%Y-%m-%d'),
            'test_end': self.window.test_end.strftime('%Y-%m-%d'),
            'train_days': self.window.train_days,
            'test_days': self.window.test_days,
            'in_sample_return_pct': self.in_sample_results.total_return_pct,
            'in_sample_sharpe': self.in_sample_results.sharpe_ratio,
            'in_sample_max_dd': self.in_sample_results.max_drawdown,
            'in_sample_trades': self.in_sample_results.total_trades,
            'in_sample_win_rate': self.in_sample_results.win_rate,
            'out_of_sample_return_pct': self.out_of_sample_results.total_return_pct,
            'out_of_sample_sharpe': self.out_of_sample_results.sharpe_ratio,
            'out_of_sample_max_dd': self.out_of_sample_results.max_drawdown,
            'out_of_sample_trades': self.out_of_sample_results.total_trades,
            'out_of_sample_win_rate': self.out_of_sample_results.win_rate,
            'efficiency_ratio': self.efficiency_ratio,
            'optimized_params': self.optimized_params
        }


@dataclass
class WalkForwardReport:
    """
    Comprehensive report from walk-forward analysis.

    Attributes:
        mode: Walk-forward mode used (anchored or rolling)
        total_windows: Number of walk-forward windows
        results: List of WalkForwardResult for each window
        aggregate_metrics: Aggregated performance metrics across all windows
        combined_equity_curve: Combined out-of-sample equity curve
    """
    mode: WalkForwardMode
    total_windows: int
    results: List[WalkForwardResult]
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    combined_equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            'mode': self.mode.value,
            'total_windows': self.total_windows,
            'aggregate_metrics': self.aggregate_metrics,
            'window_results': [r.to_dict() for r in self.results]
        }


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization and Analysis Framework.

    This class provides methods to perform walk-forward analysis on trading
    strategies, helping to validate strategy robustness and prevent overfitting.

    Walk-forward analysis divides historical data into multiple train/test periods
    and evaluates strategy performance on out-of-sample data, providing a more
    realistic assessment of expected future performance.

    Attributes:
        backtester: Backtester instance for running backtests
        config: BacktestConfig for backtesting parameters
        data: Historical price data (DataFrame with OHLCV)
        symbol: Trading symbol name

    Example Usage:
        >>> from quantsploit.utils.walk_forward import WalkForwardOptimizer
        >>> from quantsploit.utils.backtesting import Backtester, BacktestConfig
        >>>
        >>> # Create optimizer
        >>> config = BacktestConfig(initial_capital=100000)
        >>> optimizer = WalkForwardOptimizer(config, data, symbol='AAPL')
        >>>
        >>> # Run anchored walk-forward
        >>> report = optimizer.run_anchored_walk_forward(
        ...     strategy_func=my_strategy,
        ...     train_window_days=252,  # 1 year training
        ...     test_window_days=63,    # 3 months testing
        ...     step_days=63            # Step forward 3 months
        ... )
        >>>
        >>> # Generate summary report
        >>> summary = optimizer.generate_walk_forward_report(report)
    """

    def __init__(
        self,
        config: BacktestConfig,
        data: pd.DataFrame,
        symbol: str = 'symbol'
    ):
        """
        Initialize the WalkForwardOptimizer.

        Args:
            config: BacktestConfig with backtesting parameters
            data: Historical price DataFrame with DatetimeIndex and OHLCV columns
            symbol: Trading symbol name for position tracking

        Raises:
            ValueError: If data is empty or missing required columns
        """
        self.config = config
        self.symbol = symbol

        # Validate and prepare data
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be empty")

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Data missing required columns: {missing_cols}")

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        self.data = data.sort_index()
        self._results_cache: Dict[str, WalkForwardReport] = {}

    def _generate_windows(
        self,
        mode: WalkForwardMode,
        train_window_days: int,
        test_window_days: int,
        step_days: int,
        min_train_days: Optional[int] = None
    ) -> List[WalkForwardWindow]:
        """
        Generate walk-forward windows based on mode and parameters.

        Args:
            mode: ANCHORED (expanding) or ROLLING (fixed) training window
            train_window_days: Size of training window in trading days
            test_window_days: Size of test window in trading days
            step_days: How many days to step forward between windows
            min_train_days: Minimum training days for anchored mode

        Returns:
            List of WalkForwardWindow objects

        Note:
            For ANCHORED mode, the training window expands from min_train_days
            (or train_window_days if not specified) to include all prior data.
            For ROLLING mode, the training window is always fixed size.
        """
        windows = []

        data_start = self.data.index[0]
        data_end = self.data.index[-1]
        total_days = (data_end - data_start).days

        if min_train_days is None:
            min_train_days = train_window_days

        # Calculate number of windows possible
        if mode == WalkForwardMode.ANCHORED:
            # First test starts after min_train_days
            first_test_start = data_start + timedelta(days=min_train_days)
        else:
            # Rolling: first test starts after train_window_days
            first_test_start = data_start + timedelta(days=train_window_days)

        # Generate windows
        window_id = 0
        current_test_start = first_test_start

        while current_test_start + timedelta(days=test_window_days) <= data_end:
            current_test_end = current_test_start + timedelta(days=test_window_days)

            if mode == WalkForwardMode.ANCHORED:
                # Anchored: training starts from data_start, ends at test_start
                train_start = data_start
                train_end = current_test_start - timedelta(days=1)
            else:
                # Rolling: fixed training window before test
                train_end = current_test_start - timedelta(days=1)
                train_start = train_end - timedelta(days=train_window_days)

                # Ensure we don't go before data start
                if train_start < data_start:
                    train_start = data_start

            window = WalkForwardWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=current_test_start,
                test_end=current_test_end
            )
            windows.append(window)

            window_id += 1
            current_test_start += timedelta(days=step_days)

        logger.info(f"Generated {len(windows)} walk-forward windows "
                   f"({mode.value} mode)")

        return windows

    def _run_backtest_on_window(
        self,
        strategy_func: Callable,
        window_data: pd.DataFrame,
        full_data: pd.DataFrame
    ) -> BacktestResults:
        """
        Run a backtest on a specific data window.

        Args:
            strategy_func: Strategy function (bt, date, row) -> None
            window_data: Data for the specific window period
            full_data: Full historical data (for indicator lookback)

        Returns:
            BacktestResults from the backtest
        """
        backtester = Backtester(deepcopy(self.config))
        results = backtester.run_backtest(
            data=window_data,
            strategy_func=strategy_func,
            symbol=self.symbol
        )
        return results

    def _optimize_parameters(
        self,
        strategy_factory: Callable,
        param_grid: Dict[str, List[Any]],
        train_data: pd.DataFrame,
        full_data: pd.DataFrame,
        optimization_metric: str = 'sharpe_ratio'
    ) -> Tuple[Dict[str, Any], BacktestResults]:
        """
        Optimize strategy parameters on training data.

        Args:
            strategy_factory: Function that takes params and returns strategy_func
            param_grid: Dictionary of parameter names to lists of values
            train_data: Training period data
            full_data: Full historical data for indicator lookback
            optimization_metric: Metric to optimize (sharpe_ratio, total_return_pct, etc.)

        Returns:
            Tuple of (best_params, best_results)

        Note:
            Uses grid search over all parameter combinations. For large grids,
            consider implementing more efficient optimization methods.
        """
        best_params = None
        best_results = None
        best_metric = float('-inf')

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        from itertools import product
        combinations = list(product(*param_values))

        logger.info(f"Optimizing over {len(combinations)} parameter combinations")

        for combo in combinations:
            params = dict(zip(param_names, combo))

            try:
                # Create strategy with these parameters
                strategy_func = strategy_factory(params)

                # Run backtest
                results = self._run_backtest_on_window(
                    strategy_func, train_data, full_data
                )

                # Get optimization metric
                metric_value = getattr(results, optimization_metric, float('-inf'))

                if not np.isnan(metric_value) and metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params
                    best_results = results

            except Exception as e:
                logger.warning(f"Parameter combination {params} failed: {e}")
                continue

        if best_params is None:
            raise ValueError("No valid parameter combination found during optimization")

        logger.info(f"Best parameters: {best_params} "
                   f"({optimization_metric}={best_metric:.4f})")

        return best_params, best_results

    def run_anchored_walk_forward(
        self,
        strategy_func: Callable,
        train_window_days: int = 252,
        test_window_days: int = 63,
        step_days: int = 63,
        min_train_days: Optional[int] = None,
        strategy_factory: Optional[Callable] = None,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        optimization_metric: str = 'sharpe_ratio'
    ) -> WalkForwardReport:
        """
        Run anchored (expanding window) walk-forward analysis.

        In anchored walk-forward, the training window starts from the beginning
        of the data and expands forward over time. This approach uses all
        available historical data for training, which can be beneficial for
        capturing long-term patterns but may be more prone to structural breaks.

        Timeline example (train_window=252, test_window=63, step=63):
        Window 1: Train [Day 0 - Day 251], Test [Day 252 - Day 314]
        Window 2: Train [Day 0 - Day 314], Test [Day 315 - Day 377]
        Window 3: Train [Day 0 - Day 377], Test [Day 378 - Day 440]
        ...

        Args:
            strategy_func: Strategy function with signature (backtester, date, row)
            train_window_days: Initial training window size in trading days
            test_window_days: Test window size in trading days
            step_days: Days to step forward between windows
            min_train_days: Minimum training days (defaults to train_window_days)
            strategy_factory: Optional function that creates strategy from params
            param_grid: Optional parameter grid for optimization
            optimization_metric: Metric to optimize during parameter selection

        Returns:
            WalkForwardReport with results from all windows

        Raises:
            ValueError: If insufficient data for the specified window sizes

        Example:
            >>> report = optimizer.run_anchored_walk_forward(
            ...     strategy_func=my_strategy,
            ...     train_window_days=252,  # 1 year training
            ...     test_window_days=63,    # ~3 months testing
            ...     step_days=63            # Step forward 3 months
            ... )
        """
        # Generate windows
        windows = self._generate_windows(
            mode=WalkForwardMode.ANCHORED,
            train_window_days=train_window_days,
            test_window_days=test_window_days,
            step_days=step_days,
            min_train_days=min_train_days
        )

        if len(windows) == 0:
            raise ValueError(
                f"Insufficient data for walk-forward analysis. "
                f"Data spans {len(self.data)} days, but requires at least "
                f"{train_window_days + test_window_days} days."
            )

        results = []
        do_optimization = strategy_factory is not None and param_grid is not None

        for window in windows:
            logger.info(f"Processing window {window.window_id + 1}/{len(windows)}: "
                       f"Train {window.train_start.date()} to {window.train_end.date()}, "
                       f"Test {window.test_start.date()} to {window.test_end.date()}")

            # Get data slices
            train_data = self.data.loc[window.train_start:window.train_end]
            test_data = self.data.loc[window.test_start:window.test_end]

            if len(train_data) < 30 or len(test_data) < 5:
                logger.warning(f"Skipping window {window.window_id}: insufficient data "
                             f"(train={len(train_data)}, test={len(test_data)})")
                continue

            try:
                optimized_params = None

                if do_optimization:
                    # Optimize parameters on training data
                    optimized_params, in_sample_results = self._optimize_parameters(
                        strategy_factory=strategy_factory,
                        param_grid=param_grid,
                        train_data=train_data,
                        full_data=self.data,
                        optimization_metric=optimization_metric
                    )
                    # Create strategy with optimized parameters for OOS testing
                    current_strategy = strategy_factory(optimized_params)
                else:
                    # Run training backtest with provided strategy
                    in_sample_results = self._run_backtest_on_window(
                        strategy_func, train_data, self.data
                    )
                    current_strategy = strategy_func

                # Run out-of-sample backtest
                out_of_sample_results = self._run_backtest_on_window(
                    current_strategy, test_data, self.data
                )

                result = WalkForwardResult(
                    window=window,
                    in_sample_results=in_sample_results,
                    out_of_sample_results=out_of_sample_results,
                    optimized_params=optimized_params
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error in window {window.window_id}: {e}")
                continue

        # Generate report
        report = WalkForwardReport(
            mode=WalkForwardMode.ANCHORED,
            total_windows=len(windows),
            results=results
        )

        # Calculate aggregate metrics
        report.aggregate_metrics = self._calculate_aggregate_metrics(results)
        report.combined_equity_curve = self._combine_equity_curves(results)

        return report

    def run_rolling_walk_forward(
        self,
        strategy_func: Callable,
        train_window_days: int = 252,
        test_window_days: int = 63,
        step_days: int = 63,
        strategy_factory: Optional[Callable] = None,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        optimization_metric: str = 'sharpe_ratio'
    ) -> WalkForwardReport:
        """
        Run rolling (fixed window) walk-forward analysis.

        In rolling walk-forward, the training window is a fixed size that
        slides forward over time. This approach focuses on more recent data
        and can adapt to changing market conditions, but uses less historical
        data for training.

        Timeline example (train_window=252, test_window=63, step=63):
        Window 1: Train [Day 0 - Day 251], Test [Day 252 - Day 314]
        Window 2: Train [Day 63 - Day 314], Test [Day 315 - Day 377]
        Window 3: Train [Day 126 - Day 377], Test [Day 378 - Day 440]
        ...

        Args:
            strategy_func: Strategy function with signature (backtester, date, row)
            train_window_days: Fixed training window size in trading days
            test_window_days: Test window size in trading days
            step_days: Days to step forward between windows
            strategy_factory: Optional function that creates strategy from params
            param_grid: Optional parameter grid for optimization
            optimization_metric: Metric to optimize during parameter selection

        Returns:
            WalkForwardReport with results from all windows

        Raises:
            ValueError: If insufficient data for the specified window sizes

        Example:
            >>> report = optimizer.run_rolling_walk_forward(
            ...     strategy_func=my_strategy,
            ...     train_window_days=252,  # 1 year rolling training
            ...     test_window_days=63,    # ~3 months testing
            ...     step_days=21            # Step forward monthly
            ... )
        """
        # Generate windows
        windows = self._generate_windows(
            mode=WalkForwardMode.ROLLING,
            train_window_days=train_window_days,
            test_window_days=test_window_days,
            step_days=step_days
        )

        if len(windows) == 0:
            raise ValueError(
                f"Insufficient data for walk-forward analysis. "
                f"Data spans {len(self.data)} days, but requires at least "
                f"{train_window_days + test_window_days} days."
            )

        results = []
        do_optimization = strategy_factory is not None and param_grid is not None

        for window in windows:
            logger.info(f"Processing window {window.window_id + 1}/{len(windows)}: "
                       f"Train {window.train_start.date()} to {window.train_end.date()}, "
                       f"Test {window.test_start.date()} to {window.test_end.date()}")

            # Get data slices
            train_data = self.data.loc[window.train_start:window.train_end]
            test_data = self.data.loc[window.test_start:window.test_end]

            if len(train_data) < 30 or len(test_data) < 5:
                logger.warning(f"Skipping window {window.window_id}: insufficient data "
                             f"(train={len(train_data)}, test={len(test_data)})")
                continue

            try:
                optimized_params = None

                if do_optimization:
                    # Optimize parameters on training data
                    optimized_params, in_sample_results = self._optimize_parameters(
                        strategy_factory=strategy_factory,
                        param_grid=param_grid,
                        train_data=train_data,
                        full_data=self.data,
                        optimization_metric=optimization_metric
                    )
                    # Create strategy with optimized parameters for OOS testing
                    current_strategy = strategy_factory(optimized_params)
                else:
                    # Run training backtest with provided strategy
                    in_sample_results = self._run_backtest_on_window(
                        strategy_func, train_data, self.data
                    )
                    current_strategy = strategy_func

                # Run out-of-sample backtest
                out_of_sample_results = self._run_backtest_on_window(
                    current_strategy, test_data, self.data
                )

                result = WalkForwardResult(
                    window=window,
                    in_sample_results=in_sample_results,
                    out_of_sample_results=out_of_sample_results,
                    optimized_params=optimized_params
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error in window {window.window_id}: {e}")
                continue

        # Generate report
        report = WalkForwardReport(
            mode=WalkForwardMode.ROLLING,
            total_windows=len(windows),
            results=results
        )

        # Calculate aggregate metrics
        report.aggregate_metrics = self._calculate_aggregate_metrics(results)
        report.combined_equity_curve = self._combine_equity_curves(results)

        return report

    def _calculate_aggregate_metrics(
        self,
        results: List[WalkForwardResult]
    ) -> Dict[str, float]:
        """
        Calculate aggregate metrics across all walk-forward windows.

        Args:
            results: List of WalkForwardResult from all windows

        Returns:
            Dictionary of aggregate metrics
        """
        if not results:
            return {}

        # Extract metrics
        is_returns = [r.in_sample_results.total_return_pct for r in results]
        oos_returns = [r.out_of_sample_results.total_return_pct for r in results]
        is_sharpes = [r.in_sample_results.sharpe_ratio for r in results]
        oos_sharpes = [r.out_of_sample_results.sharpe_ratio for r in results]
        is_drawdowns = [r.in_sample_results.max_drawdown for r in results]
        oos_drawdowns = [r.out_of_sample_results.max_drawdown for r in results]
        is_win_rates = [r.in_sample_results.win_rate for r in results]
        oos_win_rates = [r.out_of_sample_results.win_rate for r in results]
        efficiency_ratios = [r.efficiency_ratio for r in results
                           if not np.isnan(r.efficiency_ratio)]

        # Calculate aggregates
        metrics = {
            # In-Sample Statistics
            'in_sample_avg_return': np.mean(is_returns),
            'in_sample_std_return': np.std(is_returns),
            'in_sample_avg_sharpe': np.mean([s for s in is_sharpes if not np.isnan(s)]),
            'in_sample_avg_max_dd': np.mean(is_drawdowns),
            'in_sample_avg_win_rate': np.mean(is_win_rates),

            # Out-of-Sample Statistics
            'out_of_sample_avg_return': np.mean(oos_returns),
            'out_of_sample_std_return': np.std(oos_returns),
            'out_of_sample_avg_sharpe': np.mean([s for s in oos_sharpes if not np.isnan(s)]),
            'out_of_sample_avg_max_dd': np.mean(oos_drawdowns),
            'out_of_sample_avg_win_rate': np.mean(oos_win_rates),

            # Cumulative Out-of-Sample Return
            'cumulative_oos_return': self._calculate_cumulative_return(oos_returns),

            # Walk-Forward Efficiency
            'avg_efficiency_ratio': np.mean(efficiency_ratios) if efficiency_ratios else np.nan,
            'efficiency_ratio_std': np.std(efficiency_ratios) if efficiency_ratios else np.nan,

            # Robustness Metrics
            'oos_positive_windows': sum(1 for r in oos_returns if r > 0),
            'oos_negative_windows': sum(1 for r in oos_returns if r <= 0),
            'oos_positive_rate': sum(1 for r in oos_returns if r > 0) / len(oos_returns) * 100,

            # Performance Degradation
            'avg_return_degradation': np.mean(is_returns) - np.mean(oos_returns),
            'sharpe_degradation': (np.mean([s for s in is_sharpes if not np.isnan(s)]) -
                                  np.mean([s for s in oos_sharpes if not np.isnan(s)])),

            # Total windows analyzed
            'total_windows_analyzed': len(results)
        }

        return metrics

    def _calculate_cumulative_return(self, returns: List[float]) -> float:
        """
        Calculate cumulative return from a series of period returns.

        Args:
            returns: List of percentage returns

        Returns:
            Cumulative return as a percentage
        """
        if not returns:
            return 0.0

        cumulative = 1.0
        for r in returns:
            cumulative *= (1 + r / 100)

        return (cumulative - 1) * 100

    def _combine_equity_curves(
        self,
        results: List[WalkForwardResult]
    ) -> pd.DataFrame:
        """
        Combine out-of-sample equity curves from all windows.

        This creates a continuous equity curve by chaining the out-of-sample
        periods together, providing a realistic view of strategy performance.

        Args:
            results: List of WalkForwardResult from all windows

        Returns:
            DataFrame with combined equity curve
        """
        if not results:
            return pd.DataFrame()

        combined_curves = []
        last_equity = self.config.initial_capital

        for result in results:
            oos_curve = result.out_of_sample_results.equity_curve.copy()

            if len(oos_curve) == 0:
                continue

            # Scale equity curve to start from last_equity
            if 'equity' in oos_curve.columns:
                initial_equity = oos_curve['equity'].iloc[0]
                if initial_equity > 0:
                    scale_factor = last_equity / initial_equity
                    oos_curve['equity'] = oos_curve['equity'] * scale_factor
                    oos_curve['window_id'] = result.window.window_id
                    combined_curves.append(oos_curve)
                    last_equity = oos_curve['equity'].iloc[-1]

        if combined_curves:
            return pd.concat(combined_curves).sort_index()

        return pd.DataFrame()

    def generate_walk_forward_report(
        self,
        report: WalkForwardReport
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report from walk-forward analysis.

        This method produces a detailed report suitable for display or export,
        including overall statistics, window-by-window breakdown, and
        interpretation guidance.

        Args:
            report: WalkForwardReport from anchored or rolling walk-forward

        Returns:
            Dictionary containing comprehensive analysis summary

        Example:
            >>> report = optimizer.run_anchored_walk_forward(strategy_func, ...)
            >>> summary = optimizer.generate_walk_forward_report(report)
            >>> print(summary['interpretation'])
        """
        if not report.results:
            return {
                'error': 'No valid results to report',
                'total_windows': report.total_windows
            }

        metrics = report.aggregate_metrics

        # Generate interpretation
        interpretation = self._generate_interpretation(metrics)

        # Format window results
        window_details = []
        for result in report.results:
            window_details.append({
                'window': f"Window {result.window.window_id + 1}",
                'train_period': f"{result.window.train_start.date()} to {result.window.train_end.date()}",
                'test_period': f"{result.window.test_start.date()} to {result.window.test_end.date()}",
                'in_sample_return': f"{result.in_sample_results.total_return_pct:.2f}%",
                'out_of_sample_return': f"{result.out_of_sample_results.total_return_pct:.2f}%",
                'in_sample_sharpe': f"{result.in_sample_results.sharpe_ratio:.3f}",
                'out_of_sample_sharpe': f"{result.out_of_sample_results.sharpe_ratio:.3f}",
                'efficiency_ratio': f"{result.efficiency_ratio:.3f}" if not np.isnan(result.efficiency_ratio) else "N/A",
                'oos_trades': result.out_of_sample_results.total_trades,
                'oos_win_rate': f"{result.out_of_sample_results.win_rate:.1f}%",
                'optimized_params': result.optimized_params
            })

        summary = {
            'mode': report.mode.value,
            'total_windows': report.total_windows,
            'windows_analyzed': len(report.results),

            'overall_performance': {
                'cumulative_oos_return': f"{metrics.get('cumulative_oos_return', 0):.2f}%",
                'avg_oos_return_per_window': f"{metrics.get('out_of_sample_avg_return', 0):.2f}%",
                'oos_return_std': f"{metrics.get('out_of_sample_std_return', 0):.2f}%",
                'avg_oos_sharpe': f"{metrics.get('out_of_sample_avg_sharpe', 0):.3f}",
                'avg_oos_max_drawdown': f"{metrics.get('out_of_sample_avg_max_dd', 0):.2f}%",
                'oos_positive_rate': f"{metrics.get('oos_positive_rate', 0):.1f}%"
            },

            'walk_forward_efficiency': {
                'avg_efficiency_ratio': f"{metrics.get('avg_efficiency_ratio', 0):.3f}" if not np.isnan(metrics.get('avg_efficiency_ratio', np.nan)) else "N/A",
                'efficiency_std': f"{metrics.get('efficiency_ratio_std', 0):.3f}" if not np.isnan(metrics.get('efficiency_ratio_std', np.nan)) else "N/A",
                'return_degradation': f"{metrics.get('avg_return_degradation', 0):.2f}%",
                'sharpe_degradation': f"{metrics.get('sharpe_degradation', 0):.3f}"
            },

            'in_sample_summary': {
                'avg_return': f"{metrics.get('in_sample_avg_return', 0):.2f}%",
                'avg_sharpe': f"{metrics.get('in_sample_avg_sharpe', 0):.3f}",
                'avg_max_drawdown': f"{metrics.get('in_sample_avg_max_dd', 0):.2f}%",
                'avg_win_rate': f"{metrics.get('in_sample_avg_win_rate', 0):.1f}%"
            },

            'window_details': window_details,
            'interpretation': interpretation,
            'raw_metrics': metrics
        }

        return summary

    def _generate_interpretation(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Generate human-readable interpretation of walk-forward metrics.

        Args:
            metrics: Aggregate metrics dictionary

        Returns:
            Dictionary with interpretation strings
        """
        interpretations = {}

        # Efficiency ratio interpretation
        efficiency = metrics.get('avg_efficiency_ratio', np.nan)
        if not np.isnan(efficiency):
            if efficiency >= 0.8:
                interpretations['efficiency'] = (
                    f"EXCELLENT: Average efficiency ratio of {efficiency:.3f} suggests "
                    "the strategy maintains strong performance on unseen data. "
                    "Low risk of overfitting."
                )
            elif efficiency >= 0.5:
                interpretations['efficiency'] = (
                    f"GOOD: Average efficiency ratio of {efficiency:.3f} indicates "
                    "reasonable out-of-sample performance retention. "
                    "Strategy appears robust with moderate optimization."
                )
            elif efficiency >= 0.3:
                interpretations['efficiency'] = (
                    f"MODERATE: Average efficiency ratio of {efficiency:.3f} shows "
                    "noticeable performance degradation out-of-sample. "
                    "Consider simplifying the strategy or reducing parameters."
                )
            else:
                interpretations['efficiency'] = (
                    f"POOR: Average efficiency ratio of {efficiency:.3f} indicates "
                    "significant overfitting. Out-of-sample performance is much "
                    "worse than in-sample. Strategy needs revision."
                )

        # Consistency interpretation
        positive_rate = metrics.get('oos_positive_rate', 0)
        if positive_rate >= 70:
            interpretations['consistency'] = (
                f"HIGHLY CONSISTENT: {positive_rate:.1f}% of windows produced "
                "positive out-of-sample returns. Strategy performs well across "
                "different market conditions."
            )
        elif positive_rate >= 50:
            interpretations['consistency'] = (
                f"MODERATELY CONSISTENT: {positive_rate:.1f}% of windows produced "
                "positive returns. Strategy has mixed but net positive performance."
            )
        else:
            interpretations['consistency'] = (
                f"INCONSISTENT: Only {positive_rate:.1f}% of windows produced "
                "positive returns. Strategy may not be reliable in different "
                "market conditions."
            )

        # Overall recommendation
        cum_return = metrics.get('cumulative_oos_return', 0)
        avg_sharpe = metrics.get('out_of_sample_avg_sharpe', 0)

        if cum_return > 0 and avg_sharpe > 0.5 and positive_rate >= 50:
            interpretations['recommendation'] = (
                "VIABLE: The strategy shows positive cumulative out-of-sample "
                "returns with acceptable risk-adjusted performance. Consider "
                "for live trading with appropriate position sizing."
            )
        elif cum_return > 0:
            interpretations['recommendation'] = (
                "MARGINAL: The strategy produces positive returns but with "
                "suboptimal risk characteristics. Further refinement recommended "
                "before live deployment."
            )
        else:
            interpretations['recommendation'] = (
                "NOT RECOMMENDED: The strategy fails to produce positive "
                "out-of-sample returns. Significant changes or a different "
                "approach may be needed."
            )

        return interpretations


def calculate_walk_forward_efficiency(
    in_sample_returns: List[float],
    out_of_sample_returns: List[float]
) -> Tuple[float, float]:
    """
    Calculate walk-forward efficiency ratio and its standard deviation.

    Walk-Forward Efficiency (WFE) measures how well a strategy's in-sample
    performance translates to out-of-sample performance. A ratio close to 1.0
    indicates the strategy is not overfit.

    WFE = Average(OOS Return / IS Return)

    Args:
        in_sample_returns: List of in-sample period returns
        out_of_sample_returns: List of corresponding out-of-sample returns

    Returns:
        Tuple of (average_efficiency, efficiency_std)

    Example:
        >>> is_returns = [10.5, 8.2, 12.1, 9.8]
        >>> oos_returns = [7.2, 5.1, 9.8, 6.2]
        >>> efficiency, std = calculate_walk_forward_efficiency(is_returns, oos_returns)
        >>> print(f"WFE: {efficiency:.2f} +/- {std:.2f}")
    """
    if len(in_sample_returns) != len(out_of_sample_returns):
        raise ValueError("In-sample and out-of-sample return lists must have same length")

    efficiencies = []
    for is_ret, oos_ret in zip(in_sample_returns, out_of_sample_returns):
        if is_ret != 0 and not np.isnan(is_ret) and not np.isnan(oos_ret):
            if abs(is_ret) > 0.01:  # Avoid division by very small numbers
                efficiencies.append(oos_ret / is_ret)

    if not efficiencies:
        return np.nan, np.nan

    return np.mean(efficiencies), np.std(efficiencies)


# =============================================================================
# PurgedKFoldCV - Time-Series Cross-Validation for ML
# =============================================================================

@dataclass
class PurgedKFoldSplit:
    """
    Represents a single train/test split from PurgedKFoldCV.

    Attributes:
        fold_id: Fold number (0-indexed)
        train_indices: Array of training sample indices
        test_indices: Array of test sample indices
        purge_indices: Array of purged sample indices (excluded from both)
        embargo_indices: Array of embargo sample indices (excluded from train)
    """
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    purge_indices: np.ndarray
    embargo_indices: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.train_indices)

    @property
    def n_test(self) -> int:
        return len(self.test_indices)

    @property
    def n_purged(self) -> int:
        return len(self.purge_indices)

    @property
    def n_embargo(self) -> int:
        return len(self.embargo_indices)


class PurgedKFoldCV:
    """
    Purged K-Fold Cross-Validation for Time-Series Data.

    This cross-validator is designed specifically for financial ML applications
    where standard K-Fold CV can lead to data leakage due to:
    1. Overlapping information between train/test samples
    2. Autocorrelation in time-series data
    3. Feature labels computed from future data

    Key Features:
    - **Purge Gap**: Removes samples between train/test to prevent information leakage
    - **Embargo Period**: Excludes samples after test set from training to account for
      autocorrelation decay
    - **Combinatorial Purging**: Properly handles overlapping samples

    ★ Insight ─────────────────────────────────────
    Standard K-Fold CV for financial ML is DANGEROUS because:
    - Features often include lagged values that span multiple bars
    - Labels (returns) are computed from future prices
    - Financial data exhibits strong autocorrelation

    PurgedKFoldCV addresses these by adding "buffer zones" around test folds.
    ─────────────────────────────────────────────────

    Reference:
    - De Prado, M.L. (2018). Advances in Financial Machine Learning, Ch. 7

    Args:
        n_splits: Number of folds (default 5)
        purge_gap: Number of samples to purge between train/test boundaries
                   Set this >= max feature lookback period
        embargo_pct: Percentage of test samples to embargo after test set (0.0 to 0.5)
                     Accounts for autocorrelation decay time

    Example:
        >>> from quantsploit.utils.walk_forward import PurgedKFoldCV
        >>> import numpy as np
        >>>
        >>> # Create CV with 5 folds, 10-day purge, 1% embargo
        >>> cv = PurgedKFoldCV(n_splits=5, purge_gap=10, embargo_pct=0.01)
        >>>
        >>> # Generate splits
        >>> X = np.random.randn(1000, 10)  # 1000 samples, 10 features
        >>> for train_idx, test_idx in cv.split(X):
        ...     X_train, X_test = X[train_idx], X[test_idx]
        ...     # Train and evaluate model
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 0,
        embargo_pct: float = 0.0
    ):
        """
        Initialize PurgedKFoldCV.

        Args:
            n_splits: Number of cross-validation folds (>= 2)
            purge_gap: Number of samples to exclude between train and test sets
                       This should be >= your maximum feature lookback window
            embargo_pct: Percentage of test samples to embargo (0.0 to 0.5)
                         This accounts for label overlap and autocorrelation
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if purge_gap < 0:
            raise ValueError(f"purge_gap must be >= 0, got {purge_gap}")
        if not 0.0 <= embargo_pct <= 0.5:
            raise ValueError(f"embargo_pct must be in [0.0, 0.5], got {embargo_pct}")

        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[np.ndarray] = None
    ):
        """
        Generate train/test indices for purged K-fold cross-validation.

        Args:
            X: Feature matrix with shape (n_samples, n_features)
            y: Target array (optional, not used but kept for sklearn compatibility)
            groups: Group labels (optional, not used)

        Yields:
            (train_indices, test_indices) tuples for each fold

        Note:
            Unlike standard K-Fold, this yields arrays of indices rather than
            boolean masks, and the train set may have "gaps" where purged and
            embargoed samples are excluded.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate fold sizes
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        # Calculate embargo size (same for all folds)
        test_fold_size = fold_sizes[0]  # Approximate
        embargo_size = int(np.ceil(test_fold_size * self.embargo_pct))

        current = 0
        for fold_id in range(self.n_splits):
            fold_size = fold_sizes[fold_id]

            # Test indices for this fold
            test_start = current
            test_end = current + fold_size
            test_indices = indices[test_start:test_end]

            # Purge indices: samples between train and test that overlap
            purge_before_start = max(0, test_start - self.purge_gap)
            purge_before_end = test_start
            purge_after_start = test_end
            purge_after_end = min(n_samples, test_end + self.purge_gap)

            purge_before = set(range(purge_before_start, purge_before_end))
            purge_after = set(range(purge_after_start, purge_after_end))
            purge_indices = purge_before | purge_after

            # Embargo indices: samples after test set (to account for autocorrelation)
            embargo_start = purge_after_end
            embargo_end = min(n_samples, embargo_start + embargo_size)
            embargo_indices = set(range(embargo_start, embargo_end))

            # Training indices: everything except test, purge, and embargo
            test_set = set(test_indices)
            excluded = test_set | purge_indices | embargo_indices
            train_indices = np.array([i for i in indices if i not in excluded])

            current += fold_size

            yield train_indices, test_indices

    def split_detailed(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> List[PurgedKFoldSplit]:
        """
        Generate detailed split information including purge and embargo indices.

        This method provides more detailed information about each split,
        useful for visualization and debugging.

        Args:
            X: Feature matrix with shape (n_samples, n_features)
            y: Target array (optional)

        Returns:
            List of PurgedKFoldSplit objects with detailed split information
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate fold sizes
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        # Calculate embargo size
        test_fold_size = fold_sizes[0]
        embargo_size = int(np.ceil(test_fold_size * self.embargo_pct))

        splits = []
        current = 0

        for fold_id in range(self.n_splits):
            fold_size = fold_sizes[fold_id]

            # Test indices
            test_start = current
            test_end = current + fold_size
            test_indices = indices[test_start:test_end]

            # Purge indices
            purge_before_start = max(0, test_start - self.purge_gap)
            purge_before_end = test_start
            purge_after_start = test_end
            purge_after_end = min(n_samples, test_end + self.purge_gap)

            purge_before = list(range(purge_before_start, purge_before_end))
            purge_after = list(range(purge_after_start, purge_after_end))
            purge_indices = np.array(purge_before + purge_after)

            # Embargo indices
            embargo_start = purge_after_end
            embargo_end = min(n_samples, embargo_start + embargo_size)
            embargo_indices = np.array(list(range(embargo_start, embargo_end)))

            # Training indices
            test_set = set(test_indices)
            purge_set = set(purge_indices)
            embargo_set = set(embargo_indices)
            excluded = test_set | purge_set | embargo_set
            train_indices = np.array([i for i in indices if i not in excluded])

            split = PurgedKFoldSplit(
                fold_id=fold_id,
                train_indices=train_indices,
                test_indices=test_indices,
                purge_indices=purge_indices,
                embargo_indices=embargo_indices
            )
            splits.append(split)

            current += fold_size

        return splits

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Return the number of splits (sklearn compatibility)."""
        return self.n_splits

    def visualize_splits(
        self,
        n_samples: int,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Visualize the cross-validation splits using matplotlib.

        Args:
            n_samples: Number of samples to visualize
            figsize: Figure size (width, height)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.warning("matplotlib not available for visualization")
            return

        X_dummy = np.zeros((n_samples, 1))
        splits = self.split_detailed(X_dummy)

        fig, ax = plt.subplots(figsize=figsize)

        cmap = {
            'train': '#1f77b4',    # Blue
            'test': '#2ca02c',     # Green
            'purge': '#ff7f0e',    # Orange
            'embargo': '#d62728'   # Red
        }

        for fold_id, split in enumerate(splits):
            y_pos = fold_id

            # Create color array for this fold
            colors = np.full(n_samples, '#e0e0e0')  # Default gray (unused)

            # Fill in the colors
            colors[split.train_indices] = cmap['train']
            colors[split.test_indices] = cmap['test']
            colors[split.purge_indices] = cmap['purge']
            colors[split.embargo_indices] = cmap['embargo']

            # Plot as horizontal bar segments
            for i in range(n_samples):
                ax.barh(y_pos, width=1, left=i, height=0.8,
                       color=colors[i], edgecolor='none')

        # Create legend
        legend_patches = [
            mpatches.Patch(color=cmap['train'], label=f'Train'),
            mpatches.Patch(color=cmap['test'], label=f'Test'),
            mpatches.Patch(color=cmap['purge'], label=f'Purge (gap={self.purge_gap})'),
            mpatches.Patch(color=cmap['embargo'], label=f'Embargo ({self.embargo_pct:.1%})')
        ]
        ax.legend(handles=legend_patches, loc='upper right')

        ax.set_yticks(range(self.n_splits))
        ax.set_yticklabels([f'Fold {i+1}' for i in range(self.n_splits)])
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Fold')
        ax.set_title(f'PurgedKFoldCV Splits (n_splits={self.n_splits})')
        ax.set_xlim(0, n_samples)

        plt.tight_layout()
        plt.show()

    def summary(self, n_samples: int) -> Dict[str, Any]:
        """
        Generate a summary of the cross-validation splits.

        Args:
            n_samples: Number of samples

        Returns:
            Dictionary with split statistics
        """
        X_dummy = np.zeros((n_samples, 1))
        splits = self.split_detailed(X_dummy)

        train_sizes = [s.n_train for s in splits]
        test_sizes = [s.n_test for s in splits]
        purge_sizes = [s.n_purged for s in splits]
        embargo_sizes = [s.n_embargo for s in splits]

        total_excluded = sum(purge_sizes) + sum(embargo_sizes)
        unique_excluded = len(set(
            list(np.concatenate([s.purge_indices for s in splits])) +
            list(np.concatenate([s.embargo_indices for s in splits]))
        ))

        return {
            'n_splits': self.n_splits,
            'n_samples': n_samples,
            'purge_gap': self.purge_gap,
            'embargo_pct': self.embargo_pct,
            'avg_train_size': np.mean(train_sizes),
            'avg_test_size': np.mean(test_sizes),
            'avg_purge_size': np.mean(purge_sizes),
            'avg_embargo_size': np.mean(embargo_sizes),
            'train_pct': np.mean(train_sizes) / n_samples * 100,
            'test_pct': np.mean(test_sizes) / n_samples * 100,
            'excluded_pct': unique_excluded / n_samples * 100,
            'fold_details': [
                {
                    'fold': i,
                    'train': s.n_train,
                    'test': s.n_test,
                    'purge': s.n_purged,
                    'embargo': s.n_embargo
                }
                for i, s in enumerate(splits)
            ]
        }


class CombinatorialPurgedKFoldCV:
    """
    Combinatorial Purged K-Fold Cross-Validation.

    Extends PurgedKFoldCV by generating all possible combinations of k test
    folds out of n total folds, providing more test paths for backtesting
    validation.

    This is particularly useful for:
    - Strategy backtesting where you want multiple out-of-sample paths
    - Ensemble methods that combine multiple train/test combinations
    - More robust performance estimation with limited data

    ★ Insight ─────────────────────────────────────
    With n_splits=5 and n_test_splits=2, you get C(5,2)=10 different
    train/test configurations, each with 2/5 of data as test. This
    provides much more robust performance estimates than standard K-Fold.
    ─────────────────────────────────────────────────

    Args:
        n_splits: Total number of folds to create
        n_test_splits: Number of folds to use as test set (default 2)
        purge_gap: Number of samples to purge between train/test
        embargo_pct: Percentage of test samples to embargo

    Example:
        >>> cv = CombinatorialPurgedKFoldCV(n_splits=5, n_test_splits=2, purge_gap=5)
        >>> for train_idx, test_idx in cv.split(X):
        ...     # 10 different train/test combinations
        ...     model.fit(X[train_idx], y[train_idx])
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_gap: int = 0,
        embargo_pct: float = 0.0
    ):
        if n_test_splits >= n_splits:
            raise ValueError(f"n_test_splits ({n_test_splits}) must be < n_splits ({n_splits})")

        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ):
        """
        Generate combinatorial train/test indices.

        Args:
            X: Feature matrix
            y: Target array (optional)
            groups: Group labels (optional)

        Yields:
            (train_indices, test_indices) tuples for each combination
        """
        from itertools import combinations

        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate fold boundaries
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        # Create fold index ranges
        fold_ranges = []
        current = 0
        for size in fold_sizes:
            fold_ranges.append((current, current + size))
            current += size

        # Calculate embargo size
        avg_test_size = sum(fold_sizes[i] for i in range(self.n_test_splits)) // self.n_test_splits
        embargo_size = int(np.ceil(avg_test_size * self.embargo_pct))

        # Generate all combinations of test folds
        for test_fold_combo in combinations(range(self.n_splits), self.n_test_splits):
            # Test indices: union of selected folds
            test_indices = []
            for fold_id in test_fold_combo:
                start, end = fold_ranges[fold_id]
                test_indices.extend(range(start, end))
            test_indices = np.array(test_indices)
            test_set = set(test_indices)

            # Purge and embargo indices
            purge_set = set()
            embargo_set = set()

            for fold_id in test_fold_combo:
                start, end = fold_ranges[fold_id]

                # Purge before test fold
                purge_start = max(0, start - self.purge_gap)
                for i in range(purge_start, start):
                    if i not in test_set:
                        purge_set.add(i)

                # Purge after test fold
                purge_end = min(n_samples, end + self.purge_gap)
                for i in range(end, purge_end):
                    if i not in test_set:
                        purge_set.add(i)

                # Embargo after purge
                embargo_start = purge_end
                embargo_end = min(n_samples, embargo_start + embargo_size)
                for i in range(embargo_start, embargo_end):
                    if i not in test_set and i not in purge_set:
                        embargo_set.add(i)

            # Training indices
            excluded = test_set | purge_set | embargo_set
            train_indices = np.array([i for i in indices if i not in excluded])

            yield train_indices, test_indices

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Return the number of unique combinations."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)


def run_quick_walk_forward(
    data: pd.DataFrame,
    strategy_func: Callable,
    symbol: str = 'symbol',
    mode: str = 'rolling',
    train_days: int = 252,
    test_days: int = 63,
    step_days: int = 63,
    initial_capital: float = 100000.0
) -> Dict[str, Any]:
    """
    Convenience function for quick walk-forward analysis.

    Provides a simple interface for running walk-forward analysis without
    manually configuring all components.

    Args:
        data: Historical OHLCV DataFrame with DatetimeIndex
        strategy_func: Strategy function (backtester, date, row) -> None
        symbol: Trading symbol name
        mode: 'rolling' or 'anchored'
        train_days: Training window size in days
        test_days: Test window size in days
        step_days: Step size between windows
        initial_capital: Starting capital

    Returns:
        Dictionary with walk-forward analysis results

    Example:
        >>> results = run_quick_walk_forward(
        ...     data=price_data,
        ...     strategy_func=my_strategy,
        ...     symbol='AAPL',
        ...     mode='rolling',
        ...     train_days=252,
        ...     test_days=63
        ... )
        >>> print(f"Cumulative OOS Return: {results['cumulative_oos_return']}")
    """
    config = BacktestConfig(initial_capital=initial_capital)
    optimizer = WalkForwardOptimizer(config, data, symbol)

    if mode.lower() == 'anchored':
        report = optimizer.run_anchored_walk_forward(
            strategy_func=strategy_func,
            train_window_days=train_days,
            test_window_days=test_days,
            step_days=step_days
        )
    else:
        report = optimizer.run_rolling_walk_forward(
            strategy_func=strategy_func,
            train_window_days=train_days,
            test_window_days=test_days,
            step_days=step_days
        )

    return optimizer.generate_walk_forward_report(report)


# =============================================================================
# BLOCK BOOTSTRAP FOR TIME SERIES
# =============================================================================

@dataclass
class BlockBootstrapResult:
    """
    Results from block bootstrap analysis.

    Attributes:
        statistic_name: Name of the statistic being bootstrapped
        original_statistic: Statistic from the original sample
        bootstrap_mean: Mean of bootstrap distribution
        bootstrap_std: Standard deviation of bootstrap distribution
        confidence_interval: Tuple of (lower, upper) CI bounds
        confidence_level: Confidence level used (e.g., 0.95)
        p_value: P-value for hypothesis test (if applicable)
        n_bootstrap: Number of bootstrap samples
        block_length: Block length used
        bootstrap_distribution: Array of bootstrap statistics
    """
    statistic_name: str
    original_statistic: float
    bootstrap_mean: float
    bootstrap_std: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    p_value: Optional[float]
    n_bootstrap: int
    block_length: int
    bootstrap_distribution: np.ndarray


class BlockBootstrap:
    """
    Block Bootstrap for Time Series Data.

    Block bootstrap preserves temporal dependence in time series data
    by resampling contiguous blocks rather than individual observations.
    This is essential for financial time series which exhibit serial
    correlation and volatility clustering.

    Supports multiple block bootstrap methods:
    - Non-overlapping block bootstrap (NBB)
    - Moving block bootstrap (MBB)
    - Circular block bootstrap (CBB)
    - Stationary bootstrap (Politis & Romano, 1994)

    ★ Insight ─────────────────────────────────────
    Why block bootstrap for finance?
    - Returns exhibit autocorrelation (momentum, mean reversion)
    - Volatility is clustered (GARCH effects)
    - Standard bootstrap destroys these dependencies
    - Block bootstrap preserves within-block dependence structure
    ─────────────────────────────────────────────────

    Example:
        >>> bb = BlockBootstrap(returns, block_length=20)
        >>> result = bb.bootstrap_statistic(np.mean, n_bootstrap=10000)
        >>> print(f"95% CI: {result.confidence_interval}")

    References:
        - Künsch, H.R. (1989). "The Jackknife and the Bootstrap for General
          Stationary Observations". Annals of Statistics.
        - Politis, D. & Romano, J. (1994). "The Stationary Bootstrap".
          Journal of the American Statistical Association.
        - Lahiri, S.N. (2003). "Resampling Methods for Dependent Data". Springer.
    """

    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame, np.ndarray],
        block_length: Optional[int] = None,
        method: str = 'moving',
        seed: Optional[int] = None
    ):
        """
        Initialize Block Bootstrap.

        Args:
            data: Time series data (1D or 2D)
            block_length: Length of blocks. If None, automatically selected
                         using optimal block length methods.
            method: Bootstrap method:
                   - 'moving': Moving block bootstrap (default)
                   - 'circular': Circular block bootstrap
                   - 'nonoverlapping': Non-overlapping block bootstrap
                   - 'stationary': Stationary bootstrap (random block lengths)
            seed: Random seed for reproducibility
        """
        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            self.data = data.values
            self.columns = list(data.columns)
            self.is_multivariate = True
        elif isinstance(data, pd.Series):
            self.data = data.values
            self.columns = [data.name] if data.name else ['value']
            self.is_multivariate = False
        else:
            self.data = np.array(data)
            self.columns = None
            self.is_multivariate = len(self.data.shape) > 1

        if self.is_multivariate and len(self.data.shape) == 1:
            self.data = self.data.reshape(-1, 1)

        self.n = len(self.data)
        self.method = method.lower()

        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

        # Set or estimate block length
        if block_length is not None:
            self.block_length = block_length
        else:
            self.block_length = self._optimal_block_length()

        # For stationary bootstrap, block_length is expected (mean) length
        self.p = 1 / self.block_length  # Geometric probability for stationary

    def _optimal_block_length(self) -> int:
        """
        Estimate optimal block length.

        Uses Politis & White (2004) automatic block length selection
        based on the spectral density of the data.

        For simplicity, uses the rule of thumb: n^(1/3) for variance estimation
        and n^(1/4) for distribution estimation.
        """
        # Rule of thumb: n^(1/3) is often used
        # More sophisticated methods could use spectral density estimation
        b_opt = int(np.ceil(self.n ** (1/3)))

        # Ensure reasonable bounds
        b_opt = max(5, min(b_opt, self.n // 3))

        return b_opt

    def _generate_moving_block_indices(self) -> np.ndarray:
        """Generate indices using moving block bootstrap."""
        n_blocks = int(np.ceil(self.n / self.block_length))
        indices = []

        # Sample starting positions
        max_start = self.n - self.block_length + 1
        if max_start <= 0:
            max_start = 1

        starts = np.random.randint(0, max_start, n_blocks)

        for start in starts:
            end = min(start + self.block_length, self.n)
            indices.extend(range(start, end))

        return np.array(indices[:self.n])

    def _generate_circular_block_indices(self) -> np.ndarray:
        """Generate indices using circular block bootstrap."""
        n_blocks = int(np.ceil(self.n / self.block_length))
        indices = []

        # Sample starting positions (can wrap around)
        starts = np.random.randint(0, self.n, n_blocks)

        for start in starts:
            for i in range(self.block_length):
                indices.append((start + i) % self.n)

        return np.array(indices[:self.n])

    def _generate_nonoverlapping_block_indices(self) -> np.ndarray:
        """Generate indices using non-overlapping block bootstrap."""
        n_complete_blocks = self.n // self.block_length
        n_blocks_needed = int(np.ceil(self.n / self.block_length))

        # Sample from available complete blocks
        block_starts = np.arange(0, n_complete_blocks * self.block_length, self.block_length)
        selected_blocks = np.random.choice(block_starts, n_blocks_needed, replace=True)

        indices = []
        for start in selected_blocks:
            end = min(start + self.block_length, self.n)
            indices.extend(range(start, end))

        return np.array(indices[:self.n])

    def _generate_stationary_indices(self) -> np.ndarray:
        """
        Generate indices using stationary bootstrap (Politis & Romano).

        Block lengths are random, drawn from geometric distribution.
        This produces strictly stationary bootstrap samples.
        """
        indices = []
        current_pos = np.random.randint(0, self.n)  # Random start

        while len(indices) < self.n:
            # Add current position
            indices.append(current_pos)

            # Decide whether to start new block (with probability p)
            if np.random.random() < self.p:
                # Start new block at random position
                current_pos = np.random.randint(0, self.n)
            else:
                # Continue current block (circular)
                current_pos = (current_pos + 1) % self.n

        return np.array(indices[:self.n])

    def generate_bootstrap_sample(self) -> np.ndarray:
        """
        Generate a single bootstrap sample.

        Returns:
            Bootstrap sample as numpy array
        """
        if self.method == 'moving':
            indices = self._generate_moving_block_indices()
        elif self.method == 'circular':
            indices = self._generate_circular_block_indices()
        elif self.method == 'nonoverlapping':
            indices = self._generate_nonoverlapping_block_indices()
        elif self.method == 'stationary':
            indices = self._generate_stationary_indices()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self.data[indices]

    def bootstrap_statistic(
        self,
        statistic_func: Callable,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
        statistic_name: str = 'statistic'
    ) -> BlockBootstrapResult:
        """
        Bootstrap any statistic with block bootstrap.

        Args:
            statistic_func: Function that takes data array and returns scalar
            n_bootstrap: Number of bootstrap replications
            confidence_level: Confidence level for CI (e.g., 0.95)
            statistic_name: Name for reporting

        Returns:
            BlockBootstrapResult with bootstrap distribution and CI

        Example:
            >>> # Bootstrap Sharpe ratio
            >>> def sharpe_ratio(returns):
            ...     return np.mean(returns) / np.std(returns) * np.sqrt(252)
            >>> result = bb.bootstrap_statistic(sharpe_ratio, n_bootstrap=10000)
        """
        # Calculate original statistic
        original_stat = statistic_func(self.data)

        # Generate bootstrap distribution
        bootstrap_stats = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            bootstrap_sample = self.generate_bootstrap_sample()
            bootstrap_stats[i] = statistic_func(bootstrap_sample)

        # Calculate statistics of bootstrap distribution
        bootstrap_mean = np.mean(bootstrap_stats)
        bootstrap_std = np.std(bootstrap_stats)

        # Confidence interval (percentile method)
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

        return BlockBootstrapResult(
            statistic_name=statistic_name,
            original_statistic=original_stat,
            bootstrap_mean=bootstrap_mean,
            bootstrap_std=bootstrap_std,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=confidence_level,
            p_value=None,
            n_bootstrap=n_bootstrap,
            block_length=self.block_length,
            bootstrap_distribution=bootstrap_stats
        )

    def bootstrap_sharpe_ratio(
        self,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
        annualization_factor: int = 252
    ) -> BlockBootstrapResult:
        """
        Bootstrap Sharpe ratio specifically.

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for CI
            annualization_factor: Factor to annualize (252 for daily)

        Returns:
            BlockBootstrapResult for Sharpe ratio
        """
        def sharpe_ratio(data):
            if self.is_multivariate:
                data = data[:, 0]  # Use first column
            mean_ret = np.mean(data)
            std_ret = np.std(data)
            if std_ret == 0:
                return 0
            return mean_ret / std_ret * np.sqrt(annualization_factor)

        return self.bootstrap_statistic(
            sharpe_ratio,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            statistic_name='sharpe_ratio'
        )

    def bootstrap_max_drawdown(
        self,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95
    ) -> BlockBootstrapResult:
        """
        Bootstrap maximum drawdown.

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for CI

        Returns:
            BlockBootstrapResult for max drawdown
        """
        def max_drawdown(data):
            if self.is_multivariate:
                data = data[:, 0]
            # Calculate cumulative returns
            cum_returns = np.cumprod(1 + data)
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (running_max - cum_returns) / running_max
            return np.max(drawdowns)

        return self.bootstrap_statistic(
            max_drawdown,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            statistic_name='max_drawdown'
        )

    def hypothesis_test(
        self,
        statistic_func: Callable,
        null_value: float = 0,
        alternative: str = 'two-sided',
        n_bootstrap: int = 10000
    ) -> BlockBootstrapResult:
        """
        Perform bootstrap hypothesis test.

        Tests H0: statistic = null_value vs H1 based on alternative.

        Args:
            statistic_func: Function to compute statistic
            null_value: Value under null hypothesis
            alternative: 'two-sided', 'greater', or 'less'
            n_bootstrap: Number of bootstrap samples

        Returns:
            BlockBootstrapResult with p-value

        Example:
            >>> # Test if Sharpe ratio is significantly > 0
            >>> result = bb.hypothesis_test(
            ...     sharpe_func, null_value=0, alternative='greater'
            ... )
        """
        # Get bootstrap distribution
        result = self.bootstrap_statistic(
            statistic_func,
            n_bootstrap=n_bootstrap,
            confidence_level=0.95,
            statistic_name='test_statistic'
        )

        # Center bootstrap distribution under null
        centered_dist = result.bootstrap_distribution - result.bootstrap_mean + null_value

        # Calculate p-value
        original = result.original_statistic

        if alternative == 'two-sided':
            p_value = 2 * min(
                np.mean(centered_dist >= original),
                np.mean(centered_dist <= original)
            )
        elif alternative == 'greater':
            p_value = np.mean(centered_dist >= original)
        elif alternative == 'less':
            p_value = np.mean(centered_dist <= original)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")

        return BlockBootstrapResult(
            statistic_name=result.statistic_name,
            original_statistic=result.original_statistic,
            bootstrap_mean=result.bootstrap_mean,
            bootstrap_std=result.bootstrap_std,
            confidence_interval=result.confidence_interval,
            confidence_level=result.confidence_level,
            p_value=p_value,
            n_bootstrap=n_bootstrap,
            block_length=self.block_length,
            bootstrap_distribution=result.bootstrap_distribution
        )

    def bootstrap_var(
        self,
        confidence_level: float = 0.95,
        var_level: float = 0.05,
        n_bootstrap: int = 10000
    ) -> BlockBootstrapResult:
        """
        Bootstrap Value at Risk (VaR) confidence interval.

        Args:
            confidence_level: Confidence level for CI around VaR
            var_level: VaR probability level (e.g., 0.05 for 5% VaR)
            n_bootstrap: Number of bootstrap samples

        Returns:
            BlockBootstrapResult for VaR
        """
        def var_statistic(data):
            if self.is_multivariate:
                data = data[:, 0]
            return -np.percentile(data, var_level * 100)

        return self.bootstrap_statistic(
            var_statistic,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            statistic_name=f'VaR_{int(var_level*100)}%'
        )

    def bootstrap_covariance_matrix(
        self,
        n_bootstrap: int = 5000
    ) -> Dict[str, Any]:
        """
        Bootstrap covariance matrix for multivariate data.

        Returns bootstrap distribution of covariance matrix elements.

        Args:
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary with mean covariance, std, and confidence intervals
        """
        if not self.is_multivariate or self.data.shape[1] < 2:
            raise ValueError("Need multivariate data for covariance bootstrap")

        n_assets = self.data.shape[1]
        original_cov = np.cov(self.data, rowvar=False)

        # Store bootstrap covariances
        bootstrap_covs = np.zeros((n_bootstrap, n_assets, n_assets))

        for i in range(n_bootstrap):
            bootstrap_sample = self.generate_bootstrap_sample()
            bootstrap_covs[i] = np.cov(bootstrap_sample, rowvar=False)

        mean_cov = np.mean(bootstrap_covs, axis=0)
        std_cov = np.std(bootstrap_covs, axis=0)
        ci_lower = np.percentile(bootstrap_covs, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_covs, 97.5, axis=0)

        return {
            'original_covariance': original_cov,
            'mean_covariance': mean_cov,
            'std_covariance': std_cov,
            'ci_lower_2.5%': ci_lower,
            'ci_upper_97.5%': ci_upper,
            'n_bootstrap': n_bootstrap,
            'block_length': self.block_length
        }

    def bootstrap_correlation(
        self,
        n_bootstrap: int = 5000,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Bootstrap correlation matrix for multivariate data.

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for CI

        Returns:
            Dictionary with correlation statistics
        """
        if not self.is_multivariate or self.data.shape[1] < 2:
            raise ValueError("Need multivariate data for correlation bootstrap")

        n_assets = self.data.shape[1]
        original_corr = np.corrcoef(self.data, rowvar=False)

        bootstrap_corrs = np.zeros((n_bootstrap, n_assets, n_assets))

        for i in range(n_bootstrap):
            bootstrap_sample = self.generate_bootstrap_sample()
            bootstrap_corrs[i] = np.corrcoef(bootstrap_sample, rowvar=False)

        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_corrs, alpha / 2 * 100, axis=0)
        ci_upper = np.percentile(bootstrap_corrs, (1 - alpha / 2) * 100, axis=0)

        return {
            'original_correlation': original_corr,
            'mean_correlation': np.mean(bootstrap_corrs, axis=0),
            'std_correlation': np.std(bootstrap_corrs, axis=0),
            f'ci_lower_{alpha/2*100:.1f}%': ci_lower,
            f'ci_upper_{(1-alpha/2)*100:.1f}%': ci_upper,
            'n_bootstrap': n_bootstrap,
            'block_length': self.block_length
        }

    def generate_bootstrap_paths(
        self,
        n_paths: int = 1000
    ) -> np.ndarray:
        """
        Generate multiple bootstrap return paths.

        Useful for Monte Carlo simulation with bootstrapped returns.

        Args:
            n_paths: Number of paths to generate

        Returns:
            Array of shape (n_paths, n_observations, n_features)
        """
        paths = np.zeros((n_paths, self.n) + self.data.shape[1:])

        for i in range(n_paths):
            paths[i] = self.generate_bootstrap_sample()

        return paths

    def summary(self) -> Dict[str, Any]:
        """Return summary of bootstrap configuration."""
        return {
            'n_observations': self.n,
            'method': self.method,
            'block_length': self.block_length,
            'is_multivariate': self.is_multivariate,
            'n_features': self.data.shape[1] if self.is_multivariate else 1,
            'seed': self.seed
        }
