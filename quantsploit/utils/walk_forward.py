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
