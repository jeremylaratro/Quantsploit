"""
Monte Carlo Simulation Framework for Quantsploit

This module provides comprehensive Monte Carlo simulation capabilities for
backtesting analysis, including:
- Bootstrap resampling of returns
- Trade sequence randomization
- Confidence interval calculation
- Probability of ruin analysis
- Drawdown distribution analysis
- Visualization helpers for equity curves and metric distributions

The framework integrates with the existing BacktestResults class to provide
robust statistical analysis of strategy performance.

Example Usage:
    from quantsploit.utils.monte_carlo import MonteCarloSimulator
    from quantsploit.utils.backtesting import BacktestResults, Trade

    # Create simulator from backtest results
    simulator = MonteCarloSimulator(backtest_results)

    # Run bootstrap simulation
    bootstrap_results = simulator.bootstrap_backtest(n_simulations=5000)

    # Get confidence intervals
    ci = simulator.calculate_confidence_intervals(
        bootstrap_results,
        metrics=['total_return', 'sharpe_ratio', 'max_drawdown']
    )

    # Generate distribution report
    report = simulator.generate_distribution_report(bootstrap_results)

    # Plot results
    simulator.plot_equity_distribution(bootstrap_results)
    simulator.plot_metric_histogram(bootstrap_results, 'total_return')
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings

# Import from existing backtesting module
from quantsploit.utils.backtesting import (
    BacktestResults,
    BacktestConfig,
    Trade,
    PositionSide
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomizationMethod(Enum):
    """Methods for randomizing trade sequences in Monte Carlo simulations."""
    BOOTSTRAP = "bootstrap"  # Resample with replacement
    SHUFFLE = "shuffle"      # Random permutation without replacement
    PARAMETRIC = "parametric"  # Generate from fitted distribution


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulations.

    Attributes:
        n_simulations: Number of Monte Carlo simulations to run (1,000-10,000 recommended)
        random_seed: Random seed for reproducibility (None for random)
        confidence_levels: Percentile levels for confidence intervals
        initial_capital: Starting capital for equity curve simulation
        risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        ruin_threshold: Percentage drawdown considered as ruin (e.g., 0.5 = 50% loss)
    """
    n_simulations: int = 5000
    random_seed: Optional[int] = None
    confidence_levels: Tuple[float, ...] = (5, 25, 50, 75, 95)
    initial_capital: float = 100000.0
    risk_free_rate: float = 0.02
    ruin_threshold: float = 0.5  # 50% drawdown = ruin


@dataclass
class SimulationResult:
    """Results from a single Monte Carlo simulation run.

    Attributes:
        simulation_id: Unique identifier for this simulation
        total_return: Total return in currency
        total_return_pct: Total return as percentage
        sharpe_ratio: Annualized Sharpe ratio
        sortino_ratio: Annualized Sortino ratio
        max_drawdown: Maximum drawdown percentage
        max_drawdown_duration: Duration of maximum drawdown in days
        volatility: Annualized volatility percentage
        win_rate: Percentage of winning trades
        profit_factor: Ratio of gross profits to gross losses
        calmar_ratio: Return / max drawdown ratio
        total_trades: Number of trades
        equity_curve: Array of equity values over trade sequence
        is_ruined: Whether this simulation hit the ruin threshold
    """
    simulation_id: int = 0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    total_trades: int = 0
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    is_ruined: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding large arrays."""
        result = asdict(self)
        # Convert numpy array to list for serialization, or exclude if too large
        if len(result['equity_curve']) > 1000:
            result['equity_curve'] = f"[{len(result['equity_curve'])} values]"
        else:
            result['equity_curve'] = list(result['equity_curve'])
        return result


@dataclass
class MonteCarloResults:
    """Aggregated results from Monte Carlo simulation ensemble.

    Attributes:
        config: Configuration used for simulation
        simulations: List of individual simulation results
        method: Randomization method used
        original_results: Original backtest results for comparison
        execution_time_seconds: Time taken to run all simulations
    """
    config: SimulationConfig
    simulations: List[SimulationResult] = field(default_factory=list)
    method: RandomizationMethod = RandomizationMethod.BOOTSTRAP
    original_results: Optional[BacktestResults] = None
    execution_time_seconds: float = 0.0

    def get_metric_array(self, metric: str) -> np.ndarray:
        """Extract an array of values for a specific metric across all simulations.

        Args:
            metric: Name of the metric to extract (e.g., 'total_return_pct', 'sharpe_ratio')

        Returns:
            NumPy array of metric values

        Raises:
            AttributeError: If metric doesn't exist on SimulationResult
        """
        return np.array([getattr(sim, metric) for sim in self.simulations])

    def get_percentile(self, metric: str, percentile: float) -> float:
        """Get a specific percentile value for a metric.

        Args:
            metric: Name of the metric
            percentile: Percentile value (0-100)

        Returns:
            The value at the specified percentile
        """
        values = self.get_metric_array(metric)
        return float(np.percentile(values, percentile))

    @property
    def probability_of_ruin(self) -> float:
        """Calculate the probability of ruin across all simulations."""
        if not self.simulations:
            return 0.0
        ruined = sum(1 for sim in self.simulations if sim.is_ruined)
        return ruined / len(self.simulations)

    @property
    def n_simulations(self) -> int:
        """Number of simulations in this result set."""
        return len(self.simulations)


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric.

    Attributes:
        metric_name: Name of the metric
        lower: Lower bound of the interval
        median: Median value (50th percentile)
        upper: Upper bound of the interval
        mean: Mean value across simulations
        std: Standard deviation across simulations
        confidence_level: Confidence level percentage (e.g., 90 for 90% CI)
    """
    metric_name: str
    lower: float
    median: float
    upper: float
    mean: float
    std: float
    confidence_level: float = 90.0

    def __str__(self) -> str:
        return (f"{self.metric_name}: {self.median:.4f} "
                f"[{self.lower:.4f}, {self.upper:.4f}] "
                f"({self.confidence_level:.0f}% CI)")


@dataclass
class DistributionReport:
    """Comprehensive distribution report for Monte Carlo results.

    Attributes:
        n_simulations: Number of simulations run
        method: Randomization method used
        metrics: Dictionary of metric statistics
        confidence_intervals: Dictionary of confidence intervals by metric
        probability_of_ruin: Probability of hitting ruin threshold
        expected_drawdown_stats: Statistics on drawdown distribution
        original_vs_simulated: Comparison of original results to simulation median
    """
    n_simulations: int
    method: str
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confidence_intervals: Dict[str, ConfidenceInterval] = field(default_factory=dict)
    probability_of_ruin: float = 0.0
    expected_drawdown_stats: Dict[str, float] = field(default_factory=dict)
    original_vs_simulated: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'n_simulations': self.n_simulations,
            'method': self.method,
            'probability_of_ruin': self.probability_of_ruin,
            'expected_drawdown_stats': self.expected_drawdown_stats,
            'metrics': self.metrics,
            'confidence_intervals': {
                name: {
                    'lower': ci.lower,
                    'median': ci.median,
                    'upper': ci.upper,
                    'mean': ci.mean,
                    'std': ci.std,
                    'confidence_level': ci.confidence_level
                }
                for name, ci in self.confidence_intervals.items()
            },
            'original_vs_simulated': self.original_vs_simulated
        }
        return result


class MonteCarloSimulator:
    """
    Monte Carlo Simulation framework for backtesting analysis.

    This class provides comprehensive Monte Carlo simulation capabilities for
    analyzing the statistical properties of trading strategy performance.

    Key Features:
        - Bootstrap resampling of trade returns
        - Trade sequence shuffling
        - Parametric simulation from fitted distributions
        - Confidence interval calculation
        - Probability of ruin analysis
        - Drawdown distribution analysis
        - Visualization helpers

    Example:
        >>> from quantsploit.utils.monte_carlo import MonteCarloSimulator
        >>>
        >>> # From existing backtest results
        >>> simulator = MonteCarloSimulator(backtest_results)
        >>>
        >>> # Run 5000 bootstrap simulations
        >>> results = simulator.bootstrap_backtest(n_simulations=5000)
        >>>
        >>> # Analyze results
        >>> print(f"Probability of ruin: {results.probability_of_ruin:.2%}")
        >>> print(f"Median Sharpe: {results.get_percentile('sharpe_ratio', 50):.2f}")
        >>>
        >>> # Get confidence intervals
        >>> ci = simulator.calculate_confidence_intervals(results)
        >>> for metric, interval in ci.items():
        >>>     print(interval)

    Attributes:
        config: SimulationConfig with simulation parameters
        trades: List of Trade objects from original backtest
        returns: Array of trade returns (percentage)
        pnl: Array of trade P&L (currency)
        original_results: Original BacktestResults for comparison
    """

    def __init__(
        self,
        backtest_results: Optional[BacktestResults] = None,
        trades: Optional[List[Trade]] = None,
        config: Optional[SimulationConfig] = None
    ):
        """
        Initialize the Monte Carlo simulator.

        Args:
            backtest_results: BacktestResults object containing trades and metrics.
                             Either this or trades must be provided.
            trades: List of Trade objects. Used if backtest_results not provided.
            config: SimulationConfig with simulation parameters.
                   Defaults to reasonable values if not provided.

        Raises:
            ValueError: If neither backtest_results nor trades are provided,
                       or if trades list is empty.
        """
        self.config = config or SimulationConfig()

        # Set random seed if specified
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        # Extract trades from backtest results or use provided trades
        if backtest_results is not None:
            self.original_results = backtest_results
            self.trades = [t for t in backtest_results.trades if t.exit_date is not None]
        elif trades is not None:
            self.original_results = None
            self.trades = [t for t in trades if t.exit_date is not None]
        else:
            raise ValueError("Either backtest_results or trades must be provided")

        if not self.trades:
            raise ValueError("No completed trades available for simulation")

        # Extract returns and P&L arrays
        self.returns = np.array([t.pnl_pct for t in self.trades])
        self.pnl = np.array([t.pnl for t in self.trades])

        # Calculate trade durations for parametric simulation
        self.durations = np.array([
            (t.exit_date - t.entry_date).days
            for t in self.trades
        ])

        logger.info(f"MonteCarloSimulator initialized with {len(self.trades)} trades")

    def _resample_returns(
        self,
        method: RandomizationMethod,
        size: Optional[int] = None
    ) -> np.ndarray:
        """
        Resample trade returns using the specified method.

        Args:
            method: Randomization method to use
            size: Number of samples to generate (defaults to original length)

        Returns:
            Array of resampled returns
        """
        n = size or len(self.returns)

        if method == RandomizationMethod.BOOTSTRAP:
            # Resample with replacement
            indices = np.random.choice(len(self.returns), size=n, replace=True)
            return self.returns[indices]

        elif method == RandomizationMethod.SHUFFLE:
            # Random permutation without replacement
            shuffled = self.returns.copy()
            np.random.shuffle(shuffled)
            return shuffled

        elif method == RandomizationMethod.PARAMETRIC:
            # Fit a distribution and sample from it
            mean = np.mean(self.returns)
            std = np.std(self.returns)
            skew = self._calculate_skewness(self.returns)

            # Use a skew-normal approximation if significant skewness
            if abs(skew) > 0.5:
                # Simple skewed sampling using a mixture
                samples = np.random.normal(mean, std, n)
                # Adjust for skewness
                if skew > 0:
                    samples = samples + np.random.exponential(std * 0.3, n) * 0.5
                else:
                    samples = samples - np.random.exponential(std * 0.3, n) * 0.5
            else:
                samples = np.random.normal(mean, std, n)

            return samples

        else:
            raise ValueError(f"Unknown randomization method: {method}")

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of a data array."""
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        return (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)

    def _simulate_equity_curve(
        self,
        resampled_returns: np.ndarray,
        initial_capital: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate an equity curve from resampled returns.

        Args:
            resampled_returns: Array of percentage returns
            initial_capital: Starting capital (uses config default if not provided)

        Returns:
            Array of equity values
        """
        capital = initial_capital or self.config.initial_capital

        # Convert percentage returns to multipliers
        multipliers = 1 + (resampled_returns / 100)

        # Calculate cumulative equity
        equity = capital * np.cumprod(multipliers)

        # Prepend initial capital
        equity = np.insert(equity, 0, capital)

        return equity

    def _calculate_drawdown_series(self, equity_curve: np.ndarray) -> np.ndarray:
        """
        Calculate drawdown series from equity curve.

        Args:
            equity_curve: Array of equity values

        Returns:
            Array of drawdown percentages (negative values)
        """
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak * 100
        return drawdown

    def _calculate_max_drawdown_duration(
        self,
        equity_curve: np.ndarray
    ) -> int:
        """
        Calculate the maximum drawdown duration in number of trades.

        Args:
            equity_curve: Array of equity values

        Returns:
            Maximum number of consecutive periods in drawdown
        """
        peak = np.maximum.accumulate(equity_curve)
        in_drawdown = equity_curve < peak

        if not np.any(in_drawdown):
            return 0

        # Find consecutive drawdown periods
        max_duration = 0
        current_duration = 0

        for in_dd in in_drawdown:
            if in_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def _calculate_simulation_metrics(
        self,
        resampled_returns: np.ndarray,
        equity_curve: np.ndarray,
        simulation_id: int
    ) -> SimulationResult:
        """
        Calculate all metrics for a single simulation.

        Args:
            resampled_returns: Array of resampled percentage returns
            equity_curve: Array of equity values
            simulation_id: Unique ID for this simulation

        Returns:
            SimulationResult with all calculated metrics
        """
        initial_capital = equity_curve[0]
        final_equity = equity_curve[-1]

        # Returns
        total_return = final_equity - initial_capital
        total_return_pct = (total_return / initial_capital) * 100

        # Win rate
        wins = np.sum(resampled_returns > 0)
        total_trades = len(resampled_returns)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        # Profit factor
        gross_profits = np.sum(resampled_returns[resampled_returns > 0])
        gross_losses = abs(np.sum(resampled_returns[resampled_returns <= 0]))
        profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else float('inf')

        # Volatility (annualized, assuming ~252 trading days per year)
        # Estimate trades per year based on original data
        if len(self.trades) > 1:
            total_days = (self.trades[-1].exit_date - self.trades[0].entry_date).days
            trades_per_year = len(self.trades) / (total_days / 365.25) if total_days > 0 else 252
        else:
            trades_per_year = 252

        volatility = np.std(resampled_returns) * np.sqrt(trades_per_year)

        # Sharpe ratio
        excess_returns = resampled_returns - (self.config.risk_free_rate / trades_per_year * 100)
        sharpe_ratio = 0.0
        if np.std(resampled_returns) > 0:
            sharpe_ratio = np.sqrt(trades_per_year) * np.mean(excess_returns) / np.std(resampled_returns)

        # Sortino ratio (downside deviation)
        downside_returns = resampled_returns[resampled_returns < 0]
        sortino_ratio = 0.0
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino_ratio = np.sqrt(trades_per_year) * np.mean(excess_returns) / np.std(downside_returns)

        # Maximum drawdown
        drawdown_series = self._calculate_drawdown_series(equity_curve)
        max_drawdown = abs(np.min(drawdown_series))
        max_dd_duration = self._calculate_max_drawdown_duration(equity_curve)

        # Calmar ratio
        calmar_ratio = 0.0
        if max_drawdown > 0:
            # Annualize return for Calmar
            days = total_trades * np.mean(self.durations) if np.mean(self.durations) > 0 else total_trades
            years = days / 365.25
            annual_return = ((final_equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

        # Check for ruin
        is_ruined = max_drawdown >= (self.config.ruin_threshold * 100)

        return SimulationResult(
            simulation_id=simulation_id,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            volatility=volatility,
            win_rate=win_rate,
            profit_factor=profit_factor if profit_factor != float('inf') else 999.99,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            equity_curve=equity_curve,
            is_ruined=is_ruined
        )

    def bootstrap_backtest(
        self,
        n_simulations: Optional[int] = None,
        preserve_sequence_length: bool = True
    ) -> MonteCarloResults:
        """
        Run bootstrap Monte Carlo simulation by resampling returns with replacement.

        Bootstrap resampling creates new "alternative histories" by randomly
        sampling from the original trade returns with replacement. This preserves
        the distribution of individual trade outcomes while varying their sequence.

        Args:
            n_simulations: Number of simulations to run (1,000-10,000 recommended).
                          Uses config default if not specified.
            preserve_sequence_length: If True, each simulation has the same number
                                     of trades as the original backtest.

        Returns:
            MonteCarloResults containing all simulation results and statistics.

        Example:
            >>> simulator = MonteCarloSimulator(backtest_results)
            >>> results = simulator.bootstrap_backtest(n_simulations=5000)
            >>> print(f"Median return: {results.get_percentile('total_return_pct', 50):.2f}%")
            >>> print(f"5th percentile: {results.get_percentile('total_return_pct', 5):.2f}%")
        """
        n_sims = n_simulations or self.config.n_simulations

        if n_sims < 100:
            logger.warning("n_simulations < 100 may give unreliable results")
        elif n_sims > 10000:
            logger.info(f"Running {n_sims} simulations - this may take a while")

        start_time = datetime.now()
        simulations = []

        size = len(self.returns) if preserve_sequence_length else None

        for i in range(n_sims):
            # Resample returns
            resampled = self._resample_returns(RandomizationMethod.BOOTSTRAP, size)

            # Generate equity curve
            equity_curve = self._simulate_equity_curve(resampled)

            # Calculate metrics
            result = self._calculate_simulation_metrics(resampled, equity_curve, i)
            simulations.append(result)

            # Log progress for large simulations
            if n_sims >= 1000 and (i + 1) % 1000 == 0:
                logger.info(f"Completed {i + 1}/{n_sims} simulations")

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Bootstrap simulation completed in {execution_time:.2f} seconds")

        return MonteCarloResults(
            config=self.config,
            simulations=simulations,
            method=RandomizationMethod.BOOTSTRAP,
            original_results=self.original_results,
            execution_time_seconds=execution_time
        )

    def randomize_trade_sequence(
        self,
        n_simulations: Optional[int] = None,
        method: RandomizationMethod = RandomizationMethod.SHUFFLE
    ) -> MonteCarloResults:
        """
        Run Monte Carlo simulation by randomizing the order of trades.

        This method shuffles the order of trades without replacement (SHUFFLE)
        or samples with replacement (BOOTSTRAP). Unlike bootstrap_backtest,
        SHUFFLE mode uses each original trade exactly once per simulation,
        only changing the order.

        Args:
            n_simulations: Number of simulations to run.
            method: Randomization method - SHUFFLE for permutation,
                   BOOTSTRAP for sampling with replacement,
                   PARAMETRIC for distribution-based sampling.

        Returns:
            MonteCarloResults containing all simulation results.

        Example:
            >>> simulator = MonteCarloSimulator(backtest_results)
            >>> # Shuffle trades without replacement
            >>> results = simulator.randomize_trade_sequence(
            ...     n_simulations=5000,
            ...     method=RandomizationMethod.SHUFFLE
            ... )
            >>> print(f"Probability of ruin: {results.probability_of_ruin:.2%}")
        """
        n_sims = n_simulations or self.config.n_simulations

        start_time = datetime.now()
        simulations = []

        for i in range(n_sims):
            # Resample returns using specified method
            resampled = self._resample_returns(method)

            # Generate equity curve
            equity_curve = self._simulate_equity_curve(resampled)

            # Calculate metrics
            result = self._calculate_simulation_metrics(resampled, equity_curve, i)
            simulations.append(result)

            # Log progress
            if n_sims >= 1000 and (i + 1) % 1000 == 0:
                logger.info(f"Completed {i + 1}/{n_sims} simulations")

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"{method.value} simulation completed in {execution_time:.2f} seconds")

        return MonteCarloResults(
            config=self.config,
            simulations=simulations,
            method=method,
            original_results=self.original_results,
            execution_time_seconds=execution_time
        )

    def calculate_confidence_intervals(
        self,
        mc_results: MonteCarloResults,
        metrics: Optional[List[str]] = None,
        confidence_level: float = 90.0
    ) -> Dict[str, ConfidenceInterval]:
        """
        Calculate percentile-based confidence intervals for specified metrics.

        Computes confidence intervals using the percentile method, which is
        non-parametric and makes no assumptions about the underlying distribution.

        Args:
            mc_results: MonteCarloResults from a simulation run.
            metrics: List of metric names to calculate CIs for.
                    Defaults to common metrics if not specified.
            confidence_level: Confidence level as percentage (e.g., 90 for 90% CI).
                            Default is 90%.

        Returns:
            Dictionary mapping metric names to ConfidenceInterval objects.

        Example:
            >>> results = simulator.bootstrap_backtest(n_simulations=5000)
            >>> ci = simulator.calculate_confidence_intervals(
            ...     results,
            ...     metrics=['total_return_pct', 'sharpe_ratio', 'max_drawdown'],
            ...     confidence_level=95.0
            ... )
            >>> for metric, interval in ci.items():
            ...     print(f"{metric}: {interval.median:.2f} [{interval.lower:.2f}, {interval.upper:.2f}]")
        """
        if metrics is None:
            metrics = [
                'total_return_pct',
                'sharpe_ratio',
                'sortino_ratio',
                'max_drawdown',
                'win_rate',
                'profit_factor',
                'calmar_ratio',
                'volatility'
            ]

        # Calculate percentile bounds
        lower_percentile = (100 - confidence_level) / 2
        upper_percentile = 100 - lower_percentile

        intervals = {}

        for metric in metrics:
            try:
                values = mc_results.get_metric_array(metric)

                # Filter out infinities and NaNs
                valid_values = values[np.isfinite(values)]

                if len(valid_values) == 0:
                    logger.warning(f"No valid values for metric '{metric}'")
                    continue

                intervals[metric] = ConfidenceInterval(
                    metric_name=metric,
                    lower=float(np.percentile(valid_values, lower_percentile)),
                    median=float(np.percentile(valid_values, 50)),
                    upper=float(np.percentile(valid_values, upper_percentile)),
                    mean=float(np.mean(valid_values)),
                    std=float(np.std(valid_values)),
                    confidence_level=confidence_level
                )
            except AttributeError:
                logger.warning(f"Metric '{metric}' not found in simulation results")

        return intervals

    def calculate_probability_of_ruin(
        self,
        mc_results: MonteCarloResults,
        ruin_threshold: Optional[float] = None
    ) -> float:
        """
        Calculate the probability of ruin (hitting a maximum drawdown threshold).

        Args:
            mc_results: MonteCarloResults from a simulation run.
            ruin_threshold: Maximum drawdown percentage considered as ruin.
                           Uses config default if not specified.

        Returns:
            Probability of ruin as a float between 0 and 1.

        Example:
            >>> results = simulator.bootstrap_backtest(n_simulations=10000)
            >>> prob_ruin = simulator.calculate_probability_of_ruin(results, ruin_threshold=0.25)
            >>> print(f"Probability of 25% drawdown: {prob_ruin:.2%}")
        """
        threshold = (ruin_threshold or self.config.ruin_threshold) * 100

        drawdowns = mc_results.get_metric_array('max_drawdown')
        ruined = np.sum(drawdowns >= threshold)

        return ruined / len(drawdowns)

    def calculate_expected_drawdown_distribution(
        self,
        mc_results: MonteCarloResults
    ) -> Dict[str, float]:
        """
        Calculate statistics on the expected drawdown distribution.

        Provides comprehensive statistics about the distribution of maximum
        drawdowns across all simulations.

        Args:
            mc_results: MonteCarloResults from a simulation run.

        Returns:
            Dictionary with drawdown statistics including:
            - mean, median, std: Basic statistics
            - percentiles: 5th, 25th, 75th, 95th percentiles
            - min, max: Extreme values
            - prob_above_X: Probability of drawdown exceeding X%

        Example:
            >>> results = simulator.bootstrap_backtest(n_simulations=5000)
            >>> dd_stats = simulator.calculate_expected_drawdown_distribution(results)
            >>> print(f"Expected max drawdown: {dd_stats['mean']:.2f}%")
            >>> print(f"95th percentile: {dd_stats['percentile_95']:.2f}%")
        """
        drawdowns = mc_results.get_metric_array('max_drawdown')

        stats = {
            'mean': float(np.mean(drawdowns)),
            'median': float(np.median(drawdowns)),
            'std': float(np.std(drawdowns)),
            'min': float(np.min(drawdowns)),
            'max': float(np.max(drawdowns)),
            'percentile_5': float(np.percentile(drawdowns, 5)),
            'percentile_25': float(np.percentile(drawdowns, 25)),
            'percentile_75': float(np.percentile(drawdowns, 75)),
            'percentile_95': float(np.percentile(drawdowns, 95)),
            'prob_above_10': float(np.mean(drawdowns >= 10)),
            'prob_above_20': float(np.mean(drawdowns >= 20)),
            'prob_above_30': float(np.mean(drawdowns >= 30)),
            'prob_above_50': float(np.mean(drawdowns >= 50)),
        }

        return stats

    def generate_distribution_report(
        self,
        mc_results: MonteCarloResults,
        confidence_level: float = 90.0
    ) -> DistributionReport:
        """
        Generate a comprehensive distribution report from Monte Carlo results.

        Provides a complete statistical analysis of the simulation results,
        including confidence intervals, probability of ruin, drawdown analysis,
        and comparison with original backtest results.

        Args:
            mc_results: MonteCarloResults from a simulation run.
            confidence_level: Confidence level for intervals (default 90%).

        Returns:
            DistributionReport with comprehensive statistics.

        Example:
            >>> results = simulator.bootstrap_backtest(n_simulations=5000)
            >>> report = simulator.generate_distribution_report(results)
            >>> print(f"Probability of ruin: {report.probability_of_ruin:.2%}")
            >>> for metric, stats in report.metrics.items():
            ...     print(f"{metric}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        """
        # Key metrics to analyze
        metrics = [
            'total_return_pct',
            'sharpe_ratio',
            'sortino_ratio',
            'max_drawdown',
            'win_rate',
            'profit_factor',
            'calmar_ratio',
            'volatility'
        ]

        # Calculate statistics for each metric
        metric_stats = {}
        for metric in metrics:
            try:
                values = mc_results.get_metric_array(metric)
                valid_values = values[np.isfinite(values)]

                if len(valid_values) > 0:
                    metric_stats[metric] = {
                        'mean': float(np.mean(valid_values)),
                        'median': float(np.median(valid_values)),
                        'std': float(np.std(valid_values)),
                        'min': float(np.min(valid_values)),
                        'max': float(np.max(valid_values)),
                        'skewness': float(self._calculate_skewness(valid_values)),
                        'percentile_5': float(np.percentile(valid_values, 5)),
                        'percentile_95': float(np.percentile(valid_values, 95)),
                    }
            except (AttributeError, ValueError):
                continue

        # Calculate confidence intervals
        confidence_intervals = self.calculate_confidence_intervals(
            mc_results, metrics, confidence_level
        )

        # Drawdown distribution
        dd_stats = self.calculate_expected_drawdown_distribution(mc_results)

        # Probability of ruin
        prob_ruin = self.calculate_probability_of_ruin(mc_results)

        # Compare original to simulated (if original results available)
        original_comparison = {}
        if mc_results.original_results is not None:
            orig = mc_results.original_results
            for metric in metrics:
                try:
                    orig_value = getattr(orig, metric.replace('_pct', ''), None)
                    if orig_value is None:
                        orig_value = getattr(orig, metric, None)

                    if orig_value is not None and metric in metric_stats:
                        sim_median = metric_stats[metric]['median']
                        original_comparison[metric] = {
                            'original': float(orig_value),
                            'simulated_median': sim_median,
                            'difference': float(orig_value) - sim_median,
                            'percentile_rank': float(
                                np.mean(mc_results.get_metric_array(metric) <= orig_value) * 100
                            )
                        }
                except (AttributeError, TypeError):
                    continue

        return DistributionReport(
            n_simulations=mc_results.n_simulations,
            method=mc_results.method.value,
            metrics=metric_stats,
            confidence_intervals=confidence_intervals,
            probability_of_ruin=prob_ruin,
            expected_drawdown_stats=dd_stats,
            original_vs_simulated=original_comparison
        )

    def plot_equity_distribution(
        self,
        mc_results: MonteCarloResults,
        n_paths: int = 100,
        show_percentiles: bool = True,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Create a fan chart visualization of equity curve distribution.

        Plots multiple simulated equity paths along with percentile bands
        to visualize the range of possible outcomes.

        Args:
            mc_results: MonteCarloResults from a simulation run.
            n_paths: Number of individual paths to plot (default 100).
            show_percentiles: Whether to show percentile bands (5th, 25th, 75th, 95th).
            figsize: Figure size as (width, height) tuple.
            save_path: If provided, save the figure to this path.

        Returns:
            Matplotlib figure object if matplotlib is available, None otherwise.

        Example:
            >>> results = simulator.bootstrap_backtest(n_simulations=5000)
            >>> fig = simulator.plot_equity_distribution(results, n_paths=200)
            >>> fig.savefig('equity_distribution.png')
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Get equity curves - sample if we have more than n_paths
        n_sims = len(mc_results.simulations)
        if n_sims > n_paths:
            indices = np.random.choice(n_sims, size=n_paths, replace=False)
            selected = [mc_results.simulations[i] for i in indices]
        else:
            selected = mc_results.simulations

        # Find the maximum length equity curve
        max_len = max(len(sim.equity_curve) for sim in selected)

        # Plot individual paths with low alpha
        for sim in selected:
            curve = sim.equity_curve
            x = np.arange(len(curve))
            color = 'green' if curve[-1] >= curve[0] else 'red'
            ax.plot(x, curve, color=color, alpha=0.05, linewidth=0.5)

        # Calculate and plot percentile bands
        if show_percentiles:
            # Align all curves to same length by padding with final value
            aligned_curves = np.zeros((len(mc_results.simulations), max_len))
            for i, sim in enumerate(mc_results.simulations):
                curve = sim.equity_curve
                aligned_curves[i, :len(curve)] = curve
                if len(curve) < max_len:
                    aligned_curves[i, len(curve):] = curve[-1]

            x = np.arange(max_len)

            # Calculate percentiles
            p5 = np.percentile(aligned_curves, 5, axis=0)
            p25 = np.percentile(aligned_curves, 25, axis=0)
            p50 = np.percentile(aligned_curves, 50, axis=0)
            p75 = np.percentile(aligned_curves, 75, axis=0)
            p95 = np.percentile(aligned_curves, 95, axis=0)

            # Plot percentile bands
            ax.fill_between(x, p5, p95, alpha=0.2, color='blue', label='5-95th percentile')
            ax.fill_between(x, p25, p75, alpha=0.3, color='blue', label='25-75th percentile')
            ax.plot(x, p50, color='blue', linewidth=2, label='Median')

        # Plot original equity curve if available
        if mc_results.original_results is not None:
            orig_curve = mc_results.original_results.equity_curve
            if 'equity' in orig_curve.columns:
                ax.plot(
                    np.arange(len(orig_curve)),
                    orig_curve['equity'].values,
                    color='black',
                    linewidth=2,
                    linestyle='--',
                    label='Original'
                )

        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Equity ($)')
        ax.set_title(f'Monte Carlo Equity Distribution ({mc_results.n_simulations} simulations)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        return fig

    def plot_metric_histogram(
        self,
        mc_results: MonteCarloResults,
        metric: str,
        bins: int = 50,
        show_original: bool = True,
        show_stats: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Plot histogram of a metric's distribution across simulations.

        Creates a histogram showing the distribution of a specific metric
        across all Monte Carlo simulations, with optional annotations for
        key statistics and the original backtest value.

        Args:
            mc_results: MonteCarloResults from a simulation run.
            metric: Name of the metric to plot (e.g., 'total_return_pct', 'sharpe_ratio').
            bins: Number of histogram bins.
            show_original: Whether to show the original backtest value as a vertical line.
            show_stats: Whether to annotate mean, median, and percentiles.
            figsize: Figure size as (width, height) tuple.
            save_path: If provided, save the figure to this path.

        Returns:
            Matplotlib figure object if matplotlib is available, None otherwise.

        Example:
            >>> results = simulator.bootstrap_backtest(n_simulations=5000)
            >>> fig = simulator.plot_metric_histogram(results, 'sharpe_ratio')
            >>> fig.savefig('sharpe_distribution.png')
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None

        values = mc_results.get_metric_array(metric)
        valid_values = values[np.isfinite(values)]

        if len(valid_values) == 0:
            logger.warning(f"No valid values for metric '{metric}'")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Plot histogram
        n, bins_edges, patches = ax.hist(
            valid_values, bins=bins, density=True,
            alpha=0.7, color='steelblue', edgecolor='white'
        )

        # Add KDE if scipy available
        try:
            from scipy import stats as scipy_stats
            kde = scipy_stats.gaussian_kde(valid_values)
            x_kde = np.linspace(valid_values.min(), valid_values.max(), 200)
            ax.plot(x_kde, kde(x_kde), color='darkblue', linewidth=2, label='KDE')
        except ImportError:
            pass

        # Calculate statistics
        mean_val = np.mean(valid_values)
        median_val = np.median(valid_values)
        p5 = np.percentile(valid_values, 5)
        p95 = np.percentile(valid_values, 95)

        if show_stats:
            # Add vertical lines for key statistics
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            ax.axvline(p5, color='orange', linestyle=':', linewidth=1.5, label=f'5th %ile: {p5:.2f}')
            ax.axvline(p95, color='orange', linestyle=':', linewidth=1.5, label=f'95th %ile: {p95:.2f}')

        # Add original value if available
        if show_original and mc_results.original_results is not None:
            try:
                orig_value = getattr(mc_results.original_results, metric.replace('_pct', ''), None)
                if orig_value is None:
                    orig_value = getattr(mc_results.original_results, metric, None)

                if orig_value is not None:
                    ax.axvline(
                        orig_value, color='black', linestyle='-.',
                        linewidth=2, label=f'Original: {orig_value:.2f}'
                    )

                    # Calculate percentile rank
                    percentile_rank = np.mean(valid_values <= orig_value) * 100
                    ax.text(
                        0.02, 0.98,
                        f'Original at {percentile_rank:.1f}th percentile',
                        transform=ax.transAxes,
                        verticalalignment='top',
                        fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    )
            except (AttributeError, TypeError):
                pass

        # Formatting
        metric_display = metric.replace('_', ' ').title()
        ax.set_xlabel(metric_display)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {metric_display} ({mc_results.n_simulations} simulations)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        return fig

    def plot_drawdown_distribution(
        self,
        mc_results: MonteCarloResults,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Plot the distribution of maximum drawdowns with risk annotations.

        Creates a specialized visualization for drawdown risk analysis,
        showing the probability of various drawdown levels.

        Args:
            mc_results: MonteCarloResults from a simulation run.
            figsize: Figure size as (width, height) tuple.
            save_path: If provided, save the figure to this path.

        Returns:
            Matplotlib figure object if matplotlib is available, None otherwise.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None

        drawdowns = mc_results.get_metric_array('max_drawdown')

        fig, ax = plt.subplots(figsize=figsize)

        # Plot histogram
        n, bins_edges, patches = ax.hist(
            drawdowns, bins=50, density=True,
            alpha=0.7, color='indianred', edgecolor='white'
        )

        # Color bins by risk level
        for i, patch in enumerate(patches):
            bin_center = (bins_edges[i] + bins_edges[i + 1]) / 2
            if bin_center < 10:
                patch.set_facecolor('lightgreen')
            elif bin_center < 20:
                patch.set_facecolor('yellow')
            elif bin_center < 30:
                patch.set_facecolor('orange')
            else:
                patch.set_facecolor('indianred')

        # Add risk zone annotations
        ax.axvline(10, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(20, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(30, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(50, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7)

        # Add statistics
        dd_stats = self.calculate_expected_drawdown_distribution(mc_results)

        stats_text = (
            f"Mean: {dd_stats['mean']:.1f}%\n"
            f"Median: {dd_stats['median']:.1f}%\n"
            f"95th %ile: {dd_stats['percentile_95']:.1f}%\n"
            f"P(DD > 20%): {dd_stats['prob_above_20']:.1%}\n"
            f"P(DD > 30%): {dd_stats['prob_above_30']:.1%}"
        )

        ax.text(
            0.98, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ax.set_xlabel('Maximum Drawdown (%)')
        ax.set_ylabel('Density')
        ax.set_title(f'Drawdown Risk Distribution ({mc_results.n_simulations} simulations)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        return fig

    def generate_simulation_from_backtest_results(
        self,
        original_results: BacktestResults,
        n_simulations: Optional[int] = None,
        method: RandomizationMethod = RandomizationMethod.BOOTSTRAP
    ) -> Tuple[MonteCarloResults, List[BacktestResults]]:
        """
        Generate new BacktestResults objects for each Monte Carlo simulation.

        This method creates full BacktestResults objects that can be used
        for further analysis or comparison, preserving the structure of
        the original backtest output.

        Args:
            original_results: The original BacktestResults to base simulations on.
            n_simulations: Number of simulations to run.
            method: Randomization method to use.

        Returns:
            Tuple of (MonteCarloResults, List[BacktestResults]) where each
            BacktestResults represents one simulation.

        Example:
            >>> mc_results, sim_bt_results = simulator.generate_simulation_from_backtest_results(
            ...     original_results,
            ...     n_simulations=1000
            ... )
            >>> # Each sim_bt_results[i] is a full BacktestResults object
            >>> for bt in sim_bt_results[:5]:
            ...     print(f"Return: {bt.total_return_pct:.2f}%, Sharpe: {bt.sharpe_ratio:.2f}")
        """
        n_sims = n_simulations or self.config.n_simulations

        # Run the Monte Carlo simulation
        if method == RandomizationMethod.BOOTSTRAP:
            mc_results = self.bootstrap_backtest(n_sims)
        else:
            mc_results = self.randomize_trade_sequence(n_sims, method)

        # Generate BacktestResults for each simulation
        simulated_results = []

        for sim in mc_results.simulations:
            # Create a new BacktestResults object
            bt_result = BacktestResults(
                total_return=sim.total_return,
                total_return_pct=sim.total_return_pct,
                annualized_return=0.0,  # Would need duration info
                benchmark_return=original_results.benchmark_return,
                sharpe_ratio=sim.sharpe_ratio,
                sortino_ratio=sim.sortino_ratio,
                calmar_ratio=sim.calmar_ratio,
                max_drawdown=sim.max_drawdown,
                max_drawdown_duration=sim.max_drawdown_duration,
                volatility=sim.volatility,
                total_trades=sim.total_trades,
                win_rate=sim.win_rate,
                profit_factor=sim.profit_factor,
                # Note: We don't replicate individual trades as they're resampled
            )

            # Create equity curve DataFrame
            equity_df = pd.DataFrame({
                'equity': sim.equity_curve
            })
            bt_result.equity_curve = equity_df

            simulated_results.append(bt_result)

        return mc_results, simulated_results


def run_monte_carlo_analysis(
    backtest_results: BacktestResults,
    n_simulations: int = 5000,
    confidence_level: float = 90.0,
    random_seed: Optional[int] = None,
    methods: Optional[List[RandomizationMethod]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run a complete Monte Carlo analysis.

    Runs Monte Carlo simulations using multiple methods and generates
    a comprehensive analysis report.

    Args:
        backtest_results: BacktestResults from a strategy backtest.
        n_simulations: Number of simulations per method (default 5000).
        confidence_level: Confidence level for intervals (default 90%).
        random_seed: Random seed for reproducibility.
        methods: List of RandomizationMethod to use. Defaults to all three.

    Returns:
        Dictionary containing:
        - 'bootstrap': MonteCarloResults from bootstrap method
        - 'shuffle': MonteCarloResults from shuffle method
        - 'parametric': MonteCarloResults from parametric method
        - 'reports': Dictionary of DistributionReport for each method
        - 'comparison': Cross-method comparison statistics

    Example:
        >>> from quantsploit.utils.monte_carlo import run_monte_carlo_analysis
        >>>
        >>> analysis = run_monte_carlo_analysis(backtest_results, n_simulations=5000)
        >>>
        >>> # Access bootstrap results
        >>> bootstrap_report = analysis['reports']['bootstrap']
        >>> print(f"Bootstrap P(ruin): {bootstrap_report.probability_of_ruin:.2%}")
        >>>
        >>> # Compare methods
        >>> for method, report in analysis['reports'].items():
        >>>     print(f"{method}: median return = {report.metrics['total_return_pct']['median']:.2f}%")
    """
    if methods is None:
        methods = [
            RandomizationMethod.BOOTSTRAP,
            RandomizationMethod.SHUFFLE,
            RandomizationMethod.PARAMETRIC
        ]

    config = SimulationConfig(
        n_simulations=n_simulations,
        random_seed=random_seed,
        confidence_levels=(5, 25, 50, 75, 95)
    )

    simulator = MonteCarloSimulator(backtest_results, config=config)

    results = {}
    reports = {}

    for method in methods:
        method_name = method.value
        logger.info(f"Running {method_name} Monte Carlo simulation...")

        if method == RandomizationMethod.BOOTSTRAP:
            mc_results = simulator.bootstrap_backtest(n_simulations)
        else:
            mc_results = simulator.randomize_trade_sequence(n_simulations, method)

        results[method_name] = mc_results
        reports[method_name] = simulator.generate_distribution_report(
            mc_results, confidence_level
        )

    # Generate cross-method comparison
    comparison = {}
    metrics = ['total_return_pct', 'sharpe_ratio', 'max_drawdown', 'win_rate']

    for metric in metrics:
        comparison[metric] = {}
        for method_name, report in reports.items():
            if metric in report.metrics:
                comparison[metric][method_name] = {
                    'mean': report.metrics[metric]['mean'],
                    'median': report.metrics[metric]['median'],
                    'std': report.metrics[metric]['std']
                }

    return {
        'bootstrap': results.get('bootstrap'),
        'shuffle': results.get('shuffle'),
        'parametric': results.get('parametric'),
        'reports': reports,
        'comparison': comparison,
        'simulator': simulator
    }
