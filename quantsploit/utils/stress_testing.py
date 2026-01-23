"""
Stress Testing Framework for Quantsploit

This module provides comprehensive stress testing capabilities for trading strategies:
- Historical stress tests (crisis periods)
- Hypothetical stress tests (synthetic shocks)
- Reverse stress tests (find breaking scenarios)
- Comprehensive stress test reporting

Integrates with the existing backtesting engine for strategy evaluation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import json
import copy

from quantsploit.utils.backtesting import (
    Backtester, BacktestConfig, BacktestResults, PositionSide
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Stress Testing
# =============================================================================

@dataclass
class HistoricalScenario:
    """Defines a historical crisis scenario for stress testing"""
    name: str
    start_date: str
    end_date: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'extreme'
    market_context: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class HypotheticalScenario:
    """Defines a hypothetical stress scenario"""
    name: str
    description: str
    shock_type: str  # 'market_shock', 'volatility_spike', 'correlation_breakdown', 'liquidity_crisis'
    parameters: Dict[str, Any]
    severity: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StressTestResult:
    """Results from a single stress test"""
    scenario_name: str
    scenario_type: str  # 'historical', 'hypothetical', 'reverse'
    strategy_name: str
    symbol: str

    # Performance metrics under stress
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    volatility: float

    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float

    # Stress-specific metrics
    worst_day_return: float
    best_day_return: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    recovery_time_days: Optional[int]  # Days to recover from max drawdown

    # Baseline comparison
    baseline_return: Optional[float] = None
    return_degradation: Optional[float] = None  # % worse than baseline

    # Additional context
    start_date: str = ""
    end_date: str = ""
    duration_days: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ReverseStressResult:
    """Results from reverse stress testing"""
    strategy_name: str
    symbol: str
    breaking_scenario: str
    breaking_parameters: Dict[str, Any]
    threshold_metric: str  # The metric that was used to define "breaking"
    threshold_value: float
    actual_value: float
    description: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StressTestReport:
    """Comprehensive stress test report"""
    generated_at: str
    strategy_name: str
    symbols: List[str]

    # Historical stress results
    historical_results: List[StressTestResult]

    # Hypothetical stress results
    hypothetical_results: List[StressTestResult]

    # Reverse stress results
    reverse_stress_results: List[ReverseStressResult]

    # Summary statistics
    summary: Dict[str, Any]

    # Risk assessment
    risk_rating: str  # 'low', 'medium', 'high', 'extreme'
    risk_factors: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict:
        result = asdict(self)
        result['historical_results'] = [r.to_dict() if hasattr(r, 'to_dict') else r
                                         for r in self.historical_results]
        result['hypothetical_results'] = [r.to_dict() if hasattr(r, 'to_dict') else r
                                           for r in self.hypothetical_results]
        result['reverse_stress_results'] = [r.to_dict() if hasattr(r, 'to_dict') else r
                                             for r in self.reverse_stress_results]
        return result


# =============================================================================
# Scenario Generator Class
# =============================================================================

class ScenarioGenerator:
    """
    Generates synthetic stress scenarios by applying various shocks to market data.

    Supports:
    - Market shocks (instantaneous price drops)
    - Volatility spikes (increased price variability)
    - Correlation breakdowns (all assets move together)
    - Liquidity crises (wider spreads, reduced volume)
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the scenario generator.

        Args:
            random_seed: Optional seed for reproducible scenarios
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        self.random_seed = random_seed

    def apply_market_shock(
        self,
        data: pd.DataFrame,
        shock_percentage: float,
        shock_day: Optional[int] = None,
        recovery_days: Optional[int] = None,
        recovery_type: str = 'linear'
    ) -> pd.DataFrame:
        """
        Apply an instantaneous market shock to price data.

        Args:
            data: DataFrame with OHLCV data
            shock_percentage: Percentage drop (negative) or gain (positive)
                             e.g., -0.20 for a 20% drop
            shock_day: Day index to apply shock (None = middle of period)
            recovery_days: Days to recover to pre-shock trend (None = no recovery)
            recovery_type: 'linear', 'exponential', or 'none'

        Returns:
            Modified DataFrame with shock applied
        """
        df = data.copy()
        n_days = len(df)

        if shock_day is None:
            shock_day = n_days // 2

        shock_day = max(0, min(shock_day, n_days - 1))

        # Calculate shock multiplier
        shock_mult = 1 + shock_percentage

        # Apply shock to all prices from shock_day onwards
        price_cols = ['Open', 'High', 'Low', 'Close']
        available_cols = [col for col in price_cols if col in df.columns]

        for col in available_cols:
            # Get pre-shock price
            pre_shock_price = df[col].iloc[shock_day - 1] if shock_day > 0 else df[col].iloc[0]

            # Apply immediate shock
            df.loc[df.index[shock_day:], col] = df.loc[df.index[shock_day:], col] * shock_mult

            # Apply recovery if specified
            if recovery_days and recovery_days > 0 and recovery_type != 'none':
                recovery_end = min(shock_day + recovery_days, n_days)
                days_to_recover = recovery_end - shock_day

                if days_to_recover > 0:
                    # Calculate recovery path
                    recovery_mult = 1 / shock_mult  # To get back to original level

                    for i, idx in enumerate(range(shock_day, recovery_end)):
                        if recovery_type == 'linear':
                            partial_recovery = (i / days_to_recover) * (recovery_mult - 1) + 1
                        elif recovery_type == 'exponential':
                            partial_recovery = np.exp(np.log(recovery_mult) * (i / days_to_recover))
                        else:
                            partial_recovery = 1

                        df.loc[df.index[idx], col] = df.loc[df.index[idx], col] * partial_recovery

        return df

    def apply_volatility_shock(
        self,
        data: pd.DataFrame,
        volatility_multiplier: float,
        start_day: Optional[int] = None,
        duration_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Apply a volatility spike to price data.

        Args:
            data: DataFrame with OHLCV data
            volatility_multiplier: Multiply volatility by this factor (e.g., 2.0 = 2x vol)
            start_day: Day index to start volatility spike (None = start)
            duration_days: Days the spike lasts (None = entire period)

        Returns:
            Modified DataFrame with increased volatility
        """
        df = data.copy()
        n_days = len(df)

        if start_day is None:
            start_day = 0
        if duration_days is None:
            duration_days = n_days - start_day

        end_day = min(start_day + duration_days, n_days)

        # Calculate daily returns and amplify them
        if 'Close' not in df.columns:
            return df

        close_prices = df['Close'].copy()
        returns = close_prices.pct_change()
        mean_return = returns.mean()

        # Amplify returns during spike period
        for i in range(start_day + 1, end_day):
            if i < len(returns) and not pd.isna(returns.iloc[i]):
                # Amplify deviation from mean
                deviation = returns.iloc[i] - mean_return
                amplified_deviation = deviation * volatility_multiplier
                new_return = mean_return + amplified_deviation

                # Apply new return to price
                prev_price = df['Close'].iloc[i - 1]
                new_price = prev_price * (1 + new_return)

                # Update all price columns proportionally
                price_ratio = new_price / df['Close'].iloc[i]
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in df.columns:
                        df.loc[df.index[i], col] = df.loc[df.index[i], col] * price_ratio

        return df

    def apply_correlation_shock(
        self,
        data_dict: Dict[str, pd.DataFrame],
        target_correlation: float = 0.95
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply a correlation breakdown where all assets move together.

        Args:
            data_dict: Dictionary of symbol -> DataFrame with OHLCV data
            target_correlation: Target correlation between all assets (0 to 1)

        Returns:
            Dictionary of modified DataFrames with correlated movements
        """
        if not data_dict:
            return data_dict

        result = {}

        # Use the first symbol's returns as the "market" factor
        symbols = list(data_dict.keys())
        reference_symbol = symbols[0]
        reference_data = data_dict[reference_symbol]

        if 'Close' not in reference_data.columns:
            return data_dict

        reference_returns = reference_data['Close'].pct_change().dropna()

        for symbol, df in data_dict.items():
            modified_df = df.copy()

            if 'Close' not in modified_df.columns:
                result[symbol] = modified_df
                continue

            original_returns = modified_df['Close'].pct_change()

            # Blend original returns with reference returns
            # Higher target_correlation = more weight on reference
            for i in range(1, len(modified_df)):
                if i - 1 < len(reference_returns) and not pd.isna(original_returns.iloc[i]):
                    blended_return = (
                        target_correlation * reference_returns.iloc[i - 1] +
                        (1 - target_correlation) * original_returns.iloc[i]
                    )

                    # Apply blended return
                    prev_price = modified_df['Close'].iloc[i - 1]
                    new_price = prev_price * (1 + blended_return)

                    # Update all price columns proportionally
                    if modified_df['Close'].iloc[i] != 0:
                        price_ratio = new_price / modified_df['Close'].iloc[i]
                        for col in ['Open', 'High', 'Low', 'Close']:
                            if col in modified_df.columns:
                                modified_df.loc[modified_df.index[i], col] = (
                                    modified_df.loc[modified_df.index[i], col] * price_ratio
                                )

            result[symbol] = modified_df

        return result

    def apply_liquidity_crisis(
        self,
        data: pd.DataFrame,
        spread_multiplier: float = 10.0,
        volume_reduction: float = 0.5,
        start_day: Optional[int] = None,
        duration_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Simulate a liquidity crisis with wider spreads and reduced volume.

        Args:
            data: DataFrame with OHLCV data
            spread_multiplier: Multiply the High-Low spread by this factor
            volume_reduction: Reduce volume by this fraction (0.5 = 50% reduction)
            start_day: Day index to start crisis (None = start)
            duration_days: Days the crisis lasts (None = entire period)

        Returns:
            Modified DataFrame with liquidity crisis effects
        """
        df = data.copy()
        n_days = len(df)

        if start_day is None:
            start_day = 0
        if duration_days is None:
            duration_days = n_days - start_day

        end_day = min(start_day + duration_days, n_days)

        for i in range(start_day, end_day):
            idx = df.index[i]

            # Widen the High-Low spread
            if 'High' in df.columns and 'Low' in df.columns:
                mid_price = (df.loc[idx, 'High'] + df.loc[idx, 'Low']) / 2
                original_spread = df.loc[idx, 'High'] - df.loc[idx, 'Low']
                new_spread = original_spread * spread_multiplier

                df.loc[idx, 'High'] = mid_price + new_spread / 2
                df.loc[idx, 'Low'] = mid_price - new_spread / 2

            # Reduce volume
            if 'Volume' in df.columns:
                df.loc[idx, 'Volume'] = df.loc[idx, 'Volume'] * (1 - volume_reduction)

        return df

    def generate_custom_scenario(
        self,
        data: pd.DataFrame,
        shocks: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Generate a custom scenario by applying multiple shocks in sequence.

        Args:
            data: DataFrame with OHLCV data
            shocks: List of shock definitions, each with 'type' and 'params' keys
                   Types: 'market_shock', 'volatility_shock', 'liquidity_crisis'

        Returns:
            Modified DataFrame with all shocks applied

        Example:
            shocks = [
                {'type': 'market_shock', 'params': {'shock_percentage': -0.15}},
                {'type': 'volatility_shock', 'params': {'volatility_multiplier': 2.5}},
                {'type': 'liquidity_crisis', 'params': {'spread_multiplier': 5.0}}
            ]
        """
        df = data.copy()

        for shock in shocks:
            shock_type = shock.get('type', '')
            params = shock.get('params', {})

            if shock_type == 'market_shock':
                df = self.apply_market_shock(df, **params)
            elif shock_type == 'volatility_shock':
                df = self.apply_volatility_shock(df, **params)
            elif shock_type == 'liquidity_crisis':
                df = self.apply_liquidity_crisis(df, **params)

        return df


# =============================================================================
# Stress Test Framework Class
# =============================================================================

class StressTestFramework:
    """
    Comprehensive stress testing framework for trading strategies.

    Provides:
    - Historical stress testing on actual crisis periods
    - Hypothetical stress testing with synthetic scenarios
    - Reverse stress testing to find breaking points
    - Comprehensive stress test reporting
    """

    # Predefined historical crisis scenarios
    HISTORICAL_SCENARIOS = {
        '2008_financial_crisis': HistoricalScenario(
            name='2008 Financial Crisis',
            start_date='2008-09-01',
            end_date='2009-03-31',
            description='Global financial crisis triggered by subprime mortgage collapse',
            severity='extreme',
            market_context='S&P 500 dropped ~50% from peak, massive volatility spike'
        ),
        '2020_covid_crash': HistoricalScenario(
            name='2020 COVID Crash',
            start_date='2020-02-01',
            end_date='2020-04-30',
            description='Rapid market crash due to COVID-19 pandemic',
            severity='extreme',
            market_context='Fastest 30% drop in history, followed by rapid recovery'
        ),
        '2022_rate_hikes': HistoricalScenario(
            name='2022 Rate Hikes',
            start_date='2022-01-01',
            end_date='2022-10-31',
            description='Federal Reserve aggressive rate hiking cycle',
            severity='high',
            market_context='Sustained bear market with tech sector hit hardest'
        ),
        'dotcom_bubble': HistoricalScenario(
            name='Dotcom Bubble Burst',
            start_date='2000-03-01',
            end_date='2002-10-31',
            description='Technology bubble collapse',
            severity='extreme',
            market_context='NASDAQ dropped ~78% from peak, prolonged bear market'
        ),
        'flash_crash_2010': HistoricalScenario(
            name='Flash Crash 2010',
            start_date='2010-05-06',
            end_date='2010-05-06',
            description='Flash crash with Dow dropping ~1000 points in minutes',
            severity='high',
            market_context='Single day extreme volatility event'
        ),
        '2011_debt_ceiling': HistoricalScenario(
            name='2011 Debt Ceiling Crisis',
            start_date='2011-07-01',
            end_date='2011-10-31',
            description='US debt ceiling crisis and S&P downgrade',
            severity='medium',
            market_context='~20% correction driven by political uncertainty'
        ),
        '2015_china_fears': HistoricalScenario(
            name='2015 China Slowdown Fears',
            start_date='2015-08-01',
            end_date='2016-02-28',
            description='Global market selloff on China growth concerns',
            severity='medium',
            market_context='~15% correction with elevated volatility'
        ),
        '2018_q4_selloff': HistoricalScenario(
            name='2018 Q4 Selloff',
            start_date='2018-10-01',
            end_date='2018-12-31',
            description='Sharp Q4 selloff on trade war and Fed concerns',
            severity='medium',
            market_context='~20% correction in 3 months'
        )
    }

    # Predefined hypothetical scenarios
    HYPOTHETICAL_SCENARIOS = {
        'market_shock_10pct': HypotheticalScenario(
            name='10% Market Shock',
            description='Instantaneous 10% market drop',
            shock_type='market_shock',
            parameters={'shock_percentage': -0.10},
            severity='medium'
        ),
        'market_shock_20pct': HypotheticalScenario(
            name='20% Market Shock',
            description='Instantaneous 20% market drop (crash)',
            shock_type='market_shock',
            parameters={'shock_percentage': -0.20},
            severity='high'
        ),
        'market_shock_30pct': HypotheticalScenario(
            name='30% Market Shock',
            description='Instantaneous 30% market drop (extreme crash)',
            shock_type='market_shock',
            parameters={'shock_percentage': -0.30},
            severity='extreme'
        ),
        'volatility_2x': HypotheticalScenario(
            name='2x Volatility Spike',
            description='Volatility doubles from normal levels',
            shock_type='volatility_spike',
            parameters={'volatility_multiplier': 2.0},
            severity='medium'
        ),
        'volatility_3x': HypotheticalScenario(
            name='3x Volatility Spike',
            description='Volatility triples from normal levels',
            shock_type='volatility_spike',
            parameters={'volatility_multiplier': 3.0},
            severity='high'
        ),
        'correlation_breakdown': HypotheticalScenario(
            name='Correlation Breakdown',
            description='All correlations spike to 1 (everything moves together)',
            shock_type='correlation_breakdown',
            parameters={'target_correlation': 0.95},
            severity='high'
        ),
        'liquidity_crisis': HypotheticalScenario(
            name='Liquidity Crisis',
            description='10x normal spread with 50% volume reduction',
            shock_type='liquidity_crisis',
            parameters={'spread_multiplier': 10.0, 'volume_reduction': 0.5},
            severity='high'
        ),
        'combined_shock': HypotheticalScenario(
            name='Combined Shock',
            description='15% drop + 2x volatility + liquidity crisis',
            shock_type='combined',
            parameters={
                'shocks': [
                    {'type': 'market_shock', 'params': {'shock_percentage': -0.15}},
                    {'type': 'volatility_shock', 'params': {'volatility_multiplier': 2.0}},
                    {'type': 'liquidity_crisis', 'params': {'spread_multiplier': 5.0, 'volume_reduction': 0.3}}
                ]
            },
            severity='extreme'
        )
    }

    def __init__(
        self,
        data_fetcher: Any,
        initial_capital: float = 100000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.001,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize the stress testing framework.

        Args:
            data_fetcher: DataFetcher instance for retrieving market data
            initial_capital: Starting capital for backtests
            commission_pct: Commission percentage per trade
            slippage_pct: Slippage percentage per trade
            risk_free_rate: Annual risk-free rate for Sharpe calculations
        """
        self.data_fetcher = data_fetcher
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.risk_free_rate = risk_free_rate

        self.scenario_generator = ScenarioGenerator()
        self.results: List[StressTestResult] = []
        self.reverse_results: List[ReverseStressResult] = []

    def _create_backtest_config(self) -> BacktestConfig:
        """Create a BacktestConfig with current settings"""
        return BacktestConfig(
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
            risk_free_rate=self.risk_free_rate,
            position_size=1.0,
            max_positions=1
        )

    def _calculate_risk_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate additional risk metrics from equity curve.

        Args:
            equity_curve: DataFrame with equity values

        Returns:
            Dictionary with VaR, CVaR, and other risk metrics
        """
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return {
                'var_95': 0.0,
                'cvar_95': 0.0,
                'worst_day_return': 0.0,
                'best_day_return': 0.0,
                'recovery_time_days': None
            }

        returns = equity_curve['equity'].pct_change().dropna()

        if len(returns) == 0:
            return {
                'var_95': 0.0,
                'cvar_95': 0.0,
                'worst_day_return': 0.0,
                'best_day_return': 0.0,
                'recovery_time_days': None
            }

        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5) * 100  # 5th percentile of returns

        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        if pd.isna(cvar_95):
            cvar_95 = var_95

        # Best and worst day
        worst_day = returns.min() * 100
        best_day = returns.max() * 100

        # Recovery time from max drawdown
        cummax = equity_curve['equity'].cummax()
        drawdown = (equity_curve['equity'] - cummax) / cummax

        recovery_time = None
        in_drawdown = False
        drawdown_start = None
        max_recovery_time = 0

        for i, (idx, dd) in enumerate(drawdown.items()):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                if drawdown_start is not None:
                    recovery = i - drawdown_start
                    max_recovery_time = max(max_recovery_time, recovery)

        if max_recovery_time > 0:
            recovery_time = max_recovery_time

        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'worst_day_return': worst_day,
            'best_day_return': best_day,
            'recovery_time_days': recovery_time
        }

    def run_historical_stress_test(
        self,
        strategy_func: Callable,
        strategy_name: str,
        symbols: List[str],
        scenarios: Optional[List[str]] = None,
        baseline_period: str = '1y'
    ) -> List[StressTestResult]:
        """
        Test strategy on historical crisis periods.

        Args:
            strategy_func: Strategy function that takes (backtester, date, row, symbol, data)
            strategy_name: Name of the strategy being tested
            symbols: List of symbols to test
            scenarios: List of scenario keys to test (None = all scenarios)
            baseline_period: Period for calculating baseline performance

        Returns:
            List of StressTestResult objects
        """
        results = []

        # Determine which scenarios to run
        if scenarios is None:
            scenarios = list(self.HISTORICAL_SCENARIOS.keys())

        for scenario_key in scenarios:
            if scenario_key not in self.HISTORICAL_SCENARIOS:
                logger.warning(f"Unknown scenario: {scenario_key}")
                continue

            scenario = self.HISTORICAL_SCENARIOS[scenario_key]
            logger.info(f"Running historical stress test: {scenario.name}")

            for symbol in symbols:
                try:
                    result = self._run_single_historical_test(
                        strategy_func=strategy_func,
                        strategy_name=strategy_name,
                        symbol=symbol,
                        scenario=scenario,
                        baseline_period=baseline_period
                    )
                    if result:
                        results.append(result)
                        self.results.append(result)
                except Exception as e:
                    logger.error(f"Error in historical stress test {scenario.name}/{symbol}: {e}")

        return results

    def _run_single_historical_test(
        self,
        strategy_func: Callable,
        strategy_name: str,
        symbol: str,
        scenario: HistoricalScenario,
        baseline_period: str
    ) -> Optional[StressTestResult]:
        """Run a single historical stress test"""
        # Fetch data for the crisis period
        # We need extra data before the period for indicator warmup
        start_date = pd.to_datetime(scenario.start_date)
        end_date = pd.to_datetime(scenario.end_date)

        # Calculate how much data to fetch (period + warmup buffer)
        days_needed = (end_date - start_date).days + 100  # 100 day warmup

        # Fetch extended data
        full_data = self.data_fetcher.get_stock_data(
            symbol=symbol,
            period='max',  # Get all available data
            interval='1d'
        )

        if full_data is None or len(full_data) < 30:
            logger.warning(f"Insufficient data for {symbol}")
            return None

        # Filter to crisis period for backtesting
        mask = (full_data.index >= scenario.start_date) & (full_data.index <= scenario.end_date)
        crisis_data = full_data[mask]

        if len(crisis_data) < 5:
            logger.warning(f"Insufficient data in crisis period for {symbol}")
            return None

        # Run backtest on crisis period
        config = self._create_backtest_config()
        backtester = Backtester(config)

        # Create strategy wrapper
        def strategy_wrapper(bt, date, row):
            strategy_func(bt, date, row, symbol, full_data)

        bt_results = backtester.run_backtest(crisis_data, strategy_wrapper, symbol=symbol)

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(bt_results.equity_curve)

        # Calculate baseline (normal period) performance for comparison
        baseline_return = self._calculate_baseline_performance(
            strategy_func, strategy_name, symbol, baseline_period, full_data
        )

        # Calculate return degradation
        return_degradation = None
        if baseline_return is not None and baseline_return != 0:
            return_degradation = ((bt_results.total_return_pct - baseline_return) /
                                  abs(baseline_return)) * 100

        return StressTestResult(
            scenario_name=scenario.name,
            scenario_type='historical',
            strategy_name=strategy_name,
            symbol=symbol,
            total_return=bt_results.total_return_pct,
            max_drawdown=bt_results.max_drawdown,
            sharpe_ratio=bt_results.sharpe_ratio,
            sortino_ratio=bt_results.sortino_ratio,
            volatility=bt_results.volatility,
            total_trades=bt_results.total_trades,
            win_rate=bt_results.win_rate,
            profit_factor=bt_results.profit_factor,
            worst_day_return=risk_metrics['worst_day_return'],
            best_day_return=risk_metrics['best_day_return'],
            var_95=risk_metrics['var_95'],
            cvar_95=risk_metrics['cvar_95'],
            recovery_time_days=risk_metrics['recovery_time_days'],
            baseline_return=baseline_return,
            return_degradation=return_degradation,
            start_date=scenario.start_date,
            end_date=scenario.end_date,
            duration_days=(end_date - start_date).days
        )

    def _calculate_baseline_performance(
        self,
        strategy_func: Callable,
        strategy_name: str,
        symbol: str,
        period: str,
        full_data: pd.DataFrame
    ) -> Optional[float]:
        """Calculate baseline performance for a normal period"""
        try:
            # Use the most recent data for baseline
            baseline_data = full_data.tail(252)  # ~1 year of trading days

            if len(baseline_data) < 50:
                return None

            config = self._create_backtest_config()
            backtester = Backtester(config)

            def strategy_wrapper(bt, date, row):
                strategy_func(bt, date, row, symbol, full_data)

            bt_results = backtester.run_backtest(baseline_data, strategy_wrapper, symbol=symbol)
            return bt_results.total_return_pct

        except Exception as e:
            logger.error(f"Error calculating baseline: {e}")
            return None

    def run_hypothetical_stress_test(
        self,
        strategy_func: Callable,
        strategy_name: str,
        symbols: List[str],
        scenarios: Optional[List[str]] = None,
        base_data_period: str = '1y'
    ) -> List[StressTestResult]:
        """
        Test strategy on hypothetical synthetic stress scenarios.

        Args:
            strategy_func: Strategy function
            strategy_name: Name of the strategy
            symbols: List of symbols to test
            scenarios: List of scenario keys to test (None = all scenarios)
            base_data_period: Period of real data to use as base

        Returns:
            List of StressTestResult objects
        """
        results = []

        if scenarios is None:
            scenarios = list(self.HYPOTHETICAL_SCENARIOS.keys())

        for scenario_key in scenarios:
            if scenario_key not in self.HYPOTHETICAL_SCENARIOS:
                logger.warning(f"Unknown hypothetical scenario: {scenario_key}")
                continue

            scenario = self.HYPOTHETICAL_SCENARIOS[scenario_key]
            logger.info(f"Running hypothetical stress test: {scenario.name}")

            for symbol in symbols:
                try:
                    result = self._run_single_hypothetical_test(
                        strategy_func=strategy_func,
                        strategy_name=strategy_name,
                        symbol=symbol,
                        scenario=scenario,
                        base_data_period=base_data_period
                    )
                    if result:
                        results.append(result)
                        self.results.append(result)
                except Exception as e:
                    logger.error(f"Error in hypothetical stress test {scenario.name}/{symbol}: {e}")

        return results

    def _run_single_hypothetical_test(
        self,
        strategy_func: Callable,
        strategy_name: str,
        symbol: str,
        scenario: HypotheticalScenario,
        base_data_period: str
    ) -> Optional[StressTestResult]:
        """Run a single hypothetical stress test"""
        # Fetch base data
        base_data = self.data_fetcher.get_stock_data(
            symbol=symbol,
            period=base_data_period,
            interval='1d'
        )

        if base_data is None or len(base_data) < 50:
            logger.warning(f"Insufficient data for {symbol}")
            return None

        # Apply the shock scenario
        if scenario.shock_type == 'market_shock':
            stressed_data = self.scenario_generator.apply_market_shock(
                base_data, **scenario.parameters
            )
        elif scenario.shock_type == 'volatility_spike':
            stressed_data = self.scenario_generator.apply_volatility_shock(
                base_data, **scenario.parameters
            )
        elif scenario.shock_type == 'liquidity_crisis':
            stressed_data = self.scenario_generator.apply_liquidity_crisis(
                base_data, **scenario.parameters
            )
        elif scenario.shock_type == 'combined':
            stressed_data = self.scenario_generator.generate_custom_scenario(
                base_data, scenario.parameters.get('shocks', [])
            )
        else:
            logger.warning(f"Unknown shock type: {scenario.shock_type}")
            return None

        # Run backtest on stressed data
        config = self._create_backtest_config()
        backtester = Backtester(config)

        def strategy_wrapper(bt, date, row):
            strategy_func(bt, date, row, symbol, stressed_data)

        bt_results = backtester.run_backtest(stressed_data, strategy_wrapper, symbol=symbol)

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(bt_results.equity_curve)

        # Calculate baseline on unshocked data
        baseline_backtester = Backtester(config)

        def baseline_wrapper(bt, date, row):
            strategy_func(bt, date, row, symbol, base_data)

        baseline_results = baseline_backtester.run_backtest(base_data, baseline_wrapper, symbol=symbol)
        baseline_return = baseline_results.total_return_pct

        # Calculate return degradation
        return_degradation = None
        if baseline_return != 0:
            return_degradation = ((bt_results.total_return_pct - baseline_return) /
                                  abs(baseline_return)) * 100

        start_date = stressed_data.index[0].strftime('%Y-%m-%d') if hasattr(stressed_data.index[0], 'strftime') else str(stressed_data.index[0])
        end_date = stressed_data.index[-1].strftime('%Y-%m-%d') if hasattr(stressed_data.index[-1], 'strftime') else str(stressed_data.index[-1])

        return StressTestResult(
            scenario_name=scenario.name,
            scenario_type='hypothetical',
            strategy_name=strategy_name,
            symbol=symbol,
            total_return=bt_results.total_return_pct,
            max_drawdown=bt_results.max_drawdown,
            sharpe_ratio=bt_results.sharpe_ratio,
            sortino_ratio=bt_results.sortino_ratio,
            volatility=bt_results.volatility,
            total_trades=bt_results.total_trades,
            win_rate=bt_results.win_rate,
            profit_factor=bt_results.profit_factor,
            worst_day_return=risk_metrics['worst_day_return'],
            best_day_return=risk_metrics['best_day_return'],
            var_95=risk_metrics['var_95'],
            cvar_95=risk_metrics['cvar_95'],
            recovery_time_days=risk_metrics['recovery_time_days'],
            baseline_return=baseline_return,
            return_degradation=return_degradation,
            start_date=start_date,
            end_date=end_date,
            duration_days=len(stressed_data)
        )

    def run_reverse_stress_test(
        self,
        strategy_func: Callable,
        strategy_name: str,
        symbols: List[str],
        threshold_metric: str = 'max_drawdown',
        threshold_value: float = 50.0,
        search_params: Optional[Dict[str, List[float]]] = None
    ) -> List[ReverseStressResult]:
        """
        Find scenarios that cause the strategy to fail (exceed threshold).

        Reverse stress testing works backwards: instead of testing known scenarios,
        it searches for conditions that break the strategy.

        Args:
            strategy_func: Strategy function
            strategy_name: Name of the strategy
            symbols: List of symbols to test
            threshold_metric: Metric to use for defining "failure"
                            Options: 'max_drawdown', 'total_return', 'sharpe_ratio'
            threshold_value: Value that defines failure
                            e.g., max_drawdown > 50% or total_return < -30%
            search_params: Parameters to search (None = default grid)

        Returns:
            List of ReverseStressResult objects identifying breaking scenarios
        """
        results = []

        # Default search parameters
        if search_params is None:
            search_params = {
                'market_shock': [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.40, -0.50],
                'volatility_multiplier': [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
                'spread_multiplier': [2.0, 5.0, 10.0, 20.0]
            }

        for symbol in symbols:
            logger.info(f"Running reverse stress test for {symbol}")

            # Fetch base data
            base_data = self.data_fetcher.get_stock_data(
                symbol=symbol,
                period='1y',
                interval='1d'
            )

            if base_data is None or len(base_data) < 50:
                continue

            # Test market shocks
            for shock in search_params.get('market_shock', []):
                result = self._test_breaking_scenario(
                    strategy_func, strategy_name, symbol, base_data,
                    'market_shock', {'shock_percentage': shock},
                    threshold_metric, threshold_value
                )
                if result:
                    results.append(result)
                    self.reverse_results.append(result)

            # Test volatility shocks
            for vol_mult in search_params.get('volatility_multiplier', []):
                result = self._test_breaking_scenario(
                    strategy_func, strategy_name, symbol, base_data,
                    'volatility_shock', {'volatility_multiplier': vol_mult},
                    threshold_metric, threshold_value
                )
                if result:
                    results.append(result)
                    self.reverse_results.append(result)

            # Test liquidity crises
            for spread_mult in search_params.get('spread_multiplier', []):
                result = self._test_breaking_scenario(
                    strategy_func, strategy_name, symbol, base_data,
                    'liquidity_crisis', {'spread_multiplier': spread_mult, 'volume_reduction': 0.5},
                    threshold_metric, threshold_value
                )
                if result:
                    results.append(result)
                    self.reverse_results.append(result)

        return results

    def _test_breaking_scenario(
        self,
        strategy_func: Callable,
        strategy_name: str,
        symbol: str,
        base_data: pd.DataFrame,
        shock_type: str,
        shock_params: Dict[str, Any],
        threshold_metric: str,
        threshold_value: float
    ) -> Optional[ReverseStressResult]:
        """Test if a specific scenario breaks the strategy"""
        # Apply shock
        if shock_type == 'market_shock':
            stressed_data = self.scenario_generator.apply_market_shock(base_data, **shock_params)
        elif shock_type == 'volatility_shock':
            stressed_data = self.scenario_generator.apply_volatility_shock(base_data, **shock_params)
        elif shock_type == 'liquidity_crisis':
            stressed_data = self.scenario_generator.apply_liquidity_crisis(base_data, **shock_params)
        else:
            return None

        # Run backtest
        config = self._create_backtest_config()
        backtester = Backtester(config)

        def strategy_wrapper(bt, date, row):
            strategy_func(bt, date, row, symbol, stressed_data)

        bt_results = backtester.run_backtest(stressed_data, strategy_wrapper, symbol=symbol)

        # Check if threshold is breached
        actual_value = getattr(bt_results, threshold_metric, None)
        if actual_value is None:
            return None

        breached = False
        if threshold_metric == 'max_drawdown':
            breached = actual_value > threshold_value
        elif threshold_metric == 'total_return':
            breached = bt_results.total_return_pct < threshold_value
        elif threshold_metric == 'sharpe_ratio':
            breached = bt_results.sharpe_ratio < threshold_value

        if breached:
            # Generate description
            if shock_type == 'market_shock':
                desc = f"{abs(shock_params.get('shock_percentage', 0))*100:.0f}% market crash"
            elif shock_type == 'volatility_shock':
                desc = f"{shock_params.get('volatility_multiplier', 1):.1f}x volatility spike"
            elif shock_type == 'liquidity_crisis':
                desc = f"{shock_params.get('spread_multiplier', 1):.0f}x spread widening"
            else:
                desc = shock_type

            return ReverseStressResult(
                strategy_name=strategy_name,
                symbol=symbol,
                breaking_scenario=shock_type,
                breaking_parameters=shock_params,
                threshold_metric=threshold_metric,
                threshold_value=threshold_value,
                actual_value=actual_value,
                description=f"Strategy breaks ({threshold_metric} = {actual_value:.2f}) under {desc}"
            )

        return None

    def generate_stress_report(
        self,
        strategy_name: str,
        symbols: List[str],
        historical_results: Optional[List[StressTestResult]] = None,
        hypothetical_results: Optional[List[StressTestResult]] = None,
        reverse_results: Optional[List[ReverseStressResult]] = None
    ) -> StressTestReport:
        """
        Generate a comprehensive stress test report.

        Args:
            strategy_name: Name of the strategy tested
            symbols: List of symbols tested
            historical_results: Results from historical stress tests
            hypothetical_results: Results from hypothetical stress tests
            reverse_results: Results from reverse stress tests

        Returns:
            StressTestReport with all results and analysis
        """
        if historical_results is None:
            historical_results = [r for r in self.results if r.scenario_type == 'historical']
        if hypothetical_results is None:
            hypothetical_results = [r for r in self.results if r.scenario_type == 'hypothetical']
        if reverse_results is None:
            reverse_results = self.reverse_results

        # Calculate summary statistics
        all_results = historical_results + hypothetical_results

        summary = self._calculate_summary(all_results)

        # Determine risk rating and factors
        risk_rating, risk_factors = self._assess_risk(all_results, reverse_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            all_results, reverse_results, risk_rating
        )

        return StressTestReport(
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            strategy_name=strategy_name,
            symbols=symbols,
            historical_results=historical_results,
            hypothetical_results=hypothetical_results,
            reverse_stress_results=reverse_results,
            summary=summary,
            risk_rating=risk_rating,
            risk_factors=risk_factors,
            recommendations=recommendations
        )

    def _calculate_summary(self, results: List[StressTestResult]) -> Dict[str, Any]:
        """Calculate summary statistics from stress test results"""
        if not results:
            return {}

        returns = [r.total_return for r in results]
        drawdowns = [r.max_drawdown for r in results]
        sharpes = [r.sharpe_ratio for r in results if not pd.isna(r.sharpe_ratio)]
        var_values = [r.var_95 for r in results if not pd.isna(r.var_95)]

        return {
            'total_scenarios_tested': len(results),
            'average_return': np.mean(returns) if returns else 0,
            'worst_return': np.min(returns) if returns else 0,
            'best_return': np.max(returns) if returns else 0,
            'average_max_drawdown': np.mean(drawdowns) if drawdowns else 0,
            'worst_max_drawdown': np.max(drawdowns) if drawdowns else 0,
            'average_sharpe': np.mean(sharpes) if sharpes else 0,
            'average_var_95': np.mean(var_values) if var_values else 0,
            'scenarios_with_loss': sum(1 for r in returns if r < 0),
            'scenarios_with_severe_drawdown': sum(1 for d in drawdowns if d > 30),
            'return_distribution': {
                'p5': np.percentile(returns, 5) if returns else 0,
                'p25': np.percentile(returns, 25) if returns else 0,
                'p50': np.percentile(returns, 50) if returns else 0,
                'p75': np.percentile(returns, 75) if returns else 0,
                'p95': np.percentile(returns, 95) if returns else 0
            }
        }

    def _assess_risk(
        self,
        results: List[StressTestResult],
        reverse_results: List[ReverseStressResult]
    ) -> Tuple[str, List[str]]:
        """Assess overall risk level and identify risk factors"""
        risk_factors = []
        risk_score = 0

        if not results:
            return 'unknown', ['Insufficient data for risk assessment']

        # Check for severe drawdowns
        severe_dd = [r for r in results if r.max_drawdown > 40]
        if severe_dd:
            risk_factors.append(f"Severe drawdowns (>40%) in {len(severe_dd)} scenarios")
            risk_score += 3

        moderate_dd = [r for r in results if 25 < r.max_drawdown <= 40]
        if moderate_dd:
            risk_factors.append(f"Moderate drawdowns (25-40%) in {len(moderate_dd)} scenarios")
            risk_score += 2

        # Check for large losses
        large_losses = [r for r in results if r.total_return < -30]
        if large_losses:
            risk_factors.append(f"Large losses (>30%) in {len(large_losses)} scenarios")
            risk_score += 3

        # Check VaR breaches
        high_var = [r for r in results if r.var_95 < -5]
        if high_var:
            risk_factors.append(f"High daily VaR (>5%) in {len(high_var)} scenarios")
            risk_score += 2

        # Check reverse stress test results
        if reverse_results:
            easy_breaks = [r for r in reverse_results
                          if 'market_shock' in r.breaking_scenario
                          and abs(r.breaking_parameters.get('shock_percentage', 0)) <= 0.15]
            if easy_breaks:
                risk_factors.append(f"Strategy breaks under moderate stress ({len(easy_breaks)} scenarios)")
                risk_score += 3

        # Check negative Sharpe ratios
        neg_sharpe = [r for r in results if r.sharpe_ratio < 0]
        if neg_sharpe:
            risk_factors.append(f"Negative Sharpe ratio in {len(neg_sharpe)} scenarios")
            risk_score += 1

        # Determine risk rating
        if risk_score >= 8:
            risk_rating = 'extreme'
        elif risk_score >= 5:
            risk_rating = 'high'
        elif risk_score >= 2:
            risk_rating = 'medium'
        else:
            risk_rating = 'low'

        if not risk_factors:
            risk_factors.append('No significant risk factors identified')

        return risk_rating, risk_factors

    def _generate_recommendations(
        self,
        results: List[StressTestResult],
        reverse_results: List[ReverseStressResult],
        risk_rating: str
    ) -> List[str]:
        """Generate actionable recommendations based on stress test results"""
        recommendations = []

        if risk_rating == 'extreme':
            recommendations.append(
                "CRITICAL: Strategy shows extreme vulnerability to market stress. "
                "Consider fundamental redesign or implementation of strict risk limits."
            )
        elif risk_rating == 'high':
            recommendations.append(
                "HIGH RISK: Implement robust risk management including position sizing limits "
                "and stop-loss orders."
            )

        # Specific recommendations based on results
        if results:
            avg_dd = np.mean([r.max_drawdown for r in results])
            if avg_dd > 30:
                recommendations.append(
                    f"Average max drawdown of {avg_dd:.1f}% suggests need for "
                    "drawdown-based position reduction rules."
                )

            volatile_scenarios = [r for r in results if r.volatility > 50]
            if volatile_scenarios:
                recommendations.append(
                    "Strategy shows high volatility in stress scenarios. "
                    "Consider volatility-targeting or vol-adjusted position sizing."
                )

        # Recommendations from reverse stress tests
        if reverse_results:
            market_shocks = [r for r in reverse_results if 'market_shock' in r.breaking_scenario]
            if market_shocks:
                min_shock = min([abs(r.breaking_parameters.get('shock_percentage', 0))
                                for r in market_shocks])
                recommendations.append(
                    f"Strategy breaks at {min_shock*100:.0f}% market shock. "
                    "Consider implementing portfolio hedging for tail risk protection."
                )

            vol_breaks = [r for r in reverse_results if 'volatility' in r.breaking_scenario]
            if vol_breaks:
                min_vol = min([r.breaking_parameters.get('volatility_multiplier', 1)
                              for r in vol_breaks])
                recommendations.append(
                    f"Strategy vulnerable to {min_vol:.1f}x volatility spikes. "
                    "Consider adding volatility filters to trading logic."
                )

        if not recommendations:
            recommendations.append(
                "Strategy shows reasonable resilience to tested stress scenarios. "
                "Continue monitoring and periodic stress testing."
            )

        return recommendations

    def save_report(
        self,
        report: StressTestReport,
        output_dir: str = './stress_test_results',
        format: str = 'all'
    ) -> Dict[str, str]:
        """
        Save stress test report to files.

        Args:
            report: StressTestReport to save
            output_dir: Directory to save files
            format: 'json', 'markdown', 'csv', or 'all'

        Returns:
            Dictionary of file paths created
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        files_created = {}

        if format in ['json', 'all']:
            json_file = f'{output_dir}/stress_report_{timestamp}.json'
            with open(json_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            files_created['json'] = json_file
            logger.info(f"JSON report saved to {json_file}")

        if format in ['markdown', 'all']:
            md_file = f'{output_dir}/stress_report_{timestamp}.md'
            self._generate_markdown_report(report, md_file)
            files_created['markdown'] = md_file
            logger.info(f"Markdown report saved to {md_file}")

        if format in ['csv', 'all']:
            # Save historical results
            if report.historical_results:
                hist_file = f'{output_dir}/historical_stress_{timestamp}.csv'
                df = pd.DataFrame([r.to_dict() for r in report.historical_results])
                df.to_csv(hist_file, index=False)
                files_created['historical_csv'] = hist_file

            # Save hypothetical results
            if report.hypothetical_results:
                hypo_file = f'{output_dir}/hypothetical_stress_{timestamp}.csv'
                df = pd.DataFrame([r.to_dict() for r in report.hypothetical_results])
                df.to_csv(hypo_file, index=False)
                files_created['hypothetical_csv'] = hypo_file

            # Save reverse stress results
            if report.reverse_stress_results:
                rev_file = f'{output_dir}/reverse_stress_{timestamp}.csv'
                df = pd.DataFrame([r.to_dict() for r in report.reverse_stress_results])
                df.to_csv(rev_file, index=False)
                files_created['reverse_csv'] = rev_file

        return files_created

    def _generate_markdown_report(self, report: StressTestReport, output_file: str):
        """Generate a markdown stress test report"""
        with open(output_file, 'w') as f:
            f.write(f"# Stress Test Report: {report.strategy_name}\n\n")
            f.write(f"**Generated:** {report.generated_at}\n\n")
            f.write(f"**Symbols Tested:** {', '.join(report.symbols)}\n\n")
            f.write(f"**Risk Rating:** **{report.risk_rating.upper()}**\n\n")

            # Risk Factors
            f.write("## Risk Factors\n\n")
            for factor in report.risk_factors:
                f.write(f"- {factor}\n")
            f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            for i, rec in enumerate(report.recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")

            # Summary Statistics
            f.write("## Summary Statistics\n\n")
            if report.summary:
                f.write(f"- **Total Scenarios Tested:** {report.summary.get('total_scenarios_tested', 0)}\n")
                f.write(f"- **Average Return:** {report.summary.get('average_return', 0):.2f}%\n")
                f.write(f"- **Worst Return:** {report.summary.get('worst_return', 0):.2f}%\n")
                f.write(f"- **Average Max Drawdown:** {report.summary.get('average_max_drawdown', 0):.2f}%\n")
                f.write(f"- **Worst Max Drawdown:** {report.summary.get('worst_max_drawdown', 0):.2f}%\n")
                f.write(f"- **Scenarios with Loss:** {report.summary.get('scenarios_with_loss', 0)}\n")
                f.write(f"- **Scenarios with Severe Drawdown (>30%):** {report.summary.get('scenarios_with_severe_drawdown', 0)}\n")
            f.write("\n")

            # Historical Stress Tests
            if report.historical_results:
                f.write("## Historical Stress Test Results\n\n")
                f.write("| Scenario | Symbol | Return | Max DD | Sharpe | VaR 95% | Recovery Days |\n")
                f.write("|----------|--------|--------|--------|--------|---------|---------------|\n")
                for r in report.historical_results:
                    recovery = r.recovery_time_days if r.recovery_time_days else "N/A"
                    f.write(f"| {r.scenario_name} | {r.symbol} | {r.total_return:.2f}% | "
                           f"{r.max_drawdown:.2f}% | {r.sharpe_ratio:.2f} | "
                           f"{r.var_95:.2f}% | {recovery} |\n")
                f.write("\n")

            # Hypothetical Stress Tests
            if report.hypothetical_results:
                f.write("## Hypothetical Stress Test Results\n\n")
                f.write("| Scenario | Symbol | Return | Max DD | Sharpe | Return Degradation |\n")
                f.write("|----------|--------|--------|--------|--------|--------------------|\n")
                for r in report.hypothetical_results:
                    degradation = f"{r.return_degradation:.1f}%" if r.return_degradation else "N/A"
                    f.write(f"| {r.scenario_name} | {r.symbol} | {r.total_return:.2f}% | "
                           f"{r.max_drawdown:.2f}% | {r.sharpe_ratio:.2f} | {degradation} |\n")
                f.write("\n")

            # Reverse Stress Tests
            if report.reverse_stress_results:
                f.write("## Reverse Stress Test Results (Breaking Scenarios)\n\n")
                f.write("| Symbol | Scenario | Threshold | Actual | Description |\n")
                f.write("|--------|----------|-----------|--------|-------------|\n")
                for r in report.reverse_stress_results:
                    f.write(f"| {r.symbol} | {r.breaking_scenario} | "
                           f"{r.threshold_metric} > {r.threshold_value} | "
                           f"{r.actual_value:.2f} | {r.description} |\n")
                f.write("\n")


# =============================================================================
# Convenience Functions
# =============================================================================

def run_full_stress_test(
    strategy_func: Callable,
    strategy_name: str,
    symbols: List[str],
    data_fetcher: Any,
    output_dir: str = './stress_test_results',
    include_historical: bool = True,
    include_hypothetical: bool = True,
    include_reverse: bool = True,
    historical_scenarios: Optional[List[str]] = None,
    hypothetical_scenarios: Optional[List[str]] = None
) -> StressTestReport:
    """
    Convenience function to run a complete stress test suite.

    Args:
        strategy_func: Strategy function that takes (backtester, date, row, symbol, data)
        strategy_name: Name of the strategy
        symbols: List of symbols to test
        data_fetcher: DataFetcher instance
        output_dir: Directory to save results
        include_historical: Run historical stress tests
        include_hypothetical: Run hypothetical stress tests
        include_reverse: Run reverse stress tests
        historical_scenarios: Specific historical scenarios to test
        hypothetical_scenarios: Specific hypothetical scenarios to test

    Returns:
        StressTestReport with all results
    """
    framework = StressTestFramework(data_fetcher)

    historical_results = []
    hypothetical_results = []
    reverse_results = []

    if include_historical:
        logger.info("Running historical stress tests...")
        historical_results = framework.run_historical_stress_test(
            strategy_func=strategy_func,
            strategy_name=strategy_name,
            symbols=symbols,
            scenarios=historical_scenarios
        )

    if include_hypothetical:
        logger.info("Running hypothetical stress tests...")
        hypothetical_results = framework.run_hypothetical_stress_test(
            strategy_func=strategy_func,
            strategy_name=strategy_name,
            symbols=symbols,
            scenarios=hypothetical_scenarios
        )

    if include_reverse:
        logger.info("Running reverse stress tests...")
        reverse_results = framework.run_reverse_stress_test(
            strategy_func=strategy_func,
            strategy_name=strategy_name,
            symbols=symbols
        )

    # Generate report
    report = framework.generate_stress_report(
        strategy_name=strategy_name,
        symbols=symbols,
        historical_results=historical_results,
        hypothetical_results=hypothetical_results,
        reverse_results=reverse_results
    )

    # Save report
    framework.save_report(report, output_dir)

    return report
