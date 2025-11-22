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
from quantsploit.utils.ta_compat import rsi, atr, adx, bbands

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


def parse_time_span(time_str: str) -> int:
    """
    Parse a time span string into number of days

    Args:
        time_str: Time span string (e.g., '2y', '6m', '180d', '4w')

    Returns:
        Number of days

    Examples:
        parse_time_span('2y') -> 730
        parse_time_span('6m') -> 180
        parse_time_span('4w') -> 28
        parse_time_span('90d') -> 90
    """
    import re

    time_str = time_str.strip().lower()
    match = re.match(r'^(\d+)([ymwd])$', time_str)

    if not match:
        raise ValueError(f"Invalid time span format: {time_str}. Use format like '2y', '6m', '4w', or '180d'")

    value = int(match.group(1))
    unit = match.group(2)

    # Convert to days
    if unit == 'y':
        return value * 365
    elif unit == 'm':
        return value * 30  # Approximate month as 30 days
    elif unit == 'w':
        return value * 7
    elif unit == 'd':
        return value
    else:
        raise ValueError(f"Unknown time unit: {unit}")


def parse_quarters(quarter_str: str) -> List[int]:
    """
    Parse a quarter specification string

    Args:
        quarter_str: Quarter string (e.g., '2' for Q2, '1,2,3' for Q1-Q3)

    Returns:
        List of quarter numbers (1-4)

    Examples:
        parse_quarters('2') -> [2]
        parse_quarters('1,2,3') -> [1, 2, 3]
        parse_quarters('4') -> [4]
    """
    quarter_str = quarter_str.strip()

    # Handle comma-separated quarters
    quarter_parts = [q.strip() for q in quarter_str.split(',')]
    quarters = []

    for part in quarter_parts:
        try:
            q = int(part)
            if q < 1 or q > 4:
                raise ValueError(f"Quarter must be between 1 and 4, got {q}")
            quarters.append(q)
        except ValueError as e:
            raise ValueError(f"Invalid quarter specification: {quarter_str}. Use format like '2' or '1,2,3'")

    # Remove duplicates and sort
    quarters = sorted(list(set(quarters)))

    return quarters


def get_fiscal_quarter_dates(year: int, quarter: int) -> Tuple[datetime, datetime]:
    """
    Get the start and end dates for a fiscal quarter

    Args:
        year: The year
        quarter: The quarter number (1-4)

    Returns:
        Tuple of (start_date, end_date) for the quarter

    Examples:
        get_fiscal_quarter_dates(2024, 1) -> (Jan 1, Mar 31)
        get_fiscal_quarter_dates(2024, 2) -> (Apr 1, Jun 30)
    """
    # Quarter start months: Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
    quarter_start_months = {1: 1, 2: 4, 3: 7, 4: 10}
    quarter_end_months = {1: 3, 2: 6, 3: 9, 4: 12}
    quarter_end_days = {1: 31, 2: 30, 3: 30, 4: 31}

    start_month = quarter_start_months[quarter]
    end_month = quarter_end_months[quarter]
    end_day = quarter_end_days[quarter]

    start_date = datetime(year, start_month, 1)
    end_date = datetime(year, end_month, end_day, 23, 59, 59)

    return start_date, end_date


def find_quarter_periods(quarters: List[int], num_periods: Optional[int] = None) -> List[TestPeriod]:
    """
    Find test periods based on fiscal quarters

    Args:
        quarters: List of quarter numbers (1-4)
        num_periods: Number of occurrences to include (None = most recent only)

    Returns:
        List of TestPeriod objects

    Examples:
        find_quarter_periods([2], 4) -> Last 4 Q2s
        find_quarter_periods([1, 2, 3], None) -> Most recent Q1, Q2, Q3
    """
    periods = []
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month

    # Determine current quarter
    current_quarter = (current_month - 1) // 3 + 1

    if len(quarters) == 1:
        # Single quarter mode with multiple periods
        target_quarter = quarters[0]
        periods_to_find = num_periods if num_periods else 1

        # Start from current year and work backwards
        year = current_year
        count = 0

        # If we haven't finished the current quarter yet, start from last year
        if target_quarter == current_quarter:
            # Check if we're past the end of this quarter
            _, quarter_end = get_fiscal_quarter_dates(year, target_quarter)
            if current_date <= quarter_end:
                year -= 1
        elif target_quarter > current_quarter:
            year -= 1

        while count < periods_to_find:
            start_date, end_date = get_fiscal_quarter_dates(year, target_quarter)

            # Don't include future periods
            if start_date <= current_date:
                periods.append(TestPeriod(
                    name=f'Q{target_quarter}_{year}',
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    description=f'Q{target_quarter} {year}'
                ))
                count += 1

            year -= 1

    else:
        # Range of quarters (e.g., Q1, Q2, Q3)
        # Find the most recent complete occurrence of this range
        if num_periods:
            # Multiple occurrences of the quarter range
            for occurrence in range(num_periods):
                for quarter in quarters:
                    year = current_year

                    # Adjust year based on current quarter and target quarter
                    if quarter > current_quarter:
                        year -= 1

                    # Adjust for occurrence
                    year -= occurrence

                    start_date, end_date = get_fiscal_quarter_dates(year, quarter)

                    # Don't include future periods
                    if start_date <= current_date:
                        periods.append(TestPeriod(
                            name=f'Q{quarter}_{year}',
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d'),
                            description=f'Q{quarter} {year}'
                        ))
        else:
            # Most recent occurrence of each quarter in the range
            for quarter in quarters:
                year = current_year

                # If the target quarter is in the future or current (incomplete), go back a year
                if quarter > current_quarter:
                    year -= 1
                elif quarter == current_quarter:
                    # Check if current quarter is complete
                    _, quarter_end = get_fiscal_quarter_dates(year, quarter)
                    if current_date <= quarter_end:
                        year -= 1

                start_date, end_date = get_fiscal_quarter_dates(year, quarter)

                periods.append(TestPeriod(
                    name=f'Q{quarter}_{year}',
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    description=f'Q{quarter} {year}'
                ))

    return periods


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

    def multifactor_strategy(self, backtester: Backtester, date: pd.Timestamp,
                            row: pd.Series, symbol: str, data: pd.DataFrame,
                            lookback: int = 20):
        """Multifactor Scoring Strategy Adapter"""
        history = data.loc[:date]

        if len(history) < 50:
            return

        # Calculate simplified multifactor score
        close = history['Close']

        # Momentum score (0-100)
        if len(close) >= 20:
            roc_20 = ((close.iloc[-1] / close.iloc[-20]) - 1) * 100
            momentum_score = 50 + np.clip(roc_20, -25, 25)
        else:
            momentum_score = 50

        # Technical score (0-100)
        rsi_val = rsi(close, 14)
        if len(rsi_val) > 0 and not pd.isna(rsi_val.iloc[-1]):
            current_rsi = rsi_val.iloc[-1]
            if current_rsi < 30:
                technical_score = 70
            elif current_rsi > 70:
                technical_score = 30
            else:
                technical_score = 50
        else:
            technical_score = 50

        # Composite score (simple average)
        composite_score = (momentum_score + technical_score) / 2

        current_position = backtester.positions.get(symbol)

        # Enter on high composite score
        if composite_score >= 60:
            if current_position is None:
                backtester.enter_long(symbol, date, row['Close'])

        # Exit on low composite score
        elif composite_score < 40:
            if current_position is not None:
                backtester.exit_position(symbol, date, row['Close'])

    def kalman_adaptive_strategy(self, backtester: Backtester, date: pd.Timestamp,
                                row: pd.Series, symbol: str, data: pd.DataFrame,
                                threshold: float = 0.5, process_noise: float = 0.01):
        """Kalman Filter Adaptive Strategy Adapter"""
        history = data.loc[:date]

        if len(history) < 50:
            return

        # Simple 1D Kalman filter
        prices = history['Close'].values
        n = len(prices)
        filtered = np.zeros(n)
        filtered[0] = prices[0]
        P = 1.0
        Q = process_noise
        R = 1.0

        for i in range(1, n):
            # Prediction
            x_pred = filtered[i - 1]
            P_pred = P + Q

            # Update
            K = P_pred / (P_pred + R)
            filtered[i] = x_pred + K * (prices[i] - x_pred)
            P = (1 - K) * P_pred

        # Calculate deviation
        current_price = row['Close']
        filtered_price = filtered[-1]
        deviation = (current_price - filtered_price) / filtered_price * 100

        current_position = backtester.positions.get(symbol)

        # Buy when price below filtered (oversold)
        if deviation < -threshold:
            if current_position is None:
                backtester.enter_long(symbol, date, row['Close'])

        # Sell when price above filtered (overbought)
        elif deviation > threshold:
            if current_position is not None:
                backtester.exit_position(symbol, date, row['Close'])

    def volume_profile_strategy(self, backtester: Backtester, date: pd.Timestamp,
                               row: pd.Series, symbol: str, data: pd.DataFrame,
                               profile_period: int = 20, num_levels: int = 50):
        """Volume Profile Swing Strategy Adapter"""
        history = data.loc[:date]

        if len(history) < profile_period + 1:
            return

        # Get recent window for volume profile
        window = history.iloc[-profile_period:]

        # Calculate simple volume profile
        price_min = window['Low'].min()
        price_max = window['High'].max()

        if price_max <= price_min:
            return

        price_bins = np.linspace(price_min, price_max, num_levels + 1)
        price_levels = (price_bins[:-1] + price_bins[1:]) / 2
        volume_at_levels = np.zeros(num_levels)

        # Distribute volume
        for idx, bar in window.iterrows():
            bar_low = bar['Low']
            bar_high = bar['High']
            bar_volume = bar['Volume']

            for i, price in enumerate(price_levels):
                if bar_low <= price <= bar_high:
                    volume_at_levels[i] += bar_volume / num_levels

        # Find POC and value area
        poc_idx = np.argmax(volume_at_levels)
        poc_price = price_levels[poc_idx]

        # Find value area (70% of volume)
        total_volume = volume_at_levels.sum()
        target_volume = total_volume * 0.7
        va_volume = volume_at_levels[poc_idx]
        lower_idx = upper_idx = poc_idx

        while va_volume < target_volume and (lower_idx > 0 or upper_idx < len(volume_at_levels) - 1):
            lower_vol = volume_at_levels[lower_idx - 1] if lower_idx > 0 else 0
            upper_vol = volume_at_levels[upper_idx + 1] if upper_idx < len(volume_at_levels) - 1 else 0

            if lower_vol >= upper_vol and lower_idx > 0:
                lower_idx -= 1
                va_volume += volume_at_levels[lower_idx]
            elif upper_idx < len(volume_at_levels) - 1:
                upper_idx += 1
                va_volume += volume_at_levels[upper_idx]
            else:
                break

        va_low = price_levels[lower_idx]
        va_high = price_levels[upper_idx]

        current_price = row['Close']
        current_position = backtester.positions.get(symbol)

        # Buy at/below value area low
        if current_price <= va_low:
            if current_position is None:
                backtester.enter_long(symbol, date, row['Close'])

        # Sell at/above value area high
        elif current_price >= va_high:
            if current_position is not None:
                backtester.exit_position(symbol, date, row['Close'])

    def hmm_regime_strategy(self, backtester: Backtester, date: pd.Timestamp,
                           row: pd.Series, symbol: str, data: pd.DataFrame,
                           lookback: int = 20):
        """HMM Regime Detection Strategy Adapter (Simplified)"""
        history = data.loc[:date]

        if len(history) < lookback + 20:
            return

        # Simplified regime detection based on trend and volatility
        recent = history.iloc[-lookback:]

        # Calculate regime features
        returns = recent['Close'].pct_change()
        avg_return = returns.mean()
        volatility = returns.std()

        # Simple regime classification
        # Bull: positive returns, moderate vol
        # Bear: negative returns
        # Sideways: low abs returns
        if avg_return > 0.001:
            regime = 'BULL'
        elif avg_return < -0.001:
            regime = 'BEAR'
        else:
            regime = 'SIDEWAYS'

        rsi_val = rsi(history['Close'], 14)
        current_rsi = rsi_val.iloc[-1] if len(rsi_val) > 0 and not pd.isna(rsi_val.iloc[-1]) else 50

        current_position = backtester.positions.get(symbol)

        # Regime-based strategy
        if regime == 'BULL':
            # Trend follow in bull market - buy dips
            if current_rsi < 40 and current_position is None:
                backtester.enter_long(symbol, date, row['Close'])
            elif current_rsi > 70 and current_position is not None:
                backtester.exit_position(symbol, date, row['Close'])

        elif regime == 'BEAR':
            # Defensive in bear market - exit to cash
            if current_position is not None:
                backtester.exit_position(symbol, date, row['Close'])

        else:  # SIDEWAYS
            # Mean reversion in sideways market
            if current_rsi < 30 and current_position is None:
                backtester.enter_long(symbol, date, row['Close'])
            elif current_rsi > 70 and current_position is not None:
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
            # Basic strategies (kept for comparison)
            'sma_crossover': {
                'name': 'SMA Crossover (20/50)',
                'function': self.adapter.sma_crossover_strategy,
                'params': {'short_window': 20, 'long_window': 50}
            },
            'mean_reversion': {
                'name': 'Mean Reversion (20 day)',
                'function': self.adapter.mean_reversion_strategy,
                'params': {'lookback': 20, 'entry_threshold': -60, 'exit_threshold': 40}
            },
            'momentum': {
                'name': 'Momentum (10/20/50)',
                'function': self.adapter.momentum_strategy,
                'params': {'periods': [10, 20, 50], 'entry_threshold': 60, 'exit_threshold': -40}
            },

            # Advanced strategies
            'multifactor_scoring': {
                'name': 'Multi-Factor Scoring',
                'function': self.adapter.multifactor_strategy,
                'params': {'lookback': 20}
            },
            'kalman_adaptive': {
                'name': 'Kalman Adaptive Filter',
                'function': self.adapter.kalman_adaptive_strategy,
                'params': {'threshold': 0.5, 'process_noise': 0.01}
            },
            'kalman_adaptive_sensitive': {
                'name': 'Kalman Adaptive (Sensitive)',
                'function': self.adapter.kalman_adaptive_strategy,
                'params': {'threshold': 0.3, 'process_noise': 0.05}
            },
            'volume_profile_swing': {
                'name': 'Volume Profile Swing',
                'function': self.adapter.volume_profile_strategy,
                'params': {'profile_period': 20, 'num_levels': 50}
            },
            'volume_profile_swing_fast': {
                'name': 'Volume Profile (Fast)',
                'function': self.adapter.volume_profile_strategy,
                'params': {'profile_period': 10, 'num_levels': 30}
            },
            'hmm_regime_detection': {
                'name': 'HMM Regime Detection',
                'function': self.adapter.hmm_regime_strategy,
                'params': {'lookback': 20}
            },
            'hmm_regime_detection_long': {
                'name': 'HMM Regime (Long-term)',
                'function': self.adapter.hmm_regime_strategy,
                'params': {'lookback': 50}
            }
        }

        self.results: List[StrategyPerformance] = []

    def generate_test_periods(self, years_back: int = 3,
                             tspan: Optional[str] = None,
                             bspan: Optional[str] = None,
                             num_periods: Optional[int] = None,
                             quarters: Optional[str] = None) -> List[TestPeriod]:
        """
        Generate test periods for backtesting

        Args:
            years_back: Number of years to look back (used when custom params not provided)
            tspan: Total time span (e.g., '2y', '18m'). If provided, uses custom period generation
            bspan: Backtest span for each period (e.g., '6m', '180d')
            num_periods: Number of separate backtest periods to run
            quarters: Quarter specification (e.g., '2' or '1,2,3')

        Returns:
            List of TestPeriod objects
        """
        periods = []
        end_date = datetime.now()

        # Quarterly analysis mode
        if quarters is not None:
            try:
                quarter_list = parse_quarters(quarters)
                return find_quarter_periods(quarter_list, num_periods)
            except ValueError as e:
                logger.error(f"Error parsing quarter parameters: {e}")
                logger.info("Falling back to default period generation")

        # Custom period generation if parameters provided
        if tspan is not None and bspan is not None and num_periods is not None:
            try:
                total_days = parse_time_span(tspan)
                backtest_days = parse_time_span(bspan)

                # Calculate the starting point for the total span
                total_start_date = end_date - timedelta(days=total_days)

                # Calculate the step size between periods
                # If we have 4 periods over 2 years (730 days) with 6-month (180 days) spans:
                # The periods should be distributed evenly over the total span
                # Step = (total_days - backtest_days) / (num_periods - 1) if num_periods > 1
                if num_periods == 1:
                    step_days = 0
                else:
                    step_days = (total_days - backtest_days) / (num_periods - 1)

                for i in range(num_periods):
                    # Calculate period end date (working backwards from end_date)
                    period_end = end_date - timedelta(days=i * step_days)
                    period_start = period_end - timedelta(days=backtest_days)

                    # Make sure we don't go before the total start date
                    if period_start < total_start_date:
                        period_start = total_start_date

                    periods.append(TestPeriod(
                        name=f'custom_period_{i+1}',
                        start_date=period_start.strftime('%Y-%m-%d'),
                        end_date=period_end.strftime('%Y-%m-%d'),
                        description=f'Custom Period {i+1} ({period_start.strftime("%b %Y")} - {period_end.strftime("%b %Y")})'
                    ))

                return periods

            except ValueError as e:
                logger.error(f"Error parsing time span parameters: {e}")
                logger.info("Falling back to default period generation")

        # Default period generation
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
            full_data = self.data_fetcher.get_stock_data(
                symbol=symbol,
                period='3y',  # Fetch more data to ensure we have enough
                interval='1d'
            )

            if full_data is None or len(full_data) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return None

            # Filter to get test period data for backtest loop
            test_period_data = full_data.loc[period.start_date:period.end_date]

            if len(test_period_data) < 30:
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

            # Create a closure to pass full historical data to strategy for indicator calculation
            def strategy_func(bt, date, row):
                strategy_info['function'](bt, date, row, symbol, full_data, **strategy_info['params'])

            # Run backtest on test period only, but strategies have access to full historical data
            backtester = Backtester(config)
            results = backtester.run_backtest(test_period_data, strategy_func, symbol=symbol)

            # Calculate signal accuracy from completed trades
            completed_trades = [t for t in backtester.trades if t.exit_date is not None]
            correct, total, accuracy = self.calculate_signal_accuracy(
                completed_trades, test_period_data
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
                                   max_workers: int = 4,
                                   tspan: Optional[str] = None,
                                   bspan: Optional[str] = None,
                                   num_periods: Optional[int] = None,
                                   quarters: Optional[str] = None) -> pd.DataFrame:
        """
        Run comprehensive backtests across all strategies, symbols, and periods

        Args:
            parallel: Whether to run backtests in parallel
            max_workers: Maximum number of parallel workers
            tspan: Total time span (e.g., '2y', '18m')
            bspan: Backtest span for each period (e.g., '6m', '180d')
            num_periods: Number of separate backtest periods to run
            quarters: Quarter specification (e.g., '2' or '1,2,3')

        Returns:
            DataFrame with all results
        """
        periods = self.generate_test_periods(
            tspan=tspan,
            bspan=bspan,
            num_periods=num_periods,
            quarters=quarters
        )

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
        # Get the top N rows first
        ranked = df.nlargest(top_n, metric)

        # Select columns, avoiding duplicates by using a set
        columns = ['strategy_name', 'symbol', 'period_name', metric, 'total_return',
                   'win_rate', 'signal_accuracy', 'sharpe_ratio']
        # Remove duplicates while preserving order
        unique_columns = []
        seen = set()
        for col in columns:
            if col not in seen:
                unique_columns.append(col)
                seen.add(col)

        ranked = ranked[unique_columns]
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

        # Flatten the multi-level columns and convert to a JSON-serializable format
        result = {}
        for period in grouped.index:
            result[period] = {}
            for col in grouped.columns:
                # Convert tuple column names to strings (e.g., ('total_return', 'mean') -> 'total_return_mean')
                if isinstance(col, tuple):
                    col_name = '_'.join(str(c) for c in col)
                else:
                    col_name = str(col)
                result[period][col_name] = float(grouped.loc[period, col])

        return result

    def _analyze_by_symbol(self, df: pd.DataFrame) -> Dict:
        """Analyze performance grouped by symbol"""
        grouped = df.groupby('symbol').agg({
            'total_return': ['mean', 'std'],
            'sharpe_ratio': 'mean',
            'win_rate': 'mean',
            'signal_accuracy': 'mean',
            'excess_return': 'mean'
        }).round(4)

        # Flatten the multi-level columns and convert to a JSON-serializable format
        result = {}
        for symbol in grouped.index:
            result[symbol] = {}
            for col in grouped.columns:
                # Convert tuple column names to strings (e.g., ('total_return', 'mean') -> 'total_return_mean')
                if isinstance(col, tuple):
                    col_name = '_'.join(str(c) for c in col)
                else:
                    col_name = str(col)
                result[symbol][col_name] = float(grouped.loc[symbol, col])

        return result

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


def run_comprehensive_analysis(symbols: List[str],
                              output_dir: str = './backtest_results',
                              tspan: Optional[str] = None,
                              bspan: Optional[str] = None,
                              num_periods: Optional[int] = None,
                              quarters: Optional[str] = None):
    """
    Convenience function to run a complete comprehensive backtest

    Args:
        symbols: List of stock symbols to analyze
        output_dir: Directory to save results
        tspan: Total time span (e.g., '2y', '18m')
        bspan: Backtest span for each period (e.g., '6m', '180d')
        num_periods: Number of separate backtest periods to run
        quarters: Quarter specification (e.g., '2' or '1,2,3')
    """
    # Create backtester
    backtester = ComprehensiveBacktester(
        symbols=symbols,
        initial_capital=100000,
        commission_pct=0.001,
        slippage_pct=0.001
    )

    # Run comprehensive backtest
    results_df = backtester.run_comprehensive_backtest(
        parallel=False,
        tspan=tspan,
        bspan=bspan,
        num_periods=num_periods,
        quarters=quarters
    )

    if results_df.empty:
        logger.error("No results generated")
        return None, None

    # Generate summary
    summary = backtester.generate_summary_report(results_df)

    # Save results
    backtester.save_results(results_df, summary, output_dir)

    return results_df, summary
