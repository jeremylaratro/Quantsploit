"""
VWAP Execution Strategy for Quantsploit

This module implements Volume Weighted Average Price (VWAP) execution algorithms
for minimizing market impact when executing large orders.

Key Features:
- Static VWAP scheduling based on historical volume profiles
- Dynamic VWAP adapting to real-time volume
- TWAP (Time Weighted) alternative
- Implementation shortfall tracking
- Market impact estimation
- Integration with backtesting framework

References:
    - Almgren, R. & Chriss, N. (2000). "Optimal Execution of Portfolio Transactions"
    - Kissell, R. & Glantz, M. (2003). "Optimal Trading Strategies"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ExecutionAlgorithm(Enum):
    """Execution algorithm types."""
    VWAP = "vwap"
    TWAP = "twap"
    POV = "pov"  # Percentage of Volume
    IS = "implementation_shortfall"


@dataclass
class OrderSlice:
    """
    Individual order slice for execution.

    Attributes:
        time: Scheduled execution time
        shares: Number of shares to execute
        pct_of_total: Percentage of total order
        price_limit: Optional limit price
        executed_shares: Actual shares executed
        executed_price: Actual execution price
        executed_time: Actual execution time
    """
    time: pd.Timestamp
    shares: int
    pct_of_total: float
    price_limit: Optional[float] = None
    executed_shares: int = 0
    executed_price: float = 0.0
    executed_time: Optional[pd.Timestamp] = None


@dataclass
class ExecutionReport:
    """
    Execution performance report.

    Attributes:
        symbol: Traded symbol
        side: 'buy' or 'sell'
        total_shares: Total order size
        executed_shares: Shares executed
        avg_price: Average execution price
        vwap: Market VWAP over execution window
        arrival_price: Price at order arrival
        slippage_bps: Slippage vs arrival in basis points
        vwap_deviation_bps: Deviation from VWAP in basis points
        implementation_shortfall: Total implementation shortfall
        participation_rate: Order as % of market volume
        execution_time_minutes: Total execution duration
        slices: List of executed order slices
    """
    symbol: str
    side: str
    total_shares: int
    executed_shares: int
    avg_price: float
    vwap: float
    arrival_price: float
    slippage_bps: float
    vwap_deviation_bps: float
    implementation_shortfall: float
    participation_rate: float
    execution_time_minutes: float
    slices: List[OrderSlice] = field(default_factory=list)


class VolumeProfile:
    """
    Intraday volume profile analysis.

    Analyzes historical volume patterns to predict intraday
    volume distribution for VWAP scheduling.
    """

    def __init__(
        self,
        intraday_data: pd.DataFrame,
        interval_minutes: int = 5
    ):
        """
        Initialize Volume Profile.

        Args:
            intraday_data: DataFrame with columns ['timestamp', 'volume'] or similar
            interval_minutes: Time interval for volume bucketing
        """
        self.intraday_data = intraday_data.copy()
        self.interval_minutes = interval_minutes
        self._build_profile()

    def _build_profile(self) -> None:
        """Build average volume profile from historical data."""
        df = self.intraday_data.copy()

        # Ensure datetime index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Extract time of day
        df['time_of_day'] = df.index.time

        # Create time buckets
        df['bucket'] = (df.index.hour * 60 + df.index.minute) // self.interval_minutes

        # Average volume by bucket
        if 'Volume' in df.columns:
            vol_col = 'Volume'
        elif 'volume' in df.columns:
            vol_col = 'volume'
        else:
            vol_col = df.columns[0]

        self.volume_profile = df.groupby('bucket')[vol_col].mean()
        self.volume_profile = self.volume_profile / self.volume_profile.sum()

    def get_volume_schedule(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        total_shares: int
    ) -> List[Tuple[pd.Timestamp, int]]:
        """
        Get volume-weighted execution schedule.

        Args:
            start_time: Start of execution window
            end_time: End of execution window
            total_shares: Total shares to execute

        Returns:
            List of (timestamp, shares) tuples
        """
        # Get relevant buckets
        start_bucket = (start_time.hour * 60 + start_time.minute) // self.interval_minutes
        end_bucket = (end_time.hour * 60 + end_time.minute) // self.interval_minutes

        # Filter profile to execution window
        relevant_profile = self.volume_profile.loc[start_bucket:end_bucket]

        if len(relevant_profile) == 0:
            # Fall back to equal distribution
            n_intervals = max(1, int((end_time - start_time).total_seconds() / 60 / self.interval_minutes))
            shares_per_interval = total_shares // n_intervals

            schedule = []
            current_time = start_time
            remaining = total_shares
            for i in range(n_intervals):
                shares = shares_per_interval if i < n_intervals - 1 else remaining
                schedule.append((current_time, shares))
                remaining -= shares
                current_time += timedelta(minutes=self.interval_minutes)
            return schedule

        # Renormalize profile for execution window
        relevant_profile = relevant_profile / relevant_profile.sum()

        # Create schedule
        schedule = []
        remaining = total_shares
        for bucket, pct in relevant_profile.items():
            bucket_time = start_time.replace(
                hour=int(bucket * self.interval_minutes // 60),
                minute=int(bucket * self.interval_minutes % 60),
                second=0,
                microsecond=0
            )

            shares = int(total_shares * pct)
            shares = min(shares, remaining)

            if shares > 0:
                schedule.append((bucket_time, shares))
                remaining -= shares

        # Add any remaining shares to last slice
        if remaining > 0 and schedule:
            last_time, last_shares = schedule[-1]
            schedule[-1] = (last_time, last_shares + remaining)

        return schedule


class VWAPExecutionStrategy:
    """
    VWAP Execution Strategy.

    Implements volume-weighted average price execution to minimize
    market impact by distributing orders according to historical
    volume patterns.

    ★ Insight ─────────────────────────────────────
    VWAP Execution Key Concepts:
    - VWAP = Σ(Price × Volume) / Σ(Volume)
    - Static VWAP uses historical volume profile
    - Dynamic VWAP adapts to real-time volume
    - Goal: Execute near market VWAP to minimize tracking error
    - Trade-off: Faster execution = more impact, slower = more risk
    ─────────────────────────────────────────────────

    Example:
        >>> strategy = VWAPExecutionStrategy(intraday_data)
        >>> schedule = strategy.create_vwap_schedule(
        ...     symbol='AAPL', total_shares=10000,
        ...     start_time=market_open, end_time=market_close
        ... )
        >>> report = strategy.simulate_execution(schedule, price_data)

    Attributes:
        volume_profile: Intraday volume profile
        market_impact_model: Model for estimating market impact
    """

    def __init__(
        self,
        intraday_data: Optional[pd.DataFrame] = None,
        interval_minutes: int = 5,
        market_impact_coefficient: float = 0.1,  # Impact per sqrt(participation)
        volatility: float = 0.02,  # Daily volatility
        spread_bps: float = 1.0  # Typical bid-ask spread
    ):
        """
        Initialize VWAP Execution Strategy.

        Args:
            intraday_data: Historical intraday data for volume profile
            interval_minutes: Execution interval in minutes
            market_impact_coefficient: Market impact model coefficient
            volatility: Daily volatility for impact estimation
            spread_bps: Typical bid-ask spread in basis points
        """
        self.interval_minutes = interval_minutes
        self.market_impact_coefficient = market_impact_coefficient
        self.volatility = volatility
        self.spread_bps = spread_bps

        if intraday_data is not None:
            self.volume_profile = VolumeProfile(intraday_data, interval_minutes)
        else:
            self.volume_profile = None
            self._create_default_profile()

    def _create_default_profile(self) -> None:
        """Create default U-shaped volume profile."""
        # U-shaped pattern: high at open/close, low at midday
        buckets = 78  # 6.5 hours * 60 / 5 min intervals

        # Create U-shape
        x = np.linspace(0, 1, buckets)
        profile = 1 + 2 * (4 * (x - 0.5) ** 2)  # Parabola

        # Normalize
        profile = profile / profile.sum()

        self.default_profile = pd.Series(profile, index=range(buckets))

    def estimate_market_impact(
        self,
        shares: int,
        adv: float,  # Average daily volume
        price: float,
        execution_window_minutes: int = 390  # Full day
    ) -> float:
        """
        Estimate market impact for an order.

        Uses Almgren-Chriss impact model: Impact ∝ σ × sqrt(participation)

        Args:
            shares: Order size in shares
            adv: Average daily volume
            price: Current price
            execution_window_minutes: Execution window in minutes

        Returns:
            Estimated impact as percentage
        """
        # Participation rate
        window_fraction = execution_window_minutes / 390  # vs full day
        expected_volume = adv * window_fraction
        participation = shares / expected_volume if expected_volume > 0 else 1.0

        # Square root impact model
        impact = self.market_impact_coefficient * self.volatility * np.sqrt(participation)

        # Add half-spread cost
        impact += self.spread_bps / 10000 / 2

        return impact

    def create_vwap_schedule(
        self,
        symbol: str,
        total_shares: int,
        side: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        price: Optional[float] = None,
        participation_limit: float = 0.10  # Max 10% of volume
    ) -> List[OrderSlice]:
        """
        Create VWAP execution schedule.

        Args:
            symbol: Stock symbol
            total_shares: Total shares to execute
            side: 'buy' or 'sell'
            start_time: Start of execution window
            end_time: End of execution window
            price: Current price (for limit calculation)
            participation_limit: Maximum participation rate

        Returns:
            List of OrderSlice objects
        """
        if self.volume_profile is not None:
            raw_schedule = self.volume_profile.get_volume_schedule(
                start_time, end_time, total_shares
            )
        else:
            # Use default profile
            n_intervals = max(1, int((end_time - start_time).total_seconds() / 60 / self.interval_minutes))

            # Map to default profile
            schedule = []
            remaining = total_shares
            current_time = start_time

            for i in range(n_intervals):
                pct = self.default_profile.iloc[i % len(self.default_profile)]
                shares = int(total_shares * pct)
                shares = min(shares, remaining)

                if shares > 0:
                    schedule.append((current_time, shares))
                    remaining -= shares

                current_time += timedelta(minutes=self.interval_minutes)

            raw_schedule = schedule

        # Convert to OrderSlice objects
        slices = []
        for time, shares in raw_schedule:
            slices.append(OrderSlice(
                time=time,
                shares=shares,
                pct_of_total=shares / total_shares if total_shares > 0 else 0,
                price_limit=None
            ))

        return slices

    def create_twap_schedule(
        self,
        symbol: str,
        total_shares: int,
        side: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        price: Optional[float] = None
    ) -> List[OrderSlice]:
        """
        Create TWAP (Time Weighted) execution schedule.

        Equal shares per time interval, ignoring volume profile.

        Args:
            symbol: Stock symbol
            total_shares: Total shares to execute
            side: 'buy' or 'sell'
            start_time: Start of execution window
            end_time: End of execution window
            price: Current price (optional)

        Returns:
            List of OrderSlice objects
        """
        n_intervals = max(1, int((end_time - start_time).total_seconds() / 60 / self.interval_minutes))
        shares_per_interval = total_shares // n_intervals

        slices = []
        remaining = total_shares
        current_time = start_time

        for i in range(n_intervals):
            shares = shares_per_interval if i < n_intervals - 1 else remaining

            slices.append(OrderSlice(
                time=current_time,
                shares=shares,
                pct_of_total=shares / total_shares if total_shares > 0 else 0,
                price_limit=None
            ))

            remaining -= shares
            current_time += timedelta(minutes=self.interval_minutes)

        return slices

    def create_pov_schedule(
        self,
        symbol: str,
        total_shares: int,
        side: str,
        participation_rate: float,
        start_time: pd.Timestamp,
        max_duration_minutes: int = 390,
        volume_data: Optional[pd.Series] = None
    ) -> List[OrderSlice]:
        """
        Create Percentage of Volume (POV) execution schedule.

        Execute as a fixed percentage of market volume.

        Args:
            symbol: Stock symbol
            total_shares: Total shares to execute
            side: 'buy' or 'sell'
            participation_rate: Target % of volume (e.g., 0.10 for 10%)
            start_time: Start of execution window
            max_duration_minutes: Maximum execution duration
            volume_data: Expected volume per interval

        Returns:
            List of OrderSlice objects
        """
        if volume_data is None:
            # Estimate using profile
            n_intervals = max_duration_minutes // self.interval_minutes

            if self.volume_profile is not None:
                expected_volume = self.volume_profile.volume_profile.iloc[:n_intervals]
            else:
                expected_volume = pd.Series([1.0 / n_intervals] * n_intervals)

            # Assume 1M shares ADV scaled to window
            adv = 1000000
            volume_data = expected_volume * adv

        slices = []
        remaining = total_shares
        current_time = start_time

        for interval_volume in volume_data:
            if remaining <= 0:
                break

            target_shares = int(interval_volume * participation_rate)
            shares = min(target_shares, remaining)

            if shares > 0:
                slices.append(OrderSlice(
                    time=current_time,
                    shares=shares,
                    pct_of_total=shares / total_shares if total_shares > 0 else 0,
                    price_limit=None
                ))

                remaining -= shares

            current_time += timedelta(minutes=self.interval_minutes)

        return slices

    def simulate_execution(
        self,
        schedule: List[OrderSlice],
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        symbol: str = 'UNKNOWN',
        side: str = 'buy'
    ) -> ExecutionReport:
        """
        Simulate execution of order schedule.

        Args:
            schedule: List of OrderSlice objects
            price_data: Price data with DatetimeIndex
            volume_data: Volume data for VWAP calculation
            symbol: Stock symbol
            side: 'buy' or 'sell'

        Returns:
            ExecutionReport with execution metrics
        """
        if len(schedule) == 0:
            return ExecutionReport(
                symbol=symbol,
                side=side,
                total_shares=0,
                executed_shares=0,
                avg_price=0,
                vwap=0,
                arrival_price=0,
                slippage_bps=0,
                vwap_deviation_bps=0,
                implementation_shortfall=0,
                participation_rate=0,
                execution_time_minutes=0,
                slices=[]
            )

        # Get arrival price
        arrival_time = schedule[0].time
        try:
            if 'Close' in price_data.columns:
                arrival_idx = price_data.index.get_indexer([arrival_time], method='nearest')[0]
                arrival_price = price_data.iloc[arrival_idx]['Close']
            else:
                arrival_price = price_data.iloc[0]
        except Exception:
            arrival_price = 100.0  # Default

        # Simulate each slice
        total_cost = 0
        total_shares = 0
        total_volume = 0

        for slice_order in schedule:
            try:
                # Find nearest price
                idx = price_data.index.get_indexer([slice_order.time], method='nearest')[0]

                if 'Close' in price_data.columns:
                    base_price = price_data.iloc[idx]['Close']
                else:
                    base_price = price_data.iloc[idx]

                # Add market impact
                participation = 0.05  # Assume 5% of interval volume
                impact = self.market_impact_coefficient * self.volatility * np.sqrt(participation)

                if side == 'buy':
                    exec_price = base_price * (1 + impact)
                else:
                    exec_price = base_price * (1 - impact)

                # Record execution
                slice_order.executed_shares = slice_order.shares
                slice_order.executed_price = exec_price
                slice_order.executed_time = slice_order.time

                total_cost += slice_order.shares * exec_price
                total_shares += slice_order.shares

                # Track for VWAP calculation
                if volume_data is not None:
                    try:
                        vol = volume_data.loc[slice_order.time]
                        if isinstance(vol, pd.Series):
                            vol = vol.iloc[0]
                        total_volume += vol
                    except Exception:
                        total_volume += 1000  # Default volume

            except Exception as e:
                logger.warning(f"Error executing slice at {slice_order.time}: {e}")

        # Calculate metrics
        if total_shares == 0:
            avg_price = 0
        else:
            avg_price = total_cost / total_shares

        # Calculate market VWAP over execution window
        start_time = schedule[0].time
        end_time = schedule[-1].time

        try:
            window_data = price_data.loc[start_time:end_time]
            if volume_data is not None:
                window_vol = volume_data.loc[start_time:end_time]
                if 'Close' in window_data.columns:
                    market_vwap = (window_data['Close'] * window_vol).sum() / window_vol.sum()
                else:
                    market_vwap = (window_data * window_vol).sum() / window_vol.sum()
            else:
                if 'Close' in window_data.columns:
                    market_vwap = window_data['Close'].mean()
                else:
                    market_vwap = window_data.mean()
        except Exception:
            market_vwap = avg_price

        # Calculate slippage and shortfall
        if arrival_price > 0:
            if side == 'buy':
                slippage_bps = (avg_price / arrival_price - 1) * 10000
            else:
                slippage_bps = (arrival_price / avg_price - 1) * 10000
        else:
            slippage_bps = 0

        if market_vwap > 0:
            vwap_deviation_bps = (avg_price / market_vwap - 1) * 10000
        else:
            vwap_deviation_bps = 0

        # Implementation shortfall
        implementation_shortfall = total_shares * (avg_price - arrival_price)
        if side == 'sell':
            implementation_shortfall = -implementation_shortfall

        # Execution time
        if len(schedule) > 1:
            execution_time = (schedule[-1].time - schedule[0].time).total_seconds() / 60
        else:
            execution_time = 0

        return ExecutionReport(
            symbol=symbol,
            side=side,
            total_shares=sum(s.shares for s in schedule),
            executed_shares=total_shares,
            avg_price=avg_price,
            vwap=market_vwap,
            arrival_price=arrival_price,
            slippage_bps=slippage_bps,
            vwap_deviation_bps=vwap_deviation_bps,
            implementation_shortfall=implementation_shortfall,
            participation_rate=total_shares / total_volume if total_volume > 0 else 0,
            execution_time_minutes=execution_time,
            slices=schedule
        )

    def optimize_execution_window(
        self,
        total_shares: int,
        adv: float,
        price: float,
        urgency: str = 'medium',  # low, medium, high
        risk_aversion: float = 1.0
    ) -> Dict:
        """
        Optimize execution window based on order size and urgency.

        Balances market impact vs timing risk using Almgren-Chriss framework.

        Args:
            total_shares: Total order size
            adv: Average daily volume
            price: Current price
            urgency: Urgency level
            risk_aversion: Risk aversion parameter

        Returns:
            Optimal execution parameters
        """
        participation = total_shares / adv

        # Base execution window (minutes)
        base_windows = {
            'low': 390,    # Full day
            'medium': 195,  # Half day
            'high': 60      # 1 hour
        }

        base_window = base_windows.get(urgency, 195)

        # Adjust for order size
        if participation > 0.05:  # Large order
            window_mult = np.sqrt(participation / 0.05)
            optimal_window = min(base_window * window_mult, 390)
        else:
            optimal_window = base_window

        # Estimate costs
        impact_cost = self.estimate_market_impact(
            total_shares, adv, price, int(optimal_window)
        )

        # Timing risk (opportunity cost from price moves)
        timing_risk = self.volatility * np.sqrt(optimal_window / 390) / np.sqrt(252)

        # Total expected cost
        total_cost = impact_cost + risk_aversion * timing_risk

        return {
            'optimal_window_minutes': int(optimal_window),
            'estimated_impact_bps': impact_cost * 10000,
            'timing_risk_bps': timing_risk * 10000,
            'total_expected_cost_bps': total_cost * 10000,
            'recommended_algorithm': 'VWAP' if optimal_window >= 60 else 'POV',
            'participation_rate': participation
        }

    def run_backtest(
        self,
        orders: List[Dict],
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        algorithm: str = 'vwap'
    ) -> Dict:
        """
        Backtest execution algorithm across multiple orders.

        Args:
            orders: List of order dictionaries with keys:
                    ['symbol', 'shares', 'side', 'arrival_time', 'deadline']
            price_data: Historical price data
            volume_data: Historical volume data
            algorithm: Execution algorithm ('vwap', 'twap', 'pov')

        Returns:
            Backtest results dictionary
        """
        reports = []

        for order in orders:
            # Create schedule based on algorithm
            if algorithm == 'vwap':
                schedule = self.create_vwap_schedule(
                    symbol=order['symbol'],
                    total_shares=order['shares'],
                    side=order['side'],
                    start_time=order['arrival_time'],
                    end_time=order['deadline']
                )
            elif algorithm == 'twap':
                schedule = self.create_twap_schedule(
                    symbol=order['symbol'],
                    total_shares=order['shares'],
                    side=order['side'],
                    start_time=order['arrival_time'],
                    end_time=order['deadline']
                )
            else:
                schedule = self.create_pov_schedule(
                    symbol=order['symbol'],
                    total_shares=order['shares'],
                    side=order['side'],
                    participation_rate=0.10,
                    start_time=order['arrival_time']
                )

            # Simulate execution
            report = self.simulate_execution(
                schedule=schedule,
                price_data=price_data,
                volume_data=volume_data,
                symbol=order['symbol'],
                side=order['side']
            )

            reports.append(report)

        # Aggregate metrics
        if len(reports) == 0:
            return {'error': 'No orders processed'}

        avg_slippage = np.mean([r.slippage_bps for r in reports])
        avg_vwap_dev = np.mean([abs(r.vwap_deviation_bps) for r in reports])
        total_shortfall = sum(r.implementation_shortfall for r in reports)

        # Win rate (executed better than VWAP)
        win_rate = np.mean([
            (r.avg_price <= r.vwap if r.side == 'buy' else r.avg_price >= r.vwap)
            for r in reports
        ])

        return {
            'n_orders': len(orders),
            'algorithm': algorithm,
            'avg_slippage_bps': avg_slippage,
            'avg_vwap_deviation_bps': avg_vwap_dev,
            'total_implementation_shortfall': total_shortfall,
            'vwap_beat_rate': win_rate,
            'reports': reports
        }


class DynamicVWAPExecutor:
    """
    Dynamic VWAP Executor that adapts to real-time volume.

    Adjusts execution pace based on actual market volume vs expected,
    allowing for opportunistic execution during volume spikes.

    ★ Insight ─────────────────────────────────────
    Dynamic VWAP Advantages:
    - Adapts to actual volume vs static profile
    - Can pause during low volume to reduce impact
    - Accelerates during volume spikes for natural execution
    - Better VWAP tracking in abnormal volume days
    ─────────────────────────────────────────────────
    """

    def __init__(
        self,
        expected_profile: pd.Series,
        total_shares: int,
        side: str,
        interval_minutes: int = 5,
        catchup_aggressiveness: float = 1.5
    ):
        """
        Initialize Dynamic VWAP Executor.

        Args:
            expected_profile: Expected volume profile (normalized)
            total_shares: Total shares to execute
            side: 'buy' or 'sell'
            interval_minutes: Execution interval
            catchup_aggressiveness: How aggressive to catch up (1.0 = normal)
        """
        self.expected_profile = expected_profile
        self.total_shares = total_shares
        self.side = side
        self.interval_minutes = interval_minutes
        self.catchup_aggressiveness = catchup_aggressiveness

        # State
        self.shares_executed = 0
        self.target_shares_by_now = 0
        self.current_interval = 0

    def get_next_slice(
        self,
        actual_volume: float,
        expected_volume: float,
        current_price: float
    ) -> OrderSlice:
        """
        Get next execution slice based on actual vs expected volume.

        Args:
            actual_volume: Actual volume in current interval
            expected_volume: Expected volume in current interval
            current_price: Current market price

        Returns:
            OrderSlice for next execution
        """
        # Calculate target based on volume profile
        if self.current_interval < len(self.expected_profile):
            target_pct = self.expected_profile.iloc[self.current_interval]
        else:
            target_pct = 0.01  # Default small slice

        base_shares = int(self.total_shares * target_pct)

        # Volume adjustment
        if expected_volume > 0:
            volume_ratio = actual_volume / expected_volume
        else:
            volume_ratio = 1.0

        # Scale shares by volume ratio
        adjusted_shares = int(base_shares * volume_ratio)

        # Catchup adjustment if behind schedule
        shares_behind = self.target_shares_by_now - self.shares_executed
        if shares_behind > 0:
            catchup_shares = int(shares_behind * self.catchup_aggressiveness * 0.2)
            adjusted_shares += catchup_shares

        # Don't exceed remaining
        remaining = self.total_shares - self.shares_executed
        adjusted_shares = min(adjusted_shares, remaining)
        adjusted_shares = max(0, adjusted_shares)

        # Update state
        self.shares_executed += adjusted_shares
        self.target_shares_by_now += base_shares
        self.current_interval += 1

        return OrderSlice(
            time=pd.Timestamp.now(),
            shares=adjusted_shares,
            pct_of_total=adjusted_shares / self.total_shares if self.total_shares > 0 else 0
        )

    def get_completion_status(self) -> Dict:
        """Get current execution completion status."""
        return {
            'shares_executed': self.shares_executed,
            'shares_remaining': self.total_shares - self.shares_executed,
            'pct_complete': self.shares_executed / self.total_shares if self.total_shares > 0 else 0,
            'target_pct': self.target_shares_by_now / self.total_shares if self.total_shares > 0 else 0,
            'ahead_behind': self.shares_executed - self.target_shares_by_now
        }


def compare_execution_algorithms(
    price_data: pd.DataFrame,
    volume_data: pd.DataFrame,
    order_size: int,
    side: str = 'buy',
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Compare different execution algorithms.

    Args:
        price_data: Price data
        volume_data: Volume data
        order_size: Order size in shares
        side: 'buy' or 'sell'
        start_time: Execution window start
        end_time: Execution window end

    Returns:
        DataFrame comparing algorithm performance
    """
    if start_time is None:
        start_time = price_data.index[0]
    if end_time is None:
        end_time = price_data.index[-1]

    strategy = VWAPExecutionStrategy()

    results = []

    # Test each algorithm
    algorithms = ['vwap', 'twap', 'pov']

    for algo in algorithms:
        if algo == 'vwap':
            schedule = strategy.create_vwap_schedule(
                'TEST', order_size, side, start_time, end_time
            )
        elif algo == 'twap':
            schedule = strategy.create_twap_schedule(
                'TEST', order_size, side, start_time, end_time
            )
        else:
            schedule = strategy.create_pov_schedule(
                'TEST', order_size, side, 0.10, start_time
            )

        report = strategy.simulate_execution(
            schedule, price_data, volume_data, 'TEST', side
        )

        results.append({
            'algorithm': algo.upper(),
            'avg_price': report.avg_price,
            'vwap': report.vwap,
            'slippage_bps': report.slippage_bps,
            'vwap_deviation_bps': report.vwap_deviation_bps,
            'implementation_shortfall': report.implementation_shortfall,
            'execution_time_min': report.execution_time_minutes
        })

    return pd.DataFrame(results)
