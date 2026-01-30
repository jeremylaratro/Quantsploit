"""
Earnings Momentum Strategy for Quantsploit

This module implements earnings-based momentum strategies that trade on
earnings surprise patterns and post-earnings announcement drift (PEAD).

Key Features:
- Standardized Unexpected Earnings (SUE) calculation
- Post-Earnings Announcement Drift (PEAD) trading
- Earnings revision momentum
- Analyst estimate tracking
- Integration with backtesting framework

References:
    - Ball, R. & Brown, P. (1968). "An Empirical Evaluation of Accounting Income Numbers"
    - Bernard, V. & Thomas, J. (1989). "Post-Earnings-Announcement Drift"
    - Chan et al. (1996). "Momentum Strategies"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


@dataclass
class EarningsSurprise:
    """
    Earnings surprise data.

    Attributes:
        symbol: Stock symbol
        date: Announcement date
        actual_eps: Actual EPS reported
        expected_eps: Consensus estimate
        surprise_pct: Surprise as percentage
        sue: Standardized Unexpected Earnings
        reaction: Price reaction around announcement
    """
    symbol: str
    date: pd.Timestamp
    actual_eps: float
    expected_eps: float
    surprise_pct: float
    sue: float
    reaction: Optional[float] = None


@dataclass
class EarningsMomentumSignal:
    """
    Earnings momentum trading signal.

    Attributes:
        symbol: Stock symbol
        date: Signal date
        signal_type: 'long', 'short', or 'neutral'
        strength: Signal strength (0-100)
        sue: Standardized Unexpected Earnings
        revision_momentum: Analyst revision momentum score
        consecutive_beats: Number of consecutive earnings beats/misses
    """
    symbol: str
    date: pd.Timestamp
    signal_type: str
    strength: float
    sue: float
    revision_momentum: float
    consecutive_beats: int


class EarningsMomentumStrategy:
    """
    Earnings Momentum Trading Strategy.

    Implements trading strategies based on earnings surprises and the
    well-documented Post-Earnings Announcement Drift (PEAD) anomaly.
    Stocks with positive earnings surprises tend to continue outperforming
    for 60-90 days after the announcement.

    ★ Insight ─────────────────────────────────────
    Post-Earnings Announcement Drift (PEAD):
    - Positive surprises → continued outperformance
    - Negative surprises → continued underperformance
    - Drift persists for ~60 trading days
    - Strongest in small caps and low analyst coverage
    - Can be enhanced with revision momentum
    ─────────────────────────────────────────────────

    Example:
        >>> strategy = EarningsMomentumStrategy(price_data, earnings_data)
        >>> signals = strategy.generate_signals()
        >>> results = strategy.run_backtest()

    Attributes:
        price_data: DataFrame of stock prices
        earnings_data: DataFrame of earnings announcements
        estimates_data: DataFrame of analyst estimates (optional)
    """

    def __init__(
        self,
        price_data: pd.DataFrame,
        earnings_data: Optional[pd.DataFrame] = None,
        estimates_data: Optional[pd.DataFrame] = None,
        sue_lookback: int = 8,  # Quarters for SUE calculation
        drift_window: int = 60  # Trading days for PEAD
    ):
        """
        Initialize Earnings Momentum Strategy.

        Args:
            price_data: DataFrame with columns ['Date', 'Symbol', 'Close'] or MultiIndex
            earnings_data: DataFrame with columns ['Date', 'Symbol', 'Actual_EPS', 'Expected_EPS']
            estimates_data: DataFrame of analyst estimates for revision tracking
            sue_lookback: Number of quarters for SUE standardization
            drift_window: Days for post-announcement drift window
        """
        self.price_data = price_data
        self.earnings_data = earnings_data
        self.estimates_data = estimates_data
        self.sue_lookback = sue_lookback
        self.drift_window = drift_window

        # Calculate surprises if earnings data provided
        self.surprises = {}
        if earnings_data is not None:
            self._calculate_surprises()

    def _calculate_surprises(self) -> None:
        """Calculate earnings surprises and SUE for all stocks."""
        if self.earnings_data is None:
            return

        for symbol in self.earnings_data['Symbol'].unique():
            symbol_data = self.earnings_data[self.earnings_data['Symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('Date')

            surprises = []
            for _, row in symbol_data.iterrows():
                actual = row['Actual_EPS']
                expected = row.get('Expected_EPS', row.get('Estimate_EPS', None))

                if expected is None or expected == 0:
                    continue

                # Calculate surprise
                surprise = actual - expected
                surprise_pct = (actual / expected - 1) * 100 if expected != 0 else 0

                # Calculate SUE (Standardized Unexpected Earnings)
                sue = self._calculate_sue(symbol_data, row['Date'], surprise)

                # Get price reaction if price data available
                reaction = self._calculate_price_reaction(symbol, row['Date'])

                surprises.append(EarningsSurprise(
                    symbol=symbol,
                    date=row['Date'],
                    actual_eps=actual,
                    expected_eps=expected,
                    surprise_pct=surprise_pct,
                    sue=sue,
                    reaction=reaction
                ))

            self.surprises[symbol] = surprises

    def _calculate_sue(
        self,
        earnings_history: pd.DataFrame,
        current_date: pd.Timestamp,
        current_surprise: float
    ) -> float:
        """
        Calculate Standardized Unexpected Earnings (SUE).

        SUE = (Actual - Expected) / StdDev(Historical Surprises)

        Args:
            earnings_history: Historical earnings data
            current_date: Date of current earnings
            current_surprise: Current surprise value

        Returns:
            SUE score
        """
        # Get historical surprises
        historical = earnings_history[earnings_history['Date'] < current_date]

        if len(historical) < 2:
            return current_surprise / 0.01 if current_surprise != 0 else 0

        # Calculate historical surprises
        hist_surprises = []
        for _, row in historical.tail(self.sue_lookback).iterrows():
            actual = row['Actual_EPS']
            expected = row.get('Expected_EPS', row.get('Estimate_EPS', None))
            if expected is not None:
                hist_surprises.append(actual - expected)

        if len(hist_surprises) < 2:
            return current_surprise / 0.01 if current_surprise != 0 else 0

        std = np.std(hist_surprises)
        if std == 0:
            std = 0.01

        return current_surprise / std

    def _calculate_price_reaction(
        self,
        symbol: str,
        announcement_date: pd.Timestamp,
        window: Tuple[int, int] = (-1, 3)
    ) -> Optional[float]:
        """
        Calculate price reaction around earnings announcement.

        Args:
            symbol: Stock symbol
            announcement_date: Earnings announcement date
            window: Days around announcement (before, after)

        Returns:
            Cumulative return around announcement
        """
        try:
            if isinstance(self.price_data.columns, pd.MultiIndex):
                prices = self.price_data.xs(symbol, level='Symbol', axis=1)['Close']
            else:
                # Assume wide format with symbol as column
                prices = self.price_data[symbol]

            # Find nearest date
            idx = prices.index.get_indexer([announcement_date], method='nearest')[0]

            if idx < abs(window[0]) or idx + window[1] >= len(prices):
                return None

            start_price = prices.iloc[idx + window[0]]
            end_price = prices.iloc[idx + window[1]]

            return (end_price / start_price - 1) * 100

        except Exception:
            return None

    def calculate_revision_momentum(
        self,
        symbol: str,
        lookback_days: int = 90
    ) -> float:
        """
        Calculate analyst revision momentum.

        Measures the trend in analyst estimate revisions.

        Args:
            symbol: Stock symbol
            lookback_days: Days for revision calculation

        Returns:
            Revision momentum score (-100 to 100)
        """
        if self.estimates_data is None:
            return 0

        symbol_estimates = self.estimates_data[
            self.estimates_data['Symbol'] == symbol
        ].copy()

        if len(symbol_estimates) < 2:
            return 0

        # Sort by date
        symbol_estimates = symbol_estimates.sort_values('Date')

        # Calculate revision direction
        revisions = symbol_estimates['Estimate'].diff()
        revisions = revisions.dropna()

        if len(revisions) == 0:
            return 0

        # Score: % of positive revisions - % of negative revisions
        n_up = (revisions > 0).sum()
        n_down = (revisions < 0).sum()
        total = n_up + n_down

        if total == 0:
            return 0

        return ((n_up - n_down) / total) * 100

    def count_consecutive_beats(
        self,
        symbol: str,
        as_of_date: Optional[pd.Timestamp] = None
    ) -> int:
        """
        Count consecutive earnings beats or misses.

        Positive = consecutive beats, Negative = consecutive misses

        Args:
            symbol: Stock symbol
            as_of_date: Calculate as of this date

        Returns:
            Count of consecutive beats (positive) or misses (negative)
        """
        if symbol not in self.surprises:
            return 0

        surprises = self.surprises[symbol]
        if as_of_date is not None:
            surprises = [s for s in surprises if s.date <= as_of_date]

        if len(surprises) == 0:
            return 0

        # Sort by date (most recent first)
        surprises = sorted(surprises, key=lambda x: x.date, reverse=True)

        # Count consecutive same-direction surprises
        count = 0
        direction = None

        for surprise in surprises:
            if surprise.sue > 0.5:  # Meaningful positive surprise
                if direction is None:
                    direction = 'beat'
                    count = 1
                elif direction == 'beat':
                    count += 1
                else:
                    break
            elif surprise.sue < -0.5:  # Meaningful negative surprise
                if direction is None:
                    direction = 'miss'
                    count = -1
                elif direction == 'miss':
                    count -= 1
                else:
                    break
            else:
                break  # Neutral surprise breaks streak

        return count

    def generate_signals(
        self,
        min_sue: float = 1.0,
        require_revision_confirmation: bool = False,
        min_consecutive: int = 1
    ) -> List[EarningsMomentumSignal]:
        """
        Generate earnings momentum signals.

        Args:
            min_sue: Minimum absolute SUE for signal
            require_revision_confirmation: Require revision momentum alignment
            min_consecutive: Minimum consecutive beats/misses

        Returns:
            List of EarningsMomentumSignal objects
        """
        signals = []

        for symbol, surprises in self.surprises.items():
            for surprise in surprises:
                if abs(surprise.sue) < min_sue:
                    continue

                # Get revision momentum
                revision_mom = self.calculate_revision_momentum(symbol)

                # Get consecutive count
                consecutive = self.count_consecutive_beats(symbol, surprise.date)

                # Check revision confirmation
                if require_revision_confirmation:
                    if surprise.sue > 0 and revision_mom < 0:
                        continue
                    if surprise.sue < 0 and revision_mom > 0:
                        continue

                # Check consecutive requirement
                if abs(consecutive) < min_consecutive:
                    continue

                # Determine signal type
                if surprise.sue > 0:
                    signal_type = 'long'
                else:
                    signal_type = 'short'

                # Calculate strength
                strength = self._calculate_signal_strength(
                    sue=surprise.sue,
                    revision_momentum=revision_mom,
                    consecutive=consecutive
                )

                signals.append(EarningsMomentumSignal(
                    symbol=symbol,
                    date=surprise.date,
                    signal_type=signal_type,
                    strength=strength,
                    sue=surprise.sue,
                    revision_momentum=revision_mom,
                    consecutive_beats=consecutive
                ))

        return signals

    def _calculate_signal_strength(
        self,
        sue: float,
        revision_momentum: float,
        consecutive: int
    ) -> float:
        """Calculate signal strength from 0-100."""
        # SUE component (0-40 points)
        sue_score = min(abs(sue) / 3, 1) * 40

        # Revision component (0-30 points)
        revision_score = min(abs(revision_momentum) / 50, 1) * 30

        # Consecutive beats component (0-30 points)
        consecutive_score = min(abs(consecutive) / 4, 1) * 30

        return min(100, sue_score + revision_score + consecutive_score)

    def run_backtest(
        self,
        initial_capital: float = 100000.0,
        position_size_pct: float = 0.05,
        holding_period: int = 60,
        min_sue: float = 1.0,
        commission_pct: float = 0.001
    ) -> Dict:
        """
        Backtest earnings momentum strategy.

        Args:
            initial_capital: Starting capital
            position_size_pct: Position size as percentage
            holding_period: Days to hold after signal
            min_sue: Minimum SUE for signals
            commission_pct: Commission percentage

        Returns:
            Dictionary with backtest results
        """
        signals = self.generate_signals(min_sue=min_sue)

        if len(signals) == 0:
            return {
                'error': 'No signals generated',
                'n_signals': 0
            }

        # Sort signals by date
        signals = sorted(signals, key=lambda x: x.date)

        capital = initial_capital
        positions = {}  # {symbol: {'shares': n, 'entry_price': p, 'entry_date': d}}
        trades = []

        # Get all dates from price data
        all_dates = self.price_data.index if isinstance(
            self.price_data.index, pd.DatetimeIndex
        ) else pd.DatetimeIndex(self.price_data.index)

        # Create signal lookup
        signal_dict = {}
        for s in signals:
            if s.date not in signal_dict:
                signal_dict[s.date] = []
            signal_dict[s.date].append(s)

        for date in all_dates:
            # Check for exits
            symbols_to_exit = []
            for symbol, pos in positions.items():
                days_held = (date - pos['entry_date']).days
                if days_held >= holding_period:
                    symbols_to_exit.append(symbol)

            for symbol in symbols_to_exit:
                pos = positions[symbol]
                try:
                    exit_price = self._get_price(symbol, date)
                    if exit_price is not None:
                        pnl = pos['shares'] * (exit_price - pos['entry_price'])
                        if pos['direction'] == 'short':
                            pnl = -pnl

                        commission = abs(pos['shares']) * exit_price * commission_pct
                        capital += pnl - commission

                        trades.append({
                            'symbol': symbol,
                            'entry_date': pos['entry_date'],
                            'exit_date': date,
                            'direction': pos['direction'],
                            'entry_price': pos['entry_price'],
                            'exit_price': exit_price,
                            'pnl': pnl - commission
                        })

                except Exception:
                    pass

                del positions[symbol]

            # Check for new signals
            if date in signal_dict:
                for signal in signal_dict[date]:
                    if signal.symbol in positions:
                        continue  # Already have position

                    try:
                        entry_price = self._get_price(signal.symbol, date)
                        if entry_price is None:
                            continue

                        position_value = capital * position_size_pct
                        shares = int(position_value / entry_price)

                        if shares > 0:
                            commission = shares * entry_price * commission_pct
                            capital -= commission

                            positions[signal.symbol] = {
                                'shares': shares,
                                'entry_price': entry_price,
                                'entry_date': date,
                                'direction': signal.signal_type
                            }

                    except Exception:
                        pass

        # Close remaining positions at end
        final_date = all_dates[-1]
        for symbol, pos in positions.items():
            try:
                exit_price = self._get_price(symbol, final_date)
                if exit_price is not None:
                    pnl = pos['shares'] * (exit_price - pos['entry_price'])
                    if pos['direction'] == 'short':
                        pnl = -pnl
                    capital += pnl
            except Exception:
                pass

        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital

        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            win_rate = (trades_df['pnl'] > 0).mean()
            avg_pnl = trades_df['pnl'].mean()
        else:
            win_rate = 0
            avg_pnl = 0

        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return_pct': total_return * 100,
            'n_signals': len(signals),
            'n_trades': len(trades),
            'win_rate': win_rate,
            'avg_trade_pnl': avg_pnl,
            'trades': trades
        }

    def _get_price(self, symbol: str, date: pd.Timestamp) -> Optional[float]:
        """Get stock price for symbol on date."""
        try:
            if isinstance(self.price_data.columns, pd.MultiIndex):
                return self.price_data.loc[date, (symbol, 'Close')]
            else:
                return self.price_data.loc[date, symbol]
        except Exception:
            return None
