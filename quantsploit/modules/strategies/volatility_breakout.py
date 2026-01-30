"""
Volatility Breakout Strategy for Quantsploit

This module implements volatility breakout trading strategies that capitalize
on price movements following periods of low volatility. Based on the concept
that volatility tends to cluster and expand after contraction.

Key Features:
- Bollinger Band squeeze detection
- Keltner Channel breakout signals
- ATR-based volatility expansion
- Volume confirmation filters
- Integration with backtesting framework

References:
    - Bollinger, J. (2001). "Bollinger on Bollinger Bands"
    - Keltner, C. (1960). "How to Make Money in Commodities"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BreakoutSignal:
    """
    Volatility breakout signal.

    Attributes:
        date: Signal date
        direction: 'bullish' or 'bearish'
        strength: Signal strength (0-100)
        squeeze_duration: Days in squeeze before breakout
        atr_expansion: ATR expansion ratio
        volume_confirmation: Whether volume confirms breakout
    """
    date: pd.Timestamp
    direction: str
    strength: float
    squeeze_duration: int
    atr_expansion: float
    volume_confirmation: bool


class VolatilityBreakoutStrategy:
    """
    Volatility Breakout Trading Strategy.

    Identifies and trades breakouts from periods of low volatility.
    Uses Bollinger Band squeeze (BB inside Keltner Channel) as the
    primary indicator of volatility contraction.

    ★ Insight ─────────────────────────────────────
    The Squeeze:
    - When BBands contract inside Keltner Channel = low volatility
    - Volatility is mean-reverting; expansion follows contraction
    - Direction of breakout often predicted by momentum
    - Volume confirms conviction of the move
    ─────────────────────────────────────────────────

    Example:
        >>> strategy = VolatilityBreakoutStrategy(price_df)
        >>> signals = strategy.generate_signals()
        >>> for signal in signals:
        ...     print(f"{signal.date}: {signal.direction} ({signal.strength})")

    Attributes:
        data: OHLCV DataFrame
        bb_period: Bollinger Band period
        bb_std: Bollinger Band standard deviations
        kc_period: Keltner Channel period
        kc_mult: Keltner Channel ATR multiplier
    """

    def __init__(
        self,
        data: pd.DataFrame,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_mult: float = 1.5,
        atr_period: int = 14,
        volume_ma_period: int = 20
    ):
        """
        Initialize Volatility Breakout Strategy.

        Args:
            data: OHLCV DataFrame with DatetimeIndex
            bb_period: Bollinger Band lookback period
            bb_std: Bollinger Band standard deviations
            kc_period: Keltner Channel period
            kc_mult: Keltner Channel ATR multiplier
            atr_period: ATR period for volatility measurement
            volume_ma_period: Volume moving average period
        """
        self.data = data.copy()
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_mult = kc_mult
        self.atr_period = atr_period
        self.volume_ma_period = volume_ma_period

        self._calculate_indicators()

    def _calculate_indicators(self) -> None:
        """Calculate all technical indicators needed for the strategy."""
        df = self.data

        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(self.bb_period).mean()
        df['bb_std'] = df['Close'].rolling(self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + self.bb_std * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - self.bb_std * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # True Range and ATR
        df['tr'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['atr'] = df['tr'].ewm(span=self.atr_period, adjust=False).mean()

        # Keltner Channel
        df['kc_middle'] = df['Close'].ewm(span=self.kc_period, adjust=False).mean()
        df['kc_upper'] = df['kc_middle'] + self.kc_mult * df['atr']
        df['kc_lower'] = df['kc_middle'] - self.kc_mult * df['atr']

        # Squeeze Detection (BB inside KC)
        df['squeeze'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])

        # Momentum (using momentum oscillator)
        df['momentum'] = df['Close'] - df['Close'].shift(self.bb_period)
        df['momentum_direction'] = np.where(df['momentum'] > 0, 1, -1)

        # Volume
        if 'Volume' in df.columns:
            df['volume_ma'] = df['Volume'].rolling(self.volume_ma_period).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma']
        else:
            df['volume_ratio'] = 1.0

        # ATR expansion ratio
        df['atr_ma'] = df['atr'].rolling(self.bb_period).mean()
        df['atr_ratio'] = df['atr'] / df['atr_ma']

        self.data = df

    def detect_squeeze(self) -> pd.Series:
        """
        Detect volatility squeeze conditions.

        Returns:
            Boolean Series indicating squeeze periods
        """
        return self.data['squeeze']

    def detect_squeeze_release(self) -> pd.Series:
        """
        Detect squeeze release (breakout from squeeze).

        Returns:
            Boolean Series indicating squeeze release points
        """
        squeeze = self.data['squeeze']
        squeeze_release = squeeze.shift(1) & ~squeeze
        return squeeze_release.fillna(False)

    def get_squeeze_duration(self) -> pd.Series:
        """
        Calculate the duration of each squeeze period.

        Returns:
            Series with squeeze duration in bars for each squeeze release
        """
        squeeze = self.data['squeeze'].astype(int)

        # Count consecutive squeeze bars
        # Use cumsum trick: reset counter when squeeze ends
        squeeze_groups = (~squeeze.astype(bool)).cumsum()
        durations = squeeze.groupby(squeeze_groups).cumsum()

        return durations

    def generate_signals(
        self,
        min_squeeze_duration: int = 5,
        min_atr_expansion: float = 1.2,
        require_volume_confirmation: bool = True,
        volume_threshold: float = 1.5
    ) -> List[BreakoutSignal]:
        """
        Generate volatility breakout signals.

        Args:
            min_squeeze_duration: Minimum bars in squeeze before valid breakout
            min_atr_expansion: Minimum ATR expansion ratio for confirmation
            require_volume_confirmation: Require volume spike on breakout
            volume_threshold: Volume ratio threshold for confirmation

        Returns:
            List of BreakoutSignal objects
        """
        df = self.data
        signals = []

        squeeze_release = self.detect_squeeze_release()
        squeeze_duration = self.get_squeeze_duration()

        for i in range(len(df)):
            if not squeeze_release.iloc[i]:
                continue

            date = df.index[i]

            # Check squeeze duration
            duration = squeeze_duration.iloc[i-1] if i > 0 else 0
            if duration < min_squeeze_duration:
                continue

            # Determine direction from momentum
            momentum_dir = df['momentum_direction'].iloc[i]
            direction = 'bullish' if momentum_dir > 0 else 'bearish'

            # Check ATR expansion
            atr_expansion = df['atr_ratio'].iloc[i]
            if atr_expansion < min_atr_expansion:
                continue

            # Check volume confirmation
            volume_ratio = df['volume_ratio'].iloc[i]
            volume_confirmed = volume_ratio >= volume_threshold

            if require_volume_confirmation and not volume_confirmed:
                continue

            # Calculate signal strength
            strength = self._calculate_signal_strength(
                squeeze_duration=duration,
                atr_expansion=atr_expansion,
                volume_ratio=volume_ratio
            )

            signals.append(BreakoutSignal(
                date=date,
                direction=direction,
                strength=strength,
                squeeze_duration=int(duration),
                atr_expansion=atr_expansion,
                volume_confirmation=volume_confirmed
            ))

        return signals

    def _calculate_signal_strength(
        self,
        squeeze_duration: int,
        atr_expansion: float,
        volume_ratio: float
    ) -> float:
        """Calculate signal strength from 0-100."""
        # Duration component (longer squeeze = stronger breakout potential)
        duration_score = min(squeeze_duration / 20, 1) * 30  # Max 30 points

        # ATR expansion component
        expansion_score = min((atr_expansion - 1) / 0.5, 1) * 40  # Max 40 points

        # Volume component
        volume_score = min((volume_ratio - 1) / 1, 1) * 30  # Max 30 points

        return min(100, duration_score + expansion_score + volume_score)

    def get_current_state(self) -> Dict:
        """
        Get current market state for the strategy.

        Returns:
            Dictionary with current indicator values and state
        """
        latest = self.data.iloc[-1]

        return {
            'in_squeeze': bool(latest['squeeze']),
            'bb_width': latest['bb_width'],
            'atr': latest['atr'],
            'atr_ratio': latest['atr_ratio'],
            'momentum': latest['momentum'],
            'momentum_direction': 'bullish' if latest['momentum_direction'] > 0 else 'bearish',
            'volume_ratio': latest['volume_ratio'],
            'bb_upper': latest['bb_upper'],
            'bb_lower': latest['bb_lower'],
            'kc_upper': latest['kc_upper'],
            'kc_lower': latest['kc_lower']
        }

    def run_backtest(
        self,
        initial_capital: float = 100000.0,
        position_size_pct: float = 0.1,
        stop_loss_atr: float = 2.0,
        take_profit_atr: float = 3.0,
        commission_pct: float = 0.001,
        min_squeeze_duration: int = 5,
        min_atr_expansion: float = 1.2
    ) -> Dict:
        """
        Backtest the volatility breakout strategy.

        Args:
            initial_capital: Starting capital
            position_size_pct: Position size as percentage of capital
            stop_loss_atr: Stop loss in ATR multiples
            take_profit_atr: Take profit in ATR multiples
            commission_pct: Commission percentage
            min_squeeze_duration: Minimum squeeze duration for valid signal
            min_atr_expansion: Minimum ATR expansion ratio

        Returns:
            Dictionary with backtest results
        """
        df = self.data.copy()
        signals = self.generate_signals(
            min_squeeze_duration=min_squeeze_duration,
            min_atr_expansion=min_atr_expansion,
            require_volume_confirmation=False
        )

        # Convert signals to dict for faster lookup
        signal_dict = {s.date: s for s in signals}

        capital = initial_capital
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0

        trades = []
        equity_curve = []

        for i in range(len(df)):
            date = df.index[i]
            row = df.iloc[i]

            # Check for exit if in position
            if position != 0:
                exit_triggered = False
                exit_price = 0
                exit_reason = ''

                if position > 0:  # Long position
                    if row['Low'] <= stop_loss:
                        exit_triggered = True
                        exit_price = stop_loss
                        exit_reason = 'stop_loss'
                    elif row['High'] >= take_profit:
                        exit_triggered = True
                        exit_price = take_profit
                        exit_reason = 'take_profit'
                else:  # Short position
                    if row['High'] >= stop_loss:
                        exit_triggered = True
                        exit_price = stop_loss
                        exit_reason = 'stop_loss'
                    elif row['Low'] <= take_profit:
                        exit_triggered = True
                        exit_price = take_profit
                        exit_reason = 'take_profit'

                if exit_triggered:
                    # Calculate P&L
                    if position > 0:
                        pnl = (exit_price - entry_price) * position
                    else:
                        pnl = (entry_price - exit_price) * abs(position)

                    commission = abs(position) * exit_price * commission_pct
                    net_pnl = pnl - commission

                    capital += net_pnl

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'direction': 'long' if position > 0 else 'short',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl': net_pnl,
                        'return_pct': net_pnl / (entry_price * abs(position)) * 100
                    })

                    position = 0

            # Check for entry signal if not in position
            if position == 0 and date in signal_dict:
                signal = signal_dict[date]

                # Calculate position size
                position_value = capital * position_size_pct
                shares = int(position_value / row['Close'])

                if shares > 0:
                    entry_price = row['Close']
                    entry_date = date
                    atr = row['atr']

                    if signal.direction == 'bullish':
                        position = shares
                        stop_loss = entry_price - stop_loss_atr * atr
                        take_profit = entry_price + take_profit_atr * atr
                    else:
                        position = -shares
                        stop_loss = entry_price + stop_loss_atr * atr
                        take_profit = entry_price - take_profit_atr * atr

                    commission = shares * entry_price * commission_pct
                    capital -= commission

            # Track equity
            if position != 0:
                unrealized = position * (row['Close'] - entry_price)
                equity_curve.append(capital + unrealized)
            else:
                equity_curve.append(capital)

        # Calculate metrics
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            win_rate = (trades_df['pnl'] > 0).mean()
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if any(trades_df['pnl'] > 0) else 0
            avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if any(trades_df['pnl'] < 0) else 0
            profit_factor = avg_win * win_rate / (avg_loss * (1 - win_rate)) if avg_loss > 0 and win_rate < 1 else 0
        else:
            win_rate = 0
            profit_factor = 0

        equity_curve = np.array(equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        cumulative_return = (capital - initial_capital) / initial_capital

        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0

        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return_pct': cumulative_return * 100,
            'annualized_return': (1 + cumulative_return) ** (252 / len(df)) - 1 if len(df) > 0 else 0,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'n_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trades': trades,
            'equity_curve': equity_curve,
            'signals': signals
        }
