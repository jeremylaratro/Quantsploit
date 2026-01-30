"""
Mean Reversion Strategy with Z-Score Analysis

This module provides both analysis and backtesting capabilities for
mean reversion trading based on statistical indicators.
"""

import pandas as pd
import numpy as np
from quantsploit.utils.ta_compat import ta
from typing import Dict, Any, Optional, Callable
from scipy import stats
from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher


class MeanReversion(BaseModule):
    """
    Advanced mean reversion strategy using z-score, Bollinger Bands,
    and statistical analysis to identify overbought/oversold conditions
    """

    @property
    def name(self) -> str:
        return "Mean Reversion Strategy"

    @property
    def description(self) -> str:
        return "Statistical mean reversion using Z-score, Bollinger Bands, and RSI"

    def trading_guide(self) -> str:
        return """SYNOPSIS: Z-score measures price deviation from 20-day mean. |Z| >2.0 signals
extreme, likely to revert. Uses Bollinger Bands, RSI for confirmation.

SIMULATION POSITIONS:
  - Analysis only (no backtested trades)
  - Provides signal strength score and reversion probability
  - Calculates expected return to mean

RECOMMENDED ENTRY:
  - LONG: Z-score < -2.0 (oversold) + RSI <30 + below lower BB
  - SHORT/EXIT: Z-score > +2.0 (overbought) + RSI >70 + above upper BB
  - Take profit when Z-score returns to ±0.5 (near mean)
  - Stop loss if Z-score exceeds ±3.0 (extreme extension)

POSITION SIZING:
  - Full position: Z-score < -2.5 (very oversold)
  - Half position: Z-score -1.5 to -2.0 (moderately oversold)
  - Increase size when reversion probability >70%

BEST USE: Range-bound stocks, avoid in strong trends. Works well on ETFs,
stable large-caps. Check if historical mean reversion probability >60%."""

    def show_info(self):
        info = super().show_info()
        info['trading_guide'] = self.trading_guide()
        return info

    @property
    def author(self) -> str:
        return "Quantsploit Team"

    @property
    def category(self) -> str:
        return "strategy"

    def _init_options(self):
        super()._init_options()
        self.options.update({
            "LOOKBACK": {
                "value": 20,
                "required": False,
                "description": "Lookback period for mean calculation"
            },
            "Z_THRESHOLD": {
                "value": 2.0,
                "required": False,
                "description": "Z-score threshold for signals"
            },
            "BB_PERIOD": {
                "value": 20,
                "required": False,
                "description": "Bollinger Bands period"
            },
            "BB_STD": {
                "value": 2.0,
                "required": False,
                "description": "Bollinger Bands standard deviation"
            }
        })

    def run(self) -> Dict[str, Any]:
        """Execute mean reversion analysis"""
        symbol = self.get_option("SYMBOL")
        period = self.get_option("PERIOD")
        interval = self.get_option("INTERVAL")
        lookback = int(self.get_option("LOOKBACK"))
        z_threshold = float(self.get_option("Z_THRESHOLD"))
        bb_period = int(self.get_option("BB_PERIOD"))
        bb_std = float(self.get_option("BB_STD"))

        # Fetch data
        fetcher = DataFetcher(self.framework.database)
        df = fetcher.get_stock_data(symbol, period, interval)

        if df is None or df.empty:
            return {"success": False, "error": f"Failed to fetch data for {symbol}"}

        close = df['Close']

        # Calculate rolling statistics
        rolling_mean = close.rolling(window=lookback).mean()
        rolling_std = close.rolling(window=lookback).std()

        # Z-score calculation
        df['z_score'] = (close - rolling_mean) / rolling_std

        # Bollinger Bands
        bbands = ta.bbands(close, length=bb_period, std=bb_std)
        if bbands is not None:
            df = pd.concat([df, bbands], axis=1)

        # Calculate percentile rank (where price sits in range)
        def percentile_rank(series, window):
            return series.rolling(window).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
            )

        df['percentile_rank'] = percentile_rank(close, lookback)

        # RSI for confirmation
        df['rsi'] = ta.rsi(close, length=14)

        # Current values
        current_price = close.iloc[-1]
        current_z = df['z_score'].iloc[-1]
        current_percentile = df['percentile_rank'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]

        # Generate signals
        signals = []
        signal_strength = 0

        # Z-score based signals
        if current_z < -z_threshold:
            signals.append(f"🟢 OVERSOLD: Z-score = {current_z:.2f} (< -{z_threshold})")
            signal_strength += 30
        elif current_z < -1.5:
            signals.append(f"🟡 Approaching oversold: Z-score = {current_z:.2f}")
            signal_strength += 15

        if current_z > z_threshold:
            signals.append(f"🔴 OVERBOUGHT: Z-score = {current_z:.2f} (> {z_threshold})")
            signal_strength -= 30
        elif current_z > 1.5:
            signals.append(f"🟡 Approaching overbought: Z-score = {current_z:.2f}")
            signal_strength -= 15

        # Bollinger Bands signals
        if bbands is not None:
            bb_upper = df[f'BBU_{bb_period}_{bb_std}'].iloc[-1]
            bb_lower = df[f'BBL_{bb_period}_{bb_std}'].iloc[-1]
            bb_middle = df[f'BBM_{bb_period}_{bb_std}'].iloc[-1]

            if current_price < bb_lower:
                signals.append(f"🟢 Below lower BB: Price ${current_price:.2f} < ${bb_lower:.2f}")
                signal_strength += 25
            elif current_price > bb_upper:
                signals.append(f"🔴 Above upper BB: Price ${current_price:.2f} > ${bb_upper:.2f}")
                signal_strength -= 25

        # Percentile rank signals
        if current_percentile < 0.2:
            signals.append(f"🟢 In bottom 20% of range (percentile: {current_percentile:.2%})")
            signal_strength += 20
        elif current_percentile > 0.8:
            signals.append(f"🔴 In top 20% of range (percentile: {current_percentile:.2%})")
            signal_strength -= 20

        # RSI confirmation
        if current_rsi < 30:
            signals.append(f"🟢 RSI oversold: {current_rsi:.1f}")
            signal_strength += 15
        elif current_rsi > 70:
            signals.append(f"🔴 RSI overbought: {current_rsi:.1f}")
            signal_strength -= 15

        # Mean reversion probability
        mean_reversion_prob = self._calculate_reversion_probability(df, lookback)

        # Overall signal
        if signal_strength > 50:
            overall_signal = "🟢 STRONG BUY - High probability mean reversion opportunity"
        elif signal_strength > 25:
            overall_signal = "🟢 BUY - Oversold conditions detected"
        elif signal_strength < -50:
            overall_signal = "🔴 STRONG SELL - Severely overbought"
        elif signal_strength < -25:
            overall_signal = "🔴 SELL - Overbought conditions detected"
        else:
            overall_signal = "⚪ NEUTRAL - No clear mean reversion signal"

        # Calculate expected return to mean
        expected_return = ((rolling_mean.iloc[-1] - current_price) / current_price) * 100

        results = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "rolling_mean": round(rolling_mean.iloc[-1], 2),
            "z_score": round(current_z, 2),
            "percentile_rank": round(current_percentile, 2),
            "rsi": round(current_rsi, 2),
            "signal_strength": signal_strength,
            "overall_signal": overall_signal,
            "expected_return_to_mean": f"{expected_return:+.2f}%",
            "mean_reversion_probability": f"{mean_reversion_prob:.1%}",
            "signals": signals,
            "statistics": {
                "mean": round(rolling_mean.iloc[-1], 2),
                "std_dev": round(rolling_std.iloc[-1], 2),
                "min_20d": round(close.tail(lookback).min(), 2),
                "max_20d": round(close.tail(lookback).max(), 2),
            },
            "recent_data": df[['Close', 'z_score', 'percentile_rank', 'rsi']].tail(10)
        }

        # Add Bollinger Bands data if available
        if bbands is not None:
            results["bollinger_bands"] = {
                "upper": round(bb_upper, 2),
                "middle": round(bb_middle, 2),
                "lower": round(bb_lower, 2),
                "width": round((bb_upper - bb_lower) / bb_middle * 100, 2)
            }

        return results

    def _calculate_reversion_probability(self, df: pd.DataFrame, window: int) -> float:
        """
        Calculate probability of mean reversion based on historical behavior
        """
        z_scores = df['z_score'].dropna()

        if len(z_scores) < window:
            return 0.5

        # Count how many times extreme z-scores reverted
        reversions = 0
        opportunities = 0

        for i in range(len(z_scores) - 5):
            if abs(z_scores.iloc[i]) > 2.0:  # Extreme value
                opportunities += 1
                # Check if it reverted in next 5 periods
                future = z_scores.iloc[i+1:i+6]
                if len(future) > 0:
                    if z_scores.iloc[i] > 0 and any(future < 1.0):
                        reversions += 1
                    elif z_scores.iloc[i] < 0 and any(future > -1.0):
                        reversions += 1

        return reversions / opportunities if opportunities > 0 else 0.5

    # =========================================================================
    # Backtesting Integration
    # =========================================================================

    def run_backtest(
        self,
        initial_capital: float = 100000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a full backtest of the mean reversion strategy.

        This method fetches data, calculates indicators, and runs the strategy
        through the backtesting engine to produce performance metrics.

        Args:
            initial_capital: Starting capital for backtest
            commission_pct: Commission as percentage of trade value
            slippage_pct: Slippage as percentage of trade price
            **kwargs: Additional arguments passed to BacktestConfig

        Returns:
            Dictionary containing:
            - backtest_results: Full BacktestResults object
            - summary: Performance summary
            - trades: List of executed trades
            - analysis: Original analysis results

        Example:
            >>> module.set_option('SYMBOL', 'AAPL')
            >>> results = module.run_backtest(initial_capital=50000)
            >>> print(f"Total Return: {results['summary']['total_return_pct']:.2f}%")
        """
        from quantsploit.utils.backtesting import (
            Backtester, BacktestConfig, PositionSide
        )

        symbol = self.get_option("SYMBOL")
        period = self.get_option("PERIOD")
        interval = self.get_option("INTERVAL")
        lookback = int(self.get_option("LOOKBACK"))
        z_threshold = float(self.get_option("Z_THRESHOLD"))
        bb_period = int(self.get_option("BB_PERIOD"))
        bb_std = float(self.get_option("BB_STD"))

        # Fetch data
        fetcher = DataFetcher(self.framework.database)
        df = fetcher.get_stock_data(symbol, period, interval)

        if df is None or df.empty:
            return {"success": False, "error": f"Failed to fetch data for {symbol}"}

        # Pre-calculate all indicators
        df = self._prepare_indicators(df, lookback, bb_period, bb_std)

        # Skip warmup period
        warmup = max(lookback, bb_period, 14) + 5  # Extra buffer for stability

        if len(df) < warmup + 50:
            return {
                "success": False,
                "error": f"Insufficient data for backtest. Need {warmup + 50}, got {len(df)}"
            }

        # Create backtest configuration
        config = BacktestConfig(
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            position_size=0.95,  # Use 95% of capital
            max_positions=1,     # Single position at a time
            **kwargs
        )

        # Create backtester
        backtester = Backtester(config)

        # Create strategy function with closure over parameters
        def strategy_func(bt, date, row):
            self._backtest_strategy_logic(
                bt, date, row, df, symbol,
                z_threshold=z_threshold,
                lookback=lookback
            )

        # Run backtest (skip warmup period)
        backtest_data = df.iloc[warmup:]
        results = backtester.run_backtest(backtest_data, strategy_func, symbol=symbol)

        # Run analysis for comparison
        analysis = self.run()

        return {
            "success": True,
            "backtest_results": results,
            "summary": {
                "total_return_pct": results.total_return_pct,
                "sharpe_ratio": results.sharpe_ratio,
                "sortino_ratio": results.sortino_ratio,
                "max_drawdown": results.max_drawdown,
                "total_trades": results.total_trades,
                "win_rate": results.win_rate,
                "profit_factor": results.profit_factor,
                "avg_trade_pnl": results.avg_trade_pnl,
                "volatility": results.volatility,
            },
            "trades": [t.__dict__ for t in results.closed_trades],
            "analysis": analysis,
            "equity_curve": results.equity_curve
        }

    def _prepare_indicators(
        self,
        df: pd.DataFrame,
        lookback: int,
        bb_period: int,
        bb_std: float
    ) -> pd.DataFrame:
        """
        Pre-calculate all technical indicators needed for the strategy.

        Args:
            df: Price DataFrame
            lookback: Lookback period for z-score
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation

        Returns:
            DataFrame with calculated indicators
        """
        df = df.copy()
        close = df['Close']

        # Z-score calculation
        rolling_mean = close.rolling(window=lookback).mean()
        rolling_std = close.rolling(window=lookback).std()
        df['z_score'] = (close - rolling_mean) / rolling_std
        df['rolling_mean'] = rolling_mean
        df['rolling_std'] = rolling_std

        # Bollinger Bands
        bbands = ta.bbands(close, length=bb_period, std=bb_std)
        if bbands is not None:
            df = pd.concat([df, bbands], axis=1)

        # RSI for confirmation
        df['rsi'] = ta.rsi(close, length=14)

        # Percentile rank
        df['percentile_rank'] = close.rolling(lookback).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 0 else 0.5
        )

        return df

    def _backtest_strategy_logic(
        self,
        bt,
        date,
        row,
        full_data: pd.DataFrame,
        symbol: str,
        z_threshold: float = 2.0,
        lookback: int = 20
    ):
        """
        Execute strategy logic for a single bar during backtesting.

        Entry Conditions (LONG):
        - Z-score < -z_threshold (oversold)
        - RSI < 35 (confirmation)

        Exit Conditions:
        - Z-score crosses back toward 0 (mean reversion)
        - OR Z-score exceeds +z_threshold (overbought - stop and reverse)
        - OR stop loss at Z-score < -3.0 (extreme extension)

        Args:
            bt: Backtester instance
            date: Current bar date
            row: Current bar data
            full_data: Full DataFrame with indicators
            symbol: Trading symbol
            z_threshold: Z-score threshold for signals
            lookback: Lookback period
        """
        from quantsploit.utils.backtesting import PositionSide

        # Get current indicator values from row (already in DataFrame)
        current_z = row.get('z_score', np.nan)
        current_rsi = row.get('rsi', np.nan)
        current_price = row['Close']

        # Skip if indicators not ready
        if pd.isna(current_z) or pd.isna(current_rsi):
            return

        # Get current position
        position = bt.get_position(symbol)

        # =====================================================================
        # EXIT LOGIC (check first)
        # =====================================================================
        if position is not None:
            # Calculate profit target and stop loss based on entry
            entry_z = getattr(position, '_entry_z', current_z)

            # Exit conditions for LONG position
            if position.side == PositionSide.LONG:
                # Take profit: Z-score reverts to near mean
                if current_z > -0.5:
                    bt.close_position(symbol, current_price, reason="Mean Reversion Target")
                    return

                # Stop loss: Extreme extension (Z < -3)
                if current_z < -3.0:
                    bt.close_position(symbol, current_price, reason="Stop Loss - Extreme Extension")
                    return

                # Cut loss if RSI indicates momentum reversal
                if current_rsi > 50 and current_z > entry_z + 0.5:
                    bt.close_position(symbol, current_price, reason="Momentum Shift Exit")
                    return

            # Exit conditions for SHORT position
            elif position.side == PositionSide.SHORT:
                # Take profit: Z-score reverts to near mean
                if current_z < 0.5:
                    bt.close_position(symbol, current_price, reason="Mean Reversion Target")
                    return

                # Stop loss: Extreme extension (Z > +3)
                if current_z > 3.0:
                    bt.close_position(symbol, current_price, reason="Stop Loss - Extreme Extension")
                    return

        # =====================================================================
        # ENTRY LOGIC
        # =====================================================================
        if position is None:
            # Calculate signal strength
            signal_strength = 0

            # Z-score signal
            if current_z < -z_threshold:
                signal_strength += 40
            elif current_z > z_threshold:
                signal_strength -= 40

            # RSI confirmation
            if current_rsi < 35:
                signal_strength += 20
            elif current_rsi > 65:
                signal_strength -= 20

            # Bollinger Band position
            bb_lower_col = f'BBL_{lookback}_2.0'
            bb_upper_col = f'BBU_{lookback}_2.0'

            if bb_lower_col in row.index and not pd.isna(row.get(bb_lower_col)):
                if current_price < row[bb_lower_col]:
                    signal_strength += 15
                elif current_price > row[bb_upper_col]:
                    signal_strength -= 15

            # Entry signals
            if signal_strength >= 50:
                # Strong oversold signal -> LONG
                new_pos = bt.open_position(
                    symbol=symbol,
                    price=current_price,
                    side=PositionSide.LONG,
                    reason=f"Mean Reversion LONG: Z={current_z:.2f}, RSI={current_rsi:.1f}"
                )
                if new_pos:
                    new_pos._entry_z = current_z  # Store for exit logic

            elif signal_strength <= -50:
                # Strong overbought signal -> SHORT
                new_pos = bt.open_position(
                    symbol=symbol,
                    price=current_price,
                    side=PositionSide.SHORT,
                    reason=f"Mean Reversion SHORT: Z={current_z:.2f}, RSI={current_rsi:.1f}"
                )
                if new_pos:
                    new_pos._entry_z = current_z

    def create_strategy_function(
        self,
        z_threshold: float = 2.0,
        lookback: int = 20
    ) -> Callable:
        """
        Create a standalone strategy function for use with external backtesting.

        This method creates a strategy function that can be used directly with
        the Backtester class or walk-forward analysis without needing the
        full module context.

        Args:
            z_threshold: Z-score threshold for signals
            lookback: Lookback period for calculations

        Returns:
            Strategy function with signature (backtester, date, row, symbol, data)

        Example:
            >>> strategy_func = module.create_strategy_function(z_threshold=2.5)
            >>> backtester.run_backtest(data, strategy_func, symbol='AAPL')
        """
        from quantsploit.utils.backtesting import PositionSide

        def strategy_func(bt, date, row, symbol: str, full_data: pd.DataFrame):
            """Mean reversion strategy function for backtesting."""
            # Calculate indicators if not present
            if 'z_score' not in full_data.columns:
                close = full_data['Close']
                rolling_mean = close.rolling(window=lookback).mean()
                rolling_std = close.rolling(window=lookback).std()
                full_data['z_score'] = (close - rolling_mean) / rolling_std
                full_data['rsi'] = ta.rsi(close, length=14)

            # Get current values
            try:
                current_idx = full_data.index.get_loc(date)
                current_z = full_data['z_score'].iloc[current_idx]
                current_rsi = full_data['rsi'].iloc[current_idx]
            except (KeyError, IndexError):
                return

            if pd.isna(current_z) or pd.isna(current_rsi):
                return

            current_price = row['Close']
            position = bt.get_position(symbol)

            # Exit logic
            if position is not None:
                if position.side == PositionSide.LONG:
                    if current_z > -0.5 or current_z < -3.0:
                        bt.close_position(symbol, current_price)
                        return
                elif position.side == PositionSide.SHORT:
                    if current_z < 0.5 or current_z > 3.0:
                        bt.close_position(symbol, current_price)
                        return

            # Entry logic
            if position is None:
                if current_z < -z_threshold and current_rsi < 35:
                    bt.open_position(symbol, current_price, PositionSide.LONG)
                elif current_z > z_threshold and current_rsi > 65:
                    bt.open_position(symbol, current_price, PositionSide.SHORT)

        return strategy_func
