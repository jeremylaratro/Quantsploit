"""
Advanced Momentum and Trend Following Signals

This module provides both analysis and backtesting capabilities for
momentum-based trading strategies using multi-factor scoring.
"""

import pandas as pd
import numpy as np
from quantsploit.utils.ta_compat import ta
from typing import Dict, Any, Optional, Callable
from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher


class MomentumSignals(BaseModule):
    """
    Advanced momentum strategy combining multiple indicators:
    - Rate of Change (ROC)
    - Relative Strength
    - ADX (trend strength)
    - Volume-weighted momentum
    - Acceleration indicators
    """

    @property
    def name(self) -> str:
        return "Momentum Signals"

    @property
    def description(self) -> str:
        return "Multi-factor momentum scoring using ROC, ADX, volume, MA alignment"

    def trading_guide(self) -> str:
        return """SYNOPSIS: Combines ROC, ADX, volume momentum, MA alignment, and relative strength
into composite score. Score >60 = BUY, <-60 = SELL.

SIMULATION POSITIONS:
  - Analysis only (no backtested positions)
  - Generates signal scores from -100 to +100
  - Provides STRONG BUY/BUY/NEUTRAL/SELL/STRONG SELL recommendation

RECOMMENDED ENTRY:
  - STRONG BUY (score >60): Enter full position, strong uptrend confirmed
  - BUY (score >30): Enter partial position, positive momentum detected
  - SELL signals: Exit or avoid, downtrend in progress
  - Confirm with ADX >25 for strong trend before entering

KEY SIGNALS:
  - 12-period ROC >10%: Strong momentum (+25 points)
  - Bullish MA alignment (10>20>50): Trend confirmed (+20)
  - Positive momentum acceleration (+20)
  - Trading above VWAP (+10)
  - Outperforming benchmark (+15)

BEST USE: Screening tool for trending stocks before entering swing positions."""

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
            "BENCHMARK": {
                "value": "SPY",
                "required": False,
                "description": "Benchmark for relative strength (SPY, QQQ, etc.)"
            },
            "MOMENTUM_PERIOD": {
                "value": 12,
                "required": False,
                "description": "Momentum calculation period"
            },
            "MIN_ADX": {
                "value": 25,
                "required": False,
                "description": "Minimum ADX for trend confirmation"
            }
        })

    def run(self) -> Dict[str, Any]:
        """Execute momentum analysis"""
        symbol = self.get_option("SYMBOL")
        period = self.get_option("PERIOD")
        interval = self.get_option("INTERVAL")
        benchmark = self.get_option("BENCHMARK")
        momentum_period = int(self.get_option("MOMENTUM_PERIOD"))
        min_adx = float(self.get_option("MIN_ADX"))

        # Fetch data
        fetcher = DataFetcher(self.framework.database)
        df = fetcher.get_stock_data(symbol, period, interval)

        if df is None or df.empty:
            return {"success": False, "error": f"Failed to fetch data for {symbol}"}

        # Fetch benchmark data
        benchmark_df = fetcher.get_stock_data(benchmark, period, interval)

        close = df['Close']
        volume = df['Volume']

        # 1. Rate of Change (momentum)
        df['roc_12'] = ta.roc(close, length=momentum_period)
        df['roc_6'] = ta.roc(close, length=6)
        df['roc_3'] = ta.roc(close, length=3)

        # 2. ADX (trend strength)
        adx = ta.adx(df['High'], df['Low'], close, length=14)
        if adx is not None:
            df = pd.concat([df, adx], axis=1)

        # 3. Volume-weighted momentum
        df['vwap'] = ta.vwap(df['High'], df['Low'], close, volume)
        df['volume_momentum'] = (close / df['vwap'] - 1) * 100

        # 4. Moving average alignment (trend confirmation)
        df['sma_10'] = ta.sma(close, length=10)
        df['sma_20'] = ta.sma(close, length=20)
        df['sma_50'] = ta.sma(close, length=50)

        # 5. Momentum acceleration
        df['momentum_accel'] = df['roc_6'].diff()

        # 6. Relative strength vs benchmark
        if benchmark_df is not None and not benchmark_df.empty:
            # Align dates
            combined = pd.merge(
                df[['Close']], benchmark_df[['Close']],
                left_index=True, right_index=True,
                suffixes=('', '_bench'), how='inner'
            )

            # Calculate relative strength
            stock_return = (combined['Close'] / combined['Close'].iloc[0] - 1) * 100
            bench_return = (combined['Close_bench'] / combined['Close_bench'].iloc[0] - 1) * 100
            df['relative_strength'] = stock_return - bench_return
            current_rs = df['relative_strength'].iloc[-1]
        else:
            current_rs = None

        # 7. Stochastic momentum
        stoch = ta.stoch(df['High'], df['Low'], close)
        if stoch is not None:
            df = pd.concat([df, stoch], axis=1)

        # Current values
        current_price = close.iloc[-1]
        current_roc_12 = df['roc_12'].iloc[-1]
        current_roc_6 = df['roc_6'].iloc[-1]
        current_adx = df['ADX_14'].iloc[-1] if 'ADX_14' in df.columns else None
        current_volume_mom = df['volume_momentum'].iloc[-1]
        current_accel = df['momentum_accel'].iloc[-1]

        # Moving average alignment
        ma_aligned = (df['sma_10'].iloc[-1] > df['sma_20'].iloc[-1] >
                     df['sma_50'].iloc[-1])
        ma_bearish = (df['sma_10'].iloc[-1] < df['sma_20'].iloc[-1] <
                     df['sma_50'].iloc[-1])

        # Generate signals
        signals = []
        signal_score = 0

        # ROC signals
        if current_roc_12 > 10:
            signals.append(f"🟢 Strong momentum: 12-period ROC = {current_roc_12:.2f}%")
            signal_score += 25
        elif current_roc_12 > 5:
            signals.append(f"🟢 Positive momentum: 12-period ROC = {current_roc_12:.2f}%")
            signal_score += 15
        elif current_roc_12 < -10:
            signals.append(f"🔴 Strong negative momentum: ROC = {current_roc_12:.2f}%")
            signal_score -= 25
        elif current_roc_12 < -5:
            signals.append(f"🔴 Negative momentum: ROC = {current_roc_12:.2f}%")
            signal_score -= 15

        # Short-term momentum
        if current_roc_6 > 5:
            signals.append(f"🟢 Short-term momentum positive: {current_roc_6:.2f}%")
            signal_score += 15
        elif current_roc_6 < -5:
            signals.append(f"🔴 Short-term momentum negative: {current_roc_6:.2f}%")
            signal_score -= 15

        # Momentum acceleration
        if current_accel > 2:
            signals.append(f"🟢 Momentum accelerating: {current_accel:.2f}")
            signal_score += 20
        elif current_accel < -2:
            signals.append(f"🔴 Momentum decelerating: {current_accel:.2f}")
            signal_score -= 20

        # ADX trend strength
        if current_adx:
            if current_adx > min_adx:
                trend_dir = "bullish" if current_roc_12 > 0 else "bearish"
                signals.append(f"🔷 Strong trend ({trend_dir}): ADX = {current_adx:.1f}")
                signal_score += 15 if trend_dir == "bullish" else -15
            else:
                signals.append(f"⚪ Weak trend: ADX = {current_adx:.1f}")

        # Volume-weighted momentum
        if current_volume_mom > 2:
            signals.append(f"🟢 Trading above VWAP: {current_volume_mom:+.2f}%")
            signal_score += 10
        elif current_volume_mom < -2:
            signals.append(f"🔴 Trading below VWAP: {current_volume_mom:+.2f}%")
            signal_score -= 10

        # Moving average alignment
        if ma_aligned:
            signals.append("🟢 Moving averages aligned bullish (10>20>50)")
            signal_score += 20
        elif ma_bearish:
            signals.append("🔴 Moving averages aligned bearish (10<20<50)")
            signal_score -= 20

        # Relative strength
        if current_rs is not None:
            if current_rs > 5:
                signals.append(f"🟢 Outperforming {benchmark}: +{current_rs:.2f}%")
                signal_score += 15
            elif current_rs < -5:
                signals.append(f"🔴 Underperforming {benchmark}: {current_rs:.2f}%")
                signal_score -= 15

        # Overall signal
        if signal_score > 60:
            overall_signal = "🟢 STRONG BUY - Multiple momentum confirmations"
            recommendation = "Strong uptrend with momentum acceleration"
        elif signal_score > 30:
            overall_signal = "🟢 BUY - Positive momentum detected"
            recommendation = "Uptrend in progress"
        elif signal_score < -60:
            overall_signal = "🔴 STRONG SELL - Multiple negative signals"
            recommendation = "Strong downtrend, avoid or short"
        elif signal_score < -30:
            overall_signal = "🔴 SELL - Negative momentum"
            recommendation = "Downtrend in progress"
        else:
            overall_signal = "⚪ NEUTRAL - No clear momentum"
            recommendation = "Wait for clearer signals"

        # Calculate momentum rank (0-100)
        momentum_rank = max(0, min(100, 50 + signal_score))

        results = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "signal_score": signal_score,
            "momentum_rank": round(momentum_rank, 1),
            "overall_signal": overall_signal,
            "recommendation": recommendation,
            "signals": signals,
            "momentum_metrics": {
                "roc_12_period": round(current_roc_12, 2),
                "roc_6_period": round(current_roc_6, 2),
                "momentum_acceleration": round(current_accel, 2),
                "adx": round(current_adx, 2) if current_adx else None,
                "volume_momentum": round(current_volume_mom, 2),
                "relative_strength_vs_benchmark": round(current_rs, 2) if current_rs else None
            },
            "trend_analysis": {
                "ma_alignment": "bullish" if ma_aligned else "bearish" if ma_bearish else "mixed",
                "sma_10": round(df['sma_10'].iloc[-1], 2),
                "sma_20": round(df['sma_20'].iloc[-1], 2),
                "sma_50": round(df['sma_50'].iloc[-1], 2)
            },
            "recent_data": df[['Close', 'roc_12', 'volume_momentum']].tail(10)
        }

        return results

    # =========================================================================
    # Backtesting Integration
    # =========================================================================

    def run_backtest(
        self,
        initial_capital: float = 100000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        entry_threshold: int = 30,
        exit_threshold: int = -10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a full backtest of the momentum strategy.

        This method fetches data, calculates momentum indicators, and runs the
        strategy through the backtesting engine to produce performance metrics.

        Args:
            initial_capital: Starting capital for backtest
            commission_pct: Commission as percentage of trade value
            slippage_pct: Slippage as percentage of trade price
            entry_threshold: Minimum signal score to enter position (default 30)
            exit_threshold: Signal score to exit position (default -10)
            **kwargs: Additional arguments passed to BacktestConfig

        Returns:
            Dictionary containing:
            - backtest_results: Full BacktestResults object
            - summary: Performance summary
            - trades: List of executed trades
            - analysis: Original analysis results

        Example:
            >>> module.set_option('SYMBOL', 'AAPL')
            >>> results = module.run_backtest(entry_threshold=40)
            >>> print(f"Win Rate: {results['summary']['win_rate']:.1f}%")
        """
        from quantsploit.utils.backtesting import (
            Backtester, BacktestConfig, PositionSide
        )

        symbol = self.get_option("SYMBOL")
        period = self.get_option("PERIOD")
        interval = self.get_option("INTERVAL")
        momentum_period = int(self.get_option("MOMENTUM_PERIOD"))
        min_adx = float(self.get_option("MIN_ADX"))

        # Fetch data
        fetcher = DataFetcher(self.framework.database)
        df = fetcher.get_stock_data(symbol, period, interval)

        if df is None or df.empty:
            return {"success": False, "error": f"Failed to fetch data for {symbol}"}

        # Pre-calculate all indicators
        df = self._prepare_indicators(df, momentum_period, min_adx)

        # Calculate warmup period
        warmup = max(momentum_period, 50, 14) + 10

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
            position_size=0.95,
            max_positions=1,
            **kwargs
        )

        # Create backtester
        backtester = Backtester(config)

        # Create strategy function
        def strategy_func(bt, date, row):
            self._backtest_strategy_logic(
                bt, date, row, df, symbol,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                min_adx=min_adx
            )

        # Run backtest
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
        momentum_period: int,
        min_adx: float
    ) -> pd.DataFrame:
        """
        Pre-calculate all technical indicators needed for the strategy.

        Args:
            df: Price DataFrame
            momentum_period: ROC period
            min_adx: Minimum ADX threshold

        Returns:
            DataFrame with calculated indicators
        """
        df = df.copy()
        close = df['Close']
        volume = df['Volume']

        # Rate of Change (momentum)
        df['roc_12'] = ta.roc(close, length=momentum_period)
        df['roc_6'] = ta.roc(close, length=6)
        df['roc_3'] = ta.roc(close, length=3)

        # ADX (trend strength)
        adx = ta.adx(df['High'], df['Low'], close, length=14)
        if adx is not None:
            df = pd.concat([df, adx], axis=1)

        # Volume-weighted momentum
        df['vwap'] = ta.vwap(df['High'], df['Low'], close, volume)
        df['volume_momentum'] = (close / df['vwap'] - 1) * 100

        # Moving average alignment
        df['sma_10'] = ta.sma(close, length=10)
        df['sma_20'] = ta.sma(close, length=20)
        df['sma_50'] = ta.sma(close, length=50)

        # Momentum acceleration
        df['momentum_accel'] = df['roc_6'].diff()

        # Calculate signal score for each bar
        df['signal_score'] = 0

        # ROC contribution
        df.loc[df['roc_12'] > 10, 'signal_score'] += 25
        df.loc[(df['roc_12'] > 5) & (df['roc_12'] <= 10), 'signal_score'] += 15
        df.loc[df['roc_12'] < -10, 'signal_score'] -= 25
        df.loc[(df['roc_12'] < -5) & (df['roc_12'] >= -10), 'signal_score'] -= 15

        # Short-term ROC
        df.loc[df['roc_6'] > 5, 'signal_score'] += 15
        df.loc[df['roc_6'] < -5, 'signal_score'] -= 15

        # Momentum acceleration
        df.loc[df['momentum_accel'] > 2, 'signal_score'] += 20
        df.loc[df['momentum_accel'] < -2, 'signal_score'] -= 20

        # ADX contribution
        if 'ADX_14' in df.columns:
            df.loc[(df['ADX_14'] > min_adx) & (df['roc_12'] > 0), 'signal_score'] += 15
            df.loc[(df['ADX_14'] > min_adx) & (df['roc_12'] < 0), 'signal_score'] -= 15

        # Volume momentum
        df.loc[df['volume_momentum'] > 2, 'signal_score'] += 10
        df.loc[df['volume_momentum'] < -2, 'signal_score'] -= 10

        # MA alignment
        ma_bullish = (df['sma_10'] > df['sma_20']) & (df['sma_20'] > df['sma_50'])
        ma_bearish = (df['sma_10'] < df['sma_20']) & (df['sma_20'] < df['sma_50'])
        df.loc[ma_bullish, 'signal_score'] += 20
        df.loc[ma_bearish, 'signal_score'] -= 20

        return df

    def _backtest_strategy_logic(
        self,
        bt,
        date,
        row,
        full_data: pd.DataFrame,
        symbol: str,
        entry_threshold: int = 30,
        exit_threshold: int = -10,
        min_adx: float = 25
    ):
        """
        Execute strategy logic for a single bar during backtesting.

        Entry Conditions (LONG):
        - Signal score >= entry_threshold
        - ADX >= min_adx (trend confirmation)

        Entry Conditions (SHORT):
        - Signal score <= -entry_threshold
        - ADX >= min_adx (trend confirmation)

        Exit Conditions:
        - Long exits when score drops below exit_threshold
        - Short exits when score rises above -exit_threshold
        - Also exits on trend reversal (MA cross)

        Args:
            bt: Backtester instance
            date: Current bar date
            row: Current bar data
            full_data: Full DataFrame with indicators
            symbol: Trading symbol
            entry_threshold: Score threshold for entry
            exit_threshold: Score threshold for exit
            min_adx: Minimum ADX for trend confirmation
        """
        from quantsploit.utils.backtesting import PositionSide

        # Get current indicator values
        signal_score = row.get('signal_score', 0)
        current_adx = row.get('ADX_14', 0)
        current_roc_12 = row.get('roc_12', 0)
        current_price = row['Close']
        sma_10 = row.get('sma_10', current_price)
        sma_20 = row.get('sma_20', current_price)

        # Skip if indicators not ready
        if pd.isna(signal_score) or pd.isna(current_adx):
            return

        # Get current position
        position = bt.get_position(symbol)

        # =====================================================================
        # EXIT LOGIC
        # =====================================================================
        if position is not None:
            if position.side == PositionSide.LONG:
                # Exit long if signal weakens
                if signal_score < exit_threshold:
                    bt.close_position(symbol, current_price, reason=f"Signal Weakened: {signal_score}")
                    return

                # Exit on MA crossover bearish
                if sma_10 < sma_20:
                    bt.close_position(symbol, current_price, reason="MA Cross Bearish")
                    return

                # Trailing stop: exit if momentum turns negative
                if current_roc_12 < -5:
                    bt.close_position(symbol, current_price, reason="Momentum Reversal")
                    return

            elif position.side == PositionSide.SHORT:
                # Exit short if signal strengthens
                if signal_score > -exit_threshold:
                    bt.close_position(symbol, current_price, reason=f"Signal Strengthened: {signal_score}")
                    return

                # Exit on MA crossover bullish
                if sma_10 > sma_20:
                    bt.close_position(symbol, current_price, reason="MA Cross Bullish")
                    return

                # Exit if momentum turns positive
                if current_roc_12 > 5:
                    bt.close_position(symbol, current_price, reason="Momentum Reversal")
                    return

        # =====================================================================
        # ENTRY LOGIC
        # =====================================================================
        if position is None:
            # Trend confirmation via ADX
            trend_confirmed = current_adx >= min_adx

            # Long entry: Strong positive momentum with trend confirmation
            if signal_score >= entry_threshold and trend_confirmed:
                bt.open_position(
                    symbol=symbol,
                    price=current_price,
                    side=PositionSide.LONG,
                    reason=f"Momentum LONG: Score={signal_score}, ADX={current_adx:.1f}"
                )

            # Short entry: Strong negative momentum with trend confirmation
            elif signal_score <= -entry_threshold and trend_confirmed:
                bt.open_position(
                    symbol=symbol,
                    price=current_price,
                    side=PositionSide.SHORT,
                    reason=f"Momentum SHORT: Score={signal_score}, ADX={current_adx:.1f}"
                )

    def create_strategy_function(
        self,
        entry_threshold: int = 30,
        exit_threshold: int = -10,
        min_adx: float = 25,
        momentum_period: int = 12
    ) -> Callable:
        """
        Create a standalone strategy function for use with external backtesting.

        This method creates a strategy function that can be used directly with
        the Backtester class or walk-forward analysis.

        Args:
            entry_threshold: Minimum signal score to enter position
            exit_threshold: Signal score threshold to exit position
            min_adx: Minimum ADX for trend confirmation
            momentum_period: ROC period

        Returns:
            Strategy function with signature (backtester, date, row, symbol, data)

        Example:
            >>> strategy_func = module.create_strategy_function(entry_threshold=40)
            >>> walk_forward.run_rolling_walk_forward(data, strategy_func)
        """
        from quantsploit.utils.backtesting import PositionSide

        def strategy_func(bt, date, row, symbol: str, full_data: pd.DataFrame):
            """Momentum strategy function for backtesting."""
            # Calculate indicators if not present
            if 'signal_score' not in full_data.columns:
                full_data = self._prepare_indicators(full_data, momentum_period, min_adx)

            # Get current values
            try:
                current_idx = full_data.index.get_loc(date)
                signal_score = full_data['signal_score'].iloc[current_idx]
                current_adx = full_data['ADX_14'].iloc[current_idx] if 'ADX_14' in full_data.columns else 0
                current_roc = full_data['roc_12'].iloc[current_idx]
                sma_10 = full_data['sma_10'].iloc[current_idx]
                sma_20 = full_data['sma_20'].iloc[current_idx]
            except (KeyError, IndexError):
                return

            if pd.isna(signal_score):
                return

            current_price = row['Close']
            position = bt.get_position(symbol)

            # Exit logic
            if position is not None:
                if position.side == PositionSide.LONG:
                    if signal_score < exit_threshold or sma_10 < sma_20 or current_roc < -5:
                        bt.close_position(symbol, current_price)
                        return
                elif position.side == PositionSide.SHORT:
                    if signal_score > -exit_threshold or sma_10 > sma_20 or current_roc > 5:
                        bt.close_position(symbol, current_price)
                        return

            # Entry logic
            if position is None:
                trend_confirmed = current_adx >= min_adx
                if signal_score >= entry_threshold and trend_confirmed:
                    bt.open_position(symbol, current_price, PositionSide.LONG)
                elif signal_score <= -entry_threshold and trend_confirmed:
                    bt.open_position(symbol, current_price, PositionSide.SHORT)

        return strategy_func
