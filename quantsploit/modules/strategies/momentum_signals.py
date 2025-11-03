"""
Advanced Momentum and Trend Following Signals
"""

import pandas as pd
import numpy as np
from quantsploit.utils.ta_compat import ta
from typing import Dict, Any
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
        return "Advanced momentum and trend following signals with multiple confirmations"

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
