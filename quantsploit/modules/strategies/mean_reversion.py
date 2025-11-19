"""
Mean Reversion Strategy with Z-Score Analysis
"""

import pandas as pd
import numpy as np
from quantsploit.utils.ta_compat import ta
from typing import Dict, Any
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
        return """Mean Reversion - Trades oversold/overbought extremes back to average.

SYNOPSIS: Z-score measures price deviation from 20-day mean. |Z| >2.0 signals
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
