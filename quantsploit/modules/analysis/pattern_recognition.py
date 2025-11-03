"""
Advanced Pattern Recognition Module
Detects candlestick patterns and chart patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher


class PatternRecognition(BaseModule):
    """
    Detect candlestick patterns, chart patterns, and technical setups
    """

    @property
    def name(self) -> str:
        return "Pattern Recognition"

    @property
    def description(self) -> str:
        return "Detect bullish/bearish patterns including candlesticks and chart patterns"

    @property
    def author(self) -> str:
        return "Quantsploit Team"

    @property
    def category(self) -> str:
        return "analysis"

    def _init_options(self):
        super()._init_options()
        self.options.update({
            "LOOKBACK": {
                "value": 50,
                "required": False,
                "description": "Number of candles to analyze"
            },
            "PATTERNS": {
                "value": "all",
                "required": False,
                "description": "Pattern types: candlestick, chart, all"
            }
        })

    def run(self) -> Dict[str, Any]:
        """Execute pattern recognition"""
        symbol = self.get_option("SYMBOL")
        period = self.get_option("PERIOD")
        interval = self.get_option("INTERVAL")
        lookback = int(self.get_option("LOOKBACK"))

        # Fetch data
        fetcher = DataFetcher(self.framework.database)
        df = fetcher.get_stock_data(symbol, period, interval)

        if df is None or df.empty:
            return {"success": False, "error": f"Failed to fetch data for {symbol}"}

        # Limit to lookback period
        df = df.tail(lookback)

        results = {
            "symbol": symbol,
            "current_price": df['Close'].iloc[-1],
            "candlestick_patterns": [],
            "chart_patterns": [],
            "support_resistance": {},
            "signals": []
        }

        # Detect candlestick patterns
        candlestick_patterns = self._detect_candlestick_patterns(df)
        results["candlestick_patterns"] = candlestick_patterns

        # Detect chart patterns
        chart_patterns = self._detect_chart_patterns(df)
        results["chart_patterns"] = chart_patterns

        # Find support and resistance
        results["support_resistance"] = self._find_support_resistance(df)

        # Generate signals
        results["signals"] = self._generate_signals(
            candlestick_patterns, chart_patterns, results["support_resistance"], df
        )

        # Add pattern summary
        results["pattern_summary"] = pd.DataFrame({
            "Pattern Type": ["Bullish Candlestick", "Bearish Candlestick", "Chart Patterns", "Total Signals"],
            "Count": [
                len([p for p in candlestick_patterns if p['sentiment'] == 'bullish']),
                len([p for p in candlestick_patterns if p['sentiment'] == 'bearish']),
                len(chart_patterns),
                len(results["signals"])
            ]
        })

        return results

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect candlestick patterns"""
        patterns = []

        # Get OHLC data
        O = df['Open'].values
        H = df['High'].values
        L = df['Low'].values
        C = df['Close'].values

        # Check last few candles for patterns
        for i in range(len(df) - 5, len(df)):
            if i < 2:
                continue

            # Hammer (bullish reversal)
            if self._is_hammer(O[i], H[i], L[i], C[i]):
                patterns.append({
                    "pattern": "Hammer",
                    "sentiment": "bullish",
                    "strength": "strong",
                    "index": i,
                    "description": "Bullish reversal pattern after downtrend"
                })

            # Shooting Star (bearish reversal)
            if self._is_shooting_star(O[i], H[i], L[i], C[i]):
                patterns.append({
                    "pattern": "Shooting Star",
                    "sentiment": "bearish",
                    "strength": "strong",
                    "index": i,
                    "description": "Bearish reversal pattern after uptrend"
                })

            # Engulfing patterns
            if i > 0:
                # Bullish Engulfing
                if self._is_bullish_engulfing(O[i-1], C[i-1], O[i], C[i]):
                    patterns.append({
                        "pattern": "Bullish Engulfing",
                        "sentiment": "bullish",
                        "strength": "strong",
                        "index": i,
                        "description": "Strong bullish reversal pattern"
                    })

                # Bearish Engulfing
                if self._is_bearish_engulfing(O[i-1], C[i-1], O[i], C[i]):
                    patterns.append({
                        "pattern": "Bearish Engulfing",
                        "sentiment": "bearish",
                        "strength": "strong",
                        "index": i,
                        "description": "Strong bearish reversal pattern"
                    })

            # Doji (indecision)
            if self._is_doji(O[i], C[i], H[i], L[i]):
                patterns.append({
                    "pattern": "Doji",
                    "sentiment": "neutral",
                    "strength": "medium",
                    "index": i,
                    "description": "Indecision, potential reversal"
                })

            # Morning Star (bullish)
            if i >= 2:
                if self._is_morning_star(
                    O[i-2], C[i-2], O[i-1], C[i-1], O[i], C[i]
                ):
                    patterns.append({
                        "pattern": "Morning Star",
                        "sentiment": "bullish",
                        "strength": "very_strong",
                        "index": i,
                        "description": "Very strong bullish reversal"
                    })

                # Evening Star (bearish)
                if self._is_evening_star(
                    O[i-2], C[i-2], O[i-1], C[i-1], O[i], C[i]
                ):
                    patterns.append({
                        "pattern": "Evening Star",
                        "sentiment": "bearish",
                        "strength": "very_strong",
                        "index": i,
                        "description": "Very strong bearish reversal"
                    })

        return patterns

    def _is_hammer(self, o, h, l, c):
        """Detect hammer pattern"""
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l

        if total_range == 0:
            return False

        return (lower_shadow > 2 * body and
                upper_shadow < 0.3 * body and
                body > 0.1 * total_range)

    def _is_shooting_star(self, o, h, l, c):
        """Detect shooting star pattern"""
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l

        if total_range == 0:
            return False

        return (upper_shadow > 2 * body and
                lower_shadow < 0.3 * body and
                body > 0.1 * total_range)

    def _is_bullish_engulfing(self, o1, c1, o2, c2):
        """Detect bullish engulfing"""
        return c1 < o1 and c2 > o2 and o2 < c1 and c2 > o1

    def _is_bearish_engulfing(self, o1, c1, o2, c2):
        """Detect bearish engulfing"""
        return c1 > o1 and c2 < o2 and o2 > c1 and c2 < o1

    def _is_doji(self, o, c, h, l):
        """Detect doji"""
        body = abs(c - o)
        total_range = h - l
        return total_range > 0 and body < 0.1 * total_range

    def _is_morning_star(self, o1, c1, o2, c2, o3, c3):
        """Detect morning star (3-candle bullish reversal)"""
        # First candle bearish
        if c1 >= o1:
            return False
        # Second candle small body (star)
        if abs(c2 - o2) > abs(c1 - o1) * 0.5:
            return False
        # Third candle bullish and closes above midpoint of first
        if c3 <= o3:
            return False
        midpoint = (o1 + c1) / 2
        return c3 > midpoint

    def _is_evening_star(self, o1, c1, o2, c2, o3, c3):
        """Detect evening star (3-candle bearish reversal)"""
        # First candle bullish
        if c1 <= o1:
            return False
        # Second candle small body (star)
        if abs(c2 - o2) > abs(c1 - o1) * 0.5:
            return False
        # Third candle bearish and closes below midpoint of first
        if c3 >= o3:
            return False
        midpoint = (o1 + c1) / 2
        return c3 < midpoint

    def _detect_chart_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect chart patterns"""
        patterns = []
        close = df['Close'].values

        # Double bottom/top
        if len(close) >= 20:
            # Simple double bottom detection
            min_idx = []
            for i in range(5, len(close) - 5):
                if close[i] == min(close[i-5:i+6]):
                    min_idx.append(i)

            # Check for double bottom
            for i in range(len(min_idx) - 1):
                idx1, idx2 = min_idx[i], min_idx[i+1]
                if 10 <= idx2 - idx1 <= 40:  # Reasonable distance
                    if abs(close[idx1] - close[idx2]) / close[idx1] < 0.02:  # Similar lows
                        patterns.append({
                            "pattern": "Double Bottom",
                            "sentiment": "bullish",
                            "strength": "strong",
                            "description": "Bullish reversal pattern"
                        })

        # Head and Shoulders (simplified)
        if len(close) >= 30:
            for i in range(10, len(close) - 10):
                left = close[i-10]
                head = close[i]
                right = close[i+10]

                if head > left and head > right and abs(left - right) / left < 0.03:
                    patterns.append({
                        "pattern": "Head and Shoulders",
                        "sentiment": "bearish",
                        "strength": "strong",
                        "description": "Bearish reversal pattern"
                    })
                    break

        # Ascending/Descending Triangle
        if len(close) >= 20:
            recent = close[-20:]
            highs = [recent[i] for i in range(len(recent)) if i == 0 or recent[i] > recent[i-1]]
            lows = [recent[i] for i in range(len(recent)) if i == 0 or recent[i] < recent[i-1]]

            if len(highs) >= 3 and len(lows) >= 3:
                # Check for ascending triangle (flat top, rising bottom)
                high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
                low_slope = np.polyfit(range(len(lows)), lows, 1)[0]

                if abs(high_slope) < 0.01 and low_slope > 0.01:
                    patterns.append({
                        "pattern": "Ascending Triangle",
                        "sentiment": "bullish",
                        "strength": "medium",
                        "description": "Continuation pattern, likely breakout up"
                    })

        return patterns

    def _find_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values

        # Find local maxima and minima
        resistance_levels = []
        support_levels = []

        for i in range(5, len(close) - 5):
            # Resistance (local max)
            if high[i] == max(high[i-5:i+6]):
                resistance_levels.append(high[i])

            # Support (local min)
            if low[i] == min(low[i-5:i+6]):
                support_levels.append(low[i])

        # Cluster levels
        current_price = close[-1]

        return {
            "resistance": sorted(list(set([round(r, 2) for r in resistance_levels])), reverse=True)[:5],
            "support": sorted(list(set([round(s, 2) for s in support_levels])), reverse=True)[:5],
            "current_price": round(current_price, 2),
            "nearest_resistance": min([r for r in resistance_levels if r > current_price], default=None),
            "nearest_support": max([s for s in support_levels if s < current_price], default=None)
        }

    def _generate_signals(self, candlestick_patterns, chart_patterns,
                         support_resistance, df) -> List[str]:
        """Generate trading signals from patterns"""
        signals = []

        # Count bullish/bearish candlestick patterns
        bullish_count = len([p for p in candlestick_patterns if p['sentiment'] == 'bullish'])
        bearish_count = len([p for p in candlestick_patterns if p['sentiment'] == 'bearish'])

        if bullish_count > bearish_count and bullish_count >= 2:
            signals.append("🟢 STRONG BUY - Multiple bullish patterns detected")
        elif bullish_count > 0:
            signals.append("🟢 BUY - Bullish pattern detected")

        if bearish_count > bullish_count and bearish_count >= 2:
            signals.append("🔴 STRONG SELL - Multiple bearish patterns detected")
        elif bearish_count > 0:
            signals.append("🔴 SELL - Bearish pattern detected")

        # Check chart patterns
        for pattern in chart_patterns:
            if pattern['sentiment'] == 'bullish':
                signals.append(f"🟢 BUY - {pattern['pattern']} detected")
            elif pattern['sentiment'] == 'bearish':
                signals.append(f"🔴 SELL - {pattern['pattern']} detected")

        # Support/resistance signals
        current_price = df['Close'].iloc[-1]
        nearest_support = support_resistance.get('nearest_support')
        nearest_resistance = support_resistance.get('nearest_resistance')

        if nearest_support and current_price <= nearest_support * 1.01:
            signals.append(f"🟢 Near support level at ${nearest_support:.2f} - potential bounce")

        if nearest_resistance and current_price >= nearest_resistance * 0.99:
            signals.append(f"🔴 Near resistance at ${nearest_resistance:.2f} - potential rejection")

        return signals if signals else ["⚪ NEUTRAL - No clear signals"]
