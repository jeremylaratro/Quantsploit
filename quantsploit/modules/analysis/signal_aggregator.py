"""
Signal Aggregator - Combines Multiple Strategies for Consensus Signals
"""

import pandas as pd
import numpy as np
from quantsploit.utils.ta_compat import ta
from typing import Dict, Any, List
from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher


class SignalAggregator(BaseModule):
    """
    Aggregates signals from multiple quantitative strategies to provide
    consensus buy/sell recommendations with confidence scores
    """

    @property
    def name(self) -> str:
        return "Signal Aggregator"

    @property
    def description(self) -> str:
        return "Aggregate signals from multiple strategies for consensus recommendations"

    @property
    def author(self) -> str:
        return "Quantsploit Team"

    @property
    def category(self) -> str:
        return "analysis"

    def _init_options(self):
        super()._init_options()
        self.options.update({
            "STRATEGIES": {
                "value": "all",
                "required": False,
                "description": "Strategies to aggregate: all, momentum, mean_reversion, technical, pattern"
            },
            "MIN_CONFIDENCE": {
                "value": 60,
                "required": False,
                "description": "Minimum confidence threshold for signals (0-100)"
            }
        })

    def run(self) -> Dict[str, Any]:
        """Execute signal aggregation"""
        symbol = self.get_option("SYMBOL")
        period = self.get_option("PERIOD")
        interval = self.get_option("INTERVAL")
        min_confidence = float(self.get_option("MIN_CONFIDENCE"))

        # Fetch data
        fetcher = DataFetcher(self.framework.database)
        df = fetcher.get_stock_data(symbol, period, interval)

        if df is None or df.empty:
            return {"success": False, "error": f"Failed to fetch data for {symbol}"}

        # Run all strategies
        self.log("Running momentum analysis...")
        momentum_signal = self._get_momentum_signals(df)

        self.log("Running mean reversion analysis...")
        mean_reversion_signal = self._get_mean_reversion_signals(df)

        self.log("Running technical analysis...")
        technical_signal = self._get_technical_signals(df)

        self.log("Running pattern analysis...")
        pattern_signal = self._get_pattern_signals(df)

        self.log("Running volume analysis...")
        volume_signal = self._get_volume_signals(df)

        # Aggregate signals
        all_signals = {
            "momentum": momentum_signal,
            "mean_reversion": mean_reversion_signal,
            "technical": technical_signal,
            "pattern": pattern_signal,
            "volume": volume_signal
        }

        # Calculate consensus
        consensus = self._calculate_consensus(all_signals)

        # Generate final recommendation
        final_signal, confidence = self._generate_final_signal(consensus, min_confidence)

        # Risk assessment
        risk_assessment = self._assess_risk(df, all_signals)

        return {
            "symbol": symbol,
            "current_price": df['Close'].iloc[-1],
            "final_signal": final_signal,
            "confidence": f"{confidence:.1f}%",
            "consensus_score": consensus['score'],
            "strategy_signals": {
                "momentum": momentum_signal,
                "mean_reversion": mean_reversion_signal,
                "technical": technical_signal,
                "pattern": pattern_signal,
                "volume": volume_signal
            },
            "bullish_signals": consensus['bullish_count'],
            "bearish_signals": consensus['bearish_count'],
            "neutral_signals": consensus['neutral_count'],
            "risk_assessment": risk_assessment,
            "actionable_insights": self._generate_insights(all_signals, consensus, risk_assessment)
        }

    def _get_momentum_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get momentum-based signals"""
        close = df['Close']

        # ROC
        roc_12 = ta.roc(close, length=12)
        current_roc = roc_12.iloc[-1] if roc_12 is not None else 0

        # Momentum score
        if current_roc > 10:
            signal = "bullish"
            score = 80
        elif current_roc > 5:
            signal = "bullish"
            score = 65
        elif current_roc < -10:
            signal = "bearish"
            score = 80
        elif current_roc < -5:
            signal = "bearish"
            score = 65
        else:
            signal = "neutral"
            score = 50

        return {
            "signal": signal,
            "score": score,
            "detail": f"ROC(12) = {current_roc:.2f}%"
        }

    def _get_mean_reversion_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get mean reversion signals"""
        close = df['Close']

        # Z-score
        mean = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        z_score = (close - mean) / std

        current_z = z_score.iloc[-1] if not pd.isna(z_score.iloc[-1]) else 0

        # RSI
        rsi = ta.rsi(close, length=14)
        current_rsi = rsi.iloc[-1] if rsi is not None else 50

        # Scoring
        if current_z < -2 or current_rsi < 30:
            signal = "bullish"  # Oversold
            score = 75
        elif current_z < -1.5 or current_rsi < 35:
            signal = "bullish"
            score = 60
        elif current_z > 2 or current_rsi > 70:
            signal = "bearish"  # Overbought
            score = 75
        elif current_z > 1.5 or current_rsi > 65:
            signal = "bearish"
            score = 60
        else:
            signal = "neutral"
            score = 50

        return {
            "signal": signal,
            "score": score,
            "detail": f"Z-score = {current_z:.2f}, RSI = {current_rsi:.1f}"
        }

    def _get_technical_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get technical indicator signals"""
        close = df['Close']
        high = df['High']
        low = df['Low']

        signals_count = {"bullish": 0, "bearish": 0}

        # MACD
        macd = ta.macd(close)
        if macd is not None:
            macd_line = macd['MACD_12_26_9'].iloc[-1]
            macd_signal = macd['MACDs_12_26_9'].iloc[-1]
            if macd_line > macd_signal:
                signals_count["bullish"] += 1
            else:
                signals_count["bearish"] += 1

        # Moving averages
        sma_20 = ta.sma(close, length=20)
        sma_50 = ta.sma(close, length=50)
        if sma_20 is not None and sma_50 is not None:
            if sma_20.iloc[-1] > sma_50.iloc[-1]:
                signals_count["bullish"] += 1
            else:
                signals_count["bearish"] += 1

        # Price vs MA
        if close.iloc[-1] > sma_20.iloc[-1]:
            signals_count["bullish"] += 1
        else:
            signals_count["bearish"] += 1

        # ADX
        adx = ta.adx(high, low, close, length=14)
        strong_trend = False
        if adx is not None and 'ADX_14' in adx.columns:
            if adx['ADX_14'].iloc[-1] > 25:
                strong_trend = True

        # Determine signal
        if signals_count["bullish"] > signals_count["bearish"]:
            signal = "bullish"
            score = 70 if strong_trend else 60
        elif signals_count["bearish"] > signals_count["bullish"]:
            signal = "bearish"
            score = 70 if strong_trend else 60
        else:
            signal = "neutral"
            score = 50

        return {
            "signal": signal,
            "score": score,
            "detail": f"Bull:{signals_count['bullish']}, Bear:{signals_count['bearish']}"
        }

    def _get_pattern_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get pattern-based signals"""
        O = df['Open'].values
        H = df['High'].values
        L = df['Low'].values
        C = df['Close'].values

        patterns = {"bullish": 0, "bearish": 0}

        # Check last 3 candles for patterns
        for i in range(max(0, len(df) - 3), len(df)):
            if i < 1:
                continue

            o, h, l, c = O[i], H[i], L[i], C[i]
            body = abs(c - o)
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            total_range = h - l

            if total_range == 0:
                continue

            # Hammer (bullish)
            if lower_shadow > 2 * body and upper_shadow < 0.3 * body:
                patterns["bullish"] += 1

            # Shooting star (bearish)
            if upper_shadow > 2 * body and lower_shadow < 0.3 * body:
                patterns["bearish"] += 1

            # Engulfing patterns
            if i > 0:
                o1, c1 = O[i-1], C[i-1]
                # Bullish engulfing
                if c1 < o1 and c > o and o < c1 and c > o1:
                    patterns["bullish"] += 1
                # Bearish engulfing
                if c1 > o1 and c < o and o > c1 and c < o1:
                    patterns["bearish"] += 1

        # Determine signal
        if patterns["bullish"] > patterns["bearish"] and patterns["bullish"] > 0:
            signal = "bullish"
            score = 65
        elif patterns["bearish"] > patterns["bullish"] and patterns["bearish"] > 0:
            signal = "bearish"
            score = 65
        else:
            signal = "neutral"
            score = 50

        return {
            "signal": signal,
            "score": score,
            "detail": f"Bullish patterns: {patterns['bullish']}, Bearish: {patterns['bearish']}"
        }

    def _get_volume_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get volume-based signals"""
        volume = df['Volume']
        close = df['Close']

        # Volume ratio
        avg_volume = volume.rolling(window=20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Price change
        price_change = (close.iloc[-1] / close.iloc[-2] - 1) * 100

        # OBV
        obv = ta.obv(close, volume)
        obv_trend = "neutral"
        if obv is not None and len(obv) > 5:
            if obv.iloc[-1] > obv.iloc[-5]:
                obv_trend = "bullish"
            elif obv.iloc[-1] < obv.iloc[-5]:
                obv_trend = "bearish"

        # Scoring
        if volume_ratio > 1.5 and price_change > 0 and obv_trend == "bullish":
            signal = "bullish"
            score = 75
        elif volume_ratio > 1.2 and price_change > 0:
            signal = "bullish"
            score = 60
        elif volume_ratio > 1.5 and price_change < 0 and obv_trend == "bearish":
            signal = "bearish"
            score = 75
        elif volume_ratio > 1.2 and price_change < 0:
            signal = "bearish"
            score = 60
        else:
            signal = "neutral"
            score = 50

        return {
            "signal": signal,
            "score": score,
            "detail": f"Vol ratio: {volume_ratio:.2f}x, OBV: {obv_trend}"
        }

    def _calculate_consensus(self, signals: Dict[str, Dict]) -> Dict:
        """Calculate consensus from all signals"""
        bullish_count = sum(1 for s in signals.values() if s['signal'] == 'bullish')
        bearish_count = sum(1 for s in signals.values() if s['signal'] == 'bearish')
        neutral_count = sum(1 for s in signals.values() if s['signal'] == 'neutral')

        # Weighted score
        total_score = sum(s['score'] for s in signals.values())
        bullish_score = sum(s['score'] for s in signals.values() if s['signal'] == 'bullish')
        bearish_score = sum(s['score'] for s in signals.values() if s['signal'] == 'bearish')

        # Net consensus score (-100 to +100)
        consensus_score = ((bullish_score - bearish_score) / total_score * 100) if total_score > 0 else 0

        return {
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "score": consensus_score
        }

    def _generate_final_signal(self, consensus: Dict, min_confidence: float) -> tuple:
        """Generate final signal and confidence"""
        score = consensus['score']
        bullish_count = consensus['bullish_count']
        bearish_count = consensus['bearish_count']

        # Calculate confidence (0-100)
        agreement = max(bullish_count, bearish_count) / (bullish_count + bearish_count + consensus['neutral_count'])
        confidence = agreement * 100

        # Generate signal
        if score > 30 and confidence >= min_confidence:
            if score > 60:
                signal = "🟢 STRONG BUY - High conviction"
            else:
                signal = "🟢 BUY - Moderate conviction"
        elif score < -30 and confidence >= min_confidence:
            if score < -60:
                signal = "🔴 STRONG SELL - High conviction"
            else:
                signal = "🔴 SELL - Moderate conviction"
        else:
            signal = "⚪ HOLD - Mixed signals or low confidence"

        return signal, confidence

    def _assess_risk(self, df: pd.DataFrame, signals: Dict) -> Dict:
        """Assess risk level"""
        close = df['Close']

        # Volatility
        returns = close.pct_change()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized

        # ATR
        atr = ta.atr(df['High'], df['Low'], close, length=14)
        atr_pct = (atr.iloc[-1] / close.iloc[-1]) * 100 if atr is not None else 0

        # Signal agreement (lower agreement = higher risk)
        signal_values = [s['signal'] for s in signals.values()]
        unique_signals = len(set(signal_values))
        signal_risk = "high" if unique_signals == 3 else "medium" if unique_signals == 2 else "low"

        # Overall risk
        if volatility > 40 or atr_pct > 5:
            overall_risk = "HIGH"
        elif volatility > 25 or atr_pct > 3:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"

        return {
            "overall_risk": overall_risk,
            "volatility": f"{volatility:.1f}%",
            "atr_percent": f"{atr_pct:.2f}%",
            "signal_disagreement": signal_risk
        }

    def _generate_insights(self, signals: Dict, consensus: Dict, risk: Dict) -> List[str]:
        """Generate actionable insights"""
        insights = []

        # Strategy agreement
        bullish_strategies = [name for name, sig in signals.items() if sig['signal'] == 'bullish']
        bearish_strategies = [name for name, sig in signals.items() if sig['signal'] == 'bearish']

        if len(bullish_strategies) >= 4:
            insights.append(f"✅ Strong agreement across strategies: {', '.join(bullish_strategies)}")
        elif len(bearish_strategies) >= 4:
            insights.append(f"⚠️ Multiple bearish signals from: {', '.join(bearish_strategies)}")

        # Risk warning
        if risk['overall_risk'] == "HIGH":
            insights.append("⚠️ HIGH RISK: High volatility detected - use smaller position sizes")

        # Conflicting signals
        if consensus['bullish_count'] > 0 and consensus['bearish_count'] > 0:
            insights.append("⚠️ Mixed signals detected - consider waiting for clearer setup")

        # Strong consensus
        if abs(consensus['score']) > 60:
            insights.append(f"💪 Strong consensus: {abs(consensus['score']):.0f}/100")

        return insights if insights else ["No special insights at this time"]
