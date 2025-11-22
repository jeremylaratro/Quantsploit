"""
Multi-Factor Quantitative Scoring System
Combines multiple quantitative factors to rank stocks
"""

import pandas as pd
import numpy as np
from quantsploit.utils.ta_compat import ta
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher


class MultiFactorScoring(BaseModule):
    """
    Advanced multi-factor model combining:
    - Value factors
    - Momentum factors
    - Quality factors
    - Volatility factors
    - Technical factors
    """

    @property
    def name(self) -> str:
        return "Multi-Factor Scoring"

    @property
    def description(self) -> str:
        return "Ranks stocks using momentum, technical, volatility, and volume factors"

    def trading_guide(self) -> str:
        return """SYNOPSIS: Scores each stock 0-100 in momentum, technical, volatility (inverted),
and volume. Weighted composite determines rank.

SIMULATION POSITIONS:
  - Analysis/ranking only (no backtested trades)
  - Processes multiple symbols in parallel
  - Returns sorted list by composite score

RECOMMENDED ENTRY:
  - Composite score ≥75: STRONG BUY - Enter full position
  - Composite score 60-74: BUY - Enter partial position
  - Composite score <40: Avoid or reduce exposure
  - Focus on top 3-5 ranked stocks from screened universe

DEFAULT WEIGHTS:
  - Momentum: 30% (ROC, acceleration)
  - Technical: 30% (RSI, MACD, MA crossovers, ADX)
  - Volatility: 20% (lower is better - ATR, BB width)
  - Volume: 20% (volume trend, OBV, spikes)

REAL-WORLD USE:
  1. Screen 20-50 stocks (e.g., S&P sectors, watchlist)
  2. Run multifactor scoring
  3. Take top 5 scores ≥70 for further analysis
  4. Combine with other strategies for entry timing"""

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
            "SYMBOLS": {
                "value": None,
                "required": True,
                "description": "Comma-separated symbols to score"
            },
            "FACTOR_WEIGHTS": {
                "value": "momentum:0.3,technical:0.3,volatility:0.2,volume:0.2",
                "required": False,
                "description": "Factor weights (momentum,technical,volatility,volume)"
            },
            "MAX_WORKERS": {
                "value": 10,
                "required": False,
                "description": "Parallel processing workers"
            }
        })
        self.options["SYMBOL"]["required"] = False

    def run(self) -> Dict[str, Any]:
        """Execute multi-factor scoring"""
        # Handle both SYMBOLS (plural, for standalone use) and SYMBOL (singular, from meta_analysis)
        symbols_str = self.get_option("SYMBOLS")
        if symbols_str is None:
            # Try getting SYMBOL instead (when called from meta_analysis)
            symbol = self.get_option("SYMBOL")
            if symbol:
                symbols_str = symbol
            else:
                return {"error": "No symbols provided. Set SYMBOLS or SYMBOL option."}

        period = self.get_option("PERIOD")
        interval = self.get_option("INTERVAL")
        max_workers = int(self.get_option("MAX_WORKERS"))

        # Parse factor weights
        factor_weights = self._parse_weights(self.get_option("FACTOR_WEIGHTS"))

        symbols = [s.strip().upper() for s in symbols_str.split(",")]

        self.log(f"Scoring {len(symbols)} stocks using multi-factor model...")

        # Score stocks in parallel
        # Note: Don't pass database to avoid SQLite threading issues
        fetcher = DataFetcher(database=None, cache_enabled=False)
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self._score_stock, symbol, fetcher, period, interval, factor_weights
                ): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    self.log(f"Error scoring {symbol}: {str(e)}", "warning")

        # Sort by composite score
        results.sort(key=lambda x: x['composite_score'], reverse=True)

        # Create DataFrame
        if results:
            df = pd.DataFrame(results)
        else:
            df = pd.DataFrame()

        # Analyze factor performance
        factor_analysis = self._analyze_factors(results) if results else {}

        return {
            "total_scored": len(results),
            "factor_weights": factor_weights,
            "top_picks": df.head(10) if not df.empty else pd.DataFrame(),
            "all_scores": df,
            "factor_analysis": factor_analysis,
            "recommendation": self._generate_recommendations(results)
        }

    def _parse_weights(self, weights_str: str) -> Dict[str, float]:
        """Parse factor weights from string"""
        weights = {}
        for item in weights_str.split(","):
            if ":" in item:
                factor, weight = item.split(":")
                weights[factor.strip()] = float(weight.strip())

        # Normalize to sum to 1.0
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def _score_stock(self, symbol: str, fetcher: DataFetcher,
                    period: str, interval: str, factor_weights: Dict) -> Dict[str, Any]:
        """Score a single stock across multiple factors"""
        df = fetcher.get_stock_data(symbol, period, interval)

        if df is None or df.empty or len(df) < 50:
            return None

        close = df['Close']
        volume = df['Volume']
        high = df['High']
        low = df['Low']

        # Initialize scores
        scores = {}

        # 1. MOMENTUM FACTORS (0-100)
        scores['momentum'] = self._score_momentum(close)

        # 2. TECHNICAL FACTORS (0-100)
        scores['technical'] = self._score_technical(df, close, high, low)

        # 3. VOLATILITY FACTORS (0-100) - lower volatility scores higher
        scores['volatility'] = self._score_volatility(close, high, low)

        # 4. VOLUME FACTORS (0-100)
        scores['volume'] = self._score_volume(volume, close)

        # Calculate composite score
        composite_score = sum(
            scores.get(factor, 0) * weight
            for factor, weight in factor_weights.items()
        )

        # Get current metrics
        current_price = close.iloc[-1]
        price_change_pct = ((close.iloc[-1] / close.iloc[-2]) - 1) * 100

        # Generate signal
        if composite_score >= 75:
            signal = "🟢 STRONG BUY"
        elif composite_score >= 60:
            signal = "🟢 BUY"
        elif composite_score >= 40:
            signal = "⚪ HOLD"
        elif composite_score >= 25:
            signal = "🔴 SELL"
        else:
            signal = "🔴 STRONG SELL"

        return {
            "Symbol": symbol,
            "Price": round(current_price, 2),
            "Change%": round(price_change_pct, 2),
            "composite_score": round(composite_score, 2),
            "momentum_score": round(scores['momentum'], 2),
            "technical_score": round(scores['technical'], 2),
            "volatility_score": round(scores['volatility'], 2),
            "volume_score": round(scores['volume'], 2),
            "Signal": signal
        }

    def _score_momentum(self, close: pd.Series) -> float:
        """Score momentum factors (0-100)"""
        score = 50.0

        # 12-month momentum (minus most recent month)
        if len(close) >= 252:  # ~1 year of daily data
            annual_return = ((close.iloc[-1] / close.iloc[-252]) - 1) * 100
            score += min(max(annual_return, -25), 25)  # Cap at ±25

        # Rate of change
        roc_20 = ta.roc(close, length=20)
        if roc_20 is not None and not pd.isna(roc_20.iloc[-1]):
            score += min(max(roc_20.iloc[-1], -10), 10)  # Cap at ±10

        # Momentum acceleration
        roc_5 = ta.roc(close, length=5)
        if roc_5 is not None and len(roc_5) > 1:
            accel = roc_5.iloc[-1] - roc_5.iloc[-2]
            score += min(max(accel * 2, -10), 10)

        return max(0, min(100, score))

    def _score_technical(self, df: pd.DataFrame, close: pd.Series,
                        high: pd.Series, low: pd.Series) -> float:
        """Score technical factors (0-100)"""
        score = 50.0

        # RSI (prefer 40-60 range)
        rsi = ta.rsi(close, length=14)
        if rsi is not None and not pd.isna(rsi.iloc[-1]):
            current_rsi = rsi.iloc[-1]
            if 40 <= current_rsi <= 60:
                score += 15
            elif 30 <= current_rsi < 40:
                score += 10  # Slightly oversold
            elif current_rsi < 30:
                score += 20  # Oversold opportunity
            elif current_rsi > 70:
                score -= 15  # Overbought warning

        # MACD
        macd = ta.macd(close)
        if macd is not None:
            macd_line = macd['MACD_12_26_9'].iloc[-1]
            macd_signal = macd['MACDs_12_26_9'].iloc[-1]
            if not pd.isna(macd_line) and not pd.isna(macd_signal):
                if macd_line > macd_signal:
                    score += 15
                else:
                    score -= 10

        # Moving average crossovers
        sma_20 = ta.sma(close, length=20)
        sma_50 = ta.sma(close, length=50)

        if sma_20 is not None and sma_50 is not None:
            if not pd.isna(sma_20.iloc[-1]) and not pd.isna(sma_50.iloc[-1]):
                # Golden cross
                if sma_20.iloc[-1] > sma_50.iloc[-1]:
                    score += 15
                # Price above both MAs
                if close.iloc[-1] > sma_20.iloc[-1]:
                    score += 10

        # ADX trend strength
        adx = ta.adx(high, low, close, length=14)
        if adx is not None and 'ADX_14' in adx.columns:
            adx_val = adx['ADX_14'].iloc[-1]
            if not pd.isna(adx_val) and adx_val > 25:
                score += 10

        return max(0, min(100, score))

    def _score_volatility(self, close: pd.Series, high: pd.Series, low: pd.Series) -> float:
        """Score volatility factors (0-100) - lower volatility scores higher"""
        score = 50.0

        # ATR-based volatility
        atr = ta.atr(high, low, close, length=14)
        if atr is not None and not pd.isna(atr.iloc[-1]):
            atr_pct = (atr.iloc[-1] / close.iloc[-1]) * 100
            # Lower ATR% is better (less risky)
            if atr_pct < 2:
                score += 20
            elif atr_pct < 3:
                score += 10
            elif atr_pct > 5:
                score -= 15
            elif atr_pct > 7:
                score -= 25

        # Standard deviation
        std_20 = close.rolling(window=20).std()
        if len(std_20) > 0 and not pd.isna(std_20.iloc[-1]):
            std_pct = (std_20.iloc[-1] / close.iloc[-1]) * 100
            if std_pct < 2:
                score += 15
            elif std_pct > 4:
                score -= 15

        # Bollinger Band width
        bbands = ta.bbands(close, length=20)
        if bbands is not None:
            bb_width = (bbands['BBU_20_2.0'].iloc[-1] - bbands['BBL_20_2.0'].iloc[-1]) / bbands['BBM_20_2.0'].iloc[-1]
            if not pd.isna(bb_width):
                # Narrower bands = lower volatility = higher score
                if bb_width < 0.1:
                    score += 15
                elif bb_width > 0.2:
                    score -= 10

        return max(0, min(100, score))

    def _score_volume(self, volume: pd.Series, close: pd.Series) -> float:
        """Score volume factors (0-100)"""
        score = 50.0

        # Volume trend
        avg_volume_20 = volume.rolling(window=20).mean()
        avg_volume_50 = volume.rolling(window=50).mean()

        if len(avg_volume_20) > 0 and len(avg_volume_50) > 0:
            if not pd.isna(avg_volume_20.iloc[-1]) and not pd.isna(avg_volume_50.iloc[-1]):
                # Increasing volume is positive
                if avg_volume_20.iloc[-1] > avg_volume_50.iloc[-1]:
                    score += 15

        # Recent volume spike
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(window=20).mean().iloc[-1]
        if not pd.isna(avg_volume) and avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            if volume_ratio > 2.0:
                score += 20  # Strong volume
            elif volume_ratio > 1.5:
                score += 10
            elif volume_ratio < 0.5:
                score -= 15  # Very low volume

        # On-balance volume (OBV)
        obv = ta.obv(close, volume)
        if obv is not None and len(obv) > 10:
            obv_change = ((obv.iloc[-1] / obv.iloc[-10]) - 1) * 100
            score += min(max(obv_change, -15), 15)

        return max(0, min(100, score))

    def _analyze_factors(self, results: List[Dict]) -> Dict:
        """Analyze which factors are driving scores"""
        if not results:
            return {}

        df = pd.DataFrame(results)

        return {
            "avg_momentum": round(df['momentum_score'].mean(), 2),
            "avg_technical": round(df['technical_score'].mean(), 2),
            "avg_volatility": round(df['volatility_score'].mean(), 2),
            "avg_volume": round(df['volume_score'].mean(), 2),
            "top_momentum_stocks": df.nlargest(3, 'momentum_score')['Symbol'].tolist(),
            "top_technical_stocks": df.nlargest(3, 'technical_score')['Symbol'].tolist(),
            "lowest_volatility_stocks": df.nlargest(3, 'volatility_score')['Symbol'].tolist()
        }

    def _generate_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        if not results:
            return ["No data to analyze"]

        df = pd.DataFrame(results)

        recommendations = []

        # Top picks
        top_5 = df.nlargest(5, 'composite_score')
        if not top_5.empty:
            recommendations.append(
                f"🟢 Top picks: {', '.join(top_5['Symbol'].tolist())}"
            )

        # Strong buys
        strong_buys = df[df['Signal'] == "🟢 STRONG BUY"]
        if len(strong_buys) > 0:
            recommendations.append(
                f"🟢 {len(strong_buys)} stocks rated STRONG BUY"
            )

        # Avoid
        avoid = df[df['composite_score'] < 30]
        if len(avoid) > 0:
            recommendations.append(
                f"🔴 Avoid: {', '.join(avoid['Symbol'].tolist())}"
            )

        return recommendations
