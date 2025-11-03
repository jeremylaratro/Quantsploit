"""
Top Movers and Rankings Module
Identifies and ranks top opportunities across multiple dimensions
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher


class TopMovers(BaseModule):
    """
    Identify top movers, top gainers, best momentum plays,
    and rank stocks by various quantitative criteria
    """

    @property
    def name(self) -> str:
        return "Top Movers & Rankings"

    @property
    def description(self) -> str:
        return "Identify top movers and rank stocks by multiple criteria"

    @property
    def author(self) -> str:
        return "Quantsploit Team"

    @property
    def category(self) -> str:
        return "scanner"

    def _init_options(self):
        super()._init_options()
        self.options.update({
            "SYMBOLS": {
                "value": None,
                "required": True,
                "description": "Comma-separated symbols or 'SP500', 'NASDAQ100'"
            },
            "RANKING_METHOD": {
                "value": "all",
                "required": False,
                "description": "Ranking: all, gainers, momentum, breakout, oversold, quality"
            },
            "TIMEFRAME": {
                "value": "1d",
                "required": False,
                "description": "Performance timeframe: 1d, 5d, 1mo, 3mo"
            },
            "TOP_N": {
                "value": 20,
                "required": False,
                "description": "Number of top picks to return"
            },
            "MAX_WORKERS": {
                "value": 15,
                "required": False,
                "description": "Parallel processing workers"
            }
        })
        self.options["SYMBOL"]["required"] = False

    def run(self) -> Dict[str, Any]:
        """Execute top movers analysis"""
        symbols_input = self.get_option("SYMBOLS")
        ranking_method = self.get_option("RANKING_METHOD")
        timeframe = self.get_option("TIMEFRAME")
        top_n = int(self.get_option("TOP_N"))
        max_workers = int(self.get_option("MAX_WORKERS"))

        # Parse symbols
        symbols = self._parse_symbols(symbols_input)

        self.log(f"Analyzing {len(symbols)} stocks for top movers...")

        # Analyze stocks in parallel
        fetcher = DataFetcher(self.framework.database)
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self._analyze_stock, symbol, fetcher, timeframe
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
                    self.log(f"Error analyzing {symbol}: {str(e)}", "warning")

        if not results:
            return {"success": False, "error": "No stocks analyzed successfully"}

        # Generate rankings
        rankings = self._generate_rankings(results, ranking_method, top_n)

        return {
            "total_analyzed": len(symbols),
            "successful_analysis": len(results),
            "timeframe": timeframe,
            "ranking_method": ranking_method,
            **rankings
        }

    def _parse_symbols(self, symbols_input: str) -> List[str]:
        """Parse symbol input"""
        if symbols_input.upper() == "SP500":
            return self._get_sp500_symbols()
        elif symbols_input.upper() == "NASDAQ100":
            return self._get_nasdaq100_symbols()
        else:
            return [s.strip().upper() for s in symbols_input.split(",")]

    def _get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols"""
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
                'UNH', 'JNJ', 'V', 'XOM', 'WMT', 'JPM', 'PG', 'MA', 'CVX', 'HD',
                'LLY', 'ABBV', 'MRK', 'AVGO', 'KO', 'PEP', 'COST', 'ADBE', 'TMO',
                'MCD', 'ACN', 'CSCO', 'ABT', 'DHR', 'NKE', 'NFLX', 'TXN', 'DIS',
                'CRM', 'VZ', 'CMCSA', 'ORCL', 'WFC', 'AMD', 'INTC', 'BAC', 'PM',
                'NEE', 'RTX', 'BMY', 'UPS', 'T', 'LOW', 'HON', 'MS', 'QCOM', 'SPGI']

    def _get_nasdaq100_symbols(self) -> List[str]:
        """Get NASDAQ 100 symbols"""
        return ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
                'AVGO', 'COST', 'ASML', 'NFLX', 'AMD', 'PEP', 'ADBE', 'CSCO',
                'TMUS', 'CMCSA', 'INTC', 'TXN', 'QCOM', 'INTU', 'HON', 'AMAT',
                'SBUX', 'ISRG', 'BKNG', 'MDLZ', 'GILD', 'ADP', 'REGN', 'VRTX']

    def _analyze_stock(self, symbol: str, fetcher: DataFetcher, timeframe: str) -> Dict[str, Any]:
        """Analyze a single stock"""
        # Get appropriate period for timeframe
        period_map = {"1d": "5d", "5d": "1mo", "1mo": "3mo", "3mo": "1y"}
        period = period_map.get(timeframe, "3mo")

        df = fetcher.get_stock_data(symbol, period, "1d")

        if df is None or df.empty or len(df) < 20:
            return None

        close = df['Close']
        volume = df['Volume']
        high = df['High']
        low = df['Low']

        # Calculate metrics
        current_price = close.iloc[-1]

        # Performance calculation
        periods_map = {"1d": 1, "5d": 5, "1mo": 21, "3mo": 63}
        lookback = periods_map.get(timeframe, 1)

        if len(close) > lookback:
            period_return = ((close.iloc[-1] / close.iloc[-lookback-1]) - 1) * 100
        else:
            period_return = 0

        # Momentum metrics
        roc_20 = ta.roc(close, length=20)
        momentum_score = roc_20.iloc[-1] if roc_20 is not None else 0

        # RSI
        rsi = ta.rsi(close, length=14)
        current_rsi = rsi.iloc[-1] if rsi is not None else 50

        # Volume analysis
        avg_volume = volume.rolling(window=20).mean().iloc[-1]
        volume_ratio = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1

        # Volatility
        atr = ta.atr(high, low, close, length=14)
        atr_pct = (atr.iloc[-1] / current_price) * 100 if atr is not None else 0

        # Trend strength (ADX)
        adx = ta.adx(high, low, close, length=14)
        trend_strength = adx['ADX_14'].iloc[-1] if adx is not None else 0

        # Moving averages
        sma_20 = ta.sma(close, length=20)
        sma_50 = ta.sma(close, length=50)

        above_sma_20 = close.iloc[-1] > sma_20.iloc[-1] if sma_20 is not None else False
        above_sma_50 = close.iloc[-1] > sma_50.iloc[-1] if sma_50 is not None else False

        # Breakout detection
        high_52w = close.rolling(window=min(252, len(close))).max().iloc[-1]
        breakout_distance = ((current_price / high_52w) - 1) * 100

        # Quality score
        quality_score = self._calculate_quality_score(
            momentum_score, current_rsi, volume_ratio, trend_strength, atr_pct
        )

        # Oversold score
        oversold_score = self._calculate_oversold_score(current_rsi, close)

        return {
            "Symbol": symbol,
            "Price": round(current_price, 2),
            "Return": round(period_return, 2),
            "Momentum": round(momentum_score, 2),
            "RSI": round(current_rsi, 2),
            "Volume_Ratio": round(volume_ratio, 2),
            "ATR%": round(atr_pct, 2),
            "Trend_Strength": round(trend_strength, 2),
            "Above_SMA20": above_sma_20,
            "Above_SMA50": above_sma_50,
            "Breakout%": round(breakout_distance, 2),
            "Quality_Score": round(quality_score, 2),
            "Oversold_Score": round(oversold_score, 2)
        }

    def _calculate_quality_score(self, momentum, rsi, volume_ratio, trend_strength, atr_pct):
        """Calculate overall quality score"""
        score = 50.0

        # Momentum component
        score += min(max(momentum, -20), 20)

        # RSI component (prefer 45-65)
        if 45 <= rsi <= 65:
            score += 15
        elif rsi < 30:
            score += 10  # Oversold opportunity
        elif rsi > 70:
            score -= 10

        # Volume component
        if volume_ratio > 1.5:
            score += 10

        # Trend strength
        if trend_strength > 25:
            score += 10

        # Lower volatility is better
        if atr_pct < 2:
            score += 10
        elif atr_pct > 5:
            score -= 10

        return max(0, min(100, score))

    def _calculate_oversold_score(self, rsi, close):
        """Calculate oversold opportunity score"""
        score = 0

        # RSI oversold
        if rsi < 30:
            score += 40
        elif rsi < 35:
            score += 25
        elif rsi < 40:
            score += 15

        # Z-score
        mean = close.rolling(window=20).mean().iloc[-1]
        std = close.rolling(window=20).std().iloc[-1]
        if std > 0:
            z_score = (close.iloc[-1] - mean) / std
            if z_score < -2:
                score += 40
            elif z_score < -1.5:
                score += 25

        return min(100, score)

    def _generate_rankings(self, results: List[Dict], method: str, top_n: int) -> Dict:
        """Generate rankings by specified method"""
        df = pd.DataFrame(results)

        rankings = {}

        if method in ["all", "gainers"]:
            top_gainers = df.nlargest(top_n, 'Return')
            rankings["top_gainers"] = top_gainers[
                ['Symbol', 'Price', 'Return', 'Volume_Ratio', 'Momentum']
            ]

        if method in ["all", "losers"]:
            top_losers = df.nsmallest(top_n, 'Return')
            rankings["top_losers"] = top_losers[
                ['Symbol', 'Price', 'Return', 'Volume_Ratio', 'Momentum']
            ]

        if method in ["all", "momentum"]:
            top_momentum = df.nlargest(top_n, 'Momentum')
            rankings["top_momentum"] = top_momentum[
                ['Symbol', 'Price', 'Momentum', 'Trend_Strength', 'RSI']
            ]

        if method in ["all", "breakout"]:
            near_breakout = df[df['Breakout%'] > -5].nlargest(top_n, 'Breakout%')
            rankings["near_breakout"] = near_breakout[
                ['Symbol', 'Price', 'Breakout%', 'Volume_Ratio', 'Trend_Strength']
            ]

        if method in ["all", "oversold"]:
            oversold = df.nlargest(top_n, 'Oversold_Score')
            rankings["oversold_opportunities"] = oversold[
                ['Symbol', 'Price', 'RSI', 'Oversold_Score', 'Return']
            ]

        if method in ["all", "quality"]:
            high_quality = df.nlargest(top_n, 'Quality_Score')
            rankings["high_quality_stocks"] = high_quality[
                ['Symbol', 'Price', 'Quality_Score', 'Momentum', 'Return']
            ]

        if method in ["all", "volume"]:
            high_volume = df.nlargest(top_n, 'Volume_Ratio')
            rankings["high_volume"] = high_volume[
                ['Symbol', 'Price', 'Volume_Ratio', 'Return', 'Momentum']
            ]

        # Summary statistics
        rankings["summary"] = {
            "avg_return": round(df['Return'].mean(), 2),
            "positive_stocks": len(df[df['Return'] > 0]),
            "negative_stocks": len(df[df['Return'] < 0]),
            "avg_momentum": round(df['Momentum'].mean(), 2),
            "avg_rsi": round(df['RSI'].mean(), 2),
            "high_quality_count": len(df[df['Quality_Score'] > 70])
        }

        return rankings
