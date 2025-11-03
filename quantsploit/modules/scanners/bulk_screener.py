"""
Advanced Bulk Stock Screener with Parallel Processing
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher


class BulkScreener(BaseModule):
    """
    High-performance bulk stock screener with advanced filters and parallel processing
    """

    @property
    def name(self) -> str:
        return "Advanced Bulk Screener"

    @property
    def description(self) -> str:
        return "Analyze large numbers of stocks in parallel with advanced quantitative filters"

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
                "description": "Comma-separated symbols or 'SP500', 'NASDAQ100', 'DOW30'"
            },
            "MIN_PRICE": {
                "value": 5.0,
                "required": False,
                "description": "Minimum stock price"
            },
            "MAX_PRICE": {
                "value": 10000.0,
                "required": False,
                "description": "Maximum stock price"
            },
            "MIN_VOLUME": {
                "value": 500000,
                "required": False,
                "description": "Minimum average daily volume"
            },
            "RSI_MIN": {
                "value": None,
                "required": False,
                "description": "Minimum RSI (e.g., 30 for oversold)"
            },
            "RSI_MAX": {
                "value": None,
                "required": False,
                "description": "Maximum RSI (e.g., 70 for overbought)"
            },
            "TREND_FILTER": {
                "value": None,
                "required": False,
                "description": "Trend filter: 'uptrend', 'downtrend', or None"
            },
            "MAX_WORKERS": {
                "value": 10,
                "required": False,
                "description": "Number of parallel workers"
            },
            "SORT_BY": {
                "value": "score",
                "required": False,
                "description": "Sort by: score, volume, price_change, rsi, momentum"
            },
        })
        self.options["SYMBOL"]["required"] = False

    def run(self) -> Dict[str, Any]:
        """Execute bulk screening"""
        symbols_input = self.get_option("SYMBOLS")
        period = self.get_option("PERIOD")
        interval = self.get_option("INTERVAL")
        max_workers = int(self.get_option("MAX_WORKERS"))
        sort_by = self.get_option("SORT_BY")

        # Get symbol list
        symbols = self._parse_symbols(symbols_input)

        self.log(f"Screening {len(symbols)} stocks in parallel...")

        # Screen stocks in parallel
        fetcher = DataFetcher(self.framework.database)
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._analyze_stock, symbol, fetcher, period, interval): symbol
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

        # Filter results
        filtered_results = self._apply_filters(results)

        # Sort results
        sorted_results = self._sort_results(filtered_results, sort_by)

        # Create DataFrame
        if sorted_results:
            df = pd.DataFrame(sorted_results)
        else:
            df = pd.DataFrame()

        return {
            "total_analyzed": len(symbols),
            "passed_filters": len(sorted_results),
            "filter_rate": f"{(len(sorted_results)/len(symbols)*100):.1f}%" if symbols else "0%",
            "results": df,
            "top_10": df.head(10) if not df.empty else pd.DataFrame()
        }

    def _parse_symbols(self, symbols_input: str) -> List[str]:
        """Parse symbol input including predefined lists"""
        if symbols_input.upper() == "SP500":
            return self._get_sp500_symbols()
        elif symbols_input.upper() == "NASDAQ100":
            return self._get_nasdaq100_symbols()
        elif symbols_input.upper() == "DOW30":
            return self._get_dow30_symbols()
        else:
            return [s.strip().upper() for s in symbols_input.split(",")]

    def _get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols (subset for demo)"""
        # In production, you'd fetch from Wikipedia or other source
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
                'UNH', 'JNJ', 'V', 'XOM', 'WMT', 'JPM', 'PG', 'MA', 'CVX', 'HD',
                'LLY', 'ABBV', 'MRK', 'AVGO', 'KO', 'PEP', 'COST', 'ADBE', 'TMO',
                'MCD', 'ACN', 'CSCO', 'ABT', 'DHR', 'NKE', 'NFLX', 'TXN', 'DIS',
                'CRM', 'VZ', 'CMCSA', 'ORCL', 'WFC', 'AMD', 'INTC', 'BAC', 'PM']

    def _get_nasdaq100_symbols(self) -> List[str]:
        """Get NASDAQ 100 symbols (subset)"""
        return ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
                'AVGO', 'COST', 'ASML', 'NFLX', 'AMD', 'PEP', 'ADBE', 'CSCO',
                'TMUS', 'CMCSA', 'INTC', 'TXN', 'QCOM', 'INTU', 'HON', 'AMAT']

    def _get_dow30_symbols(self) -> List[str]:
        """Get Dow 30 symbols"""
        return ['AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
                'DOW', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO',
                'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V',
                'VZ', 'WBA', 'WMT']

    def _analyze_stock(self, symbol: str, fetcher: DataFetcher,
                      period: str, interval: str) -> Dict[str, Any]:
        """Analyze a single stock"""
        df = fetcher.get_stock_data(symbol, period, interval)

        if df is None or df.empty or len(df) < 50:
            return None

        # Calculate indicators
        close = df['Close']

        # Price metrics
        current_price = close.iloc[-1]
        price_change = close.iloc[-1] - close.iloc[-2]
        price_change_pct = (price_change / close.iloc[-2]) * 100

        # Volume
        avg_volume = df['Volume'].mean()
        volume_ratio = df['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 0

        # RSI
        rsi = ta.rsi(close, length=14)
        current_rsi = rsi.iloc[-1] if rsi is not None else None

        # Moving averages
        sma_20 = ta.sma(close, length=20)
        sma_50 = ta.sma(close, length=50)
        ema_12 = ta.ema(close, length=12)

        # MACD
        macd = ta.macd(close)
        macd_line = macd['MACD_12_26_9'].iloc[-1] if macd is not None else None
        macd_signal = macd['MACDs_12_26_9'].iloc[-1] if macd is not None else None

        # ATR for volatility
        atr = ta.atr(df['High'], df['Low'], close, length=14)
        current_atr = atr.iloc[-1] if atr is not None else None

        # Momentum (rate of change)
        roc = ta.roc(close, length=10)
        momentum = roc.iloc[-1] if roc is not None else None

        # Trend determination
        trend = "neutral"
        if sma_20 is not None and sma_50 is not None:
            if not pd.isna(sma_20.iloc[-1]) and not pd.isna(sma_50.iloc[-1]):
                if sma_20.iloc[-1] > sma_50.iloc[-1] and close.iloc[-1] > sma_20.iloc[-1]:
                    trend = "uptrend"
                elif sma_20.iloc[-1] < sma_50.iloc[-1] and close.iloc[-1] < sma_20.iloc[-1]:
                    trend = "downtrend"

        # Calculate composite score
        score = self._calculate_score(
            current_rsi, momentum, volume_ratio, trend,
            current_price, sma_20.iloc[-1] if sma_20 is not None else None
        )

        return {
            "Symbol": symbol,
            "Price": round(current_price, 2),
            "Change%": round(price_change_pct, 2),
            "Volume": int(avg_volume),
            "Vol_Ratio": round(volume_ratio, 2),
            "RSI": round(current_rsi, 2) if current_rsi else None,
            "Momentum": round(momentum, 2) if momentum else None,
            "Trend": trend,
            "MACD": "Bull" if macd_line and macd_signal and macd_line > macd_signal else "Bear",
            "ATR": round(current_atr, 2) if current_atr else None,
            "Score": round(score, 2)
        }

    def _calculate_score(self, rsi, momentum, volume_ratio, trend, price, sma_20) -> float:
        """Calculate composite score for ranking"""
        score = 50.0  # Base score

        # RSI component (prefer 30-70 range)
        if rsi:
            if 30 <= rsi <= 70:
                score += 10
            elif rsi < 30:
                score += 15  # Oversold bonus
            elif rsi > 70:
                score -= 10  # Overbought penalty

        # Momentum component
        if momentum:
            score += min(momentum, 20)  # Cap at 20

        # Volume component
        if volume_ratio > 1.5:
            score += 10
        elif volume_ratio > 2.0:
            score += 20

        # Trend component
        if trend == "uptrend":
            score += 15
        elif trend == "downtrend":
            score -= 10

        # Price vs SMA
        if sma_20 and price:
            if price > sma_20:
                score += 10

        return max(0, min(100, score))  # Clamp to 0-100

    def _apply_filters(self, results: List[Dict]) -> List[Dict]:
        """Apply user-defined filters"""
        filtered = results

        min_price = float(self.get_option("MIN_PRICE"))
        max_price = float(self.get_option("MAX_PRICE"))
        min_volume = float(self.get_option("MIN_VOLUME"))
        rsi_min = self.get_option("RSI_MIN")
        rsi_max = self.get_option("RSI_MAX")
        trend_filter = self.get_option("TREND_FILTER")

        # Price filter
        filtered = [r for r in filtered if min_price <= r['Price'] <= max_price]

        # Volume filter
        filtered = [r for r in filtered if r['Volume'] >= min_volume]

        # RSI filters
        if rsi_min:
            filtered = [r for r in filtered if r['RSI'] and r['RSI'] >= float(rsi_min)]
        if rsi_max:
            filtered = [r for r in filtered if r['RSI'] and r['RSI'] <= float(rsi_max)]

        # Trend filter
        if trend_filter:
            filtered = [r for r in filtered if r['Trend'] == trend_filter.lower()]

        return filtered

    def _sort_results(self, results: List[Dict], sort_by: str) -> List[Dict]:
        """Sort results by specified column"""
        sort_map = {
            "score": "Score",
            "volume": "Volume",
            "price_change": "Change%",
            "rsi": "RSI",
            "momentum": "Momentum"
        }

        sort_column = sort_map.get(sort_by, "Score")

        return sorted(results, key=lambda x: x.get(sort_column, 0) or 0, reverse=True)
