"""
Technical Analysis Indicators Module
"""

import pandas as pd
import pandas_ta as ta
from typing import Dict, Any
from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher


class TechnicalIndicators(BaseModule):
    """
    Calculate technical indicators for a stock
    """

    @property
    def name(self) -> str:
        return "Technical Indicators"

    @property
    def description(self) -> str:
        return "Calculate common technical indicators (RSI, MACD, SMA, EMA, Bollinger Bands)"

    @property
    def author(self) -> str:
        return "Quantsploit Team"

    @property
    def category(self) -> str:
        return "analysis"

    def _init_options(self):
        super()._init_options()
        self.options.update({
            "INDICATORS": {
                "value": "RSI,MACD,SMA,EMA,BBANDS",
                "required": False,
                "description": "Comma-separated list of indicators to calculate"
            },
            "RSI_PERIOD": {
                "value": 14,
                "required": False,
                "description": "RSI period"
            },
            "SMA_PERIOD": {
                "value": 20,
                "required": False,
                "description": "SMA period"
            },
            "EMA_PERIOD": {
                "value": 12,
                "required": False,
                "description": "EMA period"
            },
        })

    def run(self) -> Dict[str, Any]:
        """Execute technical analysis"""
        symbol = self.get_option("SYMBOL")
        period = self.get_option("PERIOD")
        interval = self.get_option("INTERVAL")
        indicators_str = self.get_option("INDICATORS")

        # Fetch data
        fetcher = DataFetcher(self.framework.database)
        df = fetcher.get_stock_data(symbol, period, interval)

        if df is None or df.empty:
            return {"success": False, "error": f"Failed to fetch data for {symbol}"}

        indicators = [i.strip() for i in indicators_str.split(",")]
        results = {"symbol": symbol, "latest_price": df['Close'].iloc[-1]}

        # Calculate requested indicators
        for indicator in indicators:
            indicator = indicator.upper()

            if indicator == "RSI":
                rsi_period = int(self.get_option("RSI_PERIOD"))
                df['RSI'] = ta.rsi(df['Close'], length=rsi_period)
                results['RSI'] = df['RSI'].iloc[-1]
                results['RSI_signal'] = self._interpret_rsi(df['RSI'].iloc[-1])

            elif indicator == "MACD":
                macd = ta.macd(df['Close'])
                if macd is not None:
                    df = pd.concat([df, macd], axis=1)
                    results['MACD'] = df['MACD_12_26_9'].iloc[-1]
                    results['MACD_signal'] = df['MACDs_12_26_9'].iloc[-1]
                    results['MACD_hist'] = df['MACDh_12_26_9'].iloc[-1]
                    results['MACD_interpretation'] = self._interpret_macd(
                        df['MACD_12_26_9'].iloc[-1],
                        df['MACDs_12_26_9'].iloc[-1]
                    )

            elif indicator == "SMA":
                sma_period = int(self.get_option("SMA_PERIOD"))
                df[f'SMA_{sma_period}'] = ta.sma(df['Close'], length=sma_period)
                results[f'SMA_{sma_period}'] = df[f'SMA_{sma_period}'].iloc[-1]
                results['Price_vs_SMA'] = self._compare_to_ma(
                    df['Close'].iloc[-1],
                    df[f'SMA_{sma_period}'].iloc[-1]
                )

            elif indicator == "EMA":
                ema_period = int(self.get_option("EMA_PERIOD"))
                df[f'EMA_{ema_period}'] = ta.ema(df['Close'], length=ema_period)
                results[f'EMA_{ema_period}'] = df[f'EMA_{ema_period}'].iloc[-1]

            elif indicator == "BBANDS":
                bbands = ta.bbands(df['Close'])
                if bbands is not None:
                    df = pd.concat([df, bbands], axis=1)
                    results['BB_upper'] = df['BBU_5_2.0'].iloc[-1]
                    results['BB_middle'] = df['BBM_5_2.0'].iloc[-1]
                    results['BB_lower'] = df['BBL_5_2.0'].iloc[-1]
                    results['BB_position'] = self._interpret_bbands(
                        df['Close'].iloc[-1],
                        df['BBU_5_2.0'].iloc[-1],
                        df['BBL_5_2.0'].iloc[-1]
                    )

        # Add price change data
        results['price_change'] = df['Close'].iloc[-1] - df['Close'].iloc[-2]
        results['price_change_pct'] = ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100
        results['volume'] = df['Volume'].iloc[-1]

        # Add recent data
        results['recent_data'] = df.tail(10)

        return results

    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI value"""
        if rsi > 70:
            return "Overbought"
        elif rsi < 30:
            return "Oversold"
        else:
            return "Neutral"

    def _interpret_macd(self, macd: float, signal: float) -> str:
        """Interpret MACD"""
        if macd > signal:
            return "Bullish"
        elif macd < signal:
            return "Bearish"
        else:
            return "Neutral"

    def _compare_to_ma(self, price: float, ma: float) -> str:
        """Compare price to moving average"""
        diff_pct = ((price / ma) - 1) * 100
        if diff_pct > 2:
            return f"Above MA by {diff_pct:.2f}%"
        elif diff_pct < -2:
            return f"Below MA by {abs(diff_pct):.2f}%"
        else:
            return "Near MA"

    def _interpret_bbands(self, price: float, upper: float, lower: float) -> str:
        """Interpret Bollinger Bands position"""
        if price > upper:
            return "Above upper band (overbought)"
        elif price < lower:
            return "Below lower band (oversold)"
        else:
            position = (price - lower) / (upper - lower) * 100
            return f"{position:.1f}% within bands"
