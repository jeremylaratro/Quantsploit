"""
Market data fetcher with caching support
"""

import yfinance as yf
import pandas as pd
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta


class DataFetcher:
    """
    Fetches market data from various sources with caching
    """

    def __init__(self, database=None, cache_enabled=True, cache_duration=3600):
        self.database = database
        self.cache_enabled = cache_enabled
        self.cache_duration = cache_duration

    def get_stock_data(self, symbol: str, period: str = "1y",
                       interval: str = "1d", force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch stock price data

        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            force_refresh: Skip cache and fetch fresh data

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        if self.cache_enabled and not force_refresh and self.database:
            cached = self.database.get_cached_data(symbol, period, interval, self.cache_duration)
            if cached:
                return pd.read_json(cached)

        # Fetch fresh data
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                return None

            # Cache the data
            if self.cache_enabled and self.database:
                self.database.cache_market_data(
                    symbol, period, interval, df.to_json()
                )

            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed stock information

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with stock info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except Exception as e:
            print(f"Error fetching info for {symbol}: {str(e)}")
            return None

    def get_options_chain(self, symbol: str, expiration: Optional[str] = None) -> Optional[Dict]:
        """
        Get options chain for a symbol

        Args:
            symbol: Stock ticker symbol
            expiration: Expiration date (YYYY-MM-DD), None for nearest

        Returns:
            Dictionary with calls and puts DataFrames
        """
        try:
            ticker = yf.Ticker(symbol)

            if expiration is None:
                expirations = ticker.options
                if not expirations:
                    return None
                expiration = expirations[0]

            opt_chain = ticker.option_chain(expiration)

            return {
                "expiration": expiration,
                "calls": opt_chain.calls,
                "puts": opt_chain.puts,
                "available_expirations": ticker.options
            }

        except Exception as e:
            print(f"Error fetching options for {symbol}: {str(e)}")
            return None

    def get_multiple_stocks(self, symbols: list, period: str = "1y",
                          interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks

        Args:
            symbols: List of ticker symbols
            period: Data period
            interval: Data interval

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        for symbol in symbols:
            df = self.get_stock_data(symbol, period, interval)
            if df is not None:
                results[symbol] = df
        return results

    def get_realtime_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote data

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with current price info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            quote = {
                "symbol": symbol,
                "price": info.get("currentPrice", info.get("regularMarketPrice")),
                "change": info.get("regularMarketChange"),
                "change_percent": info.get("regularMarketChangePercent"),
                "volume": info.get("volume"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "day_high": info.get("dayHigh"),
                "day_low": info.get("dayLow"),
                "year_high": info.get("fiftyTwoWeekHigh"),
                "year_low": info.get("fiftyTwoWeekLow"),
            }

            return quote

        except Exception as e:
            print(f"Error fetching quote for {symbol}: {str(e)}")
            return None

    def search_symbols(self, query: str) -> list:
        """
        Search for stock symbols (basic implementation)

        Args:
            query: Search query

        Returns:
            List of matching symbols
        """
        # Note: This is a basic implementation
        # For production, you'd want to use a proper symbol search API
        try:
            ticker = yf.Ticker(query.upper())
            info = ticker.info
            if info and 'symbol' in info:
                return [{
                    "symbol": info.get('symbol'),
                    "name": info.get('longName', 'N/A'),
                    "type": info.get('quoteType', 'N/A')
                }]
        except:
            pass
        return []
