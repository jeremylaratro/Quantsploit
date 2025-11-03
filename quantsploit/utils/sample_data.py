"""
Demo/Sample Data Generator for Testing
When live data is unavailable, generate realistic sample data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


class SampleDataGenerator:
    """Generate realistic sample market data for testing"""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate_stock_data(self, symbol: str, period: str = "1y",
                           interval: str = "1d") -> pd.DataFrame:
        """Generate sample OHLCV data"""

        # Calculate number of data points
        periods_map = {
            "1d": 1, "5d": 5, "1mo": 21, "3mo": 63,
            "6mo": 126, "1y": 252, "2y": 504, "5y": 1260
        }

        num_periods = periods_map.get(period, 252)

        # Starting parameters based on symbol
        base_price = self._get_base_price(symbol)
        volatility = 0.02  # 2% daily volatility

        # Generate dates
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=num_periods, freq='D')

        # Generate price series using geometric brownian motion
        returns = np.random.normal(0.0005, volatility, num_periods)
        price_series = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        data = {
            'Open': [],
            'High': [],
            'Low': [],
            'Close': [],
            'Volume': []
        }

        for price in price_series:
            # Daily range
            daily_range = price * np.random.uniform(0.01, 0.03)

            open_price = price * np.random.uniform(0.995, 1.005)
            close_price = price
            high_price = max(open_price, close_price) + daily_range * 0.5
            low_price = min(open_price, close_price) - daily_range * 0.5

            volume = int(np.random.uniform(50_000_000, 150_000_000))

            data['Open'].append(open_price)
            data['High'].append(high_price)
            data['Low'].append(low_price)
            data['Close'].append(close_price)
            data['Volume'].append(volume)

        df = pd.DataFrame(data, index=dates)
        df.index.name = 'Date'

        return df

    def _get_base_price(self, symbol: str) -> float:
        """Get base price for symbol"""
        symbol_prices = {
            'AAPL': 180.0,
            'MSFT': 380.0,
            'GOOGL': 140.0,
            'AMZN': 170.0,
            'NVDA': 480.0,
            'META': 480.0,
            'TSLA': 250.0,
            'AMD': 160.0,
            'NFLX': 580.0,
            'SPY': 460.0,
            'QQQ': 390.0,
        }

        # Default price if symbol not found
        return symbol_prices.get(symbol.upper(), 100.0)

    def generate_stock_info(self, symbol: str) -> dict:
        """Generate sample stock info"""
        base_price = self._get_base_price(symbol)

        return {
            'symbol': symbol,
            'currentPrice': base_price,
            'regularMarketPrice': base_price,
            'regularMarketChange': np.random.uniform(-5, 5),
            'regularMarketChangePercent': np.random.uniform(-2, 2),
            'volume': int(np.random.uniform(50_000_000, 150_000_000)),
            'marketCap': int(base_price * 1_000_000_000),
            'trailingPE': np.random.uniform(15, 35),
            'dayHigh': base_price * 1.02,
            'dayLow': base_price * 0.98,
            'fiftyTwoWeekHigh': base_price * 1.3,
            'fiftyTwoWeekLow': base_price * 0.7,
            'longName': f'{symbol} Corporation',
            'quoteType': 'EQUITY'
        }

    def generate_options_chain(self, symbol: str, expiration: Optional[str] = None) -> dict:
        """Generate sample options chain"""
        base_price = self._get_base_price(symbol)

        # Generate expiration dates
        today = datetime.now()
        expirations = [
            (today + timedelta(days=7)).strftime('%Y-%m-%d'),
            (today + timedelta(days=14)).strftime('%Y-%m-%d'),
            (today + timedelta(days=30)).strftime('%Y-%m-%d'),
            (today + timedelta(days=60)).strftime('%Y-%m-%d'),
        ]

        if expiration is None:
            expiration = expirations[0]

        # Generate strikes around current price
        strikes = [base_price * (1 + i * 0.05) for i in range(-5, 6)]

        calls_data = []
        puts_data = []

        for strike in strikes:
            # Call options
            calls_data.append({
                'strike': strike,
                'lastPrice': max(0.01, base_price - strike + np.random.uniform(-2, 2)),
                'bid': max(0.01, base_price - strike),
                'ask': max(0.02, base_price - strike + 0.5),
                'volume': int(np.random.uniform(0, 10000)),
                'openInterest': int(np.random.uniform(100, 50000)),
                'impliedVolatility': np.random.uniform(0.2, 0.6)
            })

            # Put options
            puts_data.append({
                'strike': strike,
                'lastPrice': max(0.01, strike - base_price + np.random.uniform(-2, 2)),
                'bid': max(0.01, strike - base_price),
                'ask': max(0.02, strike - base_price + 0.5),
                'volume': int(np.random.uniform(0, 10000)),
                'openInterest': int(np.random.uniform(100, 50000)),
                'impliedVolatility': np.random.uniform(0.2, 0.6)
            })

        return {
            'expiration': expiration,
            'calls': pd.DataFrame(calls_data),
            'puts': pd.DataFrame(puts_data),
            'available_expirations': expirations
        }


# Singleton instance
_sample_generator = SampleDataGenerator()

def get_sample_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Get sample stock data"""
    return _sample_generator.generate_stock_data(symbol, period, interval)

def get_sample_info(symbol: str) -> dict:
    """Get sample stock info"""
    return _sample_generator.generate_stock_info(symbol)

def get_sample_options(symbol: str, expiration: Optional[str] = None) -> dict:
    """Get sample options chain"""
    return _sample_generator.generate_options_chain(symbol, expiration)
