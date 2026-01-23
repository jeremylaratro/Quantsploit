"""
Pytest configuration and fixtures for Quantsploit tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')

    # Generate realistic price movements
    initial_price = 100.0
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = initial_price * np.cumprod(1 + returns)

    # Generate OHLCV
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'High': prices * (1 + np.abs(np.random.uniform(0, 0.02, len(dates)))),
        'Low': prices * (1 - np.abs(np.random.uniform(0, 0.02, len(dates)))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    # Ensure High >= Open, Close and Low <= Open, Close
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

    return data


@pytest.fixture
def sample_ohlcv_bullish():
    """Generate sample OHLCV data with bullish trend"""
    np.random.seed(123)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')

    # Generate bullish price movements (positive drift)
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.015, len(dates))  # Positive mean
    prices = initial_price * np.cumprod(1 + returns)

    data = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
        'High': prices * (1 + np.abs(np.random.uniform(0, 0.015, len(dates)))),
        'Low': prices * (1 - np.abs(np.random.uniform(0, 0.015, len(dates)))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

    return data


@pytest.fixture
def sample_ohlcv_bearish():
    """Generate sample OHLCV data with bearish trend"""
    np.random.seed(456)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')

    # Generate bearish price movements (negative drift)
    initial_price = 100.0
    returns = np.random.normal(-0.001, 0.02, len(dates))  # Negative mean
    prices = initial_price * np.cumprod(1 + returns)

    data = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
        'High': prices * (1 + np.abs(np.random.uniform(0, 0.02, len(dates)))),
        'Low': prices * (1 - np.abs(np.random.uniform(0, 0.02, len(dates)))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

    return data


@pytest.fixture
def sample_ohlcv_volatile():
    """Generate highly volatile sample OHLCV data"""
    np.random.seed(789)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')

    # Generate volatile price movements
    initial_price = 100.0
    returns = np.random.normal(0, 0.04, len(dates))  # High volatility
    prices = initial_price * np.cumprod(1 + returns)

    data = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.02, 0.02, len(dates))),
        'High': prices * (1 + np.abs(np.random.uniform(0, 0.04, len(dates)))),
        'Low': prices * (1 - np.abs(np.random.uniform(0, 0.04, len(dates)))),
        'Close': prices,
        'Volume': np.random.randint(5000000, 20000000, len(dates))
    }, index=dates)

    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

    return data


@pytest.fixture
def empty_dataframe():
    """Return an empty DataFrame for edge case testing"""
    return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])


@pytest.fixture
def single_row_dataframe():
    """Return a single-row DataFrame for edge case testing"""
    return pd.DataFrame({
        'Open': [100.0],
        'High': [101.0],
        'Low': [99.0],
        'Close': [100.5],
        'Volume': [1000000]
    }, index=[datetime(2023, 1, 1)])


@pytest.fixture
def backtest_config():
    """Standard backtest configuration for testing"""
    from quantsploit.utils.backtesting import BacktestConfig
    return BacktestConfig(
        initial_capital=100000.0,
        commission_pct=0.001,
        commission_min=1.0,
        slippage_pct=0.001,
        position_size=1.0,
        max_positions=1,
        margin_requirement=1.0,
        risk_free_rate=0.02
    )


@pytest.fixture
def backtester(backtest_config):
    """Configured Backtester instance"""
    from quantsploit.utils.backtesting import Backtester
    return Backtester(backtest_config)


# Utility functions for tests
def assert_valid_ohlcv(df):
    """Assert that a DataFrame is valid OHLCV data"""
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"

    # High should be >= max(Open, Close)
    assert (df['High'] >= df[['Open', 'Close']].max(axis=1)).all(), "High < max(Open, Close)"

    # Low should be <= min(Open, Close)
    assert (df['Low'] <= df[['Open', 'Close']].min(axis=1)).all(), "Low > min(Open, Close)"

    # Volume should be non-negative
    assert (df['Volume'] >= 0).all(), "Negative volume"


def assert_valid_backtest_results(results):
    """Assert that BacktestResults are valid"""
    from quantsploit.utils.backtesting import BacktestResults

    assert isinstance(results, BacktestResults)

    # Check key metrics exist and are reasonable
    assert not np.isnan(results.total_return) or results.total_trades == 0
    assert results.total_trades >= 0
    assert 0 <= results.win_rate <= 100 or results.total_trades == 0
    assert results.max_drawdown >= 0
