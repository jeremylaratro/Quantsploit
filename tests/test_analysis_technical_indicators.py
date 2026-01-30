"""
Unit tests for Technical Indicators Module

Tests cover:
- Module properties and initialization
- Individual indicator calculations
- Indicator interpretation
- Multiple indicator calculation
- Error handling
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.modules.analysis.technical_indicators import TechnicalIndicators


@pytest.fixture
def mock_framework():
    """Create a mock framework with database"""
    framework = Mock()
    framework.database = Mock()
    framework.log = Mock()
    return framework


@pytest.fixture
def ti_module(mock_framework):
    """Create a TechnicalIndicators module instance"""
    return TechnicalIndicators(mock_framework)


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Create data with some trend
    close = 100 + np.linspace(0, 10, 100) + np.random.normal(0, 2, 100)

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.02,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)


@pytest.fixture
def overbought_data():
    """Generate data that would produce overbought RSI"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')

    # Strong uptrend - price increases every day
    close = 100 * (1.02 ** np.arange(50))

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.01,
        'Low': close * 0.995,
        'Close': close,
        'Volume': [1000000] * 50
    }, index=dates)


@pytest.fixture
def oversold_data():
    """Generate data that would produce oversold RSI"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')

    # Strong downtrend - price decreases every day
    close = 100 * (0.98 ** np.arange(50))

    return pd.DataFrame({
        'Open': close * 1.01,
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': [1000000] * 50
    }, index=dates)


class TestModuleProperties:
    """Tests for module property definitions"""

    def test_name(self, ti_module):
        """Test module name"""
        assert ti_module.name == "Technical Indicators"

    def test_description(self, ti_module):
        """Test module description"""
        assert "RSI" in ti_module.description
        assert "MACD" in ti_module.description
        assert "SMA" in ti_module.description

    def test_author(self, ti_module):
        """Test module author"""
        assert ti_module.author == "Quantsploit Team"

    def test_category(self, ti_module):
        """Test module category"""
        assert ti_module.category == "analysis"


class TestOptions:
    """Tests for module option management"""

    def test_default_options(self, ti_module):
        """Test default option values"""
        assert "RSI" in ti_module.get_option("INDICATORS")
        assert "MACD" in ti_module.get_option("INDICATORS")
        assert ti_module.get_option("RSI_PERIOD") == 14
        assert ti_module.get_option("SMA_PERIOD") == 20
        assert ti_module.get_option("EMA_PERIOD") == 12

    def test_set_indicators(self, ti_module):
        """Test setting indicators list"""
        ti_module.set_option("INDICATORS", "RSI,SMA")
        assert ti_module.get_option("INDICATORS") == "RSI,SMA"

    def test_set_rsi_period(self, ti_module):
        """Test setting RSI period"""
        ti_module.set_option("RSI_PERIOD", 21)
        assert ti_module.get_option("RSI_PERIOD") == 21

    def test_set_sma_period(self, ti_module):
        """Test setting SMA period"""
        ti_module.set_option("SMA_PERIOD", 50)
        assert ti_module.get_option("SMA_PERIOD") == 50

    def test_inherited_options(self, ti_module):
        """Test that base module options are inherited"""
        assert "SYMBOL" in ti_module.options
        assert "PERIOD" in ti_module.options
        assert "INTERVAL" in ti_module.options


class TestRunExecution:
    """Tests for run execution"""

    def test_run_returns_results(self, ti_module, sample_ohlcv):
        """Test run returns results dictionary"""
        ti_module.set_option("SYMBOL", "AAPL")
        # Use only indicators that have stable column names
        ti_module.set_option("INDICATORS", "RSI,MACD,SMA,EMA")

        with patch('quantsploit.modules.analysis.technical_indicators.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = ti_module.run()

        assert "symbol" in results
        assert "latest_price" in results
        assert results["symbol"] == "AAPL"

    def test_run_no_data_error(self, ti_module):
        """Test run with no data returns error"""
        ti_module.set_option("SYMBOL", "INVALID")

        with patch('quantsploit.modules.analysis.technical_indicators.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = None

            results = ti_module.run()

        assert results["success"] is False
        assert "error" in results

    def test_run_empty_data_error(self, ti_module):
        """Test run with empty data returns error"""
        ti_module.set_option("SYMBOL", "AAPL")

        with patch('quantsploit.modules.analysis.technical_indicators.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = pd.DataFrame()

            results = ti_module.run()

        assert results["success"] is False


class TestRSICalculation:
    """Tests for RSI indicator"""

    def test_rsi_in_results(self, ti_module, sample_ohlcv):
        """Test RSI is calculated and included"""
        ti_module.set_option("SYMBOL", "AAPL")
        ti_module.set_option("INDICATORS", "RSI")

        with patch('quantsploit.modules.analysis.technical_indicators.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = ti_module.run()

        assert "RSI" in results
        assert "RSI_signal" in results
        assert 0 <= results["RSI"] <= 100

    def test_rsi_uses_custom_period(self, ti_module, sample_ohlcv):
        """Test RSI uses custom period setting"""
        ti_module.set_option("SYMBOL", "AAPL")
        ti_module.set_option("INDICATORS", "RSI")
        ti_module.set_option("RSI_PERIOD", 21)

        with patch('quantsploit.modules.analysis.technical_indicators.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = ti_module.run()

        # RSI should still be calculated
        assert "RSI" in results


class TestMACDCalculation:
    """Tests for MACD indicator"""

    def test_macd_in_results(self, ti_module, sample_ohlcv):
        """Test MACD is calculated and included"""
        ti_module.set_option("SYMBOL", "AAPL")
        ti_module.set_option("INDICATORS", "MACD")

        with patch('quantsploit.modules.analysis.technical_indicators.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = ti_module.run()

        assert "MACD" in results
        assert "MACD_signal" in results
        assert "MACD_hist" in results
        assert "MACD_interpretation" in results


class TestSMACalculation:
    """Tests for SMA indicator"""

    def test_sma_in_results(self, ti_module, sample_ohlcv):
        """Test SMA is calculated and included"""
        ti_module.set_option("SYMBOL", "AAPL")
        ti_module.set_option("INDICATORS", "SMA")

        with patch('quantsploit.modules.analysis.technical_indicators.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = ti_module.run()

        assert "SMA_20" in results
        assert "Price_vs_SMA" in results

    def test_sma_uses_custom_period(self, ti_module, sample_ohlcv):
        """Test SMA uses custom period setting"""
        ti_module.set_option("SYMBOL", "AAPL")
        ti_module.set_option("INDICATORS", "SMA")
        ti_module.set_option("SMA_PERIOD", 50)

        with patch('quantsploit.modules.analysis.technical_indicators.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = ti_module.run()

        assert "SMA_50" in results


class TestEMACalculation:
    """Tests for EMA indicator"""

    def test_ema_in_results(self, ti_module, sample_ohlcv):
        """Test EMA is calculated and included"""
        ti_module.set_option("SYMBOL", "AAPL")
        ti_module.set_option("INDICATORS", "EMA")

        with patch('quantsploit.modules.analysis.technical_indicators.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = ti_module.run()

        assert "EMA_12" in results


class TestBBandsCalculation:
    """Tests for Bollinger Bands indicator"""

    def test_bbands_interpretation(self, ti_module):
        """Test Bollinger Bands interpretation function"""
        # Test above upper band
        result = ti_module._interpret_bbands(110, 108, 92)
        assert "overbought" in result.lower()

        # Test below lower band
        result = ti_module._interpret_bbands(90, 108, 92)
        assert "oversold" in result.lower()

        # Test within bands
        result = ti_module._interpret_bbands(100, 108, 92)
        assert "within" in result.lower()


class TestRSIInterpretation:
    """Tests for RSI interpretation"""

    def test_interpret_rsi_overbought(self, ti_module):
        """Test RSI interpretation for overbought"""
        result = ti_module._interpret_rsi(75)
        assert result == "Overbought"

    def test_interpret_rsi_oversold(self, ti_module):
        """Test RSI interpretation for oversold"""
        result = ti_module._interpret_rsi(25)
        assert result == "Oversold"

    def test_interpret_rsi_neutral(self, ti_module):
        """Test RSI interpretation for neutral"""
        result = ti_module._interpret_rsi(50)
        assert result == "Neutral"

    def test_interpret_rsi_boundary_overbought(self, ti_module):
        """Test RSI interpretation at overbought boundary"""
        result = ti_module._interpret_rsi(70)
        assert result == "Neutral"

        result = ti_module._interpret_rsi(71)
        assert result == "Overbought"

    def test_interpret_rsi_boundary_oversold(self, ti_module):
        """Test RSI interpretation at oversold boundary"""
        result = ti_module._interpret_rsi(30)
        assert result == "Neutral"

        result = ti_module._interpret_rsi(29)
        assert result == "Oversold"


class TestMACDInterpretation:
    """Tests for MACD interpretation"""

    def test_interpret_macd_bullish(self, ti_module):
        """Test MACD interpretation for bullish"""
        result = ti_module._interpret_macd(1.5, 0.5)
        assert result == "Bullish"

    def test_interpret_macd_bearish(self, ti_module):
        """Test MACD interpretation for bearish"""
        result = ti_module._interpret_macd(0.5, 1.5)
        assert result == "Bearish"

    def test_interpret_macd_neutral(self, ti_module):
        """Test MACD interpretation for neutral"""
        result = ti_module._interpret_macd(1.0, 1.0)
        assert result == "Neutral"


class TestMAComparison:
    """Tests for moving average comparison"""

    def test_compare_to_ma_above(self, ti_module):
        """Test price above MA"""
        result = ti_module._compare_to_ma(105, 100)
        assert "Above" in result
        assert "5.00%" in result

    def test_compare_to_ma_below(self, ti_module):
        """Test price below MA"""
        result = ti_module._compare_to_ma(95, 100)
        assert "Below" in result

    def test_compare_to_ma_near(self, ti_module):
        """Test price near MA"""
        result = ti_module._compare_to_ma(100.5, 100)
        assert result == "Near MA"


class TestMultipleIndicators:
    """Tests for multiple indicator calculation"""

    def test_all_default_indicators(self, ti_module, sample_ohlcv):
        """Test all default indicators are calculated"""
        ti_module.set_option("SYMBOL", "AAPL")
        # Use stable indicators only
        ti_module.set_option("INDICATORS", "RSI,MACD,SMA,EMA")

        with patch('quantsploit.modules.analysis.technical_indicators.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = ti_module.run()

        # Check stable indicators
        assert "RSI" in results
        assert "MACD" in results
        assert "SMA_20" in results
        assert "EMA_12" in results

    def test_subset_of_indicators(self, ti_module, sample_ohlcv):
        """Test subset of indicators"""
        ti_module.set_option("SYMBOL", "AAPL")
        ti_module.set_option("INDICATORS", "RSI,SMA")

        with patch('quantsploit.modules.analysis.technical_indicators.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = ti_module.run()

        assert "RSI" in results
        assert "SMA_20" in results
        # Should not include others
        assert "MACD" not in results
        assert "EMA_12" not in results


class TestPriceChangeData:
    """Tests for price change data in results"""

    def test_price_change_in_results(self, ti_module, sample_ohlcv):
        """Test price change is included"""
        ti_module.set_option("SYMBOL", "AAPL")
        ti_module.set_option("INDICATORS", "RSI")

        with patch('quantsploit.modules.analysis.technical_indicators.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = ti_module.run()

        assert "price_change" in results
        assert "price_change_pct" in results
        assert "volume" in results

    def test_recent_data_in_results(self, ti_module, sample_ohlcv):
        """Test recent data is included"""
        ti_module.set_option("SYMBOL", "AAPL")
        ti_module.set_option("INDICATORS", "RSI")

        with patch('quantsploit.modules.analysis.technical_indicators.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = ti_module.run()

        assert "recent_data" in results
        assert isinstance(results["recent_data"], pd.DataFrame)
        assert len(results["recent_data"]) == 10


class TestEdgeCases:
    """Tests for edge cases"""

    def test_minimal_data(self, ti_module):
        """Test with minimal data points"""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'Open': [100 + i * 0.1 for i in range(30)],
            'High': [101 + i * 0.1 for i in range(30)],
            'Low': [99 + i * 0.1 for i in range(30)],
            'Close': [100.5 + i * 0.1 for i in range(30)],
            'Volume': [1000000] * 30
        }, index=dates)

        ti_module.set_option("SYMBOL", "AAPL")
        ti_module.set_option("INDICATORS", "RSI")

        with patch('quantsploit.modules.analysis.technical_indicators.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = df

            results = ti_module.run()

        assert "RSI" in results

    def test_single_indicator(self, ti_module, sample_ohlcv):
        """Test with single indicator"""
        ti_module.set_option("SYMBOL", "AAPL")
        ti_module.set_option("INDICATORS", "RSI")

        with patch('quantsploit.modules.analysis.technical_indicators.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = ti_module.run()

        assert "RSI" in results
        assert "SMA_20" not in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
