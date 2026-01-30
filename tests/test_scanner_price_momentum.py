"""
Unit tests for Price Momentum Scanner Module

Tests cover:
- Module properties and initialization
- Option management
- Scanning multiple symbols
- Volume filtering
- Momentum detection
- Flag generation
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.modules.scanners.price_momentum import PriceMomentumScanner


@pytest.fixture
def mock_framework():
    """Create a mock framework with database"""
    framework = Mock()
    framework.database = Mock()
    framework.log = Mock()
    return framework


@pytest.fixture
def scanner_module(mock_framework):
    """Create a PriceMomentumScanner module instance"""
    return PriceMomentumScanner(mock_framework)


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Create trending data
    close = 100 + np.linspace(0, 20, 100) + np.random.normal(0, 1, 100)

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.02,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)


@pytest.fixture
def high_momentum_data():
    """Generate data with high momentum"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Strong uptrend
    close = 100 * (1.01 ** np.arange(100))

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.03,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(2000000, 5000000, 100)
    }, index=dates)


@pytest.fixture
def volume_spike_data():
    """Generate data with volume spike"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    close = np.full(100, 100.0) + np.random.normal(0, 1, 100)
    volume = np.full(100, 1000000)
    volume[-1] = 5000000  # Volume spike on last day

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': volume
    }, index=dates)


class TestModuleProperties:
    """Tests for module property definitions"""

    def test_name(self, scanner_module):
        """Test module name"""
        assert scanner_module.name == "Price Momentum Scanner"

    def test_description(self, scanner_module):
        """Test module description"""
        assert "momentum" in scanner_module.description.lower()
        assert "scan" in scanner_module.description.lower()

    def test_author(self, scanner_module):
        """Test module author"""
        assert scanner_module.author == "Quantsploit Team"

    def test_category(self, scanner_module):
        """Test module category"""
        assert scanner_module.category == "scanner"


class TestOptions:
    """Tests for module option management"""

    def test_default_options(self, scanner_module):
        """Test default option values"""
        assert scanner_module.get_option("SYMBOLS") is None
        assert scanner_module.get_option("MIN_VOLUME") == 1000000
        assert scanner_module.get_option("MIN_GAIN_PCT") == 5.0

    def test_set_symbols(self, scanner_module):
        """Test setting symbols"""
        scanner_module.set_option("SYMBOLS", "AAPL,MSFT,GOOGL")
        assert scanner_module.get_option("SYMBOLS") == "AAPL,MSFT,GOOGL"

    def test_set_min_volume(self, scanner_module):
        """Test setting minimum volume"""
        scanner_module.set_option("MIN_VOLUME", 500000)
        assert scanner_module.get_option("MIN_VOLUME") == 500000

    def test_set_min_gain_pct(self, scanner_module):
        """Test setting minimum gain percentage"""
        scanner_module.set_option("MIN_GAIN_PCT", 10.0)
        assert scanner_module.get_option("MIN_GAIN_PCT") == 10.0

    def test_inherited_options(self, scanner_module):
        """Test that base module options are inherited"""
        assert "PERIOD" in scanner_module.options
        assert "INTERVAL" in scanner_module.options


class TestRunExecution:
    """Tests for run execution"""

    def test_run_no_symbols_error(self, scanner_module):
        """Test run with no symbols returns error"""
        results = scanner_module.run()
        assert results["success"] is False
        assert "error" in results

    def test_run_returns_results(self, scanner_module, sample_ohlcv):
        """Test run returns results dictionary"""
        scanner_module.set_option("SYMBOLS", "AAPL,MSFT")

        with patch('quantsploit.modules.scanners.price_momentum.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = scanner_module.run()

        assert "scan_count" in results
        assert "results_found" in results
        assert "scan_results" in results

    def test_run_single_symbol(self, scanner_module, sample_ohlcv):
        """Test run with single symbol"""
        scanner_module.set_option("SYMBOLS", "AAPL")

        with patch('quantsploit.modules.scanners.price_momentum.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = scanner_module.run()

        assert results["scan_count"] == 1
        assert results["results_found"] == 1

    def test_run_multiple_symbols(self, scanner_module, sample_ohlcv):
        """Test run with multiple symbols"""
        scanner_module.set_option("SYMBOLS", "AAPL,MSFT,GOOGL")

        with patch('quantsploit.modules.scanners.price_momentum.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = scanner_module.run()

        assert results["scan_count"] == 3
        assert results["results_found"] == 3


class TestScanResults:
    """Tests for scan result structure"""

    def test_results_dataframe(self, scanner_module, sample_ohlcv):
        """Test results are returned as DataFrame"""
        scanner_module.set_option("SYMBOLS", "AAPL")

        with patch('quantsploit.modules.scanners.price_momentum.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = scanner_module.run()

        assert isinstance(results["scan_results"], pd.DataFrame)

    def test_results_columns(self, scanner_module, sample_ohlcv):
        """Test results have expected columns"""
        scanner_module.set_option("SYMBOLS", "AAPL")

        with patch('quantsploit.modules.scanners.price_momentum.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = scanner_module.run()

        expected_columns = ['Symbol', 'Price', 'Change %', 'Period Gain %',
                          'Volume', 'Vol Ratio', 'Flags']
        for col in expected_columns:
            assert col in results["scan_results"].columns


class TestFlagGeneration:
    """Tests for flag generation"""

    def test_high_volume_flag(self, scanner_module, high_momentum_data):
        """Test high volume flag is generated"""
        scanner_module.set_option("SYMBOLS", "AAPL")
        scanner_module.set_option("MIN_VOLUME", 1000000)

        with patch('quantsploit.modules.scanners.price_momentum.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = high_momentum_data

            results = scanner_module.run()

        flags = results["scan_results"]["Flags"].iloc[0]
        assert "High Volume" in flags

    def test_volume_spike_flag(self, scanner_module, volume_spike_data):
        """Test volume spike flag is generated"""
        scanner_module.set_option("SYMBOLS", "AAPL")

        with patch('quantsploit.modules.scanners.price_momentum.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = volume_spike_data

            results = scanner_module.run()

        flags = results["scan_results"]["Flags"].iloc[0]
        assert "Volume Spike" in flags

    def test_strong_momentum_flag(self, scanner_module, high_momentum_data):
        """Test strong momentum flag is generated"""
        scanner_module.set_option("SYMBOLS", "AAPL")
        scanner_module.set_option("MIN_GAIN_PCT", 5.0)

        with patch('quantsploit.modules.scanners.price_momentum.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = high_momentum_data

            results = scanner_module.run()

        flags = results["scan_results"]["Flags"].iloc[0]
        assert "Strong Momentum" in flags


class TestSorting:
    """Tests for result sorting"""

    def test_results_sorted_by_gain(self, scanner_module):
        """Test results are sorted by period gain descending"""
        scanner_module.set_option("SYMBOLS", "A,B,C")

        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

        def get_data(symbol, period, interval):
            if symbol == 'A':
                gain = 10
            elif symbol == 'B':
                gain = 30
            else:  # C
                gain = 20
            close = 100 + np.linspace(0, gain, 100)
            return pd.DataFrame({
                'Open': close * 0.99,
                'High': close * 1.01,
                'Low': close * 0.99,
                'Close': close,
                'Volume': [1000000] * 100
            }, index=dates)

        with patch('quantsploit.modules.scanners.price_momentum.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = get_data

            results = scanner_module.run()

        # B should be first (30% gain), then C (20%), then A (10%)
        symbols = results["scan_results"]["Symbol"].tolist()
        assert symbols[0] == 'B'
        assert symbols[1] == 'C'
        assert symbols[2] == 'A'


class TestNoDataHandling:
    """Tests for handling no data"""

    def test_skip_symbols_with_no_data(self, scanner_module, sample_ohlcv):
        """Test that symbols with no data are skipped"""
        scanner_module.set_option("SYMBOLS", "AAPL,INVALID")

        def get_data(symbol, period, interval):
            if symbol == 'INVALID':
                return None
            return sample_ohlcv

        with patch('quantsploit.modules.scanners.price_momentum.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = get_data

            results = scanner_module.run()

        assert results["scan_count"] == 2
        assert results["results_found"] == 1  # Only AAPL has data

    def test_empty_results(self, scanner_module):
        """Test handling when all symbols return no data"""
        scanner_module.set_option("SYMBOLS", "INVALID1,INVALID2")

        with patch('quantsploit.modules.scanners.price_momentum.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = None

            results = scanner_module.run()

        assert results["scan_count"] == 2
        assert results["results_found"] == 0
        assert results["scan_results"].empty


class TestSymbolParsing:
    """Tests for symbol parsing"""

    def test_parse_with_spaces(self, scanner_module, sample_ohlcv):
        """Test parsing symbols with spaces"""
        scanner_module.set_option("SYMBOLS", "AAPL, MSFT, GOOGL")

        with patch('quantsploit.modules.scanners.price_momentum.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = scanner_module.run()

        assert results["scan_count"] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
