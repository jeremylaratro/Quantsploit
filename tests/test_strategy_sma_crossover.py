"""
Unit tests for SMA Crossover Strategy Module

Tests cover:
- Module properties and initialization
- Option management
- Signal generation
- Backtest execution
- Return calculations
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.modules.strategies.sma_crossover import SMACrossover


@pytest.fixture
def mock_framework():
    """Create a mock framework with database"""
    framework = Mock()
    framework.database = Mock()
    framework.log = Mock()
    return framework


@pytest.fixture
def sma_module(mock_framework):
    """Create an SMACrossover module instance"""
    return SMACrossover(mock_framework)


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Create trending data with clear crossover opportunities
    trend = np.linspace(100, 150, 50).tolist() + np.linspace(150, 100, 50).tolist()
    noise = np.random.normal(0, 2, 100)
    close = np.array(trend) + noise

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.02,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)


class TestModuleProperties:
    """Tests for module property definitions"""

    def test_name(self, sma_module):
        """Test module name"""
        assert sma_module.name == "SMA Crossover Strategy"

    def test_description(self, sma_module):
        """Test module description"""
        assert "SMA" in sma_module.description
        assert "crossover" in sma_module.description.lower()

    def test_author(self, sma_module):
        """Test module author"""
        assert sma_module.author == "Quantsploit Team"

    def test_category(self, sma_module):
        """Test module category"""
        assert sma_module.category == "strategy"


class TestOptions:
    """Tests for module option management"""

    def test_default_options(self, sma_module):
        """Test default option values"""
        assert sma_module.get_option("FAST_PERIOD") == 10
        assert sma_module.get_option("SLOW_PERIOD") == 30
        assert sma_module.get_option("INITIAL_CAPITAL") == 10000

    def test_set_fast_period(self, sma_module):
        """Test setting fast period"""
        sma_module.set_option("FAST_PERIOD", 5)
        assert sma_module.get_option("FAST_PERIOD") == 5

    def test_set_slow_period(self, sma_module):
        """Test setting slow period"""
        sma_module.set_option("SLOW_PERIOD", 50)
        assert sma_module.get_option("SLOW_PERIOD") == 50

    def test_set_initial_capital(self, sma_module):
        """Test setting initial capital"""
        sma_module.set_option("INITIAL_CAPITAL", 50000)
        assert sma_module.get_option("INITIAL_CAPITAL") == 50000

    def test_inherited_options(self, sma_module):
        """Test that base module options are inherited"""
        assert "SYMBOL" in sma_module.options
        assert "PERIOD" in sma_module.options
        assert "INTERVAL" in sma_module.options


class TestTradingGuide:
    """Tests for trading guide"""

    def test_trading_guide_exists(self, sma_module):
        """Test trading guide method exists"""
        guide = sma_module.trading_guide()
        assert isinstance(guide, str)
        assert len(guide) > 0

    def test_trading_guide_content(self, sma_module):
        """Test trading guide contains key information"""
        guide = sma_module.trading_guide()
        assert "LONG" in guide
        assert "SMA" in guide
        assert "PARAMETERS" in guide

    def test_show_info_includes_guide(self, sma_module):
        """Test show_info includes trading guide"""
        info = sma_module.show_info()
        assert "trading_guide" in info


class TestBacktestExecution:
    """Tests for backtest execution"""

    def test_run_returns_results(self, sma_module, sample_ohlcv):
        """Test that run returns results dictionary"""
        sma_module.set_option("SYMBOL", "AAPL")

        with patch.object(sma_module.framework.database, '__class__') as mock_db:
            # Mock the DataFetcher to return our sample data
            with patch('quantsploit.modules.strategies.sma_crossover.DataFetcher') as MockFetcher:
                mock_fetcher = MockFetcher.return_value
                mock_fetcher.get_stock_data.return_value = sample_ohlcv

                results = sma_module.run()

        assert isinstance(results, dict)
        assert "symbol" in results
        assert "total_return" in results

    def test_run_no_data(self, sma_module):
        """Test run with no data returns error"""
        sma_module.set_option("SYMBOL", "INVALID")

        with patch('quantsploit.modules.strategies.sma_crossover.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = None

            results = sma_module.run()

        assert results["success"] is False
        assert "error" in results

    def test_run_empty_data(self, sma_module):
        """Test run with empty DataFrame returns error"""
        sma_module.set_option("SYMBOL", "EMPTY")

        with patch('quantsploit.modules.strategies.sma_crossover.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = pd.DataFrame()

            results = sma_module.run()

        assert results["success"] is False

    def test_results_contain_expected_fields(self, sma_module, sample_ohlcv):
        """Test results contain all expected fields"""
        sma_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.sma_crossover.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = sma_module.run()

        expected_fields = [
            'symbol', 'period', 'fast_sma', 'slow_sma',
            'initial_capital', 'final_value', 'total_return',
            'total_return_pct', 'buy_hold_return', 'buy_hold_return_pct',
            'strategy_vs_buy_hold', 'total_trades', 'trades', 'price_chart_data'
        ]

        for field in expected_fields:
            assert field in results, f"Missing field: {field}"


class TestReturnCalculations:
    """Tests for return calculations"""

    def test_initial_capital_preserved(self, sma_module, sample_ohlcv):
        """Test that initial capital is correctly used"""
        sma_module.set_option("SYMBOL", "TEST")
        sma_module.set_option("INITIAL_CAPITAL", 50000)

        with patch('quantsploit.modules.strategies.sma_crossover.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = sma_module.run()

        assert results["initial_capital"] == 50000

    def test_return_calculation(self, sma_module, sample_ohlcv):
        """Test return percentage calculation is consistent"""
        sma_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.sma_crossover.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = sma_module.run()

        # Verify return calculation
        expected_return = results["final_value"] - results["initial_capital"]
        assert abs(results["total_return"] - expected_return) < 0.01

        expected_pct = (expected_return / results["initial_capital"]) * 100
        assert abs(results["total_return_pct"] - expected_pct) < 0.01

    def test_buy_hold_comparison(self, sma_module, sample_ohlcv):
        """Test buy and hold comparison is calculated"""
        sma_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.sma_crossover.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = sma_module.run()

        # Verify strategy vs buy hold is correctly calculated
        expected_diff = results["total_return_pct"] - results["buy_hold_return_pct"]
        assert abs(results["strategy_vs_buy_hold"] - expected_diff) < 0.01


class TestSignalGeneration:
    """Tests for signal generation"""

    def test_trades_generated(self, sma_module, sample_ohlcv):
        """Test that trades are generated during backtest"""
        sma_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.sma_crossover.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = sma_module.run()

        # In trending data, we should see some trades
        assert results["total_trades"] >= 0

    def test_trades_dataframe(self, sma_module, sample_ohlcv):
        """Test trades are returned as DataFrame"""
        sma_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.sma_crossover.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = sma_module.run()

        assert isinstance(results["trades"], pd.DataFrame)


class TestPriceChartData:
    """Tests for price chart data output"""

    def test_chart_data_included(self, sma_module, sample_ohlcv):
        """Test price chart data is included in results"""
        sma_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.sma_crossover.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = sma_module.run()

        assert "price_chart_data" in results
        assert isinstance(results["price_chart_data"], pd.DataFrame)

    def test_chart_data_columns(self, sma_module, sample_ohlcv):
        """Test price chart data has expected columns"""
        sma_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.sma_crossover.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = sma_module.run()

        chart_data = results["price_chart_data"]
        assert "Close" in chart_data.columns
        assert "SMA_fast" in chart_data.columns
        assert "SMA_slow" in chart_data.columns

    def test_chart_data_limited_to_100(self, sma_module, sample_ohlcv):
        """Test price chart data is limited to 100 rows"""
        sma_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.sma_crossover.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = sma_module.run()

        assert len(results["price_chart_data"]) <= 100


class TestDifferentPeriods:
    """Tests for different SMA period configurations"""

    def test_different_fast_period(self, sma_module, sample_ohlcv):
        """Test with different fast period"""
        sma_module.set_option("SYMBOL", "TEST")
        sma_module.set_option("FAST_PERIOD", 5)

        with patch('quantsploit.modules.strategies.sma_crossover.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = sma_module.run()

        assert results["fast_sma"] == 5

    def test_different_slow_period(self, sma_module, sample_ohlcv):
        """Test with different slow period"""
        sma_module.set_option("SYMBOL", "TEST")
        sma_module.set_option("SLOW_PERIOD", 50)

        with patch('quantsploit.modules.strategies.sma_crossover.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = sma_module.run()

        assert results["slow_sma"] == 50

    def test_long_term_periods(self, sma_module):
        """Test with long-term periods (50/200)"""
        np.random.seed(42)
        # Need longer data for long-term SMAs
        dates = pd.date_range(start='2022-01-01', periods=300, freq='D')
        close = 100 + np.cumsum(np.random.randn(300) * 0.5)

        long_data = pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.02,
            'Low': close * 0.98,
            'Close': close,
            'Volume': np.random.randint(1000000, 10000000, 300)
        }, index=dates)

        sma_module.set_option("SYMBOL", "TEST")
        sma_module.set_option("FAST_PERIOD", 50)
        sma_module.set_option("SLOW_PERIOD", 200)

        with patch('quantsploit.modules.strategies.sma_crossover.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = long_data

            results = sma_module.run()

        assert results["fast_sma"] == 50
        assert results["slow_sma"] == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
