"""
Unit tests for Mean Reversion Strategy Module

Tests cover:
- Module properties and initialization
- Option management
- Signal generation
- Z-score analysis
- Bollinger Bands integration
- Reversion probability calculation
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.modules.strategies.mean_reversion import MeanReversion


@pytest.fixture
def mock_framework():
    """Create a mock framework with database"""
    framework = Mock()
    framework.database = Mock()
    framework.log = Mock()
    return framework


@pytest.fixture
def mean_reversion_module(mock_framework):
    """Create a MeanReversion module instance"""
    return MeanReversion(mock_framework)


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Create mean-reverting data with oscillations around 100
    trend = 100 + np.sin(np.linspace(0, 6 * np.pi, 100)) * 15
    noise = np.random.normal(0, 2, 100)
    close = trend + noise

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.02,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)


@pytest.fixture
def oversold_data():
    """Generate data with oversold conditions"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')

    # Price that drops significantly below mean
    close = np.concatenate([
        np.linspace(100, 100, 30),  # Stable period
        np.linspace(100, 70, 20)    # Sharp decline
    ])

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.01,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, 50)
    }, index=dates)


@pytest.fixture
def overbought_data():
    """Generate data with overbought conditions"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')

    # Price that rises significantly above mean
    close = np.concatenate([
        np.linspace(100, 100, 30),  # Stable period
        np.linspace(100, 130, 20)   # Sharp rise
    ])

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.02,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, 50)
    }, index=dates)


class TestModuleProperties:
    """Tests for module property definitions"""

    def test_name(self, mean_reversion_module):
        """Test module name"""
        assert mean_reversion_module.name == "Mean Reversion Strategy"

    def test_description(self, mean_reversion_module):
        """Test module description"""
        assert "mean reversion" in mean_reversion_module.description.lower()
        assert "Z-score" in mean_reversion_module.description

    def test_author(self, mean_reversion_module):
        """Test module author"""
        assert mean_reversion_module.author == "Quantsploit Team"

    def test_category(self, mean_reversion_module):
        """Test module category"""
        assert mean_reversion_module.category == "strategy"


class TestOptions:
    """Tests for module option management"""

    def test_default_options(self, mean_reversion_module):
        """Test default option values"""
        assert mean_reversion_module.get_option("LOOKBACK") == 20
        assert mean_reversion_module.get_option("Z_THRESHOLD") == 2.0
        assert mean_reversion_module.get_option("BB_PERIOD") == 20
        assert mean_reversion_module.get_option("BB_STD") == 2.0

    def test_set_lookback(self, mean_reversion_module):
        """Test setting lookback period"""
        mean_reversion_module.set_option("LOOKBACK", 30)
        assert mean_reversion_module.get_option("LOOKBACK") == 30

    def test_set_z_threshold(self, mean_reversion_module):
        """Test setting z-score threshold"""
        mean_reversion_module.set_option("Z_THRESHOLD", 2.5)
        assert mean_reversion_module.get_option("Z_THRESHOLD") == 2.5

    def test_set_bb_period(self, mean_reversion_module):
        """Test setting Bollinger Bands period"""
        mean_reversion_module.set_option("BB_PERIOD", 15)
        assert mean_reversion_module.get_option("BB_PERIOD") == 15

    def test_set_bb_std(self, mean_reversion_module):
        """Test setting Bollinger Bands standard deviation"""
        mean_reversion_module.set_option("BB_STD", 2.5)
        assert mean_reversion_module.get_option("BB_STD") == 2.5

    def test_inherited_options(self, mean_reversion_module):
        """Test that base module options are inherited"""
        assert "SYMBOL" in mean_reversion_module.options
        assert "PERIOD" in mean_reversion_module.options
        assert "INTERVAL" in mean_reversion_module.options


class TestTradingGuide:
    """Tests for trading guide"""

    def test_trading_guide_exists(self, mean_reversion_module):
        """Test trading guide method exists"""
        guide = mean_reversion_module.trading_guide()
        assert isinstance(guide, str)
        assert len(guide) > 0

    def test_trading_guide_content(self, mean_reversion_module):
        """Test trading guide contains key information"""
        guide = mean_reversion_module.trading_guide()
        assert "Z-score" in guide
        assert "LONG" in guide
        assert "oversold" in guide.lower()

    def test_show_info_includes_guide(self, mean_reversion_module):
        """Test show_info includes trading guide"""
        info = mean_reversion_module.show_info()
        assert "trading_guide" in info


class TestRunExecution:
    """Tests for run execution"""

    def test_run_returns_results(self, mean_reversion_module, sample_ohlcv):
        """Test that run returns results dictionary"""
        mean_reversion_module.set_option("SYMBOL", "AAPL")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        assert isinstance(results, dict)
        assert "symbol" in results
        assert "z_score" in results

    def test_run_no_data(self, mean_reversion_module):
        """Test run with no data returns error"""
        mean_reversion_module.set_option("SYMBOL", "INVALID")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = None

            results = mean_reversion_module.run()

        assert results["success"] is False
        assert "error" in results

    def test_run_empty_data(self, mean_reversion_module):
        """Test run with empty DataFrame returns error"""
        mean_reversion_module.set_option("SYMBOL", "EMPTY")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = pd.DataFrame()

            results = mean_reversion_module.run()

        assert results["success"] is False

    def test_results_contain_expected_fields(self, mean_reversion_module, sample_ohlcv):
        """Test results contain all expected fields"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        expected_fields = [
            'symbol', 'current_price', 'rolling_mean', 'z_score',
            'percentile_rank', 'rsi', 'signal_strength', 'overall_signal',
            'expected_return_to_mean', 'mean_reversion_probability', 'signals',
            'statistics', 'recent_data'
        ]

        for field in expected_fields:
            assert field in results, f"Missing field: {field}"


class TestZScoreAnalysis:
    """Tests for z-score calculation and signals"""

    def test_z_score_calculated(self, mean_reversion_module, sample_ohlcv):
        """Test z-score is calculated correctly"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        assert "z_score" in results
        assert isinstance(results["z_score"], float)

    def test_oversold_detection(self, mean_reversion_module, oversold_data):
        """Test detection of oversold conditions"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = oversold_data

            results = mean_reversion_module.run()

        # With a sharp decline, z-score should be negative
        assert results["z_score"] < 0

    def test_overbought_detection(self, mean_reversion_module, overbought_data):
        """Test detection of overbought conditions"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = overbought_data

            results = mean_reversion_module.run()

        # With a sharp rise, z-score should be positive
        assert results["z_score"] > 0


class TestSignalGeneration:
    """Tests for signal generation"""

    def test_signals_list_generated(self, mean_reversion_module, sample_ohlcv):
        """Test that signals list is generated"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        assert "signals" in results
        assert isinstance(results["signals"], list)

    def test_overall_signal_generated(self, mean_reversion_module, sample_ohlcv):
        """Test that overall signal is generated"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        assert "overall_signal" in results
        assert isinstance(results["overall_signal"], str)

    def test_signal_strength_calculated(self, mean_reversion_module, sample_ohlcv):
        """Test signal strength is calculated"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        assert "signal_strength" in results
        assert isinstance(results["signal_strength"], (int, float))


class TestBollingerBandsIntegration:
    """Tests for Bollinger Bands integration"""

    def test_bollinger_bands_calculated(self, mean_reversion_module, sample_ohlcv):
        """Test Bollinger Bands are included in results"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        assert "bollinger_bands" in results
        bb = results["bollinger_bands"]
        assert "upper" in bb
        assert "middle" in bb
        assert "lower" in bb
        assert "width" in bb

    def test_bb_order_correct(self, mean_reversion_module, sample_ohlcv):
        """Test BB upper > middle > lower"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        bb = results["bollinger_bands"]
        assert bb["upper"] > bb["middle"] > bb["lower"]


class TestStatistics:
    """Tests for statistics calculation"""

    def test_statistics_included(self, mean_reversion_module, sample_ohlcv):
        """Test statistics are included in results"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        assert "statistics" in results
        stats = results["statistics"]
        assert "mean" in stats
        assert "std_dev" in stats
        assert "min_20d" in stats
        assert "max_20d" in stats

    def test_min_max_order(self, mean_reversion_module, sample_ohlcv):
        """Test min < max in statistics"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        stats = results["statistics"]
        assert stats["min_20d"] <= stats["max_20d"]


class TestReversionProbability:
    """Tests for mean reversion probability calculation"""

    def test_probability_calculated(self, mean_reversion_module, sample_ohlcv):
        """Test mean reversion probability is calculated"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        assert "mean_reversion_probability" in results

    def test_probability_in_valid_range(self, mean_reversion_module, sample_ohlcv):
        """Test probability is between 0 and 100%"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        # Parse probability string like "65.0%"
        prob_str = results["mean_reversion_probability"]
        prob_value = float(prob_str.rstrip('%'))
        assert 0 <= prob_value <= 100


class TestExpectedReturn:
    """Tests for expected return calculation"""

    def test_expected_return_calculated(self, mean_reversion_module, sample_ohlcv):
        """Test expected return to mean is calculated"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        assert "expected_return_to_mean" in results

    def test_expected_return_format(self, mean_reversion_module, sample_ohlcv):
        """Test expected return has correct format with +/- sign"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        ret_str = results["expected_return_to_mean"]
        assert "%" in ret_str
        # Should have sign and percentage
        assert any(c in ret_str for c in ['+', '-'])


class TestRecentData:
    """Tests for recent data output"""

    def test_recent_data_included(self, mean_reversion_module, sample_ohlcv):
        """Test recent data is included"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        assert "recent_data" in results
        assert isinstance(results["recent_data"], pd.DataFrame)

    def test_recent_data_columns(self, mean_reversion_module, sample_ohlcv):
        """Test recent data has expected columns"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        recent = results["recent_data"]
        assert "Close" in recent.columns
        assert "z_score" in recent.columns
        assert "percentile_rank" in recent.columns
        assert "rsi" in recent.columns

    def test_recent_data_limited_to_10(self, mean_reversion_module, sample_ohlcv):
        """Test recent data is limited to 10 rows"""
        mean_reversion_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        assert len(results["recent_data"]) == 10


class TestDifferentParameters:
    """Tests for different parameter configurations"""

    def test_different_lookback(self, mean_reversion_module, sample_ohlcv):
        """Test with different lookback period"""
        mean_reversion_module.set_option("SYMBOL", "TEST")
        mean_reversion_module.set_option("LOOKBACK", 10)

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        assert "z_score" in results
        assert isinstance(results["z_score"], float)

    def test_different_z_threshold(self, mean_reversion_module, sample_ohlcv):
        """Test with different z-score threshold"""
        mean_reversion_module.set_option("SYMBOL", "TEST")
        mean_reversion_module.set_option("Z_THRESHOLD", 1.5)

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        # With lower threshold, might get different signal strength
        assert "signal_strength" in results

    def test_strict_threshold(self, mean_reversion_module, sample_ohlcv):
        """Test with strict z-score threshold"""
        mean_reversion_module.set_option("SYMBOL", "TEST")
        mean_reversion_module.set_option("Z_THRESHOLD", 3.0)

        with patch('quantsploit.modules.strategies.mean_reversion.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = mean_reversion_module.run()

        # Stricter threshold should require more extreme values
        assert "overall_signal" in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
