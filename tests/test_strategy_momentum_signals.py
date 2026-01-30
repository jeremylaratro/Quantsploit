"""
Unit tests for Momentum Signals Strategy Module

Tests cover:
- Module properties and initialization
- Option management
- Signal generation
- Momentum metrics calculation
- Trend analysis
- Relative strength calculation
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.modules.strategies.momentum_signals import MomentumSignals


@pytest.fixture
def mock_framework():
    """Create a mock framework with database"""
    framework = Mock()
    framework.database = Mock()
    framework.log = Mock()
    return framework


@pytest.fixture
def momentum_module(mock_framework):
    """Create a MomentumSignals module instance"""
    return MomentumSignals(mock_framework)


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Create trending data
    trend = np.linspace(100, 130, 100)
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
def uptrend_data():
    """Generate data with clear uptrend"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Strong uptrend
    close = 100 * (1.01 ** np.arange(100))  # 1% daily gain

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.02,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)


@pytest.fixture
def downtrend_data():
    """Generate data with clear downtrend"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Strong downtrend
    close = 100 * (0.99 ** np.arange(100))  # 1% daily loss

    return pd.DataFrame({
        'Open': close * 1.01,
        'High': close * 1.02,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)


@pytest.fixture
def benchmark_data():
    """Generate benchmark (SPY) data"""
    np.random.seed(43)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Moderate uptrend
    close = 100 + np.linspace(0, 10, 100) + np.random.normal(0, 1, 100)

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)


class TestModuleProperties:
    """Tests for module property definitions"""

    def test_name(self, momentum_module):
        """Test module name"""
        assert momentum_module.name == "Momentum Signals"

    def test_description(self, momentum_module):
        """Test module description"""
        assert "momentum" in momentum_module.description.lower()
        assert "ROC" in momentum_module.description

    def test_author(self, momentum_module):
        """Test module author"""
        assert momentum_module.author == "Quantsploit Team"

    def test_category(self, momentum_module):
        """Test module category"""
        assert momentum_module.category == "strategy"


class TestOptions:
    """Tests for module option management"""

    def test_default_options(self, momentum_module):
        """Test default option values"""
        assert momentum_module.get_option("BENCHMARK") == "SPY"
        assert momentum_module.get_option("MOMENTUM_PERIOD") == 12
        assert momentum_module.get_option("MIN_ADX") == 25

    def test_set_benchmark(self, momentum_module):
        """Test setting benchmark"""
        momentum_module.set_option("BENCHMARK", "QQQ")
        assert momentum_module.get_option("BENCHMARK") == "QQQ"

    def test_set_momentum_period(self, momentum_module):
        """Test setting momentum period"""
        momentum_module.set_option("MOMENTUM_PERIOD", 20)
        assert momentum_module.get_option("MOMENTUM_PERIOD") == 20

    def test_set_min_adx(self, momentum_module):
        """Test setting minimum ADX"""
        momentum_module.set_option("MIN_ADX", 30)
        assert momentum_module.get_option("MIN_ADX") == 30

    def test_inherited_options(self, momentum_module):
        """Test that base module options are inherited"""
        assert "SYMBOL" in momentum_module.options
        assert "PERIOD" in momentum_module.options
        assert "INTERVAL" in momentum_module.options


class TestTradingGuide:
    """Tests for trading guide"""

    def test_trading_guide_exists(self, momentum_module):
        """Test trading guide method exists"""
        guide = momentum_module.trading_guide()
        assert isinstance(guide, str)
        assert len(guide) > 0

    def test_trading_guide_content(self, momentum_module):
        """Test trading guide contains key information"""
        guide = momentum_module.trading_guide()
        assert "ROC" in guide
        assert "BUY" in guide
        assert "SELL" in guide

    def test_show_info_includes_guide(self, momentum_module):
        """Test show_info includes trading guide"""
        info = momentum_module.show_info()
        assert "trading_guide" in info


class TestRunExecution:
    """Tests for run execution"""

    def test_run_returns_results(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test that run returns results dictionary"""
        momentum_module.set_option("SYMBOL", "AAPL")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        assert isinstance(results, dict)
        assert "symbol" in results
        assert "signal_score" in results

    def test_run_no_data(self, momentum_module):
        """Test run with no data returns error"""
        momentum_module.set_option("SYMBOL", "INVALID")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = None

            results = momentum_module.run()

        assert results["success"] is False
        assert "error" in results

    def test_run_empty_data(self, momentum_module):
        """Test run with empty DataFrame returns error"""
        momentum_module.set_option("SYMBOL", "EMPTY")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = pd.DataFrame()

            results = momentum_module.run()

        assert results["success"] is False

    def test_results_contain_expected_fields(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test results contain all expected fields"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        expected_fields = [
            'symbol', 'current_price', 'signal_score', 'momentum_rank',
            'overall_signal', 'recommendation', 'signals',
            'momentum_metrics', 'trend_analysis', 'recent_data'
        ]

        for field in expected_fields:
            assert field in results, f"Missing field: {field}"


class TestMomentumMetrics:
    """Tests for momentum metrics calculation"""

    def test_momentum_metrics_included(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test momentum metrics are included"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        assert "momentum_metrics" in results
        metrics = results["momentum_metrics"]
        assert "roc_12_period" in metrics
        assert "roc_6_period" in metrics
        assert "momentum_acceleration" in metrics
        assert "adx" in metrics
        assert "volume_momentum" in metrics

    def test_roc_calculated(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test ROC values are calculated"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        metrics = results["momentum_metrics"]
        assert isinstance(metrics["roc_12_period"], float)
        assert isinstance(metrics["roc_6_period"], float)


class TestSignalGeneration:
    """Tests for signal generation"""

    def test_signals_list_generated(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test that signals list is generated"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        assert "signals" in results
        assert isinstance(results["signals"], list)

    def test_overall_signal_generated(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test that overall signal is generated"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        assert "overall_signal" in results
        assert isinstance(results["overall_signal"], str)

    def test_signal_score_calculated(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test signal score is calculated"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        assert "signal_score" in results
        assert isinstance(results["signal_score"], (int, float))


class TestTrendAnalysis:
    """Tests for trend analysis"""

    def test_trend_analysis_included(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test trend analysis is included"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        assert "trend_analysis" in results
        trend = results["trend_analysis"]
        assert "ma_alignment" in trend
        assert "sma_10" in trend
        assert "sma_20" in trend
        assert "sma_50" in trend

    def test_ma_alignment_valid_values(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test MA alignment has valid values"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        alignment = results["trend_analysis"]["ma_alignment"]
        assert alignment in ["bullish", "bearish", "mixed"]


class TestMomentumRank:
    """Tests for momentum rank calculation"""

    def test_momentum_rank_in_range(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test momentum rank is between 0 and 100"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        assert 0 <= results["momentum_rank"] <= 100


class TestRelativeStrength:
    """Tests for relative strength calculation"""

    def test_relative_strength_with_benchmark(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test relative strength is calculated when benchmark available"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        metrics = results["momentum_metrics"]
        assert "relative_strength_vs_benchmark" in metrics

    def test_relative_strength_without_benchmark(self, momentum_module, sample_ohlcv):
        """Test handling when benchmark is unavailable"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            # First call returns stock data, second returns None for benchmark
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, None]

            results = momentum_module.run()

        # Should still work, just without relative strength
        assert "momentum_metrics" in results
        metrics = results["momentum_metrics"]
        assert metrics["relative_strength_vs_benchmark"] is None


class TestUptrend:
    """Tests with uptrend data"""

    def test_uptrend_positive_roc(self, momentum_module, uptrend_data, benchmark_data):
        """Test uptrend produces positive ROC"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [uptrend_data, benchmark_data]

            results = momentum_module.run()

        metrics = results["momentum_metrics"]
        # Strong uptrend should have very positive ROC
        assert metrics["roc_12_period"] > 0

    def test_uptrend_positive_signal(self, momentum_module, uptrend_data, benchmark_data):
        """Test uptrend produces positive signal score"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [uptrend_data, benchmark_data]

            results = momentum_module.run()

        assert results["signal_score"] > 0


class TestDowntrend:
    """Tests with downtrend data"""

    def test_downtrend_negative_roc(self, momentum_module, downtrend_data, benchmark_data):
        """Test downtrend produces negative ROC"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [downtrend_data, benchmark_data]

            results = momentum_module.run()

        metrics = results["momentum_metrics"]
        # Strong downtrend should have negative ROC
        assert metrics["roc_12_period"] < 0

    def test_downtrend_negative_signal(self, momentum_module, downtrend_data, benchmark_data):
        """Test downtrend produces negative signal score"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [downtrend_data, benchmark_data]

            results = momentum_module.run()

        assert results["signal_score"] < 0


class TestRecentData:
    """Tests for recent data output"""

    def test_recent_data_included(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test recent data is included"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        assert "recent_data" in results
        assert isinstance(results["recent_data"], pd.DataFrame)

    def test_recent_data_columns(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test recent data has expected columns"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        recent = results["recent_data"]
        assert "Close" in recent.columns
        assert "roc_12" in recent.columns
        assert "volume_momentum" in recent.columns

    def test_recent_data_limited_to_10(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test recent data is limited to 10 rows"""
        momentum_module.set_option("SYMBOL", "TEST")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        assert len(results["recent_data"]) == 10


class TestDifferentParameters:
    """Tests for different parameter configurations"""

    def test_different_benchmark(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test with different benchmark"""
        momentum_module.set_option("SYMBOL", "TEST")
        momentum_module.set_option("BENCHMARK", "QQQ")

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        assert "signal_score" in results

    def test_different_momentum_period(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test with different momentum period"""
        momentum_module.set_option("SYMBOL", "TEST")
        momentum_module.set_option("MOMENTUM_PERIOD", 20)

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        assert "momentum_metrics" in results

    def test_strict_adx_threshold(self, momentum_module, sample_ohlcv, benchmark_data):
        """Test with strict ADX threshold"""
        momentum_module.set_option("SYMBOL", "TEST")
        momentum_module.set_option("MIN_ADX", 40)

        with patch('quantsploit.modules.strategies.momentum_signals.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = [sample_ohlcv, benchmark_data]

            results = momentum_module.run()

        assert "signals" in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
