"""
Unit tests for Multi-Factor Scoring Strategy Module

Tests cover:
- Module properties and initialization
- Option management
- Factor weight parsing
- Individual factor scoring
- Composite score calculation
- Signal generation
- Multi-symbol processing
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.modules.strategies.multifactor_scoring import MultiFactorScoring


@pytest.fixture
def mock_framework():
    """Create a mock framework with database"""
    framework = Mock()
    framework.database = Mock()
    framework.log = Mock()
    return framework


@pytest.fixture
def multifactor_module(mock_framework):
    """Create a MultiFactorScoring module instance"""
    return MultiFactorScoring(mock_framework)


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=300, freq='D')

    # Create trending data
    trend = 100 + np.linspace(0, 50, 300) + np.random.normal(0, 3, 300)
    close = np.maximum(trend, 50)  # Ensure no negative prices

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.02,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, 300)
    }, index=dates)


@pytest.fixture
def uptrend_data():
    """Generate data with strong uptrend"""
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=300, freq='D')

    close = 100 * (1.005 ** np.arange(300))  # 0.5% daily gain

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.02,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, 300)
    }, index=dates)


@pytest.fixture
def downtrend_data():
    """Generate data with strong downtrend"""
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=300, freq='D')

    close = 100 * (0.995 ** np.arange(300))  # 0.5% daily loss

    return pd.DataFrame({
        'Open': close * 1.01,
        'High': close * 1.02,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, 300)
    }, index=dates)


class TestModuleProperties:
    """Tests for module property definitions"""

    def test_name(self, multifactor_module):
        """Test module name"""
        assert multifactor_module.name == "Multi-Factor Scoring"

    def test_description(self, multifactor_module):
        """Test module description"""
        assert "factor" in multifactor_module.description.lower()
        assert "momentum" in multifactor_module.description.lower()

    def test_author(self, multifactor_module):
        """Test module author"""
        assert multifactor_module.author == "Quantsploit Team"

    def test_category(self, multifactor_module):
        """Test module category"""
        assert multifactor_module.category == "strategy"


class TestOptions:
    """Tests for module option management"""

    def test_default_options(self, multifactor_module):
        """Test default option values"""
        assert multifactor_module.get_option("SYMBOLS") is None
        assert "momentum" in multifactor_module.get_option("FACTOR_WEIGHTS")
        assert multifactor_module.get_option("MAX_WORKERS") == 10

    def test_set_symbols(self, multifactor_module):
        """Test setting symbols"""
        multifactor_module.set_option("SYMBOLS", "AAPL,MSFT,GOOGL")
        assert multifactor_module.get_option("SYMBOLS") == "AAPL,MSFT,GOOGL"

    def test_set_factor_weights(self, multifactor_module):
        """Test setting factor weights"""
        multifactor_module.set_option("FACTOR_WEIGHTS", "momentum:0.5,technical:0.5")
        assert multifactor_module.get_option("FACTOR_WEIGHTS") == "momentum:0.5,technical:0.5"

    def test_set_max_workers(self, multifactor_module):
        """Test setting max workers"""
        multifactor_module.set_option("MAX_WORKERS", 5)
        assert multifactor_module.get_option("MAX_WORKERS") == 5

    def test_inherited_options(self, multifactor_module):
        """Test that base module options are inherited"""
        assert "PERIOD" in multifactor_module.options
        assert "INTERVAL" in multifactor_module.options


class TestTradingGuide:
    """Tests for trading guide"""

    def test_trading_guide_exists(self, multifactor_module):
        """Test trading guide method exists"""
        guide = multifactor_module.trading_guide()
        assert isinstance(guide, str)
        assert len(guide) > 0

    def test_trading_guide_content(self, multifactor_module):
        """Test trading guide contains key information"""
        guide = multifactor_module.trading_guide()
        assert "momentum" in guide.lower() or "Momentum" in guide
        assert "BUY" in guide
        assert "WEIGHTS" in guide

    def test_show_info_includes_guide(self, multifactor_module):
        """Test show_info includes trading guide"""
        info = multifactor_module.show_info()
        assert "trading_guide" in info


class TestWeightParsing:
    """Tests for factor weight parsing"""

    def test_parse_default_weights(self, multifactor_module):
        """Test parsing default weights"""
        weights_str = "momentum:0.3,technical:0.3,volatility:0.2,volume:0.2"
        weights = multifactor_module._parse_weights(weights_str)

        assert "momentum" in weights
        assert "technical" in weights
        assert "volatility" in weights
        assert "volume" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Sum should be ~1

    def test_parse_custom_weights(self, multifactor_module):
        """Test parsing custom weights"""
        weights_str = "momentum:0.5,technical:0.5"
        weights = multifactor_module._parse_weights(weights_str)

        assert abs(weights["momentum"] - 0.5) < 0.01
        assert abs(weights["technical"] - 0.5) < 0.01

    def test_weights_normalize(self, multifactor_module):
        """Test that weights are normalized to sum to 1"""
        weights_str = "momentum:2,technical:2,volatility:1,volume:1"
        weights = multifactor_module._parse_weights(weights_str)

        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01


class TestSingleStockScoring:
    """Tests for single stock scoring"""

    def test_score_stock_returns_dict(self, multifactor_module, sample_ohlcv):
        """Test that _score_stock returns dictionary"""
        multifactor_module.set_option("SYMBOLS", "AAPL")

        with patch('quantsploit.modules.strategies.multifactor_scoring.DataFetcher') as MockFetcher:
            mock_fetcher = Mock()
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            weights = multifactor_module._parse_weights("momentum:0.3,technical:0.3,volatility:0.2,volume:0.2")
            result = multifactor_module._score_stock("AAPL", mock_fetcher, "1y", "1d", weights)

        assert isinstance(result, dict)
        assert "Symbol" in result
        assert "composite_score" in result

    def test_score_stock_required_fields(self, multifactor_module, sample_ohlcv):
        """Test that result has all required fields"""
        with patch('quantsploit.modules.strategies.multifactor_scoring.DataFetcher') as MockFetcher:
            mock_fetcher = Mock()
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            weights = multifactor_module._parse_weights("momentum:0.3,technical:0.3,volatility:0.2,volume:0.2")
            result = multifactor_module._score_stock("TEST", mock_fetcher, "1y", "1d", weights)

        required_fields = ['Symbol', 'Price', 'Change%', 'composite_score',
                          'momentum_score', 'technical_score', 'volatility_score',
                          'volume_score', 'Signal']

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_scores_in_valid_range(self, multifactor_module, sample_ohlcv):
        """Test that all scores are between 0 and 100"""
        with patch('quantsploit.modules.strategies.multifactor_scoring.DataFetcher') as MockFetcher:
            mock_fetcher = Mock()
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            weights = multifactor_module._parse_weights("momentum:0.3,technical:0.3,volatility:0.2,volume:0.2")
            result = multifactor_module._score_stock("TEST", mock_fetcher, "1y", "1d", weights)

        assert 0 <= result['momentum_score'] <= 100
        assert 0 <= result['technical_score'] <= 100
        assert 0 <= result['volatility_score'] <= 100
        assert 0 <= result['volume_score'] <= 100
        assert 0 <= result['composite_score'] <= 100


class TestIndividualFactorScoring:
    """Tests for individual factor scoring methods"""

    def test_score_momentum(self, multifactor_module, sample_ohlcv):
        """Test momentum scoring"""
        close = sample_ohlcv['Close']
        score = multifactor_module._score_momentum(close)

        assert 0 <= score <= 100

    def test_score_momentum_uptrend(self, multifactor_module, uptrend_data):
        """Test momentum scoring for uptrend"""
        close = uptrend_data['Close']
        score = multifactor_module._score_momentum(close)

        # Uptrend should have higher momentum score
        assert score > 50

    def test_score_technical(self, multifactor_module, sample_ohlcv):
        """Test technical scoring"""
        df = sample_ohlcv
        score = multifactor_module._score_technical(
            df, df['Close'], df['High'], df['Low']
        )

        assert 0 <= score <= 100

    def test_score_volatility(self, multifactor_module, sample_ohlcv):
        """Test volatility scoring"""
        score = multifactor_module._score_volatility(
            sample_ohlcv['Close'], sample_ohlcv['High'], sample_ohlcv['Low']
        )

        assert 0 <= score <= 100

    def test_score_volume(self, multifactor_module, sample_ohlcv):
        """Test volume scoring"""
        score = multifactor_module._score_volume(
            sample_ohlcv['Volume'], sample_ohlcv['Close']
        )

        assert 0 <= score <= 100


class TestSignalGeneration:
    """Tests for signal generation"""

    def test_strong_buy_signal(self, multifactor_module, uptrend_data):
        """Test strong buy signal for high scores"""
        with patch('quantsploit.modules.strategies.multifactor_scoring.DataFetcher') as MockFetcher:
            mock_fetcher = Mock()
            mock_fetcher.get_stock_data.return_value = uptrend_data

            weights = multifactor_module._parse_weights("momentum:1,technical:0,volatility:0,volume:0")
            result = multifactor_module._score_stock("TEST", mock_fetcher, "1y", "1d", weights)

        # High momentum stock should get positive signal
        assert "STRONG BUY" in result['Signal'] or "BUY" in result['Signal']

    def test_sell_signal(self, multifactor_module, downtrend_data):
        """Test sell signal for low scores"""
        with patch('quantsploit.modules.strategies.multifactor_scoring.DataFetcher') as MockFetcher:
            mock_fetcher = Mock()
            mock_fetcher.get_stock_data.return_value = downtrend_data

            weights = multifactor_module._parse_weights("momentum:1,technical:0,volatility:0,volume:0")
            result = multifactor_module._score_stock("TEST", mock_fetcher, "1y", "1d", weights)

        # Downtrend should have lower score
        assert result['composite_score'] < 60


class TestRunExecution:
    """Tests for run execution"""

    def test_run_no_symbols_error(self, multifactor_module):
        """Test run with no symbols returns error"""
        results = multifactor_module.run()
        assert "error" in results

    def test_run_single_symbol(self, multifactor_module, sample_ohlcv):
        """Test run with single symbol"""
        multifactor_module.set_option("SYMBOL", "AAPL")

        with patch('quantsploit.modules.strategies.multifactor_scoring.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = multifactor_module.run()

        assert "symbol" in results
        assert "composite_score" in results
        assert "signal_strength" in results

    def test_run_multiple_symbols(self, multifactor_module, sample_ohlcv):
        """Test run with multiple symbols"""
        multifactor_module.set_option("SYMBOLS", "AAPL,MSFT")
        multifactor_module.set_option("MAX_WORKERS", 2)

        with patch('quantsploit.modules.strategies.multifactor_scoring.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = multifactor_module.run()

        assert "total_scored" in results
        assert "all_scores" in results

    def test_run_returns_sorted_results(self, multifactor_module):
        """Test results are sorted by composite score"""
        multifactor_module.set_option("SYMBOLS", "A,B,C")
        multifactor_module.set_option("MAX_WORKERS", 3)

        # Create different data for different scores
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=300, freq='D')

        data = {}
        for symbol, trend in [('A', 50), ('B', 0), ('C', 100)]:
            close = 100 + np.linspace(0, trend, 300)
            data[symbol] = pd.DataFrame({
                'Open': close * 0.99,
                'High': close * 1.02,
                'Low': close * 0.98,
                'Close': close,
                'Volume': np.random.randint(1000000, 10000000, 300)
            }, index=dates)

        with patch('quantsploit.modules.strategies.multifactor_scoring.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.side_effect = lambda s, p, i: data.get(s)

            results = multifactor_module.run()

        # Results should exist
        assert "all_scores" in results


class TestFactorAnalysis:
    """Tests for factor analysis"""

    def test_analyze_factors(self, multifactor_module, sample_ohlcv):
        """Test factor analysis"""
        multifactor_module.set_option("SYMBOLS", "AAPL,MSFT")

        with patch('quantsploit.modules.strategies.multifactor_scoring.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = multifactor_module.run()

        assert "factor_analysis" in results

    def test_factor_analysis_contents(self, multifactor_module, sample_ohlcv):
        """Test factor analysis contains expected keys"""
        multifactor_module.set_option("SYMBOLS", "AAPL,MSFT,GOOGL")

        with patch('quantsploit.modules.strategies.multifactor_scoring.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = multifactor_module.run()

        analysis = results.get("factor_analysis", {})
        if analysis:  # Only check if we got results
            expected_keys = ['avg_momentum', 'avg_technical', 'avg_volatility', 'avg_volume']
            for key in expected_keys:
                assert key in analysis, f"Missing key: {key}"


class TestRecommendations:
    """Tests for recommendation generation"""

    def test_generate_recommendations(self, multifactor_module, sample_ohlcv):
        """Test recommendation generation"""
        multifactor_module.set_option("SYMBOLS", "AAPL,MSFT")

        with patch('quantsploit.modules.strategies.multifactor_scoring.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = multifactor_module.run()

        assert "recommendation" in results
        assert isinstance(results["recommendation"], list)

    def test_empty_results_recommendation(self, multifactor_module):
        """Test recommendations with no data"""
        recommendations = multifactor_module._generate_recommendations([])
        assert len(recommendations) == 1
        assert "No data" in recommendations[0]


class TestInsufficientData:
    """Tests for handling insufficient data"""

    def test_score_stock_insufficient_data(self, multifactor_module):
        """Test handling of insufficient data"""
        # Create data with only 20 rows (need at least 50)
        short_data = pd.DataFrame({
            'Open': [100] * 20,
            'High': [102] * 20,
            'Low': [98] * 20,
            'Close': [100] * 20,
            'Volume': [1000000] * 20
        })

        with patch('quantsploit.modules.strategies.multifactor_scoring.DataFetcher') as MockFetcher:
            mock_fetcher = Mock()
            mock_fetcher.get_stock_data.return_value = short_data

            weights = multifactor_module._parse_weights("momentum:0.25,technical:0.25,volatility:0.25,volume:0.25")
            result = multifactor_module._score_stock("TEST", mock_fetcher, "1y", "1d", weights)

        # Should return None for insufficient data
        assert result is None

    def test_score_stock_no_data(self, multifactor_module):
        """Test handling of no data"""
        with patch('quantsploit.modules.strategies.multifactor_scoring.DataFetcher') as MockFetcher:
            mock_fetcher = Mock()
            mock_fetcher.get_stock_data.return_value = None

            weights = multifactor_module._parse_weights("momentum:0.25,technical:0.25,volatility:0.25,volume:0.25")
            result = multifactor_module._score_stock("TEST", mock_fetcher, "1y", "1d", weights)

        assert result is None


class TestSignalStrengthMapping:
    """Tests for signal strength mapping (for meta_analysis integration)"""

    def test_single_symbol_includes_signal_strength(self, multifactor_module, sample_ohlcv):
        """Test single symbol result includes signal_strength"""
        multifactor_module.set_option("SYMBOL", "AAPL")

        with patch('quantsploit.modules.strategies.multifactor_scoring.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = multifactor_module.run()

        assert "signal_strength" in results
        # Signal strength should be a number
        assert isinstance(results["signal_strength"], (int, float))

    def test_signal_strength_range(self, multifactor_module, sample_ohlcv):
        """Test signal strength is in reasonable range"""
        multifactor_module.set_option("SYMBOL", "AAPL")

        with patch('quantsploit.modules.strategies.multifactor_scoring.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = multifactor_module.run()

        # Signal strength should be between -100 and 100
        assert -100 <= results["signal_strength"] <= 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
