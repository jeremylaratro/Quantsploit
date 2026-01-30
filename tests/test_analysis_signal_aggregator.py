"""
Unit tests for Signal Aggregator Module

Tests cover:
- Module properties and initialization
- Individual strategy signal generation
- Consensus calculation
- Final signal generation
- Risk assessment
- Insight generation
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.modules.analysis.signal_aggregator import SignalAggregator


@pytest.fixture
def mock_framework():
    """Create a mock framework with database"""
    framework = Mock()
    framework.database = Mock()
    framework.log = Mock()
    return framework


@pytest.fixture
def signal_module(mock_framework):
    """Create a SignalAggregator module instance"""
    return SignalAggregator(mock_framework)


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
def bullish_data():
    """Generate strongly bullish data"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Strong uptrend
    close = 100 * (1.01 ** np.arange(100))

    return pd.DataFrame({
        'Open': close * 0.995,
        'High': close * 1.02,
        'Low': close * 0.99,
        'Close': close,
        'Volume': np.linspace(1000000, 5000000, 100)  # Increasing volume
    }, index=dates)


@pytest.fixture
def bearish_data():
    """Generate strongly bearish data"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Strong downtrend
    close = 100 * (0.99 ** np.arange(100))

    return pd.DataFrame({
        'Open': close * 1.005,
        'High': close * 1.01,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.linspace(5000000, 1000000, 100)  # Decreasing volume
    }, index=dates)


@pytest.fixture
def volatile_data():
    """Generate highly volatile data"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # High volatility
    close = 100 + np.random.normal(0, 10, 100).cumsum()

    return pd.DataFrame({
        'Open': close * 0.95,
        'High': close * 1.10,
        'Low': close * 0.90,
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)


class TestModuleProperties:
    """Tests for module property definitions"""

    def test_name(self, signal_module):
        """Test module name"""
        assert signal_module.name == "Signal Aggregator"

    def test_description(self, signal_module):
        """Test module description"""
        assert "aggregate" in signal_module.description.lower()
        assert "signal" in signal_module.description.lower()

    def test_author(self, signal_module):
        """Test module author"""
        assert signal_module.author == "Quantsploit Team"

    def test_category(self, signal_module):
        """Test module category"""
        assert signal_module.category == "analysis"


class TestOptions:
    """Tests for module option management"""

    def test_default_options(self, signal_module):
        """Test default option values"""
        assert signal_module.get_option("STRATEGIES") == "all"
        assert signal_module.get_option("MIN_CONFIDENCE") == 60

    def test_set_strategies(self, signal_module):
        """Test setting strategies filter"""
        signal_module.set_option("STRATEGIES", "momentum")
        assert signal_module.get_option("STRATEGIES") == "momentum"

    def test_set_min_confidence(self, signal_module):
        """Test setting minimum confidence"""
        signal_module.set_option("MIN_CONFIDENCE", 80)
        assert signal_module.get_option("MIN_CONFIDENCE") == 80

    def test_inherited_options(self, signal_module):
        """Test that base module options are inherited"""
        assert "SYMBOL" in signal_module.options
        assert "PERIOD" in signal_module.options
        assert "INTERVAL" in signal_module.options


class TestRunExecution:
    """Tests for run execution"""

    def test_run_returns_results(self, signal_module, sample_ohlcv):
        """Test run returns results dictionary"""
        signal_module.set_option("SYMBOL", "AAPL")

        with patch('quantsploit.modules.analysis.signal_aggregator.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = signal_module.run()

        assert "symbol" in results
        assert "final_signal" in results
        assert "confidence" in results
        assert "consensus_score" in results
        assert "strategy_signals" in results

    def test_run_no_data_error(self, signal_module):
        """Test run with no data returns error"""
        signal_module.set_option("SYMBOL", "INVALID")

        with patch('quantsploit.modules.analysis.signal_aggregator.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = None

            results = signal_module.run()

        assert results["success"] is False
        assert "error" in results

    def test_run_empty_data_error(self, signal_module):
        """Test run with empty data returns error"""
        signal_module.set_option("SYMBOL", "AAPL")

        with patch('quantsploit.modules.analysis.signal_aggregator.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = pd.DataFrame()

            results = signal_module.run()

        assert results["success"] is False


class TestMomentumSignals:
    """Tests for momentum signal generation"""

    def test_momentum_returns_dict(self, signal_module, sample_ohlcv):
        """Test momentum signals return dictionary"""
        result = signal_module._get_momentum_signals(sample_ohlcv)

        assert isinstance(result, dict)
        assert "signal" in result
        assert "score" in result
        assert "detail" in result

    def test_momentum_signal_values(self, signal_module, sample_ohlcv):
        """Test momentum signal values are valid"""
        result = signal_module._get_momentum_signals(sample_ohlcv)

        assert result["signal"] in ["bullish", "bearish", "neutral"]
        assert 0 <= result["score"] <= 100

    def test_momentum_bullish_on_uptrend(self, signal_module, bullish_data):
        """Test bullish momentum signal on uptrend"""
        result = signal_module._get_momentum_signals(bullish_data)

        # Strong uptrend should produce bullish signal
        assert result["signal"] == "bullish"


class TestMeanReversionSignals:
    """Tests for mean reversion signal generation"""

    def test_mean_reversion_returns_dict(self, signal_module, sample_ohlcv):
        """Test mean reversion signals return dictionary"""
        result = signal_module._get_mean_reversion_signals(sample_ohlcv)

        assert isinstance(result, dict)
        assert "signal" in result
        assert "score" in result
        assert "detail" in result

    def test_mean_reversion_signal_values(self, signal_module, sample_ohlcv):
        """Test mean reversion signal values are valid"""
        result = signal_module._get_mean_reversion_signals(sample_ohlcv)

        assert result["signal"] in ["bullish", "bearish", "neutral"]
        assert 0 <= result["score"] <= 100

    def test_mean_reversion_detail_contains_zscore(self, signal_module, sample_ohlcv):
        """Test mean reversion detail includes z-score"""
        result = signal_module._get_mean_reversion_signals(sample_ohlcv)
        assert "Z-score" in result["detail"]


class TestTechnicalSignals:
    """Tests for technical indicator signals"""

    def test_technical_returns_dict(self, signal_module, sample_ohlcv):
        """Test technical signals return dictionary"""
        result = signal_module._get_technical_signals(sample_ohlcv)

        assert isinstance(result, dict)
        assert "signal" in result
        assert "score" in result
        assert "detail" in result

    def test_technical_signal_values(self, signal_module, sample_ohlcv):
        """Test technical signal values are valid"""
        result = signal_module._get_technical_signals(sample_ohlcv)

        assert result["signal"] in ["bullish", "bearish", "neutral"]
        assert 0 <= result["score"] <= 100

    def test_technical_detail_contains_counts(self, signal_module, sample_ohlcv):
        """Test technical detail includes bull/bear counts"""
        result = signal_module._get_technical_signals(sample_ohlcv)
        assert "Bull:" in result["detail"]
        assert "Bear:" in result["detail"]


class TestPatternSignals:
    """Tests for pattern signal generation"""

    def test_pattern_returns_dict(self, signal_module, sample_ohlcv):
        """Test pattern signals return dictionary"""
        result = signal_module._get_pattern_signals(sample_ohlcv)

        assert isinstance(result, dict)
        assert "signal" in result
        assert "score" in result
        assert "detail" in result

    def test_pattern_signal_values(self, signal_module, sample_ohlcv):
        """Test pattern signal values are valid"""
        result = signal_module._get_pattern_signals(sample_ohlcv)

        assert result["signal"] in ["bullish", "bearish", "neutral"]
        assert 0 <= result["score"] <= 100


class TestVolumeSignals:
    """Tests for volume signal generation"""

    def test_volume_returns_dict(self, signal_module, sample_ohlcv):
        """Test volume signals return dictionary"""
        result = signal_module._get_volume_signals(sample_ohlcv)

        assert isinstance(result, dict)
        assert "signal" in result
        assert "score" in result
        assert "detail" in result

    def test_volume_signal_values(self, signal_module, sample_ohlcv):
        """Test volume signal values are valid"""
        result = signal_module._get_volume_signals(sample_ohlcv)

        assert result["signal"] in ["bullish", "bearish", "neutral"]
        assert 0 <= result["score"] <= 100

    def test_volume_detail_contains_ratio(self, signal_module, sample_ohlcv):
        """Test volume detail includes ratio"""
        result = signal_module._get_volume_signals(sample_ohlcv)
        assert "ratio" in result["detail"].lower()


class TestConsensusCalculation:
    """Tests for consensus calculation"""

    def test_consensus_returns_dict(self, signal_module):
        """Test consensus calculation returns dictionary"""
        signals = {
            "momentum": {"signal": "bullish", "score": 70},
            "mean_reversion": {"signal": "neutral", "score": 50},
            "technical": {"signal": "bullish", "score": 65},
            "pattern": {"signal": "neutral", "score": 50},
            "volume": {"signal": "bullish", "score": 60}
        }

        result = signal_module._calculate_consensus(signals)

        assert isinstance(result, dict)
        assert "bullish_count" in result
        assert "bearish_count" in result
        assert "neutral_count" in result
        assert "score" in result

    def test_consensus_counts_correct(self, signal_module):
        """Test consensus counts are correct"""
        signals = {
            "momentum": {"signal": "bullish", "score": 70},
            "mean_reversion": {"signal": "neutral", "score": 50},
            "technical": {"signal": "bullish", "score": 65},
            "pattern": {"signal": "bearish", "score": 60},
            "volume": {"signal": "bullish", "score": 60}
        }

        result = signal_module._calculate_consensus(signals)

        assert result["bullish_count"] == 3
        assert result["bearish_count"] == 1
        assert result["neutral_count"] == 1

    def test_consensus_score_positive_on_bullish(self, signal_module):
        """Test consensus score is positive when bullish dominates"""
        signals = {
            "momentum": {"signal": "bullish", "score": 80},
            "mean_reversion": {"signal": "bullish", "score": 70},
            "technical": {"signal": "bullish", "score": 75},
            "pattern": {"signal": "neutral", "score": 50},
            "volume": {"signal": "neutral", "score": 50}
        }

        result = signal_module._calculate_consensus(signals)

        assert result["score"] > 0

    def test_consensus_score_negative_on_bearish(self, signal_module):
        """Test consensus score is negative when bearish dominates"""
        signals = {
            "momentum": {"signal": "bearish", "score": 80},
            "mean_reversion": {"signal": "bearish", "score": 70},
            "technical": {"signal": "bearish", "score": 75},
            "pattern": {"signal": "neutral", "score": 50},
            "volume": {"signal": "neutral", "score": 50}
        }

        result = signal_module._calculate_consensus(signals)

        assert result["score"] < 0


class TestFinalSignalGeneration:
    """Tests for final signal generation"""

    def test_final_signal_returns_tuple(self, signal_module):
        """Test final signal returns tuple"""
        consensus = {
            "bullish_count": 3,
            "bearish_count": 1,
            "neutral_count": 1,
            "score": 40
        }

        result = signal_module._generate_final_signal(consensus, 60)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_buy_signal_on_high_consensus(self, signal_module):
        """Test BUY signal on high positive consensus"""
        consensus = {
            "bullish_count": 4,
            "bearish_count": 0,
            "neutral_count": 1,
            "score": 50
        }

        signal, confidence = signal_module._generate_final_signal(consensus, 60)

        assert "BUY" in signal

    def test_sell_signal_on_negative_consensus(self, signal_module):
        """Test SELL signal on negative consensus"""
        consensus = {
            "bullish_count": 0,
            "bearish_count": 4,
            "neutral_count": 1,
            "score": -50
        }

        signal, confidence = signal_module._generate_final_signal(consensus, 60)

        assert "SELL" in signal

    def test_hold_signal_on_low_confidence(self, signal_module):
        """Test HOLD signal on low confidence"""
        consensus = {
            "bullish_count": 2,
            "bearish_count": 2,
            "neutral_count": 1,
            "score": 10
        }

        signal, confidence = signal_module._generate_final_signal(consensus, 80)

        assert "HOLD" in signal

    def test_strong_buy_on_very_high_consensus(self, signal_module):
        """Test STRONG BUY on very high consensus"""
        consensus = {
            "bullish_count": 5,
            "bearish_count": 0,
            "neutral_count": 0,
            "score": 70
        }

        signal, confidence = signal_module._generate_final_signal(consensus, 60)

        assert "STRONG" in signal and "BUY" in signal


class TestRiskAssessment:
    """Tests for risk assessment"""

    def test_risk_returns_dict(self, signal_module, sample_ohlcv):
        """Test risk assessment returns dictionary"""
        signals = {
            "momentum": {"signal": "bullish"},
            "mean_reversion": {"signal": "neutral"},
            "technical": {"signal": "bullish"}
        }

        result = signal_module._assess_risk(sample_ohlcv, signals)

        assert isinstance(result, dict)
        assert "overall_risk" in result
        assert "volatility" in result
        assert "signal_disagreement" in result

    def test_risk_overall_values(self, signal_module, sample_ohlcv):
        """Test overall risk has valid values"""
        signals = {"momentum": {"signal": "neutral"}}

        result = signal_module._assess_risk(sample_ohlcv, signals)

        assert result["overall_risk"] in ["HIGH", "MEDIUM", "LOW"]

    def test_high_risk_on_volatile_data(self, signal_module, volatile_data):
        """Test high risk on volatile data"""
        signals = {"momentum": {"signal": "neutral"}}

        result = signal_module._assess_risk(volatile_data, signals)

        # Volatile data should produce higher risk
        assert result["overall_risk"] in ["HIGH", "MEDIUM"]

    def test_signal_disagreement_high_on_mixed(self, signal_module, sample_ohlcv):
        """Test signal disagreement is high on mixed signals"""
        signals = {
            "momentum": {"signal": "bullish"},
            "mean_reversion": {"signal": "bearish"},
            "technical": {"signal": "neutral"}
        }

        result = signal_module._assess_risk(sample_ohlcv, signals)

        assert result["signal_disagreement"] == "high"


class TestInsightGeneration:
    """Tests for insight generation"""

    def test_insights_returns_list(self, signal_module):
        """Test insights returns list"""
        signals = {
            "momentum": {"signal": "bullish"},
            "mean_reversion": {"signal": "bullish"},
            "technical": {"signal": "neutral"},
            "pattern": {"signal": "neutral"},
            "volume": {"signal": "neutral"}
        }
        consensus = {"bullish_count": 2, "bearish_count": 0, "neutral_count": 3, "score": 30}
        risk = {"overall_risk": "LOW"}

        result = signal_module._generate_insights(signals, consensus, risk)

        assert isinstance(result, list)

    def test_high_agreement_insight(self, signal_module):
        """Test insight on high agreement"""
        signals = {
            "momentum": {"signal": "bullish"},
            "mean_reversion": {"signal": "bullish"},
            "technical": {"signal": "bullish"},
            "pattern": {"signal": "bullish"},
            "volume": {"signal": "neutral"}
        }
        consensus = {"bullish_count": 4, "bearish_count": 0, "neutral_count": 1, "score": 60}
        risk = {"overall_risk": "LOW"}

        result = signal_module._generate_insights(signals, consensus, risk)

        assert any("agreement" in insight.lower() for insight in result)

    def test_high_risk_warning(self, signal_module):
        """Test warning on high risk"""
        signals = {"momentum": {"signal": "neutral"}}
        consensus = {"bullish_count": 0, "bearish_count": 0, "neutral_count": 1, "score": 0}
        risk = {"overall_risk": "HIGH"}

        result = signal_module._generate_insights(signals, consensus, risk)

        assert any("high risk" in insight.lower() or "volatility" in insight.lower() for insight in result)

    def test_mixed_signals_insight(self, signal_module):
        """Test insight on mixed signals"""
        signals = {
            "momentum": {"signal": "bullish"},
            "mean_reversion": {"signal": "bearish"},
            "technical": {"signal": "neutral"},
            "pattern": {"signal": "neutral"},
            "volume": {"signal": "neutral"}
        }
        consensus = {"bullish_count": 1, "bearish_count": 1, "neutral_count": 3, "score": 10}
        risk = {"overall_risk": "LOW"}

        result = signal_module._generate_insights(signals, consensus, risk)

        assert any("mixed" in insight.lower() for insight in result)


class TestStrategySignalsInResults:
    """Tests for strategy signals in run results"""

    def test_all_strategy_signals_present(self, signal_module, sample_ohlcv):
        """Test all strategy signals are present in results"""
        signal_module.set_option("SYMBOL", "AAPL")

        with patch('quantsploit.modules.analysis.signal_aggregator.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = signal_module.run()

        strategy_signals = results["strategy_signals"]

        assert "momentum" in strategy_signals
        assert "mean_reversion" in strategy_signals
        assert "technical" in strategy_signals
        assert "pattern" in strategy_signals
        assert "volume" in strategy_signals

    def test_strategy_signals_have_required_fields(self, signal_module, sample_ohlcv):
        """Test strategy signals have required fields"""
        signal_module.set_option("SYMBOL", "AAPL")

        with patch('quantsploit.modules.analysis.signal_aggregator.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = signal_module.run()

        for name, sig in results["strategy_signals"].items():
            assert "signal" in sig
            assert "score" in sig
            assert "detail" in sig


class TestEdgeCases:
    """Tests for edge cases"""

    def test_minimal_data(self, signal_module):
        """Test with minimal data points"""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'Open': [100] * 30,
            'High': [101] * 30,
            'Low': [99] * 30,
            'Close': [100.5] * 30,
            'Volume': [1000000] * 30
        }, index=dates)

        signal_module.set_option("SYMBOL", "AAPL")

        with patch('quantsploit.modules.analysis.signal_aggregator.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = df

            results = signal_module.run()

        assert "final_signal" in results

    def test_zero_volume(self, signal_module):
        """Test handling of zero volume"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        close = [100 + i * 0.1 for i in range(50)]

        df = pd.DataFrame({
            'Open': [c * 0.99 for c in close],
            'High': [c * 1.01 for c in close],
            'Low': [c * 0.98 for c in close],
            'Close': close,
            'Volume': [0] * 50
        }, index=dates)

        # This may produce warnings but should not crash
        result = signal_module._get_volume_signals(df)
        assert isinstance(result, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
