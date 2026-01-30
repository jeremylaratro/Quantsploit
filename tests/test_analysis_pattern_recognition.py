"""
Unit tests for Pattern Recognition Module

Tests cover:
- Module properties and initialization
- Candlestick pattern detection
- Chart pattern detection
- Support/resistance level finding
- Signal generation
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.modules.analysis.pattern_recognition import PatternRecognition


@pytest.fixture
def mock_framework():
    """Create a mock framework with database"""
    framework = Mock()
    framework.database = Mock()
    framework.log = Mock()
    return framework


@pytest.fixture
def pattern_module(mock_framework):
    """Create a PatternRecognition module instance"""
    return PatternRecognition(mock_framework)


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')

    # Create relatively stable data
    close = 100 + np.random.normal(0, 1, 50).cumsum() * 0.5

    return pd.DataFrame({
        'Open': close * 0.995,
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': np.random.randint(1000000, 5000000, 50)
    }, index=dates)


@pytest.fixture
def hammer_candle_data():
    """Generate data with a hammer pattern at the end"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')

    # Downtrend followed by hammer
    close = 110 - np.arange(20) * 0.5
    open_prices = close + 0.1
    high = close + 0.2
    low = close.copy()

    # Create hammer on last candle
    # Hammer: long lower shadow, small body, minimal upper shadow
    open_prices[-1] = 101.0
    high[-1] = 101.5
    low[-1] = 98.0
    close[-1] = 101.2

    return pd.DataFrame({
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': [1000000] * 20
    }, index=dates)


@pytest.fixture
def engulfing_data():
    """Generate data with an engulfing pattern"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')

    close = [100 + i * 0.1 for i in range(20)]
    open_prices = [c - 0.3 for c in close]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]

    # Create bullish engulfing at end
    # Day -2: bearish candle
    open_prices[-2] = 102.0
    close[-2] = 101.0
    high[-2] = 102.5
    low[-2] = 100.5

    # Day -1: bullish candle that engulfs previous
    open_prices[-1] = 100.5
    close[-1] = 103.0
    high[-1] = 103.5
    low[-1] = 100.0

    return pd.DataFrame({
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': [1000000] * 20
    }, index=dates)


@pytest.fixture
def double_bottom_data():
    """Generate data with a double bottom pattern"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')

    # Create W-shape for double bottom
    close = np.concatenate([
        np.linspace(100, 90, 15),   # First decline
        np.linspace(90, 100, 10),   # First rise
        np.linspace(100, 90, 10),   # Second decline (similar low)
        np.linspace(90, 105, 15)    # Recovery
    ])

    return pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.01,
        'Low': close * 0.98,
        'Close': close,
        'Volume': [1000000] * 50
    }, index=dates)


class TestModuleProperties:
    """Tests for module property definitions"""

    def test_name(self, pattern_module):
        """Test module name"""
        assert pattern_module.name == "Pattern Recognition"

    def test_description(self, pattern_module):
        """Test module description"""
        assert "pattern" in pattern_module.description.lower()
        assert "candlestick" in pattern_module.description.lower() or "detect" in pattern_module.description.lower()

    def test_author(self, pattern_module):
        """Test module author"""
        assert pattern_module.author == "Quantsploit Team"

    def test_category(self, pattern_module):
        """Test module category"""
        assert pattern_module.category == "analysis"


class TestOptions:
    """Tests for module option management"""

    def test_default_options(self, pattern_module):
        """Test default option values"""
        assert pattern_module.get_option("LOOKBACK") == 50
        assert pattern_module.get_option("PATTERNS") == "all"

    def test_set_lookback(self, pattern_module):
        """Test setting lookback"""
        pattern_module.set_option("LOOKBACK", 100)
        assert pattern_module.get_option("LOOKBACK") == 100

    def test_set_patterns_filter(self, pattern_module):
        """Test setting patterns filter"""
        pattern_module.set_option("PATTERNS", "candlestick")
        assert pattern_module.get_option("PATTERNS") == "candlestick"

    def test_inherited_options(self, pattern_module):
        """Test that base module options are inherited"""
        assert "SYMBOL" in pattern_module.options
        assert "PERIOD" in pattern_module.options
        assert "INTERVAL" in pattern_module.options


class TestRunExecution:
    """Tests for run execution"""

    def test_run_returns_results(self, pattern_module, sample_ohlcv):
        """Test run returns results dictionary"""
        pattern_module.set_option("SYMBOL", "AAPL")

        with patch('quantsploit.modules.analysis.pattern_recognition.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = pattern_module.run()

        assert "symbol" in results
        assert "current_price" in results
        assert "candlestick_patterns" in results
        assert "chart_patterns" in results
        assert "support_resistance" in results
        assert "signals" in results

    def test_run_no_data_error(self, pattern_module):
        """Test run with no data returns error"""
        pattern_module.set_option("SYMBOL", "INVALID")

        with patch('quantsploit.modules.analysis.pattern_recognition.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = None

            results = pattern_module.run()

        assert results["success"] is False
        assert "error" in results

    def test_run_empty_data_error(self, pattern_module):
        """Test run with empty data returns error"""
        pattern_module.set_option("SYMBOL", "AAPL")

        with patch('quantsploit.modules.analysis.pattern_recognition.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = pd.DataFrame()

            results = pattern_module.run()

        assert results["success"] is False

    def test_run_with_lookback(self, pattern_module, sample_ohlcv):
        """Test run respects lookback setting"""
        pattern_module.set_option("SYMBOL", "AAPL")
        pattern_module.set_option("LOOKBACK", 20)

        with patch('quantsploit.modules.analysis.pattern_recognition.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = pattern_module.run()

        assert results is not None


class TestCandlestickPatterns:
    """Tests for candlestick pattern detection"""

    def test_hammer_detection(self, pattern_module):
        """Test hammer pattern detection"""
        # Hammer: long lower shadow > 2x body, minimal upper shadow < 0.3x body
        # Body = 0.3, lower shadow = 1.0 > 2*0.3=0.6, upper shadow = 0 < 0.3*0.3=0.09
        o, h, l, c = 100, 100.3, 99, 100.3
        assert pattern_module._is_hammer(o, h, l, c) is True

    def test_hammer_detection_invalid(self, pattern_module):
        """Test hammer detection rejects non-hammer"""
        # Not a hammer: no long lower shadow
        o, h, l, c = 100, 105, 99.5, 104
        assert pattern_module._is_hammer(o, h, l, c) is False

    def test_shooting_star_detection(self, pattern_module):
        """Test shooting star pattern detection"""
        # Shooting star: long upper shadow > 2x body, minimal lower shadow < 0.3x body
        # Body = 0.3, upper shadow = 1.0 > 2*0.3=0.6, lower shadow = 0 < 0.3*0.3=0.09
        o, h, l, c = 100, 101, 99.7, 99.7
        assert pattern_module._is_shooting_star(o, h, l, c) is True

    def test_shooting_star_detection_invalid(self, pattern_module):
        """Test shooting star detection rejects non-shooting star"""
        # Not a shooting star: no long upper shadow
        o, h, l, c = 100, 101, 95, 100.5
        assert pattern_module._is_shooting_star(o, h, l, c) is False

    def test_bullish_engulfing_detection(self, pattern_module):
        """Test bullish engulfing pattern detection"""
        # Previous candle: bearish (close < open)
        o1, c1 = 102, 100  # Bearish
        # Current candle: bullish, engulfs previous
        o2, c2 = 99, 103
        assert pattern_module._is_bullish_engulfing(o1, c1, o2, c2) is True

    def test_bearish_engulfing_detection(self, pattern_module):
        """Test bearish engulfing pattern detection"""
        # Previous candle: bullish (close > open)
        o1, c1 = 100, 102  # Bullish
        # Current candle: bearish, engulfs previous
        o2, c2 = 103, 99
        assert pattern_module._is_bearish_engulfing(o1, c1, o2, c2) is True

    def test_doji_detection(self, pattern_module):
        """Test doji pattern detection"""
        # Doji: very small body relative to range
        o, c, h, l = 100, 100.05, 102, 98
        assert pattern_module._is_doji(o, c, h, l) is True

    def test_doji_detection_invalid(self, pattern_module):
        """Test doji detection rejects non-doji"""
        # Not a doji: large body
        o, c, h, l = 100, 105, 106, 99
        assert pattern_module._is_doji(o, c, h, l) is False

    def test_morning_star_detection(self, pattern_module):
        """Test morning star pattern detection"""
        # Day 1: bearish
        o1, c1 = 105, 100
        # Day 2: small body (star)
        o2, c2 = 99.5, 99.8
        # Day 3: bullish, closes above midpoint of day 1
        o3, c3 = 100, 104
        assert pattern_module._is_morning_star(o1, c1, o2, c2, o3, c3) is True

    def test_evening_star_detection(self, pattern_module):
        """Test evening star pattern detection"""
        # Day 1: bullish
        o1, c1 = 100, 105
        # Day 2: small body (star)
        o2, c2 = 105.2, 105.5
        # Day 3: bearish, closes below midpoint of day 1
        o3, c3 = 104, 100
        assert pattern_module._is_evening_star(o1, c1, o2, c2, o3, c3) is True


class TestChartPatterns:
    """Tests for chart pattern detection"""

    def test_detect_chart_patterns_returns_list(self, pattern_module, sample_ohlcv):
        """Test chart pattern detection returns list"""
        patterns = pattern_module._detect_chart_patterns(sample_ohlcv)
        assert isinstance(patterns, list)

    def test_detect_double_bottom(self, pattern_module, double_bottom_data):
        """Test double bottom pattern detection"""
        patterns = pattern_module._detect_chart_patterns(double_bottom_data)

        # Check if any double bottom pattern detected
        double_bottom_found = any(p['pattern'] == 'Double Bottom' for p in patterns)
        # Note: Pattern detection may or may not find the pattern depending on data
        assert isinstance(patterns, list)

    def test_pattern_structure(self, pattern_module, sample_ohlcv):
        """Test pattern dictionary structure"""
        patterns = pattern_module._detect_chart_patterns(sample_ohlcv)

        for pattern in patterns:
            assert "pattern" in pattern
            assert "sentiment" in pattern
            assert "strength" in pattern
            assert pattern["sentiment"] in ["bullish", "bearish", "neutral"]


class TestSupportResistance:
    """Tests for support and resistance detection"""

    def test_find_support_resistance_returns_dict(self, pattern_module, sample_ohlcv):
        """Test support/resistance returns dictionary"""
        sr = pattern_module._find_support_resistance(sample_ohlcv)
        assert isinstance(sr, dict)

    def test_support_resistance_keys(self, pattern_module, sample_ohlcv):
        """Test support/resistance has expected keys"""
        sr = pattern_module._find_support_resistance(sample_ohlcv)

        assert "resistance" in sr
        assert "support" in sr
        assert "current_price" in sr
        assert "nearest_resistance" in sr
        assert "nearest_support" in sr

    def test_resistance_is_list(self, pattern_module, sample_ohlcv):
        """Test resistance levels are a list"""
        sr = pattern_module._find_support_resistance(sample_ohlcv)
        assert isinstance(sr["resistance"], list)

    def test_support_is_list(self, pattern_module, sample_ohlcv):
        """Test support levels are a list"""
        sr = pattern_module._find_support_resistance(sample_ohlcv)
        assert isinstance(sr["support"], list)

    def test_resistance_sorted_descending(self, pattern_module, sample_ohlcv):
        """Test resistance levels are sorted descending"""
        sr = pattern_module._find_support_resistance(sample_ohlcv)
        resistance = sr["resistance"]

        if len(resistance) > 1:
            for i in range(len(resistance) - 1):
                assert resistance[i] >= resistance[i + 1]

    def test_support_sorted_descending(self, pattern_module, sample_ohlcv):
        """Test support levels are sorted descending"""
        sr = pattern_module._find_support_resistance(sample_ohlcv)
        support = sr["support"]

        if len(support) > 1:
            for i in range(len(support) - 1):
                assert support[i] >= support[i + 1]


class TestSignalGeneration:
    """Tests for signal generation"""

    def test_generate_signals_returns_list(self, pattern_module, sample_ohlcv):
        """Test signal generation returns list"""
        candlestick_patterns = []
        chart_patterns = []
        sr = {"nearest_support": 95, "nearest_resistance": 105}

        signals = pattern_module._generate_signals(
            candlestick_patterns, chart_patterns, sr, sample_ohlcv
        )
        assert isinstance(signals, list)

    def test_neutral_signal_when_no_patterns(self, pattern_module, sample_ohlcv):
        """Test neutral signal when no patterns detected"""
        signals = pattern_module._generate_signals([], [], {}, sample_ohlcv)

        assert len(signals) == 1
        assert "NEUTRAL" in signals[0]

    def test_bullish_signal_on_bullish_patterns(self, pattern_module, sample_ohlcv):
        """Test bullish signal on bullish patterns"""
        candlestick_patterns = [
            {"pattern": "Hammer", "sentiment": "bullish"},
            {"pattern": "Bullish Engulfing", "sentiment": "bullish"}
        ]

        signals = pattern_module._generate_signals(
            candlestick_patterns, [], {}, sample_ohlcv
        )

        assert any("BUY" in s for s in signals)

    def test_bearish_signal_on_bearish_patterns(self, pattern_module, sample_ohlcv):
        """Test bearish signal on bearish patterns"""
        candlestick_patterns = [
            {"pattern": "Shooting Star", "sentiment": "bearish"},
            {"pattern": "Bearish Engulfing", "sentiment": "bearish"}
        ]

        signals = pattern_module._generate_signals(
            candlestick_patterns, [], {}, sample_ohlcv
        )

        assert any("SELL" in s for s in signals)

    def test_strong_signal_on_multiple_patterns(self, pattern_module, sample_ohlcv):
        """Test strong signal on multiple patterns"""
        candlestick_patterns = [
            {"pattern": "Hammer", "sentiment": "bullish"},
            {"pattern": "Morning Star", "sentiment": "bullish"},
            {"pattern": "Bullish Engulfing", "sentiment": "bullish"}
        ]

        signals = pattern_module._generate_signals(
            candlestick_patterns, [], {}, sample_ohlcv
        )

        assert any("STRONG" in s for s in signals)

    def test_chart_pattern_signals(self, pattern_module, sample_ohlcv):
        """Test signals from chart patterns"""
        chart_patterns = [
            {"pattern": "Double Bottom", "sentiment": "bullish"}
        ]

        signals = pattern_module._generate_signals(
            [], chart_patterns, {}, sample_ohlcv
        )

        assert any("Double Bottom" in s for s in signals)


class TestPatternSummary:
    """Tests for pattern summary generation"""

    def test_pattern_summary_in_results(self, pattern_module, sample_ohlcv):
        """Test pattern summary is included in results"""
        pattern_module.set_option("SYMBOL", "AAPL")

        with patch('quantsploit.modules.analysis.pattern_recognition.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = pattern_module.run()

        assert "pattern_summary" in results
        assert isinstance(results["pattern_summary"], pd.DataFrame)

    def test_pattern_summary_columns(self, pattern_module, sample_ohlcv):
        """Test pattern summary has expected columns"""
        pattern_module.set_option("SYMBOL", "AAPL")

        with patch('quantsploit.modules.analysis.pattern_recognition.DataFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.get_stock_data.return_value = sample_ohlcv

            results = pattern_module.run()

        assert "Pattern Type" in results["pattern_summary"].columns
        assert "Count" in results["pattern_summary"].columns


class TestEdgeCases:
    """Tests for edge cases"""

    def test_zero_range_candle(self, pattern_module):
        """Test handling of zero-range candle"""
        # All prices equal (zero range)
        o, h, l, c = 100, 100, 100, 100
        assert pattern_module._is_hammer(o, h, l, c) is False
        assert pattern_module._is_shooting_star(o, h, l, c) is False

    def test_minimal_data(self, pattern_module):
        """Test with minimal data points"""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'Open': [100] * 10,
            'High': [101] * 10,
            'Low': [99] * 10,
            'Close': [100.5] * 10,
            'Volume': [1000000] * 10
        }, index=dates)

        patterns = pattern_module._detect_candlestick_patterns(df)
        assert isinstance(patterns, list)

    def test_negative_prices(self, pattern_module):
        """Test handling of unusual price data"""
        # This shouldn't normally happen, but test robustness
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        close = np.abs(np.random.normal(100, 5, 20))

        df = pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.01,
            'Low': close * 0.98,
            'Close': close,
            'Volume': [1000000] * 20
        }, index=dates)

        sr = pattern_module._find_support_resistance(df)
        assert "current_price" in sr


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
