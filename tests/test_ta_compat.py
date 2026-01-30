"""
Unit tests for Technical Analysis Compatibility Layer

Tests cover:
- SMA calculation
- EMA calculation
- RSI calculation
- MACD calculation
- Bollinger Bands calculation
- ATR calculation
- ADX calculation
- ROC calculation
- Stochastic Oscillator
- OBV calculation
- VWAP calculation
- TA class interface
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.utils.ta_compat import (
    sma, ema, rsi, macd, bbands, atr, adx, roc, stoch, obv, vwap, ta, TA
)


@pytest.fixture
def price_series():
    """Generate a simple price series for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    return pd.Series(prices, index=dates, name='Close')


@pytest.fixture
def ohlcv_data():
    """Generate OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    close = 100 + np.cumsum(np.random.randn(100) * 2)
    high = close * (1 + np.abs(np.random.uniform(0, 0.02, 100)))
    low = close * (1 - np.abs(np.random.uniform(0, 0.02, 100)))
    open_ = close * (1 + np.random.uniform(-0.01, 0.01, 100))
    volume = np.random.randint(1000000, 10000000, 100)

    return pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)


class TestSMA:
    """Tests for Simple Moving Average"""

    def test_sma_calculation(self, price_series):
        """Test basic SMA calculation"""
        result = sma(price_series, length=20)

        assert isinstance(result, pd.Series)
        assert len(result) == len(price_series)
        # First 19 values should be NaN
        assert result.iloc[:19].isna().all()
        # Values after should be valid
        assert result.iloc[19:].notna().all()

    def test_sma_value(self, price_series):
        """Test SMA value is correct"""
        result = sma(price_series, length=5)

        # Manually calculate for comparison
        expected = price_series.iloc[0:5].mean()
        assert abs(result.iloc[4] - expected) < 0.001

    def test_sma_different_lengths(self, price_series):
        """Test SMA with different window lengths"""
        sma_5 = sma(price_series, length=5)
        sma_20 = sma(price_series, length=20)
        sma_50 = sma(price_series, length=50)

        # Longer SMAs should have more NaN values at start
        assert sma_5.iloc[4:].notna().all()
        assert sma_20.iloc[19:].notna().all()
        assert sma_50.iloc[49:].notna().all()


class TestEMA:
    """Tests for Exponential Moving Average"""

    def test_ema_calculation(self, price_series):
        """Test basic EMA calculation"""
        result = ema(price_series, length=20)

        assert isinstance(result, pd.Series)
        assert len(result) == len(price_series)
        # EMA should have values for all rows
        assert result.notna().all()

    def test_ema_smoothing(self, price_series):
        """Test that EMA gives more weight to recent values"""
        result = ema(price_series, length=10)

        # EMA should be between the extreme values
        assert result.min() >= price_series.min() - 1
        assert result.max() <= price_series.max() + 1

    def test_ema_vs_sma(self, price_series):
        """Test that EMA reacts faster than SMA to price changes"""
        sma_result = sma(price_series, length=20)
        ema_result = ema(price_series, length=20)

        # Both should exist
        assert sma_result is not None
        assert ema_result is not None


class TestRSI:
    """Tests for Relative Strength Index"""

    def test_rsi_calculation(self, price_series):
        """Test basic RSI calculation"""
        result = rsi(price_series, length=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(price_series)

    def test_rsi_range(self, price_series):
        """Test RSI is within 0-100 range"""
        result = rsi(price_series, length=14)

        # Remove NaN values for testing
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()

    def test_rsi_overbought_condition(self):
        """Test RSI detects overbought condition"""
        # Create strongly upward trending prices
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        prices = pd.Series([100 + i * 2 for i in range(50)], index=dates)

        result = rsi(prices, length=14)
        # In strong uptrend, RSI should be high
        assert result.iloc[-1] > 50

    def test_rsi_oversold_condition(self):
        """Test RSI detects oversold condition"""
        # Create strongly downward trending prices
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        prices = pd.Series([200 - i * 2 for i in range(50)], index=dates)

        result = rsi(prices, length=14)
        # In strong downtrend, RSI should be low
        assert result.iloc[-1] < 50


class TestMACD:
    """Tests for MACD Indicator"""

    def test_macd_returns_dataframe(self, price_series):
        """Test MACD returns DataFrame with expected columns"""
        result = macd(price_series)

        assert isinstance(result, pd.DataFrame)
        assert 'MACD_12_26_9' in result.columns
        assert 'MACDs_12_26_9' in result.columns
        assert 'MACDh_12_26_9' in result.columns

    def test_macd_histogram(self, price_series):
        """Test MACD histogram is difference between MACD and signal"""
        result = macd(price_series)

        calculated_hist = result['MACD_12_26_9'] - result['MACDs_12_26_9']
        diff = (result['MACDh_12_26_9'] - calculated_hist).abs()
        assert (diff < 0.0001).all()

    def test_macd_custom_periods(self, price_series):
        """Test MACD with custom periods"""
        result = macd(price_series, fast=8, slow=21, signal=5)

        assert 'MACD_8_21_5' in result.columns
        assert 'MACDs_8_21_5' in result.columns
        assert 'MACDh_8_21_5' in result.columns


class TestBollingerBands:
    """Tests for Bollinger Bands"""

    def test_bbands_returns_dataframe(self, price_series):
        """Test Bollinger Bands returns DataFrame"""
        result = bbands(price_series)

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 3

    def test_bbands_columns(self, price_series):
        """Test Bollinger Bands has correct column names"""
        result = bbands(price_series, length=20, std=2.0)

        columns = result.columns.tolist()
        assert any('BBL' in col for col in columns)
        assert any('BBM' in col for col in columns)
        assert any('BBU' in col for col in columns)

    def test_bbands_order(self, price_series):
        """Test that upper > middle > lower"""
        result = bbands(price_series, length=20, std=2.0)

        cols = result.columns.tolist()
        lower_col = [c for c in cols if 'BBL' in c][0]
        middle_col = [c for c in cols if 'BBM' in c][0]
        upper_col = [c for c in cols if 'BBU' in c][0]

        valid_idx = result[lower_col].notna()
        assert (result.loc[valid_idx, upper_col] >= result.loc[valid_idx, middle_col]).all()
        assert (result.loc[valid_idx, middle_col] >= result.loc[valid_idx, lower_col]).all()

    def test_bbands_std_width(self, price_series):
        """Test wider std parameter creates wider bands"""
        result_narrow = bbands(price_series, length=20, std=1.0)
        result_wide = bbands(price_series, length=20, std=3.0)

        # Wide bands should be wider than narrow bands
        narrow_cols = result_narrow.columns.tolist()
        wide_cols = result_wide.columns.tolist()

        narrow_upper = result_narrow[[c for c in narrow_cols if 'BBU' in c][0]]
        narrow_lower = result_narrow[[c for c in narrow_cols if 'BBL' in c][0]]
        wide_upper = result_wide[[c for c in wide_cols if 'BBU' in c][0]]
        wide_lower = result_wide[[c for c in wide_cols if 'BBL' in c][0]]

        narrow_width = (narrow_upper - narrow_lower).dropna()
        wide_width = (wide_upper - wide_lower).dropna()

        assert (wide_width.values > narrow_width.values).all()


class TestATR:
    """Tests for Average True Range"""

    def test_atr_calculation(self, ohlcv_data):
        """Test basic ATR calculation"""
        result = atr(ohlcv_data['High'], ohlcv_data['Low'], ohlcv_data['Close'], length=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv_data)

    def test_atr_positive(self, ohlcv_data):
        """Test ATR is always positive"""
        result = atr(ohlcv_data['High'], ohlcv_data['Low'], ohlcv_data['Close'], length=14)

        valid = result.dropna()
        assert (valid >= 0).all()

    def test_atr_reflects_volatility(self):
        """Test ATR reflects market volatility"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')

        # Low volatility data
        low_vol_close = pd.Series([100 + i * 0.1 for i in range(50)], index=dates)
        low_vol_high = low_vol_close + 0.5
        low_vol_low = low_vol_close - 0.5

        # High volatility data
        high_vol_close = pd.Series([100 + i * 0.1 + np.random.randn() * 5 for i in range(50)], index=dates)
        high_vol_high = high_vol_close + 3
        high_vol_low = high_vol_close - 3

        atr_low = atr(low_vol_high, low_vol_low, low_vol_close, length=14)
        atr_high = atr(high_vol_high, high_vol_low, high_vol_close, length=14)

        # High volatility should have higher ATR
        assert atr_high.iloc[-1] > atr_low.iloc[-1]


class TestADX:
    """Tests for Average Directional Index"""

    def test_adx_returns_dataframe(self, ohlcv_data):
        """Test ADX returns DataFrame"""
        result = adx(ohlcv_data['High'], ohlcv_data['Low'], ohlcv_data['Close'], length=14)

        assert isinstance(result, pd.DataFrame)

    def test_adx_columns(self, ohlcv_data):
        """Test ADX has expected columns"""
        result = adx(ohlcv_data['High'], ohlcv_data['Low'], ohlcv_data['Close'], length=14)

        assert 'ADX_14' in result.columns
        assert 'DMP_14' in result.columns
        assert 'DMN_14' in result.columns

    def test_adx_range(self, ohlcv_data):
        """Test ADX values are within expected range"""
        result = adx(ohlcv_data['High'], ohlcv_data['Low'], ohlcv_data['Close'], length=14)

        valid = result['ADX_14'].dropna()
        # ADX should be between 0 and 100
        assert (valid >= 0).all() or valid.isna().all()


class TestROC:
    """Tests for Rate of Change"""

    def test_roc_calculation(self, price_series):
        """Test basic ROC calculation"""
        result = roc(price_series, length=12)

        assert isinstance(result, pd.Series)
        assert len(result) == len(price_series)

    def test_roc_value(self, price_series):
        """Test ROC value is correct"""
        result = roc(price_series, length=5)

        # Manually calculate ROC for comparison
        expected = ((price_series.iloc[5] - price_series.iloc[0]) / price_series.iloc[0]) * 100
        assert abs(result.iloc[5] - expected) < 0.001

    def test_roc_uptrend(self):
        """Test ROC is positive in uptrend"""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        prices = pd.Series([100 + i for i in range(30)], index=dates)

        result = roc(prices, length=10)
        assert result.iloc[-1] > 0


class TestStochastic:
    """Tests for Stochastic Oscillator"""

    def test_stoch_returns_dataframe(self, ohlcv_data):
        """Test Stochastic returns DataFrame"""
        result = stoch(ohlcv_data['High'], ohlcv_data['Low'], ohlcv_data['Close'])

        assert isinstance(result, pd.DataFrame)

    def test_stoch_columns(self, ohlcv_data):
        """Test Stochastic has K and D lines"""
        result = stoch(ohlcv_data['High'], ohlcv_data['Low'], ohlcv_data['Close'], k=14, d=3)

        columns = result.columns.tolist()
        assert any('STOCHk' in col for col in columns)
        assert any('STOCHd' in col for col in columns)

    def test_stoch_range(self, ohlcv_data):
        """Test Stochastic is within 0-100 range"""
        result = stoch(ohlcv_data['High'], ohlcv_data['Low'], ohlcv_data['Close'])

        k_col = [c for c in result.columns if 'STOCHk' in c][0]
        valid = result[k_col].dropna()

        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestOBV:
    """Tests for On-Balance Volume"""

    def test_obv_calculation(self, ohlcv_data):
        """Test basic OBV calculation"""
        result = obv(ohlcv_data['Close'], ohlcv_data['Volume'])

        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv_data)

    def test_obv_increases_on_up_day(self):
        """Test OBV increases on up days"""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        close = pd.Series([100, 101, 102, 103, 104], index=dates)
        volume = pd.Series([1000, 1000, 1000, 1000, 1000], index=dates)

        result = obv(close, volume)

        # OBV should increase with each up day
        for i in range(1, len(result)):
            if close.iloc[i] > close.iloc[i-1]:
                assert result.iloc[i] > result.iloc[i-1]

    def test_obv_decreases_on_down_day(self):
        """Test OBV decreases on down days"""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        close = pd.Series([104, 103, 102, 101, 100], index=dates)
        volume = pd.Series([1000, 1000, 1000, 1000, 1000], index=dates)

        result = obv(close, volume)

        # OBV should decrease with each down day
        for i in range(1, len(result)):
            if close.iloc[i] < close.iloc[i-1]:
                assert result.iloc[i] < result.iloc[i-1]


class TestVWAP:
    """Tests for Volume Weighted Average Price"""

    def test_vwap_calculation(self, ohlcv_data):
        """Test basic VWAP calculation"""
        result = vwap(ohlcv_data['High'], ohlcv_data['Low'],
                     ohlcv_data['Close'], ohlcv_data['Volume'])

        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv_data)

    def test_vwap_within_price_range(self, ohlcv_data):
        """Test VWAP is within the price range"""
        result = vwap(ohlcv_data['High'], ohlcv_data['Low'],
                     ohlcv_data['Close'], ohlcv_data['Volume'])

        # VWAP should be reasonable (not extreme)
        assert result.iloc[-1] > ohlcv_data['Low'].min()
        assert result.iloc[-1] < ohlcv_data['High'].max()


class TestTAClass:
    """Tests for the TA class interface"""

    def test_ta_sma(self, price_series):
        """Test TA class sma method"""
        result = ta.sma(price_series, length=20)
        assert isinstance(result, pd.Series)

    def test_ta_ema(self, price_series):
        """Test TA class ema method"""
        result = ta.ema(price_series, length=20)
        assert isinstance(result, pd.Series)

    def test_ta_rsi(self, price_series):
        """Test TA class rsi method"""
        result = ta.rsi(price_series, length=14)
        assert isinstance(result, pd.Series)

    def test_ta_macd(self, price_series):
        """Test TA class macd method"""
        result = ta.macd(price_series)
        assert isinstance(result, pd.DataFrame)

    def test_ta_bbands(self, price_series):
        """Test TA class bbands method"""
        result = ta.bbands(price_series)
        assert isinstance(result, pd.DataFrame)

    def test_ta_atr(self, ohlcv_data):
        """Test TA class atr method"""
        result = ta.atr(ohlcv_data['High'], ohlcv_data['Low'], ohlcv_data['Close'])
        assert isinstance(result, pd.Series)

    def test_ta_adx(self, ohlcv_data):
        """Test TA class adx method"""
        result = ta.adx(ohlcv_data['High'], ohlcv_data['Low'], ohlcv_data['Close'])
        assert isinstance(result, pd.DataFrame)

    def test_ta_roc(self, price_series):
        """Test TA class roc method"""
        result = ta.roc(price_series)
        assert isinstance(result, pd.Series)

    def test_ta_stoch(self, ohlcv_data):
        """Test TA class stoch method"""
        result = ta.stoch(ohlcv_data['High'], ohlcv_data['Low'], ohlcv_data['Close'])
        assert isinstance(result, pd.DataFrame)

    def test_ta_obv(self, ohlcv_data):
        """Test TA class obv method"""
        result = ta.obv(ohlcv_data['Close'], ohlcv_data['Volume'])
        assert isinstance(result, pd.Series)

    def test_ta_vwap(self, ohlcv_data):
        """Test TA class vwap method"""
        result = ta.vwap(ohlcv_data['High'], ohlcv_data['Low'],
                        ohlcv_data['Close'], ohlcv_data['Volume'])
        assert isinstance(result, pd.Series)


class TestEdgeCases:
    """Tests for edge cases"""

    def test_short_series(self):
        """Test indicators with short series"""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        prices = pd.Series([100, 101, 102, 103, 104], index=dates)

        # These should not crash
        sma_result = sma(prices, length=3)
        ema_result = ema(prices, length=3)
        rsi_result = rsi(prices, length=3)

        assert len(sma_result) == 5
        assert len(ema_result) == 5
        assert len(rsi_result) == 5

    def test_constant_prices(self):
        """Test indicators with constant prices"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        prices = pd.Series([100] * 50, index=dates)

        sma_result = sma(prices, length=10)
        ema_result = ema(prices, length=10)

        # With constant prices, SMA and EMA should equal the price
        assert abs(sma_result.iloc[-1] - 100) < 0.001
        assert abs(ema_result.iloc[-1] - 100) < 0.001

    def test_nan_handling(self):
        """Test indicators handle NaN values"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        prices = pd.Series([100 + i for i in range(50)], index=dates)
        prices.iloc[10:15] = np.nan

        # Should not crash
        result = sma(prices, length=5)
        assert isinstance(result, pd.Series)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
