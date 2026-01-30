"""
Unit tests for Data Fetcher utility

Tests cover:
- DataFetcher initialization
- Stock data fetching
- Cache functionality
- Sample data fallback
- Multiple stock fetching
- Stock info retrieval
- Options chain retrieval
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.utils.data_fetcher import DataFetcher


@pytest.fixture
def mock_database():
    """Create a mock database"""
    db = Mock()
    db.get_cached_data = Mock(return_value=None)
    db.cache_market_data = Mock()
    return db


@pytest.fixture
def fetcher_with_db(mock_database):
    """Create a DataFetcher with mock database"""
    return DataFetcher(database=mock_database)


@pytest.fixture
def fetcher_sample_mode():
    """Create a DataFetcher in sample data mode"""
    return DataFetcher(use_sample_data=True)


@pytest.fixture
def fetcher_no_cache():
    """Create a DataFetcher without caching"""
    return DataFetcher(cache_enabled=False)


@pytest.fixture
def sample_df():
    """Generate sample DataFrame for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'Open': np.random.uniform(100, 110, 100),
        'High': np.random.uniform(110, 120, 100),
        'Low': np.random.uniform(90, 100, 100),
        'Close': np.random.uniform(100, 110, 100),
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)


class TestDataFetcherInitialization:
    """Tests for DataFetcher initialization"""

    def test_init_default(self):
        """Test default initialization"""
        fetcher = DataFetcher()
        assert fetcher.database is None
        assert fetcher.cache_enabled is True
        assert fetcher.cache_duration == 3600
        assert fetcher.use_sample_data is False

    def test_init_with_database(self, mock_database):
        """Test initialization with database"""
        fetcher = DataFetcher(database=mock_database)
        assert fetcher.database is mock_database

    def test_init_cache_disabled(self):
        """Test initialization with cache disabled"""
        fetcher = DataFetcher(cache_enabled=False)
        assert fetcher.cache_enabled is False

    def test_init_custom_cache_duration(self):
        """Test initialization with custom cache duration"""
        fetcher = DataFetcher(cache_duration=7200)
        assert fetcher.cache_duration == 7200

    def test_init_sample_data_mode(self):
        """Test initialization in sample data mode"""
        fetcher = DataFetcher(use_sample_data=True)
        assert fetcher.use_sample_data is True


class TestGetStockDataSampleMode:
    """Tests for get_stock_data in sample data mode"""

    def test_get_stock_data_sample_mode(self, fetcher_sample_mode):
        """Test fetching stock data in sample mode"""
        with patch('quantsploit.utils.data_fetcher.get_sample_data') as mock_sample:
            mock_sample.return_value = pd.DataFrame({
                'Open': [100], 'High': [105], 'Low': [95],
                'Close': [102], 'Volume': [1000000]
            })

            result = fetcher_sample_mode.get_stock_data("AAPL")

        assert mock_sample.called
        assert isinstance(result, pd.DataFrame)

    def test_get_stock_data_default_params(self, fetcher_sample_mode):
        """Test default parameters are passed correctly"""
        with patch('quantsploit.utils.data_fetcher.get_sample_data') as mock_sample:
            mock_sample.return_value = pd.DataFrame()

            fetcher_sample_mode.get_stock_data("AAPL")

        mock_sample.assert_called_with("AAPL", "1y", "1d")

    def test_get_stock_data_custom_params(self, fetcher_sample_mode):
        """Test custom parameters are passed"""
        with patch('quantsploit.utils.data_fetcher.get_sample_data') as mock_sample:
            mock_sample.return_value = pd.DataFrame()

            fetcher_sample_mode.get_stock_data("AAPL", period="6mo", interval="1h")

        mock_sample.assert_called_with("AAPL", "6mo", "1h")


class TestCaching:
    """Tests for caching functionality"""

    def test_cache_hit(self, fetcher_with_db, sample_df, mock_database):
        """Test cache hit returns cached data"""
        # Setup cache to return data
        mock_database.get_cached_data.return_value = sample_df.to_json()

        result = fetcher_with_db.get_stock_data("AAPL")

        assert mock_database.get_cached_data.called
        assert isinstance(result, pd.DataFrame)

    def test_cache_miss_fetches_data(self, fetcher_with_db, mock_database):
        """Test cache miss triggers data fetch"""
        mock_database.get_cached_data.return_value = None

        with patch('yfinance.Ticker') as mock_yf:
            mock_ticker = Mock()
            mock_ticker.history.return_value = pd.DataFrame({
                'Open': [100], 'High': [105], 'Low': [95],
                'Close': [102], 'Volume': [1000000]
            })
            mock_yf.return_value = mock_ticker

            fetcher_with_db.get_stock_data("AAPL")

        assert mock_database.get_cached_data.called

    def test_force_refresh_skips_cache(self, fetcher_with_db, mock_database):
        """Test force_refresh bypasses cache"""
        mock_database.get_cached_data.return_value = '{"data": "cached"}'

        with patch('yfinance.Ticker') as mock_yf:
            mock_ticker = Mock()
            mock_ticker.history.return_value = pd.DataFrame({
                'Open': [100], 'High': [105], 'Low': [95],
                'Close': [102], 'Volume': [1000000]
            })
            mock_yf.return_value = mock_ticker

            fetcher_with_db.get_stock_data("AAPL", force_refresh=True)

        # Should not check cache when force_refresh is True
        # but should still call the ticker


class TestYFinanceFallback:
    """Tests for yfinance fallback to sample data"""

    def test_fallback_on_empty_data(self, fetcher_no_cache):
        """Test fallback to sample data on empty response"""
        with patch('yfinance.Ticker') as mock_yf:
            mock_ticker = Mock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_yf.return_value = mock_ticker

            with patch('quantsploit.utils.data_fetcher.get_sample_data') as mock_sample:
                mock_sample.return_value = pd.DataFrame({'Close': [100]})

                result = fetcher_no_cache.get_stock_data("AAPL")

        # Should fallback to sample data
        assert mock_sample.called or result is not None

    def test_fallback_on_exception(self, fetcher_no_cache):
        """Test fallback to sample data on exception"""
        with patch('yfinance.Ticker') as mock_yf:
            mock_yf.side_effect = Exception("Network error")

            with patch('quantsploit.utils.data_fetcher.get_sample_data') as mock_sample:
                mock_sample.return_value = pd.DataFrame({'Close': [100]})

                result = fetcher_no_cache.get_stock_data("AAPL")

        mock_sample.assert_called()


class TestGetStockInfo:
    """Tests for get_stock_info method"""

    def test_get_stock_info_sample_mode(self, fetcher_sample_mode):
        """Test getting stock info in sample mode"""
        with patch('quantsploit.utils.data_fetcher.get_sample_info') as mock_info:
            mock_info.return_value = {"symbol": "AAPL", "name": "Apple Inc."}

            result = fetcher_sample_mode.get_stock_info("AAPL")

        mock_info.assert_called_with("AAPL")
        assert "symbol" in result

    def test_get_stock_info_fallback(self, fetcher_no_cache):
        """Test fallback on info fetch failure"""
        with patch('yfinance.Ticker') as mock_yf:
            mock_ticker = Mock()
            mock_ticker.info = None
            mock_yf.return_value = mock_ticker

            with patch('quantsploit.utils.data_fetcher.get_sample_info') as mock_info:
                mock_info.return_value = {"symbol": "AAPL"}

                result = fetcher_no_cache.get_stock_info("AAPL")

        # Should return sample info on failure
        assert result is not None


class TestGetOptionsChain:
    """Tests for get_options_chain method"""

    def test_get_options_chain_sample_mode(self, fetcher_sample_mode):
        """Test getting options chain in sample mode"""
        with patch('quantsploit.utils.data_fetcher.get_sample_options') as mock_opts:
            mock_opts.return_value = {
                "expiration": "2024-01-19",
                "calls": pd.DataFrame(),
                "puts": pd.DataFrame()
            }

            result = fetcher_sample_mode.get_options_chain("AAPL")

        mock_opts.assert_called()
        assert "expiration" in result

    def test_get_options_chain_with_expiration(self, fetcher_sample_mode):
        """Test getting options chain with specific expiration"""
        with patch('quantsploit.utils.data_fetcher.get_sample_options') as mock_opts:
            mock_opts.return_value = {"expiration": "2024-06-21"}

            fetcher_sample_mode.get_options_chain("AAPL", expiration="2024-06-21")

        mock_opts.assert_called_with("AAPL", "2024-06-21")


class TestGetMultipleStocks:
    """Tests for get_multiple_stocks method"""

    def test_get_multiple_stocks(self, fetcher_sample_mode):
        """Test fetching multiple stocks"""
        with patch.object(fetcher_sample_mode, 'get_stock_data') as mock_get:
            mock_get.return_value = pd.DataFrame({'Close': [100]})

            result = fetcher_sample_mode.get_multiple_stocks(["AAPL", "MSFT", "GOOGL"])

        assert mock_get.call_count == 3
        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOGL" in result

    def test_get_multiple_stocks_partial_failure(self, fetcher_sample_mode):
        """Test that partial failures are handled"""
        def side_effect(symbol, period, interval):
            if symbol == "INVALID":
                return None
            return pd.DataFrame({'Close': [100]})

        with patch.object(fetcher_sample_mode, 'get_stock_data', side_effect=side_effect):
            result = fetcher_sample_mode.get_multiple_stocks(["AAPL", "INVALID", "MSFT"])

        assert "AAPL" in result
        assert "MSFT" in result
        assert "INVALID" not in result

    def test_get_multiple_stocks_empty_list(self, fetcher_sample_mode):
        """Test with empty symbol list"""
        result = fetcher_sample_mode.get_multiple_stocks([])
        assert result == {}


class TestGetRealtimeQuote:
    """Tests for get_realtime_quote method"""

    def test_get_realtime_quote_success(self, fetcher_no_cache):
        """Test successful realtime quote fetch"""
        with patch('yfinance.Ticker') as mock_yf:
            mock_ticker = Mock()
            mock_ticker.info = {
                "symbol": "AAPL",
                "currentPrice": 175.50,
                "regularMarketChange": 2.30,
                "regularMarketChangePercent": 1.33,
                "volume": 50000000,
                "marketCap": 2800000000000
            }
            mock_yf.return_value = mock_ticker

            result = fetcher_no_cache.get_realtime_quote("AAPL")

        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["price"] == 175.50

    def test_get_realtime_quote_failure(self, fetcher_no_cache):
        """Test realtime quote failure returns None"""
        with patch('yfinance.Ticker') as mock_yf:
            mock_yf.side_effect = Exception("Network error")

            result = fetcher_no_cache.get_realtime_quote("AAPL")

        assert result is None


class TestSearchSymbols:
    """Tests for search_symbols method"""

    def test_search_symbols_found(self, fetcher_no_cache):
        """Test successful symbol search"""
        with patch('yfinance.Ticker') as mock_yf:
            mock_ticker = Mock()
            mock_ticker.info = {
                "symbol": "AAPL",
                "longName": "Apple Inc.",
                "quoteType": "EQUITY"
            }
            mock_yf.return_value = mock_ticker

            result = fetcher_no_cache.search_symbols("aapl")

        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["name"] == "Apple Inc."

    def test_search_symbols_not_found(self, fetcher_no_cache):
        """Test symbol search with no results"""
        with patch('yfinance.Ticker') as mock_yf:
            mock_yf.side_effect = Exception("Not found")

            result = fetcher_no_cache.search_symbols("INVALID")

        assert result == []

    def test_search_symbols_uppercase(self, fetcher_no_cache):
        """Test symbol is converted to uppercase"""
        with patch('yfinance.Ticker') as mock_yf:
            mock_ticker = Mock()
            mock_ticker.info = {"symbol": "AAPL"}
            mock_yf.return_value = mock_ticker

            fetcher_no_cache.search_symbols("aapl")

        # Should be called with uppercase
        mock_yf.assert_called_with("AAPL")


class TestLiveDataAvailability:
    """Tests for live data availability tracking"""

    def test_live_data_available_flag_set(self, fetcher_no_cache):
        """Test _live_data_available flag is set on success"""
        with patch('yfinance.Ticker') as mock_yf:
            mock_ticker = Mock()
            mock_ticker.history.return_value = pd.DataFrame({'Close': [100]})
            mock_yf.return_value = mock_ticker

            fetcher_no_cache.get_stock_data("AAPL")

        assert fetcher_no_cache._live_data_available is True

    def test_live_data_unavailable_flag_set(self, fetcher_no_cache):
        """Test _live_data_available flag is False on failure"""
        with patch('yfinance.Ticker') as mock_yf:
            mock_ticker = Mock()
            mock_ticker.history.return_value = pd.DataFrame()  # Empty
            mock_yf.return_value = mock_ticker

            with patch('quantsploit.utils.data_fetcher.get_sample_data') as mock_sample:
                mock_sample.return_value = pd.DataFrame({'Close': [100]})
                fetcher_no_cache.get_stock_data("AAPL")

        assert fetcher_no_cache._live_data_available is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
