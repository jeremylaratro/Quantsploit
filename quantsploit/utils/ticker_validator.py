"""
Ticker Validation Utility
Validates stock tickers against a comprehensive dictionary of valid symbols
"""

import os
import json
import re
from pathlib import Path
from typing import Set, List, Optional
from datetime import datetime, timedelta

class TickerValidator:
    """
    Validates stock tickers against a comprehensive database of valid symbols.
    Supports NYSE, NASDAQ, AMEX, and other major exchanges.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the ticker validator

        Args:
            cache_dir: Directory to store ticker cache (default: quantsploit/data)
        """
        if cache_dir is None:
            # Default to quantsploit/data directory
            cache_dir = Path(__file__).parent.parent / "data"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ticker_cache_file = self.cache_dir / "valid_tickers.json"
        self.cache_metadata_file = self.cache_dir / "ticker_cache_metadata.json"

        self._valid_tickers: Optional[Set[str]] = None
        self._load_or_initialize_cache()

    def _load_or_initialize_cache(self):
        """Load ticker cache from disk or initialize with comprehensive list"""
        if self.ticker_cache_file.exists():
            try:
                with open(self.ticker_cache_file, 'r') as f:
                    data = json.load(f)
                    self._valid_tickers = set(data.get('tickers', []))
                print(f"[TickerValidator] Loaded {len(self._valid_tickers)} valid tickers from cache")
                return
            except Exception as e:
                print(f"[TickerValidator] Error loading cache: {e}, regenerating...")

        # Initialize with comprehensive ticker list
        self._valid_tickers = self._get_comprehensive_ticker_list()
        self._save_cache()

    def _get_comprehensive_ticker_list(self) -> Set[str]:
        """
        Get comprehensive list of valid US stock tickers.
        This combines multiple sources for maximum coverage.
        """
        tickers = set()

        # Try to fetch from yfinance if available
        try:
            import yfinance as yf
            import pandas as pd

            # Get S&P 500 tickers
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            try:
                sp500_df = pd.read_html(sp500_url)[0]
                sp500_tickers = sp500_df['Symbol'].str.replace('.', '-').tolist()
                tickers.update(sp500_tickers)
                print(f"[TickerValidator] Fetched {len(sp500_tickers)} S&P 500 tickers")
            except Exception as e:
                print(f"[TickerValidator] Could not fetch S&P 500 list: {e}")

            # Get NASDAQ 100 tickers
            nasdaq_url = "https://en.wikipedia.org/wiki/NASDAQ-100"
            try:
                nasdaq_df = pd.read_html(nasdaq_url)[4]
                nasdaq_tickers = nasdaq_df['Ticker'].str.replace('.', '-').tolist()
                tickers.update(nasdaq_tickers)
                print(f"[TickerValidator] Fetched {len(nasdaq_tickers)} NASDAQ-100 tickers")
            except Exception as e:
                print(f"[TickerValidator] Could not fetch NASDAQ-100 list: {e}")

        except ImportError:
            print("[TickerValidator] yfinance not available, using static list")

        # Add comprehensive static list of popular tickers
        static_tickers = self._get_static_ticker_list()
        tickers.update(static_tickers)

        print(f"[TickerValidator] Initialized with {len(tickers)} valid tickers")
        return tickers

    def _get_static_ticker_list(self) -> Set[str]:
        """
        Comprehensive static list of major US stock tickers.
        This ensures validation works even without internet access.
        """
        return {
            # Major Tech
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC',
            'NFLX', 'ADBE', 'CRM', 'ORCL', 'CSCO', 'AVGO', 'QCOM', 'TXN', 'INTU', 'IBM',
            'AMAT', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'FTNT', 'PANW', 'CRWD',
            'ZS', 'DDOG', 'NET', 'SNOW', 'PLTR', 'U', 'RBLX', 'COIN', 'SQ', 'PYPL',

            # Financial
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'V',
            'MA', 'BRK.B', 'BRK.A', 'BX', 'KKR', 'APO', 'PNC', 'USB', 'TFC', 'COF',

            # Healthcare & Pharma
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'LLY', 'DHR', 'BMY',
            'AMGN', 'GILD', 'CVS', 'CI', 'REGN', 'VRTX', 'ISRG', 'MRNA', 'BIIB', 'ZTS',

            # Consumer & Retail
            'WMT', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW', 'DIS', 'CMCSA', 'COST',
            'PEP', 'KO', 'PM', 'MO', 'CL', 'PG', 'EL', 'ULTA', 'LULU', 'ROST',

            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO', 'PSX', 'OXY',

            # Industrial & Manufacturing
            'BA', 'CAT', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'MMM', 'DE', 'EMR',

            # Automotive
            'F', 'GM', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI',

            # Airlines & Transportation
            'AAL', 'DAL', 'UAL', 'LUV', 'JBLU', 'UBER', 'LYFT', 'DASH',

            # Telecom & Media
            'T', 'VZ', 'TMUS', 'NFLX', 'PARA', 'WBD', 'SPOT', 'TWLO',

            # Real Estate & REITs
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'VICI', 'DLR', 'AVB',

            # Semiconductors (additional)
            'TSM', 'ASML', 'QRVO', 'SWKS', 'MCHP', 'ADI', 'NXPI', 'ON',

            # China Tech (US-listed)
            'BABA', 'JD', 'PDD', 'BIDU', 'TCOM', 'IQ', 'BILI', 'TME',

            # Meme Stocks & Reddit Favorites
            'GME', 'AMC', 'BB', 'BBBY', 'NOK', 'CLOV', 'WISH', 'WKHS', 'SPCE', 'HOOD',
            'SOFI', 'AFRM', 'UPST', 'OPEN', 'SKLZ', 'DKNG', 'PENN',

            # SPACs & Recent IPOs
            'RIVN', 'LCID', 'GGPI', 'DWAC', 'IRNT', 'OPAD',

            # Crypto-related
            'MSTR', 'RIOT', 'MARA', 'CLSK', 'HUT', 'BITF', 'SI', 'SOS',

            # Major ETFs (commonly discussed)
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'LQD',
            'HYG', 'TLT', 'GLD', 'SLV', 'USO', 'UNG', 'XLE', 'XLF', 'XLK', 'XLV',
            'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE', 'VNQ', 'EEM', 'EWJ', 'EWZ',
            'FXI', 'KWEB', 'ARKK', 'ARKG', 'ARKF', 'ARKW', 'ICLN', 'TAN', 'QCLN',
            'SQQQ', 'TQQQ', 'SPXU', 'UPRO', 'SOXL', 'SOXS', 'UVXY', 'SVXY',

            # Biotech & Small Cap
            'SAVA', 'OCGN', 'CLOV', 'ATER', 'PROG', 'CEI', 'BBIG', 'DWAC',

            # Additional Popular Stocks
            'SHOP', 'SQ', 'ZM', 'DOCU', 'OKTA', 'TEAM', 'WDAY', 'NOW', 'SPLK', 'MDB',
            'ESTC', 'DBX', 'BOX', 'ZI', 'VEEV', 'RNG', 'HUBS', 'COUP', 'BILL', 'PCTY',

            # Defense & Aerospace (additional)
            'NOC', 'GD', 'LHX', 'TXT', 'HII',

            # Materials & Chemicals
            'LIN', 'APD', 'ECL', 'DD', 'DOW', 'PPG', 'SHW', 'NEM', 'FCX',

            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'XEL', 'ED',

            # Miscellaneous Popular
            'SNAP', 'PINS', 'TWTR', 'ROKU', 'BMBL', 'MTCH', 'YELP', 'GRUB', 'UBER',
            'Z', 'RDFN', 'BYND', 'TTWO', 'EA', 'ATVI', 'RBLX', 'UNITY', 'EXPE', 'BKNG',

            # Banking & Regional Banks
            'BAC', 'JPM', 'WFC', 'C', 'USB', 'PNC', 'TFC', 'KEY', 'FITB', 'RF',
            'CFG', 'HBAN', 'MTB', 'ZION', 'CMA', 'SIVB', 'SBNY', 'FRC',

            # Insurance
            'BRK.A', 'BRK.B', 'PGR', 'ALL', 'TRV', 'AIG', 'MET', 'PRU', 'AFL', 'HIG',

            # Pharmaceutical (additional)
            'NVO', 'RHHBY', 'AZN', 'SNY', 'GSK', 'TAK', 'NVS', 'TEVA',
        }

    def _save_cache(self):
        """Save ticker cache to disk"""
        try:
            # Save tickers
            with open(self.ticker_cache_file, 'w') as f:
                json.dump({
                    'tickers': sorted(list(self._valid_tickers)),
                    'count': len(self._valid_tickers)
                }, f, indent=2)

            # Save metadata
            with open(self.cache_metadata_file, 'w') as f:
                json.dump({
                    'last_updated': datetime.now().isoformat(),
                    'ticker_count': len(self._valid_tickers)
                }, f, indent=2)

            print(f"[TickerValidator] Saved {len(self._valid_tickers)} tickers to cache")
        except Exception as e:
            print(f"[TickerValidator] Error saving cache: {e}")

    def is_valid(self, ticker: str) -> bool:
        """
        Check if a ticker is valid

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            True if ticker is valid, False otherwise
        """
        if not ticker or not isinstance(ticker, str):
            return False

        # Normalize ticker (uppercase, handle special chars)
        normalized = ticker.upper().strip()

        # Handle special cases like BRK.A/BRK.B
        if '.' in normalized:
            # Check both with and without period
            if normalized in self._valid_tickers:
                return True
            # Also check hyphenated version (some sources use BRK-A)
            hyphenated = normalized.replace('.', '-')
            if hyphenated in self._valid_tickers:
                return True

        return normalized in self._valid_tickers

    def validate_batch(self, tickers: List[str]) -> List[str]:
        """
        Validate a batch of tickers and return only valid ones

        Args:
            tickers: List of ticker symbols

        Returns:
            List of valid tickers
        """
        return [t for t in tickers if self.is_valid(t)]

    def get_invalid_tickers(self, tickers: List[str]) -> List[str]:
        """
        Get list of invalid tickers from a batch

        Args:
            tickers: List of ticker symbols

        Returns:
            List of invalid tickers
        """
        return [t for t in tickers if not self.is_valid(t)]

    def add_ticker(self, ticker: str):
        """
        Manually add a ticker to the valid list

        Args:
            ticker: Stock ticker symbol to add
        """
        normalized = ticker.upper().strip()
        self._valid_tickers.add(normalized)
        self._save_cache()

    def add_tickers(self, tickers: List[str]):
        """
        Manually add multiple tickers to the valid list

        Args:
            tickers: List of ticker symbols to add
        """
        for ticker in tickers:
            normalized = ticker.upper().strip()
            self._valid_tickers.add(normalized)
        self._save_cache()

    def refresh_cache(self):
        """Force refresh of the ticker cache from sources"""
        self._valid_tickers = self._get_comprehensive_ticker_list()
        self._save_cache()

    def get_ticker_count(self) -> int:
        """Get count of valid tickers in database"""
        return len(self._valid_tickers)

    def get_all_tickers(self) -> List[str]:
        """Get all valid tickers as a sorted list"""
        return sorted(list(self._valid_tickers))


# Global instance for easy access
_global_validator: Optional[TickerValidator] = None

def get_validator() -> TickerValidator:
    """Get or create global ticker validator instance"""
    global _global_validator
    if _global_validator is None:
        _global_validator = TickerValidator()
    return _global_validator
