"""
Options Analysis Module
"""

import pandas as pd
from typing import Dict, Any
from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher
from datetime import datetime


class OptionsAnalyzer(BaseModule):
    """
    Analyze options chain for a stock
    """

    @property
    def name(self) -> str:
        return "Options Analyzer"

    @property
    def description(self) -> str:
        return "Analyze options chain, calculate Greeks, and identify opportunities"

    @property
    def author(self) -> str:
        return "Quantsploit Team"

    @property
    def category(self) -> str:
        return "options"

    def _init_options(self):
        super()._init_options()
        self.options.update({
            "EXPIRATION": {
                "value": None,
                "required": False,
                "description": "Option expiration date (YYYY-MM-DD), leave empty for nearest"
            },
            "OPTION_TYPE": {
                "value": "both",
                "required": False,
                "description": "Option type to analyze (calls/puts/both)"
            },
            "MIN_VOLUME": {
                "value": 10,
                "required": False,
                "description": "Minimum volume filter"
            },
        })

    def run(self) -> Dict[str, Any]:
        """Execute options analysis"""
        symbol = self.get_option("SYMBOL")
        expiration = self.get_option("EXPIRATION")
        option_type = self.get_option("OPTION_TYPE").lower()
        min_volume = int(self.get_option("MIN_VOLUME"))

        # Fetch options data
        fetcher = DataFetcher(self.framework.database)
        options_data = fetcher.get_options_chain(symbol, expiration)

        if not options_data:
            return {"success": False, "error": f"Failed to fetch options for {symbol}"}

        # Get current stock price
        stock_info = fetcher.get_stock_info(symbol)
        current_price = stock_info.get('currentPrice', stock_info.get('regularMarketPrice', 0))

        results = {
            "symbol": symbol,
            "current_price": current_price,
            "expiration": options_data['expiration'],
            "available_expirations": options_data['available_expirations']
        }

        # Analyze calls
        if option_type in ["calls", "both"]:
            calls = options_data['calls']
            calls_filtered = calls[calls['volume'] >= min_volume].copy()

            if not calls_filtered.empty:
                # Find ATM, ITM, OTM
                calls_filtered['moneyness'] = calls_filtered['strike'].apply(
                    lambda x: self._classify_moneyness(x, current_price, 'call')
                )

                # Calculate some metrics
                calls_filtered['bid_ask_spread'] = calls_filtered['ask'] - calls_filtered['bid']
                calls_filtered['spread_pct'] = (calls_filtered['bid_ask_spread'] / calls_filtered['ask']) * 100

                results['calls_summary'] = {
                    'total_options': len(calls),
                    'liquid_options': len(calls_filtered),
                    'total_volume': int(calls_filtered['volume'].sum()),
                    'total_open_interest': int(calls_filtered['openInterest'].sum()),
                    'avg_iv': calls_filtered['impliedVolatility'].mean() * 100
                }

                results['top_calls_by_volume'] = calls_filtered.nlargest(10, 'volume')[
                    ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility', 'moneyness']
                ]

        # Analyze puts
        if option_type in ["puts", "both"]:
            puts = options_data['puts']
            puts_filtered = puts[puts['volume'] >= min_volume].copy()

            if not puts_filtered.empty:
                # Find ATM, ITM, OTM
                puts_filtered['moneyness'] = puts_filtered['strike'].apply(
                    lambda x: self._classify_moneyness(x, current_price, 'put')
                )

                # Calculate some metrics
                puts_filtered['bid_ask_spread'] = puts_filtered['ask'] - puts_filtered['bid']
                puts_filtered['spread_pct'] = (puts_filtered['bid_ask_spread'] / puts_filtered['ask']) * 100

                results['puts_summary'] = {
                    'total_options': len(puts),
                    'liquid_options': len(puts_filtered),
                    'total_volume': int(puts_filtered['volume'].sum()),
                    'total_open_interest': int(puts_filtered['openInterest'].sum()),
                    'avg_iv': puts_filtered['impliedVolatility'].mean() * 100
                }

                results['top_puts_by_volume'] = puts_filtered.nlargest(10, 'volume')[
                    ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility', 'moneyness']
                ]

        # Calculate Put/Call ratio
        if option_type == "both":
            total_call_volume = options_data['calls']['volume'].sum()
            total_put_volume = options_data['puts']['volume'].sum()
            if total_call_volume > 0:
                results['put_call_ratio'] = total_put_volume / total_call_volume
                results['put_call_interpretation'] = self._interpret_pcr(results['put_call_ratio'])

        return results

    def _classify_moneyness(self, strike: float, spot: float, option_type: str) -> str:
        """Classify option as ITM, ATM, or OTM"""
        if option_type == 'call':
            if strike < spot * 0.98:
                return "ITM"
            elif strike <= spot * 1.02:
                return "ATM"
            else:
                return "OTM"
        else:  # put
            if strike > spot * 1.02:
                return "ITM"
            elif strike >= spot * 0.98:
                return "ATM"
            else:
                return "OTM"

    def _interpret_pcr(self, pcr: float) -> str:
        """Interpret put/call ratio"""
        if pcr > 1.0:
            return "Bearish (more puts)"
        elif pcr < 0.7:
            return "Bullish (more calls)"
        else:
            return "Neutral"
