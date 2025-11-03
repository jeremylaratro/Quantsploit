"""
Price Momentum Scanner Module
"""

import pandas as pd
from typing import Dict, Any, List
from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher


class PriceMomentumScanner(BaseModule):
    """
    Scan multiple stocks for price momentum
    """

    @property
    def name(self) -> str:
        return "Price Momentum Scanner"

    @property
    def description(self) -> str:
        return "Scan multiple stocks for price momentum and volume patterns"

    @property
    def author(self) -> str:
        return "Quantsploit Team"

    @property
    def category(self) -> str:
        return "scanner"

    def _init_options(self):
        super()._init_options()
        self.options.update({
            "SYMBOLS": {
                "value": None,
                "required": True,
                "description": "Comma-separated list of symbols to scan"
            },
            "MIN_VOLUME": {
                "value": 1000000,
                "required": False,
                "description": "Minimum average volume"
            },
            "MIN_GAIN_PCT": {
                "value": 5.0,
                "required": False,
                "description": "Minimum percentage gain to flag"
            },
        })
        # Remove SYMBOL requirement for scanners
        self.options["SYMBOL"]["required"] = False

    def run(self) -> Dict[str, Any]:
        """Execute momentum scan"""
        symbols_str = self.get_option("SYMBOLS")
        period = self.get_option("PERIOD")
        interval = self.get_option("INTERVAL")
        min_volume = float(self.get_option("MIN_VOLUME"))
        min_gain_pct = float(self.get_option("MIN_GAIN_PCT"))

        if not symbols_str:
            return {"success": False, "error": "SYMBOLS option is required"}

        symbols = [s.strip().upper() for s in symbols_str.split(",")]

        # Fetch data for all symbols
        fetcher = DataFetcher(self.framework.database)
        scan_results = []

        self.log(f"Scanning {len(symbols)} symbols...")

        for symbol in symbols:
            self.log(f"Fetching data for {symbol}...")
            df = fetcher.get_stock_data(symbol, period, interval)

            if df is None or df.empty:
                self.log(f"No data for {symbol}", "warning")
                continue

            # Calculate metrics
            latest_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            price_change_pct = ((latest_price / prev_price) - 1) * 100

            avg_volume = df['Volume'].mean()
            latest_volume = df['Volume'].iloc[-1]
            volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 0

            # Calculate gain/loss over period
            period_start_price = df['Close'].iloc[0]
            period_gain_pct = ((latest_price / period_start_price) - 1) * 100

            # Flag criteria
            flags = []
            if avg_volume >= min_volume:
                flags.append("High Volume")
            if abs(price_change_pct) >= min_gain_pct:
                flags.append("Large Move")
            if volume_ratio > 2:
                flags.append("Volume Spike")
            if period_gain_pct >= min_gain_pct:
                flags.append("Strong Momentum")

            scan_results.append({
                "Symbol": symbol,
                "Price": f"${latest_price:.2f}",
                "Change %": f"{price_change_pct:+.2f}%",
                "Period Gain %": f"{period_gain_pct:+.2f}%",
                "Volume": f"{latest_volume:,.0f}",
                "Vol Ratio": f"{volume_ratio:.2f}x",
                "Flags": ", ".join(flags) if flags else "None"
            })

        # Create DataFrame for results
        results_df = pd.DataFrame(scan_results)

        # Sort by period gain
        if not results_df.empty:
            results_df = results_df.sort_values(
                by="Period Gain %",
                ascending=False,
                key=lambda x: x.str.rstrip('%').astype(float)
            )

        return {
            "scan_count": len(symbols),
            "results_found": len(scan_results),
            "scan_results": results_df
        }
