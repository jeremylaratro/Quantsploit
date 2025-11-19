"""
Volume Profile Swing Trading Strategy

This strategy uses volume profile analysis to identify optimal swing trade entries:
- Volume Profile calculation (distribution of volume by price level)
- Point of Control (POC) - price level with highest volume
- Value Area (VA) - price range containing 70% of volume
- High/Low Volume Nodes for support/resistance
- Delta analysis (buying vs selling pressure)

State-of-the-art techniques:
- Volume-weighted price levels
- Delta divergence detection
- Liquidity zone identification
- Volume imbalance analysis
- Multi-timeframe volume confirmation
"""

from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher
from quantsploit.utils.ta_compat import rsi, atr, vwap
from quantsploit.utils.backtesting import (
    Backtester, BacktestConfig, BacktestResults
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class VolumeProfileSwingStrategy(BaseModule):
    """
    Volume Profile Swing Trading Strategy

    Uses volume distribution analysis for swing trading
    """

    name = "volume_profile_swing"
    description = "Swing trading using volume profile and POC analysis"
    category = "strategies"

    options = {
        "SYMBOL": {
            "description": "Stock symbol to analyze",
            "required": True,
            "default": "AAPL"
        },
        "PERIOD": {
            "description": "Historical data period (1y, 2y)",
            "required": False,
            "default": "1y"
        },
        "INTERVAL": {
            "description": "Data interval (1d for daily, 1h for hourly)",
            "required": False,
            "default": "1d"
        },
        "PROFILE_PERIOD": {
            "description": "Number of bars for volume profile calculation",
            "required": False,
            "default": 20
        },
        "NUM_PRICE_LEVELS": {
            "description": "Number of price levels for volume distribution",
            "required": False,
            "default": 50
        },
        "VALUE_AREA_PCT": {
            "description": "Percentage of volume for value area (0.70 = 70%)",
            "required": False,
            "default": 0.70
        },
        "POC_TOLERANCE": {
            "description": "POC proximity tolerance as % (e.g., 0.5 = 0.5%)",
            "required": False,
            "default": 0.5
        },
        "INITIAL_CAPITAL": {
            "description": "Initial capital for backtesting",
            "required": False,
            "default": 100000
        },
        "POSITION_SIZE": {
            "description": "Position size as fraction of capital",
            "required": False,
            "default": 0.5
        },
        "USE_DELTA": {
            "description": "Use delta analysis (requires intraday data)",
            "required": False,
            "default": False
        },
    }

    def calculate_volume_profile(
        self,
        df: pd.DataFrame,
        num_levels: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate volume profile for a price range

        Args:
            df: DataFrame with OHLCV data
            num_levels: Number of price levels

        Returns:
            Tuple of (price_levels, volume_at_levels)
        """
        # Define price range
        price_min = df['Low'].min()
        price_max = df['High'].max()

        # Create price bins
        price_bins = np.linspace(price_min, price_max, num_levels + 1)
        price_levels = (price_bins[:-1] + price_bins[1:]) / 2

        # Initialize volume array
        volume_at_levels = np.zeros(num_levels)

        # Distribute volume across price levels
        for idx, row in df.iterrows():
            bar_low = row['Low']
            bar_high = row['High']
            bar_volume = row['Volume']

            # Find which price levels this bar spans
            for i, price in enumerate(price_levels):
                if bar_low <= price <= bar_high:
                    # Distribute volume proportionally
                    volume_at_levels[i] += bar_volume / num_levels

        return price_levels, volume_at_levels

    def find_poc(
        self,
        price_levels: np.ndarray,
        volume_at_levels: np.ndarray
    ) -> float:
        """
        Find Point of Control (price with highest volume)

        Args:
            price_levels: Array of price levels
            volume_at_levels: Array of volume at each level

        Returns:
            POC price
        """
        poc_idx = np.argmax(volume_at_levels)
        poc_price = price_levels[poc_idx]

        return poc_price

    def find_value_area(
        self,
        price_levels: np.ndarray,
        volume_at_levels: np.ndarray,
        value_area_pct: float
    ) -> Tuple[float, float, float]:
        """
        Find Value Area (price range containing X% of volume)

        Args:
            price_levels: Array of price levels
            volume_at_levels: Array of volume at each level
            value_area_pct: Percentage of volume (e.g., 0.70 for 70%)

        Returns:
            Tuple of (value_area_high, value_area_low, poc)
        """
        total_volume = volume_at_levels.sum()
        target_volume = total_volume * value_area_pct

        # Start from POC and expand outward
        poc_idx = np.argmax(volume_at_levels)

        # Initialize value area with POC
        va_indices = {poc_idx}
        va_volume = volume_at_levels[poc_idx]

        # Expand value area
        lower_idx = poc_idx
        upper_idx = poc_idx

        while va_volume < target_volume:
            # Check volume at boundaries
            lower_volume = volume_at_levels[lower_idx - 1] if lower_idx > 0 else 0
            upper_volume = volume_at_levels[upper_idx + 1] if upper_idx < len(volume_at_levels) - 1 else 0

            # Expand to side with more volume
            if lower_volume >= upper_volume and lower_idx > 0:
                lower_idx -= 1
                va_indices.add(lower_idx)
                va_volume += volume_at_levels[lower_idx]
            elif upper_idx < len(volume_at_levels) - 1:
                upper_idx += 1
                va_indices.add(upper_idx)
                va_volume += volume_at_levels[upper_idx]
            else:
                break

        value_area_high = price_levels[upper_idx]
        value_area_low = price_levels[lower_idx]
        poc = price_levels[poc_idx]

        return value_area_high, value_area_low, poc

    def find_high_volume_nodes(
        self,
        price_levels: np.ndarray,
        volume_at_levels: np.ndarray,
        threshold_percentile: float = 75
    ) -> List[float]:
        """
        Find high volume nodes (HVN) - areas of strong support/resistance

        Args:
            price_levels: Array of price levels
            volume_at_levels: Array of volume at each level
            threshold_percentile: Percentile threshold for HVN

        Returns:
            List of HVN prices
        """
        threshold = np.percentile(volume_at_levels, threshold_percentile)

        hvn_prices = price_levels[volume_at_levels >= threshold].tolist()

        return hvn_prices

    def find_low_volume_nodes(
        self,
        price_levels: np.ndarray,
        volume_at_levels: np.ndarray,
        threshold_percentile: float = 25
    ) -> List[float]:
        """
        Find low volume nodes (LVN) - areas of weak support (price moves fast through these)

        Args:
            price_levels: Array of price levels
            volume_at_levels: Array of volume at each level
            threshold_percentile: Percentile threshold for LVN

        Returns:
            List of LVN prices
        """
        threshold = np.percentile(volume_at_levels, threshold_percentile)

        lvn_prices = price_levels[volume_at_levels <= threshold].tolist()

        return lvn_prices

    def calculate_delta(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate buying vs selling pressure (delta)

        For daily data, we approximate:
        - Buying pressure = Volume * (Close - Low) / (High - Low)
        - Selling pressure = Volume * (High - Close) / (High - Low)
        - Delta = Buying - Selling

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series of delta values
        """
        # Avoid division by zero
        price_range = df['High'] - df['Low']
        price_range = price_range.replace(0, np.nan)

        # Calculate buying and selling pressure
        buying_pressure = df['Volume'] * (df['Close'] - df['Low']) / price_range
        selling_pressure = df['Volume'] * (df['High'] - df['Close']) / price_range

        delta = buying_pressure - selling_pressure

        return delta.fillna(0)

    def generate_volume_signals(
        self,
        df: pd.DataFrame,
        profile_period: int,
        num_levels: int,
        value_area_pct: float,
        poc_tolerance: float,
        use_delta: bool
    ) -> pd.DataFrame:
        """
        Generate trading signals based on volume profile analysis

        Args:
            df: Price data
            profile_period: Period for volume profile
            num_levels: Number of price levels
            value_area_pct: Value area percentage
            poc_tolerance: POC proximity tolerance
            use_delta: Whether to use delta analysis

        Returns:
            DataFrame with signals
        """
        signals = df.copy()
        signals['signal'] = 0
        signals['poc'] = np.nan
        signals['va_high'] = np.nan
        signals['va_low'] = np.nan

        # Calculate delta if requested
        if use_delta:
            signals['delta'] = self.calculate_delta(df)
            signals['delta_sma'] = signals['delta'].rolling(10).mean()

        # Calculate volume profile for rolling windows
        for i in range(profile_period, len(df)):
            # Get window
            window_start = i - profile_period
            window = df.iloc[window_start:i]

            # Calculate volume profile
            price_levels, volume_at_levels = self.calculate_volume_profile(
                window,
                num_levels
            )

            # Find POC and value area
            va_high, va_low, poc = self.find_value_area(
                price_levels,
                volume_at_levels,
                value_area_pct
            )

            signals.iloc[i, signals.columns.get_loc('poc')] = poc
            signals.iloc[i, signals.columns.get_loc('va_high')] = va_high
            signals.iloc[i, signals.columns.get_loc('va_low')] = va_low

            current_price = df.iloc[i]['Close']

            # Generate signals based on volume profile
            poc_distance = abs(current_price - poc) / poc * 100

            # Signal 1: Price near POC (potential reversal)
            if poc_distance < poc_tolerance:
                # Check if price is bouncing off POC
                if i > 0:
                    prev_price = df.iloc[i - 1]['Close']
                    if prev_price < poc and current_price >= poc:
                        signals.iloc[i, signals.columns.get_loc('signal')] = 1  # Buy
                    elif prev_price > poc and current_price <= poc:
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1  # Sell

            # Signal 2: Price at value area boundaries
            elif current_price <= va_low:
                # At or below value area low - potential buy
                signals.iloc[i, signals.columns.get_loc('signal')] = 1

            elif current_price >= va_high:
                # At or above value area high - potential sell
                signals.iloc[i, signals.columns.get_loc('signal')] = -1

            # Signal 3: Delta divergence (if using delta)
            if use_delta and i >= profile_period + 10:
                delta = signals.iloc[i]['delta']
                delta_sma = signals.iloc[i]['delta_sma']

                # Positive delta divergence (buying pressure increasing)
                if delta > delta_sma and current_price < poc:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1

                # Negative delta divergence (selling pressure increasing)
                elif delta < delta_sma and current_price > poc:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1

        return signals

    def backtest_strategy(
        self,
        signals: pd.DataFrame,
        initial_capital: float,
        position_size: float
    ) -> BacktestResults:
        """Backtest the volume profile strategy"""
        config = BacktestConfig(
            initial_capital=initial_capital,
            commission_pct=0.001,
            slippage_pct=0.001,
            position_size=position_size,
            max_positions=1
        )

        backtester = Backtester(config)
        in_position = False

        def strategy_logic(bt: Backtester, date: datetime, row: pd.Series):
            nonlocal in_position
            symbol = "SYMBOL"

            signal = row['signal']

            # Entry
            if not in_position and signal == 1:
                success = bt.enter_long(symbol, date, row['Close'])
                if success:
                    in_position = True

            # Exit
            elif in_position and signal == -1:
                bt.exit_position(symbol, date, row['Close'])
                in_position = False

        results = backtester.run_backtest(signals, strategy_logic)
        return results

    def run(self):
        """Execute the volume profile swing strategy"""
        symbol = self.options["SYMBOL"]["value"]
        period = self.options["PERIOD"]["value"]
        interval = self.options["INTERVAL"]["value"]
        profile_period = int(self.options["PROFILE_PERIOD"]["value"])
        num_levels = int(self.options["NUM_PRICE_LEVELS"]["value"])
        value_area_pct = float(self.options["VALUE_AREA_PCT"]["value"])
        poc_tolerance = float(self.options["POC_TOLERANCE"]["value"])
        initial_capital = float(self.options["INITIAL_CAPITAL"]["value"])
        position_size = float(self.options["POSITION_SIZE"]["value"])
        use_delta = self.options["USE_DELTA"]["value"]

        self.print_status(f"Running Volume Profile Swing Strategy for {symbol}")

        # Fetch data
        data_fetcher = DataFetcher(self.database)
        df = data_fetcher.get_stock_data(symbol, period=period, interval=interval)

        if df is None or len(df) < profile_period * 2:
            self.print_error("Insufficient data for analysis")
            return {"error": "Insufficient data"}

        self.print_info(f"Loaded {len(df)} bars of data")

        # Calculate current volume profile
        self.print_status("Calculating volume profile...")

        recent_df = df.iloc[-profile_period:]
        price_levels, volume_at_levels = self.calculate_volume_profile(
            recent_df,
            num_levels
        )

        # Find POC and value area
        va_high, va_low, poc = self.find_value_area(
            price_levels,
            volume_at_levels,
            value_area_pct
        )

        # Find HVN and LVN
        hvn_prices = self.find_high_volume_nodes(price_levels, volume_at_levels)
        lvn_prices = self.find_low_volume_nodes(price_levels, volume_at_levels)

        # Display volume profile analysis
        self.print_good("\n=== Volume Profile Analysis ===")
        self.print_info(f"Profile Period: Last {profile_period} bars")
        self.print_info(f"Point of Control (POC): ${poc:.2f}")
        self.print_info(f"Value Area High: ${va_high:.2f}")
        self.print_info(f"Value Area Low: ${va_low:.2f}")
        self.print_info(f"Value Area Range: ${va_high - va_low:.2f} ({(va_high - va_low) / poc * 100:.2f}%)")

        self.print_info(f"\nHigh Volume Nodes (support/resistance):")
        for hvn in sorted(hvn_prices)[-5:]:  # Show top 5
            self.print_info(f"  ${hvn:.2f}")

        # Generate signals
        self.print_status("\nGenerating volume-based signals...")
        signals = self.generate_volume_signals(
            df,
            profile_period,
            num_levels,
            value_area_pct,
            poc_tolerance,
            use_delta
        )

        # Backtest
        self.print_status("Running backtest...")
        results = self.backtest_strategy(signals, initial_capital, position_size)

        # Display results
        self.print_good("\n=== Backtest Results ===")
        results_dict = results.to_dict()

        for key, value in results_dict.items():
            self.print_info(f"{key}: {value}")

        # Current analysis
        self.print_status("\n=== Current Analysis ===")

        current_price = df['Close'].iloc[-1]
        self.print_info(f"Current Price: ${current_price:.2f}")

        # Position relative to volume profile
        if current_price > va_high:
            position = "Above Value Area"
            interpretation = "Price is expensive, potential short or wait for pullback"
            self.print_warning(f"Position: {position}")
            self.print_info(f"Interpretation: {interpretation}")

        elif current_price < va_low:
            position = "Below Value Area"
            interpretation = "Price is cheap, potential long opportunity"
            self.print_good(f"Position: {position}")
            self.print_info(f"Interpretation: {interpretation}")

        elif abs(current_price - poc) / poc * 100 < poc_tolerance:
            position = "At Point of Control"
            interpretation = "Key decision point - watch for direction"
            self.print_info(f"Position: {position}")
            self.print_info(f"Interpretation: {interpretation}")

        else:
            position = "Within Value Area"
            interpretation = "Fair value zone, wait for better setup"
            self.print_info(f"Position: {position}")
            self.print_info(f"Interpretation: {interpretation}")

        # Trading recommendation
        current_signal = signals['signal'].iloc[-1]

        self.print_info("\n=== Trading Recommendation ===")
        if current_signal == 1:
            self.print_good("Signal: BUY")
            self.print_info(f"Entry: ${current_price:.2f}")
            self.print_info(f"Target: ${poc:.2f} (POC)")
            self.print_info(f"Stop Loss: ${va_low * 0.98:.2f}")

        elif current_signal == -1:
            self.print_warning("Signal: SELL/SHORT")
            self.print_info(f"Entry: ${current_price:.2f}")
            self.print_info(f"Target: ${poc:.2f} (POC)")
            self.print_info(f"Stop Loss: ${va_high * 1.02:.2f}")

        else:
            self.print_info("Signal: NEUTRAL - Wait for better setup")

        return {
            "symbol": symbol,
            "current_price": float(current_price),
            "poc": float(poc),
            "value_area_high": float(va_high),
            "value_area_low": float(va_low),
            "position": position,
            "signal": "BUY" if current_signal == 1 else "SELL" if current_signal == -1 else "NEUTRAL",
            "backtest_results": results_dict,
            "total_return_pct": results.total_return_pct,
            "sharpe_ratio": results.sharpe_ratio,
            "win_rate": results.win_rate,
            "total_trades": results.total_trades
        }
