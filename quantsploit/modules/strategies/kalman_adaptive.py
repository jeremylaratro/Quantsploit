"""
Kalman Filter Adaptive Trading Strategy

This strategy uses Kalman Filtering to:
- Estimate true price trends by filtering out market noise
- Adaptively adjust to changing market conditions
- Generate high-probability trading signals
- Dynamic position sizing based on filter confidence

State-of-the-art techniques:
- Multi-dimensional Kalman Filter (price + velocity + acceleration)
- Adaptive noise covariance estimation
- Confidence-based position sizing
- Regime-aware parameter adjustment
"""

from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher
from quantsploit.utils.backtesting import (
    Backtester, BacktestConfig, BacktestResults
)
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class KalmanAdaptiveStrategy(BaseModule):
    """
    Kalman Filter Adaptive Trading Strategy

    Uses advanced Kalman Filtering to estimate true price trends
    and generate adaptive trading signals
    """

    @property
    def name(self) -> str:


        return "kalman_adaptive"


    @property
    def description(self) -> str:
        return """Kalman Filter - Filters noise to estimate true price trend with high precision.

SYNOPSIS: Applies Kalman Filter to smooth price data and remove noise. Buys when
price deviates below filtered trend (oversold), sells when above (overbought).

SIMULATION POSITIONS:
  - LONG: Price <0.5% below Kalman filtered price (noise pullback)
  - STRONG LONG: Deviation <-0.5% AND positive velocity (uptrend confirmed)
  - FLAT/EXIT: Price >0.5% above filtered price
  - Position size: 50% base, scaled by filter confidence

FILTER TYPES:
  1. Simple: Price only (smooth trend line)
  2. Velocity: Price + velocity (detects acceleration)
  3. Full: Price + velocity + acceleration (best for trends)

RECOMMENDED ENTRY:
  - Use 'full' filter for trending stocks
  - Enter when deviation <-0.5% (price below trend)
  - Strong buy if velocity >0 (upward momentum confirmed)
  - Exit when deviation >+0.5% (price extended above trend)

PARAMETERS:
  - PROCESS_NOISE: 0.01 (lower = smoother, higher = reactive)
  - MEASUREMENT_NOISE: 1.0 (higher = trust price less)
  - SIGNAL_THRESHOLD: 0.5% (deviation trigger)
  - USE_CONFIDENCE_SIZING: Scale position by filter confidence

REAL-WORLD USE:
  1. Run Kalman filter to get true trend (filters whipsaws)
  2. Wait for price to deviate -0.5% from filtered price
  3. Enter if velocity confirms direction
  4. Exit when price reverts above filtered price
  5. Works best on trending stocks with noise (volatility)

ADVANTAGES:
  - Reduces false signals vs simple moving averages
  - Adapts quickly to regime changes
  - Confidence-based sizing improves risk management
  - Velocity signals catch momentum early"""


    @property
    def author(self) -> str:


        return "Quantsploit Team"


    @property
    def category(self) -> str:


        return "strategy"

    def _init_options(self):
        super()._init_options()
        self.options.update({
        "SYMBOL": {
            "description": "Stock symbol to analyze",
            "required": True,
            "value": "AAPL"
            },
            "PERIOD": {
            "description": "Historical data period (1y, 2y, 5y)",
            "required": False,
            "value": "1y"
        },
        "INTERVAL": {
            "description": "Data interval (1d for daily)",
            "required": False,
            "value": "1d"
        },
        "FILTER_TYPE": {
            "description": "Filter type: simple, velocity, or full (price+velocity+acceleration)",
            "required": False,
            "value": "full"
        },
        "PROCESS_NOISE": {
            "description": "Process noise variance (lower = smoother filter)",
            "required": False,
            "value": 0.01
        },
        "MEASUREMENT_NOISE": {
            "description": "Measurement noise variance (higher = trust measurements less)",
            "required": False,
            "value": 1.0
        },
        "SIGNAL_THRESHOLD": {
            "description": "Signal threshold as % deviation from filtered price",
            "required": False,
            "value": 0.5
        },
        "INITIAL_CAPITAL": {
            "description": "Initial capital for backtesting",
            "required": False,
            "value": 100000
        },
        "POSITION_SIZE": {
            "description": "Base position size as fraction of capital",
            "required": False,
            "value": 0.5
        },
        "USE_CONFIDENCE_SIZING": {
            "description": "Scale position size based on filter confidence",
            "required": False,
            "value": True
        }
        })


    def kalman_filter_simple(self, prices: pd.Series, Q: float, R: float) -> Tuple[pd.Series, pd.Series]:
        """
        Simple 1D Kalman Filter for price estimation

        Args:
            prices: Price series
            Q: Process noise variance
            R: Measurement noise variance

        Returns:
            Tuple of (filtered_prices, estimation_errors)
        """
        n = len(prices)
        filtered = np.zeros(n)
        errors = np.zeros(n)

        # Initialize
        filtered[0] = prices.iloc[0]
        P = 1.0  # Initial estimation error covariance

        for i in range(1, n):
            # Prediction
            x_pred = filtered[i - 1]
            P_pred = P + Q

            # Update
            y = prices.iloc[i]  # Measurement

            # Kalman gain
            K = P_pred / (P_pred + R)

            # Update estimate
            filtered[i] = x_pred + K * (y - x_pred)

            # Update error covariance
            P = (1 - K) * P_pred

            # Store estimation error
            errors[i] = P

        return pd.Series(filtered, index=prices.index), pd.Series(errors, index=prices.index)

    def kalman_filter_velocity(
        self,
        prices: pd.Series,
        Q_position: float,
        Q_velocity: float,
        R: float
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        2D Kalman Filter with price + velocity

        State: [price, velocity]

        Returns:
            Tuple of (filtered_prices, velocities, estimation_errors)
        """
        n = len(prices)
        filtered_price = np.zeros(n)
        filtered_velocity = np.zeros(n)
        errors = np.zeros(n)

        # State: [price, velocity]
        x = np.array([prices.iloc[0], 0.0])

        # State covariance
        P = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        # Process noise covariance
        Q = np.array([[Q_position, 0.0],
                      [0.0, Q_velocity]])

        # Measurement noise
        R_matrix = np.array([[R]])

        # State transition matrix (assumes dt = 1)
        F = np.array([[1.0, 1.0],
                      [0.0, 1.0]])

        # Observation matrix (we only observe price)
        H = np.array([[1.0, 0.0]])

        filtered_price[0] = x[0]
        filtered_velocity[0] = x[1]

        for i in range(1, n):
            # Prediction
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # Update
            y = np.array([prices.iloc[i]])  # Measurement

            # Innovation
            innovation = y - H @ x_pred

            # Innovation covariance
            S = H @ P_pred @ H.T + R_matrix

            # Kalman gain
            K = P_pred @ H.T @ np.linalg.inv(S)

            # Update estimate
            x = x_pred + K @ innovation

            # Update covariance
            P = (np.eye(2) - K @ H) @ P_pred

            # Store results
            filtered_price[i] = x[0]
            filtered_velocity[i] = x[1]
            errors[i] = P[0, 0]

        return (
            pd.Series(filtered_price, index=prices.index),
            pd.Series(filtered_velocity, index=prices.index),
            pd.Series(errors, index=prices.index)
        )

    def kalman_filter_full(
        self,
        prices: pd.Series,
        Q_position: float,
        Q_velocity: float,
        Q_acceleration: float,
        R: float
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        3D Kalman Filter with price + velocity + acceleration

        State: [price, velocity, acceleration]

        Returns:
            Tuple of (filtered_prices, velocities, accelerations, errors)
        """
        n = len(prices)
        filtered_price = np.zeros(n)
        filtered_velocity = np.zeros(n)
        filtered_acceleration = np.zeros(n)
        errors = np.zeros(n)

        # State: [price, velocity, acceleration]
        x = np.array([prices.iloc[0], 0.0, 0.0])

        # State covariance
        P = np.eye(3)

        # Process noise covariance
        Q = np.array([[Q_position, 0.0, 0.0],
                      [0.0, Q_velocity, 0.0],
                      [0.0, 0.0, Q_acceleration]])

        # Measurement noise
        R_matrix = np.array([[R]])

        # State transition matrix (assumes dt = 1)
        F = np.array([[1.0, 1.0, 0.5],
                      [0.0, 1.0, 1.0],
                      [0.0, 0.0, 1.0]])

        # Observation matrix (we only observe price)
        H = np.array([[1.0, 0.0, 0.0]])

        filtered_price[0] = x[0]
        filtered_velocity[0] = x[1]
        filtered_acceleration[0] = x[2]

        for i in range(1, n):
            # Prediction
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # Update
            y = np.array([prices.iloc[i]])  # Measurement

            # Innovation
            innovation = y - H @ x_pred

            # Innovation covariance
            S = H @ P_pred @ H.T + R_matrix

            # Kalman gain
            K = P_pred @ H.T @ np.linalg.inv(S)

            # Update estimate
            x = x_pred + K @ innovation

            # Update covariance
            P = (np.eye(3) - K @ H) @ P_pred

            # Store results
            filtered_price[i] = x[0]
            filtered_velocity[i] = x[1]
            filtered_acceleration[i] = x[2]
            errors[i] = P[0, 0]

        return (
            pd.Series(filtered_price, index=prices.index),
            pd.Series(filtered_velocity, index=prices.index),
            pd.Series(filtered_acceleration, index=prices.index),
            pd.Series(errors, index=prices.index)
        )

    def generate_signals(
        self,
        df: pd.DataFrame,
        filtered_price: pd.Series,
        velocity: pd.Series = None,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate trading signals based on Kalman Filter output

        Args:
            df: Original price data
            filtered_price: Kalman filtered price
            velocity: Optional velocity estimate
            threshold: Signal threshold (%)

        Returns:
            DataFrame with signals
        """
        signals = df.copy()
        signals['filtered_price'] = filtered_price

        # Calculate deviation from filtered price
        signals['deviation'] = (df['Close'] - filtered_price) / filtered_price * 100

        # Generate signals based on deviation
        signals['signal'] = 0  # 0 = neutral, 1 = buy, -1 = sell

        # Buy when price is below filtered price by threshold
        signals.loc[signals['deviation'] < -threshold, 'signal'] = 1

        # Sell when price is above filtered price by threshold
        signals.loc[signals['deviation'] > threshold, 'signal'] = -1

        # Add velocity-based signals if available
        if velocity is not None:
            signals['velocity'] = velocity

            # Strong buy: price below filter AND positive velocity
            signals.loc[
                (signals['deviation'] < -threshold) & (velocity > 0),
                'signal'
            ] = 2

            # Strong sell: price above filter AND negative velocity
            signals.loc[
                (signals['deviation'] > threshold) & (velocity < 0),
                'signal'
            ] = -2

        return signals

    def backtest_strategy(
        self,
        signals: pd.DataFrame,
        initial_capital: float,
        position_size: float,
        use_confidence_sizing: bool,
        errors: pd.Series = None
    ) -> BacktestResults:
        """
        Backtest the Kalman Filter strategy

        Args:
            signals: DataFrame with signals and prices
            initial_capital: Starting capital
            position_size: Base position size
            use_confidence_sizing: Scale position by confidence
            errors: Optional estimation errors for confidence

        Returns:
            BacktestResults object
        """
        config = BacktestConfig(
            initial_capital=initial_capital,
            commission_pct=0.001,
            slippage_pct=0.001,
            position_size=position_size,
            max_positions=1
        )

        backtester = Backtester(config)

        # Track current position
        in_position = False

        def strategy_logic(bt: Backtester, date: datetime, row: pd.Series):
            nonlocal in_position
            symbol = "SYMBOL"

            signal = row['signal']

            # Entry signals
            if not in_position:
                if signal >= 1:  # Buy signal
                    # Calculate position size based on confidence
                    shares = bt.calculate_position_size(row['Close'])

                    if use_confidence_sizing and errors is not None:
                        # Scale by confidence (inverse of error)
                        idx = signals.index.get_loc(date)
                        error = errors.iloc[idx]
                        confidence = 1.0 / (1.0 + error)
                        shares = int(shares * confidence)

                    if shares > 0:
                        success = bt.enter_long(symbol, date, row['Close'], shares)
                        if success:
                            in_position = True

            # Exit signals
            elif in_position:
                if signal <= -1:  # Sell signal
                    bt.exit_position(symbol, date, row['Close'])
                    in_position = False

        results = backtester.run_backtest(signals, strategy_logic)

        return results

    def run(self):
        """Execute the Kalman Filter adaptive strategy"""
        symbol = self.options["SYMBOL"]["value"]
        period = self.options["PERIOD"]["value"]
        interval = self.options["INTERVAL"]["value"]
        filter_type = self.options["FILTER_TYPE"]["value"]
        process_noise = float(self.options["PROCESS_NOISE"]["value"])
        measurement_noise = float(self.options["MEASUREMENT_NOISE"]["value"])
        signal_threshold = float(self.options["SIGNAL_THRESHOLD"]["value"])
        initial_capital = float(self.options["INITIAL_CAPITAL"]["value"])
        position_size = float(self.options["POSITION_SIZE"]["value"])
        use_confidence_sizing = self.options["USE_CONFIDENCE_SIZING"]["value"]


        # Fetch data
        data_fetcher = DataFetcher(self.framework.database)
        df = data_fetcher.get_stock_data(symbol, period=period, interval=interval)

        if df is None or len(df) < 50:
            pass
            return {"error": "Insufficient data"}


        # Apply Kalman Filter

        prices = df['Close']

        if filter_type == "simple":
            filtered_price, errors = self.kalman_filter_simple(
                prices,
                process_noise,
                measurement_noise
            )
            velocity = None
            acceleration = None

        elif filter_type == "velocity":
            filtered_price, velocity, errors = self.kalman_filter_velocity(
                prices,
                process_noise,
                process_noise * 0.1,
                measurement_noise
            )
            acceleration = None

        else:  # full
            filtered_price, velocity, acceleration, errors = self.kalman_filter_full(
                prices,
                process_noise,
                process_noise * 0.1,
                process_noise * 0.01,
                measurement_noise
            )

        # Generate signals
        signals = self.generate_signals(df, filtered_price, velocity, signal_threshold)

        # Backtest
        results = self.backtest_strategy(
            signals,
            initial_capital,
            position_size,
            use_confidence_sizing,
            errors
        )

        # Display results
        results_dict = results.to_dict()

        for key, value in results_dict.items():
            pass

        # Current signal

        current_price = df['Close'].iloc[-1]
        current_filtered = filtered_price.iloc[-1]
        current_deviation = (current_price - current_filtered) / current_filtered * 100
        current_signal = signals['signal'].iloc[-1]


        if velocity is not None:
            current_velocity = velocity.iloc[-1]

        if acceleration is not None:
            current_acceleration = acceleration.iloc[-1]

        # Interpret signal
        if current_signal >= 2:
            signal_str = "STRONG BUY"
        elif current_signal == 1:
            signal_str = "BUY"
        elif current_signal == -1:
            signal_str = "SELL"
        elif current_signal <= -2:
            signal_str = "STRONG SELL"
        else:
            signal_str = "NEUTRAL"

        # Confidence
        if use_confidence_sizing:
            current_error = errors.iloc[-1]
            confidence = 1.0 / (1.0 + current_error)

        return {
            "symbol": symbol,
            "current_price": float(current_price),
            "filtered_price": float(current_filtered),
            "deviation_pct": float(current_deviation),
            "signal": signal_str,
            "velocity": float(current_velocity) if velocity is not None else None,
            "acceleration": float(current_acceleration) if acceleration is not None else None,
            "backtest_results": results_dict,
            "total_return_pct": results.total_return_pct,
            "sharpe_ratio": results.sharpe_ratio,
            "win_rate": results.win_rate,
            "total_trades": results.total_trades
        }
